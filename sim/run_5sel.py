"""Phase 5sel driver: does a byte allowance and/or a ranking model remove the overhead?

Run end to end with:

    python sim/run_5sel.py

The protocol (``docs/phase-5sel-protocol.md``) was written before this ran. This module:

1. reconstructs the *12 prediction-relevant cells* of Phase 4b+ from its published CSV,
   applying the pre-declared relevance rule rather than copying the count;
2. trains the P3 ranking model with leave-trajectory-out folds, three seeds and two
   losses, and picks the loss and the allowance factor ``f`` on the three validation
   trajectories only;
3. replays the 12 cells under every policy - the uncapped 4b+ comparators, the
   rank-and-cap selector at each ``f``, and P3 - recording per-trajectory metrics;
4. writes ``sim/results_5sel/{summary.md,results.csv}``, a deficit-versus-bytes Pareto
   PNG, and the P3 model in the Phase 3b export layout under
   ``training/experiments/phase5rank/``.

Nothing existing is edited; the 4b+ CSV, the feature table, the split manifest and the
score bundle are read only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim import policies_5sel as P
from sim import scene
from sim.simulate import RunConfig, build_sim_data, summarise
from sim.summary_4bplus import paired_ci, resample_matrix

ROOT = Path(__file__).resolve().parents[1]
CONFIG_4BPLUS = ROOT / "sim" / "config_4bplus.yaml"
RESULTS_4BPLUS = ROOT / "sim" / "results_4bplus" / "results.csv"
FEATURE_CONFIG = ROOT / "training" / "configs" / "features_v1.yaml"
SPLIT_V1 = ROOT / "dataset" / "manifests" / "experimental_split_v1.json"
OUTPUT = ROOT / "sim" / "results_5sel"
MODEL_DIR = ROOT / "training" / "experiments" / "phase5rank"

F_VALUES = (0.1, 0.25, 0.5, 1.0)
TAU_VALUES = (0.0, 0.1, 0.3, 0.5)
SELECTOR_BASES = ("mlp_b_tau0", "mlp_b_tau0.5", "heuristic_occ_tau0",
                  "heuristic_occ_tau0.5")
REACTIVE = ("reactive_frustum", "reactive_visible", "reactive_visible_lag")
RELEVANCE_GAP = 0.02
SUCCESS_TOLERANCE = 1.10          # deficit within 10% (relative) of mlp_b_tau0
SUCCESS_BYTE_CAP = 2.0            # <= 2.0x reactive_visible bytes
SUCCESS_MIN_CELLS = 10
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260501
THREADS = 4


# --- the 12 prediction-relevant cells -----------------------------------------

def relevant_cells(results_csv: Path, nominal_bandwidth: float) -> list[dict]:
    """Apply the 4b+ relevance rule to the published CSV and return the matching cells."""
    values: dict[tuple, dict[str, dict[str, float]]] = {}
    with results_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "deficit":
                continue
            cell = (float(row["budget_fraction"]), float(row["load_time_scale"]),
                    float(row["latency_seconds"]))
            values.setdefault(cell, {}).setdefault(row["policy"], {})[row["trajectory"]] = \
                float(row["value"])

    cells = []
    for (fraction, scale, latency), per_policy in sorted(values.items()):
        trajectories = sorted(next(iter(per_policy.values())))
        best_reactive = min(np.mean([per_policy[p][t] for t in trajectories])
                            for p in REACTIVE)
        oracle = np.mean([per_policy["oracle_matched"][t] for t in trajectories])
        if best_reactive - oracle >= RELEVANCE_GAP - 1e-12:
            cells.append({"budget_fraction": fraction, "load_time_scale": scale,
                          "latency_seconds": latency,
                          "bandwidth_mb_s": nominal_bandwidth / scale})
    return cells


# --- policy set ---------------------------------------------------------------

def policy_names() -> list[str]:
    names = ["reactive_visible", "oracle_matched"]
    for base in ("mlp_b", "heuristic_occ"):
        for tau in TAU_VALUES:
            names.append(f"{base}_tau{tau:g}")
    for base in SELECTOR_BASES:
        for f in F_VALUES:
            names.append(f"{base}@f{f:g}")
    names.append("p3")
    for f in F_VALUES:
        names.append(f"p3@f{f:g}")
    return names


def cell_policies(fixed, oracle, p3_priority, allowance_base) -> dict:
    """Every policy as ``(priority_array, allowance_bytes)`` for one cell."""
    choices = {"reactive_visible": (fixed["reactive_visible"], math.inf),
               "oracle_matched": (oracle, math.inf)}
    for base in ("mlp_b", "heuristic_occ"):
        for tau in TAU_VALUES:
            choices[f"{base}_tau{tau:g}"] = (fixed[f"{base}_tau{tau:g}"], math.inf)
    for base in SELECTOR_BASES:
        for f in F_VALUES:
            choices[f"{base}@f{f:g}"] = (fixed[base], f * allowance_base)
    choices["p3"] = (p3_priority, math.inf)
    for f in F_VALUES:
        choices[f"p3@f{f:g}"] = (p3_priority, f * allowance_base)
    return choices


# --- the grid -----------------------------------------------------------------

def run_grid(cells, data, fixed, size_per_row, p3_priority, sample_interval,
             window_samples, margin, log):
    """Replay every cell under every policy, returning per (cell, policy, trajectory)."""
    names = policy_names()
    per_trajectory: dict[tuple, dict] = {}
    completed = 0
    total = len(cells) * len(names) * len(data.views)
    started = time.perf_counter()
    peaks = {view.trajectory_id: scene.peak_working_set(
        view.rows_by_sample, data.ground_truth_visible, data.object_ids, _SIZES_CACHE,
        window_samples) for view in data.views}
    for cell in cells:
        fraction = cell["budget_fraction"]
        bandwidth_bytes = cell["bandwidth_mb_s"] * 1e6
        allowance_base = bandwidth_bytes * sample_interval
        oracle = P.oracle_matched_priority(data, size_per_row, cell["latency_seconds"],
                                           bandwidth_bytes, margin)
        choices = cell_policies(fixed, oracle, p3_priority, allowance_base)
        for view in data.views:
            config = RunConfig(budget_bytes=max(peaks[view.trajectory_id] * fraction, 1.0),
                               size_multiplier=1.0,
                               latency_seconds=cell["latency_seconds"],
                               bandwidth_bytes_per_s=bandwidth_bytes,
                               sample_interval=sample_interval)
            for name in names:
                priority, allowance = choices[name]
                metrics = summarise(P.simulate_selector(
                    view, priority, _SIZES_CACHE, data.ground_truth_visible, data.coverage,
                    data.object_ids, config, allowance_bytes=allowance), config)
                per_trajectory[(fraction, cell["load_time_scale"],
                                cell["latency_seconds"], name,
                                view.trajectory_id)] = metrics
                completed += 1
        log(f"    cell budget={fraction:.0%} scale={cell['load_time_scale']:g} "
            f"latency={cell['latency_seconds']:g}s ({completed}/{total}, "
            f"{time.perf_counter() - started:.0f}s)")
    return per_trajectory


_SIZES_CACHE: dict = {}


# --- P3 training --------------------------------------------------------------

def train_p3(table, dataset, size_bytes, paths, split, seeds, losses, max_epochs, log):
    """Train P3 leave-trajectory-out; return OOF scores per loss and the fold list."""
    import torch
    from training.run_3b import snap_yaw_row_mask, trajectory_rows
    from training import splits as S, train as T

    torch.set_num_threads(THREADS)
    snap = snap_yaw_row_mask(table, Path(paths["sidecar"]))
    gain = P.gain_target(table, dataset, size_bytes)
    codes = P.group_codes(table)
    trainable = T.unified_primary(table) & ~snap

    folds = S.build_folds(sorted(table.trajectories), 7, 2, seed=20261105)
    fold_report = [{"fold": f.index, "train": list(f.train),
                    "validation": list(f.validation), "test": list(f.test)}
                   for f in folds]
    log("    folds (train | validation | test):")
    for f in fold_report:
        log(f"      {f['fold']}: {len(f['train'])} train, "
            f"val={','.join(f['validation'])}, test={','.join(f['test'])}")

    config = {"training": {"max_epochs": int(max_epochs)}}
    oof = {loss: {seed: np.full(len(table), np.nan) for seed in seeds} for loss in losses}
    epochs = {loss: {seed: None for seed in seeds} for loss in losses}
    parameters = None

    for loss in losses:
        for seed in seeds:
            started = time.perf_counter()
            for fold in folds:
                train_rows = trajectory_rows(table, fold.train) & trainable
                val_rows_mask = trajectory_rows(table, fold.validation) & trainable
                Xn, _mean, _std = P.normalise_fold(table.X, train_rows, table.X)
                fit = P.train_rank_fold(
                    Xn, gain, P.group_lists(codes, train_rows),
                    np.flatnonzero(val_rows_mask), P.group_lists(codes, val_rows_mask),
                    config, seed, loss)
                test_rows = trajectory_rows(table, fold.test)
                oof[loss][seed][test_rows] = fit["raw_scores"][test_rows]
                epochs[loss][seed] = fit["best_epoch"]
                parameters = fit["parameters"]
            log(f"      P3 loss={loss} seed={seed}: {time.perf_counter() - started:.1f}s, "
                f"epochs={epochs[loss][seed]}, {parameters} params")

    for loss in losses:
        for seed in seeds:
            if np.isnan(oof[loss][seed]).any():
                raise RuntimeError(f"P3 {loss} seed {seed} left rows unscored")

    validation = [t["trajectory_id"] for t in split["splits"]["validation"]]
    val_rows_mask = trajectory_rows(table, validation) & trainable
    val_groups = P.group_lists(codes, val_rows_mask)
    val_rows = np.flatnonzero(val_rows_mask)
    local_groups = [np.searchsorted(val_rows, rows) for rows in val_groups]
    ndcg = {}
    for loss in losses:
        averaged = np.mean([oof[loss][seed] for seed in seeds], axis=0)
        ndcg[loss] = P.ndcg_at_k(averaged[val_rows], gain[val_rows], local_groups, P.NDCG_K)
    chosen_loss = max(losses, key=lambda name: (ndcg[name], name == "listwise"))
    log("    loss choice (validation NDCG@10): "
        + ", ".join(f"{name}={ndcg[name]:.4f}" for name in losses) + f" -> {chosen_loss}")
    chosen_oof = np.mean([oof[chosen_loss][seed] for seed in seeds], axis=0)
    return {"chosen_oof": chosen_oof, "loss": chosen_loss, "ndcg": ndcg, "oof": oof,
            "epochs": epochs, "folds": fold_report, "parameters": parameters}


# --- aggregation --------------------------------------------------------------

def cell_values(per_trajectory, cells, policy, trajectories, metric):
    """Per-cell values: mean for deficit, sum for counters."""
    result = []
    for cell in cells:
        key = (cell["budget_fraction"], cell["load_time_scale"], cell["latency_seconds"])
        entries = [per_trajectory[key + (policy, t)] for t in trajectories]
        if metric == "deficit":
            result.append(float(np.mean([e[metric] for e in entries])))
        else:
            result.append(float(np.sum([e[metric] for e in entries])))
    return result


def policy_aggregate(per_trajectory, cells, policy, trajectories):
    deficits = cell_values(per_trajectory, cells, policy, trajectories, "deficit")
    byte_values = cell_values(per_trajectory, cells, policy, trajectories, "bytes_streamed")
    wasted = cell_values(per_trajectory, cells, policy, trajectories, "wasted_prefetches")
    return {"deficit": float(np.mean(deficits)), "bytes": float(np.sum(byte_values)),
            "wasted": float(np.sum(wasted))}


def trajectory_deficit_vector(per_trajectory, cells, policy, trajectories):
    return np.array([float(np.mean([
        per_trajectory[(cell["budget_fraction"], cell["load_time_scale"],
                        cell["latency_seconds"], policy, t)]["deficit"]
        for cell in cells])) for t in trajectories])


# --- writing ------------------------------------------------------------------

def write_csv(path: Path, per_trajectory, cells, policies, trajectories) -> None:
    fields = ["budget_fraction", "load_time_scale", "latency_seconds", "bandwidth_mb_s",
              "policy", "trajectory", "metric", "value"]
    metric_names = ["deficit", "bytes_streamed", "wasted_prefetches", "loads_completed",
                    "pop_in_events", "peak_resident_bytes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            key = (cell["budget_fraction"], cell["load_time_scale"],
                   cell["latency_seconds"])
            for policy in policies:
                for trajectory in trajectories:
                    metrics = per_trajectory[key + (policy, trajectory)]
                    for metric in metric_names:
                        writer.writerow({
                            "budget_fraction": key[0], "load_time_scale": key[1],
                            "latency_seconds": key[2],
                            "bandwidth_mb_s": cell["bandwidth_mb_s"], "policy": policy,
                            "trajectory": trajectory, "metric": metric,
                            "value": f"{metrics[metric]:.6g}"})


def pareto_png(path: Path, aggregates, reactive_bytes) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6.5))
    for name, agg in aggregates.items():
        x, y = agg["bytes"] / reactive_bytes, agg["deficit"]
        if name in ("mlp_b_tau0", "oracle_matched", "reactive_visible", "p3"):
            axis.scatter([x], [y], s=70, marker="D", zorder=3, label=name)
            axis.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4),
                          fontsize=8)
        elif "@f" in name:
            axis.scatter([x], [y], s=22, color="tab:red", zorder=2)
        elif name.startswith("p3"):
            axis.scatter([x], [y], s=45, color="tab:green", marker="s", zorder=3)
        else:
            axis.scatter([x], [y], s=22, color="tab:gray", zorder=1)
    axis.scatter([], [], s=22, color="tab:red", label="selector (*@f*)")
    axis.scatter([], [], s=22, color="tab:gray", label="tau family (uncapped)")
    axis.set_xlabel("bytes / reactive_visible bytes")
    axis.set_ylabel("mean deficit over the 12 relevant cells (all 14 trajectories)")
    axis.set_title("Phase 5sel: deficit vs bytes")
    axis.axvline(SUCCESS_BYTE_CAP, color="black", linestyle=":", linewidth=1)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


# --- main ---------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 5sel sweep")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--losses", default="listwise,hinge")
    parser.add_argument("--max-epochs", type=int, default=P.MAX_EPOCHS)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args(argv)


def run(argv=None) -> dict:
    arguments = parse_args(argv)
    config = yaml.safe_load(CONFIG_4BPLUS.read_text(encoding="utf-8"))
    paths = config["paths"]
    nominal = float(config["nominal_bandwidth_mb_s"])
    sample_interval = float(config["sample_interval_seconds"])
    window_samples = int(round(float(config["working_set_window_seconds"]) / sample_interval))
    snap_window = int(config["snap_window_samples"])
    margin = float(config["matched_margin_seconds"])
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in arguments.seeds.split(",")]
    losses = [s.strip() for s in arguments.losses.split(",")]

    def log(message):
        print(message, flush=True)

    log("=" * 78)
    log("Phase 5sel - rank-and-cap selector and a ranking model")
    log("=" * 78)

    cells = relevant_cells(RESULTS_4BPLUS, nominal)
    log(f"\n  {len(cells)} prediction-relevant cells (rule: best_reactive - "
        f"oracle_matched >= {RELEVANCE_GAP})")

    data = build_sim_data(paths["oof_scores"], paths["dataset"], snap_window)
    records = scene.load_objects(paths["object_index"])
    size_bytes = scene.sized_bytes(records, 1.0)
    _SIZES_CACHE.clear()
    _SIZES_CACHE.update(size_bytes)
    size_per_row = np.array([size_bytes[oid] for oid in data.object_ids], dtype=np.float64)
    fixed = P.fixed_priorities(data)
    log(f"  {len(data.object_ids)} rows, {len(data.trajectories)} trajectories")

    log("\n[1/3] P3 ranking model")
    from training import dataset as D
    feature_config = yaml.safe_load(FEATURE_CONFIG.read_text(encoding="utf-8"))
    table = D.build_table(paths["dataset"], feature_config)
    if not np.array_equal(np.array(data.object_ids, dtype=object),
                          np.array(table.object_id, dtype=object)):
        raise RuntimeError("simulator rows and feature-table rows are not aligned")
    split = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    p3 = train_p3(table, paths["dataset"], size_bytes, paths, split, seeds, losses,
                  arguments.max_epochs, log)
    p3_priority = np.where(data.ground_truth_visible, 2.0, P.sigmoid(p3["chosen_oof"]))

    log("\n[2/3] replaying the 12 cells")
    started = time.perf_counter()
    per_trajectory = run_grid(cells, data, fixed, size_per_row, p3_priority,
                              sample_interval, window_samples, margin, log)
    log(f"    {len(per_trajectory)} runs in {time.perf_counter() - started:.0f}s")

    all_trajectories = [view.trajectory_id for view in data.views]
    validation = [t["trajectory_id"] for t in split["splits"]["validation"]]
    test = [t["trajectory_id"] for t in split["splits"]["test"]]
    policies = policy_names()

    # --- f selection on validation trajectories only -------------------------
    mlp_deficit_val = policy_aggregate(per_trajectory, cells, "mlp_b_tau0",
                                       validation)["deficit"]
    reactive_bytes_val = policy_aggregate(per_trajectory, cells, "reactive_visible",
                                          validation)["bytes"]
    f_report = []
    for f in F_VALUES:
        agg = policy_aggregate(per_trajectory, cells, f"p3@f{f:g}", validation)
        f_report.append({"f": f, "deficit": agg["deficit"], "bytes": agg["bytes"],
                         "bytes_ratio": agg["bytes"] / reactive_bytes_val,
                         "qualifies": bool(agg["deficit"] <= SUCCESS_TOLERANCE
                                           * mlp_deficit_val
                                           and agg["bytes"] <= SUCCESS_BYTE_CAP
                                           * reactive_bytes_val)})
    qualifying = [entry for entry in f_report if entry["qualifies"]]
    chosen_f = (min(qualifying, key=lambda e: (e["deficit"], e["f"]))["f"]
                if qualifying else min(f_report, key=lambda e: e["deficit"])["f"])
    log("\n[3/3] f selection (validation trajectories only):")
    for entry in f_report:
        log(f"    f={entry['f']:<4} deficit={entry['deficit']:.4f} "
            f"bytes={entry['bytes_ratio']:.2f}x "
            f"{'qualifies' if entry['qualifies'] else ''}")
    log(f"    chosen f = {chosen_f:g}")

    # --- aggregates ---------------------------------------------------------
    aggregates_all = {name: policy_aggregate(per_trajectory, cells, name, all_trajectories)
                      for name in policies}
    aggregates_test = {name: policy_aggregate(per_trajectory, cells, name, test)
                       for name in policies}
    reactive_bytes_all = aggregates_all["reactive_visible"]["bytes"]
    reactive_bytes_test = aggregates_test["reactive_visible"]["bytes"]

    picks_all = resample_matrix(len(all_trajectories), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    picks_test = resample_matrix(len(test), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)

    def ci_block(trajectories, picks):
        reference = trajectory_deficit_vector(per_trajectory, cells, "mlp_b_tau0",
                                              trajectories)
        return {name: paired_ci(
            trajectory_deficit_vector(per_trajectory, cells, name, trajectories),
            reference, picks) for name in policies}

    ci_all = ci_block(all_trajectories, picks_all)
    ci_test = ci_block(test, picks_test)

    # --- success rule (test trajectories) -----------------------------------
    success = []
    for name in [p for p in policies if "@f" in p or p == "p3"]:
        passing = 0
        for cell in cells:
            key = (cell["budget_fraction"], cell["load_time_scale"],
                   cell["latency_seconds"])
            deficit = float(np.mean([per_trajectory[key + (name, t)]["deficit"]
                                     for t in test]))
            bytes_ = sum(per_trajectory[key + (name, t)]["bytes_streamed"] for t in test)
            base_deficit = float(np.mean([per_trajectory[key + ("mlp_b_tau0", t)]["deficit"]
                                          for t in test]))
            base_bytes = sum(per_trajectory[key + ("reactive_visible", t)]["bytes_streamed"]
                             for t in test)
            if (deficit <= SUCCESS_TOLERANCE * base_deficit
                    and bytes_ <= SUCCESS_BYTE_CAP * base_bytes):
                passing += 1
        agg, base = aggregates_test[name], aggregates_test["mlp_b_tau0.5"]
        dominates = (agg["bytes"] <= base["bytes"] + 1e-6
                     and agg["deficit"] <= base["deficit"] + 1e-9
                     and (agg["bytes"] < base["bytes"] - 1e-6
                          or agg["deficit"] < base["deficit"] - 1e-9))
        success.append({"policy": name, "cells": passing, "beats_tau05": bool(dominates),
                        "success": bool(passing >= SUCCESS_MIN_CELLS and dominates)})

    # --- write --------------------------------------------------------------
    write_csv(output / "results.csv", per_trajectory, cells, policies, all_trajectories)
    pareto_png(output / "pareto_deficit_vs_bytes.png", aggregates_all, reactive_bytes_all)
    (output / "summary.md").write_text(
        render_summary(cells, policies, aggregates_all, aggregates_test, reactive_bytes_all,
                       reactive_bytes_test, f_report, chosen_f, p3, success, test,
                       validation, ci_all, ci_test), encoding="utf-8")

    manifest = {"cells": cells, "policies": policies, "chosen_f": chosen_f,
                "p3_loss": p3["loss"], "p3_ndcg": p3["ndcg"], "seeds": seeds,
                "losses": losses, "f_report": f_report, "success": success,
                "folds": p3["folds"], "parameters": p3["parameters"],
                "all_trajectories": all_trajectories, "test_trajectories": test,
                "validation_trajectories": validation,
                "grid_declared_before_running": True}
    (output / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")

    won = any(entry["success"] for entry in success)
    model_paths = []
    if not arguments.skip_model:
        model_paths = save_model(table, size_bytes, paths, split, p3["loss"], seeds[0],
                                 arguments.max_epochs, log)

    log(f"\n  -> {output / 'results.csv'}")
    log(f"  -> {output / 'summary.md'}")
    log(f"  success: {'yes' if won else 'no'} "
        f"({sum(e['success'] for e in success)} policies met the rule)")
    return {"success": success, "model_paths": model_paths}


def save_model(table, size_bytes, paths, split, loss, seed, max_epochs, log):
    """Export P3 trained on split-v1 train, early-stopped on split-v1 validation."""
    import torch
    from training import dataset as D
    from training.models import export_bundle
    from training.run_3b import snap_yaw_row_mask, trajectory_rows
    from training import train as T

    torch.set_num_threads(THREADS)
    snap = snap_yaw_row_mask(table, Path(paths["sidecar"]))
    gain = P.gain_target(table, paths["dataset"], size_bytes)
    codes = P.group_codes(table)
    trainable = T.unified_primary(table) & ~snap
    train_ids = [entry["trajectory_id"] for entry in split["splits"]["train"]]
    validation_ids = [entry["trajectory_id"] for entry in split["splits"]["validation"]]
    train_rows = trajectory_rows(table, train_ids) & trainable
    val_rows_mask = trajectory_rows(table, validation_ids) & trainable
    Xn, mean, std = P.normalise_fold(table.X, train_rows, table.X)
    fit = P.train_rank_fold(Xn, gain, P.group_lists(codes, train_rows),
                            np.flatnonzero(val_rows_mask), P.group_lists(codes, val_rows_mask),
                            {"training": {"max_epochs": int(max_epochs)}}, seed, loss)

    normaliser = D.Normaliser(names=list(table.names), mean=mean.astype(np.float64),
                              std=std.astype(np.float64),
                              counts=np.zeros(len(table.names), dtype=np.int64),
                              fit_rows=int(train_rows.sum()))
    stem = MODEL_DIR / f"phase5rank_p3_{loss}_seed{seed}"
    info = export_bundle(fit["model"], table.names, ["p3_score"], normaliser, stem)
    (MODEL_DIR / "results.json").write_text(json.dumps({
        "phase": "5sel", "loss": loss, "seed": seed, "hidden": P.HIDDEN,
        "dropout": P.DROPOUT, "best_epoch": fit["best_epoch"],
        "best_val_ndcg": fit["best_val_ndcg"], "parameters": fit["parameters"],
        "trained_on": list(split["splits"]["train"]),
        "early_stopped_on": list(split["splits"]["validation"]),
        "target": f"{P.GAIN_FIELD} / full-texture bytes at 1x", "export": info},
        indent=2) + "\n", encoding="utf-8")
    log(f"    saved model -> {info['weights']}")
    return [info["weights"], info["header"], str(MODEL_DIR / "results.json")]


def render_summary(cells, policies, aggregates_all, aggregates_test, reactive_bytes_all,
                   reactive_bytes_test, f_report, chosen_f, p3, success, test, validation,
                   ci_all, ci_test) -> str:
    lines = ["# Phase 5sel summary", "",
             "Rank-and-cap selector and a ranking model against the Phase 4b+ byte "
             "overhead. Deficit is the coverage-weighted fraction of visible object-steps "
             "not resident; bytes are total streamed as a multiple of `reactive_visible`. "
             "The protocol and success rule are in `docs/phase-5sel-protocol.md`.", "",
             f"- {len(cells)} prediction-relevant cells, {len(policies)} policies",
             f"- P3 loss chosen on validation: **{p3['loss']}** (NDCG@10 "
             + ", ".join(f"{k}={v:.4f}" for k, v in p3["ndcg"].items()) + ")",
             f"- `f` chosen on validation trajectories: **{chosen_f:g}**",
             f"- test: {', '.join(test)}; validation: {', '.join(validation)}", ""]

    for label, aggregates, reactive_bytes, ci_block in (
            ("all 14 trajectories", aggregates_all, reactive_bytes_all, ci_all),
            ("split-v1 test trajectories", aggregates_test, reactive_bytes_test, ci_test)):
        lines += [f"## Deficit, bytes and paired CI - {label}", "",
                  "`CI` is the 95% bootstrap interval (over trajectories) of the deficit "
                  "difference versus `mlp_b_tau0`; negative means lower deficit.", "",
                  "| policy | deficit | bytes / reactive_visible | wasted loads | "
                  "CI vs mlp_b_tau0 |", "|---|---|---|---|---|"]
        for name in policies:
            agg = aggregates[name]
            _mean, low, high = ci_block[name]
            lines.append(f"| `{name}` | {agg['deficit']:.4f} | "
                         f"{agg['bytes'] / reactive_bytes:.2f}x | {int(agg['wasted'])} | "
                         f"[{low:+.4f}, {high:+.4f}] |")
        lines.append("")

    lines += ["## f-selection table (validation trajectories only)", "",
              "| f | deficit | bytes / reactive_visible | qualifies |", "|---|---|---|---|"]
    for entry in f_report:
        lines.append(f"| {entry['f']:g} | {entry['deficit']:.4f} | "
                     f"{entry['bytes_ratio']:.3f} | "
                     f"{'yes' if entry['qualifies'] else 'no'} |")
    lines.append("")

    lines += ["## Success rule (test trajectories)", "",
              "A policy succeeds iff it is within 10% of `mlp_b_tau0`'s deficit at "
              "<= 2.0x `reactive_visible` bytes in >= 10 of 12 cells **and** Pareto-beats "
              "`mlp_b_tau0.5`.", "",
              "| policy | cells within tolerance at <=2.0x | beats mlp_b_tau0.5 | success |",
              "|---|---|---|---|"]
    for entry in success:
        lines.append(f"| `{entry['policy']}` | {entry['cells']}/12 | "
                     f"{'yes' if entry['beats_tau05'] else 'no'} | "
                     f"**{'yes' if entry['success'] else 'no'}** |")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run()
