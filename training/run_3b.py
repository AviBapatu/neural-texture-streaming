"""Phase 3b - train and evaluate the visibility MLP against the frozen heuristic.

    python training/run_3b.py --config training/configs/model_v1.yaml

Protocol
--------
Primary: leave-trajectory-out grouped folds over all 14 trajectories. Inside each
fold, epochs and the decision threshold are chosen on that fold's validation
trajectories; the fold's test trajectories are scored but never influence any
choice. Composing the seven folds gives out-of-fold predictions for every
trajectory.

Secondary: the fixed split v1 test partition, scored once, after every choice has
been frozen.

Nothing here retunes the heuristic, changes a dataset, or writes outside
``training/experiments/phase3b``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import baselines as B
from training import dataset as D
from training import evaluate as E
from training import features as F
from training import models as M
from training import splits as S
from training import train as T

OUTPUT_DIR = Path("training/experiments/phase3b")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3b visibility MLP")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--variants", default=None,
                        help="comma-separated subset of variant keys (default: all)")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seed override (default: config)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--heuristic-only", action="store_true", dest="heuristic_only",
                        help="recompute only the frozen-heuristic reference and the paired "
                             "comparisons, reusing the saved MLP results; no retraining")
    return parser.parse_args(argv)


# --- helpers ------------------------------------------------------------------

def feature_config(base: dict, history_k: int) -> dict:
    """The feature config with a different history depth, and the allow-list adjusted.

    With K=0 no ``hist_*`` column is produced, so leaving them in the allow-list
    would make ``assemble_matrix`` raise. Dropping them is the ablation: the model
    loses all causal pose history.
    """
    adjusted = copy.deepcopy(base)
    adjusted["history_k"] = int(history_k)
    if history_k == 0:
        adjusted["features"]["allow"] = [name for name in base["features"]["allow"]
                                         if not name.startswith("hist_")]
    return adjusted


def snap_yaw_row_mask(table, sidecar_path: Path) -> np.ndarray:
    """Per-row flag: within +/-10 samples of a yaw snap. From the Phase 3a.1 sidecar."""
    lookup = {name: index for index, name in enumerate(table.trajectories)}
    flags = np.zeros(len(table), dtype=bool)
    with sidecar_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trajectory_id = row["trajectory_id"]
            if trajectory_id not in lookup:
                continue
            flags[(table.trajectory == lookup[trajectory_id])
                  & (table.sample_index == int(row["sample_index"]))] = row["snap_affected_yaw"] == "1"
    return flags


def trajectory_rows(table, names) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(table.trajectories)}
    indices = [lookup[name] for name in names if name in lookup]
    if not indices:
        return np.zeros(len(table), dtype=bool)
    return np.isin(table.trajectory, indices)


def frozen_heuristic(config: dict) -> tuple[dict, dict]:
    """The Phase 3a parameters, one set per target, read and never retuned.

    Phase 3a selected *different* parameters for the two targets: the occlusion target
    was tuned with `occlusion_aware=True` and a 30 degree occlusion penalty. Scoring
    the occlusion target with the frustum-tuned score would handicap the baseline and
    inflate the MLP's apparent advantage, so both are built here and each target is
    scored with its own.
    """
    payload = json.loads(Path(config["heuristic"]["results"]).read_text(encoding="utf-8"))
    horizon = float(config["heuristic"]["threshold_horizon"])
    tuned = {}
    for key in ("frustum", "occlusion"):
        chosen = payload["tuning"]["chosen"][key]
        tuned[key] = (
            B.HeuristicParams(damping=chosen["damping"],
                              angular_damping=chosen["angular_damping"],
                              integrate_steps=chosen["integrate_steps"],
                              margin_threshold_deg=0.0,
                              occlusion_penalty_deg=chosen["occlusion_penalty_deg"]),
            float(payload["tuning"]["chosen_thresholds_deg"][key][f"{horizon:g}"]))
    return tuned, payload


def output_specs(config: dict) -> list[T.OutputSpec]:
    return [T.OutputSpec(name=entry["name"], target=entry["target"],
                         horizon=float(entry["horizon"]))
            for entry in config["targets"]["outputs"]]


def masked_rows(table, spec: T.OutputSpec, rows: np.ndarray, snap_yaw: np.ndarray,
                drop_snap: bool) -> np.ndarray:
    """Rows usable for scoring one output: valid label, that target's primary subset."""
    _, valid, primary = E.target_arrays(table, spec.target, spec.horizon_index)
    mask = np.asarray(rows, dtype=bool) & valid & primary
    if drop_snap:
        mask = mask & ~np.asarray(snap_yaw, dtype=bool)
    return mask


def score_output(table, spec: T.OutputSpec, rows: np.ndarray, score: np.ndarray,
                 threshold: float, ks=(5, 10), predicted: np.ndarray | None = None) -> dict:
    """Full metric set for one output on one row subset.

    ``predicted`` lets the caller supply per-fold decisions (each fold's threshold
    was chosen on its own validation rows) instead of applying one global threshold.
    """
    y_true, _, _ = E.target_arrays(table, spec.target, spec.horizon_index)
    if rows.sum() == 0:
        return {"n": 0}
    decisions = predicted[rows] if predicted is not None else (score[rows] >= threshold)
    metrics = E.binary_metrics(y_true[rows], decisions, score[rows])
    metrics["threshold"] = float(threshold)
    for k in ks:
        metrics[f"recall_at_{k}"] = E.recall_at_k(table, y_true, score, k, eligible=rows)
    return metrics


# --- one fold, one variant ----------------------------------------------------

def run_fold_mlp(table, specs, fold, bundle_config, config, seed, heuristic_score,
                 snap_yaw, keep_drop_snap=True):
    """Train on the fold's train trajectories, early-stop on validation, score test."""
    train_rows = trajectory_rows(table, fold.train) & T.unified_primary(table)
    validation_rows = trajectory_rows(table, fold.validation) & T.unified_primary(table)
    test_rows = trajectory_rows(table, fold.test) & T.unified_primary(table)

    extra = heuristic_score.reshape(-1, 1) if bundle_config.get("heuristic_input") else None
    bundle = T.build_bundle(table, specs, train_rows,
                            drop_snap_yaw=config["masking"]["drop_snap_yaw_from_training"],
                            snap_yaw_rows=snap_yaw, extra_inputs=extra)
    fit = T.train_mlp(bundle, specs, validation_rows, table, config, seed,
                      exclude_snap=snap_yaw)
    scores = T.predict_mlp(fit.model, bundle)

    # Threshold per output, chosen on validation rows only.
    thresholds = {}
    for column, spec in enumerate(specs):
        rows = masked_rows(table, spec, validation_rows, snap_yaw, True)
        y_true, _, _ = E.target_arrays(table, spec.target, spec.horizon_index)
        thresholds[spec.name] = T.choose_threshold(y_true[rows], scores[rows, column]) \
            if rows.sum() else 0.5

    return {"fit": fit, "scores": scores, "thresholds": thresholds,
            "test_rows": test_rows, "train_rows": int(train_rows.sum()),
            "validation_rows": int(validation_rows.sum())}


def run_fold_sklearn(table, specs, fold, config, seed, settings):
    """A separate HistGradientBoosting model per output; no shared trunk."""
    train_rows = trajectory_rows(table, fold.train) & T.unified_primary(table)
    validation_rows = trajectory_rows(table, fold.validation) & T.unified_primary(table)
    test_rows = trajectory_rows(table, fold.test) & T.unified_primary(table)
    snap_placeholder = np.zeros(len(table), dtype=bool)

    y, _ = T.per_output_targets(table, specs)
    X = table.X.astype(np.float64)
    scores = np.zeros((len(table), len(specs)), dtype=np.float64)
    thresholds, fitted, parameters = {}, [], 0

    for column, spec in enumerate(specs):
        _, valid, primary = E.target_arrays(table, spec.target, spec.horizon_index)
        fit_rows = train_rows & valid & primary
        model = T.train_sklearn(X, y[:, column], fit_rows, seed, settings)
        fitted.append(model)
        parameters += M.count_parameters(model)
        scores[:, column] = T.predict_sklearn(model, X)

        rows = masked_rows(table, spec, validation_rows, snap_placeholder, False) & \
            valid
        thresholds[spec.name] = T.choose_threshold(y[rows, column], scores[rows, column]) \
            if rows.sum() else 0.5

    return {"fit": None, "models": fitted, "scores": scores, "thresholds": thresholds,
            "test_rows": test_rows, "train_rows": int(train_rows.sum()),
            "validation_rows": int(validation_rows.sum()), "parameters": parameters}


# --- aggregate out-of-fold predictions ---------------------------------------

def aggregate_oof(table, specs, per_fold: list[dict], snap_yaw: np.ndarray,
                  drop_snap: bool) -> dict:
    """Per-trajectory, mean/std, pooled - from out-of-fold predictions."""
    rows_by_fold = per_fold
    per_output = {}
    for column, spec in enumerate(specs):
        # Assemble the out-of-fold score and decision for every trajectory. Each
        # fold contributes its own threshold, chosen on that fold's validation rows.
        pooled_rows = np.zeros(len(table), dtype=bool)
        pooled_score = np.full(len(table), np.nan)
        pooled_predicted = np.zeros(len(table), dtype=bool)
        thresholds = []
        for entry in rows_by_fold:
            test = entry["test_rows"]
            threshold = float(entry["thresholds"][spec.name])
            thresholds.append(threshold)
            pooled_rows |= test
            pooled_score[test] = entry["scores"][test, column]
            pooled_predicted[test] = entry["scores"][test, column] >= threshold
        usable = masked_rows(table, spec, pooled_rows, snap_yaw, drop_snap)

        y_true, _, _ = E.target_arrays(table, spec.target, spec.horizon_index)
        threshold = float(np.mean(thresholds))

        per_trajectory = {}
        for index, trajectory_id in enumerate(table.trajectories):
            mask = usable & (table.trajectory == index)
            if not mask.any():
                continue
            if int(y_true[mask].sum()) == 0:
                per_trajectory[trajectory_id] = {
                    "n": int(mask.sum()), "positives": 0, "pr_auc": None,
                    "note": "no positives; PR-AUC undefined"}
                continue
            metrics = score_output(table, spec, mask, pooled_score, threshold,
                                   predicted=pooled_predicted)
            metrics["positives"] = int(y_true[mask].sum())
            per_trajectory[trajectory_id] = metrics

        finite = [v["pr_auc"] for v in per_trajectory.values()
                  if isinstance(v.get("pr_auc"), (int, float)) and np.isfinite(v["pr_auc"])]
        pooled = score_output(table, spec, usable, pooled_score, threshold,
                              predicted=pooled_predicted) if usable.any() else {"n": 0}
        per_output[spec.name] = {
            "pooled": pooled,
            "per_trajectory": per_trajectory,
            "trajectory_mean": float(np.mean(finite)) if finite else None,
            "trajectory_std": float(np.std(finite)) if finite else None,
            "trajectories_scored": len(finite),
            "threshold_mean": threshold,
        }
    return per_output


def heuristic_metrics(table, specs, scores_by_target: dict, snap_yaw, drop_snap) -> dict:
    """The frozen heuristic scored on the same rows and masks as the MLP.

    Each target uses its own tuned parameters and its own score.
    """
    result = {}
    for spec in specs:
        score = scores_by_target[spec.target]
        _, valid, primary = E.target_arrays(table, spec.target, spec.horizon_index)
        usable = valid & primary
        if drop_snap:
            usable = usable & ~snap_yaw
        y_true, _, _ = E.target_arrays(table, spec.target, spec.horizon_index)
        per_trajectory = {}
        for index, trajectory_id in enumerate(table.trajectories):
            mask = usable & (table.trajectory == index)
            if not mask.any():
                continue
            if int(y_true[mask].sum()) == 0:
                per_trajectory[trajectory_id] = {"n": int(mask.sum()), "positives": 0,
                                                 "pr_auc": None}
                continue
            per_trajectory[trajectory_id] = {
                "n": int(mask.sum()), "positives": int(y_true[mask].sum()),
                "pr_auc": float(E.average_precision(y_true[mask], score[mask])),
            }
        finite = [v["pr_auc"] for v in per_trajectory.values()
                  if isinstance(v.get("pr_auc"), (int, float)) and np.isfinite(v["pr_auc"])]
        result[spec.name] = {
            "per_trajectory": per_trajectory,
            "trajectory_mean": float(np.mean(finite)) if finite else None,
            "trajectory_std": float(np.std(finite)) if finite else None,
            "trajectories_scored": len(finite),
            "pooled": {
                "n": int(usable.sum()),
                "positives": int(y_true[usable].sum()),
                "pr_auc": float(E.average_precision(y_true[usable], score[usable]))
                if usable.any() else None,
            },
        }
    return result


def paired_comparison(spec_name, mlp_block, heuristic_block) -> dict:
    """Per-trajectory difference in PR-AUC between the MLP and the frozen heuristic."""
    rows = {}
    improved = worse = tied = 0
    deltas = []
    for trajectory_id, mlp in mlp_block["per_trajectory"].items():
        other = heuristic_block["per_trajectory"].get(trajectory_id, {})
        a, b = mlp.get("pr_auc"), other.get("pr_auc")
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            rows[trajectory_id] = {"mlp": a, "heuristic": b, "delta": None,
                                   "positives": mlp.get("positives", 0)}
            continue
        delta = float(a - b)
        rows[trajectory_id] = {"mlp": float(a), "heuristic": float(b), "delta": delta,
                               "positives": mlp.get("positives", 0)}
        deltas.append(delta)
        if delta > 1e-6:
            improved += 1
        elif delta < -1e-6:
            worse += 1
        else:
            tied += 1
    return {"output": spec_name, "per_trajectory": rows, "improved": improved,
            "worse": worse, "tied": tied,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "median_delta": float(np.median(deltas)) if deltas else None}


# --- main ---------------------------------------------------------------------

def run(argv: list[str] | None = None) -> dict:
    arguments = parse_args(argv)
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    base_feature_config = yaml.safe_load(
        Path(config["feature_config"]).read_text(encoding="utf-8"))
    output = Path(arguments.output or config["output"])
    output.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        if not arguments.quiet:
            print(message, flush=True)

    import torch
    # Thread count is from the config so a run is reproducible: results are
    # deterministic for a *fixed* thread count, and this model is small enough that
    # oversubscribing is actively harmful (12 threads measured ~8x slower than 1).
    threads = int(config["training"].get("threads", 1))
    torch.set_num_threads(threads)

    dataset = Path(config["dataset"])
    specs = output_specs(config)

    log("=" * 78)
    log("Phase 3b - visibility MLP vs the frozen heuristic")
    log("=" * 78)

    # --- data -----------------------------------------------------------------
    log("\n[1/6] loading features")
    table = D.build_table(dataset, base_feature_config)
    snap_yaw = snap_yaw_row_mask(table, Path(config["sidecar"]["eval_flags"]))
    log(f"      {len(table)} rows x {len(table.names)} features, "
        f"{len(table.trajectories)} trajectories")
    log(f"      snap-yaw rows: {int(snap_yaw.sum())} "
        f"({100.0 * snap_yaw.mean():.2f}%) - excluded from training loss and headline eval")
    log(f"      unified primary subset: {int(T.unified_primary(table).sum())} rows")

    tuned, phase3a = frozen_heuristic(config)
    catalogue = F.ObjectCatalogue.from_manifest(base_feature_config["paths"]["object_index"])
    cache = B.build_row_pose_cache(table, B.load_pose_arrays(dataset), catalogue.centre)
    heuristic_scores = {
        "frustum": B.predictive_score(table, cache, tuned["frustum"][0], 0.5,
                                      occlusion_aware=False),
        "occlusion": B.predictive_score(table, cache, tuned["occlusion"][0], 0.5,
                                        occlusion_aware=True),
    }
    # Variant B takes the heuristic as a single extra input; the frustum-tuned score is
    # *the* heuristic and is what it receives, for both of its targets.
    heuristic_score = heuristic_scores["frustum"]
    for target, (params, threshold) in tuned.items():
        log(f"      frozen heuristic [{target}]: damping={params.damping} "
            f"steps={params.integrate_steps} occlusion_penalty={params.occlusion_penalty_deg} "
            f"threshold={threshold:g}")

    if arguments.heuristic_only:
        return recompute_heuristic_reference(
            table, specs, heuristic_scores, snap_yaw, output, arguments, log)

    # History ablation needs its own feature table.
    ablation_feature_config = feature_config(base_feature_config, 0)
    log("      building the K=0 feature table for the ablation")
    table_k0 = D.build_table(dataset, ablation_feature_config)
    log(f"      K=0 table: {len(table_k0.names)} features "
        f"({len(table.names) - len(table_k0.names)} history columns removed)")

    # --- folds ----------------------------------------------------------------
    folds = S.build_folds(sorted(table.trajectories), int(base_feature_config["splits"]["folds"]),
                          int(base_feature_config["splits"]["fold_size"]),
                          int(base_feature_config["splits"]["group_seed"]))
    log(f"\n[2/6] {len(folds)} leave-trajectory-out folds")

    variant_keys = [v.strip() for v in arguments.variants.split(",")] if arguments.variants \
        else list(config["variants"])
    seeds = [int(s) for s in arguments.seeds.split(",")] if arguments.seeds \
        else [int(s) for s in config["training"]["seeds"]]

    results: dict = {
        "phase": "3b",
        "config": arguments.config,
        "dataset": str(dataset),
        "rows": len(table),
        "features": list(table.names),
        "feature_count": len(table.names),
        "outputs": [spec.name for spec in specs],
        "snap_yaw_rows": int(snap_yaw.sum()),
        "unified_primary_rows": int(T.unified_primary(table).sum()),
        "heuristic": {target: {"parameters": params.as_dict(), "threshold_deg": threshold,
                               "occlusion_aware": target == "occlusion"}
                      for target, (params, threshold) in tuned.items()},
        "heuristic_source": config["heuristic"]["results"],
        "heuristic_retuned": False,
        "seeds": seeds,
        "protocol": {
            "primary": "leave-trajectory-out grouped folds; epochs and threshold chosen on "
                       "the fold's validation trajectories only",
            "secondary": "split v1 test, scored once after all choices were frozen",
            "seeds_per_mlp_config": len(seeds),
        },
        "variants": {},
    }

    # --- primary protocol -----------------------------------------------------
    log(f"\n[3/6] primary protocol: {len(variant_keys)} variants x {len(seeds)} seeds")
    oof_scores: dict[str, dict[int, dict]] = {}
    for key in variant_keys:
        variant = config["variants"][key]
        log(f"\n  variant {key}: {variant['label']}")
        is_sklearn = variant["model_class"] == "sklearn_histgb"
        variant_table = table_k0 if int(variant.get("history_k", 4)) == 0 else table
        if variant_table is not table:
            snap_k0 = snap_yaw          # same rows, table_k0 is row-aligned
        else:
            snap_k0 = snap_yaw
        seed_results = {}
        for seed in seeds:
            started = time.time()
            per_fold = []
            for fold in folds:
                fold_started = time.time()
                if is_sklearn:
                    per_fold.append(run_fold_sklearn(
                        variant_table, specs, fold, config, seed, variant.get("sklearn", {})))
                else:
                    per_fold.append(run_fold_mlp(
                        variant_table, specs, fold, variant, config, seed, heuristic_score,
                        snap_k0))
                log(f"    seed {seed} fold {fold.index}/{len(folds)} "
                    f"test={list(fold.test)[0]}... "
                    f"{round(time.time() - fold_started, 1)}s")
            aggregate = aggregate_oof(variant_table, specs, per_fold, snap_k0, True)
            epochs = [entry["fit"].best_epoch for entry in per_fold if entry["fit"]]
            parameters = per_fold[0].get("parameters") or (
                per_fold[0]["fit"].parameters if per_fold[0]["fit"] else 0)
            seed_results[seed] = {
                "aggregate": aggregate,
                "best_epochs": epochs,
                "parameters": parameters,
                "seconds": round(time.time() - started, 2),
            }
            oof_scores.setdefault(key, {})[seed] = per_fold
            log(f"    seed {seed}: {round(time.time() - started, 1)}s, "
                f"epochs {epochs[:3]}{'...' if len(epochs) > 3 else ''}, "
                f"{parameters} parameters")
            for spec in specs:
                block = aggregate[spec.name]
                log(f"      {spec.name:<14} pooled PR-AUC {block['pooled'].get('pr_auc', float('nan')):.4f} "
                    f"traj-mean {block['trajectory_mean']:.4f}")

        # Mean and std across seeds, per output.
        across_seeds = {}
        for spec in specs:
            pooled = [seed_results[s]["aggregate"][spec.name]["pooled"].get("pr_auc")
                      for s in seeds]
            traj_mean = [seed_results[s]["aggregate"][spec.name]["trajectory_mean"] for s in seeds]
            pooled = [v for v in pooled if isinstance(v, (int, float)) and np.isfinite(v)]
            traj_mean = [v for v in traj_mean if isinstance(v, (int, float)) and np.isfinite(v)]
            across_seeds[spec.name] = {
                "pooled_pr_auc_mean": float(np.mean(pooled)) if pooled else None,
                "pooled_pr_auc_std": float(np.std(pooled)) if pooled else None,
                "trajectory_mean_pr_auc_mean": float(np.mean(traj_mean)) if traj_mean else None,
                "trajectory_mean_pr_auc_std": float(np.std(traj_mean)) if traj_mean else None,
            }
        results["variants"][key] = {
            "label": variant["label"],
            "model_class": variant["model_class"],
            "history_k": int(variant.get("history_k", 4)),
            "heuristic_input": bool(variant.get("heuristic_input", False)),
            "per_seed": {str(s): {"parameters": seed_results[s]["parameters"],
                                  "best_epochs": seed_results[s]["best_epochs"],
                                  "seconds": seed_results[s]["seconds"],
                                  "aggregate": seed_results[s]["aggregate"]}
                         for s in seeds},
            "across_seeds": across_seeds,
        }
        # Persist after every variant so a later failure cannot discard earlier work.
        (output / "results.partial.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8")
        log(f"    (partial results written after variant {key})")

    # --- heuristic reference and paired comparison -----------------------------
    log("\n[4/6] frozen heuristic reference and paired comparison")
    heuristic = heuristic_metrics(table, specs, heuristic_scores, snap_yaw, True)
    results["heuristic_reference"] = heuristic
    for key in variant_keys:
        paired = {}
        for spec in specs:
            # Use the seed closest to the variant's mean, so the comparison is one model.
            seed = seeds[0]
            paired[spec.name] = paired_comparison(
                spec.name, results["variants"][key]["per_seed"][str(seed)]["aggregate"][spec.name],
                heuristic[spec.name])
        results["variants"][key]["paired_vs_heuristic"] = paired
        for spec in specs:
            p = paired[spec.name]
            log(f"  {key} {spec.name:<14} improved {p['improved']}, worse {p['worse']}, "
                f"tied {p['tied']}, mean delta {p['mean_delta']}")

    # --- snap-affected rows, reported separately -------------------------------
    log("\n[5/6] snap-affected rows reported separately")
    snap_report = {}
    for spec in specs:
        _, valid, primary = E.target_arrays(table, spec.target, spec.horizon_index)
        y_true, _, _ = E.target_arrays(table, spec.target, spec.horizon_index)
        mask = valid & primary & snap_yaw
        snap_report[spec.name] = {
            "rows": int(mask.sum()),
            "positives": int(y_true[mask].sum()),
            "heuristic_pr_auc": float(E.average_precision(y_true[mask], heuristic_score[mask]))
            if mask.any() and y_true[mask].sum() else None,
        }
        if snap_report[spec.name]["rows"]:
            log(f"  {spec.name:<14} {snap_report[spec.name]['rows']} rows, "
                f"{snap_report[spec.name]['positives']} positives, "
                f"heuristic PR-AUC {snap_report[spec.name]['heuristic_pr_auc']}")
    results["snap_affected_report"] = snap_report

    # --- secondary: split v1, scored once --------------------------------------
    log("\n[6/6] secondary: split v1, scored once")
    split_v1 = json.loads(Path(config["split_v1"]).read_text(encoding="utf-8"))
    parts = {name: [entry["trajectory_id"] for entry in entries]
             for name, entries in split_v1["splits"].items()}
    v1_train = trajectory_rows(table, parts["train"]) & T.unified_primary(table)
    v1_validation = trajectory_rows(table, parts["validation"]) & T.unified_primary(table)
    v1_test = trajectory_rows(table, parts["test"]) & T.unified_primary(table)
    log(f"      split v1 rows: train {int(v1_train.sum())}, validation "
        f"{int(v1_validation.sum())}, test {int(v1_test.sum())}")

    v1_results = {}
    best_key, best_score = None, float("-inf")
    for key in variant_keys:
        variant = config["variants"][key]
        variant_table = table_k0 if int(variant.get("history_k", 4)) == 0 else table
        is_sklearn = variant["model_class"] == "sklearn_histgb"
        extra = heuristic_score.reshape(-1, 1) if variant.get("heuristic_input") else None
        bundle = T.build_bundle(variant_table, specs, v1_train,
                               drop_snap_yaw=config["masking"]["drop_snap_yaw_from_training"],
                               snap_yaw_rows=snap_yaw, extra_inputs=extra)
        if is_sklearn:
            y, _ = T.per_output_targets(variant_table, specs)
            X = variant_table.X.astype(np.float64)
            scores = np.zeros((len(variant_table), len(specs)))
            thresholds, parameters = {}, 0
            for column, spec in enumerate(specs):
                _, valid, primary = E.target_arrays(variant_table, spec.target,
                                                    spec.horizon_index)
                rows = v1_train & valid & primary
                model = T.train_sklearn(X, y[:, column], rows, seeds[0],
                                        variant.get("sklearn", {}))
                parameters += M.count_parameters(model)
                scores[:, column] = T.predict_sklearn(model, X)
                vrows = masked_rows(variant_table, spec, v1_validation, snap_yaw, True) & \
                    valid
                thresholds[spec.name] = T.choose_threshold(y[vrows, column],
                                                           scores[vrows, column]) \
                    if vrows.sum() else 0.5
            fit = None
            epochs = []
        else:
            fit = T.train_mlp(bundle, specs, v1_validation, variant_table, config, seeds[0],
                              exclude_snap=snap_yaw)
            scores = T.predict_mlp(fit.model, bundle)
            parameters = fit.parameters
            epochs = [fit.best_epoch]
            thresholds = {}
            for column, spec in enumerate(specs):
                rows = masked_rows(variant_table, spec, v1_validation, snap_yaw, True)
                y_true, _, _ = E.target_arrays(variant_table, spec.target, spec.horizon_index)
                thresholds[spec.name] = T.choose_threshold(y_true[rows], scores[rows, column]) \
                    if rows.sum() else 0.5

        blocks = {}
        for column, spec in enumerate(specs):
            rows = masked_rows(variant_table, spec, v1_test, snap_yaw, True)
            blocks[spec.name] = score_output(variant_table, spec, rows, scores[:, column],
                                             thresholds[spec.name])
        v1_results[key] = {"label": variant["label"], "parameters": parameters,
                           "best_epochs": epochs, "thresholds": thresholds,
                           "test": blocks}
        mean_auc = float(np.nanmean([blocks[spec.name].get("pr_auc", np.nan)
                                     for spec in specs]))
        summary = ", ".join(f"{spec.name} {_fmt(blocks[spec.name].get('pr_auc'))}"
                            for spec in specs)
        log(f"      {key}: {summary}")
        if np.isfinite(mean_auc) and mean_auc > best_score:
            best_score, best_key = mean_auc, key
    results["split_v1"] = {"manifest": config["split_v1"],
                           "manifest_sha256": __import__("hashlib").sha256(
                               Path(config["split_v1"]).read_bytes()).hexdigest(),
                           "partitions": {k: len(v) for k, v in parts.items()},
                           "scored_once_after_freezing": True,
                           "variants": v1_results,
                           "best_variant": best_key}

    # --- export ---------------------------------------------------------------
    log("\n[7/7] exporting weights")
    export_key = best_key or variant_keys[0]
    variant = config["variants"][export_key]
    variant_table = table_k0 if int(variant.get("history_k", 4)) == 0 else table
    extra = heuristic_score.reshape(-1, 1) if variant.get("heuristic_input") else None
    bundle = T.build_bundle(variant_table, specs, v1_train,
                            drop_snap_yaw=config["masking"]["drop_snap_yaw_from_training"],
                            snap_yaw_rows=snap_yaw, extra_inputs=extra)
    export_model = None
    if variant["model_class"] != "sklearn_histgb":
        final_fit = T.train_mlp(bundle, specs, v1_validation, variant_table, config, seeds[0],
                                exclude_snap=snap_yaw)
        export_model = final_fit.model
        stem = output / f"visibility_mlp_{export_key}_seed{seeds[0]}"
        export_info = M.export_bundle(export_model, bundle.feature_names,
                                      bundle.output_names, bundle.normaliser, stem)
        inference = T.time_inference(export_model, bundle,
                                     int(config["evaluation"]["inference_batch"]),
                                     int(config["evaluation"]["inference_repeats"]))
        results["model_size"] = {"parameters": M.count_parameters(export_model),
                                 "dense_layers": len(export_model.dense_layers()),
                                 "hidden": export_model.hidden,
                                 "input_dim": export_model.input_dim}
        results["inference"] = inference
        log(f"      {export_info['parameters']} parameters, "
            f"{inference['mean_ms']:.3f} ms for {inference['rows']} candidates "
            f"({inference['threads']} thread)")
        log(f"      -> {export_info['weights']}")
    else:
        export_info = {}
        results["model_size"] = {"parameters": v1_results[export_key]["parameters"]}
    results["export"] = {"variant": export_key, **export_info}

    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n",
                                         encoding="utf-8")
    write_csv(output / "results.csv", results)
    (output / "summary.md").write_text(render_markdown(results), encoding="utf-8")
    log(f"\n-> {output}")
    return results


def recompute_heuristic_reference(table, specs, heuristic_scores, snap_yaw, output,
                                  arguments, log) -> dict:
    """Rewrite the heuristic reference and paired comparisons without retraining.

    The MLP's out-of-fold PR-AUCs do not depend on the heuristic, so they are reused
    from the saved results. Only the baseline side is recomputed, with each target
    scored by its own tuned parameters.
    """
    path = output / "results.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run the full pipeline first")
    results = json.loads(path.read_text(encoding="utf-8"))
    seeds = [int(s) for s in results["seeds"]]

    heuristic = heuristic_metrics(table, specs, heuristic_scores, snap_yaw, True)
    results["heuristic_reference"] = heuristic
    results["heuristic_reference_correction"] = {
        "reason": "the occlusion target was previously scored with the frustum-tuned "
                  "heuristic and occlusion_aware=False, which handicapped the baseline",
        "now": "each target is scored with the parameters Phase 3a tuned for that target; "
               "the occlusion score uses occlusion_aware=True",
        "mlp_results_reused_not_retrained": True,
    }

    for key, variant in results["variants"].items():
        paired = {}
        for spec in specs:
            aggregate = variant["per_seed"][str(seeds[0])]["aggregate"][spec.name]
            paired[spec.name] = paired_comparison(spec.name, aggregate, heuristic[spec.name])
        variant["paired_vs_heuristic"] = paired

    (path).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "results.csv", results)
    (output / "summary.md").write_text(render_markdown(results), encoding="utf-8")

    log("\n  corrected heuristic reference (per-target tuned parameters):")
    for spec in specs:
        block = heuristic[spec.name]
        log(f"    {spec.name:<15} pooled PR-AUC {block['pooled']['pr_auc']:.4f}  "
            f"traj-mean {block['trajectory_mean']:.4f}")
    for key, variant in results["variants"].items():
        log(f"  {key}:")
        for spec in specs:
            p = variant["paired_vs_heuristic"][spec.name]
            log(f"    {spec.name:<15} mlp {p['mean_delta']:+.4f} mean delta vs heuristic "
                f"(improved {p['improved']}, worse {p['worse']}, tied {p['tied']})")
    log(f"\n-> {output} (heuristic reference recomputed; MLP results reused)")
    return results


def write_csv(path: Path, results: dict) -> None:
    rows = []
    for key, variant in results["variants"].items():
        for seed, payload in variant["per_seed"].items():
            for spec_name, block in payload["aggregate"].items():
                pooled = block.get("pooled", {})
                rows.append({"variant": key, "seed": seed, "output": spec_name,
                             "scope": "pooled",
                             **{k: v for k, v in pooled.items() if not isinstance(v, dict)}})
                rows.append({"variant": key, "seed": seed, "output": spec_name,
                             "scope": "trajectory_mean",
                             "pr_auc": block.get("trajectory_mean"),
                             "pr_auc_std": block.get("trajectory_std"),
                             "n": None, "positives": None})
        for spec_name, paired in variant.get("paired_vs_heuristic", {}).items():
            rows.append({"variant": key, "seed": "-", "output": spec_name,
                         "scope": "paired_vs_heuristic", "improved": paired["improved"],
                         "worse": paired["worse"], "tied": paired["tied"],
                         "mean_delta": paired["mean_delta"]})
    for spec_name, block in results.get("heuristic_reference", {}).items():
        rows.append({"variant": "heuristic", "seed": "-", "output": spec_name,
                     "scope": "pooled", "pr_auc": block["pooled"].get("pr_auc"),
                     "n": block["pooled"].get("n"),
                     "positives": block["pooled"].get("positives"),
                     "trajectory_mean": block.get("trajectory_mean")})
    for key, payload in results.get("split_v1", {}).get("variants", {}).items():
        for spec_name, block in payload["test"].items():
            rows.append({"variant": key, "seed": "-", "output": spec_name,
                         "scope": "split_v1_test",
                         **{k: v for k, v in block.items() if not isinstance(v, dict)}})
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(results: dict) -> str:
    lines = ["# Phase 3b results", "",
             f"- rows: {results['rows']}, features: {results['feature_count']}",
             f"- snap-yaw rows excluded: {results['snap_yaw_rows']}",
             f"- seeds: {results['seeds']}",
             f"- heuristic: frozen from `{results.get('heuristic_source')}`, retuned: "
             f"{results.get('heuristic_retuned')}", "",
             "Heuristic parameters, one tuned set per target:",
             f"```\n{json.dumps(results.get('heuristic', {}), indent=2)}\n```", ""]
    for key, variant in results["variants"].items():
        lines.append(f"## Variant {key} - {variant['label']}")
        lines.append("")
        lines.append("| output | pooled PR-AUC (mean over seeds) | traj-mean PR-AUC |")
        lines.append("|---|---|---|")
        for spec_name, block in variant["across_seeds"].items():
            lines.append(f"| {spec_name} | "
                         f"{_fmt(block['pooled_pr_auc_mean'])} ± {_fmt(block['pooled_pr_auc_std'])} | "
                         f"{_fmt(block['trajectory_mean_pr_auc_mean'])} ± "
                         f"{_fmt(block['trajectory_mean_pr_auc_std'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    run()
