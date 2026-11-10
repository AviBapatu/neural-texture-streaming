"""Run the Phase 3a baselines and write the experiment results.

    python training/run_baselines.py \
        --config training/configs/features_v1.yaml \
        --dataset dataset/processed/run_full_p25 \
        --split dataset/manifests/runs/run_full_p25/splits.json

Hyper-parameter selection uses the split manifest's **validation** partition only.
The test partition is never scored during selection. The grouped folds are reported
for variance; because they rotate every trajectory through the test role, fold-test
metrics are a variance estimate, not a selection signal - which is why selection is
done on the fixed validation split instead.

No model is defined, trained or persisted. The only artefacts are JSON/CSV results,
normalisation statistics for Phase 3b, and a markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import baselines as B
from training import dataset as D
from training import evaluate as E
from training import features as F
from training import splits as S

EXPERIMENT_DIR = Path("training/experiments/phase3a")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3a non-neural baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default=None,
                        help="approved experimental split manifest; optional")
    parser.add_argument("--phase2_5-split", dest="phase2_5_split", default=None,
                        help="override the Phase 2.5 split manifest path")
    parser.add_argument("--output", default=str(EXPERIMENT_DIR))
    parser.add_argument("--max-tuning-combinations", type=int, default=0,
                        help="0 = full grid")
    return parser.parse_args(argv)


def trajectory_mask(table, wanted: set[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(table.trajectories)}
    indices = {lookup[name] for name in wanted if name in lookup}
    return np.isin(table.trajectory, list(indices)) if indices else np.zeros(len(table), dtype=bool)


def score_bundle(table, cache: B.RowPoseCache, horizon: float,
                 chosen: dict[str, B.HeuristicParams]) -> dict[str, np.ndarray]:
    """Every baseline's score for one horizon on one row subset.

    The frustum and occlusion targets have separately tuned parameters, so the two
    predictive scores differ by more than the occlusion penalty.
    """
    return {
        "reactive": B.reactive_score(table, "frustum"),
        "reactive_occlusion": B.reactive_score(table, "occlusion"),
        "predictive": B.predictive_score(table, cache, chosen["frustum"], horizon,
                                         occlusion_aware=False),
        "predictive_occlusion": B.predictive_score(table, cache, chosen["occlusion"], horizon,
                                                   occlusion_aware=True),
    }


def run(argv: list[str] | None = None) -> dict:
    arguments = parse_args(argv)
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    if arguments.phase2_5_split:
        config["paths"]["phase2_5_split"] = arguments.phase2_5_split

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    log = print

    log("=" * 78)
    log("Phase 3a - leakage-safe features, evaluation harness, non-neural baselines")
    log("=" * 78)

    # --- STEP 1: load ----------------------------------------------------------
    log("\n[1/7] loading dataset and building features")
    table = D.build_table(arguments.dataset, config)
    poses = B.load_pose_arrays(arguments.dataset)
    catalogue = F.ObjectCatalogue.from_manifest(config["paths"]["object_index"])
    cache = B.build_row_pose_cache(table, poses, catalogue.centre)
    log(f"      {len(table)} rows x {len(table.names)} features "
        f"across {len(table.trajectories)} trajectories")
    log(f"      horizons {table.horizons}")

    # --- STEP 2: splits --------------------------------------------------------
    log("\n[2/7] resolving splits")
    phase2_5 = S.load_split_manifest(config["paths"]["phase2_5_split"])
    log("      phase 2.5 split:")
    for line in S.summary_lines(phase2_5):
        log(line)

    experimental_path = arguments.split or config["splits"].get("experimental")
    experimental = S.load_experimental_split(experimental_path)
    if experimental is None:
        log(f"      experimental split manifest: NOT FOUND "
            f"({experimental_path!r}) - grouped folds used instead")
    elif arguments.split and Path(arguments.split).resolve() == \
            Path(config["paths"]["phase2_5_split"]).resolve():
        # The command line names a path that is in fact the Phase 2.5 manifest. Say
        # so rather than reporting an "experimental" manifest that does not exist.
        log(f"      NOTE: --split {arguments.split} is the same file as the Phase 2.5 "
            f"manifest; no separate approved experimental manifest exists.")
    else:
        log(f"      experimental split manifest loaded from {experimental_path}")

    # Folds are built over the trajectories actually present in the dataset, so a
    # dataset slice (as used by the tests) folds over its own trajectories.
    present = sorted(set(table.trajectories))
    folds = S.build_folds(present, int(config["splits"]["folds"]),
                          int(config["splits"]["fold_size"]),
                          int(config["splits"]["group_seed"]))
    log(f"      {len(folds)} grouped folds (seed {config['splits']['group_seed']}):")
    for fold in folds:
        log(f"        fold {fold.index}: test={list(fold.test)} validation={list(fold.validation)}")

    second_seed = S.find_second_seed_dataset(config["splits"].get("second_seed_dataset"))
    log(f"      second-seed dataset: {second_seed if second_seed else 'NOT PRESENT - skipped'}")

    train_rows = trajectory_mask(table, {t for t, v in phase2_5.items() if v == "train"})
    validation_rows = trajectory_mask(table, {t for t, v in phase2_5.items() if v == "validation"})
    test_rows = trajectory_mask(table, {t for t, v in phase2_5.items() if v == "test"})
    log(f"      rows: train={int(train_rows.sum())} validation={int(validation_rows.sum())} "
        f"test={int(test_rows.sum())}")
    assert not (train_rows & validation_rows).any()
    assert not (train_rows & test_rows).any()
    assert not (validation_rows & test_rows).any()

    # --- STEP 3: normalisers ---------------------------------------------------
    log("\n[3/7] fitting normalisation statistics on training rows only")
    normaliser_dir = output / "normalisers"
    fold_normalisers = []
    for fold in folds:
        rows = trajectory_mask(table, set(fold.train))
        normaliser = D.fit_normaliser(table, rows)
        D.save_normaliser(normaliser, normaliser_dir / f"fold_{fold.index}.json")
        fold_normalisers.append(normaliser)
        log(f"      fold {fold.index}: {normaliser.fit_rows} training rows, "
            f"{len(normaliser.names)} features")
    final_normaliser = D.fit_normaliser(table, train_rows)
    D.save_normaliser(final_normaliser, normaliser_dir / "final.json")
    log(f"      final (phase 2.5 train split): {final_normaliser.fit_rows} rows -> "
        f"{normaliser_dir / 'final.json'}")

    # --- STEP 4: tuning on validation only -------------------------------------
    log("\n[4/7] tuning the predictive heuristic on VALIDATION rows only")
    validation_table = table.subset(validation_rows)
    validation_cache = B.build_row_pose_cache(validation_table, poses, catalogue.centre)
    grid = B.default_parameter_grid(config)
    thresholds = B.margin_thresholds(config)
    log(f"      grid: {json.dumps(grid, sort_keys=True)}")
    log(f"      margin thresholds: {thresholds}")

    tuning_log = []
    for occlusion_aware in (False, True):
        for params in B.iter_parameter_grid(grid, occlusion_aware=occlusion_aware):
            entry = {"occlusion_aware": occlusion_aware, **params.as_dict(), "per_horizon": {}}
            scores = 0.0
            counted = 0
            for column, horizon in enumerate(validation_table.horizons):
                score = B.predictive_score(validation_table, validation_cache, params, horizon,
                                           occlusion_aware=occlusion_aware)
                target = "occlusion" if occlusion_aware else "frustum"
                y_true, valid, primary = E.target_arrays(validation_table, target, column)
                mask = valid & primary
                value = E.average_precision(y_true[mask], score[mask])
                entry["per_horizon"][f"{horizon:g}"] = float(value)
                if np.isfinite(value):
                    scores += value
                    counted += 1
            entry["selection_score"] = float(scores / counted) if counted else float("nan")
            tuning_log.append(entry)

    tuning_log.sort(key=lambda e: (-(e["selection_score"] if np.isfinite(e["selection_score"])
                                     else -1.0), e["occlusion_aware"], e["damping"],
                                   e["angular_damping"], e["integrate_steps"],
                                   e["occlusion_penalty_deg"]))

    # Each target gets its own parameters. Tuning a single global winner would let
    # the frustum target's optimum decide the occlusion penalty, and the best global
    # combination sets occlusion_penalty_deg to 0 - which would silently make the
    # occlusion-aware variant identical to the plain one and the comparison
    # meaningless.
    chosen: dict[str, B.HeuristicParams] = {}
    best_by_target: dict[str, dict] = {}
    for target, occlusion_aware in (("frustum", False), ("occlusion", True)):
        candidates = [entry for entry in tuning_log if entry["occlusion_aware"] == occlusion_aware]
        winner = candidates[0]
        best_by_target[target] = winner
        chosen[target] = B.HeuristicParams(
            damping=winner["damping"], angular_damping=winner["angular_damping"],
            integrate_steps=winner["integrate_steps"], margin_threshold_deg=0.0,
            occlusion_penalty_deg=winner["occlusion_penalty_deg"])
        log(f"      chosen for {target}: damping={winner['damping']} "
            f"angular_damping={winner['angular_damping']} "
            f"integrate_steps={winner['integrate_steps']} "
            f"occlusion_penalty_deg={winner['occlusion_penalty_deg']} "
            f"(validation PR-AUC on the primary subset {winner['selection_score']:.4f})")
    log(f"      {len(tuning_log)} combinations evaluated")

    # The decision threshold is chosen on validation too, per target and horizon.
    chosen_thresholds = {"frustum": {}, "occlusion": {}}
    for column, horizon in enumerate(validation_table.horizons):
        for target, occlusion_aware in (("frustum", False), ("occlusion", True)):
            score = B.predictive_score(validation_table, validation_cache, chosen[target],
                                       horizon, occlusion_aware=occlusion_aware)
            y_true, valid, primary = E.target_arrays(validation_table, target, column)
            mask = valid & primary
            best_f1, best_threshold = -1.0, 0.0
            for threshold in thresholds:
                predicted = B.threshold_predictions(score[mask], threshold)
                metrics = E.binary_metrics(y_true[mask], predicted)
                if metrics["f1"] > best_f1:
                    best_f1, best_threshold = metrics["f1"], threshold
            chosen_thresholds[target][f"{horizon:g}"] = float(best_threshold)
            log(f"      threshold {target} @ {horizon:g}s: {best_threshold:g} "
                f"(validation F1 on primary subset {best_f1:.4f})")

    # --- STEP 5: evaluate on folds ---------------------------------------------
    log("\n[5/7] evaluating on each fold's held-out test partition")
    fold_results = []
    for fold in folds:
        rows = trajectory_mask(table, set(fold.test))
        subset = table.subset(rows)
        subset_cache = B.build_row_pose_cache(subset, poses, catalogue.centre)
        entry = {"fold": fold.index, "test_trajectories": list(fold.test),
                 "validation_trajectories": list(fold.validation),
                 "rows": int(rows.sum()), "horizons": {}, "resolution": None}
        for column, horizon in enumerate(subset.horizons):
            bundle = score_bundle(subset, subset_cache, horizon, chosen)
            entry["horizons"][f"{horizon:g}"] = evaluate_horizon(
                subset, [column][0], bundle, chosen_thresholds, horizon)
        entry["resolution"] = evaluate_resolution(subset, subset_cache, column=0)
        fold_results.append(entry)
        log(f"      fold {fold.index}: {int(rows.sum())} rows, test={list(fold.test)}")

    # --- STEP 6: evaluate on the fixed final test ------------------------------
    log("\n[6/7] evaluating on the fixed Phase 2.5 test split")
    test_table = table.subset(test_rows)
    test_cache = B.build_row_pose_cache(test_table, poses, catalogue.centre)
    final = {"test_trajectories": sorted({test_table.trajectories[i]
                                          for i in test_table.trajectory}),
             "rows": len(test_table),
             "horizons": {}, "resolution": None}
    for column, horizon in enumerate(test_table.horizons):
        bundle = score_bundle(test_table, test_cache, horizon, chosen)
        final["horizons"][f"{horizon:g}"] = evaluate_horizon(
            test_table, column, bundle, chosen_thresholds, horizon)
    final["resolution"] = evaluate_resolution(test_table, test_cache, column=0)
    log(f"      {final['rows']} rows, trajectories {final['test_trajectories']}")

    # --- STEP 7: write ---------------------------------------------------------
    log("\n[7/7] writing results")
    summary = {
        "phase": "3a",
        "config": arguments.config,
        "dataset": arguments.dataset,
        "row_count": len(table),
        "feature_count": len(table.names),
        "features": list(table.names),
        "horizons": table.horizons,
        "splits": {
            "phase2_5_manifest": config["paths"]["phase2_5_split"],
            "experimental_manifest": experimental_path if experimental else None,
            "experimental_manifest_found": experimental is not None,
            "phase2_5": _split_counts(phase2_5),
            "fold_seed": config["splits"]["group_seed"],
            "folds": [fold.as_dict() for fold in folds],
            "second_seed_dataset": second_seed,
        },
        "tuning": {
            "selection_data": "phase 2.5 validation partition only",
            "test_rows_scored_during_selection": 0,
            "grid": grid,
            "margin_thresholds_deg": thresholds,
            "combinations": len(tuning_log),
            "top_combinations": tuning_log[:10],
            "chosen": {target: params.as_dict() for target, params in chosen.items()},
            "chosen_validation_score": {
                target: best_by_target[target]["selection_score"] for target in chosen},
            "chosen_thresholds_deg": chosen_thresholds,
        },
        "folds": fold_results,
        "final_test": final,
        "fold_summary": summarise_folds(fold_results),
        "unknown_occlusion": {
            "current_unknown_rows": int((table.occlusion_state == -1).sum()),
            "target_unknown_rows_0_5s": int((table.y_occlusion[:, 0] == -1).sum()),
            "target_unknown_rows_1_0s": int((table.y_occlusion[:, 1] == -1).sum()),
            "policy": "excluded from occlusion metrics and reported here; never treated as negative",
        },
        "normalisers": {
            "final": str(normaliser_dir / "final.json"),
            "folds": [str(normaliser_dir / f"fold_{fold.index}.json") for fold in folds],
        },
    }

    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "tuning.json").write_text(json.dumps(tuning_log, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "results.csv", summary)
    (output / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    log(f"      -> {output / 'results.json'}")
    log(f"      -> {output / 'results.csv'}")
    log(f"      -> {output / 'tuning.json'}")
    log(f"      -> {output / 'summary.md'}")
    log("\ndone")
    return summary


def _split_counts(assignment: dict[str, str]) -> dict:
    counts: dict[str, list[str]] = {}
    for trajectory_id, name in sorted(assignment.items()):
        counts.setdefault(name, []).append(trajectory_id)
    return {name: {"count": len(ids), "trajectories": ids} for name, ids in sorted(counts.items())}


def evaluate_horizon(table, column: int, bundle: dict[str, np.ndarray],
                     thresholds: dict, horizon: float) -> dict:
    """All baselines for one horizon, per target."""
    result: dict = {}
    for target in ("frustum", "occlusion"):
        y_true, valid, primary = E.target_arrays(table, target, column)
        result[target] = {
            "valid_rows": int(valid.sum()),
            "excluded_unknown_rows": int((~valid).sum()),
            "primary_subset_rows": int((valid & primary).sum()),
            "baselines": {},
        }
        reactive_key = "reactive" if target == "frustum" else "reactive_occlusion"
        predictive_key = "predictive" if target == "frustum" else "predictive_occlusion"
        for name, key in (("reactive", reactive_key), ("predictive", predictive_key)):
            score = bundle[key]
            threshold = thresholds[target][f"{horizon:g}"]
            predicted = B.threshold_predictions(score, threshold) if name == "predictive" \
                else (score >= 1.0)
            masked_valid = valid
            evaluation = E.evaluate_target(
                table.subset(masked_valid), y_true[masked_valid], predicted[masked_valid],
                score[masked_valid], primary[masked_valid],
                percentiles=tuple(int(k) for k in (5, 10, 20)))
            evaluation["threshold_deg"] = threshold if name == "predictive" else None
            result[target]["baselines"][name] = evaluation
    return result


def evaluate_resolution(table, cache: B.RowPoseCache, column: int) -> dict:
    """Coverage-threshold resolution baseline against the future requirement."""
    y_true = table.y_resolution[:, column]
    eligible = np.isfinite(y_true)
    if not eligible.any():
        return {"n": 0}
    predicted = B.coverage_resolution_predictions(table, cache)
    metrics = E.resolution_metrics(y_true[eligible].astype(np.int64),
                                   predicted[eligible])
    metrics["eligible_rows"] = int(eligible.sum())
    metrics["excluded_ineligible_rows"] = int((~eligible).sum())
    metrics["target"] = "frustum_required_resolution"
    return metrics


def summarise_folds(fold_results: list[dict]) -> dict:
    summary = {}
    for horizon_key in sorted({k for fold in fold_results for k in fold["horizons"]}):
        block = {}
        for target in ("frustum", "occlusion"):
            per_baseline = {}
            for baseline in ("reactive", "predictive"):
                node = {}
                for path in (("full", "pr_auc"), ("primary_subset", "pr_auc"),
                             ("primary_subset", "precision"), ("primary_subset", "recall"),
                             ("primary_subset", "f1")):
                    values = []
                    for fold in fold_results:
                        entry = fold["horizons"].get(horizon_key, {}).get(target, {}) \
                            .get("baselines", {}).get(baseline, {})
                        node_value = entry
                        for key in path:
                            node_value = node_value.get(key, {}) if isinstance(node_value, dict) else {}
                        if isinstance(node_value, (int, float)) and np.isfinite(node_value):
                            values.append(float(node_value))
                    node["_".join(path)] = {
                        "mean": float(np.mean(values)) if values else float("nan"),
                        "std": float(np.std(values)) if values else float("nan"),
                        "folds": len(values),
                    }
                per_baseline[baseline] = node
            block[target] = per_baseline
        summary[horizon_key] = block

    resolution_under = [fold["resolution"].get("under_prediction_rate") for fold in fold_results
                        if isinstance(fold.get("resolution"), dict)]
    resolution_under = [v for v in resolution_under if isinstance(v, (int, float))
                        and np.isfinite(v)]
    summary["resolution"] = {
        "under_prediction_rate": {
            "mean": float(np.mean(resolution_under)) if resolution_under else float("nan"),
            "std": float(np.std(resolution_under)) if resolution_under else float("nan"),
            "folds": len(resolution_under),
        }
    }
    return summary


def write_csv(path: Path, summary: dict) -> None:
    rows = []
    for scope, block in (("fold", {"folds": summary["folds"]}),
                         ("final_test", summary["final_test"])):
        if scope == "fold":
            for fold in summary["folds"]:
                for horizon_key, targets in fold["horizons"].items():
                    for target, info in targets.items():
                        for baseline, evaluation in info["baselines"].items():
                            rows.extend(_csv_rows(
                                f"fold{fold['fold']}.{target}.{baseline}.{horizon_key}",
                                evaluation))
                if isinstance(fold.get("resolution"), dict):
                    rows.extend(_csv_rows(f"fold{fold['fold']}.resolution", fold["resolution"]))
        else:
            for horizon_key, targets in block["horizons"].items():
                for target, info in targets.items():
                    for baseline, evaluation in info["baselines"].items():
                        rows.extend(_csv_rows(f"final.{target}.{baseline}.{horizon_key}", evaluation))
            if isinstance(block.get("resolution"), dict):
                rows.extend(_csv_rows("final.resolution", block["resolution"]))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "metric", "value"])
        for row in rows:
            writer.writerow([row["scope"], row["metric"], row["value"]])


def _csv_rows(scope: str, evaluation: dict, prefix: str = "") -> list[dict]:
    rows = []
    for key, value in evaluation.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(_csv_rows(scope, value, prefix=f"{name}."))
        elif isinstance(value, list):
            continue
        else:
            rows.append({"scope": scope, "metric": name, "value": value})
    return rows


def render_markdown(summary: dict) -> str:
    lines = ["# Phase 3a results", "",
             f"- rows: {summary['row_count']}, features: {summary['feature_count']}",
             f"- horizons: {summary['horizons']}",
             f"- selection data: {summary['tuning']['selection_data']}",
             f"- chosen heuristic: `{json.dumps(summary['tuning']['chosen'], sort_keys=True)}`",
             f"- chosen thresholds (deg): `{json.dumps(summary['tuning']['chosen_thresholds_deg'])}`",
             "", "## Fixed final test", ""]
    lines.append("| horizon | target | baseline | n | pos rate | precision | recall | F1 | PR-AUC |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for horizon_key, targets in summary["final_test"]["horizons"].items():
        for target, info in targets.items():
            for baseline, evaluation in info["baselines"].items():
                full = evaluation["full"]
                lines.append(
                    f"| {horizon_key} | {target} | {baseline} | {full['n']} | "
                    f"{full['positive_rate']:.3f} | {full['precision']:.3f} | "
                    f"{full['recall']:.3f} | {full['f1']:.3f} | {full['pr_auc']:.4f} |")
    lines += ["", "## Fixed final test - primary subset (not currently visible)", ""]
    lines.append("| horizon | target | baseline | n | pos rate | precision | recall | F1 | PR-AUC |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for horizon_key, targets in summary["final_test"]["horizons"].items():
        for target, info in targets.items():
            for baseline, evaluation in info["baselines"].items():
                sub = evaluation["primary_subset"]
                lines.append(
                    f"| {horizon_key} | {target} | {baseline} | {sub['n']} | "
                    f"{sub['positive_rate']:.3f} | {sub['precision']:.3f} | "
                    f"{sub['recall']:.3f} | {sub['f1']:.3f} | {sub['pr_auc']:.4f} |")
    resolution = summary["final_test"].get("resolution", {})
    if resolution.get("n"):
        lines += ["", "## Fixed final test - resolution", "",
                  f"- rows scored: {resolution['n']}",
                  f"- macro-F1: {resolution['macro_f1']:.4f}",
                  f"- under-prediction rate: {resolution['under_prediction_rate']:.4f}",
                  f"- exact match: {resolution['exact_resolution_rate']:.4f}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run()
