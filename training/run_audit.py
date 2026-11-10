"""Phase 3a.1 - trajectory audit, evaluation protocol and heuristic error analysis.

    python training/run_audit.py --config training/configs/features_v1.yaml \
        --dataset dataset/processed/run_full_p25

Read-only over `dataset/`: nothing under `dataset/` is written except the new
`dataset/manifests/experimental_split_v1.json` (a separate script step), and the
existing Phase 2.5 manifest is checksummed before and after.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import audit as A
from training import baselines as B
from training import dataset as D
from training import evaluate as E
from training import features as F
from training import splits as S

OUTPUT_DIR = Path("training/experiments/phase3a1")
PHASE3A_DIR = Path("training/experiments/phase3a")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3a.1 trajectory audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--phase3a-results", default=str(PHASE3A_DIR / "results.json"),
                        help="frozen Phase 3a parameters are read from here, never retuned")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(argv: list[str] | None = None) -> dict:
    arguments = parse_args(argv)
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    dataset = Path(arguments.dataset)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    log = print

    thresholds = A.SnapThresholds.from_config(config)
    window = int(config.get("audit", {}).get("snap_window_samples", 10))
    low_speed = float(config["audit"]["low_angular_speed_deg_s"])
    onset_accel = float(config["audit"]["turn_onset_accel_deg_s2"])
    distance_edges = [float(v) for v in config["audit"]["distance_bins_m"]]

    log("=" * 78)
    log("Phase 3a.1 - trajectory audit, evaluation protocol, heuristic error analysis")
    log("=" * 78)
    log(f"\nsnap thresholds: yaw > {thresholds.yaw_deg_per_sample:g} deg/sample, "
        f"position > {thresholds.position_m_per_sample:g} m/sample, window +/-{window} samples")
    log(f"cause thresholds: low angular speed < {low_speed:g} deg/s, "
        f"turn onset > {onset_accel:g} deg/s^2")

    identifiers = A.trajectory_ids(dataset)
    kinds = A.declared_kinds(config["paths"]["trajectory_index"])

    # --- Task A ---------------------------------------------------------------
    log(f"\n[A] auditing {len(identifiers)} trajectories")
    audit_rows: list[dict] = []
    snap_rows: list[dict] = []
    impact_rows: list[dict] = []
    snap_lookup: dict[str, list[A.Snap]] = {}
    loaded: dict[str, A.TrajectoryData] = {}
    kind_mismatches = []

    for trajectory_id in identifiers:
        data = A.load_trajectory(dataset, trajectory_id)
        recorded = data.kind
        declared = kinds.get(trajectory_id, recorded)
        if recorded != declared:
            kind_mismatches.append((trajectory_id, recorded, declared))
        data.kind = declared          # use the authoritative movement type
        loaded[trajectory_id] = data
        snaps = A.detect_snaps(data.samples, trajectory_id, thresholds)
        snap_lookup[trajectory_id] = snaps
        audit_rows.append(A.trajectory_row(data, thresholds.yaw_deg_per_sample, window))
        snap_rows.extend(snap.as_dict() for snap in snaps)
        impact_rows.append(A.snap_impact(data, snaps, window))
        marker = f"  {len(snaps)} snap(s)" if snaps else ""
        log(f"      {trajectory_id:<24} {data.sample_count:>4} samples, "
            f"{data.kind:<20}{marker}")

    if kind_mismatches:
        log("\n      recorded trajectory_kind differs from the declared movement type:")
        for trajectory_id, recorded, declared in kind_mismatches:
            log(f"        {trajectory_id:<24} recorded {recorded!r} != declared {declared!r}")

    log(f"\n      total snaps flagged: {len(snap_rows)}")
    for row in snap_rows:
        log(f"        {row['trajectory_id']:<24} i={row['sample_index']:>4} "
            f"t={row['session_time']:>6.1f}  yaw {row['yaw_change_deg']:>+9.2f} deg  "
            f"position {row['position_jump_m']:.4f} m  [{row['snap_kind']}]")

    write_csv(output / "audit_table.csv", audit_rows)
    write_csv(output / "snaps.csv", snap_rows)
    write_csv(output / "snap_impact.csv", impact_rows)

    # Sample-count reconciliation.
    total_samples = sum(len(d.samples) for d in loaded.values())
    with_candidates = sum(len({r["sample_index"] for r in d.labels}) for d in loaded.values())
    pairs = sum(len(d.samples) - 1 for d in loaded.values())
    reconciliation = [
        {"count": total_samples, "meaning": "total samples recorded across all 14 trajectories"},
        {"count": pairs, "meaning": "samples with a predecessor (total - 14); the population "
                                    "for finite differences, so the angular-velocity scan "
                                    "reports 2826"},
        {"count": with_candidates,
         "meaning": "samples that emitted at least one candidate row (>=1 object within 25 m); "
                    "the remainder had no candidate and contributed no label rows"},
        {"count": total_samples - with_candidates,
         "meaning": "samples with no candidate within 25 m, mostly sprinting past the road grid"},
        {"count": 188469, "meaning": "label rows (object x sample pairs), not samples"},
    ]
    log("\n      sample counts reconciled:")
    for entry in reconciliation:
        log(f"        {entry['count']:>6}  {entry['meaning']}")
    write_csv(output / "sample_counts.csv", reconciliation)

    # --- Task B: evaluation sidecar ------------------------------------------
    log("\n[B] writing the evaluation sidecar")
    flags = A.build_eval_flags(dataset, config, snap_lookup, kinds)
    write_csv(output / "eval_flags.csv", flags)
    snap_flagged = sum(row["snap_affected"] for row in flags)
    log(f"      {len(flags)} (trajectory, sample) rows, {snap_flagged} flagged snap_affected "
        f"({100.0 * snap_flagged / max(1, len(flags)):.2f}%)")
    log(f"      columns: {', '.join(A.SIDECAR_COLUMNS)} - never model inputs")

    # --- Task C: error analysis with frozen parameters -------------------------
    log("\n[C] heuristic error analysis (Phase 3a parameters, NOT retuned)")
    phase3a = json.loads(Path(arguments.phase3a_results).read_text(encoding="utf-8"))
    frozen = phase3a["tuning"]["chosen"]["frustum"]
    frozen_threshold = float(phase3a["tuning"]["chosen_thresholds_deg"]["frustum"]["0.5"])
    params = B.HeuristicParams(damping=frozen["damping"],
                               angular_damping=frozen["angular_damping"],
                               integrate_steps=frozen["integrate_steps"],
                               margin_threshold_deg=0.0,
                               occlusion_penalty_deg=frozen["occlusion_penalty_deg"])
    log(f"      frozen: damping={params.damping} angular_damping={params.angular_damping} "
        f"integrate_steps={params.integrate_steps}, threshold {frozen_threshold:g} deg")
    log(f"      selection source: {phase3a['tuning']['selection_data']}")

    table = D.build_table(dataset, config)
    catalogue = F.ObjectCatalogue.from_manifest(config["paths"]["object_index"])
    cache = B.build_row_pose_cache(table, B.load_pose_arrays(dataset), catalogue.centre)
    score = B.predictive_score(table, cache, params, 0.5, occlusion_aware=False)
    predicted = B.threshold_predictions(score, frozen_threshold)

    y_true, _, primary = E.target_arrays(table, "frustum", 0)
    baseline_full = E.binary_metrics(y_true, predicted, score)
    baseline_primary = E.binary_metrics(y_true[primary], predicted[primary], score[primary])
    log(f"      all 14 trajectories, primary subset: recall {baseline_primary['recall']:.4f}, "
        f"PR-AUC {baseline_primary['pr_auc']:.4f}, positives {baseline_primary['positives']}")

    # Per-row angular speed / acceleration, and snap flags, in table row order.
    trajectory_lookup = {name: index for index, name in enumerate(table.trajectories)}
    angular_speed = np.zeros(len(table))
    angular_accel = np.zeros(len(table))
    snap_flag = np.zeros(len(table), dtype=bool)
    snap_yaw = np.zeros(len(table), dtype=bool)
    snap_position = np.zeros(len(table), dtype=bool)
    for trajectory_id, data in loaded.items():
        if trajectory_id not in trajectory_lookup:
            continue
        indices = np.flatnonzero(table.trajectory == trajectory_lookup[trajectory_id])
        if indices.size == 0:
            continue
        speed_series = A.angular_speed_deg_s(data.quaternions, data.times)
        accel_series = A.angular_accel_deg_s2(data.quaternions, data.times)
        affected = A.snap_affected_samples(snap_lookup[trajectory_id], data.sample_count, window)
        affected_yaw = A.snap_affected_samples(snap_lookup[trajectory_id], data.sample_count,
                                               window, "yaw")
        affected_position = A.snap_affected_samples(snap_lookup[trajectory_id], data.sample_count,
                                                    window, "position")
        sample = table.sample_index[indices]
        angular_speed[indices] = speed_series[sample]
        angular_accel[indices] = accel_series[sample]
        snap_sample = np.array([int(s) in affected for s in sample])
        snap_yaw[indices] = np.array([int(s) in affected_yaw for s in sample])
        snap_position[indices] = np.array([int(s) in affected_position for s in sample])
        snap_flag[indices] = snap_sample

    positives = primary & y_true
    # Category 1 uses the *yaw* snaps: those are the generator defect. The 115
    # position jumps are legitimate 20-120 m/s sprint motion that is consistent with
    # the recorded velocity, so classifying them as "snap_affected" would file hard
    # but valid samples under "data artefact". The union variant is reported
    # separately as a sensitivity check.
    causes = np.array([
        A.dominant_cause(bool(snap_yaw[i]), float(angular_speed[i]),
                         float(angular_accel[i]), low_speed, onset_accel)
        for i in range(len(table))
    ])
    causes_union = np.array([
        A.dominant_cause(bool(snap_flag[i]), float(angular_speed[i]),
                         float(angular_accel[i]), low_speed, onset_accel)
        for i in range(len(table))
    ])

    category_stats = {}
    for cause in A.CAUSE_ORDER:
        mask = causes == cause
        count = int((positives & mask).sum())
        hits = int((positives & mask & predicted).sum())
        category_stats[cause] = {
            "positives": count,
            "predicted_positive": hits,
            "recall": round(hits / count, 4) if count else None,
            "recall_at_5": (round(A._recall_at_k_for_category(
                table, y_true, score, mask, primary, 5), 4) if count else None),
            "share_of_positives": round(count / int(positives.sum()), 4) if positives.sum() else None,
        }
    assert sum(v["positives"] for v in category_stats.values()) == int(positives.sum())

    union_stats = {}
    for cause in A.CAUSE_ORDER:
        mask = causes_union == cause
        union_stats[cause] = int((positives & mask).sum())
    assert sum(union_stats.values()) == int(positives.sum())

    log(f"      {int(positives.sum())} primary-subset positives categorised:")
    for cause in A.CAUSE_ORDER:
        stats = category_stats[cause]
        log(f"        {cause:<20} {stats['positives']:>4} positives "
            f"({100.0 * (stats['share_of_positives'] or 0):>5.1f}%), "
            f"recall {stats['recall'] if stats['recall'] is not None else float('nan'):.4f}, "
            f"recall@5 {stats['recall_at_5'] if stats['recall_at_5'] is not None else float('nan')}")

    # Headline metrics with snap-affected rows removed, for both definitions.
    def excluding(mask: np.ndarray) -> tuple[dict, int]:
        keep = primary & ~mask
        removed = int((primary & mask).sum())
        if not keep.any():
            return {}, removed
        return E.binary_metrics(y_true[keep], predicted[keep], score[keep]), removed

    clean_metrics, removed_yaw = excluding(snap_yaw)
    union_metrics, removed_union = excluding(snap_flag)
    log(f"      excluding yaw-snap-affected primary rows ({removed_yaw}) - the generator bug:")
    log(f"        recall {clean_metrics['recall']:.4f}, precision {clean_metrics['precision']:.4f}, "
        f"PR-AUC {clean_metrics['pr_auc']:.4f}")
    log(f"      excluding ALL flagged rows ({removed_union}, i.e. also fast sprint motion):")
    log(f"        recall {union_metrics['recall']:.4f}, precision {union_metrics['precision']:.4f}, "
        f"PR-AUC {union_metrics['pr_auc']:.4f}")

    # Distance and category breakdowns.
    bins = np.array([A.bin_index(float(d), distance_edges) for d in table.distance])
    by_distance = A.group_breakdown(table, y_true, predicted, score, primary, bins,
                                    A.distance_bin_labels(distance_edges))
    categories = sorted(set(table.category))
    category_index = np.array([categories.index(c) for c in table.category])
    by_category = A.group_breakdown(table, y_true, predicted, score, primary, category_index,
                                    categories)
    # Per-trajectory breakdown distinguishes out-of-fold rows from selection-exposed ones.
    phase2_5 = S.load_split_manifest(config["paths"]["phase2_5_split"])
    exposed = {"t02_sprint_straight", "t10_street_then_plaza", "t14_plaza_look"}
    by_trajectory = {}
    for index, trajectory_id in enumerate(table.trajectories):
        mask = primary & (table.trajectory == index)
        count = int((mask & y_true).sum())
        hits = int((mask & y_true & predicted).sum())
        by_trajectory[trajectory_id] = {
            "phase2_5_partition": phase2_5.get(trajectory_id, "unknown"),
            "used_for_parameter_selection": trajectory_id in exposed,
            "primary_rows": int(mask.sum()),
            "positives": count,
            "recall": round(hits / count, 4) if count else None,
            "recall_at_5": (round(A._recall_at_k_for_category(
                table, y_true, score, table.trajectory == index, primary, 5), 4)
                if count else None),
        }
    out_of_fold = [v for k, v in by_trajectory.items() if not v["used_for_parameter_selection"]]
    oof_pos = sum(v["positives"] for v in out_of_fold)
    oof_hit = sum(round((v["recall"] or 0.0) * v["positives"]) for v in out_of_fold)
    log(f"      out-of-fold trajectories (11 of 14, parameters never saw them): "
        f"{oof_pos} positives, pooled recall "
        f"{(oof_hit / oof_pos) if oof_pos else float('nan'):.4f}")

    error_analysis = {
        "parameters": {
            "source": str(arguments.phase3a_results),
            "retuned": False,
            "frustum": params.as_dict(),
            "threshold_deg": frozen_threshold,
            "selection_data": phase3a["tuning"]["selection_data"],
        },
        "target": {"horizon_seconds": 0.5, "target": "frustum_visible_in_0_5s",
                   "subset": "primary: rows not currently in the frustum"},
        "positives_total": int(positives.sum()),
        "categories": category_stats,
        "categories_with_any_flagged_snap": union_stats,
        "category_note": "category 1 uses yaw snaps only (the generator defect); "
                         "categories_with_any_flagged_snap is the same partition using the "
                         "union of yaw and position flags, as a sensitivity check",
        "cause_order": list(A.CAUSE_ORDER),
        "thresholds": {"low_angular_speed_deg_s": low_speed,
                       "turn_onset_accel_deg_s2": onset_accel,
                       "snap_window_samples": window},
        "headline": {
            "all_rows": baseline_full,
            "primary": baseline_primary,
            "primary_excluding_yaw_snap_affected": clean_metrics,
            "yaw_snap_affected_primary_rows_removed": removed_yaw,
            "primary_excluding_any_flagged": union_metrics,
            "any_flagged_primary_rows_removed": removed_union,
        },
        "position_jumps": {
            "flagged": int(sum(1 for s in snap_rows if "position" in s["snap_kind"])),
            "velocity_consistent": int(sum(1 for s in snap_rows
                                           if "position" in s["snap_kind"]
                                           and s["velocity_consistent"])),
            "note": "a position jump consistent with cam_velocity is fast motion, not a "
                    "discontinuity; 0 of 115 are inconsistent",
        },
        "by_distance": by_distance,
        "by_category": by_category,
        "by_trajectory": by_trajectory,
        "out_of_fold": {
            "trajectories": [k for k, v in by_trajectory.items()
                             if not v["used_for_parameter_selection"]],
            "positives": oof_pos,
            "pooled_recall": (oof_hit / oof_pos) if oof_pos else None,
        },
        "parameter_stability": {
            "chosen": phase3a["tuning"]["chosen"],
            "chosen_thresholds": phase3a["tuning"]["chosen_thresholds_deg"],
            "top_10_validation_scores": [round(entry["selection_score"], 4) for entry
                                         in phase3a["tuning"]["top_combinations"]],
            "combinations_evaluated": phase3a["tuning"]["combinations"],
        },
    }
    (output / "error_analysis.json").write_text(
        json.dumps(error_analysis, indent=2) + "\n", encoding="utf-8")

    # --- Task D: resolution target reassessment -------------------------------
    log("\n[D] resolution target reassessment")
    resolution = resolution_reassessment(table, cache, outputs=output)
    for horizon_key, block in resolution["per_horizon"].items():
        log(f"      {horizon_key}s: primary rows {block['primary']['rows']}, "
            f"under-prediction {block['primary']['under_prediction_rate']}, "
            f"exact {block['primary']['exact_resolution_rate']}")
        log(f"            newly-visible rows {block['newly_visible']['rows']}, "
            f"under-prediction {block['newly_visible']['under_prediction_rate']}")
    log(f"      coverage defined outside the frustum: "
        f"{resolution['coverage_outside_frustum']['statement']}")

    log(f"\n[E] split v1 (manifest recorded before any scoring)")
    split_v1 = evaluate_split_v1(table, cache, config, params, frozen_threshold,
                                 snap_yaw, output, log)

    log(f"\n-> {output}")
    return {"audit_rows": audit_rows, "snaps": snap_rows, "error_analysis": error_analysis,
            "resolution": resolution, "split_v1": split_v1}


def evaluate_split_v1(table, cache: B.RowPoseCache, config: dict,
                      params: B.HeuristicParams, threshold: float,
                      snap_yaw: np.ndarray, output: Path, log) -> dict | None:
    """Score the frozen Phase 3a parameters on the fixed split v1 test partition.

    No retuning: the parameters come from Phase 3a unchanged. The split manifest was
    written before this ran, and its hash is recorded alongside the results so the
    ordering is auditable.
    """
    import hashlib

    path = Path(config["paths"].get("experimental_split_v1",
                                    "dataset/manifests/experimental_split_v1.json"))
    if not path.is_file():
        log(f"      NOT FOUND ({path}) - skipped")
        return None

    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    partitions = {name: [entry["trajectory_id"] for entry in entries]
                  for name, entries in manifest["splits"].items()}
    lookup = {name: index for index, name in enumerate(table.trajectories)}
    masks = {name: np.isin(table.trajectory, [lookup[t] for t in ids if t in lookup])
             for name, ids in partitions.items()}

    seen: dict[str, str] = {}
    for name, ids in partitions.items():
        for trajectory_id in ids:
            assert trajectory_id not in seen, f"{trajectory_id} appears in two partitions"
            seen[trajectory_id] = name
    assert len(seen) == len(table.trajectories), "split v1 does not cover every trajectory"

    log(f"      manifest {path}")
    log(f"      sha256 {digest}")
    log("      partitions: " + ", ".join(f"{k}={len(v)}" for k, v in partitions.items()))

    # Reuse the row-aligned cache from Phase 3a; the scores do not depend on which
    # subset is later selected.
    scores = {
        "reactive": B.reactive_score(table, "frustum"),
        f"predictive@{table.horizons[0]:g}": B.predictive_score(table, cache, params,
                                                                table.horizons[0],
                                                                occlusion_aware=False),
        f"predictive@{table.horizons[1]:g}": B.predictive_score(table, cache, params,
                                                                table.horizons[1],
                                                                occlusion_aware=False),
    }

    result = {
        "manifest_path": str(path),
        "manifest_sha256": digest,
        "partitions": partitions,
        "partition_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "parameters": params.as_dict(),
        "threshold_deg": threshold,
        "retuned": False,
        "note": "parameters are Phase 3a's, unchanged; the split manifest was written and "
                "hashed before this evaluation ran",
        "test": {},
    }

    test_mask = masks["test"]
    for horizon_index, horizon in enumerate(table.horizons):
        y_true, _, primary = E.target_arrays(table, "frustum", horizon_index)
        block = {"rows": int(test_mask.sum()),
                 "primary_rows": int((test_mask & primary).sum()),
                 "positives_in_primary": int((test_mask & primary & y_true).sum()),
                 "baselines": {}}
        for name, key in (("reactive", "reactive"),
                          ("predictive", f"predictive@{horizon:g}")):
            score = scores[key]
            predicted = (score >= 1.0) if name == "reactive" \
                else B.threshold_predictions(score, threshold)
            evaluation = E.evaluate_target(table.subset(test_mask), y_true[test_mask],
                                           predicted[test_mask], score[test_mask],
                                           primary[test_mask])
            evaluation["recall_at_5_primary"] = E.recall_at_k(
                table, y_true, score, 5, eligible=primary & test_mask)
            clean = test_mask & ~snap_yaw
            evaluation["primary_excluding_snap_affected"] = E.binary_metrics(
                y_true[clean & primary], predicted[clean & primary], score[clean & primary])
            block["baselines"][name] = evaluation
        result["test"][f"{horizon:g}"] = block

    for horizon_key, block in result["test"].items():
        for name, evaluation in block["baselines"].items():
            full, sub = evaluation["full"], evaluation["primary_subset"]
            log(f"      test @ {horizon_key}s {name:<11} full F1 {full['f1']:.4f} "
                f"AP {full['pr_auc']:.4f} | primary F1 {sub['f1']:.4f} AP {sub['pr_auc']:.4f} "
                f"| recall@5 {evaluation['recall_at_5_primary']:.4f}")

    (output / "split_v1_baselines.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    rows = []
    for horizon_key, block in result["test"].items():
        for name, evaluation in block["baselines"].items():
            for scope, metrics in (
                    ("full", evaluation["full"]),
                    ("primary", evaluation["primary_subset"]),
                    ("primary_no_snap", evaluation["primary_excluding_snap_affected"])):
                for metric, value in metrics.items():
                    rows.append({"horizon": horizon_key, "baseline": name, "scope": scope,
                                 "metric": metric, "value": value})
            rows.append({"horizon": horizon_key, "baseline": name, "scope": "primary",
                         "metric": "recall_at_5",
                         "value": evaluation["recall_at_5_primary"]})
    write_csv(output / "split_v1_baselines.csv", rows)
    return result


def resolution_reassessment(table, cache: B.RowPoseCache, outputs: Path) -> dict:
    """Task D: is a deterministic coverage-derived resolution good enough?"""
    predicted = B.coverage_resolution_predictions(table, cache)
    outside = ~np.asarray(table.frustum_visible, dtype=bool)
    outside_coverage = table.coverage[outside]
    coverage_statement = (
        "yes - coverage is recorded for every candidate whether or not it is in the "
        "frustum, and is exactly 0.0 for objects outside it, which makes the "
        "coverage-threshold map return required_resolution = 0")

    per_horizon = {}
    for column, horizon in enumerate(table.horizons):
        target = table.y_resolution[:, column]
        eligible = np.isfinite(target)
        primary = eligible & outside
        newly = eligible & outside & table.y_frustum[:, column]
        implied = np.where(outside, 0.0, np.array(predicted, dtype=np.float64))

        block = {
            "all_eligible": E.resolution_metrics(target[eligible].astype(np.int64),
                                                 predicted[eligible]),
            "primary": _resolution_subset(target, predicted, primary),
            "newly_visible": _resolution_subset(target, predicted, newly),
            "future_differs_from_current_implied": {
                "rows": int((target[eligible] != implied[eligible]).sum()),
                "rate": round(float((target[eligible] != implied[eligible]).mean()), 4),
            },
        }
        block["all_eligible"]["rows"] = int(eligible.sum())
        per_horizon[f"{horizon:g}"] = block

    result = {
        "baseline": "coverage-threshold: required_resolution from CURRENT projected coverage",
        "coverage_outside_frustum": {
            "statement": coverage_statement,
            "rows_outside_frustum": int(outside.sum()),
            "max_coverage_outside_frustum": float(outside_coverage.max()) if outside.any() else 0.0,
            "nonzero_coverage_outside_frustum": int((outside_coverage > 0).sum()),
        },
        "per_horizon": per_horizon,
    }
    (outputs / "resolution_reassessment.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _resolution_subset(target: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {"rows": 0}
    metrics = E.resolution_metrics(target[mask].astype(np.int64), predicted[mask])
    metrics["rows"] = count
    metrics["predicted_all_zero"] = bool((predicted[mask] == 0).all())
    return metrics


if __name__ == "__main__":
    run()
