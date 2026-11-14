"""Phase 4a step 1: save out-of-fold prediction scores for the simulator.

The simulator needs a prediction for *every* candidate row, not just the primary
subset, and it needs the guarantee that a row's score comes from a model that never
saw that row's trajectory. So the existing leave-trajectory-out folds are re-run
once, seed 1, for the two model variants worth simulating:

* ``b`` - the exported MLP (occlusion-aware, portable to GDScript)
* ``c`` - HistGradientBoosting (the offline reference, not portable)

and the two frozen heuristic scores are recomputed alongside. Nothing is retrained
or retuned beyond these; the parameters are read from ``model_v1.yaml`` and the
Phase 3a results exactly as Phase 3b used them.

Output: ``oof_scores.npz``, one row per feature-table row, aligned with the table's
row order, plus the key columns and the fold assignment so the disjointness
property can be checked without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Usable as both `python -m training.save_oof_scores` and
# `python training/save_oof_scores.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import baselines as B
from training import dataset as D
from training import evaluate as E
from training import features as F
from training import splits as S
from training import train as T
from training.run_3b import (frozen_heuristic, output_specs, snap_yaw_row_mask,
                             trajectory_rows)

DEFAULT_OUTPUT = "training/experiments/phase4a"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="save Phase 4a out-of-fold scores")
    parser.add_argument("--config", default="training/configs/model_v1.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=1,
                        help="the single seed the simulator scores come from")
    return parser.parse_args(argv)


def mlp_fold_scores(table, specs, fold, variant_config, config, seed, heuristic_score,
                    snap_yaw) -> np.ndarray:
    """Train one fold and predict **every** row.

    Rows outside the test trajectories are predicted too, but the caller only ever
    reads the test trajectories, so the out-of-fold property holds.
    """
    train_rows = trajectory_rows(table, fold.train) & T.unified_primary(table)
    validation_rows = trajectory_rows(table, fold.validation) & T.unified_primary(table)
    extra = heuristic_score.reshape(-1, 1) if variant_config.get("heuristic_input") else None
    bundle = T.build_bundle(
        table, specs, train_rows,
        drop_snap_yaw=config["masking"]["drop_snap_yaw_from_training"],
        snap_yaw_rows=snap_yaw, extra_inputs=extra)
    fit = T.train_mlp(bundle, specs, validation_rows, table, config, seed,
                      exclude_snap=snap_yaw)
    return T.predict_mlp(fit.model, bundle)


def sklearn_fold_scores(table, specs, fold, config, seed, settings) -> np.ndarray:
    """One HistGradientBoosting model per output, predicted for every row."""
    train_rows = trajectory_rows(table, fold.train) & T.unified_primary(table)
    y, _ = T.per_output_targets(table, specs)
    X = table.X.astype(np.float64)
    scores = np.zeros((len(table), len(specs)), dtype=np.float64)
    for column, spec in enumerate(specs):
        _, valid, primary = E.target_arrays(table, spec.target, spec.horizon_index)
        model = T.train_sklearn(X, y[:, column], train_rows & valid & primary, seed, settings)
        scores[:, column] = T.predict_sklearn(model, X)
    return scores


def assemble_oof(table, folds, per_fold: list[np.ndarray], label: str) -> np.ndarray:
    """Fill each trajectory's rows with the scores from the fold that held it out."""
    count, width = len(table), per_fold[0].shape[1]
    oof = np.full((count, width), np.nan, dtype=np.float64)
    for fold, scores in zip(folds, per_fold):
        rows = trajectory_rows(table, fold.test)
        oof[rows] = scores[rows]
    missing = np.isnan(oof)
    if missing.any():
        raise RuntimeError(
            f"{label}: {int(missing.any(axis=1).sum())} rows left unscored; the folds "
            f"must cover every trajectory exactly once as test")
    return oof


def main(argv: list[str] | None = None) -> dict:
    arguments = parse_args(argv)
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    base_feature_config = yaml.safe_load(
        Path(config["feature_config"]).read_text(encoding="utf-8"))
    output = Path(arguments.output or DEFAULT_OUTPUT)
    output.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    import torch
    torch.set_num_threads(int(config["training"].get("threads", 1)))

    dataset = Path(config["dataset"])
    specs = output_specs(config)

    log("=" * 78)
    log("Phase 4a step 1 - out-of-fold scores for the simulator")
    log("=" * 78)

    table = D.build_table(dataset, base_feature_config)
    snap_yaw = snap_yaw_row_mask(table, Path(config["sidecar"]["eval_flags"]))
    log(f"\n  {len(table)} rows x {len(table.names)} features, "
        f"{len(table.trajectories)} trajectories")
    log(f"  snap-yaw rows: {int(snap_yaw.sum())}")

    tuned, _ = frozen_heuristic(config)
    catalogue = F.ObjectCatalogue.from_manifest(base_feature_config["paths"]["object_index"])
    cache = B.build_row_pose_cache(table, B.load_pose_arrays(dataset), catalogue.centre)
    heuristic = {
        "heuristic_frustum": B.predictive_score(table, cache, tuned["frustum"][0], 0.5,
                                                occlusion_aware=False),
        "heuristic_occlusion": B.predictive_score(table, cache, tuned["occlusion"][0], 0.5,
                                                  occlusion_aware=True),
    }
    log(f"  frozen heuristic: frustum {tuned['frustum'][0].as_dict()}")
    log(f"                    occlusion {tuned['occlusion'][0].as_dict()}")

    fold_settings = base_feature_config["splits"]
    folds = S.build_folds(table.trajectories, folds=int(fold_settings["folds"]),
                          fold_size=int(fold_settings["fold_size"]),
                          seed=int(fold_settings["group_seed"]))
    log(f"\n  {len(folds)} leave-trajectory-out folds: " +
        "; ".join(f"{f.index}:test={','.join(f.test)}" for f in folds))

    variant_b = config["variants"]["b"]
    variant_c = config["variants"]["c"]

    log(f"\n[1/3] variant b (MLP, seed {arguments.seed})")
    b_scores = []
    for fold in folds:
        scores = mlp_fold_scores(table, specs, fold, variant_b, config, arguments.seed,
                                 heuristic["heuristic_frustum"], snap_yaw)
        b_scores.append(scores)
        log(f"      fold {fold.index} test={','.join(fold.test)} done")
    mlp_b = assemble_oof(table, folds, b_scores, "mlp_b")

    log("\n[2/3] variant c (HistGradientBoosting)")
    c_scores = []
    for fold in folds:
        scores = sklearn_fold_scores(table, specs, fold, config, arguments.seed,
                                     variant_c.get("sklearn", {}))
        c_scores.append(scores)
        log(f"      fold {fold.index} test={','.join(fold.test)} done")
    gbdt_c = assemble_oof(table, folds, c_scores, "gbdt_c")

    log("\n[3/3] writing")
    payload = {
        "trajectory": np.array(table.trajectories, dtype=object)[table.trajectory],
        "sample_index": table.sample_index.astype(np.int32),
        "object_id": np.array(table.object_id, dtype=object),
        "frustum_visible": table.frustum_visible.astype(np.int8),
        "occlusion_state": table.occlusion_state.astype(np.int8),
        "coverage": table.coverage.astype(np.float32),
        "snap_affected_yaw": snap_yaw.astype(np.int8),
        "heuristic_frustum": heuristic["heuristic_frustum"].astype(np.float32),
        "heuristic_occlusion": heuristic["heuristic_occlusion"].astype(np.float32),
    }
    for column, spec in enumerate(specs):
        payload[f"mlp_b__{spec.name}"] = mlp_b[:, column].astype(np.float32)
        payload[f"gbdt_c__{spec.name}"] = gbdt_c[:, column].astype(np.float32)

    np.savez_compressed(output / "oof_scores.npz", **payload)

    manifest = {
        "rows": int(len(table)),
        "trajectories": list(table.trajectories),
        "outputs": [spec.name for spec in specs],
        "seed": int(arguments.seed),
        "variants": {
            "mlp_b": config["variants"]["b"]["label"],
            "gbdt_c": config["variants"]["c"]["label"],
        },
        "folds": [fold.as_dict() for fold in folds],
        "fold_seed": int(fold_settings["group_seed"]),
        "key_columns": ["trajectory", "sample_index", "object_id"],
        "heuristic": {target: {"parameters": params.as_dict(), "threshold_deg": threshold}
                      for target, (params, threshold) in tuned.items()},
        "config": arguments.config,
        "note": "each row's model score comes from the fold that held its trajectory "
                "out; the fold assignment is recorded here so that is checkable",
    }
    (output / "oof_scores.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    log(f"\n  -> {output / 'oof_scores.npz'} "
        f"({(output / 'oof_scores.npz').stat().st_size / 1e6:.1f} MB)")
    log(f"  -> {output / 'oof_scores.manifest.json'}")
    return manifest


if __name__ == "__main__":
    main()
