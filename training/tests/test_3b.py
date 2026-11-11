"""Phase 3b tests: normalisation scope, masking, determinism, export round-trip.

These use the three-trajectory fixture and a deliberately tiny model so the suite
stays fast. Anything that would be slow is asserted structurally instead.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import dataset as D
from training import models as M
from training import run_3b as R
from training import train as T

pytest.importorskip("torch")


def _model_config(base_features: Path) -> dict:
    config = yaml.safe_load(
        (ROOT / "training" / "configs" / "model_v1.yaml").read_text(encoding="utf-8"))
    config["feature_config"] = str(base_features)
    config["training"] = {**config["training"], "seeds": [1], "max_epochs": 4,
                          "batch_size": 4096, "early_stopping_patience": 2}
    config["model"] = {"hidden": [8, 4], "dropout": 0.0, "activation": "relu"}
    return config


@pytest.fixture(scope="module")
def features_config():
    return yaml.safe_load(
        (ROOT / "training" / "configs" / "features_v1.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def table_and_snap(fixture_dir, features_config):
    table = D.build_table(fixture_dir, features_config)
    # The fixture trajectories t01/t06/t12 carry no yaw snaps, so synthesise the flag
    # the way the sidecar would: a window of rows around a chosen sample.
    snap = np.zeros(len(table), dtype=bool)
    snap[(table.trajectory == 1) & (np.abs(table.sample_index - 40) <= 10)] = True
    return table, snap


@pytest.fixture(scope="module")
def specs():
    return [T.OutputSpec(name="frustum_0_5", target="frustum", horizon=0.5),
            T.OutputSpec(name="frustum_1_0", target="frustum", horizon=1.0),
            T.OutputSpec(name="occlusion_0_5", target="occlusion", horizon=0.5),
            T.OutputSpec(name="occlusion_1_0", target="occlusion", horizon=1.0)]


# --- normalisation ------------------------------------------------------------

def test_normaliser_is_fitted_on_training_rows_only(table_and_snap, specs):
    table, snap = table_and_snap
    train_rows = T.unified_primary(table) & (table.trajectory == 0)

    bundle = T.build_bundle(table, specs, train_rows, drop_snap_yaw=True,
                            snap_yaw_rows=snap)
    mask = train_rows & ~snap
    expected_rows = int(mask.sum())
    assert bundle.normaliser.fit_rows == expected_rows, (
        f"normaliser saw {bundle.normaliser.fit_rows} rows, expected {expected_rows}")

    # Corrupting rows outside the fitting mask must not move the statistics. The
    # statistics are in raw space, so the comparison is against a raw-space refit.
    reference = D.fit_normaliser(table, mask)
    raw = table.X.copy()
    raw[~mask] = 1e6
    refit = D.fit_normaliser(T._table_with_raw(table, raw, table.names), mask)
    assert np.allclose(refit.mean, reference.mean), "non-training rows influenced the mean"
    assert np.allclose(refit.std, reference.std), "non-training rows influenced the std"
    assert np.allclose(refit.mean, bundle.normaliser.mean)


def test_normaliser_does_not_use_validation_or_test_rows(table_and_snap, specs):
    """Changing a validation trajectory's features must not change the statistics."""
    table, snap = table_and_snap
    train_rows = T.unified_primary(table) & (table.trajectory == 0)
    first = T.build_bundle(table, specs, train_rows, drop_snap_yaw=True, snap_yaw_rows=snap)

    altered = copy.deepcopy(table)
    held_out = table.trajectory == 1
    altered.X[held_out] *= 7.0
    second = T.build_bundle(altered, specs, train_rows, drop_snap_yaw=True, snap_yaw_rows=snap)

    assert np.allclose(first.normaliser.mean, second.normaliser.mean)
    assert np.allclose(first.normaliser.std, second.normaliser.std)


# --- masking ------------------------------------------------------------------

def test_snap_yaw_rows_lose_all_loss_weight(table_and_snap, specs):
    table, snap = table_and_snap
    assert snap.sum() > 0, "fixture has no snap rows; the test would be vacuous"

    dropped = T.build_bundle(table, specs, T.unified_primary(table), drop_snap_yaw=True,
                             snap_yaw_rows=snap)
    kept = T.build_bundle(table, specs, T.unified_primary(table), drop_snap_yaw=False,
                          snap_yaw_rows=snap)

    assert not dropped.loss_mask[snap].any(), "a snap-yaw row still contributes to the loss"
    assert not dropped.row_mask[snap].any(), "a snap-yaw row is still marked trainable"
    assert kept.loss_mask[snap].any(), "the flag is being dropped even when disabled"


def test_unknown_occlusion_rows_are_masked_from_the_occlusion_outputs(table_and_snap, specs):
    table, snap = table_and_snap
    bundle = T.build_bundle(table, specs, T.unified_primary(table), drop_snap_yaw=True,
                            snap_yaw_rows=snap)

    unknown = table.y_occlusion == D.OCCLUSION_UNKNOWN
    assert unknown.any(), "fixture has no unknown occlusion labels"
    for column, spec in enumerate(specs):
        if spec.target != "occlusion":
            continue
        # Each occlusion output reads its own horizon's label.
        rows = unknown[:, spec.horizon_index] & bundle.row_mask
        assert rows.sum() > 0, f"{spec.name}: no unknown rows to check"
        assert not bundle.loss_mask[rows, column].any(), (
            f"{spec.name}: an unknown occlusion label carries loss weight")
        assert not bundle.y[rows, column].any(), (
            f"{spec.name}: an unknown occlusion label was encoded as a positive")


def test_occlusion_masking_does_not_touch_the_frustum_outputs(table_and_snap, specs):
    """Frustum labels are always defined, so their mask must be exactly the row mask."""
    table, snap = table_and_snap
    bundle = T.build_bundle(table, specs, T.unified_primary(table), drop_snap_yaw=True,
                            snap_yaw_rows=snap)
    for column, spec in enumerate(specs):
        if spec.target != "frustum":
            continue
        assert np.array_equal(bundle.loss_mask[:, column] > 0.5, bundle.row_mask)


def test_sidecar_flags_are_never_features(table_and_snap, specs):
    """The sidecar columns must not reach the feature matrix, at any history depth."""
    table, snap = table_and_snap
    bundle = T.build_bundle(table, specs, T.unified_primary(table), drop_snap_yaw=True,
                            snap_yaw_rows=snap)
    from training import audit as A
    for column in A.SIDECAR_COLUMNS:
        assert column not in bundle.feature_names, f"{column} leaked into the features"
        assert column not in table.names
    assert snap.dtype == bool and len(snap) == len(table)


def test_unified_primary_is_the_union_of_the_two_primary_subsets(table_and_snap):
    table, _ = table_and_snap
    unified = T.unified_primary(table)

    from training import evaluate as E
    _, _, frustum_primary = E.target_arrays(table, "frustum", 0)
    _, _, occlusion_primary = E.target_arrays(table, "occlusion", 0)

    assert np.array_equal(unified, frustum_primary | occlusion_primary), (
        "the shared training row set is not the union of the two primary subsets")


# --- determinism --------------------------------------------------------------

def test_training_is_deterministic_given_a_seed(table_and_snap, specs, features_config):
    table, snap = table_and_snap
    config = _model_config(Path(features_config and ROOT / "training/configs/features_v1.yaml"))
    config["seeds"] = [7]

    train_rows = T.unified_primary(table) & (table.trajectory == 0)
    validation_rows = T.unified_primary(table) & (table.trajectory == 1)

    first = T.build_bundle(table, specs, train_rows, True, snap)
    second = T.build_bundle(table, specs, train_rows, True, snap)

    fit_a = T.train_mlp(first, specs, validation_rows, table, config, seed=7,
                        exclude_snap=snap)
    fit_b = T.train_mlp(second, specs, validation_rows, table, config, seed=7,
                        exclude_snap=snap)

    assert fit_a.best_epoch == fit_b.best_epoch
    assert np.allclose(fit_a.best_validation_score, fit_b.best_validation_score)
    predictions_a = T.predict_mlp(fit_a.model, first)
    predictions_b = T.predict_mlp(fit_b.model, second)
    assert np.array_equal(predictions_a, predictions_b), \
        "two runs with the same seed produced different predictions"


def test_different_seeds_produce_different_models(table_and_snap, specs, features_config):
    table, snap = table_and_snap
    config = _model_config(ROOT / "training/configs/features_v1.yaml")
    train_rows = T.unified_primary(table) & (table.trajectory == 0)
    validation_rows = T.unified_primary(table) & (table.trajectory == 1)
    bundle = T.build_bundle(table, specs, train_rows, True, snap)

    one = T.train_mlp(bundle, specs, validation_rows, table, config, seed=1, exclude_snap=snap)
    two = T.train_mlp(bundle, specs, validation_rows, table, config, seed=2, exclude_snap=snap)
    assert not np.array_equal(T.predict_mlp(one.model, bundle),
                              T.predict_mlp(two.model, bundle))


# --- export -------------------------------------------------------------------

def test_export_round_trip_matches_the_torch_forward(table_and_snap, specs, tmp_path):
    """The plain JSON/NPZ bundle must be executable and reproduce the torch output."""
    table, snap = table_and_snap
    config = _model_config(ROOT / "training/configs/features_v1.yaml")
    train_rows = T.unified_primary(table) & (table.trajectory == 0)
    validation_rows = T.unified_primary(table) & (table.trajectory == 1)
    bundle = T.build_bundle(table, specs, train_rows, True, snap)
    fit = T.train_mlp(bundle, specs, validation_rows, table, config, seed=1, exclude_snap=snap)

    stem = tmp_path / "bundle"
    info = M.export_bundle(fit.model, bundle.feature_names, bundle.output_names,
                           bundle.normaliser, stem)
    assert Path(info["weights"]).is_file() and Path(info["header"]).is_file()

    header, arrays = M.load_bundle(stem)
    assert header["feature_names"] == bundle.feature_names
    assert header["output_names"] == bundle.output_names
    assert len(header["normaliser"]["mean"]) == len(bundle.feature_names)

    reference = M.forward_numpy(header, arrays, bundle.raw_X[:256])
    torch_output = T.predict_mlp(fit.model, bundle)[:256]
    # The reference applies the export's own normaliser to the raw features, and
    # sigmoid is applied outside; compare in logit space.
    logits_reference = reference
    logits_torch = np.log(np.clip(torch_output, 1e-12, 1 - 1e-12)
                          / np.clip(1 - torch_output, 1e-12, 1))
    assert np.allclose(logits_reference, logits_torch, atol=1e-4), \
        "the exported bundle does not reproduce the network"


def test_export_records_parameter_count(table_and_snap, specs, tmp_path):
    table, snap = table_and_snap
    config = _model_config(ROOT / "training/configs/features_v1.yaml")
    train_rows = T.unified_primary(table) & (table.trajectory == 0)
    validation_rows = T.unified_primary(table) & (table.trajectory == 1)
    bundle = T.build_bundle(table, specs, train_rows, True, snap)
    fit = T.train_mlp(bundle, specs, validation_rows, table, config, seed=1, exclude_snap=snap)

    expected = sum(p.numel() for p in fit.model.parameters() if p.requires_grad)
    assert M.count_parameters(fit.model) == expected
    # dense stack: (d*h1 + h1) + (h1*h2 + h2) + (h2*4 + 4)
    d, h1, h2, o = bundle.X.shape[1], 8, 4, 4
    assert expected == d * h1 + h1 + h1 * h2 + h2 + h2 * o + o


# --- configuration ------------------------------------------------------------

def test_history_ablation_config_removes_only_history_columns():
    base = yaml.safe_load(
        (ROOT / "training" / "configs" / "features_v1.yaml").read_text(encoding="utf-8"))
    adjusted = R.feature_config(base, 0)
    removed = set(base["features"]["allow"]) - set(adjusted["features"]["allow"])
    assert removed, "the K=0 ablation removed nothing"
    assert all(name.startswith("hist_") for name in removed), \
        f"the ablation removed non-history features: {sorted(removed)}"
    assert adjusted["history_k"] == 0
    assert base["features"]["allow"] == yaml.safe_load(
        (ROOT / "training" / "configs" / "features_v1.yaml").read_text(
            encoding="utf-8"))["features"]["allow"], "the base config was mutated"


def test_frozen_heuristic_parameters_are_read_per_target_not_retuned():
    """Both tuned parameter sets are read verbatim, and they are genuinely different.

    Phase 3a tuned a separate set per target. If the occlusion target were scored with
    the frustum set the baseline would be handicapped and the MLP's advantage inflated,
    so this asserts the two sets are distinct and each matches the stored values.
    """
    config = yaml.safe_load(
        (ROOT / "training" / "configs" / "model_v1.yaml").read_text(encoding="utf-8"))
    tuned, payload = R.frozen_heuristic(config)

    assert set(tuned) == {"frustum", "occlusion"}
    for target, (params, threshold) in tuned.items():
        stored = payload["tuning"]["chosen"][target]
        assert params.damping == stored["damping"]
        assert params.angular_damping == stored["angular_damping"]
        assert params.integrate_steps == stored["integrate_steps"]
        assert params.occlusion_penalty_deg == stored["occlusion_penalty_deg"]
        assert threshold == payload["tuning"]["chosen_thresholds_deg"][target]["0.5"]

    frustum_params, frustum_threshold = tuned["frustum"]
    occlusion_params, occlusion_threshold = tuned["occlusion"]
    assert (frustum_params.damping, frustum_params.integrate_steps,
            frustum_params.occlusion_penalty_deg) != (
        occlusion_params.damping, occlusion_params.integrate_steps,
        occlusion_params.occlusion_penalty_deg), \
        "the two targets resolved to the same parameters; the occlusion baseline would be wrong"
    assert frustum_threshold != occlusion_threshold


def test_occlusion_target_is_scored_occlusion_aware():
    """The heuristic reference must use the occlusion-aware score for that target."""
    import inspect
    source = inspect.getsource(R.heuristic_metrics)
    assert "scores_by_target[spec.target]" in source, \
        "heuristic_metrics does not select the score by target"
    run_source = inspect.getsource(R.run)
    assert 'occlusion_aware=True' in run_source, \
        "no occlusion-aware heuristic score is built"


def test_positive_weights_are_bounded(table_and_snap, specs):
    table, snap = table_and_snap
    bundle = T.build_bundle(table, specs, T.unified_primary(table), True, snap)
    weights = T.positive_weights(bundle, cap=20.0)
    assert np.all(weights >= 1.0 - 1e-9), "pos_weight fell below 1"
    assert np.all(weights <= 20.0 + 1e-9), "pos_weight exceeded the cap"
