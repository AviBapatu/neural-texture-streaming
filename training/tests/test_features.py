"""Feature correctness, causality of the recorded data, and the mask/normaliser rules."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import baselines as B
from training import dataset as D
from training import features as F


# --- the camera frame and the resolution mapping ------------------------------

def test_quaternion_matrix_matches_recorded_view_direction(fixture_dir):
    """The world-from-local convention must reproduce the recorded view direction."""
    samples = D.load_labels(fixture_dir / "t06_look_around.samples.jsonl")
    worst = 0.0
    for sample in samples:
        matrix = F.quaternion_to_matrix(sample["cam_quaternion"])
        # Godot cameras look down local -Z.
        worst = max(worst, float(np.max(np.abs(-matrix[:, 2] - np.array(sample["cam_direction"])))))
    assert worst < 1e-3, f"camera frame convention is wrong (max error {worst})"


def test_resolution_port_matches_recorded(fixture_dir):
    """The Python port of the labeller's coverage->resolution map must be exact.

    The canonical implementation is GDScript and cannot be imported, so equivalence
    is established against every visible row of the fixture rather than assumed.
    """
    samples = D.load_labels(fixture_dir / "t06_look_around.samples.jsonl")
    labels = D.load_labels(fixture_dir / "t06_look_around.labels.jsonl")
    width, height = samples[0]["viewport"]
    checked = mismatches = 0
    for row in labels:
        if not row["frustum_visible"]:
            continue
        checked += 1
        predicted = F.resolution_from_coverage(row["coverage"], width, height)
        if predicted != row["required_resolution"]:
            mismatches += 1
    assert checked > 1000, "not enough visible rows to make the check meaningful"
    assert mismatches == 0, f"{mismatches}/{checked} rows disagree with the recorder"


def test_resolution_port_boundaries():
    """Exact ladder boundaries, including the zero-coverage case."""
    assert F.resolution_from_coverage(0.0, 1280, 720) == 0
    assert F.resolution_from_coverage(-1.0, 1280, 720) == 0
    assert F.resolution_from_coverage(0.5, 0, 720) == 0
    # sqrt(coverage * 1280 * 720) just under and just over a ladder step
    for step in F.RESOLUTION_LADDER:
        below = ((step - 1e-6) ** 2) / (1280 * 720)
        assert F.resolution_from_coverage(below, 1280, 720) == step


def test_horizon_suffix_matches_the_labeller():
    """The target column names are built here and read from the labeller's output."""
    tools = ROOT / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from label_future_demand import horizon_suffix as canonical
    for horizon in (0.5, 1.0, 2.0):
        assert D.horizon_suffix(horizon) == canonical(horizon)
    assert D.horizon_suffix(0.5) == "0_5"
    assert D.horizon_suffix(1.0) == "1_0"


# --- causality of derived signals ---------------------------------------------

def test_bearing_rate_does_not_spike_at_the_azimuth_wrap(fixture_dir):
    """Azimuth is reported in [-180, 180]; an unwrapped difference jumps by 360.

    A 3600 deg/s bearing rate would mean the object teleported around the camera.
    """
    table = _fixture_table(fixture_dir)
    rates = table.X[:, table.names.index("bearing_rate_deg_s")]
    # 180 / 0.1 s = 1800 deg/s is the largest physically meaningful rate for a
    # sample-to-sample change, and only for an object passing the camera.
    assert np.abs(rates).max() <= 1800.0 + 1e-6


def test_angular_velocity_correction_handles_quaternion_double_cover():
    """A near-360-degree quaternion delta must not be read as a fast rotation."""
    # Two orientations 18 deg/s apart, but the second stored as the -q
    # representation, which is the same rotation.
    angle = math.radians(1.8)          # 18 deg/s at 0.1 s
    q_prev = np.array([[0.0, 0.0, 0.0, 1.0]])
    q_next = np.array([[0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)]])
    times = np.array([0.0, 0.1])

    velocity, flagged = F.angular_velocity_from_quaternions(
        np.vstack([q_prev, q_next]), times)
    assert not flagged[1]
    assert abs(np.degrees(np.linalg.norm(velocity[1])) - 18.0) < 0.1

    flipped = np.vstack([q_prev, -q_next])
    velocity_flipped, flagged_flipped = F.angular_velocity_from_quaternions(flipped, times)
    assert flagged_flipped[1], "double cover was not detected"
    assert abs(np.degrees(np.linalg.norm(velocity_flipped[1])) - 18.0) < 0.1, \
        "the shorter arc must be chosen"


def test_recorded_angular_velocity_defect_is_bounded(fixture_dir):
    """The recorded field disagrees with the corrected one only on a few samples.

    This does not assert the defect is absent - it is present in the recordings and
    cannot be fixed without touching the recorder. It asserts the *extent* is small
    enough that recomputing the value is a correction, not a rewrite.
    """
    poses = B.load_pose_arrays(fixture_dir)
    total = flagged = 0
    for trajectory_id in poses.quaternion:
        rows = D.load_labels(fixture_dir / f"{trajectory_id}.samples.jsonl")
        recorded = np.array([row["cam_angular_velocity"] for row in rows])
        corrected = poses.angular_velocity[trajectory_id]
        total += len(rows) - 1
        flagged += int((np.linalg.norm(recorded[1:] - corrected[1:], axis=1) > 0.5).sum())
    assert total > 0
    assert flagged > 0, "the double-cover defect was not found in the fixture"
    assert flagged / total < 0.05, f"{flagged}/{total} samples disagree"


# --- masks and normalisation --------------------------------------------------

def test_history_is_masked_at_the_start_of_a_trajectory(fixture_dir, config):
    """The first K samples have no history; they are flagged, not silently zeroed."""
    sample_features = F.compute_sample_features(
        D.load_labels(fixture_dir / "t06_look_around.samples.jsonl"), history_k=4)
    for k in range(1, 5):
        valid = sample_features.history[f"hist_valid_{k}"]
        assert np.all(valid[:k] == 0.0), f"hist_valid_{k} claims history before t={k}"
        assert np.all(valid[k:] == 1.0), f"hist_valid_{k} missing valid samples"


def test_normaliser_depends_only_on_the_rows_it_is_given(fixture_dir, tiny_config):
    """Statistics must be fit on training rows only, and change with that set."""
    table = _fixture_table(fixture_dir, tiny_config)
    train = np.arange(len(table)) % 3 != 0
    other = ~train

    first = D.fit_normaliser(table, train)
    second = D.fit_normaliser(table, train)
    assert np.allclose(first.mean, second.mean)
    assert first.fit_rows == int(train.sum())

    different = D.fit_normaliser(table, other)
    assert not np.allclose(first.mean, different.mean), \
        "statistics did not change when the fitting rows changed"

    # Corrupting rows outside the fitting mask must not move the statistics.
    corrupted = _fixture_table(fixture_dir, tiny_config, copy=True)
    corrupted.X[other] *= 100.0
    third = D.fit_normaliser(corrupted, train)
    assert np.allclose(first.mean, third.mean), "non-training rows influenced the statistics"
    assert np.allclose(first.std, third.std)


def test_normaliser_excludes_invalid_masked_values(fixture_dir, tiny_config):
    """A masked history slot must not contribute a spurious zero to the mean."""
    table = _fixture_table(fixture_dir, tiny_config)
    train = np.ones(len(table), dtype=bool)
    normaliser = D.fit_normaliser(table, train)
    column = table.names.index("hist_dpos_1")
    validity = table.names.index("hist_valid_1")
    keep = table.X[:, validity] > 0.5
    assert keep.sum() < len(table), "no invalid rows in the fixture; test is vacuous"
    assert abs(normaliser.mean[column] - table.X[keep, column].mean()) < 1e-5


def test_normaliser_json_round_trip(tmp_path, fixture_dir, tiny_config):
    table = _fixture_table(fixture_dir, tiny_config)
    normaliser = D.fit_normaliser(table, np.ones(len(table), dtype=bool))
    path = tmp_path / "normaliser.json"
    D.save_normaliser(normaliser, path)
    restored = D.load_normaliser(path)
    assert restored.names == normaliser.names
    assert np.allclose(restored.mean, normaliser.mean)
    assert np.allclose(restored.std, normaliser.std)


def test_unknown_occlusion_and_ineligible_rows_are_masked(fixture_dir, tiny_config):
    """Unknown occlusion is excluded from the target, never counted as negative."""
    table = _fixture_table(fixture_dir, tiny_config)
    from training import evaluate as E

    for column in range(len(table.horizons)):
        y_true, valid, _ = E.target_arrays(table, "occlusion", column)
        unknown = table.y_occlusion[:, column] == D.OCCLUSION_UNKNOWN
        assert np.array_equal(~valid, unknown), \
            "the occlusion validity mask does not match the unknown rows"
        assert not np.any(y_true[unknown]), "an unknown row was treated as a negative label"

    # Resolution targets exist only for eligible rows.
    ineligible = ~table.resolution_eligible
    assert np.all(np.isnan(table.y_resolution[ineligible])), \
        "an ineligible row carries a resolution target"


def test_resolution_targets_are_on_the_ladder(fixture_dir, tiny_config):
    table = _fixture_table(fixture_dir, tiny_config)
    values = table.y_resolution[np.isfinite(table.y_resolution)]
    assert values.size > 0
    assert set(np.unique(values).astype(int)) <= set(F.RESOLUTION_LADDER)


# --- assembly -----------------------------------------------------------------

def test_feature_matrix_is_finite_and_uses_the_allow_list(fixture_dir, config):
    table = _fixture_table(fixture_dir)
    assert np.isfinite(table.X).all(), "feature matrix contains NaN or infinity"
    expected = list(config["features"]["allow"])
    assert table.names[:len(expected)] == expected, "matrix columns do not follow the allow-list"


def test_assemble_matrix_rejects_unknown_feature_names():
    rows = [{"a": 1.0}]
    with pytest.raises(KeyError):
        F.assemble_matrix(rows, ["a", "typo_feature"])


def _fixture_table(fixture_dir, config_override=None, copy=False):
    import copy as copy_module
    config = config_override if config_override is not None else _default_config()
    if copy:
        config = copy_module.deepcopy(config)
    return D.build_table(fixture_dir, config)


def _default_config():
    import yaml
    return yaml.safe_load(
        (ROOT / "training" / "configs" / "features_v1.yaml").read_text(encoding="utf-8"))
