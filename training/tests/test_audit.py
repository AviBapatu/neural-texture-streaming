"""Audit-specific tests: snap detection, split v1 integrity, category partition, sidecar safety."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import audit as A
from training import features as F

# The Phase 2.5 manifest must never change. Pinned here so any future edit to it
# fails the suite rather than passing silently.
PHASE2_5_MANIFEST = ROOT / "dataset" / "manifests" / "runs" / "run_full_p25" / "splits.json"
PHASE2_5_SHA256 = "60b271b9aa7fa6556378b7affc7a669db1593c32e24ff68763c54217a53603e0"
SPLIT_V1 = ROOT / "dataset" / "manifests" / "experimental_split_v1.json"


def _config():
    return yaml.safe_load(
        (ROOT / "training" / "configs" / "features_v1.yaml").read_text(encoding="utf-8"))


# --- synthetic sample builders ------------------------------------------------

def _sample(yaw_deg: float, position=(0.0, 0.0, 0.0), time: float = 0.0,
            velocity=(0.0, 0.0, 0.0), kind: str = "walk_straight") -> dict:
    """A minimal sample carrying only the fields the snap detector reads."""
    radians = np.radians(yaw_deg)
    direction = [-np.sin(radians), 0.0, -np.cos(radians)]
    return {
        "cam_direction": list(direction),
        "cam_position": list(position),
        "cam_velocity": list(velocity),
        "session_time": time,
        "trajectory_kind": kind,
    }


def _sequence(yaws, positions=None, times=None, velocity=(0.0, 0.0, 0.0)):
    count = len(yaws)
    positions = positions or [(0.0, 0.0, 0.0)] * count
    times = times or [i * 0.1 for i in range(count)]
    return [_sample(yaws[i], positions[i], times[i], velocity) for i in range(count)]


# --- shortest arc -------------------------------------------------------------

def test_shortest_arc_wraps_through_zero():
    assert A.shortest_arc_degrees(359.0, 1.0) == pytest.approx(2.0)
    assert A.shortest_arc_degrees(1.0, 359.0) == pytest.approx(-2.0)
    assert A.shortest_arc_degrees(-179.0, 179.0) == pytest.approx(-2.0)
    assert A.shortest_arc_degrees(0.0, 190.0) == pytest.approx(-170.0)
    # A half turn is the boundary of the interval and lands on -180; +180 and -180
    # describe the same rotation, so only the magnitude is meaningful.
    assert abs(A.shortest_arc_degrees(0.0, 180.0)) == pytest.approx(180.0)
    assert abs(A.shortest_arc_degrees(-90.0, 90.0)) == pytest.approx(180.0)


def test_yaw_recovered_from_direction():
    for yaw in (0.0, 45.0, 90.0, -90.0, 179.0, -179.0):
        sample = _sample(yaw)
        assert A.yaw_degrees_from_direction(sample["cam_direction"]) == pytest.approx(yaw, abs=1e-6)


# --- snap detection -----------------------------------------------------------

def test_normal_wraparound_is_not_flagged():
    """359 -> 1 degrees is an ordinary wrap of +2 deg, not a 358 deg snap."""
    samples = _sequence([359.0, 1.0, 3.0, 5.0])
    snaps = A.detect_snaps(samples, "synthetic_wrap", A.SnapThresholds())
    assert snaps == [], f"a normal 359->1 wrap was flagged as a snap: {snaps}"


def test_smooth_rotation_is_not_flagged():
    samples = _sequence([i * 2.0 for i in range(50)])       # 20 deg/s
    assert A.detect_snaps(samples, "synthetic_smooth", A.SnapThresholds()) == []


def test_yaw_snap_is_flagged_with_the_right_magnitude():
    samples = _sequence([0.0, 0.0, 180.0, 180.0])
    snaps = A.detect_snaps(samples, "synthetic_snap", A.SnapThresholds())
    assert len(snaps) == 1
    assert snaps[0].sample_index == 2
    assert abs(snaps[0].yaw_change_deg) == pytest.approx(180.0)
    assert snaps[0].is_yaw and not snaps[0].is_position


def test_threshold_is_configurable():
    samples = _sequence([0.0, 30.0, 60.0])                  # 30 deg per sample
    assert A.detect_snaps(samples, "s", A.SnapThresholds(yaw_deg_per_sample=45.0)) == []
    assert len(A.detect_snaps(samples, "s", A.SnapThresholds(yaw_deg_per_sample=20.0))) == 2


def test_first_sample_is_never_flagged():
    """There is no predecessor, and a trajectory boundary must not be crossed."""
    samples = _sequence([0.0, 90.0])
    snaps = A.detect_snaps(samples, "s", A.SnapThresholds())
    assert [s.sample_index for s in snaps] == [1]


def test_fast_motion_consistent_with_velocity_is_a_position_jump():
    # 12 m in 0.1 s = 120 m/s, and cam_velocity agrees -> fast motion.
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 12.0)]
    samples = _sequence([0.0, 0.0], positions=positions, velocity=(0.0, 0.0, 120.0))
    snaps = A.detect_snaps(samples, "s", A.SnapThresholds())
    assert len(snaps) == 1
    assert snaps[0].is_position
    assert snaps[0].displacement_speed_m_s == pytest.approx(120.0)
    assert snaps[0].velocity_consistent is True


def test_teleport_inconsistent_with_velocity_is_not_consistent():
    # 12 m in 0.1 s while the recorded velocity says the camera was still: a teleport.
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 12.0)]
    samples = _sequence([0.0, 0.0], positions=positions, velocity=(0.0, 0.0, 0.0))
    snaps = A.detect_snaps(samples, "s", A.SnapThresholds())
    assert len(snaps) == 1
    assert snaps[0].velocity_consistent is False


def test_snap_window_is_symmetric_and_clipped():
    samples = _sequence([0.0, 180.0] + [180.0] * 28)
    snaps = A.detect_snaps(samples, "s", A.SnapThresholds())
    affected = A.snap_affected_samples(snaps, len(samples), window=10)
    assert affected == set(range(0, 12))          # index 1 +/- 10, clipped at 0
    assert min(affected) == 0 and max(affected) == 11


def test_snap_kind_filter_separates_yaw_from_position():
    yaw_samples = _sequence([0.0, 180.0] + [180.0] * 20)
    yaw_snaps = A.detect_snaps(yaw_samples, "s", A.SnapThresholds())
    positions = [(0.0, 0.0, 0.0)] + [(0.0, 0.0, i * 3.0) for i in range(1, 22)]
    position_snaps = A.detect_snaps(
        _sequence([0.0] * 22, positions=positions, velocity=(0.0, 0.0, 30.0)), "s",
        A.SnapThresholds())

    assert A.snap_affected_samples(yaw_snaps, 22, 2, "yaw")
    assert not A.snap_affected_samples(yaw_snaps, 22, 2, "position")
    assert A.snap_affected_samples(position_snaps, 22, 2, "position")
    assert not A.snap_affected_samples(position_snaps, 22, 2, "yaw")


# --- cause partition ----------------------------------------------------------

def test_categories_are_exhaustive_and_mutually_exclusive():
    """Every combination of the three inputs lands in exactly one category.

    Also cross-checks against the live audit output, where the categories must sum
    to the primary-subset positive count exactly.
    """
    low, onset = 5.0, 50.0
    seen: dict[str, int] = {}
    for snap in (False, True):
        for speed in (0.0, 4.9, 5.0, 100.0):
            for accel in (0.0, 49.9, 50.1, 500.0):
                cause = A.dominant_cause(snap, speed, accel, low, onset)
                assert cause in A.CAUSE_ORDER, f"{cause} is not a declared category"
                seen[cause] = seen.get(cause, 0) + 1
    assert sum(seen.values()) == 2 * 4 * 4, "the partition lost or duplicated a combination"
    assert seen.get("other", 0) == 0, "the residual category should be unreachable here"

    # snap wins over everything
    assert A.dominant_cause(True, 0.0, 0.0, low, onset) == "snap_affected"
    assert A.dominant_cause(True, 999.0, 999.0, low, onset) == "snap_affected"
    # then slow, then onset, then steady
    assert A.dominant_cause(False, 1.0, 999.0, low, onset) == "low_angular_speed"
    assert A.dominant_cause(False, 50.0, 100.0, low, onset) == "turn_onset"
    assert A.dominant_cause(False, 50.0, 1.0, low, onset) == "steady_rotation"


def test_live_error_analysis_categories_sum_to_positives():
    """The recorded analysis must partition the positives exactly."""
    path = ROOT / "training" / "experiments" / "phase3a1" / "error_analysis.json"
    if not path.is_file():
        pytest.skip("run_audit.py has not been run")
    analysis = json.loads(path.read_text(encoding="utf-8"))
    total = sum(entry["positives"] for entry in analysis["categories"].values())
    assert set(analysis["categories"]) == set(A.CAUSE_ORDER)
    assert total == analysis["positives_total"], (
        f"categories sum to {total}, positives_total is {analysis['positives_total']}")
    assert sum(analysis["categories_with_any_flagged_snap"].values()) == analysis["positives_total"]


# --- sidecar safety -----------------------------------------------------------

def test_sidecar_columns_cannot_enter_the_feature_matrix():
    """Every sidecar column must be rejected by the allow-list."""
    config = _config()
    allow = list(config["features"]["allow"])
    for column in A.SIDECAR_COLUMNS:
        assert column not in allow, f"sidecar column {column!r} is in the allow-list"

    rows = [{"coverage": 0.5}]
    for column in A.SIDECAR_COLUMNS:
        with pytest.raises(KeyError):
            F.assemble_matrix(rows, ["coverage", column])


def test_live_sidecar_has_the_declared_columns_and_no_extras():
    path = ROOT / "training" / "experiments" / "phase3a1" / "eval_flags.csv"
    if not path.is_file():
        pytest.skip("run_audit.py has not been run")
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(A.SIDECAR_COLUMNS)


# --- split v1 -----------------------------------------------------------------

def test_phase2_5_manifest_is_unchanged():
    """The original split manifest must be byte-identical to its recorded hash."""
    assert PHASE2_5_MANIFEST.is_file()
    digest = hashlib.sha256(PHASE2_5_MANIFEST.read_bytes()).hexdigest()
    assert digest == PHASE2_5_SHA256, (
        "the Phase 2.5 split manifest changed; it must never be modified")


def test_split_v1_is_disjoint_complete_and_intact():
    manifest = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    partitions = {name: [entry["trajectory_id"] for entry in entries]
                  for name, entries in manifest["splits"].items()}

    seen: dict[str, str] = {}
    for name, ids in partitions.items():
        for trajectory_id in ids:
            assert trajectory_id not in seen, (
                f"{trajectory_id} appears in both {seen[trajectory_id]} and {name}")
            seen[trajectory_id] = name

    index = json.loads(
        (ROOT / "dataset" / "trajectories" / "full" / "index.json").read_text(encoding="utf-8"))
    declared = {entry["trajectory_id"]: entry["kind"] for entry in index["trajectories"]}
    assert set(seen) == set(declared), "split v1 does not cover exactly the 14 trajectories"
    assert len(seen) == 14

    # Trajectories are intact: each appears once and kinds match the index.
    for name, entries in manifest["splits"].items():
        for entry in entries:
            assert declared[entry["trajectory_id"]] == entry["kind"], (
                f"{entry['trajectory_id']} kind disagrees with the trajectory index")
    assert manifest["counts"] == {"train": len(partitions["train"]),
                                  "validation": len(partitions["validation"]),
                                  "test": len(partitions["test"])}


def test_split_v1_test_covers_the_four_required_roles():
    manifest = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    index = json.loads(
        (ROOT / "dataset" / "trajectories" / "full" / "index.json").read_text(encoding="utf-8"))
    declared = {entry["trajectory_id"]: entry["kind"] for entry in index["trajectories"]}
    test_kinds = {declared[entry["trajectory_id"]] for entry in manifest["splits"]["test"]}
    assert "sprint_corner" in test_kinds or "sprint_straight" in test_kinds, "no sprinting"
    assert "look_around" in test_kinds, "no pure rotation trajectory"
    assert "walk_straight" in test_kinds, "no translation-dominant trajectory"
    assert "walk_corner" in test_kinds, "no walking trajectory"


def test_split_v1_keeps_every_replicated_movement_type_in_training():
    manifest = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    index = json.loads(
        (ROOT / "dataset" / "trajectories" / "full" / "index.json").read_text(encoding="utf-8"))
    declared = {entry["trajectory_id"]: entry["kind"] for entry in index["trajectories"]}

    counts: dict[str, int] = {}
    for kind in declared.values():
        counts[kind] = counts.get(kind, 0) + 1
    replicated = {kind for kind, n in counts.items() if n > 1}

    train_kinds = {declared[entry["trajectory_id"]] for entry in manifest["splits"]["train"]}
    missing = replicated - train_kinds
    assert not missing, f"replicated movement types missing from training: {sorted(missing)}"


def test_split_v1_keeps_snap_contaminated_trajectories_out_of_evaluation():
    manifest = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    contaminated = set(manifest["snap_policy"]["snap_contaminated_trajectories"])
    for name in ("validation", "test"):
        ids = {entry["trajectory_id"] for entry in manifest["splits"][name]}
        assert not (ids & contaminated), f"{name} contains a snap-contaminated trajectory"


def test_declared_kinds_match_the_index():
    config = _config()
    kinds = A.declared_kinds(ROOT / config["paths"]["trajectory_index"])
    assert len(kinds) == 14
    assert kinds["t11_slow_walk_down"] == "walk_straight"
    assert kinds["t12_sprint_weave"] == "sprint_corner"
    assert kinds["t14_plaza_look"] == "look_around"


def test_recorded_trajectory_kind_column_is_not_the_declared_kind():
    """Documents the known defect: the samples column is the id suffix for 3 trajectories."""
    config = _config()
    declared = A.declared_kinds(ROOT / config["paths"]["trajectory_index"])
    dataset = ROOT / "dataset" / "processed" / "run_full_p25"
    mismatched = []
    for trajectory_id in A.trajectory_ids(dataset):
        recorded = json.loads(
            (dataset / f"{trajectory_id}.samples.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )["trajectory_kind"]
        if recorded != declared[trajectory_id]:
            mismatched.append(trajectory_id)
    assert sorted(mismatched) == ["t11_slow_walk_down", "t12_sprint_weave", "t14_plaza_look"], (
        f"expected exactly 3 mismatches, got {mismatched}")
