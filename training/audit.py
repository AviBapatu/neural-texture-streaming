"""Phase 3a.1 trajectory audit and heuristic error analysis.

Read-only over `dataset/`. Nothing here is a model input: the evaluation sidecar this
module produces uses future information for stratification only, and
`training/tests/test_audit.py` proves the allow-list rejects those columns.

The angular speeds used throughout are recomputed from the recorded orientations with
the shortest arc chosen, not read from the recorded `cam_angular_velocity` field, which
is defective (see `docs/known-issues.md`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import baselines as B
from . import dataset as D
from . import evaluate as E
from . import features as F


# --- small geometry helpers ---------------------------------------------------

def yaw_degrees_from_direction(direction: Sequence[float]) -> float:
    """Recover the camera's yaw from its view direction.

    With a Y-only rotation, ``view = (-sin(yaw), 0, -cos(yaw))``, so
    ``yaw = atan2(-view.x, -view.z)``. Returns degrees in (-180, 180].
    """
    return math.degrees(math.atan2(-float(direction[0]), -float(direction[2])))


def shortest_arc_degrees(a: float, b: float) -> float:
    """Signed change from ``a`` to ``b`` wrapped into [-180, 180).

    A 359 deg -> 1 deg change is +2 deg, not -358 deg. This is the whole point of
    the helper: the snap detector must not fire on an ordinary wrap.

    A half turn lands on -180 rather than +180 because ``[-180, 180)`` is the
    half-open interval the modulo produces; the two are the same rotation, so
    magnitudes are unaffected.
    """
    return (float(b) - float(a) + 180.0) % 360.0 - 180.0


def angular_speed_deg_s(quaternions: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Shortest-arc angular speed per sample, in degrees per second."""
    velocity, _ = F.angular_velocity_from_quaternions(quaternions, times)
    return np.degrees(np.linalg.norm(velocity, axis=1))


def angular_accel_deg_s2(quaternions: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Magnitude of the backward difference of angular velocity, in degrees per second^2."""
    velocity, _ = F.angular_velocity_from_quaternions(quaternions, times)
    speed = np.linalg.norm(velocity, axis=1)
    accel = np.zeros_like(speed)
    if speed.size > 1:
        delta = np.diff(times)
        safe = np.where(delta > 0.0, delta, np.nan)
        accel[1:] = np.nan_to_num(np.abs(np.diff(speed)) / safe)
    return np.degrees(accel)


# --- snap detection -----------------------------------------------------------

@dataclass
class SnapThresholds:
    yaw_deg_per_sample: float = 45.0
    position_m_per_sample: float = 2.0

    @classmethod
    def from_config(cls, config: dict) -> "SnapThresholds":
        audit = config.get("audit", {})
        return cls(yaw_deg_per_sample=float(audit.get("yaw_threshold_deg_per_sample", 45.0)),
                   position_m_per_sample=float(audit.get("position_threshold_m_per_sample", 2.0)))


@dataclass
class Snap:
    trajectory_id: str
    sample_index: int
    session_time: float
    yaw_change_deg: float
    position_jump_m: float
    kind: str                  # "yaw" | "position" | "yaw+position"
    declared_kind: str
    displacement_speed_m_s: float = 0.0
    recorded_speed_m_s: float = 0.0
    velocity_consistent: bool = True

    @property
    def is_yaw(self) -> bool:
        return "yaw" in self.kind

    @property
    def is_position(self) -> bool:
        return "position" in self.kind

    def as_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "sample_index": self.sample_index,
            "session_time": round(self.session_time, 3),
            "yaw_change_deg": round(self.yaw_change_deg, 4),
            "position_jump_m": round(self.position_jump_m, 4),
            "snap_kind": self.kind,
            "trajectory_kind": self.declared_kind,
            "displacement_speed_m_s": round(self.displacement_speed_m_s, 3),
            "recorded_speed_m_s": round(self.recorded_speed_m_s, 3),
            "velocity_consistent": int(self.velocity_consistent),
        }


def detect_snaps(samples: Sequence[dict], trajectory_id: str,
                 thresholds: SnapThresholds) -> list[Snap]:
    """Flag every sample whose yaw change or position jump exceeds the thresholds.

    Sample 0 has no predecessor and is never flagged; trajectory boundaries are
    therefore never crossed.

    A position jump is additionally checked against the recorded velocity. A large
    displacement that matches ``cam_velocity`` is a fast camera, not a
    discontinuity - the sprint trajectories legitimately move 20-120 m/s and would
    otherwise be flagged wholesale. Only a displacement inconsistent with the
    recorded velocity is evidence of a teleport.
    """
    snaps: list[Snap] = []
    if len(samples) < 2:
        return snaps
    kind = str(samples[0].get("trajectory_kind", ""))
    yaws = [yaw_degrees_from_direction(s["cam_direction"]) for s in samples]
    for index in range(1, len(samples)):
        yaw_change = shortest_arc_degrees(yaws[index - 1], yaws[index])
        a = np.asarray(samples[index - 1]["cam_position"], dtype=np.float64)
        b = np.asarray(samples[index]["cam_position"], dtype=np.float64)
        jump = float(np.linalg.norm(b - a))
        delta_t = float(samples[index]["session_time"]) - float(samples[index - 1]["session_time"])
        displacement_speed = jump / delta_t if delta_t > 0 else float("inf")
        recorded_speed = float(np.linalg.norm(
            np.asarray(samples[index - 1]["cam_velocity"], dtype=np.float64)))
        consistent = (recorded_speed > 1e-9
                      and abs(displacement_speed - recorded_speed) / recorded_speed <= 0.15)

        yaw_hit = abs(yaw_change) > thresholds.yaw_deg_per_sample
        position_hit = jump > thresholds.position_m_per_sample
        if not (yaw_hit or position_hit):
            continue
        snaps.append(Snap(
            trajectory_id=trajectory_id,
            sample_index=index,
            session_time=float(samples[index]["session_time"]),
            yaw_change_deg=yaw_change,
            position_jump_m=jump,
            kind="yaw+position" if (yaw_hit and position_hit) else ("yaw" if yaw_hit else "position"),
            declared_kind=kind,
            displacement_speed_m_s=displacement_speed,
            recorded_speed_m_s=recorded_speed,
            velocity_consistent=consistent,
        ))
    return snaps


def snap_affected_samples(snaps: Iterable[Snap], sample_count: int, window: int,
                          kinds: str = "any") -> set[int]:
    """Sample indices within ``window`` samples of a snap.

    ``kinds`` selects which snaps count: "any", "yaw" or "position". The yaw-only
    view isolates the generator defect; the position-only view isolates fast motion.
    """
    affected: set[int] = set()
    for snap in snaps:
        if kinds == "yaw" and not snap.is_yaw:
            continue
        if kinds == "position" and not snap.is_position:
            continue
        low = max(0, snap.sample_index - window)
        high = min(sample_count - 1, snap.sample_index + window)
        affected.update(range(low, high + 1))
    return affected


# --- loading ------------------------------------------------------------------

@dataclass
class TrajectoryData:
    trajectory_id: str
    kind: str
    samples: list[dict]
    labels: list[dict]
    quaternions: np.ndarray = field(repr=False, default=None)
    times: np.ndarray = field(repr=False, default=None)

    @property
    def sample_count(self) -> int:
        return len(self.samples)


def load_trajectory(directory: Path, trajectory_id: str) -> TrajectoryData:
    samples = D.load_labels(directory / f"{trajectory_id}.samples.jsonl")
    labels = D.load_labels(directory / f"{trajectory_id}.labels.jsonl")
    quaternions = np.array([s["cam_quaternion"] for s in samples], dtype=np.float64)
    times = np.array([float(s["session_time"]) for s in samples], dtype=np.float64)
    kind = str(samples[0].get("trajectory_kind", "")) if samples else ""
    return TrajectoryData(trajectory_id=trajectory_id, kind=kind, samples=samples,
                          labels=labels, quaternions=quaternions, times=times)


def trajectory_ids(directory: Path) -> list[str]:
    return sorted(p.name.replace(".labels.jsonl", "") for p in directory.glob("*.labels.jsonl"))


# The processed samples carry a `trajectory_kind` column, but for three trajectories
# it holds the trajectory-id suffix rather than the declared movement type:
# t11_slow_walk_down -> "slow_walk_down" (declared "walk_straight"), t12_sprint_weave ->
# "sprint_weave" (declared "sprint_corner"), t14_plaza_look -> "plaza_look" (declared
# "look_around"). The labeller derives it as `trajectory_id.split("_", 1)[-1]`
# (tools/label_future_demand.py). The trajectory index carries the true value, so the
# audit reads movement type from there instead. See docs/known-issues.md.
def declared_kinds(index_path: str | Path) -> dict[str, str]:
    """Movement type per trajectory, from the trajectory index (authoritative)."""
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    entries = payload["trajectories"] if isinstance(payload, dict) else payload
    return {entry["trajectory_id"]: str(entry.get("kind", "")) for entry in entries}


# --- Task A: the audit table --------------------------------------------------

def trajectory_row(data: TrajectoryData, threshold_deg: float, window: int,
                   frustum_visible_only: bool = True) -> dict:
    """One row of the per-trajectory audit table."""
    samples = data.samples
    count = len(samples)
    speed = np.array([float(np.linalg.norm(s["cam_velocity"])) for s in samples])
    angular = angular_speed_deg_s(data.quaternions, data.times)

    labels = data.labels
    def count_where(predicate) -> int:
        return sum(1 for row in labels if predicate(row))

    row = {
        "trajectory_id": data.trajectory_id,
        "movement_type": data.kind,
        "duration_s": round(float(samples[-1]["session_time"]), 2) if samples else 0.0,
        "samples": count,
        "samples_with_candidates": len({row["sample_index"] for row in labels}),
        "label_rows": len(labels),
        "mean_linear_speed_m_s": round(float(speed.mean()), 3),
        "max_linear_speed_m_s": round(float(speed.max()), 3),
        "mean_angular_speed_deg_s": round(float(angular.mean()), 3),
        "max_angular_speed_deg_s": round(float(angular.max()), 3),
    }
    for horizon, suffix in (("0.5", "0_5"), ("1.0", "1_0")):
        tag = horizon.replace(".", "_")
        frustum_key = f"frustum_visible_in_{suffix}s"
        occlusion_key = f"occlusion_visible_in_{suffix}s"
        row[f"frustum_pos_{tag}s"] = count_where(lambda r, k=frustum_key: bool(r[k]))
        row[f"frustum_new_{tag}s"] = count_where(
            lambda r, k=frustum_key: bool(r[k]) and not r["frustum_visible"])
        row[f"occlusion_pos_{tag}s"] = count_where(
            lambda r, k=occlusion_key: r[k] is True)
        row[f"occlusion_new_{tag}s"] = count_where(
            lambda r, k=occlusion_key: r[k] is True and r["occlusion_visible"] is not True)
        row[f"occlusion_unknown_{tag}s"] = count_where(
            lambda r, k=occlusion_key: r[k] is None)
    return row


def newly_visible_samples(data: TrajectoryData, horizon_suffix: str = "0_5") -> dict[int, int]:
    """Count of newly-visible events (0.5 s frustum) per sample index."""
    key = f"frustum_visible_in_{horizon_suffix}s"
    result: dict[int, int] = {}
    for row in data.labels:
        if bool(row[key]) and not row["frustum_visible"]:
            result[int(row["sample_index"])] = result.get(int(row["sample_index"]), 0) + 1
    return result


def snap_impact(data: TrajectoryData, snaps: list[Snap], window: int) -> dict:
    """Fraction of newly-visible events lying within ``window`` samples of a snap."""
    affected = snap_affected_samples(snaps, data.sample_count, window)
    per_sample = newly_visible_samples(data)
    total = sum(per_sample.values())
    hit = sum(count for index, count in per_sample.items() if index in affected)
    return {
        "trajectory_id": data.trajectory_id,
        "newly_visible_0_5s": total,
        "newly_visible_near_snap": hit,
        "fraction_near_snap": round(hit / total, 4) if total else 0.0,
        "snap_affected_samples": len(affected),
        "samples": data.sample_count,
    }


# --- Task B: evaluation sidecar ----------------------------------------------

# These column names must never be features. The test asserts the allow-list
# rejects them; the names are listed here so that assertion has something to
# check against.
SIDECAR_COLUMNS = ("trajectory_id", "sample_index", "snap_affected",
                   "snap_affected_yaw", "snap_affected_position", "movement_type")


def build_eval_flags(dataset_dir: Path, config: dict,
                     snap_lookup: dict[str, list[Snap]],
                     kinds: dict[str, str] | None = None) -> list[dict]:
    """Per-(trajectory, sample) evaluation flags.

    Uses future information (a snap later in the trajectory affects earlier samples)
    for stratification only. These rows never enter the feature matrix.

    ``snap_affected`` is the union the brief asks for. The two split columns exist
    because the union conflates two very different things in this dataset: 3 real
    yaw defects and 115 legitimate fast-sprint position jumps. At *row* level the two
    currently coincide, because the sprint jumps happen where no object is within
    25 m and therefore produce no label rows at all.
    """
    window = int(config.get("audit", {}).get("snap_window_samples", 10))
    kinds = kinds or {}
    flags: list[dict] = []
    for trajectory_id in trajectory_ids(dataset_dir):
        data = load_trajectory(dataset_dir, trajectory_id)
        snaps = snap_lookup.get(trajectory_id, [])
        any_affected = snap_affected_samples(snaps, data.sample_count, window, "any")
        yaw_affected = snap_affected_samples(snaps, data.sample_count, window, "yaw")
        position_affected = snap_affected_samples(snaps, data.sample_count, window, "position")
        movement_type = kinds.get(trajectory_id, data.kind)
        for index in range(data.sample_count):
            flags.append({
                "trajectory_id": trajectory_id,
                "sample_index": index,
                "snap_affected": int(index in any_affected),
                "snap_affected_yaw": int(index in yaw_affected),
                "snap_affected_position": int(index in position_affected),
                "movement_type": movement_type,
            })
    return flags


# --- Task C: heuristic error analysis ----------------------------------------

CAUSE_ORDER = ("snap_affected", "low_angular_speed", "turn_onset", "steady_rotation", "other")


def dominant_cause(snap_affected: bool, angular_speed_deg_s_value: float,
                   angular_accel_deg_s2_value: float, low_speed: float, onset_accel: float) -> str:
    """Assign one mutually exclusive cause, first match wins.

    Precedence: snap > low angular speed (translation or corner reveal, the object
    appears without the camera rotating) > turn onset (accelerating into a turn) >
    steady rotation > other.
    """
    if snap_affected:
        return "snap_affected"
    if angular_speed_deg_s_value < low_speed:
        return "low_angular_speed"
    if angular_accel_deg_s2_value > onset_accel:
        return "turn_onset"
    if angular_speed_deg_s_value >= low_speed:
        return "steady_rotation"
    return "other"


def _recall_at_k_for_category(table, y_true: np.ndarray, score: np.ndarray,
                              category_mask: np.ndarray, primary: np.ndarray,
                              k: int) -> float:
    """recall@k restricted to one cause category.

    Only samples containing at least one positive of this category contribute, and
    the ranking is over the primary subset (not-currently-visible rows) in that
    sample - the same pool the headline recall@k uses.
    """
    keys = table.trajectory.astype(np.int64) * 1_000_000 + table.sample_index.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(keys[order])) + 1)
    values: list[float] = []
    for group in groups:
        pool = group[primary[group]]
        if pool.size == 0:
            continue
        positives = pool[y_true[pool] & category_mask[pool]]
        if positives.size == 0:
            continue
        ranked = pool[np.argsort(-score[pool], kind="stable")][:k]
        values.append(float(np.isin(ranked, positives).sum()) / positives.size)
    return float(np.mean(values)) if values else float("nan")


def distance_bin_labels(edges: Sequence[float]) -> list[str]:
    return [f"{edges[i]:g}-{edges[i + 1]:g}m" for i in range(len(edges) - 1)]


def bin_index(value: float, edges: Sequence[float]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return len(edges) - 2


def group_breakdown(table, y_true: np.ndarray, predicted: np.ndarray,
                    score: np.ndarray, primary: np.ndarray, groups: np.ndarray,
                    labels: Sequence[str]) -> dict:
    """Recall (and count) of positives per group, plus recall@5 where defined."""
    result = {}
    for index, label in enumerate(labels):
        mask = primary & (groups == index)
        positives = mask & y_true
        count = int(positives.sum())
        hits = int((positives & predicted).sum())
        result[label] = {
            "positives": count,
            "recall": round(hits / count, 4) if count else None,
            "recall_at_5": round(_recall_at_k_for_category(
                table, y_true, score, groups == index, primary, 5), 4) if count else None,
        }
    return result
