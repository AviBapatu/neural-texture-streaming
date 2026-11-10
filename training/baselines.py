"""Non-neural baselines.

There are three, in increasing order of sophistication:

1. ``reactive`` - the object is wanted because it is visible right now. This is
   what a streaming system does with no prediction at all, and it is the honest
   floor every predictive method must beat.
2. ``coverage_resolution`` - predict the resolution an object will need from its
   *current* projected coverage, using the same mapping the labeller uses.
3. ``predictive`` - extrapolate the camera pose forward over the horizon under
   constant linear and angular velocity (both optionally damped), then ask whether
   the object's centre falls inside the frustum. The score is the angular margin to
   the frustum boundary, which gives a continuous ranking.

Nothing here learns from test data. The heuristic's hyper-parameters are searched on
validation folds only; ``run_baselines.py`` reports the whole grid.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import features as F


# --- camera pose cache --------------------------------------------------------

@dataclass
class PoseArrays:
    """Per-trajectory camera state, including world-frame values the model never sees."""

    position: dict[str, np.ndarray]
    matrix: dict[str, np.ndarray]
    quaternion: dict[str, np.ndarray]
    velocity: dict[str, np.ndarray]
    angular_velocity: dict[str, np.ndarray]
    fov: dict[str, np.ndarray]
    viewport: dict[str, np.ndarray]

    def horizontal_half_angle_deg(self, trajectory_id: str, index: int) -> float:
        vertical = math.radians(float(self.fov[trajectory_id][index]))
        width, height = self.viewport[trajectory_id][index]
        aspect = float(width) / float(height) if height else 1.0
        return math.degrees(math.atan(math.tan(vertical * 0.5) * aspect))

    def vertical_half_angle_deg(self, trajectory_id: str, index: int) -> float:
        return float(self.fov[trajectory_id][index]) * 0.5


def load_pose_arrays(dataset_dir: str | Path) -> PoseArrays:
    """Load camera state for every trajectory, with the orientation-derived fix applied."""
    directory = Path(dataset_dir)
    position, matrix, quaternion, velocity, angular_velocity, fov, viewport = ({} for _ in range(7))
    for path in sorted(directory.glob("*.samples.jsonl")):
        trajectory_id = path.name.replace(".samples.jsonl", "")
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        quats = np.array([r["cam_quaternion"] for r in rows], dtype=np.float64)
        times = np.array([float(r["session_time"]) for r in rows], dtype=np.float64)
        corrected, _ = F.angular_velocity_from_quaternions(quats, times)
        quaternion[trajectory_id] = quats
        position[trajectory_id] = np.array([r["cam_position"] for r in rows], dtype=np.float64)
        matrix[trajectory_id] = np.array([F.quaternion_to_matrix(q) for q in quats])
        velocity[trajectory_id] = np.array([r["cam_velocity"] for r in rows], dtype=np.float64)
        angular_velocity[trajectory_id] = corrected
        fov[trajectory_id] = np.array([float(r["cam_fov"]) for r in rows], dtype=np.float64)
        viewport[trajectory_id] = np.array([r["viewport"] for r in rows], dtype=np.float64)
    return PoseArrays(position=position, matrix=matrix, quaternion=quaternion,
                      velocity=velocity, angular_velocity=angular_velocity,
                      fov=fov, viewport=viewport)


# --- shared helpers -----------------------------------------------------------

@dataclass
class RowPoseCache:
    """Per-row camera and object state, gathered once and reused across the grid.

    The baselines are evaluated over a few hundred hyper-parameter combinations, so
    the per-trajectory lookup is done once here and everything after it is a
    vectorised operation over all rows.
    """

    position: np.ndarray
    matrix: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    half_horizontal: np.ndarray
    half_vertical: np.ndarray
    centre: np.ndarray
    viewport_width: np.ndarray
    viewport_height: np.ndarray

    def __len__(self) -> int:
        return self.position.shape[0]


def build_row_pose_cache(table, poses: PoseArrays,
                         object_centre: dict[str, np.ndarray]) -> RowPoseCache:
    """Gather per-row camera state. Fourteen slicing passes, no per-row Python."""
    count = len(table)
    position = np.zeros((count, 3))
    matrix = np.zeros((count, 3, 3))
    velocity = np.zeros((count, 3))
    angular = np.zeros((count, 3))
    half_h = np.zeros(count)
    half_v = np.zeros(count)
    width = np.zeros(count)
    height = np.zeros(count)

    for index, trajectory_id in enumerate(table.trajectories):
        mask = table.trajectory == index
        if not mask.any():
            continue
        sample = table.sample_index[mask]
        position[mask] = poses.position[trajectory_id][sample]
        matrix[mask] = poses.matrix[trajectory_id][sample]
        velocity[mask] = poses.velocity[trajectory_id][sample]
        angular[mask] = poses.angular_velocity[trajectory_id][sample]
        viewport = poses.viewport[trajectory_id][sample]
        width[mask] = viewport[:, 0]
        height[mask] = viewport[:, 1]
        vertical = np.radians(poses.fov[trajectory_id][sample])
        aspect = np.divide(viewport[:, 0], viewport[:, 1],
                           out=np.ones_like(viewport[:, 0]), where=viewport[:, 1] > 0)
        half_v[mask] = np.degrees(vertical * 0.5)
        half_h[mask] = np.degrees(np.arctan(np.tan(vertical * 0.5) * aspect))

    centre = np.array([object_centre[object_id] for object_id in table.object_id])
    return RowPoseCache(position=position, matrix=matrix, velocity=velocity,
                        angular_velocity=angular, half_horizontal=half_h,
                        half_vertical=half_v, centre=centre, viewport_width=width,
                        viewport_height=height)


def _rotation_matrices(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Batched axis-angle rotation matrices, shape (n, 3, 3)."""
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.stack([
        np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], axis=1),
        np.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], axis=1),
        np.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], axis=1),
    ], axis=1)


def margin_deg(cache: RowPoseCache, at_offsets: Sequence[float], damping: float,
               angular_damping: float) -> np.ndarray:
    """Max frustum margin over forward offsets, in degrees. Positive = inside.

        margin = min(vertical_half_fov - |elevation|, horizontal_half_fov - |azimuth|)

    For each offset the camera is extrapolated under constant linear velocity scaled
    by ``(1 - damping)`` and constant angular velocity scaled by
    ``(1 - angular_damping)``, and the object's centre is projected into that pose.
    Taking the maximum over the window approximates "visible at any point in
    [t, t+h]", which is what the label asks.
    """
    best = np.full(len(cache), -np.inf)
    speed = np.linalg.norm(cache.angular_velocity, axis=1)
    moving = speed > 1e-9
    safe_speed = np.where(moving, speed, 1.0)
    axis = cache.angular_velocity / safe_speed[:, None]

    for offset in at_offsets:
        if offset <= 0.0:
            position_at = cache.position
            matrix_at = cache.matrix
        else:
            position_at = cache.position + cache.velocity * (1.0 - damping) * offset
            angle = speed * (1.0 - angular_damping) * offset
            delta = _rotation_matrices(axis, angle)
            delta[~moving] = np.eye(3)
            matrix_at = delta @ cache.matrix

        local = np.einsum("nij,nj->ni", np.transpose(matrix_at, (0, 2, 1)),
                          cache.centre - position_at)
        azimuth = np.degrees(np.arctan2(local[:, 0], -local[:, 2]))
        elevation = np.degrees(np.arctan2(local[:, 1], np.hypot(local[:, 0], local[:, 2])))
        value = np.minimum(cache.half_vertical - np.abs(elevation),
                           cache.half_horizontal - np.abs(azimuth))
        best = np.maximum(best, value)
    return best


# --- baseline 1: reactive -----------------------------------------------------

def reactive_score(table, target: str) -> np.ndarray:
    """Score 1.x for currently visible objects, then order the rest by proximity.

    The tie-break matters: a pure 0/1 reactive score leaves recall@k undefined
    among the (large) invisible majority. Falling back to a distance ordering makes
    the ranking total and reproduces what a system would do with no prediction -
    prefer what is on screen, then what is closest.
    """
    proximity = 1.0 / (1.0 + np.asarray(table.distance, dtype=np.float64))
    if target == "frustum":
        visible = np.asarray(table.frustum_visible, dtype=bool)
    else:
        visible = np.asarray(table.occlusion_state) == 1
    return np.where(visible, 1.0 + proximity, proximity)


# --- baseline 2: coverage threshold resolution --------------------------------

def coverage_resolution_predictions(table, cache: RowPoseCache) -> np.ndarray:
    """Predict the needed resolution from the *current* coverage.

    Uses ``features.resolution_from_coverage``, the verified port of the labeller's
    mapping, so the baseline and the label come from the same rule. The only
    difference is that the baseline cannot see the future.
    """
    coverage = np.asarray(table.coverage, dtype=np.float64)
    required_px = np.sqrt(np.maximum(coverage, 0.0) * cache.viewport_width *
                          cache.viewport_height)
    predictions = np.zeros(len(table), dtype=np.int32)
    for step in F.RESOLUTION_LADDER:
        predictions = np.where((predictions == 0) & (required_px <= float(step)),
                               step, predictions)
    predictions = np.where(required_px <= 0.0, 0, predictions)
    predictions = np.where((predictions == 0) & (required_px > 0.0),
                           F.RESOLUTION_LADDER[-1], predictions)
    return predictions


# --- baseline 3: predictive heuristic -----------------------------------------

@dataclass(frozen=True)
class HeuristicParams:
    damping: float
    angular_damping: float
    integrate_steps: int
    margin_threshold_deg: float
    occlusion_penalty_deg: float = 0.0

    def as_dict(self) -> dict:
        return {
            "damping": self.damping,
            "angular_damping": self.angular_damping,
            "integrate_steps": self.integrate_steps,
            "margin_threshold_deg": self.margin_threshold_deg,
            "occlusion_penalty_deg": self.occlusion_penalty_deg,
        }


def predictive_score(table, cache: RowPoseCache, params: HeuristicParams, horizon: float,
                     occlusion_aware: bool = False) -> np.ndarray:
    """Continuous futuribility score for one horizon.

    The horizon is discretised into ``integrate_steps`` forward offsets. With 10
    steps and a 1.0 s horizon the offsets land exactly on the 0.1 s samples the
    labels are computed from, so the discretisation is no coarser than the label.
    """
    steps = max(1, int(params.integrate_steps))
    offsets = [horizon * (k + 1) / steps for k in range(steps)]
    # Offset 0 means "where it is now", which the reactive baseline already covers.
    # Including it would let a currently-visible object score high with no prediction.
    score = margin_deg(cache, offsets, params.damping, params.angular_damping)
    if occlusion_aware:
        blocked = np.asarray(table.occlusion_state) == 0
        score = np.where(blocked, score - params.occlusion_penalty_deg, score)
    return score


def threshold_predictions(score: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(score, dtype=np.float64) >= float(threshold)


def default_parameter_grid(config: dict) -> dict[str, list]:
    """Hyper-parameter grid from the config.

    ``margin_thresholds_deg`` is a binarisation threshold, not a knob that changes
    the score, so it lives outside this grid and is applied afterwards. That keeps
    the expensive extrapolation from being repeated once per threshold.
    """
    grid = {key: list(values)
            for key, values in config["baselines"]["predictive"]["grid"].items()}
    if config["baselines"]["predictive"].get("occlusion_variants", False):
        grid["occlusion_penalty_deg"] = [0.0, 10.0, 30.0]
    return grid


def margin_thresholds(config: dict) -> list[float]:
    return [float(v) for v in config["baselines"]["predictive"]["margin_thresholds_deg"]]


def iter_parameter_grid(grid: dict[str, list], occlusion_aware: bool = False):
    """Deterministic cartesian product of the grid (thresholds excluded)."""
    import itertools
    keys = sorted(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        chosen = dict(zip(keys, values))
        if not occlusion_aware and chosen.get("occlusion_penalty_deg", 0.0) != 0.0:
            continue
        yield HeuristicParams(
            damping=float(chosen["damping"]),
            angular_damping=float(chosen["angular_damping"]),
            integrate_steps=int(chosen["integrate_steps"]),
            margin_threshold_deg=0.0,
            occlusion_penalty_deg=float(chosen.get("occlusion_penalty_deg", 0.0)),
        )
