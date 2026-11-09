"""Causal feature extraction for future texture demand.

Every feature in this module is a function of the sample at time ``t`` and earlier
samples of the *same trajectory*. Nothing reads a later sample, a label, or a
future-derived mask. ``training/tests/test_leakage.py`` enforces that by rebuilding
features from a trajectory with all samples after ``t`` deleted and requiring an
exact match.

The feature matrix is assembled from an explicit allow-list in the config, so a new
column appearing in the dataset cannot silently become a model input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# The ladder used by the recording side (godot/scripts/visibility_service.gd).
RESOLUTION_LADDER = (64, 128, 256, 512, 1024, 2048)

# Object categories emitted by the environment. Kept alphabetical so the one-hot
# column order is stable.
CATEGORIES = ("bld", "plz", "prp", "rod", "shw", "wal")


# --- the resolution mapping ---------------------------------------------------

def resolution_from_coverage(coverage: float, viewport_width: float,
                             viewport_height: float) -> int:
    """Port of ``visibility_service.gd::required_resolution``.

    required_px = sqrt(coverage * viewport_w * viewport_h), then the smallest ladder
    step at least that large. Returns 0 when the object is not on screen.

    The GDScript is the canonical implementation and cannot be imported from
    Python, so this port is verified against the recorded ``required_resolution``
    values for every visible row rather than assumed to be equivalent
    (training/tests/test_features.py::test_resolution_port_matches_recorded).
    """
    if coverage <= 0.0 or viewport_width <= 0.0 or viewport_height <= 0.0:
        return 0
    required_px = math.sqrt(coverage * viewport_width * viewport_height)
    for step in RESOLUTION_LADDER:
        if float(step) >= required_px:
            return step
    return RESOLUTION_LADDER[-1]


# --- geometric helpers --------------------------------------------------------

def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Rotation matrix (world-from-local) for a Godot quaternion ``[x, y, z, w]``.

    Column 2 is the camera's local +Z, so the view direction is ``-matrix[:, 2]``.
    """
    x, y, z, w = (float(v) for v in quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def world_to_camera_batch(points: np.ndarray, camera_position: np.ndarray,
                          matrix: np.ndarray) -> np.ndarray:
    """World points -> camera-local coordinates. ``points`` is (n, 3)."""
    return (np.asarray(points, dtype=np.float64) - camera_position) @ matrix


def aabb_centre(bounds: dict) -> np.ndarray:
    position = np.asarray(bounds["position"], dtype=np.float64)
    size = np.asarray(bounds["size"], dtype=np.float64)
    return position + size * 0.5


def aabb_radius(bounds: dict) -> float:
    size = np.asarray(bounds["size"], dtype=np.float64)
    return float(np.linalg.norm(size) * 0.5)


def point_to_aabb_distance(point: np.ndarray, bounds: dict) -> float:
    """Same metric the labeller uses to pick candidates (distance to the box)."""
    position = np.asarray(bounds["position"], dtype=np.float64)
    size = np.asarray(bounds["size"], dtype=np.float64)
    low, high = position, position + size
    deltas = np.maximum(np.maximum(low - point, 0.0), point - high)
    return float(np.linalg.norm(deltas))


def angular_rate_about_axes(world_angular_velocity: np.ndarray,
                            matrix: np.ndarray) -> np.ndarray:
    """Project a world-space angular velocity onto the camera's local axes.

    Camera-local X is right, Y is up, Z is backwards, so the components are
    (pitch, yaw, roll) rates in rad/s.
    """
    return matrix.T @ np.asarray(world_angular_velocity, dtype=np.float64)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, batched. Quaternions are `[x, y, z, w]`."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=1)


def angular_velocity_from_quaternions(quaternions: np.ndarray,
                                      times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angular velocity from consecutive camera orientations, in world rad/s.

    Returns ``(angular_velocity, double_cover_flags)``.

    This deliberately replaces the recorded ``cam_angular_velocity`` field. The
    recorder computes ``q_prev.inverse() * q`` and takes ``get_angle()`` without
    choosing the shorter arc, so when the quaternion difference lands on the ``-q``
    representation it reports roughly ``2*pi - theta`` with a flipped axis. That is
    not a rounding error: on `t06_look_around` a genuine 18 deg/s rotation is
    recorded as 3582 deg/s, which then produces a 36005 deg/s^2 acceleration spike.

    Picking the shorter arc (negating the delta when its scalar part is negative)
    is exact for any rotation under 180 degrees per sample, which the 10 Hz
    sampling guarantees. The flag array marks the samples where the two disagree, so
    the extent of the recorded defect stays measurable.
    """
    count = quaternions.shape[0]
    angular = np.zeros((count, 3), dtype=np.float64)
    flagged = np.zeros(count, dtype=bool)
    if count < 2:
        return angular, flagged

    inverse = quaternions[:-1].copy()
    inverse[:, :3] *= -1.0
    delta = quaternion_multiply(inverse, quaternions[1:])
    norm = np.linalg.norm(delta, axis=1)
    norm[norm < 1e-12] = 1.0
    delta = delta / norm[:, None]
    negate = delta[:, 3] < 0.0
    flagged[1:] = negate
    delta[negate] *= -1.0

    angle = 2.0 * np.arccos(np.clip(delta[:, 3], -1.0, 1.0))
    sin_half = np.sqrt(np.maximum(0.0, 1.0 - delta[:, 3] * delta[:, 3]))
    delta_t = np.diff(times)
    safe = np.where(delta_t > 0.0, delta_t, np.nan)

    moving = sin_half > 1e-8
    axis = np.zeros((delta.shape[0], 3), dtype=np.float64)
    axis[moving] = delta[moving, :3] / sin_half[moving, None]
    angular[1:] = np.nan_to_num(axis * (angle / safe)[:, None])
    return angular, flagged


# --- per-sample causal features ----------------------------------------------

@dataclass
class SampleFeatures:
    """Backward-looking per-sample arrays. Index ``i`` uses only samples 0..i."""

    count: int
    position: np.ndarray
    matrix: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    local_angular_velocity: np.ndarray
    speed: np.ndarray
    lin_accel: np.ndarray
    ang_accel: np.ndarray
    times: np.ndarray
    history: dict[str, np.ndarray] = field(default_factory=dict)
    double_cover: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def history_names(self) -> list[str]:
        return sorted(self.history)


def compute_sample_features(samples: Sequence[dict], history_k: int = 4) -> SampleFeatures:
    """Camera-only per-sample features, all computed from the past.

    A truncation of ``samples`` at any index leaves the values up to that index
    unchanged, which is what makes the leakage test possible.
    """
    count = len(samples)
    position = np.array([s["cam_position"] for s in samples], dtype=np.float64)
    quaternions = np.array([s["cam_quaternion"] for s in samples], dtype=np.float64)
    matrix = np.array([quaternion_to_matrix(s["cam_quaternion"]) for s in samples])
    velocity = np.array([s["cam_velocity"] for s in samples], dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=1)

    times = np.array([float(s["session_time"]) for s in samples], dtype=np.float64)
    delta_t = np.diff(times) if count > 1 else np.zeros(0)

    # Angular velocity is recomputed from the orientations rather than taken from
    # the recorded field, which is wrong wherever the quaternion difference lands on
    # the -q representation. See angular_velocity_from_quaternions.
    angular, double_cover = angular_velocity_from_quaternions(quaternions, times)
    local_angular = np.einsum("nij,nj->ni", np.transpose(matrix, (0, 2, 1)), angular)

    # Backward finite differences. Index 0 has no predecessor and stays zero; the
    # accompanying *_valid mask records that rather than pretending it is a value.
    lin_accel = np.zeros_like(velocity)
    ang_accel = np.zeros(count, dtype=np.float64)
    if count > 1:
        safe = np.where(delta_t > 0.0, delta_t, np.nan)
        lin_accel[1:] = np.nan_to_num((velocity[1:] - velocity[:-1]) / safe[:, None])
        ang_accel[1:] = np.nan_to_num(
            np.linalg.norm(angular[1:] - angular[:-1], axis=1) / safe)
    lin_accel_mag = np.linalg.norm(lin_accel, axis=1)

    # Pose history: the k-th most recent interval, i.e. from t-k to t-k+1.
    history: dict[str, np.ndarray] = {}
    for k in range(1, history_k + 1):
        dpos = np.zeros(count, dtype=np.float64)
        dyaw = np.zeros(count, dtype=np.float64)
        dpitch = np.zeros(count, dtype=np.float64)
        hspeed = np.zeros(count, dtype=np.float64)
        valid = np.zeros(count, dtype=np.float64)
        for i in range(k, count):
            j = i - k                      # interval [j, j+1]
            dpos[i] = float(np.linalg.norm(position[j + 1] - position[j]))
            dyaw[i] = float(local_angular[j + 1][1] - local_angular[j][1])
            dpitch[i] = float(local_angular[j + 1][0] - local_angular[j][0])
            hspeed[i] = float(speed[j + 1])
            valid[i] = 1.0
        history[f"hist_dpos_{k}"] = dpos
        history[f"hist_dyaw_{k}"] = dyaw
        history[f"hist_dpitch_{k}"] = dpitch
        history[f"hist_speed_{k}"] = hspeed
        history[f"hist_valid_{k}"] = valid

    return SampleFeatures(count=count, position=position, matrix=matrix, velocity=velocity,
                          angular_velocity=angular, local_angular_velocity=local_angular,
                          speed=speed, lin_accel=lin_accel, ang_accel=ang_accel,
                          times=times, history=history, double_cover=double_cover)


# --- object catalogue ---------------------------------------------------------

@dataclass
class ObjectCatalogue:
    """Static, seed-derived object facts. Independent of any trajectory."""

    bounds: dict[str, dict]
    centre: dict[str, np.ndarray]
    radius: dict[str, float]
    category: dict[str, str]
    resolution_eligible: dict[str, bool]
    current_resolution: dict[str, int]
    available_resolutions: dict[str, list[int]]
    texture_ids: dict[str, list[str]]
    has_variants: dict[str, bool]

    @classmethod
    def from_manifest(cls, path: str | Path) -> "ObjectCatalogue":
        import json
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        bounds, centre, radius, category = {}, {}, {}, {}
        eligible, current, available, textures, variants = {}, {}, {}, {}, {}
        for entry in manifest["objects"]:
            oid = entry["object_id"]
            box = entry["world_bounds"]
            bounds[oid] = box
            centre[oid] = aabb_centre(box)
            radius[oid] = aabb_radius(box)
            category[oid] = entry["category"]
            eligible[oid] = bool(entry.get("resolution_eligible",
                                           entry.get("has_variants", False)))
            available[oid] = list(entry.get("available_resolutions") or [])
            current[oid] = max(available[oid]) if available[oid] else 0
            textures[oid] = list(entry.get("texture_ids") or [])
            variants[oid] = bool(entry.get("has_variants", False))
        return cls(bounds=bounds, centre=centre, radius=radius, category=category,
                   resolution_eligible=eligible, current_resolution=current,
                   available_resolutions=available, texture_ids=textures,
                   has_variants=variants)


# --- row-level assembly -------------------------------------------------------

def geometry_features(catalogue: ObjectCatalogue, sample_features: SampleFeatures,
                      sample_index: int, object_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Camera-frame geometry for every object of one sample, vectorised."""
    matrix = sample_features.matrix[sample_index]
    camera_position = sample_features.position[sample_index]
    centres = np.array([catalogue.centre[oid] for oid in object_ids], dtype=np.float64)
    local = world_to_camera_batch(centres, camera_position, matrix)

    forward = local[:, 2]                      # negative in front of the camera
    horizontal = np.hypot(local[:, 0], local[:, 2])
    azimuth = np.degrees(np.arctan2(local[:, 0], -local[:, 2]))
    elevation = np.degrees(np.arctan2(local[:, 1], horizontal))
    distance = np.array([point_to_aabb_distance(camera_position, catalogue.bounds[oid])
                         for oid in object_ids], dtype=np.float64)
    radius = np.array([catalogue.radius[oid] for oid in object_ids], dtype=np.float64)

    # Bearing and closing rates need the previous sample: causal, and masked at i=0.
    bearing_rate = np.zeros(len(object_ids), dtype=np.float64)
    closing_speed = np.zeros(len(object_ids), dtype=np.float64)
    rate_valid = 0.0
    if sample_index >= 1:
        previous_matrix = sample_features.matrix[sample_index - 1]
        previous_position = sample_features.position[sample_index - 1]
        previous_local = world_to_camera_batch(centres, previous_position, previous_matrix)
        previous_azimuth = np.degrees(np.arctan2(previous_local[:, 0], -previous_local[:, 2]))
        dt = _sample_delta(sample_features, sample_index)
        if dt > 0.0:
            # Azimuth is reported in [-180, 180], so a raw difference jumps by 360
            # when the bearing crosses the wrap point. Wrap the delta into [-180, 180]
            # or a crossing looks like a 3600 deg/s rotation.
            wrapped = (azimuth - previous_azimuth + 180.0) % 360.0 - 180.0
            bearing_rate = wrapped / dt
            previous_distance = np.array(
                [point_to_aabb_distance(previous_position, catalogue.bounds[oid])
                 for oid in object_ids], dtype=np.float64)
            closing_speed = (distance - previous_distance) / dt
            rate_valid = 1.0

    return {
        "obj_cam_x": local[:, 0],
        "obj_cam_y": local[:, 1],
        "obj_cam_z": local[:, 2],
        "obj_cam_radius": radius,
        "distance": distance,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "bearing_rate_deg_s": bearing_rate,
        "bearing_rate_valid": np.full(len(object_ids), rate_valid),
        "closing_speed_m_s": closing_speed,
        "closing_speed_valid": np.full(len(object_ids), rate_valid),
        "forward_component": forward,
    }


def _sample_delta(sample_features: SampleFeatures, sample_index: int) -> float:
    """Seconds between ``sample_index - 1`` and ``sample_index``.

    Recovered from the stored session times rather than assumed, so a change to the
    sampling interval cannot silently corrupt finite differences.
    """
    return float(sample_features.times[sample_index] - sample_features.times[sample_index - 1])


def camera_feature_columns(sample_features: SampleFeatures, sample_index: int) -> dict[str, float]:
    """Scalar camera features for one sample."""
    matrix = sample_features.matrix[sample_index]
    velocity = sample_features.velocity[sample_index]
    local_velocity = matrix.T @ velocity
    local_angular = sample_features.local_angular_velocity[sample_index]
    accel = sample_features.lin_accel[sample_index]
    local_accel = matrix.T @ accel
    return {
        "cam_speed": float(sample_features.speed[sample_index]),
        "cam_vel_cam_x": float(local_velocity[0]),
        "cam_vel_cam_y": float(local_velocity[1]),
        "cam_vel_cam_z": float(local_velocity[2]),
        "cam_lin_accel_mag": float(np.linalg.norm(accel)),
        "cam_lin_accel_valid": 1.0 if sample_index >= 1 else 0.0,
        "cam_accel_cam_x": float(local_accel[0]),
        "cam_accel_cam_y": float(local_accel[1]),
        "cam_accel_cam_z": float(local_accel[2]),
        "cam_yaw_rate_deg_s": math.degrees(float(local_angular[1])),
        "cam_pitch_rate_deg_s": math.degrees(float(local_angular[0])),
        "cam_roll_rate_deg_s": math.degrees(float(local_angular[2])),
        "cam_ang_speed_deg_s": math.degrees(float(np.linalg.norm(
            sample_features.angular_velocity[sample_index]))),
        "cam_ang_accel_deg_s2": math.degrees(float(sample_features.ang_accel[sample_index])),
        "cam_ang_accel_valid": 1.0 if sample_index >= 1 else 0.0,
    }


def object_feature_columns(catalogue: ObjectCatalogue, object_id: str) -> dict[str, float]:
    """Static, non-geometric object facts (available at time t by definition)."""
    available = catalogue.available_resolutions[object_id]
    current = catalogue.current_resolution[object_id]
    best = max(available) if available else 0
    columns = {
        "current_resolution": float(current),
        "n_available_resolutions": float(len(available)),
        "max_available_resolution": float(best),
        "has_variants": 1.0 if catalogue.has_variants[object_id] else 0.0,
        "resolution_ratio": float(current) / float(best) if best else 0.0,
    }
    for name in CATEGORIES:
        columns[f"cat_{name}"] = 1.0 if catalogue.category[object_id] == name else 0.0
    return columns


def screen_state_columns(row: dict) -> dict[str, float]:
    """Current on-screen state. Read from the label row, all of it available at t."""
    occlusion = row.get("occlusion_visible")
    coverage = float(row.get("coverage") or 0.0)
    required = float(row.get("required_resolution") or 0.0)
    return {
        "coverage": coverage,
        "log_coverage": math.log1p(coverage * 1e6),
        "frustum_visible": 1.0 if row.get("frustum_visible") else 0.0,
        "occlusion_true": 1.0 if occlusion is True else 0.0,
        "occlusion_false": 1.0 if occlusion is False else 0.0,
        "occlusion_unknown": 1.0 if occlusion is None else 0.0,
        "required_resolution": required,
        "log_required_resolution": math.log2(required) if required > 0 else 0.0,
    }


def assemble_matrix(rows: list[dict], allow: Sequence[str]) -> np.ndarray:
    """Build the feature matrix from the allow-list.

    Raises if the allow-list names a feature that was not produced, so a typo in the
    config fails loudly instead of yielding a silent column of zeros.
    """
    if not rows:
        return np.zeros((0, len(allow)), dtype=np.float32)
    missing = sorted({name for name in allow if name not in rows[0]})
    if missing:
        raise KeyError(f"allow-list names features that were not produced: {missing}")
    return np.array([[float(row[name]) for name in allow] for row in rows], dtype=np.float32)
