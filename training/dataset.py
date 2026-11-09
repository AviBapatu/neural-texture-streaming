"""Assembly of the model-ready feature table, targets and masks.

The table keeps features, targets, masks and bookkeeping metadata in separate
arrays. Only the feature block is ever handed to a model; targets and masks are
read by the evaluation harness.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import features as F
from .splits import load_split_manifest

# Occlusion target encoding. -1 is "unknown" and is never treated as negative.
OCCLUSION_FALSE = 0
OCCLUSION_TRUE = 1
OCCLUSION_UNKNOWN = -1


def horizon_suffix(horizon: float) -> str:
    """Field-name suffix for a horizon: 0.5 -> '0_5', 1.0 -> '1_0'.

    Mirrors ``tools/label_future_demand.horizon_suffix``, which is the canonical
    definition. training/tests/test_features.py asserts the two agree so the format
    cannot drift apart silently.
    """
    return f"{horizon:.1f}".replace(".", "_")


def _validity_column(name: str) -> str | None:
    """The mask column that says whether ``name`` is meaningful for a row."""
    if name.startswith(("hist_dpos_", "hist_dyaw_", "hist_dpitch_", "hist_speed_")):
        return "hist_valid_" + name.rsplit("_", 1)[1]
    if name == "bearing_rate_deg_s":
        return "bearing_rate_valid"
    if name == "closing_speed_m_s":
        return "closing_speed_valid"
    if name.startswith("cam_accel_") or name == "cam_lin_accel_mag":
        return "cam_lin_accel_valid"
    if name == "cam_ang_accel_deg_s2":
        return "cam_ang_accel_valid"
    return None


@dataclass
class Normaliser:
    """Per-feature mean/std, fit on training rows of the active fold only."""

    names: list[str]
    mean: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    fit_rows: int

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        return ((np.asarray(matrix, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def as_dict(self) -> dict:
        return {
            "names": self.names,
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "counts": [int(v) for v in self.counts],
            "fit_rows": int(self.fit_rows),
        }


@dataclass
class FeatureTable:
    """Features, targets, masks and metadata for one dataset run."""

    names: list[str]
    X: np.ndarray                      # (n, d) float32
    trajectory: np.ndarray             # (n,) int index into `trajectories`
    trajectories: list[str]
    sample_index: np.ndarray           # (n,) int
    object_id: list[str]
    category: list[str]
    split: list[str]
    distance: np.ndarray
    coverage: np.ndarray
    frustum_visible: np.ndarray        # (n,) bool
    occlusion_state: np.ndarray        # (n,) int8: 1 true, 0 false, -1 unknown
    resolution_eligible: np.ndarray    # (n,) bool
    # targets, shape (n, horizons)
    y_frustum: np.ndarray              # bool
    y_occlusion: np.ndarray            # int8, -1 unknown
    y_resolution: np.ndarray           # float32, nan where not applicable
    horizons: list[float]

    def __len__(self) -> int:
        return self.X.shape[0]

    def horizon_column(self, horizon: float) -> int:
        return self.horizons.index(horizon)

    def subset(self, mask: np.ndarray) -> "FeatureTable":
        mask = np.asarray(mask, dtype=bool)
        return FeatureTable(
            names=self.names, X=self.X[mask], trajectory=self.trajectory[mask],
            trajectories=self.trajectories, sample_index=self.sample_index[mask],
            object_id=[oid for oid, keep in zip(self.object_id, mask) if keep],
            category=[c for c, keep in zip(self.category, mask) if keep],
            split=[s for s, keep in zip(self.split, mask) if keep],
            distance=self.distance[mask], coverage=self.coverage[mask],
            frustum_visible=self.frustum_visible[mask],
            occlusion_state=self.occlusion_state[mask],
            resolution_eligible=self.resolution_eligible[mask],
            y_frustum=self.y_frustum[mask], y_occlusion=self.y_occlusion[mask],
            y_resolution=self.y_resolution[mask], horizons=self.horizons,
        )


def load_labels(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def group_by_sample(rows: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_index"]), []).append(row)
    for rows_at_sample in grouped.values():
        rows_at_sample.sort(key=lambda r: r["object_id"])
    return grouped


def build_trajectory_rows(samples: Sequence[dict], labels: Sequence[dict],
                          catalogue: F.ObjectCatalogue, history_k: int,
                          allow: Sequence[str],
                          texture_ids: Sequence[str] | None = None,
                          include_texture_identity: bool = False
                          ) -> tuple[list[dict], list[dict]]:
    """Feature dicts and metadata for one trajectory.

    Only ``samples`` up to each row's ``sample_index`` are read, so passing a
    truncated ``samples`` list cannot change the rows it still covers. That is the
    property the leakage test exercises.
    """
    sample_features = F.compute_sample_features(samples, history_k)
    grouped = group_by_sample(labels)
    feature_rows: list[dict] = []
    meta_rows: list[dict] = []
    texture_ids = list(texture_ids or [])

    for sample_index in sorted(grouped):
        rows = grouped[sample_index]
        ids = [row["object_id"] for row in rows]
        geometry = F.geometry_features(catalogue, sample_features, sample_index, ids)
        camera = F.camera_feature_columns(sample_features, sample_index)

        for position, row in enumerate(rows):
            values: dict[str, float] = dict(camera)
            for name, array in geometry.items():
                values[name] = float(array[position])
            for name, array in sample_features.history.items():
                values[name] = float(array[sample_index])
            values.update(F.screen_state_columns(row))
            values.update(F.object_feature_columns(catalogue, row["object_id"]))
            if include_texture_identity:
                for texture_id in texture_ids:
                    values[f"tex_{texture_id}"] = 0.0
                for texture_id in catalogue.texture_ids[row["object_id"]]:
                    key = f"tex_{texture_id}"
                    if key in values:
                        values[key] = 1.0
            feature_rows.append(values)
            meta_rows.append(row)

    return feature_rows, meta_rows


def build_table(dataset_dir: str | Path, config: dict) -> FeatureTable:
    """Load a processed dataset run and build the full feature table."""
    directory = Path(dataset_dir)
    allow = list(config["features"]["allow"])
    history_k = int(config["history_k"])
    include_texture_identity = bool(config["features"].get("include_texture_identity", False))

    catalogue = F.ObjectCatalogue.from_manifest(config["paths"]["object_index"])
    texture_index = json.loads(Path(config["paths"]["texture_index"]).read_text(encoding="utf-8"))
    texture_ids = sorted(texture_index["textures"])

    horizons = [float(h) for h in config["horizons"]]
    split_assignment: dict[str, str] = {}
    split_path = Path(config["paths"]["phase2_5_split"])
    if split_path.is_file():
        split_assignment = load_split_manifest(split_path)

    collected: list[dict] = []
    meta: list[dict] = []
    trajectories: list[str] = []
    for label_path in sorted(directory.glob("*.labels.jsonl")):
        trajectory_id = label_path.name.replace(".labels.jsonl", "")
        samples = load_labels(directory / f"{trajectory_id}.samples.jsonl")
        labels = load_labels(label_path)
        rows, rows_meta = build_trajectory_rows(
            samples, labels, catalogue, history_k, allow,
            texture_ids=texture_ids, include_texture_identity=include_texture_identity)
        index = len(trajectories)
        trajectories.append(trajectory_id)
        for row_meta in rows_meta:
            row_meta["_trajectory_index"] = index
        collected.extend(rows)
        meta.extend(rows_meta)

    names = list(allow) + ([f"tex_{tid}" for tid in texture_ids] if include_texture_identity else [])
    X = F.assemble_matrix(collected, names)

    count = len(meta)
    y_frustum = np.zeros((count, len(horizons)), dtype=bool)
    y_occlusion = np.full((count, len(horizons)), OCCLUSION_UNKNOWN, dtype=np.int8)
    y_resolution = np.full((count, len(horizons)), np.nan, dtype=np.float32)
    occlusion_state = np.full(count, OCCLUSION_UNKNOWN, dtype=np.int8)
    eligible = np.zeros(count, dtype=bool)
    available = np.zeros((count, len(horizons)), dtype=bool)

    frustum_key = config["targets"]["frustum"]
    occlusion_key = config["targets"]["occlusion"]
    resolution_key = config["targets"]["resolution"]

    for index, row in enumerate(meta):
        state = row.get("occlusion_visible")
        occlusion_state[index] = (OCCLUSION_UNKNOWN if state is None
                                  else (OCCLUSION_TRUE if state else OCCLUSION_FALSE))
        eligible[index] = bool(row.get("resolution_eligible", False))
        for column, horizon in enumerate(horizons):
            suffix = horizon_suffix(horizon)
            y_frustum[index, column] = bool(row[frustum_key.format(h=suffix)])
            target = row[occlusion_key.format(h=suffix)]
            y_occlusion[index, column] = (OCCLUSION_UNKNOWN if target is None
                                          else (OCCLUSION_TRUE if target else OCCLUSION_FALSE))
            available[index, column] = row.get(resolution_key.format(h=suffix)) is not None
            if eligible[index] and available[index, column]:
                y_resolution[index, column] = float(row[resolution_key.format(h=suffix)])

    return FeatureTable(
        names=names, X=X,
        trajectory=np.array([r["_trajectory_index"] for r in meta], dtype=np.int32),
        trajectories=trajectories,
        sample_index=np.array([int(r["sample_index"]) for r in meta], dtype=np.int32),
        object_id=[r["object_id"] for r in meta],
        category=[r["category"] for r in meta],
        split=[split_assignment.get(t, "unassigned") for t in
               (trajectories[r["_trajectory_index"]] for r in meta)],
        distance=np.array([float(r["distance"]) for r in meta], dtype=np.float32),
        coverage=np.array([float(r["coverage"]) for r in meta], dtype=np.float32),
        frustum_visible=np.array([bool(r["frustum_visible"]) for r in meta]),
        occlusion_state=occlusion_state,
        resolution_eligible=eligible,
        y_frustum=y_frustum, y_occlusion=y_occlusion, y_resolution=y_resolution,
        horizons=horizons,
    )


def fit_normaliser(table: FeatureTable, mask: np.ndarray,
                   population_std: bool = True) -> Normaliser:
    """Fit per-feature statistics on the rows selected by ``mask``.

    Masked-out values (a history slot before the start of a trajectory, or an
    acceleration at t=0) are excluded rather than averaged in as zeros, which is
    what the mask columns are for.
    """
    mask = np.asarray(mask, dtype=bool)
    subset = table.X[mask]
    if subset.shape[0] == 0:
        raise ValueError("cannot fit a normaliser on zero rows")

    mean = np.zeros(len(table.names))
    std = np.ones(len(table.names))
    counts = np.zeros(len(table.names), dtype=np.int64)
    for column, name in enumerate(table.names):
        column_values = subset[:, column]
        validity_name = _validity_column(name)
        if validity_name is not None and validity_name in table.names:
            validity = subset[:, table.names.index(validity_name)] > 0.5
            if validity.any():
                column_values = column_values[validity]
        counts[column] = column_values.size
        if column_values.size == 0:
            continue
        mean[column] = float(np.mean(column_values))
        deviation = float(np.std(column_values, ddof=0 if population_std else 1))
        std[column] = deviation if deviation > 1e-12 else 1.0

    return Normaliser(names=list(table.names), mean=mean, std=std,
                      counts=counts, fit_rows=int(subset.shape[0]))


def save_normaliser(normaliser: Normaliser, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normaliser.as_dict(), indent=2) + "\n", encoding="utf-8")


def load_normaliser(path: str | Path) -> Normaliser:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Normaliser(names=payload["names"],
                      mean=np.array(payload["mean"], dtype=np.float64),
                      std=np.array(payload["std"], dtype=np.float64),
                      counts=np.array(payload["counts"], dtype=np.int64),
                      fit_rows=int(payload["fit_rows"]))
