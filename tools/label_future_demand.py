#!/usr/bin/env python3
"""Turn raw telemetry into model-ready rows with future texture-demand labels.

For every sample the pipeline emits one row per *candidate* object - every object
whose bounding box is within `--candidate-distance` metres of the camera. For each
candidate and each prediction horizon it records whether the object is visible in
that window, when it first appears, how large it gets, and the texture resolution it
needs.

All labels are derived from observations at or after the sample they are attached
to; for an object that is not currently on screen every value comes purely from
future observations, so nothing leaks backwards.

Outputs, per trajectory, under `dataset/processed/<run>/`:

  <id>.samples.jsonl  one row per sample: camera pose, velocity, fov, viewport
  <id>.labels.jsonl   one row per (sample_index, object_id): object state + labels
  <id>.csv            optional convenience export of the labels (--csv)

Splitting the two keeps the per-sample camera state out of ~200k duplicated rows;
join them on (trajectory_id, sample_index).

Usage:
    python3 tools/label_future_demand.py --run run_full
    python3 tools/label_future_demand.py --run run_full --horizons 0.5,1.0,2.0
    python3 tools/label_future_demand.py --run run_full --csv
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_HORIZONS = (0.5, 1.0)
DEFAULT_CANDIDATE_DISTANCE = 25.0
SCHEMA_VERSION = 1
RAW_ROOT = "dataset/raw"
PROCESSED_ROOT = "dataset/processed"
OBJECT_INDEX = "dataset/manifests/object_index.json"
TEXTURE_INDEX = "dataset/manifests/texture_index.json"
SPLITS = "dataset/manifests/splits.json"


def horizon_suffix(horizon: float) -> str:
    """0.5 -> '0_5', 1.0 -> '1_0', 2.0 -> '2_0'."""
    return f"{horizon:.1f}".replace(".", "_")


# --- geometry ----------------------------------------------------------------

def point_to_aabb_distance(point: list[float], bounds: dict) -> float:
    """Distance from a point to an axis-aligned box (0 when inside)."""
    position, size = bounds["position"], bounds["size"]
    deltas = []
    for axis in range(3):
        low, high = position[axis], position[axis] + size[axis]
        deltas.append(max(low - point[axis], 0.0, point[axis] - high))
    return math.sqrt(sum(delta * delta for delta in deltas))


# --- input loading -----------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def object_lookup(object_index: dict, texture_index: dict) -> dict[str, dict]:
    """Static per-object facts resolved from the manifests.

    Also decides resolution-label eligibility: only texture identities that actually
    have generated variants can be scored against a resolution target. Visibility
    targets are unaffected - they apply to every object regardless of texture.
    """
    textures = texture_index.get("textures", {})
    lookup: dict[str, dict] = {}
    for entry in object_index["objects"]:
        available: set[int] = set()
        variant_ids: list[str] = []
        variant_resolutions: set[int] = set()
        for texture_id in entry.get("texture_ids", []):
            texture = textures.get(texture_id)
            if texture is None:
                continue
            available.update(texture["available_resolutions"])
            if texture.get("has_variants"):
                variant_ids.append(texture_id)
                variant_resolutions.update(texture["available_resolutions"])
        lookup[entry["object_id"]] = {
            "object_id": entry["object_id"],
            "category": entry["category"],
            "world_bounds": entry["world_bounds"],
            "texture_ids": entry.get("texture_ids", []),
            "available_resolutions": sorted(available),
            "current_resolution": max(available) if available else 0,
            # Resolution-prediction eligibility mask inputs.
            "resolution_eligible": bool(variant_ids),
            "variant_texture_ids": sorted(variant_ids),
            "resolution_variants": sorted(variant_resolutions),
        }
    return lookup


def load_splits(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    assignment: dict[str, str] = {}
    for split, entries in load_json(path).get("splits", {}).items():
        for entry in entries:
            assignment[entry["trajectory_id"]] = split
    return assignment


# --- visible-event index -----------------------------------------------------

def build_visibility_index(samples: list[dict]) -> dict[str, dict]:
    """Per object: the sample indices where it is on screen, and the data there.

    Lets a look-ahead query be an O(log n) bisect instead of a linear scan.
    """
    index: dict[str, dict] = {}
    for sample_index, sample in enumerate(samples):
        session_time = float(sample["session_time"])
        for observation in sample.get("observations", []):
            if not observation.get("frustum_visible"):
                continue
            entry = index.setdefault(observation["object_id"], {
                "sample_index": [], "session_time": [], "coverage": [],
                "required_resolution": [], "occlusion_visible": [],
            })
            entry["sample_index"].append(sample_index)
            entry["session_time"].append(session_time)
            entry["coverage"].append(float(observation.get("coverage", 0.0)))
            entry["required_resolution"].append(int(observation.get("required_resolution", 0)))
            entry["occlusion_visible"].append(observation.get("occlusion_visible"))
    return index


EMPTY_ENTRY = {"sample_index": [], "session_time": [], "coverage": [],
               "required_resolution": [], "occlusion_visible": []}


def look_ahead(entry: dict, sample_index: int, sample_time: float, horizon: float,
               current_resolution: int, current: dict | None) -> tuple[dict, dict]:
    """Compute both future targets for one object over [now, now + horizon].

    Returns `(frustum, occlusion)`.

    **Frustum demand** (`frustum`) is a plain boolean target: will the object's
    bounding box enter the camera frustum? It is identical to the Phase 2
    `visible_in_*` label, which is kept alongside it for compatibility.

    **Occlusion-aware visible demand** (`occlusion`) is tri-state, because "will it
    actually become visible" cannot always be answered:

      * `True`  - at least one observation in the window was frustum-visible *and*
                  the raycast pass confirmed it was not blocked
      * `False` - the object was never frustum-visible in the window, or it was
                  frustum-visible only at moments the raycast confirmed were blocked
      * `None`  - it was frustum-visible but every such moment has an unknown
                  occlusion state, so the question is genuinely unanswered

    Leaving the frustum is *not* a "maybe": being outside the frustum is sufficient
    to know an object is not visible, so an empty window is `False` and never `None`.
    Unknown occlusion is therefore reported as unknown and never silently counted as
    invisible.
    """
    times = entry["session_time"]
    indices = entry["sample_index"]
    window: list[tuple[float, float, int, object]] = []
    # An observation entry exists for every object within the observation radius,
    # including ones outside the frustum, so presence is not the same as visibility.
    if current is not None and current.get("frustum_visible"):
        window.append((sample_time, float(current["coverage"]),
                       int(current["required_resolution"]), current["occlusion_visible"]))
    position = bisect.bisect_right(indices, sample_index)
    while position < len(indices) and times[position] <= sample_time + horizon + 1e-9:
        window.append((times[position], entry["coverage"][position],
                       entry["required_resolution"][position],
                       entry["occlusion_visible"][position]))
        position += 1

    if window:
        best = max(window, key=lambda item: item[1])
        frustum = {
            "visible_in": True,
            "first_visible_time": round(window[0][0] - sample_time, 3),
            "max_coverage": round(max(item[1] for item in window), 6),
            "required_resolution": window[0][2],
            "required_resolution_max": best[2],
        }
    else:
        frustum = {"visible_in": False, "first_visible_time": None, "max_coverage": 0.0,
                   "required_resolution": None, "required_resolution_max": None}

    confirmed = [item for item in window if item[3] is True]
    unknown = [item for item in window if item[3] is None]
    if confirmed:
        occlusion = {
            "visible_in": True,
            "first_visible_time": round(confirmed[0][0] - sample_time, 3),
            "max_coverage": round(max(item[1] for item in confirmed), 6),
            "required_resolution": confirmed[0][2],
            "unknown_in": bool(unknown),
        }
    elif unknown:
        occlusion = {"visible_in": None, "first_visible_time": None, "max_coverage": 0.0,
                     "required_resolution": None, "unknown_in": True}
    else:
        occlusion = {"visible_in": False, "first_visible_time": None, "max_coverage": 0.0,
                     "required_resolution": None, "unknown_in": False}
    return frustum, occlusion


# --- labelling ---------------------------------------------------------------

def sample_row(sample: dict, sample_index: int, run_id: str, trajectory_id: str,
               trajectory_kind: str, split: str) -> dict:
    """Per-sample camera state, written once instead of once per candidate."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "trajectory_kind": trajectory_kind,
        "split": split,
        "sample_index": sample_index,
        "physics_tick": sample.get("physics_tick"),
        "timestamp": sample["timestamp"],
        "session_time": round(float(sample["session_time"]), 3),
        "environment_seed": sample["environment_seed"],
        "cam_position": sample["cam_position"],
        "cam_quaternion": sample["cam_quaternion"],
        "cam_direction": sample["cam_direction"],
        "cam_velocity": sample["cam_velocity"],
        "cam_angular_velocity": sample.get("cam_angular_velocity"),
        "cam_fov": sample["cam_fov"],
        "viewport": sample["viewport"],
    }


def label_rows(samples: list[dict], objects: dict[str, dict], horizons: list[float],
               candidate_distance: float, split: str, run_id: str,
               trajectory_id: str, keep_legacy_labels: bool = True) -> tuple[list[dict], list[dict]]:
    visibility = build_visibility_index(samples)
    samples_out: list[dict] = []
    labels_out: list[dict] = []

    for sample_index, sample in enumerate(samples):
        session_time = float(sample["session_time"])
        camera_position = sample["cam_position"]
        samples_out.append(sample_row(sample, sample_index, run_id, trajectory_id,
                                      trajectory_id.split("_", 1)[-1], split))

        current = {observation["object_id"]: observation
                   for observation in sample.get("observations", [])}
        candidates = []
        for object_id, info in objects.items():
            distance = point_to_aabb_distance(camera_position, info["world_bounds"])
            if distance <= candidate_distance:
                candidates.append((distance, object_id))
        candidates.sort()

        for distance, object_id in candidates:
            info = objects[object_id]
            observation = current.get(object_id)
            row = {
                "schema_version": SCHEMA_VERSION,
                "trajectory_id": trajectory_id,
                "split": split,
                "sample_index": sample_index,
                "object_id": object_id,
                "category": info["category"],
                "distance": round(distance, 3),
                "frustum_visible": bool(observation["frustum_visible"]) if observation else False,
                "approx_visible": bool(observation["approx_visible"]) if observation else False,
                "occlusion_visible": observation["occlusion_visible"] if observation else None,
                "coverage": float(observation["coverage"]) if observation else 0.0,
                "required_resolution": int(observation["required_resolution"]) if observation else 0,
                "current_resolution": info["current_resolution"],
                # Eligibility mask for resolution training. Visibility targets apply to
                # every object; only these rows may contribute to a resolution loss.
                "resolution_eligible": info["resolution_eligible"],
            }
            if info["resolution_eligible"]:
                row["resolution_variants"] = info["resolution_variants"]
                row["variant_texture_ids"] = info["variant_texture_ids"]
            entry = visibility.get(object_id, EMPTY_ENTRY)
            for horizon in horizons:
                suffix = horizon_suffix(horizon)
                frustum, occlusion = look_ahead(entry, sample_index, session_time, horizon,
                                                info["current_resolution"], observation)

                if keep_legacy_labels:
                    # Phase 2 fields, unchanged. `visible_in_*` is the frustum target.
                    row[f"visible_in_{suffix}s"] = frustum["visible_in"]
                    row[f"first_visible_time_{suffix}s"] = frustum["first_visible_time"]
                    row[f"max_coverage_{suffix}s"] = frustum["max_coverage"]
                    row[f"required_resolution_{suffix}s"] = frustum["required_resolution"]
                    row[f"required_resolution_max_{suffix}s"] = frustum["required_resolution_max"]
                    row[f"occlusion_visible_{suffix}s"] = (
                        observation["occlusion_visible"] if observation else None)
                    row[f"resolution_insufficient_{suffix}s"] = bool(
                        frustum["required_resolution"] is not None
                        and frustum["required_resolution"] > info["current_resolution"])

                # Target 1: frustum demand.
                row[f"frustum_visible_in_{suffix}s"] = frustum["visible_in"]
                row[f"frustum_first_visible_time_{suffix}s"] = frustum["first_visible_time"]
                row[f"frustum_max_coverage_{suffix}s"] = frustum["max_coverage"]
                row[f"frustum_required_resolution_{suffix}s"] = frustum["required_resolution"]

                # Target 2: occlusion-aware visible demand, tri-state.
                row[f"occlusion_visible_in_{suffix}s"] = occlusion["visible_in"]
                row[f"occlusion_first_visible_time_{suffix}s"] = occlusion["first_visible_time"]
                row[f"occlusion_max_coverage_{suffix}s"] = occlusion["max_coverage"]
                row[f"occlusion_required_resolution_{suffix}s"] = occlusion["required_resolution"]
                row[f"occlusion_unknown_in_{suffix}s"] = occlusion["unknown_in"]
            labels_out.append(row)
    return samples_out, labels_out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


CSV_BASE_COLUMNS = [
    "trajectory_id", "split", "sample_index", "object_id", "category", "distance",
    "frustum_visible", "approx_visible", "occlusion_visible", "coverage",
    "required_resolution", "current_resolution", "resolution_eligible",
]

# Target 1 (frustum demand) and target 2 (occlusion-aware visible demand).
CSV_TARGET_SUFFIXES = [
    "visible_in_{h}s",
    "first_visible_time_{h}s",
    "max_coverage_{h}s",
    "required_resolution_{h}s",
    "occlusion_visible_in_{h}s",
    "occlusion_first_visible_time_{h}s",
    "occlusion_max_coverage_{h}s",
    "occlusion_required_resolution_{h}s",
    "occlusion_unknown_in_{h}s",
]


def csv_columns(horizons: list[float]) -> list[str]:
    columns = list(CSV_BASE_COLUMNS)
    for horizon in horizons:
        suffix = horizon_suffix(horizon)
        for template in CSV_TARGET_SUFFIXES:
            name = template.format(h=suffix)
            # Frustum target columns are prefixed so they read distinctly.
            columns.append(name if name.startswith("occlusion_") else f"frustum_{name}")
    return columns


def write_csv(path: Path, rows: list[dict], horizons: list[float]) -> None:
    columns = csv_columns(horizons)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if row.get(column) is None else row.get(column)
                             for column in columns])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run id under dataset/raw/<run>")
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
                        help="comma separated prediction horizons in seconds")
    parser.add_argument("--candidate-distance", type=float, default=DEFAULT_CANDIDATE_DISTANCE,
                        dest="candidate_distance",
                        help="maximum camera-to-bounding-box distance for a candidate object")
    parser.add_argument("--csv", action="store_true",
                        help="also write a flat CSV convenience export of the labels")
    parser.add_argument("--drop-legacy-labels", action="store_true",
                        help="omit the Phase 2 label columns (visible_in_*, occlusion_visible_*, "
                             "...); they are byte-identical to the new frustum target")
    arguments = parser.parse_args(argv)

    horizons = sorted(float(part) for part in arguments.horizons.split(",") if part.strip())
    if not horizons:
        print("no horizons given", file=sys.stderr)
        return 2

    raw_directory = Path(RAW_ROOT) / arguments.run
    if not raw_directory.is_dir():
        print(f"raw run not found: {raw_directory}", file=sys.stderr)
        return 2

    objects = object_lookup(load_json(Path(OBJECT_INDEX)), load_json(Path(TEXTURE_INDEX)))
    splits = load_splits(Path(SPLITS))

    output_directory = Path(PROCESSED_ROOT) / arguments.run
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"labelling run '{arguments.run}'")
    print(f"  horizons           : {horizons}")
    print(f"  candidate distance : {arguments.candidate_distance} m")
    print(f"  objects in index   : {len(objects)}")

    total_samples = 0
    total_rows = 0
    eligible_rows = 0
    tallies = {horizon_suffix(h): collections.Counter() for h in horizons}
    for path in sorted(raw_directory.glob("*.jsonl")):
        trajectory_id = path.stem
        samples = load_samples(path)
        sample_out, rows = label_rows(samples, objects, horizons, arguments.candidate_distance,
                                      splits.get(trajectory_id, "unassigned"), arguments.run,
                                      trajectory_id,
                                      keep_legacy_labels=not arguments.drop_legacy_labels)
        write_jsonl(output_directory / f"{trajectory_id}.samples.jsonl", sample_out)
        write_jsonl(output_directory / f"{trajectory_id}.labels.jsonl", rows)
        if arguments.csv:
            write_csv(output_directory / f"{trajectory_id}.csv", rows, horizons)
        total_samples += len(sample_out)
        total_rows += len(rows)
        eligible_rows += sum(1 for row in rows if row["resolution_eligible"])
        for row in rows:
            for horizon in horizons:
                suffix = horizon_suffix(horizon)
                counter = tallies[suffix]
                frustum_target = row[f"frustum_visible_in_{suffix}s"]
                occlusion_target = row[f"occlusion_visible_in_{suffix}s"]
                counter["frustum_true"] += int(bool(frustum_target))
                counter["frustum_new"] += int(bool(frustum_target) and not row["frustum_visible"])
                counter["occlusion_true"] += int(occlusion_target is True)
                counter["occlusion_false"] += int(occlusion_target is False)
                counter["occlusion_unknown"] += int(occlusion_target is None)
                counter["occlusion_new"] += int(occlusion_target is True and not row["frustum_visible"])
                counter["occlusion_disagrees_with_frustum"] += int(
                    (occlusion_target is False) and bool(frustum_target))
        print(f"  {trajectory_id:<26} {len(samples):>4} samples -> {len(rows):>7} label rows")

    print(f"\n{len(list(raw_directory.glob('*.jsonl')))} trajectories, "
          f"{total_samples} samples, {total_rows} label rows")
    print(f"  resolution-eligible rows: {eligible_rows} / {total_rows} "
          f"({100.0 * eligible_rows / max(1, total_rows):.1f}%)")
    for horizon in horizons:
        suffix = horizon_suffix(horizon)
        counter = tallies[suffix]
        print(f"  horizon {horizon}s: frustum {counter['frustum_true']} true "
              f"({counter['frustum_new']} newly), occlusion {counter['occlusion_true']} true / "
              f"{counter['occlusion_false']} false / {counter['occlusion_unknown']} unknown "
              f"({counter['occlusion_new']} newly)")
    print(f"-> {output_directory}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": arguments.run,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment_seed": object_index_seed(),
        "horizons_seconds": horizons,
        "candidate_distance_m": arguments.candidate_distance,
        "trajectory_count": len(list(raw_directory.glob("*.jsonl"))),
        "samples": total_samples,
        "label_rows": total_rows,
        "resolution_eligible_rows": eligible_rows,
        "keep_legacy_labels": not arguments.drop_legacy_labels,
        "targets": {
            "frustum_demand": "will the object's bounding box enter the camera frustum; "
                              "boolean, fields frustum_visible_in_<h>s",
            "occlusion_aware_visible_demand": "will the object actually become visible once "
                                              "occlusion is accounted for; TRI-STATE, fields "
                                              "occlusion_visible_in_<h>s, where null means the "
                                              "occlusion state was unknown and never that the "
                                              "object is invisible",
        },
        "resolution_eligibility": {
            "rule": "an object is eligible only if at least one of its texture identities has "
                    "generated resolution variants",
            "visibility_targets_apply_to_all_textures": True,
            "eligible_rows": eligible_rows,
            "ineligible_rows": total_rows - eligible_rows,
            "field": "resolution_eligible (per row); resolution_variants and "
                     "variant_texture_ids are present only for eligible rows",
        },
        "target_tallies": {suffix: dict(counter) for suffix, counter in tallies.items()},
        "raw_directory": f"{RAW_ROOT}/{arguments.run}",
        "processed_directory": f"{PROCESSED_ROOT}/{arguments.run}",
        "join_key": ["trajectory_id", "sample_index"],
        "label_semantics": "labels cover the window [now, now + horizon]; an object that is "
                           "not visible now but visible in the window has visible_in_* true "
                           "and first_visible_time_* > 0",
        "required_resolution_is_estimate": True,
    }
    manifest_path = Path("dataset/manifests/dataset_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(manifest, indent=2) + "\n"
    manifest_path.write_text(serialised, encoding="utf-8")
    run_path = Path("dataset/manifests/runs") / arguments.run / "dataset_manifest.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(serialised, encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"wrote {run_path}")
    return 0


def object_index_seed() -> int | None:
    path = Path(OBJECT_INDEX)
    return load_json(path).get("environment_seed") if path.is_file() else None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
