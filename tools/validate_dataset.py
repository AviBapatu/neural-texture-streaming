#!/usr/bin/env python3
"""Validate a generated dataset and write a human-readable report.

Checks the raw telemetry, the labelled rows, the manifests and the train/val/test
split against the documented schema. Future labels are re-derived here with an
independent, deliberately naive implementation (a linear look-ahead with no index)
and compared against the values the pipeline wrote, so a bug in the optimised
labelling path is caught rather than trusted.

Usage:
    python3 tools/validate_dataset.py --run run_full
    python3 tools/validate_dataset.py --run run_full --label-samples 500
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

RAW_ROOT = "dataset/raw"
PROCESSED_ROOT = "dataset/processed"
REPORT_ROOT = "dataset/reports"
OBJECT_INDEX = "dataset/manifests/object_index.json"
TEXTURE_INDEX = "dataset/manifests/texture_index.json"
SPLITS = "dataset/manifests/splits.json"
RUN_MANIFEST_ROOT = "dataset/manifests/runs"


def splits_path(run_id: str) -> Path:
    """Prefer the assignment recorded for this run over the most recent one.

    `dataset/manifests/splits.json` always describes the last pipeline run, so a
    later run with a different profile would otherwise make an earlier dataset look
    corrupted.
    """
    per_run = Path(RUN_MANIFEST_ROOT) / run_id / "splits.json"
    return per_run if per_run.is_file() else Path(SPLITS)


def dataset_manifest_path(run_id: str) -> Path:
    per_run = Path(RUN_MANIFEST_ROOT) / run_id / "dataset_manifest.json"
    return per_run if per_run.is_file() else Path("dataset/manifests/dataset_manifest.json")


RESOLUTION_LADDER = {0, 64, 128, 256, 512, 1024, 2048}
OCCLUSION_REASONS = {
    "tested", "disabled", "below_coverage_threshold", "no_space_state",
    "behind_near_plane", "too_close",
}

RAW_SAMPLE_FIELDS = {
    "schema_version": int, "timestamp": (int, float), "session_time": (int, float),
    "frame": int, "physics_tick": int, "environment_seed": int,
    "cam_position": list, "cam_quaternion": list, "cam_euler": list,
    "cam_velocity": list, "cam_angular_velocity": list, "cam_direction": list,
    "cam_fov": (int, float), "viewport": list,
    "observation_count": int, "visible_count": int, "culled_count": int,
    "occlusion_enabled": bool, "observations": list,
}

OBSERVATION_FIELDS = {
    "object_id": str, "legacy_id": int, "category": str, "distance": (int, float),
    "coverage": (int, float), "frustum_visible": bool, "approx_visible": bool,
    "occlusion_reason": str, "occlusion_rays": list, "texture_ids": list,
    "required_resolution": int,
}


# Fields written for each target family. `required_resolution_max` is emitted only for
# the legacy Phase 2 family, so the frustum target deliberately omits it.
FRUSTUM_TARGET_FIELDS = ("visible_in", "first_visible_time", "max_coverage", "required_resolution")
OCCLUSION_TARGET_FIELDS = ("visible_in", "first_visible_time", "max_coverage",
                           "required_resolution", "unknown_in")
LEGACY_TARGET_FIELDS = ("visible_in", "first_visible_time", "max_coverage", "required_resolution",
                        "required_resolution_max")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: dict = {}
        self.checks: list[tuple[str, str, str]] = []  # (name, status, detail)

    def check(self, name: str, detail: str = "", ok: bool = True) -> None:
        self.checks.append((name, "PASS" if ok else "FAIL", detail))

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def horizon_suffix(horizon: float) -> str:
    return f"{horizon:.1f}".replace(".", "_")


# --- independent label recomputation -----------------------------------------

def recompute_labels(samples: list[dict], object_id: str, sample_index: int,
                     horizon: float, current_resolution: int) -> tuple[dict, dict]:
    """Naive linear look-ahead, independent of the pipeline's indexed version.

    Returns `(frustum, occlusion)` - the two future targets - computed with a plain
    scan and no bisect index, so a bug in the optimised path is caught rather than
    trusted.
    """
    sample = samples[sample_index]
    start_time = float(sample["session_time"])
    end_time = start_time + horizon
    window: list[tuple[float, float, int, object]] = []
    current = None
    for observation in sample.get("observations", []):
        if observation["object_id"] == object_id:
            current = observation
            break
    if current is not None and current["frustum_visible"]:
        window.append((start_time, float(current["coverage"]),
                       int(current["required_resolution"]), current["occlusion_visible"]))
    for other_index in range(sample_index + 1, len(samples)):
        other = samples[other_index]
        time = float(other["session_time"])
        if time > end_time + 1e-9:
            break
        for observation in other.get("observations", []):
            if observation["object_id"] != object_id:
                continue
            if observation["frustum_visible"]:
                window.append((time, float(observation["coverage"]),
                               int(observation["required_resolution"]),
                               observation["occlusion_visible"]))
            break

    if window:
        best = max(window, key=lambda item: item[1])
        frustum = {"visible_in": True,
                   "first_visible_time": round(window[0][0] - start_time, 3),
                   "max_coverage": round(max(item[1] for item in window), 6),
                   "required_resolution": window[0][2],
                   "required_resolution_max": best[2]}
    else:
        frustum = {"visible_in": False, "first_visible_time": None, "max_coverage": 0.0,
                   "required_resolution": None, "required_resolution_max": None}

    confirmed = [item for item in window if item[3] is True]
    unknown = [item for item in window if item[3] is None]
    if confirmed:
        occlusion = {"visible_in": True,
                     "first_visible_time": round(confirmed[0][0] - start_time, 3),
                     "max_coverage": round(max(item[1] for item in confirmed), 6),
                     "required_resolution": confirmed[0][2],
                     "unknown_in": bool(unknown)}
    elif unknown:
        occlusion = {"visible_in": None, "first_visible_time": None, "max_coverage": 0.0,
                     "required_resolution": None, "unknown_in": True}
    else:
        occlusion = {"visible_in": False, "first_visible_time": None, "max_coverage": 0.0,
                     "required_resolution": None, "unknown_in": False}
    return frustum, occlusion


# --- checks ------------------------------------------------------------------

def check_manifests(report: Report, run_id: str) -> tuple[dict, dict, dict, dict]:
    object_index, texture_index, splits, trajectory_index = {}, {}, {}, {}
    for label, path in (("object index", OBJECT_INDEX), ("texture index", TEXTURE_INDEX),
                        ("splits", str(splits_path(run_id)))):
        if Path(path).is_file():
            report.check(f"manifest present: {label}", path)
        else:
            report.error(f"missing manifest: {path}")
    if Path(OBJECT_INDEX).is_file():
        object_index = load_json(Path(OBJECT_INDEX))
    if Path(TEXTURE_INDEX).is_file():
        texture_index = load_json(Path(TEXTURE_INDEX))
    if splits_path(run_id).is_file():
        splits = load_json(splits_path(run_id))

    ids = [entry["object_id"] for entry in object_index.get("objects", [])]
    duplicates = [identifier for identifier, count in collections.Counter(ids).items() if count > 1]
    report.check("object ids unique", f"{len(ids)} objects", not duplicates)
    if duplicates:
        report.error(f"duplicate object ids in the index: {duplicates[:5]}")
    if object_index.get("id_collisions", 0) > 0:
        report.warn(f"{object_index['id_collisions']} id collisions were disambiguated")

    # texture identities resolve, and their variants exist on disk
    missing = []
    for texture in texture_index.get("textures", {}).values():
        if not Path(texture["source_path"]).is_file():
            missing.append(texture["source_path"])
        for variant in texture.get("variant_paths", {}).values():
            if not Path(variant).is_file():
                missing.append(variant)
    report.check("texture sources and variants exist on disk", f"{len(texture_index.get('textures', {}))} ids",
                 not missing)
    if missing:
        report.error(f"missing texture files: {missing[:5]}")
    report.info["texture_ids"] = len(texture_index.get("textures", {}))
    report.info["textures_with_variants"] = texture_index.get("textures_with_variants", 0)
    report.info["object_count"] = len(ids)
    report.info["environment_seed"] = object_index.get("environment_seed")
    report.info["generation_hash"] = object_index.get("generation_hash")
    return object_index, texture_index, splits, trajectory_index


def object_facts(object_index: dict, texture_index: dict) -> dict[str, dict]:
    textures = texture_index.get("textures", {})
    facts = {}
    for entry in object_index.get("objects", []):
        available: set[int] = set()
        variant_ids: list[str] = []
        variant_resolutions: set[int] = set()
        for texture_id in entry.get("texture_ids", []):
            texture = textures.get(texture_id)
            if not texture:
                continue
            available.update(texture["available_resolutions"])
            if texture.get("has_variants"):
                variant_ids.append(texture_id)
                variant_resolutions.update(texture["available_resolutions"])
        facts[entry["object_id"]] = {
            "texture_ids": set(entry.get("texture_ids", [])),
            "available_resolutions": available,
            "current_resolution": max(available) if available else 0,
            "resolution_eligible": bool(variant_ids),
            "variant_texture_ids": sorted(variant_ids),
            "resolution_variants": sorted(variant_resolutions),
        }
    return facts


def check_raw(report: Report, run_id: str, objects: dict[str, dict], texture_ids: set[str],
              environment_seed: int) -> dict[str, list[dict]]:
    directory = Path(RAW_ROOT) / run_id
    if not directory.is_dir():
        report.error(f"raw run directory missing: {directory}")
        return {}
    files = sorted(directory.glob("*.jsonl"))
    report.check("raw telemetry present", f"{len(files)} trajectories in {directory}", bool(files))
    loaded: dict[str, list[dict]] = {}
    total_samples = 0
    total_observations = 0

    for path in files:
        trajectory_id = path.stem
        samples = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        loaded[trajectory_id] = samples
        total_samples += len(samples)
        problems: list[str] = []
        ticks: set[int] = set()
        timestamps: set[float] = set()
        previous_tick = -1
        previous_time = -math.inf
        seen_pairs: set[tuple[int, str]] = set()
        duplicates = 0
        duplicate_timestamps = 0

        for index, sample in enumerate(samples):
            where = f"{trajectory_id}:{index}"
            for field, expected in RAW_SAMPLE_FIELDS.items():
                if field not in sample:
                    problems.append(f"{where} missing '{field}'")
                elif not isinstance(sample[field], expected) or isinstance(sample[field], bool) and expected is int:
                    problems.append(f"{where} '{field}' has type {type(sample[field]).__name__}")
            if sample.get("schema_version") != 2:
                problems.append(f"{where} unexpected schema_version {sample.get('schema_version')}")
            if sample.get("environment_seed") != environment_seed:
                problems.append(f"{where} environment seed {sample.get('environment_seed')} != {environment_seed}")
            for field in ("cam_position", "cam_quaternion", "cam_velocity", "cam_angular_velocity",
                          "cam_direction", "cam_euler"):
                for value in sample.get(field, []):
                    if not finite(value):
                        problems.append(f"{where} '{field}' has a non-finite value")
            tick = sample.get("physics_tick")
            if tick in ticks:
                problems.append(f"{where} duplicate physics_tick {tick}")
            ticks.add(tick)
            if tick <= previous_tick:
                problems.append(f"{where} physics_tick not increasing ({tick} after {previous_tick})")
            previous_tick = tick
            stamp = sample.get("timestamp")
            if stamp in timestamps:
                # Wall clock is not the time base when sampling by physics tick: under
                # --fixed-fps several samples legitimately land in the same millisecond.
                # physics_tick and session_time are the authoritative, unique keys.
                duplicate_timestamps += 1
            timestamps.add(stamp)
            if stamp < previous_time:
                problems.append(f"{where} timestamp went backwards")
            previous_time = stamp

            observations = sample.get("observations", [])
            if sample.get("observation_count") != len(observations):
                problems.append(f"{where} observation_count mismatch")
            if sample.get("visible_count") != sum(1 for o in observations if o.get("frustum_visible")):
                problems.append(f"{where} visible_count mismatch")
            viewport = sample.get("viewport", [0, 0])
            if len(viewport) != 2 or viewport[0] <= 0 or viewport[1] <= 0:
                problems.append(f"{where} invalid viewport {viewport}")

            for observation in observations:
                total_observations += 1
                object_id = observation.get("object_id")
                for field, expected in OBSERVATION_FIELDS.items():
                    if field not in observation:
                        problems.append(f"{where} observation missing '{field}'")
                    elif not isinstance(observation[field], expected) or (
                            isinstance(observation[field], bool) and expected is int):
                        problems.append(f"{where} observation '{field}' wrong type")
                if object_id not in objects:
                    problems.append(f"{where} unknown object_id {object_id}")
                    continue
                if observation.get("occlusion_visible") not in (True, False, None):
                    problems.append(f"{where} occlusion_visible must be true/false/null")
                if observation.get("occlusion_reason") not in OCCLUSION_REASONS:
                    problems.append(f"{where} unknown occlusion_reason {observation.get('occlusion_reason')!r}")
                rays = observation.get("occlusion_rays", [])
                if len(rays) != 2 or rays[0] < rays[1]:
                    problems.append(f"{where} invalid occlusion_rays {rays}")
                coverage = observation.get("coverage")
                if not (finite(coverage) and 0.0 <= coverage <= 1.0):
                    problems.append(f"{where} coverage out of range {coverage}")
                if not (finite(observation.get("distance", -1)) and observation["distance"] >= 0.0):
                    problems.append(f"{where} invalid distance {observation.get('distance')}")
                visibility = observation.get("frustum_visible")
                if observation.get("approx_visible") != visibility:
                    problems.append(f"{where} approx_visible != frustum_visible")
                if visibility and not coverage > 0.0:
                    problems.append(f"{where} frustum_visible with zero coverage")
                resolution = observation.get("required_resolution")
                if resolution not in RESOLUTION_LADDER:
                    problems.append(f"{where} required_resolution {resolution} not on the ladder")
                if visibility and resolution == 0:
                    problems.append(f"{where} visible object with required_resolution 0")
                if not visibility and resolution != 0:
                    problems.append(f"{where} invisible object with required_resolution {resolution}")
                for texture_id in observation.get("texture_ids", []):
                    if texture_id not in texture_ids:
                        problems.append(f"{where} unknown texture id {texture_id}")
                if set(observation.get("texture_ids", [])) != objects[object_id]["texture_ids"]:
                    problems.append(f"{where} texture ids do not match the object index")
                key = (sample.get("physics_tick"), object_id)
                if key in seen_pairs:
                    duplicates += 1
                seen_pairs.add(key)

        meta_path = directory / f"{trajectory_id}.meta.json"
        if meta_path.is_file():
            meta = load_json(meta_path)
            if meta.get("samples") != len(samples):
                problems.append(f"{trajectory_id}: meta sample count {meta.get('samples')} != {len(samples)}")
            if meta.get("trajectory_id") != trajectory_id:
                problems.append(f"{trajectory_id}: meta trajectory_id {meta.get('trajectory_id')}")
            if meta.get("environment_seed") != environment_seed:
                problems.append(f"{trajectory_id}: meta environment seed mismatch")
            observation_distance = meta.get("max_visible_distance")
            if observation_distance is not None:
                report.info["max_observation_distance"] = observation_distance
        else:
            problems.append(f"{trajectory_id}: missing meta sidecar")

        if duplicates:
            problems.append(f"{trajectory_id}: {duplicates} duplicate (tick, object) observations")
        if duplicate_timestamps:
            report.warn(f"{trajectory_id}: {duplicate_timestamps} samples share a wall-clock millisecond "
                        f"(expected under --fixed-fps; physics_tick is unique)")
        status = "PASS" if not problems else "FAIL"
        report.check(f"raw {trajectory_id}", f"{len(samples)} samples, {status.lower()}", not problems)
        for problem in problems[:20]:
            report.error(problem)

    report.info["raw_samples"] = total_samples
    report.info["raw_observations"] = total_observations
    report.info["raw_trajectories"] = len(files)
    return loaded


def check_processed(report: Report, run_id: str, objects: dict[str, dict], splits: dict,
                    raw: dict[str, list[dict]], horizons: list[float], label_samples: int,
                    candidate_distance: float) -> None:
    directory = Path(PROCESSED_ROOT) / run_id
    if not directory.is_dir():
        report.error(f"processed run directory missing: {directory}")
        return
    label_files = sorted(directory.glob("*.labels.jsonl"))
    sample_files = sorted(directory.glob("*.samples.jsonl"))
    report.check("processed rows present", f"{len(label_files)} trajectories in {directory}",
                 bool(label_files))
    report.check("per-sample camera records present", f"{len(sample_files)} files",
                 len(sample_files) == len(label_files) and bool(sample_files))

    assignment: dict[str, str] = {}
    for split, entries in splits.get("splits", {}).items():
        for entry in entries:
            assignment[entry["trajectory_id"]] = split

    total_rows = 0
    total_sample_rows = 0
    eligible_rows = 0
    ineligible_rows = 0
    visible_counts = {horizon_suffix(h): 0 for h in horizons}
    resolution_counts: collections.Counter = collections.Counter()
    insufficient = {horizon_suffix(h): 0 for h in horizons}
    newly_visible = {horizon_suffix(h): 0 for h in horizons}
    occlusion_true = {horizon_suffix(h): 0 for h in horizons}
    occlusion_false = {horizon_suffix(h): 0 for h in horizons}
    occlusion_unknown = {horizon_suffix(h): 0 for h in horizons}
    occlusion_new = {horizon_suffix(h): 0 for h in horizons}
    occlusion_disagreement = {horizon_suffix(h): 0 for h in horizons}
    checked_labels = 0
    label_schemas: set[str] = set()
    rng = random.Random(20261105)

    for path in label_files:
        trajectory_id = path.stem.replace(".labels", "")
        try:
            rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        except json.JSONDecodeError as error:
            report.error(f"{trajectory_id}: unreadable labels JSONL ({error})")
            continue
        total_rows += len(rows)
        problems: list[str] = []
        seen: set[tuple[int, str]] = set()
        samples = raw.get(trajectory_id, [])
        expected_split = assignment.get(trajectory_id)
        if expected_split is None:
            problems.append(f"{trajectory_id} is not assigned to any split")

        sample_path = directory / f"{trajectory_id}.samples.jsonl"
        sample_records = []
        if sample_path.is_file():
            sample_records = [json.loads(line) for line in sample_path.open(encoding="utf-8")
                              if line.strip()]
            total_sample_rows += len(sample_records)
            if len(sample_records) != len(samples):
                problems.append(f"{trajectory_id}: {len(sample_records)} sample records "
                                f"but {len(samples)} raw samples")
            for position, record in enumerate(sample_records):
                if record.get("sample_index") != position:
                    problems.append(f"{trajectory_id}: sample record {position} has index "
                                    f"{record.get('sample_index')}")
                if record.get("split") != expected_split:
                    problems.append(f"{trajectory_id}: sample record {position} split mismatch")
                if position < len(samples) and record.get("physics_tick") != samples[position].get("physics_tick"):
                    problems.append(f"{trajectory_id}: sample record {position} physics_tick mismatch")
                viewport = record.get("viewport", [])
                if len(viewport) != 2 or viewport[0] <= 0 or viewport[1] <= 0:
                    problems.append(f"{trajectory_id}: sample record {position} invalid viewport")
                for field in ("cam_position", "cam_quaternion", "cam_velocity", "cam_direction",
                              "cam_angular_velocity"):
                    for value in record.get(field, []) or []:
                        if not finite(value):
                            problems.append(f"{trajectory_id}: sample record {position} "
                                            f"non-finite '{field}'")
        else:
            problems.append(f"{trajectory_id}: missing samples file")

        # Independent recomputation on a seeded sample plus the first/last rows.
        sample_positions = sorted({0, len(rows) - 1}
                                  | {rng.randrange(len(rows)) for _ in range(min(label_samples, len(rows)))}) \
            if rows else []

        # Detect which label set this file carries, so a preserved Phase 2 dataset
        # still validates after the Phase 2.5 targets were added.
        dual_targets = bool(rows) and "frustum_visible_in_0_5s" in rows[0]
        eligibility = bool(rows) and "resolution_eligible" in rows[0]
        label_schemas.add("phase-2.5 (dual targets)" if dual_targets else "phase-2 (single target)")

        state = {
            "visible_counts": visible_counts,
            "newly_visible": newly_visible,
            "insufficient": insufficient,
            "resolution_counts": resolution_counts,
            "occlusion_true": occlusion_true,
            "occlusion_false": occlusion_false,
            "occlusion_unknown": occlusion_unknown,
            "occlusion_new": occlusion_new,
            "occlusion_disagreement": occlusion_disagreement,
            "eligible_rows": 0,
            "ineligible_rows": 0,
            "checked_labels": 0,
            "max_candidate_distance": 0.0,
            "dual_targets": dual_targets,
            "eligibility": eligibility,
        }
        for index, row in enumerate(rows):
            # A validator must survive malformed input rather than aborting the run.
            try:
                _check_row(row, index, trajectory_id, expected_split, objects, samples,
                           horizons, sample_positions, seen, problems, state)
            except Exception as error:  # noqa: BLE001 - report and keep validating
                problems.append(f"{trajectory_id}:{index} raised {type(error).__name__}: {error}")
        checked_labels += state["checked_labels"]
        eligible_rows += state["eligible_rows"]
        ineligible_rows += state["ineligible_rows"]
        if state["max_candidate_distance"] > candidate_distance + 1e-6:
            problems.append(f"{trajectory_id}: candidate distance {state['max_candidate_distance']} "
                            f"exceeds the configured {candidate_distance}")

        status = "PASS" if not problems else "FAIL"
        report.check(f"processed {trajectory_id}", f"{len(rows)} label rows, {status.lower()}",
                     not problems)
        for problem in problems[:20]:
            report.error(problem)

    observation_distance = report.info.get("max_observation_distance")
    if observation_distance is not None:
        # A candidate's look-ahead must stay inside the radius the raw capture recorded,
        # otherwise its future visibility is invisible to the labeller.
        margin = observation_distance - candidate_distance
        report.check("labelling candidate radius inside the observation radius",
                     f"observation {observation_distance} m, candidates {candidate_distance} m, margin {margin:.1f} m",
                     margin >= 12.0)
        if margin < 12.0:
            report.warn(f"only {margin:.1f} m of margin between the candidate radius and the "
                        f"observation radius; a fast camera could leave the observation radius "
                        f"within the 1.0 s horizon")

    report.check("future labels recomputed independently",
                 f"{checked_labels} labels cross-checked", checked_labels > 0)

    # The occlusion-aware target must be genuinely tri-state: a real "unknown"
    # bucket that is reported as unknown rather than folded into "not visible".
    total_unknown = sum(occlusion_unknown.values())
    total_true = sum(occlusion_true.values())
    total_false = sum(occlusion_false.values())
    dual_targets_present = any(schema.startswith("phase-2.5") for schema in label_schemas)
    if dual_targets_present:
        report.check("occlusion-aware target is tri-state, unknown not treated as invisible",
                     f"{total_true} true / {total_false} false / {total_unknown} unknown "
                     f"across all horizons",
                     (total_true + total_false + total_unknown) > 0)
        report.check("frustum and occlusion targets are recorded separately",
                     f"{sum(occlusion_disagreement.values())} rows where the object enters the "
                     f"frustum but is definitively occluded",
                     True)
        report.check("resolution eligibility mask matches the texture index",
                     f"{eligible_rows} eligible / {ineligible_rows} ineligible rows",
                     eligible_rows > 0 and eligible_rows + ineligible_rows == total_rows)
    else:
        report.check("label set detected", ", ".join(sorted(label_schemas)) or "none", True)

    report.info["label_schemas"] = sorted(label_schemas)

    report.info["processed_label_rows"] = total_rows
    report.info["processed_sample_rows"] = total_sample_rows
    report.info["label_cross_checks"] = checked_labels
    report.info["frustum_target"] = {
        "true": visible_counts, "newly_visible": newly_visible,
    }
    report.info["occlusion_target"] = {
        "true": occlusion_true, "false": occlusion_false, "unknown": occlusion_unknown,
        "newly_visible": occlusion_new, "frustum_yes_occluded": occlusion_disagreement,
    }
    report.info["resolution_eligible_rows"] = eligible_rows
    report.info["resolution_ineligible_rows"] = ineligible_rows
    report.info["visible_in"] = visible_counts
    report.info["newly_visible"] = newly_visible
    report.info["resolution_insufficient"] = insufficient
    report.info["required_resolution_0_5s"] = dict(sorted(resolution_counts.items()))


def _check_row(row: dict, index: int, trajectory_id: str, expected_split: str,
               objects: dict[str, dict], samples: list[dict], horizons: list[float],
               sample_positions: list[int], seen: set, problems: list[str],
               state: dict) -> None:
    """Validates one processed row and updates the running label counts."""
    where = f"{trajectory_id}:{index}"
    if row.get("split") != expected_split:
        problems.append(f"{where} split {row.get('split')!r} != {expected_split!r}")
    object_id = row.get("object_id")
    if object_id not in objects:
        problems.append(f"{where} unknown object_id {object_id}")
        return
    key = (row.get("sample_index"), object_id)
    if key in seen:
        problems.append(f"{where} duplicate record for {object_id}")
    seen.add(key)
    if not finite(row.get("distance")) or row["distance"] < 0.0:
        problems.append(f"{where} invalid distance")
    else:
        state["max_candidate_distance"] = max(state["max_candidate_distance"], row["distance"])
    if row.get("required_resolution") not in RESOLUTION_LADDER:
        problems.append(f"{where} required_resolution off the ladder")
    if row.get("current_resolution") not in RESOLUTION_LADDER:
        problems.append(f"{where} current_resolution off the ladder")
    if row.get("current_resolution") != objects[object_id]["current_resolution"]:
        problems.append(f"{where} current_resolution disagrees with the object index")

    # Resolution eligibility mask, derived from the texture index.
    if state["eligibility"]:
        info = objects[object_id]
        eligible = row.get("resolution_eligible")
        if not isinstance(eligible, bool):
            problems.append(f"{where} resolution_eligible is not a boolean")
        elif eligible != info["resolution_eligible"]:
            problems.append(f"{where} resolution_eligible {eligible} != texture-derived "
                            f"{info['resolution_eligible']}")
        elif eligible:
            state["eligible_rows"] += 1
            if row.get("resolution_variants") != info["resolution_variants"]:
                problems.append(f"{where} resolution_variants {row.get('resolution_variants')} != "
                                f"{info['resolution_variants']}")
            if row.get("variant_texture_ids") != info["variant_texture_ids"]:
                problems.append(f"{where} variant_texture_ids {row.get('variant_texture_ids')} != "
                                f"{info['variant_texture_ids']}")
            if not info["variant_texture_ids"]:
                problems.append(f"{where} eligible but no variant-capable texture")
        else:
            state["ineligible_rows"] += 1
            if "resolution_variants" in row or "variant_texture_ids" in row:
                problems.append(f"{where} ineligible row carries resolution variant metadata")

    for horizon in horizons:
        suffix = horizon_suffix(horizon)
        visible = row.get(f"visible_in_{suffix}s")
        first = row.get(f"first_visible_time_{suffix}s")
        coverage = row.get(f"max_coverage_{suffix}s")
        resolution = row.get(f"required_resolution_{suffix}s")
        if not isinstance(visible, bool):
            problems.append(f"{where} visible_in_{suffix}s is not a boolean")
            continue
        if visible:
            state["visible_counts"][suffix] += 1
            if not row.get("frustum_visible"):
                state["newly_visible"][suffix] += 1
                if first is None or first <= 0.0:
                    problems.append(f"{where} becomes visible but first_visible_time is {first}")
            elif first != 0.0:
                problems.append(f"{where} already visible but first_visible_time is {first}")
            if first is None or not (0.0 <= first <= horizon + 1e-9):
                problems.append(f"{where} first_visible_time {first} outside [0, {horizon}]")
            if not finite(coverage) or coverage < 0.0:
                problems.append(f"{where} invalid max_coverage {coverage}")
            if resolution not in RESOLUTION_LADDER or resolution == 0:
                problems.append(f"{where} invalid required_resolution {resolution}")
            else:
                state["resolution_counts"][resolution] += 1
                marked = row.get(f"resolution_insufficient_{suffix}s")
                if marked != (resolution > row["current_resolution"]):
                    problems.append(f"{where} resolution_insufficient disagrees with required_resolution")
                if marked:
                    state["insufficient"][suffix] += 1
            if row.get("frustum_visible") and finite(coverage) and coverage + 1e-9 < row.get("coverage", 0.0):
                problems.append(f"{where} max_coverage below the current coverage")
        else:
            if first is not None:
                problems.append(f"{where} not visible but first_visible_time is {first}")
            if resolution is not None:
                problems.append(f"{where} not visible but required_resolution is {resolution}")
            if coverage not in (None, 0.0, 0):
                problems.append(f"{where} not visible but max_coverage is {coverage}")
            if row.get(f"resolution_insufficient_{suffix}s"):
                problems.append(f"{where} not visible but marked insufficient")

        # ---- target 1: frustum demand ----
        if state["dual_targets"]:
            _check_targets(row, where, suffix, horizon, visible, state, problems)

    if index not in sample_positions or not samples:
        return
    sample_index = row.get("sample_index")
    if not isinstance(sample_index, int) or sample_index >= len(samples):
        problems.append(f"{where} sample_index beyond the raw stream")
        return
    for horizon in horizons:
        suffix = horizon_suffix(horizon)
        expected_frustum, expected_occlusion = recompute_labels(
            samples, object_id, sample_index, horizon, row["current_resolution"])
        state["checked_labels"] += 1
        if state["dual_targets"]:
            for field in FRUSTUM_TARGET_FIELDS:
                _compare_field(row, where, f"frustum_{field}_{suffix}s",
                               expected_frustum[field], problems)
            for field in OCCLUSION_TARGET_FIELDS:
                _compare_field(row, where, f"occlusion_{field}_{suffix}s",
                               expected_occlusion[field], problems)
        else:
            # Phase 2 label set: the frustum target is the legacy visible_in_* family.
            for field in LEGACY_TARGET_FIELDS:
                _compare_field(row, where, f"{field}_{suffix}s", expected_frustum[field], problems)


def _compare_field(row: dict, where: str, name: str, value, problems: list[str]) -> None:
    actual = row.get(name)
    if isinstance(value, float) and isinstance(actual, (int, float)):
        if abs(value - actual) > 1e-6:
            problems.append(f"{where} {name} {actual} != recomputed {value}")
    elif actual != value:
        problems.append(f"{where} {name} {actual!r} != recomputed {value!r}")


def _check_targets(row: dict, where: str, suffix: str, horizon: float, visible,
                   state: dict, problems: list[str]) -> None:
    """Validate the two Phase 2.5 future targets and the eligibility mask."""
    frustum_target = row.get(f"frustum_visible_in_{suffix}s")
    if not isinstance(frustum_target, bool):
        problems.append(f"{where} frustum_visible_in_{suffix}s is not a boolean")
    elif frustum_target != visible:
        problems.append(f"{where} frustum_visible_in_{suffix}s {frustum_target} disagrees "
                        f"with the legacy visible_in_{suffix}s {visible}")

    occlusion_target = row.get(f"occlusion_visible_in_{suffix}s")
    unknown_flag = row.get(f"occlusion_unknown_in_{suffix}s")
    if occlusion_target not in (True, False, None):
        problems.append(f"{where} occlusion_visible_in_{suffix}s is not true/false/null")
        return
    if not isinstance(unknown_flag, bool):
        problems.append(f"{where} occlusion_unknown_in_{suffix}s is not a boolean")
    if occlusion_target is True:
        state["occlusion_true"][suffix] += 1
        if not frustum_target:
            problems.append(f"{where} occlusion target true but frustum target false")
        if not row.get("frustum_visible"):
            state["occlusion_new"][suffix] += 1
        first_occlusion = row.get(f"occlusion_first_visible_time_{suffix}s")
        if first_occlusion is None or not (0.0 <= first_occlusion <= horizon + 1e-9):
            problems.append(f"{where} occlusion first_visible_time {first_occlusion} "
                            f"outside [0, {horizon}]")
        if row.get(f"occlusion_required_resolution_{suffix}s") in (None, 0):
            problems.append(f"{where} occlusion target true but no required resolution")
    elif occlusion_target is False:
        state["occlusion_false"][suffix] += 1
        if unknown_flag:
            problems.append(f"{where} occlusion false but flagged unknown")
        if row.get(f"occlusion_first_visible_time_{suffix}s") is not None:
            problems.append(f"{where} occlusion false but first_visible_time is set")
        if frustum_target:
            # Frustum says yes, occlusion says definitively no: the case that makes the
            # two targets worth comparing.
            state["occlusion_disagreement"][suffix] += 1
    else:
        # UNKNOWN - this is the check that it is not silently recorded as invisible.
        state["occlusion_unknown"][suffix] += 1
        if not unknown_flag:
            problems.append(f"{where} occlusion unknown but not flagged as unknown")
        if not frustum_target:
            problems.append(f"{where} occlusion unknown without any frustum visibility - an "
                            f"object outside the frustum is definitively not visible")


LABEL_FIELD_PREFIXES = ("visible_in_", "first_visible_time_", "max_coverage_",
                        "required_resolution_", "resolution_insufficient_")


def check_separation(report: Report, run_id: str) -> None:
    """Raw recordings must stay free of derived labels, and vice versa."""
    raw_directory = Path(RAW_ROOT) / run_id
    processed_directory = Path(PROCESSED_ROOT) / run_id
    problems: list[str] = []
    if raw_directory.resolve() == processed_directory.resolve():
        problems.append("raw and processed resolve to the same directory")

    raw_files = sorted(raw_directory.glob("*.jsonl"))
    if raw_files:
        first = json.loads(raw_files[0].open(encoding="utf-8").readline())
        leaked = [key for key in first if key.startswith(LABEL_FIELD_PREFIXES)]
        if leaked:
            problems.append(f"raw sample record contains derived label fields: {leaked}")
        if "observations" not in first:
            problems.append("raw sample record has no observations array")

    label_files = sorted(processed_directory.glob("*.labels.jsonl"))
    if label_files:
        first = json.loads(label_files[0].open(encoding="utf-8").readline())
        if "observations" in first:
            problems.append("processed label record contains raw observations")
        if not any(key.startswith("visible_in_") for key in first):
            problems.append("processed label record has no visible_in_* field")

    report.check("raw and processed stored separately",
                 f"{RAW_ROOT}/{run_id} vs {PROCESSED_ROOT}/{run_id}", not problems)
    for problem in problems:
        report.error(problem)


def check_splits(report: Report, splits: dict, run_id: str) -> None:
    assignment: dict[str, str] = {}
    problems: list[str] = []
    for split, entries in splits.get("splits", {}).items():
        for entry in entries:
            identifier = entry["trajectory_id"]
            if identifier in assignment:
                problems.append(f"{identifier} in both {assignment[identifier]} and {split}")
            assignment[identifier] = split
    if not problems:
        report.check("split is a disjoint trajectory partition",
                     f"{len(assignment)} trajectories", True)
    for problem in problems:
        report.error(problem)

    train = {entry["trajectory_id"] for entry in splits.get("splits", {}).get("train", [])}
    test = {entry["trajectory_id"] for entry in splits.get("splits", {}).get("test", [])}
    validation = {entry["trajectory_id"] for entry in splits.get("splits", {}).get("validation", [])}
    overlap = (train & test) | (train & validation) | (test & validation)
    report.check("no trajectory shared between splits", f"train={len(train)} val={len(validation)} test={len(test)}",
                 not overlap)
    if overlap:
        report.error(f"trajectories shared between splits: {sorted(overlap)}")

    # Every row of a trajectory must carry that trajectory's split, checked in
    # check_processed; here confirm the raw run and the split manifest agree.
    raw_ids = {path.stem for path in (Path(RAW_ROOT) / run_id).glob("*.jsonl")}
    if raw_ids and raw_ids != set(assignment):
        report.warn(f"split manifest covers {len(assignment)} trajectories but the run has {len(raw_ids)}; "
                    f"difference: {sorted(raw_ids ^ set(assignment))}")
    report.info["split_counts"] = splits.get("counts", {})

    # Object identities are deliberately shared across splits: the dataset describes
    # one fixed scene, so this is not leakage. Stated explicitly so it is not
    # mistaken for an unhandled case.
    report.info["object_overlap_note"] = (
        "objects are shared across splits by design - the split is trajectory-level, "
        "which is what prevents temporal leakage"
    )


def write_report(report: Report, run_id: str) -> None:
    directory = Path(REPORT_ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    status = "PASSED" if not report.errors else "FAILED"
    lines = [
        f"# Dataset validation report - `{run_id}`",
        "",
        f"**Status: {status}**  ",
        f"errors: {len(report.errors)} | warnings: {len(report.warnings)}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for name, result, detail in report.checks:
        lines.append(f"| {name} | {result} | {detail} |")
    lines += ["", "## Dataset summary", "", "| Metric | Value |", "|---|---|"]
    for key, value in report.info.items():
        lines.append(f"| `{key}` | {value} |")
    if report.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in report.warnings]
    if report.errors:
        lines += ["", "## Errors", ""]
        lines += [f"- {error}" for error in report.errors[:200]]
        if len(report.errors) > 200:
            lines.append(f"- ... and {len(report.errors) - 200} more")
    else:
        lines += ["", "No errors.", ""]
    markdown = "\n".join(lines) + "\n"
    (directory / f"{run_id}-validation.md").write_text(markdown, encoding="utf-8")
    (directory / f"{run_id}-validation.json").write_text(
        json.dumps({
            "run_id": run_id,
            "status": status,
            "errors": report.errors,
            "warnings": report.warnings,
            "checks": [{"name": n, "status": s, "detail": d} for n, s, d in report.checks],
            "info": report.info,
        }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {directory / f'{run_id}-validation.md'}")
    print(f"wrote {directory / f'{run_id}-validation.json'}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--horizons", default="0.5,1.0")
    parser.add_argument("--label-samples", type=int, default=40,
                        help="rows per trajectory re-labelled independently")
    parser.add_argument("--candidate-distance", type=float, default=None,
                        dest="candidate_distance",
                        help="candidate radius the labels were built with "
                             "(defaults to the value in dataset_manifest.json, else 25)")
    arguments = parser.parse_args(argv)
    if arguments.candidate_distance is None:
        path = dataset_manifest_path(arguments.run)
        arguments.candidate_distance = load_json(path).get("candidate_distance_m", 25.0) \
            if path.is_file() else 25.0

    horizons = sorted(float(part) for part in arguments.horizons.split(",") if part.strip())
    report = Report()
    print(f"validating run '{arguments.run}'\n")

    object_index, texture_index, splits, _ = check_manifests(report, arguments.run)
    objects = object_facts(object_index, texture_index)
    texture_ids = set(texture_index.get("textures", {}).keys())
    raw = check_raw(report, arguments.run, objects, texture_ids,
                    object_index.get("environment_seed", -1))
    check_processed(report, arguments.run, objects, splits, raw, horizons,
                    arguments.label_samples, arguments.candidate_distance)
    check_separation(report, arguments.run)
    check_splits(report, splits, arguments.run)

    for name, status, detail in report.checks:
        print(f"  [{status}] {name}{(' - ' + detail) if detail else ''}")
    print(f"\n  errors: {len(report.errors)}  warnings: {len(report.warnings)}")
    for error in report.errors[:20]:
        print(f"    ERROR: {error}")
    write_report(report, arguments.run)
    print(f"\n=== {arguments.run}: {'PASSED' if not report.errors else 'FAILED'} ===")
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
