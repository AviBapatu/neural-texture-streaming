#!/usr/bin/env python3
"""Validate a telemetry session written by `godot/scripts/telemetry_recorder.gd`.

Checks that every record carries the fields the dataset pipeline needs, that the
numbers are finite and within sane ranges, and that visible-object entries are
well formed. Exits non-zero when a session does not validate.

Usage:
    python3 tools/validate_telemetry.py dataset/telemetry/session_x.jsonl
    python3 tools/validate_telemetry.py dataset/telemetry            # newest session
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REQUIRED_SAMPLE_FIELDS = {
    "timestamp": (int, float),
    "session_time": (int, float),
    "frame": int,
    "cam_position": list,
    "cam_quaternion": list,
    "cam_euler": list,
    "cam_velocity": list,
    "cam_direction": list,
    "visible": list,
}

REQUIRED_VISIBLE_FIELDS = {
    "id": int,
    "name": str,
    "source": str,
    "distance": (int, float),
    "coverage": (int, float),
    "textures": list,
}

CSV_COLUMNS = [
    "timestamp", "session_time", "frame",
    "cam_x", "cam_y", "cam_z",
    "quat_x", "quat_y", "quat_z", "quat_w",
    "euler_x", "euler_y", "euler_z",
    "vel_x", "vel_y", "vel_z",
    "dir_x", "dir_y", "dir_z",
    "object_id", "object_name", "object_source", "distance", "coverage", "textures",
]


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.notes: list[str] = []

    def problem(self, message: str) -> None:
        self.problems.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def newest_session(directory: Path) -> Path:
    sessions = sorted(directory.glob("session_*.jsonl"))
    if not sessions:
        raise FileNotFoundError(f"no telemetry sessions found in {directory}")
    return sessions[-1]


def check_number(report: Report, label: str, value, where: str) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        report.problem(f"{where}: {label} is not numeric ({value!r})")
        return False
    if not math.isfinite(float(value)):
        report.problem(f"{where}: {label} is not finite ({value!r})")
        return False
    return True


def validate_jsonl(path: Path, report: Report) -> dict:
    samples = 0
    observations = 0
    coverage_min, coverage_max = math.inf, -math.inf
    distance_min, distance_max = math.inf, -math.inf
    texture_ids: set[str] = set()
    last_frame = -1
    empty_samples = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            where = f"{path.name}:{line_number}"
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                report.problem(f"{where}: invalid JSON ({error})")
                continue
            samples += 1
            for field, expected in REQUIRED_SAMPLE_FIELDS.items():
                if field not in sample:
                    report.problem(f"{where}: missing field '{field}'")
                elif not isinstance(sample[field], expected):
                    report.problem(f"{where}: field '{field}' has type {type(sample[field]).__name__}")
            if len(sample.get("cam_position", [])) != 3:
                report.problem(f"{where}: cam_position must have 3 components")
            if len(sample.get("cam_quaternion", [])) != 4:
                report.problem(f"{where}: cam_quaternion must have 4 components")
            if len(sample.get("cam_euler", [])) != 3:
                report.problem(f"{where}: cam_euler must have 3 components")
            if len(sample.get("cam_velocity", [])) != 3:
                report.problem(f"{where}: cam_velocity must have 3 components")
            if len(sample.get("cam_direction", [])) != 3:
                report.problem(f"{where}: cam_direction must have 3 components")
            for field in ("cam_position", "cam_quaternion", "cam_euler", "cam_velocity", "cam_direction"):
                for value in sample.get(field, []):
                    check_number(report, field, value, where)

            frame = sample.get("frame", -1)
            if isinstance(frame, int):
                if frame <= last_frame:
                    report.problem(f"{where}: frame did not advance ({frame} after {last_frame})")
                last_frame = frame

            visible = sample.get("visible", [])
            if not visible:
                empty_samples += 1
            for entry in visible:
                observations += 1
                for field, expected in REQUIRED_VISIBLE_FIELDS.items():
                    if field not in entry:
                        report.problem(f"{where}: visible entry missing '{field}'")
                    elif not isinstance(entry[field], expected) or isinstance(entry[field], bool):
                        report.problem(f"{where}: visible.{field} has type {type(entry[field]).__name__}")
                if "textures" in entry:
                    if not entry["textures"]:
                        report.problem(f"{where}: visible entry has no texture identifiers")
                    for texture in entry["textures"]:
                        if not isinstance(texture, str) or not texture:
                            report.problem(f"{where}: invalid texture identifier {texture!r}")
                        else:
                            texture_ids.add(texture)
                if check_number(report, "distance", entry.get("distance", 0), where):
                    if entry["distance"] < 0:
                        report.problem(f"{where}: negative distance {entry['distance']}")
                    distance_min = min(distance_min, entry["distance"])
                    distance_max = max(distance_max, entry["distance"])
                if check_number(report, "coverage", entry.get("coverage", 0), where):
                    if not 0.0 <= entry["coverage"] <= 1.0:
                        report.problem(f"{where}: coverage out of range {entry['coverage']}")
                    coverage_min = min(coverage_min, entry["coverage"])
                    coverage_max = max(coverage_max, entry["coverage"])

    if samples == 0:
        report.problem(f"{path.name}: contains no samples")
    if empty_samples == samples and samples > 0:
        report.problem(f"{path.name}: every sample has an empty 'visible' list")

    return {
        "samples": samples,
        "observations": observations,
        "coverage": (coverage_min, coverage_max),
        "distance": (distance_min, distance_max),
        "texture_ids": texture_ids,
    }


def validate_csv(path: Path, report: Report, expected_rows: int | None) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            report.problem(f"{path.name}: unexpected header\n    got     {reader.fieldnames}\n    expected {CSV_COLUMNS}")
            return 0
        rows = 0
        for line_number, row in enumerate(reader, start=2):
            rows += 1
            for column in ("timestamp", "session_time", "distance", "coverage"):
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    report.problem(f"{path.name}:{line_number}: '{column}' is not a number ({row[column]!r})")
                    continue
                if not math.isfinite(value):
                    report.problem(f"{path.name}:{line_number}: '{column}' is not finite")
            if not row.get("object_id", "").isdigit():
                report.problem(f"{path.name}:{line_number}: object_id is not an integer")
            if not row.get("textures"):
                report.problem(f"{path.name}:{line_number}: empty textures column")
    if expected_rows is not None and rows != expected_rows:
        report.problem(f"{path.name}: {rows} rows but JSONL has {expected_rows} observations")
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="a .jsonl session file or the telemetry directory")
    parser.add_argument("--min-samples", type=int, default=1)
    arguments = parser.parse_args(argv)

    target = Path(arguments.path)
    jsonl_path = newest_session(target) if target.is_dir() else target
    if not jsonl_path.is_file():
        print(f"not a file: {jsonl_path}", file=sys.stderr)
        return 2

    report = Report()
    stats = validate_jsonl(jsonl_path, report)
    csv_path = jsonl_path.with_suffix(".csv")
    if csv_path.is_file():
        rows = validate_csv(csv_path, report, stats["observations"] or None)
        report.note(f"CSV rows: {rows}")
    else:
        report.note(f"no CSV companion for {csv_path.name}")

    if stats["samples"] < arguments.min_samples:
        report.problem(f"{jsonl_path.name}: {stats['samples']} samples < required {arguments.min_samples}")

    print(f"session        : {jsonl_path}")
    print(f"samples        : {stats['samples']}")
    print(f"observations   : {stats['observations']}")
    if stats["observations"]:
        coverage_min, coverage_max = stats["coverage"]
        distance_min, distance_max = stats["distance"]
        print(f"coverage range : {coverage_min:.6f} .. {coverage_max:.6f}")
        print(f"distance range : {distance_min:.3f} .. {distance_max:.3f}")
        print(f"texture ids    : {len(stats['texture_ids'])} distinct")
        for texture in sorted(stats["texture_ids"]):
            print(f"                 {texture}")
    for note in report.notes:
        print(f"note           : {note}")
    if report.problems:
        print(f"\nINVALID - {len(report.problems)} problem(s):")
        for problem in report.problems[:40]:
            print(f"  - {problem}")
        return 1
    print("\nVALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
