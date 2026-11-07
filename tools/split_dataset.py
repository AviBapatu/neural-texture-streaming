#!/usr/bin/env python3
"""Assign trajectories to train / validation / test splits.

The split is made at **trajectory** level: every sample of a trajectory lands in
exactly one split, so no frame from a trajectory can appear in two splits. This is
what prevents leakage between training and evaluation - splitting individual frames
would put neighbouring moments of the same camera path on both sides.

Usage:
    python3 tools/split_dataset.py --profile full
    python3 tools/split_dataset.py --profile quick
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

TRAJECTORY_ROOT = "dataset/trajectories"
OUTPUT = "dataset/manifests/splits.json"
RUN_MANIFEST_ROOT = "dataset/manifests/runs"
DEFAULT_SEED = 20261105
SPLITS = ("train", "validation", "test")


def allocate(count: int) -> dict[str, int]:
    """Deterministic 60/20/20 split with at least one trajectory per split."""
    if count < len(SPLITS):
        raise ValueError(f"need at least {len(SPLITS)} trajectories, got {count}")
    test = max(1, round(count * 0.2))
    validation = max(1, round(count * 0.2))
    train = count - validation - test
    if train < 1:
        train, validation, test = count - 2, 1, 1
    return {"train": train, "validation": validation, "test": test}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=("full", "quick"), default="full")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--run-id", default="",
                        help="also write a copy under dataset/manifests/runs/<run-id>/ so the "
                             "assignment for a run survives later runs with other profiles")
    arguments = parser.parse_args(argv)

    index_path = Path(TRAJECTORY_ROOT) / arguments.profile / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"trajectory index not found: {index_path}\n"
                         f"run: python3 tools/generate_trajectories.py --profile {arguments.profile}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    trajectories = sorted(index["trajectories"], key=lambda entry: entry["trajectory_id"])
    counts = allocate(len(trajectories))

    # Deterministic shuffle: the same seed always yields the same assignment.
    shuffled = list(trajectories)
    random.Random(arguments.seed).shuffle(shuffled)

    assignment: dict[str, list[dict]] = {split: [] for split in SPLITS}
    cursor = 0
    for split in SPLITS:
        for _ in range(counts[split]):
            entry = shuffled[cursor]
            cursor += 1
            assignment[split].append({
                "trajectory_id": entry["trajectory_id"],
                "kind": entry["kind"],
                "duration_seconds": entry["duration_seconds"],
                "file": entry["file"],
            })

    # Invariants that make the split trustworthy.
    seen: dict[str, str] = {}
    problems: list[str] = []
    for split, entries in assignment.items():
        for entry in entries:
            identifier = entry["trajectory_id"]
            if identifier in seen:
                problems.append(f"{identifier} appears in both {seen[identifier]} and {split}")
            seen[identifier] = split
    for entry in trajectories:
        if entry["trajectory_id"] not in seen:
            problems.append(f"{entry['trajectory_id']} is not assigned to any split")
    if problems:
        for problem in problems:
            print(f"  ERROR: {problem}")
        return 1

    for split, entries in assignment.items():
        kinds = sorted({entry["kind"] for entry in entries})
        print(f"  {split:<11} {len(entries):>2} trajectories, kinds: {', '.join(kinds)}")
        for entry in entries:
            print(f"      {entry['trajectory_id']:<26} {entry['kind']}")

    manifest = {
        "schema_version": 1,
        "profile": arguments.profile,
        "seed": arguments.seed,
        "rule": "trajectory-level split; every sample of a trajectory belongs to exactly one "
                "split; test trajectories are never used for training or model selection",
        "source_index": str(index_path).replace("\\", "/"),
        "environment_seed": index["environment_seed"],
        "trajectory_count": len(trajectories),
        "counts": {split: len(assignment[split]) for split in SPLITS},
        "splits": assignment,
    }
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(manifest, indent=2) + "\n"
    output_path.write_text(serialised, encoding="utf-8")
    print(f"\nwrote {output_path}")
    if arguments.run_id:
        # The top-level manifest always describes the most recent run; keep a stable
        # per-run copy so an earlier dataset stays self-consistent.
        run_path = Path(RUN_MANIFEST_ROOT) / arguments.run_id / "splits.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(serialised, encoding="utf-8")
        print(f"wrote {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
