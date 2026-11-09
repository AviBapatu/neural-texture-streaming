"""Trajectory-level splits and grouped folds.

Every split in this project is a partition of *trajectories*, never of individual
rows. A sample from a trajectory in the test partition must never be visible to
training or to hyper-parameter selection, because consecutive samples of one
trajectory are almost perfectly correlated - a row-level split would leak.

Two things live here:

* the approved Phase 2.5 split, loaded by name, which provides the fixed final
  test set;
* deterministic grouped leave-trajectory-out folds used for variance estimates.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# The split names a caller may ask for. "experimental" resolves to the approved
# experimental manifest when one exists on disk (see docs/phase-3a-report.md - at
# the time of writing it does not, and the grouped folds are used instead).
KNOWN_SPLITS = ("phase2_5", "experimental")


@dataclass(frozen=True)
class Fold:
    """One grouped leave-trajectory-out fold."""

    index: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "fold": self.index,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split_manifest(path: str | Path) -> dict[str, str]:
    """Read a split manifest into `{trajectory_id: split_name}`.

    Accepts the `dataset/manifests/runs/<run>/splits.json` shape, whose `splits`
    map holds lists of objects with a `trajectory_id`.
    """
    manifest = load_json(path)
    if "splits" not in manifest:
        raise ValueError(f"{path} is not a split manifest (no 'splits' key)")

    assignment: dict[str, str] = {}
    for split_name, entries in manifest["splits"].items():
        # A plain list of ids is also accepted, for hand-written manifests.
        for entry in entries:
            trajectory_id = entry["trajectory_id"] if isinstance(entry, dict) else str(entry)
            if trajectory_id in assignment:
                raise ValueError(
                    f"{path} assigns trajectory {trajectory_id} to both "
                    f"{assignment[trajectory_id]} and {split_name}"
                )
            assignment[trajectory_id] = split_name
    return assignment


def load_experimental_split(path: str | Path | None) -> dict[str, str] | None:
    """Load the approved experimental split manifest, or return None.

    Returning None (rather than inventing a manifest) is deliberate: an approval
    that cannot be read from disk must not be silently simulated.
    """
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return load_split_manifest(candidate)


def assert_disjoint(trajectory_ids: Iterable[str], partitions: dict[str, Sequence[str]],
                    context: str = "") -> None:
    """Fail loudly if any trajectory appears in more than one partition."""
    seen: dict[str, str] = {}
    for name, ids in partitions.items():
        for trajectory_id in ids:
            if trajectory_id in seen:
                raise ValueError(
                    f"trajectory {trajectory_id} appears in both {seen[trajectory_id]} and "
                    f"{name}{f' ({context})' if context else ''}"
                )
            seen[trajectory_id] = name
    for trajectory_id in trajectory_ids:
        if trajectory_id not in seen:
            raise ValueError(
                f"trajectory {trajectory_id} is missing from every partition"
                f"{f' ({context})' if context else ''}"
            )


def build_folds(trajectory_ids: Sequence[str], folds: int = 7, fold_size: int = 2,
                seed: int = 20261105) -> list[Fold]:
    """Deterministically group trajectories and rotate them through each role.

    The trajectories are shuffled with a fixed seed and cut into `folds` groups of
    `fold_size`. Fold *i* holds group *i* out as test and the next group as
    validation, so over the whole set every trajectory serves as test exactly once
    and as validation exactly once, and no trajectory is ever both.

    The shuffle is seeded, so the grouping is reproducible; it is *not*
    stratified by trajectory kind, which the report calls out as a limitation.
    """
    ordered = sorted(trajectory_ids)
    if folds * fold_size != len(ordered):
        raise ValueError(
            f"{folds} folds of {fold_size} need {folds * fold_size} trajectories, "
            f"got {len(ordered)}"
        )
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    groups = [tuple(shuffled[i * fold_size:(i + 1) * fold_size]) for i in range(folds)]

    result: list[Fold] = []
    for index, test_group in enumerate(groups):
        validation_group = groups[(index + 1) % folds]
        held_out = set(test_group) | set(validation_group)
        train = tuple(t for t in shuffled if t not in held_out)
        fold = Fold(index=index, train=train, validation=validation_group, test=test_group)
        assert_disjoint(shuffled, {"train": fold.train, "validation": fold.validation,
                                   "test": fold.test}, context=f"fold {index}")
        result.append(fold)
    return result


def find_second_seed_dataset(path: str | Path | None) -> str | None:
    """Return the second-seed dataset directory if it exists, else None.

    A held-out run from a different environment seed would test whether features
    generalise across city layouts. None exists today, and this returns None
    rather than failing so the caller can log it and continue.
    """
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_dir():
        return None
    if not any(candidate.glob("*.labels.jsonl")):
        return None
    return str(candidate)


def trajectories_in(dataset_dir: str | Path) -> list[str]:
    """Trajectory ids present in a processed dataset directory."""
    directory = Path(dataset_dir)
    return sorted(path.name.replace(".labels.jsonl", "")
                  for path in directory.glob("*.labels.jsonl"))


def summary_lines(assignment: dict[str, str]) -> list[str]:
    """Human-readable rendering of a split assignment."""
    grouped: dict[str, list[str]] = {}
    for trajectory_id, split_name in sorted(assignment.items()):
        grouped.setdefault(split_name, []).append(trajectory_id)
    lines = []
    for split_name in sorted(grouped):
        lines.append(f"  {split_name:<11} {len(grouped[split_name]):>2}  "
                     f"{', '.join(grouped[split_name])}")
    return lines
