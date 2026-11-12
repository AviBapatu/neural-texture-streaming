"""Static scene facts and the working-set definition.

Texture memory is derived from the object index, which records the original
resolution of every texture an object uses. An object's "full" residency cost is the
sum over its textures of ``width * height * 4 * 4/3`` bytes - RGBA at 4 bytes per
pixel, plus a third again for the mip chain - multiplied by the sweep's size
multiplier.

Objects that have no generated variants use their original dimensions, which is what
the index records for them anyway; for the seven texture identities that do have
variants the original is also the largest available, so "original dimensions" and
"largest available" agree throughout this dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BYTES_PER_PIXEL = 4          # RGBA8
MIP_FACTOR = 4.0 / 3.0       # sum of a mip chain


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    category: str
    textures: tuple[tuple[str, int, int], ...]     # (texture_id, width, height)

    @property
    def full_bytes_at_1x(self) -> float:
        return float(sum(width * height * BYTES_PER_PIXEL * MIP_FACTOR
                         for _tid, width, height in self.textures))


def load_objects(path: str | Path) -> dict[str, ObjectRecord]:
    """Per-object texture dimensions from the object index."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    records: dict[str, ObjectRecord] = {}
    for entry in manifest["objects"]:
        textures = []
        for texture in entry.get("textures") or []:
            resolution = texture.get("original_resolution")
            if not resolution:
                available = texture.get("available_resolutions") or []
                if not available:
                    continue
                resolution = [max(available), max(available)]
            textures.append((texture["texture_id"], int(resolution[0]), int(resolution[1])))
        records[entry["object_id"]] = ObjectRecord(
            object_id=entry["object_id"], category=entry["category"],
            textures=tuple(textures))
    return records


def full_bytes(records: dict[str, ObjectRecord], object_id: str,
               size_multiplier: float) -> float:
    record = records.get(object_id)
    if record is None:
        raise KeyError(f"object {object_id!r} is not in the object index")
    return record.full_bytes_at_1x * float(size_multiplier)


def sized_bytes(records: dict[str, ObjectRecord], size_multiplier: float) -> dict[str, float]:
    """Every object's full-texture cost, precomputed once per sweep configuration."""
    return {oid: record.full_bytes_at_1x * float(size_multiplier)
            for oid, record in records.items()}


# --- working set --------------------------------------------------------------

def peak_working_set(trajectory_rows: dict[int, np.ndarray], visible: np.ndarray,
                     object_ids: np.ndarray, sizes: dict[str, float],
                     window_samples: int) -> float:
    """Peak bytes of distinct candidate objects visible in any sliding window.

    This is the denominator the budget percentages are taken from. For each window of
    ``window_samples`` consecutive samples (20 samples = 2 s at 10 Hz) the set of
    distinct objects visible in *any* sample of that window is collected, its full
    sizes are summed, and the largest window sum over the trajectory is returned.

    Using distinct objects rather than a per-sample sum is what makes it a working
    set: an object visible across the whole window is counted once, not twenty times.
    """
    if not window_samples or window_samples < 1:
        raise ValueError("window_samples must be at least 1")
    order = sorted(trajectory_rows)
    peak = 0.0
    for start in range(len(order)):
        window = order[start:start + window_samples]
        if not window:
            continue
        rows = np.concatenate([trajectory_rows[sample] for sample in window])
        # Distinct objects, not distinct rows: two rows for the same object at
        # different samples are the same resident texture.
        total = 0.0
        seen: set[str] = set()
        for row in rows:
            if not visible[row]:
                continue
            oid = object_ids[row]
            if oid in seen:
                continue
            seen.add(oid)
            total += sizes.get(oid, 0.0)
        peak = max(peak, total)
    return float(peak)
