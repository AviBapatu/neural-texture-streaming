"""Trace-driven replay: two-level residency, shared bandwidth, measurable deficit.

The model is deliberately small and fully deterministic:

* A low-resolution placeholder is always resident and costs nothing. It is never
  counted, never loaded and never evicted - it stands in for the engine's low mips,
  which is exactly why it is free.
* An object counts as *good* only once its full texture is resident. Full texture
  cost is ``width * height * 4 * 4/3`` per texture, times the sweep's size
  multiplier.
* Requests are served in priority order from a shared bandwidth pool of
  ``bandwidth * dt`` bytes per step. A request becomes eligible for streaming
  ``latency`` seconds after it is first made.
* Residency is capped at the budget. When a completing load does not fit, the
  lowest-priority residents below the incoming object's priority are evicted; if it
  still does not fit the load is abandoned and will be re-requested, which is the
  thrashing behaviour a too-small budget should produce.

Ground truth for the deficit is independent of every policy: an object is visible at
a step if it is frustum visible and not *definitively* occluded. Unknown occlusion
counts as visible, matching the project's convention everywhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from training.dataset import OCCLUSION_FALSE

from . import scene
from .policies import Policy, occlusion_visible

SAMPLE_INTERVAL_SECONDS = 0.1


# --- data ---------------------------------------------------------------------

@dataclass
class TrajectoryView:
    """One trajectory's rows, grouped by sample, in sample order."""

    trajectory_id: str
    samples: list[int]
    rows_by_sample: dict[int, np.ndarray]
    excluded_samples: frozenset[int]     # within +/- snap_window of a yaw snap
    all_rows: np.ndarray


@dataclass
class SimData:
    object_ids: np.ndarray
    trajectory: np.ndarray
    trajectories: list[str]
    sample_index: np.ndarray
    frustum_visible: np.ndarray
    occlusion_state: np.ndarray
    coverage: np.ndarray
    scores: dict[str, np.ndarray]
    occlusion_visible_in_1_0s: np.ndarray
    occlusion_first_visible_time_1_0s: np.ndarray
    views: list[TrajectoryView]

    @property
    def ground_truth_visible(self) -> np.ndarray:
        return occlusion_visible(self.frustum_visible, self.occlusion_state)

    def fold_rows(self, name: str) -> np.ndarray:
        return self.scores[name]


def load_labels_by_key(dataset_dir: str | Path, fields: tuple[str, ...]) -> dict[tuple, tuple]:
    """Read selected label fields keyed by (trajectory, sample, object).

    The simulator needs the *future* occlusion label for the oracle policy, which is
    not part of the score bundle. Reading it here keeps the score bundle to scores.
    """
    import json
    directory = Path(dataset_dir)
    table: dict[tuple, tuple] = {}
    for path in sorted(directory.glob("*.labels.jsonl")):
        trajectory_id = path.name.replace(".labels.jsonl", "")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (trajectory_id, int(row["sample_index"]), row["object_id"])
                table[key] = tuple(row.get(field) for field in fields)
    return table


def build_sim_data(oof_path: str | Path, dataset_dir: str | Path,
                   snap_window_samples: int) -> SimData:
    """Load the score bundle and the oracle columns, aligned by key."""
    with np.load(oof_path, allow_pickle=True) as payload:
        trajectory = np.array(payload["trajectory"], dtype=object)
        sample_index = payload["sample_index"].astype(np.int64)
        object_id = np.array(payload["object_id"], dtype=object)
        frustum_visible = payload["frustum_visible"].astype(bool)
        occlusion_state = payload["occlusion_state"].astype(np.int8)
        coverage = payload["coverage"].astype(np.float64)
        snap_affected = payload["snap_affected_yaw"].astype(bool)
        scores = {key: payload[key].astype(np.float64) for key in payload.files
                  if "__" in key or key.startswith("heuristic_")}

    lookup = load_labels_by_key(dataset_dir, ("occlusion_visible_in_1_0s",
                                              "occlusion_first_visible_time_1_0s"))
    future_visible = np.zeros(len(object_id), dtype=bool)
    first_visible = np.full(len(object_id), np.nan, dtype=np.float64)
    missing = 0
    for index in range(len(object_id)):
        key = (trajectory[index], int(sample_index[index]), object_id[index])
        hit = lookup.get(key)
        if hit is None:
            missing += 1
            continue
        future_visible[index] = bool(hit[0])
        if hit[1] is not None:
            first_visible[index] = float(hit[1])
    if missing:
        raise RuntimeError(f"{missing} rows had no matching label row")

    order = sorted(set(trajectory))
    index_of = {name: position for position, name in enumerate(order)}
    trajectory_codes = np.array([index_of[t] for t in trajectory], dtype=np.int32)

    views: list[TrajectoryView] = []
    for position, name in enumerate(order):
        rows = np.flatnonzero(trajectory_codes == position)
        rows = rows[np.lexsort((object_id[rows], sample_index[rows]))]
        samples = sorted(set(sample_index[rows].tolist()))
        rows_by_sample = {sample: rows[sample_index[rows] == sample] for sample in samples}
        snapped = {int(s) for s in sample_index[rows][snap_affected[rows]].tolist()}
        excluded = frozenset(s + offset for s in snapped
                             for offset in range(-snap_window_samples, snap_window_samples + 1))
        views.append(TrajectoryView(name, samples, rows_by_sample, excluded, rows))

    return SimData(object_ids=object_id, trajectory=trajectory_codes,
                   trajectories=order, sample_index=sample_index,
                   frustum_visible=frustum_visible, occlusion_state=occlusion_state,
                   coverage=coverage, scores=scores,
                   occlusion_visible_in_1_0s=future_visible,
                   occlusion_first_visible_time_1_0s=first_visible, views=views)


# --- configuration ------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    budget_bytes: float
    size_multiplier: float
    latency_seconds: float
    bandwidth_bytes_per_s: float
    sample_interval: float = SAMPLE_INTERVAL_SECONDS

    @property
    def label(self) -> str:
        return (f"budget={self.budget_bytes / 1e6:.1f}MB,size={self.size_multiplier:g}x,"
                f"latency={self.latency_seconds:g}s")


@dataclass
class StepMetrics:
    deficit: float = 0.0
    visible_coverage: float = 0.0
    pop_in_events: int = 0
    pop_in_durations: list[int] = field(default_factory=list)
    wasted_prefetches: int = 0
    bytes_streamed: float = 0.0
    peak_resident_bytes: float = 0.0
    steps: int = 0
    scored_steps: int = 0
    loads_completed: int = 0


# --- the engine ---------------------------------------------------------------

def simulate(view: TrajectoryView, policy: Policy, priorities: np.ndarray,
             sizes: dict[str, float], ground_truth: np.ndarray, coverage: np.ndarray,
             object_ids: np.ndarray, config: RunConfig,
             stop_sample: int | None = None) -> StepMetrics:
    """Replay one trajectory under one policy. Deterministic; no RNG anywhere.

    ``stop_sample`` replays only samples strictly below it. It exists so the causal
    policies can be tested exactly: a reactive policy's first k steps must be
    identical whether the trajectory ends at k or continues, and that is only
    observable if the same run can be cut short.
    """
    dt = config.sample_interval
    resident: dict[str, float] = {}
    resident_bytes = 0.0
    loaded: dict[str, float] = {}
    requested_at: dict[str, float] = {}
    ever_visible: dict[str, bool] = {}
    episode_open: dict[str, int] = {}
    extended: set[str] = set()

    metrics = StepMetrics()

    def close_episode(object_id: str) -> None:
        if object_id in episode_open:
            metrics.pop_in_durations.append(episode_open.pop(object_id))

    def evict(object_id: str) -> None:
        nonlocal resident_bytes
        resident_bytes -= resident.pop(object_id)
        if not ever_visible.get(object_id, False):
            metrics.wasted_prefetches += 1
        requested_at.pop(object_id, None)

    for sample in view.samples:
        if stop_sample is not None and sample >= stop_sample:
            break
        t = sample * dt
        rows = view.rows_by_sample[sample]
        scored = sample not in view.excluded_samples
        metrics.steps += 1
        metrics.scored_steps += int(scored)

        # --- object-level priority: an object can appear at most once per sample
        object_priority: dict[str, float] = {}
        row_of: dict[str, int] = {}
        for row in rows:
            oid = object_ids[row]
            value = float(priorities[row])
            if value > object_priority.get(oid, 0.0):
                object_priority[oid] = value
                row_of[oid] = row

        # --- requests
        for oid, value in object_priority.items():
            if value <= 0.0 or oid in resident:
                continue
            requested_at.setdefault(oid, t)

        # --- stream the step's bandwidth, in priority order
        available = config.bandwidth_bytes_per_s * dt
        eligible = [oid for oid, since in requested_at.items()
                    if oid not in resident and t - since >= config.latency_seconds - 1e-9]
        eligible.sort(key=lambda o: (-object_priority.get(o, 0.0), requested_at[o], o))

        completing: list[str] = []
        for oid in eligible:
            if available <= 0.0:
                break
            need = sizes[oid] - loaded.get(oid, 0.0)
            if need <= 0.0:
                completing.append(oid)
                continue
            take = min(need, available)
            loaded[oid] = loaded.get(oid, 0.0) + take
            metrics.bytes_streamed += take
            available -= take
            if sizes[oid] - loaded[oid] <= 1e-9:
                completing.append(oid)

        # --- complete loads, enforcing the budget by eviction
        for oid in completing:
            size = sizes[oid]
            if size > config.budget_bytes:
                # Can never be resident under this budget. Drop the partial load so it
                # is not counted as progress and let it be re-requested.
                loaded.pop(oid, None)
                requested_at.pop(oid, None)
                continue
            if resident_bytes + size > config.budget_bytes:
                evictable = sorted(
                    (o for o in resident if object_priority.get(o, 0.0)
                     < object_priority.get(oid, 0.0)),
                    key=lambda o: (object_priority.get(o, 0.0), requested_at.get(o, 0.0), o))
                for victim in evictable:
                    if resident_bytes + size <= config.budget_bytes:
                        break
                    evict(victim)
            if resident_bytes + size <= config.budget_bytes:
                resident[oid] = size
                resident_bytes += size
                ever_visible.setdefault(oid, False)
                loaded.pop(oid, None)
                requested_at.pop(oid, None)
                metrics.loads_completed += 1
            else:
                loaded.pop(oid, None)
                requested_at.pop(oid, None)

        metrics.peak_resident_bytes = max(metrics.peak_resident_bytes, resident_bytes)

        # --- measure against ground truth
        extended.clear()
        for row in rows:
            if not ground_truth[row]:
                continue
            oid = object_ids[row]
            covered = float(coverage[row])
            if scored:
                metrics.visible_coverage += covered
            if oid in resident:
                ever_visible[oid] = True
                close_episode(oid)
            else:
                if scored:
                    metrics.deficit += covered
                extended.add(oid)
                if oid not in episode_open:
                    metrics.pop_in_events += 1
                    episode_open[oid] = 0
                episode_open[oid] += 1

        for oid in list(episode_open):
            if oid not in extended:
                close_episode(oid)

    for oid in list(episode_open):
        close_episode(oid)
    for oid in list(resident):
        evict(oid)

    return metrics


def summarise(metrics: StepMetrics, config: RunConfig | None = None) -> dict:
    """Flatten to a metrics dict. ``config`` adds the budget context when supplied."""
    durations = metrics.pop_in_durations
    result = {
        "deficit": (metrics.deficit / metrics.visible_coverage
                    if metrics.visible_coverage > 0 else 0.0),
        "deficit_bytes_weighted": metrics.deficit,
        "visible_coverage": metrics.visible_coverage,
        "pop_in_events": metrics.pop_in_events,
        "pop_in_mean_duration_samples": float(np.mean(durations)) if durations else 0.0,
        "pop_in_max_duration_samples": int(max(durations)) if durations else 0,
        "wasted_prefetches": metrics.wasted_prefetches,
        "loads_completed": metrics.loads_completed,
        "bytes_streamed": metrics.bytes_streamed,
        "peak_resident_bytes": metrics.peak_resident_bytes,
        "steps": metrics.steps,
        "scored_steps": metrics.scored_steps,
    }
    if config is not None:
        result["budget_bytes"] = config.budget_bytes
        result["peak_over_budget"] = (metrics.peak_resident_bytes / config.budget_bytes
                                      if config.budget_bytes > 0 else float("inf"))
    return result
