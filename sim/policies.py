"""Streaming policies.

A policy answers one question per row per step: how much do we want this object's
full texture right now? The answer is a priority, and the convention is fixed:

* **2.0** - the object is visible *now* under this policy's definition, so it is being
  displayed and its full texture is wanted immediately.
* **(0, 1]** - the policy predicts the object will matter; higher means sooner or more
  likely. This is where a predictor earns its keep.
* **0.0** - not requested. Nothing is loaded for this object.

There is no threshold and nothing is tuned per policy. The value of a predictor is
carried entirely by the *ordering* it induces within the (0, 1] band, because that is
what decides who gets bandwidth when several objects want the same texture budget.

The two reactive policies have no prediction, so their score band is empty: they ask
for exactly what is on screen and nothing else.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

import numpy as np

from training.dataset import OCCLUSION_FALSE, OCCLUSION_TRUE, OCCLUSION_UNKNOWN

# The margin-to-score map used for the frozen heuristic policies. The heuristic's
# raw output is a frustum margin in degrees, unbounded in both directions. ``sigma``
# is a monotone squashing into (0, 1) that leaves the ranking untouched - which is
# all the simulator consumes - so it is a units change, not a tuning knob.
HEURISTIC_SIGMA_DEG = 10.0

VISIBLE_NOW_PRIORITY = 2.0
ORACLE_HORIZON_SECONDS = 1.0

# Phase 4b: safety margin added to the matched oracle's horizon. It buys slack for the
# one-step granularity of the replay - a load that completes exactly on the sample an
# object becomes visible is already one step late for that sample's measurement.
MATCHED_MARGIN_SECONDS = 0.2


@dataclass(frozen=True)
class Policy:
    """One streaming policy, evaluated as per-row arrays over the whole dataset."""

    name: str
    label: str
    visible_now: np.ndarray       # (n,) bool: priority 2
    score: np.ndarray | None      # (n,) float in [0, 1]: priority when not visible now
    deployable: bool              # False only for the oracle upper bounds
    # True only for oracle_matched, whose score band depends on the run's latency,
    # bandwidth and object sizes. Everything else is a fixed array.
    config_dependent: bool = False

    def priority(self, rows: np.ndarray) -> np.ndarray:
        """Priority for a subset of rows. Zero means "do not load"."""
        if self.score is None:
            raise ValueError(
                f"policy {self.name!r} needs a run configuration; call "
                "matched_priorities(...) instead of policy.priority(...)")
        return np.where(self.visible_now[rows], VISIBLE_NOW_PRIORITY, self.score[rows])


def squash(margin_deg: np.ndarray, sigma: float = HEURISTIC_SIGMA_DEG) -> np.ndarray:
    """Monotone map from degrees to (0, 1). Order-preserving, so ranking is unchanged."""
    return 1.0 / (1.0 + np.exp(-np.asarray(margin_deg, dtype=np.float64) / float(sigma)))


def occlusion_visible(frustum_visible: np.ndarray, occlusion_state: np.ndarray) -> np.ndarray:
    """Visible under the project's occlusion convention: **unknown counts as visible**.

    This is the same definition the deficit ground truth uses, so ``reactive_visible``
    is the best a policy can do with no prediction at all.
    """
    return (np.asarray(frustum_visible, dtype=bool)
            & (np.asarray(occlusion_state) != OCCLUSION_FALSE))


def lagged_occlusion_state(data, lag_samples: int) -> np.ndarray:
    """The occlusion state as the engine would have known it ``lag_samples`` earlier.

    Phase 4b's baselines all use ground-truth occlusion *now*, which is free and
    perfect. This is the one baseline that pays for visibility: at sample s it reads
    what the occlusion buffer said at s - lag, and if that object had no row back then
    it falls back to the most recent earlier reading. If there is none at all the state
    is unknown, which the project's convention counts as visible.

    No future information is used anywhere: the value at s depends only on samples
    strictly at or before s - lag.
    """
    if lag_samples <= 0:
        return np.asarray(data.occlusion_state, dtype=np.int8)
    result = np.full(len(data.object_ids), OCCLUSION_UNKNOWN, dtype=np.int8)
    for view in data.views:
        history: dict[str, list[tuple[int, int]]] = {}
        for sample in view.samples:
            for row in view.rows_by_sample[sample]:
                history.setdefault(data.object_ids[row], []).append(
                    (sample, int(data.occlusion_state[row])))
        prepared = {oid: (np.array([s for s, _ in entries], dtype=np.int64),
                          np.array([state for _, state in entries], dtype=np.int8))
                    for oid, entries in history.items()}
        for sample in view.samples:
            target = sample - lag_samples
            for row in view.rows_by_sample[sample]:
                samples_arr, states_arr = prepared[data.object_ids[row]]
                index = int(np.searchsorted(samples_arr, target, side="right")) - 1
                if index >= 0:
                    result[row] = states_arr[index]
    return result


def time_to_next_visible(data, sample_interval: float) -> np.ndarray:
    """Seconds from each row's own sample until that object is next visible.

    Computed from the trajectory's **own future samples**, not from the 1.0 s label
    column: the label saturates at the horizon and oracle_matched needs the real
    distance, which for a slow load can be several seconds. Zero while the object is
    visible; ``inf`` if it is never visible again before the trajectory ends.

    The visibility convention is the same ground truth the deficit uses, so a row is
    visible here exactly when the simulator would count it as a hit.
    """
    ground_truth = data.ground_truth_visible
    result = np.full(len(data.object_ids), np.inf, dtype=np.float64)
    for view in data.views:
        visible_samples: dict[str, list[int]] = {}
        for sample in view.samples:
            for row in view.rows_by_sample[sample]:
                if ground_truth[row]:
                    visible_samples.setdefault(data.object_ids[row], []).append(sample)
        per_object = {oid: np.asarray(sorted(values), dtype=np.int64)
                      for oid, values in visible_samples.items()}
        for sample in view.samples:
            for row in view.rows_by_sample[sample]:
                positions = per_object.get(data.object_ids[row])
                if positions is None:
                    continue
                index = int(np.searchsorted(positions, sample, side="left"))
                if index < len(positions):
                    result[row] = (int(positions[index]) - sample) * sample_interval
    return result


def matched_priorities(visible_now: np.ndarray, time_to_visible: np.ndarray,
                       size_bytes_per_row: np.ndarray, latency_seconds: float,
                       bandwidth_bytes_per_s: float,
                       margin_seconds: float = MATCHED_MARGIN_SECONDS) -> np.ndarray:
    """Priority for oracle_matched: a horizon matched to each object's own load time.

    ``H = latency + bytes / bandwidth + margin``, so a texture that takes ten seconds
    to load is requested ten seconds further ahead than one that takes one. Objects
    further than H away score 0 and are left alone; they come into range later, as
    soonness grows, which is what makes this a horizon rather than a one-shot filter.
    """
    if bandwidth_bytes_per_s <= 0:
        raise ValueError("bandwidth must be positive")
    horizon = (float(latency_seconds)
               + np.asarray(size_bytes_per_row, dtype=np.float64) / float(bandwidth_bytes_per_s)
               + float(margin_seconds))
    with np.errstate(invalid="ignore"):
        soonness = 1.0 - np.asarray(time_to_visible, dtype=np.float64) / horizon
    in_range = np.isfinite(time_to_visible) & (time_to_visible <= horizon)
    score = np.where(in_range, np.clip(soonness, 0.0, 1.0), 0.0)
    return np.where(visible_now, VISIBLE_NOW_PRIORITY, score)


TAU_NAME = re.compile(r"^(?P<base>.+)_tau(?P<tau>[0-9]+(?:\.[0-9]+)?)$")


def parse_tau_name(name: str) -> tuple[str, float] | None:
    """Split ``heuristic_occ_tau0.1`` into ``("heuristic_occ", 0.1)``. None if not one."""
    match = TAU_NAME.match(name)
    if match is None:
        return None
    return match.group("base"), float(match.group("tau"))


def tau_policy_name(base: str, tau: float) -> str:
    return f"{base}_tau{float(tau):g}"


def expand_policy_names(config: dict) -> list[str]:
    """The config's policies, with the threshold variants appended in declaration order.

    Phase 4a's and 4b's configs have no ``prefetch_thresholds`` block, so for them this
    is exactly ``config["policies"]`` and their runs are untouched.
    """
    names = list(config["policies"])
    spec = config.get("prefetch_thresholds")
    if spec:
        for base in spec["bases"]:
            for tau in spec["taus"]:
                names.append(tau_policy_name(base, float(tau)))
    return names


def threshold_score(score: np.ndarray, tau: float) -> np.ndarray:
    """Zero anything that does not clear ``tau``. ``tau = 0`` is the identity.

    A row's score is 0 exactly when the policy does not want the object, so keeping
    only ``score > 0`` at tau = 0 reproduces the unthresholded policy bit for bit.
    """
    if tau <= 0.0:
        return np.asarray(score, dtype=np.float64)
    return np.where(np.asarray(score, dtype=np.float64) > tau, score, 0.0)


def build_policies(data, config: dict) -> dict[str, Policy]:
    """Every policy, as per-row arrays, in the order the config declares them."""
    names = expand_policy_names(config)
    scores = data.scores
    frustum = np.asarray(data.frustum_visible, dtype=bool)
    now_visible = occlusion_visible(frustum, data.occlusion_state)

    mlp_occlusion = np.maximum(scores["mlp_b__occlusion_0_5"],
                               scores["mlp_b__occlusion_1_0"])
    gbdt_occlusion = np.maximum(scores["gbdt_c__occlusion_0_5"],
                                scores["gbdt_c__occlusion_1_0"])

    future_visible = np.asarray(data.occlusion_visible_in_1_0s, dtype=bool)
    first_visible = np.asarray(data.occlusion_first_visible_time_1_0s, dtype=np.float64)
    # Sooner is better, and everything that becomes visible within the horizon must
    # outrank everything that does not. 0.5 + 0.5 * (1 - t/horizon) maps t=0 to 1.0
    # and t=horizon to 0.5, both above the 0.0 floor.
    soonness = np.clip(1.0 - first_visible / ORACLE_HORIZON_SECONDS, 0.0, 1.0)
    oracle_score = np.where(future_visible, 0.5 + 0.5 * soonness, 0.0)

    zero = np.zeros(len(frustum), dtype=np.float64)
    candidates = {
        "reactive_frustum": Policy(
            "reactive_frustum", "reactive: currently in frustum", frustum, zero, True),
        "reactive_visible": Policy(
            "reactive_visible", "reactive: currently occlusion-aware visible",
            now_visible, zero, True),
        "heuristic_frustum": Policy(
            "heuristic_frustum", "frozen heuristic, frustum-tuned",
            frustum, squash(scores["heuristic_frustum"]), True),
        "heuristic_occ": Policy(
            "heuristic_occ", "frozen heuristic, occlusion-tuned (occlusion-aware score)",
            now_visible, squash(scores["heuristic_occlusion"]), True),
        "mlp_b": Policy(
            "mlp_b", "MLP variant b: max of the two occlusion outputs",
            now_visible, mlp_occlusion.astype(np.float64), True),
        "gbdt_c": Policy(
            "gbdt_c", "HistGradientBoosting variant c: max of the two occlusion outputs",
            now_visible, gbdt_occlusion.astype(np.float64), True),
        "oracle_1s": Policy(
            "oracle_1s", "oracle: true occlusion-aware visibility within 1.0 s (upper bound)",
            now_visible, oracle_score, False),
        # Phase 4b's matched oracle. Its score band is filled in per configuration, so
        # it is built with score=None and is never accidentally used as a fixed array.
        "oracle_matched": Policy(
            "oracle_matched",
            "oracle: horizon matched to each object's own load time (upper bound)",
            now_visible, None, False, config_dependent=True),
    }
    # Phase 4a spelled this policy `oracle`. Keep the alias so sim/config.yaml still
    # reproduces sim/results/ byte-for-byte; the object is identical apart from its name.
    candidates["oracle"] = dataclasses.replace(candidates["oracle_1s"], name="oracle")

    if "reactive_visible_lag" in names:
        # Built only when asked for: the lagged state costs a pass over every row, and
        # the 4a/4b configs never request it.
        lag_seconds = float(config.get("reactive_lag_seconds", 0.2))
        sample_interval = float(config.get("sample_interval_seconds", 0.1))
        lag_samples = int(round(lag_seconds / sample_interval))
        lagged_visible = occlusion_visible(frustum, lagged_occlusion_state(data, lag_samples))
        candidates["reactive_visible_lag"] = Policy(
            "reactive_visible_lag",
            f"reactive: occlusion-aware visible using a {lag_seconds:g}s-stale occlusion buffer",
            lagged_visible, zero, True)

    for name in names:
        if name in candidates:
            continue
        parsed = parse_tau_name(name)
        if parsed is None:
            raise KeyError(f"config names policy {name!r}, which does not exist")
        base, tau = parsed
        if base not in candidates:
            raise KeyError(f"threshold policy {name!r} names unknown base {base!r}")
        base_policy = candidates[base]
        if base_policy.score is None:
            raise KeyError(f"threshold policy {name!r} has no thresholdable score")
        candidates[name] = dataclasses.replace(
            base_policy, name=name, label=f"{base_policy.label} @ tau={tau:g}",
            score=threshold_score(base_policy.score, tau))

    ordered: dict[str, Policy] = {}
    for name in names:
        if name not in candidates:
            raise KeyError(f"config names policy {name!r}, which does not exist")
        ordered[name] = candidates[name]
    return ordered


def deployable_names(policies: dict[str, Policy]) -> list[str]:
    """Policies that could run on device. The oracle is an upper bound, not an option."""
    return [name for name, policy in policies.items() if policy.deployable]
