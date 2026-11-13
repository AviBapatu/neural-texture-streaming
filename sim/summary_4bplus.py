"""Phase 4b+ summary: regime map, crossover table, tau curve, bootstrap CIs.

The decision rules were written into docs/phase-4bplus-report.md before the sweep ran.
This module applies them mechanically and prints the counts; it does not choose a
headline, and it reports absolute deficit reductions rather than "% of gap recovered".

Two things here are worth reading the code for:

* the bootstrap resamples **trajectories** and recomputes ``best_reactive`` *inside*
  each resample, so the interval is an interval for the same quantity the point
  estimate reports, rather than a fixed reference with error bars bolted on;
* the resample matrix is drawn **once** and reused for every cell (common random
  numbers), which both makes the run deterministic and stops cell-to-cell differences
  in the CI from being resampling noise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REACTIVE = ("reactive_frustum", "reactive_visible", "reactive_visible_lag")
DEFAULTS = {"prediction_relevant_gap": 0.02, "help_min_reduction": 0.01,
            "bootstrap_resamples": 1000, "bootstrap_seed": 20260501}
TAU_TIE_BREAK = 0.002


def validation_trajectories(config: dict) -> list[str]:
    """The split-v1 validation trajectories. Read, never hard-coded."""
    path = Path(config["paths"]["split"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return [entry["trajectory_id"] for entry in manifest["splits"]["validation"]]


def resample_matrix(n_trajectories: int, n_resamples: int, seed: int) -> np.ndarray:
    """The shared bootstrap index matrix. Fixed seed, so the whole report is reproducible."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_trajectories, size=(n_resamples, n_trajectories))


def reduction_ci(reactive_vectors: list[np.ndarray], policy: np.ndarray,
                 picks: np.ndarray) -> tuple[float, float, float]:
    """(point estimate, lo, hi) of best_reactive - policy, in absolute deficit.

    ``best_reactive`` is the per-resample minimum of the reactive baselines' means, so a
    cell where the baselines are close together has a wider interval - which is what
    should happen and would not if the reference were held fixed.
    """
    reactive_means = np.min(np.stack([vector[picks].mean(axis=1)
                                      for vector in reactive_vectors]), axis=0)
    policy_means = policy[picks].mean(axis=1)
    resampled = reactive_means - policy_means

    observed_reactive = min(float(vector.mean()) for vector in reactive_vectors)
    observed = observed_reactive - float(policy.mean())
    low, high = np.percentile(resampled, [2.5, 97.5])
    return observed, float(low), float(high)


def paired_ci(left: np.ndarray, right: np.ndarray,
              picks: np.ndarray) -> tuple[float, float, float]:
    """(point estimate, lo, hi) of mean(left - right), resampling trajectories."""
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    resampled = difference[picks].mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)


def excludes_zero(low: float, high: float) -> bool:
    return low > 0.0 or high < 0.0


def build_4bplus_summary(config: dict, data, policies, per_trajectory, peaks) -> str:
    reporting = config["reporting"]
    gap_threshold = float(reporting.get("prediction_relevant_gap",
                                        DEFAULTS["prediction_relevant_gap"]))
    min_reduction = float(reporting.get("help_min_reduction",
                                        DEFAULTS["help_min_reduction"]))
    n_resamples = int(reporting.get("bootstrap_resamples",
                                    DEFAULTS["bootstrap_resamples"]))
    seed = int(reporting.get("bootstrap_seed", DEFAULTS["bootstrap_seed"]))

    nominal = float(config["nominal_bandwidth_mb_s"])
    fractions = [float(v) for v in config["sweep"]["budget_fraction_of_peak_working_set"]]
    multipliers = [float(v) for v in config["sweep"]["size_multiplier"]]
    latencies = [float(v) for v in config["sweep"]["latency_seconds"]]
    scales = [float(v) for v in config["sweep"]["load_time_scale"]]
    threshold_spec = config["prefetch_thresholds"]
    bases = list(threshold_spec["bases"])
    taus = [float(v) for v in threshold_spec["taus"]]

    names = list(policies)
    reactive = [n for n in names if n in REACTIVE]
    deployable = [n for n in names if policies[n].deployable]
    labelled = [n for n in names if any(n.startswith(f"{b}_tau") for b in bases)]
    if not reactive:
        raise KeyError("the 4b+ summary needs at least one reactive baseline")
    if "oracle_matched" not in names:
        raise KeyError("the 4b+ summary needs the oracle_matched reference column")
    all_trajectories = [view.trajectory_id for view in data.views]
    validation = [t for t in validation_trajectories(config) if t in all_trajectories]
    picks = resample_matrix(len(all_trajectories), n_resamples, seed)

    cells = [(f, m, l, s) for f in fractions for m in multipliers
             for l in latencies for s in scales]

    def vector(cell, policy, trajectories):
        fraction, multiplier, latency, scale = cell
        bandwidth = nominal / scale
        return np.array([
            per_trajectory[(fraction, multiplier, latency, bandwidth, policy, t)]["deficit"]
            for t in trajectories], dtype=np.float64)

    def scalar(cell, policy, trajectories):
        return float(vector(cell, policy, trajectories).mean())

    def total(cell, policy, metric_name, trajectories):
        fraction, multiplier, latency, scale = cell
        bandwidth = nominal / scale
        return float(sum(
            per_trajectory[(fraction, multiplier, latency, bandwidth, policy, t)][metric_name]
            for t in trajectories))

    def label(cell):
        fraction, multiplier, latency, scale = cell
        return (f"{fraction:.0%} | {scale:g} | {latency:g}s | "
                f"{nominal / scale:g}MB/s")

    # --- relevance, per the pre-declared rule
    best_reactive_of = {cell: min(scalar(cell, p, all_trajectories) for p in reactive)
                        for cell in cells}
    oracle_of = {cell: scalar(cell, "oracle_matched", all_trajectories) for cell in cells}
    relevant = [cell for cell in cells
                if best_reactive_of[cell] - oracle_of[cell] >= gap_threshold - 1e-12]
    relevant_set = set(relevant)

    # --- helps, per the pre-declared rule
    def evaluate(cell, policy):
        observed, low, high = reduction_ci(
            [vector(cell, p, all_trajectories) for p in reactive],
            vector(cell, policy, all_trajectories), picks)
        return observed, low, high, excludes_zero(low, high) and observed >= min_reduction

    outcome = {name: {cell: evaluate(cell, name) for cell in cells} for name in names}

    lines = ["# Phase 4b+ sweep summary", "",
             "Offline trace-driven simulation. Deficit is the coverage-weighted fraction "
             "of visible object-steps whose full texture was not resident, excluding "
             "steps within +/-10 samples of a yaw snap. Reductions are **absolute deficit "
             "reductions** versus `best_reactive`; no percentage-of-gap figures appear "
             "anywhere in this report.", "",
             f"- Grid: {len(fractions)} budgets x {len(multipliers)} size x "
             f"{len(latencies)} latencies x {len(scales)} load-time scales = "
             f"**{len(cells)} configurations** x {len(names)} policies x 14 trajectories",
             f"- `load_time_scale` multiplies bytes/bandwidth; effective bandwidth is "
             f"{nominal:g} MB/s / scale, so scale 0.25 is four times faster than nominal.",
             "- The grid was declared in `sim/config_4bplus.yaml` before the sweep ran.",
             f"- Splits: `experimental_split_v1`. Validation trajectories "
             f"({len(validation)}): {', '.join(validation)}.", ""]

    lines += ["## Pre-declared decision rules (fixed before the sweep ran)", "",
              f"- `best_reactive` is the per-cell minimum deficit among "
              f"{', '.join('`%s`' % p for p in reactive)} - a reference, not a policy.",
              f"- A cell is **prediction-relevant** if "
              f"`best_reactive - oracle_matched >= {gap_threshold:g}`.",
              f"- A policy **helps** if the 95% bootstrap CI of its reduction versus "
              f"`best_reactive` excludes 0 **and** the point estimate is "
              f">= {min_reduction:g}.",
              f"- Bootstrap: {n_resamples} resamples over trajectories, seed {seed}, "
              f"`best_reactive` recomputed inside each resample.", ""]

    # --- regime map
    lines += ["## Regime map", "",
              "`reduction` and its CI are versus `best_reactive`, in absolute deficit; "
              "positive means the policy is better. `deployable beats oracle` names the "
              "best deployable policy when it materially beats `oracle_matched`, which is "
              "a reference column only.", "",
              "| budget | scale | latency | bandwidth | best_reactive | oracle_matched | "
              "reference gap | relevant | best deployable | reduction | 95% CI | helps | "
              "deployable beats oracle |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        best_name, best_value = min(
            ((n, scalar(cell, n, all_trajectories)) for n in deployable),
            key=lambda kv: kv[1])
        observed, low, high, helps = outcome[best_name][cell]
        gap = best_reactive_of[cell] - oracle_of[cell]
        beaten = [(n, oracle_of[cell] - scalar(cell, n, all_trajectories))
                  for n in deployable
                  if scalar(cell, n, all_trajectories) < oracle_of[cell] - 0.005]
        beaten.sort(key=lambda kv: -kv[1])
        flag = f"`{beaten[0][0]}` {beaten[0][1]:+.4f}" if beaten else "-"
        lines.append(
            f"| {label(cell)} | {best_reactive_of[cell]:.4f} | {oracle_of[cell]:.4f} | "
            f"{gap:+.4f} | {'**yes**' if cell in relevant_set else 'no'} | "
            f"`{best_name}` | {observed:+.4f} | [{low:+.4f}, {high:+.4f}] | "
            f"{'yes' if helps else 'no'} | {flag} |")
    lines.append("")

    lines += [f"**{len(relevant)} of {len(cells)}** configurations are "
              f"prediction-relevant (best_reactive - oracle_matched >= "
              f"{gap_threshold:g}).", ""]

    # --- decision rule outcome
    lines += ["## Decision rule outcome", "",
              "Number of prediction-relevant cells in which each policy helps, per the "
              "pre-declared rule (CI excludes 0 and reduction >= "
              f"{min_reduction:g}).", "",
              f"| policy | helps in (of {len(relevant)}) | median reduction over relevant |",
              "|---|---|---|"]
    for name in names:
        helped = [outcome[name][cell][0] for cell in relevant
                  if outcome[name][cell][3]]
        median = f"{float(np.median(helped)):+.4f}" if helped else "n/a"
        kind = "" if policies[name].deployable else " _(reference)_"
        lines.append(f"| `{name}`{kind} | **{len(helped)}** | {median} |")
    lines.append("")

    # --- crossover
    lines += ["## Crossover: the smallest latency at which prediction helps", "",
              "For each load-time scale, the smallest latency at which the policy helps "
              "(per the rule above) in **at least one of the three budgets**, and how "
              "many of the three it helps in. \"none in range\" means it helps at no "
              "latency tested, up to the 1.5 s maximum.", "",
              "| load_time_scale | bandwidth | " + " | ".join(f"`{n}`" for n in labelled)
              + " |",
              "|---|---|" + "---|" * len(labelled)]
    for scale in scales:
        cells_at = [c for c in cells if c[3] == scale]
        columns = []
        for name in labelled:
            crossing = None
            for latency in latencies:
                at = [c for c in cells_at if c[2] == latency]
                hits = [c for c in at if outcome[name][c][3]]
                if hits:
                    crossing = f"{latency:g}s ({len(hits)}/{len(at)})"
                    break
            columns.append(crossing or "none in range")
        lines.append(f"| {scale:g} | {nominal / scale:g}MB/s | " + " | ".join(columns) + " |")
    lines.append("")

    # --- mlp_b versus heuristic_occ
    lines += ["## Is `mlp_b` better than `heuristic_occ`?", "",
              "Per prediction-relevant cell, the 95% bootstrap CI of the paired "
              "difference `deficit(heuristic_occ) - deficit(mlp_b)`. Positive means "
              "`mlp_b` has the lower deficit.", "",
              "| policy pair | mlp_b reliably better | heuristic_occ reliably better | "
              "indistinguishable |",
              "|---|---|---|---|"]
    for tau in taus:
        left = f"heuristic_occ_tau{tau:g}"
        right = f"mlp_b_tau{tau:g}"
        if left not in names or right not in names:
            continue
        better = worse = same = 0
        for cell in relevant:
            _, low, high = paired_ci(vector(cell, left, all_trajectories),
                                     vector(cell, right, all_trajectories), picks)
            if low > 0.0:
                better += 1
            elif high < 0.0:
                worse += 1
            else:
                same += 1
        lines.append(f"| tau={tau:g}: `heuristic_occ` vs `mlp_b` | {better} | {worse} | "
                     f"{same} |")
    lines.append("")

    # --- tau curve
    lines += ["## The tau curve: what a threshold costs", "",
              "Aggregated over the prediction-relevant configurations, all 14 "
              "trajectories. `deficit` is the mean deficit, `bytes` and `wasted` are "
              "summed over those cells. `validation deficit` is the same mean computed "
              "on the three split-v1 validation trajectories only - that is the column "
              "the recommendation is allowed to look at.", "",
              "| base | tau | deficit (all 14) | validation deficit | bytes loaded (GB) | "
              "wasted prefetches |",
              "|---|---|---|---|---|---|"]
    recommendations: dict[str, float] = {}
    for base in bases:
        by_tau: dict[float, float] = {}
        for tau in taus:
            name = f"{base}_tau{tau:g}"
            if name not in names:
                continue
            if not relevant:
                lines.append(f"| `{base}` | {tau:g} | n/a | n/a | n/a | n/a |")
                continue
            mean_all = float(np.mean([scalar(c, name, all_trajectories)
                                      for c in relevant]))
            mean_val = float(np.mean([scalar(c, name, validation) for c in relevant]))
            by_tau[tau] = mean_val
            loaded = sum(total(c, name, "bytes_streamed", all_trajectories)
                         for c in relevant)
            wasted = sum(total(c, name, "wasted_prefetches", all_trajectories)
                         for c in relevant)
            lines.append(f"| `{base}` | {tau:g} | {mean_all:.4f} | {mean_val:.4f} | "
                         f"{loaded / 1e9:.2f} | {int(wasted)} |")
        if by_tau:
            # The argmin itself is always eligible, so this cannot come back empty.
            best = min(by_tau.values())
            eligible = sorted(t for t, d in by_tau.items() if d <= best + TAU_TIE_BREAK)
            recommendations[base] = eligible[0]
        lines.append("")

    lines += ["### Recommended tau", ""]
    if not recommendations:
        lines += ["No threshold is recommended: there are no prediction-relevant cells "
                  "in this grid, so there is nothing for a threshold to trade against.", ""]
    else:
        lines += [f"Chosen on the validation trajectories only: the tau with the lowest "
                  f"mean deficit over the prediction-relevant cells, ties within "
                  f"{TAU_TIE_BREAK:g} broken toward the lower tau (less speculative "
                  f"traffic for the same deficit).", ""]
        for base, tau in recommendations.items():
            name = f"{base}_tau{tau:g}"
            mean_val = float(np.mean([scalar(c, name, validation) for c in relevant]))
            mean_all = float(np.mean([scalar(c, name, all_trajectories)
                                      for c in relevant]))
            helped = sum(1 for c in relevant if outcome[name][c][3])
            lines.append(
                f"- `{base}`: **tau = {tau:g}**. Validation deficit {mean_val:.4f}, "
                f"all-14 deficit {mean_all:.4f} (difference {mean_all - mean_val:+.4f}), "
                f"helps in {helped} of {len(relevant)} relevant cells.")
        lines.append("")

    # --- per-trajectory at the representative configs
    lines += ["## Per-trajectory deficit at representative configurations", ""]
    for entry in reporting["representative_configs"]:
        # size is held at a single value in this phase, so a representative cell may
        # leave it out; the pre-declared config does exactly that.
        cell = (float(entry["budget_fraction"]),
                float(entry.get("size_multiplier", multipliers[0])),
                float(entry["latency_seconds"]), float(entry["load_time_scale"]))
        lines += [f"### {label(cell)}", "",
                  "| trajectory | peak WS (MB) | " + " | ".join(f"`{n}`" for n in names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for trajectory in all_trajectories:
            peak = peaks[cell[1]][trajectory]
            values = " | ".join(
                f"{float(vector(cell, n, [trajectory])[0]):.4f}" for n in names)
            lines.append(f"| {trajectory} | {peak / 1e6:.1f} | {values} |")
        lines.append("")

    return "\n".join(lines) + "\n"
