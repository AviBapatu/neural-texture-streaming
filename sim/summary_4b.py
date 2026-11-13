"""Phase 4b summary: the regime map and the pre-declared decision rules, applied.

The decision rules were written into docs/phase-4b-report.md before the sweep ran. This
module applies them mechanically and prints the counts; it does not choose a headline.

Rules, from the config:

* **prediction-relevant**: ``deficit(reactive_visible) - deficit(oracle_matched) >= gap``
  (default 0.02) on the mean over trajectories.
* **helps**: within prediction-relevant configurations, ``recovered >= 0.25`` and
  ``wins >= 10`` of 14 trajectories, where ``recovered`` is the share of the
  reactive_visible -> oracle_matched gap the policy closes.
"""

from __future__ import annotations

import numpy as np

DEFAULTS = {"prediction_relevant_gap": 0.02, "help_recovery_fraction": 0.25,
            "help_min_trajectories": 10}
TOLERANCE = 1e-9


def _policy_labels(names: list[str]) -> list[str]:
    return list(names)


def build_regime_summary(config: dict, data, policies, per_trajectory, peaks) -> str:
    reporting = config["reporting"]
    gap_threshold = float(reporting.get("prediction_relevant_gap",
                                        DEFAULTS["prediction_relevant_gap"]))
    recovery_threshold = float(reporting.get("help_recovery_fraction",
                                             DEFAULTS["help_recovery_fraction"]))
    min_trajectories = int(reporting.get("help_min_trajectories",
                                         DEFAULTS["help_min_trajectories"]))

    fractions = [float(v) for v in config["sweep"]["budget_fraction_of_peak_working_set"]]
    multipliers = [float(v) for v in config["sweep"]["size_multiplier"]]
    latencies = [float(v) for v in config["sweep"]["latency_seconds"]]
    bandwidths = [float(v) for v in config["sweep"]["bandwidth_mb_s"]]
    names = list(policies)
    deployable = [n for n in names if policies[n].deployable]
    trajectories = [view.trajectory_id for view in data.views]
    ideal = bool(config.get("ideal_budget", False))

    cells = [(f, m, l, b) for f in fractions for m in multipliers
             for l in latencies for b in bandwidths]
    ideal_label = float("inf")

    def deficit(cell, policy, trajectory, kind="fraction"):
        fraction = ideal_label if kind == "ideal" else cell[0]
        _, multiplier, latency, bandwidth = cell
        return per_trajectory[(fraction, multiplier, latency, bandwidth, policy,
                               trajectory)]["deficit"]

    def mean_deficit(cell, policy, kind="fraction"):
        return float(np.mean([deficit(cell, policy, t, kind) for t in trajectories]))

    def series(cell, policy, metric_name, kind="fraction"):
        fraction = ideal_label if kind == "ideal" else cell[0]
        _, multiplier, latency, bandwidth = cell
        return [per_trajectory[(fraction, multiplier, latency, bandwidth, policy, t)][metric_name]
                for t in trajectories]

    def label(cell):
        fraction, multiplier, latency, bandwidth = cell
        return (f"{fraction:.0%} | {multiplier:g}x | {latency:g}s | {bandwidth:g}MB/s")

    lines = ["# Phase 4b sweep summary", "",
             "Offline trace-driven simulation, regime chosen to favour prediction: memory "
             "is sufficient (100/150/300% of peak working set) and loads are slow "
             "(50/200 MB/s, 0.1-1.0 s latency). Deficit is the coverage-weighted fraction "
             "of visible object-steps whose full texture was not resident, excluding "
             "steps within +/-10 samples of a yaw snap.", "",
             f"- Grid: {len(fractions)} budgets x {len(multipliers)} sizes x "
             f"{len(latencies)} latencies x {len(bandwidths)} bandwidths = **{len(cells)} "
             f"configurations** x {len(names)} policies x 14 trajectories",
             "- The grid was declared in `sim/config_4b.yaml` before the sweep ran.",
             "- `oracle_1s` and `oracle_matched` are upper-bound diagnostics, never "
             "deployable policies.", "",
             "## Pre-declared decision rules (fixed before the sweep ran)", "",
             f"- A configuration is **prediction-relevant** if "
             f"`deficit(reactive_visible) - deficit(oracle_matched) >= {gap_threshold:g}`.",
             f"- Within those, a deployable policy **helps** if it recovers "
             f"`>= {recovery_threshold:.0%}` of that gap and beats `reactive_visible` on "
             f"`>= {min_trajectories}` of 14 trajectories.", ""]

    # --- full deficit grid
    lines += ["## Deficit by policy and configuration (mean over 14 trajectories)", "",
              "Lower is better.", "",
              "| budget | size | latency | bandwidth | " + " | ".join(names) + " |",
              "|---|---|---|---|" + "---|" * len(names)]
    for cell in cells:
        values = " | ".join(f"{mean_deficit(cell, n):.4f}" for n in names)
        fraction, multiplier, latency, bandwidth = cell
        lines.append(f"| {fraction:.0%} | {multiplier:g}x | {latency:g}s | "
                     f"{bandwidth:g}MB/s | {values} |")
    lines.append("")

    # --- regime map
    lines += ["## Regime map: where prediction has room", "",
              "The `gap` column is the ceiling on what any predictor could recover in "
              "that configuration - the part of the deficit caused purely by "
              "`reactive_visible` being unable to see the future. `best deployable` is "
              "the lowest-deficit deployable policy of that configuration; it is chosen "
              "per configuration and is therefore a *selected* best, not a single policy "
              "that would be deployed.", "",
              "| budget | size | latency | bandwidth | reactive_visible | oracle_matched | "
              "gap | prediction-relevant | best deployable | its deficit | recovered |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    relevant: list[tuple] = []
    for cell in cells:
        rv = mean_deficit(cell, "reactive_visible")
        om = mean_deficit(cell, "oracle_matched")
        gap = rv - om
        best_name, best_value = min(
            ((n, mean_deficit(cell, n)) for n in deployable), key=lambda kv: kv[1])
        recovered = "n/a" if gap <= TOLERANCE else f"{(rv - best_value) / gap:+.1%}"
        marker = "**yes**" if gap >= gap_threshold - TOLERANCE else "no"
        if gap >= gap_threshold - TOLERANCE:
            relevant.append(cell)
        lines.append(f"| {label(cell)} | {rv:.4f} | {om:.4f} | {gap:+.4f} | {marker} | "
                     f"`{best_name}` | {best_value:.4f} | {recovered} |")
    lines.append("")

    # --- prediction-relevant count, by bandwidth and latency
    lines += ["### Where the prediction-relevant configurations are", "",
              "| bandwidth | latency | relevant / total |", "|---|---|---|"]
    for bandwidth in bandwidths:
        for latency in latencies:
            subset = [c for c in cells if c[2] == latency and c[3] == bandwidth]
            hits = [c for c in subset if mean_deficit(c, "reactive_visible")
                    - mean_deficit(c, "oracle_matched") >= gap_threshold - TOLERANCE]
            lines.append(f"| {bandwidth:g}MB/s | {latency:g}s | {len(hits)} / {len(subset)} |")
    lines.append("")
    lines += [f"**{len(relevant)} of {len(cells)}** configurations are "
              f"prediction-relevant (gap >= {gap_threshold:g}).", ""]

    # --- decision rule outcome
    lines += ["## Decision rule outcome", "",
              f"Applied to the {len(relevant)} prediction-relevant configurations, per "
              f"Rule 2: recover >= {recovery_threshold:.0%} of the gap **and** beat "
              f"`reactive_visible` on >= {min_trajectories} of 14 trajectories.", "",
              "| policy | helps in (of " + str(len(relevant)) + " relevant) "
              "| of all 54 | median recovered |",
              "|---|---|---|---|"]
    for name in deployable:
        helped = 0
        recoveries = []
        for cell in relevant:
            rv = mean_deficit(cell, "reactive_visible")
            om = mean_deficit(cell, "oracle_matched")
            gap = rv - om
            value = mean_deficit(cell, name)
            recovered = (rv - value) / gap if gap > TOLERANCE else 0.0
            wins = sum(1 for t in trajectories
                       if deficit(cell, name, t) < deficit(cell, "reactive_visible", t)
                       - TOLERANCE)
            recoveries.append(recovered)
            if recovered >= recovery_threshold - TOLERANCE and wins >= min_trajectories:
                helped += 1
        median = float(np.median(recoveries)) if recoveries else float("nan")
        lines.append(f"| `{name}` | **{helped}** | {helped} / {len(cells)} | "
                     f"{median:+.1%} |")
    lines.append("")

    # --- cost of predicting
    lines += ["## Cost of predicting, in the prediction-relevant configurations", "",
              "Aggregated over the prediction-relevant configurations only, summed over "
              "14 trajectories. Wasted prefetches are objects loaded and evicted without "
              "ever being visible.", "",
              "| policy | wasted prefetches | bytes loaded (GB) | loads completed |",
              "|---|---|---|---|"]
    for name in names:
        wasted = sum(sum(series(c, name, "wasted_prefetches")) for c in relevant)
        loaded = sum(sum(series(c, name, "bytes_streamed")) for c in relevant)
        loads = sum(sum(series(c, name, "loads_completed")) for c in relevant)
        lines.append(f"| `{name}` | {wasted} | {loaded / 1e9:.2f} | {loads} |")
    lines.append("")

    # --- ideal budget
    if ideal:
        lines += ["## Ideal budget: capacity-caused vs latency-caused deficit", "",
                  "The same policies under an infinite budget. What remains is the "
                  "deficit that latency and bandwidth cause by themselves, with capacity "
                  "removed from the picture. `capacity share` is the fraction of the "
                  "finite-budget deficit that a sufficient budget would have removed.", "",
                  "| budget | size | latency | bandwidth | reactive_visible | "
                  "reactive_visible ideal | oracle_matched | oracle_matched ideal | "
                  "capacity share |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for cell in cells:
            rv = mean_deficit(cell, "reactive_visible")
            rv_ideal = mean_deficit(cell, "reactive_visible", "ideal")
            om = mean_deficit(cell, "oracle_matched")
            om_ideal = mean_deficit(cell, "oracle_matched", "ideal")
            share = ((rv - rv_ideal) / rv) if rv > TOLERANCE else 0.0
            lines.append(f"| {label(cell)} | {rv:.4f} | {rv_ideal:.4f} | {om:.4f} | "
                         f"{om_ideal:.4f} | {share:.1%} |")
        lines.append("")
        finite = np.mean([mean_deficit(c, "reactive_visible") for c in cells])
        infinite = np.mean([mean_deficit(c, "reactive_visible", "ideal") for c in cells])
        lines += [f"Mean `reactive_visible` deficit: **{finite:.4f}** at the swept budgets, "
                  f"**{infinite:.4f}** with an infinite budget. So "
                  f"**{infinite / finite:.0%}** of the deficit in this memory-rich regime "
                  f"is caused by latency and bandwidth alone - it is the irreducible part "
                  f"that no policy can remove by holding more, only by asking earlier.", ""]

    # --- per-trajectory at the representative configs
    lines += ["## Per-trajectory deficit at representative configurations", ""]
    for entry in config["reporting"]["representative_configs"]:
        cell = (float(entry["budget_fraction"]), float(entry["size_multiplier"]),
                float(entry["latency_seconds"]), float(entry["bandwidth_mb_s"]))
        lines += [f"### {label(cell)}", "",
                  "| trajectory | peak WS (MB) | " + " | ".join(names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for trajectory in trajectories:
            peak = peaks[cell[1]][trajectory]
            values = " | ".join(f"{deficit(cell, n, trajectory):.4f}" for n in names)
            lines.append(f"| {trajectory} | {peak / 1e6:.1f} | {values} |")
        lines.append("")

    return "\n".join(lines) + "\n"
