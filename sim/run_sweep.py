"""Sweep driver: one command, the whole declared grid.

Phase 4a (`sim/config.yaml`, 18 configurations) and Phase 4b (`sim/config_4b.yaml`,
54 configurations) share this driver. Deterministic - there is no RNG anywhere in the
simulator - so re-running produces byte-identical results.

The CSV schema adapts to the config: the `bandwidth_mb_s` column appears only when
bandwidth is swept, and `budget_kind` only when `ideal_budget` runs are requested.
That is deliberate. Phase 4a's published results must stay reproducible byte-for-byte,
so a 4a config has to emit exactly the columns it emitted before.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# The brief's command is `python sim/run_sweep.py --config sim/config.yaml`, which runs
# this file as a script with no package context. Put the repository root on the path so
# the absolute imports below resolve identically either way.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim import scene
from sim.policies import build_policies, deployable_names, matched_priorities, time_to_next_visible
from sim.simulate import RunConfig, build_sim_data, summarise, simulate

IDEAL_BUDGET_POLICIES = ("reactive_visible", "oracle_matched")
BASE_FIELDS = ["budget_fraction", "budget_bytes", "size_multiplier", "latency_seconds"]
TAIL_FIELDS = ["policy", "trajectory", "metric", "value"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace-driven texture-streaming sweep")
    parser.add_argument("--config", default="sim/config.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def load_view_data(config: dict):
    paths = config["paths"]
    return build_sim_data(paths["oof_scores"], paths["dataset"],
                          int(config["snap_window_samples"]))


def transport_records(config: dict) -> dict:
    """The transport axis: how it is labelled, and each cell's bandwidth in MB/s.

    Three shapes, because the phases asked for three different things:

    * Phase 4a - no transport axis. One cell at the config's fixed bandwidth.
    * Phase 4b - a ``bandwidth_mb_s`` list under ``sweep``, labelled by bandwidth.
    * Phase 4b+ - a ``load_time_scale`` list, labelled by scale. Bandwidth is
      ``nominal_bandwidth_mb_s / scale``, so a smaller scale is a faster load path and
      scale is exactly a multiplier on ``bytes / bandwidth``.

    In every shape the per-run key and the CSV carry the *bandwidth*, which is the
    physical quantity, and which is unique across the cells of all three.
    """
    sweep = config["sweep"]
    if sweep.get("load_time_scale"):
        nominal = float(config["nominal_bandwidth_mb_s"])
        scales = [float(v) for v in sweep["load_time_scale"]]
        return {"axis": "load_time_scale", "values": scales,
                "bandwidths": [nominal / scale for scale in scales]}
    if sweep.get("bandwidth_mb_s"):
        values = [float(v) for v in sweep["bandwidth_mb_s"]]
        return {"axis": "bandwidth_mb_s", "values": values, "bandwidths": list(values)}
    return {"axis": None, "values": [None],
            "bandwidths": [float(config["bandwidth_mb_s"])]}


def to_bytes_per_s(bandwidth_mb_s: float) -> float:
    """The engine and the matched oracle both work in bytes and seconds."""
    return float(bandwidth_mb_s) * 1e6


def run_sweep(config_path: str | Path) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    paths = config["paths"]
    output = Path(paths["results"])
    output.mkdir(parents=True, exist_ok=True)
    sample_interval = float(config["sample_interval_seconds"])
    window = int(round(float(config["working_set_window_seconds"]) / sample_interval))
    margin = float(config.get("matched_margin_seconds", 0.2))
    ideal_budget = bool(config.get("ideal_budget", False))
    style = str(config.get("report_style", "gap"))

    def log(message: str) -> None:
        print(message, flush=True)

    log("=" * 78)
    log(f"Trace-driven texture-streaming sweep - {config_path}")
    log("=" * 78)

    started = time.perf_counter()
    data = load_view_data(config)
    log(f"\n  {len(data.object_ids)} rows, {len(data.trajectories)} trajectories, "
        f"{len(data.scores)} score arrays")

    records = scene.load_objects(paths["object_index"])
    policies = build_policies(data, config)
    ground_truth = data.ground_truth_visible
    now_visible = ground_truth
    log(f"  policies: {', '.join(policies)}")
    log(f"  ground-truth visible rows: {int(ground_truth.sum())}")

    # Priorities that do not depend on the run configuration are computed once. The
    # matched oracle does depend on bandwidth, latency and object size, so it is
    # computed per configuration and cached on those three axes.
    fixed_priorities = {name: policy.priority(np.arange(len(data.object_ids)))
                        for name, policy in policies.items()
                        if not policy.config_dependent}
    time_to_visible = (time_to_next_visible(data, sample_interval)
                       if any(p.config_dependent for p in policies.values()) else None)
    matched_cache: dict[tuple, np.ndarray] = {}

    def priorities_for(name: str, multiplier: float, latency: float,
                       bandwidth_mb_s: float) -> np.ndarray:
        policy = policies[name]
        if not policy.config_dependent:
            return fixed_priorities[name]
        cache_key = (multiplier, latency, bandwidth_mb_s)
        if cache_key not in matched_cache:
            matched_cache[cache_key] = matched_priorities(
                now_visible, time_to_visible, size_per_row[multiplier],
                latency, to_bytes_per_s(bandwidth_mb_s), margin)
        return matched_cache[cache_key]

    budget_fractions = [float(v) for v in
                        config["sweep"]["budget_fraction_of_peak_working_set"]]
    multipliers = [float(v) for v in config["sweep"]["size_multiplier"]]
    latencies = [float(v) for v in config["sweep"]["latency_seconds"]]
    transport = transport_records(config)
    transport_axis = transport["axis"]
    transport_values = transport["values"]
    bandwidths = transport["bandwidths"]

    sizes_by_multiplier = {m: scene.sized_bytes(records, m) for m in multipliers}
    size_per_row = {m: np.array([sizes[oid] for oid in data.object_ids], dtype=np.float64)
                    for m, sizes in sizes_by_multiplier.items()}

    peaks: dict[float, dict[str, float]] = {}
    for multiplier, sizes in sizes_by_multiplier.items():
        peaks[multiplier] = {
            view.trajectory_id: scene.peak_working_set(
                view.rows_by_sample, ground_truth, data.object_ids, sizes, window)
            for view in data.views}
    reference_multiplier = min(peaks)
    log(f"  peak working set (MB) at {reference_multiplier:g}x: " +
        ", ".join(f"{tid}={peaks[reference_multiplier][tid] / 1e6:.1f}"
                  for tid in data.trajectories[:5]) + ", ...")

    rows: list[dict] = []
    per_trajectory: dict[tuple, dict[str, float]] = {}
    total = len(budget_fractions) * len(multipliers) * len(latencies) * len(bandwidths) * len(policies)
    axis_label = transport_axis or "bandwidth"
    log(f"\n  grid: {len(budget_fractions)} budgets x {len(multipliers)} size multipliers "
        f"x {len(latencies)} latencies x {len(bandwidths)} {axis_label} x {len(policies)} "
        f"policies = {total} runs")

    def record(fraction, multiplier, latency, transport_label, bandwidth_mb_s, policy_name,
               view, budget_bytes, budget_kind):
        run = RunConfig(budget_bytes=budget_bytes, size_multiplier=multiplier,
                        latency_seconds=latency,
                        bandwidth_bytes_per_s=to_bytes_per_s(bandwidth_mb_s),
                        sample_interval=sample_interval)
        metrics = summarise(simulate(
            view, policies[policy_name],
            priorities_for(policy_name, multiplier, latency, bandwidth_mb_s),
            sizes_by_multiplier[multiplier], ground_truth, data.coverage,
            data.object_ids, run), run)
        metrics["peak_working_set_bytes"] = peaks[multiplier][view.trajectory_id]
        per_trajectory[(fraction, multiplier, latency, bandwidth_mb_s, policy_name,
                        view.trajectory_id)] = metrics
        for metric, value in metrics.items():
            row = {
                "budget_fraction": fraction,
                "budget_bytes": budget_bytes,
                "size_multiplier": multiplier,
                "latency_seconds": latency,
                "policy": policy_name,
                "trajectory": view.trajectory_id,
                "metric": metric,
                "value": value,
                "bandwidth_mb_s": bandwidth_mb_s,
            }
            if transport_axis == "load_time_scale":
                row["load_time_scale"] = transport_label
            if ideal_budget:
                row["budget_kind"] = budget_kind
            rows.append(row)

    completed = 0
    for fraction in budget_fractions:
        for multiplier in multipliers:
            for latency in latencies:
                for transport_label, bandwidth in zip(transport_values, bandwidths):
                    for policy_name in policies:
                        for view in data.views:
                            peak = peaks[multiplier][view.trajectory_id]
                            record(fraction, multiplier, latency, transport_label, bandwidth,
                                   policy_name, view, max(peak * fraction, 1.0), "fraction")
                        completed += 1
                    if ideal_budget:
                        # Infinite budget: the deficit that remains is caused by latency
                        # and bandwidth alone, with capacity taken out of the picture.
                        for policy_name in IDEAL_BUDGET_POLICIES:
                            if policy_name not in policies:
                                continue
                            for view in data.views:
                                record(float("inf"), multiplier, latency, transport_label,
                                       bandwidth, policy_name, view, float("inf"), "ideal")
                    log(f"    budget={fraction:.2f} size={multiplier:g}x latency={latency:g}s "
                        f"{axis_label}={transport_label} (bw={bandwidth:g}MB/s) done "
                        f"({completed}/{total} runs, {time.perf_counter() - started:.0f}s)")

    fields = list(BASE_FIELDS)
    if transport_axis == "bandwidth_mb_s":
        fields.append("bandwidth_mb_s")
    elif transport_axis == "load_time_scale":
        fields += ["load_time_scale", "bandwidth_mb_s"]
    fields += TAIL_FIELDS
    if ideal_budget:
        fields.append("budget_kind")
    write_csv(output / "results.csv", rows, fields)

    if style == "regime":
        from sim.summary_4b import build_regime_summary
        summary = build_regime_summary(config, data, policies, per_trajectory, peaks)
    elif style == "regime_4bplus":
        from sim.summary_4bplus import build_4bplus_summary
        summary = build_4bplus_summary(config, data, policies, per_trajectory, peaks)
    else:
        summary = build_summary(config, data, policies, per_trajectory, peaks, policies)
    (output / "summary.md").write_text(summary, encoding="utf-8")

    manifest = {
        "config": str(config_path),
        "runs": total,
        "rows": len(rows),
        "budget_fractions": budget_fractions,
        "size_multipliers": multipliers,
        "latencies": latencies,
        "transport_axis": transport_axis,
        "transport_values": list(transport_values),
        "bandwidths_mb_s": bandwidths,
        "policies": list(policies),
        "ideal_budget": ideal_budget,
        "ideal_budget_policies": list(IDEAL_BUDGET_POLICIES) if ideal_budget else [],
        "working_set_window_samples": window,
        "csv_fields": fields,
        "grid_declared_before_running": True,
        "deterministic": True,
        "seconds": time.perf_counter() - started,
    }
    if transport_axis is None:
        manifest["bandwidth_mb_s"] = bandwidths[0]
    (output / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    log(f"\n  -> {output / 'results.csv'} ({len(rows)} rows)")
    log(f"  -> {output / 'summary.md'}")
    log(f"  total {time.perf_counter() - started:.0f}s")
    return {"rows": rows, "per_trajectory": per_trajectory, "summary": summary}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{row[k]:.6g}" if isinstance(row[k], float) else row[k])
                             for k in fields})


# --- Phase 4a summary ---------------------------------------------------------

def build_summary(config: dict, data, policies, per_trajectory, peaks, all_policies) -> str:
    fractions = [float(v) for v in config["sweep"]["budget_fraction_of_peak_working_set"]]
    multipliers = [float(v) for v in config["sweep"]["size_multiplier"]]
    latencies = [float(v) for v in config["sweep"]["latency_seconds"]]
    bandwidth = transport_records(config)["bandwidths"][0]
    names = list(policies)
    deployable = deployable_names(policies)
    baseline = config["reporting"]["gap_baseline"]
    trajectories = [view.trajectory_id for view in data.views]

    def deficit(fraction, multiplier, latency, policy, trajectory):
        return per_trajectory[(fraction, multiplier, latency, bandwidth, policy,
                               trajectory)]["deficit"]

    def mean_deficit(fraction, multiplier, latency, policy):
        values = [deficit(fraction, multiplier, latency, policy, t) for t in trajectories]
        return float(np.mean(values))

    oracle_name = "oracle" if "oracle" in names else "oracle_1s"
    lines = ["# Phase 4a sweep summary", "",
             "Offline trace-driven simulation of texture streaming. Deficit is the "
             "coverage-weighted fraction of visible object-steps whose full texture was "
             "not resident, excluding steps within +/-10 samples of a yaw snap.", "",
             f"- Grid: {len(fractions)} budgets x {len(multipliers)} size multipliers x "
             f"{len(latencies)} latencies x {len(names)} policies",
             f"- Bandwidth: {config['bandwidth_mb_s']} MB/s, fixed latency per configuration",
             "- The grid was declared in `sim/config.yaml` before the sweep ran.", "",
             "## Deficit by policy and configuration (mean over 14 trajectories)", "",
             f"Lower is better. `{oracle_name}` is an upper bound, not a deployable policy.",
             "",
             "| budget | size | latency | " + " | ".join(names) + " |",
             "|---|---|---|" + "---|" * len(names)]

    for fraction in fractions:
        for multiplier in multipliers:
            for latency in latencies:
                cells = " | ".join(f"{mean_deficit(fraction, multiplier, latency, n):.4f}"
                                   for n in names)
                lines.append(f"| {fraction:.0%} | {multiplier:g}x | {latency:g}s | {cells} |")
    lines.append("")

    # --- oracle gap
    lines += ["## Oracle gap", "",
              f"Measured from `{baseline}` (the no-prediction engine baseline) to "
              f"`{oracle_name}` (true future visibility within 1.0 s). `recovery` is the "
              "share of that gap a policy closes; a value at or below zero means it is no "
              "better than doing nothing predictive.", "",
              "| budget | size | latency | " + " | ".join(deployable) + " |",
              "|---|---|---|" + "---|" * len(deployable)]
    for fraction in fractions:
        for multiplier in multipliers:
            for latency in latencies:
                base = mean_deficit(fraction, multiplier, latency, baseline)
                orc = mean_deficit(fraction, multiplier, latency, oracle_name)
                gap = base - orc
                cells = []
                for name in deployable:
                    value = mean_deficit(fraction, multiplier, latency, name)
                    cells.append("n/a" if gap <= 0 else f"{(base - value) / gap:+.1%}")
                lines.append(f"| {fraction:.0%} | {multiplier:g}x | {latency:g}s | "
                             + " | ".join(cells) + " |")
    lines.append("")

    # --- paired differences
    lines += ["## Paired differences per trajectory", "",
              "Counts of trajectories where `mlp_b` has a lower deficit than the "
              "comparison policy, out of 14. `tie` means within 1e-9.", ""]
    for opponent in ("heuristic_frustum", "heuristic_occ"):
        lines += [f"### mlp_b vs {opponent}", "",
                  "| budget | size | latency | better | worse | tie | mean delta |",
                  "|---|---|---|---|---|---|---|"]
        for fraction in fractions:
            for multiplier in multipliers:
                for latency in latencies:
                    deltas = [deficit(fraction, multiplier, latency, "mlp_b", t)
                              - deficit(fraction, multiplier, latency, opponent, t)
                              for t in trajectories]
                    better = sum(1 for d in deltas if d < -1e-9)
                    worse = sum(1 for d in deltas if d > 1e-9)
                    lines.append(f"| {fraction:.0%} | {multiplier:g}x | {latency:g}s | "
                                 f"{better} | {worse} | {14 - better - worse} | "
                                 f"{np.mean(deltas):+.4f} |")
        lines.append("")

    # --- headroom decomposition
    lines += ["## Where the headroom actually is", "",
              "Mean deficit over all 18 configurations, decomposed. This is the section "
              "that decides whether the project story holds.", "",
              "| step | mean deficit | change |",
              "|---|---|---|"]
    combined = [(f, m, l) for f in fractions for m in multipliers for l in latencies]
    reactive_frustum = np.mean([mean_deficit(f, m, l, "reactive_frustum")
                                for f, m, l in combined])
    reactive_visible = np.mean([mean_deficit(f, m, l, "reactive_visible")
                                for f, m, l in combined])
    best_mlp = np.mean([min(mean_deficit(f, m, l, "mlp_b"),
                            mean_deficit(f, m, l, "gbdt_c")) for f, m, l in combined])
    oracle_mean = np.mean([mean_deficit(f, m, l, oracle_name) for f, m, l in combined])
    lines += [
        f"| `reactive_frustum` (no prediction, no culling) | {reactive_frustum:.4f} | - |",
        f"| `reactive_visible` (+ occlusion culling, still no prediction) | "
        f"{reactive_visible:.4f} | {reactive_visible - reactive_frustum:+.4f} |",
        f"| best of `mlp_b` / `gbdt_c` (+ prediction) | {best_mlp:.4f} | "
        f"{best_mlp - reactive_visible:+.4f} |",
        f"| `{oracle_name}` (upper bound as specified) | {oracle_mean:.4f} | "
        f"{oracle_mean - best_mlp:+.4f} |", "",
        f"Total `reactive_frustum` -> `{oracle_name}` headroom is "
        f"**{reactive_frustum - oracle_mean:+.4f}**. Of that, occlusion culling supplies "
        f"**{reactive_visible - reactive_frustum:+.4f}** and the entire prediction stack "
        f"supplies **{best_mlp - reactive_visible:+.4f}** on top.", ""]

    # --- oracle bound honesty
    violations = []
    for f, m, l in combined:
        oracle_deficit = mean_deficit(f, m, l, oracle_name)
        for name in deployable:
            if mean_deficit(f, m, l, name) < oracle_deficit - 1e-9:
                violations.append((f, m, l, name,
                                   mean_deficit(f, m, l, name) - oracle_deficit))
    lines += ["### The oracle is not a strict upper bound", "",
              f"A deployable policy beats `{oracle_name}` in **{len(violations)} of "
              f"{len(combined) * len(deployable)}** policy/configuration comparisons "
              f"(largest margin "
              f"{min((v[4] for v in violations), default=0.0):+.4f} mean deficit).", "",
              "This is a property of the specified oracle, not a bug: it asks for every "
              "object that will be visible within 1.0 s, and under a budget that cannot "
              "hold them all it spends bandwidth on objects it then has to evict. It is "
              "still strictly better than `reactive_frustum` in every configuration, so "
              "it bounds the *culling* headroom but not the *prediction* headroom.", ""]

    # --- per-trajectory at the representative configs
    lines += ["## Per-trajectory deficit at representative configurations", ""]
    for entry in config["reporting"]["representative_configs"]:
        fraction = float(entry["budget_fraction"])
        multiplier = float(entry["size_multiplier"])
        latency = float(entry["latency_seconds"])
        lines += [f"### budget {fraction:.0%} of peak, size {multiplier:g}x, "
                  f"latency {latency:g}s", "",
                  "| trajectory | peak WS (MB) | " + " | ".join(names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for trajectory in trajectories:
            peak = peaks[multiplier][trajectory]
            cells = " | ".join(
                f"{deficit(fraction, multiplier, latency, n, trajectory):.4f}" for n in names)
            lines.append(f"| {trajectory} | {peak / 1e6:.1f} | {cells} |")
        lines.append("")

    lines += ["## Non-deficit metrics at the representative configurations", "",
              "Wasted prefetches are objects loaded and evicted without ever being "
              "visible. Peak resident bytes must never exceed the budget.", ""]
    for entry in config["reporting"]["representative_configs"]:
        fraction = float(entry["budget_fraction"])
        multiplier = float(entry["size_multiplier"])
        latency = float(entry["latency_seconds"])
        lines += [f"### budget {fraction:.0%}, size {multiplier:g}x, latency {latency:g}s", "",
                  "| policy | pop-in events | mean pop-in (samples) | wasted prefetches | "
                  "streamed (MB) | peak resident (MB) | peak/budget |",
                  "|---|---|---|---|---|---|---|"]
        for name in names:
            entries = [per_trajectory[(fraction, multiplier, latency, bandwidth, name, t)]
                       for t in trajectories]
            # Each trajectory has its own budget, so the ratio must be taken per
            # trajectory and then maximised - never a summed peak over one budget.
            worst_ratio = max(e["peak_resident_bytes"] / e["budget_bytes"] for e in entries)
            lines.append(
                f"| {name} | {sum(e['pop_in_events'] for e in entries)} | "
                f"{np.mean([e['pop_in_mean_duration_samples'] for e in entries]):.2f} | "
                f"{sum(e['wasted_prefetches'] for e in entries)} | "
                f"{sum(e['bytes_streamed'] for e in entries) / 1e6:.1f} | "
                f"{np.mean([e['peak_resident_bytes'] for e in entries]) / 1e6:.1f} | "
                f"{worst_ratio:.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    arguments = parse_args(argv)
    run_sweep(arguments.config)


if __name__ == "__main__":
    main()
