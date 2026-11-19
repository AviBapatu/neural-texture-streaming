#!/usr/bin/env python3
"""Compare the Godot engine's streaming metrics against sim/simulate.py.

The simulator is run scoped to the same 53 objects as the engine (via
sim/config_5engine.yaml and the filtered bundle from export_scores_for_godot.py),
so a difference can only come from the port, not the object set.

Usage:
  python sim/run_sweep.py --config sim/config_5engine.yaml
  godot --headless --path godot --script res://tests/streaming_parity_test.gd -- \
      --out=dataset/streaming/runs/phase5_engine
  python tools/check_godot_parity.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

METRICS = ("deficit", "bytes_streamed", "wasted_prefetches", "loads_completed",
           "pop_in_events", "peak_resident_bytes")
TOLERANCES = {"deficit": 0.01, "bytes_streamed": 1.0, "wasted_prefetches": 0,
              "loads_completed": 0, "pop_in_events": 0, "peak_resident_bytes": 1.0}


def load_sim(path: Path) -> dict:
    table: dict[tuple[str, str], dict] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            table.setdefault((row["trajectory"], row["policy"]), {})[row["metric"]] = \
                float(row["value"])
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="dataset/streaming/runs/phase5_engine/engine/engine_metrics.json")
    parser.add_argument("--sim", default="dataset/streaming/runs/phase5_engine/sim_results/results.csv")
    parser.add_argument("--focus", default="t01_walk_straight")
    parser.add_argument("--policies", default="reactive_visible,heuristic_occ,mlp_b")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    engine = json.loads((REPO / args.engine).read_text())
    sim = load_sim(REPO / args.sim)
    policies = args.policies.split(",")
    metrics = engine["metrics"]

    print(f"scope: {engine['scope']}")
    print(f"cell: budget={engine['budget_fraction']:.0%} latency={engine['latency_seconds']}s "
          f"bandwidth={engine['bandwidth_mb_s']}MB/s")
    print(f"heuristic port: rows={engine['heuristic_port']['rows_checked']} "
          f"max_abs_diff={engine['heuristic_port']['max_abs_diff_deg']:.3e} deg")
    print()

    header = f"{'trajectory':22s} {'policy':16s} " + " ".join(f"{m:>18s}" for m in METRICS)
    print(header)
    print("-" * len(header))

    worst: dict[str, float] = {m: 0.0 for m in METRICS}
    offenders: list[tuple] = []
    report: dict = {"focus": args.focus, "comparisons": []}
    for traj in sorted(metrics):
        for policy in policies:
            if policy not in metrics[traj]:
                continue
            eng = metrics[traj][policy]
            ref = sim.get((traj, policy))
            if ref is None:
                continue
            cells = []
            row_ok = True
            for m in METRICS:
                e = float(eng.get(m, 0.0))
                s = float(ref.get(m, 0.0))
                # deficits are fractions; the others are counts/bytes
                diff = abs(e - s)
                if m == "deficit":
                    scale_diff = diff
                elif s > 0:
                    scale_diff = diff / abs(s)
                else:
                    scale_diff = diff
                worst[m] = max(worst[m], scale_diff)
                tol = TOLERANCES[m]
                ok = scale_diff <= tol
                row_ok = row_ok and ok
                cells.append(f"{s:>10.4f}/{e:>6.3f}{'' if ok else '!!':1s}")
            print(f"{traj:22s} {policy:16s} " + " ".join(f"{c:>18s}" for c in cells))
            if not row_ok:
                offenders.append((traj, policy))
            report["comparisons"].append({
                "trajectory": traj, "policy": policy,
                "sim": {m: float(ref.get(m, 0.0)) for m in METRICS},
                "engine": {m: float(eng.get(m, 0.0)) for m in METRICS},
                "ok": row_ok,
            })

    print()
    print("max normalized differences (sim / engine):")
    for m in METRICS:
        print(f"  {m:22s} {worst[m]:.3e}")
    print()
    if offenders:
        print(f"OUT OF TOLERANCE ({len(offenders)}): {offenders}")
    else:
        print("all cells within tolerance")
    if args.json_out:
        (REPO / args.json_out).write_text(json.dumps(report, indent=1) + "\n")
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
