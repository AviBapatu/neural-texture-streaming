#!/usr/bin/env python3
"""Export Phase 5 engine inputs from the frozen Phase 4a score bundle.

Everything the Godot side needs, plus a filtered copy of the simulator's own
inputs so the engine and the simulator can be compared on identical data:

  dataset/streaming/runs/phase5_engine/
    manifest.json          streamable objects (the 7 ambientCG identities) + trajectories
    scores/<traj>.json     per-sample pose + per-(sample, object) signals and scores
    cells.json             the parity cell (budget/latency/bandwidth + budget_bytes)
    object_index_53.json   object index restricted to the 53 streamable objects
    oof_scores_53.npz      score bundle restricted to the 53 streamable objects

The 53 objects are the scene's objects that use one of the 7 texture identities
that have generated variants (plz 4 + wal 42 + shw 7). The simulator is run
against the filtered index/bundle so its numbers are scoped to exactly the same
objects the engine streams.

Nothing existing is modified: all outputs are written under
dataset/streaming/runs/phase5_engine/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training import features as F  # noqa: E402
from sim import scene  # noqa: E402

VARIANT_IDS = [
    "ambientcg/Asphalt021_Color", "ambientcg/Bricks066_Color", "ambientcg/Concrete012_Color",
    "ambientcg/Grass004_Color", "ambientcg/Ground037_Color", "ambientcg/Metal063_Color",
    "ambientcg/Road001_Color",
]

SAMPLE_INTERVAL = 0.1
WORKING_SET_WINDOW = 20  # 2 s at 10 Hz


def streamable(entry: dict) -> bool:
    return any(t.get("texture_id") in VARIANT_IDS for t in (entry.get("textures") or []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="training/experiments/phase4a/oof_scores.npz")
    parser.add_argument("--object-index", default="dataset/manifests/object_index.json")
    parser.add_argument("--samples-dir", default="dataset/processed/run_full_p25")
    parser.add_argument("--out", default="dataset/streaming/runs/phase5_engine")
    parser.add_argument("--snap-window", type=int, default=10)
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--latency", type=float, default=0.5)
    parser.add_argument("--bandwidth-mb-s", type=float, default=200.0)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)

    index = json.loads(Path(args.object_index).read_text(encoding="utf-8"))
    keep = {e["object_id"] for e in index["objects"] if streamable(e)}
    records = {e["object_id"]: e for e in index["objects"] if e["object_id"] in keep}
    if len(keep) == 0:
        print("no streamable objects found", file=sys.stderr)
        return 2

    # --- filtered simulator inputs -------------------------------------------
    filtered_index = dict(index)
    filtered_index["objects"] = [e for e in index["objects"] if e["object_id"] in keep]
    filtered_index["object_count"] = len(filtered_index["objects"])
    (out / "object_index_53.json").write_text(
        json.dumps(filtered_index, indent=1) + "\n", encoding="utf-8")

    with np.load(args.oof, allow_pickle=True) as payload:
        arrays = {k: payload[k] for k in payload.files}
    object_id = arrays["object_id"]
    mask = np.array([oid in keep for oid in object_id], dtype=bool)
    filtered = {k: (v[mask] if v.shape and v.shape[0] == mask.size else v)
                for k, v in arrays.items()}
    np.savez(out / "oof_scores_53.npz", **filtered)

    # --- object facts ---------------------------------------------------------
    sizes = {oid: scene.full_bytes(scene.load_objects(args.object_index), oid, 1.0)
             for oid in keep}
    objects = []
    for oid, entry in sorted(records.items()):
        centre = F.aabb_centre(entry["world_bounds"])
        identity = next(t["texture_id"] for t in (entry.get("textures") or [])
                        if t["texture_id"] in VARIANT_IDS)
        source = next(t["source_path"] for t in (entry.get("textures") or [])
                      if t["texture_id"] == identity)
        available = next(t["available_resolutions"] for t in (entry.get("textures") or [])
                         if t["texture_id"] == identity)
        objects.append({
            "object_id": oid,
            "category": entry["category"],
            "texture_id": identity,
            "source_path": source,
            "available_resolutions": available,
            "full_bytes": sizes[oid],
            "centre": [float(v) for v in centre],
            "world_bounds": entry["world_bounds"],
        })

    # --- poses (per trajectory, from the processed samples) -------------------
    poses: dict[str, dict] = {}
    for path in sorted(Path(args.samples_dir).glob("*.samples.jsonl")):
        tid = path.name.replace(".samples.jsonl", "")
        samples = []
        for line in path.open(encoding="utf-8"):
            samples.append(json.loads(line))
        samples.sort(key=lambda r: int(r["sample_index"]))
        quats = np.array([r["cam_quaternion"] for r in samples], dtype=np.float64)
        times = np.array([float(r["session_time"]) for r in samples], dtype=np.float64)
        angular, _ = F.angular_velocity_from_quaternions(quats, times)
        by_sample = {}
        for i, row in enumerate(samples):
            by_sample[int(row["sample_index"])] = {
                "position": [float(v) for v in row["cam_position"]],
                "quaternion": [float(v) for v in row["cam_quaternion"]],
                "velocity": [float(v) for v in row["cam_velocity"]],
                "angular_velocity": [float(v) for v in angular[i]],
                "fov": float(row["cam_fov"]),
                "viewport": [float(row["viewport"][0]), float(row["viewport"][1])],
            }
        poses[tid] = by_sample

    # --- per-trajectory rows --------------------------------------------------
    trajectories = sorted(set(str(t) for t in arrays["trajectory"]))
    per_traj_rows: dict[str, list] = {t: [] for t in trajectories}
    for i in range(len(object_id)):
        if not mask[i]:
            continue
        per_traj_rows[str(arrays["trajectory"][i])].append(i)

    peak_ws: dict[str, float] = {}
    for tid in trajectories:
        rows = per_traj_rows[tid]
        sample_index = np.array([int(arrays["sample_index"][i]) for i in rows], dtype=np.int64)
        order = np.argsort(sample_index, kind="stable")
        rows = [rows[j] for j in order]
        sample_index = sample_index[order]
        out_rows = {}
        samples_out = sorted(set(sample_index.tolist()))
        for sample in samples_out:
            sel = [r for r, s in zip(rows, sample_index) if int(s) == sample]
            out_rows[sample] = [[
                str(arrays["object_id"][r]),
                float(arrays["coverage"][r]),
                int(arrays["frustum_visible"][r]),
                int(arrays["occlusion_state"][r]),
                float(arrays["heuristic_occlusion"][r]),
                float(arrays["mlp_b__occlusion_0_5"][r]),
                float(arrays["mlp_b__occlusion_1_0"][r]),
            ] for r in sel]
        snap = sorted({int(arrays["sample_index"][r]) for r in rows
                       if int(arrays["snap_affected_yaw"][r]) == 1})
        (out / "scores" / f"{tid}.json").write_text(json.dumps({
            "trajectory_id": tid,
            "samples": samples_out,
            "pose": poses.get(tid, {}),
            "rows": {str(k): v for k, v in out_rows.items()},
            "snap_samples": snap,
        }, separators=(",", ":")), encoding="utf-8")

        # peak working set over the 53 (the budget denominator), mirroring sim/scene.py
        vis = np.array([int(arrays["frustum_visible"][r]) == 1
                        and int(arrays["occlusion_state"][r]) != 0 for r in rows])
        oids = np.array([str(arrays["object_id"][r]) for r in rows], dtype=object)
        rows_by_sample = {s: np.flatnonzero(sample_index == s) for s in samples_out}
        peak_ws[tid] = scene.peak_working_set(rows_by_sample, vis, oids,
                                              {k: v for k, v in sizes.items()},
                                              WORKING_SET_WINDOW)

    # --- manifest + cells -----------------------------------------------------
    (out / "manifest.json").write_text(json.dumps({
        "streamable_count": len(objects),
        "variant_ids": VARIANT_IDS,
        "objects": objects,
        "trajectories": trajectories,
        "peak_working_set_bytes": peak_ws,
        "sample_interval_seconds": SAMPLE_INTERVAL,
        "working_set_window_samples": WORKING_SET_WINDOW,
    }, indent=1) + "\n", encoding="utf-8")

    budgets = {tid: max(peak_ws[tid] * args.budget_fraction, 1.0) for tid in trajectories}
    (out / "cells.json").write_text(json.dumps({
        "budget_fraction": args.budget_fraction,
        "latency_seconds": args.latency,
        "bandwidth_mb_s": args.bandwidth_mb_s,
        "bandwidth_bytes_per_s": args.bandwidth_mb_s * 1e6,
        "size_multiplier": 1.0,
        "snap_window_samples": args.snap_window,
        "budget_bytes": budgets,
        "policies": ["reactive_visible", "heuristic_occ", "mlp_b"],
    }, indent=1) + "\n", encoding="utf-8")

    print(f"streamable objects: {len(objects)}")
    print(f"trajectories: {len(trajectories)}")
    print("peak working set (MB) over 53:", {t: round(peak_ws[t] / 1e6, 2)
                                            for t in trajectories[:5]})
    print("budget (MB) over 53:", {t: round(budgets[t] / 1e6, 2) for t in trajectories[:5]})
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
