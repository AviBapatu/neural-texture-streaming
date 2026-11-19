#!/usr/bin/env python3
"""Export the Phase 3b MLP for Godot-consumable inference.

Reads training/experiments/phase3b/visibility_mlp_b_seed1.{npz,json} and writes
godot/assets/models/visibility_mlp_b_seed1.json with the feature order, output
names, architecture and the dense layers (weights + biases) in forward order.

The bundle has 66 inputs: training/configs/features_v1.yaml allows 65 and the
exported headers append one more column, `extra_0` - the frozen frustum-tuned
heuristic score (training/train.py build_bundle). So the network is 66 -> 128 ->
64 -> 4, and the 66th feature is the frozen heuristic, not a new geometric feature.

It also writes a small probe file (16 deterministic feature rows plus the logits
from training/models.py forward_numpy) so the GDScript port can be checked against
the reference implementation.

Usage:
  python tools/export_model_for_godot.py
  godot --headless --path godot --script res://tests/mlp_parity_test.gd
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

from training import models as M  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="training/experiments/phase3b/visibility_mlp_b_seed1")
    parser.add_argument("--out-dir", default="godot/assets/models")
    parser.add_argument("--probe-rows", type=int, default=16)
    parser.add_argument("--probe-seed", type=int, default=1234)
    args = parser.parse_args()

    header, arrays = M.load_bundle(REPO / args.bundle)
    feature_names = list(header["feature_names"])
    output_names = list(header["output_names"])
    layers = []
    layer = 0
    while f"weight_{layer}" in arrays:
        layers.append({
            "weight": arrays[f"weight_{layer}"].astype(float).tolist(),
            "bias": arrays[f"bias_{layer}"].astype(float).tolist(),
        })
        layer += 1

    model = {
        "format": "commandcode.phase5.visibility_mlp.for_godot.v1",
        "source": "training/experiments/phase3b/visibility_mlp_b_seed1",
        "architecture": "dense_stack",
        "activation": header["activation"],
        "input_dim": int(header["input_dim"]),
        "output_dim": int(header["output_dim"]),
        "hidden": list(header["hidden"]),
        "feature_names": feature_names,
        "output_names": output_names,
        "normaliser": {
            "mean": [float(v) for v in header["normaliser"]["mean"]],
            "std": [float(v) for v in header["normaliser"]["std"]],
            "apply": "(x - mean) / std before the first Linear layer",
        },
        "layers": layers,
        "extra_feature_note": (
            "66 inputs: features_v1.yaml allows 65; the 66th ('extra_0') is the frozen "
            "frustum-tuned heuristic score appended by training/train.py build_bundle."),
    }
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "visibility_mlp_b_seed1.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")

    # --- probe: deterministic rows + reference logits -------------------------
    rng = np.random.RandomState(int(args.probe_seed))
    mean = np.asarray(header["normaliser"]["mean"], dtype=np.float64)
    std = np.asarray(header["normaliser"]["std"], dtype=np.float64)
    features = mean + std * rng.standard_normal((int(args.probe_rows), len(feature_names)))
    logits = M.forward_numpy(header, arrays, features)
    probe = {
        "feature_names": feature_names,
        "output_names": output_names,
        "features": features.astype(float).tolist(),
        "expected_logits": logits.astype(float).tolist(),
        "tolerance": 1e-4,
    }
    probe_path = out_dir / "visibility_mlp_b_seed1_probe.json"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")

    # --- self-check: re-parse the written model and reproduce forward_numpy ----
    reloaded = json.loads(model_path.read_text())
    w = [np.asarray(l["weight"], dtype=np.float64) for l in reloaded["layers"]]
    b = [np.asarray(l["bias"], dtype=np.float64) for l in reloaded["layers"]]
    x = (features - np.asarray(reloaded["normaliser"]["mean"])) / np.asarray(reloaded["normaliser"]["std"])
    for i in range(len(w)):
        x = x @ w[i].T + b[i]
        if i < len(w) - 1:
            x = np.maximum(x, 0.0) if reloaded["activation"] == "relu" else np.tanh(x)
    diff = float(np.max(np.abs(x - logits)))

    print(f"model     -> {model_path} ({len(layers)} layers, {header['input_dim']} inputs)")
    print(f"probe     -> {probe_path} ({len(features)} rows)")
    print(f"extra feature (66th): {feature_names[-1]!r} = frozen frustum-tuned heuristic score")
    print(f"self-check vs forward_numpy: max_abs_diff={diff:.3e} (tolerance 1e-9)")
    return 0 if diff < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
