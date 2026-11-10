"""Phase 3b model definitions and weight export.

One shared MLP with four output logits. The architecture is a plain dense stack -
``Linear -> ReLU -> Dropout`` repeated - because the Phase 4 runtime port has to
reimplement it in GDScript, where anything more exotic would be a liability. There
are no convolutions, no residuals, no attention and no normalisation layers inside
the network; normalisation is a separate, exportable affine transform.

The exported bundle is deliberately dumb: a JSON header plus an NPZ of weight
arrays, with the feature name order and target order written down, so a port cannot
silently disagree about column ordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # torch and sklearn are optional: the rest of the package must import without them
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    torch = None
    nn = object  # type: ignore[assignment]
    TORCH_AVAILABLE = False


ACTIVATIONS = {"relu": "relu", "tanh": "tanh"}


# --- torch model --------------------------------------------------------------

if TORCH_AVAILABLE:

    class VisibilityMLP(nn.Module):
        """Plain dense stack: Linear -> activation -> dropout, then a linear head.

        ``forward`` is deliberately a bare loop over ``self.layers`` so the GDScript
        port is a transcription rather than a translation.
        """

        def __init__(self, input_dim: int, hidden: Sequence[int], output_dim: int,
                     dropout: float = 0.1, activation: str = "relu") -> None:
            super().__init__()
            if activation not in ACTIVATIONS:
                raise ValueError(f"unknown activation {activation!r}")
            self.input_dim = int(input_dim)
            self.hidden = [int(h) for h in hidden]
            self.output_dim = int(output_dim)
            self.dropout_p = float(dropout)
            self.activation_name = activation

            blocks: list[nn.Module] = []
            previous = self.input_dim
            for width in self.hidden:
                blocks.append(nn.Linear(previous, width))
                blocks.append(nn.ReLU() if activation == "relu" else nn.Tanh())
                if self.dropout_p > 0.0:
                    blocks.append(nn.Dropout(self.dropout_p))
                previous = width
            blocks.append(nn.Linear(previous, self.output_dim))
            self.layers = nn.Sequential(*blocks)

        def forward(self, features):  # noqa: D102 - mirrors a GDScript func exactly
            return self.layers(features)

        def dense_layers(self) -> list[tuple[np.ndarray, np.ndarray]]:
            """The weight matrices and biases, in forward order."""
            result = []
            for module in self.layers:
                if isinstance(module, nn.Linear):
                    result.append((module.weight.detach().numpy().copy(),
                                   module.bias.detach().numpy().copy()))
            return result


def count_parameters(model) -> int:
    """Trainable scalar parameter count."""
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # sklearn estimators
    total = 0
    for attribute in ("_predictors", "estimators_"):
        for stage in getattr(model, attribute, None) or []:
            for estimator in (stage if isinstance(stage, (list, tuple)) else [stage]):
                tree = getattr(estimator, "tree_", None)
                if tree is not None:
                    total += int(tree.node_count)
    return total


# --- export -------------------------------------------------------------------

@dataclass
class BundleHeader:
    feature_names: list[str]
    output_names: list[str]
    hidden: list[int]
    activation: str
    dropout: float
    normaliser_mean: list[float]
    normaliser_std: list[float]
    input_dim: int
    output_dim: int

    def as_dict(self) -> dict:
        return {
            "format": "commandcode.phase3b.visibility_mlp.v1",
            "architecture": "dense_stack",
            "layer_order": "for each block: Linear(weight, bias) then activation then dropout",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden": self.hidden,
            "activation": self.activation,
            "dropout": self.dropout,
            "feature_names": self.feature_names,
            "output_names": self.output_names,
            "normaliser": {
                "mean": self.normaliser_mean,
                "std": self.normaliser_std,
                "note": "apply as (x - mean) / std before the first Linear layer",
            },
        }


def export_bundle(model, feature_names: Sequence[str], output_names: Sequence[str],
                  normaliser, path_stem: str | Path) -> dict:
    """Write ``<stem>.json`` (header) and ``<stem>.npz`` (weights).

    The NPZ holds ``weight_0, bias_0, weight_1, bias_1, ...`` in forward order, so a
    port reads them in sequence and cannot get the order wrong.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available")
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    header = BundleHeader(
        feature_names=list(feature_names), output_names=list(output_names),
        hidden=list(model.hidden), activation=model.activation_name,
        dropout=float(model.dropout_p),
        normaliser_mean=[float(v) for v in normaliser.mean],
        normaliser_std=[float(v) for v in normaliser.std],
        input_dim=int(model.input_dim), output_dim=int(model.output_dim))

    arrays: dict[str, np.ndarray] = {}
    for index, (weight, bias) in enumerate(model.dense_layers()):
        arrays[f"weight_{index}"] = weight.astype(np.float32)
        arrays[f"bias_{index}"] = bias.astype(np.float32)
    np.savez(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text(json.dumps(header.as_dict(), indent=2) + "\n",
                                         encoding="utf-8")
    return {"header": str(stem.with_suffix(".json")),
            "weights": str(stem.with_suffix(".npz")),
            "dense_layers": len(model.dense_layers()),
            "parameters": count_parameters(model)}


def load_bundle(stem: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read an exported bundle back. Used by the export round-trip test."""
    stem = Path(stem)
    header = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(stem.with_suffix(".npz")) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return header, arrays


def forward_numpy(header: dict, arrays: dict[str, np.ndarray],
                  features: np.ndarray) -> np.ndarray:
    """A tiny reference implementation of the exported network.

    This exists so the port target is executable and testable: it reads exactly the
    JSON/NPZ an external implementation would read, and nothing else.
    """
    mean = np.asarray(header["normaliser"]["mean"], dtype=np.float64)
    std = np.asarray(header["normaliser"]["std"], dtype=np.float64)
    activations = np.asarray((features - mean) / std, dtype=np.float64)

    layer = 0
    while f"weight_{layer}" in arrays:
        weight = arrays[f"weight_{layer}"].astype(np.float64)
        bias = arrays[f"bias_{layer}"].astype(np.float64)
        activations = activations @ weight.T + bias
        if f"weight_{layer + 1}" in arrays:            # hidden layer, apply activation
            if header["activation"] == "relu":
                activations = np.maximum(activations, 0.0)
            else:
                activations = np.tanh(activations)
        layer += 1
    return activations


# --- sklearn variant ----------------------------------------------------------

def build_sklearn_classifier(seed: int, settings: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=int(settings.get("max_iter", 300)),
        learning_rate=float(settings.get("learning_rate", 0.1)),
        max_leaf_nodes=int(settings.get("max_leaf_nodes", 31)),
        early_stopping=bool(settings.get("early_stopping", True)),
        validation_fraction=float(settings.get("validation_fraction", 0.15)),
        n_iter_no_change=int(settings.get("n_iter_no_change", 15)),
        random_state=int(seed))
