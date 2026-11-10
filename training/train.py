"""Training utilities for the Phase 3b visibility MLP.

Determinism is a hard requirement, so:
  * the data order comes from ``numpy.random.default_rng(seed)``, not torch's RNG;
  * ``torch.manual_seed(seed)`` is set immediately before model construction, and
    the seed is not reset again, so dropout consumes a reproducible stream;
  * no shuffling, dropout sampling or initialisation depends on wall-clock or
    thread scheduling.

Everything the model consumes is a row of the existing feature matrix. Nothing from
the evaluation sidecar is ever an input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import dataset as D
from . import evaluate as E
from .models import TORCH_AVAILABLE, VisibilityMLP, count_parameters

if TORCH_AVAILABLE:
    import torch
    from torch import nn


# --- target assembly ----------------------------------------------------------

@dataclass
class OutputSpec:
    """One model output: which target, which horizon, and how to mask it."""

    name: str
    target: str          # "frustum" | "occlusion"
    horizon: float

    @property
    def horizon_index(self) -> int:
        return 0 if abs(self.horizon - 0.5) < 1e-9 else 1


@dataclass
class TrainingBundle:
    X: np.ndarray            # (n, d) float32, already normalised
    y: np.ndarray            # (n, k) float32
    loss_mask: np.ndarray    # (n, k) float32, 1 where the output contributes to the loss
    row_mask: np.ndarray     # (n,) bool, rows eligible for training at all
    normaliser: D.Normaliser
    feature_names: list[str]
    output_names: list[str]
    raw_X: np.ndarray        # (n, d) float32, unnormalised, for reference/diagnostics


def per_output_targets(table, specs: Sequence[OutputSpec]
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(y, valid, primary, snap_free)`` arrays for the given outputs.

    ``valid`` is where the label exists (unknown occlusion is invalid), ``primary``
    is the target's own not-currently-visible mask, and ``snap_free`` marks rows that
    are not within +/-10 samples of a yaw snap.
    """
    count = len(table)
    y = np.zeros((count, len(specs)), dtype=np.float32)
    valid = np.ones((count, len(specs)), dtype=bool)
    for column, spec in enumerate(specs):
        y_true, is_valid, _ = E.target_arrays(table, spec.target, spec.horizon_index)
        y[:, column] = y_true.astype(np.float32)
        valid[:, column] = is_valid
    return y, valid


def unified_primary(table) -> np.ndarray:
    """The shared training row set: objects not currently *actually* visible.

    The frustum primary is ``not frustum_visible``; the occlusion primary is
    ``not confirmed visible``. Their union is ``not (frustum_visible and
    occlusion_visible)``. Per-target evaluation still uses each target's own mask
    through ``evaluate.target_arrays``; this union only decides what the shared
    model is trained on.
    """
    currently_visible = (np.asarray(table.frustum_visible, dtype=bool)
                         & (np.asarray(table.occlusion_state) == D.OCCLUSION_TRUE))
    return ~currently_visible


def build_bundle(table, specs: Sequence[OutputSpec], train_rows: np.ndarray,
                 drop_snap_yaw: bool, snap_yaw_rows: np.ndarray,
                 extra_inputs: np.ndarray | None = None) -> TrainingBundle:
    """Assemble targets, masks and a normaliser fitted on ``train_rows`` only."""
    y, valid = per_output_targets(table, specs)
    row_mask = np.asarray(train_rows, dtype=bool).copy()
    if drop_snap_yaw:
        row_mask &= ~np.asarray(snap_yaw_rows, dtype=bool)

    loss_mask = valid.astype(np.float32)
    # A row outside the training set contributes nothing anywhere.
    loss_mask[~row_mask] = 0.0

    raw = table.X if extra_inputs is None else np.concatenate(
        [table.X, np.asarray(extra_inputs, dtype=np.float32).reshape(len(table), -1)], axis=1)
    names = list(table.names)
    if extra_inputs is not None:
        names += [f"extra_{i}" for i in range(raw.shape[1] - len(table.names))]

    normaliser = D.fit_normaliser(
        _table_with_raw(table, raw, names), row_mask)
    normalised = normaliser.apply(raw)

    return TrainingBundle(X=normalised.astype(np.float32), y=y, loss_mask=loss_mask,
                          row_mask=row_mask, normaliser=normaliser, feature_names=names,
                          output_names=[spec.name for spec in specs], raw_X=raw)


def _table_with_raw(table, raw: np.ndarray, names: list[str]):
    """A shallow copy of ``table`` whose X is ``raw``, for the existing normaliser."""
    clone = D.FeatureTable(
        names=names, X=raw, trajectory=table.trajectory, trajectories=table.trajectories,
        sample_index=table.sample_index, object_id=table.object_id, category=table.category,
        split=table.split, distance=table.distance, coverage=table.coverage,
        frustum_visible=table.frustum_visible, occlusion_state=table.occlusion_state,
        resolution_eligible=table.resolution_eligible, y_frustum=table.y_frustum,
        y_occlusion=table.y_occlusion, y_resolution=table.y_resolution,
        horizons=table.horizons)
    return clone


def positive_weights(bundle: TrainingBundle, cap: float = 20.0) -> np.ndarray:
    """``n_negative / n_positive`` per output, from training rows only, clipped."""
    weights = np.ones(bundle.y.shape[1], dtype=np.float32)
    for column in range(bundle.y.shape[1]):
        mask = bundle.loss_mask[:, column] > 0.5
        if not mask.any():
            continue
        labels = bundle.y[mask, column]
        positives = float(labels.sum())
        negatives = float(labels.size - positives)
        if positives <= 0.0:
            weights[column] = 1.0
        else:
            weights[column] = float(min(negatives / positives, cap))
    return weights


# --- evaluation helpers -------------------------------------------------------

def selection_score(table, specs: Sequence[OutputSpec], scores: np.ndarray,
                    rows: np.ndarray, exclude_snap: np.ndarray | None = None) -> float:
    """Mean PR-AUC over the outputs, on each target's primary subset.

    Used for early stopping. It reads validation trajectories only - the caller
    decides which rows those are.
    """
    values = []
    for column, spec in enumerate(specs):
        y_true, valid, primary = E.target_arrays(table, spec.target, spec.horizon_index)
        mask = np.asarray(rows, dtype=bool) & valid & primary
        if exclude_snap is not None:
            mask = mask & ~np.asarray(exclude_snap, dtype=bool)
        if not mask.any():
            continue
        value = E.average_precision(y_true[mask], scores[mask, column])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def choose_threshold(y_true: np.ndarray, score: np.ndarray,
                     candidates: Sequence[float] | None = None) -> float:
    """Threshold maximising F1 on the rows it is given. Caller passes validation rows.

    ``y_true`` is coerced to boolean: callers hold labels as float32, and ``&`` on a
    float array raises rather than doing the obvious thing.
    """
    y_true = np.asarray(y_true, dtype=bool).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if y_true.size == 0:
        return 0.5
    if candidates is None:
        candidates = np.unique(np.quantile(score, np.linspace(0.50, 0.999, 120)))
    best_f1, best = -1.0, 0.5
    for threshold in candidates:
        predicted = score >= threshold
        tp = float(np.sum(y_true & predicted))
        fp = float(np.sum(~y_true & predicted))
        fn = float(np.sum(y_true & ~predicted))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1, best = f1, float(threshold)
    return best


# --- torch training -----------------------------------------------------------

@dataclass
class FitResult:
    model: object
    best_epoch: int
    best_validation_score: float
    history: list[dict]
    parameters: int
    feature_names: list[str]


class SelectionScorer:
    """Early-stopping scorer that only touches the rows it needs.

    The obvious implementation runs a forward pass over every row each epoch and
    then masks. That is ~8x more work than necessary here, because a fold's
    validation rows are two trajectories out of fourteen, and it dominated the
    runtime. The scorer precomputes the index of rows that can possibly contribute
    and the labels for those rows, so each epoch forwards only those.
    """

    def __init__(self, table, specs: Sequence[OutputSpec], rows: np.ndarray,
                 exclude_snap: np.ndarray | None = None) -> None:
        self.specs = list(specs)
        self.index = np.flatnonzero(np.asarray(rows, dtype=bool))
        self.blocks = []
        for spec in self.specs:
            y_true, valid, primary = E.target_arrays(table, spec.target, spec.horizon_index)
            mask = np.asarray(rows, dtype=bool) & valid & primary
            if exclude_snap is not None:
                mask = mask & ~np.asarray(exclude_snap, dtype=bool)
            local = mask[self.index]
            self.blocks.append((spec, y_true[self.index][local], local))

    @property
    def rows(self) -> int:
        return int(self.index.size)

    def __call__(self, scores: np.ndarray) -> float:
        """``scores`` is aligned with ``self.index``, shape (rows, outputs)."""
        values = []
        for column, (_spec, y_true, local) in enumerate(self.blocks):
            if y_true.size == 0:
                continue
            value = E.average_precision(y_true, scores[local, column])
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else float("nan")


def train_mlp(bundle: TrainingBundle, specs: Sequence[OutputSpec], validation_rows: np.ndarray,
              table, config: dict, seed: int, exclude_snap: np.ndarray | None = None,
              extra_validation_rows: np.ndarray | None = None) -> FitResult:
    """Deterministic MLP training with early stopping on validation rows."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available")

    settings = config["training"]
    model_config = config["model"]
    weight_cap = float(settings.get("pos_weight_max", 20.0))
    pos_weight = positive_weights(bundle, cap=weight_cap)

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = VisibilityMLP(input_dim=bundle.X.shape[1], hidden=model_config["hidden"],
                          output_dim=len(specs), dropout=float(model_config["dropout"]),
                          activation=str(model_config.get("activation", "relu")))
    optimiser = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]),
                                 weight_decay=float(settings["weight_decay"]))

    train_index = np.flatnonzero(bundle.row_mask)
    X_all = torch.from_numpy(bundle.X)
    y_all = torch.from_numpy(bundle.y)
    mask_all = torch.from_numpy(bundle.loss_mask)
    pos_weight_tensor = torch.from_numpy(pos_weight)
    if pos_weight_tensor.dim() == 1:
        pos_weight_tensor = pos_weight_tensor.reshape(1, -1)

    selection_rows = np.asarray(validation_rows, dtype=bool)
    if extra_validation_rows is not None:
        selection_rows = selection_rows | np.asarray(extra_validation_rows, dtype=bool)
    scorer = SelectionScorer(table, specs, selection_rows, exclude_snap)
    selection_tensor = torch.from_numpy(scorer.index)

    batch_size = int(settings["batch_size"])
    max_epochs = int(settings["max_epochs"])
    patience = int(settings["early_stopping_patience"])
    min_epochs = int(settings.get("min_epochs", 1))

    rng = np.random.default_rng(int(seed))
    best_state, best_epoch, best_score = None, -1, -math.inf
    history: list[dict] = []
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(train_index)
        epoch_loss, batches = 0.0, 0
        for start in range(0, order.size, batch_size):
            index = torch.from_numpy(order[start:start + batch_size])
            optimiser.zero_grad()
            logits = model(X_all[index])
            elementwise = nn.functional.binary_cross_entropy_with_logits(
                logits, y_all[index], reduction="none", pos_weight=pos_weight_tensor)
            weights = mask_all[index]
            loss = (elementwise * weights).sum() / weights.sum().clamp_min(1.0)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            batches += 1

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(X_all[selection_tensor])).numpy()
        score = scorer(scores)
        history.append({"epoch": epoch, "loss": epoch_loss / max(1, batches),
                        "validation_score": score})
        if np.isfinite(score) and score > best_score + 1e-6:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience and epoch >= min_epochs:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return FitResult(model=model, best_epoch=best_epoch, best_validation_score=best_score,
                     history=history, parameters=count_parameters(model),
                     feature_names=list(bundle.feature_names))


def predict_mlp(model, bundle: TrainingBundle) -> np.ndarray:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(bundle.X))
        return torch.sigmoid(logits).numpy()


def time_inference(model, bundle: TrainingBundle, batch: int, repeats: int) -> dict:
    """CPU inference time for one batch of ``batch`` candidates, mean over repeats."""
    if not TORCH_AVAILABLE:
        return {}
    model.eval()
    take = min(batch, bundle.X.shape[0])
    sample = torch.from_numpy(bundle.X[:take])
    with torch.no_grad():
        for _ in range(3):
            model(sample)                       # warm up
        timings = []
        for _ in range(int(repeats)):
            import time
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000.0)
    return {"rows": int(take), "repeats": int(repeats),
            "mean_ms": float(np.mean(timings)), "std_ms": float(np.std(timings)),
            "min_ms": float(np.min(timings)), "max_ms": float(np.max(timings)),
            "threads": int(torch.get_num_threads())}


# --- sklearn training ---------------------------------------------------------

def train_sklearn(X: np.ndarray, y: np.ndarray, rows: np.ndarray, seed: int,
                  settings: dict):
    from .models import build_sklearn_classifier
    model = build_sklearn_classifier(seed, settings)
    mask = np.asarray(rows, dtype=bool)
    model.fit(X[mask], y[mask].astype(np.int8))
    return model


def predict_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float64)
