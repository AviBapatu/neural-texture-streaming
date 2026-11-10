"""Metric primitives and the evaluation harness.

Everything is computed from numpy arrays with no third-party metric library, so the
definitions are visible and testable. ``training/tests/test_features.py`` recomputes
a sample of these with an independent naive implementation and compares.

Two conventions matter and are applied everywhere:

* **Unknown occlusion is not negative.** Rows whose occlusion target is unknown are
  removed from the occlusion target's metrics and counted separately.
* **The primary subset** is the pool where prefetching actually matters: for the
  frustum target, rows that are not currently in the frustum; for the occlusion
  target, rows that are not currently confirmed visible. Full-set metrics are
  dominated by objects that are already on screen and are easy to get right.
"""

from __future__ import annotations

import numpy as np

from .dataset import OCCLUSION_TRUE, OCCLUSION_UNKNOWN


# --- primitives ---------------------------------------------------------------

def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision (the area under the precision-recall curve).

    AP = sum over the ranking of (recall_n - recall_{n-1}) * precision_n. Returns
    NaN when there are no positives, which is reported as "undefined" rather than
    as zero.
    """
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ordered = y_true[order]
    true_positive = np.cumsum(ordered)
    false_positive = np.cumsum(~ordered)
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / positives
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous_recall) * precision))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   score: np.ndarray | None = None) -> dict:
    """Precision/recall/F1/PR-AUC/confusion for one boolean target."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    counts = confusion(y_true, y_pred)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    result = {
        "n": int(y_true.size),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if y_true.size else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision(y_true, score)) if score is not None else float("nan"),
    }
    result.update(counts)
    return result


def recall_at_k(table, y_true: np.ndarray, score: np.ndarray, k: int,
                eligible: np.ndarray | None = None) -> float:
    """Mean, over samples containing at least one newly-visible object, of the
    fraction of those objects that appear in the top ``k`` by score.

    Samples with no newly-visible object are skipped: they contribute no recall to
    measure and would drag the mean toward zero.
    """
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    if eligible is not None:
        keep = np.asarray(eligible, dtype=bool)
    else:
        keep = np.ones(y_true.shape[0], dtype=bool)

    # Group rows by (trajectory, sample) without a per-row dict.
    keys = table.trajectory.astype(np.int64) * 1_000_000 + table.sample_index.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    boundaries = np.flatnonzero(np.diff(keys[order])) + 1
    groups = np.split(order, boundaries)

    values: list[float] = []
    for group in groups:
        active = group[keep[group]]
        if active.size == 0:
            continue
        positives = active[y_true[active]]
        if positives.size == 0:
            continue
        ranked = active[np.argsort(-score[active], kind="stable")][:k]
        hits = int(np.isin(ranked, positives).sum())
        values.append(hits / positives.size)
    return float(np.mean(values)) if values else float("nan")


def resolution_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       classes: np.ndarray | None = None) -> dict:
    """Per-class precision/recall/F1, macro-F1, confusion matrix and under-prediction rate.

    Under-prediction (predicted below required) is the failure that shows up as
    visible quality loss, so it is reported on its own rather than folded into
    accuracy.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred])) if y_true.size else np.zeros(0)
    classes = np.asarray(classes, dtype=np.int64)

    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    index = {int(c): i for i, c in enumerate(classes)}
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[index[int(pred_value)], index[int(true_value)]] += 1

    per_class = {}
    f1_scores = []
    for position, value in enumerate(classes):
        tp = int(matrix[position, position])
        fp = int(matrix[position, :].sum() - tp)
        fn = int(matrix[:, position].sum() - tp)
        support = int(matrix[:, position].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[int(value)] = {
            "support": support,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
        }
        # Macro-F1 averages over classes that actually occur. A class with no
        # support (e.g. resolution 0, which only ever appears as a prediction for a
        # non-visible object) would otherwise contribute a spurious zero.
        if support > 0:
            f1_scores.append(f1)

    under = int(np.sum(y_pred < y_true))
    over = int(np.sum(y_pred > y_true))
    return {
        "n": int(y_true.size),
        "classes": [int(c) for c in classes],
        "classes_with_support": [int(value) for value in classes
                                 if per_class[int(value)]["support"] > 0],
        "confusion_matrix": matrix.tolist(),
        "confusion_row_predicted_col_true": True,
        "per_class": per_class,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else float("nan"),
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else float("nan"),
        "under_prediction_rate": float(under / y_true.size) if y_true.size else float("nan"),
        "over_prediction_rate": float(over / y_true.size) if y_true.size else float("nan"),
        "exact_resolution_rate": float(np.mean(y_true == y_pred)) if y_true.size else float("nan"),
    }


# --- target-level evaluation --------------------------------------------------

def evaluate_target(table, y_true: np.ndarray, y_pred: np.ndarray,
                    score: np.ndarray | None, primary: np.ndarray,
                    percentiles: tuple[int, ...] = (5, 10, 20)) -> dict:
    """Full-set and primary-subset metrics plus recall@k and a per-trajectory breakdown."""
    primary = np.asarray(primary, dtype=bool)
    full = binary_metrics(y_true, y_pred, score)
    subset = binary_metrics(y_true[primary], y_pred[primary],
                            score[primary] if score is not None else None)
    result = {
        "full": full,
        "primary_subset": subset,
        "newly_visible_positives": int(np.sum(np.asarray(y_true, dtype=bool) & primary)),
    }
    if score is not None:
        result["recall_at_k"] = {
            str(k): recall_at_k(table, y_true, score, k) for k in percentiles
        }
        result["recall_at_k_primary"] = {
            str(k): recall_at_k(table, y_true, score, k, eligible=primary) for k in percentiles
        }
    result["per_trajectory"] = {}
    for index, trajectory_id in enumerate(table.trajectories):
        mask = table.trajectory == index
        if not mask.any():
            continue
        result["per_trajectory"][trajectory_id] = binary_metrics(
            y_true[mask], y_pred[mask], score[mask] if score is not None else None)
    return result


def target_arrays(table, target: str, horizon_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(y_true, valid_mask, primary_mask)`` for one target and horizon.

    ``valid_mask`` is what the metrics are restricted to: everything for the frustum
    target, and non-unknown rows for the occlusion target. Unknown occlusion is
    excluded here and counted by the caller, never treated as negative.
    """
    if target == "frustum":
        y_true = table.y_frustum[:, horizon_index].astype(bool)
        valid = np.ones(y_true.shape[0], dtype=bool)
        primary = ~np.asarray(table.frustum_visible, dtype=bool)
    elif target == "occlusion":
        y_true = table.y_occlusion[:, horizon_index] == OCCLUSION_TRUE
        valid = table.y_occlusion[:, horizon_index] != OCCLUSION_UNKNOWN
        primary = np.asarray(table.occlusion_state) != OCCLUSION_TRUE
    else:
        raise ValueError(f"unknown target {target!r}")
    return y_true, valid, primary
