"""Metrics and dev-only threshold selection for official benchmark runs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .schema import SENTIMENT_ORDER


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, average: str = "macro") -> Optional[float]:
    values = np.asarray(y_true)
    columns = values.reshape(-1, 1) if values.ndim == 1 else values
    if any(np.unique(columns[:, idx]).size < 2 for idx in range(columns.shape[1])):
        return None
    try:
        score = float(roc_auc_score(y_true, y_prob, average=average))
        return score if math.isfinite(score) else None
    except ValueError:
        return None


def _safe_average_precision(
    y_true: np.ndarray, y_prob: np.ndarray, average: str = "macro"
) -> Optional[float]:
    values = np.asarray(y_true)
    columns = values.reshape(-1, 1) if values.ndim == 1 else values
    if any(np.unique(columns[:, idx]).size < 2 for idx in range(columns.shape[1])):
        return None
    try:
        score = float(average_precision_score(y_true, y_prob, average=average))
        return score if math.isfinite(score) else None
    except ValueError:
        return None


def tune_acsa_thresholds(
    true_m: np.ndarray,
    true_s: np.ndarray,
    prob_m: np.ndarray,
    prob_s: np.ndarray,
    minimum: float = 0.1,
    maximum: float = 0.9,
    steps: int = 17,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tune per-aspect/class thresholds exclusively on a development split."""

    grid = np.linspace(minimum, maximum, steps, dtype=np.float32)
    num_aspects = true_m.shape[1]
    thresholds_m = np.full(num_aspects, 0.5, dtype=np.float32)
    thresholds_s = np.full((num_aspects, 3), 0.5, dtype=np.float32)

    for aspect_idx in range(num_aspects):
        target = true_m[:, aspect_idx]
        if np.unique(target).size >= 2:
            scores = [
                f1_score(target, prob_m[:, aspect_idx] >= threshold, zero_division=0)
                for threshold in grid
            ]
            thresholds_m[aspect_idx] = grid[int(np.argmax(scores))]

        mentioned = target == 1
        if not mentioned.any():
            continue
        for sentiment_idx in range(3):
            sentiment_target = true_s[mentioned, aspect_idx, sentiment_idx]
            if np.unique(sentiment_target).size < 2:
                thresholds_s[aspect_idx, sentiment_idx] = 0.5
                continue
            scores = [
                f1_score(
                    sentiment_target,
                    prob_s[mentioned, aspect_idx, sentiment_idx] >= threshold,
                    zero_division=0,
                )
                for threshold in grid
            ]
            thresholds_s[aspect_idx, sentiment_idx] = grid[int(np.argmax(scores))]
    return thresholds_m, thresholds_s


def apply_acsa_thresholds(
    prob_m: np.ndarray,
    prob_s: np.ndarray,
    thresholds_m: np.ndarray,
    thresholds_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_m = (prob_m >= thresholds_m[None, :]).astype(np.float32)
    pred_s = (prob_s >= thresholds_s[None, :, :]).astype(np.float32)
    pred_s *= pred_m[:, :, None]
    no_sentiment = (pred_m == 1) & (pred_s.sum(axis=2) == 0)
    pred_s[:, :, 2][no_sentiment] = 1
    return pred_m, pred_s


def compute_acsa_metrics(
    true_m: np.ndarray,
    pred_m: np.ndarray,
    true_s: np.ndarray,
    pred_s: np.ndarray,
    prob_m: Optional[np.ndarray] = None,
    prob_s: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    true_m_flat = true_m.reshape(-1)
    pred_m_flat = pred_m.reshape(-1)

    metrics["mention_precision_macro"] = float(
        precision_score(true_m_flat, pred_m_flat, average="macro", zero_division=0)
    )
    metrics["mention_recall_macro"] = float(
        recall_score(true_m_flat, pred_m_flat, average="macro", zero_division=0)
    )
    for average in ("macro", "micro", "weighted"):
        metrics[f"mention_f1_{average}"] = float(
            f1_score(true_m_flat, pred_m_flat, average=average, zero_division=0)
        )
    metrics["mention_auc_roc"] = (
        _safe_roc_auc(true_m_flat, prob_m.reshape(-1)) if prob_m is not None else None
    )
    metrics["mention_auc_pr"] = (
        _safe_average_precision(true_m_flat, prob_m.reshape(-1))
        if prob_m is not None
        else None
    )

    gold_mentioned = true_m.astype(bool).reshape(-1)
    true_s_gold = true_s.reshape(-1, 3)[gold_mentioned]
    pred_s_gold = pred_s.reshape(-1, 3)[gold_mentioned]
    prob_s_gold = prob_s.reshape(-1, 3)[gold_mentioned] if prob_s is not None else None
    if len(true_s_gold):
        metrics["sentiment_precision_macro"] = float(
            precision_score(true_s_gold, pred_s_gold, average="macro", zero_division=0)
        )
        metrics["sentiment_recall_macro"] = float(
            recall_score(true_s_gold, pred_s_gold, average="macro", zero_division=0)
        )
        for average in ("macro", "micro", "weighted", "samples"):
            metrics[f"sentiment_f1_{average}"] = float(
                f1_score(true_s_gold, pred_s_gold, average=average, zero_division=0)
            )
        metrics["sentiment_auc_roc"] = (
            _safe_roc_auc(true_s_gold, prob_s_gold) if prob_s_gold is not None else None
        )
        metrics["sentiment_auc_pr"] = (
            _safe_average_precision(true_s_gold, prob_s_gold)
            if prob_s_gold is not None
            else None
        )
    else:
        for key in (
            "sentiment_precision_macro",
            "sentiment_recall_macro",
            "sentiment_f1_macro",
            "sentiment_f1_micro",
            "sentiment_f1_weighted",
            "sentiment_f1_samples",
            "sentiment_auc_roc",
            "sentiment_auc_pr",
        ):
            metrics[key] = None

    gated_pred_s = pred_s * pred_m[:, :, None]
    true_pairs = true_s.reshape(len(true_s), -1)
    pred_pairs = gated_pred_s.reshape(len(gated_pred_s), -1)
    metrics["end_to_end_precision_micro"] = float(
        precision_score(true_pairs, pred_pairs, average="micro", zero_division=0)
    )
    metrics["end_to_end_recall_micro"] = float(
        recall_score(true_pairs, pred_pairs, average="micro", zero_division=0)
    )
    metrics["end_to_end_f1_micro"] = float(
        f1_score(true_pairs, pred_pairs, average="micro", zero_division=0)
    )
    metrics["end_to_end_f1_macro"] = float(
        f1_score(true_pairs, pred_pairs, average="macro", zero_division=0)
    )
    metrics["exact_match_accuracy"] = float(np.mean(np.all(true_pairs == pred_pairs, axis=1)))

    if metrics.get("sentiment_f1_macro") is not None:
        metrics["combined_f1_macro"] = float(
            0.5 * metrics["mention_f1_macro"] + 0.5 * metrics["sentiment_f1_macro"]
        )
        metrics["combined_f1_micro"] = float(
            0.5 * metrics["mention_f1_micro"] + 0.5 * metrics["sentiment_f1_micro"]
        )
        metrics["combined_f1_weighted"] = float(
            0.5 * metrics["mention_f1_weighted"]
            + 0.5 * metrics["sentiment_f1_weighted"]
        )
    else:
        metrics["combined_f1_macro"] = None
        metrics["combined_f1_micro"] = None
        metrics["combined_f1_weighted"] = None

    combined_inputs = (
        metrics.get("mention_f1_macro"),
        metrics.get("mention_auc_roc"),
        metrics.get("mention_auc_pr"),
        metrics.get("sentiment_f1_samples"),
        metrics.get("sentiment_f1_macro"),
        metrics.get("sentiment_auc_roc"),
        metrics.get("sentiment_auc_pr"),
    )
    if all(value is not None for value in combined_inputs):
        metrics["combined_score"] = float(
            0.20 * combined_inputs[0]
            + 0.10 * combined_inputs[1]
            + 0.10 * combined_inputs[2]
            + 0.20 * combined_inputs[3]
            + 0.15 * combined_inputs[4]
            + 0.15 * combined_inputs[5]
            + 0.10 * combined_inputs[6]
        )
    else:
        metrics["combined_score"] = None
    metrics["combined_f1"] = metrics["combined_score"]
    return metrics


def compute_global_sentiment_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
    }
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            labels=[0, 1, 2],
            average=average,
            zero_division=0,
        )
        metrics[f"precision_{average}"] = float(precision)
        metrics[f"recall_{average}"] = float(recall)
        metrics[f"f1_{average}"] = float(f1)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    metrics["per_class"] = {
        name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, name in enumerate(SENTIMENT_ORDER)
    }
    metrics["confusion_matrix"] = confusion_matrix(
        true_labels, pred_labels, labels=[0, 1, 2]
    ).astype(int).tolist()
    return metrics
