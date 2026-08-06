"""Metrics and development-only threshold selection for multi-polarity ABSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .schema import ASPECTS, POLARITIES


EVALUATION_PROTOCOL = "separate-aspect-polarity/1.0.0"


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int8).reshape(-1)
    tp = int(np.logical_and(truth == 1, prediction == 1).sum())
    fp = int(np.logical_and(truth == 0, prediction == 1).sum())
    fn = int(np.logical_and(truth == 1, prediction == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(truth.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.ndim == 1:
        return _binary_prf(truth, prediction)["f1"]
    return float(
        np.mean(
            [
                _binary_prf(truth[..., index], prediction[..., index])["f1"]
                for index in range(truth.shape[-1])
            ]
        )
    )


@dataclass(frozen=True)
class Thresholds:
    mention: np.ndarray
    sentiment: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "mention": np.asarray(self.mention, dtype=float).tolist(),
            "sentiment": np.asarray(self.sentiment, dtype=float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Thresholds":
        mention = np.asarray(payload["mention"], dtype=np.float64)
        sentiment = np.asarray(payload["sentiment"], dtype=np.float64)
        if mention.shape != (len(ASPECTS),):
            raise ValueError("mention threshold shape mismatch")
        if sentiment.shape != (len(ASPECTS), len(POLARITIES)):
            raise ValueError("sentiment threshold shape mismatch")
        return cls(mention=mention, sentiment=sentiment)


def tune_thresholds(
    mention_true: np.ndarray,
    sentiment_true: np.ndarray,
    mention_prob: np.ndarray,
    sentiment_prob: np.ndarray,
    *,
    grid: Iterable[float] = tuple(np.linspace(0.15, 0.85, 15)),
) -> Thresholds:
    """Tune every threshold exclusively on development labels."""

    mention_true = np.asarray(mention_true, dtype=np.int8)
    sentiment_true = np.asarray(sentiment_true, dtype=np.int8)
    mention_prob = np.asarray(mention_prob, dtype=np.float64)
    sentiment_prob = np.asarray(sentiment_prob, dtype=np.float64)
    if mention_true.shape != mention_prob.shape:
        raise ValueError("mention probability shape mismatch")
    if sentiment_true.shape != sentiment_prob.shape:
        raise ValueError("sentiment probability shape mismatch")
    values = np.asarray(tuple(grid), dtype=np.float64)
    if not len(values):
        raise ValueError("threshold grid cannot be empty")

    mention_thresholds = np.full(len(ASPECTS), 0.5, dtype=np.float64)
    sentiment_thresholds = np.full(
        (len(ASPECTS), len(POLARITIES)), 0.5, dtype=np.float64
    )
    for aspect_index in range(len(ASPECTS)):
        target = mention_true[:, aspect_index]
        scores = [
            _binary_prf(target, mention_prob[:, aspect_index] >= threshold)["f1"]
            for threshold in values
        ]
        mention_thresholds[aspect_index] = values[int(np.argmax(scores))]
        mentioned = target.astype(bool)
        for polarity_index in range(len(POLARITIES)):
            polarity_target = sentiment_true[mentioned, aspect_index, polarity_index]
            polarity_prob = sentiment_prob[mentioned, aspect_index, polarity_index]
            if not len(polarity_target) or len(np.unique(polarity_target)) < 2:
                sentiment_thresholds[aspect_index, polarity_index] = 0.5
                continue
            scores = [
                _binary_prf(polarity_target, polarity_prob >= threshold)["f1"]
                for threshold in values
            ]
            sentiment_thresholds[aspect_index, polarity_index] = values[
                int(np.argmax(scores))
            ]
    return Thresholds(mention_thresholds, sentiment_thresholds)


def apply_thresholds(
    mention_prob: np.ndarray,
    sentiment_prob: np.ndarray,
    thresholds: Thresholds,
    *,
    gate_polarity_by_mention: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply thresholds and enforce the canonical polarity constraints.

    Evaluation can keep polarity decisions independent from predicted aspect
    mentions. Production inference gates those decisions by predicted mentions.
    """

    mention_prob = np.asarray(mention_prob, dtype=np.float64)
    sentiment_prob = np.asarray(sentiment_prob, dtype=np.float64)
    predicted_mention = mention_prob >= thresholds.mention[None, :]
    predicted_sentiment = (
        sentiment_prob >= thresholds.sentiment[None, :, :]
    )

    batch_size, num_aspects = predicted_mention.shape
    for sample_index in range(batch_size):
        for aspect_index in range(num_aspects):
            values = predicted_sentiment[sample_index, aspect_index]
            probabilities = sentiment_prob[sample_index, aspect_index]
            if values[2] and (values[0] or values[1]):
                if probabilities[2] >= max(probabilities[0], probabilities[1]):
                    values[0] = False
                    values[1] = False
                else:
                    values[2] = False
            if not values.any():
                values[int(np.argmax(probabilities))] = True
    if gate_polarity_by_mention:
        predicted_sentiment[~predicted_mention] = False
    return predicted_mention.astype(np.int8), predicted_sentiment.astype(np.int8)


def compute_metrics(
    mention_true: np.ndarray,
    sentiment_true: np.ndarray,
    mention_pred: np.ndarray,
    sentiment_pred: np.ndarray,
) -> dict[str, Any]:
    """Score aspect detection and gold-aspect polarity classification separately."""
    mention_true = np.asarray(mention_true, dtype=np.int8)
    sentiment_true = np.asarray(sentiment_true, dtype=np.int8)
    mention_pred = np.asarray(mention_pred, dtype=np.int8)
    sentiment_pred = np.asarray(sentiment_pred, dtype=np.int8)
    if mention_true.shape != mention_pred.shape:
        raise ValueError("mention prediction shape mismatch")
    if sentiment_true.shape != sentiment_pred.shape:
        raise ValueError("sentiment prediction shape mismatch")

    mention_per_aspect = {
        aspect: _binary_prf(
            mention_true[:, aspect_index],
            mention_pred[:, aspect_index],
        )
        for aspect_index, aspect in enumerate(ASPECTS)
    }
    polarity_per_label: dict[str, dict[str, float]] = {}
    for aspect_index, aspect in enumerate(ASPECTS):
        mentioned = mention_true[:, aspect_index].astype(bool)
        for polarity_index, polarity in enumerate(POLARITIES):
            polarity_per_label[f"{aspect}::{polarity}"] = _binary_prf(
                sentiment_true[mentioned, aspect_index, polarity_index],
                sentiment_pred[mentioned, aspect_index, polarity_index],
            )

    gold_mention_mask = mention_true.astype(bool)
    polarity_true = sentiment_true[gold_mention_mask]
    polarity_pred = sentiment_pred[gold_mention_mask]
    flat_true = polarity_true.reshape(-1)
    flat_pred = polarity_pred.reshape(-1)

    sample_intersections: list[int] = []
    sample_unions: list[int] = []
    sample_exact: list[bool] = []
    for sample_index in range(len(sentiment_true)):
        sample_mask = gold_mention_mask[sample_index]
        sample_true = sentiment_true[sample_index, sample_mask].reshape(-1)
        sample_pred = sentiment_pred[sample_index, sample_mask].reshape(-1)
        sample_intersections.append(
            int(np.logical_and(sample_true, sample_pred).sum())
        )
        sample_unions.append(int(np.logical_or(sample_true, sample_pred).sum()))
        sample_exact.append(bool(np.array_equal(sample_true, sample_pred)))
    intersections = np.asarray(sample_intersections, dtype=np.int64)
    unions = np.asarray(sample_unions, dtype=np.int64)
    jaccard = np.where(unions > 0, intersections / np.maximum(unions, 1), 1.0)
    exact_match = np.asarray(sample_exact, dtype=bool)

    mixed_true = np.logical_and(polarity_true[:, 0], polarity_true[:, 1])
    mixed_pred = np.logical_and(polarity_pred[:, 0], polarity_pred[:, 1])
    polarity_micro = _binary_prf(flat_true, flat_pred)
    polarity_per_class = {
        polarity: _binary_prf(
            polarity_true[:, polarity_index],
            polarity_pred[:, polarity_index],
        )
        for polarity_index, polarity in enumerate(POLARITIES)
    }
    return {
        "num_samples": int(len(mention_true)),
        "mention_micro": _binary_prf(mention_true, mention_pred),
        "mention_macro_f1": float(
            np.mean([metrics["f1"] for metrics in mention_per_aspect.values()])
        ),
        "mention_per_aspect": mention_per_aspect,
        "polarity_micro": polarity_micro,
        "polarity_macro_f1": float(
            np.mean([metrics["f1"] for metrics in polarity_per_class.values()])
        ),
        "polarity_per_class": polarity_per_class,
        "polarity_per_label": polarity_per_label,
        "exact_set_match": float(exact_match.mean()),
        "sample_jaccard": float(jaccard.mean()),
        "hamming_loss": (
            float(np.not_equal(flat_true, flat_pred).mean())
            if flat_true.size
            else 0.0
        ),
        "mixed": _binary_prf(mixed_true, mixed_pred),
    }
