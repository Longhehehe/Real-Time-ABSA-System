"""Checkpoint-bound single/batch inference with evidence extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import pickle

import numpy as np
import torch

from .metrics import Thresholds, apply_thresholds
from .model_registry import build_neural_model, get_model_spec
from .schema import ASPECTS, POLARITIES
from .tokenization import load_offset_tokenizer
from .training import load_checkpoint, resolve_device


class ABSAPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device_name: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.device = resolve_device(device_name)
        self.checkpoint_path = checkpoint_path.resolve()
        if self.checkpoint_path.suffix.lower() == ".pkl":
            self._load_classical_checkpoint()
            return
        checkpoint = load_checkpoint(checkpoint_path, device="cpu")
        self.model_name = str(checkpoint.get("model_name", "phobert"))
        model_spec = get_model_spec(self.model_name)
        if not model_spec.iterative:
            raise ValueError("a classical checkpoint must use the .pkl extension")
        backbone_name = str(model_spec.tokenizer_name)
        self.max_length = int(checkpoint["max_length"])
        self.tokenizer = load_offset_tokenizer(
            backbone_name,
            local_files_only=local_files_only,
        )
        training_config = dict(checkpoint.get("training_config", {}))
        training_config["local_files_only"] = bool(local_files_only)
        self.model = build_neural_model(
            self.model_name,
            tokenizer=self.tokenizer,
            config=training_config,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.thresholds = Thresholds.from_dict(checkpoint["thresholds"])
        self.supports_evidence = model_spec.supports_evidence
        self.classical_model = None
        self.checkpoint_metadata = {
            "schema_version": checkpoint["schema_version"],
            "data_release_id": checkpoint["data_release_id"],
            "best_epoch": checkpoint["best_epoch"],
            "model_name": self.model_name,
            "fold": checkpoint.get("cross_validation", {}).get("fold"),
        }

    def _load_classical_checkpoint(self) -> None:
        with self.checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("schema_version") != "absa-classical-checkpoint/1.0.0":
            raise ValueError("unsupported classical checkpoint schema_version")
        self.model_name = str(checkpoint["model_name"])
        model_spec = get_model_spec(self.model_name)
        if model_spec.iterative:
            raise ValueError("invalid classical checkpoint model family")
        self.classical_model = checkpoint["model"]
        self.model = None
        self.tokenizer = None
        self.max_length = None
        self.supports_evidence = False
        self.thresholds = Thresholds.from_dict(checkpoint["thresholds"])
        self.checkpoint_metadata = {
            "schema_version": checkpoint["schema_version"],
            "data_release_id": checkpoint["data_release_id"],
            "best_epoch": None,
            "model_name": self.model_name,
            "fold": checkpoint.get("cross_validation", {}).get("fold"),
        }

    @staticmethod
    def _positive_probability(estimator: Any, features: Any) -> np.ndarray:
        probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        classes = np.asarray(estimator.classes_)
        positive = np.flatnonzero(classes == 1)
        if len(positive):
            return probabilities[:, int(positive[0])]
        return (
            np.ones(len(probabilities), dtype=np.float64)
            if classes[0] == 1
            else np.zeros(len(probabilities), dtype=np.float64)
        )

    def _predict_classical_probabilities(
        self, texts: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        fitted = self.classical_model
        features = fitted["vectorizer"].transform(list(texts))
        mention_prob = np.column_stack(
            [
                self._positive_probability(estimator, features)
                for estimator in fitted["mention_estimators"]
            ]
        )
        sentiment_prob = np.empty(
            (len(texts), len(ASPECTS), len(POLARITIES)), dtype=np.float64
        )
        for aspect_index, estimators in enumerate(fitted["sentiment_estimators"]):
            for polarity_index, estimator in enumerate(estimators):
                sentiment_prob[:, aspect_index, polarity_index] = (
                    self._positive_probability(estimator, features)
                )
        return mention_prob, sentiment_prob

    @torch.no_grad()
    def predict(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("predict requires non-empty review strings")
        offsets = None
        polarity_attention = None
        if self.classical_model is not None:
            mention_prob, sentiment_prob = self._predict_classical_probabilities(texts)
        else:
            encoded = self.tokenizer(
                list(texts),
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_offsets_mapping=(
                    self.supports_evidence
                    and getattr(self.tokenizer, "is_fast", False)
                ),
            )
            special_tokens_mask = encoded.pop("special_tokens_mask")
            offsets = encoded.pop("offset_mapping", None)
            content_mask = encoded["attention_mask"] * (1 - special_tokens_mask)
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            output = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                content_mask=content_mask.to(self.device),
            )
            mention_prob = torch.sigmoid(output.mention_logits).cpu().numpy()
            sentiment_prob = torch.sigmoid(output.sentiment_logits).cpu().numpy()
            if self.supports_evidence:
                polarity_attention = output.polarity_attention.cpu().numpy()
        mention_pred, sentiment_pred = apply_thresholds(
            mention_prob,
            sentiment_prob,
            self.thresholds,
        )
        results: list[dict[str, Any]] = []
        for sample_index, text in enumerate(texts):
            aspects: list[dict[str, Any]] = []
            for aspect_index, aspect in enumerate(ASPECTS):
                if not mention_pred[sample_index, aspect_index]:
                    continue
                polarities = [
                    POLARITIES[polarity_index]
                    for polarity_index in range(len(POLARITIES))
                    if sentiment_pred[
                        sample_index, aspect_index, polarity_index
                    ]
                ]
                evidence: list[dict[str, Any]] = []
                if offsets is not None and polarity_attention is not None:
                    raw_offsets = offsets[sample_index].tolist()
                    for polarity_index, polarity in enumerate(POLARITIES):
                        if polarity not in polarities:
                            continue
                        token_index = int(
                            np.argmax(
                                polarity_attention[
                                    sample_index,
                                    aspect_index,
                                    polarity_index,
                                ]
                            )
                        )
                        start, end = raw_offsets[token_index]
                        evidence.append(
                            {
                                "polarity": polarity,
                                "start": int(start),
                                "end": int(end),
                                "text": text[int(start) : int(end)],
                                "attention": float(
                                    polarity_attention[
                                        sample_index,
                                        aspect_index,
                                        polarity_index,
                                        token_index,
                                    ]
                                ),
                            }
                        )
                aspects.append(
                    {
                        "aspect": aspect,
                        "mention_probability": float(
                            mention_prob[sample_index, aspect_index]
                        ),
                        "polarities": polarities,
                        "polarity_probabilities": {
                            polarity: float(
                                sentiment_prob[
                                    sample_index,
                                    aspect_index,
                                    polarity_index,
                                ]
                            )
                            for polarity_index, polarity in enumerate(
                                POLARITIES
                            )
                        },
                        "evidence_peaks": evidence,
                    }
                )
            results.append(
                {
                    "schema_version": "absa-prediction/1.0.0",
                    "reviewContent": text,
                    "aspects": aspects,
                    "checkpoint": self.checkpoint_metadata,
                }
            )
        return results
