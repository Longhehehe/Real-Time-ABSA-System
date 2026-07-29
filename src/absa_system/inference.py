"""Checkpoint-bound single/batch inference with evidence extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .metrics import Thresholds, apply_thresholds
from .model import AspectEvidenceModel
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
        checkpoint = load_checkpoint(checkpoint_path, device="cpu")
        model_config = checkpoint["model_config"]
        backbone_name = str(model_config["backbone_name"])
        self.max_length = int(checkpoint["max_length"])
        self.tokenizer = load_offset_tokenizer(
            backbone_name,
            local_files_only=local_files_only,
        )
        self.model = AspectEvidenceModel(
            backbone_name=backbone_name,
            local_files_only=local_files_only,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.thresholds = Thresholds.from_dict(checkpoint["thresholds"])
        self.checkpoint_metadata = {
            "schema_version": checkpoint["schema_version"],
            "data_release_id": checkpoint["data_release_id"],
            "best_epoch": checkpoint["best_epoch"],
        }

    @torch.no_grad()
    def predict(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("predict requires non-empty review strings")
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=getattr(self.tokenizer, "is_fast", False),
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
        mention_pred, sentiment_pred = apply_thresholds(
            mention_prob,
            sentiment_prob,
            self.thresholds,
        )
        polarity_attention = output.polarity_attention.cpu().numpy()

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
                if offsets is not None:
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
