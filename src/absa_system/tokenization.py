"""Tokenizer loading and evidence-to-token alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from .schema import ASPECTS, POLARITIES


def load_offset_tokenizer(
    model_name_or_path: str,
    *,
    local_files_only: bool = False,
):
    """Load a fast tokenizer, upgrading PhoBERT's bundled tokenizer.json.

    Hugging Face currently resolves ``vinai/phobert-base`` to the Python slow
    tokenizer even when the cached snapshot contains a compatible
    ``tokenizer.json``. Evidence supervision needs character offsets, so this
    function upgrades that exact tokenizer while verifying vocabulary IDs are
    inherited from the same artifact.
    """

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if getattr(tokenizer, "is_fast", False):
        return tokenizer
    tokenizer_file = tokenizer.init_kwargs.get("tokenizer_file")
    if not tokenizer_file or not Path(tokenizer_file).is_file():
        warnings.warn(
            "Tokenizer is slow and has no tokenizer.json; evidence supervision "
            "will be unavailable.",
            RuntimeWarning,
        )
        return tokenizer
    special_tokens = {
        key: value
        for key, value in tokenizer.special_tokens_map.items()
        if isinstance(value, str)
    }
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        **special_tokens,
    )
    fast.name_or_path = tokenizer.name_or_path
    return fast


def build_evidence_masks(
    *,
    offsets: Sequence[Sequence[int]],
    special_tokens_mask: Sequence[int],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Map canonical character evidence to mention/polarity token targets."""

    sequence_length = len(offsets)
    if len(special_tokens_mask) != sequence_length:
        raise ValueError("offset/special token mask length mismatch")
    mention_mask = np.zeros((len(ASPECTS), sequence_length), dtype=np.float32)
    polarity_mask = np.zeros(
        (len(ASPECTS), len(POLARITIES), sequence_length),
        dtype=np.float32,
    )
    for span in evidence:
        aspect_index = int(span["aspect_index"])
        polarity_index = int(span["polarity_index"])
        start = int(span["start"])
        end = int(span["end"])
        if not (0 <= aspect_index < len(ASPECTS)):
            raise ValueError("evidence aspect index out of range")
        if not (0 <= polarity_index < len(POLARITIES)):
            raise ValueError("evidence polarity index out of range")
        for token_index, raw_offset in enumerate(offsets):
            token_start, token_end = int(raw_offset[0]), int(raw_offset[1])
            if special_tokens_mask[token_index]:
                continue
            if token_end <= token_start:
                continue
            if max(start, token_start) < min(end, token_end):
                mention_mask[aspect_index, token_index] = 1.0
                polarity_mask[
                    aspect_index,
                    polarity_index,
                    token_index,
                ] = 1.0
    mention_available = (mention_mask.sum(axis=-1) > 0).astype(np.float32)
    polarity_available = (polarity_mask.sum(axis=-1) > 0).astype(np.float32)
    return {
        "mention_evidence_mask": mention_mask,
        "polarity_evidence_mask": polarity_mask,
        "mention_evidence_available": mention_available,
        "polarity_evidence_available": polarity_available,
    }


def tokenize_model_record(
    row: Mapping[str, Any],
    tokenizer,
    *,
    max_length: int,
    include_evidence: bool,
) -> dict[str, Any]:
    request_offsets = include_evidence and getattr(tokenizer, "is_fast", False)
    encoded = tokenizer(
        row["reviewContent"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_offsets_mapping=request_offsets,
    )
    special_tokens_mask = encoded.pop("special_tokens_mask")
    offsets = encoded.pop("offset_mapping", None)
    content_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
    content_mask = content_mask * (
        1 - np.asarray(special_tokens_mask, dtype=np.int64)
    )
    output: dict[str, Any] = {
        "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
        "content_mask": content_mask,
        "mention_labels": np.asarray(row["mention_labels"], dtype=np.float32),
        "sentiment_labels": np.asarray(
            row["sentiment_labels"], dtype=np.float32
        ),
        "sample_id": row["sample_id"],
        "leakage_group_id": row["leakage_group_id"],
    }
    if request_offsets and offsets is not None:
        output.update(
            build_evidence_masks(
                offsets=offsets,
                special_tokens_mask=special_tokens_mask,
                evidence=row.get("evidence", []),
            )
        )
    else:
        sequence_length = len(output["input_ids"])
        output.update(
            {
                "mention_evidence_mask": np.zeros(
                    (len(ASPECTS), sequence_length), dtype=np.float32
                ),
                "polarity_evidence_mask": np.zeros(
                    (len(ASPECTS), len(POLARITIES), sequence_length),
                    dtype=np.float32,
                ),
                "mention_evidence_available": np.zeros(
                    len(ASPECTS), dtype=np.float32
                ),
                "polarity_evidence_available": np.zeros(
                    (len(ASPECTS), len(POLARITIES)), dtype=np.float32
                ),
            }
        )
    return output
