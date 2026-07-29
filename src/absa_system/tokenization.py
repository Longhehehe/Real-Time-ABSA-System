"""Tokenizer loading and evidence-to-token alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from .schema import ASPECTS, POLARITIES


class _ModelVocabAlignedFastTokenizer:
    """Expose fast offsets while returning the slow tokenizer's model IDs.

    PhoBERT's tokenizer.json stores some added/special tokens at backend IDs
    that are not the IDs expected by the pretrained embedding table. VinAI's
    historical ``PhobertTokenizerFast`` remapped those IDs in
    ``_convert_encoding``. This adapter performs the same compatibility step
    through the public tokenizer call so it remains usable across supported
    Transformers 4.x releases.
    """

    def __init__(self, fast_tokenizer, reference_tokenizer) -> None:
        self._fast_tokenizer = fast_tokenizer
        self._reference_tokenizer = reference_tokenizer

        source_vocab = {
            str(token): int(token_id)
            for token, token_id in fast_tokenizer.get_vocab().items()
        }
        target_vocab = {
            str(token): int(token_id)
            for token, token_id in reference_tokenizer.get_vocab().items()
        }
        if not source_vocab or not target_vocab:
            raise RuntimeError("tokenizer vocabulary is empty")
        source_ids = list(source_vocab.values())
        if min(source_ids) < 0:
            raise RuntimeError("fast tokenizer contains a negative token ID")

        unknown_id = int(reference_tokenizer.unk_token_id)
        source_max = max(source_ids)
        id_remap = np.full(source_max + 1, unknown_id, dtype=np.int64)
        self.model_vocab_size = max(target_vocab.values()) + 1
        preserved = min(self.model_vocab_size, len(id_remap))
        id_remap[:preserved] = np.arange(preserved, dtype=np.int64)
        self._id_remap = id_remap
        self.model_vocab_id_alignment = "fast_backend_to_reference_vocab"
        self.remapped_vocab_entries = sum(
            int(token_id >= self.model_vocab_size)
            for token_id in source_ids
        )

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return self.model_vocab_size

    def __len__(self) -> int:
        return self.model_vocab_size

    def __getattr__(self, name: str):
        try:
            reference = object.__getattribute__(
                self, "_reference_tokenizer"
            )
            return getattr(reference, name)
        except AttributeError:
            fast = object.__getattribute__(self, "_fast_tokenizer")
            return getattr(fast, name)

    def get_vocab(self, *args, **kwargs) -> dict[str, int]:
        del args, kwargs
        return {
            str(token): int(token_id)
            for token, token_id in self._reference_tokenizer.get_vocab().items()
        }

    def _validate_source_range(self, minimum: int, maximum: int) -> None:
        if minimum < 0 or maximum >= len(self._id_remap):
            raise ValueError(
                "fast tokenizer emitted an ID outside its declared vocabulary: "
                f"min={minimum}, max={maximum}, size={len(self._id_remap)}"
            )

    def _remap_input_ids(self, values):
        if isinstance(values, np.ndarray):
            if values.size == 0:
                return values.astype(np.int64, copy=False)
            self._validate_source_range(
                int(values.min()),
                int(values.max()),
            )
            return self._id_remap[values.astype(np.int64, copy=False)]

        try:
            import torch
        except ImportError:  # pragma: no cover - ML extra supplies torch.
            torch = None
        if torch is not None and isinstance(values, torch.Tensor):
            if values.numel() == 0:
                return values.to(dtype=torch.long)
            self._validate_source_range(
                int(values.min().item()),
                int(values.max().item()),
            )
            lookup = torch.as_tensor(
                self._id_remap,
                dtype=torch.long,
                device=values.device,
            )
            return lookup[values.to(dtype=torch.long)]

        if isinstance(values, tuple):
            return tuple(self._remap_input_ids(list(values)))
        if isinstance(values, list):
            if not values:
                return []
            if isinstance(values[0], (list, tuple, np.ndarray)):
                return [self._remap_input_ids(item) for item in values]
            numeric = np.asarray(values, dtype=np.int64)
            return self._remap_input_ids(numeric).tolist()
        raise TypeError(
            f"unsupported tokenizer input_ids type: {type(values).__name__}"
        )

    def __call__(self, *args, **kwargs):
        encoded = self._fast_tokenizer(*args, **kwargs)
        if "input_ids" not in encoded:
            raise RuntimeError("fast tokenizer output has no input_ids")
        encoded["input_ids"] = self._remap_input_ids(encoded["input_ids"])
        return encoded


def _tokenizer_probe_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=32,
    )
    values = encoded["input_ids"]
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return [int(value) for value in values]


def _tokenizers_match_model_ids(candidate, reference) -> bool:
    probes = (
        "",
        "sản phẩm tốt",
        "Máy hút mạnh, nhưng giao hàng chậm và hộp bị móp.",
        "t_triển ấn_đề ặc_biệt",
    )
    return all(
        _tokenizer_probe_ids(candidate, text)
        == _tokenizer_probe_ids(reference, text)
        for text in probes
    )


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

    reference = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        local_files_only=local_files_only,
    )
    candidate = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    model_vocab_size = max(
        int(token_id) for token_id in reference.get_vocab().values()
    ) + 1
    if getattr(candidate, "is_fast", False) and _tokenizers_match_model_ids(
        candidate, reference
    ):
        candidate.model_vocab_size = model_vocab_size
        candidate.model_vocab_id_alignment = "native"
        return candidate

    tokenizer_file = reference.init_kwargs.get("tokenizer_file")
    if not tokenizer_file or not Path(tokenizer_file).is_file():
        warnings.warn(
            "Tokenizer is slow and has no tokenizer.json; evidence supervision "
            "will be unavailable.",
            RuntimeWarning,
        )
        return reference
    special_tokens = {
        key: value
        for key, value in reference.special_tokens_map.items()
        if isinstance(value, str)
    }
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        model_max_length=int(reference.model_max_length),
        padding_side=str(reference.padding_side),
        truncation_side=str(reference.truncation_side),
        **special_tokens,
    )
    fast.name_or_path = reference.name_or_path
    aligned = _ModelVocabAlignedFastTokenizer(fast, reference)
    if not _tokenizers_match_model_ids(aligned, reference):
        raise RuntimeError(
            "fast tokenizer cannot reproduce PhoBERT model-compatible IDs"
        )
    return aligned


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
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    model_vocab_size = getattr(tokenizer, "model_vocab_size", None)
    if model_vocab_size is not None and input_ids.size:
        minimum_id = int(input_ids.min())
        maximum_id = int(input_ids.max())
        if minimum_id < 0 or maximum_id >= int(model_vocab_size):
            raise ValueError(
                "tokenizer emitted an ID outside the model vocabulary: "
                f"min={minimum_id}, max={maximum_id}, "
                f"model_vocab_size={int(model_vocab_size)}"
            )
    output: dict[str, Any] = {
        "input_ids": input_ids,
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
