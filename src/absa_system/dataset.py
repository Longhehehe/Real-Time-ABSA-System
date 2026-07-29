"""PyTorch datasets backed by immutable model-ready JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import validate_model_record
from .tokenization import tokenize_model_record


def load_model_records(
    release_dir: Path,
    split: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"invalid split: {split}")
    path = release_dir.resolve() / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing model split: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                validate_model_record(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if row["split"] != split:
                raise ValueError(f"{path}:{line_number}: embedded split mismatch")
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"{path}: split is empty")
    return records


class ABSADataset(Dataset):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer,
        *,
        max_length: int,
        include_evidence: bool,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.include_evidence = bool(include_evidence)
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return tokenize_model_record(
            self.records[index],
            self.tokenizer,
            max_length=self.max_length,
            include_evidence=self.include_evidence,
        )


def collate_absa(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    tensor_keys = {
        "input_ids": torch.long,
        "attention_mask": torch.long,
        "content_mask": torch.long,
        "mention_labels": torch.float32,
        "sentiment_labels": torch.float32,
        "mention_evidence_mask": torch.float32,
        "polarity_evidence_mask": torch.float32,
        "mention_evidence_available": torch.float32,
        "polarity_evidence_available": torch.float32,
    }
    batch: dict[str, Any] = {}
    for key, dtype in tensor_keys.items():
        values = [np.asarray(item[key]) for item in items]
        batch[key] = torch.as_tensor(np.stack(values), dtype=dtype)
    batch["sample_id"] = [str(item["sample_id"]) for item in items]
    batch["leakage_group_id"] = [
        str(item["leakage_group_id"]) for item in items
    ]
    return batch
