"""Shared data contracts for benchmark preparation and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np


SENTIMENT_ORDER = ("NEG", "POS", "NEU")
SENTIMENT_TO_INDEX = {name: idx for idx, name in enumerate(SENTIMENT_ORDER)}


@dataclass
class SplitData:
    """One official dataset split in the normalized internal representation."""

    profile_id: str
    task: str
    split: str
    texts: List[str]
    sample_ids: List[str]
    aspects: List[str] = field(default_factory=list)
    labels_m: Optional[np.ndarray] = None
    labels_s: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    def validate(self) -> None:
        n = len(self.texts)
        if len(self.sample_ids) != n:
            raise ValueError(f"{self.profile_id}/{self.split}: sample_ids length mismatch")
        if any(not isinstance(text, str) or not text.strip() for text in self.texts):
            raise ValueError(f"{self.profile_id}/{self.split}: empty text found")

        if self.task == "acsa":
            a = len(self.aspects)
            if not a:
                raise ValueError(f"{self.profile_id}: ACSA profile has no aspects")
            if self.labels_m is None or self.labels_s is None:
                raise ValueError(f"{self.profile_id}/{self.split}: missing ACSA labels")
            if self.labels_m.shape != (n, a):
                raise ValueError(
                    f"{self.profile_id}/{self.split}: labels_m={self.labels_m.shape}, expected {(n, a)}"
                )
            if self.labels_s.shape != (n, a, 3):
                raise ValueError(
                    f"{self.profile_id}/{self.split}: labels_s={self.labels_s.shape}, expected {(n, a, 3)}"
                )
            if np.any((self.labels_m != 0) & (self.labels_m != 1)):
                raise ValueError(f"{self.profile_id}/{self.split}: labels_m must be binary")
            if np.any((self.labels_s != 0) & (self.labels_s != 1)):
                raise ValueError(f"{self.profile_id}/{self.split}: labels_s must be binary")
            if np.any(self.labels_s.sum(axis=2) > 0) and np.any(
                (self.labels_s.sum(axis=2) > 0) & (self.labels_m == 0)
            ):
                raise ValueError(f"{self.profile_id}/{self.split}: sentiment without aspect mention")
        elif self.task == "global_sentiment":
            if self.labels is None or self.labels.shape != (n,):
                shape = None if self.labels is None else self.labels.shape
                raise ValueError(
                    f"{self.profile_id}/{self.split}: labels={shape}, expected {(n,)}"
                )
            if np.any((self.labels < 0) | (self.labels >= 3)):
                raise ValueError(f"{self.profile_id}/{self.split}: sentiment index outside [0, 2]")
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def subset(self, indices: List[int]) -> "SplitData":
        idx = np.asarray(indices, dtype=int)
        return SplitData(
            profile_id=self.profile_id,
            task=self.task,
            split=self.split,
            texts=[self.texts[i] for i in indices],
            sample_ids=[self.sample_ids[i] for i in indices],
            aspects=list(self.aspects),
            labels_m=None if self.labels_m is None else self.labels_m[idx],
            labels_s=None if self.labels_s is None else self.labels_s[idx],
            labels=None if self.labels is None else self.labels[idx],
        )

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for i, text in enumerate(self.texts):
                row: Dict[str, Any] = {
                    "id": self.sample_ids[i],
                    "text": text,
                }
                if self.task == "acsa":
                    row["labels_m"] = self.labels_m[i].astype(int).tolist()
                    row["labels_s"] = self.labels_s[i].astype(int).tolist()
                else:
                    row["label"] = int(self.labels[i])
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        profile_id: str,
        task: str,
        split: str,
        aspects: List[str],
    ) -> "SplitData":
        texts: List[str] = []
        sample_ids: List[str] = []
        labels_m: List[Any] = []
        labels_s: List[Any] = []
        labels: List[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_ids.append(str(row["id"]))
                texts.append(str(row["text"]))
                if task == "acsa":
                    labels_m.append(row["labels_m"])
                    labels_s.append(row["labels_s"])
                else:
                    labels.append(int(row["label"]))
        result = cls(
            profile_id=profile_id,
            task=task,
            split=split,
            texts=texts,
            sample_ids=sample_ids,
            aspects=list(aspects),
            labels_m=np.asarray(labels_m, dtype=np.float32) if task == "acsa" else None,
            labels_s=np.asarray(labels_s, dtype=np.float32) if task == "acsa" else None,
            labels=np.asarray(labels, dtype=np.int64) if task == "global_sentiment" else None,
        )
        result.validate()
        return result


@dataclass
class PreparedProfile:
    """A complete train/dev/test benchmark profile plus its reproducibility audit."""

    profile_id: str
    task: str
    aspects: List[str]
    train: SplitData
    dev: SplitData
    test: SplitData
    audit: Dict[str, Any]
    metadata: Dict[str, Any]

    def validate(self) -> None:
        for split in (self.train, self.dev, self.test):
            split.validate()
            if split.profile_id != self.profile_id or split.task != self.task:
                raise ValueError("Prepared split profile/task mismatch")
            if split.aspects != self.aspects:
                raise ValueError("Prepared split aspect taxonomy mismatch")

        train_texts = set(self.train.texts)
        if train_texts.intersection(self.dev.texts):
            raise ValueError(f"{self.profile_id}: train/dev exact-text leakage remains")
        if train_texts.intersection(self.test.texts):
            raise ValueError(f"{self.profile_id}: train/test exact-text leakage remains")
