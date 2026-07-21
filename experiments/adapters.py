"""Raw dataset adapters and reproducibility audit generation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
import hashlib
import json
import re
import unicodedata

import numpy as np

from .profiles import PROJECT_ROOT
from .schema import SENTIMENT_TO_INDEX, SplitData


def normalize_text(value: str) -> str:
    """Normalize representation only; do not perform semantic cleaning."""

    return " ".join(unicodedata.normalize("NFC", str(value).replace("\ufeff", "")).split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_acsa_arrays(n: int, num_aspects: int) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((n, num_aspects), dtype=np.float32),
        np.zeros((n, num_aspects, 3), dtype=np.float32),
    )


def load_uit_jsonl(
    profile_id: str,
    split: str,
    path: Path,
    config: Mapping[str, Any],
) -> Tuple[SplitData, Dict[str, Any]]:
    aspects = list(config["aspects"])
    aspect_to_idx = {name: idx for idx, name in enumerate(aspects)}
    polarity_map = config["polarity_map"]
    rows: List[Dict[str, Any]] = []
    invalid_spans: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            raw_text = str(raw.get("text", ""))
            if not raw_text.strip():
                raise ValueError(f"{path}:{line_number}: empty text")
            encoded: List[Tuple[int, int]] = []
            for label in raw.get("labels", []):
                if not isinstance(label, list) or len(label) != 3:
                    raise ValueError(f"{path}:{line_number}: malformed span label {label!r}")
                start, end, combined = int(label[0]), int(label[1]), str(label[2])
                if start < 0 or end < start or end > len(raw_text):
                    invalid_spans.append(
                        {"line": line_number, "start": start, "end": end, "text_length": len(raw_text)}
                    )
                    # This benchmark view evaluates review-level ACSA, not span
                    # boundaries. Keep the category/polarity label while making
                    # the source annotation issue explicit in dataset_audit.json.
                if "#" not in combined:
                    raise ValueError(f"{path}:{line_number}: malformed label {combined!r}")
                aspect, polarity = combined.rsplit("#", 1)
                if aspect not in aspect_to_idx:
                    raise ValueError(f"{path}:{line_number}: unknown aspect {aspect!r}")
                normalized_polarity = polarity_map.get(polarity)
                if normalized_polarity not in SENTIMENT_TO_INDEX:
                    raise ValueError(f"{path}:{line_number}: unknown polarity {polarity!r}")
                encoded.append((aspect_to_idx[aspect], SENTIMENT_TO_INDEX[normalized_polarity]))
            rows.append({"id": f"{split}-{line_number}", "text": normalize_text(raw_text), "encoded": encoded})

    labels_m, labels_s = _empty_acsa_arrays(len(rows), len(aspects))
    for row_idx, row in enumerate(rows):
        for aspect_idx, sentiment_idx in row["encoded"]:
            labels_m[row_idx, aspect_idx] = 1
            labels_s[row_idx, aspect_idx, sentiment_idx] = 1

    data = SplitData(
        profile_id=profile_id,
        task="acsa",
        split=split,
        texts=[row["text"] for row in rows],
        sample_ids=[row["id"] for row in rows],
        aspects=aspects,
        labels_m=labels_m,
        labels_s=labels_s,
    )
    data.validate()
    return data, {"invalid_spans": invalid_spans, "invalid_span_count": len(invalid_spans)}


_VLSP_LABEL_RE = re.compile(r"\{([^{}]+),\s*([^{}]+)\}")


def load_vlsp2018_txt(
    profile_id: str,
    split: str,
    path: Path,
    config: Mapping[str, Any],
) -> Tuple[SplitData, Dict[str, Any]]:
    aspects = list(config["aspects"])
    aspect_to_idx = {name: idx for idx, name in enumerate(aspects)}
    polarity_map = config["polarity_map"]
    content = path.read_text(encoding="utf-8-sig")
    blocks = [block for block in re.split(r"\r?\n\s*\r?\n", content.strip()) if block.strip()]
    rows: List[Dict[str, Any]] = []

    for block_number, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"{path}: malformed block {block_number}: expected at least 3 lines")
        sample_id = lines[0].lstrip("#") or f"{split}-{block_number}"
        label_line = lines[-1]
        text = normalize_text(" ".join(lines[1:-1]))
        if not text:
            raise ValueError(f"{path}: empty review in block {block_number}")
        encoded: List[Tuple[int, int]] = []
        for aspect, polarity in _VLSP_LABEL_RE.findall(label_line):
            aspect = aspect.strip()
            polarity = polarity.strip().lower()
            if aspect not in aspect_to_idx:
                raise ValueError(f"{path}: unknown aspect {aspect!r} in block {block_number}")
            normalized_polarity = polarity_map.get(polarity)
            if normalized_polarity not in SENTIMENT_TO_INDEX:
                raise ValueError(f"{path}: unknown polarity {polarity!r} in block {block_number}")
            encoded.append((aspect_to_idx[aspect], SENTIMENT_TO_INDEX[normalized_polarity]))
        if not encoded and "{" in label_line:
            raise ValueError(f"{path}: could not parse labels in block {block_number}")
        rows.append({"id": f"{split}-{sample_id}", "text": text, "encoded": encoded})

    labels_m, labels_s = _empty_acsa_arrays(len(rows), len(aspects))
    for row_idx, row in enumerate(rows):
        for aspect_idx, sentiment_idx in row["encoded"]:
            labels_m[row_idx, aspect_idx] = 1
            labels_s[row_idx, aspect_idx, sentiment_idx] = 1

    data = SplitData(
        profile_id=profile_id,
        task="acsa",
        split=split,
        texts=[row["text"] for row in rows],
        sample_ids=[row["id"] for row in rows],
        aspects=aspects,
        labels_m=labels_m,
        labels_s=labels_s,
    )
    data.validate()
    return data, {"invalid_spans": [], "invalid_span_count": 0}


def load_vlsp2016_tsv(
    profile_id: str,
    split: str,
    path: Path,
    config: Mapping[str, Any],
) -> Tuple[SplitData, Dict[str, Any]]:
    polarity_map = config["polarity_map"]
    texts: List[str] = []
    labels: List[int] = []
    ids: List[str] = []
    skipped_empty_lines: List[int] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                text, raw_label = line.rstrip("\r\n").rsplit("\t", 1)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: expected text<TAB>label") from exc
            normalized_label = polarity_map.get(raw_label.strip())
            if normalized_label not in SENTIMENT_TO_INDEX:
                raise ValueError(f"{path}:{line_number}: unknown label {raw_label!r}")
            normalized_text = normalize_text(text)
            if not normalized_text:
                skipped_empty_lines.append(line_number)
                continue
            ids.append(f"{split}-{line_number}")
            texts.append(normalized_text)
            labels.append(SENTIMENT_TO_INDEX[normalized_label])

    data = SplitData(
        profile_id=profile_id,
        task="global_sentiment",
        split=split,
        texts=texts,
        sample_ids=ids,
        aspects=[],
        labels=np.asarray(labels, dtype=np.int64),
    )
    data.validate()
    return data, {
        "invalid_spans": [],
        "invalid_span_count": 0,
        "skipped_empty_text_count": len(skipped_empty_lines),
        "skipped_empty_text_lines": skipped_empty_lines,
    }


LOADERS = {
    "uit_jsonl": load_uit_jsonl,
    "vlsp2018_txt": load_vlsp2018_txt,
    "vlsp2016_tsv": load_vlsp2016_tsv,
}


def _duplicate_count(texts: List[str]) -> int:
    return len(texts) - len(set(texts))


def _label_distribution(split: SplitData) -> Dict[str, Any]:
    if split.task == "global_sentiment":
        counts = Counter(int(value) for value in split.labels.tolist())
        return {name: counts.get(idx, 0) for idx, name in enumerate(("NEG", "POS", "NEU"))}
    aspect_mentions = split.labels_m.sum(axis=0).astype(int)
    sentiment_counts = split.labels_s.sum(axis=(0, 1)).astype(int)
    return {
        "aspect_mentions": {
            aspect: int(aspect_mentions[idx]) for idx, aspect in enumerate(split.aspects)
        },
        "sentiments": {
            name: int(sentiment_counts[idx]) for idx, name in enumerate(("NEG", "POS", "NEU"))
        },
        "empty_label_reviews": int((split.labels_m.sum(axis=1) == 0).sum()),
    }


def prepare_profile(
    profile_id: str,
    config: Mapping[str, Any],
    output_root: Path,
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    loader_name = str(config["loader"])
    if loader_name not in LOADERS:
        raise ValueError(f"Unsupported loader {loader_name!r} for {profile_id}")
    loader = LOADERS[loader_name]
    source_manifest: Dict[str, Any] = {}
    loaded: Dict[str, SplitData] = {}
    loader_audits: Dict[str, Any] = {}

    for split in ("train", "dev", "test"):
        raw_path = (project_root / config["splits"][split]).resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw split for {profile_id}/{split}: {raw_path}")
        source_manifest[split] = {
            "path": str(raw_path.relative_to(project_root)),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        }
        loaded[split], loader_audits[split] = loader(profile_id, split, raw_path, config)

    diagnostic_audit = None
    diagnostic_path_value = config.get("splits", {}).get("diagnostic")
    if diagnostic_path_value:
        diagnostic_path = (project_root / diagnostic_path_value).resolve()
        if diagnostic_path.exists():
            diagnostic, diagnostic_loader_audit = loader(
                profile_id, "diagnostic", diagnostic_path, config
            )
            diagnostic_audit = {
                "rows": len(diagnostic.texts),
                "internal_exact_duplicates": _duplicate_count(diagnostic.texts),
                "overlap_with_raw_train": len(set(diagnostic.texts).intersection(loaded["train"].texts)),
                "loader": diagnostic_loader_audit,
                "path": str(diagnostic_path.relative_to(project_root)),
                "bytes": diagnostic_path.stat().st_size,
                "sha256": sha256_file(diagnostic_path),
            }

    raw_train = loaded["train"]
    dev_texts = set(loaded["dev"].texts)
    test_texts = set(loaded["test"].texts)
    forbidden = dev_texts | test_texts
    keep_indices = [idx for idx, text in enumerate(raw_train.texts) if text not in forbidden]
    removed_indices = [idx for idx, text in enumerate(raw_train.texts) if text in forbidden]
    clean_train = raw_train.subset(keep_indices)
    clean_train.validate()
    loaded["train"] = clean_train

    split_stats: Dict[str, Any] = {}
    for split, data in loaded.items():
        split_stats[split] = {
            "rows": len(data.texts),
            "unique_texts": len(set(data.texts)),
            "internal_exact_duplicates": _duplicate_count(data.texts),
            "label_distribution": _label_distribution(data),
            **loader_audits[split],
        }

    audit: Dict[str, Any] = {
        "schema_version": 1,
        "profile_id": profile_id,
        "task": config["task"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_train_rows": len(raw_train.texts),
        "clean_train_rows": len(clean_train.texts),
        "removed_train_rows_due_to_dev_test_overlap": len(removed_indices),
        "raw_overlaps": {
            "train_dev_unique_texts": len(set(raw_train.texts).intersection(dev_texts)),
            "train_test_unique_texts": len(set(raw_train.texts).intersection(test_texts)),
            "dev_test_unique_texts": len(dev_texts.intersection(test_texts)),
        },
        "clean_overlaps": {
            "train_dev_unique_texts": len(set(clean_train.texts).intersection(dev_texts)),
            "train_test_unique_texts": len(set(clean_train.texts).intersection(test_texts)),
        },
        "splits": split_stats,
        "diagnostic_split": diagnostic_audit,
        "source_manifest": source_manifest,
    }

    profile_root = output_root / profile_id
    profile_root.mkdir(parents=True, exist_ok=True)
    for split, data in loaded.items():
        data.to_jsonl(profile_root / f"{split}.jsonl")
    (profile_root / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "profile_id": profile_id,
        "task": config["task"],
        "aspects": config.get("aspects", []),
        "sentiment_order": ["NEG", "POS", "NEU"],
        "description": config.get("description"),
        "source_url": config.get("source_url"),
        "citation": config.get("citation"),
        "span_policy": config.get("span_policy"),
        "source_manifest": source_manifest,
    }
    (profile_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit
