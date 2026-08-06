"""Build a frozen, annotation-ready release from canonical V2 crawl records.

Only records using ``substantive_vi_v2`` are admitted.  Raw crawl files are
never modified.  The release contains exact legacy-compatible CSV templates,
separate provenance, conservative duplicate flags, and SHA-256 inventories.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import unicodedata
from typing import Any, Iterable
from urllib.parse import urlparse

from lazada_collector.dom import stable_dom_review_id
from lazada_collector.normalization import duplicate_key
from lazada_collector.plan import load_plan
from lazada_collector.quality import evaluate_review
from lazada_collector.schema import SCHEMA_VERSION


POLICY = "substantive_vi_v2"
ASPECT_COLUMNS = [
    "Chất lượng sản phẩm",
    "Hiệu năng & Trải nghiệm",
    "Đúng mô tả",
    "Giá cả & Khuyến mãi",
    "Vận chuyển",
    "Đóng gói",
    "Dịch vụ & Thái độ Shop",
    "Bảo hành & Đổi trả",
    "Tính xác thực",
]
LEGACY_COLUMNS = ["reviewContent", *ASPECT_COLUMNS]
TOKEN_RE = re.compile(r"\b[^\W_]+\b", flags=re.UNICODE)
CLAUSE_CAPTURE_RE = re.compile(r"([^.!?…;\n]+)([.!?…;\n]+|$)")
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REQUIRED_REVIEW_FIELDS = {
    "schema_version",
    "crawl_id",
    "collected_at",
    "collector_version",
    "collection_transport",
    "sampling_frame",
    "query",
    "product_id",
    "seller_id",
    "source_url",
    "review_id",
    "review_text",
    "rating",
    "review_time",
    "sku_info",
    "has_images",
    "verified_purchase",
    "page_number",
    "response_sha256",
    "category",
    "selection_policy",
    "quality_score",
    "char_count",
    "word_count",
    "unique_word_ratio",
    "vietnamese_signal_count",
    "foreign_script_ratio",
}
FORBIDDEN_PERSONAL_OR_SECRET_KEYS = {
    "avatar",
    "buyer",
    "buyer_id",
    "buyer_name",
    "cookie",
    "cookies",
    "email",
    "phone",
    "username",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_RE.findall(normalized)


def punctuation_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters = (
        character if character.isalnum() or character.isspace() else " "
        for character in normalized
    )
    return " ".join("".join(characters).split())


def collapse_internal_repetition(text: str) -> tuple[bool, str]:
    """Return whether a substantive clause repeats and a de-inflated view.

    The original text is never changed in the canonical corpus.  The collapsed
    text exists only to detect reviews that passed the minimum-length policy
    because the same sentence was copied repeatedly.
    """
    seen: set[str] = set()
    kept: list[str] = []
    repeated = False
    for match in CLAUSE_CAPTURE_RE.finditer(text):
        original = match.group(0)
        clause = match.group(1).strip()
        if not clause:
            continue
        key = " ".join(
            unicodedata.normalize("NFKC", clause).casefold().split()
        )
        is_substantive = len(key) >= 20 and len(TOKEN_RE.findall(key)) >= 4
        if is_substantive and key in seen:
            repeated = True
            continue
        if is_substantive:
            seen.add(key)
        kept.append(original)
    collapsed = "".join(kept).strip()
    return repeated, collapsed


def has_internal_repeated_sentence(text: str) -> bool:
    repeated, _collapsed = collapse_internal_repetition(text)
    return repeated


def _five_grams(text: str) -> frozenset[tuple[str, ...]]:
    tokens = _tokens(text)
    values: set[tuple[str, ...]] = set()
    for index in range(max(0, len(tokens) - 4)):
        values.add(tuple(tokens[index : index + 5]))
    return frozenset(values)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def find_near_duplicate_pairs(
    texts: list[str],
    *,
    threshold: float = 0.85,
) -> tuple[
    list[tuple[int, int, float]],
    list[frozenset[tuple[str, ...]]],
]:
    """Find all pairs meeting the exact 5-gram Jaccard threshold.

    This is the deterministic AllPairs prefix-filter algorithm.  Unlike
    probing a fixed number of rare grams, its candidate generation has no
    false negatives for the requested threshold; every candidate is then
    verified with exact integer arithmetic.
    """
    ratio = Fraction(str(threshold)).limit_denominator(1_000_000)
    numerator = ratio.numerator
    denominator = ratio.denominator
    gram_sets = [_five_grams(text) for text in texts]
    document_frequency: Counter[tuple[str, ...]] = Counter()
    for grams in gram_sets:
        for gram in grams:
            document_frequency[gram] += 1

    ordered_grams = [
        sorted(
            grams,
            key=lambda gram: (document_frequency[gram], gram),
        )
        for grams in gram_sets
    ]
    processing_order = sorted(
        range(len(texts)),
        key=lambda index: (len(gram_sets[index]), index),
    )
    prefix_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    candidate_pairs: set[tuple[int, int]] = set()
    for current in processing_order:
        current_size = len(gram_sets[current])
        if not current_size:
            continue
        required_overlap = (
            numerator * current_size + denominator - 1
        ) // denominator
        prefix_length = current_size - required_overlap + 1
        for gram in ordered_grams[current][:prefix_length]:
            for previous in prefix_index[gram]:
                previous_size = len(gram_sets[previous])
                if denominator * previous_size < numerator * current_size:
                    continue
                candidate_pairs.add(
                    (min(previous, current), max(previous, current))
                )
        for gram in ordered_grams[current][:prefix_length]:
            prefix_index[gram].append(current)

    matches: list[tuple[int, int, float]] = []
    for left, right in sorted(candidate_pairs):
        left_grams = gram_sets[left]
        right_grams = gram_sets[right]
        intersection = len(left_grams & right_grams)
        union = len(left_grams | right_grams)
        if union and denominator * intersection >= numerator * union:
            score = intersection / union
            matches.append((left, right, score))
    return matches, gram_sets


def _sample_id(row: dict[str, Any]) -> str:
    payload = "\0".join(
        (
            str(row.get("product_id") or ""),
            str(row.get("review_id") or ""),
            duplicate_key(str(row.get("review_text") or "")),
        )
    )
    return "lzv1-" + _sha256_bytes(payload.encode("utf-8"))[:24]


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).strip().casefold()
            if normalized_key in FORBIDDEN_PERSONAL_OR_SECRET_KEYS:
                return str(key)
            found = _contains_forbidden_key(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _contains_forbidden_key(nested)
            if found:
                return found
    return None


def _validate_review_row(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    quality_result: Any,
) -> None:
    location = f"{path}:{line_number}"
    missing = sorted(REQUIRED_REVIEW_FIELDS - row.keys())
    if missing:
        raise ValueError(f"Missing required fields at {location}: {missing}")
    for key in (
        "crawl_id",
        "collected_at",
        "collector_version",
        "sampling_frame",
        "product_id",
        "source_url",
        "review_id",
        "review_text",
    ):
        if not str(row.get(key) or "").strip():
            raise ValueError(f"Required value {key!r} is blank at {location}")
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected schema_version at {location}: "
            f"{row.get('schema_version')!r}"
        )
    forbidden = _contains_forbidden_key(row)
    if forbidden:
        raise ValueError(f"Forbidden personal/secret key at {location}: {forbidden}")

    if str(row.get("crawl_id")) != path.parent.name:
        raise ValueError(f"crawl_id does not match run directory at {location}")
    if row.get("sampling_frame") != "natural":
        raise ValueError(f"Non-natural sampling frame at {location}")
    try:
        timestamp = datetime.fromisoformat(
            str(row.get("collected_at")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(f"Invalid collected_at at {location}") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"collected_at lacks timezone at {location}")

    rating = row.get("rating")
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError(f"Invalid rating at {location}: {rating!r}")
    page_number = row.get("page_number")
    if (
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number < 1
    ):
        raise ValueError(f"Invalid page_number at {location}: {page_number!r}")

    source_url = str(row.get("source_url") or "")
    parsed_source_url = urlparse(source_url)
    hostname = (parsed_source_url.hostname or "").casefold()
    if parsed_source_url.scheme.casefold() != "https":
        raise ValueError(f"Non-HTTPS source_url at {location}: {source_url!r}")
    if hostname != "lazada.vn" and not hostname.endswith(".lazada.vn"):
        raise ValueError(f"Non-Lazada source_url at {location}: {source_url!r}")
    response_sha256 = str(row.get("response_sha256") or "")
    if not HEX_SHA256_RE.fullmatch(response_sha256):
        raise ValueError(f"Invalid response_sha256 at {location}")

    transport = str(row.get("collection_transport") or "")
    review_id = str(row.get("review_id") or "")
    if transport == "selenium_dom":
        expected_review_id = stable_dom_review_id(
            str(row.get("product_id") or ""),
            str(row.get("review_text") or ""),
            str(row.get("review_time") or ""),
            str(row.get("sku_info") or ""),
        )
        if review_id != expected_review_id:
            raise ValueError(f"DOM review_id checksum mismatch at {location}")
    elif transport in {"requests", "requests_cookie"}:
        if not review_id.isdigit():
            raise ValueError(f"API review_id is not numeric at {location}")
    else:
        raise ValueError(f"Unknown collection_transport at {location}: {transport!r}")

    stored_metrics = {
        "quality_score": quality_result.score,
        "char_count": quality_result.char_count,
        "word_count": quality_result.word_count,
        "unique_word_ratio": quality_result.unique_word_ratio,
        "vietnamese_signal_count": quality_result.vietnamese_signal_count,
        "foreign_script_ratio": quality_result.foreign_script_ratio,
    }
    mismatches = {
        key: {"stored": row.get(key), "recomputed": expected}
        for key, expected in stored_metrics.items()
        if row.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Stored quality metrics mismatch at {location}: {mismatches}")


def _validate_raw_date(value: str | None, *, option: str) -> str | None:
    if value is None:
        return None
    if not RAW_DATE_RE.fullmatch(value):
        raise ValueError(f"{option} must use YYYY-MM-DD format")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{option} is not a valid calendar date") from exc
    return value


def _path_in_raw_date_scope(
    path: Path,
    raw_root: Path,
    *,
    raw_date_from: str | None,
    raw_date_through: str | None,
) -> bool:
    if raw_date_from is None and raw_date_through is None:
        return True
    relative = path.relative_to(raw_root)
    if not relative.parts or not RAW_DATE_RE.fullmatch(relative.parts[0]):
        return False
    raw_date = relative.parts[0]
    if raw_date_from is not None and raw_date < raw_date_from:
        return False
    if raw_date_through is not None and raw_date > raw_date_through:
        return False
    return True


def _scoped_paths(
    raw_root: Path,
    pattern: str,
    *,
    raw_date_from: str | None,
    raw_date_through: str | None,
) -> list[Path]:
    return sorted(
        path
        for path in raw_root.rglob(pattern)
        if _path_in_raw_date_scope(
            path,
            raw_root,
            raw_date_from=raw_date_from,
            raw_date_through=raw_date_through,
        )
    )


def _read_canonical(
    raw_root: Path,
    plan_path: Path,
    *,
    raw_date_from: str | None,
    raw_date_through: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = load_plan(plan_path).quality
    rows: list[dict[str, Any]] = []
    raw_count = 0
    excluded_policy = Counter()
    malformed = 0
    response_origins: dict[str, set[tuple[str, int, str]]] = defaultdict(set)

    for path in _scoped_paths(
        raw_root,
        "reviews.jsonl",
        raw_date_from=raw_date_from,
        raw_date_through=raw_date_through,
    ):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed += 1
                    raise ValueError(
                        f"Malformed JSON in {path}:{line_number}"
                    ) from exc
                selection_policy = str(row.get("selection_policy") or "")
                if selection_policy != POLICY:
                    excluded_policy[selection_policy or "<missing>"] += 1
                    continue
                result = evaluate_review(
                    str(row.get("review_text") or ""),
                    policy,
                )
                if not result.accepted:
                    raise ValueError(
                        f"Canonical record fails current quality policy: "
                        f"{path}:{line_number}"
                    )
                _validate_review_row(
                    row,
                    path=path,
                    line_number=line_number,
                    quality_result=result,
                )
                copied = dict(row)
                copied["_source_relative_path"] = str(
                    path.relative_to(raw_root.parent.parent)
                ).replace("\\", "/")
                copied["_source_line"] = line_number
                copied["_sample_id"] = _sample_id(row)
                rows.append(copied)
                response_origins[str(row["response_sha256"])].add(
                    (
                        str(row["product_id"]),
                        int(row["page_number"]),
                        str(row["collection_transport"]),
                    )
                )

    rows.sort(
        key=lambda row: (
            str(row.get("product_id") or ""),
            str(row.get("review_id") or ""),
            str(row.get("collected_at") or ""),
            str(row.get("crawl_id") or ""),
            str(row.get("_source_relative_path") or ""),
            int(row.get("_source_line") or 0),
        )
    )
    ambiguous_digests = {
        digest: sorted(origins)
        for digest, origins in response_origins.items()
        if len(origins) > 1
    }
    if ambiguous_digests:
        digest, origins = next(iter(sorted(ambiguous_digests.items())))
        raise ValueError(
            "response_sha256 maps to multiple product/page/transport origins: "
            f"{digest} -> {origins}"
        )
    review_ids = [str(row.get("review_id") or "") for row in rows]
    text_keys = [
        duplicate_key(str(row.get("review_text") or ""))
        for row in rows
    ]
    sample_ids = [str(row["_sample_id"]) for row in rows]
    if len(set(review_ids)) != len(rows):
        raise ValueError("Canonical V2 contains duplicate review IDs")
    if len(set(text_keys)) != len(rows):
        raise ValueError("Canonical V2 contains exact normalized duplicate text")
    if len(set(sample_ids)) != len(rows):
        raise ValueError("Stable sample ID collision detected")
    return rows, {
        "raw_review_records": raw_count,
        "canonical_records": len(rows),
        "excluded_by_policy": dict(excluded_policy),
        "malformed_records": malformed,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
    return count


def _legacy_row(review_text: str) -> list[str]:
    return [review_text, *([""] * len(ASPECT_COLUMNS))]


def _write_legacy_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> int:
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(LEGACY_COLUMNS)
        for row in rows:
            writer.writerow(_legacy_row(str(row.get("review_text") or "")))
            count += 1
    return count


def _source_inventory(
    raw_root: Path,
    project_root: Path,
    *,
    raw_date_from: str | None,
    raw_date_through: str | None,
) -> tuple[int, str, str]:
    source_files = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file()
        and _path_in_raw_date_scope(
            path,
            raw_root,
            raw_date_from=raw_date_from,
            raw_date_through=raw_date_through,
        )
    )
    temporary_files = [
        path for path in source_files if path.name.endswith(".tmp")
    ]
    if temporary_files:
        raise ValueError(
            "Raw snapshot contains unfinished temporary files: "
            + ", ".join(str(path) for path in temporary_files[:5])
        )
    lines = []
    for path in source_files:
        relative = str(path.relative_to(project_root)).replace("\\", "/")
        lines.append(f"{_sha256_file(path)}  {relative}")
    text = "\n".join(lines) + "\n"
    return (
        len(source_files),
        text,
        _sha256_bytes(text.encode("utf-8")),
    )


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSON in {path}:{line_number}")
            count += 1
    return count


def _validate_manifests(
    raw_root: Path,
    *,
    raw_date_from: str | None,
    raw_date_through: str | None,
) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    aggregate = Counter()
    manifest_count = 0
    for path in _scoped_paths(
        raw_root,
        "manifest.json",
        raw_date_from=raw_date_from,
        raw_date_through=raw_date_through,
    ):
        manifest_count += 1
        manifest = json.loads(path.read_text(encoding="utf-8"))
        crawl_id = str(manifest.get("crawl_id") or "")
        if crawl_id != path.parent.name:
            raise ValueError(f"Manifest crawl_id mismatch: {path}")
        status = str(manifest.get("status") or "<missing>")
        if status == "running":
            raise ValueError(f"Refusing to release a running crawl: {crawl_id}")
        statuses[status] += 1

        physical = {
            "products": _jsonl_count(path.parent / "products.jsonl"),
            "reviews_written": _jsonl_count(path.parent / "reviews.jsonl"),
            "reviews_rejected_quality": _jsonl_count(
                path.parent / "rejections.jsonl"
            ),
        }
        counts = manifest.get("counts") or {}
        expected = {
            "products": int(counts.get("products") or 0),
            "reviews_written": int(counts.get("reviews_written") or 0),
            "reviews_rejected_quality": int(
                counts.get("reviews_rejected_quality") or 0
            ),
        }
        if physical != expected:
            raise ValueError(
                f"Manifest counters do not match physical JSONL files for "
                f"{crawl_id}: physical={physical}, manifest={expected}"
            )
        aggregate.update(physical)
    if manifest_count == 0:
        raise ValueError(f"No crawl manifests found below {raw_root}")
    return {
        "manifests": manifest_count,
        "aggregate_physical_counts": {
            key: aggregate[key] for key in sorted(aggregate)
        },
        "statuses": {key: statuses[key] for key in sorted(statuses)},
    }


def _write_code_inventory(
    project_root: Path,
    destination: Path,
) -> tuple[int, str]:
    relative_paths = [
        Path("scripts/build_corpus_release.py"),
        Path("src/lazada_collector/dom.py"),
        Path("src/lazada_collector/normalization.py"),
        Path("src/lazada_collector/plan.py"),
        Path("src/lazada_collector/quality.py"),
        Path("src/lazada_collector/schema.py"),
        Path("pyproject.toml"),
    ]
    lines = []
    for relative in relative_paths:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Release code dependency not found: {path}")
        lines.append(
            f"{_sha256_file(path)}  "
            f"{str(relative).replace(chr(92), '/')}"
        )
    text = "\n".join(lines) + "\n"
    destination.write_text(text, encoding="utf-8", newline="\n")
    return len(lines), _sha256_bytes(text.encode("utf-8"))


def _artifact(path: Path, root: Path, records: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def build_release(
    *,
    raw_root: Path,
    raw_date_from: str | None,
    raw_date_through: str | None,
    output: Path,
    plan_path: Path,
    collector_config: Path,
    annotation_guideline: Path,
    batch_size: int,
    near_threshold: float,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Release output already exists: {output}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    project_root = Path.cwd().resolve()
    raw_root = raw_root.resolve()
    output = output.resolve()
    raw_date_from = _validate_raw_date(
        raw_date_from,
        option="--raw-date-from",
    )
    raw_date_through = _validate_raw_date(
        raw_date_through,
        option="--raw-date-through",
    )
    if (
        raw_date_from is not None
        and raw_date_through is not None
        and raw_date_from > raw_date_through
    ):
        raise ValueError("--raw-date-from must not be after --raw-date-through")
    try:
        raw_root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("raw_root must be inside the project workspace") from exc
    try:
        output.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("Release output must not be inside the raw source tree")
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"Temporary build output already exists: {temporary}")

    (
        source_file_count,
        initial_source_inventory,
        source_inventory_sha256,
    ) = _source_inventory(
        raw_root,
        project_root,
        raw_date_from=raw_date_from,
        raw_date_through=raw_date_through,
    )
    manifest_validation = _validate_manifests(
        raw_root,
        raw_date_from=raw_date_from,
        raw_date_through=raw_date_through,
    )
    rows, source_counts = _read_canonical(
        raw_root,
        plan_path,
        raw_date_from=raw_date_from,
        raw_date_through=raw_date_through,
    )
    manifest_review_total = manifest_validation[
        "aggregate_physical_counts"
    ]["reviews_written"]
    if source_counts["raw_review_records"] != manifest_review_total:
        raise ValueError(
            "Aggregate manifest review count does not match parsed raw rows: "
            f"{manifest_review_total} != "
            f"{source_counts['raw_review_records']}"
        )
    texts = [str(row.get("review_text") or "") for row in rows]
    plan = load_plan(plan_path)
    allowed_categories = {segment.name for segment in plan.segments}

    repeated_sentence_indices: set[int] = set()
    repetition_inflated_indices: set[int] = set()
    for index, text in enumerate(texts):
        repeated, collapsed = collapse_internal_repetition(text)
        if not repeated:
            continue
        repeated_sentence_indices.add(index)
        if not evaluate_review(collapsed, plan.quality).accepted:
            repetition_inflated_indices.add(index)

    def representative_rank(index: int) -> tuple[Any, ...]:
        row = rows[index]
        return (
            index in repetition_inflated_indices,
            -float(row.get("quality_score") or 0.0),
            -int(row.get("word_count") or 0),
            -int(row.get("char_count") or 0),
            str(row["_sample_id"]),
        )

    punctuation_groups: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(texts):
        punctuation_groups[punctuation_key(text)].append(index)
    punctuation_duplicate_indices: set[int] = set()
    punctuation_group_for_index: dict[int, str] = {}
    punctuation_representative_for_index: dict[int, int] = {}
    punctuation_group_count = 0
    for indices in punctuation_groups.values():
        if len(indices) <= 1:
            continue
        punctuation_group_count += 1
        representative = min(indices, key=representative_rank)
        member_ids = sorted(str(rows[index]["_sample_id"]) for index in indices)
        group_id = "punct-" + _sha256_bytes(
            "\0".join(member_ids).encode("utf-8")
        )[:16]
        for index in indices:
            punctuation_group_for_index[index] = group_id
            punctuation_representative_for_index[index] = representative
            if index != representative:
                punctuation_duplicate_indices.add(index)

    near_pairs, gram_sets = find_near_duplicate_pairs(
        texts,
        threshold=near_threshold,
    )
    union_find = _UnionFind(len(rows))
    for left, right, _score in near_pairs:
        union_find.union(left, right)
    near_components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        near_components[union_find.find(index)].append(index)
    near_components = {
        root: indices
        for root, indices in near_components.items()
        if len(indices) > 1
    }

    cluster_for_index: dict[int, str] = {}
    representative_for_index: dict[int, int] = {}
    near_duplicate_indices: set[int] = set()
    for indices in near_components.values():
        representative = min(indices, key=representative_rank)
        cluster_payload = "\0".join(
            sorted(str(rows[index]["_sample_id"]) for index in indices)
        )
        threshold_ratio = Fraction(str(near_threshold)).limit_denominator(
            1_000_000
        )
        cluster_id = (
            f"nd5w-j{threshold_ratio.numerator}of"
            f"{threshold_ratio.denominator}-"
            + _sha256_bytes(
            cluster_payload.encode("utf-8")
            )[:16]
        )
        for index in indices:
            cluster_for_index[index] = cluster_id
            representative_for_index[index] = representative
            if index != representative:
                near_duplicate_indices.add(index)

    audit_rows: list[dict[str, Any]] = []
    primary_indices: list[int] = []
    review_required_indices: list[int] = []
    flags_by_index: dict[int, list[str]] = {}
    hard_review_indices = (
        punctuation_duplicate_indices
        | near_duplicate_indices
        | repetition_inflated_indices
    )
    for index, row in enumerate(rows):
        flags: list[str] = []
        if index in punctuation_duplicate_indices:
            flags.append("punctuation_variant_duplicate")
        if index in near_duplicate_indices:
            flags.append("near_duplicate_nonrepresentative")
        if index in repeated_sentence_indices:
            flags.append("internal_repeated_sentence")
        if index in repetition_inflated_indices:
            flags.append("repetition_inflated_quality")
        category = str(row.get("category") or "").strip()
        if not category:
            flags.append("category_missing")
        elif category not in allowed_categories:
            flags.append("category_unmapped")
        flags_by_index[index] = flags
        if index in hard_review_indices:
            review_required_indices.append(index)
        else:
            primary_indices.append(index)
        representative = representative_for_index.get(index, index)
        punctuation_representative = punctuation_representative_for_index.get(
            index,
            index,
        )
        review_text = str(row.get("review_text") or "")
        audit_rows.append(
            {
                "sample_id": row["_sample_id"],
                "review_id": row.get("review_id"),
                "product_id": row.get("product_id"),
                "review_text_sha256": _sha256_bytes(
                    review_text.encode("utf-8")
                ),
                "normalized_text_sha256": duplicate_key(review_text),
                "flags": flags,
                "near_duplicate_cluster": cluster_for_index.get(index),
                "near_duplicate_representative_sample_id": rows[
                    representative
                ]["_sample_id"],
                "punctuation_duplicate_group": punctuation_group_for_index.get(
                    index
                ),
                "punctuation_representative_sample_id": rows[
                    punctuation_representative
                ]["_sample_id"],
                "annotation_eligible": index not in hard_review_indices,
                "requires_adjudication": index in hard_review_indices,
                "source_relative_path": row["_source_relative_path"],
                "source_line": row["_source_line"],
            }
        )

    near_pair_rows = []
    for left, right, score in near_pairs:
        intersection = len(gram_sets[left] & gram_sets[right])
        union = len(gram_sets[left] | gram_sets[right])
        near_pair_rows.append(
            {
                "left_sample_id": rows[left]["_sample_id"],
                "right_sample_id": rows[right]["_sample_id"],
                "intersection_5grams": intersection,
                "union_5grams": union,
                "jaccard": round(score, 12),
            }
        )

    near_cluster_rows = []
    for indices in sorted(
        near_components.values(),
        key=lambda values: sorted(
            str(rows[index]["_sample_id"]) for index in values
        ),
    ):
        representative = representative_for_index[indices[0]]
        near_cluster_rows.append(
            {
                "cluster_id": cluster_for_index[indices[0]],
                "representative_sample_id": rows[representative]["_sample_id"],
                "member_sample_ids": sorted(
                    str(rows[index]["_sample_id"]) for index in indices
                ),
            }
        )

    temporary.mkdir(parents=True)
    annotation_dir = temporary / "annotation"
    batches_dir = annotation_dir / "primary_batches"
    provenance_dir = temporary / "provenance"
    batches_dir.mkdir(parents=True)
    provenance_dir.mkdir(parents=True)

    canonical_output_rows = []
    for row in rows:
        clean = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        clean["sample_id"] = row["_sample_id"]
        canonical_output_rows.append(clean)
    canonical_path = temporary / "reviews_canonical.jsonl"
    canonical_count = _write_jsonl(canonical_path, canonical_output_rows)

    audit_path = temporary / "record_audit.jsonl"
    audit_count = _write_jsonl(audit_path, audit_rows)
    near_pairs_path = temporary / "near_duplicate_pairs.jsonl"
    near_pairs_count = _write_jsonl(near_pairs_path, near_pair_rows)
    near_clusters_path = temporary / "near_duplicate_clusters.jsonl"
    near_clusters_count = _write_jsonl(
        near_clusters_path,
        near_cluster_rows,
    )
    exclusions_path = temporary / "annotation_exclusions.jsonl"
    exclusion_rows = [
        {
            "sample_id": rows[index]["_sample_id"],
            "excluded_from": "primary_annotation_pool",
            "reason_flags": [
                flag
                for flag in flags_by_index[index]
                if flag
                in {
                    "near_duplicate_nonrepresentative",
                    "punctuation_variant_duplicate",
                    "repetition_inflated_quality",
                }
            ],
            "reversible": True,
        }
        for index in review_required_indices
    ]
    exclusions_count = _write_jsonl(exclusions_path, exclusion_rows)

    all_unlabeled_path = annotation_dir / "all_unlabeled.csv"
    all_unlabeled_count = _write_legacy_csv(all_unlabeled_path, rows)

    review_required_path = annotation_dir / "review_required.csv"
    review_required_count = _write_legacy_csv(
        review_required_path,
        (rows[index] for index in review_required_indices),
    )

    batch_assignments: dict[int, tuple[str, int]] = {}
    batch_artifacts: list[dict[str, Any]] = []
    for batch_number, start in enumerate(
        range(0, len(primary_indices), batch_size),
        1,
    ):
        selected = primary_indices[start : start + batch_size]
        filename = f"batch_{batch_number:03d}.csv"
        path = batches_dir / filename
        count = _write_legacy_csv(path, (rows[index] for index in selected))
        batch_artifacts.append(_artifact(path, temporary, count))
        for row_number, index in enumerate(selected, 1):
            batch_assignments[index] = (
                f"annotation/primary_batches/{filename}",
                row_number,
            )

    metadata_path = annotation_dir / "index.csv"
    metadata_columns = [
        "canonical_row",
        "sample_id",
        "annotation_queue",
        "batch_file",
        "batch_data_row",
        "review_id",
        "product_id",
        "rating",
        "category",
        "source_url",
        "crawl_id",
        "collected_at",
        "collection_transport",
        "review_text_sha256",
        "normalized_text_sha256",
        "audit_flags",
        "near_duplicate_cluster",
        "near_duplicate_representative_sample_id",
        "punctuation_duplicate_group",
        "punctuation_representative_sample_id",
        "source_relative_path",
        "source_line",
    ]
    manual_row_number = 0
    with metadata_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metadata_columns,
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            if index in batch_assignments:
                batch_file, batch_row = batch_assignments[index]
                queue = "primary"
            else:
                manual_row_number += 1
                batch_file = "annotation/review_required.csv"
                batch_row = manual_row_number
                queue = "review_required"
            representative = representative_for_index.get(index, index)
            writer.writerow(
                {
                    "canonical_row": index + 1,
                    "sample_id": row["_sample_id"],
                    "annotation_queue": queue,
                    "batch_file": batch_file,
                    "batch_data_row": batch_row,
                    "review_id": row.get("review_id"),
                    "product_id": row.get("product_id"),
                    "rating": row.get("rating"),
                    "category": row.get("category"),
                    "source_url": row.get("source_url"),
                    "crawl_id": row.get("crawl_id"),
                    "collected_at": row.get("collected_at"),
                    "collection_transport": row.get(
                        "collection_transport"
                    ),
                    "review_text_sha256": _sha256_bytes(
                        str(row.get("review_text") or "").encode("utf-8")
                    ),
                    "normalized_text_sha256": duplicate_key(
                        str(row.get("review_text") or "")
                    ),
                    "audit_flags": "|".join(flags_by_index[index]),
                    "near_duplicate_cluster": cluster_for_index.get(index, ""),
                    "near_duplicate_representative_sample_id": rows[
                        representative
                    ][
                        "_sample_id"
                    ],
                    "punctuation_duplicate_group": (
                        punctuation_group_for_index.get(index, "")
                    ),
                    "punctuation_representative_sample_id": rows[
                        punctuation_representative_for_index.get(index, index)
                    ]["_sample_id"],
                    "source_relative_path": row["_source_relative_path"],
                    "source_line": row["_source_line"],
                }
            )

    decisions_path = annotation_dir / "review_decisions.csv"
    with decisions_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "reason_flags",
                "decision",
                "notes",
            ]
        )
        for index in review_required_indices:
            writer.writerow(
                [
                    rows[index]["_sample_id"],
                    "|".join(flags_by_index[index]),
                    "",
                    "",
                ]
            )

    schema = {
        "review_column": "reviewContent",
        "aspect_columns": ASPECT_COLUMNS,
        "allowed_labels": [-1, 0, 1, 2, "1, -1"],
        "label_semantics": {
            "-1": "negative",
            "0": "neutral_mention",
            "1": "positive",
            "2": "not_mentioned",
            "1, -1": "positive_and_negative_for_same_aspect",
        },
        "unlabeled_representation": "blank_cell",
        "final_training_columns": LEGACY_COLUMNS,
        "metadata_join": ["batch_file", "batch_data_row"],
        "metadata_join_safety": (
            "Verify review_text_sha256 before importing labels; row order "
            "alone is not a stable identifier."
        ),
    }
    schema_path = annotation_dir / "schema.json"
    schema_path.write_text(
        json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    copied_plan = provenance_dir / "collection_plan.toml"
    copied_config = provenance_dir / "collector.toml"
    copied_guideline = provenance_dir / "annotation_guideline.txt"
    shutil.copy2(plan_path, copied_plan)
    shutil.copy2(collector_config, copied_config)
    shutil.copy2(annotation_guideline, copied_guideline)

    source_checksums_path = provenance_dir / "SOURCE_SHA256SUMS.txt"
    source_checksums_path.write_text(
        initial_source_inventory,
        encoding="utf-8",
        newline="\n",
    )
    code_checksums_path = provenance_dir / "CODE_SHA256SUMS.txt"
    code_file_count, code_inventory_sha256 = _write_code_inventory(
        project_root,
        code_checksums_path,
    )

    def sorted_counts(values: Iterable[str]) -> dict[str, int]:
        counter = Counter(values)
        return {key: counter[key] for key in sorted(counter)}

    distributions = {
        "transport": sorted_counts(
            str(row.get("collection_transport")) for row in rows
        ),
        "rating": sorted_counts(str(row.get("rating")) for row in rows),
        "category": sorted_counts(
            str(row.get("category") or "<blank>") for row in rows
        ),
        "crawl_manifest_status": manifest_validation["statuses"],
    }

    artifacts = [
        _artifact(canonical_path, temporary, canonical_count),
        _artifact(audit_path, temporary, audit_count),
        _artifact(near_pairs_path, temporary, near_pairs_count),
        _artifact(near_clusters_path, temporary, near_clusters_count),
        _artifact(exclusions_path, temporary, exclusions_count),
        _artifact(all_unlabeled_path, temporary, all_unlabeled_count),
        _artifact(
            review_required_path,
            temporary,
            review_required_count,
        ),
        _artifact(decisions_path, temporary, review_required_count),
        _artifact(metadata_path, temporary, len(rows)),
        _artifact(schema_path, temporary),
        *batch_artifacts,
        _artifact(source_checksums_path, temporary, source_file_count),
        _artifact(code_checksums_path, temporary, code_file_count),
        _artifact(copied_plan, temporary),
        _artifact(copied_config, temporary),
        _artifact(copied_guideline, temporary),
    ]

    threshold_fraction = Fraction(str(near_threshold)).limit_denominator(
        1_000_000
    )
    source_cutoff_at = max(
        str(row.get("collected_at") or "") for row in rows
    )
    release_id = (
        f"lazada-vi-{POLICY}-"
        f"{source_inventory_sha256[:12]}"
    )
    manifest = {
        "release_schema_version": 2,
        "release_name": output.name,
        "release_id": release_id,
        "source_cutoff_at": source_cutoff_at,
        "source": {
            "raw_root": str(raw_root.relative_to(project_root)).replace(
                "\\",
                "/",
            ),
            "raw_date_from": raw_date_from,
            "raw_date_through": raw_date_through,
            **source_counts,
            "source_files_hashed": source_file_count,
            "source_inventory_sha256": source_inventory_sha256,
            "code_files_hashed": code_file_count,
            "code_inventory_sha256": code_inventory_sha256,
            "manifest_validation": manifest_validation,
        },
        "selection": {
            "required_policy": POLICY,
            "exact_review_id_unique": True,
            "exact_normalized_text_unique": True,
            "quality_revalidated": True,
            "near_duplicate_action": (
                "flag_nonrepresentatives_for_manual_review; "
                "do_not_delete_from_canonical"
            ),
        },
        "counts": {
            "canonical_records": len(rows),
            "primary_annotation_records": len(primary_indices),
            "manual_review_records": len(review_required_indices),
            "record_audit_records": len(audit_rows),
            "annotation_batches": math.ceil(
                len(primary_indices) / batch_size
            ),
            "punctuation_duplicate_groups": punctuation_group_count,
            "punctuation_duplicate_nonrepresentatives": len(
                punctuation_duplicate_indices
            ),
            "near_duplicate_pairs": len(near_pairs),
            "near_duplicate_clusters": len(near_components),
            "near_duplicate_nonrepresentatives": len(
                near_duplicate_indices
            ),
            "internal_repeated_sentence_records": len(
                repeated_sentence_indices
            ),
            "repetition_inflated_quality_records": len(
                repetition_inflated_indices
            ),
            "category_missing_records": sum(
                not str(row.get("category") or "").strip()
                for row in rows
            ),
            "category_unmapped_records": sum(
                bool(str(row.get("category") or "").strip())
                and str(row.get("category") or "").strip()
                not in allowed_categories
                for row in rows
            ),
        },
        "near_duplicate_algorithm": {
            "tokens": "unicode_word_tokens_casefolded_nfkc",
            "feature": "exact_word_5gram_tuples",
            "candidate_generation": (
                "allpairs_global_df_prefix_index_with_length_filter"
            ),
            "verification": "exact_set_intersection_and_union",
            "threshold": near_threshold,
            "threshold_numerator": threshold_fraction.numerator,
            "threshold_denominator": threshold_fraction.denominator,
            "automatic_deletion": False,
        },
        "annotation": schema,
        "distributions": distributions,
        "artifacts": artifacts,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readme_path = temporary / "README.md"
    readme_path.write_text(
        "\n".join(
            (
                f"# {output.name}",
                "",
                f"Canonical records: {len(rows):,}",
                f"Primary annotation queue: {len(primary_indices):,}",
                f"Manual-review queue: {len(review_required_indices):,}",
                "",
                "`reviews_canonical.jsonl` is the immutable canonical V2",
                "corpus. No review is deleted because of heuristic",
                "near-duplicate detection.",
                "",
                "`annotation/primary_batches/*.csv` uses the exact legacy",
                "10-column schema. All nine label cells are intentionally",
                "blank. Allowed labels are -1, 0, 1, 2, and `1, -1`.",
                "",
                "`annotation/review_required.csv` contains suspicious",
                "duplicate/repetition records requiring human adjudication.",
                "Record keep/drop decisions in",
                "`annotation/review_decisions.csv`; do not put them in the",
                "nine ABSA label columns.",
                "Use `annotation/index.csv` to join each row to provenance.",
                "Verify `review_text_sha256` when importing completed labels.",
                "",
                "`record_audit.jsonl`, `near_duplicate_pairs.jsonl`, and",
                "`annotation_exclusions.jsonl` make every conservative flag",
                "and primary-pool exclusion reversible and auditable.",
                "",
                "Verify this release with `SHA256SUMS.txt`; verify the frozen",
                "raw source inventory with",
                "`provenance/SOURCE_SHA256SUMS.txt`.",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )

    final_files = sorted(
        path
        for path in temporary.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum_lines = [
        (
            f"{_sha256_file(path)}  "
            f"{str(path.relative_to(temporary)).replace(chr(92), '/')}"
        )
        for path in final_files
    ]
    checksum_path = temporary / "SHA256SUMS.txt"
    checksum_path.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    (
        final_source_file_count,
        final_source_inventory,
        final_source_inventory_sha256,
    ) = _source_inventory(
        raw_root,
        project_root,
        raw_date_from=raw_date_from,
        raw_date_through=raw_date_through,
    )
    if (
        final_source_file_count != source_file_count
        or final_source_inventory != initial_source_inventory
        or final_source_inventory_sha256 != source_inventory_sha256
    ):
        raise RuntimeError(
            "Raw source changed while the release was being built; "
            "the .building directory was retained for inspection."
        )

    temporary.replace(output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--raw-date-from",
        help="Optional inclusive YYYY-MM-DD raw directory cutoff.",
    )
    parser.add_argument(
        "--raw-date-through",
        help="Optional inclusive YYYY-MM-DD raw directory cutoff.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/releases/lazada_vi_reviews_v1_20260725"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/collection_plan.toml"),
    )
    parser.add_argument(
        "--collector-config",
        type=Path,
        default=Path("configs/collector.toml"),
    )
    parser.add_argument(
        "--annotation-guideline",
        type=Path,
        default=Path("docs/legacy-annotation-guideline.txt"),
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.85,
    )
    args = parser.parse_args()
    if not 0 < args.near_duplicate_threshold <= 1:
        parser.error("--near-duplicate-threshold must be in (0, 1]")
    manifest = build_release(
        raw_root=args.raw_root,
        raw_date_from=args.raw_date_from,
        raw_date_through=args.raw_date_through,
        output=args.output,
        plan_path=args.plan,
        collector_config=args.collector_config,
        annotation_guideline=args.annotation_guideline,
        batch_size=args.batch_size,
        near_threshold=args.near_duplicate_threshold,
    )
    print(
        json.dumps(
            {
                "release": str(args.output.resolve()),
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
