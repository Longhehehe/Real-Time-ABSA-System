"""Read prior crawl outputs for cached discovery and cross-run deduplication."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Dict, List, Set

from .normalization import duplicate_key, looks_like_mojibake
from .quality import SELECTION_POLICY
from .schema import ProductRecord


@dataclass(slots=True)
class CollectionHistory:
    cached_products: List[ProductRecord]
    attempted_product_ids: Set[str]
    accepted_review_ids: Set[str]
    accepted_text_keys: Set[str]
    files_scanned: int = 0
    excluded_review_records: int = 0


def _read_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _product_from_row(row: Dict) -> ProductRecord | None:
    allowed = {field.name for field in fields(ProductRecord)}
    values = {key: value for key, value in row.items() if key in allowed}
    if not values.get("product_id") or not values.get("url"):
        return None
    try:
        return ProductRecord(**values)
    except (TypeError, ValueError):
        return None


def load_collection_history(output_root: Path) -> CollectionHistory:
    if not output_root.exists():
        return CollectionHistory([], set(), set(), set(), 0, 0)

    product_files = sorted(
        output_root.rglob("products.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    review_files = sorted(output_root.rglob("reviews.jsonl"))
    rejection_files = sorted(output_root.rglob("rejections.jsonl"))
    cached_by_id: Dict[str, ProductRecord] = {}
    attempted_product_ids: Set[str] = set()
    accepted_review_ids: Set[str] = set()
    accepted_text_keys: Set[str] = set()
    excluded_review_records = 0

    for path in product_files:
        for row in _read_jsonl(path):
            product = _product_from_row(row)
            if product and product.product_id not in cached_by_id:
                cached_by_id[product.product_id] = product

    for path in review_files:
        for row in _read_jsonl(path):
            product_id = str(row.get("product_id") or "")
            review_id = str(row.get("review_id") or "")
            text = str(row.get("review_text") or "")
            if product_id:
                attempted_product_ids.add(product_id)
            is_current_quality_record = (
                row.get("selection_policy") == SELECTION_POLICY
            )
            if (
                not is_current_quality_record
                or not text
                or looks_like_mojibake(text)
            ):
                excluded_review_records += 1
                continue
            if review_id:
                accepted_review_ids.add(review_id)
            if text:
                accepted_text_keys.add(duplicate_key(text))

    for path in rejection_files:
        for row in _read_jsonl(path):
            product_id = str(row.get("product_id") or "")
            if product_id:
                attempted_product_ids.add(product_id)

    return CollectionHistory(
        cached_products=list(cached_by_id.values()),
        attempted_product_ids=attempted_product_ids,
        accepted_review_ids=accepted_review_ids,
        accepted_text_keys=accepted_text_keys,
        files_scanned=len(product_files) + len(review_files) + len(rejection_files),
        excluded_review_records=excluded_review_records,
    )
