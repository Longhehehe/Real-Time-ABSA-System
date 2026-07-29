"""Incremental JSONL output and an auditable run manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import uuid

from .normalization import duplicate_key
from .quality import QualityResult
from .schema import ProductRecord, ReviewRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CrawlRun:
    def __init__(
        self,
        output_root: Path,
        command: str,
        parameters: Dict[str, Any],
        existing_review_ids: Optional[Iterable[str]] = None,
        existing_text_keys: Optional[Iterable[str]] = None,
    ):
        stamp = datetime.now(timezone.utc)
        self.crawl_id = f"{stamp:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.run_dir = output_root / f"{stamp:%Y-%m-%d}" / self.crawl_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.reviews_path = self.run_dir / "reviews.jsonl"
        self.products_path = self.run_dir / "products.jsonl"
        self.rejections_path = self.run_dir / "rejections.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self._reviews = self.reviews_path.open("a", encoding="utf-8")
        self._products = self.products_path.open("a", encoding="utf-8")
        self._rejections = self.rejections_path.open("a", encoding="utf-8")
        self._seen_review_ids = set(existing_review_ids or ())
        self._seen_texts = set(existing_text_keys or ())
        self._closed = False
        self.manifest: Dict[str, Any] = {
            "schema_version": 2,
            "crawl_id": self.crawl_id,
            "command": command,
            "started_at": _utc_now(),
            "finished_at": None,
            "status": "running",
            "parameters": parameters,
            "cross_run_dedup": {
                "existing_review_ids": len(self._seen_review_ids),
                "existing_text_keys": len(self._seen_texts),
            },
            "counts": {
                "products": 0,
                "reviews_written": 0,
                "reviews_deduplicated": 0,
                "review_candidates": 0,
                "reviews_rejected_quality": 0,
                "products_completed": 0,
                "products_shortfall": 0,
                "products_skipped": 0,
                "errors": 0,
            },
            "errors": [],
            "product_shortfalls": [],
            "skipped_products": [],
            "files": {
                "products": self.products_path.name,
                "reviews": self.reviews_path.name,
                "rejections": self.rejections_path.name,
            },
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def checkpoint(self) -> None:
        """Flush append-only outputs and atomically persist current progress."""
        if self._closed:
            return
        self._reviews.flush()
        self._products.flush()
        self._rejections.flush()
        self.manifest["checkpoint_at"] = _utc_now()
        self._write_manifest()

    def add_product(self, product: ProductRecord) -> None:
        self._products.write(
            json.dumps(product.to_dict(), ensure_ascii=False) + "\n"
        )
        self._products.flush()
        self.manifest["counts"]["products"] += 1

    def add_review(self, review: ReviewRecord) -> bool:
        text_key = duplicate_key(review.review_text)
        if review.review_id in self._seen_review_ids or text_key in self._seen_texts:
            self.manifest["counts"]["reviews_deduplicated"] += 1
            return False
        self._seen_review_ids.add(review.review_id)
        self._seen_texts.add(text_key)
        self._reviews.write(
            json.dumps(review.to_dict(), ensure_ascii=False) + "\n"
        )
        self._reviews.flush()
        self.manifest["counts"]["reviews_written"] += 1
        return True

    def record_candidate(self) -> None:
        self.manifest["counts"]["review_candidates"] += 1

    def add_rejection(
        self,
        review: ReviewRecord,
        result: QualityResult,
    ) -> None:
        row = {
            "schema_version": 1,
            "crawl_id": self.crawl_id,
            "review_id": review.review_id,
            "product_id": review.product_id,
            "text_sha256": duplicate_key(review.review_text),
            "char_count": result.char_count,
            "word_count": result.word_count,
            "unique_word_ratio": result.unique_word_ratio,
            "vietnamese_signal_count": result.vietnamese_signal_count,
            "foreign_script_ratio": result.foreign_script_ratio,
            "quality_score": result.score,
            "reasons": list(result.reasons),
        }
        self._rejections.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rejections.flush()
        self.manifest["counts"]["reviews_rejected_quality"] += 1

    def record_product_result(
        self,
        product: ProductRecord,
        accepted_reviews: int,
        target_reviews: int,
    ) -> None:
        self.manifest["counts"]["products_completed"] += 1
        if accepted_reviews < target_reviews:
            self.manifest["counts"]["products_shortfall"] += 1
            self.manifest["product_shortfalls"].append(
                {
                    "product_id": product.product_id,
                    "query": product.query,
                    "category": product.category,
                    "accepted_reviews": accepted_reviews,
                    "target_reviews": target_reviews,
                }
            )

    def record_skipped_product(
        self,
        product: ProductRecord,
        reason: str,
    ) -> None:
        self.manifest["counts"]["products_skipped"] += 1
        self.manifest["skipped_products"].append(
            {
                "product_id": product.product_id,
                "query": product.query,
                "category": product.category,
                "reason": reason,
            }
        )

    def add_error(
        self,
        scope: str,
        message: str,
        code: str = "ERROR",
    ) -> None:
        self.manifest["counts"]["errors"] += 1
        self.manifest["errors"].append(
            {
                "at": _utc_now(),
                "scope": scope,
                "code": code,
                "message": message,
            }
        )
        self._write_manifest()

    def close(self, status: Optional[str] = None) -> None:
        if self._closed:
            return
        self._reviews.close()
        self._products.close()
        self._rejections.close()
        if status is None:
            status = "completed_with_errors" if self.manifest["errors"] else "completed"
        self.manifest["status"] = status
        self.manifest["finished_at"] = _utc_now()
        self._write_manifest()
        self._closed = True

    def __enter__(self) -> "CrawlRun":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(status="failed" if exc is not None else None)
