"""Reproducible, category-stratified automatic collection plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import List, Tuple

from .quality import QualityPolicy


@dataclass(frozen=True, slots=True)
class Segment:
    name: str
    queries: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AutomaticPlan:
    products_per_query: int
    reviews_per_product: int
    min_product_reviews: int
    scan_multiplier: int
    max_products_total: int
    seed: int
    include_sponsored: bool
    rating_filter: int
    quality: QualityPolicy
    segments: Tuple[Segment, ...]

    def validate(self) -> None:
        if self.products_per_query < 1:
            raise ValueError("products_per_query must be positive")
        if self.reviews_per_product < 1:
            raise ValueError("reviews_per_product must be positive")
        if self.min_product_reviews < 0:
            raise ValueError("min_product_reviews cannot be negative")
        if self.scan_multiplier < 1:
            raise ValueError("scan_multiplier must be positive")
        if self.max_products_total < 1:
            raise ValueError("max_products_total must be positive")
        if self.rating_filter not in range(0, 6):
            raise ValueError("rating_filter must be in [0, 5]")
        if not self.segments:
            raise ValueError("At least one segment is required")
        for segment in self.segments:
            if not segment.name.strip() or not segment.queries:
                raise ValueError("Each segment needs a name and query list")
            if any(not query.strip() for query in segment.queries):
                raise ValueError("Segment queries cannot be empty")
        self.quality.validate()

    @property
    def query_count(self) -> int:
        return sum(len(segment.queries) for segment in self.segments)

    @property
    def maximum_accepted_reviews(self) -> int:
        return self.max_products_total * self.reviews_per_product


def load_plan(path: Path) -> AutomaticPlan:
    if not path.exists():
        raise FileNotFoundError(f"Automatic collection plan not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    automatic = raw.get("automatic") or {}
    quality = QualityPolicy(**(raw.get("quality") or {}))
    segments: List[Segment] = []
    for item in raw.get("segments") or []:
        segments.append(
            Segment(
                name=str(item.get("name") or ""),
                queries=tuple(str(query) for query in item.get("queries") or []),
            )
        )
    plan = AutomaticPlan(
        products_per_query=int(automatic.get("products_per_query", 2)),
        reviews_per_product=int(automatic.get("reviews_per_product", 30)),
        min_product_reviews=int(automatic.get("min_product_reviews", 50)),
        scan_multiplier=int(automatic.get("scan_multiplier", 8)),
        max_products_total=int(automatic.get("max_products_total", 24)),
        seed=int(automatic.get("seed", 20260723)),
        include_sponsored=bool(automatic.get("include_sponsored", False)),
        rating_filter=int(automatic.get("rating_filter", 0)),
        quality=quality,
        segments=tuple(segments),
    )
    plan.validate()
    return plan

