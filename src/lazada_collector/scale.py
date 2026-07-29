"""Configuration and deterministic discovery order for large crawl jobs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import random
from typing import Dict, Iterator, Tuple

from .plan import AutomaticPlan, Segment


@dataclass(frozen=True, slots=True)
class ScaleSettings:
    """Runtime limits for a resumable 30k-50k review collection."""

    target_reviews: int = 30_000
    reviews_per_product: int = 50
    max_search_pages: int = 40
    products_per_page: int = 15
    cooldown_seconds: float = 300.0
    max_cooldown_seconds: float = 900.0
    max_cooldowns: int = 3
    failure_threshold: int = 3
    max_products: int | None = None

    def validate(self) -> None:
        if self.target_reviews < 1:
            raise ValueError("target_reviews must be positive")
        if self.reviews_per_product < 1:
            raise ValueError("reviews_per_product must be positive")
        if self.max_search_pages < 1:
            raise ValueError("max_search_pages must be positive")
        if self.products_per_page < 1:
            raise ValueError("products_per_page must be positive")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        if self.max_cooldown_seconds < self.cooldown_seconds:
            raise ValueError(
                "max_cooldown_seconds must be >= cooldown_seconds"
            )
        if self.max_cooldowns < 0:
            raise ValueError("max_cooldowns cannot be negative")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if self.max_products is not None and self.max_products < 1:
            raise ValueError("max_products must be positive when provided")

    def cooldown_delay(self, cooldown_number: int) -> float:
        if cooldown_number < 1:
            raise ValueError("cooldown_number must be positive")
        exponential = self.cooldown_seconds * (2 ** min(cooldown_number - 1, 4))
        return min(exponential, self.max_cooldown_seconds)


def iter_query_pages(
    plan: AutomaticPlan,
    settings: ScaleSettings,
) -> Iterator[Tuple[Segment, str, int]]:
    """Yield a balanced, reproducible page-first search schedule."""
    queries = [
        (segment, query)
        for segment in plan.segments
        for query in segment.queries
    ]
    for page in range(1, settings.max_search_pages + 1):
        page_queries = list(queries)
        random.Random(plan.seed + page).shuffle(page_queries)
        for segment, query in page_queries:
            yield segment, query, page


def build_scale_summary(
    plan: AutomaticPlan,
    settings: ScaleSettings,
    existing_reviews: int,
) -> Dict[str, int | float | bool]:
    remaining = max(0, settings.target_reviews - existing_reviews)
    estimated_products = ceil(remaining / settings.reviews_per_product)
    discovery_capacity = (
        plan.query_count
        * settings.max_search_pages
        * settings.products_per_page
    )
    if settings.max_products is not None:
        discovery_capacity = min(discovery_capacity, settings.max_products)
    accepted_capacity = discovery_capacity * settings.reviews_per_product
    return {
        "target_reviews_total": settings.target_reviews,
        "reviews_already_collected": existing_reviews,
        "reviews_remaining": remaining,
        "estimated_products_needed_at_full_quota": estimated_products,
        "maximum_products_this_run": discovery_capacity,
        "maximum_accepted_review_capacity": accepted_capacity,
        "configured_search_requests": (
            plan.query_count * settings.max_search_pages
        ),
        "candidate_scan_limit_per_product": (
            settings.reviews_per_product * plan.scan_multiplier
        ),
        "configured_capacity_is_sufficient": accepted_capacity >= remaining,
    }
