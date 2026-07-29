"""Versioned records written by the collector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


SCHEMA_VERSION = "1.2"


@dataclass(frozen=True, slots=True)
class ProductRecord:
    product_id: str
    name: str
    url: str
    query: str = ""
    seller_id: str = ""
    seller_name: str = ""
    category: str = ""
    price: str = ""
    rating: Optional[float] = None
    review_count: Optional[int] = None
    location: str = ""
    is_sponsored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    crawl_id: str
    collected_at: str
    collector_version: str
    collection_transport: str
    sampling_frame: str
    query: str
    product_id: str
    seller_id: str
    source_url: str
    review_id: str
    review_text: str
    rating: Optional[int]
    review_time: str
    sku_info: str
    has_images: bool
    verified_purchase: Optional[bool]
    page_number: int
    response_sha256: str
    category: str = ""
    selection_policy: str = "unfiltered"
    quality_score: Optional[float] = None
    char_count: Optional[int] = None
    word_count: Optional[int] = None
    unique_word_ratio: Optional[float] = None
    vietnamese_signal_count: Optional[int] = None
    foreign_script_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}
