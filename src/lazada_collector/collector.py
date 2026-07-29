"""High-level product search and review collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Iterator, List, Optional

from . import __version__
from .errors import BlockedError, CollectorError, ResponseSchemaError
from .normalization import duplicate_key, normalize_text
from .schema import ProductRecord, ReviewRecord
from .transport import (
    RequestsTransport,
    SeleniumTransport,
    extract_item_id,
    parse_review_payload,
)


@dataclass(slots=True)
class CollectorConfig:
    transport: str = "auto"
    page_size: int = 20
    max_pages: int = 100
    min_delay_seconds: float = 2.0
    max_delay_seconds: float = 5.0
    headless: bool = True
    profile_dir: Optional[Path] = None
    cookie_file: Optional[Path] = None
    review_browser_fallback: bool = False

    def validate(self) -> None:
        if self.transport not in {"auto", "requests", "selenium"}:
            raise ValueError("transport must be auto, requests, or selenium")
        if not 1 <= self.page_size <= 50:
            raise ValueError("page_size must be in [1, 50]")
        if self.max_pages < 1:
            raise ValueError("max_pages must be positive")
        if self.min_delay_seconds < 0:
            raise ValueError("min_delay_seconds cannot be negative")
        if self.max_delay_seconds < self.min_delay_seconds:
            raise ValueError("max_delay_seconds must be >= min_delay_seconds")


class LazadaCollector:
    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.config.validate()
        self.requests = RequestsTransport(cookie_file=self.config.cookie_file)
        self.browser: Optional[SeleniumTransport] = None
        self._requests_review_blocked = False
        self._requests_search_blocked = False

    def search(
        self,
        query: str,
        limit: int = 20,
        page: int = 1,
    ) -> List[ProductRecord]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if page < 1:
            raise ValueError("page must be positive")
        cleaned_query = query.strip()
        if self.config.transport == "selenium":
            products, _ = self._browser_transport().search(cleaned_query, page=page)
            return products[:limit]
        if self.config.transport == "requests" or not self._requests_search_blocked:
            try:
                products, _ = self.requests.search(cleaned_query, page=page)
                return products[:limit]
            except (BlockedError, ResponseSchemaError):
                if self.config.transport == "requests":
                    raise
                self._requests_search_blocked = True
        products, _ = self._browser_transport().search(cleaned_query, page=page)
        return products[:limit]

    def reset_transports(self) -> None:
        """Start fresh sessions after a bounded rate-limit cooldown."""
        self.requests.reset()
        self._requests_review_blocked = False
        self._requests_search_blocked = False
        if self.browser is not None:
            self.browser.close()
            self.browser = None

    def _browser_transport(self) -> SeleniumTransport:
        if self.browser is None:
            self.browser = SeleniumTransport(
                headless=self.config.headless,
                profile_dir=self.config.profile_dir,
                cookie_file=self.config.cookie_file,
            )
        return self.browser

    def _fetch_review_page(
        self,
        item_id: str,
        page_number: int,
        rating_filter: int,
    ):
        if self.config.transport == "selenium":
            transport = self._browser_transport()
            payload, digest = transport.fetch_review_page(
                item_id,
                page_number,
                self.config.page_size,
                rating_filter,
            )
            return payload, digest, transport.name

        if self.config.transport == "requests" or not self._requests_review_blocked:
            try:
                payload, digest = self.requests.fetch_review_page(
                    item_id,
                    page_number,
                    self.config.page_size,
                    rating_filter,
                )
                return payload, digest, self.requests.name
            except BlockedError:
                # A browser is not a CAPTCHA bypass. Preserve the BLOCKED
                # signal so a scale job can stop/cool down immediately.
                self._requests_review_blocked = True
                if not self.config.review_browser_fallback:
                    raise
            except ResponseSchemaError:
                if self.config.transport == "requests":
                    raise
                time.sleep(self.config.max_delay_seconds)
                self.requests.reset()
                try:
                    payload, digest = self.requests.fetch_review_page(
                        item_id,
                        page_number,
                        self.config.page_size,
                        rating_filter,
                    )
                    return payload, digest, self.requests.name
                except (BlockedError, ResponseSchemaError):
                    self._requests_review_blocked = True
                    if not self.config.review_browser_fallback:
                        raise

        transport = self._browser_transport()
        payload, digest = transport.fetch_review_page(
            item_id,
            page_number,
            self.config.page_size,
            rating_filter,
        )
        return payload, digest, transport.name

    def iter_reviews(
        self,
        product: ProductRecord,
        crawl_id: str,
        max_reviews: int,
        rating_filter: int = 0,
        max_pages: Optional[int] = None,
    ) -> Iterator[ReviewRecord]:
        if max_reviews < 1:
            raise ValueError("max_reviews must be positive")
        item_id = product.product_id or extract_item_id(product.url)
        if not item_id:
            raise ValueError(f"Cannot determine product ID from {product.url!r}")

        # A challenge on one product/page must not force every later product to
        # use the browser transport without first trying the lightweight API.
        self._requests_review_blocked = False
        seen_review_ids = set()
        seen_texts = set()
        seen_response_digests = set()
        emitted = 0
        sampling_frame = "natural" if rating_filter == 0 else "sentiment_enriched"
        page_limit = self.config.max_pages
        if max_pages is not None:
            if max_pages < 1:
                raise ValueError("max_pages must be positive when provided")
            page_limit = min(page_limit, max_pages)

        for page_number in range(1, page_limit + 1):
            payload, response_sha256, transport_name = self._fetch_review_page(
                item_id,
                page_number,
                rating_filter,
            )
            # Some challenged/changed endpoint variants ignore ``pageNo`` and
            # return page 1 repeatedly. Stop as soon as the exact payload is
            # seen again instead of spending the remaining page budget on
            # duplicate requests.
            if response_sha256 in seen_response_digests:
                break
            seen_response_digests.add(response_sha256)
            items, page_count = parse_review_payload(payload)
            if not items:
                break

            for item in items:
                text = normalize_text(str(item.get("reviewContent") or ""))
                if not text:
                    continue
                review_id = str(item.get("reviewRateId") or "").strip()
                text_key = duplicate_key(text)
                if review_id and review_id in seen_review_ids:
                    continue
                if text_key in seen_texts:
                    continue
                if review_id:
                    seen_review_ids.add(review_id)
                seen_texts.add(text_key)
                if not review_id:
                    review_id = f"text-{text_key[:24]}"

                rating = item.get("rating")
                try:
                    rating = int(rating) if rating is not None else None
                except (TypeError, ValueError):
                    rating = None
                sku_info = item.get("skuInfo") or ""
                if not isinstance(sku_info, str):
                    sku_info = json.dumps(sku_info, ensure_ascii=False, sort_keys=True)
                images = item.get("images")
                verified = item.get("isPurchased")
                if verified is None:
                    verified = item.get("verifiedPurchase")
                if verified is not None:
                    verified = bool(verified)

                yield ReviewRecord(
                    crawl_id=crawl_id,
                    collected_at=datetime.now(timezone.utc).isoformat(),
                    collector_version=__version__,
                    collection_transport=transport_name,
                    sampling_frame=sampling_frame,
                    query=product.query,
                    product_id=item_id,
                    seller_id=product.seller_id,
                    source_url=product.url,
                    review_id=review_id,
                    review_text=text,
                    rating=rating,
                    review_time=str(
                        item.get("reviewTime")
                        or item.get("zonedReviewTime")
                        or item.get("boughtDate")
                        or ""
                    ),
                    sku_info=sku_info,
                    has_images=bool(images),
                    verified_purchase=verified,
                    page_number=page_number,
                    response_sha256=response_sha256,
                    category=product.category,
                )
                emitted += 1
                if emitted >= max_reviews:
                    return

            if page_count is not None and page_number >= page_count:
                break
            if page_number >= page_limit:
                break
            time.sleep(
                random.uniform(
                    self.config.min_delay_seconds,
                    self.config.max_delay_seconds,
                )
            )

    def close(self) -> None:
        self.requests.close()
        if self.browser is not None:
            self.browser.close()
            self.browser = None

    def __enter__(self) -> "LazadaCollector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
