"""Self-hosted Selenium collection from Lazada's rendered product pages.

This transport reads the same review cards a person can see in Chrome.  It
does not call the rate-limited public review JSON endpoint and it deliberately
does not contain CAPTCHA bypass or browser-fingerprint spoofing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib.parse import urlencode

from . import __version__
from .errors import (
    BlockedError,
    ProductUnavailableError,
    RateLimitedError,
    ResponseSchemaError,
    TransportError,
)
from .normalization import normalize_text
from .schema import ProductRecord, ReviewRecord
from .transport import CATALOG_URL, SeleniumTransport, extract_item_id


DEFAULT_SEED_ROOT = Path("legacy/data/crawl_url_lists")
_URL = re.compile(r"https?://[^\s\"']+", flags=re.IGNORECASE)
_PAGE_COUNT = re.compile(r"(?:out\s+of|/)\s*(\d+)", flags=re.IGNORECASE)
_BLOCK_MARKERS = (
    "captcha",
    "security check",
    "unusual traffic",
    "xác minh",
    "verify your identity",
)


@dataclass(frozen=True, slots=True)
class DomPage:
    page_number: int
    total_pages: Optional[int]
    reviews: tuple[ReviewRecord, ...]
    is_last: bool


@dataclass(frozen=True, slots=True)
class DomSettings:
    max_pages_per_product: int = 100
    page_delay_min: float = 2.0
    page_delay_max: float = 4.0
    load_timeout_seconds: float = 35.0

    def validate(self) -> None:
        if self.max_pages_per_product < 1:
            raise ValueError("max_pages_per_product must be positive")
        if self.page_delay_min < 0:
            raise ValueError("page_delay_min cannot be negative")
        if self.page_delay_max < self.page_delay_min:
            raise ValueError("page_delay_max must be >= page_delay_min")
        if self.load_timeout_seconds <= 0:
            raise ValueError("load_timeout_seconds must be positive")


def stable_dom_review_id(
    product_id: str,
    review_text: str,
    review_time: str = "",
    sku_info: str = "",
) -> str:
    """Create a repeatable pseudonymous ID without retaining buyer identity."""
    payload = "\0".join(
        (
            product_id.strip(),
            normalize_text(review_text).casefold(),
            normalize_text(review_time).casefold(),
            normalize_text(sku_info).casefold(),
        )
    )
    return f"dom-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"


def load_seed_products(seed_root: Path = DEFAULT_SEED_ROOT) -> List[ProductRecord]:
    """Load and deduplicate the repository's historical product URL seeds."""
    if not seed_root.exists():
        return []
    products: Dict[str, ProductRecord] = {}
    for path in sorted(seed_root.glob("*.txt")):
        category = ""
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            match = _URL.search(line)
            if not match:
                if line:
                    category = line.strip("\"' ")
                continue
            url = match.group(0).rstrip(".,;)]}")
            product_id = extract_item_id(url)
            if not product_id or product_id in products:
                continue
            products[product_id] = ProductRecord(
                product_id=product_id,
                name=f"Lazada seed {product_id}",
                url=url,
                query=f"seed:{path.stem}",
                category=category,
            )
    return list(products.values())


def load_completed_dom_products(output_root: Path) -> set[str]:
    """Read products whose configured DOM page sample was completed."""
    completed: set[str] = set()
    if not output_root.exists():
        return completed
    for path in output_root.rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("command") != "crawl-dom-scale":
            continue
        dom = manifest.get("dom")
        if not isinstance(dom, dict):
            continue
        values = dom.get("completed_product_ids")
        if isinstance(values, list):
            completed.update(str(value) for value in values if value)
    return completed


def load_dom_page_progress(output_root: Path) -> Dict[str, int]:
    """Return the highest checkpointed DOM review page for each product."""
    progress: Dict[str, int] = {}
    if not output_root.exists():
        return progress

    for path in output_root.rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("command") != "crawl-dom-scale":
            continue
        dom = manifest.get("dom")
        values = dom.get("page_progress") if isinstance(dom, dict) else None
        if not isinstance(values, dict):
            continue
        for product_id, page_number in values.items():
            try:
                page = int(page_number)
            except (TypeError, ValueError):
                continue
            if product_id and page > progress.get(str(product_id), 0):
                progress[str(product_id)] = page

    # Older DOM manifests did not have page_progress. Accepted-review records
    # still provide a safe lower-bound checkpoint for those runs.
    for path in output_root.rglob("reviews.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("collection_transport") != "selenium_dom":
                continue
            product_id = str(row.get("product_id") or "")
            try:
                page = int(row.get("page_number") or 0)
            except (TypeError, ValueError):
                continue
            if product_id and page > progress.get(product_id, 0):
                progress[product_id] = page
    return progress


class SeleniumDomCollector:
    """Collect visible review cards and search results in one Chrome session."""

    def __init__(
        self,
        *,
        headless: bool = True,
        profile_dir: Optional[Path] = None,
        cookie_file: Optional[Path] = None,
        settings: Optional[DomSettings] = None,
        seed: int = 20260723,
    ):
        self.settings = settings or DomSettings()
        self.settings.validate()
        self.transport = SeleniumTransport(
            headless=headless,
            profile_dir=profile_dir,
            timeout_seconds=self.settings.load_timeout_seconds,
            cookie_file=cookie_file,
        )
        self.driver = self.transport.driver
        self._by = self.transport._by
        self._random = random.Random(seed)

    def _body_text(self) -> str:
        try:
            return (
                self.driver.find_element(self._by.TAG_NAME, "body").text or ""
            )
        except Exception:
            return ""

    def _raise_if_challenged(self) -> None:
        lowered = self._body_text()[:8000].casefold()
        if any(marker in lowered for marker in _BLOCK_MARKERS):
            raise BlockedError(
                "Lazada displayed a browser challenge; no bypass was attempted"
            )

    def _wait_for_review_module(self) -> Any:
        deadline = time.monotonic() + self.settings.load_timeout_seconds
        module = None
        scroll_y = 0
        suppressed_since: Optional[float] = None
        while time.monotonic() < deadline:
            self._raise_if_challenged()
            body_text = self._body_text()[:4000].casefold()
            if (
                "this product is no longer available" in body_text
                or "sản phẩm này không còn tồn tại" in body_text
            ):
                raise ProductUnavailableError(
                    "Lazada product is no longer available"
                )
            found = self.driver.find_elements(
                self._by.ID,
                "module_product_review",
            )
            if found:
                module = found[0]
                items = module.find_elements(self._by.CSS_SELECTOR, ".item")
                pagination = module.find_elements(
                    self._by.CSS_SELECTOR,
                    ".review-pagination",
                )
                if items or pagination:
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'})",
                        module,
                    )
                    time.sleep(0.8)
                    return module
                empty = module.find_elements(
                    self._by.CSS_SELECTOR,
                    ".mod-empty .empty-text",
                )
                title = module.find_elements(
                    self._by.CSS_SELECTOR,
                    ".mod-title .title-text",
                )
                empty_text = empty[0].text.casefold() if empty else ""
                title_text = title[0].text if title else ""
                count_match = re.search(r"\(([\d.,]+)\)", title_text)
                count = 0
                if count_match:
                    try:
                        count = int(
                            count_match.group(1).replace(".", "").replace(",", "")
                        )
                    except ValueError:
                        count = 0
                claims_no_reviews = (
                    "this product has no reviews" in empty_text
                    or "sản phẩm này chưa có đánh giá" in empty_text
                )
                if claims_no_reviews and count <= 0:
                    return module
                if claims_no_reviews and count > 0:
                    if suppressed_since is None:
                        suppressed_since = time.monotonic()
                    elif time.monotonic() - suppressed_since >= 8.0:
                        raise RateLimitedError(
                            "Review cards were suppressed despite a positive "
                            f"review count ({count}); likely temporary rate limit"
                        )
                else:
                    suppressed_since = None
            scroll_y += 650
            self.driver.execute_script("window.scrollTo(0, arguments[0])", scroll_y)
            time.sleep(0.3)
        if module is not None:
            return module
        raise ResponseSchemaError(
            "Rendered product page has no review module (removed product or DOM change)"
        )

    @staticmethod
    def _page_digest(rows: Sequence[Dict[str, Any]]) -> str:
        canonical = json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _extract_rows(self) -> List[Dict[str, Any]]:
        rows = self.driver.execute_script(
            """
            return Array.from(
              document.querySelectorAll('#module_product_review .item')
            ).map(root => {
              const reviewNode = root.querySelector(
                '.item-content-main-content-reviews, [class$="-reviews"]'
              );
              const skuNode = root.querySelector(
                '.item-content-main-content-skuInfo, [class*="skuInfo"]'
              );
              const starNodes = Array.from(
                root.querySelectorAll('.review-star .i-rate-star')
              );
              let rating = 0;
              for (const star of starNodes) {
                const colored = Array.from(star.querySelectorAll('path[mask]'))
                  .find(path => /255\\s*,\\s*200\\s*,\\s*60/.test(
                    path.getAttribute('style') || ''
                  ));
                if (!colored) continue;
                const mask = colored.getAttribute('mask') || '';
                const match = mask.match(/half_(\\d+)%/);
                if (match) rating += Number(match[1]) / 100;
              }
              return {
                review_text: (reviewNode?.innerText || '').trim(),
                review_time: (
                  root.querySelector('.time')?.innerText || ''
                ).trim(),
                sku_info: (skuNode?.innerText || '').trim(),
                has_images: root.querySelectorAll(
                  '.item-content-main-imgs .img-wrapper'
                ).length > 0,
                rating: rating > 0 ? Math.round(rating) : null
              };
            });
            """
        )
        return rows if isinstance(rows, list) else []

    def _page_state(self) -> tuple[Optional[int], bool]:
        result = self.driver.execute_script(
            """
            const root = document.querySelector(
              '#module_product_review .review-pagination'
            );
            if (!root) return {totalText: '', disabled: true};
            const next = root.querySelector('.iweb-pagination-next');
            return {
              totalText: (
                root.querySelector('.iweb-pagination-total-text')?.innerText || ''
              ).trim(),
              disabled: !next ||
                next.classList.contains('iweb-pagination-disabled') ||
                next.getAttribute('aria-disabled') === 'true'
            };
            """
        )
        text = str((result or {}).get("totalText") or "")
        match = _PAGE_COUNT.search(text)
        total_pages = int(match.group(1)) if match else None
        return total_pages, bool((result or {}).get("disabled", False))

    def _active_page(self) -> int:
        value = self.driver.execute_script(
            """
            return (
              document.querySelector(
                '#module_product_review .iweb-pagination-item-active'
              )?.innerText || '1'
            ).trim();
            """
        )
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1

    def _click_and_wait(
        self,
        clickable: Any,
        *,
        old_digest: str,
        old_page: int,
        expected_page: Optional[int] = None,
    ) -> int:
        self.driver.execute_script("arguments[0].click()", clickable)
        deadline = time.monotonic() + self.settings.load_timeout_seconds
        active_changed_at: Optional[float] = None
        while time.monotonic() < deadline:
            self._raise_if_challenged()
            rows = self._extract_rows()
            new_digest = self._page_digest(rows)
            active = self._active_page()
            page_changed = (
                active == expected_page
                if expected_page is not None
                else active > old_page
            )
            if page_changed:
                if active_changed_at is None:
                    active_changed_at = time.monotonic()
                # Usually the digest changes immediately. The grace period
                # also handles two adjacent pages containing identical cards.
                if (
                    new_digest != old_digest
                    or time.monotonic() - active_changed_at >= 2.0
                ):
                    module = self.driver.find_element(
                        self._by.ID,
                        "module_product_review",
                    )
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'})",
                        module,
                    )
                    return active
            time.sleep(0.25)
        target = expected_page if expected_page is not None else f">{old_page}"
        raise TransportError(f"Timed out waiting for review page {target}")

    def _click_next(self, old_digest: str, old_page: int) -> None:
        next_items = self.driver.find_elements(
            self._by.CSS_SELECTOR,
            "#module_product_review .review-pagination .iweb-pagination-next",
        )
        if not next_items:
            raise ResponseSchemaError("Review pagination has no next-page control")
        next_item = next_items[0]
        target = next_item.find_elements(self._by.CSS_SELECTOR, "button, a")
        clickable = target[0] if target else next_item
        self._click_and_wait(
            clickable,
            old_digest=old_digest,
            old_page=old_page,
            expected_page=old_page + 1,
        )

    def _go_to_page(self, target_page: int) -> None:
        if target_page <= 1:
            return
        total_pages, _ = self._page_state()
        if total_pages is not None and target_page > total_pages:
            raise ResponseSchemaError(
                f"Resume page {target_page} exceeds current total {total_pages}"
            )

        current_page = self._active_page()
        steps = 0
        while current_page < target_page:
            steps += 1
            if steps > target_page:
                raise TransportError(
                    f"Could not navigate to resume page {target_page}"
                )
            rows = self._extract_rows()
            old_digest = self._page_digest(rows)
            direct = self.driver.find_elements(
                self._by.CSS_SELECTOR,
                (
                    "#module_product_review "
                    f".iweb-pagination-item-{target_page}"
                ),
            )
            expected_page: Optional[int] = None
            if direct:
                control = direct[0]
                expected_page = target_page
            else:
                jumps = self.driver.find_elements(
                    self._by.CSS_SELECTOR,
                    (
                        "#module_product_review "
                        ".iweb-pagination-jump-next"
                    ),
                )
                if jumps:
                    control = jumps[0]
                else:
                    next_items = self.driver.find_elements(
                        self._by.CSS_SELECTOR,
                        (
                            "#module_product_review "
                            ".iweb-pagination-next"
                        ),
                    )
                    if not next_items:
                        raise ResponseSchemaError(
                            f"Cannot reach resume page {target_page}"
                        )
                    control = next_items[0]
                    expected_page = current_page + 1
            nested = control.find_elements(
                self._by.CSS_SELECTOR,
                "button, a",
            )
            clickable = nested[0] if nested else control
            current_page = self._click_and_wait(
                clickable,
                old_digest=old_digest,
                old_page=current_page,
                expected_page=expected_page,
            )
            if current_page < target_page and self.settings.page_delay_min:
                time.sleep(self.settings.page_delay_min)

    def iter_review_pages(
        self,
        product: ProductRecord,
        *,
        crawl_id: str,
        start_page: int = 1,
    ) -> Iterator[DomPage]:
        product_id = product.product_id or extract_item_id(product.url)
        if not product_id:
            raise ValueError(f"Cannot determine product ID from {product.url!r}")
        try:
            self.driver.get(product.url)
            self._wait_for_review_module()
        except (
            BlockedError,
            ProductUnavailableError,
            RateLimitedError,
            ResponseSchemaError,
        ):
            raise
        except Exception as exc:
            raise TransportError("Browser failed while loading product page") from exc

        if start_page < 1:
            raise ValueError("start_page must be positive")
        self._go_to_page(start_page)

        for page_number in range(
            start_page,
            self.settings.max_pages_per_product + 1,
        ):
            self._raise_if_challenged()
            rows = self._extract_rows()
            response_sha256 = self._page_digest(rows)
            total_pages, next_disabled = self._page_state()
            is_last = (
                not rows
                or next_disabled
                or (
                    total_pages is not None
                    and page_number >= total_pages
                )
            )
            reviews: list[ReviewRecord] = []
            for row in rows:
                text = normalize_text(str(row.get("review_text") or ""))
                if not text:
                    continue
                review_time = normalize_text(str(row.get("review_time") or ""))
                sku_info = normalize_text(str(row.get("sku_info") or ""))
                rating = row.get("rating")
                try:
                    rating = int(rating) if rating is not None else None
                except (TypeError, ValueError):
                    rating = None
                if rating is not None and not 1 <= rating <= 5:
                    rating = None
                reviews.append(
                    ReviewRecord(
                        crawl_id=crawl_id,
                        collected_at=datetime.now(timezone.utc).isoformat(),
                        collector_version=__version__,
                        collection_transport="selenium_dom",
                        sampling_frame="natural",
                        query=product.query,
                        product_id=product_id,
                        seller_id=product.seller_id,
                        source_url=product.url,
                        review_id=stable_dom_review_id(
                            product_id,
                            text,
                            review_time,
                            sku_info,
                        ),
                        review_text=text,
                        rating=rating,
                        review_time=review_time,
                        sku_info=sku_info,
                        has_images=bool(row.get("has_images")),
                        verified_purchase=None,
                        page_number=page_number,
                        response_sha256=response_sha256,
                        category=product.category,
                    )
                )
            yield DomPage(
                page_number=page_number,
                total_pages=total_pages,
                reviews=tuple(reviews),
                is_last=is_last,
            )
            if is_last:
                return
            delay = self._random.uniform(
                self.settings.page_delay_min,
                self.settings.page_delay_max,
            )
            if delay:
                time.sleep(delay)
            self._click_next(response_sha256, page_number)

    def search_products(
        self,
        query: str,
        *,
        page: int = 1,
        category: str = "",
        limit: int = 40,
    ) -> List[ProductRecord]:
        """Discover products from the rendered catalogue, without JSON APIs."""
        url = f"{CATALOG_URL}?{urlencode({'q': query, 'page': page})}"
        try:
            self.driver.get(url)
        except Exception as exc:
            raise TransportError("Browser failed while loading catalogue") from exc

        deadline = time.monotonic() + self.settings.load_timeout_seconds
        raw_products: list[dict[str, str]] = []
        while time.monotonic() < deadline:
            self._raise_if_challenged()
            raw_products = self.driver.execute_script(
                """
                return Array.from(document.querySelectorAll(
                  'a[href*="/products/"][href*="-i"]'
                )).map(anchor => ({
                  url: anchor.href,
                  name: (
                    anchor.getAttribute('title') ||
                    anchor.innerText ||
                    anchor.querySelector('img')?.getAttribute('alt') ||
                    ''
                  ).trim()
                }));
                """
            )
            if raw_products:
                break
            self.driver.execute_script(
                "window.scrollBy(0, Math.max(700, window.innerHeight))"
            )
            time.sleep(0.4)
        products: Dict[str, ProductRecord] = {}
        for row in raw_products or []:
            product_url = str(row.get("url") or "")
            product_id = extract_item_id(product_url)
            if not product_id or product_id in products:
                continue
            products[product_id] = ProductRecord(
                product_id=product_id,
                name=normalize_text(str(row.get("name") or ""))
                or f"Lazada product {product_id}",
                url=product_url,
                query=query,
                category=category,
            )
            if len(products) >= limit:
                break
        return list(products.values())

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "SeleniumDomCollector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
