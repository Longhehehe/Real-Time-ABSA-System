"""HTTP and browser transports for Lazada public catalogue/review pages."""

from __future__ import annotations

import hashlib
from http.cookiejar import LoadError, MozillaCookieJar
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from .errors import (
    AuthenticationError,
    BlockedError,
    DependencyError,
    ResponseSchemaError,
    TransportError,
)
from .schema import ProductRecord


CATALOG_URL = "https://www.lazada.vn/catalog/"
REVIEW_URL = "https://my.lazada.vn/pdp/review/getReviewList"
HOME_URL = "https://www.lazada.vn/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": HOME_URL,
}

_BLOCK_MARKERS = (
    "captcha",
    "rgv587",
    "verify",
    "unusual traffic",
    "security check",
)


def load_lazada_cookies(path: Path) -> requests.cookies.RequestsCookieJar:
    """Load only non-expired Lazada cookies from a Netscape-format export."""
    if not path.is_file():
        raise AuthenticationError(f"Cookie file not found: {path}")

    source = MozillaCookieJar(str(path))
    try:
        source.load(ignore_discard=True, ignore_expires=True)
    except (LoadError, OSError) as exc:
        raise AuthenticationError(
            "Cookie file is not a valid Netscape cookie export"
        ) from exc

    filtered = requests.cookies.RequestsCookieJar()
    for cookie in source:
        domain = cookie.domain.lstrip(".").casefold()
        is_lazada = domain == "lazada.vn" or domain.endswith(".lazada.vn")
        if not is_lazada or cookie.is_expired():
            continue
        filtered.set_cookie(cookie)

    if not filtered:
        raise AuthenticationError(
            "Cookie export contains no active lazada.vn cookies"
        )
    return filtered


def browser_cookie_payloads(path: Path) -> List[Dict[str, Any]]:
    """Convert the filtered cookie export to Chrome DevTools cookie payloads."""
    payloads: List[Dict[str, Any]] = []
    for cookie in load_lazada_cookies(path):
        payload: Dict[str, Any] = {
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path or "/",
            "secure": bool(cookie.secure),
            "httpOnly": any(
                str(key).casefold() == "httponly"
                for key in (cookie._rest or {})
            ),
        }
        if cookie.expires is not None:
            payload["expires"] = float(cookie.expires)
        payloads.append(payload)
    return payloads


def extract_item_id(url: str) -> Optional[str]:
    patterns = (
        r"-i(\d+)-s",
        r"-i(\d+)\.",
        r"-i(\d+)$",
        r"itemId=(\d+)",
        r"/i(\d+)\?",
        r"/i(\d+)$",
        r"/(\d{6,})[-.]",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _absolute_url(value: str) -> str:
    if value.startswith("//"):
        return f"https:{value}"
    if value.startswith("/"):
        return f"https://www.lazada.vn{value}"
    return value


def _as_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return False


def parse_search_payload(payload: Dict[str, Any], query: str = "") -> List[ProductRecord]:
    try:
        items = payload["mods"]["listItems"]
    except (KeyError, TypeError) as exc:
        raise ResponseSchemaError("Search response has no mods.listItems array") from exc
    if not isinstance(items, list):
        raise ResponseSchemaError("Search response mods.listItems is not an array")

    products: List[ProductRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        product_id = str(item.get("itemId") or item.get("nid") or "").strip()
        if not product_id:
            continue
        raw_url = str(item.get("itemUrl") or item.get("productUrl") or "")
        url = _absolute_url(raw_url)
        if not url:
            url = f"https://www.lazada.vn/products/-i{product_id}.html"
        categories = item.get("categories")
        category = ""
        if isinstance(categories, list) and categories:
            category = str(categories[-1])
        review_count = item.get("review")
        try:
            review_count = int(review_count) if review_count is not None else None
        except (TypeError, ValueError):
            review_count = None
        rating = item.get("ratingScore")
        try:
            rating = float(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating = None
        sponsored_value = (
            item.get("isSponsored")
            if "isSponsored" in item
            else item.get("adFlag")
        )
        products.append(
            ProductRecord(
                product_id=product_id,
                name=str(item.get("name") or "Unknown product"),
                url=url,
                query=query,
                seller_id=str(item.get("sellerId") or ""),
                seller_name=str(item.get("sellerName") or ""),
                category=category,
                price=str(item.get("priceShow") or item.get("price") or ""),
                rating=rating,
                review_count=review_count,
                location=str(item.get("location") or ""),
                is_sponsored=_as_boolean(sponsored_value),
            )
        )
    return products


def parse_review_payload(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if payload.get("success") is False:
        raise ResponseSchemaError("Review endpoint returned success=false")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ResponseSchemaError("Review response has no model object")
    items = model.get("items")
    if not isinstance(items, list):
        raise ResponseSchemaError("Review response model.items is not an array")
    page_count = model.get("pageCount")
    try:
        page_count = int(page_count) if page_count is not None else None
    except (TypeError, ValueError):
        page_count = None
    return items, page_count


class RequestsTransport:
    name = "requests"

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        cookie_file: Optional[Path] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.cookie_file = cookie_file
        if cookie_file is not None:
            self.name = "requests_cookie"
        self.session = self._new_session()

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        if self.cookie_file is not None:
            session.cookies = load_lazada_cookies(self.cookie_file)
        return session

    def reset(self) -> None:
        self.session.close()
        self.session = self._new_session()

    def _get_json(
        self,
        url: str,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise TransportError(f"Request failed: {type(exc).__name__}") from exc

        body = response.content
        digest = hashlib.sha256(body).hexdigest()
        if response.status_code != 200:
            if response.status_code in {403, 429}:
                raise BlockedError(
                    f"Lazada rate-limited the cookie session (HTTP {response.status_code})"
                )
            raise TransportError(f"HTTP {response.status_code} from Lazada")

        content_type = response.headers.get("Content-Type", "").lower()
        if "json" not in content_type:
            lowered = response.text[:5000].lower()
            if "html" in content_type or any(marker in lowered for marker in _BLOCK_MARKERS):
                raise BlockedError(
                    "Lazada returned an HTML challenge instead of JSON"
                )
            raise ResponseSchemaError(
                f"Expected JSON response, received {content_type or 'unknown content type'}"
            )
        try:
            # Lazada occasionally declares an incorrect response charset.
            # Decode the original JSON bytes explicitly to prevent mojibake.
            payload = json.loads(body.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponseSchemaError("Lazada returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise ResponseSchemaError("Lazada JSON root is not an object")
        return payload, digest

    def search(
        self,
        query: str,
        page: int = 1,
    ) -> Tuple[List[ProductRecord], str]:
        payload, digest = self._get_json(
            CATALOG_URL,
            {"q": query, "page": page, "ajax": "true"},
        )
        return parse_search_payload(payload, query=query), digest

    def fetch_review_page(
        self,
        item_id: str,
        page_number: int,
        page_size: int,
        rating_filter: int = 0,
    ) -> Tuple[Dict[str, Any], str]:
        return self._get_json(
            REVIEW_URL,
            {
                "itemId": item_id,
                "pageSize": page_size,
                "filter": rating_filter,
                "sort": 0,
                "pageNo": page_number,
            },
        )

    def close(self) -> None:
        self.session.close()


class SeleniumTransport:
    """Use a normal browser session when the requests client is challenged."""

    name = "selenium"

    def __init__(
        self,
        headless: bool = True,
        profile_dir: Optional[Path] = None,
        timeout_seconds: float = 30.0,
        cookie_file: Optional[Path] = None,
    ):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
        except ImportError as exc:
            raise DependencyError(
                "Selenium is required for browser transport. "
                "Install the project with: pip install -e \".[browser]\""
            ) from exc

        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-notifications")
        options.add_argument("--lang=vi-VN")
        options.add_argument("--window-size=1440,1200")
        if profile_dir is not None:
            profile_dir.mkdir(parents=True, exist_ok=True)
            options.add_argument(f"--user-data-dir={profile_dir.resolve()}")

        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as exc:
            raise DependencyError(
                "Chrome could not be started by Selenium Manager"
            ) from exc
        self._by = By
        self._wait = WebDriverWait(self.driver, timeout_seconds)
        self.cookie_file_configured = cookie_file is not None
        self.cookies_imported = 0
        self.cookie_import_failed = False
        if cookie_file is not None:
            try:
                cookie_payloads = browser_cookie_payloads(cookie_file)
            except AuthenticationError:
                # DOM collection can still work through its persistent browser
                # profile when the API cookie export has expired.
                self.cookie_import_failed = True
            else:
                for payload in cookie_payloads:
                    try:
                        result = self.driver.execute_cdp_cmd(
                            "Network.setCookie",
                            payload,
                        )
                    except Exception:
                        continue
                    if not isinstance(result, dict) or result.get("success", True):
                        self.cookies_imported += 1
        self.driver.get(HOME_URL)

    def _get_json(self, url: str) -> Tuple[Dict[str, Any], str]:
        try:
            self.driver.get(url)
            body = self._wait.until(
                lambda driver: driver.find_element(self._by.TAG_NAME, "body")
            )
            text = (body.get_attribute("innerText") or body.text or "").strip()
        except Exception as exc:
            raise TransportError("Browser failed while loading Lazada") from exc

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        lowered = text[:5000].lower()
        if any(marker in lowered for marker in _BLOCK_MARKERS):
            raise BlockedError("Lazada displayed a browser challenge")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ResponseSchemaError(
                "Browser page did not contain a JSON response"
            ) from exc
        if not isinstance(payload, dict):
            raise ResponseSchemaError("Lazada JSON root is not an object")
        return payload, digest

    def fetch_review_page(
        self,
        item_id: str,
        page_number: int,
        page_size: int,
        rating_filter: int = 0,
    ) -> Tuple[Dict[str, Any], str]:
        query = urlencode(
            {
                "itemId": item_id,
                "pageSize": page_size,
                "filter": rating_filter,
                "sort": 0,
                "pageNo": page_number,
            }
        )
        return self._get_json(f"{REVIEW_URL}?{query}")

    def search(
        self,
        query: str,
        page: int = 1,
    ) -> Tuple[List[ProductRecord], str]:
        encoded = urlencode({"q": query, "page": page, "ajax": "true"})
        payload, digest = self._get_json(f"{CATALOG_URL}?{encoded}")
        return parse_search_payload(payload, query=query), digest

    def close(self) -> None:
        self.driver.quit()
