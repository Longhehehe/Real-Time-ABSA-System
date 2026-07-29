"""Conservative text normalization used only for duplicate detection."""

from __future__ import annotations

import hashlib
import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")
_STRONG_MOJIBAKE_MARKERS = ("ðŸ", "â€", "ï¿½")
_MOJIBAKE_MARKERS = ("Ã", "Â", "Ð", "Ñ", "áº", "á»")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    return _WHITESPACE.sub(" ", normalized).strip()


def looks_like_mojibake(text: str) -> bool:
    """Detect common UTF-8 decoded-as-Latin-1 corruption without repairing it."""
    normalized = normalize_text(text)
    if any(marker in normalized for marker in _STRONG_MOJIBAKE_MARKERS):
        return True
    hits = sum(normalized.count(marker) for marker in _MOJIBAKE_MARKERS)
    return hits >= 2


def duplicate_key(text: str) -> str:
    canonical = normalize_text(text).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
