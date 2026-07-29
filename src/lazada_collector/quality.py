"""Transparent heuristics for retaining substantive Vietnamese review text."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Tuple
import unicodedata

from .normalization import looks_like_mojibake, normalize_text
from .schema import ReviewRecord


SELECTION_POLICY = "substantive_vi_v2"
_WORD = re.compile(r"\b[^\W_]+\b", flags=re.UNICODE)
_SENTENCE_MARK = re.compile(r"[.!?…]")
_VIETNAMESE_CHARS = frozenset(
    "ăâđêôơư"
    "àáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễ"
    "ìíịỉĩòóọỏõồốộổỗờớợởỡ"
    "ùúụủũừứựửữỳýỵỷỹ"
)
_VIETNAMESE_WORDS = frozenset(
    {
        "anh",
        "bao",
        "ben",
        "bi",
        "chat",
        "cho",
        "co",
        "cua",
        "dung",
        "duoc",
        "giao",
        "gia",
        "hang",
        "kha",
        "khi",
        "khong",
        "lam",
        "minh",
        "mua",
        "nhan",
        "nhieu",
        "nhung",
        "on",
        "pham",
        "rat",
        "san",
        "shop",
        "tot",
        "trong",
        "va",
        "voi",
    }
)
_FOREIGN_SCRIPT_PREFIXES = (
    "ARABIC",
    "BENGALI",
    "CJK",
    "CYRILLIC",
    "DEVANAGARI",
    "HANGUL",
    "HEBREW",
    "HIRAGANA",
    "KATAKANA",
    "THAI",
)


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    min_chars: int = 80
    min_words: int = 15
    min_unique_word_ratio: float = 0.40
    min_meaningful_words: int = 8
    min_score: float = 0.55
    require_vietnamese: bool = True
    min_vietnamese_signals: int = 2
    max_foreign_script_ratio: float = 0.20
    reject_suspect_encoding: bool = True

    def validate(self) -> None:
        if self.min_chars < 1 or self.min_words < 1:
            raise ValueError("Quality length thresholds must be positive")
        if not 0 <= self.min_unique_word_ratio <= 1:
            raise ValueError("min_unique_word_ratio must be in [0, 1]")
        if self.min_meaningful_words < 1:
            raise ValueError("min_meaningful_words must be positive")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be in [0, 1]")
        if self.min_vietnamese_signals < 0:
            raise ValueError("min_vietnamese_signals cannot be negative")
        if not 0 <= self.max_foreign_script_ratio <= 1:
            raise ValueError("max_foreign_script_ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class QualityResult:
    accepted: bool
    score: float
    char_count: int
    word_count: int
    unique_word_ratio: float
    meaningful_word_count: int
    vietnamese_signal_count: int
    foreign_script_ratio: float
    reasons: Tuple[str, ...]


def _language_metrics(text: str, words: list[str]) -> tuple[int, float]:
    signals = sum(
        1
        for word in words
        if word in _VIETNAMESE_WORDS
        or any(character in _VIETNAMESE_CHARS for character in word)
    )
    letters = [character for character in text if character.isalpha()]
    foreign_letters = 0
    for character in letters:
        name = unicodedata.name(character, "")
        if name.startswith(_FOREIGN_SCRIPT_PREFIXES):
            foreign_letters += 1
    ratio = foreign_letters / len(letters) if letters else 0.0
    return signals, ratio


def evaluate_review(text: str, policy: QualityPolicy) -> QualityResult:
    policy.validate()
    normalized = normalize_text(text)
    words = [word.casefold() for word in _WORD.findall(normalized)]
    meaningful = [word for word in words if len(word) >= 3 and not word.isdigit()]
    char_count = len(normalized)
    word_count = len(words)
    unique_ratio = len(set(words)) / word_count if word_count else 0.0
    vietnamese_signals, foreign_script_ratio = _language_metrics(
        normalized,
        words,
    )

    length_score = min(1.0, char_count / 220.0)
    word_score = min(1.0, word_count / 35.0)
    diversity_score = min(1.0, unique_ratio / 0.70)
    sentence_score = 1.0 if _SENTENCE_MARK.search(normalized) else 0.5
    score = round(
        0.30 * length_score
        + 0.30 * word_score
        + 0.30 * diversity_score
        + 0.10 * sentence_score,
        4,
    )

    reasons = []
    if char_count < policy.min_chars:
        reasons.append("TOO_SHORT_CHARS")
    if word_count < policy.min_words:
        reasons.append("TOO_FEW_WORDS")
    if unique_ratio < policy.min_unique_word_ratio:
        reasons.append("LOW_LEXICAL_DIVERSITY")
    if len(meaningful) < policy.min_meaningful_words:
        reasons.append("TOO_FEW_MEANINGFUL_WORDS")
    if score < policy.min_score:
        reasons.append("LOW_QUALITY_SCORE")
    if policy.reject_suspect_encoding and looks_like_mojibake(normalized):
        reasons.append("SUSPECT_ENCODING")
    if (
        policy.require_vietnamese
        and vietnamese_signals < policy.min_vietnamese_signals
    ):
        reasons.append("INSUFFICIENT_VIETNAMESE_SIGNAL")
    if foreign_script_ratio > policy.max_foreign_script_ratio:
        reasons.append("FOREIGN_SCRIPT_DOMINANT")

    return QualityResult(
        accepted=not reasons,
        score=score,
        char_count=char_count,
        word_count=word_count,
        unique_word_ratio=round(unique_ratio, 4),
        meaningful_word_count=len(meaningful),
        vietnamese_signal_count=vietnamese_signals,
        foreign_script_ratio=round(foreign_script_ratio, 4),
        reasons=tuple(reasons),
    )


def attach_quality(
    review: ReviewRecord,
    result: QualityResult,
    selection_policy: str = SELECTION_POLICY,
) -> ReviewRecord:
    return replace(
        review,
        selection_policy=selection_policy,
        quality_score=result.score,
        char_count=result.char_count,
        word_count=result.word_count,
        unique_word_ratio=result.unique_word_ratio,
        vietnamese_signal_count=result.vietnamese_signal_count,
        foreign_script_ratio=result.foreign_script_ratio,
    )
