"""Deterministic, audit-first helpers for semantic corpus curation.

These functions never mutate raw records.  They create comparison views,
evidence, and derived text while retaining the original text verbatim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
import hashlib
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_CLAUSE_SPLIT_RE = re.compile(r"[,.!?…;:\n\r|•]+")
SKU_FIELD_LABELS = (
    "color family",
    "colour family",
    "màu sắc",
    "nhóm màu",
    "phân loại hàng",
    "phân loại",
    "variation",
    "kích thước",
    "size",
    "khối lượng",
    "trọng lượng",
    "số lượng lon",
    "số lượng",
    "dung tích",
    "thể tích",
    "hương vị",
    "mùi",
    "model",
    "loại",
)
SKU_FIELD_LABEL_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(label) for label in sorted(
        SKU_FIELD_LABELS,
        key=lambda value: (-len(value), value),
    ))
    + r")\s*:",
    flags=re.IGNORECASE,
)
ABSOLUTE_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d %b %Y",
    "%d %B %Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
)
ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "but",
        "by",
        "can",
        "for",
        "from",
        "good",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "my",
        "not",
        "of",
        "on",
        "or",
        "product",
        "so",
        "that",
        "the",
        "this",
        "to",
        "very",
        "was",
        "with",
        "you",
    }
)
TAGALOG_FUNCTION_WORDS = frozenset(
    {
        "ako",
        "ang",
        "bumili",
        "dahil",
        "divisoria",
        "gaan",
        "gawa",
        "hindi",
        "ito",
        "ka",
        "karton",
        "lang",
        "may",
        "mga",
        "mo",
        "na",
        "naka",
        "naman",
        "napaka",
        "ng",
        "para",
        "pero",
        "sa",
        "sapatos",
        "siya",
        "tig",
        "wala",
        "walang",
        "wlang",
        "yung",
    }
)
VIETNAMESE_DIACRITICS = frozenset(
    "ăâđêôơư"
    "àáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễ"
    "ìíịỉĩòóọỏõồốộổỗờớợởỡ"
    "ùúụủũừứựửữỳýỵỷỹ"
)
NATURAL_ELONGATION_ROOTS = frozenset(
    {
        "cham",
        "dep",
        "duoc",
        "khong",
        "khoong",
        "mau",
        "mem",
        "ngon",
        "nhanh",
        "nhe",
        "ok",
        "on",
        "re",
        "sao",
        "thom",
        "tiep",
        "tot",
        "tuyet",
        "xinh",
        "xin",
        "yeu",
    }
)
REWARD_ALLOWED_TOKENS = frozenset(
    {
        "50",
        "anh",
        "cho",
        "chi",
        "chu",
        "co",
        "day",
        "de",
        "du",
        "dung",
        "gian",
        "hinh",
        "hoa",
        "khong",
        "kiem",
        "ky",
        "lay",
        "lien",
        "mang",
        "minh",
        "nha",
        "nhan",
        "noi",
        "quan",
        "tc",
        "the",
        "thoi",
        "tinh",
        "tu",
        "video",
        "viet",
        "xu",
    }
)
CATALOGUE_OPENERS = (
    "an toan",
    "am tram",
    "bao bi tien loi",
    "bao ve",
    "ben",
    "be mat chong",
    "cac thanh phan",
    "chat lieu",
    "chiet xuat",
    "chong",
    "chuc nang chong",
    "cong thuc",
    "cung cap",
    "dam bao",
    "da cam thay",
    "dat chung nhan",
    "de dang",
    "di kem",
    "dinh duong",
    "duong am",
    "gia tri",
    "giu toc",
    "giup",
    "hoan hao",
    "ho tro",
    "huong vi vani",
    "kich thuoc",
    "khong gay",
    "lam sang",
    "lua chon",
    "ly tuong",
    "mang den",
    "mang lai",
    "micro",
    "mon qua hoan hao",
    "ngan chua",
    "nhap khau",
    "nhe nhang",
    "nhieu kich thuoc",
    "num vu",
    "phu hop",
    "tao hieu ung",
    "tang cuong",
    "thanh phan",
    "than thien",
    "thiet ke",
    "tien loi",
    "toi uu",
    "trai nghiem ca phe",
    "tuyet voi",
)
CATALOGUE_HIGH_PHRASE_RE = re.compile(
    r"\b(?:hoan hao (?:cho|de)|gia tri tuyet voi|"
    r"thiet ke (?:hien dai|thoi trang)|cong thuc khong|"
    r"cung cap do|tang cuong|nhap khau tu|dat chung nhan|"
    r"bao bi tien loi|chiet xuat tu nhien|phu hop voi moi|"
    r"mon qua hoan hao|goi ranh tay|am thanh song dong|"
    r"chong va dap|co san trong nhieu)\b",
    flags=re.IGNORECASE,
)
BUYER_PRONOUN_RE = re.compile(
    r"\b(?:mình|minh|tôi|toi|tui|mik|mk|em|"
    r"nhà mình|nha minh|bé nhà|be nha|vợ|vo|chồng|chong)\b",
    flags=re.IGNORECASE,
)
BUYER_ACTION_RE = re.compile(
    r"\b(?:mua|đặt|dat|nhận|nhan|giao|xài|xai|sài|sai|"
    r"dùng|dung|sử dụng|su dung|thử|thu|test|giặt|giat|"
    r"gội|goi|uống|uong|ăn|an|mang|đeo|deo|sạc|sac|xay|"
    r"nấu|nau|pha|mở|mo|khui|đổi|doi|hoàn|hoan|"
    r"hài lòng|hai long|ưng|ung|thích|thich)\b|"
    r"\b(?:trả hàng|tra hang|trả lại|tra lai)\b",
    flags=re.IGNORECASE,
)
TRANSACTION_RE = re.compile(
    r"\b(?:shop|shipper|giao hàng|giao hang|nhận hàng|nhan hang|"
    r"đặt hàng|dat hang|đóng gói|dong goi|"
    r"giao (?:sai|nhầm|nham|thiếu|thieu)|"
    r"không (?:đúng|dung|giống|giong)|"
    r"thiếu hàng|thieu hang|đổi hàng|doi hang|"
    r"trả hàng|tra hang|hoàn tiền|hoan tien)\b",
    flags=re.IGNORECASE,
)
CONCRETE_DEFECT_RE = re.compile(
    r"\b(?:bị|bi) (?:rách|rach|hỏng|hong|móp|mop|xước|xuoc|"
    r"lỗi|loi|gãy|gay|thiếu|thieu|lỏng|long|nứt|nut|vỡ|vo)\b|"
    r"\b(?:hết pin|het pin|không hoạt động|khong hoat dong|"
    r"không vừa|khong vua|kém chất lượng|kem chat luong|"
    r"hàng giả|hang gia|hàng fake|hang fake|sai màu|sai mau|"
    r"sai size|nhỏ hơn|nho hon|to hơn|to hon)\b",
    flags=re.IGNORECASE,
)
CONTRAST_RE = re.compile(
    r"\b(?:nhưng|nhung|tuy nhiên|tuy nhien|mặc dù|mac du|"
    r"hơi|hoi|kém|kem|tệ|te|thất vọng|that vong|"
    r"không nên|khong nen)\b",
    flags=re.IGNORECASE,
)
INFORMAL_REVIEW_RE = re.compile(
    r"\b(?:ok|oke|okie|nha|nhé|nhe|ko|dc|sp|sop|mn|"
    r"vl|huhu|giá rẻ|gia re|giá mềm|gia mem|"
    r"được tặng|duoc tang)\b",
    flags=re.IGNORECASE,
)
GENERIC_RECOMMENDATION_RE = re.compile(
    r"\b(?:ai thich.{0,40}mua|moi nguoi.{0,40}mua|"
    r"cac ban.{0,40}mua|nen mua|nhanh tay)\b",
    flags=re.IGNORECASE,
)
UI_FIELD_VALUE_RE = re.compile(
    r"\b(?:quality|texture|effectiveness|fragrance|price|taste|"
    r"flavou?r|convenience|performance|design|fit|style|size|"
    r"material|capacity|portability|charging speed|sound quality|"
    r"battery life|chat luong|huong thom|hieu qua|ket cau|mui vi|"
    r"dang tien|tinh di dong|toc do sac|dung luong|do ben|do bam|"
    r"su thoai mai)\s*:\s*([^\s,.;:!?]{1,20})",
    flags=re.IGNORECASE,
)
REVIEWER_PRONOUN_TOKEN_RE = re.compile(
    r"\b(?:mình|minh|tôi|toi|tui|mik|mk|em|vợ|vo|chồng|chong)\b",
    flags=re.IGNORECASE,
)
REVIEWER_DEFECT_CONTRAST_TOKEN_RE = re.compile(
    r"\b(?:nhưng|nhung|bị|bi|đổi|doi|trả|tra)\b",
    flags=re.IGNORECASE,
)
REVIEWER_ACTOR_TOKEN_RE = re.compile(
    r"\b(?:shop|shipper|sop)\b",
    flags=re.IGNORECASE,
)
REVIEWER_PHRASE_RE = re.compile(
    r"\b(?:sau khi|cảm thấy|cam thay|săn sale|san sale|"
    r"ủng hộ shop|ung ho shop|tặng shop|tang shop|"
    r"lần sau|lan sau|sẽ mua lại|se mua lai|"
    r"nhận được|nhan duoc|nhận hàng|nhan hang|"
    r"tuy nhiên|tuy nhien|không đúng|khong dung|"
    r"không giống|khong giong|thất vọng|that vong)\b",
    flags=re.IGNORECASE,
)
REVIEWER_TEMPORAL_ACTION_RE = re.compile(
    r"\b(?:đã|da|mới|moi|vừa|vua)\b"
    r"(?:\s+\w+){0,7}\s+"
    r"(?:mua|đặt|dat|nhận|nhan|dùng|dung|xài|xai|"
    r"sài|sai|thử|thu|khui|mở|mo|uống|uong|ăn|an|"
    r"mang|mặc|mac|đeo|deo|nấu|nau|giặt|giat|"
    r"gội|goi|sạc|sac|test)\b",
    flags=re.IGNORECASE,
)
EXPANDED_SPEC_CLAUSE_RE = re.compile(
    r"^(?:nguyen lieu|thanh phan|khong (?:chua|chat|co)|"
    r"han su dung|goi dung|tui (?:tra|co the|loc)|bao bi|"
    r"dung tich|dung luong|pin |thoi luong pin|cong suat|"
    r"kich thuoc|chat lieu|bao hanh|xuat xu|san xuat|"
    r"nhap khau|mau sac|mau |thiet ke|tinh nang|he thong|"
    r"luc hut|cong thuc|microphone|phich cam|do tre|"
    r"dai nhiet do|dieu khien|tuoi tho|trong luong|"
    r"ket noi|cam bien)\b",
    flags=re.IGNORECASE,
)
GLUED_CATALOGUE_SUFFIX_RE = re.compile(
    r"^(?:tuyet voi cho|hoan hao cho|thiet ke|phu hop|"
    r"chong|chat lieu|tang |giam |giup |mang lai|de lai|"
    r"dung tich|dung luong|pin |tuoi tho|bao bi|cong thuc|"
    r"huong thom|thanh phan|tinh nang|ket noi|sac nhanh|"
    r"ho tro|lua chon|nhe va|nhe nhang|tien loi|san pham|"
    r"trai nghiem|luc hut|dieu khien|cong nghe|khoi dong|"
    r"man hinh|kich thuoc|mau sac|nhieu |goi dung|tui tra|"
    r"hieu qua)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Clause:
    index: int
    start: int
    end: int
    raw: str
    content: str
    key: str
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SimilarPair:
    left: int
    right: int
    intersection: int
    union: int

    @property
    def score(self) -> float:
        return self.intersection / self.union if self.union else 0.0


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(
        " ",
        unicodedata.normalize("NFKC", str(text)),
    ).strip()


def word_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return tuple(WORD_RE.findall(normalized))


def word_key(text: str) -> str:
    return " ".join(word_tokens(text))


def punctuation_symbol_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(
        "".join(
            character
            if character.isalnum() or character.isspace()
            else " "
            for character in normalized
        ).split()
    )


def accent_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text).casefold())
    folded = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return folded.replace("đ", "d")


def stable_rank(seed: str, sample_id: str) -> str:
    return sha256_text(f"{seed}\0{sample_id}")


def canonicalize_headings(
    text: str,
    aliases: dict[str, str],
) -> str:
    result = unicodedata.normalize("NFKC", str(text))
    for alias in sorted(aliases, key=lambda value: (-len(value), value)):
        canonical = aliases[alias]
        result = re.sub(
            rf"(?<!\w){re.escape(alias)}\s*:",
            f"{canonical}:",
            result,
            flags=re.IGNORECASE,
        )
    return result


def canonical_body_key(text: str, aliases: dict[str, str]) -> str:
    return word_key(canonicalize_headings(text, aliases))


def normalize_sku(text: str) -> str:
    """Compare SKU values while ignoring translated UI field names."""
    # DOM extraction sometimes concatenates fields without commas, for
    # example ``Khối lượng: 800g Size: 800 g Số lượng lon: 1``.
    # Replacing known bilingual labels with a separator makes that form
    # comparable with structured API metadata while retaining only values.
    prepared = SKU_FIELD_LABEL_RE.sub("|", normalize_whitespace(text))
    values: list[str] = []
    for raw_part in re.split(r"[,;|]+", prepared):
        part = raw_part.strip()
        if not part:
            continue
        value = part.split(":", 1)[1] if ":" in part else part
        key = punctuation_symbol_key(value)
        if key:
            values.append(key)
    return "|".join(sorted(set(values)))


def parse_absolute_review_date(text: str) -> str | None:
    value = normalize_whitespace(text)
    if not value:
        return None
    iso_prefix = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
    if iso_prefix:
        try:
            return date.fromisoformat(iso_prefix.group(1)).isoformat()
        except ValueError:
            return None
    vietnamese = re.search(
        r"\b(\d{1,2})\s+(?:thg|tháng)\s+(\d{1,2})[,\s]+(\d{4})\b",
        value,
        flags=re.IGNORECASE,
    )
    if vietnamese:
        try:
            return date(
                int(vietnamese.group(3)),
                int(vietnamese.group(2)),
                int(vietnamese.group(1)),
            ).isoformat()
        except ValueError:
            return None
    for date_format in ABSOLUTE_DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            continue
    return None


def word_ngrams(text: str, size: int = 5) -> frozenset[tuple[str, ...]]:
    tokens = word_tokens(text)
    if len(tokens) < size:
        return frozenset()
    return frozenset(
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    )


def length_ratio(left: str, right: str) -> float:
    left_size = len(word_tokens(left))
    right_size = len(word_tokens(right))
    if not left_size or not right_size:
        return 0.0
    return min(left_size, right_size) / max(left_size, right_size)


def jaccard_sets(
    left: frozenset[Any] | set[Any],
    right: frozenset[Any] | set[Any],
) -> tuple[int, int, float]:
    intersection = len(left & right)
    union = len(left | right)
    return intersection, union, intersection / union if union else 0.0


def extract_clauses(
    text: str,
    *,
    min_tokens: int = 4,
    min_characters: int = 20,
    splitter: re.Pattern[str] = DEFAULT_CLAUSE_SPLIT_RE,
) -> list[Clause]:
    clauses: list[Clause] = []
    cursor = 0
    index = 0
    for delimiter in splitter.finditer(text):
        end = delimiter.end()
        raw = text[cursor:end]
        content = text[cursor : delimiter.start()].strip()
        tokens = word_tokens(content)
        key = " ".join(tokens) if (
            len(tokens) >= min_tokens and len(content) >= min_characters
        ) else ""
        if content:
            clauses.append(
                Clause(
                    index=index,
                    start=cursor,
                    end=end,
                    raw=raw,
                    content=content,
                    key=key,
                    tokens=tokens,
                )
            )
            index += 1
        cursor = end
    if cursor < len(text):
        raw = text[cursor:]
        content = raw.strip()
        tokens = word_tokens(content)
        key = " ".join(tokens) if (
            len(tokens) >= min_tokens and len(content) >= min_characters
        ) else ""
        if content:
            clauses.append(
                Clause(
                    index=index,
                    start=cursor,
                    end=len(text),
                    raw=raw,
                    content=content,
                    key=key,
                    tokens=tokens,
                )
            )
    return clauses


def _starts_with_uppercase_letter(text: str) -> bool:
    for character in str(text):
        if character.isalpha():
            return character.isupper()
    return False


def _reviewer_anchor_veto(text: str) -> bool:
    """Detect conservative reviewer evidence without accent folding.

    Accent preservation is intentional: folding would make Vietnamese words
    such as ``tối``, ``túi``, ``trà`` or ``chống`` collide with reviewer
    markers ``toi``, ``tui``, ``tra`` and ``chong``.
    """
    key = punctuation_symbol_key(text)
    return any(
        pattern.search(key)
        for pattern in (
            REVIEWER_PRONOUN_TOKEN_RE,
            REVIEWER_DEFECT_CONTRAST_TOKEN_RE,
            REVIEWER_ACTOR_TOKEN_RE,
            REVIEWER_PHRASE_RE,
            REVIEWER_TEMPORAL_ACTION_RE,
        )
    )


def _glued_catalogue_boundaries(text: str) -> list[dict[str, Any]]:
    """Return lower-to-upper glue boundaries that start catalogue copy."""
    value = str(text)
    evidence: list[dict[str, Any]] = []
    for boundary in range(1, len(value)):
        left = value[boundary - 1]
        right = value[boundary]
        if not (
            left.isalpha()
            and left.islower()
            and right.isalpha()
            and right.isupper()
        ):
            continue
        folded_suffix = punctuation_symbol_key(
            accent_fold(value[boundary:])
        )
        match = GLUED_CATALOGUE_SUFFIX_RE.match(folded_suffix)
        if match is None:
            continue
        evidence.append(
            {
                "boundary": boundary,
                "opener": match.group(0),
                "suffix_sha256": sha256_text(value[boundary:]),
            }
        )
    return evidence


def structural_catalogue_evidence(
    text: str,
    *,
    minimum_segments: int,
    minimum_list_delimiters: int,
    short_segment_min_tokens: int,
    short_segment_max_tokens: int,
    minimum_short_segment_ratio: float,
    minimum_catalogue_openers: int,
    global_recurrent_clause_count: int = 0,
    global_template_coverage: float = 0.0,
    product_recurrent_clause_count: int = 0,
    product_template_coverage: float = 0.0,
    weak_global_min_clauses: int = 1,
    weak_global_min_coverage: float = 0.20,
    weak_product_min_clauses: int = 2,
    weak_product_min_coverage: float = 0.55,
    unique_min_marketing_ratio: float = 0.50,
    low_density_min_marketing_ratio: float = 1 / 3,
    modular_min_segments: int = 2,
    modular_trailing_min_short_ratio: float = 0.75,
    modular_cleaned_min_short_ratio: float = 0.60,
    modular_min_capitalized_ratio: float = 0.60,
    expanded_spec_min_clauses: int = 2,
    glued_min_segments: int = 3,
    was_cleaned: bool = False,
) -> dict[str, Any]:
    """Flag list-like catalogue copy lacking buyer/transaction anchors.

    This is deliberately a quarantine signal, never an automatic deletion
    rule.  It targets modular feature lists that exact clause-frequency rules
    miss when wording is unique or lightly varied.
    """
    segments = [
        clause
        for clause in extract_clauses(
            text,
            min_tokens=1,
            min_characters=1,
        )
        if clause.tokens
    ]
    segment_count = len(segments)
    short_segments = sum(
        short_segment_min_tokens
        <= len(clause.tokens)
        <= short_segment_max_tokens
        for clause in segments
    )
    short_ratio = (
        short_segments / segment_count if segment_count else 0.0
    )
    capitalized_initial_count = sum(
        _starts_with_uppercase_letter(clause.content)
        for clause in segments
    )
    capitalized_initial_ratio = (
        capitalized_initial_count / segment_count
        if segment_count
        else 0.0
    )
    opener_count = 0
    marketing_flags = []
    folded_segments = []
    buyer_segments = []
    for clause in segments:
        folded = punctuation_symbol_key(accent_fold(clause.content))
        folded_segments.append(folded)
        buyer_segments.append(punctuation_symbol_key(clause.content))
        is_marketing = any(
            folded == opener or folded.startswith(opener + " ")
            for opener in CATALOGUE_OPENERS
        ) or bool(CATALOGUE_HIGH_PHRASE_RE.search(folded))
        marketing_flags.append(is_marketing)
        if is_marketing:
            opener_count += 1
    list_delimiters = len(re.findall(r"[,;|\n\r•]", text))
    buyer_residual_clauses = 0
    for buyer_text, is_marketing in zip(
        buyer_segments,
        marketing_flags,
    ):
        if is_marketing:
            continue
        pronoun_action = (
            BUYER_PRONOUN_RE.search(buyer_text)
            and BUYER_ACTION_RE.search(buyer_text)
        )
        substantive = any(
            pattern.search(buyer_text)
            for pattern in (
                TRANSACTION_RE,
                CONCRETE_DEFECT_RE,
                CONTRAST_RE,
            )
        )
        informal = (
            INFORMAL_REVIEW_RE.search(buyer_text)
            and not GENERIC_RECOMMENDATION_RE.search(
                punctuation_symbol_key(accent_fold(buyer_text))
            )
        )
        if pronoun_action or substantive or informal:
            buyer_residual_clauses += 1
    ui_field_values = [
        value
        for value in UI_FIELD_VALUE_RE.findall(
            normalize_whitespace(accent_fold(text))
        )
        if value.strip()
    ]
    strong_experience = (
        buyer_residual_clauses > 0
        or len(ui_field_values) >= 2
    )
    list_like = (
        segment_count >= minimum_segments
        and list_delimiters >= minimum_list_delimiters
        and short_ratio >= minimum_short_segment_ratio
    )
    trailing_list_delimiter = bool(re.search(r"[,;|•]\s*$", text))
    recurrent_weak = (
        (
            global_recurrent_clause_count >= weak_global_min_clauses
            and global_template_coverage >= weak_global_min_coverage
        )
        or (
            product_recurrent_clause_count >= weak_product_min_clauses
            and product_template_coverage >= weak_product_min_coverage
        )
    )
    marketing_ratio = (
        opener_count / segment_count if segment_count else 0.0
    )
    reviewer_anchor_veto = _reviewer_anchor_veto(text)
    expanded_spec_clause_count = sum(
        bool(
            EXPANDED_SPEC_CLAUSE_RE.match(
                punctuation_symbol_key(accent_fold(clause.content))
            )
        )
        for clause in segments
    )
    glued_catalogue_boundaries = _glued_catalogue_boundaries(text)
    recurrent_list_rule = (
        list_like
        and recurrent_weak
        and opener_count >= 1
        and not strong_experience
        and (
            opener_count >= minimum_catalogue_openers
            or trailing_list_delimiter
        )
    )
    unique_dense_rule = (
        list_like
        and opener_count >= minimum_catalogue_openers
        and marketing_ratio >= unique_min_marketing_ratio
        and not strong_experience
        and (
            trailing_list_delimiter
            or opener_count >= 3
            or recurrent_weak
            or was_cleaned
        )
    )
    recurrent_spec_list_rule = (
        list_like
        and recurrent_weak
        and trailing_list_delimiter
        and not strong_experience
    )
    low_density_brochure_rule = (
        list_like
        and opener_count >= 1
        and marketing_ratio >= low_density_min_marketing_ratio
        and not strong_experience
        and (
            trailing_list_delimiter
            or was_cleaned
        )
    )
    narrow_two_clause_rule = (
        segment_count == 2
        and short_ratio == 1.0
        and opener_count == 2
        and not strong_experience
        and (
            trailing_list_delimiter
            or was_cleaned
            or recurrent_weak
        )
    )
    cleaned_residue_rule = (
        was_cleaned
        and recurrent_weak
        and opener_count >= 2
        and not strong_experience
        and product_template_coverage >= 0.60
    )
    modular_title_list_rule = (
        segment_count >= modular_min_segments
        and (
            (
                trailing_list_delimiter
                and short_ratio
                >= modular_trailing_min_short_ratio
            )
            or (
                was_cleaned
                and short_ratio
                >= modular_cleaned_min_short_ratio
            )
        )
        and (
            capitalized_initial_ratio
            >= modular_min_capitalized_ratio
            or recurrent_weak
        )
        and not reviewer_anchor_veto
    )
    expanded_recurrent_spec_list_rule = (
        list_like
        and recurrent_weak
        and buyer_residual_clauses == 0
        and expanded_spec_clause_count >= expanded_spec_min_clauses
        and capitalized_initial_ratio
        >= modular_min_capitalized_ratio
    )
    glued_catalogue_clause_rule = (
        segment_count >= glued_min_segments
        and bool(glued_catalogue_boundaries)
        and (
            trailing_list_delimiter
            or was_cleaned
        )
    )
    rule_hits = [
        name
        for name, hit in (
            ("RECURRENT_LIST", recurrent_list_rule),
            ("UNIQUE_LEXICON_DENSE", unique_dense_rule),
            ("RECURRENT_SPEC_LIST", recurrent_spec_list_rule),
            ("LOW_DENSITY_BROCHURE_LIST", low_density_brochure_rule),
            ("NARROW_TWO_CLAUSE", narrow_two_clause_rule),
            ("CLEANED_TEMPLATE_RESIDUE", cleaned_residue_rule),
            ("MODULAR_TITLE_LIST", modular_title_list_rule),
            (
                "EXPANDED_RECURRENT_SPEC_LIST",
                expanded_recurrent_spec_list_rule,
            ),
            (
                "GLUED_CATALOGUE_CLAUSE",
                glued_catalogue_clause_rule,
            ),
        )
        if hit
    ]
    flagged = bool(rule_hits)
    return {
        "flagged": flagged,
        "rule_hits": rule_hits,
        "list_like": list_like,
        "segment_count": segment_count,
        "list_delimiter_count": list_delimiters,
        "short_segment_count": short_segments,
        "short_segment_ratio": round(short_ratio, 6),
        "capitalized_initial_count": capitalized_initial_count,
        "capitalized_initial_ratio": round(
            capitalized_initial_ratio,
            6,
        ),
        "catalogue_opener_count": opener_count,
        "marketing_clause_ratio": round(marketing_ratio, 6),
        "trailing_list_delimiter": trailing_list_delimiter,
        "recurrent_weak": recurrent_weak,
        "buyer_residual_clause_count": buyer_residual_clauses,
        "ui_field_value_count": len(ui_field_values),
        "has_experience_anchor": strong_experience,
        "reviewer_anchor_veto": reviewer_anchor_veto,
        "expanded_spec_clause_count": expanded_spec_clause_count,
        "glued_catalogue_boundaries": glued_catalogue_boundaries,
        "was_cleaned": was_cleaned,
    }


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(
        (max(0, start), max(0, end))
        for start, end in ranges
        if end > start
    )
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def remove_ranges(
    text: str,
    ranges: Iterable[tuple[int, int]],
) -> tuple[str, list[dict[str, Any]]]:
    merged = merge_ranges(ranges)
    pieces: list[str] = []
    transformations: list[dict[str, Any]] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        removed = text[start:end]
        transformations.append(
            {
                "start": start,
                "end": end,
                "removed_sha256": sha256_text(removed),
                "removed_characters": len(removed),
            }
        )
        cursor = end
    pieces.append(text[cursor:])
    curated = normalize_whitespace(" ".join(pieces))
    curated = re.sub(r"\s+([,.!?…;:])", r"\1", curated)
    curated = re.sub(r"([,.!?…;:])\1+", r"\1", curated)
    return curated.strip(" ,;|"), transformations


def collapse_exact_repeated_clauses(
    text: str,
    *,
    min_tokens: int = 4,
    min_characters: int = 20,
) -> tuple[str, list[dict[str, Any]]]:
    seen: set[str] = set()
    duplicate_ranges: list[tuple[int, int]] = []
    duplicate_keys: list[str] = []
    for clause in extract_clauses(
        text,
        min_tokens=min_tokens,
        min_characters=min_characters,
    ):
        if not clause.key:
            continue
        if clause.key in seen:
            duplicate_ranges.append((clause.start, clause.end))
            duplicate_keys.append(clause.key)
        else:
            seen.add(clause.key)
    if not duplicate_ranges:
        return text, []
    curated, transformations = remove_ranges(text, duplicate_ranges)
    key_hashes = [sha256_text(key) for key in duplicate_keys]
    for item in transformations:
        item["type"] = "INTERNAL_CLAUSE_REPEAT"
        item["clause_key_sha256s"] = key_hashes
    return curated, transformations


def signature_matches(
    text: str,
    signatures: dict[str, Sequence[str]],
) -> list[str]:
    folded = punctuation_symbol_key(accent_fold(text))
    matches = []
    for reason, phrases in signatures.items():
        if any(
            punctuation_symbol_key(accent_fold(phrase)) in folded
            for phrase in phrases
        ):
            matches.append(reason)
    return sorted(matches)


def reward_clause_kind(
    clause: Clause,
    reward_signatures: Sequence[str],
) -> str | None:
    folded = punctuation_symbol_key(accent_fold(clause.content))
    matched = any(
        punctuation_symbol_key(accent_fold(signature)) in folded
        for signature in reward_signatures
    )
    if not matched:
        return None
    residual = [
        token for token in folded.split() if token not in REWARD_ALLOWED_TOKENS
    ]
    return "pure" if len(residual) <= 2 else "mixed"


def remove_pure_reward_clauses(
    text: str,
    reward_signatures: Sequence[str],
    *,
    min_tokens: int = 2,
    min_characters: int = 8,
) -> tuple[str, list[dict[str, Any]], bool]:
    ranges = []
    mixed = False
    keys = []
    for clause in extract_clauses(
        text,
        min_tokens=min_tokens,
        min_characters=min_characters,
    ):
        kind = reward_clause_kind(clause, reward_signatures)
        if kind == "pure":
            ranges.append((clause.start, clause.end))
            keys.append(clause.key or word_key(clause.content))
        elif kind == "mixed":
            mixed = True
    if not ranges:
        return text, [], mixed
    curated, transformations = remove_ranges(text, ranges)
    key_hashes = [sha256_text(key) for key in keys]
    for item in transformations:
        item["type"] = "REWARD_DISCLAIMER_REMOVED"
        item["clause_key_sha256s"] = key_hashes
    return curated, transformations, mixed


def redact_sensitive_spans(
    text: str,
    *,
    phone_pattern: str,
    email_pattern: str,
    url_pattern: str,
) -> tuple[str, list[dict[str, Any]]]:
    matches: list[tuple[int, int, str]] = []
    for kind, pattern in (
        ("EMAIL_REDACTED", email_pattern),
        ("PHONE_REDACTED", phone_pattern),
        ("URL_REDACTED", url_pattern),
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((match.start(), match.end(), kind))
    selected: list[tuple[int, int, str]] = []
    for start, end, kind in sorted(matches, key=lambda item: (item[0], -item[1])):
        if any(start < old_end and end > old_start for old_start, old_end, _ in selected):
            continue
        selected.append((start, end, kind))
    if not selected:
        return text, []
    pieces: list[str] = []
    transformations = []
    cursor = 0
    for start, end, kind in sorted(selected):
        pieces.append(text[cursor:start])
        replacement = {
            "EMAIL_REDACTED": "[EMAIL]",
            "PHONE_REDACTED": "[PHONE]",
            "URL_REDACTED": "[URL]",
        }[kind]
        pieces.append(replacement)
        transformations.append(
            {
                "type": kind,
                "start": start,
                "end": end,
                "removed_sha256": sha256_text(text[start:end]),
                "removed_characters": end - start,
                "replacement": replacement,
            }
        )
        cursor = end
    pieces.append(text[cursor:])
    return normalize_whitespace("".join(pieces)), transformations


def internal_ngram_repetition_evidence(
    text: str,
    *,
    ngram_size: int = 4,
    minimum_repeated_ngrams: int = 3,
    minimum_later_token_coverage: float = 0.15,
) -> dict[str, Any]:
    """Measure non-overlapping repeated token blocks inside one review.

    Exact clause collapse handles identical comma/sentence units.  This
    secondary signal catches a longer sequence that was pasted twice with a
    changed boundary or small punctuation difference.  It is evidence for
    quarantine only; callers must not delete text from this signal.
    """
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least 2")
    if minimum_repeated_ngrams < 1:
        raise ValueError("minimum_repeated_ngrams must be positive")
    if not 0 <= minimum_later_token_coverage <= 1:
        raise ValueError("minimum_later_token_coverage must be in [0, 1]")

    tokens = word_tokens(text)
    positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(max(0, len(tokens) - ngram_size + 1)):
        positions[tuple(tokens[index : index + ngram_size])].append(index)

    repeated: list[tuple[tuple[str, ...], int]] = []
    later_token_indices: set[int] = set()
    for ngram, starts in sorted(positions.items()):
        first = starts[0]
        later = next(
            (
                start
                for start in starts[1:]
                if start - first >= ngram_size
            ),
            None,
        )
        if later is None:
            continue
        repeated.append((ngram, later))
        later_token_indices.update(range(later, later + ngram_size))

    coverage = (
        len(later_token_indices) / len(tokens)
        if tokens
        else 0.0
    )
    flagged = (
        len(repeated) >= minimum_repeated_ngrams
        and coverage >= minimum_later_token_coverage
    )
    return {
        "flagged": flagged,
        "ngram_size": ngram_size,
        "repeated_ngram_count": len(repeated),
        "later_token_count": len(later_token_indices),
        "later_token_coverage": round(coverage, 6),
        "repeated_ngram_hashes": [
            sha256_text(" ".join(ngram))
            for ngram, _ in repeated
        ],
    }


def _collapsed_character_runs(token: str) -> str:
    return re.sub(r"([a-z])\1+", r"\1", token)


def _is_short_periodic_token(token: str) -> bool:
    for period in range(1, min(4, len(token) // 3) + 1):
        if len(token) % period:
            continue
        unit = token[:period]
        if unit * (len(token) // period) == token:
            return True
    return False


def _maximum_consonant_run(token: str) -> int:
    chunks = re.findall(r"[bcdfghjklmnpqrstvwxz]+", token)
    return max((len(chunk) for chunk in chunks), default=0)


def keyboard_token_document_frequencies(
    texts: Iterable[str],
) -> Counter[str]:
    """Count normalized token roots once per document for smash detection."""
    frequencies: Counter[str] = Counter()
    for text in texts:
        roots = set()
        for token in word_tokens(accent_fold(text)):
            if token.isascii() and token.isalpha():
                roots.add(_collapsed_character_runs(token))
        frequencies.update(roots)
    return frequencies


def terminal_consonant_junk_evidence(
    text: str,
    *,
    token_document_frequency: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    """Detect a rare terminal ASCII token with no vowel.

    This narrow detector exists for short terminal junk such as
    ``bxbzbbz`` that is below the general keyboard-smash length threshold.
    It returns evidence only; the caller must still verify residual quality
    and veto query/SKU tokens before applying a reversible transformation.
    """
    value = str(text)
    matches = list(WORD_RE.finditer(value))
    if not matches:
        return None
    match = matches[-1]
    token = match.group(0)
    if not (
        7 <= len(token) <= 12
        and token.isascii()
        and token.isalpha()
        and token.islower()
    ):
        return None
    suffix = value[match.end():]
    if not re.fullmatch(r"[.!?…;,]*\s*", suffix):
        return None
    if match.start() <= 0 or not value[match.start() - 1].isspace():
        return None
    residual = value[:match.start()].rstrip()
    if not residual or residual[-1] in ":/-":
        return None
    folded = accent_fold(token)
    if any(character in "aeiouy" for character in folded):
        return None
    root = _collapsed_character_runs(folded)
    document_frequency = token_document_frequency or {}
    root_df = int(document_frequency.get(root, 0))
    if root_df > 2:
        return None
    return {
        "start": match.start(),
        "token_end": match.end(),
        "end": len(value),
        "token_sha256": sha256_text(token),
        "suffix_sha256": sha256_text(value[match.start():]),
        "token_length": len(token),
        "root_document_frequency": root_df,
        "character_diversity": len(set(folded)),
        "maximum_consonant_run": _maximum_consonant_run(folded),
    }


def keyboard_smash_evidence(
    text: str,
    *,
    token_document_frequency: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return conservative, span-preserving keyboard-smash evidence.

    A single medium-strength token is never sufficient.  Common token roots,
    short periodic strings (for example ``okokok``), and natural expressive
    elongations are explicitly protected from this detector.
    """
    document_frequency = token_document_frequency or {}
    strong_tokens: list[dict[str, Any]] = []
    medium_tokens: list[dict[str, Any]] = []
    for match in WORD_RE.finditer(str(text)):
        original_token = match.group(0)
        token = accent_fold(original_token)
        if not token.isascii() or not token.isalpha() or len(token) < 8:
            continue
        root = _collapsed_character_runs(token)
        root_df = int(document_frequency.get(root, 0))
        is_periodic = _is_short_periodic_token(token)
        is_elongated = (
            len(token) - len(root) >= 2
            and (
                root in NATURAL_ELONGATION_ROOTS
                or root_df >= 5
            )
        )
        if is_periodic or is_elongated:
            continue

        length = len(token)
        rare_letters = sum(character in "fjqwz" for character in token)
        vowels = sum(character in "aeiouy" for character in token)
        consonants = length - vowels
        consonant_ratio = consonants / length
        max_consonant_run = _maximum_consonant_run(token)
        diversity = len(set(token))
        strong = (
            (
                length >= 13
                and (
                    (
                        rare_letters >= 3
                        and rare_letters / length >= 0.16
                    )
                    or (
                        consonant_ratio >= 0.76
                        and vowels <= 4
                    )
                    or max_consonant_run >= 8
                )
            )
            or (
                length >= 24
                and rare_letters >= 2
                and diversity >= 7
            )
        )
        medium = (
            not strong
            and length >= 8
            and root_df <= 2
            and (
                (
                    rare_letters >= 3
                    and rare_letters / length >= 0.18
                )
                or (
                    consonant_ratio >= 0.76
                    and vowels <= 3
                )
                or max_consonant_run >= 7
            )
        )
        if not strong and not medium:
            continue
        evidence = {
            "token_sha256": sha256_text(original_token),
            "start": match.start(),
            "end": match.end(),
            "length": length,
            "root_document_frequency": root_df,
            "rare_letter_count": rare_letters,
            "consonant_ratio": round(consonant_ratio, 6),
            "maximum_consonant_run": max_consonant_run,
            "character_diversity": diversity,
        }
        if strong:
            strong_tokens.append(evidence)
        else:
            medium_tokens.append(evidence)

    folded_tokens = word_tokens(accent_fold(text))
    one_letter_tokens = sum(
        len(token) == 1 and token.isalpha()
        for token in folded_tokens
    )
    one_letter_flagged = (
        len(folded_tokens) >= 20
        and one_letter_tokens >= 8
        and one_letter_tokens / len(folded_tokens) >= 0.20
    )
    return {
        "flagged": (
            bool(strong_tokens)
            or len(medium_tokens) >= 2
            or one_letter_flagged
        ),
        "strong_tokens": strong_tokens,
        "medium_tokens": medium_tokens,
        "one_letter_token_count": one_letter_tokens,
        "one_letter_flagged": one_letter_flagged,
    }


def language_artifact_flags(
    text: str,
    *,
    token_document_frequency: Mapping[str, int] | None = None,
) -> list[str]:
    tokens = word_tokens(text)
    if not tokens:
        return ["EMPTY_AFTER_CLEANING"]
    flags = []
    english_hits = sum(token in ENGLISH_FUNCTION_WORDS for token in tokens)
    diacritic_hits = sum(
        any(character in VIETNAMESE_DIACRITICS for character in token)
        for token in tokens
    )
    if (
        len(tokens) >= 15
        and english_hits >= 8
        and english_hits / len(tokens) >= 0.35
        and diacritic_hits / len(tokens) < 0.1
    ):
        flags.append("LANGUAGE_ENGLISH_DOMINANT")
    tagalog_hits = [
        token
        for token in tokens
        if token in TAGALOG_FUNCTION_WORDS
    ]
    if (
        len(tokens) >= 15
        and len(tagalog_hits) >= 5
        and len(set(tagalog_hits)) >= 4
        and len(tagalog_hits) / len(tokens) >= 0.15
    ):
        flags.append("LANGUAGE_TAGALOG_DOMINANT")
    if re.search(r"\\\s*[nrt]", text, flags=re.IGNORECASE):
        flags.append("LITERAL_ESCAPE_ARTIFACT")
    repeated_character_runs = 0
    for match in re.finditer(r"([A-Za-z0-9])\1{5,}", text):
        word_start = match.start()
        while word_start and text[word_start - 1].isalpha():
            word_start -= 1
        word_end = match.end()
        while word_end < len(text) and text[word_end].isalpha():
            word_end += 1
        token = accent_fold(text[word_start:word_end]).casefold()
        root = _collapsed_character_runs(token)
        root_df = int((token_document_frequency or {}).get(root, 0))
        is_natural_elongation = (
            len(token) - len(root) >= 2
            and (
                root in NATURAL_ELONGATION_ROOTS
                or root_df >= 5
            )
        )
        if not is_natural_elongation:
            repeated_character_runs += len(match.group(0))
    alphabetic = sum(character.isalpha() for character in text)
    if alphabetic and repeated_character_runs / alphabetic >= 0.35:
        flags.append("GIBBERISH_CHARACTER_RUNS")
    folded = punctuation_symbol_key(accent_fold(text))
    consonant_chunks = re.findall(
        r"\b[bcdfghjklmnpqrstvwxyz]{5,}\b",
        folded,
        flags=re.IGNORECASE,
    )
    if (
        any(len(chunk) >= 18 for chunk in consonant_chunks)
        or (
            len(consonant_chunks) >= 3
            and sum(len(chunk) for chunk in consonant_chunks) >= 18
        )
    ):
        flags.append("GIBBERISH_CONSONANT_RUNS")
    smash_evidence = keyboard_smash_evidence(
        text,
        token_document_frequency=token_document_frequency,
    )
    if smash_evidence["flagged"]:
        flags.append("GIBBERISH_KEYBOARD_SMASH")
    return flags


def build_clause_frequencies(
    texts: Sequence[str],
    product_ids: Sequence[str],
    *,
    min_tokens: int,
    min_characters: int,
) -> tuple[
    list[list[Clause]],
    Counter[str],
    dict[str, set[str]],
    dict[str, Counter[str]],
]:
    if len(texts) != len(product_ids):
        raise ValueError(
            "texts and product_ids must contain the same number of records"
        )
    parsed: list[list[Clause]] = []
    review_df: Counter[str] = Counter()
    product_sets: dict[str, set[str]] = defaultdict(set)
    product_review_df: dict[str, Counter[str]] = defaultdict(Counter)
    for text, product_id in zip(texts, product_ids):
        clauses = extract_clauses(
            text,
            min_tokens=min_tokens,
            min_characters=min_characters,
        )
        parsed.append(clauses)
        keys = {clause.key for clause in clauses if clause.key}
        review_df.update(keys)
        for key in keys:
            product_sets[key].add(product_id)
            product_review_df[product_id][key] += 1
    return parsed, review_df, product_sets, product_review_df


def template_evidence(
    text: str,
    clauses: Sequence[Clause],
    *,
    product_id: str,
    review_df: Counter[str],
    product_sets: dict[str, set[str]],
    product_review_df: dict[str, Counter[str]],
    global_review_df: int,
    global_product_df: int,
    product_df: int,
) -> dict[str, Any]:
    total_tokens = max(1, len(word_tokens(text)))
    global_keys = sorted(
        {
            clause.key
            for clause in clauses
            if clause.key
            and review_df[clause.key] >= global_review_df
            and len(product_sets[clause.key]) >= global_product_df
        }
    )
    local_keys = sorted(
        {
            clause.key
            for clause in clauses
            if clause.key
            and product_review_df[product_id][clause.key] >= product_df
        }
    )
    global_coverage = sum(
        len(clause.tokens)
        for clause in clauses
        if clause.key in set(global_keys)
    ) / total_tokens
    local_coverage = sum(
        len(clause.tokens)
        for clause in clauses
        if clause.key in set(local_keys)
    ) / total_tokens
    family_keys = sorted(set(global_keys) | set(local_keys))
    family_id = (
        "tpl-"
        + sha256_text("\0".join(family_keys))[:20]
        if family_keys
        else None
    )
    return {
        "global_recurrent_clause_count": len(global_keys),
        "global_template_coverage": round(global_coverage, 6),
        "product_recurrent_clause_count": len(local_keys),
        "product_template_coverage": round(local_coverage, 6),
        "template_family_id": family_id,
        "global_clause_hashes": [
            sha256_text(key) for key in global_keys
        ],
        "product_clause_hashes": [
            sha256_text(key) for key in local_keys
        ],
    }


def find_allpairs(
    feature_sets: Sequence[frozenset[Any]],
    *,
    threshold: float,
    stable_ids: Sequence[str] | None = None,
) -> list[SimilarPair]:
    """Complete AllPairs prefix join with exact Jaccard verification."""
    ratio = Fraction(str(threshold)).limit_denominator(1_000_000)
    numerator = ratio.numerator
    denominator = ratio.denominator
    document_frequency: Counter[Any] = Counter()
    for values in feature_sets:
        document_frequency.update(values)
    ordered_values = [
        sorted(values, key=lambda value: (document_frequency[value], value))
        for values in feature_sets
    ]
    if stable_ids is None:
        stable_ids = [str(index) for index in range(len(feature_sets))]
    processing_order = sorted(
        range(len(feature_sets)),
        key=lambda index: (
            len(feature_sets[index]),
            stable_ids[index],
        ),
    )
    prefix_index: dict[Any, list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for current in processing_order:
        size = len(feature_sets[current])
        if not size:
            continue
        required_overlap = (
            numerator * size + denominator - 1
        ) // denominator
        prefix_length = size - required_overlap + 1
        for value in ordered_values[current][:prefix_length]:
            for previous in prefix_index[value]:
                previous_size = len(feature_sets[previous])
                if denominator * previous_size < numerator * size:
                    continue
                candidates.add(
                    (min(previous, current), max(previous, current))
                )
        for value in ordered_values[current][:prefix_length]:
            prefix_index[value].append(current)
    matches = []
    for left, right in sorted(candidates):
        intersection = len(feature_sets[left] & feature_sets[right])
        union = len(feature_sets[left] | feature_sets[right])
        if union and denominator * intersection >= numerator * union:
            matches.append(
                SimilarPair(
                    left=left,
                    right=right,
                    intersection=intersection,
                    union=union,
                )
            )
    return matches


def connected_components(
    size: int,
    pairs: Iterable[tuple[int, int]],
) -> list[list[int]]:
    union_find = UnionFind(size)
    involved = set()
    for left, right in pairs:
        union_find.union(left, right)
        involved.add(left)
        involved.add(right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in sorted(involved):
        groups[union_find.find(index)].append(index)
    return sorted(groups.values(), key=lambda values: (len(values), values))


def text_artifact_penalty(text: str) -> tuple[int, int, int, int, int]:
    """Rank obvious extraction/junk artifacts without interpreting sentiment."""
    normalized = normalize_whitespace(text)
    if not normalized:
        return (1, 1, 1_000_000, 1_000_000, 1_000_000)
    leading_junk = int(
        bool(re.match(r"^[^\wÀ-ỹĐđ]+(?=\w)", normalized, flags=re.UNICODE))
    )
    trailing_junk = int(
        bool(
            re.search(
                r"(?:[^\wÀ-ỹĐđ\s]{2,}|(?:[.,;:/\\-]\s*){3,})$",
                normalized,
                flags=re.UNICODE,
            )
        )
    )
    repeated_run_characters = sum(
        max(0, len(match.group(0)) - 3)
        for match in re.finditer(r"(.)\1{3,}", normalized, flags=re.DOTALL)
    )
    punctuation_anomalies = len(
        re.findall(r"(?:[!?][,;]|[,;][!?])", normalized)
    ) + int(bool(re.search(r"[,;]\s*$", normalized)))
    nonsemantic_characters = sum(
        not character.isalnum()
        and not character.isspace()
        and character not in ".,!?…;:'\"()-/%+&"
        for character in normalized
    )
    scaled_nonsemantic_ratio = round(
        1_000_000 * nonsemantic_characters / max(1, len(normalized))
    )
    return (
        leading_junk,
        trailing_junk,
        punctuation_anomalies,
        repeated_run_characters,
        scaled_nonsemantic_ratio,
    )


def choose_representative(
    rows: Sequence[dict[str, Any]],
    indices: Sequence[int],
    *,
    prefer_api_identity: bool = True,
) -> int:
    def rank(index: int) -> tuple[Any, ...]:
        row = rows[index]
        transport = str(row.get("collection_transport") or "")
        review_id = str(row.get("review_id") or "")
        api_identity = transport in {"requests", "requests_cookie"} and review_id.isdigit()
        comparison_text = str(
            row.get("curated_review_text")
            or row.get("review_text")
            or ""
        )
        metadata = sum(
            bool(row.get(key))
            for key in (
                "review_time",
                "sku_info",
                "verified_purchase",
                "seller_id",
                "source_url",
            )
        )
        return (
            not api_identity if prefer_api_identity else False,
            text_artifact_penalty(comparison_text),
            -metadata,
            -float(row.get("quality_score") or 0.0),
            str(row.get("collected_at") or ""),
            str(row.get("sample_id") or ""),
        )

    return min(indices, key=rank)


def deterministic_stratified_sample(
    rows: Sequence[dict[str, Any]],
    *,
    size: int,
    seed: str,
    keys: Sequence[str],
) -> list[int]:
    if size <= 0:
        return []
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = tuple(str(row.get(key) or "<blank>") for key in keys)
        groups[group].append(index)
    for indices in groups.values():
        indices.sort(
            key=lambda index: stable_rank(
                seed,
                str(rows[index].get("sample_id") or index),
            )
        )
    selected: list[int] = []
    ordered_groups = sorted(groups)
    cursor = {group: 0 for group in ordered_groups}
    while len(selected) < min(size, len(rows)):
        progressed = False
        for group in ordered_groups:
            position = cursor[group]
            if position >= len(groups[group]):
                continue
            selected.append(groups[group][position])
            cursor[group] += 1
            progressed = True
            if len(selected) >= min(size, len(rows)):
                break
        if not progressed:
            break
    return selected


def ratio_meets(
    intersection: int,
    union: int,
    threshold: float,
) -> bool:
    if not union:
        return False
    ratio = Fraction(str(threshold)).limit_denominator(1_000_000)
    return ratio.denominator * intersection >= ratio.numerator * union
