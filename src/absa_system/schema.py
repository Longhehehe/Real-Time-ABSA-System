"""Strict schemas for the active multi-polarity ABSA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import hashlib


ASPECTS: tuple[str, ...] = (
    "Chất lượng sản phẩm",
    "Hiệu năng & Trải nghiệm",
    "Đúng mô tả",
    "Giá cả & Khuyến mãi",
    "Vận chuyển",
    "Đóng gói",
    "Dịch vụ & Thái độ Shop",
    "Bảo hành & Đổi trả",
    "Tính xác thực",
)

POLARITIES: tuple[str, ...] = ("negative", "positive", "neutral")
POLARITY_TO_INDEX = {name: index for index, name in enumerate(POLARITIES)}
TRAINABLE_STATUSES = frozenset({"LABELED", "ESCALATE"})


class AnnotationSchemaError(ValueError):
    """Raised when an annotation violates the canonical ABSA contract."""


@dataclass(frozen=True)
class EvidenceSpan:
    aspect_index: int
    polarity_index: int
    start: int
    end: int
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "aspect_index": self.aspect_index,
            "polarity_index": self.polarity_index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class ABSAExample:
    sample_id: str
    text: str
    review_text_sha256: str
    annotation_status: str
    mention_labels: tuple[int, ...]
    sentiment_labels: tuple[tuple[int, int, int], ...]
    evidence: tuple[EvidenceSpan, ...]
    source_package: str
    source_domain: str
    source_metadata: Mapping[str, Any]
    label_provenance: str

    @property
    def flattened_sentiment_labels(self) -> tuple[int, ...]:
        return tuple(value for row in self.sentiment_labels for value in row)

    def as_model_record(self, *, split: str, leakage_group_id: str) -> dict[str, Any]:
        return {
            "schema_version": "absa-model-example/1.0.0",
            "sample_id": self.sample_id,
            "reviewContent": self.text,
            "review_text_sha256": self.review_text_sha256,
            "leakage_group_id": leakage_group_id,
            "split": split,
            "annotation_status": self.annotation_status,
            "label_provenance": self.label_provenance,
            "mention_labels": list(self.mention_labels),
            "sentiment_labels": [list(row) for row in self.sentiment_labels],
            "evidence": [span.as_dict() for span in self.evidence],
            "source": {
                "package": self.source_package,
                "domain": self.source_domain,
                **dict(self.source_metadata),
            },
        }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_mixed_label(value: Any) -> Any:
    if isinstance(value, str):
        compact = value.replace(" ", "")
        if compact in {"1,-1", "-1,1"}:
            return "mixed"
        if compact in {"-1", "0", "1", "2"}:
            return int(compact)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            values = {int(item) for item in value}
        except (TypeError, ValueError):
            return value
        if values == {-1, 1}:
            return "mixed"
    return value


def encode_aspect_label(value: Any) -> tuple[int, tuple[int, int, int]]:
    """Encode one five-state aspect label into mention and multi-hot polarity."""

    normalized = _normalize_mixed_label(value)
    if normalized == 2:
        return 0, (0, 0, 0)
    if normalized == -1:
        return 1, (1, 0, 0)
    if normalized == 1:
        return 1, (0, 1, 0)
    if normalized == 0:
        return 1, (0, 0, 1)
    if normalized == "mixed":
        return 1, (1, 1, 0)
    raise AnnotationSchemaError(f"Unsupported aspect label: {value!r}")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationSchemaError(f"{field} must be a non-empty string")
    return value


def _parse_evidence(
    raw_evidence: Iterable[Mapping[str, Any]],
    *,
    text: str,
    aspect_index: int,
) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for position, raw in enumerate(raw_evidence):
        polarity = raw.get("polarity")
        if polarity not in POLARITY_TO_INDEX:
            raise AnnotationSchemaError(
                f"aspect {aspect_index} evidence {position}: invalid polarity {polarity!r}"
            )
        start = raw.get("start")
        end = raw.get("end")
        evidence_text = raw.get("text")
        if not isinstance(start, int) or not isinstance(end, int):
            raise AnnotationSchemaError(
                f"aspect {aspect_index} evidence {position}: offsets must be integers"
            )
        if start < 0 or end <= start or end > len(text):
            raise AnnotationSchemaError(
                f"aspect {aspect_index} evidence {position}: invalid span [{start}, {end})"
            )
        if evidence_text != text[start:end]:
            raise AnnotationSchemaError(
                f"aspect {aspect_index} evidence {position}: text/offset mismatch"
            )
        spans.append(
            EvidenceSpan(
                aspect_index=aspect_index,
                polarity_index=POLARITY_TO_INDEX[polarity],
                start=start,
                end=end,
                text=evidence_text,
            )
        )
    return spans


def parse_ai_annotation_record(
    row: Mapping[str, Any],
    *,
    source_package: str,
    source_domain: str,
) -> ABSAExample:
    """Parse one canonical AI annotation record and fail closed on label drift."""

    sample_id = _require_nonempty_string(row.get("sample_id"), "sample_id")
    text = _require_nonempty_string(row.get("reviewContent"), "reviewContent")
    text_hash = _require_nonempty_string(
        row.get("review_text_sha256"), "review_text_sha256"
    )
    if sha256_text(text) != text_hash:
        raise AnnotationSchemaError(f"{sample_id}: review_text_sha256 mismatch")

    annotation = row.get("annotation")
    if not isinstance(annotation, Mapping):
        raise AnnotationSchemaError(f"{sample_id}: annotation must be an object")
    status = annotation.get("annotation_status")
    if status not in {"LABELED", "ESCALATE", "REJECT_NON_REVIEW"}:
        raise AnnotationSchemaError(f"{sample_id}: invalid annotation status {status!r}")
    if status == "REJECT_NON_REVIEW":
        raise AnnotationSchemaError(
            f"{sample_id}: REJECT_NON_REVIEW has no trainable label vector"
        )

    raw_aspects = annotation.get("aspects")
    if not isinstance(raw_aspects, list) or len(raw_aspects) != len(ASPECTS):
        raise AnnotationSchemaError(
            f"{sample_id}: expected exactly {len(ASPECTS)} aspect rows"
        )
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw_aspect in raw_aspects:
        if not isinstance(raw_aspect, Mapping):
            raise AnnotationSchemaError(f"{sample_id}: aspect row must be an object")
        name = raw_aspect.get("aspect")
        if name in by_name:
            raise AnnotationSchemaError(f"{sample_id}: duplicate aspect {name!r}")
        by_name[name] = raw_aspect
    if set(by_name) != set(ASPECTS):
        missing = sorted(set(ASPECTS) - set(by_name))
        unknown = sorted(set(by_name) - set(ASPECTS))
        raise AnnotationSchemaError(
            f"{sample_id}: aspect taxonomy mismatch; missing={missing}, unknown={unknown}"
        )

    mentions: list[int] = []
    sentiments: list[tuple[int, int, int]] = []
    evidence: list[EvidenceSpan] = []
    for aspect_index, aspect_name in enumerate(ASPECTS):
        raw_aspect = by_name[aspect_name]
        mention, sentiment = encode_aspect_label(raw_aspect.get("label"))
        mentions.append(mention)
        sentiments.append(sentiment)
        raw_evidence = raw_aspect.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raise AnnotationSchemaError(
                f"{sample_id}: evidence for {aspect_name!r} must be a list"
            )
        evidence.extend(
            _parse_evidence(
                raw_evidence,
                text=text,
                aspect_index=aspect_index,
            )
        )
        if not mention and raw_evidence:
            raise AnnotationSchemaError(
                f"{sample_id}: absent aspect {aspect_name!r} has evidence"
            )

    source = row.get("source")
    source_metadata = dict(source) if isinstance(source, Mapping) else {}
    provenance = row.get("artifact_status") or "UNKNOWN_LABEL_PROVENANCE"
    return ABSAExample(
        sample_id=sample_id,
        text=text,
        review_text_sha256=text_hash,
        annotation_status=status,
        mention_labels=tuple(mentions),
        sentiment_labels=tuple(sentiments),
        evidence=tuple(evidence),
        source_package=source_package,
        source_domain=source_domain,
        source_metadata=source_metadata,
        label_provenance=str(provenance),
    )


def validate_model_record(row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != "absa-model-example/1.0.0":
        raise AnnotationSchemaError("invalid model example schema_version")
    _require_nonempty_string(row.get("sample_id"), "sample_id")
    text = _require_nonempty_string(row.get("reviewContent"), "reviewContent")
    if sha256_text(text) != row.get("review_text_sha256"):
        raise AnnotationSchemaError("model record review_text_sha256 mismatch")
    if row.get("split") not in {"train", "dev", "test"}:
        raise AnnotationSchemaError("model record has invalid split")
    _require_nonempty_string(row.get("leakage_group_id"), "leakage_group_id")
    mentions = row.get("mention_labels")
    sentiments = row.get("sentiment_labels")
    if not isinstance(mentions, list) or len(mentions) != len(ASPECTS):
        raise AnnotationSchemaError("mention_labels shape mismatch")
    if any(value not in {0, 1} for value in mentions):
        raise AnnotationSchemaError("mention_labels must be binary")
    if not isinstance(sentiments, list) or len(sentiments) != len(ASPECTS):
        raise AnnotationSchemaError("sentiment_labels aspect shape mismatch")
    for aspect_index, values in enumerate(sentiments):
        if (
            not isinstance(values, list)
            or len(values) != len(POLARITIES)
            or any(value not in {0, 1} for value in values)
        ):
            raise AnnotationSchemaError("sentiment_labels polarity shape mismatch")
        if not mentions[aspect_index] and any(values):
            raise AnnotationSchemaError("sentiment present for absent aspect")
        if values[2] and (values[0] or values[1]):
            raise AnnotationSchemaError("neutral cannot coexist with positive/negative")
