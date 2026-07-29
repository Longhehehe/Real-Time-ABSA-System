"""Strict contract for AI suggestions shown in human-verification mode."""

from __future__ import annotations

import json
from typing import Any, Mapping

from human_annotation_ui.common import canonical_json, sha256_text
from human_annotation_ui.validate_export import (
    annotation_to_validator_payload,
    reject_forbidden_keys,
)
from lazada_collector.llm_annotation import (
    AnnotationValidationError,
    validate_and_normalize_annotation,
)


SUGGESTION_SCHEMA_VERSION = "human-absa-ai-suggestions/1.0.0"
ENVELOPE_FIELDS = {
    "schema_version",
    "payload",
    "payload_sha256",
}
PAYLOAD_FIELDS = {
    "suggestion_set_id",
    "assignment_id",
    "reference_id",
    "assignment_payload_sha256",
    "guideline_version",
    "guideline_sha256",
    "item_count",
    "created_at",
    "source_method",
    "records",
}
RECORD_FIELDS = {
    "annotation_id",
    "review_text_sha256",
    "annotation_status",
    "aspects",
    "review_uncertainty_codes",
    "notes",
}
ANNOTATION_FIELDS = {
    "annotation_status",
    "aspects",
    "review_uncertainty_codes",
    "notes",
}


class SuggestionIntegrityError(ValueError):
    """Raised when suggestions cannot be safely paired with an assignment."""


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise SuggestionIntegrityError(f"{context} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise SuggestionIntegrityError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def verify_suggestions(
    value: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_keys(value, ENVELOPE_FIELDS, context="Suggestion envelope")
    if value.get("schema_version") != SUGGESTION_SCHEMA_VERSION:
        raise SuggestionIntegrityError("Suggestion schema version mismatch")
    payload = value.get("payload")
    require_exact_keys(payload, PAYLOAD_FIELDS, context="Suggestion payload")
    reject_forbidden_keys(payload)
    if value.get("payload_sha256") != sha256_text(canonical_json(payload)):
        raise SuggestionIntegrityError("Suggestion payload checksum mismatch")
    expected_pairs = {
        "assignment_id": assignment["assignment_id"],
        "reference_id": assignment["reference_id"],
        "assignment_payload_sha256": assignment[
            "assignment_payload_sha256"
        ],
        "guideline_version": assignment["guideline"]["version"],
        "guideline_sha256": assignment["guideline"]["sha256"],
        "item_count": assignment["item_count"],
    }
    for field, expected in expected_pairs.items():
        if payload.get(field) != expected:
            raise SuggestionIntegrityError(
                f"Suggestion/assignment mismatch: {field}"
            )
    for field in ("suggestion_set_id", "created_at", "source_method"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise SuggestionIntegrityError(
                f"Suggestion field {field} must be a non-empty string"
            )
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or len(records) != assignment["item_count"]
    ):
        raise SuggestionIntegrityError("Suggestion records count mismatch")

    for index, (record, item) in enumerate(
        zip(records, assignment["records"], strict=True),
        1,
    ):
        require_exact_keys(
            record,
            RECORD_FIELDS,
            context=f"Suggestion record {index}",
        )
        if (
            record.get("annotation_id") != item["annotation_id"]
            or record.get("review_text_sha256")
            != item["review_text_sha256"]
        ):
            raise SuggestionIntegrityError(
                f"Suggestion identity/order mismatch at record {index}"
            )
        annotation = {
            key: record[key]
            for key in ANNOTATION_FIELDS
        }
        try:
            validator_payload = annotation_to_validator_payload(
                item["reviewContent"],
                annotation,
                context=f"Suggestion record {index}",
            )
            validate_and_normalize_annotation(
                item["reviewContent"],
                validator_payload,
            )
        except (AnnotationValidationError, ValueError) as exc:
            raise SuggestionIntegrityError(
                f"Invalid suggestion at record {index}: {exc}"
            ) from exc

    return json.loads(canonical_json(value))
