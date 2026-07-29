"""Strict contract for A/B comparisons shown in adjudication mode."""

from __future__ import annotations

import json
from typing import Any, Mapping

from human_annotation_ui.common import canonical_json, sha256_text
from human_annotation_ui.validate_export import (
    annotation_to_validator_payload,
)
from lazada_collector.llm_annotation import (
    AnnotationValidationError,
    validate_and_normalize_annotation,
)


ADJUDICATION_SCHEMA_VERSION = "human-absa-adjudication-input/1.0.0"
ENVELOPE_FIELDS = {
    "schema_version",
    "payload",
    "payload_sha256",
}
PAYLOAD_FIELDS = {
    "adjudication_set_id",
    "assignment_id",
    "reference_id",
    "assignment_payload_sha256",
    "guideline_version",
    "guideline_sha256",
    "item_count",
    "created_at",
    "source_a_export_payload_sha256",
    "source_b_export_payload_sha256",
    "source_a_annotator_id",
    "source_b_annotator_id",
    "iaa_report_sha256",
    "records",
}
RECORD_FIELDS = {
    "annotation_id",
    "review_text_sha256",
    "source_a",
    "source_b",
    "disagreement_fields",
}
ANNOTATION_FIELDS = {
    "annotation_status",
    "aspects",
    "review_uncertainty_codes",
    "notes",
}


class AdjudicationIntegrityError(ValueError):
    """Raised when an adjudication input cannot be trusted."""


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise AdjudicationIntegrityError(f"{context} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise AdjudicationIntegrityError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def validate_annotation(
    review_text: str,
    annotation: Mapping[str, Any],
    *,
    context: str,
) -> None:
    require_exact_keys(annotation, ANNOTATION_FIELDS, context=context)
    try:
        payload = annotation_to_validator_payload(
            review_text,
            annotation,
            context=context,
        )
        validate_and_normalize_annotation(review_text, payload)
    except (AnnotationValidationError, ValueError) as exc:
        raise AdjudicationIntegrityError(
            f"Invalid annotation at {context}: {exc}"
        ) from exc


def verify_adjudication(
    value: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_keys(value, ENVELOPE_FIELDS, context="Adjudication envelope")
    if value.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
        raise AdjudicationIntegrityError(
            "Adjudication schema version mismatch"
        )
    payload = value.get("payload")
    require_exact_keys(payload, PAYLOAD_FIELDS, context="Adjudication payload")
    if value.get("payload_sha256") != sha256_text(canonical_json(payload)):
        raise AdjudicationIntegrityError(
            "Adjudication payload checksum mismatch"
        )
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
            raise AdjudicationIntegrityError(
                f"Adjudication/assignment mismatch: {field}"
            )
    for field in (
        "adjudication_set_id",
        "created_at",
        "source_a_annotator_id",
        "source_b_annotator_id",
    ):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise AdjudicationIntegrityError(
                f"Adjudication field {field} must be a non-empty string"
            )
    for field in (
        "source_a_export_payload_sha256",
        "source_b_export_payload_sha256",
        "iaa_report_sha256",
    ):
        if (
            not isinstance(payload.get(field), str)
            or len(payload[field]) != 64
        ):
            raise AdjudicationIntegrityError(
                f"Adjudication field {field} must be SHA-256"
            )
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or len(records) != assignment["item_count"]
    ):
        raise AdjudicationIntegrityError(
            "Adjudication records count mismatch"
        )
    for index, (record, item) in enumerate(
        zip(records, assignment["records"], strict=True),
        1,
    ):
        require_exact_keys(
            record,
            RECORD_FIELDS,
            context=f"Adjudication record {index}",
        )
        if (
            record.get("annotation_id") != item["annotation_id"]
            or record.get("review_text_sha256")
            != item["review_text_sha256"]
            or not isinstance(record.get("disagreement_fields"), list)
            or not record["disagreement_fields"]
            or not all(
                isinstance(field, str) and field
                for field in record["disagreement_fields"]
            )
            or len(set(record["disagreement_fields"]))
            != len(record["disagreement_fields"])
        ):
            raise AdjudicationIntegrityError(
                f"Adjudication identity/disagreement mismatch at {index}"
            )
        validate_annotation(
            item["reviewContent"],
            record["source_a"],
            context=f"Adjudication record {index}.source_a",
        )
        validate_annotation(
            item["reviewContent"],
            record["source_b"],
            context=f"Adjudication record {index}.source_b",
        )
        if canonical_json(record["source_a"]) == canonical_json(
            record["source_b"]
        ):
            raise AdjudicationIntegrityError(
                f"Adjudication record {index} has no actual disagreement"
            )
    return json.loads(canonical_json(value))
