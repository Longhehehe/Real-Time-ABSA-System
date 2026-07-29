"""Shared, dependency-free integrity helpers for the annotation workbench."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from human_annotation_ui import (
    ASSIGNMENT_SCHEMA_VERSION,
    EXPORT_SCHEMA_VERSION,
    LEGACY_EXPORT_SCHEMA_VERSION,
)


ASSIGNMENT_FIELDS = {
    "schema_version",
    "assignment_id",
    "reference_id",
    "role",
    "item_count",
    "guideline",
    "created_at",
    "records",
    "assignment_payload_sha256",
}
ASSIGNMENT_PAYLOAD_FIELDS = ASSIGNMENT_FIELDS - {
    "assignment_payload_sha256"
}
ASSIGNMENT_RECORD_FIELDS = {
    "annotation_id",
    "reviewContent",
    "review_text_sha256",
}
GUIDELINE_FIELDS = {
    "document_id",
    "version",
    "sha256",
}
EXPORT_ENVELOPE_FIELDS = {
    "schema_version",
    "payload",
    "payload_sha256",
}


class IntegrityError(ValueError):
    """Raised when an assignment/export is malformed or has been changed."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Không đọc được JSON hợp lệ: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON gốc phải là object: {path}")
    return value


def verify_assignment(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError("Assignment phải là một JSON object")
    if set(value) != ASSIGNMENT_FIELDS:
        raise IntegrityError(
            "Assignment có field thiếu/thừa: "
            f"{sorted(set(value) ^ ASSIGNMENT_FIELDS)}"
        )
    if value.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION:
        raise IntegrityError("Sai assignment schema version")
    role = value.get("role")
    if role not in {"A", "B", "REVIEWER", "ADJUDICATOR"}:
        raise IntegrityError(
            "Assignment role phải là A, B, REVIEWER hoặc ADJUDICATOR"
        )
    for field in ("assignment_id", "reference_id", "created_at"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise IntegrityError(f"Assignment field {field} không hợp lệ")
    guideline = value.get("guideline")
    if not isinstance(guideline, Mapping) or set(guideline) != GUIDELINE_FIELDS:
        raise IntegrityError("Guideline descriptor không hợp lệ")
    if (
        guideline.get("document_id") != "ABSA-ANNOTATION-GUIDELINE-V2"
        or guideline.get("version") != "2.0.0"
        or not isinstance(guideline.get("sha256"), str)
        or len(guideline["sha256"]) != 64
    ):
        raise IntegrityError("Guideline version/hash không hợp lệ")
    records = value.get("records")
    item_count = value.get("item_count")
    if (
        not isinstance(item_count, int)
        or isinstance(item_count, bool)
        or item_count <= 0
        or not isinstance(records, list)
        or len(records) != item_count
    ):
        raise IntegrityError("item_count không khớp records")

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise IntegrityError(f"Record {index} không phải object")
        if set(record) != ASSIGNMENT_RECORD_FIELDS:
            raise IntegrityError(
                f"Record {index} có field cấm hoặc field thiếu/thừa"
            )
        annotation_id = record.get("annotation_id")
        review = record.get("reviewContent")
        review_hash = record.get("review_text_sha256")
        if not isinstance(annotation_id, str) or not annotation_id:
            raise IntegrityError(f"Record {index} thiếu annotation_id")
        if annotation_id in seen_ids:
            raise IntegrityError(f"Trùng annotation_id: {annotation_id}")
        if not isinstance(review, str) or not review:
            raise IntegrityError(f"Record {index} có review rỗng")
        if review_hash != sha256_text(review):
            raise IntegrityError(
                f"Record {index} sai review_text_sha256"
            )
        if review_hash in seen_hashes:
            raise IntegrityError(
                f"Assignment có review text trùng tại record {index}"
            )
        seen_ids.add(annotation_id)
        seen_hashes.add(review_hash)

    payload = {key: value[key] for key in ASSIGNMENT_PAYLOAD_FIELDS}
    expected_hash = sha256_text(canonical_json(payload))
    if value.get("assignment_payload_sha256") != expected_hash:
        raise IntegrityError("Assignment payload checksum không khớp")
    return json.loads(canonical_json(value))


def verify_export_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != EXPORT_ENVELOPE_FIELDS:
        raise IntegrityError("Export envelope có field thiếu/thừa")
    if value.get("schema_version") not in {
        EXPORT_SCHEMA_VERSION,
        LEGACY_EXPORT_SCHEMA_VERSION,
    }:
        raise IntegrityError("Sai export schema version")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise IntegrityError("Export payload phải là object")
    expected = sha256_text(canonical_json(payload))
    if value.get("payload_sha256") != expected:
        raise IntegrityError("Export payload checksum không khớp")
    return json.loads(canonical_json(value))
