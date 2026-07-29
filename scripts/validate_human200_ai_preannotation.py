"""Fail-closed validation for the 200-record AI human-check package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from human_annotation_ui.ai_review import verify_suggestions
from human_annotation_ui.common import (
    canonical_json,
    load_json,
    sha256_file,
    verify_assignment,
)
from human_annotation_ui.validate_export import (
    annotation_to_validator_payload,
    validate_export,
)
from scripts.build_human200_ai_preannotation import (
    compare_calibration,
    distribution_rows,
    load_raw_batches,
    semantic_summary,
)
from lazada_collector.llm_annotation import (
    AnnotationValidationError,
    validate_and_normalize_annotation,
)


PREANNOTATION_ROW_FIELDS = {
    "assignment_position",
    "annotation_id",
    "review_text_sha256",
    "reference_id",
    "guideline_version",
    "annotation_method",
    "source_batch",
    "source_line",
    "annotation",
}


class PackageValidationError(ValueError):
    """Raised when a package artifact or invariant is invalid."""


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise PackageValidationError(f"{context} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise PackageValidationError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def validate_checksums(root: Path) -> dict[str, str]:
    checksum_path = root / "SHA256SUMS.txt"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PackageValidationError("Missing SHA256SUMS.txt") from exc
    expected: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if "  " not in line:
            raise PackageValidationError(
                f"Malformed checksum line {line_number}"
            )
        digest, relative = line.split("  ", 1)
        posix = PurePosixPath(relative)
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or posix.is_absolute()
            or ".." in posix.parts
            or relative in expected
        ):
            raise PackageValidationError(
                f"Unsafe/invalid checksum line {line_number}"
            )
        expected[relative] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(expected) != actual_files:
        raise PackageValidationError(
            "Checksum inventory mismatch; "
            f"missing={sorted(actual_files - set(expected))}, "
            f"extra={sorted(set(expected) - actual_files)}"
        )
    for relative, digest in expected.items():
        if sha256_file(root / Path(*PurePosixPath(relative).parts)) != digest:
            raise PackageValidationError(
                f"Checksum mismatch: {relative}"
            )
    return expected


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line:
            raise PackageValidationError(
                f"Blank JSONL line: {path.name}:{line_number}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PackageValidationError(
                f"Invalid JSONL: {path.name}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise PackageValidationError(
                f"JSONL row is not an object: {path.name}:{line_number}"
            )
        rows.append(value)
    return rows


def normalize_suggestion_annotation(
    review_text: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    annotation = {
        "annotation_status": record["annotation_status"],
        "aspects": record["aspects"],
        "review_uncertainty_codes": record[
            "review_uncertainty_codes"
        ],
        "notes": record["notes"],
    }
    payload = annotation_to_validator_payload(
        review_text,
        annotation,
        context=f"Suggestion {record['annotation_id']}",
    )
    try:
        return validate_and_normalize_annotation(review_text, payload)
    except AnnotationValidationError as exc:
        raise PackageValidationError(
            f"Invalid suggestion semantics: {record['annotation_id']}"
        ) from exc


def read_distribution(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                "aspect": row["aspect"],
                "label": row["label"],
                "count": int(row["count"]),
            }
            for row in csv.DictReader(handle)
        ]


def validate_package(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    checksum_inventory = validate_checksums(root)
    manifest = load_json(root / "manifest.json")
    if (
        manifest.get("schema_version")
        != "human-absa-ai-preannotation-package/1.0.0"
        or manifest.get("status")
        != "AI_PREANNOTATION_PENDING_HUMAN_VERIFICATION"
        or manifest.get("records") != 200
        or manifest.get("scientific_use", {}).get("is_human_gold")
        is not False
        or manifest.get("scientific_use", {}).get(
            "is_double_blind_human_annotation"
        )
        is not False
    ):
        raise PackageValidationError("Manifest scientific/status gate failed")

    review_assignment = verify_assignment(
        load_json(root / "human_check" / "ai_review.assignment.json")
    )
    suggestions = verify_suggestions(
        load_json(root / "human_check" / "ai_suggestions.json"),
        review_assignment,
    )
    if (
        review_assignment["assignment_id"]
        != manifest["review_assignment_id"]
        or suggestions["payload"]["suggestion_set_id"]
        != manifest["suggestion_set_id"]
    ):
        raise PackageValidationError("Manifest ID linkage mismatch")

    source_assignment = verify_assignment(
        load_json(
            root
            / "provenance"
            / "source_annotator_a.assignment.json"
        )
    )
    if source_assignment["reference_id"] != review_assignment["reference_id"]:
        raise PackageValidationError("Source/review reference mismatch")
    source_items = source_assignment["records"]
    preannotations = load_jsonl(root / "ai_preannotations.jsonl")
    if len(preannotations) != len(source_items):
        raise PackageValidationError("Preannotation count mismatch")
    suggestion_records = suggestions["payload"]["records"]
    normalized_rows: list[dict[str, Any]] = []
    for position, (row, item, suggestion) in enumerate(
        zip(preannotations, source_items, suggestion_records, strict=True),
        1,
    ):
        require_exact_keys(
            row,
            PREANNOTATION_ROW_FIELDS,
            context=f"Preannotation row {position}",
        )
        if (
            row["assignment_position"] != position
            or row["annotation_id"] != item["annotation_id"]
            or row["review_text_sha256"] != item["review_text_sha256"]
            or row["reference_id"] != source_assignment["reference_id"]
            or row["guideline_version"]
            != source_assignment["guideline"]["version"]
            or suggestion["annotation_id"] != row["annotation_id"]
        ):
            raise PackageValidationError(
                f"Preannotation identity mismatch at position {position}"
            )
        normalized = normalize_suggestion_annotation(
            item["reviewContent"],
            suggestion,
        )
        if canonical_json(normalized) != canonical_json(row["annotation"]):
            raise PackageValidationError(
                f"Suggestion/preannotation mismatch at position {position}"
            )
        normalized_rows.append(
            {
                "assignment_position": position,
                "annotation_id": row["annotation_id"],
                "review_text_sha256": row["review_text_sha256"],
                "source_batch": row["source_batch"],
                "source_line": row["source_line"],
                "annotation": normalized,
            }
        )

    raw_batch_paths = sorted(
        (root / "provenance" / "annotation_batches").glob("*.jsonl")
    )
    replayed = load_raw_batches(raw_batch_paths, source_assignment)
    for expected, actual in zip(
        normalized_rows,
        replayed,
        strict=True,
    ):
        if (
            expected["assignment_position"]
            != actual["assignment_position"]
            or expected["annotation_id"] != actual["annotation_id"]
            or canonical_json(expected["annotation"])
            != canonical_json(actual["annotation"])
        ):
            raise PackageValidationError(
                "Raw-batch replay mismatch at position "
                f"{expected['assignment_position']}"
            )

    draft_value = load_json(
        root / "provenance" / "calibration_human_draft.json"
    )
    draft_result = validate_export(
        assignment_value=source_assignment,
        export_value=draft_value,
        require_final=False,
    )
    comparisons, calibration_report = compare_calibration(
        draft_result["normalized_rows"],
        normalized_rows,
        source_assignment,
    )
    stored_comparisons = load_jsonl(
        root / "calibration_comparison.jsonl"
    )
    stored_report = load_json(root / "calibration_report.json")
    if (
        canonical_json(stored_comparisons) != canonical_json(comparisons)
        or canonical_json(stored_report)
        != canonical_json(calibration_report)
    ):
        raise PackageValidationError("Calibration recomputation mismatch")

    summary = semantic_summary(normalized_rows)
    if canonical_json(summary) != canonical_json(
        load_json(root / "annotation_summary.json")
    ):
        raise PackageValidationError("Annotation summary mismatch")
    if read_distribution(root / "label_distribution.csv") != (
        distribution_rows(normalized_rows)
    ):
        raise PackageValidationError("Label distribution mismatch")
    if canonical_json(manifest["summary"]) != canonical_json(summary):
        raise PackageValidationError("Manifest summary mismatch")

    return {
        "validation_status": "VALID_AI_PREANNOTATION_PACKAGE",
        "package": str(root),
        "records": len(normalized_rows),
        "calibration_records": len(draft_result["normalized_rows"]),
        "checksummed_files": len(checksum_inventory),
        "review_assignment_id": review_assignment["assignment_id"],
        "suggestion_set_id": suggestions["payload"]["suggestion_set_id"],
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an AI-assisted human-check package."
    )
    parser.add_argument("--package", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = validate_package(args.package.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
