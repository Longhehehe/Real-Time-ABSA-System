"""Strictly validate a UI backup/final export against its frozen assignment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from human_annotation_ui import (
    EXPORT_SCHEMA_VERSION,
    LEGACY_EXPORT_SCHEMA_VERSION,
    LEGACY_UI_VERSION,
    UI_VERSION,
)
from human_annotation_ui.common import (
    canonical_json,
    load_json,
    sha256_file,
    sha256_text,
    verify_assignment,
    verify_export_envelope,
)
from lazada_collector.llm_annotation import (
    ASPECT_COLUMNS,
    AnnotationValidationError,
    validate_and_normalize_annotation,
)


PAYLOAD_FIELDS_V1 = {
    "workspace_schema_version",
    "export_status",
    "ui_version",
    "assignment_id",
    "reference_id",
    "role",
    "assignment_payload_sha256",
    "guideline_version",
    "guideline_sha256",
    "annotator_id",
    "item_count",
    "workspace_created_at",
    "exported_at",
    "finalized_at",
    "records",
    "audit_events",
}
PAYLOAD_FIELDS_V2 = PAYLOAD_FIELDS_V1 | {"workflow"}
WORKFLOW_ENVELOPE_FIELDS = {
    "schema_version",
    "payload",
    "payload_sha256",
}
WORKFLOW_PAYLOAD_FIELDS = {
    "workflow_id",
    "mode",
    "assignment_id",
    "assignment_payload_sha256",
    "suggestions_available",
    "suggestion_set_id",
    "suggestions_payload_sha256",
    "adjudication_available",
    "adjudication_set_id",
    "adjudication_payload_sha256",
}
RECORD_FIELDS = {
    "annotation_id",
    "review_text_sha256",
    "annotation_status",
    "aspects",
    "review_uncertainty_codes",
    "notes",
    "read_complete",
    "revisit",
    "complete",
    "completed_at",
    "updated_at",
    "revision_number",
    "revisions",
}
ASPECT_FIELDS = {
    "aspect",
    "label",
    "evidence",
    "uncertainty_codes",
}
EVIDENCE_FIELDS = {
    "quote",
    "start",
    "end",
    "polarity",
}
REVISION_FIELDS = {
    "revision",
    "event",
    "at",
    "annotation",
}
ANNOTATION_FIELDS = {
    "annotation_status",
    "aspects",
    "review_uncertainty_codes",
    "notes",
}
FORBIDDEN_EXPORT_KEYS = {
    "reviewcontent",
    "rating",
    "category",
    "product_id",
    "seller_id",
    "shop_id",
    "source_url",
    "query",
    "collection_transport",
    "old_labels",
    "llm_labels",
    "model_predictions",
}


class HumanExportValidationError(ValueError):
    """Raised when a human annotation export fails closed."""


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise HumanExportValidationError(f"{context} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise HumanExportValidationError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def reject_forbidden_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_EXPORT_KEYS:
                raise HumanExportValidationError(
                    f"Forbidden key in blinded export at {path}: {key}"
                )
            reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, path=f"{path}[{index}]")


def verify_workflow(
    value: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_keys(
        value,
        WORKFLOW_ENVELOPE_FIELDS,
        context="Workflow envelope",
    )
    if value.get("schema_version") != "human-absa-workflow/2.0.0":
        raise HumanExportValidationError(
            "Workflow schema version mismatch"
        )
    workflow_payload = value.get("payload")
    require_exact_keys(
        workflow_payload,
        WORKFLOW_PAYLOAD_FIELDS,
        context="Workflow payload",
    )
    if value.get("payload_sha256") != sha256_text(
        canonical_json(workflow_payload)
    ):
        raise HumanExportValidationError(
            "Workflow payload checksum mismatch"
        )
    if (
        workflow_payload.get("assignment_id")
        != assignment["assignment_id"]
        or workflow_payload.get("assignment_payload_sha256")
        != assignment["assignment_payload_sha256"]
        or not isinstance(workflow_payload.get("workflow_id"), str)
        or not workflow_payload["workflow_id"]
    ):
        raise HumanExportValidationError(
            "Workflow/assignment binding mismatch"
        )
    mode = workflow_payload.get("mode")
    if mode not in {
        "BLINDED_INDEPENDENT_ANNOTATION",
        "AI_ASSISTED_HUMAN_VERIFICATION",
        "EXPERT_ADJUDICATION",
    }:
        raise HumanExportValidationError("Invalid workflow mode")
    ai_mode = mode == "AI_ASSISTED_HUMAN_VERIFICATION"
    adjudication_mode = mode == "EXPERT_ADJUDICATION"
    if (
        workflow_payload.get("suggestions_available") is not ai_mode
        or workflow_payload.get("adjudication_available")
        is not adjudication_mode
    ):
        raise HumanExportValidationError(
            "Workflow auxiliary-input flags mismatch"
        )
    for enabled, identifier_field, hash_field in (
        (
            ai_mode,
            "suggestion_set_id",
            "suggestions_payload_sha256",
        ),
        (
            adjudication_mode,
            "adjudication_set_id",
            "adjudication_payload_sha256",
        ),
    ):
        identifier = workflow_payload.get(identifier_field)
        digest = workflow_payload.get(hash_field)
        if enabled:
            if (
                not isinstance(identifier, str)
                or not identifier
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise HumanExportValidationError(
                    f"Workflow {identifier_field}/{hash_field} invalid"
                )
        elif identifier is not None or digest is not None:
            raise HumanExportValidationError(
                f"Workflow disabled {identifier_field}/{hash_field} "
                "must be null"
            )
    if (
        mode == "BLINDED_INDEPENDENT_ANNOTATION"
        and assignment["role"] not in {"A", "B"}
    ):
        raise HumanExportValidationError(
            "Blind workflow requires role A/B"
        )
    if (
        mode == "EXPERT_ADJUDICATION"
        and assignment["role"] != "ADJUDICATOR"
    ):
        raise HumanExportValidationError(
            "Adjudication workflow requires ADJUDICATOR role"
        )
    return json.loads(canonical_json(value))


def occurrence_at_offset(text: str, quote: str, start: int) -> int:
    if text[start : start + len(quote)] != quote:
        return 0
    occurrence = 0
    cursor = 0
    while True:
        index = text.find(quote, cursor)
        if index < 0 or index > start:
            return 0
        occurrence += 1
        if index == start:
            return occurrence
        cursor = index + 1


def annotation_to_validator_payload(
    review_text: str,
    annotation: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    require_exact_keys(annotation, ANNOTATION_FIELDS, context=context)
    raw_aspects = annotation.get("aspects")
    if not isinstance(raw_aspects, list) or len(raw_aspects) != len(
        ASPECT_COLUMNS
    ):
        raise HumanExportValidationError(
            f"{context}.aspects must contain exactly nine rows"
        )
    aspects: list[dict[str, Any]] = []
    for index, (expected_aspect, row) in enumerate(
        zip(ASPECT_COLUMNS, raw_aspects, strict=True),
        1,
    ):
        require_exact_keys(
            row,
            ASPECT_FIELDS,
            context=f"{context}.aspects[{index}]",
        )
        if row.get("aspect") != expected_aspect:
            raise HumanExportValidationError(
                f"{context}.aspects[{index}] order/name mismatch"
            )
        raw_evidence = row.get("evidence")
        if not isinstance(raw_evidence, list):
            raise HumanExportValidationError(
                f"{context}.aspects[{index}].evidence must be a list"
            )
        evidence: list[dict[str, Any]] = []
        for evidence_index, item in enumerate(raw_evidence, 1):
            require_exact_keys(
                item,
                EVIDENCE_FIELDS,
                context=(
                    f"{context}.aspects[{index}]."
                    f"evidence[{evidence_index}]"
                ),
            )
            quote = item.get("quote")
            start = item.get("start")
            end = item.get("end")
            if (
                not isinstance(quote, str)
                or not quote
                or not quote.strip()
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or review_text[start:end] != quote
            ):
                raise HumanExportValidationError(
                    f"{context}.aspects[{index}]."
                    f"evidence[{evidence_index}] offset/text mismatch"
                )
            occurrence = occurrence_at_offset(review_text, quote, start)
            if occurrence < 1:
                raise HumanExportValidationError(
                    f"{context}.aspects[{index}]."
                    f"evidence[{evidence_index}] occurrence mismatch"
                )
            evidence.append(
                {
                    "quote": quote,
                    "occurrence": occurrence,
                    "polarity": item.get("polarity"),
                }
            )
        aspects.append(
            {
                "aspect": expected_aspect,
                "label": row.get("label"),
                "evidence": evidence,
                "uncertainty_codes": row.get("uncertainty_codes"),
            }
        )
    return {
        "annotation_status": annotation.get("annotation_status"),
        "aspects": aspects,
        "review_uncertainty_codes": annotation.get(
            "review_uncertainty_codes"
        ),
        "notes": annotation.get("notes"),
    }


def current_annotation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ANNOTATION_FIELDS
    }


def validate_export(
    *,
    assignment_value: Mapping[str, Any],
    export_value: Mapping[str, Any],
    require_final: bool,
) -> dict[str, Any]:
    assignment = verify_assignment(assignment_value)
    envelope = verify_export_envelope(export_value)
    payload = envelope["payload"]
    export_schema = envelope["schema_version"]
    is_v2 = export_schema == EXPORT_SCHEMA_VERSION
    require_exact_keys(
        payload,
        PAYLOAD_FIELDS_V2 if is_v2 else PAYLOAD_FIELDS_V1,
        context="Export payload",
    )
    reject_forbidden_keys(payload)

    expected_workspace = (
        "human-absa-workspace/2.0.0"
        if is_v2
        else "human-absa-workspace/1.0.0"
    )
    expected_ui = UI_VERSION if is_v2 else LEGACY_UI_VERSION
    if payload.get("workspace_schema_version") != expected_workspace:
        raise HumanExportValidationError("Workspace schema version mismatch")
    if payload.get("ui_version") != expected_ui:
        raise HumanExportValidationError("UI version mismatch")
    workflow = (
        verify_workflow(payload["workflow"], assignment)
        if is_v2
        else None
    )
    export_status = payload.get("export_status")
    if export_status not in {"DRAFT", "FINAL"}:
        raise HumanExportValidationError("Invalid export_status")
    if require_final and export_status != "FINAL":
        raise HumanExportValidationError("--require-final received DRAFT")
    for field in (
        "assignment_id",
        "reference_id",
        "role",
        "assignment_payload_sha256",
    ):
        if payload.get(field) != assignment[field]:
            raise HumanExportValidationError(
                f"Export/assignment mismatch: {field}"
            )
    if (
        payload.get("guideline_version")
        != assignment["guideline"]["version"]
        or payload.get("guideline_sha256")
        != assignment["guideline"]["sha256"]
    ):
        raise HumanExportValidationError("Guideline version/hash mismatch")
    annotator_id = payload.get("annotator_id")
    if not isinstance(annotator_id, str):
        raise HumanExportValidationError("annotator_id must be a string")
    if export_status == "FINAL" and not annotator_id.strip():
        raise HumanExportValidationError(
            "FINAL requires a non-empty annotator_id"
        )
    if payload.get("item_count") != assignment["item_count"]:
        raise HumanExportValidationError("item_count mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != assignment[
        "item_count"
    ]:
        raise HumanExportValidationError("Export records count mismatch")
    if not isinstance(payload.get("audit_events"), list):
        raise HumanExportValidationError("audit_events must be a list")
    if export_status == "FINAL":
        if (
            not isinstance(payload.get("finalized_at"), str)
            or not payload["finalized_at"]
        ):
            raise HumanExportValidationError(
                "FINAL requires finalized_at"
            )
    elif payload.get("finalized_at") is not None:
        raise HumanExportValidationError(
            "DRAFT must have finalized_at=null"
        )

    assignment_by_id = {
        item["annotation_id"]: item for item in assignment["records"]
    }
    seen_ids: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    invalid_draft_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for index, record in enumerate(records, 1):
        require_exact_keys(
            record,
            RECORD_FIELDS,
            context=f"records[{index}]",
        )
        annotation_id = record.get("annotation_id")
        if (
            not isinstance(annotation_id, str)
            or annotation_id in seen_ids
            or annotation_id not in assignment_by_id
        ):
            raise HumanExportValidationError(
                f"records[{index}] invalid/duplicate annotation_id"
            )
        seen_ids.add(annotation_id)
        assignment_row = assignment_by_id[annotation_id]
        if record.get("review_text_sha256") != assignment_row[
            "review_text_sha256"
        ]:
            raise HumanExportValidationError(
                f"records[{index}] review hash mismatch"
            )
        review_text = assignment_row["reviewContent"]
        if sha256_text(review_text) != record["review_text_sha256"]:
            raise HumanExportValidationError(
                f"records[{index}] assignment text hash mismatch"
            )
        if not isinstance(record.get("read_complete"), bool):
            raise HumanExportValidationError(
                f"records[{index}].read_complete must be boolean"
            )
        if not isinstance(record.get("revisit"), bool):
            raise HumanExportValidationError(
                f"records[{index}].revisit must be boolean"
            )
        if not isinstance(record.get("complete"), bool):
            raise HumanExportValidationError(
                f"records[{index}].complete must be boolean"
            )
        revision_number = record.get("revision_number")
        revisions = record.get("revisions")
        if (
            not isinstance(revision_number, int)
            or isinstance(revision_number, bool)
            or revision_number < 0
            or not isinstance(revisions, list)
            or len(revisions) != revision_number
        ):
            raise HumanExportValidationError(
                f"records[{index}] revision ledger mismatch"
            )
        for revision_index, revision in enumerate(revisions, 1):
            require_exact_keys(
                revision,
                REVISION_FIELDS,
                context=(
                    f"records[{index}].revisions[{revision_index}]"
                ),
            )
            if (
                revision.get("revision") != revision_index
                or revision.get("event") != "COMPLETED"
                or not isinstance(revision.get("at"), str)
            ):
                raise HumanExportValidationError(
                    f"records[{index}] invalid revision sequence/event"
                )
            revision_payload = annotation_to_validator_payload(
                review_text,
                revision.get("annotation"),
                context=(
                    f"records[{index}].revisions[{revision_index}]."
                    "annotation"
                ),
            )
            try:
                validate_and_normalize_annotation(
                    review_text,
                    revision_payload,
                )
            except AnnotationValidationError as exc:
                raise HumanExportValidationError(
                    f"Invalid revision annotation at records[{index}]: {exc}"
                ) from exc

        completed = record["complete"]
        if completed:
            if (
                not record["read_complete"]
                or revision_number < 1
                or not isinstance(record.get("completed_at"), str)
                or not record["completed_at"]
            ):
                raise HumanExportValidationError(
                    f"records[{index}] completed-state invariant failed"
                )
            validator_payload = annotation_to_validator_payload(
                review_text,
                current_annotation(record),
                context=f"records[{index}].annotation",
            )
            try:
                normalized = validate_and_normalize_annotation(
                    review_text,
                    validator_payload,
                )
            except AnnotationValidationError as exc:
                raise HumanExportValidationError(
                    f"Invalid completed annotation at records[{index}]: {exc}"
                ) from exc
            status = normalized["annotation_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            normalized_rows.append(
                {
                    "annotation_id": annotation_id,
                    "review_text_sha256": record[
                        "review_text_sha256"
                    ],
                    "annotator_id": annotator_id,
                    "assignment_id": assignment["assignment_id"],
                    "reference_id": assignment["reference_id"],
                    "role": assignment["role"],
                    "workflow_mode": (
                        workflow["payload"]["mode"]
                        if workflow is not None
                        else "LEGACY_UNBOUND_WORKFLOW"
                    ),
                    "workflow_id": (
                        workflow["payload"]["workflow_id"]
                        if workflow is not None
                        else None
                    ),
                    "guideline_version": payload["guideline_version"],
                    "guideline_sha256": payload["guideline_sha256"],
                    "completed_at": record["completed_at"],
                    **normalized,
                }
            )
        else:
            if record.get("completed_at") is not None:
                raise HumanExportValidationError(
                    f"records[{index}] incomplete row has completed_at"
                )
            invalid_draft_rows.append(
                {
                    "annotation_id": annotation_id,
                    "reason": "INCOMPLETE_NOT_SEMANTICALLY_VALIDATED",
                }
            )

    if seen_ids != set(assignment_by_id):
        raise HumanExportValidationError("Export/assignment ID bijection failed")
    if export_status == "FINAL" and invalid_draft_rows:
        raise HumanExportValidationError(
            "FINAL contains incomplete records"
        )
    if export_status == "FINAL" and len(normalized_rows) != assignment[
        "item_count"
    ]:
        raise HumanExportValidationError(
            "FINAL does not contain every validated assignment record"
        )
    if workflow is not None:
        mode = workflow["payload"]["mode"]
        audit_events = payload["audit_events"]
        if mode == "BLINDED_INDEPENDENT_ANNOTATION":
            forbidden_events = {
                "AI_SUGGESTIONS_SEEDED_FOR_HUMAN_VERIFICATION",
                "ADJUDICATION_SOURCE_APPLIED",
                "ADJUDICATION_DECISION",
            }
            if any(
                event.get("event") in forbidden_events
                for event in audit_events
                if isinstance(event, Mapping)
            ):
                raise HumanExportValidationError(
                    "Blind export contains assisted/adjudication event"
                )
        elif mode == "AI_ASSISTED_HUMAN_VERIFICATION":
            seeded = [
                event
                for event in audit_events
                if isinstance(event, Mapping)
                and event.get("event")
                == "AI_SUGGESTIONS_SEEDED_FOR_HUMAN_VERIFICATION"
                and event.get("suggestion_set_id")
                == workflow["payload"]["suggestion_set_id"]
                and event.get("suggestions_payload_sha256")
                == workflow["payload"]["suggestions_payload_sha256"]
            ]
            if not seeded:
                raise HumanExportValidationError(
                    "AI-review export lacks matching suggestion-seed event"
                )
        elif export_status == "FINAL":
            decisions = {
                (
                    event.get("annotation_id"),
                    event.get("revision"),
                ): event
                for event in audit_events
                if isinstance(event, Mapping)
                and event.get("event") == "ADJUDICATION_DECISION"
            }
            for record in records:
                key = (
                    record["annotation_id"],
                    record["revision_number"],
                )
                event = decisions.get(key)
                if (
                    event is None
                    or event.get("resolution_basis")
                    not in {
                        "ACCEPT_A",
                        "ACCEPT_B",
                        "MATCHES_BOTH",
                        "MANUAL_OVERRIDE",
                    }
                    or not isinstance(
                        event.get("disagreement_fields"),
                        list,
                    )
                    or not event["disagreement_fields"]
                    or (
                        event["resolution_basis"] == "MANUAL_OVERRIDE"
                        and not record["notes"].strip()
                    )
                ):
                    raise HumanExportValidationError(
                        "Adjudication FINAL lacks a valid decision event"
                    )

    return {
        "validation_status": (
            "VALID_FINAL"
            if export_status == "FINAL"
            else "VALID_DRAFT_STRUCTURE"
        ),
        "export_schema_version": export_schema,
        "assignment_id": assignment["assignment_id"],
        "reference_id": assignment["reference_id"],
        "role": assignment["role"],
        "workflow_mode": (
            workflow["payload"]["mode"]
            if workflow is not None
            else "LEGACY_UNBOUND_WORKFLOW"
        ),
        "workflow_id": (
            workflow["payload"]["workflow_id"]
            if workflow is not None
            else None
        ),
        "export_status": export_status,
        "records_total": len(records),
        "records_completed_valid": len(normalized_rows),
        "records_incomplete": len(invalid_draft_rows),
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "payload_sha256": envelope["payload_sha256"],
        "normalized_rows": normalized_rows,
        "incomplete_rows": invalid_draft_rows,
    }


def publish_validation(
    *,
    result: dict[str, Any],
    assignment: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        normalized_path = temp_root / "validated_annotations.jsonl"
        with normalized_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in result["normalized_rows"]:
                handle.write(canonical_json(row))
                handle.write("\n")
        report = {
            key: value
            for key, value in result.items()
            if key not in {"normalized_rows", "incomplete_rows"}
        }
        report["validated_at"] = datetime.now(timezone.utc).isoformat()
        report_path = temp_root / "validation_report.json"
        report_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        assignment_by_id = {
            row["annotation_id"]: row for row in assignment["records"]
        }
        labels_path = temp_root / "labels_with_status.csv"
        with labels_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            fields = [
                "annotation_id",
                "review_text_sha256",
                "annotation_status",
                "reviewContent",
                *ASPECT_COLUMNS,
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in result["normalized_rows"]:
                output_row: dict[str, Any] = {
                    "annotation_id": row["annotation_id"],
                    "review_text_sha256": row["review_text_sha256"],
                    "annotation_status": row["annotation_status"],
                    "reviewContent": assignment_by_id[
                        row["annotation_id"]
                    ]["reviewContent"],
                }
                for aspect in row["aspects"]:
                    output_row[aspect["aspect"]] = aspect["label"]
                writer.writerow(output_row)

        paths = [normalized_path, report_path, labels_path]
        checksums_path = temp_root / "SHA256SUMS.txt"
        checksums_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in sorted(paths, key=lambda item: item.name)
            ),
            encoding="utf-8",
            newline="\n",
        )
        temp_root.replace(output)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "validated_annotations": len(result["normalized_rows"]),
        "sha256sums": str(output / "SHA256SUMS.txt"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly validate a human ABSA UI export."
    )
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--export", dest="export_path", type=Path, required=True)
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new versioned directory for validated artifacts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    assignment = verify_assignment(load_json(args.assignment))
    export_value = load_json(args.export_path)
    result = validate_export(
        assignment_value=assignment,
        export_value=export_value,
        require_final=args.require_final,
    )
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"normalized_rows", "incomplete_rows"}
    }
    if args.output is not None:
        summary["published"] = publish_validation(
            result=result,
            assignment=assignment,
            output=args.output,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
