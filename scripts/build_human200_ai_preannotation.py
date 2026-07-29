"""Validate and package auditable AI pre-annotations for human review.

This builder deliberately keeps three things separate:

1. the frozen blinded assignment;
2. the annotator's calibration DRAFT;
3. AI suggestions that still require explicit human verification.

Raw batch files are JSONL. Each line must contain exactly:

    {
      "assignment_position": 1,
      "annotation_id": "...",
      "review_text_sha256": "...",
      "annotation": {
        "annotation_status": "LABELED|ESCALATE|REJECT_NON_REVIEW",
        "aspects": [...],
        "review_uncertainty_codes": [...],
        "notes": "..."
      }
    }

Evidence in raw batches uses the LLM annotation contract:
``quote``, one-based literal ``occurrence``, and ``polarity``. The existing
strict validator replays every quote against canonical review text and emits
Unicode code-point offsets for the browser review workbench.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from human_annotation_ui.common import (
    canonical_json,
    load_json,
    sha256_file,
    sha256_text,
    verify_assignment,
)
from human_annotation_ui.validate_export import validate_export
from lazada_collector.llm_annotation import (
    ASPECT_COLUMNS,
    AnnotationValidationError,
    label_vector,
    validate_and_normalize_annotation,
)


RAW_ROW_FIELDS = {
    "assignment_position",
    "annotation_id",
    "review_text_sha256",
    "annotation",
}
SUGGESTION_SCHEMA_VERSION = "human-absa-ai-suggestions/1.0.0"
SUGGESTION_PAYLOAD_FIELDS = {
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
SUGGESTION_RECORD_FIELDS = {
    "annotation_id",
    "review_text_sha256",
    "annotation_status",
    "aspects",
    "review_uncertainty_codes",
    "notes",
}


class PreannotationBuildError(ValueError):
    """Raised when an input or generated artifact fails closed."""


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise PreannotationBuildError(f"{context} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise PreannotationBuildError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def parse_built_at(raw: str | None) -> str:
    if raw is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PreannotationBuildError(
            "--built-at must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise PreannotationBuildError("--built-at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_text(
        path,
        "".join(f"{canonical_json(row)}\n" for row in rows),
    )


def load_raw_batches(
    batch_paths: list[Path],
    assignment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    assignment_records = assignment["records"]
    by_position: dict[int, dict[str, Any]] = {}
    for batch_path in batch_paths:
        try:
            lines = batch_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise PreannotationBuildError(
                f"Cannot read batch: {batch_path}"
            ) from exc
        if not lines:
            raise PreannotationBuildError(f"Empty batch: {batch_path}")
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise PreannotationBuildError(
                    f"Blank JSONL line: {batch_path}:{line_number}"
                )
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreannotationBuildError(
                    f"Invalid JSON: {batch_path}:{line_number}: {exc}"
                ) from exc
            require_exact_keys(
                raw,
                RAW_ROW_FIELDS,
                context=f"{batch_path.name}:{line_number}",
            )
            position = raw.get("assignment_position")
            if (
                not isinstance(position, int)
                or isinstance(position, bool)
                or not 1 <= position <= len(assignment_records)
            ):
                raise PreannotationBuildError(
                    f"Invalid assignment_position at "
                    f"{batch_path.name}:{line_number}"
                )
            if position in by_position:
                raise PreannotationBuildError(
                    f"Duplicate assignment_position: {position}"
                )
            item = assignment_records[position - 1]
            if (
                raw.get("annotation_id") != item["annotation_id"]
                or raw.get("review_text_sha256")
                != item["review_text_sha256"]
            ):
                raise PreannotationBuildError(
                    f"Assignment identity mismatch at position {position}"
                )
            try:
                annotation = validate_and_normalize_annotation(
                    item["reviewContent"],
                    raw.get("annotation"),
                )
            except AnnotationValidationError as exc:
                raise PreannotationBuildError(
                    f"Invalid annotation at position {position} "
                    f"({batch_path.name}:{line_number}): {exc}"
                ) from exc
            by_position[position] = {
                "assignment_position": position,
                "annotation_id": item["annotation_id"],
                "review_text_sha256": item["review_text_sha256"],
                "source_batch": batch_path.name,
                "source_line": line_number,
                "annotation": annotation,
            }

    expected = set(range(1, len(assignment_records) + 1))
    actual = set(by_position)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PreannotationBuildError(
            f"Batch coverage mismatch; missing={missing}, extra={extra}"
        )
    return [by_position[position] for position in sorted(by_position)]


def ui_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "annotation_status": annotation["annotation_status"],
        "aspects": [
            {
                "aspect": aspect["aspect"],
                "label": aspect["label"],
                "evidence": [
                    {
                        "quote": evidence["text"],
                        "start": evidence["start"],
                        "end": evidence["end"],
                        "polarity": evidence["polarity"],
                    }
                    for evidence in aspect["evidence"]
                ],
                "uncertainty_codes": aspect["uncertainty_codes"],
            }
            for aspect in annotation["aspects"]
        ],
        "review_uncertainty_codes": annotation[
            "review_uncertainty_codes"
        ],
        "notes": annotation["notes"],
    }


def build_review_assignment(
    source_assignment: Mapping[str, Any],
    annotations: list[Mapping[str, Any]],
    *,
    built_at: str,
) -> dict[str, Any]:
    semantic_fingerprint = sha256_text(
        canonical_json(
            [
                {
                    "annotation_id": row["annotation_id"],
                    "status": row["annotation"]["annotation_status"],
                    "labels": list(label_vector(row["annotation"])),
                }
                for row in annotations
            ]
        )
    )
    payload = {
        "schema_version": source_assignment["schema_version"],
        "assignment_id": (
            "hra-ai-review-"
            + sha256_text(
                source_assignment["assignment_payload_sha256"]
                + semantic_fingerprint
            )[:16]
        ),
        "reference_id": source_assignment["reference_id"],
        "role": "A",
        "item_count": source_assignment["item_count"],
        "guideline": source_assignment["guideline"],
        "created_at": built_at,
        "records": source_assignment["records"],
    }
    assignment = {
        **payload,
        "assignment_payload_sha256": sha256_text(canonical_json(payload)),
    }
    return verify_assignment(assignment)


def build_suggestions(
    review_assignment: Mapping[str, Any],
    annotations: list[Mapping[str, Any]],
    *,
    built_at: str,
    source_method: str,
) -> dict[str, Any]:
    records = [
        {
            "annotation_id": row["annotation_id"],
            "review_text_sha256": row["review_text_sha256"],
            **ui_annotation(row["annotation"]),
        }
        for row in annotations
    ]
    for index, record in enumerate(records, 1):
        require_exact_keys(
            record,
            SUGGESTION_RECORD_FIELDS,
            context=f"suggestions.records[{index}]",
        )
    suggestion_set_id = (
        "ai-suggestions-"
        + sha256_text(
            canonical_json(
                {
                    "assignment_payload_sha256": review_assignment[
                        "assignment_payload_sha256"
                    ],
                    "records": records,
                }
            )
        )[:16]
    )
    payload = {
        "suggestion_set_id": suggestion_set_id,
        "assignment_id": review_assignment["assignment_id"],
        "reference_id": review_assignment["reference_id"],
        "assignment_payload_sha256": review_assignment[
            "assignment_payload_sha256"
        ],
        "guideline_version": review_assignment["guideline"]["version"],
        "guideline_sha256": review_assignment["guideline"]["sha256"],
        "item_count": review_assignment["item_count"],
        "created_at": built_at,
        "source_method": source_method,
        "records": records,
    }
    require_exact_keys(
        payload,
        SUGGESTION_PAYLOAD_FIELDS,
        context="suggestions.payload",
    )
    return {
        "schema_version": SUGGESTION_SCHEMA_VERSION,
        "payload": payload,
        "payload_sha256": sha256_text(canonical_json(payload)),
    }


def compare_calibration(
    human_rows: list[Mapping[str, Any]],
    ai_rows: list[Mapping[str, Any]],
    assignment: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    human_by_id = {
        row["annotation_id"]: row
        for row in human_rows
    }
    ai_by_id = {
        row["annotation_id"]: row
        for row in ai_rows
    }
    position_by_id = {
        row["annotation_id"]: index
        for index, row in enumerate(assignment["records"], 1)
    }
    comparisons: list[dict[str, Any]] = []
    aspect_agreements = Counter()
    aspect_totals = Counter()
    status_agreements = 0
    vector_agreements = 0
    cell_agreements = 0
    cell_total = 0

    for annotation_id in sorted(
        human_by_id,
        key=position_by_id.__getitem__,
    ):
        human = human_by_id[annotation_id]
        ai = ai_by_id[annotation_id]["annotation"]
        human_labels = {
            row["aspect"]: row["label"]
            for row in human["aspects"]
        }
        ai_labels = {
            row["aspect"]: row["label"]
            for row in ai["aspects"]
        }
        differences: list[dict[str, Any]] = []
        for aspect in ASPECT_COLUMNS:
            same = human_labels[aspect] == ai_labels[aspect]
            aspect_totals[aspect] += 1
            cell_total += 1
            if same:
                aspect_agreements[aspect] += 1
                cell_agreements += 1
            else:
                differences.append(
                    {
                        "aspect": aspect,
                        "human_label": human_labels[aspect],
                        "ai_label": ai_labels[aspect],
                    }
                )
        status_same = (
            human["annotation_status"] == ai["annotation_status"]
        )
        if status_same:
            status_agreements += 1
        vector_same = not differences
        if vector_same:
            vector_agreements += 1
        comparisons.append(
            {
                "assignment_position": position_by_id[annotation_id],
                "annotation_id": annotation_id,
                "review_text_sha256": human["review_text_sha256"],
                "status_agreement": status_same,
                "vector_agreement": vector_same,
                "differences": differences,
                "human_annotation": {
                    "annotation_status": human["annotation_status"],
                    "aspects": human["aspects"],
                    "review_uncertainty_codes": human[
                        "review_uncertainty_codes"
                    ],
                    "notes": human["notes"],
                },
                "ai_annotation": ai,
            }
        )

    count = len(comparisons)
    report = {
        "calibration_records": count,
        "status_exact_count": status_agreements,
        "status_exact_rate": (
            status_agreements / count if count else None
        ),
        "full_vector_exact_count": vector_agreements,
        "full_vector_exact_rate": (
            vector_agreements / count if count else None
        ),
        "aspect_cell_exact_count": cell_agreements,
        "aspect_cell_total": cell_total,
        "aspect_cell_exact_rate": (
            cell_agreements / cell_total if cell_total else None
        ),
        "per_aspect_exact": {
            aspect: {
                "agreements": aspect_agreements[aspect],
                "total": aspect_totals[aspect],
                "rate": (
                    aspect_agreements[aspect] / aspect_totals[aspect]
                    if aspect_totals[aspect]
                    else None
                ),
            }
            for aspect in ASPECT_COLUMNS
        },
        "interpretation": (
            "Diagnostic calibration comparison only. Ten records are too "
            "small for IAA or model-performance claims, and the AI saw these "
            "records during calibration."
        ),
    }
    return comparisons, report


def distribution_rows(
    annotations: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    labels = (2, -1, 0, 1, "1, -1", None)
    counts = {
        aspect: Counter()
        for aspect in ASPECT_COLUMNS
    }
    for row in annotations:
        for aspect in row["annotation"]["aspects"]:
            counts[aspect["aspect"]][aspect["label"]] += 1
    output: list[dict[str, Any]] = []
    for aspect in ASPECT_COLUMNS:
        for label in labels:
            output.append(
                {
                    "aspect": aspect,
                    "label": "null" if label is None else str(label),
                    "count": counts[aspect][label],
                }
            )
    return output


def semantic_summary(
    annotations: list[Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(
        row["annotation"]["annotation_status"]
        for row in annotations
    )
    review_multi_polarity = 0
    mixed_cells = 0
    mentioned_cells = 0
    uncertainty_records = 0
    for row in annotations:
        annotation = row["annotation"]
        labels = [
            aspect["label"]
            for aspect in annotation["aspects"]
        ]
        mentioned_cells += sum(
            label not in {2, None}
            for label in labels
        )
        mixed_cells += labels.count("1, -1")
        has_positive = any(
            label in {1, "1, -1"}
            for label in labels
        )
        has_negative = any(
            label in {-1, "1, -1"}
            for label in labels
        )
        if has_positive and has_negative:
            review_multi_polarity += 1
        if (
            annotation["review_uncertainty_codes"]
            or any(
                aspect["uncertainty_codes"]
                for aspect in annotation["aspects"]
            )
        ):
            uncertainty_records += 1
    return {
        "records": len(annotations),
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "mentioned_aspect_cells": mentioned_cells,
        "mixed_aspect_cells": mixed_cells,
        "review_level_multi_polarity_records": review_multi_polarity,
        "uncertainty_records": uncertainty_records,
    }


def write_distribution_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["aspect", "label", "count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_package(
    *,
    assignment_path: Path,
    draft_path: Path,
    batch_paths: list[Path],
    output: Path,
    built_at: str,
    source_method: str,
    audit_artifact_paths: list[Path],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    assignment = verify_assignment(load_json(assignment_path))
    draft_value = load_json(draft_path)
    draft_result = validate_export(
        assignment_value=assignment,
        export_value=draft_value,
        require_final=False,
    )
    human_rows = draft_result["normalized_rows"]
    if not human_rows:
        raise PreannotationBuildError(
            "Calibration DRAFT has no completed records"
        )
    annotations = load_raw_batches(batch_paths, assignment)
    review_assignment = build_review_assignment(
        assignment,
        annotations,
        built_at=built_at,
    )
    suggestions = build_suggestions(
        review_assignment,
        annotations,
        built_at=built_at,
        source_method=source_method,
    )
    comparison, calibration_report = compare_calibration(
        human_rows,
        annotations,
        assignment,
    )
    summary = semantic_summary(annotations)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        normalized_rows = [
            {
                "assignment_position": row["assignment_position"],
                "annotation_id": row["annotation_id"],
                "review_text_sha256": row["review_text_sha256"],
                "reference_id": assignment["reference_id"],
                "guideline_version": assignment["guideline"]["version"],
                "annotation_method": source_method,
                "source_batch": row["source_batch"],
                "source_line": row["source_line"],
                "annotation": row["annotation"],
            }
            for row in annotations
        ]
        write_jsonl(
            temp_root / "ai_preannotations.jsonl",
            normalized_rows,
        )
        write_jsonl(
            temp_root / "calibration_comparison.jsonl",
            comparison,
        )
        write_json(
            temp_root / "calibration_report.json",
            calibration_report,
        )
        write_json(
            temp_root / "annotation_summary.json",
            summary,
        )
        write_distribution_csv(
            temp_root / "label_distribution.csv",
            distribution_rows(annotations),
        )
        write_json(
            temp_root / "human_check" / "ai_review.assignment.json",
            review_assignment,
        )
        write_json(
            temp_root / "human_check" / "ai_suggestions.json",
            suggestions,
        )

        provenance = temp_root / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            assignment_path,
            provenance / "source_annotator_a.assignment.json",
        )
        shutil.copyfile(
            draft_path,
            provenance / "calibration_human_draft.json",
        )
        shutil.copyfile(
            Path(__file__).resolve(),
            provenance / Path(__file__).name,
        )
        guideline_source = assignment_path.parent.parent / (
            "ABSA_ANNOTATION_GUIDELINE_V2.md"
        )
        shutil.copyfile(
            guideline_source,
            provenance / guideline_source.name,
        )
        annotation_batches_root = provenance / "annotation_batches"
        annotation_batches_root.mkdir(parents=True, exist_ok=True)
        for batch_path in batch_paths:
            shutil.copyfile(
                batch_path,
                annotation_batches_root / batch_path.name,
            )
        cross_audit_root = provenance / "cross_audit"
        if audit_artifact_paths:
            cross_audit_root.mkdir(parents=True, exist_ok=True)
        seen_audit_names: set[str] = set()
        for audit_path in audit_artifact_paths:
            if not audit_path.is_file():
                raise FileNotFoundError(audit_path)
            if audit_path.name in seen_audit_names:
                raise PreannotationBuildError(
                    f"Duplicate audit artifact filename: {audit_path.name}"
                )
            seen_audit_names.add(audit_path.name)
            shutil.copyfile(
                audit_path,
                cross_audit_root / audit_path.name,
            )

        manifest = {
            "schema_version": "human-absa-ai-preannotation-package/1.0.0",
            "status": "AI_PREANNOTATION_PENDING_HUMAN_VERIFICATION",
            "built_at": built_at,
            "source_method": source_method,
            "source_assignment": {
                "filename": assignment_path.name,
                "sha256": sha256_file(assignment_path),
                "assignment_id": assignment["assignment_id"],
                "assignment_payload_sha256": assignment[
                    "assignment_payload_sha256"
                ],
            },
            "calibration_draft": {
                "filename": draft_path.name,
                "sha256": sha256_file(draft_path),
                "completed_valid": draft_result[
                    "records_completed_valid"
                ],
                "incomplete": draft_result["records_incomplete"],
                "payload_sha256": draft_result["payload_sha256"],
            },
            "annotation_batches": [
                {
                    "filename": path.name,
                    "sha256": sha256_file(path),
                }
                for path in batch_paths
            ],
            "cross_audit_artifacts": [
                {
                    "filename": path.name,
                    "sha256": sha256_file(path),
                }
                for path in audit_artifact_paths
            ],
            "records": assignment["item_count"],
            "review_assignment_id": review_assignment["assignment_id"],
            "suggestion_set_id": suggestions["payload"][
                "suggestion_set_id"
            ],
            "guideline": assignment["guideline"],
            "summary": summary,
            "calibration_report": calibration_report,
            "scientific_use": {
                "is_human_gold": False,
                "is_double_blind_human_annotation": False,
                "permitted_current_use": (
                    "AI suggestion set for explicit per-record human "
                    "verification and correction."
                ),
                "required_next_step": (
                    "A human reviewer must inspect all 200 canonical texts, "
                    "confirm or edit labels/evidence, and export a validated "
                    "FINAL before any human-verified claim."
                ),
            },
        }
        write_json(temp_root / "manifest.json", manifest)
        instructions = (
            "# Human-check AI pre-annotations\n\n"
            "This package contains AI suggestions, not human gold.\n\n"
            "Start the dedicated review UI from the repository root:\n\n"
            "```powershell\n"
            ".\\human_annotation_ui\\start_ai_review.ps1\n"
            "```\n\n"
            "The review assignment has a separate assignment ID, so it does "
            "not overwrite the original role-A browser session. Read every "
            "review, inspect every suggested label/evidence span, correct "
            "errors, then mark the record complete. Export DRAFT backups "
            "regularly and export FINAL only after 200/200 records have been "
            "explicitly reviewed.\n"
        )
        write_text(
            temp_root / "human_check" / "README.md",
            instructions,
        )

        checksum_targets = sorted(
            (
                path
                for path in temp_root.rglob("*")
                if path.is_file()
                and path != temp_root / "SHA256SUMS.txt"
            ),
            key=lambda path: path.relative_to(temp_root).as_posix(),
        )
        write_text(
            temp_root / "SHA256SUMS.txt",
            "".join(
                f"{sha256_file(path)}  "
                f"{path.relative_to(temp_root).as_posix()}\n"
                for path in checksum_targets
            ),
        )
        temp_root.replace(output)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return {
        "output": str(output),
        "records": len(annotations),
        "calibration_records": len(human_rows),
        "review_assignment_id": review_assignment["assignment_id"],
        "suggestion_set_id": suggestions["payload"]["suggestion_set_id"],
        "summary": summary,
        "sha256sums": str(output / "SHA256SUMS.txt"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate 200 AI ABSA annotations and build an auditable "
            "human-check package."
        )
    )
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument(
        "--batch",
        dest="batches",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every raw JSONL batch.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--built-at",
        help="Optional fixed ISO-8601 timestamp for reproducible builds.",
    )
    parser.add_argument(
        "--source-method",
        default="codex_guideline_v2_calibrated_preannotation",
    )
    parser.add_argument(
        "--audit-artifact",
        dest="audit_artifacts",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional cross-audit, adjudication, or reconciliation artifact "
            "copied into package provenance. Repeat as needed."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_package(
        assignment_path=args.assignment.resolve(),
        draft_path=args.draft.resolve(),
        batch_paths=[path.resolve() for path in args.batches],
        output=args.output.resolve(),
        built_at=parse_built_at(args.built_at),
        source_method=args.source_method,
        audit_artifact_paths=[
            path.resolve()
            for path in args.audit_artifacts
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
