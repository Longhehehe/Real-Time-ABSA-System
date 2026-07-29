"""Apply explicit cross-audit decisions without mutating raw annotations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from human_annotation_ui.common import (
    canonical_json,
    load_json,
    sha256_file,
    verify_assignment,
)
from lazada_collector.llm_annotation import (
    AnnotationValidationError,
    annotation_fingerprint,
    label_vector,
    validate_and_normalize_annotation,
)
from scripts.build_human200_ai_preannotation import load_raw_batches


AUDIT_FIELDS = {
    "assignment_position",
    "annotation_id",
    "severity",
    "rationale",
    "proposed_annotation",
}
DECISION_FIELDS = {
    "audit_file",
    "assignment_position",
    "annotation_id",
    "decision",
    "rationale",
    "replacement_annotation",
}
DECISIONS = {
    "APPLY_PROPOSED",
    "APPLY_CUSTOM",
    "RETAIN_ORIGINAL",
    "DEFER_HUMAN_CHECK",
}


class ReconciliationError(ValueError):
    """Raised when audit evidence and decisions do not close exactly."""


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ReconciliationError(f"{context} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ReconciliationError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line:
            raise ReconciliationError(
                f"Blank JSONL line: {path.name}:{line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReconciliationError(
                f"Invalid JSON: {path.name}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ReconciliationError(
                f"Row is not an object: {path.name}:{line_number}"
            )
        rows.append(row)
    return rows


def validate_annotation(
    annotation: Mapping[str, Any],
    review_text: str,
    *,
    context: str,
) -> dict[str, Any]:
    try:
        return validate_and_normalize_annotation(
            review_text,
            annotation,
        )
    except AnnotationValidationError as exc:
        raise ReconciliationError(
            f"{context} annotation invalid: {exc}"
        ) from exc


def load_audits(
    paths: list[Path],
    assignment: Mapping[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    audits: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for line_number, row in enumerate(read_jsonl(path), 1):
            require_exact_keys(
                row,
                AUDIT_FIELDS,
                context=f"{path.name}:{line_number}",
            )
            position = row.get("assignment_position")
            if (
                not isinstance(position, int)
                or isinstance(position, bool)
                or not 1 <= position <= assignment["item_count"]
            ):
                raise ReconciliationError(
                    f"Invalid audit position: {path.name}:{line_number}"
                )
            item = assignment["records"][position - 1]
            if row.get("annotation_id") != item["annotation_id"]:
                raise ReconciliationError(
                    f"Audit identity mismatch: {path.name}:{line_number}"
                )
            if row.get("severity") not in {"CHANGE", "CHECK"}:
                raise ReconciliationError(
                    f"Invalid audit severity: {path.name}:{line_number}"
                )
            if (
                not isinstance(row.get("rationale"), str)
                or not row["rationale"].strip()
            ):
                raise ReconciliationError(
                    f"Audit rationale missing: {path.name}:{line_number}"
                )
            proposed = row.get("proposed_annotation")
            if row["severity"] == "CHANGE" and proposed is None:
                raise ReconciliationError(
                    f"CHANGE lacks proposal: {path.name}:{line_number}"
                )
            if proposed is not None:
                validate_annotation(
                    proposed,
                    item["reviewContent"],
                    context=f"{path.name}:{line_number} proposed",
                )
            key = (path.name, position)
            if key in audits:
                raise ReconciliationError(f"Duplicate audit key: {key}")
            audits[key] = {
                **row,
                "audit_file": path.name,
                "audit_line": line_number,
            }
    return audits


def load_decisions(
    path: Path,
    audits: Mapping[tuple[str, int], Mapping[str, Any]],
    assignment: Mapping[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    decisions: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), 1):
        require_exact_keys(
            row,
            DECISION_FIELDS,
            context=f"{path.name}:{line_number}",
        )
        key = (row.get("audit_file"), row.get("assignment_position"))
        if key not in audits or key in decisions:
            raise ReconciliationError(
                f"Unknown/duplicate decision key: {key}"
            )
        audit = audits[key]
        if row.get("annotation_id") != audit["annotation_id"]:
            raise ReconciliationError(
                f"Decision identity mismatch: {key}"
            )
        decision = row.get("decision")
        if decision not in DECISIONS:
            raise ReconciliationError(
                f"Invalid decision {decision!r}: {key}"
            )
        if (
            not isinstance(row.get("rationale"), str)
            or not row["rationale"].strip()
        ):
            raise ReconciliationError(f"Decision rationale missing: {key}")
        replacement = row.get("replacement_annotation")
        if decision == "APPLY_PROPOSED":
            if audit["proposed_annotation"] is None or replacement is not None:
                raise ReconciliationError(
                    f"APPLY_PROPOSED contract failed: {key}"
                )
        elif decision == "APPLY_CUSTOM":
            if replacement is None:
                raise ReconciliationError(
                    f"APPLY_CUSTOM lacks replacement: {key}"
                )
        elif replacement is not None:
            raise ReconciliationError(
                f"{decision} must not include replacement: {key}"
            )
        if replacement is not None:
            item = assignment["records"][row["assignment_position"] - 1]
            validate_annotation(
                replacement,
                item["reviewContent"],
                context=f"{path.name}:{line_number} replacement",
            )
        decisions[key] = row
    if set(decisions) != set(audits):
        raise ReconciliationError(
            "Decision coverage mismatch; "
            f"missing={sorted(set(audits) - set(decisions))}, "
            f"extra={sorted(set(decisions) - set(audits))}"
        )
    return decisions


def reconcile(
    *,
    assignment_path: Path,
    batch_paths: list[Path],
    audit_paths: list[Path],
    decisions_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    assignment = verify_assignment(load_json(assignment_path))
    raw_rows = load_raw_batches(batch_paths, assignment)
    audits = load_audits(audit_paths, assignment)
    decisions = load_decisions(
        decisions_path,
        audits,
        assignment,
    )
    raw_payload_by_position: dict[int, dict[str, Any]] = {}
    for batch_path in batch_paths:
        for raw in read_jsonl(batch_path):
            position = raw["assignment_position"]
            if position in raw_payload_by_position:
                raise ReconciliationError(
                    f"Duplicate raw position while preserving payload: "
                    f"{position}"
                )
            raw_payload_by_position[position] = raw
    final_by_position = {
        row["assignment_position"]: {
            "assignment_position": row["assignment_position"],
            "annotation_id": row["annotation_id"],
            "review_text_sha256": row["review_text_sha256"],
            "annotation": raw_payload_by_position[
                row["assignment_position"]
            ]["annotation"],
        }
        for row in raw_rows
    }
    audit_results: list[dict[str, Any]] = []
    position_changes: Counter[int] = Counter()
    for key in sorted(audits, key=lambda item: (item[1], item[0])):
        audit = audits[key]
        decision = decisions[key]
        position = audit["assignment_position"]
        item = assignment["records"][position - 1]
        before = final_by_position[position]["annotation"]
        if decision["decision"] == "APPLY_PROPOSED":
            after = audit["proposed_annotation"]
        elif decision["decision"] == "APPLY_CUSTOM":
            after = decision["replacement_annotation"]
        else:
            after = before
        before_normalized = validate_annotation(
            before,
            item["reviewContent"],
            context=f"position {position} before",
        )
        after_normalized = validate_annotation(
            after,
            item["reviewContent"],
            context=f"position {position} after",
        )
        changed = (
            canonical_json(before_normalized)
            != canonical_json(after_normalized)
        )
        if decision["decision"].startswith("APPLY_") and not changed:
            raise ReconciliationError(
                f"Applied decision has no semantic/evidence change: {key}"
            )
        if not decision["decision"].startswith("APPLY_") and changed:
            raise ReconciliationError(
                f"Retained/deferred decision unexpectedly changed: {key}"
            )
        if changed:
            position_changes[position] += 1
            if position_changes[position] > 1:
                raise ReconciliationError(
                    f"Multiple applied changes target position {position}"
                )
            final_by_position[position]["annotation"] = after
        audit_results.append(
            {
                "audit_file": audit["audit_file"],
                "audit_line": audit["audit_line"],
                "assignment_position": position,
                "annotation_id": audit["annotation_id"],
                "severity": audit["severity"],
                "decision": decision["decision"],
                "decision_rationale": decision["rationale"],
                "changed": changed,
                "before_status": before_normalized["annotation_status"],
                "after_status": after_normalized["annotation_status"],
                "before_labels": list(label_vector(before_normalized)),
                "after_labels": list(label_vector(after_normalized)),
                "before_fingerprint": annotation_fingerprint(
                    before_normalized
                ),
                "after_fingerprint": annotation_fingerprint(
                    after_normalized
                ),
            }
        )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        by_source: dict[str, list[dict[str, Any]]] = {
            path.name: []
            for path in batch_paths
        }
        source_for_position = {
            row["assignment_position"]: row["source_batch"]
            for row in raw_rows
        }
        for position in sorted(final_by_position):
            source = source_for_position[position]
            by_source[source].append(final_by_position[position])
        reconciled_paths: list[Path] = []
        for source_path in batch_paths:
            name = source_path.name.replace(
                "batch_",
                "reconciled_",
                1,
            )
            target = temporary / name
            target.write_text(
                "".join(
                    f"{canonical_json(row)}\n"
                    for row in by_source[source_path.name]
                ),
                encoding="utf-8",
                newline="\n",
            )
            reconciled_paths.append(target)
        report = {
            "schema_version": "human-absa-cross-audit-reconciliation/1.0.0",
            "records": assignment["item_count"],
            "audit_items": len(audits),
            "decision_counts": dict(
                sorted(
                    Counter(
                        row["decision"]
                        for row in decisions.values()
                    ).items()
                )
            ),
            "records_changed": len(position_changes),
            "changed_positions": sorted(position_changes),
            "results": audit_results,
        }
        report_path = temporary / "reconciliation_report.json"
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
        checksum_targets = [*reconciled_paths, report_path]
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in sorted(
                    checksum_targets,
                    key=lambda item: item.name,
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "records": assignment["item_count"],
        "audit_items": len(audits),
        "records_changed": len(position_changes),
        "changed_positions": sorted(position_changes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply explicit audit decisions to new annotation batches."
    )
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument(
        "--batch",
        dest="batches",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--audit",
        dest="audits",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = reconcile(
        assignment_path=args.assignment.resolve(),
        batch_paths=[path.resolve() for path in args.batches],
        audit_paths=[path.resolve() for path in args.audits],
        decisions_path=args.decisions.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
