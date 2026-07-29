"""Publish a complete primary run as an auditable ABSA pseudo-label release.

This command is intentionally fail-closed: it writes nothing unless the
prepared package, leakage exclusions, diagnostic gate, and all primary run
records validate.  The release remains explicitly pending human verification.
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
from typing import Any, Mapping

from lazada_collector.ai_tranche import count_labels, sha256_file
from lazada_collector.llm_annotation import canonical_json

try:
    from scripts.validate_ai_annotation_tranche import (
        ARTIFACT_STATUS,
        CSV_FIELDS,
        DECISION_LEDGER_SCHEMA_VERSION,
        FINAL_MANIFEST_SCHEMA_VERSION,
        FINAL_SUMMARY_SCHEMA_VERSION,
        HUMAN_QUEUE_SCHEMA_VERSION,
        HUMAN_VERIFICATION_STATUS,
        PSEUDO_RECORD_SCHEMA_VERSION,
        TERMINAL_STATE,
        TRANCHE_SIZE,
        _load_primary_records,
        _validate_prepared_package,
        deterministic_stratified_audit_plan,
        pseudo_record_sha256,
        queue_priority,
        release_provenance_source_plan,
        risk_flags,
        run_execution_config,
        stable_queue_id,
        validate_release,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from validate_ai_annotation_tranche import (  # type: ignore[no-redef]
        ARTIFACT_STATUS,
        CSV_FIELDS,
        DECISION_LEDGER_SCHEMA_VERSION,
        FINAL_MANIFEST_SCHEMA_VERSION,
        FINAL_SUMMARY_SCHEMA_VERSION,
        HUMAN_QUEUE_SCHEMA_VERSION,
        HUMAN_VERIFICATION_STATUS,
        PSEUDO_RECORD_SCHEMA_VERSION,
        TERMINAL_STATE,
        TRANCHE_SIZE,
        _load_primary_records,
        _validate_prepared_package,
        deterministic_stratified_audit_plan,
        pseudo_record_sha256,
        queue_priority,
        release_provenance_source_plan,
        risk_flags,
        run_execution_config,
        stable_queue_id,
        validate_release,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def _artifact(
    path: Path,
    root: Path,
    *,
    records: int | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        item["records"] = records
    return item


def _source_metadata(private: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: private.get(key)
        for key in (
            "source_release_id",
            "parent_canonical_row",
            "curation_status",
            "category",
            "rating",
            "collection_transport",
            "product_id",
        )
    }


def _write_compatibility_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            labels = {
                aspect["aspect"]: (
                    "" if aspect["label"] is None else str(aspect["label"])
                )
                for aspect in row["annotation"]["aspects"]
            }
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "annotation_id": row["annotation_id"],
                    "selection_rank": row["selection_rank"],
                    "reviewContent": row["reviewContent"],
                    "annotation_status": row["annotation"][
                        "annotation_status"
                    ],
                    **labels,
                    "artifact_status": ARTIFACT_STATUS,
                    "human_verification_status": (
                        HUMAN_VERIFICATION_STATUS
                    ),
                    "review_text_sha256": row["review_text_sha256"],
                }
            )


def finalize(
    *,
    package: Path,
    output: Path,
    expected_records: int = TRANCHE_SIZE,
) -> dict[str, Any]:
    """Create and independently validate a versioned pseudo-label release."""

    package = package.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if expected_records <= 0:
        raise ValueError("expected_records must be positive")

    context = _validate_prepared_package(
        package,
        expected_records=expected_records,
    )
    run_records, run_summary, run_hashes = _load_primary_records(
        package,
        context,
        expected_records=expected_records,
    )
    if len(run_records) != expected_records:
        raise ValueError(
            f"Primary run is incomplete: {len(run_records)}/{expected_records}"
        )
    run_by_id = {row["annotation_id"]: row for row in run_records}
    tranche_id = context["manifest"]["tranche_id"]

    pseudo_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    risk_counts: Counter[str] = Counter()
    audit_reason_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()

    for blind in context["blind_rows"]:
        annotation_id = blind["annotation_id"]
        private = context["private_by_id"][annotation_id]
        run = run_by_id[annotation_id]
        pseudo = {
            "schema_version": PSEUDO_RECORD_SCHEMA_VERSION,
            "artifact_status": ARTIFACT_STATUS,
            "human_verification_status": HUMAN_VERIFICATION_STATUS,
            "tranche_id": tranche_id,
            "selection_rank": blind["selection_rank"],
            "annotation_id": annotation_id,
            "sample_id": private["sample_id"],
            "reviewContent": blind["reviewContent"],
            "review_text_sha256": blind["review_text_sha256"],
            "source": _source_metadata(private),
            "annotation": run["annotation"],
            "normalization_repairs": run["normalization_repairs"],
            "generation": run["generation"],
        }
        pseudo_rows.append(pseudo)

    audit_plan = deterministic_stratified_audit_plan(
        pseudo_rows,
        tranche_id=tranche_id,
    )
    for pseudo in pseudo_rows:
        annotation_id = pseudo["annotation_id"]
        private = context["private_by_id"][annotation_id]
        run = run_by_id[annotation_id]
        flags = risk_flags(pseudo)
        audit_reasons = audit_plan.get(annotation_id, [])
        queued = bool(flags) or bool(audit_reasons)
        ledger = {
            "schema_version": DECISION_LEDGER_SCHEMA_VERSION,
            "artifact_status": ARTIFACT_STATUS,
            "annotation_id": annotation_id,
            "sample_id": private["sample_id"],
            "selection_rank": pseudo["selection_rank"],
            "review_text_sha256": pseudo["review_text_sha256"],
            "primary_run_record_sha256": run_hashes[annotation_id],
            "pseudo_record_sha256": pseudo_record_sha256(pseudo),
            "validation_decision": (
                "ACCEPT_SCHEMA_VALID_AI_PSEUDO_LABEL"
            ),
            "terminal_state": TERMINAL_STATE,
            "risk_flags": flags,
            "deterministic_audit_10_percent": (
                "DETERMINISTIC_RANDOM_AUDIT_10_PERCENT"
                in audit_reasons
            ),
            "deterministic_audit_reasons": audit_reasons,
            "human_review_queued": queued,
        }
        ledger_rows.append(ledger)
        risk_counts.update(flags)
        audit_reason_counts.update(audit_reasons)
        terminal_counts[TERMINAL_STATE] += 1
        if queued:
            queue_rows.append(
                {
                    "schema_version": HUMAN_QUEUE_SCHEMA_VERSION,
                    "artifact_status": ARTIFACT_STATUS,
                    "queue_id": stable_queue_id(
                        tranche_id,
                        annotation_id,
                    ),
                    "annotation_id": annotation_id,
                    "sample_id": private["sample_id"],
                    "selection_rank": pseudo["selection_rank"],
                    "priority": queue_priority(
                        flags,
                        audit_only=not flags,
                    ),
                    "queue_reasons": flags + audit_reasons,
                    "human_verification_status": (
                        HUMAN_VERIFICATION_STATUS
                    ),
                    "reviewContent": pseudo["reviewContent"],
                    "review_text_sha256": pseudo["review_text_sha256"],
                    "ai_annotation": run["annotation"],
                }
            )

    priority_order = {"HIGH": 0, "MEDIUM": 1, "AUDIT": 2}
    queue_rows.sort(
        key=lambda row: (
            priority_order[row["priority"]],
            row["selection_rank"],
        )
    )
    summary = {
        "schema_version": FINAL_SUMMARY_SCHEMA_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "target_records": expected_records,
        "pseudo_label_records": len(pseudo_rows),
        "human_review_queue_records": len(queue_rows),
        "human_verification_completed": 0,
        "terminal_state_counts": dict(sorted(terminal_counts.items())),
        "risk_flag_counts": dict(sorted(risk_counts.items())),
        "deterministic_audit_reason_counts": dict(
            sorted(audit_reason_counts.items())
        ),
        "label_summary": count_labels(pseudo_rows),
        "review_queue_policy": {
            "mandatory": (
                "All ESCALATE, REJECT_NON_REVIEW, normalized-repair, "
                "neutral, mixed, all-absent, many-mentioned, uncertain, "
                "marketing-like, non-empty-note, cross-aspect reused-span, "
                "or overlong-evidence records."
            ),
            "audit": (
                "Deterministic SHA-256 10% sample plus explicit coverage "
                "for each observed rating, category, transport, and rare "
                "aspect-label stratum."
            ),
            "human_decisions_imported": False,
        },
        "interpretation": (
            "Schema-valid AI pseudo-labels only; no record is human-verified "
            "by this publication step."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        pseudo_path = temporary / "ai_pseudo_labels.jsonl"
        csv_path = temporary / "ai_pseudo_labels_compat.csv"
        ledger_path = temporary / "decision_ledger.jsonl"
        queue_path = temporary / "human_review_queue.jsonl"
        summary_path = temporary / "summary.json"
        _write_jsonl(pseudo_path, pseudo_rows)
        _write_compatibility_csv(csv_path, pseudo_rows)
        _write_jsonl(ledger_path, ledger_rows)
        _write_jsonl(queue_path, queue_rows)
        _write_json(summary_path, summary)

        provenance_plan = release_provenance_source_plan(
            package,
            context,
        )
        for relative, source in sorted(provenance_plan.items()):
            destination = temporary.joinpath(*Path(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        artifacts = [
            _artifact(pseudo_path, temporary, records=len(pseudo_rows)),
            _artifact(csv_path, temporary, records=len(pseudo_rows)),
            _artifact(ledger_path, temporary, records=len(ledger_rows)),
            _artifact(queue_path, temporary, records=len(queue_rows)),
            _artifact(summary_path, temporary),
        ]
        artifacts.extend(
            _artifact(
                temporary.joinpath(*Path(relative).parts),
                temporary,
            )
            for relative in sorted(provenance_plan)
        )
        diagnostic_summary = context["_diagnostic_summary"]
        audit_manifest_path = (
            package
            / "audits"
            / "semantic_audit_60_v1"
            / "manifest.json"
        )
        audit_sums_path = audit_manifest_path.with_name("SHA256SUMS.txt")
        audit_manifest = json.loads(
            audit_manifest_path.read_text(encoding="utf-8")
        )
        manifest = {
            "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_STATUS,
            "status": ARTIFACT_STATUS,
            "tranche_id": tranche_id,
            "target_records": expected_records,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "prepared_package": {
                "logical_location": "provenance/prepared",
                "prepare_manifest_sha256": sha256_file(
                    package / "prepare_manifest.json"
                ),
                "input_checksums_sha256": sha256_file(
                    package / "INPUT_SHA256SUMS.txt"
                ),
                "ordered_membership_sha256": context["manifest"][
                    "selection"
                ]["ordered_membership_sha256"],
                "source_release_id": context["manifest"][
                    "source_release"
                ]["release_id"],
                "human_reference_manifest_sha256": context["manifest"][
                    "selection"
                ]["human_reference_manifest_sha256"],
                "group_reservations_sha256": context["manifest"][
                    "selection"
                ]["group_reservations_sha256"],
            },
            "diagnostic_run": {
                "summary": {
                    "path": "provenance/diagnostic/run_summary.json",
                    "sha256": sha256_file(
                        package
                        / "runs"
                        / "diagnostic"
                        / "run_summary.json"
                    ),
                },
                "metrics": {
                    "path": (
                        "provenance/diagnostic/diagnostic_metrics.json"
                    ),
                    "sha256": sha256_file(
                        package
                        / "calibration"
                        / "diagnostic_metrics.json"
                    ),
                },
                "gate": {
                    "path": "provenance/diagnostic/diagnostic_gate.json",
                    "sha256": sha256_file(
                        package
                        / "calibration"
                        / "diagnostic_gate.json"
                    ),
                    "status": "PASS",
                },
                "sealed_manifest": {
                    "path": "provenance/diagnostic/run_manifest.json",
                    "sha256": sha256_file(
                        package
                        / "runs"
                        / "diagnostic"
                        / "run_manifest.json"
                    ),
                },
                "sealed_checksums": {
                    "path": (
                        "provenance/diagnostic/RUN_SHA256SUMS.txt"
                    ),
                    "sha256": sha256_file(
                        package
                        / "runs"
                        / "diagnostic"
                        / "RUN_SHA256SUMS.txt"
                    ),
                },
                "execution_config": run_execution_config(
                    diagnostic_summary
                ),
            },
            "primary_run": {
                "summary": {
                    "path": "provenance/primary/run_summary.json",
                    "sha256": sha256_file(
                        package
                        / "runs"
                        / "primary"
                        / "run_summary.json"
                    ),
                },
                "sealed_manifest": {
                    "path": "provenance/primary/run_manifest.json",
                    "sha256": sha256_file(
                        package
                        / "runs"
                        / "primary"
                        / "run_manifest.json"
                    ),
                },
                "sealed_checksums": {
                    "path": "provenance/primary/RUN_SHA256SUMS.txt",
                    "sha256": sha256_file(
                        package
                        / "runs"
                        / "primary"
                        / "RUN_SHA256SUMS.txt"
                    ),
                },
                "execution_config": run_execution_config(run_summary),
            },
            "semantic_audit": {
                "artifact_type": "AI_SEMANTIC_AUDIT_NOT_HUMAN_GOLD",
                "status": audit_manifest["status"],
                "human_accuracy_claim_permitted": False,
                "manifest": {
                    "path": "provenance/semantic_audit/manifest.json",
                    "sha256": sha256_file(audit_manifest_path),
                },
                "checksums": {
                    "path": (
                        "provenance/semantic_audit/SHA256SUMS.txt"
                    ),
                    "sha256": sha256_file(audit_sums_path),
                },
            },
            "human_verification": {
                "status": HUMAN_VERIFICATION_STATUS,
                "completed_records": 0,
                "queue_records": len(queue_rows),
            },
            "software": {
                "finalizer": {
                    "path": (
                        "provenance/software/"
                        "finalize_ai_annotation_tranche.py"
                    ),
                    "sha256": sha256_file(
                        temporary
                        / "provenance"
                        / "software"
                        / "finalize_ai_annotation_tranche.py"
                    ),
                },
                "validator": {
                    "path": (
                        "provenance/software/"
                        "validate_ai_annotation_tranche.py"
                    ),
                    "sha256": sha256_file(
                        temporary
                        / "provenance"
                        / "software"
                        / "validate_ai_annotation_tranche.py"
                    ),
                },
                "run_sealer": {
                    "path": (
                        "provenance/software/"
                        "seal_ai_annotation_runs.py"
                    ),
                    "sha256": sha256_file(
                        temporary
                        / "provenance"
                        / "software"
                        / "seal_ai_annotation_runs.py"
                    ),
                },
            },
            "limitations": [
                "All annotations are AI pseudo-labels pending human review.",
                "A PASS diagnostic gate measures alignment to an AI-assisted "
                "human-confirmed holdout; it is not an independent accuracy "
                "or inter-annotator-agreement estimate.",
                "The compatibility CSV is a projection; JSONL is canonical "
                "because it preserves evidence, uncertainty, and provenance.",
                "The bundled AI semantic audit is not human accuracy evidence "
                "and remains pending independent human adjudication.",
            ],
            "artifacts": sorted(artifacts, key=lambda row: row["path"]),
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        sums = [
            (item["sha256"], item["path"]) for item in artifacts
        ] + [(sha256_file(manifest_path), "manifest.json")]
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(
                f"{checksum}  {relative}\n"
                for checksum, relative in sorted(
                    sums,
                    key=lambda item: item[1],
                )
            ),
            encoding="utf-8",
            newline="\n",
        )

        validation = validate_release(
            package=package,
            release=temporary,
            expected_records=expected_records,
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": ARTIFACT_STATUS,
        "records": expected_records,
        "human_review_queue_records": len(queue_rows),
        "output": str(output),
        "validation": validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("data/annotations/absa_ai_tranche_5000_v1_20260727"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/annotations/absa_ai_tranche_5000_v1_20260727/final"
        ),
    )
    parser.add_argument(
        "--expected-records",
        type=int,
        default=None,
        help=(
            "Expected record count. Defaults to target_records in the "
            "prepared manifest."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected_records = args.expected_records
    if expected_records is None:
        prepared_manifest = json.loads(
            (args.package / "prepare_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected_records = prepared_manifest.get("target_records")
    if (
        not isinstance(expected_records, int)
        or isinstance(expected_records, bool)
        or expected_records <= 0
    ):
        raise ValueError("Prepared target_records is invalid")
    result = finalize(
        package=args.package,
        output=args.output,
        expected_records=expected_records,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
