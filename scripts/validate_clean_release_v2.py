"""Fail-closed validation for an ABSA curation V2 release."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable

from lazada_collector.curation import (
    internal_ngram_repetition_evidence,
    keyboard_token_document_frequencies,
    language_artifact_flags,
    sha256_text,
    structural_catalogue_evidence,
)
from lazada_collector.quality import QualityPolicy, evaluate_review


EXPECTED_STATUSES = {
    "KEEP",
    "KEEP_CLEANED",
    "QUARANTINE",
    "EXCLUDE_AUTO",
}
QUARANTINE_REQUIRED_REASON_CODES = {
    "QC_AUDIT_BORDERLINE",
    "QC_AUDIT_MIXED_NOISY",
    "QC_AUDIT_NONREVIEW",
    "QC_AUDIT_PURE_TEMPLATE",
    "STRUCTURAL_CATALOGUE_NO_EXPERIENCE",
    "TEMPLATE_GLOBAL_CANDIDATE",
    "TEMPLATE_GLOBAL_HIGH",
    "TEMPLATE_PRODUCT_HIGH",
}
ASPECT_COLUMNS = [
    "Chất lượng sản phẩm",
    "Hiệu năng & Trải nghiệm",
    "Đúng mô tả",
    "Giá cả & Khuyến mãi",
    "Vận chuyển",
    "Đóng gói",
    "Dịch vụ & Thái độ Shop",
    "Bảo hành & Đổi trả",
    "Tính xác thực",
]
ANNOTATION_COLUMNS = ["reviewContent", *ASPECT_COLUMNS]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_CURATION_FIELDS = {
    "curation_schema_version",
    "parent_release_id",
    "parent_canonical_row",
    "sample_id",
    "product_id",
    "review_id",
    "rule_version",
    "status",
    "primary_reason",
    "exclusion_trigger",
    "reason_codes",
    "decision_source",
    "annotation_eligible",
    "raw_text_sha256",
    "curated_review_text",
    "curated_text_sha256",
    "transformation_ids",
    "duplicate_cluster_id",
    "representative_sample_id",
    "near_duplicate_cluster_id",
    "near_duplicate_representative_sample_id",
    "template_family_id",
    "template_evidence",
    "structural_template_evidence",
    "internal_repetition_evidence",
    "post_clean_quality",
    "product_cap",
    "source_relative_path",
    "source_line",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row is not an object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV lacks a header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _checksum_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("  ", 1)
            if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
                raise ValueError(
                    f"Invalid checksum at {path}:{line_number}"
                )
            relative = parts[1].replace("\\", "/")
            if relative in values:
                raise ValueError(f"Duplicate checksum path: {relative}")
            values[relative] = parts[0]
    return values


def _verify_checksum_closure(root: Path) -> dict[str, str]:
    expected = _checksum_map(root / "SHA256SUMS.txt")
    physical = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if set(expected) != physical:
        missing = sorted(physical - set(expected))
        extra = sorted(set(expected) - physical)
        raise ValueError(
            f"Checksum closure mismatch; missing={missing}, extra={extra}"
        )
    for relative, digest in sorted(expected.items()):
        actual = _sha256_file(root / relative)
        if actual != digest:
            raise ValueError(
                f"Checksum mismatch for {relative}: {actual} != {digest}"
            )
    return expected


def _quality_policy(config: dict[str, Any]) -> QualityPolicy:
    values = config["post_clean_quality"]
    return QualityPolicy(
        min_chars=int(values["min_chars"]),
        min_words=int(values["min_words"]),
        min_unique_word_ratio=float(values["min_unique_word_ratio"]),
        min_meaningful_words=int(values["min_meaning_words"])
        if "min_meaning_words" in values
        else int(values["min_meaningful_words"]),
        min_score=float(values["min_score"]),
        require_vietnamese=bool(values["require_vietnamese"]),
        min_vietnamese_signals=int(values["min_vietnamese_signals"]),
        max_foreign_script_ratio=float(values["max_foreign_script_ratio"]),
        reject_suspect_encoding=bool(values["reject_suspect_encoding"]),
    )


def _quality_dict(result: Any) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "score": result.score,
        "char_count": result.char_count,
        "word_count": result.word_count,
        "unique_word_ratio": result.unique_word_ratio,
        "meaningful_word_count": result.meaningful_word_count,
        "vietnamese_signal_count": result.vietnamese_signal_count,
        "foreign_script_ratio": result.foreign_script_ratio,
        "reasons": list(result.reasons),
    }


def _record_counts(path: Path) -> int | None:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        return len(_read_jsonl(path))
    if suffix == ".csv":
        return len(_read_csv(path)[1])
    return None


def _validate_artifact_manifest(
    root: Path,
    manifest: dict[str, Any],
) -> None:
    seen = set()
    for artifact in manifest["artifacts"]:
        relative = str(artifact["path"])
        if relative in seen:
            raise ValueError(f"Duplicate manifest artifact: {relative}")
        seen.add(relative)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest artifact missing: {path}")
        if path.stat().st_size != int(artifact["bytes"]):
            raise ValueError(f"Manifest byte count mismatch: {relative}")
        if _sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Manifest hash mismatch: {relative}")
        if "records" in artifact:
            actual_records = _record_counts(path)
            if actual_records != int(artifact["records"]):
                raise ValueError(
                    f"Manifest record count mismatch for {relative}: "
                    f"{actual_records} != {artifact['records']}"
                )


def _validate_transformations(
    curation_by_id: dict[str, dict[str, Any]],
    parent_by_id: dict[str, dict[str, Any]],
    transformations: list[dict[str, Any]],
) -> None:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transformation_ids = set()
    for row in transformations:
        transformation_id = str(row.get("transformation_id") or "")
        sample_id = str(row.get("sample_id") or "")
        if not transformation_id or transformation_id in transformation_ids:
            raise ValueError("Blank or duplicate transformation_id")
        if sample_id not in curation_by_id:
            raise ValueError(f"Transformation references unknown sample: {sample_id}")
        if "removed_text" in json.dumps(row, ensure_ascii=False):
            raise ValueError(
                f"Transformation leaks removed text: {transformation_id}"
            )
        transformation_ids.add(transformation_id)
        by_sample[sample_id].append(row)

    for sample_id, curation in curation_by_id.items():
        rows = sorted(
            by_sample.get(sample_id, []),
            key=lambda row: int(row["sequence"]),
        )
        expected_ids = list(curation["transformation_ids"])
        actual_ids = [str(row["transformation_id"]) for row in rows]
        if actual_ids != expected_ids:
            raise ValueError(
                f"Transformation ID/order mismatch for {sample_id}"
            )
        if [int(row["sequence"]) for row in rows] != list(
            range(1, len(rows) + 1)
        ):
            raise ValueError(
                f"Non-contiguous transformation sequence for {sample_id}"
            )
        current_hash = sha256_text(parent_by_id[sample_id]["review_text"])
        for row in rows:
            if row["input_text_sha256"] != current_hash:
                raise ValueError(
                    f"Broken transformation input chain for {sample_id}"
                )
            current_hash = str(row["output_text_sha256"])
        if current_hash != curation["curated_text_sha256"]:
            raise ValueError(
                f"Broken transformation output chain for {sample_id}"
            )


def _validate_annotation(
    root: Path,
    curation_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, int]:
    schema = json.loads(
        (root / "annotation" / "schema.json").read_text(encoding="utf-8")
    )
    if schema["review_column"] != "reviewContent":
        raise ValueError("Annotation schema has wrong review column")
    if schema["aspect_columns"] != ASPECT_COLUMNS:
        raise ValueError("Annotation schema aspect order mismatch")
    if schema["allowed_labels"] != config["annotation"]["allowed_labels"]:
        raise ValueError("Annotation allowed labels differ from config")
    if schema["unlabeled_representation"] != "blank":
        raise ValueError("Annotation blank semantics are not explicit")

    pilot_header, pilot = _read_csv(
        root / "annotation" / "pilot_candidate.csv"
    )
    if pilot_header != ANNOTATION_COLUMNS:
        raise ValueError("Pilot CSV does not use exact 10-column schema")
    pilot_texts = []
    for row_number, row in enumerate(pilot, 2):
        if any(row[column] != "" for column in ASPECT_COLUMNS):
            raise ValueError(f"Pilot label is not blank at row {row_number}")
        pilot_texts.append(row["reviewContent"])

    index_header, index = _read_csv(root / "annotation" / "index.csv")
    expected_index_header = [
        "sample_id",
        "curated_text_sha256",
        "package_role",
        "batch_file",
        "batch_data_row",
        "parent_canonical_row",
        "curation_status",
        "rule_version",
        "guideline_version",
    ]
    if index_header != expected_index_header:
        raise ValueError("Annotation index header mismatch")
    clean_ids = {
        sample_id
        for sample_id, row in curation_by_id.items()
        if row["annotation_eligible"]
    }
    index_ids = [row["sample_id"] for row in index]
    if len(index_ids) != len(set(index_ids)) or set(index_ids) != clean_ids:
        raise ValueError("Annotation index is not one-to-one with clean-core")
    pilot_ids = set()
    occupied_positions = set()
    for row in index:
        sample_id = row["sample_id"]
        curation = curation_by_id[sample_id]
        if row["curated_text_sha256"] != curation["curated_text_sha256"]:
            raise ValueError(f"Annotation index hash mismatch: {sample_id}")
        if row["curation_status"] not in {"KEEP", "KEEP_CLEANED"}:
            raise ValueError(f"Non-clean row in annotation index: {sample_id}")
        role = row["package_role"]
        if role == "PILOT_CANDIDATE":
            if row["batch_file"] != "annotation/pilot_candidate.csv":
                raise ValueError(f"Pilot file mapping mismatch: {sample_id}")
            data_row = int(row["batch_data_row"])
            position = data_row - 2
            if position < 0 or position >= len(pilot):
                raise ValueError(f"Pilot row is out of range: {sample_id}")
            if position in occupied_positions:
                raise ValueError("Duplicate pilot row position")
            occupied_positions.add(position)
            if (
                sha256_text(pilot[position]["reviewContent"])
                != curation["curated_text_sha256"]
            ):
                raise ValueError(f"Pilot text/hash mismatch: {sample_id}")
            pilot_ids.add(sample_id)
        elif role == "PENDING_MAIN_AFTER_PILOT_GATE":
            if row["batch_file"] or row["batch_data_row"]:
                raise ValueError(f"Pending main row has batch mapping: {sample_id}")
        else:
            raise ValueError(f"Unknown annotation package role: {role}")
    if len(pilot_ids) != len(pilot) or occupied_positions != set(
        range(len(pilot))
    ):
        raise ValueError("Pilot/index mapping is not bijective")

    audit_header, audit = _read_csv(
        root / "annotation" / "curation_audit_1000.csv"
    )
    if audit_header != [
        "audit_id",
        "reviewContent",
        "curation_decision",
        "reviewer_id",
        "notes",
    ]:
        raise ValueError("Clean-core audit header mismatch")
    audit_index_header, audit_index = _read_csv(
        root / "annotation" / "curation_audit_index.csv"
    )
    if audit_index_header != [
        "audit_id",
        "sample_id",
        "category",
        "rating",
        "collection_transport",
        "curated_text_sha256",
    ]:
        raise ValueError("Clean-core audit index header mismatch")
    audit_by_id = {row["audit_id"]: row for row in audit}
    audit_index_by_id = {row["audit_id"]: row for row in audit_index}
    if (
        len(audit_by_id) != len(audit)
        or len(audit_index_by_id) != len(audit_index)
        or set(audit_by_id) != set(audit_index_by_id)
    ):
        raise ValueError("Clean-core audit mapping is not one-to-one")
    for audit_id, row in audit_by_id.items():
        metadata = audit_index_by_id[audit_id]
        sample_id = metadata["sample_id"]
        if sample_id not in clean_ids:
            raise ValueError("Audit includes non-clean sample")
        expected_hash = curation_by_id[sample_id]["curated_text_sha256"]
        if sha256_text(row["reviewContent"]) != expected_hash:
            raise ValueError(f"Audit text/hash mismatch: {audit_id}")
        if metadata["curated_text_sha256"] != expected_hash:
            raise ValueError(f"Audit index hash mismatch: {audit_id}")
        if (
            row["curation_decision"]
            or row["reviewer_id"]
            or row["notes"]
        ):
            raise ValueError("Input clean-core audit contains completed fields")

    calibration_header, calibration = _read_csv(
        root / "annotation" / "template_calibration.csv"
    )
    if calibration_header != [
        "calibration_id",
        "reviewContent",
        "composition_decision",
        "nonreview_decision",
        "reviewer_id",
        "notes",
    ]:
        raise ValueError("Template calibration header mismatch")
    calibration_index_header, calibration_index = _read_csv(
        root / "annotation" / "template_calibration_index.csv"
    )
    if calibration_index_header != [
        "calibration_id",
        "sample_id",
        "stratum",
        "curation_status",
        "curated_text_sha256",
        "global_recurrent_clause_count",
        "global_template_coverage",
        "product_recurrent_clause_count",
        "product_template_coverage",
        "template_family_id",
    ]:
        raise ValueError("Template calibration index header mismatch")
    calibration_by_id = {
        row["calibration_id"]: row for row in calibration
    }
    calibration_index_by_id = {
        row["calibration_id"]: row for row in calibration_index
    }
    if (
        len(calibration_by_id) != len(calibration)
        or len(calibration_index_by_id) != len(calibration_index)
        or set(calibration_by_id) != set(calibration_index_by_id)
    ):
        raise ValueError("Template calibration mapping is not one-to-one")
    calibration_strata = Counter()
    for calibration_id, row in calibration_by_id.items():
        metadata = calibration_index_by_id[calibration_id]
        sample_id = metadata["sample_id"]
        if sample_id not in curation_by_id:
            raise ValueError("Template calibration references unknown sample")
        curation = curation_by_id[sample_id]
        reasons = set(curation["reason_codes"])
        has_global_candidate = "TEMPLATE_GLOBAL_CANDIDATE" in reasons
        has_global_high = "TEMPLATE_GLOBAL_HIGH" in reasons
        has_product_high = "TEMPLATE_PRODUCT_HIGH" in reasons
        has_structural = (
            "STRUCTURAL_CATALOGUE_NO_EXPERIENCE" in reasons
        )
        has_global = has_global_candidate or has_global_high
        if has_global and has_product_high:
            expected_stratum = "GLOBAL_AND_PRODUCT"
        elif has_global_high:
            expected_stratum = "GLOBAL_HIGH_ONLY"
        elif has_global_candidate:
            expected_stratum = "GLOBAL_CANDIDATE_ONLY"
        elif has_product_high:
            expected_stratum = "PRODUCT_HIGH_ONLY"
        elif has_structural:
            expected_stratum = "STRUCTURAL_ONLY"
        else:
            raise ValueError("Calibration sample lacks template evidence")
        if metadata["stratum"] != expected_stratum:
            raise ValueError("Template calibration stratum mismatch")
        if curation["status"] == "EXCLUDE_AUTO":
            raise ValueError("Excluded row entered template calibration")
        expected_hash = curation["curated_text_sha256"]
        if (
            metadata["curated_text_sha256"] != expected_hash
            or sha256_text(row["reviewContent"]) != expected_hash
        ):
            raise ValueError("Template calibration text/hash mismatch")
        if any(
            row[field]
            for field in (
                "composition_decision",
                "nonreview_decision",
                "reviewer_id",
                "notes",
            )
        ):
            raise ValueError(
                "Template calibration input contains completed fields"
            )
        calibration_strata[expected_stratum] += 1
    maximum_per_stratum = int(
        config["annotation"]["template_calibration_per_stratum"]
    )
    if calibration_strata and max(calibration_strata.values()) > maximum_per_stratum:
        raise ValueError("Template calibration exceeds stratum quota")

    queue_header, queue = _read_csv(
        root / "annotation" / "curation_review_queue.csv"
    )
    if queue_header != [
        "sample_id",
        "proposed_status",
        "primary_reason",
        "reason_codes",
        "review_text_original",
        "review_text_curated",
        "human_decision",
        "human_primary_reason",
        "reviewer_id",
        "notes",
    ]:
        raise ValueError("Curation review queue header mismatch")
    flagged_ids = {
        sample_id
        for sample_id, row in curation_by_id.items()
        if row["status"] in {"QUARANTINE", "EXCLUDE_AUTO"}
    }
    queue_ids = [row["sample_id"] for row in queue]
    if len(queue_ids) != len(set(queue_ids)) or set(queue_ids) != flagged_ids:
        raise ValueError("Curation queue is not one-to-one with flagged records")
    for row in queue:
        curation = curation_by_id[row["sample_id"]]
        if row["proposed_status"] != curation["status"]:
            raise ValueError("Curation queue proposed status mismatch")
        if row["review_text_curated"] != curation["curated_review_text"]:
            raise ValueError("Curation queue curated text mismatch")
        if any(
            row[field]
            for field in (
                "human_decision",
                "human_primary_reason",
                "reviewer_id",
                "notes",
            )
        ):
            raise ValueError("Curation input queue contains completed fields")

    return {
        "annotation_eligible": len(clean_ids),
        "annotation_index": len(index),
        "pilot_candidate": len(pilot),
        "clean_core_audit": len(audit),
        "template_calibration": len(calibration),
        "flagged_human_review": len(queue),
    }


def _distribution(
    rows: Iterable[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "<blank>") for row in rows)
    return dict(sorted(counts.items()))


def validate(
    release_root: Path,
    parent_root: Path,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    parent_root = parent_root.resolve()
    checksums = _verify_checksum_closure(release_root)
    _verify_checksum_closure(parent_root)
    manifest = json.loads(
        (release_root / "manifest.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (
            release_root / "provenance" / "cleaning_v2.json"
        ).read_text(encoding="utf-8")
    )
    parent_manifest = json.loads(
        (parent_root / "manifest.json").read_text(encoding="utf-8")
    )
    if set(config["statuses"]) != EXPECTED_STATUSES:
        raise ValueError("Cleaning config status contract mismatch")
    if manifest["parent"]["release_id"] != parent_manifest["release_id"]:
        raise ValueError("Manifest parent release mismatch")
    if (
        manifest["parent"]["manifest_sha256"]
        != _sha256_file(parent_root / "manifest.json")
    ):
        raise ValueError("Parent manifest hash mismatch")
    if (
        manifest["provenance"]["config_sha256"]
        != _sha256_file(release_root / "provenance" / "cleaning_v2.json")
    ):
        raise ValueError("Cleaning config hash mismatch")
    _validate_artifact_manifest(release_root, manifest)

    parent_rows = _read_jsonl(parent_root / "reviews_canonical.jsonl")
    parent_by_id = {
        str(row["sample_id"]): row
        for row in parent_rows
    }
    if len(parent_by_id) != len(parent_rows):
        raise ValueError("Parent sample IDs are duplicated")
    curation = _read_jsonl(release_root / "curation_records.jsonl")
    curation_by_id: dict[str, dict[str, Any]] = {}
    quality_policy = _quality_policy(config)
    keyboard_document_frequency = keyboard_token_document_frequencies(
        row["curated_review_text"]
        for row in curation
    )
    for row_number, row in enumerate(curation, 1):
        missing = REQUIRED_CURATION_FIELDS - row.keys()
        if missing:
            raise ValueError(
                f"Missing curation fields at row {row_number}: {sorted(missing)}"
            )
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in curation_by_id:
            raise ValueError("Blank or duplicate curation sample_id")
        if sample_id not in parent_by_id:
            raise ValueError(f"Unknown parent sample: {sample_id}")
        parent = parent_by_id[sample_id]
        if int(row["parent_canonical_row"]) != row_number:
            raise ValueError("Parent canonical row order is not stable")
        if row["product_id"] != parent["product_id"]:
            raise ValueError(f"Product mismatch: {sample_id}")
        if row["review_id"] != parent["review_id"]:
            raise ValueError(f"Review ID mismatch: {sample_id}")
        status = str(row["status"])
        if status not in EXPECTED_STATUSES:
            raise ValueError(f"Unknown curation status: {status}")
        if row["annotation_eligible"] != (
            status in {"KEEP", "KEEP_CLEANED"}
        ):
            raise ValueError(f"Annotation eligibility mismatch: {sample_id}")
        if status == "EXCLUDE_AUTO" and not row["exclusion_trigger"]:
            raise ValueError(f"Excluded row lacks trigger: {sample_id}")
        if status != "EXCLUDE_AUTO" and row["exclusion_trigger"] is not None:
            raise ValueError(
                f"Non-excluded row has exclusion trigger: {sample_id}"
            )
        if row["raw_text_sha256"] != sha256_text(parent["review_text"]):
            raise ValueError(f"Raw text hash mismatch: {sample_id}")
        if row["curated_text_sha256"] != sha256_text(
            row["curated_review_text"]
        ):
            raise ValueError(f"Curated text hash mismatch: {sample_id}")
        reason_codes = row["reason_codes"]
        if (
            not isinstance(reason_codes, list)
            or reason_codes != sorted(set(reason_codes))
        ):
            raise ValueError(f"Reason codes are not stable/unique: {sample_id}")
        leaked_quarantine_reasons = (
            set(reason_codes) & QUARANTINE_REQUIRED_REASON_CODES
        )
        if (
            leaked_quarantine_reasons
            and status in {"KEEP", "KEEP_CLEANED"}
        ):
            raise ValueError(
                "Quarantine-required evidence entered clean-core for "
                f"{sample_id}: {sorted(leaked_quarantine_reasons)}"
            )
        result = evaluate_review(row["curated_review_text"], quality_policy)
        if row["post_clean_quality"] != _quality_dict(result):
            raise ValueError(f"Post-clean quality mismatch: {sample_id}")
        structural_config = config["structural_template"]
        structural_evidence = structural_catalogue_evidence(
            row["curated_review_text"],
            minimum_segments=int(
                structural_config["minimum_segments"]
            ),
            minimum_list_delimiters=int(
                structural_config["minimum_list_delimiters"]
            ),
            short_segment_min_tokens=int(
                structural_config["short_segment_min_tokens"]
            ),
            short_segment_max_tokens=int(
                structural_config["short_segment_max_tokens"]
            ),
            minimum_short_segment_ratio=float(
                structural_config["minimum_short_segment_ratio"]
            ),
            minimum_catalogue_openers=int(
                structural_config["minimum_catalogue_openers"]
            ),
            global_recurrent_clause_count=int(
                row["template_evidence"][
                    "global_recurrent_clause_count"
                ]
            ),
            global_template_coverage=float(
                row["template_evidence"]["global_template_coverage"]
            ),
            product_recurrent_clause_count=int(
                row["template_evidence"][
                    "product_recurrent_clause_count"
                ]
            ),
            product_template_coverage=float(
                row["template_evidence"]["product_template_coverage"]
            ),
            weak_global_min_clauses=int(
                structural_config["weak_global_min_clauses"]
            ),
            weak_global_min_coverage=float(
                structural_config["weak_global_min_coverage"]
            ),
            weak_product_min_clauses=int(
                structural_config["weak_product_min_clauses"]
            ),
            weak_product_min_coverage=float(
                structural_config["weak_product_min_coverage"]
            ),
            unique_min_marketing_ratio=float(
                structural_config["unique_min_marketing_ratio"]
            ),
            low_density_min_marketing_ratio=float(
                structural_config[
                    "low_density_min_marketing_ratio"
                ]
            ),
            modular_min_segments=int(
                structural_config["modular_min_segments"]
            ),
            modular_trailing_min_short_ratio=float(
                structural_config[
                    "modular_trailing_min_short_ratio"
                ]
            ),
            modular_cleaned_min_short_ratio=float(
                structural_config[
                    "modular_cleaned_min_short_ratio"
                ]
            ),
            modular_min_capitalized_ratio=float(
                structural_config[
                    "modular_min_capitalized_ratio"
                ]
            ),
            expanded_spec_min_clauses=int(
                structural_config["expanded_spec_min_clauses"]
            ),
            glued_min_segments=int(
                structural_config["glued_min_segments"]
            ),
            was_cleaned=bool(row["transformation_ids"]),
        )
        if row["structural_template_evidence"] != structural_evidence:
            raise ValueError(
                f"Structural-template evidence mismatch: {sample_id}"
            )
        if (
            structural_evidence["flagged"]
            and row["status"] in {"KEEP", "KEEP_CLEANED"}
        ):
            raise ValueError(
                f"Structural template did not enter quarantine: {sample_id}"
            )
        repetition_config = config["internal_repetition"]
        repetition_evidence = internal_ngram_repetition_evidence(
            row["curated_review_text"],
            ngram_size=int(repetition_config["ngram_size"]),
            minimum_repeated_ngrams=int(
                repetition_config["minimum_repeated_ngrams"]
            ),
            minimum_later_token_coverage=float(
                repetition_config["minimum_later_token_coverage"]
            ),
        )
        if row["internal_repetition_evidence"] != repetition_evidence:
            raise ValueError(
                f"Internal-repetition evidence mismatch: {sample_id}"
            )
        if (
            repetition_evidence["flagged"]
            and row["status"] in {"KEEP", "KEEP_CLEANED"}
        ):
            raise ValueError(
                f"Internal repeated sequence entered clean-core: {sample_id}"
            )
        artifact_flags = language_artifact_flags(
            row["curated_review_text"],
            token_document_frequency=keyboard_document_frequency,
        )
        missing_artifact_reasons = (
            set(artifact_flags) - set(reason_codes)
        )
        if missing_artifact_reasons:
            raise ValueError(
                "Language-artifact evidence is missing reason codes for "
                f"{sample_id}: {sorted(missing_artifact_reasons)}"
            )
        if artifact_flags and status in {"KEEP", "KEEP_CLEANED"}:
            raise ValueError(
                f"Language artifact entered clean-core: {sample_id}"
            )
        if status == "KEEP":
            if (
                row["curated_review_text"] != parent["review_text"]
                or row["transformation_ids"]
                or row["raw_text_sha256"] != row["curated_text_sha256"]
            ):
                raise ValueError(f"KEEP invariant violated: {sample_id}")
        if status == "KEEP_CLEANED":
            if (
                not row["curated_review_text"]
                or not row["transformation_ids"]
                or row["raw_text_sha256"] == row["curated_text_sha256"]
                or not result.accepted
            ):
                raise ValueError(f"KEEP_CLEANED invariant violated: {sample_id}")
        if status in {"KEEP", "KEEP_CLEANED"} and not result.accepted:
            raise ValueError(f"Non-quality row in clean-core: {sample_id}")
        curation_by_id[sample_id] = row
    if set(curation_by_id) != set(parent_by_id):
        raise ValueError("Curation records do not exactly cover the parent")

    partitions = {
        "clean_core": _read_jsonl(release_root / "clean_core.jsonl"),
        "quarantine": _read_jsonl(release_root / "quarantine.jsonl"),
        "excluded": _read_jsonl(release_root / "excluded.jsonl"),
    }
    expected_statuses = {
        "clean_core": {"KEEP", "KEEP_CLEANED"},
        "quarantine": {"QUARANTINE"},
        "excluded": {"EXCLUDE_AUTO"},
    }
    partition_ids: dict[str, set[str]] = {}
    for name, rows in partitions.items():
        ids = set()
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in ids:
                raise ValueError(f"Duplicate sample in {name}: {sample_id}")
            ids.add(sample_id)
            curation_row = curation_by_id[sample_id]
            if curation_row["status"] not in expected_statuses[name]:
                raise ValueError(f"Wrong status in {name}: {sample_id}")
            if row["review_text"] != parent_by_id[sample_id]["review_text"]:
                raise ValueError(f"Parent text changed in {name}: {sample_id}")
            if row["curated_review_text"] != curation_row[
                "curated_review_text"
            ]:
                raise ValueError(f"Curated text mismatch in {name}: {sample_id}")
            if row["curation"]["status"] != curation_row["status"]:
                raise ValueError(f"Embedded curation mismatch in {name}")
        partition_ids[name] = ids
    if any(
        partition_ids[left] & partition_ids[right]
        for left, right in (
            ("clean_core", "quarantine"),
            ("clean_core", "excluded"),
            ("quarantine", "excluded"),
        )
    ):
        raise ValueError("Curation partitions overlap")
    if set().union(*partition_ids.values()) != set(curation_by_id):
        raise ValueError("Curation partitions do not cover all records")

    transformations = _read_jsonl(
        release_root / "transformations.jsonl"
    )
    _validate_transformations(
        curation_by_id,
        parent_by_id,
        transformations,
    )

    aliases = _read_jsonl(release_root / "duplicate_aliases.jsonl")
    alias_ids = set()
    cluster_representatives: dict[str, str] = {}
    for row in aliases:
        representative = str(row["representative_sample_id"])
        alias = str(row["alias_sample_id"])
        cluster_id = str(row["cluster_id"])
        if representative == alias:
            raise ValueError("Duplicate alias points to itself")
        if representative not in curation_by_id or alias not in curation_by_id:
            raise ValueError("Duplicate alias references unknown sample")
        if alias in alias_ids:
            raise ValueError(f"Sample is duplicate alias twice: {alias}")
        alias_ids.add(alias)
        if curation_by_id[alias]["status"] != "EXCLUDE_AUTO":
            raise ValueError(f"Confirmed alias is not excluded: {alias}")
        if curation_by_id[alias]["representative_sample_id"] != representative:
            raise ValueError(f"Duplicate representative mismatch: {alias}")
        old_representative = cluster_representatives.setdefault(
            cluster_id,
            representative,
        )
        if old_representative != representative:
            raise ValueError("Duplicate cluster has multiple representatives")
        if row["match_type"] == "CROSS_TRANSPORT":
            if not row["auto_merge"]:
                raise ValueError("Unconfirmed cross candidate in alias ledger")
            duplicate_config = config["duplicate"]
            if (
                float(row["word_5gram_jaccard"])
                < float(
                    duplicate_config[
                        "cross_transport_word_5gram_jaccard"
                    ]
                )
                or float(row["length_ratio"])
                < float(
                    duplicate_config[
                        "cross_transport_min_length_ratio"
                    ]
                )
            ):
                raise ValueError("Cross-transport alias below threshold")
    if alias_ids & partition_ids["clean_core"]:
        raise ValueError("Confirmed duplicate alias leaked into clean-core")

    near = _read_jsonl(
        release_root / "near_duplicate_candidates.jsonl"
    )
    for row in near:
        left = str(row["left_sample_id"])
        right = str(row["right_sample_id"])
        representative = str(row["representative_sample_id"])
        if (
            left not in curation_by_id
            or right not in curation_by_id
            or representative not in curation_by_id
        ):
            raise ValueError("Near duplicate references unknown sample")
        if float(row["jaccard"]) <= 0 or float(row["length_ratio"]) <= 0:
            raise ValueError("Near duplicate has invalid similarity")

    for row in _read_jsonl(release_root / "template_families.jsonl"):
        if row["decision"] != "HUMAN_CALIBRATION_COMPONENT":
            raise ValueError("Template family has unsafe automatic decision")
        if len(row["member_sample_ids"]) < 2:
            raise ValueError("Template family contains fewer than two members")
        if len(row["shared_recurrent_clause_hashes"]) < 2:
            raise ValueError(
                "Template family lacks two shared recurrent clauses"
            )
        for sample_id in row["member_sample_ids"]:
            if sample_id not in curation_by_id:
                raise ValueError("Template family references unknown sample")
            if curation_by_id[sample_id]["status"] == "EXCLUDE_AUTO":
                other_reasons = set(
                    curation_by_id[sample_id]["reason_codes"]
                ) - {
                    "TEMPLATE_GLOBAL_CANDIDATE",
                    "TEMPLATE_GLOBAL_HIGH",
                    "TEMPLATE_PRODUCT_HIGH",
                }
                if not other_reasons:
                    raise ValueError(
                        "Frequency-only template was automatically excluded"
                    )

    if bool(config["product_cap"]["enabled"]):
        maximum = int(config["product_cap"]["maximum_reviews"])
        product_counts = Counter(
            str(parent_by_id[sample_id]["product_id"])
            for sample_id in partition_ids["clean_core"]
        )
        if product_counts and max(product_counts.values()) > maximum:
            raise ValueError("Product cap violated in clean-core")

    privacy = config["privacy"]
    for sample_id in partition_ids["clean_core"]:
        text = curation_by_id[sample_id]["curated_review_text"]
        for key in ("phone_pattern", "email_pattern", "url_pattern"):
            if re.search(privacy[key], text, flags=re.IGNORECASE):
                raise ValueError(
                    f"Unredacted {key} in clean-core: {sample_id}"
                )

    annotation_counts = _validate_annotation(
        release_root,
        curation_by_id,
        config,
    )
    status_counts = Counter(
        row["status"] for row in curation_by_id.values()
    )
    primary_reason_counts = Counter(
        row["primary_reason"] for row in curation_by_id.values()
    )
    exclusion_trigger_counts = Counter(
        row["exclusion_trigger"]
        for row in curation_by_id.values()
        if row["exclusion_trigger"]
    )
    if manifest["counts"]["status"] != dict(sorted(status_counts.items())):
        raise ValueError("Manifest status counts mismatch")
    if manifest["counts"]["primary_reason"] != dict(
        sorted(primary_reason_counts.items())
    ):
        raise ValueError("Manifest primary-reason counts mismatch")
    if manifest["counts"]["exclusion_trigger"] != dict(
        sorted(exclusion_trigger_counts.items())
    ):
        raise ValueError("Manifest exclusion-trigger counts mismatch")
    if manifest["counts"]["parent_records"] != len(parent_rows):
        raise ValueError("Manifest parent count mismatch")
    if manifest["counts"]["annotation"] != annotation_counts:
        raise ValueError("Manifest annotation counts mismatch")
    if manifest["distributions"]["all"] != {
        "rating": _distribution(parent_rows, "rating"),
        "category": _distribution(parent_rows, "category"),
        "transport": _distribution(
            parent_rows,
            "collection_transport",
        ),
    }:
        raise ValueError("Manifest all-record distributions mismatch")
    clean_rows = [
        parent_by_id[sample_id]
        for sample_id in partition_ids["clean_core"]
    ]
    if manifest["distributions"]["clean_core"] != {
        "rating": _distribution(clean_rows, "rating"),
        "category": _distribution(clean_rows, "category"),
        "transport": _distribution(
            clean_rows,
            "collection_transport",
        ),
    }:
        raise ValueError("Manifest clean-core distributions mismatch")

    return {
        "status": "VALID",
        "release_id": manifest["release_id"],
        "parent_records": len(parent_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "transformations": len(transformations),
        "confirmed_duplicate_aliases": len(aliases),
        "near_duplicate_pairs": len(near),
        "annotation": annotation_counts,
        "checksum_entries": len(checksums),
    }


def _rebuild_check(
    release_root: Path,
    parent_root: Path,
    builder_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(
        (release_root / "manifest.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(
        prefix=".curation-rebuild-check-",
        dir=release_root.parent,
    ) as temporary_directory:
        rebuilt = Path(temporary_directory) / release_root.name
        command = [
            sys.executable,
            str(builder_path),
            "--config",
            str(release_root / "provenance" / "cleaning_v2.json"),
            "--parent-release",
            str(parent_root),
            "--output",
            str(rebuilt),
            "--guideline",
            str(
                release_root
                / "provenance"
                / "ABSA_ANNOTATION_GUIDELINE_V2.md"
            ),
            "--built-at",
            str(manifest["built_at"]),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        original = _checksum_map(release_root / "SHA256SUMS.txt")
        reproduced = _checksum_map(rebuilt / "SHA256SUMS.txt")
        if original != reproduced:
            differing = sorted(
                key
                for key in set(original) | set(reproduced)
                if original.get(key) != reproduced.get(key)
            )
            raise ValueError(
                f"Byte reproducibility check failed: {differing}"
            )
    return {
        "status": "BYTE_REPRODUCIBLE",
        "compared_files": len(original),
        "built_at_reused": manifest["built_at"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "release",
        type=Path,
        nargs="?",
        default=Path(
            "data/releases/lazada_vi_absa_curation_v2_1_2_20260725"
        ),
    )
    parser.add_argument(
        "--parent-release",
        type=Path,
        default=Path("data/releases/lazada_vi_reviews_v1_20260725"),
    )
    parser.add_argument(
        "--rebuild-check",
        action="store_true",
    )
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path("scripts/build_clean_release_v2.py"),
    )
    args = parser.parse_args()
    result = validate(args.release, args.parent_release)
    if args.rebuild_check:
        result["reproducibility"] = _rebuild_check(
            args.release.resolve(),
            args.parent_release.resolve(),
            args.builder.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
