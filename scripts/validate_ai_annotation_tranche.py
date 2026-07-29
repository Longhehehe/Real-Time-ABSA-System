"""Independently validate a published 5,000-record ABSA pseudo-label tranche.

The validator deliberately replays the complete chain of custody:

* the frozen prepared-package checksum closure;
* the blind/private/selection joins and the source-release join;
* exclusion of both exact human-reference rows and their reserved leakage
  groups;
* every primary run record and every exact evidence offset;
* the published JSONL, compatibility CSV, decision ledger, review queue,
  manifest, and SHA-256 closure.

Passing this validator does *not* turn AI annotations into human gold.  The
only publishable status in this tranche version is
``AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

from lazada_collector.ai_tranche import count_labels, sha256_file
from lazada_collector.llm_annotation import (
    ASPECT_COLUMNS,
    LLM_ANNOTATION_SCHEMA_VERSION,
    AnnotationValidationError,
    canonical_json,
    sha256_text,
    validate_and_normalize_annotation,
)


TRANCHE_SIZE = 5000
ARTIFACT_STATUS = "AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION"
HUMAN_VERIFICATION_STATUS = "PENDING"
TERMINAL_STATE = "PUBLISHED_PENDING_HUMAN_VERIFICATION"

PSEUDO_RECORD_SCHEMA_VERSION = "absa-ai-pseudo-label-record/1.0.0"
DECISION_LEDGER_SCHEMA_VERSION = "absa-ai-decision-ledger/1.1.0"
HUMAN_QUEUE_SCHEMA_VERSION = "absa-ai-human-review-queue/1.1.0"
FINAL_MANIFEST_SCHEMA_VERSION = "absa-ai-tranche-release-manifest/1.1.0"
FINAL_SUMMARY_SCHEMA_VERSION = "absa-ai-tranche-summary/1.1.0"
RUN_RECORD_SCHEMA_VERSION = "absa-ai-run-record/1.0.0"
RUN_MANIFEST_SCHEMA_VERSION = "absa-ai-run-manifest/1.0.0"
RUN_MANIFEST_ARTIFACT_TYPE = "ABSA_AI_RUN_PROVENANCE_SEAL"
RUN_MANIFEST_STATUS = "SEALED_REPLAY_VALID"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_SUMS_NAME = "RUN_SHA256SUMS.txt"

FINAL_ROOT_ARTIFACTS = {
    "ai_pseudo_labels.jsonl",
    "ai_pseudo_labels_compat.csv",
    "decision_ledger.jsonl",
    "human_review_queue.jsonl",
    "summary.json",
}
RUN_RECORD_KEYS = {
    "schema_version",
    "artifact_type",
    "scope",
    "annotation_id",
    "selection_rank",
    "review_text_sha256",
    "annotation",
    "normalization_repairs",
    "generation",
}
GENERATION_KEYS = {
    "backend",
    "batch_id",
    "valid_attempt",
    "model",
    "prompt_version",
    "system_prompt_sha256",
    "human_example_ids",
    "reasoning_effort",
    "enable_thinking",
    "codex_schema_sha256",
}
RUN_SUMMARY_KEYS = {
    "artifact_type",
    "scope",
    "status",
    "started_at",
    "completed_at",
    "selected_records",
    "valid_records",
    "missing_annotation_ids",
    "provider",
    "prompt",
    "provider_calls_this_invocation",
    "provider_usage_this_invocation",
    "label_summary",
}
RUN_PROVIDER_KEYS = {
    "backend",
    "endpoint",
    "model",
    "api_key_env",
    "batch_size",
    "workers",
    "max_retries",
    "max_tokens",
    "timeout_seconds",
    "reasoning_effort",
    "enable_thinking",
    "codex_schema",
}
SOURCE_METADATA_KEYS = (
    "source_release_id",
    "parent_canonical_row",
    "curation_status",
    "category",
    "rating",
    "collection_transport",
    "product_id",
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"\A[A-Za-z]:[\\/]")
_WINDOWS_USER_PATH_RE = re.compile(
    r"(?i)(?:\A|\s)[A-Za-z]:[\\/]Users[\\/]"
)
_MARKETING_RE = re.compile(
    r"(?iu)(?:"
    r"(?:mô\s+tả|thông\s*tin|tên)\s+sản\s*phẩm|"
    r"hướng\s*dẫn\s+sử\s*dụng|công\s*dụng\s*:|chất\s*liệu\s*:|"
    r"cam\s*kết\s+(?:chất\s*lượng|chính\s*hãng)|"
    r"(?:inbox|liên\s*hệ)\s+(?:shop|để)|"
    r"giá\s*(?:sỉ|buôn)|khuyến\s*mãi\s*(?:sốc|đặc\s*biệt)"
    r")"
)
CSV_FIELDS = (
    "sample_id",
    "annotation_id",
    "selection_rank",
    "reviewContent",
    "annotation_status",
    *ASPECT_COLUMNS,
    "artifact_status",
    "human_verification_status",
    "review_text_sha256",
)


class TrancheValidationError(ValueError):
    """Raised when a tranche violates a frozen publication invariant."""


def _expected_selection_decision(
    manifest: Mapping[str, Any],
    *,
    selected_rows: Iterable[Mapping[str, Any]] = (),
) -> str:
    selection = manifest.get("selection")
    tranche = (
        selection.get("tranche")
        if isinstance(selection, Mapping)
        else None
    )
    if isinstance(tranche, str):
        normalized = tranche.upper().replace("-", "_")
        suffix = normalized.removeprefix("TRANCHE_")
        if (
            normalized.startswith("TRANCHE_")
            and len(suffix) == 4
            and suffix.isdigit()
        ):
            return f"SELECT_{normalized}"
    # Backward-compatible validation for frozen test/legacy fixtures that
    # predate the explicit tranche-name field. The decision must still be
    # unique and conform to the same closed form.
    decisions = {
        row.get("decision")
        for row in selected_rows
        if isinstance(row.get("decision"), str)
    }
    if len(decisions) == 1:
        decision = next(iter(decisions))
        if re.fullmatch(r"SELECT_TRANCHE_\d{4}", decision):
            return decision
    raise TrancheValidationError("Prepared tranche name is invalid")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise TrancheValidationError(
            f"{context} keys mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrancheValidationError(f"{context} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TrancheValidationError(f"{context} must be a lowercase SHA-256")
    return value


def _require_int(
    value: Any,
    *,
    context: str,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise TrancheValidationError(
            f"{context} must be an integer >= {minimum}"
        )
    return value


def _require_portable_manifest(value: Any, *, context: str = "manifest") -> None:
    """Reject host-absolute paths anywhere in a published manifest."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_portable_manifest(
                child,
                context=f"{context}.{key}",
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _require_portable_manifest(
                child,
                context=f"{context}[{index}]",
            )
        return
    if not isinstance(value, str):
        return
    if (
        _WINDOWS_ABSOLUTE_RE.match(value)
        or _WINDOWS_USER_PATH_RE.search(value)
        or value.startswith(("/", "\\"))
    ):
        raise TrancheValidationError(
            f"Published manifest contains a host-absolute path at {context}"
        )


def _duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise TrancheValidationError(f"Duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejector,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TrancheValidationError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrancheValidationError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise TrancheValidationError(
                        f"Blank JSONL line at {path}:{line_number}"
                    )
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_duplicate_rejector,
                    )
                except json.JSONDecodeError as exc:
                    raise TrancheValidationError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise TrancheValidationError(
                        f"Expected object at {path}:{line_number}"
                    )
                rows.append(value)
    except OSError as exc:
        raise TrancheValidationError(f"Cannot read {path}: {exc}") from exc
    return rows


def _safe_artifact_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TrancheValidationError("Artifact path must be a non-empty string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise TrancheValidationError(f"Unsafe artifact path: {relative!r}")
    candidate = root.joinpath(*pure.parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise TrancheValidationError(
            f"Artifact escapes package root: {relative!r}"
        ) from exc
    return candidate


def _read_sums(path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise TrancheValidationError(
                f"Malformed checksum line at {path}:{line_number}"
            )
        checksum, relative = parts
        if (
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise TrancheValidationError(
                f"Malformed SHA-256 at {path}:{line_number}"
            )
        if relative in sums:
            raise TrancheValidationError(f"Duplicate checksum path: {relative}")
        sums[relative] = checksum
    return sums


def _require_unique(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    *,
    context: str,
) -> dict[Any, Mapping[str, Any]]:
    output: dict[Any, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            raise TrancheValidationError(f"{context} has empty {field}")
        if value in output:
            raise TrancheValidationError(
                f"Duplicate {field} in {context}: {value!r}"
            )
        output[value] = row
    return output


def _resolve_external(raw_path: Any, *, package: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise TrancheValidationError("Manifest external path is missing")
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (package / path).resolve()


def _verify_artifact_inventory(
    *,
    root: Path,
    artifacts: Any,
    sums: Mapping[str, str],
    manifest_name: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, list):
        raise TrancheValidationError("Manifest artifacts must be a list")
    inventory: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise TrancheValidationError("Malformed manifest artifact")
        relative = item["path"]
        if relative in inventory:
            raise TrancheValidationError(
                f"Duplicate manifest artifact: {relative}"
            )
        path = _safe_artifact_path(root, relative)
        if not path.is_file():
            raise TrancheValidationError(f"Missing artifact: {relative}")
        actual = sha256_file(path)
        if item.get("sha256") != actual or sums.get(relative) != actual:
            raise TrancheValidationError(
                f"Artifact checksum mismatch: {relative}"
            )
        if item.get("bytes") != path.stat().st_size:
            raise TrancheValidationError(
                f"Artifact byte-count mismatch: {relative}"
            )
        inventory[relative] = item
    manifest_path = root / manifest_name
    if sums.get(manifest_name) != sha256_file(manifest_path):
        raise TrancheValidationError(
            f"Manifest checksum mismatch: {manifest_name}"
        )
    expected_sums = set(inventory) | {manifest_name}
    if set(sums) != expected_sums:
        raise TrancheValidationError(
            "Checksum closure mismatch; "
            f"missing={sorted(expected_sums - set(sums))}, "
            f"extra={sorted(set(sums) - expected_sums)}"
        )
    return inventory


def _verify_human_reference_ledgers(
    package: Path,
    manifest: Mapping[str, Any],
) -> tuple[set[str], set[str], set[str], set[str]]:
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise TrancheValidationError("Prepared manifest has no selection block")
    human_manifest_path = _resolve_external(
        selection.get("human_reference_manifest_path"),
        package=package,
    )
    expected_human_manifest_sha = selection.get(
        "human_reference_manifest_sha256"
    )
    if sha256_file(human_manifest_path) != expected_human_manifest_sha:
        raise TrancheValidationError("Human-reference manifest hash mismatch")
    human_root = human_manifest_path.parent
    human_manifest = _read_json(human_manifest_path)
    human_inventory = {
        item["path"]: item
        for item in human_manifest.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    human_sums = _read_sums(human_root / "SHA256SUMS.txt")

    crosswalk_path = human_root / "private" / "crosswalk.jsonl"
    reservations_path = _resolve_external(
        selection.get("group_reservations_path"),
        package=package,
    )
    for path in (crosswalk_path, reservations_path):
        try:
            relative = path.relative_to(human_root).as_posix()
        except ValueError as exc:
            raise TrancheValidationError(
                "Human-reference ledgers are from different roots"
            ) from exc
        actual = sha256_file(path)
        item = human_inventory.get(relative)
        if (
            not isinstance(item, Mapping)
            or item.get("sha256") != actual
            or item.get("bytes") != path.stat().st_size
            or human_sums.get(relative) != actual
        ):
            raise TrancheValidationError(
                f"Human-reference ledger closure mismatch: {relative}"
            )
    if (
        sha256_file(reservations_path)
        != selection.get("group_reservations_sha256")
    ):
        raise TrancheValidationError("Group-reservation hash mismatch")

    reference_rows = _read_jsonl(crosswalk_path)
    reservation_rows = _read_jsonl(reservations_path)
    reference_ids = set(
        _require_unique(
            reference_rows,
            "sample_id",
            context="human-reference crosswalk",
        )
    )
    reserved_ids = set(
        _require_unique(
            reservation_rows,
            "sample_id",
            context="group-reservation ledger",
        )
    )
    if not reference_ids.issubset(reserved_ids):
        raise TrancheValidationError(
            "Group reservations omit an exact reference record"
        )
    reference_hashes = set(
        _require_unique(
            reference_rows,
            "review_text_sha256",
            context="human-reference crosswalk text hashes",
        )
    )
    reserved_hashes = set(
        _require_unique(
            reservation_rows,
            "review_text_sha256",
            context="group-reservation text hashes",
        )
    )
    if not reference_hashes.issubset(reserved_hashes):
        raise TrancheValidationError(
            "Group reservations omit an exact reference text hash"
        )
    expected_reference = selection.get("reserved_human_reference_records")
    expected_reserved = selection.get("reserved_reference_group_records")
    if expected_reference is not None and len(reference_ids) != expected_reference:
        raise TrancheValidationError("Reference-record count mismatch")
    if expected_reserved is not None and len(reserved_ids) != expected_reserved:
        raise TrancheValidationError("Reserved-group count mismatch")
    return reference_ids, reserved_ids, reference_hashes, reserved_hashes


def _validate_calibration_package(
    package: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    required_paths = {
        "calibration/human_confirmed.jsonl",
        "calibration/diagnostic_split.json",
    }
    declared = {
        item.get("path")
        for item in manifest.get("artifacts", [])
        if isinstance(item, Mapping)
    }
    if not required_paths.issubset(declared):
        raise TrancheValidationError(
            "Prepared package omits frozen calibration artifacts"
        )
    calibration_rows = _read_jsonl(
        package / "calibration" / "human_confirmed.jsonl"
    )
    calibration_by_id: dict[str, dict[str, Any]] = {}
    accepted_by_id: dict[str, dict[str, Any]] = {}
    excluded_by_id: dict[str, dict[str, Any]] = {}
    for row in calibration_rows:
        _require_exact_keys(
            row,
            {
                "schema_version",
                "calibration_id",
                "reviewContent",
                "review_text_sha256",
                "annotation",
                "human_provenance",
                "semantic_audit",
            },
            context="human calibration row",
        )
        calibration_id = _require_nonempty_string(
            row.get("calibration_id"),
            context="calibration_id",
        )
        if calibration_id in calibration_by_id:
            raise TrancheValidationError(
                f"Duplicate calibration ID: {calibration_id}"
            )
        review_text = row.get("reviewContent")
        if (
            row.get("schema_version") != "absa-human-calibration/1.0.0"
            or not isinstance(review_text, str)
            or not review_text.strip()
            or row.get("review_text_sha256") != sha256_text(review_text)
        ):
            raise TrancheValidationError(
                f"Calibration source mismatch: {calibration_id}"
            )
        validate_canonical_annotation(
            review_text,
            row.get("annotation"),
        )
        provenance = row.get("human_provenance")
        audit = row.get("semantic_audit")
        if not isinstance(provenance, Mapping) or not isinstance(
            audit,
            Mapping,
        ):
            raise TrancheValidationError(
                f"Calibration provenance is malformed: {calibration_id}"
            )
        _require_exact_keys(
            provenance,
            {
                "annotator_id",
                "assignment_id",
                "completed_at",
                "confirmation_mode",
                "export_payload_sha256",
            },
            context=f"human provenance {calibration_id}",
        )
        _require_exact_keys(
            audit,
            {
                "audit_actor",
                "decision",
                "human_label_mutated",
                "reason",
            },
            context=f"semantic audit {calibration_id}",
        )
        _require_sha256(
            provenance.get("export_payload_sha256"),
            context=f"calibration export payload {calibration_id}",
        )
        if not isinstance(audit.get("human_label_mutated"), bool):
            raise TrancheValidationError(
                f"Calibration mutation flag is invalid: {calibration_id}"
            )
        decision = audit.get("decision")
        if decision == "CALIBRATION_ACCEPT":
            accepted_by_id[calibration_id] = row
        elif decision == "EXCLUDE_PENDING_EXPERT_ADJUDICATION":
            excluded_by_id[calibration_id] = row
        else:
            raise TrancheValidationError(
                f"Calibration semantic audit is open: {calibration_id}"
            )
        calibration_by_id[calibration_id] = row

    split = _read_json(
        package / "calibration" / "diagnostic_split.json"
    )
    _require_exact_keys(
        split,
        {
            "schema_version",
            "records_total",
            "diagnostic_holdout_records",
            "prompt_calibration_records",
            "diagnostic_holdout_ids",
            "prompt_calibration_ids",
            "method",
        },
        context="diagnostic split",
    )
    if split.get("schema_version") != "absa-calibration-split/1.0.0":
        raise TrancheValidationError("Diagnostic split schema mismatch")
    holdout_ids = split.get("diagnostic_holdout_ids")
    prompt_ids = split.get("prompt_calibration_ids")
    if not isinstance(holdout_ids, list) or not isinstance(prompt_ids, list):
        raise TrancheValidationError("Diagnostic split IDs must be lists")
    if (
        any(not isinstance(item, str) or not item for item in holdout_ids)
        or any(not isinstance(item, str) or not item for item in prompt_ids)
        or len(set(holdout_ids)) != len(holdout_ids)
        or len(set(prompt_ids)) != len(prompt_ids)
    ):
        raise TrancheValidationError("Diagnostic split IDs are malformed")
    holdout_set = set(holdout_ids)
    prompt_set = set(prompt_ids)
    if (
        holdout_set.intersection(prompt_set)
        or holdout_set | prompt_set != set(accepted_by_id)
        or split.get("records_total") != len(calibration_rows)
        or split.get("diagnostic_holdout_records") != len(holdout_set)
        or split.get("prompt_calibration_records") != len(prompt_set)
    ):
        raise TrancheValidationError(
            "Diagnostic split is not a closed partition of accepted calibration"
        )
    human_block = manifest.get("human_calibration")
    if not isinstance(human_block, Mapping):
        raise TrancheValidationError(
            "Prepared human-calibration provenance is missing"
        )
    if (
        human_block.get("completed_valid") != len(calibration_rows)
        or human_block.get("semantic_audit_accepted")
        != len(accepted_by_id)
        or human_block.get("semantic_audit_excluded_pending_adjudication")
        != len(excluded_by_id)
    ):
        raise TrancheValidationError(
            "Prepared calibration counts do not replay"
        )
    return {
        "calibration_rows": calibration_rows,
        "accepted_calibration_by_id": accepted_by_id,
        "excluded_calibration_by_id": excluded_by_id,
        "diagnostic_split": split,
        "diagnostic_holdout_ids": tuple(sorted(holdout_set)),
        "prompt_calibration_ids": tuple(sorted(prompt_set)),
    }


def _validate_prepared_package(
    package: Path,
    *,
    expected_records: int,
) -> dict[str, Any]:
    package = package.resolve()
    manifest_path = package / "prepare_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "PREPARED_NOT_LABELED":
        raise TrancheValidationError("Prepared package status is invalid")
    if manifest.get("target_records") != expected_records:
        raise TrancheValidationError(
            f"Prepared package must target exactly {expected_records} records"
        )
    sums = _read_sums(package / "INPUT_SHA256SUMS.txt")
    inventory = _verify_artifact_inventory(
        root=package,
        artifacts=manifest.get("artifacts"),
        sums=sums,
        manifest_name="prepare_manifest.json",
    )
    for relative in (
        "input/blind_reviews.jsonl",
        "input/private_index.jsonl",
        "input/selection_ledger.jsonl",
    ):
        if relative not in inventory:
            raise TrancheValidationError(
                f"Prepared package omits required artifact: {relative}"
            )

    blind_rows = _read_jsonl(package / "input" / "blind_reviews.jsonl")
    private_rows = _read_jsonl(package / "input" / "private_index.jsonl")
    ledger_rows = _read_jsonl(package / "input" / "selection_ledger.jsonl")
    if len(blind_rows) != expected_records or len(private_rows) != expected_records:
        raise TrancheValidationError("Prepared input row count is not exact")
    blind_by_id = _require_unique(
        blind_rows,
        "annotation_id",
        context="blind input",
    )
    private_by_id = _require_unique(
        private_rows,
        "annotation_id",
        context="private input",
    )
    if set(blind_by_id) != set(private_by_id):
        raise TrancheValidationError("Blind/private annotation ID join mismatch")

    ranks: set[int] = set()
    sample_ids: set[str] = set()
    text_hashes: set[str] = set()
    for annotation_id, blind in blind_by_id.items():
        private = private_by_id[annotation_id]
        _require_exact_keys(
            blind,
            {
                "schema_version",
                "selection_rank",
                "annotation_id",
                "reviewContent",
                "review_text_sha256",
            },
            context=f"blind input {annotation_id}",
        )
        _require_exact_keys(
            private,
            {
                "schema_version",
                "selection_rank",
                "annotation_id",
                "sample_id",
                "review_text_sha256",
                "parent_canonical_row",
                "curation_status",
                "category",
                "rating",
                "collection_transport",
                "product_id",
                "source_release_id",
            },
            context=f"private input {annotation_id}",
        )
        if (
            blind.get("schema_version") != "absa-ai-tranche-input/1.0.0"
            or private.get("schema_version")
            != "absa-ai-tranche-private-index/1.0.0"
        ):
            raise TrancheValidationError(
                f"Prepared input schema mismatch: {annotation_id}"
            )
        rank = blind.get("selection_rank")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
            or rank > expected_records
            or rank in ranks
            or private.get("selection_rank") != rank
        ):
            raise TrancheValidationError(
                f"Invalid selection rank for {annotation_id}"
            )
        ranks.add(rank)
        review_text = blind.get("reviewContent")
        text_hash = blind.get("review_text_sha256")
        if (
            not isinstance(review_text, str)
            or not review_text.strip()
            or text_hash != sha256_text(review_text)
            or private.get("review_text_sha256") != text_hash
            or text_hash in text_hashes
        ):
            raise TrancheValidationError(
                f"Review text/hash invariant failed for {annotation_id}"
            )
        text_hashes.add(text_hash)
        sample_id = private.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise TrancheValidationError(
                f"Invalid/duplicate sample_id for {annotation_id}"
            )
        sample_ids.add(sample_id)
    if ranks != set(range(1, expected_records + 1)):
        raise TrancheValidationError("Selection ranks are not contiguous")

    selected_ledger = [row for row in ledger_rows if row.get("selected") is True]
    selected_by_sample = _require_unique(
        selected_ledger,
        "sample_id",
        context="selected membership ledger",
    )
    if set(selected_by_sample) != sample_ids:
        raise TrancheValidationError(
            "Selection ledger/private-index membership mismatch"
        )
    expected_selection_decision = _expected_selection_decision(
        manifest,
        selected_rows=selected_ledger,
    )
    for private in private_rows:
        selected = selected_by_sample[private["sample_id"]]
        if (
            selected.get("selection_rank") != private["selection_rank"]
            or selected.get("review_text_sha256")
            != private["review_text_sha256"]
            or selected.get("decision") != expected_selection_decision
        ):
            raise TrancheValidationError(
                f"Selection ledger join mismatch: {private['sample_id']}"
            )

    ordered_private = sorted(private_rows, key=lambda row: row["selection_rank"])
    membership_sha = hashlib.sha256(
        "".join(
            f"{row['sample_id']}\t{row['review_text_sha256']}\n"
            for row in ordered_private
        ).encode("utf-8")
    ).hexdigest()
    if (
        manifest.get("selection", {}).get("ordered_membership_sha256")
        != membership_sha
    ):
        raise TrancheValidationError("Ordered membership hash mismatch")

    (
        reference_ids,
        reserved_ids,
        reference_hashes,
        reserved_hashes,
    ) = _verify_human_reference_ledgers(package, manifest)
    exact_overlap = sample_ids.intersection(reference_ids)
    group_overlap = sample_ids.intersection(reserved_ids)
    selected_hashes = {
        str(row["review_text_sha256"]) for row in private_rows
    }
    exact_hash_overlap = selected_hashes.intersection(reference_hashes)
    group_hash_overlap = selected_hashes.intersection(reserved_hashes)
    if (
        exact_overlap
        or group_overlap
        or exact_hash_overlap
        or group_hash_overlap
    ):
        raise TrancheValidationError(
            "Human-reference leakage detected; "
            f"exact_id={len(exact_overlap)}, "
            f"reserved_group_id={len(group_overlap)}, "
            f"exact_hash={len(exact_hash_overlap)}, "
            f"reserved_group_hash={len(group_hash_overlap)}"
        )

    source = manifest.get("source_release")
    if not isinstance(source, Mapping):
        raise TrancheValidationError("Prepared manifest has no source release")
    source_root = _resolve_external(source.get("path"), package=package)
    source_manifest_path = source_root / "manifest.json"
    records_relative = source.get("records_path", "clean_core.jsonl")
    if records_relative not in {"clean_core.jsonl", "quarantine.jsonl"}:
        raise TrancheValidationError(
            "Prepared source record partition is unsupported"
        )
    source_records_path = source_root / records_relative
    if sha256_file(source_manifest_path) != source.get("manifest_sha256"):
        raise TrancheValidationError("Source-release manifest hash mismatch")
    expected_records_sha = source.get(
        "records_sha256",
        source.get("clean_core_sha256"),
    )
    if sha256_file(source_records_path) != expected_records_sha:
        raise TrancheValidationError("Source record-partition hash mismatch")
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("release_id") != source.get("release_id"):
        raise TrancheValidationError("Source release ID mismatch")
    source_rows = _read_jsonl(source_records_path)
    source_by_sample = _require_unique(
        source_rows,
        "sample_id",
        context="source record partition",
    )
    expected_records = source.get(
        "records_count",
        source.get("clean_core_records"),
    )
    if expected_records != len(source_rows):
        raise TrancheValidationError(
            "Source record-partition count mismatch"
        )
    if not sample_ids.issubset(source_by_sample):
        raise TrancheValidationError(
            "Private samples are outside source record partition"
        )
    source_partition = manifest.get("selection", {}).get(
        "source_partition"
    )
    expected_partition_path = {
        "clean_core": "clean_core.jsonl",
        "quarantine": "quarantine.jsonl",
    }.get(source_partition)
    if (
        source_partition is not None
        and expected_partition_path != records_relative
    ):
        raise TrancheValidationError(
            "Selection/source-partition binding mismatch"
        )
    allowed_curation_statuses = {
        "clean_core": {"KEEP", "KEEP_CLEANED"},
        "quarantine": {"QUARANTINE"},
    }.get(source_partition)
    for private in private_rows:
        source_row = source_by_sample[private["sample_id"]]
        source_text = source_row.get("curated_review_text")
        curation = source_row.get("curation")
        if not isinstance(curation, Mapping):
            raise TrancheValidationError(
                f"Source curation metadata missing: {private['sample_id']}"
            )
        source_hash = curation.get("curated_text_sha256")
        blind = blind_by_id[private["annotation_id"]]
        expected_source_metadata = {
            "source_release_id": source.get("release_id"),
            "parent_canonical_row": curation.get("parent_canonical_row"),
            "curation_status": curation.get("status"),
            "category": source_row.get("category"),
            "rating": source_row.get("rating"),
            "collection_transport": source_row.get(
                "collection_transport"
            ),
            "product_id": source_row.get("product_id"),
        }
        private_source_metadata = {
            key: private.get(key) for key in SOURCE_METADATA_KEYS
        }
        if (
            source_text != blind["reviewContent"]
            or source_hash != private["review_text_sha256"]
            or sha256_text(str(source_text)) != source_hash
            or (
                allowed_curation_statuses is not None
                and curation.get("status") not in allowed_curation_statuses
            )
            or canonical_json(private_source_metadata)
            != canonical_json(expected_source_metadata)
        ):
            raise TrancheValidationError(
                f"Source join mismatch: {private['sample_id']}"
            )

    calibration_context = _validate_calibration_package(
        package,
        manifest,
    )
    return {
        "manifest": manifest,
        "inventory": inventory,
        "blind_rows": sorted(blind_rows, key=lambda row: row["selection_rank"]),
        "private_rows": sorted(private_rows, key=lambda row: row["selection_rank"]),
        "blind_by_id": blind_by_id,
        "private_by_id": private_by_id,
        "reference_ids": reference_ids,
        "reserved_ids": reserved_ids,
        "source_by_sample": source_by_sample,
        **calibration_context,
    }


def validate_canonical_annotation(
    review_text: str,
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay a normalized annotation and return its canonical equivalent."""

    if not isinstance(annotation, Mapping):
        raise TrancheValidationError("Annotation must be an object")
    expected_root = {
        "schema_version",
        "annotation_status",
        "aspects",
        "review_uncertainty_codes",
        "notes",
    }
    if set(annotation) != expected_root:
        raise TrancheValidationError("Canonical annotation root keys mismatch")
    if annotation.get("schema_version") != LLM_ANNOTATION_SCHEMA_VERSION:
        raise TrancheValidationError("Canonical annotation schema mismatch")
    raw_aspects: list[dict[str, Any]] = []
    aspects = annotation.get("aspects")
    if not isinstance(aspects, list) or len(aspects) != len(ASPECT_COLUMNS):
        raise TrancheValidationError("Canonical annotation aspect count mismatch")
    for expected_aspect, aspect in zip(ASPECT_COLUMNS, aspects, strict=True):
        if not isinstance(aspect, Mapping) or set(aspect) != {
            "aspect",
            "label",
            "evidence",
            "uncertainty_codes",
        }:
            raise TrancheValidationError("Canonical aspect keys mismatch")
        if aspect.get("aspect") != expected_aspect:
            raise TrancheValidationError("Canonical aspect order mismatch")
        raw_evidence: list[dict[str, Any]] = []
        evidence_rows = aspect.get("evidence")
        if not isinstance(evidence_rows, list):
            raise TrancheValidationError("Canonical evidence must be a list")
        for evidence in evidence_rows:
            if not isinstance(evidence, Mapping) or set(evidence) != {
                "text",
                "start",
                "end",
                "polarity",
            }:
                raise TrancheValidationError("Canonical evidence keys mismatch")
            text = evidence.get("text")
            start = evidence.get("start")
            end = evidence.get("end")
            if (
                not isinstance(text, str)
                or not text
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end != start + len(text)
                or review_text[start:end] != text
            ):
                raise TrancheValidationError(
                    f"Evidence offsets do not replay for {expected_aspect}"
                )
            positions: list[int] = []
            cursor = 0
            while True:
                found = review_text.find(text, cursor)
                if found < 0:
                    break
                positions.append(found)
                cursor = found + 1
            if start not in positions:
                raise TrancheValidationError(
                    f"Evidence occurrence does not replay for {expected_aspect}"
                )
            raw_evidence.append(
                {
                    "quote": text,
                    "occurrence": positions.index(start) + 1,
                    "polarity": evidence.get("polarity"),
                }
            )
        raw_aspects.append(
            {
                "aspect": expected_aspect,
                "label": aspect.get("label"),
                "evidence": raw_evidence,
                "uncertainty_codes": aspect.get("uncertainty_codes"),
            }
        )
    raw = {
        "annotation_status": annotation.get("annotation_status"),
        "aspects": raw_aspects,
        "review_uncertainty_codes": annotation.get(
            "review_uncertainty_codes"
        ),
        "notes": annotation.get("notes"),
    }
    try:
        normalized = validate_and_normalize_annotation(review_text, raw)
    except AnnotationValidationError as exc:
        raise TrancheValidationError(
            f"Canonical annotation is invalid: {exc}"
        ) from exc
    if canonical_json(normalized) != canonical_json(dict(annotation)):
        raise TrancheValidationError(
            "Canonical annotation changes after exact replay"
        )
    return normalized


def _validate_normalization_repairs(
    *,
    review_text: str,
    annotation: Mapping[str, Any],
    repairs: Any,
) -> None:
    if not isinstance(repairs, list):
        raise TrancheValidationError("normalization_repairs must be a list")
    seen_locations: set[tuple[int, int]] = set()
    aspects = annotation["aspects"]
    for repair in repairs:
        if not isinstance(repair, Mapping) or set(repair) != {
            "aspect_index",
            "evidence_index",
            "original_quote",
            "repaired_quote",
            "method",
        }:
            raise TrancheValidationError(
                "Normalization-repair schema mismatch"
            )
        aspect_index = repair.get("aspect_index")
        evidence_index = repair.get("evidence_index")
        if (
            not isinstance(aspect_index, int)
            or isinstance(aspect_index, bool)
            or not isinstance(evidence_index, int)
            or isinstance(evidence_index, bool)
            or not 1 <= aspect_index <= len(aspects)
        ):
            raise TrancheValidationError(
                "Normalization-repair index is invalid"
            )
        evidence_rows = aspects[aspect_index - 1]["evidence"]
        if not 1 <= evidence_index <= len(evidence_rows):
            raise TrancheValidationError(
                "Normalization repair does not join to canonical evidence"
            )
        location = (aspect_index, evidence_index)
        if location in seen_locations:
            raise TrancheValidationError(
                "Duplicate normalization repair for one evidence item"
            )
        seen_locations.add(location)
        original = repair.get("original_quote")
        repaired = repair.get("repaired_quote")
        method = repair.get("method")
        if (
            not isinstance(original, str)
            or not original
            or not isinstance(repaired, str)
            or not repaired
            or original == repaired
            or repaired not in review_text
            or evidence_rows[evidence_index - 1]["text"] != repaired
        ):
            raise TrancheValidationError(
                "Normalization repair does not replay"
            )
        if method == "CASE_ONLY":
            if original.casefold() != repaired.casefold():
                raise TrancheValidationError("Invalid CASE_ONLY repair")
        else:
            raise TrancheValidationError(
                f"Unsupported normalization-repair method: {method!r}"
            )


def _target_membership_sha(
    targets: Iterable[Mapping[str, Any]],
) -> str:
    ordered = sorted(targets, key=lambda row: row["selection_rank"])
    return sha256_text(
        "".join(
            f"{row['selection_rank']}\t{row['annotation_id']}\t"
            f"{row['review_text_sha256']}\n"
            for row in ordered
        )
    )


def _validate_run_summary(
    summary: Mapping[str, Any],
    *,
    scope: str,
    target_count: int,
) -> dict[str, Any]:
    expected_keys = set(RUN_SUMMARY_KEYS)
    if scope == "diagnostic":
        expected_keys.add("diagnostic_gate")
    _require_exact_keys(
        summary,
        expected_keys,
        context=f"{scope} run summary",
    )
    if (
        summary.get("artifact_type")
        != "ABSA_AI_PREANNOTATION_RUN_SUMMARY"
        or summary.get("scope") != scope
        or summary.get("status") != "COMPLETED"
        or summary.get("selected_records") != target_count
        or summary.get("valid_records") != target_count
        or summary.get("missing_annotation_ids") != []
    ):
        raise TrancheValidationError(
            f"{scope.capitalize()} run summary is not complete"
        )
    _require_nonempty_string(
        summary.get("started_at"),
        context=f"{scope} run started_at",
    )
    _require_nonempty_string(
        summary.get("completed_at"),
        context=f"{scope} run completed_at",
    )
    _require_int(
        summary.get("provider_calls_this_invocation"),
        context=f"{scope} provider call count",
    )
    usage = summary.get("provider_usage_this_invocation")
    if not isinstance(usage, Mapping):
        raise TrancheValidationError(f"{scope} provider usage is malformed")
    _require_exact_keys(
        usage,
        {"prompt_tokens", "completion_tokens", "total_tokens"},
        context=f"{scope} provider usage",
    )
    for key, value in usage.items():
        _require_int(value, context=f"{scope} usage {key}")

    prompt = summary.get("prompt")
    provider = summary.get("provider")
    if not isinstance(prompt, Mapping) or not isinstance(provider, Mapping):
        raise TrancheValidationError(
            f"{scope} execution configuration is malformed"
        )
    _require_exact_keys(
        prompt,
        {"version", "system_prompt_sha256"},
        context=f"{scope} prompt configuration",
    )
    _require_nonempty_string(
        prompt.get("version"),
        context=f"{scope} prompt version",
    )
    _require_sha256(
        prompt.get("system_prompt_sha256"),
        context=f"{scope} system prompt",
    )
    _require_exact_keys(
        provider,
        RUN_PROVIDER_KEYS,
        context=f"{scope} provider configuration",
    )
    backend = provider.get("backend")
    if backend not in {"nvidia", "codex"}:
        raise TrancheValidationError(
            f"{scope} run backend is unsupported: {backend!r}"
        )
    for field in ("endpoint", "model", "api_key_env"):
        _require_nonempty_string(
            provider.get(field),
            context=f"{scope} provider {field}",
        )
    for field in ("batch_size", "workers", "max_tokens"):
        _require_int(
            provider.get(field),
            context=f"{scope} provider {field}",
            minimum=1,
        )
    _require_int(
        provider.get("max_retries"),
        context=f"{scope} provider max_retries",
    )
    timeout = provider.get("timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
        or not math.isfinite(float(timeout))
    ):
        raise TrancheValidationError(
            f"{scope} provider timeout_seconds must be positive and finite"
        )
    reasoning = provider.get("reasoning_effort")
    thinking = provider.get("enable_thinking")
    if reasoning is not None and (
        not isinstance(reasoning, str) or not reasoning
    ):
        raise TrancheValidationError(
            f"{scope} reasoning_effort is malformed"
        )
    if thinking is not None and not isinstance(thinking, bool):
        raise TrancheValidationError(
            f"{scope} enable_thinking is malformed"
        )
    schema = provider.get("codex_schema")
    if backend == "codex":
        if not isinstance(schema, Mapping):
            raise TrancheValidationError(
                f"{scope} Codex schema provenance is missing"
            )
        _require_exact_keys(
            schema,
            {"path", "sha256"},
            context=f"{scope} Codex schema",
        )
        _require_nonempty_string(
            schema.get("path"),
            context=f"{scope} Codex schema path",
        )
        _require_sha256(
            schema.get("sha256"),
            context=f"{scope} Codex schema",
        )
    elif schema is not None:
        raise TrancheValidationError(
            f"{scope} non-Codex run declares a Codex schema"
        )
    return dict(provider)


def run_execution_config(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable cross-scope generation configuration."""

    provider = summary["provider"]
    prompt = summary["prompt"]
    schema = provider.get("codex_schema")
    return {
        "backend": provider.get("backend"),
        "model": provider.get("model"),
        "prompt_version": prompt.get("version"),
        "system_prompt_sha256": prompt.get("system_prompt_sha256"),
        "reasoning_effort": provider.get("reasoning_effort"),
        "enable_thinking": provider.get("enable_thinking"),
        "codex_schema_sha256": (
            schema.get("sha256") if isinstance(schema, Mapping) else None
        ),
    }


def _validate_generation(
    generation: Any,
    *,
    summary: Mapping[str, Any],
    annotation_id: str,
    allowed_example_ids: set[str],
) -> None:
    if not isinstance(generation, Mapping):
        raise TrancheValidationError(
            f"Generation metadata is not an object: {annotation_id}"
        )
    _require_exact_keys(
        generation,
        GENERATION_KEYS,
        context=f"generation metadata {annotation_id}",
    )
    config = run_execution_config(summary)
    for key in (
        "backend",
        "model",
        "prompt_version",
        "system_prompt_sha256",
        "reasoning_effort",
        "enable_thinking",
        "codex_schema_sha256",
    ):
        if generation.get(key) != config[key]:
            raise TrancheValidationError(
                f"Run-summary/generation {key} mismatch: {annotation_id}"
            )
    _require_nonempty_string(
        generation.get("batch_id"),
        context=f"generation batch_id {annotation_id}",
    )
    _require_int(
        generation.get("valid_attempt"),
        context=f"generation valid_attempt {annotation_id}",
        minimum=1,
    )
    example_ids = generation.get("human_example_ids")
    if (
        not isinstance(example_ids, list)
        or any(not isinstance(item, str) or not item for item in example_ids)
        or len(example_ids) != len(set(example_ids))
        or not set(example_ids).issubset(allowed_example_ids)
    ):
        raise TrancheValidationError(
            f"Generation human-example binding mismatch: {annotation_id}"
        )


def _verify_run_seal(
    package: Path,
    *,
    scope: str,
    targets: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = package / "runs" / scope
    manifest_path = run_root / RUN_MANIFEST_NAME
    sums_path = run_root / RUN_SUMS_NAME
    if not manifest_path.is_file() or not sums_path.is_file():
        raise TrancheValidationError(
            f"{scope} run is not sealed; run "
            "scripts/seal_ai_annotation_runs.py after completion"
        )
    manifest = _read_json(manifest_path)
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "status",
            "scope",
            "sealed_at",
            "path_basis",
            "tranche_id",
            "input_bindings",
            "target_source",
            "target_records",
            "target_membership_sha256",
            "provider",
            "prompt",
            "sealing_implementation",
            "replay",
            "inventory",
            "diagnostic",
            "artifacts",
            "limitations",
        },
        context=f"{scope} run seal",
    )
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_type") != RUN_MANIFEST_ARTIFACT_TYPE
        or manifest.get("status") != RUN_MANIFEST_STATUS
        or manifest.get("scope") != scope
        or manifest.get("path_basis") != "package_root"
        or manifest.get("tranche_id")
        != _read_json(package / "prepare_manifest.json").get("tranche_id")
        or manifest.get("target_records") != len(targets)
        or manifest.get("target_membership_sha256")
        != _target_membership_sha(targets)
        or canonical_json(manifest.get("provider"))
        != canonical_json(summary.get("provider"))
        or canonical_json(manifest.get("prompt"))
        != canonical_json(summary.get("prompt"))
    ):
        raise TrancheValidationError(f"{scope} run-seal binding mismatch")
    _require_nonempty_string(
        manifest.get("sealed_at"),
        context=f"{scope} seal timestamp",
    )
    implementation = manifest.get("sealing_implementation")
    if not isinstance(implementation, Mapping):
        raise TrancheValidationError(
            f"{scope} sealing implementation is malformed"
        )
    _require_exact_keys(
        implementation,
        {"path", "sha256", "schema_version"},
        context=f"{scope} sealing implementation",
    )
    seal_script = Path(__file__).resolve().with_name(
        "seal_ai_annotation_runs.py"
    )
    if (
        implementation.get("path")
        != "scripts/seal_ai_annotation_runs.py"
        or implementation.get("schema_version")
        != RUN_MANIFEST_SCHEMA_VERSION
        or implementation.get("sha256") != sha256_file(seal_script)
    ):
        raise TrancheValidationError(
            f"{scope} sealing implementation binding mismatch"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TrancheValidationError(
            f"{scope} sealed artifact inventory is malformed"
        )
    artifact_by_path: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise TrancheValidationError(
                f"{scope} sealed artifact is malformed"
            )
        _require_exact_keys(
            item,
            {"path", "role", "bytes", "sha256"},
            context=f"{scope} sealed artifact",
        )
        relative = item.get("path")
        if not isinstance(relative, str) or relative in artifact_by_path:
            raise TrancheValidationError(
                f"{scope} sealed path is duplicate or malformed"
            )
        path = _safe_artifact_path(package, relative)
        if (
            not path.is_file()
            or item.get("bytes") != path.stat().st_size
            or item.get("sha256") != sha256_file(path)
        ):
            raise TrancheValidationError(
                f"{scope} sealed artifact mismatch: {relative}"
            )
        artifact_by_path[relative] = item
    expected_run_files = {
        path.relative_to(package).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
        and path.name not in {RUN_MANIFEST_NAME, RUN_SUMS_NAME}
    }
    if scope == "diagnostic":
        expected_run_files.update(
            {
                "calibration/diagnostic_metrics.json",
                "calibration/diagnostic_gate.json",
            }
        )
    if set(artifact_by_path) != expected_run_files:
        raise TrancheValidationError(
            f"{scope} sealed run inventory is not closed"
        )
    sums = _read_sums(sums_path)
    manifest_relative = manifest_path.relative_to(package).as_posix()
    expected_sums = {
        **{
            relative: str(item["sha256"])
            for relative, item in artifact_by_path.items()
        },
        manifest_relative: sha256_file(manifest_path),
    }
    if sums != expected_sums:
        raise TrancheValidationError(
            f"{scope} run-seal checksum closure mismatch"
        )
    summary_relative = f"runs/{scope}/run_summary.json"
    record_paths = {
        relative
        for relative, item in artifact_by_path.items()
        if item.get("role") == "record"
    }
    if (
        artifact_by_path.get(summary_relative, {}).get("role")
        != "run_summary"
        or len(record_paths) != len(targets)
    ):
        raise TrancheValidationError(
            f"{scope} sealed run lacks required summary/records"
        )
    if scope == "diagnostic":
        if (
            artifact_by_path.get(
                "calibration/diagnostic_metrics.json",
                {},
            ).get("role")
            != "diagnostic_metrics"
            or artifact_by_path.get(
                "calibration/diagnostic_gate.json",
                {},
            ).get("role")
            != "diagnostic_gate"
        ):
            raise TrancheValidationError(
                "Diagnostic seal omits metric/gate bindings"
            )
    replay = manifest.get("replay")
    if not isinstance(replay, Mapping):
        raise TrancheValidationError(f"{scope} replay seal is malformed")
    _require_exact_keys(
        replay,
        {
            "parser",
            "attempts",
            "attempt_outcomes",
            "records_bound_to_unique_valid_attempts",
            "record_attempt_binding_sha256",
            "record_attempt_binding_serialization",
            "normalization_repairs",
            "recovered_failure_markers",
            "unrecovered_failures",
        },
        context=f"{scope} replay seal",
    )
    binding_parts: list[str] = []
    for target in sorted(targets, key=lambda row: row["selection_rank"]):
        annotation_id = str(target["annotation_id"])
        record_relative = (
            f"runs/{scope}/records/"
            f"{int(target['selection_rank']):05d}-{annotation_id}.json"
        )
        record = _read_json(package / record_relative)
        generation = record.get("generation")
        if not isinstance(generation, Mapping):
            raise TrancheValidationError(
                f"{scope} sealed record has no generation metadata"
            )
        batch_id = generation.get("batch_id")
        valid_attempt = generation.get("valid_attempt")
        if (
            not isinstance(batch_id, str)
            or not batch_id
            or isinstance(valid_attempt, bool)
            or not isinstance(valid_attempt, int)
            or valid_attempt < 1
        ):
            raise TrancheValidationError(
                f"{scope} sealed record attempt binding is malformed"
            )
        attempt_relative = (
            f"runs/{scope}/attempts/{batch_id}/"
            f"attempt-{valid_attempt:02d}.json"
        )
        record_artifact = artifact_by_path.get(record_relative)
        attempt_artifact = artifact_by_path.get(attempt_relative)
        if (
            not isinstance(record_artifact, Mapping)
            or record_artifact.get("role") != "record"
            or not isinstance(attempt_artifact, Mapping)
            or attempt_artifact.get("role") != "attempt"
        ):
            raise TrancheValidationError(
                f"{scope} record/attempt seal binding is missing: "
                f"{annotation_id}"
            )
        binding_parts.append(
            f"{target['selection_rank']}\t{annotation_id}\t"
            f"{batch_id}\t{valid_attempt}\t"
            f"{record_artifact['sha256']}\t"
            f"{attempt_artifact['sha256']}\n"
        )
    if (
        replay.get("parser") != "parse_compact_batch_partial"
        or replay.get("records_bound_to_unique_valid_attempts")
        != len(targets)
        or replay.get("unrecovered_failures") != 0
        or replay.get("record_attempt_binding_sha256")
        != sha256_text("".join(binding_parts))
    ):
        raise TrancheValidationError(
            f"{scope} replay seal binding mismatch"
        )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "sums_path": sums_path,
        "artifacts": artifact_by_path,
    }


def _load_run_records(
    package: Path,
    *,
    scope: str,
    targets: list[Mapping[str, Any]],
    allowed_example_ids: set[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
]:
    run_root = package / "runs" / scope
    summary = _read_json(run_root / "run_summary.json")
    _validate_run_summary(
        summary,
        scope=scope,
        target_count=len(targets),
    )
    target_by_id = {
        str(row["annotation_id"]): row for row in targets
    }
    if len(target_by_id) != len(targets):
        raise TrancheValidationError(
            f"{scope} target IDs are not unique"
        )
    seal = _verify_run_seal(
        package,
        scope=scope,
        targets=targets,
        summary=summary,
    )
    record_paths = sorted((run_root / "records").glob("*.json"))
    if len(record_paths) != len(targets):
        raise TrancheValidationError(
            f"{scope.capitalize()} run must contain exactly "
            f"{len(targets)} records"
        )
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in record_paths:
        record = _read_json(path)
        _require_exact_keys(
            record,
            RUN_RECORD_KEYS,
            context=f"{scope} run record {path.name}",
        )
        annotation_id = _require_nonempty_string(
            record.get("annotation_id"),
            context=f"{scope} run annotation_id",
        )
        target = target_by_id.get(annotation_id)
        if target is None or annotation_id in hashes:
            raise TrancheValidationError(
                f"{scope} run record is duplicate/outside target: "
                f"{annotation_id}"
            )
        expected_name = (
            f"{int(target['selection_rank']):05d}-{annotation_id}.json"
        )
        if (
            path.name != expected_name
            or record.get("schema_version") != RUN_RECORD_SCHEMA_VERSION
            or record.get("artifact_type") != ARTIFACT_STATUS
            or record.get("scope") != scope
            or record.get("selection_rank") != target["selection_rank"]
            or record.get("review_text_sha256")
            != target["review_text_sha256"]
            or not isinstance(record.get("normalization_repairs"), list)
        ):
            raise TrancheValidationError(
                f"{scope} run record contract mismatch: {annotation_id}"
            )
        _validate_generation(
            record.get("generation"),
            summary=summary,
            annotation_id=annotation_id,
            allowed_example_ids=allowed_example_ids,
        )
        normalized = validate_canonical_annotation(
            str(target["reviewContent"]),
            record.get("annotation"),
        )
        _validate_normalization_repairs(
            review_text=str(target["reviewContent"]),
            annotation=normalized,
            repairs=record["normalization_repairs"],
        )
        records.append(record)
        hashes[annotation_id] = sha256_file(path)
    if set(hashes) != set(target_by_id):
        raise TrancheValidationError(
            f"{scope} run/target ID join mismatch"
        )
    for failure_path in (run_root / "failed").glob("*.json"):
        failure = _read_json(failure_path)
        if failure.get("annotation_id") not in hashes:
            raise TrancheValidationError(
                f"{scope} run has an unrecovered terminal failure: "
                f"{failure_path.name}"
            )
    records.sort(key=lambda row: row["selection_rank"])
    if canonical_json(summary.get("label_summary")) != canonical_json(
        count_labels(records)
    ):
        raise TrancheValidationError(
            f"{scope} run-summary label counts mismatch"
        )
    return records, summary, hashes, seal


def _recompute_calibration_metrics(
    expected_rows: list[Mapping[str, Any]],
    predicted_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independent implementation of the frozen diagnostic metrics."""

    expected = {
        str(row["calibration_id"]): row["annotation"]
        for row in expected_rows
    }
    predicted = {
        str(row["annotation_id"]): row["annotation"]
        for row in predicted_rows
    }
    if (
        len(expected) != len(expected_rows)
        or len(predicted) != len(predicted_rows)
        or set(expected) != set(predicted)
    ):
        raise TrancheValidationError(
            "Diagnostic prediction/holdout ID join mismatch"
        )
    status_correct = vector_correct = cell_correct = 0
    mentioned_tp = mentioned_fp = mentioned_fn = 0
    joint_mentioned = joint_polarity_correct = 0
    confusion: Counter[tuple[str, str]] = Counter()
    per_aspect: dict[str, Counter[str]] = {
        aspect: Counter() for aspect in ASPECT_COLUMNS
    }
    differences: list[dict[str, Any]] = []
    for calibration_id in sorted(expected):
        human = expected[calibration_id]
        ai = predicted[calibration_id]
        human_labels = [
            aspect["label"] for aspect in human["aspects"]
        ]
        ai_labels = [aspect["label"] for aspect in ai["aspects"]]
        status_match = (
            human["annotation_status"] == ai["annotation_status"]
        )
        vector_match = human_labels == ai_labels
        status_correct += int(status_match)
        vector_correct += int(vector_match)
        record_differences: list[dict[str, Any]] = []
        for index, (aspect, human_label, ai_label) in enumerate(
            zip(ASPECT_COLUMNS, human_labels, ai_labels, strict=True)
        ):
            match = human_label == ai_label
            cell_correct += int(match)
            per_aspect[aspect]["correct"] += int(match)
            per_aspect[aspect]["total"] += 1
            confusion[(str(human_label), str(ai_label))] += 1
            human_mentioned = human_label not in {2, None}
            ai_mentioned = ai_label not in {2, None}
            if human_mentioned and ai_mentioned:
                mentioned_tp += 1
                joint_mentioned += 1
                joint_polarity_correct += int(match)
            elif ai_mentioned:
                mentioned_fp += 1
            elif human_mentioned:
                mentioned_fn += 1
            if not match:
                record_differences.append(
                    {
                        "aspect_index": index + 1,
                        "aspect": aspect,
                        "human": human_label,
                        "ai": ai_label,
                    }
                )
        if not status_match or record_differences:
            differences.append(
                {
                    "calibration_id": calibration_id,
                    "human_status": human["annotation_status"],
                    "ai_status": ai["annotation_status"],
                    "aspect_differences": record_differences,
                }
            )
    total = len(expected)
    cell_total = total * len(ASPECT_COLUMNS)
    precision_denominator = mentioned_tp + mentioned_fp
    recall_denominator = mentioned_tp + mentioned_fn
    precision = (
        mentioned_tp / precision_denominator
        if precision_denominator
        else 1.0
    )
    recall = (
        mentioned_tp / recall_denominator
        if recall_denominator
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "scope": "AI_ALIGNMENT_TO_AI_ASSISTED_HUMAN_CONFIRMED_HOLDOUT",
        "accuracy_claim_permitted": False,
        "records": total,
        "status_exact": {
            "numerator": status_correct,
            "denominator": total,
            "rate": status_correct / total if total else math.nan,
        },
        "full_vector_exact": {
            "numerator": vector_correct,
            "denominator": total,
            "rate": vector_correct / total if total else math.nan,
        },
        "aspect_cell_exact": {
            "numerator": cell_correct,
            "denominator": cell_total,
            "rate": cell_correct / cell_total if cell_total else math.nan,
        },
        "mentioned_detection": {
            "true_positive": mentioned_tp,
            "false_positive": mentioned_fp,
            "false_negative": mentioned_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "polarity_exact_when_both_mentioned": {
            "numerator": joint_polarity_correct,
            "denominator": joint_mentioned,
            "rate": (
                joint_polarity_correct / joint_mentioned
                if joint_mentioned
                else math.nan
            ),
        },
        "per_aspect_cell_exact": {
            aspect: {
                "numerator": counts["correct"],
                "denominator": counts["total"],
                "rate": (
                    counts["correct"] / counts["total"]
                    if counts["total"]
                    else math.nan
                ),
            }
            for aspect, counts in per_aspect.items()
        },
        "label_confusion": {
            f"{human}->{ai}": count
            for (human, ai), count in sorted(confusion.items())
        },
        "records_with_differences": differences,
        "limitations": [
            "The human-confirmed records were reviewed from seeded AI suggestions, "
            "so confirmation bias is possible.",
            "The holdout is diagnostic prompt alignment, not independent accuracy "
            "or inter-annotator agreement.",
        ],
    }


def _recompute_calibration_gate(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "status_exact_at_least_0_85": (
            metrics["status_exact"]["rate"] >= 0.85
        ),
        "aspect_cell_exact_at_least_0_90": (
            metrics["aspect_cell_exact"]["rate"] >= 0.90
        ),
        "mentioned_f1_at_least_0_80": (
            metrics["mentioned_detection"]["f1"] >= 0.80
        ),
        "joint_polarity_at_least_0_80": (
            metrics["polarity_exact_when_both_mentioned"]["rate"] >= 0.80
        ),
        "full_vector_exact_at_least_0_50": (
            metrics["full_vector_exact"]["rate"] >= 0.50
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "interpretation": (
            "Technical alignment gate only; it does not establish human-gold "
            "accuracy."
        ),
    }


def _load_diagnostic_records(
    package: Path,
    context: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    targets = [
        {
            "annotation_id": calibration_id,
            "reviewContent": context["accepted_calibration_by_id"][
                calibration_id
            ]["reviewContent"],
            "review_text_sha256": context[
                "accepted_calibration_by_id"
            ][calibration_id]["review_text_sha256"],
            "selection_rank": index,
        }
        for index, calibration_id in enumerate(
            context["diagnostic_holdout_ids"],
            1,
        )
    ]
    records, summary, hashes, seal = _load_run_records(
        package,
        scope="diagnostic",
        targets=targets,
        allowed_example_ids=set(context["prompt_calibration_ids"]),
    )
    expected_rows = [
        context["accepted_calibration_by_id"][calibration_id]
        for calibration_id in context["diagnostic_holdout_ids"]
    ]
    predictions = [
        {
            "annotation_id": record["annotation_id"],
            "annotation": record["annotation"],
        }
        for record in records
    ]
    recomputed_metrics = _recompute_calibration_metrics(
        expected_rows,
        predictions,
    )
    recomputed_gate = _recompute_calibration_gate(recomputed_metrics)
    stored_metrics = _read_json(
        package / "calibration" / "diagnostic_metrics.json"
    )
    stored_gate = _read_json(
        package / "calibration" / "diagnostic_gate.json"
    )
    if canonical_json(stored_metrics) != canonical_json(
        recomputed_metrics
    ):
        raise TrancheValidationError(
            "Diagnostic metrics differ from independent recomputation"
        )
    if canonical_json(stored_gate) != canonical_json(recomputed_gate):
        raise TrancheValidationError(
            "Diagnostic gate differs from independent recomputation"
        )
    if canonical_json(summary.get("diagnostic_gate")) != canonical_json(
        recomputed_gate
    ):
        raise TrancheValidationError(
            "Diagnostic run-summary gate differs from recomputation"
        )
    if recomputed_gate.get("status") != "PASS":
        raise TrancheValidationError(
            "Independently recomputed diagnostic gate is not PASS"
        )
    return (
        records,
        summary,
        hashes,
        seal,
        recomputed_metrics,
        recomputed_gate,
    )


def _load_primary_records(
    package: Path,
    context: dict[str, Any],
    *,
    expected_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    (
        _diagnostic_records,
        diagnostic_summary,
        _diagnostic_hashes,
        diagnostic_seal,
        diagnostic_metrics,
        diagnostic_gate,
    ) = _load_diagnostic_records(package, context)
    records, summary, hashes, primary_seal = _load_run_records(
        package,
        scope="primary",
        targets=list(context["blind_rows"]),
        allowed_example_ids=set(
            context["accepted_calibration_by_id"]
        ),
    )
    if len(records) != expected_records:
        raise TrancheValidationError(
            f"Primary run is incomplete: {len(records)}/{expected_records}"
        )
    diagnostic_config = run_execution_config(diagnostic_summary)
    primary_config = run_execution_config(summary)
    if canonical_json(diagnostic_config) != canonical_json(primary_config):
        raise TrancheValidationError(
            "Diagnostic/primary execution configuration mismatch"
        )
    context["_diagnostic_records"] = _diagnostic_records
    context["_diagnostic_summary"] = diagnostic_summary
    context["_diagnostic_seal"] = diagnostic_seal
    context["_diagnostic_metrics"] = diagnostic_metrics
    context["_diagnostic_gate"] = diagnostic_gate
    context["_primary_seal"] = primary_seal
    return records, summary, hashes


def risk_flags(record: Mapping[str, Any]) -> list[str]:
    """Return deterministic human-review risk flags for one pseudo-label."""

    annotation = record["annotation"]
    labels = [aspect["label"] for aspect in annotation["aspects"]]
    flags: list[str] = []
    status = annotation["annotation_status"]
    if status == "ESCALATE":
        flags.append("STATUS_ESCALATE")
    elif status == "REJECT_NON_REVIEW":
        flags.append("STATUS_REJECT_NON_REVIEW")
    if record.get("normalization_repairs"):
        flags.append("NORMALIZATION_REPAIR")
    if 0 in labels:
        flags.append("NEUTRAL_ASPECT")
    if "1, -1" in labels:
        flags.append("MIXED_ASPECT")
    mentioned = sum(label not in {2, None} for label in labels)
    if status != "REJECT_NON_REVIEW" and mentioned == 0:
        flags.append("ALL_ASPECTS_ABSENT")
    if mentioned >= 5:
        flags.append("MANY_MENTIONED_ASPECTS")
    has_uncertainty = bool(annotation["review_uncertainty_codes"]) or any(
        aspect["uncertainty_codes"] for aspect in annotation["aspects"]
    )
    if has_uncertainty:
        flags.append("UNCERTAINTY_PRESENT")
    if str(annotation.get("notes", "")).strip():
        flags.append("NONEMPTY_ANNOTATION_NOTES")
    review_text = str(record.get("reviewContent", ""))
    if _MARKETING_RE.search(review_text):
        flags.append("MARKETING_LIKE_TEXT")
    evidence_owners: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    overlong = False
    for aspect_index, aspect in enumerate(annotation["aspects"]):
        for evidence in aspect["evidence"]:
            text = str(evidence["text"])
            span = (
                int(evidence["start"]),
                int(evidence["end"]),
                text,
            )
            evidence_owners[span].add(aspect_index)
            if len(text) > 160 or (
                len(text) > 80
                and review_text
                and len(text) / len(review_text) >= 0.8
            ):
                overlong = True
    if any(len(owners) > 1 for owners in evidence_owners.values()):
        flags.append("CROSS_ASPECT_REUSED_EVIDENCE")
    if overlong:
        flags.append("OVERLONG_EVIDENCE")
    return flags


def deterministic_audit_selected(
    tranche_id: str,
    annotation_id: str,
) -> bool:
    digest = hashlib.sha256(
        f"{tranche_id}\0HUMAN_AUDIT_10_PERCENT\0{annotation_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") < (1 << 64) // 10


def _audit_rank(
    tranche_id: str,
    dimension: str,
    key: str,
    annotation_id: str,
) -> tuple[bytes, str]:
    return (
        hashlib.sha256(
            f"{tranche_id}\0{dimension}\0{key}\0{annotation_id}".encode(
                "utf-8"
            )
        ).digest(),
        annotation_id,
    )


def deterministic_stratified_audit_plan(
    records: Iterable[Mapping[str, Any]],
    *,
    tranche_id: str,
) -> dict[str, list[str]]:
    """Build a stable audit union with source-stratum and rare-label coverage."""

    rows = list(records)
    by_id = {
        str(row["annotation_id"]): row for row in rows
    }
    if len(by_id) != len(rows):
        raise TrancheValidationError(
            "Cannot stratify audit over duplicate annotation IDs"
        )
    reasons: dict[str, set[str]] = {
        annotation_id: set() for annotation_id in by_id
    }
    for annotation_id in by_id:
        if deterministic_audit_selected(tranche_id, annotation_id):
            reasons[annotation_id].add(
                "DETERMINISTIC_RANDOM_AUDIT_10_PERCENT"
            )

    source_dimensions = {
        "rating": "STRATIFIED_RATING",
        "category": "STRATIFIED_CATEGORY",
        "collection_transport": "STRATIFIED_TRANSPORT",
    }
    for field, reason_prefix in source_dimensions.items():
        groups: dict[str, list[str]] = defaultdict(list)
        display_keys: dict[str, str] = {}
        for annotation_id, row in by_id.items():
            source = row.get("source")
            if not isinstance(source, Mapping):
                raise TrancheValidationError(
                    f"Audit source metadata missing: {annotation_id}"
                )
            raw_key = source.get(field)
            key = canonical_json(raw_key)
            display_keys[key] = (
                "<blank>" if raw_key == "" else str(raw_key)
            )
            groups[key].append(annotation_id)
        for key, members in sorted(groups.items()):
            selected = min(
                members,
                key=lambda annotation_id: _audit_rank(
                    tranche_id,
                    field,
                    key,
                    annotation_id,
                ),
            )
            reasons[selected].add(
                f"{reason_prefix}:{display_keys[key]}"
            )

    label_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for annotation_id, row in by_id.items():
        for aspect in row["annotation"]["aspects"]:
            label = aspect["label"]
            if label not in {2, None}:
                label_groups[
                    (str(aspect["aspect"]), str(label))
                ].append(annotation_id)
    rare_limit = max(10, math.ceil(len(rows) * 0.01))
    for (aspect, label), members in sorted(label_groups.items()):
        if len(members) > rare_limit:
            continue
        key = f"{aspect}\0{label}"
        selected = min(
            members,
            key=lambda annotation_id: _audit_rank(
                tranche_id,
                "rare_label",
                key,
                annotation_id,
            ),
        )
        reasons[selected].add(
            f"STRATIFIED_RARE_LABEL:{aspect}:{label}"
        )
    return {
        annotation_id: sorted(values)
        for annotation_id, values in reasons.items()
        if values
    }


def queue_priority(flags: Iterable[str], *, audit_only: bool) -> str:
    flag_set = set(flags)
    if flag_set.intersection(
        {
            "STATUS_ESCALATE",
            "STATUS_REJECT_NON_REVIEW",
            "NORMALIZATION_REPAIR",
            "ALL_ASPECTS_ABSENT",
            "UNCERTAINTY_PRESENT",
            "MARKETING_LIKE_TEXT",
            "CROSS_ASPECT_REUSED_EVIDENCE",
            "OVERLONG_EVIDENCE",
        }
    ):
        return "HIGH"
    if flag_set:
        return "MEDIUM"
    return "AUDIT" if audit_only else "MEDIUM"


def stable_queue_id(tranche_id: str, annotation_id: str) -> str:
    digest = sha256_text(f"{tranche_id}\0{annotation_id}\0HUMAN_REVIEW")
    return f"ahr-{digest[:20]}"


def pseudo_record_sha256(record: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(record)))


def _csv_label(value: Any) -> str:
    return "" if value is None else str(value)


def release_provenance_source_plan(
    package: Path,
    context: Mapping[str, Any],
) -> dict[str, Path]:
    """Return the exact self-contained provenance copy plan."""

    package = package.resolve()
    plan: dict[str, Path] = {
        "provenance/prepared/prepare_manifest.json": (
            package / "prepare_manifest.json"
        ),
        "provenance/prepared/INPUT_SHA256SUMS.txt": (
            package / "INPUT_SHA256SUMS.txt"
        ),
        "provenance/diagnostic/diagnostic_metrics.json": (
            package / "calibration" / "diagnostic_metrics.json"
        ),
        "provenance/diagnostic/diagnostic_gate.json": (
            package / "calibration" / "diagnostic_gate.json"
        ),
        "provenance/diagnostic/run_summary.json": (
            package / "runs" / "diagnostic" / "run_summary.json"
        ),
        "provenance/diagnostic/run_manifest.json": (
            package / "runs" / "diagnostic" / RUN_MANIFEST_NAME
        ),
        "provenance/diagnostic/RUN_SHA256SUMS.txt": (
            package / "runs" / "diagnostic" / RUN_SUMS_NAME
        ),
        "provenance/primary/run_summary.json": (
            package / "runs" / "primary" / "run_summary.json"
        ),
        "provenance/primary/run_manifest.json": (
            package / "runs" / "primary" / RUN_MANIFEST_NAME
        ),
        "provenance/primary/RUN_SHA256SUMS.txt": (
            package / "runs" / "primary" / RUN_SUMS_NAME
        ),
    }
    audit_root = package / "audits" / "semantic_audit_60_v1"
    audit_manifest_path = audit_root / "manifest.json"
    audit_sums_path = audit_root / "SHA256SUMS.txt"
    if not audit_manifest_path.is_file() or not audit_sums_path.is_file():
        raise TrancheValidationError(
            "Frozen AI semantic audit is missing; this audit is required "
            "for publication but is not human accuracy evidence"
        )
    audit_manifest = _read_json(audit_manifest_path)
    if (
        audit_manifest.get("artifact_type")
        != "AI_SEMANTIC_AUDIT_NOT_HUMAN_GOLD"
        or audit_manifest.get("status")
        != "FROZEN_PENDING_INDEPENDENT_HUMAN_ADJUDICATION"
        or audit_manifest.get("label_mutations") != 0
    ):
        raise TrancheValidationError(
            "AI semantic-audit status/interpretation mismatch"
        )
    audit_sums = _read_sums(audit_sums_path)
    audit_artifacts = audit_manifest.get("artifacts")
    if not isinstance(audit_artifacts, list):
        raise TrancheValidationError(
            "AI semantic-audit inventory is malformed"
        )
    expected_audit_files = {"manifest.json", "SHA256SUMS.txt"}
    expected_audit_sums = {"manifest.json": sha256_file(audit_manifest_path)}
    for item in audit_artifacts:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("path"),
            str,
        ):
            raise TrancheValidationError(
                "AI semantic-audit artifact is malformed"
            )
        relative = str(item["path"])
        source = _safe_artifact_path(audit_root, relative)
        if (
            not source.is_file()
            or item.get("sha256") != sha256_file(source)
            or item.get("bytes") != source.stat().st_size
        ):
            raise TrancheValidationError(
                f"AI semantic-audit artifact mismatch: {relative}"
            )
        expected_audit_files.add(relative)
        expected_audit_sums[relative] = sha256_file(source)
    if audit_sums != expected_audit_sums:
        raise TrancheValidationError(
            "AI semantic-audit checksum closure mismatch"
        )
    actual_audit_files = {
        path.relative_to(audit_root).as_posix()
        for path in audit_root.rglob("*")
        if path.is_file()
    }
    if actual_audit_files != expected_audit_files:
        raise TrancheValidationError(
            "AI semantic-audit file inventory is not closed"
        )
    source_hashes = audit_manifest.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or source_hashes.get("prepare_manifest_sha256")
        != sha256_file(package / "prepare_manifest.json")
        or source_hashes.get("blind_reviews_sha256")
        != sha256_file(package / "input" / "blind_reviews.jsonl")
        or source_hashes.get("guideline_sha256")
        != context["manifest"]["guideline"]["sha256"]
    ):
        raise TrancheValidationError(
            "AI semantic audit is not bound to this prepared package"
        )
    for source in sorted(audit_root.rglob("*")):
        if source.is_file():
            relative = source.relative_to(audit_root).as_posix()
            plan[f"provenance/semantic_audit/{relative}"] = source
    for relative in sorted(context["inventory"]):
        if relative.startswith("provenance/") or relative in {
            "calibration/human_confirmed.jsonl",
            "calibration/diagnostic_split.json",
        }:
            destination = f"provenance/prepared/{relative}"
            plan[destination] = package / relative
    diagnostic_seal = context.get("_diagnostic_seal")
    if not isinstance(diagnostic_seal, Mapping):
        raise TrancheValidationError(
            "Diagnostic seal context was not validated"
        )
    for relative, item in diagnostic_seal["artifacts"].items():
        if item.get("role") == "record":
            source = package / relative
            plan[
                f"provenance/diagnostic/records/{source.name}"
            ] = source

    scripts_root = Path(__file__).resolve().parent
    software_sources = {
        "finalize_ai_annotation_tranche.py": (
            scripts_root / "finalize_ai_annotation_tranche.py"
        ),
        "validate_ai_annotation_tranche.py": Path(__file__).resolve(),
        "seal_ai_annotation_runs.py": (
            scripts_root / "seal_ai_annotation_runs.py"
        ),
    }
    for name, source in software_sources.items():
        if not source.is_file():
            raise TrancheValidationError(
                f"Required publication software is missing: {source}"
            )
        plan[f"provenance/software/{name}"] = source
    for relative, source in plan.items():
        if not source.is_file():
            raise TrancheValidationError(
                f"Required provenance source is missing: {relative}"
            )
    return plan


def validate_release(
    *,
    package: Path,
    release: Path,
    expected_records: int = TRANCHE_SIZE,
) -> dict[str, Any]:
    """Validate a final pseudo-label release from independent source files."""

    package = package.resolve()
    release = release.resolve()
    context = _validate_prepared_package(
        package,
        expected_records=expected_records,
    )
    run_records, run_summary, run_hashes = _load_primary_records(
        package,
        context,
        expected_records=expected_records,
    )
    run_by_id = {
        row["annotation_id"]: row for row in run_records
    }

    manifest_path = release / "manifest.json"
    manifest = _read_json(manifest_path)
    _require_portable_manifest(manifest)
    if (
        manifest.get("schema_version") != FINAL_MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_type") != ARTIFACT_STATUS
        or manifest.get("status") != ARTIFACT_STATUS
        or manifest.get("target_records") != expected_records
        or manifest.get("tranche_id")
        != context["manifest"].get("tranche_id")
    ):
        raise TrancheValidationError("Final manifest contract mismatch")
    sums = _read_sums(release / "SHA256SUMS.txt")
    inventory = _verify_artifact_inventory(
        root=release,
        artifacts=manifest.get("artifacts"),
        sums=sums,
        manifest_name="manifest.json",
    )
    provenance_plan = release_provenance_source_plan(package, context)
    expected_artifacts = FINAL_ROOT_ARTIFACTS | set(provenance_plan)
    if set(inventory) != expected_artifacts:
        raise TrancheValidationError(
            "Final artifact inventory is not exactly the frozen release set"
        )
    for relative, source in provenance_plan.items():
        published = release / Path(relative)
        if (
            not relative.startswith("provenance/software/")
            and sha256_file(published) != sha256_file(source)
        ):
            raise TrancheValidationError(
                f"Published provenance differs from frozen source: {relative}"
            )
    actual_files = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file()
    }
    expected_files = expected_artifacts | {
        "manifest.json",
        "SHA256SUMS.txt",
    }
    if actual_files != expected_files:
        raise TrancheValidationError(
            "Untracked or missing final-release files; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    pseudo_rows = _read_jsonl(release / "ai_pseudo_labels.jsonl")
    ledger_rows = _read_jsonl(release / "decision_ledger.jsonl")
    queue_rows = _read_jsonl(release / "human_review_queue.jsonl")
    if len(pseudo_rows) != expected_records or len(ledger_rows) != expected_records:
        raise TrancheValidationError("Final JSONL record count is not exact")
    pseudo_by_id = _require_unique(
        pseudo_rows,
        "annotation_id",
        context="published pseudo-labels",
    )
    ledger_by_id = _require_unique(
        ledger_rows,
        "annotation_id",
        context="decision ledger",
    )
    if (
        set(pseudo_by_id) != set(context["blind_by_id"])
        or set(ledger_by_id) != set(pseudo_by_id)
    ):
        raise TrancheValidationError("Published ID join is not closed")

    queue_by_id = _require_unique(
        queue_rows,
        "annotation_id",
        context="human-review queue",
    )
    expected_queue_ids: set[str] = set()
    terminal_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    audit_reason_counts: Counter[str] = Counter()
    tranche_id = manifest["tranche_id"]
    audit_plan = deterministic_stratified_audit_plan(
        pseudo_rows,
        tranche_id=tranche_id,
    )
    for annotation_id, pseudo in pseudo_by_id.items():
        blind = context["blind_by_id"][annotation_id]
        private = context["private_by_id"][annotation_id]
        run = run_by_id[annotation_id]
        expected_pseudo_keys = {
            "schema_version",
            "artifact_status",
            "human_verification_status",
            "tranche_id",
            "selection_rank",
            "annotation_id",
            "sample_id",
            "reviewContent",
            "review_text_sha256",
            "source",
            "annotation",
            "normalization_repairs",
            "generation",
        }
        if (
            set(pseudo) != expected_pseudo_keys
            or pseudo.get("schema_version") != PSEUDO_RECORD_SCHEMA_VERSION
            or pseudo.get("artifact_status") != ARTIFACT_STATUS
            or pseudo.get("human_verification_status")
            != HUMAN_VERIFICATION_STATUS
            or pseudo.get("tranche_id") != tranche_id
            or pseudo.get("selection_rank") != blind["selection_rank"]
            or pseudo.get("sample_id") != private["sample_id"]
            or pseudo.get("reviewContent") != blind["reviewContent"]
            or pseudo.get("review_text_sha256")
            != blind["review_text_sha256"]
            or canonical_json(pseudo.get("annotation"))
            != canonical_json(run["annotation"])
            or canonical_json(pseudo.get("generation"))
            != canonical_json(run["generation"])
            or canonical_json(pseudo.get("normalization_repairs"))
            != canonical_json(run["normalization_repairs"])
        ):
            raise TrancheValidationError(
                f"Published pseudo-label join mismatch: {annotation_id}"
            )
        expected_source = {
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
        if canonical_json(pseudo.get("source")) != canonical_json(expected_source):
            raise TrancheValidationError(
                f"Published source metadata mismatch: {annotation_id}"
            )
        validate_canonical_annotation(
            pseudo["reviewContent"],
            pseudo["annotation"],
        )

        ledger = ledger_by_id[annotation_id]
        flags = risk_flags(pseudo)
        audit_reasons = audit_plan.get(annotation_id, [])
        audit_selected = bool(audit_reasons)
        queued = bool(flags) or bool(audit_reasons)
        if queued:
            expected_queue_ids.add(annotation_id)
        terminal_counts[ledger.get("terminal_state")] += 1
        flag_counts.update(flags)
        audit_reason_counts.update(audit_reasons)
        expected_ledger = {
            "schema_version": DECISION_LEDGER_SCHEMA_VERSION,
            "artifact_status": ARTIFACT_STATUS,
            "annotation_id": annotation_id,
            "sample_id": private["sample_id"],
            "selection_rank": blind["selection_rank"],
            "review_text_sha256": blind["review_text_sha256"],
            "primary_run_record_sha256": run_hashes[annotation_id],
            "pseudo_record_sha256": pseudo_record_sha256(pseudo),
            "validation_decision": "ACCEPT_SCHEMA_VALID_AI_PSEUDO_LABEL",
            "terminal_state": TERMINAL_STATE,
            "risk_flags": flags,
            "deterministic_audit_10_percent": (
                "DETERMINISTIC_RANDOM_AUDIT_10_PERCENT"
                in audit_reasons
            ),
            "deterministic_audit_reasons": audit_reasons,
            "human_review_queued": queued,
        }
        if canonical_json(ledger) != canonical_json(expected_ledger):
            raise TrancheValidationError(
                f"Decision-ledger mismatch: {annotation_id}"
            )

    if set(queue_by_id) != expected_queue_ids:
        raise TrancheValidationError("Human-review queue membership mismatch")
    for annotation_id, queue in queue_by_id.items():
        pseudo = pseudo_by_id[annotation_id]
        flags = risk_flags(pseudo)
        audit_reasons = audit_plan.get(annotation_id, [])
        audit_only = not flags
        expected_queue = {
            "schema_version": HUMAN_QUEUE_SCHEMA_VERSION,
            "artifact_status": ARTIFACT_STATUS,
            "queue_id": stable_queue_id(tranche_id, annotation_id),
            "annotation_id": annotation_id,
            "sample_id": pseudo["sample_id"],
            "selection_rank": pseudo["selection_rank"],
            "priority": queue_priority(flags, audit_only=audit_only),
            "queue_reasons": flags + audit_reasons,
            "human_verification_status": HUMAN_VERIFICATION_STATUS,
            "reviewContent": pseudo["reviewContent"],
            "review_text_sha256": pseudo["review_text_sha256"],
            "ai_annotation": pseudo["annotation"],
        }
        if canonical_json(queue) != canonical_json(expected_queue):
            raise TrancheValidationError(
                f"Human-review queue row mismatch: {annotation_id}"
            )

    csv_path = release / "ai_pseudo_labels_compat.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise TrancheValidationError("Compatibility CSV header mismatch")
        csv_rows = list(reader)
    if len(csv_rows) != expected_records:
        raise TrancheValidationError("Compatibility CSV row count mismatch")
    csv_by_id = _require_unique(
        csv_rows,
        "annotation_id",
        context="compatibility CSV",
    )
    if set(csv_by_id) != set(pseudo_by_id):
        raise TrancheValidationError("Compatibility CSV ID join mismatch")
    for annotation_id, row in csv_by_id.items():
        pseudo = pseudo_by_id[annotation_id]
        labels = {
            aspect["aspect"]: _csv_label(aspect["label"])
            for aspect in pseudo["annotation"]["aspects"]
        }
        expected_csv = {
            "sample_id": pseudo["sample_id"],
            "annotation_id": annotation_id,
            "selection_rank": str(pseudo["selection_rank"]),
            "reviewContent": pseudo["reviewContent"],
            "annotation_status": pseudo["annotation"]["annotation_status"],
            **labels,
            "artifact_status": ARTIFACT_STATUS,
            "human_verification_status": HUMAN_VERIFICATION_STATUS,
            "review_text_sha256": pseudo["review_text_sha256"],
        }
        if row != expected_csv:
            raise TrancheValidationError(
                f"Compatibility CSV row mismatch: {annotation_id}"
            )

    summary = _read_json(release / "summary.json")
    _require_exact_keys(
        summary,
        {
            "schema_version",
            "artifact_status",
            "target_records",
            "pseudo_label_records",
            "human_review_queue_records",
            "human_verification_completed",
            "terminal_state_counts",
            "risk_flag_counts",
            "deterministic_audit_reason_counts",
            "label_summary",
            "review_queue_policy",
            "interpretation",
        },
        context="final summary",
    )
    expected_summary_values = {
        "schema_version": FINAL_SUMMARY_SCHEMA_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "target_records": expected_records,
        "pseudo_label_records": expected_records,
        "human_review_queue_records": len(queue_rows),
        "human_verification_completed": 0,
        "terminal_state_counts": {TERMINAL_STATE: expected_records},
        "risk_flag_counts": dict(sorted(flag_counts.items())),
        "deterministic_audit_reason_counts": dict(
            sorted(audit_reason_counts.items())
        ),
        "label_summary": count_labels(pseudo_rows),
    }
    for key, expected in expected_summary_values.items():
        if canonical_json(summary.get(key)) != canonical_json(expected):
            raise TrancheValidationError(f"Final summary mismatch: {key}")
    expected_queue_policy = {
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
    }
    if canonical_json(summary.get("review_queue_policy")) != canonical_json(
        expected_queue_policy
    ) or summary.get("interpretation") != (
        "Schema-valid AI pseudo-labels only; no record is human-verified "
        "by this publication step."
    ):
        raise TrancheValidationError(
            "Final summary interpretation/policy mismatch"
        )
    if terminal_counts != Counter({TERMINAL_STATE: expected_records}):
        raise TrancheValidationError("Decision ledger terminal states are invalid")

    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "status",
            "tranche_id",
            "target_records",
            "built_at",
            "prepared_package",
            "diagnostic_run",
            "primary_run",
            "semantic_audit",
            "human_verification",
            "software",
            "limitations",
            "artifacts",
        },
        context="final manifest",
    )
    _require_nonempty_string(
        manifest.get("built_at"),
        context="final manifest built_at",
    )
    expected_human_verification = {
        "status": HUMAN_VERIFICATION_STATUS,
        "completed_records": 0,
        "queue_records": len(queue_rows),
    }
    if canonical_json(
        manifest.get("human_verification")
    ) != canonical_json(expected_human_verification):
        raise TrancheValidationError(
            "Final manifest human-verification state mismatch"
        )
    expected_limitations = [
        "All annotations are AI pseudo-labels pending human review.",
        "A PASS diagnostic gate measures alignment to an AI-assisted "
        "human-confirmed holdout; it is not an independent accuracy "
        "or inter-annotator-agreement estimate.",
        "The compatibility CSV is a projection; JSONL is canonical "
        "because it preserves evidence, uncertainty, and provenance.",
        "The bundled AI semantic audit is not human accuracy evidence "
        "and remains pending independent human adjudication.",
    ]
    if manifest.get("limitations") != expected_limitations:
        raise TrancheValidationError(
            "Final manifest limitations/claim boundary mismatch"
        )
    diagnostic = manifest.get("diagnostic_run")
    primary = manifest.get("primary_run")
    diagnostic_summary = context["_diagnostic_summary"]
    expected_diagnostic_block = {
        "summary": {
            "path": "provenance/diagnostic/run_summary.json",
            "sha256": sha256_file(
                package / "runs" / "diagnostic" / "run_summary.json"
            ),
        },
        "metrics": {
            "path": "provenance/diagnostic/diagnostic_metrics.json",
            "sha256": sha256_file(
                package / "calibration" / "diagnostic_metrics.json"
            ),
        },
        "gate": {
            "path": "provenance/diagnostic/diagnostic_gate.json",
            "sha256": sha256_file(
                package / "calibration" / "diagnostic_gate.json"
            ),
            "status": "PASS",
        },
        "sealed_manifest": {
            "path": "provenance/diagnostic/run_manifest.json",
            "sha256": sha256_file(
                package / "runs" / "diagnostic" / RUN_MANIFEST_NAME
            ),
        },
        "sealed_checksums": {
            "path": "provenance/diagnostic/RUN_SHA256SUMS.txt",
            "sha256": sha256_file(
                package / "runs" / "diagnostic" / RUN_SUMS_NAME
            ),
        },
        "execution_config": run_execution_config(diagnostic_summary),
    }
    expected_primary_block = {
        "summary": {
            "path": "provenance/primary/run_summary.json",
            "sha256": sha256_file(
                package / "runs" / "primary" / "run_summary.json"
            ),
        },
        "sealed_manifest": {
            "path": "provenance/primary/run_manifest.json",
            "sha256": sha256_file(
                package / "runs" / "primary" / RUN_MANIFEST_NAME
            ),
        },
        "sealed_checksums": {
            "path": "provenance/primary/RUN_SHA256SUMS.txt",
            "sha256": sha256_file(
                package / "runs" / "primary" / RUN_SUMS_NAME
            ),
        },
        "execution_config": run_execution_config(run_summary),
    }
    if canonical_json(diagnostic) != canonical_json(
        expected_diagnostic_block
    ):
        raise TrancheValidationError(
            "Final manifest diagnostic-run provenance mismatch"
        )
    if canonical_json(primary) != canonical_json(expected_primary_block):
        raise TrancheValidationError(
            "Final manifest primary-run provenance mismatch"
        )
    audit_manifest_path = (
        package
        / "audits"
        / "semantic_audit_60_v1"
        / "manifest.json"
    )
    audit_sums_path = audit_manifest_path.with_name("SHA256SUMS.txt")
    audit_manifest = _read_json(audit_manifest_path)
    expected_semantic_audit = {
        "artifact_type": "AI_SEMANTIC_AUDIT_NOT_HUMAN_GOLD",
        "status": audit_manifest["status"],
        "human_accuracy_claim_permitted": False,
        "manifest": {
            "path": "provenance/semantic_audit/manifest.json",
            "sha256": sha256_file(audit_manifest_path),
        },
        "checksums": {
            "path": "provenance/semantic_audit/SHA256SUMS.txt",
            "sha256": sha256_file(audit_sums_path),
        },
    }
    if canonical_json(manifest.get("semantic_audit")) != canonical_json(
        expected_semantic_audit
    ):
        raise TrancheValidationError(
            "Final manifest AI semantic-audit provenance mismatch"
        )
    prepared = manifest.get("prepared_package")
    expected_prepared = {
        "logical_location": "provenance/prepared",
        "prepare_manifest_sha256": sha256_file(
            package / "prepare_manifest.json"
        ),
        "input_checksums_sha256": sha256_file(
            package / "INPUT_SHA256SUMS.txt"
        ),
        "ordered_membership_sha256": context["manifest"]["selection"][
            "ordered_membership_sha256"
        ],
        "source_release_id": context["manifest"]["source_release"][
            "release_id"
        ],
        "human_reference_manifest_sha256": context["manifest"]["selection"][
            "human_reference_manifest_sha256"
        ],
        "group_reservations_sha256": context["manifest"]["selection"][
            "group_reservations_sha256"
        ],
    }
    if canonical_json(prepared) != canonical_json(expected_prepared):
        raise TrancheValidationError(
            "Final manifest prepared-package provenance mismatch"
        )
    software = manifest.get("software")
    expected_software: dict[str, dict[str, str]] = {}
    for role, name in (
        ("finalizer", "finalize_ai_annotation_tranche.py"),
        ("validator", "validate_ai_annotation_tranche.py"),
        ("run_sealer", "seal_ai_annotation_runs.py"),
    ):
        relative = f"provenance/software/{name}"
        expected_software[role] = {
            "path": relative,
            "sha256": sha256_file(release / relative),
        }
    if canonical_json(software) != canonical_json(expected_software):
        raise TrancheValidationError(
            "Final manifest software provenance mismatch"
        )
    return {
        "status": "VALID",
        "artifact_status": ARTIFACT_STATUS,
        "records": expected_records,
        "human_review_queue_records": len(queue_rows),
        "reference_overlap": 0,
        "reserved_group_overlap": 0,
        "terminal_states": dict(sorted(terminal_counts.items())),
        "release_manifest_sha256": sha256_file(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("data/annotations/absa_ai_tranche_5000_v1_20260727"),
    )
    parser.add_argument(
        "--release",
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
        prepared_manifest = _read_json(
            args.package / "prepare_manifest.json"
        )
        expected_records = prepared_manifest.get("target_records")
    if (
        not isinstance(expected_records, int)
        or isinstance(expected_records, bool)
        or expected_records <= 0
    ):
        raise TrancheValidationError(
            "Prepared target_records is invalid"
        )
    report = validate_release(
        package=args.package,
        release=args.release,
        expected_records=expected_records,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
