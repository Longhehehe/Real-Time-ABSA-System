"""Validate and compare two LLM ABSA pilot passes.

This evaluator is deliberately fail-closed.  It verifies the frozen pilot
package, both run checksum closures, terminal records and failures, request
attempt ledgers, and deterministic evidence-offset normalization before it
computes any agreement statistic.

The resulting agreement is a repeat-consistency measurement, not an accuracy
measurement.  Without an independently adjudicated human reference set the
semantic quality gate remains ``NOT_EVALUATED_NO_HUMAN_GOLD``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lazada_collector.llm_annotation import (  # noqa: E402
    ASPECT_COLUMNS,
    LLM_ANNOTATION_SCHEMA_VERSION,
    PROMPT_VERSION,
    AnnotationValidationError,
    annotation_fingerprint,
    build_user_message,
    canonical_json,
    compare_annotation_passes,
    label_vector,
    parse_model_json,
    sha256_text,
    validate_and_normalize_annotation,
    validate_output_schema_contract,
)


EVALUATION_SCHEMA_VERSION = "absa-llm-pilot-evaluation/1.0.0"
BLIND_INPUT_SCHEMA_VERSION = "absa-llm-blind-input/1.0.0"
SEMANTIC_GATE_STATUS = "NOT_EVALUATED_NO_HUMAN_GOLD"
BLIND_ID_RE = re.compile(r"\Allmp-[0-9a-f]{20}\Z")
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

PILOT_REQUIRED_ARTIFACTS = {
    "pilot_blind.jsonl",
    "pilot_private_index.jsonl",
    "system_prompt_direct.txt",
    "system_prompt_evidence_first.txt",
    "legacy_annotation_schema.json",
    "llm_annotation_output_schema.json",
    "ABSA_ANNOTATION_GUIDELINE_V2.md",
}

PRIVATE_INDEX_FIELDS = {
    "blind_id",
    "pilot_rank",
    "sample_id",
    "review_text_sha256",
    "parent_canonical_row",
    "curation_status",
    "category",
    "rating",
    "collection_transport",
}

FINAL_RECORD_FIELDS = {
    "annotation",
    "annotation_fingerprint",
    "annotation_id",
    "annotation_output_schema_sha256",
    "artifact_type",
    "attempt_count",
    "attempts",
    "completed_at",
    "finish_reason",
    "guideline_sha256",
    "model",
    "pass_id",
    "prompt_sha256",
    "prompt_variant",
    "prompt_version",
    "provider_request_id",
    "raw_response",
    "raw_response_sha256",
    "request_sha256",
    "review_text_sha256",
    "run_fingerprint",
    "schema_version",
    "usage",
}

FAILURE_FIELDS = {
    "annotation_id",
    "artifact_type",
    "attempts",
    "failed_at",
    "pass_id",
    "review_text_sha256",
    "run_fingerprint",
}

RUN_MANIFEST_FIELDS = {
    "annotation_records",
    "artifact_type",
    "aspect_label_counts",
    "completed_at",
    "failed_records",
    "missing_annotation_ids",
    "run_config",
    "selected_records",
    "status",
    "status_counts",
    "stop_reason",
}

RUN_CONFIG_FIELDS = {
    "annotation_output_schema_sha256",
    "backend",
    "code_sha256",
    "config_sha256",
    "data_transmission_notice",
    "generation",
    "guideline_sha256",
    "input_sha256",
    "pass_id",
    "pilot",
    "prompt_sha256",
    "prompt_variant",
    "prompt_version",
    "run_fingerprint",
    "schema_version",
    "seed",
    "selected_records",
}

ATTEMPT_SUMMARY_FIELDS = {
    "attempt",
    "attempt_path",
    "outcome",
    "request_sha256",
}

USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_eval_count",
    "eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = parse_model_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, AnnotationValidationError) as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot open JSONL: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"Blank lines are forbidden in JSONL: {path}:{line_number}"
                )
            try:
                value = parse_model_json(line)
            except AnnotationValidationError as exc:
                raise ValueError(
                    f"Invalid JSON object at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _resolve_relative(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Artifact path must be a non-empty string")
    if "\\" in relative_path:
        raise ValueError(
            f"Artifact paths must use forward slashes: {relative_path!r}"
        )
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe artifact path: {relative_path!r}")
    candidate = root.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Artifact escapes its package: {relative_path!r}"
        ) from exc
    return candidate


def _read_checksum_file(
    checksum_path: Path,
    *,
    root: Path,
    verify_files: bool,
) -> dict[str, str]:
    entries: dict[str, str] = {}
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.rstrip("\n")
            if not stripped:
                raise ValueError(
                    f"Blank checksum line: {checksum_path}:{line_number}"
                )
            parts = stripped.split("  ", 1)
            if len(parts) != 2 or SHA256_RE.fullmatch(parts[0]) is None:
                raise ValueError(
                    f"Invalid checksum line: {checksum_path}:{line_number}"
                )
            checksum, relative_path = parts
            if relative_path in entries:
                raise ValueError(
                    f"Duplicate checksum path in {checksum_path}: "
                    f"{relative_path}"
                )
            artifact_path = _resolve_relative(root, relative_path)
            if verify_files:
                if not artifact_path.is_file():
                    raise FileNotFoundError(artifact_path)
                if _sha256_file(artifact_path) != checksum:
                    raise ValueError(
                        f"SHA256SUMS mismatch: {artifact_path}"
                    )
            entries[relative_path] = checksum
    return entries


def _strict_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _record_name(row: Mapping[str, Any]) -> str:
    return f"{row['pilot_rank']:04d}-{row['annotation_id']}.json"


def _validate_pilot_package(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    required_paths = {
        "manifest": package_dir / "manifest.json",
        "checksums": package_dir / "SHA256SUMS.txt",
        "blind": package_dir / "pilot_blind.jsonl",
        "private": package_dir / "pilot_private_index.jsonl",
        "guideline": package_dir / "ABSA_ANNOTATION_GUIDELINE_V2.md",
        "llm_schema": package_dir / "llm_annotation_output_schema.json",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = _read_json(required_paths["manifest"])
    if manifest.get("artifact_type") != "LLM_PSEUDO_LABEL_PILOT_INPUT":
        raise ValueError("Unexpected pilot artifact_type")
    if manifest.get("status") != "PREPARED_NOT_LABELED":
        raise ValueError("Pilot package is not frozen in prepared state")
    if manifest.get("blind_input_schema_version") != (
        BLIND_INPUT_SCHEMA_VERSION
    ):
        raise ValueError("Pilot blind-input schema version mismatch")
    if manifest.get("annotation_output_schema_version") != (
        LLM_ANNOTATION_SCHEMA_VERSION
    ):
        raise ValueError("Pilot annotation schema version mismatch")
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Pilot prompt version mismatch")
    if manifest.get("model_visible_fields") != [
        "schema_version",
        "blind_id",
        "reviewContent",
    ]:
        raise ValueError("Pilot model-visible field allowlist mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Pilot manifest has no artifact inventory")
    inventory: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Pilot artifact inventory entry is not an object")
        if not set(artifact).issubset({"path", "bytes", "sha256", "records"}):
            raise ValueError("Pilot artifact inventory has unexpected fields")
        if not {"path", "bytes", "sha256"}.issubset(artifact):
            raise ValueError("Pilot artifact inventory entry is incomplete")
        relative_path = artifact.get("path")
        path = _resolve_relative(package_dir, relative_path)
        if relative_path in inventory:
            raise ValueError(f"Duplicate pilot artifact: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if (
            not isinstance(artifact.get("bytes"), int)
            or isinstance(artifact.get("bytes"), bool)
            or artifact["bytes"] < 0
            or artifact["bytes"] != path.stat().st_size
        ):
            raise ValueError(f"Pilot artifact byte mismatch: {relative_path}")
        actual_sha = _sha256_file(path)
        if artifact.get("sha256") != actual_sha:
            raise ValueError(
                f"Pilot artifact checksum mismatch: {relative_path}"
            )
        inventory[str(relative_path)] = artifact
    if not PILOT_REQUIRED_ARTIFACTS.issubset(inventory):
        raise ValueError(
            "Pilot manifest omits required artifacts: "
            f"{sorted(PILOT_REQUIRED_ARTIFACTS - set(inventory))}"
        )

    checksum_entries = _read_checksum_file(
        required_paths["checksums"],
        root=package_dir,
        verify_files=True,
    )
    expected_checksum_paths = set(inventory) | {"manifest.json"}
    if set(checksum_entries) != expected_checksum_paths:
        raise ValueError(
            "Pilot SHA256SUMS closure mismatch; "
            f"missing={sorted(expected_checksum_paths - set(checksum_entries))}, "
            f"extra={sorted(set(checksum_entries) - expected_checksum_paths)}"
        )

    guideline_sha = _sha256_file(required_paths["guideline"])
    llm_schema_sha = _sha256_file(required_paths["llm_schema"])
    if manifest.get("guideline_sha256") != guideline_sha:
        raise ValueError("Pilot guideline checksum pin mismatch")
    if manifest.get("llm_output_schema_sha256") != llm_schema_sha:
        raise ValueError("Pilot output-schema checksum pin mismatch")
    validate_output_schema_contract(_read_json(required_paths["llm_schema"]))

    blind_rows = _read_jsonl(required_paths["blind"])
    private_rows = _read_jsonl(required_paths["private"])
    if not blind_rows:
        raise ValueError("Pilot input is empty")
    if len(blind_rows) != len(private_rows):
        raise ValueError("Pilot/private-index record count mismatch")
    if manifest.get("pilot_records") != len(blind_rows):
        raise ValueError("Pilot manifest record count mismatch")
    for relative_path, rows in (
        ("pilot_blind.jsonl", blind_rows),
        ("pilot_private_index.jsonl", private_rows),
    ):
        recorded = inventory[relative_path].get("records")
        if recorded != len(rows):
            raise ValueError(
                f"Pilot inventory record count mismatch: {relative_path}"
            )

    normalized_rows: list[dict[str, Any]] = []
    private_by_id: dict[str, dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    for position, private_row in enumerate(private_rows, 1):
        _strict_keys(
            private_row,
            PRIVATE_INDEX_FIELDS,
            context=f"Private-index row {position}",
        )
        blind_id = private_row.get("blind_id")
        if (
            not isinstance(blind_id, str)
            or BLIND_ID_RE.fullmatch(blind_id) is None
        ):
            raise ValueError(
                f"Invalid private-index blind_id at row {position}"
            )
        if blind_id in private_by_id:
            raise ValueError(f"Duplicate private-index blind_id: {blind_id}")
        rank = private_row.get("pilot_rank")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank != position
        ):
            raise ValueError(
                f"Private-index pilot_rank must be contiguous: {blind_id}"
            )
        sample_id = private_row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Invalid sample_id for {blind_id}")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate private-index sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        text_hash = private_row.get("review_text_sha256")
        if (
            not isinstance(text_hash, str)
            or SHA256_RE.fullmatch(text_hash) is None
        ):
            raise ValueError(f"Invalid private review hash for {blind_id}")
        private_by_id[blind_id] = private_row

    seen_blind_ids: set[str] = set()
    for position, blind_row in enumerate(blind_rows, 1):
        _strict_keys(
            blind_row,
            {"schema_version", "blind_id", "reviewContent"},
            context=f"Blind-input row {position}",
        )
        if blind_row.get("schema_version") != BLIND_INPUT_SCHEMA_VERSION:
            raise ValueError(f"Blind schema mismatch at row {position}")
        blind_id = blind_row.get("blind_id")
        if (
            not isinstance(blind_id, str)
            or BLIND_ID_RE.fullmatch(blind_id) is None
        ):
            raise ValueError(f"Invalid blind_id at row {position}")
        if blind_id in seen_blind_ids:
            raise ValueError(f"Duplicate blind_id: {blind_id}")
        seen_blind_ids.add(blind_id)
        text = blind_row.get("reviewContent")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Empty review text: {blind_id}")
        text_hash = sha256_text(text)
        private_row = private_by_id.get(blind_id)
        if private_row is None:
            raise ValueError(f"Blind ID absent from private index: {blind_id}")
        if private_row["pilot_rank"] != position:
            raise ValueError(f"Pilot rank mismatch for {blind_id}")
        if private_row["review_text_sha256"] != text_hash:
            raise ValueError(f"Review text hash mismatch for {blind_id}")
        normalized_rows.append(
            {
                "annotation_id": blind_id,
                "pilot_rank": position,
                "review_text": text,
                "review_text_sha256": text_hash,
                "private": private_row,
            }
        )
    if seen_blind_ids != set(private_by_id):
        raise ValueError("Pilot/private-index ID bijection mismatch")

    prompt_paths = {
        "direct": package_dir / "system_prompt_direct.txt",
        "evidence_first": (
            package_dir / "system_prompt_evidence_first.txt"
        ),
    }
    prompts = {
        variant: path.read_text(encoding="utf-8")
        for variant, path in prompt_paths.items()
    }
    return {
        "package_dir": package_dir,
        "manifest": manifest,
        "manifest_sha256": _sha256_file(required_paths["manifest"]),
        "checksums_sha256": _sha256_file(required_paths["checksums"]),
        "blind_sha256": _sha256_file(required_paths["blind"]),
        "private_index_sha256": _sha256_file(required_paths["private"]),
        "guideline_sha256": guideline_sha,
        "llm_schema_sha256": llm_schema_sha,
        "rows": normalized_rows,
        "prompts": prompts,
        "prompt_sha256": {
            variant: sha256_text(text)
            for variant, text in prompts.items()
        },
    }


def _validate_usage(value: Any, *, context: str) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} usage must be an object")
    if not set(value).issubset(USAGE_FIELDS):
        raise ValueError(
            f"{context} usage contains unsupported fields: "
            f"{sorted(set(value) - USAGE_FIELDS)}"
        )
    output: dict[str, int | float] = {}
    for key, item in value.items():
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or item < 0
        ):
            raise ValueError(f"{context} usage value is invalid: {key}")
        if key.endswith("_tokens") or key.endswith("_count"):
            if not isinstance(item, int):
                raise ValueError(
                    f"{context} token/count usage must be an integer: {key}"
                )
        output[key] = item
    return output


def _add_usage(
    destination: Counter[str],
    usage: Mapping[str, int | float],
) -> None:
    for key, value in usage.items():
        destination[key] += value


def _attempt_user_message(
    row: Mapping[str, Any],
    validation_feedback: str | None,
) -> str:
    base = json.loads(
        build_user_message(
            blind_id=row["annotation_id"],
            review_text=row["review_text"],
            review_text_sha256=row["review_text_sha256"],
        )
    )
    if validation_feedback:
        base["pipeline_retry_instruction"] = (
            "Return a complete corrected annotation for exactly this review. "
            "Fix every validation error and repeat all nine aspects."
        )
        base["pipeline_validation_feedback"] = validation_feedback[:800]
    return canonical_json(base)


def _validate_attempt_ledger(
    *,
    run_dir: Path,
    row: Mapping[str, Any],
    summaries: Any,
    run_config: Mapping[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    if not isinstance(summaries, list) or not summaries:
        raise ValueError(
            f"Terminal artifact has no attempts: {row['annotation_id']}"
        )
    validation_feedback: str | None = None
    valid_attempt: dict[str, Any] | None = None
    validated_attempts: list[dict[str, Any]] = []
    usage_sums: Counter[str] = Counter()
    record_stem = Path(_record_name(row)).stem
    expected_summaries: list[dict[str, Any]] = []

    for expected_number, summary in enumerate(summaries, 1):
        if not isinstance(summary, dict):
            raise ValueError("Attempt summary must be an object")
        _strict_keys(
            summary,
            ATTEMPT_SUMMARY_FIELDS,
            context=f"Attempt summary for {row['annotation_id']}",
        )
        expected_relative = (
            f"attempts/{record_stem}-attempt-{expected_number}.json"
        )
        if summary.get("attempt") != expected_number:
            raise ValueError(
                f"Attempt numbers are not contiguous: {row['annotation_id']}"
            )
        if summary.get("attempt_path") != expected_relative:
            raise ValueError(
                f"Attempt path mismatch: {row['annotation_id']}"
            )
        attempt_path = _resolve_relative(run_dir, expected_relative)
        if not attempt_path.is_file():
            raise FileNotFoundError(attempt_path)
        attempt = _read_json(attempt_path)
        required_base = {
            "annotation_id",
            "attempt",
            "pass_id",
            "request_sha256",
            "review_text_sha256",
            "run_fingerprint",
            "started_at",
            "completed_at",
            "outcome",
        }
        if not required_base.issubset(attempt):
            raise ValueError(f"Incomplete attempt record: {attempt_path}")
        if attempt.get("annotation_id") != row["annotation_id"]:
            raise ValueError(f"Attempt annotation_id mismatch: {attempt_path}")
        if attempt.get("attempt") != expected_number:
            raise ValueError(f"Attempt number mismatch: {attempt_path}")
        if attempt.get("pass_id") != run_config["pass_id"]:
            raise ValueError(f"Attempt pass_id mismatch: {attempt_path}")
        if attempt.get("review_text_sha256") != row["review_text_sha256"]:
            raise ValueError(f"Attempt review hash mismatch: {attempt_path}")
        if attempt.get("run_fingerprint") != (
            run_config["run_fingerprint"]
        ):
            raise ValueError(f"Attempt run fingerprint mismatch: {attempt_path}")

        expected_user_message = _attempt_user_message(
            row,
            validation_feedback,
        )
        expected_request_sha = sha256_text(
            system_prompt + "\0" + expected_user_message
        )
        if attempt.get("request_sha256") != expected_request_sha:
            raise ValueError(f"Attempt request hash mismatch: {attempt_path}")
        outcome = attempt.get("outcome")
        if outcome not in {"VALID", "SCHEMA_INVALID", "BACKEND_ERROR"}:
            raise ValueError(f"Unsupported attempt outcome: {attempt_path}")
        if valid_attempt is not None:
            raise ValueError(f"Attempt exists after VALID: {attempt_path}")

        if outcome in {"VALID", "SCHEMA_INVALID"}:
            response_required = {
                "finish_reason",
                "provider_model",
                "provider_request_id",
                "raw_response",
                "raw_response_sha256",
                "usage",
            }
            if not response_required.issubset(attempt):
                raise ValueError(f"Incomplete provider response: {attempt_path}")
            expected_fields = required_base | response_required
            if outcome == "SCHEMA_INVALID":
                expected_fields.add("validation_error")
            _strict_keys(
                attempt,
                expected_fields,
                context=f"Attempt record {attempt_path}",
            )
            if attempt.get("provider_model") != (
                run_config["backend"]["model"]
            ):
                raise ValueError(f"Provider model mismatch: {attempt_path}")
            raw_response = attempt.get("raw_response")
            if not isinstance(raw_response, str):
                raise ValueError(f"Attempt raw response missing: {attempt_path}")
            if attempt.get("raw_response_sha256") != sha256_text(raw_response):
                raise ValueError(
                    f"Attempt raw-response checksum mismatch: {attempt_path}"
                )
            normalization_error: AnnotationValidationError | None = None
            normalized: dict[str, Any] | None = None
            try:
                normalized = validate_and_normalize_annotation(
                    row["review_text"],
                    parse_model_json(raw_response),
                )
            except AnnotationValidationError as exc:
                normalization_error = exc
            if outcome == "VALID":
                if normalization_error is not None or normalized is None:
                    raise ValueError(
                        f"Attempt marked VALID but normalization failed: "
                        f"{attempt_path}"
                    ) from normalization_error
                attempt["_normalized_annotation"] = normalized
                valid_attempt = attempt
            else:
                if normalization_error is None:
                    raise ValueError(
                        f"Attempt marked SCHEMA_INVALID but validates: "
                        f"{attempt_path}"
                    )
                if attempt.get("validation_error") != str(
                    normalization_error
                ):
                    raise ValueError(
                        f"Stored validation error mismatch: {attempt_path}"
                    )
                validation_feedback = str(normalization_error)
            usage = _validate_usage(
                attempt.get("usage"),
                context=f"Attempt {attempt_path}",
            )
            _add_usage(usage_sums, usage)
        else:
            backend_required = {
                "backend_error",
                "retry_after_seconds",
                "retryable",
                "status_code",
            }
            if not backend_required.issubset(attempt):
                raise ValueError(f"Incomplete backend failure: {attempt_path}")
            _strict_keys(
                attempt,
                required_base | backend_required,
                context=f"Attempt record {attempt_path}",
            )
            if not isinstance(attempt.get("retryable"), bool):
                raise ValueError(f"Invalid retryable flag: {attempt_path}")
            forbidden = {"raw_response", "raw_response_sha256", "usage"}
            if forbidden.intersection(attempt):
                raise ValueError(
                    f"Backend failure contains response payload: {attempt_path}"
                )

        expected_summary = {
            "attempt": expected_number,
            "attempt_path": expected_relative,
            "outcome": outcome,
            "request_sha256": expected_request_sha,
        }
        if canonical_json(summary) != canonical_json(expected_summary):
            raise ValueError(f"Attempt summary mismatch: {attempt_path}")
        expected_summaries.append(expected_summary)
        validated_attempts.append(attempt)

    return {
        "attempts": validated_attempts,
        "attempt_paths": {
            summary["attempt_path"] for summary in expected_summaries
        },
        "valid_attempt": valid_attempt,
        "usage_sums": dict(usage_sums),
        "outcome_counts": dict(
            sorted(Counter(item["outcome"] for item in validated_attempts).items())
        ),
    }


def _validate_final_record(
    *,
    path: Path,
    row: Mapping[str, Any],
    run_config: Mapping[str, Any],
    system_prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _read_json(path)
    _strict_keys(
        record,
        FINAL_RECORD_FIELDS,
        context=f"Final record {path}",
    )
    if record.get("artifact_type") != "LLM_PSEUDO_LABEL_RECORD":
        raise ValueError(f"Final record artifact_type mismatch: {path}")
    identity_checks = {
        "annotation_id": row["annotation_id"],
        "review_text_sha256": row["review_text_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
        "pass_id": run_config["pass_id"],
        "prompt_sha256": run_config["prompt_sha256"],
        "prompt_variant": run_config["prompt_variant"],
        "prompt_version": run_config["prompt_version"],
        "guideline_sha256": run_config["guideline_sha256"],
        "annotation_output_schema_sha256": run_config[
            "annotation_output_schema_sha256"
        ],
        "schema_version": run_config["schema_version"],
        "model": run_config["backend"]["model"],
    }
    for field, expected in identity_checks.items():
        if record.get(field) != expected:
            raise ValueError(f"Final record {field} mismatch: {path}")

    ledger = _validate_attempt_ledger(
        run_dir=path.parents[1],
        row=row,
        summaries=record.get("attempts"),
        run_config=run_config,
        system_prompt=system_prompt,
    )
    valid_attempt = ledger["valid_attempt"]
    if valid_attempt is None:
        raise ValueError(f"Final record has no VALID attempt: {path}")
    if record.get("attempt_count") != valid_attempt.get("attempt"):
        raise ValueError(f"Final attempt count mismatch: {path}")
    for field in (
        "raw_response",
        "raw_response_sha256",
        "request_sha256",
        "finish_reason",
        "provider_request_id",
    ):
        if record.get(field) != valid_attempt.get(field):
            raise ValueError(f"Final/attempt {field} mismatch: {path}")
    valid_usage = _validate_usage(
        valid_attempt.get("usage"),
        context=f"Valid attempt for {path}",
    )
    record_usage = _validate_usage(
        record.get("usage"),
        context=f"Final record {path}",
    )
    if canonical_json(valid_usage) != canonical_json(record_usage):
        raise ValueError(f"Final/attempt usage mismatch: {path}")

    raw_response = record.get("raw_response")
    if not isinstance(raw_response, str):
        raise ValueError(f"Missing final raw response: {path}")
    normalized = validate_and_normalize_annotation(
        row["review_text"],
        parse_model_json(raw_response),
    )
    if canonical_json(record.get("annotation")) != canonical_json(normalized):
        raise ValueError(f"Stored normalized annotation mismatch: {path}")
    expected_fingerprint = annotation_fingerprint(normalized)
    if record.get("annotation_fingerprint") != expected_fingerprint:
        raise ValueError(f"Annotation fingerprint mismatch: {path}")
    return record, ledger


def _validate_failure(
    *,
    path: Path,
    row: Mapping[str, Any],
    run_config: Mapping[str, Any],
    system_prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failure = _read_json(path)
    _strict_keys(
        failure,
        FAILURE_FIELDS,
        context=f"Failure record {path}",
    )
    if failure.get("artifact_type") != "LLM_PSEUDO_LABEL_FAILURE":
        raise ValueError(f"Failure artifact_type mismatch: {path}")
    for field, expected in {
        "annotation_id": row["annotation_id"],
        "review_text_sha256": row["review_text_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
        "pass_id": run_config["pass_id"],
    }.items():
        if failure.get(field) != expected:
            raise ValueError(f"Failure {field} mismatch: {path}")
    ledger = _validate_attempt_ledger(
        run_dir=path.parents[1],
        row=row,
        summaries=failure.get("attempts"),
        run_config=run_config,
        system_prompt=system_prompt,
    )
    if ledger["valid_attempt"] is not None:
        raise ValueError(f"Failure contains a VALID attempt: {path}")
    return failure, ledger


def _validate_run_file_layout(run_dir: Path) -> None:
    allowed_root = {"manifest.json", "run_config.json", "SHA256SUMS.txt"}
    allowed_directories = {
        "attempts": ".json",
        "records": ".json",
        "failed": ".json",
        "failure_history": ".json",
        "checksum_history": ".txt",
    }
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if len(relative.parts) == 1:
            if path.name not in allowed_root:
                raise ValueError(f"Unexpected run artifact: {path}")
            continue
        if len(relative.parts) != 2:
            raise ValueError(f"Unexpected nested run artifact: {path}")
        directory, name = relative.parts
        suffix = allowed_directories.get(directory)
        if suffix is None or not name.endswith(suffix):
            raise ValueError(f"Unexpected run artifact: {path}")


def _manifest_count_map(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(
        record["annotation"]["annotation_status"] for record in records
    )
    aspect_counts: dict[str, Counter[str]] = {
        aspect: Counter() for aspect in ASPECT_COLUMNS
    }
    for record in records:
        for aspect, label in zip(
            ASPECT_COLUMNS,
            label_vector(record["annotation"]),
            strict=True,
        ):
            aspect_counts[aspect][str(label)] += 1
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "aspect_label_counts": {
            aspect: dict(sorted(counter.items()))
            for aspect, counter in aspect_counts.items()
        },
    }


def _merge_counters(
    destination: Counter[str],
    values: Mapping[str, int | float],
) -> None:
    for key, value in values.items():
        destination[key] += value


def _validate_history_failures(
    *,
    run_dir: Path,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    active_ledgers: Mapping[str, list[dict[str, Any]]],
    run_config: Mapping[str, Any],
) -> None:
    history_dir = run_dir / "failure_history"
    if not history_dir.exists():
        return
    pattern = re.compile(
        r"\A(?P<rank>[0-9]{4})-(?P<id>llmp-[0-9a-f]{20})"
        r"-through-attempt-(?P<count>[1-9][0-9]*)\.json\Z"
    )
    for path in sorted(history_dir.glob("*.json")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid failure-history filename: {path}")
        annotation_id = match.group("id")
        row = rows_by_id.get(annotation_id)
        if row is None:
            raise ValueError(f"History failure is outside selection: {path}")
        if int(match.group("rank")) != row["pilot_rank"]:
            raise ValueError(f"History failure rank mismatch: {path}")
        value = _read_json(path)
        _strict_keys(
            value,
            FAILURE_FIELDS,
            context=f"Failure history {path}",
        )
        if value.get("artifact_type") != "LLM_PSEUDO_LABEL_FAILURE":
            raise ValueError(f"Failure history artifact_type mismatch: {path}")
        if value.get("annotation_id") != annotation_id:
            raise ValueError(f"Failure history annotation_id mismatch: {path}")
        if value.get("review_text_sha256") != row["review_text_sha256"]:
            raise ValueError(f"Failure history review hash mismatch: {path}")
        if value.get("pass_id") != run_config["pass_id"]:
            raise ValueError(f"Failure history pass mismatch: {path}")
        if value.get("run_fingerprint") != run_config["run_fingerprint"]:
            raise ValueError(f"Failure history run mismatch: {path}")
        attempts = value.get("attempts")
        count = int(match.group("count"))
        active = active_ledgers.get(annotation_id)
        if (
            not isinstance(attempts, list)
            or len(attempts) != count
            or active is None
            or canonical_json(attempts) != canonical_json(active[:count])
        ):
            raise ValueError(f"Failure history ledger mismatch: {path}")


def _validate_checksum_history(run_dir: Path) -> None:
    history_dir = run_dir / "checksum_history"
    if not history_dir.exists():
        return
    pattern = re.compile(r"\ASHA256SUMS-([0-9a-f]{16})\.txt\Z")
    for path in sorted(history_dir.glob("*.txt")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid checksum-history filename: {path}")
        if not _sha256_file(path).startswith(match.group(1)):
            raise ValueError(f"Checksum-history identity mismatch: {path}")
        _read_checksum_file(path, root=run_dir, verify_files=False)


def _validate_run(
    *,
    run_dir: Path,
    pilot: Mapping[str, Any],
    expected_pass_id: str,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    _validate_run_file_layout(run_dir)
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "run_config.json"
    checksum_path = run_dir / "SHA256SUMS.txt"
    for path in (manifest_path, config_path, checksum_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    checksum_entries = _read_checksum_file(
        checksum_path,
        root=run_dir,
        verify_files=True,
    )
    current_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path != checksum_path
        and not path.name.endswith(".tmp")
    }
    if set(checksum_entries) != current_files:
        raise ValueError(
            f"Run SHA256SUMS closure mismatch: {run_dir}; "
            f"missing={sorted(current_files - set(checksum_entries))}, "
            f"extra={sorted(set(checksum_entries) - current_files)}"
        )

    manifest = _read_json(manifest_path)
    run_config = _read_json(config_path)
    _strict_keys(
        manifest,
        RUN_MANIFEST_FIELDS,
        context=f"Run manifest {manifest_path}",
    )
    _strict_keys(
        run_config,
        RUN_CONFIG_FIELDS,
        context=f"Run config {config_path}",
    )
    if manifest.get("artifact_type") != "LLM_PSEUDO_LABEL_PASS":
        raise ValueError(f"Unexpected run artifact_type: {run_dir}")
    if canonical_json(manifest.get("run_config")) != canonical_json(
        run_config
    ):
        raise ValueError(f"Run manifest/config mismatch: {run_dir}")
    if run_config.get("pass_id") != expected_pass_id:
        raise ValueError(
            f"Expected {expected_pass_id}, got {run_config.get('pass_id')}"
        )
    if run_config.get("schema_version") != LLM_ANNOTATION_SCHEMA_VERSION:
        raise ValueError(f"Run annotation schema mismatch: {run_dir}")
    if run_config.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"Run prompt version mismatch: {run_dir}")

    fingerprint = run_config.get("run_fingerprint")
    if not isinstance(fingerprint, str) or SHA256_RE.fullmatch(
        fingerprint
    ) is None:
        raise ValueError(f"Invalid run fingerprint: {run_dir}")
    fingerprint_material = dict(run_config)
    fingerprint_material.pop("run_fingerprint")
    if sha256_text(canonical_json(fingerprint_material)) != fingerprint:
        raise ValueError(f"Run fingerprint cannot be replayed: {run_dir}")

    selected_count = run_config.get("selected_records")
    if (
        not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or not 0 < selected_count <= len(pilot["rows"])
    ):
        raise ValueError(f"Invalid selected-record count: {run_dir}")
    if manifest.get("selected_records") != selected_count:
        raise ValueError(f"Manifest selected-record count mismatch: {run_dir}")
    rows = pilot["rows"][:selected_count]
    rows_by_id = {row["annotation_id"]: row for row in rows}
    expected_ids = set(rows_by_id)

    pilot_provenance = run_config.get("pilot")
    if not isinstance(pilot_provenance, dict):
        raise ValueError(f"Run pilot provenance missing: {run_dir}")
    required_pilot_provenance = {
        "pilot_id": pilot["manifest"].get("pilot_id"),
        "pilot_manifest_sha256": pilot["manifest_sha256"],
        "pilot_private_index_sha256": pilot["private_index_sha256"],
        "source_release_id": pilot["manifest"].get("release_id"),
        "source_release_manifest_sha256": pilot["manifest"].get(
            "release_manifest_sha256"
        ),
    }
    for field, expected in required_pilot_provenance.items():
        if pilot_provenance.get(field) != expected:
            raise ValueError(
                f"Run pilot provenance {field} mismatch: {run_dir}"
            )
    if run_config.get("input_sha256") != pilot["blind_sha256"]:
        raise ValueError(f"Run input checksum mismatch: {run_dir}")
    if run_config.get("guideline_sha256") != pilot["guideline_sha256"]:
        raise ValueError(f"Run guideline checksum mismatch: {run_dir}")
    if run_config.get("annotation_output_schema_sha256") != (
        pilot["llm_schema_sha256"]
    ):
        raise ValueError(f"Run output-schema checksum mismatch: {run_dir}")
    backend = run_config.get("backend")
    if (
        not isinstance(backend, dict)
        or not isinstance(backend.get("model"), str)
        or not backend["model"]
    ):
        raise ValueError(f"Run backend/model is invalid: {run_dir}")
    _strict_keys(
        backend,
        {
            "api_key_env",
            "endpoint",
            "model",
            "request_json_mode",
            "type",
        },
        context=f"Run backend config {run_dir}",
    )
    for hash_field in (
        "annotation_output_schema_sha256",
        "config_sha256",
        "guideline_sha256",
        "input_sha256",
        "prompt_sha256",
    ):
        value = run_config.get(hash_field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"Invalid run hash field {hash_field}: {run_dir}")
    code_sha256 = run_config.get("code_sha256")
    if not isinstance(code_sha256, dict) or not code_sha256:
        raise ValueError(f"Run code checksum inventory missing: {run_dir}")
    for relative_path, checksum in code_sha256.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or not isinstance(checksum, str)
            or SHA256_RE.fullmatch(checksum) is None
        ):
            raise ValueError(f"Invalid run code checksum entry: {run_dir}")
    if not isinstance(run_config.get("generation"), dict):
        raise ValueError(f"Run generation config is invalid: {run_dir}")
    if (
        not isinstance(run_config.get("seed"), int)
        or isinstance(run_config.get("seed"), bool)
    ):
        raise ValueError(f"Run seed is invalid: {run_dir}")
    prompt_variant = run_config.get("prompt_variant")
    if prompt_variant not in pilot["prompts"]:
        raise ValueError(f"Unsupported run prompt variant: {run_dir}")
    if run_config.get("prompt_sha256") != (
        pilot["prompt_sha256"][prompt_variant]
    ):
        raise ValueError(f"Run prompt checksum mismatch: {run_dir}")
    system_prompt = pilot["prompts"][prompt_variant]

    records: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    active_attempt_paths: set[str] = set()
    active_ledgers: dict[str, list[dict[str, Any]]] = {}
    usage_sums: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    successful_usage: Counter[str] = Counter()

    actual_record_names = {
        path.name for path in (run_dir / "records").glob("*.json")
    }
    actual_failure_names = {
        path.name for path in (run_dir / "failed").glob("*.json")
    }
    expected_names = {_record_name(row) for row in rows}
    if not actual_record_names.issubset(expected_names):
        raise ValueError(f"Run contains out-of-selection records: {run_dir}")
    if not actual_failure_names.issubset(expected_names):
        raise ValueError(f"Run contains out-of-selection failures: {run_dir}")

    for row in rows:
        name = _record_name(row)
        record_path = run_dir / "records" / name
        failure_path = run_dir / "failed" / name
        if record_path.is_file() and failure_path.is_file():
            raise ValueError(
                f"ID exists in both records and failed: {row['annotation_id']}"
            )
        if record_path.is_file():
            record, ledger = _validate_final_record(
                path=record_path,
                row=row,
                run_config=run_config,
                system_prompt=system_prompt,
            )
            records[row["annotation_id"]] = record
            _add_usage(
                successful_usage,
                _validate_usage(
                    record["usage"],
                    context=f"Successful record {record_path}",
                ),
            )
        elif failure_path.is_file():
            failure, ledger = _validate_failure(
                path=failure_path,
                row=row,
                run_config=run_config,
                system_prompt=system_prompt,
            )
            failures[row["annotation_id"]] = failure
        else:
            continue
        active_attempt_paths.update(ledger["attempt_paths"])
        active_ledgers[row["annotation_id"]] = (
            record["attempts"]
            if record_path.is_file()
            else failure["attempts"]
        )
        _merge_counters(usage_sums, ledger["usage_sums"])
        _merge_counters(outcome_counts, ledger["outcome_counts"])

    attempts_dir = run_dir / "attempts"
    actual_attempt_paths = {
        path.relative_to(run_dir).as_posix()
        for path in attempts_dir.glob("*.json")
    }
    if actual_attempt_paths != active_attempt_paths:
        raise ValueError(
            f"Run attempt-ledger closure mismatch: {run_dir}; "
            f"unclaimed={sorted(actual_attempt_paths - active_attempt_paths)}, "
            f"missing={sorted(active_attempt_paths - actual_attempt_paths)}"
        )
    _validate_history_failures(
        run_dir=run_dir,
        rows_by_id=rows_by_id,
        active_ledgers=active_ledgers,
        run_config=run_config,
    )
    _validate_checksum_history(run_dir)

    completed_ids = set(records)
    failed_ids = set(failures)
    missing_ids = expected_ids - completed_ids - failed_ids
    if manifest.get("annotation_records") != len(records):
        raise ValueError(f"Manifest annotation count mismatch: {run_dir}")
    if manifest.get("failed_records") != len(failures):
        raise ValueError(f"Manifest failure count mismatch: {run_dir}")
    if manifest.get("missing_annotation_ids") != sorted(missing_ids):
        raise ValueError(f"Manifest missing-ID closure mismatch: {run_dir}")
    count_maps = _manifest_count_map(records.values())
    if canonical_json(manifest.get("status_counts")) != canonical_json(
        count_maps["status_counts"]
    ):
        raise ValueError(f"Manifest status counts mismatch: {run_dir}")
    if canonical_json(manifest.get("aspect_label_counts")) != canonical_json(
        count_maps["aspect_label_counts"]
    ):
        raise ValueError(f"Manifest aspect-label counts mismatch: {run_dir}")

    status = manifest.get("status")
    allowed_statuses = {
        "COMPLETED",
        "COMPLETED_WITH_FAILURES",
        "INCOMPLETE",
        "PAUSED_RATE_LIMIT",
        "ABORTED_BACKEND_FATAL",
    }
    if status not in allowed_statuses:
        raise ValueError(f"Unsupported run status {status!r}: {run_dir}")
    if status == "COMPLETED" and (
        failures or missing_ids or len(records) != selected_count
    ):
        raise ValueError(f"COMPLETED run is not closed: {run_dir}")
    if status == "COMPLETED_WITH_FAILURES" and (
        missing_ids or not failures
    ):
        raise ValueError(
            f"COMPLETED_WITH_FAILURES state is inconsistent: {run_dir}"
        )
    if status == "INCOMPLETE" and not missing_ids:
        raise ValueError(f"INCOMPLETE run has no missing IDs: {run_dir}")

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "manifest_sha256": _sha256_file(manifest_path),
        "checksums_sha256": _sha256_file(checksum_path),
        "run_config": run_config,
        "run_config_sha256": _sha256_file(config_path),
        "selected_rows": rows,
        "selected_ids": expected_ids,
        "records": records,
        "failures": failures,
        "missing_ids": missing_ids,
        "attempt_metrics": {
            "attempts": sum(outcome_counts.values()),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "all_response_usage": dict(sorted(usage_sums.items())),
            "successful_terminal_usage": dict(
                sorted(successful_usage.items())
            ),
        },
    }


def _validate_cross_pass_invariants(
    pass_a: Mapping[str, Any],
    pass_b: Mapping[str, Any],
) -> None:
    config_a = pass_a["run_config"]
    config_b = pass_b["run_config"]
    invariant_fields = {
        "annotation_output_schema_sha256",
        "backend",
        "code_sha256",
        "config_sha256",
        "generation",
        "guideline_sha256",
        "input_sha256",
        "pilot",
        "prompt_version",
        "schema_version",
        "selected_records",
    }
    for field in sorted(invariant_fields):
        if canonical_json(config_a.get(field)) != canonical_json(
            config_b.get(field)
        ):
            raise ValueError(f"Cross-pass invariant differs: {field}")
    if pass_a["selected_ids"] != pass_b["selected_ids"]:
        raise ValueError("Passes selected different annotation IDs")
    if config_a.get("backend", {}).get("model") != (
        config_b.get("backend", {}).get("model")
    ):
        raise ValueError("Passes used different configured models")
    if config_a.get("pass_id") == config_b.get("pass_id"):
        raise ValueError("Two-pass evaluation requires distinct pass IDs")
    if {config_a.get("prompt_variant"), config_b.get("prompt_variant")} != {
        "direct",
        "evidence_first",
    }:
        raise ValueError(
            "Two-pass evaluation requires direct and evidence_first prompts"
        )
    if config_a.get("seed") == config_b.get("seed"):
        raise ValueError("Two-pass evaluation requires distinct fixed seeds")


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": (
            round(numerator / denominator, 8)
            if denominator
            else None
        ),
    }


def _pass_state(
    run: Mapping[str, Any],
    annotation_id: str,
) -> dict[str, Any]:
    record = run["records"].get(annotation_id)
    if record is not None:
        return {
            "state": "RECORD",
            "annotation_status": record["annotation"]["annotation_status"],
            "labels": list(label_vector(record["annotation"])),
            "annotation_fingerprint": record["annotation_fingerprint"],
        }
    if annotation_id in run["failures"]:
        return {"state": "FAILURE"}
    if annotation_id in run["missing_ids"]:
        return {"state": "MISSING"}
    raise ValueError(f"Unclosed run state for {annotation_id}")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _sum_pass_usage(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    key: str,
) -> dict[str, int | float]:
    combined: Counter[str] = Counter()
    _merge_counters(combined, first["attempt_metrics"][key])
    _merge_counters(combined, second["attempt_metrics"][key])
    return dict(sorted(combined.items()))


def _build_evaluation_rows(
    *,
    pilot: Mapping[str, Any],
    pass_a: Mapping[str, Any],
    pass_b: Mapping[str, Any],
    evaluation_id: str,
) -> dict[str, Any]:
    selected_rows = pass_a["selected_rows"]
    per_aspect_equal: Counter[str] = Counter()
    per_aspect_total: Counter[str] = Counter()
    pair_label_equal = 0
    pair_label_total = 0
    paired_records = 0
    full_vector_equal = 0
    status_equal = 0
    exact_semantic_equal = 0
    both_labeled = 0
    consensus_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    queue_reason_counts: Counter[str] = Counter()
    consensus_label_counts: dict[str, Counter[str]] = {
        aspect: Counter() for aspect in ASPECT_COLUMNS
    }

    for row in selected_rows:
        annotation_id = row["annotation_id"]
        record_a = pass_a["records"].get(annotation_id)
        record_b = pass_b["records"].get(annotation_id)
        reasons: list[str] = []
        disagreement_aspects: list[str] = []
        comparison: dict[str, Any] | None = None
        consensus_exported = False

        if record_a is None:
            _append_reason(
                reasons,
                "PASS_A_FAILURE"
                if annotation_id in pass_a["failures"]
                else "PASS_A_MISSING",
            )
        if record_b is None:
            _append_reason(
                reasons,
                "PASS_B_FAILURE"
                if annotation_id in pass_b["failures"]
                else "PASS_B_MISSING",
            )

        if record_a is not None and record_b is not None:
            annotation_a = record_a["annotation"]
            annotation_b = record_b["annotation"]
            comparison = compare_annotation_passes(
                annotation_a,
                annotation_b,
            )
            paired_records += 1
            labels_a = list(label_vector(annotation_a))
            labels_b = list(label_vector(annotation_b))
            for aspect, left, right in zip(
                ASPECT_COLUMNS,
                labels_a,
                labels_b,
                strict=True,
            ):
                per_aspect_total[aspect] += 1
                pair_label_total += 1
                if left == right:
                    per_aspect_equal[aspect] += 1
                    pair_label_equal += 1
            if comparison["label_agreement"]:
                full_vector_equal += 1
            else:
                disagreement_aspects = comparison["aspect_disagreements"]
                _append_reason(reasons, "LABEL_DISAGREEMENT")
            if comparison["status_agreement"]:
                status_equal += 1
            else:
                _append_reason(reasons, "STATUS_DISAGREEMENT")
            if comparison["exact_semantic_agreement"]:
                exact_semantic_equal += 1
            else:
                _append_reason(
                    reasons,
                    "EVIDENCE_OR_UNCERTAINTY_SEMANTIC_DISAGREEMENT",
                )

            status_a = annotation_a["annotation_status"]
            status_b = annotation_b["annotation_status"]
            if status_a == "ESCALATE":
                _append_reason(reasons, "PASS_A_UNCERTAINTY")
            if status_b == "ESCALATE":
                _append_reason(reasons, "PASS_B_UNCERTAINTY")
            if status_a == "REJECT_NON_REVIEW":
                _append_reason(reasons, "PASS_A_NON_REVIEW")
            if status_b == "REJECT_NON_REVIEW":
                _append_reason(reasons, "PASS_B_NON_REVIEW")

            consensus_eligible = (
                status_a == "LABELED"
                and status_b == "LABELED"
                and comparison["label_agreement"]
                and len(labels_a) == len(ASPECT_COLUMNS)
            )
            if status_a == "LABELED" and status_b == "LABELED":
                both_labeled += 1
            if consensus_eligible:
                consensus_exported = True
                risk_items: list[dict[str, Any]] = []
                for aspect, label in zip(
                    ASPECT_COLUMNS,
                    labels_a,
                    strict=True,
                ):
                    consensus_label_counts[aspect][str(label)] += 1
                    if label == "1, -1":
                        _append_reason(reasons, "AGREED_MIXED")
                        risk_items.append(
                            {"aspect": aspect, "label": label}
                        )
                    elif label == -1:
                        _append_reason(reasons, "AGREED_NEGATIVE")
                        risk_items.append(
                            {"aspect": aspect, "label": label}
                        )
                    elif label == 0:
                        _append_reason(reasons, "AGREED_NEUTRAL")
                        risk_items.append(
                            {"aspect": aspect, "label": label}
                        )
                consensus_rows.append(
                    {
                        "annotation_id": annotation_id,
                        "annotation_status": "LABELED",
                        "artifact_type": (
                            "LLM_PSEUDO_LABEL_CONSENSUS_RECORD"
                        ),
                        "consensus_rule": (
                            "BOTH_PASSES_LABELED_AND_NINE_LABELS_AGREE"
                        ),
                        "evaluation_id": evaluation_id,
                        "exact_semantic_agreement": comparison[
                            "exact_semantic_agreement"
                        ],
                        "human_review_reasons": reasons,
                        "human_review_required": bool(reasons),
                        "label_class": "LLM_PSEUDO_LABEL_NOT_HUMAN_GOLD",
                        "labels": [
                            {"aspect": aspect, "label": label}
                            for aspect, label in zip(
                                ASPECT_COLUMNS,
                                labels_a,
                                strict=True,
                            )
                        ],
                        "pass_annotations": {
                            "pass_a": annotation_a,
                            "pass_b": annotation_b,
                        },
                        "pass_provenance": {
                            "pass_a": {
                                "annotation_fingerprint": record_a[
                                    "annotation_fingerprint"
                                ],
                                "run_fingerprint": pass_a["run_config"][
                                    "run_fingerprint"
                                ],
                            },
                            "pass_b": {
                                "annotation_fingerprint": record_b[
                                    "annotation_fingerprint"
                                ],
                                "run_fingerprint": pass_b["run_config"][
                                    "run_fingerprint"
                                ],
                            },
                        },
                        "pilot_rank": row["pilot_rank"],
                        "reviewContent": row["review_text"],
                        "review_text_sha256": row["review_text_sha256"],
                        "risk_labels_for_human_review": risk_items,
                        "sample_id": row["private"]["sample_id"],
                        "schema_version": EVALUATION_SCHEMA_VERSION,
                        "semantic_gate": SEMANTIC_GATE_STATUS,
                    }
                )

        if reasons:
            for reason in reasons:
                queue_reason_counts[reason] += 1
            queue_rows.append(
                {
                    "annotation_id": annotation_id,
                    "artifact_type": "LLM_PSEUDO_LABEL_HUMAN_REVIEW_ITEM",
                    "category": row["private"]["category"],
                    "collection_transport": row["private"][
                        "collection_transport"
                    ],
                    "curation_status": row["private"]["curation_status"],
                    "disagreement_aspects": disagreement_aspects,
                    "evaluation_id": evaluation_id,
                    "parent_canonical_row": row["private"][
                        "parent_canonical_row"
                    ],
                    "pass_a_annotation": (
                        record_a["annotation"]
                        if record_a is not None
                        else None
                    ),
                    "pass_a_state": _pass_state(pass_a, annotation_id),
                    "pass_b_annotation": (
                        record_b["annotation"]
                        if record_b is not None
                        else None
                    ),
                    "pass_b_state": _pass_state(pass_b, annotation_id),
                    "pilot_rank": row["pilot_rank"],
                    "queue_reasons": reasons,
                    "rating": row["private"]["rating"],
                    "reviewContent": row["review_text"],
                    "review_text_sha256": row["review_text_sha256"],
                    "sample_id": row["private"]["sample_id"],
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                }
            )

        decision_rows.append(
            {
                "annotation_id": annotation_id,
                "artifact_type": "LLM_PSEUDO_LABEL_EVALUATION_DECISION",
                "consensus_exported": consensus_exported,
                "disagreement_aspects": disagreement_aspects,
                "evaluation_id": evaluation_id,
                "exact_semantic_agreement": (
                    comparison["exact_semantic_agreement"]
                    if comparison is not None
                    else None
                ),
                "human_review_reasons": reasons,
                "human_review_required": bool(reasons),
                "label_agreement": (
                    comparison["label_agreement"]
                    if comparison is not None
                    else None
                ),
                "pass_a_state": _pass_state(pass_a, annotation_id),
                "pass_b_state": _pass_state(pass_b, annotation_id),
                "pilot_rank": row["pilot_rank"],
                "review_text_sha256": row["review_text_sha256"],
                "sample_id": row["private"]["sample_id"],
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status_agreement": (
                    comparison["status_agreement"]
                    if comparison is not None
                    else None
                ),
            }
        )

    selected_count = len(selected_rows)
    agreement = {
        "paired_terminal_records": _ratio(
            paired_records,
            selected_count,
        ),
        "overall_aspect_label_agreement_paired": _ratio(
            pair_label_equal,
            pair_label_total,
        ),
        "overall_aspect_label_agreement_selected_denominator": _ratio(
            pair_label_equal,
            selected_count * len(ASPECT_COLUMNS),
        ),
        "per_aspect_label_agreement_paired": {
            aspect: _ratio(
                per_aspect_equal[aspect],
                per_aspect_total[aspect],
            )
            for aspect in ASPECT_COLUMNS
        },
        "full_nine_label_vector_agreement_paired": _ratio(
            full_vector_equal,
            paired_records,
        ),
        "full_nine_label_vector_agreement_selected_denominator": _ratio(
            full_vector_equal,
            selected_count,
        ),
        "status_agreement_paired": _ratio(
            status_equal,
            paired_records,
        ),
        "exact_semantic_agreement_paired": _ratio(
            exact_semantic_equal,
            paired_records,
        ),
        "both_passes_labeled": _ratio(
            both_labeled,
            selected_count,
        ),
    }
    return {
        "agreement": agreement,
        "consensus_label_counts": {
            aspect: dict(sorted(counter.items()))
            for aspect, counter in consensus_label_counts.items()
        },
        "consensus_rows": consensus_rows,
        "decision_rows": decision_rows,
        "queue_reason_counts": dict(sorted(queue_reason_counts.items())),
        "queue_rows": queue_rows,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
            count += 1
    return count


def _write_legacy_csv(
    path: Path,
    consensus_rows: Iterable[Mapping[str, Any]],
) -> int:
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reviewContent", *ASPECT_COLUMNS],
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in consensus_rows:
            labels = {
                item["aspect"]: item["label"] for item in row["labels"]
            }
            writer.writerow(
                {
                    "reviewContent": row["reviewContent"],
                    **labels,
                }
            )
            count += 1
    return count


def _artifact(
    path: Path,
    root: Path,
    *,
    records: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _combined_attempt_metrics(
    pass_a: Mapping[str, Any],
    pass_b: Mapping[str, Any],
) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    _merge_counters(
        outcomes,
        pass_a["attempt_metrics"]["outcome_counts"],
    )
    _merge_counters(
        outcomes,
        pass_b["attempt_metrics"]["outcome_counts"],
    )
    return {
        "attempts": (
            pass_a["attempt_metrics"]["attempts"]
            + pass_b["attempt_metrics"]["attempts"]
        ),
        "outcome_counts": dict(sorted(outcomes.items())),
        "all_response_usage": _sum_pass_usage(
            pass_a,
            pass_b,
            "all_response_usage",
        ),
        "successful_terminal_usage": _sum_pass_usage(
            pass_a,
            pass_b,
            "successful_terminal_usage",
        ),
    }


def evaluate(
    *,
    pilot_package: Path,
    pass_a_run: Path,
    pass_b_run: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    resolved_inputs = {
        pilot_package.resolve(),
        pass_a_run.resolve(),
        pass_b_run.resolve(),
    }
    if len(resolved_inputs) != 3:
        raise ValueError("Pilot and pass directories must be distinct")
    for input_path in resolved_inputs:
        try:
            output.relative_to(input_path)
        except ValueError:
            pass
        else:
            if input_path in {
                pass_a_run.resolve(),
                pass_b_run.resolve(),
            }:
                raise ValueError("Output must not be inside a run directory")

    pilot = _validate_pilot_package(pilot_package)
    pass_a = _validate_run(
        run_dir=pass_a_run,
        pilot=pilot,
        expected_pass_id="pass_a",
    )
    pass_b = _validate_run(
        run_dir=pass_b_run,
        pilot=pilot,
        expected_pass_id="pass_b",
    )
    _validate_cross_pass_invariants(pass_a, pass_b)

    identity_material = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluator_code_sha256": _sha256_file(Path(__file__).resolve()),
        "pass_a_manifest_sha256": pass_a["manifest_sha256"],
        "pass_a_run_config_sha256": pass_a["run_config_sha256"],
        "pass_b_manifest_sha256": pass_b["manifest_sha256"],
        "pass_b_run_config_sha256": pass_b["run_config_sha256"],
        "pilot_manifest_sha256": pilot["manifest_sha256"],
        "selected_annotation_ids": [
            row["annotation_id"] for row in pass_a["selected_rows"]
        ],
    }
    evaluation_id = (
        "llm-pilot-evaluation-"
        + sha256_text(canonical_json(identity_material))[:20]
    )
    built = _build_evaluation_rows(
        pilot=pilot,
        pass_a=pass_a,
        pass_b=pass_b,
        evaluation_id=evaluation_id,
    )

    semantic_gate = {
        "reason": (
            "No independently human-annotated and adjudicated gold labels "
            "were supplied; two-pass agreement measures repeat consistency "
            "only and must not be reported as accuracy."
        ),
        "status": SEMANTIC_GATE_STATUS,
    }
    coverage = {
        "selected_records": len(pass_a["selected_rows"]),
        "pass_a": {
            "records": len(pass_a["records"]),
            "failures": len(pass_a["failures"]),
            "missing": len(pass_a["missing_ids"]),
            "status": pass_a["manifest"]["status"],
        },
        "pass_b": {
            "records": len(pass_b["records"]),
            "failures": len(pass_b["failures"]),
            "missing": len(pass_b["missing_ids"]),
            "status": pass_b["manifest"]["status"],
        },
        "consensus_pseudo_labels": len(built["consensus_rows"]),
        "human_review_queue": len(built["queue_rows"]),
    }
    metrics = {
        "agreement": built["agreement"],
        "artifact_type": "LLM_PSEUDO_LABEL_TECHNICAL_METRICS",
        "consensus_label_counts": built["consensus_label_counts"],
        "coverage": coverage,
        "evaluation_id": evaluation_id,
        "interpretation": {
            "agreement_type": "SAME_MODEL_TWO_PASS_REPEAT_CONSISTENCY",
            "accuracy_claim_permitted": False,
            "independent_annotators": False,
            "technical_validation": "PASS",
        },
        "queue_reason_counts": built["queue_reason_counts"],
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "semantic_gate": semantic_gate,
        "tokens_and_provider_usage": {
            "pass_a": pass_a["attempt_metrics"],
            "pass_b": pass_b["attempt_metrics"],
            "combined": _combined_attempt_metrics(pass_a, pass_b),
            "scope": (
                "Usage from every checksum-verified provider response "
                "attempt, including schema-invalid responses."
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.building-",
        dir=output.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        consensus_path = temporary / "consensus_pseudo_labels.jsonl"
        legacy_path = temporary / "consensus_legacy.csv"
        queue_path = temporary / "human_review_queue.jsonl"
        ledger_path = temporary / "evaluation_decision_ledger.jsonl"
        metrics_path = temporary / "technical_metrics.json"

        consensus_count = _write_jsonl(
            consensus_path,
            built["consensus_rows"],
        )
        legacy_count = _write_legacy_csv(
            legacy_path,
            built["consensus_rows"],
        )
        queue_count = _write_jsonl(queue_path, built["queue_rows"])
        decision_count = _write_jsonl(
            ledger_path,
            built["decision_rows"],
        )
        _write_json(metrics_path, metrics)
        if consensus_count != legacy_count:
            raise ValueError("Consensus JSONL/legacy CSV count mismatch")
        if decision_count != len(pass_a["selected_rows"]):
            raise ValueError("Evaluation decision ledger is not closed")

        artifacts = [
            _artifact(
                consensus_path,
                temporary,
                records=consensus_count,
            ),
            _artifact(legacy_path, temporary, records=legacy_count),
            _artifact(queue_path, temporary, records=queue_count),
            _artifact(ledger_path, temporary, records=decision_count),
            _artifact(metrics_path, temporary),
        ]
        manifest = {
            "artifact_type": "LLM_PSEUDO_LABEL_PILOT_EVALUATION",
            "artifacts": artifacts,
            "built_at": _utc_now(),
            "counts": {
                "consensus_pseudo_labels": consensus_count,
                "decision_ledger": decision_count,
                "human_review_queue": queue_count,
                "legacy_consensus_rows": legacy_count,
            },
            "evaluation_id": evaluation_id,
            "identity_material": identity_material,
            "legacy_csv_notice": (
                "The CSV is a compatibility view and omits evidence and "
                "provenance; consensus_pseudo_labels.jsonl is authoritative."
            ),
            "model": pass_a["run_config"]["backend"]["model"],
            "pseudo_label_policy": (
                "Export only when both passes have status LABELED and all "
                "nine canonical labels agree. These remain LLM pseudo-labels."
            ),
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "semantic_gate": semantic_gate,
            "source": {
                "pass_a": {
                    "manifest_sha256": pass_a["manifest_sha256"],
                    "run_config_sha256": pass_a["run_config_sha256"],
                    "run_fingerprint": pass_a["run_config"][
                        "run_fingerprint"
                    ],
                    "sha256sums_sha256": pass_a["checksums_sha256"],
                },
                "pass_b": {
                    "manifest_sha256": pass_b["manifest_sha256"],
                    "run_config_sha256": pass_b["run_config_sha256"],
                    "run_fingerprint": pass_b["run_config"][
                        "run_fingerprint"
                    ],
                    "sha256sums_sha256": pass_b["checksums_sha256"],
                },
                "pilot": {
                    "pilot_id": pilot["manifest"].get("pilot_id"),
                    "manifest_sha256": pilot["manifest_sha256"],
                    "private_index_sha256": pilot[
                        "private_index_sha256"
                    ],
                    "sha256sums_sha256": pilot["checksums_sha256"],
                },
            },
            "status": "TECHNICAL_VALIDATION_PASS",
            "validation_scope": "TECHNICAL_ONLY",
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        checksum_paths = [
            *(temporary / artifact["path"] for artifact in artifacts),
            manifest_path,
        ]
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{_sha256_file(path)}  "
                f"{path.relative_to(temporary).as_posix()}"
                for path in sorted(
                    checksum_paths,
                    key=lambda item: item.relative_to(
                        temporary
                    ).as_posix(),
                )
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)

    return {
        "consensus_pseudo_labels": len(built["consensus_rows"]),
        "evaluation_id": evaluation_id,
        "human_review_queue": len(built["queue_rows"]),
        "manifest": str(output / "manifest.json"),
        "output": str(output),
        "selected_records": len(pass_a["selected_rows"]),
        "semantic_gate": SEMANTIC_GATE_STATUS,
        "status": "TECHNICAL_VALIDATION_PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed technical QA and agreement evaluation for two "
            "checkpointed LLM ABSA pilot passes."
        )
    )
    parser.add_argument(
        "--pilot-package",
        type=Path,
        required=True,
        help="Frozen pilot package containing blind input/private index.",
    )
    parser.add_argument(
        "--pass-a-run",
        type=Path,
        required=True,
        help="Completed or closed pass_a run directory.",
    )
    parser.add_argument(
        "--pass-b-run",
        type=Path,
        required=True,
        help="Completed or closed pass_b run directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New versioned output directory; it must not already exist.",
    )
    args = parser.parse_args()
    result = evaluate(
        pilot_package=args.pilot_package,
        pass_a_run=args.pass_a_run,
        pass_b_run=args.pass_b_run,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
