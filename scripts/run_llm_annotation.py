"""Run one resumable, fail-closed LLM pseudo-labeling pass."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from lazada_collector.llm_annotation import (
    ASPECT_COLUMNS,
    LLM_ANNOTATION_SCHEMA_VERSION,
    PROMPT_VERSION,
    AnnotationValidationError,
    annotation_fingerprint,
    build_system_prompt,
    build_user_message,
    canonical_json,
    label_vector,
    parse_model_json,
    sha256_text,
    validate_and_normalize_annotation,
    validate_output_schema_contract,
)
from lazada_collector.llm_backends import (
    GenerationRequest,
    LLMBackendError,
    generate_json,
    sanitized_endpoint,
)


DEFAULT_CONFIG = Path("configs/llm_annotation_v1.json")
BLIND_INPUT_SCHEMA_VERSION = "absa-llm-blind-input/1.0.0"
BLIND_ID_RE = re.compile(r"\Allmp-[0-9a-f]{20}\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = parse_model_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = parse_model_json(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _resolve_artifact_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Artifact path must be a non-empty relative path")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Artifact path escapes pilot directory: {relative_path}"
        ) from exc
    return candidate


def _validate_pilot_package(
    *,
    input_path: Path,
    rows: list[dict[str, Any]],
    guideline_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    package_dir = input_path.parent.resolve()
    manifest_path = package_dir / "manifest.json"
    checksum_path = package_dir / "SHA256SUMS.txt"
    private_index_path = package_dir / "pilot_private_index.jsonl"
    for required in (manifest_path, checksum_path, private_index_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "LLM_PSEUDO_LABEL_PILOT_INPUT":
        raise ValueError("Unexpected pilot manifest artifact_type")
    if manifest.get("status") != "PREPARED_NOT_LABELED":
        raise ValueError("Pilot package is not in prepared state")
    if manifest.get("blind_input_schema_version") != (
        BLIND_INPUT_SCHEMA_VERSION
    ):
        raise ValueError("Pilot manifest blind schema version mismatch")
    if manifest.get("annotation_output_schema_version") != (
        LLM_ANNOTATION_SCHEMA_VERSION
    ):
        raise ValueError("Pilot manifest annotation schema version mismatch")
    if manifest.get("model_visible_fields") != [
        "schema_version",
        "blind_id",
        "reviewContent",
    ]:
        raise ValueError("Pilot manifest model-visible allowlist mismatch")
    if manifest.get("pilot_records") != len(rows):
        raise ValueError("Pilot manifest record count mismatch")
    if manifest.get("guideline_sha256") != _sha256_file(guideline_path):
        raise ValueError("Pilot guideline checksum mismatch")
    if manifest.get("llm_output_schema_sha256") != _sha256_file(schema_path):
        raise ValueError("Pilot output-schema checksum mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Pilot manifest has no artifact inventory")
    inventory_paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Pilot manifest artifact entry must be an object")
        relative_path = artifact.get("path")
        artifact_path = _resolve_artifact_path(package_dir, relative_path)
        if relative_path in inventory_paths:
            raise ValueError(f"Duplicate pilot artifact: {relative_path}")
        inventory_paths.add(relative_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if artifact.get("bytes") != artifact_path.stat().st_size:
            raise ValueError(f"Pilot artifact byte count mismatch: {relative_path}")
        if artifact.get("sha256") != _sha256_file(artifact_path):
            raise ValueError(f"Pilot artifact checksum mismatch: {relative_path}")

    required_inventory = {
        input_path.relative_to(package_dir).as_posix(),
        private_index_path.relative_to(package_dir).as_posix(),
    }
    if not required_inventory.issubset(inventory_paths):
        raise ValueError("Pilot manifest omits blind input or private index")

    checksum_entries: dict[str, str] = {}
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError(
                    f"Invalid pilot checksum line {line_number}"
                )
            checksum, relative_path = parts
            if relative_path in checksum_entries:
                raise ValueError(
                    f"Duplicate pilot checksum path: {relative_path}"
                )
            artifact_path = _resolve_artifact_path(
                package_dir,
                relative_path,
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            if checksum != _sha256_file(artifact_path):
                raise ValueError(
                    f"Pilot SHA256SUMS mismatch: {relative_path}"
                )
            checksum_entries[relative_path] = checksum
    expected_checksum_paths = inventory_paths | {"manifest.json"}
    if set(checksum_entries) != expected_checksum_paths:
        raise ValueError("Pilot SHA256SUMS closure mismatch")

    private_rows = _read_jsonl(private_index_path)
    if len(private_rows) != len(rows):
        raise ValueError("Pilot private-index count mismatch")
    private_by_id: dict[str, dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    expected_private_fields = {
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
    for private_row in private_rows:
        if set(private_row) != expected_private_fields:
            raise ValueError("Pilot private-index field schema mismatch")
        blind_id = private_row.get("blind_id")
        if not isinstance(blind_id, str) or not blind_id:
            raise ValueError("Pilot private index has invalid blind_id")
        if blind_id in private_by_id:
            raise ValueError(f"Duplicate private-index blind_id: {blind_id}")
        private_by_id[blind_id] = private_row
        sample_id = private_row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Pilot private index has invalid sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate private-index sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
    for row in rows:
        private_row = private_by_id.get(row["annotation_id"])
        if private_row is None:
            raise ValueError(
                f"Blind ID missing from private index: {row['annotation_id']}"
            )
        if private_row.get("pilot_rank") != row["pilot_rank"]:
            raise ValueError(
                f"Pilot rank mismatch: {row['annotation_id']}"
            )
        if private_row.get("review_text_sha256") != row["review_text_sha256"]:
            raise ValueError(
                f"Review hash mismatch in private index: "
                f"{row['annotation_id']}"
            )

    return {
        "pilot_id": manifest.get("pilot_id"),
        "pilot_manifest_sha256": _sha256_file(manifest_path),
        "pilot_private_index_sha256": _sha256_file(private_index_path),
        "source_release_id": manifest.get("release_id"),
        "source_release_manifest_sha256": manifest.get(
            "release_manifest_sha256"
        ),
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _record_name(row: dict[str, Any]) -> str:
    rank = row.get("pilot_rank")
    annotation_id = row.get("annotation_id")
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank <= 0
        or not isinstance(annotation_id, str)
        or not annotation_id
    ):
        raise ValueError("Pilot input requires positive pilot_rank/annotation_id")
    return f"{rank:04d}-{annotation_id}.json"


def _normalize_blind_input(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allowed = {"schema_version", "blind_id", "reviewContent"}
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for pilot_rank, row in enumerate(rows, 1):
        extra = set(row) - allowed
        missing = allowed - set(row)
        if extra or missing:
            raise ValueError(
                "Blind input fields mismatch; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if row.get("schema_version") != BLIND_INPUT_SCHEMA_VERSION:
            raise ValueError("Blind input schema version mismatch")
        annotation_id = row.get("blind_id")
        text = row.get("reviewContent")
        if (
            not isinstance(annotation_id, str)
            or BLIND_ID_RE.fullmatch(annotation_id) is None
        ):
            raise ValueError("Invalid blind_id")
        if annotation_id in seen_ids:
            raise ValueError(f"Duplicate blind_id: {annotation_id}")
        seen_ids.add(annotation_id)
        if not isinstance(text, str) or not text:
            raise ValueError(f"Empty reviewContent for {annotation_id}")
        normalized.append(
            {
                "annotation_id": annotation_id,
                "pilot_rank": pilot_rank,
                "review_text": text,
                "review_text_sha256": sha256_text(text),
            }
        )
    return normalized


def _pass_config(config: dict[str, Any], pass_id: str) -> dict[str, Any]:
    passes = config.get("passes")
    if not isinstance(passes, list):
        raise ValueError("Config passes must be a list")
    matches = [
        value
        for value in passes
        if isinstance(value, dict) and value.get("id") == pass_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one config pass: {pass_id}")
    return matches[0]


def _retry_delay(attempt: int, error: LLMBackendError | None = None) -> float:
    if error and error.retry_after_seconds is not None:
        return error.retry_after_seconds
    return min(30.0, float(2 ** max(0, attempt - 1)))


def _attempt_user_message(
    *,
    row: dict[str, Any],
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


def _scan_attempt_ledger(
    *,
    output: Path,
    row: dict[str, Any],
    pass_id: str,
    run_fingerprint: str,
    system_prompt: str,
) -> tuple[
    list[dict[str, Any]],
    int,
    str | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Replay and validate append-only attempts after a crash or resume."""

    record_stem = Path(_record_name(row)).stem
    attempt_paths = list(
        (output / "attempts").glob(f"{record_stem}-attempt-*.json")
    )
    numbered_paths: dict[int, Path] = {}
    pattern = re.compile(
        rf"\A{re.escape(record_stem)}-attempt-([1-9][0-9]*)\.json\Z"
    )
    for path in attempt_paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid attempt filename: {path}")
        attempt_number = int(match.group(1))
        if attempt_number in numbered_paths:
            raise ValueError(
                f"Duplicate attempt number {attempt_number}: {path}"
            )
        numbered_paths[attempt_number] = path
    expected_numbers = list(range(1, len(numbered_paths) + 1))
    if sorted(numbered_paths) != expected_numbers:
        raise ValueError(
            f"Attempt ledger is not contiguous for {row['annotation_id']}"
        )

    summaries: list[dict[str, Any]] = []
    validation_feedback: str | None = None
    valid_attempt: dict[str, Any] | None = None
    last_attempt_record: dict[str, Any] | None = None
    for attempt_number in expected_numbers:
        attempt_path = numbered_paths[attempt_number]
        attempt_record = _read_json(attempt_path)
        if attempt_record.get("attempt") != attempt_number:
            raise ValueError(f"Attempt number mismatch in {attempt_path}")
        if attempt_record.get("annotation_id") != row["annotation_id"]:
            raise ValueError(f"annotation_id mismatch in {attempt_path}")
        if attempt_record.get("review_text_sha256") != (
            row["review_text_sha256"]
        ):
            raise ValueError(f"review hash mismatch in {attempt_path}")
        if attempt_record.get("pass_id") != pass_id:
            raise ValueError(f"pass_id mismatch in {attempt_path}")
        if attempt_record.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"run fingerprint mismatch in {attempt_path}")
        expected_user_message = _attempt_user_message(
            row=row,
            validation_feedback=validation_feedback,
        )
        expected_request_hash = sha256_text(
            system_prompt + "\0" + expected_user_message
        )
        if attempt_record.get("request_sha256") != expected_request_hash:
            raise ValueError(f"request hash mismatch in {attempt_path}")

        outcome = attempt_record.get("outcome")
        if outcome not in {"VALID", "SCHEMA_INVALID", "BACKEND_ERROR"}:
            raise ValueError(f"Invalid attempt outcome in {attempt_path}")
        if valid_attempt is not None:
            raise ValueError(
                f"Attempt exists after a valid response: {attempt_path}"
            )
        if outcome in {"VALID", "SCHEMA_INVALID"}:
            raw_response = attempt_record.get("raw_response")
            if not isinstance(raw_response, str):
                raise ValueError(f"Missing raw response in {attempt_path}")
            if attempt_record.get("raw_response_sha256") != sha256_text(
                raw_response
            ):
                raise ValueError(
                    f"Raw-response checksum mismatch in {attempt_path}"
                )
            try:
                normalized = validate_and_normalize_annotation(
                    row["review_text"],
                    parse_model_json(raw_response),
                )
            except AnnotationValidationError as exc:
                if outcome != "SCHEMA_INVALID":
                    raise ValueError(
                        f"Attempt marked VALID but is invalid: {attempt_path}"
                    ) from exc
                if attempt_record.get("validation_error") != str(exc):
                    raise ValueError(
                        f"Validation error mismatch in {attempt_path}"
                    )
                validation_feedback = str(exc)
            else:
                if outcome != "VALID":
                    raise ValueError(
                        f"Attempt marked invalid but validates: {attempt_path}"
                    )
                attempt_record["_normalized_annotation"] = normalized
                valid_attempt = attempt_record
        elif not isinstance(attempt_record.get("retryable"), bool):
            raise ValueError(
                f"Backend attempt has invalid retryable flag: {attempt_path}"
            )
        summaries.append(
            {
                "attempt": attempt_number,
                "attempt_path": attempt_path.relative_to(output).as_posix(),
                "outcome": outcome,
                "request_sha256": expected_request_hash,
            }
        )
        last_attempt_record = attempt_record
    return (
        summaries,
        len(expected_numbers),
        validation_feedback,
        valid_attempt,
        last_attempt_record,
    )


def _validate_prior_failure(
    *,
    failure_path: Path,
    row: dict[str, Any],
    run_fingerprint: str,
    attempt_summaries: list[dict[str, Any]],
) -> int:
    failure = _read_json(failure_path)
    if failure.get("annotation_id") != row["annotation_id"]:
        raise ValueError(f"annotation_id mismatch in {failure_path}")
    if failure.get("review_text_sha256") != row["review_text_sha256"]:
        raise ValueError(f"review hash mismatch in {failure_path}")
    if failure.get("run_fingerprint") != run_fingerprint:
        raise ValueError(f"Run fingerprint mismatch in {failure_path}")
    stored_attempts = failure.get("attempts")
    if not isinstance(stored_attempts, list):
        raise ValueError(f"Invalid attempt ledger in {failure_path}")
    if len(stored_attempts) > len(attempt_summaries):
        raise ValueError(f"Attempt ledger is truncated in {failure_path}")
    if canonical_json(stored_attempts) != canonical_json(
        attempt_summaries[: len(stored_attempts)]
    ):
        raise ValueError(f"Attempt ledger mismatch in {failure_path}")
    return len(stored_attempts)


def _archive_failure(
    *,
    output: Path,
    failure_path: Path,
    last_attempt: int,
) -> None:
    if not failure_path.is_file():
        return
    history_dir = output / "failure_history"
    history_dir.mkdir(exist_ok=True)
    history_path = (
        history_dir
        / f"{failure_path.stem}-through-attempt-{last_attempt}.json"
    )
    if history_path.exists():
        raise FileExistsError(f"Failure history already exists: {history_path}")
    failure_path.replace(history_path)


def _validate_existing_record(
    path: Path,
    *,
    row: dict[str, Any],
    run_fingerprint: str,
    pass_id: str,
    prompt_sha256: str,
    schema_sha256: str,
    attempt_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    record = _read_json(path)
    if record.get("run_fingerprint") != run_fingerprint:
        raise ValueError(f"Run fingerprint mismatch in {path}")
    if record.get("annotation_id") != row["annotation_id"]:
        raise ValueError(f"annotation_id mismatch in {path}")
    if record.get("review_text_sha256") != row["review_text_sha256"]:
        raise ValueError(f"review hash mismatch in {path}")
    if record.get("pass_id") != pass_id:
        raise ValueError(f"pass_id mismatch in {path}")
    if record.get("prompt_sha256") != prompt_sha256:
        raise ValueError(f"prompt checksum mismatch in {path}")
    if record.get("schema_version") != LLM_ANNOTATION_SCHEMA_VERSION:
        raise ValueError(f"schema version mismatch in {path}")
    if record.get("annotation_output_schema_sha256") != schema_sha256:
        raise ValueError(f"schema checksum mismatch in {path}")
    if canonical_json(record.get("attempts")) != canonical_json(
        attempt_summaries
    ):
        raise ValueError(f"attempt ledger mismatch in {path}")
    raw_response = record.get("raw_response")
    if not isinstance(raw_response, str):
        raise ValueError(f"Missing raw_response in {path}")
    if record.get("raw_response_sha256") != sha256_text(raw_response):
        raise ValueError(f"raw response checksum mismatch in {path}")
    normalized = validate_and_normalize_annotation(
        row["review_text"],
        parse_model_json(raw_response),
    )
    if canonical_json(normalized) != canonical_json(record.get("annotation")):
        raise ValueError(f"Canonical annotation mismatch in {path}")
    if record.get("annotation_fingerprint") != annotation_fingerprint(
        normalized
    ):
        raise ValueError(f"Annotation fingerprint mismatch in {path}")
    return record


def _build_final_record(
    *,
    row: dict[str, Any],
    attempt_record: dict[str, Any],
    normalized: dict[str, Any],
    attempt_summaries: list[dict[str, Any]],
    run_config: dict[str, Any],
    run_fingerprint: str,
    pass_id: str,
    prompt_variant: str,
) -> dict[str, Any]:
    raw_response = attempt_record.get("raw_response")
    if not isinstance(raw_response, str):
        raise ValueError("Valid attempt is missing raw_response")
    return {
        "annotation": normalized,
        "annotation_fingerprint": annotation_fingerprint(normalized),
        "annotation_id": row["annotation_id"],
        "annotation_output_schema_sha256": run_config[
            "annotation_output_schema_sha256"
        ],
        "artifact_type": "LLM_PSEUDO_LABEL_RECORD",
        "attempt_count": attempt_record["attempt"],
        "attempts": attempt_summaries,
        "completed_at": attempt_record["completed_at"],
        "finish_reason": attempt_record.get("finish_reason"),
        "guideline_sha256": run_config["guideline_sha256"],
        "model": attempt_record["provider_model"],
        "pass_id": pass_id,
        "prompt_sha256": run_config["prompt_sha256"],
        "prompt_variant": prompt_variant,
        "prompt_version": PROMPT_VERSION,
        "provider_request_id": attempt_record.get("provider_request_id"),
        "raw_response": raw_response,
        "raw_response_sha256": sha256_text(raw_response),
        "request_sha256": attempt_record["request_sha256"],
        "review_text_sha256": row["review_text_sha256"],
        "run_fingerprint": run_fingerprint,
        "schema_version": LLM_ANNOTATION_SCHEMA_VERSION,
        "usage": attempt_record.get("usage", {}),
    }


def _write_checksums(run_dir: Path) -> None:
    checksum_path = run_dir / "SHA256SUMS.txt"
    files = sorted(
        (
            path
            for path in run_dir.rglob("*")
            if path.is_file()
            and path != checksum_path
            and not path.name.endswith(".tmp")
        ),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    )
    checksum_path.write_text(
        "\n".join(
            f"{_sha256_file(path)}  {path.relative_to(run_dir).as_posix()}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _verify_run_checksums(run_dir: Path) -> None:
    checksum_path = run_dir / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        return
    recorded: dict[str, str] = {}
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError(f"Invalid run checksum line {line_number}")
            checksum, relative_path = parts
            if relative_path in recorded:
                raise ValueError(f"Duplicate run checksum: {relative_path}")
            artifact_path = _resolve_artifact_path(run_dir, relative_path)
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            if checksum != _sha256_file(artifact_path):
                raise ValueError(f"Run checksum mismatch: {relative_path}")
            recorded[relative_path] = checksum
    current_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path != checksum_path
        and not path.name.endswith(".tmp")
    }
    if set(recorded) != current_files:
        raise ValueError("Run SHA256SUMS closure mismatch")


def _archive_run_checksums_for_retry(run_dir: Path) -> None:
    checksum_path = run_dir / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        return
    checksum_sha = _sha256_file(checksum_path)
    history_dir = run_dir / "checksum_history"
    history_dir.mkdir(exist_ok=True)
    history_path = history_dir / f"SHA256SUMS-{checksum_sha[:16]}.txt"
    if history_path.exists():
        raise FileExistsError(f"Checksum history already exists: {history_path}")
    checksum_path.replace(history_path)


def _summarize(
    *,
    run_dir: Path,
    selected_rows: list[dict[str, Any]],
    run_config: dict[str, Any],
    stop_status: str | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in selected_rows:
        name = _record_name(row)
        record_path = run_dir / "records" / name
        failure_path = run_dir / "failed" / name
        if record_path.is_file():
            records.append(_read_json(record_path))
        elif failure_path.is_file():
            failures.append(_read_json(failure_path))

    status_counts = Counter(
        record["annotation"]["annotation_status"] for record in records
    )
    aspect_label_counts: dict[str, Counter] = {
        aspect: Counter() for aspect in ASPECT_COLUMNS
    }
    for record in records:
        for aspect, label in zip(
            ASPECT_COLUMNS,
            label_vector(record["annotation"]),
            strict=True,
        ):
            aspect_label_counts[aspect][str(label)] += 1
    completed_ids = {record["annotation_id"] for record in records}
    failed_ids = {record["annotation_id"] for record in failures}
    selected_ids = {row["annotation_id"] for row in selected_rows}
    missing_ids = sorted(selected_ids - completed_ids - failed_ids)
    if completed_ids.intersection(failed_ids):
        raise ValueError("An annotation_id exists in records and failed")

    if stop_status is not None:
        status = stop_status
    else:
        status = (
            "COMPLETED"
            if len(records) == len(selected_rows) and not failures
            else "COMPLETED_WITH_FAILURES"
            if len(records) + len(failures) == len(selected_rows)
            else "INCOMPLETE"
        )
    return {
        "artifact_type": "LLM_PSEUDO_LABEL_PASS",
        "annotation_records": len(records),
        "aspect_label_counts": {
            aspect: dict(sorted(counter.items()))
            for aspect, counter in aspect_label_counts.items()
        },
        "completed_at": _utc_now(),
        "failed_records": len(failures),
        "missing_annotation_ids": missing_ids,
        "run_config": run_config,
        "selected_records": len(selected_rows),
        "status": status,
        "status_counts": dict(sorted(status_counts.items())),
        "stop_reason": stop_reason,
    }


def run_pass(
    *,
    config_path: Path,
    pass_id: str,
    output: Path,
    max_records: int | None,
    resume: bool,
    retry_failures: bool,
    dry_run: bool,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Config prompt_version does not match code")
    if config.get("schema_version") != LLM_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("Config schema_version does not match code")
    base_dir = config_path.parent.parent
    input_path = (base_dir / config["pilot_input"]).resolve()
    guideline_path = (base_dir / config["guideline"]).resolve()
    schema_path = (base_dir / config["annotation_output_schema"]).resolve()
    source_release = (base_dir / config["source_release"]).resolve()
    source_release_manifest_path = source_release / "manifest.json"
    for path in (
        input_path,
        guideline_path,
        schema_path,
        source_release_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    pinned_sha256 = config.get("pinned_sha256")
    if not isinstance(pinned_sha256, dict):
        raise ValueError("Config pinned_sha256 must be an object")
    if pinned_sha256.get("guideline") != _sha256_file(guideline_path):
        raise ValueError("Configured guideline checksum pin mismatch")
    if pinned_sha256.get("annotation_output_schema") != _sha256_file(
        schema_path
    ):
        raise ValueError("Configured output-schema checksum pin mismatch")
    validate_output_schema_contract(_read_json(schema_path))
    rows = _normalize_blind_input(_read_jsonl(input_path))
    pilot_provenance = _validate_pilot_package(
        input_path=input_path,
        rows=rows,
        guideline_path=guideline_path,
        schema_path=schema_path,
    )
    source_release_manifest = _read_json(source_release_manifest_path)
    if source_release_manifest.get("release_id") != pilot_provenance.get(
        "source_release_id"
    ):
        raise ValueError("Configured source release ID mismatch")
    if _sha256_file(source_release_manifest_path) != pilot_provenance.get(
        "source_release_manifest_sha256"
    ):
        raise ValueError("Configured source release manifest mismatch")
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("--max-records must be positive")
        rows = rows[:max_records]
    if not rows:
        raise ValueError("No pilot rows selected")

    pass_config = _pass_config(config, pass_id)
    prompt_variant = pass_config.get("prompt_variant")
    seed = pass_config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"Pass seed must be an integer: {pass_id}")
    guideline_text = guideline_path.read_text(encoding="utf-8")
    system_prompt = build_system_prompt(
        guideline_text,
        prompt_variant=prompt_variant,
    )
    backend = config["backend"]
    generation = config["generation"]
    endpoint = sanitized_endpoint(str(backend["endpoint"]))
    api_key_env = backend.get("api_key_env")

    public_backend_config = {
        "api_key_env": api_key_env,
        "endpoint": endpoint,
        "model": backend["model"],
        "request_json_mode": bool(backend.get("request_json_mode", True)),
        "type": backend["type"],
    }
    run_config = {
        "annotation_output_schema_sha256": _sha256_file(schema_path),
        "backend": public_backend_config,
        "config_sha256": _sha256_file(config_path),
        "code_sha256": {
            "scripts/run_llm_annotation.py": _sha256_file(
                Path(__file__).resolve()
            ),
            "src/lazada_collector/llm_annotation.py": _sha256_file(
                base_dir / "src" / "lazada_collector" / "llm_annotation.py"
            ),
            "src/lazada_collector/llm_backends.py": _sha256_file(
                base_dir / "src" / "lazada_collector" / "llm_backends.py"
            ),
        },
        "data_transmission_notice": (
            "Review text is transmitted to the configured external endpoint "
            "when the endpoint is not localhost."
        ),
        "generation": generation,
        "guideline_sha256": _sha256_file(guideline_path),
        "input_sha256": _sha256_file(input_path),
        "pass_id": pass_id,
        "pilot": pilot_provenance,
        "prompt_sha256": sha256_text(system_prompt),
        "prompt_variant": prompt_variant,
        "prompt_version": PROMPT_VERSION,
        "schema_version": LLM_ANNOTATION_SCHEMA_VERSION,
        "seed": seed,
        "selected_records": len(rows),
    }
    run_fingerprint = sha256_text(canonical_json(run_config))
    run_config["run_fingerprint"] = run_fingerprint

    if dry_run:
        return {
            "status": "DRY_RUN_VALID",
            "output": str(output.resolve()),
            "run_config": run_config,
        }

    if (
        backend["type"] == "openai_compatible"
        and api_key_env
        and not os.environ.get(api_key_env)
    ):
        raise RuntimeError(
            f"Required environment variable is not set: {api_key_env}"
        )

    output = output.resolve()
    if output.exists() and not resume:
        raise FileExistsError(
            f"Output exists; pass --resume to continue: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    stored_config_path = output / "run_config.json"
    if stored_config_path.exists():
        stored = _read_json(stored_config_path)
        if canonical_json(stored) != canonical_json(run_config):
            raise ValueError("Existing run_config.json does not match this run")
    else:
        _atomic_write_json(stored_config_path, run_config)
    (output / "records").mkdir(exist_ok=True)
    (output / "failed").mkdir(exist_ok=True)
    (output / "attempts").mkdir(exist_ok=True)
    _verify_run_checksums(output)
    mutated = False
    existing_manifest_path = output / "manifest.json"
    if existing_manifest_path.is_file():
        existing_status = _read_json(existing_manifest_path).get("status")
        if existing_status == "ABORTED_BACKEND_FATAL":
            raise RuntimeError(
                "This run was aborted by a fatal backend/configuration error; "
                "fix configuration and start a new output directory."
            )
        if existing_status == "PAUSED_RATE_LIMIT" and not retry_failures:
            raise RuntimeError(
                "This run is paused by rate limiting; resume with "
                "--retry-failures after the provider cooldown."
            )
    if resume and retry_failures:
        if (output / "SHA256SUMS.txt").is_file():
            mutated = True
        _archive_run_checksums_for_retry(output)

    max_retries = int(generation["max_retries"])
    if max_retries <= 0:
        raise ValueError("generation.max_retries must be positive")
    max_retry_after_seconds = float(
        generation.get("max_retry_after_seconds", 120.0)
    )
    if max_retry_after_seconds <= 0:
        raise ValueError(
            "generation.max_retry_after_seconds must be positive"
        )
    completed = 0
    failed = 0
    skipped = 0
    stop_status: str | None = None
    stop_reason: str | None = None
    for position, row in enumerate(rows, 1):
        name = _record_name(row)
        record_path = output / "records" / name
        failure_path = output / "failed" / name
        (
            attempt_summaries,
            attempt_offset,
            validation_feedback,
            valid_attempt,
            last_attempt_record,
        ) = _scan_attempt_ledger(
            output=output,
            row=row,
            pass_id=pass_id,
            run_fingerprint=run_fingerprint,
            system_prompt=system_prompt,
        )
        prior_failure_attempts = 0
        if failure_path.is_file():
            prior_failure_attempts = _validate_prior_failure(
                failure_path=failure_path,
                row=row,
                run_fingerprint=run_fingerprint,
                attempt_summaries=attempt_summaries,
            )

        if (
            record_path.is_file() is False
            and failure_path.is_file() is False
            and last_attempt_record is not None
            and last_attempt_record.get("outcome") == "BACKEND_ERROR"
            and last_attempt_record.get("retryable") is False
        ):
            _atomic_write_json(
                failure_path,
                {
                    "annotation_id": row["annotation_id"],
                    "artifact_type": "LLM_PSEUDO_LABEL_FAILURE",
                    "attempts": attempt_summaries,
                    "failed_at": _utc_now(),
                    "pass_id": pass_id,
                    "review_text_sha256": row["review_text_sha256"],
                    "run_fingerprint": run_fingerprint,
                },
            )
            mutated = True
            failed += 1
            stop_status = "ABORTED_BACKEND_FATAL"
            stop_reason = (
                "recovered_nonretryable_backend_error"
                f";status_code={last_attempt_record.get('status_code')}"
            )
            print(
                f"[llm-stop] pass={pass_id} {position}/{len(rows)} "
                f"status={stop_status} reason={stop_reason}"
            )
            break

        if record_path.is_file():
            record = _validate_existing_record(
                record_path,
                row=row,
                run_fingerprint=run_fingerprint,
                pass_id=pass_id,
                prompt_sha256=run_config["prompt_sha256"],
                schema_sha256=run_config[
                    "annotation_output_schema_sha256"
                ],
                attempt_summaries=attempt_summaries,
            )
            if valid_attempt is None:
                raise ValueError(
                    f"Terminal record has no valid attempt: {record_path}"
                )
            if record.get("attempt_count") != valid_attempt.get("attempt"):
                raise ValueError(
                    f"Terminal record attempt mismatch: {record_path}"
                )
            if record.get("raw_response_sha256") != valid_attempt.get(
                "raw_response_sha256"
            ):
                raise ValueError(
                    f"Terminal/attempt response mismatch: {record_path}"
                )
            if record.get("request_sha256") != valid_attempt.get(
                "request_sha256"
            ):
                raise ValueError(
                    f"Terminal/attempt request mismatch: {record_path}"
                )
            if failure_path.is_file():
                _archive_failure(
                    output=output,
                    failure_path=failure_path,
                    last_attempt=prior_failure_attempts,
                )
                mutated = True
            completed += 1
            skipped += 1
            print(
                f"[llm-progress] pass={pass_id} {position}/{len(rows)} "
                f"status=resume-skip ok={completed} failed={failed}"
            )
            continue

        if valid_attempt is not None:
            normalized = valid_attempt.pop("_normalized_annotation")
            final_record = _build_final_record(
                row=row,
                attempt_record=valid_attempt,
                normalized=normalized,
                attempt_summaries=attempt_summaries,
                run_config=run_config,
                run_fingerprint=run_fingerprint,
                pass_id=pass_id,
                prompt_variant=prompt_variant,
            )
            _atomic_write_json(record_path, final_record)
            mutated = True
            _archive_failure(
                output=output,
                failure_path=failure_path,
                last_attempt=prior_failure_attempts,
            )
            completed += 1
            print(
                f"[llm-progress] pass={pass_id} {position}/{len(rows)} "
                f"status=recovered-valid-attempt ok={completed} "
                f"failed={failed} skipped={skipped}"
            )
            continue

        if failure_path.is_file():
            if not resume:
                raise ValueError(f"Unexpected prior failure: {failure_path}")
            if not retry_failures:
                if prior_failure_attempts != len(attempt_summaries):
                    _atomic_write_json(
                        failure_path,
                        {
                            "annotation_id": row["annotation_id"],
                            "artifact_type": "LLM_PSEUDO_LABEL_FAILURE",
                            "attempts": attempt_summaries,
                            "failed_at": _utc_now(),
                            "pass_id": pass_id,
                            "review_text_sha256": row[
                                "review_text_sha256"
                            ],
                            "run_fingerprint": run_fingerprint,
                        },
                    )
                    mutated = True
                failed += 1
                skipped += 1
                print(
                    f"[llm-progress] pass={pass_id} {position}/{len(rows)} "
                    f"status=resume-skip-failed ok={completed} failed={failed}"
                )
                continue

        final_record: dict[str, Any] | None = None
        last_backend_error: LLMBackendError | None = None
        if failure_path.is_file():
            attempt_budget_end = prior_failure_attempts + max_retries
        else:
            attempt_budget_end = max_retries
        for attempt in range(attempt_offset + 1, attempt_budget_end + 1):
            user_message = _attempt_user_message(
                row=row,
                validation_feedback=validation_feedback,
            )
            request_hash = sha256_text(
                system_prompt + "\0" + user_message
            )
            started_at = _utc_now()
            attempt_record: dict[str, Any] = {
                "annotation_id": row["annotation_id"],
                "attempt": attempt,
                "pass_id": pass_id,
                "request_sha256": request_hash,
                "review_text_sha256": row["review_text_sha256"],
                "run_fingerprint": run_fingerprint,
                "started_at": started_at,
            }
            backend_error: LLMBackendError | None = None
            try:
                response = generate_json(
                    GenerationRequest(
                        backend=backend["type"],
                        endpoint=endpoint,
                        model=backend["model"],
                        system_prompt=system_prompt,
                        user_message=user_message,
                        temperature=float(generation["temperature"]),
                        max_tokens=int(generation["max_tokens"]),
                        seed=seed,
                        api_key_env=api_key_env,
                        timeout_seconds=float(generation["timeout_seconds"]),
                        request_json_mode=bool(
                            backend.get("request_json_mode", True)
                        ),
                    )
                )
                last_backend_error = None
                response_sha = sha256_text(response.content)
                attempt_record.update(
                    {
                        "completed_at": _utc_now(),
                        "finish_reason": response.finish_reason,
                        "outcome": "RESPONSE_RECEIVED",
                        "provider_model": response.provider_model,
                        "provider_request_id": response.provider_request_id,
                        "raw_response": response.content,
                        "raw_response_sha256": response_sha,
                        "usage": response.usage,
                    }
                )
                try:
                    normalized = validate_and_normalize_annotation(
                        row["review_text"],
                        parse_model_json(response.content),
                    )
                except AnnotationValidationError as exc:
                    validation_feedback = str(exc)
                    attempt_record["outcome"] = "SCHEMA_INVALID"
                    attempt_record["validation_error"] = validation_feedback
                else:
                    attempt_record["outcome"] = "VALID"
                    final_record = {}
            except LLMBackendError as exc:
                backend_error = exc
                last_backend_error = exc
                attempt_record.update(
                    {
                        "backend_error": str(exc),
                        "completed_at": _utc_now(),
                        "outcome": "BACKEND_ERROR",
                        "retry_after_seconds": exc.retry_after_seconds,
                        "retryable": exc.retryable,
                        "status_code": exc.status_code,
                    }
                )

            attempt_path = (
                output
                / "attempts"
                / f"{Path(name).stem}-attempt-{attempt}.json"
            )
            _atomic_write_json(attempt_path, attempt_record)
            mutated = True
            attempt_summaries.append(
                {
                    "attempt": attempt,
                    "attempt_path": attempt_path.relative_to(output).as_posix(),
                    "outcome": attempt_record["outcome"],
                    "request_sha256": request_hash,
                }
            )
            if final_record is not None:
                final_record = _build_final_record(
                    row=row,
                    attempt_record=attempt_record,
                    normalized=normalized,
                    attempt_summaries=attempt_summaries,
                    run_config=run_config,
                    run_fingerprint=run_fingerprint,
                    pass_id=pass_id,
                    prompt_variant=prompt_variant,
                )
                _atomic_write_json(record_path, final_record)
                _archive_failure(
                    output=output,
                    failure_path=failure_path,
                    last_attempt=prior_failure_attempts,
                )
                completed += 1
                break
            if backend_error and not backend_error.retryable:
                stop_status = "ABORTED_BACKEND_FATAL"
                stop_reason = (
                    "nonretryable_backend_error"
                    f";status_code={backend_error.status_code}"
                )
                break
            retry_delay = _retry_delay(
                attempt - attempt_offset,
                backend_error,
            )
            if (
                backend_error
                and backend_error.status_code == 429
                and retry_delay > max_retry_after_seconds
            ):
                stop_status = "PAUSED_RATE_LIMIT"
                stop_reason = (
                    "retry_after_exceeds_bound"
                    f";retry_after_seconds={retry_delay}"
                )
                break
            if attempt < attempt_budget_end:
                time.sleep(retry_delay)

        if final_record is None:
            _atomic_write_json(
                failure_path,
                {
                    "annotation_id": row["annotation_id"],
                    "artifact_type": "LLM_PSEUDO_LABEL_FAILURE",
                    "attempts": attempt_summaries,
                    "failed_at": _utc_now(),
                    "pass_id": pass_id,
                    "review_text_sha256": row["review_text_sha256"],
                    "run_fingerprint": run_fingerprint,
                },
            )
            mutated = True
            failed += 1
            if (
                stop_status is None
                and last_backend_error is not None
                and last_backend_error.status_code == 429
            ):
                stop_status = "PAUSED_RATE_LIMIT"
                stop_reason = "rate_limit_retry_budget_exhausted"
        print(
            f"[llm-progress] pass={pass_id} {position}/{len(rows)} "
            f"status={'ok' if final_record else 'failed'} "
            f"ok={completed} failed={failed} skipped={skipped}"
        )
        if stop_status is not None:
            print(
                f"[llm-stop] status={stop_status} reason={stop_reason}"
            )
            break

    if (
        not mutated
        and skipped == len(rows)
        and existing_manifest_path.is_file()
        and (output / "SHA256SUMS.txt").is_file()
    ):
        existing_manifest = _read_json(existing_manifest_path)
        return {
            "status": existing_manifest["status"],
            "selected_records": len(rows),
            "annotation_records": existing_manifest[
                "annotation_records"
            ],
            "failed_records": existing_manifest["failed_records"],
            "output": str(output),
            "manifest": str(existing_manifest_path),
            "no_op_resume": True,
        }

    manifest = _summarize(
        run_dir=output,
        selected_rows=rows,
        run_config=run_config,
        stop_status=stop_status,
        stop_reason=stop_reason,
    )
    _atomic_write_json(output / "manifest.json", manifest)
    _write_checksums(output)
    return {
        "status": manifest["status"],
        "selected_records": len(rows),
        "annotation_records": manifest["annotation_records"],
        "failed_records": manifest["failed_records"],
        "output": str(output),
        "manifest": str(output / "manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="On resume, append new attempts for prior terminal failures.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_pass(
        config_path=args.config,
        pass_id=args.pass_id,
        output=args.output,
        max_records=args.max_records,
        resume=args.resume,
        retry_failures=args.retry_failures,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    status = result.get("status")
    if status in {"COMPLETED", "DRY_RUN_VALID"}:
        return 0
    if status == "PAUSED_RATE_LIMIT":
        return 2
    if status == "COMPLETED_WITH_FAILURES":
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
