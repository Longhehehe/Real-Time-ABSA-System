"""Run resumable calibrated ABSA AI pre-annotation.

Two scopes are supported:

``diagnostic``
    Predict a frozen 20-record holdout using only the other completed human
    records as examples and publish an alignment gate.

``primary``
    Label the blinded 5,000-review tranche.  This scope refuses to start until
    the diagnostic gate passes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence

from lazada_collector.ai_tranche import (
    COMPACT_PROMPT_VERSION,
    build_compact_system_prompt,
    build_compact_user_message,
    calibration_gate,
    calibration_metrics,
    count_labels,
    parse_compact_batch_partial,
    retrieve_examples,
    sha256_file,
    stable_id,
)
from lazada_collector.llm_annotation import (
    AnnotationValidationError,
    canonical_json,
    sha256_text,
)
from lazada_collector.llm_backends import (
    GenerationRequest,
    GenerationResponse,
    LLMBackendError,
    generate_json,
    sanitized_endpoint,
)


DEFAULT_PACKAGE = Path(
    "data/annotations/absa_ai_tranche_5000_v1_20260727"
)
DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mistral-medium-3.5-128b"
DEFAULT_CODEX_MODEL = "gpt-5.6-terra"
DEFAULT_CODEX_SCHEMA = Path(
    "configs/absa_compact_batch_output_schema_v1.json"
)
RUN_RECORD_SCHEMA_VERSION = "absa-ai-run-record/1.0.0"
ATTEMPT_SCHEMA_VERSION = "absa-ai-run-attempt/1.0.0"
CODEX_DISABLED_FEATURES = (
    "shell_tool",
    "apps",
    "browser_use",
    "computer_use",
    "image_generation",
    "multi_agent",
    "plugins",
)
CODEX_ENV_ALLOWLIST = (
    "APPDATA",
    "CODEX_HOME",
    "COMSPEC",
    "HOMEDRIVE",
    "HOMEPATH",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invocation_id(
    *,
    tranche_id: str,
    scope: str,
    started_at: str,
    process_id: int,
) -> str:
    return stable_id(
        "inv-",
        tranche_id,
        scope,
        started_at,
        str(process_id),
        length=16,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _verify_prepared_package(package: Path) -> dict[str, Any]:
    manifest_path = package / "prepare_manifest.json"
    sums_path = package / "INPUT_SHA256SUMS.txt"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "PREPARED_NOT_LABELED":
        raise ValueError("Package is not in prepared state")
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        checksum, relative = line.split("  ", 1)
        if relative in sums:
            raise ValueError(f"Duplicate input checksum: {relative}")
        sums[relative] = checksum
    expected = {"prepare_manifest.json"}
    for artifact in manifest.get("artifacts", []):
        relative = artifact["path"]
        path = package / relative
        actual = sha256_file(path)
        if artifact.get("sha256") != actual:
            raise ValueError(f"Prepared artifact mismatch: {relative}")
        if artifact.get("bytes") != path.stat().st_size:
            raise ValueError(f"Prepared byte count mismatch: {relative}")
        if sums.get(relative) != actual:
            raise ValueError(f"Input SHA256SUMS mismatch: {relative}")
        expected.add(relative)
    if sums.get("prepare_manifest.json") != sha256_file(manifest_path):
        raise ValueError("Prepared manifest checksum mismatch")
    if set(sums) != expected:
        raise ValueError("Prepared checksum closure mismatch")
    return manifest


def _batches(
    rows: Sequence[dict[str, Any]],
    size: int,
) -> list[list[dict[str, Any]]]:
    return [list(rows[index:index + size]) for index in range(0, len(rows), size)]


def _codex_subprocess_env() -> dict[str, str]:
    """Build a minimal environment without forwarding provider/API secrets."""

    environment = {
        key: value
        for key in CODEX_ENV_ALLOWLIST
        if (value := os.environ.get(key))
    }
    environment["NO_COLOR"] = "1"
    return environment


class Runner:
    def __init__(
        self,
        *,
        package: Path,
        scope: str,
        backend: str,
        endpoint: str,
        model: str,
        api_key_env: str,
        batch_size: int,
        workers: int,
        max_retries: int,
        timeout_seconds: float,
        max_tokens: int,
        reasoning_effort: str | None,
        enable_thinking: bool | None,
        codex_schema: Path,
    ) -> None:
        self.package = package
        self.scope = scope
        self.backend = backend
        self.endpoint = (
            sanitized_endpoint(endpoint)
            if backend == "nvidia"
            else "codex-cli://local-authenticated-session"
        )
        self.model = model
        self.api_key_env = api_key_env
        self.batch_size = batch_size
        self.workers = workers
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.codex_schema = codex_schema.resolve()
        self.codex_schema_sha256: str | None = None
        if self.backend == "codex":
            if not self.codex_schema.is_file():
                raise FileNotFoundError(self.codex_schema)
            self.codex_schema_sha256 = sha256_file(self.codex_schema)
            self.codex_executable = shutil.which("codex")
            if not self.codex_executable:
                raise FileNotFoundError("codex executable was not found")
        else:
            self.codex_executable = None
        self.manifest = _verify_prepared_package(package)
        self.run_root = package / "runs" / scope
        self.records_root = self.run_root / "records"
        self.attempts_root = self.run_root / "attempts"
        self.failures_root = self.run_root / "failed"
        self.print_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.fatal_backend_lock = threading.Lock()
        self.fatal_backend_error: str | None = None
        self.completed = 0
        self.failed = 0
        self.calls = 0
        self.usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        calibration_rows = _read_jsonl(
            package / "calibration" / "human_confirmed.jsonl"
        )
        split = _read_json(
            package / "calibration" / "diagnostic_split.json"
        )
        accepted_rows = [
            row
            for row in calibration_rows
            if row.get("semantic_audit", {}).get("decision")
            == "CALIBRATION_ACCEPT"
        ]
        excluded_rows = [
            row
            for row in calibration_rows
            if row.get("semantic_audit", {}).get("decision")
            == "EXCLUDE_PENDING_EXPERT_ADJUDICATION"
        ]
        if len(accepted_rows) + len(excluded_rows) != len(calibration_rows):
            raise ValueError("Calibration semantic-audit state is incomplete")
        calibration_by_id = {
            row["calibration_id"]: row for row in accepted_rows
        }
        holdout_ids = set(split["diagnostic_holdout_ids"])
        prompt_ids = set(split["prompt_calibration_ids"])
        if holdout_ids.intersection(prompt_ids):
            raise ValueError("Calibration/holdout split overlap")
        if holdout_ids | prompt_ids != set(calibration_by_id):
            raise ValueError("Calibration split is not closed")

        if scope == "diagnostic":
            self.examples = [
                calibration_by_id[item] for item in sorted(prompt_ids)
            ]
            self.targets = [
                {
                    "annotation_id": item,
                    "reviewContent": calibration_by_id[item]["reviewContent"],
                    "review_text_sha256": calibration_by_id[item][
                        "review_text_sha256"
                    ],
                    "selection_rank": index + 1,
                }
                for index, item in enumerate(sorted(holdout_ids))
            ]
            self.expected_holdout = [
                calibration_by_id[item] for item in sorted(holdout_ids)
            ]
        else:
            gate_path = package / "calibration" / "diagnostic_gate.json"
            if not gate_path.is_file():
                raise ValueError(
                    "Primary run requires completed diagnostic gate"
                )
            gate = _read_json(gate_path)
            if gate.get("status") != "PASS":
                raise ValueError("Diagnostic gate did not pass")
            self.examples = accepted_rows
            self.targets = _read_jsonl(
                package / "input" / "blind_reviews.jsonl"
            )
            self.expected_holdout = []

        calibration_payload_sha = self.manifest["human_calibration"][
            "calibration_payload_sha256"
        ]
        self.system_prompt = build_compact_system_prompt(
            guideline_sha256=self.manifest["guideline"]["sha256"],
            calibration_payload_sha256=calibration_payload_sha,
        )
        self.system_prompt_sha256 = sha256_text(self.system_prompt)

    def _generate_response(
        self,
        *,
        batch_id: str,
        attempt: int,
        system_prompt: str,
        user_message: str,
        seed: int,
    ) -> GenerationResponse:
        if self.backend == "nvidia":
            return generate_json(
                GenerationRequest(
                    backend="openai_compatible",
                    endpoint=self.endpoint,
                    model=self.model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    seed=seed,
                    api_key_env=self.api_key_env,
                    timeout_seconds=self.timeout_seconds,
                    request_json_mode=True,
                    reasoning_effort=self.reasoning_effort,
                    enable_thinking=self.enable_thinking,
                )
            )

        invocation_root = (
            self.run_root
            / "codex_invocations"
            / batch_id
            / f"attempt-{attempt:02d}"
        )
        invocation_root.mkdir(parents=True, exist_ok=True)
        output_path = invocation_root / "last_message.json"
        command = [
            str(self.codex_executable),
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
        ]
        if self.reasoning_effort is not None:
            command.extend(
                (
                    "-c",
                    f'model_reasoning_effort="{self.reasoning_effort}"',
                )
            )
        for feature in CODEX_DISABLED_FEATURES:
            command.extend(("--disable", feature))
        command.extend(
            [
                "--output-schema",
                str(self.codex_schema),
                "--output-last-message",
                str(output_path),
                "--model",
                self.model,
                "--cd",
                str(invocation_root),
                "-",
            ]
        )
        prompt = (
            system_prompt
            + "\n\n===== BATCH INPUT =====\n"
            + user_message
            + "\n\nReturn only the schema-conforming JSON object."
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=_codex_subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMBackendError(
                f"Codex CLI timed out after {self.timeout_seconds}s",
                retryable=True,
            ) from exc
        duration = time.monotonic() - started
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-1000:]
            normalized_detail = detail.casefold()
            invalid_request = any(
                marker in normalized_detail
                for marker in (
                    "invalid_json_schema",
                    "invalid_request_error",
                )
            )
            raise LLMBackendError(
                f"Codex CLI exit {completed.returncode}: {detail}",
                retryable=not invalid_request,
                status_code=400 if invalid_request else None,
            )
        if not output_path.is_file():
            raise LLMBackendError(
                "Codex CLI did not create --output-last-message",
                retryable=True,
            )
        content = output_path.read_text(encoding="utf-8")
        if not content.strip():
            raise LLMBackendError(
                "Codex CLI returned an empty final message",
                retryable=True,
            )
        return GenerationResponse(
            content=content,
            provider_model=self.model,
            finish_reason="completed",
            usage={"wall_seconds": duration},
            provider_request_id=None,
        )

    def _record_path(self, row: Mapping[str, Any]) -> Path:
        rank = int(row["selection_rank"])
        return self.records_root / (
            f"{rank:05d}-{row['annotation_id']}.json"
        )

    def _load_existing(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._record_path(row)
        if not path.is_file():
            return None
        value = _read_json(path)
        if value.get("schema_version") != RUN_RECORD_SCHEMA_VERSION:
            raise ValueError(f"Existing run record schema mismatch: {path}")
        if value.get("annotation_id") != row["annotation_id"]:
            raise ValueError(f"Existing run record ID mismatch: {path}")
        if value.get("review_text_sha256") != row["review_text_sha256"]:
            raise ValueError(f"Existing run record hash mismatch: {path}")
        generation = value.get("generation", {})
        if generation.get("backend") != self.backend:
            raise ValueError(f"Existing run record backend mismatch: {path}")
        if generation.get("model") != self.model:
            raise ValueError(f"Existing run record model mismatch: {path}")
        if generation.get("system_prompt_sha256") != self.system_prompt_sha256:
            raise ValueError(f"Existing run record prompt mismatch: {path}")
        if generation.get("reasoning_effort") != self.reasoning_effort:
            raise ValueError(
                f"Existing run record reasoning-effort mismatch: {path}"
            )
        if generation.get("enable_thinking") != self.enable_thinking:
            raise ValueError(
                f"Existing run record thinking-mode mismatch: {path}"
            )
        if generation.get("codex_schema_sha256") != (
            self.codex_schema_sha256
        ):
            raise ValueError(f"Existing run record Codex schema mismatch: {path}")
        return value

    def _attempt_path(
        self,
        *,
        batch_id: str,
        attempt: int,
    ) -> Path:
        return self.attempts_root / batch_id / f"attempt-{attempt:02d}.json"

    def _write_attempt(
        self,
        *,
        batch_id: str,
        attempt: int,
        target_ids: list[str],
        user_message_sha256: str,
        started_at: str,
        outcome: str,
        response_content: str | None,
        error: str | None,
        response_metadata: Mapping[str, Any] | None,
    ) -> None:
        metadata = dict(response_metadata or {})
        value = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "scope": self.scope,
            "batch_id": batch_id,
            "attempt": attempt,
            "target_ids": target_ids,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "outcome": outcome,
            "system_prompt_sha256": self.system_prompt_sha256,
            "user_message_sha256": user_message_sha256,
            "provider": {
                "backend": self.backend,
                "endpoint": self.endpoint,
                "requested_model": self.model,
                "requested_reasoning_effort": self.reasoning_effort,
                "requested_enable_thinking": self.enable_thinking,
                "codex_schema_sha256": self.codex_schema_sha256,
                **metadata,
            },
            "response_content": response_content,
            "error": error,
        }
        _atomic_json(
            self._attempt_path(batch_id=batch_id, attempt=attempt),
            value,
        )

    def _save_records(
        self,
        rows: Sequence[dict[str, Any]],
        normalized: Sequence[dict[str, Any]],
        *,
        batch_id: str,
        attempt: int,
        example_ids: list[str],
    ) -> None:
        normalized_by_id = {
            row["annotation_id"]: row for row in normalized
        }
        for source in rows:
            annotation_id = source["annotation_id"]
            result = normalized_by_id[annotation_id]
            value = {
                "schema_version": RUN_RECORD_SCHEMA_VERSION,
                "artifact_type": "AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION",
                "scope": self.scope,
                "annotation_id": annotation_id,
                "selection_rank": source["selection_rank"],
                "review_text_sha256": source["review_text_sha256"],
                "annotation": result["annotation"],
                "normalization_repairs": result.get(
                    "normalization_repairs",
                    [],
                ),
                "generation": {
                    "backend": self.backend,
                    "batch_id": batch_id,
                    "valid_attempt": attempt,
                    "model": self.model,
                    "prompt_version": COMPACT_PROMPT_VERSION,
                    "system_prompt_sha256": self.system_prompt_sha256,
                    "human_example_ids": example_ids,
                    "reasoning_effort": self.reasoning_effort,
                    "enable_thinking": self.enable_thinking,
                    "codex_schema_sha256": self.codex_schema_sha256,
                },
            }
            _atomic_json(self._record_path(source), value)

    def _fatal_backend_snapshot(self) -> str | None:
        with self.fatal_backend_lock:
            return self.fatal_backend_error

    def _open_fatal_backend_circuit(self, error: str) -> None:
        with self.fatal_backend_lock:
            if self.fatal_backend_error is None:
                self.fatal_backend_error = error

    def _save_failures(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        last_error: str,
        batch_id: str,
        backend_fatal: bool,
    ) -> None:
        for row in rows:
            failure = {
                "schema_version": "absa-ai-run-failure/1.0.0",
                "scope": self.scope,
                "annotation_id": row["annotation_id"],
                "review_text_sha256": row["review_text_sha256"],
                "failed_at": _utc_now(),
                "last_error": last_error,
                "batch_id": batch_id,
                "backend_fatal": backend_fatal,
            }
            _atomic_json(
                self.failures_root / f"{row['annotation_id']}.json",
                failure,
            )

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        *,
        lineage: str,
    ) -> tuple[int, int]:
        pending = [row for row in rows if self._load_existing(row) is None]
        if not pending:
            return len(rows), 0
        batch_id = stable_id(
            f"{self.scope}-",
            lineage,
            *[row["annotation_id"] for row in pending],
            length=18,
        )
        existing_fatal = self._fatal_backend_snapshot()
        if existing_fatal is not None:
            self._save_failures(
                pending,
                last_error=existing_fatal,
                batch_id=batch_id,
                backend_fatal=True,
            )
            return 0, len(pending)
        remaining = list(pending)
        valid_total = 0
        last_error = ""
        fatal_backend = False
        for attempt in range(1, self.max_retries + 1):
            existing_fatal = self._fatal_backend_snapshot()
            if existing_fatal is not None:
                last_error = existing_fatal
                fatal_backend = True
                break
            examples = retrieve_examples(
                remaining,
                self.examples,
                per_target=3,
                maximum=14,
            )
            example_ids = [row["calibration_id"] for row in examples]
            user_message = build_compact_user_message(remaining, examples)
            started_at = _utc_now()
            attempt_message = user_message
            if last_error:
                attempt_message = (
                    user_message
                    + "\nLần trước không hợp lệ. Hãy trả lại toàn bộ JSON đã sửa. "
                    + "Lỗi validator: "
                    + last_error[:800]
                )
            seed = int(
                hashlib.sha256(
                    f"{batch_id}\0{attempt}".encode("utf-8")
                ).hexdigest()[:8],
                16,
            )
            response = None
            try:
                response = self._generate_response(
                    batch_id=batch_id,
                    attempt=attempt,
                    system_prompt=self.system_prompt,
                    user_message=attempt_message,
                    seed=seed,
                )
                with self.progress_lock:
                    self.calls += 1
                    for key in self.usage:
                        value = response.usage.get(key)
                        if isinstance(value, int):
                            self.usage[key] += value
                normalized, row_errors = parse_compact_batch_partial(
                    response.content,
                    expected_rows=remaining,
                )
                valid_ids = {
                    row["annotation_id"] for row in normalized
                }
                valid_sources = [
                    row
                    for row in remaining
                    if row["annotation_id"] in valid_ids
                ]
                if normalized:
                    self._save_records(
                        valid_sources,
                        normalized,
                        batch_id=batch_id,
                        attempt=attempt,
                        example_ids=example_ids,
                    )
                    valid_total += len(normalized)
                outcome = "VALID" if not row_errors else "PARTIAL_VALID"
                self._write_attempt(
                    batch_id=batch_id,
                    attempt=attempt,
                    target_ids=[
                        row["annotation_id"] for row in remaining
                    ],
                    user_message_sha256=sha256_text(attempt_message),
                    started_at=started_at,
                    outcome=outcome,
                    response_content=response.content,
                    error=(
                        None
                        if not row_errors
                        else canonical_json(row_errors)
                    ),
                    response_metadata={
                        "provider_model": response.provider_model,
                        "finish_reason": response.finish_reason,
                        "provider_request_id": response.provider_request_id,
                        "usage": response.usage,
                    },
                )
                if not row_errors:
                    return valid_total, 0
                last_error = canonical_json(row_errors)
                remaining = [
                    row
                    for row in remaining
                    if row["annotation_id"] in row_errors
                ]
            except AnnotationValidationError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                content = (
                    response.content
                    if response is not None
                    and isinstance(response.content, str)
                    else None
                )
                metadata = (
                    {
                        "provider_model": response.provider_model,
                        "finish_reason": response.finish_reason,
                        "provider_request_id": response.provider_request_id,
                        "usage": response.usage,
                    }
                    if response is not None
                    else {}
                )
                self._write_attempt(
                    batch_id=batch_id,
                    attempt=attempt,
                    target_ids=[
                        row["annotation_id"] for row in remaining
                    ],
                    user_message_sha256=sha256_text(attempt_message),
                    started_at=started_at,
                    outcome="SCHEMA_INVALID",
                    response_content=content,
                    error=last_error,
                    response_metadata=metadata,
                )
            except LLMBackendError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._write_attempt(
                    batch_id=batch_id,
                    attempt=attempt,
                    target_ids=[
                        row["annotation_id"] for row in remaining
                    ],
                    user_message_sha256=sha256_text(attempt_message),
                    started_at=started_at,
                    outcome="BACKEND_ERROR",
                    response_content=None,
                    error=last_error,
                    response_metadata={
                        "status_code": exc.status_code,
                        "retryable": exc.retryable,
                    },
                )
                if not exc.retryable:
                    self._open_fatal_backend_circuit(last_error)
                    fatal_backend = True
                    break
                wait = exc.retry_after_seconds
                if wait is None:
                    wait = min(30.0, 2.0 ** (attempt - 1))
                time.sleep(max(0.0, wait) + random.random() * 0.25)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._write_attempt(
                    batch_id=batch_id,
                    attempt=attempt,
                    target_ids=[
                        row["annotation_id"] for row in remaining
                    ],
                    user_message_sha256=sha256_text(attempt_message),
                    started_at=started_at,
                    outcome="UNEXPECTED_ERROR",
                    response_content=None,
                    error=last_error,
                    response_metadata={},
                )
            if attempt < self.max_retries:
                time.sleep(min(5.0, 0.5 * attempt))

        if fatal_backend:
            self._save_failures(
                remaining,
                last_error=last_error,
                batch_id=batch_id,
                backend_fatal=True,
            )
            return valid_total, len(remaining)

        if len(remaining) > 1:
            middle = len(remaining) // 2
            left_ok, left_failed = self._process_batch(
                remaining[:middle],
                lineage=f"{lineage}-a",
            )
            right_ok, right_failed = self._process_batch(
                remaining[middle:],
                lineage=f"{lineage}-b",
            )
            return (
                valid_total + left_ok + right_ok,
                left_failed + right_failed,
            )

        self._save_failures(
            remaining,
            last_error=last_error,
            batch_id=batch_id,
            backend_fatal=False,
        )
        return valid_total, 1

    def run(self) -> dict[str, Any]:
        self.records_root.mkdir(parents=True, exist_ok=True)
        batches = _batches(self.targets, self.batch_size)
        started_at = _utc_now()
        # A resume invocation must never reuse an earlier batch directory.
        # Reusing only the frozen target IDs caused attempt-01/02 from a later
        # invocation to overwrite earlier retry artifacts while leaving an
        # older attempt-03 behind.  Binding the batch lineage to this
        # invocation preserves every attempt and keeps replay order closed.
        invocation_id = _invocation_id(
            tranche_id=self.manifest["tranche_id"],
            scope=self.scope,
            started_at=started_at,
            process_id=os.getpid(),
        )
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self._process_batch,
                    batch,
                    lineage=f"{invocation_id}-batch-{index:05d}",
                ): index
                for index, batch in enumerate(batches, 1)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    ok, failed = future.result()
                except Exception as exc:
                    ok, failed = 0, len(batches[index - 1])
                    with self.print_lock:
                        print(
                            f"[{self.scope}] batch={index}/{len(batches)} "
                            f"fatal={type(exc).__name__}:{exc}",
                            flush=True,
                        )
                with self.progress_lock:
                    self.completed += ok
                    self.failed += failed
                    completed = self.completed
                    failures = self.failed
                    calls = self.calls
                with self.print_lock:
                    print(
                        f"[{self.scope}] records={completed}/{len(self.targets)} "
                        f"failed={failures} calls={calls}",
                        flush=True,
                    )

        run_records = [
            self._load_existing(row) for row in self.targets
        ]
        available = [row for row in run_records if row is not None]
        missing = [
            row["annotation_id"]
            for row, result in zip(self.targets, run_records, strict=True)
            if result is None
        ]
        summary: dict[str, Any] = {
            "artifact_type": "ABSA_AI_PREANNOTATION_RUN_SUMMARY",
            "scope": self.scope,
            "status": "COMPLETED" if not missing else "COMPLETED_WITH_FAILURES",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "selected_records": len(self.targets),
            "valid_records": len(available),
            "missing_annotation_ids": missing,
            "provider": {
                "backend": self.backend,
                "endpoint": self.endpoint,
                "model": self.model,
                "api_key_env": self.api_key_env,
                "batch_size": self.batch_size,
                "workers": self.workers,
                "max_retries": self.max_retries,
                "max_tokens": self.max_tokens,
                "timeout_seconds": self.timeout_seconds,
                "reasoning_effort": self.reasoning_effort,
                "enable_thinking": self.enable_thinking,
                "codex_schema": (
                    {
                        "path": str(self.codex_schema),
                        "sha256": self.codex_schema_sha256,
                    }
                    if self.backend == "codex"
                    else None
                ),
            },
            "prompt": {
                "version": COMPACT_PROMPT_VERSION,
                "system_prompt_sha256": self.system_prompt_sha256,
            },
            "provider_calls_this_invocation": self.calls,
            "provider_usage_this_invocation": self.usage,
            "label_summary": count_labels(available),
        }
        _atomic_json(self.run_root / "run_summary.json", summary)

        if self.scope == "diagnostic" and not missing:
            predictions = [
                {
                    "annotation_id": row["annotation_id"],
                    "annotation": row["annotation"],
                }
                for row in available
            ]
            metrics = calibration_metrics(
                self.expected_holdout,
                predictions,
            )
            gate = calibration_gate(metrics)
            _atomic_json(
                self.package / "calibration" / "diagnostic_metrics.json",
                metrics,
            )
            _atomic_json(
                self.package / "calibration" / "diagnostic_gate.json",
                gate,
            )
            summary["diagnostic_gate"] = gate
            _atomic_json(self.run_root / "run_summary.json", summary)
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scope",
        choices=("diagnostic", "primary"),
    )
    parser.add_argument(
        "--backend",
        choices=("nvidia", "codex"),
        default="nvidia",
    )
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument(
        "--codex-schema",
        type=Path,
        default=DEFAULT_CODEX_SCHEMA,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    batch_size = args.batch_size or (4 if args.scope == "diagnostic" else 6)
    workers = args.workers or (2 if args.scope == "diagnostic" else 4)
    model = args.model or (
        DEFAULT_CODEX_MODEL if args.backend == "codex" else DEFAULT_MODEL
    )
    if batch_size <= 0 or workers <= 0 or args.max_retries <= 0:
        raise ValueError("batch-size/workers/max-retries must be positive")
    runner = Runner(
        package=args.package.resolve(),
        scope=args.scope,
        backend=args.backend,
        endpoint=args.endpoint,
        model=model,
        api_key_env=args.api_key_env,
        batch_size=batch_size,
        workers=workers,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=(
            None
            if args.thinking_mode == "auto"
            else args.thinking_mode == "on"
        ),
        codex_schema=args.codex_schema,
    )
    result = runner.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
