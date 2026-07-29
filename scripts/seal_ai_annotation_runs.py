"""Fail-closed provenance sealing for completed ABSA AI tranche runs.

This command does not trust terminal run records.  It independently replays
every stored provider response through ``parse_compact_batch_partial``, binds
each terminal record to the exact ``batch_id``/``valid_attempt`` that produced
it, and verifies the frozen prompt and provider configuration before writing a
per-scope manifest.

The default command seals both the diagnostic and primary scopes:

    python -X utf8 scripts/seal_ai_annotation_runs.py

Run the command only after the annotation runner has stopped.  Existing seals
are verified before they may be regenerated, so a post-seal mutation cannot be
silently accepted by running the command again.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

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
)
from lazada_collector.llm_annotation import (
    AnnotationValidationError,
    canonical_json,
    sha256_text,
)


DEFAULT_PACKAGE = Path(
    "data/annotations/absa_ai_tranche_5000_v1_20260727"
)
RUN_RECORD_SCHEMA_VERSION = "absa-ai-run-record/1.0.0"
ATTEMPT_SCHEMA_VERSION = "absa-ai-run-attempt/1.0.0"
FAILURE_SCHEMA_VERSION = "absa-ai-run-failure/1.0.0"
RUN_MANIFEST_SCHEMA_VERSION = "absa-ai-run-manifest/1.0.0"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_SUMS_NAME = "RUN_SHA256SUMS.txt"
ARTIFACT_TYPE = "AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION"
RETRY_PREFIX = (
    "\nLần trước không hợp lệ. Hãy trả lại toàn bộ JSON đã sửa. "
    "Lỗi validator: "
)
_ATTEMPT_NAME_RE = re.compile(r"\Aattempt-(0*[1-9][0-9]*)\.json\Z")


class RunSealError(ValueError):
    """Raised when run provenance cannot be closed without assumptions."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSealError(f"Could not read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RunSealError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RunSealError(
                        f"Expected object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSealError(f"Could not read JSONL: {path}") from exc
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise RunSealError(
            f"{context} keys mismatch; "
            f"missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunSealError(f"{context} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunSealError(f"{context} must be a non-negative integer")
    return value


def _resolve_package_path(package: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RunSealError("Artifact path must be a non-empty string")
    candidate_path = Path(relative)
    if candidate_path.is_absolute() or "\\" in relative:
        raise RunSealError(f"Artifact path is not portable: {relative!r}")
    root = package.resolve()
    candidate = (root / candidate_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RunSealError(f"Artifact escapes package root: {relative}") from exc
    return candidate


def _parse_sums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RunSealError(f"Missing checksum ledger: {path}")
    sums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line:
            continue
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RunSealError(
                f"Malformed checksum line at {path}:{line_number}"
            ) from exc
        if (
            not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or not relative
            or relative in sums
        ):
            raise RunSealError(
                f"Invalid or duplicate checksum at {path}:{line_number}"
            )
        sums[relative] = checksum
    return sums


def _verify_prepared_package(
    package: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = package / "prepare_manifest.json"
    sums_path = package / "INPUT_SHA256SUMS.txt"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "PREPARED_NOT_LABELED":
        raise RunSealError("Package is not in PREPARED_NOT_LABELED state")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RunSealError("Prepared manifest artifacts must be a list")

    sums = _parse_sums(sums_path)
    by_path: dict[str, dict[str, Any]] = {}
    expected_sums = {"prepare_manifest.json"}
    for index, artifact in enumerate(artifacts, 1):
        if not isinstance(artifact, dict):
            raise RunSealError(
                f"Prepared artifact {index} must be an object"
            )
        relative = artifact.get("path")
        if not isinstance(relative, str) or relative in by_path:
            raise RunSealError(
                f"Duplicate or invalid prepared artifact path: {relative!r}"
            )
        path = _resolve_package_path(package, relative)
        if not path.is_file():
            raise RunSealError(f"Missing prepared artifact: {relative}")
        actual_hash = sha256_file(path)
        if artifact.get("sha256") != actual_hash:
            raise RunSealError(f"Prepared artifact hash mismatch: {relative}")
        if artifact.get("bytes") != path.stat().st_size:
            raise RunSealError(f"Prepared artifact size mismatch: {relative}")
        if sums.get(relative) != actual_hash:
            raise RunSealError(
                f"Prepared checksum ledger mismatch: {relative}"
            )
        by_path[relative] = artifact
        expected_sums.add(relative)

    if sums.get("prepare_manifest.json") != sha256_file(manifest_path):
        raise RunSealError("Prepared manifest checksum mismatch")
    if set(sums) != expected_sums:
        raise RunSealError(
            "Prepared checksum closure mismatch; "
            f"missing={sorted(expected_sums - set(sums))} "
            f"extra={sorted(set(sums) - expected_sums)}"
        )
    return manifest, by_path


def _artifact_binding(
    package: Path,
    prepared_artifacts: Mapping[str, Mapping[str, Any]],
    relative: str,
) -> dict[str, Any]:
    artifact = prepared_artifacts.get(relative)
    if artifact is None:
        raise RunSealError(
            f"Prepared manifest does not bind required artifact: {relative}"
        )
    path = _resolve_package_path(package, relative)
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **(
            {"records": artifact["records"]}
            if "records" in artifact
            else {}
        ),
    }


def _verify_runner_recovery_binding(
    package: Path,
    *,
    frozen_runner_sha256: str,
    runtime_runner_sha256: str,
) -> dict[str, Any]:
    """Verify the checksummed pre-fix/fixed runner recovery bridge."""

    recovery_root = (
        package / "recovery" / "resume_attempt_collision_v1"
    )
    manifest_path = recovery_root / "manifest.json"
    sums_path = recovery_root / "SHA256SUMS.txt"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise RunSealError(
            "Runtime runner differs from frozen provenance and no verified "
            "resume-collision recovery bridge exists"
        )
    sums = _parse_sums(sums_path)
    actual_files = {
        path.relative_to(recovery_root).as_posix(): path
        for path in recovery_root.rglob("*")
        if path.is_file() and path != sums_path
    }
    if set(sums) != set(actual_files):
        raise RunSealError(
            "Resume-collision recovery checksum closure mismatch"
        )
    for relative, path in actual_files.items():
        if sums.get(relative) != sha256_file(path):
            raise RunSealError(
                f"Resume-collision recovery artifact mismatch: {relative}"
            )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version")
        != "absa-ai-resume-collision-recovery/1.0.0"
        or manifest.get("artifact_type")
        != "NON_LABEL_MUTATING_PROVENANCE_RECOVERY"
        or manifest.get("status")
        != "ARCHIVED_BEFORE_ACTIVE_REPLAY_REPAIR"
        or manifest.get("label_mutations") != 0
    ):
        raise RunSealError(
            "Resume-collision recovery manifest is invalid"
        )
    pre_fix_path = (
        recovery_root
        / "software"
        / "run_ai_annotation_tranche.pre_fix.py"
    )
    fixed_path = (
        recovery_root
        / "software"
        / "run_ai_annotation_tranche.fixed.py"
    )
    if (
        sha256_file(pre_fix_path) != frozen_runner_sha256
        or sha256_file(fixed_path) != runtime_runner_sha256
    ):
        raise RunSealError(
            "Resume-collision recovery runner bindings do not match"
        )
    return {
        "path": manifest_path.relative_to(package).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(sums_path),
        "frozen_runner_sha256": frozen_runner_sha256,
        "runtime_runner_sha256": runtime_runner_sha256,
        "label_mutations": 0,
    }


def _verify_frozen_implementation(
    package: Path,
    prepared_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    repository_root = Path(__file__).resolve().parents[1]
    required = {
        "parser": (
            "provenance/ai_tranche.py",
            repository_root / "src" / "lazada_collector" / "ai_tranche.py",
        ),
        "annotation_contract": (
            "provenance/llm_annotation.py",
            repository_root
            / "src"
            / "lazada_collector"
            / "llm_annotation.py",
        ),
        "backend_contract": (
            "provenance/llm_backends.py",
            repository_root / "src" / "lazada_collector" / "llm_backends.py",
        ),
        "runner": (
            "provenance/run_ai_annotation_tranche.py",
            repository_root / "scripts" / "run_ai_annotation_tranche.py",
        ),
        "structured_output_schema": (
            "provenance/absa_compact_batch_output_schema_v1.json",
            repository_root
            / "configs"
            / "absa_compact_batch_output_schema_v1.json",
        ),
    }
    bindings: dict[str, dict[str, Any]] = {}
    for name, (relative, runtime_path) in required.items():
        binding = _artifact_binding(
            package,
            prepared_artifacts,
            relative,
        )
        if not runtime_path.is_file():
            raise RunSealError(
                f"Runtime implementation file is missing: {runtime_path}"
            )
        runtime_hash = sha256_file(runtime_path)
        if runtime_hash != binding["sha256"]:
            if name != "runner":
                raise RunSealError(
                    "Runtime implementation differs from frozen run "
                    f"provenance: {relative}"
                )
            recovery = _verify_runner_recovery_binding(
                package,
                frozen_runner_sha256=binding["sha256"],
                runtime_runner_sha256=runtime_hash,
            )
        else:
            recovery = None
        bindings[name] = {
            **binding,
            "runtime_sha256": runtime_hash,
            **(
                {"runtime_recovery_bridge": recovery}
                if recovery is not None
                else {}
            ),
        }
    return bindings


def _target_membership_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{row['selection_rank']}\t{row['annotation_id']}\t"
        f"{row['review_text_sha256']}\n"
        for row in rows
    )
    return sha256_text(payload)


def _validate_target_rows(
    rows: Sequence[dict[str, Any]],
    *,
    scope: str,
) -> None:
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for expected_rank, row in enumerate(rows, 1):
        annotation_id = _require_nonempty_string(
            row.get("annotation_id"),
            context=f"{scope} target annotation_id",
        )
        if annotation_id in seen_ids:
            raise RunSealError(
                f"Duplicate {scope} target annotation_id: {annotation_id}"
            )
        rank = row.get("selection_rank")
        if rank != expected_rank or rank in seen_ranks:
            raise RunSealError(
                f"{scope} target selection ranks are not exactly 1..N"
            )
        review = row.get("reviewContent")
        if not isinstance(review, str) or not review.strip():
            raise RunSealError(
                f"{scope} target has empty review: {annotation_id}"
            )
        expected_hash = sha256_text(review)
        if row.get("review_text_sha256") != expected_hash:
            raise RunSealError(
                f"{scope} target review hash mismatch: {annotation_id}"
            )
        seen_ids.add(annotation_id)
        seen_ranks.add(rank)


def _load_context(
    package: Path,
    prepared_manifest: Mapping[str, Any],
    prepared_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    calibration_path = package / "calibration" / "human_confirmed.jsonl"
    split_path = package / "calibration" / "diagnostic_split.json"
    blind_path = package / "input" / "blind_reviews.jsonl"
    calibration_rows = _read_jsonl(calibration_path)
    split = _read_json(split_path)
    blind_rows = _read_jsonl(blind_path)

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    accepted_by_id: dict[str, dict[str, Any]] = {}
    for row in calibration_rows:
        calibration_id = _require_nonempty_string(
            row.get("calibration_id"),
            context="calibration_id",
        )
        if row.get("review_text_sha256") != sha256_text(
            str(row.get("reviewContent", ""))
        ):
            raise RunSealError(
                f"Calibration review hash mismatch: {calibration_id}"
            )
        decision = row.get("semantic_audit", {}).get("decision")
        if decision == "CALIBRATION_ACCEPT":
            if calibration_id in accepted_by_id:
                raise RunSealError(
                    f"Duplicate accepted calibration ID: {calibration_id}"
                )
            accepted.append(row)
            accepted_by_id[calibration_id] = row
        elif decision == "EXCLUDE_PENDING_EXPERT_ADJUDICATION":
            excluded.append(row)
        else:
            raise RunSealError(
                f"Incomplete calibration semantic audit: {calibration_id}"
            )
    if len(accepted) + len(excluded) != len(calibration_rows):
        raise RunSealError("Calibration semantic audit is not closed")

    holdout_ids = split.get("diagnostic_holdout_ids")
    prompt_ids = split.get("prompt_calibration_ids")
    if not isinstance(holdout_ids, list) or not isinstance(prompt_ids, list):
        raise RunSealError("Diagnostic split ID fields must be lists")
    if (
        len(set(holdout_ids)) != len(holdout_ids)
        or len(set(prompt_ids)) != len(prompt_ids)
    ):
        raise RunSealError("Diagnostic split contains duplicate IDs")
    holdout_set = set(holdout_ids)
    prompt_set = set(prompt_ids)
    if holdout_set.intersection(prompt_set):
        raise RunSealError("Diagnostic prompt/holdout split overlaps")
    if holdout_set | prompt_set != set(accepted_by_id):
        raise RunSealError("Diagnostic split does not close accepted calibration")

    diagnostic_targets = [
        {
            "annotation_id": calibration_id,
            "reviewContent": accepted_by_id[calibration_id]["reviewContent"],
            "review_text_sha256": accepted_by_id[calibration_id][
                "review_text_sha256"
            ],
            "selection_rank": index,
        }
        for index, calibration_id in enumerate(sorted(holdout_set), 1)
    ]
    _validate_target_rows(diagnostic_targets, scope="diagnostic")
    _validate_target_rows(blind_rows, scope="primary")
    if prepared_manifest.get("target_records") != len(blind_rows):
        raise RunSealError("Prepared target_records does not match blind input")
    if split.get("diagnostic_holdout_records") != len(diagnostic_targets):
        raise RunSealError("Diagnostic holdout count does not match split")

    guideline = prepared_manifest.get("guideline")
    calibration = prepared_manifest.get("human_calibration")
    if not isinstance(guideline, Mapping) or not isinstance(
        calibration,
        Mapping,
    ):
        raise RunSealError("Prepared prompt provenance is incomplete")
    guideline_sha = guideline.get("sha256")
    calibration_payload_sha = calibration.get(
        "calibration_payload_sha256"
    )
    if not isinstance(guideline_sha, str) or not isinstance(
        calibration_payload_sha,
        str,
    ):
        raise RunSealError("Prepared prompt hashes are incomplete")
    system_prompt = build_compact_system_prompt(
        guideline_sha256=guideline_sha,
        calibration_payload_sha256=calibration_payload_sha,
    )
    implementation_bindings = _verify_frozen_implementation(
        package,
        prepared_artifacts,
    )

    return {
        "targets": {
            "diagnostic": diagnostic_targets,
            "primary": blind_rows,
        },
        "examples": {
            "diagnostic": [
                accepted_by_id[item] for item in sorted(prompt_set)
            ],
            "primary": accepted,
        },
        "expected_holdout": [
            accepted_by_id[item] for item in sorted(holdout_set)
        ],
        "system_prompt": system_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "input_bindings": {
            "prepare_manifest": {
                "path": "prepare_manifest.json",
                "sha256": sha256_file(package / "prepare_manifest.json"),
            },
            "calibration": _artifact_binding(
                package,
                prepared_artifacts,
                "calibration/human_confirmed.jsonl",
            ),
            "diagnostic_split": _artifact_binding(
                package,
                prepared_artifacts,
                "calibration/diagnostic_split.json",
            ),
            "primary_targets": _artifact_binding(
                package,
                prepared_artifacts,
                "input/blind_reviews.jsonl",
            ),
            "implementation": implementation_bindings,
        },
        "prepared_artifacts": prepared_artifacts,
    }


def _validate_summary_shape(
    summary: Mapping[str, Any],
    *,
    scope: str,
    target_count: int,
    system_prompt_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
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
    if scope == "diagnostic":
        expected_keys.add("diagnostic_gate")
    _require_exact_keys(summary, expected_keys, context=f"{scope} run summary")
    if (
        summary.get("artifact_type") != "ABSA_AI_PREANNOTATION_RUN_SUMMARY"
        or summary.get("scope") != scope
        or summary.get("status") != "COMPLETED"
        or summary.get("selected_records") != target_count
        or summary.get("valid_records") != target_count
        or summary.get("missing_annotation_ids") != []
    ):
        raise RunSealError(f"{scope} run summary is not complete")
    _require_nonempty_string(
        summary.get("started_at"),
        context=f"{scope} summary started_at",
    )
    _require_nonempty_string(
        summary.get("completed_at"),
        context=f"{scope} summary completed_at",
    )
    _require_nonnegative_int(
        summary.get("provider_calls_this_invocation"),
        context=f"{scope} provider calls",
    )
    usage = summary.get("provider_usage_this_invocation")
    if not isinstance(usage, Mapping):
        raise RunSealError(f"{scope} provider usage must be an object")
    _require_exact_keys(
        usage,
        {"prompt_tokens", "completion_tokens", "total_tokens"},
        context=f"{scope} provider usage",
    )
    for key, value in usage.items():
        _require_nonnegative_int(
            value,
            context=f"{scope} provider usage {key}",
        )

    prompt = summary.get("prompt")
    if not isinstance(prompt, Mapping):
        raise RunSealError(f"{scope} prompt config must be an object")
    _require_exact_keys(
        prompt,
        {"version", "system_prompt_sha256"},
        context=f"{scope} prompt config",
    )
    if (
        prompt.get("version") != COMPACT_PROMPT_VERSION
        or prompt.get("system_prompt_sha256") != system_prompt_sha256
    ):
        raise RunSealError(f"{scope} frozen prompt mismatch")

    provider = summary.get("provider")
    if not isinstance(provider, Mapping):
        raise RunSealError(f"{scope} provider config must be an object")
    _require_exact_keys(
        provider,
        {
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
        },
        context=f"{scope} provider config",
    )
    backend = provider.get("backend")
    if backend not in {"nvidia", "codex"}:
        raise RunSealError(f"Unsupported {scope} backend: {backend!r}")
    _require_nonempty_string(
        provider.get("endpoint"),
        context=f"{scope} provider endpoint",
    )
    _require_nonempty_string(
        provider.get("model"),
        context=f"{scope} provider model",
    )
    _require_nonempty_string(
        provider.get("api_key_env"),
        context=f"{scope} api_key_env",
    )
    for field in ("batch_size", "workers", "max_retries", "max_tokens"):
        value = provider.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RunSealError(
                f"{scope} provider {field} must be a positive integer"
            )
    timeout = provider.get("timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise RunSealError(
            f"{scope} provider timeout_seconds must be positive"
        )
    if provider.get("reasoning_effort") not in {
        None,
        "low",
        "medium",
        "high",
    }:
        raise RunSealError(f"{scope} reasoning_effort is invalid")
    thinking = provider.get("enable_thinking")
    if thinking is not None and not isinstance(thinking, bool):
        raise RunSealError(f"{scope} enable_thinking is invalid")
    codex_schema = provider.get("codex_schema")
    if backend == "nvidia":
        if codex_schema is not None:
            raise RunSealError("NVIDIA run unexpectedly declares Codex schema")
    else:
        if not isinstance(codex_schema, Mapping):
            raise RunSealError("Codex run has no output-schema binding")
        _require_exact_keys(
            codex_schema,
            {"path", "sha256"},
            context=f"{scope} Codex schema",
        )
        _require_nonempty_string(
            codex_schema.get("path"),
            context=f"{scope} Codex schema path",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", str(codex_schema.get("sha256"))):
            raise RunSealError(f"{scope} Codex schema hash is invalid")
    return dict(provider)


def _expected_provider_core(
    provider: Mapping[str, Any],
) -> dict[str, Any]:
    codex_schema = provider.get("codex_schema")
    return {
        "backend": provider["backend"],
        "endpoint": provider["endpoint"],
        "requested_model": provider["model"],
        "requested_reasoning_effort": provider["reasoning_effort"],
        "requested_enable_thinking": provider["enable_thinking"],
        "codex_schema_sha256": (
            codex_schema["sha256"]
            if isinstance(codex_schema, Mapping)
            else None
        ),
    }


def _validate_usage(value: Any, *, context: str) -> None:
    if not isinstance(value, Mapping):
        raise RunSealError(f"{context} usage must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or item < 0
        ):
            raise RunSealError(f"{context} usage is invalid")


def _run_input_paths(run_root: Path) -> list[Path]:
    excluded = {
        (run_root / RUN_MANIFEST_NAME).resolve(),
        (run_root / RUN_SUMS_NAME).resolve(),
    }
    return sorted(
        (
            path
            for path in run_root.rglob("*")
            if path.is_file() and path.resolve() not in excluded
        ),
        key=lambda path: path.relative_to(run_root).as_posix(),
    )


def _classify_run_paths(
    run_root: Path,
) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {
        "run_summary": [],
        "records": [],
        "attempts": [],
        "failures": [],
        "codex_invocations": [],
    }
    for path in _run_input_paths(run_root):
        relative = path.relative_to(run_root)
        parts = relative.parts
        if relative.as_posix() == "run_summary.json":
            groups["run_summary"].append(path)
        elif len(parts) == 2 and parts[0] == "records":
            if path.suffix != ".json":
                raise RunSealError(f"Unexpected record artifact: {relative}")
            groups["records"].append(path)
        elif len(parts) == 3 and parts[0] == "attempts":
            if _ATTEMPT_NAME_RE.fullmatch(parts[2]) is None:
                raise RunSealError(f"Unexpected attempt artifact: {relative}")
            groups["attempts"].append(path)
        elif len(parts) == 2 and parts[0] == "failed":
            if path.suffix != ".json":
                raise RunSealError(f"Unexpected failure artifact: {relative}")
            groups["failures"].append(path)
        elif (
            len(parts) == 4
            and parts[0] == "codex_invocations"
            and _ATTEMPT_NAME_RE.fullmatch(parts[2] + ".json") is not None
            and parts[3] == "last_message.json"
        ):
            groups["codex_invocations"].append(path)
        else:
            raise RunSealError(f"Unrecognized run artifact: {relative}")
    if len(groups["run_summary"]) != 1:
        raise RunSealError(
            f"Expected exactly one run_summary.json in {run_root}"
        )
    return groups


def _attempt_number_from_path(path: Path) -> int:
    match = _ATTEMPT_NAME_RE.fullmatch(path.name)
    if match is None:
        raise RunSealError(f"Invalid attempt filename: {path}")
    number = int(match.group(1))
    if path.name != f"attempt-{number:02d}.json":
        raise RunSealError(f"Non-canonical attempt filename: {path}")
    return number


def _expected_examples(
    target_rows: Sequence[dict[str, Any]],
    examples: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    selected = retrieve_examples(
        target_rows,
        examples,
        per_target=3,
        maximum=14,
    )
    return selected, [row["calibration_id"] for row in selected]


def _validate_attempts(
    paths: Sequence[Path],
    *,
    run_root: Path,
    scope: str,
    targets: Sequence[dict[str, Any]],
    examples: Sequence[dict[str, Any]],
    provider: Mapping[str, Any],
    system_prompt_sha256: str,
) -> dict[str, Any]:
    target_by_id = {row["annotation_id"]: row for row in targets}
    target_order = {
        row["annotation_id"]: row["selection_rank"] for row in targets
    }
    grouped: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = {}
    for path in paths:
        attempt = _read_json(path)
        _require_exact_keys(
            attempt,
            {
                "schema_version",
                "scope",
                "batch_id",
                "attempt",
                "target_ids",
                "started_at",
                "completed_at",
                "outcome",
                "system_prompt_sha256",
                "user_message_sha256",
                "provider",
                "response_content",
                "error",
            },
            context=f"attempt {path}",
        )
        if (
            attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION
            or attempt.get("scope") != scope
        ):
            raise RunSealError(f"Attempt schema/scope mismatch: {path}")
        batch_id = _require_nonempty_string(
            attempt.get("batch_id"),
            context=f"attempt batch_id {path}",
        )
        if path.parent.name != batch_id:
            raise RunSealError(f"Attempt batch/path mismatch: {path}")
        number = _attempt_number_from_path(path)
        if attempt.get("attempt") != number:
            raise RunSealError(f"Attempt number/path mismatch: {path}")
        if number in grouped.setdefault(batch_id, {}):
            raise RunSealError(
                f"Duplicate attempt number {number} for {batch_id}"
            )
        grouped[batch_id][number] = (path, attempt)

    provider_core = _expected_provider_core(provider)
    claims: dict[
        tuple[str, int, str],
        dict[str, Any],
    ] = {}
    claim_by_target: dict[str, tuple[str, int, str]] = {}
    attempt_inventory: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    response_paths: dict[tuple[str, int], str] = {}

    for batch_id in sorted(grouped):
        numbered = grouped[batch_id]
        numbers = sorted(numbered)
        if numbers != list(range(1, len(numbers) + 1)):
            raise RunSealError(
                f"Attempt sequence has gaps for batch {batch_id}: {numbers}"
            )
        previous_target_ids: list[str] | None = None
        previous_next_ids: list[str] | None = None
        previous_error = ""
        previous_terminal = False
        for number in numbers:
            path, attempt = numbered[number]
            if previous_terminal:
                raise RunSealError(
                    f"Attempt exists after terminal attempt: {path}"
                )
            target_ids = attempt.get("target_ids")
            if (
                not isinstance(target_ids, list)
                or not target_ids
                or any(not isinstance(item, str) for item in target_ids)
                or len(set(target_ids)) != len(target_ids)
            ):
                raise RunSealError(f"Invalid target_ids in attempt: {path}")
            unknown = set(target_ids) - set(target_by_id)
            if unknown:
                raise RunSealError(
                    f"Attempt references unknown targets: {sorted(unknown)}"
                )
            if target_ids != sorted(
                target_ids,
                key=lambda item: target_order[item],
            ):
                raise RunSealError(
                    f"Attempt target order does not match frozen input: {path}"
                )
            if previous_target_ids is not None and (
                target_ids != previous_next_ids
            ):
                raise RunSealError(
                    f"Attempt retry target set mismatch: {path}"
                )
            rows = [target_by_id[item] for item in target_ids]
            selected_examples, example_ids = _expected_examples(rows, examples)
            user_message = build_compact_user_message(rows, selected_examples)
            attempt_message = (
                user_message
                if not previous_error
                else user_message + RETRY_PREFIX + previous_error[:800]
            )
            if attempt.get("user_message_sha256") != sha256_text(
                attempt_message
            ):
                raise RunSealError(
                    f"Attempt user-message hash mismatch: {path}"
                )
            if attempt.get("system_prompt_sha256") != system_prompt_sha256:
                raise RunSealError(
                    f"Attempt system-prompt hash mismatch: {path}"
                )
            _require_nonempty_string(
                attempt.get("started_at"),
                context=f"attempt started_at {path}",
            )
            _require_nonempty_string(
                attempt.get("completed_at"),
                context=f"attempt completed_at {path}",
            )

            attempt_provider = attempt.get("provider")
            if not isinstance(attempt_provider, Mapping):
                raise RunSealError(f"Attempt provider must be object: {path}")
            outcome = attempt.get("outcome")
            if outcome not in {
                "VALID",
                "PARTIAL_VALID",
                "SCHEMA_INVALID",
                "BACKEND_ERROR",
                "UNEXPECTED_ERROR",
            }:
                raise RunSealError(
                    f"Unsupported attempt outcome {outcome!r}: {path}"
                )
            expected_provider_keys = set(provider_core)
            if outcome in {"VALID", "PARTIAL_VALID", "SCHEMA_INVALID"}:
                expected_provider_keys.update(
                    {
                        "provider_model",
                        "finish_reason",
                        "provider_request_id",
                        "usage",
                    }
                )
            elif outcome == "BACKEND_ERROR":
                expected_provider_keys.update({"status_code", "retryable"})
            _require_exact_keys(
                attempt_provider,
                expected_provider_keys,
                context=f"attempt provider {path}",
            )
            for key, expected in provider_core.items():
                if attempt_provider.get(key) != expected:
                    raise RunSealError(
                        f"Attempt provider config mismatch ({key}): {path}"
                    )

            content = attempt.get("response_content")
            stored_error = attempt.get("error")
            valid: list[dict[str, Any]] = []
            row_errors: dict[str, str] = {}
            replay_outcome: str
            if outcome in {"VALID", "PARTIAL_VALID", "SCHEMA_INVALID"}:
                if not isinstance(content, str) or not content.strip():
                    raise RunSealError(
                        f"Response-bearing attempt has no response: {path}"
                    )
                if attempt_provider.get("provider_model") != provider["model"]:
                    raise RunSealError(
                        f"Provider response model mismatch: {path}"
                    )
                if (
                    attempt_provider.get("finish_reason") is not None
                    and not isinstance(
                        attempt_provider.get("finish_reason"),
                        str,
                    )
                ):
                    raise RunSealError(
                        f"Provider finish_reason is invalid: {path}"
                    )
                if (
                    attempt_provider.get("provider_request_id") is not None
                    and not isinstance(
                        attempt_provider.get("provider_request_id"),
                        str,
                    )
                ):
                    raise RunSealError(
                        f"Provider request ID is invalid: {path}"
                    )
                _validate_usage(
                    attempt_provider.get("usage"),
                    context=f"attempt {path}",
                )
                try:
                    valid, row_errors = parse_compact_batch_partial(
                        content,
                        expected_rows=rows,
                    )
                except AnnotationValidationError as exc:
                    replay_outcome = "SCHEMA_INVALID"
                    expected_error = f"{type(exc).__name__}: {exc}"
                else:
                    replay_outcome = (
                        "PARTIAL_VALID" if row_errors else "VALID"
                    )
                    expected_error = (
                        canonical_json(row_errors) if row_errors else None
                    )
                if outcome != replay_outcome:
                    raise RunSealError(
                        f"Stored/replayed outcome mismatch at {path}; "
                        f"stored={outcome} replayed={replay_outcome}"
                    )
                if stored_error != expected_error:
                    raise RunSealError(
                        f"Stored/replayed validation error mismatch: {path}"
                    )
                response_paths[(batch_id, number)] = content
            elif outcome == "BACKEND_ERROR":
                status_code = attempt_provider.get("status_code")
                if (
                    content is not None
                    or not isinstance(stored_error, str)
                    or not stored_error
                    or not isinstance(
                        attempt_provider.get("retryable"),
                        bool,
                    )
                    or (
                        status_code is not None
                        and (
                            isinstance(status_code, bool)
                            or not isinstance(status_code, int)
                            or status_code < 100
                            or status_code > 599
                        )
                    )
                ):
                    raise RunSealError(
                        f"Malformed backend-error attempt: {path}"
                    )
                replay_outcome = outcome
                expected_error = stored_error
            else:
                if (
                    content is not None
                    or not isinstance(stored_error, str)
                    or not stored_error
                ):
                    raise RunSealError(
                        f"Malformed unexpected-error attempt: {path}"
                    )
                replay_outcome = outcome
                expected_error = stored_error

            for normalized in valid:
                annotation_id = normalized["annotation_id"]
                key = (batch_id, number, annotation_id)
                if annotation_id in claim_by_target:
                    raise RunSealError(
                        "A target validates in more than one attempt: "
                        f"{annotation_id}"
                    )
                claims[key] = {
                    "normalized": normalized,
                    "human_example_ids": example_ids,
                    "attempt_path": path,
                }
                claim_by_target[annotation_id] = key

            outcome_counts[outcome] += 1
            attempt_inventory.append(
                {
                    "batch_id": batch_id,
                    "attempt": number,
                    "path": path,
                    "outcome": outcome,
                    "target_ids": list(target_ids),
                }
            )
            previous_target_ids = list(target_ids)
            if replay_outcome == "PARTIAL_VALID":
                previous_next_ids = [
                    item for item in target_ids if item in row_errors
                ]
            else:
                previous_next_ids = list(target_ids)
            previous_error = "" if stored_error is None else str(stored_error)
            previous_terminal = replay_outcome == "VALID" or (
                replay_outcome == "BACKEND_ERROR"
                and attempt_provider.get("retryable") is False
            )

    return {
        "claims": claims,
        "claim_by_target": claim_by_target,
        "attempt_inventory": attempt_inventory,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "response_paths": response_paths,
    }


def _validate_records(
    paths: Sequence[Path],
    *,
    run_root: Path,
    scope: str,
    targets: Sequence[dict[str, Any]],
    provider: Mapping[str, Any],
    system_prompt_sha256: str,
    attempt_validation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_paths = {
        (
            run_root
            / "records"
            / f"{row['selection_rank']:05d}-{row['annotation_id']}.json"
        ): row
        for row in targets
    }
    if set(paths) != set(expected_paths):
        raise RunSealError(
            f"{scope} record inventory mismatch; "
            f"missing={sorted(str(path) for path in set(expected_paths)-set(paths))} "
            f"extra={sorted(str(path) for path in set(paths)-set(expected_paths))}"
        )
    claims = attempt_validation["claims"]
    used_claims: set[tuple[str, int, str]] = set()
    records: list[dict[str, Any]] = []
    expected_schema_sha = (
        provider["codex_schema"]["sha256"]
        if isinstance(provider.get("codex_schema"), Mapping)
        else None
    )
    expected_generation_keys = {
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
    for path, target in sorted(
        expected_paths.items(),
        key=lambda item: item[1]["selection_rank"],
    ):
        record = _read_json(path)
        _require_exact_keys(
            record,
            {
                "schema_version",
                "artifact_type",
                "scope",
                "annotation_id",
                "selection_rank",
                "review_text_sha256",
                "annotation",
                "normalization_repairs",
                "generation",
            },
            context=f"run record {path}",
        )
        if (
            record.get("schema_version") != RUN_RECORD_SCHEMA_VERSION
            or record.get("artifact_type") != ARTIFACT_TYPE
            or record.get("scope") != scope
            or record.get("annotation_id") != target["annotation_id"]
            or record.get("selection_rank") != target["selection_rank"]
            or record.get("review_text_sha256")
            != target["review_text_sha256"]
        ):
            raise RunSealError(f"Run record/source binding mismatch: {path}")
        generation = record.get("generation")
        if not isinstance(generation, Mapping):
            raise RunSealError(f"Run record generation is not an object: {path}")
        _require_exact_keys(
            generation,
            expected_generation_keys,
            context=f"run record generation {path}",
        )
        if (
            generation.get("backend") != provider["backend"]
            or generation.get("model") != provider["model"]
            or generation.get("prompt_version") != COMPACT_PROMPT_VERSION
            or generation.get("system_prompt_sha256")
            != system_prompt_sha256
            or generation.get("reasoning_effort")
            != provider["reasoning_effort"]
            or generation.get("enable_thinking")
            != provider["enable_thinking"]
            or generation.get("codex_schema_sha256") != expected_schema_sha
        ):
            raise RunSealError(
                f"Run record generation config mismatch: {path}"
            )
        batch_id = generation.get("batch_id")
        attempt_number = generation.get("valid_attempt")
        if (
            not isinstance(batch_id, str)
            or isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number <= 0
        ):
            raise RunSealError(
                f"Run record has invalid attempt binding: {path}"
            )
        claim_key = (batch_id, attempt_number, target["annotation_id"])
        claim = claims.get(claim_key)
        if claim is None:
            raise RunSealError(
                f"Run record has no replay-valid attempt: {path}"
            )
        normalized = claim["normalized"]
        if canonical_json(record.get("annotation")) != canonical_json(
            normalized["annotation"]
        ):
            raise RunSealError(
                f"Run record annotation differs from replay: {path}"
            )
        if canonical_json(record.get("normalization_repairs")) != canonical_json(
            normalized.get("normalization_repairs", [])
        ):
            raise RunSealError(
                f"Run record normalization repairs differ from replay: {path}"
            )
        if generation.get("human_example_ids") != claim["human_example_ids"]:
            raise RunSealError(
                f"Run record human-example binding mismatch: {path}"
            )
        used_claims.add(claim_key)
        records.append(record)
    if used_claims != set(claims):
        unused = sorted(set(claims) - used_claims)
        raise RunSealError(
            f"Replay-valid attempt rows are not bound to records: {unused}"
        )
    return records


def _validate_failures(
    paths: Sequence[Path],
    *,
    scope: str,
    target_by_id: Mapping[str, Mapping[str, Any]],
    record_ids: set[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        failure = _read_json(path)
        _require_exact_keys(
            failure,
            {
                "schema_version",
                "scope",
                "annotation_id",
                "review_text_sha256",
                "failed_at",
                "last_error",
                "batch_id",
                "backend_fatal",
            },
            context=f"failure marker {path}",
        )
        annotation_id = failure.get("annotation_id")
        target = target_by_id.get(str(annotation_id))
        if (
            failure.get("schema_version") != FAILURE_SCHEMA_VERSION
            or failure.get("scope") != scope
            or target is None
            or failure.get("review_text_sha256")
            != target["review_text_sha256"]
            or path.stem != annotation_id
            or annotation_id in seen
            or not isinstance(failure.get("backend_fatal"), bool)
            or not isinstance(failure.get("last_error"), str)
            or not failure.get("last_error")
            or not isinstance(failure.get("batch_id"), str)
            or not failure.get("batch_id")
        ):
            raise RunSealError(f"Invalid failure marker: {path}")
        _require_nonempty_string(
            failure.get("failed_at"),
            context=f"failure timestamp {path}",
        )
        if annotation_id not in record_ids:
            raise RunSealError(
                f"Unrecovered terminal failure: {annotation_id}"
            )
        seen.add(annotation_id)
        failures.append(failure)
    return failures


def _validate_codex_invocations(
    paths: Sequence[Path],
    *,
    run_root: Path,
    backend: str,
    response_paths: Mapping[tuple[str, int], str],
) -> None:
    if backend != "codex":
        if paths:
            raise RunSealError(
                "Non-Codex run contains Codex invocation artifacts"
            )
        return
    expected: dict[Path, str] = {
        (
            run_root
            / "codex_invocations"
            / batch_id
            / f"attempt-{attempt:02d}"
            / "last_message.json"
        ): content
        for (batch_id, attempt), content in response_paths.items()
    }
    if set(paths) != set(expected):
        raise RunSealError(
            "Codex invocation-output inventory does not match responses"
        )
    for path, content in expected.items():
        if path.read_text(encoding="utf-8") != content:
            raise RunSealError(
                f"Codex invocation output differs from attempt: {path}"
            )


def _artifact_entry(
    package: Path,
    path: Path,
    *,
    role: str,
) -> dict[str, Any]:
    return {
        "path": path.relative_to(package).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _artifact_set_sha(
    artifacts: Iterable[Mapping[str, Any]],
    *,
    role: str,
) -> str:
    payload = "".join(
        f"{item['path']}\t{item['bytes']}\t{item['sha256']}\n"
        for item in sorted(
            (item for item in artifacts if item["role"] == role),
            key=lambda item: item["path"],
        )
    )
    return sha256_text(payload)


def _validate_scope(
    package: Path,
    *,
    scope: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = package / "runs" / scope
    if not run_root.is_dir():
        raise RunSealError(f"Missing run directory: {run_root}")
    groups = _classify_run_paths(run_root)
    summary_path = groups["run_summary"][0]
    summary = _read_json(summary_path)
    targets = context["targets"][scope]
    provider = _validate_summary_shape(
        summary,
        scope=scope,
        target_count=len(targets),
        system_prompt_sha256=context["system_prompt_sha256"],
    )

    if provider["backend"] == "codex":
        frozen_schema = context["prepared_artifacts"].get(
            "provenance/absa_compact_batch_output_schema_v1.json"
        )
        if frozen_schema is None:
            raise RunSealError(
                "Codex run lacks frozen output-schema provenance"
            )
        if provider["codex_schema"]["sha256"] != frozen_schema["sha256"]:
            raise RunSealError(
                f"{scope} Codex schema differs from frozen package schema"
            )

    attempts = _validate_attempts(
        groups["attempts"],
        run_root=run_root,
        scope=scope,
        targets=targets,
        examples=context["examples"][scope],
        provider=provider,
        system_prompt_sha256=context["system_prompt_sha256"],
    )
    records = _validate_records(
        groups["records"],
        run_root=run_root,
        scope=scope,
        targets=targets,
        provider=provider,
        system_prompt_sha256=context["system_prompt_sha256"],
        attempt_validation=attempts,
    )
    record_ids = {record["annotation_id"] for record in records}
    if record_ids != {row["annotation_id"] for row in targets}:
        raise RunSealError(f"{scope} has unrecovered target records")
    failures = _validate_failures(
        groups["failures"],
        scope=scope,
        target_by_id={row["annotation_id"]: row for row in targets},
        record_ids=record_ids,
    )
    _validate_codex_invocations(
        groups["codex_invocations"],
        run_root=run_root,
        backend=provider["backend"],
        response_paths=attempts["response_paths"],
    )

    expected_label_summary = count_labels(records)
    if canonical_json(summary.get("label_summary")) != canonical_json(
        expected_label_summary
    ):
        raise RunSealError(f"{scope} run-summary label counts mismatch")

    artifacts: list[dict[str, Any]] = []
    role_by_group = {
        "run_summary": "run_summary",
        "records": "record",
        "attempts": "attempt",
        "failures": "recovered_failure_marker",
        "codex_invocations": "codex_invocation_output",
    }
    for group, paths in groups.items():
        for path in paths:
            artifacts.append(
                _artifact_entry(
                    package,
                    path,
                    role=role_by_group[group],
                )
            )

    diagnostic: dict[str, Any] | None = None
    if scope == "diagnostic":
        metrics_path = package / "calibration" / "diagnostic_metrics.json"
        gate_path = package / "calibration" / "diagnostic_gate.json"
        metrics = _read_json(metrics_path)
        gate = _read_json(gate_path)
        predictions = [
            {
                "annotation_id": record["annotation_id"],
                "annotation": record["annotation"],
            }
            for record in records
        ]
        expected_metrics = calibration_metrics(
            context["expected_holdout"],
            predictions,
        )
        expected_gate = calibration_gate(expected_metrics)
        if canonical_json(metrics) != canonical_json(expected_metrics):
            raise RunSealError("Diagnostic metrics differ from replay")
        if canonical_json(gate) != canonical_json(expected_gate):
            raise RunSealError("Diagnostic gate differs from replay")
        if canonical_json(summary.get("diagnostic_gate")) != canonical_json(
            gate
        ):
            raise RunSealError("Diagnostic summary gate mismatch")
        artifacts.extend(
            (
                _artifact_entry(
                    package,
                    metrics_path,
                    role="diagnostic_metrics",
                ),
                _artifact_entry(
                    package,
                    gate_path,
                    role="diagnostic_gate",
                ),
            )
        )
        diagnostic = {
            "metrics_path": "calibration/diagnostic_metrics.json",
            "metrics_sha256": sha256_file(metrics_path),
            "gate_path": "calibration/diagnostic_gate.json",
            "gate_sha256": sha256_file(gate_path),
            "gate_status": gate.get("status"),
        }

    artifacts.sort(key=lambda item: item["path"])
    role_counts = Counter(item["role"] for item in artifacts)
    artifact_hash_by_path = {
        item["path"]: item["sha256"] for item in artifacts
    }
    binding_payload_parts: list[str] = []
    for record in records:
        generation = record["generation"]
        record_relative = (
            run_root
            / "records"
            / (
                f"{record['selection_rank']:05d}-"
                f"{record['annotation_id']}.json"
            )
        ).relative_to(package).as_posix()
        attempt_relative = (
            run_root
            / "attempts"
            / generation["batch_id"]
            / f"attempt-{generation['valid_attempt']:02d}.json"
        ).relative_to(package).as_posix()
        binding_payload_parts.append(
            f"{record['selection_rank']}\t{record['annotation_id']}\t"
            f"{generation['batch_id']}\t{generation['valid_attempt']}\t"
            f"{artifact_hash_by_path[record_relative]}\t"
            f"{artifact_hash_by_path[attempt_relative]}\n"
        )
    scope_input_bindings = dict(context["input_bindings"])
    if scope == "primary":
        diagnostic_gate_path = (
            package / "calibration" / "diagnostic_gate.json"
        )
        if not diagnostic_gate_path.is_file():
            raise RunSealError("Primary run has no diagnostic gate artifact")
        scope_input_bindings["upstream_diagnostic_gate"] = {
            "path": "calibration/diagnostic_gate.json",
            "bytes": diagnostic_gate_path.stat().st_size,
            "sha256": sha256_file(diagnostic_gate_path),
        }
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "ABSA_AI_RUN_PROVENANCE_SEAL",
        "status": "SEALED_REPLAY_VALID",
        "scope": scope,
        "sealed_at": None,
        "path_basis": "package_root",
        "tranche_id": context["prepared_manifest"].get("tranche_id"),
        "input_bindings": scope_input_bindings,
        "target_source": (
            context["input_bindings"]["diagnostic_split"]
            if scope == "diagnostic"
            else context["input_bindings"]["primary_targets"]
        ),
        "target_records": len(targets),
        "target_membership_sha256": _target_membership_sha(targets),
        "provider": provider,
        "prompt": {
            "version": COMPACT_PROMPT_VERSION,
            "system_prompt_sha256": context["system_prompt_sha256"],
        },
        "sealing_implementation": {
            "path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[1]
            ).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        },
        "replay": {
            "parser": "parse_compact_batch_partial",
            "attempts": len(attempts["attempt_inventory"]),
            "attempt_outcomes": attempts["outcome_counts"],
            "records_bound_to_unique_valid_attempts": len(records),
            "record_attempt_binding_sha256": sha256_text(
                "".join(binding_payload_parts)
            ),
            "record_attempt_binding_serialization": (
                "selection_rank<TAB>annotation_id<TAB>batch_id<TAB>"
                "valid_attempt<TAB>record_sha256<TAB>attempt_sha256<LF>"
            ),
            "normalization_repairs": sum(
                len(record["normalization_repairs"]) for record in records
            ),
            "recovered_failure_markers": len(failures),
            "unrecovered_failures": 0,
        },
        "inventory": {
            "artifact_files": len(artifacts),
            "role_counts": dict(sorted(role_counts.items())),
            "record_artifact_set_sha256": _artifact_set_sha(
                artifacts,
                role="record",
            ),
            "attempt_artifact_set_sha256": _artifact_set_sha(
                artifacts,
                role="attempt",
            ),
        },
        "diagnostic": diagnostic,
        "artifacts": artifacts,
        "limitations": [
            "This seal proves replay and provenance consistency; it does not "
            "turn AI pseudo-labels into human gold.",
            "The diagnostic holdout is an alignment gate, not an independent "
            "accuracy or inter-annotator-agreement estimate.",
        ],
    }
    return {
        "scope": scope,
        "run_root": run_root,
        "manifest": manifest,
        "input_snapshot": {
            item["path"]: (item["bytes"], item["sha256"])
            for item in artifacts
        },
    }


def _current_artifact_snapshot(
    package: Path,
    validation: Mapping[str, Any],
) -> dict[str, tuple[int, str]]:
    snapshot: dict[str, tuple[int, str]] = {}
    for artifact in validation["manifest"]["artifacts"]:
        path = _resolve_package_path(package, artifact["path"])
        if not path.is_file():
            raise RunSealError(
                f"Artifact disappeared during sealing: {artifact['path']}"
            )
        snapshot[artifact["path"]] = (path.stat().st_size, sha256_file(path))
    return snapshot


def _verify_existing_seal(package: Path, *, scope: str) -> None:
    run_root = package / "runs" / scope
    manifest_path = run_root / RUN_MANIFEST_NAME
    sums_path = run_root / RUN_SUMS_NAME
    if not manifest_path.exists() and not sums_path.exists():
        return
    if not manifest_path.is_file() or not sums_path.is_file():
        raise RunSealError(f"Incomplete existing {scope} seal")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("scope") != scope
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise RunSealError(f"Invalid existing {scope} run manifest")
    sums = _parse_sums(sums_path)
    expected: dict[str, str] = {}
    expected_sizes: dict[str, int] = {}
    for artifact in manifest["artifacts"]:
        if not isinstance(artifact, Mapping):
            raise RunSealError(
                f"Invalid artifact in existing {scope} manifest"
            )
        _require_exact_keys(
            artifact,
            {"path", "role", "bytes", "sha256"},
            context=f"existing {scope} manifest artifact",
        )
        relative = artifact.get("path")
        checksum = artifact.get("sha256")
        size = artifact.get("bytes")
        if (
            not isinstance(relative, str)
            or relative in expected
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise RunSealError(
                f"Invalid artifact entry in existing {scope} manifest"
            )
        expected[relative] = checksum
        expected_sizes[relative] = size
    manifest_relative = manifest_path.relative_to(package).as_posix()
    expected[manifest_relative] = sha256_file(manifest_path)
    if (
        None in expected
        or set(sums) != set(expected)
        or any(sums[path] != checksum for path, checksum in expected.items())
    ):
        raise RunSealError(f"Existing {scope} seal checksum closure mismatch")
    for relative, checksum in expected.items():
        path = _resolve_package_path(package, str(relative))
        if not path.is_file() or sha256_file(path) != checksum:
            raise RunSealError(
                f"Post-seal artifact mutation detected: {relative}"
            )
        if (
            relative in expected_sizes
            and path.stat().st_size != expected_sizes[relative]
        ):
            raise RunSealError(
                f"Post-seal artifact size mutation detected: {relative}"
            )
    current_run_paths = {
        path.relative_to(package).as_posix()
        for path in _run_input_paths(run_root)
    }
    sealed_run_paths = {
        str(artifact["path"])
        for artifact in manifest["artifacts"]
        if str(artifact.get("path", "")).startswith(
            f"runs/{scope}/"
        )
    }
    if current_run_paths != sealed_run_paths:
        raise RunSealError(
            f"Existing {scope} seal inventory closure mismatch"
        )


def _write_seal(
    package: Path,
    validation: Mapping[str, Any],
    *,
    sealed_at: str,
) -> dict[str, Any]:
    manifest = dict(validation["manifest"])
    manifest["sealed_at"] = sealed_at
    run_root = validation["run_root"]
    manifest_path = run_root / RUN_MANIFEST_NAME
    sums_path = run_root / RUN_SUMS_NAME
    _atomic_write_json(manifest_path, manifest)
    sums_entries = {
        artifact["path"]: artifact["sha256"]
        for artifact in manifest["artifacts"]
    }
    sums_entries[manifest_path.relative_to(package).as_posix()] = sha256_file(
        manifest_path
    )
    _atomic_write_text(
        sums_path,
        "".join(
            f"{checksum}  {relative}\n"
            for relative, checksum in sorted(sums_entries.items())
        ),
    )
    return manifest


def seal_runs(
    package: Path,
    *,
    scopes: Sequence[str] = ("diagnostic", "primary"),
) -> dict[str, Any]:
    """Validate and seal one or both completed scopes.

    Validation of every requested scope completes before any manifest is
    written.  Existing manifests are verified first, and the artifact snapshot
    is checked both immediately before and after the atomic writes.
    """

    package = package.resolve()
    normalized_scopes = tuple(scopes)
    if (
        not normalized_scopes
        or len(set(normalized_scopes)) != len(normalized_scopes)
        or any(scope not in {"diagnostic", "primary"} for scope in scopes)
    ):
        raise RunSealError(f"Invalid seal scopes: {list(scopes)}")
    for scope in normalized_scopes:
        _verify_existing_seal(package, scope=scope)
    if "primary" in normalized_scopes and "diagnostic" not in normalized_scopes:
        _verify_existing_seal(package, scope="diagnostic")
        diagnostic_manifest = _read_json(
            package / "runs" / "diagnostic" / RUN_MANIFEST_NAME
        )
        diagnostic = diagnostic_manifest.get("diagnostic")
        if (
            diagnostic_manifest.get("status") != "SEALED_REPLAY_VALID"
            or not isinstance(diagnostic, Mapping)
            or diagnostic.get("gate_status") != "PASS"
        ):
            raise RunSealError(
                "Primary-only sealing requires a valid PASS diagnostic seal"
            )
    prepared_manifest, prepared_artifacts = _verify_prepared_package(package)
    context = _load_context(
        package,
        prepared_manifest,
        prepared_artifacts,
    )
    context["prepared_manifest"] = prepared_manifest
    validations = [
        _validate_scope(package, scope=scope, context=context)
        for scope in normalized_scopes
    ]
    if "primary" in normalized_scopes:
        gate = _read_json(
            package / "calibration" / "diagnostic_gate.json"
        )
        if gate.get("status") != "PASS":
            raise RunSealError("Primary run cannot be sealed after a failed gate")

    for validation in validations:
        if _current_artifact_snapshot(package, validation) != validation[
            "input_snapshot"
        ]:
            raise RunSealError(
                f"{validation['scope']} artifacts changed during validation"
            )
    sealed_at = _utc_now()
    manifests = {
        validation["scope"]: _write_seal(
            package,
            validation,
            sealed_at=sealed_at,
        )
        for validation in validations
    }
    for validation in validations:
        if _current_artifact_snapshot(package, validation) != validation[
            "input_snapshot"
        ]:
            raise RunSealError(
                f"{validation['scope']} artifacts changed while sealing"
            )
        _verify_existing_seal(package, scope=validation["scope"])
    return {
        "status": "SEALED_REPLAY_VALID",
        "package": str(package),
        "sealed_at": sealed_at,
        "scopes": {
            scope: {
                "target_records": manifest["target_records"],
                "attempts": manifest["replay"]["attempts"],
                "recovered_failure_markers": manifest["replay"][
                    "recovered_failure_markers"
                ],
                "manifest": str(
                    package / "runs" / scope / RUN_MANIFEST_NAME
                ),
                "checksums": str(
                    package / "runs" / scope / RUN_SUMS_NAME
                ),
            }
            for scope, manifest in manifests.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and seal completed diagnostic and primary ABSA AI runs."
        )
    )
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--scope",
        choices=("both", "diagnostic", "primary"),
        default="both",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scopes = (
        ("diagnostic", "primary")
        if args.scope == "both"
        else (args.scope,)
    )
    result = seal_runs(args.package, scopes=scopes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
