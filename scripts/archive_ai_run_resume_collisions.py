"""Archive stale retry tails created by the pre-fix resume batch-ID collision.

The original runner derived a batch ID only from the frozen target IDs.  A
resume invocation therefore reused the same attempt directory, overwrote
attempt-01/02, and could leave an older attempt-03 after a newer terminal
attempt.  This tool does not alter annotations.  It copies every surviving
stale artifact into a checksummed recovery archive before removing only those
stale copies from the active replay tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = Path(
    "data/annotations/absa_ai_remainder_8976_v1_20260728"
)
ARCHIVE_RELATIVE = Path("recovery/resume_attempt_collision_v1")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _started_at(path: Path) -> datetime:
    value = _read_json(path).get("started_at")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Attempt has no started_at: {path}")
    return datetime.fromisoformat(value)


def _stale_attempts(run_root: Path) -> list[Path]:
    stale: list[Path] = []
    attempts_root = run_root / "attempts"
    for batch_root in sorted(
        (path for path in attempts_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        attempts = sorted(batch_root.glob("attempt-*.json"))
        if not attempts:
            continue
        timestamps = [_started_at(path) for path in attempts]
        first_inversion: int | None = None
        latest = timestamps[0]
        for index, timestamp in enumerate(timestamps[1:], 1):
            if timestamp < latest:
                first_inversion = index
                break
            latest = timestamp
        if first_inversion is not None:
            stale.extend(attempts[first_inversion:])
    return stale


def archive_collisions(
    *,
    package: Path,
    scope: str,
) -> dict[str, Any]:
    package = package.resolve()
    run_root = (package / "runs" / scope).resolve()
    try:
        run_root.relative_to(package)
    except ValueError as exc:
        raise ValueError("Run root escaped the package") from exc
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)

    output = (package / ARCHIVE_RELATIVE).resolve()
    try:
        output.relative_to(package)
    except ValueError as exc:
        raise ValueError("Archive root escaped the package") from exc
    if output.exists():
        raise FileExistsError(
            f"Recovery archive already exists: {output}"
        )

    stale_attempts = _stale_attempts(run_root)
    if not stale_attempts:
        raise ValueError("No chronological resume collision was detected")

    sources: list[tuple[Path, str, dict[str, Any] | None]] = []
    for attempt_path in stale_attempts:
        attempt = _read_json(attempt_path)
        attempt_relative = attempt_path.relative_to(run_root).as_posix()
        sources.append((attempt_path, attempt_relative, attempt))
        invocation = (
            run_root
            / "codex_invocations"
            / attempt_path.parent.name
            / attempt_path.stem
            / "last_message.json"
        )
        if invocation.is_file():
            sources.append(
                (
                    invocation,
                    invocation.relative_to(run_root).as_posix(),
                    None,
                )
            )

    recovery_parent = output.parent
    recovery_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            dir=recovery_parent,
        )
    )
    try:
        artifact_rows: list[dict[str, Any]] = []
        for source, original_relative, attempt in sources:
            destination = temporary / "artifacts" / original_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if _sha256(source) != _sha256(destination):
                raise ValueError(
                    f"Recovery copy checksum mismatch: {source}"
                )
            item: dict[str, Any] = {
                "original_run_path": original_relative,
                "archive_path": (
                    Path("artifacts") / original_relative
                ).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "role": (
                    "STALE_ATTEMPT"
                    if attempt is not None
                    else "STALE_CODEX_OUTPUT"
                ),
            }
            if attempt is not None:
                item.update(
                    {
                        "batch_id": attempt.get("batch_id"),
                        "attempt": attempt.get("attempt"),
                        "started_at": attempt.get("started_at"),
                        "outcome": attempt.get("outcome"),
                        "target_ids": attempt.get("target_ids"),
                    }
                )
            artifact_rows.append(item)

        software_root = temporary / "software"
        software_root.mkdir(parents=True, exist_ok=True)
        software_sources = {
            Path(__file__).resolve(): (
                software_root / "archive_ai_run_resume_collisions.py"
            ),
            REPOSITORY_ROOT
            / "scripts"
            / "run_ai_annotation_tranche.py": (
                software_root / "run_ai_annotation_tranche.fixed.py"
            ),
            package
            / "provenance"
            / "run_ai_annotation_tranche.py": (
                software_root / "run_ai_annotation_tranche.pre_fix.py"
            ),
        }
        software_rows: list[dict[str, Any]] = []
        for source, destination in software_sources.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination)
            software_rows.append(
                {
                    "path": destination.relative_to(temporary).as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )

        manifest = {
            "schema_version": "absa-ai-resume-collision-recovery/1.0.0",
            "artifact_type": "NON_LABEL_MUTATING_PROVENANCE_RECOVERY",
            "status": "ARCHIVED_BEFORE_ACTIVE_REPLAY_REPAIR",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package": package.as_posix(),
            "scope": scope,
            "cause": (
                "The pre-fix runner reused a deterministic batch ID across "
                "resume invocations. New attempt-01/02 files could overwrite "
                "older files while an older higher-numbered retry survived."
            ),
            "method": (
                "Detect the first started_at inversion within each numbered "
                "attempt sequence; copy that stale tail and matching Codex "
                "output into this archive; verify byte hashes; only then "
                "remove the stale copies from the active replay tree."
            ),
            "label_mutations": 0,
            "archived_artifacts": sorted(
                artifact_rows,
                key=lambda row: row["archive_path"],
            ),
            "software": sorted(
                software_rows,
                key=lambda row: row["path"],
            ),
            "limitations": [
                "Attempt files overwritten before this repair cannot be "
                "reconstructed; the surviving failure marker records the "
                "last pre-resume error for every recovered target.",
                "This archive is supplementary provenance and is not part of "
                "the active attempt sequence used to validate annotations.",
            ],
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        checksum_rows = [
            (row["sha256"], row["archive_path"])
            for row in artifact_rows
        ] + [
            (row["sha256"], row["path"])
            for row in software_rows
        ] + [(_sha256(manifest_path), "manifest.json")]
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(
                f"{checksum}  {relative}\n"
                for checksum, relative in sorted(
                    checksum_rows,
                    key=lambda row: row[1],
                )
            ),
            encoding="utf-8",
            newline="\n",
        )

        # Only exact files already copied and hash-verified are removed from
        # the active tree. Their recoverable copies remain in the archive.
        for source, original_relative, _ in sources:
            archived = temporary / "artifacts" / original_relative
            if _sha256(source) != _sha256(archived):
                raise ValueError(
                    f"Pre-removal checksum mismatch: {source}"
                )
        for source, _, _ in sources:
            source.unlink()

        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "ARCHIVED_BEFORE_ACTIVE_REPLAY_REPAIR",
        "output": str(output),
        "stale_attempts_archived": len(stale_attempts),
        "artifacts_archived": len(sources),
        "label_mutations": 0,
        "manifest_sha256": _sha256(output / "manifest.json"),
        "checksums_sha256": _sha256(output / "SHA256SUMS.txt"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--scope",
        choices=("diagnostic", "primary"),
        default="primary",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = archive_collisions(
        package=args.package,
        scope=args.scope,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
