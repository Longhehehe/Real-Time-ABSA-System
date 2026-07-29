"""Independently validate the Q1 logical dataset snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "data" / "releases" / "q1_dataset_snapshot_v1_20260728"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object: {path}:{line_number}")
            yield value


def read_sums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in rows:
            raise ValueError(f"Duplicate checksum path: {relative}")
        rows[relative] = digest
    return rows


def validate(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    manifest_path = snapshot / "manifest.json"
    sums_path = snapshot / "SHA256SUMS.txt"
    manifest = read_json(manifest_path)
    if (
        manifest.get("artifact_type")
        != "CHECKSUM_BOUND_DATASET_SNAPSHOT"
        or manifest.get("status") != "FROZEN_LOGICAL_SNAPSHOT"
    ):
        raise ValueError("Snapshot status/type mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Snapshot artifact inventory is malformed")
    expected_sums = {"manifest.json": sha256_file(manifest_path)}
    expected_files = {"manifest.json", "SHA256SUMS.txt"}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Malformed snapshot artifact")
        path = (snapshot / item["path"]).resolve()
        try:
            path.relative_to(snapshot)
        except ValueError as exc:
            raise ValueError("Artifact escapes snapshot root") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != item.get("sha256") or path.stat().st_size != item.get(
            "bytes"
        ):
            raise ValueError(f"Artifact hash/size mismatch: {item['path']}")
        expected_sums[item["path"]] = digest
        expected_files.add(item["path"])
    if read_sums(sums_path) != expected_sums:
        raise ValueError("Snapshot checksum ledger is not closed")
    actual_files = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Snapshot file inventory is not closed")

    raw_rows = list(read_jsonl(snapshot / "raw_file_inventory.jsonl"))
    raw_by_path = {row["path"]: row for row in raw_rows}
    actual_raw = {
        path.relative_to(ROOT).as_posix(): path
        for path in (ROOT / "data" / "raw").rglob("*")
        if path.is_file()
    }
    if set(raw_by_path) != set(actual_raw):
        raise ValueError("Current raw file membership differs from snapshot")
    for relative, row in raw_by_path.items():
        path = actual_raw[relative]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row[
            "sha256"
        ]:
            raise ValueError(f"Raw file drift: {relative}")

    bindings = list(read_jsonl(snapshot / "release_bindings.jsonl"))
    for row in bindings:
        manifest_source = ROOT / row["manifest_path"]
        sums_source = ROOT / row["checksums_path"]
        if (
            sha256_file(manifest_source) != row["manifest_sha256"]
            or sha256_file(sums_source) != row["checksums_sha256"]
        ):
            raise ValueError(f"Published release drift: {row['name']}")

    scope = manifest["scope"]
    if (
        scope["raw_files"] != len(raw_rows)
        or scope["raw_bytes"] != sum(row["bytes"] for row in raw_rows)
        or scope["release_bindings"] != len(bindings)
    ):
        raise ValueError("Snapshot scope counts do not reconcile")
    state = manifest["measured_dataset_state"]
    if (
        state["crawled_records_sent_through_llm_workflow"]
        + state["reserved_reference_group_records"]
        + state["confirmed_duplicate_exclusions"]
        != state["accepted_unique_crawled_reviews"]
    ):
        raise ValueError("Measured crawled-data state does not close")
    if sum(state["crawled_terminal_status"].values()) != state[
        "crawled_records_sent_through_llm_workflow"
    ]:
        raise ValueError("Crawled terminal status does not close")
    return {
        "status": "VALID",
        "snapshot_id": manifest["snapshot_id"],
        "raw_files": len(raw_rows),
        "raw_bytes": sum(row["bytes"] for row in raw_rows),
        "release_bindings": len(bindings),
        "manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(sums_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.snapshot),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
