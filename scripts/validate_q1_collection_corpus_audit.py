"""Independently validate the frozen Q1 collection/corpus audit release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    ROOT / "docs" / "audits" / "q1_collection_corpus_audit_v1_20260728"
)
SNAPSHOT = ROOT / "data" / "releases" / "q1_dataset_snapshot_v1_20260728"


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


def read_sums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in rows:
            raise ValueError(f"Duplicate checksum path: {relative}")
        rows[relative] = digest
    return rows


def validate(audit: Path) -> dict[str, Any]:
    audit = audit.resolve()
    manifest_path = audit / "manifest.json"
    sums_path = audit / "SHA256SUMS.txt"
    report_path = audit / "report.json"
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    snapshot = read_json(SNAPSHOT / "manifest.json")
    if (
        manifest.get("artifact_type")
        != "Q1_COLLECTION_AND_CORPUS_AUDIT_RELEASE"
        or manifest.get("status") != "VALID"
        or report.get("status") != "VALID"
    ):
        raise ValueError("Audit status/type mismatch")
    if (
        manifest.get("snapshot_id") != snapshot["snapshot_id"]
        or report["snapshot"]["snapshot_id"] != snapshot["snapshot_id"]
        or report["snapshot"]["manifest_sha256"]
        != sha256_file(SNAPSHOT / "manifest.json")
        or report["snapshot"]["checksums_sha256"]
        != sha256_file(SNAPSHOT / "SHA256SUMS.txt")
    ):
        raise ValueError("Audit is not bound to the frozen snapshot")
    if manifest.get("report_sha256") != sha256_file(report_path):
        raise ValueError("Audit report hash mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Audit artifact inventory is malformed")
    expected_sums = {"manifest.json": sha256_file(manifest_path)}
    expected_files = {"manifest.json", "SHA256SUMS.txt"}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Malformed audit artifact")
        path = (audit / item["path"]).resolve()
        try:
            path.relative_to(audit)
        except ValueError as exc:
            raise ValueError("Audit artifact escapes release root") from exc
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise ValueError(f"Audit artifact mismatch: {item['path']}")
        expected_sums[item["path"]] = item["sha256"]
        expected_files.add(item["path"])
    if read_sums(sums_path) != expected_sums:
        raise ValueError("Audit checksum ledger is not closed")
    actual_files = {
        path.relative_to(audit).as_posix()
        for path in audit.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Audit file inventory is not closed")

    collection = report["collection"]
    if (
        collection["reviews_written"]
        + collection["reviews_rejected_quality"]
        + collection["reviews_deduplicated"]
        != collection["review_candidates"]
        or collection["open_runs"] != 0
        or sum(collection["run_status"].values()) != collection["runs"]
    ):
        raise ValueError("Collection accounting is not closed")
    accepted = report["accepted_corpus"]
    curation = report["curation"]
    labeling = report["labeling"]
    if (
        accepted["records"] != 32_918
        or sum(curation["status"].values()) != accepted["records"]
        or labeling["records"] != 26_130
        or sum(labeling["annotation_status"].values()) != labeling["records"]
        or labeling["old_records"] != 9_772
    ):
        raise ValueError("Corpus/curation/labeling counts do not close")
    for table in (audit / "tables").glob("*.csv"):
        with table.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty audit table: {table.name}")
    return {
        "status": "VALID",
        "audit_id": manifest["audit_id"],
        "snapshot_id": manifest["snapshot_id"],
        "collection_runs": collection["runs"],
        "accepted_records": accepted["records"],
        "workflow_records": labeling["records"],
        "manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(sums_path),
        "report_sha256": sha256_file(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.audit),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
