"""Independently validate the Q1 human-gold sampling-plan release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    ROOT / "docs" / "audits" / "q1_human_gold_sampling_plan_v1_20260728"
)
SNAPSHOT = ROOT / "data" / "releases" / "q1_dataset_snapshot_v1_20260728"
CORPUS_AUDIT = (
    ROOT / "docs" / "audits" / "q1_collection_corpus_audit_v1_20260728"
)
CONFIG = ROOT / "configs" / "q1_human_gold_sampling_v1.json"


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


def validate(plan_root: Path) -> dict[str, Any]:
    plan_root = plan_root.resolve()
    manifest_path = plan_root / "manifest.json"
    report_path = plan_root / "report.json"
    sums_path = plan_root / "SHA256SUMS.txt"
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    snapshot = read_json(SNAPSHOT / "manifest.json")
    corpus_audit = read_json(CORPUS_AUDIT / "manifest.json")
    config = read_json(CONFIG)
    if (
        manifest.get("artifact_type")
        != "Q1_HUMAN_GOLD_SAMPLING_PLAN_RELEASE"
        or manifest.get("status") != "VALID_FEASIBLE"
        or report.get("status") != "VALID_FEASIBLE"
    ):
        raise ValueError("Sampling plan status/type mismatch")
    if (
        manifest["snapshot_id"] != snapshot["snapshot_id"]
        or report["snapshot_id"] != snapshot["snapshot_id"]
        or manifest["corpus_audit_id"] != corpus_audit["audit_id"]
        or report["corpus_audit_id"] != corpus_audit["audit_id"]
        or report["config_sha256"] != sha256_file(CONFIG)
        or report["plan"] != config
    ):
        raise ValueError("Sampling plan source/config binding mismatch")
    if manifest["report_sha256"] != sha256_file(report_path):
        raise ValueError("Sampling report hash mismatch")

    expected_sums = {"manifest.json": sha256_file(manifest_path)}
    expected_files = {"manifest.json", "SHA256SUMS.txt"}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Sampling-plan artifact inventory is malformed")
    for item in artifacts:
        path = (plan_root / item["path"]).resolve()
        try:
            path.relative_to(plan_root)
        except ValueError as exc:
            raise ValueError("Sampling artifact escapes release root") from exc
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"Sampling artifact mismatch: {item['path']}")
        expected_sums[item["path"]] = item["sha256"]
        expected_files.add(item["path"])
    if read_sums(sums_path) != expected_sums:
        raise ValueError("Sampling checksum ledger is not closed")
    actual_files = {
        path.relative_to(plan_root).as_posix()
        for path in plan_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Sampling-plan file inventory is not closed")

    checks = report["feasibility"]["global_checks"]
    if not checks or not all(checks.values()):
        raise ValueError("Sampling global feasibility gate did not pass")
    if any(
        row["gate"] != "PASS"
        for row in report["feasibility"]["core_category"]
    ):
        raise ValueError("A core category feasibility gate failed")
    challenge = report["feasibility"]["challenge_bins"]
    if (
        any(row["status"] != "PASS" for row in challenge)
        or sum(row["required"] for row in challenge) != 400
        or config["target_unique_reviews"] != 1_200
        or config["annotation_design"]["double_blind_records"] != 1_200
    ):
        raise ValueError("Challenge/target annotation design mismatch")
    eligibility = report["eligibility"]
    if (
        eligibility["eligible_records"] < 1_200
        or eligibility["prior_reference_records"] != 200
        or eligibility["prior_reserved_group_records"] != 6_646
        or eligibility["old_records_included"] != 0
        or eligibility["confirmed_duplicates_included"] != 0
    ):
        raise ValueError("Sampling eligibility closure mismatch")
    return {
        "status": "VALID",
        "plan_id": manifest["plan_id"],
        "snapshot_id": manifest["snapshot_id"],
        "eligible_records": eligibility["eligible_records"],
        "target_records": config["target_unique_reviews"],
        "manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(sums_path),
        "report_sha256": sha256_file(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.plan),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
