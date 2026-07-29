"""Independently validate the Q1 workbench validation release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = (
    ROOT / "docs" / "audits"
    / "q1_annotation_workbench_validation_v1_20260728"
)
PACKAGE = (
    ROOT / "data" / "annotations"
    / "q1_human_gold_1200_v1_20260728"
)
PLAN = (
    ROOT / "docs" / "audits"
    / "q1_human_gold_sampling_plan_v1_1_20260728"
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


def read_sums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in rows:
            raise ValueError(f"Duplicate checksum path: {relative}")
        rows[relative] = digest
    return rows


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    report_path = root / "report.json"
    sums_path = root / "SHA256SUMS.txt"
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    package = read_json(PACKAGE / "manifest.json")
    plan = read_json(PLAN / "manifest.json")
    if (
        manifest.get("artifact_type")
        != "Q1_ANNOTATION_WORKBENCH_VALIDATION_RELEASE"
        or manifest.get("status") != "VALID"
        or report.get("artifact_type")
        != "Q1_ANNOTATION_WORKBENCH_VALIDATION"
        or report.get("status") != "VALID"
        or manifest.get("package_id") != package["package_id"]
        or report.get("package_id") != package["package_id"]
        or manifest.get("plan_id") != plan["plan_id"]
        or report.get("plan_id") != plan["plan_id"]
        or report.get("package_manifest_sha256")
        != sha256_file(PACKAGE / "manifest.json")
        or report.get("plan_manifest_sha256")
        != sha256_file(PLAN / "manifest.json")
        or manifest.get("report_sha256") != sha256_file(report_path)
    ):
        raise ValueError("Validation release source/status binding mismatch")

    expected_sums = {"manifest.json": sha256_file(manifest_path)}
    expected_files = {"manifest.json", "SHA256SUMS.txt"}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Validation artifact inventory malformed")
    for item in artifacts:
        path = (root / item["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Artifact escapes release root") from exc
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"Artifact mismatch: {item['path']}")
        expected_sums[item["path"]] = item["sha256"]
        expected_files.add(item["path"])
    if read_sums(sums_path) != expected_sums:
        raise ValueError("Validation checksum ledger is not closed")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Validation file inventory is not closed")

    tests = report.get("tests")
    if (
        not isinstance(tests, list)
        or len(tests) != 13
        or report.get("tests_passed") != 13
        or report.get("tests_failed") != 0
        or any(row.get("status") != "PASS" for row in tests)
        or len({row.get("test_id") for row in tests}) != 13
    ):
        raise ValueError("Validation test closure mismatch")
    for row in tests:
        log_path = root / row["log_path"]
        if (
            sha256_file(log_path) != row["log_sha256"]
            or "RETURN_CODE: 0" not in log_path.read_text(
                encoding="utf-8"
            )
        ):
            raise ValueError(f"Test log mismatch: {row['test_id']}")

    source_pairs = [
        (
            ROOT / "human_annotation_ui" / "app.js",
            root / "provenance" / "human_annotation_ui" / "app.js",
        ),
        (
            ROOT / "human_annotation_ui" / "serve.py",
            root / "provenance" / "human_annotation_ui" / "serve.py",
        ),
        (
            ROOT / "human_annotation_ui" / "validate_export.py",
            root / "provenance" / "human_annotation_ui"
            / "validate_export.py",
        ),
        (
            ROOT / "tests" / "browser_smoke_q1_workbench.py",
            root / "provenance" / "tests"
            / "browser_smoke_q1_workbench.py",
        ),
    ]
    if any(
        sha256_file(source) != sha256_file(frozen)
        for source, frozen in source_pairs
    ):
        raise ValueError("Current workbench source drifted from validation")
    screenshot_paths = list((root / "screenshots").glob("*.png"))
    if len(screenshot_paths) != 3 or any(
        path.stat().st_size < 1_000 for path in screenshot_paths
    ):
        raise ValueError("Browser screenshot evidence mismatch")
    return {
        "status": "VALID",
        "validation_id": manifest["validation_id"],
        "package_id": manifest["package_id"],
        "plan_id": manifest["plan_id"],
        "tests_passed": len(tests),
        "screenshots": len(screenshot_paths),
        "manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(sums_path),
        "report_sha256": sha256_file(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.release),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
