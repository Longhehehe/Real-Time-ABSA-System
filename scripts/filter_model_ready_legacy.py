"""Publish a model-ready release after filtering one package by curation.

This is a lossless projection for included rows: model records, labels,
leakage groups, and existing split assignments are copied byte-semantically
from the validated parent release.  Every parent model record receives one
decision in the new ledger.  ``KEEP_CLEANED`` is intentionally excluded when
only ``KEEP`` is allowed because transformed text requires re-annotation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from absa_system.data import (
    MODEL_RELEASE_STATUS,
    _distribution,
    sha256_file,
    stable_hash,
    validate_model_ready_release,
)


CONFIG_SCHEMA = "absa-model-filter-config/1.0.0"
LEDGER_SCHEMA = "absa-model-filter-decision/1.0.0"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    return count


def _verify_checksum_closure(root: Path) -> int:
    checksum_path = root / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Checksum closure missing: {checksum_path}")
    entries = 0
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw_line)
        if not match:
            raise ValueError(
                f"Malformed checksum at {checksum_path}:{line_number}"
            )
        expected, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe checksum path: {relative}")
        target = root / relative_path
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"Checksum mismatch: {target}")
        entries += 1
    if not entries:
        raise ValueError(f"Empty checksum closure: {checksum_path}")
    return entries


def _artifact(
    path: Path,
    root: Path,
    *,
    records: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _decision_for_status(status: str) -> str:
    return {
        "KEEP_CLEANED": "EXCLUDE_LEGACY_REQUIRES_RELABEL",
        "QUARANTINE": "EXCLUDE_LEGACY_QUARANTINE",
        "EXCLUDE_AUTO": "EXCLUDE_LEGACY_CONFIRMED_DUPLICATE",
    }.get(status, "EXCLUDE_LEGACY_CURRATION_STATUS")


def build(
    *,
    project_root: Path,
    config_path: Path,
    output: Path | None,
    built_at: str,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Unsupported filter config: {config.get('schema_version')}")
    parent = (project_root / str(config["parent_release"])).resolve()
    curation = (project_root / str(config["curation_release"])).resolve()
    output = (
        output.resolve()
        if output is not None
        else (project_root / str(config["output_release"])).resolve()
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {output}")

    parent_validation = validate_model_ready_release(parent)
    curation_checksum_entries = _verify_checksum_closure(curation)
    parent_manifest = _read_json(parent / "manifest.json")
    curation_manifest = _read_json(curation / "manifest.json")
    target_package = str(config["target_package"])
    allowed_statuses = {str(value) for value in config["allowed_statuses"]}
    if allowed_statuses != {"KEEP"}:
        raise ValueError(
            "This strict label-reuse filter currently requires allowed_statuses=[KEEP]"
        )

    curation_rows = _read_jsonl(curation / "curation_records.jsonl")
    curation_by_id = {
        str(row.get("sample_id") or ""): row for row in curation_rows
    }
    if len(curation_by_id) != len(curation_rows):
        raise ValueError("Curation sample IDs are blank or duplicated")

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    ledger: list[dict[str, Any]] = []
    seen_parent_ids: set[str] = set()
    selected_target_ids: set[str] = set()
    target_parent_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        for row in _read_jsonl(parent / f"{split}.jsonl"):
            sample_id = str(row["sample_id"])
            if sample_id in seen_parent_ids:
                raise ValueError(f"Duplicate parent sample ID: {sample_id}")
            seen_parent_ids.add(sample_id)
            package = str(row.get("source", {}).get("package") or "")
            base = {
                "schema_version": LEDGER_SCHEMA,
                "parent_release_id": parent_manifest["release_id"],
                "sample_id": sample_id,
                "review_text_sha256": row["review_text_sha256"],
                "parent_split": split,
                "source_package": package,
            }
            if package != target_package:
                decision = "INCLUDE_PARENT_SPLIT"
                records_by_split[split].append(row)
                ledger.append(
                    {
                        **base,
                        "curation_status": None,
                        "decision": decision,
                        "reason": "non_target_package_unchanged",
                    }
                )
                continue

            target_parent_ids.add(sample_id)
            curation_row = curation_by_id.get(sample_id)
            if curation_row is None:
                raise ValueError(f"Target record missing curation decision: {sample_id}")
            raw_hash = str(curation_row.get("raw_text_sha256") or "")
            if raw_hash != row["review_text_sha256"]:
                raise ValueError(f"Parent/curation text hash mismatch: {sample_id}")
            status = str(curation_row.get("status") or "")
            if status in allowed_statuses:
                if curation_row.get("transformation_ids"):
                    raise ValueError(f"KEEP record unexpectedly transformed: {sample_id}")
                if curation_row.get("curated_text_sha256") != raw_hash:
                    raise ValueError(f"KEEP record text changed: {sample_id}")
                selected_target_ids.add(sample_id)
                records_by_split[split].append(row)
                decision = "INCLUDE_LEGACY_KEEP"
                reason = "curation_status_KEEP_and_text_unchanged"
            else:
                decision = _decision_for_status(status)
                reason = f"curation_status={status}"
            ledger.append(
                {
                    **base,
                    "curation_status": status,
                    "curation_primary_reason": curation_row.get("primary_reason"),
                    "curation_reason_codes": curation_row.get("reason_codes", []),
                    "decision": decision,
                    "reason": reason,
                }
            )

    curation_target_ids = set(curation_by_id)
    if target_parent_ids != curation_target_ids:
        raise ValueError(
            "Target package and curation membership differ: "
            f"parent_only={len(target_parent_ids-curation_target_ids)}, "
            f"curation_only={len(curation_target_ids-target_parent_ids)}"
        )
    expected_parent_records = int(parent_manifest["counts"]["model_ready_unique"])
    if len(seen_parent_ids) != expected_parent_records:
        raise ValueError("Parent record count does not match its manifest")

    kept_rows = [
        row for split_rows in records_by_split.values() for row in split_rows
    ]
    kept_ids = {str(row["sample_id"]) for row in kept_rows}
    kept_hashes = {str(row["review_text_sha256"]) for row in kept_rows}
    if len(kept_ids) != len(kept_rows) or len(kept_hashes) != len(kept_rows):
        raise ValueError("Filtered output has duplicate sample IDs or text hashes")
    decision_counts = Counter(str(row["decision"]) for row in ledger)
    ledger.sort(key=lambda row: str(row["sample_id"]))

    groups_by_split = {
        split: {str(row["leakage_group_id"]) for row in rows}
        for split, rows in records_by_split.items()
    }
    overlap = {
        "train_dev": len(groups_by_split["train"] & groups_by_split["dev"]),
        "train_test": len(groups_by_split["train"] & groups_by_split["test"]),
        "dev_test": len(groups_by_split["dev"] & groups_by_split["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Leakage group overlap after filtering: {overlap}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        artifacts: list[dict[str, Any]] = []
        split_counts: dict[str, int] = {}
        for split, rows in records_by_split.items():
            path = temporary / f"{split}.jsonl"
            split_counts[split] = _write_jsonl(path, rows)
            artifacts.append(
                _artifact(path, temporary, records=split_counts[split])
            )
        ledger_path = temporary / "decision_ledger.jsonl"
        ledger_count = _write_jsonl(ledger_path, ledger)
        artifacts.append(_artifact(ledger_path, temporary, records=ledger_count))
        snapshot_path = temporary / "filter_config.json"
        _write_json(snapshot_path, config)
        artifacts.append(_artifact(snapshot_path, temporary))
        readme_path = temporary / "README.md"
        readme_path.write_text(
            "\n".join(
                (
                    f"# {output.name}",
                    "",
                    f"Status: `{MODEL_RELEASE_STATUS}`.",
                    "",
                    "Derived from a validated model-ready parent by excluding",
                    "legacy records that are not unchanged curation `KEEP` rows.",
                    "Split assignments and labels of included records are unchanged.",
                    "",
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        artifacts.append(_artifact(readme_path, temporary))

        release_id = stable_hash(
            "absa-model-ready",
            [
                f"{row['sample_id']}:{row['leakage_group_id']}:{row['split']}"
                for row in kept_rows
            ]
            + [
                sha256_file(config_path),
                sha256_file(parent / "manifest.json"),
                sha256_file(curation / "manifest.json"),
            ],
        )
        manifest = {
            "schema_version": "absa-model-ready-release/1.0.0",
            "release_id": release_id,
            "status": MODEL_RELEASE_STATUS,
            "created_at": built_at,
            "source_config": {
                "path": config_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(config_path),
            },
            "parent_model_ready": {
                "path": parent.relative_to(project_root).as_posix(),
                "release_id": parent_manifest["release_id"],
                "manifest_sha256": sha256_file(parent / "manifest.json"),
                "validation": parent_validation,
            },
            "filter_curation": {
                "path": curation.relative_to(project_root).as_posix(),
                "release_id": curation_manifest["release_id"],
                "manifest_sha256": sha256_file(curation / "manifest.json"),
                "checksum_entries_verified": curation_checksum_entries,
                "target_package": target_package,
                "allowed_statuses": sorted(allowed_statuses),
            },
            "source_packages": parent_manifest.get("source_packages", []),
            "reservation_ledgers": parent_manifest.get("reservation_ledgers", []),
            "curation_sources": parent_manifest.get("curation_sources", [])
            + [
                {
                    "path": (
                        curation / "curation_records.jsonl"
                    ).relative_to(project_root).as_posix(),
                    "sha256": sha256_file(curation / "curation_records.jsonl"),
                }
            ],
            "policy": {
                "operation": "FILTER_VALIDATED_MODEL_READY_PARENT",
                "target_package": target_package,
                "included_curation_statuses": sorted(allowed_statuses),
                "exclude_keep_cleaned_until_relabel": True,
                "labels_modified": False,
                "texts_modified": False,
                "split_assignments_preserved": True,
                "leakage_groups_preserved": True,
                "evaluation_label_warning": parent_manifest["policy"][
                    "evaluation_label_warning"
                ],
            },
            "counts": {
                "source_records": len(seen_parent_ids),
                "candidate_before_exact_dedup": len(seen_parent_ids),
                "model_ready_unique": len(kept_rows),
                "target_package_parent_records": len(target_parent_ids),
                "target_package_included": len(selected_target_ids),
                "decisions": dict(sorted(decision_counts.items())),
                "splits": split_counts,
            },
            "distributions": {
                split: _distribution(rows)
                for split, rows in records_by_split.items()
            },
            "leakage_audit": {
                "group_overlap": overlap,
                "sample_id_unique": len(kept_ids) == len(kept_rows),
                "review_text_sha256_unique": len(kept_hashes) == len(kept_rows),
                "parent_split_preserved": True,
                "parent_leakage_group_preserved": True,
            },
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        checksum_entries = artifacts + [_artifact(manifest_path, temporary)]
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(
                f"{entry['sha256']}  {entry['path']}\n"
                for entry in sorted(checksum_entries, key=lambda item: item["path"])
            ),
            encoding="utf-8",
            newline="\n",
        )
        validation = validate_model_ready_release(temporary)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "VALID",
        "release_id": release_id,
        "output": str(output),
        "counts": manifest["counts"],
        "validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_filter_legacy_keep_v3_20260806.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--built-at", default=None)
    args = parser.parse_args()
    built_at = args.built_at or datetime.now(timezone.utc).isoformat()
    try:
        datetime.fromisoformat(built_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("--built-at must be ISO-8601") from exc
    root = Path(__file__).resolve().parents[1]
    result = build(
        project_root=root,
        config_path=(root / args.config if not args.config.is_absolute() else args.config),
        output=(
            None
            if args.output is None
            else (root / args.output if not args.output.is_absolute() else args.output)
        ),
        built_at=built_at,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
