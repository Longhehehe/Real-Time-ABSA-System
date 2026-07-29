"""Freeze the current Q1 dataset state as a checksum-bound logical snapshot.

The snapshot does not duplicate immutable raw/release payloads.  It binds
every raw file plus each published release's manifest and checksum ledger.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "releases" / "q1_dataset_snapshot_v1_20260728"

RELEASE_SPECS = (
    ("accepted_base", "data/releases/lazada_vi_reviews_v1_20260725", 31_928),
    ("accepted_delta", "data/releases/lazada_vi_reviews_delta_v1_20260728", 990),
    (
        "curation_base",
        "data/releases/lazada_vi_absa_curation_v2_1_2_20260725",
        31_928,
    ),
    (
        "curation_delta",
        "data/releases/lazada_vi_absa_delta_curation_v1_20260728",
        990,
    ),
    (
        "human_reference",
        "data/annotations/human_reference_v1_20260726",
        200,
    ),
    (
        "pseudo_clean_5000",
        "data/annotations/absa_ai_tranche_5000_v1_20260727/final",
        5_000,
    ),
    (
        "pseudo_clean_8976",
        "data/annotations/absa_ai_remainder_8976_v1_20260728/final",
        8_976,
    ),
    (
        "pseudo_delta_clean_613",
        "data/annotations/absa_ai_delta_v1_20260728/final",
        613,
    ),
    (
        "pseudo_quarantine_base_11166",
        (
            "data/annotations/"
            "absa_ai_quarantine_base_11166_v1_20260728/final"
        ),
        11_166,
    ),
    (
        "pseudo_quarantine_delta_375",
        (
            "data/annotations/"
            "absa_ai_quarantine_delta_375_v1_20260728/final"
        ),
        375,
    ),
    (
        "pseudo_legacy_old_9772",
        "data/annotations/absa_legacy_old_relabel_9772_v1_20260728/final",
        9_772,
    ),
)

PROVENANCE_INPUTS = (
    "configs/collector.toml",
    "configs/collection_plan.toml",
    "configs/cleaning_v2.json",
    "configs/cleaning_delta_v1_20260728.json",
    "docs/ABSA_ANNOTATION_GUIDELINE_V2.md",
    "docs/audits/QUARANTINE_LABELING_REPORT_20260728.json",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
            count += 1
    return count


def artifact(path: Path, root: Path, *, records: int | None = None) -> dict:
    value = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def raw_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "data" / "raw").rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": (
                    "run_manifest"
                    if path.name == "manifest.json"
                    else "accepted_reviews"
                    if path.name == "reviews.jsonl"
                    else "collection_evidence"
                ),
            }
        )
    if not rows:
        raise ValueError("Raw inventory is empty")
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Raw inventory contains duplicate paths")
    return rows


def release_bindings() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, relative, expected_records in RELEASE_SPECS:
        root = ROOT / relative
        manifest_path = root / "manifest.json"
        sums_path = root / "SHA256SUMS.txt"
        if not manifest_path.is_file() or not sums_path.is_file():
            raise FileNotFoundError(f"Unsealed release: {root}")
        manifest = read_json(manifest_path)
        rows.append(
            {
                "name": name,
                "root": root.relative_to(ROOT).as_posix(),
                "expected_records": expected_records,
                "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "checksums_path": sums_path.relative_to(ROOT).as_posix(),
                "checksums_sha256": sha256_file(sums_path),
                "release_id": manifest.get("release_id"),
                "artifact_type": manifest.get("artifact_type"),
                "status": manifest.get("status"),
            }
        )
    return rows


def build() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {OUTPUT}")
    raw_rows = raw_inventory()
    bindings = release_bindings()
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    )
    try:
        raw_path = temporary / "raw_file_inventory.jsonl"
        write_jsonl(raw_path, raw_rows)
        bindings_path = temporary / "release_bindings.jsonl"
        write_jsonl(bindings_path, bindings)

        copied: list[Path] = []
        for relative in PROVENANCE_INPUTS:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = temporary / "provenance" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)
        software = temporary / "provenance" / "software" / Path(__file__).name
        software.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), software)
        copied.append(software)

        readme = temporary / "README.md"
        readme.write_text(
            "# Q1 dataset logical snapshot v1\n\n"
            "This directory freezes the dataset state used for Q1-oriented "
            "collection, annotation and evaluation design. It does not copy "
            "raw or published release payloads. Instead, it binds every raw "
            "file by SHA-256 and binds each versioned release through its "
            "manifest and checksum ledger.\n\n"
            "The snapshot is not a claim that AI pseudo-labels are human "
            "gold. Quarantine-origin outputs remain a separate provenance "
            "partition pending human verification.\n",
            encoding="utf-8",
            newline="\n",
        )

        raw_membership = hashlib.sha256(
            "".join(
                f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n"
                for row in raw_rows
            ).encode("utf-8")
        ).hexdigest()
        release_membership = hashlib.sha256(
            "".join(
                f"{row['name']}\t{row['root']}\t"
                f"{row['manifest_sha256']}\t{row['checksums_sha256']}\n"
                for row in bindings
            ).encode("utf-8")
        ).hexdigest()
        raw_role_counts: dict[str, int] = {}
        for row in raw_rows:
            raw_role_counts[row["role"]] = raw_role_counts.get(
                row["role"], 0
            ) + 1
        artifacts = [
            artifact(raw_path, temporary, records=len(raw_rows)),
            artifact(bindings_path, temporary, records=len(bindings)),
            artifact(readme, temporary),
            *(artifact(path, temporary) for path in sorted(copied)),
        ]
        manifest = {
            "schema_version": "q1-dataset-logical-snapshot/1.0.0",
            "artifact_type": "CHECKSUM_BOUND_DATASET_SNAPSHOT",
            "status": "FROZEN_LOGICAL_SNAPSHOT",
            "snapshot_id": (
                "q1-dataset-snapshot-"
                + hashlib.sha256(
                    f"{raw_membership}\0{release_membership}".encode("ascii")
                ).hexdigest()[:16]
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": {
                "raw_root": "data/raw",
                "raw_files": len(raw_rows),
                "raw_bytes": sum(row["bytes"] for row in raw_rows),
                "raw_role_counts": dict(sorted(raw_role_counts.items())),
                "release_bindings": len(bindings),
                "raw_membership_sha256": raw_membership,
                "release_membership_sha256": release_membership,
            },
            "measured_dataset_state": {
                "accepted_unique_crawled_reviews": 32_918,
                "crawled_records_sent_through_llm_workflow": 26_130,
                "crawled_terminal_status": {
                    "LABELED": 23_848,
                    "ESCALATE": 1_663,
                    "REJECT_NON_REVIEW": 619,
                },
                "reserved_reference_group_records": 6_646,
                "confirmed_duplicate_exclusions": 142,
                "all_pseudo_label_records_including_old": 35_902,
                "all_exact_unique_review_text_sha256": 35_901,
            },
            "immutability_contract": {
                "raw_payloads_copied": False,
                "published_release_payloads_copied": False,
                "raw_files_bound_individually_by_sha256": True,
                "published_releases_bound_by_manifest_and_checksum_ledger": True,
                "source_files_may_not_be_modified_in_place": True,
            },
            "artifacts": sorted(artifacts, key=lambda row: row["path"]),
            "limitations": [
                "This is a logical checksum freeze, not an operating-system "
                "write-protection mechanism.",
                "AI pseudo-label releases remain pending human verification.",
                "A later dataset release must reference this snapshot ID and "
                "must not silently replace any bound source artifact.",
            ],
        }
        manifest_path = temporary / "manifest.json"
        write_json(manifest_path, manifest)
        checksum_rows = [
            (row["sha256"], row["path"]) for row in artifacts
        ] + [(sha256_file(manifest_path), "manifest.json")]
        sums_path = temporary / "SHA256SUMS.txt"
        sums_path.write_text(
            "".join(
                f"{digest}  {relative}\n"
                for digest, relative in sorted(
                    checksum_rows, key=lambda row: row[1]
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(OUTPUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "FROZEN_LOGICAL_SNAPSHOT",
        "output": str(OUTPUT),
        "snapshot_id": manifest["snapshot_id"],
        "raw_files": len(raw_rows),
        "raw_bytes": sum(row["bytes"] for row in raw_rows),
        "release_bindings": len(bindings),
        "manifest_sha256": sha256_file(OUTPUT / "manifest.json"),
        "checksums_sha256": sha256_file(OUTPUT / "SHA256SUMS.txt"),
    }


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
