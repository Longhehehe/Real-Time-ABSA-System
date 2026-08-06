"""Project the exact legacy model-ready membership into a label-blind corpus.

The current curation builder consumes a corpus-style parent release, while the
frozen legacy source exposes ``clean_core.jsonl``.  This adapter binds the
exact legacy records present in a model-ready release back to that frozen
label-blind source and publishes the minimum corpus fields required by the
current cleaning pipeline.  Annotation labels, evidence, status, and split
membership are deliberately not copied into the projected records.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from typing import Any, Iterable


DEFAULT_MODEL_READY = Path("data/model_ready/absa_pseudo_v2_20260806")
DEFAULT_SOURCE_RELEASE = Path(
    "data/releases/legacy_old_reviews_label_blind_v1_20260728"
)
DEFAULT_OUTPUT = Path(
    "data/releases/legacy_old_model_ready_8549_parent_v1_20260806"
)
DEFAULT_PACKAGE = "absa_legacy_old_relabel_9772_v1_20260728"
EXPECTED_COUNT = 8549
CANONICAL_SCHEMA_VERSION = "legacy-model-ready-parent/1.0.0"
AUDIT_SCHEMA_VERSION = "legacy-model-ready-parent-audit/1.0.0"
MANIFEST_SCHEMA_VERSION = "legacy-model-ready-parent-manifest/1.0.0"

FORBIDDEN_RECORD_KEYS = {
    "annotation_status",
    "evidence",
    "label",
    "label_provenance",
    "labels",
    "mention_labels",
    "sentiment_labels",
    "split",
    "target",
    "targets",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
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
        if not target.is_file():
            raise FileNotFoundError(f"Checksum target missing: {target}")
        actual = _sha256_file(target)
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {target}")
        entries += 1
    if not entries:
        raise ValueError(f"Empty checksum closure: {checksum_path}")
    return entries


def _normalized_text_hash(text: str) -> str:
    normalized = re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", text).casefold(),
    ).strip()
    return _sha256_text(normalized)


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            folded = str(key).casefold()
            if folded in FORBIDDEN_RECORD_KEYS:
                return str(key)
            found = _contains_forbidden_key(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _contains_forbidden_key(nested)
            if found:
                return found
    return None


def _source_location(parent_canonical_row: str) -> tuple[str, str, int | None]:
    parts = parent_canonical_row.rsplit(":", 2)
    if len(parts) != 3:
        return parent_canonical_row, "", None
    workbook, sheet, raw_row = parts
    try:
        row_number = int(raw_row)
    except ValueError:
        row_number = None
    return workbook, sheet, row_number


def _artifact(path: Path, root: Path, records: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _membership_rows(
    model_ready: Path,
    package: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    partition_counts: dict[str, int] = {}
    for partition in ("train", "dev", "test"):
        rows = _read_jsonl(model_ready / f"{partition}.jsonl")
        members = [
            row
            for row in rows
            if str(row.get("source", {}).get("package") or "") == package
        ]
        partition_counts[partition] = len(members)
        selected.extend(members)
    return selected, partition_counts


def build(
    *,
    model_ready: Path,
    source_release: Path,
    output: Path,
    package: str,
    expected_count: int,
    built_at: str,
) -> dict[str, Any]:
    model_ready = model_ready.resolve()
    source_release = source_release.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output release already exists: {output}")

    model_ready_checksum_entries = _verify_checksum_closure(model_ready)
    source_checksum_entries = _verify_checksum_closure(source_release)
    model_manifest = json.loads(
        (model_ready / "manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        (source_release / "manifest.json").read_text(encoding="utf-8")
    )
    members, partition_counts = _membership_rows(model_ready, package)
    if len(members) != expected_count:
        raise ValueError(
            f"Expected {expected_count} model-ready members for {package}, "
            f"got {len(members)}"
        )
    member_ids = [str(row.get("sample_id") or "") for row in members]
    if not all(member_ids) or len(set(member_ids)) != len(member_ids):
        raise ValueError("Selected model-ready sample IDs are blank or duplicated")

    source_rows = _read_jsonl(source_release / "clean_core.jsonl")
    source_by_id = {
        str(row.get("sample_id") or ""): row for row in source_rows
    }
    if len(source_by_id) != len(source_rows):
        raise ValueError("Frozen legacy clean core has duplicate sample IDs")
    missing = sorted(set(member_ids) - set(source_by_id))
    if missing:
        raise ValueError(
            f"Model-ready legacy members missing from label-blind source: "
            f"{missing[:5]}"
        )

    model_by_id = {str(row["sample_id"]): row for row in members}
    canonical_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for sample_id in sorted(member_ids):
        member = model_by_id[sample_id]
        source = source_by_id[sample_id]
        text = str(source.get("curated_review_text") or "")
        source_hash = str(
            source.get("curation", {}).get("curated_text_sha256") or ""
        )
        member_text = str(member.get("reviewContent") or "")
        member_hash = str(member.get("review_text_sha256") or "")
        actual_hash = _sha256_text(text)
        if not text.strip():
            raise ValueError(f"Blank source text for {sample_id}")
        if text != member_text or actual_hash not in {source_hash, member_hash}:
            raise ValueError(f"Text binding mismatch for {sample_id}")
        if source_hash != actual_hash or member_hash != actual_hash:
            raise ValueError(f"Text SHA-256 mismatch for {sample_id}")

        parent_row = str(
            source.get("curation", {}).get("parent_canonical_row") or ""
        )
        workbook, sheet, source_line = _source_location(parent_row)
        synthetic_product_id = f"legacy-unknown-{sample_id}"
        canonical = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "sample_id": sample_id,
            "review_id": sample_id,
            "product_id": synthetic_product_id,
            "product_id_provenance": "SYNTHETIC_UNIQUE_UNKNOWN",
            "category": str(source.get("category") or ""),
            "collection_transport": "legacy_xlsx",
            "rating": None,
            "review_time": "",
            "query": "",
            "sku_info": "",
            "review_text": text,
            "review_text_sha256": actual_hash,
            "source_release_id": source_manifest["release_id"],
            "source_parent_canonical_row": parent_row,
        }
        forbidden = _contains_forbidden_key(canonical)
        if forbidden:
            raise ValueError(
                f"Forbidden annotation field {forbidden!r} in projected record"
            )
        canonical_rows.append(canonical)
        audit_rows.append(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "sample_id": sample_id,
                "review_id": sample_id,
                "product_id": synthetic_product_id,
                "review_text_sha256": actual_hash,
                "normalized_text_sha256": _normalized_text_hash(text),
                "source_relative_path": (
                    f"legacy/data/old_dataset/{workbook}"
                    if workbook
                    else "legacy/data/old_dataset"
                ),
                "source_sheet": sheet,
                "source_line": source_line,
                "source_parent_canonical_row": parent_row,
                "source_release_id": source_manifest["release_id"],
                "model_ready_release_id": model_manifest["release_id"],
                "membership_package": package,
                "flags": [
                    "LEGACY_PRODUCT_ID_UNAVAILABLE",
                    "LEGACY_RATING_UNAVAILABLE",
                ],
                "annotation_eligible": True,
                "requires_adjudication": False,
            }
        )

    membership_sha256 = _sha256_text("\n".join(sorted(member_ids)) + "\n")
    bindings = {
        "schema_version": "legacy-model-ready-input-bindings/1.0.0",
        "model_ready": {
            "path": model_ready.as_posix(),
            "release_id": model_manifest["release_id"],
            "manifest_sha256": _sha256_file(model_ready / "manifest.json"),
            "checksum_entries_verified": model_ready_checksum_entries,
            "selected_package": package,
            "selected_records": len(members),
            "selected_partition_counts": partition_counts,
        },
        "label_blind_source": {
            "path": source_release.as_posix(),
            "release_id": source_manifest["release_id"],
            "manifest_sha256": _sha256_file(source_release / "manifest.json"),
            "clean_core_sha256": _sha256_file(
                source_release / "clean_core.jsonl"
            ),
            "checksum_entries_verified": source_checksum_entries,
        },
        "membership_sample_ids_sha256": membership_sha256,
        "binding_assertions": {
            "sample_ids_one_to_one": True,
            "review_text_equal": True,
            "review_text_sha256_equal": True,
            "annotation_fields_copied": False,
            "split_membership_copied_to_records": False,
        },
        "missing_product_metadata_policy": (
            "Assign one unique synthetic product_id per record so unknown "
            "legacy products are not falsely pooled by product-scoped rules."
        ),
    }
    repository_root = Path.cwd().resolve()
    source_inventory_paths = [
        model_ready / "SHA256SUMS.txt",
        model_ready / "manifest.json",
        model_ready / "train.jsonl",
        model_ready / "dev.jsonl",
        model_ready / "test.jsonl",
        source_release / "SHA256SUMS.txt",
        source_release / "manifest.json",
        source_release / "clean_core.jsonl",
    ]
    source_inventory_lines: list[str] = []
    for path in source_inventory_paths:
        try:
            display_path = path.relative_to(repository_root).as_posix()
        except ValueError:
            display_path = path.as_posix()
        source_inventory_lines.append(f"{_sha256_file(path)}  {display_path}")
    source_inventory_text = "\n".join(source_inventory_lines) + "\n"
    source_inventory_sha256 = _sha256_text(source_inventory_text)
    release_id = (
        "legacy-model-ready-parent-"
        + _sha256_text(
            "\0".join(
                (
                    str(model_manifest["release_id"]),
                    str(source_manifest["release_id"]),
                    package,
                    membership_sha256,
                )
            )
        )[:16]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        canonical_path = temporary / "reviews_canonical.jsonl"
        audit_path = temporary / "record_audit.jsonl"
        bindings_path = temporary / "provenance" / "input_bindings.json"
        source_sums_path = (
            temporary / "provenance" / "SOURCE_SHA256SUMS.txt"
        )
        _write_jsonl(canonical_path, canonical_rows)
        _write_jsonl(audit_path, audit_rows)
        _write_json(bindings_path, bindings)
        source_sums_path.write_text(
            source_inventory_text,
            encoding="utf-8",
            newline="\n",
        )
        artifacts = [
            _artifact(canonical_path, temporary, records=len(canonical_rows)),
            _artifact(audit_path, temporary, records=len(audit_rows)),
            _artifact(bindings_path, temporary),
            _artifact(source_sums_path, temporary),
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "release_id": release_id,
            "release_name": output.name,
            "release_type": "LABEL_BLIND_CORPUS_PARENT_PROJECTION",
            "status": "FROZEN_LABEL_BLIND",
            "built_at": built_at,
            "source_cutoff_at": str(
                model_manifest.get("created_at")
                or source_manifest.get("built_at")
                or built_at
            ),
            "source": {
                "canonical_records": len(canonical_rows),
                "source_inventory_sha256": source_inventory_sha256,
                "membership_sample_ids_sha256": membership_sha256,
                "product_metadata_policy": "SYNTHETIC_UNIQUE_UNKNOWN",
            },
            "counts": {
                "canonical_records": len(canonical_rows),
                "record_audit_records": len(audit_rows),
                "model_ready_members_selected": len(members),
                "source_clean_core_records": len(source_rows),
            },
            "label_blind_projection": {
                "annotation_fields_copied": False,
                "split_membership_copied_to_records": False,
                "review_text_bound_to_frozen_source": True,
            },
            "limitations": [
                "Legacy product_id, rating, query, SKU, and review time are unavailable.",
                "Product-scoped duplicate and template evidence cannot be reconstructed.",
                "Synthetic unique product IDs preserve global rules without pooling unknown products.",
            ],
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        }
        _write_json(temporary / "manifest.json", manifest)
        checksum_targets = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}"
                for path in checksum_targets
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate(output, expected_count=expected_count)


def validate(output: Path, *, expected_count: int | None = None) -> dict[str, Any]:
    output = output.resolve()
    checksum_entries = _verify_checksum_closure(output)
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    canonical_rows = _read_jsonl(output / "reviews_canonical.jsonl")
    audit_rows = _read_jsonl(output / "record_audit.jsonl")
    manifest_count = int(manifest["counts"]["canonical_records"])
    if len(canonical_rows) != manifest_count or len(audit_rows) != manifest_count:
        raise ValueError("Manifest, canonical, and audit counts do not agree")
    if expected_count is not None and manifest_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} projected records, got {manifest_count}"
        )
    canonical_ids = [str(row.get("sample_id") or "") for row in canonical_rows]
    audit_ids = [str(row.get("sample_id") or "") for row in audit_rows]
    if not all(canonical_ids) or len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("Canonical sample IDs are blank or duplicated")
    if set(canonical_ids) != set(audit_ids) or len(set(audit_ids)) != len(audit_ids):
        raise ValueError("Audit is not a one-to-one canonical index")
    for row in canonical_rows:
        forbidden = _contains_forbidden_key(row)
        if forbidden:
            raise ValueError(f"Forbidden annotation field in canonical rows: {forbidden}")
        text = str(row.get("review_text") or "")
        if not text.strip() or _sha256_text(text) != row.get("review_text_sha256"):
            raise ValueError(f"Invalid projected text binding: {row.get('sample_id')}")
    return {
        "status": "VALID",
        "release_id": manifest["release_id"],
        "output": str(output),
        "canonical_records": len(canonical_rows),
        "audit_records": len(audit_rows),
        "checksum_entries": checksum_entries,
        "annotation_fields_copied": manifest["label_blind_projection"][
            "annotation_fields_copied"
        ],
        "membership_sample_ids_sha256": manifest["source"][
            "membership_sample_ids_sha256"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ready", type=Path, default=DEFAULT_MODEL_READY)
    parser.add_argument("--source-release", type=Path, default=DEFAULT_SOURCE_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--built-at", default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate(args.output, expected_count=args.expected_count)
    else:
        built_at = args.built_at or datetime.now(timezone.utc).isoformat()
        try:
            datetime.fromisoformat(built_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("--built-at must be ISO-8601") from exc
        result = build(
            model_ready=args.model_ready,
            source_release=args.source_release,
            output=args.output,
            package=args.package,
            expected_count=args.expected_count,
            built_at=built_at,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
