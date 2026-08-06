"""Fail-closed validation for a frozen Lazada annotation release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


ASPECT_COLUMNS = [
    "Chất lượng sản phẩm",
    "Hiệu năng & Trải nghiệm",
    "Đúng mô tả",
    "Giá cả & Khuyến mãi",
    "Vận chuyển",
    "Đóng gói",
    "Dịch vụ & Thái độ Shop",
    "Bảo hành & Đổi trả",
    "Tính xác thực",
]
LEGACY_COLUMNS = ["reviewContent", *ASPECT_COLUMNS]
TOKEN_RE = re.compile(r"\b[^\W_]+\b", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
RAW_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FORBIDDEN_KEYS = {
    "avatar",
    "buyer",
    "buyer_id",
    "buyer_name",
    "cookie",
    "cookies",
    "email",
    "phone",
    "username",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text_sha256(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _exact_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSON in {path}:{line_number}")
            rows.append(value)
    return rows


def _read_checksum_file(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise ValueError(f"Invalid checksum row in {path}:{line_number}")
        digest, relative = parts
        if relative in entries:
            raise ValueError(f"Duplicate checksum path in {path}: {relative}")
        entries[relative] = digest
    return entries


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / Path(relative.replace("/", "\\"))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes validation root: {relative}") from exc
    return path


def _path_in_raw_date_scope(
    path: Path,
    raw_root: Path,
    *,
    raw_date_from: str | None,
    raw_date_through: str | None,
) -> bool:
    if raw_date_from is None and raw_date_through is None:
        return True
    for name, value in (
        ("raw_date_from", raw_date_from),
        ("raw_date_through", raw_date_through),
    ):
        if value is not None and not RAW_DATE_RE.fullmatch(value):
            raise ValueError(f"Invalid source {name}: {value!r}")
    if (
        raw_date_from is not None
        and raw_date_through is not None
        and raw_date_from > raw_date_through
    ):
        raise ValueError("Source raw date range is reversed")
    relative = path.relative_to(raw_root)
    if not relative.parts or not RAW_DATE_RE.fullmatch(relative.parts[0]):
        return False
    raw_date = relative.parts[0]
    if raw_date_from is not None and raw_date < raw_date_from:
        return False
    if raw_date_through is not None and raw_date > raw_date_through:
        return False
    return True


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).strip().casefold()
            if normalized in FORBIDDEN_KEYS:
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


def _load_legacy_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty annotation CSV: {path}") from exc
        if header != LEGACY_COLUMNS:
            raise ValueError(f"Wrong legacy columns in {path}: {header}")
        rows = list(reader)
    for data_row, row in enumerate(rows, 1):
        if len(row) != len(LEGACY_COLUMNS):
            raise ValueError(f"Wrong column count in {path}, data row {data_row}")
        if any(value != "" for value in row[1:]):
            raise ValueError(f"Nonblank label in {path}, data row {data_row}")
    return rows


def _five_grams(text: str) -> set[tuple[str, ...]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = TOKEN_RE.findall(normalized)
    return {
        tuple(tokens[index : index + 5])
        for index in range(max(0, len(tokens) - 4))
    }


def validate_release(
    release_root: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    project_root = project_root.resolve()
    manifest = json.loads(
        (release_root / "manifest.json").read_text(encoding="utf-8")
    )
    counts = manifest["counts"]

    release_checksums = _read_checksum_file(
        release_root / "SHA256SUMS.txt"
    )
    actual_release_files = {
        str(path.relative_to(release_root)).replace("\\", "/")
        for path in release_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if set(release_checksums) != actual_release_files:
        missing = sorted(actual_release_files - set(release_checksums))
        extra = sorted(set(release_checksums) - actual_release_files)
        raise ValueError(
            f"Release checksum inventory mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected in release_checksums.items():
        path = _safe_child(release_root, relative)
        if _sha256_file(path) != expected:
            raise ValueError(f"Release checksum mismatch: {relative}")

    for artifact in manifest["artifacts"]:
        path = _safe_child(release_root, str(artifact["path"]))
        if not path.is_file():
            raise ValueError(f"Manifest artifact is missing: {path}")
        if path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Manifest byte count mismatch: {path}")
        if _sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Manifest artifact checksum mismatch: {path}")
        if "records" in artifact:
            if path.suffix == ".jsonl":
                observed = len(_read_jsonl(path))
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    observed = max(0, sum(1 for _row in csv.reader(handle)) - 1)
            else:
                observed = len(
                    [
                        line
                        for line in path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line
                    ]
                )
            if observed != artifact["records"]:
                raise ValueError(
                    f"Manifest record count mismatch for {path}: "
                    f"{observed} != {artifact['records']}"
                )

    source_inventory_path = (
        release_root / "provenance" / "SOURCE_SHA256SUMS.txt"
    )
    source_inventory = _read_checksum_file(source_inventory_path)
    if len(source_inventory) != manifest["source"]["source_files_hashed"]:
        raise ValueError("Source inventory count does not match manifest")
    inventory_digest = hashlib.sha256(
        source_inventory_path.read_bytes()
    ).hexdigest()
    if inventory_digest != manifest["source"]["source_inventory_sha256"]:
        raise ValueError("Source inventory digest does not match manifest")
    for relative, expected in source_inventory.items():
        path = _safe_child(project_root, relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen raw source changed or is missing: {relative}")
    raw_root = _safe_child(project_root, manifest["source"]["raw_root"])
    raw_date_from = manifest["source"].get("raw_date_from")
    raw_date_through = manifest["source"].get("raw_date_through")
    actual_raw_files = {
        str(path.relative_to(project_root)).replace("\\", "/")
        for path in raw_root.rglob("*")
        if path.is_file()
        and _path_in_raw_date_scope(
            path,
            raw_root,
            raw_date_from=raw_date_from,
            raw_date_through=raw_date_through,
        )
    }
    if actual_raw_files != set(source_inventory):
        raise ValueError("Current raw file set differs from frozen source inventory")

    running_manifests = []
    for path in sorted(raw_root.rglob("manifest.json")):
        if not _path_in_raw_date_scope(
            path,
            raw_root,
            raw_date_from=raw_date_from,
            raw_date_through=raw_date_through,
        ):
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "running":
            running_manifests.append(str(path))
    if running_manifests:
        raise ValueError(f"Raw tree still has running manifests: {running_manifests}")

    canonical = _read_jsonl(release_root / "reviews_canonical.jsonl")
    if len(canonical) != counts["canonical_records"]:
        raise ValueError("Canonical record count does not match manifest")
    sample_ids: set[str] = set()
    review_ids: set[str] = set()
    normalized_hashes: set[str] = set()
    canonical_by_sample: dict[str, dict[str, Any]] = {}
    formula_prefix_records = 0
    for row_number, row in enumerate(canonical, 1):
        sample_id = str(row.get("sample_id") or "")
        review_id = str(row.get("review_id") or "")
        review_text = str(row.get("review_text") or "")
        normalized_hash = _normalized_text_sha256(review_text)
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"Invalid/duplicate sample_id at canonical row {row_number}")
        if not review_id or review_id in review_ids:
            raise ValueError(f"Invalid/duplicate review_id at canonical row {row_number}")
        if normalized_hash in normalized_hashes:
            raise ValueError(f"Duplicate normalized text at canonical row {row_number}")
        forbidden = _contains_forbidden_key(row)
        if forbidden:
            raise ValueError(
                f"Forbidden personal/secret key {forbidden!r} at canonical "
                f"row {row_number}"
            )
        if row.get("selection_policy") != "substantive_vi_v2":
            raise ValueError(f"Wrong policy at canonical row {row_number}")
        if review_text.lstrip().startswith(("=", "+", "-", "@")):
            formula_prefix_records += 1
        sample_ids.add(sample_id)
        review_ids.add(review_id)
        normalized_hashes.add(normalized_hash)
        canonical_by_sample[sample_id] = row

    audit_rows = _read_jsonl(release_root / "record_audit.jsonl")
    if len(audit_rows) != counts["record_audit_records"]:
        raise ValueError("Audit record count does not match manifest")
    audit_by_sample = {str(row["sample_id"]): row for row in audit_rows}
    if set(audit_by_sample) != sample_ids or len(audit_by_sample) != len(audit_rows):
        raise ValueError("record_audit sample IDs are not a 1:1 canonical mapping")

    all_unlabeled = _load_legacy_csv(
        release_root / "annotation" / "all_unlabeled.csv"
    )
    if len(all_unlabeled) != len(canonical):
        raise ValueError("all_unlabeled count differs from canonical")
    for index, (csv_row, canonical_row) in enumerate(
        zip(all_unlabeled, canonical),
        1,
    ):
        if csv_row[0] != canonical_row["review_text"]:
            raise ValueError(f"all_unlabeled text mismatch at row {index}")

    index_path = release_root / "annotation" / "index.csv"
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != len(canonical):
        raise ValueError("Annotation index count differs from canonical")
    if {str(row["sample_id"]) for row in index_rows} != sample_ids:
        raise ValueError("Annotation index sample IDs differ from canonical")

    annotation_cache: dict[str, list[list[str]]] = {}
    used_positions: set[tuple[str, int]] = set()
    primary_samples: set[str] = set()
    review_required_samples: set[str] = set()
    for index_row in index_rows:
        sample_id = str(index_row["sample_id"])
        canonical_row = canonical_by_sample[sample_id]
        if index_row["review_text_sha256"] != _exact_text_sha256(
            str(canonical_row["review_text"])
        ):
            raise ValueError(f"Exact text hash mismatch in index: {sample_id}")
        if index_row["normalized_text_sha256"] != _normalized_text_sha256(
            str(canonical_row["review_text"])
        ):
            raise ValueError(f"Normalized text hash mismatch in index: {sample_id}")
        relative = str(index_row["batch_file"])
        data_row = int(index_row["batch_data_row"])
        if relative not in annotation_cache:
            annotation_cache[relative] = _load_legacy_csv(
                _safe_child(release_root, relative)
            )
        rows = annotation_cache[relative]
        if not 1 <= data_row <= len(rows):
            raise ValueError(f"Out-of-range annotation row for {sample_id}")
        position = (relative, data_row)
        if position in used_positions:
            raise ValueError(f"Duplicate annotation position: {position}")
        used_positions.add(position)
        if rows[data_row - 1][0] != canonical_row["review_text"]:
            raise ValueError(f"Annotation text mismatch for {sample_id}")
        queue = str(index_row["annotation_queue"])
        if queue == "primary":
            primary_samples.add(sample_id)
        elif queue == "review_required":
            review_required_samples.add(sample_id)
        else:
            raise ValueError(f"Unknown annotation queue: {queue}")

    if len(primary_samples) != counts["primary_annotation_records"]:
        raise ValueError("Primary annotation count does not match manifest")
    if len(review_required_samples) != counts["manual_review_records"]:
        raise ValueError("Review-required count does not match manifest")
    if primary_samples & review_required_samples:
        raise ValueError("Annotation queues overlap")
    if primary_samples | review_required_samples != sample_ids:
        raise ValueError("Annotation queues do not cover canonical samples")

    decisions_path = release_root / "annotation" / "review_decisions.csv"
    with decisions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        decision_rows = list(csv.DictReader(handle))
    if [str(row["sample_id"]) for row in decision_rows] != [
        str(row["sample_id"])
        for row in index_rows
        if row["annotation_queue"] == "review_required"
    ]:
        raise ValueError("Decision sheet is not aligned to review-required queue")
    if any(row["decision"] or row["notes"] for row in decision_rows):
        raise ValueError("Decision sheet must be blank in a fresh release")

    exclusion_rows = _read_jsonl(
        release_root / "annotation_exclusions.jsonl"
    )
    if {str(row["sample_id"]) for row in exclusion_rows} != review_required_samples:
        raise ValueError("Exclusion ledger differs from review-required queue")

    pair_rows = _read_jsonl(release_root / "near_duplicate_pairs.jsonl")
    if len(pair_rows) != counts["near_duplicate_pairs"]:
        raise ValueError("Near-duplicate pair count does not match manifest")
    for pair in pair_rows:
        left_id = str(pair["left_sample_id"])
        right_id = str(pair["right_sample_id"])
        if left_id not in sample_ids or right_id not in sample_ids:
            raise ValueError("Near-duplicate pair references unknown sample")
        left_grams = _five_grams(str(canonical_by_sample[left_id]["review_text"]))
        right_grams = _five_grams(str(canonical_by_sample[right_id]["review_text"]))
        intersection = len(left_grams & right_grams)
        union = len(left_grams | right_grams)
        if (
            intersection != pair["intersection_5grams"]
            or union != pair["union_5grams"]
            or 20 * intersection < 17 * union
        ):
            raise ValueError(f"Invalid near-duplicate evidence: {left_id}, {right_id}")

    return {
        "status": "valid",
        "release": str(release_root),
        "release_id": manifest["release_id"],
        "canonical_records": len(canonical),
        "primary_annotation_records": len(primary_samples),
        "manual_review_records": len(review_required_samples),
        "annotation_batches": counts["annotation_batches"],
        "release_files_verified": len(release_checksums),
        "raw_source_files_verified": len(source_inventory),
        "raw_running_manifests": 0,
        "formula_prefix_records": formula_prefix_records,
        "all_label_cells_blank": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/lazada_vi_reviews_v1_20260725"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate_release(
        args.release,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
