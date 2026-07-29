"""Prepare a leakage-controlled ABSA AI-labeling tranche.

The default invocation reproduces the frozen 5,000-review first tranche.
An explicit ``--exclude-package`` plus a new tranche name prepares a
non-overlapping continuation from the same leakage-controlled safe frame.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from human_annotation_ui.validate_export import validate_export
from lazada_collector.ai_tranche import (
    HUMAN_CALIBRATION_SCHEMA_VERSION,
    TRANCHE_INPUT_SCHEMA_VERSION,
    TRANCHE_PRIVATE_INDEX_SCHEMA_VERSION,
    count_labels,
    sha256_file,
    stable_id,
)
from lazada_collector.llm_annotation import canonical_json, label_vector, sha256_text


DEFAULT_RELEASE = Path(
    "data/releases/lazada_vi_absa_curation_v2_1_2_20260725"
)
DEFAULT_ASSIGNMENT = Path(
    "data/annotations/human_reference_ai_preannotation_v1_20260726/"
    "human_check/ai_review.assignment.json"
)
DEFAULT_HUMAN_EXPORT = Path(
    "data/annotations/human_reference_v1_20260726/"
    "hra-ai-review-033a7c9c9ab4eda6-draft-2026-07-26T17-12-40-170Z.json"
)
DEFAULT_CROSSWALK = Path(
    "data/annotations/human_reference_v1_20260726/private/crosswalk.jsonl"
)
DEFAULT_GROUP_RESERVATIONS = Path(
    "data/annotations/human_reference_v1_20260726/private/"
    "group_reservations.jsonl"
)
DEFAULT_GUIDELINE = Path("docs/ABSA_ANNOTATION_GUIDELINE_V2.md")
DEFAULT_OUTPUT = Path(
    "data/annotations/absa_ai_tranche_5000_v1_20260727"
)
SELECTION_SPEC_VERSION = "absa-pseudolabel-selection/1.0.0"
CONTINUATION_SELECTION_SPEC_VERSION = "absa-pseudolabel-selection/1.1.0"
INCREMENTAL_SELECTION_SPEC_VERSION = "absa-pseudolabel-selection/1.2.0"
TRANCHE_NAME = "tranche-0001"
TRANCHE_SIZE = 5000

# These records are structurally valid human confirmations but conflict with
# Guideline V2 on substantive semantic boundaries.  They remain preserved in
# provenance and are excluded from prompting/diagnostic scoring pending expert
# adjudication.  This prevents a small number of assisted-review mistakes from
# being amplified across 5,000 pseudo-labels.
CALIBRATION_EXCLUSIONS: dict[str, str] = {
    "hra-8d94239249a91af8320a": (
        "ESCALATE case; Guideline V2 requires adjudication before it can serve "
        "as a frozen prompt precedent."
    ),
    "hra-a6b84e7b18885355a174": (
        "Contains two mixed aspect labels; early mixed cases require expert "
        "adjudication before calibration use."
    ),
    "hra-048a41d4524755e555bd": (
        "REJECT_NON_REVIEW cases require curator adjudication before prompt use."
    ),
    "hra-177f0cda51a9506c3bdf": (
        "ESCALATE plus mixed case requires expert adjudication."
    ),
    "hra-6c6085d72dab6b0c159b": (
        "Mixed Quality case requires expert adjudication."
    ),
    "hra-2cfaf3d256af83ffe3dd": (
        "Mixed Quality and Authenticity/component boundary require expert "
        "adjudication."
    ),
    "hra-7622b45e946f9e413069": (
        "Mixed Performance case requires expert adjudication."
    ),
    "hra-2603e6762068b56d765a": (
        "Mixed Packaging includes counterfactual/causal evidence and requires "
        "expert adjudication."
    ),
    "hra-f1ed6e27c20247858f98": (
        "REJECT_NON_REVIEW marketing-style text requires curator adjudication."
    ),
    "hra-44e9f9358d0a9d5c3611": (
        "Mixed Performance boundary requires expert adjudication."
    ),
    "hra-1febe3dcfbf50d6dc611": (
        "ESCALATE case requires expert adjudication before calibration use."
    ),
    "hra-dd0eb40722c3849d3942": (
        "Substantive product review mixed with noisy tail must ESCALATE with "
        "BOILERPLATE_MIXED_WITH_REVIEW, not REJECT_NON_REVIEW."
    ),
    "hra-33f3512ff487234f9570": (
        "Faded/unclear appearance is explicit physical-quality evidence; "
        "human removal of Quality is unresolved."
    ),
    "hra-7bfde8cd4a74373b1a65": (
        "'Shop lừa đảo' and 'như hàng mã' do not alone establish counterfeit "
        "authenticity under the frozen boundary."
    ),
    "hra-b602f3e91d23fe4530fe": (
        "The text is a substantive receipt/initial-impression review and "
        "cannot be rejected as non-review."
    ),
    "hra-34335c008304d0f64e78": (
        "Distant expiry/shelf life is not warranty/return evidence; the human "
        "Quality-to-Warranty move is unresolved."
    ),
    "hra-c34e05f52a2195674f81": (
        "Physical lack of sturdiness belongs to Quality; convenience belongs "
        "to Performance, so the human mixed Performance label conflicts."
    ),
    "hra-f5130003f7bf068efa35": (
        "Clear positive product and price review cannot be REJECT_NON_REVIEW."
    ),
    "hra-2c951241556fdd2d2a44": (
        "Mixed Quality case requires expert adjudication."
    ),
    "hra-e0a81d19a0e11b266a71": (
        "REJECT_NON_REVIEW catalogue/gibberish case requires curator "
        "adjudication."
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
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


def _read_sums(path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        checksum, relative = line.split("  ", 1)
        if relative in sums:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        sums[relative] = checksum
    return sums


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def _artifact(path: Path, root: Path, *, records: int | None = None) -> dict:
    item: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        item["records"] = records
    return item


def _source_inventory(
    release: Path,
    *,
    records_relative: str = "clean_core.jsonl",
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = release / "manifest.json"
    sums_path = release / "SHA256SUMS.txt"
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Source release has no artifact inventory")
    inventory = {
        item["path"]: item
        for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        checksum, relative = line.split("  ", 1)
        sums[relative] = checksum
    if records_relative not in {"clean_core.jsonl", "quarantine.jsonl"}:
        raise ValueError(
            f"Unsupported source record partition: {records_relative}"
        )
    for relative in (records_relative, "annotation/index.csv"):
        path = release / relative
        actual = sha256_file(path)
        artifact = inventory.get(relative, {})
        if artifact.get("sha256") != actual:
            raise ValueError(f"Source manifest mismatch: {relative}")
        if (
            artifact.get("bytes") is not None
            and artifact.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"Source byte count mismatch: {relative}")
        if sums.get(relative) != actual:
            raise ValueError(f"Source SHA256SUMS mismatch: {relative}")
    manifest_sha = sha256_file(manifest_path)
    if sums.get("manifest.json") != manifest_sha:
        raise ValueError("Source manifest checksum is not closed")
    return manifest, sums


def _verify_human_reference_sources(
    *,
    crosswalk_path: Path,
    group_reservations_path: Path,
) -> tuple[Path, str]:
    reference_root = crosswalk_path.parent.parent
    if group_reservations_path.parent.parent != reference_root:
        raise ValueError("Human reference ledgers are from different packages")
    manifest_path = reference_root / "manifest.json"
    sums_path = reference_root / "SHA256SUMS.txt"
    manifest = _read_json(manifest_path)
    inventory = {
        item["path"]: item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        checksum, relative = line.split("  ", 1)
        sums[relative] = checksum
    for path in (crosswalk_path, group_reservations_path):
        relative = path.relative_to(reference_root).as_posix()
        actual = sha256_file(path)
        if inventory.get(relative, {}).get("sha256") != actual:
            raise ValueError(
                f"Human-reference manifest mismatch: {relative}"
            )
        if inventory.get(relative, {}).get("bytes") != path.stat().st_size:
            raise ValueError(
                f"Human-reference byte count mismatch: {relative}"
            )
        if sums.get(relative) != actual:
            raise ValueError(
                f"Human-reference SHA256SUMS mismatch: {relative}"
            )
    manifest_sha = sha256_file(manifest_path)
    if sums.get("manifest.json") != manifest_sha:
        raise ValueError("Human-reference manifest checksum mismatch")
    return manifest_path, manifest_sha


def _has_label(row: dict[str, Any], value: Any) -> bool:
    return value in label_vector(row["annotation"])


def _choose_holdout(calibration: list[dict[str, Any]]) -> list[str]:
    """Choose a deterministic 20-record clear-case diagnostic holdout."""

    by_status: dict[str, list[dict[str, Any]]] = {}
    for row in calibration:
        by_status.setdefault(row["annotation"]["annotation_status"], []).append(
            row
        )
    for rows in by_status.values():
        rows.sort(
            key=lambda row: sha256_text(
                f"holdout\0{row['calibration_id']}"
            )
        )

    selected: list[dict[str, Any]] = []
    labeled = by_status.get("LABELED", [])
    selected_ids = {row["calibration_id"] for row in selected}
    neutral = [
        row
        for row in labeled
        if _has_label(row, 0) and row["calibration_id"] not in selected_ids
    ]
    for row in neutral[:4]:
        selected.append(row)
        selected_ids.add(row["calibration_id"])

    for row in labeled:
        if len(selected) >= 20:
            break
        if row["calibration_id"] in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row["calibration_id"])
    if len(selected) != 20:
        raise ValueError("Could not construct 20-record calibration holdout")
    return [row["calibration_id"] for row in selected]


def _hamilton_allocate(
    counts: dict[Any, int],
    *,
    total: int,
) -> dict[Any, int]:
    if not isinstance(total, int) or isinstance(total, bool):
        raise ValueError("Hamilton target must be an integer")
    if not counts:
        raise ValueError("Hamilton counts must not be empty")
    for key, count in counts.items():
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            raise ValueError(
                f"Hamilton count must be a positive integer: {key!r}"
            )
    population = sum(counts.values())
    if population <= 0:
        raise ValueError("Hamilton population must be positive")
    if total < 0 or total > population:
        raise ValueError("Hamilton target exceeds population")
    allocations: dict[Any, int] = {}
    remainders: list[tuple[Fraction, str, Any]] = []
    allocated = 0
    for key, count in counts.items():
        ideal = Fraction(total * count, population)
        floor = ideal.numerator // ideal.denominator
        allocations[key] = floor
        allocated += floor
        key_text = canonical_json(
            list(key) if isinstance(key, tuple) else key
        )
        remainders.append((ideal - floor, key_text, key))
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, _, key in remainders[: total - allocated]:
        allocations[key] += 1
    if sum(allocations.values()) != total:
        raise ValueError("Hamilton allocation total mismatch")
    return allocations


def _hierarchical_hamilton_select(
    rows: list[dict[str, Any]],
    *,
    total: int,
    release_id: str,
    release_manifest_sha256: str,
    human_reference_manifest_sha256: str,
    group_reservations_sha256: str,
    selection_spec_version: str = SELECTION_SPEC_VERSION,
    tranche_name: str = TRANCHE_NAME,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze rating, then category×transport, at natural proportions."""

    rating_counts = Counter(int(row["rating"]) for row in rows)
    rating_allocations = _hamilton_allocate(
        dict(rating_counts),
        total=total,
    )
    joint_allocations: dict[tuple[int, str, str], int] = {}
    selected: list[dict[str, Any]] = []

    def normalized(value: Any) -> str:
        return str(value) if value not in {None, ""} else "<blank>"

    def rank_digest(row: dict[str, Any]) -> str:
        return _selection_rank_digest(
            row,
            release_id=release_id,
            release_manifest_sha256=release_manifest_sha256,
            human_reference_manifest_sha256=(
                human_reference_manifest_sha256
            ),
            group_reservations_sha256=group_reservations_sha256,
            selection_spec_version=selection_spec_version,
            tranche_name=tranche_name,
        )

    for rating in sorted(rating_allocations):
        rating_rows = [
            row for row in rows if int(row["rating"]) == rating
        ]
        joint: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rating_rows:
            key = (
                normalized(row.get("category")),
                normalized(row.get("collection_transport")),
            )
            joint.setdefault(key, []).append(row)
        allocations = _hamilton_allocate(
            {key: len(members) for key, members in joint.items()},
            total=rating_allocations[rating],
        )
        for key, members in joint.items():
            members.sort(
                key=lambda row: (rank_digest(row), row["sample_id"])
            )
            take = allocations[key]
            selected.extend(members[:take])
            joint_allocations[(rating, key[0], key[1])] = take
    if len(selected) != total:
        raise ValueError("Hierarchical Hamilton selection size mismatch")
    selected.sort(key=lambda row: (rank_digest(row), row["sample_id"]))
    return selected, {
        "rating": {
            str(key): value
            for key, value in sorted(rating_allocations.items())
        },
        "rating_category_transport": {
            "|".join((str(key[0]), key[1], key[2])): value
            for key, value in sorted(joint_allocations.items())
        },
    }


def _selection_rank_digest(
    row: dict[str, Any],
    *,
    release_id: str,
    release_manifest_sha256: str,
    human_reference_manifest_sha256: str,
    group_reservations_sha256: str,
    selection_spec_version: str = SELECTION_SPEC_VERSION,
    tranche_name: str = TRANCHE_NAME,
) -> str:
    material = "\0".join(
        [
            selection_spec_version,
            tranche_name,
            release_id,
            release_manifest_sha256,
            human_reference_manifest_sha256,
            group_reservations_sha256,
            row["sample_id"],
            row["curation"]["curated_text_sha256"],
        ]
    )
    return sha256_text(material)


def _tranche_decision(tranche_name: str) -> str:
    normalized = tranche_name.upper().replace("-", "_")
    if not normalized.startswith("TRANCHE_"):
        raise ValueError(
            "Tranche name must use the form tranche-NNNN"
        )
    suffix = normalized.removeprefix("TRANCHE_")
    if len(suffix) != 4 or not suffix.isdigit():
        raise ValueError(
            "Tranche name must use the form tranche-NNNN"
        )
    return f"SELECT_{normalized}"


def _verify_excluded_package(
    package: Path,
    *,
    source_release_id: str,
    source_manifest_sha256: str,
    clean_core_sha256: str,
    human_reference_manifest_sha256: str,
    group_reservations_sha256: str,
) -> tuple[set[str], dict[str, Any]]:
    """Verify and bind a previously published tranche before excluding it."""

    package = package.resolve()
    manifest_path = package / "prepare_manifest.json"
    sums_path = package / "INPUT_SHA256SUMS.txt"
    manifest = _read_json(manifest_path)
    sums = _read_sums(sums_path)
    if manifest.get("status") != "PREPARED_NOT_LABELED":
        raise ValueError("Excluded package is not a frozen prepared package")
    expected = {"prepare_manifest.json"}
    for artifact in manifest.get("artifacts", []):
        if not isinstance(artifact, dict):
            raise ValueError("Excluded package artifact inventory is malformed")
        relative = artifact.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Excluded package artifact path is malformed")
        path = package / relative
        actual = sha256_file(path)
        if (
            artifact.get("sha256") != actual
            or artifact.get("bytes") != path.stat().st_size
            or sums.get(relative) != actual
        ):
            raise ValueError(
                f"Excluded package artifact mismatch: {relative}"
            )
        expected.add(relative)
    if sums.get("prepare_manifest.json") != sha256_file(manifest_path):
        raise ValueError("Excluded package manifest checksum mismatch")
    if set(sums) != expected:
        raise ValueError("Excluded package checksum closure mismatch")

    source = manifest.get("source_release", {})
    selection = manifest.get("selection", {})
    if (
        source.get("release_id") != source_release_id
        or source.get("manifest_sha256") != source_manifest_sha256
        or source.get("clean_core_sha256") != clean_core_sha256
        or selection.get("human_reference_manifest_sha256")
        != human_reference_manifest_sha256
        or selection.get("group_reservations_sha256")
        != group_reservations_sha256
    ):
        raise ValueError(
            "Excluded package is bound to different source/reference inputs"
        )

    private_path = package / "input" / "private_index.jsonl"
    private_rows = _read_jsonl(private_path)
    target_records = manifest.get("target_records")
    if (
        not isinstance(target_records, int)
        or isinstance(target_records, bool)
        or target_records <= 0
        or len(private_rows) != target_records
    ):
        raise ValueError("Excluded package target count is invalid")
    sample_ids: set[str] = set()
    text_hashes: set[str] = set()
    ranks: set[int] = set()
    ordered = sorted(private_rows, key=lambda row: row["selection_rank"])
    for row in ordered:
        sample_id = row.get("sample_id")
        text_hash = row.get("review_text_sha256")
        rank = row.get("selection_rank")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not isinstance(text_hash, str)
            or len(text_hash) != 64
            or text_hash in text_hashes
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank in ranks
        ):
            raise ValueError("Excluded package private index is malformed")
        sample_ids.add(sample_id)
        text_hashes.add(text_hash)
        ranks.add(rank)
    if ranks != set(range(1, target_records + 1)):
        raise ValueError("Excluded package ranks are not contiguous")
    membership_sha = hashlib.sha256(
        "".join(
            f"{row['sample_id']}\t{row['review_text_sha256']}\n"
            for row in ordered
        ).encode("utf-8")
    ).hexdigest()
    if selection.get("ordered_membership_sha256") != membership_sha:
        raise ValueError("Excluded package membership hash mismatch")

    final_manifest_path = package / "final" / "manifest.json"
    final_sums_path = package / "final" / "SHA256SUMS.txt"
    final_manifest = _read_json(final_manifest_path)
    final_sums = _read_sums(final_sums_path)
    if (
        final_manifest.get("status")
        != "AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION"
        or final_manifest.get("target_records") != target_records
        or final_manifest.get("tranche_id") != manifest.get("tranche_id")
        or final_sums.get("manifest.json")
        != sha256_file(final_manifest_path)
    ):
        raise ValueError("Excluded package is not a matching published tranche")

    return sample_ids, {
        "path": package.as_posix(),
        "tranche_id": manifest["tranche_id"],
        "tranche": selection["tranche"],
        "target_records": target_records,
        "ordered_membership_sha256": membership_sha,
        "prepare_manifest_sha256": sha256_file(manifest_path),
        "input_checksums_sha256": sha256_file(sums_path),
        "final_manifest_sha256": sha256_file(final_manifest_path),
        "final_checksums_sha256": sha256_file(final_sums_path),
    }


def _validate_reference_joins(
    *,
    assignment: dict[str, Any],
    crosswalk: list[dict[str, Any]],
    group_reservations: list[dict[str, Any]],
    clean_by_sample: dict[str, dict[str, Any]],
    source_release_id: str,
) -> None:
    """Close assignment/reference/reservation joins against the frozen core."""

    if len(crosswalk) != 200:
        raise ValueError("Human-reference crosswalk must contain 200 rows")
    role = assignment.get("role")
    annotation_field = {
        "A": "annotator_a_id",
        "B": "annotator_b_id",
    }.get(role)
    if annotation_field is None:
        raise ValueError(f"Unsupported human assignment role: {role!r}")

    assignment_records = assignment.get("records")
    if not isinstance(assignment_records, list):
        raise ValueError("Human assignment records must be an array")
    if assignment.get("item_count") != len(assignment_records):
        raise ValueError("Human assignment item_count mismatch")

    cross_by_annotation: dict[str, dict[str, Any]] = {}
    cross_by_sample: dict[str, dict[str, Any]] = {}
    reference_item_ids: set[str] = set()
    for row in crosswalk:
        annotation_id = row.get(annotation_field)
        sample_id = row.get("sample_id")
        reference_item_id = row.get("reference_item_id")
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("Crosswalk annotation ID is invalid")
        if annotation_id in cross_by_annotation:
            raise ValueError(f"Duplicate crosswalk annotation ID: {annotation_id}")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Crosswalk sample_id is invalid")
        if sample_id in cross_by_sample:
            raise ValueError(f"Duplicate crosswalk sample_id: {sample_id}")
        if (
            not isinstance(reference_item_id, str)
            or not reference_item_id
            or reference_item_id in reference_item_ids
        ):
            raise ValueError("Duplicate or invalid crosswalk reference_item_id")
        if row.get("source_release_id") != source_release_id:
            raise ValueError("Crosswalk source release mismatch")
        source = clean_by_sample.get(sample_id)
        if source is None:
            raise ValueError(f"Crosswalk sample absent from clean core: {sample_id}")
        source_hash = source["curation"]["curated_text_sha256"]
        if row.get("review_text_sha256") != source_hash:
            raise ValueError(f"Crosswalk/core text hash mismatch: {sample_id}")
        cross_by_annotation[annotation_id] = row
        cross_by_sample[sample_id] = row
        reference_item_ids.add(reference_item_id)

    assignment_ids: set[str] = set()
    for row in assignment_records:
        annotation_id = row.get("annotation_id")
        review_text = row.get("reviewContent")
        review_hash = row.get("review_text_sha256")
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("Assignment annotation ID is invalid")
        if annotation_id in assignment_ids:
            raise ValueError(f"Duplicate assignment annotation ID: {annotation_id}")
        if not isinstance(review_text, str) or not review_text:
            raise ValueError(f"Assignment review text is invalid: {annotation_id}")
        if review_hash != sha256_text(review_text):
            raise ValueError(f"Assignment text hash mismatch: {annotation_id}")
        linked = cross_by_annotation.get(annotation_id)
        if linked is None:
            raise ValueError(
                f"Assignment annotation absent from crosswalk: {annotation_id}"
            )
        if linked.get("review_text_sha256") != review_hash:
            raise ValueError(
                f"Assignment/crosswalk text hash mismatch: {annotation_id}"
            )
        assignment_ids.add(annotation_id)
    if assignment_ids != set(cross_by_annotation):
        raise ValueError("Assignment/crosswalk annotation ID set mismatch")

    reservation_by_sample: dict[str, dict[str, Any]] = {}
    for row in group_reservations:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Reservation sample_id is invalid")
        if sample_id in reservation_by_sample:
            raise ValueError(f"Duplicate reservation sample_id: {sample_id}")
        source = clean_by_sample.get(sample_id)
        if source is None:
            raise ValueError(
                f"Reservation sample absent from clean core: {sample_id}"
            )
        source_hash = source["curation"]["curated_text_sha256"]
        if row.get("review_text_sha256") != source_hash:
            raise ValueError(f"Reservation/core text hash mismatch: {sample_id}")
        should_be_reference = sample_id in cross_by_sample
        if row.get("is_reference_row") is not should_be_reference:
            raise ValueError(
                f"Reservation reference flag mismatch: {sample_id}"
            )
        if should_be_reference:
            cross_row = cross_by_sample[sample_id]
            if row.get("leakage_group_id") != cross_row.get(
                "leakage_group_id"
            ):
                raise ValueError(
                    f"Reference leakage-group mismatch: {sample_id}"
                )
        reservation_by_sample[sample_id] = row
    if not set(cross_by_sample).issubset(reservation_by_sample):
        raise ValueError("Reservation ledger omits a crosswalk sample")


def prepare(
    *,
    release: Path,
    assignment_path: Path,
    human_export_path: Path,
    crosswalk_path: Path,
    group_reservations_path: Path,
    guideline_path: Path,
    output: Path,
    limit: int,
    exclude_package: Path | None = None,
    tranche_name: str = TRANCHE_NAME,
    reference_release: Path | None = None,
    incremental_source: bool = False,
    expected_safe_frame: int | None = None,
    source_partition: str = "clean_core",
) -> dict[str, Any]:
    release = release.resolve()
    assignment_path = assignment_path.resolve()
    human_export_path = human_export_path.resolve()
    crosswalk_path = crosswalk_path.resolve()
    group_reservations_path = group_reservations_path.resolve()
    guideline_path = guideline_path.resolve()
    output = output.resolve()
    reference_release = (
        reference_release.resolve()
        if reference_release is not None
        else release
    )
    exclude_package = (
        exclude_package.resolve()
        if exclude_package is not None
        else None
    )
    source_partition_files = {
        "clean_core": "clean_core.jsonl",
        "quarantine": "quarantine.jsonl",
    }
    source_records_relative = source_partition_files.get(source_partition)
    if source_records_relative is None:
        raise ValueError(
            f"Unsupported source partition: {source_partition!r}"
        )
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    selected_decision = _tranche_decision(tranche_name)
    if incremental_source and exclude_package is not None:
        raise ValueError(
            "Incremental-source selection cannot use --exclude-package"
        )
    if source_partition != "clean_core" and not incremental_source:
        raise ValueError(
            "Non-clean-core selection requires --incremental-source"
        )
    if incremental_source and expected_safe_frame is None:
        raise ValueError(
            "Incremental-source selection requires --expected-safe-frame"
        )
    if not incremental_source and exclude_package is None and (
        limit != TRANCHE_SIZE or tranche_name != TRANCHE_NAME
    ):
        raise ValueError(
            "The original selection remains frozen at tranche-0001 with "
            f"{TRANCHE_SIZE} records; continuation requires "
            "--exclude-package and a new tranche name"
        )
    if exclude_package is not None and tranche_name == TRANCHE_NAME:
        raise ValueError(
            "Continuation must use a tranche name other than tranche-0001"
        )
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
    ):
        raise ValueError("Tranche limit must be a positive integer")
    if incremental_source:
        selection_spec_version = INCREMENTAL_SELECTION_SPEC_VERSION
    else:
        selection_spec_version = (
            CONTINUATION_SELECTION_SPEC_VERSION
            if exclude_package is not None
            else SELECTION_SPEC_VERSION
        )

    source_manifest, _ = _source_inventory(
        release,
        records_relative=source_records_relative,
    )
    (
        human_reference_manifest_path,
        human_reference_manifest_sha,
    ) = _verify_human_reference_sources(
        crosswalk_path=crosswalk_path,
        group_reservations_path=group_reservations_path,
    )
    assignment = _read_json(assignment_path)
    export = _read_json(human_export_path)
    validation = validate_export(
        assignment_value=assignment,
        export_value=export,
        require_final=False,
    )
    if validation["records_completed_valid"] < 20:
        raise ValueError("At least 20 completed human records are required")
    guideline_sha = sha256_file(guideline_path)
    if assignment["guideline"]["sha256"] != guideline_sha:
        raise ValueError("Assignment/guideline checksum mismatch")

    assignment_by_id = {
        row["annotation_id"]: row for row in assignment["records"]
    }
    calibration: list[dict[str, Any]] = []
    for row in validation["normalized_rows"]:
        assignment_row = assignment_by_id[row["annotation_id"]]
        annotation = {
            key: row[key]
            for key in (
                "schema_version",
                "annotation_status",
                "aspects",
                "review_uncertainty_codes",
                "notes",
            )
        }
        exclusion_reason = CALIBRATION_EXCLUSIONS.get(row["annotation_id"])
        calibration.append(
            {
                "schema_version": HUMAN_CALIBRATION_SCHEMA_VERSION,
                "calibration_id": row["annotation_id"],
                "reviewContent": assignment_row["reviewContent"],
                "review_text_sha256": row["review_text_sha256"],
                "annotation": annotation,
                "human_provenance": {
                    "annotator_id": row["annotator_id"],
                    "assignment_id": row["assignment_id"],
                    "completed_at": row["completed_at"],
                    "export_payload_sha256": validation["payload_sha256"],
                    "confirmation_mode": "AI_ASSISTED_HUMAN_CONFIRMED",
                },
                "semantic_audit": {
                    "decision": (
                        "EXCLUDE_PENDING_EXPERT_ADJUDICATION"
                        if exclusion_reason
                        else "CALIBRATION_ACCEPT"
                    ),
                    "reason": exclusion_reason or (
                        "No frozen-boundary contradiction identified in the "
                        "targeted semantic audit."
                    ),
                    "audit_actor": "AI_SEMANTIC_QA",
                    "human_label_mutated": False,
                },
            }
        )
    calibration.sort(key=lambda row: row["calibration_id"])
    accepted_calibration = [
        row
        for row in calibration
        if row["semantic_audit"]["decision"] == "CALIBRATION_ACCEPT"
    ]
    holdout_ids = set(_choose_holdout(accepted_calibration))
    calibration_ids = {
        row["calibration_id"] for row in accepted_calibration
    } - holdout_ids

    crosswalk = _read_jsonl(crosswalk_path)
    reference_sample_ids = {
        row["sample_id"] for row in crosswalk
    }
    if len(reference_sample_ids) != 200:
        raise ValueError("Expected exactly 200 reserved human-reference samples")
    group_reservations = _read_jsonl(group_reservations_path)
    reserved_group_sample_ids = {
        row["sample_id"] for row in group_reservations
    }
    if len(reserved_group_sample_ids) != len(group_reservations):
        raise ValueError("Duplicate sample_id in group reservation ledger")
    if not reference_sample_ids.issubset(reserved_group_sample_ids):
        raise ValueError(
            "Group reservation ledger omits a human-reference sample"
        )

    def load_clean_index(
        root: Path,
        *,
        context: str,
        records_relative: str = "clean_core.jsonl",
    ) -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
        clean_path = root / records_relative
        rows = _read_jsonl(clean_path)
        index: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{context} row has no sample_id")
            if sample_id in index:
                raise ValueError(
                    f"Duplicate {context} sample_id: {sample_id}"
                )
            text = row.get("curated_review_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Empty {context} curated review: {sample_id}"
                )
            expected_hash = row.get("curation", {}).get(
                "curated_text_sha256"
            )
            if expected_hash != sha256_text(text):
                raise ValueError(
                    f"{context} curated text hash mismatch: {sample_id}"
                )
            index[sample_id] = row
        return clean_path, rows, index

    clean_core_path, clean_rows, clean_by_sample = load_clean_index(
        release,
        context="source",
        records_relative=source_records_relative,
    )
    if incremental_source:
        reference_source_manifest, _ = _source_inventory(reference_release)
        (
            reference_clean_core_path,
            _,
            reference_clean_by_sample,
        ) = load_clean_index(
            reference_release,
            context="reference source",
        )
    else:
        reference_source_manifest = source_manifest
        reference_clean_core_path = clean_core_path
        reference_clean_by_sample = clean_by_sample

    if not reference_sample_ids.issubset(reference_clean_by_sample):
        raise ValueError(
            "Human-reference crosswalk is outside reference clean core"
        )
    if not reserved_group_sample_ids.issubset(reference_clean_by_sample):
        raise ValueError(
            "Group reservation ledger is outside reference clean core"
        )
    _validate_reference_joins(
        assignment=assignment,
        crosswalk=crosswalk,
        group_reservations=group_reservations,
        clean_by_sample=reference_clean_by_sample,
        source_release_id=reference_source_manifest["release_id"],
    )

    reference_text_hashes = {
        str(row["review_text_sha256"]) for row in crosswalk
    }
    reserved_group_text_hashes = {
        str(row["review_text_sha256"]) for row in group_reservations
    }
    if not reference_text_hashes.issubset(reserved_group_text_hashes):
        raise ValueError(
            "Group reservations omit a human-reference text hash"
        )
    local_reference_sample_ids = {
        sample_id
        for sample_id, row in clean_by_sample.items()
        if row["curation"]["curated_text_sha256"]
        in reference_text_hashes
    }
    local_reserved_sample_ids = {
        sample_id
        for sample_id, row in clean_by_sample.items()
        if row["curation"]["curated_text_sha256"]
        in reserved_group_text_hashes
    }
    eligible = [
        row
        for sample_id, row in clean_by_sample.items()
        if sample_id not in local_reserved_sample_ids
    ]
    expected_eligible = (
        expected_safe_frame
        if expected_safe_frame is not None
        else 13976
    )
    if len(eligible) != expected_eligible:
        raise ValueError(
            f"Safe selection frame must contain {expected_eligible} rows, got "
            f"{len(eligible)}"
        )
    source_manifest_sha = sha256_file(release / "manifest.json")
    clean_core_sha = sha256_file(clean_core_path)
    reference_source_manifest_sha = sha256_file(
        reference_release / "manifest.json"
    )
    reference_clean_core_sha = sha256_file(reference_clean_core_path)
    group_reservations_sha = sha256_file(group_reservations_path)
    excluded_sample_ids: set[str] = set()
    excluded_package_binding: dict[str, Any] | None = None
    if exclude_package is not None:
        (
            excluded_sample_ids,
            excluded_package_binding,
        ) = _verify_excluded_package(
            exclude_package,
            source_release_id=source_manifest["release_id"],
            source_manifest_sha256=source_manifest_sha,
            clean_core_sha256=clean_core_sha,
            human_reference_manifest_sha256=human_reference_manifest_sha,
            group_reservations_sha256=group_reservations_sha,
        )
        eligible_sample_ids = {
            row["sample_id"] for row in eligible
        }
        if not excluded_sample_ids.issubset(eligible_sample_ids):
            raise ValueError(
                "Excluded package contains records outside the safe frame"
            )
    selection_frame = [
        row
        for row in eligible
        if row["sample_id"] not in excluded_sample_ids
    ]
    if len(selection_frame) < limit:
        raise ValueError(
            "Not enough safe, non-reference, non-prior source records"
        )
    selected, stratum_allocations = _hierarchical_hamilton_select(
        selection_frame,
        total=limit,
        release_id=source_manifest["release_id"],
        release_manifest_sha256=source_manifest_sha,
        human_reference_manifest_sha256=human_reference_manifest_sha,
        group_reservations_sha256=group_reservations_sha,
        selection_spec_version=selection_spec_version,
        tranche_name=tranche_name,
    )
    selected_sample_ids = {row["sample_id"] for row in selected}
    selected_text_hashes = {
        row["curation"]["curated_text_sha256"] for row in selected
    }
    if (
        len(selected_sample_ids) != limit
        or len(selected_text_hashes) != limit
        or selected_sample_ids.intersection(local_reserved_sample_ids)
        or selected_text_hashes.intersection(reserved_group_text_hashes)
        or selected_text_hashes.intersection(reference_text_hashes)
        or selected_sample_ids.intersection(excluded_sample_ids)
    ):
        raise ValueError("Selected tranche uniqueness/leakage invariant failed")
    tranche_id = stable_id(
        "absa-ai-tranche-",
        source_manifest["release_id"],
        selection_spec_version,
        tranche_name,
        source_manifest_sha,
        human_reference_manifest_sha,
        group_reservations_sha,
        (
            excluded_package_binding["ordered_membership_sha256"]
            if excluded_package_binding is not None
            else ""
        ),
        str(limit),
        length=16,
    )

    blind_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for rank, source in enumerate(selected, 1):
        sample_id = source["sample_id"]
        text = source["curated_review_text"]
        annotation_id = stable_id(
            (
                "aiq-"
                if source_partition == "quarantine"
                else (
                    "aid-"
                    if incremental_source
                    else ("a5k-" if exclude_package is None else "air-")
                )
            ),
            tranche_id,
            sample_id,
            length=20,
        )
        text_sha = sha256_text(text)
        blind_rows.append(
            {
                "schema_version": TRANCHE_INPUT_SCHEMA_VERSION,
                "selection_rank": rank,
                "annotation_id": annotation_id,
                "reviewContent": text,
                "review_text_sha256": text_sha,
            }
        )
        curation = source["curation"]
        private_rows.append(
            {
                "schema_version": TRANCHE_PRIVATE_INDEX_SCHEMA_VERSION,
                "selection_rank": rank,
                "annotation_id": annotation_id,
                "sample_id": sample_id,
                "review_text_sha256": text_sha,
                "parent_canonical_row": curation["parent_canonical_row"],
                "curation_status": curation["status"],
                "category": source.get("category", ""),
                "rating": source.get("rating"),
                "collection_transport": source.get("collection_transport"),
                "product_id": source.get("product_id"),
                "source_release_id": source_manifest["release_id"],
            }
        )
    selected_rank_by_sample = {
        row["sample_id"]: index
        for index, row in enumerate(selected, 1)
    }
    selection_ledger: list[dict[str, Any]] = []
    for source in sorted(eligible, key=lambda row: row["sample_id"]):
        sample_id = source["sample_id"]
        selection_ledger.append(
            {
                "schema_version": "absa-pseudolabel-selection-ledger/1.0.0",
                "sample_id": sample_id,
                "review_text_sha256": source["curation"][
                    "curated_text_sha256"
                ],
                "rating": source["rating"],
                "category": source.get("category", "") or "<blank>",
                "collection_transport": (
                    source.get("collection_transport", "") or "<blank>"
                ),
                "selection_rank_digest": _selection_rank_digest(
                    source,
                    release_id=source_manifest["release_id"],
                    release_manifest_sha256=source_manifest_sha,
                    human_reference_manifest_sha256=(
                        human_reference_manifest_sha
                    ),
                    group_reservations_sha256=group_reservations_sha,
                    selection_spec_version=selection_spec_version,
                    tranche_name=tranche_name,
                ),
                "selected": sample_id in selected_rank_by_sample,
                "selection_rank": selected_rank_by_sample.get(sample_id),
                "decision": (
                    selected_decision
                    if sample_id in selected_rank_by_sample
                    else (
                        "EXCLUDE_ALREADY_PUBLISHED_TRANCHE"
                        if sample_id in excluded_sample_ids
                        else "RETAIN_SAFE_FRAME_FOR_FUTURE_TRANCHE"
                    )
                ),
            }
        )
    membership_sha = hashlib.sha256(
        "".join(
            f"{row['sample_id']}\t{row['review_text_sha256']}\n"
            for row in private_rows
        ).encode("utf-8")
    ).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        input_root = temporary / "input"
        calibration_root = temporary / "calibration"
        blind_path = input_root / "blind_reviews.jsonl"
        private_path = input_root / "private_index.jsonl"
        selection_ledger_path = input_root / "selection_ledger.jsonl"
        calibration_path = calibration_root / "human_confirmed.jsonl"
        split_path = calibration_root / "diagnostic_split.json"
        _write_jsonl(blind_path, blind_rows)
        _write_jsonl(private_path, private_rows)
        _write_jsonl(selection_ledger_path, selection_ledger)
        _write_jsonl(calibration_path, calibration)
        split = {
            "schema_version": "absa-calibration-split/1.0.0",
            "method": (
                "Deterministic clear-case holdout enriched with up to four "
                "neutral-label records; excluded uncertainty/non-review/mixed "
                "cases are not treated as calibration truth. This is not a "
                "representative accuracy sample."
            ),
            "records_total": len(calibration),
            "prompt_calibration_ids": sorted(calibration_ids),
            "diagnostic_holdout_ids": sorted(holdout_ids),
            "prompt_calibration_records": len(calibration_ids),
            "diagnostic_holdout_records": len(holdout_ids),
        }
        _write_json(split_path, split)

        provenance_root = temporary / "provenance"
        provenance_root.mkdir(parents=True, exist_ok=True)
        provenance_sources = {
            guideline_path: (
                provenance_root / "ABSA_ANNOTATION_GUIDELINE_V2.md"
            ),
            Path(__file__).resolve(): (
                provenance_root / "prepare_ai_annotation_tranche.py"
            ),
            (Path(__file__).resolve().parent / "run_ai_annotation_tranche.py"): (
                provenance_root / "run_ai_annotation_tranche.py"
            ),
            (
                Path(__file__).resolve().parents[1]
                / "src"
                / "lazada_collector"
                / "ai_tranche.py"
            ): provenance_root / "ai_tranche.py",
            (
                Path(__file__).resolve().parents[1]
                / "src"
                / "lazada_collector"
                / "llm_annotation.py"
            ): provenance_root / "llm_annotation.py",
            (
                Path(__file__).resolve().parents[1]
                / "src"
                / "lazada_collector"
                / "llm_backends.py"
            ): provenance_root / "llm_backends.py",
            (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "absa_compact_batch_output_schema_v1.json"
            ): (
                provenance_root
                / "absa_compact_batch_output_schema_v1.json"
            ),
        }
        for source, destination in provenance_sources.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination)

        human_completed_payload_sha = sha256_text(
            canonical_json(
                [
                    {
                        "calibration_id": row["calibration_id"],
                        "review_text_sha256": row["review_text_sha256"],
                        "annotation": row["annotation"],
                    }
                    for row in calibration
                ]
            )
        )
        calibration_payload_sha = sha256_text(
            canonical_json(
                [
                    {
                        "calibration_id": row["calibration_id"],
                        "review_text_sha256": row["review_text_sha256"],
                        "annotation": row["annotation"],
                    }
                    for row in accepted_calibration
                ]
            )
        )
        source_categories = Counter(
            row.get("category", "") for row in clean_rows
        )
        selected_categories = Counter(
            row.get("category", "") for row in selected
        )
        selected_status = Counter(
            row["curation"]["status"] for row in selected
        )
        artifacts = [
            _artifact(blind_path, temporary, records=len(blind_rows)),
            _artifact(private_path, temporary, records=len(private_rows)),
            _artifact(
                selection_ledger_path,
                temporary,
                records=len(selection_ledger),
            ),
            _artifact(
                calibration_path,
                temporary,
                records=len(calibration),
            ),
            _artifact(split_path, temporary),
        ]
        artifacts.extend(
            _artifact(path, temporary)
            for path in sorted(provenance_root.iterdir())
            if path.is_file()
        )
        manifest = {
            "artifact_type": "CALIBRATED_ABSA_AI_TRANCHE_WORK_PACKAGE",
            "status": "PREPARED_NOT_LABELED",
            "tranche_id": tranche_id,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "target_records": limit,
            "selection": {
                "selection_spec_version": selection_spec_version,
                "source_mode": (
                    "QUARANTINE_PARTITION"
                    if source_partition == "quarantine"
                    else (
                        "INCREMENTAL_CROSS_RELEASE"
                        if incremental_source
                        else "FROZEN_OR_CONTINUATION"
                    )
                ),
                "source_partition": source_partition,
                "tranche": tranche_name,
                "method": (
                    "Hierarchical Hamilton largest-remainder allocation: rating "
                    "first, then category × collection_transport within rating; "
                    "stable SHA-256 rank bound to source/reference/reservation "
                    "hashes within each stratum. For continuation tranches, "
                    "all records in the cryptographically bound prior "
                    "published tranche are excluded before allocation."
                ),
                "stratum_order": [
                    "rating",
                    "category",
                    "collection_transport",
                ],
                "stratum_allocations": stratum_allocations,
                "ordered_membership_sha256": membership_sha,
                "ordered_membership_serialization": (
                    "UTF-8 sample_id<TAB>review_text_sha256<LF> repeated in "
                    "selection_rank order, including terminal LF"
                ),
                "human_reference_manifest_path": (
                    human_reference_manifest_path.as_posix()
                ),
                "human_reference_manifest_sha256": (
                    human_reference_manifest_sha
                ),
                "eligible_after_reference_exclusion": len(eligible),
                "eligible_after_prior_tranche_exclusion": len(
                    selection_frame
                ),
                "excluded_prior_tranche_records": len(
                    excluded_sample_ids
                ),
                "excluded_prior_tranche": excluded_package_binding,
                "reserved_human_reference_records": len(reference_sample_ids),
                "reserved_reference_group_records": len(
                    reserved_group_sample_ids
                ),
                "source_exact_reference_hash_records_excluded": len(
                    local_reference_sample_ids
                ),
                "source_reserved_group_hash_records_excluded": len(
                    local_reserved_sample_ids
                ),
                "group_reservations_path": (
                    group_reservations_path.as_posix()
                ),
                "group_reservations_sha256": group_reservations_sha,
                "overlap_with_human_reference": 0,
                "overlap_with_reserved_reference_groups": 0,
                "overlap_with_prior_tranche": 0,
                "reference_source_release": {
                    "path": reference_release.as_posix(),
                    "release_id": reference_source_manifest["release_id"],
                    "manifest_sha256": reference_source_manifest_sha,
                    "clean_core_sha256": reference_clean_core_sha,
                },
            },
            "source_release": {
                "path": release.as_posix(),
                "release_id": source_manifest["release_id"],
                "manifest_sha256": sha256_file(release / "manifest.json"),
                "records_path": source_records_relative,
                "records_sha256": clean_core_sha,
                "records_count": len(clean_rows),
                **(
                    {
                        "clean_core_sha256": clean_core_sha,
                        "clean_core_records": len(clean_rows),
                    }
                    if source_partition == "clean_core"
                    else {}
                ),
            },
            "guideline": {
                "path": guideline_path.as_posix(),
                "version": assignment["guideline"]["version"],
                "sha256": guideline_sha,
            },
            "human_calibration": {
                "mode": "AI_ASSISTED_HUMAN_CONFIRMED",
                "human_export_path": human_export_path.as_posix(),
                "human_export_file_sha256": sha256_file(human_export_path),
                "human_export_payload_sha256": validation["payload_sha256"],
                "human_export_status": validation["export_status"],
                "completed_valid": len(calibration),
                "semantic_audit_accepted": len(accepted_calibration),
                "semantic_audit_excluded_pending_adjudication": (
                    len(calibration) - len(accepted_calibration)
                ),
                "incomplete_excluded": validation["records_incomplete"],
                "human_completed_payload_sha256": (
                    human_completed_payload_sha
                ),
                "calibration_payload_sha256": calibration_payload_sha,
                "label_summary": count_labels(calibration),
                "limitations": [
                    "The source human export is DRAFT, not FINAL.",
                    "Only complete=true records were admitted.",
                    "Annotator reviewed seeded AI suggestions; confirmation bias "
                    "is possible.",
                    "Twenty uncertainty/non-review/mixed or semantically "
                    "conflicting confirmations are preserved but excluded from "
                    "prompts pending expert adjudication.",
                ],
            },
            "distributions": {
                "source_categories": dict(sorted(source_categories.items())),
                "selected_categories": dict(sorted(selected_categories.items())),
                "selected_curation_status": dict(
                    sorted(selected_status.items())
                ),
            },
            "data_transmission_notice": (
                "Only blinded annotation_id and review text are sent to the "
                "configured HTTPS LLM endpoint during a run. Private source IDs "
                "remain local."
            ),
            "artifacts": sorted(artifacts, key=lambda row: row["path"]),
        }
        manifest_path = temporary / "prepare_manifest.json"
        _write_json(manifest_path, manifest)

        sums_rows = [
            (artifact["sha256"], artifact["path"]) for artifact in artifacts
        ]
        sums_rows.append((sha256_file(manifest_path), "prepare_manifest.json"))
        sums_path = temporary / "INPUT_SHA256SUMS.txt"
        sums_path.write_text(
            "".join(
                f"{checksum}  {relative}\n"
                for checksum, relative in sorted(
                    sums_rows,
                    key=lambda row: row[1],
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "PREPARED_NOT_LABELED",
        "tranche_id": tranche_id,
        "output": str(output),
        "source_partition": source_partition,
        "target_records": len(blind_rows),
        "human_confirmed_calibration": len(calibration),
        "diagnostic_holdout": len(holdout_ids),
        "reference_overlap": 0,
        "reserved_group_overlap": 0,
        "source_reference_hash_records_excluded": len(
            local_reference_sample_ids
        ),
        "source_reserved_group_hash_records_excluded": len(
            local_reserved_sample_ids
        ),
        "prior_tranche_overlap": 0,
        "excluded_prior_tranche_records": len(excluded_sample_ids),
        "eligible_after_prior_tranche_exclusion": len(selection_frame),
        "semantic_calibration_accepted": len(accepted_calibration),
        "semantic_calibration_excluded": (
            len(calibration) - len(accepted_calibration)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--assignment", type=Path, default=DEFAULT_ASSIGNMENT)
    parser.add_argument(
        "--human-export",
        type=Path,
        default=DEFAULT_HUMAN_EXPORT,
    )
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument(
        "--group-reservations",
        type=Path,
        default=DEFAULT_GROUP_RESERVATIONS,
    )
    parser.add_argument("--guideline", type=Path, default=DEFAULT_GUIDELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=TRANCHE_SIZE)
    parser.add_argument(
        "--exclude-package",
        type=Path,
        default=None,
        help=(
            "Previously published tranche to exclude from a continuation "
            "selection."
        ),
    )
    parser.add_argument(
        "--tranche-name",
        default=TRANCHE_NAME,
        help="Stable tranche name in the form tranche-NNNN.",
    )
    parser.add_argument(
        "--reference-release",
        type=Path,
        default=None,
        help=(
            "Release that owns the human-reference crosswalk and group "
            "reservations. Defaults to --release."
        ),
    )
    parser.add_argument(
        "--incremental-source",
        action="store_true",
        help=(
            "Prepare a new independently frozen incremental source while "
            "excluding human-reference leakage by curated-text hash."
        ),
    )
    parser.add_argument(
        "--expected-safe-frame",
        type=int,
        default=None,
        help=(
            "Fail-closed expected eligible count after reference-group hash "
            "exclusion; required with --incremental-source."
        ),
    )
    parser.add_argument(
        "--source-partition",
        choices=("clean_core", "quarantine"),
        default="clean_core",
        help=(
            "Versioned curation partition to annotate. Quarantine remains "
            "quarantine-origin data and requires --incremental-source."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare(
        release=args.release,
        assignment_path=args.assignment,
        human_export_path=args.human_export,
        crosswalk_path=args.crosswalk,
        group_reservations_path=args.group_reservations,
        guideline_path=args.guideline,
        output=args.output,
        limit=args.limit,
        exclude_package=args.exclude_package,
        tranche_name=args.tranche_name,
        reference_release=args.reference_release,
        incremental_source=args.incremental_source,
        expected_safe_frame=args.expected_safe_frame,
        source_partition=args.source_partition,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
