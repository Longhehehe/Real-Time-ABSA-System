"""Build a locked 200-review, double-blind human-reference package.

The design is deliberately hybrid:

* 150 representative rows sampled proportionally from the clean core;
* 50 challenge rows sampled from the frozen 450-row LLM pilot frame using
  preregistered text-cue/metadata proxy bins.

Both annotators receive the same source reviews with independent order and
opaque IDs. Public assignments contain only the exact canonical review text,
its SHA-256, an opaque ID, and frozen guideline metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import unicodedata
from typing import Any, Callable, Iterable

from human_annotation_ui import ASSIGNMENT_SCHEMA_VERSION
from human_annotation_ui.common import (
    ASSIGNMENT_PAYLOAD_FIELDS,
    canonical_json,
    load_json,
    sha256_file,
    sha256_text,
    verify_assignment,
)


SELECTION_SPEC_VERSION = "human-reference-selection/1.0.0"
DEFAULT_RELEASE = Path(
    "data/releases/lazada_vi_absa_curation_v2_1_2_20260725"
)
DEFAULT_PILOT = Path("data/annotations/llm_pilot_v1_1_20260725")
DEFAULT_SEMANTIC_HOLDOUT = Path(
    "docs/audits/v2_1_2_semantic_holdout_250.csv"
)
DEFAULT_GUIDELINE = Path("docs/ABSA_ANNOTATION_GUIDELINE_V2.md")
DEFAULT_OUTPUT = Path(
    "data/annotations/human_reference_v1_20260726"
)

REPRESENTATIVE_COUNT = 150
CHALLENGE_COUNT = 50
REFERENCE_COUNT = REPRESENTATIVE_COUNT + CHALLENGE_COUNT
REPRESENTATIVE_GROUP_CAP = 2
CHALLENGE_CATEGORY_CAP = 10

REPRESENTATIVE_SALT = "human-reference-200-v1-representative-row"
REPRESENTATIVE_FILL_SALT = "human-reference-200-v1-representative-fill"
ROLE_ORDER_SALT = "human-reference-200-v1-role-order"

CHALLENGE_BIN_QUOTAS: tuple[tuple[str, int], ...] = (
    ("WARRANTY_RETURN_CUE", 5),
    ("AUTHENTICITY_CUE", 5),
    ("SHIPPING_PACKAGING_BOUNDARY", 5),
    ("SHOP_SERVICE_CUE", 5),
    ("CONTRAST", 8),
    ("NEGATION", 5),
    ("LOW_RATING", 5),
    ("MID_RATING", 4),
    ("CLEANED_OR_LONG", 4),
    ("MINORITY_METADATA", 2),
    ("REMAINDER", 2),
)

WARRANTY_RE = re.compile(
    r"\b(?:bảo\s*hành|đổi\s*trả|đổi\s*hàng|trả\s*hàng|"
    r"hoàn\s*tiền|khiếu\s*nại)\b",
    flags=re.IGNORECASE,
)
AUTHENTICITY_RE = re.compile(
    r"\b(?:chính\s*hãng|hàng\s*giả|fake|authentic|tem|"
    r"mã\s*qr|nguồn\s*gốc)\b",
    flags=re.IGNORECASE,
)
SHIPPING_RE = re.compile(
    r"\b(?:giao\s*hàng|giao\s+nhanh|giao\s+chậm|vận\s*chuyển|"
    r"ship(?:per|ping)?|nhận\s+hàng)\b",
    flags=re.IGNORECASE,
)
PACKAGING_RE = re.compile(
    r"\b(?:đóng\s*gói|bao\s*bì|hộp|bọc|chống\s*sốc|"
    r"gói\s*hàng|kiện\s+hàng)\b",
    flags=re.IGNORECASE,
)
SHOP_SERVICE_RE = re.compile(
    r"\b(?:tư\s*vấn|phản\s*hồi|trả\s*lời|rep|thái\s*độ|"
    r"hỗ\s*trợ|xử\s*lý|nhắn\s*tin|chăm\s*sóc)\b",
    flags=re.IGNORECASE,
)
CONTRAST_RE = re.compile(
    r"\b(?:nhưng|tuy\s*nhiên|mặc\s*dù|dù\s+.{0,24}\s+nhưng|"
    r"bù\s*lại|cơ\s*mà|tuy\s+.{0,24}\s+nhưng)\b",
    flags=re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(?:không|chưa|chẳng|chả|đâu\s+có|không\s+hề|"
    r"không\s+được|không\s+như)\b",
    flags=re.IGNORECASE,
)


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
            count += 1
    return count


def stable_rank(salt: str, *parts: object) -> str:
    material = "\0".join((salt, *(str(part) for part in parts)))
    return sha256_text(material)


def hamilton_quotas(
    counts: dict[Any, int],
    target: int,
) -> dict[Any, int]:
    if target < 0:
        raise ValueError("Hamilton target cannot be negative")
    total = sum(counts.values())
    if total <= 0:
        if target == 0:
            return {key: 0 for key in counts}
        raise ValueError("Cannot allocate a positive target over an empty frame")
    if target > total:
        raise ValueError("Hamilton target exceeds frame size")
    quotas: dict[Any, int] = {}
    remainders: list[tuple[Fraction, str, Any]] = []
    allocated = 0
    for key, count in counts.items():
        exact = Fraction(count * target, total)
        floor = exact.numerator // exact.denominator
        quotas[key] = floor
        allocated += floor
        remainders.append((exact - floor, canonical_json(key), key))
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, _, key in remainders[: target - allocated]:
        quotas[key] += 1
    if sum(quotas.values()) != target:
        raise AssertionError("Hamilton allocation did not close")
    return quotas


def verify_release_artifact(
    release: Path,
    manifest: dict[str, Any],
    relative_path: str,
) -> str:
    inventory = {
        item["path"]: item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    expected = inventory.get(relative_path)
    if expected is None:
        raise ValueError(f"Missing release artifact: {relative_path}")
    path = release / relative_path
    actual = sha256_file(path)
    if (
        actual != expected.get("sha256")
        or path.stat().st_size != expected.get("bytes")
    ):
        raise ValueError(f"Release artifact checksum mismatch: {relative_path}")
    return actual


def verify_pilot_artifact(
    pilot: Path,
    manifest: dict[str, Any],
    relative_path: str,
) -> str:
    inventory = {
        item["path"]: item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    expected = inventory.get(relative_path)
    if expected is None:
        raise ValueError(f"Missing pilot artifact: {relative_path}")
    path = pilot / relative_path
    actual = sha256_file(path)
    if (
        actual != expected.get("sha256")
        or path.stat().st_size != expected.get("bytes")
    ):
        raise ValueError(f"Pilot artifact checksum mismatch: {relative_path}")
    return actual


def build_leakage_groups(
    core_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
    template_rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    sample_ids = {str(row["sample_id"]) for row in core_rows}
    union_find = UnionFind(sample_ids)

    by_product: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    for row in core_rows:
        sample_id = str(row["sample_id"])
        product_id = str(row.get("product_id") or "").strip()
        text_hash = str(row["curation"]["curated_text_sha256"])
        if product_id:
            prior = by_product.setdefault(product_id, sample_id)
            union_find.union(prior, sample_id)
        prior_hash = by_hash.setdefault(text_hash, sample_id)
        union_find.union(prior_hash, sample_id)
        curation = row.get("curation") or {}
        for key in (
            "representative_sample_id",
            "near_duplicate_representative_sample_id",
        ):
            peer = curation.get(key)
            if isinstance(peer, str) and peer in sample_ids:
                union_find.union(sample_id, peer)

    for row in near_rows:
        left = row.get("left_sample_id")
        right = row.get("right_sample_id")
        representative = row.get("representative_sample_id")
        members = [
            value
            for value in (left, right, representative)
            if isinstance(value, str) and value in sample_ids
        ]
        for peer in members[1:]:
            union_find.union(members[0], peer)

    for row in template_rows:
        members = [
            value
            for value in row.get("member_sample_ids", [])
            if isinstance(value, str) and value in sample_ids
        ]
        for peer in members[1:]:
            union_find.union(members[0], peer)

    components_by_root: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(sample_ids):
        components_by_root[union_find.find(sample_id)].append(sample_id)
    group_by_sample: dict[str, str] = {}
    members_by_group: dict[str, list[str]] = {}
    for members in components_by_root.values():
        group_id = "lkg-" + sha256_text("\n".join(members))[:20]
        members_by_group[group_id] = members
        for sample_id in members:
            group_by_sample[sample_id] = group_id
    return group_by_sample, members_by_group


def normalize_cue_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def challenge_matches(
    row: dict[str, Any],
    bin_name: str,
    long_threshold: int,
) -> bool:
    text = normalize_cue_text(row["curated_review_text"])
    if bin_name == "WARRANTY_RETURN_CUE":
        return bool(WARRANTY_RE.search(text))
    if bin_name == "AUTHENTICITY_CUE":
        return bool(AUTHENTICITY_RE.search(text))
    if bin_name == "SHIPPING_PACKAGING_BOUNDARY":
        return bool(SHIPPING_RE.search(text) and PACKAGING_RE.search(text))
    if bin_name == "SHOP_SERVICE_CUE":
        return bool(SHOP_SERVICE_RE.search(text))
    if bin_name == "CONTRAST":
        return bool(CONTRAST_RE.search(text))
    if bin_name == "NEGATION":
        return bool(NEGATION_RE.search(text))
    if bin_name == "LOW_RATING":
        return int(row["rating"]) in {1, 2}
    if bin_name == "MID_RATING":
        return int(row["rating"]) in {3, 4}
    if bin_name == "CLEANED_OR_LONG":
        return (
            row["curation"]["status"] == "KEEP_CLEANED"
            or int(
                row.get("char_count") or len(row["curated_review_text"])
            )
            >= long_threshold
        )
    if bin_name == "MINORITY_METADATA":
        return (
            row.get("collection_transport") == "selenium_dom"
            or not str(row.get("category") or "").strip()
        )
    if bin_name == "REMAINDER":
        return True
    raise ValueError(f"Unknown challenge bin: {bin_name}")


def representative_selection(
    core_rows: list[dict[str, Any]],
    *,
    excluded_sample_ids: set[str],
    group_by_sample: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hash_counts = Counter(
        str(row["curation"]["curated_text_sha256"]) for row in core_rows
    )
    eligible = [
        row
        for row in core_rows
        if str(row["sample_id"]) not in excluded_sample_ids
        and hash_counts[str(row["curation"]["curated_text_sha256"])] == 1
    ]
    rating_counts = Counter(int(row["rating"]) for row in core_rows)
    rating_quotas = hamilton_quotas(dict(rating_counts), REPRESENTATIVE_COUNT)
    expected_rating_quotas = {1: 3, 2: 1, 3: 2, 4: 4, 5: 140}
    if rating_quotas != expected_rating_quotas:
        raise ValueError(
            f"Source distribution drifted; rating quotas={rating_quotas}"
        )

    stratum_quotas: dict[tuple[int, str, str], int] = {}
    stratum_frame_counts: Counter[tuple[int, str, str]] = Counter()
    for row in core_rows:
        stratum = (
            int(row["rating"]),
            str(row.get("category") or "<blank>"),
            str(row.get("collection_transport") or "<blank>"),
        )
        stratum_frame_counts[stratum] += 1
    for rating, rating_target in sorted(rating_quotas.items()):
        within = {
            stratum: count
            for stratum, count in stratum_frame_counts.items()
            if stratum[0] == rating
        }
        stratum_quotas.update(hamilton_quotas(within, rating_target))

    candidates_by_stratum: dict[
        tuple[int, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in eligible:
        stratum = (
            int(row["rating"]),
            str(row.get("category") or "<blank>"),
            str(row.get("collection_transport") or "<blank>"),
        )
        candidates_by_stratum[stratum].append(row)
    eligible_stratum_counts = {
        stratum: len(rows)
        for stratum, rows in candidates_by_stratum.items()
    }
    for rows in candidates_by_stratum.values():
        rows.sort(
            key=lambda row: stable_rank(
                REPRESENTATIVE_SALT,
                row["sample_id"],
            )
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    stratum_selected_counts: Counter[tuple[int, str, str]] = Counter()
    for stratum in sorted(
        stratum_quotas,
        key=lambda value: canonical_json(value),
    ):
        target = stratum_quotas[stratum]
        if target == 0:
            continue
        for row in candidates_by_stratum.get(stratum, []):
            group = group_by_sample[str(row["sample_id"])]
            if group_counts[group] >= REPRESENTATIVE_GROUP_CAP:
                continue
            selected.append(row)
            selected_ids.add(str(row["sample_id"]))
            group_counts[group] += 1
            stratum_selected_counts[stratum] += 1
            if stratum_selected_counts[stratum] == target:
                break

    # Preserve the exact rating margins if a fine stratum is short under the
    # leakage-group cap. This fallback is deterministic and is recorded.
    for rating, rating_target in sorted(rating_quotas.items()):
        current = sum(
            1 for row in selected if int(row["rating"]) == rating
        )
        if current >= rating_target:
            continue
        fill_candidates = sorted(
            (
                row
                for row in eligible
                if int(row["rating"]) == rating
                and str(row["sample_id"]) not in selected_ids
            ),
            key=lambda row: stable_rank(
                REPRESENTATIVE_FILL_SALT,
                rating,
                row["sample_id"],
            ),
        )
        for row in fill_candidates:
            group = group_by_sample[str(row["sample_id"])]
            if group_counts[group] >= REPRESENTATIVE_GROUP_CAP:
                continue
            selected.append(row)
            selected_ids.add(str(row["sample_id"]))
            group_counts[group] += 1
            stratum = (
                int(row["rating"]),
                str(row.get("category") or "<blank>"),
                str(row.get("collection_transport") or "<blank>"),
            )
            stratum_selected_counts[stratum] += 1
            current += 1
            if current == rating_target:
                break
        if current != rating_target:
            raise ValueError(
                f"Cannot fill representative rating {rating}: "
                f"{current}/{rating_target}"
            )

    if len(selected) != REPRESENTATIVE_COUNT:
        raise ValueError(
            f"Representative selection did not close: {len(selected)}"
        )
    selected.sort(
        key=lambda row: (
            int(row["rating"]),
            stable_rank(REPRESENTATIVE_SALT, row["sample_id"]),
        )
    )
    diagnostics = {
        "eligible_records": len(eligible),
        "excluded_records": len(core_rows) - len(eligible),
        "rating_frame_counts": dict(sorted(rating_counts.items())),
        "rating_quotas": dict(sorted(rating_quotas.items())),
        "stratum_frame_counts": {
            "|".join(map(str, key)): value
            for key, value in sorted(stratum_frame_counts.items())
        },
        "stratum_quotas": {
            "|".join(map(str, key)): value
            for key, value in sorted(stratum_quotas.items())
        },
        "eligible_stratum_counts": {
            "|".join(map(str, key)): value
            for key, value in sorted(eligible_stratum_counts.items())
        },
        "stratum_selected_counts": {
            "|".join(map(str, key)): value
            for key, value in sorted(stratum_selected_counts.items())
        },
        "selected_group_counts": dict(group_counts),
    }
    return selected, diagnostics


def round_robin_select(
    rows: list[dict[str, Any]],
    *,
    bin_name: str,
    quota: int,
    group_by_sample: dict[str, str],
    used_groups: set[str],
    category_counts: Counter[str],
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category") or "<blank>")].append(row)
    categories = sorted(
        by_category,
        key=lambda category: stable_rank(
            f"human-reference-200-v1-challenge-{bin_name}-category",
            category,
        ),
    )
    for category, category_rows in by_category.items():
        category_rows.sort(
            key=lambda row: stable_rank(
                f"human-reference-200-v1-challenge-{bin_name}",
                row["sample_id"],
            )
        )

    selected: list[dict[str, Any]] = []
    cursor: Counter[str] = Counter()
    while len(selected) < quota:
        progressed = False
        for category in categories:
            if len(selected) == quota:
                break
            if category_counts[category] >= CHALLENGE_CATEGORY_CAP:
                continue
            bucket = by_category[category]
            while cursor[category] < len(bucket):
                row = bucket[cursor[category]]
                cursor[category] += 1
                group = group_by_sample[str(row["sample_id"])]
                if group in used_groups:
                    continue
                selected.append(row)
                used_groups.add(group)
                category_counts[category] += 1
                progressed = True
                break
        if not progressed:
            break
    if len(selected) != quota:
        raise ValueError(
            f"Challenge bin {bin_name} shortfall: {len(selected)}/{quota}"
        )
    return selected


def challenge_selection(
    pilot_rows: list[dict[str, Any]],
    *,
    semantic_ids: set[str],
    representative_groups: set[str],
    group_by_sample: dict[str, str],
    long_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        row
        for row in pilot_rows
        if int(row["pilot_rank"]) > 5
        and str(row["sample_id"]) not in semantic_ids
        and group_by_sample[str(row["sample_id"])]
        not in representative_groups
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_groups = set(representative_groups)
    category_counts: Counter[str] = Counter()
    bin_pool_counts: dict[str, int] = {}
    for bin_name, quota in CHALLENGE_BIN_QUOTAS:
        bin_candidates = [
            row
            for row in eligible
            if str(row["sample_id"]) not in selected_ids
            and challenge_matches(row, bin_name, long_threshold)
        ]
        bin_pool_counts[bin_name] = len(bin_candidates)
        chosen = round_robin_select(
            bin_candidates,
            bin_name=bin_name,
            quota=quota,
            group_by_sample=group_by_sample,
            used_groups=used_groups,
            category_counts=category_counts,
        )
        for row in chosen:
            row["_challenge_bin"] = bin_name
            selected_ids.add(str(row["sample_id"]))
        selected.extend(chosen)
    if len(selected) != CHALLENGE_COUNT:
        raise ValueError(f"Challenge selection did not close: {len(selected)}")
    if len({group_by_sample[str(row["sample_id"])] for row in selected}) != (
        CHALLENGE_COUNT
    ):
        raise ValueError("Challenge rows are not group-disjoint")
    diagnostics = {
        "eligible_records": len(eligible),
        "long_char_threshold": long_threshold,
        "bin_pool_counts": bin_pool_counts,
        "bin_quotas": dict(CHALLENGE_BIN_QUOTAS),
        "category_counts": dict(sorted(category_counts.items())),
    }
    return selected, diagnostics


def distribution(
    rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], object],
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(key(row)) for row in rows).items(),
            key=lambda item: item[0],
        )
    )


def build_assignment(
    *,
    selected_rows: list[dict[str, Any]],
    reference_id: str,
    role: str,
    guideline_sha: str,
    built_at: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ordered = sorted(
        selected_rows,
        key=lambda row: stable_rank(
            ROLE_ORDER_SALT,
            role,
            reference_id,
            row["sample_id"],
        ),
    )
    public_rows: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for order, row in enumerate(ordered, 1):
        sample_id = str(row["sample_id"])
        annotation_id = (
            f"hr{role.lower()}-"
            + stable_rank(
                "human-reference-200-v1-opaque-item",
                reference_id,
                role,
                sample_id,
            )[:20]
        )
        public_rows.append(
            {
                "annotation_id": annotation_id,
                "reviewContent": row["curated_review_text"],
                "review_text_sha256": row["curation"][
                    "curated_text_sha256"
                ],
            }
        )
        mapping[sample_id] = {
            "annotation_id": annotation_id,
            "order": order,
        }
    assignment_id = (
        f"hra-{role.lower()}-"
        + stable_rank(
            "human-reference-200-v1-assignment",
            reference_id,
            role,
            "\n".join(row["annotation_id"] for row in public_rows),
        )[:16]
    )
    payload = {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "assignment_id": assignment_id,
        "reference_id": reference_id,
        "role": role,
        "item_count": len(public_rows),
        "guideline": {
            "document_id": "ABSA-ANNOTATION-GUIDELINE-V2",
            "version": "2.0.0",
            "sha256": guideline_sha,
        },
        "created_at": built_at,
        "records": public_rows,
    }
    if set(payload) != ASSIGNMENT_PAYLOAD_FIELDS:
        raise AssertionError("Assignment payload field drift")
    assignment = {
        **payload,
        "assignment_payload_sha256": sha256_text(
            canonical_json(payload)
        ),
    }
    verify_assignment(assignment)
    return assignment, mapping


def artifact_entry(
    path: Path,
    root: Path,
    *,
    records: int | None = None,
    visibility: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "visibility": visibility,
    }
    if records is not None:
        entry["records"] = records
    return entry


def prepare(
    *,
    release: Path,
    pilot: Path,
    semantic_holdout: Path,
    guideline: Path,
    output: Path,
    built_at: str | None = None,
) -> dict[str, Any]:
    release = release.resolve()
    pilot = pilot.resolve()
    semantic_holdout = semantic_holdout.resolve()
    guideline = guideline.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    required = (
        release / "manifest.json",
        release / "clean_core.jsonl",
        release / "near_duplicate_candidates.jsonl",
        release / "template_families.jsonl",
        pilot / "manifest.json",
        pilot / "pilot_private_index.jsonl",
        semantic_holdout,
        guideline,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    release_manifest = load_json(release / "manifest.json")
    pilot_manifest = load_json(pilot / "manifest.json")
    source_hashes = {
        "release_manifest": sha256_file(release / "manifest.json"),
        "clean_core": verify_release_artifact(
            release, release_manifest, "clean_core.jsonl"
        ),
        "near_duplicate_candidates": verify_release_artifact(
            release, release_manifest, "near_duplicate_candidates.jsonl"
        ),
        "template_families": verify_release_artifact(
            release, release_manifest, "template_families.jsonl"
        ),
        "pilot_manifest": sha256_file(pilot / "manifest.json"),
        "pilot_private_index": verify_pilot_artifact(
            pilot, pilot_manifest, "pilot_private_index.jsonl"
        ),
        "semantic_holdout": sha256_file(semantic_holdout),
        "guideline": sha256_file(guideline),
    }
    if source_hashes["guideline"] != release_manifest["provenance"][
        "guideline_sha256"
    ]:
        raise ValueError("Guideline is not the release-frozen version")

    core_rows = read_jsonl(release / "clean_core.jsonl")
    if len(core_rows) != 20622:
        raise ValueError(f"Clean-core count drifted: {len(core_rows)}")
    core_by_id = {str(row["sample_id"]): row for row in core_rows}
    if len(core_by_id) != len(core_rows):
        raise ValueError("Duplicate sample_id in clean core")
    for row in core_rows:
        text = row.get("curated_review_text")
        expected_hash = row.get("curation", {}).get("curated_text_sha256")
        if not isinstance(text, str) or sha256_text(text) != expected_hash:
            raise ValueError(
                f"Curated review hash mismatch: {row.get('sample_id')}"
            )

    near_rows = read_jsonl(release / "near_duplicate_candidates.jsonl")
    template_rows = read_jsonl(release / "template_families.jsonl")
    group_by_sample, members_by_group = build_leakage_groups(
        core_rows,
        near_rows,
        template_rows,
    )

    with semantic_holdout.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        semantic_ids = {
            row["sample_id"]
            for row in csv.DictReader(handle)
            if row.get("sample_id")
        }
    pilot_index = read_jsonl(pilot / "pilot_private_index.jsonl")
    if len(pilot_index) != 450:
        raise ValueError(f"Pilot count drifted: {len(pilot_index)}")
    pilot_ids = {str(row["sample_id"]) for row in pilot_index}
    pilot_rows: list[dict[str, Any]] = []
    for pilot_row in pilot_index:
        sample_id = str(pilot_row["sample_id"])
        source = core_by_id.get(sample_id)
        if source is None:
            raise ValueError(f"Pilot sample missing from clean core: {sample_id}")
        if (
            source["curation"]["curated_text_sha256"]
            != pilot_row["review_text_sha256"]
        ):
            raise ValueError(f"Pilot text hash mismatch: {sample_id}")
        merged = dict(source)
        merged["pilot_rank"] = int(pilot_row["pilot_rank"])
        pilot_rows.append(merged)

    representative, representative_diagnostics = representative_selection(
        core_rows,
        excluded_sample_ids=pilot_ids | semantic_ids,
        group_by_sample=group_by_sample,
    )
    representative_groups = {
        group_by_sample[str(row["sample_id"])] for row in representative
    }
    char_lengths = sorted(
        int(row.get("char_count") or len(row["curated_review_text"]))
        for row in pilot_rows
    )
    long_threshold = char_lengths[
        max(0, int(0.95 * len(char_lengths)) - 1)
    ]
    challenge, challenge_diagnostics = challenge_selection(
        pilot_rows,
        semantic_ids=semantic_ids,
        representative_groups=representative_groups,
        group_by_sample=group_by_sample,
        long_threshold=long_threshold,
    )
    for row in representative:
        row["_panel"] = "REPRESENTATIVE"
        row["_selection_bin"] = "PROPORTIONAL_STRATUM"
    for row in challenge:
        row["_panel"] = "CHALLENGE"
        row["_selection_bin"] = row["_challenge_bin"]
    selected = representative + challenge

    selected_ids = [str(row["sample_id"]) for row in selected]
    if len(selected) != REFERENCE_COUNT or len(set(selected_ids)) != len(
        selected
    ):
        raise ValueError("Selected reference does not contain 200 unique rows")
    challenge_groups = {
        group_by_sample[str(row["sample_id"])] for row in challenge
    }
    if representative_groups & challenge_groups:
        raise ValueError("Representative/challenge leakage-group overlap")
    if len(challenge_groups) != CHALLENGE_COUNT:
        raise ValueError("Challenge panel is not group-disjoint")
    if set(selected_ids) & semantic_ids:
        raise ValueError("Semantic holdout leaked into human reference")
    if set(str(row["sample_id"]) for row in representative) & pilot_ids:
        raise ValueError("Pilot rows leaked into representative panel")

    reference_id = (
        "human-absa-reference-"
        + stable_rank(
            "human-reference-200-v1",
            release_manifest["release_id"],
            source_hashes["guideline"],
            "\n".join(selected_ids),
        )[:16]
    )
    if built_at is None:
        built_at = datetime.now(timezone.utc).isoformat()
    else:
        try:
            parsed_built_at = datetime.fromisoformat(
                built_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("--built-at must be an ISO-8601 timestamp") from exc
        if parsed_built_at.tzinfo is None:
            raise ValueError("--built-at must include a UTC offset")
    assignment_a, mapping_a = build_assignment(
        selected_rows=selected,
        reference_id=reference_id,
        role="A",
        guideline_sha=source_hashes["guideline"],
        built_at=built_at,
    )
    assignment_b, mapping_b = build_assignment(
        selected_rows=selected,
        reference_id=reference_id,
        role="B",
        guideline_sha=source_hashes["guideline"],
        built_at=built_at,
    )
    order_a = [
        row["review_text_sha256"] for row in assignment_a["records"]
    ]
    order_b = [
        row["review_text_sha256"] for row in assignment_b["records"]
    ]
    if set(order_a) != set(order_b) or order_a == order_b:
        raise ValueError("A/B must have the same set and different order")
    if {
        row["annotation_id"] for row in assignment_a["records"]
    } & {row["annotation_id"] for row in assignment_b["records"]}:
        raise ValueError("A/B opaque annotation IDs must be disjoint")

    temp_parent = output.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=temp_parent)
    )
    try:
        assignments_dir = temp_root / "assignments"
        private_dir = temp_root / "private"
        assignments_dir.mkdir(parents=True)
        private_dir.mkdir(parents=True)
        assignment_a_path = (
            assignments_dir / "annotator_a.assignment.json"
        )
        assignment_b_path = (
            assignments_dir / "annotator_b.assignment.json"
        )
        write_json(assignment_a_path, assignment_a)
        write_json(assignment_b_path, assignment_b)
        guideline_output = (
            temp_root / "ABSA_ANNOTATION_GUIDELINE_V2.md"
        )
        shutil.copyfile(guideline, guideline_output)

        selected_by_id = {
            str(row["sample_id"]): row for row in selected
        }
        rep_stratum_counts = Counter(
            (
                int(row["rating"]),
                str(row.get("category") or "<blank>"),
                str(row.get("collection_transport") or "<blank>"),
            )
            for row in representative
        )
        rep_frame_counts = {
            tuple(key.split("|", 2)): value
            for key, value in representative_diagnostics[
                "eligible_stratum_counts"
            ].items()
        }
        crosswalk_rows: list[dict[str, Any]] = []
        for reference_rank, row in enumerate(selected, 1):
            sample_id = str(row["sample_id"])
            group = group_by_sample[sample_id]
            stratum = (
                str(int(row["rating"])),
                str(row.get("category") or "<blank>"),
                str(row.get("collection_transport") or "<blank>"),
            )
            stratum_selected = rep_stratum_counts.get(
                (int(stratum[0]), stratum[1], stratum[2]),
                0,
            )
            stratum_frame = rep_frame_counts.get(stratum, 0)
            nominal_weight = (
                round(stratum_frame / stratum_selected, 12)
                if row["_panel"] == "REPRESENTATIVE"
                and stratum_selected
                else None
            )
            crosswalk_rows.append(
                {
                    "reference_rank": reference_rank,
                    "reference_item_id": (
                        "href-"
                        + stable_rank(reference_id, sample_id)[:20]
                    ),
                    "sample_id": sample_id,
                    "review_text_sha256": row["curation"][
                        "curated_text_sha256"
                    ],
                    "panel": row["_panel"],
                    "selection_bin": row["_selection_bin"],
                    "design_stratum": "|".join(stratum),
                    "nominal_analysis_weight": nominal_weight,
                    "analysis_weight_status": (
                        "APPROXIMATE_GROUP_CAP_DEPENDENCE"
                        if nominal_weight is not None
                        else "NOT_APPLICABLE_NONPROBABILITY_CHALLENGE"
                    ),
                    "leakage_group_id": group,
                    "leakage_group_size": len(members_by_group[group]),
                    "product_id": row.get("product_id"),
                    "category": row.get("category") or "<blank>",
                    "rating": row.get("rating"),
                    "collection_transport": row.get(
                        "collection_transport"
                    ),
                    "curation_status": row["curation"]["status"],
                    "parent_canonical_row": row["curation"][
                        "parent_canonical_row"
                    ],
                    "pilot_rank": row.get("pilot_rank"),
                    "annotator_a_id": mapping_a[sample_id][
                        "annotation_id"
                    ],
                    "annotator_a_order": mapping_a[sample_id]["order"],
                    "annotator_b_id": mapping_b[sample_id][
                        "annotation_id"
                    ],
                    "annotator_b_order": mapping_b[sample_id]["order"],
                    "source_release_id": release_manifest["release_id"],
                }
            )
        crosswalk_path = private_dir / "crosswalk.jsonl"
        crosswalk_count = write_jsonl(crosswalk_path, crosswalk_rows)

        pilot_rank_by_id = {
            str(item["sample_id"]): int(item["pilot_rank"])
            for item in pilot_index
        }
        curated_hash_counts = Counter(
            str(row["curation"]["curated_text_sha256"])
            for row in core_rows
        )
        selection_ledger_rows: list[dict[str, Any]] = []
        for row in core_rows:
            sample_id = str(row["sample_id"])
            exclusion_reasons: list[str] = []
            if sample_id in pilot_ids:
                exclusion_reasons.append("PILOT_ROW")
            if sample_id in semantic_ids:
                exclusion_reasons.append("SEMANTIC_HOLDOUT_ROW")
            hash_value = row["curation"]["curated_text_sha256"]
            if curated_hash_counts[str(hash_value)] != 1:
                exclusion_reasons.append("NON_BIJECTIVE_TEXT_HASH")
            selected_row = selected_by_id.get(sample_id)
            pilot_rank = pilot_rank_by_id.get(sample_id)
            challenge_exclusion: list[str] = []
            if pilot_rank is None:
                challenge_exclusion.append("NOT_IN_PILOT_FRAME")
            else:
                if pilot_rank <= 5:
                    challenge_exclusion.append("LLM_PREFLIGHT_RANK_1_5")
                if sample_id in semantic_ids:
                    challenge_exclusion.append(
                        "SEMANTIC_HOLDOUT_ROW"
                    )
                if (
                    group_by_sample[sample_id]
                    in representative_groups
                ):
                    challenge_exclusion.append(
                        "REPRESENTATIVE_GROUP_OVERLAP"
                    )
            selection_ledger_rows.append(
                {
                    "sample_id": sample_id,
                    "review_text_sha256": hash_value,
                    "leakage_group_id": group_by_sample[sample_id],
                    "representative_eligible": not exclusion_reasons,
                    "representative_exclusion_reasons": exclusion_reasons,
                    "representative_rank_sha256": stable_rank(
                        REPRESENTATIVE_SALT, sample_id
                    ),
                    "pilot_rank": pilot_rank,
                    "challenge_eligible": not challenge_exclusion,
                    "challenge_exclusion_reasons": challenge_exclusion,
                    "selected": selected_row is not None,
                    "selected_panel": (
                        selected_row["_panel"]
                        if selected_row is not None
                        else None
                    ),
                    "selected_bin": (
                        selected_row["_selection_bin"]
                        if selected_row is not None
                        else None
                    ),
                }
            )
        selection_ledger_path = private_dir / "selection_ledger.jsonl"
        ledger_count = write_jsonl(
            selection_ledger_path, selection_ledger_rows
        )

        reserved_groups = representative_groups | challenge_groups
        reservation_rows = [
            {
                "sample_id": str(row["sample_id"]),
                "review_text_sha256": row["curation"][
                    "curated_text_sha256"
                ],
                "leakage_group_id": group_by_sample[str(row["sample_id"])],
                "is_reference_row": str(row["sample_id"])
                in selected_by_id,
                "reference_panel": (
                    selected_by_id[str(row["sample_id"])]["_panel"]
                    if str(row["sample_id"]) in selected_by_id
                    else None
                ),
                "reservation_policy": (
                    "RESERVE_IF_REFERENCE_USED_FOR_MODEL_EVALUATION"
                ),
            }
            for row in core_rows
            if group_by_sample[str(row["sample_id"])] in reserved_groups
        ]
        reservation_path = private_dir / "group_reservations.jsonl"
        reservation_count = write_jsonl(
            reservation_path, reservation_rows
        )

        provenance_dir = temp_root / "provenance"
        provenance_dir.mkdir(parents=True)
        selection_code_path = provenance_dir / "prepare_reference.py"
        common_code_path = provenance_dir / "common.py"
        package_code_path = provenance_dir / "__init__.py"
        shutil.copyfile(Path(__file__).resolve(), selection_code_path)
        shutil.copyfile(
            Path(__file__).resolve().with_name("common.py"),
            common_code_path,
        )
        shutil.copyfile(
            Path(__file__).resolve().with_name("__init__.py"),
            package_code_path,
        )

        public_forbidden_fields = sorted(
            {
                "sample_id",
                "review_id",
                "rating",
                "category",
                "product_id",
                "seller_id",
                "shop_id",
                "source_url",
                "query",
                "sku_info",
                "collection_transport",
                "curation_status",
                "cleaning_flags",
                "panel",
                "selection_bin",
                "old_labels",
                "human_labels",
                "llm_labels",
                "model_predictions",
            }
        )
        manifest: dict[str, Any] = {
            "artifact_type": "DOUBLE_BLIND_HUMAN_ABSA_REFERENCE_INPUT",
            "status": "LOCKED_INPUT_PENDING_TWO_HUMAN_ANNOTATIONS",
            "selection_spec_version": SELECTION_SPEC_VERSION,
            "reference_id": reference_id,
            "built_at": built_at,
            "source_release": {
                "release_id": release_manifest["release_id"],
                "release_name": release_manifest["release_name"],
                "clean_core_records": len(core_rows),
            },
            "source_sha256": source_hashes,
            "runtime": {
                "python": sys.version.split()[0],
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
            "guideline": {
                "document_id": "ABSA-ANNOTATION-GUIDELINE-V2",
                "version": "2.0.0",
                "sha256": source_hashes["guideline"],
            },
            "counts": {
                "reference_records": len(selected),
                "representative": len(representative),
                "challenge": len(challenge),
                "annotator_a_records": len(assignment_a["records"]),
                "annotator_b_records": len(assignment_b["records"]),
                "reserved_group_records_if_model_evaluation": (
                    reservation_count
                ),
                "remaining_core_if_all_groups_reserved": (
                    len(core_rows) - reservation_count
                ),
            },
            "selection_design": {
                "representative": {
                    "target": REPRESENTATIVE_COUNT,
                    "method": (
                        "Hamilton proportional rating then "
                        "category-by-transport; deterministic hash rank"
                    ),
                    "group_cap": REPRESENTATIVE_GROUP_CAP,
                    "excluded_exact_rows": [
                        "all 450 LLM pilot rows",
                        "250 semantic-holdout rows",
                        "non-bijective curated text hashes",
                    ],
                    "diagnostics": representative_diagnostics,
                },
                "challenge": {
                    "target": CHALLENGE_COUNT,
                    "frame": "frozen 450-row LLM pilot candidate",
                    "exclusive_priority_quotas": dict(
                        CHALLENGE_BIN_QUOTAS
                    ),
                    "category_cap": CHALLENGE_CATEGORY_CAP,
                    "group_cap": 1,
                    "cue_warning": (
                        "Bins are coverage proxies, not human-confirmed "
                        "aspect/polarity labels."
                    ),
                    "diagnostics": challenge_diagnostics,
                },
                "analysis_rule": (
                    "Report representative and challenge metrics separately. "
                    "Do not report a single population estimate from all 200 "
                    "without a preregistered weighting rule."
                ),
            },
            "blindness": {
                "public_assignment_root_fields": sorted(
                    assignment_a.keys()
                ),
                "public_record_fields": [
                    "annotation_id",
                    "reviewContent",
                    "review_text_sha256",
                ],
                "forbidden_fields": public_forbidden_fields,
                "same_source_set_for_a_and_b": True,
                "different_order_for_a_and_b": True,
                "disjoint_opaque_ids_for_a_and_b": True,
            },
            "distributions": {
                "representative": {
                    "rating": distribution(
                        representative, lambda row: row["rating"]
                    ),
                    "category": distribution(
                        representative,
                        lambda row: row.get("category") or "<blank>",
                    ),
                    "transport": distribution(
                        representative,
                        lambda row: row.get("collection_transport")
                        or "<blank>",
                    ),
                    "leakage_groups": len(representative_groups),
                },
                "challenge": {
                    "rating": distribution(
                        challenge, lambda row: row["rating"]
                    ),
                    "category": distribution(
                        challenge,
                        lambda row: row.get("category") or "<blank>",
                    ),
                    "transport": distribution(
                        challenge,
                        lambda row: row.get("collection_transport")
                        or "<blank>",
                    ),
                    "selection_bin": distribution(
                        challenge, lambda row: row["_selection_bin"]
                    ),
                    "leakage_groups": len(challenge_groups),
                },
            },
            "protocol_constraints": [
                "A and B independently annotate all 200 before adjudication.",
                "A and B must not share files, browser profiles, or labels.",
                "Practice/qualification samples must be separate from this set.",
                "IAA is calculated on A/B raw labels before adjudication.",
                "If this reference is used for model evaluation, reserve every "
                "row listed in private/group_reservations.jsonl from training.",
                "LLM output is never displayed to annotators.",
            ],
            "artifacts": [],
        }

        artifacts = [
            artifact_entry(
                assignment_a_path,
                temp_root,
                records=REFERENCE_COUNT,
                visibility="ANNOTATOR_A",
            ),
            artifact_entry(
                assignment_b_path,
                temp_root,
                records=REFERENCE_COUNT,
                visibility="ANNOTATOR_B",
            ),
            artifact_entry(
                guideline_output,
                temp_root,
                visibility="PUBLIC_TO_ANNOTATORS",
            ),
            artifact_entry(
                crosswalk_path,
                temp_root,
                records=crosswalk_count,
                visibility="CURATOR_PRIVATE",
            ),
            artifact_entry(
                selection_ledger_path,
                temp_root,
                records=ledger_count,
                visibility="CURATOR_PRIVATE",
            ),
            artifact_entry(
                reservation_path,
                temp_root,
                records=reservation_count,
                visibility="CURATOR_PRIVATE",
            ),
            artifact_entry(
                selection_code_path,
                temp_root,
                visibility="PUBLIC_AUDIT",
            ),
            artifact_entry(
                common_code_path,
                temp_root,
                visibility="PUBLIC_AUDIT",
            ),
            artifact_entry(
                package_code_path,
                temp_root,
                visibility="PUBLIC_AUDIT",
            ),
        ]
        manifest["artifacts"] = artifacts
        manifest_path = temp_root / "manifest.json"
        write_json(manifest_path, manifest)
        all_checksum_paths = [
            *(temp_root / item["path"] for item in artifacts),
            manifest_path,
        ]
        checksum_path = temp_root / "SHA256SUMS.txt"
        checksum_path.write_text(
            "".join(
                f"{sha256_file(path)}  "
                f"{path.relative_to(temp_root).as_posix()}\n"
                for path in sorted(
                    all_checksum_paths,
                    key=lambda value: value.relative_to(
                        temp_root
                    ).as_posix(),
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        temp_root.replace(output)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return {
        "reference_id": reference_id,
        "status": "LOCKED_INPUT_PENDING_TWO_HUMAN_ANNOTATIONS",
        "counts": {
            "representative": len(representative),
            "challenge": len(challenge),
            "total": len(selected),
            "reserved_group_records_if_model_evaluation": len(
                reservation_rows
            ),
        },
        "assignment_a": str(
            output / "assignments" / "annotator_a.assignment.json"
        ),
        "assignment_b": str(
            output / "assignments" / "annotator_b.assignment.json"
        ),
        "manifest": str(output / "manifest.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the locked 150-representative + 50-challenge "
            "double-blind human ABSA reference package."
        )
    )
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument(
        "--semantic-holdout",
        type=Path,
        default=DEFAULT_SEMANTIC_HOLDOUT,
    )
    parser.add_argument("--guideline", type=Path, default=DEFAULT_GUIDELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--built-at",
        help=(
            "Reuse an ISO-8601 build timestamp for byte-reproducible "
            "rebuild checks."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = prepare(
        release=args.release,
        pilot=args.pilot,
        semantic_holdout=args.semantic_holdout,
        guideline=args.guideline,
        output=args.output,
        built_at=args.built_at,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
