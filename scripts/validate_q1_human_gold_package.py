"""Independently validate the Q1 1,200-review blind package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_annotation_ui.common import (  # noqa: E402
    sha256_file,
    sha256_text,
    verify_assignment,
)


DEFAULT_PACKAGE = (
    ROOT / "data" / "annotations"
    / "q1_human_gold_1200_v1_20260728"
)
PLAN = (
    ROOT / "docs" / "audits"
    / "q1_human_gold_sampling_plan_v1_1_20260728"
)
CONFIG = ROOT / "configs" / "q1_human_gold_sampling_v1_1.json"
GUIDELINE = ROOT / "docs" / "ABSA_ANNOTATION_GUIDELINE_V2.md"
CURATION_RELEASES = (
    ROOT / "data" / "releases"
    / "lazada_vi_absa_curation_v2_1_2_20260725",
    ROOT / "data" / "releases"
    / "lazada_vi_absa_delta_curation_v1_20260728",
)
CRAWLED_PACKAGES = (
    ROOT / "data" / "annotations"
    / "absa_ai_tranche_5000_v1_20260727",
    ROOT / "data" / "annotations"
    / "absa_ai_remainder_8976_v1_20260728",
    ROOT / "data" / "annotations"
    / "absa_ai_delta_v1_20260728",
    ROOT / "data" / "annotations"
    / "absa_ai_quarantine_base_11166_v1_20260728",
    ROOT / "data" / "annotations"
    / "absa_ai_quarantine_delta_375_v1_20260728",
)
PRIOR_RESERVATIONS = (
    ROOT / "data" / "annotations" / "human_reference_v1_20260726"
    / "private" / "group_reservations.jsonl"
)
PRIOR_CROSSWALK = (
    ROOT / "data" / "annotations" / "human_reference_v1_20260726"
    / "private" / "crosswalk.jsonl"
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object: {path}:{line_number}")
            rows.append(value)
    return rows


def read_sums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in rows:
            raise ValueError(f"Duplicate checksum path: {relative}")
        rows[relative] = digest
    return rows


def build_groups(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    eligible = [row for row in rows if row["status"] != "EXCLUDE_AUTO"]
    ids = {row["sample_id"] for row in eligible}
    union_find = UnionFind(ids)
    indexes: tuple[tuple[str, dict[str, str]], ...] = (
        ("product_id", {}),
        ("curated_text_sha256", {}),
        ("template_family_id", {}),
    )
    for row in eligible:
        sample_id = row["sample_id"]
        for field, index in indexes:
            value = str(row.get(field) or "")
            if value:
                prior = index.setdefault(value, sample_id)
                union_find.union(prior, sample_id)
        for field in (
            "representative_sample_id",
            "near_duplicate_representative_sample_id",
        ):
            peer = row.get(field)
            if isinstance(peer, str) and peer in ids:
                union_find.union(sample_id, peer)
    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(ids):
        components[union_find.find(sample_id)].append(sample_id)
    group_by_sample: dict[str, str] = {}
    members_by_group: dict[str, list[str]] = {}
    for members in components.values():
        group_id = "q1lkg-" + sha256_text("\n".join(members))[:20]
        members_by_group[group_id] = members
        for sample_id in members:
            group_by_sample[sample_id] = group_id
    return group_by_sample, members_by_group


def mentioned_labels(row: Mapping[str, Any]) -> list[Any]:
    return [
        aspect["label"]
        for aspect in row["annotation"]["aspects"]
        if aspect["label"] not in {2, None}
    ]


def challenge_class(
    row: Mapping[str, Any],
    major_ids: set[str],
) -> str | None:
    if row["sample_id"] in major_ids:
        return "SEMANTIC_AUDIT_MAJOR"
    status = row["annotation"]["annotation_status"]
    if status == "ESCALATE":
        return "ESCALATE"
    if status == "REJECT_NON_REVIEW":
        return "REJECT_NON_REVIEW"
    labels = mentioned_labels(row)
    if "1, -1" in labels:
        return "MIXED_POLARITY"
    if 0 in labels:
        return "NEUTRAL_POLARITY"
    if -1 in labels or any(
        aspect["aspect"] in {"Bảo hành & Đổi trả", "Tính xác thực"}
        and aspect["label"] not in {2, None}
        for aspect in row["annotation"]["aspects"]
    ):
        return "NEGATIVE_OR_RARE_ASPECT"
    return None


def validate(package_root: Path) -> dict[str, Any]:
    package_root = package_root.resolve()
    manifest_path = package_root / "manifest.json"
    sums_path = package_root / "SHA256SUMS.txt"
    report_path = package_root / "selection_report.json"
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    config = read_json(CONFIG)
    plan = read_json(PLAN / "manifest.json")
    if (
        manifest.get("schema_version")
        != "q1-human-gold-package-manifest/1.0.0"
        or manifest.get("artifact_type")
        != "Q1_DOUBLE_BLIND_HUMAN_GOLD_PACKAGE_RELEASE"
        or manifest.get("status") != "BUILT_PENDING_HUMAN_ANNOTATION"
        or report.get("status") != "BUILT_PENDING_HUMAN_ANNOTATION"
        or manifest.get("plan_id") != plan["plan_id"]
        or report.get("plan_id") != plan["plan_id"]
        or report.get("selection_seed") != config["selection_seed"]
        or manifest.get("selection_report_sha256")
        != sha256_file(report_path)
    ):
        raise ValueError("Package type/status/plan/report binding mismatch")

    expected_sums = {"manifest.json": sha256_file(manifest_path)}
    expected_files = {"manifest.json", "SHA256SUMS.txt"}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Package artifact inventory is malformed")
    for item in artifacts:
        path = (package_root / item["path"]).resolve()
        try:
            path.relative_to(package_root)
        except ValueError as exc:
            raise ValueError("Package artifact escapes root") from exc
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"Artifact mismatch: {item['path']}")
        expected_sums[item["path"]] = item["sha256"]
        expected_files.add(item["path"])
    if read_sums(sums_path) != expected_sums:
        raise ValueError("Package checksum ledger is not closed")
    actual_files = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Package file inventory is not closed")

    assignment_a = verify_assignment(
        read_json(
            package_root / "assignments"
            / "annotator_a.assignment.json"
        )
    )
    assignment_b = verify_assignment(
        read_json(
            package_root / "assignments"
            / "annotator_b.assignment.json"
        )
    )
    if (
        assignment_a["role"] != "A"
        or assignment_b["role"] != "B"
        or assignment_a["reference_id"] != assignment_b["reference_id"]
        or assignment_a["reference_id"] != manifest["reference_id"]
        or assignment_a["item_count"] != 1_200
        or assignment_b["item_count"] != 1_200
        or assignment_a["guideline"]["sha256"] != sha256_file(GUIDELINE)
        or assignment_b["guideline"]["sha256"] != sha256_file(GUIDELINE)
        or sha256_file(
            package_root / "ABSA_ANNOTATION_GUIDELINE_V2.md"
        )
        != sha256_file(GUIDELINE)
    ):
        raise ValueError("Assignment identity/count/guideline mismatch")
    a_hashes = [
        row["review_text_sha256"] for row in assignment_a["records"]
    ]
    b_hashes = [
        row["review_text_sha256"] for row in assignment_b["records"]
    ]
    a_ids = {row["annotation_id"] for row in assignment_a["records"]}
    b_ids = {row["annotation_id"] for row in assignment_b["records"]}
    if (
        set(a_hashes) != set(b_hashes)
        or len(set(a_hashes)) != 1_200
        or a_hashes == b_hashes
        or a_ids & b_ids
    ):
        raise ValueError("Assignment blindness/order/ID independence failed")
    a_by_id = {
        row["annotation_id"]: row for row in assignment_a["records"]
    }
    b_by_id = {
        row["annotation_id"]: row for row in assignment_b["records"]
    }

    selection = read_jsonl(
        package_root / "private" / "selection_ledger.jsonl"
    )
    crosswalk = read_jsonl(
        package_root / "private" / "crosswalk.jsonl"
    )
    reservations = read_jsonl(
        package_root / "private" / "group_reservations.jsonl"
    )
    if len(selection) != 1_200 or len(crosswalk) != 1_200:
        raise ValueError("Private selection/crosswalk count mismatch")
    selection_by_id = {row["sample_id"]: row for row in selection}
    crosswalk_by_id = {row["sample_id"]: row for row in crosswalk}
    if (
        len(selection_by_id) != 1_200
        or set(selection_by_id) != set(crosswalk_by_id)
    ):
        raise ValueError("Selection/crosswalk ID bijection failed")

    curation_rows: list[dict[str, Any]] = []
    for release in CURATION_RELEASES:
        curation_rows.extend(
            read_jsonl(release / "curation_records.jsonl")
        )
    curation_by_id = {
        row["sample_id"]: row for row in curation_rows
    }
    group_by_sample, members_by_group = build_groups(curation_rows)
    prior_ids = {
        row["sample_id"] for row in read_jsonl(PRIOR_RESERVATIONS)
    } | {
        row["sample_id"] for row in read_jsonl(PRIOR_CROSSWALK)
    }
    prior_groups = {
        group_by_sample[sample_id]
        for sample_id in prior_ids
        if sample_id in group_by_sample
    }

    pseudo_by_id: dict[str, dict[str, Any]] = {}
    major_ids: set[str] = set()
    for source_package in CRAWLED_PACKAGES:
        rows = read_jsonl(
            source_package / "final" / "ai_pseudo_labels.jsonl"
        )
        annotation_to_sample = {
            row["annotation_id"]: row["sample_id"] for row in rows
        }
        hash_to_sample = {
            row["review_text_sha256"]: row["sample_id"] for row in rows
        }
        for row in rows:
            if row["sample_id"] in pseudo_by_id:
                raise ValueError("Pseudo source sample ID overlap")
            row["_source_package"] = source_package.name
            pseudo_by_id[row["sample_id"]] = row
        audits = read_jsonl(
            source_package / "audits" / "semantic_audit_60_v1"
            / "audit_ledger.jsonl"
        )
        for row in audits:
            if row["audit_severity"] != "MAJOR":
                continue
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str):
                sample_id = annotation_to_sample.get(
                    row.get("annotation_id")
                )
            if not isinstance(sample_id, str):
                sample_id = hash_to_sample.get(
                    row.get("review_text_sha256")
                )
            if not isinstance(sample_id, str):
                raise ValueError("Cannot join semantic-major audit row")
            major_ids.add(sample_id)
    if len(pseudo_by_id) != 26_130 or len(major_ids) != 53:
        raise ValueError("Frozen pseudo/major source inventory drifted")

    selected_ids = set(selection_by_id)
    selected_groups = {
        group_by_sample[sample_id] for sample_id in selected_ids
    }
    if selected_groups & prior_groups:
        raise ValueError("Selected group intersects prior calibration reserve")
    if len(selected_groups) > 240:
        raise ValueError("Selected-group cap exceeded")

    panel_counts = Counter()
    challenge_counts = Counter()
    category_counts = Counter()
    group_total = Counter()
    group_core = Counter()
    clean_core = 0
    quarantine_core = 0
    selenium_core = 0
    low_by_category = Counter()
    mid_by_category = Counter()
    for sample_id, ledger in selection_by_id.items():
        if sample_id not in pseudo_by_id or sample_id not in curation_by_id:
            raise ValueError(f"Selected sample missing source: {sample_id}")
        source = pseudo_by_id[sample_id]
        group_id = group_by_sample[sample_id]
        expected_fields = {
            "review_text_sha256": source["review_text_sha256"],
            "leakage_group_id": group_id,
            "leakage_group_size": len(members_by_group[group_id]),
            "category": source["source"]["category"],
            "rating": source["source"]["rating"],
            "collection_transport": source["source"][
                "collection_transport"
            ],
            "curation_status": source["source"]["curation_status"],
            "product_id": source["source"]["product_id"],
            "source_package": source["_source_package"],
            "pseudo_annotation_status": source["annotation"][
                "annotation_status"
            ],
            "is_semantic_audit_major": sample_id in major_ids,
        }
        if any(ledger.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(f"Selection ledger/source mismatch: {sample_id}")
        cross = crosswalk_by_id[sample_id]
        if any(cross.get(key) != ledger.get(key) for key in expected_fields):
            raise ValueError(f"Crosswalk/selection mismatch: {sample_id}")
        a_row = a_by_id.get(cross["annotator_a_id"])
        b_row = b_by_id.get(cross["annotator_b_id"])
        if (
            a_row is None
            or b_row is None
            or a_row["review_text_sha256"] != source["review_text_sha256"]
            or b_row["review_text_sha256"] != source["review_text_sha256"]
            or a_row["reviewContent"] != source["reviewContent"]
            or b_row["reviewContent"] != source["reviewContent"]
            or assignment_a["records"][
                cross["annotator_a_order"] - 1
            ]["annotation_id"]
            != cross["annotator_a_id"]
            or assignment_b["records"][
                cross["annotator_b_order"] - 1
            ]["annotation_id"]
            != cross["annotator_b_id"]
        ):
            raise ValueError(f"Crosswalk/public assignment mismatch: {sample_id}")
        panel = ledger["panel"]
        panel_counts[panel] += 1
        group_total[group_id] += 1
        if panel == "CHALLENGE_DIAGNOSTIC":
            actual_bin = challenge_class(source, major_ids)
            if ledger["selection_bin"] != actual_bin:
                raise ValueError(
                    f"Challenge-bin mismatch for {sample_id}: "
                    f"{ledger['selection_bin']} != {actual_bin}"
                )
            challenge_counts[actual_bin] += 1
        elif panel == "CORE_BALANCED":
            group_core[group_id] += 1
            category = source["source"]["category"]
            rating = source["source"]["rating"]
            origin = source["source"]["curation_status"]
            category_counts[category] += 1
            low_by_category[category] += rating in {1, 2}
            mid_by_category[category] += rating in {3, 4}
            clean_core += origin in {"KEEP", "KEEP_CLEANED"}
            quarantine_core += origin == "QUARANTINE"
            selenium_core += (
                source["source"]["collection_transport"] == "selenium_dom"
            )
        else:
            raise ValueError(f"Invalid panel: {panel}")

    core_spec = config["panels"]["CORE_BALANCED"]
    challenge_spec = config["panels"]["CHALLENGE_DIAGNOSTIC"]
    if (
        panel_counts
        != Counter({"CORE_BALANCED": 800, "CHALLENGE_DIAGNOSTIC": 400})
        or any(
            category_counts[category] != 100
            or low_by_category[category] < 10
            or mid_by_category[category] < 10
            for category in core_spec["mapped_categories"]
        )
        or selenium_core < core_spec["minimum_selenium_dom_total"]
        or clean_core < core_spec["minimum_clean_origin_total"]
        or quarantine_core < core_spec["minimum_quarantine_origin_total"]
        or challenge_counts
        != Counter(challenge_spec["ordered_disjoint_bin_quotas"])
        or max(group_core.values()) > 8
        or max(group_total.values()) > 12
    ):
        raise ValueError("Panel marginal/group-cap validation failed")

    expected_reserved_ids = {
        sample_id
        for group_id in selected_groups
        for sample_id in members_by_group[group_id]
    }
    reservation_by_id = {
        row["sample_id"]: row for row in reservations
    }
    if (
        len(reservation_by_id) != len(reservations)
        or set(reservation_by_id) != expected_reserved_ids
    ):
        raise ValueError("Selected-group reservation closure failed")
    for sample_id, row in reservation_by_id.items():
        curation = curation_by_id[sample_id]
        if (
            row["leakage_group_id"] != group_by_sample[sample_id]
            or row["review_text_sha256"]
            != curation["curated_text_sha256"]
            or row["is_gold_selected_row"] != (sample_id in selected_ids)
        ):
            raise ValueError(f"Reservation/source mismatch: {sample_id}")

    measured = report["measured"]
    expected_measured = {
        "selected_records": 1_200,
        "selected_unique_text_hashes": 1_200,
        "core_records": 800,
        "challenge_records": 400,
        "selected_groups": len(selected_groups),
        "reserved_group_records": len(reservations),
        "maximum_total_records_in_selected_group": max(
            group_total.values()
        ),
        "maximum_core_records_in_selected_group": max(
            group_core.values()
        ),
        "core_category_counts": dict(sorted(category_counts.items())),
        "core_rating_1_2_by_category": dict(sorted(low_by_category.items())),
        "core_rating_3_4_by_category": dict(sorted(mid_by_category.items())),
        "core_selenium_dom": selenium_core,
        "core_clean_origin": clean_core,
        "core_quarantine_origin": quarantine_core,
        "challenge_bin_counts": dict(sorted(challenge_counts.items())),
        "role_a_b_same_text_set": True,
        "role_a_b_identical_order": False,
        "role_a_b_annotation_id_overlap": 0,
        "prior_reference_or_reserved_group_overlap": 0,
        "old_records_included": 0,
        "confirmed_duplicates_included": 0,
    }
    if measured != expected_measured:
        raise ValueError("Selection report measured metrics mismatch")
    return {
        "status": "VALID",
        "package_id": manifest["package_id"],
        "reference_id": manifest["reference_id"],
        "plan_id": manifest["plan_id"],
        "selected_records": 1_200,
        "core_records": 800,
        "challenge_records": 400,
        "selected_groups": len(selected_groups),
        "reserved_group_records": len(reservations),
        "core_selenium_dom": selenium_core,
        "core_clean_origin": clean_core,
        "core_quarantine_origin": quarantine_core,
        "maximum_core_per_group": max(group_core.values()),
        "maximum_total_per_group": max(group_total.values()),
        "manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(sums_path),
        "selection_report_sha256": sha256_file(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.package),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
