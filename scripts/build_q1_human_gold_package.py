"""Build the frozen Q1 1,200-review double-blind annotation package."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_annotation_ui import ASSIGNMENT_SCHEMA_VERSION
from human_annotation_ui.common import (
    ASSIGNMENT_PAYLOAD_FIELDS,
    canonical_json,
    sha256_file,
    sha256_text,
    verify_assignment,
)
from scripts.design_q1_human_gold_sampling import (
    build_groups,
    challenge_bin,
)

CONFIG = ROOT / "configs" / "q1_human_gold_sampling_v1_1.json"
PLAN = (
    ROOT / "docs" / "audits"
    / "q1_human_gold_sampling_plan_v1_1_20260728"
)
SNAPSHOT = ROOT / "data" / "releases" / "q1_dataset_snapshot_v1_20260728"
CORPUS_AUDIT = (
    ROOT / "docs" / "audits"
    / "q1_collection_corpus_audit_v1_20260728"
)
OUTPUT = (
    ROOT / "data" / "annotations"
    / "q1_human_gold_1200_v1_20260728"
)
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

PACKAGE_SCHEMA = "q1-human-gold-package/1.0.0"
SELECTION_SCHEMA = "q1-human-gold-selection/1.0.0"
SELECTION_SALT = "q1-human-gold-v1.1-20260728"
CHALLENGE_SALT = f"{SELECTION_SALT}:challenge"
CORE_SALT = f"{SELECTION_SALT}:core"
ROLE_SALT = f"{SELECTION_SALT}:role"
SELENIUM_TARGETS = {
    "beauty_personal_care": 10,
    "electronics": 5,
    "fashion": 0,
    "food_beverage": 0,
    "home_appliances": 14,
    "home_living": 13,
    "mother_baby": 8,
    "sports_outdoors": 0,
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object: {path}:{line_number}")
            yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
            count += 1
    return count


def stable_rank(salt: str, *parts: object) -> str:
    return sha256_text("\0".join((salt, *(str(part) for part in parts))))


def load_sampling_frame() -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    set[str],
]:
    curation_rows: list[dict[str, Any]] = []
    for release in CURATION_RELEASES:
        curation_rows.extend(read_jsonl(release / "curation_records.jsonl"))
    curation_by_id = {row["sample_id"]: row for row in curation_rows}
    group_by_sample, members_by_group = build_groups(curation_rows)
    prior_ids = {
        row["sample_id"] for row in read_jsonl(PRIOR_RESERVATIONS)
    }
    prior_reference_ids = {
        row["sample_id"] for row in read_jsonl(PRIOR_CROSSWALK)
    }
    contaminated_groups = {
        group_by_sample[sample_id]
        for sample_id in prior_ids | prior_reference_ids
        if sample_id in group_by_sample
    }

    pseudo_rows: list[dict[str, Any]] = []
    major_ids: set[str] = set()
    for package in CRAWLED_PACKAGES:
        rows = list(
            read_jsonl(package / "final" / "ai_pseudo_labels.jsonl")
        )
        for row in rows:
            row["_source_package"] = package.name
        pseudo_rows.extend(rows)
        sample_by_annotation = {
            row["annotation_id"]: row["sample_id"] for row in rows
        }
        sample_by_hash = {
            row["review_text_sha256"]: row["sample_id"] for row in rows
        }
        ledger = (
            package / "audits" / "semantic_audit_60_v1"
            / "audit_ledger.jsonl"
        )
        for audit_row in read_jsonl(ledger):
            if audit_row["audit_severity"] != "MAJOR":
                continue
            sample_id = audit_row.get("sample_id")
            if not isinstance(sample_id, str):
                sample_id = sample_by_annotation.get(
                    audit_row.get("annotation_id")
                )
            if not isinstance(sample_id, str):
                sample_id = sample_by_hash.get(
                    audit_row.get("review_text_sha256")
                )
            if not isinstance(sample_id, str):
                raise ValueError("Cannot join semantic-major row")
            major_ids.add(sample_id)
    frame = [
        row
        for row in pseudo_rows
        if group_by_sample[row["sample_id"]] not in contaminated_groups
    ]
    if (
        len(pseudo_rows) != 26_130
        or len(frame) != 15_553
        or len(major_ids) != 53
        or len(major_ids & {row["sample_id"] for row in frame}) != 23
    ):
        raise ValueError("Frozen sampling-frame inventory drifted")
    return (
        frame,
        group_by_sample,
        members_by_group,
        curation_by_id,
        major_ids,
    )


def choose_challenge(
    frame: Sequence[dict[str, Any]],
    *,
    group_by_sample: Mapping[str, str],
    major_ids: set[str],
    quotas: Mapping[str, int],
) -> tuple[list[tuple[dict[str, Any], str]], Counter[str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    selected_ids: set[str] = set()
    group_total: Counter[str] = Counter()
    selenium_by_group = Counter(
        group_by_sample[row["sample_id"]]
        for row in frame
        if row["source"]["collection_transport"] == "selenium_dom"
    )
    for bin_name, target in quotas.items():
        candidates = [
            row
            for row in frame
            if challenge_bin(row, major_ids) == bin_name
            and row["sample_id"] not in selected_ids
        ]
        chosen = 0
        while chosen < target:
            feasible = [
                row
                for row in candidates
                if row["sample_id"] not in selected_ids
                and group_total[group_by_sample[row["sample_id"]]] < 12
            ]
            if not feasible:
                raise ValueError(
                    f"Cannot close challenge bin {bin_name}: "
                    f"{chosen}/{target}"
                )
            yield_by_group = Counter(
                group_by_sample[row["sample_id"]] for row in feasible
            )
            best_group = min(
                yield_by_group,
                key=lambda group_id: (
                    1 if selenium_by_group[group_id] else 0,
                    0 if group_total[group_id] else 1,
                    -min(
                        yield_by_group[group_id],
                        12 - group_total[group_id],
                    ),
                    stable_rank(CHALLENGE_SALT, bin_name, group_id),
                ),
            )
            capacity = min(
                target - chosen,
                12 - group_total[best_group],
            )
            rows = sorted(
                (
                    row
                    for row in feasible
                    if group_by_sample[row["sample_id"]] == best_group
                ),
                key=lambda row: stable_rank(
                    CHALLENGE_SALT,
                    bin_name,
                    row["sample_id"],
                ),
            )[:capacity]
            if not rows:
                raise AssertionError("Selected an empty challenge group")
            for row in rows:
                selected.append((row, bin_name))
                selected_ids.add(row["sample_id"])
                group_total[best_group] += 1
                chosen += 1
    if len(selected) != 400:
        raise ValueError(f"Challenge selection did not close: {len(selected)}")
    return selected, group_total


def condition_names(
    row: Mapping[str, Any],
    deficits: Mapping[str, int],
) -> tuple[str, ...]:
    names: list[str] = []
    rating = int(row["source"]["rating"])
    origin = str(row["source"]["curation_status"])
    if deficits["rating_1_2"] > 0 and rating in {1, 2}:
        names.append("rating_1_2")
    if deficits["rating_3_4"] > 0 and rating in {3, 4}:
        names.append("rating_3_4")
    if (
        deficits["selenium_dom"] > 0
        and row["source"]["collection_transport"] == "selenium_dom"
    ):
        names.append("selenium_dom")
    if (
        deficits["clean_origin"] > 0
        and origin in {"KEEP", "KEEP_CLEANED"}
    ):
        names.append("clean_origin")
    if deficits["quarantine_origin"] > 0 and origin == "QUARANTINE":
        names.append("quarantine_origin")
    return tuple(names)


def choose_core(
    frame: Sequence[dict[str, Any]],
    *,
    categories: Sequence[str],
    selected_ids: set[str],
    group_by_sample: Mapping[str, str],
    group_total: Counter[str],
) -> tuple[list[tuple[dict[str, Any], str]], Counter[str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    group_core: Counter[str] = Counter()
    for category in categories:
        candidates = [
            row
            for row in frame
            if row["source"]["category"] == category
            and row["sample_id"] not in selected_ids
        ]
        category_selected: list[dict[str, Any]] = []
        deficits: dict[str, int] = {
            "rating_1_2": 10,
            "rating_3_4": 10,
            "selenium_dom": SELENIUM_TARGETS[category],
            "clean_origin": 40,
            "quarantine_origin": 40,
        }
        while any(value > 0 for value in deficits.values()):
            feasible = [
                row
                for row in candidates
                if row["sample_id"] not in selected_ids
                and group_core[group_by_sample[row["sample_id"]]] < 8
                and group_total[group_by_sample[row["sample_id"]]] < 12
                and condition_names(row, deficits)
            ]
            if not feasible or len(category_selected) >= 100:
                raise ValueError(
                    f"Cannot close core margins for {category}: {deficits}"
                )
            available_by_condition = Counter()
            for row in feasible:
                available_by_condition.update(
                    condition_names(row, deficits)
                )

            def candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
                names = condition_names(row, deficits)
                scarcity = sum(
                    1.0 / available_by_condition[name] for name in names
                )
                group_id = group_by_sample[str(row["sample_id"])]
                return (
                    0 if "selenium_dom" in names else 1,
                    -len(names),
                    -scarcity,
                    0 if group_total[group_id] else 1,
                    -group_total[group_id],
                    stable_rank(CORE_SALT, category, row["sample_id"]),
                )

            row = min(feasible, key=candidate_key)
            sample_id = row["sample_id"]
            group_id = group_by_sample[sample_id]
            for name in condition_names(row, deficits):
                deficits[name] = max(0, deficits[name] - 1)
            selected_ids.add(sample_id)
            category_selected.append(row)
            selected.append((row, "CATEGORY_BALANCED_MARGINS"))
            group_core[group_id] += 1
            group_total[group_id] += 1

        while len(category_selected) < 100:
            feasible = [
                row
                for row in candidates
                if row["sample_id"] not in selected_ids
                and group_core[group_by_sample[row["sample_id"]]] < 8
                and group_total[group_by_sample[row["sample_id"]]] < 12
            ]
            if not feasible:
                raise ValueError(
                    f"Cannot fill core category {category}: "
                    f"{len(category_selected)}/100"
                )
            row = min(
                feasible,
                key=lambda item: (
                    0
                    if group_total[group_by_sample[item["sample_id"]]]
                    else 1,
                    -group_total[group_by_sample[item["sample_id"]]],
                    stable_rank(CORE_SALT, category, item["sample_id"]),
                ),
            )
            sample_id = row["sample_id"]
            group_id = group_by_sample[sample_id]
            selected_ids.add(sample_id)
            category_selected.append(row)
            selected.append((row, "CATEGORY_BALANCED_FILL"))
            group_core[group_id] += 1
            group_total[group_id] += 1
    if len(selected) != 800:
        raise ValueError(f"Core selection did not close: {len(selected)}")
    return selected, group_core


def build_assignment(
    *,
    role: str,
    reference_id: str,
    created_at: str,
    guideline_hash: str,
    selected_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, tuple[str, int]]]:
    ordered = sorted(
        selected_rows,
        key=lambda row: stable_rank(ROLE_SALT, role, row["sample_id"]),
    )
    crosswalk: dict[str, tuple[str, int]] = {}
    public_rows: list[dict[str, Any]] = []
    for order, row in enumerate(ordered, 1):
        annotation_id = (
            f"q1{role.casefold()}-"
            + stable_rank(
                f"{ROLE_SALT}:annotation-id",
                reference_id,
                role,
                row["sample_id"],
            )[:24]
        )
        crosswalk[row["sample_id"]] = (annotation_id, order)
        public_rows.append(
            {
                "annotation_id": annotation_id,
                "reviewContent": row["reviewContent"],
                "review_text_sha256": row["review_text_sha256"],
            }
        )
    assignment = {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "assignment_id": (
            f"{reference_id}-{role.casefold()}-"
            + stable_rank(ROLE_SALT, reference_id, role)[:12]
        ),
        "reference_id": reference_id,
        "role": role,
        "item_count": len(public_rows),
        "guideline": {
            "document_id": "ABSA-ANNOTATION-GUIDELINE-V2",
            "version": "2.0.0",
            "sha256": guideline_hash,
        },
        "created_at": created_at,
        "records": public_rows,
    }
    payload = {
        key: assignment[key] for key in ASSIGNMENT_PAYLOAD_FIELDS
    }
    assignment["assignment_payload_sha256"] = sha256_text(
        canonical_json(payload)
    )
    return verify_assignment(assignment), crosswalk


def artifact(
    path: Path,
    root: Path,
    *,
    records: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        row["records"] = records
    return row


def render_readme(
    *,
    reference_id: str,
    selected_groups: int,
    reservation_records: int,
) -> str:
    return f"""# Q1 Human-Gold 1,200 v1

Reference ID: `{reference_id}`

This immutable package contains 1,200 unique reviews: 800
`CORE_BALANCED` and 400 `CHALLENGE_DIAGNOSTIC`. Both roles annotate all
1,200 reviews independently. Public assignments contain only opaque IDs,
review text and review-text SHA-256.

- Selected leakage groups: {selected_groups}
- Reserved records across those groups: {reservation_records}
- Existing 200-review package is calibration-only and is not included.
- AI pseudo-labels were used privately for deterministic sampling strata;
  they do not appear in either public assignment.
- Do not open an AI-review view for these samples until both A and B FINAL
  exports have been independently validated and frozen.

The browser draft is not a research artifact. A FINAL export must pass
`human_annotation_ui.validate_export` before IAA is computed. IAA must be
published before expert adjudication.
"""


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite package: {OUTPUT}")
    config = read_json(CONFIG)
    plan_manifest = read_json(PLAN / "manifest.json")
    snapshot_manifest = read_json(SNAPSHOT / "manifest.json")
    if (
        plan_manifest["status"] != "VALID_FEASIBLE"
        or plan_manifest["snapshot_id"] != snapshot_manifest["snapshot_id"]
        or config["snapshot_id"] != snapshot_manifest["snapshot_id"]
    ):
        raise ValueError("Plan/snapshot/config binding mismatch")
    (
        frame,
        group_by_sample,
        members_by_group,
        curation_by_id,
        major_ids,
    ) = load_sampling_frame()
    challenge, group_total = choose_challenge(
        frame,
        group_by_sample=group_by_sample,
        major_ids=major_ids,
        quotas=config["panels"]["CHALLENGE_DIAGNOSTIC"][
            "ordered_disjoint_bin_quotas"
        ],
    )
    selected_ids = {row["sample_id"] for row, _ in challenge}
    core, group_core = choose_core(
        frame,
        categories=config["panels"]["CORE_BALANCED"][
            "mapped_categories"
        ],
        selected_ids=selected_ids,
        group_by_sample=group_by_sample,
        group_total=group_total,
    )
    selected_pairs = [
        *((row, "CHALLENGE_DIAGNOSTIC", bin_name) for row, bin_name in challenge),
        *((row, "CORE_BALANCED", bin_name) for row, bin_name in core),
    ]
    selected_rows = [row for row, _, _ in selected_pairs]
    selected_ids = {row["sample_id"] for row in selected_rows}
    selected_groups = {
        group_by_sample[sample_id] for sample_id in selected_ids
    }
    if (
        len(selected_rows) != 1_200
        or len(selected_ids) != 1_200
        or len(selected_groups)
        > config["leakage_grouping"]["maximum_selected_groups"]
        or max(group_total.values()) > 12
        or max(group_core.values()) > 8
    ):
        raise ValueError("Selection/group-cap closure failed")

    created_at = datetime.now(timezone.utc).isoformat()
    selection_digest = sha256_text(
        "\n".join(sorted(selected_ids))
        + "\0"
        + plan_manifest["plan_id"]
    )
    reference_id = f"q1-human-gold-1200-{selection_digest[:16]}"
    guideline_hash = sha256_file(GUIDELINE)
    assignment_a, crosswalk_a = build_assignment(
        role="A",
        reference_id=reference_id,
        created_at=created_at,
        guideline_hash=guideline_hash,
        selected_rows=selected_rows,
    )
    assignment_b, crosswalk_b = build_assignment(
        role="B",
        reference_id=reference_id,
        created_at=created_at,
        guideline_hash=guideline_hash,
        selected_rows=selected_rows,
    )
    if (
        [row["review_text_sha256"] for row in assignment_a["records"]]
        == [row["review_text_sha256"] for row in assignment_b["records"]]
        or {
            row["annotation_id"] for row in assignment_a["records"]
        }
        & {row["annotation_id"] for row in assignment_b["records"]}
    ):
        raise ValueError("Role order/opaque-ID independence failed")

    panel_by_sample = {
        row["sample_id"]: (panel, selection_bin)
        for row, panel, selection_bin in selected_pairs
    }
    selected_rank = {
        sample_id: index
        for index, sample_id in enumerate(
            sorted(
                selected_ids,
                key=lambda value: stable_rank(
                    f"{SELECTION_SALT}:ledger", value
                ),
            ),
            1,
        )
    }
    selection_rows: list[dict[str, Any]] = []
    crosswalk_rows: list[dict[str, Any]] = []
    pseudo_by_id = {row["sample_id"]: row for row in selected_rows}
    for sample_id in sorted(selected_ids):
        row = pseudo_by_id[sample_id]
        panel, selection_bin = panel_by_sample[sample_id]
        group_id = group_by_sample[sample_id]
        base = {
            "sample_id": sample_id,
            "review_text_sha256": row["review_text_sha256"],
            "panel": panel,
            "selection_bin": selection_bin,
            "selection_rank": selected_rank[sample_id],
            "leakage_group_id": group_id,
            "leakage_group_size": len(members_by_group[group_id]),
            "category": row["source"]["category"],
            "rating": row["source"]["rating"],
            "collection_transport": row["source"]["collection_transport"],
            "curation_status": row["source"]["curation_status"],
            "product_id": row["source"]["product_id"],
            "source_package": row["_source_package"],
            "pseudo_annotation_status": row["annotation"][
                "annotation_status"
            ],
            "is_semantic_audit_major": sample_id in major_ids,
        }
        selection_rows.append(
            {
                "schema_version": SELECTION_SCHEMA,
                **base,
            }
        )
        crosswalk_rows.append(
            {
                **base,
                "reference_id": reference_id,
                "annotator_a_id": crosswalk_a[sample_id][0],
                "annotator_a_order": crosswalk_a[sample_id][1],
                "annotator_b_id": crosswalk_b[sample_id][0],
                "annotator_b_order": crosswalk_b[sample_id][1],
            }
        )

    reservation_rows: list[dict[str, Any]] = []
    for group_id in sorted(selected_groups):
        for sample_id in members_by_group[group_id]:
            curation = curation_by_id[sample_id]
            selected = sample_id in selected_ids
            panel = panel_by_sample[sample_id][0] if selected else None
            reservation_rows.append(
                {
                    "sample_id": sample_id,
                    "review_text_sha256": curation[
                        "curated_text_sha256"
                    ],
                    "leakage_group_id": group_id,
                    "is_gold_selected_row": selected,
                    "gold_panel": panel,
                    "reservation_policy": (
                        "EXCLUDE_FROM_ALL_FUTURE_MODEL_TRAINING_AND_"
                        "AI_REVIEW_UNTIL_BLIND_GOLD_IS_FROZEN"
                    ),
                    "reference_id": reference_id,
                }
            )

    core_rows = [row for row, panel, _ in selected_pairs if panel == "CORE_BALANCED"]
    challenge_rows = [
        row for row, panel, _ in selected_pairs
        if panel == "CHALLENGE_DIAGNOSTIC"
    ]
    category_counts = Counter(row["source"]["category"] for row in core_rows)
    challenge_counts = Counter(
        selection_bin
        for _, panel, selection_bin in selected_pairs
        if panel == "CHALLENGE_DIAGNOSTIC"
    )
    measured = {
        "selected_records": len(selected_rows),
        "selected_unique_text_hashes": len(
            {row["review_text_sha256"] for row in selected_rows}
        ),
        "core_records": len(core_rows),
        "challenge_records": len(challenge_rows),
        "selected_groups": len(selected_groups),
        "reserved_group_records": len(reservation_rows),
        "maximum_total_records_in_selected_group": max(
            group_total.values()
        ),
        "maximum_core_records_in_selected_group": max(
            group_core.values()
        ),
        "core_category_counts": dict(sorted(category_counts.items())),
        "core_rating_1_2_by_category": {
            category: sum(
                row["source"]["rating"] in {1, 2}
                for row in core_rows
                if row["source"]["category"] == category
            )
            for category in sorted(category_counts)
        },
        "core_rating_3_4_by_category": {
            category: sum(
                row["source"]["rating"] in {3, 4}
                for row in core_rows
                if row["source"]["category"] == category
            )
            for category in sorted(category_counts)
        },
        "core_selenium_dom": sum(
            row["source"]["collection_transport"] == "selenium_dom"
            for row in core_rows
        ),
        "core_clean_origin": sum(
            row["source"]["curation_status"] in {"KEEP", "KEEP_CLEANED"}
            for row in core_rows
        ),
        "core_quarantine_origin": sum(
            row["source"]["curation_status"] == "QUARANTINE"
            for row in core_rows
        ),
        "challenge_bin_counts": dict(sorted(challenge_counts.items())),
        "role_a_b_same_text_set": (
            {
                row["review_text_sha256"] for row in assignment_a["records"]
            }
            == {
                row["review_text_sha256"] for row in assignment_b["records"]
            }
        ),
        "role_a_b_identical_order": False,
        "role_a_b_annotation_id_overlap": 0,
        "prior_reference_or_reserved_group_overlap": 0,
        "old_records_included": 0,
        "confirmed_duplicates_included": 0,
    }

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    )
    try:
        paths: list[tuple[Path, int | None]] = []
        assignment_a_path = (
            temporary / "assignments" / "annotator_a.assignment.json"
        )
        assignment_b_path = (
            temporary / "assignments" / "annotator_b.assignment.json"
        )
        write_json(assignment_a_path, assignment_a)
        write_json(assignment_b_path, assignment_b)
        paths.extend(
            [
                (assignment_a_path, 1_200),
                (assignment_b_path, 1_200),
            ]
        )
        selection_path = temporary / "private" / "selection_ledger.jsonl"
        crosswalk_path = temporary / "private" / "crosswalk.jsonl"
        reservation_path = (
            temporary / "private" / "group_reservations.jsonl"
        )
        paths.extend(
            [
                (
                    selection_path,
                    write_jsonl(selection_path, selection_rows),
                ),
                (
                    crosswalk_path,
                    write_jsonl(crosswalk_path, crosswalk_rows),
                ),
                (
                    reservation_path,
                    write_jsonl(reservation_path, reservation_rows),
                ),
            ]
        )
        guideline_path = temporary / "ABSA_ANNOTATION_GUIDELINE_V2.md"
        shutil.copy2(GUIDELINE, guideline_path)
        paths.append((guideline_path, None))
        report = {
            "schema_version": PACKAGE_SCHEMA,
            "artifact_type": "Q1_DOUBLE_BLIND_HUMAN_GOLD_PACKAGE",
            "status": "BUILT_PENDING_HUMAN_ANNOTATION",
            "created_at": created_at,
            "reference_id": reference_id,
            "snapshot_id": snapshot_manifest["snapshot_id"],
            "plan_id": plan_manifest["plan_id"],
            "selection_seed": config["selection_seed"],
            "guideline_sha256": guideline_hash,
            "measured": measured,
            "executed": [
                "Rebuilt leakage components from frozen curation records.",
                "Excluded every component connected to the prior calibration reserve.",
                "Selected ordered-disjoint challenge quotas deterministically.",
                "Selected category/rating/transport/curation-balanced core deterministically.",
                "Created role-specific opaque IDs and independent role order.",
                "Reserved every record in every selected leakage component.",
            ],
            "not_executed": [
                "No human annotation has been performed.",
                "No IAA has been calculated.",
                "No expert adjudication has been performed.",
                "No dev/test split has been published.",
            ],
            "next_dependency": (
                "Validate this package independently, integrate it into "
                "the workbench, then obtain two blind FINAL exports."
            ),
        }
        report_path = temporary / "selection_report.json"
        write_json(report_path, report)
        paths.append((report_path, None))
        readme_path = temporary / "README.md"
        readme_path.write_text(
            render_readme(
                reference_id=reference_id,
                selected_groups=len(selected_groups),
                reservation_records=len(reservation_rows),
            ),
            encoding="utf-8",
            newline="\n",
        )
        paths.append((readme_path, None))

        provenance_sources: list[tuple[Path, str]] = [
            (
                Path(__file__).resolve(),
                "provenance/software/build_q1_human_gold_package.py",
            ),
            (
                CONFIG,
                "provenance/config/q1_human_gold_sampling_v1.json",
            ),
            (
                PLAN / "manifest.json",
                "provenance/sampling_plan/manifest.json",
            ),
            (
                PLAN / "report.json",
                "provenance/sampling_plan/report.json",
            ),
            (
                SNAPSHOT / "manifest.json",
                "provenance/snapshot/manifest.json",
            ),
            (
                CORPUS_AUDIT / "manifest.json",
                "provenance/corpus_audit/manifest.json",
            ),
        ]
        for package in CRAWLED_PACKAGES:
            provenance_sources.extend(
                [
                    (
                        package / "final" / "manifest.json",
                        f"provenance/source_packages/{package.name}/manifest.json",
                    ),
                    (
                        package / "final" / "SHA256SUMS.txt",
                        f"provenance/source_packages/{package.name}/SHA256SUMS.txt",
                    ),
                ]
            )
        for source, relative in provenance_sources:
            destination = temporary.joinpath(*Path(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            paths.append((destination, None))

        artifacts = sorted(
            (
                artifact(path, temporary, records=records)
                for path, records in paths
            ),
            key=lambda row: row["path"],
        )
        manifest = {
            "schema_version": "q1-human-gold-package-manifest/1.0.0",
            "artifact_type": "Q1_DOUBLE_BLIND_HUMAN_GOLD_PACKAGE_RELEASE",
            "status": "BUILT_PENDING_HUMAN_ANNOTATION",
            "package_id": (
                "q1-human-gold-package-"
                + sha256_text(
                    reference_id
                    + "\0"
                    + assignment_a["assignment_payload_sha256"]
                    + "\0"
                    + assignment_b["assignment_payload_sha256"]
                )[:16]
            ),
            "created_at": created_at,
            "reference_id": reference_id,
            "snapshot_id": snapshot_manifest["snapshot_id"],
            "plan_id": plan_manifest["plan_id"],
            "selection_report_sha256": sha256_file(report_path),
            "artifacts": artifacts,
        }
        manifest_path = temporary / "manifest.json"
        write_json(manifest_path, manifest)
        sums = {
            item["path"]: item["sha256"] for item in artifacts
        }
        sums["manifest.json"] = sha256_file(manifest_path)
        sums_path = temporary / "SHA256SUMS.txt"
        sums_path.write_text(
            "".join(
                f"{digest}  {relative}\n"
                for relative, digest in sorted(sums.items())
            ),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(OUTPUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "BUILT_PENDING_HUMAN_ANNOTATION",
        "output": str(OUTPUT),
        "package_id": manifest["package_id"],
        "reference_id": reference_id,
        **measured,
        "manifest_sha256": sha256_file(OUTPUT / "manifest.json"),
        "checksums_sha256": sha256_file(OUTPUT / "SHA256SUMS.txt"),
    }


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
