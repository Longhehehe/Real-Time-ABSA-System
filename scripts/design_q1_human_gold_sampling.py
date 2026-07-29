"""Preflight and freeze the Q1 human-gold/IAA sampling design."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "q1_human_gold_sampling_v1_1.json"
SNAPSHOT = ROOT / "data" / "releases" / "q1_dataset_snapshot_v1_20260728"
CORPUS_AUDIT = (
    ROOT / "docs" / "audits" / "q1_collection_corpus_audit_v1_20260728"
)
OUTPUT = (
    ROOT / "docs" / "audits"
    / "q1_human_gold_sampling_plan_v1_1_20260728"
)
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
RESERVATIONS = (
    ROOT / "data" / "annotations" / "human_reference_v1_20260726"
    / "private" / "group_reservations.jsonl"
)
REFERENCE_CROSSWALK = (
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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_groups(
    curation_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    eligible = [
        row for row in curation_rows if row["status"] != "EXCLUDE_AUTO"
    ]
    sample_ids = {row["sample_id"] for row in eligible}
    union_find = UnionFind(sample_ids)
    by_product: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    by_template: dict[str, str] = {}
    for row in eligible:
        sample_id = row["sample_id"]
        for value, index in (
            (str(row.get("product_id") or ""), by_product),
            (str(row.get("curated_text_sha256") or ""), by_hash),
            (str(row.get("template_family_id") or ""), by_template),
        ):
            if not value:
                continue
            prior = index.setdefault(value, sample_id)
            union_find.union(prior, sample_id)
        for field in (
            "representative_sample_id",
            "near_duplicate_representative_sample_id",
        ):
            peer = row.get(field)
            if isinstance(peer, str) and peer in sample_ids:
                union_find.union(sample_id, peer)
    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(sample_ids):
        components[union_find.find(sample_id)].append(sample_id)
    group_by_sample: dict[str, str] = {}
    members_by_group: dict[str, list[str]] = {}
    for members in components.values():
        group_id = "q1lkg-" + sha256_text("\n".join(members))[:20]
        members_by_group[group_id] = members
        for sample_id in members:
            group_by_sample[sample_id] = group_id
    return group_by_sample, members_by_group


def labels(row: Mapping[str, Any]) -> list[Any]:
    return [
        aspect["label"]
        for aspect in row["annotation"]["aspects"]
        if aspect["label"] not in {2, None}
    ]


def is_negative_or_rare(row: Mapping[str, Any]) -> bool:
    mentioned = labels(row)
    if -1 in mentioned:
        return True
    return any(
        aspect["aspect"] in {"Bảo hành & Đổi trả", "Tính xác thực"}
        and aspect["label"] not in {2, None}
        for aspect in row["annotation"]["aspects"]
    )


def challenge_bin(
    row: Mapping[str, Any], major_ids: set[str]
) -> str | None:
    if row["sample_id"] in major_ids:
        return "SEMANTIC_AUDIT_MAJOR"
    status = row["annotation"]["annotation_status"]
    if status == "ESCALATE":
        return "ESCALATE"
    if status == "REJECT_NON_REVIEW":
        return "REJECT_NON_REVIEW"
    mentioned = labels(row)
    if "1, -1" in mentioned:
        return "MIXED_POLARITY"
    if 0 in mentioned:
        return "NEUTRAL_POLARITY"
    if is_negative_or_rare(row):
        return "NEGATIVE_OR_RARE_ASPECT"
    return None


def render_markdown(report: Mapping[str, Any]) -> str:
    plan = report["plan"]
    feasibility = report["feasibility"]
    categories = "\n".join(
        f"| `{row['category']}` | {row['available']:,} | "
        f"{row['rating_1_2']:,} | {row['rating_3_4']:,} | "
        f"{row['selenium_dom']:,} |"
        for row in feasibility["core_category"]
    )
    challenge = "\n".join(
        f"| `{row['bin']}` | {row['required']:,} | "
        f"{row['available_disjoint']:,} | `{row['status']}` |"
        for row in feasibility["challenge_bins"]
    )
    return f"""# Q1 Human-Gold & IAA Sampling Plan v1.1

Status: **{report['status']}**  
Snapshot: `{report['snapshot_id']}`  
Target: **{plan['target_unique_reviews']:,} unique reviews**, double-blind by
two human annotators.

## 1. Separation from previous calibration

- Existing 200 human-reference records are calibration-only.
- All 6,646 previously reserved reference/leakage-group records remain
excluded.
- Old historical data and confirmed duplicates are excluded.
- New selected leakage groups must be removed from future pseudo-label
training before dev/test publication.
- AI labels are sampling metadata only and are never shown in blind mode.

## 2. Panels

- `CORE_BALANCED`: **800** records, 100 per each of eight mapped categories.
  Every category requires at least 10 rating-1/2 and 10 rating-3/4 examples.
  The panel additionally requires at least
  {plan['panels']['CORE_BALANCED']['minimum_selenium_dom_total']}
  Selenium-DOM, 320 clean-origin and 320 quarantine-origin records.
- `CHALLENGE_DIAGNOSTIC`: **400** records from ordered disjoint bins. This
  panel is diagnostic and must not be reported as prevalence.

### Core-frame feasibility

| Category | Available | Rating 1/2 | Rating 3/4 | Selenium |
|---|---:|---:|---:|---:|
{categories}

### Challenge-bin feasibility

| Ordered disjoint bin | Required | Available | Gate |
|---|---:|---:|---|
{challenge}

## 3. Leakage control

The preflight rebuilds leakage components across product ID, exact curated
text, near-duplicate representative and template-family ID. The preparation
step may select at most 240 components, at most eight core records and twelve
total records per component. Every selected component is reserved from future
training.

## 4. Annotation and adjudication

- Both annotators receive all 1,200 items with independent opaque IDs/order.
- Agreement is computed before adjudication: status agreement, aspect mention
  F1, per-aspect polarity Cohen's kappa, Krippendorff's alpha and confusion
  matrices.
- Every disagreement is adjudicated. No AI suggestion is visible before both
  blind submissions are frozen.
- After adjudication, core is split group-wise into 300 dev and 500 test;
  challenge remains a separate 400-record diagnostic set.

## 5. Interpretation

This is a deliberately balanced evaluation design, not a prevalence sample.
Primary reporting must use per-aspect/per-polarity macro metrics and keep
challenge results separate. The 95.25% five-star source skew and all other
collection limitations remain part of the paper.
"""


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite plan: {OUTPUT}")
    config = read_json(CONFIG)
    snapshot = read_json(SNAPSHOT / "manifest.json")
    audit_manifest = read_json(CORPUS_AUDIT / "manifest.json")
    if config["snapshot_id"] != snapshot["snapshot_id"]:
        raise ValueError("Sampling config snapshot binding mismatch")
    if audit_manifest["snapshot_id"] != snapshot["snapshot_id"]:
        raise ValueError("Corpus audit snapshot binding mismatch")

    curation_rows: list[dict[str, Any]] = []
    for root in CURATION_RELEASES:
        curation_rows.extend(read_jsonl(root / "curation_records.jsonl"))
    group_by_sample, members_by_group = build_groups(curation_rows)
    reservation_ids = {
        row["sample_id"] for row in read_jsonl(RESERVATIONS)
    }
    reference_ids = {
        row["sample_id"] for row in read_jsonl(REFERENCE_CROSSWALK)
    }
    contaminated_groups = {
        group_by_sample[sample_id]
        for sample_id in reservation_ids | reference_ids
        if sample_id in group_by_sample
    }

    pseudo_rows: list[dict[str, Any]] = []
    major_ids: set[str] = set()
    for package in CRAWLED_PACKAGES:
        package_rows = list(
            read_jsonl(package / "final" / "ai_pseudo_labels.jsonl")
        )
        pseudo_rows.extend(package_rows)
        sample_by_annotation = {
            row["annotation_id"]: row["sample_id"] for row in package_rows
        }
        sample_by_hash = {
            row["review_text_sha256"]: row["sample_id"]
            for row in package_rows
        }
        ledger = package / "audits" / "semantic_audit_60_v1" / "audit_ledger.jsonl"
        for row in read_jsonl(ledger):
            if row["audit_severity"] == "MAJOR":
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str):
                    sample_id = sample_by_annotation.get(
                        row.get("annotation_id")
                    )
                if not isinstance(sample_id, str):
                    sample_id = sample_by_hash.get(
                        row.get("review_text_sha256")
                    )
                if not isinstance(sample_id, str):
                    raise ValueError(
                        "Cannot join semantic-audit row to source sample"
                    )
                major_ids.add(sample_id)
    if len(pseudo_rows) != 26_130 or len(major_ids) != 53:
        raise ValueError("Pseudo-label/semantic-major inventory mismatch")
    pseudo_ids = {row["sample_id"] for row in pseudo_rows}
    if pseudo_ids & reference_ids or pseudo_ids & reservation_ids:
        raise ValueError("Pseudo frame overlaps prior reference reservation")
    frame = [
        row for row in pseudo_rows
        if group_by_sample[row["sample_id"]] not in contaminated_groups
    ]
    frame_ids = {row["sample_id"] for row in frame}
    if len(frame_ids) != len(frame):
        raise ValueError("Sampling frame contains duplicate sample IDs")
    excluded_group_extension = len(pseudo_rows) - len(frame)

    categories = config["panels"]["CORE_BALANCED"]["mapped_categories"]
    core_rows: list[dict[str, Any]] = []
    core_pass = True
    for category in categories:
        rows = [row for row in frame if row["source"]["category"] == category]
        ratings_12 = sum(row["source"]["rating"] in {1, 2} for row in rows)
        ratings_34 = sum(row["source"]["rating"] in {3, 4} for row in rows)
        selenium = sum(
            row["source"]["collection_transport"] == "selenium_dom"
            for row in rows
        )
        checks = {
            "records_100": len(rows) >= 100,
            "rating_1_2_min_10": ratings_12 >= 10,
            "rating_3_4_min_10": ratings_34 >= 10,
        }
        core_pass = core_pass and all(checks.values())
        core_rows.append(
            {
                "category": category,
                "available": len(rows),
                "rating_1_2": ratings_12,
                "rating_3_4": ratings_34,
                "rating_5": sum(row["source"]["rating"] == 5 for row in rows),
                "selenium_dom": selenium,
                "clean_origin": sum(
                    row["source"]["curation_status"]
                    in {"KEEP", "KEEP_CLEANED"}
                    for row in rows
                ),
                "quarantine_origin": sum(
                    row["source"]["curation_status"] == "QUARANTINE"
                    for row in rows
                ),
                "gate": "PASS" if all(checks.values()) else "FAIL",
            }
        )

    quotas = config["panels"]["CHALLENGE_DIAGNOSTIC"][
        "ordered_disjoint_bin_quotas"
    ]
    disjoint_counts = Counter()
    for row in frame:
        bin_name = challenge_bin(row, major_ids)
        if bin_name is not None:
            disjoint_counts[bin_name] += 1
    challenge_rows = []
    challenge_pass = True
    for bin_name, required in quotas.items():
        available = disjoint_counts[bin_name]
        passed = available >= required
        challenge_pass = challenge_pass and passed
        challenge_rows.append(
            {
                "bin": bin_name,
                "required": required,
                "available_disjoint": available,
                "headroom": available - required,
                "status": "PASS" if passed else "FAIL",
            }
        )
    group_sizes = [
        len(
            [
                sample_id
                for sample_id in members
                if sample_id in frame_ids
            ]
        )
        for group_id, members in members_by_group.items()
        if group_id not in contaminated_groups
        and any(sample_id in frame_ids for sample_id in members)
    ]
    maximum_core_per_group = config["leakage_grouping"][
        "maximum_core_records_per_group"
    ]
    selenium_minimum = config["panels"]["CORE_BALANCED"][
        "minimum_selenium_dom_total"
    ]
    selenium_by_group = Counter(
        group_by_sample[row["sample_id"]]
        for row in frame
        if row["source"]["collection_transport"] == "selenium_dom"
        and row["source"]["category"] in categories
    )
    selenium_group_aware_capacity = sum(
        min(maximum_core_per_group, count)
        for count in selenium_by_group.values()
    )
    global_checks = {
        "frame_at_least_target": len(frame) >= config["target_unique_reviews"],
        "all_eligible_semantic_major_are_quota_bound": (
            quotas["SEMANTIC_AUDIT_MAJOR"]
            == len(major_ids & frame_ids)
        ),
        "selenium_record_availability_meets_minimum": sum(
            row["source"]["collection_transport"] == "selenium_dom"
            for row in frame
        ) >= selenium_minimum,
        "selenium_group_aware_capacity_meets_minimum": (
            selenium_group_aware_capacity >= selenium_minimum
        ),
        "clean_total_at_least_320": sum(
            row["source"]["curation_status"] in {"KEEP", "KEEP_CLEANED"}
            for row in frame
        ) >= 320,
        "quarantine_total_at_least_320": sum(
            row["source"]["curation_status"] == "QUARANTINE"
            for row in frame
        ) >= 320,
        "core_category_margins_feasible": core_pass,
        "challenge_quotas_feasible": challenge_pass,
        "quota_sum_is_400": sum(quotas.values()) == 400,
        "panel_sum_is_1200": sum(
            panel["records"] for panel in config["panels"].values()
        ) == config["target_unique_reviews"],
    }
    status = "VALID_FEASIBLE" if all(global_checks.values()) else "INVALID"
    if status != "VALID_FEASIBLE":
        raise ValueError(
            "Human-gold sampling design is infeasible: "
            + canonical_json(
                {
                    "checks": global_checks,
                    "eligible_semantic_major": len(major_ids & frame_ids),
                    "challenge_available": dict(disjoint_counts),
                    "eligible_records": len(frame),
                    "excluded_group_extension": excluded_group_extension,
                }
            )
        )
    group_summary = {
        "groups": len(group_sizes),
        "records": sum(group_sizes),
        "min": min(group_sizes),
        "median": sorted(group_sizes)[len(group_sizes) // 2],
        "mean": sum(group_sizes) / len(group_sizes),
        "max": max(group_sizes),
    }
    report = {
        "schema_version": "q1-human-gold-sampling-plan/1.1.0",
        "artifact_type": "Q1_HUMAN_GOLD_SAMPLING_PLAN",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_manifest_sha256": sha256_file(SNAPSHOT / "manifest.json"),
        "corpus_audit_id": audit_manifest["audit_id"],
        "corpus_audit_manifest_sha256": sha256_file(
            CORPUS_AUDIT / "manifest.json"
        ),
        "config_sha256": sha256_file(CONFIG),
        "plan": config,
        "eligibility": {
            "pseudo_frame_records": len(pseudo_rows),
            "prior_reference_records": len(reference_ids),
            "prior_reserved_group_records": len(reservation_ids),
            "expanded_contaminated_groups": len(contaminated_groups),
            "records_excluded_by_expanded_group_intersection": (
                excluded_group_extension
            ),
            "eligible_records": len(frame),
            "eligible_unique_sample_ids": len(frame_ids),
            "semantic_audit_major_records": len(major_ids),
            "old_records_included": 0,
            "confirmed_duplicates_included": 0,
        },
        "leakage_groups": group_summary,
        "group_aware_capacity": {
            "selenium_eligible_groups": len(selenium_by_group),
            "selenium_eligible_records": sum(selenium_by_group.values()),
            "maximum_core_records_per_group": maximum_core_per_group,
            "selenium_core_capacity_upper_bound": (
                selenium_group_aware_capacity
            ),
            "selenium_required": selenium_minimum,
            "headroom": selenium_group_aware_capacity - selenium_minimum,
        },
        "feasibility": {
            "global_checks": global_checks,
            "core_category": core_rows,
            "challenge_bins": challenge_rows,
        },
        "decisions": [
            "Supersede sampling plan v1 before selection because its Selenium quota was infeasible under the frozen group cap.",
            "Use one blind gold package with two independent annotator assignments.",
            "Use CORE_BALANCED for dev/test and CHALLENGE_DIAGNOSTIC only for diagnostic reporting.",
            "Never show AI suggestions before both blind exports are frozen.",
            "Reserve every selected leakage component from future model training.",
            "Complete blind gold work before AI-assisted review on the same sample.",
        ],
        "limitations": [
            "CORE_BALANCED is not a prevalence sample and must not be used to estimate source-corpus label prevalence.",
            "AI labels are used only to form challenge bins; they are hidden from annotators.",
            "Feasibility does not substitute for two human annotators or expert adjudication.",
            "Post-adjudication dev/test publication remains blocked until human work is complete.",
        ],
        "next_dependency": (
            "Run the deterministic package builder against this frozen plan, "
            "then validate assignment blindness, group reservation closure "
            "and independent role order."
        ),
    }

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    )
    try:
        report_path = temporary / "report.json"
        write_json(report_path, report)
        markdown_path = temporary / "report.md"
        markdown_path.write_text(
            render_markdown(report), encoding="utf-8", newline="\n"
        )
        core_path = temporary / "tables" / "core_category_feasibility.csv"
        challenge_path = (
            temporary / "tables" / "challenge_bin_feasibility.csv"
        )
        write_csv(core_path, core_rows)
        write_csv(challenge_path, challenge_rows)
        group_path = temporary / "tables" / "leakage_group_summary.csv"
        write_csv(group_path, [group_summary])
        provenance = []
        for source, relative in (
            (Path(__file__).resolve(), "provenance/software/design_q1_human_gold_sampling.py"),
            (CONFIG, "provenance/config/q1_human_gold_sampling_v1_1.json"),
            (SNAPSHOT / "manifest.json", "provenance/snapshot/manifest.json"),
            (CORPUS_AUDIT / "manifest.json", "provenance/corpus_audit/manifest.json"),
        ):
            destination = temporary.joinpath(*Path(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            provenance.append(destination)
        artifacts = [
            artifact(report_path, temporary),
            artifact(markdown_path, temporary),
            artifact(core_path, temporary),
            artifact(challenge_path, temporary),
            artifact(group_path, temporary),
            *(artifact(path, temporary) for path in provenance),
        ]
        manifest = {
            "schema_version": "q1-human-gold-sampling-plan-manifest/1.1.0",
            "artifact_type": "Q1_HUMAN_GOLD_SAMPLING_PLAN_RELEASE",
            "status": status,
            "plan_id": (
                "q1-human-gold-plan-"
                + sha256_text(
                    snapshot["snapshot_id"]
                    + "\0"
                    + sha256_file(CONFIG)
                    + "\0"
                    + sha256_file(report_path)
                )[:16]
            ),
            "created_at": report["created_at"],
            "snapshot_id": snapshot["snapshot_id"],
            "corpus_audit_id": audit_manifest["audit_id"],
            "report_sha256": sha256_file(report_path),
            "artifacts": sorted(artifacts, key=lambda row: row["path"]),
        }
        manifest_path = temporary / "manifest.json"
        write_json(manifest_path, manifest)
        sums = [
            (row["sha256"], row["path"]) for row in artifacts
        ] + [(sha256_file(manifest_path), "manifest.json")]
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(
                f"{digest}  {relative}\n"
                for digest, relative in sorted(sums, key=lambda row: row[1])
            ),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(OUTPUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": status,
        "output": str(OUTPUT),
        "plan_id": manifest["plan_id"],
        "eligible_records": len(frame),
        "target_records": config["target_unique_reviews"],
        "semantic_major_records": len(major_ids),
        "manifest_sha256": sha256_file(OUTPUT / "manifest.json"),
        "checksums_sha256": sha256_file(OUTPUT / "SHA256SUMS.txt"),
        "report_sha256": sha256_file(OUTPUT / "report.json"),
    }


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
