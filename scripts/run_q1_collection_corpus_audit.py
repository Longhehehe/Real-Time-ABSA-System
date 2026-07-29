"""Build a publication-oriented collection and corpus audit for the Q1 snapshot."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "releases" / "q1_dataset_snapshot_v1_20260728"
OUTPUT = (
    ROOT / "docs" / "audits" / "q1_collection_corpus_audit_v1_20260728"
)
ACCEPTED_RELEASES = (
    ROOT / "data" / "releases" / "lazada_vi_reviews_v1_20260725",
    ROOT / "data" / "releases" / "lazada_vi_reviews_delta_v1_20260728",
)
CURATION_RELEASES = (
    ROOT / "data" / "releases"
    / "lazada_vi_absa_curation_v2_1_2_20260725",
    ROOT / "data" / "releases"
    / "lazada_vi_absa_delta_curation_v1_20260728",
)
CRAWLED_FINALS = (
    ROOT / "data" / "annotations"
    / "absa_ai_tranche_5000_v1_20260727" / "final",
    ROOT / "data" / "annotations"
    / "absa_ai_remainder_8976_v1_20260728" / "final",
    ROOT / "data" / "annotations"
    / "absa_ai_delta_v1_20260728" / "final",
    ROOT / "data" / "annotations"
    / "absa_ai_quarantine_base_11166_v1_20260728" / "final",
    ROOT / "data" / "annotations"
    / "absa_ai_quarantine_delta_375_v1_20260728" / "final",
)
OLD_FINAL = (
    ROOT / "data" / "annotations"
    / "absa_legacy_old_relabel_9772_v1_20260728" / "final"
)
ASPECTS = (
    "Chất lượng sản phẩm",
    "Hiệu năng & Trải nghiệm",
    "Đúng mô tả",
    "Giá cả & Khuyến mãi",
    "Vận chuyển",
    "Đóng gói",
    "Dịch vụ & Thái độ Shop",
    "Bảo hành & Đổi trả",
    "Tính xác thực",
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def key(value: Any) -> str:
    if value is None or value == "":
        return "MISSING"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def distribution_rows(
    counter: Counter[str], *, dimension: str, total: int
) -> list[dict[str, Any]]:
    return [
        {
            "dimension": dimension,
            "value": value,
            "records": count,
            "share": round(count / total, 8) if total else 0.0,
        }
        for value, count in sorted(
            counter.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"records": 0}
    return {
        "records": len(values),
        "min": round(min(values), 4),
        "p25": round(percentile(values, 0.25), 4),
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "p75": round(percentile(values, 0.75), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def gini(counts: Sequence[int]) -> float:
    if not counts or sum(counts) == 0:
        return 0.0
    ordered = sorted(counts)
    n = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (n * sum(ordered)) - (n + 1) / n


def parse_review_date(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for pattern in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(value.strip(), pattern)
        except ValueError:
            pass
    return None


def load_rows(roots: Sequence[Path], filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        rows.extend(read_jsonl(root / filename))
    return rows


def coverage_table(
    accepted: Sequence[dict[str, Any]],
    workflow_ids: set[str],
    labeled_ids: set[str],
    accessor: Callable[[dict[str, Any]], Any],
    dimension: str,
) -> list[dict[str, Any]]:
    accepted_counts: Counter[str] = Counter()
    workflow_counts: Counter[str] = Counter()
    labeled_counts: Counter[str] = Counter()
    for row in accepted:
        value = key(accessor(row))
        accepted_counts[value] += 1
        if row["sample_id"] in workflow_ids:
            workflow_counts[value] += 1
        if row["sample_id"] in labeled_ids:
            labeled_counts[value] += 1
    return [
        {
            "dimension": dimension,
            "value": value,
            "accepted_records": accepted_counts[value],
            "workflow_records": workflow_counts[value],
            "workflow_coverage": round(
                workflow_counts[value] / accepted_counts[value], 8
            ),
            "labeled_records": labeled_counts[value],
            "labeled_coverage": round(
                labeled_counts[value] / accepted_counts[value], 8
            ),
        }
        for value in sorted(accepted_counts)
    ]


def artifact(path: Path, root: Path, *, records: int | None = None) -> dict:
    value: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def render_markdown(report: Mapping[str, Any]) -> str:
    collection = report["collection"]
    accepted = report["accepted_corpus"]
    curation = report["curation"]
    labeling = report["labeling"]
    bias = report["bias_indicators"]
    status_lines = "\n".join(
        f"| `{name}` | {count:,} |"
        for name, count in collection["run_status"].items()
    )
    rating_lines = "\n".join(
        f"| {row['value']} | {row['records']:,} | {row['share']:.2%} |"
        for row in accepted["rating_distribution"]
    )
    category_lines = "\n".join(
        f"| `{row['value']}` | {row['records']:,} | {row['share']:.2%} |"
        for row in accepted["category_distribution"]
    )
    transport_lines = "\n".join(
        f"| `{row['value']}` | {row['records']:,} | {row['share']:.2%} |"
        for row in accepted["transport_distribution"]
    )
    return f"""# Q1 Collection & Corpus Audit v1

Snapshot: `{report['snapshot']['snapshot_id']}`  
Audit status: **{report['status']}**  
Generated: `{report['created_at']}`

## 1. Collection process

- Crawl runs: **{collection['runs']:,}**; open/running runs:
  **{collection['open_runs']:,}**.
- Collection window: `{collection['started_at_min']}` to
  `{collection['finished_at_max']}`.
- Candidate accounting: **{collection['review_candidates']:,}** candidates =
  **{collection['reviews_written']:,}** written +
  **{collection['reviews_rejected_quality']:,}** quality-rejected +
  **{collection['reviews_deduplicated']:,}** deduplicated.
- Run-level errors recorded: **{collection['errors']:,}**.
- Current substantive policy requires at least 80 characters, 15 words,
  Vietnamese signals, unique-word ratio 0.4 and quality score 0.55.

| Run status | Runs |
|---|---:|
{status_lines}

## 2. Accepted crawled corpus

- Accepted unique records: **{accepted['records']:,}** across
  **{accepted['unique_products']:,}** products and
  **{accepted['unique_queries']:,}** stored queries.
- Review-date parse coverage: **{accepted['review_date_parse_coverage']:.2%}**;
  observed review dates `{accepted['review_date_min']}` to
  `{accepted['review_date_max']}`.
- Verified purchase: **{accepted['verified_purchase_share']:.2%}**; has image:
  **{accepted['has_images_share']:.2%}**.
- Product concentration: top-1 **{accepted['product_concentration']['top1_share']:.2%}**,
  top-10 **{accepted['product_concentration']['top10_share']:.2%}**,
  HHI `{accepted['product_concentration']['hhi']:.6f}`, Gini
  `{accepted['product_concentration']['gini']:.4f}`.

### Rating

| Rating | Records | Share |
|---|---:|---:|
{rating_lines}

### Category

| Category | Records | Share |
|---|---:|---:|
{category_lines}

### Collection transport

| Transport | Records | Share |
|---|---:|---:|
{transport_lines}

## 3. Curation

- Closure: **{curation['records']:,}** =
  **{curation['status']['KEEP'] + curation['status']['KEEP_CLEANED']:,}**
  clean-core + **{curation['status']['QUARANTINE']:,}** quarantine +
  **{curation['status']['EXCLUDE_AUTO']:,}** duplicate exclusion.
- Quarantine share: **{curation['quarantine_share']:.2%}**.
- Every source record remains represented in a versioned decision ledger.

## 4. Labeling coverage

- Crawled records sent through LLM workflow: **{labeling['records']:,}**
  (**{labeling['coverage_of_accepted']:.2%}** of accepted corpus).
- Terminal status: **{labeling['annotation_status']['LABELED']:,}**
  `LABELED`, **{labeling['annotation_status']['ESCALATE']:,}**
  `ESCALATE`, **{labeling['annotation_status']['REJECT_NON_REVIEW']:,}**
  `REJECT_NON_REVIEW`.
- Old canonical pseudo-label records are reported separately:
  **{labeling['old_records']:,}**.

## 5. Bias indicators and interpretation

- Five-star share: **{bias['five_star_share']:.2%}**.
- Largest category share: **{bias['largest_category_share']:.2%}**.
- Selenium-DOM share: **{bias['selenium_dom_share']:.2%}**.
- Reviews are intentionally length/quality selected; this corpus is not a
  representative sample of all Lazada reviews.
- Platform, query, product availability, rate limits, long-review filtering,
  quarantine rules and human-reference reservation all create selection
  effects that must be stated in the paper.
- No demographic, region or seller-population frame is available, so
  population-level representativeness cannot be claimed.

## 6. Publication decision

This audit supports a **dataset-construction report**, not a claim that all
pseudo-labels are human gold. Gold sampling must be stratified using the
category, rating, transport, curation-origin, annotation-status, aspect and
polarity tables produced here. Development and test partitions must be
human-verified and group-isolated from model training and prompt examples.
"""


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {OUTPUT}")
    snapshot = read_json(SNAPSHOT / "manifest.json")
    accepted = load_rows(ACCEPTED_RELEASES, "reviews_canonical.jsonl")
    curation = load_rows(CURATION_RELEASES, "curation_records.jsonl")
    crawled_labels = load_rows(CRAWLED_FINALS, "ai_pseudo_labels.jsonl")
    old_labels = list(read_jsonl(OLD_FINAL / "ai_pseudo_labels.jsonl"))
    if len(accepted) != 32_918 or len(curation) != 32_918:
        raise ValueError("Accepted/curation corpus count mismatch")
    if len(crawled_labels) != 26_130 or len(old_labels) != 9_772:
        raise ValueError("Pseudo-label corpus count mismatch")
    accepted_ids = {row["sample_id"] for row in accepted}
    curation_ids = {row["sample_id"] for row in curation}
    workflow_ids = {row["sample_id"] for row in crawled_labels}
    if (
        len(accepted_ids) != len(accepted)
        or accepted_ids != curation_ids
        or len(workflow_ids) != len(crawled_labels)
        or not workflow_ids.issubset(accepted_ids)
    ):
        raise ValueError("Corpus membership/closure mismatch")

    run_manifests: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    rejection_rows = 0
    for path in sorted((ROOT / "data" / "raw").rglob("manifest.json")):
        run_manifests.append(read_json(path))
    for path in sorted((ROOT / "data" / "raw").rglob("rejections.jsonl")):
        for row in read_jsonl(path):
            rejection_rows += 1
            rejection_reasons.update(key(value) for value in row.get("reasons", []))
    if len(run_manifests) != 411:
        raise ValueError("Expected 411 closed collection manifests")
    count_totals = Counter()
    for manifest in run_manifests:
        count_totals.update(manifest.get("counts", {}))
    if (
        count_totals["reviews_written"]
        + count_totals["reviews_rejected_quality"]
        + count_totals["reviews_deduplicated"]
        != count_totals["review_candidates"]
    ):
        raise ValueError("Raw candidate accounting is not closed")
    run_status = Counter(key(row.get("status")) for row in run_manifests)
    open_runs = sum(
        count for status, count in run_status.items()
        if status.casefold() in {"running", "started", "in_progress"}
    )
    commands = Counter(key(row.get("command")) for row in run_manifests)
    parameter_transports = Counter(
        key(row.get("parameters", {}).get("transport"))
        for row in run_manifests
    )
    started = sorted(
        row["started_at"] for row in run_manifests
        if isinstance(row.get("started_at"), str)
    )
    finished = sorted(
        row["finished_at"] for row in run_manifests
        if isinstance(row.get("finished_at"), str)
    )

    accepted_category = Counter(key(row.get("category")) for row in accepted)
    accepted_rating = Counter(key(row.get("rating")) for row in accepted)
    accepted_transport = Counter(
        key(row.get("collection_transport")) for row in accepted
    )
    accepted_query = Counter(key(row.get("query")) for row in accepted)
    product_counts = Counter(key(row.get("product_id")) for row in accepted)
    product_total = sum(product_counts.values())
    product_ordered = product_counts.most_common()
    product_shares = [
        count / product_total for _, count in product_ordered
    ]
    review_dates = [
        parsed for row in accepted
        if (parsed := parse_review_date(row.get("review_time"))) is not None
    ]
    verified_count = sum(row.get("verified_purchase") is True for row in accepted)
    image_count = sum(row.get("has_images") is True for row in accepted)

    curation_status = Counter(
        key(row["status"]) for row in curation
    )
    curation_reason = Counter(
        key(row["primary_reason"]) for row in curation
    )
    curation_by_id = {row["sample_id"]: row for row in curation}

    annotation_status = Counter(
        key(row["annotation"]["annotation_status"])
        for row in crawled_labels
    )
    labeled_ids = {
        row["sample_id"]
        for row in crawled_labels
        if row["annotation"]["annotation_status"] == "LABELED"
    }
    source_curation_status = Counter(
        key(row["source"].get("curation_status")) for row in crawled_labels
    )
    aspect_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for row in crawled_labels:
        for aspect in row["annotation"]["aspects"]:
            aspect_labels[aspect["aspect"]][key(aspect["label"])] += 1
    if set(aspect_labels) != set(ASPECTS):
        raise ValueError("Aspect inventory mismatch")

    category_coverage = coverage_table(
        accepted, workflow_ids, labeled_ids,
        lambda row: row.get("category"), "category"
    )
    rating_coverage = coverage_table(
        accepted, workflow_ids, labeled_ids,
        lambda row: row.get("rating"), "rating"
    )
    transport_coverage = coverage_table(
        accepted, workflow_ids, labeled_ids,
        lambda row: row.get("collection_transport"), "collection_transport"
    )

    length_groups = {
        "accepted_all": [float(row.get("char_count", 0)) for row in accepted],
        "clean_core": [
            float(row.get("char_count", len(row.get("curated_review_text", ""))))
            for row in curation
            if row["status"] in {"KEEP", "KEEP_CLEANED"}
        ],
        "quarantine": [
            float(row.get("char_count", len(row.get("curated_review_text", ""))))
            for row in curation
            if row["status"] == "QUARANTINE"
        ],
        "llm_workflow": [
            float(len(row["reviewContent"])) for row in crawled_labels
        ],
        "terminal_labeled": [
            float(len(row["reviewContent"]))
            for row in crawled_labels
            if row["annotation"]["annotation_status"] == "LABELED"
        ],
        "old_canonical": [
            float(len(row["reviewContent"])) for row in old_labels
        ],
    }
    length_summary = {
        name: numeric_summary(values) for name, values in length_groups.items()
    }

    accepted_category_rows = distribution_rows(
        accepted_category, dimension="category", total=len(accepted)
    )
    accepted_rating_rows = distribution_rows(
        accepted_rating, dimension="rating", total=len(accepted)
    )
    accepted_transport_rows = distribution_rows(
        accepted_transport,
        dimension="collection_transport",
        total=len(accepted),
    )
    accepted_query_rows = distribution_rows(
        accepted_query, dimension="query", total=len(accepted)
    )
    product_rows = []
    cumulative = 0
    for rank, (product_id, count) in enumerate(product_ordered[:50], 1):
        cumulative += count
        sample = next(
            row for row in accepted if key(row.get("product_id")) == product_id
        )
        product_rows.append(
            {
                "rank": rank,
                "product_id": product_id,
                "category": key(sample.get("category")),
                "records": count,
                "share": round(count / len(accepted), 8),
                "cumulative_share": round(cumulative / len(accepted), 8),
            }
        )
    aspect_rows = []
    for aspect in ASPECTS:
        for label, count in sorted(
            aspect_labels[aspect].items(), key=lambda item: item[0]
        ):
            aspect_rows.append(
                {
                    "aspect": aspect,
                    "label": label,
                    "records": count,
                    "share_of_workflow": round(
                        count / len(crawled_labels), 8
                    ),
                }
            )
    curation_status_rows = distribution_rows(
        curation_status, dimension="curation_status", total=len(curation)
    )
    curation_reason_rows = distribution_rows(
        curation_reason, dimension="curation_primary_reason", total=len(curation)
    )
    annotation_status_rows = distribution_rows(
        annotation_status,
        dimension="annotation_status",
        total=len(crawled_labels),
    )
    length_rows = [
        {"group": group, **metrics}
        for group, metrics in length_summary.items()
    ]
    rejection_rows_table = distribution_rows(
        rejection_reasons,
        dimension="quality_rejection_reason",
        total=rejection_rows,
    )

    five_star_share = accepted_rating["5"] / len(accepted)
    largest_category_share = accepted_category.most_common(1)[0][1] / len(accepted)
    selenium_dom_share = accepted_transport["selenium_dom"] / len(accepted)
    workflow_coverage = len(workflow_ids) / len(accepted)
    product_concentration = {
        "unique_products": len(product_counts),
        "reviews_per_product": numeric_summary(list(product_counts.values())),
        "top1_share": sum(product_shares[:1]),
        "top5_share": sum(product_shares[:5]),
        "top10_share": sum(product_shares[:10]),
        "top20_share": sum(product_shares[:20]),
        "hhi": sum(share * share for share in product_shares),
        "gini": gini(list(product_counts.values())),
    }
    report = {
        "schema_version": "q1-collection-corpus-audit/1.0.0",
        "artifact_type": "Q1_COLLECTION_AND_CORPUS_AUDIT",
        "status": "VALID",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": {
            "snapshot_id": snapshot["snapshot_id"],
            "manifest_sha256": sha256_file(SNAPSHOT / "manifest.json"),
            "checksums_sha256": sha256_file(SNAPSHOT / "SHA256SUMS.txt"),
        },
        "collection": {
            "runs": len(run_manifests),
            "open_runs": open_runs,
            "started_at_min": started[0],
            "finished_at_max": finished[-1],
            "run_status": dict(sorted(run_status.items())),
            "commands": dict(sorted(commands.items())),
            "parameter_transport": dict(sorted(parameter_transports.items())),
            "products_processed_encounters": count_totals["products"],
            "review_candidates": count_totals["review_candidates"],
            "reviews_written": count_totals["reviews_written"],
            "reviews_rejected_quality": count_totals[
                "reviews_rejected_quality"
            ],
            "reviews_deduplicated": count_totals["reviews_deduplicated"],
            "errors": count_totals["errors"],
            "quality_rejection_rows": rejection_rows,
            "quality_rejection_reason_counts": dict(
                sorted(rejection_reasons.items())
            ),
            "frozen_current_policy": {
                "min_chars": 80,
                "min_words": 15,
                "min_unique_word_ratio": 0.4,
                "min_meaningful_words": 8,
                "min_score": 0.55,
                "require_vietnamese": True,
                "min_vietnamese_signals": 2,
                "max_foreign_script_ratio": 0.2,
                "reject_suspect_encoding": True,
            },
        },
        "accepted_corpus": {
            "records": len(accepted),
            "unique_sample_ids": len(accepted_ids),
            "unique_products": len(product_counts),
            "unique_queries": len(
                {value for value in accepted_query if value != "MISSING"}
            ),
            "category_distribution": accepted_category_rows,
            "rating_distribution": accepted_rating_rows,
            "transport_distribution": accepted_transport_rows,
            "query_distribution": accepted_query_rows,
            "length": length_summary["accepted_all"],
            "word_count": numeric_summary(
                [float(row.get("word_count", 0)) for row in accepted]
            ),
            "quality_score": numeric_summary(
                [float(row.get("quality_score", 0)) for row in accepted]
            ),
            "review_date_parsed": len(review_dates),
            "review_date_parse_coverage": len(review_dates) / len(accepted),
            "review_date_min": min(review_dates).date().isoformat(),
            "review_date_max": max(review_dates).date().isoformat(),
            "verified_purchase_share": verified_count / len(accepted),
            "has_images_share": image_count / len(accepted),
            "product_concentration": product_concentration,
        },
        "curation": {
            "records": len(curation),
            "status": dict(sorted(curation_status.items())),
            "primary_reason": dict(sorted(curation_reason.items())),
            "quarantine_share": curation_status["QUARANTINE"] / len(curation),
            "source_membership_equals_accepted": curation_ids == accepted_ids,
        },
        "labeling": {
            "records": len(crawled_labels),
            "coverage_of_accepted": workflow_coverage,
            "annotation_status": dict(sorted(annotation_status.items())),
            "source_curation_status": dict(
                sorted(source_curation_status.items())
            ),
            "aspect_labels": {
                aspect: dict(sorted(counter.items()))
                for aspect, counter in sorted(aspect_labels.items())
            },
            "old_records": len(old_labels),
            "old_is_reported_outside_crawled_collection_frame": True,
        },
        "coverage": {
            "by_category": category_coverage,
            "by_rating": rating_coverage,
            "by_transport": transport_coverage,
        },
        "length_comparison": length_summary,
        "bias_indicators": {
            "five_star_share": five_star_share,
            "largest_category_share": largest_category_share,
            "selenium_dom_share": selenium_dom_share,
            "workflow_coverage_of_accepted": workflow_coverage,
            "quarantine_share": curation_status["QUARANTINE"] / len(curation),
            "long_review_selection": True,
            "single_platform": "Lazada Vietnam",
            "population_sampling_frame_available": False,
        },
        "measured_limitations": [
            "The collector intentionally retains substantive Vietnamese "
            "reviews and rejects short/low-quality text; the corpus is not "
            "representative of all Lazada review lengths.",
            "Availability, search queries, product review counts, platform "
            "rate limits and transport fallback affect product inclusion.",
            "Rating, category, transport and product concentration are "
            "observed sample properties, not population estimates.",
            "No customer demographic, geographic population frame or full "
            "seller catalogue frame is available.",
            "Quarantine and human-reference group reservation change LLM "
            "workflow coverage by design.",
            "Old canonical data is excluded from collection-distribution "
            "claims and is reported only as an external historical corpus.",
            "AI pseudo-label status is not human-gold accuracy.",
        ],
        "sampling_dependency": (
            "Build a human-gold sampling frame that jointly stratifies "
            "category, rating, collection transport, curation origin, "
            "annotation status, aspect/polarity rarity and product group."
        ),
    }

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    )
    try:
        tables = temporary / "tables"
        table_specs = {
            "accepted_by_category.csv": accepted_category_rows,
            "accepted_by_rating.csv": accepted_rating_rows,
            "accepted_by_transport.csv": accepted_transport_rows,
            "accepted_by_query.csv": accepted_query_rows,
            "top_products.csv": product_rows,
            "quality_rejection_reasons.csv": rejection_rows_table,
            "curation_status.csv": curation_status_rows,
            "curation_primary_reason.csv": curation_reason_rows,
            "annotation_status.csv": annotation_status_rows,
            "aspect_label_distribution.csv": aspect_rows,
            "coverage_by_category.csv": category_coverage,
            "coverage_by_rating.csv": rating_coverage,
            "coverage_by_transport.csv": transport_coverage,
            "review_length_summary.csv": length_rows,
        }
        table_paths: list[Path] = []
        for name, rows in table_specs.items():
            path = tables / name
            write_csv(path, rows)
            table_paths.append(path)
        report_path = temporary / "report.json"
        write_json(report_path, report)
        markdown_path = temporary / "report.md"
        markdown_path.write_text(
            render_markdown(report), encoding="utf-8", newline="\n"
        )
        provenance_paths = []
        for source, relative in (
            (
                Path(__file__).resolve(),
                "provenance/software/run_q1_collection_corpus_audit.py",
            ),
            (
                SNAPSHOT / "manifest.json",
                "provenance/snapshot/manifest.json",
            ),
            (
                SNAPSHOT / "SHA256SUMS.txt",
                "provenance/snapshot/SHA256SUMS.txt",
            ),
            (
                ROOT / "configs" / "collection_plan.toml",
                "provenance/configs/collection_plan.toml",
            ),
            (
                ROOT / "configs" / "collector.toml",
                "provenance/configs/collector.toml",
            ),
        ):
            destination = temporary.joinpath(*Path(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            provenance_paths.append(destination)
        artifacts = [
            artifact(report_path, temporary),
            artifact(markdown_path, temporary),
            *(artifact(path, temporary) for path in table_paths),
            *(artifact(path, temporary) for path in provenance_paths),
        ]
        manifest = {
            "schema_version": "q1-collection-corpus-audit-manifest/1.0.0",
            "artifact_type": "Q1_COLLECTION_AND_CORPUS_AUDIT_RELEASE",
            "status": "VALID",
            "audit_id": (
                "q1-corpus-audit-"
                + hashlib.sha256(
                    (
                        snapshot["snapshot_id"]
                        + "\0"
                        + sha256_file(report_path)
                    ).encode("utf-8")
                ).hexdigest()[:16]
            ),
            "created_at": report["created_at"],
            "snapshot_id": snapshot["snapshot_id"],
            "report_sha256": sha256_file(report_path),
            "artifacts": sorted(artifacts, key=lambda row: row["path"]),
            "interpretation": (
                "Descriptive audit of the frozen sample; not a population "
                "representativeness or human-label accuracy claim."
            ),
        }
        manifest_path = temporary / "manifest.json"
        write_json(manifest_path, manifest)
        sums = [
            (row["sha256"], row["path"]) for row in artifacts
        ] + [(sha256_file(manifest_path), "manifest.json")]
        sums_path = temporary / "SHA256SUMS.txt"
        sums_path.write_text(
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
        "status": "VALID",
        "output": str(OUTPUT),
        "audit_id": manifest["audit_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "accepted_records": len(accepted),
        "collection_runs": len(run_manifests),
        "workflow_records": len(crawled_labels),
        "manifest_sha256": sha256_file(OUTPUT / "manifest.json"),
        "checksums_sha256": sha256_file(OUTPUT / "SHA256SUMS.txt"),
        "report_sha256": sha256_file(OUTPUT / "report.json"),
    }


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
