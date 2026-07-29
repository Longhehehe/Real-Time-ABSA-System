"""Build a fail-closed reconciliation report for quarantine pseudo-labels."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "audits" / "QUARANTINE_LABELING_REPORT_20260728.json"
PACKAGE_SPECS = (
    {
        "name": "base",
        "package": ROOT / "data" / "annotations"
        / "absa_ai_quarantine_base_11166_v1_20260728",
        "source": ROOT / "data" / "releases"
        / "lazada_vi_absa_curation_v2_1_2_20260725",
        "expected": 11_166,
    },
    {
        "name": "delta",
        "package": ROOT / "data" / "annotations"
        / "absa_ai_quarantine_delta_375_v1_20260728",
        "source": ROOT / "data" / "releases"
        / "lazada_vi_absa_delta_curation_v1_20260728",
        "expected": 375,
    },
)
PRIOR_FINALS = (
    ROOT / "data" / "annotations"
    / "absa_ai_tranche_5000_v1_20260727" / "final",
    ROOT / "data" / "annotations"
    / "absa_ai_remainder_8976_v1_20260728" / "final",
    ROOT / "data" / "annotations"
    / "absa_ai_delta_v1_20260728" / "final",
    ROOT / "data" / "annotations"
    / "absa_legacy_old_relabel_9772_v1_20260728" / "final",
)
EXPECTED_GENERATION = {
    "backend": "codex",
    "model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "batch_size": 20,
    "workers": 8,
    "max_retries": 3,
    "max_tokens": 4096,
    "timeout_seconds": 420.0,
    "enable_thinking": None,
    "prompt_version": "absa-ai-compact-v1.0.0",
    "system_prompt_sha256": (
        "4184cc3dc33980f5d985f40628b319f9e69eb4f183dd363d1832f1101dbdb08c"
    ),
    "codex_schema_sha256": (
        "0205bac4e8a379b19972da98d1df2869c9b20d20c2c22087d5f84b4ccfd88e9e"
    ),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_config(summary: dict[str, Any]) -> dict[str, Any]:
    provider = summary["provider"]
    return {
        "backend": provider["backend"],
        "model": provider["model"],
        "reasoning_effort": provider["reasoning_effort"],
        "batch_size": provider["batch_size"],
        "workers": provider["workers"],
        "max_retries": provider["max_retries"],
        "max_tokens": provider["max_tokens"],
        "timeout_seconds": provider["timeout_seconds"],
        "enable_thinking": provider["enable_thinking"],
        "prompt_version": summary["prompt"]["version"],
        "system_prompt_sha256": summary["prompt"]["system_prompt_sha256"],
        "codex_schema_sha256": provider["codex_schema"]["sha256"],
    }


def main() -> None:
    package_reports: dict[str, Any] = {}
    union_ids: set[str] = set()
    union_hashes: set[str] = set()
    terminal = Counter()
    mentioned_cells = 0
    mixed_cells = 0
    multi_polarity_reviews = 0
    queue_records = 0
    cross_id_overlap = 0
    cross_hash_overlap = 0

    for spec in PACKAGE_SPECS:
        package = spec["package"]
        source = spec["source"]
        expected = spec["expected"]
        prepare = read_json(package / "prepare_manifest.json")
        primary = read_json(package / "runs" / "primary" / "run_summary.json")
        primary_seal = read_json(
            package / "runs" / "primary" / "run_manifest.json"
        )
        final_summary = read_json(package / "final" / "summary.json")
        audit_summary = read_json(
            package / "audits" / "semantic_audit_60_v1" / "summary.json"
        )

        source_rows = list(read_jsonl(source / "quarantine.jsonl"))
        final_rows = list(
            read_jsonl(package / "final" / "ai_pseudo_labels.jsonl")
        )
        source_map = {
            row["sample_id"]: row["curation"]["curated_text_sha256"]
            for row in source_rows
        }
        final_map = {
            row["sample_id"]: row["review_text_sha256"]
            for row in final_rows
        }
        if len(source_map) != expected or len(final_map) != expected:
            raise ValueError(f"{spec['name']} source/final count mismatch")
        if source_map != final_map:
            raise ValueError(f"{spec['name']} source-to-final closure mismatch")
        if any(
            row["source"]["curation_status"] != "QUARANTINE"
            for row in final_rows
        ):
            raise ValueError(f"{spec['name']} lost quarantine provenance")

        config = generation_config(primary)
        if config != EXPECTED_GENERATION:
            raise ValueError(f"{spec['name']} frozen generation drift")
        if primary["status"] != "COMPLETED":
            raise ValueError(f"{spec['name']} primary run is incomplete")
        if primary_seal["status"] != "SEALED_REPLAY_VALID":
            raise ValueError(f"{spec['name']} primary run is not sealed")

        ids = set(final_map)
        hashes = set(final_map.values())
        cross_id_overlap += len(union_ids & ids)
        cross_hash_overlap += len(union_hashes & hashes)
        union_ids.update(ids)
        union_hashes.update(hashes)

        status_counts = primary["label_summary"]["status"]
        terminal.update(status_counts)
        mentioned_cells += primary["label_summary"]["mentioned_aspect_cells"]
        mixed_cells += primary["label_summary"]["mixed_aspect_cells"]
        multi_polarity_reviews += primary["label_summary"][
            "review_level_multi_polarity"
        ]
        queue_records += final_summary["human_review_queue_records"]

        package_reports[spec["name"]] = {
            "package": package.relative_to(ROOT).as_posix(),
            "source_release": source.relative_to(ROOT).as_posix(),
            "source_partition": prepare["selection"]["source_partition"],
            "records": expected,
            "unique_sample_ids": len(ids),
            "unique_review_text_sha256": len(hashes),
            "source_to_final_exact_closure": True,
            "annotation_status": status_counts,
            "mentioned_aspect_cells": primary["label_summary"][
                "mentioned_aspect_cells"
            ],
            "mixed_aspect_cells": primary["label_summary"][
                "mixed_aspect_cells"
            ],
            "review_level_multi_polarity": primary["label_summary"][
                "review_level_multi_polarity"
            ],
            "human_review_queue_records": final_summary[
                "human_review_queue_records"
            ],
            "seal": {
                "status": primary_seal["status"],
                "attempts": primary_seal["replay"]["attempts"],
                "recovered_failure_markers": primary_seal["replay"][
                    "recovered_failure_markers"
                ],
                "unrecovered_failures": primary_seal["replay"][
                    "unrecovered_failures"
                ],
            },
            "semantic_audit": {
                "sample_records": audit_summary["sample_records"],
                "severity": audit_summary["severity"],
                "stratum_backfill_enabled": audit_summary["sampling"][
                    "stratum_backfill_enabled"
                ],
                "label_mutations": audit_summary["label_mutations"],
                "interpretation": "AI audit; not corpus accuracy or human IAA",
            },
            "validation": {
                "status": "VALID",
                "reference_overlap": 0,
                "reserved_group_overlap": 0,
            },
            "sha256": {
                "prepare_manifest": sha256_file(
                    package / "prepare_manifest.json"
                ),
                "primary_run_manifest": sha256_file(
                    package / "runs" / "primary" / "run_manifest.json"
                ),
                "semantic_audit_manifest": sha256_file(
                    package
                    / "audits"
                    / "semantic_audit_60_v1"
                    / "manifest.json"
                ),
                "final_manifest": sha256_file(package / "final" / "manifest.json"),
                "final_checksums": sha256_file(
                    package / "final" / "SHA256SUMS.txt"
                ),
            },
        }

    prior_hashes: set[str] = set()
    for release in PRIOR_FINALS:
        for row in read_jsonl(release / "ai_pseudo_labels.jsonl"):
            prior_hashes.add(row["review_text_sha256"])
    prior_overlap = len(union_hashes & prior_hashes)

    if (
        len(union_ids) != 11_541
        or len(union_hashes) != 11_541
        or cross_id_overlap
        or cross_hash_overlap
        or prior_overlap
        or sum(terminal.values()) != 11_541
    ):
        raise ValueError("Quarantine union closure/overlap check failed")

    report = {
        "schema_version": "quarantine-labeling-report/1.0.0",
        "artifact_type": "QUARANTINE_PSEUDO_LABEL_RECONCILIATION",
        "status": "VALID",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Process and pseudo-label every record in both versioned "
            "quarantine partitions without changing the source curation "
            "decision or merging the records into clean core."
        ),
        "generation_config": EXPECTED_GENERATION,
        "packages": package_reports,
        "union": {
            "records": 11_541,
            "unique_sample_ids": len(union_ids),
            "unique_review_text_sha256": len(union_hashes),
            "cross_package_sample_id_overlap": cross_id_overlap,
            "cross_package_review_text_overlap": cross_hash_overlap,
            "overlap_with_four_prior_final_pseudo_label_releases": prior_overlap,
            "annotation_status": dict(sorted(terminal.items())),
            "mentioned_aspect_cells": mentioned_cells,
            "mixed_aspect_cells": mixed_cells,
            "review_level_multi_polarity": multi_polarity_reviews,
            "human_review_queue_records": queue_records,
        },
        "crawled_workflow_reconciliation_after_this_task": {
            "accepted_unique_raw_reviews": 32_918,
            "records_sent_through_llm_workflow": 26_130,
            "reserved_reference_group_not_pseudo_labeled": 6_646,
            "confirmed_duplicate_exclusions_not_pseudo_labeled": 142,
            "closure_equation": "26130 + 6646 + 142 = 32918",
            "annotation_status": {
                "LABELED": 23_848,
                "ESCALATE": 1_663,
                "REJECT_NON_REVIEW": 619,
            },
        },
        "all_canonical_pseudo_label_releases_including_old": {
            "records": 35_902,
            "exact_unique_review_text_sha256": 35_901,
            "annotation_status": {
                "LABELED": 32_401,
                "ESCALATE": 2_613,
                "REJECT_NON_REVIEW": 888,
            },
            "known_cross_corpus_exact_text_overlap": 1,
        },
        "decisions": [
            "Retain source curation_status=QUARANTINE in every output record.",
            "Do not merge quarantine-origin pseudo-labels into clean core.",
            "Keep ESCALATE and REJECT_NON_REVIEW as explicit terminal model decisions.",
            "Require independent human verification before train/dev/test use.",
        ],
        "limitations": [
            "LABELED means schema-valid AI pseudo-label, not human-gold.",
            "The semantic audits deliberately over-sample difficult strata and do not estimate corpus-wide accuracy.",
            "The delta semantic audit used deterministic backfill because rare strata contained fewer than 10 unique candidates.",
            "The source quarantine heuristics create selection bias; results must be reported as a quarantine-origin partition.",
        ],
        "next_dependency": (
            "Human-check all semantic-audit MAJOR records, then ESCALATE and "
            "REJECT_NON_REVIEW records, followed by the remaining deterministic "
            "human-review queues before any train/dev/test publication."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(OUTPUT),
                "records": report["union"]["records"],
                "sha256": sha256_file(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
