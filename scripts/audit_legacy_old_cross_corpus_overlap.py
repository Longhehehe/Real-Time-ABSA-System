"""Audit legacy-old text overlap with calibration, reference, and new corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


DEFAULT_OLD = Path(
    "data/releases/legacy_old_reviews_label_blind_v1_20260728/clean_core.jsonl"
)
DEFAULT_CALIBRATION = Path(
    "data/annotations/absa_ai_remainder_8976_v1_20260728/"
    "calibration/human_confirmed.jsonl"
)
DEFAULT_RESERVATIONS = Path(
    "data/annotations/human_reference_v1_20260726/"
    "private/group_reservations.jsonl"
)
DEFAULT_NEW_PACKAGES = (
    Path("data/annotations/absa_ai_tranche_5000_v1_20260727/final"),
    Path("data/annotations/absa_ai_remainder_8976_v1_20260728/final"),
)
DEFAULT_OUTPUT = Path(
    "docs/audits/legacy_old_cross_corpus_overlap_20260728.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", value).casefold(),
    ).strip()


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


def _text_index(
    rows: list[dict[str, Any]],
    *,
    text_key: str,
    id_key: str,
) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for row in rows:
        text = row[text_key]
        identifier = row[id_key]
        exact[_sha256_text(text)] = identifier
        normalized[_sha256_text(_normalized(text))] = identifier
    return exact, normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--reservations",
        type=Path,
        default=DEFAULT_RESERVATIONS,
    )
    parser.add_argument(
        "--new-package",
        type=Path,
        action="append",
        dest="new_packages",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    new_packages = tuple(args.new_packages or DEFAULT_NEW_PACKAGES)

    old_rows = _read_jsonl(args.old)
    old_exact, old_normalized = _text_index(
        old_rows,
        text_key="curated_review_text",
        id_key="sample_id",
    )
    calibration_rows = _read_jsonl(args.calibration)
    calibration_exact, calibration_normalized = _text_index(
        calibration_rows,
        text_key="reviewContent",
        id_key="calibration_id",
    )
    reservation_rows = _read_jsonl(args.reservations)
    reservation_exact = {
        row["review_text_sha256"]: row["sample_id"]
        for row in reservation_rows
    }

    comparisons: list[dict[str, Any]] = []
    calibration_exact_overlap = set(old_exact).intersection(calibration_exact)
    calibration_normalized_overlap = set(old_normalized).intersection(
        calibration_normalized
    )
    comparisons.append(
        {
            "target": "prompt_calibration",
            "records": len(calibration_rows),
            "source_sha256": _sha256_file(args.calibration),
            "exact_text_hash_overlap": len(calibration_exact_overlap),
            "normalized_text_hash_overlap": len(
                calibration_normalized_overlap
            ),
            "overlap_old_sample_ids": sorted(
                old_exact[item] for item in calibration_exact_overlap
            ),
        }
    )
    reservation_overlap = set(old_exact).intersection(reservation_exact)
    comparisons.append(
        {
            "target": "human_reference_reserved_groups",
            "records": len(reservation_rows),
            "source_sha256": _sha256_file(args.reservations),
            "exact_text_hash_overlap": len(reservation_overlap),
            "normalized_text_hash_overlap": None,
            "overlap_old_sample_ids": sorted(
                old_exact[item] for item in reservation_overlap
            ),
            "overlap_reference_sample_ids": sorted(
                reservation_exact[item] for item in reservation_overlap
            ),
        }
    )

    union_exact: set[str] = set()
    union_normalized: set[str] = set()
    for package in new_packages:
        pseudo_path = package / "ai_pseudo_labels.jsonl"
        rows = _read_jsonl(pseudo_path)
        exact, normalized = _text_index(
            rows,
            text_key="reviewContent",
            id_key="sample_id",
        )
        exact_overlap = set(old_exact).intersection(exact)
        normalized_overlap = set(old_normalized).intersection(normalized)
        union_exact.update(exact_overlap)
        union_normalized.update(normalized_overlap)
        comparisons.append(
            {
                "target": package.parent.name,
                "records": len(rows),
                "source_sha256": _sha256_file(pseudo_path),
                "exact_text_hash_overlap": len(exact_overlap),
                "normalized_text_hash_overlap": len(normalized_overlap),
                "overlap_old_sample_ids_exact": sorted(
                    old_exact[item] for item in exact_overlap
                ),
                "overlap_old_sample_ids_normalized": sorted(
                    old_normalized[item] for item in normalized_overlap
                ),
            }
        )

    output = {
        "schema_version": "legacy-old-cross-corpus-overlap-audit/1.0.0",
        "status": "COMPLETED",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "old_source": {
            "path": args.old.as_posix(),
            "records": len(old_rows),
            "sha256": _sha256_file(args.old),
        },
        "normalization": "NFKC + casefold + whitespace collapse + strip",
        "comparisons": comparisons,
        "new_pseudo_label_union": {
            "exact_text_hash_overlap": len(union_exact),
            "normalized_text_hash_overlap": len(union_normalized),
            "overlap_old_sample_ids_exact": sorted(
                old_exact[item] for item in union_exact
            ),
            "overlap_old_sample_ids_normalized": sorted(
                old_normalized[item] for item in union_normalized
            ),
        },
        "decisions": [
            "No prompt-calibration text overlap is permitted.",
            "Reserved-reference and prior-pseudo overlaps are disclosed and "
            "must not be counted as independent benchmark evidence.",
            "Cross-corpus overlap does not import or reuse a historical label.",
        ],
    }
    if calibration_exact_overlap or calibration_normalized_overlap:
        raise ValueError("Legacy targets overlap prompt calibration text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
