"""Reproduce and package the frozen 60-record ABSA semantic audit.

This script does not alter model annotations.  It reconstructs the exact
six-stratum sample that was inspected while the primary run was in progress,
verifies the frozen input and annotation closures, and emits an AI semantic
audit ledger.  The ledger is diagnostic evidence pending independent human
adjudication; it is not human gold, IAA, or an accuracy estimate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lazada_collector.ai_tranche import exact_occurrence, sha256_file
from lazada_collector.llm_annotation import (
    ASPECT_COLUMNS,
    canonical_json,
    label_vector,
    sha256_text,
    validate_and_normalize_annotation,
)


DEFAULT_PACKAGE = Path(
    "data/annotations/absa_ai_tranche_5000_v1_20260727"
)
DEFAULT_AUDIT_RELATIVE = Path("audits/semantic_audit_60_v1")

AUDIT_ID = "semantic-audit-60-v1"
AUDIT_SCHEMA_VERSION = "absa-ai-semantic-audit-ledger/1.0.0"
SUMMARY_SCHEMA_VERSION = "absa-ai-semantic-audit-summary/1.0.0"
MANIFEST_SCHEMA_VERSION = "absa-ai-semantic-audit-manifest/1.0.0"

AUDIT_SEED = "semantic-audit-v1"
SNAPSHOT_MAX_SELECTION_RANK = 1060
SNAPSHOT_RECORDS = 1060
STRATUM_SIZE = 10
AUDIT_FROZEN_AT_UTC = "2026-07-27T08:12:10Z"
SNAPSHOT_BOUNDARY_COMPLETED_AT_UTC = "2026-07-27T08:02:37.9907101Z"

EXPECTED_PREPARE_MANIFEST_SHA256 = (
    "067ed254b6075c7b3d93e8c63d200ec5a5041f41ac92b7e7cf9f846355304726"
)
EXPECTED_BLIND_INPUT_SHA256 = (
    "7a58187242eef20380b51121a2bd71d7ccab4516ffb87e28ce97b5655b8e8ad3"
)
EXPECTED_GUIDELINE_SHA256 = (
    "58ae9f19d1921fa6ce9933f85a1af2ed47a72601bfa8b1a91f48b991fb3aa1f9"
)
EXPECTED_SNAPSHOT_CLOSURE_SHA256 = (
    "a74803bd43a8423a49376bbad230140f55534b918417749301a93a5fd01ab398"
)
EXPECTED_SAMPLE_MEMBERSHIP_SHA256 = (
    "5f5e89c6073213dd83e90a13ad5792986ce3e9e11c0bafd203309e469d667c8b"
)
EXPECTED_SELECTED_ANNOTATION_CLOSURE_SHA256 = (
    "1a0248729e91bd40a444da9533dbbdb1011812ac634e7e3de76a7c91e1b4336b"
)
EXPECTED_SELECTED_REVIEW_CLOSURE_SHA256 = (
    "87f5cf1e5a9cc825ea20e2380245ee2bacef2ffb76ac698e74661c9791eb7293"
)

SEVERITY_OK = "NO_MATERIAL_ISSUE"
SEVERITY_MAJOR = "MAJOR"
SEVERITY_MINOR = "MINOR"
VALID_SEVERITIES = {SEVERITY_OK, SEVERITY_MAJOR, SEVERITY_MINOR}

STRATA = (
    "reject",
    "escalate",
    "neutral",
    "mixed",
    "high",
    "clear",
)

# Ordered exactly as the original SHA-256 selection.  The script independently
# reconstructs this order and refuses to publish if any member changes.
EXPECTED_MEMBERSHIP: tuple[tuple[str, int, str], ...] = (
    ("reject", 254, "a5k-1f7e14efc733cca6f325"),
    ("reject", 1057, "a5k-0a814a2fd410c0285343"),
    ("reject", 533, "a5k-bf46708a3a941b729c99"),
    ("reject", 190, "a5k-ee4044bf25e2760965da"),
    ("reject", 34, "a5k-7384043e52cbcb388a81"),
    ("reject", 922, "a5k-c7189e4b0d799bd1d393"),
    ("reject", 300, "a5k-38f3af2a52dfefd35fee"),
    ("reject", 999, "a5k-9569cd3a8c257b716d70"),
    ("reject", 718, "a5k-b59d1fff471b3d54456b"),
    ("reject", 888, "a5k-d71d0519ea3678be977e"),
    ("escalate", 104, "a5k-4e04b09e292069311197"),
    ("escalate", 509, "a5k-8b7a9db998bc7e398b83"),
    ("escalate", 374, "a5k-9b129e122ab80865f148"),
    ("escalate", 838, "a5k-5e903366ba955b64c58f"),
    ("escalate", 795, "a5k-1cbe4db91d32146c2c66"),
    ("escalate", 116, "a5k-406ee9fe2ab147dbf77d"),
    ("escalate", 524, "a5k-521cd48342f5ce06a41a"),
    ("escalate", 954, "a5k-846a51575228a541359f"),
    ("escalate", 266, "a5k-f8e2b90f0e42da792190"),
    ("escalate", 522, "a5k-d82e8076c709cb17966f"),
    ("neutral", 1023, "a5k-0a5fb3ee3a09b1c36796"),
    ("neutral", 384, "a5k-17183919be7a7f28003f"),
    ("neutral", 163, "a5k-29e4d4081b18626fd6fe"),
    ("neutral", 270, "a5k-5235b6aeb2640c91b548"),
    ("neutral", 916, "a5k-fc6fe52a3a70dc7005dd"),
    ("neutral", 202, "a5k-03729acfccbf6084d555"),
    ("neutral", 5, "a5k-06ba5936965aff164041"),
    ("neutral", 480, "a5k-29d94fb883a08509f558"),
    ("neutral", 48, "a5k-9d65f0f2e2e624db5b68"),
    ("neutral", 213, "a5k-980ddf846db21eee1902"),
    ("mixed", 661, "a5k-1248b2525f7b9e03a9ea"),
    ("mixed", 229, "a5k-167d125c62bd541d4154"),
    ("mixed", 100, "a5k-ef003df93aaa0714b5dc"),
    ("mixed", 802, "a5k-e1cbdc205a82352e0aa8"),
    ("mixed", 804, "a5k-9b4ac1afcde7fdb3ec3e"),
    ("mixed", 205, "a5k-991bd18a74dcc2dd3838"),
    ("mixed", 58, "a5k-a20c7a7d91c46482c3d0"),
    ("mixed", 846, "a5k-6a4561f2f2e1807923e4"),
    ("mixed", 967, "a5k-de7c0e2cbc54b3ddee07"),
    ("mixed", 840, "a5k-e4ccdb9ffc32e79c3327"),
    ("high", 56, "a5k-e1990c84cb86cb8c2bbf"),
    ("high", 657, "a5k-b084816c5e6a989b7173"),
    ("high", 178, "a5k-bd267dc83a7bc8593ada"),
    ("high", 1052, "a5k-b6df198c4944bb1026a1"),
    ("high", 360, "a5k-2b9022fde732937cc871"),
    ("high", 806, "a5k-5d981a80ee3ed08ef76d"),
    ("high", 179, "a5k-36d027e13b031d588165"),
    ("high", 155, "a5k-80f569ac5b95e1dbaf3d"),
    ("high", 885, "a5k-c82bbccd0fddcc8b7427"),
    ("high", 109, "a5k-a243248e0e54289ae609"),
    ("clear", 826, "a5k-d75cb9df0f47e6e42467"),
    ("clear", 519, "a5k-a57a42c525545505a258"),
    ("clear", 1059, "a5k-108e20fb3e918b3dff3d"),
    ("clear", 304, "a5k-1c3c015306e38b712afd"),
    ("clear", 733, "a5k-557757384b637ae724e6"),
    ("clear", 584, "a5k-c4d3fd52822efde39442"),
    ("clear", 159, "a5k-63aee23c1c04b218531f"),
    ("clear", 895, "a5k-7501d83d21bb3a060e2d"),
    ("clear", 543, "a5k-c6a713a3e12c103932b4"),
    ("clear", 769, "a5k-3b14a20cd2343f5d68ba"),
)


class SemanticAuditError(ValueError):
    """Raised when the frozen audit cannot be reproduced safely."""


@dataclass(frozen=True)
class AuditDecision:
    severity: str
    issue_codes: tuple[str, ...]
    issue_text: str


ISSUES: dict[str, AuditDecision] = {
    "a5k-b59d1fff471b3d54456b": AuditDecision(
        SEVERITY_MAJOR,
        ("STATUS_FALSE_REJECT",),
        "Nội dung mua đủ combo là phát biểu mua hàng có nghĩa; "
        "REJECT_NON_REVIEW quá mạnh và cần curator/human adjudication.",
    ),
    "a5k-8b7a9db998bc7e398b83": AuditDecision(
        SEVERITY_MAJOR,
        ("POLARITY_ERROR", "AUTHENTICITY_UNCERTAINTY"),
        "'Chất kem lỏng' chưa đủ căn cứ cho Quality=-1; "
        "'có thể là hàng thật' phải là Authenticity=0 thay vì +1.",
    ),
    "a5k-846a51575228a541359f": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_OMISSION",),
        "Bỏ sót Shipping=+1 từ exact evidence 'giao hàng nhanh'.",
    ),
    "a5k-d82e8076c709cb17966f": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_BOUNDARY", "POLARITY_ERROR"),
        "'Không phải kaki' thuộc Description=-1, không phải Quality negative; "
        "Quality nên +1 và Description nên mixed với 'Kích thước đúng'.",
    ),
    "a5k-06ba5936965aff164041": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_BOUNDARY",),
        "'Đặt nhầm size' là lỗi chọn của người mua, không phải Description=0; "
        "Performance cũng cần xem lại phạm vi êm/không đau so với hơi chật.",
    ),
    "a5k-29d94fb883a08509f558": AuditDecision(
        SEVERITY_MAJOR,
        ("STATUS_FALSE_LABEL", "NON_REVIEW"),
        "Văn bản quảng cáo/khuyên dùng Bakkaland không có trải nghiệm người mua; "
        "không nên LABELED với Quality=0.",
    ),
    "a5k-980ddf846db21eee1902": AuditDecision(
        SEVERITY_MAJOR,
        ("POLARITY_ERROR",),
        "'Nồi chiên của Philip số 1 rồi' hỗ trợ Quality=+1; "
        "'chưa dùng' không biện minh cho Quality=0.",
    ),
    "a5k-108e20fb3e918b3dff3d": AuditDecision(
        SEVERITY_MAJOR,
        ("STATUS_FALSE_LABEL", "NON_REVIEW"),
        "Chuỗi claim tính năng mang phong cách quảng cáo, không có trải nghiệm "
        "người mua; cần REJECT hoặc ESCALATE NON_REVIEW.",
    ),
    "a5k-1c3c015306e38b712afd": AuditDecision(
        SEVERITY_MAJOR,
        ("STATUS_FALSE_LABEL", "NON_REVIEW"),
        "Chuỗi claim quảng cáo pin sạc không có trải nghiệm người mua; "
        "không nên phát hành như LABELED.",
    ),
    "a5k-1248b2525f7b9e03a9ea": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_BOUNDARY",),
        "Calibration đã chấp nhận ánh xạ mùi/vị sang Performance: "
        "Performance nên mixed; Quality chỉ còn positive từ tự nhiên/không phẩm màu.",
    ),
    "a5k-e1cbdc205a82352e0aa8": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_OMISSION",),
        "Bỏ sót Performance negative hoặc mixed từ một bình bình thường và "
        "một bình bị giọt/rò nước.",
    ),
    "a5k-991bd18a74dcc2dd3838": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_BOUNDARY", "ASPECT_OMISSION"),
        "'Vạch màu cam hơi khó nhìn' là trải nghiệm/readability "
        "Performance=-1; Quality có bằng chứng positive riêng.",
    ),
    "a5k-a20c7a7d91c46482c3d0": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_BOUNDARY", "SPURIOUS_POLARITY"),
        "Thiếu hộp/bao tay là product-content Description, không phải transport "
        "Packaging; Packaging không nên mixed từ cùng evidence này.",
    ),
    "a5k-5d981a80ee3ed08ef76d": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_OMISSION",),
        "Bỏ sót Quality=+1 từ ý 'size lớn, thích quá' trong câu có lỗi gõ.",
    ),
    "a5k-c82bbccd0fddcc8b7427": AuditDecision(
        SEVERITY_MAJOR,
        ("ASPECT_OMISSION",),
        "Bỏ sót ShopService=+1 từ phát biểu có đích rõ "
        "'cửa hàng rất OK'.",
    ),
    "a5k-e1990c84cb86cb8c2bbf": AuditDecision(
        SEVERITY_MINOR,
        ("CALIBRATION_INCONSISTENCY",),
        "'Tặng shop 5 sao' có thể tạo ShopService=+1 khi đích là shop, "
        "nhưng calibration hiện xử lý các câu gần giống nhau không nhất quán.",
    ),
    "a5k-ef003df93aaa0714b5dc": AuditDecision(
        SEVERITY_MINOR,
        ("EVIDENCE_OMISSION", "POLARITY_SCOPE"),
        "Bỏ sót evidence Quality-positive 'Kích thước:rộng rãi'; "
        "'Thiết kế:ổn' còn ranh giới neutral/positive.",
    ),
    "a5k-9b4ac1afcde7fdb3ec3e": AuditDecision(
        SEVERITY_MINOR,
        ("EVIDENCE_POLARITY", "EVIDENCE_OMISSION"),
        "'Âm lượng vừa phải' nên neutral thay vì positive và pin 3h bị bỏ sót; "
        "vector Performance vẫn mixed nhờ các evidence khác.",
    ),
    "a5k-e4ccdb9ffc32e79c3327": AuditDecision(
        SEVERITY_MINOR,
        ("POLARITY_SCOPE",),
        "Phạm vi 'xài chống cháy thì ok, vẫn nên kiếm cục khác' chưa được giữ "
        "trọn; mixed hay negative cần human adjudication.",
    ),
}

NO_ISSUE = AuditDecision(
    SEVERITY_OK,
    (),
    "Không phát hiện lỗi ngữ nghĩa vật chất trong lần AI semantic audit này.",
)

_RECORD_FILENAME = re.compile(r"^(?P<rank>\d{5})-(?P<annotation_id>a5k-.+)\.json$")


def _hash_rank(seed: str, annotation_id: str) -> str:
    return sha256_text(f"{seed}|{annotation_id}")


def _labels(record: Mapping[str, Any]) -> list[Any]:
    annotation = record.get("annotation")
    aspects = annotation.get("aspects") if isinstance(annotation, Mapping) else None
    if not isinstance(aspects, list) or len(aspects) != len(ASPECT_COLUMNS):
        raise SemanticAuditError(
            f"Malformed annotation aspects for {record.get('annotation_id')!r}"
        )
    return [item.get("label") if isinstance(item, Mapping) else None for item in aspects]


def select_semantic_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: str = AUDIT_SEED,
    stratum_size: int = STRATUM_SIZE,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Select six disjoint SHA-ranked strata in the frozen priority order."""

    if stratum_size < 1:
        raise SemanticAuditError("stratum_size must be positive")
    used: set[str] = set()
    selected: list[tuple[str, Mapping[str, Any]]] = []

    def status(record: Mapping[str, Any]) -> Any:
        annotation = record.get("annotation")
        return (
            annotation.get("annotation_status")
            if isinstance(annotation, Mapping)
            else None
        )

    def mentioned_count(record: Mapping[str, Any]) -> int:
        return sum(label is not None and label != 2 for label in _labels(record))

    predicates: tuple[
        tuple[str, Callable[[Mapping[str, Any]], bool]], ...
    ] = (
        ("reject", lambda row: status(row) == "REJECT_NON_REVIEW"),
        ("escalate", lambda row: status(row) == "ESCALATE"),
        (
            "neutral",
            lambda row: status(row) == "LABELED"
            and 0 in _labels(row)
            and "1, -1" not in _labels(row),
        ),
        (
            "mixed",
            lambda row: status(row) == "LABELED"
            and "1, -1" in _labels(row),
        ),
        (
            "high",
            lambda row: status(row) == "LABELED"
            and 0 not in _labels(row)
            and "1, -1" not in _labels(row)
            and mentioned_count(row) >= 4,
        ),
        (
            "clear",
            lambda row: status(row) == "LABELED"
            and 0 not in _labels(row)
            and "1, -1" not in _labels(row)
            and mentioned_count(row) <= 3,
        ),
    )

    for stratum, predicate in predicates:
        pool = [
            row
            for row in records
            if row.get("annotation_id") not in used and predicate(row)
        ]
        pool.sort(key=lambda row: _hash_rank(seed, str(row["annotation_id"])))
        if len(pool) < stratum_size:
            raise SemanticAuditError(
                f"Stratum {stratum!r} has {len(pool)} records; "
                f"needs {stratum_size}"
            )
        for row in pool[:stratum_size]:
            annotation_id = row.get("annotation_id")
            if not isinstance(annotation_id, str):
                raise SemanticAuditError("Selected record has no annotation_id")
            used.add(annotation_id)
            selected.append((stratum, row))
    return selected


def membership_rows(
    sample: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[tuple[str, int, str], ...]:
    output: list[tuple[str, int, str]] = []
    for stratum, row in sample:
        rank = row.get("selection_rank")
        annotation_id = row.get("annotation_id")
        if not isinstance(rank, int) or not isinstance(annotation_id, str):
            raise SemanticAuditError("Selected membership fields are malformed")
        output.append((stratum, rank, annotation_id))
    return tuple(output)


def membership_sha256(
    membership: Iterable[tuple[str, int, str]],
) -> str:
    serialized = "".join(
        f"{stratum}\t{rank}\t{annotation_id}\n"
        for stratum, rank, annotation_id in membership
    )
    return sha256_text(serialized)


def require_expected_membership(
    actual: Sequence[tuple[str, int, str]],
    expected: Sequence[tuple[str, int, str]] = EXPECTED_MEMBERSHIP,
) -> None:
    actual_tuple = tuple(actual)
    expected_tuple = tuple(expected)
    if actual_tuple != expected_tuple:
        first_difference: dict[str, Any] | None = None
        for index in range(max(len(actual_tuple), len(expected_tuple))):
            actual_item = actual_tuple[index] if index < len(actual_tuple) else None
            expected_item = (
                expected_tuple[index] if index < len(expected_tuple) else None
            )
            if actual_item != expected_item:
                first_difference = {
                    "index": index,
                    "expected": expected_item,
                    "actual": actual_item,
                }
                break
        raise SemanticAuditError(
            "Frozen semantic-audit membership changed; refusing to write. "
            f"first_difference={first_difference}"
        )
    actual_sha = membership_sha256(actual_tuple)
    expected_sha = (
        EXPECTED_SAMPLE_MEMBERSHIP_SHA256
        if expected_tuple == EXPECTED_MEMBERSHIP
        else membership_sha256(expected_tuple)
    )
    if actual_sha != expected_sha:
        raise SemanticAuditError(
            "Frozen semantic-audit membership hash mismatch: "
            f"{actual_sha}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticAuditError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticAuditError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise SemanticAuditError(
                        f"Blank JSONL line at {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SemanticAuditError(
                        f"Expected JSON object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticAuditError(f"Cannot read JSONL {path}: {exc}") from exc
    return rows


def _unique_by(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    *,
    context: str,
) -> dict[Any, Mapping[str, Any]]:
    output: dict[Any, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            raise SemanticAuditError(f"{context} has empty {key}")
        if value in output:
            raise SemanticAuditError(
                f"Duplicate {key} in {context}: {value!r}"
            )
        output[value] = row
    return output


def _load_snapshot_records(package: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    records_dir = package / "runs" / "primary" / "records"
    if not records_dir.is_dir():
        raise SemanticAuditError(f"Primary records directory missing: {records_dir}")

    rows: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for path in records_dir.glob("*.json"):
        match = _RECORD_FILENAME.fullmatch(path.name)
        if match is None:
            continue
        filename_rank = int(match.group("rank"))
        if filename_rank > SNAPSHOT_MAX_SELECTION_RANK:
            continue
        row = _read_json(path)
        annotation_id = row.get("annotation_id")
        rank = row.get("selection_rank")
        if annotation_id != match.group("annotation_id") or rank != filename_rank:
            raise SemanticAuditError(
                f"Record filename/payload mismatch: {path.name}"
            )
        if annotation_id in paths:
            raise SemanticAuditError(
                f"Duplicate snapshot annotation_id: {annotation_id}"
            )
        paths[str(annotation_id)] = path
        rows.append(row)

    rows.sort(key=lambda row: int(row["selection_rank"]))
    if len(rows) != SNAPSHOT_RECORDS:
        raise SemanticAuditError(
            f"Frozen snapshot needs {SNAPSHOT_RECORDS} records, found {len(rows)}"
        )
    ranks = [row.get("selection_rank") for row in rows]
    if ranks != list(range(1, SNAPSHOT_RECORDS + 1)):
        raise SemanticAuditError("Frozen snapshot ranks are not exactly 1..1060")

    snapshot_serialized = ""
    for row in rows:
        annotation = row.get("annotation")
        if not isinstance(annotation, Mapping):
            raise SemanticAuditError(
                f"Missing annotation in {row.get('annotation_id')!r}"
            )
        annotation_sha = sha256_text(canonical_json(annotation))
        snapshot_serialized += (
            f"{row['selection_rank']}\t{row['annotation_id']}\t"
            f"{row.get('review_text_sha256')}\t{annotation_sha}\n"
        )
    snapshot_sha = sha256_text(snapshot_serialized)
    if snapshot_sha != EXPECTED_SNAPSHOT_CLOSURE_SHA256:
        raise SemanticAuditError(
            "Frozen 1,060-record snapshot closure changed; refusing to write. "
            f"actual={snapshot_sha}"
        )
    return rows, paths


def _denormalize_annotation(
    review_text: str,
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    aspects: list[dict[str, Any]] = []
    raw_aspects = annotation.get("aspects")
    if not isinstance(raw_aspects, list):
        raise SemanticAuditError("Normalized annotation has no aspects")
    for raw_aspect in raw_aspects:
        if not isinstance(raw_aspect, Mapping):
            raise SemanticAuditError("Malformed normalized aspect")
        evidence: list[dict[str, Any]] = []
        raw_evidence = raw_aspect.get("evidence")
        if not isinstance(raw_evidence, list):
            raise SemanticAuditError("Malformed normalized evidence")
        for item in raw_evidence:
            if not isinstance(item, Mapping):
                raise SemanticAuditError("Malformed evidence item")
            quote = item.get("text")
            start = item.get("start")
            end = item.get("end")
            if (
                not isinstance(quote, str)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or review_text[start:end] != quote
            ):
                raise SemanticAuditError(
                    "Evidence text/offset does not replay exactly"
                )
            evidence.append(
                {
                    "quote": quote,
                    "occurrence": exact_occurrence(review_text, quote, start),
                    "polarity": item.get("polarity"),
                }
            )
        aspects.append(
            {
                "aspect": raw_aspect.get("aspect"),
                "label": raw_aspect.get("label"),
                "evidence": evidence,
                "uncertainty_codes": raw_aspect.get("uncertainty_codes"),
            }
        )
    return {
        "annotation_status": annotation.get("annotation_status"),
        "aspects": aspects,
        "review_uncertainty_codes": annotation.get(
            "review_uncertainty_codes"
        ),
        "notes": annotation.get("notes"),
    }


def _verify_and_build(
    package: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    package = package.resolve()
    prepare_path = package / "prepare_manifest.json"
    blind_path = package / "input" / "blind_reviews.jsonl"
    guideline_path = package / "provenance" / "ABSA_ANNOTATION_GUIDELINE_V2.md"
    sums_path = package / "INPUT_SHA256SUMS.txt"

    source_hashes = {
        "prepare_manifest_sha256": sha256_file(prepare_path),
        "blind_reviews_sha256": sha256_file(blind_path),
        "guideline_sha256": sha256_file(guideline_path),
    }
    expected_hashes = {
        "prepare_manifest_sha256": EXPECTED_PREPARE_MANIFEST_SHA256,
        "blind_reviews_sha256": EXPECTED_BLIND_INPUT_SHA256,
        "guideline_sha256": EXPECTED_GUIDELINE_SHA256,
    }
    if source_hashes != expected_hashes:
        raise SemanticAuditError(
            "Frozen source hash mismatch; refusing to write. "
            f"expected={expected_hashes}, actual={source_hashes}"
        )

    prepare = _read_json(prepare_path)
    artifacts = {
        item.get("path"): item
        for item in prepare.get("artifacts", [])
        if isinstance(item, Mapping)
    }
    blind_artifact = artifacts.get("input/blind_reviews.jsonl")
    if (
        not isinstance(blind_artifact, Mapping)
        or blind_artifact.get("sha256") != EXPECTED_BLIND_INPUT_SHA256
        or blind_artifact.get("records") != 5000
    ):
        raise SemanticAuditError("Prepared manifest blind-input binding changed")
    if prepare.get("guideline", {}).get("sha256") != EXPECTED_GUIDELINE_SHA256:
        raise SemanticAuditError("Prepared manifest guideline binding changed")

    sums = sums_path.read_text(encoding="utf-8").splitlines()
    sum_map: dict[str, str] = {}
    for line in sums:
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise SemanticAuditError("Malformed INPUT_SHA256SUMS.txt")
        checksum, relative = parts
        if relative in sum_map:
            raise SemanticAuditError(f"Duplicate input checksum path: {relative}")
        sum_map[relative] = checksum
    if (
        sum_map.get("prepare_manifest.json")
        != EXPECTED_PREPARE_MANIFEST_SHA256
        or sum_map.get("input/blind_reviews.jsonl")
        != EXPECTED_BLIND_INPUT_SHA256
        or sum_map.get("provenance/ABSA_ANNOTATION_GUIDELINE_V2.md")
        != EXPECTED_GUIDELINE_SHA256
    ):
        raise SemanticAuditError("INPUT_SHA256SUMS source bindings changed")

    blind_rows = _read_jsonl(blind_path)
    if len(blind_rows) != 5000:
        raise SemanticAuditError(
            f"Blind input needs 5,000 rows, found {len(blind_rows)}"
        )
    blind_by_id = _unique_by(
        blind_rows,
        "annotation_id",
        context="blind_reviews.jsonl",
    )

    snapshot, record_paths = _load_snapshot_records(package)
    sample = select_semantic_sample(snapshot)
    actual_membership = membership_rows(sample)
    require_expected_membership(actual_membership)

    annotation_closure = ""
    review_closure = ""
    ledger: list[dict[str, Any]] = []
    evidence_spans = 0
    records_with_repairs = 0
    severity_by_stratum: dict[str, Counter[str]] = {
        stratum: Counter() for stratum in STRATA
    }

    for audit_order, (stratum, record) in enumerate(sample, 1):
        annotation_id = str(record["annotation_id"])
        rank = int(record["selection_rank"])
        blind = blind_by_id.get(annotation_id)
        if blind is None:
            raise SemanticAuditError(f"Selected ID missing from blind input: {annotation_id}")
        review_text = blind.get("reviewContent")
        review_sha = blind.get("review_text_sha256")
        if (
            not isinstance(review_text, str)
            or not isinstance(review_sha, str)
            or sha256_text(review_text) != review_sha
            or record.get("review_text_sha256") != review_sha
            or blind.get("selection_rank") != rank
        ):
            raise SemanticAuditError(
                f"Review/rank binding mismatch for {annotation_id}"
            )

        annotation = record.get("annotation")
        if not isinstance(annotation, Mapping):
            raise SemanticAuditError(f"Missing annotation: {annotation_id}")
        normalized = validate_and_normalize_annotation(
            review_text,
            _denormalize_annotation(review_text, annotation),
        )
        if canonical_json(normalized) != canonical_json(annotation):
            raise SemanticAuditError(
                f"Normalized annotation replay mismatch: {annotation_id}"
            )
        annotation_sha = sha256_text(canonical_json(annotation))
        annotation_closure += (
            f"{stratum}\t{rank}\t{annotation_id}\t{annotation_sha}\n"
        )
        review_closure += (
            f"{stratum}\t{rank}\t{annotation_id}\t{review_sha}\n"
        )

        aspects = annotation["aspects"]
        record_evidence = sum(len(item["evidence"]) for item in aspects)
        evidence_spans += record_evidence
        repairs = record.get("normalization_repairs")
        if not isinstance(repairs, list):
            raise SemanticAuditError(
                f"normalization_repairs must be a list: {annotation_id}"
            )
        if repairs:
            records_with_repairs += 1

        decision = ISSUES.get(annotation_id, NO_ISSUE)
        if decision.severity not in VALID_SEVERITIES:
            raise SemanticAuditError(
                f"Invalid audit severity for {annotation_id}"
            )
        severity_by_stratum[stratum][decision.severity] += 1
        ledger.append(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "audit_id": AUDIT_ID,
                "audit_order": audit_order,
                "audit_stratum": stratum,
                "selection_rank": rank,
                "annotation_id": annotation_id,
                "review_text_sha256": review_sha,
                "source_run_record": record_paths[annotation_id]
                .relative_to(package)
                .as_posix(),
                "source_annotation_sha256": annotation_sha,
                "original_annotation_status": annotation.get(
                    "annotation_status"
                ),
                "original_label_vector": label_vector(annotation),
                "evidence_spans_verified": record_evidence,
                "normalization_repairs_observed": len(repairs),
                "audit_severity": decision.severity,
                "issue_codes": list(decision.issue_codes),
                "issue_text": decision.issue_text,
                "human_adjudication_status": (
                    "NOT_REQUESTED_BY_AI_AUDIT"
                    if decision.severity == SEVERITY_OK
                    else "PENDING"
                ),
                "label_mutation_performed": False,
                "auditor_type": "AI_SEMANTIC_REVIEW",
                "independent_human_gold": False,
                "guideline_version": "2.0.0",
            }
        )

    annotation_closure_sha = sha256_text(annotation_closure)
    review_closure_sha = sha256_text(review_closure)
    if annotation_closure_sha != EXPECTED_SELECTED_ANNOTATION_CLOSURE_SHA256:
        raise SemanticAuditError(
            "Selected annotation closure changed; refusing to write. "
            f"actual={annotation_closure_sha}"
        )
    if review_closure_sha != EXPECTED_SELECTED_REVIEW_CLOSURE_SHA256:
        raise SemanticAuditError(
            "Selected review closure changed; refusing to write. "
            f"actual={review_closure_sha}"
        )
    if evidence_spans != 188 or records_with_repairs != 0:
        raise SemanticAuditError(
            "Frozen technical audit counts changed; "
            f"evidence_spans={evidence_spans}, "
            f"records_with_repairs={records_with_repairs}"
        )

    overall_severity = Counter(
        row["audit_severity"] for row in ledger
    )
    expected_severity = Counter(
        {
            SEVERITY_OK: 41,
            SEVERITY_MAJOR: 15,
            SEVERITY_MINOR: 4,
        }
    )
    if overall_severity != expected_severity:
        raise SemanticAuditError(
            f"Frozen audit decision counts changed: {overall_severity}"
        )

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "artifact_type": "AI_SEMANTIC_AUDIT_NOT_HUMAN_GOLD",
        "status": "FROZEN_PENDING_INDEPENDENT_HUMAN_ADJUDICATION",
        "audit_frozen_at_utc": AUDIT_FROZEN_AT_UTC,
        "scope": {
            "package_root": ".",
            "tranche_id": prepare.get("tranche_id"),
            "snapshot_records": SNAPSHOT_RECORDS,
            "snapshot_max_selection_rank": SNAPSHOT_MAX_SELECTION_RANK,
            "snapshot_boundary_completed_at_utc": (
                SNAPSHOT_BOUNDARY_COMPLETED_AT_UTC
            ),
            "sample_records": len(ledger),
            "strata": {stratum: STRATUM_SIZE for stratum in STRATA},
        },
        "selection_method": {
            "seed": AUDIT_SEED,
            "method": (
                "Within the frozen selection_rank<=1060 snapshot, process "
                "reject, escalate, neutral, mixed, high-aspect, then clear; "
                "exclude earlier members and take the first 10 by ascending "
                "SHA256(seed + '|' + annotation_id) in each stratum."
            ),
            "high_definition": (
                "LABELED, no neutral/mixed cell, at least four mentioned aspects"
            ),
            "clear_definition": (
                "LABELED, no neutral/mixed cell, at most three mentioned aspects"
            ),
            "membership_sha256": EXPECTED_SAMPLE_MEMBERSHIP_SHA256,
            "membership_serialization": (
                "UTF-8 stratum<TAB>selection_rank<TAB>annotation_id<LF>, "
                "in stratum/hash selection order, including terminal LF"
            ),
        },
        "technical_verification": {
            "records_with_nine_aspects": 60,
            "evidence_spans_replayed": evidence_spans,
            "evidence_offset_or_text_mismatches": 0,
            "records_with_normalization_repairs": records_with_repairs,
            "snapshot_closure_sha256": EXPECTED_SNAPSHOT_CLOSURE_SHA256,
            "selected_annotation_closure_sha256": annotation_closure_sha,
            "selected_review_closure_sha256": review_closure_sha,
        },
        "semantic_findings": {
            "severity": dict(sorted(overall_severity.items())),
            "by_stratum": {
                stratum: dict(sorted(severity_by_stratum[stratum].items()))
                for stratum in STRATA
            },
            "major_issue_records": sorted(
                row["annotation_id"]
                for row in ledger
                if row["audit_severity"] == SEVERITY_MAJOR
            ),
            "minor_issue_records": sorted(
                row["annotation_id"]
                for row in ledger
                if row["audit_severity"] == SEVERITY_MINOR
            ),
            "dominant_patterns": [
                "Missed or misallocated secondary aspects.",
                "Marketing-like non-review text incorrectly left LABELED.",
                "Quality/Performance/Description/Packaging boundary errors.",
                "Inconsistent calibration treatment of shop-targeted five-star text.",
            ],
            "continuation_decision": (
                "Supports continued candidate pseudo-label generation only; "
                "does not support direct training-gold publication."
            ),
        },
        "source_hashes": source_hashes,
        "limitations": [
            "The sample deliberately over-represents difficult strata; its "
            "severity proportions are not a corpus-wide accuracy estimate.",
            "The human calibration was AI-seeded and human-reviewed, not an "
            "independent double-blind gold set, and contains unresolved "
            "convention inconsistencies.",
            "This ledger is an AI semantic audit. It is not human accuracy, "
            "IAA, expert adjudication, or permission to mutate labels.",
            "All source annotations remain AI pseudo-labels pending human "
            "verification.",
        ],
        "label_mutations": 0,
    }
    return ledger, summary, source_hashes


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")


def _artifact(path: str, payload: bytes, *, records: int | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if records is not None:
        item["records"] = records
    return item


def build_artifact_payloads(
    package: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    ledger, summary, source_hashes = _verify_and_build(package)
    ledger_bytes = _jsonl_bytes(ledger)
    summary_bytes = _json_bytes(summary)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "artifact_type": "AI_SEMANTIC_AUDIT_NOT_HUMAN_GOLD",
        "status": "FROZEN_PENDING_INDEPENDENT_HUMAN_ADJUDICATION",
        "audit_frozen_at_utc": AUDIT_FROZEN_AT_UTC,
        "label_mutations": 0,
        "source_hashes": source_hashes,
        "closures": {
            "snapshot_sha256": EXPECTED_SNAPSHOT_CLOSURE_SHA256,
            "sample_membership_sha256": EXPECTED_SAMPLE_MEMBERSHIP_SHA256,
            "selected_annotation_sha256": (
                EXPECTED_SELECTED_ANNOTATION_CLOSURE_SHA256
            ),
            "selected_review_sha256": EXPECTED_SELECTED_REVIEW_CLOSURE_SHA256,
        },
        "artifacts": [
            _artifact("audit_ledger.jsonl", ledger_bytes, records=len(ledger)),
            _artifact("summary.json", summary_bytes, records=1),
        ],
        "publication_note": (
            "These artifacts may accompany final pseudo-label publication, "
            "but do not convert any annotation into human gold."
        ),
    }
    manifest_bytes = _json_bytes(manifest)
    sums_rows = [
        (hashlib.sha256(ledger_bytes).hexdigest(), "audit_ledger.jsonl"),
        (hashlib.sha256(summary_bytes).hexdigest(), "summary.json"),
        (hashlib.sha256(manifest_bytes).hexdigest(), "manifest.json"),
    ]
    sums_bytes = "".join(
        f"{checksum}  {relative}\n"
        for checksum, relative in sums_rows
    ).encode("utf-8")
    return {
        "audit_ledger.jsonl": ledger_bytes,
        "summary.json": summary_bytes,
        "manifest.json": manifest_bytes,
        "SHA256SUMS.txt": sums_bytes,
    }, manifest


def _require_audit_destination(package: Path, output: Path) -> None:
    package = package.resolve()
    output = output.resolve()
    audit_root = (package / "audits").resolve()
    try:
        output.relative_to(audit_root)
    except ValueError as exc:
        raise SemanticAuditError(
            f"Audit output must remain below {audit_root}: {output}"
        ) from exc
    if output == audit_root:
        raise SemanticAuditError("Audit output cannot be the audits root itself")


def validate_existing_artifacts(
    output: Path,
    expected_payloads: Mapping[str, bytes],
) -> None:
    if not output.is_dir():
        raise SemanticAuditError(f"Audit directory does not exist: {output}")
    actual_names = {
        path.name for path in output.iterdir() if path.is_file()
    }
    if actual_names != set(expected_payloads):
        raise SemanticAuditError(
            "Audit artifact inventory mismatch; "
            f"expected={sorted(expected_payloads)}, actual={sorted(actual_names)}"
        )
    for name, expected in expected_payloads.items():
        actual = (output / name).read_bytes()
        if actual != expected:
            raise SemanticAuditError(
                f"Audit artifact bytes changed: {name}; "
                f"expected_sha256={hashlib.sha256(expected).hexdigest()}, "
                f"actual_sha256={hashlib.sha256(actual).hexdigest()}"
            )


def write_audit(
    package: Path,
    output: Path,
    *,
    validate_only: bool = False,
) -> dict[str, Any]:
    package = package.resolve()
    output = output.resolve()
    _require_audit_destination(package, output)
    payloads, manifest = build_artifact_payloads(package)

    if validate_only:
        validate_existing_artifacts(output, payloads)
        return manifest
    if output.exists():
        raise SemanticAuditError(
            f"Audit output already exists: {output}. "
            "Use --validate-only; this command will not overwrite it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".semantic-audit-60-",
        dir=output.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        for name, payload in payloads.items():
            (temporary_root / name).write_bytes(payload)
        for name, payload in payloads.items():
            if (temporary_root / name).read_bytes() != payload:
                raise SemanticAuditError(f"Temporary write verification failed: {name}")
        temporary_root.replace(output)

    validate_existing_artifacts(output, payloads)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen 60-record AI semantic audit."
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=DEFAULT_PACKAGE,
        help="Prepared ABSA AI tranche package.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Versioned audit directory. Defaults to "
            "<package>/audits/semantic_audit_60_v1."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Recompute every binding and validate existing artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package = args.package.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else (package / DEFAULT_AUDIT_RELATIVE).resolve()
    )
    manifest = write_audit(
        package,
        output,
        validate_only=args.validate_only,
    )
    result = {
        "audit_id": AUDIT_ID,
        "status": (
            "validated" if args.validate_only else "created_and_validated"
        ),
        "output": str(output),
        "sample_records": 60,
        "severity": {
            SEVERITY_OK: 41,
            SEVERITY_MAJOR: 15,
            SEVERITY_MINOR: 4,
        },
        "sample_membership_sha256": EXPECTED_SAMPLE_MEMBERSHIP_SHA256,
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "artifacts": manifest["artifacts"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
