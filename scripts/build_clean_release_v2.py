"""Build an audit-first ABSA curation release from the frozen V1 corpus.

The parent release and raw crawl files are read-only.  Every parent sample is
assigned exactly one status: KEEP, KEEP_CLEANED, QUARANTINE, or EXCLUDE_AUTO.
Heuristic frequency and near-similarity evidence never cause automatic
deletion; uncertain records are routed to human curation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Sequence

from lazada_collector.curation import (
    accent_fold,
    build_clause_frequencies,
    canonical_body_key,
    choose_representative,
    collapse_exact_repeated_clauses,
    connected_components,
    deterministic_stratified_sample,
    extract_clauses,
    find_allpairs,
    internal_ngram_repetition_evidence,
    keyboard_smash_evidence,
    keyboard_token_document_frequencies,
    language_artifact_flags,
    length_ratio,
    normalize_sku,
    normalize_whitespace,
    parse_absolute_review_date,
    punctuation_symbol_key,
    redact_sensitive_spans,
    remove_pure_reward_clauses,
    sha256_text,
    signature_matches,
    stable_rank,
    structural_catalogue_evidence,
    template_evidence,
    terminal_consonant_junk_evidence,
    word_ngrams,
    word_tokens,
)
from lazada_collector.quality import QualityPolicy, evaluate_review


CURATION_SCHEMA_VERSION = "2.1"
EXPECTED_STATUSES = {
    "KEEP",
    "KEEP_CLEANED",
    "QUARANTINE",
    "EXCLUDE_AUTO",
}
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
ANNOTATION_COLUMNS = ["reviewContent", *ASPECT_COLUMNS]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTACT_PROMO_RE = re.compile(
    r"\b(?:liên\s+hệ|lien\s+he|zalo|inbox|sđt|sdt|hotline|"
    r"call|lh)\b",
    flags=re.IGNORECASE,
)
FORBIDDEN_KEYS = {
    "avatar",
    "buyer",
    "buyer_id",
    "buyer_name",
    "cookie",
    "cookies",
    "email",
    "phone",
    "token",
    "username",
}
PRIMARY_REASON_BY_CODE = {
    "INVALID_SCHEMA": "INVALID_SCHEMA",
    "SOURCE_ID_DUPLICATE": "DUPLICATE",
    "PUNCTUATION_EXACT_DUPLICATE": "DUPLICATE",
    "CROSS_TRANSPORT_DUPLICATE": "DUPLICATE",
    "CROSS_TRANSPORT_AMBIGUOUS": "DUPLICATE",
    "HARD_NONREVIEW_PURE": "HARD_NONREVIEW",
    "HARD_NONREVIEW_MATCH": "HARD_NONREVIEW",
    "HARD_NONREVIEW_TAIL_REMOVED": "HARD_NONREVIEW",
    "NONREVIEW_GBOARD": "HARD_NONREVIEW",
    "NONREVIEW_SHARE_CARD": "HARD_NONREVIEW",
    "NONREVIEW_SMS_TELECOM": "HARD_NONREVIEW",
    "SUSPECT_SMS_TELECOM": "NONREVIEW_SUSPECT",
    "SUSPECT_JOB_POST": "NONREVIEW_SUSPECT",
    "SUSPECT_NEWS_SOCIAL": "NONREVIEW_SUSPECT",
    "SUSPECT_DOCUMENT_EDUCATIONAL": "NONREVIEW_SUSPECT",
    "SUSPECT_PRIVATE_NOTIFICATION": "NONREVIEW_SUSPECT",
    "SUSPECT_PLATFORM_GAME": "NONREVIEW_SUSPECT",
    "SUSPECT_SYSTEM_TEXT": "HARD_NONREVIEW",
    "SUSPECT_ADVERTISEMENT": "NONREVIEW_SUSPECT",
    "PROMOTIONAL_CTA_CLUSTER": "NONREVIEW_SUSPECT",
    "SUSPECT_COPIED_CONTENT": "NONREVIEW_SUSPECT",
    "SUSPECT_OBFUSCATED_CONTACT": "PRIVACY",
    "SUSPECT_PERSONAL_ABUSE": "PRIVACY",
    "SUSPECT_GENERIC_PURCHASE_ADVICE": "NONREVIEW_SUSPECT",
    "SUSPECT_NO_EVALUATION": "NONREVIEW_SUSPECT",
    "LONG_TEXT_MANUAL_AUDIT": "NONREVIEW_SUSPECT",
    "NONREVIEW_NEWS_LEGAL": "HARD_NONREVIEW",
    "PII_REDACTED": "PRIVACY",
    "CONTACT_PROMO_WITH_PII": "PRIVACY",
    "TEMPLATE_GLOBAL_CANDIDATE": "PLATFORM_TEMPLATE",
    "TEMPLATE_GLOBAL_HIGH": "PLATFORM_TEMPLATE",
    "TEMPLATE_PRODUCT_HIGH": "PLATFORM_TEMPLATE",
    "STRUCTURAL_CATALOGUE_NO_EXPERIENCE": "PLATFORM_TEMPLATE",
    "REWARD_DISCLOSURE_REMOVED": "REWARD_DISCLOSURE",
    "REWARD_DISCLOSURE_MIXED": "REWARD_DISCLOSURE",
    "REWARD_DISCLOSURE_TYPO": "REWARD_DISCLOSURE",
    "REWARD_DISCLOSURE_SUSPECT": "REWARD_DISCLOSURE",
    "INTERNAL_CLAUSE_REPEAT": "INTERNAL_REPETITION",
    "INTERNAL_NGRAM_REPEAT": "INTERNAL_REPETITION",
    "POST_CLEAN_QUALITY_FAILED": "POST_CLEAN_QUALITY",
    "EMPTY_AFTER_CLEANING": "POST_CLEAN_QUALITY",
    "LANGUAGE_ENGLISH_DOMINANT": "POST_CLEAN_QUALITY",
    "LANGUAGE_TAGALOG_DOMINANT": "POST_CLEAN_QUALITY",
    "LITERAL_ESCAPE_ARTIFACT": "POST_CLEAN_QUALITY",
    "GIBBERISH_CHARACTER_RUNS": "POST_CLEAN_QUALITY",
    "GIBBERISH_CONSONANT_RUNS": "POST_CLEAN_QUALITY",
    "GIBBERISH_KEYBOARD_SMASH": "POST_CLEAN_QUALITY",
    "GIBBERISH_SUFFIX_REMOVED": "POST_CLEAN_QUALITY",
    "QC_AUDIT_BORDERLINE": "NONREVIEW_SUSPECT",
    "QC_AUDIT_MIXED_NOISY": "POST_CLEAN_QUALITY",
    "QC_AUDIT_NONREVIEW": "NONREVIEW_SUSPECT",
    "QC_AUDIT_PURE_TEMPLATE": "PLATFORM_TEMPLATE",
    "NEAR_DUPLICATE_NONREPRESENTATIVE": "NEAR_DUPLICATE",
    "PRODUCT_CAP_OVERFLOW": "PRODUCT_CAP",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_json_line(row))
            handle.write("\n")
            count += 1
    return count


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
    *,
    utf8_bom: bool = True,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if utf8_bom else "utf-8"
    count = 0
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _verify_checksum_file(root: Path, checksum_path: Path) -> int:
    count = 0
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("  ", 1)
            if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
                raise ValueError(
                    f"Invalid checksum row at {checksum_path}:{line_number}"
                )
            relative = parts[1].replace("/", os.sep)
            target = root / relative
            if not target.is_file():
                raise FileNotFoundError(f"Checksum target missing: {target}")
            actual = _sha256_file(target)
            if actual != parts[0]:
                raise ValueError(
                    f"Checksum mismatch for {target}: {actual} != {parts[0]}"
                )
            count += 1
    return count


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().casefold() in FORBIDDEN_KEYS:
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


def _policy_from_config(config: dict[str, Any]) -> QualityPolicy:
    values = config["post_clean_quality"]
    return QualityPolicy(
        min_chars=int(values["min_chars"]),
        min_words=int(values["min_words"]),
        min_unique_word_ratio=float(values["min_unique_word_ratio"]),
        min_meaningful_words=int(values["min_meaningful_words"]),
        min_score=float(values["min_score"]),
        require_vietnamese=bool(values["require_vietnamese"]),
        min_vietnamese_signals=int(values["min_vietnamese_signals"]),
        max_foreign_script_ratio=float(values["max_foreign_script_ratio"]),
        reject_suspect_encoding=bool(values["reject_suspect_encoding"]),
    )


def _quality_dict(result: Any) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "score": result.score,
        "char_count": result.char_count,
        "word_count": result.word_count,
        "unique_word_ratio": result.unique_word_ratio,
        "meaningful_word_count": result.meaningful_word_count,
        "vietnamese_signal_count": result.vietnamese_signal_count,
        "foreign_script_ratio": result.foreign_script_ratio,
        "reasons": list(result.reasons),
    }


def _new_state(
    row: dict[str, Any],
    row_number: int,
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    text = str(row["review_text"])
    return {
        "row": row,
        "parent_canonical_row": row_number,
        "source_audit": source_audit,
        "raw_text": text,
        "curated_text": text,
        "reason_codes": set(),
        "force_exclude": False,
        "force_quarantine": False,
        "transformation_ids": [],
        "transformations": [],
        "duplicate_cluster_id": None,
        "representative_sample_id": str(row["sample_id"]),
        "near_duplicate_cluster_id": None,
        "near_duplicate_representative_sample_id": str(row["sample_id"]),
        "template_evidence": {
            "global_recurrent_clause_count": 0,
            "global_template_coverage": 0.0,
            "product_recurrent_clause_count": 0,
            "product_template_coverage": 0.0,
            "template_family_id": None,
            "global_clause_hashes": [],
            "product_clause_hashes": [],
        },
        "structural_template_evidence": {
            "flagged": False,
            "segment_count": 0,
            "list_delimiter_count": 0,
            "short_segment_count": 0,
            "short_segment_ratio": 0.0,
            "catalogue_opener_count": 0,
            "has_experience_anchor": False,
        },
        "internal_repetition_evidence": {
            "flagged": False,
            "ngram_size": 0,
            "repeated_ngram_count": 0,
            "later_token_count": 0,
            "later_token_coverage": 0.0,
            "repeated_ngram_hashes": [],
        },
        "post_clean_quality": None,
        "product_cap": {
            "enabled": False,
            "maximum_reviews": None,
            "product_candidate_count": None,
            "rank": None,
            "selected": None,
            "selection_seed": None,
        },
    }


def _add_reason(state: dict[str, Any], *reason_codes: str) -> None:
    state["reason_codes"].update(reason_codes)


def _apply_text_operation(
    state: dict[str, Any],
    operation: str,
    output_text: str,
    *,
    spans: Sequence[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
) -> None:
    input_text = state["curated_text"]
    if output_text == input_text:
        return
    sequence = len(state["transformations"]) + 1
    input_hash = sha256_text(input_text)
    output_hash = sha256_text(output_text)
    identity_payload = "\0".join(
        (
            str(state["row"]["sample_id"]),
            str(sequence),
            operation,
            input_hash,
            output_hash,
        )
    )
    transformation_id = "tr-" + sha256_text(identity_payload)[:24]
    cleaned_spans = []
    for span in spans or []:
        cleaned_spans.append(
            {
                key: value
                for key, value in span.items()
                if key not in {"removed_text", "content", "raw"}
            }
        )
    record = {
        "transformation_id": transformation_id,
        "sample_id": state["row"]["sample_id"],
        "sequence": sequence,
        "operation": operation,
        "coordinate_system": "stage_input_python_codepoint_offsets",
        "input_text_sha256": input_hash,
        "output_text_sha256": output_hash,
        "spans": cleaned_spans,
        "evidence": evidence or {},
    }
    state["curated_text"] = output_text
    state["transformation_ids"].append(transformation_id)
    state["transformations"].append(record)


def _prepare_text_states(
    states: list[dict[str, Any]],
    config: dict[str, Any],
    quality_policy: QualityPolicy,
) -> None:
    signatures = config["hard_nonreview_signatures"]
    reward_signatures = config["reward_signatures"]
    privacy = config["privacy"]
    for state in states:
        normalized = normalize_whitespace(state["curated_text"])
        _apply_text_operation(
            state,
            "UNICODE_WHITESPACE_NORMALIZED",
            normalized,
        )

        clauses = extract_clauses(
            state["curated_text"],
            min_tokens=1,
            min_characters=1,
        )
        matched_clauses = []
        matched_reasons: set[str] = set()
        for clause in clauses:
            matches = signature_matches(clause.content, signatures)
            if matches:
                matched_clauses.append(clause)
                matched_reasons.update(matches)
        full_matches = set(signature_matches(state["curated_text"], signatures))
        matched_reasons.update(full_matches)
        if matched_reasons:
            _add_reason(state, *sorted(matched_reasons))
            _add_reason(state, "HARD_NONREVIEW_MATCH")
            state["force_quarantine"] = True

        reward_curated, reward_spans, reward_mixed = (
            remove_pure_reward_clauses(
                state["curated_text"],
                reward_signatures,
                min_tokens=2,
                min_characters=8,
            )
        )
        if reward_spans:
            _apply_text_operation(
                state,
                "REWARD_DISCLAIMER_REMOVED",
                reward_curated,
                spans=reward_spans,
            )
            _add_reason(state, "REWARD_DISCLOSURE_REMOVED")
            state["force_quarantine"] = True
        if reward_mixed:
            _add_reason(state, "REWARD_DISCLOSURE_MIXED")
            state["force_quarantine"] = True
        reward_fallback = re.search(
            config["reward_typo_pattern"],
            state["curated_text"],
            flags=re.IGNORECASE,
        )
        if reward_fallback and not reward_spans and not reward_mixed:
            matched_reward = reward_fallback.group(0).casefold()
            reason = (
                "REWARD_DISCLOSURE_TYPO"
                if matched_reward.endswith("su")
                else "REWARD_DISCLOSURE_MIXED"
            )
            _add_reason(state, reason)
            state["force_quarantine"] = True
        reward_text = normalize_whitespace(
            state["curated_text"]
        ).casefold()
        reward_suspect_groups = config.get(
            "reward_suspect_patterns",
            {},
        )
        explicit_reward = any(
            re.search(pattern, reward_text, flags=re.IGNORECASE)
            for pattern in reward_suspect_groups.get(
                "explicit_reward",
                [],
            )
        )
        generic_disclaimer = any(
            re.search(pattern, reward_text, flags=re.IGNORECASE)
            for pattern in reward_suspect_groups.get(
                "generic_disclaimer",
                [],
            )
        )
        reward_context_guard = any(
            re.search(pattern, reward_text, flags=re.IGNORECASE)
            for pattern in config.get(
                "reward_context_guard_patterns",
                [],
            )
        )
        if explicit_reward or (
            generic_disclaimer and not reward_context_guard
        ):
            _add_reason(state, "REWARD_DISCLOSURE_SUSPECT")
            state["force_quarantine"] = True

        repeated_curated, repeated_spans = collapse_exact_repeated_clauses(
            state["curated_text"],
            min_tokens=int(config["internal_repetition"]["min_tokens"]),
            min_characters=int(
                config["internal_repetition"]["min_characters"]
            ),
        )
        if repeated_spans:
            _apply_text_operation(
                state,
                "INTERNAL_CLAUSE_REPEAT",
                repeated_curated,
                spans=repeated_spans,
            )
            _add_reason(state, "INTERNAL_CLAUSE_REPEAT")

        redacted, privacy_spans = redact_sensitive_spans(
            state["curated_text"],
            phone_pattern=privacy["phone_pattern"],
            email_pattern=privacy["email_pattern"],
            url_pattern=privacy["url_pattern"],
        )
        if privacy_spans:
            input_before_redaction = state["curated_text"]
            _apply_text_operation(
                state,
                "PII_REDACTION",
                redacted,
                spans=privacy_spans,
            )
            _add_reason(state, "PII_REDACTED")
            state["force_quarantine"] = True
            if CONTACT_PROMO_RE.search(input_before_redaction):
                _add_reason(state, "CONTACT_PROMO_WITH_PII")
                state["force_quarantine"] = True

    keyboard_document_frequency = keyboard_token_document_frequencies(
        state["curated_text"]
        for state in states
    )
    for state in states:
        text_before_smash_check = state["curated_text"]
        smash_evidence = keyboard_smash_evidence(
            text_before_smash_check,
            token_document_frequency=keyboard_document_frequency,
        )
        strong_tokens = smash_evidence["strong_tokens"]
        if (
            smash_evidence["flagged"]
            and len(strong_tokens) == 1
            and not smash_evidence["medium_tokens"]
            and not smash_evidence["one_letter_flagged"]
        ):
            strong = strong_tokens[0]
            token_start = int(strong["start"])
            token_end = int(strong["end"])
            residual = text_before_smash_check[:token_start].rstrip()
            token_is_terminal = (
                token_end == len(text_before_smash_check.rstrip())
            )
            has_safe_boundary = (
                token_start > 0
                and text_before_smash_check[token_start - 1].isspace()
                and bool(re.search(r"[.!?…;,:]$", residual))
            )
            residual_quality = evaluate_review(residual, quality_policy)
            if (
                token_is_terminal
                and has_safe_boundary
                and residual_quality.accepted
            ):
                trim_start = len(residual)
                removed = text_before_smash_check[trim_start:]
                _apply_text_operation(
                    state,
                    "TERMINAL_KEYBOARD_SMASH_REMOVED",
                    residual,
                    spans=[
                        {
                            "type": "KEYBOARD_SMASH_SUFFIX",
                            "start": trim_start,
                            "end": len(text_before_smash_check),
                            "replacement": "",
                            "removed_sha256": sha256_text(removed),
                        }
                    ],
                    evidence={
                        "detector_version": "keyboard-smash-v2.1",
                        "token_sha256": strong["token_sha256"],
                        "token_length": strong["length"],
                        "rare_letter_count": strong[
                            "rare_letter_count"
                        ],
                        "consonant_ratio": strong[
                            "consonant_ratio"
                        ],
                        "maximum_consonant_run": strong[
                            "maximum_consonant_run"
                        ],
                        "residual_quality": _quality_dict(
                            residual_quality
                        ),
                    },
                )
                _add_reason(state, "GIBBERISH_SUFFIX_REMOVED")

        terminal_text = state["curated_text"]
        terminal_evidence = terminal_consonant_junk_evidence(
            terminal_text,
            token_document_frequency=keyboard_document_frequency,
        )
        if terminal_evidence is not None:
            token_start = int(terminal_evidence["start"])
            token_end = int(terminal_evidence["token_end"])
            terminal_token = accent_fold(
                terminal_text[token_start:token_end]
            )
            metadata_tokens = set(
                word_tokens(
                    accent_fold(
                        " ".join(
                            (
                                str(state["row"].get("query", "")),
                                str(state["row"].get("sku_info", "")),
                            )
                        )
                    )
                )
            )
            residual = terminal_text[:token_start].rstrip()
            residual_quality = evaluate_review(residual, quality_policy)
            if (
                terminal_token not in metadata_tokens
                and residual_quality.accepted
            ):
                trim_start = len(residual)
                removed = terminal_text[trim_start:]
                _apply_text_operation(
                    state,
                    "TERMINAL_KEYBOARD_SMASH_REMOVED",
                    residual,
                    spans=[
                        {
                            "type": "KEYBOARD_SMASH_SUFFIX",
                            "start": trim_start,
                            "end": len(terminal_text),
                            "replacement": "",
                            "removed_sha256": sha256_text(removed),
                        }
                    ],
                    evidence={
                        "detector_version": "keyboard-smash-v2.1.2",
                        "rule": "TERMINAL_ZERO_VOWEL_TOKEN",
                        "token_sha256": terminal_evidence[
                            "token_sha256"
                        ],
                        "token_length": terminal_evidence[
                            "token_length"
                        ],
                        "root_document_frequency": terminal_evidence[
                            "root_document_frequency"
                        ],
                        "character_diversity": terminal_evidence[
                            "character_diversity"
                        ],
                        "maximum_consonant_run": terminal_evidence[
                            "maximum_consonant_run"
                        ],
                        "residual_quality": _quality_dict(
                            residual_quality
                        ),
                    },
                )
                _add_reason(state, "GIBBERISH_SUFFIX_REMOVED")

    final_keyboard_document_frequency = (
        keyboard_token_document_frequencies(
            state["curated_text"]
            for state in states
        )
    )
    for state in states:
        repetition_config = config["internal_repetition"]
        repetition_evidence = internal_ngram_repetition_evidence(
            state["curated_text"],
            ngram_size=int(repetition_config["ngram_size"]),
            minimum_repeated_ngrams=int(
                repetition_config["minimum_repeated_ngrams"]
            ),
            minimum_later_token_coverage=float(
                repetition_config["minimum_later_token_coverage"]
            ),
        )
        state["internal_repetition_evidence"] = repetition_evidence
        if repetition_evidence["flagged"]:
            _add_reason(state, "INTERNAL_NGRAM_REPEAT")
            state["force_quarantine"] = True

        for reason in language_artifact_flags(
            state["curated_text"],
            token_document_frequency=final_keyboard_document_frequency,
        ):
            _add_reason(state, reason)
            state["force_quarantine"] = True

        result = evaluate_review(state["curated_text"], quality_policy)
        state["post_clean_quality"] = _quality_dict(result)
        if not result.accepted:
            _add_reason(state, "POST_CLEAN_QUALITY_FAILED")
            state["force_quarantine"] = True


def _soft_nonreview_pass(
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Route high-precision but non-definitive non-review matches to humans."""
    pattern_groups = config.get(
        "soft_nonreview_patterns_accent_folded",
        {},
    )
    compiled = {
        reason: [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in patterns
        ]
        for reason, patterns in pattern_groups.items()
    }
    cta_patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in config.get(
            "promotional_cta_patterns_accent_folded",
            [],
        )
    ]
    cta_by_product: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        folded = accent_fold(state["curated_text"])
        for reason, patterns in compiled.items():
            if any(pattern.search(folded) for pattern in patterns):
                _add_reason(state, reason)
                state["force_quarantine"] = True
        if cta_patterns and any(
            pattern.search(folded)
            for pattern in cta_patterns
        ):
            cta_by_product[str(state["row"]["product_id"])].append(
                index
            )
        if len(state["curated_text"]) >= int(
            config["artifact"]["long_text_manual_audit_chars"]
        ):
            _add_reason(state, "LONG_TEXT_MANUAL_AUDIT")
            state["force_quarantine"] = True
    minimum_cluster_size = int(
        config.get("promotional_cta_min_product_records", 2)
    )
    for indices in cta_by_product.values():
        distinct_texts = {
            punctuation_symbol_key(states[index]["curated_text"])
            for index in indices
        }
        if len(distinct_texts) < minimum_cluster_size:
            continue
        for index in indices:
            _add_reason(states[index], "PROMOTIONAL_CTA_CLUSTER")
            states[index]["force_quarantine"] = True


def _structural_template_pass(
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    structural_config = config["structural_template"]
    for state in states:
        template_evidence_row = state["template_evidence"]
        evidence = structural_catalogue_evidence(
            state["curated_text"],
            minimum_segments=int(
                structural_config["minimum_segments"]
            ),
            minimum_list_delimiters=int(
                structural_config["minimum_list_delimiters"]
            ),
            short_segment_min_tokens=int(
                structural_config["short_segment_min_tokens"]
            ),
            short_segment_max_tokens=int(
                structural_config["short_segment_max_tokens"]
            ),
            minimum_short_segment_ratio=float(
                structural_config["minimum_short_segment_ratio"]
            ),
            minimum_catalogue_openers=int(
                structural_config["minimum_catalogue_openers"]
            ),
            global_recurrent_clause_count=int(
                template_evidence_row["global_recurrent_clause_count"]
            ),
            global_template_coverage=float(
                template_evidence_row["global_template_coverage"]
            ),
            product_recurrent_clause_count=int(
                template_evidence_row["product_recurrent_clause_count"]
            ),
            product_template_coverage=float(
                template_evidence_row["product_template_coverage"]
            ),
            weak_global_min_clauses=int(
                structural_config["weak_global_min_clauses"]
            ),
            weak_global_min_coverage=float(
                structural_config["weak_global_min_coverage"]
            ),
            weak_product_min_clauses=int(
                structural_config["weak_product_min_clauses"]
            ),
            weak_product_min_coverage=float(
                structural_config["weak_product_min_coverage"]
            ),
            unique_min_marketing_ratio=float(
                structural_config["unique_min_marketing_ratio"]
            ),
            low_density_min_marketing_ratio=float(
                structural_config[
                    "low_density_min_marketing_ratio"
                ]
            ),
            modular_min_segments=int(
                structural_config["modular_min_segments"]
            ),
            modular_trailing_min_short_ratio=float(
                structural_config[
                    "modular_trailing_min_short_ratio"
                ]
            ),
            modular_cleaned_min_short_ratio=float(
                structural_config[
                    "modular_cleaned_min_short_ratio"
                ]
            ),
            modular_min_capitalized_ratio=float(
                structural_config[
                    "modular_min_capitalized_ratio"
                ]
            ),
            expanded_spec_min_clauses=int(
                structural_config["expanded_spec_min_clauses"]
            ),
            glued_min_segments=int(
                structural_config["glued_min_segments"]
            ),
            was_cleaned=bool(state["transformation_ids"]),
        )
        state["structural_template_evidence"] = evidence
        if evidence["flagged"]:
            _add_reason(
                state,
                "STRUCTURAL_CATALOGUE_NO_EXPERIENCE",
            )
            state["force_quarantine"] = True


def _audit_escalation_pass(
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Route content-audit findings to humans without deleting records."""
    configured = config.get("qc_audit_escalations", {})
    state_by_id = {
        str(state["row"]["sample_id"]): state
        for state in states
    }
    for reason_code, sample_ids in configured.items():
        unknown = sorted(set(sample_ids) - set(state_by_id))
        if unknown:
            raise ValueError(
                f"QC audit escalation references unknown samples: {unknown}"
            )
        for sample_id in sorted(set(sample_ids)):
            state = state_by_id[sample_id]
            _add_reason(state, str(reason_code))
            state["force_quarantine"] = True


def _sku_value_set(value: str) -> frozenset[str]:
    return frozenset(part for part in normalize_sku(value).split("|") if part)


def _sku_compatible(left: str, right: str) -> bool:
    left_values = _sku_value_set(left)
    right_values = _sku_value_set(right)
    if not left_values and not right_values:
        return True
    if not left_values or not right_values:
        return False
    return (
        left_values == right_values
        or left_values.issubset(right_values)
        or right_values.issubset(left_values)
    )


def _cross_transport_duplicates(
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duplicate_config = config["duplicate"]
    threshold = float(
        duplicate_config["cross_transport_word_5gram_jaccard"]
    )
    minimum_length_ratio = float(
        duplicate_config["cross_transport_min_length_ratio"]
    )
    required_margin = float(
        duplicate_config["cross_transport_ambiguous_margin"]
    )
    aliases = config["bilingual_heading_aliases"]
    blocks: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        if state["force_exclude"]:
            continue
        row = state["row"]
        review_date = parse_absolute_review_date(
            str(row.get("review_time") or "")
        )
        if not review_date:
            continue
        blocks[
            (
                str(row["product_id"]),
                int(row["rating"]),
                review_date,
            )
        ].append(index)

    all_evidence: list[dict[str, Any]] = []
    confirmed_aliases: list[dict[str, Any]] = []
    for block_key in sorted(blocks):
        indices = blocks[block_key]
        api = [
            index
            for index in indices
            if states[index]["row"]["collection_transport"]
            in {"requests", "requests_cookie"}
        ]
        dom = [
            index
            for index in indices
            if states[index]["row"]["collection_transport"] == "selenium_dom"
        ]
        if not api or not dom:
            continue

        canonical = {
            index: canonical_body_key(
                states[index]["curated_text"],
                aliases,
            )
            for index in indices
        }
        grams = {
            index: word_ngrams(canonical[index], 5)
            for index in indices
        }
        candidates: list[dict[str, Any]] = []
        for api_index in api:
            for dom_index in dom:
                if not _sku_compatible(
                    str(states[api_index]["row"].get("sku_info") or ""),
                    str(states[dom_index]["row"].get("sku_info") or ""),
                ):
                    continue
                left_grams = grams[api_index]
                right_grams = grams[dom_index]
                union = len(left_grams | right_grams)
                if not union:
                    continue
                intersection = len(left_grams & right_grams)
                score = intersection / union
                text_length_ratio = length_ratio(
                    canonical[api_index],
                    canonical[dom_index],
                )
                if (
                    score < threshold
                    or text_length_ratio < minimum_length_ratio
                ):
                    continue
                candidates.append(
                    {
                        "api_index": api_index,
                        "dom_index": dom_index,
                        "intersection": intersection,
                        "union": union,
                        "jaccard": score,
                        "length_ratio": text_length_ratio,
                    }
                )
        if not candidates:
            continue

        by_api: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_dom: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            by_api[candidate["api_index"]].append(candidate)
            by_dom[candidate["dom_index"]].append(candidate)

        def ordered(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                values,
                key=lambda item: (
                    -item["jaccard"],
                    -item["length_ratio"],
                    states[item["api_index"]]["row"]["sample_id"],
                    states[item["dom_index"]]["row"]["sample_id"],
                ),
            )

        accepted_pairs: set[tuple[int, int]] = set()
        ambiguous_indices: set[int] = set()
        for candidate in sorted(
            candidates,
            key=lambda item: (
                states[item["api_index"]]["row"]["sample_id"],
                states[item["dom_index"]]["row"]["sample_id"],
            ),
        ):
            api_ranked = ordered(by_api[candidate["api_index"]])
            dom_ranked = ordered(by_dom[candidate["dom_index"]])
            if api_ranked[0] is not candidate or dom_ranked[0] is not candidate:
                continue
            api_gap = (
                candidate["jaccard"] - api_ranked[1]["jaccard"]
                if len(api_ranked) > 1
                else 1.0
            )
            dom_gap = (
                candidate["jaccard"] - dom_ranked[1]["jaccard"]
                if len(dom_ranked) > 1
                else 1.0
            )
            candidate["api_best_margin"] = round(api_gap, 12)
            candidate["dom_best_margin"] = round(dom_gap, 12)
            if api_gap >= required_margin and dom_gap >= required_margin:
                accepted_pairs.add(
                    (candidate["api_index"], candidate["dom_index"])
                )
            else:
                ambiguous_indices.add(candidate["api_index"])
                ambiguous_indices.add(candidate["dom_index"])

        for candidate in candidates:
            api_index = candidate["api_index"]
            dom_index = candidate["dom_index"]
            accepted = (api_index, dom_index) in accepted_pairs
            evidence = {
                "cluster_id": None,
                "representative_sample_id": states[api_index]["row"]["sample_id"],
                "alias_sample_id": states[dom_index]["row"]["sample_id"],
                "match_type": "CROSS_TRANSPORT",
                "auto_merge": accepted,
                "product_compatible": True,
                "rating_compatible": True,
                "date_compatible": True,
                "sku_compatible": True,
                "word_5gram_intersection": candidate["intersection"],
                "word_5gram_union": candidate["union"],
                "word_5gram_jaccard": round(candidate["jaccard"], 12),
                "length_ratio": round(candidate["length_ratio"], 12),
                "api_best_margin": candidate.get("api_best_margin"),
                "dom_best_margin": candidate.get("dom_best_margin"),
                "required_ambiguity_margin": required_margin,
                "rule_version": config["rule_version"],
            }
            if accepted:
                pair_key = "\0".join(
                    sorted(
                        (
                            str(states[api_index]["row"]["sample_id"]),
                            str(states[dom_index]["row"]["sample_id"]),
                        )
                    )
                )
                cluster_id = "dup-xtr-" + sha256_text(pair_key)[:20]
                evidence["cluster_id"] = cluster_id
                states[api_index]["duplicate_cluster_id"] = cluster_id
                states[dom_index]["duplicate_cluster_id"] = cluster_id
                states[dom_index]["representative_sample_id"] = str(
                    states[api_index]["row"]["sample_id"]
                )
                _add_reason(states[dom_index], "CROSS_TRANSPORT_DUPLICATE")
                states[dom_index]["force_exclude"] = True
                confirmed_aliases.append(dict(evidence))
            all_evidence.append(evidence)

        accepted_members = {
            index for pair in accepted_pairs for index in pair
        }
        candidate_members = {
            candidate["api_index"]
            for candidate in candidates
        } | {
            candidate["dom_index"]
            for candidate in candidates
        }
        ambiguous_indices.update(candidate_members - accepted_members)
        for index in sorted(ambiguous_indices - accepted_members):
            _add_reason(states[index], "CROSS_TRANSPORT_AMBIGUOUS")
            states[index]["force_quarantine"] = True

    order = lambda row: (
        row["representative_sample_id"],
        row["alias_sample_id"],
    )
    return (
        sorted(confirmed_aliases, key=order),
        sorted(all_evidence, key=order),
    )


def _confirmed_key_duplicates(
    states: list[dict[str, Any]],
    *,
    key_name: str,
    key_function: Any,
    match_type: str,
    reason_code: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    ranking_rows = [
        {
            **state["row"],
            "curated_review_text": state["curated_text"],
        }
        for state in states
    ]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        if state["force_exclude"]:
            continue
        key = key_function(state)
        if key:
            groups[str(key)].append(index)
    aliases = []
    for key in sorted(groups):
        indices = groups[key]
        if len(indices) < 2:
            continue
        representative = choose_representative(
            ranking_rows,
            indices,
            prefer_api_identity=match_type == "SOURCE_ID",
        )
        member_ids = sorted(
            str(states[index]["row"]["sample_id"])
            for index in indices
        )
        cluster_id = (
            f"dup-{key_name}-"
            + sha256_text("\0".join(member_ids))[:20]
        )
        for index in indices:
            states[index]["duplicate_cluster_id"] = cluster_id
            states[index]["representative_sample_id"] = str(
                states[representative]["row"]["sample_id"]
            )
            if index == representative:
                continue
            _add_reason(states[index], reason_code)
            states[index]["force_exclude"] = True
            aliases.append(
                {
                    "cluster_id": cluster_id,
                    "representative_sample_id": states[representative]["row"][
                        "sample_id"
                    ],
                    "alias_sample_id": states[index]["row"]["sample_id"],
                    "match_type": match_type,
                    "auto_merge": True,
                    "key_sha256": sha256_text(key),
                    "rule_version": config["rule_version"],
                }
            )
    return aliases


def _template_pass(
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    active_indices = [
        index for index, state in enumerate(states)
        if not state["force_exclude"]
    ]
    texts = [states[index]["curated_text"] for index in active_indices]
    products = [
        str(states[index]["row"]["product_id"])
        for index in active_indices
    ]
    clause_config = config["clause"]
    parsed, review_df, product_sets, product_review_df = (
        build_clause_frequencies(
            texts,
            products,
            min_tokens=int(clause_config["min_tokens"]),
            min_characters=int(clause_config["min_characters"]),
        )
    )
    template_config = config["template"]
    template_candidate_indices: list[int] = []
    for local_index, state_index in enumerate(active_indices):
        state = states[state_index]
        evidence = template_evidence(
            state["curated_text"],
            parsed[local_index],
            product_id=str(state["row"]["product_id"]),
            review_df=review_df,
            product_sets=product_sets,
            product_review_df=product_review_df,
            global_review_df=int(template_config["global_review_df"]),
            global_product_df=int(template_config["global_product_df"]),
            product_df=int(template_config["product_review_df"]),
        )
        # The helper's exact-set family hash is useful as record evidence but
        # is too fragmented for leakage grouping.  A connected family based
        # on at least two shared recurrent clauses is assigned below.
        evidence["template_family_id"] = None
        state["template_evidence"] = evidence
        clause_count = sum(
            1 for clause in parsed[local_index] if clause.key
        )
        global_count = evidence["global_recurrent_clause_count"]
        global_coverage = evidence["global_template_coverage"]
        product_count = evidence["product_recurrent_clause_count"]
        product_coverage = evidence["product_template_coverage"]
        list_like = clause_count >= int(template_config["list_min_segments"])
        flagged = False
        quarantine = False
        if (
            list_like
            and global_count
            >= int(template_config["global_min_recurrent_clauses"])
            and global_coverage
            >= float(template_config["global_high_coverage"])
        ):
            _add_reason(state, "TEMPLATE_GLOBAL_HIGH")
            flagged = True
            quarantine = True
        elif (
            list_like
            and global_count
            >= int(
                template_config[
                    "global_candidate_min_recurrent_clauses"
                ]
            )
            and global_coverage
            >= float(template_config["global_candidate_coverage"])
        ):
            _add_reason(state, "TEMPLATE_GLOBAL_CANDIDATE")
            flagged = True
            quarantine = True
        if (
            list_like
            and product_count
            >= int(template_config["product_min_recurrent_clauses"])
            and product_coverage
            >= float(template_config["product_high_coverage"])
        ):
            _add_reason(state, "TEMPLATE_PRODUCT_HIGH")
            flagged = True
            quarantine = True
        if flagged:
            template_candidate_indices.append(state_index)
        if quarantine:
            state["force_quarantine"] = True

    minimum_shared = int(template_config["family_min_shared_clauses"])
    if minimum_shared != 2:
        raise ValueError(
            "Template family construction currently requires exactly two "
            "shared recurrent clauses"
        )
    maximum_signature_members = int(
        template_config["family_max_signature_members"]
    )
    signature_members: dict[tuple[str, str], list[int]] = defaultdict(list)
    hashes_by_index: dict[int, tuple[str, ...]] = {}
    for state_index in template_candidate_indices:
        evidence = states[state_index]["template_evidence"]
        hashes = tuple(
            sorted(
                set(evidence["global_clause_hashes"])
                | set(evidence["product_clause_hashes"])
            )
        )
        hashes_by_index[state_index] = hashes
        for signature in combinations(hashes, 2):
            signature_members[signature].append(state_index)
    family_edges: list[tuple[int, int]] = []
    for signature in sorted(signature_members):
        members = sorted(set(signature_members[signature]))
        if not 2 <= len(members) <= maximum_signature_members:
            continue
        family_edges.extend(
            (members[0], member)
            for member in members[1:]
        )
    components = connected_components(len(states), family_edges)
    families = []
    for component in components:
        members = sorted(
            str(states[index]["row"]["sample_id"])
            for index in component
        )
        family_id = "tplcc-" + sha256_text("\0".join(members))[:20]
        clause_support: Counter[str] = Counter()
        for index in component:
            clause_support.update(set(hashes_by_index[index]))
            states[index]["template_evidence"][
                "template_family_id"
            ] = family_id
        shared_hashes = sorted(
            clause_hash
            for clause_hash, support in clause_support.items()
            if support >= 2
        )
        families.append(
            {
                "template_family_id": family_id,
                "shared_recurrent_clause_hashes": shared_hashes,
                "member_sample_ids": members,
                "member_count": len(members),
                "decision": "HUMAN_CALIBRATION_COMPONENT",
                "thresholds": template_config,
                "rule_version": config["rule_version"],
            }
        )
    return sorted(families, key=lambda row: row["template_family_id"])


def _near_duplicate_pass(
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    active_indices = [
        index for index, state in enumerate(states)
        if not state["force_exclude"]
    ]
    feature_sets = [
        word_ngrams(states[index]["curated_text"], 5)
        for index in active_indices
    ]
    stable_ids = [
        str(states[index]["row"]["sample_id"])
        for index in active_indices
    ]
    duplicate_config = config["duplicate"]
    global_pairs = find_allpairs(
        feature_sets,
        threshold=float(
            duplicate_config["global_clean_word_5gram_jaccard"]
        ),
        stable_ids=stable_ids,
    )
    pair_evidence: dict[tuple[int, int], dict[str, Any]] = {}
    for pair in global_pairs:
        left = active_indices[pair.left]
        right = active_indices[pair.right]
        pair_length_ratio = length_ratio(
            states[left]["curated_text"],
            states[right]["curated_text"],
        )
        if pair_length_ratio < float(
            duplicate_config["global_clean_min_length_ratio"]
        ):
            continue
        pair_evidence[(min(left, right), max(left, right))] = {
            "scopes": {"GLOBAL"},
            "intersection_5grams": pair.intersection,
            "union_5grams": pair.union,
            "jaccard": pair.score,
            "length_ratio": pair_length_ratio,
        }

    by_product: dict[str, list[int]] = defaultdict(list)
    for index in active_indices:
        by_product[str(states[index]["row"]["product_id"])].append(index)
    for product_id in sorted(by_product):
        indices = by_product[product_id]
        if len(indices) < 2:
            continue
        local_features = [
            word_ngrams(states[index]["curated_text"], 5)
            for index in indices
        ]
        local_ids = [
            str(states[index]["row"]["sample_id"])
            for index in indices
        ]
        local_pairs = find_allpairs(
            local_features,
            threshold=float(
                duplicate_config["product_clean_word_5gram_jaccard"]
            ),
            stable_ids=local_ids,
        )
        for pair in local_pairs:
            left = indices[pair.left]
            right = indices[pair.right]
            pair_length_ratio = length_ratio(
                states[left]["curated_text"],
                states[right]["curated_text"],
            )
            if pair_length_ratio < float(
                duplicate_config["product_clean_min_length_ratio"]
            ):
                continue
            key = (min(left, right), max(left, right))
            if key in pair_evidence:
                pair_evidence[key]["scopes"].add("PRODUCT")
            else:
                pair_evidence[key] = {
                    "scopes": {"PRODUCT"},
                    "intersection_5grams": pair.intersection,
                    "union_5grams": pair.union,
                    "jaccard": pair.score,
                    "length_ratio": pair_length_ratio,
                }

    components = connected_components(
        len(states),
        pair_evidence.keys(),
    )
    component_for_index: dict[int, tuple[str, int]] = {}
    rows_for_ranking = [
        {
            **state["row"],
            "curated_review_text": state["curated_text"],
        }
        for state in states
    ]
    for component in components:
        representative = choose_representative(
            rows_for_ranking,
            component,
            prefer_api_identity=False,
        )
        member_ids = sorted(
            str(states[index]["row"]["sample_id"])
            for index in component
        )
        cluster_id = "near-" + sha256_text("\0".join(member_ids))[:20]
        for index in component:
            component_for_index[index] = (cluster_id, representative)
            states[index]["near_duplicate_cluster_id"] = cluster_id
            states[index]["near_duplicate_representative_sample_id"] = str(
                states[representative]["row"]["sample_id"]
            )
            if index != representative:
                _add_reason(
                    states[index],
                    "NEAR_DUPLICATE_NONREPRESENTATIVE",
                )
                states[index]["force_quarantine"] = True

    output = []
    for (left, right), evidence in sorted(
        pair_evidence.items(),
        key=lambda item: (
            states[item[0][0]]["row"]["sample_id"],
            states[item[0][1]]["row"]["sample_id"],
        ),
    ):
        cluster_id, representative = component_for_index[left]
        output.append(
            {
                "cluster_id": cluster_id,
                "representative_sample_id": states[representative]["row"][
                    "sample_id"
                ],
                "left_sample_id": states[left]["row"]["sample_id"],
                "right_sample_id": states[right]["row"]["sample_id"],
                "scopes": sorted(evidence["scopes"]),
                "intersection_5grams": evidence["intersection_5grams"],
                "union_5grams": evidence["union_5grams"],
                "jaccard": round(evidence["jaccard"], 12),
                "length_ratio": round(evidence["length_ratio"], 12),
                "decision": "QUARANTINE_NONREPRESENTATIVE",
                "rule_version": config["rule_version"],
            }
        )
    return output


def _apply_product_cap(
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    cap = config["product_cap"]
    enabled = bool(cap["enabled"])
    maximum = int(cap["maximum_reviews"])
    seed = str(cap["selection_seed"])
    candidates_by_product: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        if state["force_exclude"]:
            continue
        candidates_by_product[str(state["row"]["product_id"])].append(index)
    for product_id in sorted(candidates_by_product):
        indices = sorted(
            candidates_by_product[product_id],
            key=lambda index: (
                stable_rank(
                    seed,
                    str(states[index]["row"]["sample_id"]),
                ),
                str(states[index]["row"]["sample_id"]),
            ),
        )
        for rank, index in enumerate(indices, 1):
            selected = not enabled or rank <= maximum
            states[index]["product_cap"] = {
                "enabled": enabled,
                "maximum_reviews": maximum,
                "product_candidate_count": len(indices),
                "rank": rank,
                "selected": selected,
                "selection_seed": seed,
            }
            if not selected and not states[index]["force_quarantine"]:
                _add_reason(states[index], "PRODUCT_CAP_OVERFLOW")
                states[index]["force_quarantine"] = True


def _primary_reason(
    state: dict[str, Any],
    config: dict[str, Any],
) -> str:
    broad_reasons = {
        PRIMARY_REASON_BY_CODE[code]
        for code in state["reason_codes"]
        if code in PRIMARY_REASON_BY_CODE
    }
    for reason in config["precedence"]:
        if reason in broad_reasons:
            return reason
    return "KEEP"


def _exclusion_trigger(state: dict[str, Any]) -> str | None:
    if not state["force_exclude"]:
        return None
    reasons = state["reason_codes"]
    if {
        "SOURCE_ID_DUPLICATE",
        "PUNCTUATION_EXACT_DUPLICATE",
        "CROSS_TRANSPORT_DUPLICATE",
    } & reasons:
        return "CONFIRMED_DUPLICATE"
    if "HARD_NONREVIEW_PURE" in reasons:
        return "HARD_NONREVIEW_NO_SUBSTANTIVE_RESIDUAL"
    if "POST_CLEAN_QUALITY_FAILED" in reasons:
        operations = [
            transformation["operation"]
            for transformation in state["transformations"]
        ]
        if operations:
            return (
                "POST_CLEAN_QUALITY_FAILED_AFTER_"
                + "+".join(operations)
            )
        return "POST_CLEAN_QUALITY_FAILED"
    return "FAIL_CLOSED_UNCLASSIFIED"


def _finalize_states(
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    for state in states:
        if state["force_exclude"]:
            status = "EXCLUDE_AUTO"
        elif state["force_quarantine"]:
            status = "QUARANTINE"
        elif state["transformation_ids"]:
            status = "KEEP_CLEANED"
        else:
            status = "KEEP"
        if status not in EXPECTED_STATUSES:
            raise AssertionError(f"Unexpected curation status: {status}")
        state["status"] = status
        state["primary_reason"] = _primary_reason(state, config)
        state["exclusion_trigger"] = _exclusion_trigger(state)
        state["decision_source"] = (
            "MANUAL_PENDING"
            if status == "QUARANTINE"
            else "AUTOMATIC_RULE"
        )
        state["annotation_eligible"] = status in {
            "KEEP",
            "KEEP_CLEANED",
        }


def _curation_row(
    state: dict[str, Any],
    parent_release_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    row = state["row"]
    raw_hash = sha256_text(state["raw_text"])
    curated_hash = sha256_text(state["curated_text"])
    return {
        "curation_schema_version": CURATION_SCHEMA_VERSION,
        "parent_release_id": parent_release_id,
        "parent_canonical_row": state["parent_canonical_row"],
        "sample_id": row["sample_id"],
        "product_id": row["product_id"],
        "review_id": row["review_id"],
        "rule_version": config["rule_version"],
        "status": state["status"],
        "primary_reason": state["primary_reason"],
        "exclusion_trigger": state["exclusion_trigger"],
        "reason_codes": sorted(state["reason_codes"]),
        "decision_source": state["decision_source"],
        "annotation_eligible": state["annotation_eligible"],
        "raw_text_sha256": raw_hash,
        "curated_review_text": state["curated_text"],
        "curated_text_sha256": curated_hash,
        "transformation_ids": state["transformation_ids"],
        "duplicate_cluster_id": state["duplicate_cluster_id"],
        "representative_sample_id": state["representative_sample_id"],
        "near_duplicate_cluster_id": state["near_duplicate_cluster_id"],
        "near_duplicate_representative_sample_id": state[
            "near_duplicate_representative_sample_id"
        ],
        "template_family_id": state["template_evidence"][
            "template_family_id"
        ],
        "template_evidence": state["template_evidence"],
        "structural_template_evidence": state[
            "structural_template_evidence"
        ],
        "internal_repetition_evidence": state[
            "internal_repetition_evidence"
        ],
        "post_clean_quality": state["post_clean_quality"],
        "product_cap": state["product_cap"],
        "source_relative_path": state["source_audit"].get(
            "source_relative_path"
        ),
        "source_line": state["source_audit"].get("source_line"),
    }


def _full_partition_row(
    state: dict[str, Any],
    curation_row: dict[str, Any],
) -> dict[str, Any]:
    value = dict(state["row"])
    value["curated_review_text"] = state["curated_text"]
    value["curation"] = {
        key: curation_row[key]
        for key in (
            "curation_schema_version",
            "parent_release_id",
            "parent_canonical_row",
            "rule_version",
            "status",
            "primary_reason",
            "exclusion_trigger",
            "reason_codes",
            "decision_source",
            "annotation_eligible",
            "raw_text_sha256",
            "curated_text_sha256",
            "transformation_ids",
            "duplicate_cluster_id",
            "representative_sample_id",
            "near_duplicate_cluster_id",
            "near_duplicate_representative_sample_id",
            "template_family_id",
            "template_evidence",
            "structural_template_evidence",
            "internal_repetition_evidence",
            "post_clean_quality",
            "product_cap",
            "source_relative_path",
            "source_line",
        )
    }
    return value


def _artifact(path: Path, root: Path, records: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _distribution(
    states: list[dict[str, Any]],
    field: str,
    *,
    statuses: set[str] | None = None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for state in states:
        if statuses is not None and state["status"] not in statuses:
            continue
        counts[str(state["row"].get(field) or "<blank>")] += 1
    return dict(sorted(counts.items()))


def _write_annotation_package(
    root: Path,
    states: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, int]:
    annotation_root = root / "annotation"
    eligible_indices = [
        index for index, state in enumerate(states)
        if state["annotation_eligible"]
    ]
    eligible_rows = [
        {
            "sample_id": states[index]["row"]["sample_id"],
            "category": states[index]["row"].get("category") or "<blank>",
            "rating": states[index]["row"]["rating"],
            "collection_transport": states[index]["row"][
                "collection_transport"
            ],
        }
        for index in eligible_indices
    ]
    annotation_config = config["annotation"]
    pilot_local = deterministic_stratified_sample(
        eligible_rows,
        size=int(annotation_config["pilot_size"]),
        seed=str(config["product_cap"]["selection_seed"]) + "-pilot",
        keys=("category", "rating"),
    )
    pilot_indices = [eligible_indices[index] for index in pilot_local]
    pilot_set = set(pilot_indices)
    pilot_path = annotation_root / "pilot_candidate.csv"
    pilot_count = _write_csv(
        pilot_path,
        ANNOTATION_COLUMNS,
        (
            {
                "reviewContent": states[index]["curated_text"],
                **{column: "" for column in ASPECT_COLUMNS},
            }
            for index in pilot_indices
        ),
    )

    index_rows = []
    pilot_position = {
        index: position
        for position, index in enumerate(pilot_indices, 2)
    }
    for index in eligible_indices:
        state = states[index]
        is_pilot = index in pilot_set
        index_rows.append(
            {
                "sample_id": state["row"]["sample_id"],
                "curated_text_sha256": sha256_text(state["curated_text"]),
                "package_role": (
                    "PILOT_CANDIDATE"
                    if is_pilot
                    else "PENDING_MAIN_AFTER_PILOT_GATE"
                ),
                "batch_file": (
                    "annotation/pilot_candidate.csv"
                    if is_pilot
                    else ""
                ),
                "batch_data_row": (
                    pilot_position[index] if is_pilot else ""
                ),
                "parent_canonical_row": state["parent_canonical_row"],
                "curation_status": state["status"],
                "rule_version": config["rule_version"],
                "guideline_version": "2.0",
            }
        )
    index_count = _write_csv(
        annotation_root / "index.csv",
        [
            "sample_id",
            "curated_text_sha256",
            "package_role",
            "batch_file",
            "batch_data_row",
            "parent_canonical_row",
            "curation_status",
            "rule_version",
            "guideline_version",
        ],
        index_rows,
    )

    clean_candidates = [
        index for index, state in enumerate(states)
        if state["annotation_eligible"]
    ]
    audit_rows = [
        {
            "sample_id": states[index]["row"]["sample_id"],
            "category": states[index]["row"].get("category") or "<blank>",
            "rating": states[index]["row"]["rating"],
            "collection_transport": states[index]["row"][
                "collection_transport"
            ],
        }
        for index in clean_candidates
    ]
    audit_local = deterministic_stratified_sample(
        audit_rows,
        size=int(annotation_config["unflagged_manual_audit_size"]),
        seed=str(config["product_cap"]["selection_seed"]) + "-clean-audit",
        keys=("category", "rating", "collection_transport"),
    )
    audit_indices = [clean_candidates[index] for index in audit_local]
    clean_audit_count = _write_csv(
        annotation_root / "curation_audit_1000.csv",
        [
            "audit_id",
            "reviewContent",
            "curation_decision",
            "reviewer_id",
            "notes",
        ],
        (
            {
                "audit_id": (
                    "audit-"
                    + sha256_text(
                        str(config["product_cap"]["selection_seed"])
                        + "\0"
                        + str(states[index]["row"]["sample_id"])
                    )[:16]
                ),
                "reviewContent": states[index]["curated_text"],
                "curation_decision": "",
                "reviewer_id": "",
                "notes": "",
            }
            for index in audit_indices
        ),
    )
    _write_csv(
        annotation_root / "curation_audit_index.csv",
        [
            "audit_id",
            "sample_id",
            "category",
            "rating",
            "collection_transport",
            "curated_text_sha256",
        ],
        (
            {
                "audit_id": (
                    "audit-"
                    + sha256_text(
                        str(config["product_cap"]["selection_seed"])
                        + "\0"
                        + str(states[index]["row"]["sample_id"])
                    )[:16]
                ),
                "sample_id": states[index]["row"]["sample_id"],
                "category": states[index]["row"].get("category") or "",
                "rating": states[index]["row"]["rating"],
                "collection_transport": states[index]["row"][
                    "collection_transport"
                ],
                "curated_text_sha256": sha256_text(
                    states[index]["curated_text"]
                ),
            }
            for index in audit_indices
        ),
    )

    template_strata: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        if state["status"] == "EXCLUDE_AUTO":
            continue
        reasons = state["reason_codes"]
        has_global_candidate = "TEMPLATE_GLOBAL_CANDIDATE" in reasons
        has_global_high = "TEMPLATE_GLOBAL_HIGH" in reasons
        has_product_high = "TEMPLATE_PRODUCT_HIGH" in reasons
        has_structural = (
            "STRUCTURAL_CATALOGUE_NO_EXPERIENCE" in reasons
        )
        has_global = has_global_candidate or has_global_high
        if has_global and has_product_high:
            stratum = "GLOBAL_AND_PRODUCT"
        elif has_global_high:
            stratum = "GLOBAL_HIGH_ONLY"
        elif has_global_candidate:
            stratum = "GLOBAL_CANDIDATE_ONLY"
        elif has_product_high:
            stratum = "PRODUCT_HIGH_ONLY"
        elif has_structural:
            stratum = "STRUCTURAL_ONLY"
        else:
            continue
        template_strata[stratum].append(index)
    calibration_indices: list[tuple[str, int]] = []
    calibration_size = int(
        annotation_config["template_calibration_per_stratum"]
    )
    for stratum in sorted(template_strata):
        ordered = sorted(
            template_strata[stratum],
            key=lambda index: (
                stable_rank(
                    str(config["product_cap"]["selection_seed"])
                    + "-template-calibration-"
                    + stratum,
                    str(states[index]["row"]["sample_id"]),
                ),
                str(states[index]["row"]["sample_id"]),
            ),
        )
        calibration_indices.extend(
            (stratum, index)
            for index in ordered[:calibration_size]
        )
    calibration_rows = []
    calibration_index_rows = []
    for stratum, index in calibration_indices:
        sample_id = str(states[index]["row"]["sample_id"])
        calibration_id = (
            "tplcal-"
            + sha256_text(
                str(config["product_cap"]["selection_seed"])
                + "\0"
                + stratum
                + "\0"
                + sample_id
            )[:16]
        )
        calibration_rows.append(
            {
                "calibration_id": calibration_id,
                "reviewContent": states[index]["curated_text"],
                "composition_decision": "",
                "nonreview_decision": "",
                "reviewer_id": "",
                "notes": "",
            }
        )
        evidence = states[index]["template_evidence"]
        calibration_index_rows.append(
            {
                "calibration_id": calibration_id,
                "sample_id": sample_id,
                "stratum": stratum,
                "curation_status": states[index]["status"],
                "curated_text_sha256": sha256_text(
                    states[index]["curated_text"]
                ),
                "global_recurrent_clause_count": evidence[
                    "global_recurrent_clause_count"
                ],
                "global_template_coverage": evidence[
                    "global_template_coverage"
                ],
                "product_recurrent_clause_count": evidence[
                    "product_recurrent_clause_count"
                ],
                "product_template_coverage": evidence[
                    "product_template_coverage"
                ],
                "template_family_id": evidence["template_family_id"] or "",
            }
        )
    template_calibration_count = _write_csv(
        annotation_root / "template_calibration.csv",
        [
            "calibration_id",
            "reviewContent",
            "composition_decision",
            "nonreview_decision",
            "reviewer_id",
            "notes",
        ],
        calibration_rows,
    )
    _write_csv(
        annotation_root / "template_calibration_index.csv",
        [
            "calibration_id",
            "sample_id",
            "stratum",
            "curation_status",
            "curated_text_sha256",
            "global_recurrent_clause_count",
            "global_template_coverage",
            "product_recurrent_clause_count",
            "product_template_coverage",
            "template_family_id",
        ],
        calibration_index_rows,
    )

    flagged_indices = [
        index for index, state in enumerate(states)
        if state["status"] in {"QUARANTINE", "EXCLUDE_AUTO"}
    ]
    flagged_review_count = _write_csv(
        annotation_root / "curation_review_queue.csv",
        [
            "sample_id",
            "proposed_status",
            "primary_reason",
            "reason_codes",
            "review_text_original",
            "review_text_curated",
            "human_decision",
            "human_primary_reason",
            "reviewer_id",
            "notes",
        ],
        (
            {
                "sample_id": states[index]["row"]["sample_id"],
                "proposed_status": states[index]["status"],
                "primary_reason": states[index]["primary_reason"],
                "reason_codes": "|".join(
                    sorted(states[index]["reason_codes"])
                ),
                "review_text_original": states[index]["raw_text"],
                "review_text_curated": states[index]["curated_text"],
                "human_decision": "",
                "human_primary_reason": "",
                "reviewer_id": "",
                "notes": "",
            }
            for index in flagged_indices
        ),
    )

    schema = {
        "schema_version": "2.0",
        "review_column": "reviewContent",
        "aspect_columns": ASPECT_COLUMNS,
        "allowed_labels": annotation_config["allowed_labels"],
        "unlabeled_representation": "blank",
        "blank_semantics": "UNLABELED_ONLY; blank is never label 2",
        "legacy_tensor_polarity_order": [
            "negative",
            "positive",
            "neutral",
        ],
        "pilot_file_status": (
            "CANDIDATE_ONLY; annotate only after curation QC is approved"
        ),
        "main_annotation_status": (
            "NOT_GENERATED; requires pilot IAA and guideline gate"
        ),
        "blind_fields_excluded": [
            "rating",
            "product_id",
            "seller_id",
            "collection_transport",
            "model_prediction",
        ],
    }
    _write_json(annotation_root / "schema.json", schema)
    return {
        "annotation_eligible": len(eligible_indices),
        "pilot_candidate": pilot_count,
        "annotation_index": index_count,
        "clean_core_audit": clean_audit_count,
        "template_calibration": template_calibration_count,
        "flagged_human_review": flagged_review_count,
    }


def _write_data_card(
    root: Path,
    parent_manifest: dict[str, Any],
    config: dict[str, Any],
    counts: dict[str, Any],
    distributions: dict[str, Any],
) -> None:
    status_counts = counts["status"]
    reason_counts = counts["primary_reason"]
    text = f"""# Lazada Vietnamese ABSA Curation V2.1

## Release status

This is a **pre-annotation curation release**, not a labeled gold dataset.
It is derived from the frozen parent release
`{parent_manifest["release_id"]}` without modifying raw data.

Every one of the {counts["parent_records"]:,} parent records has exactly one
status:

| Status | Records | Meaning |
|---|---:|---|
| KEEP | {status_counts.get("KEEP", 0):,} | Original text admitted to clean-core |
| KEEP_CLEANED | {status_counts.get("KEEP_CLEANED", 0):,} | Audited deterministic text transformation |
| QUARANTINE | {status_counts.get("QUARANTINE", 0):,} | Human curation is required |
| EXCLUDE_AUTO | {status_counts.get("EXCLUDE_AUTO", 0):,} | Confirmed text/source duplicate only |

The clean-core currently contains
{status_counts.get("KEEP", 0) + status_counts.get("KEEP_CLEANED", 0):,}
records.  Quarantine decisions and the deterministic stratified
1,000-record clean-core audit must be completed before the corpus is called
final.

## Sequential method

1. Verify all parent SHA-256 entries and the one-to-one parent audit index.
2. Preserve `review_text` and derive `curated_review_text`.
3. Normalize only Unicode NFKC and whitespace.
4. Quarantine high-precision system/non-review matches; remove only pure
   reward clauses and exact repeated clauses; redact phone, email, and URL
   spans in the derived text.
5. Re-run the frozen substantive Vietnamese quality policy.
6. Resolve high-confidence API--DOM and punctuation-exact duplicates.
7. Compute corpus-wide and product-local clause-template evidence.
8. Compute conservative global/product word-5-gram near-duplicate candidates.
9. Compute deterministic within-product ranks for audit.  The canonical
   natural-observed corpus does not apply a product cap; cap-50/cap-200 may
   only be created later as train-only ablation views.
10. Partition all records and emit reversible ledgers plus checksums.

Frequency-only template evidence and fuzzy similarity never cause automatic
deletion.  They create a human-review queue.

## Corpus flow by primary reason

| Primary reason | Records |
|---|---:|
"""
    for reason, value in sorted(reason_counts.items()):
        text += f"| {reason} | {value:,} |\n"
    text += f"""

## Annotation state

`annotation/pilot_candidate.csv` contains
{counts["annotation"]["pilot_candidate"]:,} blinded, unlabeled candidates.
It is intentionally marked as a candidate: main annotation batches are not
generated until manual curation QC, pilot agreement, and guideline gates pass.
Blank cells mean **unlabeled**, while label `2` means **aspect absent**.

## Distribution warning

The parent candidate corpus intentionally selected substantive, relatively
long reviews and is not representative of all Lazada reviews.  Its natural
within-frame rating distribution is strongly five-star-skewed.  Rating is
retained only for audit/stratification and is never used to derive ABSA labels.

## Privacy and release ethics

Annotation files exclude product, seller, transport, rating, URLs, and model
predictions.  Potential phone/email/URL spans are replaced only in the curated
text and recorded by hash in the transformation ledger.  The full local
partition files retain parent provenance and therefore require a separate
legal/terms/privacy review before public redistribution.

## Reproducibility

- Cleaning rule version: `{config["rule_version"]}`
- Curation schema version: `{CURATION_SCHEMA_VERSION}`
- Parent source inventory:
  `{parent_manifest["source"]["source_inventory_sha256"]}`
- Checksums: `SHA256SUMS.txt`
- Detailed methodology:
  `docs/DATASET_CONSTRUCTION_PROTOCOL.docx`
"""
    (root / "DATA_CARD.md").write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )


def _write_readme(
    root: Path,
    counts: dict[str, Any],
    *,
    release_name: str,
) -> None:
    statuses = counts["status"]
    text = f"""# {release_name}

Pre-annotation ABSA curation release.

- Parent records: {counts["parent_records"]:,}
- Clean-core: {statuses.get("KEEP", 0) + statuses.get("KEEP_CLEANED", 0):,}
- Quarantine: {statuses.get("QUARANTINE", 0):,}
- Auto-excluded: {statuses.get("EXCLUDE_AUTO", 0):,}

Start with `DATA_CARD.md` and `manifest.json`.

Do not train from `annotation/pilot_candidate.csv`: its nine label columns are
blank by design.  Main annotation batches have not been released because the
pilot and manual curation gates have not yet passed.
"""
    (root / "README.md").write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )


def _copy_provenance(
    root: Path,
    parent_root: Path,
    config_path: Path,
    guideline_path: Path,
    code_paths: Sequence[Path],
) -> dict[str, str]:
    provenance = root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_root / "manifest.json", provenance / "parent_manifest.json")
    shutil.copy2(config_path, provenance / "cleaning_v2.json")
    shutil.copy2(guideline_path, provenance / "ABSA_ANNOTATION_GUIDELINE_V2.md")
    source_sums = parent_root / "provenance" / "SOURCE_SHA256SUMS.txt"
    shutil.copy2(source_sums, provenance / "SOURCE_SHA256SUMS.txt")
    code_rows = []
    for path in sorted(
        (path.resolve() for path in code_paths if path.is_file()),
        key=lambda value: value.as_posix(),
    ):
        try:
            relative = path.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            relative = path.name
        code_rows.append(f"{_sha256_file(path)}  {relative}")
    code_path = provenance / "CODE_SHA256SUMS.txt"
    code_path.write_text("\n".join(code_rows) + "\n", encoding="utf-8")
    return {
        "parent_manifest_sha256": _sha256_file(parent_root / "manifest.json"),
        "config_sha256": _sha256_file(config_path),
        "guideline_sha256": _sha256_file(guideline_path),
        "source_inventory_sha256": _sha256_file(source_sums),
        "code_inventory_sha256": _sha256_file(code_path),
    }


def _build_release(
    parent_root: Path,
    output_root: Path,
    config_path: Path,
    guideline_path: Path,
    *,
    built_at: str,
) -> dict[str, Any]:
    parent_root = parent_root.resolve()
    output_root = output_root.resolve()
    config_path = config_path.resolve()
    guideline_path = guideline_path.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output release already exists and will not be overwritten: {output_root}"
        )
    if not guideline_path.is_file():
        raise FileNotFoundError(f"Annotation guideline missing: {guideline_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(config["statuses"]) != EXPECTED_STATUSES:
        raise ValueError("Config must define exactly the four curation statuses")
    parent_checksum_entries = _verify_checksum_file(
        parent_root,
        parent_root / "SHA256SUMS.txt",
    )
    parent_manifest = json.loads(
        (parent_root / "manifest.json").read_text(encoding="utf-8")
    )
    parent_rows = _read_jsonl(parent_root / "reviews_canonical.jsonl")
    parent_audit_rows = _read_jsonl(parent_root / "record_audit.jsonl")
    expected_parent_count = int(parent_manifest["counts"]["canonical_records"])
    if len(parent_rows) != expected_parent_count:
        raise ValueError("Parent canonical count does not match parent manifest")
    parent_ids = [str(row.get("sample_id") or "") for row in parent_rows]
    if not all(parent_ids) or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("Parent sample_id values are blank or duplicated")
    audit_by_sample = {
        str(row.get("sample_id") or ""): row
        for row in parent_audit_rows
    }
    if len(audit_by_sample) != len(parent_audit_rows):
        raise ValueError("Parent audit sample IDs are duplicated")
    if set(audit_by_sample) != set(parent_ids):
        raise ValueError("Parent audit is not a one-to-one canonical index")
    for row_number, row in enumerate(parent_rows, 1):
        forbidden = _contains_forbidden_key(row)
        if forbidden:
            raise ValueError(
                f"Forbidden personal/secret metadata key at row {row_number}: "
                f"{forbidden}"
            )
        if not str(row.get("review_text") or "").strip():
            raise ValueError(f"Blank parent review text at row {row_number}")

    quality_policy = _policy_from_config(config)
    states = [
        _new_state(row, row_number, audit_by_sample[str(row["sample_id"])])
        for row_number, row in enumerate(parent_rows, 1)
    ]
    _prepare_text_states(states, config, quality_policy)
    _soft_nonreview_pass(states, config)

    duplicate_aliases, cross_transport_candidates = (
        _cross_transport_duplicates(states, config)
    )
    duplicate_aliases.extend(
        _confirmed_key_duplicates(
            states,
            key_name="source",
            key_function=lambda state: (
                f"{state['row']['product_id']}\0{state['row']['review_id']}"
            ),
            match_type="SOURCE_ID",
            reason_code="SOURCE_ID_DUPLICATE",
            config=config,
        )
    )
    duplicate_aliases.extend(
        _confirmed_key_duplicates(
            states,
            key_name="punct",
            key_function=lambda state: punctuation_symbol_key(
                state["curated_text"]
            ),
            match_type="PUNCTUATION_EXACT_BODY",
            reason_code="PUNCTUATION_EXACT_DUPLICATE",
            config=config,
        )
    )
    duplicate_aliases.sort(
        key=lambda row: (
            str(row["representative_sample_id"]),
            str(row["alias_sample_id"]),
            str(row["match_type"]),
        )
    )

    template_families = _template_pass(states, config)
    _structural_template_pass(states, config)
    _audit_escalation_pass(states, config)
    near_duplicate_candidates = _near_duplicate_pass(states, config)
    _apply_product_cap(states, config)
    _finalize_states(states, config)

    release_parent = output_root.parent
    release_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.building-",
        dir=release_parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        curation_rows = [
            _curation_row(
                state,
                str(parent_manifest["release_id"]),
                config,
            )
            for state in states
        ]
        full_rows = [
            _full_partition_row(state, curation)
            for state, curation in zip(states, curation_rows)
        ]
        artifact_records: dict[str, int | None] = {}
        artifact_records["curation_records.jsonl"] = _write_jsonl(
            temporary / "curation_records.jsonl",
            curation_rows,
        )
        for status, filename in (
            ({"KEEP", "KEEP_CLEANED"}, "clean_core.jsonl"),
            ({"QUARANTINE"}, "quarantine.jsonl"),
            ({"EXCLUDE_AUTO"}, "excluded.jsonl"),
        ):
            artifact_records[filename] = _write_jsonl(
                temporary / filename,
                (
                    row
                    for row, state in zip(full_rows, states)
                    if state["status"] in status
                ),
            )
        transformations = [
            transformation
            for state in states
            for transformation in state["transformations"]
        ]
        artifact_records["transformations.jsonl"] = _write_jsonl(
            temporary / "transformations.jsonl",
            transformations,
        )
        artifact_records["duplicate_aliases.jsonl"] = _write_jsonl(
            temporary / "duplicate_aliases.jsonl",
            duplicate_aliases,
        )
        artifact_records["cross_transport_candidates.jsonl"] = _write_jsonl(
            temporary / "cross_transport_candidates.jsonl",
            cross_transport_candidates,
        )
        artifact_records["near_duplicate_candidates.jsonl"] = _write_jsonl(
            temporary / "near_duplicate_candidates.jsonl",
            near_duplicate_candidates,
        )
        artifact_records["template_families.jsonl"] = _write_jsonl(
            temporary / "template_families.jsonl",
            template_families,
        )

        annotation_counts = _write_annotation_package(
            temporary,
            states,
            config,
        )
        artifact_records["annotation/pilot_candidate.csv"] = (
            annotation_counts["pilot_candidate"]
        )
        artifact_records["annotation/index.csv"] = (
            annotation_counts["annotation_index"]
        )
        artifact_records["annotation/curation_audit_1000.csv"] = (
            annotation_counts["clean_core_audit"]
        )
        artifact_records["annotation/curation_audit_index.csv"] = (
            annotation_counts["clean_core_audit"]
        )
        artifact_records["annotation/template_calibration.csv"] = (
            annotation_counts["template_calibration"]
        )
        artifact_records["annotation/template_calibration_index.csv"] = (
            annotation_counts["template_calibration"]
        )
        artifact_records["annotation/curation_review_queue.csv"] = (
            annotation_counts["flagged_human_review"]
        )
        artifact_records["annotation/schema.json"] = None

        code_paths = [
            Path(__file__),
            Path("scripts/validate_clean_release_v2.py"),
            Path("src/lazada_collector/curation.py"),
            Path("src/lazada_collector/quality.py"),
            Path("tests/test_curation.py"),
            Path("scripts/render_dataset_protocol.py"),
        ]
        provenance_hashes = _copy_provenance(
            temporary,
            parent_root,
            config_path,
            guideline_path,
            code_paths,
        )
        for relative in (
            "provenance/parent_manifest.json",
            "provenance/cleaning_v2.json",
            "provenance/ABSA_ANNOTATION_GUIDELINE_V2.md",
            "provenance/SOURCE_SHA256SUMS.txt",
            "provenance/CODE_SHA256SUMS.txt",
        ):
            artifact_records[relative] = None

        status_counts = Counter(state["status"] for state in states)
        primary_reason_counts = Counter(
            state["primary_reason"] for state in states
        )
        exclusion_trigger_counts = Counter(
            state["exclusion_trigger"]
            for state in states
            if state["exclusion_trigger"]
        )
        reason_code_counts = Counter(
            reason
            for state in states
            for reason in state["reason_codes"]
        )
        counts = {
            "parent_records": len(states),
            "status": dict(sorted(status_counts.items())),
            "primary_reason": dict(sorted(primary_reason_counts.items())),
            "exclusion_trigger": dict(
                sorted(exclusion_trigger_counts.items())
            ),
            "reason_code_flags": dict(sorted(reason_code_counts.items())),
            "transformations": len(transformations),
            "duplicate_aliases": len(duplicate_aliases),
            "confirmed_cross_transport_aliases": sum(
                row["match_type"] == "CROSS_TRANSPORT"
                for row in duplicate_aliases
            ),
            "cross_transport_candidates": len(
                cross_transport_candidates
            ),
            "ambiguous_cross_transport_candidates": sum(
                "CROSS_TRANSPORT_AMBIGUOUS" in state["reason_codes"]
                for state in states
            ),
            "near_duplicate_pairs": len(near_duplicate_candidates),
            "near_duplicate_clusters": len(
                {
                    row["cluster_id"]
                    for row in near_duplicate_candidates
                }
            ),
            "template_families": len(template_families),
            "annotation": annotation_counts,
        }
        distributions = {
            "all": {
                "rating": _distribution(states, "rating"),
                "category": _distribution(states, "category"),
                "transport": _distribution(
                    states,
                    "collection_transport",
                ),
            },
            "clean_core": {
                "rating": _distribution(
                    states,
                    "rating",
                    statuses={"KEEP", "KEEP_CLEANED"},
                ),
                "category": _distribution(
                    states,
                    "category",
                    statuses={"KEEP", "KEEP_CLEANED"},
                ),
                "transport": _distribution(
                    states,
                    "collection_transport",
                    statuses={"KEEP", "KEEP_CLEANED"},
                ),
            },
        }
        _write_data_card(
            temporary,
            parent_manifest,
            config,
            counts,
            distributions,
        )
        _write_readme(
            temporary,
            counts,
            release_name=output_root.name,
        )
        artifact_records["DATA_CARD.md"] = None
        artifact_records["README.md"] = None

        artifacts = []
        for relative, records in sorted(artifact_records.items()):
            artifacts.append(
                _artifact(
                    temporary / relative,
                    temporary,
                    records=records,
                )
            )
        release_id_payload = "\0".join(
            (
                str(parent_manifest["release_id"]),
                provenance_hashes["config_sha256"],
                provenance_hashes["code_inventory_sha256"],
                str(config["product_cap"]["selection_seed"]),
            )
        )
        release_id = "lazada-vi-absa-curation-" + sha256_text(
            release_id_payload
        )[:16]
        manifest = {
            "release_id": release_id,
            "release_name": output_root.name,
            "release_type": "PRE_ANNOTATION_CURATION",
            "release_schema_version": CURATION_SCHEMA_VERSION,
            "rule_version": config["rule_version"],
            "built_at": built_at,
            "source_cutoff_at": parent_manifest["source_cutoff_at"],
            "parent": {
                "release_id": parent_manifest["release_id"],
                "release_name": parent_manifest["release_name"],
                "canonical_records": expected_parent_count,
                "checksum_entries_verified": parent_checksum_entries,
                "manifest_sha256": provenance_hashes[
                    "parent_manifest_sha256"
                ],
                "source_inventory_sha256": parent_manifest["source"][
                    "source_inventory_sha256"
                ],
            },
            "provenance": provenance_hashes,
            "counts": counts,
            "distributions": distributions,
            "annotation": {
                "schema": {
                    "columns": ANNOTATION_COLUMNS,
                    "allowed_labels": config["annotation"][
                        "allowed_labels"
                    ],
                    "unlabeled_representation": "blank",
                    "legacy_tensor_polarity_order": [
                        "negative",
                        "positive",
                        "neutral",
                    ],
                },
                "pilot_status": (
                    "CANDIDATE_ONLY_PENDING_CURATION_QC"
                ),
                "main_annotation_status": (
                    "NOT_GENERATED_PENDING_PILOT_GATE"
                ),
            },
            "automatic_exclusion_policy": {
                "allowed": [
                    "confirmed source/punctuation duplicate",
                    "one-to-one unambiguous cross-transport duplicate",
                ],
                "forbidden": [
                    "hard non-review or privacy heuristic without human review",
                    "post-clean quality failure",
                    "frequency-only template evidence",
                    "near-duplicate similarity alone",
                    "rating, category, or sentiment assumption",
                    "product-cap overflow",
                ],
            },
            "artifacts": artifacts,
        }
        _write_json(temporary / "manifest.json", manifest)

        checksum_targets = sorted(
            (
                path
                for path in temporary.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS.txt"
            ),
            key=lambda path: path.relative_to(temporary).as_posix(),
        )
        checksum_lines = [
            f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}"
            for path in checksum_targets
        ]
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output_root)

    return {
        "release_id": release_id,
        "output": str(output_root),
        "counts": counts,
        "manifest": str(output_root / "manifest.json"),
        "checksums": str(output_root / "SHA256SUMS.txt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cleaning_v2.json"),
    )
    parser.add_argument(
        "--parent-release",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--guideline",
        type=Path,
        default=Path("docs/ABSA_ANNOTATION_GUIDELINE_V2.md"),
    )
    parser.add_argument(
        "--built-at",
        default=None,
        help=(
            "ISO-8601 build time. It is recorded as provenance; pass the same "
            "value for byte-reproducible rebuilds."
        ),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = args.parent_release or Path(config["parent_release"])
    output = args.output or Path(config["output_release"])
    built_at = args.built_at or datetime.now(timezone.utc).isoformat()
    try:
        datetime.fromisoformat(built_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("--built-at must be ISO-8601") from exc
    result = _build_release(
        parent,
        output,
        args.config,
        args.guideline,
        built_at=built_at,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
