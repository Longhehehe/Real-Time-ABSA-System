"""Prepare a blinded, versioned LLM pseudo-labeling pilot from curation v2.1.2."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from lazada_collector.llm_annotation import (
    ASPECT_COLUMNS,
    LLM_ANNOTATION_SCHEMA_VERSION,
    PROMPT_VERSION,
    build_system_prompt,
    canonical_json,
    parse_model_json,
    sha256_text,
    validate_output_schema_contract,
)


DEFAULT_RELEASE = Path(
    "data/releases/lazada_vi_absa_curation_v2_1_2_20260725"
)
DEFAULT_GUIDELINE = Path("docs/ABSA_ANNOTATION_GUIDELINE_V2.md")
DEFAULT_LLM_SCHEMA = Path("configs/llm_annotation_output_schema_v1.json")
DEFAULT_OUTPUT = Path("data/annotations/llm_pilot_v1_1_20260725")
BLIND_INPUT_SCHEMA_VERSION = "absa-llm-blind-input/1.0.0"

MODEL_VISIBLE_FIELDS = (
    "schema_version",
    "blind_id",
    "reviewContent",
)

BLIND_FIELDS_EXCLUDED = (
    "sample_id",
    "review_id",
    "rating",
    "category",
    "product_id",
    "seller_id",
    "shop_id",
    "product_url",
    "query",
    "sku",
    "review_date",
    "crawl_date",
    "collection_transport",
    "verified_purchase",
    "quality_score",
    "curation_status",
    "cleaning_flags",
    "raw_review_text",
    "old_labels",
    "human_labels",
    "model_predictions",
    "product_title",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = parse_model_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = parse_model_json(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def _annotation_id(sample_id: str, pilot_id: str) -> str:
    token = sha256_text(f"{pilot_id}\0{sample_id}")[:20]
    return f"llmp-{token}"


def _artifact(path: Path, root: Path, *, records: int | None = None) -> dict:
    value: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _validate_release_sources(
    *,
    release: Path,
    manifest: dict[str, Any],
    referenced_paths: list[Path],
) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Source release manifest has no artifact inventory")
    inventory: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Source release artifact entry must be an object")
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("Source release artifact path is invalid")
        if relative_path in inventory:
            raise ValueError(
                f"Duplicate source release artifact: {relative_path}"
            )
        inventory[relative_path] = artifact

    checksum_path = release / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    checksum_entries: dict[str, str] = {}
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError(
                    f"Invalid source checksum line {line_number}"
                )
            checksum, relative_path = parts
            if relative_path in checksum_entries:
                raise ValueError(
                    f"Duplicate source checksum path: {relative_path}"
                )
            checksum_entries[relative_path] = checksum

    verified: dict[str, str] = {}
    for path in referenced_paths:
        resolved = path.resolve()
        try:
            relative_path = resolved.relative_to(release).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Referenced source is outside release: {resolved}"
            ) from exc
        artifact = inventory.get(relative_path)
        if artifact is None:
            raise ValueError(
                f"Referenced source is absent from release manifest: "
                f"{relative_path}"
            )
        actual_sha = _sha256_file(resolved)
        if artifact.get("sha256") != actual_sha:
            raise ValueError(
                f"Source manifest checksum mismatch: {relative_path}"
            )
        if artifact.get("bytes") != resolved.stat().st_size:
            raise ValueError(
                f"Source manifest byte count mismatch: {relative_path}"
            )
        if checksum_entries.get(relative_path) != actual_sha:
            raise ValueError(
                f"Source SHA256SUMS mismatch: {relative_path}"
            )
        verified[relative_path] = actual_sha

    manifest_path = release / "manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    if checksum_entries.get("manifest.json") != manifest_sha:
        raise ValueError("Source SHA256SUMS mismatch: manifest.json")
    verified["manifest.json"] = manifest_sha
    verified["SHA256SUMS.txt"] = _sha256_file(checksum_path)
    return verified


def prepare(
    *,
    release: Path,
    guideline: Path,
    llm_schema: Path,
    output: Path,
    limit: int | None,
) -> dict[str, Any]:
    release = release.resolve()
    guideline = guideline.resolve()
    llm_schema = llm_schema.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    manifest_path = release / "manifest.json"
    clean_core_path = release / "clean_core.jsonl"
    pilot_path = release / "annotation" / "pilot_candidate.csv"
    schema_path = release / "annotation" / "schema.json"
    for required in (
        manifest_path,
        clean_core_path,
        pilot_path,
        schema_path,
        guideline,
        llm_schema,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    release_manifest = _read_json(manifest_path)
    source_checksums = _validate_release_sources(
        release=release,
        manifest=release_manifest,
        referenced_paths=[clean_core_path, pilot_path, schema_path],
    )
    annotation_schema = _read_json(schema_path)
    validate_output_schema_contract(_read_json(llm_schema))
    if annotation_schema.get("aspect_columns") != list(ASPECT_COLUMNS):
        raise ValueError("Release annotation schema aspect order mismatch")
    guideline_text = guideline.read_text(encoding="utf-8")
    guideline_sha = _sha256_file(guideline)
    legacy_schema_sha = _sha256_file(schema_path)
    llm_schema_sha = _sha256_file(llm_schema)

    core_by_text: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(clean_core_path):
        text = row.get("curated_review_text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Invalid clean-core text: {row.get('sample_id')}")
        core_by_text.setdefault(text, []).append(row)

    with pilot_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = ["reviewContent", *ASPECT_COLUMNS]
        if reader.fieldnames != expected_header:
            raise ValueError(
                "Pilot candidate header mismatch; expected exact legacy order"
            )
        candidate_rows = list(reader)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        candidate_rows = candidate_rows[:limit]
    if not candidate_rows:
        raise ValueError("Pilot candidate file is empty")

    mapped: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for rank, candidate in enumerate(candidate_rows, 1):
        text = candidate.get("reviewContent")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Pilot row {rank} has empty reviewContent")
        nonblank_labels = {
            aspect: candidate.get(aspect)
            for aspect in ASPECT_COLUMNS
            if (candidate.get(aspect) or "").strip()
        }
        if nonblank_labels:
            raise ValueError(
                f"Pilot row {rank} contains pre-existing labels: "
                f"{sorted(nonblank_labels)}"
            )
        matches = core_by_text.get(text, [])
        if len(matches) != 1:
            raise ValueError(
                f"Pilot row {rank} maps to {len(matches)} clean-core records"
            )
        source = matches[0]
        sample_id = str(source["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate pilot sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        text_hash = sha256_text(text)
        recorded_hash = source["curation"]["curated_text_sha256"]
        if text_hash != recorded_hash:
            raise ValueError(f"Curated text hash mismatch: {sample_id}")
        mapped.append(
            {
                "pilot_rank": rank,
                "sample_id": sample_id,
                "review_text": text,
                "review_text_sha256": text_hash,
                "parent_canonical_row": source["curation"][
                    "parent_canonical_row"
                ],
                "curation_status": source["curation"]["status"],
                "category": source.get("category") or "<blank>",
                "rating": source.get("rating"),
                "collection_transport": source.get("collection_transport"),
            }
        )

    pilot_seed_material = "\n".join(row["sample_id"] for row in mapped)
    pilot_id = (
        "llm-absa-pilot-"
        + sha256_text(
            "\0".join(
                (
                    str(release_manifest["release_id"]),
                    guideline_sha,
                    llm_schema_sha,
                    PROMPT_VERSION,
                    LLM_ANNOTATION_SCHEMA_VERSION,
                    pilot_seed_material,
                )
            )
        )[:16]
    )

    blind_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for row in mapped:
        annotation_id = _annotation_id(row["sample_id"], pilot_id)
        blind_rows.append(
            {
                "schema_version": BLIND_INPUT_SCHEMA_VERSION,
                "blind_id": annotation_id,
                "reviewContent": row["review_text"],
            }
        )
        index_rows.append(
            {
                "blind_id": annotation_id,
                "pilot_rank": row["pilot_rank"],
                "sample_id": row["sample_id"],
                "review_text_sha256": row["review_text_sha256"],
                "parent_canonical_row": row["parent_canonical_row"],
                "curation_status": row["curation_status"],
                "category": row["category"],
                "rating": row["rating"],
                "collection_transport": row["collection_transport"],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.building-",
        dir=output.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        blind_path = temporary / "pilot_blind.jsonl"
        index_path = temporary / "pilot_private_index.jsonl"
        direct_prompt_path = temporary / "system_prompt_direct.txt"
        evidence_prompt_path = temporary / "system_prompt_evidence_first.txt"
        copied_legacy_schema_path = temporary / "legacy_annotation_schema.json"
        copied_llm_schema_path = temporary / "llm_annotation_output_schema.json"
        copied_guideline_path = temporary / "ABSA_ANNOTATION_GUIDELINE_V2.md"

        _write_jsonl(blind_path, blind_rows)
        _write_jsonl(index_path, index_rows)
        direct_prompt_path.write_text(
            build_system_prompt(
                guideline_text,
                prompt_variant="direct",
            ),
            encoding="utf-8",
            newline="\n",
        )
        evidence_prompt_path.write_text(
            build_system_prompt(
                guideline_text,
                prompt_variant="evidence_first",
            ),
            encoding="utf-8",
            newline="\n",
        )
        shutil.copyfile(schema_path, copied_legacy_schema_path)
        shutil.copyfile(llm_schema, copied_llm_schema_path)
        shutil.copyfile(guideline, copied_guideline_path)

        artifacts = [
            _artifact(blind_path, temporary, records=len(blind_rows)),
            _artifact(index_path, temporary, records=len(index_rows)),
            _artifact(direct_prompt_path, temporary),
            _artifact(evidence_prompt_path, temporary),
            _artifact(copied_legacy_schema_path, temporary),
            _artifact(copied_llm_schema_path, temporary),
            _artifact(copied_guideline_path, temporary),
        ]
        manifest: dict[str, Any] = {
            "artifact_type": "LLM_PSEUDO_LABEL_PILOT_INPUT",
            "annotation_output_schema_version": LLM_ANNOTATION_SCHEMA_VERSION,
            "artifacts": artifacts,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "blind_fields_excluded": list(BLIND_FIELDS_EXCLUDED),
            "blind_input_schema_version": BLIND_INPUT_SCHEMA_VERSION,
            "guideline_sha256": guideline_sha,
            "pilot_id": pilot_id,
            "pilot_records": len(blind_rows),
            "prompt_version": PROMPT_VERSION,
            "release_id": release_manifest["release_id"],
            "release_manifest_sha256": _sha256_file(manifest_path),
            "release_name": release_manifest["release_name"],
            "legacy_schema_sha256": legacy_schema_sha,
            "llm_output_schema_sha256": llm_schema_sha,
            "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
            "source_artifact_sha256": source_checksums,
            "source_pilot_candidate_sha256": _sha256_file(pilot_path),
            "status": "PREPARED_NOT_LABELED",
        }
        manifest_output = temporary / "manifest.json"
        _write_json(manifest_output, manifest)

        checksum_paths = [
            *(temporary / artifact["path"] for artifact in artifacts),
            manifest_output,
        ]
        checksum_lines = [
            f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}"
            for path in sorted(
                checksum_paths,
                key=lambda item: item.relative_to(temporary).as_posix(),
            )
        ]
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)

    return {
        "status": "PREPARED_NOT_LABELED",
        "pilot_id": pilot_id,
        "pilot_records": len(blind_rows),
        "output": str(output),
        "manifest": str(output / "manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--guideline", type=Path, default=DEFAULT_GUIDELINE)
    parser.add_argument("--llm-schema", type=Path, default=DEFAULT_LLM_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        help="Prepare only the first N frozen pilot candidates.",
    )
    args = parser.parse_args()
    result = prepare(
        release=args.release,
        guideline=args.guideline,
        llm_schema=args.llm_schema,
        output=args.output,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
