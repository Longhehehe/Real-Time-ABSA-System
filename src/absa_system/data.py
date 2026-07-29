"""Build and validate leakage-controlled model-ready ABSA releases."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
import hashlib
import json
import os
import random
import shutil
import tempfile

import numpy as np

from .schema import (
    ABSAExample,
    ASPECTS,
    POLARITIES,
    AnnotationSchemaError,
    parse_ai_annotation_record,
    validate_model_record,
)


MODEL_RELEASE_STATUS = "DEVELOPMENT_PSEUDO_MODEL_READY_NOT_GOLD"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield line_number, row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(prefix: str, values: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:20]}"


class UnionFind:
    def __init__(self, values: Iterable[str] = ()) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
        for value in values:
            self.add(value)

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
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


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    domain: str


@dataclass(frozen=True)
class Candidate:
    example: ABSAExample
    source_line: int


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "absa-model-data-config/1.0.0":
        raise ValueError("unsupported model data config schema_version")
    return config


def _resolve_sources(project_root: Path, config: Mapping[str, Any]) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for row in config.get("sources", []):
        if not isinstance(row, Mapping):
            raise ValueError("source config rows must be objects")
        path = (project_root / str(row["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"annotation source does not exist: {path}")
        sources.append(
            SourceSpec(
                name=str(row["name"]),
                path=path,
                domain=str(row["domain"]),
            )
        )
    if not sources:
        raise ValueError("model data config has no sources")
    return sources


def _load_reservations(
    project_root: Path,
    config: Mapping[str, Any],
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    sample_ids: set[str] = set()
    text_hashes: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for relative in config.get("reservation_ledgers", []):
        path = (project_root / str(relative)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"reservation ledger does not exist: {path}")
        count = 0
        for _, row in iter_jsonl(path):
            sample_id = row.get("sample_id")
            text_hash = row.get("review_text_sha256")
            if isinstance(sample_id, str) and sample_id:
                sample_ids.add(sample_id)
            if isinstance(text_hash, str) and text_hash:
                text_hashes.add(text_hash)
            count += 1
        provenance.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "records": count,
                "sha256": sha256_file(path),
            }
        )
    return sample_ids, text_hashes, provenance


def _add_group_key(
    key_owner: dict[str, str],
    union_find: UnionFind,
    sample_id: str,
    namespace: str,
    value: Any,
) -> None:
    if value is None:
        return
    normalized = str(value).strip()
    if not normalized:
        return
    key = f"{namespace}:{normalized}"
    owner = key_owner.get(key)
    if owner is None:
        key_owner[key] = sample_id
    else:
        union_find.union(owner, sample_id)


def build_leakage_groups(
    examples: Sequence[ABSAExample],
    *,
    curation_paths: Sequence[Path],
) -> dict[str, str]:
    """Connect records by product, exact text, duplicate and template families."""

    sample_ids = {example.sample_id for example in examples}
    union_find = UnionFind(sample_ids)
    key_owner: dict[str, str] = {}
    by_id = {example.sample_id: example for example in examples}
    for example in examples:
        _add_group_key(
            key_owner,
            union_find,
            example.sample_id,
            "text",
            example.review_text_sha256,
        )
        _add_group_key(
            key_owner,
            union_find,
            example.sample_id,
            "product",
            example.source_metadata.get("product_id"),
        )

    for path in curation_paths:
        if not path.is_file():
            raise FileNotFoundError(f"curation source does not exist: {path}")
        for _, row in iter_jsonl(path):
            sample_id = row.get("sample_id")
            if sample_id not in sample_ids:
                continue
            _add_group_key(key_owner, union_find, sample_id, "product", row.get("product_id"))
            _add_group_key(
                key_owner,
                union_find,
                sample_id,
                "duplicate",
                row.get("duplicate_cluster_id"),
            )
            _add_group_key(
                key_owner,
                union_find,
                sample_id,
                "near_duplicate",
                row.get("near_duplicate_cluster_id"),
            )
            _add_group_key(
                key_owner,
                union_find,
                sample_id,
                "near_representative",
                row.get("near_duplicate_representative_sample_id"),
            )
            template_evidence = row.get("template_evidence")
            template_family = row.get("template_family_id")
            if template_family is None and isinstance(template_evidence, Mapping):
                template_family = template_evidence.get("template_family_id")
            _add_group_key(
                key_owner,
                union_find,
                sample_id,
                "template",
                template_family,
            )

    members: dict[str, list[str]] = defaultdict(list)
    for sample_id in by_id:
        members[union_find.find(sample_id)].append(sample_id)
    group_id_by_sample: dict[str, str] = {}
    for component in members.values():
        group_id = stable_hash("absa-lkg", component)
        for sample_id in component:
            group_id_by_sample[sample_id] = group_id
    return group_id_by_sample


def _stratified_group_split_partition(
    examples: Sequence[ABSAExample],
    group_id_by_sample: Mapping[str, str],
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Deterministic greedy group split balancing size and 27 label supports."""

    split_names = ("train", "dev", "test")
    ratio_array = np.asarray([float(ratios[name]) for name in split_names])
    if np.any(ratio_array <= 0) or not np.isclose(ratio_array.sum(), 1.0):
        raise ValueError("split ratios must be positive and sum to one")

    grouped: dict[str, list[ABSAExample]] = defaultdict(list)
    for example in examples:
        grouped[group_id_by_sample[example.sample_id]].append(example)
    if len(grouped) < len(split_names):
        raise ValueError("not enough leakage groups for train/dev/test")

    domains = sorted({example.source_domain for example in examples})
    categories = sorted(
        {
            str(example.source_metadata.get("category"))
            for example in examples
            if example.source_metadata.get("category") not in {None, ""}
        }
    )
    domain_index = {value: index for index, value in enumerate(domains)}
    category_index = {value: index for index, value in enumerate(categories)}
    base_dimension = len(ASPECTS) * len(POLARITIES)

    def feature_vector(example: ABSAExample) -> np.ndarray:
        vector = np.zeros(
            base_dimension + len(domains) + len(categories),
            dtype=np.float64,
        )
        vector[:base_dimension] = np.asarray(
            example.flattened_sentiment_labels,
            dtype=np.float64,
        )
        vector[
            base_dimension + domain_index[example.source_domain]
        ] = 1.0
        category = example.source_metadata.get("category")
        if category not in {None, ""}:
            vector[
                base_dimension
                + len(domains)
                + category_index[str(category)]
            ] = 1.0
        return vector

    total_size = len(examples)
    total_labels = np.sum([feature_vector(example) for example in examples], axis=0)
    target_sizes = ratio_array * total_size
    target_labels = ratio_array[:, None] * total_labels[None, :]
    current_sizes = np.zeros(len(split_names), dtype=np.float64)
    current_labels = np.zeros_like(target_labels)
    assignments: dict[str, str] = {}

    randomizer = random.Random(seed)
    tie_breakers = {group_id: randomizer.random() for group_id in grouped}

    def rarity_score(group_examples: Sequence[ABSAExample]) -> float:
        counts = np.sum(
            [feature_vector(example) for example in group_examples], axis=0
        )
        return float(np.sum(counts / np.maximum(total_labels, 1.0)))

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            -rarity_score(item[1]),
            -len(item[1]),
            tie_breakers[item[0]],
            item[0],
        ),
    )
    for group_id, group_examples in ordered_groups:
        group_size = float(len(group_examples))
        group_labels = np.sum(
            [feature_vector(example) for example in group_examples], axis=0
        )
        best_index = 0
        best_cost: float | None = None
        for split_index in range(len(split_names)):
            proposed_sizes = current_sizes.copy()
            proposed_labels = current_labels.copy()
            proposed_sizes[split_index] += group_size
            proposed_labels[split_index] += group_labels
            size_cost = np.mean(
                ((proposed_sizes - target_sizes) / np.maximum(target_sizes, 1.0)) ** 2
            )
            active = total_labels > 0
            label_cost = np.mean(
                (
                    (proposed_labels[:, active] - target_labels[:, active])
                    / np.maximum(target_labels[:, active], 1.0)
                )
                ** 2
            )
            overflow = np.maximum(proposed_sizes - target_sizes * 1.08, 0.0)
            overflow_cost = float(
                np.sum((overflow / np.maximum(target_sizes, 1.0)) ** 2)
            )
            cost = float(size_cost + 2.0 * label_cost + 4.0 * overflow_cost)
            if best_cost is None or cost < best_cost - 1e-12:
                best_index = split_index
                best_cost = cost
        assignments[group_id] = split_names[best_index]
        current_sizes[best_index] += group_size
        current_labels[best_index] += group_labels

    observed = Counter(assignments.values())
    if any(observed[name] == 0 for name in split_names):
        raise RuntimeError(f"group splitter created an empty split: {observed}")
    return assignments


def stratified_group_split(
    examples: Sequence[ABSAExample],
    group_id_by_sample: Mapping[str, str],
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Split disconnected crawled/legacy corpora independently, then merge.

    Product-sized crawled groups can otherwise fill development/test before
    the many singleton legacy records are considered, producing a misleading
    domain shift. The two source families are disconnected by construction;
    this wrapper verifies that assumption before balancing each family with
    the same label/domain/category objective.
    """

    partitions: dict[str, list[ABSAExample]] = defaultdict(list)
    for example in examples:
        family = "legacy_old" if example.source_domain == "legacy_old" else "crawled"
        partitions[family].append(example)
    if len(partitions) == 1:
        return _stratified_group_split_partition(
            examples,
            group_id_by_sample,
            ratios=ratios,
            seed=seed,
        )

    family_by_group: dict[str, str] = {}
    for family, family_examples in partitions.items():
        for example in family_examples:
            group_id = group_id_by_sample[example.sample_id]
            previous = family_by_group.setdefault(group_id, family)
            if previous != family:
                raise ValueError(
                    f"leakage group {group_id} crosses source families "
                    f"{previous}/{family}"
                )
    assignments: dict[str, str] = {}
    for offset, family in enumerate(sorted(partitions)):
        family_examples = partitions[family]
        family_groups = {
            group_id_by_sample[example.sample_id]
            for example in family_examples
        }
        if len(family_groups) < 3:
            raise ValueError(
                f"source family {family!r} has fewer than three leakage groups"
            )
        assignments.update(
            _stratified_group_split_partition(
                family_examples,
                group_id_by_sample,
                ratios=ratios,
                seed=seed + offset * 1009,
            )
        )
    return assignments


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def _distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    mention = np.asarray([row["mention_labels"] for row in records], dtype=np.int64)
    sentiment = np.asarray(
        [row["sentiment_labels"] for row in records], dtype=np.int64
    )
    return {
        "records": len(records),
        "groups": len({row["leakage_group_id"] for row in records}),
        "mentions_by_aspect": {
            aspect: int(mention[:, index].sum())
            for index, aspect in enumerate(ASPECTS)
        },
        "sentiments_by_aspect": {
            aspect: {
                polarity: int(sentiment[:, aspect_index, polarity_index].sum())
                for polarity_index, polarity in enumerate(POLARITIES)
            }
            for aspect_index, aspect in enumerate(ASPECTS)
        },
        "mixed_aspect_instances": int(
            np.logical_and(sentiment[:, :, 0], sentiment[:, :, 1]).sum()
        ),
    }


def _artifact_entry(path: Path, root: Path, *, records: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        entry["records"] = records
    return entry


def build_model_ready_release(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish a versioned, closed-inventory model-ready pseudo-label release."""

    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {output_dir}")
    config = load_config(config_path)
    sources = _resolve_sources(project_root, config)
    allowed_statuses = set(config.get("allowed_statuses", ["LABELED"]))
    if not allowed_statuses or not allowed_statuses <= {"LABELED", "ESCALATE"}:
        raise ValueError("allowed_statuses must be a subset of LABELED/ESCALATE")
    reserved_ids, reserved_hashes, reservation_provenance = _load_reservations(
        project_root, config
    )

    candidates: list[Candidate] = []
    ledger: list[dict[str, Any]] = []
    source_provenance: list[dict[str, Any]] = []
    for source in sources:
        source_count = 0
        for line_number, row in iter_jsonl(source.path):
            source_count += 1
            sample_id = row.get("sample_id")
            text_hash = row.get("review_text_sha256")
            status = (row.get("annotation") or {}).get("annotation_status")
            ledger_base = {
                "schema_version": "absa-model-decision/1.0.0",
                "source_package": source.name,
                "source_line": line_number,
                "sample_id": sample_id,
                "review_text_sha256": text_hash,
                "annotation_status": status,
            }
            if status not in {"LABELED", "ESCALATE", "REJECT_NON_REVIEW"}:
                raise AnnotationSchemaError(
                    f"{source.path}:{line_number}: invalid annotation status"
                )
            if status == "REJECT_NON_REVIEW":
                ledger.append(
                    {
                        **ledger_base,
                        "decision": "EXCLUDE_NON_REVIEW",
                        "reason": "annotation_status=REJECT_NON_REVIEW",
                    }
                )
                continue
            if status not in allowed_statuses:
                ledger.append(
                    {
                        **ledger_base,
                        "decision": "EXCLUDE_ANNOTATION_STATUS",
                        "reason": f"annotation_status={status}",
                    }
                )
                continue
            if sample_id in reserved_ids or text_hash in reserved_hashes:
                ledger.append(
                    {
                        **ledger_base,
                        "decision": "EXCLUDE_HUMAN_GOLD_RESERVATION",
                        "reason": "sample_id_or_text_hash_in_reservation_ledger",
                    }
                )
                continue
            example = parse_ai_annotation_record(
                row,
                source_package=source.name,
                source_domain=source.domain,
            )
            candidates.append(Candidate(example, line_number))
            ledger.append(
                {
                    **ledger_base,
                    "decision": "CANDIDATE",
                    "reason": "passed_status_reservation_and_schema_gates",
                }
            )
        source_provenance.append(
            {
                "name": source.name,
                "domain": source.domain,
                "path": source.path.relative_to(project_root).as_posix(),
                "records": source_count,
                "sha256": sha256_file(source.path),
            }
        )

    # Source order is an explicit priority order. Keep one canonical text only.
    kept_by_hash: dict[str, Candidate] = {}
    duplicate_ids: set[str] = set()
    for candidate in candidates:
        existing = kept_by_hash.get(candidate.example.review_text_sha256)
        if existing is None:
            kept_by_hash[candidate.example.review_text_sha256] = candidate
        else:
            duplicate_ids.add(candidate.example.sample_id)
    if duplicate_ids:
        for row in ledger:
            if row.get("sample_id") in duplicate_ids and row["decision"] == "CANDIDATE":
                row["decision"] = "EXCLUDE_EXACT_TEXT_DUPLICATE"
                row["reason"] = "lower_priority_duplicate_of_canonical_text_hash"

    examples = [candidate.example for candidate in kept_by_hash.values()]
    curation_paths = [
        (project_root / str(relative)).resolve()
        for relative in config.get("curation_sources", [])
    ]
    group_id_by_sample = build_leakage_groups(
        examples,
        curation_paths=curation_paths,
    )
    group_assignments = stratified_group_split(
        examples,
        group_id_by_sample,
        ratios=config["split"],
        seed=int(config["seed"]),
    )
    split_by_sample = {
        example.sample_id: group_assignments[group_id_by_sample[example.sample_id]]
        for example in examples
    }
    for row in ledger:
        if row["decision"] == "CANDIDATE":
            sample_id = row["sample_id"]
            row["decision"] = f"INCLUDE_{split_by_sample[sample_id].upper()}"
            row["reason"] = "canonical_eligible_record_group_assigned"
            row["leakage_group_id"] = group_id_by_sample[sample_id]

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    for example in examples:
        split = split_by_sample[example.sample_id]
        records_by_split[split].append(
            example.as_model_record(
                split=split,
                leakage_group_id=group_id_by_sample[example.sample_id],
            )
        )
    for rows in records_by_split.values():
        rows.sort(key=lambda row: row["sample_id"])
    ledger.sort(
        key=lambda row: (
            row["source_package"],
            int(row["source_line"]),
        )
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.building-",
            dir=output_dir.parent,
        )
    )
    try:
        record_counts: dict[str, int] = {}
        artifact_entries: list[dict[str, Any]] = []
        for split, rows in records_by_split.items():
            path = temporary / f"{split}.jsonl"
            record_counts[split] = _write_jsonl(path, rows)
            artifact_entries.append(
                _artifact_entry(path, temporary, records=record_counts[split])
            )
        ledger_path = temporary / "decision_ledger.jsonl"
        ledger_count = _write_jsonl(ledger_path, ledger)
        artifact_entries.append(
            _artifact_entry(ledger_path, temporary, records=ledger_count)
        )
        config_snapshot = temporary / "data_config.json"
        _write_json(config_snapshot, config)
        artifact_entries.append(_artifact_entry(config_snapshot, temporary))
        readme_path = temporary / "README.md"
        readme_path.write_text(
            "\n".join(
                [
                    f"# {output_dir.name}",
                    "",
                    f"Status: `{MODEL_RELEASE_STATUS}`.",
                    "",
                    "This release is a leakage-controlled engineering input for",
                    "model development. It is not a human-gold benchmark.",
                    "",
                    "Canonical files: `train.jsonl`, `dev.jsonl`, `test.jsonl`,",
                    "`decision_ledger.jsonl`, `manifest.json`, and `SHA256SUMS.txt`.",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        artifact_entries.append(_artifact_entry(readme_path, temporary))

        split_groups = {
            split: {row["leakage_group_id"] for row in rows}
            for split, rows in records_by_split.items()
        }
        overlap = {
            "train_dev": len(split_groups["train"] & split_groups["dev"]),
            "train_test": len(split_groups["train"] & split_groups["test"]),
            "dev_test": len(split_groups["dev"] & split_groups["test"]),
        }
        if any(overlap.values()):
            raise RuntimeError(f"leakage group overlap detected: {overlap}")
        manifest = {
            "schema_version": "absa-model-ready-release/1.0.0",
            "release_id": stable_hash(
                "absa-model-ready",
                [
                    (
                        f"{example.sample_id}:"
                        f"{group_id_by_sample[example.sample_id]}:"
                        f"{split_by_sample[example.sample_id]}"
                    )
                    for example in examples
                ]
                + [sha256_file(config_path)],
            ),
            "status": MODEL_RELEASE_STATUS,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_config": {
                "path": config_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(config_path),
            },
            "source_packages": source_provenance,
            "reservation_ledgers": reservation_provenance,
            "curation_sources": [
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in curation_paths
            ],
            "policy": {
                "allowed_statuses": sorted(allowed_statuses),
                "exact_text_deduplication": True,
                "reservation_exclusion": True,
                "grouping": [
                    "product_id",
                    "review_text_sha256",
                    "duplicate_cluster_id",
                    "near_duplicate_cluster_id",
                    "near_duplicate_representative_sample_id",
                    "template_family_id",
                ],
                "stratification": [
                    "27 aspect-polarity labels",
                    "source_domain",
                    "category",
                ],
                "split_ratios": config["split"],
                "seed": int(config["seed"]),
                "evaluation_label_warning": (
                    "All labels remain pseudo labels pending human verification; "
                    "dev/test are engineering splits, not final gold benchmarks."
                ),
            },
            "counts": {
                "source_records": len(ledger),
                "candidate_before_exact_dedup": len(candidates),
                "model_ready_unique": len(examples),
                "decisions": dict(Counter(row["decision"] for row in ledger)),
                "splits": record_counts,
            },
            "distributions": {
                split: _distribution(rows)
                for split, rows in records_by_split.items()
            },
            "leakage_audit": {
                "group_overlap": overlap,
                "sample_id_unique": len(examples)
                == len({example.sample_id for example in examples}),
                "review_text_sha256_unique": len(examples)
                == len({example.review_text_sha256 for example in examples}),
                "reserved_sample_overlap": len(
                    {example.sample_id for example in examples} & reserved_ids
                ),
                "reserved_text_overlap": len(
                    {example.review_text_sha256 for example in examples}
                    & reserved_hashes
                ),
            },
            "artifacts": artifact_entries,
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        checksum_entries = artifact_entries + [_artifact_entry(manifest_path, temporary)]
        checksum_path = temporary / "SHA256SUMS.txt"
        checksum_path.write_text(
            "".join(
                f"{entry['sha256']}  {entry['path']}\n"
                for entry in sorted(checksum_entries, key=lambda item: item["path"])
            ),
            encoding="utf-8",
            newline="\n",
        )

        validate_model_ready_release(temporary)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_json(output_dir / "manifest.json")


def validate_model_ready_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "absa-model-ready-release/1.0.0":
        raise ValueError("unsupported model-ready release schema")
    if manifest.get("status") != MODEL_RELEASE_STATUS:
        raise ValueError("model-ready release status mismatch")

    expected_artifacts = {entry["path"]: entry for entry in manifest["artifacts"]}
    for relative, entry in expected_artifacts.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing release artifact: {relative}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"{relative}: byte count mismatch")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{relative}: checksum mismatch")
    checksum_path = root / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise FileNotFoundError("missing SHA256SUMS.txt")
    checksum_rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("malformed SHA256SUMS.txt") from exc
        checksum_rows[relative] = digest
    expected_checksums = set(expected_artifacts) | {"manifest.json"}
    if set(checksum_rows) != expected_checksums:
        raise ValueError("SHA256SUMS inventory mismatch")
    for relative, digest in checksum_rows.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{relative}: SHA256SUMS verification failed")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = expected_checksums | {"SHA256SUMS.txt"}
    if actual_files != expected_files:
        raise ValueError(
            "closed file inventory mismatch; "
            f"unexpected={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}"
        )

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    groups_by_split: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        path = root / f"{split}.jsonl"
        groups: set[str] = set()
        count = 0
        for _, row in iter_jsonl(path):
            validate_model_record(row)
            if row["split"] != split:
                raise ValueError(f"{path}: embedded split mismatch")
            sample_id = row["sample_id"]
            text_hash = row["review_text_sha256"]
            if sample_id in seen_ids:
                raise ValueError(f"duplicate model-ready sample_id: {sample_id}")
            if text_hash in seen_hashes:
                raise ValueError(f"duplicate model-ready review text: {text_hash}")
            seen_ids.add(sample_id)
            seen_hashes.add(text_hash)
            groups.add(row["leakage_group_id"])
            count += 1
        groups_by_split[split] = groups
        counts[split] = count
        if count != int(manifest["counts"]["splits"][split]):
            raise ValueError(f"{split}: record count mismatch")
    if groups_by_split["train"] & groups_by_split["dev"]:
        raise ValueError("train/dev leakage group overlap")
    if groups_by_split["train"] & groups_by_split["test"]:
        raise ValueError("train/test leakage group overlap")
    if groups_by_split["dev"] & groups_by_split["test"]:
        raise ValueError("dev/test leakage group overlap")
    return {
        "status": "VALID",
        "records": sum(counts.values()),
        "splits": counts,
        "groups": {
            split: len(groups)
            for split, groups in groups_by_split.items()
        },
    }
