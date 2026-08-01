"""Deterministic multi-label, leakage-group-aware fold construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence
import random

import numpy as np

from .schema import ASPECTS, POLARITIES


def _source_value(record: Mapping[str, Any], key: str) -> str:
    source = record.get("source")
    if not isinstance(source, Mapping):
        return ""
    value = source.get(key)
    return "" if value is None else str(value).strip()


def build_stratified_group_folds(
    records: Sequence[Mapping[str, Any]],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign each leakage group to one fold while balancing 27 labels.

    The feature vector also includes source domain and category so that the
    corpus mixture is approximately preserved. Returned fold indices are
    zero-based and deterministic for the same records, seed and fold count.
    """

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not records:
        raise ValueError("cannot construct folds from an empty record set")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        group_id = str(record.get("leakage_group_id", "")).strip()
        if not group_id:
            raise ValueError("every record must have a leakage_group_id")
        grouped[group_id].append(record)
    if len(grouped) < folds:
        raise ValueError(
            f"not enough leakage groups for {folds} folds: {len(grouped)}"
        )

    domains = sorted(
        {value for record in records if (value := _source_value(record, "domain"))}
    )
    categories = sorted(
        {
            value
            for record in records
            if (value := _source_value(record, "category"))
        }
    )
    domain_index = {value: index for index, value in enumerate(domains)}
    category_index = {value: index for index, value in enumerate(categories)}
    label_dimensions = len(ASPECTS) * len(POLARITIES)
    dimensions = label_dimensions + len(domains) + len(categories)

    def feature_vector(record: Mapping[str, Any]) -> np.ndarray:
        labels = np.asarray(record["sentiment_labels"], dtype=np.float64)
        expected = (len(ASPECTS), len(POLARITIES))
        if labels.shape != expected:
            raise ValueError(
                f"sentiment_labels shape must be {expected}, got {labels.shape}"
            )
        vector = np.zeros(dimensions, dtype=np.float64)
        vector[:label_dimensions] = labels.reshape(-1)
        domain = _source_value(record, "domain")
        if domain:
            vector[label_dimensions + domain_index[domain]] = 1.0
        category = _source_value(record, "category")
        if category:
            vector[
                label_dimensions + len(domains) + category_index[category]
            ] = 1.0
        return vector

    group_sizes: dict[str, float] = {}
    group_features: dict[str, np.ndarray] = {}
    for group_id, group_records in grouped.items():
        group_sizes[group_id] = float(len(group_records))
        group_features[group_id] = np.sum(
            [feature_vector(record) for record in group_records],
            axis=0,
        )

    total_size = float(len(records))
    total_features = np.sum(list(group_features.values()), axis=0)
    target_size = total_size / folds
    target_features = total_features / folds
    active_features = total_features > 0
    randomizer = random.Random(seed)
    group_ties = {group_id: randomizer.random() for group_id in grouped}

    def rarity_score(group_id: str) -> float:
        return float(
            np.sum(group_features[group_id] / np.maximum(total_features, 1.0))
        )

    ordered_groups = sorted(
        grouped,
        key=lambda group_id: (
            -rarity_score(group_id),
            -group_sizes[group_id],
            group_ties[group_id],
            group_id,
        ),
    )
    current_sizes = np.zeros(folds, dtype=np.float64)
    current_features = np.zeros((folds, dimensions), dtype=np.float64)
    assignments: dict[str, int] = {}

    for group_number, group_id in enumerate(ordered_groups):
        size = group_sizes[group_id]
        features = group_features[group_id]
        if group_number < folds:
            best_fold = group_number
        else:
            fold_order = list(range(folds))
            randomizer.shuffle(fold_order)
            best_fold = fold_order[0]
            best_cost: float | None = None
            for fold_index in fold_order:
                proposed_sizes = current_sizes.copy()
                proposed_features = current_features.copy()
                proposed_sizes[fold_index] += size
                proposed_features[fold_index] += features
                size_cost = float(
                    np.mean(
                        ((proposed_sizes - target_size) / max(target_size, 1.0))
                        ** 2
                    )
                )
                if np.any(active_features):
                    label_cost = float(
                        np.mean(
                            (
                                (
                                    proposed_features[:, active_features]
                                    - target_features[active_features]
                                )
                                / np.maximum(
                                    target_features[active_features], 1.0
                                )
                            )
                            ** 2
                        )
                    )
                else:
                    label_cost = 0.0
                overflow = np.maximum(
                    proposed_sizes - target_size * 1.08,
                    0.0,
                )
                overflow_cost = float(
                    np.sum((overflow / max(target_size, 1.0)) ** 2)
                )
                cost = size_cost + 2.0 * label_cost + 4.0 * overflow_cost
                if best_cost is None or cost < best_cost - 1e-12:
                    best_fold = fold_index
                    best_cost = cost
        assignments[group_id] = best_fold
        current_sizes[best_fold] += size
        current_features[best_fold] += features

    observed = Counter(assignments.values())
    if len(observed) != folds or any(observed[index] == 0 for index in range(folds)):
        raise RuntimeError(f"fold splitter created an empty fold: {observed}")
    return assignments


def validate_fold_assignments(
    records: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
    *,
    folds: int,
) -> dict[str, Any]:
    """Validate exact sample coverage and group isolation."""

    sample_ids: set[str] = set()
    groups_by_fold = {fold: set() for fold in range(folds)}
    records_by_fold = Counter()
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        group_id = str(record.get("leakage_group_id", "")).strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"missing or duplicate sample_id: {sample_id!r}")
        sample_ids.add(sample_id)
        if group_id not in assignments:
            raise ValueError(f"group has no fold assignment: {group_id}")
        fold = int(assignments[group_id])
        if fold < 0 or fold >= folds:
            raise ValueError(f"invalid fold index for {group_id}: {fold}")
        groups_by_fold[fold].add(group_id)
        records_by_fold[fold] += 1

    assigned_groups = set(assignments)
    observed_groups = set().union(*groups_by_fold.values())
    if assigned_groups != observed_groups:
        raise ValueError("fold assignments contain missing or unexpected groups")
    for left in range(folds):
        for right in range(left + 1, folds):
            if groups_by_fold[left] & groups_by_fold[right]:
                raise ValueError(f"leakage groups overlap folds {left} and {right}")
    if any(records_by_fold[index] == 0 for index in range(folds)):
        raise ValueError("one or more validation folds are empty")
    return {
        "status": "VALID",
        "folds": folds,
        "samples": len(sample_ids),
        "groups": len(observed_groups),
        "records_by_fold": {
            str(index + 1): records_by_fold[index] for index in range(folds)
        },
        "groups_by_fold": {
            str(index + 1): len(groups_by_fold[index]) for index in range(folds)
        },
    }
