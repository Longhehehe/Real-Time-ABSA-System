"""Fair six-model benchmark orchestration on leakage-controlled folds."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Mapping, Sequence
import os
import pickle
import sys
import tempfile
import time

import numpy as np
from tqdm.auto import tqdm

from .cross_validation import (
    _aggregate_fold_metrics,
    _concatenate_predictions,
    _deterministic_limit,
    _ensemble_test_predictions,
    _write_fold_assignments,
    seal_kfold_training_run,
    train_kfold_model,
    validate_kfold_training_run,
)
from .data import read_json, sha256_file, validate_model_ready_release
from .dataset import load_model_records
from .folds import build_stratified_group_folds, validate_fold_assignments
from .model_registry import MODEL_NAMES, get_model_spec
from .results import (
    validate_suite_comparison,
    write_run_metric_exports,
    write_suite_comparison,
)
from .schema import ASPECTS, POLARITIES
from .training import (
    _TQDM_BAR_FORMAT,
    _emit_console_event,
    _metric_summary,
    _write_json,
    evaluate_probabilities,
)


def _atomic_pickle(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _text(record: Mapping[str, Any]) -> str:
    return str(record["reviewContent"])


def _fit_binary_estimator(
    model_name: str,
    features: Any,
    target: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> Any:
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    target = np.asarray(target, dtype=np.int8)
    unique = np.unique(target)
    if len(unique) < 2:
        estimator = DummyClassifier(strategy="constant", constant=int(unique[0]))
    elif model_name == "logistic_regression":
        estimator = LogisticRegression(
            C=float(config.get("logistic_c", 1.0)),
            class_weight="balanced",
            max_iter=int(config.get("logistic_max_iter", 1000)),
            random_state=seed,
            solver="liblinear",
        )
    elif model_name == "naive_bayes":
        estimator = MultinomialNB(alpha=float(config.get("naive_bayes_alpha", 1.0)))
    else:
        raise ValueError(f"unsupported classical model: {model_name}")
    estimator.fit(features, target)
    return estimator


def _positive_probability(estimator: Any, features: Any) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive):
        return probabilities[:, int(positive[0])]
    return np.ones(len(probabilities), dtype=np.float64) if classes[0] == 1 else np.zeros(
        len(probabilities), dtype=np.float64
    )


def _fit_classical_fold(
    model_name: str,
    train_records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    seed: int,
    show_progress: bool,
) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        ngram_range=(
            int(config.get("tfidf_ngram_min", 1)),
            int(config.get("tfidf_ngram_max", 2)),
        ),
        max_features=int(config.get("tfidf_max_features", 10000)),
        min_df=int(config.get("tfidf_min_df", 2)),
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform([_text(row) for row in train_records])
    mention_true = np.asarray(
        [row["mention_labels"] for row in train_records], dtype=np.int8
    )
    sentiment_true = np.asarray(
        [row["sentiment_labels"] for row in train_records], dtype=np.int8
    )
    mention_estimators: list[Any] = []
    sentiment_estimators: list[list[Any]] = [
        [] for _ in range(len(ASPECTS))
    ]
    tasks = len(ASPECTS) + len(ASPECTS) * len(POLARITIES)
    progress = tqdm(
        total=tasks,
        desc=f"[{model_name}] fit heads",
        unit="head",
        disable=not show_progress,
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
        bar_format=_TQDM_BAR_FORMAT,
        file=sys.stdout,
    )
    try:
        for aspect_index in range(len(ASPECTS)):
            mention_estimators.append(
                _fit_binary_estimator(
                    model_name,
                    features,
                    mention_true[:, aspect_index],
                    config=config,
                    seed=seed + aspect_index,
                )
            )
            progress.update(1)
        for aspect_index in range(len(ASPECTS)):
            mentioned = mention_true[:, aspect_index].astype(bool)
            training_features = features[mentioned] if mentioned.any() else features
            for polarity_index in range(len(POLARITIES)):
                target = (
                    sentiment_true[mentioned, aspect_index, polarity_index]
                    if mentioned.any()
                    else np.zeros(len(train_records), dtype=np.int8)
                )
                sentiment_estimators[aspect_index].append(
                    _fit_binary_estimator(
                        model_name,
                        training_features,
                        target,
                        config=config,
                        seed=seed + 100 + aspect_index * len(POLARITIES) + polarity_index,
                    )
                )
                progress.update(1)
    finally:
        progress.close()
    return {
        "vectorizer": vectorizer,
        "mention_estimators": mention_estimators,
        "sentiment_estimators": sentiment_estimators,
        "aspects": list(ASPECTS),
        "polarities": list(POLARITIES),
        "model_name": model_name,
    }


def _classical_probabilities(
    fitted: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    features = fitted["vectorizer"].transform([_text(row) for row in records])
    mention_prob = np.column_stack(
        [
            _positive_probability(estimator, features)
            for estimator in fitted["mention_estimators"]
        ]
    )
    sentiment_prob = np.empty(
        (len(records), len(ASPECTS), len(POLARITIES)), dtype=np.float64
    )
    for aspect_index, estimators in enumerate(fitted["sentiment_estimators"]):
        for polarity_index, estimator in enumerate(estimators):
            sentiment_prob[:, aspect_index, polarity_index] = _positive_probability(
                estimator, features
            )
    return {
        "mention_true": np.asarray(
            [row["mention_labels"] for row in records], dtype=np.int8
        ),
        "sentiment_true": np.asarray(
            [row["sentiment_labels"] for row in records], dtype=np.int8
        ),
        "mention_prob": mention_prob,
        "sentiment_prob": sentiment_prob,
        "sample_ids": [str(row["sample_id"]) for row in records],
        "leakage_group_ids": [str(row["leakage_group_id"]) for row in records],
    }


def train_classical_kfold_model(
    *,
    model_name: str,
    data_release: Path,
    output_dir: Path,
    config_path: Path,
    folds_override: int | None = None,
    max_development_samples: int | None = None,
    max_test_samples: int | None = None,
    show_progress_override: bool | None = None,
) -> dict[str, Any]:
    """Train one classical baseline with the exact neural CV/test contract."""

    spec = get_model_spec(model_name)
    if spec.iterative:
        raise ValueError(f"{model_name} is not a classical model")
    data_release = data_release.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite K-fold run: {output_dir}")
    data_validation = validate_model_ready_release(data_release)
    data_manifest = read_json(data_release / "manifest.json")
    config = read_json(config_path)
    if config.get("schema_version") != "absa-training-config/1.0.0":
        raise ValueError("unsupported training config schema_version")
    if folds_override is not None:
        if folds_override < 2:
            raise ValueError("folds override must be at least 2")
        config["k_folds"] = int(folds_override)
    config["model_name"] = model_name
    config["runtime_overrides"] = {
        key: value
        for key, value in {
            "k_folds": folds_override,
            "show_progress": show_progress_override,
        }.items()
        if value is not None
    }
    if show_progress_override is not None:
        config["show_progress"] = bool(show_progress_override)
    folds = int(config.get("k_folds", 3))
    seed = int(config["seed"])
    show_progress = bool(config.get("show_progress", True))
    development_records = _deterministic_limit(
        [
            *load_model_records(data_release, "train"),
            *load_model_records(data_release, "dev"),
        ],
        max_development_samples,
        seed=seed,
    )
    test_records = _deterministic_limit(
        load_model_records(data_release, "test"),
        max_test_samples,
        seed=seed + 1,
    )
    assignments = build_stratified_group_folds(
        development_records, folds=folds, seed=seed
    )
    assignment_validation = validate_fold_assignments(
        development_records, assignments, folds=folds
    )
    development_groups = set(assignments)
    test_groups = {str(row["leakage_group_id"]) for row in test_records}
    if development_groups & test_groups:
        raise ValueError("locked test leakage groups overlap K-fold development pool")
    group_sizes = Counter(str(row["leakage_group_id"]) for row in development_records)
    largest_group_id, largest_group_records = group_sizes.most_common(1)[0]
    fold_counts = assignment_validation["records_by_fold"]
    fold_size_ratio = max(fold_counts.values()) / min(fold_counts.values())
    target_fold_records = len(development_records) / folds
    warnings: list[str] = []
    if largest_group_records > target_fold_records:
        warnings.append(
            "The largest indivisible leakage group exceeds the target fold size."
        )
    if fold_size_ratio > 1.25:
        warnings.append("Validation fold sizes differ by more than 25%.")

    output_dir.mkdir(parents=True)
    _write_json(output_dir / "training_config.json", config)
    _write_fold_assignments(
        output_dir / "fold_assignments.jsonl", development_records, assignments
    )
    _write_json(
        output_dir / "cross_validation.json",
        {
            "schema_version": "absa-kfold-split/1.0.0",
            "method": "deterministic_greedy_multilabel_stratified_group_kfold",
            "folds": folds,
            "seed": seed,
            "development_sources": ["train", "dev"],
            "locked_test_source": "test",
            "features_balanced": [
                "27_aspect_polarity_labels",
                "source_domain",
                "source_category",
            ],
            "validation": assignment_validation,
            "largest_leakage_group": {
                "leakage_group_id": largest_group_id,
                "records": largest_group_records,
                "fraction_of_development": largest_group_records
                / len(development_records),
            },
            "target_records_per_fold": target_fold_records,
            "max_to_min_fold_size_ratio": fold_size_ratio,
            "warnings": warnings,
            "test_development_group_overlap": 0,
        },
    )
    data_manifest_sha256 = sha256_file(data_release / "manifest.json")
    run_metadata: dict[str, Any] = {
        "schema_version": "absa-kfold-training-run/1.0.0",
        "status": "RUNNING",
        "model_name": model_name,
        "model_family": spec.family,
        "checkpoint_filename": spec.checkpoint_filename,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_release": str(data_release),
        "data_release_id": data_manifest["release_id"],
        "data_manifest_sha256": data_manifest_sha256,
        "data_validation": data_validation,
        "config_sha256": sha256_file(config_path),
        "numpy_version": np.__version__,
        "scikit_learn_version": distribution_version("scikit-learn"),
        "tqdm_version": distribution_version("tqdm"),
        "seed": seed,
        "folds": folds,
        "early_stopping": {
            "enabled": False,
            "reason": "non_iterative_classical_estimator",
        },
        "sample_limits": {
            "development": max_development_samples,
            "test": max_test_samples,
        },
        "records": {
            "development": len(development_records),
            "locked_test": len(test_records),
        },
        "test_policy": (
            "No fold-level test metric is computed. Mean fold probabilities "
            "are evaluated once after all folds using pooled-OOF thresholds."
        ),
    }
    _write_json(output_dir / "run.json", run_metadata)
    _emit_console_event(
        "classical_kfold_started",
        model=model_name,
        folds=folds,
        development_records=len(development_records),
        locked_test_records=len(test_records),
    )

    fold_results: list[dict[str, Any]] = []
    oof_predictions: list[Mapping[str, Any]] = []
    test_predictions: list[Mapping[str, Any]] = []
    started = time.monotonic()
    for fold_index in range(folds):
        fold_number = fold_index + 1
        fold_started = time.monotonic()
        fold_seed = seed + fold_index * int(config.get("fold_seed_stride", 1009))
        validation_records = [
            row
            for row in development_records
            if assignments[str(row["leakage_group_id"])] == fold_index
        ]
        train_records = [
            row
            for row in development_records
            if assignments[str(row["leakage_group_id"])] != fold_index
        ]
        fold_dir = output_dir / "folds" / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True)
        fitted = _fit_classical_fold(
            model_name,
            train_records,
            config=config,
            seed=fold_seed,
            show_progress=show_progress,
        )
        validation_predictions = _classical_probabilities(fitted, validation_records)
        validation_metrics, thresholds = evaluate_probabilities(
            validation_predictions, thresholds=None
        )
        locked_test_predictions = _classical_probabilities(fitted, test_records)
        checkpoint = {
            "schema_version": "absa-classical-checkpoint/1.0.0",
            "model_name": model_name,
            "model_family": spec.family,
            "model": fitted,
            "training_config": config,
            "data_release_id": data_manifest["release_id"],
            "data_manifest_sha256": data_manifest_sha256,
            "cross_validation": {
                "fold": fold_number,
                "folds": folds,
                "seed": fold_seed,
            },
            "thresholds": thresholds.as_dict(),
        }
        checkpoint_path = fold_dir / spec.checkpoint_filename
        _atomic_pickle(checkpoint, checkpoint_path)
        _write_json(
            fold_dir / "thresholds.json",
            {
                **thresholds.as_dict(),
                "selected_on": "validation_fold",
                "fold": fold_number,
                "epoch": None,
            },
        )
        _write_json(fold_dir / "validation_metrics.json", validation_metrics)
        elapsed = time.monotonic() - fold_started
        with (fold_dir / "epochs.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            import json

            handle.write(
                json.dumps(
                    {
                        "event": "fit_completed",
                        "fold": fold_number,
                        "epoch": None,
                        "validation_primary_metric": "end_to_end_macro_f1",
                        "validation_primary_value": validation_metrics[
                            "end_to_end_macro_f1"
                        ],
                        "validation_metrics": validation_metrics,
                        "elapsed_seconds": elapsed,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        checkpoint_metadata = {
            "path": spec.checkpoint_filename,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        }
        fold_summary = {
            "schema_version": "absa-kfold-fold/1.0.0",
            "status": "COMPLETED",
            "model_name": model_name,
            "model_family": spec.family,
            "fold": fold_number,
            "folds": folds,
            "seed": fold_seed,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "test_records_for_blinded_ensemble_prediction": len(test_records),
            "train_groups": len(
                {str(row["leakage_group_id"]) for row in train_records}
            ),
            "validation_groups": len(
                {str(row["leakage_group_id"]) for row in validation_records}
            ),
            "best_epoch": None,
            "best_validation_end_to_end_macro_f1": validation_metrics[
                "end_to_end_macro_f1"
            ],
            "validation_metrics": _metric_summary(validation_metrics),
            "checkpoint": checkpoint_metadata,
        }
        _write_json(fold_dir / "fold.json", fold_summary)
        fold_results.append(
            {
                "fold": fold_number,
                "best_epoch": None,
                "best_validation_end_to_end_macro_f1": validation_metrics[
                    "end_to_end_macro_f1"
                ],
                "validation_metrics": validation_metrics,
                "validation_predictions": validation_predictions,
                "test_predictions": locked_test_predictions,
                "checkpoint": checkpoint_metadata,
            }
        )
        oof_predictions.append(validation_predictions)
        test_predictions.append(locked_test_predictions)
        _emit_console_event(
            "classical_fold_completed",
            model=model_name,
            fold=fold_number,
            folds=folds,
            validation_metrics=_metric_summary(validation_metrics),
            elapsed_seconds=elapsed,
        )

    pooled_oof_predictions = _concatenate_predictions(oof_predictions)
    if len(set(pooled_oof_predictions["sample_ids"])) != len(development_records):
        raise RuntimeError("OOF predictions do not cover development samples exactly once")
    pooled_oof_metrics, pooled_oof_thresholds = evaluate_probabilities(
        pooled_oof_predictions, thresholds=None
    )
    ensemble_predictions = _ensemble_test_predictions(test_predictions)
    test_metrics, _ = evaluate_probabilities(
        ensemble_predictions, thresholds=pooled_oof_thresholds
    )
    fold_summaries = [
        _metric_summary(result["validation_metrics"]) for result in fold_results
    ]
    aggregate_metrics = {
        "schema_version": "absa-kfold-aggregate-metrics/1.0.0",
        "folds": [
            {
                "fold": result["fold"],
                "best_epoch": None,
                "best_validation_end_to_end_macro_f1": result[
                    "best_validation_end_to_end_macro_f1"
                ],
                "validation_metrics": fold_summaries[index],
            }
            for index, result in enumerate(fold_results)
        ],
        "cross_fold_mean_std": _aggregate_fold_metrics(fold_summaries),
        "pooled_oof_metrics": _metric_summary(pooled_oof_metrics),
    }
    _write_json(output_dir / "aggregate_metrics.json", aggregate_metrics)
    _write_json(output_dir / "oof_metrics.json", pooled_oof_metrics)
    _write_json(output_dir / "test_metrics.json", test_metrics)
    _write_json(
        output_dir / "thresholds.json",
        {
            **pooled_oof_thresholds.as_dict(),
            "selected_on": "pooled_oof",
            "folds": folds,
        },
    )
    checkpoints = []
    for result in fold_results:
        checkpoint_metadata = dict(result["checkpoint"])
        checkpoint_metadata["fold"] = result["fold"]
        checkpoint_metadata["path"] = (
            f"folds/fold_{result['fold']:02d}/{spec.checkpoint_filename}"
        )
        checkpoints.append(checkpoint_metadata)
    final = {
        **run_metadata,
        "status": "COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "fold_results": aggregate_metrics["folds"],
        "cross_fold_mean_std": aggregate_metrics["cross_fold_mean_std"],
        "pooled_oof_metrics": pooled_oof_metrics,
        "test_metrics": test_metrics,
        "checkpoints": checkpoints,
        "metric_exports": {
            "fold_metrics": "fold_metrics.csv",
            "epoch_metrics": "epoch_metrics.csv",
        },
    }
    _write_json(output_dir / "run.json", final)
    write_run_metric_exports(output_dir, model_name)
    seal_kfold_training_run(output_dir)
    final["artifact_validation"] = validate_kfold_training_run(output_dir)
    return final


def train_benchmark_suite(
    *,
    data_release: Path,
    results_dir: Path,
    config_path: Path,
    suite_id: str,
    models: Sequence[str] = MODEL_NAMES,
    device_name: str | None = None,
    folds_override: int | None = None,
    max_development_samples: int | None = None,
    max_test_samples: int | None = None,
    max_epochs_override: int | None = None,
    max_length_override: int | None = None,
    batch_size_override: int | None = None,
    gradient_accumulation_override: int | None = None,
    show_progress_override: bool | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run selected models sequentially and publish a comparison table."""

    selected = list(dict.fromkeys(str(model) for model in models))
    if not selected:
        raise ValueError("at least one model must be selected")
    for model_name in selected:
        get_model_spec(model_name)
    results_dir = results_dir.resolve()
    completed: list[dict[str, Any]] = []
    assignment_sha256: str | None = None
    for model_name in selected:
        spec = get_model_spec(model_name)
        run_dir = results_dir / model_name / suite_id
        if run_dir.exists() and resume:
            validate_kfold_training_run(run_dir)
            result = read_json(run_dir / "run.json")
            if result.get("model_name") != model_name:
                raise ValueError(f"resume model identity mismatch: {run_dir}")
        elif spec.iterative:
            result = train_kfold_model(
                model_name=model_name,
                data_release=data_release,
                output_dir=run_dir,
                config_path=config_path,
                device_name=device_name,
                folds_override=folds_override,
                max_development_samples=max_development_samples,
                max_test_samples=max_test_samples,
                max_epochs_override=max_epochs_override,
                max_length_override=max_length_override,
                batch_size_override=batch_size_override,
                gradient_accumulation_override=gradient_accumulation_override,
                show_progress_override=show_progress_override,
            )
        else:
            result = train_classical_kfold_model(
                model_name=model_name,
                data_release=data_release,
                output_dir=run_dir,
                config_path=config_path,
                folds_override=folds_override,
                max_development_samples=max_development_samples,
                max_test_samples=max_test_samples,
                show_progress_override=show_progress_override,
            )
        current_assignment_sha256 = sha256_file(
            run_dir / "fold_assignments.jsonl"
        )
        if assignment_sha256 is None:
            assignment_sha256 = current_assignment_sha256
        elif current_assignment_sha256 != assignment_sha256:
            raise RuntimeError(
                "model runs do not use identical leakage-group fold assignments"
            )
        completed.append(
            {
                **result,
                "run_dir": str(run_dir),
            }
        )
    comparison_dir = results_dir / "comparisons" / suite_id
    if comparison_dir.exists() and resume:
        validation = validate_suite_comparison(comparison_dir)
        if set(validation["models"]) != set(selected):
            raise ValueError("existing comparison model set does not match resume request")
        comparison = read_json(comparison_dir / "all_models_comparison.json")
    else:
        comparison = write_suite_comparison(
            comparison_dir, suite_id=suite_id, runs=completed
        )
    _emit_console_event(
        "benchmark_completed",
        suite_id=suite_id,
        models=comparison["models"],
        comparison_dir=str(comparison_dir),
    )
    return {
        "status": "COMPLETED",
        "suite_id": suite_id,
        "models": selected,
        "fold_assignments_sha256": assignment_sha256,
        "runs": [run["run_dir"] for run in completed],
        "comparison_dir": str(comparison_dir),
        "comparison": comparison,
    }
