"""Validate heavy experiment artifacts and publish a Git-friendly report subset."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
import csv
import hashlib
import json
import shutil

from .artifacts import read_json, write_json
from .modeling import MODEL_NAMES, NEURAL_MODEL_NAMES
from .profiles import PROJECT_ROOT


LIGHT_SUFFIXES = {".json", ".csv", ".png"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_artifacts(model_dir: Path, allow_partial: bool = False) -> Dict[str, Any]:
    required = [
        "config.json",
        "results.json",
        "metrics_comparison.png",
        "seed_scores.png",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise ValueError(f"{model_dir}: missing {', '.join(missing)}")
    result = read_json(model_dir / "results.json")
    if result.get("schema_version") != 2:
        raise ValueError(f"{model_dir}: unsupported results schema")
    if not allow_partial and result.get("n_runs") != 3:
        raise ValueError(f"{model_dir}: expected 3 completed seeds, got {result.get('n_runs')}")
    if result.get("task") == "acsa" and not (model_dir / "thresholds.json").exists():
        raise ValueError(f"{model_dir}: ACSA result has no thresholds.json")
    if result.get("model") in NEURAL_MODEL_NAMES and not (model_dir / "training_loss_by_epoch.png").exists():
        raise ValueError(f"{model_dir}: neural result has no training loss plot")
    if result.get("task") == "global_sentiment" and not (model_dir / "confusion_matrix.png").exists():
        raise ValueError(f"{model_dir}: global sentiment result has no confusion matrix plot")
    primary = result.get("primary_metric")
    if primary not in result.get("avg_metrics", {}):
        raise ValueError(f"{model_dir}: primary metric missing from avg_metrics")
    return result


def validate_artifact_tree(
    artifact_root: Path,
    profile_ids: Iterable[str],
    allow_partial: bool = False,
) -> Dict[str, Dict[str, Any]]:
    validated: Dict[str, Dict[str, Any]] = {}
    for profile_id in profile_ids:
        profile_dir = artifact_root / profile_id
        if not profile_dir.exists():
            raise ValueError(f"Missing profile artifacts: {profile_dir}")
        if not (profile_dir / "dataset_audit.json").exists():
            raise ValueError(f"{profile_dir}: missing dataset_audit.json")
        if not (profile_dir / "all_models_comparison.json").exists():
            raise ValueError(f"{profile_dir}: missing all_models_comparison.json")
        validated[profile_id] = {}
        for model_name in MODEL_NAMES:
            model_dir = profile_dir / model_name
            if allow_partial and not model_dir.exists():
                continue
            validated[profile_id][model_name] = validate_model_artifacts(
                model_dir, allow_partial=allow_partial
            )
        if not allow_partial and set(validated[profile_id]) != set(MODEL_NAMES):
            raise ValueError(f"{profile_dir}: incomplete six-model matrix")
    return validated


def _copy_model_light_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in LIGHT_SUFFIXES:
            shutil.copy2(path, destination / path.name)


def _write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "profile_id",
        "task",
        "model",
        "primary_metric",
        "test_mean",
        "test_std",
        "best_seed",
        "best_dev_score",
        "n_runs",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def publish_results(
    artifact_root: Path,
    report_root: Path,
    profile_ids: Iterable[str],
    allow_partial: bool = False,
) -> Dict[str, Any]:
    profile_ids = list(profile_ids)
    validated = validate_artifact_tree(artifact_root, profile_ids, allow_partial)
    report_root.mkdir(parents=True, exist_ok=True)
    acsa_rows: List[Dict[str, Any]] = []
    sentiment_rows: List[Dict[str, Any]] = []

    for profile_id, model_results in validated.items():
        source_profile = artifact_root / profile_id
        destination_profile = report_root / profile_id
        destination_profile.mkdir(parents=True, exist_ok=True)
        for filename in (
            "dataset_audit.json",
            "all_models_comparison.json",
            "all_models_metrics_comparison.png",
        ):
            source = source_profile / filename
            if source.exists():
                shutil.copy2(source, destination_profile / filename)
        for model_name, result in model_results.items():
            _copy_model_light_files(
                source_profile / model_name,
                destination_profile / model_name,
            )
            primary = result["primary_metric"]
            row = {
                "profile_id": profile_id,
                "task": result["task"],
                "model": model_name,
                "primary_metric": primary,
                "test_mean": result["avg_metrics"][primary],
                "test_std": result["avg_metrics"].get(f"{primary}_std"),
                "best_seed": result["best_seed"],
                "best_dev_score": result["best_dev_score"],
                "n_runs": result["n_runs"],
            }
            (acsa_rows if result["task"] == "acsa" else sentiment_rows).append(row)

    _write_summary(report_root / "summary_acsa.csv", acsa_rows)
    _write_summary(report_root / "summary_sentiment.csv", sentiment_rows)
    files = []
    for path in sorted(report_root.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            files.append(
                {
                    "path": path.relative_to(report_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    try:
        artifact_label = artifact_root.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        artifact_label = artifact_root.name
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": artifact_label,
        "profiles": profile_ids,
        "allow_partial": allow_partial,
        "files": files,
    }
    write_json(report_root / "run_manifest.json", manifest)
    return manifest
