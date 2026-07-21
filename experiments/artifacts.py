"""Artifact aggregation, plotting and lightweight result contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import math
import os
import shutil

import numpy as np


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _scalar_metrics(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(
        {
            key
            for result in seed_results
            for key, value in result["test_metrics"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    aggregated: Dict[str, Any] = {}
    for key in keys:
        values = [
            result["test_metrics"].get(key)
            for result in seed_results
            if isinstance(result["test_metrics"].get(key), (int, float))
            and not isinstance(result["test_metrics"].get(key), bool)
            and math.isfinite(float(result["test_metrics"][key]))
        ]
        if values:
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        else:
            aggregated[key] = None
            aggregated[f"{key}_std"] = None
    return aggregated


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _save_metrics_plot(metrics: Dict[str, Any], task: str, path: Path, title: str) -> None:
    plt = _get_plt()
    if plt is None:
        return
    keys = (
        [
            "mention_f1_macro",
            "sentiment_f1_macro",
            "sentiment_f1_samples",
            "end_to_end_f1_micro",
            "end_to_end_f1_macro",
            "exact_match_accuracy",
        ]
        if task == "acsa"
        else ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_micro", "f1_weighted"]
    )
    labels = [key for key in keys if isinstance(metrics.get(key), (int, float))]
    values = [float(metrics[key]) for key in labels]
    if not values:
        return
    plt.figure(figsize=(max(8, len(labels) * 1.5), 6))
    bars = plt.bar(range(len(labels)), values, color="#4C78A8")
    plt.xticks(range(len(labels)), [label.replace("_", " ") for label in labels], rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_seed_plot(
    seed_results: List[Dict[str, Any]], primary_metric: str, path: Path, title: str
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    seeds = [str(result["seed"]) for result in seed_results]
    dev = [float(result["dev_metrics"][primary_metric]) for result in seed_results]
    test = [float(result["test_metrics"][primary_metric]) for result in seed_results]
    x = np.arange(len(seeds))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, dev, width, label="Dev")
    plt.bar(x + width / 2, test, width, label="Test")
    plt.xticks(x, [f"Seed {seed}" for seed in seeds])
    plt.ylim(0, 1.0)
    plt.ylabel(primary_metric.replace("_", " "))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_loss_plot(seed_results: List[Dict[str, Any]], path: Path, title: str) -> None:
    plt = _get_plt()
    if plt is None:
        return
    losses = [result.get("training_loss", []) for result in seed_results if result.get("training_loss")]
    if not losses:
        return
    max_length = max(len(values) for values in losses)
    array = np.full((len(losses), max_length), np.nan, dtype=float)
    for idx, values in enumerate(losses):
        array[idx, : len(values)] = values
    mean = np.nanmean(array, axis=0)
    std = np.nanstd(array, axis=0)
    epochs = np.arange(1, max_length + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean, marker="o", label="Mean loss")
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, label="±1 std")
    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_confusion_matrix(matrix: List[List[int]], path: Path, title: str) -> None:
    plt = _get_plt()
    if plt is None:
        return
    values = np.asarray(matrix, dtype=int)
    plt.figure(figsize=(7, 6))
    image = plt.imshow(values, cmap="Blues")
    plt.colorbar(image)
    labels = ["NEG", "POS", "NEU"]
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for row in range(3):
        for col in range(3):
            plt.text(col, row, str(values[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def aggregate_model_runs(
    model_dir: Path,
    profile_id: str,
    task: str,
    model_name: str,
    aspects: List[str],
    requested_seeds: Iterable[int],
    dataset_stats: Dict[str, Any],
    git_commit: str,
    environment: Dict[str, Any],
) -> Dict[str, Any]:
    seed_results: List[Dict[str, Any]] = []
    for seed in requested_seeds:
        path = model_dir / "runs" / f"seed_{seed}" / "results.json"
        if path.exists():
            result = read_json(path)
            if result.get("status") == "completed":
                seed_results.append(result)
    if not seed_results:
        raise ValueError(f"No completed runs found under {model_dir}")

    primary_metric = "end_to_end_f1_micro" if task == "acsa" else "f1_macro"
    best = max(seed_results, key=lambda result: float(result["dev_metrics"][primary_metric]))
    avg_metrics = _scalar_metrics(seed_results)
    output = {
        "schema_version": 2,
        "profile_id": profile_id,
        "task": task,
        "model": model_name,
        "protocol": "official_train_dev_test",
        "seeds": [int(result["seed"]) for result in seed_results],
        "n_runs": len(seed_results),
        "dataset_stats": dataset_stats,
        "aspects": aspects,
        "sentiment_order": ["NEG", "POS", "NEU"],
        "primary_metric": primary_metric,
        "best_seed": int(best["seed"]),
        "best_dev_score": float(best["dev_metrics"][primary_metric]),
        "avg_metrics": avg_metrics,
        "seed_results": seed_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "environment": environment,
    }
    write_json(model_dir / "results.json", output)

    best_run_dir = model_dir / "runs" / f"seed_{best['seed']}"
    checkpoint_name = best.get("checkpoint_file")
    if checkpoint_name and (best_run_dir / checkpoint_name).exists():
        extension = Path(checkpoint_name).suffix
        _link_or_copy(best_run_dir / checkpoint_name, model_dir / f"best_model{extension}")
    if task == "acsa" and (best_run_dir / "thresholds.json").exists():
        shutil.copy2(best_run_dir / "thresholds.json", model_dir / "thresholds.json")

    root_config = dict(best.get("config", {}))
    root_config.update({"best_seed": int(best["seed"]), "profile_id": profile_id, "model": model_name})
    write_json(model_dir / "config.json", root_config)
    _save_metrics_plot(
        avg_metrics,
        task,
        model_dir / "metrics_comparison.png",
        f"{profile_id} — {model_name} test metrics",
    )
    _save_seed_plot(
        seed_results,
        primary_metric,
        model_dir / "seed_scores.png",
        f"{profile_id} — {model_name} seed scores",
    )
    _save_loss_plot(
        seed_results,
        model_dir / "training_loss_by_epoch.png",
        f"{profile_id} — {model_name} training loss",
    )
    if task == "global_sentiment":
        matrix = best["test_metrics"].get("confusion_matrix")
        if matrix:
            _save_confusion_matrix(
                matrix,
                model_dir / "confusion_matrix.png",
                f"{profile_id} — {model_name}",
            )
    return output


def aggregate_profile(model_results: Dict[str, Dict[str, Any]], profile_dir: Path) -> None:
    if not model_results:
        return
    task = next(iter(model_results.values()))["task"]
    primary_metric = "end_to_end_f1_micro" if task == "acsa" else "f1_macro"
    comparison = {
        "schema_version": 2,
        "profile_id": next(iter(model_results.values()))["profile_id"],
        "task": task,
        "primary_metric": primary_metric,
        "models": {
            model: {
                "best_seed": result["best_seed"],
                "best_dev_score": result["best_dev_score"],
                "avg_metrics": result["avg_metrics"],
            }
            for model, result in model_results.items()
        },
    }
    write_json(profile_dir / "all_models_comparison.json", comparison)

    plt = _get_plt()
    if plt is None:
        return
    names = list(model_results)
    values = [float(model_results[name]["avg_metrics"][primary_metric]) for name in names]
    errors = [float(model_results[name]["avg_metrics"].get(f"{primary_metric}_std", 0.0)) for name in names]
    plt.figure(figsize=(max(10, len(names) * 1.6), 6))
    bars = plt.bar(range(len(names)), values, yerr=errors, capsize=4, color="#4C78A8")
    plt.xticks(range(len(names)), [name.replace("_", " ") for name in names], rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel(primary_metric.replace("_", " "))
    plt.title(f"{comparison['profile_id']} — all models")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(profile_dir / "all_models_metrics_comparison.png", dpi=150)
    plt.close()
