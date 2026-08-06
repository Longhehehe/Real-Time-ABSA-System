"""Reproducible training, evaluation and checkpoint handling."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math
import os
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import read_json, sha256_file, validate_model_ready_release
from .dataset import ABSADataset, collate_absa, load_model_records
from .losses import compute_absa_loss
from .metrics import (
    EVALUATION_PROTOCOL,
    Thresholds,
    apply_thresholds,
    compute_metrics,
    tune_thresholds,
)
from .model import AspectEvidenceModel
from .schema import ASPECTS, POLARITIES
from .tokenization import load_offset_tokenizer


_TQDM_BAR_FORMAT = (
    "{desc:<30} {percentage:3.0f}%|{bar:24}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _class_weights(records: Iterable[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = list(records)
    mention = np.asarray([row["mention_labels"] for row in rows], dtype=np.float64)
    sentiment = np.asarray(
        [row["sentiment_labels"] for row in rows], dtype=np.float64
    )
    mention_positive = mention.sum(axis=0)
    mention_negative = len(rows) - mention_positive
    mention_weight = mention_negative / np.maximum(mention_positive, 1.0)
    mention_weight = np.clip(mention_weight, 1.0, 20.0)

    mentioned = mention.sum(axis=0)[:, None]
    sentiment_positive = sentiment.sum(axis=0)
    sentiment_negative = mentioned - sentiment_positive
    sentiment_weight = sentiment_negative / np.maximum(sentiment_positive, 1.0)
    sentiment_weight = np.clip(sentiment_weight, 1.0, 20.0)
    return (
        torch.as_tensor(mention_weight, dtype=torch.float32),
        torch.as_tensor(sentiment_weight, dtype=torch.float32),
    )


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
    )


@torch.no_grad()
def collect_probabilities(
    model: AspectEvidenceModel,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
    show_progress: bool = False,
    description: str = "evaluate",
) -> dict[str, Any]:
    model.eval()
    mention_true: list[np.ndarray] = []
    sentiment_true: list[np.ndarray] = []
    mention_prob: list[np.ndarray] = []
    sentiment_prob: list[np.ndarray] = []
    sample_ids: list[str] = []
    leakage_group_ids: list[str] = []
    progress = tqdm(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
        bar_format=_TQDM_BAR_FORMAT,
        file=sys.stdout,
        disable=not show_progress,
    )
    for raw_batch in progress:
        batch = _move_batch(raw_batch, device)
        with _autocast(device, amp):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                content_mask=batch["content_mask"],
            )
        mention_true.append(batch["mention_labels"].cpu().numpy())
        sentiment_true.append(batch["sentiment_labels"].cpu().numpy())
        mention_prob.append(
            torch.sigmoid(output.mention_logits).float().cpu().numpy()
        )
        sentiment_prob.append(
            torch.sigmoid(output.sentiment_logits).float().cpu().numpy()
        )
        sample_ids.extend(raw_batch["sample_id"])
        leakage_group_ids.extend(raw_batch["leakage_group_id"])
    return {
        "mention_true": np.concatenate(mention_true),
        "sentiment_true": np.concatenate(sentiment_true),
        "mention_prob": np.concatenate(mention_prob),
        "sentiment_prob": np.concatenate(sentiment_prob),
        "sample_ids": sample_ids,
        "leakage_group_ids": leakage_group_ids,
    }


def evaluate_probabilities(
    predictions: Mapping[str, Any],
    *,
    thresholds: Thresholds | None,
) -> tuple[dict[str, Any], Thresholds]:
    if thresholds is None:
        thresholds = tune_thresholds(
            predictions["mention_true"],
            predictions["sentiment_true"],
            predictions["mention_prob"],
            predictions["sentiment_prob"],
        )
    mention_pred, sentiment_pred = apply_thresholds(
        predictions["mention_prob"],
        predictions["sentiment_prob"],
        thresholds,
        gate_polarity_by_mention=False,
    )
    metrics = compute_metrics(
        predictions["mention_true"],
        predictions["sentiment_true"],
        mention_pred,
        sentiment_pred,
    )
    return metrics, thresholds


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable console subset while full metrics stay in artifacts."""

    polarity_micro = metrics["polarity_micro"]
    mention_micro = metrics["mention_micro"]
    mixed = metrics["mixed"]
    per_class = metrics.get("polarity_per_class", {})
    polarity_f1 = {
        polarity: float(per_class.get(polarity, {}).get("f1", 0.0))
        for polarity in POLARITIES
    }
    return {
        "num_samples": int(metrics["num_samples"]),
        "polarity_macro_f1": float(metrics["polarity_macro_f1"]),
        "polarity_micro": {
            "precision": float(polarity_micro["precision"]),
            "recall": float(polarity_micro["recall"]),
            "f1": float(polarity_micro["f1"]),
        },
        "mention_macro_f1": float(metrics["mention_macro_f1"]),
        "mention_micro_f1": float(mention_micro["f1"]),
        "exact_set_match": float(metrics["exact_set_match"]),
        "sample_jaccard": float(metrics["sample_jaccard"]),
        "hamming_loss": float(metrics["hamming_loss"]),
        "mixed": {
            "precision": float(mixed["precision"]),
            "recall": float(mixed["recall"]),
            "f1": float(mixed["f1"]),
            "support": int(mixed["support"]),
        },
        "polarity_f1": polarity_f1,
    }


def _console_number(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _console_duration(value: Any) -> str:
    try:
        total = max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return "-"
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _console_metric_lines(
    title: str,
    metrics: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(metrics, Mapping):
        return [f"  {title}: unavailable"]
    micro = metrics.get("polarity_micro", {})
    mixed = metrics.get("mixed", {})
    polarity = metrics.get("polarity_f1", {})
    return [
        f"  {title}",
        (
            "    Polarity macro-F1: "
            f"{_console_number(metrics.get('polarity_macro_f1'))}"
            "    | Mention macro-F1: "
            f"{_console_number(metrics.get('mention_macro_f1'))}"
        ),
        (
            "    Polarity micro P/R/F1: "
            f"{_console_number(micro.get('precision'))} / "
            f"{_console_number(micro.get('recall'))} / "
            f"{_console_number(micro.get('f1'))}"
        ),
        (
            "    Polarity Exact/Jaccard/Hamming: "
            f"{_console_number(metrics.get('exact_set_match'))} / "
            f"{_console_number(metrics.get('sample_jaccard'))} / "
            f"{_console_number(metrics.get('hamming_loss'))}"
        ),
        (
            "    Polarity F1 neg/pos/neu: "
            f"{_console_number(polarity.get('negative'))} / "
            f"{_console_number(polarity.get('positive'))} / "
            f"{_console_number(polarity.get('neutral'))}"
            "    | Mixed-F1: "
            f"{_console_number(mixed.get('f1'))} (n={mixed.get('support', '-')})"
        ),
    ]


def _emit_console_event(event: str, **payload: Any) -> None:
    """Write readable, tqdm-safe progress; JSON console output remains opt-in."""

    if os.environ.get("ABSA_CONSOLE_FORMAT", "human").strip().lower() == "json":
        tqdm.write(
            json.dumps({"event": event, **payload}, ensure_ascii=False),
            file=sys.stdout,
        )
        return

    separator = "=" * 78
    lines: list[str]
    if event == "training_started":
        lines = [
            "",
            separator,
            "TRAINING STARTED",
            separator,
            (
                f"  Device: {payload.get('device')} | Epochs: {payload.get('epochs')} "
                f"| Total updates: {payload.get('total_updates')}"
            ),
            (
                f"  Records train/dev/test: {payload.get('train_records')} / "
                f"{payload.get('dev_records')} / {payload.get('test_records')}"
            ),
            (
                f"  Batches train/dev/test: {payload.get('train_batches')} / "
                f"{payload.get('dev_batches')} / {payload.get('test_batches')}"
            ),
        ]
    elif event in {"epoch_completed", "fold_epoch_completed"}:
        fold_prefix = (
            f"FOLD {payload.get('fold')}/{payload.get('folds')} | "
            if event == "fold_epoch_completed"
            else ""
        )
        epoch_total = payload.get("max_epochs", payload.get("epochs"))
        marker = " | NEW BEST" if payload.get("is_best") else ""
        losses = payload.get("train_loss", {})
        metrics = payload.get("validation_metrics", payload.get("dev_metrics"))
        lines = [
            "",
            separator,
            f"{fold_prefix}EPOCH {payload.get('epoch')}/{epoch_total} COMPLETED{marker}",
            separator,
            (
                "  Train loss total/mention/sentiment/evidence: "
                f"{_console_number(losses.get('total'))} / "
                f"{_console_number(losses.get('mention'))} / "
                f"{_console_number(losses.get('sentiment'))} / "
                f"{_console_number(losses.get('evidence'))}"
            ),
            *_console_metric_lines("Validation metrics", metrics),
            (
                f"  Early stopping: {payload.get('epochs_without_improvement', 0)}"
                f"/{payload.get('patience', '-')} without improvement | "
                f"best epoch={payload.get('best_epoch')} | best macro-F1="
                f"{_console_number(payload.get('best_validation_polarity_macro_f1', payload.get('best_dev_polarity_macro_f1')))}"
            ),
            f"  Elapsed: {_console_duration(payload.get('elapsed_seconds'))}",
        ]
    elif event in {"early_stopping", "fold_early_stopping"}:
        fold_prefix = (
            f"Fold {payload.get('fold')}/{payload.get('folds')}: "
            if event == "fold_early_stopping"
            else ""
        )
        lines = [
            "",
            f"[EARLY STOP] {fold_prefix}stopped at epoch "
            f"{payload.get('stopped_epoch', payload.get('epoch'))}; "
            f"best epoch={payload.get('best_epoch')}, best macro-F1="
            f"{_console_number(payload.get('best_validation_polarity_macro_f1', payload.get('best_dev_polarity_macro_f1')))}, "
            f"patience={payload.get('patience')}.",
        ]
    elif event == "test_completed":
        lines = [
            "",
            separator,
            f"LOCKED TEST COMPLETED | checkpoint epoch {payload.get('selected_checkpoint_epoch')}",
            separator,
            *_console_metric_lines("Test metrics", payload.get("metrics")),
        ]
    elif event == "kfold_started":
        validation = payload.get("fold_assignment_validation", {})
        early = payload.get("early_stopping", {})
        lines = [
            "",
            separator,
            f"K-FOLD RUN STARTED | model={payload.get('model', '-')} | folds={payload.get('folds')}",
            separator,
            (
                f"  Device: {payload.get('device')} | Development/test: "
                f"{payload.get('development_records')} / {payload.get('locked_test_records')}"
            ),
            (
                f"  Leakage groups: {payload.get('development_groups')} | Fold sizes: "
                f"{validation.get('records_by_fold', {})}"
            ),
            (
                f"  Early stopping: patience={early.get('patience')} | "
                f"max epochs={early.get('max_epochs')} | monitor={early.get('monitor')}"
            ),
        ]
        for warning in payload.get("split_warnings", []):
            lines.append(f"  WARNING: {warning}")
    elif event == "fold_started":
        lines = [
            "",
            separator,
            f"FOLD {payload.get('fold')}/{payload.get('folds')} STARTED | model={payload.get('model', '-')}",
            separator,
            (
                f"  Train/validation records: {payload.get('train_records')} / "
                f"{payload.get('validation_records')} | seed={payload.get('seed')}"
            ),
            (
                f"  Max epochs: {payload.get('max_epochs')} | "
                f"early-stopping patience: {payload.get('patience')}"
            ),
        ]
    elif event in {"fold_completed", "classical_fold_completed"}:
        lines = [
            "",
            f"[FOLD {payload.get('fold')}/{payload.get('folds')} COMPLETED] "
            f"model={payload.get('model', '-')} | best epoch="
            f"{payload.get('best_epoch', 'n/a')} | elapsed="
            f"{_console_duration(payload.get('elapsed_seconds'))}",
            *_console_metric_lines(
                "Validation metrics", payload.get("validation_metrics")
            ),
        ]
    elif event == "classical_kfold_started":
        lines = [
            "",
            separator,
            f"CLASSICAL K-FOLD STARTED | model={payload.get('model')} | folds={payload.get('folds')}",
            separator,
            (
                f"  Development/test records: {payload.get('development_records')} / "
                f"{payload.get('locked_test_records')}"
            ),
            "  Early stopping: not applicable (non-epoch estimator)",
        ]
    elif event == "kfold_completed":
        aggregate = payload.get("cross_fold_mean_std", {}).get(
            "polarity_macro_f1", {}
        )
        lines = [
            "",
            separator,
            f"K-FOLD RUN COMPLETED | model={payload.get('model', '-')} | folds={payload.get('folds')}",
            separator,
            (
                "  Cross-fold polarity macro-F1 mean +/- SD: "
                f"{_console_number(aggregate.get('mean'))} +/- "
                f"{_console_number(aggregate.get('std'))}"
            ),
            *_console_metric_lines("Pooled OOF metrics", payload.get("pooled_oof_metrics")),
            *_console_metric_lines("Locked test metrics", payload.get("locked_test_metrics")),
            f"  Total elapsed: {_console_duration(payload.get('elapsed_seconds'))}",
        ]
    elif event == "benchmark_completed":
        lines = [
            "",
            separator,
            f"BENCHMARK SUITE COMPLETED | run={payload.get('suite_id')}",
            separator,
            "  Rank  Model                    OOF macro-F1  Test macro-F1",
            "  ----  -----------------------  ------------  -------------",
        ]
        for row in payload.get("models", []):
            lines.append(
                f"  {int(row.get('rank', 0)):>4}  "
                f"{str(row.get('model', '-')):<23}  "
                f"{_console_number(row.get('oof_polarity_macro_f1')):>12}  "
                f"{_console_number(row.get('test_polarity_macro_f1')):>13}"
            )
        lines.append(f"  Results: {payload.get('comparison_dir')}")
    else:
        lines = ["", f"[{event}]", json.dumps(payload, ensure_ascii=False, indent=2)]
    tqdm.write("\n".join(lines), file=sys.stdout)


def _optimizer(
    model: torch.nn.Module,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> AdamW:
    backbone_parameters = []
    head_parameters = []
    encoder = getattr(model, "encoder", None)
    backbone_ids = (
        {id(parameter) for parameter in encoder.parameters()}
        if isinstance(encoder, torch.nn.Module)
        else set()
    )
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in backbone_ids:
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    groups = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            }
        )
    if head_parameters:
        groups.append(
            {
                "params": head_parameters,
                "lr": head_lr,
                "weight_decay": weight_decay,
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return AdamW(groups)


def _linear_warmup_decay(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_ratio: float,
):
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        denominator = max(total_steps - warmup_steps, 1)
        return float(remaining) / float(denominator)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _artifact_entry(path: Path, root: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            entry["records"] = sum(1 for line in stream if line.strip())
    return entry


def seal_training_run(output_dir: str | Path) -> dict[str, Any]:
    """Seal a completed run without introducing checksum cycles."""

    root = Path(output_dir).resolve()
    required_names = (
        "epochs.jsonl",
        "model.pt",
        "run.json",
        "test_metrics.json",
        "thresholds.json",
        "training_config.json",
    )
    missing = [name for name in required_names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"cannot seal incomplete training run; missing artifacts: {missing}"
        )

    run_payload = json.loads((root / "run.json").read_text(encoding="utf-8"))
    sample_limits = run_payload.get("sample_limits", {})
    is_smoke = any(value is not None for value in sample_limits.values())
    manifest = {
        "schema_version": "absa-training-artifact-manifest/1.0.0",
        "status": "SEALED_SMOKE_RUN" if is_smoke else "SEALED_TRAINING_RUN",
        "run_status": run_payload.get("status"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_release_id": run_payload.get("data_release_id"),
        "data_manifest_sha256": run_payload.get("data_manifest_sha256"),
        "checkpoint_sha256": run_payload.get("checkpoint", {}).get("sha256"),
        "artifacts": [
            _artifact_entry(root / name, root) for name in sorted(required_names)
        ],
    }
    _write_json(root / "manifest.json", manifest)

    checksum_paths = [root / name for name in required_names] + [
        root / "manifest.json"
    ]
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(checksum_paths, key=lambda item: item.name)
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def validate_training_run(output_dir: str | Path) -> dict[str, Any]:
    """Fail closed when a sealed checkpoint run is changed or incomplete."""

    root = Path(output_dir).resolve()
    manifest_path = root / "manifest.json"
    checksum_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            "training run is not sealed with manifest.json and SHA256SUMS"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = {
        "epochs.jsonl",
        "model.pt",
        "run.json",
        "test_metrics.json",
        "thresholds.json",
        "training_config.json",
        "manifest.json",
        "SHA256SUMS",
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            "training artifact closure failed; "
            f"missing={missing}, unexpected={unexpected}"
        )

    expected_checksums: dict[str, str] = {}
    for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative_path = raw_line.split("  ", 1)
        expected_checksums[relative_path] = digest
    checksum_targets = expected_files - {"SHA256SUMS"}
    if set(expected_checksums) != checksum_targets:
        raise ValueError(
            "SHA256SUMS does not enumerate the complete sealed artifact set"
        )
    for relative_path, expected_digest in expected_checksums.items():
        if sha256_file(root / relative_path) != expected_digest:
            raise ValueError(f"checksum mismatch for {relative_path}")

    run_payload = json.loads((root / "run.json").read_text(encoding="utf-8"))
    thresholds = json.loads(
        (root / "thresholds.json").read_text(encoding="utf-8")
    )
    checkpoint_digest = sha256_file(root / "model.pt")
    run_checkpoint_digest = run_payload.get("checkpoint", {}).get("sha256")
    if run_payload.get("status") != "COMPLETED":
        raise ValueError("only completed training runs can be sealed")
    if manifest.get("run_status") != "COMPLETED":
        raise ValueError("manifest status is inconsistent with run.json")
    if run_checkpoint_digest != checkpoint_digest:
        raise ValueError("checkpoint digest in run.json is inconsistent")
    if manifest.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("checkpoint digest in manifest.json is inconsistent")
    if thresholds.get("selected_on") != "dev":
        raise ValueError("decision thresholds were not selected on dev")
    if manifest.get("data_release_id") != run_payload.get("data_release_id"):
        raise ValueError("data release identity is inconsistent")

    return {
        "status": "VALID",
        "run_dir": str(root),
        "manifest_status": manifest.get("status"),
        "data_release_id": manifest.get("data_release_id"),
        "checkpoint_sha256": checkpoint_digest,
        "files": len(actual_files),
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def train_model(
    *,
    data_release: Path,
    output_dir: Path,
    config_path: Path,
    device_name: str | None = None,
    max_train_samples: int | None = None,
    max_dev_samples: int | None = None,
    max_test_samples: int | None = None,
    max_epochs_override: int | None = None,
    max_length_override: int | None = None,
    batch_size_override: int | None = None,
    gradient_accumulation_override: int | None = None,
    show_progress_override: bool | None = None,
) -> dict[str, Any]:
    """Train one deterministic run and evaluate test once at the selected epoch."""

    data_release = data_release.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite training run: {output_dir}")
    data_validation = validate_model_ready_release(data_release)
    data_manifest = read_json(data_release / "manifest.json")
    config = read_json(config_path)
    if config.get("schema_version") != "absa-training-config/1.0.0":
        raise ValueError("unsupported training config schema_version")
    runtime_overrides = {
        "max_epochs": max_epochs_override,
        "max_length": max_length_override,
        "batch_size": batch_size_override,
        "gradient_accumulation_steps": gradient_accumulation_override,
    }
    for key, value in runtime_overrides.items():
        if value is not None:
            if int(value) <= 0:
                raise ValueError(f"{key} override must be positive")
            config[key] = int(value)
    config["runtime_overrides"] = {
        key: value
        for key, value in runtime_overrides.items()
        if value is not None
    }
    if show_progress_override is not None:
        config["show_progress"] = bool(show_progress_override)
        config["runtime_overrides"]["show_progress"] = bool(
            show_progress_override
        )
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(device_name or config.get("device"))
    amp = bool(config.get("amp", True)) and device.type in {"cuda", "cpu"}
    show_progress = bool(config.get("show_progress", True))

    train_records = load_model_records(
        data_release, "train", limit=max_train_samples
    )
    dev_records = load_model_records(data_release, "dev", limit=max_dev_samples)
    test_records = load_model_records(
        data_release, "test", limit=max_test_samples
    )
    local_files_only = bool(config.get("local_files_only", False))
    tokenizer = load_offset_tokenizer(
        str(config["backbone_name"]),
        local_files_only=local_files_only,
    )
    include_evidence = bool(config.get("evidence_weight", 0.0) > 0)
    if include_evidence and not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "evidence_weight > 0 requires a fast tokenizer with offsets"
        )
    max_length = int(config["max_length"])
    train_dataset = ABSADataset(
        train_records,
        tokenizer,
        max_length=max_length,
        include_evidence=include_evidence,
    )
    dev_dataset = ABSADataset(
        dev_records,
        tokenizer,
        max_length=max_length,
        include_evidence=include_evidence,
    )
    test_dataset = ABSADataset(
        test_records,
        tokenizer,
        max_length=max_length,
        include_evidence=include_evidence,
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_absa,
        pin_memory=device.type == "cuda",
    )
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_absa,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_absa,
        pin_memory=device.type == "cuda",
    )

    model = AspectEvidenceModel(
        backbone_name=str(config["backbone_name"]),
        dropout=float(config.get("dropout", 0.2)),
        local_files_only=local_files_only,
    )
    tokenizer_vocab_size = getattr(tokenizer, "model_vocab_size", None)
    encoder_vocab_size = int(
        model.encoder.get_input_embeddings().num_embeddings
    )
    if (
        tokenizer_vocab_size is not None
        and int(tokenizer_vocab_size) > encoder_vocab_size
    ):
        raise RuntimeError(
            "tokenizer/model vocabulary mismatch: "
            f"tokenizer={int(tokenizer_vocab_size)}, "
            f"encoder={encoder_vocab_size}"
        )
    if config.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()
    model.to(device)
    optimizer = _optimizer(
        model,
        backbone_lr=float(config["backbone_lr"]),
        head_lr=float(config["head_lr"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    epochs = int(config["max_epochs"])
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    total_updates = updates_per_epoch * epochs
    scheduler = _linear_warmup_decay(
        optimizer,
        total_steps=total_updates,
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp and device.type == "cuda",
    )
    mention_pos_weight, sentiment_pos_weight = _class_weights(train_records)
    mention_pos_weight = mention_pos_weight.to(device)
    sentiment_pos_weight = sentiment_pos_weight.to(device)

    output_dir.mkdir(parents=True)
    _write_json(output_dir / "training_config.json", config)
    run_metadata = {
        "schema_version": "absa-training-run/2.0.0",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_release": str(data_release),
        "data_release_id": data_manifest["release_id"],
        "data_manifest_sha256": sha256_file(data_release / "manifest.json"),
        "data_validation": data_validation,
        "config_sha256": sha256_file(config_path),
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "transformers_version": distribution_version("transformers"),
        "tqdm_version": distribution_version("tqdm"),
        "numpy_version": np.__version__,
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "model_vocab_size": (
                int(tokenizer_vocab_size)
                if tokenizer_vocab_size is not None
                else None
            ),
            "encoder_vocab_size": encoder_vocab_size,
            "id_alignment": getattr(
                tokenizer,
                "model_vocab_id_alignment",
                "unknown",
            ),
            "remapped_vocab_entries": getattr(
                tokenizer,
                "remapped_vocab_entries",
                0,
            ),
        },
        "seed": seed,
        "sample_limits": {
            "train": max_train_samples,
            "dev": max_dev_samples,
            "test": max_test_samples,
        },
    }
    _write_json(output_dir / "run.json", run_metadata)
    _emit_console_event(
        "training_started",
        device=str(device),
        train_records=len(train_records),
        dev_records=len(dev_records),
        test_records=len(test_records),
        train_batches=len(train_loader),
        dev_batches=len(dev_loader),
        test_batches=len(test_loader),
        epochs=epochs,
        updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,
        progress=show_progress,
    )

    best_metric = -1.0
    best_epoch = 0
    best_thresholds: Thresholds | None = None
    patience = int(config.get("patience", 3))
    epochs_without_improvement = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: dict[str, float] = defaultdict(float)
        batches = 0
        train_progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"[Epoch {epoch}/{epochs}] train",
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
            bar_format=_TQDM_BAR_FORMAT,
            file=sys.stdout,
            disable=not show_progress,
        )
        for batch_index, raw_batch in enumerate(train_progress, start=1):
            batch = _move_batch(raw_batch, device)
            with _autocast(device, amp):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    content_mask=batch["content_mask"],
                )
                breakdown = compute_absa_loss(
                    output,
                    batch,
                    mention_weight=float(config.get("mention_weight", 2.0)),
                    sentiment_weight=float(
                        config.get("sentiment_weight", 5.0)
                    ),
                    evidence_weight=float(config.get("evidence_weight", 0.2)),
                    consistency_weight=float(
                        config.get("consistency_weight", 0.1)
                    ),
                    neutral_exclusivity_weight=float(
                        config.get("neutral_exclusivity_weight", 0.05)
                    ),
                    focal_gamma=float(config.get("focal_gamma", 2.0)),
                    mention_pos_weight=mention_pos_weight,
                    sentiment_pos_weight=sentiment_pos_weight,
                )
                scaled_loss = breakdown.total / accumulation
            scaler.scale(scaled_loss).backward()
            for name, value in breakdown.detached().items():
                epoch_losses[name] += value
            batches += 1
            should_update = (
                batch_index % accumulation == 0
                or batch_index == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config.get("max_grad_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            if show_progress:
                train_progress.set_postfix(
                    loss=f"{epoch_losses['total'] / batches:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    updates=global_step,
                    refresh=False,
                )

        dev_predictions = collect_probabilities(
            model,
            dev_loader,
            device=device,
            amp=amp,
            show_progress=show_progress,
            description=f"[Epoch {epoch}/{epochs}] validate",
        )
        dev_metrics, thresholds = evaluate_probabilities(
            dev_predictions,
            thresholds=None,
        )
        primary = float(dev_metrics["polarity_macro_f1"])
        epoch_log = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": {
                name: value / max(batches, 1)
                for name, value in epoch_losses.items()
            },
            "dev_primary_metric": "polarity_macro_f1",
            "dev_primary_value": primary,
            "dev_metrics": dev_metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        _append_jsonl(output_dir / "epochs.jsonl", epoch_log)
        is_best = primary > best_metric
        if is_best:
            best_metric = primary
            best_epoch = epoch
            best_thresholds = thresholds
            epochs_without_improvement = 0
            checkpoint = {
                "schema_version": "absa-checkpoint/1.0.0",
                "model_config": model.export_config(),
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "thresholds": thresholds.as_dict(),
                "aspects": list(ASPECTS),
                "polarities": list(POLARITIES),
                "max_length": max_length,
                "best_epoch": best_epoch,
                "best_dev_polarity_macro_f1": best_metric,
                "training_config": config,
                "data_release_id": data_manifest["release_id"],
                "data_manifest_sha256": run_metadata[
                    "data_manifest_sha256"
                ],
            }
            _atomic_torch_save(checkpoint, output_dir / "model.pt")
            _write_json(
                output_dir / "thresholds.json",
                {
                    **thresholds.as_dict(),
                    "selected_on": "dev",
                    "epoch": best_epoch,
                },
            )
        else:
            epochs_without_improvement += 1
        _emit_console_event(
            "epoch_completed",
            epoch=epoch,
            epochs=epochs,
            global_step=global_step,
            train_loss=epoch_log["loss"],
            dev_metrics=_metric_summary(dev_metrics),
            is_best=is_best,
            best_epoch=best_epoch,
            best_dev_polarity_macro_f1=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            patience=patience,
            elapsed_seconds=epoch_log["elapsed_seconds"],
        )
        if epochs_without_improvement >= patience:
            _emit_console_event(
                "early_stopping",
                epoch=epoch,
                patience=patience,
                best_epoch=best_epoch,
                best_dev_polarity_macro_f1=best_metric,
            )
            break

    if best_thresholds is None or not (output_dir / "model.pt").is_file():
        raise RuntimeError("training did not produce a valid checkpoint")
    checkpoint = load_checkpoint(output_dir / "model.pt", device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_predictions = collect_probabilities(
        model,
        test_loader,
        device=device,
        amp=amp,
        show_progress=show_progress,
        description="[Locked test] inference",
    )
    test_metrics, _ = evaluate_probabilities(
        test_predictions,
        thresholds=best_thresholds,
    )
    _emit_console_event(
        "test_completed",
        selected_checkpoint_epoch=best_epoch,
        metrics=_metric_summary(test_metrics),
    )
    final = {
        **run_metadata,
        "status": "COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "best_epoch": best_epoch,
        "best_dev_polarity_macro_f1": best_metric,
        "test_metrics": test_metrics,
        "checkpoint": {
            "path": "model.pt",
            "bytes": (output_dir / "model.pt").stat().st_size,
            "sha256": sha256_file(output_dir / "model.pt"),
        },
    }
    _write_json(output_dir / "run.json", final)
    _write_json(output_dir / "test_metrics.json", test_metrics)
    seal_training_run(output_dir)
    final["artifact_validation"] = validate_training_run(output_dir)
    return final


def load_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dictionary")
    if payload.get("schema_version") != "absa-checkpoint/1.0.0":
        raise ValueError("unsupported checkpoint schema_version")
    if payload.get("aspects") != list(ASPECTS):
        raise ValueError("checkpoint aspect taxonomy mismatch")
    if payload.get("polarities") != list(POLARITIES):
        raise ValueError("checkpoint polarity order mismatch")
    return payload
