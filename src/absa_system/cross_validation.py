"""Leakage-controlled K-fold training with early stopping and OOF calibration."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import read_json, sha256_file, validate_model_ready_release
from .dataset import ABSADataset, collate_absa, load_model_records
from .folds import build_stratified_group_folds, validate_fold_assignments
from .losses import compute_absa_loss
from .metrics import Thresholds
from .model_registry import (
    build_neural_model,
    effective_neural_config,
    get_model_spec,
)
from .results import write_run_metric_exports
from .schema import ASPECTS, POLARITIES
from .tokenization import load_offset_tokenizer
from .training import (
    _append_jsonl,
    _artifact_entry,
    _atomic_torch_save,
    _autocast,
    _class_weights,
    _emit_console_event,
    _linear_warmup_decay,
    _metric_summary,
    _move_batch,
    _optimizer,
    _TQDM_BAR_FORMAT,
    _write_json,
    collect_probabilities,
    evaluate_probabilities,
    load_checkpoint,
    resolve_device,
    set_seed,
)


def _deterministic_limit(
    records: Sequence[Mapping[str, Any]],
    limit: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rows = [dict(record) for record in records]
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("sample limits must be positive")

    def sample_key(record: Mapping[str, Any]) -> tuple[str, str]:
        sample_id = str(record["sample_id"])
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
        return digest, sample_id

    return sorted(rows, key=sample_key)[:limit]


def _write_fold_assignments(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda row: str(row["sample_id"])):
            group_id = str(record["leakage_group_id"])
            payload = {
                "sample_id": str(record["sample_id"]),
                "leakage_group_id": group_id,
                "original_split": str(record["split"]),
                "fold": int(assignments[group_id]) + 1,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _concatenate_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot concatenate an empty prediction list")
    return {
        "mention_true": np.concatenate(
            [np.asarray(row["mention_true"]) for row in predictions], axis=0
        ),
        "sentiment_true": np.concatenate(
            [np.asarray(row["sentiment_true"]) for row in predictions], axis=0
        ),
        "mention_prob": np.concatenate(
            [np.asarray(row["mention_prob"]) for row in predictions], axis=0
        ),
        "sentiment_prob": np.concatenate(
            [np.asarray(row["sentiment_prob"]) for row in predictions], axis=0
        ),
        "sample_ids": [
            sample_id
            for row in predictions
            for sample_id in row["sample_ids"]
        ],
        "leakage_group_ids": [
            group_id
            for row in predictions
            for group_id in row["leakage_group_ids"]
        ],
    }


def _ensemble_test_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot ensemble an empty prediction list")
    reference = predictions[0]
    reference_ids = list(reference["sample_ids"])
    for fold_predictions in predictions[1:]:
        if list(fold_predictions["sample_ids"]) != reference_ids:
            raise ValueError("test sample order differs between folds")
        if not np.array_equal(
            fold_predictions["mention_true"], reference["mention_true"]
        ) or not np.array_equal(
            fold_predictions["sentiment_true"], reference["sentiment_true"]
        ):
            raise ValueError("test targets differ between folds")
    return {
        "mention_true": np.asarray(reference["mention_true"]),
        "sentiment_true": np.asarray(reference["sentiment_true"]),
        "mention_prob": np.mean(
            [np.asarray(row["mention_prob"]) for row in predictions], axis=0
        ),
        "sentiment_prob": np.mean(
            [np.asarray(row["sentiment_prob"]) for row in predictions], axis=0
        ),
        "sample_ids": reference_ids,
        "leakage_group_ids": list(reference["leakage_group_ids"]),
    }


def _scalar_fold_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    metrics = {
        "end_to_end_macro_f1": float(summary["end_to_end_macro_f1"]),
        "end_to_end_micro_f1": float(summary["end_to_end_micro"]["f1"]),
        "mention_macro_f1": float(summary["mention_macro_f1"]),
        "mention_micro_f1": float(summary["mention_micro_f1"]),
        "exact_set_match": float(summary["exact_set_match"]),
        "sample_jaccard": float(summary["sample_jaccard"]),
        "hamming_loss": float(summary["hamming_loss"]),
        "mixed_f1": float(summary["mixed"]["f1"]),
    }
    for polarity, value in summary["polarity_macro_f1"].items():
        metrics[f"polarity_{polarity}_macro_f1"] = float(value)
    return metrics


def _aggregate_fold_metrics(
    fold_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scalar_rows = [_scalar_fold_metrics(summary) for summary in fold_summaries]
    names = sorted(scalar_rows[0])
    result: dict[str, Any] = {}
    for name in names:
        values = np.asarray([row[name] for row in scalar_rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "values": values.tolist(),
        }
    return result


def _train_one_fold(
    *,
    model_name: str,
    fold_index: int,
    folds: int,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config: Mapping[str, Any],
    device: torch.device,
    amp: bool,
    show_progress: bool,
    output_dir: Path,
    data_release_id: str,
    data_manifest_sha256: str,
) -> dict[str, Any]:
    model_spec = get_model_spec(model_name)
    if not model_spec.iterative:
        raise ValueError(f"{model_name} is not a neural iterative model")
    checkpoint_filename = model_spec.checkpoint_filename
    fold_number = fold_index + 1
    fold_seed = int(config["seed"]) + fold_index * int(
        config.get("fold_seed_stride", 1009)
    )
    set_seed(fold_seed)
    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    include_evidence = bool(config.get("evidence_weight", 0.0) > 0)
    max_length = int(config["max_length"])
    batch_size = int(config["batch_size"])
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    num_workers = int(config.get("num_workers", 0))
    train_dataset = ABSADataset(
        train_records,
        tokenizer,
        max_length=max_length,
        include_evidence=include_evidence,
    )
    validation_dataset = ABSADataset(
        validation_records,
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
    generator = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_absa,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
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

    model = build_neural_model(
        model_name,
        tokenizer=tokenizer,
        config=config,
    )
    tokenizer_vocab_size = getattr(tokenizer, "model_vocab_size", None)
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        encoder_vocab_size = int(encoder.get_input_embeddings().num_embeddings)
    else:
        encoder_vocab_size = int(getattr(model, "vocab_size"))
    if (
        tokenizer_vocab_size is not None
        and int(tokenizer_vocab_size) > encoder_vocab_size
    ):
        raise RuntimeError(
            "tokenizer/model vocabulary mismatch: "
            f"tokenizer={int(tokenizer_vocab_size)}, encoder={encoder_vocab_size}"
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
    max_epochs = int(config["max_epochs"])
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    scheduler = _linear_warmup_decay(
        optimizer,
        total_steps=updates_per_epoch * max_epochs,
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp and device.type == "cuda",
    )
    mention_pos_weight, sentiment_pos_weight = _class_weights(train_records)
    mention_pos_weight = mention_pos_weight.to(device)
    sentiment_pos_weight = sentiment_pos_weight.to(device)
    fold_metadata: dict[str, Any] = {
        "schema_version": "absa-kfold-fold/1.0.0",
        "status": "RUNNING",
        "model_name": model_name,
        "model_family": model_spec.family,
        "fold": fold_number,
        "folds": folds,
        "seed": fold_seed,
        "started_at": started_at,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "test_records_for_blinded_ensemble_prediction": len(test_records),
        "train_groups": len(
            {str(row["leakage_group_id"]) for row in train_records}
        ),
        "validation_groups": len(
            {str(row["leakage_group_id"]) for row in validation_records}
        ),
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
    }
    _write_json(output_dir / "fold.json", fold_metadata)
    _emit_console_event(
        "fold_started",
        model=model_name,
        fold=fold_number,
        folds=folds,
        seed=fold_seed,
        train_records=len(train_records),
        validation_records=len(validation_records),
        max_epochs=max_epochs,
        patience=int(config.get("patience", 3)),
    )

    best_metric = -1.0
    best_epoch = 0
    best_thresholds: Thresholds | None = None
    best_validation_metrics: dict[str, Any] | None = None
    patience = int(config.get("patience", 3))
    epochs_without_improvement = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses: dict[str, float] = defaultdict(float)
        batches = 0
        train_progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"[F{fold_number}/{folds} E{epoch}/{max_epochs}] train",
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
                    sentiment_weight=float(config.get("sentiment_weight", 5.0)),
                    evidence_weight=float(config.get("evidence_weight", 0.2)),
                    consistency_weight=float(config.get("consistency_weight", 0.1)),
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

        validation_predictions = collect_probabilities(
            model,
            validation_loader,
            device=device,
            amp=amp,
            show_progress=show_progress,
            description=(
                f"[F{fold_number}/{folds} E{epoch}/{max_epochs}] validate"
            ),
        )
        validation_metrics, thresholds = evaluate_probabilities(
            validation_predictions,
            thresholds=None,
        )
        primary = float(validation_metrics["end_to_end_macro_f1"])
        epoch_log = {
            "fold": fold_number,
            "epoch": epoch,
            "global_step": global_step,
            "loss": {
                name: value / max(batches, 1)
                for name, value in epoch_losses.items()
            },
            "validation_primary_metric": "end_to_end_macro_f1",
            "validation_primary_value": primary,
            "validation_metrics": validation_metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        _append_jsonl(output_dir / "epochs.jsonl", epoch_log)
        is_best = primary > best_metric
        if is_best:
            best_metric = primary
            best_epoch = epoch
            best_thresholds = thresholds
            best_validation_metrics = validation_metrics
            epochs_without_improvement = 0
            checkpoint = {
                "schema_version": "absa-checkpoint/1.0.0",
                "model_name": model_name,
                "model_family": model_spec.family,
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
                "best_validation_end_to_end_macro_f1": best_metric,
                "training_config": dict(config),
                "data_release_id": data_release_id,
                "data_manifest_sha256": data_manifest_sha256,
                "cross_validation": {
                    "fold": fold_number,
                    "folds": folds,
                    "seed": fold_seed,
                },
            }
            _atomic_torch_save(checkpoint, output_dir / checkpoint_filename)
            _write_json(
                output_dir / "thresholds.json",
                {
                    **thresholds.as_dict(),
                    "selected_on": "validation_fold",
                    "fold": fold_number,
                    "epoch": best_epoch,
                },
            )
        else:
            epochs_without_improvement += 1
        _emit_console_event(
            "fold_epoch_completed",
            fold=fold_number,
            folds=folds,
            epoch=epoch,
            max_epochs=max_epochs,
            global_step=global_step,
            train_loss=epoch_log["loss"],
            validation_metrics=_metric_summary(validation_metrics),
            is_best=is_best,
            best_epoch=best_epoch,
            best_validation_end_to_end_macro_f1=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            patience=patience,
            elapsed_seconds=epoch_log["elapsed_seconds"],
        )
        if epochs_without_improvement >= patience:
            _emit_console_event(
                "fold_early_stopping",
                fold=fold_number,
                folds=folds,
                stopped_epoch=epoch,
                patience=patience,
                best_epoch=best_epoch,
                best_validation_end_to_end_macro_f1=best_metric,
            )
            break

    if (
        best_thresholds is None
        or best_validation_metrics is None
        or not (output_dir / checkpoint_filename).is_file()
    ):
        raise RuntimeError(f"fold {fold_number} did not produce a checkpoint")
    checkpoint = load_checkpoint(output_dir / checkpoint_filename, device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    best_validation_predictions = collect_probabilities(
        model,
        validation_loader,
        device=device,
        amp=amp,
        show_progress=show_progress,
        description=f"[F{fold_number}/{folds}] best validation",
    )
    recomputed_metrics, _ = evaluate_probabilities(
        best_validation_predictions,
        thresholds=best_thresholds,
    )
    if not math.isclose(
        float(recomputed_metrics["end_to_end_macro_f1"]),
        float(best_validation_metrics["end_to_end_macro_f1"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("best checkpoint validation metric is not reproducible")
    _write_json(output_dir / "validation_metrics.json", recomputed_metrics)

    # Test probabilities are accumulated without fold-level scoring. Labels are
    # consumed only once, after the complete ensemble and OOF thresholds exist.
    test_predictions = collect_probabilities(
        model,
        test_loader,
        device=device,
        amp=amp,
        show_progress=show_progress,
        description=f"[F{fold_number}/{folds}] test probabilities",
    )
    fold_summary = _metric_summary(recomputed_metrics)
    fold_metadata.update(
        {
            "status": "COMPLETED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "best_epoch": best_epoch,
            "best_validation_end_to_end_macro_f1": best_metric,
            "validation_metrics": fold_summary,
            "checkpoint": {
                "path": checkpoint_filename,
                "bytes": (output_dir / checkpoint_filename).stat().st_size,
                "sha256": sha256_file(output_dir / checkpoint_filename),
            },
        }
    )
    _write_json(output_dir / "fold.json", fold_metadata)
    _emit_console_event(
        "fold_completed",
        model=model_name,
        fold=fold_number,
        folds=folds,
        best_epoch=best_epoch,
        validation_metrics=fold_summary,
        elapsed_seconds=fold_metadata["elapsed_seconds"],
    )
    return {
        "fold": fold_number,
        "best_epoch": best_epoch,
        "best_validation_end_to_end_macro_f1": best_metric,
        "validation_metrics": recomputed_metrics,
        "validation_predictions": best_validation_predictions,
        "test_predictions": test_predictions,
        "checkpoint": fold_metadata["checkpoint"],
        "model": model,
    }


def seal_kfold_training_run(output_dir: str | Path) -> dict[str, Any]:
    """Seal every root and fold artifact without checksum cycles."""

    root = Path(output_dir).resolve()
    required_root = {
        "aggregate_metrics.json",
        "cross_validation.json",
        "fold_assignments.jsonl",
        "oof_metrics.json",
        "run.json",
        "test_metrics.json",
        "thresholds.json",
        "training_config.json",
    }
    missing = sorted(name for name in required_root if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"incomplete K-fold run; missing={missing}")
    pre_manifest_files = sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
        ],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    run_payload = read_json(root / "run.json")
    manifest = {
        "schema_version": "absa-kfold-artifact-manifest/1.0.0",
        "status": "SEALED_KFOLD_SMOKE_RUN"
        if run_payload.get("sample_limits", {}).get("development") is not None
        or run_payload.get("sample_limits", {}).get("test") is not None
        else "SEALED_KFOLD_TRAINING_RUN",
        "run_status": run_payload.get("status"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_release_id": run_payload.get("data_release_id"),
        "data_manifest_sha256": run_payload.get("data_manifest_sha256"),
        "folds": run_payload.get("folds"),
        "artifacts": [_artifact_entry(path, root) for path in pre_manifest_files],
    }
    _write_json(root / "manifest.json", manifest)
    checksum_paths = pre_manifest_files + [root / "manifest.json"]
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(
            checksum_paths,
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def validate_kfold_training_run(output_dir: str | Path) -> dict[str, Any]:
    """Validate recursive artifact closure, hashes and completed fold states."""

    root = Path(output_dir).resolve()
    manifest_path = root / "manifest.json"
    checksum_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("K-fold run is not sealed")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "absa-kfold-artifact-manifest/1.0.0":
        raise ValueError("unsupported K-fold artifact manifest")
    expected_checksums: dict[str, str] = {}
    for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            digest, relative_path = raw_line.split("  ", 1)
            expected_checksums[relative_path] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = set(expected_checksums) | {"SHA256SUMS"}
    if actual_files != expected_files:
        raise ValueError(
            "K-fold artifact closure failed; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    for relative_path, expected_digest in expected_checksums.items():
        if sha256_file(root / relative_path) != expected_digest:
            raise ValueError(f"checksum mismatch for {relative_path}")
    manifest_artifacts = {
        str(entry["path"]): str(entry["sha256"])
        for entry in manifest.get("artifacts", [])
    }
    expected_manifest_artifacts = {
        path: digest
        for path, digest in expected_checksums.items()
        if path != "manifest.json"
    }
    if manifest_artifacts != expected_manifest_artifacts:
        raise ValueError("manifest artifact inventory is inconsistent")
    run_payload = read_json(root / "run.json")
    folds = int(run_payload.get("folds", 0))
    if run_payload.get("status") != "COMPLETED" or folds < 2:
        raise ValueError("K-fold run is not completed")
    if manifest.get("run_status") != "COMPLETED" or manifest.get("folds") != folds:
        raise ValueError("manifest and run metadata are inconsistent")
    root_thresholds = read_json(root / "thresholds.json")
    if (
        root_thresholds.get("selected_on") != "pooled_oof"
        or int(root_thresholds.get("folds", 0)) != folds
    ):
        raise ValueError("root thresholds were not selected from pooled OOF data")
    split_payload = read_json(root / "cross_validation.json")
    if (
        int(split_payload.get("folds", 0)) != folds
        or int(split_payload.get("test_development_group_overlap", -1)) != 0
    ):
        raise ValueError("cross-validation split contract is inconsistent")
    run_checkpoints = {
        int(row.get("fold", 0)): row
        for row in run_payload.get("checkpoints", [])
        if isinstance(row, Mapping)
    }
    if set(run_checkpoints) != set(range(1, folds + 1)):
        raise ValueError("run metadata does not enumerate every fold checkpoint")
    for fold_number in range(1, folds + 1):
        fold_dir = root / "folds" / f"fold_{fold_number:02d}"
        run_checkpoint = run_checkpoints[fold_number]
        checkpoint_relative_path = str(run_checkpoint.get("path", ""))
        expected_prefix = f"folds/fold_{fold_number:02d}/"
        if not checkpoint_relative_path.startswith(expected_prefix):
            raise ValueError(
                f"run metadata for fold {fold_number} checkpoint path is invalid"
            )
        checkpoint_filename = checkpoint_relative_path.removeprefix(expected_prefix)
        if not checkpoint_filename or Path(checkpoint_filename).name != checkpoint_filename:
            raise ValueError(
                f"run metadata for fold {fold_number} checkpoint filename is invalid"
            )
        required_fold_files = {
            "epochs.jsonl",
            "fold.json",
            checkpoint_filename,
            "thresholds.json",
            "validation_metrics.json",
        }
        missing_fold_files = sorted(
            name for name in required_fold_files if not (fold_dir / name).is_file()
        )
        if missing_fold_files:
            raise ValueError(
                f"fold {fold_number} is missing artifacts: {missing_fold_files}"
            )
        fold_path = fold_dir / "fold.json"
        fold_payload = read_json(fold_path)
        if (
            fold_payload.get("status") != "COMPLETED"
            or int(fold_payload.get("fold", 0)) != fold_number
        ):
            raise ValueError(f"fold {fold_number} is incomplete or inconsistent")
        fold_thresholds = read_json(fold_dir / "thresholds.json")
        if (
            fold_thresholds.get("selected_on") != "validation_fold"
            or int(fold_thresholds.get("fold", 0)) != fold_number
        ):
            raise ValueError(f"fold {fold_number} threshold provenance is invalid")
        checkpoint_digest = sha256_file(fold_dir / checkpoint_filename)
        if fold_payload.get("checkpoint", {}).get("sha256") != checkpoint_digest:
            raise ValueError(f"fold {fold_number} checkpoint digest is inconsistent")
        expected_checkpoint_path = f"{expected_prefix}{checkpoint_filename}"
        if (
            run_checkpoint.get("path") != expected_checkpoint_path
            or run_checkpoint.get("sha256") != checkpoint_digest
        ):
            raise ValueError(
                f"run metadata for fold {fold_number} checkpoint is inconsistent"
            )
    return {
        "status": "VALID",
        "run_dir": str(root),
        "manifest_status": manifest.get("status"),
        "data_release_id": manifest.get("data_release_id"),
        "folds": folds,
        "files": len(actual_files),
    }


def train_kfold_model(
    *,
    model_name: str = "phobert",
    data_release: Path,
    output_dir: Path,
    config_path: Path,
    device_name: str | None = None,
    folds_override: int | None = None,
    max_development_samples: int | None = None,
    max_test_samples: int | None = None,
    max_epochs_override: int | None = None,
    max_length_override: int | None = None,
    batch_size_override: int | None = None,
    gradient_accumulation_override: int | None = None,
    show_progress_override: bool | None = None,
) -> dict[str, Any]:
    """Run group-aware K-fold CV and one locked-test ensemble evaluation."""

    model_spec = get_model_spec(model_name)
    if not model_spec.iterative:
        raise ValueError(
            f"{model_name} is a classical estimator; use the benchmark trainer"
        )
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
    runtime_overrides = {
        "k_folds": folds_override,
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
        key: value for key, value in runtime_overrides.items() if value is not None
    }
    if show_progress_override is not None:
        config["show_progress"] = bool(show_progress_override)
        config["runtime_overrides"]["show_progress"] = bool(show_progress_override)
    config = effective_neural_config(model_name, config)
    folds = int(config.get("k_folds", 5))
    if folds < 2:
        raise ValueError("k_folds must be at least 2")
    seed = int(config["seed"])
    device = resolve_device(device_name or config.get("device"))
    amp = bool(config.get("amp", True)) and device.type in {"cuda", "cpu"}
    show_progress = bool(config.get("show_progress", True))
    train_records = load_model_records(data_release, "train")
    dev_records = load_model_records(data_release, "dev")
    development_records = _deterministic_limit(
        [*train_records, *dev_records],
        max_development_samples,
        seed=seed,
    )
    test_records = _deterministic_limit(
        load_model_records(data_release, "test"),
        max_test_samples,
        seed=seed + 1,
    )
    assignments = build_stratified_group_folds(
        development_records,
        folds=folds,
        seed=seed,
    )
    assignment_validation = validate_fold_assignments(
        development_records,
        assignments,
        folds=folds,
    )
    test_groups = {str(row["leakage_group_id"]) for row in test_records}
    development_groups = set(assignments)
    if test_groups & development_groups:
        raise ValueError("locked test leakage groups overlap K-fold development pool")
    development_group_sizes = Counter(
        str(row["leakage_group_id"]) for row in development_records
    )
    largest_group_id, largest_group_records = development_group_sizes.most_common(1)[0]
    target_fold_records = len(development_records) / folds
    fold_record_counts = assignment_validation["records_by_fold"]
    fold_size_ratio = max(fold_record_counts.values()) / min(
        fold_record_counts.values()
    )
    split_warnings: list[str] = []
    if largest_group_records > target_fold_records:
        split_warnings.append(
            "The largest indivisible leakage group is larger than the target "
            "fold size; exact fold balance is impossible without weakening "
            "the leakage boundary."
        )
    if fold_size_ratio > 1.25:
        split_warnings.append(
            "Validation fold sizes differ by more than 25%; use pooled OOF "
            "metrics alongside unweighted cross-fold mean and standard deviation."
        )

    output_dir.mkdir(parents=True)
    _write_json(output_dir / "training_config.json", config)
    _write_fold_assignments(
        output_dir / "fold_assignments.jsonl",
        development_records,
        assignments,
    )
    cross_validation_payload = {
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
            "fraction_of_development": (
                largest_group_records / len(development_records)
            ),
        },
        "target_records_per_fold": target_fold_records,
        "max_to_min_fold_size_ratio": fold_size_ratio,
        "warnings": split_warnings,
        "test_development_group_overlap": 0,
    }
    _write_json(output_dir / "cross_validation.json", cross_validation_payload)
    data_manifest_sha256 = sha256_file(data_release / "manifest.json")
    local_files_only = bool(config.get("local_files_only", False))
    tokenizer = load_offset_tokenizer(
        str(model_spec.tokenizer_name or config["backbone_name"]),
        local_files_only=local_files_only,
    )
    if bool(config.get("evidence_weight", 0.0) > 0) and not getattr(
        tokenizer, "is_fast", False
    ):
        raise RuntimeError("evidence_weight > 0 requires a fast tokenizer")
    tokenizer_vocab_size = getattr(tokenizer, "model_vocab_size", None)
    run_metadata: dict[str, Any] = {
        "schema_version": "absa-kfold-training-run/1.0.0",
        "status": "RUNNING",
        "model_name": model_name,
        "model_family": model_spec.family,
        "checkpoint_filename": model_spec.checkpoint_filename,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_release": str(data_release),
        "data_release_id": data_manifest["release_id"],
        "data_manifest_sha256": data_manifest_sha256,
        "data_validation": data_validation,
        "config_sha256": sha256_file(config_path),
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
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
        "folds": folds,
        "early_stopping": {
            "enabled": True,
            "monitor": "validation_end_to_end_macro_f1",
            "mode": "max",
            "patience": int(config.get("patience", 3)),
            "max_epochs": int(config["max_epochs"]),
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
        "kfold_started",
        model=model_name,
        folds=folds,
        development_records=len(development_records),
        development_groups=len(development_groups),
        locked_test_records=len(test_records),
        device=str(device),
        progress=show_progress,
        early_stopping=run_metadata["early_stopping"],
        fold_assignment_validation=assignment_validation,
        split_warnings=split_warnings,
    )

    fold_results: list[dict[str, Any]] = []
    oof_predictions: list[Mapping[str, Any]] = []
    test_predictions: list[Mapping[str, Any]] = []
    started = time.monotonic()
    for fold_index in range(folds):
        validation_records = [
            row
            for row in development_records
            if assignments[str(row["leakage_group_id"])] == fold_index
        ]
        fold_train_records = [
            row
            for row in development_records
            if assignments[str(row["leakage_group_id"])] != fold_index
        ]
        train_groups = {
            str(row["leakage_group_id"]) for row in fold_train_records
        }
        validation_groups = {
            str(row["leakage_group_id"]) for row in validation_records
        }
        if train_groups & validation_groups:
            raise RuntimeError(f"leakage group overlap in fold {fold_index + 1}")
        fold_dir = output_dir / "folds" / f"fold_{fold_index + 1:02d}"
        result = _train_one_fold(
            model_name=model_name,
            fold_index=fold_index,
            folds=folds,
            train_records=fold_train_records,
            validation_records=validation_records,
            test_records=test_records,
            tokenizer=tokenizer,
            config=config,
            device=device,
            amp=amp,
            show_progress=show_progress,
            output_dir=fold_dir,
            data_release_id=str(data_manifest["release_id"]),
            data_manifest_sha256=data_manifest_sha256,
        )
        fold_results.append(result)
        oof_predictions.append(result["validation_predictions"])
        test_predictions.append(result["test_predictions"])
        del result["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pooled_oof_predictions = _concatenate_predictions(oof_predictions)
    if len(set(pooled_oof_predictions["sample_ids"])) != len(development_records):
        raise RuntimeError("OOF predictions do not cover development samples exactly once")
    pooled_oof_metrics, pooled_oof_thresholds = evaluate_probabilities(
        pooled_oof_predictions,
        thresholds=None,
    )
    ensemble_predictions = _ensemble_test_predictions(test_predictions)
    locked_test_metrics, _ = evaluate_probabilities(
        ensemble_predictions,
        thresholds=pooled_oof_thresholds,
    )
    fold_summaries = [
        _metric_summary(result["validation_metrics"]) for result in fold_results
    ]
    aggregate_metrics = {
        "schema_version": "absa-kfold-aggregate-metrics/1.0.0",
        "folds": [
            {
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
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
    _write_json(output_dir / "test_metrics.json", locked_test_metrics)
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
        checkpoint = dict(result["checkpoint"])
        checkpoint["fold"] = result["fold"]
        checkpoint["path"] = (
            f"folds/fold_{result['fold']:02d}/{model_spec.checkpoint_filename}"
        )
        checkpoints.append(checkpoint)
    final = {
        **run_metadata,
        "status": "COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "fold_results": aggregate_metrics["folds"],
        "cross_fold_mean_std": aggregate_metrics["cross_fold_mean_std"],
        "pooled_oof_metrics": pooled_oof_metrics,
        "test_metrics": locked_test_metrics,
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
    _emit_console_event(
        "kfold_completed",
        model=model_name,
        folds=folds,
        cross_fold_mean_std=aggregate_metrics["cross_fold_mean_std"],
        pooled_oof_metrics=_metric_summary(pooled_oof_metrics),
        locked_test_metrics=_metric_summary(locked_test_metrics),
        elapsed_seconds=final["elapsed_seconds"],
    )
    return final
