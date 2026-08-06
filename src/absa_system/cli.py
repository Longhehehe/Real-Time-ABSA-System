"""Command-line interface for data preparation, training and inference."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import json

from .cross_validation import train_kfold_model, validate_kfold_training_run
from .data import build_model_ready_release, validate_model_ready_release
from .inference import ABSAPredictor
from .model_registry import MODEL_NAMES, get_model_spec
from .results import validate_suite_comparison
from .training import _metric_summary, train_model, validate_training_run


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="absa-system",
        description="Vietnamese evidence-aware multi-polarity ABSA pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Build a versioned leakage-controlled model-ready release.",
    )
    prepare.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_data_v1.json"),
    )
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("data/model_ready/absa_pseudo_v1_2_20260729"),
    )

    validate = subparsers.add_parser(
        "validate-data",
        help="Validate checksums, schemas and group isolation.",
    )
    validate.add_argument("release", type=Path)

    validate_run = subparsers.add_parser(
        "validate-run",
        help="Validate closure and checksums of a sealed training run.",
    )
    validate_run.add_argument("run_dir", type=Path)

    validate_kfold_run = subparsers.add_parser(
        "validate-kfold-run",
        help="Validate recursive closure and checksums of a K-fold run.",
    )
    validate_kfold_run.add_argument("run_dir", type=Path)

    validate_benchmark = subparsers.add_parser(
        "validate-benchmark",
        help="Validate checksums and closure of a suite comparison directory.",
    )
    validate_benchmark.add_argument("comparison_dir", type=Path)

    train = subparsers.add_parser(
        "train",
        help="Train, select thresholds on dev and evaluate test once.",
    )
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_v1.json"),
    )
    train.add_argument("--device")
    train.add_argument("--max-train-samples", type=int)
    train.add_argument("--max-dev-samples", type=int)
    train.add_argument("--max-test-samples", type=int)
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--max-length", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--gradient-accumulation-steps", type=int)
    train.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm train/dev/test progress bars.",
    )

    train_kfold = subparsers.add_parser(
        "train-kfold",
        help=(
            "Run group-aware K-fold CV with per-fold early stopping and one "
            "locked-test ensemble evaluation."
        ),
    )
    train_kfold.add_argument("--data", type=Path, required=True)
    train_kfold.add_argument("--output", type=Path, required=True)
    train_kfold.add_argument(
        "--model",
        choices=[name for name in MODEL_NAMES if get_model_spec(name).iterative],
        default="phobert",
        help="One neural model; use train-benchmark for the complete suite.",
    )
    train_kfold.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_v1.json"),
    )
    train_kfold.add_argument("--device")
    train_kfold.add_argument("--folds", type=int)
    train_kfold.add_argument("--max-development-samples", type=int)
    train_kfold.add_argument("--max-test-samples", type=int)
    train_kfold.add_argument("--max-epochs", type=int)
    train_kfold.add_argument("--max-length", type=int)
    train_kfold.add_argument("--batch-size", type=int)
    train_kfold.add_argument("--gradient-accumulation-steps", type=int)
    train_kfold.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars but keep structured fold/metric events.",
    )

    benchmark = subparsers.add_parser(
        "train-benchmark",
        help=(
            "Train the original six-model family on identical group-aware "
            "folds and save per-model plus comparison results."
        ),
    )
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--results-dir", type=Path, default=Path("results"))
    benchmark.add_argument(
        "--run-id",
        help="Shared run ID; defaults to a UTC timestamp.",
    )
    benchmark.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_NAMES),
        default=list(MODEL_NAMES),
    )
    benchmark.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_v1.json"),
    )
    benchmark.add_argument("--device")
    benchmark.add_argument("--folds", type=int)
    benchmark.add_argument("--max-development-samples", type=int)
    benchmark.add_argument("--max-test-samples", type=int)
    benchmark.add_argument("--max-epochs", type=int)
    benchmark.add_argument("--max-length", type=int)
    benchmark.add_argument("--batch-size", type=int)
    benchmark.add_argument("--gradient-accumulation-steps", type=int)
    benchmark.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars but keep structured fold/metric events.",
    )
    benchmark.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only already completed per-model runs that pass validation.",
    )

    predict = subparsers.add_parser(
        "predict",
        help="Run checkpoint-bound inference on one or more reviews.",
    )
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--text", action="append", required=True)
    predict.add_argument("--device")
    predict.add_argument("--local-files-only", action="store_true")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _compact_cli_metrics(metrics: dict) -> dict:
    summary = _metric_summary(metrics)
    return {
        "polarity_macro_f1": summary["polarity_macro_f1"],
        "polarity_micro_f1": summary["polarity_micro"]["f1"],
        "mention_macro_f1": summary["mention_macro_f1"],
        "exact_set_match": summary["exact_set_match"],
        "sample_jaccard": summary["sample_jaccard"],
        "hamming_loss": summary["hamming_loss"],
        "mixed_f1": summary["mixed"]["f1"],
        "polarity_f1": summary["polarity_f1"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args: Namespace = build_parser().parse_args(argv)
    root = _project_root()
    if args.command == "prepare":
        manifest = build_model_ready_release(
            project_root=root,
            config_path=_resolve(root, args.config),
            output_dir=_resolve(root, args.output),
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "release_id": manifest["release_id"],
                    "output": str(_resolve(root, args.output)),
                    "counts": manifest["counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "validate-data":
        result = validate_model_ready_release(_resolve(root, args.release))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-run":
        result = validate_training_run(_resolve(root, args.run_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-kfold-run":
        result = validate_kfold_training_run(_resolve(root, args.run_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-benchmark":
        result = validate_suite_comparison(_resolve(root, args.comparison_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        result = train_model(
            data_release=_resolve(root, args.data),
            output_dir=_resolve(root, args.output),
            config_path=_resolve(root, args.config),
            device_name=args.device,
            max_train_samples=args.max_train_samples,
            max_dev_samples=args.max_dev_samples,
            max_test_samples=args.max_test_samples,
            max_epochs_override=args.max_epochs,
            max_length_override=args.max_length,
            batch_size_override=args.batch_size,
            gradient_accumulation_override=args.gradient_accumulation_steps,
            show_progress_override=(False if args.no_progress else None),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "best_epoch": result["best_epoch"],
                    "best_dev_polarity_macro_f1": result[
                        "best_dev_polarity_macro_f1"
                    ],
                    "test_metrics": _compact_cli_metrics(result["test_metrics"]),
                    "output": str(_resolve(root, args.output)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "train-kfold":
        result = train_kfold_model(
            model_name=args.model,
            data_release=_resolve(root, args.data),
            output_dir=_resolve(root, args.output),
            config_path=_resolve(root, args.config),
            device_name=args.device,
            folds_override=args.folds,
            max_development_samples=args.max_development_samples,
            max_test_samples=args.max_test_samples,
            max_epochs_override=args.max_epochs,
            max_length_override=args.max_length,
            batch_size_override=args.batch_size,
            gradient_accumulation_override=args.gradient_accumulation_steps,
            show_progress_override=(False if args.no_progress else None),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "folds": result["folds"],
                    "model": args.model,
                    "cross_fold_polarity_macro_f1": result[
                        "cross_fold_mean_std"
                    ]["polarity_macro_f1"],
                    "pooled_oof_metrics": _compact_cli_metrics(
                        result["pooled_oof_metrics"]
                    ),
                    "test_metrics": _compact_cli_metrics(result["test_metrics"]),
                    "output": str(_resolve(root, args.output)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "train-benchmark":
        from .benchmark import train_benchmark_suite

        suite_id = args.run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        result = train_benchmark_suite(
            data_release=_resolve(root, args.data),
            results_dir=_resolve(root, args.results_dir),
            config_path=_resolve(root, args.config),
            suite_id=suite_id,
            models=args.models,
            device_name=args.device,
            folds_override=args.folds,
            max_development_samples=args.max_development_samples,
            max_test_samples=args.max_test_samples,
            max_epochs_override=args.max_epochs,
            max_length_override=args.max_length,
            batch_size_override=args.batch_size,
            gradient_accumulation_override=args.gradient_accumulation_steps,
            show_progress_override=(False if args.no_progress else None),
            resume=args.resume,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "suite_id": result["suite_id"],
                    "models": result["models"],
                    "fold_assignments_sha256": result[
                        "fold_assignments_sha256"
                    ],
                    "runs": result["runs"],
                    "comparison_dir": result["comparison_dir"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "predict":
        predictor = ABSAPredictor(
            _resolve(root, args.checkpoint),
            device_name=args.device,
            local_files_only=args.local_files_only,
        )
        result = predictor.predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")
