"""Command-line interface for data preparation, training and inference."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence
import json

from .data import build_model_ready_release, validate_model_ready_release
from .inference import ABSAPredictor
from .training import train_model, validate_training_run


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
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "best_epoch": result["best_epoch"],
                    "best_dev_end_to_end_macro_f1": result[
                        "best_dev_end_to_end_macro_f1"
                    ],
                    "test_metrics": result["test_metrics"],
                    "output": str(_resolve(root, args.output)),
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
