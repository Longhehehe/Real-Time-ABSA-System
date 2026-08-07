"""Single training entrypoint for local, server, and Kaggle runs.

Examples:
    python train.py --models logistic_regression naive_bayes --device cpu
    python train.py --models bilstm cnn_bilstm --device cuda
    python train.py --models xlm_roberta phobert --device cuda
    python train.py --device cuda  # all models, lightest to heaviest

Selected models always run in the canonical light-to-heavy order, regardless
of the order supplied to ``--models``.  PhoBERT is always last when selected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
IS_KAGGLE = KAGGLE_INPUT.is_dir() and KAGGLE_WORKING.is_dir()
if IS_KAGGLE:
    os.environ.setdefault("HF_HOME", str(KAGGLE_WORKING / "huggingface"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from absa_system.benchmark import train_benchmark_suite  # noqa: E402
from absa_system.data import validate_model_ready_release  # noqa: E402
from absa_system.results import validate_suite_comparison  # noqa: E402


MODEL_ORDER = (
    "logistic_regression",
    "naive_bayes",
    "bilstm",
    "cnn_bilstm",
    "xlm_roberta",
    "phobert",
)
DATA_RELEASE_NAME = "absa_pseudo_v3_legacy_clean_20260806"
DEFAULT_DATA = Path("data/model_ready") / DATA_RELEASE_NAME
DEFAULT_CONFIG = Path("configs/training_v1.json")


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def discover_data(explicit: Path | None) -> Path:
    if explicit is not None:
        data = resolve(explicit)
        if not data.is_dir():
            raise FileNotFoundError(f"Dataset does not exist: {data}")
        return data
    local = resolve(DEFAULT_DATA)
    if local.is_dir():
        return local
    if IS_KAGGLE:
        matches = sorted(
            path.parent
            for path in KAGGLE_INPUT.rglob(
                f"{DATA_RELEASE_NAME}/manifest.json"
            )
        )
        if len(matches) == 1:
            return matches[0].resolve()
        if not matches:
            raise FileNotFoundError(
                "Dataset v3 not found under /kaggle/input; attach it or pass --data."
            )
        raise ValueError(
            "Multiple v3 datasets found; select one with --data: "
            + ", ".join(str(path) for path in matches)
        )
    raise FileNotFoundError(f"Dataset does not exist: {local}")


def results_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return resolve(explicit)
    if IS_KAGGLE:
        return (KAGGLE_WORKING / "absa_results").resolve()
    return resolve(Path("results"))


def ordered_models(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(MODEL_ORDER)
    requested_set = set(requested)
    return [model for model in MODEL_ORDER if model in requested_set]


def check_existing_comparison(
    output_root: Path,
    run_id: str,
    models: list[str],
    *,
    resume: bool,
) -> dict[str, Any] | None:
    comparison = output_root / "comparisons" / run_id
    if not comparison.exists():
        return None
    if not resume:
        raise FileExistsError(
            f"Comparison already exists with --no-resume: {comparison}"
        )
    validation = validate_suite_comparison(comparison)
    if set(validation["models"]) != set(models):
        raise ValueError(
            "Existing comparison has a different model set; use a new --run-id."
        )
    return validation


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Train selected ABSA models with one shared 3-fold protocol."
    )
    value.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=None,
        help="Models to train. Default: all models in light-to-heavy order.",
    )
    value.add_argument("--data", type=Path, default=None)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--results-dir", type=Path, default=None)
    value.add_argument(
        "--run-id",
        default=None,
        help="Default: full_v3_YYYYMMDD (UTC).",
    )
    value.add_argument("--folds", type=int, default=3)
    value.add_argument("--device", default="cuda")
    value.add_argument("--max-development-samples", type=int)
    value.add_argument("--max-test-samples", type=int)
    value.add_argument("--max-epochs", type=int)
    value.add_argument("--max-length", type=int)
    value.add_argument("--batch-size", type=int)
    value.add_argument("--gradient-accumulation-steps", type=int)
    value.add_argument(
        "--multi-gpu",
        action="store_true",
        help=(
            "Use CUDA GPUs 0 and 1 with DataParallel and AMP for selected "
            "PhoBERT/XLM-RoBERTa runs."
        ),
    )
    value.add_argument("--no-progress", action="store_true")
    value.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume sealed model runs; enabled by default.",
    )
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    models = ordered_models(args.models)
    data = discover_data(args.data)
    config = resolve(args.config)
    output_root = results_dir(args.results_dir)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "full_v3_%Y%m%d"
    )
    if not config.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config}")
    data_validation = validate_model_ready_release(data)
    comparison_validation = check_existing_comparison(
        output_root,
        run_id,
        models,
        resume=bool(args.resume),
    )
    plan = {
        "status": "DRY_RUN" if args.dry_run else "STARTING",
        "run_id": run_id,
        "models": models,
        "data": str(data),
        "data_validation": data_validation,
        "config": str(config),
        "results_dir": str(output_root),
        "folds": args.folds,
        "device": args.device,
        "multi_gpu": bool(args.multi_gpu),
        "amp": "forced_on_for_transformers" if args.multi_gpu else "from_config",
        "resume": bool(args.resume),
        "kaggle": IS_KAGGLE,
        "existing_comparison": comparison_validation,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    result = train_benchmark_suite(
        data_release=data,
        results_dir=output_root,
        config_path=config,
        suite_id=run_id,
        models=models,
        device_name=args.device,
        folds_override=args.folds,
        max_development_samples=args.max_development_samples,
        max_test_samples=args.max_test_samples,
        max_epochs_override=args.max_epochs,
        max_length_override=args.max_length,
        batch_size_override=args.batch_size,
        gradient_accumulation_override=args.gradient_accumulation_steps,
        show_progress_override=(False if args.no_progress else None),
        resume=bool(args.resume),
        multi_gpu=bool(args.multi_gpu),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["suite_id"],
                "models": result["models"],
                "fold_assignments_sha256": result[
                    "fold_assignments_sha256"
                ],
                "comparison_dir": result["comparison_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
