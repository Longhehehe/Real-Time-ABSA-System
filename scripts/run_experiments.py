#!/usr/bin/env python3
"""CLI for the 6-model × 4-profile official benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.modeling import MODEL_NAMES
from experiments.profiles import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    load_registry,
    resolve_profile_ids,
)
from experiments.runner import DEFAULT_ARTIFACT_ROOT, ExperimentRunner, RunOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official multi-profile ABSA experiments")
    parser.add_argument("--profiles", nargs="+", default=["all"])
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 64/32/32 samples and one epoch unless explicit limits are supplied",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = load_registry(args.config)
    profile_ids = resolve_profile_ids(args.profiles, registry)

    if args.smoke:
        args.epochs = 1
        args.patience = 1
        args.max_train_samples = args.max_train_samples or 64
        args.max_dev_samples = args.max_dev_samples or 32
        args.max_test_samples = args.max_test_samples or 32

    options = RunOptions(
        artifact_root=args.artifact_root,
        cache_root=args.cache_root,
        config_path=args.config,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
        max_test_samples=args.max_test_samples,
        device=args.device,
    )
    runner = ExperimentRunner(options)
    result = runner.run_matrix(
        profiles=profile_ids,
        models=args.models,
        seeds=args.seeds,
        resume=args.resume,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
