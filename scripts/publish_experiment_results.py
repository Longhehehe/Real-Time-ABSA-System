#!/usr/bin/env python3
"""Publish JSON/CSV/PNG experiment results while excluding checkpoints and logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.profiles import DEFAULT_CONFIG_PATH, load_registry, resolve_profile_ids
from experiments.publish import publish_results
from experiments.runner import DEFAULT_ARTIFACT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish lightweight experiment reports")
    parser.add_argument("--profiles", nargs="+", default=["all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports" / "experiments")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = load_registry(args.config)
    profile_ids = resolve_profile_ids(args.profiles, registry)
    manifest = publish_results(
        args.artifact_root.resolve(),
        args.report_root.resolve(),
        profile_ids,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
