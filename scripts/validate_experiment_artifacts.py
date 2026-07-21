#!/usr/bin/env python3
"""Validate the experiment output contract without publishing files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.profiles import DEFAULT_CONFIG_PATH, load_registry, resolve_profile_ids
from experiments.publish import validate_artifact_tree
from experiments.runner import DEFAULT_ARTIFACT_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate experiment artifact schema")
    parser.add_argument("--profiles", nargs="+", default=["all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    registry = load_registry(args.config)
    profile_ids = resolve_profile_ids(args.profiles, registry)
    validated = validate_artifact_tree(
        args.artifact_root.resolve(), profile_ids, allow_partial=args.allow_partial
    )
    print(json.dumps({key: sorted(value) for key, value in validated.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
