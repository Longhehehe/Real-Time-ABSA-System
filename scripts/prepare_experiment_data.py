#!/usr/bin/env python3
"""Prepare and audit all external benchmark profiles.

Raw third-party data stays outside Git. The normalized JSONL cache and source
checksums are written under .experiment_cache/processed by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import prepare_profile
from experiments.profiles import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    load_prepared_profile,
    load_registry,
    resolve_profile_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare official ABSA benchmark profiles")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["all"],
        help="Profile IDs or 'all'",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CACHE_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = load_registry(args.config)
    profile_ids = resolve_profile_ids(args.profiles, registry)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audits = {}

    for profile_id in profile_ids:
        print(f"\n{'=' * 72}\nPreparing {profile_id}\n{'=' * 72}")
        audit = prepare_profile(
            profile_id=profile_id,
            config=registry["profiles"][profile_id],
            output_root=output_root,
            project_root=ROOT,
        )
        # Reload from disk to validate the exact cache consumed by training.
        prepared = load_prepared_profile(profile_id, output_root, args.config)
        audits[profile_id] = audit
        print(
            f"train={len(prepared.train.texts)} | dev={len(prepared.dev.texts)} | "
            f"test={len(prepared.test.texts)} | aspects={len(prepared.aspects)}"
        )
        print(
            "removed train leakage:",
            audit["removed_train_rows_due_to_dev_test_overlap"],
        )

    manifest = {
        "schema_version": 1,
        "profiles": {
            profile_id: {
                "task": audits[profile_id]["task"],
                "raw_train_rows": audits[profile_id]["raw_train_rows"],
                "clean_train_rows": audits[profile_id]["clean_train_rows"],
                "source_manifest": audits[profile_id]["source_manifest"],
            }
            for profile_id in profile_ids
        },
    }
    manifest_path = output_root / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPrepared cache: {output_root}")
    print(f"Source manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
