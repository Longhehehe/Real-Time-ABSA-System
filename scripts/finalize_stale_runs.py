"""Safely finalize crawl manifests left in ``running`` after an external stop.

The script never changes JSONL data or counters.  Before changing a manifest,
it verifies that the physical product/review/rejection line counts match the
stored counters.  Run without ``--apply`` for a read-only preview.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Non-object JSON record in {path}:{line_number}"
                )
            count += 1
    return count


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finalize(
    raw_root: Path,
    *,
    apply: bool,
    crawl_ids: set[str],
) -> dict[str, Any]:
    inspected = 0
    candidates: list[dict[str, Any]] = []
    recovered_at = datetime.now(timezone.utc).isoformat()

    for manifest_path in sorted(raw_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "running":
            continue
        crawl_id = str(manifest.get("crawl_id") or manifest_path.parent.name)
        if crawl_ids and crawl_id not in crawl_ids:
            continue
        inspected += 1

        counts = manifest.get("counts") or {}
        physical = {
            "products": _jsonl_count(manifest_path.parent / "products.jsonl"),
            "reviews_written": _jsonl_count(
                manifest_path.parent / "reviews.jsonl"
            ),
            "reviews_rejected_quality": _jsonl_count(
                manifest_path.parent / "rejections.jsonl"
            ),
        }
        expected = {
            "products": int(counts.get("products") or 0),
            "reviews_written": int(counts.get("reviews_written") or 0),
            "reviews_rejected_quality": int(
                counts.get("reviews_rejected_quality") or 0
            ),
        }
        if physical != expected:
            raise ValueError(
                f"Refusing to finalize {crawl_id}: "
                f"physical={physical}, manifest={expected}"
            )

        item = {
            "crawl_id": crawl_id,
            "manifest": str(manifest_path),
            "command": manifest.get("command"),
            "checkpoint_at": manifest.get("checkpoint_at"),
            "counts": physical,
            "action": "would_finalize",
        }
        if apply:
            manifest["status"] = "interrupted_external"
            manifest["finished_at"] = (
                manifest.get("checkpoint_at")
                or manifest.get("started_at")
                or recovered_at
            )
            manifest["recovery_audit"] = {
                "recovered_at": recovered_at,
                "previous_status": "running",
                "reason": "no_active_collector_process_after_external_stop",
                "jsonl_counts_verified": True,
                "data_files_modified": False,
            }
            _atomic_json_write(manifest_path, manifest)
            item["action"] = "finalized"
        candidates.append(item)

    missing = sorted(
        crawl_ids - {item["crawl_id"] for item in candidates}
    )
    if missing:
        raise ValueError(
            "Requested crawl IDs were not running or not found: "
            + ", ".join(missing)
        )
    return {
        "raw_root": str(raw_root.resolve()),
        "apply": apply,
        "running_manifests_inspected": inspected,
        "runs": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--crawl-id", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = finalize(
        args.raw_root,
        apply=args.apply,
        crawl_ids=set(args.crawl_id),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
