"""Tabular exports and suite-level comparisons for K-fold experiments."""

from __future__ import annotations

from csv import DictWriter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

from .data import read_json, sha256_file
from .metrics import EVALUATION_PROTOCOL


SUMMARY_METRICS = (
    "polarity_macro_f1",
    "mention_macro_f1",
    "exact_set_match",
    "sample_jaccard",
    "hamming_loss",
)


def _flatten_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {name: metrics.get(name) for name in SUMMARY_METRICS}
    for prefix in ("mention_micro", "polarity_micro", "mixed"):
        nested = metrics.get(prefix, {})
        if isinstance(nested, Mapping):
            for name in ("precision", "recall", "f1", "support"):
                row[f"{prefix}_{name}"] = nested.get(name)
    return row


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_run_metric_exports(output_dir: str | Path, model_name: str) -> dict[str, str]:
    """Create human-readable CSV tables before a run is cryptographically sealed."""

    root = Path(output_dir).resolve()
    run = read_json(root / "run.json")
    folds = int(run["folds"])
    fold_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for fold_number in range(1, folds + 1):
        fold_dir = root / "folds" / f"fold_{fold_number:02d}"
        fold = read_json(fold_dir / "fold.json")
        full_metrics = read_json(fold_dir / "validation_metrics.json")
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold_number,
                "best_epoch": fold.get("best_epoch"),
                "train_records": fold.get("train_records"),
                "validation_records": fold.get("validation_records"),
                "elapsed_seconds": fold.get("elapsed_seconds"),
                **_flatten_summary(full_metrics),
            }
        )
        epoch_path = fold_dir / "epochs.jsonl"
        with epoch_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                validation = event.get("validation_metrics", {})
                losses = event.get("loss", {})
                epoch_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "epoch": event.get("epoch"),
                        "event": event.get("event", "epoch_completed"),
                        "global_step": event.get("global_step"),
                        "loss_total": (
                            losses.get("total") if isinstance(losses, Mapping) else None
                        ),
                        "validation_primary_value": event.get(
                            "validation_primary_value"
                        ),
                        "elapsed_seconds": event.get("elapsed_seconds"),
                        **(
                            _flatten_summary(validation)
                            if isinstance(validation, Mapping) and validation
                            else {}
                        ),
                    }
                )
    _write_csv(root / "fold_metrics.csv", fold_rows)
    _write_csv(root / "epoch_metrics.csv", epoch_rows)
    return {
        "fold_metrics": "fold_metrics.csv",
        "epoch_metrics": "epoch_metrics.csv",
    }


def write_suite_comparison(
    output_dir: str | Path,
    *,
    suite_id: str,
    runs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish JSON and CSV rankings from completed model runs."""

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for run in runs:
        test_metrics = run["test_metrics"]
        oof_metrics = run["pooled_oof_metrics"]
        rows.append(
            {
                "model": run["model_name"],
                "family": run["model_family"],
                "folds": run["folds"],
                "run_dir": run["run_dir"],
                "oof_polarity_macro_f1": oof_metrics["polarity_macro_f1"],
                "test_polarity_macro_f1": test_metrics["polarity_macro_f1"],
                "test_mention_macro_f1": test_metrics["mention_macro_f1"],
                "test_exact_set_match": test_metrics["exact_set_match"],
                "test_sample_jaccard": test_metrics["sample_jaccard"],
                "test_hamming_loss": test_metrics["hamming_loss"],
            }
        )
    rows.sort(
        key=lambda row: (-float(row["test_polarity_macro_f1"]), row["model"])
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    payload = {
        "schema_version": "absa-model-comparison/2.0.0",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "suite_id": suite_id,
        "ranking_metric": "locked_test_polarity_macro_f1",
        "models": rows,
    }
    with (root / "all_models_comparison.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    _write_csv(root / "all_models_comparison.csv", rows)
    artifact_names = ["all_models_comparison.csv", "all_models_comparison.json"]
    manifest = {
        "schema_version": "absa-model-comparison-manifest/1.0.0",
        "status": "SEALED_MODEL_COMPARISON",
        "suite_id": suite_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": [row["model"] for row in rows],
        "artifacts": [
            {
                "path": name,
                "bytes": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in artifact_names
        ],
    }
    with (root / "suite_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    checksum_names = [*artifact_names, "suite_manifest.json"]
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in checksum_names),
        encoding="utf-8",
        newline="\n",
    )
    return payload


def validate_suite_comparison(output_dir: str | Path) -> dict[str, Any]:
    """Validate the sealed four-file comparison artifact."""

    root = Path(output_dir).resolve()
    required = {
        "all_models_comparison.csv",
        "all_models_comparison.json",
        "suite_manifest.json",
        "SHA256SUMS",
    }
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != required:
        raise ValueError(
            f"comparison closure mismatch; missing={sorted(required - actual)}, "
            f"unexpected={sorted(actual - required)}"
        )
    expected: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name or name == "SHA256SUMS":
            raise ValueError("invalid comparison checksum path")
        expected[name] = digest
    if set(expected) != required - {"SHA256SUMS"}:
        raise ValueError("comparison checksum inventory mismatch")
    for name, digest in expected.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"comparison checksum mismatch: {name}")
    manifest = read_json(root / "suite_manifest.json")
    comparison = read_json(root / "all_models_comparison.json")
    if (
        comparison.get("schema_version") != "absa-model-comparison/2.0.0"
        or comparison.get("evaluation_protocol") != EVALUATION_PROTOCOL
    ):
        raise ValueError("comparison uses an unsupported evaluation protocol")
    if (
        manifest.get("status") != "SEALED_MODEL_COMPARISON"
        or manifest.get("suite_id") != comparison.get("suite_id")
    ):
        raise ValueError("comparison manifest identity mismatch")
    return {
        "status": "VALID",
        "suite_id": comparison["suite_id"],
        "models": [row["model"] for row in comparison["models"]],
        "files": len(actual),
    }
