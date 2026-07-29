"""Run and freeze the Q1 annotation-workbench validation suite."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "docs" / "audits"
    / "q1_annotation_workbench_validation_v1_20260728"
)
PACKAGE = (
    ROOT / "data" / "annotations"
    / "q1_human_gold_1200_v1_20260728"
)
PLAN = (
    ROOT / "docs" / "audits"
    / "q1_human_gold_sampling_plan_v1_1_20260728"
)
PYTHON = Path(sys.executable).resolve()
NODE = shutil.which("node")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| `{row['test_id']}` | `{row['status']}` | "
        f"{row['duration_seconds']:.2f}s |"
        for row in report["tests"]
    )
    return f"""# Q1 Annotation Workbench Validation v1

Status: **{report['status']}**  
UI: `{report['ui_version']}`  
Package: `{report['package_id']}`  
Reference: `{report['reference_id']}`

## Test matrix

| Test | Status | Duration |
|---|---|---:|
{rows}

## Verified behavior

- One local-only workbench exposes explicit blind, AI-review and adjudication
  modes.
- Workflow mode and input hashes are bound into browser session/export.
- Q1 A public assignment exposes 1,200 review texts with no sampling/source
  metadata.
- Autosave survives reload on the same browser profile/workflow.
- Blind, AI-review and adjudication browser fixtures pass; Q1 blind package
  passes at 1,200 items.
- No external resource host is used and mobile horizontal overflow is zero in
  recorded browser smoke tests.
- Legacy v1 drafts remain structurally valid through the backward-compatible
  Python validator.

## Limitations

- The adjudication browser test uses a synthetic A/B conflict fixture because
  real Q1 A/B FINAL exports and IAA do not yet exist.
- Browser smoke validates interaction/integrity plumbing, not semantic human
  label accuracy.
- No human annotation, IAA, adjudication outcome or dev/test split was
  produced by this validation task.
"""


def run_command(
    *,
    test_id: str,
    command: list[str],
    temporary: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    duration = time.perf_counter() - started
    log = (
        f"COMMAND: {json.dumps(command, ensure_ascii=False)}\n"
        f"RETURN_CODE: {result.returncode}\n"
        f"DURATION_SECONDS: {duration:.6f}\n\n"
        "STDOUT\n"
        f"{result.stdout}\n"
        "STDERR\n"
        f"{result.stderr}\n"
    )
    log_path = temporary / "logs" / f"{test_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log, encoding="utf-8", newline="\n")
    if result.returncode != 0:
        raise RuntimeError(
            f"Validation test failed: {test_id}; see {log_path}"
        )
    return {
        "test_id": test_id,
        "status": "PASS",
        "duration_seconds": duration,
        "command": command,
        "log_path": log_path.relative_to(temporary).as_posix(),
        "log_sha256": sha256_file(log_path),
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite validation release: {OUTPUT}"
        )
    if NODE is None:
        raise FileNotFoundError("node executable not found")
    package_manifest = json.loads(
        (PACKAGE / "manifest.json").read_text(encoding="utf-8")
    )
    plan_manifest = json.loads(
        (PLAN / "manifest.json").read_text(encoding="utf-8")
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    )
    try:
        commands = [
            (
                "python_compileall",
                [
                    str(PYTHON),
                    "-m",
                    "compileall",
                    "-q",
                    "human_annotation_ui",
                    "tests",
                ],
            ),
            (
                "javascript_syntax",
                [
                    NODE,
                    "--check",
                    "human_annotation_ui/app.js",
                ],
            ),
            (
                "annotation_core_unit",
                [
                    NODE,
                    "--test",
                    "human_annotation_ui/tests/annotation_core.test.mjs",
                ],
            ),
            (
                "workflow_contract_unit",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "tests/test_workflow_contracts.py",
                ],
            ),
            (
                "server_contract_unit",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "tests/test_annotation_server_contracts.py",
                ],
            ),
            (
                "blind_browser_fixture",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "tests/browser_smoke_human_annotation_ui.py",
                ],
            ),
            (
                "ai_review_browser_legacy_package",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "tests/browser_smoke_ai_review_ui.py",
                ],
            ),
            (
                "adjudication_browser_fixture",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "tests/browser_smoke_adjudication_ui.py",
                ],
            ),
            (
                "q1_1200_blind_browser",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "tests/browser_smoke_q1_workbench.py",
                ],
            ),
            (
                "q1_package_independent_validator",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "scripts/validate_q1_human_gold_package.py",
                ],
            ),
            (
                "q1_plan_independent_validator",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "scripts/validate_q1_human_gold_sampling_plan_v1_1.py",
                ],
            ),
            (
                "legacy_blind_draft_backward_compatibility",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "-m",
                    "human_annotation_ui.validate_export",
                    "--assignment",
                    (
                        "data/annotations/human_reference_v1_20260726/"
                        "assignments/annotator_a.assignment.json"
                    ),
                    "--export",
                    (
                        "data/annotations/human_reference_v1_20260726/"
                        "hra-a-974b86c75d3cc5a4-draft-"
                        "2026-07-26T08-54-12-069Z.json"
                    ),
                ],
            ),
            (
                "legacy_ai_review_draft_backward_compatibility",
                [
                    str(PYTHON),
                    "-X",
                    "utf8",
                    "-m",
                    "human_annotation_ui.validate_export",
                    "--assignment",
                    (
                        "data/annotations/"
                        "human_reference_ai_preannotation_v1_20260726/"
                        "human_check/ai_review.assignment.json"
                    ),
                    "--export",
                    (
                        "data/annotations/human_reference_v1_20260726/"
                        "hra-ai-review-033a7c9c9ab4eda6-draft-"
                        "2026-07-26T17-12-40-170Z.json"
                    ),
                ],
            ),
        ]
        tests = [
            run_command(
                test_id=test_id,
                command=command,
                temporary=temporary,
            )
            for test_id, command in commands
        ]

        screenshot_sources = (
            ROOT / ".tmp" / "human_annotation_ui_smoke.png",
            ROOT / ".tmp" / "human_annotation_ai_review_smoke.png",
            ROOT / ".tmp" / "q1_workbench_blind_smoke.png",
        )
        copied_screenshots = []
        for source in screenshot_sources:
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = temporary / "screenshots" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_screenshots.append(destination)

        provenance_sources = [
            ROOT / "human_annotation_ui" / "__init__.py",
            ROOT / "human_annotation_ui" / "common.py",
            ROOT / "human_annotation_ui" / "serve.py",
            ROOT / "human_annotation_ui" / "adjudication.py",
            ROOT / "human_annotation_ui" / "validate_export.py",
            ROOT / "human_annotation_ui" / "annotation_core.mjs",
            ROOT / "human_annotation_ui" / "app.js",
            ROOT / "human_annotation_ui" / "index.html",
            ROOT / "human_annotation_ui" / "styles.css",
            ROOT / "human_annotation_ui" / "start.ps1",
            ROOT / "human_annotation_ui" / "start_workbench.ps1",
            ROOT / "human_annotation_ui" / "README.md",
            ROOT / "tests" / "test_workflow_contracts.py",
            ROOT / "tests" / "test_annotation_server_contracts.py",
            ROOT / "tests" / "browser_smoke_human_annotation_ui.py",
            ROOT / "tests" / "browser_smoke_ai_review_ui.py",
            ROOT / "tests" / "browser_smoke_adjudication_ui.py",
            ROOT / "tests" / "browser_smoke_q1_workbench.py",
        ]
        copied_provenance = []
        for source in provenance_sources:
            relative = source.relative_to(ROOT)
            destination = temporary / "provenance" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_provenance.append(destination)

        created_at = datetime.now(timezone.utc).isoformat()
        report = {
            "schema_version": (
                "q1-annotation-workbench-validation/1.0.0"
            ),
            "artifact_type": "Q1_ANNOTATION_WORKBENCH_VALIDATION",
            "status": "VALID",
            "created_at": created_at,
            "ui_version": "human-absa-ui/2.0.0",
            "export_schema_version": "human-absa-export/2.0.0",
            "workspace_schema_version": "human-absa-workspace/2.0.0",
            "package_id": package_manifest["package_id"],
            "reference_id": package_manifest["reference_id"],
            "package_manifest_sha256": sha256_file(
                PACKAGE / "manifest.json"
            ),
            "plan_id": plan_manifest["plan_id"],
            "plan_manifest_sha256": sha256_file(
                PLAN / "manifest.json"
            ),
            "tests_passed": len(tests),
            "tests_failed": 0,
            "tests": tests,
            "browser_modes_exercised": [
                "BLINDED_INDEPENDENT_ANNOTATION",
                "AI_ASSISTED_HUMAN_VERIFICATION",
                "EXPERT_ADJUDICATION",
            ],
            "q1_browser_items": 1_200,
            "legacy_v1_exports_backward_compatible": True,
            "real_human_annotations_created": 0,
            "real_adjudications_created": 0,
            "limitations": [
                "Adjudication browser coverage uses a synthetic A/B conflict fixture because real Q1 A/B FINAL exports and IAA do not exist yet.",
                "Browser smoke validates UI/integrity behavior, not semantic annotation accuracy.",
                "A real second human and expert adjudicator remain external dependencies.",
            ],
            "next_dependency": (
                "Run blind Q1 assignment A and B with two humans; "
                "independently validate/freeze both FINAL exports."
            ),
        }
        report_path = temporary / "report.json"
        write_json(report_path, report)
        markdown_path = temporary / "report.md"
        markdown_path.write_text(
            render_markdown(report),
            encoding="utf-8",
            newline="\n",
        )
        artifact_paths = [
            report_path,
            markdown_path,
            *sorted(
                (temporary / "logs").glob("*.log"),
                key=lambda path: path.name,
            ),
            *copied_screenshots,
            *copied_provenance,
        ]
        artifacts = sorted(
            (artifact(path, temporary) for path in artifact_paths),
            key=lambda row: row["path"],
        )
        manifest = {
            "schema_version": (
                "q1-annotation-workbench-validation-manifest/1.0.0"
            ),
            "artifact_type": (
                "Q1_ANNOTATION_WORKBENCH_VALIDATION_RELEASE"
            ),
            "status": "VALID",
            "validation_id": (
                "q1-workbench-validation-"
                + sha256_text(
                    package_manifest["package_id"]
                    + "\0"
                    + sha256_file(report_path)
                )[:16]
            ),
            "created_at": created_at,
            "package_id": package_manifest["package_id"],
            "plan_id": plan_manifest["plan_id"],
            "report_sha256": sha256_file(report_path),
            "artifacts": artifacts,
        }
        manifest_path = temporary / "manifest.json"
        write_json(manifest_path, manifest)
        sums = {
            row["path"]: row["sha256"] for row in artifacts
        }
        sums["manifest.json"] = sha256_file(manifest_path)
        sums_path = temporary / "SHA256SUMS.txt"
        sums_path.write_text(
            "".join(
                f"{digest}  {relative}\n"
                for relative, digest in sorted(sums.items())
            ),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(OUTPUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "VALID",
        "output": str(OUTPUT),
        "validation_id": manifest["validation_id"],
        "tests_passed": len(tests),
        "manifest_sha256": sha256_file(OUTPUT / "manifest.json"),
        "checksums_sha256": sha256_file(OUTPUT / "SHA256SUMS.txt"),
        "report_sha256": sha256_file(OUTPUT / "report.json"),
    }


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
