"""Render and validate the living dataset-construction DOCX with Pandoc."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def render(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Protocol source does not exist: {source}")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Pandoc is required to render the protocol DOCX")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="dataset-protocol-",
        dir=output.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory) / "protocol.docx"
        subprocess.run(
            [
                pandoc,
                str(source),
                "--from=gfm",
                "--to=docx",
                "--standalone",
                "--toc",
                "--number-sections",
                "--metadata=lang:vi-VN",
                f"--output={temporary}",
            ],
            check=True,
        )
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("Pandoc output is not a valid DOCX container")
        with zipfile.ZipFile(temporary) as archive:
            required = {"[Content_Types].xml", "word/document.xml"}
            if not required.issubset(archive.namelist()):
                raise RuntimeError("DOCX is missing required OOXML entries")
        plain_text = subprocess.run(
            [pandoc, str(temporary), "--to=plain"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        if len(plain_text.strip()) < 1_000:
            raise RuntimeError("Rendered DOCX is unexpectedly short")
        temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("docs/DATASET_CONSTRUCTION_PROTOCOL.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/DATASET_CONSTRUCTION_PROTOCOL.docx"),
    )
    args = parser.parse_args()
    render(args.source, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
