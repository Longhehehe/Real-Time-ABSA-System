"""Compatibility entry point for the cookie-authenticated scale collector.

The former Colab script called Lazada's review endpoint directly, sampled a
fixed quota from each star rating, and wrote the complete API payload.  This
entry point keeps its efficient 50-review API transport while routing records
through the maintained collector's natural sampling, quality filtering,
cross-run deduplication, privacy controls, manifests, and checkpoints.

Run from the repository root:

    .\\.venv\\Scripts\\python .\\src\\crawl.py --target-reviews 30000
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from lazada_collector.cli import main as collector_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COOKIE_FILE = PROJECT_ROOT / "src" / "cookies.txt"


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    os.chdir(PROJECT_ROOT)
    defaults = [
        "crawl-scale",
        "--transport",
        "requests",
        "--cookie-file",
        str(COOKIE_FILE),
        "--review-pages-per-product",
        "5",
        "--max-cooldowns",
        "0",
    ]
    if not any(
        argument == "--target-reviews"
        or argument.startswith("--target-reviews=")
        for argument in arguments
    ):
        arguments = ["--target-reviews", "30000", *arguments]
    return collector_main([*defaults, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
