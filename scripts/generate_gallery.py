"""Script to generate visualization gallery assets for documentation and publications.

Executes TROPICOR's regional benchmark figure generation pipeline and writes
production-grade images and interactive maps to the target output directory.

Usage:
    uv run python scripts/generate_gallery.py [--output-dir local/figures]
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.regional_figures import OUTPUT_DIR, generate_all_figures  # noqa: E402


def main() -> None:
    """Parse CLI options and generate documentation gallery figures."""
    parser = argparse.ArgumentParser(
        description="Generate TROPICOR publication figure gallery."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to save generated figures (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    target_dir = args.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Generating TROPICOR gallery figures into: {target_dir}")

    generate_all_figures()
    print("[+] Figure gallery successfully generated.")


if __name__ == "__main__":
    main()
