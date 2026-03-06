#!/usr/bin/env python3
"""
MVTec AD2 data download helper.

The dataset requires accepting the MVTec licence agreement.
This script prints the download URL and expected folder structure,
then optionally extracts a downloaded archive.

Usage:
    # Print instructions only
    python scripts/download_data.py

    # Extract a manually downloaded archive
    python scripts/download_data.py --archive /path/to/mvtec_ad2.zip --dest data/
"""

import argparse
import os
import zipfile

DOWNLOAD_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad-2"

CATEGORIES = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
]

EXPECTED_STRUCTURE = """
data/
└── <category>/
    ├── train/good/
    ├── validation/good/
    └── test_public/
        ├── bad/
        ├── good/
        └── ground_truth/bad/
"""


def print_instructions() -> None:
    print("=" * 60)
    print("MVTec AD2 Dataset Download Instructions")
    print("=" * 60)
    print(f"\n1. Visit: {DOWNLOAD_URL}")
    print("2. Accept the licence agreement and download the dataset.")
    print("3. Extract the archive so the folder structure matches:")
    print(EXPECTED_STRUCTURE)
    print("4. Re-run this script with --archive to auto-extract, or")
    print("   place the data manually under data/\n")


def extract(archive: str, dest: str) -> None:
    print(f"Extracting {archive} → {dest} ...")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)
    print("Done.")

    # Verify expected categories
    missing = [c for c in CATEGORIES if not os.path.isdir(os.path.join(dest, c))]
    if missing:
        print(f"⚠  Missing categories after extraction: {missing}")
        print("   Check that the archive matches the expected structure above.")
    else:
        print("✅ All 8 categories found.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=None, help="Path to downloaded .zip archive.")
    parser.add_argument("--dest", default="data", help="Destination directory (default: data/).")
    args = parser.parse_args()

    print_instructions()
    if args.archive:
        extract(args.archive, args.dest)


if __name__ == "__main__":
    main()
