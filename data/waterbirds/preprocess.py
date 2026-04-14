"""Waterbirds dataset preprocessing.

Organizes the Waterbirds dataset into the layout expected by
``release_benchmark.datasets.waterbirds_dataset.WaterbirdsDataset``.

Expected output structure
-------------------------
<output_dir>/
├── metadata.csv                   (columns: split, place, y, img_filename)
└── <img_filename directories>     (images referenced in metadata.csv)

Source: https://github.com/kohpangwei/group_DRO
"""

import argparse
import os
import shutil
import sys

import pandas as pd

METADATA_FILE = "metadata.csv"
REQUIRED_COLUMNS = {"split", "place", "y", "img_filename"}


def organize(raw_dir: str, output_dir: str) -> None:
    """Copy or symlink raw files into the expected layout."""
    os.makedirs(output_dir, exist_ok=True)

    src_csv = os.path.join(raw_dir, METADATA_FILE)
    dst_csv = os.path.join(output_dir, METADATA_FILE)
    if os.path.exists(src_csv) and not os.path.exists(dst_csv):
        print(f"  Copying {METADATA_FILE}")
        shutil.copy2(src_csv, dst_csv)

    for item in os.listdir(raw_dir):
        src = os.path.join(raw_dir, item)
        dst = os.path.join(output_dir, item)
        if os.path.isdir(src) and not os.path.exists(dst):
            print(f"  Linking {item}/ -> {dst}")
            os.symlink(os.path.abspath(src), dst)


def verify(output_dir: str) -> bool:
    """Return *True* when all expected files are in place."""
    ok = True

    csv_path = os.path.join(output_dir, METADATA_FILE)
    if not os.path.exists(csv_path):
        print(f"  [MISSING] {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"  [ERROR]   {METADATA_FILE} missing columns: {missing}")
        ok = False
    else:
        print(f"  [OK]      {csv_path} ({len(df)} rows)")

    split_counts = df["split"].value_counts().to_dict()
    for split_id, name in [(0, "train"), (1, "val"), (2, "test")]:
        n = split_counts.get(split_id, 0)
        print(f"            {name}: {n} samples")

    sample_file = df["img_filename"].iloc[0]
    img_path = os.path.join(output_dir, sample_file)
    if not os.path.exists(img_path):
        print(f"  [WARN]    Sample image not found: {img_path}")
        print("            Images should be reachable at <output_dir>/<img_filename>.")
        ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the Waterbirds dataset.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="waterbirds/raw",
        help="Directory with raw Waterbirds download.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="waterbirds", help="Output directory."
    )
    args = parser.parse_args()

    if os.path.isdir(args.raw_dir):
        print("Organizing raw files ...")
        organize(args.raw_dir, args.output_dir)

    print("\nVerifying output structure ...")
    if verify(args.output_dir):
        print("\nWaterbirds preprocessing complete.")
    else:
        print(
            "\nSome files are missing.  Please check the instructions in "
            "data/README.md."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
