"""FairFace dataset preprocessing.

Organizes the FairFace dataset into the layout expected by
``release_benchmark.datasets.fairface_dataset.FairFaceDataset``.

Expected output structure
-------------------------
<output_dir>/
├── fairface_label_train.csv
├── fairface_label_val.csv
├── train/
│   └── *.jpg
└── val/
    └── *.jpg

Source: https://github.com/joojs/fairface; https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view; https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view; https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view
"""

import argparse
import os
import shutil
import sys

import pandas as pd

REQUIRED_CSVS = [
    "fairface_label_train.csv",
    "fairface_label_val.csv",
]
EXPECTED_COLUMNS = {"file", "age", "gender", "race", "service_test"}


def organize(raw_dir: str, output_dir: str) -> None:
    """Copy CSVs and image directories into the expected layout."""
    os.makedirs(output_dir, exist_ok=True)

    for csv_name in REQUIRED_CSVS:
        src = os.path.join(raw_dir, csv_name)
        dst = os.path.join(output_dir, csv_name)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"  Copying {csv_name}")
            shutil.copy2(src, dst)

    for subdir in ("train", "val"):
        src = os.path.join(raw_dir, subdir)
        dst = os.path.join(output_dir, subdir)
        if os.path.isdir(src) and not os.path.exists(dst):
            print(f"  Linking {subdir}/ -> {dst}")
            os.symlink(os.path.abspath(src), dst)


def verify(output_dir: str) -> bool:
    """Return *True* when all expected files are in place."""
    ok = True

    for csv_name in REQUIRED_CSVS:
        path = os.path.join(output_dir, csv_name)
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            ok = False
        else:
            df = pd.read_csv(path)
            missing_cols = EXPECTED_COLUMNS - set(df.columns)
            if missing_cols:
                print(f"  [ERROR]   {csv_name} missing columns: {missing_cols}")
                ok = False
            else:
                print(f"  [OK]      {csv_name} ({len(df)} rows)")

            first_file = df["file"].iloc[0]
            img_path = os.path.join(output_dir, first_file)
            if not os.path.exists(img_path):
                print(f"  [WARN]    Sample image not found: {img_path}")
                print(
                    "            Images should be reachable via <output_dir>/<file column>"
                )
                ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the FairFace dataset.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="fairface/raw",
        help="Directory containing downloaded FairFace files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="fairface", help="Output directory."
    )
    args = parser.parse_args()

    if os.path.isdir(args.raw_dir):
        print("Organizing raw files ...")
        organize(args.raw_dir, args.output_dir)

    print("\nVerifying output structure ...")
    if verify(args.output_dir):
        print("\nFairFace preprocessing complete.")
    else:
        print(
            "\nSome files are missing.  Please check the instructions in "
            "data/README.md."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
