"""Fitzpatrick17k dataset preprocessing.

Organizes the Fitzpatrick17k dataset into the layout expected by
``release_benchmark.datasets.fitz17k_dataset.FitzDataset``.

Expected output structure
-------------------------
<output_dir>/
└── processed_fitz17k/
    ├── fitzpatrick17k.csv
    └── images/
        └── *.jpg   (skin condition images)

Source: https://github.com/mattgroh/fitzpatrick17k

Dataset notes
-------------
- We denote ``fitzpatrick_scale`` as ``skin_type`` in the CSV and remove
  those rows where ``fitzpatrick_scale == -1``.
- During dataset construction some image URLs were already invalid, so a
  few dozen images may be missing from the original csv.
- As of the date this repository was published **most of original URLs have
  expired**.  For the original images please contact the source authors at
  https://github.com/mattgroh/fitzpatrick17k
"""

import argparse
import os
import shutil
import sys

import pandas as pd

CSV_FILE = "fitz17k.csv"
REQUIRED_COLUMNS = {
    "skin_type",
    "three_partition_label",
    "nine_partition_label",
    "label",
    "file",
}


def organize(raw_dir: str, output_dir: str) -> None:
    """Copy the bundled CSV and images into the processed output directory."""
    fitz_dir = os.path.join(output_dir, "processed_fitz17k")
    img_dir = os.path.join(fitz_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # --- CSV ---
    # The loader expects "fitzpatrick17k.csv" in the processed dir.
    src_csv = os.path.join(raw_dir, CSV_FILE)
    dst_csv = os.path.join(fitz_dir, "fitzpatrick17k.csv")
    if not os.path.exists(src_csv):
        print(f"  [ERROR] Bundled CSV not found: {src_csv}")
        sys.exit(1)
    if not os.path.exists(dst_csv):
        print(f"  Copying {CSV_FILE} -> fitzpatrick17k.csv")
        shutil.copy2(src_csv, dst_csv)
    else:
        print(f"  fitzpatrick17k.csv already present, skipping copy.")

    # --- Images ---
    raw_img_dir = os.path.join(raw_dir, "images")
    if not os.path.isdir(raw_img_dir):
        print(f"  [WARN] Image directory not found: {raw_img_dir}")
        print("         Place the skin-condition images in that directory and re-run.")
        return

    imgs = [
        f
        for f in os.listdir(raw_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    copied = 0
    for fname in imgs:
        dst = os.path.join(img_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(raw_img_dir, fname), dst)
            copied += 1
    if copied:
        print(f"  Copied {copied} images from {raw_img_dir}")
    else:
        print("  All images already present.")


def verify(output_dir: str) -> bool:
    """Return *True* when all expected files are in place."""
    fitz_dir = os.path.join(output_dir, "processed_fitz17k")
    ok = True

    csv_path = os.path.join(fitz_dir, "fitzpatrick17k.csv")
    if not os.path.exists(csv_path):
        print(f"  [MISSING] {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"  [ERROR]   {CSV_FILE} missing columns: {missing_cols}")
        ok = False
    else:
        print(f"  [OK]      {csv_path} ({len(df)} rows)")

    if "skin_type" in df.columns:
        n_malignant = (df["three_partition_label"] == "malignant").sum()
        print(
            f"            skin_type distribution:\n"
            f"            {df['skin_type'].value_counts().sort_index().to_dict()}"
        )
        print(
            f"            Malignant: {n_malignant}, "
            f"Non-malignant: {len(df) - n_malignant}"
        )

    img_dir = os.path.join(fitz_dir, "images")
    if not os.path.isdir(img_dir):
        print(f"  [MISSING] {img_dir}/")
        ok = False
    else:
        n = len(
            [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        print(f"  [OK]      {img_dir}/ ({n} images)")
        if n == 0:
            print("  [WARN]    No images found.  As of this repository's release all")
            print("            original URLs have expired.  Please contact the source")
            print("            authors at https://github.com/mattgroh/fitzpatrick17k")
            ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess the Fitzpatrick17k dataset."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="fitz/raw",
        help="Directory containing the bundled fitzpatrick17k.csv "
        "and an images/ sub-directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="fitz", help="Output directory."
    )
    args = parser.parse_args()

    if os.path.isdir(args.raw_dir):
        print("Organizing raw files ...")
        organize(args.raw_dir, args.output_dir)
    else:
        print(f"  [WARN] raw_dir not found: {args.raw_dir} — skipping copy step.")

    print("\nVerifying output structure ...")
    if verify(args.output_dir):
        print("\nFitzpatrick17k preprocessing complete.")
    else:
        print(
            "\nSome files are missing.  Please check the instructions in "
            "data/README.md."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
