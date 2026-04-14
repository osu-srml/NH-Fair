"""HAM10000 dataset preprocessing.

Organizes the HAM10000 dataset into the layout expected by
``release_benchmark.datasets.ham10000_dataset.HAM10000Dataset``.

Expected output structure
-------------------------
<output_dir>/
└── base/
    ├── HAM10000_metadata.csv
    └── HAM10000_images/
        └── *.jpg   (10 015 dermoscopic images)

Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
"""

import argparse
import glob
import os
import shutil
import sys

import pandas as pd

METADATA_FILE = "HAM10000_metadata.csv"
IMAGE_DIR = "HAM10000_images"
REQUIRED_COLUMNS = {
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
}


def organize(raw_dir: str, output_dir: str) -> None:
    """Copy metadata and images into <output_dir>/base/."""
    base_dir = os.path.join(output_dir, "base")
    os.makedirs(base_dir, exist_ok=True)

    src_csv = os.path.join(raw_dir, METADATA_FILE)
    dst_csv = os.path.join(base_dir, METADATA_FILE)
    if os.path.exists(src_csv) and not os.path.exists(dst_csv):
        print(f"  Copying {METADATA_FILE}")
        shutil.copy2(src_csv, dst_csv)

    dst_img = os.path.join(base_dir, IMAGE_DIR)
    os.makedirs(dst_img, exist_ok=True)

    src_patterns = [
        os.path.join(raw_dir, IMAGE_DIR, "*.jpg"),
        os.path.join(raw_dir, "HAM10000_images_part_1", "*.jpg"),
        os.path.join(raw_dir, "HAM10000_images_part_2", "*.jpg"),
        os.path.join(raw_dir, "*.jpg"),
    ]

    copied = 0
    for pattern in src_patterns:
        for src in glob.glob(pattern):
            fname = os.path.basename(src)
            dst = os.path.join(dst_img, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1

    if copied:
        print(f"  Copied {copied} images to {dst_img}")


def verify(output_dir: str) -> bool:
    """Return *True* when all expected files are in place."""
    base_dir = os.path.join(output_dir, "base")
    ok = True

    csv_path = os.path.join(base_dir, METADATA_FILE)
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

    df_clean = df.dropna(subset=["age", "sex"])
    n_dropped = len(df) - len(df_clean)
    if n_dropped:
        print(
            f"            {n_dropped} rows with missing age/sex "
            f"(will be dropped at training time)"
        )

    dx_counts = df["dx"].value_counts().to_dict()
    positive = sum(dx_counts.get(d, 0) for d in ("mel", "akiec"))
    negative = len(df) - positive
    print(f"            Label distribution: positive={positive}, negative={negative}")

    img_dir = os.path.join(base_dir, IMAGE_DIR)
    if not os.path.isdir(img_dir):
        print(f"  [MISSING] {img_dir}/")
        ok = False
    else:
        n_imgs = len(glob.glob(os.path.join(img_dir, "*.jpg")))
        print(f"  [OK]      {img_dir}/ ({n_imgs} images)")
        if n_imgs == 0:
            ok = False

        expected_ids = set(df["image_id"].values)
        found_ids = {
            os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")
        }
        missing_imgs = expected_ids - found_ids
        if missing_imgs:
            print(
                f"  [WARN]    {len(missing_imgs)} images referenced in "
                f"metadata are missing from {IMAGE_DIR}/"
            )
            ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the HAM10000 dataset.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="ham/raw",
        help="Directory with raw HAM10000 downloads.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="ham", help="Output directory."
    )
    args = parser.parse_args()

    if os.path.isdir(args.raw_dir):
        print("Organizing raw files ...")
        organize(args.raw_dir, args.output_dir)

    print("\nVerifying output structure ...")
    if verify(args.output_dir):
        print("\nHAM10000 preprocessing complete.")
    else:
        print(
            "\nSome files are missing.  Please check the instructions in "
            "data/README.md."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
