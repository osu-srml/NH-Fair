"""UTKFace dataset preprocessing.

Organizes the UTKFace "In-the-Wild" aligned & cropped images into the layout
expected by ``release_benchmark.datasets.utk_dataset.UTKDataset``.

Expected output structure
-------------------------
<output_dir>/
└── UTKface_inthewild/
    └── *.jpg   (images named <age>_<gender>_<race>_<timestamp>.jpg)

Source: https://susanqq.github.io/UTKFace/
"""

import argparse
import os
import re
import shutil
import sys

IMAGE_DIR = "UTKface_inthewild"
FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)_\d+\..*$")


def organize(raw_dir: str, output_dir: str) -> None:
    """Copy or symlink raw images into the expected layout."""
    dst_dir = os.path.join(output_dir, IMAGE_DIR)
    os.makedirs(dst_dir, exist_ok=True)

    candidates = [
        f for f in os.listdir(raw_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not candidates:
        for sub in os.listdir(raw_dir):
            sub_path = os.path.join(raw_dir, sub)
            if os.path.isdir(sub_path):
                candidates = [
                    f
                    for f in os.listdir(sub_path)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ]
                if candidates:
                    raw_dir = sub_path
                    break

    copied = 0
    skipped = 0
    for fname in candidates:
        if FILENAME_PATTERN.match(fname):
            src = os.path.join(raw_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1

    print(f"  Copied {copied} images, skipped {skipped} (invalid filename format).")


def verify(output_dir: str) -> bool:
    """Return *True* when all expected files are in place."""
    img_dir = os.path.join(output_dir, IMAGE_DIR)
    ok = True

    if not os.path.isdir(img_dir):
        print(f"  [MISSING] {img_dir}/")
        return False

    images = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    valid = [f for f in images if FILENAME_PATTERN.match(f)]

    print(
        f"  [OK]      {img_dir}/ ({len(images)} images, {len(valid)} valid filenames)"
    )

    if len(valid) == 0:
        print("  [ERROR]   No valid UTKFace images found.")
        ok = False
    elif len(valid) < len(images):
        print(
            f"  [WARN]    {len(images) - len(valid)} images have "
            f"unexpected filenames and will be ignored at training time."
        )

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the UTKFace dataset.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="utk/raw",
        help="Directory containing downloaded UTKFace images.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="utk", help="Output directory."
    )
    args = parser.parse_args()

    if os.path.isdir(args.raw_dir):
        print("Organizing raw files ...")
        organize(args.raw_dir, args.output_dir)

    print("\nVerifying output structure ...")
    if verify(args.output_dir):
        print("\nUTKFace preprocessing complete.")
    else:
        print(
            "\nSome files are missing.  Please check the instructions in "
            "data/README.md."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
