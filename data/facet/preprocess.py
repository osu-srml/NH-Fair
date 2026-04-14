"""FACET dataset preprocessing.

Organizes the FACET dataset and resizes images to 224x224 as expected by
``release_benchmark.datasets.facet_dataset.FacetDataset``.

Expected output structure
-------------------------
<output_dir>/
├── annotations/
│   └── annotations.csv
└── img/
    └── *.jpg

Source: https://ai.meta.com/datasets/facet-downloads/
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from tqdm import tqdm

ANNOTATIONS_SUBDIR = "annotations"
ANNOTATIONS_FILE = "annotations.csv"
IMAGE_OUTPUT_DIR = "img224"


def _resize_single(args_tuple):
    """Resize a single image (pickleable for multiprocessing)."""
    src_path, dst_path, size = args_tuple
    try:
        with Image.open(src_path) as img:
            img_resized = img.resize(size, Image.LANCZOS)
            img_resized.save(dst_path)
    except Exception as e:
        return f"Error: {src_path}: {e}"
    return None


def resize_images(
    raw_img_dir: str, output_dir: str, size: tuple = (224, 224), num_workers: int = 16
) -> None:
    """Resize all images in *raw_img_dir* to *size* and save to *output_dir*/img224."""
    dst_dir = os.path.join(output_dir, IMAGE_OUTPUT_DIR)
    os.makedirs(dst_dir, exist_ok=True)

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(extensions)]

    already_done = set(os.listdir(dst_dir))
    to_process = [f for f in files if f not in already_done]

    if not to_process:
        print(f"  All {len(files)} images already resized.")
        return

    print(
        f"  Resizing {len(to_process)} images "
        f"({len(files) - len(to_process)} already done) ..."
    )

    tasks = [
        (os.path.join(raw_img_dir, f), os.path.join(dst_dir, f), size)
        for f in to_process
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        results = list(
            tqdm(
                pool.map(_resize_single, tasks),
                total=len(tasks),
                desc="Resizing",
                unit="img",
            )
        )

    errors = [r for r in results if r is not None]
    if errors:
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


def organize(raw_dir: str, output_dir: str, num_workers: int = 16) -> None:
    """Copy annotations and resize images."""
    ann_src = os.path.join(raw_dir, ANNOTATIONS_SUBDIR)
    ann_dst = os.path.join(output_dir, ANNOTATIONS_SUBDIR)
    os.makedirs(ann_dst, exist_ok=True)

    src_csv = os.path.join(ann_src, ANNOTATIONS_FILE)
    if not os.path.exists(src_csv):
        src_csv = os.path.join(raw_dir, ANNOTATIONS_FILE)
    dst_csv = os.path.join(ann_dst, ANNOTATIONS_FILE)

    if os.path.exists(src_csv) and not os.path.exists(dst_csv):
        print(f"  Copying {ANNOTATIONS_FILE}")
        shutil.copy2(src_csv, dst_csv)

    raw_img_dir = None
    for candidate in ("img", "images"):
        p = os.path.join(raw_dir, candidate)
        if os.path.isdir(p):
            imgs = [
                f
                for f in os.listdir(p)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            if imgs:
                raw_img_dir = p
                break

    if raw_img_dir is None:
        print(
            "  [WARN] Could not find raw images directory. "
            "Please place images in <raw_dir>/img/ or <raw_dir>/images/"
        )
    else:
        resize_images(raw_img_dir, output_dir, num_workers=num_workers)


def verify(output_dir: str) -> bool:
    """Return *True* when all expected files are in place."""
    ok = True

    csv_path = os.path.join(output_dir, ANNOTATIONS_SUBDIR, ANNOTATIONS_FILE)
    if not os.path.exists(csv_path):
        print(f"  [MISSING] {csv_path}")
        ok = False
    else:
        import pandas as pd

        df = pd.read_csv(csv_path)
        required_cols = {
            "gender_presentation_masc",
            "gender_presentation_fem",
            "filename",
            "class1",
        }
        missing = required_cols - set(df.columns)
        if missing:
            print(f"  [ERROR]   {ANNOTATIONS_FILE} missing columns: {missing}")
            ok = False
        else:
            print(f"  [OK]      {csv_path} ({len(df)} rows)")

    img_dir = os.path.join(output_dir, IMAGE_OUTPUT_DIR)
    if not os.path.isdir(img_dir):
        print(f"  [MISSING] {img_dir}/")
        ok = False
    else:
        n = len(
            [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
        )
        print(f"  [OK]      {img_dir}/ ({n} images)")
        if n == 0:
            ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the FACET dataset.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="facet/raw",
        help="Directory with raw FACET downloads.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="facet", help="Output directory."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Parallel workers for image resizing.",
    )
    args = parser.parse_args()

    if os.path.isdir(args.raw_dir):
        print("Organizing raw files ...")
        organize(args.raw_dir, args.output_dir, num_workers=args.num_workers)

    print("\nVerifying output structure ...")
    if verify(args.output_dir):
        print("\nFACET preprocessing complete.")
    else:
        print(
            "\nSome files are missing.  Please check the instructions in "
            "data/README.md."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
