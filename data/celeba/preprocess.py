"""CelebA dataset preprocessing.

Downloads (optionally) and organizes the CelebA dataset into the layout
expected by ``release_benchmark.datasets.celeba_dataset.CelebaDataset``.

The training code calls ``torchvision.datasets.CelebA(root=data_path)``
which expects a ``celeba/`` subdirectory inside *data_path*.  When running
from the ``data/`` directory with ``--output_dir celeba``, the files are
placed directly in ``celeba/`` so that ``--data_path data/`` works.

Expected output structure (relative to data/)
----------------------------------------------
celeba/
├── img_align_celeba/
│   └── *.jpg              (202 599 aligned face images)
├── list_attr_celeba.txt
├── list_eval_partition.txt
└── identity_CelebA.txt

Source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
"""

import argparse
import os
import shutil
import sys

REQUIRED_FILES = [
    "list_attr_celeba.txt",
    "list_eval_partition.txt",
    "identity_CelebA.txt",
    "list_bbox_celeba.txt",
    "list_landmarks_align_celeba.txt",
]
IMAGE_DIR = "img_align_celeba"


def download_via_torchvision(output_dir: str) -> None:
    """Use torchvision to download CelebA.

    ``torchvision.datasets.CelebA(root=parent)`` creates a ``celeba/``
    folder inside *parent*, so we pass the parent of *output_dir*.
    """
    try:
        from torchvision import datasets
    except ImportError:
        print(
            "torchvision is required for automatic download.  "
            "Install it with: pip install torchvision"
        )
        sys.exit(1)

    parent = os.path.dirname(os.path.abspath(output_dir))
    if not parent:
        parent = "."
    print("Downloading CelebA via torchvision (this may take a while) ...")
    datasets.CelebA(root=parent, split="all", download=True)
    print("Download complete.")


def organize(raw_dir: str, output_dir: str) -> None:
    """Copy raw files directly into *output_dir* (the ``celeba/`` folder)."""
    os.makedirs(output_dir, exist_ok=True)

    for fname in REQUIRED_FILES:
        src = os.path.join(raw_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"  Copying {fname}")
            shutil.copy2(src, dst)

    src_img = os.path.join(raw_dir, IMAGE_DIR)
    dst_img = os.path.join(output_dir, IMAGE_DIR)
    if os.path.isdir(src_img) and not os.path.exists(dst_img):
        print(f"  Linking {IMAGE_DIR}/ -> {dst_img}")
        os.symlink(os.path.abspath(src_img), dst_img)


def verify(output_dir: str) -> bool:
    """Return *True* when all expected files are in place."""
    ok = True

    for fname in REQUIRED_FILES:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            ok = False
        else:
            print(f"  [OK]      {path}")

    img_dir = os.path.join(output_dir, IMAGE_DIR)
    if not os.path.isdir(img_dir):
        print(f"  [MISSING] {img_dir}/")
        ok = False
    else:
        n = len(
            [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
        )
        print(f"  [OK]      {img_dir}/ ({n} images)")
        if n == 0:
            ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the CelebA dataset.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="celeba/raw",
        help="Directory containing downloaded CelebA files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="celeba",
        help="Output directory (the 'celeba' folder itself).",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download CelebA via torchvision."
    )
    args = parser.parse_args()

    if args.download:
        download_via_torchvision(args.output_dir)

    if os.path.isdir(args.raw_dir):
        print("Organising raw files ...")
        organize(args.raw_dir, args.output_dir)

    print("\nVerifying output structure ...")
    if verify(args.output_dir):
        print("\nCelebA preprocessing complete.")
    else:
        print(
            "\nSome files are missing.  Please check the instructions in "
            "data/README.md."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
