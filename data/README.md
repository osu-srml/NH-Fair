# Dataset Preparation

This directory contains preprocessing scripts for all 7 datasets used in the NH-Fair benchmark.
We do not redistribute the original datasets due to licensing restrictions.
Please download each dataset from its official source and then run the corresponding preprocessing script.

## Quick Start

```bash
# From the release/ directory
cd data

# 1. Download raw data into each dataset folder (see per-dataset instructions below)
# 2. Run preprocessing for each dataset
python celeba/preprocess.py   --raw_dir celeba/raw   --output_dir celeba
# or: python celeba/preprocess.py --download
python utk/preprocess.py      --raw_dir utk/raw      --output_dir utk
python fairface/preprocess.py --raw_dir fairface/raw  --output_dir fairface
python facet/preprocess.py    --raw_dir facet/raw     --output_dir facet
python waterbirds/preprocess.py --raw_dir waterbirds/raw --output_dir waterbirds
python ham/preprocess.py      --raw_dir ham/raw       --output_dir ham
python fitz/preprocess.py     --raw_dir fitz/raw      --output_dir fitz

# 3. Train
python -m release_benchmark.cli.train --data_path data/ --dataset celeba ...
```

## Expected Directory Structure After Preprocessing

```
data/
├── celeba/
│   ├── img_align_celeba/            # 202,599 face images
│   ├── list_attr_celeba.txt         # 40 binary attributes
│   ├── list_eval_partition.txt
│   └── identity_CelebA.txt
├── utk/
│   └── UTKface_inthewild/           # ~24K images named age_gender_race_timestamp.jpg
├── fairface/
│   ├── fairface_label_train.csv
│   ├── fairface_label_val.csv
│   ├── train/                       # Training images
│   └── val/                         # Validation images
├── facet/
│   ├── annotations/
│   │   └── annotations.csv         # FACET annotations
│   └── img224/                      # Images resized to 224×224
├── waterbirds/
│   ├── metadata.csv
│   └── (image directories)
├── ham/
│   └── base/
│       ├── HAM10000_metadata.csv
│       └── HAM10000_images/         # Dermoscopic images
└── fitz/
    └── finalfitz17k/
        ├── fitzpatrick17k.csv
        └── images/                  # Skin condition images
```

## Per-Dataset Instructions

### CelebA

| | |
|---|---|
| **Source** | [CelebA Website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or `torchvision.datasets.CelebA` |
| **Target** | 40 binary attributes (default: Wavy Hair, index 33) |
| **Sensitive Attr** | Gender (index 20) |

**Download options:**

1. **Via torchvision** (recommended): The preprocessing script can download automatically.
2. **Manual**: Download `img_align_celeba.zip`, `list_attr_celeba.txt`, `list_eval_partition.txt`, and `identity_CelebA.txt` from the CelebA website.

```bash
python celeba/preprocess.py --raw_dir celeba/raw --output_dir celeba
# Add --download to auto-download via torchvision
```

### UTKFace

| | |
|---|---|
| **Source** | [UTKFace Website](https://susanqq.github.io/UTKFace/) |
| **Target** | Gender |
| **Sensitive Attr** | Ethnicity |

Download the **"In-the-wild"** aligned & cropped faces. Place images in the raw directory.

```bash
python utk/preprocess.py --raw_dir utk/raw --output_dir utk
```

### FairFace

| | |
|---|---|
| **Source** | [FairFace GitHub](https://github.com/joojs/fairface) |
| **Target** | Race (7 classes) |
| **Sensitive Attr** | Gender |

Download `fairface_label_train.csv`, `fairface_label_val.csv`, and the image archives from the FairFace GitHub page (Google Drive links).

```bash
python fairface/preprocess.py --raw_dir fairface/raw --output_dir fairface
```

### FACET

| | |
|---|---|
| **Source** | [FACET](https://ai.meta.com/datasets/facet-downloads/) |
| **Target** | Visible face |
| **Sensitive Attr** | Gender presentation |

Download the annotations CSV and images from Meta's FACET page. The preprocessing script resizes images to 224×224.

```bash
python facet/preprocess.py --raw_dir facet/raw --output_dir facet --num_workers 8
```

### Waterbirds

| | |
|---|---|
| **Source** | [Waterbirds (group_DRO)](https://github.com/kohpangwei/group_DRO) |
| **Target** | Bird species (waterbird vs. landbird) |
| **Sensitive Attr** | Background (land vs. water) |

Download the preprocessed Waterbirds dataset from the group_DRO repository.

```bash
python waterbirds/preprocess.py --raw_dir waterbirds/raw --output_dir waterbirds
```

### HAM10000

| | |
|---|---|
| **Source** | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) |
| **Target** | Binary diagnosis (melanoma/akiec vs. others) |
| **Sensitive Attr** | Sex (or sex × age) |

Download `HAM10000_metadata.csv` and image archives (`HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`).

```bash
python ham/preprocess.py --raw_dir ham/raw --output_dir ham
```

### Fitzpatrick17k

| | |
|---|---|
| **Source** | [Fitzpatrick17k GitHub](https://github.com/mattgroh/fitzpatrick17k) |
| **Target** | Binary diagnosis (malignant vs. non-malignant) |
| **Sensitive Attr** | Skin type (binarized: types I–III vs. IV–VI) |

Download `fitz17k.csv` (or `fitzpatrick17k.csv`) and the images from the GitHub repo. Place the CSV and an `images/` folder under `fitz/raw/`, then run the preprocessing script.

```bash
python fitz/preprocess.py --raw_dir fitz/raw --output_dir fitz --num_workers 8
```

## Training

After preprocessing, train with:

```bash
python -m release_benchmark.cli.train \
  --data_path data/ \
  --dataset celeba --method erm --sa sex --ta 33 \
  --model resnet18 --pretrain 1 --lr 0.001 --bs 128 \
  --epochs 30 --gpu 0
```

See the main [README](../README.md) for full usage instructions.
