# NH-Fair: Benchmarking Bias Mitigation Toward Fairness Without Harm

Official code for **"Benchmarking Bias Mitigation Toward Fairness Without Harm from Vision to LVLMs"** (ICLR 2026).

[[Paper]](https://arxiv.org/abs/2602.03895)

NH-Fair is a unified fairness benchmark that covers both vision models and large vision-language models (LVLMs) under standardized data, metrics, and training protocols. It provides a tuning-aware, sweep-first pipeline for rigorous, harm-aware fairness evaluation.

## Installation

From the `release/` directory (this folder):

```bash
pip install -e .
```

You can also use [uv](https://docs.astral.sh/uv/): run `uv sync` (and optional extras such as `uv sync --extra lavis` or `uv sync --extra llm` as defined in `pyproject.toml`), then e.g. `uv run python -m release_benchmark.cli.train --help`.

Run CLIs as `python -m release_benchmark.cli.<command>` from any working directory after installation, or set `PYTHONPATH=src` when running from `release/` without installing.

## Key Features

- **7 datasets**: CelebA, UTKFace, FairFace, FACET, Waterbirds, HAM10000, Fitzpatrick17k
- **18+ bias mitigation methods**: ERM, GroupDRO, LAFTR, DFR, FairMixup, CLIP-based, OxonFair, and more
- **10 LVLMs**: Qwen2.5-VL, LLaMA 3.2/4, Gemma 3, LLaVA-NeXT — local **transformers** or **OpenAI-compatible gateway** (vLLM, TGI, …)
- **Standardized metrics**: Accuracy, AUC, DP, EqOpp, EqOdd, worst-group accuracy, accuracy gap
## Dataset Setup

Each dataset has a dedicated preprocessing script under `data/<dataset>/preprocess.py`.
See [data/README.md](data/README.md) for detailed download links and step-by-step instructions.

**Quick start:**

```bash
cd data

# Download raw files into <dataset>/raw/, then run:
python celeba/preprocess.py   --raw_dir celeba/raw   --output_dir celeba
python utk/preprocess.py      --raw_dir utk/raw      --output_dir utk
python fairface/preprocess.py --raw_dir fairface/raw  --output_dir fairface
python facet/preprocess.py    --raw_dir facet/raw     --output_dir facet    --num_workers 8
python waterbirds/preprocess.py --raw_dir waterbirds/raw --output_dir waterbirds
python ham/preprocess.py      --raw_dir ham/raw       --output_dir ham
python fitz/preprocess.py     --raw_dir fitz/raw      --output_dir fitz     --num_workers 8
```


| Dataset        | Source                                                                                      | Sensitive Attr | Target               |
| -------------- | ------------------------------------------------------------------------------------------- | -------------- | -------------------- |
| CelebA         | [torchvision](https://pytorch.org/vision/stable/datasets.html#celeba)                       | Gender         | 40 binary attributes |
| UTKFace        | [UTKFace](https://susanqq.github.io/UTKFace/)                                               | Race / Gender  | Gender / Race        |
| FairFace       | [FairFace](https://github.com/joojs/fairface)                                               | Race / Gender  | Gender / Race        |
| FACET          | [FACET](https://ai.meta.com/datasets/facet-downloads/)                                      | Gender         | Face visibility      |
| Waterbirds     | [Waterbirds](https://github.com/kohpangwei/group_DRO)                                       | Background     | Bird type            |
| HAM10000       | [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | Sex / Age      | Diagnosis            |
| Fitzpatrick17k | [Fitz17k](https://github.com/mattgroh/fitzpatrick17k)                                       | Skin type      | Diagnosis            |


## Usage

### Supervised Training

```bash
python -m release_benchmark.cli.train \
  --dataset celeba --method erm --sa sex --ta 33 \
  --model resnet18 --pretrain 1 --lr 0.001 --bs 128 \
  --epochs 30 --gpu 0
```

### Zero-shot LLM Evaluation

```bash
python -m release_benchmark.cli.zeroshot \
  --dataset celeba --method qwen --model Qwen/Qwen2.5-VL-7B-Instruct --sa sex --ta 33 \
  --image_direct --bs 1 --gpu 0

# Same LVLM via OpenAI-compatible server (start one with scripts/launch_llm_gateway_qwen.sh, or use vendors' API)
python -m release_benchmark.cli.zeroshot \
  --dataset celeba --method qwen --vlm_backend gateway \
  --llm_gateway_url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-VL-7B-Instruct \
  --sa sex --ta 33 --image_direct --bs 1 --gpu 0

python -m release_benchmark.cli.zeroshot \
  --dataset waterbirds --method blip2 --bs 32 --gpu 0

python -m release_benchmark.cli.zeroshot \
  --dataset celeba --method clip --model resnet --sa sex --ta 33 --bs 32 --gpu 0
```

### Sweep (Hyperparameter Search via W&B)

```bash
python -m release_benchmark.cli.sweep \
  --dataset celeba --method erm --sa sex --ta 33 --gpu auto
```

## Implemented Methods


| Category     | Methods                                       |
| ------------ | --------------------------------------------- |
| Baseline     | ERM, RandAug, Resample                  |
| Fairness     | GroupDRO, LAFTR, DFR, GapReg, MCDP, FairMixup |
| Data-centric | FIS, BM (Bias Mimicking), FSCL+               |
| CLIP-based   | CLIP, CLIP-SFID, CLIP-Fairer, BLIP2           |
| Post-hoc     | OxonFair, Decoupled                           |
| LVLMs        | Qwen2.5-VL, LLaMA 3.2/4, Gemma 3, LLaVA-NeXT  |


## Project Structure

```
release/
├── data/
│   ├── manifests/        # Dataset manifest examples
│   ├── celeba/           # CelebA preprocessing
│   ├── utk/              # UTKFace preprocessing
│   ├── fairface/         # FairFace preprocessing
│   ├── facet/            # FACET preprocessing
│   ├── waterbirds/       # Waterbirds preprocessing
│   ├── ham/              # HAM10000 preprocessing
│   └── fitz/             # Fitzpatrick17k preprocessing
├── docs/                 # Method audit and sweep guide
├── scripts/              # Shell scripts for common workflows
├── src/release_benchmark/
│   ├── cli/              # Entry points: train, zeroshot, sweep
│   ├── configs/          # Sweep and dataset YAML templates (packaged with the lib)
│   ├── datasets/         # Dataset loaders (FairDataset base class)
│   ├── methods/
│   │   ├── cv/           # Vision methods (ERM, GroupDRO, LAFTR, ...)
│   │   ├── vlm/          # CLIP/BLIP-2; LVLMs (HF + gateway); see scripts/launch_llm_gateway_*.sh
│   │   └── registry.py   # Method name -> class resolution
│   ├── metrics/          # Fairness and performance metrics
│   └── utils/            # Seeds, logging, helpers
└── tests/                # Smoke and registry tests
```

## Environment Variables


| Variable        | Description                                    |
| --------------- | ---------------------------------------------- |
| `WANDB_API_KEY` | Weights & Biases API key for sweep logging     |
| `HF_TOKEN`      | Hugging Face token (required for gated models) |


## Citation

```bibtex
@inproceedings{
tan2026benchmarking,
title={Benchmarking Bias Mitigation Toward Fairness Without Harm from Vision to {LVLM}s},
author={Xuwei Tan and Ziyu Hu and Xueru Zhang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=GLPmZhhCAE}
}
```

## License

This project is released under the [MIT License](LICENSE).