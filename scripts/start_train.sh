#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Quick-start examples: one ERM run per dataset with small settings.
# Edit flags below or pass overrides to explore other methods.
# ================================================================

# -- CelebA -------------------------------------------------------
# ta = CelebA attribute index (e.g. 33 = Wavy_Hair)
# sa = not used by load_dataset for celeba (gender is hard-coded to attr 20)
python -m release_benchmark.cli.train \
    --method erm --dataset celeba \
    --ta 33 --sa sex \
    --data_path ./data --model resnet18 \
    --lr 0.001 --bs 128 --epochs 5 --optim sgd \
    --pretrain 0 --weight_decay 1e-4 --seed 42 \
    --gpu 0 --save_path ./results/erm_celeba --nowandb

# -- UTKFace ------------------------------------------------------
# sa in {ethnicity, ethnicity_age}  ta in {gender}
python -m release_benchmark.cli.train \
    --method erm --dataset utk \
    --ta gender --sa ethnicity \
    --data_path ./data --model resnet18 \
    --lr 0.001 --bs 128 --epochs 5 --optim sgd \
    --pretrain 0 --weight_decay 1e-4 --seed 42 \
    --gpu 0 --save_path ./results/erm_utk --nowandb

# -- FairFace -----------------------------------------------------
# sa in {race, sex/gender}  ta is auto-assigned based on sa
python -m release_benchmark.cli.train \
    --method erm --dataset fairface \
    --ta race --sa sex \
    --data_path ./data --model resnet18 \
    --lr 0.001 --bs 128 --epochs 5 --optim sgd \
    --pretrain 0 --weight_decay 1e-4 --seed 42 \
    --gpu 0 --save_path ./results/erm_fairface --nowandb

# -- HAM10000 -----------------------------------------------------
# sa in {sex, age, sex_age}  ta not used (label is malignant vs benign)
python -m release_benchmark.cli.train \
    --method erm --dataset ham \
    --ta None --sa age \
    --data_path ./data --model resnet18 \
    --lr 0.001 --bs 128 --epochs 5 --optim sgd \
    --pretrain 0 --weight_decay 1e-4 --seed 42 \
    --gpu 0 --save_path ./results/erm_ham --nowandb

# -- Fitzpatrick17k -----------------------------------------------
# sa in {skin_binary, skin_type}  ta not used (label is malignant vs benign)
python -m release_benchmark.cli.train \
    --method erm --dataset fitz \
    --ta None --sa skin_binary \
    --data_path ./data --model resnet18 \
    --lr 0.001 --bs 128 --epochs 5 --optim sgd \
    --pretrain 0 --weight_decay 1e-4 --seed 42 \
    --gpu 0 --save_path ./results/erm_fitz --nowandb

# -- FACET --------------------------------------------------------
# ta in {class, class_num, visible_face}  sa = gender (built-in)
python -m release_benchmark.cli.train \
    --method erm --dataset facet \
    --ta visible_face --sa gender \
    --data_path ./data --model resnet18 \
    --lr 0.001 --bs 128 --epochs 5 --optim sgd \
    --pretrain 0 --weight_decay 1e-4 --seed 42 \
    --gpu 0 --save_path ./results/erm_facet --nowandb

# -- Waterbirds ---------------------------------------------------
# ta = species (waterbird/landbird), sa = background (water/land)
python -m release_benchmark.cli.train \
    --method erm --dataset waterbirds \
    --ta species --sa background \
    --data_path ./data --model resnet18 \
    --lr 0.001 --bs 128 --epochs 5 --optim sgd \
    --pretrain 0 --weight_decay 1e-4 --seed 42 \
    --gpu 0 --save_path ./results/erm_waterbirds --nowandb
