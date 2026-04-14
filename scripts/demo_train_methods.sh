#!/usr/bin/env bash
# ================================================================
# Quick demo: run every training method x dataset with tiny data.
# Verifies the full train pipeline works end-to-end.
#
# Usage:
#   bash scripts/demo_train_methods.sh                       # run all
#   bash scripts/demo_train_methods.sh --methods erm mixup   # specific methods
#   bash scripts/demo_train_methods.sh --datasets celeba utk # specific datasets
# ================================================================
set -uo pipefail

LOGFILE="demo_train_methods_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "Logging to $LOGFILE"

# -- defaults -----------------------------------------------------
ALL_METHODS=(erm randaug mixup bm decoupled gapreg mcdp fis fscl groupdro laftr)
ALL_DATASETS=(celeba utk fairface ham fitz facet waterbirds)

SAVE_ROOT="./tmp/demo_train"
MAX_SAMPLES=800
EPOCHS=5
BS=32
NUM_WORKERS=0

declare -A DS_SA DS_TA
DS_SA=([celeba]=sex [utk]=ethnicity [fairface]=sex [ham]=age [fitz]=skin_binary [facet]=gender [waterbirds]=background)
DS_TA=([celeba]=33 [utk]=gender [fairface]=race [ham]=None [fitz]=None [facet]=visible_face [waterbirds]=species)

# -- argument parsing ---------------------------------------------
METHODS=()
DATASETS=()
mode=""
for arg in "$@"; do
    case "$arg" in
        --methods)  mode="methods"  ; continue ;;
        --datasets) mode="datasets" ; continue ;;
        *)
            if   [[ "$mode" == "methods"  ]]; then METHODS+=("$arg")
            elif [[ "$mode" == "datasets" ]]; then DATASETS+=("$arg")
            fi
            ;;
    esac
done
[[ ${#METHODS[@]}  -eq 0 ]] && METHODS=("${ALL_METHODS[@]}")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${ALL_DATASETS[@]}")

# -- run loop -----------------------------------------------------
PASS=0
FAIL=0
FAILED_LIST=()

total=$(( ${#METHODS[@]} * ${#DATASETS[@]} ))
count=0

for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        count=$((count + 1))
        sa="${DS_SA[$dataset]}"
        ta="${DS_TA[$dataset]}"
        save_path="${SAVE_ROOT}/${method}_${dataset}"

        echo ""
        echo "============================================================"
        echo "  [${count}/${total}] METHOD=${method}  DATASET=${dataset}  sa=${sa}  ta=${ta}"
        echo "============================================================"

        python -m release_benchmark.cli.train \
            --method  "$method"  \
            --dataset "$dataset" \
            --sa      "$sa"      \
            --ta      "$ta"      \
            --epochs      "$EPOCHS"      \
            --bs          "$BS"          \
            --max_samples "$MAX_SAMPLES" \
            --nowandb                    \
            --save_path   "$save_path"   \
            --num_workers "$NUM_WORKERS" \
            --max_patience 3

        if [[ $? -eq 0 ]]; then
            echo "  >>> PASS"
            ((PASS++))
        else
            echo "  >>> FAIL"
            ((FAIL++))
            FAILED_LIST+=("${method}/${dataset}")
        fi
    done
done

# -- summary ------------------------------------------------------
echo ""
echo "============================================================"
echo "  SUMMARY: ${PASS} passed, ${FAIL} failed (out of ${total})"
echo "============================================================"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    for f in "${FAILED_LIST[@]}"; do
        echo "  [FAIL] $f"
    done
fi

exit "$FAIL"
