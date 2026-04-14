#!/usr/bin/env bash
# ================================================================
# Quick demo: run LVLM methods (local HF transformers) x datasets.
# Uses only 100 data points, --image_direct for PIL input.
#
# Uncomment additional model entries below to test more checkpoints.
#
# Usage:
#   bash scripts/demo_lvlm_methods.sh                              # run all
#   bash scripts/demo_lvlm_methods.sh --methods qwen llama         # specific methods
#   bash scripts/demo_lvlm_methods.sh --datasets celeba utk        # specific datasets
# ================================================================
set -uo pipefail

LOGFILE="demo_lvlm_methods_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "Logging to $LOGFILE"

# -- defaults -----------------------------------------------------
ALL_DATASETS=(celeba utk fairface ham fitz facet waterbirds)

SAVE_ROOT="./tmp/demo_lvlm"
MAX_SAMPLES=100
BS=1
NUM_WORKERS=0

declare -A DS_SA DS_TA
DS_SA=([celeba]=sex [utk]=ethnicity [fairface]=sex [ham]=age [fitz]=skin_binary [facet]=gender [waterbirds]=background)
DS_TA=([celeba]=33 [utk]=gender [fairface]=race [ham]=None [fitz]=None [facet]=visible_face [waterbirds]=species)

# -- method -> model list -----------------------------------------
# Each entry is  "method_name|huggingface_model_id"
# Uncomment lines to add more models; each uncommented entry will be tested.
ALL_ENTRIES=(
    # Qwen2.5-VL
    "qwen|Qwen/Qwen2.5-VL-7B-Instruct"
    # "qwen|Qwen/Qwen2.5-VL-32B-Instruct"
    # "qwen|Qwen/Qwen2.5-VL-72B-Instruct"
    # Gemma 3
    # "gemma|google/gemma-3-4b-it"
    # "gemma|google/gemma-3-12b-it"
    # "gemma|google/gemma-3-27b-it"
    # Llama 3.2 Vision / Llama 4 Scout
    # "llama|meta-llama/Llama-3.2-11B-Vision-Instruct"
    # "llama|meta-llama/Llama-3.2-90B-Vision-Instruct"
    # "llama|meta-llama/Llama-4-Scout-17B-16E-Instruct"
    # LLaVA-NeXT
    # "llava_next|llava-hf/llava-v1.6-vicuna-7b-hf"
    # "llava_next|llava-hf/llava-v1.6-vicuna-13b-hf"
    # "llava_next|llava-hf/llava-v1.6-34b-hf"
)

# -- argument parsing ---------------------------------------------
FILTER_METHODS=()
DATASETS=()
mode=""
for arg in "$@"; do
    case "$arg" in
        --methods)  mode="methods"  ; continue ;;
        --datasets) mode="datasets" ; continue ;;
        *)
            if   [[ "$mode" == "methods"  ]]; then FILTER_METHODS+=("$arg")
            elif [[ "$mode" == "datasets" ]]; then DATASETS+=("$arg")
            fi
            ;;
    esac
done
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${ALL_DATASETS[@]}")

# Build the final list of entries (filter by method if requested)
ENTRIES=()
if [[ ${#FILTER_METHODS[@]} -eq 0 ]]; then
    ENTRIES=("${ALL_ENTRIES[@]}")
else
    for entry in "${ALL_ENTRIES[@]}"; do
        m="${entry%%|*}"
        for fm in "${FILTER_METHODS[@]}"; do
            if [[ "$m" == "$fm" ]]; then
                ENTRIES+=("$entry")
                break
            fi
        done
    done
fi

# -- run loop -----------------------------------------------------
PASS=0
FAIL=0
FAILED_LIST=()

total=$(( ${#ENTRIES[@]} * ${#DATASETS[@]} ))
count=0

for entry in "${ENTRIES[@]}"; do
    method="${entry%%|*}"
    model="${entry#*|}"
    model_short="${model##*/}"

    for dataset in "${DATASETS[@]}"; do
        count=$((count + 1))
        sa="${DS_SA[$dataset]}"
        ta="${DS_TA[$dataset]}"
        save_path="${SAVE_ROOT}/${method}_${model_short}_${dataset}"

        echo ""
        echo "============================================================"
        echo "  [${count}/${total}] METHOD=${method}  MODEL=${model_short}"
        echo "                  DATASET=${dataset}  sa=${sa}  ta=${ta}"
        echo "============================================================"

        python -m release_benchmark.cli.zeroshot \
            --method  "$method"     \
            --model   "$model"      \
            --dataset "$dataset"    \
            --sa      "$sa"         \
            --ta      "$ta"         \
            --bs          "$BS"          \
            --max_samples "$MAX_SAMPLES" \
            --image_direct               \
            --nowandb                    \
            --save_path   "$save_path"   \
            --num_workers "$NUM_WORKERS"

        if [[ $? -eq 0 ]]; then
            echo "  >>> PASS"
            ((PASS++))
        else
            echo "  >>> FAIL"
            ((FAIL++))
            FAILED_LIST+=("${method}/${model_short}/${dataset}")
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
