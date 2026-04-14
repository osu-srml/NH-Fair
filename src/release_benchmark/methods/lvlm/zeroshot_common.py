"""Shared zeroshot LVLM logic: prompts, logits from text, metrics, optional save."""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from release_benchmark.methods.lvlm.llm_utils import (
    build_conversation,
    build_conversation_open,
    evaluate_open_generation,
    extract_classification_from_open_text,
    get_valid_labels,
    predict_and_get_probs,
)
from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.utils.common import AverageMeter


def build_prompt_text(args) -> str:
    """Single user text string for gateway or for building chat templates."""
    task_mode = getattr(args, "task_mode", "classification")
    if task_mode == "open_generation":
        return build_conversation_open(
            args.dataset,
            prompt_style=args.prompt_style,
            task_attr=args.ta,
            sensitive_attr=args.sa,
            textonly=True,
        )
    return build_conversation(
        args.dataset,
        prompt_style=args.prompt_style,
        task_attr=args.ta,
        sensitive_attr=args.sa,
        textonly=True,
    )


def max_output_tokens(task_mode: str, args) -> int:
    if task_mode == "classification":
        return 32
    if task_mode == "open_generation":
        return min(1000, getattr(args, "max_tokens", 1000))
    return 256


def append_outputs_from_generated(
    output_text: list[str],
    task_mode: str,
    args,
    tol_output: list,
    tol_generated_texts: list,
) -> bool:
    """Parse model strings into tol_output rows. Returns False if batch skipped."""
    if not output_text:
        return False
    if output_text[0] in (None, "None", "Unknown"):
        return False

    tol_generated_texts.extend(output_text)

    if task_mode == "classification":
        try:
            probs, _ = predict_and_get_probs(
                output_text, dataset=args.dataset, sensitive_attr=args.sa
            )
        except ValueError as e:
            print(
                f"{output_text} Error processing output: {e}; manually check"
            )  # Cannot be automated, need to manually check
            return False
        tol_output += probs
        return True

    try:
        extracted = extract_classification_from_open_text(
            output_text, dataset=args.dataset, sensitive_attr=args.sa
        )
        valid_labels = get_valid_labels(args.dataset, args.sa)
        for label in extracted:
            if label is None:
                prob = [1.0 / len(valid_labels)] * len(valid_labels)
            else:
                prob = [0.0] * len(valid_labels)
                if label in valid_labels:
                    prob[valid_labels.index(label)] = 1.0
                else:
                    prob = [1.0 / len(valid_labels)] * len(valid_labels)
            tol_output.append(prob)
        return True
    except Exception as e:
        print(f"Error extracting classification from open text: {e}")
        return False


def finalize_zeroshot_metrics(
    tol_output: list,
    tol_target: list,
    tol_sensitive: list,
    tol_index: list,
    tol_generated_texts: list,
    args,
    task_mode: str,
    num_classes: int,
    sensitive_attributes: int,
    mode: str,
    epoch: int,
    loss_meter: AverageMeter,
) -> dict:
    print(len(tol_output), len(tol_target), len(tol_sensitive), len(tol_index))
    assert len(tol_output) == len(tol_target)
    log_dict, _, _ = calculate_metrics(
        tol_output,
        tol_target,
        tol_sensitive,
        tol_index,
        sensitive_attributes,
        num_class=num_classes,
        skip_auc=True,
    )
    if task_mode == "open_generation" and tol_generated_texts:
        open_eval = evaluate_open_generation(
            tol_generated_texts, dataset=args.dataset, sensitive_attr=args.sa
        )
        log_dict["open_gen_avg_length"] = open_eval["avg_length"]
        log_dict["open_gen_texts_sample"] = tol_generated_texts[:5]
        print("\n--- Open Generation Metrics ---")
        print(f"Average text length: {open_eval['avg_length']:.2f} words")
        print("Sample outputs:")
        for i, text in enumerate(tol_generated_texts[:3]):
            print(f"  [{i + 1}] {text[:200]}...")
    print(
        f"\n##################################### {mode.capitalize()} {epoch} #####################################"
    )
    print(log_dict)
    return log_dict


def save_generation_artifacts(
    args,
    task_mode: str,
    mode: str,
    epoch: int,
    generated_texts: list,
    targets: list,
    sensitive_attrs: list,
    predictions: np.ndarray,
    num_classes: int,
    gateway_extra: dict | None = None,
):
    """JSON + CSV + summary (qwen-style layout)."""
    save_dir = os.path.join(
        getattr(args, "save_dir", None) or "./outputs",
        "generated_texts",
        args.dataset,
        args.method,
    )
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tms = task_mode if task_mode != "classification" else "cls"
    ps = (
        f"_p{args.prompt_style}"
        if getattr(args, "prompt_style", None) is not None
        else ""
    )
    sd = f"_s{args.seed}" if getattr(args, "seed", None) is not None else ""
    base = f"{mode}_epoch{epoch}_{tms}{ps}{sd}_{ts}"

    pred_labels = np.argmax(predictions, axis=1).tolist()
    valid_labels = get_valid_labels(args.dataset, getattr(args, "sa", None))
    records = []
    for i in range(len(generated_texts)):
        records.append(
            {
                "index": i,
                "generated_text": generated_texts[i],
                "target_label": targets[i],
                "target_label_name": valid_labels[targets[i]]
                if targets[i] < len(valid_labels)
                else "unknown",
                "predicted_label": pred_labels[i],
                "predicted_label_name": valid_labels[pred_labels[i]]
                if pred_labels[i] < len(valid_labels)
                else "unknown",
                "sensitive_attr": sensitive_attrs[i],
                "is_correct": pred_labels[i] == targets[i],
                "prediction_probs": predictions[i].tolist()
                if hasattr(predictions[i], "tolist")
                else list(predictions[i]),
            }
        )

    meta = {
        "dataset": args.dataset,
        "method": args.method,
        "model": getattr(args, "model", "unknown"),
        "task_mode": task_mode,
        "prompt_style": getattr(args, "prompt_style", None),
        "seed": getattr(args, "seed", None),
        "mode": mode,
        "epoch": epoch,
        "timestamp": ts,
        "total_samples": len(generated_texts),
        "num_classes": num_classes,
    }
    if gateway_extra:
        meta.update(gateway_extra)
    json_path = os.path.join(save_dir, f"{base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "data": records}, f, indent=2, ensure_ascii=False)

    rows = []
    for r in records:
        row = {k: r[k] for k in r if k != "prediction_probs"}
        for j, p in enumerate(r["prediction_probs"]):
            row[f"prob_class_{j}"] = p
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, f"{base}.csv"), index=False, encoding="utf-8"
    )

    summary_path = os.path.join(save_dir, f"{base}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            f"Dataset: {args.dataset}\nMethod: {args.method}\nTask Mode: {task_mode}\n"
        )
        if gateway_extra:
            for k, v in gateway_extra.items():
                f.write(f"{k}: {v}\n")
        correct = sum(1 for r in records if r["is_correct"])
        f.write(
            f"Overall Accuracy: {correct}/{len(records)} = {100 * correct / max(len(records), 1):.2f}%\n"
        )
    print("\n--- Saved outputs ---")
    print(f"JSON: {json_path}")
    print(f"CSV: {os.path.join(save_dir, f'{base}.csv')}")


def print_gateway_banner(base_url: str):
    print(f"Using OpenAI-compatible gateway at: {base_url}")
