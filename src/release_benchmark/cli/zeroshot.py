"""Zero-shot / LLM evaluation CLI for the NH-Fair benchmark."""

import argparse
import os
import sys
import traceback

import torch
import wandb
from torch.utils.data import DataLoader

from release_benchmark.datasets.dataset import load_dataset
from release_benchmark.utils.common import DualWriter, get_et_time, set_seed

# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args, wb):
    train_set, val_set, test_set, sensitive_attributes, num_classes = load_dataset(args)
    args.sensitive_attributes = sensitive_attributes
    args.num_classes = num_classes

    if getattr(args, "max_samples", 0) > 0:
        n = args.max_samples
        if isinstance(train_set, list):
            for ds in train_set:
                ds.indices = ds.indices[:n]
        else:
            train_set.indices = train_set.indices[:n]
        val_set.indices = val_set.indices[:n]
        test_set.indices = test_set.indices[:n]

    from release_benchmark.methods.registry import get_zeroshot_method

    Method = get_zeroshot_method(args.method)
    method = Method(args)

    if args.image_direct:

        def pil_collate_fn(batch):
            images, labels, sens = zip(*batch)
            return list(images), torch.tensor(labels), torch.tensor(sens)

        val_set.transform = None
        test_set.transform = None
        val_loader = DataLoader(
            val_set,
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=pil_collate_fn,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=pil_collate_fn,
        )
    else:
        val_loader = DataLoader(
            val_set, batch_size=args.bs, num_workers=args.num_workers, shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=args.bs, num_workers=args.num_workers, shuffle=False
        )

    if hasattr(method, "fit") and callable(method.fit):
        train_loader = DataLoader(
            train_set, batch_size=args.bs, num_workers=args.num_workers, shuffle=False
        )
        method.fit(train_loader, args)

    test_loss, test_log_dict = method.test(test_loader, 0, args)
    val_loss, val_log_dict = method.validate(val_loader, 0, args)

    log_info = {
        "epoch": 0,
        "Validation loss": val_loss,
        "Validation Accuracy": val_log_dict["Overall Acc"],
        "Validation Worst Accuracy": val_log_dict["worst_acc"],
        "Validation Accuracy Gap": val_log_dict["gap_acc"],
        "Validation DP": val_log_dict["DP"],
        "Validation EqOpp": val_log_dict.get("EqOpp1", val_log_dict.get("EqOppAvg")),
        "Validation EqOdd": val_log_dict["EqOdd"],
        "Test loss": test_loss,
        "Test Accuracy": test_log_dict["Overall Acc"],
        "Test Worst Accuracy": test_log_dict["worst_acc"],
        "Test Accuracy Gap": test_log_dict["gap_acc"],
        "Test DP": test_log_dict["DP"],
        "Test EqOpp": test_log_dict.get("EqOpp1", test_log_dict.get("EqOppAvg")),
        "Test EqOdd": test_log_dict["EqOdd"],
    }

    if wb is not None:
        wb.log(log_info)


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------


def _init_wandb(args, current_time):
    if args.nowandb:
        return None

    wb_key = os.getenv("WANDB_API_KEY")
    if wb_key:
        wandb.login(key=wb_key)

    if args.sweep:
        wb_dir = f"./wandb/{args.dataset}/{args.method}/{args.ta}/{args.sa}"
        os.makedirs(wb_dir, exist_ok=True)
        wandb.init(resume="auto", dir=wb_dir)
    else:
        wb_dir = f"./wandb/{args.dataset}/{args.method}/{args.ta}/{args.sa}"
        os.makedirs(wb_dir, exist_ok=True)
        wandb.init(
            project=f"{args.dataset}_{args.method}",
            name=f"{args.experiment} {args.dataset} {args.method} {args.ta} {args.sa}",
            tags=[current_time],
            dir=wb_dir,
            config=vars(args),
        )

    wandb.define_metric("Validation loss", summary="min")
    return wandb


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser():
    p = argparse.ArgumentParser(
        description="NH-Fair benchmark: LVLM zero-shot evaluation"
    )

    # Core
    p.add_argument("--method", type=str, default="qwen")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--dataset", type=str, default="celeba")
    p.add_argument("--ta", type=str, default="0")
    p.add_argument("--sa", type=str, default="race")
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--save_path", type=str, default="debug")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--bs", type=int, default=128, help="batch size")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--experiment", type=str, default=None)

    # Dataset loader
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--augment", type=str, default="weak", choices=["weak", "strong"])
    p.add_argument("--load_memory", action="store_true", default=False)
    p.add_argument("--use_clip_tranform", type=int, default=0)

    # LVLM-specific
    p.add_argument("--image_direct", action="store_true", default=False)
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--prompt_style", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument(
        "--task_mode",
        type=str,
        default="classification",
        choices=["classification", "open_generation"],
    )
    p.add_argument(
        "--vlm_backend",
        type=str,
        default="transformers",
        choices=["transformers", "gateway"],
        help="transformers=local HF; gateway=OpenAI-compatible HTTP (set --llm_gateway_url). "
        "Default: transformers.",
    )
    p.add_argument(
        "--llm_gateway_url",
        type=str,
        default=None,
        help="OpenAI API base URL, e.g. http://127.0.0.1:8000/v1 (vLLM/TGI/compatible).",
    )
    p.add_argument(
        "--llm_gateway_api_key",
        type=str,
        default=None,
        help="Optional API key for the gateway (many local servers use EMPTY).",
    )
    p.add_argument(
        "--vllm_port",
        type=int,
        default=8000,
        help="If --llm_gateway_url is unset, gateway URL defaults to http://127.0.0.1:PORT/v1",
    )

    # Runtime
    p.add_argument("--sweep", action="store_true", default=False)
    p.add_argument("--nowandb", action="store_true", default=False)

    # Testing / debugging
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, truncate every split to at most this many samples (for smoke tests).",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_time = get_et_time()
    wb = _init_wandb(args, current_time)

    model_short = args.model.replace("/", "_")
    run_detail = f"{args.method}_{model_short}_prompt_{args.prompt_style}_bs_{args.bs}"
    if args.experiment is not None:
        run_detail += f"_{args.experiment}"

    args.parent_path = os.path.join(args.save_path, run_detail + f"_seed_{args.seed}")
    args.save_path = os.path.join(args.parent_path, current_time)

    os.makedirs(args.save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, "log.txt")
    sys.stdout = DualWriter(log_file_path)
    print(" ".join(sys.argv))
    print(args)
    print(f"save path: {args.save_path}")
    set_seed(args)

    try:
        evaluate(args, wb)
    except Exception as e:
        print(f"\nException during evaluation: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if hasattr(sys.stdout, "log"):
            sys.stdout.log.close()
            sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
