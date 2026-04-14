"""Training CLI for the NH-Fair benchmark (supervised methods)."""

import argparse
import os
import sys
import traceback

import torch
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler

from release_benchmark.datasets.common import add_sampled_to_labeled
from release_benchmark.datasets.dataset import load_dataset
from release_benchmark.utils.common import DualWriter, get_et_time, set_seed

# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------


def train(args, wb):
    if args.method == "randaug":
        args.augment = "strong"
    train_set, val_set, test_set, sensitive_attributes, num_classes = load_dataset(args)
    args.sensitive_attributes = sensitive_attributes
    args.num_classes = num_classes

    if getattr(args, "max_samples", 0) > 0:
        n = args.max_samples

        def _truncate(ds):
            ds.indices = ds.indices[:n]
            for attr in ("targets_bin", "group_weights", "groups_idx"):
                v = getattr(ds, attr, None)
                if v is not None:
                    setattr(ds, attr, v[:n])

        if isinstance(train_set, list):
            for ds in train_set:
                _truncate(ds)
        else:
            _truncate(train_set)
        _truncate(val_set)
        _truncate(test_set)

    if args.method == "resample":
        weights = train_set.get_weights(resample_which=args.resample_mode)
        g = torch.Generator()
        g.manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=g
        )
        train_loader = DataLoader(
            train_set, batch_size=args.bs, sampler=sampler, num_workers=args.num_workers
        )
        args.method = "erm"
    elif args.method == "fis":
        train_set, unlabeled_trainset = train_set
        train_loader = DataLoader(
            train_set,
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
        )
        unlabeled_train_loader = DataLoader(
            unlabeled_trainset,
            batch_size=args.fis_bs,
            shuffle=False,
            num_workers=args.num_workers,
        )
    elif args.method == "groupdro":
        g = torch.Generator()
        g.manual_seed(args.seed)
        args.groupdro_group_count = train_set.group_counts()[1]
        weights = train_set.get_weights(resample_which="group")
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=g
        )
        train_loader = DataLoader(
            train_set, batch_size=args.bs, sampler=sampler, num_workers=args.num_workers
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.bs, num_workers=args.num_workers, shuffle=True
        )

    val_loader = DataLoader(
        val_set, batch_size=args.bs, num_workers=args.num_workers, shuffle=False
    )
    test_loader = DataLoader(
        test_set, batch_size=args.bs, num_workers=args.num_workers, shuffle=False
    )

    from release_benchmark.methods.registry import get_train_method

    Method = get_train_method(args.method)
    method = Method(args)

    if args.method == "fis":
        method.unlabeled_train_loader = unlabeled_train_loader

    val_loss, log_dict = method.validate(val_loader, -1, args)
    best_val_loss = val_loss
    best_AUC = log_dict.get("Overall AUC", 0)
    best_ACC = log_dict["Overall Acc"]
    patience = 0
    last_epoch = None

    for epoch in range(args.epochs):
        if args.method in ("dfr", "oxonfair_method"):
            loaders = [train_loader, val_loader, test_loader]
            train_loss, val_loss, val_log_dict, test_loss, test_log_dict = method.train(
                loaders, epoch, args
            )
        else:
            train_loss = method.train(train_loader, epoch, args)
            val_loss, val_log_dict = method.validate(val_loader, epoch, args)
            test_loss, test_log_dict = method.test(test_loader, epoch, args)

        if (
            epoch >= args.fis_warm
            and args.method == "fis"
            and "selected_indices" in val_log_dict
        ):
            selected_indices = val_log_dict["selected_indices"]
            train_set, unlabeled_trainset = add_sampled_to_labeled(
                train_set, unlabeled_trainset, selected_indices
            )

        log_info = {
            "epoch": epoch,
            "Training loss": train_loss,
            "Validation loss": val_loss,
            "Validation Accuracy": val_log_dict["Overall Acc"],
            "Validation Worst Accuracy": val_log_dict["worst_acc"],
            "Validation Accuracy Gap": val_log_dict["gap_acc"],
            "Validation AUC": val_log_dict.get("Overall AUC", 0),
            "Validation DP": val_log_dict["DP"],
            "Validation EqOpp": val_log_dict.get(
                "EqOpp1", val_log_dict.get("EqOppAvg")
            ),
            "Validation EqOdd": val_log_dict["EqOdd"],
            "Test loss": test_loss,
            "Test Accuracy": test_log_dict["Overall Acc"],
            "Test Worst Accuracy": test_log_dict["worst_acc"],
            "Test Accuracy Gap": test_log_dict["gap_acc"],
            "Test AUC": test_log_dict.get("Overall AUC", 0),
            "Test DP": test_log_dict["DP"],
            "Test EqOpp": test_log_dict.get("EqOpp1", test_log_dict.get("EqOppAvg")),
            "Test EqOdd": test_log_dict["EqOdd"],
        }

        for key in ("worst_AUC", "gap_AUC"):
            if key in val_log_dict:
                log_info[f"Validation {key}"] = val_log_dict[key]
                log_info[f"Test {key}"] = test_log_dict[key]

        if wb is not None:
            wb.log(log_info)

        val_flag = False
        if "ham" in args.dataset or "fitz" in args.dataset:
            if val_log_dict.get("Overall AUC", 0) > best_AUC:
                best_AUC = val_log_dict["Overall AUC"]
                val_flag = True
        else:
            if val_log_dict["Overall Acc"] > best_ACC:
                best_ACC = val_log_dict["Overall Acc"]
                val_flag = True

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_flag = True

        if val_flag:
            torch.save(
                {"model": method.model.state_dict(), "epoch": epoch},
                os.path.join(args.save_path, "best.pth"),
            )
            torch.save(
                {"epoch": epoch}, os.path.join(args.save_path, f"best_{epoch}.pt")
            )
            if last_epoch is not None:
                old = os.path.join(args.save_path, f"best_{last_epoch}.pt")
                if os.path.exists(old):
                    os.remove(old)
            last_epoch = epoch
            patience = 0
        else:
            patience += 1

        if method.fe_scheduler is not None and args.StepLR_size > 1:
            method.fe_scheduler.step()
        if patience >= args.max_patience:
            print("---early stop---")
            break


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------


def _init_wandb(args, current_time):
    """Initialize Weights & Biases for the run. Returns the wb module or None."""
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


def _build_run_detail(args):
    """Build a descriptive run-detail string from args."""
    parts = [
        f"optim_{args.optim}",
        f"scheduler_{args.scheduler}{args.StepLR_size}",
        f"lr_{args.lr}",
        f"pretrain_{args.pretrain}",
        f"bs_{args.bs}",
        f"wd_{args.weight_decay}",
        f"aug_{args.augment}",
        f"model_{args.model}",
        f"mom_{args.momentum}",
    ]

    method_parts = {
        "gapreg": [f"diff_{args.diff_metric}", f"lam_{args.diff_lambda}"],
        "mcdp": [
            f"diff_{args.diff_metric}",
            f"lam_{args.diff_lambda}",
            f"temp_{args.diff_temperature}",
        ],
        "laftr": [
            f"aud_{args.aud_steps}",
            f"cls_{args.class_coeff}",
            f"fair_{args.fair_coeff}",
            f"var_{args.model_var}",
        ],
        "mixup": [
            f"alpha_{args.mixup_alpha}",
            f"lam_{args.mixup_lam}",
            f"mode_{args.mixup_mode}",
        ],
        "fis": [
            f"ratio_{args.fis_ratio}",
            f"metric_{args.fis_metric}",
            f"warm_{args.fis_warm}",
        ],
        "bm": [f"mode_{args.bm_mode}"],
        "fscl": [f"gnorm_{args.group_norm}"],
        "dfr": [
            f"ntv_{args.dfr_notrain_val}",
            f"tcwt_{args.dfr_tune_class_weights_train}",
            f"ref_{args.dfr_ref}",
            f"mode_{args.dfr_mode}",
        ],
        "resample": [f"mode_{args.resample_mode}"],
        "oxonfair_method": [f"mode_{args.oxonfair_mode}"],
        "groupdro": [f"alpha_{args.groupdro_alpha}", f"gamma_{args.groupdro_gamma}"],
        "clip_sfid": [
            f"thr_{args.sfid_threshold}",
            f"prune_{args.sfid_image_prune_num}",
        ],
        "clip_fairer": [
            f"tzi_{args.clipfairer_tau_z_i:.3f}",
            f"ti_{args.clipfairer_tau_i:.3f}",
            f"tt_{args.clipfairer_tau_t:.3f}",
            f"tzt_{args.clipfairer_tau_z_t:.3f}",
            f"rff_{args.clipfairer_rff_dim}",
            f"sig_{args.clipfairer_sigma_max}",
            f"gi_{args.clipfairer_gamma_i:.1e}",
            f"gt_{args.clipfairer_gamma_t:.1e}",
        ],
    }

    if args.method in method_parts:
        parts.extend(method_parts[args.method])

    if args.experiment is not None:
        parts.append(args.experiment)

    return "_".join(parts)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser():
    p = argparse.ArgumentParser(description="NH-Fair benchmark: supervised training")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=128, help="batch size")
    p.add_argument("--dataset", type=str, default="celeba")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--method", type=str, default="erm")
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--save_path", type=str, default="debug")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument(
        "--optim",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw", "adagrad", "adadelta"],
    )
    p.add_argument("--scheduler", type=str, default="step")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--StepLR_size", type=int, default=30)
    p.add_argument("--max_patience", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--no_progress", action="store_true", default=False)
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument("--ta", type=str, default="0")
    p.add_argument("--sa", type=str, default="race")
    p.add_argument("--pretrain", type=int, default=0, choices=[0, 1])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--load_memory", action="store_true", default=False)
    p.add_argument("--freeze", type=int, default=0)
    p.add_argument("--augment", type=str, default="weak", choices=["weak", "strong"])
    p.add_argument("--img_size", type=int, default=224)

    # LAFTR
    p.add_argument("--aud_steps", type=int, default=1)
    p.add_argument("--class_coeff", type=float, default=1.0)
    p.add_argument("--fair_coeff", type=float, default=0.1)
    p.add_argument("--model_var", type=str, default="laftr-dp")

    # Gap regularization / MCDP
    p.add_argument("--diff_metric", type=str, default="dp")
    p.add_argument("--diff_lambda", type=float, default=1.0)
    p.add_argument("--diff_temperature", type=float, default=1.0)

    # Mixup
    p.add_argument("--mixup_alpha", type=float, default=1.0)
    p.add_argument("--mixup_lam", type=float, default=5)
    p.add_argument("--mixup_mode", type=str, default="group")

    # Resample
    p.add_argument(
        "--resample_mode", type=str, default="group", choices=["group", "balanced"]
    )

    # FIS
    p.add_argument("--fis_ratio", type=float, default=0.2)
    p.add_argument("--fis_warm", type=int, default=5)
    p.add_argument("--fis_metric", type=str, default="dp")
    p.add_argument("--fis_bs", type=int, default=128)

    # BM
    p.add_argument("--bm_mode", type=str, default="uw")

    # FSCL+
    p.add_argument("--group_norm", type=int, default=0, choices=[0, 1])

    # GroupDRO
    p.add_argument("--groupdro_alpha", type=float, default=20)
    p.add_argument("--groupdro_gamma", type=float, default=20)
    p.add_argument("--groupdro_btl", type=int, default=0, choices=[0, 1])

    # DFR
    p.add_argument("--dfr_notrain_val", type=int, default=0, choices=[0, 1])
    p.add_argument(
        "--dfr_tune_class_weights_train", type=int, default=0, choices=[0, 1]
    )
    p.add_argument("--dfr_ref", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument(
        "--dfr_mode", type=str, default="validation", choices=["validation", "train"]
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the pretrained checkpoint used by post-processing methods such as DFR.",
    )

    # OxonFair
    p.add_argument(
        "--oxonfair_fairness_metrics",
        type=str,
        nargs="+",
        default=["dp", "eopp", "eodd"],
    )
    p.add_argument(
        "--oxonfair_performance_metrics",
        type=str,
        nargs="+",
        default=["accuracy", "balanced_accuracy"],
    )
    p.add_argument("--oxonfair_num_retrains", type=int, default=20)
    p.add_argument("--oxonfair_use_validation", type=int, default=1, choices=[0, 1])
    p.add_argument("--oxonfair_lambda_fairness", type=float, default=1.0)
    p.add_argument("--oxonfair_mode", type=str, default="accuracy_noharm")

    # CLIP / VLM
    p.add_argument("--use_clip_tranform", type=int, default=0)
    p.add_argument("--fine_tune", type=int, default=0)

    # SFID
    p.add_argument("--sfid_threshold", type=float, default=0.7)
    p.add_argument("--sfid_image_prune_num", type=int, default=50)

    # Fairer CLIP
    p.add_argument("--clipfairer_tau_z_i", type=float, default=0.7)
    p.add_argument("--clipfairer_tau_i", type=float, default=0.7)
    p.add_argument("--clipfairer_tau_t", type=float, default=0.1)
    p.add_argument("--clipfairer_tau_z_t", type=float, default=0.1)
    p.add_argument("--clipfairer_rff_dim", type=int, default=4000)
    p.add_argument("--clipfairer_sigma_max", type=int, default=4000)
    p.add_argument("--clipfairer_gamma_i", type=float, default=3e-5)
    p.add_argument("--clipfairer_gamma_t", type=float, default=3e-4)

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

    current_time = get_et_time()
    wb = _init_wandb(args, current_time)

    run_detail = _build_run_detail(args)
    args.parent_path = os.path.join(args.save_path, run_detail + f"_seed_{args.seed}")
    args.save_path = os.path.join(args.parent_path, current_time)

    os.makedirs(args.save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, "log.txt")
    sys.stdout = DualWriter(log_file_path)
    print(" ".join(sys.argv))
    print(args)
    print(f"save path: {args.save_path}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args)

    try:
        train(args, wb)
    except Exception as e:
        print(f"\nException during training: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if hasattr(sys.stdout, "log"):
            sys.stdout.log.close()
            sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
