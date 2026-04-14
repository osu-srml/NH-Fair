"""Create and launch Weights & Biases hyperparameter sweeps (reads YAML templates from the package)."""

import argparse
import os
import subprocess
from pathlib import Path

import wandb
import yaml

from release_benchmark.paths import sweep_config_dir

_gpu_status = {}


def get_free_gpus(threshold=30):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    free = []
    for idx, memory in enumerate(result.stdout.strip().split("\n")):
        if not memory.strip():
            continue
        used, total = map(int, memory.split(","))
        usage = (used / total) * 100
        if usage < threshold and not _gpu_status.get(idx, False):
            free.append(idx)
    return free


def launch_agent(sweep_path, gpu_id, screen=False, count=0, name="sweep"):
    _gpu_status[gpu_id] = True
    if screen:
        session = f"{name}_gpu{gpu_id}"
        extra = f" --count {count}" if count > 0 else ""
        cmd = f"screen -dmS {session} bash -lc 'CUDA_VISIBLE_DEVICES={gpu_id} wandb agent {sweep_path}{extra}'"
        subprocess.run(cmd, shell=True, check=False)
        print(f"launched screen session {session}")
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = ["wandb", "agent", sweep_path]
    if count > 0:
        cmd += ["--count", str(count)]
    print(f"launching {' '.join(cmd)} on gpu {gpu_id}")
    return subprocess.Popen(cmd, env=env)


def load_sweep_template(config_dir: Path, method: str):
    cfg_path = config_dir / f"{method}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_program_from_template(template: dict) -> str:
    program = str(template.get("program", "train2.py"))
    if "zeroshot" in program:
        return "release_benchmark.cli.zeroshot"
    return "release_benchmark.cli.train"


def make_release_command(program_module: str, method: str):
    base = ["${env}", "python", "-m", program_module, "${args}", "--method", method]
    if program_module.endswith("zeroshot"):
        base += ["--image_direct"]
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Create and launch W&B sweeps for release_benchmark"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--sa", type=str, required=True)
    parser.add_argument("--ta", type=str, required=True)
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--sweep_name", type=str, default="release")
    parser.add_argument("--savepath", type=str, default="results_sweep")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--StepLR_size", type=int, default=30)
    parser.add_argument("--fis_bs", type=int, default=128)
    parser.add_argument("--no_progress", action="store_true", default=False)
    parser.add_argument("--load_memory", action="store_true", default=False)
    parser.add_argument("--print", action="store_true", default=False)
    parser.add_argument("--screen", action="store_true", default=False)
    parser.add_argument(
        "--gpu", type=str, default=None, help="comma-separated GPU ids or 'auto'"
    )
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--slurm", type=int, default=0)
    args = parser.parse_args()

    wb_key = os.getenv("WANDB_API_KEY")
    if wb_key:
        wandb.login(key=wb_key)

    project = args.project_name or f"{args.dataset}_{args.method}"
    wandb.init(project=project)

    config_dir = sweep_config_dir()
    cfg = load_sweep_template(config_dir, args.method)

    module = choose_program_from_template(cfg)
    cfg["program"] = f"python -m {module}"
    cfg["name"] = (
        f"{args.dataset} {args.method} {args.ta} {args.sa} sweep {args.sweep_name}"
    )
    cfg["dir"] = (
        f"./wandb/{args.dataset}/{args.method}/{args.ta}/{args.sa} sweep {args.sweep_name}"
    )

    command = make_release_command(module, args.method)
    command += [
        "--dataset",
        args.dataset,
        "--save_path",
        f"./{args.savepath}/{args.method}/{args.dataset}/{args.sa}-{args.ta}/",
        "--sa",
        args.sa,
        "--ta",
        args.ta,
        "--img_size",
        str(args.img_size),
        "--StepLR_size",
        str(args.StepLR_size),
        "--gpu",
        "0",
        "--sweep",
        "--fis_bs",
        str(args.fis_bs),
    ]
    if args.model is not None:
        command += ["--model", args.model]
    if args.no_progress:
        command += ["--no_progress"]
    if args.load_memory:
        command += ["--load_memory"]

    cfg["command"] = command

    sweep_id = wandb.sweep(cfg, project=project)
    sweep_path = (
        f"{wandb.api.default_entity}/{project}/{sweep_id}"
        if wandb.api.default_entity
        else f"{project}/{sweep_id}"
    )
    print(f"Created sweep: {sweep_id}")
    print(f"Agent command: wandb agent {sweep_path}")

    if args.print:
        return

    if args.slurm > 0:
        slurm_script = config_dir / "slurm_sweep_count.sh"
        for _ in range(args.slurm):
            subprocess.run(
                ["sbatch", str(slurm_script), "--sweep_id", sweep_id], check=False
            )
        return

    gpu_arg = args.gpu
    if gpu_arg:
        if gpu_arg == "auto":
            gpu_ids = get_free_gpus()
        else:
            gpu_ids = [int(x) for x in gpu_arg.split(",") if x.strip()]
    else:
        gpu_ids = get_free_gpus()

    procs = []
    for gid in gpu_ids:
        p = launch_agent(
            sweep_path,
            gid,
            screen=args.screen,
            count=(args.count // max(1, len(gpu_ids))) if args.count else 0,
            name=project,
        )
        if p is not None:
            procs.append((p, gid))

    for p, gid in procs:
        p.wait()
        _gpu_status[gid] = False


if __name__ == "__main__":
    main()
