"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import argparse
import glob
import logging
import math
import os
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf

OmegaConf.register_resolver(
    "pi_op",
    lambda operation, expr=None: {
        "div": np.pi / eval(expr) if expr else np.pi,
        "mul": np.pi * eval(expr) if expr else np.pi,
        "raw": np.pi,
    }[operation],
)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def parse_args():
    """Parse command-line arguments including Hydra config overrides."""
    parser = argparse.ArgumentParser(description="Extra arguments parser")

    # Standard arguments
    parser.add_argument("--job", "-j", type=str, default="1056700", help="Job ID")
    parser.add_argument("--ckpt", "-c", type=int, default=200, help="Checkpoint ID")
    parser.add_argument("--on_local", action="store_true", help="Run on cluster")

    # Hydra config overrides (flexible handling)
    args, hydra_overrides = parser.parse_known_args()
    args.overrides = hydra_overrides  # Store unknown args as Hydra overrides

    return args


def find_job_folder(base_path, job_id):
    """Find the job-specific folder safely."""
    matching_folders = glob.glob(os.path.join(base_path, f"{job_id}*"))

    if not matching_folders:
        raise FileNotFoundError(
            f"No matching job folder found for ID: {job_id} in {base_path}"
        )
    if len(matching_folders) > 1:
        print(f"Warning: Multiple job folders found. Using {matching_folders[0]}.")

    return matching_folders[0]  # Return first match


def main():
    args = parse_args()

    # Determine checkpoint directory
    CKPT_PATH = "log/tomato_plate" if not args.on_local else "ckpts/"

    # Find job folder safely
    job_folder = find_job_folder(CKPT_PATH, args.job)

    # Determine config and checkpoint paths
    cfg_path = (
        os.path.join(job_folder, ".hydra/config.yaml")
        if not args.on_local
        else os.path.join(job_folder, "config.yaml")
    )
    ckpt_path = (
        os.path.join(job_folder, f"checkpoint/state_{args.ckpt}.pt")
        if not args.on_local
        else os.path.join(job_folder, f"state_{args.ckpt}.pt")
    )
    norm_stats_path = os.path.join(job_folder, "norm.npz")

    # Check if paths exist
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Load Hydra config
    cfg = OmegaConf.load(cfg_path)

    # Apply command-line overrides
    if args.overrides:
        override_dict = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, override_dict)

    # # add datetime to logdir
    # cfg.logdir = os.path.join(
    #     cfg.logdir, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # )
    # Debug output
    print(f"Using checkpoint: {ckpt_path}")

    # Initialize and run the agent
    cfg.gpu_id = 0
    cfg._target_ = "guided_dc.agent.ValLossAgent"
    cfg.policy.model_path = ckpt_path
    cfg.normalization_stats_path = norm_stats_path

    # Initialize environment
    np.set_printoptions(suppress=True, precision=3)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to: {cfg.seed}")

    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
