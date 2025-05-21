"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

# import pretty_errors
import logging
import math
import os
import signal
import sys

import hydra
import numpy as np
import torch
import torch.distributed as dist
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


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Process group destroyed.")


def signal_handler(sig, frame):
    print(f"Received signal: {sig}. Cleaning up...")
    cleanup()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Function to run in each process
def _main(cfg: OmegaConf):
    checkpoint_dir = os.path.join(cfg.logdir, "checkpoint")
    if os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir):
        log.info(f"Checkpoint directory {checkpoint_dir} already exists. Exiting.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        from torch.distributed import destroy_process_group, init_process_group

        def ddp_setup():
            # os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split(",")[0]
            # os.environ["MASTER_PORT"] = "29500"
            # os.environ["NCCL_DEBUG"] = "INFO"
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            torch.cuda.empty_cache()
            # torch.cuda.set_device(rank)
            init_process_group(backend="nccl")

        ddp_setup()
        cfg["gpu_id"] = int(os.environ["LOCAL_RANK"])
    else:
        cfg["gpu_id"] = 0

    logging.info(cfg)

    # Initialize and run the agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()

    if num_gpus > 1:
        destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "guided_dc/cfg"),
    config_name="train.yaml",
)
def main(cfg: OmegaConf):
    # # Launch training in multiple processes if needed
    # if torch.cuda.device_count() > 1:
    #     import torch.multiprocessing as mp
    #     mp.spawn(_main, args=(cfg,), nprocs=torch.cuda.device_count())
    # else:
    #     _main(0, cfg)  # Pass rank 0 when not using multiple processes

    _main(cfg)


if __name__ == "__main__":
    main()
