"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import argparse
import datetime
import glob
import itertools
import logging
import math
import os
import sys

import cv2
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf

from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode
from guided_dc.utils.io_utils import dict_to_omegaconf_format
from guided_dc.utils.video_utils import merge_rgb_array_videos


import logging

from openpi_client import websocket_client_policy

import cv2, os
import numpy as np

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

gdc_assets_dir = os.getenv("GDC_ASSETS_DIR")

def split_list(lst, k):
    """
    Splits a list into sublists of equal length k, dropping any leftover elements.
    Args:
        lst (list): The input list to split.
        k (int): The length of each sublist.
    Returns:
        list of lists: Sublists of length k.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    n = len(lst)
    leftover = n % k

    if leftover > 0:
        log.warning(
            f"Dropping {leftover} element(s) as the list cannot be evenly divided."
        )

    return [lst[i : i + k] for i in range(0, n - leftover, k)]

def parse_args():
    """Parse command-line arguments including Hydra config overrides."""
    parser = argparse.ArgumentParser(description="Extra arguments parser")

    # Standard arguments
    parser.add_argument("--job", "-j", type=str, default="1056700", help="Job ID")
    parser.add_argument("--eval_name", default=None, help="Evaluation job name")
    parser.add_argument("--eval_type", default="grid", help="Evaluation type")
    parser.add_argument("--save_video", action="store_true", help="Flag to save video")
    parser.add_argument("--remote_host", type=str)
    parser.add_argument("--remote_port", type=str)
    parser.add_argument("--policy_dir", type=str)

    # Randomization flags
    parser.add_argument("--num_eval_instances", type=int, default=10)
    parser.add_argument("--randomize_table_texture", "-tt", action="store_true")
    parser.add_argument("--randomize_ambient", "-a", action="store_true")
    parser.add_argument("--randomize_camera_pose", "-cp", action="store_true")
    parser.add_argument("--randomize_background", "-b", action="store_true")
    parser.add_argument("--randomize_distractor", "-dis", action="store_true")
    parser.add_argument("--randomize_directional", "-dir", action="store_true")

    # Hydra config overrides (flexible handling)
    args, hydra_overrides = parser.parse_known_args()
    args.overrides = hydra_overrides  # Store unknown args as Hydra overrides

    return args

def save_images(images, video_dir):
    for k, v in images.items():
        if len(v.shape) == 4:  # (batch_size, h, w, c)
            batch_size, h, w, c = v.shape
            n = math.isqrt(batch_size)  # Closest integer square root
            p = math.ceil(batch_size / n)  # Ensure all images fit

            # Pad if necessary
            pad_size = n * p - batch_size
            if pad_size > 0:
                pad = np.zeros((pad_size, h, w, c), dtype=v.dtype)
                v = np.concatenate([v, pad], axis=0)

            # Reshape and stack
            v = v.reshape(n, p, h, w, c).swapaxes(
                1, 2
            )  # Shape: (n, h, p, w, c)
            stacked_img = v.reshape(
                n * h, p * w, c
            )  # Merge into a single image

        else:
            stacked_img = v

        cv2.imwrite(
            os.path.join(video_dir, f"{k}.png"),
            cv2.cvtColor(stacked_img, cv2.COLOR_BGR2RGB),
        )

def main():
    args = parse_args()

    # policy = create_policy(args)
    # Connect to the policy server
    print(f"Connecting to policy server at {args.remote_host}:{args.remote_port}")
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host, args.remote_port
    )

    override_dict = None
    # Apply command-line overrides
    if args.overrides:
        override_dict = args.overrides

    hydra.initialize(config_path="../../guided_dc/cfg")  # Ensure it's the directory containing `cfg.yaml`
    cfg = hydra.compose(config_name="train", overrides=override_dict)
    OmegaConf.set_struct(cfg, False)  # Disable struct mode

    ckpt_num = args.policy_dir.split("/")[-1]
    dataset_name = args.policy_dir.split("/")[-2]

    # add datetime to logdir
    cfg.logdir = os.path.join(
       f"log/pi0/{dataset_name}", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Initialize and run the agent
    cfg.gpu_id = 0
    cfg._target_ = "guided_dc.agent.eval.eval_agent_sim.EvalAgentSimPi"

    # Initialize environment
    np.set_printoptions(suppress=True, precision=3)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to: {cfg.seed}")

    env_cfg = cfg.simulation
    variation_factors = env_cfg.randomization
    base_factors = variation_factors.base

    if not env_cfg.quiet:
        log.info(f"Loaded configuration: \n{OmegaConf.to_yaml(env_cfg)}")

    env_cfg.control_mode = "pd_joint_pos"

    env = gym.make(env_cfg.env_id, cfg=env_cfg)

    output_dir = env_cfg.record.output_dir
    # output_dir = "videos_pi0"

    if args.eval_name is not None:
        sub_dir = f"{args.job}_{args.eval_name}_{args.eval_name}/{ckpt_num}"
    else:
        sub_dir = f"{args.job}_{args.eval_type}/{ckpt_num}"

    output_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Save env_cfg to a yaml file
    cfg_name = "config.yaml"
    cfg_name = os.path.join(output_dir, cfg_name)
    with open(cfg_name, "w") as f:
        OmegaConf.save(env_cfg, f)
    env_cfg.record.output_dir = output_dir

    render_mode = env_cfg.render_mode
    if (output_dir and render_mode != "human") and args.save_video:
        log.info(f"Recording environment episodes to: {output_dir}")
        env = RecordEpisode(
            env,  # type: ignore
            max_steps_per_video=env._max_episode_steps,  # type: ignore
            **env_cfg.record,
        )

    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg, env, policy_client)

    def generate_eval_env():
        
        init_manip_pose = np.load(env_cfg.eval_base_manip_poses_file)
        init_goal_pose = np.load(env_cfg.eval_base_goal_poses_file)
        init_manip_pose = split_list(init_manip_pose, env_cfg.num_envs)
        init_goal_pose = split_list(init_goal_pose, env_cfg.num_envs)

        env_configs = []

        for key in list(variation_factors.keys()):
            if not getattr(args, f"randomize_{key}", False):
                del variation_factors[key]
            else:
                variation_factors[key] = variation_factors[key][:args.num_eval_instances]

        # Vary one factor at a time, while keeping others at their base values
        for k, v in variation_factors.items():
            for factor_value in v:
                for manip_pose, goal_pose in zip(init_manip_pose, init_goal_pose):
                    env_config = {
                        "manip_obj_pose": manip_pose,
                        "goal_obj_pose": goal_pose,
                        k: factor_value,
                    }
                    # Add base values for all other factors
                    for base_key, base_value in base_factors.items():
                        if base_key != k:
                            env_config[base_key] = base_value[
                                0
                            ]  # Assume single base value
                    env_configs.append(env_config)
        return env_configs

    eval_env_configs = generate_eval_env()

    total_num_success = 0
    total_num_trials = 0
    last_config = {}

    _num_trials = 1

    for _, eval_env_config in enumerate(eval_env_configs):
        video_dir = os.path.join(
            output_dir,
            "_".join(
                f"{k}_{(next((i for i, fv in enumerate(variation_factors[k]) if (isinstance(v, dict) and isinstance(fv, dict) and fv == v) or (isinstance(fv, np.ndarray) and np.array_equal(fv, v)) or fv == v), 'base'))}"  # Compare normal values
                if k in variation_factors
                and any(
                    (isinstance(v, dict) and isinstance(fv, dict) and fv == v)
                    or (isinstance(fv, np.ndarray) and np.array_equal(fv, v))
                    or (fv == v)
                    for fv in variation_factors[k]
                )
                else f"{k}_base"
                for k, v in eval_env_config.items()
                if k not in {"manip_obj_pose", "goal_obj_pose"}
            ),
        )

        if os.path.exists(video_dir):
            num_result_files = len(glob.glob(os.path.join(video_dir, "*.txt")))
            result_id = num_result_files
            if result_id >= 2:
                continue
        else:
            result_id = 0

        os.makedirs(video_dir, exist_ok=True)

        for trial in range(_num_trials):
            reconfigure = any(
                eval_env_config[k] != last_config.get(k)
                for k in eval_env_config
                if k not in {"manip_obj_pose", "goal_obj_pose", "camera_pose"}
            )
            eval_env_config.update({"reconfigure": reconfigure})
            obs, info = env.reset(options=eval_env_config)
            last_config = eval_env_config.copy()
            images = agent.process_sim_observation(obs)["image"]

            save_images(images, video_dir)


            env_states: dict = agent.run()
            if args.save_video:
                env.flush_video()  # type: ignore

            num_success = sum(env_states["success"])
            num_trials = len(env_states["success"])
            total_num_success += num_success
            total_num_trials += num_trials

            if args.save_video:
                video_name = os.path.join(
                    video_dir, f"{num_success}_{num_trials}_{result_id}_{trial}.mp4"
                )
                merge_rgb_array_videos(
                    input_path=output_dir,
                    output_video=video_name,
                    num_videos=env_cfg.num_envs,
                    fps=30,
                )
            file_name = os.path.join(
                video_dir, f"{num_success}_{num_trials}_{result_id}_{trial}.txt"
            )
            with open(file_name, "w") as f:
                f.write("")

            states_name = os.path.join(
                video_dir, f"env_states_{result_id}_{trial}.yaml"
            )
            with open(states_name, "w") as f:
                OmegaConf.save(
                    OmegaConf.create(dict_to_omegaconf_format(env_states)), f
                )

    with open(
        os.path.join(output_dir, f"{total_num_success / total_num_trials * 100}%.txt"),
        "w",
    ) as f:
        f.write(
            f"Total number of success: {total_num_success}, Total number of trials: {total_num_trials}"
        )

    env.close()



if __name__ == "__main__":
    main()
