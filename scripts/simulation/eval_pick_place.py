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

import re
from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode
from guided_dc.utils.io_utils import dict_to_omegaconf_format
from guided_dc.utils.video_utils import merge_rgb_array_videos

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
    parser.add_argument("--ckpt", "-c", type=int, default=None, help="Checkpoint ID")
    parser.add_argument("--on_local", action="store_true", help="Run on cluster")
    parser.add_argument("--eval_name", default=None, help="Evaluation job name")
    parser.add_argument("--eval_type", default="grid", help="Evaluation type")
    parser.add_argument("--save_video", action="store_true", help="Flag to save video")
    parser.add_argument("--env_name", default="tomato_plate")

    # Randomization flags
    parser.add_argument("--num_eval_instances", type=int, default=10)
    parser.add_argument("--strat", type=str, default="strat5", help="Eval strategy")
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
    CKPT_PATH = f"log/{args.env_name}" if not args.on_local else "ckpts/"

    # Find job folder safely
    job_folder = find_job_folder(CKPT_PATH, args.job)

    # Determine config and checkpoint paths
    cfg_path = (
        os.path.join(job_folder, ".hydra/config.yaml")
        if not args.on_local
        else os.path.join(job_folder, "config.yaml")
    )
    if args.ckpt is None:
        if not args.on_local:
            ckpt_folder = os.path.join(job_folder, "checkpoint")
        else:
            ckpt_folder = job_folder
        ckpt_files = glob.glob(os.path.join(ckpt_folder, "state_*.pt"))
        assert len(ckpt_files) == 1, "Expected exactly one checkpoint file."
        ckpt_path = ckpt_files[0]

        match = re.search(r"state_(\d+)\.pt", os.path.basename(ckpt_path))
        assert match is not None, f"Failed to extract checkpoint number from {ckpt_path}"
        ckpt_number = int(match.group(1))
        args.ckpt = ckpt_number
    else:
        ckpt_path = (
            os.path.join(job_folder, f"checkpoint/state_{args.ckpt}.pt")
            if not args.on_local
            else os.path.join(job_folder, f"state_{args.ckpt}.pt")
        )

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

    # add datetime to logdir
    # cfg.logdir = os.path.join(
    #     cfg.logdir, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # )
    # Debug output
    print(f"Using checkpoint: {ckpt_path}")

    # Initialize and run the agent
    cfg.gpu_id = 0

    if args.env_name == "tomato_plate":
        target_cls = "EvalAgentSim"
    elif args.env_name == "peg_insertion":
        target_cls = "EvalAgentPeg"
    else:
        raise NotImplementedError(f"Unknown environment name: {args.env_name}")

    cfg._target_ = f"guided_dc.agent.eval.eval_agent_sim.{target_cls}"
    cfg.policy.model_path = ckpt_path
    cfg.normalization_stats_path = f"{job_folder}/norm.npz"
    cfg.n_steps = 80
    cfg.act_steps = 8
    if cfg.task.num_views == 2:
        cfg.camera_indices = ["0", "2"]
    else:
        assert cfg.task.num_views == 1
        cfg.camera_indices = ["0"]

    # Set up control and proprio
    if cfg.task.dataset_name.startswith("eefg"):
        cfg.ordered_obs_keys = ["cartesian_position", "gripper_position"]
    elif cfg.task.dataset_name.startswith("jsg"):
        cfg.ordered_obs_keys = ["joint_positions", "gripper_position"]
    else:
        raise NotImplementedError
    if "_eefg" in cfg.task.dataset_name:
        cfg.action_space = "pd_ee_pose"
    elif "_jsg" in cfg.task.dataset_name:
        cfg.action_space = "pd_joint_pos"
    else:
        raise NotImplementedError

    # Initialize environment
    np.set_printoptions(suppress=True, precision=3)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to: {cfg.seed}")

    env_cfg = cfg.simulation
    variation_factors = env_cfg.randomization
    base_factors = variation_factors.pop("base")

    # env_cfg = OmegaConf.load("guided_dc/cfg/simulation/peg_insertion_sim.yaml")
    # variation_factors = OmegaConf.load(
    #     "guided_dc/cfg/simulation/randomization/eval_sim_peg.yaml"
    # )
    # base_factors = variation_factors.base
    print(env_cfg.num_envs)

    if not env_cfg.quiet:
        log.info(f"Loaded configuration: \n{OmegaConf.to_yaml(env_cfg)}")

    env_cfg.control_mode = cfg.action_space
    # if env_cfg.shader == "rt":
    #     env_cfg.num_envs = 1
    #     env_cfg.enable_shadow = False

    env = gym.make(env_cfg.env_id, cfg=env_cfg)

    output_dir = env_cfg.record.output_dir

    if args.eval_name:
        sub_dir = f"{args.job}_{args.eval_name}/{args.ckpt}"
    else:
        sub_dir = f"{args.job}/{args.ckpt}"
    output_dir = os.path.join(output_dir, sub_dir)
    cfg.logdir = output_dir
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

    # Log environment details if verbose output is enabled
    if not env_cfg.quiet:
        log.info(f"Observation space: {env.observation_space}")
        log.info(f"Action space: {env.action_space}")
        log.info(f"Control mode: {env.control_mode}")  # type: ignore
        log.info(f"Reward mode: {env.reward_mode}")  # type: ignore

    if (
        (not render_mode == "human")
        and (env_cfg.record.save_trajectory)
        and (args.save_video)
    ):
        raise NotImplementedError

    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg, env)

    def generate_eval_env(strat="strat5"):
        init_manip_pose = []
        init_goal_pose = []
        if args.eval_type == "dataset":
            if args.on_local:
                pick_poses = np.load("data/pick_obj_poses.npy")
                place_poses = np.load("data/place_obj_poses.npy")
            else:
                pick_poses = np.load(
                    "data/sim_default_with_shadow/tomato_plate_dec1/pick_obj_poses.npy"
                )
                place_poses = np.load(
                    "data/sim_default_with_shadow/tomato_plate_dec1/place_obj_poses.npy"
                )
            for i in range(len(pick_poses)):
                init_manip_pose.append(pick_poses[i])
                init_goal_pose.append(place_poses[i])
                pick_pose, place_pose = pick_poses[i], place_poses[i]
                for _ in range(3):
                    new_pick_pose = pick_pose + np.concatenate(
                        [np.random.uniform(-0.05, 0.05, size=2), [0, 0, 0, 0]]
                    )
                    new_place_pose = place_pose + np.concatenate(
                        [np.random.uniform(-0.05, 0.05, size=2), [0, 0, 0, 0]]
                    )
                    init_manip_pose.append(new_pick_pose)
                    init_goal_pose.append(new_place_pose)
        elif args.eval_type == "grid":
            if hasattr(env_cfg, 'eval_manip_pose_file') and hasattr(env_cfg, 'eval_goal_pose_file'):
                init_manip_pose = np.load(env_cfg.eval_manip_pose_file)
                init_goal_pose = np.load(env_cfg.eval_goal_pose_file)
            else:
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

        if strat == "strat1":
            # Full combinatorial pairing
            all_pairs = list(
                itertools.product(
                    *variation_factors.values(), zip(init_manip_pose, init_goal_pose)
                )
            )
            for *factor_values, (manip_pose, goal_pose) in all_pairs:
                env_configs.append(
                    {
                        "manip_obj_pose": manip_pose,
                        "goal_obj_pose": goal_pose,
                        **dict(zip(variation_factors.keys(), factor_values)),
                    }
                )
        elif strat == "strat2":
            # Distribute init poses evenly among factors
            factor_keys = list(variation_factors.keys())
            factor_lists = list(variation_factors.values())

            for i, (manip_pose, goal_pose) in enumerate(
                zip(init_manip_pose, init_goal_pose)
            ):
                factor_values = [
                    factor_lists[j][i % len(factor_lists[j])]
                    for j in range(len(factor_keys))
                ]
                env_configs.append(
                    {
                        "manip_obj_pose": manip_pose,
                        "goal_obj_pose": goal_pose,
                        **dict(zip(factor_keys, factor_values)),
                    }
                )
        elif strat == "strat3":
            # Vary one at a time, keeping the rest as base
            all_pairs = []
            for k, v in variation_factors.items():
                pairs = list(
                    itertools.product(
                        v,
                        zip(init_manip_pose, init_goal_pose),
                    )
                )
                for factor_values, (manip_pose, goal_pose) in pairs:
                    if factor_values is None:
                        continue
                    env_configs.append(
                        {
                            "manip_obj_pose": manip_pose,
                            "goal_obj_pose": goal_pose,
                            k: factor_values,
                        }
                    )
        elif strat == "strat4":
            # Vary one factor at a time, while randomly sampling values for others
            rng = np.random.default_rng(
                seed=args.seed if hasattr(args, "seed") else None
            )

            for k, v in variation_factors.items():
                for factor_value in v:
                    for manip_pose, goal_pose in zip(init_manip_pose, init_goal_pose):
                        random_factors = {
                            key: rng.choice(values)
                            for key, values in variation_factors.items()
                            if key != k
                        }
                        env_configs.append(
                            {
                                "manip_obj_pose": manip_pose,
                                "goal_obj_pose": goal_pose,
                                k: factor_value,
                                **random_factors,
                            }
                        )
        elif strat == "strat5":
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

    eval_env_configs = generate_eval_env(strat=args.strat)

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

        # if os.path.exists(video_dir):
        #     num_result_files = len(glob.glob(os.path.join(video_dir, "*.txt")))
        #     if num_result_files >= _num_trials:
        #         continue

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
