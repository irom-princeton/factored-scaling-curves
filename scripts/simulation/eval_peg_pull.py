"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import argparse
import glob
import logging
import math
import os
import re
import sys

import cv2
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf

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


def parse_args():
    """Parse command-line arguments including Hydra config overrides."""
    parser = argparse.ArgumentParser(description="Extra arguments parser")

    # Standard arguments
    parser.add_argument("--job", "-j", type=str, default="1056700", help="Job ID")
    parser.add_argument("--ckpt", "-c", type=int, default=None, help="Checkpoint ID")
    parser.add_argument("--on_local", action="store_true", help="Run on cluster")
    parser.add_argument("--eval_name", default=None, help="Evaluation job name")
    parser.add_argument("--eval_type", default="grid", help="Evaluation type")
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--save_video", action="store_true", help="Flag to save video")

    # Randomization flags
    parser.add_argument("--num_eval_instances", type=int, default=10)
    parser.add_argument("--randomize_table_texture", "-tt", action="store_true")
    parser.add_argument("--randomize_camera_pose", "-cp", action="store_true")
    parser.add_argument("--randomize_background", "-b", action="store_true")
    parser.add_argument("--randomize_distractor", "-dis", action="store_true")
    parser.add_argument("--randomize_directional", "-dir", action="store_true")
    parser.add_argument("--randomize_table_height", "-th", action="store_true")
    parser.add_argument("--randomize_delta_qpos", "-dq", action="store_true")
    parser.add_argument("--randomize_obj_pose", "-op", action="store_true")

    # Hydra config overrides (flexible handling)
    args, hydra_overrides = parser.parse_known_args()
    args.overrides = hydra_overrides  # Store unknown args as Hydra overrides

    return args


def find_job_folder(base_path, job_id):
    """Find the job-specific folder safely."""
    matching_folders = glob.glob(os.path.join(base_path, f"{job_id}_*"))

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
        assert match is not None, (
            f"Failed to extract checkpoint number from {ckpt_path}"
        )
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
    if args.env_name == "peg_insertion":
        target_cls = "guided_dc.agent.eval.eval_agent_sim.EvalAgentPeg"
    elif args.env_name == "pull_cube_tool":
        target_cls = "guided_dc.agent.eval.eval_agent_sim.EvalAgentPull"
    else:
        target_cls = "guided_dc.agent.eval.eval_agent_sim.EvalAgentSim"
    cfg._target_ = target_cls
    cfg.policy.model_path = ckpt_path
    cfg.normalization_stats_path = (
        f"{job_folder}/norm.npz"
        if os.path.exists(f"{job_folder}/norm.npz")
        else f"{os.environ.get('GDC_DATA_DIR')}/{cfg.task.dataset_name}/norm.npz"
    )
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
    cfg.ordered_obs_keys += ["forces"] if "with_force" in cfg.task.dataset_name else []
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
    base_visual_factors = env_cfg.randomization.base
    visual_factors = env_cfg.randomization

    if not env_cfg.quiet:
        log.info(f"Loaded configuration: \n{OmegaConf.to_yaml(env_cfg)}")

    env_cfg.control_mode = cfg.action_space

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

    total_num_success = 0
    total_num_trials = 0
    last_config = {}

    eval_base_manip_poses = np.load(env_cfg.eval_base_manip_poses_file)
    eval_base_goal_poses = np.load(env_cfg.eval_base_goal_poses_file)
    eval_factor_manip_poses = np.load(env_cfg.eval_factor_manip_poses_file)
    eval_factor_goal_poses = np.load(env_cfg.eval_factor_goal_poses_file)

    eval_base_delta_qpos = np.load(env_cfg.eval_base_delta_qpos_file)
    eval_factor_delta_qpos = np.load(env_cfg.eval_factor_delta_qpos_file)

    num_pose_sets = len(eval_base_manip_poses) // env_cfg.num_envs
    num_variations = args.num_eval_instances

    factors = [
        "table_texture",
        "directional",
        "background",
        "distractor",
        "camera_pose",
        "table_height",
        "delta_qpos",
        "obj_pose",
    ]

    if not args.randomize_table_texture:
        factors.remove("table_texture")
    if not args.randomize_directional:
        factors.remove("directional")
    if not args.randomize_background:
        factors.remove("background")
    if not args.randomize_distractor:
        factors.remove("distractor")
    if not args.randomize_camera_pose:
        factors.remove("camera_pose")
    if not args.randomize_table_height:
        factors.remove("table_height")
    if not args.randomize_delta_qpos:
        factors.remove("delta_qpos")
    if not args.randomize_obj_pose:
        factors.remove("obj_pose")

    for factor in factors:
        for i in range(num_variations):
            video_dir = os.path.join(output_dir, f"{factor}_{i}")
            if os.path.exists(video_dir):
                num_result_files = len(glob.glob(os.path.join(video_dir, "*.txt")))
                result_id = num_result_files
                if result_id >= num_pose_sets:
                    continue
            else:
                result_id = 0
            os.makedirs(video_dir, exist_ok=True)

            for pose_set_idx in range(num_pose_sets):
                if factor != "obj_pose":
                    manip_obj_pose = eval_base_manip_poses[
                        pose_set_idx * env_cfg.num_envs : (pose_set_idx + 1)
                        * env_cfg.num_envs
                    ]
                    goal_obj_pose = eval_base_goal_poses[
                        pose_set_idx * env_cfg.num_envs : (pose_set_idx + 1)
                        * env_cfg.num_envs
                    ]
                else:
                    manip_obj_pose = eval_factor_manip_poses[
                        i * 2 * env_cfg.num_envs + pose_set_idx * env_cfg.num_envs : i
                        * 2
                        * env_cfg.num_envs
                        + (pose_set_idx + 1) * env_cfg.num_envs
                    ]
                    goal_obj_pose = eval_factor_goal_poses[
                        i * 2 * env_cfg.num_envs + pose_set_idx * env_cfg.num_envs : i
                        * 2
                        * env_cfg.num_envs
                        + (pose_set_idx + 1) * env_cfg.num_envs
                    ]

                if factor != "delta_qpos":
                    delta_qpos = eval_base_delta_qpos[
                        pose_set_idx * env_cfg.num_envs : (pose_set_idx + 1)
                        * env_cfg.num_envs
                    ]
                else:
                    delta_qpos = eval_factor_delta_qpos[
                        i * 2 * env_cfg.num_envs + pose_set_idx * env_cfg.num_envs : i
                        * 2
                        * env_cfg.num_envs
                        + (pose_set_idx + 1) * env_cfg.num_envs
                    ]

                options = env.get_option_values(
                    factor,
                    visual_factors,
                    base_visual_factors,
                    manip_obj_pose=manip_obj_pose,
                    goal_obj_pose=goal_obj_pose,
                    delta_qpos=delta_qpos,
                    discrete_factor_idx=i,
                    num_train_variations=5,
                    num_eval_variations=num_variations,
                )
                print(options)
                reconfigure = any(
                    not np.array_equal(options[k], last_config.get(k))
                    if isinstance(options[k], np.ndarray)
                    else options[k] != last_config.get(k)
                    for k in options
                    if k
                    not in {
                        "manip_obj_pose",
                        "goal_obj_pose",
                        "camera_pose",
                        "delta_qpos",
                    }
                )
                if i == 0 and pose_set_idx == 0:
                    reconfigure = True
                last_config = options.copy()
                options["reconfigure"] = reconfigure

                obs, info = env.reset(options=options)
                last_config = options.copy()
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

                # if "pull_percentage" in env_states:
                #     num_success = env_states["pull_percentage"].sum()
                #     num_trials = len(env_states["success"])
                # else:
                num_success = sum(env_states["success"])
                num_trials = len(env_states["success"])
                total_num_success += num_success
                total_num_trials += num_trials

                if args.save_video:
                    video_name = os.path.join(
                        video_dir,
                        f"{num_success}_{num_trials}_{result_id}_{pose_set_idx}.mp4",
                    )
                    merge_rgb_array_videos(
                        input_path=output_dir,
                        output_video=video_name,
                        num_videos=env_cfg.num_envs,
                        fps=30,
                    )
                file_name = os.path.join(
                    video_dir,
                    f"{num_success}_{num_trials}_{result_id}_{pose_set_idx}.txt",
                )
                with open(file_name, "w") as f:
                    f.write("")

                states_name = os.path.join(
                    video_dir, f"env_states_{result_id}_{pose_set_idx}.yaml"
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
