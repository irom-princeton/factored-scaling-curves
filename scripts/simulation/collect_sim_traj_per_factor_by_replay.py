"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import logging
import math
import os
import sys
from collections import Counter

import gymnasium as gym
import hydra
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from guided_dc.utils.hdf5_utils import (
    load_hdf5,
    save_dict_to_hdf5,
)
from guided_dc.utils.pose_utils import quaternion_to_euler_xyz
from guided_dc.utils.video_utils import stack_videos_horizontally

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
log = logging.getLogger(__name__)
DATA_DIR = os.environ["GDC_DATA_DIR"]
ASSET_DIR = os.environ["GDC_ASSETS_DIR"]
img_size = (192, 192)
resize_transform = T.Resize(img_size)

all_traj_folders = {
    "lighting_new": 200,
    "camera_pose_new": 200,
    "distractor_new": 200,
    "background_new": 200,
    "table_texture_new": 200,
    "old_raw_data/camera_pose_only": 140,
    "old_raw_data/table_texture": 76,
    "old_raw_data/tomato_plate_dec1": 48,
    "old_raw_data/tomato_plate_jan12": 67,
    "old_raw_data/tomato_plate_jan24": 91,
    "old_raw_data/tomato_plate_trials-date": 52,
}


def get_factor_paths(factor_name, num_factors=5, traj_per_factor=300):
    real_traj_paths = []
    num_traj_available = 0
    for traj_folder in all_traj_folders:
        traj_paths = [
            os.path.join(f"data/{traj_folder}", traj_name)
            for traj_name in os.listdir(f"data/{traj_folder}")
            if traj_name.endswith(".h5")
        ]
        num_traj_available += len(traj_paths)
        real_traj_paths.extend(traj_paths)

    print("Number of avaliable trajectories:", num_traj_available)

    total_needed = num_factors * traj_per_factor

    if total_needed > 2 * num_traj_available:
        raise ValueError("Not enough elements to satisfy the constraints.")

    num_sampled_twice = total_needed - num_traj_available
    num_sampled_once = num_traj_available - num_sampled_twice

    # Step 1: Randomly shuffle the available trajectories
    np.random.shuffle(real_traj_paths)

    # Step 2: Split into sampled-once and sampled-twice sets
    sampled_once = real_traj_paths[:num_sampled_once]
    sampled_twice = real_traj_paths[
        num_sampled_once : num_sampled_once + num_sampled_twice
    ]

    # Step 3: Construct the full list with correct frequency
    full_list = np.array(sampled_once + sampled_twice * 2)

    # Step 4: Shuffle to randomize element distribution
    np.random.shuffle(full_list)

    # Step 5: Split into sublists
    sublists = [
        full_list[i * traj_per_factor : (i + 1) * traj_per_factor]
        for i in range(num_factors)
    ]

    # Flatten the list of lists into a single list of elements
    all_elements = np.concatenate(sublists)

    # Count occurrences of each element
    element_counts = Counter(all_elements)

    # Output verification
    print(
        "Each element appears at least once:",
        all(count >= 1 for count in element_counts.values()),
    )
    print(
        "No element appears more than twice:",
        all(count <= 2 for count in element_counts.values()),
    )
    print(f"Number of trajectories sampled once: {num_sampled_once}")
    print(f"Number of trajectories sampled twice: {num_sampled_twice}")

    # _keys = ["camera_pose", "distractor", "directional", "table_texture", "background"]
    _keys = ["table_height"]
    data_dict = {}

    # Verify the sublists (optionally print the sublists or their lengths)
    for idx, sublist in enumerate(sublists):
        print(f"Sublists {idx + 1}: {len(sublist)} elements")
        data_dict[_keys[idx]] = sublist

    return data_dict[factor_name]


def find_pick_and_place_times(gripper_position):
    pick_start = -1
    place_start = -1

    # Find the start of the last subsequence of 1
    for i in range(len(gripper_position) - 1, -1, -1):
        if gripper_position[i] == 1:
            pick_start = i
        elif pick_start != -1:
            # Found the start of the last subsequence of 1
            break

    # Find the start of the immediate subsequence of 0 after pick_start
    if pick_start != -1:
        for i in range(pick_start + 1, len(gripper_position)):
            if gripper_position[i] == 0:
                place_start = i
                break

    return pick_start, place_start


def transform_to_global(relative_pos, base_position, base_orientation):
    """
    Transforms a position from the robot base's frame to the global frame.

    :param relative_pos: np.array, shape (3,), position relative to the robot base
    :param base_position: np.array, shape (3,), position of the robot base in the global frame
    :param base_orientation: np.array, shape (3,)
    :return: np.array, shape (3,), position in the global frame
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_euler("xyz", base_orientation)  # Quaternion format: [x, y, z, w]
    rotation_matrix = rotation.as_matrix()

    # Transform the position
    global_pos = rotation_matrix @ relative_pos + base_position
    return global_pos


def resize_tensor_images(tensor_images):
    # tensor_images shape: (batch_size, 480, 640, 3)
    # Convert the tensor to (batch_size, 3, 480, 640) for PyTorch operations
    tensor_images = tensor_images.permute(
        0, 3, 1, 2
    )  # Convert to (batch_size, 3, 480, 640)

    resized_images = torch.stack([resize_transform(image) for image in tensor_images])
    # Convert back to (batch_size, 192, 192, 3)
    resized_images = resized_images.permute(0, 2, 3, 1)
    return resized_images


def process_sim_observation(raw_obs):
    if isinstance(raw_obs, dict):
        raw_obs = [raw_obs]
    joint_state = raw_obs[0]["agent"]["qpos"][:, :7].cpu().numpy()
    gripper_state = raw_obs[0]["agent"]["qpos"][:, 7:8].cpu().numpy()
    # assert (gripper_state <= 0.04).all(), gripper_state
    # gripper_state = 1 - gripper_state / 0.04

    eef_pos_quat = raw_obs[0]["extra"]["tcp_pose"].cpu().numpy()
    # conver quaternion to euler angles
    eef_pos_euler = np.zeros((eef_pos_quat.shape[0], 6))
    eef_pos_euler[:, :3] = eef_pos_quat[:, :3]
    eef_pos_euler[:, 3:] = quaternion_to_euler_xyz(eef_pos_quat[:, 3:])

    images = {}
    images["0"] = (
        resize_tensor_images(raw_obs[0]["sensor_data"]["sensor_0"]["rgb"]).cpu().numpy()
    )
    images["2"] = (
        resize_tensor_images(raw_obs[0]["sensor_data"]["hand_camera"]["rgb"])
        .cpu()
        .numpy()
    )

    return joint_state, gripper_state, images, eef_pos_euler


def step(env):
    """
    Take a step through the environment with an action. Actions are automatically clipped to the action space.

    If ``action`` is None, the environment will proceed forward in time without sending any actions/control signals to the agent
    """
    info = env.get_info()
    obs = env.get_obs(info)
    return (
        obs,
        info,
    )


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


def load_trajs(factor_name):
    base_position = np.array(
        [-0.547, -0.527, -0.143]
    )  # Position of the robot base in the global frame
    base_orientation = np.array([0, 0, np.pi / 4])

    action_list = []
    eef_action_list = []
    global_pick_pos_list = []
    global_place_pos_list = []
    pick_obj_timesteps = []
    place_obj_timesteps = []

    traj_paths = get_factor_paths(factor_name, num_factors=1, traj_per_factor=3)
    # traj_paths = traj_paths[270:]

    for traj_path in tqdm(traj_paths):
        try:
            real_traj_dict, _ = load_hdf5(
                file_path=traj_path,
                action_keys=[
                    "joint_position",
                    "gripper_position",
                    "cartesian_position",
                ],
                observation_keys=[
                    "joint_positions",
                    "gripper_position",
                    "cartesian_position",
                ],
                load_image=True,
            )
        except:
            continue

        pick_obj_timestep, place_obj_timestep = find_pick_and_place_times(
            real_traj_dict["action/gripper_position"]
        )
        relative_pick_pos = np.array(
            real_traj_dict["observation/robot_state/cartesian_position"][
                pick_obj_timestep
            ][:3]
        )
        global_pick_pos = transform_to_global(
            relative_pick_pos, base_position, base_orientation
        )

        relative_place_pos = np.array(
            real_traj_dict["observation/robot_state/cartesian_position"][
                place_obj_timestep
            ][:3]
        )
        global_place_pos = transform_to_global(
            relative_place_pos, base_position, base_orientation
        )

        real_js = real_traj_dict["observation/robot_state/joint_positions"]
        real_gs = real_traj_dict["observation/robot_state/gripper_position"]
        real_gs = (real_gs > 0.2).astype(np.float32)
        real_gs = -real_gs * 2 + 1
        # For the gripper position, we round values <0 to -1, and values >0 to 1
        for i in range(len(real_gs)):
            if real_gs[i] < 0:
                real_gs[i] = -1
            else:
                real_gs[i] = 1

        actions = np.concatenate(
            [
                real_js,
                real_gs[:, None],
            ],
            axis=1,
        )

        eef_actions = real_traj_dict["action/cartesian_position"]

        action_list.append(actions)
        eef_action_list.append(eef_actions)
        global_pick_pos_list.append(global_pick_pos)
        global_place_pos_list.append(global_place_pos)
        pick_obj_timesteps.append(pick_obj_timestep)
        place_obj_timesteps.append(place_obj_timestep)

        for _ in range(100):
            # Create a large random tensor
            x = torch.randn(10000, 10000, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()

        print(actions[:, -1])
    return (
        action_list,
        eef_action_list,
        global_pick_pos_list,
        global_place_pos_list,
        pick_obj_timesteps,
        place_obj_timesteps,
    )


def generate_eval_env(
    variation_factors,
    base_factors,
    manip_pose,
    goal_pose,
    eval_config_idx,
    only_pose=False,
    no_randomization=False,
):
    if only_pose:
        env_configs = [
            {
                "manip_obj_pose": manip_pose,
                "goal_obj_pose": goal_pose,
            }
        ]
    else:
        if no_randomization:
            env_config = {
                "manip_obj_pose": manip_pose,
                "goal_obj_pose": goal_pose,
            }
            for base_key, base_value in base_factors.items():
                env_config[base_key] = base_value[0]  # Assume single base value
            env_configs = [env_config]
        else:
            env_configs = []
            # Vary one factor at a time, while keeping others at their base values
            assert len(variation_factors) == 1
            for k, v in variation_factors.items():
                factor_value = v[eval_config_idx]
                env_config = {
                    "manip_obj_pose": manip_pose,
                    "goal_obj_pose": goal_pose,
                    k: factor_value,
                }
                # Add base values for all other factors
                for base_key, base_value in base_factors.items():
                    if base_key != k:
                        env_config[base_key] = base_value[0]  # Assume single base value
                env_configs.append(env_config)
    assert len(env_configs) == 1
    return env_configs


def pad_to_longest(traj_list):
    # Pad with the last element
    longest = max(len(traj) for traj in traj_list)
    last_idx = [len(traj) - 1 for traj in traj_list]
    for i, traj in enumerate(traj_list):
        if len(traj) < longest:
            traj_list[i] = np.pad(traj, ((0, longest - len(traj)), (0, 0)), mode="edge")
    return np.array(traj_list), np.array(last_idx)


@hydra.main(
    config_path=os.path.join(os.getcwd(), "guided_dc/cfg/simulation"),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg):
    env_cfg = cfg
    base_factors = env_cfg.randomization.base
    variation_factors = {cfg.factor: env_cfg.randomization[cfg.factor]}

    # Initialize environment
    np.set_printoptions(suppress=True, precision=3)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to: {cfg.seed}")

    env_cfg.control_mode = "pd_joint_pos"
    env = gym.make(env_cfg.env_id, cfg=env_cfg)

    (
        action_list,
        eef_action_list,
        global_pick_pos_list,
        global_place_pos_list,
        pick_obj_timesteps,
        place_obj_timesteps,
    ) = load_trajs(cfg.factor)
    action_list = split_list(action_list, env_cfg.num_envs)
    eef_action_list = split_list(eef_action_list, env_cfg.num_envs)
    global_pick_pos_list = split_list(global_pick_pos_list, env_cfg.num_envs)
    global_place_pos_list = split_list(global_place_pos_list, env_cfg.num_envs)
    pick_obj_timesteps = split_list(pick_obj_timesteps, env_cfg.num_envs)
    place_obj_timesteps = split_list(place_obj_timesteps, env_cfg.num_envs)

    for action_list_idx, (
        actions,
        eef_actions,
        global_pick_pos,
        global_place_pos,
        pick_obj_timestep,
        place_obj_timestep,
    ) in enumerate(
        zip(
            action_list,
            eef_action_list,
            global_pick_pos_list,
            global_place_pos_list,
            pick_obj_timesteps,
            place_obj_timesteps,
        )
    ):
        actions, end_indices = pad_to_longest(
            actions
        )  # (num_envs, max_len, 8), (num_envs,)
        # eef_actions, _ = pad_to_longest(eef_actions)
        global_pick_pos = np.array(global_pick_pos)
        global_place_pos = np.array(global_place_pos)

        pick_offset = np.zeros_like(global_pick_pos)
        place_offset = np.zeros_like(global_place_pos)
        traj_start_js = actions[:, 0, :-1]

        for trial in range(2):
            manip_obj_pose = np.zeros((env_cfg.num_envs, 6))
            goal_obj_pose = np.zeros((env_cfg.num_envs, 6))

            manip_obj_pose[:, :2] = global_pick_pos[:, :2] + pick_offset[:, :2]
            goal_obj_pose[:, :2] = global_place_pos[:, :2] + place_offset[:, :2]
            manip_obj_pose[:, 2] = 0
            goal_obj_pose[:, 2] = 0.01
            manip_obj_pose[:, 3:] = np.array([-1.57265, -0.0464526, 3.0965])
            goal_obj_pose[:, 3:] = np.array([1.569, 2.9, -0.001066])
            goal_obj_pose[:, 4] = np.random.uniform(
                -np.pi, np.pi, size=env_cfg.num_envs
            )

            eval_env_configs = generate_eval_env(
                variation_factors,
                base_factors,
                manip_obj_pose,
                goal_obj_pose,
                only_pose=True if trial == 0 else False,
                no_randomization=False,
                eval_config_idx=action_list_idx,
            )
            last_config = {}

            eval_env_config = eval_env_configs[0]

            pick_obj_poses = []
            place_obj_poses = []
            reconfigure = any(
                eval_env_config[k] != last_config.get(k)
                for k in eval_env_config
                if k not in {"manip_obj_pose", "goal_obj_pose", "camera_pose"}
            )
            eval_env_config.update({"reconfigure": reconfigure})
            obs, info = env.reset(options=eval_env_config)
            last_config = eval_env_config.copy()
            env_start_js = env.agent.robot.get_qpos().cpu().numpy()[:, :7]
            qpos_to_set = env.agent.robot.get_qpos().cpu().numpy()

            norm_diff = np.linalg.norm(traj_start_js - env_start_js, axis=1)
            mask = norm_diff > 0.1
            qpos_to_set[mask] = np.hstack(
                (traj_start_js[mask], np.full((mask.sum(), 2), 0.04))
            )
            env.agent.robot.set_qpos(qpos_to_set)
            obs, _ = step(env)

            if trial == 1:
                # Preallocate NumPy arrays for efficiency
                max_timesteps = actions.shape[1]

                jss = np.zeros(
                    (env_cfg.num_envs, max_timesteps, 7)
                )  # Assuming `js.shape[-1]` is known
                gss = np.zeros((env_cfg.num_envs, max_timesteps, 1))
                wrist_imgs = np.zeros(
                    (env_cfg.num_envs, max_timesteps, *img_size, 3), dtype=np.uint8
                )
                side_imgs = np.zeros(
                    (env_cfg.num_envs, max_timesteps, *img_size, 3), dtype=np.uint8
                )
                acts = np.zeros((env_cfg.num_envs, max_timesteps, 8))
                eef_poses = np.zeros((env_cfg.num_envs, max_timesteps, 6))
                success = np.zeros(
                    env_cfg.num_envs, dtype=bool
                )  # Use boolean for efficiency

            for timestep in range(actions.shape[1]):
                action = actions[:, timestep]  # (num_envs, 8)
                js, gs, img, eef_pose = process_sim_observation(
                    obs
                )  # (num_envs, 7), (num_envs, 1), (num_envs, 2, 84, 84), (num_envs, 6)

                # Change above to batch version
                _pick_flag = [
                    i for i, x in enumerate(pick_obj_timestep) if x == timestep
                ]
                if len(_pick_flag) > 0:
                    tcp_pos = env.agent.tcp.pose.p.cpu().numpy()
                    pick_offset[_pick_flag] = (
                        tcp_pos[_pick_flag] - global_pick_pos[_pick_flag]
                    )

                _place_flag = [
                    i for i, x in enumerate(place_obj_timestep) if x == timestep
                ]
                if len(_place_flag) > 0:
                    tcp_pos = env.agent.tcp.pose.p.cpu().numpy()
                    place_offset[_place_flag] = (
                        tcp_pos[_place_flag] - global_place_pos[_place_flag]
                    )

                obs, rew, terminated, truncated, info = env.step(action)
                if trial == 1:
                    mask = timestep <= end_indices
                    jss[mask, timestep] = js[mask]
                    gss[mask, timestep] = gs[mask]
                    side_imgs[mask, timestep] = img["0"][mask]
                    wrist_imgs[mask, timestep] = img["2"][mask]
                    acts[mask, timestep] = action[mask]
                    eef_poses[mask, timestep] = eef_pose[mask]

                    # Vectorized logical OR operation for success tracking
                    success |= terminated.cpu().numpy()

            if trial == 1:
                traj_folder = f"{DATA_DIR}/sim_new_variation/{cfg.factor}"
                os.makedirs(traj_folder, exist_ok=True)
                # Save the trajectory data per environment
                for i in range(env_cfg.num_envs):
                    pick_obj_poses.append(manip_obj_pose[i])
                    place_obj_poses.append(goal_obj_pose[i])
                    data_dict = {
                        "observation": {
                            "image": {
                                "0": side_imgs[i][: end_indices[i] + 1],
                                "2": wrist_imgs[i][: end_indices[i] + 1],
                            },
                            "robot_state": {
                                "joint_positions": jss[i][: end_indices[i] + 1],
                                "gripper_position": gss[i][: end_indices[i] + 1],
                                "cartesian_position": eef_poses[i][
                                    : end_indices[i] + 1
                                ],
                            },
                        },
                        "action": {
                            "joint_position": acts[i, :, :-1][: end_indices[i] + 1],
                            "gripper_position": acts[i, :, -1][: end_indices[i] + 1],
                            # "cartesian_position": eef_actions[i][:end_indices[i] + 1],
                        },
                        "manip_obj_pose": manip_obj_pose[i][: end_indices[i] + 1],
                        "goal_obj_pose": goal_obj_pose[i][: end_indices[i] + 1],
                        "pick_obj_timestep": pick_obj_timestep[i],
                        "place_obj_timestep": place_obj_timestep[i],
                    }
                    traj_idx = action_list_idx * env_cfg.num_envs + i
                    if success[i]:
                        save_dict_to_hdf5(
                            traj_folder + f"/{traj_idx}_sim.h5",
                            data_dict,
                        )
                        stack_videos_horizontally(
                            data_dict["observation"]["image"]["0"],
                            data_dict["observation"]["image"]["2"],
                            traj_folder + f"/{traj_idx}_sim.mp4",
                        )
                    else:
                        save_dict_to_hdf5(
                            traj_folder + f"/{traj_idx}_sim_failed.h5",
                            data_dict,
                        )
                        stack_videos_horizontally(
                            data_dict["observation"]["image"]["0"],
                            data_dict["observation"]["image"]["2"],
                            traj_folder + f"/{traj_idx}_sim_failed.mp4",
                        )
                np.save(traj_folder + "/pick_obj_poses.npy", pick_obj_poses)
                np.save(traj_folder + "/place_obj_poses.npy", place_obj_poses)

    env.close()


if __name__ == "__main__":
    main()
