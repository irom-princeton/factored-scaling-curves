import math
import os

import gymnasium as gym
import hydra
import numpy as np
import sapien
import torch
import torchvision.transforms as T
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from omegaconf import OmegaConf

from guided_dc.envs.tasks.peg_insertion import PegInsertionSideEnv
from guided_dc.utils.hdf5_utils import (
    save_dict_to_hdf5,
)
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

img_size = (256, 256)
resize_transform = T.Resize(img_size)


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

    # eef_pos_quat = raw_obs[0]["extra"]["tcp_pose"].cpu().numpy()
    # # conver quaternion to euler angles
    # eef_pos_euler = np.zeros((eef_pos_quat.shape[0], 6))
    # eef_pos_euler[:, :3] = eef_pos_quat[:, :3]
    # eef_pos_euler[:, 3:] = quaternion_to_euler_xyz(eef_pos_quat[:, 3:])

    images = {}
    images["0"] = (
        resize_tensor_images(raw_obs[0]["sensor_data"]["base_camera"]["rgb"])
        .cpu()
        .numpy()
    )
    images["2"] = (
        resize_tensor_images(raw_obs[0]["sensor_data"]["hand_camera"]["rgb"])
        .cpu()
        .numpy()
    )

    return joint_state, gripper_state, images


class RecordEpisode(gym.Wrapper):
    def __init__(
        self,
        env,
    ) -> None:
        super().__init__(env)

        self.clear_buffer()

        self._manip_obj_poses = []
        self._goal_obj_poses = []
        self._delta_qpos = []

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def base_env(self):
        return self.env.unwrapped

    def clear_buffer(self):
        self._joint_states = []
        self._gripper_states = []
        self._wrist_imgs = []
        self._side_imgs = []
        self._actions = []
        self._success = []
        self._forces = []
        self._step_count = 0

    def save_episode(self, save_path, only_video=False):
        if len(self._joint_states) > 0 and len(self._actions) > 0:
            assert len(self._joint_states) == len(self._actions)
            assert len(self._wrist_imgs) == len(self._actions)
            assert len(self._side_imgs) == len(self._actions)

            # Save the trajectory data per environment
            data_dict = {
                "observation": {
                    "image": {
                        "0": np.array(self._side_imgs),
                        "2": np.array(self._wrist_imgs),
                    },
                    "robot_state": {
                        "joint_positions": np.array(self._joint_states),
                        "gripper_position": np.array(self._gripper_states),
                        "forces": np.array(self._forces),
                    },
                },
                "action": {
                    "joint_position": np.array(self._actions)[:, :-1],
                    "gripper_position": np.array(self._actions)[:, -1:],
                },
            }

            if not only_video:
                save_dict_to_hdf5(
                    save_path,
                    data_dict,
                )

            stack_videos_horizontally(
                data_dict["observation"]["image"]["0"],
                data_dict["observation"]["image"]["2"],
                save_path.replace(".h5", ".mp4"),
            )
        self.clear_buffer()

    def save_poses(self, save_path):
        np.save(save_path + "manip_obj_poses.npy", self._manip_obj_poses)
        np.save(save_path + "goal_obj_poses.npy", self._goal_obj_poses)
        np.save(save_path + "delta_qpos.npy", self._delta_qpos)

        self._manip_obj_poses = []
        self._goal_obj_poses = []
        self._delta_qpos = []

    def reset(
        self,
        seed,
        options,
        **kwargs,
    ):
        obs, info = super().reset(seed=seed, options=options, **kwargs)
        return obs, info

    def record_pose(self, options):
        self._manip_obj_poses.append(options["manip_obj_pose"][0])
        self._goal_obj_poses.append(options["goal_obj_pose"][0])
        self._delta_qpos.append(options["delta_qpos"][0])

    def step(self, action):
        obs = self.env.get_obs()
        force = torch.norm(self.env.goal_obj.get_net_contact_forces()).item()
        js, gs, img = process_sim_observation(
            obs
        )  # (num_envs, 7), (num_envs, 1), (num_envs, 2, 84, 84), (num_envs, 6)
        self._joint_states.append(js)
        self._gripper_states.append(gs)
        self._wrist_imgs.append(img["2"].squeeze(0))
        self._side_imgs.append(img["0"].squeeze(0))
        self._forces.append(force)

        obs, rew, terminated, truncated, info = super().step(action)

        self._actions.append(action)
        self._success.append(info["success"])
        if self._step_count % 100 == 0:
            print(f"Step: {self._step_count}")
        if self._step_count >= 300:
            terminated = True
        self._step_count += 1

        return obs, rew, terminated, truncated, info

    def close(self):
        self.env.close()


def get_option_values(
    get_options_range,
    factor,
    visual_factors,
    visual_factors_base,
    discrete_factor_idx=None,
    num_train_variations=5,
    num_eval_variations=10,
):
    options_range = get_options_range()
    options_value = {}

    def get_base_values(v):
        return np.array(v["base"]) + np.random.uniform(
            -np.array(v["base_range"]),
            np.array(v["base_range"]),
            size=len(v["base"]) if hasattr(v["base"], "__len__") else 1,
        )

    def get_discrete_factor_values(v):
        total_variations = num_train_variations + num_eval_variations
        values = np.array(v["base"]) + np.linspace(
            -np.array(v["factor_range"]), np.array(v["factor_range"]), total_variations
        )
        train_values = values[:: total_variations // num_train_variations]
        return train_values

    def get_continuous_factor_values(v):
        return np.array(v["base"]) + np.random.uniform(
            -np.array(v["factor_range"]),
            np.array(v["factor_range"]),
            size=len(v["base"]) if hasattr(v["base"], "__len__") else 1,
        )

    if factor in list(visual_factors.keys()):
        options_value[factor] = visual_factors[factor][discrete_factor_idx]

    for k in visual_factors_base:
        if factor != k:
            options_value[k] = visual_factors_base[k][0]

    for k, v in options_range.items():
        if factor == "obj_pose":
            if k in ("manip_obj_pose", "goal_obj_pose"):
                options_value[k] = get_continuous_factor_values(v)
            else:
                options_value[k] = get_base_values(v)
        elif factor == "delta_qpos":
            if k == factor:
                options_value[k] = get_continuous_factor_values(v)
            else:
                options_value[k] = get_base_values(v)
        else:
            if k == factor:
                options_value[k] = get_discrete_factor_values(v)[discrete_factor_idx]
            else:
                options_value[k] = get_base_values(v)
    # Post processing to align with simulation interface

    options_value["camera_pose"] = [
        {
            "eye": options_value["camera_pose"][:3],
            "target": options_value["camera_pose"][3:],
        }
    ]

    options_value["delta_qpos"] = options_value["delta_qpos"][None, :]
    options_value["manip_obj_pose"][-1] = math.sqrt(
        1 - options_value["manip_obj_pose"][3] ** 2
    )
    options_value["goal_obj_pose"][-1] = math.sqrt(
        1 - options_value["goal_obj_pose"][3] ** 2
    )
    options_value["manip_obj_pose"] = [options_value["manip_obj_pose"]]
    options_value["goal_obj_pose"] = [options_value["goal_obj_pose"]]
    options_value["table_height"] = options_value["table_height"].item()
    options_value["load_table_struc"] = False

    return options_value


@hydra.main(
    config_path=os.path.join(os.getcwd(), "guided_dc/cfg/simulation"),
    config_name="peg_insertion_mp",
    version_base=None,
)
def main(cfg):
    cfg.num_envs = 1
    cfg.obs_mode = "rgb"
    cfg.control_mode = "pd_joint_pos"
    cfg.render_mode = "sensors"
    cfg.reward_mode = "dense"
    cfg.sim_backend = "cpu"
    cfg.render_backend = "cpu"
    env = RecordEpisode(gym.make(cfg.env_id, cfg=cfg))

    base_visual_factors = cfg.randomization.base
    visual_factors = cfg.randomization

    folder = "data/raw_data/sim_peg"

    num_demos_per_variation = 25
    num_variations = 4
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
    # factors = [cfg.factor]

    last_config = {}

    for factor in factors:
        img_idx = 0
        for i in range(num_variations):
            trial = 0
            while trial < num_demos_per_variation:
                options = get_option_values(
                    env.get_options_range,
                    factor,
                    visual_factors,
                    base_visual_factors,
                    discrete_factor_idx=i,
                    num_train_variations=num_variations,
                    num_eval_variations=10,
                )
                print(
                    options["table_height"],
                    options["radii"],
                    options["lengths"],
                    options["centers"],
                    options["camera_pose"],
                )
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
                if trial == 0:
                    reconfigure = True
                last_config = options.copy()
                options["reconfigure"] = reconfigure
                obs, _ = env.reset(
                    seed=None,
                    options=options,
                )

                trial_idx = i * num_demos_per_variation + trial

                os.makedirs(f"{folder}/{factor}", exist_ok=True)
                save_path = f"{folder}/{factor}/{trial_idx}.h5"

                img_idx += 1

                res = solve(env, debug=False, vis=False)
                if res == -1:
                    env.save_episode(
                        f"{folder}/{factor}/{img_idx}_failed.h5",
                        only_video=True,
                    )
                else:
                    print(res[4]["success"])
                    if res[4]["success"]:
                        trial += 1
                        env.save_episode(save_path)
                        env.record_pose(options)
                    else:
                        env.save_episode(
                            f"{folder}/{factor}/{img_idx}_failed.h5",
                            only_video=True,
                        )
                for _ in range(100):
                    # Create a large random tensor
                    x = torch.randn(10000, 10000, device="cuda")
                    y = torch.matmul(x, x)
                    torch.cuda.synchronize()
        env.save_poses(f"{folder}/{factor}/")
    env.close()


def noise(p=0.3, scale=0.02):
    _flag = torch.bernoulli(torch.tensor(p)).item()
    if isinstance(scale, float):
        noise = ((torch.rand(1, 7) - 0.5) * scale) * _flag
    else:
        noise = ((torch.rand(1, 7) - 0.5) * torch.tensor(scale)[None, :]) * _flag
    # random_noise = (torch.rand(1, 7) - 0.5) * 0.04
    # random_noise[0, 1] = torch.rand(1).item() * 0.06
    return noise


def noise2(p=0.3):
    _flag = torch.bernoulli(torch.tensor(p)).item()
    noise = (torch.rand(1, 7) - 0.5) * 0.04
    noise[0, 1] = (torch.rand(1).item() - 0.5) * 0.06
    noise = noise * _flag
    print(noise)
    return noise


def solve(
    env: PegInsertionSideEnv,
    debug=False,
    vis=False,
):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.manip_obj)
    approaching = np.array([0, 0, -1])
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )

    peg_init_pose = env.manip_obj.pose

    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([-max(0.05, env.peg_half_sizes[0, 0] / 2 + 0.01), 0, 0.02])
    grasp_pose = grasp_pose * (offset)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    _reach_first = torch.bernoulli(torch.tensor(1.0)).item()
    if _reach_first:
        reach_pose = grasp_pose * (
            sapien.Pose(
                [np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02), -0.05]
            )
        )

        # reach_pose = torch.tensor(np.concatenate([reach_pose.p, reach_pose.q]))[None, :]

        # res = planner.move_to_pose_with_RRTConnect(
        #     reach_pose + noise(p=0.8, scale=0.06), rrt_range=0.23
        # )
        # res = planner.move_to_pose_with_screw(reach_pose + noise(p=0.0, scale=0.03))
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1:
            return res

        # # Refine the grasping pose
        # for _ in range(1):
        #     delta_pose = env.agent.tcp.pose.inv() * reach_pose  # Compute correction
        #     refined_reach_pose = delta_pose * reach_pose
        #     res = planner.move_to_pose_with_RRTConnect(refined_reach_pose)
        #     if res == -1:
        #         return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    grasp_pose = torch.tensor(np.concatenate([grasp_pose.p, grasp_pose.q]))[None, :]

    # res = planner.move_to_pose_with_RRTConnect(
    #     grasp_pose + noise(p=1.0, scale=[0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]),
    #     rrt_range=0.3 if not _reach_first else 0.2,
    # )
    res = planner.move_to_pose_with_screw(
        grasp_pose + noise(p=0.0, scale=[0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]),
    )
    if res == -1:
        return res

    # # Refine the grasping pose
    # for _ in range(1):
    #     delta_pose = env.agent.tcp.pose.inv() * grasp_pose  # Compute correction
    #     refined_grasp_pose = delta_pose * grasp_pose
    #     res = planner.move_to_pose_with_RRTConnect(refined_grasp_pose)
    #     if res == -1:
    #         return res

    # res = planner.move_to_pose_with_RRTConnect(grasp_pose)
    # if res == -1:
    #     return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Align Peg
    # -------------------------------------------------------------------------- #

    # align the peg with the hole
    insert_pose = env.goal_pose * peg_init_pose.inv() * grasp_pose
    _add_noise = torch.bernoulli(torch.tensor(0.75)).item()
    if _add_noise:
        depth_offset = np.random.uniform(-0.01, 0.02)
        noise_offset = sapien.Pose(
            (
                np.array([depth_offset - env.peg_half_sizes[0, 0], 0, -0.02])
                + noise(p=1.0)[0, :3].numpy()
            ).tolist()
        )
        pre_insert_pose_noise = insert_pose * (noise_offset)

        offset = sapien.Pose([-env.peg_half_sizes[0, 0], 0, -0.02])
        pre_insert_pose = insert_pose * (offset)

        # res = planner.move_to_pose_with_RRTConnect(
        #     pre_insert_pose.raw_pose + noise2(p=0.3), rrt_range=0.23
        # )
        res = planner.move_to_pose_with_screw(pre_insert_pose_noise.raw_pose)
        if res == -1:
            return res
        # refine the insertion pose
        while True:
            delta_pose = (
                env.goal_pose
                * (sapien.Pose([-env.peg_half_sizes[0, 0], 0, 0]))
                * env.manip_obj.pose.inv()
            )
            if torch.norm(delta_pose.p) < 0.001:
                break
            pre_insert_pose_noise = delta_pose * pre_insert_pose_noise
            res = planner.move_to_pose_with_screw(pre_insert_pose_noise)
            if res == -1:
                return res

    else:
        offset = sapien.Pose(
            [
                np.random.uniform(-0.01, 0.01) - env.peg_half_sizes[0, 0],
                0,
                -0.02 + np.random.uniform(-0.005, 0.005),
            ]
        )
        pre_insert_pose = insert_pose * (offset)
        res = planner.move_to_pose_with_screw(pre_insert_pose.raw_pose)
        if res == -1:
            return res

    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(
        insert_pose * (sapien.Pose([0.05, 0, 0])),
        interpolate_factor=2,
        early_termination=False,
    )
    if res == -1:
        return res
    planner.close()
    return res


if __name__ == "__main__":
    main()
