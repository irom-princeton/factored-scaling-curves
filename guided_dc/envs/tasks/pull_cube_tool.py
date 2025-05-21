from typing import Any, ClassVar, Dict, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from sapien import physx
from transforms3d.euler import euler2quat

from guided_dc.envs.agents import PandaWristCamPeg
from guided_dc.envs.factor_env import TwoObjectEnv
from guided_dc.envs.scenes.tabletop_scene_builder import TabletopSceneBuilder


@register_env("CustomPullCubeTool-v1", max_episode_steps=600)
class CustomPullCubeToolEnv(TwoObjectEnv):
    """
    **Task Description**
    Given an L-shaped tool that is within the reach of the robot, leverage the
    tool to pull a cube that is out of it's reach

    **Randomizations**
    - The cube's position (x,y) is randomized on top of a table in the region "<out of manipulator
    reach, but within reach of tool>". It is placed flat on the table
    - The target goal region is the region on top of the table marked by "<within reach of arm>"

    **Success Conditions**
    - The cube's xy position is within the goal region of the arm's base (marked by reachability)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PullCubeTool-v1_rt.mp4"

    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["panda_wristcam", "panda_wristcam_peg"]

    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[PandaWristCam, PandaWristCamPeg]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.2
    hook_length = 0.03
    width = 0.03
    height = 0.03
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, cfg):
        super().__init__(cfg)

    @property
    def _default_sensor_configs(self):
        # pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        # return [
        #     CameraConfig(
        #         "base_camera",
        #         pose=pose,
        #         width=128,
        #         height=128,
        #         fov=np.pi / 2,
        #         near=0.01,
        #         far=100,
        #     )
        # ]
        base_camera_config = CameraConfig(
            uid="base_camera",
            pose=Pose.create_from_pq(p=[0, 0, 0], q=[1, 0, 0, 0]),
            mount=self.cam_mount[0],
            width=256,
            height=256,
            near=0.01,
            far=100,
            fov=np.pi / 2,
        )

        # pose = sapien_utils.look_at([0.3, 0.35, 0.4], [0.1, 0.0, 0.0])
        return [
            base_camera_config,
            # CameraConfig(
            #     uid="hand_camera",
            #     pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
            #     width=128,
            #     height=128,
            #     fov=np.pi / 2,
            #     near=0.01,
            #     far=100,
            #     mount=self.agent.robot.links_map["camera_link"],
            # ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
            )
        ]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        hook_width = width * 2

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, hook_width, 0]),
            [hook_length / 2, hook_width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, hook_width, 0]),
            [hook_length / 2, hook_width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TabletopSceneBuilder(
                self, cfg=self.base_cfg.scene_builder
            )
            self.table_scene.build(options)
            self.cam_mount = [
                self.scene.create_actor_builder().build_kinematic("base_camera_mount")
            ]

            self.build_distractor(options)
            sapien.set_log_level("off")
            self._build_objects(options)

    def _build_objects(self, options):
        self.goal_obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        self.manip_obj = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )

    def _after_reconfigure(self, options: dict):
        with torch.device(self.device):
            if hasattr(self, "distractors"):
                for _, (distractor, _distractors) in self.distractors.items():
                    heights = []
                    for d in _distractors:
                        collision_mesh = d.get_first_collision_mesh()
                        heights.append(-collision_mesh.bounding_box.bounds[0, 2])
                    distractor.height = common.to_tensor(heights)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # b = len(env_idx)
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    # 0,
                    0.04,
                    0.04,
                ]
            )
            delta_qpos = np.array(
                options.get(
                    "delta_qpos",
                    self._episode_rng.normal(0, 0.02, (len(env_idx), len(qpos))),
                )
            )

            qpos = delta_qpos + qpos
            qpos[:, -2:] = 0.04
            self.table_scene.initialize(
                env_idx,
                options={
                    # "robot_init_pos": [-0.615, 0, 0],
                    "robot_init_qpos": qpos,
                    "robot_init_euler": [0, 0, 0],
                    "table_orientation": euler2quat(0, 0, np.pi / 2),
                    "table_height": options.get("table_height", 0),
                },
            )
            manip_obj_pose = options.get("manip_obj_pose", None)
            goal_obj_pose = options.get("goal_obj_pose", None)
            camera_pose = options.get("camera_pose", None)
            distractor_cfgs = options.get("distractor", None)
            table_height = torch.tensor(options.get("table_height", 0))
            if manip_obj_pose is None:
                b = len(env_idx)
                tool_xyz = torch.zeros((b, 3), device=self.device)
                tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
                tool_xyz[..., 2] = self.height / 2
                tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

                manip_obj_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
                cube_xyz = torch.zeros((b, 3), device=self.device)
                cube_xyz[..., 0] = (
                    self.arm_reach
                    + torch.rand(b, device=self.device) * (self.handle_length)
                    - 0.3
                )
                cube_xyz[..., 1] = torch.rand(b, device=self.device) * 0.3 - 0.25
                cube_xyz[..., 2] = self.cube_size / 2 + 0.015

                cube_q = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                    bounds=(-np.pi / 6, np.pi / 6),
                    device=self.device,
                )
                goal_obj_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
                self.manip_obj.set_pose(manip_obj_pose)
                self.goal_obj.set_pose(goal_obj_pose)
            else:
                assert goal_obj_pose is not None
                for i in range(len(manip_obj_pose)):
                    manip_obj_pose[i][2] = self.height / 2 + table_height.item()
                    goal_obj_pose[i][2] = self.cube_half_size + table_height.item()
                self.set_obj_pose(manip_obj_pose, self.manip_obj)
                self.set_obj_pose(goal_obj_pose, self.goal_obj)

            if camera_pose is not None:
                self.camera_pose = camera_pose
            if distractor_cfgs:
                for distractor_cfg in distractor_cfgs:
                    name = f"distractor-{distractor_cfg['obj_name']}"
                    distractor_pose = self.get_distractor_pose(
                        name,
                        min_distance=0.1,
                        x_min=-0.25,
                        x_max=-0.15,
                        y_min=-0.3,
                        y_max=0.3,
                    )
                    distractor_pose[:, 2] += table_height
                    self.distractors[name][0].set_pose(distractor_pose)

            if physx.is_gpu_enabled():
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()  # type: ignore
                self.scene.px.step()
                self.scene._gpu_fetch_all()
            else:
                self.scene.px.step()

        self.initial_cube_to_base_dist = torch.linalg.norm(
            self.goal_obj.pose.p[:, :2] - self.agent.robot.get_links()[0].pose.p[:, :2],
            dim=1,
        )
        print("Initialized episode.")

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            obs.update(
                cube_pose=self.goal_obj.pose.raw_pose,
                tool_pose=self.manip_obj.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        cube_pos = self.goal_obj.pose.p

        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        cube_to_base_dist = torch.linalg.norm(
            cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
        )

        target_dist = 0.47
        eps = 1e-6  # protects against divide-by-zero

        # --- success flag ---
        cube_pulled_close = cube_to_base_dist < target_dist

        # --- normalized progress: 0 → at initial pose, 1 → reached target ---
        denom = (self.initial_cube_to_base_dist - target_dist).clamp_min(
            eps
        )  # make safe
        pull_percentage = (self.initial_cube_to_base_dist - cube_to_base_dist) / denom
        pull_percentage = torch.clamp(pull_percentage, 0.0, 1.0)

        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "cube_progress": progress.mean(),
            "cube_distance": cube_to_workspace_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
            "pull_percentage": pull_percentage,
            "success_45": cube_to_base_dist < 0.45,
            "success_50": cube_to_base_dist < 0.50,
            "success_55": cube_to_base_dist < 0.55,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.goal_obj.pose.p
        tool_pos = self.manip_obj.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.manip_obj, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        if "success" in info:
            reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward

    def get_options_range(self):
        options = {}
        # add some extra options for the peg and box
        options["manip_obj_pose"] = {
            "base": [-0.1, 0, 0, 1, 0, 0, 0],
            "base_range": [0.04, 0.08, 0, 0.0, 0, 0, 0],
            "factor_range": [0.05, 0.1, 0, 0.0, 0, 0, 0],
        }
        options["goal_obj_pose"] = {
            "base": [0.2, 0, 0, 0.95, 0, 0, 0],
            "base_range": [0.04, 0.08, 0, 0.04, 0, 0, 0],
            "factor_range": [0.05, 0.1, 0, 0.05, 0, 0, 0],
        }

        options["delta_qpos"] = {
            "base": np.zeros(9),
            "base_range": np.ones(9) * 0.01,
            "factor_range": np.ones(9) * 0.02,
        }

        options["table_height"] = {
            "base": 0,
            "base_range": 0.0,
            "factor_range": 0.025,
        }

        options["camera_pose"] = {
            "base": [0.25, 0.3, 0.3, 0.1, 0.05, 0.0],
            "base_range": [0, 0, 0, 0, 0, 0],
            "factor_range": [0.05, 0.05, 0.05, 0, 0, 0],
        }

        return options

    def get_option_values(
        self,
        factor,
        visual_factors,
        visual_factors_base,
        manip_obj_pose,
        goal_obj_pose,
        delta_qpos,
        discrete_factor_idx=0,
        num_train_variations=5,
        num_eval_variations=10,
    ):
        options_range = self.get_options_range()
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
                -np.array(v["factor_range"]),
                np.array(v["factor_range"]),
                total_variations,
            )
            # train_values = values[:: total_variations // num_train_variations]
            eval_values = values[1 :: total_variations // num_eval_variations]
            return eval_values

        if factor in list(visual_factors.keys()):
            options_value[factor] = visual_factors[factor][discrete_factor_idx]

        for k in visual_factors_base:
            if factor != k:
                options_value[k] = visual_factors_base[k][0]

        for k, v in options_range.items():
            if k == "manip_obj_pose" or k == "goal_obj_pose" or k == "delta_qpos":
                continue
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
        options_value["manip_obj_pose"] = manip_obj_pose
        options_value["goal_obj_pose"] = goal_obj_pose
        options_value["delta_qpos"] = delta_qpos

        # options_value["lengths"] = np.repeat(options_value["lengths"], self.num_envs)
        # options_value["radii"] = np.repeat(options_value["radii"], self.num_envs)
        # options_value["centers"] = np.repeat(options_value["centers"], num_envs).reshape(num_envs, -1)
        options_value["table_height"] = options_value["table_height"].item()
        options_value["load_table_struc"] = False

        return options_value

    def generate_eval_poses(
        self, num_base_pairs=60, num_factor_pairs=600, visualize=True
    ):
        import matplotlib.pyplot as plt

        options = self.get_options_range()

        def sample_uniform(base, variation, num_samples):
            base = np.array(base)
            variation = np.array(variation)
            return base + np.random.uniform(
                -variation, variation, (num_samples, len(base))
            )

        base_manip_poses = sample_uniform(
            options["manip_obj_pose"]["base"],
            options["manip_obj_pose"]["base_range"],
            num_base_pairs,
        )

        base_goal_poses = sample_uniform(
            options["goal_obj_pose"]["base"],
            options["goal_obj_pose"]["base_range"],
            num_base_pairs,
        )
        factor_manip_poses = sample_uniform(
            options["manip_obj_pose"]["base"],
            options["manip_obj_pose"]["factor_range"],
            num_factor_pairs,
        )
        factor_goal_poses = sample_uniform(
            options["goal_obj_pose"]["base"],
            options["goal_obj_pose"]["factor_range"],
            num_factor_pairs,
        )

        base_manip_poses[:, -1] = np.sqrt(1 - base_manip_poses[:, 3] ** 2)
        base_goal_poses[:, -1] = np.sqrt(1 - base_goal_poses[:, 3] ** 2)
        factor_manip_poses[:, -1] = np.sqrt(1 - factor_manip_poses[:, 3] ** 2)
        factor_goal_poses[:, -1] = np.sqrt(1 - factor_goal_poses[:, 3] ** 2)

        base_delta_qpos = sample_uniform(
            options["delta_qpos"]["base"],
            options["delta_qpos"]["base_range"],
            num_base_pairs,
        )
        factor_delta_qpos = sample_uniform(
            options["delta_qpos"]["base"],
            options["delta_qpos"]["factor_range"],
            num_factor_pairs,
        )

        def visualize_pose_pairs(base_pairs, factor_pairs):
            plt.figure(figsize=(8, 6))

            # Extract XY positions
            base_manip_poses = base_pairs[0]
            base_goal_poses = base_pairs[1]
            factor_manip_poses = factor_pairs[0]
            factor_goal_poses = factor_pairs[1]
            base_manip_xy = base_manip_poses[:, :2]
            base_goal_xy = base_goal_poses[:, :2]
            factor_manip_xy = factor_manip_poses[:, :2]
            factor_goal_xy = factor_goal_poses[:, :2]

            # Plot base pairs
            plt.scatter(
                base_manip_xy[:, 0],
                base_manip_xy[:, 1],
                color="blue",
                label="Base Manip Pose",
                alpha=0.5,
            )
            plt.scatter(
                base_goal_xy[:, 0],
                base_goal_xy[:, 1],
                color="green",
                label="Base Goal Pose",
                alpha=0.5,
            )

            # Plot factor pairs
            plt.scatter(
                factor_manip_xy[:, 0],
                factor_manip_xy[:, 1],
                color="red",
                label="Factor Manip Pose",
                alpha=0.5,
            )
            plt.scatter(
                factor_goal_xy[:, 0],
                factor_goal_xy[:, 1],
                color="orange",
                label="Factor Goal Pose",
                alpha=0.5,
            )

            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend()
            plt.title("Visualization of Manip and Goal Object Poses")
            plt.savefig(
                f"data/eval_poses/pose_pairs_{num_base_pairs}_{num_factor_pairs}_pull.png"
            )

        np.save(
            f"data/eval_poses/eval_base_manip_poses_file_pull_{num_base_pairs}.npy",
            base_manip_poses,
        )
        np.save(
            f"data/eval_poses/eval_base_goal_poses_file_pull_{num_base_pairs}.npy",
            base_goal_poses,
        )
        np.save(
            f"data/eval_poses/eval_factor_manip_poses_file_pull_{num_factor_pairs}.npy",
            factor_manip_poses,
        )
        np.save(
            f"data/eval_poses/eval_factor_goal_poses_file_pull_{num_factor_pairs}.npy",
            factor_goal_poses,
        )
        np.save(
            f"data/eval_poses/eval_base_delta_qpos_file_pull_{num_base_pairs}.npy",
            base_delta_qpos,
        )
        np.save(
            f"data/eval_poses/eval_factor_delta_qpos_file_pull_{num_factor_pairs}.npy",
            factor_delta_qpos,
        )
        if visualize:
            visualize_pose_pairs(
                (base_manip_poses, base_goal_poses),
                (factor_manip_poses, factor_goal_poses),
            )
