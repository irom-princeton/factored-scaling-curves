from typing import Any, ClassVar, Dict, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose
from sapien import physx
from transforms3d.euler import euler2quat

from guided_dc.envs.agents import PandaWristCamPeg
from guided_dc.envs.factor_env import TwoObjectEnv
from guided_dc.envs.scenes.tabletop_scene_builder import TabletopSceneBuilder


def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


@register_env("CustomPegInsertionSide-v1", max_episode_steps=1000)
class PegInsertionSideEnv(TwoObjectEnv):
    """
    **Task Description:**
    Pick up a orange-white peg and insert the orange end into the box with a hole in it.

    **Randomizations:**
    - Peg half length is randomized between 0.085 and 0.125 meters. Box half length is the same value. (during reconfiguration)
    - Peg radius/half-width is randomized between 0.015 and 0.025 meters. Box hole's radius is same value + 0.003m of clearance. (during reconfiguration)
    - Peg is laid flat on table and has it's xy position and z-axis rotation randomized
    - Box is laid flat on table and has it's xy position and z-axis rotation randomized

    **Success Conditions:**
    - The white end of the peg is within 0.015m of the center of the box (inserted mid way).
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionSide-v1_rt.mp4"

    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["panda_wristcam", "panda_wristcam_peg"]  # type: ignore
    agent: Union[PandaWristCam, PandaWristCamPeg]
    _clearance = 0.01

    def __init__(self, cfg):
        super().__init__(cfg)

    @property
    def _default_sensor_configs(self):
        # pose = sapien_utils.look_at([0.25, -0.1, 0.3], [0.05, 0.0, 0.1])
        # return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]
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

        wrist_camera_config = CameraConfig(
            uid="hand_camera",
            pose=sapien.Pose(p=[0.01, -0.015, -0.01], q=[1, 0, -0.5, 0]),
            width=256,
            height=256,
            fov=np.pi / 2,
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["camera_link"],
        )

        return [base_camera_config, wrist_camera_config]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 256, 256, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

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

    def _after_reconfigure(self, options: dict):
        with torch.device(self.device):
            if hasattr(self, "distractors"):
                for _, (distractor, _distractors) in self.distractors.items():
                    heights = []
                    for d in _distractors:
                        collision_mesh = d.get_first_collision_mesh()
                        heights.append(-collision_mesh.bounding_box.bounds[0, 2])
                    distractor.height = common.to_tensor(heights)

    def _build_objects(self, options):
        lengths = options.get(
            "lengths", self._batched_episode_rng.uniform(0.085, 0.125)
        )
        radii = options.get("radii", self._batched_episode_rng.uniform(0.015, 0.025))
        centers = options.get(
            "centers", self._batched_episode_rng.uniform(-1, 1, size=(2,))
        )

        # make sure lengths and radii are the same size as num_envs
        if isinstance(lengths, (np.ndarray)):
            if len(lengths) == 1:
                lengths = np.full(self.num_envs, lengths[0])
        elif isinstance(lengths, (float, int)):
            lengths = np.full(self.num_envs, lengths)

        if isinstance(radii, (np.ndarray)):
            if len(radii) == 1:
                radii = np.full(self.num_envs, radii[0])
        elif isinstance(radii, (float, int)):
            radii = np.full(self.num_envs, radii)

        if len(centers.shape) == 1:
            centers = np.tile(centers, (self.num_envs, 1))
        centers = 0.5 * (lengths - radii)[:, None] * centers

        # save some useful values for use later
        self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
        peg_head_offsets = torch.zeros((self.num_envs, 3))
        peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
        self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

        box_hole_offsets = torch.zeros((self.num_envs, 3))
        box_hole_offsets[:, 1:] = common.to_tensor(centers)
        self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
        self.box_hole_radii = common.to_tensor(radii + self._clearance)

        # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
        pegs = []
        boxes = []

        for i in range(self.num_envs):
            scene_idxs = [i]
            length = lengths[i]
            radius = radii[i]
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[length, radius, radius])
            # peg head
            mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#EC7357"),
                roughness=0.5,
                specular=0.5,
            )
            builder.add_box_visual(
                sapien.Pose([length / 2, 0, 0]),
                half_size=[length / 2, radius, radius],
                material=mat,
            )
            # peg tail
            mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#EDF6F9"),
                roughness=0.5,
                specular=0.5,
            )
            builder.add_box_visual(
                sapien.Pose([-length / 2, 0, 0]),
                half_size=[length / 2, radius, radius],
                material=mat,
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            builder.set_scene_idxs(scene_idxs)
            peg = builder.build(f"peg_{i}")
            self.remove_from_state_dict_registry(peg)
            # box with hole

            inner_radius, outer_radius, depth = (
                radius + self._clearance,
                length,
                length,
            )
            builder = _build_box_with_hole(
                self.scene, inner_radius, outer_radius, depth, center=centers[i]
            )
            builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
            builder.set_scene_idxs(scene_idxs)
            box = builder.build_kinematic(f"box_with_hole_{i}")
            self.remove_from_state_dict_registry(box)
            pegs.append(peg)
            boxes.append(box)
        self.manip_obj = Actor.merge(pegs, "peg")
        self.goal_obj = Actor.merge(boxes, "box_with_hole")

        # to support heterogeneous simulation state dictionaries we register merged versions
        # of the parallel actors
        self.add_to_state_dict_registry(self.manip_obj)
        self.add_to_state_dict_registry(self.goal_obj)

    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.manip_obj.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.manip_obj.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.goal_obj.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.goal_obj.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
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
                    "robot_init_pos": [-0.615, 0, 0],
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
                # initialize the box and peg
                b = len(env_idx)
                assert goal_obj_pose is None
                xy = randomization.uniform(
                    low=torch.tensor([-0.1, -0.3]),
                    high=torch.tensor([0.1, 0]),
                    size=(b, 2),
                )
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self.peg_half_sizes[env_idx, 2] + table_height
                quat = randomization.random_quaternions(
                    b,
                    self.device,
                    lock_x=True,
                    lock_y=True,
                    bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
                )
                self.manip_obj.set_pose(Pose.create_from_pq(pos, quat))

                xy = randomization.uniform(
                    low=torch.tensor([-0.05, 0.2]),
                    high=torch.tensor([0.05, 0.4]),
                    size=(b, 2),
                )
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self.peg_half_sizes[env_idx, 0] + table_height
                quat = randomization.random_quaternions(
                    b,
                    self.device,
                    lock_x=True,
                    lock_y=True,
                    bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                )
                self.goal_obj.set_pose(Pose.create_from_pq(pos, quat))
            else:
                assert goal_obj_pose is not None
                for i in range(len(manip_obj_pose)):
                    manip_obj_pose[i][2] = (
                        self.peg_half_sizes[env_idx[0], 2].item() + table_height.item()
                    )
                    goal_obj_pose[i][2] = (
                        self.peg_half_sizes[env_idx[0], 0].item() + table_height.item()
                    )
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

        print("Initialized episode.")

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.025 <= peg_head_pos_at_hole[:, 0]  # -0.015
        y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (  # type: ignore
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (  # type: ignore
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.manip_obj.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Encourage gripper to be rotated to be lined up with the peg

        # Stage 2: Encourage gripper to move close to peg tail and grasp it
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.manip_obj.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose * (offset)
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose.p, axis=1
        )

        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
        is_grasped = self.agent.is_grasping(self.manip_obj, max_angle=20)  # type: ignore
        reward = reaching_reward + is_grasped

        # Stage 3: Orient the grasped peg properly towards the hole

        # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.manip_obj.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        pre_insertion_reward = 3 * (
            1
            - torch.tanh(
                0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
            )
        )
        reward += pre_insertion_reward * is_grasped
        # stage 3 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )

        # Stage 4: Insert the peg into the hole once it is grasped and lined up
        peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
        insertion_reward = 5 * (
            1
            - torch.tanh(
                5.0 * torch.linalg.norm(peg_head_wrt_goal_inside_hole.p, axis=1)
            )
        )
        reward += insertion_reward * (is_grasped & pre_inserted)

        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10

    def get_options_range(self):
        options = {}
        # add some extra options for the peg and box
        options["lengths"] = {"base": 0.1, "base_range": 0.01, "factor_range": 0}
        options["radii"] = {"base": 0.02, "base_range": 0.002, "factor_range": 0}
        options["centers"] = {
            "base": [0, 0],
            "base_range": [-0.3, 0.3],
            "factor_range": [0, 0],
        }
        options["manip_obj_pose"] = {
            "base": [0, -0.15, 0, 0.8, 0, 0, 0],
            "base_range": [0.04, 0.04, 0, 0.04, 0, 0, 0],
            "factor_range": [0.05, 0.05, 0, 0.05, 0, 0, 0],
        }
        options["goal_obj_pose"] = {
            "base": [0, 0.45, 0, 0.525, 0, 0, 0],
            "base_range": [0.04, 0.04, 0, 0.04, 0, 0, 0],
            "factor_range": [0.05, 0.05, 0, 0.05, 0, 0, 0],
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
            "base": [0.25, -0.1, 0.3, 0.05, 0.0, 0.1],
            "base_range": [0, 0, 0, 0, 0, 0],
            "factor_range": [0.025, 0.025, 0.025, 0, 0, 0],
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

        options_value["lengths"] = np.repeat(options_value["lengths"], self.num_envs)
        options_value["radii"] = np.repeat(options_value["radii"], self.num_envs)
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
                f"data/eval_poses/pose_pairs_{num_base_pairs}_{num_factor_pairs}_peg.png"
            )

        np.save(
            f"data/eval_poses/eval_base_manip_poses_file_peg_{num_base_pairs}.npy",
            base_manip_poses,
        )
        np.save(
            f"data/eval_poses/eval_base_goal_poses_file_peg_{num_base_pairs}.npy",
            base_goal_poses,
        )
        np.save(
            f"data/eval_poses/eval_factor_manip_poses_file_peg_{num_factor_pairs}.npy",
            factor_manip_poses,
        )
        np.save(
            f"data/eval_poses/eval_factor_goal_poses_file_peg_{num_factor_pairs}.npy",
            factor_goal_poses,
        )
        np.save(
            f"data/eval_poses/eval_base_delta_qpos_file_peg_{num_base_pairs}.npy",
            base_delta_qpos,
        )
        np.save(
            f"data/eval_poses/eval_factor_delta_qpos_file_peg_{num_factor_pairs}.npy",
            factor_delta_qpos,
        )
        if visualize:
            visualize_pose_pairs(
                (base_manip_poses, base_goal_poses),
                (factor_manip_poses, factor_goal_poses),
            )
