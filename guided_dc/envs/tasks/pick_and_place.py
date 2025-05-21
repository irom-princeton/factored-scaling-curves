"""
Overridable Arguments in `options`:

1. robot_init_qpos (array): Initial joint positions of the robot.
2. robot_init_euler (array): Initial Euler angles (rotation) of the robot.
3. robot_init_pos (array): Initial position of the robot.
4. table_model_file (str): File path to the table model.
5. floor_texture_file (str): File path to the floor texture.
6. table_height (float): Height of the table.
7. background_pos (array): Position of the background object.
8. background_rot (array): Rotation of the background object (Euler angles).
9. background_model_file (str): File path to the background model.
10. table_texture (str or list): Either a texture file path or an RGBA color list for the table material.
11. camera_pose (array)
12. manip_obj_pose (array)
13. goal_obj_pose (array)
14. lighting (float)

These arguments can be passed via the `options` dictionary to override default configurations.
"""

from typing import Any, ClassVar, Dict, List

import numpy as np
import sapien
import sapien.physx as physx
import torch
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import Array

from guided_dc.envs.agents.panda_wristcam import PandaWristCamIROM
from guided_dc.envs.factor_env import TwoObjectEnv
from guided_dc.envs.scenes.tabletop_scene_builder import TabletopSceneBuilder

TOP_DOWN_GRASP_BIAS: float = 1e-4
NUM_SAMPLES = 20


@register_env("PickAndPlace-v1", max_episode_steps=500)
class PickAndPlace(TwoObjectEnv):
    SUPPORTED_ROBOTS: ClassVar[List[str]] = ["panda_wristcam_irom"]  # type: ignore

    agent: PandaWristCamIROM

    def __init__(self, cfg):
        super().__init__(cfg)

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.

        self.table_scene = TabletopSceneBuilder(
            env=self,
            cfg=self.base_cfg.scene_builder,
            robot_init_qpos_noise=self.base_cfg.scene_builder.robot_init_qpos_noise,
        )
        self.table_scene.build(options)
        self.cam_mount = [
            self.scene.create_actor_builder().build_kinematic(f"camera_mount_{i}")
            for i in range(len(self.base_cfg.camera.sensor.eye))
        ]
        # self.wrist_mount = self.scene.create_actor_builder().build_kinematic(
        #     "wrist_mount"
        # )
        self.build_distractor(options)

        sapien.set_log_level("off")
        self._build_pick_and_place_obj()
        # sapien.set_log_level("warn")

    def _after_reconfigure(self, options: dict):
        with torch.device(self.device):
            for merged_obj, obj_list in [
                (self.manip_obj, self._manip_objs),
                (self.goal_obj, self._goal_objs),
            ]:
                heights = []
                meshes = []
                for obj in obj_list:
                    collision_mesh = obj.get_first_collision_mesh()
                    meshes.append(collision_mesh)
                    heights.append(-collision_mesh.bounding_box.bounds[0, 2])
                merged_obj.height = common.to_tensor(heights)
                # xyz = torch.zeros((len(obj_list), 3))
                # xyz[:, 2] = merged_obj.height
                # merged_obj.set_pose(Pose.create_from_pq(p=xyz, q=[1,0,0,0]))
            if hasattr(self, "distractors"):
                for _, (distractor, _distractors) in self.distractors.items():
                    heights = []
                    for d in _distractors:
                        collision_mesh = d.get_first_collision_mesh()
                        heights.append(-collision_mesh.bounding_box.bounds[0, 2])
                    distractor.height = common.to_tensor(heights)

    def _build_pick_and_place_obj(self):
        self.manip_obj, self._manip_objs = self._load_assets(
            asset_type=self.base_cfg.manip_obj.type,
            physics=self.base_cfg.manip_obj.physics,
            filepath=self.base_cfg.manip_obj.get("filepath", None),
            obj_name=self.base_cfg.manip_obj.get("obj_name", None),
            scale=self.base_cfg.manip_obj.get("scale", [1, 1, 1]),
            visuals=self.base_cfg.manip_obj.get("visuals", None),
            mass=self.base_cfg.goal_obj.get("mass", None),
            # mass=100,
        )  # type: ignore
        self.goal_obj, self._goal_objs = self._load_assets(
            asset_type=self.base_cfg.goal_obj.type,
            physics=self.base_cfg.goal_obj.physics,
            filepath=self.base_cfg.goal_obj.get("filepath", None),
            obj_name=self.base_cfg.goal_obj.get("obj_name", None),
            scale=self.base_cfg.goal_obj.get("scale", [1, 1, 1]),
            visuals=self.base_cfg.goal_obj.get("visuals", None),
            mass=self.base_cfg.goal_obj.get("mass", None),
            # mass=100,
        )  # type: ignore

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            qpos = np.array(
                [0, -0.62831853, 0, -2.51327412, 0, 1.88495559, 0, 0.04, 0.04]
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
                    "robot_init_qpos": qpos,
                    "table_height": options.get("table_height", 0),
                },
            )
            # self.randomize_robot_pose()
            # self.randomize_lighting()
            # self.randomize_camera()

            manip_obj_pose = options.get("manip_obj_pose", None)
            goal_obj_pose = options.get("goal_obj_pose", None)
            camera_pose = options.get("camera_pose", None)
            distractor_cfgs = options.get("distractor", None)
            table_height = torch.tensor(options.get("table_height", 0))
            if manip_obj_pose is None:
                assert goal_obj_pose is None
                self.set_obj_pose(self.base_cfg.goal_obj.pose, self.goal_obj)
                self.set_obj_pose(self.base_cfg.manip_obj.pose, self.manip_obj)
            else:
                assert goal_obj_pose is not None
                for i in range(len(manip_obj_pose)):
                    manip_obj_pose[i][2] += table_height.item()
                    goal_obj_pose[i][2] += table_height.item()
                self.set_obj_pose(manip_obj_pose, self.manip_obj)
                self.set_obj_pose(goal_obj_pose, self.goal_obj)

            if camera_pose is not None:
                self.camera_pose = camera_pose
            if distractor_cfgs:
                for distractor_cfg in distractor_cfgs:
                    name = f"distractor-{distractor_cfg['obj_name']}"
                    distractor_pose = self.get_distractor_pose(name)
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

    def evaluate(self):
        pos_pick = self.manip_obj.pose.p
        pos_place = self.goal_obj.pose.p
        offset = pos_pick - pos_place
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.07  # 2025.4.6: 0.07
        # print(offset)
        # print("xy:", xy_flag)
        # print(self.goal_obj.height + self.manip_obj.height)
        # print(self.goal_obj.height)
        # print(self.manip_obj.height)
        z_flag = (
            torch.abs(offset[..., 2])
            <= self.goal_obj.height + self.manip_obj.height  # 0.01
        )
        # print("z", z_flag)
        is_obj_on_place = torch.logical_and(xy_flag, z_flag)
        is_obj_static = self.manip_obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.manip_obj)
        success = is_obj_on_place & is_obj_static & (~is_obj_grasped)
        # success = is_obj_on_place & (~is_obj_grasped)
        # print("is_grasping:", is_obj_grasped)
        # print("is_on_place:", is_obj_on_place)
        # print("success:", success)
        return {
            # "is_obj_grasped": is_obj_grasped,
            # "is_obj_on_place": is_obj_on_place,
            # "is_obj_static": is_obj_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose.clone(),  # type: ignore
            gripper=self.agent.robot.get_qpos()[:, -1].clone(),
            bin_pos=self.goal_obj.pose.p,
        )

        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.manip_obj.pose.raw_pose,
                tcp_to_obj_pos=self.manip_obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # # reaching reward
        # tcp_pose = self.agent.tcp.pose.p
        # obj_pos = self.manip_obj.pose.p
        # obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        # reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # # grasp and place reward TODO: need object specific metrics
        # obj_pos = self.manip_obj.pose.p
        # place_obj_top_pos = self.goal_obj.pose.p.clone()
        # place_obj_top_pos[:, 2] = place_obj_top_pos[:, 2] + self.manip_obj.height/2 + self.manip_obj.height/2
        # obj_to_place_obj_top_dist = torch.linalg.norm(place_obj_top_pos - obj_pos, axis=1)
        # place_reward = 1 - torch.tanh(5.0 * obj_to_place_obj_top_dist)
        # reward[info["is_obj_grasped"]] = (4 + place_reward)[info["is_obj_grasped"]]

        # # ungrasp and static reward
        # gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        # is_obj_grasped = info["is_obj_grasped"]
        # ungrasp_reward = (
        #     torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        # )
        # ungrasp_reward[
        #     ~is_obj_grasped
        # ] = 16.0  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        # v = torch.linalg.norm(self.manip_obj.linear_velocity, axis=1)
        # av = torch.linalg.norm(self.manip_obj.angular_velocity, axis=1)
        # static_reward = 1 - torch.tanh(v * 10 + av)
        # robot_static_reward = self.agent.is_static(
        #     0.2
        # )  # keep the robot static at the end state, since the sphere may spin when being placed on top
        # reward[info["is_obj_on_place"]] = (
        #     6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        # )[info["is_obj_on_place"]]

        # # success reward
        # reward[info["success"]] = 13
        # return reward

        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        # max_reward = 13.0
        # return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
        return 0

    def get_options_range(self):
        options = {}
        # add some extra options for the peg and box
        options["manip_obj_pose"] = {
            "base": [-0.2, 0, 0, -1.57265, -0.0464526, 3.0965],
            "base_range": [0.1, 0.05, 0, 0.0, 0, 0],
            "factor_range": [0.15, 0.075, 0, 0.0, 0, 0],
        }
        options["goal_obj_pose"] = {
            "base": [-0.4, 0.0, 0.01, 1.569, 2.9, -0.001066],
            "base_range": [0.1, 0.05, 0, 0, 0.25, 0],
            "factor_range": [0.15, 0.075, 0, 0, 0.5, 0],
        }

        options["delta_qpos"] = {
            "base": np.zeros(9),
            "base_range": np.ones(9) * 0.01,
            "factor_range": np.ones(9) * 0.02,
        }

        options["table_height"] = {
            "base": 0,
            "base_range": 0.0,
            "factor_range": 0.05,
        }

        options["camera_pose"] = {
            "base": [
                -0.56870818,
                0.51039636,
                0.5691,
                -0.77100993,
                -0.25428228,
                -0.3064509,
                0.49696282,
            ],
            "base_range": [0.0] * 7,
            "factor_range": [0.025] * 7,
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

        options_value["camera_pose"] = [options_value["camera_pose"]]
        options_value["manip_obj_pose"] = manip_obj_pose
        options_value["goal_obj_pose"] = goal_obj_pose
        options_value["delta_qpos"] = delta_qpos

        options_value["table_height"] = options_value["table_height"].item()
        options_value["load_table_struc"] = True

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
                f"data/eval_poses/pose_pairs_{num_base_pairs}_{num_factor_pairs}.png"
            )

        np.save(
            f"data/eval_poses/eval_base_manip_poses_file_{num_base_pairs}.npy",
            base_manip_poses,
        )
        np.save(
            f"data/eval_poses/eval_base_goal_poses_file_{num_base_pairs}.npy",
            base_goal_poses,
        )
        np.save(
            f"data/eval_poses/eval_factor_manip_poses_file_{num_factor_pairs}.npy",
            factor_manip_poses,
        )
        np.save(
            f"data/eval_poses/eval_factor_goal_poses_file_{num_factor_pairs}.npy",
            factor_goal_poses,
        )
        np.save(
            f"data/eval_poses/eval_base_delta_qpos_file_{num_base_pairs}.npy",
            base_delta_qpos,
        )
        np.save(
            f"data/eval_poses/eval_factor_delta_qpos_file_{num_factor_pairs}.npy",
            factor_delta_qpos,
        )
        if visualize:
            visualize_pose_pairs(
                (base_manip_poses, base_goal_poses),
                (factor_manip_poses, factor_goal_poses),
            )


# class PickAndPlaceMP(PickAndPlace):
#     def __init__(self, cfg, do_motion_planning: bool = True):
#         self.do_motion_planning = do_motion_planning
#         super().__init__(cfg)

#     def _build_pick_and_place_obj(self):
#         super()._build_pick_and_place_obj()

#         if self.do_motion_planning:
#             self.grasp_point = actors.build_box(
#                 self.scene,
#                 half_sizes=[0.05, 0.01, 0.01],
#                 color=[0, 1, 0, 1],
#                 name="handle_link_goal",
#                 body_type="kinematic",
#                 add_collision=False,
#             )

#     def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
#         # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
#         super()._initialize_episode(env_idx, options)
#         with torch.device(self.device):
#             if self.do_motion_planning:
#                 self.manip_obj_pcds = []
#                 self.goal_obj_pcds = []
#                 if len(self.manip_obj_meshes_raw) > 1:
#                     for i in range(len(self.manip_obj_meshes_raw)):
#                         manip_mesh = self.manip_obj_meshes_raw[i].copy()
#                         goal_mesh = self.goal_obj_meshes_raw[i].copy()
#                         manip_pose = self.manip_obj.pose[i]
#                         goal_pose = self.goal_obj.pose[i]

#                         manip_mesh.apply_transform(
#                             manip_pose.sp.to_transformation_matrix()
#                         )
#                         goal_mesh.apply_transform(
#                             goal_pose.sp.to_transformation_matrix()
#                         )

#                         samples = trimesh.sample.sample_surface_even(
#                             manip_mesh, NUM_SAMPLES
#                         )
#                         pcd = PointCloud(
#                             xyz_pts=samples[0].__array__().copy(),
#                             normals=manip_mesh.face_normals[samples[1]].copy(),
#                         )
#                         self.manip_obj_pcds.append(pcd)

#                         # visualize_trimesh(manip_mesh, extra_pts=(samples[0].__array__().copy(), manip_mesh.face_normals[samples[1]].copy()))

#                         samples = trimesh.sample.sample_surface_even(
#                             goal_mesh, NUM_SAMPLES
#                         )
#                         pcd = PointCloud(
#                             xyz_pts=samples[0].__array__().copy(),
#                             normals=goal_mesh.face_normals[samples[1]].copy(),
#                         )
#                         self.goal_obj_pcds.append(pcd)
#                 else:
#                     manip_mesh = self.manip_obj_meshes_raw.copy()
#                     goal_mesh = self.goal_obj_meshes_raw.copy()
#                     manip_pose = self.manip_obj.pose
#                     goal_pose = self.goal_obj.pose

#                     manip_mesh.apply_transform(manip_pose.sp.to_transformation_matrix())
#                     goal_mesh.apply_transform(goal_pose.sp.to_transformation_matrix())

#                     samples = trimesh.sample.sample_surface_even(
#                         manip_mesh, NUM_SAMPLES
#                     )
#                     pcd = PointCloud(
#                         xyz_pts=samples[0].__array__().copy(),
#                         normals=manip_mesh.face_normals[samples[1]].copy(),
#                     )
#                     self.manip_obj_pcds.append(pcd)

#                     samples = trimesh.sample.sample_surface_even(goal_mesh, NUM_SAMPLES)
#                     pcd = PointCloud(
#                         xyz_pts=samples[0].__array__().copy(),
#                         normals=goal_mesh.face_normals[samples[1]].copy(),
#                     )
#                     self.goal_obj_pcds.append(pcd)

#         print("Initialized episode.")

#     def get_grasp_pose(self):
#         pose = self.get_feasible_candidate(self.manip_obj_pcds)
#         self.grasp_point.set_pose(pose)
#         if physx.is_gpu_enabled():
#             self.scene._gpu_apply_all()
#             self.scene.px.gpu_update_articulation_kinematics()
#             self.scene.px.step()
#             self.scene._gpu_fetch_all()
#         return pose

#     def get_target_qpos(self):
#         goal_cfg_at_global: sapien.Pose = copy.deepcopy(self.get_grasp_pose())
#         q0 = self.agent.robot.get_qpos().clone()
#         goal_cfg_at_base = self.agent.robot.pose.inv() * goal_cfg_at_global
#         goal_qpos_for_arm = self.agent.controller.controllers[
#             "arm"
#         ].kinematics.compute_ik(
#             target_pose=goal_cfg_at_base, q0=q0, use_delta_ik_solver=False
#         )
#         goal_qpos_for_gripper = torch.ones_like(goal_qpos_for_arm)[
#             :, :2
#         ]  # 2 dim, one gripper hand mimic the other so just set them the same
#         qpos_to_set = torch.cat((goal_qpos_for_arm, goal_qpos_for_gripper), dim=-1)

#         return qpos_to_set, None

#     def get_feasible_candidate(
#         self, link_point_clouds: List[PointCloud], pushin_more: bool = True
#     ):
#         grasp_p = np.zeros((len(link_point_clouds), 3))
#         grasp_q = np.zeros((len(link_point_clouds), 4))

#         for pcd_idx, link_point_cloud in enumerate(link_point_clouds):
#             normals = link_point_cloud.normals
#             pointing_down = normals[:, 2] < 0.0

#             p = np.ones(shape=len(link_point_cloud), dtype=np.float64)
#             if not pointing_down.all():
#                 # sample the ones point up more often
#                 p = np.exp(normals[:, 2] * TOP_DOWN_GRASP_BIAS)
#             p /= p.sum()

#             candidate_indices = self._episode_rng.choice(
#                 len(link_point_cloud),
#                 size=min(
#                     self.base_cfg.grasp.num_action_candidates, len(link_point_cloud)
#                 ),
#                 p=p,
#                 replace=True,
#             )
#             for i, idx in enumerate(candidate_indices):
#                 assert i <= 1
#                 position = link_point_cloud.xyz_pts[idx].copy()
#                 grasp_to_ee = (
#                     self.agent.tcp.pose.p.cpu().numpy()[pcd_idx] - position
#                 )  # in global frame, no broadcasting
#                 grasp_to_ee /= np.linalg.norm(grasp_to_ee)  # if broadcasting, axis=1
#                 if pointing_down.all():
#                     # disallow bottom up grasps, so using point to ee
#                     # as normal, with some random exploration
#                     grasp_to_ee += (
#                         self._episode_rng.randn(3) * 0.1
#                     )  # if broadcasting, randn is 2D
#                     grasp_to_ee /= np.linalg.norm(
#                         grasp_to_ee
#                     )  # if broadcasting, axis=1
#                     normal = grasp_to_ee
#                 else:
#                     normal = normals[idx]
#                     # # Expand normal to (len(grasp_to_ee), 3)
#                     # normal = np.tile(normal, (len(grasp_to_ee), 1))

#                 ## Orient normal towards end effector
#                 ## For broadcasting, get dot product of each row of grasp_to_ee and normal
#                 # dot_product = np.einsum('ij,ij->i', grasp_to_ee, normal)
#                 # neg_indices = dot_product < 0
#                 # normal[neg_indices] *= -1.0
#                 if np.dot(grasp_to_ee, normal) < 0:
#                     normal *= -1.0
#                 # compute base orientation, randomize along Z-axis
#                 try:
#                     # base_orientation = quaternion_multiply(  # If broadcasting, using batch_quaternion_multiply
#                     #     normal_to_forward_quat(normal),
#                     #     euler2quat(-np.pi, 0, np.pi),
#                     # )
#                     base_orientation = normal_to_forward_quat(normal)
#                 except np.linalg.LinAlgError as e:
#                     logging.warning(e)
#                     base_orientation = euler2quat(0, 0, 0)
#                 # z_angle = self._episode_rng.uniform(np.pi, -np.pi)
#                 # z_orn = euler2quat(0, 0, z_angle)
#                 # base_orientation = quaternion_multiply(base_orientation, z_orn)
#                 # pregrasp_distance = self._episode_rng.uniform(0.1, 0.2)
#                 if pushin_more:
#                     # prioritize grasps that push in more, if possible, but slowly
#                     # back off to prevent all grasps colliding.
#                     pushin_distance = (len(candidate_indices) - i) / len(
#                         candidate_indices
#                     ) * (
#                         self.base_cfg.grasp.max_pushin_dist
#                         - self.base_cfg.grasp.min_pushin_dist
#                     ) + self.base_cfg.grasp.min_pushin_dist
#                 else:
#                     # prioritize grasps that push in less. Useful for "pushing"
#                     pushin_distance = (
#                         i
#                         / len(candidate_indices)
#                         * (
#                             self.base_cfg.grasp.max_pushin_dist
#                             - self.base_cfg.grasp.min_pushin_dist
#                         )
#                         + self.base_cfg.grasp.min_pushin_dist
#                     )

#                 grasp_p[pcd_idx] = position - normal * pushin_distance
#                 grasp_q[pcd_idx] = base_orientation

#         # compute grasp pose relative to object
#         grasp_pose = Pose.create_from_pq(
#             p=grasp_p,
#             q=grasp_q,
#         )

#         return grasp_pose

#     #     candidate = GraspLinkPoseAction(
#     #         link_path=self.link_path,
#     #         pose=grasp_pose,
#     #         with_backup=self.with_backup,
#     #         backup_distance=pushin_distance + pregrasp_distance,
#     #     )
#     #     try:
#     #         candidate.check_feasibility(env=env)
#     #         logging.info(
#     #             f"[{i}|{env.episode_id}] "
#     #             + f"pushin_distance: {pushin_distance} ({self.pushin_more})"
#     #         )
#     #         break
#     #     except Action.InfeasibleAction as e:
#     #         errors.append(e)
#     #         candidate = None
#     #         continue
#     # del normals, link_point_cloud, grasp_to_ee, candidate_indices, p

#     # # see `tests/test_memory_leak.py``, this needs to be here,
#     # # otherwise, a few MBs will leak each time
#     # if candidate is not None:
#     #     return ActionListPolicy(actions=[candidate])
#     # raise Action.InfeasibleAction(
#     #     action_class_name=type(self).__name__,
#     #     message=f"all candidates failed {self}:"
#     #     + ",".join(list({str(e) for e in errors})[:3]),
#     # )
