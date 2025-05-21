import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import objaverse
import sapien
import sapien.physx as physx
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Actor, Pose
from sapien.render import RenderMaterial
from transforms3d.euler import euler2quat

from guided_dc.envs.objects.generate_obj import get_obj_asset
from guided_dc.utils.pose_utils import add_euler
from guided_dc.utils.randomization import randomize_array

log = logging.getLogger(__name__)

FORCE_THRESHOLD = 1e-5


class FactorEnv(BaseEnv):
    def __init__(self, cfg):
        self.base_cfg = cfg
        self.cam_mount = []

        # Check if we are rendering in a single scene
        parallel_in_single_scene = cfg.render_mode == "human"
        if cfg.render_mode == "human" and cfg.obs_mode in [
            "sensor_data",
            "rgb",
            "rgbd",
            "depth",
            "point_cloud",
        ]:
            log.info(
                "Disabling parallel single scene/GUI render as observation mode is a visual one. "
                "Change observation mode to 'state' or 'state_dict' to enable parallel rendering."
            )
            parallel_in_single_scene = False

        if cfg.render_mode == "human" and cfg.num_envs == 1:
            parallel_in_single_scene = False

        super().__init__(
            num_envs=cfg.num_envs,
            obs_mode=cfg.obs_mode,
            reward_mode=cfg.reward_mode,
            control_mode=cfg.control_mode,
            render_mode=cfg.render_mode,
            shader_dir=cfg.get("shader_dir", None),
            enable_shadow=cfg.get("enable_shadow", False),
            sensor_configs=dict(shader_pack=cfg.shader),
            human_render_camera_configs=dict(shader_pack=cfg.shader),
            viewer_camera_configs=dict(shader_pack=cfg.shader),
            robot_uids=cfg.robot_uids,
            sim_config=cfg.get("sim_config", dict()),
            reconfiguration_freq=cfg.get("reconfiguration_freq", None),
            sim_backend=cfg.get("sim_backend", "auto"),
            render_backend=cfg.get("render_backend", "gpu"),
            parallel_in_single_scene=parallel_in_single_scene,
            enhanced_determinism=cfg.get("enhanced_determinism", False),
        )

    def _camera_template(self, camera_cfg, camera_type, use_mount=True):
        camera_configs = []
        for i, cam_mount in enumerate(self.cam_mount):
            if camera_cfg.use_pose:
                pose_list = camera_cfg.pose[i]
                pose = Pose.create_from_pq(p=pose_list[:3], q=pose_list[3:])
                log.info(f"Using user-defined camera pose for {camera_type}_{i}.")
            else:
                lookat_pose = sapien_utils.look_at(
                    eye=camera_cfg.eye[i], target=camera_cfg.target[i]
                )
                pose = Pose.create(lookat_pose)
            fov = camera_cfg.get("fov", None)
            intrinsic = camera_cfg.get("intrinsic", None)
            if fov is not None:
                assert intrinsic is None, (
                    "Intrinsic should not be provided if fov is provided."
                )
                log.info(f"Using fov for {camera_type} intrinsic.")
            else:
                assert intrinsic is not None, (
                    f"Intrinsic must be provided for {camera_type}."
                )
                log.info(f"Using intrinsic for {camera_type}.")
            if not use_mount:
                uid = camera_type
            else:
                uid = f"{camera_type}_{i}"
                cam_mount.set_pose(pose)
                pose = Pose.create_from_pq(p=[0, 0, 0], q=[1, 0, 0, 0])  # type: ignore
            camera_configs.append(
                CameraConfig(
                    uid=uid,
                    pose=pose,
                    mount=cam_mount if use_mount else None,  # type: ignore
                    width=camera_cfg.width,
                    height=camera_cfg.height,
                    near=camera_cfg.near,
                    far=camera_cfg.far,
                    fov=fov,
                    intrinsic=intrinsic,
                )
            )
        return camera_configs

    # Specify default simulation/gpu memory configurations to override any default values
    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         gpu_memory_config=GPUMemoryConfig(
    #             found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
    #         )
    #     )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster

        # 1. Side camera
        assert isinstance(self.cam_mount, list)
        sensor_camera_configs = self._camera_template(
            self.base_cfg.camera.sensor, "sensor"
        )

        # 2. Wrist camera
        wrist_camera_cfg = self.base_cfg.camera.get("wrist_camera", None)
        if wrist_camera_cfg:
            fov = wrist_camera_cfg.get("fov", None)
            intrinsic = wrist_camera_cfg.get("intrinsic", None)
            if fov is not None:
                assert intrinsic is None, (
                    "Intrinsic should not be provided if fov is provided."
                )
                log.info("Using fov for wrist camera intrinsic.")
            else:
                assert intrinsic is not None, (
                    "Intrinsic must be provided for wrist camera."
                )
                log.info("Using intrinsic for wrist camera.")
            assert wrist_camera_cfg.use_pose, "Wrist camera pose must be provided."
            pose_list = wrist_camera_cfg.pose[0]
            pose = Pose.create_from_pq(p=pose_list[:3], q=pose_list[3:])
            # pose = Pose.create_from_pq(
            #     p=[0.07422515, -0.07422515, 0.0264795],
            #     q=[0.4267, -0.3394, -0.8195, -0.1767],
            # )
            wrist_camera_configs = [
                CameraConfig(
                    uid="hand_camera",
                    pose=pose,
                    # pose=Pose.create_from_pq(p=[0, 0, 0], q=[1, 0, 0, 0]),
                    width=wrist_camera_cfg.width,
                    height=wrist_camera_cfg.height,
                    near=wrist_camera_cfg.near,
                    far=wrist_camera_cfg.far,
                    fov=fov,
                    intrinsic=intrinsic,
                    # mount=self.agent.robot.links_map["camera_reference_link"],
                    # mount=self.wrist_mount,
                    # mount=self.agent.robot.links_map["panda_hand"],
                    mount=self.agent.robot.links_map["panda_link8"],
                )
            ]
        else:
            wrist_camera_configs = []

        # 3. Additional cameras
        # additional_confgs = [
        #     CameraConfig(
        #         uid="camera_50",
        #         pose=sapien_utils.look_at(eye=[1, 0.5, 0.5], target=[0, 0, 0]),
        #         width=640,
        #         height=480,
        #         fov=np.pi / 4,
        #     )
        # ]
        additional_confgs = []
        return sensor_camera_configs + wrist_camera_configs + additional_confgs

    @property
    def _default_viewer_camera_configs(
        self,
    ) -> CameraConfig:
        """Default configuration for the viewer camera, controlling shader, fov, etc. By default if there is a human render camera called "render_camera" then the viewer will use that camera's pose."""
        return self._camera_template(
            self.base_cfg.camera.viewer, "viewer", use_mount=False
        )  # type: ignore

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        return self._camera_template(
            self.base_cfg.camera.human_render, "render_camera", use_mount=False
        )

    @property
    def camera_pose(self):
        return [cam_mount.pose.raw_pose.cpu().numpy() for cam_mount in self.cam_mount]

    @camera_pose.setter
    def camera_pose(self, camera_pose):
        for i, cam_mount in enumerate(self.cam_mount):
            if hasattr(camera_pose[i], "keys"):
                pose = sapien_utils.look_at(
                    eye=camera_pose[i]["eye"], target=camera_pose[i]["target"]
                )
                pose = Pose.create(pose)
                cam_mount.set_pose(pose)
                continue
            if isinstance(camera_pose[i], float):
                camera_pose[i] = [camera_pose[i]]
            pose = Pose.create_from_pq(p=camera_pose[i][:3], q=camera_pose[i][3:])
            cam_mount.set_pose(pose)

    @property
    def lighting(self):
        return {"ambient": self.scene.ambient_light.copy()}

    @property
    def robot_qpos(self):
        return self.agent.robot.get_qpos().cpu().numpy()

    @property
    def robot_base_pose(self):
        return self.agent.robot.pose.raw_pose.cpu().numpy()

    def set_ambient_light(self, ambient_light):
        self.scene.set_ambient_light(ambient_light)

    # Overload the _load_lighting() functions in sapien_env
    def _load_lighting(self, options: dict):
        """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""

        shadow = self.enable_shadow
        # shadow = False
        # self.scene.add_spot_light(
        #     position=[0, 0, 2],
        #     direction=[0, 0, -1],
        #     inner_fov=30,
        #     outer_fov=60,
        #     color=[1, 1, 1],
        # )
        # self.scene.add_point_light(position=[0, 0, 0], color=[1, 1, 1])
        ambient_override = options.get("ambient", None)
        if ambient_override:
            if isinstance(ambient_override, float):
                self.scene.set_ambient_light([ambient_override] * 3)
            else:
                assert len(ambient_override) == 3
                self.scene.set_ambient_light(ambient_override)
        else:
            self.scene.set_ambient_light(self.base_cfg.lighting.ambient)
        directional = options.get("directional", None)
        if directional:
            self.scene.add_directional_light(
                direction=[-1, -1, -0.3]
                if not len(directional) == 2
                else directional[0],
                color=directional if not len(directional) == 2 else directional[1],
                # position=[0, 0, 100],
                shadow=shadow,
                shadow_scale=10,
                shadow_map_size=2048,
            )
        else:
            self.scene.add_directional_light(
                direction=[0, 0.4, -1],
                color=[0.6, 0.6, 0.6],
                # position=[0, 0, 100],
                shadow=shadow,
                shadow_scale=10,
                shadow_map_size=2048,
            )
        # self.scene.add_directional_light(
        #     [1, 0, -1],
        #     [0.9, 0.9, 0.9],
        #     position=[0, 0, 100],
        #     shadow=shadow,
        #     shadow_scale=5,
        #     shadow_map_size=2048,
        # )

        # self.scene.add_area_light_for_ray_tracing(
        #     pose=sapien.Pose(p=[0, 0, 4], q=euler2quat(0, 3.14, 0)),
        #     color=[1, 1, 1],
        #     half_height=50,
        #     half_width=50,
        # )
        # self.scene.add_directional_light(
        #     [0, 0.6, -1],
        #     [0.7, 0.7, 0.7],
        #     shadow=shadow,
        #     shadow_scale=5,
        #     shadow_map_size=2048,
        # )

    def _load_glb_assets(
        self,
        path,
        physic_material=None,
        visual_material=None,
        scale=[1, 1, 1],
        mass=None,
        kinematic=False,
    ):
        _objs = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([i])
            builder.add_visual_from_file(
                filename=path, scale=scale, material=visual_material
            )
            builder.add_convex_collision_from_file(
                filename=path, scale=scale, material=physic_material
            )
            builder.initial_pose = sapien.Pose(q=euler2quat(0, 0, 0))  # type: ignore
            if mass is not None:
                builder._mass = mass
            if not kinematic:
                _objs.append(builder.build(name=f"{path}-{i}"))
            else:
                _objs.append(builder.build_kinematic(name=f"{path}-{i}"))
            self.remove_from_state_dict_registry(_objs[-1])
        obj = Actor.merge(_objs, name=f"{path}")
        self.add_to_state_dict_registry(obj)
        return obj, _objs

    def _load_assets(
        self,
        asset_type,
        physics=None,
        visuals=None,
        filepath=None,
        obj_name=None,
        scale=[1, 1, 1],
        mass=None,
        kinematic=False,
    ):
        if physics:
            physic_material = sapien.Scene().create_physical_material(**physics)
        else:
            physic_material = None
        if visuals is not None:
            visual_material = RenderMaterial(**visuals)
        else:
            visual_material = None
        if asset_type == "custom":
            return self._load_glb_assets(
                filepath, physic_material, visual_material, scale, mass, kinematic
            )
        elif asset_type == "ai2thor":
            return self._load_ai2thor_assets(
                obj_name, physic_material, visual_material, kinetic=kinematic
            )
        elif asset_type == "objaverse":
            return self._load_objaverse_assets(
                obj_name, physic_material, visual_material, scale, kinetic=kinematic
            )

    def _load_objaverse_assets(
        self,
        obj_name,
        material=None,
        visual_material=None,
        scale=[1, 1, 1],
        kinetic=False,
    ):
        lvis_annotations = objaverse.load_lvis_annotations()
        uids_to_load = lvis_annotations[obj_name]
        # processes = multiprocessing.cpu_count()
        random.seed(3)
        uids_to_load = random.sample(uids_to_load, 1)
        obj_dicts = objaverse.load_objects(uids=uids_to_load, download_processes=1)
        asset_path_list = list(obj_dicts.values())

        rand_idx = torch.randperm(len(asset_path_list))
        model_files = [asset_path_list[idx] for idx in rand_idx]
        model_files = np.concatenate(
            [model_files] * np.ceil(self.num_envs / len(asset_path_list)).astype(int)
        )[: self.num_envs]
        print(model_files)
        if (
            self.num_envs > 1
            and self.num_envs < len(asset_path_list)
            and self.reconfiguration_freq <= 0
        ):
            log.info(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be > 1."""
            )

        _objs: List[Actor] = []
        for i, model_file in enumerate(model_files):
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            model_name = Path(model_file).name
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([i])

            obj_pose = sapien.Pose(q=euler2quat(0, 0, 0))  # type: ignore

            if material is not None:
                builder.add_convex_collision_from_file(
                    filename=model_file, scale=scale, pose=obj_pose, material=material
                )
            else:
                log.info(
                    f"Using default physical properties for the object {model_name}."
                )
                builder.add_convex_collision_from_file(
                    filename=model_file, scale=scale, pose=obj_pose
                )

            builder.add_visual_from_file(
                filename=model_file,
                scale=scale,
                pose=obj_pose,
                material=visual_material,
            )

            builder.initial_pose = obj_pose  # type: ignore
            if not kinetic:
                _objs.append(builder.build(name=f"{model_name}-{i}"))
            else:
                _objs.append(builder.build_kinematic(name=f"{model_name}-{i}"))
            self.remove_from_state_dict_registry(_objs[-1])
        obj = Actor.merge(_objs, name=f"{obj_name}")
        self.add_to_state_dict_registry(obj)
        return obj, _objs

    def _load_ai2thor_assets(
        self,
        obj_name,
        material=None,
        visual_material=None,
        scale=[1, 1, 1],
        kinetic=False,
    ):
        asset_path_list: list = get_obj_asset(obj_name)
        rand_idx = torch.randperm(len(asset_path_list))
        model_files = [asset_path_list[idx] for idx in rand_idx]
        model_files = np.concatenate(
            [model_files] * np.ceil(self.num_envs / len(asset_path_list)).astype(int)
        )[: self.num_envs]
        if (
            self.num_envs > 1
            and self.num_envs < len(asset_path_list)
            and self.reconfiguration_freq <= 0
        ):
            log.info(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be > 1."""
            )

        _objs: List[Actor] = []
        for i, model_file in enumerate(model_files):
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            model_name = Path(model_file).name
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([i])

            obj_pose = sapien.Pose(q=euler2quat(0, 0, 0))  # type: ignore

            if material is not None:
                builder.add_convex_collision_from_file(
                    filename=model_file, scale=scale, pose=obj_pose, material=material
                )
            else:
                log.info(
                    f"Using default physical properties for the object {model_name}."
                )
                builder.add_convex_collision_from_file(
                    filename=model_file, scale=scale, pose=obj_pose
                )

            builder.add_visual_from_file(
                filename=model_file,
                scale=scale,
                pose=obj_pose,
                material=visual_material,
            )
            builder.initial_pose = obj_pose  # type: ignore
            if not kinetic:
                _objs.append(builder.build(name=f"{model_name}-{i}"))
            else:
                _objs.append(builder.build_kinematic(name=f"{model_name}-{i}"))
            self.remove_from_state_dict_registry(_objs[-1])
        obj = Actor.merge(_objs, name=f"{obj_name}")
        self.add_to_state_dict_registry(obj)
        return obj, _objs

    def build_distractor(self, options):
        override = options.get("distractor", [])
        if override:
            cfgs = override
        else:
            cfgs = self.base_cfg.get("distractor", [])
        self.distractors = {}
        for cfg in cfgs:
            distractor, _distractors = self._load_assets(
                asset_type=cfg.type,
                filepath=cfg.get("model_file", None),
                obj_name=cfg.obj_name,
                scale=cfg.scale,
                kinematic=True,
            )  # type: ignore
            aabb = (
                distractor._objs[0]
                .find_component_by_type(sapien.render.RenderBodyComponent)
                .compute_global_aabb_tight()
            )
            length = aabb[1, 0] - aabb[0, 0]
            w = aabb[1, 1] - aabb[0, 1]
            h = aabb[1, 2] - aabb[0, 2]
            log.info(
                f"Adding distractor '{cfg.obj_name}' with dimention: {length}, {w}, {h}"
            )
            self.distractors[f"distractor-{cfg.obj_name}"] = [distractor, _distractors]
            # self.table_scene.scene_objects.append(distractor)


class TwoObjectEnv(FactorEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.manip_obj = None
        # self.goal_obj = None
        # self.table_scene = None

    @property
    def manip_obj_pose(self):
        return self.manip_obj.pose.raw_pose.clone().cpu().numpy()

    @property
    def goal_obj_pose(self):
        return self.goal_obj.pose.raw_pose.clone().cpu().numpy()

    def set_obj_pose(self, pose: list, obj):
        if isinstance(pose[0], list) or isinstance(pose[0], np.ndarray):
            # assert self.num_envs > 1 and len(pose) == self.num_envs
            if len(pose[0]) == 6:
                p = np.stack([pose[i][:3] for i in range(self.num_envs)], axis=0)
                q = np.stack(
                    [
                        euler2quat(*pose[i][3:], axes="rxyz")
                        for i in range(self.num_envs)
                    ],
                    axis=0,
                )  # type: ignore
            elif len(pose[0]) == 7:
                p = np.stack([pose[i][:3] for i in range(self.num_envs)], axis=0)
                q = np.stack([pose[i][3:] for i in range(self.num_envs)], axis=0)
        else:
            assert len(pose) == 6, "rotation should be euler angles"
            p = pose[:3]
            q = euler2quat(pose[3], pose[4], pose[5], "rxyz")
        obj.set_pose(Pose.create_from_pq(p=p, q=q))  # type: ignore

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            # self.randomize_robot_pose()
            # self.randomize_lighting()
            # self.randomize_camera()
            manip_obj_pose = options.get("manip_obj_pose", None)
            goal_obj_pose = options.get("goal_obj_pose", None)
            camera_pose = options.get("camera_pose", None)
            distractor_cfgs = options.get("distractor", None)
            if manip_obj_pose is None:
                self.set_obj_pose(self.base_cfg.manip_obj.pose, self.manip_obj)
            else:
                self.set_obj_pose(manip_obj_pose, self.manip_obj)
            if goal_obj_pose is None:
                self.set_obj_pose(self.base_cfg.goal_obj.pose, self.goal_obj)
            else:
                self.set_obj_pose(goal_obj_pose, self.goal_obj)
            if camera_pose is not None:
                self.camera_pose = camera_pose
            if distractor_cfgs:
                for distractor_cfg in distractor_cfgs:
                    name = f"distractor-{distractor_cfg['obj_name']}"
                    self.distractors[name][0].set_pose(self.get_distractor_pose(name))

            if physx.is_gpu_enabled():
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()  # type: ignore
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            # # Initialize the robot
            # qpos = np.array(
            #     [
            #         0.0,
            #         np.pi / 8,
            #         0,
            #         -np.pi * 5 / 8,
            #         0,
            #         np.pi * 3 / 4,
            #         -np.pi / 4,
            #         0.04,
            #         0.04,
            #     ]
            # )
            # qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            # qpos[:, -2:] = 0.04
            # self.agent.robot.set_qpos(qpos)
            # self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

        print("Initialized episode.")

    def get_distractor_pose(
        self,
        name,
        min_distance=0.25,
        x_min=-0.5,
        x_max=0,
        y_min=-0.25,
        y_max=0.25,
        max_attempts=1000,
    ):
        num_envs = self.manip_obj_pose.shape[0]

        # Extract (x, y) positions of existing objects
        manip_xy = common.to_tensor(self.manip_obj_pose[:, :2])  # (num_envs, 2)
        goal_xy = common.to_tensor(self.goal_obj_pose[:, :2])  # (num_envs, 2)

        # Sample distractor (x, y)
        distractor_xy = torch.empty_like(manip_xy)

        for i in range(num_envs):
            for _ in range(max_attempts):
                sample_x = (
                    torch.rand(1) * (x_max - x_min) + x_min
                )  # Sample x from [x_min, x_max]
                sample_y = (
                    torch.rand(1) * (y_max - y_min) + y_min
                )  # Sample y from [y_min, y_max]
                sample_xy = torch.cat([sample_x, sample_y])  # Combine into (2,) tensor

                if (
                    torch.norm(sample_xy - manip_xy[i]) > min_distance
                    and torch.norm(sample_xy - goal_xy[i]) > min_distance
                ):
                    distractor_xy[i] = sample_xy
                    break
            else:
                raise RuntimeError(
                    f"Failed to sample a valid distractor position for environment {i}"
                )
        # Set distractor pose
        distractor_pose = torch.zeros((num_envs, 7))
        distractor_pose[:, :2] = distractor_xy  # Set x, y
        distractor_pose[:, 2] = self.distractors[name][0].height  # Set z
        distractor_pose[:, 3:] = torch.tensor(
            [0, 0, 0, 1]
        )  # Default quaternion (no rotation)

        return distractor_pose


class MotionPlanningEnv(FactorEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rand_cfg = cfg.randomization
        self.start_state = None

    def randomize_lighting(self):
        """
        Randomizes the lighting based on the configuration provided in `self.rand_cfg.lighting`.
        It uses the `randomize` function to compute the deltas for the lighting parameters.

        Returns:
            torch.Tensor: The randomized ambient light for all environments.
        """
        assert self.rand_cfg.get("lighting", None) is not None, (
            "Lighting configuration is not provided in the randomization config."
        )
        ambient_light = None
        if "ambient" in self.rand_cfg.lighting:
            current_ambient_light = self.scene.ambient_light
            ambient_light = randomize_array(
                **self.rand_cfg.lighting.ambient, base=current_ambient_light
            )
            self.set_ambient_light(ambient_light)
        log.info("Randomizing lighting. Only ambient light is supported for now.")
        return ambient_light

    def _pose_randomization(self, init_pos, init_rot, pos_to_rand, rot_to_rand):
        if pos_to_rand:
            pos_delta = []
            for _ in range(self.num_envs):
                pos_delta.append(
                    torch.tensor(randomize_array(**pos_to_rand), device=self.device)
                )
            pos_delta = torch.stack(pos_delta, dim=0)
        else:
            pos_delta = torch.zeros(self.num_envs, 3)
        if rot_to_rand:
            rot_delta = []
            for _ in range(self.num_envs):
                rot_delta.append(
                    torch.tensor(randomize_array(**rot_to_rand), device=self.device)
                )
            rot_delta = torch.stack(rot_delta, dim=0)
        else:
            rot_delta = torch.zeros(self.num_envs, 3)

        rand_pos = init_pos + pos_delta
        rand_quat = add_euler(init_rot, rot_delta, return_quat=True)

        rand_pose = Pose.create_from_pq(p=rand_pos, q=rand_quat)
        return rand_pose

    def randomize_sensor_camera(self):
        """
        Randomizes the camera position and rotation (Euler angles) based on the configuration
        provided in `self.rand_cfg.camera`. It uses the `randomize` function to compute the
        deltas for position and rotation.

        Returns:
            torch.Tensor: The stack of randomized camera poses for all environments.
        """

        log.info(
            "Randomizing camera. Only position and rotation are supported for now."
        )

        rot_to_rand = self.rand_cfg.camera.get("rot", None)
        pos_to_rand = self.rand_cfg.camera.get("pos", None)

        poses = []
        for i, cam_mount in enumerate(self.cam_mount):
            if self.cfg.camera.sensor.use_pose:
                init_pos = torch.tensor(
                    self.cfg.camera.sensor.pose[i][:3], device=self.device
                ).repeat(self.num_envs, 1)
                init_rot = torch.tensor(
                    self.cfg.camera.sensor.pose[i][3:], device=self.device
                ).repeat(self.num_envs, 1)
            else:
                eye = self.cfg.camera.sensor.eye[i]
                target = self.cfg.camera.sensor.target[i]
                init_pose = sapien_utils.look_at(eye=eye, target=target)
                init_pose = Pose.create(init_pose)
                init_pos = init_pose.p
                init_rot = init_pose.q

            rand_pose = self._pose_randomization(
                init_pos, init_rot, pos_to_rand, rot_to_rand
            )
            cam_mount.set_pose(rand_pose)

            poses.append(rand_pose.raw_pose.cpu())
        return torch.stack(poses, dim=0)

    def randomize_robot_pose(self):
        """
        Randomizes the robot's position and rotation (Euler angles) based on the configuration
        provided in `self.rand_cfg.robot`. It uses the `randomize` function to compute the
        deltas for position and rotation.

        Returns:
            torch.Tensor: The stack of randomized robot poses for all environments.
        """

        log.info("Randomizing robot pose. Only single agent is supported for now.")

        rot_to_rand = self.rand_cfg.robot.get("rot", None)
        pos_to_rand = self.rand_cfg.robot.get("pos", None)

        init_pos = torch.tensor(
            self.cfg.scene_builder.robot_init_pos, device=self.device
        ).repeat(self.num_envs, 1)
        init_rot = torch.tensor(
            self.cfg.scene_builder.robot_init_rot, device=self.device
        ).repeat(self.num_envs, 1)

        rand_pose = self._pose_randomization(
            init_pos, init_rot, pos_to_rand, rot_to_rand
        )

        self.agent.robot.set_pose(rand_pose)
        return rand_pose.raw_pose

    def randomize_obj_pose(self, obj, obj_type, set_z_to_height=True):
        """
        Randomizes the object's position and rotation (Euler angles) based on the configuration
        provided in `self.rand_cfg.object`. It uses the `randomize` function to compute the
        deltas for position and rotation.

        Args:
            obj (Actor): The object to randomize the pose.
            env_idx (torch.Tensor): The indices of the environments to randomize the object pose.

        Returns:
            None
        """

        log.info("Using cfg obj pose.")

        assert obj_type in [
            "manip_obj",
            "goal_obj",
        ], "Object type must be either 'manip_obj' or 'goal_obj'."

        rot_to_rand = self.rand_cfg.get(obj_type).get("rot", None)
        pos_to_rand = self.rand_cfg.get(obj_type).get("pos", None)

        init_pos = torch.tensor(
            self.cfg.get(obj_type).get("pos"), device=self.device
        ).repeat(self.num_envs, 1)
        init_rot = torch.tensor(
            self.cfg.get(obj_type).get("rot"), device=self.device
        ).repeat(self.num_envs, 1)
        if set_z_to_height:
            init_pos[:, 2] = obj.height

        rand_pose = self._pose_randomization(
            init_pos, init_rot, pos_to_rand, rot_to_rand
        )
        obj.set_pose(rand_pose)

    #############################################
    ########### For motion planning #############
    #############################################

    def collision_checker(self, new_cfg, active_env_idx):
        # TODO: make sure new_cfg is qpos
        qpos_to_set = self.agent.robot.get_qpos()
        qpos_to_set[active_env_idx] = new_cfg[active_env_idx]
        init_state = self.get_state_dict()
        self.agent.robot.set_qpos(qpos_to_set)
        action = np.zeros_like(self.agent.action_space.sample())
        if len(action.shape) == 1:
            action = action[None]
        self.step(action)
        if physx.is_gpu_enabled():
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()
        forces = self.agent.robot.get_net_contact_forces(
            list(self.agent.robot.links_map.keys())
        )
        has_collision = torch.linalg.norm(forces, dim=-1).sum(-1) > FORCE_THRESHOLD
        if has_collision.any():
            nonzero_env = [i.item() for i in has_collision.nonzero()]
            log.info(
                f"************ Collision detected for {nonzero_env}! **************"
            )
        self.set_state_dict(init_state)
        return has_collision

    def uniform_cfg_sampler(
        self, active_env_idx, pad_value, active_joint_indices, pad_inactive_joints_value
    ):
        qlimits = self.agent.robot.get_qlimits()  # (num_envs, 9, 2)

        # Separate the minimum and maximum limits
        q_min = qlimits[..., 0]  # Shape: (num_envs, num_joints)
        q_max = qlimits[..., 1]  # Shape: (num_envs, num_joints)

        # Select only the limits for the active joint indices
        active_q_min = q_min[
            ..., active_joint_indices
        ]  # Shape: (num_envs, len(active_joint_indices))
        active_q_max = q_max[
            ..., active_joint_indices
        ]  # Shape: (num_envs, len(active_joint_indices))

        # Initialize the output tensor with pad_value
        num_envs = qlimits.shape[0]
        num_joints = qlimits.shape[1]
        qpos = torch.full(
            (num_envs, num_joints), pad_value, device=self.device
        )  # Shape: (num_envs, num_joints)

        # # Uniformly sample random numbers between 0 and 1 for active joints
        # random_samples = torch.rand(active_q_min.shape, device=self.device)  # Shape: (num_active_envs, len(active_joint_indices))

        if active_env_idx.dtype == torch.bool:
            active_env_true_idx = torch.nonzero(active_env_idx).squeeze(1)
            inactive_env_true_idx = torch.nonzero(~active_env_idx).squeeze(1)

        else:
            active_env_true_idx = active_env_idx
            all_env_idx = torch.arange(
                self.num_envs, device=active_env_idx.device
            )  # Shape: (self.num_envs,)
            # Find the inactive indices using set difference
            inactive_env_true_idx = torch.tensor(
                list(
                    set(all_env_idx.cpu().numpy())
                    - set(active_env_true_idx.cpu().numpy())
                ),
                device=self.device,
            )

            # Optionally, sort the inactive indices
            inactive_env_true_idx = inactive_env_true_idx.sort().values

        # Uniformly sample random numbers between 0 and 1 for active joints in active environments
        random_samples = torch.rand(
            (len(active_env_true_idx), len(active_joint_indices)), device=self.device
        )  # Shape: (num_active_envs, len(active_joint_indices))

        # Sample only for active environments
        sampled_qpos = active_q_min[active_env_true_idx] + random_samples * (
            active_q_max[active_env_true_idx] - active_q_min[active_env_true_idx]
        )

        # Assign sampled values to the active joint indices for the active environments
        qpos[
            torch.tensor(active_env_true_idx, device=self.device).unsqueeze(1),
            torch.tensor(active_joint_indices, device=self.device),
        ] = sampled_qpos

        # Create a mask for inactive joint indices
        inactive_joint_mask = torch.ones(
            num_joints, dtype=torch.bool, device=self.device
        )
        inactive_joint_mask[active_joint_indices] = False  # Mark active joints as False

        # Set inactive joint positions to pad_inactive_joints_value
        qpos[:, inactive_joint_mask] = pad_inactive_joints_value
        qpos[inactive_env_true_idx, ...] = pad_value

        return qpos

    def save_start_state(self, extra_state_dict):
        state_dict = (
            self.get_state_dict()
        )  # Only include states of actors and articulations
        state_dict.update(extra_state_dict)
        self.start_state = state_dict

    def load_start_state(self, start_state_dict):
        sys_dict = {}
        randomization_dict = {}
        for key, value in start_state_dict.items():
            if key == "actors" or key == "articulations":
                sys_dict[key] = value
            else:
                randomization_dict[key] = value

        self.set_state_dict(sys_dict)
        assert set(randomization_dict.keys()) == {
            "lighting",
            "camera",
        }, "Randomization dict should have only keys lighting and camera."
        self._randomize_lighting(lighting=randomization_dict["lighting"])
        self.camera_pose = randomization_dict["camera"]
