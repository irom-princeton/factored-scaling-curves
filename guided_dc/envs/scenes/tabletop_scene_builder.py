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

These arguments can be passed via the `options` dictionary to override default configurations.
"""

from typing import Any, Dict

import numpy as np
import sapien
import torch
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from sapien.render import RenderMaterial, RenderTexture2D
from transforms3d.euler import euler2quat


class TabletopSceneBuilder(SceneBuilder):
    def __init__(self, env, cfg, robot_init_qpos_noise=0.02):
        super().__init__(env=env, robot_init_qpos_noise=robot_init_qpos_noise)
        self.base_cfg = cfg
        self.default_options = {
            "robot_init_qpos": cfg.robot_init_qpos,
            "robot_init_euler": cfg.robot_init_rot,
            "robot_init_pos": np.array(cfg.robot_init_pos),
            "table_model_file": cfg.table.model_file,
            "floor_texture_file": cfg.floor_texture_file,
            "table_height": cfg.table.table_height,
            "table_scale": cfg.table.get("scale", [1, 1, 1]),
            "background": cfg.background,
            "table_orientation": euler2quat(0, 0, 0),
        }
        self.options = None

    @property
    def floor_texture_file(self):
        return self.options.get("floor_texture_file", None)

    @property
    def table_model_file(self):
        return self.options.get("table_model_file", None)

    def build_background_and_floor(self, options: Dict[str, Any]):
        background = options.get("background", {})
        background_model_file = background.get("model_file")

        load_table_struc = options.get("load_table_struc", True)

        if load_table_struc:
            background_pos = np.array(
                background.get("pos", self.base_cfg.background.pos)
            )
            background_rot = np.array(
                background.get("rot", self.base_cfg.background.rot)
            )
            background_pose = sapien.Pose(
                p=background_pos + np.array([0, 0, 0]),
                q=euler2quat(*background_rot),
            )
            builder2 = self.scene.create_actor_builder()
            builder2.add_visual_from_file(
                filename="guided_dc/assets/background/table_struc.obj",
                scale=[1, 1, 1],
                pose=background_pose,
            )
            builder2.initial_pose = background_pose
            background_fixed = builder2.build_kinematic(name="background-fixed")
            self.scene_objects.append(background_fixed)

        if background_model_file.endswith(("jpg", "png", "jpeg")):
            # Create a virtual plane for displaying the image
            floor_width = 500 if self.scene.parallel_in_single_scene else 100
            self.ground = build_ground(
                self.scene,
                floor_width=floor_width,
                altitude=-self.base_cfg.table.thickness
                - self.base_cfg.table.leg_length,
                texture_file=background_model_file,
            )
        else:
            if load_table_struc:
                assert background_model_file.endswith("obj")
                background_pos = np.array(
                    background.get("pos", self.base_cfg.background.pos)
                )
                background_rot = np.array(
                    background.get("rot", self.base_cfg.background.rot)
                )
                background_pose = sapien.Pose(
                    p=background_pos + np.array([0, 0, 0]),
                    q=euler2quat(*background_rot),
                )
                builder = self.scene.create_actor_builder()
                builder.add_visual_from_file(
                    filename=background_model_file,
                    scale=[1, 1, 1],
                    pose=background_pose,
                )
                builder.initial_pose = background_pose
                background = builder.build_kinematic(name="background")
                self.scene_objects.append(background)

            floor_width = 500 if self.scene.parallel_in_single_scene else 100
            self.ground = build_ground(
                self.scene,
                floor_width=floor_width,
                altitude=-self.base_cfg.table.thickness
                - self.base_cfg.table.leg_length,
                texture_file=options["floor_texture_file"],
            )
        self.scene_objects.append(self.ground)

    def build(self, options: Dict[str, Any]):
        options = {**self.default_options, **options}  # Override defaults with options
        self.options = options

        # 1. Build the table
        builder = self.scene.create_actor_builder()
        table_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        builder.add_nonconvex_collision_from_file(
            filename=options["table_model_file"],
            scale=options.get("table_scale", [1.0, 1.0, 1.0]),
            pose=table_pose,
        )
        # builder.add_box_collision(  # TODO: Table-specific, but potentially faster.
        #     pose=sapien.Pose(p=[0, 0, -0.025]),
        #     half_size=(0.7, 0.3575, 0.025),
        # )
        builder.add_visual_from_file(
            filename=options["table_model_file"],
            scale=options.get("table_scale", [1.0, 1.0, 1.0]),
            pose=table_pose,
            material=self._get_table_material(options),
        )
        builder.initial_pose = table_pose
        self.table = builder.build_kinematic(name="table-workspace")
        # aabb = (
        #     table._objs[0]
        #     .find_component_by_type(sapien.render.RenderBodyComponent)
        #     .compute_global_aabb_tight()
        # )
        # self.table_length = aabb[1, 0] - aabb[0, 0]
        # self.table_width = aabb[1, 1] - aabb[0, 1]
        # self.table_height = aabb[1, 2] - aabb[0, 2]
        # print(self.table_length, self.table_width)
        # breakpoint()
        # self.table_thickness = self.base_cfg.table.thickness
        self.scene_objects = [self.table]
        self.build_background_and_floor(options)

    def _get_table_material(self, options: Dict[str, Any]):
        table_material = RenderMaterial()
        material_override = options.get("table_texture", None)

        if material_override:
            if isinstance(material_override, str):
                table_material.base_color_texture = RenderTexture2D(
                    filename=material_override, srgb=True
                )
                table_material.diffuse_texture = RenderTexture2D(
                    filename=material_override, srgb=True
                )
            elif isinstance(material_override, list) and len(material_override) == 4:
                table_material.base_color = material_override
        else:
            if self.base_cfg.table.material_type == "file":
                table_material.base_color_texture = RenderTexture2D(
                    filename=self.base_cfg.table.material_file, srgb=True
                )
                table_material.diffuse_texture = RenderTexture2D(
                    filename=self.base_cfg.table.material_file, srgb=True
                )
            elif self.base_cfg.table.material_type == "color":
                table_material.base_color = self.base_cfg.table.material_color
            else:
                raise NotImplementedError("Unknown table material type")

        table_material.ior = 1.4  # Typical for plastic/laminated surfaces
        table_material.specular = 0.6  # Moderate specular reflection
        table_material.roughness = 0.9  # Very rough, like paper
        # table_material.metallic = 0.4
        # Specular reflection (minimal for paper)
        # table_material.specular = 0.6  # Low specular reflection
        return table_material

    def initialize(
        self, env_idx: torch.Tensor, options: Dict[str, Any] = {}, randomize_qpos=False
    ):
        options = {**self.default_options, **options}  # Override defaults with options
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(
                p=[0, 0, options.get("table_height", 0)], q=options["table_orientation"]
            )
        )

        if self.env.robot_uids in [
            "panda",
            "panda_wristcam",
            "panda_wristcam_irom",
            "panda_wristcam_peg",
        ]:
            qpos = np.array(options["robot_init_qpos"])
            if randomize_qpos:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
                qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose(
                    p=options["robot_init_pos"],
                    q=euler2quat(*options["robot_init_euler"]),
                )
            )
        elif self.env.robot_uids in [
            ("panda", "panda"),
            ("panda_wristcam", "panda_wristcam"),
        ]:
            assert (
                isinstance(options["robot_init_pos"], list)
                and isinstance(options["robot_init_euler"], list)
                and len(options["robot_init_pos"]) == 2
                and len(options["robot_init_euler"]) == 2
            ), "robot_init_pos and robot_init_euler should be lists of length 2"

            agent: MultiAgent = self.env.agent
            qpos = np.array(options["robot_init_qpos"])
            if randomize_qpos:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
                qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose(
                    p=options["robot_init_pos"][0],
                    q=euler2quat(*options["robot_init_euler"][0]),
                )
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose(
                    p=options["robot_init_pos"][1],
                    q=euler2quat(*options["robot_init_euler"][1]),
                )
            )
        else:
            raise NotImplementedError("Unknown robot_uids")
