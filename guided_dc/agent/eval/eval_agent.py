"""
Parent eval agent class.

"""

import logging
import os
import random

import hydra
import numpy as np
import torch

log = logging.getLogger(__name__)


class EvalAgent:
    def __init__(self, cfg, env=None):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # shm_manager = SharedMemoryManager()

        # Set obs dim -  we will save the different obs in batch in a dict
        self.ordered_obs_keys = cfg.ordered_obs_keys
        self.normalization_stats_path = cfg.normalization_stats_path
        self.normalization_stats = np.load(
            self.normalization_stats_path, allow_pickle=True
        )
        self.obs_min = self.normalization_stats["obs_min"]
        self.obs_max = self.normalization_stats["obs_max"]
        self.action_min = self.normalization_stats["action_min"]
        self.action_max = self.normalization_stats["action_max"]

        self.n_cond_step = cfg.task.cond_steps
        self.n_img_cond_step = cfg.task.img_cond_steps
        self.obs_dim = cfg.task.obs_dim
        self.action_dim = cfg.task.action_dim
        self.horizon_steps = cfg.task.horizon_steps
        self.act_steps = min(cfg.act_steps, self.horizon_steps)
        self.use_delta_actions = cfg.get("use_delta_actions", False)
        self.camera_indices = cfg.camera_indices
        if self.use_delta_actions:
            self.delta_min = self.normalization_stats["delta_min"].astype(np.float32)
            self.delta_max = self.normalization_stats["delta_max"].astype(np.float32)
        self.debug = cfg.get("eval_debug", False)

        # Initialize environment
        if env is None:
            if self.debug:
                from guided_dc.envs.dummy_env import DummyEnv

                self.env = DummyEnv(img_size=cfg.task.img_size)
            else:
                from droid.robot_env import RobotEnv  # type: ignore

                self.env = RobotEnv(
                    robot_type="panda",
                    action_space=cfg.action_space,
                    gripper_action_space="position",
                    camera_resolution=(
                        cfg.task.img_size[2],
                        cfg.task.img_size[1],
                    ),  # TODO: resolutions for different cameras
                )
        else:
            self.env = env

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.policy)

        # Eval params
        self.n_steps = cfg.n_steps

        # Logging
        self.logdir = cfg.logdir
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.logdir, exist_ok=True)

    def run(self):
        raise NotImplementedError

    def reset_env(self, randomize_reset=False):
        self.env.reset(randomize=randomize_reset)
        obs = self.env.get_observation()
        return obs

    def reset_sim_env(self):
        obs, info = self.env.reset()
        return obs

    def unnormalize_action(self, naction):
        action = (naction + 1) * (
            self.action_max - self.action_min + 1e-6
        ) / 2 + self.action_min
        return action

    def unnormalize_delta_action(self, naction: np.ndarray, state: np.ndarray):
        action = (naction[..., :-1] + 1) * (
            self.delta_max - self.delta_min + 1e-6
        ) / 2 + self.delta_min
        if len(action.shape) == 2:
            action += state[:, -1, :-1]  # skip gripper
        else:
            action += state[:, -1:, :-1]
        gripper_action = (naction[..., -1:] + 1) * (
            self.action_max[-1] - self.action_min[-1] + 1e-6
        ) / 2 + self.action_min[-1]
        return np.concatenate([action, gripper_action], axis=-1)

    def normalize_obs(self, obs):
        """concatenate and normalize obs"""
        processed_obs = []
        for i in range(len(self.ordered_obs_keys)):
            o = obs["robot_state"][self.ordered_obs_keys[i]]
            o = np.array(o)
            if len(o.shape) == 1:
                o = o.reshape(1, -1)
            elif len(o.shape) == 0:
                o = o.reshape(1, 1)
            assert len(o.shape) == 2, o.shape
            processed_obs.append(o)

        processed_obs = np.concatenate(processed_obs, axis=1)
        processed_obs = (
            2 * (processed_obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6)
            - 1
        )
        processed_obs = np.clip(processed_obs, -1, 1)
        # If any of the processed_obs reach the limits, print a warning and show which obs reached the limits
        # if np.any(np.abs(processed_obs[:, :-1]) == 1):
        #     raise ValueError(
        #         f"obs reached limits, {np.where(np.abs(processed_obs[:, :-1]) == 1)}"
        #     )

        # processed_obs[0,3] = np.abs(processed_obs[0,3])
        return processed_obs

    def unnormalize_obs(self, state):
        return (state + 1) * (self.obs_max - self.obs_min + 1e-6) / 2 + self.obs_min

    def process_multistep_state(self, obs, prev_obs=None):
        if self.n_cond_step == 2:
            assert prev_obs
            ret = np.stack(
                (
                    self.normalize_obs(prev_obs),
                    self.normalize_obs(obs),
                ),
                axis=1,
            )
        else:
            assert self.n_cond_step == 1
            ret = self.normalize_obs(obs)[:, None]
        ret = torch.from_numpy(ret).float().to(self.device)

        return ret

    def process_multistep_img(self, obs, camera_indices, prev_obs=None, bgr2rgb=False):
        if self.n_img_cond_step == 2:  # TODO: better logic
            assert prev_obs
            images = {}
            for idx in camera_indices:
                if len(obs["image"][idx].shape) == 3:
                    image_1 = obs["image"][idx][None].transpose(0, 3, 1, 2).copy()
                else:
                    image_1 = obs["image"][idx].transpose(0, 3, 1, 2).copy()
                if len(prev_obs["image"][idx].shape) == 3:
                    image_2 = prev_obs["image"][idx][None].transpose(0, 3, 1, 2).copy()
                else:
                    image_2 = prev_obs["image"][idx].transpose(0, 3, 1, 2).copy()
                images[idx] = np.stack(
                    (
                        image_1,
                        image_2,
                    ),
                    axis=1,
                )
                if bgr2rgb:
                    images[idx] = images[idx][:, :, ::-1, :, :].copy()
                images[idx] = torch.from_numpy(images[idx]).to(self.device).float()
                # assert images[idx].shape == (1, 2, 3, 96, 96), images[idx].shape
        else:
            images = {}

            for idx in camera_indices:
                image = obs["image"][idx]
                if len(image.shape) == 3:
                    image = image[None]
                image = image.transpose(0, 3, 1, 2).copy()

                # Insert a cond_step dimension
                images[idx] = np.expand_dims(image, axis=1)
                if bgr2rgb:
                    images[idx] = images[idx][:, :, ::-1, :, :].copy()

                images[idx] = torch.from_numpy(images[idx]).to(self.device).float()
                # assert images[idx].shape == (2, 1, 3, 192, 192), images[idx].shape
        return images
