"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import logging
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from guided_dc.agent.eval.eval_agent import EvalAgent
from guided_dc.utils.pose_utils import quaternion_to_euler_xyz

log = logging.getLogger(__name__)


def update(env_states, env_success, terminated, info, eval_step=0):
    new_env_success = torch.logical_or(env_success, terminated)
    env_states["success"] = new_env_success.cpu().numpy().copy()
    for i in range(len(new_env_success)):
        if new_env_success[i] != env_success[i]:
            env_states["eval_steps"][i] = eval_step
            env_states["env_elapsed_steps"][i] = info["elapsed_steps"].cpu().numpy()[i]
    env_success = new_env_success
    return env_states, env_success


def step_without_action(env):
    info = env.get_info()
    obs = env.get_obs(info)
    return (
        obs,
        info,
    )


class EvalAgentSim(EvalAgent):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def predict_single_step(self, obs, camera_indices=["0", "2"], bgr2rgb=False):
        self.model.eval()
        with torch.no_grad():
            cond = {}
            cond["state"] = self.process_multistep_state(obs=obs)
            cond["rgb"] = self.process_multistep_img(
                obs=obs,
                camera_indices=camera_indices,
                bgr2rgb=bgr2rgb,
            )
            print(
                "RGB dimensions:",
                [cond["rgb"][idx].shape for idx in cond["rgb"].keys()],
            )
            print("State dimensions:", cond["state"].shape)
            samples = (
                self.model.sample(cond=cond, deterministic=True)
                .trajectories.cpu()
                .numpy()
            )
            print(
                "Predicted action chunk dimensions:",
                samples.shape,
            )
            naction = samples[:, : self.act_steps]  # remove batch dimension
            if self.use_delta_actions:
                cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
                action = self.unnormalized_sim_delta_action(naction, cur_state)
                print("using delta")
            else:
                action = self.unnormalize_action(naction)
            return action

    def process_sim_observation(self, raw_obs):
        if isinstance(raw_obs, dict):
            raw_obs = [raw_obs]
        joint_state = raw_obs[0]["agent"]["qpos"][:, :7].cpu().numpy()
        gripper_state = raw_obs[0]["agent"]["qpos"][:, 7:8].cpu().numpy()
        # assert (gripper_state <= 0.04).all(), gripper_state
        gripper_state = 1 - gripper_state / 0.04  # 1 is closed, 0 is open
        gripper_state = np.where(gripper_state > 0.2, 1.0, 0.0)

        eef_pos_quat = raw_obs[0]["extra"]["tcp_pose"].cpu().numpy()
        # conver quaternion to euler angles
        eef_pos_euler = np.zeros((eef_pos_quat.shape[0], 6))
        eef_pos_euler[:, :3] = eef_pos_quat[:, :3]
        eef_pos_euler[:, 3:] = quaternion_to_euler_xyz(eef_pos_quat[:, 3:])

        images = {}
        try:
            images[self.camera_indices[0]] = (
                raw_obs[0]["sensor_data"]["sensor_0"]["rgb"].cpu().numpy()
            )
        except Exception:
            images[self.camera_indices[0]] = (
                raw_obs[0]["sensor_data"]["base_camera"]["rgb"].cpu().numpy()
            )
        if len(self.camera_indices) > 1:
            assert len(self.camera_indices) == 2
            images[self.camera_indices[1]] = (
                raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()
            )

        # wrist_img_resolution = (320, 240)
        # wrist_img = np.zeros(
        #     (len(images["2"]), wrist_img_resolution[1], wrist_img_resolution[0], 3)
        # )
        # for i in range(len(images["2"])):
        #     wrist_img[i] = cv2.resize(images["2"][i], wrist_img_resolution)
        # images["2"] = wrist_img

        obs = {
            "robot_state": {
                "joint_positions": joint_state,
                "gripper_position": gripper_state,
                "cartesian_position": eef_pos_euler,
            },
            "image": images,
        }
        return obs

    def postprocess_sim_gripper_action(self, action):
        action[..., -1] = -(action[..., -1] * 2 - 1)
        return action

    def run(self):
        # Reset env before iteration starts
        self.model.eval()
        if self.env.control_mode == "pd_ee_pose":
            action = self.env.agent.tcp_pose
        elif self.env.control_mode == "pd_joint_pos":
            action = self.env.robot_qpos[:, :8]
            action[:, -1] = 1.0
        else:
            raise NotImplementedError

        for _ in range(10):
            prev_obs, rew, terminated, truncated, info = self.env.step(action)
            self.env.render()

        env_success = torch.zeros(self.env.num_envs).to(torch.bool)
        env_states = {
            "object": {
                "manip_obj": self.env.manip_obj_pose,
                "goal_obj": self.env.goal_obj_pose,
            },
            "robot": {
                "qpos": self.env.robot_qpos,
                "base_pose": self.env.robot_base_pose,
            },
            "lighting": self.env.lighting.copy(),
            "camera_pose": self.env.camera_pose.copy(),
            "floor_texture_file": self.env.table_scene.floor_texture_file,
            "table_model_file": self.env.table_scene.table_model_file,
            "success": env_success.cpu().numpy(),
            "eval_steps": [0] * self.env.num_envs,
            "env_elapsed_steps": [0] * self.env.num_envs,
        }

        env_states, env_success = update(
            env_states,
            env_success,
            terminated.cpu(),  # type: ignore
            info,  # type: ignore
        )

        prev_obs = self.process_sim_observation(prev_obs)  # type: ignore
        obs = prev_obs
        np.set_printoptions(precision=3, suppress=True)
        # Check inference
        print("Warming up policy inference")
        with torch.no_grad():
            cond = {}
            cond["state"] = self.process_multistep_state(obs=obs, prev_obs=prev_obs)
            cond["rgb"] = self.process_multistep_img(
                obs=obs,
                camera_indices=self.camera_indices,
                prev_obs=prev_obs,
                bgr2rgb=False,
            )
            print(
                "RGB dimensions:",
                [cond["rgb"][idx].shape for idx in cond["rgb"].keys()],
            )
            print("State dimensions:", cond["state"].shape)
            samples = (
                self.model.sample(cond=cond, deterministic=True)
                .trajectories.cpu()
                .numpy()
            )
            print(
                "Predicted action chunk dimensions:",
                samples.shape,
            )
            naction = samples[:, : self.act_steps]
            prev_obs = obs
        if self.use_delta_actions:
            cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
            action = self.unnormalize_delta_action(naction, cur_state)
            print("*using delta")
        else:
            action = self.unnormalize_action(naction)
        action = self.postprocess_sim_gripper_action(action)
        print("Action:", action)
        print("States:", cond["state"].cpu().numpy())

        # Check images
        for i in self.camera_indices:
            plt.imshow(
                cond["rgb"][i][0, 0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            )
            plt.axis("off")  # Turn off the axes
            plt.gca().set_position(
                [0, 0, 1, 1]
            )  # Remove white space by adjusting the position of the axes
            plt.savefig(f"image_{i}.png", bbox_inches="tight", pad_inches=0)
            # plt.close()

        try:
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Run policy inference with observations
                pre_obs_start_time = time.time()
                with torch.no_grad():
                    cond = {}
                    cond["state"] = self.process_multistep_state(
                        obs=obs,
                        prev_obs=prev_obs,
                    )
                    cond["rgb"] = self.process_multistep_img(
                        obs=obs,
                        camera_indices=self.camera_indices,
                        prev_obs=prev_obs,
                        bgr2rgb=False,
                    )

                    pre_obs_end_time = time.time()
                    print(f"Pre-obs time: {pre_obs_end_time - pre_obs_start_time}")

                    # Run forward pass
                    samples = (
                        self.model.sample(cond=cond, deterministic=True)
                        .trajectories.cpu()
                        .numpy()
                    )
                    naction = samples[
                        :, : self.act_steps
                    ]  # (num_envs, act_steps, action_dim)

                if self.use_delta_actions:
                    cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
                    action = self.unnormalize_delta_action(naction, cur_state)
                    print("using delta")
                else:
                    action = self.unnormalize_action(naction)
                action = self.postprocess_sim_gripper_action(action)

                model_inf_end_time = time.time()
                print(f"Model inference time: {model_inf_end_time - pre_obs_end_time}")
                print("Action: ", action)

                # Run action chunk
                for action_step in range(self.act_steps):
                    a = action[:, action_step]
                    prev_obs = obs
                    step_start_time = time.time()

                    obs, rew, terminated, truncated, info = self.env.step(a)
                    # obs, rew, terminated, truncated, info = self.env.step(a)
                    env_states, env_success = update(
                        env_states, env_success, terminated.cpu(), info, eval_step=step
                    )
                    self.env.render()
                    step_end_time = time.time()
                    print(f"Step time: {step_end_time - step_start_time}")
                    obs = self.process_sim_observation(obs)

                    # time.sleep(0.03)
                if env_success.all():
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

        return env_states


class EvalAgentSimPi(EvalAgent):
    def __init__(self, cfg, env, policy):
        from openpi_client import image_tools

        self.image_tools = image_tools
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.n_cond_step = 1
        self.n_img_cond_step = 1
        self.open_loop_horizon = 8
        self.max_timesteps = 600
        self.camera_indices = ["0", "2"]

        self.env = env
        self.model = policy

        # Logging
        self.logdir = cfg.logdir
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.logdir, exist_ok=True)

    def process_sim_observation(self, raw_obs):
        if isinstance(raw_obs, dict):
            raw_obs = [raw_obs]
        joint_state = raw_obs[0]["agent"]["qpos"][:, :7].cpu().numpy()
        gripper_state = raw_obs[0]["agent"]["qpos"][:, 7:8].cpu().numpy()
        # assert (gripper_state <= 0.04).all(), gripper_state
        gripper_state = 1 - gripper_state / 0.04  # 1 is closed, 0 is open
        gripper_state = np.where(gripper_state > 0.2, 1.0, 0.0)

        eef_pos_quat = raw_obs[0]["extra"]["tcp_pose"].cpu().numpy()
        # conver quaternion to euler angles
        eef_pos_euler = np.zeros((eef_pos_quat.shape[0], 6))
        eef_pos_euler[:, :3] = eef_pos_quat[:, :3]
        eef_pos_euler[:, 3:] = quaternion_to_euler_xyz(eef_pos_quat[:, 3:])

        images = {}
        try:
            images[self.camera_indices[0]] = (
                raw_obs[0]["sensor_data"]["sensor_0"]["rgb"].cpu().numpy()
            )
        except Exception:
            images[self.camera_indices[0]] = (
                raw_obs[0]["sensor_data"]["base_camera"]["rgb"].cpu().numpy()
            )
        if len(self.camera_indices) > 1:
            assert len(self.camera_indices) == 2
            images[self.camera_indices[1]] = (
                raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()
            )

        obs = {
            "robot_state": {
                "joint_positions": joint_state,
                "gripper_position": gripper_state,
                "cartesian_position": eef_pos_euler,
            },
            "image": images,
        }
        return obs

    def postprocess_sim_gripper_action(self, action):
        action[..., -1] = -(action[..., -1] * 2 - 1)
        return action

    def run(self):
        # Reset env before iteration starts
        if self.env.control_mode == "pd_ee_pose":
            action = self.env.agent.tcp_pose
        elif self.env.control_mode == "pd_joint_pos":
            action = self.env.robot_qpos[:, :8]
            action[:, -1] = 1.0
        else:
            raise NotImplementedError

        for _ in range(10):
            obs, rew, terminated, truncated, info = self.env.step(action)
            self.env.render()

        env_success = torch.zeros(self.env.num_envs).to(torch.bool)
        env_states = {
            "object": {
                "manip_obj": self.env.manip_obj_pose,
                "goal_obj": self.env.goal_obj_pose,
            },
            "robot": {
                "qpos": self.env.robot_qpos,
                "base_pose": self.env.robot_base_pose,
            },
            "lighting": self.env.lighting.copy(),
            "camera_pose": self.env.camera_pose.copy(),
            "floor_texture_file": self.env.table_scene.floor_texture_file,
            "table_model_file": self.env.table_scene.table_model_file,
            "success": env_success.cpu().numpy(),
            "eval_steps": [0] * self.env.num_envs,
            "env_elapsed_steps": [0] * self.env.num_envs,
        }

        env_states, env_success = update(
            env_states,
            env_success,
            terminated.cpu(),  # type: ignore
            info,  # type: ignore
        )

        obs = self.process_sim_observation(obs)  # type: ignore
        np.set_printoptions(precision=3, suppress=True)
        # Check inference

        instruction = "pick up the tomato and put it into the metal plate"

        state = np.concatenate(
            [
                obs["robot_state"]["joint_positions"],
                obs["robot_state"]["gripper_position"],
            ],
            axis=1,
        )
        images = obs["image"]
        request_data = {
            "observation/image": self.image_tools.resize_with_pad(
                images["0"], 224, 224
            ),
            "observation/wrist_image": self.image_tools.resize_with_pad(
                images["2"], 224, 224
            ),
            "observation/state": state,
            "prompt": instruction,
            "batch_size": self.env.num_envs,
        }
        print(
            "RGB dimensions:",
            [images[idx].shape for idx in images.keys()],
        )
        print("State dimensions:", state.shape)
        while True:
            try:
                samples = self.model.infer(request_data)["actions"]
                break
            except:
                time.sleep(1)
        print(
            "Predicted action chunk dimensions:",
            samples.shape,
        )
        action = np.array(samples, copy=True)
        action = self.postprocess_sim_gripper_action(action)
        print("Action:", action)
        print("States:", state)

        # Check images
        for i in self.camera_indices:
            plt.imshow(images[i][0].astype(np.uint8))
            plt.axis("off")  # Turn off the axes
            plt.gca().set_position(
                [0, 0, 1, 1]
            )  # Remove white space by adjusting the position of the axes
            plt.savefig(f"image_{i}.png", bbox_inches="tight", pad_inches=0)
            # plt.close()

        bar = tqdm.tqdm(range(self.max_timesteps))

        for t_step in bar:
            state = np.concatenate(
                [
                    obs["robot_state"]["joint_positions"],
                    obs["robot_state"]["gripper_position"],
                ],
                axis=1,
            )
            images = obs["image"]
            request_data = {
                "observation/image": self.image_tools.resize_with_pad(
                    images["0"], 224, 224
                ),
                "observation/wrist_image": self.image_tools.resize_with_pad(
                    images["2"], 224, 224
                ),
                "observation/state": state,
                "prompt": instruction,
                "batch_size": self.env.num_envs,
            }
            st = time.time()
            while True:
                try:
                    pred_action_chunk = self.model.infer(request_data)["actions"]
                    break
                except Exception:
                    time.sleep(1)
            pred_action_chunk = np.array(pred_action_chunk, copy=True)
            et = time.time()
            print(f"Model inference time: {et - st}")
            # print("Action: ", pred_action_chunk)

            pred_action_chunk = self.postprocess_sim_gripper_action(pred_action_chunk)

            for action_step in range(self.open_loop_horizon):
                action = pred_action_chunk[:, action_step]
                obs, rew, terminated, truncated, info = self.env.step(action)
                env_states, env_success = update(
                    env_states,
                    env_success,
                    terminated.cpu() if hasattr(terminated, "cpu") else terminated,
                    info,
                    eval_step=t_step,
                )
                self.env.render()
                # time.sleep(0.03)
            obs = self.process_sim_observation(obs)
            if env_success.all():
                break

        return env_states


class EvalAgentPeg(EvalAgentSim):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def process_sim_observation(self, raw_obs):
        if isinstance(raw_obs, dict):
            raw_obs = [raw_obs]
        joint_state = raw_obs[0]["agent"]["qpos"][:, :7].cpu().numpy()
        gripper_state = raw_obs[0]["agent"]["qpos"][:, 7:8].cpu().numpy()
        # assert (gripper_state <= 0.04).all(), gripper_state
        gripper_state = 1 - gripper_state / 0.04  # 1 is closed, 0 is open
        gripper_state = np.where(gripper_state > 0.2, 1.0, 0.0)

        eef_pos_quat = raw_obs[0]["extra"]["tcp_pose"].cpu().numpy()
        # conver quaternion to euler angles
        eef_pos_euler = np.zeros((eef_pos_quat.shape[0], 6))
        eef_pos_euler[:, :3] = eef_pos_quat[:, :3]
        eef_pos_euler[:, 3:] = quaternion_to_euler_xyz(eef_pos_quat[:, 3:])

        force = torch.norm(
            self.env.goal_obj.get_net_contact_forces(), dim=-1, keepdim=True
        )

        images = {}
        try:
            images[self.camera_indices[0]] = (
                raw_obs[0]["sensor_data"]["sensor_0"]["rgb"].cpu().numpy()
            )
        except Exception:
            images[self.camera_indices[0]] = (
                raw_obs[0]["sensor_data"]["base_camera"]["rgb"].cpu().numpy()
            )
        if len(self.camera_indices) > 1:
            assert len(self.camera_indices) == 2
            images[self.camera_indices[1]] = (
                raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()
            )

        obs = {
            "robot_state": {
                "joint_positions": joint_state,
                "gripper_position": gripper_state,
                "cartesian_position": eef_pos_euler,
                "forces": force.cpu().numpy(),
            },
            "image": images,
        }
        return obs


class EvalAgentPull(EvalAgentSim):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def update(self, env_states, env_success, terminated, info, eval_step=0):
        new_env_success = torch.logical_or(env_success, terminated)
        env_states["success"] = new_env_success.cpu().numpy().copy()
        for i in range(len(new_env_success)):
            if new_env_success[i] != env_success[i]:
                env_states["eval_steps"][i] = eval_step
                env_states["env_elapsed_steps"][i] = (
                    info["elapsed_steps"].cpu().numpy()[i]
                )
        env_success = new_env_success
        pull_pct_new = info["pull_percentage"].cpu().numpy()  # (B,)
        # keep the larger of (stored, new) for every element
        better_mask = pull_pct_new > env_states["pull_percentage"]
        env_states["pull_percentage"][better_mask] = pull_pct_new[better_mask]

        self.success_45 = torch.logical_or(self.success_45, info["success_45"].cpu())
        self.success_50 = torch.logical_or(self.success_50, info["success_50"].cpu())
        self.success_55 = torch.logical_or(self.success_55, info["success_55"].cpu())

        env_states["success_45"] = self.success_45
        env_states["success_50"] = self.success_50
        env_states["success_55"] = self.success_55

        return env_states, env_success

    def run(self):
        # Reset env before iteration starts
        self.model.eval()
        if self.env.control_mode == "pd_ee_pose":
            action = self.env.agent.tcp_pose
        elif self.env.control_mode == "pd_joint_pos":
            action = self.env.robot_qpos[:, :8]
            action[:, -1] = 1.0
        else:
            raise NotImplementedError

        for _ in range(10):
            prev_obs, rew, terminated, truncated, info = self.env.step(action)
            self.env.render()

        env_success = torch.zeros(self.env.num_envs).to(torch.bool)

        self.success_45 = torch.zeros(self.env.num_envs).to(torch.bool)
        self.success_50 = torch.zeros(self.env.num_envs).to(torch.bool)
        self.success_55 = torch.zeros(self.env.num_envs).to(torch.bool)

        env_states = {
            "object": {
                "manip_obj": self.env.manip_obj_pose,
                "goal_obj": self.env.goal_obj_pose,
            },
            "robot": {
                "qpos": self.env.robot_qpos,
                "base_pose": self.env.robot_base_pose,
            },
            "lighting": self.env.lighting.copy(),
            "camera_pose": self.env.camera_pose.copy(),
            "floor_texture_file": self.env.table_scene.floor_texture_file,
            "table_model_file": self.env.table_scene.table_model_file,
            "success": env_success.cpu().numpy(),
            "eval_steps": [0] * self.env.num_envs,
            "env_elapsed_steps": [0] * self.env.num_envs,
            "pull_percentage": np.zeros(self.env.num_envs),
            "success_45": self.success_45,
            "success_50": self.success_50,
            "success_55": self.success_55,
        }

        env_states, env_success = self.update(
            env_states,
            env_success,
            terminated.cpu(),  # type: ignore
            info,  # type: ignore
        )

        prev_obs = self.process_sim_observation(prev_obs)  # type: ignore
        obs = prev_obs
        np.set_printoptions(precision=3, suppress=True)
        # Check inference
        print("Warming up policy inference")
        with torch.no_grad():
            cond = {}
            cond["state"] = self.process_multistep_state(obs=obs, prev_obs=prev_obs)
            cond["rgb"] = self.process_multistep_img(
                obs=obs,
                camera_indices=self.camera_indices,
                prev_obs=prev_obs,
                bgr2rgb=False,
            )
            print(
                "RGB dimensions:",
                [cond["rgb"][idx].shape for idx in cond["rgb"].keys()],
            )
            print("State dimensions:", cond["state"].shape)
            samples = (
                self.model.sample(cond=cond, deterministic=True)
                .trajectories.cpu()
                .numpy()
            )
            print(
                "Predicted action chunk dimensions:",
                samples.shape,
            )
            naction = samples[:, : self.act_steps]
            prev_obs = obs
        if self.use_delta_actions:
            cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
            action = self.unnormalize_delta_action(naction, cur_state)
            print("*using delta")
        else:
            action = self.unnormalize_action(naction)
        action = self.postprocess_sim_gripper_action(action)
        print("Action:", action)
        print("States:", cond["state"].cpu().numpy())

        # Check images
        for i in self.camera_indices:
            plt.imshow(
                cond["rgb"][i][0, 0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            )
            plt.axis("off")  # Turn off the axes
            plt.gca().set_position(
                [0, 0, 1, 1]
            )  # Remove white space by adjusting the position of the axes
            plt.savefig(f"image_{i}.png", bbox_inches="tight", pad_inches=0)
            # plt.close()
        try:
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Run policy inference with observations
                pre_obs_start_time = time.time()
                with torch.no_grad():
                    cond = {}
                    cond["state"] = self.process_multistep_state(
                        obs=obs,
                        prev_obs=prev_obs,
                    )
                    cond["rgb"] = self.process_multistep_img(
                        obs=obs,
                        camera_indices=self.camera_indices,
                        prev_obs=prev_obs,
                        bgr2rgb=False,
                    )
                    pre_obs_end_time = time.time()
                    print(f"Pre-obs time: {pre_obs_end_time - pre_obs_start_time}")

                    # Run forward pass
                    samples = (
                        self.model.sample(cond=cond, deterministic=True)
                        .trajectories.cpu()
                        .numpy()
                    )
                    naction = samples[
                        :, : self.act_steps
                    ]  # (num_envs, act_steps, action_dim)

                if self.use_delta_actions:
                    cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
                    action = self.unnormalize_delta_action(naction, cur_state)
                    print("using delta")
                else:
                    action = self.unnormalize_action(naction)
                action = self.postprocess_sim_gripper_action(action)
                # actions.append(action)

                # Debug
                model_inf_end_time = time.time()
                print(f"Model inference time: {model_inf_end_time - pre_obs_end_time}")
                print("Action: ", action)

                # Run action chunk
                for action_step in range(self.act_steps):
                    a = action[:, action_step]
                    prev_obs = obs
                    step_start_time = time.time()

                    obs, rew, terminated, truncated, info = self.env.step(a)
                    env_states, env_success = self.update(
                        env_states, env_success, terminated.cpu(), info, eval_step=step
                    )
                    self.env.render()
                    step_end_time = time.time()
                    print(f"Step time: {step_end_time - step_start_time}")
                    obs = self.process_sim_observation(obs)
                if env_success.all():
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

        env_states["success_45"] = env_states["success_45"].numpy()
        env_states["success_50"] = env_states["success_50"].numpy()
        env_states["success_55"] = env_states["success_55"].numpy()

        return env_states
