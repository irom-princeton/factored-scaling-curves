"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from guided_dc.agent.eval.eval_agent import EvalAgent

log = logging.getLogger(__name__)


class EvalAgentReal(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg, env=None)

    def run(self):
        # Reset env before iteration starts
        self.model.eval()
        prev_obs = self.reset_env()
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
                bgr2rgb=True,
            )
            print("Using camera indices:", self.camera_indices)
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
            naction = samples[0, : self.act_steps]  # remove batch dimension
            prev_obs = obs
        if self.use_delta_actions:
            cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
            action = self.unnormalize_delta_action(naction, cur_state)
        else:
            action = self.unnormalize_action(naction)
        print("Action:", action)

        # Check images
        for i in self.camera_indices:
            plt.imshow(
                cond["rgb"][i][0, 0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            )
            plt.savefig(f"image_{i}.png")
        input("Check images and then press anything to continue...")

        # TODO: some safety check making sure the sample predicted actions are not crazy

        # Run
        cond_states = []
        images = [[], []]
        actions = []
        robot_states = [
            np.concatenate(
                [
                    obs["robot_state"]["cartesian_position"],
                    [obs["robot_state"]["gripper_position"]],
                ]
            )
        ]
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
                        bgr2rgb=True,
                    )
                    cond_states.append(cond["state"].cpu().numpy())
                    for i, k in enumerate(cond["rgb"].keys()):
                        images[i].append(cond["rgb"][k].cpu().numpy())

                    print(
                        "State: ",
                        self.unnormalize_obs(cond["state"].cpu().numpy()),
                    )
                    pre_obs_end_time = time.time()
                    print(f"Pre-obs time: {pre_obs_end_time - pre_obs_start_time}")

                    # Run forward pass
                    samples = (
                        self.model.sample(cond=cond, deterministic=True)
                        .trajectories.cpu()
                        .numpy()
                    )
                    naction = samples[0, : self.act_steps]  # remove batch dimension
                if self.use_delta_actions:
                    cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
                    action = self.unnormalize_delta_action(naction, cur_state)
                else:
                    action = self.unnormalize_action(naction)
                actions.append(action)

                # Debug
                model_inf_end_time = time.time()
                print(f"Model inference time: {model_inf_end_time - pre_obs_end_time}")
                print("Action: ", action)

                # Run action chunk
                for a in action:
                    prev_obs = obs
                    self.env.step(a)
                    obs = self.env.get_observation()
                    robot_states.append(
                        np.concatenate(
                            [
                                obs["robot_state"]["cartesian_position"],
                                [obs["robot_state"]["gripper_position"]],
                            ]
                        )
                    )
                    time.sleep(0.06)

                # Debug
                step_end_time = time.time()
                print(f"Step time: {step_end_time - model_inf_end_time}")
        except KeyboardInterrupt:
            print("Interrupted by user")

        # Save data
        np.savez_compressed(
            self.result_path,
            actions=np.array(actions),
            robot_states=np.array(robot_states),
            cond_states=np.array(cond_states),
            images=np.array(images),
        )
