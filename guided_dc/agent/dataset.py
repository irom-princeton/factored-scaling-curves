"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

import logging
import os
import random
from collections import namedtuple

import cv2
import numpy as np
import torch

Batch = namedtuple("Batch", "actions conditions")
log = logging.getLogger(__name__)


class BaseSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        logdir,
        dataset_norm_stats_path=None,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cuda:0",
        store_gpu=False,
        use_delta_actions=True,
        use_raw=False,
        get_anomaly_dataset_flag=True,
    ):
        assert img_cond_steps <= cond_steps, (
            "consider using more cond_steps than img_cond_steps"
        )
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path
        self.store_gpu = store_gpu
        self.use_raw = use_raw
        # Load dataset to device specified
        assert dataset_path.endswith(".npz")
        dataset = np.load(dataset_path, allow_pickle=True)  # only np arrays
        images = dataset["images"]
        if images.dtype == np.dtype("O"):
            images = images.item()
        else:
            raise NotImplementedError("Only support dict of images for now")

        # Use first max_n_episodes episodes
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)

        # Subsample if needed
        # Load states and actions
        if self.use_raw:
            assert total_num_steps == len(dataset["raw_states"])
            assert total_num_steps == len(dataset["raw_actions"])
            self.states = dataset["raw_states"][:total_num_steps].astype(np.float32)
            self.actions = dataset["raw_actions"][:total_num_steps].astype(np.float32)
        else:
            self.states = dataset["states"][:total_num_steps].astype(np.float32)
            self.actions = dataset["actions"][:total_num_steps].astype(np.float32)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps, self.cond_steps)

        if store_gpu:
            self.states = torch.from_numpy(self.states).to(device)
            self.actions = torch.from_numpy(self.actions).to(device)
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")

        # Load images
        if self.use_img:
            self.images = {}
            for idx in images.keys():
                self.images[idx] = images[idx][:total_num_steps]
                if store_gpu:
                    self.images[idx] = torch.from_numpy(self.images[idx]).to(device)
            log.info(f"Loading multiple images from {len(self.images)} cameras")
            log.info(
                f"Images shape/type: {self.images[next(iter(images.keys()))].shape, self.images[next(iter(images.keys()))].dtype}"
            )

        if dataset_norm_stats_path:
            normalization = np.load(dataset_norm_stats_path)
            np.savez(
                f"{logdir}/norm.npz",
                obs_min=normalization["obs_min"],
                obs_max=normalization["obs_max"],
                action_min=normalization["action_min"],
                action_max=normalization["action_max"],
                delta_min=normalization["delta_min"],
                delta_max=normalization["delta_max"],
            )

        # Delta actions --- action chunk relative to current state
        self.use_delta_actions = use_delta_actions
        if self.use_delta_actions:
            assert dataset_norm_stats_path
            normalization = np.load(dataset_norm_stats_path)
            self.delta_min = normalization["delta_min"].astype(np.float32)
            self.delta_max = normalization["delta_max"].astype(np.float32)
            self.raw_states = dataset["raw_states"][:total_num_steps].astype(np.float32)
            self.raw_actions = dataset["raw_actions"][:total_num_steps].astype(
                np.float32
            )
            if store_gpu:
                self.delta_min = torch.from_numpy(self.delta_min).to(device)
                self.delta_max = torch.from_numpy(self.delta_max).to(device)
                self.raw_states = torch.from_numpy(self.raw_states).to(device)
                self.raw_actions = torch.from_numpy(self.raw_actions).to(device)

        if get_anomaly_dataset_flag:
            num_traj = len(traj_lengths)
            anomaly_images = {key: [] for key in self.images.keys()}
            first_frame_idx = 0
            for traj_idx in range(num_traj):
                for key in self.images.keys():
                    anomaly_images[key].append(self.images[key][first_frame_idx])
                first_frame_idx += traj_lengths[traj_idx]
            np.savez(
                dataset_path.replace("dataset.npz", "anomaly_dataset.npz"),
                images=anomaly_images,
            )

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        stack_pkg = torch if self.store_gpu else np

        # extract states and actions
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : (start + 1)]
        states = stack_pkg.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            images = {}
            for idx in self.images.keys():
                images[idx] = self.images[idx][(start - num_before_start) : end]
                images[idx] = stack_pkg.stack(
                    [
                        images[idx][max(num_before_start - t, 0)]
                        for t in reversed(range(self.img_cond_steps))
                    ]
                )
            conditions["rgb"] = images

        # extract actions
        # TODO: assume absolute action right now, and both state and action use joint or cartesian
        if self.use_delta_actions:  # subtrct current state
            if self.store_gpu:
                raw_action = self.raw_actions[start:end][
                    :, :-1
                ].clone()  # skip the gripper
            else:
                raw_action = self.raw_actions[start:end][:, :-1].copy()
            raw_cur_state = self.raw_states[start : (start + 1), : raw_action.shape[1]]
            raw_action -= raw_cur_state
            action = (
                2
                * (raw_action - self.delta_min)
                / (self.delta_max - self.delta_min + 1e-6)
                - 1
            )
            gripper_action = self.actions[start:end, -1:]  # normalized
            if self.store_gpu:
                actions = torch.cat([action, gripper_action], dim=-1)
            else:
                actions = np.concatenate([action, gripper_action], axis=-1)
        else:
            actions = self.actions[start:end]  # normalized absolute actions
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps, cond_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for _, traj_length in enumerate(traj_lengths):
            min_start = cur_traj_index + cond_steps - 1
            max_start = cur_traj_index + traj_length - horizon_steps
            indices_to_add = [
                (i, i - cur_traj_index) for i in range(min_start, max_start + 1)
            ]
            indices += indices_to_add
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [ind for ind in self.indices if ind not in train_indices]
        self.set_indices(train_indices)
        return val_indices

    def set_indices(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)


class RealSequenceDataset(BaseSequenceDataset, torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    reset_state = np.array(
        [0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0, 0]
    )

    def __init__(
        self,
        dataset_path,
        logdir,
        dataset_norm_stats_path=None,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cuda:0",
        store_gpu=False,
        use_delta_actions=True,
        use_raw=False,
        get_anomaly_dataset_flag=True,
        sim_traj_num=0,
        filter_first_state=False,
    ):
        torch.utils.data.Dataset.__init__(self)
        assert img_cond_steps <= cond_steps, (
            "consider using more cond_steps than img_cond_steps"
        )

        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path
        self.store_gpu = store_gpu
        self.use_raw = use_raw
        self.sim_traj_num = sim_traj_num
        # Load dataset to device specified
        assert dataset_path.endswith(".npz")
        dataset = np.load(dataset_path, allow_pickle=True)  # only np arrays
        images = dataset["images"]
        if images.dtype == np.dtype("O"):
            images = images.item()
        else:
            raise NotImplementedError("Only support dict of images for now")

        # Use first max_n_episodes episodes
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)

        # Load states and actions
        if self.use_raw:
            assert total_num_steps == len(dataset["raw_states"])
            assert total_num_steps == len(dataset["raw_actions"])
            self.states = dataset["raw_states"][:total_num_steps].astype(np.float32)
            self.actions = dataset["raw_actions"][:total_num_steps].astype(np.float32)
        else:
            self.states = dataset["states"][:total_num_steps].astype(np.float32)
            self.actions = dataset["actions"][:total_num_steps].astype(np.float32)
        if dataset_norm_stats_path:
            normalization = np.load(dataset_norm_stats_path)
            np.savez(
                f"{logdir}/norm.npz",
                obs_min=normalization["obs_min"],
                obs_max=normalization["obs_max"],
                action_min=normalization["action_min"],
                action_max=normalization["action_max"],
                delta_min=normalization["delta_min"],
                delta_max=normalization["delta_max"],
            )

        # Set up indices for sampling
        self.indices, self.sim_traj_flags = self.make_indices(
            traj_lengths, horizon_steps, self.cond_steps
        )

        if store_gpu:
            self.states = torch.from_numpy(self.states).to(device)
            self.actions = torch.from_numpy(self.actions).to(device)
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")

        # Load images
        if self.use_img:
            self.images = {}
            for idx in images.keys():
                self.images[idx] = images[idx][:total_num_steps]
                if store_gpu:
                    self.images[idx] = torch.from_numpy(self.images[idx]).to(device)
            log.info(f"Loading multiple images from {len(self.images)} cameras")
            log.info(
                f"Images shape/type: {self.images[next(iter(images.keys()))].shape, self.images[next(iter(images.keys()))].dtype}"
            )

        # Delta actions --- action chunk relative to current state
        self.use_delta_actions = use_delta_actions
        if self.use_delta_actions:
            assert dataset_norm_stats_path
            normalization = np.load(dataset_norm_stats_path)
            self.delta_min = normalization["delta_min"].astype(np.float32)
            self.delta_max = normalization["delta_max"].astype(np.float32)
            self.raw_states = dataset["raw_states"][:total_num_steps].astype(np.float32)
            self.raw_actions = dataset["raw_actions"][:total_num_steps].astype(
                np.float32
            )
            if store_gpu:
                self.delta_min = torch.from_numpy(self.delta_min).to(device)
                self.delta_max = torch.from_numpy(self.delta_max).to(device)
                self.raw_states = torch.from_numpy(self.raw_states).to(device)
                self.raw_actions = torch.from_numpy(self.raw_actions).to(device)

        if get_anomaly_dataset_flag:
            states = dataset["raw_states"]
            num_traj = len(traj_lengths)
            anomaly_images = {key: [] for key in self.images.keys()}

            first_frame_idx = 0

            for traj_idx in range(num_traj - sim_traj_num):
                if filter_first_state:
                    states_i = states[
                        first_frame_idx : first_frame_idx + traj_lengths[traj_idx]
                    ]
                    found_reset_state = False
                    # Check if the states are close to the reset state
                    state_idx = 0
                    for state_idx in range(len(states_i)):
                        if np.all(
                            np.abs(states_i[state_idx] - self.reset_state) < 5e-2
                        ):
                            found_reset_state = True
                            state_idx += 4
                            break
                    if found_reset_state:
                        for key in self.images.keys():
                            anomaly_images[key].append(
                                self.images[key][first_frame_idx + state_idx]
                            )
                else:
                    for key in self.images.keys():
                        anomaly_images[key].append(self.images[key][first_frame_idx])
                first_frame_idx += traj_lengths[traj_idx]

            np.savez(
                dataset_path.replace("dataset.npz", "anomaly_dataset.npz"),
                images=anomaly_images,
            )

    def make_indices(self, traj_lengths, horizon_steps, cond_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        sim_traj_flags = []
        total_traj_num = len(traj_lengths)
        cur_traj_index = 0
        for traj_idx, traj_length in enumerate(traj_lengths):
            min_start = cur_traj_index + cond_steps - 1
            max_start = cur_traj_index + traj_length - horizon_steps
            indices_to_add = [
                (i, i - cur_traj_index) for i in range(min_start, max_start + 1)
            ]
            indices += indices_to_add
            cur_traj_index += traj_length
            if traj_idx < total_traj_num - self.sim_traj_num:
                sim_traj_flags += [False] * len(indices_to_add)
            else:
                sim_traj_flags += [True] * len(indices_to_add)
        return indices, sim_traj_flags


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_path):
        self.image_pairs = {}

        # Collect and group image paths by position index
        for image_name in os.listdir(images_path):
            if image_name.endswith(".png"):
                parts = image_name.replace(".png", "").split(
                    "_"
                )  # Remove extension before splitting
                if len(parts) < 3:
                    continue  # Skip malformed filenames

                try:
                    pos_index, cam_index = (
                        int(parts[1]),
                        int(parts[2]),
                    )  # Convert to integers
                except ValueError:
                    print(f"Skipping invalid file: {image_name}")
                    continue  # Skip files that don't match the expected format

                if pos_index not in self.image_pairs:
                    self.image_pairs[pos_index] = {}

                # Load image using OpenCV and convert BGR to RGB
                image = cv2.imread(os.path.join(images_path, image_name))
                if image is None:
                    print(f"Warning: Could not read {image_name}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.transpose(2, 0, 1)  # Convert to CHW format

                self.image_pairs[pos_index][cam_index] = image

        # Ensure all positions have both camera views (0 and 1)
        self.image_pairs = [
            self.image_pairs[k]
            for k in sorted(self.image_pairs.keys())
            # if 0 in self.image_pairs[k] and 1 in self.image_pairs[k]
        ]

    def __getitem__(self, idx):
        if 1 in self.image_pairs[idx]:
            return {0: self.image_pairs[idx][0][None], 1: self.image_pairs[idx][1][None]}
        else:
            return {0: self.image_pairs[idx][0][None]}

    def __len__(self):
        return len(self.image_pairs)


class AnomalyNominalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        # for dataset_path in dataset_paths:
        #     self.load_dataset(dataset_path)
        images = self.load_dataset(dataset_path)

        self.images = images
        self.images_keys = list(images.keys())

    def load_dataset(self, dataset_path):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=True)  # only np arrays
            images = dataset["images"]
            if images.dtype == np.dtype("O"):
                images = images.item()
            else:
                raise NotImplementedError("Only support dict of images for now")
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        return images

    def __getitem__(self, idx):
        if 1 in self.images_keys:
            return {
                0: self.images[self.images_keys[0]][idx][None],
                1: self.images[self.images_keys[1]][idx][None],
            }
        else:
            return {
                0: self.images[self.images_keys[0]][idx][None],
            }

    def __len__(self):
        return len(self.images[self.images_keys[0]])
