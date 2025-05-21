"""
Script for processing raw teleop data for policy training.

Create a new folder and then save dataset and normalization values in the folder. Also save the config in txt.

"""

import os
import time
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

from guided_dc.utils.hdf5_utils import (
    filter_hdf5,
    load_hdf5,
    load_sim_hdf5_for_training,
)
from guided_dc.utils.video_utils import save_array_to_video, stack_videos_horizontally


def sort_path(traj_paths):
    """
    Sort a list of file paths by the integer filename (e.g. '25' in '…/distractor/25.h5').
    """
    return sorted(
        traj_paths,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[0]),
    )


def resize_image(args):
    img, img_resolution = args
    return cv2.resize(img, (img_resolution[1], img_resolution[0]))  # (W, H)


def resize_images_multiprocessing(raw_img, img_resolution, num_thread=10):
    args = [(raw_img[i], img_resolution) for i in range(raw_img.shape[0])]

    # Use Pool for multiprocessing
    with Pool(processes=num_thread) as pool:
        resized_images = pool.map(resize_image, args)

    resized_img = np.array(resized_images, dtype=np.uint8)
    return resized_img


def filter_raw_data(
    input_dir,
    output_dir,
    keys_to_exclude=[
        "observation/image/2",
        "observation/controller_info",
        "observation/timestamp",
    ],
    rename_map={"observation/image/3": "observation/image/0"},
    start_idx=0,
):
    traj_paths = [
        os.path.join(input_dir, traj_name)
        for traj_name in os.listdir(input_dir)
        if traj_name.endswith(".h5")
    ]
    num_trajs = len(traj_paths)
    print(f"Processing {num_trajs} trajectories from {input_dir}.")

    for traj_path in tqdm(traj_paths):
        new_file_name = os.path.join(output_dir, f"{start_idx}.h5")
        filter_hdf5(
            file_path=traj_path,
            keys_to_exclude=keys_to_exclude,
            rename_map=rename_map,
            new_file_name=new_file_name,
        )
        start_idx += 1


def sample_paths_rand(paths, num_samples, ratio=None):
    """Helper function to sample paths based on either count or ratio."""
    sampled_paths = []
    if num_samples is None and ratio is None:
        sampled_paths = [path for sublist in sampled_paths for path in sublist]
        return sampled_paths

    if len(num_samples) > 1:
        if len(num_samples) == len(paths) - 1:
            print("Automatically merging last two data paths then sample")
            paths[-2].extend(paths[-1])
            paths.pop(-1)
        sampled_paths = [
            list(np.random.choice(paths[i], num_samples[i], replace=False))
            for i in range(len(paths))
        ]
        sampled_paths = [
            path for sublist in sampled_paths for path in sublist
        ]  # Flatten
    elif len(num_samples) == 1:
        paths = [path for sublist in paths for path in sublist]  # Flatten
        sampled_paths = list(np.random.choice(paths, num_samples[0], replace=False))
    elif ratio is not None:
        if len(ratio) > 1:
            assert len(ratio) == len(paths), "Mismatched list lengths"
            sampled_paths = [
                list(
                    np.random.choice(
                        paths[i], int(len(paths[i]) * ratio[i]), replace=False
                    )
                )
                for i in range(len(paths))
            ]
            sampled_paths = [
                path for sublist in sampled_paths for path in sublist
            ]  # Flatten
        else:
            paths = [path for sublist in paths for path in sublist]  # Flatten
            sampled_paths = list(
                np.random.choice(paths, int(len(paths) * ratio[0]), replace=False)
            )
    else:
        sampled_paths = paths  # No change if neither num_samples nor ratio is provided

    return sampled_paths


def get_data_paths(input_paths):
    # concatenate all paths
    real_traj_paths = []
    sim_traj_paths = []
    num_traj_available = 0
    for path in input_paths:
        if "sim" in path:
            sim_traj_paths.append(
                [
                    os.path.join(path, traj_name)
                    for traj_name in os.listdir(path)
                    # if traj_name.endswith(".h5") and "failed" not in traj_name
                    if traj_name.endswith(".h5")
                ]
            )
            num_traj_available += len(sim_traj_paths[-1])
        else:
            real_traj_paths.append(
                [
                    os.path.join(path, traj_name)
                    for traj_name in os.listdir(path)
                    if traj_name.endswith(".h5")
                ]
            )
            num_traj_available += len(real_traj_paths[-1])
    return real_traj_paths, sim_traj_paths, num_traj_available


def sample_paths(
    traj_paths,
    num_trajs,
    ratio=None,
    sample_strat="uniform",
    delta=20,
    num_instances=5,
    num_per_instance=30,
):
    if sample_strat == "rand":
        output_paths = sample_paths_rand(traj_paths, num_trajs, ratio)

    elif sample_strat == "uniform":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sort_path(traj_path)
            num_samples = num_trajs[i]
            # if "object_pose" in sorted_paths[0] or "qpos" in sorted_paths[0]:
            #     idx_per_variation = {0: 0}
            # else:
            idx_per_variation = {num_per_instance * i: 0 for i in range(num_instances)}
            variation_keys = list(idx_per_variation.keys())
            paths = []
            j = 0
            while len(paths) < num_samples:
                variation_idx = j % len(idx_per_variation)
                sample_idx = idx_per_variation[variation_keys[variation_idx]]
                index = variation_keys[variation_idx] + sample_idx
                if index < len(sorted_paths):  # Ensure the index is valid
                    if "failed" not in sorted_paths[index]:
                        paths.append(sorted_paths[index])
                        idx_per_variation[variation_keys[variation_idx]] += 1
                    else:
                        add_path = True
                        while "failed" in sorted_paths[index]:
                            index += 1
                            if (
                                index
                                >= variation_keys[variation_idx] + num_per_instance
                            ):
                                # raise ValueError(
                                #     f"Too many failed paths for variation {variation_idx}. {sorted_paths}"
                                # )
                                add_path = False
                                break
                        if add_path:
                            paths.append(sorted_paths[index])
                            idx_per_variation[variation_keys[variation_idx]] = (
                                index - variation_keys[variation_idx] + 1
                            )
                else:
                    raise ValueError(
                        f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping."
                    )
                j += 1
            output_paths += paths

    elif sample_strat == "factor":
        factor_paths = traj_paths.pop(-1)
        num_samples = num_trajs.pop(-1)
        sorted_paths = sort_path(factor_paths)

        output_paths = sample_paths(
            traj_paths,
            num_trajs,
            ratio=None,
            sample_strat="uniform",
            delta=None,
            num_instances=num_instances,
            num_per_instance=num_per_instance,
        )

        num_instances = (num_samples + delta - 1) // delta
        samples_per_instance = [delta] * (num_instances - 1) + [
            num_samples - delta * (num_instances - 1)
        ]
        for instance, num in zip(range(num_instances), samples_per_instance):
            count = 0
            i = instance * num_per_instance
            while count < num and i < len(sorted_paths):
                if "failed" not in sorted_paths[i]:
                    output_paths.append(sorted_paths[i])
                    count += 1
                i += 1  # Move to the next path

    elif sample_strat == "ordered":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sort_path(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]
            samples_added = 0
            for path in sorted_paths:
                if "failed" not in path:
                    output_paths.append(path)
                    samples_added += 1
                if samples_added >= num_samples:
                    break

    elif sample_strat == "val":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sort_path(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]
            samples_added = 0
            for path in sorted_paths[num_instances * num_per_instance - 1 :: -1]:
                if "failed" not in path:
                    output_paths.append(path)
                    samples_added += 1
                if samples_added >= num_samples:
                    break

    else:
        raise NotImplementedError(f"Unknown sampling strategy: {sample_strat}")

    return output_paths


def process_dataset(
    input_paths,
    output_parent_dir,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    horizon_steps=16,
    img_resolution=(192, 192),
    camera_indices=[0, 2],
    num_thread=10,
    skip_image=False,
    keep_bgr=False,
    use_obs_as_action=False,
    obs_gripper_threshold=0.2,
    additional_name="",
    num_real_traj=None,
    num_sim_traj=None,
    sim_data_ratio=None,
    real_data_ratio=None,
    seed=None,
    visualize_image=False,
    sample_real_strat="uniform",
    keep_only_first=False,
    sample_sim_strat="uniform",
    delta=20,
    num_instances=5,
    num_per_instance=30,
):
    save_image = not skip_image
    bgr2rgb = not keep_bgr

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(42)

    print(additional_name)

    # Ensure only one of num_sim_traj or sim_data_ratio is specified
    assert not (num_sim_traj is not None and sim_data_ratio is not None), (
        "Only one of sim_data_ratio and num_sim_traj should be specified"
    )
    assert not (num_real_traj is not None and real_data_ratio is not None), (
        "Only one of real_data_ratio and num_real_traj should be specified"
    )

    real_traj_paths, sim_traj_paths, num_traj_available = get_data_paths(input_paths)

    sim_traj_paths = sample_paths(
        sim_traj_paths,
        num_sim_traj,
        sim_data_ratio,
        sample_sim_strat,
        delta=delta,
        num_instances=num_instances,
        num_per_instance=num_per_instance,
    )
    real_traj_paths = sample_paths(
        real_traj_paths,
        num_real_traj,
        real_data_ratio,
        sample_real_strat,
        delta=delta,
        num_instances=num_instances,
        num_per_instance=num_per_instance,
    )

    print(real_traj_paths)
    print(sim_traj_paths)

    num_real_traj = len(real_traj_paths)
    num_sim_traj = len(sim_traj_paths)

    traj_paths = list(real_traj_paths) + list(sim_traj_paths)
    num_traj = len(traj_paths)

    print(
        f"Processing {num_traj}/{num_traj_available} trajectories with {cpu_count()} cpu threads..."
    )
    # Configure dataset name based on keys
    dataset_name = ""
    if "cartesian_position" in observation_keys:
        dataset_name += "eef"
    if "joint_positions" in observation_keys:
        assert "cartesian_position" not in observation_keys
        dataset_name += "js"
    if "joint_velocities" in observation_keys:
        assert "cartesian_position" not in observation_keys
        dataset_name += "jv"
    if "gripper_position" in observation_keys:
        dataset_name += "g"
    dataset_name += "_"
    if "cartesian_position" in action_keys:
        dataset_name += "eef"
    if "joint_position" in action_keys:
        assert "cartesian_position" not in action_keys
        dataset_name += "js"
    if "joint_velocities" in action_keys:
        assert "cartesian_position" not in action_keys
        dataset_name += "jv"
    if "gripper_position" in action_keys:
        dataset_name += "g"
    if save_image:
        dataset_name += "_"
        dataset_name += f"{len(camera_indices)}cam"
        dataset_name += f"_{img_resolution[0]}"
    if additional_name:
        dataset_name += f"_{additional_name}"
    if seed is not None:
        dataset_name += f"_seed{seed}"
    dataset_name += f"_sim{num_sim_traj}_real{num_real_traj}"

    dataset_path = os.path.join(f"{output_parent_dir}", f"{dataset_name}/dataset.npz")
    print(dataset_path)

    if os.path.exists(dataset_path):
        try:
            data = np.load(dataset_path, allow_pickle=True)
            _ = data["images"].item()
            print("Dataset already exists, skipping processing.")
            return
        except Exception:
            print(
                "Dataset file exists but could not be loaded. Proceeding with processing."
            )

    # initialize output dictionary
    output = {
        "traj_lengths": [],
        "actions": [],
        "states": [],
        "images": {index: [] for index in camera_indices},
    }
    state_action_diff_mins = []
    state_action_diff_maxs = []
    for traj_path in tqdm(traj_paths):
        # load trajectory from h5
        s1 = time.time()
        try:
            if "sim" in traj_path:
                print("Loading sim data")
                traj, camera_indices_raw = load_sim_hdf5_for_training(
                    traj_path,
                    action_keys=action_keys,
                    observation_keys=observation_keys,
                    load_image=save_image,
                )
                traj["action/gripper_position"] = (
                    1 - (traj["action/gripper_position"] + 1) / 2
                )
                traj["observation/robot_state/gripper_position"] = 1 - (
                    traj["observation/robot_state/gripper_position"] / 0.04
                )
            else:
                traj, camera_indices_raw = load_hdf5(
                    traj_path,
                    action_keys=action_keys,
                    observation_keys=observation_keys,
                    load_image=save_image,
                )
            print("Time to load h5:", time.time() - s1)
        except Exception:
            print("Failed to load", traj_path)
            continue

        # skip idle actions (skip_action == True)
        keep_idx = ~traj["observation/timestamp/skip_action"]
        if keep_only_first:
            # Only keep the first 1 step
            keep_idx = np.zeros_like(keep_idx)
            keep_idx[0] = 1
        traj_length = keep_idx.sum()
        output["traj_lengths"].append(traj_length)
        print("Path:", traj_path, "Length:", traj_length)

        # set gripper position to binary: 0 for open, 1 for closed
        if "gripper_position" in action_keys:
            traj["action/gripper_position"] = (
                traj["action/gripper_position"] > 0.5
            ).astype(np.float32)
        if "gripper_position" in observation_keys:
            traj["observation/robot_state/gripper_position"] = (
                traj["observation/robot_state/gripper_position"] > obs_gripper_threshold
            ).astype(np.float32)

        # set roll to always be positive
        if "cartesian_position" in action_keys:
            traj["action/cartesian_position"][:, 3] = np.abs(
                traj["action/cartesian_position"][:, 3]
            )
        if "cartesian_position" in observation_keys:
            traj["observation/robot_state/cartesian_position"][:, 3] = np.abs(
                traj["observation/robot_state/cartesian_position"][:, 3]
            )
        if "joint_positions" in observation_keys:
            if len(traj["observation/robot_state/joint_positions"].shape) == 3:
                traj["observation/robot_state/joint_positions"] = traj[
                    "observation/robot_state/joint_positions"
                ].squeeze()
            assert len(traj["observation/robot_state/joint_positions"].shape) == 2

        # add dimension to gripper position
        if "gripper_position" in action_keys:
            traj["action/gripper_position"] = traj["action/gripper_position"].squeeze()[
                :, None
            ]
            traj["action/gripper_position"] = traj["action/gripper_position"].squeeze()[
                :, None
            ]
        if "gripper_position" in observation_keys:
            traj["observation/robot_state/gripper_position"] = traj[
                "observation/robot_state/gripper_position"
            ].squeeze()[:, None]
            traj["observation/robot_state/gripper_position"] = traj[
                "observation/robot_state/gripper_position"
            ].squeeze()[:, None]

        if "forces" in observation_keys:
            traj["observation/robot_state/forces"] = traj[
                "observation/robot_state/forces"
            ][:, None]
            assert len(traj["observation/robot_state/forces"].shape) == 2
        print(traj["observation/robot_state/gripper_position"].shape)
        print(traj["action/gripper_position"].shape)
        print(traj["observation/robot_state/joint_positions"].shape)
        print(traj["action/joint_position"].shape)
        # get the maximum difference between the starting state and each action of the chunk at each timestep
        if "joint_positions" in observation_keys and "joint_position" in action_keys:
            state = traj["observation/robot_state/joint_positions"][keep_idx]
            action = traj["action/joint_position"][keep_idx]
        elif (
            "cartesian_position" in observation_keys
            and "cartesian_position" in action_keys
        ):
            state = traj["observation/robot_state/cartesian_position"][keep_idx]
            action = traj["action/cartesian_position"][keep_idx]
        else:
            raise NotImplementedError(
                "For getting the state-action difference, need consistent keys"
            )
        if not keep_only_first:
            diffs = np.empty((0, action.shape[1]))
            for step in range(horizon_steps // 4, horizon_steps):  # skip early steps
                diff = action[step:] - state[:-step]
                diffs = np.concatenate([diffs, diff], axis=0)
            state_action_diff_mins.append(np.min(diffs, axis=0))
            state_action_diff_maxs.append(np.max(diffs, axis=0))
            if np.isnan(np.sum(diffs)) > 0 or np.isnan(np.sum(diffs)) > 0:
                raise ValueError("NaN in state-action difference")

        # concatenate states and actions
        if use_obs_as_action:
            states = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx][
                        :-1
                    ]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            actions = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx][1:]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            output["traj_lengths"][-1] = output["traj_lengths"][-1] - 1
        else:
            states = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            actions = np.concatenate(
                [
                    traj[f"action/{action_keys[i]}"][keep_idx]
                    for i in range(len(action_keys))
                ],
                axis=1,
            )
        assert len(states) == output["traj_lengths"][-1]
        assert len(actions) == output["traj_lengths"][-1]
        output["states"].append(states)
        output["actions"].append(actions)

        # add images
        if save_image:
            # verify camera indices
            if len(camera_indices_raw) == 4:
                if len(camera_indices) == 2:
                    camera_indices_chosen = [camera_indices_raw[idx] for idx in [1, 3]]
                else:
                    camera_indices_chosen = [camera_indices_raw[idx] for idx in [1]]
            elif len(camera_indices_raw) == 2:
                camera_indices_chosen = [0, 1]
            else:
                camera_indices_chosen = [
                    camera_indices_raw[idx] for idx in camera_indices
                ]
            print(f"Using raw camera indices: {camera_indices_chosen}")
            for raw_idx, idx in zip(camera_indices_chosen, camera_indices):
                raw_img = traj[f"observation/image/{raw_idx}"][keep_idx]  # (T, H, W, C)
                assert raw_img.dtype == np.uint8

                # resize with multiprocessing
                s1 = time.time()
                resized_img = resize_images_multiprocessing(
                    raw_img,
                    img_resolution,
                    num_thread,
                )
                print("Time to resize images:", time.time() - s1)

                # Transpose to (T, C, H, W)
                resized_img = resized_img.transpose(0, 3, 1, 2)

                # Change BGR (cv2 default) to RGB
                # if bgr2rgb:
                #     if "sim" not in traj_path:
                #         resized_img = resized_img[:, [2, 1, 0]]

                if use_obs_as_action:
                    resized_img = resized_img[:-1]

                # save
                assert len(resized_img) == output["traj_lengths"][-1]
                output["images"][idx].append(resized_img)

    # Convert to numpy arrays
    output["traj_lengths"] = np.array(output["traj_lengths"])
    output["actions"] = np.concatenate(output["actions"], axis=0)
    output["states"] = np.concatenate(output["states"], axis=0)

    for idx in camera_indices:
        output["images"][idx] = np.concatenate(output["images"][idx], axis=0)
    print("\n\n=========\nImages shape: ", output["images"][camera_indices[0]].shape)

    # Normalize states and actions to [-1, 1]
    obs_min = np.min(output["states"], axis=0)
    obs_max = np.max(output["states"], axis=0)
    action_min = np.min(output["actions"], axis=0)
    action_max = np.max(output["actions"], axis=0)
    output["raw_states"] = output["states"].copy()
    output["raw_actions"] = output["actions"].copy()
    output["states"] = 2 * (output["states"] - obs_min) / (obs_max - obs_min + 1e-6) - 1
    output["actions"] = (
        2 * (output["actions"] - action_min) / (action_max - action_min + 1e-6) - 1
    )
    print("States min (after normalization):", np.min(output["states"], axis=0))
    print("States max (after normalization):", np.max(output["states"], axis=0))
    print("Actions min (after normalization):", np.min(output["actions"], axis=0))
    print("Actions max (after normalization):", np.max(output["actions"], axis=0))

    if not keep_only_first:
        # Get min and max of state-action difference
        state_action_diff_min = np.min(np.stack(state_action_diff_mins), axis=0)
        state_action_diff_max = np.max(np.stack(state_action_diff_maxs), axis=0)

    # Create output directory
    output_dir = os.path.join(output_parent_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    if visualize_image:
        if len(camera_indices) == 2:
            stack_videos_horizontally(
                *[
                    output["images"][i][::100].transpose(0, 2, 3, 1)
                    for i in camera_indices
                ],
                os.path.join(output_dir, "images.mp4"),
                bgr2rgb=False,
            )
        else:
            assert len(camera_indices) == 1
            save_array_to_video(
                os.path.join(output_dir, "images.mp4"),
                output["images"][camera_indices[0]][::100].transpose(0, 2, 3, 1),
                bgr2rgb=False,
            )
    if not keep_only_first:
        # Save config into a text file
        config = {
            "action_keys": action_keys,
            "observation_keys": observation_keys,
            "img_resolution": img_resolution,
            "camera_indices": camera_indices,
            "bgr2rgb": bgr2rgb,
            "obs_min": obs_min,
            "obs_max": obs_max,
            "action_min": action_min,
            "action_max": action_max,
            "delta_min": state_action_diff_min,
            "delta_max": state_action_diff_max,
            "num_traj": len(traj_paths),
        }
        with open(os.path.join(output_dir, "config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        # Save the normalization values and processed dataset
        np.savez(
            os.path.join(output_dir, "norm.npz"),
            obs_min=obs_min,
            obs_max=obs_max,
            action_min=action_min,
            action_max=action_max,
            delta_min=state_action_diff_min,
            delta_max=state_action_diff_max,
        )
        np.savez_compressed(
            os.path.join(output_dir, "dataset.npz"),
            **output,
        )
    else:
        # Save config into a text file
        config = {
            "action_keys": action_keys,
            "observation_keys": observation_keys,
            "img_resolution": img_resolution,
            "camera_indices": camera_indices,
            "bgr2rgb": bgr2rgb,
            "obs_min": obs_min,
            "obs_max": obs_max,
            "action_min": action_min,
            "action_max": action_max,
            "num_traj": len(traj_paths),
        }
        with open(os.path.join(output_dir, "config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        # Save the normalization values and processed dataset
        np.savez(
            os.path.join(output_dir, "norm.npz"),
            obs_min=obs_min,
            obs_max=obs_max,
            action_min=action_min,
            action_max=action_max,
        )
        np.savez_compressed(
            os.path.join(output_dir, "dataset.npz"),
            **output,
        )
    print("Data and normalization values saved in", output_dir)

    if sample_real_strat == "val" or sample_sim_strat == "val":
        image_keys = output["images"].keys()
        anomaly_images = {key: [] for key in image_keys}
        first_frame_idx = 0
        for traj_idx in range(num_traj):
            for key in image_keys:
                anomaly_images[key].append(output["images"][key][first_frame_idx])
            first_frame_idx += output["traj_lengths"][traj_idx]
        np.savez(
            dataset_path.replace("dataset.npz", "anomaly_dataset.npz"),
            images=anomaly_images,
        )


def get_sampled_paths(
    input_paths,
    num_real_traj=None,
    num_sim_traj=None,
    sim_data_ratio=None,
    real_data_ratio=None,
    seed=None,
    sample_real_strat="uniform",
    sample_sim_strat="uniform",
    delta=20,
    num_instances=5,
    num_per_instance=30,
):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(42)

    # Ensure only one of num_sim_traj or sim_data_ratio is specified
    assert not (num_sim_traj is not None and sim_data_ratio is not None), (
        "Only one of sim_data_ratio and num_sim_traj should be specified"
    )
    assert not (num_real_traj is not None and real_data_ratio is not None), (
        "Only one of real_data_ratio and num_real_traj should be specified"
    )

    real_traj_paths, sim_traj_paths, num_traj_available = get_data_paths(input_paths)

    sim_traj_paths = sample_paths(
        sim_traj_paths,
        num_sim_traj,
        sim_data_ratio,
        sample_sim_strat,
        delta=delta,
        num_instances=num_instances,
        num_per_instance=num_per_instance,
    )
    real_traj_paths = sample_paths(
        real_traj_paths, num_real_traj, real_data_ratio, sample_real_strat, delta=delta
    )

    print(real_traj_paths)
    print(sim_traj_paths)

    return sim_traj_paths + real_traj_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-in",
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--output_parent_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--num_thread",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-c",
        "--camera_indices",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Raw data uses index from cv2, which might be something like [2, 4, 8]. We will use the index from the raw data, in this case, 0, 1, 2",
    )
    parser.add_argument(
        "-res",
        "--img_resolution",
        type=int,
        nargs=2,
        default=[192, 192],
    )
    parser.add_argument(
        "-a",
        "--action_keys",
        type=str,
        nargs="+",
        default=[
            "joint_position",
            "gripper_position",
        ],  # "cartesian_position"
    )
    parser.add_argument(
        "-o",
        "--observation_keys",
        type=str,
        nargs="+",
        default=[
            "joint_positions",
            "gripper_position",
        ],  # "joint_velocities", "cartesian_positions"
    )
    parser.add_argument(
        "-tp",
        "--horizon_steps",
        type=int,
        default=16,  # we are not saving action chunks, but to get the maximum difference between the state and each step of action, for delta actions
    )
    parser.add_argument(
        "--skip_image",
        action="store_true",
    )
    parser.add_argument(
        "--keep_bgr",
        action="store_true",
    )
    parser.add_argument(
        "--use_obs_as_action",
        action="store_true",
    )
    parser.add_argument(
        "--obs_gripper_threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--additional_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--sim_data_ratio",
        default=None,
    )
    parser.add_argument(
        "--real_data_ratio",
        default=None,
    )
    parser.add_argument(
        "--num_sim_traj",
        type=int,
        nargs="+",  # Accepts one or more integers
        default=None,
    )
    parser.add_argument(
        "--num_real_traj",
        type=int,
        nargs="+",  # Accepts one or more integers
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--visualize_image",
        action="store_true",
    )
    parser.add_argument(
        "--sample_real_strat",
        type=str,
        default="uniform",
    )
    parser.add_argument(
        "--keep_only_first",
        action="store_true",
    )
    parser.add_argument(
        "--sample_sim_strat",
        type=str,
        default="uniform",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num_per_instance",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    process_dataset(**vars(args))
