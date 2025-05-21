import argparse
import os
import re

import numpy as np

from guided_dc.utils.video_utils import stack_videos_horizontally


def read_config(file_path) -> dict:
    config_dict = {}

    with open(file_path, "r") as file:
        for line in file:
            # Remove comments and strip whitespace
            line = line.split("#")[0].strip()
            if not line:
                continue

            # Parse key-value pairs
            match = re.match(r"^(\w+):\s*(.*)$", line)
            if match:
                key, value = match.groups()

                # Convert value to appropriate type
                try:
                    if "[" in value and "]" in value:  # Handle lists
                        # # Add commas if missing between elements
                        # value = re.sub(
                        #     r"\s+", ",", value.strip("[")
                        # )  # Replace spaces with commas
                        value = eval(f"{value}")
                    elif "." in value:  # Handle floats
                        value = float(value)
                    else:  # Handle integers
                        value = int(value)
                except (ValueError, SyntaxError):
                    pass  # Keep value as string if conversion fails
                if value == "True":
                    value = True
                config_dict[key] = value

    return config_dict


def load_dataset(dataset_folder):
    dataset_path = os.path.join(dataset_folder, "dataset.npz")
    config_path = os.path.join(dataset_folder, "config.txt")
    # norm_stats_path = os.path.join(dataset_folder, "norm.npz")

    config = read_config(config_path)
    # print(config)
    camera_indices = config["camera_indices"]
    # action_keys = config["action_keys"]
    # bservation_keys = config["observation_keys"]
    # img_resolution = config["img_resolution"]
    # bgr2rgb = config["bgr2rgb"]
    # num_traj = config["num_traj"]

    # norm_stats = np.load(norm_stats_path)
    # obs_min = norm_stats["action_min"]
    # obs_max = norm_stats["action_max"]
    # action_min = norm_stats["action_min"]
    # action_max = norm_stats["action_max"]
    # delta_min = norm_stats["delta_min"]
    # delta_max = norm_stats["delta_max"]

    dataset = np.load(dataset_path, allow_pickle=True)
    images = dataset["images"].item()
    # actions = dataset["actions"]
    # states = dataset["states"]

    # # 1. Check actions and states
    # # assert np.all(actions >= action_min) and np.all(actions <= action_max)
    # # assert np.all(states >= obs_min) and np.all(states <= obs_max)
    # print("Image max:", [np.max(images[i]) for i in camera_indices])
    # print("Image min:", [np.min(images[i]) for i in camera_indices])
    # print("Image shape:", [images[i].shape for i in camera_indices])
    # print("Action max:", np.max(actions, axis=0))
    # print("Action min:", np.min(actions, axis=0))
    # print("State max:", np.max(states, axis=0))
    # print("State min:", np.min(states, axis=0))

    # fig, ax = plt.subplots(4, 2, figsize=(10, 10))
    # for i in range(8):
    #     ax[i // 2, i % 2].plot(states[:, i], label=f"State {i}")
    #     ax[i // 2, i % 2].plot(actions[:, i], label=f"Action {i}")
    #     ax[i // 2, i % 2].legend()
    # plt.savefig(os.path.join(dataset_folder, "actions_states.png"))
    # plt.plot(actions[:, -1], label="State 0")
    # plt.savefig(os.path.join(dataset_folder, "gripper.png"))

    # 2. Visualize images
    stack_videos_horizontally(
        *[images[i][::100].transpose(0, 2, 3, 1) for i in camera_indices],
        os.path.join(dataset_folder, "images.mp4"),
        bgr2rgb=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="jsg_jsg_2cam_192_default_seed0_sim67_real22",
    )
    args = parser.parse_args()
    load_dataset(args.dataset_folder)
