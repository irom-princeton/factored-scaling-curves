import gc

import h5py
import numpy as np
import torch

from guided_dc.utils.video_utils import save_array_to_video


def to_cpu(var):
    if isinstance(var, torch.Tensor):
        return var.cpu()  # Returns True if tensor is on GPU, False otherwise
    return var  # If not a torch tensor, return False


def load_hdf5(
    file_path,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    load_image=True,
):
    """also get the raw indices of camera images"""
    keys_to_load = ["observation/timestamp/skip_action"]
    for key in action_keys:
        keys_to_load.append(f"action/{key}")
        if key == "joint_position":
            keys_to_load.append("action/joint_positions")
    for key in observation_keys:
        keys_to_load.append(f"observation/robot_state/{key}")
    if load_image:
        keys_to_load.append("observation/image")

    output = {}
    camera_indices_raw = []
    h5_file = h5py.File(file_path, "r")
    for key in keys_to_load:
        if key in h5_file:
            if "image" in key:
                for cam in h5_file[key].keys():
                    output[f"{key}/{cam}"] = h5_file[f"{key}/{cam}"][()]
                    camera_indices_raw.append(int(cam))
            else:
                output[key] = h5_file[key][()]
        else:
            print(f"Key '{key}' not found in the HDF5 file.")

    if "action/joint_positions" in output:
        output["action/joint_position"] = output["action/joint_positions"]
        del output["action/joint_positions"]

    # make sure to close h5 file
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):
            try:
                obj.close()
            except Exception:
                pass
    h5_file.close()

    return output, camera_indices_raw


def load_sim_hdf5_for_training(
    file_path,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    load_image=True,
):
    """also get the raw indices of camera images"""
    keys_to_load = []
    for key in action_keys:
        keys_to_load.append(f"action/{key}")
    for key in observation_keys:
        keys_to_load.append(f"observation/robot_state/{key}")
    if load_image:
        keys_to_load.append("observation/image")

    output = {}
    h5_file = h5py.File(file_path, "r")
    for key in keys_to_load:
        if key in h5_file:
            if "image" in key:
                for cam in h5_file[key].keys():
                    output[f"{key}/{cam}"] = h5_file[f"{key}/{cam}"][()]
            else:
                output[key] = h5_file[key][()]
        else:
            print(f"Key '{key}' not found in the HDF5 file.")

    if "pick_obj_pos" in h5_file:
        output["pick_obj_pos"] = h5_file["pick_obj_pos"][()]
        output["pick_obj_rot"] = h5_file["pick_obj_rot"][()]
        output["place_obj_pos"] = h5_file["place_obj_pos"][()]
        output["place_obj_rot"] = h5_file["place_obj_rot"][()]

    output["observation/timestamp/skip_action"] = np.zeros(
        len(output["action/gripper_position"])
    ).astype(bool)

    camera_indices_raw = [0, 1, 2]

    # make sure to close h5 file
    # for obj in gc.get_objects():
    #     if isinstance(obj, h5py.File):
    #         try:
    #             obj.close()
    #         except Exception:
    #             pass
    h5_file.close()
    return output, camera_indices_raw


def save_dict_to_hdf5(hdf5_group, data_dict):
    """Recursively saves a nested dictionary to an HDF5 group."""
    _save = False
    if isinstance(hdf5_group, str):
        _save = True
        hdf5_group = h5py.File(hdf5_group, "w")
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Create a subgroup for the nested dictionary
            subgroup = hdf5_group.create_group(key)
            save_dict_to_hdf5(subgroup, value)  # Recursively save the nested dict
        else:
            # Save the value as a dataset
            if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
                value = np.array(to_cpu(value))  # Convert lists/tuples to numpy arrays
            hdf5_group.create_dataset(key, data=value)
    if _save:
        hdf5_group.close()


def save_extra_info_to_hdf5(env, kwargs):
    traj_id = "traj_{}".format(env._episode_id)
    group = env._h5_file[traj_id]
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            value = np.array(to_cpu(value))
            group.create_dataset(key, data=value)
        elif isinstance(value, dict):
            group.create_group(key)
            save_dict_to_hdf5(group[key], to_cpu(value))
        elif isinstance(value, str):
            # Create a dataset with variable-length strings
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(key, data=value, dtype=dt)


def filter_hdf5(file_path, keys_to_exclude, rename_map, new_file_name):
    """
    Create a filtered HDF5 file by excluding specified keys and renaming specified keys.

    `rename_map` should map full source paths to desired full destination paths.
    """
    with (
        h5py.File(file_path, "r") as src_file,
        h5py.File(new_file_name, "w") as dest_file,
    ):
        found_excluded_keys = {k: False for k in keys_to_exclude}
        found_renamed_keys = {k: False for k in rename_map}

        def copy_recursive(src, src_path, dest, dest_path):
            """
            Recursively copy items from src to dest.

            src_path: current full path in the source (used for mapping lookups)
            dest_path: current full path in the destination (after renaming)
            """
            for key in src:
                # Build the full source path (for mapping & exclusion lookups)
                current_src_path = f"{src_path}/{key}" if src_path else key

                # Exclude keys if they match the criteria
                if current_src_path in keys_to_exclude:
                    found_excluded_keys[current_src_path] = True
                    print(f"Excluding key: {current_src_path}")
                    continue

                # Compute the default destination path (if no rename is applied)
                default_dest_path = f"{dest_path}/{key}" if dest_path else key
                # Apply renaming if a mapping exists
                current_dest_path = rename_map.get(current_src_path, default_dest_path)
                if current_dest_path != default_dest_path:
                    found_renamed_keys[current_src_path] = True

                if isinstance(src[key], h5py.Group):
                    # For groups, create the group in the destination using the full destination path.
                    group = dest.require_group(current_dest_path)
                    # Recursively copy the group's content.
                    copy_recursive(src[key], current_src_path, dest, current_dest_path)
                else:
                    # For datasets, ensure the destination parent group exists.
                    parent_path = "/".join(current_dest_path.split("/")[:-1])
                    parent_group = (
                        dest.require_group(parent_path) if parent_path else dest
                    )
                    new_name = current_dest_path.split("/")[-1]
                    # Use `key` here since `src` is already the current group.
                    src.copy(key, parent_group, name=new_name)

        # Start copying from the root of the source file.
        copy_recursive(src_file, "", dest_file, "")

        print("Excluded keys:", found_excluded_keys)
        print("Renamed keys:", found_renamed_keys)

    # Ensure all HDF5 files are closed
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):
            try:
                obj.close()
            except Exception:
                pass


def visualize_hdf5_images(file_path, video_filename="output_video.mp4"):
    """Visualize all observation/image/* values as a video."""
    with h5py.File(file_path, "r") as h5_file:
        image_keys = sorted(
            [key for key in h5_file["observation/image"].keys()], key=lambda x: int(x)
        )
        images = [h5_file[f"observation/image/{key}"][()] for key in image_keys]

    for idx, imgs in enumerate(images):
        save_array_to_video(str(idx) + video_filename, imgs)
    print(f"Video saved as {video_filename}")


if __name__ == "__main__":
    # save_dict_to_hdf5(h5py.File("data.h5", "w"), {"a": 1, "b": {"c": 2, "d": 3}})

    visualize_hdf5_images(
        "/n/fs/robot-data/guided-data-collection/data/tomato_plate_trials-date/17.h5"
    )
