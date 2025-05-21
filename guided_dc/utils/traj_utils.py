import h5py
import torch
import torch.nn.functional as F

from guided_dc.utils.pose_utils import quaternion_to_euler_xyz


def smooth_trajectories(trajectories, window_size=5, batch_first=True):
    """
    Smooths the batch of end effector trajectories using a moving average filter.
    The first and last points remain unchanged.

    Args:
        trajectories (torch.Tensor): Batch of trajectories with shape (T, B, D) or (B, T, D) when batch_first=True,
                                     where 7 represents position (x, y, z) and quaternion (w, x, y, z).
        window_size (int): The size of the window for the moving average filter.

    Returns:
        torch.Tensor: Smoothed trajectories with the same shape as input.
    """
    _, _, D = trajectories.shape

    # Ensure the window size is odd for symmetry
    if window_size % 2 == 0:
        window_size += 1

    # Create the moving average filter (1D convolution kernel)
    kernel = (
        torch.ones((D, 1, window_size), device=trajectories.device) / window_size
    )  # One filter for each of the 7 dimensions

    # Apply smoothing only to the middle part (excluding the first and last points)
    if batch_first:
        raw_trajectories = trajectories.clone().permute(
            0, 2, 1
        )  # Shape: (B, T, D) -> (B, D, T)
    else:
        raw_trajectories = trajectories.clone().permute(
            1, 2, 0
        )  # Shape: (T, B, D) -> (B, D, T)

    # Perform convolution on the middle part
    smoothed_middle = F.conv1d(
        raw_trajectories,  # Shape: (B, D, T)
        kernel,
        padding=(window_size // 2),
        groups=D,  # Apply the same filter independently for each of the D dimensions
    )

    # Rebuild the trajectories by concatenating the first point, smoothed middle points, and last point
    if batch_first:
        smoothed_trajectories = smoothed_middle.permute(
            0, 2, 1
        )  # (B, D, T) -> (B, T, D)
        smoothed_trajectories[:, 0] = trajectories[:, 0]
        smoothed_trajectories[:, -1] = trajectories[:, -1]
    else:
        smoothed_trajectories = smoothed_middle.permute(
            2, 0, 1
        )  # (B, D, T) -> (T, B, D)
        smoothed_trajectories[0] = trajectories[0]
        smoothed_trajectories[-1] = trajectories[-1]

    return smoothed_trajectories


# def smooth_trajectories(trajectories, base_window_size=5, batch_first=True, smooth_factor=2):
#     """
#     Smooths the batch of end effector trajectories using a variable moving average filter.
#     The first and last parts of the trajectory are smoothed more than the middle part.

#     Args:
#         trajectories (torch.Tensor): Batch of trajectories with shape (T, B, D) or (B, T, D) when batch_first=True,
#                                      where D represents position (x, y, z) and quaternion (w, x, y, z).
#         base_window_size (int): The base size of the window for the moving average filter.
#         smooth_factor (int): The multiplier to increase the smoothing at the start and end of the trajectory.

#     Returns:
#         torch.Tensor: Smoothed trajectories with the same shape as input.
#     """
#     if base_window_size % 2 == 0:
#         base_window_size += 1  # Ensure odd window size for symmetry

#     # Get dimensions
#     B, T, D = trajectories.shape if batch_first else trajectories.permute(1, 0, 2).shape  # (B, T, D)

#     # Create copy of original trajectory
#     smoothed_trajectories = trajectories.clone()

#     # Function to apply smoothing for a specific window size
#     def apply_smoothing(data, window_size):
#         kernel = torch.ones((D, 1, window_size), device=data.device) / window_size
#         smoothed = F.conv1d(data.permute(1, 2, 0), kernel, padding=(window_size // 2), groups=D).permute(2, 0, 1)
#         return smoothed

#     # Smooth start and end of the trajectory with increased window size
#     if batch_first:
#         for i in range(T):
#             if i < T // 4:  # First quarter (smooth more)
#                 window_size = min(base_window_size * smooth_factor, T)  # Larger window for more smoothing
#             elif i > 3 * T // 4:  # Last quarter (smooth more)
#                 window_size = min(base_window_size * smooth_factor, T)
#             else:
#                 window_size = base_window_size

#             # Ensure window size is odd
#             if window_size % 2 == 0:
#                 window_size += 1

#             # Apply smoothing to this section of the trajectory
#             smoothed_trajectories[:, i:i+1, :] = apply_smoothing(trajectories[:, i:i+1, :], window_size)

#         # Keep first and last points unchanged
#         smoothed_trajectories[:, 0] = trajectories[:, 0]
#         smoothed_trajectories[:, -1] = trajectories[:, -1]

#     else:
#         for i in range(T):
#             if i < T // 4:  # First quarter (smooth more)
#                 window_size = min(base_window_size * smooth_factor, T)
#             elif i > 3 * T // 4:  # Last quarter (smooth more)
#                 window_size = min(base_window_size * smooth_factor, T)
#             else:
#                 window_size = base_window_size

#             if window_size % 2 == 0:
#                 window_size += 1

#             smoothed_trajectories[i:i+1, :, :] = apply_smoothing(trajectories[i:i+1, :, :], window_size)

#         # Keep first and last points unchanged
#         smoothed_trajectories[0] = trajectories[0]
#         smoothed_trajectories[-1] = trajectories[-1]

#     return smoothed_trajectories


def get_waypoints(trajectories, curvature_threshold=0.01, num_waypoints=10):
    """
    Automatically pick waypoints from the smoothed trajectories based on curvature
    and trajectory length, ensuring that key corner points are kept. Then interpolates
    the waypoints to have a fixed number of waypoints for all trajectories.

    Args:
        trajectories (torch.Tensor): Batch of smoothed trajectories with shape (T, B, 7),
                                     where 7 represents position and quaternion.
        curvature_threshold (float): Threshold for curvature to identify corner points.
        num_waypoints (int): Number of waypoints to interpolate for each trajectory.

    Returns:
        torch.Tensor: A tensor of interpolated waypoints with shape (B, num_waypoints, 7).
    """
    T, B, _ = trajectories.shape
    waypoints = []

    for i in range(B):
        traj = trajectories[:, i, :3]  # Extract position only (shape: T, 3)

        # Compute differences between consecutive points (for detecting curvature)
        diff1 = traj[1:] - traj[:-1]  # Velocity vector (T-1, 3)
        diff2 = diff1[1:] - diff1[:-1]  # Change in velocity (acceleration, T-2, 3)

        # Compute "curvature" as the norm of the second difference (i.e., sharp changes in direction)
        curvature = torch.norm(diff2, dim=1)  # (T-2,)

        # Select corner points where the curvature exceeds a threshold
        corner_points = torch.nonzero(curvature > curvature_threshold).squeeze()

        # Ensure corner_points is always a list, even if it's a single value
        if corner_points.numel() == 1:
            corner_points = corner_points.unsqueeze(0)

        corner_points = corner_points.tolist()  # Convert to list

        # Ensure first and last points are always included as waypoints
        keypoints = [0, *corner_points, T - 1]

        # Collect the waypoints based on the selected indices
        traj_waypoints = trajectories[keypoints, i, :]  # Shape: (N, 7)

        # Interpolate waypoints to have exactly num_waypoints points
        interp_indices = torch.linspace(
            0, traj_waypoints.size(0) - 1, num_waypoints, device=trajectories.device
        )
        interp_waypoints = torch.stack(
            [
                torch.lerp(traj_waypoints[int(floor)], traj_waypoints[int(ceil)], frac)
                for floor, ceil, frac in zip(
                    interp_indices.floor().long(),
                    interp_indices.ceil().long(),
                    interp_indices.frac(),
                    strict=False,
                )
            ]
        )

        waypoints.append(interp_waypoints)

    # Stack the interpolated waypoints into a single tensor (B, num_waypoints, 7)
    return torch.stack(waypoints, dim=1)


def compute_delta_ee_pose_euler(ee_pose):
    """
    Convert end effector pose to delta end effector pose.

    :param ee_pose: Tensor of shape (num_envs, traj_length, 7) containing (x, y, z, q_w, q_x, q_y, q_z).
    :return: Tensor of shape (num_envs, traj_length-1, 6) containing (delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw).
    """
    # Extract position and quaternion
    positions = ee_pose[:, :, :3]  # Shape (num_envs, traj_length, 3)
    quaternions = ee_pose[:, :, 3:]  # Shape (num_envs, traj_length, 4)

    # Compute delta position
    delta_position = (
        positions[:, 1:] - positions[:, :-1]
    )  # Shape (num_envs, traj_length-1, 3)

    # Convert quaternions to Euler angles
    euler_angles = quaternion_to_euler_xyz(
        quaternions
    )  # Shape (num_envs, traj_length, 3)

    # Compute delta euler angles
    delta_euler = (
        euler_angles[:, 1:] - euler_angles[:, :-1]
    )  # Shape (num_envs, traj_length-1, 3)

    # Concatenate delta position and delta euler angles
    delta_ee_pose = torch.cat(
        (delta_position, delta_euler), dim=-1
    )  # Shape (num_envs, traj_length-1, 6)

    return delta_ee_pose


def interpolate_trajectory(ee_pose_path, k):
    """
    Linearly interpolate a trajectory to add intermediate points in a batch-processed way.

    Args:
        ee_pose_path (torch.Tensor): Original trajectory of shape (batch_num, traj_len, cfg_dim).
        k (int): Multiplication factor for the length of the trajectory.

    Returns:
        torch.Tensor: Interpolated trajectory of shape (batch_num, k*traj_len, cfg_dim).
    """
    batch_num, traj_len, cfg_dim = ee_pose_path.shape
    new_traj_len = k * traj_len

    # Original time points, shaped to broadcast across all batches
    torch.linspace(0, 1, traj_len, device=ee_pose_path.device)  # Shape: (traj_len,)
    new_time = torch.linspace(
        0, 1, new_traj_len, device=ee_pose_path.device
    )  # Shape: (new_traj_len,)

    # Compute the indices of the two neighboring time points for each new point
    ratio = new_time * (traj_len - 1)  # Scale new_time to the range [0, traj_len-1]
    low_idx = ratio.floor().long()  # Lower bound index (Shape: (new_traj_len,))
    high_idx = torch.clamp(
        low_idx + 1, max=traj_len - 1
    )  # Upper bound index, clamped (Shape: (new_traj_len,))

    # Compute interpolation weights
    low_weight = 1 - (ratio - low_idx)  # Weight for the lower index
    high_weight = 1 - low_weight  # Weight for the upper index

    # Gather corresponding positions from the original trajectory
    low_values = ee_pose_path[
        :, low_idx, :
    ]  # Shape: (batch_num, new_traj_len, cfg_dim)
    high_values = ee_pose_path[
        :, high_idx, :
    ]  # Shape: (batch_num, new_traj_len, cfg_dim)

    # Perform linear interpolation using torch.lerp (low_weight * low_values + high_weight * high_values)
    interpolated_traj = torch.lerp(low_values, high_values, high_weight.unsqueeze(-1))

    return interpolated_traj


def process_paths(ee_paths, goal_poses):
    batch_size, traj_len, ee_pose_dim = ee_paths.shape

    # Calculate the L2 distance between each point in the trajectory and the goal pose
    distances = torch.norm(ee_paths - goal_poses.unsqueeze(1), dim=-1)

    # Find the index of the nearest point to the goal pose for each trajectory
    nearest_indices = torch.argmin(distances, dim=1)

    # Initialize a tensor to store the processed paths
    processed_paths = torch.zeros_like(ee_paths)

    # Process each trajectory in the batch
    for i in range(batch_size):
        nearest_idx = nearest_indices[i]
        processed_paths[i, : nearest_idx + 1] = ee_paths[i, : nearest_idx + 1]
        processed_paths[i, nearest_idx + 1 :] = ee_paths[
            i, nearest_idx : nearest_idx + 1
        ].repeat(traj_len - nearest_idx - 1, 1)

    return processed_paths, nearest_indices


def load_hdf5_to_dict(hdf5_group):
    """Recursively loads an HDF5 group back into a nested dictionary."""
    result_dict = {}
    for key, item in hdf5_group.items():
        if isinstance(item, h5py.Group):
            # If it's a group, recursively load it as a nested dict
            result_dict[key] = load_hdf5_to_dict(item)
        else:
            # Otherwise, load the dataset (which are leaf nodes)
            result_dict[key] = item[()]  # Extract the data from the dataset
    return result_dict


def load_traj_to_replay(traj_path, idx):
    # Load hdf5 file
    with h5py.File(traj_path, "r") as f:
        # Get the keys of the groups in the hdf5 file, e.g., 'traj_0', 'traj_1', ...
        keys = list(f.keys())
        traj = f[keys[idx]]
        start_state = load_hdf5_to_dict(traj["start_state"])
        traj_to_replay = traj["actions"][()]
        control_mode = traj["control_mode"][()]
    f.close()
    return start_state, traj_to_replay, control_mode


if __name__ == "__main__":
    traj_path = "videos/PickAndPlace-v1_numenvs4_datarnd1_seedNone_20241010_210747.h5"
    traj_idx = 2
    start_state, traj_to_replay, control_mode = load_traj_to_replay(traj_path, traj_idx)
    print(traj_to_replay.shape)
