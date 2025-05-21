from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation as R
from transforms3d import affines, quaternions


def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions, supports both list/numpy array and torch.Tensor input.

    Parameters:
    q1 : list, numpy array, or torch.Tensor
        First quaternion [w1, x1, y1, z1]
    q2 : list, numpy array, or torch.Tensor
        Second quaternion [w2, x2, y2, z2]

    Returns:
    result : list or torch.Tensor
        Resultant quaternion after multiplication [w, x, y, z]
        Output is on GPU if input is torch.Tensor on GPU.
    """

    if isinstance(q1, torch.Tensor):
        # Ensure the inputs are torch tensors
        if q1.device != q2.device:
            raise ValueError("q1 and q2 must be on the same device (CPU or GPU).")

        # Extract components of the quaternions
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        # Perform quaternion multiplication
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return torch.stack([w, x, y, z], dim=-1)

    elif isinstance(q1, (list, np.ndarray, tuple)):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return np.array([w, x, y, z])

    else:
        raise NotImplementedError


def batch_quaternion_multiply(q1, q2):
    # Determine whether q1 or q2 is batched. If both are batched, the batch size must match. If only one is batched, the other is broadcasted.
    pkg = torch if isinstance(q1, torch.Tensor) else np
    if len(q1.shape) == 2 and len(q2.shape) == 2:
        assert q1.shape[0] == q2.shape[0], "Batch size mismatch between q1 and q2."
        return pkg.stack(
            [quaternion_multiply(q1[i], q2[i]) for i in range(q1.shape[0])]
        )
    elif len(q1.shape) == 2:
        return pkg.stack([quaternion_multiply(q1[i], q2) for i in range(q1.shape[0])])
    elif len(q2.shape) == 2:
        return pkg.stack([quaternion_multiply(q1, q2[i]) for i in range(q2.shape[0])])
    else:
        return quaternion_multiply(q1, q2)


def euler_to_quaternion(
    euler_angles: Union[torch.Tensor, np.ndarray, list], order="XYZ"
):
    """
    Convert Euler angles to quaternions.

    Parameters:
    euler_angles : torch.Tensor, np.ndarray, or list
        Euler angles in radians, shape (3,) or (N, 3).
    order : str
        Order of Euler angles, e.g. 'XYZ', 'ZXY', etc.

    Returns:
    torch.Tensor or np.ndarray
        Quaternions in (w, x, y, z) format, shape (4,) or (N, 4).
    """
    # Check the type of input and convert to torch.tensor if it's not already
    if isinstance(euler_angles, np.ndarray):
        euler_angles = torch.from_numpy(euler_angles)
    elif isinstance(euler_angles, list):
        euler_angles = torch.tensor(euler_angles)

    # Ensure the tensor is float type
    euler_angles = euler_angles.float()

    # Handle cases where the input is not on the CPU (for torch tensors)
    device = euler_angles.device

    # Convert Euler angles to quaternions
    if len(euler_angles.shape) == 1:
        euler_angles = euler_angles[None]

    quaternions = R.from_euler(order, euler_angles.cpu()).as_quat()

    # from xyzw to wxyz
    quaternions = reshape_quaternion_xyzw_to_wxyz(quaternions)

    if isinstance(euler_angles, torch.Tensor):
        return torch.from_numpy(quaternions).squeeze().to(device)
    elif isinstance(euler_angles, np.ndarray):
        return quaternions.squeeze()
    elif isinstance(euler_angles, list):
        return quaternions.squeeze().tolist()


def axis_angle_to_quaternion(axis_angle: Union[torch.Tensor, np.ndarray]):
    """
    Input: [Batch_size, 4], where [:, :3] is the axis and [:, 3] is the angle
    Output: wxyz format quaternion
    """
    # Check the type of input and convert to torch.tensor if it's not already
    if isinstance(axis_angle, np.ndarray):
        axis_angle = torch.from_numpy(axis_angle)

    # Ensure the tensor is float type
    axis_angle = axis_angle.float()

    # Handle cases where the input is not on the CPU (for torch tensors)
    device = axis_angle.device if isinstance(axis_angle, torch.Tensor) else "cpu"

    # Split the axis vector (first three elements) and angle (last element)
    if len(axis_angle.shape) == 1:
        axis_angle = axis_angle[None]

    axis = axis_angle[:, :3]
    angle = axis_angle[:, 3]

    # Normalize the axis vector
    axis_norm = torch.norm(axis, dim=1, keepdim=True)
    normalized_axis = axis / axis_norm.clamp(min=1e-8)  # To avoid division by zero

    # Compute quaternion components
    half_angle = angle / 2
    cos_half_angle = torch.cos(half_angle)
    sin_half_angle = torch.sin(half_angle)

    quaternion = torch.zeros((axis_angle.shape[0], 4), device=device)
    quaternion[:, 0] = cos_half_angle  # w
    quaternion[:, 1:] = normalized_axis * sin_half_angle.unsqueeze(-1)  # (x, y, z)

    if isinstance(axis_angle, np.ndarray):
        return quaternion.squeeze().cpu().numpy()
    else:
        return quaternion.squeeze()


def compute_delta_rot_batch(q_current: torch.Tensor, q_target: torch.Tensor):
    """
    Compute delta rotation (Euler angles) between batch of current and target quaternions.

    Args:
        q_current (torch.Tensor): Batch of current quaternions, shape (N, 4) in (w, x, y, z) format.
        q_target (torch.Tensor): Batch of target quaternions, shape (N, 4) in (w, x, y, z) format.

    Returns:
        torch.Tensor: Batch of delta rotations in Euler angles (XYZ order), shape (N, 3).
    """
    # Normalize quaternions to avoid issues with rounding errors
    q_current = F.normalize(q_current, dim=-1)
    q_target = F.normalize(q_target, dim=-1)

    # Invert the current quaternion: q_current_inv = (w, -x, -y, -z)
    q_current_inv = torch.cat([q_current[:, :1], -q_current[:, 1:]], dim=-1)

    # Compute delta quaternion: q_delta = q_target * q_current_inv
    w1, x1, y1, z1 = q_target.unbind(-1)
    w2, x2, y2, z2 = q_current_inv.unbind(-1)

    q_delta = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

    # Convert delta quaternion to Euler angles (XYZ order)
    delta_rot = quaternion_to_euler_xyz(q_delta)
    return delta_rot


def quaternion_to_euler_xyz(q: Union[torch.Tensor, np.ndarray]):
    """
    Convert a batch of quaternions to Euler angles (XYZ order).

    Args:
        q (torch.Tensor): Batch of quaternions, shape (N, 4).

    Returns:
        torch.Tensor: Batch of Euler angles (XYZ order), shape (N, 3).
    """
    # Ensure quaternion is normalized
    if isinstance(q, list):
        q = np.array(q)
    if len(q.shape) == 1:
        q = q[None]
    if isinstance(q, torch.Tensor):
        q = F.normalize(q, dim=-1)

        w, x, y, z = q.unbind(-1)

        # Euler angles formula from quaternion
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch_y = torch.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)

        return torch.stack([roll_x, pitch_y, yaw_z], dim=-1)
    else:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Euler angles formula from quaternion
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.stack([roll_x, pitch_y, yaw_z], axis=-1)


def transform_matrices_to_quaternions(matrices: torch.Tensor):
    """
    Convert a batch of transformation matrices to quaternions using SciPy.

    Args:
        matrices (torch.Tensor): A tensor of shape (N, 4, 4) where N is the number of matrices.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) containing quaternions (w, x, y, z).
    """
    assert matrices.shape[1] == 4 and matrices.shape[2] == 4, (
        f"Input must be of shape (N, 4, 4), but the given shape is {matrices.shape}"
    )

    # Move matrices to CPU and convert to NumPy for SciPy
    matrices_np = matrices.cpu().numpy()

    # Extract the rotation part of the transformation matrices (N, 3, 3)
    rotation_matrices = matrices_np[:, :3, :3]

    # Convert to quaternions using SciPy
    quaternions = np.array([R.from_matrix(rot).as_quat() for rot in rotation_matrices])

    # Convert the result back to a PyTorch tensor and ensure correct device and dtype
    quaternions_tensor = torch.tensor(
        quaternions, dtype=matrices.dtype, device=matrices.device
    )

    # Reshape from (x, y, z, w) to (w, x, y, z)
    quaternions_tensor = reshape_quaternion_xyzw_to_wxyz(quaternions_tensor)

    return quaternions_tensor


def batch_transform_to_pos_quat(transformation_matrices):
    """
    Convert batch of transformation matrices to positions and quaternions.

    Args:
        transformation_matrices (torch.Tensor): Batch of transformation matrices of shape (B, 4, 4).

    Returns:
        torch.Tensor: Batch of positions of shape (B, 3).
        torch.Tensor: Batch of quaternions of shape (B, 4) in (x, y, z, w) format.
    """

    # Extract positions from the top-right part of the transformation matrix
    positions = transformation_matrices[:, :3, 3]  # (B, 3)

    # Extract rotation matrices from the top-left 3x3 part of the transformation matrix
    rotation_matrices = transformation_matrices[:, :3, :3].cpu().numpy()  # (B, 3, 3)

    # Convert rotation matrices to quaternions using scipy
    quaternions = R.from_matrix(rotation_matrices).as_quat()  # (B, 4)

    # Convert quaternions back to torch.Tensor
    quaternions = torch.tensor(
        quaternions, dtype=torch.float32, device=transformation_matrices.device
    )  # (B, 4)

    return torch.cat((positions, quaternions), dim=-1)


def reshape_quaternion_xyzw_to_wxyz(q_batch: torch.Tensor):
    """
    Reshape batch of quaternions from (x, y, z, w) to (w, x, y, z) format.

    Args:
        q_batch (torch.Tensor): Batch of quaternions, shape (N, 4) in (x, y, z, w) format.

    Returns:
        torch.Tensor: Batch of quaternions in (w, x, y, z) format.
    """
    # Extract each component from the input tensor
    x = q_batch[:, 0:1]  # Shape (N, 1)
    y = q_batch[:, 1:2]  # Shape (N, 1)
    z = q_batch[:, 2:3]  # Shape (N, 1)
    w = q_batch[:, 3:4]  # Shape (N, 1)

    if isinstance(q_batch, np.ndarray):
        return np.concatenate((w, x, y, z), axis=-1)
    else:
        # Concatenate in the order (w, x, y, z)
        q_wxyz = torch.cat((w, x, y, z), dim=-1)  # Shape (N, 4)
        return q_wxyz


def get_longest_axis_direction(
    mesh: trimesh.Trimesh, return_tensor=False, device="cuda:0"
):
    bounding_box = mesh.bounding_box_oriented
    extents = bounding_box.primitive.extents
    longest_axis_index = np.argmax(extents)
    rotation_matrix = bounding_box.primitive.transform[:3, :3]  # 3x3 rotation matrix
    longest_axis_direction = rotation_matrix[:, longest_axis_index]
    longest_axis_direction = longest_axis_direction.copy()
    if len(longest_axis_direction.shape) == 1:
        longest_axis_direction[2] = 0
        longest_axis_direction = longest_axis_direction / np.linalg.norm(
            longest_axis_direction
        )
    elif len(longest_axis_direction.shape) == 2:
        longest_axis_direction[:, 2] = 0
        longest_axis_direction = (
            longest_axis_direction
            / np.linalg.norm(longest_axis_direction, axis=1)[:, None]
        )
    else:
        raise NotImplementedError
    if return_tensor:
        return torch.tensor(longest_axis_direction, device=device)
    return longest_axis_direction


def get_normal_axis_direction(
    mesh: trimesh.Trimesh, return_tensor=False, device="cuda:0"
):
    longest_axis_direction = get_longest_axis_direction(mesh, return_tensor, device)
    if len(longest_axis_direction.shape) == 2:
        if return_tensor:
            normal_axis_direction = torch.column_stack(
                (-longest_axis_direction[:, 1], longest_axis_direction[:, 0], 0)
            )
        else:
            normal_axis_direction = np.column_stack(
                (-longest_axis_direction[:, 1], longest_axis_direction[:, 0], 0)
            )
        # mask = normal_axis_direction[:, 0] > 0
        # normal_axis_direction[mask] *= -1
    elif len(longest_axis_direction.shape) == 1:
        if return_tensor:
            normal_axis_direction = torch.tensor(
                (-longest_axis_direction[1], longest_axis_direction[0], 0)
            )
        else:
            normal_axis_direction = np.array(
                (-longest_axis_direction[1], longest_axis_direction[0], 0)
            )
        # if normal_axis_direction[0] > 0:
        #     normal_axis_direction *= -1
    else:
        raise NotImplementedError
    if return_tensor:
        return torch.tensor(normal_axis_direction, device=device)
    return normal_axis_direction


def rotate_vectors(vectors, q_delta_zrot):
    # If q_delta_zrot is a scalar, expand it to a vector of the same shape as vectors
    if q_delta_zrot.dim() == 0:
        q_delta_zrot = q_delta_zrot.expand(vectors.shape[0])

    # Calculate cosine and sine of the angles
    cos_angles = torch.cos(q_delta_zrot)
    sin_angles = torch.sin(q_delta_zrot)

    # Create the rotation matrix
    rotation_matrix = torch.stack(
        (
            cos_angles,
            -sin_angles,
            torch.zeros_like(cos_angles),
            sin_angles,
            cos_angles,
            torch.zeros_like(sin_angles),
            torch.zeros_like(cos_angles),
            torch.zeros_like(sin_angles),
            torch.ones_like(cos_angles),
        ),
        dim=1,
    ).reshape(-1, 3, 3)

    # Rotate the vectors using batch matrix multiplication
    rotated_vectors = torch.bmm(rotation_matrix, vectors.unsqueeze(-1)).squeeze(-1)

    return rotated_vectors


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        quaternion (torch.Tensor): Quaternion (w, x, y, z) of shape (4,).

    Returns:
        torch.Tensor: Corresponding 3x3 rotation matrix.
    """
    w, x, y, z = quaternion
    R = torch.tensor(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )
    return R


def normal_to_forward_quat(normal: np.ndarray) -> np.ndarray:
    if len(normal.shape) == 1:
        pos = np.array([0, 0, 0])
        forward = pos + normal
        u = np.array([0, 0, 1])
        if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
            u = np.array([0, 1, 0])
        s = np.cross(forward, u)
        s = s / np.linalg.norm(s)
        u = np.cross(s, forward)
        view_matrix = np.array(
            [
                s[0],
                u[0],
                -forward[0],
                0,
                s[1],
                u[1],
                -forward[1],
                0,
                s[2],
                u[2],
                -forward[2],
                0,
                -np.dot(s, pos),
                -np.dot(u, pos),
                np.dot(forward, pos),
                1,
            ]
        )
        view_matrix = view_matrix.reshape(4, 4).T
        pose_matrix = np.linalg.inv(view_matrix)
        return quaternions.mat2quat(affines.decompose(pose_matrix)[1])

    else:  # iterate among the batch
        quats = []
        for n in normal:
            pos = np.array([0, 0, 0])
            forward = pos + n
            u = np.array([0, 0, 1])
            if np.allclose(n, [0, 0, 1]) or np.allclose(n, [0, 0, -1]):
                u = np.array([0, 1, 0])
            s = np.cross(forward, u)
            s = s / np.linalg.norm(s)
            u = np.cross(s, forward)
            view_matrix = np.array(
                [
                    s[0],
                    u[0],
                    -forward[0],
                    0,
                    s[1],
                    u[1],
                    -forward[1],
                    0,
                    s[2],
                    u[2],
                    -forward[2],
                    0,
                    -np.dot(s, pos),
                    -np.dot(u, pos),
                    np.dot(forward, pos),
                    1,
                ]
            )
            view_matrix = view_matrix.reshape(4, 4).T
            pose_matrix = np.linalg.inv(view_matrix)
            quats.append(quaternions.mat2quat(affines.decompose(pose_matrix)[1]))
        return np.array(quats)


def add_euler(e1, e2, order="XYZ", return_quat=True):
    """
    Add two batched Euler angles in radians.

    Args:
        e1 (torch.Tensor): First Euler angles in radians, shape (N, 3).
        e2 (torch.Tensor): Second Euler angles in radians, shape (N, 3).
        order (str): Order of Euler angles, e.g. 'XYZ', 'ZXY', etc.

    Returns:
        torch.Tensor: Sum of Euler angles in radians, shape (N, 3).
    """
    # Convert Euler angles to quaternions
    q1 = euler_to_quaternion(e1, order)
    q2 = euler_to_quaternion(e2, order)

    # Multiply quaternions
    q_sum = batch_quaternion_multiply(q1, q2)

    if return_quat:
        return q_sum

    # Convert back to Euler angles
    e_sum = quaternion_to_euler_xyz(q_sum)

    return e_sum


if __name__ == "__main__":
    # pose1 = torch.tensor([0.7073882691671998, 0.706825181105366, 0, 0])[None,:]
    # pose2 = torch.tensor([0.0343991, 0.240845, -0.0556856, 0.968354])[None,:]
    # print(compute_delta_rot_batch(pose1, pose2))
    # pose1 = torch.tensor([0.0416438, 0.00134037, -0.688737, 0.723813])[None, :]
    # print(quaternion_to_euler_xyz(pose1))
    from transforms3d.euler import euler2quat

    euler_1 = add_euler(
        torch.tensor([-1.57265, -0.0464526, 3.0965]),
        torch.tensor([0, 0, 0]),
        return_quat=True,
    )

    euler_2 = euler2quat(-1.57265, -0.0464526, 3.0965)
    breakpoint()
