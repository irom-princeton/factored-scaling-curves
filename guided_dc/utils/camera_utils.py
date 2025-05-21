import numpy as np
from scipy.spatial.transform import Rotation as R


def get_sim_camera_global_pose_from_real(
    robot_base_global_pose, camera_robot_base_pose
):
    # Unpack inputs
    pos_b, quat_b = (
        np.array(robot_base_global_pose[:3]),
        np.array(robot_base_global_pose[3:]),
    )
    pos_c, quat_c = (
        np.array(camera_robot_base_pose[:3]),
        np.array(camera_robot_base_pose[3:]),
    )

    # Normalize quaternions
    quat_b /= np.linalg.norm(quat_b)
    quat_c /= np.linalg.norm(quat_c)

    quat_b = np.roll(quat_b, -1)  # from w, x, y, z to x, y, z, w
    quat_c = np.roll(quat_c, -1)

    # Convert quaternion to rotation matrix for base
    R_b = R.from_quat(quat_b).as_matrix()  # Quaternion format: [x, y, z, w]

    # Debugging: Print rotation matrix
    print("Rotation matrix R_b:\n", R_b)

    # Transform position
    pos_g = pos_b + R_b @ pos_c

    # Debugging: Print intermediate positions
    print("Base global position (pos_b):", pos_b)
    print("Camera relative position (pos_c):", pos_c)
    print("Transformed global position (pos_g):", pos_g)

    # Transform orientation (quaternion multiplication)
    quat_b_rot = R.from_quat(quat_b)
    quat_c_rot = R.from_quat(quat_c)
    quat_g_rot_oepncv = quat_b_rot * quat_c_rot  # Quaternion multiplication

    # from x, y, z, w to w, x, y, z
    # quat_g_opencv = np.roll(quat_g, 1)

    opencv2ros_rot = R.from_euler("xyz", [90, -90, 0], degrees=True)

    quat_g_rot_ros = quat_g_rot_oepncv * opencv2ros_rot

    quat_g = quat_g_rot_ros.as_quat()
    quat_g = np.roll(quat_g, 1)

    print("Resulting global quaternion (quat_g):", quat_g)
    return np.concatenate([pos_g, quat_g])


def rot_opencv_to_sapien(euler_xyz=[0.609621798630839, -0.0015185817574139282, -3.1243143293420315]):
    r_gc = R.from_euler(
        "xyz", euler_xyz
    )

    r_cs = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ]
    )

    r_gc = r_gc.as_matrix()
    r_gs = np.dot(r_gc, r_cs)

    new_quat = R.from_matrix(r_gs).as_quat()
    new_quat = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
    print(new_quat)
    return new_quat


if __name__ == "__main__":
    # Example input
    robot_base_global_pose = [
        -0.547,
        -0.527,
        -0.143,
        0.9238795325127119,
        -0.0,
        -0.0,
        0.3826834323616491,
    ]
    camera_robot_base_pose = [
        0.7182223102548958,
        0.7488545305630568,
        0.7120899543920938,
        0.03282993931453109,
        -0.1992157118799819,
        -0.907170672593098,
        0.36915669574280047
    ]

    # Compute camera global pose
    camera_global_pose = get_sim_camera_global_pose_from_real(
        robot_base_global_pose, camera_robot_base_pose
    )
    print("Camera's pose in the global frame:", camera_global_pose)
