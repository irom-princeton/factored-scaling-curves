import numpy as np


class DummyEnv:
    def __init__(self, img_size):
        self.img_size = img_size
        assert len(self.img_size) == 3 and self.img_size[0] == 3
        self.img_size = (self.img_size[1], self.img_size[2], self.img_size[0])

        class CameraReader:
            def set_trajectory_mode(self):
                pass

        self.camera_reader = CameraReader()

    def reset(self, randomize=False):
        return None, None

    def get_observation(self):
        img = {f"{i}": np.random.randint(0, 255, size=(self.img_size)) for i in [0, 3]}
        obs = {
            "image": img,
            "robot_state": {
                "cartesian_position": np.random.rand(6),
                "joint_positions": np.random.rand(7),
                "gripper_position": np.random.rand(1).item(),
            },
        }
        return obs

    def step(self, action):
        return None, None, None, None, None
