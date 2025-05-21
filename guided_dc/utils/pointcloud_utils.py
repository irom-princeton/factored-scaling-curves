from __future__ import annotations

from typing import (
    Dict,
)

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

RGB_DTYPE = np.uint8
DEPTH_DTYPE = np.float32
SEGMENTATION_DTYPE = bool


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class PointCloud:
    xyz_pts: np.ndarray
    normals: np.ndarray
    rgb_pts: np.ndarray = None
    segmentation_pts: Dict[str, np.ndarray] = None

    @classmethod
    def rgb_dtype(cls, rgb_pts: np.ndarray):
        if (
            (rgb_pts.dtype in {np.float32, np.float64})
            and rgb_pts.max() < 1.0
            and rgb_pts.min() > 0.0
        ):
            rgb_pts = rgb_pts * 255
            return rgb_pts.astype(RGB_DTYPE)
        elif rgb_pts.dtype == RGB_DTYPE:
            return rgb_pts
        else:
            raise ValueError(f"`rgb_pts` in unexpected format: dtype {rgb_pts.dtype}")

    @classmethod
    def segmentation_pts_shape(cls, v: Dict[str, np.ndarray]):
        for pts in v.values():
            if len(pts.shape) > 2:
                raise ValueError(f"points.shape should N, but got {pts.shape}")
        return v

    @classmethod
    def xyz_pts_shape(cls, v: np.ndarray):
        if len(v.shape) != 2 or v.shape[1] != 3:
            raise ValueError("points should be Nx3")
        return v

    @classmethod
    def same_len(cls, v: np.ndarray, values):
        if "rgb_pts" in values and len(values["rgb_pts"]) != len(v):
            raise ValueError("`len(rgb_pts) != len(xyz_pts)`")
        if "segmentation_pts" in values and not all(
            len(pts) == len(v) for pts in values["segmentation_pts"].values()
        ):
            raise ValueError("`len(segmentation_pts) != len(xyz_pts)`")
        return v

    def __len__(self):
        return len(self.xyz_pts)

    def __add__(self, other: PointCloud):
        return PointCloud(
            xyz_pts=np.concatenate((self.xyz_pts, other.xyz_pts), axis=0),
            rgb_pts=np.concatenate((self.rgb_pts, other.rgb_pts), axis=0),
            segmentation_pts={
                k: np.concatenate(
                    (self.segmentation_pts[k], other.segmentation_pts[k]), axis=0
                )
                for k in self.segmentation_pts.keys()
            },
        )

    def to_open3d(self, use_segmentation_pts: bool = False) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz_pts)
        if use_segmentation_pts:
            pcd.colors = o3d.utility.Vector3dVector(
                np.stack(
                    [
                        np.stack(list(self.segmentation_pts.values()), axis=1).argmax(
                            axis=1
                        )
                    ]
                    * 3,
                    axis=1,
                )
            )
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.rgb_pts / 255)
        return pcd

    def visualize_o3d(self, use_segmentation_pts: bool = False):
        o3d.visualization.draw_geometries([self.to_open3d(use_segmentation_pts)])
        breakpoint()

    def voxel_downsample(
        self, voxel_dim: float = 0.015, skip_segmentation: bool = False
    ) -> PointCloud:
        pcd = self.to_open3d()
        pcd = pcd.voxel_down_sample(voxel_dim)
        xyz_pts = np.array(pcd.points).astype(DEPTH_DTYPE)
        rgb_pts = (np.array(pcd.colors) * 255).astype(RGB_DTYPE)
        if skip_segmentation:
            return PointCloud(
                xyz_pts=xyz_pts,
                rgb_pts=rgb_pts,
                segmentation_pts={},
            )
        assert (
            len(self.xyz_pts) < 20000
        ), f"{len(self.xyz_pts)} is too many points, will consume too much RAM"
        distances = ((xyz_pts[None, ...] - self.xyz_pts[:, None, :]) ** 2).sum(axis=2)
        indices = distances.argmin(axis=0)

        segmentation_pts = {k: v[indices] for k, v in self.segmentation_pts.items()}
        return PointCloud(
            xyz_pts=xyz_pts,
            rgb_pts=rgb_pts,
            segmentation_pts=segmentation_pts,
        )

    # @property
    # def normals(self) -> np.ndarray:
    #     pcd = self.to_open3d()
    #     pcd.estimate_normals(
    #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
    #     )
    #     # visualize
    #     return np.asarray(pcd.normals)

    def filter_bounds(self, bounds):
        in_bounds_mask = np.logical_and(
            (self.xyz_pts > np.array(bounds[0])).all(axis=1),
            (self.xyz_pts < np.array(bounds[1])).all(axis=1),
        )
        return PointCloud(
            xyz_pts=self.xyz_pts[in_bounds_mask],
            rgb_pts=self.rgb_pts[in_bounds_mask],
            segmentation_pts={
                k: self.segmentation_pts[k][in_bounds_mask]
                for k in self.segmentation_pts.keys()
            },
        )

    def subsample(
        self, num_pts: int, numpy_random: np.random.RandomState
    ) -> PointCloud:
        indices = numpy_random.choice(
            len(self), size=num_pts, replace=num_pts > len(self)
        )
        return PointCloud(
            xyz_pts=self.xyz_pts[indices],
            rgb_pts=self.rgb_pts[indices],
            segmentation_pts={k: v[indices] for k, v in self.segmentation_pts.items()},
        )

    def __getitem__(self, key: str) -> PointCloud:
        assert key in self.segmentation_pts

        seg_mask = self.segmentation_pts[key]
        if not seg_mask.any():
            return PointCloud(
                xyz_pts=np.empty((0, 3), dtype=DEPTH_DTYPE),
                rgb_pts=np.empty((0, 3), dtype=RGB_DTYPE),
                segmentation_pts={key: np.ones(seg_mask.sum(), dtype=bool)},
            )
        link_point_cloud = PointCloud(
            xyz_pts=self.xyz_pts[seg_mask],
            rgb_pts=self.rgb_pts[seg_mask],
            segmentation_pts={key: np.ones(seg_mask.sum(), dtype=bool)},
        )
        if len(link_point_cloud) == 0:
            return link_point_cloud

        # help remove outliers due to noisy segmentations
        # this is actually quite expensive, and can be improved
        _, ind = link_point_cloud.to_open3d().remove_radius_outlier(
            nb_points=32, radius=0.02
        )
        return PointCloud(
            xyz_pts=link_point_cloud.xyz_pts[ind],
            rgb_pts=link_point_cloud.rgb_pts[ind],
            segmentation_pts={key: np.ones(len(ind), dtype=bool)},
        )

    def visualize_mesh(self, meshes):
        for i, handle_mesh in enumerate(meshes):
            # Sample points from the mesh surface (point cloud)
            num_points = 10000
            point_cloud = handle_mesh.sample(num_points)
            face_centroids = self.handle_link_pos[i].cpu().numpy()
            face_normals = self.handle_link_normal[i].cpu().numpy()
            # face_centroids = handle_mesh.bounding_box.center_mass
            # face_normals = get_normal_axis_direction(handle_mesh).squeeze()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Plot the point cloud of the mesh
            ax.scatter(
                point_cloud[:, 0],
                point_cloud[:, 1],
                point_cloud[:, 2],
                color="cyan",
                s=2,
                label="Point Cloud",
            )

            # Plot the face normals (centroids + normal vectors)
            ax.quiver(
                face_centroids[0],
                face_centroids[1],
                face_centroids[2],
                face_normals[0],
                face_normals[1],
                face_normals[2],
                length=0.05,
                color="red",
                label="Face Normals",
            )

            # Labels and plot settings
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()

            # Save the plot as an image file (e.g., PNG)
            # plt.savefig('mesh_normals_with_grasp_pose.png', dpi=300)
            plt.show()

            # Optional: close the plot if you're running this in a script
            plt.close()
