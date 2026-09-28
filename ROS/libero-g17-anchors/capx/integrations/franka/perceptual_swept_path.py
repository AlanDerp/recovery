"""Observation-only swept-path audit for LIBERO acquisition candidates.

This module never reads simulator bodies, contacts, or privileged poses.  It
uses the same RGB-D point clouds available to the robot, measured joints, and
the robot URDF collision model.  v296a is audit-only: reports cannot change
candidate selection or execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import pyroki as pk
from scipy.spatial import cKDTree

from capx.utils.depth_utils import depth_to_pointcloud


_SCENE_POINTS = 2048
_TARGET_POINTS = 1024
_PATH_SAMPLES = 25
_POINT_RADIUS_M = 0.008
_TARGET_ASSOCIATION_M = 0.025
_ROBOT_POINT_REJECTION_M = 0.012


def enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V296A_PT_SWEPT_AUDIT", "0") == "1"


def _voxel_sample(
    points: np.ndarray, *, voxel_m: float, count: int
) -> tuple[np.ndarray, int]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[
        (pts[:, 0] > -0.5)
        & (pts[:, 0] < 1.5)
        & (pts[:, 1] > -1.0)
        & (pts[:, 1] < 1.0)
        & (pts[:, 2] > -0.15)
        & (pts[:, 2] < 1.5)
    ]
    if len(pts) == 0:
        return np.full((count, 3), 50.0, dtype=np.float64), 0
    keys = np.floor(pts / float(voxel_m)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    sampled = pts[np.sort(first)]
    if len(sampled) > count:
        indices = np.linspace(0, len(sampled) - 1, count, dtype=np.int64)
        sampled = sampled[indices]
    valid = len(sampled)
    if valid < count:
        pad = np.full((count - valid, 3), 50.0, dtype=np.float64)
        pad[:, 0] += np.arange(count - valid, dtype=np.float64) * 0.01
        sampled = np.concatenate([sampled, pad], axis=0)
    return sampled, int(valid)


def _distances(
    api: Any, configs: np.ndarray, points: np.ndarray, *, radius_m: float
) -> np.ndarray:
    cfg = np.asarray(configs, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    robot_geom = api._robot_coll.at_config(api._robot, jnp.asarray(cfg))
    world = pk.collision.Sphere.from_center_and_radius(
        jnp.asarray(pts), jnp.full((len(pts),), float(radius_m))
    )
    return np.asarray(
        pk.collision.collide(
            robot_geom.reshape((len(cfg), api._robot_coll.num_links, 1)),
            world.reshape((1, 1, len(pts))),
        )
    )


@dataclass(frozen=True)
class PerceptualScene:
    scene_points: np.ndarray
    scene_valid: int
    target_points: np.ndarray
    target_valid: int
    input_point_count: int
    rejected_robot_points: int
    rejected_target_points: int


def prepare_scene_from_observation(
    api: Any, pc_target: np.ndarray, measured_cfg: np.ndarray
) -> PerceptualScene:
    """Rebuild scene at the exact command boundary from current robot RGB-D."""
    obs = api.get_observation()
    parts = []
    for camera_name in (api.camera_name, api.wrist_camera_name):
        camera = obs[camera_name]
        points_camera = depth_to_pointcloud(
            camera["images"]["depth"], camera["intrinsics"], subsample_factor=1
        )
        homogeneous = np.concatenate(
            [points_camera, np.ones((len(points_camera), 1))], axis=1
        )
        parts.append((camera["pose_mat"] @ homogeneous.T).T[:, :3])
    return prepare_scene(
        api, np.concatenate(parts), np.asarray(pc_target), measured_cfg
    )


def prepare_scene(
    api: Any,
    pc_full: np.ndarray,
    pc_target: np.ndarray,
    measured_cfg: np.ndarray,
) -> PerceptualScene:
    target, target_valid = _voxel_sample(
        pc_target, voxel_m=0.012, count=_TARGET_POINTS
    )
    full, full_valid = _voxel_sample(pc_full, voxel_m=0.018, count=8192)
    live_full = full[:full_valid]
    live_target = target[:target_valid]

    target_mask = np.zeros(len(live_full), dtype=bool)
    if len(live_full) and len(live_target):
        target_mask = (
            cKDTree(live_target).query(live_full, k=1, workers=1)[0]
            <= _TARGET_ASSOCIATION_M
        )
    non_target = live_full[~target_mask]

    rejected_robot = 0
    if len(non_target):
        current_dist = _distances(
            api,
            np.asarray(measured_cfg, dtype=np.float64)[None, :],
            non_target,
            radius_m=0.0,
        )[0]
        robot_surface = np.min(current_dist, axis=0) <= _ROBOT_POINT_REJECTION_M
        rejected_robot = int(np.sum(robot_surface))
        non_target = non_target[~robot_surface]

    scene, scene_valid = _voxel_sample(
        non_target, voxel_m=0.020, count=_SCENE_POINTS
    )
    return PerceptualScene(
        scene_points=scene,
        scene_valid=scene_valid,
        target_points=target,
        target_valid=target_valid,
        input_point_count=int(full_valid),
        rejected_robot_points=rejected_robot,
        rejected_target_points=int(np.sum(target_mask)),
    )


def audit_candidate_path(
    api: Any,
    scene: PerceptualScene,
    measured_cfg: np.ndarray,
    candidate_cfg: np.ndarray,
) -> dict[str, Any]:
    start = np.asarray(measured_cfg, dtype=np.float64).reshape(-1)
    goal = np.asarray(candidate_cfg, dtype=np.float64).reshape(-1)
    alpha = np.linspace(0.0, 1.0, _PATH_SAMPLES, dtype=np.float64)[:, None]
    path = start[None, :] + alpha * (goal[None, :] - start[None, :])
    names = tuple(str(name) for name in api._robot_coll.link_names)
    monitored = np.array(
        [
            idx
            for idx, name in enumerate(names)
            if name
            not in {
                "panda_link0",
                "panda_link1",
                "panda_link2",
                "panda_link3",
            }
        ],
        dtype=np.int64,
    )
    arm_target = np.array(
        [
            idx
            for idx, name in enumerate(names)
            if name.startswith("panda_link") and name not in {"panda_link0", "panda_link1", "panda_link2", "panda_link3"}
        ],
        dtype=np.int64,
    )

    scene_all_dist = _distances(
        api, path, scene.scene_points, radius_m=_POINT_RADIUS_M
    )
    scene_dist = scene_all_dist[:, monitored, : scene.scene_valid]
    if scene_dist.size:
        flat = int(np.argmin(scene_dist))
        ti, li, _ = np.unravel_index(flat, scene_dist.shape)
        scene_min = float(scene_dist[ti, li].min())
        first = np.argwhere(np.min(scene_dist, axis=(1, 2)) <= 0.0)
        first_scene = int(first[0, 0]) if len(first) else None
        scene_link = names[int(monitored[li])]
    else:
        scene_min, first_scene, scene_link = float("inf"), None, None

    # Manipulation necessarily brings the hand/fingers into observed object
    # surfaces.  Keep that conservative whole-robot diagnostic above, but
    # separately measure the failure mode v296 is intended to identify:
    # proximal arm links sweeping through non-target scene geometry.
    scene_arm_dist = scene_all_dist[:, arm_target, : scene.scene_valid]
    if scene_arm_dist.size:
        flat = int(np.argmin(scene_arm_dist))
        ti, li, _ = np.unravel_index(flat, scene_arm_dist.shape)
        scene_arm_min_sample = int(ti)
        scene_arm_min = float(scene_arm_dist[ti, li].min())
        first = np.argwhere(np.min(scene_arm_dist, axis=(1, 2)) <= 0.0)
        first_scene_arm = int(first[0, 0]) if len(first) else None
        scene_arm_link = names[int(arm_target[li])]
        endpoint_flat = int(np.argmin(scene_arm_dist[-1]))
        endpoint_li, _ = np.unravel_index(
            endpoint_flat, scene_arm_dist[-1].shape
        )
        scene_arm_endpoint_min = float(scene_arm_dist[-1, endpoint_li].min())
        scene_arm_endpoint_link = names[int(arm_target[endpoint_li])]
    else:
        scene_arm_min_sample = None
        scene_arm_min, first_scene_arm, scene_arm_link = float("inf"), None, None
        scene_arm_endpoint_min, scene_arm_endpoint_link = float("inf"), None

    target_dist = _distances(
        api, path, scene.target_points, radius_m=_POINT_RADIUS_M
    )[:, arm_target, : scene.target_valid]
    if target_dist.size:
        flat = int(np.argmin(target_dist))
        ti, li, _ = np.unravel_index(flat, target_dist.shape)
        target_min = float(target_dist[ti, li].min())
        first = np.argwhere(np.min(target_dist, axis=(1, 2)) <= 0.0)
        first_target = int(first[0, 0]) if len(first) else None
        target_link = names[int(arm_target[li])]
    else:
        target_min, first_target, target_link = float("inf"), None, None

    return {
        "schema": "capx_libero_v296a_pt_swept_path_v1",
        "source": "robot_rgbd_plus_measured_joints_plus_urdf",
        "behavior_effect": "audit_only",
        "path_samples": _PATH_SAMPLES,
        "scene_valid_points": scene.scene_valid,
        "target_valid_points": scene.target_valid,
        "input_voxel_points": scene.input_point_count,
        "rejected_robot_points": scene.rejected_robot_points,
        "rejected_target_points": scene.rejected_target_points,
        "scene_min_distance_m": scene_min,
        "scene_first_predicted_collision_sample": first_scene,
        "scene_closest_link": scene_link,
        "scene_arm_min_distance_m": scene_arm_min,
        "scene_arm_min_distance_sample": scene_arm_min_sample,
        "scene_arm_first_predicted_collision_sample": first_scene_arm,
        "scene_arm_closest_link": scene_arm_link,
        "scene_arm_endpoint_min_distance_m": scene_arm_endpoint_min,
        "scene_arm_endpoint_closest_link": scene_arm_endpoint_link,
        "predicted_arm_obstacle_path_clear": bool(scene_arm_min > 0.0),
        "target_arm_min_distance_m": target_min,
        "target_arm_first_predicted_collision_sample": first_target,
        "target_arm_closest_link": target_link,
        "predicted_native_path_clear": bool(scene_min > 0.0 and target_min > 0.0),
    }
