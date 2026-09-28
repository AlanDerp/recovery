"""RGB-D reconstruction helpers for inverted colored cups."""

from __future__ import annotations

from typing import Any

import numpy as np


def reconstruct_inverted_cup_pose(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    mask: np.ndarray,
    *,
    cup_height_m: float,
    minimum_points: int = 80,
) -> dict[str, Any]:
    """Recover an inverted cup's upward-facing base and downward rim centers."""
    depth = np.asarray(depth_m, dtype=float)
    binary = np.asarray(mask, dtype=bool)
    camera_matrix = np.asarray(intrinsics, dtype=float).reshape(3, 3)
    transform = np.asarray(base_from_camera, dtype=float).reshape(4, 4)
    if depth.ndim != 2 or binary.shape != depth.shape:
        raise ValueError("cup mask and depth image must have the same 2D shape")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(transform).all():
        raise ValueError("cup RGB-D calibration must be finite")
    height = float(cup_height_m)
    if not 0.02 <= height <= 0.30:
        raise ValueError("cup height must be between 0.02m and 0.30m")

    ys, xs = np.nonzero(binary)
    values = depth[ys, xs]
    valid = np.isfinite(values) & (values >= 0.10) & (values <= 2.50)
    xs = xs[valid]
    ys = ys[valid]
    values = values[valid]
    if values.size < int(minimum_points):
        raise ValueError(
            f"cup mask has too few valid depth points: "
            f"{values.size} < {minimum_points}"
        )

    camera_points = np.column_stack(
        (
            (xs - camera_matrix[0, 2]) * values / camera_matrix[0, 0],
            (ys - camera_matrix[1, 2]) * values / camera_matrix[1, 1],
            values,
            np.ones_like(values),
        )
    )
    base_points = (transform @ camera_points.T).T[:, :3]
    finite = np.isfinite(base_points).all(axis=1)
    base_points = base_points[finite]
    if base_points.shape[0] < int(minimum_points):
        raise ValueError("cup point cloud has too few finite base-frame points")

    top_threshold = float(np.percentile(base_points[:, 2], 70.0))
    top_points = base_points[base_points[:, 2] >= top_threshold]
    top_center = np.median(top_points, axis=0)
    top_center[2] = float(np.median(top_points[:, 2]))
    rim_center = top_center.copy()
    rim_center[2] -= height
    body_center = top_center.copy()
    body_center[2] -= height / 2.0
    residuals = top_points[:, 2] - top_center[2]
    return {
        "top_center_base_m": top_center.tolist(),
        "body_center_base_m": body_center.tolist(),
        "rim_center_base_m": rim_center.tolist(),
        "cup_height_m": height,
        "valid_depth_points": int(base_points.shape[0]),
        "top_surface_points": int(top_points.shape[0]),
        "top_surface_z_mad_m": float(
            np.median(np.abs(residuals - np.median(residuals)))
        ),
        "position_semantics": {
            "top_center": "upward-facing closed cup base",
            "body_center": "geometric center inferred from measured cup height",
            "rim_center": "downward open rim center inferred from measured cup height",
        },
    }
