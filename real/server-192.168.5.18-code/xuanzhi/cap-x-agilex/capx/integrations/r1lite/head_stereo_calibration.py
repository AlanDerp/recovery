"""Calibration helpers for the rectified R1 Lite head-stereo image domain."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _camera_matrix_at_size(
    matrix: Sequence[float],
    source_size: Sequence[int],
    target_size: Sequence[int],
) -> np.ndarray:
    source_width, source_height = (float(value) for value in source_size)
    target_width, target_height = (float(value) for value in target_size)
    if min(source_width, source_height, target_width, target_height) <= 0.0:
        raise ValueError("camera image sizes must be positive")
    result = np.asarray(matrix, dtype=np.float64).reshape(3, 3).copy()
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    result[0, :] *= scale_x
    result[1, :] *= scale_y
    result[2, :] = [0.0, 0.0, 1.0]
    return result


def calibration_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_rectified_plane_calibration(
    raw_calibration: Mapping[str, Any],
    *,
    left_camera_matrix: Sequence[float],
    left_distortion: Sequence[float],
    left_info_size: Sequence[int],
    right_camera_matrix: Sequence[float],
    right_distortion: Sequence[float],
    right_info_size: Sequence[int],
    left_to_right_rotation: Sequence[float],
    left_to_right_translation_m: Sequence[float],
    target_size: Sequence[int] = (640, 512),
    alpha: float = 0.0,
) -> dict[str, Any]:
    """Convert raw-image table correspondences into the stereo-rectified domain."""
    raw_points = list(raw_calibration.get("points", []))
    if len(raw_points) < 4:
        raise ValueError("raw head-plane calibration needs at least four points")
    raw_size = np.asarray(raw_calibration.get("image_size", []), dtype=float)
    target = np.asarray(target_size, dtype=int)
    if raw_size.shape != (2,) or target.shape != (2,) or np.any(raw_size <= 0):
        raise ValueError("invalid raw or target image size")
    if np.any(target <= 0):
        raise ValueError("target image size must be positive")

    target_tuple = (int(target[0]), int(target[1]))
    k1 = _camera_matrix_at_size(
        left_camera_matrix, left_info_size, target_tuple
    )
    k2 = _camera_matrix_at_size(
        right_camera_matrix, right_info_size, target_tuple
    )
    d1 = np.asarray(left_distortion, dtype=np.float64).reshape(-1, 1)
    d2 = np.asarray(right_distortion, dtype=np.float64).reshape(-1, 1)
    stereo_rotation = np.asarray(
        left_to_right_rotation, dtype=np.float64
    ).reshape(3, 3)
    stereo_translation = np.asarray(
        left_to_right_translation_m, dtype=np.float64
    ).reshape(3, 1)
    r1, _r2, p1, _p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        k1,
        d1,
        k2,
        d2,
        target_tuple,
        stereo_rotation,
        stereo_translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )

    raw_pixels = np.asarray(
        [point["pixel"] for point in raw_points], dtype=np.float64
    ).reshape(-1, 2)
    processing_pixels = raw_pixels * (target.astype(float) / raw_size)
    rectified_pixels = cv2.undistortPoints(
        processing_pixels.reshape(-1, 1, 2),
        k1,
        d1,
        R=r1,
        P=p1,
    ).reshape(-1, 2)

    plane_z = float(raw_calibration["plane_z"])
    object_points = np.asarray(
        [[*point["base_xy"], plane_z] for point in raw_points],
        dtype=np.float64,
    )
    rectified_intrinsics = p1[:3, :3].copy()
    success, rotation_vector, translation = cv2.solvePnP(
        object_points,
        rectified_pixels,
        rectified_intrinsics,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("rectified solvePnP failed")
    camera_from_base = np.eye(4, dtype=np.float64)
    camera_from_base[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    camera_from_base[:3, 3] = translation.reshape(3)
    base_from_camera = np.linalg.inv(camera_from_base)

    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        rectified_intrinsics,
        None,
    )
    reprojection_errors = np.linalg.norm(
        projected.reshape(-1, 2) - rectified_pixels, axis=1
    )
    depths = (
        camera_from_base[:3, :3] @ object_points.T
        + camera_from_base[:3, 3:4]
    )[2]
    if (
        not np.isfinite(base_from_camera).all()
        or np.any(depths <= 0.05)
        or float(np.max(reprojection_errors)) > 10.0
    ):
        raise RuntimeError(
            "invalid rectified pose solution: "
            f"maximum_reprojection_error_px={float(np.max(reprojection_errors)):.3f}"
        )

    homography, _ = cv2.findHomography(rectified_pixels, object_points[:, :2])
    if homography is None:
        raise RuntimeError("rectified image-to-table homography solve failed")
    points = [
        {
            **dict(point),
            "raw_pixel": [float(value) for value in point["pixel"]],
            "pixel": rectified_pixel.tolist(),
        }
        for point, rectified_pixel in zip(
            raw_points, rectified_pixels, strict=True
        )
    ]
    return {
        "schema": "capx.r1lite.head_stereo_rectified_plane_calibration.v1",
        "image_domain": "head_left_stereo_rectified",
        "source": "converted_raw_bimanual_chessboard_correspondences",
        "source_calibration_sha256": calibration_sha256(raw_calibration),
        "image_size": target.tolist(),
        "plane_z": plane_z,
        "rectified_intrinsics": rectified_intrinsics.tolist(),
        "left_rectification_rotation": r1.tolist(),
        "left_to_right_rotation": stereo_rotation.tolist(),
        "left_to_right_translation_m": stereo_translation.reshape(3).tolist(),
        "alpha": float(alpha),
        "base_from_rectified_left_camera": base_from_camera.tolist(),
        "image_to_base_xy_homography": homography.tolist(),
        "reprojection_rmse_px": float(
            np.sqrt(np.mean(np.square(reprojection_errors)))
        ),
        "maximum_reprojection_error_px": float(np.max(reprojection_errors)),
        "reprojection_errors_px": reprojection_errors.tolist(),
        "num_points": len(points),
        "points": points,
    }
