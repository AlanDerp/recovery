import cv2
import numpy as np
from types import SimpleNamespace

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel
from capx.integrations.r1lite.head_stereo_calibration import (
    build_rectified_plane_calibration,
)


def test_raw_chessboard_pixels_are_converted_to_rectified_domain():
    image_size = (640, 512)
    intrinsics = np.array(
        [[420.0, 0.0, 320.0], [0.0, 418.0, 256.0], [0.0, 0.0, 1.0]]
    )
    camera_from_base = np.eye(4)
    camera_from_base[:3, 3] = [0.02, -0.03, 1.10]
    base_points = np.asarray(
        [
            [x, y, 0.0]
            for x in np.linspace(-0.20, 0.20, 6)
            for y in np.linspace(-0.14, 0.14, 5)
        ]
    )
    pixels, _ = cv2.projectPoints(
        base_points,
        np.zeros(3),
        camera_from_base[:3, 3],
        intrinsics,
        None,
    )
    raw = {
        "image_size": list(image_size),
        "plane_z": 0.0,
        "points": [
            {"pixel": pixel.tolist(), "base_xy": point[:2].tolist()}
            for pixel, point in zip(
                pixels.reshape(-1, 2), base_points, strict=True
            )
        ],
    }

    result = build_rectified_plane_calibration(
        raw,
        left_camera_matrix=intrinsics.reshape(-1),
        left_distortion=[0.0] * 5,
        left_info_size=image_size,
        right_camera_matrix=intrinsics.reshape(-1),
        right_distortion=[0.0] * 5,
        right_info_size=image_size,
        left_to_right_rotation=np.eye(3).reshape(-1),
        left_to_right_translation_m=[-0.065, 0.0, 0.0],
        target_size=image_size,
    )

    assert result["image_domain"] == "head_left_stereo_rectified"
    assert result["image_size"] == [640, 512]
    assert result["num_points"] == len(base_points)
    assert result["reprojection_rmse_px"] < 1e-6
    np.testing.assert_allclose(
        result["base_from_rectified_left_camera"],
        np.linalg.inv(camera_from_base),
        atol=1e-7,
    )


def test_conversion_rejects_missing_correspondences():
    try:
        build_rectified_plane_calibration(
            {"image_size": [640, 512], "plane_z": 0.0, "points": []},
            left_camera_matrix=np.eye(3).reshape(-1),
            left_distortion=[],
            left_info_size=(640, 512),
            right_camera_matrix=np.eye(3).reshape(-1),
            right_distortion=[],
            right_info_size=(640, 512),
            left_to_right_rotation=np.eye(3).reshape(-1),
            left_to_right_translation_m=[-0.065, 0.0, 0.0],
        )
    except ValueError as error:
        assert "at least four points" in str(error)
    else:
        raise AssertionError("missing correspondences must be rejected")


def test_head_rgbd_temporal_median_uses_fresh_depth_frames():
    state = {"stamp": 0}

    def get_head_rgbd(*, camera, require_depth, newer_than_ns):
        assert camera == "left"
        assert require_depth is True
        assert newer_than_ns == (None if state["stamp"] == 0 else state["stamp"])
        state["stamp"] += 1
        value = float(state["stamp"])
        return {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.full((2, 2), value),
            "intrinsics": np.eye(3),
            "frame_id": "camera_head_left",
            "stamp_ns": state["stamp"],
        }

    fake = SimpleNamespace(
        head_rgbd_median_sample_count=15,
        get_head_rgbd=get_head_rgbd,
    )
    result = R1LiteRealLowLevel.get_head_rgbd_median(fake)

    assert result["depth_sample_count"] == 15
    assert result["depth_stamp_range_ns"] == [1, 15]
    np.testing.assert_allclose(result["depth"], 8.0)
