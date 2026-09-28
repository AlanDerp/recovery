from __future__ import annotations

import numpy as np

from capx.utils.depth_utils import masked_depth_color_to_pointcloud


def test_masked_pointcloud_uses_the_same_depth_validity_domain() -> None:
    depth = np.array([[1.0, np.nan], [0.0, 2.0]], dtype=np.float64)
    image = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    intrinsics = np.eye(3, dtype=np.float64)
    mask = np.ones((2, 2), dtype=bool)

    points, colors = masked_depth_color_to_pointcloud(
        depth, image, intrinsics, mask
    )

    assert points.shape == (2, 3)
    assert np.allclose(points[:, 2], [1.0, 2.0])
    assert np.allclose(colors, [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_masked_pointcloud_returns_empty_when_mask_has_no_valid_depth() -> None:
    depth = np.array([[np.nan, 0.0]], dtype=np.float64)
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    intrinsics = np.eye(3, dtype=np.float64)

    points, colors = masked_depth_color_to_pointcloud(
        depth, image, intrinsics, np.ones((1, 2), dtype=bool)
    )

    assert points.shape == (0, 3)
    assert colors.shape == (0, 3)
