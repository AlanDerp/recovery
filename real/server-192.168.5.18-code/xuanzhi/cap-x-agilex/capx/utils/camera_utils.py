"""
Camera utility functions for processing observation data.
"""

from typing import Any

import numpy as np


def scale_intrinsics_for_image(
    intrinsics: np.ndarray,
    target_shape: tuple[int, int],
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Scale pinhole intrinsics from ``ref_shape`` (H, W) to ``target_shape`` (H, W)."""
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    ref_h, ref_w = int(ref_shape[0]), int(ref_shape[1])
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3).copy()
    if target_h == ref_h and target_w == ref_w:
        return k
    scale_x = target_w / ref_w
    scale_y = target_h / ref_h
    k[0, 0] *= scale_x
    k[1, 1] *= scale_y
    k[0, 2] *= scale_x
    k[1, 2] *= scale_y
    return k


def obs_get_rgb(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Recursively search through observation dictionary to find RGB images.

    Args:
        obs: Observation dictionary that may contain nested camera data

    Returns:
        Dictionary mapping camera names to RGB image arrays
    """
    rgb_dict = {}

    for key, value in obs.items():
        if isinstance(value, dict):
            # Check if this dict contains images with rgb data
            if "images" in value and isinstance(value["images"], dict):
                if "rgb" in value["images"]:
                    rgb_dict[key] = value["images"]["rgb"]
            else:
                # Recursively search in nested dictionaries
                nested_rgb = obs_get_rgb(value)
                rgb_dict.update(nested_rgb)

    return rgb_dict
