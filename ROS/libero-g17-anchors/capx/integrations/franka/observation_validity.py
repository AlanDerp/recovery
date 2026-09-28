"""Validity checks for rigid-object pose observations."""

from __future__ import annotations

import numpy as np


# A rigid body cannot change a full side length by this factor between two
# observations. The bound remains deliberately broad for noisy OBB fits while
# rejecting masks that absorb a neighboring object or the table.
MAX_RIGID_SCALE_CHANGE = 3.0


def validate_obb_geometry(
    center: np.ndarray, rotation: np.ndarray, extent: np.ndarray
) -> float:
    """Validate an OBB and return its rotation-invariant characteristic scale."""
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)

    if center.shape != (3,) or rotation.shape != (3, 3) or extent.shape != (3,):
        raise ValueError("object-pose OBB has an invalid shape")
    if not (
        np.all(np.isfinite(center))
        and np.all(np.isfinite(rotation))
        and np.all(np.isfinite(extent))
    ):
        raise ValueError("object-pose OBB contains non-finite values")
    if np.any(extent <= 0.0):
        raise ValueError("object-pose OBB has a non-positive extent")

    scale = float(np.linalg.norm(extent))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("object-pose OBB has an invalid characteristic scale")
    return scale


def rigid_extent_signature(extent: np.ndarray) -> np.ndarray:
    """Return an axis-order-independent full-extent signature."""
    extent = np.asarray(extent, dtype=np.float64)
    if extent.shape != (3,) or not np.all(np.isfinite(extent)):
        raise ValueError("object-pose OBB extent has an invalid shape or value")
    if np.any(extent <= 0.0):
        raise ValueError("object-pose OBB has a non-positive extent")
    return np.sort(extent)


def rigid_scale_is_consistent(current_scale: float, prior_scale: float | None) -> bool:
    """Return whether two observations can represent the same rigid object."""
    if prior_scale is None:
        return True
    if not (
        np.isfinite(current_scale)
        and np.isfinite(prior_scale)
        and current_scale > 0.0
        and prior_scale > 0.0
    ):
        return False
    ratio = max(current_scale / prior_scale, prior_scale / current_scale)
    return ratio <= MAX_RIGID_SCALE_CHANGE


def rigid_extent_is_consistent(
    current_extent: np.ndarray,
    prior_extent: np.ndarray | None,
) -> bool:
    """Check every rigid-object dimension against its frozen first observation.

    OBB axes may permute as the object rotates, so dimensions are compared after
    sorting.  Comparing only the extent norm misses a collapsed single axis.
    """
    if prior_extent is None:
        return True
    try:
        current = rigid_extent_signature(current_extent)
        prior = rigid_extent_signature(prior_extent)
    except ValueError:
        return False
    ratios = np.maximum(current / prior, prior / current)
    return bool(np.all(ratios <= MAX_RIGID_SCALE_CHANGE))
