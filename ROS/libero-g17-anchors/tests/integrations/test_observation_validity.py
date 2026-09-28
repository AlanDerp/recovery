import numpy as np
import pytest

from capx.integrations.franka.observation_validity import (
    rigid_extent_is_consistent,
    rigid_extent_signature,
    rigid_scale_is_consistent,
    validate_obb_geometry,
)


def test_valid_obb_scale_is_rotation_invariant() -> None:
    extent = np.array([0.08, 0.06, 0.04])
    scale = validate_obb_geometry(np.zeros(3), np.eye(3), extent)

    assert scale == pytest.approx(np.linalg.norm(extent))
    assert rigid_scale_is_consistent(scale, np.linalg.norm(extent[::-1]))


def test_collapsed_rigid_object_observation_is_rejected() -> None:
    prior = np.linalg.norm([0.08, 0.06, 0.04])
    collapsed = np.linalg.norm([0.0023, 0.0016, 0.00003])

    assert not rigid_scale_is_consistent(collapsed, prior)


def test_single_axis_collapse_is_rejected_even_when_norm_is_plausible() -> None:
    prior = np.array([0.0663, 0.0547, 0.0323])
    collapsed = np.array([0.0433, 0.0285, 0.0019])

    assert rigid_scale_is_consistent(np.linalg.norm(collapsed), np.linalg.norm(prior))
    assert not rigid_extent_is_consistent(collapsed, prior)


def test_mask_expansion_seen_in_stack_is_rejected() -> None:
    prior = np.array([0.07123433, 0.05290246, 0.04557170])
    expanded = np.array([0.24644953, 0.13612460, 0.12954742])

    assert not rigid_extent_is_consistent(expanded, prior)


def test_extent_signature_is_axis_order_independent() -> None:
    extent = np.array([0.08, 0.04, 0.06])

    assert np.array_equal(
        rigid_extent_signature(extent),
        rigid_extent_signature(extent[::-1]),
    )
    assert rigid_extent_is_consistent(extent, extent[::-1])


def test_frozen_reference_rejects_gradual_scale_drift() -> None:
    reference = 1.0
    rolling_observations = [0.5, 0.2, 0.1]

    assert all(
        rigid_scale_is_consistent(current, prior)
        for prior, current in zip(
            [reference, *rolling_observations[:-1]],
            rolling_observations,
            strict=True,
        )
    )
    assert not rigid_scale_is_consistent(
        rolling_observations[-1], reference
    )


@pytest.mark.parametrize(
    ("center", "rotation", "extent"),
    [
        (np.array([np.nan, 0.0, 0.0]), np.eye(3), np.ones(3)),
        (np.zeros(3), np.eye(3), np.array([1.0, 0.0, 1.0])),
        (np.zeros(2), np.eye(3), np.ones(3)),
    ],
)
def test_invalid_obb_geometry_is_rejected(center, rotation, extent) -> None:
    with pytest.raises(ValueError):
        validate_obb_geometry(center, rotation, extent)
