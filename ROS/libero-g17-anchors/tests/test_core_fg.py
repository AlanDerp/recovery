from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from capx.verification.core_fg import (
    CoreFacts,
    _axis_relation,
    _height_relation,
    _horizontal_goal_projection,
    _rotation_matrix_wxyz,
    extract_core_facts,
    format_public_facts,
)


def test_axis_relations_are_closed_at_boundaries() -> None:
    assert _axis_relation(-0.026, 0.025, "x") == "target_center_negative_x_of_goal_band"
    assert _axis_relation(-0.025, 0.025, "x") == "target_center_within_goal_x_band"
    assert _axis_relation(0.025, 0.025, "x") == "target_center_within_goal_x_band"
    assert _axis_relation(0.026, 0.025, "x") == "target_center_positive_x_of_goal_band"


def test_height_relation_is_geometry_scaled_and_closed() -> None:
    assert _height_relation(-0.026, 0.02, 0.025) == "target_below_goal_height_band"
    assert _height_relation(0.0, 0.02, 0.025) == "target_within_goal_height_band"
    assert _height_relation(0.026, 0.02, 0.025) == "target_above_goal_height_band"


def test_quaternion_rotation_uses_wxyz() -> None:
    angle = np.pi / 2
    rotation = _rotation_matrix_wxyz([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_horizontal_projection_uses_one_frame_for_axes_and_footprint() -> None:
    goal_rotation = _rotation_matrix_wxyz(
        [-0.6881370398318637, 0.16268808933490214, -0.6881370398318624, -0.16268808933490203]
    )
    band_x, band_y, inside = _horizontal_goal_projection(
        goal_rotation,
        np.array([0.025, 0.025, 0.025]),
        np.array([0.062831903, 0.037666623]),
    )
    assert inside is False
    assert _axis_relation(0.062831903, band_x, "x") == "target_center_positive_x_of_goal_band"
    assert _axis_relation(0.037666623, band_y, "y") == "target_center_positive_y_of_goal_band"


def test_horizontal_projection_includes_closed_rotated_boundary() -> None:
    angle = np.pi / 4
    rotation = _rotation_matrix_wxyz(
        [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    )
    band_x, band_y, inside = _horizontal_goal_projection(
        rotation,
        np.array([0.025, 0.025, 0.025]),
        np.array([np.sqrt(2) * 0.025, 0.0]),
    )
    assert inside is True
    assert _axis_relation(np.sqrt(2) * 0.025, band_x, "x") == (
        "target_center_within_goal_x_band"
    )
    assert _axis_relation(0.0, band_y, "y") == "target_center_within_goal_y_band"


def test_public_packet_has_only_fixed_factual_fields() -> None:
    facts = CoreFacts(
        target_grasp="target_not_grasped_by_robot",
        relative_x="target_center_within_goal_x_band",
        relative_y="target_center_positive_y_of_goal_band",
        relative_z="target_above_goal_height_band",
        footprint="target_center_inside_goal_footprint",
        target_goal_contact="target_contacting_goal",
        target_table_contact="target_separated_from_table",
    )
    message = format_public_facts(facts)
    assert message.splitlines()[2:] == [
        "TARGET_GRASP: target_not_grasped_by_robot",
        "RELATIVE_X: target_center_within_goal_x_band",
        "RELATIVE_Y: target_center_positive_y_of_goal_band",
        "RELATIVE_Z: target_above_goal_height_band",
        "FOOTPRINT: target_center_inside_goal_footprint",
        "TARGET_GOAL_CONTACT: target_contacting_goal",
        "TARGET_TABLE_CONTACT: target_separated_from_table",
    ]
    lowered = message.lower()
    for forbidden in ("reward", "success", "task_completed", "move to", "open_gripper"):
        assert forbidden not in lowered


def test_full_extractor_uses_simulator_predicates() -> None:
    class FakeSimulator:
        cubeA = SimpleNamespace(size=[0.02, 0.02, 0.02])
        cubeB = SimpleNamespace(size=[0.025, 0.025, 0.025])
        cubeA_body_id = 0
        cubeB_body_id = 1
        robots = [SimpleNamespace(gripper=object())]
        sim = SimpleNamespace(
            data=SimpleNamespace(
                body_xpos=np.array([[0.01, -0.04, 0.10], [0.0, 0.0, 0.05]]),
                body_xquat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            )
        )

        def _check_grasp(self, *, gripper, object_geoms) -> bool:
            return True

        def check_contact(self, first, second) -> bool:
            return second == "table_collision"

    class FakeLowLevel:
        robosuite_env = FakeSimulator()

        def _cube_pose_dict(self, observation):
            return {
                "primary": [*observation["cubeA_pos"], 1.0, 0.0, 0.0, 0.0],
                "secondary": [*observation["cubeB_pos"], 1.0, 0.0, 0.0, 0.0],
            }

    facts = extract_core_facts(SimpleNamespace(low_level_env=FakeLowLevel()))
    assert facts == CoreFacts(
        target_grasp="target_grasped_by_robot",
        relative_x="target_center_within_goal_x_band",
        relative_y="target_center_negative_y_of_goal_band",
        relative_z="target_above_goal_height_band",
        footprint="target_center_outside_goal_footprint",
        target_goal_contact="target_separated_from_goal",
        target_table_contact="target_contacting_table",
    )
