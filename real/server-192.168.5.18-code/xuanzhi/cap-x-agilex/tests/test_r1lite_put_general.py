from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/r1lite_execute_put_general.py"
SPEC = importlib.util.spec_from_file_location("put_general", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PUT_GENERAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUT_GENERAL)


@pytest.mark.parametrize(
    "order",
    list(itertools.permutations(("tennis", "peach", "apple"))),
)
def test_every_put_object_order_is_supported(
    order: tuple[str, str, str],
) -> None:
    normalized = PUT_GENERAL.normalize_object_order(order)
    assert set(normalized) == {"tennis ball", "peach", "apple"}
    assert len(normalized) == 3


@pytest.mark.parametrize(
    "order",
    [
        ("tennis", "tennis", "apple"),
        ("tennis", "peach"),
        ("tennis", "peach", "orange"),
    ],
)
def test_invalid_put_object_orders_are_rejected(
    order: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="unique permutation"):
        PUT_GENERAL.normalize_object_order(order)


def test_held_object_prediction_tracks_tool_rotation() -> None:
    tcp = np.array([0.4, -0.2, 1.0])
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    offset = np.array([0.0, 0.0, -0.08])
    predicted = PUT_GENERAL.predicted_held_object_center(
        tcp, identity, offset
    )
    assert predicted.tolist() == pytest.approx([0.4, -0.2, 0.92])


def test_grasp_orientation_candidates_remain_near_top_down() -> None:
    topdown = np.array([0.0, 1.0, 0.0, 0.0])
    candidates = PUT_GENERAL.build_grasp_orientation_candidates(topdown)
    assert np.allclose(candidates[0], topdown)
    assert len(candidates) > 1
    for candidate in candidates:
        tool_z = PUT_GENERAL.quaternion_matrix(candidate)[:, 2]
        tilt_rad = np.arccos(np.clip(-tool_z[2], -1.0, 1.0))
        assert tilt_rad <= np.deg2rad(20.0) + 1e-9


def test_tilted_grasp_tcp_places_fingertip_at_object_center() -> None:
    center = np.array([0.44, -0.16, 0.89])
    topdown = np.array([0.0, 1.0, 0.0, 0.0])
    tilted = PUT_GENERAL.build_grasp_orientation_candidates(topdown)[7]
    tcp = PUT_GENERAL.grasp_tcp_for_orientation(center, tilted, 0.078)
    fingertip = (
        tcp
        + PUT_GENERAL.quaternion_matrix(tilted)
        @ np.array([0.0, 0.0, 0.078])
    )
    assert fingertip.tolist() == pytest.approx(center.tolist())


def test_grasp_ik_seeds_only_vary_j2_and_j4() -> None:
    current = np.array([0.1, 0.4, -0.2, 0.3, 0.5, -0.1])
    seeds = PUT_GENERAL.build_grasp_ik_seed_candidates(current)
    assert np.allclose(seeds[0], current)
    assert len(seeds) == 7
    for seed in seeds:
        assert np.allclose(seed[[0, 2, 4, 5]], current[[0, 2, 4, 5]])
        assert 0.05 <= seed[1] <= 3.05
        assert -1.45 <= seed[3] <= 1.45


def test_seeded_grasp_endpoint_waits_for_iterative_convergence() -> None:
    target = np.zeros(3)

    class Robot:
        def __init__(self) -> None:
            self.solutions = [
                np.array([0.10, 0.2, 0.0, 0.0, 0.0, 0.0]),
                np.array([0.01, 0.2, 0.0, 0.0, 0.0, 0.0]),
            ]
            self.validated: np.ndarray | None = None

        def get_torso_joints(self) -> np.ndarray:
            return np.zeros(3)

        def get_relaxed_ik_candidate(
            self,
            _position: np.ndarray,
            _quaternion: np.ndarray,
            **_kwargs: object,
        ) -> np.ndarray:
            return self.solutions.pop(0)

        def get_local_planning_fk_pose(
            self,
            joints: np.ndarray,
            _torso: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.array([joints[0], 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            )

        def validate_arm_joint_feedback_target(
            self,
            joints: np.ndarray,
            *,
            context: str,
        ) -> np.ndarray:
            assert "converged" in context
            self.validated = joints.copy()
            return joints

    robot = Robot()
    joints, error = PUT_GENERAL.converge_seeded_grasp_endpoint(
        robot,
        target,
        np.array([0.0, 1.0, 0.0, 0.0]),
        maximum_attempts=2,
        position_tolerance_m=0.025,
    )

    assert error == pytest.approx(0.01)
    assert np.allclose(joints, robot.validated)


def test_frozen_pose_plan_executes_validated_joint_waypoints() -> None:
    target = np.array([0.4, -0.2, 1.0])

    class Robot:
        goto_pose_joint_waypoint_step_rad = 0.1

        def __init__(self) -> None:
            self.commands: list[np.ndarray] = []

        def move_joint_waypoints_adaptive(
            self,
            joints: list[float],
            *,
            waypoint_step_rad: float,
            context: str,
        ) -> dict[str, object]:
            self.commands.append(np.asarray(joints, dtype=float))
            return {
                "waypoint_step_rad": waypoint_step_rad,
                "context": context,
            }

        def get_local_planning_fk_pose(
            self,
        ) -> tuple[np.ndarray, np.ndarray]:
            return target.copy(), np.array([0.0, 1.0, 0.0, 0.0])

    robot = Robot()
    plan = {
        "waypoints": [
            {"joint_feedback": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
            {"joint_feedback": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
        ]
    }
    result = PUT_GENERAL.execute_frozen_pose_plan(
        robot,
        plan,
        target,
        context="unit_frozen",
    )

    assert len(robot.commands) == 2
    assert np.allclose(robot.commands[0], plan["waypoints"][0]["joint_feedback"])
    assert np.allclose(robot.commands[1], plan["waypoints"][1]["joint_feedback"])
    assert result["position_error_m"] == pytest.approx(0.0)


def test_release_gate_requires_box_xy_and_height() -> None:
    box = {
        "opening_xy_min_base_m": [0.3, -0.3],
        "opening_xy_max_base_m": [0.5, -0.1],
        "rim_z_base_m": 0.9,
    }
    ready, details = PUT_GENERAL.object_center_is_over_box(
        np.array([0.4, -0.2, 1.02]),
        box,
        radius_m=0.04,
    )
    assert ready is True
    assert details["xy_inside"] is True
    outside, _ = PUT_GENERAL.object_center_is_over_box(
        np.array([0.55, -0.2, 1.02]),
        box,
        radius_m=0.04,
    )
    too_low, _ = PUT_GENERAL.object_center_is_over_box(
        np.array([0.4, -0.2, 0.93]),
        box,
        radius_m=0.04,
    )
    assert outside is False
    assert too_low is False
