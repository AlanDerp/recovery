from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from capx.integrations.r1lite.control import R1LiteIkHarness
from capx.integrations.r1lite.cup_stack_llm import (
    INTENT_SCHEMA,
    compile_general_plan,
    compile_llm_plan,
    execute_generated_code_dry_run,
    expected_operations,
    nearest_top_down_quaternion,
    validate_generated_code,
    validate_general_color_order,
    validate_llm_intent,
)

ROOT = Path(__file__).resolve().parents[1]


def intent() -> dict:
    return {
        "schema": INTENT_SCHEMA,
        "phase_order": ["orange_on_purple", "green_on_orange"],
        "phases": [
            {
                "id": "orange_on_purple",
                "source": "orange cup",
                "support": "purple cup",
                "grasp_height_ratio": 0.55,
                "pregrasp_clearance_m": 0.10,
                "lift_clearance_m": 0.15,
                "grasp_descent_steps": 12,
                "release_clearance_m": 0.03,
                "nested_height_increment_m": 0.0,
            },
            {
                "id": "green_on_orange",
                "source": "green cup",
                "support": "orange cup",
                "grasp_height_ratio": 0.55,
                "pregrasp_clearance_m": 0.11,
                "lift_clearance_m": 0.16,
                "grasp_descent_steps": 14,
                "release_clearance_m": 0.04,
                "nested_height_increment_m": 0.0,
            },
        ],
    }


def scene() -> dict:
    cups = {}
    for name, x, y in (
        ("orange cup", 0.42, -0.18),
        ("purple cup", 0.50, -0.02),
        ("green cup", 0.43, 0.16),
    ):
        cups[name] = {
            "rim_center_base_m": [x, y, 0.80],
            "body_center_base_m": [x, y, 0.85],
            "top_center_base_m": [x, y, 0.90],
        }
    return {
        "frame_id": "base_link",
        "cups": cups,
        "home_gripper_position_base_m": [0.30, -0.25, 1.10],
        "home_gripper_quaternion_xyzw": [0.7, 0.7, 0.1, 0.1],
        "home_torso_joints_rad": [-0.7, 1.36, 0.92],
    }


def compiled_plan() -> dict:
    return compile_llm_plan(
        intent(),
        scene(),
        cup_height_m=0.10,
        cup_diameter_m=0.07,
        gripper_close_position=20.0,
    )


def compiled_general_plan(colors: tuple[str, str, str]) -> dict:
    fixed = scene()
    fixed["pose_source"] = "fixed_user_measurement"
    return compile_general_plan(
        colors,
        fixed,
        cup_height_m=0.10,
        cup_diameter_m=0.07,
        gripper_close_position=20.0,
    )


def render(operations: list[dict]) -> str:
    lines = []
    for operation in operations:
        kwargs = ", ".join(
            f"{key}={value!r}"
            for key, value in operation.items()
            if key != "call"
        )
        lines.append(f"{operation['call']}({kwargs})")
    return "\n".join(lines)


def test_llm_intent_cannot_reorder_cups() -> None:
    value = intent()
    value["phase_order"].reverse()
    with pytest.raises(ValueError, match="phase order"):
        validate_llm_intent(value)


def test_llm_intent_is_compiled_to_same_height_nested_geometry() -> None:
    plan = compiled_plan()
    assert plan["phase_order"] == ["orange_on_purple", "green_on_orange"]
    assert plan["gripper_open_position"] == 60.0
    assert plan["orientation_policy"] == (
        "nearest_top_down_orientation_to_home"
    )
    expected_orientation = pytest.approx(
        [2**-0.5, 2**-0.5, 0.0, 0.0]
    )
    assert plan["phases"][0]["orientation_quaternion_xyzw"] == expected_orientation
    assert plan["phases"][1]["orientation_quaternion_xyzw"] == expected_orientation
    assert plan["home_to_top_down_rotation_rad"] < 0.30
    assert plan["ik_path_limits"] == {
        "max_cartesian_step_m": 0.06,
        "max_orientation_step_rad": 0.25,
        "expected_home_to_pregrasp_waypoints": 3,
        "low_level_joint_interpolation_retained": True,
        "cartesian_error_check_stages": [
            "grasp",
            "lift",
            "above_support",
            "release_hover",
        ],
    }
    assert [phase["support_rim_z_m"] for phase in plan["phases"]] == [
        0.8,
        0.8,
    ]
    assert plan["phases"][1]["grasp_descent_steps"] == 14
    assert plan["phases"][0]["grasp_point_policy"] == (
        "fingertip_fixed_distance_below_cup_top"
    )
    assert plan["phases"][0]["grasp_below_cup_top_m"] == pytest.approx(0.07)
    assert plan["phases"][0]["grasp_height_ratio"] == pytest.approx(0.30)
    first_grasp = next(
        waypoint
        for waypoint in plan["phases"][0]["waypoints"]
        if waypoint["id"] == "grasp"
    )
    assert first_grasp["fingertip_position_base_m"][2] == pytest.approx(0.83)
    first_waypoints = {
        waypoint["id"]: waypoint
        for waypoint in plan["phases"][0]["waypoints"]
    }
    assert plan["phases"][0]["initial_vertical_lift_m"] == pytest.approx(0.05)
    assert plan["phases"][0]["loaded_transfer_policy"] == (
        "obstacle_clearance_multi_via_search"
    )
    assert plan["phases"][0][
        "minimum_loaded_transport_tcp_z_m"
    ] == pytest.approx(1.03)
    assert (
        first_waypoints["lift"]["tcp_position_base_m"][2]
        - first_waypoints["grasp"]["tcp_position_base_m"][2]
    ) == pytest.approx(0.05)
    assert (
        first_waypoints["above_support"]["tcp_position_base_m"][2]
        - first_waypoints["grasp"]["tcp_position_base_m"][2]
    ) == pytest.approx(0.10)
    assert plan["phases"][0]["release_policy"] == (
        "gravity_drop_without_place_descent"
    )
    first_release = plan["phases"][0][
        "release_hover_tcp_position_base_m"
    ]
    second_release = plan["phases"][1][
        "release_hover_tcp_position_base_m"
    ]
    assert first_release[:2] == second_release[:2]
    assert first_release[2] == pytest.approx(0.938)
    assert second_release[2] == pytest.approx(0.948)


def test_generated_code_is_hash_bound_and_ast_restricted() -> None:
    plan = compiled_plan()
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()
    expected = expected_operations(plan, plan_sha256=digest)
    stages = [
        operation.get("stage_id")
        for operation in expected
        if operation["call"] == "execute_cup_stage"
    ]
    assert "release_hover" in stages
    assert "place" not in stages
    code = render(expected)
    assert validate_generated_code(code, expected=expected) == expected
    assert execute_generated_code_dry_run(code, expected=expected) == expected
    with pytest.raises(ValueError, match="forbidden"):
        validate_generated_code(
            code.replace("return_home()", "__import__('os')"),
            expected=expected,
        )


@pytest.mark.parametrize(
    "colors",
    list(itertools.permutations(("orange", "purple", "green"))),
)
def test_general_plan_supports_every_color_permutation(
    colors: tuple[str, str, str],
) -> None:
    plan = compiled_general_plan(colors)
    first_color, second_color, base_color = colors
    assert plan["pipeline"] == "beizi-general"
    assert plan["llm_intent"]["color_order"] == list(colors)
    assert [phase["source"] for phase in plan["phases"]] == [
        f"{first_color} cup",
        f"{second_color} cup",
    ]
    assert [phase["support"] for phase in plan["phases"]] == [
        f"{base_color} cup",
        f"{base_color} cup",
    ]
    expected_support_xy = scene()["cups"][f"{base_color} cup"][
        "rim_center_base_m"
    ][:2]
    expected_source_xy = [
        scene()["cups"][f"{color} cup"]["rim_center_base_m"][:2]
        for color in (first_color, second_color)
    ]
    for index, phase in enumerate(plan["phases"]):
        waypoints = {item["id"]: item for item in phase["waypoints"]}
        assert waypoints["grasp"]["tcp_position_base_m"][:2] == pytest.approx(
            expected_source_xy[index]
        )
        assert waypoints["above_support"][
            "tcp_position_base_m"
        ][:2] == pytest.approx(expected_support_xy)
        assert phase["grasp_below_cup_top_m"] == pytest.approx(0.07)
        assert phase["loaded_transfer_rear_x_offset_m"] == pytest.approx(-0.16)
        assert phase["loaded_transfer_transit_tcp_z_m"] == pytest.approx(1.10)
        assert phase["minimum_loaded_transport_tcp_z_m"] == pytest.approx(0.95)
        assert phase["loaded_transfer_orientation_policy"] == (
            "free_per_waypoint_topdown_at_above_support"
        )
        assert phase["loaded_transfer_route_policy"] == (
            "mandatory_rear_retract_lift_rotate_then_advance"
        )
        assert phase["release_policy"] == (
            "gravity_drop_directly_from_above_support"
        )
        assert phase["release_trigger_policy"] == (
            "xy_aligned_before_above_support"
        )
        assert phase["release_maximum_xy_error_m"] == pytest.approx(0.01)
        assert "saved_stage_joint_positions_rad" not in phase
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()
    expected = expected_operations(plan, plan_sha256=digest)
    code = render(expected)
    assert execute_generated_code_dry_run(code, expected=expected) == expected


@pytest.mark.parametrize(
    "colors",
    [
        ("orange", "purple"),
        ("orange", "orange", "green"),
        ("orange", "purple", "red"),
    ],
)
def test_general_plan_rejects_invalid_color_orders(
    colors: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="colors|three"):
        validate_general_color_order(colors)


def test_adaptive_gripper_close_stops_on_stable_obstruction() -> None:
    script = ROOT / "scripts/r1lite_execute_llm_cup_stack.py"
    spec = importlib.util.spec_from_file_location("cup_executor", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeApi:
        def command_gripper(
            self,
            command: float,
            *,
            duration_s: float,
            allow_contact: bool,
        ) -> float:
            assert duration_s == 0.6
            assert allow_contact is True
            return max(float(command), 40.0)

    result = module.close_gripper_until_contact(
        FakeApi(),
        minimum_command=0.0,
    )
    assert result["contact_detected"] is True
    assert result["hold_command"] == 36.0
    assert result["feedback"] == 40.0


def test_adaptive_gripper_does_not_treat_frozen_feedback_as_contact() -> None:
    module_path = ROOT / "scripts/r1lite_execute_llm_cup_stack.py"
    spec = importlib.util.spec_from_file_location(
        "cup_executor_frozen", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FrozenApi:
        def command_gripper(
            self,
            command: float,
            *,
            duration_s: float,
            allow_contact: bool,
        ) -> float:
            return 60.0

    with pytest.raises(RuntimeError, match="minimum without stable contact"):
        module.close_gripper_until_contact(
            FrozenApi(),
            minimum_command=0.0,
        )


def test_nearest_top_down_preserves_home_jaw_yaw() -> None:
    target, rotation = nearest_top_down_quaternion(
        [0.7, 0.7, 0.1, 0.1]
    )
    assert target.tolist() == pytest.approx(
        [2**-0.5, 2**-0.5, 0.0, 0.0]
    )
    assert rotation < 0.30


def test_fixed_user_scene_disables_runtime_cup_reperception() -> None:
    fixed = scene()
    fixed["pose_source"] = "fixed_user_measurement"
    plan = compile_llm_plan(
        intent(),
        fixed,
        cup_height_m=0.10,
        cup_diameter_m=0.07,
        gripper_close_position=0.0,
    )
    assert plan["cup_pose_source"] == "fixed_user_measurement"
    assert plan["runtime_cup_reperception_enabled"] is False


def test_transit_waypoints_skip_only_cartesian_error_checks() -> None:
    harness = R1LiteIkHarness()
    waypoint = {
        "incremental_joint_delta_rad": 0.01,
        "total_joint_delta_rad": 0.01,
        "fk_position_error_m": 0.20,
        "fk_orientation_error_rad": 0.40,
        "forward_step_m": -0.01,
        "actual_forward_progress_m": 0.0,
        "expected_forward_progress_m": 0.10,
    }
    harness.validate_waypoint(waypoint, check_cartesian_error=False)
    with pytest.raises(RuntimeError, match="FK does not reproduce"):
        harness.validate_waypoint(waypoint, check_cartesian_error=True)


def test_loaded_transfer_routes_include_source_and_outside_rises() -> None:
    module_path = ROOT / "scripts/r1lite_execute_llm_cup_stack.py"
    spec = importlib.util.spec_from_file_location(
        "cup_executor_routes", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    routes = module.build_loaded_transfer_route_candidates(
        [0.369, -0.352, 1.038],
        [0.369, -0.269, 1.138],
        [-0.08, -0.12],
    )
    assert len(routes) == 5
    direct_first_high = next(
        point for point in routes[0] if point[2] == 1.138
    )
    assert direct_first_high[0] == pytest.approx(0.369)
    for route in routes[1:]:
        first_high = next(point for point in route if point[2] == 1.138)
        assert first_high[0] != pytest.approx(0.369)
        assert route[-1][2] == pytest.approx(1.138)
    monitored = module.interpolate_cartesian_segment(
        [0.249, -0.352, 1.138],
        [0.369, -0.269, 1.138],
        maximum_step_m=0.03,
    )
    previous = [0.249, -0.352, 1.138]
    for point in monitored:
        distance = sum((a - b) ** 2 for a, b in zip(point, previous)) ** 0.5
        assert distance <= 0.03 + 1e-9
        previous = point
