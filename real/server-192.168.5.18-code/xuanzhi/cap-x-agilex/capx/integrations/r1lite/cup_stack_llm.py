"""Validated LLM intent and generated-code contract for three-cup stacking."""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable
from typing import Any, Mapping

import numpy as np

from capx.integrations.r1lite.cup_stack_plan import (
    PHASES,
    PHASE_ORDER,
    build_cup_phase_plan,
)

INTENT_SCHEMA = "capx.r1lite.three_cup_stack.llm_intent.v2"
GENERAL_INTENT_SCHEMA = "capx.r1lite.beizi_general.intent.v1"
PLAN_SCHEMA = "capx.r1lite.three_cup_stack.llm_plan.v12"
SUPPORTED_CUP_COLORS = ("orange", "purple", "green")
GRASP_BELOW_CUP_TOP_M = 0.07
INITIAL_VERTICAL_LIFT_M = 0.05
HIGH_TRANSFER_CLEARANCE_M = 0.10
MINIMUM_LOADED_TRANSPORT_TCP_Z_M = 1.03
ALLOWED_CALLS = {
    "bind_cup_plan",
    "open_gripper",
    "execute_cup_stage",
    "close_gripper",
    "verify_grasp",
    "verify_cup_stack",
    "return_home",
}


def nearest_top_down_quaternion(
    home_quaternion_xyzw: Any,
) -> tuple[np.ndarray, float]:
    """Return the top-down tool pose with minimum rotation from Home."""
    home = np.asarray(home_quaternion_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(home))
    if not np.isfinite(home).all() or norm < 1e-9:
        raise ValueError("Home gripper orientation quaternion is invalid")
    home /= norm
    # Every normalized [x, y, 0, 0] quaternion maps tool +Z to base -Z.
    # Projecting Home onto this family preserves the nearest in-plane jaw yaw.
    target = np.array([home[0], home[1], 0.0, 0.0], dtype=float)
    target_norm = float(np.linalg.norm(target))
    if target_norm < 1e-9:
        target = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        target /= target_norm
    rotation = float(
        2.0
        * np.arccos(
            np.clip(abs(np.dot(home, target)), 0.0, 1.0)
        )
    )
    return target, rotation


def validate_llm_intent(intent: Mapping[str, Any]) -> None:
    if intent.get("schema") != INTENT_SCHEMA:
        raise ValueError("unexpected LLM cup-stack intent schema")
    if tuple(intent.get("phase_order", ())) != PHASE_ORDER:
        raise ValueError("LLM changed the immutable cup-stack phase order")
    phases = intent.get("phases")
    if not isinstance(phases, list) or len(phases) != len(PHASE_ORDER):
        raise ValueError("LLM intent must contain exactly two phases")
    grasp_ratios = []
    for expected_phase, phase in zip(PHASE_ORDER, phases, strict=True):
        if phase.get("id") != expected_phase:
            raise ValueError("LLM cup-stack phases are missing or out of order")
        definition = PHASES[expected_phase]
        if phase.get("source") != definition["source"]:
            raise ValueError("LLM changed a cup-stack source")
        if phase.get("support") != definition["support"]:
            raise ValueError("LLM changed a cup-stack support")
        ratio = float(phase.get("grasp_height_ratio"))
        pregrasp = float(phase.get("pregrasp_clearance_m"))
        lift = float(phase.get("lift_clearance_m"))
        grasp_steps = int(phase.get("grasp_descent_steps"))
        release_clearance = float(phase.get("release_clearance_m"))
        if not 0.20 <= ratio <= 0.70:
            raise ValueError("LLM grasp height ratio is outside [0.20, 0.70]")
        if not 0.08 <= pregrasp <= 0.15:
            raise ValueError("LLM pregrasp clearance is outside [0.08m, 0.15m]")
        if not 0.12 <= lift <= 0.25:
            raise ValueError("LLM lift clearance is outside [0.12m, 0.25m]")
        if not 10 <= grasp_steps <= 20:
            raise ValueError("LLM grasp descent steps are outside [10, 20]")
        if not 0.02 <= release_clearance <= 0.06:
            raise ValueError(
                "LLM gravity-release clearance is outside [0.02m, 0.06m]"
            )
        if float(phase.get("nested_height_increment_m", 0.0)) != 0.0:
            raise ValueError("nested inverted cups must keep the same rim height")
        grasp_ratios.append(ratio)
    if not np.allclose(grasp_ratios, grasp_ratios[0]):
        raise ValueError(
            "identical cups require one shared grasp height ratio"
        )


def compile_llm_plan(
    intent: Mapping[str, Any],
    scene: Mapping[str, Any],
    *,
    cup_height_m: float,
    cup_diameter_m: float,
    gripper_close_position: float,
) -> dict[str, Any]:
    """Compile bounded LLM decisions into deterministic Cartesian geometry."""
    validate_llm_intent(intent)
    cups = scene.get("cups")
    if not isinstance(cups, Mapping) or set(cups) != {
        "orange cup",
        "purple cup",
        "green cup",
    }:
        raise ValueError("RGB-D scene must contain exactly three colored cups")
    home_orientation = np.asarray(
        scene.get("home_gripper_quaternion_xyzw"),
        dtype=float,
    )
    if home_orientation.shape != (4,) or not np.isfinite(
        home_orientation
    ).all():
        raise ValueError("RGB-D scene is missing the Home gripper orientation")
    orientation_norm = float(np.linalg.norm(home_orientation))
    if orientation_norm < 1e-9:
        raise ValueError("Home gripper orientation quaternion is zero")
    home_orientation /= orientation_norm
    grasp_orientation, home_to_top_down = nearest_top_down_quaternion(
        home_orientation
    )
    if home_to_top_down > np.deg2rad(100.0):
        raise ValueError(
            "Home orientation needs more than 100 degrees to become top-down"
        )
    table_z = float(
        np.median(
            [
                np.asarray(item["rim_center_base_m"], dtype=float)[2]
                for item in cups.values()
            ]
        )
    )
    cup_height = float(cup_height_m)
    if cup_height <= GRASP_BELOW_CUP_TOP_M:
        raise ValueError("cup height must exceed the grasp top offset")
    fixed_grasp_height_ratio = (
        cup_height - GRASP_BELOW_CUP_TOP_M
    ) / cup_height
    compiled_phases = []
    for phase_intent in intent["phases"]:
        phase_id = str(phase_intent["id"])
        definition = PHASES[phase_id]
        source = np.asarray(
            cups[definition["source"]]["rim_center_base_m"],
            dtype=float,
        )
        support = np.asarray(
            cups[definition["support"]]["rim_center_base_m"],
            dtype=float,
        )
        if phase_id == "green_on_orange":
            support = np.asarray(
                cups["purple cup"]["rim_center_base_m"],
                dtype=float,
            )
        support[2] = table_z
        phase_plan = build_cup_phase_plan(
            phase=phase_id,
            source_table_position_base_m=source,
            support_position_base_m=support,
            source_height_m=float(cup_height_m),
            support_rim_z_m=table_z,
            cup_diameter_m=float(cup_diameter_m),
            nested_height_increment_m=0.0,
            grasp_height_ratio=fixed_grasp_height_ratio,
            pregrasp_clearance_m=float(
                phase_intent["pregrasp_clearance_m"]
            ),
            initial_lift_clearance_m=INITIAL_VERTICAL_LIFT_M,
            lift_clearance_m=HIGH_TRANSFER_CLEARANCE_M,
            orientation_quaternion_xyzw=grasp_orientation,
        )
        phase_plan["grasp_descent_steps"] = int(
            phase_intent["grasp_descent_steps"]
        )
        phase_plan["grasp_point_policy"] = (
            "fingertip_fixed_distance_below_cup_top"
        )
        phase_plan["grasp_below_cup_top_m"] = GRASP_BELOW_CUP_TOP_M
        phase_plan["grasp_height_ratio"] = fixed_grasp_height_ratio
        phase_plan["llm_requested_grasp_height_ratio"] = float(
            phase_intent["grasp_height_ratio"]
        )
        phase_plan["initial_vertical_lift_m"] = INITIAL_VERTICAL_LIFT_M
        phase_plan["high_transfer_clearance_m"] = HIGH_TRANSFER_CLEARANCE_M
        phase_plan["llm_requested_lift_clearance_m"] = float(
            phase_intent["lift_clearance_m"]
        )
        phase_plan["transfer_height_policy"] = (
            "short_vertical_lift_then_obstacle_clearance_route_search"
        )
        place_tcp = next(
            np.asarray(item["tcp_position_base_m"], dtype=float)
            for item in phase_plan["waypoints"]
            if item["id"] == "place"
        )
        release_clearance = float(phase_intent["release_clearance_m"])
        release_hover_tcp = place_tcp + np.array(
            [0.0, 0.0, release_clearance]
        )
        phase_plan["release_clearance_m"] = release_clearance
        phase_plan["release_hover_tcp_position_base_m"] = (
            release_hover_tcp.tolist()
        )
        phase_plan["release_policy"] = "gravity_drop_without_place_descent"
        above_support_tcp = next(
            np.asarray(item["tcp_position_base_m"], dtype=float)
            for item in phase_plan["waypoints"]
            if item["id"] == "above_support"
        )
        phase_plan["loaded_transfer_policy"] = (
            "obstacle_clearance_multi_via_search"
        )
        phase_plan["loaded_transfer_corridor_x_offsets_m"] = [
            -0.08,
            -0.12,
            -0.16,
            0.08,
            0.12,
        ]
        phase_plan["minimum_loaded_transport_tcp_z_m"] = float(
            MINIMUM_LOADED_TRANSPORT_TCP_Z_M
        )
        compiled_phases.append(phase_plan)
    home_position = np.asarray(
        scene.get("home_gripper_position_base_m"),
        dtype=float,
    ).reshape(3)
    first_pregrasp = np.asarray(
        compiled_phases[0]["waypoints"][0]["tcp_position_base_m"],
        dtype=float,
    )
    expected_initial_waypoints = max(
        1,
        int(np.ceil(np.linalg.norm(first_pregrasp - home_position) / 0.06)),
        int(np.ceil(home_to_top_down / 0.25)),
    )
    return {
        "schema": PLAN_SCHEMA,
        "phase_order": list(PHASE_ORDER),
        "cup_pose_source": scene.get("pose_source", "live_rgbd"),
        "runtime_cup_reperception_enabled": (
            scene.get("pose_source") != "fixed_user_measurement"
        ),
        "llm_intent": dict(intent),
        "rgbd_scene": dict(scene),
        "cup_height_m": float(cup_height_m),
        "cup_diameter_m": float(cup_diameter_m),
        "gripper_open_position": 60.0,
        "gripper_close_position": float(gripper_close_position),
        "table_rim_z_m": table_z,
        "home_gripper_quaternion_xyzw": home_orientation.tolist(),
        "grasp_quaternion_xyzw": grasp_orientation.tolist(),
        "home_to_top_down_rotation_rad": home_to_top_down,
        "orientation_policy": "nearest_top_down_orientation_to_home",
        "ik_path_limits": {
            "max_cartesian_step_m": 0.06,
            "max_orientation_step_rad": 0.25,
            "expected_home_to_pregrasp_waypoints": expected_initial_waypoints,
            "low_level_joint_interpolation_retained": True,
            "cartesian_error_check_stages": [
                "grasp",
                "lift",
                "above_support",
                "release_hover",
            ],
        },
        "phases": compiled_phases,
        "failure_policy": "hold_and_abort",
    }


def validate_general_color_order(color_order: Any) -> tuple[str, str, str]:
    """Validate and normalize the three positional beizi-general colors."""
    if isinstance(color_order, str):
        colors = tuple(
            item.strip().lower()
            for item in color_order.split(",")
            if item.strip()
        )
    else:
        colors = tuple(str(item).strip().lower() for item in color_order)
    if len(colors) != 3:
        raise ValueError("beizi-general requires exactly three colors")
    if set(colors) != set(SUPPORTED_CUP_COLORS):
        raise ValueError(
            "colors must be one unique permutation of orange, purple, green"
        )
    return colors


def apply_beizi2_green_motion_policy(
    phase: dict[str, Any],
    *,
    minimum_transport_z_m: float = 0.95,
) -> None:
    """Apply the current beizi-2 green transport and release policy."""
    minimum_z = float(minimum_transport_z_m)
    if minimum_z < 0.95:
        raise ValueError("loaded transport minimum Z must be at least 0.95m")
    phase.pop("saved_stage_joint_positions_rad", None)
    phase.pop("saved_stage_joint_position_policy", None)
    phase.pop("saved_stage_joint_position_sources", None)
    phase["minimum_loaded_transport_tcp_z_m"] = minimum_z
    phase["above_support_joint_policy"] = (
        "regenerate_with_online_ik_each_execution"
    )
    phase["loaded_transfer_orientation_policy"] = (
        "free_per_waypoint_topdown_at_above_support"
    )
    phase["loaded_transfer_route_policy"] = (
        "mandatory_rear_retract_lift_rotate_then_advance"
    )
    phase["loaded_transfer_rear_x_offset_m"] = -0.16
    phase["loaded_transfer_transit_tcp_z_m"] = 1.10
    phase["release_policy"] = "gravity_drop_directly_from_above_support"
    phase["release_trigger_policy"] = "xy_aligned_before_above_support"
    phase["release_maximum_xy_error_m"] = 0.01


def compile_general_plan(
    color_order: Any,
    scene: Mapping[str, Any],
    *,
    cup_height_m: float,
    cup_diameter_m: float,
    gripper_close_position: float,
) -> dict[str, Any]:
    """Compile c1->c3 then c2->c3 using fixed saved cup positions."""
    first_color, second_color, base_color = validate_general_color_order(
        color_order
    )
    cups = scene.get("cups")
    if not isinstance(cups, Mapping):
        raise ValueError("fixed scene is missing colored cup positions")
    required_keys = {f"{color} cup" for color in SUPPORTED_CUP_COLORS}
    if set(cups) != required_keys:
        raise ValueError("fixed scene must contain orange, purple, and green cups")

    # Reuse the proven deterministic compiler by mapping the requested source
    # colors onto its canonical c1/c2/base slots. Geometry is then relabeled.
    canonical_scene = copy.deepcopy(dict(scene))
    canonical_scene["cups"] = {
        "orange cup": copy.deepcopy(cups[f"{first_color} cup"]),
        "purple cup": copy.deepcopy(cups[f"{base_color} cup"]),
        "green cup": copy.deepcopy(cups[f"{second_color} cup"]),
    }
    canonical_intent = {
        "schema": INTENT_SCHEMA,
        "phase_order": list(PHASE_ORDER),
        "phases": [
            {
                "id": "orange_on_purple",
                "source": "orange cup",
                "support": "purple cup",
                "grasp_height_ratio": 0.55,
                "pregrasp_clearance_m": 0.11,
                "lift_clearance_m": 0.16,
                "grasp_descent_steps": 14,
                "release_clearance_m": 0.04,
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
    plan = compile_llm_plan(
        canonical_intent,
        canonical_scene,
        cup_height_m=cup_height_m,
        cup_diameter_m=cup_diameter_m,
        gripper_close_position=gripper_close_position,
    )
    phase_ids = (
        f"phase_1_{first_color}_on_{base_color}",
        f"phase_2_{second_color}_on_{base_color}",
    )
    sources = (first_color, second_color)
    for phase, phase_id, source_color in zip(
        plan["phases"], phase_ids, sources, strict=True
    ):
        phase["phase"] = phase_id
        phase["source"] = f"{source_color} cup"
        phase["support"] = f"{base_color} cup"
        apply_beizi2_green_motion_policy(phase)
    general_intent = {
        "schema": GENERAL_INTENT_SCHEMA,
        "color_order": [first_color, second_color, base_color],
        "phase_order": list(phase_ids),
        "phases": [
            {
                "id": phase_id,
                "source": f"{source_color} cup",
                "support": f"{base_color} cup",
            }
            for phase_id, source_color in zip(
                phase_ids, sources, strict=True
            )
        ],
    }
    plan["phase_order"] = list(phase_ids)
    plan["llm_intent"] = general_intent
    plan["rgbd_scene"] = copy.deepcopy(dict(scene))
    plan["cup_pose_source"] = scene.get(
        "pose_source", "fixed_user_measurement"
    )
    plan["runtime_cup_reperception_enabled"] = False
    plan["pipeline"] = "beizi-general"
    plan["verification"] = {
        "base_cup": f"{base_color} cup",
        "top_cup": f"{second_color} cup",
        "expected_top_center_base_m": copy.deepcopy(
            cups[f"{base_color} cup"]["top_center_base_m"]
        ),
    }
    return plan


def expected_operations(
    plan: Mapping[str, Any],
    *,
    plan_sha256: str,
) -> list[dict[str, Any]]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unexpected compiled LLM cup-stack plan schema")
    phase_order = tuple(str(item) for item in plan.get("phase_order", ()))
    phases = plan.get("phases")
    if len(phase_order) != 2 or len(set(phase_order)) != 2:
        raise ValueError("compiled cup-stack plan must contain two unique phases")
    if not isinstance(phases, list) or tuple(
        str(phase.get("phase")) for phase in phases
    ) != phase_order:
        raise ValueError("compiled cup-stack phase order is inconsistent")
    operations: list[dict[str, Any]] = [
        {"call": "bind_cup_plan", "sha256": plan_sha256}
    ]
    for phase_id in phase_order:
        operations.append({"call": "open_gripper"})
        for stage_id in ("pregrasp", "grasp"):
            operations.append(
                {
                    "call": "execute_cup_stage",
                    "phase": phase_id,
                    "stage_id": stage_id,
                }
            )
        operations.append({"call": "close_gripper"})
        operations.append(
            {
                "call": "execute_cup_stage",
                "phase": phase_id,
                "stage_id": "lift",
            }
        )
        operations.append({"call": "verify_grasp"})
        for stage_id in ("above_support", "release_hover"):
            operations.append(
                {
                    "call": "execute_cup_stage",
                    "phase": phase_id,
                    "stage_id": stage_id,
                }
            )
        operations.append({"call": "open_gripper"})
        operations.append(
            {
                "call": "execute_cup_stage",
                "phase": phase_id,
                "stage_id": "retract",
            }
        )
    operations.extend(
        [
            {"call": "verify_cup_stack"},
            {"call": "return_home"},
        ]
    )
    return operations


def _literal_keywords(call: ast.Call) -> dict[str, Any]:
    if call.args:
        raise ValueError("LLM-generated code cannot use positional arguments")
    result: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in result:
            raise ValueError("invalid LLM-generated keyword arguments")
        result[keyword.arg] = ast.literal_eval(keyword.value)
    return result


def validate_generated_code(
    code: str,
    *,
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != len(expected):
        raise ValueError("LLM code does not contain the exact required calls")
    observed = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            raise ValueError("LLM code may contain only call expressions")
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id not in ALLOWED_CALLS:
            raise ValueError("LLM code contains a forbidden function call")
        observed.append(
            {"call": call.func.id, **_literal_keywords(call)}
        )
    if observed != expected:
        raise ValueError("LLM code differs from the hash-bound safe program")
    return observed


def execute_generated_code_dry_run(
    code: str,
    *,
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validate_generated_code(code, expected=expected)
    trace: list[dict[str, Any]] = []

    def recorder(name: str) -> Callable[..., None]:
        def record(**kwargs: Any) -> None:
            trace.append({"call": name, **kwargs})

        return record

    namespace = {name: recorder(name) for name in ALLOWED_CALLS}
    exec(
        compile(code, "<r1lite-cup-stack-llm-code>", "exec"),
        {"__builtins__": {}, **namespace},
        {},
    )
    if trace != expected:
        raise RuntimeError("LLM-generated cup-stack dry-run trace mismatch")
    return trace
