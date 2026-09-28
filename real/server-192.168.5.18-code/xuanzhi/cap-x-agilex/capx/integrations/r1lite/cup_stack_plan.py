"""Deterministic geometry for the R1 Lite three-color cup stack."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

PHASE_ORDER = ("orange_on_purple", "green_on_orange")
PHASES = {
    "orange_on_purple": {"source": "orange cup", "support": "purple cup"},
    "green_on_orange": {"source": "green cup", "support": "orange cup"},
}


def _vector(value: Sequence[float], *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a finite XYZ vector")
    return result


def build_cup_phase_plan(
    *,
    phase: str,
    source_table_position_base_m: Sequence[float],
    support_position_base_m: Sequence[float],
    source_height_m: float,
    support_rim_z_m: float,
    cup_diameter_m: float,
    nested_height_increment_m: float = 0.0,
    gripper_max_span_m: float = 0.090,
    grasp_height_ratio: float = 0.55,
    tcp_to_fingertip_m: float = 0.078,
    pregrasp_clearance_m: float = 0.100,
    initial_lift_clearance_m: float | None = None,
    lift_clearance_m: float = 0.150,
    orientation_quaternion_xyzw: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build one vertical pick-and-place phase without issuing robot commands."""
    if phase not in PHASES:
        raise ValueError(f"unsupported cup-stack phase {phase!r}")
    source = _vector(source_table_position_base_m, field="source position")
    support = _vector(support_position_base_m, field="support position")
    height = float(source_height_m)
    diameter = float(cup_diameter_m)
    if not 0.02 <= height <= 0.30:
        raise ValueError("cup height must be between 0.02m and 0.30m")
    if not 0.01 <= diameter <= float(gripper_max_span_m):
        raise ValueError("cup diameter exceeds the configured gripper span")
    if not 0.20 <= float(grasp_height_ratio) <= 0.80:
        raise ValueError("grasp height ratio must stay on the cup side wall")
    nesting_increment = float(nested_height_increment_m)
    if not 0.0 <= nesting_increment <= 0.03:
        raise ValueError("nested height increment must be between 0m and 0.03m")
    if abs(float(support_rim_z_m) - support[2]) > 0.03:
        raise ValueError("support rim height is inconsistent with the table plane")
    orientation = np.asarray(
        (
            [1.0, 0.0, 0.0, 0.0]
            if orientation_quaternion_xyzw is None
            else orientation_quaternion_xyzw
        ),
        dtype=float,
    ).reshape(4)
    orientation_norm = float(np.linalg.norm(orientation))
    if not np.isfinite(orientation).all() or orientation_norm < 1e-9:
        raise ValueError("orientation quaternion must be finite and non-zero")
    orientation /= orientation_norm

    jaw_center_offset = height * float(grasp_height_ratio)
    grasp_fingertip = source.copy()
    grasp_fingertip[2] += jaw_center_offset
    place_fingertip = support.copy()
    place_fingertip[2] = (
        float(support_rim_z_m) + nesting_increment + jaw_center_offset
    )
    tcp_offset = np.array([0.0, 0.0, float(tcp_to_fingertip_m)])
    safe_fingertip_z = max(
        float(grasp_fingertip[2]),
        float(place_fingertip[2]),
    ) + float(lift_clearance_m)
    initial_lift_z = float(grasp_fingertip[2]) + float(
        lift_clearance_m
        if initial_lift_clearance_m is None
        else initial_lift_clearance_m
    )
    if initial_lift_z > safe_fingertip_z:
        raise ValueError("initial lift cannot exceed the transfer safe height")

    def waypoint(name: str, fingertip: np.ndarray) -> dict[str, Any]:
        return {
            "id": name,
            "fingertip_position_base_m": fingertip.tolist(),
            "tcp_position_base_m": (fingertip + tcp_offset).tolist(),
        }

    waypoints = [
        waypoint(
            "pregrasp",
            grasp_fingertip + np.array([0.0, 0.0, pregrasp_clearance_m]),
        ),
        waypoint("grasp", grasp_fingertip),
        waypoint(
            "lift",
            np.array(
                [grasp_fingertip[0], grasp_fingertip[1], initial_lift_z]
            ),
        ),
        waypoint(
            "above_support",
            np.array(
                [place_fingertip[0], place_fingertip[1], safe_fingertip_z]
            ),
        ),
        waypoint("place", place_fingertip),
        waypoint(
            "retract",
            place_fingertip + np.array([0.0, 0.0, pregrasp_clearance_m]),
        ),
    ]
    plan = {
        "schema": "capx.r1lite.three_cup_stack.phase.v3",
        "phase": phase,
        **PHASES[phase],
        "source_height_m": height,
        "cup_diameter_m": diameter,
        "support_rim_z_m": float(support_rim_z_m),
        "nested_height_increment_m": nesting_increment,
        "stacking_mode": "same_orientation_nested_inverted_cups",
        "orientation_quaternion_xyzw": orientation.tolist(),
        "motion_policy": "short_vertical_lift_then_diagonal_safe_z_transfer",
        "waypoints": waypoints,
    }
    validate_cup_phase_plan(plan)
    return plan


def validate_cup_phase_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != "capx.r1lite.three_cup_stack.phase.v3":
        raise ValueError("unexpected cup-stack phase schema")
    phase = str(plan.get("phase"))
    if phase not in PHASES:
        raise ValueError("unknown cup-stack phase")
    if {
        "source": plan.get("source"),
        "support": plan.get("support"),
    } != PHASES[phase]:
        raise ValueError("cup-stack source/support order changed")
    orientation = np.asarray(
        plan.get("orientation_quaternion_xyzw"),
        dtype=float,
    )
    if (
        orientation.shape != (4,)
        or not np.isfinite(orientation).all()
        or not np.isclose(np.linalg.norm(orientation), 1.0, atol=1e-6)
    ):
        raise ValueError("cup-stack orientation quaternion must be normalized")
    waypoints = plan.get("waypoints")
    expected = (
        "pregrasp",
        "grasp",
        "lift",
        "above_support",
        "place",
        "retract",
    )
    if not isinstance(waypoints, list) or tuple(
        item.get("id") for item in waypoints
    ) != expected:
        raise ValueError("cup-stack waypoint order is immutable")
    values = {
        item["id"]: _vector(
            item["tcp_position_base_m"],
            field=f"{item['id']} TCP",
        )
        for item in waypoints
    }
    if not np.allclose(values["pregrasp"][:2], values["grasp"][:2]):
        raise ValueError("pregrasp descent must keep source XY fixed")
    if not np.allclose(values["above_support"][:2], values["place"][:2]):
        raise ValueError("placement descent must keep support XY fixed")
    if values["pregrasp"][2] <= values["grasp"][2]:
        raise ValueError("pregrasp must be above grasp")
    if values["lift"][2] <= values["grasp"][2]:
        raise ValueError("initial lift must be above grasp")
    if values["above_support"][2] <= values["place"][2]:
        raise ValueError("above-support waypoint must be above place")
    if values["lift"][2] > values["above_support"][2]:
        raise ValueError("loaded transfer cannot descend before above-support")
