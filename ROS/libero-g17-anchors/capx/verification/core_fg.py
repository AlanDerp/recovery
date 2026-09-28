"""Deterministic, sanitized factual grounding for cube placement tasks.

The extractor intentionally reports only current-state predicates. It does not
infer task stage, diagnose failure, recommend an action, or expose coordinates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "core-fg-cube-v2"
PUBLIC_HEADER = "GT_BACKED_FACTUAL_CONTEXT:"
PUBLIC_BRIDGE = "These are current GT-backed facts; treat them as factual grounding."


@dataclass(frozen=True)
class CoreFacts:
    target_grasp: str
    relative_x: str
    relative_y: str
    relative_z: str
    footprint: str
    target_goal_contact: str
    target_table_contact: str


def _rotation_matrix_wxyz(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("Goal quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Goal quaternion has zero norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_relation(delta: float, half_extent: float, axis: str) -> str:
    if not np.isfinite(delta) or not np.isfinite(half_extent) or half_extent <= 0:
        raise ValueError(f"Invalid {axis}-axis relation inputs")
    scale = max(1.0, abs(float(delta)), abs(float(half_extent)))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    if delta < -half_extent - tolerance:
        return f"target_center_negative_{axis}_of_goal_band"
    if delta > half_extent + tolerance:
        return f"target_center_positive_{axis}_of_goal_band"
    return f"target_center_within_goal_{axis}_band"


def _height_relation(delta_z: float, target_half_height: float, goal_half_height: float) -> str:
    band = max(float(target_half_height), float(goal_half_height))
    if not np.isfinite(delta_z) or not np.isfinite(band) or band <= 0:
        raise ValueError("Invalid height relation inputs")
    if delta_z < -band:
        return "target_below_goal_height_band"
    if delta_z > band:
        return "target_above_goal_height_band"
    return "target_within_goal_height_band"


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Return the counter-clockwise convex hull of finite 2D points."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Projected footprint points must be finite 2D values")
    unique = sorted({(float(point[0]), float(point[1])) for point in values})
    if len(unique) < 3:
        raise ValueError("Projected footprint must contain at least three unique points")

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] < 3:
        raise ValueError("Projected footprint hull is degenerate")
    return hull


def _point_in_closed_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Test point inclusion in a counter-clockwise convex polygon."""
    point_value = np.asarray(point, dtype=np.float64)
    polygon_value = np.asarray(polygon, dtype=np.float64)
    if point_value.shape != (2,) or not np.all(np.isfinite(point_value)):
        raise ValueError("Footprint query point must contain two finite values")
    if (
        polygon_value.ndim != 2
        or polygon_value.shape[0] < 3
        or polygon_value.shape[1] != 2
        or not np.all(np.isfinite(polygon_value))
    ):
        raise ValueError("Footprint polygon must contain finite 2D vertices")

    scale = max(1.0, float(np.max(np.abs(polygon_value))), float(np.max(np.abs(point_value))))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale * scale
    for start, end in zip(polygon_value, np.roll(polygon_value, -1, axis=0)):
        edge = end - start
        relative = point_value - start
        if edge[0] * relative[1] - edge[1] * relative[0] < -tolerance:
            return False
    return True


def _horizontal_goal_projection(
    goal_rotation_task: np.ndarray,
    goal_size: np.ndarray,
    delta_xy_task: np.ndarray,
) -> tuple[float, float, bool]:
    """Project the goal OBB onto the task-frame horizontal plane."""
    rotation = np.asarray(goal_rotation_task, dtype=np.float64)
    size = np.asarray(goal_size, dtype=np.float64)
    delta_xy = np.asarray(delta_xy_task, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Goal rotation must be a finite 3x3 matrix")
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("Goal half-extents must be finite and positive")
    if delta_xy.shape != (2,) or not np.all(np.isfinite(delta_xy)):
        raise ValueError("Task-frame horizontal displacement must be finite")

    signs = np.asarray(
        [
            [x_sign, y_sign, z_sign]
            for x_sign in (-1.0, 1.0)
            for y_sign in (-1.0, 1.0)
            for z_sign in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    projected = ((signs * size) @ rotation.T)[:, :2]
    hull = _convex_hull_2d(projected)
    half_extent_x = float(np.max(np.abs(projected[:, 0])))
    half_extent_y = float(np.max(np.abs(projected[:, 1])))
    inside = _point_in_closed_convex_polygon(delta_xy, hull)
    return half_extent_x, half_extent_y, inside


def _unwrap_cube_environment(env: Any) -> tuple[Any, Any]:
    low_level = getattr(env, "low_level_env", None)
    simulator = getattr(low_level, "robosuite_env", None)
    required = ("cubeA", "cubeB", "cubeA_body_id", "cubeB_body_id", "robots", "sim")
    if low_level is None or simulator is None or not all(
        hasattr(simulator, name) for name in required
    ):
        raise TypeError(
            "Core-FG cube adapter requires a CaP-X Robosuite Stack low-level environment"
        )
    return low_level, simulator


def extract_core_facts(env: Any) -> CoreFacts:
    """Project the current simulator state into a closed sanitized vocabulary."""
    low_level, simulator = _unwrap_cube_environment(env)
    sim_data = simulator.sim.data

    target_world = np.asarray(
        sim_data.body_xpos[simulator.cubeA_body_id], dtype=np.float64
    ).copy()
    goal_world = np.asarray(
        sim_data.body_xpos[simulator.cubeB_body_id], dtype=np.float64
    ).copy()
    goal_quat_wxyz = np.asarray(
        sim_data.body_xquat[simulator.cubeB_body_id], dtype=np.float64
    ).copy()
    if target_world.shape != (3,) or goal_world.shape != (3,):
        raise ValueError("Cube centers must each contain three values")
    if not np.all(np.isfinite(target_world)) or not np.all(np.isfinite(goal_world)):
        raise ValueError("Cube centers must be finite")

    target_size = np.asarray(simulator.cubeA.size, dtype=np.float64)
    goal_size = np.asarray(simulator.cubeB.size, dtype=np.float64)
    if (
        target_size.shape != (3,)
        or goal_size.shape != (3,)
        or not np.all(np.isfinite(target_size))
        or not np.all(np.isfinite(goal_size))
        or np.any(target_size <= 0)
        or np.any(goal_size <= 0)
    ):
        raise ValueError("Cube half-extents must be finite and positive")

    # Match the task API's robot-base frame for directional axis predicates.
    pose_input = {
        "cubeA_pos": target_world,
        "cubeB_pos": goal_world,
        # Robosuite's cube pose sensor exposes XYZW.
        "cubeA_quat": np.roll(
            np.asarray(sim_data.body_xquat[simulator.cubeA_body_id], dtype=np.float64),
            -1,
        ),
        "cubeB_quat": np.roll(goal_quat_wxyz, -1),
    }
    poses = low_level._cube_pose_dict(pose_input)
    target_task = np.asarray(poses["primary"][:3], dtype=np.float64)
    goal_task = np.asarray(poses["secondary"][:3], dtype=np.float64)
    delta_task = target_task - goal_task
    if not np.all(np.isfinite(delta_task)):
        raise ValueError("Task-frame cube displacement must be finite")

    goal_rotation_task = _rotation_matrix_wxyz(poses["secondary"][3:7])
    goal_band_x, goal_band_y, inside_footprint = _horizontal_goal_projection(
        goal_rotation_task,
        goal_size,
        delta_task[:2],
    )
    relative_x = _axis_relation(float(delta_task[0]), goal_band_x, "x")
    relative_y = _axis_relation(float(delta_task[1]), goal_band_y, "y")
    if inside_footprint and (
        relative_x != "target_center_within_goal_x_band"
        or relative_y != "target_center_within_goal_y_band"
    ):
        raise RuntimeError("Horizontal footprint and axis predicates are inconsistent")

    grasped = bool(
        simulator._check_grasp(
            gripper=simulator.robots[0].gripper,
            object_geoms=simulator.cubeA,
        )
    )
    goal_contact = bool(simulator.check_contact(simulator.cubeA, simulator.cubeB))
    table_contact = bool(simulator.check_contact(simulator.cubeA, "table_collision"))

    return CoreFacts(
        target_grasp=(
            "target_grasped_by_robot" if grasped else "target_not_grasped_by_robot"
        ),
        relative_x=relative_x,
        relative_y=relative_y,
        relative_z=_height_relation(
            float(delta_task[2]), float(target_size[2]), float(goal_size[2])
        ),
        footprint=(
            "target_center_inside_goal_footprint"
            if inside_footprint
            else "target_center_outside_goal_footprint"
        ),
        target_goal_contact=(
            "target_contacting_goal" if goal_contact else "target_separated_from_goal"
        ),
        target_table_contact=(
            "target_contacting_table" if table_contact else "target_separated_from_table"
        ),
    )


def format_public_facts(facts: CoreFacts) -> str:
    """Render a fixed-order, coordinate-free public packet."""
    return "\n".join(
        (
            PUBLIC_HEADER,
            PUBLIC_BRIDGE,
            f"TARGET_GRASP: {facts.target_grasp}",
            f"RELATIVE_X: {facts.relative_x}",
            f"RELATIVE_Y: {facts.relative_y}",
            f"RELATIVE_Z: {facts.relative_z}",
            f"FOOTPRINT: {facts.footprint}",
            f"TARGET_GOAL_CONTACT: {facts.target_goal_contact}",
            f"TARGET_TABLE_CONTACT: {facts.target_table_contact}",
        )
    )


def build_and_record_core_facts(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    turn: int,
) -> str:
    """Extract public facts and persist their enum-only derivation record."""
    facts = extract_core_facts(env)
    public_text = format_public_facts(facts)
    trace_dir = Path(output_dir) / "core_fg_traces" / f"trial_{trial:02d}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "trial": int(trial),
        "turn": int(turn),
        "facts": asdict(facts),
        "public_text": public_text,
        "contains_coordinates": False,
        "contains_reward_or_success": False,
        "contains_action_advice": False,
    }
    (trace_dir / f"turn_{turn:02d}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return public_text
