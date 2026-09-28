#!/usr/bin/env python3
"""Build a deterministic right-arm config for the three-color cup stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cup-height-m", type=float, required=True)
    parser.add_argument("--cup-diameter-m", type=float, required=True)
    parser.add_argument("--gripper-close-position", type=float, required=True)
    return parser.parse_args()


def render_oracle(
    *,
    cup_height_m: float,
    cup_diameter_m: float,
    gripper_close_position: float,
) -> str:
    return f"""import time
import numpy as np
from capx.integrations.r1lite.cup_stack_plan import build_cup_phase_plan

CUP_HEIGHT_M = {cup_height_m!r}
CUP_DIAMETER_M = {cup_diameter_m!r}
GRIPPER_CLOSE_POSITION = {gripper_close_position!r}


def execute_phase(phase, detections, support_rim_z_m):
    definition = {{
        "orange_on_purple": ("orange cup", "purple cup"),
        "green_on_orange": ("green cup", "orange cup"),
    }}[phase]
    source_name, support_name = definition
    source = np.asarray(detections[source_name]["rim_center_base_m"], dtype=float)
    support = np.asarray(detections[support_name]["rim_center_base_m"], dtype=float)
    plan = build_cup_phase_plan(
        phase=phase,
        source_table_position_base_m=source,
        support_position_base_m=support,
        source_height_m=CUP_HEIGHT_M,
        support_rim_z_m=support_rim_z_m,
        cup_diameter_m=CUP_DIAMETER_M,
        nested_height_increment_m=0.0,
    )
    points = {{
        item["id"]: np.asarray(item["tcp_position_base_m"], dtype=float)
        for item in plan["waypoints"]
    }}
    quat = np.asarray(plan["orientation_quaternion_xyzw"], dtype=float)

    open_gripper()
    goto_pose(points["pregrasp"], quat, gripper=100.0)
    descend_z_in_steps(
        points["grasp"],
        quat,
        start_z=points["pregrasp"][2],
        num_steps=12,
        gripper=100.0,
    )
    close_gripper(position=GRIPPER_CLOSE_POSITION)
    time.sleep(0.8)
    goto_pose(points["lift"], quat, gripper=GRIPPER_CLOSE_POSITION)
    retained, retention = verify_grasp(save_debug=True)
    if not retained:
        raise RuntimeError(f"{{phase}} grasp retention failed: {{retention}}")
    goto_pose(points["above_support"], quat, gripper=GRIPPER_CLOSE_POSITION)
    descend_z_in_steps(
        points["place"],
        quat,
        start_z=points["above_support"][2],
        num_steps=14,
        gripper=GRIPPER_CLOSE_POSITION,
    )
    open_gripper()
    time.sleep(0.8)
    goto_pose(points["retract"], quat, gripper=100.0)
    return plan


save_head_rgb("outputs/r1lite/three_cup_stack_initial.png")
initial = locate_colored_cups_rgbd(
    cup_height_m=CUP_HEIGHT_M,
    save_debug=True,
    require_separation=True,
)
table_z = float(initial["purple cup"]["rim_center_base_m"][2])

orange_plan = execute_phase(
    "orange_on_purple",
    initial,
    support_rim_z_m=table_z,
)

nested = dict(initial)
nested["orange cup"] = dict(initial["orange cup"])
nested["orange cup"]["rim_center_base_m"] = list(
    initial["purple cup"]["rim_center_base_m"]
)
green_plan = execute_phase(
    "green_on_orange",
    nested,
    support_rim_z_m=table_z,
)

success, details = verify_cup_stack(
    expected_top_center_base_m=initial["purple cup"]["top_center_base_m"],
    cup_height_m=CUP_HEIGHT_M,
    nested_occlusion_expected=True,
)
print("verify_cup_stack:", success, details)
if not success:
    raise RuntimeError(f"final cup-stack verification failed: {{details}}")
return_home()
RESULT = {{
    "success": True,
    "details": details,
    "phase_order": ["orange_on_purple", "green_on_orange"],
    "plans": [orange_plan, green_plan],
}}
"""


def build_config(
    source: dict,
    *,
    cup_height_m: float,
    cup_diameter_m: float,
    gripper_close_position: float,
) -> dict:
    if not 0.02 <= cup_height_m <= 0.30:
        raise ValueError("cup height must be between 0.02m and 0.30m")
    if not 0.01 <= cup_diameter_m <= 0.09:
        raise ValueError("cup diameter must be between 0.01m and 0.09m")
    if not 0.0 <= gripper_close_position <= 100.0:
        raise ValueError("gripper close position must be between 0 and 100")

    config = dict(source)
    env = config["env"]["cfg"]
    low = env["low_level"]
    low["arm_target_topic"] = "/motion_target/target_joint_state_arm_right"
    low["gripper_target_topic"] = "/motion_target/target_position_gripper_right"
    low["relaxed_ik_target_topic"] = "/capx/relaxed_ik/target_pose_arm_right"
    low["wrist_camera_enabled"] = False
    low["r1lite_motion_validation_enabled"] = True
    low["r1lite_require_pose_arrival"] = True
    low["r1lite_ik_harness_enabled"] = True
    low["max_joint_step"] = 0.15
    low["goto_pose_joint_waypoint_step_rad"] = 0.15
    low["object_prompts"] = {
        "orange cup": "orange plastic cup",
        "purple cup": "purple plastic cup",
        "green cup": "green plastic cup",
    }
    env["task_only_prompt"] = (
        "Stack the orange cup on the purple cup, then stack the green cup "
        "on the orange cup."
    )
    env["prompt"] = (
        "Execute the deterministic three-color cup stacking policy. "
        "The immutable order is orange on purple, then green on orange."
    )
    env["oracle_code"] = render_oracle(
        cup_height_m=cup_height_m,
        cup_diameter_m=cup_diameter_m,
        gripper_close_position=gripper_close_position,
    )
    config["r1lite_staged_tennis_execution"] = False
    config["r1lite_staged_use_fixed_plan"] = False
    config["use_wrist_camera"] = False
    config["use_oracle_code"] = True
    config["output_dir"] = "./outputs/r1lite_three_cup_stack"
    return config


def main() -> None:
    args = parse_args()
    source = yaml.safe_load(args.source_config.read_text())
    config = build_config(
        source,
        cup_height_m=args.cup_height_m,
        cup_diameter_m=args.cup_diameter_m,
        gripper_close_position=args.gripper_close_position,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
