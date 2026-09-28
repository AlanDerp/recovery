#!/usr/bin/env python3
"""Use RGB-D and an LLM to plan the bounded three-cup stacking task."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel  # noqa: E402
from capx.integrations.r1lite.control import R1LiteControlApi  # noqa: E402
from capx.integrations.r1lite.cup_stack_llm import (  # noqa: E402
    GRASP_BELOW_CUP_TOP_M,
    INTENT_SCHEMA,
    compile_llm_plan,
    validate_llm_intent,
)
from capx.llm.client import ModelQueryArgs, query_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8110/chat/completions",
    )
    parser.add_argument("--cup-height-m", type=float, required=True)
    parser.add_argument("--cup-diameter-m", type=float, required=True)
    parser.add_argument("--gripper-close-position", type=float, required=True)
    parser.add_argument("--fixed-scene", type=Path)
    return parser.parse_args()


def extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        begin, end = text.find("{"), text.rfind("}")
        if begin < 0 or end <= begin:
            raise ValueError("LLM response contains no JSON object")
        result = json.loads(text[begin : end + 1])
    if not isinstance(result, dict):
        raise ValueError("LLM cup-stack intent must be a JSON object")
    return result


def load_robot(config_path: Path) -> R1LiteRealLowLevel:
    config = yaml.safe_load(config_path.read_text())
    low_config = dict(config["env"]["cfg"]["low_level"])
    low_config.pop("_target_", None)
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {
        name: value
        for name, value in low_config.items()
        if name in signature.parameters
    }
    kwargs["wrist_camera_enabled"] = False
    return R1LiteRealLowLevel(**kwargs)


def main() -> None:
    args = parse_args()
    api_key = args.api_key_file.read_text().strip()
    if not api_key:
        raise RuntimeError("LLM API key file is empty")
    robot = load_robot(args.config)
    try:
        api = R1LiteControlApi(robot)
        current_joints = robot.get_arm_joints()
        if current_joints.shape != (6,) or not np.isfinite(
            current_joints
        ).all():
            raise RuntimeError("right-arm Home feedback is invalid")
        home_position, home_orientation = robot.get_planning_ee_pose()
        home_torso_joints = robot.get_torso_joints()
        if args.fixed_scene is None:
            cups = api.locate_colored_cups_rgbd(
                cup_height_m=args.cup_height_m,
                save_debug=True,
                require_separation=True,
            )
            scene_source = {
                "schema": "capx.r1lite.three_cup_rgbd_scene.v1",
                "frame_id": "base_link",
                "pose_source": "live_rgbd",
                "cups": cups,
            }
        else:
            scene_source = json.loads(args.fixed_scene.read_text())
            if scene_source.get("pose_source") != "fixed_user_measurement":
                raise ValueError("fixed scene must be a user-measurement scene")
            cups = scene_source["cups"]
    finally:
        robot.close()
    scene = {
        **scene_source,
        "home_gripper_position_base_m": home_position.tolist(),
        "home_gripper_quaternion_xyzw": home_orientation.tolist(),
        "home_arm_joints_rad": current_joints.tolist(),
        "home_torso_joints_rad": home_torso_joints.tolist(),
        "robot_motion_commands_sent": False,
    }
    compact_scene = {
        name: {
            "top_center_base_m": value["top_center_base_m"],
            "body_center_base_m": value["body_center_base_m"],
            "rim_center_base_m": value["rim_center_base_m"],
        }
        for name, value in cups.items()
    }
    grasp_below_cup_top_m = GRASP_BELOW_CUP_TOP_M
    if float(args.cup_height_m) <= grasp_below_cup_top_m:
        raise ValueError(
            "cup height must exceed the configured grasp top offset"
        )
    fixed_grasp_height_ratio = (
        float(args.cup_height_m) - grasp_below_cup_top_m
    ) / float(args.cup_height_m)
    reference = {
        "schema": INTENT_SCHEMA,
        "phase_order": ["orange_on_purple", "green_on_orange"],
        "phases": [
            {
                "id": "orange_on_purple",
                "source": "orange cup",
                "support": "purple cup",
                "grasp_height_ratio": fixed_grasp_height_ratio,
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
                "grasp_height_ratio": fixed_grasp_height_ratio,
                "pregrasp_clearance_m": 0.10,
                "lift_clearance_m": 0.15,
                "grasp_descent_steps": 12,
                "release_clearance_m": 0.03,
                "nested_height_increment_m": 0.0,
            },
        ],
        "rationale": "short safety explanation",
    }
    prompt = f"""
Plan a real R1 Lite right-arm task using the measured RGB-D cup poses below.
All cups are inverted and nest to the same final height. The immutable task is:
first orange cup onto purple cup, then green cup onto the nested orange cup.

Measured base_link poses:
{json.dumps(compact_scene, ensure_ascii=False)}

Use this immutable grasp geometry for both phases:
- the fingertip center is exactly {grasp_below_cup_top_m}m below the cup top
- grasp_height_ratio must be exactly {fixed_grasp_height_ratio}

Choose only these remaining bounded parameters for each phase:
- pregrasp_clearance_m: 0.08 to 0.15
- lift_clearance_m: 0.12 to 0.25 (the high transport clearance)
- grasp_descent_steps: integer 10 to 20
- release_clearance_m: 0.02 to 0.06
- nested_height_increment_m must be exactly 0.0

After grasping, the compiler performs a strict 0.05m vertical lift, moves away
from the cup row, searches multiple obstacle-clearance routes and IK branches,
and rises in the clear corridor. Horizontal motion toward the support is
forbidden until measured TCP height passes the transport-clearance gate. Every
valid route must end at the exact above-support waypoint before descending to
the non-contact release hover. There is no place descent and gravity performs
nesting.

Do not change cup identities, phase order, coordinates, gripper command, or
the captured Home gripper orientation. Return one JSON object only, matching
this structure:
{json.dumps(reference, ensure_ascii=False)}
""".strip()
    response = query_model(
        ModelQueryArgs(
            model=args.model,
            server_url=args.server_url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=4096,
            reasoning_effort="medium",
        ),
        [{"role": "user", "content": prompt}],
    )
    raw_content = str(response["content"])
    intent = extract_json(raw_content)
    validate_llm_intent(intent)
    plan = compile_llm_plan(
        intent,
        scene,
        cup_height_m=args.cup_height_m,
        cup_diameter_m=args.cup_diameter_m,
        gripper_close_position=args.gripper_close_position,
    )
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False)
    )
    plan_sha256 = hashlib.sha256(args.output_plan.read_bytes()).hexdigest()
    audit = {
        "schema": "capx.r1lite.three_cup_stack.llm_plan_audit.v1",
        "model": args.model,
        "generated_at_unix": time.time(),
        "plan_path": str(args.output_plan.resolve()),
        "plan_sha256": plan_sha256,
        "rgbd_scene": scene,
        "validated_llm_intent": intent,
        "raw_llm_response": raw_content,
        "robot_motion_commands_sent": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
