#!/usr/bin/env python3
"""Compile a hash-bound beizi-general plan from saved cup positions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.cup_stack_llm import (  # noqa: E402
    compile_general_plan,
    execute_generated_code_dry_run,
    expected_operations,
    validate_generated_code,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_color")
    parser.add_argument("second_color")
    parser.add_argument("base_color")
    parser.add_argument("--fixed-scene", type=Path, required=True)
    parser.add_argument("--template-plan", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-code", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def render_program(operations: list[dict[str, object]]) -> str:
    lines = []
    for operation in operations:
        kwargs = ", ".join(
            f"{key}={value!r}"
            for key, value in operation.items()
            if key != "call"
        )
        lines.append(f"{operation['call']}({kwargs})")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    fixed_scene = json.loads(args.fixed_scene.read_text())
    template = json.loads(args.template_plan.read_text())
    scene = copy.deepcopy(template["rgbd_scene"])
    scene["cups"] = copy.deepcopy(fixed_scene["cups"])
    scene["pose_source"] = "fixed_user_measurement"
    scene["fixed_scene_schema"] = fixed_scene.get("schema")
    scene["fixed_scene_path"] = str(args.fixed_scene.resolve())
    for field in (
        "measurement_note",
        "reference_gripper_x_m",
        "fixed_cup_x_m",
        "table_height_m",
    ):
        if field in fixed_scene:
            scene[field] = copy.deepcopy(fixed_scene[field])

    colors = (args.first_color, args.second_color, args.base_color)
    plan = compile_general_plan(
        colors,
        scene,
        cup_height_m=float(template["cup_height_m"]),
        cup_diameter_m=float(template["cup_diameter_m"]),
        gripper_close_position=float(template["gripper_close_position"]),
    )
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    )
    plan_sha256 = hashlib.sha256(args.output_plan.read_bytes()).hexdigest()
    operations = expected_operations(plan, plan_sha256=plan_sha256)
    code = render_program(operations)
    validate_generated_code(code, expected=operations)
    execute_generated_code_dry_run(code, expected=operations)
    args.output_code.parent.mkdir(parents=True, exist_ok=True)
    args.output_code.write_text(code)
    code_sha256 = hashlib.sha256(args.output_code.read_bytes()).hexdigest()

    audit = {
        "schema": "capx.r1lite.beizi_general.compile_audit.v1",
        "pipeline": "beizi-general",
        "color_order": list(colors),
        "fixed_scene_path": str(args.fixed_scene.resolve()),
        "template_plan_path": str(args.template_plan.resolve()),
        "plan_path": str(args.output_plan.resolve()),
        "plan_sha256": plan_sha256,
        "code_path": str(args.output_code.resolve()),
        "code_sha256": code_sha256,
        "restricted_dry_run_passed": True,
        "robot_motion_commands_sent": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
