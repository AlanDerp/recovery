#!/usr/bin/env python3
"""Generate and validate all LLM Cartesian anchors; never move the arm."""

from __future__ import annotations

import argparse
import copy
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

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel
from capx.integrations.r1lite.control import R1LiteControlApi
from capx.llm.client import ModelQueryArgs, query_model


REQUIRED_ROLES = (
    "above_ball",
    "pregrasp",
    "approach",
    "grasp",
    "transfer",
    "above_box",
)
ALLOWED_ROLES = set(REQUIRED_ROLES) | {"transit"}
DEFAULT_MAX_CARTESIAN_STEP_M = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--template-plan", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8110/chat/completions",
    )
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument(
        "--max-cartesian-step-m",
        type=float,
        default=DEFAULT_MAX_CARTESIAN_STEP_M,
    )
    return parser.parse_args()


def extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        begin = text.find("{")
        end = text.rfind("}")
        if begin < 0 or end <= begin:
            raise ValueError("LLM response contains no JSON object")
        payload = json.loads(text[begin : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM plan must be a JSON object")
    return payload


def validate_anchors(
    payload: dict[str, Any],
    ball: np.ndarray,
    box: np.ndarray,
    workspace: tuple[tuple[float, float], ...],
) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("LLM plan must contain a frames list")
    if not 6 <= len(frames) <= 24:
        raise ValueError("LLM plan must contain 6-24 semantic anchors")

    anchors: list[dict[str, Any]] = []
    ids: set[str] = set()
    observed_required_roles: list[str] = []
    for index, item in enumerate(frames):
        if not isinstance(item, dict):
            raise ValueError(f"frame {index} must be an object")
        frame_id = str(item.get("id", "")).strip()
        role = str(item.get("role", "")).strip()
        if not frame_id or frame_id in ids:
            raise ValueError(f"frame IDs must be non-empty and unique: {frame_id!r}")
        ids.add(frame_id)
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{frame_id} has unsupported role {role!r}")
        if role != "transit":
            observed_required_roles.append(role)
        position = np.asarray(item.get("position_base_m"), dtype=float).reshape(3)
        if not np.isfinite(position).all():
            raise ValueError(f"non-finite target for {frame_id}")
        for axis, (lower, upper) in enumerate(workspace):
            if not lower <= position[axis] <= upper:
                raise ValueError(
                    f"{frame_id} axis {axis}={position[axis]:.4f} "
                    f"outside [{lower:.4f}, {upper:.4f}]"
                )
        anchors.append({"id": frame_id, "role": role, "position": position})

    if observed_required_roles != list(REQUIRED_ROLES):
        raise ValueError(
            "non-transit roles must occur exactly once and in order: "
            f"{REQUIRED_ROLES}; got {observed_required_roles}"
        )

    by_role = {
        anchor["role"]: anchor
        for anchor in anchors
        if anchor["role"] != "transit"
    }
    pick = [by_role[role]["position"] for role in REQUIRED_ROLES[:4]]
    for role, position in zip(REQUIRED_ROLES[:4], pick, strict=True):
        if np.linalg.norm(position[:2] - ball[:2]) > 0.12:
            raise ValueError(f"{role} is more than 12cm from ball XY")
    offsets = [float(position[2] - ball[2]) for position in pick]
    if not (0.15 <= offsets[0] <= 0.35):
        raise ValueError("above_ball must be 15-35cm above the ball")
    if not (0.08 <= offsets[1] < offsets[0]):
        raise ValueError("pregrasp must descend and remain at least 8cm high")
    if not (0.04 <= offsets[2] < offsets[1]):
        raise ValueError("approach must descend and remain at least 4cm high")
    if not (0.02 <= offsets[3] < offsets[2]):
        raise ValueError("grasp must descend and remain above the ball")

    place = by_role["above_box"]["position"]
    if np.linalg.norm(place[:2] - box[:2]) > 0.15:
        raise ValueError("above_box must be within 15cm of box-center XY")
    place_height = float(place[2] - box[2])
    if not 0.10 <= place_height <= 0.35:
        raise ValueError("above_box must be 10-35cm above the box center")
    return anchors


def densify_anchors(
    start_position: np.ndarray,
    anchors: list[dict[str, Any]],
    *,
    max_step_m: float,
) -> list[dict[str, Any]]:
    if not 0.0 < float(max_step_m) <= DEFAULT_MAX_CARTESIAN_STEP_M:
        raise ValueError("max Cartesian step must be in (0, 0.10] meters")
    previous = np.asarray(start_position, dtype=float).reshape(3)
    expanded: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(anchors):
        target = np.asarray(anchor["position"], dtype=float).reshape(3)
        delta = target - previous
        distance = float(np.linalg.norm(delta))
        segment_count = max(1, int(np.ceil(distance / float(max_step_m))))
        for segment_index in range(1, segment_count + 1):
            is_anchor = segment_index == segment_count
            position = previous + delta * (segment_index / segment_count)
            expanded.append(
                {
                    "id": (
                        str(anchor["id"])
                        if is_anchor
                        else (
                            f"interp_{anchor_index:02d}_"
                            f"{segment_index:02d}_of_{segment_count:02d}"
                        )
                    ),
                    "role": str(anchor["role"]) if is_anchor else "transit",
                    "position": position,
                    "target_source": (
                        "llm_runtime_anchor"
                        if is_anchor
                        else "system_interpolation"
                    ),
                    "is_interpolated": not is_anchor,
                    "anchor_id": str(anchor["id"]),
                }
            )
        previous = target
    for left, right in zip(
        [np.asarray(start_position, dtype=float)] + [
            np.asarray(item["position"], dtype=float) for item in expanded[:-1]
        ],
        [np.asarray(item["position"], dtype=float) for item in expanded],
        strict=True,
    ):
        if float(np.linalg.norm(right - left)) > float(max_step_m) + 1e-9:
            raise AssertionError("densified plan exceeds Cartesian step bound")
    return expanded


def main() -> None:
    args = parse_args()
    key = args.api_key_file.read_text().strip()
    if not key:
        raise RuntimeError("API key file is empty")

    cfg = yaml.safe_load(args.config.read_text())
    low_cfg = dict(cfg["env"]["cfg"]["low_level"])
    low_cfg.pop("_target_", None)
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {name: value for name, value in low_cfg.items() if name in signature.parameters}

    r1 = R1LiteRealLowLevel(**kwargs)
    try:
        api = R1LiteControlApi(r1)
        box_position, _ = api.get_object_pose("cardboard box")
        ball_position, _ = api.get_object_pose("tennis ball")
        box = np.asarray(box_position, dtype=float).reshape(3)
        ball = np.asarray(ball_position, dtype=float).reshape(3)
        if not np.isfinite(box).all() or not np.isfinite(ball).all():
            raise RuntimeError("perceived object position is non-finite")

        workspace = (
            tuple(float(v) for v in r1.workspace_x_limits),
            tuple(float(v) for v in r1.workspace_y_limits),
            tuple(float(v) for v in r1.workspace_z_limits),
        )
        plan = json.loads(args.template_plan.read_text())
        template_waypoints = plan["waypoints"]
        if len(template_waypoints) < 3:
            raise RuntimeError("template plan must contain F0, F1, and an IK waypoint")
        f1_position = np.asarray(
            template_waypoints[1]["target_ee_position_base_m"], dtype=float
        ).reshape(3)
        prompt = f"""
You are planning all Cartesian target positions for one R1 Lite tennis-ball
pick-and-place run.
All coordinates use base_link meters. Perception has measured:
- tennis_ball_center_base_m: {ball.tolist()}
- box_center_base_m: {box.tolist()}

Return JSON only with key "frames", containing 6-24 ordered semantic anchors.
Each frame must contain:
- "id": a unique descriptive string
- "role": one of {sorted(ALLOWED_ROLES)}
- "position_base_m": three finite numbers

The non-transit roles must occur exactly once in this order:
{list(REQUIRED_ROLES)}
You may insert any number of "transit" anchors between them.

Semantics:
- above_ball, pregrasp, approach, and grasp remain within 0.12m of ball XY.
- above_ball is 0.15-0.35m above the ball.
- pregrasp is at least 0.08m above the ball and below above_ball.
- approach is at least 0.04m above the ball and below pregrasp.
- grasp is at least 0.02m above the ball and below approach.
- transfer moves the already-grasped ball safely toward the box.
- above_box is within 0.15m of box XY and 0.10-0.35m above box center.
- Workspace x/y/z limits: {workspace}.
The system will deterministically interpolate every segment to at most
{args.max_cartesian_step_m:.3f}m. Do not output joint values, actions, prose, or Markdown.
""".strip()
        result = query_model(
            ModelQueryArgs(
                model=args.model,
                server_url=args.server_url,
                api_key=key,
                temperature=0.0,
                max_tokens=2048,
                reasoning_effort="medium",
            ),
            [{"role": "user", "content": prompt}],
        )
        content = str(result["content"])
        payload = extract_json(content)
        anchors = validate_anchors(payload, ball, box, workspace)
        expanded = densify_anchors(
            f1_position,
            anchors,
            max_step_m=args.max_cartesian_step_m,
        )
        runtime_torso_joints = r1.get_torso_joints()
        if not np.isfinite(runtime_torso_joints).all():
            raise RuntimeError("runtime torso joint feedback is non-finite")

        preserved = copy.deepcopy(template_waypoints[:2])
        preserved[0]["semantic_role"] = "initial"
        preserved[1]["semantic_role"] = "f1_anchor"
        ik_template = template_waypoints[2]
        generated_waypoints = []
        for expanded_index, item in enumerate(expanded, start=2):
            waypoint = copy.deepcopy(ik_template)
            waypoint["index"] = expanded_index
            waypoint["id"] = str(item["id"])
            waypoint["label"] = (
                f"LLM anchor: {item['role']}"
                if not item["is_interpolated"]
                else f"System interpolation toward {item['anchor_id']}"
            )
            waypoint["target_ee_position_base_m"] = np.asarray(
                item["position"], dtype=float
            ).tolist()
            waypoint["target_source"] = str(item["target_source"])
            waypoint["semantic_role"] = str(item["role"])
            waypoint["semantic_action"] = (
                "close_gripper"
                if item["role"] == "grasp"
                else "release_and_return_f1"
                if item["role"] == "above_box"
                else "move"
            )
            waypoint["is_interpolated"] = bool(item["is_interpolated"])
            waypoint["anchor_id"] = str(item["anchor_id"])
            waypoint["torso_joints_rad"] = runtime_torso_joints.tolist()
            waypoint["torso_joints_source"] = "runtime_feedback"
            waypoint["skip_xyz_correction"] = False
            generated_waypoints.append(waypoint)
        plan["waypoints"] = preserved + generated_waypoints
        objects = plan.setdefault("objects", {})
        objects["tennis_ball_center_base_m"] = ball.tolist()
        objects["box_center_base_m"] = box.tolist()
        objects["drop_position_base_m"] = (
            box + np.array([0.0, 0.0, 0.10])
        ).tolist()
        plan["fourth_pipeline"] = {
            "planner": "llm_variable_semantic_cartesian_targets",
            "model": args.model,
            "generated_at_unix": time.time(),
            "variable_frame_plan": True,
            "coupled_grasp_enabled": False,
            "max_cartesian_step_m": float(args.max_cartesian_step_m),
            "llm_anchor_count": len(anchors),
            "expanded_waypoint_count": len(generated_waypoints),
            "f0_f1_source": "template_preserved",
            "generated_orientation_source": "template_preserved",
            "generated_torso_joints_source": "runtime_feedback",
            "runtime_torso_joints_rad": runtime_torso_joints.tolist(),
            "execution_round_count": 1,
        }

        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        args.output_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        output_plan_sha256 = hashlib.sha256(
            args.output_plan.read_bytes()
        ).hexdigest()
        audit = {
            "model": args.model,
            "server_url": args.server_url,
            "ball_position_base_m": ball.tolist(),
            "box_position_base_m": box.tolist(),
            "runtime_torso_joints_rad": runtime_torso_joints.tolist(),
            "max_cartesian_step_m": float(args.max_cartesian_step_m),
            "llm_anchors": [
                {
                    "id": anchor["id"],
                    "role": anchor["role"],
                    "position_base_m": anchor["position"].tolist(),
                }
                for anchor in anchors
            ],
            "expanded_waypoints": [
                {
                    "id": item["id"],
                    "role": item["role"],
                    "position_base_m": item["position"].tolist(),
                    "target_source": item["target_source"],
                    "anchor_id": item["anchor_id"],
                }
                for item in expanded
            ],
            "raw_response": content,
            "robot_motion_commands_sent": False,
            "output_plan": str(args.output_plan),
            "output_plan_sha256": output_plan_sha256,
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False)
        )
        print(json.dumps(audit, ensure_ascii=False))
    finally:
        r1.close()


if __name__ == "__main__":
    main()
