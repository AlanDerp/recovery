"""Perception + geometry success check for Agilex tennis-ball-in-box tasks.

Uses SAM3 segmentation and RGB-D back-projection to estimate ball/box poses in
the robot world frame, then applies distance thresholds similar to pick-place
success logic used in manipulation benchmarks.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from capx.integrations.agilex.control import AgilexControlApi
from capx.llm.client import ModelQueryArgs, query_model

SuccessBackend = Literal["geometric", "vlm", "hybrid"]


def _encode_rgb_base64(rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _get_top_camera_rgb(control_api: AgilexControlApi) -> np.ndarray:
    obs = control_api.get_observation()
    rgb = obs.get("robot0_robotview", {}).get("images", {}).get("rgb")
    if rgb is None:
        raise RuntimeError("missing top-camera RGB in observation")
    return np.asarray(rgb, dtype=np.uint8)


def verify_tennis_ball_in_box_geometric(
    control_api: AgilexControlApi,
    *,
    xy_margin: float = 0.12,
    max_z_above_box: float = 0.15,
    min_z_below_box: float = -0.10,
    min_ball_points: int = 20,
) -> tuple[bool, dict[str, Any]]:
    """RoboSuite-style geometric check using SAM3 world-frame object poses."""
    details: dict[str, Any] = {"method": "geometric", "success": False}
    try:
        box_pos, _ = control_api.get_object_pose("cardboard box")
        ball_pos, _ = control_api.get_object_pose("tennis ball")
    except Exception as exc:
        details["error"] = str(exc)
        details["reason"] = "perception_failed"
        return False, details

    box_pos = np.asarray(box_pos, dtype=float).reshape(3)
    ball_pos = np.asarray(ball_pos, dtype=float).reshape(3)
    xy_dist = float(np.linalg.norm(ball_pos[:2] - box_pos[:2]))
    z_delta = float(ball_pos[2] - box_pos[2])

    points = control_api._perception_models.get("last_object_points_world")
    ball_point_count = int(len(points)) if points is not None else 0

    in_box_xy = xy_dist <= xy_margin
    in_box_z = min_z_below_box <= z_delta <= max_z_above_box
    on_table_outside = (not in_box_xy) and z_delta < min_z_below_box

    success = in_box_xy and in_box_z and not on_table_outside
    if ball_point_count < min_ball_points:
        details["warning"] = "low_ball_point_count"

    if success:
        reason = "ball_in_box"
    elif not in_box_xy:
        reason = "ball_outside_box_xy"
    elif on_table_outside:
        reason = "ball_still_on_table"
    else:
        reason = "ball_not_at_box_height"

    details.update(
        {
            "success": success,
            "reason": reason,
            "box_pos": box_pos.tolist(),
            "ball_pos": ball_pos.tolist(),
            "xy_distance_m": xy_dist,
            "z_delta_m": z_delta,
            "ball_point_count": ball_point_count,
            "thresholds": {
                "xy_margin": xy_margin,
                "max_z_above_box": max_z_above_box,
                "min_z_below_box": min_z_below_box,
            },
        }
    )
    return success, details


def _parse_vlm_success_response(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"raw_response": text.strip()}
    if not text.strip():
        parsed["reason"] = "empty_vlm_response"
        parsed["success"] = False
        return parsed

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict):
                parsed.update(payload)
        except json.JSONDecodeError:
            pass

    success_token = str(parsed.get("success", "")).strip().lower()
    if success_token in {"true", "yes", "1", "success", "succeeded"}:
        parsed["success"] = True
    elif success_token in {"false", "no", "0", "fail", "failed", "failure"}:
        parsed["success"] = False
    elif re.search(r"\bSUCCESS\s*:\s*(YES|TRUE|1)\b", text, flags=re.IGNORECASE):
        parsed["success"] = True
    elif re.search(r"\bSUCCESS\s*:\s*(NO|FALSE|0)\b", text, flags=re.IGNORECASE):
        parsed["success"] = False
    else:
        parsed.setdefault("success", False)
        parsed.setdefault("reason", "unparsed_vlm_response")

    if "ball_location" not in parsed:
        loc_match = re.search(r"BALL_LOCATION\s*:\s*([A-Za-z_]+)", text, flags=re.IGNORECASE)
        if loc_match:
            parsed["ball_location"] = loc_match.group(1).lower()

    parsed["method"] = "vlm"
    return parsed


def verify_tennis_ball_in_box_vlm(
    control_api: AgilexControlApi,
    *,
    model: str = "qwen/qwen3.7-max",
    server_url: str = "http://127.0.0.1:8110/chat/completions",
    api_key: str | None = None,
    task_prompt: str = "Pick up one tennis ball and place it into the cardboard box.",
) -> tuple[bool, dict[str, Any]]:
    """VLM judge on the current top-camera frame."""
    details: dict[str, Any] = {"method": "vlm", "success": False}
    try:
        rgb = _get_top_camera_rgb(control_api)
    except Exception as exc:
        details["error"] = str(exc)
        details["reason"] = "missing_camera_frame"
        return False, details

    image_url = _encode_rgb_base64(rgb)
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a strict real-robot task success evaluator. "
                "Judge only from the image. Do not assume success unless the tennis ball "
                "is clearly resting inside the cardboard box and not still on the table "
                "or in the gripper."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Task: {task_prompt}\n\n"
                        "Look at the top-camera image after robot execution.\n"
                        "Return ONLY JSON with keys:\n"
                        '  "success": true or false,\n'
                        '  "ball_location": "in_box" | "on_table" | "in_gripper" | "not_visible",\n'
                        '  "confidence": "high" | "medium" | "low",\n'
                        '  "reason": "short explanation"\n'
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    if not api_key and os.path.isfile(".openrouterkey"):
        api_key = next(
            (
                line.strip()
                for line in Path(".openrouterkey").read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ),
            None,
        )

    result = query_model(
        ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=512,
        ),
        prompt,
    )
    parsed = _parse_vlm_success_response(result.get("content") or "")
    parsed["model"] = model
    success = bool(parsed.get("success"))
    parsed.setdefault("reason", "vlm_rejected" if not success else "vlm_confirmed")
    return success, parsed


def evaluate_tennis_task_success(
    control_api: AgilexControlApi,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Run configured success backend(s) for the tennis-ball task."""
    backend: SuccessBackend = config.get("agilex_success_backend", "geometric")
    task_prompt = (
        config.get("agilex_success_task_prompt")
        or config.get("task_only_prompt")
        or "Pick up one tennis ball and place it into the cardboard box."
    )

    geometric_ok, geometric_details = verify_tennis_ball_in_box_geometric(
        control_api,
        xy_margin=float(config.get("agilex_visual_success_xy_margin", 0.12)),
        max_z_above_box=float(config.get("agilex_visual_success_max_z_above_box", 0.15)),
        min_z_below_box=float(config.get("agilex_visual_success_min_z_below_box", -0.10)),
    )

    if backend == "geometric":
        return geometric_ok, {
            "backend": backend,
            "success": geometric_ok,
            "reason": geometric_details.get("reason", "geometric"),
            "geometric": geometric_details,
        }

    vlm_ok, vlm_details = verify_tennis_ball_in_box_vlm(
        control_api,
        model=str(config.get("agilex_success_vlm_model", "qwen/qwen3.7-max")),
        server_url=str(
            config.get(
                "agilex_success_vlm_server_url",
                "http://127.0.0.1:8110/chat/completions",
            )
        ),
        api_key=config.get("agilex_success_vlm_api_key"),
        task_prompt=task_prompt,
    )

    if backend == "vlm":
        return vlm_ok, {
            "backend": backend,
            "success": vlm_ok,
            "reason": vlm_details.get("reason", "vlm"),
            "vlm": vlm_details,
        }

    success = geometric_ok and vlm_ok
    if success:
        reason = "geometric_and_vlm_agree"
    elif not geometric_ok and not vlm_ok:
        reason = "geometric_and_vlm_reject"
    elif not geometric_ok:
        reason = "geometric_reject"
    else:
        reason = "vlm_reject"

    return success, {
        "backend": backend,
        "success": success,
        "reason": reason,
        "geometric": geometric_details,
        "vlm": vlm_details,
    }


# Backward-compatible alias used by earlier code.
verify_tennis_ball_in_box = verify_tennis_ball_in_box_geometric
