#!/usr/bin/env python3
"""Plan one R1 Lite tennis grasp with Contact-GraspNet without moving the robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader
from capx.integrations.r1lite.control import R1LiteControlApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="env_configs/r1lite/r1lite_tennis_ball_in_box.yaml",
    )
    parser.add_argument("--no-sam3", action="store_true")
    parser.add_argument("--skip-head", action="store_true")
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--dynamic-view-only", action="store_true")
    args = parser.parse_args()

    config = DictLoader.load([args.config])
    if config.get("r1lite_graspnet_mandatory") is not True:
        raise RuntimeError("Config must enable r1lite_graspnet_mandatory")
    if config.get("r1lite_graspnet_allow_fallback") is not False:
        raise RuntimeError("Config must explicitly disable GraspNet fallback")

    low_level = instantiate(config["env"]["cfg"]["low_level"])
    wrist_enabled = bool(getattr(low_level, "wrist_camera_enabled", True))
    try:
        api = R1LiteControlApi(low_level)
        api.save_head_rgb("outputs/r1lite/graspnet_smoke_head.png")
        if args.dynamic_view_only:
            if not wrist_enabled:
                raise RuntimeError(
                    "dynamic-view preflight requires wrist_camera_enabled=true"
                )
            detection = api.locate_object_head(
                "tennis ball",
                use_sam3=not args.no_sam3,
                save_debug=True,
            )
            if not detection.get("ok"):
                raise RuntimeError(f"Head tennis detection failed: {detection}")
            ball_position = low_level.pixel_to_base_on_plane(detection["pixel"])
            ball_position[2] += low_level.tennis_ball_radius_m
            observation = api.plan_wrist_observation_pose(ball_position)
            wrist = None
            last_error = None
            for _ in range(4):
                try:
                    wrist = low_level.get_wrist_rgbd(require_depth=True)
                    break
                except (RuntimeError, TimeoutError) as exc:
                    last_error = exc
            if wrist is None:
                raise RuntimeError(
                    f"Wrist RGB-D/TF preflight failed: {last_error}"
                )
            base_from_camera = low_level.lookup_wrist_camera_to_base(
                frame_id=wrist["frame_id"],
                stamp_ns=wrist["stamp_ns"],
            )
            print(json.dumps({
                "ok": True,
                "dynamic_view_only": True,
                "safe_to_execute": False,
                "head_ball_position_base": np.asarray(
                    ball_position,
                    dtype=float,
                ).tolist(),
                "target_ee_position": observation["target_ee_position"],
                "target_ee_quaternion_xyzw": (
                    observation["target_ee_quaternion_xyzw"]
                ),
                "geometry": observation["geometry"],
                "waypoint_count": observation["ik_plan"]["waypoint_count"],
                "max_total_joint_delta_rad": max(
                    waypoint["total_joint_delta_rad"]
                    for waypoint in observation["ik_plan"]["waypoints"]
                ),
                "base_from_current_camera": base_from_camera.tolist(),
            }, indent=2))
            return
        if wrist_enabled:
            api.save_wrist_rgb("outputs/r1lite/graspnet_smoke_wrist.png")
        plan = api.plan_tennis_graspnet(
            use_sam3=not args.no_sam3,
            save_debug=True,
            confirm_head=not args.skip_head,
            camera_only=args.camera_only,
            camera="wrist" if wrist_enabled else "head",
        )
        if plan.get("camera_only"):
            print(json.dumps({
                "ok": plan["ok"],
                "camera_only": True,
                "safe_to_execute": plan["safe_to_execute"],
                "reason": plan["reason"],
                "frame_id": plan["frame_id"],
                "pc_full_points": plan["pc_full_points"],
                "pc_segment_points": plan["pc_segment_points"],
                "num_candidates": plan["num_candidates"],
                "scores": plan["scores"],
            }, indent=2))
            return
        selected = plan["selected"]
        summary = {
            "ok": plan["ok"],
            "mandatory_graspnet": plan["mandatory_graspnet"],
            "fallback_used": plan["fallback_used"],
            "frame_id": plan["frame_id"],
            "stamp_ns": plan["stamp_ns"],
            "pc_full_points": plan["pc_full_points"],
            "pc_segment_points": plan["pc_segment_points"],
            "selected_index": selected["index"],
            "selected_score": selected["score"],
            "grasp_position": selected["grasp_position"],
            "grasp_quaternion_xyzw": selected["grasp_quaternion_xyzw"],
            "pregrasp_position": selected["pregrasp_position"],
            "approach_axis_base": selected["approach_axis_base"],
            "r1lite_ik_harness": selected["r1lite_ik_harness"],
            "rejected_candidates": plan["rejected_candidates"],
        }
        print(json.dumps(summary, indent=2))
    finally:
        low_level.close()


if __name__ == "__main__":
    main()
