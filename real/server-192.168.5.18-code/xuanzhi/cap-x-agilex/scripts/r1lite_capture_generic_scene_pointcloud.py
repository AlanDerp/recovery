#!/usr/bin/env python3
"""Capture SAM3-segmented head stereo point clouds without robot motion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.control import R1LiteControlApi  # noqa: E402
from scripts.r1lite_execute_llm_cup_stack import load_robot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--object",
        action="append",
        nargs=2,
        metavar=("NAME", "PROMPT"),
        required=True,
    )
    args = parser.parse_args()
    robot = load_robot(args.config)
    api = R1LiteControlApi(robot)
    try:
        observation = robot.get_head_rgbd_median(camera="left")
        rgb = np.asarray(observation["rgb"], dtype=np.uint8)
        depth = np.asarray(observation["depth"], dtype=float)
        intrinsics = np.asarray(
            observation["intrinsics"], dtype=float
        ).reshape(3, 3)
        base_from_camera = robot._head_left_camera_to_base_from_calibration()
        table_z = float(robot.load_head_plane_calibration()["plane_z"])
        base_from_camera, table_leveling = api._level_head_rgbd_to_table(
            rgb,
            depth,
            intrinsics,
            base_from_camera,
            table_z=table_z,
            save_debug=True,
        )
        detections = {}
        for name, prompt in args.object:
            slug = name.lower().replace(" ", "_")
            segmentation = api._locate_object_head_sam3(
                rgb,
                prompt=prompt,
                timeout_s=30.0,
                min_score=0.15,
                min_pixels=80,
                save_debug=True,
                slug=f"generic_scene_{slug}",
                include_mask=True,
            )
            if not segmentation.get("ok"):
                raise RuntimeError(
                    f"SAM3 failed for {name}: "
                    f"{segmentation.get('reason', segmentation)}"
                )
            mask = api._resize_mask(
                np.asarray(segmentation.pop("mask"), dtype=bool),
                depth.shape,
            )
            _full, camera_points = api._rgbd_point_clouds(
                depth,
                intrinsics,
                mask,
                camera_label=f"generic_{slug}",
            )
            base_points = (
                base_from_camera[:3, :3] @ camera_points.T
            ).T + base_from_camera[:3, 3]
            plausible = base_points[
                np.isfinite(base_points).all(axis=1)
                & (base_points[:, 2] >= table_z - 0.03)
                & (base_points[:, 2] <= table_z + 0.35)
            ]
            if len(plausible) < 30:
                raise RuntimeError(
                    f"{name} has insufficient valid points near the table"
                )
            indices = np.linspace(
                0, len(plausible) - 1, min(1500, len(plausible))
            ).astype(int)
            sampled = plausible[indices]
            center = np.median(plausible, axis=0)
            detection = {
                **segmentation,
                "position_base_m": center.tolist(),
                "table_z_base_m": table_z,
                "point_cloud_base_m": sampled.tolist(),
            }
            if "box" in name.lower():
                outer_min = np.percentile(
                    plausible[:, :2], 5.0, axis=0
                )
                outer_max = np.percentile(
                    plausible[:, :2], 95.0, axis=0
                )
                opening_min = outer_min + 0.025
                opening_max = outer_max - 0.025
                rim_z = float(
                    np.clip(
                        np.percentile(plausible[:, 2], 85.0),
                        table_z + 0.03,
                        table_z + 0.25,
                    )
                )
                detection.update(
                    {
                        "opening_xy_min_base_m": opening_min.tolist(),
                        "opening_xy_max_base_m": opening_max.tolist(),
                        "rim_z_base_m": rim_z,
                    }
                )
            detections[name] = detection
        audit = {
            "schema": "capx.r1lite.generic_scene_pointcloud.v1",
            "generated_at_unix": time.time(),
            "depth_sample_count": int(
                observation.get("depth_sample_count", 1)
            ),
            "depth_stamp_range_ns": observation.get("depth_stamp_range_ns"),
            "table_leveling": table_leveling,
            "events": [
                {
                    "event": "initial_head_rgbd_scene",
                    "detections": detections,
                }
            ],
            "robot_motion_commands_sent": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
        )
        print(args.output.resolve())
    finally:
        robot.close()


if __name__ == "__main__":
    main()
