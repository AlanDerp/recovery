#!/usr/bin/env python3
"""Record measured R1 Lite right-wrist poses to an atomically updated JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class WristTrajectoryRecorder(Node):
    def __init__(
        self,
        topic: str,
        output: Path,
        ee_from_camera: np.ndarray,
    ) -> None:
        super().__init__(f"capx_wrist_trajectory_recorder_{time.time_ns()}")
        self.output = output
        self.ee_from_camera = ee_from_camera
        self.started_at = time.monotonic()
        self.samples: list[dict[str, Any]] = []
        qos = QoSProfile(depth=20)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(PoseStamped, topic, self._on_pose, qos)
        self.create_timer(0.25, self.write)

    def _on_pose(self, message: PoseStamped) -> None:
        position = message.pose.position
        orientation = message.pose.orientation
        ee_position = np.array(
            [position.x, position.y, position.z],
            dtype=float,
        )
        quaternion = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=float,
        )
        camera_position = ee_position + (
            Rotation.from_quat(quaternion).as_matrix()
            @ self.ee_from_camera[:3, 3]
        )
        self.samples.append(
            {
                "elapsed_s": time.monotonic() - self.started_at,
                "frame_id": message.header.frame_id or "base_link",
                "source_topic": "/motion_control/pose_ee_arm_right",
                "position_kind": "right_wrist_camera_center",
                "x_m": float(camera_position[0]),
                "y_m": float(camera_position[1]),
                "z_m": float(camera_position[2]),
                "ee_position_base_m": ee_position.tolist(),
                "quaternion_xyzw": quaternion.tolist(),
            }
        )

    def write(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.samples, indent=2))
        temporary.replace(self.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        default="/motion_control/pose_ee_arm_right",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration-s", type=float, default=900.0)
    parser.add_argument(
        "--ee-camera-calibration",
        type=Path,
        default=Path(
            "outputs/r1lite/right_touch_hand_eye_calibration/T_ee_camera.yaml"
        ),
    )
    args = parser.parse_args()

    payload = yaml.safe_load(args.ee_camera_calibration.read_text())
    ee_from_camera = np.asarray(payload["T_ee_camera"], dtype=float).reshape(4, 4)
    rclpy.init()
    recorder = WristTrajectoryRecorder(args.topic, args.output, ee_from_camera)
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    deadline = time.monotonic() + args.max_duration_s
    try:
        while not stopping and time.monotonic() < deadline:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    finally:
        recorder.write()
        print(
            f"Saved {len(recorder.samples)} measured wrist poses to {args.output}",
            flush=True,
        )
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
