#!/usr/bin/env python3
"""Latch the live R1 Lite torso pose and continuously hold that target."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--maximum-drift-rad", type=float, default=0.03)
    return parser.parse_args()


class TorsoHoldNode(Node):
    def __init__(
        self,
        *,
        ready_file: Path,
        rate_hz: float,
        maximum_drift_rad: float,
    ) -> None:
        super().__init__("capx_r1lite_torso_hold")
        self._ready_file = ready_file
        self._maximum_drift = float(maximum_drift_rad)
        self._reference: np.ndarray | None = None
        self._latest: np.ndarray | None = None
        self._violation = False
        self._publisher = self.create_publisher(
            JointState,
            "/motion_target/target_joint_state_torso",
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            "/hdas/feedback_torso",
            self._feedback,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / float(rate_hz), self._publish_hold)

    def _write_status(self) -> None:
        payload = {
            "pid": os.getpid(),
            "reference_position_rad": (
                None if self._reference is None else self._reference.tolist()
            ),
            "latest_position_rad": (
                None if self._latest is None else self._latest.tolist()
            ),
            "maximum_drift_rad": self._maximum_drift,
            "violation": self._violation,
            "updated_at_unix": time.time(),
        }
        temporary = self._ready_file.with_suffix(
            self._ready_file.suffix + ".tmp"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(self._ready_file)

    def _feedback(self, message: JointState) -> None:
        position = np.asarray(message.position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            return
        self._latest = position
        if self._reference is None:
            self._reference = position.copy()
            self._write_status()
            self.get_logger().info(
                f"latched torso hold target {self._reference.tolist()}"
            )
            return
        drift = float(np.max(np.abs(position - self._reference)))
        if drift > self._maximum_drift and not self._violation:
            self._violation = True
            self._write_status()
            self.get_logger().error(
                f"torso drift {drift:.4f}rad exceeds "
                f"{self._maximum_drift:.4f}rad"
            )

    def _publish_hold(self) -> None:
        if self._reference is None:
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = ["torso"]
        message.position = self._reference.tolist()
        self._publisher.publish(message)


def main() -> None:
    args = parse_args()
    if not 10.0 <= args.rate_hz <= 500.0:
        raise ValueError("torso hold rate must be in [10Hz, 500Hz]")
    if not 0.005 <= args.maximum_drift_rad <= 0.05:
        raise ValueError("maximum torso drift must be in [0.005rad, 0.05rad]")
    rclpy.init()
    node = TorsoHoldNode(
        ready_file=args.ready_file,
        rate_hz=args.rate_hz,
        maximum_drift_rad=args.maximum_drift_rad,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
