#!/usr/bin/env python3
"""Move the R1 Lite left arm to Home and hold it during right-arm tasks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from hdas_msg.msg import MotorControl
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState


DEFAULT_LEFT_HOME = (
    0.01574468085106383,
    0.21404255319148935,
    -0.01276595744680851,
    0.01851063829787234,
    0.02468085106382979,
    0.02468085106382979,
)


def bounded_feedback_target(
    current: np.ndarray,
    target: np.ndarray,
    maximum_step_rad: float,
) -> np.ndarray:
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    distance = float(np.max(np.abs(delta)))
    if distance <= maximum_step_rad:
        return np.asarray(target, dtype=float).copy()
    return np.asarray(current, dtype=float) + delta * (maximum_step_rad / distance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--target", nargs=6, type=float, default=DEFAULT_LEFT_HOME)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--maximum-step-rad", type=float, default=0.003)
    parser.add_argument("--arrival-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--maximum-feedback-step-rad", type=float, default=0.06)
    return parser.parse_args()


class LeftArmHomeHold(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("capx_r1lite_left_arm_home_hold")
        self.args = args
        self.target = np.asarray(args.target, dtype=float).reshape(6)
        self.feedback: np.ndarray | None = None
        self.control: np.ndarray | None = None
        self.previous_feedback: np.ndarray | None = None
        self.arrived = False
        self.violation = False
        self.last_status_write = 0.0
        self.publisher = self.create_publisher(
            JointState,
            "/motion_target/target_joint_state_arm_left",
            QoSProfile(depth=5),
        )
        self.create_subscription(
            JointState,
            "/hdas/feedback_arm_left",
            self.on_feedback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            MotorControl,
            "/motion_control/control_arm_left",
            self.on_control,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / float(args.rate_hz), self.publish_target)

    def on_feedback(self, message: JointState) -> None:
        if len(message.position) < 6:
            return
        feedback = np.asarray(message.position[:6], dtype=float)
        if not np.isfinite(feedback).all():
            return
        if self.previous_feedback is not None:
            step = float(np.max(np.abs(feedback - self.previous_feedback)))
            if step > float(self.args.maximum_feedback_step_rad):
                self.violation = True
        self.previous_feedback = feedback.copy()
        self.feedback = feedback
        error = float(np.max(np.abs(self.target - feedback)))
        self.arrived = error <= float(self.args.arrival_tolerance_rad)
        now = time.monotonic()
        if self.violation or self.arrived or now - self.last_status_write >= 1.0:
            self.write_status()
            self.last_status_write = now

    def on_control(self, message: MotorControl) -> None:
        if len(message.p_des) < 6:
            return
        control = np.asarray(message.p_des[:6], dtype=float)
        if np.isfinite(control).all():
            self.control = control

    def publish_target(self) -> None:
        if self.feedback is None or self.control is None or self.violation:
            return
        next_feedback = bounded_feedback_target(
            self.feedback,
            self.target,
            float(self.args.maximum_step_rad),
        )
        command = self.control + next_feedback - self.feedback
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.position = command.tolist()
        message.velocity = [1.0] * 6
        self.publisher.publish(message)

    def write_status(self) -> None:
        error = (
            None
            if self.feedback is None
            else float(np.max(np.abs(self.target - self.feedback)))
        )
        payload = {
            "pid": os.getpid(),
            "target_position_rad": self.target.tolist(),
            "latest_position_rad": (
                None if self.feedback is None else self.feedback.tolist()
            ),
            "maximum_error_rad": error,
            "arrived": self.arrived,
            "violation": self.violation,
            "updated_at_unix": time.time(),
        }
        temporary = self.args.ready_file.with_suffix(
            self.args.ready_file.suffix + ".tmp"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(self.args.ready_file)


def main() -> None:
    args = parse_args()
    if not 10.0 <= args.rate_hz <= 100.0:
        raise ValueError("rate-hz must be in [10, 100]")
    if not 0.001 <= args.maximum_step_rad <= 0.01:
        raise ValueError("maximum-step-rad must be in [0.001, 0.01]")
    if not 0.005 <= args.arrival_tolerance_rad <= 0.03:
        raise ValueError("arrival-tolerance-rad must be in [0.005, 0.03]")
    rclpy.init()
    node = LeftArmHomeHold(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
