#!/usr/bin/env python3
"""Smoke-test validated R1 Lite right-arm joint and gripper control.

This script avoids the full CaP-X dependency stack and talks directly to the
validated ROS 2 topics:

- /motion_target/target_joint_state_arm_right
- /motion_target/target_position_gripper_right
- /hdas/feedback_arm_right
- /hdas/feedback_gripper_right
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import rclpy
from PIL import Image
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from hdas_msg.srv import FunctionFrame


# Right-arm boot reset command.  The paired feedback target is the measured
# left-arm boot pose from 2026-07-16; the left arm had not been moved.
SAFE_HOME_COMMAND = np.array(
    [0.010470, 0.222985, -0.022772, 0.008466, 0.035299, 0.006095],
    dtype=float,
)
SAFE_HOME_FEEDBACK = np.array(
    [0.027234, 0.448298, -0.005745, 0.028936, 0.058085, 0.032340],
    dtype=float,
)


class R1LiteSafeSmoke(Node):
    def __init__(self) -> None:
        super().__init__("capx_r1lite_safe_smoke")
        self.arm_feedback: np.ndarray | None = None
        self.gripper_feedback: float | None = None
        self.wrist_rgb: np.ndarray | None = None
        self.arm_pub = self.create_publisher(
            JointState,
            "/capx/motion_target/target_joint_state_arm_right",
            10,
        )
        self.gripper_pub = self.create_publisher(
            JointState,
            "/capx/motion_target/target_position_gripper_right",
            10,
        )
        self.brake_pub = self.create_publisher(Bool, "/motion_target/brake_mode", 10)
        self.function_client = self.create_client(
            FunctionFrame, "/function_frame_right_arm"
        )
        self.create_subscription(
            JointState,
            "/hdas/feedback_arm_right",
            self._on_arm,
            10,
        )
        self.create_subscription(
            JointState,
            "/hdas/feedback_gripper_right",
            self._on_gripper,
            10,
        )
        self.create_subscription(
            ImageMsg,
            "/hdas/camera_wrist_right/color/image_raw",
            self._on_wrist_rgb,
            qos_profile_sensor_data,
        )

    def _on_arm(self, msg: JointState) -> None:
        if len(msg.position) >= 6:
            self.arm_feedback = np.asarray(msg.position[:6], dtype=float)

    def _on_gripper(self, msg: JointState) -> None:
        if not msg.position:
            return
        self.gripper_feedback = float(msg.position[0])

    def _on_wrist_rgb(self, msg: ImageMsg) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        channels = max(1, int(msg.step) // max(1, int(msg.width)))
        rgb = arr.reshape(int(msg.height), int(msg.width), channels)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        encoding = str(msg.encoding).lower()
        if encoding in {"bgr8", "bgra8"}:
            rgb = rgb[..., ::-1]
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        self.wrist_rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    def wait_for_feedback(self, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.arm_feedback is not None and self.gripper_feedback is not None:
                return
        raise TimeoutError("Timed out waiting for R1 Lite arm/gripper feedback")

    def save_wrist_rgb(self, path: str, timeout_s: float = 5.0) -> str:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.wrist_rgb is not None:
                out = Path(path)
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(self.wrist_rgb).save(out)
                return str(out)
        raise TimeoutError("Timed out waiting for right wrist RGB frame")

    def _publish_joint_state(self, pub, name: str, position: list[float]) -> None:
        msg = JointState()
        msg.name = []
        msg.position = [float(x) for x in position]
        if name == "right_arm":
            msg.velocity = [1.6, 1.6, 1.6, 4.0, 4.0, 4.0]
        pub.publish(msg)

    def enable_right_arm(self) -> None:
        if not self.function_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("right-arm FunctionFrame service is unavailable")
        for command in (2, 5, 1):
            request = FunctionFrame.Request()
            request.command = command
            future = self.function_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError(f"FunctionFrame command {command} failed")
        brake = Bool()
        brake.data = True
        deadline = time.time() + 1.0
        while time.time() < deadline:
            self.brake_pub.publish(brake)
            rclpy.spin_once(self, timeout_sec=0.05)

    def command_gripper(self, position: float, duration_s: float = 3.0) -> None:
        deadline = time.time() + duration_s
        while time.time() < deadline:
            self._publish_joint_state(
                self.gripper_pub,
                "right_gripper",
                [float(position)],
            )
            rclpy.spin_once(self, timeout_sec=0.05)

    def move_joints_safe(
        self,
        target: np.ndarray,
        *,
        max_step: float = 0.03,
        duration_s: float = 3.0,
        tolerance_rad: float = 0.025,
        divergence_limit_rad: float = 0.04,
        expected_feedback: np.ndarray | None = None,
    ) -> None:
        self.wait_for_feedback()
        assert self.arm_feedback is not None
        current = self.arm_feedback.copy()
        target = np.asarray(target, dtype=float).reshape(6)
        # R1 Lite arm feedback is approximately twice the MOCA joint target
        # displacement around the paired boot pose.
        current_command = SAFE_HOME_COMMAND + 0.5 * (current - SAFE_HOME_FEEDBACK)
        expected = (
            target
            if expected_feedback is None
            else np.asarray(expected_feedback, dtype=float).reshape(6)
        )
        initial_error = float(np.max(np.abs(expected - current)))
        steps = max(
            1,
            int(np.ceil(np.max(np.abs(target - current_command)) / max_step)),
        )
        for idx in range(1, steps + 1):
            waypoint = current_command + (target - current_command) * (idx / steps)
            self._publish_joint_state(self.arm_pub, "right_arm", waypoint.tolist())
            rclpy.spin_once(self, timeout_sec=0.05)
        deadline = time.time() + duration_s
        while time.time() < deadline:
            self._publish_joint_state(self.arm_pub, "right_arm", target.tolist())
            rclpy.spin_once(self, timeout_sec=0.05)
            assert self.arm_feedback is not None
            target_error = float(np.max(np.abs(expected - self.arm_feedback)))
            measured_motion = float(np.max(np.abs(self.arm_feedback - current)))
            if measured_motion >= 0.005 and target_error > initial_error + divergence_limit_rad:
                raise RuntimeError(
                    "Arm moved away from target; command publication stopped. "
                    "The servo may be passive or controlled elsewhere: "
                    f"before={current.tolist()}, target={target.tolist()}, "
                    f"measured={self.arm_feedback.tolist()}"
                )
        assert self.arm_feedback is not None
        target_error = float(np.max(np.abs(expected - self.arm_feedback)))
        if target_error > tolerance_rad:
            raise TimeoutError(
                "Arm target did not converge: "
                f"target={target.tolist()}, measured={self.arm_feedback.tolist()}, "
                f"max_error={target_error:.4f}, tolerance={tolerance_rad:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", dest="open_position", type=float, default=14.5)
    parser.add_argument("--close", dest="close_position", type=float, default=5.5)
    parser.add_argument("--max-step", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.025,
        help="Maximum allowed error against the paired feedback safe pose.",
    )
    parser.add_argument(
        "--no-gripper-cycle",
        action="store_true",
        help="Only hold the safe arm pose; do not open/close the gripper.",
    )
    parser.add_argument(
        "--save-rgb",
        default=None,
        help="Optional path to save one right-wrist RGB frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = R1LiteSafeSmoke()
    try:
        node.wait_for_feedback()
        print("Initial arm:", node.arm_feedback)
        print("Initial gripper:", node.gripper_feedback)

        node.enable_right_arm()
        print("Moving to validated safe joint command:", SAFE_HOME_COMMAND.tolist())
        node.move_joints_safe(
            SAFE_HOME_COMMAND,
            max_step=args.max_step,
            duration_s=args.duration,
            tolerance_rad=args.tolerance,
            expected_feedback=SAFE_HOME_FEEDBACK,
        )
        print("Arm after safe move:", node.arm_feedback)

        if args.save_rgb:
            saved = node.save_wrist_rgb(args.save_rgb)
            print("Saved wrist RGB:", saved)

        if not args.no_gripper_cycle:
            print(f"Opening gripper to {args.open_position}")
            node.command_gripper(args.open_position, duration_s=args.duration)
            print("Gripper after open:", node.gripper_feedback)

            print(f"Closing gripper to {args.close_position}")
            node.command_gripper(args.close_position, duration_s=args.duration)
            print("Gripper after close:", node.gripper_feedback)

            print(f"Re-opening gripper to {args.open_position}")
            node.command_gripper(args.open_position, duration_s=args.duration)
            print("Gripper after final open:", node.gripper_feedback)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
