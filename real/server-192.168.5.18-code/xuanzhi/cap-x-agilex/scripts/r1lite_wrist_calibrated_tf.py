#!/usr/bin/env python3
"""Publish the calibrated R1 Lite wrist-camera tree in base_link."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def _transform_to_matrix(transform: Any) -> np.ndarray:
    translation = transform.translation
    rotation = transform.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    ).as_matrix()
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def _pose_to_matrix(pose: Any) -> np.ndarray:
    position = pose.position
    orientation = pose.orientation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        [orientation.x, orientation.y, orientation.z, orientation.w]
    ).as_matrix()
    matrix[:3, 3] = [position.x, position.y, position.z]
    return matrix


def _load_calibration(path: Path) -> np.ndarray:
    payload = yaml.safe_load(path.read_text())
    matrix = np.asarray(payload["T_ee_camera"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise RuntimeError(f"T_ee_camera in {path} must be 4x4")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RuntimeError("T_ee_camera has an invalid homogeneous row")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5):
        raise RuntimeError("T_ee_camera rotation is not orthonormal")
    if np.linalg.det(matrix[:3, :3]) < 0.999:
        raise RuntimeError("T_ee_camera rotation is not right-handed")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        default=(
            "outputs/r1lite/right_touch_hand_eye_calibration/"
            "T_ee_camera.yaml"
        ),
    )
    parser.add_argument(
        "--pose-topic",
        default="/motion_control/pose_ee_arm_right",
    )
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-root-frame", default="camera_wrist_left_link")
    parser.add_argument(
        "--camera-optical-frame",
        default="camera_wrist_right_color_optical_frame",
    )
    args = parser.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformBroadcaster, TransformListener

    ee_from_optical = _load_calibration(Path(args.calibration))
    rclpy.init()
    node = Node("r1lite_wrist_calibrated_tf")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    broadcaster = TransformBroadcaster(node)
    root_from_optical: np.ndarray | None = None
    latest_message: TransformStamped | None = None

    def publish(pose_message: PoseStamped) -> None:
        nonlocal latest_message, root_from_optical
        if pose_message.header.frame_id.strip("/") != args.base_frame.strip("/"):
            node.get_logger().error(
                f"Expected EE pose in {args.base_frame}, "
                f"got {pose_message.header.frame_id!r}"
            )
            return
        if root_from_optical is None:
            try:
                static_tf = buffer.lookup_transform(
                    args.camera_root_frame,
                    args.camera_optical_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
            except Exception as error:
                node.get_logger().warn(
                    f"Waiting for camera static TF: {error}",
                    throttle_duration_sec=2.0,
                )
                return
            root_from_optical = _transform_to_matrix(static_tf.transform)
            node.get_logger().info(
                "Loaded camera static chain "
                f"{args.camera_root_frame} <- {args.camera_optical_frame}"
            )

        base_from_ee = _pose_to_matrix(pose_message.pose)
        base_from_optical = base_from_ee @ ee_from_optical
        base_from_root = base_from_optical @ np.linalg.inv(root_from_optical)
        quaternion = Rotation.from_matrix(base_from_root[:3, :3]).as_quat()

        message = TransformStamped()
        message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = args.base_frame
        message.child_frame_id = args.camera_root_frame
        message.transform.translation.x = float(base_from_root[0, 3])
        message.transform.translation.y = float(base_from_root[1, 3])
        message.transform.translation.z = float(base_from_root[2, 3])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        latest_message = message
        broadcaster.sendTransform(message)

    def republish_latest() -> None:
        if latest_message is None:
            return
        message = TransformStamped()
        message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = latest_message.header.frame_id
        message.child_frame_id = latest_message.child_frame_id
        message.transform = latest_message.transform
        broadcaster.sendTransform(message)

    node.create_subscription(
        PoseStamped,
        args.pose_topic,
        publish,
        qos_profile_sensor_data,
    )
    node.create_timer(0.05, republish_latest)
    node.get_logger().info(
        f"Publishing calibrated {args.base_frame} <- "
        f"{args.camera_root_frame} from {args.pose_topic}"
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
