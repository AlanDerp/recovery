#!/usr/bin/env python3
"""Record the R1 Lite MOCA joint command path without moving the robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import rclpy
from hdas_msg.msg import MotorControl
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


TOPICS = {
    "capx_target": "/capx/motion_target/target_joint_state_arm_right",
    "vendor_target": "/motion_target/target_joint_state_arm_right",
    "motor_control": "/motion_control/control_arm_right",
    "feedback": "/hdas/feedback_arm_right",
}


def endpoint_dict(info: Any) -> dict[str, Any]:
    qos = info.qos_profile
    return {
        "node_name": info.node_name,
        "node_namespace": info.node_namespace,
        "topic_type": info.topic_type,
        "endpoint_gid": bytes(info.endpoint_gid).hex(),
        "endpoint_type": str(info.endpoint_type),
        "qos": {
            "reliability": str(qos.reliability),
            "durability": str(qos.durability),
            "history": str(qos.history),
            "depth": int(qos.depth),
        },
    }


class MocaAudit(Node):
    def __init__(self) -> None:
        super().__init__("capx_moca_joint_path_audit")
        self.start_monotonic = time.monotonic()
        self.samples: dict[str, list[dict[str, Any]]] = {
            key: [] for key in TOPICS
        }
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        self.create_subscription(
            JointState,
            TOPICS["capx_target"],
            lambda msg: self.on_joint("capx_target", msg),
            qos,
        )
        self.create_subscription(
            JointState,
            TOPICS["vendor_target"],
            lambda msg: self.on_joint("vendor_target", msg),
            qos,
        )
        self.create_subscription(
            MotorControl,
            TOPICS["motor_control"],
            self.on_motor,
            qos,
        )
        self.create_subscription(
            JointState,
            TOPICS["feedback"],
            lambda msg: self.on_joint("feedback", msg),
            qos,
        )

    def elapsed(self) -> float:
        return time.monotonic() - self.start_monotonic

    def on_joint(self, key: str, msg: JointState) -> None:
        if len(self.samples[key]) >= 5000:
            return
        self.samples[key].append(
            {
                "t_s": self.elapsed(),
                "header_frame_id": msg.header.frame_id,
                "header_stamp_ns": (
                    int(msg.header.stamp.sec) * 1_000_000_000
                    + int(msg.header.stamp.nanosec)
                ),
                "name": list(msg.name),
                "position": [float(value) for value in msg.position],
                "velocity": [float(value) for value in msg.velocity],
                "effort": [float(value) for value in msg.effort],
            }
        )

    def on_motor(self, msg: MotorControl) -> None:
        key = "motor_control"
        if len(self.samples[key]) >= 5000:
            return
        self.samples[key].append(
            {
                "t_s": self.elapsed(),
                "header_frame_id": msg.header.frame_id,
                "header_stamp_ns": (
                    int(msg.header.stamp.sec) * 1_000_000_000
                    + int(msg.header.stamp.nanosec)
                ),
                "name": msg.name,
                "p_des": [float(value) for value in msg.p_des],
                "v_des": [float(value) for value in msg.v_des],
                "kp": [float(value) for value in msg.kp],
                "kd": [float(value) for value in msg.kd],
                "t_ff": [float(value) for value in msg.t_ff],
                "mode": int(msg.mode),
            }
        )

    def graph(self) -> dict[str, Any]:
        result = {}
        for key, topic in TOPICS.items():
            result[key] = {
                "topic": topic,
                "publishers": [
                    endpoint_dict(info)
                    for info in self.get_publishers_info_by_topic(topic)
                ],
                "subscriptions": [
                    endpoint_dict(info)
                    for info in self.get_subscriptions_info_by_topic(topic)
                ],
            }
        return result


def stream_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    times = [row["t_s"] for row in samples]
    intervals = np.diff(times)
    return {
        "sample_count": len(samples),
        "first_t_s": times[0] if times else None,
        "last_t_s": times[-1] if times else None,
        "median_rate_hz": (
            float(1.0 / statistics.median(intervals))
            if len(intervals) and statistics.median(intervals) > 0.0
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-active-pdes", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = MocaAudit()
    graph_before = node.graph()
    hold_result = None
    if args.hold_active_pdes:
        initial_deadline = time.monotonic() + 1.5
        while time.monotonic() < initial_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not node.samples["motor_control"] or not node.samples["feedback"]:
            raise RuntimeError("MOCA motor control or arm feedback is unavailable")
        active_pdes = node.samples["motor_control"][-1]["p_des"][:6]
        baseline = np.asarray(
            node.samples["feedback"][-1]["position"][:6],
            dtype=float,
        )
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        publisher = node.create_publisher(
            JointState,
            TOPICS["capx_target"],
            qos,
        )
        message = JointState()
        message.position = active_pdes
        max_drift = 0.0
        aborted = False
        deadline = time.monotonic() + max(0.1, args.duration_s)
        next_publish = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                publisher.publish(message)
                next_publish = now + 0.05
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.samples["feedback"]:
                measured = np.asarray(
                    node.samples["feedback"][-1]["position"][:6],
                    dtype=float,
                )
                max_drift = max(
                    max_drift,
                    float(np.max(np.abs(measured - baseline))),
                )
                if max_drift > 0.003:
                    aborted = True
                    break
        hold_result = {
            "command_semantic": "hold_current_MOCA_motor_p_des",
            "published_position_rad": active_pdes,
            "baseline_feedback_rad": baseline.tolist(),
            "max_feedback_drift_rad": max_drift,
            "aborted": aborted,
            "passed": not aborted and max_drift <= 0.003,
        }
        node.destroy_publisher(publisher)
    else:
        deadline = time.monotonic() + max(0.1, args.duration_s)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    graph_after = node.graph()

    result: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "duration_s": args.duration_s,
        "graph_before": graph_before,
        "graph_after": graph_after,
        "summary": {
            key: stream_summary(rows)
            for key, rows in node.samples.items()
        },
        "samples": node.samples,
    }
    if hold_result is not None:
        result["hold_active_pdes_test"] = hold_result
    motor = node.samples["motor_control"]
    feedback = node.samples["feedback"]
    if motor and feedback and len(motor[-1]["p_des"]) >= 6 and len(
        feedback[-1]["position"]
    ) >= 6:
        p_des = np.asarray(motor[-1]["p_des"][:6], dtype=float)
        measured = np.asarray(feedback[-1]["position"][:6], dtype=float)
        result["latest_feedback_minus_motor_p_des_rad"] = (
            measured - p_des
        ).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    print(json.dumps(result["summary"], indent=2))
    if "latest_feedback_minus_motor_p_des_rad" in result:
        print(
            "feedback_minus_p_des",
            result["latest_feedback_minus_motor_p_des_rad"],
        )
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
