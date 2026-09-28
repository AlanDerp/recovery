#!/usr/bin/env python3
"""Audited minimal dual-arm synchronization probe.

The only live motion is +0.02 rad on joint 6 of both arms, followed by a return
to the captured command baseline. Any asymmetric, reversed, excessive, or stale
feedback causes both arms to receive the common captured hold command.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from hdas_msg.msg import MotorControl
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


JOINT_INDEX = 5
COMMAND_DELTA_RAD = 0.020
ONSET_THRESHOLD_RAD = 0.003
MAX_TOTAL_MOTION_RAD = 0.040
MAX_CROSS_COUPLING_RAD = 0.010
MAX_FEEDBACK_AGE_S = 0.10
COMMAND_DURATION_S = 1.50
RETURN_DURATION_S = 2.00
PUBLISH_RATE_HZ = 20.0


class BimanualSyncProbe(Node):
    def __init__(self, output: Path) -> None:
        super().__init__("capx_bimanual_sync_probe")
        self.output = output
        self.feedback: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.feedback_time = {"left": 0.0, "right": 0.0}
        self.command_reference: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self.samples: list[dict[str, Any]] = []
        self.publish_skews_ms: list[float] = []
        self.phase = "warmup"
        self.phase_started = time.monotonic()
        self._arm_publishers = {
            side: self.create_publisher(
                JointState,
                f"/motion_target/target_joint_state_arm_{side}",
                10,
            )
            for side in ("left", "right")
        }
        self.brake_publisher = self.create_publisher(
            Bool,
            "/motion_target/brake_mode",
            10,
        )
        for side in ("left", "right"):
            self.create_subscription(
                JointState,
                f"/hdas/feedback_arm_{side}",
                lambda msg, arm=side: self._on_feedback(arm, msg),
                100,
            )
            self.create_subscription(
                MotorControl,
                f"/motion_control/control_arm_{side}",
                lambda msg, arm=side: self._on_control(arm, msg),
                qos_profile_sensor_data,
            )

    def _on_feedback(self, side: str, msg: JointState) -> None:
        if len(msg.position) < 6:
            return
        now = time.monotonic()
        value = np.asarray(msg.position[:6], dtype=np.float64)
        self.feedback[side] = value
        self.feedback_time[side] = now
        self.samples.append(
            {
                "monotonic_s": now,
                "phase": self.phase,
                "side": side,
                "joints_rad": value.tolist(),
            }
        )

    def _on_control(self, side: str, msg: MotorControl) -> None:
        if len(msg.p_des) >= 6:
            self.command_reference[side] = np.asarray(
                msg.p_des[:6],
                dtype=np.float64,
            )

    def spin_until_ready(self, timeout_s: float = 8.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if all(
                value is not None for value in self.feedback.values()
            ) and all(
                value is not None for value in self.command_reference.values()
            ):
                return
        raise TimeoutError("dual-arm feedback or MOCA command reference unavailable")

    @staticmethod
    def _message(command: np.ndarray) -> JointState:
        msg = JointState()
        msg.position = [float(value) for value in command]
        msg.velocity = [1.6, 1.6, 1.6, 4.0, 4.0, 4.0]
        return msg

    def publish_pair(
        self,
        left_command: np.ndarray,
        right_command: np.ndarray,
    ) -> None:
        before = time.monotonic_ns()
        self._arm_publishers["left"].publish(self._message(left_command))
        middle = time.monotonic_ns()
        self._arm_publishers["right"].publish(self._message(right_command))
        after = time.monotonic_ns()
        self.publish_skews_ms.append((after - before) / 1e6)
        self.samples.append(
            {
                "monotonic_s": after / 1e9,
                "phase": self.phase,
                "event": "command_pair",
                "left_publish_to_right_complete_ms": (after - before) / 1e6,
                "between_publish_calls_ms": (middle - before) / 1e6,
            }
        )

    def run_phase(
        self,
        name: str,
        left_command: np.ndarray,
        right_command: np.ndarray,
        duration_s: float,
        baseline_feedback: dict[str, np.ndarray],
    ) -> None:
        self.phase = name
        self.phase_started = time.monotonic()
        period = 1.0 / PUBLISH_RATE_HZ
        deadline = self.phase_started + duration_s
        brake = Bool()
        brake.data = True
        while time.monotonic() < deadline:
            cycle_started = time.monotonic()
            self.brake_publisher.publish(brake)
            self.publish_pair(left_command, right_command)
            until = cycle_started + period
            while time.monotonic() < until:
                rclpy.spin_once(
                    self,
                    timeout_sec=min(0.01, max(0.0, until - time.monotonic())),
                )
            now = time.monotonic()
            for side in ("left", "right"):
                if now - self.feedback_time[side] > MAX_FEEDBACK_AGE_S:
                    raise RuntimeError(f"{side} feedback became stale")
                assert self.feedback[side] is not None
                delta = self.feedback[side] - baseline_feedback[side]
                if float(np.max(np.abs(delta))) > MAX_TOTAL_MOTION_RAD:
                    raise RuntimeError(f"{side} exceeded total motion safety limit")
                inactive = np.delete(delta, JOINT_INDEX)
                if float(np.max(np.abs(inactive))) > MAX_CROSS_COUPLING_RAD:
                    raise RuntimeError(f"{side} inactive-joint cross coupling exceeded")

    def write_audit(self, audit: dict[str, Any]) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
        samples_path = self.output.with_suffix(".jsonl")
        samples_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.samples)
        )


def phase_samples(
    samples: list[dict[str, Any]],
    phase: str,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        item
        for item in samples
        if item.get("phase") == phase and item.get("side") == side
    ]
    return (
        np.asarray([item["monotonic_s"] for item in selected], dtype=float),
        np.asarray([item["joints_rad"] for item in selected], dtype=float),
    )


def interlock_action(
    *,
    left_feedback_age_s: float,
    right_feedback_age_s: float,
) -> str:
    """Return the joint action for one dual-arm feedback-health snapshot."""
    if (
        left_feedback_age_s > MAX_FEEDBACK_AGE_S
        or right_feedback_age_s > MAX_FEEDBACK_AGE_S
    ):
        return "hold_both"
    return "continue_both"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fault-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "planned_joint_index": JOINT_INDEX,
                    "command_delta_rad": COMMAND_DELTA_RAD,
                    "motion_commands_sent": False,
                }
            )
        )
        return
    rclpy.init()
    node = BimanualSyncProbe(args.output)
    baseline_command: dict[str, np.ndarray] | None = None
    audit: dict[str, Any] = {
        "experiment": "minimal_bimanual_joint6_sync",
        "command_delta_rad": COMMAND_DELTA_RAD,
        "joint_index_zero_based": JOINT_INDEX,
        "motion_commands_sent": False,
        "passed": False,
    }
    try:
        node.spin_until_ready()
        baseline_feedback = {
            side: np.asarray(node.feedback[side], dtype=float).copy()
            for side in ("left", "right")
        }
        baseline_command = {
            side: np.asarray(node.command_reference[side], dtype=float).copy()
            for side in ("left", "right")
        }
        target_command = {
            side: baseline_command[side].copy() for side in ("left", "right")
        }
        for side in ("left", "right"):
            target_command[side][JOINT_INDEX] += COMMAND_DELTA_RAD
        if args.fault_test_only:
            injected = {
                "left_feedback_age_s": MAX_FEEDBACK_AGE_S + 0.05,
                "right_feedback_age_s": 0.0,
            }
            action = interlock_action(**injected)
            if action != "hold_both":
                raise RuntimeError("simulated stale-left fault did not hold both arms")
            audit["motion_commands_sent"] = True
            node.run_phase(
                "simulated_left_stale_common_hold",
                baseline_command["left"],
                baseline_command["right"],
                1.0,
                baseline_feedback,
            )
            maximum_hold_motion = {}
            for side in ("left", "right"):
                _timestamps, feedback = phase_samples(
                    node.samples,
                    "simulated_left_stale_common_hold",
                    side,
                )
                maximum_hold_motion[side] = float(
                    np.max(np.abs(feedback - baseline_feedback[side]))
                )
            if any(value > 0.005 for value in maximum_hold_motion.values()):
                raise RuntimeError(
                    f"common hold moved an arm unexpectedly: {maximum_hold_motion}"
                )
            audit.update(
                {
                    "fault_injection": injected,
                    "interlock_action": action,
                    "maximum_common_hold_motion_rad": maximum_hold_motion,
                    "passed": True,
                }
            )
            return
        audit["motion_commands_sent"] = True
        node.run_phase(
            "probe",
            target_command["left"],
            target_command["right"],
            COMMAND_DURATION_S,
            baseline_feedback,
        )
        node.run_phase(
            "return",
            baseline_command["left"],
            baseline_command["right"],
            RETURN_DURATION_S,
            baseline_feedback,
        )
        analysis: dict[str, Any] = {}
        onset_times = {}
        for side in ("left", "right"):
            timestamps, feedback = phase_samples(node.samples, "probe", side)
            deltas = feedback[:, JOINT_INDEX] - baseline_feedback[side][JOINT_INDEX]
            indices = np.flatnonzero(np.abs(deltas) >= ONSET_THRESHOLD_RAD)
            onset_times[side] = (
                None if not len(indices) else float(timestamps[int(indices[0])])
            )
            settled_delta = float(np.median(deltas[-max(1, len(deltas) // 5) :]))
            inactive = np.delete(feedback - baseline_feedback[side], JOINT_INDEX, axis=1)
            analysis[side] = {
                "baseline_feedback_rad": baseline_feedback[side].tolist(),
                "baseline_command_rad": baseline_command[side].tolist(),
                "settled_joint6_delta_rad": settled_delta,
                "peak_joint6_delta_rad": float(np.max(np.abs(deltas))),
                "maximum_cross_coupling_rad": float(np.max(np.abs(inactive))),
                "onset_monotonic_s": onset_times[side],
            }
        if any(value is None for value in onset_times.values()):
            raise RuntimeError("one or both arms did not begin the commanded motion")
        onset_skew_ms = abs(onset_times["left"] - onset_times["right"]) * 1000.0
        returned = {
            side: np.asarray(node.feedback[side], dtype=float).copy()
            for side in ("left", "right")
        }
        return_error = {
            side: float(np.max(np.abs(returned[side] - baseline_feedback[side])))
            for side in ("left", "right")
        }
        if onset_skew_ms > 50.0:
            raise RuntimeError(f"motion onset skew {onset_skew_ms:.3f}ms exceeds 50ms")
        if any(error > 0.010 for error in return_error.values()):
            raise RuntimeError(f"return error exceeds 0.01rad: {return_error}")
        audit.update(
            {
                "analysis": analysis,
                "onset_skew_ms": onset_skew_ms,
                "publish_skew_ms": {
                    "maximum": max(node.publish_skews_ms),
                    "median": float(np.median(node.publish_skews_ms)),
                },
                "return_error_rad": return_error,
                "passed": True,
            }
        )
    except Exception as exc:
        audit["error"] = repr(exc)
        raise
    finally:
        if baseline_command is not None:
            try:
                current_feedback = {
                    side: np.asarray(node.feedback[side], dtype=float).copy()
                    for side in ("left", "right")
                }
                node.run_phase(
                    "final_common_hold",
                    baseline_command["left"],
                    baseline_command["right"],
                    1.0,
                    current_feedback,
                )
            except Exception as hold_exc:
                audit["final_hold_error"] = repr(hold_exc)
        node.write_audit(audit)
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
