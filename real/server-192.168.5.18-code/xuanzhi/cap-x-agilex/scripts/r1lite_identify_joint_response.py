#!/usr/bin/env python3
"""Safely identify the local command/feedback response of R1 Lite joint 3.

The script is read-only unless ``--execute`` is supplied.  During execution it
holds the five inactive joints at the validated home command and probes joint 3
with independent, bidirectional steps.  Every probe returns to the same
baseline; command deltas are never accumulated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError:  # Allow the pure analysis helpers to be unit tested without ROS.
    rclpy = None
    Node = object
    JointState = Any


ARM_NAME = "right_arm"
ARM_TARGET_TOPIC = "/capx/motion_target/target_joint_state_arm_right"
ARM_FEEDBACK_TOPIC = "/hdas/feedback_arm_right"
JOINT_INDEX = 2
ALLOWED_TARGET_PUBLISHERS = {"moca_adapter"}
SAFE_HOME_COMMAND = np.array(
    [-0.005, 1.063, -1.541, 0.088, 0.003, 0.002], dtype=np.float64
)
SAFE_HOME_FEEDBACK = np.array(
    [0.012766, 1.347447, -1.485500, 0.116383, 0.028511, 0.023830],
    dtype=np.float64,
)


class SafetyAbort(RuntimeError):
    """Raised when a live probe violates a safety gate."""


@dataclass(frozen=True)
class SafetyLimits:
    feedback_timeout_s: float = 0.5
    sample_jump_rad: float = 0.02
    cross_coupling_rad: float = 0.01
    return_error_rad: float = 0.01
    response_threshold_rad: float = 0.001


def analyze_phase(
    samples: list[dict[str, Any]],
    *,
    command_delta_rad: float,
    response_threshold_rad: float = 0.001,
) -> dict[str, Any]:
    """Summarize one probe phase from timestamped six-axis feedback samples."""
    if not samples:
        raise ValueError("Cannot analyze an empty phase")
    feedback = np.asarray([sample["feedback"] for sample in samples], dtype=np.float64)
    timestamps = np.asarray([sample["monotonic_s"] for sample in samples], dtype=np.float64)
    if feedback.ndim != 2 or feedback.shape[1] != 6:
        raise ValueError("Each feedback sample must contain six joints")

    baseline = feedback[0]
    tail_count = max(1, int(np.ceil(len(feedback) * 0.2)))
    settled = np.median(feedback[-tail_count:], axis=0)
    settled_delta = settled - baseline
    target_delta = float(settled_delta[JOINT_INDEX])
    peak_target_delta = float(
        feedback[np.argmax(np.abs(feedback[:, JOINT_INDEX] - baseline[JOINT_INDEX])), JOINT_INDEX]
        - baseline[JOINT_INDEX]
    )
    inactive = np.delete(settled_delta, JOINT_INDEX)
    cross_coupling = float(np.max(np.abs(inactive)))
    gain = (
        target_delta / float(command_delta_rad)
        if abs(command_delta_rad) > 1e-12
        else None
    )

    response_indices = np.flatnonzero(
        np.abs(feedback[:, JOINT_INDEX] - baseline[JOINT_INDEX])
        >= float(response_threshold_rad)
    )
    latency_s = (
        None
        if not len(response_indices)
        else float(timestamps[int(response_indices[0])] - timestamps[0])
    )
    if abs(target_delta) < response_threshold_rad:
        classification = "no_response"
    elif target_delta * command_delta_rad < 0.0:
        classification = "reversed"
    else:
        classification = "responsive"

    return {
        "command_delta_rad": float(command_delta_rad),
        "sample_count": int(len(samples)),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "baseline_feedback": baseline.tolist(),
        "settled_feedback": settled.tolist(),
        "settled_delta": settled_delta.tolist(),
        "target_delta_rad": target_delta,
        "peak_target_delta_rad": peak_target_delta,
        "gain": gain,
        "latency_s": latency_s,
        "cross_coupling_rad": cross_coupling,
        "classification": classification,
    }


def classify_sweep(
    probe_summaries: list[dict[str, Any]],
    *,
    response_threshold_rad: float = 0.001,
) -> dict[str, Any]:
    """Classify a completed bidirectional sweep without hiding raw results."""
    probes = [item for item in probe_summaries if item["phase"] == "probe"]
    if not probes:
        return {"classification": "no_probe_data", "deadband_rad": None}
    reversed_items = [
        item for item in probes if item["analysis"]["classification"] == "reversed"
    ]
    responsive = [
        item
        for item in probes
        if abs(item["analysis"]["target_delta_rad"]) >= response_threshold_rad
        and item["analysis"]["classification"] == "responsive"
    ]
    max_cross_coupling = max(
        float(item["analysis"]["cross_coupling_rad"]) for item in probes
    )
    if reversed_items:
        classification = "direction_reversed"
    elif not responsive:
        classification = "no_response_or_control_path_fault"
    else:
        signs = {int(np.sign(item["command_delta_rad"])) for item in responsive}
        classification = (
            "normal_local_response"
            if signs == {-1, 1}
            else "asymmetric_or_deadband"
        )
    deadband_candidates = [
        abs(float(item["command_delta_rad"]))
        for item in responsive
    ]
    deadband = min(deadband_candidates) if deadband_candidates else None
    gains = [
        float(item["analysis"]["gain"])
        for item in responsive
        if item["analysis"]["gain"] is not None
    ]
    latencies = [
        float(item["analysis"]["latency_s"])
        for item in responsive
        if item["analysis"]["latency_s"] is not None
    ]
    return {
        "classification": classification,
        "deadband_upper_bound_rad": deadband,
        "median_local_gain": None if not gains else float(np.median(gains)),
        "median_latency_s": None if not latencies else float(np.median(latencies)),
        "max_cross_coupling_rad": max_cross_coupling,
        "probe_count": len(probes),
    }


def validate_feedback_sample(
    current: np.ndarray,
    previous: np.ndarray,
    phase_start: np.ndarray,
    *,
    phase: str,
    command_delta_rad: float,
    limits: SafetyLimits,
) -> dict[str, float]:
    """Apply per-sample motion gates independently of ROS."""
    current = np.asarray(current, dtype=np.float64).reshape(6)
    previous = np.asarray(previous, dtype=np.float64).reshape(6)
    phase_start = np.asarray(phase_start, dtype=np.float64).reshape(6)
    sample_jump = float(np.max(np.abs(current - previous)))
    if sample_jump > limits.sample_jump_rad:
        raise SafetyAbort(
            f"Feedback sample jump {sample_jump:.4f} rad exceeds "
            f"{limits.sample_jump_rad:.4f}"
        )
    inactive_indices = [idx for idx in range(6) if idx != JOINT_INDEX]
    cross_coupling = float(
        np.max(np.abs(current[inactive_indices] - phase_start[inactive_indices]))
    )
    if cross_coupling > limits.cross_coupling_rad:
        raise SafetyAbort(
            f"Inactive-joint motion {cross_coupling:.4f} rad exceeds "
            f"{limits.cross_coupling_rad:.4f}"
        )
    target_motion = float(current[JOINT_INDEX] - phase_start[JOINT_INDEX])
    if (
        phase == "probe"
        and abs(target_motion) >= limits.response_threshold_rad
        and target_motion * command_delta_rad < 0.0
    ):
        raise SafetyAbort(
            f"Joint 3 moved opposite the probe: command_delta={command_delta_rad:.4f}, "
            f"feedback_delta={target_motion:.4f}"
        )
    return {
        "sample_jump_rad": sample_jump,
        "cross_coupling_rad": cross_coupling,
        "target_motion_rad": target_motion,
    }


class R1LiteJointResponseIdentifier(Node):
    def __init__(
        self,
        *,
        output_dir: Path,
        limits: SafetyLimits,
        publish_rate_hz: float,
    ) -> None:
        super().__init__("capx_r1lite_joint3_identifier")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = output_dir / "joint3_response_samples.jsonl"
        self.summary_path = output_dir / "joint3_response_summary.json"
        self.raw_path.write_text("", encoding="utf-8")
        self.limits = limits
        self.publish_rate_hz = float(publish_rate_hz)
        self.feedback: np.ndarray | None = None
        self.feedback_monotonic: float | None = None
        self.feedback_sequence = 0
        self.stop_requested = False
        self.arm_pub = self.create_publisher(JointState, ARM_TARGET_TOPIC, 10)
        self.create_subscription(
            JointState, ARM_FEEDBACK_TOPIC, self._on_feedback, 10
        )

    def _on_feedback(self, msg: JointState) -> None:
        if len(msg.position) < 6:
            return
        self.feedback = np.asarray(msg.position[:6], dtype=np.float64)
        self.feedback_monotonic = time.monotonic()
        self.feedback_sequence += 1

    def request_stop(self, *_args: Any) -> None:
        self.stop_requested = True

    def wait_for_feedback(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.feedback is not None:
                return self.feedback.copy()
        raise TimeoutError(f"No feedback received on {ARM_FEEDBACK_TOPIC}")

    def _publish(self, command: np.ndarray) -> None:
        msg = JointState()
        msg.name = [ARM_NAME]
        msg.position = np.asarray(command, dtype=np.float64).reshape(6).tolist()
        self.arm_pub.publish(msg)

    def preflight(self, *, safe_pose_tolerance_rad: float) -> dict[str, Any]:
        first = self.wait_for_feedback()
        start_sequence = self.feedback_sequence
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.feedback_sequence - start_sequence < 5:
            raise RuntimeError("Arm feedback is not updating continuously")
        current = self.feedback.copy()
        home_error = float(np.max(np.abs(current - SAFE_HOME_FEEDBACK)))
        if home_error > safe_pose_tolerance_rad:
            raise SafetyAbort(
                f"Current arm is outside the validated home neighborhood: "
                f"max_error={home_error:.4f} rad, current={current.tolist()}"
            )

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and self.arm_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.arm_pub.get_subscription_count() < 1:
            raise RuntimeError(f"No subscriber is matched to {ARM_TARGET_TOPIC}")

        publishers = self.get_publishers_info_by_topic(ARM_TARGET_TOPIC)
        foreign_publishers = [
            f"{item.node_namespace.rstrip('/')}/{item.node_name}"
            for item in publishers
            if item.node_name != self.get_name()
            and item.node_name not in ALLOWED_TARGET_PUBLISHERS
        ]
        if foreign_publishers:
            raise SafetyAbort(
                f"Other publishers are active on {ARM_TARGET_TOPIC}: "
                f"{foreign_publishers}"
            )
        return {
            "current_feedback": current.tolist(),
            "initial_feedback": first.tolist(),
            "home_error_rad": home_error,
            "target_subscribers": self.arm_pub.get_subscription_count(),
            "foreign_target_publishers": foreign_publishers,
        }

    def _record_sample(
        self,
        *,
        run_id: str,
        phase: str,
        command: np.ndarray,
        command_delta_rad: float,
        phase_feedback_start: np.ndarray,
        previous_feedback: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray]:
        now = time.monotonic()
        if self.feedback is None or self.feedback_monotonic is None:
            raise SafetyAbort("Arm feedback disappeared")
        age = now - self.feedback_monotonic
        if age > self.limits.feedback_timeout_s:
            raise SafetyAbort(f"Arm feedback is stale by {age:.3f}s")
        current = self.feedback.copy()
        metrics = validate_feedback_sample(
            current,
            previous_feedback,
            phase_feedback_start,
            phase=phase,
            command_delta_rad=command_delta_rad,
            limits=self.limits,
        )
        record = {
            "wall_time_s": time.time(),
            "monotonic_s": now,
            "run_id": run_id,
            "phase": phase,
            "command": command.tolist(),
            "command_delta_rad": float(command_delta_rad),
            "feedback": current.tolist(),
            "feedback_age_s": age,
            **metrics,
        }
        with self.raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return record, current

    def run_phase(
        self,
        *,
        run_id: str,
        phase: str,
        command: np.ndarray,
        command_delta_rad: float,
        duration_s: float,
    ) -> list[dict[str, Any]]:
        phase_start = self.wait_for_feedback().copy()
        previous = phase_start.copy()
        samples: list[dict[str, Any]] = []
        deadline = time.monotonic() + duration_s
        interval = 1.0 / self.publish_rate_hz
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise KeyboardInterrupt("Stop requested")
            self._publish(command)
            rclpy.spin_once(self, timeout_sec=min(0.02, interval))
            record, previous = self._record_sample(
                run_id=run_id,
                phase=phase,
                command=command,
                command_delta_rad=command_delta_rad,
                phase_feedback_start=phase_start,
                previous_feedback=previous,
            )
            samples.append(record)
            remaining = interval - 0.02
            if remaining > 0:
                time.sleep(remaining)
        return samples

    def hold_baseline(self, *, run_id: str, duration_s: float) -> list[dict[str, Any]]:
        return self.run_phase(
            run_id=run_id,
            phase="baseline",
            command=SAFE_HOME_COMMAND,
            command_delta_rad=0.0,
            duration_s=duration_s,
        )

    def emergency_return(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        interval = 1.0 / self.publish_rate_hz
        while time.monotonic() < deadline:
            self._publish(SAFE_HOME_COMMAND)
            rclpy.spin_once(self, timeout_sec=min(0.02, interval))
            time.sleep(max(0.0, interval - 0.02))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually publish bounded joint commands; omitted means preflight only.",
    )
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[0.002, 0.004, 0.006, 0.010],
        help="Positive joint-3 command step magnitudes in radians.",
    )
    parser.add_argument("--phase-duration", type=float, default=2.5)
    parser.add_argument("--return-duration", type=float, default=2.5)
    parser.add_argument("--publish-rate", type=float, default=20.0)
    parser.add_argument("--safe-pose-tolerance", type=float, default=0.08)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/r1lite/joint3_identification"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    steps = np.asarray(args.steps, dtype=np.float64)
    if len(steps) == 0 or not np.isfinite(steps).all():
        raise ValueError("--steps must contain finite values")
    if np.any(steps <= 0.0) or np.any(steps > 0.010 + 1e-12):
        raise ValueError("Each step must be in the interval (0, 0.010] rad")
    if args.phase_duration <= 0.0 or args.return_duration <= 0.0:
        raise ValueError("Phase durations must be positive")
    if args.publish_rate < 20.0:
        raise ValueError("--publish-rate must be at least 20 Hz")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are required")
    rclpy.init()
    node = R1LiteJointResponseIdentifier(
        output_dir=args.output_dir,
        limits=SafetyLimits(),
        publish_rate_hz=args.publish_rate,
    )
    signal.signal(signal.SIGINT, node.request_stop)
    signal.signal(signal.SIGTERM, node.request_stop)
    summaries: list[dict[str, Any]] = []
    status = "preflight_only"
    error: str | None = None
    try:
        preflight = node.preflight(safe_pose_tolerance_rad=args.safe_pose_tolerance)
        print("Preflight passed:", json.dumps(preflight, indent=2))
        if not args.execute:
            print("Read-only preflight complete. Pass --execute to run bounded probes.")
            return 0

        status = "running"
        baseline_samples = node.hold_baseline(
            run_id="initial_baseline", duration_s=args.return_duration
        )
        initial_baseline_feedback = np.median(
            np.asarray([item["feedback"] for item in baseline_samples])[-10:], axis=0
        )
        for magnitude in args.steps:
            for direction in (1.0, -1.0):
                delta = float(direction * magnitude)
                run_id = f"joint3_{delta:+.4f}"
                command = SAFE_HOME_COMMAND.copy()
                command[JOINT_INDEX] += delta
                samples = node.run_phase(
                    run_id=run_id,
                    phase="probe",
                    command=command,
                    command_delta_rad=delta,
                    duration_s=args.phase_duration,
                )
                summaries.append(
                    {
                        "run_id": run_id,
                        "phase": "probe",
                        "command_delta_rad": delta,
                        "analysis": analyze_phase(
                            samples,
                            command_delta_rad=delta,
                            response_threshold_rad=node.limits.response_threshold_rad,
                        ),
                    }
                )
                return_samples = node.hold_baseline(
                    run_id=f"{run_id}_return", duration_s=args.return_duration
                )
                returned = np.median(
                    np.asarray([item["feedback"] for item in return_samples])[-10:], axis=0
                )
                return_error = float(
                    np.max(np.abs(returned - initial_baseline_feedback))
                )
                summaries.append(
                    {
                        "run_id": f"{run_id}_return",
                        "phase": "return",
                        "command_delta_rad": 0.0,
                        "return_error_rad": return_error,
                    }
                )
                if return_error > node.limits.return_error_rad:
                    raise SafetyAbort(
                        f"Arm did not return to baseline after {run_id}: "
                        f"error={return_error:.4f} rad"
                    )
        status = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "aborted"
        print(error, file=sys.stderr)
        if args.execute:
            try:
                node.emergency_return(args.return_duration)
            except Exception as return_exc:
                error += f"; emergency return failed: {return_exc!r}"
    finally:
        report = {
            "status": status,
            "error": error,
            "joint_number": 3,
            "joint_index": JOINT_INDEX,
            "safe_home_command": SAFE_HOME_COMMAND.tolist(),
            "safe_home_feedback": SAFE_HOME_FEEDBACK.tolist(),
            "steps_rad": [float(item) for item in args.steps],
            "summaries": summaries,
            "sweep_analysis": classify_sweep(
                summaries,
                response_threshold_rad=node.limits.response_threshold_rad,
            ),
            "raw_samples_path": str(node.raw_path),
        }
        node.summary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("Summary:", node.summary_path)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if status in {"preflight_only", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
