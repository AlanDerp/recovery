#!/usr/bin/env python3
"""Distinguish joint settling from true cross-joint command coupling."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml


REPO = Path("/home/nvidia/xuanzhi/cap-x-agilex")
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel


CONFIG = REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
STABILITY_WINDOW_S = 1.0
STABILITY_SPAN_RAD = 0.001
PROBE_DELTA_RAD = -0.01
MAX_NON_TARGET_MOTION_RAD = 0.002
MAX_TOTAL_MOTION_RAD = 0.015


def load_low_level() -> R1LiteRealLowLevel:
    config = yaml.safe_load(CONFIG.read_text())
    values = dict(config["env"]["cfg"]["low_level"])
    values.pop("_target_", None)
    values["r1lite_motion_validation_enabled"] = True
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {key: value for key, value in values.items() if key in signature.parameters}
    return R1LiteRealLowLevel(**kwargs)


def sample_window(
    r1: R1LiteRealLowLevel,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    samples = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        samples.append(r1.get_arm_joints())
        time.sleep(0.02)
    values = np.asarray(samples, dtype=float)
    return values, np.ptp(values, axis=0)


def active_command(r1: R1LiteRealLowLevel) -> np.ndarray:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with r1._lock:
            command = (
                None
                if r1._last_joint_command is None
                else r1._last_joint_command.copy()
            )
        if command is not None:
            return command
        r1._spin_once_or_sleep()
    raise RuntimeError("MOCA p_des unavailable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output exists; use a new run ID")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "status": "running",
        "started_at_unix": time.time(),
        "thresholds": {
            "stability_window_s": STABILITY_WINDOW_S,
            "stability_span_rad": STABILITY_SPAN_RAD,
            "joint6_probe_delta_rad": PROBE_DELTA_RAD,
            "max_non_target_motion_rad": MAX_NON_TARGET_MOTION_RAD,
            "max_total_motion_rad": MAX_TOTAL_MOTION_RAD,
        },
    }
    r1 = load_low_level()
    try:
        before_samples, before_span = sample_window(
            r1,
            STABILITY_WINDOW_S,
        )
        result["pre_probe_first_feedback_rad"] = before_samples[0].tolist()
        result["pre_probe_last_feedback_rad"] = before_samples[-1].tolist()
        result["pre_probe_span_rad"] = before_span.tolist()
        if float(np.max(before_span)) > STABILITY_SPAN_RAD:
            raise RuntimeError(
                "Pre-probe stability gate failed; no command was sent"
            )
        before = before_samples[-1]
        command_before = active_command(r1)
        target = before.copy()
        target[5] += PROBE_DELTA_RAD
        after = r1.move_joints_safe(
            target,
            max_step=abs(PROBE_DELTA_RAD),
            duration_s=0.7,
        )
        command_after = active_command(r1)
        delta = after - before
        non_target = np.abs(delta[:5])
        result["probe"] = {
            "before_feedback_rad": before.tolist(),
            "target_feedback_rad": target.tolist(),
            "after_feedback_rad": after.tolist(),
            "feedback_delta_rad": delta.tolist(),
            "command_before_rad": command_before.tolist(),
            "command_after_rad": command_after.tolist(),
            "command_delta_rad": (
                command_after - command_before
            ).tolist(),
            "joint2_motion_rad": float(delta[1]),
            "joint6_gain": float(delta[5] / PROBE_DELTA_RAD),
            "max_non_target_motion_rad": float(np.max(non_target)),
            "max_total_motion_rad": float(np.max(np.abs(delta))),
        }
        if (
            delta[5] >= 0.0
            or float(np.max(non_target)) > MAX_NON_TARGET_MOTION_RAD
            or float(np.max(np.abs(delta))) > MAX_TOTAL_MOTION_RAD
        ):
            raise RuntimeError("Settled joint6 probe violated safety limits")
        after_samples, after_span = sample_window(
            r1,
            STABILITY_WINDOW_S,
        )
        result["post_probe_first_feedback_rad"] = after_samples[0].tolist()
        result["post_probe_last_feedback_rad"] = after_samples[-1].tolist()
        result["post_probe_span_rad"] = after_span.tolist()
        result["post_probe_drift_rad"] = (
            after_samples[-1] - after_samples[0]
        ).tolist()
        result["classification"] = (
            "previous joint2 movement was unsettled drift, not repeatable "
            "joint6-to-joint2 command coupling"
        )
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "rejected_or_safety_aborted"
        result["error"] = repr(exc)
    finally:
        result["finished_at_unix"] = time.time()
        result["final_feedback_rad"] = r1.get_arm_joints().tolist()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(args.output)
        print(json.dumps(result, indent=2))
        r1.close()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
