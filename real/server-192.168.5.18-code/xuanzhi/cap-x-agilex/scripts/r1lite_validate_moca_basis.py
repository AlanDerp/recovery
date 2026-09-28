#!/usr/bin/env python3
"""Safely identify the local MOCA command-to-feedback response matrix."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml


REPO = Path("/home/nvidia/xuanzhi/cap-x-agilex")
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel


CONFIG = REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
URDF = Path(
    "/home/nvidia/galaxea/install/r1lite_urdf/share/r1lite_urdf/urdf/"
    "r1lite_2026.urdf"
)


def load_low_level() -> R1LiteRealLowLevel:
    config = yaml.safe_load(CONFIG.read_text())
    values = dict(config["env"]["cfg"]["low_level"])
    values.pop("_target_", None)
    values["r1lite_motion_validation_enabled"] = True
    values["r1lite_adaptive_joint_control_enabled"] = True
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {key: value for key, value in values.items() if key in signature.parameters}
    return R1LiteRealLowLevel(**kwargs)


def arm_joint_limits() -> list[tuple[float, float]]:
    root = ET.parse(URDF).getroot()
    by_name = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    result = []
    for index in range(1, 7):
        limit = by_name[f"right_arm_joint{index}"].find("limit")
        assert limit is not None
        result.append(
            (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-rad", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-delta-rad", type=float, default=0.02)
    args = parser.parse_args()

    result: dict[str, object] = {
        "timestamp_unix": time.time(),
        "status": "running",
        "delta_rad": args.delta_rad,
        "tests": [],
    }
    r1 = load_low_level()
    try:
        baseline = r1.get_arm_joints()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with r1._lock:
                active_command = (
                    None
                    if r1._last_joint_command is None
                    else r1._last_joint_command.copy()
                )
            if active_command is not None:
                break
            r1._spin_once_or_sleep()
        if active_command is None:
            raise RuntimeError("MOCA active p_des was not observed")

        zero_after = r1.move_joints_safe(
            baseline,
            max_step=args.delta_rad,
            duration_s=0.5,
        )
        zero_drift = float(np.max(np.abs(zero_after - baseline)))
        result["initial_feedback_rad"] = baseline.tolist()
        result["initial_moca_p_des_rad"] = active_command.tolist()
        result["zero_hold_max_drift_rad"] = zero_drift
        if zero_drift > 0.003:
            raise RuntimeError(
                f"zero-motion hold drifted {zero_drift:.4f}rad"
            )

        response_rows = []
        limits = arm_joint_limits()
        for joint_index in range(6):
            before = r1.get_arm_joints()
            with r1._lock:
                command_before = r1._last_joint_command.copy()
            lower, upper = limits[joint_index]
            direction = (
                -1.0
                if command_before[joint_index] + args.delta_rad
                > upper - 1e-6
                else 1.0
            )
            target = before.copy()
            target[joint_index] += direction * args.delta_rad
            after = r1.move_joints_safe(
                target,
                max_step=args.delta_rad,
                duration_s=0.5,
            )
            delta = after - before
            non_target = np.delete(np.abs(delta), joint_index)
            row = {
                "joint_index": joint_index,
                "direction": direction,
                "command_limit_rad": [lower, upper],
                "command_before_rad": command_before.tolist(),
                "before_rad": before.tolist(),
                "target_feedback_rad": target.tolist(),
                "after_rad": after.tolist(),
                "feedback_delta_rad": delta.tolist(),
                "target_gain": float(
                    delta[joint_index] / (direction * args.delta_rad)
                ),
                "max_non_target_motion_rad": float(np.max(non_target)),
                "max_total_motion_rad": float(np.max(np.abs(delta))),
            }
            result["tests"].append(row)
            if (
                delta[joint_index] * direction <= 0.0
                or float(np.max(non_target)) > 0.002
                or float(np.max(np.abs(delta))) > 0.015
            ):
                result["status"] = "safety_aborted"
                result["abort_reason"] = (
                    f"joint {joint_index + 1} response violated basis limits"
                )
                break
            returned = r1.move_joints_safe(
                before,
                max_step=args.delta_rad,
                duration_s=0.5,
            )
            return_error = float(np.max(np.abs(returned - before)))
            row["return_feedback_rad"] = returned.tolist()
            row["return_error_rad"] = return_error
            if return_error > 0.005:
                result["status"] = "safety_aborted"
                result["abort_reason"] = (
                    f"joint {joint_index + 1} failed to return"
                )
                break
            response_rows.append(delta / (direction * args.delta_rad))
        else:
            matrix = np.asarray(response_rows, dtype=float).T
            result["response_matrix_feedback_per_command"] = matrix.tolist()
            before = r1.get_arm_joints()
            target = before.copy()
            target[5] += args.final_delta_rad
            after = r1.move_joints_safe(
                target,
                max_step=args.delta_rad,
                duration_s=0.6,
            )
            delta = after - before
            result["final_joint6_probe"] = {
                "requested_delta_rad": args.final_delta_rad,
                "feedback_delta_rad": delta.tolist(),
                "max_non_target_motion_rad": float(
                    np.max(np.abs(delta[:5]))
                ),
            }
            if (
                delta[5] <= 0.0
                or float(np.max(np.abs(delta[:5]))) > 0.003
                or float(np.max(np.abs(delta))) > 0.03
            ):
                result["status"] = "safety_aborted"
                result["abort_reason"] = "final 0.02rad probe violated limits"
            else:
                returned = r1.move_joints_safe(
                    before,
                    max_step=args.delta_rad,
                    duration_s=0.6,
                )
                result["final_joint6_probe"]["return_error_rad"] = float(
                    np.max(np.abs(returned - before))
                )
                result["status"] = "passed"
    except Exception as exc:
        if result["status"] == "running":
            result["status"] = "failed"
        result["error"] = repr(exc)
    finally:
        result["final_feedback_rad"] = r1.get_arm_joints().tolist()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(args.output)
        print(json.dumps(result, indent=2))
        r1.close()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
