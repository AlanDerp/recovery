#!/usr/bin/env python3
"""Calibrate grasp-table contact from motor-effort residuals.

The probe follows an already certified ingress path, opens the active gripper,
and then lowers the certified grasp TCP in bounded 0.5 mm increments.  It never
closes the gripper.  The probe always retracts along the reverse certified path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_relaxed_ik import (
    IsolatedBimanualRelaxedIK,
)
from capx.integrations.r1lite.bimanual_runtime import (
    BimanualRuntimeConfig,
    R1LiteBimanualRuntime,
)
from capx.integrations.r1lite.execution_bundle import load_execution_bundle
from capx.integrations.r1lite.kinematics import R1LiteCandidateKinematics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-mm", type=float, default=0.5)
    parser.add_argument("--maximum-descent-mm", type=float, default=7.0)
    parser.add_argument("--provisional-stop-residual-nm", type=float, default=0.5)
    return parser.parse_args()


def sample_telemetry(
    controller: R1LiteBimanualRuntime,
    *,
    side: str,
    count: int = 40,
) -> dict[str, list[float] | float]:
    settle_deadline = time.monotonic() + 3.0
    settled_frames = 0
    while time.monotonic() < settle_deadline and settled_frames < 10:
        controller.refresh_feedback(0.01)
        current = controller.get_joint_telemetry(side)
        maximum_velocity = float(
            np.max(np.abs(np.asarray(current["velocity"], dtype=float)))
        )
        settled_frames = settled_frames + 1 if maximum_velocity <= 0.05 else 0
    if settled_frames < 10:
        raise RuntimeError(f"{side} arm did not settle before effort sampling")
    efforts = []
    velocities = []
    for _ in range(count):
        controller.refresh_feedback(0.01)
        sample = controller.get_joint_telemetry(side)
        efforts.append(np.asarray(sample["effort"], dtype=float))
        velocities.append(np.asarray(sample["velocity"], dtype=float))
    effort_array = np.asarray(efforts)
    velocity_array = np.asarray(velocities)
    return {
        "median_effort_nm": np.median(effort_array, axis=0).tolist(),
        "median_abs_velocity_rad_s": np.median(
            np.abs(velocity_array), axis=0
        ).tolist(),
        "maximum_abs_velocity_rad_s": float(np.max(np.abs(velocity_array))),
    }


def main() -> None:
    args = parse_args()
    if args.step_mm <= 0 or args.maximum_descent_mm <= 0:
        raise ValueError("descent distances must be positive")
    bundle = load_execution_bundle(args.bundle)
    stages = {stage["id"]: stage for stage in bundle["stages"]}
    if any(stage_id not in stages for stage_id in ("pregrasp", "align_grasp", "grasp")):
        raise ValueError("bundle lacks the required right-arm ingress stages")
    ingress = []
    for stage_id in ("pregrasp", "align_grasp", "grasp"):
        ingress.extend(stages[stage_id]["right"]["joint_waypoints_rad"])
    initial = bundle["initial_state"]
    audit = {
        "schema": "capx.r1lite.grasp_effort_contact_calibration.v1",
        "bundle": str(args.bundle.resolve()),
        "robot_motion_commands_sent": False,
        "passed": False,
        "probe": [],
        "retraction": [],
    }
    controller = R1LiteBimanualRuntime(
        config=BimanualRuntimeConfig(
            arrival_timeout_s=8.0,
            arrival_hard_threshold_rad=0.06,
        )
    )
    visited: list[list[float]] = []
    at_probe = False
    try:
        controller.start()
        controller.wait_until_ready(require_tool_pose=True)
        current = controller.get_feedback_pair()
        initial_error = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(current[side])
                        - np.asarray(initial[f"{side}_joints_rad"])
                    )
                )
            )
            for side in ("left", "right")
        )
        if initial_error > 0.012:
            raise RuntimeError(
                f"initial joint state mismatch: {initial_error:.6f}rad"
            )
        warmup_deadline = time.monotonic() + 1.0
        while time.monotonic() < warmup_deadline:
            controller.refresh_feedback(0.01)
        controller.open_gripper("right")
        settle_deadline = time.monotonic() + 1.0
        while time.monotonic() < settle_deadline:
            controller.refresh_feedback(0.01)
        left_hold = initial["left_joints_rad"]
        for index, waypoint in enumerate(ingress, 1):
            controller.execute_single_arm_motion(
                side="right",
                target=waypoint,
                inactive_hold=left_hold,
                progress_steps=1,
            )
            visited.append(waypoint)
            audit["probe"].append({"event": "certified_ingress", "index": index})
        at_probe = True
        baseline = sample_telemetry(controller, side="right")
        baseline_effort = np.asarray(baseline["median_effort_nm"])
        previous_effort = baseline_effort
        audit["baseline"] = baseline

        solver = IsolatedBimanualRelaxedIK(
            controller, initial["torso_joints_rad"]
        )
        fk = R1LiteCandidateKinematics(side="right")
        grasp = stages["grasp"]["right"]
        base_position = np.asarray(grasp["target_position_base_m"], dtype=float)
        quaternion = grasp["target_quaternion_xyzw"]
        reference = np.asarray(controller.get_feedback_pair()["right"])
        steps = int(np.ceil(args.maximum_descent_mm / args.step_mm))
        detected = None
        for index in range(1, steps + 1):
            descent_mm = min(index * args.step_mm, args.maximum_descent_mm)
            target_position = base_position.copy()
            target_position[2] -= descent_mm / 1000.0
            candidates = solver.solve_candidates(
                side="right",
                target_position_base_m=target_position,
                target_quaternion_xyzw=quaternion,
                reference_joints_rad=reference,
                probes=8,
            )
            ranked = []
            for candidate in candidates:
                joints = np.asarray(candidate, dtype=float)
                pose = fk.fk(joints, initial["torso_joints_rad"])
                position_error = float(
                    np.linalg.norm(pose.position_base_m - target_position)
                )
                joint_delta = float(np.max(np.abs(joints - reference)))
                if position_error <= 0.015 and joint_delta <= 0.25:
                    ranked.append((position_error, joint_delta, joints, pose))
            if not ranked:
                raise RuntimeError(
                    f"no bounded IK candidate at descent {descent_mm:.3f}mm"
                )
            _, _, selected, pose = min(ranked, key=lambda item: item[:2])
            controller.execute_single_arm_motion(
                side="right",
                target=selected,
                inactive_hold=left_hold,
                progress_steps=1,
            )
            reference = selected
            telemetry = sample_telemetry(controller, side="right")
            residual = (
                np.asarray(telemetry["median_effort_nm"]) - baseline_effort
            )
            incremental = (
                np.asarray(telemetry["median_effort_nm"]) - previous_effort
            )
            previous_effort = np.asarray(telemetry["median_effort_nm"])
            maximum_residual = float(np.max(np.abs(residual)))
            maximum_incremental = float(np.max(np.abs(incremental)))
            record = {
                "event": "descent_sample",
                "descent_mm": descent_mm,
                "target_tcp_position_base_m": target_position.tolist(),
                "fk_tcp_position_base_m": pose.position_base_m.tolist(),
                "effort_residual_nm": residual.tolist(),
                "maximum_abs_effort_residual_nm": maximum_residual,
                "incremental_effort_change_nm": incremental.tolist(),
                "maximum_abs_incremental_effort_change_nm": maximum_incremental,
                **telemetry,
            }
            audit["probe"].append(record)
            if (
                descent_mm >= 7.5
                and maximum_incremental >= args.provisional_stop_residual_nm
            ):
                detected = record
                break
            if maximum_residual >= 5.0:
                raise RuntimeError("absolute effort residual emergency stop")
        audit["provisional_contact"] = detected
        audit["passed"] = detected is not None
        if detected is None:
            raise RuntimeError("bounded probe found no calibrated effort transition")
    finally:
        try:
            if at_probe:
                controller.execute_single_arm_motion(
                    side="right",
                    target=stages["grasp"]["right"]["joints_rad"],
                    inactive_hold=initial["left_joints_rad"],
                    progress_steps=1,
                )
            for index, waypoint in enumerate(reversed(visited), 1):
                controller.execute_single_arm_motion(
                    side="right",
                    target=waypoint,
                    inactive_hold=initial["left_joints_rad"],
                    progress_steps=1,
                )
                audit["retraction"].append(
                    {"event": "reverse_certified_waypoint", "index": index}
                )
            if visited:
                controller.execute_single_arm_motion(
                    side="right",
                    target=initial["right_joints_rad"],
                    inactive_hold=initial["left_joints_rad"],
                    progress_steps=1,
                )
                audit["retraction"].append({"event": "home"})
        except Exception as exc:
            audit["retraction_error"] = repr(exc)
            try:
                controller.hold()
            except Exception:
                pass
        finally:
            audit["robot_motion_commands_sent"] = (
                controller.command_pair_count > 0
            )
            controller.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
