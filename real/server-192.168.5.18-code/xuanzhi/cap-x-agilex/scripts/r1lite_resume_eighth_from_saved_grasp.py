#!/usr/bin/env python3
"""Resume a certified pipeline-eight bundle from a separately saved grasp pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_kinematics import (
    R1LiteCombinedCollisionKinematics,
)
from capx.integrations.r1lite.bimanual_relaxed_ik import (
    IsolatedBimanualRelaxedIK,
)
from capx.integrations.r1lite.bimanual_runtime import (
    BimanualRuntime,
    BimanualRuntimeConfig,
)
from capx.integrations.r1lite.bimanual_towel_executor import (
    BimanualTowelExecutor,
)
from capx.integrations.r1lite.capability_certificate import (
    load_capability_certificate,
)
from capx.integrations.r1lite.execution_bundle import sha256_file


LEGACY_PLAN_SCHEMA = "capx.r1lite.eighth_bimanual_recenter.v2"
LEGACY_BUNDLE_SCHEMA = "capx.r1lite.eighth_execution_bundle.v2"
LEGACY_POLICY_SCHEMA = "capx.r1lite.eighth_fixed_policy.v2"
SNAPSHOT_SCHEMA = "capx.r1lite.current_pose_snapshot.v1"
RESUME_ORDER = ("lift", "translate_center", "release_retract", "home")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main() -> None:
    args = parse_args()
    plan = load(args.plan)
    policy = load(args.policy)
    bundle = load(args.bundle)
    snapshot = load(args.snapshot)
    plan_sha = sha256_file(args.plan)
    policy_sha = sha256_file(args.policy)
    bundle_sha = sha256_file(args.bundle)
    certificate_sha = sha256_file(args.certificate)
    audit: dict[str, Any] = {
        "schema": "capx.r1lite.saved_grasp_resume_audit.v1",
        "passed": False,
        "plan_sha256": plan_sha,
        "policy_sha256": policy_sha,
        "execution_bundle_sha256": bundle_sha,
        "capability_certificate_sha256": certificate_sha,
        "snapshot_sha256": sha256_file(args.snapshot),
        "robot_motion_commands_sent": False,
        "home_completed": False,
        "events": [],
    }
    controller = BimanualRuntime(
        config=BimanualRuntimeConfig(arrival_timeout_s=5.0)
    )
    executor: BimanualTowelExecutor | None = None
    try:
        if plan.get("schema") != LEGACY_PLAN_SCHEMA or plan.get("pipeline") != 8:
            raise ValueError("resume requires the approved pipeline-eight v2 plan")
        if bundle.get("schema") != LEGACY_BUNDLE_SCHEMA:
            raise ValueError("resume requires the approved pipeline-eight v2 bundle")
        if policy.get("schema") != LEGACY_POLICY_SCHEMA:
            raise ValueError("resume requires the approved pipeline-eight v2 policy")
        if snapshot.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("saved grasp snapshot schema mismatch")
        load_capability_certificate(
            args.certificate,
            expected_sha256=certificate_sha,
            plan_sha256=plan_sha,
            code_sha256=policy_sha,
            execution_bundle_sha256=bundle_sha,
        )
        grasp = next(
            stage for stage in bundle["stages"] if stage["id"] == "grasp"
        )["waypoints"][-1]
        saved = {
            side: np.asarray(
                snapshot["arms"][side]["joint_positions_rad"], dtype=float
            )
            for side in ("left", "right")
        }
        grasp_errors = {
            side: float(
                np.max(
                    np.abs(
                        saved[side]
                        - np.asarray(grasp[side]["joints_rad"], dtype=float)
                    )
                )
            )
            for side in ("left", "right")
        }
        if any(error > 0.01 for error in grasp_errors.values()):
            raise RuntimeError(
                f"saved pose does not match certified grasp: {grasp_errors}"
            )
        controller.start()
        controller.wait_until_ready(
            timeout_s=10.0, require_tool_pose=True
        )
        current = controller.get_feedback_pair()
        collision = R1LiteCombinedCollisionKinematics(
            table_z_m=float(plan["safety"]["table_z_m"]),
            tcp_to_fingertip_m=float(
                plan["safety"].get("tcp_to_fingertip_m", 0.078)
            ),
        )
        snapshot_errors = {
            side: float(
                np.max(
                    np.abs(
                        np.asarray(current[side], dtype=float) - saved[side]
                    )
                )
            )
            for side in ("left", "right")
        }
        audit["saved_to_certified_grasp_joint_error_rad"] = grasp_errors
        audit["current_to_saved_joint_error_rad"] = snapshot_errors
        if any(error > 0.01 for error in snapshot_errors.values()):
            maximum_delta = max(snapshot_errors.values())
            if maximum_delta > 3.0:
                raise RuntimeError(
                    "direct saved-pose replay exceeds 3.0rad hard gate"
                )
            reports = collision.validate_interpolated_segment(
                np.asarray(current["left"]),
                np.asarray(current["right"]),
                saved["left"],
                saved["right"],
                np.asarray(bundle["initial_state"]["torso_joints_rad"]),
                samples=50,
                fingers_open=True,
            )
            audit["direct_saved_pose_replay"] = {
                "samples": len(reports),
                "maximum_joint_delta_rad": maximum_delta,
                "passed": True,
            }
            controller.execute_joint_motion(
                saved["left"], saved["right"], progress_steps=1
            )
            audit["robot_motion_commands_sent"] = True
            current = controller.get_feedback_pair()
            arrival_errors = {
                side: float(
                    np.max(
                        np.abs(
                            np.asarray(current[side], dtype=float)
                            - saved[side]
                        )
                    )
                )
                for side in ("left", "right")
            }
            audit["saved_pose_arrival_error_rad"] = arrival_errors
            if any(error > 0.01 for error in arrival_errors.values()):
                raise RuntimeError(
                    f"saved-pose arrival gate failed: {arrival_errors}"
                )
        collision.validate(
            np.asarray(current["left"]),
            np.asarray(current["right"]),
            np.asarray(bundle["initial_state"]["torso_joints_rad"]),
            fingers_open=True,
        )
        solver = IsolatedBimanualRelaxedIK(
            controller, bundle["initial_state"]["torso_joints_rad"]
        )
        executor = BimanualTowelExecutor(
            plan=plan,
            plan_sha256=plan_sha,
            bundle=bundle,
            controller=controller,
            solver=solver,
            safety=collision,
        )
        executor.bind_plan(sha256=plan_sha)
        executor._completed = ["grasp"]
        executor.close_both_grippers()
        audit["robot_motion_commands_sent"] = True
        for stage_id in RESUME_ORDER:
            if stage_id == "release_retract":
                executor.open_both_grippers()
            executor.execute_eighth_stage(stage_id=stage_id)
        audit["events"] = executor.trace
        audit["home_completed"] = any(
            event.get("event") == "eighth_stage_completed"
            and event.get("stage_id") == "home"
            for event in executor.trace
        )
        if not audit["home_completed"]:
            raise RuntimeError("resume did not complete home")
        audit["passed"] = True
    except Exception as exc:
        audit["error"] = repr(exc)
        if executor is not None:
            audit["events"] = executor.trace
        try:
            controller.hold()
            audit["common_hold_attempted"] = True
        except Exception as hold_exc:
            audit["common_hold_error"] = repr(hold_exc)
        raise
    finally:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        controller.close()


if __name__ == "__main__":
    main()
