#!/usr/bin/env python3
"""Independently verify an R1 Lite nominal plan without ROS or robot motion."""

from __future__ import annotations

import argparse
import hashlib
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

from scripts.r1lite_audit_tcp_frames import (
    FULL_URDF,
    RELAXED_URDF,
    UrdfChain,
    arm_joint_map,
    full_joint_map,
    pose_matrix,
    residual,
)
from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel


SETTINGS = Path(
    "/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/"
    "R1_Lite/settings_right.yaml"
)
LOW_LEVEL_CONFIG = (
    REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
)
MAX_POSITION_ERROR_M = 0.01
MAX_ORIENTATION_ERROR_RAD = 0.05
MAX_INCREMENTAL_JOINT_DELTA_RAD = 0.2
MAX_DUAL_URDF_POSITION_ERROR_M = 1e-8
MAX_DUAL_URDF_ORIENTATION_ERROR_RAD = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_limits() -> list[tuple[float, float]]:
    root = ET.parse(FULL_URDF).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    result = []
    for index in range(1, 7):
        limit = joints[f"right_arm_joint{index}"].find("limit")
        assert limit is not None
        result.append(
            (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        )
    return result


def pose_error(target: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    return residual(target, actual)


def run_live_gate(
    *,
    plan_path: Path,
    verified_report_path: Path,
    output_path: Path,
    run_id: str,
) -> int:
    if output_path.exists():
        raise FileExistsError("Live gate output exists; use a new run ID")
    plan = json.loads(plan_path.read_text())
    verified = json.loads(verified_report_path.read_text())
    errors = []
    if verified.get("status") != "verified_for_single_step":
        errors.append("Independent verifier status is not verified_for_single_step")
    if verified.get("plan_sha256") != sha256(plan_path):
        errors.append("Verifier report is bound to a different plan SHA-256")
    run_dir = REPO / "outputs/r1lite_audit_runs" / run_id
    if not (run_dir / "before/manifest.json").exists():
        errors.append("Governed run has no before snapshot")
    anchors = plan.get("anchors", [])
    if len(anchors) < 2 or not anchors[1].get("microsteps"):
        errors.append("Plan has no independently verified first executable microstep")
    else:
        initial = np.asarray(plan["initial_arm_feedback_rad"], dtype=float)
        first_solution = np.asarray(
            anchors[1]["microsteps"][0][
                "solution_feedback_joints_rad"
            ],
            dtype=float,
        )
        first_delta = float(np.max(np.abs(first_solution - initial)))
        result_first_delta = first_delta
        if first_delta >= 0.02:
            errors.append(
                "First executable microstep exceeds the <0.02rad release limit"
            )
    result: dict[str, object] = {
        "status": "rejected_for_execution",
        "run_id": run_id,
        "plan_path": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "verified_report_path": str(verified_report_path),
        "errors": errors,
    }
    if "result_first_delta" in locals():
        result["first_microstep_joint_delta_rad"] = result_first_delta
    if not errors:
        config = yaml.safe_load(LOW_LEVEL_CONFIG.read_text())
        values = dict(config["env"]["cfg"]["low_level"])
        values.pop("_target_", None)
        signature = inspect.signature(R1LiteRealLowLevel.__init__)
        kwargs = {
            key: value
            for key, value in values.items()
            if key in signature.parameters
        }
        r1 = R1LiteRealLowLevel(**kwargs)
        try:
            samples = []
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                samples.append(r1.get_arm_joints())
                time.sleep(0.02)
            values_array = np.asarray(samples)
            span = np.ptp(values_array, axis=0)
            current_arm = values_array[-1]
            current_torso = r1._base_from_torso_link3()[1]
            seed_arm = np.asarray(plan["initial_arm_feedback_rad"], dtype=float)
            seed_torso = np.asarray(
                plan["initial_torso_feedback_rad"],
                dtype=float,
            )
            arm_seed_error = float(np.max(np.abs(current_arm - seed_arm)))
            torso_seed_error = float(
                np.max(np.abs(current_torso - seed_torso))
            )
            topic = r1._arm_pub.topic_name
            subscribers = r1._node.get_subscriptions_info_by_topic(topic)
            publishers = r1._node.get_publishers_info_by_topic(topic)
            subscriber_names = sorted(
                {info.node_name for info in subscribers}
            )
            publisher_names = sorted(
                {info.node_name for info in publishers}
            )
            result.update(
                {
                    "feedback_span_rad": span.tolist(),
                    "current_arm_feedback_rad": current_arm.tolist(),
                    "current_torso_feedback_rad": current_torso.tolist(),
                    "arm_seed_error_rad": arm_seed_error,
                    "torso_seed_error_rad": torso_seed_error,
                    "arm_topic": topic,
                    "subscriber_names": subscriber_names,
                    "publisher_names": publisher_names,
                }
            )
            if float(np.max(span)) >= 0.001:
                errors.append("Arm feedback is not stable below 0.001rad span")
            if arm_seed_error >= 0.005:
                errors.append("Current arm feedback does not match plan seed")
            if torso_seed_error >= 0.005:
                errors.append("Current torso feedback does not match plan seed")
            if subscriber_names.count("moca_adapter") != 1:
                errors.append("MOCA is not the unique expected arm subscriber")
            forbidden = {
                "homogeneous_teleoperation",
                "relaxed_ik_right",
            }
            if forbidden.intersection(publisher_names):
                errors.append("Competing arm command publisher is present")
        finally:
            r1.close()
    if not errors:
        result["status"] = "live_gate_passed_for_first_microstep"
    result["errors"] = errors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(output_path)
    print(result["status"])
    print("\n".join(errors))
    return 0 if not errors else 1


def verify_pose(
    *,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    joints: np.ndarray,
    torso: np.ndarray,
    relaxed_chain: UrdfChain,
    full_chain: UrdfChain,
) -> dict[str, object]:
    target = pose_matrix(target_position, target_quaternion)
    torso_map = {
        f"torso_joint{index + 1}": float(value)
        for index, value in enumerate(torso)
    }
    base_from_torso = full_chain.fk(
        "base_link",
        "torso_link3",
        torso_map,
    )
    relaxed_fk = base_from_torso @ relaxed_chain.fk(
        "torso_link3",
        "right_gripper_tool_z",
        arm_joint_map(joints),
    )
    full_fk = full_chain.fk(
        "base_link",
        "right_gripper_tool_z",
        full_joint_map(joints, torso),
    )
    relaxed_error = pose_error(target, relaxed_fk)
    full_error = pose_error(target, full_fk)
    dual_error = pose_error(relaxed_fk, full_fk)
    return {
        "target_matrix": target.tolist(),
        "relaxed_urdf_fk_matrix": relaxed_fk.tolist(),
        "full_urdf_fk_matrix": full_fk.tolist(),
        "target_vs_relaxed": relaxed_error,
        "target_vs_full": full_error,
        "relaxed_vs_full": dual_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--live-gate", action="store_true")
    parser.add_argument("--verified-report", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.live_gate:
        if args.verified_report is None or args.run_id is None:
            parser.error("--live-gate requires --verified-report and --run-id")
        return run_live_gate(
            plan_path=args.plan,
            verified_report_path=args.verified_report,
            output_path=args.report,
            run_id=args.run_id,
        )
    if args.report.exists():
        raise FileExistsError("Verifier report exists; use a new output path")
    plan = json.loads(args.plan.read_text())
    report: dict[str, object] = {
        "verifier_schema_version": 1,
        "plan_path": str(args.plan),
        "plan_sha256": sha256(args.plan),
        "status": "rejected_for_execution",
        "thresholds": {
            "max_position_error_m": MAX_POSITION_ERROR_M,
            "max_orientation_error_rad": MAX_ORIENTATION_ERROR_RAD,
            "max_incremental_joint_delta_rad": (
                MAX_INCREMENTAL_JOINT_DELTA_RAD
            ),
            "max_dual_urdf_position_error_m": (
                MAX_DUAL_URDF_POSITION_ERROR_M
            ),
            "max_dual_urdf_orientation_error_rad": (
                MAX_DUAL_URDF_ORIENTATION_ERROR_RAD
            ),
        },
        "errors": [],
        "warnings": [],
        "anchors": [],
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]
    if plan.get("schema_version") != "r1lite.nominal_plan.v2":
        errors.append("Plan schema is not r1lite.nominal_plan.v2")
    if plan.get("status") != "candidate_unverified":
        errors.append("Candidate plan status must be candidate_unverified")
    if plan.get("pose_semantic") != "T_base_right_gripper_tool_z":
        errors.append("Plan pose semantic is not T_base_right_gripper_tool_z")
    if plan.get("quaternion_order") != "xyzw":
        errors.append("Plan quaternion order is not xyzw")

    expected_sources = {
        "relaxed_ik_settings": SETTINGS,
        "relaxed_ik_urdf": RELAXED_URDF,
        "full_r1lite_urdf": FULL_URDF,
        "low_level_config": LOW_LEVEL_CONFIG,
    }
    actual_hashes = {
        name: sha256(path) for name, path in expected_sources.items()
    }
    report["actual_source_sha256"] = actual_hashes
    declared_hashes = plan.get("source_sha256", {})
    for name, actual in actual_hashes.items():
        if declared_hashes.get(name) != actual:
            errors.append(f"Source hash mismatch: {name}")

    try:
        torso = np.asarray(
            plan["initial_torso_feedback_rad"],
            dtype=float,
        ).reshape(3)
        initial_arm = np.asarray(
            plan["initial_arm_feedback_rad"],
            dtype=float,
        ).reshape(6)
        anchors = plan["anchors"]
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"Missing or malformed plan provenance: {exc}")
        torso = np.zeros(3)
        initial_arm = np.zeros(6)
        anchors = []

    relaxed_chain = UrdfChain(RELAXED_URDF)
    full_chain = UrdfChain(FULL_URDF)
    limits = arm_limits()
    previous_solution = initial_arm
    for anchor_index, anchor in enumerate(anchors):
        anchor_errors = []
        anchor_report: dict[str, object] = {
            "index": anchor_index,
            "id": anchor.get("id"),
            "errors": anchor_errors,
        }
        try:
            position = np.asarray(
                anchor["target_position_base_m"],
                dtype=float,
            ).reshape(3)
            quaternion = np.asarray(
                anchor["target_tool_z_quaternion_xyzw"],
                dtype=float,
            ).reshape(4)
            quaternion_norm = float(np.linalg.norm(quaternion))
            if abs(quaternion_norm - 1.0) > 1e-6:
                anchor_errors.append(
                    f"Quaternion norm is {quaternion_norm:.9f}"
                )
            solution = np.asarray(
                anchor["solution_feedback_joints_rad"],
                dtype=float,
            ).reshape(6)
            seed = np.asarray(
                anchor["segment_seed_feedback_joints_rad"],
                dtype=float,
            ).reshape(6)
            if float(np.max(np.abs(seed - previous_solution))) > 1e-9:
                anchor_errors.append(
                    "Segment seed does not equal previous anchor solution"
                )
            for joint_index, (value, (lower, upper)) in enumerate(
                zip(solution, limits)
            ):
                if value < lower - 1e-9 or value > upper + 1e-9:
                    overrun = max(lower - value, value - upper)
                    if anchor_index == 0 and overrun <= 0.01:
                        warnings.append(
                            f"Observed initial joint {joint_index + 1} is "
                            f"{overrun:.6f}rad outside the nominal URDF limit; "
                            "this current-state anchor is not executable"
                        )
                    else:
                        anchor_errors.append(
                            f"Joint {joint_index + 1} outside URDF limits"
                        )
            pose_check = verify_pose(
                target_position=position,
                target_quaternion=quaternion,
                joints=solution,
                torso=torso,
                relaxed_chain=relaxed_chain,
                full_chain=full_chain,
            )
            anchor_report["independent_fk"] = pose_check
            target_error = pose_check["target_vs_relaxed"]
            dual_error = pose_check["relaxed_vs_full"]
            if target_error["translation_norm_m"] >= MAX_POSITION_ERROR_M:
                anchor_errors.append("Anchor FK position error exceeds limit")
            if target_error["rotation_error_rad"] >= MAX_ORIENTATION_ERROR_RAD:
                anchor_errors.append(
                    "Anchor FK orientation error exceeds limit"
                )
            if (
                dual_error["translation_norm_m"]
                >= MAX_DUAL_URDF_POSITION_ERROR_M
                or dual_error["rotation_error_rad"]
                >= MAX_DUAL_URDF_ORIENTATION_ERROR_RAD
            ):
                anchor_errors.append("Dual-URDF FK disagreement")

            microsteps = anchor.get("microsteps", [])
            if anchor_index > 0 and not microsteps:
                anchor_errors.append("Non-initial anchor has no microstep path")
            micro_previous = seed
            step_reports = []
            for step_index, step in enumerate(microsteps):
                step_solution = np.asarray(
                    step["solution_feedback_joints_rad"],
                    dtype=float,
                ).reshape(6)
                step_seed = np.asarray(
                    step["solver_seed_feedback_joints_rad"],
                    dtype=float,
                ).reshape(6)
                if float(np.max(np.abs(step_seed - micro_previous))) > 1e-9:
                    anchor_errors.append(
                        f"Microstep {step_index} seed discontinuity"
                    )
                incremental_delta = float(
                    np.max(np.abs(step_solution - micro_previous))
                )
                if incremental_delta >= MAX_INCREMENTAL_JOINT_DELTA_RAD:
                    anchor_errors.append(
                        f"Microstep {step_index} joint delta exceeds limit"
                    )
                step_check = verify_pose(
                    target_position=np.asarray(
                        step["target_position_base_m"],
                        dtype=float,
                    ).reshape(3),
                    target_quaternion=np.asarray(
                        step["target_tool_z_quaternion_xyzw"],
                        dtype=float,
                    ).reshape(4),
                    joints=step_solution,
                    torso=torso,
                    relaxed_chain=relaxed_chain,
                    full_chain=full_chain,
                )
                step_error = step_check["target_vs_relaxed"]
                if (
                    step_error["translation_norm_m"]
                    >= MAX_POSITION_ERROR_M
                    or step_error["rotation_error_rad"]
                    >= MAX_ORIENTATION_ERROR_RAD
                ):
                    anchor_errors.append(
                        f"Microstep {step_index} FK residual exceeds limit"
                    )
                step_reports.append(
                    {
                        "index": step_index,
                        "incremental_joint_delta_rad": incremental_delta,
                        "independent_fk": step_check,
                    }
                )
                micro_previous = step_solution
            if microsteps and float(
                np.max(np.abs(micro_previous - solution))
            ) > 1e-9:
                anchor_errors.append(
                    "Final microstep does not equal anchor solution"
                )
            anchor_report["microsteps"] = step_reports
            previous_solution = solution
        except (KeyError, TypeError, ValueError) as exc:
            anchor_errors.append(f"Malformed anchor: {exc}")
        if anchor_errors:
            errors.extend(
                f"anchor[{anchor_index}]: {message}"
                for message in anchor_errors
            )
        report["anchors"].append(anchor_report)

    if not anchors:
        errors.append("Plan contains no anchors")
    if not errors:
        report["status"] = "verified_for_single_step"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.report)
    print(report["status"])
    print("\n".join(errors))
    return 0 if report["status"] == "verified_for_single_step" else 1


if __name__ == "__main__":
    raise SystemExit(main())
