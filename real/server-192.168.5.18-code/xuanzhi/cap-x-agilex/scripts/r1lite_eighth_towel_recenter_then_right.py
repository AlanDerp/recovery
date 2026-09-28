#!/usr/bin/env python3
"""Prepare and execute pipeline eight with integrated right-arm takeover."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from capx.integrations.r1lite.bimanual_relaxed_ik import (  # noqa: E402
    IsolatedBimanualRelaxedIK,
)
from capx.integrations.r1lite.bimanual_runtime import (  # noqa: E402
    BimanualRuntime,
    BimanualRuntimeConfig,
)
from capx.integrations.r1lite.bimanual_towel_executor import (  # noqa: E402
    BimanualTowelExecutor,
)
from capx.integrations.r1lite.capability_certificate import (  # noqa: E402
    load_capability_certificate,
)
from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    load_eighth_execution_bundle,
    sha256_file,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    FULL_URDF,
    LEFT_RELAXED_URDF,
    RELAXED_URDF,
    R1LiteCandidateKinematics,
)
from capx.integrations.r1lite.towel_perception import (  # noqa: E402
    calibration_polygon,
    extract_nearest_towel_corner_grasp_points,
)
from capx.integrations.r1lite.towel_plan import (  # noqa: E402
    build_eighth_bimanual_recenter_plan,
    validate_eighth_bimanual_recenter_plan,
)
from r1lite_seventh_dual_arm_towel_fold import (  # noqa: E402
    load_camera_robot,
    segment_towel,
)
from r1lite_right_joint2_control_offset import (  # noqa: E402
    RightJoint2ControlOffsetMixin,
)


MAXIMUM_TWO_FRAME_DRIFT_M = 0.015
MAXIMUM_KEYPOINT_EXTRAPOLATION_M = 0.15
POLICY_SCHEMA = "capx.r1lite.eighth_fixed_policy.v27"


class RightJoint2ControlOffsetRuntime(
    RightJoint2ControlOffsetMixin, BimanualRuntime
):
    """Pipeline-8 runtime with explicit application-layer compensation."""


HANDOFF_SCHEMA = "capx.r1lite.eighth_to_seventh_handoff.v1"
COMPLETED_HANDOFF_SCHEMA = "capx.r1lite.eighth_to_seventh_handoff.completed.v1"
POLICY_OPERATIONS = (
    ("gripper", "pregrasp_open_40"),
    ("stage", "grasp"),
    ("contact_probe", "both"),
    ("gripper", "close_both"),
    ("stage", "lift"),
    ("stage", "pullback"),
    ("stage", "translate_center"),
    ("gripper", "open_both"),
    ("stage", "release_retract"),
    ("stage", "home"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def build_fixed_policy(plan_sha256: str) -> dict[str, Any]:
    return {
        "schema": POLICY_SCHEMA,
        "plan_sha256": plan_sha256,
        "operations": [
            {"kind": kind, "id": operation} for kind, operation in POLICY_OPERATIONS
        ],
        "online_ik_refinement": True,
        "grasp_refinement_strategy": (
            "independent_preclose_worst_error_first"
        ),
        "initial_grasp_seed_policy": (
            "best_joint_bounded_candidate_without_ik_fk_acceptance"
        ),
        "initial_grasp_seed_maximum_substep_joint_delta_rad": 1.0,
        "joint_command_progress_steps": 5,
        "pregrasp_gripper_position": 40.0,
        "grasp_x_tolerance_m": 0.005,
        "grasp_maximum_inward_x_error_m": 0.005,
        "contact_probe_lock_xy_to_grasp_target": True,
        "contact_probe_freeze_on_first_effort_indication": True,
        "right_grasp_robotward_x_offset_m": -0.005,
        "translation_control_mode": (
            "midpoint_plus_fixed_grasp_relative_vector"
        ),
        "gripper_close_requires_both_tcp_within_m": None,
        "micro_adjustment_completion_policy": (
            "force_continue_after_maximum_iterations"
        ),
        "tcp_and_relative_error_gates_required_after_refinement": False,
        "post_release_policy": "skip_retract_direct_home",
        "collision_policy": "first_phase_collision_disabled",
        "collision_checked_stage_ids": [],
        "second_phase_perception_policy": "fresh_head_camera_reperception",
        "failure_policy": "hold_both_and_abort",
    }


def validate_fixed_policy(
    policy: Mapping[str, Any], *, plan_sha256: str
) -> dict[str, Any]:
    expected = build_fixed_policy(plan_sha256)
    if dict(policy) != expected:
        raise ValueError("eighth policy is not the fixed plan-bound policy")
    return expected


def keypoint_repeatability_m(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> float:
    return max(
        float(
            np.linalg.norm(
                np.asarray(first["keypoints"][name]["base_xyz_m"], dtype=float)
                - np.asarray(second["keypoints"][name]["base_xyz_m"], dtype=float)
            )
        )
        for name in ("center", "left_grasp", "right_grasp")
    )


def expected_tcp_definition(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "left": "left_gripper_tool_z",
        "right": "right_gripper_tool_z",
        "tcp_to_fingertip_m": float(plan["safety"]["tcp_to_fingertip_m"]),
    }


def expected_collision_model() -> dict[str, Any]:
    return {
        "name": "first_phase_collision_disabled",
        "planning_checks_enabled": False,
        "certification_checks_enabled": False,
        "execution_checks_enabled": False,
        "checked_stage_ids": [],
        "sweep_samples_per_segment": 0,
        "restore_at": "fresh_seventh_phase",
    }


def capture_eighth_geometry(
    robot: Any,
    calibration: dict[str, Any],
    *,
    raw_path: Path | None,
    sam_minimum_score: float = 0.15,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    rgb = robot.get_head_rgb("left")
    expected_size = list(calibration.get("image_size", []))
    if expected_size != [rgb.shape[1], rgb.shape[0]]:
        raise RuntimeError(
            "head image does not match homography resolution: "
            f"image={[rgb.shape[1], rgb.shape[0]]}, calibration={expected_size}"
        )
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(raw_path)
    segmentation = segment_towel(
        rgb,
        service_url=str(getattr(robot, "sam3_service_url", "http://127.0.0.1:8114")),
        prompts=("towel",),
        minimum_score=float(sam_minimum_score),
    )
    geometry = extract_nearest_towel_corner_grasp_points(
        segmentation["mask"],
        calibration,
        maximum_extrapolation_m=MAXIMUM_KEYPOINT_EXTRAPOLATION_M,
    )
    return rgb, segmentation, geometry


def render_eighth_preview(
    rgb: np.ndarray,
    mask: np.ndarray,
    geometry: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    preview_path: Path,
    mask_path: Path,
) -> None:
    source = np.asarray(rgb, dtype=np.uint8)
    overlay = source.copy()
    overlay[np.asarray(mask, dtype=bool)] = [255, 80, 80]
    image = Image.fromarray((0.65 * source + 0.35 * overlay).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    polygon = calibration_polygon(dict(calibration))
    draw.line(
        [tuple(point) for point in polygon] + [tuple(polygon[0])],
        fill=(0, 220, 255),
        width=4,
    )
    colors = {
        "center": (255, 255, 255),
        "left_grasp": (0, 255, 0),
        "right_grasp": (80, 120, 255),
    }
    for name, item in geometry["keypoints"].items():
        u, v = item["pixel_uv"]
        radius = 10
        draw.ellipse(
            [u - radius, v - radius, u + radius, v + radius],
            outline=colors[name],
            width=5,
        )
        draw.text((u + 12, v - 12), name, fill=colors[name])
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(mask_path)


def preview(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        args.config,
        args.calibration,
        args.output_plan,
        args.output_policy,
        args.audit_output,
        args.raw_image,
        args.preview_image,
        args.mask_image,
    )
    if any(value is None for value in required):
        raise ValueError("preview requires all config, image, plan, policy and audit paths")
    if not 0.0 < float(args.sam_minimum_score) <= 1.0:
        raise ValueError("SAM3 minimum score must lie in (0, 1]")
    calibration_bytes = args.calibration.read_bytes()
    calibration = json.loads(calibration_bytes)
    robot = load_camera_robot(args.config)
    try:
        first_rgb, first_segmentation, first_geometry = capture_eighth_geometry(
            robot,
            calibration,
            raw_path=None,
            sam_minimum_score=args.sam_minimum_score,
        )
        time.sleep(0.5)
        second_rgb, second_segmentation, second_geometry = capture_eighth_geometry(
            robot,
            calibration,
            raw_path=args.raw_image,
            sam_minimum_score=args.sam_minimum_score,
        )
    finally:
        robot.close()
    repeatability = keypoint_repeatability_m(first_geometry, second_geometry)
    if repeatability > MAXIMUM_TWO_FRAME_DRIFT_M:
        raise RuntimeError(
            f"two-frame lower-contour repeatability failed: {repeatability:.6f}m"
        )
    plan = build_eighth_bimanual_recenter_plan(
        second_geometry,
        table_z_m=float(calibration["plane_z"]),
        first_center_base_m=first_geometry["keypoints"]["center"]["base_xyz_m"],
    )
    plan["perception"]["artifact_bindings"] = {
        "calibration_sha256": sha256_bytes(calibration_bytes),
        "first_image_sha256": sha256_bytes(first_rgb.tobytes()),
        "second_image_sha256": sha256_bytes(second_rgb.tobytes()),
    }
    plan["perception"]["two_frame_repeatability_m"] = repeatability
    plan["perception"]["sam3"] = {
        "first_prompt": first_segmentation["prompt"],
        "first_score": first_segmentation["score"],
        "second_prompt": second_segmentation["prompt"],
        "second_score": second_segmentation["score"],
        "minimum_score": float(args.sam_minimum_score),
    }
    validate_eighth_bimanual_recenter_plan(plan)
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    plan_sha = sha256_file(args.output_plan)
    policy = build_fixed_policy(plan_sha)
    args.output_policy.parent.mkdir(parents=True, exist_ok=True)
    args.output_policy.write_text(json.dumps(policy, indent=2, ensure_ascii=False))
    render_eighth_preview(
        second_rgb,
        second_segmentation["mask"],
        second_geometry,
        calibration,
        preview_path=args.preview_image,
        mask_path=args.mask_image,
    )
    audit = {
        "pipeline": 8,
        "mode": "preview",
        "passed": True,
        "plan_sha256": plan_sha,
        "policy_sha256": sha256_file(args.output_policy),
        "calibration_sha256": sha256_bytes(calibration_bytes),
        "first_image_sha256": sha256_bytes(first_rgb.tobytes()),
        "second_image_sha256": sha256_bytes(second_rgb.tobytes()),
        "keypoint_repeatability_m": repeatability,
        "sam_minimum_score": float(args.sam_minimum_score),
        "robot_motion_commands_sent": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


def _require_hash(path: Path | None, expected: str | None, label: str) -> str:
    if path is None or not expected:
        raise ValueError(f"execute requires {label} path and expected SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"approved {label} SHA-256 mismatch")
    return actual


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.config is None or args.audit_output is None:
        raise ValueError("execute requires config and audit output")
    plan_sha = _require_hash(args.approved_plan, args.expected_plan_sha256, "plan")
    policy_sha = _require_hash(
        args.approved_policy, args.expected_policy_sha256, "policy"
    )
    bundle_sha = _require_hash(
        args.execution_bundle, args.expected_execution_bundle_sha256, "bundle"
    )
    certificate_sha = _require_hash(
        args.capability_certificate,
        args.expected_capability_certificate_sha256,
        "certificate",
    )
    plan = load_object(args.approved_plan)
    validate_eighth_bimanual_recenter_plan(plan)
    validate_fixed_policy(load_object(args.approved_policy), plan_sha256=plan_sha)
    bundle = load_eighth_execution_bundle(
        args.execution_bundle,
        expected_file_sha256=bundle_sha,
        plan_sha256=plan_sha,
        config_sha256=sha256_file(args.config),
        full_urdf_sha256=sha256_file(FULL_URDF),
        left_relaxed_urdf_sha256=sha256_file(LEFT_RELAXED_URDF),
        right_relaxed_urdf_sha256=sha256_file(RELAXED_URDF),
        tcp_definition=expected_tcp_definition(plan),
        collision_model=expected_collision_model(),
    )
    load_capability_certificate(
        args.capability_certificate,
        expected_sha256=certificate_sha,
        plan_sha256=plan_sha,
        code_sha256=policy_sha,
        execution_bundle_sha256=bundle_sha,
    )
    saved_grasp_refinement = None
    if args.saved_grasp_refinement_sequence is not None:
        saved_grasp_refinement = load_object(
            args.saved_grasp_refinement_sequence
        )
        if saved_grasp_refinement.get("schema") != (
            "capx.r1lite.saved_eighth_grasp_refinement.v1"
        ):
            raise ValueError("unexpected saved eighth grasp refinement schema")
        if saved_grasp_refinement.get("plan_sha256") != plan_sha:
            raise ValueError("saved eighth grasp refinement plan binding mismatch")
    original_contact_confirmations = int(
        plan["safety"]["contact_probe_policy"]["confirmation_samples"]
    )
    if args.accept_first_contact_indication:
        plan["safety"]["contact_probe_policy"]["confirmation_samples"] = 1
    audit: dict[str, Any] = {
        "pipeline": 8,
        "mode": "execute",
        "passed": False,
        "plan_sha256": plan_sha,
        "policy_sha256": policy_sha,
        "execution_bundle_sha256": bundle_sha,
        "capability_certificate_sha256": certificate_sha,
        "events": [],
        "operation_events": [],
        "home_completed": False,
        "robot_motion_commands_sent": False,
        "motion_command_pair_count": 0,
        "right_joint2_control_offset_deg": float(
            args.right_joint2_control_offset_deg
        ),
        "joint_arrival_hard_gate": {
            "enabled": bool(np.isfinite(args.arrival_hard_threshold_rad)),
            "threshold_rad": (
                float(args.arrival_hard_threshold_rad)
                if np.isfinite(args.arrival_hard_threshold_rad)
                else None
            ),
        },
        "saved_grasp_refinement_replay": {
            "enabled": saved_grasp_refinement is not None,
            "path": (
                None
                if args.saved_grasp_refinement_sequence is None
                else str(args.saved_grasp_refinement_sequence)
            ),
        },
        "contact_confirmation_runtime_override": {
            "enabled": bool(args.accept_first_contact_indication),
            "artifact_confirmation_samples": original_contact_confirmations,
            "runtime_confirmation_samples": int(
                plan["safety"]["contact_probe_policy"]["confirmation_samples"]
            ),
            "acceptance_rule": (
                "first_threshold_crossing_with_stable_velocity"
                if args.accept_first_contact_indication
                else "artifact_policy"
            ),
        },
    }
    controller = RightJoint2ControlOffsetRuntime(
        np.deg2rad(args.right_joint2_control_offset_deg),
        config=BimanualRuntimeConfig(
            arrival_timeout_s=5.0,
            arrival_soft_threshold_rad=0.02,
            arrival_hard_threshold_rad=float(
                args.arrival_hard_threshold_rad
            ),
            arrival_velocity_threshold_rad_s=0.10,
            arrival_settle_time_s=0.15,
            record_joint_progress_tcp=True,
        )
    )
    runtime: BimanualTowelExecutor | None = None
    try:
        controller.start()
        controller.wait_until_ready(require_tool_pose=True)
        current = controller.get_feedback_pair()
        expected_initial = {
            side: np.asarray(
                bundle["initial_state"][f"{side}_joints_rad"], dtype=float
            ).copy()
            for side in ("left", "right")
        }
        expected_initial["right"][1] += np.deg2rad(
            args.right_joint2_control_offset_deg
        )
        initial_errors = {
            side: float(
                np.max(
                    np.abs(
                        np.asarray(current[side], dtype=float)
                        - expected_initial[side]
                    )
                )
            )
            for side in ("left", "right")
        }
        if any(initial_errors[side] > 0.02 for side in ("left", "right")):
            raise RuntimeError(f"initial joint gate failed: {initial_errors}")
        audit["initial_joint_error_rad"] = initial_errors
        poses = controller.get_tool_pose_pair()
        solver = IsolatedBimanualRelaxedIK(
            controller, bundle["initial_state"]["torso_joints_rad"]
        )
        roundtrip = {}
        roundtrip_pose = {}
        for side in ("left", "right"):
            for attempt in range(1, 4):
                try:
                    probe = np.asarray(
                        solver.solve_candidates(
                            side=side,
                            target_position_base_m=poses[side][0],
                            target_quaternion_xyzw=poses[side][1],
                            reference_joints_rad=current[side],
                            probes=1,
                        )[0],
                        dtype=float,
                    )
                    break
                except TimeoutError:
                    if attempt == 3:
                        raise
                    time.sleep(2.0)
            current_joints = np.asarray(current[side], dtype=float)
            roundtrip[side] = float(
                np.max(np.abs(probe - current_joints))
            )
            kinematics = R1LiteCandidateKinematics(side=side)
            current_fk = kinematics.fk(
                current_joints,
                bundle["initial_state"]["torso_joints_rad"],
            )
            probe_fk = kinematics.fk(
                probe,
                bundle["initial_state"]["torso_joints_rad"],
            )
            roundtrip_pose[side] = {
                "position_error_m": float(
                    np.linalg.norm(
                        probe_fk.position_base_m
                        - current_fk.position_base_m
                    )
                ),
                "orientation_error_rad": float(
                    (
                        Rotation.from_quat(current_fk.quaternion_xyzw).inv()
                        * Rotation.from_quat(probe_fk.quaternion_xyzw)
                    ).magnitude()
                ),
            }
        failed_sides = [
            side
            for side in ("left", "right")
            if roundtrip[side] > 0.05
            and (
                roundtrip_pose[side]["position_error_m"] > 0.015
                or roundtrip_pose[side]["orientation_error_rad"] > 0.05
            )
        ]
        if failed_sides:
            raise RuntimeError(
                "dual-side IK roundtrip preflight failed: "
                f"joint={roundtrip}, pose={roundtrip_pose}, "
                f"failed_sides={failed_sides}"
            )
        audit["ik_roundtrip_error_rad"] = roundtrip
        audit["ik_roundtrip_pose_error"] = roundtrip_pose
        audit["collision_policy"] = "first_phase_collision_disabled"
        audit["grasp_table_collision_checked"] = False
        audit["initial_combined_collision_checked"] = False
        runtime = BimanualTowelExecutor(
            plan=plan,
            plan_sha256=plan_sha,
            bundle=bundle,
            controller=controller,
            solver=solver,
            safety=None,
            saved_eighth_grasp_refinement=saved_grasp_refinement,
        )
        runtime.bind_plan(sha256=plan_sha)
        for kind, operation in POLICY_OPERATIONS:
            if kind == "gripper":
                if operation == "pregrasp_open_40":
                    runtime.set_both_grippers(
                        float(plan["safety"]["pregrasp_gripper_position"])
                    )
                elif operation == "open_both":
                    runtime.open_both_grippers()
                elif operation == "close_both":
                    runtime.close_both_grippers()
                else:
                    raise AssertionError("unknown fixed gripper operation")
            elif kind == "contact_probe":
                if operation != "both":
                    raise AssertionError("unknown contact probe operation")
                runtime.trace.append(
                    {
                        "event": "right_joint2_control_offset_suspended",
                        "reason": "contact_probe_ik_candidates_are_feedback_space",
                    }
                )
                with controller.suspend_right_joint2_control_offset():
                    runtime.execute_eighth_contact_probe()
            else:
                runtime.execute_eighth_stage(stage_id=operation)
            audit["operation_events"].append(
                {"kind": kind, "id": operation, "completed": True}
            )
        audit["events"] = runtime.trace
        audit["home_completed"] = any(
            event.get("event") == "home_joint_gate" and event.get("passed") is True
            for event in runtime.trace
        ) and any(
            event.get("event") == "eighth_stage_completed"
            and event.get("stage_id") == "home"
            for event in runtime.trace
        )
        if not audit["home_completed"]:
            raise RuntimeError("eighth home completion evidence is missing")
        audit["passed"] = True
        return audit
    except Exception as exc:
        audit["error"] = repr(exc)
        if runtime is not None:
            audit["events"] = runtime.trace
        try:
            controller.hold()
            audit["common_hold_attempted"] = True
        except Exception as hold_exc:
            audit["common_hold_error"] = repr(hold_exc)
        raise
    finally:
        controller.close()
        audit["motion_command_pair_count"] = controller.command_pair_count
        audit["joint_arrival_trace"] = controller.joint_arrival_trace
        audit["joint_progress_tcp_trace"] = controller.joint_progress_tcp_trace
        audit["robot_motion_commands_sent"] = controller.command_pair_count > 0
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))


def validate_eighth_execution_audit(
    audit: Mapping[str, Any], *, expected_hashes: Mapping[str, str]
) -> None:
    if audit.get("pipeline") != 8 or audit.get("mode") != "execute":
        raise ValueError("handoff requires an eighth execute audit")
    if audit.get("passed") is not True or audit.get("home_completed") is not True:
        raise ValueError("handoff requires passed eighth execution and completed home")
    for field, expected in expected_hashes.items():
        if audit.get(field) != expected:
            raise ValueError(f"eighth execute audit {field} binding mismatch")
    events = audit.get("events")
    if not isinstance(events, list) or not any(
        isinstance(event, dict)
        and event.get("event") == "home_joint_gate"
        and event.get("passed") is True
        for event in events
    ):
        raise ValueError("eighth execute audit lacks passing home gate evidence")


def build_handoff_manifest(
    *,
    eighth_artifact_hashes: Mapping[str, str],
    eighth_execution_audit_sha256: str,
    seventh_plan_sha256: str,
    seventh_perception_audit_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": HANDOFF_SCHEMA,
        "passed": True,
        "eighth": {
            "artifact_sha256": dict(eighth_artifact_hashes),
            "execution_audit_sha256": eighth_execution_audit_sha256,
        },
        "seventh_reperception": {
            "plan_sha256": seventh_plan_sha256,
            "candidate_plan_sha256": seventh_plan_sha256,
            "perception_audit_sha256": seventh_perception_audit_sha256,
            "fresh_perception_required": True,
            "first_eighth_images_reused": False,
            "collision_policy": "active_arm_table_only",
            "collision_checks_restored": True,
        },
        "right_execution_authorized": False,
        "next_action": "independently certify the new seventh right-arm plan",
    }


def validate_right_execution_audit(
    audit: Mapping[str, Any],
    *,
    plan_sha256: str,
    code_sha256: str,
    execution_bundle_sha256: str,
) -> None:
    expected = {
        "pipeline": 7,
        "mode": "execute",
        "passed": True,
        "plan_sha256": plan_sha256,
        "code_sha256": code_sha256,
        "execution_bundle_sha256": execution_bundle_sha256,
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ValueError(f"right execution audit {field} binding mismatch")
    events = audit.get("events")
    if not isinstance(events, list) or not any(
        isinstance(event, dict) and event.get("event") == "home_completed"
        for event in events
    ):
        raise ValueError("right execution audit lacks home completion evidence")


def build_completed_handoff_manifest(
    *,
    pending_manifest: Mapping[str, Any],
    right_code_sha256: str,
    right_execution_bundle_sha256: str,
    right_capability_certificate_sha256: str,
    right_collision_audit_sha256: str,
    right_execution_audit_sha256: str,
    right_selected_plan_sha256: str | None = None,
    right_candidate_selection_audit_sha256: str | None = None,
) -> dict[str, Any]:
    if pending_manifest.get("schema") != HANDOFF_SCHEMA:
        raise ValueError("unexpected pending handoff manifest schema")
    if pending_manifest.get("passed") is not True:
        raise ValueError("pending handoff manifest did not pass")
    return {
        **dict(pending_manifest),
        "schema": COMPLETED_HANDOFF_SCHEMA,
        "right_certification": {
            "selected_plan_sha256": (
                right_selected_plan_sha256
                or pending_manifest["seventh_reperception"]["plan_sha256"]
            ),
            "code_sha256": right_code_sha256,
            "execution_bundle_sha256": right_execution_bundle_sha256,
            "capability_certificate_sha256": right_capability_certificate_sha256,
            "collision_audit_sha256": right_collision_audit_sha256,
            "passed": True,
        },
        "right_candidate_selection": {
            "audit_sha256": right_candidate_selection_audit_sha256,
            "passed": True,
        },
        "right_execution": {
            "audit_sha256": right_execution_audit_sha256,
            "passed": True,
            "home_completed": True,
        },
        "right_execution_authorized": True,
        "next_action": "complete",
    }


def finalize_handoff(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        args.handoff_manifest,
        args.right_plan,
        args.right_code,
        args.right_execution_bundle,
        args.right_capability_certificate,
        args.right_collision_audit,
        args.right_execution_audit,
        args.right_candidate_selection_audit,
    )
    if any(value is None for value in required):
        raise ValueError("finalize-handoff requires all right-phase artifacts")
    pending = load_object(args.handoff_manifest)
    plan_sha = sha256_file(args.right_plan)
    perception = pending.get("seventh_reperception", {})
    candidate_plan_sha = perception.get(
        "candidate_plan_sha256", perception.get("plan_sha256")
    )
    selection = load_object(args.right_candidate_selection_audit)
    if (
        selection.get("passed") is not True
        or selection.get("source_plan_sha256") != candidate_plan_sha
        or selection.get("selected_plan_sha256") != plan_sha
    ):
        raise ValueError("right contour candidate selection binding mismatch")
    code_sha = sha256_file(args.right_code)
    bundle_sha = sha256_file(args.right_execution_bundle)
    collision = load_object(args.right_collision_audit)
    if (
        collision.get("passed") is not True
        or collision.get("plan_sha256") != plan_sha
        or collision.get("robot_motion_commands_sent") is not False
    ):
        raise ValueError("right collision certification binding did not pass")
    certificate_sha = sha256_file(args.right_capability_certificate)
    load_capability_certificate(
        args.right_capability_certificate,
        expected_sha256=certificate_sha,
        plan_sha256=plan_sha,
        code_sha256=code_sha,
        execution_bundle_sha256=bundle_sha,
    )
    execution = load_object(args.right_execution_audit)
    validate_right_execution_audit(
        execution,
        plan_sha256=plan_sha,
        code_sha256=code_sha,
        execution_bundle_sha256=bundle_sha,
    )
    completed = build_completed_handoff_manifest(
        pending_manifest=pending,
        right_code_sha256=code_sha,
        right_execution_bundle_sha256=bundle_sha,
        right_capability_certificate_sha256=certificate_sha,
        right_collision_audit_sha256=sha256_file(args.right_collision_audit),
        right_execution_audit_sha256=sha256_file(args.right_execution_audit),
        right_selected_plan_sha256=plan_sha,
        right_candidate_selection_audit_sha256=sha256_file(
            args.right_candidate_selection_audit
        ),
    )
    args.handoff_manifest.write_text(
        json.dumps(completed, indent=2, ensure_ascii=False)
    )
    return completed


def handoff(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        args.config,
        args.calibration,
        args.approved_plan,
        args.approved_policy,
        args.execution_bundle,
        args.capability_certificate,
        args.eighth_execution_audit,
        args.handoff_manifest,
        args.right_plan,
        args.right_perception_audit,
        args.right_raw_image,
        args.right_preview_image,
        args.right_mask_image,
    )
    if any(value is None for value in required):
        raise ValueError("handoff requires all eighth inputs and fresh seventh outputs")
    artifacts = {
        "plan_sha256": sha256_file(args.approved_plan),
        "policy_sha256": sha256_file(args.approved_policy),
        "execution_bundle_sha256": sha256_file(args.execution_bundle),
        "capability_certificate_sha256": sha256_file(args.capability_certificate),
    }
    audit = load_object(args.eighth_execution_audit)
    validate_eighth_execution_audit(audit, expected_hashes=artifacts)
    command = [
        sys.executable,
        str(REPO / "scripts/r1lite_seventh_dual_arm_towel_fold.py"),
        "--mode",
        "preview",
        "--config",
        str(args.config),
        "--calibration",
        str(args.calibration),
        "--output-plan",
        str(args.right_plan),
        "--audit-output",
        str(args.right_perception_audit),
        "--raw-image",
        str(args.right_raw_image),
        "--preview-image",
        str(args.right_preview_image),
        "--mask-image",
        str(args.right_mask_image),
        "--sequential-side",
        "right",
    ]
    # No --original-center-json: pipeline seven must use this fresh perception's center.
    subprocess.run(command, cwd=REPO, check=True)
    right_audit = load_object(args.right_perception_audit)
    right_plan = load_object(args.right_plan)
    right_plan_sha = sha256_file(args.right_plan)
    if (
        right_audit.get("pipeline") != 7
        or right_audit.get("mode") != "preview"
        or right_audit.get("plan_sha256") != right_plan_sha
        or right_audit.get("robot_motion_commands_sent") is not False
    ):
        raise RuntimeError("fresh seventh preview audit did not bind the new plan")
    if (
        right_plan.get("pipeline") != 7
        or right_plan.get("safety", {}).get("collision_policy")
        != "active_arm_table_only"
    ):
        raise RuntimeError(
            "seventh handoff did not restore the active-arm table collision policy"
        )
    manifest = build_handoff_manifest(
        eighth_artifact_hashes=artifacts,
        eighth_execution_audit_sha256=sha256_file(args.eighth_execution_audit),
        seventh_plan_sha256=right_plan_sha,
        seventh_perception_audit_sha256=sha256_file(args.right_perception_audit),
    )
    args.handoff_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.handoff_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preview", "execute", "handoff", "finalize-handoff"),
        required=True,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--output-policy", type=Path)
    parser.add_argument("--raw-image", type=Path)
    parser.add_argument("--preview-image", type=Path)
    parser.add_argument("--mask-image", type=Path)
    parser.add_argument("--sam-minimum-score", type=float, default=0.15)
    parser.add_argument("--approved-plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--approved-policy", type=Path)
    parser.add_argument("--expected-policy-sha256")
    parser.add_argument("--execution-bundle", type=Path)
    parser.add_argument("--expected-execution-bundle-sha256")
    parser.add_argument("--capability-certificate", type=Path)
    parser.add_argument("--expected-capability-certificate-sha256")
    parser.add_argument("--eighth-execution-audit", type=Path)
    parser.add_argument("--handoff-manifest", type=Path)
    parser.add_argument("--right-plan", type=Path)
    parser.add_argument("--right-perception-audit", type=Path)
    parser.add_argument("--right-raw-image", type=Path)
    parser.add_argument("--right-preview-image", type=Path)
    parser.add_argument("--right-mask-image", type=Path)
    parser.add_argument("--right-code", type=Path)
    parser.add_argument("--right-execution-bundle", type=Path)
    parser.add_argument("--right-capability-certificate", type=Path)
    parser.add_argument("--right-collision-audit", type=Path)
    parser.add_argument("--right-execution-audit", type=Path)
    parser.add_argument("--right-candidate-selection-audit", type=Path)
    parser.add_argument("--accept-first-contact-indication", action="store_true")
    parser.add_argument("--saved-grasp-refinement-sequence", type=Path)
    parser.add_argument(
        "--right-joint2-control-offset-deg",
        type=float,
        default=float(
            os.environ.get("R1LITE_RIGHT_JOINT2_CONTROL_OFFSET_DEG", "0.0")
        ),
    )
    parser.add_argument(
        "--arrival-hard-threshold-rad",
        type=float,
        default=float(
            os.environ.get("R1LITE_ARRIVAL_HARD_THRESHOLD_RAD", "0.05")
        ),
        help="set to inf to disable the joint-arrival hard gate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preview":
        result = preview(args)
    elif args.mode == "execute":
        result = execute(args)
    elif args.mode == "handoff":
        result = handoff(args)
    elif args.mode == "finalize-handoff":
        result = finalize_handoff(args)
    else:
        raise AssertionError("unreachable eighth pipeline mode")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
