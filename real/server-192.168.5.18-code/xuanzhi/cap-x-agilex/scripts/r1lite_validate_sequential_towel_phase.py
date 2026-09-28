#!/usr/bin/env python3
"""Offline IK and swept-collision certification for one sequential towel phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteBimanualKinematics,
    R1LiteTableCollisionKinematics,
    strict_topdown_edge_rotation,
    tcp_position_for_fingertip,
)
from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    build_execution_bundle,
    canonical_json_bytes,
    sha256_file,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    FULL_URDF,
    LEFT_RELAXED_URDF,
    RELAXED_URDF,
    R1LiteCandidateKinematics,
)
from capx.integrations.r1lite.towel_plan import (  # noqa: E402
    SINGLE_ARM_STAGE_ORDER,
    rebuild_sequential_single_arm_plan_for_candidate,
    validate_sequential_single_arm_plan,
)
from r1lite_validate_towel_swept_collision import (  # noqa: E402
    FK_ORIENTATION_TOL_RAD,
    FK_POSITION_TOL_M,
    MAX_STAGE_JOINT_DELTA_RAD,
    SWEEP_SAMPLES_PER_SEGMENT,
    directed_tilted_quaternion,
    make_robot,
    orientation_error,
    solve_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-bundle-output", type=Path, required=True)
    parser.add_argument("--selected-plan-output", type=Path)
    return parser.parse_args()


def strict_topdown_edge_quaternion(stage: dict[str, Any]) -> np.ndarray:
    """Return tool-Z down with tool-Y aligned to the towel inward normal."""
    inward = np.asarray(stage["jaw_closing_axis_base_xy"], dtype=float).reshape(2)
    inward /= np.linalg.norm(inward)
    rotation = strict_topdown_edge_rotation(inward)
    quaternion = Rotation.from_matrix(rotation).as_quat()
    if not np.allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-9):
        raise ValueError("strict grasp quaternion does not point tool_z down")
    if not np.allclose(rotation[:2, 1], inward, atol=1e-9):
        raise ValueError("strict grasp quaternion does not align tool_y to edge normal")
    return quaternion


def edge_normal_tilt_quaternion(
    stage: dict[str, Any], tilt_deg: float
) -> np.ndarray:
    """Tilt about tool-Y, preserving a horizontal edge-normal jaw axis."""
    strict = Rotation.from_quat(strict_topdown_edge_quaternion(stage))
    local_tilt = Rotation.from_rotvec(
        np.array([0.0, math.radians(float(tilt_deg)), 0.0])
    )
    rotation = (strict * local_tilt).as_matrix()
    inward = np.asarray(stage["jaw_closing_axis_base_xy"], dtype=float)
    inward /= np.linalg.norm(inward)
    if not np.allclose(rotation[:, 1], [inward[0], inward[1], 0.0], atol=1e-9):
        raise ValueError("tilted grasp does not preserve the horizontal jaw axis")
    if rotation[2, 2] >= 0.0:
        raise ValueError("tilted grasp tool_z must retain a downward component")
    return Rotation.from_matrix(rotation).as_quat()


def _inactive_item(
    *,
    joints: np.ndarray,
    fk: Any,
) -> dict[str, Any]:
    quaternion = [float(value) for value in fk.quaternion_xyzw]
    self_orientation_error = 2.0 * math.acos(
        min(
            1.0,
            max(
                0.0,
                abs(sum(actual * target for actual, target in zip(
                    quaternion,
                    quaternion,
                ))),
            ),
        )
    )
    return {
        "label": "session_initial_hold",
        "joints_rad": joints.tolist(),
        "target_position_base_m": fk.position_base_m.tolist(),
        "target_quaternion_xyzw": quaternion,
        "fk_position_base_m": fk.position_base_m.tolist(),
        "fk_quaternion_xyzw": quaternion,
        "fk_position_error_m": 0.0,
        "fk_orientation_error_rad": self_orientation_error,
        "max_joint_delta_rad": 0.0,
    }


def _stage_collision_kwargs(
    plan: dict[str, Any], stage_id: str
) -> dict[str, Any]:
    if stage_id != "grasp":
        return {}
    safety = plan["safety"]
    return {
        "allowed_contact_finger_link": safety[
            "grasp_allowed_table_contact_link"
        ],
        "minimum_noncontact_clearance_m": float(
            safety["grasp_noncontact_min_clearance_m"]
        ),
        "minimum_contact_clearance_m": float(
            safety["grasp_designated_contact_min_clearance_m"]
        ),
    }


def _solve_active_transit(
    *,
    stage_id: str,
    side: str,
    robot: Any,
    local_fk: R1LiteCandidateKinematics,
    safety: R1LiteBimanualKinematics,
    torso: np.ndarray,
    reference: np.ndarray,
    inactive_side: str,
    inactive_joints: np.ndarray,
    final_target: tuple[str, np.ndarray, np.ndarray],
    collision_kwargs: dict[str, Any] | None = None,
    post_close_z_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[list[float]], list[dict[str, Any]]]:
    label, final_position, final_quaternion = final_target
    start_fk = local_fk.fk(reference, torso)
    distance = float(np.linalg.norm(start_fk.position_base_m - final_position))
    angle = orientation_error(start_fk.quaternion_xyzw, final_quaternion)
    count = max(
        1,
        int(np.ceil(distance / 0.08)),
        int(np.ceil(np.rad2deg(angle) / 15.0)),
    )
    rotations = Rotation.from_quat(
        np.vstack([start_fk.quaternion_xyzw, final_quaternion])
    )
    waypoints: list[list[float]] = []
    segments: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    current = reference.copy()
    for index in range(1, count + 1):
        alpha = index / count
        position = (
            start_fk.position_base_m * (1.0 - alpha) + final_position * alpha
        )
        quaternion = Slerp([0.0, 1.0], rotations)([alpha]).as_quat()[0]
        qualified, rejected = solve_candidates(
            robot,
            local_fk,
            side=side,
            tool_targets=[
                (
                    f"{stage_id}_{index:02d}_of_{count:02d}_{label}",
                    position,
                    quaternion,
                )
            ],
            reference_joints=current,
            torso_joints=torso,
            max_joint_delta_rad=MAX_STAGE_JOINT_DELTA_RAD,
            position_xy_tolerance_m=(0.005 if stage_id == "grasp" else None),
            position_z_tolerance_m=(0.002 if stage_id == "grasp" else None),
        )
        if not qualified and index < count:
            current_fk = local_fk.fk(current, torso)
            current_final_error = float(
                np.linalg.norm(current_fk.position_base_m - final_position)
            )
            progress = []
            for item in rejected:
                if (
                    "fk_position_base_m" not in item
                    or item["max_joint_delta_rad"] > MAX_STAGE_JOINT_DELTA_RAD
                    or item["fk_orientation_error_rad"] > FK_ORIENTATION_TOL_RAD
                ):
                    continue
                final_error = float(
                    np.linalg.norm(
                        np.asarray(item["fk_position_base_m"]) - final_position
                    )
                )
                if current_final_error - final_error >= 0.001:
                    progress.append((final_error, item))
            if progress:
                progress.sort(key=lambda value: value[0])
                qualified = [
                    {
                        **progress[0][1],
                        "accepted_via": "predicted_final_improvement",
                    }
                ]
        pre_final_grasp_waiver = stage_id == "align_grasp" or (
            stage_id == "grasp" and index < count
        )
        if not qualified and pre_final_grasp_waiver:
            transit_candidates = [
                item
                for item in rejected
                if (
                    "joints_rad" in item
                    and item["max_joint_delta_rad"] <= MAX_STAGE_JOINT_DELTA_RAD
                )
            ]
            if transit_candidates:
                qualified = [
                    {
                        **min(
                            transit_candidates,
                            key=lambda item: item["fk_position_error_m"],
                        ),
                        "accepted_via": (
                            "pre_final_grasp_fk_pose_waiver"
                        ),
                    }
                ]
        if post_close_z_policy and post_close_z_policy.get("enabled") is True:
            samples = int(
                post_close_z_policy.get(
                    "fk_samples_per_segment", SWEEP_SAMPLES_PER_SEGMENT
                )
            )
            tolerance = float(
                post_close_z_policy.get(
                    "planned_decrease_tolerance_m", 0.0002
                )
            )
            monotonic = []
            for item in qualified:
                candidate_joints = np.asarray(item["joints_rad"], dtype=float)
                z_values = [
                    float(
                        local_fk.fk(
                            current * (1.0 - alpha)
                            + candidate_joints * alpha,
                            torso,
                        ).position_base_m[2]
                    )
                    for alpha in np.linspace(0.0, 1.0, samples + 1)
                ]
                minimum_delta = float(min(np.diff(z_values), default=0.0))
                total_delta = float(z_values[-1] - z_values[0])
                if minimum_delta >= -tolerance and total_delta >= -tolerance:
                    monotonic.append(
                        {
                            **item,
                            "post_close_tcp_z_minimum_sample_delta_m": minimum_delta,
                            "post_close_tcp_z_total_delta_m": total_delta,
                        }
                    )
                else:
                    rejected.append(
                        {
                            **item,
                            "reason": "post_close_tcp_z_decreased",
                            "post_close_tcp_z_minimum_sample_delta_m": minimum_delta,
                            "post_close_tcp_z_total_delta_m": total_delta,
                        }
                    )
            qualified = monotonic
        if not qualified:
            raise RuntimeError(
                f"{stage_id} active transit {index}/{count} failed: {rejected}"
            )
        selected = min(
            qualified,
            key=lambda item: (
                item["fk_position_error_m"],
                item["max_joint_delta_rad"],
            ),
        )
        candidate = np.asarray(selected["joints_rad"], dtype=float)
        pair_start = {
            side: current,
            inactive_side: inactive_joints,
        }
        pair_end = {
            side: candidate,
            inactive_side: inactive_joints,
        }
        reports = safety.validate_interpolated_segment(
            pair_start["left"],
            pair_start["right"],
            pair_end["left"],
            pair_end["right"],
            torso,
            samples=SWEEP_SAMPLES_PER_SEGMENT,
            **(collision_kwargs or {}),
        )
        segments.append(
            {
                "minimum_estimated_link_clearance_m": min(
                    report.estimated_link_clearance_m for report in reports
                ),
                "minimum_tcp_separation_m": min(
                    report.tcp_separation_m for report in reports
                ),
                **(
                    {
                        "post_close_tcp_z_minimum_sample_delta_m": selected[
                            "post_close_tcp_z_minimum_sample_delta_m"
                        ],
                        "post_close_tcp_z_total_delta_m": selected[
                            "post_close_tcp_z_total_delta_m"
                        ],
                    }
                    if "post_close_tcp_z_minimum_sample_delta_m" in selected
                    else {}
                ),
            }
        )
        current = candidate
        waypoints.append(candidate.tolist())
    assert selected is not None
    return selected, waypoints, segments


def select_and_certify_candidate_plan(args: argparse.Namespace) -> None:
    source_bytes = args.plan.read_bytes()
    source_plan = json.loads(source_bytes)
    validate_sequential_single_arm_plan(source_plan)
    if source_plan.get("phase_side") != "right":
        raise ValueError("contour candidate selection is right-arm-only")
    assert args.selected_plan_output is not None
    robot = make_robot(args.config, "right")
    torso = robot.get_torso_joints()
    initial = {
        "left": make_robot(args.config, "left").get_arm_joints(),
        "right": robot.get_arm_joints(),
    }
    local_fk = R1LiteCandidateKinematics(side="right")
    safety = R1LiteTableCollisionKinematics(
        active_side="right",
        table_z_m=float(source_plan["safety"]["table_z_m"]),
        tcp_to_fingertip_m=float(source_plan["safety"]["tcp_to_fingertip_m"]),
        distal_capsule_radius_m=float(
            source_plan["safety"]["distal_link_capsule_radius_m"]
        ),
        minimum_table_clearance_m=float(
            source_plan["safety"]["minimum_table_clearance_m"]
        ),
    )
    terminal_reports: list[dict[str, Any]] = []
    qualified_plans: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for candidate in source_plan["perception"]["right_side_grasp_candidates"]:
        report: dict[str, Any] = {
            "candidate_id": candidate["id"],
            "base_xyz_m": candidate["base_xyz_m"],
            "qualified_final_grasp_count": 0,
            "passed": False,
            "tilt_reports": [],
        }
        all_qualified: list[tuple[dict[str, Any], float, dict[str, Any]]] = []
        all_rejected: list[dict[str, Any]] = []
        for tilt_deg in source_plan["grasp_geometry"]["tilt_candidates_deg"]:
            selected_plan = rebuild_sequential_single_arm_plan_for_candidate(
                source_plan,
                candidate_id=str(candidate["id"]),
                selected_grasp_tilt_deg=float(tilt_deg),
            )
            grasp = next(
                stage for stage in selected_plan["stages"] if stage["id"] == "grasp"
            )
            quaternion = edge_normal_tilt_quaternion(grasp, float(tilt_deg))
            position = tcp_position_for_fingertip(
                np.asarray(grasp["active_fingertip_position_base_m"], dtype=float),
                quaternion,
                float(selected_plan["safety"]["tcp_to_fingertip_m"]),
            )
            qualified, rejected = solve_candidates(
                robot,
                local_fk,
                side="right",
                tool_targets=[
                    (
                        f"candidate_{candidate['id']}_tilt_{tilt_deg:+.0f}",
                        position,
                        quaternion,
                    )
                ],
                reference_joints=initial["right"],
                torso_joints=torso,
                max_joint_delta_rad=3.0,
                max_qualified=5,
                position_xy_tolerance_m=0.005,
                position_z_tolerance_m=0.002,
            )
            report["tilt_reports"].append(
                {
                    "tilt_deg": float(tilt_deg),
                    "qualified_count": len(qualified),
                }
            )
            all_qualified.extend(
                (item, float(tilt_deg), selected_plan) for item in qualified
            )
            all_rejected.extend(rejected)
        report["qualified_final_grasp_count"] = len(all_qualified)
        if all_qualified:
            ranked_qualified = sorted(
                all_qualified,
                key=lambda entry: (
                    entry[0]["fk_position_error_m"],
                    entry[0]["max_joint_delta_rad"],
                    abs(entry[1]),
                ),
            )
            collision_errors = []
            for best, best_tilt, selected_plan in ranked_qualified:
                try:
                    collision = safety.validate(
                        initial["left"],
                        np.asarray(best["joints_rad"], dtype=float),
                        torso,
                        **_stage_collision_kwargs(selected_plan, "grasp"),
                    )
                except Exception as exc:
                    collision_errors.append(
                        {"tilt_deg": best_tilt, "error": repr(exc)}
                    )
                    continue
                clearance = float(collision.estimated_link_clearance_m)
                report.update(
                    {
                        "passed": True,
                        "selected_tilt_deg": best_tilt,
                        "fk_position_error_m": best["fk_position_error_m"],
                        "fk_orientation_error_rad": best[
                            "fk_orientation_error_rad"
                        ],
                        "maximum_joint_delta_rad": best["max_joint_delta_rad"],
                        "estimated_link_clearance_m": clearance,
                    }
                )
                qualified_plans.append(
                    (
                        (
                            float(best["fk_position_error_m"]),
                            float(best["max_joint_delta_rad"]),
                            -clearance,
                        ),
                        selected_plan,
                    )
                )
                break
            if report["passed"] is not True:
                report["collision_errors"] = collision_errors
        elif all_rejected:
            best_rejected = min(
                (item for item in all_rejected if "fk_position_error_m" in item),
                key=lambda item: item["fk_position_error_m"],
                default=None,
            )
            if best_rejected is not None:
                report.update(
                    {
                        "best_fk_position_error_m": best_rejected[
                            "fk_position_error_m"
                        ],
                        "best_fk_orientation_error_rad": best_rejected[
                            "fk_orientation_error_rad"
                        ],
                    }
                )
        terminal_reports.append(report)
    qualified_plans.sort(key=lambda item: item[0])
    selection_audit: dict[str, Any] = {
        "experiment": "pipeline7_right_side_contour_candidate_selection",
        "source_plan_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "robot_motion_commands_sent": False,
        "candidate_terminal_reports": terminal_reports,
        "terminal_qualified_count": len(qualified_plans),
        "passed": False,
        "full_certification_attempts": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r1lite_right_candidate_") as temp:
        temporary = Path(temp)
        for rank, (_, selected_plan) in enumerate(qualified_plans, start=1):
            candidate_id = selected_plan["grasp_geometry"]["selected_candidate_id"]
            plan_path = temporary / f"{rank:02d}_{candidate_id}_plan.json"
            audit_path = temporary / f"{rank:02d}_{candidate_id}_audit.json"
            bundle_path = temporary / f"{rank:02d}_{candidate_id}_bundle.json"
            plan_bytes = json.dumps(
                selected_plan, indent=2, ensure_ascii=False
            ).encode()
            plan_path.write_bytes(plan_bytes)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(args.config),
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(audit_path),
                    "--execution-bundle-output",
                    str(bundle_path),
                ],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
            )
            candidate_audit = (
                json.loads(audit_path.read_text()) if audit_path.exists() else {}
            )
            attempt = {
                "rank": rank,
                "candidate_id": candidate_id,
                "returncode": completed.returncode,
                "passed": candidate_audit.get("passed") is True,
                "error": candidate_audit.get("error"),
            }
            selection_audit["full_certification_attempts"].append(attempt)
            if completed.returncode != 0 or candidate_audit.get("passed") is not True:
                continue
            args.selected_plan_output.parent.mkdir(parents=True, exist_ok=True)
            args.execution_bundle_output.parent.mkdir(parents=True, exist_ok=True)
            args.selected_plan_output.write_bytes(plan_bytes)
            args.execution_bundle_output.write_bytes(bundle_path.read_bytes())
            selection_audit.update(
                {
                    "passed": True,
                    "selected_candidate_id": candidate_id,
                    "selected_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                    "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                    "execution_bundle_sha256": sha256_file(
                        args.execution_bundle_output
                    ),
                    "certification": candidate_audit,
                }
            )
            args.output.write_text(
                json.dumps(selection_audit, indent=2, ensure_ascii=False)
            )
            print(json.dumps(selection_audit, ensure_ascii=False))
            return
    selection_audit["error"] = (
        "no right-side contour candidate passed terminal and full certification"
    )
    args.output.write_text(json.dumps(selection_audit, indent=2, ensure_ascii=False))
    raise RuntimeError(selection_audit["error"])


def main() -> None:
    args = parse_args()
    if args.selected_plan_output is not None:
        select_and_certify_candidate_plan(args)
        return
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    validate_sequential_single_arm_plan(plan)
    side = str(plan["phase_side"])
    inactive = "right" if side == "left" else "left"
    robots = {
        "left": make_robot(args.config, "left"),
        "right": make_robot(args.config, "right"),
    }
    audit: dict[str, Any] = {
        "experiment": "pipeline7_sequential_single_arm_phase_certification",
        "phase_side": side,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "robot_motion_commands_sent": False,
        "passed": False,
        "stages": [],
        "segments": [],
    }
    try:
        torso = robots["right"].get_torso_joints()
        initial = {
            arm: robots[arm].get_arm_joints() for arm in ("left", "right")
        }
        references = {arm: value.copy() for arm, value in initial.items()}
        local_fk = {
            arm: R1LiteCandidateKinematics(side=arm)
            for arm in ("left", "right")
        }
        safety = R1LiteTableCollisionKinematics(
            active_side=side,
            table_z_m=float(plan["safety"]["table_z_m"]),
            tcp_to_fingertip_m=float(
                plan["safety"]["tcp_to_fingertip_m"]
            ),
            distal_capsule_radius_m=float(
                plan["safety"]["distal_link_capsule_radius_m"]
            ),
            minimum_table_clearance_m=float(
                plan["safety"]["minimum_table_clearance_m"]
            ),
        )
        inactive_fk = local_fk[inactive].fk(initial[inactive], torso)
        chosen = []
        for stage in plan["stages"]:
            if stage["id"] == "home":
                reports = safety.validate_interpolated_segment(
                    references["left"],
                    references["right"],
                    initial["left"],
                    initial["right"],
                    torso,
                    samples=SWEEP_SAMPLES_PER_SEGMENT,
                )
                selected = {
                    arm: _inactive_item(
                        joints=initial[arm],
                        fk=local_fk[arm].fk(initial[arm], torso),
                    )
                    for arm in ("left", "right")
                }
                home_steps = max(
                    1,
                    int(
                        np.ceil(
                            max(
                                float(
                                    np.max(
                                        np.abs(
                                            initial[arm] - references[arm]
                                        )
                                    )
                                )
                                for arm in ("left", "right")
                            )
                            / MAX_STAGE_JOINT_DELTA_RAD
                        )
                    ),
                )
                waypoints = {
                    arm: [
                        (
                            references[arm] * (1.0 - index / home_steps)
                            + initial[arm] * (index / home_steps)
                        ).tolist()
                        for index in range(1, home_steps + 1)
                    ]
                    for arm in ("left", "right")
                }
                stage_segments = [
                    {
                        "minimum_estimated_link_clearance_m": min(
                            report.estimated_link_clearance_m for report in reports
                        ),
                        "minimum_tcp_separation_m": min(
                            report.tcp_separation_m for report in reports
                        ),
                    }
                ]
            else:
                fingertip = np.asarray(
                    stage["active_fingertip_position_base_m"], dtype=float
                )
                if stage["orientation"] == "edge_normal_constrained_tilt":
                    orientation_candidates = [
                        (
                            f"edge_normal_tilt_{tilt:+.0f}",
                            edge_normal_tilt_quaternion(stage, float(tilt)),
                        )
                        for tilt in stage["tilt_search_deg"]
                    ]
                else:
                    outward_y = 1.0 if side == "left" else -1.0
                    directions = [
                        ("forward", np.array([1.0, 0.0, 0.0])),
                        ("outward", np.array([0.0, outward_y, 0.0])),
                        (
                            "forward_outward",
                            np.array([1.0, outward_y, 0.0]) / np.sqrt(2.0),
                        ),
                    ]
                    orientation_candidates = [
                        (
                            f"dir{direction_name}_tilt{tilt:.0f}_roll{roll:+.0f}",
                            directed_tilted_quaternion(
                                direction,
                                tilt_deg=tilt,
                                roll_deg=roll,
                            ),
                        )
                        for direction_name, direction in directions
                        for tilt in reversed(stage["tilt_search_deg"])
                        for roll in (0.0, -90.0, 90.0, 180.0)
                    ]
                errors = []
                solved = None
                for orientation_label, quaternion in orientation_candidates:
                    position = tcp_position_for_fingertip(
                        fingertip,
                        quaternion,
                        float(plan["safety"]["tcp_to_fingertip_m"]),
                    )
                    try:
                        solved = _solve_active_transit(
                            stage_id=stage["id"],
                            side=side,
                            robot=robots[side],
                            local_fk=local_fk[side],
                            safety=safety,
                            torso=torso,
                            reference=references[side],
                            inactive_side=inactive,
                            inactive_joints=initial[inactive],
                            final_target=(
                                orientation_label,
                                position,
                                quaternion,
                            ),
                            collision_kwargs=_stage_collision_kwargs(
                                plan, stage["id"]
                            ),
                            post_close_z_policy=(
                                plan["safety"].get("post_close_tcp_z_policy")
                                if stage["id"]
                                in plan["safety"]
                                .get("post_close_tcp_z_policy", {})
                                .get("protected_stage_ids", [])
                                else None
                            ),
                        )
                        break
                    except Exception as exc:
                        errors.append(repr(exc))
                if solved is None:
                    raise RuntimeError(
                        f"{stage['id']} has no certified active-arm path: {errors}"
                    )
                active_item, active_waypoints, stage_segments = solved
                selected = {
                    side: active_item,
                    inactive: _inactive_item(
                        joints=initial[inactive],
                        fk=inactive_fk,
                    ),
                }
                waypoints = {
                    side: active_waypoints,
                    inactive: [
                        initial[inactive].tolist() for _ in active_waypoints
                    ],
                }
            references = {
                arm: np.asarray(selected[arm]["joints_rad"], dtype=float)
                for arm in ("left", "right")
            }
            chosen.append(
                {
                    "id": stage["id"],
                    "left": selected["left"],
                    "right": selected["right"],
                    "joint_waypoints_rad": waypoints,
                }
            )
            segment = {
                "to_stage": stage["id"],
                "samples": SWEEP_SAMPLES_PER_SEGMENT,
                "minimum_estimated_link_clearance_m": min(
                    item["minimum_estimated_link_clearance_m"]
                    for item in stage_segments
                ),
                "minimum_tcp_separation_m": min(
                    item["minimum_tcp_separation_m"] for item in stage_segments
                ),
                "passed": True,
            }
            z_stage_segments = [
                item
                for item in stage_segments
                if "post_close_tcp_z_minimum_sample_delta_m" in item
            ]
            if z_stage_segments:
                segment.update(
                    {
                        "post_close_tcp_z_minimum_sample_delta_m": min(
                            item["post_close_tcp_z_minimum_sample_delta_m"]
                            for item in z_stage_segments
                        ),
                        "post_close_tcp_z_total_delta_m": sum(
                            item["post_close_tcp_z_total_delta_m"]
                            for item in z_stage_segments
                        ),
                    }
                )
            audit["stages"].append(
                {
                    "id": stage["id"],
                    "active_side": side,
                    "waypoint_count": len(waypoints[side]),
                    "passed": True,
                }
            )
            audit["segments"].append(segment)

        bundle_stages = []
        previous = {arm: initial[arm] for arm in ("left", "right")}
        for selected in chosen:
            bundle_stage: dict[str, Any] = {"id": selected["id"]}
            for arm in ("left", "right"):
                waypoint_values = selected["joint_waypoints_rad"][arm]
                maximum_delta = 0.0
                prior = previous[arm]
                for waypoint in waypoint_values:
                    waypoint_array = np.asarray(waypoint, dtype=float)
                    maximum_delta = max(
                        maximum_delta,
                        float(np.max(np.abs(waypoint_array - prior))),
                    )
                    prior = waypoint_array
                item = selected[arm]
                bundle_stage[arm] = {
                    "joints_rad": item["joints_rad"],
                    "joint_waypoints_rad": waypoint_values,
                    "target_position_base_m": item["target_position_base_m"],
                    "target_quaternion_xyzw": item["target_quaternion_xyzw"],
                    "fk_position_base_m": item["fk_position_base_m"],
                    "fk_quaternion_xyzw": item["fk_quaternion_xyzw"],
                    "fk_position_error_m": item["fk_position_error_m"],
                    "fk_orientation_error_rad": item[
                        "fk_orientation_error_rad"
                    ],
                    "maximum_joint_delta_rad": maximum_delta,
                }
                previous[arm] = np.asarray(item["joints_rad"], dtype=float)
            bundle_stages.append(bundle_stage)
        minimum_clearance = min(
            item["minimum_estimated_link_clearance_m"]
            for item in audit["segments"]
        )
        minimum_tcp = min(
            item["minimum_tcp_separation_m"] for item in audit["segments"]
        )
        z_segments = [
            item
            for item in audit["segments"]
            if "post_close_tcp_z_minimum_sample_delta_m" in item
        ]
        if plan["safety"].get("post_close_tcp_z_policy", {}).get("enabled"):
            if not z_segments:
                raise RuntimeError("post-close TCP Z policy produced no audit segments")
            audit["post_close_tcp_z_monotonic"] = {
                "passed": True,
                "protected_stage_ids": plan["safety"][
                    "post_close_tcp_z_policy"
                ]["protected_stage_ids"],
                "minimum_sample_delta_m": min(
                    item["post_close_tcp_z_minimum_sample_delta_m"]
                    for item in z_segments
                ),
                "segments_checked": len(z_segments),
            }
        collision_model = {
            "name": (
                "active_arm_distal_links_and_two_open_finger_mesh_aabbs"
                "_vs_table_plane"
            ),
            "active_side": side,
            "table_z_m": safety.table_z_m,
            "tcp_to_fingertip_m": safety.tcp_to_fingertip_m,
            "link_capsule_radius_m": safety.link_capsule_radius_m,
            "gripper_finger_open_travel_m": safety.FINGER_MAX_TRAVEL_M,
            "gripper_finger_joint_origins_m": [
                value.tolist() for value in safety.FINGER_JOINT_ORIGINS_M
            ],
            "gripper_finger_mesh_bounds_m": [
                [lower.tolist(), upper.tolist()]
                for lower, upper in safety.FINGER_MESH_BOUNDS_M
            ],
            "minimum_link_clearance_m": safety.minimum_link_clearance_m,
            "minimum_tcp_separation_m": 0.0,
            "sweep_interpolation": "linear_joint_space",
            "sweep_samples_per_segment": SWEEP_SAMPLES_PER_SEGMENT,
        }
        bundle = build_execution_bundle(
            plan_sha256=audit["plan_sha256"],
            config_sha256=sha256_file(args.config),
            full_urdf_sha256=sha256_file(FULL_URDF),
            left_relaxed_urdf_sha256=sha256_file(LEFT_RELAXED_URDF),
            right_relaxed_urdf_sha256=sha256_file(RELAXED_URDF),
            tcp_definition=plan["tcp"],
            collision_model=collision_model,
            initial_left_joints_rad=initial["left"],
            initial_right_joints_rad=initial["right"],
            torso_joints_rad=torso,
            stages=bundle_stages,
            sweeps={
                "samples_per_segment": SWEEP_SAMPLES_PER_SEGMENT,
                "segments": [
                    {
                        key: value
                        for key, value in segment.items()
                        if not key.startswith("post_close_tcp_z_")
                    }
                    for segment in audit["segments"]
                ],
                "minimum_swept_link_clearance_m": minimum_clearance,
                "minimum_swept_tcp_separation_m": minimum_tcp,
            },
            maximum_stage_joint_delta_rad=MAX_STAGE_JOINT_DELTA_RAD,
            fk_position_tolerance_m=FK_POSITION_TOL_M,
            fk_orientation_tolerance_rad=FK_ORIENTATION_TOL_RAD,
            execution_mode="sequential_single_arm",
            stage_order=SINGLE_ARM_STAGE_ORDER,
        )
        args.execution_bundle_output.parent.mkdir(parents=True, exist_ok=True)
        args.execution_bundle_output.write_bytes(canonical_json_bytes(bundle))
        audit["passed"] = True
        audit["execution_bundle_sha256"] = sha256_file(
            args.execution_bundle_output
        )
    except Exception as exc:
        audit["error"] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
        for robot in robots.values():
            robot.close()
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
