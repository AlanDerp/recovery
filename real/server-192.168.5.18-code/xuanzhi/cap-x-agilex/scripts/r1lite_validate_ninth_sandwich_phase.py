#!/usr/bin/env python3
"""Auto-select and certify one no-motion pipeline-nine sandwich phase."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteCombinedCollisionKinematics,
)
from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    build_execution_bundle,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    FULL_URDF,
    LEFT_RELAXED_URDF,
    RELAXED_URDF,
    R1LiteCandidateKinematics,
)
from capx.integrations.r1lite.sandwich_plan import (  # noqa: E402
    STAGE_ORDER,
    apply_grasp_orientation_branch,
    build_policy,
    select_arm,
    validate_sandwich_intent,
)
from r1lite_validate_sequential_towel_phase import (  # noqa: E402
    _inactive_item,
    _solve_active_transit,
)
from r1lite_validate_towel_swept_collision import (  # noqa: E402
    FK_ORIENTATION_TOL_RAD,
    FK_POSITION_TOL_M,
    MAX_STAGE_JOINT_DELTA_RAD,
    SWEEP_SAMPLES_PER_SEGMENT,
    make_robot,
    solve_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--selected-plan-output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    parser.add_argument("--execution-bundle-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _preflight_grasp_endpoint(
    *,
    side: str,
    plan: dict[str, Any],
    robot: Any,
    torso: np.ndarray,
    initial_joints: np.ndarray,
) -> dict[str, Any]:
    """Quickly reject yaw branches whose pregrasp endpoint is not reachable."""
    pregrasp = next(stage for stage in plan["stages"] if stage["id"] == "pregrasp")
    local_fk = R1LiteCandidateKinematics(side=side)
    qualified, rejected = solve_candidates(
        robot,
        local_fk,
        side=side,
        tool_targets=[
            (
                "ninth_pregrasp_orientation_preflight",
                np.asarray(pregrasp["active_tcp_position_base_m"], dtype=float),
                np.asarray(pregrasp["target_quaternion_xyzw"], dtype=float),
            )
        ],
        reference_joints=initial_joints,
        torso_joints=torso,
        max_joint_delta_rad=MAX_STAGE_JOINT_DELTA_RAD,
    )
    best_rejected = (
        min(
            rejected,
            key=lambda item: float(item.get("fk_position_error_m", float("inf"))),
        )
        if rejected
        else None
    )
    return {
        "passed": bool(qualified),
        "best_fk_position_error_m": (
            min(float(item["fk_position_error_m"]) for item in qualified)
            if qualified
            else None
        ),
        "rejected_count": len(rejected),
        "best_rejected": (
            {
                key: best_rejected[key]
                for key in (
                    "reason",
                    "target_position_base_m",
                    "fk_position_base_m",
                    "fk_position_error_m",
                    "fk_orientation_error_rad",
                    "max_joint_delta_rad",
                )
                if key in best_rejected
            }
            if best_rejected is not None
            else None
        ),
    }


def _certify_side(
    *,
    side: str,
    plan: dict[str, Any],
    robot: Any,
    torso: np.ndarray,
    initial: dict[str, np.ndarray],
) -> dict[str, Any]:
    inactive = "right" if side == "left" else "left"
    local_fk = {
        arm: R1LiteCandidateKinematics(side=arm) for arm in ("left", "right")
    }
    safety = R1LiteCombinedCollisionKinematics(
        table_z_m=float(plan["safety"]["table_z_m"]),
        tcp_to_fingertip_m=float(plan["safety"]["tcp_to_fingertip_m"]),
        minimum_link_clearance_m=float(
            plan["safety"]["minimum_link_clearance_m"]
        ),
        minimum_table_clearance_m=float(
            plan["safety"]["minimum_table_clearance_m"]
        ),
        minimum_gripper_envelope_separation_m=float(
            plan["safety"]["minimum_gripper_envelope_separation_m"]
        ),
    )
    references = {arm: value.copy() for arm, value in initial.items()}
    inactive_fk = local_fk[inactive].fk(initial[inactive], torso)
    chosen: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
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
            home_steps = max(
                1,
                int(
                    np.ceil(
                        max(
                            float(
                                np.max(
                                    np.abs(initial[arm] - references[arm])
                                )
                            )
                            for arm in ("left", "right")
                        )
                        / MAX_STAGE_JOINT_DELTA_RAD
                    )
                ),
            )
            selected = {
                arm: _inactive_item(
                    joints=initial[arm],
                    fk=local_fk[arm].fk(initial[arm], torso),
                )
                for arm in ("left", "right")
            }
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
            position = np.asarray(
                stage["active_tcp_position_base_m"], dtype=float
            )
            quaternion = np.asarray(
                stage["target_quaternion_xyzw"], dtype=float
            )
            active_item, active_waypoints, stage_segments = (
                _solve_active_transit(
                    stage_id=stage["id"],
                    side=side,
                    robot=robot,
                    local_fk=local_fk[side],
                    safety=safety,
                    torso=torso,
                    reference=references[side],
                    inactive_side=inactive,
                    inactive_joints=initial[inactive],
                    final_target=(
                        "strict_topdown_object_axis",
                        position,
                        quaternion,
                    ),
                )
            )
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
        segments.append(
            {
                "to_stage": stage["id"],
                "samples": SWEEP_SAMPLES_PER_SEGMENT,
                "minimum_estimated_link_clearance_m": min(
                    item["minimum_estimated_link_clearance_m"]
                    for item in stage_segments
                ),
                "minimum_tcp_separation_m": min(
                    item["minimum_tcp_separation_m"]
                    for item in stage_segments
                ),
                "passed": True,
            }
        )
    bundle_stages: list[dict[str, Any]] = []
    previous = {arm: initial[arm].copy() for arm in ("left", "right")}
    maximum_joint_delta = 0.0
    maximum_fk_position_error = 0.0
    for selected in chosen:
        bundle_stage: dict[str, Any] = {"id": selected["id"]}
        for arm in ("left", "right"):
            waypoint_values = selected["joint_waypoints_rad"][arm]
            stage_maximum_delta = 0.0
            prior = previous[arm]
            for waypoint in waypoint_values:
                waypoint_array = np.asarray(waypoint, dtype=float)
                stage_maximum_delta = max(
                    stage_maximum_delta,
                    float(np.max(np.abs(waypoint_array - prior))),
                )
                prior = waypoint_array
            item = selected[arm]
            maximum_joint_delta = max(
                maximum_joint_delta, stage_maximum_delta
            )
            maximum_fk_position_error = max(
                maximum_fk_position_error,
                float(item["fk_position_error_m"]),
            )
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
                "maximum_joint_delta_rad": stage_maximum_delta,
            }
            previous[arm] = np.asarray(item["joints_rad"], dtype=float)
        bundle_stages.append(bundle_stage)
    minimum_clearance = min(
        segment["minimum_estimated_link_clearance_m"]
        for segment in segments
    )
    minimum_tcp = min(
        segment["minimum_tcp_separation_m"] for segment in segments
    )
    score = (
        maximum_fk_position_error * 100.0
        + maximum_joint_delta
        - min(minimum_clearance, 0.10)
        - min(minimum_tcp, 0.30) * 0.1
    )
    return {
        "side": side,
        "passed": True,
        "score": float(score),
        "maximum_joint_delta_rad": maximum_joint_delta,
        "maximum_fk_position_error_m": maximum_fk_position_error,
        "minimum_swept_link_clearance_m": minimum_clearance,
        "minimum_swept_tcp_separation_m": minimum_tcp,
        "segments": segments,
        "bundle_stages": bundle_stages,
    }


def main() -> None:
    args = parse_args()
    intent = json.loads(args.intent.read_text())
    validate_sandwich_intent(intent)
    robots = {
        side: make_robot(args.config, side) for side in ("left", "right")
    }
    audit: dict[str, Any] = {
        "schema": "capx.r1lite.ninth_sandwich_certification_audit.v1",
        "pipeline": 9,
        "phase": intent["phase"],
        "intent_sha256": sha256_file(args.intent),
        "robot_motion_commands_sent": False,
        "arm_candidates": {},
        "passed": False,
    }
    try:
        torso = np.asarray(robots["right"].get_torso_joints(), dtype=float)
        initial = {
            side: np.asarray(robots[side].get_arm_joints(), dtype=float)
            for side in ("left", "right")
        }
        candidates = []
        source_yaw_offsets = (
            (180.0, 90.0, -90.0, 45.0, -45.0, 135.0, -135.0, 0.0)
            if intent["phase_definition"]["source"] == "donut"
            else (180.0, 0.0)
        )
        target_yaw_offsets = (0.0, 180.0)
        source_name = intent["phase_definition"]["source"]
        source_y = float(
            intent["perception"]["scene"]["objects"][source_name][
                "center_base_m"
            ][1]
        )
        side_order = ("right", "left") if source_y < 0.0 else ("left", "right")
        for side in side_order:
            base_plan = select_arm(intent, side)
            branch_audits: list[dict[str, Any]] = []
            selected_for_side: tuple[dict[str, Any], dict[str, Any]] | None = None
            for source_yaw in source_yaw_offsets:
                preflight_plan = apply_grasp_orientation_branch(
                    base_plan,
                    source_yaw_offset_deg=source_yaw,
                    target_yaw_offset_deg=0.0,
                )
                preflight = _preflight_grasp_endpoint(
                    side=side,
                    plan=preflight_plan,
                    robot=robots[side],
                    torso=torso,
                    initial_joints=initial[side],
                )
                if not preflight["passed"]:
                    branch_audits.append(
                        {
                            "source_yaw_offset_deg": source_yaw,
                            "target_yaw_offset_deg": None,
                            "passed": False,
                            "preflight": preflight,
                        }
                    )
                    continue
                for target_yaw in target_yaw_offsets:
                    plan = apply_grasp_orientation_branch(
                        base_plan,
                        source_yaw_offset_deg=source_yaw,
                        target_yaw_offset_deg=target_yaw,
                    )
                    branch = {
                        "source_yaw_offset_deg": source_yaw,
                        "target_yaw_offset_deg": target_yaw,
                        "passed": False,
                        "preflight": preflight,
                    }
                    try:
                        result = _certify_side(
                            side=side,
                            plan=plan,
                            robot=robots[side],
                            torso=torso,
                            initial=initial,
                        )
                        branch["passed"] = True
                        branch["score"] = result["score"]
                        branch_audits.append(branch)
                        selected_for_side = (result, plan)
                        break
                    except Exception as exc:
                        branch["error"] = repr(exc)
                        branch_audits.append(branch)
                if selected_for_side is not None:
                    break
            if selected_for_side is None:
                audit["arm_candidates"][side] = {
                    "side": side,
                    "passed": False,
                    "orientation_branches": branch_audits,
                }
                continue
            result, plan = selected_for_side
            candidates.append((result, plan))
            audit["arm_candidates"][side] = {
                **{
                    key: value
                    for key, value in result.items()
                    if key != "bundle_stages"
                },
                "orientation_branches": branch_audits,
                "selected_orientation_branch": plan[
                    "grasp_orientation_branch"
                ],
            }
            break
        if not candidates:
            raise RuntimeError("neither arm has a certified sandwich path")
        selected_result, selected_plan = min(
            candidates, key=lambda item: item[0]["score"]
        )
        plan_bytes = canonical_json_bytes(selected_plan)
        plan_sha256 = sha256_bytes(plan_bytes)
        policy = build_policy(selected_plan, plan_sha256=plan_sha256)
        collision_model = {
            "name": "combined_interarm_table_and_open_gripper",
            "active_side": selected_result["side"],
            "table_z_m": float(selected_plan["safety"]["table_z_m"]),
            "tcp_to_fingertip_m": float(
                selected_plan["safety"]["tcp_to_fingertip_m"]
            ),
            "minimum_link_clearance_m": float(
                selected_plan["safety"]["minimum_link_clearance_m"]
            ),
            "minimum_table_clearance_m": float(
                selected_plan["safety"]["minimum_table_clearance_m"]
            ),
            "minimum_gripper_envelope_separation_m": float(
                selected_plan["safety"][
                    "minimum_gripper_envelope_separation_m"
                ]
            ),
            "sweep_interpolation": "linear_joint_space",
            "sweep_samples_per_segment": SWEEP_SAMPLES_PER_SEGMENT,
        }
        bundle = build_execution_bundle(
            plan_sha256=plan_sha256,
            config_sha256=sha256_file(args.config),
            full_urdf_sha256=sha256_file(FULL_URDF),
            left_relaxed_urdf_sha256=sha256_file(LEFT_RELAXED_URDF),
            right_relaxed_urdf_sha256=sha256_file(RELAXED_URDF),
            tcp_definition=selected_plan["tcp"],
            collision_model=collision_model,
            initial_left_joints_rad=initial["left"],
            initial_right_joints_rad=initial["right"],
            torso_joints_rad=torso,
            stages=selected_result["bundle_stages"],
            sweeps={
                "samples_per_segment": SWEEP_SAMPLES_PER_SEGMENT,
                "segments": selected_result["segments"],
                "minimum_swept_link_clearance_m": selected_result[
                    "minimum_swept_link_clearance_m"
                ],
                "minimum_swept_tcp_separation_m": selected_result[
                    "minimum_swept_tcp_separation_m"
                ],
            },
            maximum_stage_joint_delta_rad=MAX_STAGE_JOINT_DELTA_RAD,
            fk_position_tolerance_m=FK_POSITION_TOL_M,
            fk_orientation_tolerance_rad=FK_ORIENTATION_TOL_RAD,
            execution_mode="sequential_single_arm",
            stage_order=STAGE_ORDER,
        )
        args.selected_plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.selected_plan_output.write_bytes(plan_bytes)
        args.policy_output.write_bytes(canonical_json_bytes(policy))
        args.execution_bundle_output.write_bytes(canonical_json_bytes(bundle))
        audit.update(
            {
                "passed": True,
                "selected_side": selected_result["side"],
                "selection_score": selected_result["score"],
                "plan_sha256": plan_sha256,
                "policy_sha256": sha256_file(args.policy_output),
                "execution_bundle_sha256": sha256_file(
                    args.execution_bundle_output
                ),
            }
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
