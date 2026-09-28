#!/usr/bin/env python3
"""Replay one certified Pipeline-7 joint sequence without online IK/refinement."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_runtime import (  # noqa: E402
    BimanualRuntimeConfig,
    FeedbackStaleError,
    R1LiteBimanualRuntime,
)
from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteTableCollisionKinematics,
)
from capx.integrations.r1lite.bimanual_relaxed_ik import (  # noqa: E402
    IsolatedBimanualRelaxedIK,
)
from capx.integrations.r1lite.bimanual_towel_executor import (  # noqa: E402
    BimanualTowelExecutor,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    R1LiteCandidateKinematics,
)
from r1lite_right_joint2_control_offset import (  # noqa: E402
    HistoricalMappingRightJoint2ControlOffsetMixin,
    RightJoint2ControlOffsetMixin,
)


def load_historical_moca_samples(
    path: Path,
    *,
    expected_bundle_sha256: str,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], dict]:
    raw = path.read_bytes()
    historical = json.loads(raw)
    if historical.get("passed") is not True:
        raise ValueError("historical MOCA source audit did not pass")
    if historical.get("bundle_sha256") != expected_bundle_sha256:
        raise ValueError("historical MOCA source audit uses a different bundle")
    samples = {"left": [], "right": []}
    for trace in historical.get("joint_arrival_trace", []):
        before = trace.get("before_publish", {})
        for side in ("left", "right"):
            item = before.get(side, {})
            target = item.get("target_rad")
            command = item.get("published_command_rad")
            if target is None or command is None:
                continue
            target_array = np.asarray(target, dtype=float)
            command_array = np.asarray(command, dtype=float)
            if (
                target_array.shape != (6,)
                or command_array.shape != (6,)
                or not np.all(np.isfinite(target_array))
                or not np.all(np.isfinite(command_array))
            ):
                raise ValueError("historical MOCA sample is invalid")
            samples[side].append((target_array, command_array))
    if any(not samples[side] for side in ("left", "right")):
        raise ValueError("historical MOCA source audit has no paired commands")
    statistics = {}
    for side in ("left", "right"):
        offsets = np.asarray(
            [target - command for target, command in samples[side]],
            dtype=float,
        )
        statistics[side] = {
            "sample_count": int(len(offsets)),
            "median_offset_rad": np.median(offsets, axis=0).tolist(),
            "offset_span_rad": np.ptp(offsets, axis=0).tolist(),
        }
    return samples, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "statistics": statistics,
    }


class HistoricalRawMocaRuntime(R1LiteBimanualRuntime):
    """Replay feedback targets through nearest historical raw MOCA commands."""

    def __init__(
        self,
        historical_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]],
        *,
        maximum_reference_distance_rad: float = 0.35,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.historical_samples = historical_samples
        self.maximum_reference_distance_rad = float(
            maximum_reference_distance_rad
        )
        self.historical_mapping_enabled = True
        self.historical_mapping_trace: list[dict] = []

    def _feedback_targets_to_commands(self, targets):
        if not self.historical_mapping_enabled:
            return super()._feedback_targets_to_commands(targets)
        commands = []
        for side, target in zip(("left", "right"), targets):
            target_array = np.asarray(target, dtype=float)
            candidates = self.historical_samples[side]
            distances = [
                float(np.max(np.abs(target_array - historical_target)))
                for historical_target, _ in candidates
            ]
            selected_index = int(np.argmin(distances))
            distance = distances[selected_index]
            if distance > self.maximum_reference_distance_rad:
                raise RuntimeError(
                    f"{side} historical MOCA reference is too far: "
                    f"{distance:.6f}rad"
                )
            historical_target, historical_command = candidates[selected_index]
            command = historical_command + target_array - historical_target
            commands.append(tuple(float(value) for value in command))
            self.historical_mapping_trace.append(
                {
                    "side": side,
                    "reference_index": selected_index,
                    "reference_distance_rad": distance,
                    "target_rad": target_array.tolist(),
                    "raw_command_rad": command.tolist(),
                    "historical_offset_rad": (
                        historical_target - historical_command
                    ).tolist(),
                }
            )
        return commands[0], commands[1]


class RightJoint2ControlOffsetRuntime(
    RightJoint2ControlOffsetMixin, R1LiteBimanualRuntime
):
    """Pipeline-7 runtime with explicit application-layer compensation."""


class HistoricalRawMocaOffsetRuntime(
    HistoricalMappingRightJoint2ControlOffsetMixin,
    HistoricalRawMocaRuntime,
):
    """Combine fixed historical MOCA mapping with a right joint-2 offset."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--refine-grasp-before-close", action="store_true")
    parser.add_argument("--grasp-refinement-plan", type=Path)
    parser.add_argument("--ignore-grasp-refinement-collision", action="store_true")
    parser.add_argument("--ignore-absolute-contact-height", action="store_true")
    parser.add_argument("--saved-grasp-joint-sequence", type=Path)
    parser.add_argument("--feedback-downward-step-m", type=float)
    parser.add_argument(
        "--feedback-downward-step-candidates-m",
        type=float,
        nargs="+",
    )
    parser.add_argument("--stop-on-effort-contact", action="store_true")
    parser.add_argument("--ignore-table-height-clamp", action="store_true")
    parser.add_argument(
        "--ignore-local-fk-interpolation-z-gate",
        action="store_true",
    )
    parser.add_argument("--feedback-downward-refinement-iterations", type=int, default=0)
    parser.add_argument("--resume-after-saved-grasp-descent", action="store_true")
    parser.add_argument(
        "--resume-feedback-refinement-after-saved-descent",
        action="store_true",
    )
    parser.add_argument("--resume-after-place-release", action="store_true")
    parser.add_argument("--resume-after-pregrasp", action="store_true")
    parser.add_argument("--resume-initial-tolerance-rad", type=float, default=0.10)
    parser.add_argument("--historical-raw-moca-audit", type=Path)
    parser.add_argument("--live-mapping-after-grasp", action="store_true")
    parser.add_argument("--live-mapping-after-release", action="store_true")
    parser.add_argument(
        "--arrival-soft-threshold-rad",
        type=float,
        default=0.02,
    )
    parser.add_argument("--stop-after-pregrasp", action="store_true")
    parser.add_argument(
        "--right-joint2-control-offset-deg",
        type=float,
        default=float(
            os.environ.get("R1LITE_RIGHT_JOINT2_CONTROL_OFFSET_DEG", "0.0")
        ),
    )
    parser.add_argument(
        "--right-gripper-open-position",
        type=float,
        default=70.0,
    )
    parser.add_argument(
        "--right-gripper-release-position",
        type=float,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.right_gripper_open_position <= 100.0:
        raise ValueError("--right-gripper-open-position must be within [0, 100]")
    right_gripper_release_position = (
        args.right_gripper_open_position
        if args.right_gripper_release_position is None
        else args.right_gripper_release_position
    )
    if not 0.0 <= right_gripper_release_position <= 100.0:
        raise ValueError(
            "--right-gripper-release-position must be within [0, 100]"
        )
    raw = args.bundle.read_bytes()
    bundle = json.loads(raw)
    bundle_sha256 = hashlib.sha256(raw).hexdigest()
    historical_moca_samples = None
    historical_moca_metadata = None
    if args.historical_raw_moca_audit is not None:
        historical_moca_samples, historical_moca_metadata = (
            load_historical_moca_samples(
                args.historical_raw_moca_audit,
                expected_bundle_sha256=bundle_sha256,
            )
        )
    saved_grasp_sequence = None
    if args.saved_grasp_joint_sequence is not None:
        if args.refine_grasp_before_close:
            raise ValueError(
                "saved grasp joint replay and online refinement are mutually exclusive"
            )
        saved_grasp_sequence = json.loads(
            args.saved_grasp_joint_sequence.read_text()
        )
        if saved_grasp_sequence.get("schema") != (
            "capx.r1lite.saved_grasp_joint_sequence.v1"
        ):
            raise ValueError("unexpected saved grasp joint sequence schema")
        if saved_grasp_sequence.get(
            "source_execution_bundle_sha256"
        ) != bundle_sha256:
            raise ValueError("saved grasp sequence does not match replay bundle")
    plan = None
    plan_sha256 = None
    if (
        args.refine_grasp_before_close
        or args.feedback_downward_refinement_iterations > 0
    ):
        if (
            args.feedback_downward_refinement_iterations > 0
            and (
                args.feedback_downward_step_m is None
                or args.feedback_downward_step_m <= 0
            )
        ):
            raise ValueError(
                "positive --feedback-downward-step-m is required"
            )
        if args.grasp_refinement_plan is None:
            raise ValueError(
                "--grasp-refinement-plan is required with grasp refinement"
            )
        plan_raw = args.grasp_refinement_plan.read_bytes()
        plan = json.loads(plan_raw)
        plan_sha256 = hashlib.sha256(plan_raw).hexdigest()
        if bundle["artifact_bindings"]["plan_sha256"] != plan_sha256:
            raise ValueError("grasp refinement plan does not match saved bundle")
        if args.refine_grasp_before_close and args.ignore_grasp_refinement_collision:
            plan = deepcopy(plan)
            plan["safety"]["grasp_post_arrival_probe_policy"][
                "check_collision_during_absolute_height_refinement"
            ] = False
        if args.refine_grasp_before_close and args.ignore_absolute_contact_height:
            plan = deepcopy(plan)
            probe_policy = plan["safety"]["grasp_post_arrival_probe_policy"]
            probe_policy["ignore_absolute_contact_height"] = True
            probe_policy["incremental_refinement_step_m"] = 0.0005
    elif args.ignore_grasp_refinement_collision:
        raise ValueError(
            "collision bypass requires --refine-grasp-before-close"
        )
    elif args.ignore_absolute_contact_height:
        raise ValueError(
            "absolute-height bypass requires --refine-grasp-before-close"
        )
    stages = {stage["id"]: stage for stage in bundle["stages"]}
    order = [
        "pregrasp",
        "align_grasp",
        "grasp",
        "lift",
        "place_center",
        "release_retract",
        "home",
    ]
    if bundle.get("stage_order") != order:
        raise ValueError("saved bundle stage order is not the expected Pipeline-7 order")
    if args.feedback_downward_refinement_iterations > 0:
        if saved_grasp_sequence is None:
            raise ValueError(
                "feedback downward refinement requires a saved grasp joint sequence"
            )
        if args.feedback_downward_step_m is None or not (
            0.0001 <= args.feedback_downward_step_m <= 0.01
        ):
            raise ValueError("feedback downward step must be within [0.1, 10]mm")
        if args.feedback_downward_step_candidates_m is not None and any(
            not 0.0001 <= step <= 0.01
            for step in args.feedback_downward_step_candidates_m
        ):
            raise ValueError(
                "feedback downward step candidates must be within [0.1, 10]mm"
            )
        if args.feedback_downward_refinement_iterations > 20:
            raise ValueError("feedback downward refinements cannot exceed 20")

    audit = {
        "schema": "capx.r1lite.saved_ik_replay.v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "bundle": str(args.bundle),
        "bundle_sha256": bundle_sha256,
        "online_ik_enabled": bool(
            args.refine_grasp_before_close
            or args.feedback_downward_refinement_iterations > 0
        ),
        "cartesian_refinement_enabled": bool(
            args.refine_grasp_before_close
            or args.feedback_downward_refinement_iterations > 0
        ),
        "refinement_stage_ids": (
            ["grasp"]
            if args.refine_grasp_before_close
            or saved_grasp_sequence is not None
            else []
        ),
        "saved_grasp_joint_sequence": (
            None
            if args.saved_grasp_joint_sequence is None
            else str(args.saved_grasp_joint_sequence)
        ),
        "saved_grasp_joint_replay_enabled": saved_grasp_sequence is not None,
        "grasp_refinement_collision_check_enabled": (
            False
            if saved_grasp_sequence is not None
            else not bool(args.ignore_grasp_refinement_collision)
        ),
        "absolute_contact_height_ignored": bool(
            args.ignore_absolute_contact_height
        ),
        "progress_steps_per_saved_waypoint": 1,
        "feedback_downward_refinement": {
            "enabled": args.feedback_downward_refinement_iterations > 0,
            "step_m": args.feedback_downward_step_m,
            "step_candidates_m": args.feedback_downward_step_candidates_m,
            "maximum_iterations": args.feedback_downward_refinement_iterations,
            "stop_condition": (
                "effort_contact_or_maximum_iterations"
                if args.stop_on_effort_contact
                else "fingertip_at_table_or_maximum_iterations"
            ),
            "table_z_m": (
                None
                if plan is None
                else float(plan["safety"]["table_z_m"])
            ),
            "tcp_to_fingertip_m": (
                None
                if plan is None
                else float(plan["safety"]["tcp_to_fingertip_m"])
            ),
            "xy_drift_check_enabled": False,
            "effort_contact_stop_enabled": args.stop_on_effort_contact,
            "table_height_clamp_enabled": not args.ignore_table_height_clamp,
            "fk_candidate_error_axis": "z_only",
            "maximum_fk_z_error_m": 0.015,
            "local_fk_interpolation_z_gate_enabled": (
                not args.ignore_local_fk_interpolation_z_gate
            ),
            "local_fk_maximum_upward_step_m": 0.0001,
        },
        "resume_after_saved_grasp_descent": bool(
            args.resume_after_saved_grasp_descent
        ),
        "resume_feedback_refinement_after_saved_descent": bool(
            args.resume_feedback_refinement_after_saved_descent
        ),
        "resume_after_pregrasp": bool(args.resume_after_pregrasp),
        "command_mapping": (
            {
                "mode": "nearest_historical_raw_moca_command",
                **historical_moca_metadata,
            }
            if historical_moca_metadata is not None
            else {"mode": "live_feedback_control_reference_offset"}
        ),
        "stop_after_pregrasp": bool(args.stop_after_pregrasp),
        "live_mapping_after_grasp": bool(args.live_mapping_after_grasp),
        "live_mapping_after_release": bool(args.live_mapping_after_release),
        "arrival_soft_threshold_rad": float(
            args.arrival_soft_threshold_rad
        ),
        "right_joint2_control_offset_deg": float(
            args.right_joint2_control_offset_deg
        ),
        "right_gripper_open_position": float(
            args.right_gripper_open_position
        ),
        "right_gripper_release_position": float(
            right_gripper_release_position
        ),
        "passed": False,
        "events": [],
    }
    if (
        historical_moca_samples is not None
        and abs(args.right_joint2_control_offset_deg) > 1e-12
    ):
        controller_class = HistoricalRawMocaOffsetRuntime
    elif historical_moca_samples is not None:
        controller_class = HistoricalRawMocaRuntime
    elif abs(args.right_joint2_control_offset_deg) > 1e-12:
        controller_class = RightJoint2ControlOffsetRuntime
    else:
        controller_class = R1LiteBimanualRuntime
    controller_kwargs = {
        "config": BimanualRuntimeConfig(
            arrival_timeout_s=8.0,
            arrival_soft_threshold_rad=args.arrival_soft_threshold_rad,
            arrival_velocity_threshold_rad_s=0.10,
            arrival_settle_time_s=0.20,
            gripper_open_position=args.right_gripper_open_position,
        )
    }
    if historical_moca_samples is not None:
        controller_kwargs["historical_samples"] = historical_moca_samples
    if controller_class in (
        RightJoint2ControlOffsetRuntime,
        HistoricalRawMocaOffsetRuntime,
    ):
        controller_kwargs["right_joint2_control_offset_rad"] = np.deg2rad(
            args.right_joint2_control_offset_deg
        )
    controller = controller_class(
        **controller_kwargs
    )

    def replay_stage(
        stage_id: str,
        *,
        bypass_arrival_gate: bool = False,
    ) -> None:
        stage = stages[stage_id]
        pairs = zip(
            stage["left"]["joint_waypoints_rad"],
            stage["right"]["joint_waypoints_rad"],
        )
        for index, (left, right) in enumerate(pairs, start=1):
            progress_steps = 20 if stage_id == "pregrasp" and index == 1 else 1
            if bypass_arrival_gate:
                left_target = np.asarray(left, dtype=float)
                right_target = np.asarray(right, dtype=float)
                controller.publish_joint_targets_for(
                    left_target,
                    right_target,
                    duration_s=2.0,
                )
            else:
                controller.execute_joint_motion(
                    np.asarray(left, dtype=float),
                    np.asarray(right, dtype=float),
                    progress_steps=progress_steps,
                )
            audit["events"].append(
                {
                    "event": "saved_joint_waypoint_replayed",
                    "stage_id": stage_id,
                    "waypoint_index": index,
                    "left_joints_rad": left,
                    "right_joints_rad": right,
                    "progress_steps": progress_steps,
                    "arrival_gate_bypassed": bypass_arrival_gate,
                    "fixed_dwell_s": 2.0 if bypass_arrival_gate else None,
                }
            )

    def close_right_gripper_and_wait_stable() -> None:
        target = controller.config.gripper_close_position
        feedback_deadline = time.monotonic() + 1.0
        while True:
            try:
                initial_position = controller.get_gripper_feedback("right")
                break
            except FeedbackStaleError:
                if time.monotonic() >= feedback_deadline:
                    raise
                controller.refresh_feedback(0.02)
        started_at = time.monotonic()
        deadline = started_at + controller.config.gripper_timeout_s
        samples: list[tuple[float, float]] = []
        while time.monotonic() < deadline:
            controller.publish_gripper("right", target)
            controller.refresh_feedback(0.02)
            now = time.monotonic()
            position = controller.get_gripper_feedback("right")
            samples.append((now, position))
            samples = [
                sample for sample in samples if sample[0] >= now - 0.5
            ]
            reached = (
                abs(position - target)
                <= controller.config.gripper_tolerance
            )
            moved_for_contact = abs(position - initial_position) >= 2.0
            stable = (
                len(samples) >= 5
                and samples[-1][0] - samples[0][0] >= 0.4
                and max(value for _, value in samples)
                - min(value for _, value in samples)
                <= 0.5
            )
            if now - started_at >= 2.0 and (
                reached or (moved_for_contact and stable)
            ):
                audit["events"].append(
                    {
                        "event": "right_gripper_close_stabilized",
                        "initial_position": initial_position,
                        "final_position": position,
                        "reached_close_target": reached,
                        "stable_contact_position": stable,
                        "minimum_close_command_duration_s": 2.0,
                    }
                )
                return
        raise RuntimeError(
            "right gripper did not reach a stable closed/contact position"
        )

    def sample_right_contact(
        reference_effort: np.ndarray | None,
        *,
        sample_count: int | None = None,
        residual_threshold_nm: float | None = None,
        require_settled_velocity: bool = True,
    ) -> tuple[np.ndarray | None, bool, dict]:
        assert plan is not None
        policy = plan["safety"]["grasp_effort_contact_policy"]
        efforts = []
        velocities = []
        count = (
            int(policy["sample_count"])
            if sample_count is None
            else int(sample_count)
        )
        for _ in range(count):
            controller.refresh_feedback(0.01)
            telemetry = controller.get_joint_telemetry("right")
            efforts.append(np.asarray(telemetry["effort"], dtype=float))
            velocities.append(np.asarray(telemetry["velocity"], dtype=float))
        effort = np.median(np.asarray(efforts), axis=0)
        velocity = np.median(np.abs(np.asarray(velocities)), axis=0)
        if reference_effort is None:
            return effort, False, {
                "available": True,
                "detected": False,
                "effort_nm": effort.tolist(),
            }
        indices = [int(value) for value in policy["joint_indices_zero_based"]]
        residual = effort - reference_effort
        score = float(np.sum(np.abs(residual[indices])))
        maximum_velocity = float(np.max(velocity))
        threshold = (
            float(policy["sum_abs_residual_threshold_nm"])
            if residual_threshold_nm is None
            else float(residual_threshold_nm)
        )
        detected = (
            score >= threshold
            and (
                not require_settled_velocity
                or maximum_velocity
                <= float(policy["maximum_median_abs_velocity_rad_s"])
            )
        )
        return effort, detected, {
            "available": True,
            "detected": detected,
            "effort_nm": effort.tolist(),
            "effort_residual_nm": residual.tolist(),
            "joint_residual_score_nm": score,
            "residual_threshold_nm": threshold,
            "median_max_abs_velocity_rad_s": maximum_velocity,
            "settled_velocity_required": require_settled_velocity,
        }

    def execute_right_descent_with_contact_monitor(
        *,
        target: np.ndarray,
        inactive_hold: np.ndarray,
        reference_effort: np.ndarray,
    ) -> tuple[bool, str | None, list[dict]]:
        """Publish one descent target and inspect contact during arrival."""

        def fresh_right_tool_position() -> np.ndarray:
            deadline = time.monotonic() + 1.0
            while True:
                try:
                    return np.asarray(
                        controller.get_tool_pose("right")[0],
                        dtype=float,
                    )
                except FeedbackStaleError:
                    if time.monotonic() >= deadline:
                        raise
                    controller.refresh_feedback(0.02)

        before_position = fresh_right_tool_position()
        fixed_reference_effort = reference_effort.copy()
        samples = []

        def contact_stop_requested() -> bool:
            _, detected, evidence = sample_right_contact(
                fixed_reference_effort,
                sample_count=5,
                residual_threshold_nm=2.5,
                require_settled_velocity=False,
            )
            position = fresh_right_tool_position()
            samples.append(
                {
                    "sample_index": len(samples) + 1,
                    "target_right_joints_rad": target.tolist(),
                    "tcp_position_base_m": position.tolist(),
                    **evidence,
                }
            )
            if detected:
                samples[-1]["stop_reason"] = "effort_contact"
                return True
            return False

        stopped = controller.execute_single_arm_motion(
            side="right",
            target=target,
            inactive_hold=inactive_hold,
            progress_steps=1,
            stop_requested=contact_stop_requested,
        )
        if stopped:
            return True, "effort_contact", samples
        after_position = fresh_right_tool_position()
        if after_position[2] - before_position[2] >= 0.0001:
            controller.hold()
            return False, "measured_tcp_z_did_not_decrease", samples
        return False, None, samples

    try:
        controller.start()
        controller.wait_until_ready(require_tool_pose=True)
        feedback = controller.get_feedback_pair()
        resume_mode_count = sum(
            (
                args.resume_after_pregrasp,
                args.resume_after_saved_grasp_descent,
                args.resume_feedback_refinement_after_saved_descent,
                args.resume_after_place_release,
            )
        )
        if resume_mode_count > 1:
            raise ValueError("resume modes are mutually exclusive")
        if args.resume_after_place_release:
            expected_initial = {
                side: np.asarray(
                    stages["place_center"][side]["joint_waypoints_rad"][-1],
                    dtype=float,
                )
                for side in ("left", "right")
            }
        elif (
            args.resume_after_saved_grasp_descent
            or args.resume_feedback_refinement_after_saved_descent
        ):
            if saved_grasp_sequence is None:
                raise ValueError(
                    "resume after grasp descent requires saved grasp sequence"
                )
            command_index = (
                1
                if args.resume_feedback_refinement_after_saved_descent
                else -1
            )
            expected_initial = {
                side: np.asarray(
                    saved_grasp_sequence["commands"][command_index][
                        f"{side}_joints_rad"
                    ],
                    dtype=float,
                )
                for side in ("left", "right")
            }
        elif args.resume_after_pregrasp:
            expected_initial = {
                side: np.asarray(
                    stages["pregrasp"][side]["joint_waypoints_rad"][-1],
                    dtype=float,
                )
                for side in ("left", "right")
            }
        else:
            expected_initial = {
                side: np.asarray(
                    bundle["initial_state"][f"{side}_joints_rad"],
                    dtype=float,
                )
                for side in ("left", "right")
            }
        if abs(args.right_joint2_control_offset_deg) > 1e-12:
            expected_initial["right"] = expected_initial["right"].copy()
            expected_initial["right"][1] += np.deg2rad(
                args.right_joint2_control_offset_deg
            )
        initial_errors = {
            side: float(
                np.max(
                    np.abs(
                        np.asarray(feedback[side], dtype=float)
                        - expected_initial[side]
                    )
                )
            )
            for side in ("left", "right")
        }
        audit["initial_joint_error_rad"] = initial_errors
        if (
            not args.resume_after_place_release
            and max(initial_errors.values()) > args.resume_initial_tolerance_rad
        ):
            raise RuntimeError(f"saved replay initial gate failed: {initial_errors}")
        if args.resume_after_place_release:
            audit["events"].append(
                {"event": "place_release_resume_gate_passed"}
            )
            controller.publish_gripper(
                "right", right_gripper_release_position
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                controller.refresh_feedback()
            audit["events"].append(
                {
                    "event": "right_gripper_open_published_without_arrival_gate",
                    "target_position": right_gripper_release_position,
                    "seconds": 2.0,
                }
            )
            for stage_id in ("release_retract", "home"):
                replay_stage(stage_id, bypass_arrival_gate=True)
            audit["passed"] = True
            return
        if args.resume_after_saved_grasp_descent:
            audit["events"].append(
                {"event": "saved_grasp_descent_resume_gate_passed"}
            )
            close_right_gripper_and_wait_stable()
            audit["events"].append({"event": "right_gripper_closed"})
            if (
                args.live_mapping_after_grasp
                and isinstance(controller, HistoricalRawMocaRuntime)
            ):
                controller.historical_mapping_enabled = False
                audit["events"].append(
                    {
                        "event": "command_mapping_switched_after_grasp",
                        "mode": "live_feedback_control_reference_offset",
                    }
                )
            for stage_id in ("lift", "place_center"):
                replay_stage(stage_id, bypass_arrival_gate=True)
            controller.publish_gripper(
                "right", right_gripper_release_position
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                controller.refresh_feedback()
            audit["events"].append(
                {
                    "event": "right_gripper_open_published_without_arrival_gate",
                    "target_position": right_gripper_release_position,
                    "seconds": 2.0,
                }
            )
            for stage_id in ("release_retract", "home"):
                replay_stage(stage_id, bypass_arrival_gate=True)
            audit["passed"] = True
            return

        if args.resume_feedback_refinement_after_saved_descent:
            audit["events"].append(
                {
                    "event": (
                        "feedback_refinement_after_saved_descent_"
                        "resume_gate_passed"
                    )
                }
            )
        elif args.resume_after_pregrasp:
            audit["events"].append(
                {"event": "historical_raw_moca_pregrasp_resume_gate_passed"}
            )
        else:
            initial_gripper_open_position = (
                controller.config.gripper_open_position
            )
            controller.command_gripper("right", initial_gripper_open_position)
            audit["events"].append(
                {
                    "event": "right_gripper_opened",
                    "target_position": initial_gripper_open_position,
                }
            )
            replay_stage("pregrasp")
        if args.stop_after_pregrasp:
            if args.resume_after_pregrasp:
                raise ValueError(
                    "cannot stop after pregrasp while resuming after pregrasp"
                )
            controller.hold()
            audit["events"].append(
                {"event": "historical_raw_moca_pregrasp_hold"}
            )
            audit["passed"] = True
            return
        if not args.resume_feedback_refinement_after_saved_descent:
            replay_stage("align_grasp")
        if saved_grasp_sequence is not None:
            commands = saved_grasp_sequence.get("commands")
            expected_ids = [
                "saved_grasp",
                "initial_10mm_descent",
                "incremental_0.5mm_refinement",
            ]
            if not isinstance(commands, list) or [
                item.get("id") for item in commands
            ] != expected_ids:
                raise ValueError("saved grasp command order is invalid")
            contact_reference_effort = None
            saved_descent_stop_reason = None
            saved_descent_contact_detected = False
            commands_to_replay = commands
            command_index_start = 1
            if args.resume_feedback_refinement_after_saved_descent:
                (
                    contact_reference_effort,
                    _,
                    baseline_evidence,
                ) = sample_right_contact(None)
                audit["events"].append(
                    {
                        "event": "feedback_downward_contact_baseline",
                        "capture": (
                            "resume_at_saved_initial_10mm_descent_endpoint"
                        ),
                        **baseline_evidence,
                    }
                )
                commands_to_replay = [commands[2]]
                command_index_start = 3
            for index, command in enumerate(
                commands_to_replay,
                start=command_index_start,
            ):
                left = np.asarray(command["left_joints_rad"], dtype=float)
                right = np.asarray(command["right_joints_rad"], dtype=float)
                if (
                    left.shape != (6,)
                    or right.shape != (6,)
                    or not np.isfinite(left).all()
                    or not np.isfinite(right).all()
                ):
                    raise ValueError("saved grasp joint command is invalid")
                if (
                    command["id"] == "incremental_0.5mm_refinement"
                    and args.feedback_downward_refinement_iterations > 0
                ):
                    audit["events"].append(
                        {
                            "event": "saved_grasp_joint_command_skipped",
                            "command_index": index,
                            "command_id": command["id"],
                            "reason": "replaced_by_feedback_table_refinement",
                        }
                    )
                    continue
                skip_arrival_gate = (
                    command["id"] == "incremental_0.5mm_refinement"
                )
                monitored_initial_descent = False
                monitor_samples = None
                if monitored_initial_descent:
                    assert contact_reference_effort is not None
                    (
                        saved_descent_contact_detected,
                        saved_descent_stop_reason,
                        monitor_samples,
                    ) = execute_right_descent_with_contact_monitor(
                        target=right,
                        inactive_hold=left,
                        reference_effort=contact_reference_effort,
                    )
                elif not skip_arrival_gate:
                    controller.execute_joint_motion(
                        left,
                        right,
                        progress_steps=1,
                    )
                audit["events"].append(
                    {
                        "event": "saved_grasp_joint_command_replayed",
                        "command_index": index,
                        "command_id": command["id"],
                        "left_joints_rad": left.tolist(),
                        "right_joints_rad": right.tolist(),
                        "arrival_gate_skipped": skip_arrival_gate,
                        "contact_monitored": monitored_initial_descent,
                        "contact_monitor_samples": monitor_samples,
                        "stop_reason": (
                            saved_descent_stop_reason
                            if monitored_initial_descent
                            else None
                        ),
                    }
                )
                if (
                    command["id"] == "saved_grasp"
                    and args.stop_on_effort_contact
                ):
                    (
                        contact_reference_effort,
                        _,
                        baseline_evidence,
                    ) = sample_right_contact(None)
                    audit["events"].append(
                        {
                            "event": "feedback_downward_contact_baseline",
                            **baseline_evidence,
                        }
                    )
                if monitored_initial_descent and saved_descent_stop_reason:
                    continue
                if (
                    command["id"] == "initial_10mm_descent"
                    and args.stop_on_effort_contact
                ):
                    (
                        contact_reference_effort,
                        _,
                        baseline_evidence,
                    ) = sample_right_contact(None)
                    audit["events"].append(
                        {
                            "event": "feedback_downward_contact_baseline",
                            "capture": "after_unsegmented_initial_10mm_descent",
                            **baseline_evidence,
                        }
                    )
                if skip_arrival_gate:
                    controller.publish_joint_targets_for(
                        left,
                        right,
                        duration_s=2.0,
                    )
                    audit["events"].append(
                        {
                            "event": "post_refinement_pre_close_dwell_completed",
                            "seconds": 2.0,
                        }
                    )
            if (
                args.feedback_downward_refinement_iterations > 0
                and saved_descent_stop_reason is None
            ):
                assert plan is not None
                solver = IsolatedBimanualRelaxedIK(
                    controller,
                    bundle["initial_state"]["torso_joints_rad"],
                )
                local_fk = R1LiteCandidateKinematics(side="right")
                refinement_events = []
                fingertip_reached_table = False
                effort_contact_detected = False
                forced_close_reason = None
                table_z_m = float(plan["safety"]["table_z_m"])
                tcp_to_fingertip_m = float(
                    plan["safety"]["tcp_to_fingertip_m"]
                )
                table_tcp_z_m = table_z_m + tcp_to_fingertip_m
                table_reached_tolerance_m = 0.0002
                step_candidates_m = (
                    args.feedback_downward_step_candidates_m
                    if args.feedback_downward_step_candidates_m is not None
                    else [float(args.feedback_downward_step_m)]
                )
                for iteration in range(
                    1, args.feedback_downward_refinement_iterations + 1
                ):
                    before_position, before_quaternion = (
                        controller.get_tool_pose_pair()["right"]
                    )
                    before = np.asarray(before_position, dtype=float)
                    before_fingertip_z_m = (
                        float(before[2]) - tcp_to_fingertip_m
                    )
                    if (
                        not args.ignore_table_height_clamp
                        and
                        before_fingertip_z_m
                        <= table_z_m + table_reached_tolerance_m
                    ):
                        fingertip_reached_table = True
                        refinement_events.append(
                            {
                                "iteration": iteration,
                                "completed": False,
                                "reason": "fingertip_already_at_table",
                                "before_tcp_position_base_m": before.tolist(),
                                "before_fingertip_z_m": before_fingertip_z_m,
                                "table_z_m": table_z_m,
                            }
                        )
                        controller.hold()
                        break
                    current = controller.get_feedback_pair()
                    current_right = np.asarray(current["right"], dtype=float)
                    depth_attempts = []
                    selected_result = None
                    for requested_step_m in step_candidates_m:
                        target = before.copy()
                        requested_target_z_m = (
                            float(before[2]) - requested_step_m
                        )
                        target[2] = (
                            requested_target_z_m
                            if args.ignore_table_height_clamp
                            else max(requested_target_z_m, table_tcp_z_m)
                        )
                        qualified = []
                        rejection_counts = {
                            "fk_z_error": 0,
                            "joint_delta": 0,
                            "interpolation_z_upward": 0,
                        }
                        candidates = solver.solve_candidates(
                            side="right",
                            target_position_base_m=target,
                            target_quaternion_xyzw=before_quaternion,
                            reference_joints_rad=current_right,
                            probes=20,
                        )
                        for candidate in candidates:
                            joints = np.asarray(candidate, dtype=float)
                            pose = local_fk.fk(
                                joints,
                                np.asarray(
                                    bundle["initial_state"]["torso_joints_rad"],
                                    dtype=float,
                                ),
                            )
                            fk_z_error = float(
                                abs(pose.position_base_m[2] - target[2])
                            )
                            joint_delta = float(
                                np.max(np.abs(joints - current_right))
                            )
                            midpoint = (current_right + joints) / 2.0
                            interpolation_z = np.asarray(
                                [
                                    local_fk.fk(
                                        value,
                                        np.asarray(
                                            bundle["initial_state"][
                                                "torso_joints_rad"
                                            ],
                                            dtype=float,
                                        ),
                                    ).position_base_m[2]
                                    for value in (
                                        current_right,
                                        midpoint,
                                        joints,
                                    )
                                ]
                            )
                            maximum_upward_step = float(
                                np.max(np.diff(interpolation_z))
                            )
                            if fk_z_error > 0.015:
                                rejection_counts["fk_z_error"] += 1
                            if joint_delta > 0.25:
                                rejection_counts["joint_delta"] += 1
                            if (
                                not args.ignore_local_fk_interpolation_z_gate
                                and maximum_upward_step > 0.0001
                            ):
                                rejection_counts[
                                    "interpolation_z_upward"
                                ] += 1
                            if (
                                fk_z_error <= 0.015
                                and joint_delta <= 0.25
                                and (
                                    args.ignore_local_fk_interpolation_z_gate
                                    or maximum_upward_step <= 0.0001
                                )
                            ):
                                qualified.append(
                                    (
                                        fk_z_error,
                                        joint_delta,
                                        maximum_upward_step,
                                        joints,
                                    )
                                )
                        depth_attempts.append(
                            {
                                "requested_step_m": requested_step_m,
                                "effective_step_m": float(
                                    before[2] - target[2]
                                ),
                                "target_tcp_position_base_m": target.tolist(),
                                "candidate_count": len(candidates),
                                "qualified_candidate_count": len(qualified),
                                "rejection_counts": rejection_counts,
                            }
                        )
                        if qualified:
                            selected_result = (
                                target,
                                requested_step_m,
                                *min(qualified, key=lambda item: item[:2]),
                            )
                            break
                    if selected_result is None:
                        forced_close_reason = "no_qualified_ik_candidate"
                        refinement_events.append(
                            {
                                "iteration": iteration,
                                "completed": False,
                                "reason": forced_close_reason,
                                "depth_attempts": depth_attempts,
                            }
                        )
                        break
                    (
                        target,
                        selected_step_m,
                        fk_z_error,
                        joint_delta,
                        maximum_upward_step,
                        selected,
                    ) = selected_result
                    (
                        effort_contact_detected,
                        monitored_stop_reason,
                        contact_monitor_samples,
                    ) = execute_right_descent_with_contact_monitor(
                        target=selected,
                        inactive_hold=np.asarray(
                            current["left"],
                            dtype=float,
                        ),
                        reference_effort=contact_reference_effort,
                    )
                    after_position, _ = controller.get_tool_pose("right")
                    after = np.asarray(after_position, dtype=float)
                    measured_delta = after - before
                    after_fingertip_z_m = (
                        float(after[2]) - tcp_to_fingertip_m
                    )
                    contact_evidence = (
                        contact_monitor_samples[-1]
                        if contact_monitor_samples
                        else {}
                    )
                    refinement_event = {
                        "iteration": iteration,
                        "before_tcp_position_base_m": before.tolist(),
                        "target_tcp_position_base_m": target.tolist(),
                        "after_tcp_position_base_m": after.tolist(),
                        "measured_delta_base_m": measured_delta.tolist(),
                        "selected_requested_step_m": selected_step_m,
                        "selected_effective_step_m": float(
                            before[2] - target[2]
                        ),
                        "fk_z_error_m": fk_z_error,
                        "maximum_joint_delta_rad": joint_delta,
                        "local_fk_maximum_upward_step_m": maximum_upward_step,
                        "before_fingertip_z_m": before_fingertip_z_m,
                        "after_fingertip_z_m": after_fingertip_z_m,
                        "table_z_m": table_z_m,
                        "depth_attempts": depth_attempts,
                        "contact_monitor_samples": contact_monitor_samples,
                        **contact_evidence,
                    }
                    if effort_contact_detected:
                        refinement_event.update(
                            {
                                "completed": True,
                                "stop_reason": "effort_contact",
                            }
                        )
                        refinement_events.append(refinement_event)
                        controller.hold()
                        break
                    if monitored_stop_reason is not None:
                        forced_close_reason = monitored_stop_reason
                        refinement_event.update(
                            {
                                "completed": False,
                                "reason": forced_close_reason,
                            }
                        )
                        refinement_events.append(refinement_event)
                        controller.hold()
                        break
                    refinement_events.append(refinement_event)
                    if (
                        not args.ignore_table_height_clamp
                        and
                        after_fingertip_z_m
                        <= table_z_m + table_reached_tolerance_m
                    ):
                        fingertip_reached_table = True
                        controller.hold()
                        break
                audit["events"].append(
                    {
                        "event": "feedback_downward_table_refinement_completed",
                        "fingertip_reached_table": fingertip_reached_table,
                        "effort_contact_detected": effort_contact_detected,
                        "completed_refinement_iterations": len(
                            refinement_events
                        ),
                        "close_authorized_by": (
                            "effort_contact"
                            if effort_contact_detected
                            else (
                                "fingertip_reached_table"
                                if fingertip_reached_table
                                else (
                                    f"forced_close_after_{forced_close_reason}"
                                    if forced_close_reason is not None
                                    else (
                                        "maximum_"
                                        f"{args.feedback_downward_refinement_iterations}"
                                        "_refinements_completed"
                                    )
                                )
                            )
                        ),
                        "refinements": refinement_events,
                    }
                )
            elif saved_descent_stop_reason is not None:
                audit["events"].append(
                    {
                        "event": "feedback_downward_table_refinement_completed",
                        "fingertip_reached_table": False,
                        "effort_contact_detected": (
                            saved_descent_contact_detected
                        ),
                        "completed_refinement_iterations": 0,
                        "close_authorized_by": (
                            saved_descent_stop_reason
                        ),
                        "refinements": [],
                    }
                )
        elif args.refine_grasp_before_close:
            assert plan is not None and plan_sha256 is not None
            safety = R1LiteTableCollisionKinematics(
                active_side="right",
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
            executor = BimanualTowelExecutor(
                plan=plan,
                plan_sha256=plan_sha256,
                bundle=bundle,
                controller=controller,
                solver=IsolatedBimanualRelaxedIK(
                    controller,
                    bundle["initial_state"]["torso_joints_rad"],
                ),
                safety=safety,
            )
            executor.bind_plan(sha256=plan_sha256)
            executor._completed = ["pregrasp", "align_grasp"]
            executor.execute_single_arm_stage(side="right", stage_id="grasp")
            audit["events"].extend(executor.trace)
        else:
            replay_stage("grasp")
        close_right_gripper_and_wait_stable()
        audit["events"].append({"event": "right_gripper_closed"})
        if (
            args.live_mapping_after_grasp
            and isinstance(controller, HistoricalRawMocaRuntime)
        ):
            controller.historical_mapping_enabled = False
            audit["events"].append(
                {
                    "event": "command_mapping_switched_after_grasp",
                    "mode": "live_feedback_control_reference_offset",
                }
            )
        for stage_id in ("lift", "place_center"):
            replay_stage(stage_id, bypass_arrival_gate=True)
        controller.publish_gripper("right", right_gripper_release_position)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            controller.refresh_feedback()
        audit["events"].append(
            {
                "event": "right_gripper_open_published_without_arrival_gate",
                "target_position": right_gripper_release_position,
                "seconds": 2.0,
            }
        )
        if (
            args.live_mapping_after_release
            and isinstance(controller, HistoricalRawMocaRuntime)
        ):
            controller.historical_mapping_enabled = False
            audit["events"].append(
                {
                    "event": "command_mapping_switched_after_release",
                    "mode": "live_feedback_control_reference_offset",
                }
            )
        for stage_id in ("release_retract", "home"):
            replay_stage(stage_id, bypass_arrival_gate=True)
        audit["passed"] = True
    except Exception as exc:
        audit["error"] = repr(exc)
        try:
            controller.hold()
        except Exception:
            pass
        raise
    finally:
        controller.close()
        audit["joint_arrival_trace"] = controller.joint_arrival_trace
        if isinstance(controller, HistoricalRawMocaRuntime):
            audit["historical_moca_mapping_trace"] = (
                controller.historical_mapping_trace
            )
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
