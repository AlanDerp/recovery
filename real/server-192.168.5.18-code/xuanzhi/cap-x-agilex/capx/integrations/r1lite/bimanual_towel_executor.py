"""Closed-loop dispatcher for an approved dual-arm towel execution bundle."""

from __future__ import annotations

import json
from itertools import product
import os
import time
from typing import Any, Protocol, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.r1lite.bimanual_kinematics import (
    R1LiteBimanualKinematics,
    R1LiteBimanualTableCollisionKinematics,
    R1LiteTableCollisionKinematics,
)
from capx.integrations.r1lite.bimanual_runtime import (
    FeedbackStaleError,
    R1LiteBimanualRuntime,
)
from capx.integrations.r1lite.kinematics import R1LiteCandidateKinematics
from capx.integrations.r1lite.execution_bundle import EIGHTH_SCHEMA


class BimanualIKSolver(Protocol):
    def solve_candidates(
        self,
        *,
        side: str,
        target_position_base_m: Sequence[float],
        target_quaternion_xyzw: Sequence[float],
        reference_joints_rad: Sequence[float],
        probes: int,
    ) -> list[Sequence[float]]: ...


class BimanualTowelExecutor:
    """Execute certified seventh- and eighth-pipeline towel stages."""

    def __init__(
        self,
        *,
        plan: dict[str, Any],
        plan_sha256: str,
        bundle: dict[str, Any],
        controller: R1LiteBimanualRuntime,
        solver: BimanualIKSolver,
        safety: Any | None = None,
        saved_eighth_grasp_refinement: dict[str, Any] | None = None,
    ) -> None:
        self.plan = plan
        self.plan_sha256 = plan_sha256
        self.bundle = bundle
        self.controller = controller
        self.solver = solver
        self.saved_eighth_grasp_refinement = saved_eighth_grasp_refinement
        self._is_eighth = (
            plan.get("pipeline") == 8
            or bundle.get("schema") == EIGHTH_SCHEMA
        )
        self._collision_stage_ids = set(
            plan.get("safety", {}).get("collision_checked_stage_ids", [])
        )
        self._eighth_joint_command_progress_steps = int(
            plan.get("safety", {}).get("joint_command_progress_steps", 1)
        )
        if self._is_eighth and self._eighth_joint_command_progress_steps != 5:
            raise ValueError("pipeline eight requires exactly 5 joint progress steps")
        self._collision_checks_enabled = (
            bool(self._collision_stage_ids) if self._is_eighth else True
        )
        if safety is not None:
            self.safety = safety
        elif self._is_eighth and self._collision_checks_enabled:
            self.safety = R1LiteBimanualTableCollisionKinematics(
                table_z_m=float(plan["safety"]["table_z_m"]),
                tcp_to_fingertip_m=float(
                    plan["safety"].get("tcp_to_fingertip_m", 0.078)
                ),
            )
        elif not self._is_eighth:
            self.safety = R1LiteBimanualKinematics(
                minimum_tcp_separation_m=0.12552056969962836
            )
        else:
            self.safety = None
        self.torso = np.asarray(
            bundle["initial_state"]["torso_joints_rad"], dtype=float
        )
        self.local_fk = {
            side: R1LiteCandidateKinematics(side=side)
            for side in ("left", "right")
        }
        self.stages = {stage["id"]: stage for stage in bundle["stages"]}
        self.plan_stages = {stage["id"]: stage for stage in plan["stages"]}
        self.policy = plan["safety"].get("cartesian_refinement_policy", {})
        self.effort_contact_policy = plan["safety"].get(
            "grasp_effort_contact_policy", {}
        )
        self.post_arrival_probe_policy = plan["safety"].get(
            "grasp_post_arrival_probe_policy", {}
        )
        self.contact_probe_policy = plan["safety"].get(
            "contact_probe_policy", {}
        )
        self.trace: list[dict[str, Any]] = []
        self._bound = False
        self._completed: list[str] = []
        self._contact_probe_completed = False
        self._closed_gripper_sides: set[str] = set()
        self._post_close_tcp_z: dict[str, float] = {}
        self._post_close_z_policy = plan.get("safety", {}).get(
            "post_close_tcp_z_policy", {}
        )
        self._strict_p7_contact = (
            os.environ.get("R1LITE_P7_STRICT_CONTACT", "0") == "1"
        )
        self._p7_legacy_replay_refinement = (
            os.environ.get(
                "R1LITE_P7_LEGACY_REPLAY_REFINEMENT", "0"
            )
            == "1"
        )
        raw_candidate_descents = os.environ.get(
            "R1LITE_P7_CANDIDATE_DESCENTS_MM", ""
        )
        self._p7_candidate_descents_m = tuple(
            float(value.strip()) / 1000.0
            for value in raw_candidate_descents.split(",")
            if value.strip()
        )
        if any(value <= 0.0 for value in self._p7_candidate_descents_m):
            raise ValueError("P7 candidate descent distances must be positive")
        self._p7_horizontal_place_z_tolerance_m = float(
            os.environ.get(
                "R1LITE_P7_HORIZONTAL_PLACE_Z_TOLERANCE_M", "0"
            )
        )
        if self._p7_horizontal_place_z_tolerance_m < 0.0:
            raise ValueError("P7 horizontal place Z tolerance must be non-negative")
        self._p7_disable_place_z_guard = (
            os.environ.get("R1LITE_P7_DISABLE_PLACE_Z_GUARD", "0") == "1"
        )
        self._horizontal_place_planned_z_reference: float | None = None
        self._horizontal_place_measured_z_reference: float | None = None
        raw_post_grasp_override = os.environ.get(
            "R1LITE_PIPELINE7_POST_GRASP_PATH_JSON"
        )
        self._post_grasp_joint_override = (
            json.loads(raw_post_grasp_override)
            if raw_post_grasp_override
            else {}
        )
        self._post_release_right_tcp: np.ndarray | None = None
        self.right_takeover_safety = (
            R1LiteTableCollisionKinematics(
                active_side="right",
                table_z_m=float(plan["safety"]["table_z_m"]),
                tcp_to_fingertip_m=float(
                    plan["safety"].get("tcp_to_fingertip_m", 0.078)
                ),
            )
            if self._is_eighth and "right_regrasp" in self.plan_stages
            else None
        )

    def bind_plan(self, *, sha256: str) -> None:
        if sha256 != self.plan_sha256:
            raise ValueError("runtime plan SHA-256 binding mismatch")
        if self.bundle["artifact_bindings"]["plan_sha256"] != sha256:
            raise ValueError("execution bundle is not bound to runtime plan")
        self._bound = True
        self.trace.append({"event": "plan_bound", "sha256": sha256})
        for side in ("left", "right"):
            self._record_actual_tcp(
                side=side,
                stage_id="initial",
                phase="initial",
                sample_index=0,
            )

    def _record_actual_tcp(
        self,
        *,
        side: str,
        stage_id: str,
        phase: str,
        sample_index: int,
    ) -> None:
        pose = self._get_fresh_tool_pose(side)
        self.trace.append(
            {
                "event": "actual_tcp_sample",
                "side": side,
                "stage_id": stage_id,
                "phase": phase,
                "sample_index": int(sample_index),
                "position_base_m": [float(value) for value in pose[0]],
                "quaternion_xyzw": [float(value) for value in pose[1]],
            }
        )

    def _get_fresh_tool_pose_pair(
        self, *, timeout_s: float = 1.0
    ) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                return self.controller.get_tool_pose_pair()
            except FeedbackStaleError:
                if time.monotonic() >= deadline:
                    raise
                refresh = getattr(self.controller, "refresh_feedback", None)
                if refresh is not None:
                    refresh(0.02)
                else:
                    time.sleep(0.02)

    def _get_fresh_tool_pose(
        self, side: str, *, timeout_s: float = 1.0
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                getter = getattr(self.controller, "get_tool_pose", None)
                if getter is not None:
                    return getter(side)
                return self.controller.get_tool_pose_pair()[side]
            except FeedbackStaleError:
                if time.monotonic() >= deadline:
                    raise
                refresh = getattr(self.controller, "refresh_feedback", None)
                if refresh is not None:
                    refresh(0.02)
                else:
                    time.sleep(0.02)

    def _right_joint2_control_offset_rad(self) -> float:
        enabled = bool(
            getattr(
                self.controller,
                "right_joint2_control_offset_enabled",
                False,
            )
        )
        if not enabled:
            return 0.0
        return float(
            getattr(
                self.controller,
                "right_joint2_control_offset_rad",
                0.0,
            )
        )

    def _logical_feedback_pair(
        self, feedback: dict[str, Sequence[float]]
    ) -> dict[str, np.ndarray]:
        logical = {
            side: np.asarray(feedback[side], dtype=float).copy()
            for side in ("left", "right")
        }
        logical["right"][1] -= self._right_joint2_control_offset_rad()
        return logical

    def _offset_aware_tool_positions(
        self,
        poses: dict[str, tuple[Sequence[float], Sequence[float]]],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        actual = {
            side: np.asarray(poses[side][0], dtype=float).copy()
            for side in ("left", "right")
        }
        correction = np.zeros(3, dtype=float)
        offset = self._right_joint2_control_offset_rad()
        if abs(offset) > 1e-12:
            feedback = self.controller.get_feedback_pair()
            reported = np.asarray(feedback["right"], dtype=float)
            logical = reported.copy()
            logical[1] -= offset
            reported_fk = self.local_fk["right"].fk(
                reported, self.torso
            ).position_base_m
            logical_fk = self.local_fk["right"].fk(
                logical, self.torso
            ).position_base_m
            correction = np.asarray(logical_fk - reported_fk, dtype=float)
            actual["right"] += correction
        return actual, correction

    def open_both_grippers(self) -> None:
        self._require_bound()
        clear_hold = getattr(self.controller, "clear_gripper_hold", None)
        if clear_hold is not None:
            clear_hold()
        self.controller.open_both_grippers()
        self.trace.append(
            {
                "event": "grippers_opened",
                "continuous_close_hold_cleared": True,
            }
        )

    def set_both_grippers(self, position: float) -> None:
        self._require_bound()
        target = float(position)
        if not 0.0 <= target <= 100.0:
            raise ValueError("gripper position must be within [0, 100]")
        clear_hold = getattr(self.controller, "clear_gripper_hold", None)
        if clear_hold is not None:
            clear_hold()
        self.controller.command_grippers(target)
        self.trace.append(
            {"event": "grippers_positioned", "target_position": target}
        )

    def close_both_grippers(self) -> None:
        self._require_bound()
        if (
            self._is_eighth
            and self.contact_probe_policy.get("enabled") is True
            and not self._contact_probe_completed
        ):
            raise RuntimeError(
                "pipeline-eight grippers cannot close before dual-arm contact probe"
            )
        if self._is_eighth:
            evidence = self.controller.close_both_grippers_stable_and_hold()
            self.trace.append(
                {
                    "event": "grippers_close_stabilized",
                    **evidence,
                }
            )
            self.trace.append(
                {
                    "event": "grippers_closed",
                    "continuous_close_hold_enabled": True,
                }
            )
        else:
            self.controller.close_both_grippers()
            self.trace.append({"event": "grippers_closed"})

    def execute_eighth_contact_probe(self) -> None:
        """Lower each open gripper independently until effort contact is confirmed."""
        self._require_bound()
        if not self._is_eighth:
            raise ValueError("contact probe requires pipeline eight")
        if not self._completed or self._completed[-1] != "grasp":
            raise RuntimeError("contact probe must run immediately after grasp")
        if self._contact_probe_completed:
            raise RuntimeError("contact probe already completed")
        policy = self.contact_probe_policy
        if policy.get("enabled") is not True:
            raise RuntimeError("contact probe policy is disabled")

        preprobe_joints = {
            side: np.asarray(values, dtype=float)
            for side, values in self.controller.get_feedback_pair().items()
        }
        contacts: dict[str, dict[str, Any]] = {}
        try:
            for side in policy["probe_order"]:
                inactive = "right" if side == "left" else "left"
                start_pose = self._get_fresh_tool_pose(side)
                start_position = np.asarray(start_pose[0], dtype=float)
                quaternion = np.asarray(start_pose[1], dtype=float)
                grasp_target = np.asarray(
                    self.stages["grasp"]["waypoints"][-1][side][
                        "target_position_base_m"
                    ],
                    dtype=float,
                )
                x_error = float(start_position[0] - grasp_target[0])
                x_tolerance = float(
                    self.plan["safety"]["grasp_x_tolerance_m"]
                )
                maximum_inward = float(
                    self.plan["safety"][
                        "grasp_maximum_inward_x_error_m"
                    ]
                )
                measured_x_gate_passed = (
                    abs(x_error) <= x_tolerance
                    and x_error <= maximum_inward
                )
                saved_replay_bypass = (
                    self.saved_eighth_grasp_refinement is not None
                )
                x_gate_passed = measured_x_gate_passed or saved_replay_bypass
                self.trace.append(
                    {
                        "event": "eighth_contact_probe_pre_xy_gate",
                        "side": side,
                        "actual_x_m": float(start_position[0]),
                        "target_x_m": float(grasp_target[0]),
                        "x_error_m": x_error,
                        "x_tolerance_m": x_tolerance,
                        "maximum_inward_x_error_m": maximum_inward,
                        "measured_passed": measured_x_gate_passed,
                        "saved_replay_bypass": saved_replay_bypass,
                        "passed": x_gate_passed,
                    }
                )
                if not x_gate_passed:
                    raise RuntimeError(
                        f"{side} contact probe blocked by grasp X error: "
                        f"{x_error:.6f}m"
                    )
                probe_origin = start_position.copy()
                if policy.get("lock_xy_to_grasp_target") is True:
                    probe_origin[:2] = grasp_target[:2]
                baseline = self._sample_median_joint_telemetry(
                    side, sample_count=int(policy["sample_count"])
                )
                if baseline is None:
                    raise FeedbackStaleError(
                        f"{side} effort telemetry unavailable before contact probe"
                    )
                baseline_effort = baseline["effort"]
                self.trace.append(
                    {
                        "event": "eighth_contact_probe_baseline",
                        "side": side,
                        "effort_nm": baseline_effort.tolist(),
                        "median_abs_velocity_rad_s": (
                            baseline["median_abs_velocity"].tolist()
                        ),
                        "tcp_position_base_m": start_position.tolist(),
                    }
                )
                step = float(policy["step_m"])
                maximum = float(policy["maximum_descent_m"])
                count = int(round(maximum / step))
                contacted = None
                for index in range(1, count + 1):
                    descent = min(index * step, maximum)
                    target_position = probe_origin + np.asarray(
                        policy["axis_base"], dtype=float
                    ) * descent
                    current = self.controller.get_feedback_pair()
                    candidates = self.solver.solve_candidates(
                        side=side,
                        target_position_base_m=target_position,
                        target_quaternion_xyzw=quaternion,
                        reference_joints_rad=current[side],
                        probes=int(policy["ik_probes_per_step"]),
                    )
                    ranked = []
                    for candidate in candidates:
                        joints = np.asarray(candidate, dtype=float)
                        pose = self.local_fk[side].fk(joints, self.torso)
                        fk_error = float(
                            np.linalg.norm(
                                pose.position_base_m - target_position
                            )
                        )
                        joint_delta = float(
                            np.max(np.abs(joints - np.asarray(current[side])))
                        )
                        if (
                            fk_error
                            <= float(policy["maximum_fk_position_error_m"])
                            and joint_delta
                            <= float(
                                policy["maximum_incremental_joint_delta_rad"]
                            )
                        ):
                            ranked.append(
                                (fk_error, joint_delta, joints, pose)
                            )
                    if not ranked:
                        raise RuntimeError(
                            f"{side} contact probe has no bounded IK candidate "
                            f"at {descent * 1000.0:.1f}mm"
                        )
                    fk_error, joint_delta, selected, pose = min(
                        ranked, key=lambda item: item[:2]
                    )
                    self.controller.execute_single_arm_motion(
                        side=side,
                        target=selected,
                        inactive_hold=current[inactive],
                        progress_steps=1,
                        trace_context={
                            "stage_id": "contact_probe",
                            "phase": f"{side}_descent",
                            "waypoint_index": index,
                        },
                    )
                    indication_seen = False
                    stable_confirmations = 0
                    stationary_attempt = 0
                    while True:
                        telemetry = self._sample_median_joint_telemetry(
                            side, sample_count=int(policy["sample_count"])
                        )
                        if telemetry is None:
                            raise FeedbackStaleError(
                                f"{side} effort telemetry became unavailable"
                            )
                        residual = telemetry["effort"] - baseline_effort
                        indices = [
                            int(value)
                            for value in policy["joint_indices_zero_based"]
                        ]
                        score = float(np.sum(np.abs(residual[indices])))
                        maximum_residual = float(np.max(np.abs(residual)))
                        maximum_velocity = float(
                            np.max(telemetry["median_abs_velocity"])
                        )
                        if maximum_residual >= float(
                            policy["single_joint_emergency_threshold_nm"]
                        ):
                            raise RuntimeError(
                                f"{side} contact probe emergency effort residual: "
                                f"{maximum_residual:.3f}Nm"
                            )
                        raw_effort_indication = score >= float(
                            policy["sum_abs_residual_threshold_nm"]
                        )
                        velocity_stable = maximum_velocity <= float(
                            policy["maximum_median_abs_velocity_rad_s"]
                        )
                        actual_pose = self._get_fresh_tool_pose(side)
                        contact_indication = raw_effort_indication
                        detected = contact_indication and velocity_stable
                        evidence = {
                            "event": "eighth_contact_probe_sample",
                            "side": side,
                            "descent_m": descent,
                            "step_index": index,
                            "confirmation_index": stable_confirmations + 1,
                            "stationary_attempt": stationary_attempt,
                            "target_tcp_position_base_m": (
                                target_position.tolist()
                            ),
                            "fk_tcp_position_base_m": (
                                pose.position_base_m.tolist()
                            ),
                            "actual_tcp_position_base_m": list(
                                actual_pose[0]
                            ),
                            "actual_quaternion_xyzw": list(actual_pose[1]),
                            "fk_position_error_m": fk_error,
                            "maximum_joint_delta_rad": joint_delta,
                            "effort_nm": telemetry["effort"].tolist(),
                            "effort_reference_nm": baseline_effort.tolist(),
                            "effort_residual_reference": "probe_start_fixed",
                            "effort_residual_nm": residual.tolist(),
                            "joint_residual_score_nm": score,
                            "raw_effort_indication": raw_effort_indication,
                            "maximum_abs_effort_residual_nm": (
                                maximum_residual
                            ),
                            "median_max_abs_velocity_rad_s": (
                                maximum_velocity
                            ),
                            "contact_indication": contact_indication,
                            "velocity_stable": velocity_stable,
                            "descent_frozen": (
                                indication_seen or contact_indication
                            ),
                            "detected": detected,
                        }
                        self.trace.append(evidence)
                        if contact_indication and not indication_seen:
                            indication_seen = True
                            self.controller.hold()
                            self.trace.append(
                                {
                                    "event": (
                                        "eighth_contact_probe_descent_frozen"
                                    ),
                                    "side": side,
                                    "descent_m": descent,
                                    "reason": (
                                        "first_valid_effort_contact_indication"
                                    ),
                                    "joint_residual_score_nm": score,
                                    "median_max_abs_velocity_rad_s": (
                                        maximum_velocity
                                    ),
                                }
                            )
                        if not indication_seen:
                            break
                        stationary_attempt += 1
                        if detected:
                            stable_confirmations += 1
                        else:
                            stable_confirmations = 0
                        if stable_confirmations >= int(
                            policy["confirmation_samples"]
                        ):
                            contacted = {
                                "side": side,
                                "descent_m": descent,
                                "joints_rad": selected.tolist(),
                            }
                            contacts[side] = contacted
                            self.trace.append(
                                {
                                    "event": "eighth_contact_probe_detected",
                                    **contacted,
                                    "stationary_attempts": stationary_attempt,
                                }
                            )
                            break
                        if stationary_attempt >= int(
                            policy[
                                "stationary_confirmation_maximum_attempts"
                            ]
                        ):
                            raise RuntimeError(
                                f"{side} contact indication did not stabilize "
                                f"at {descent * 1000.0:.1f}mm; "
                                "descent remained frozen"
                            )
                    if contacted is not None:
                        break
                if contacted is None:
                    raise RuntimeError(
                        f"{side} contact not detected within "
                        f"{maximum * 1000.0:.1f}mm"
                    )
            if set(contacts) != {"left", "right"}:
                raise RuntimeError("both-arm contact confirmation is required")
            self._contact_probe_completed = True
            self.trace.append(
                {
                    "event": "eighth_contact_probe_completed",
                    "contacts": contacts,
                }
            )
        except Exception:
            self.controller.hold()
            try:
                self.controller.execute_joint_motion(
                    preprobe_joints["left"],
                    preprobe_joints["right"],
                    progress_steps=self._eighth_joint_command_progress_steps,
                    trace_context={
                        "stage_id": "contact_probe",
                        "phase": "failure_retract",
                        "waypoint_index": 0,
                    },
                )
                self.trace.append(
                    {"event": "eighth_contact_probe_failure_retracted"}
                )
            except Exception as retract_error:
                self.trace.append(
                    {
                        "event": "eighth_contact_probe_retraction_failed",
                        "error": repr(retract_error),
                    }
                )
            raise

    def open_gripper(self, *, side: str) -> None:
        self._require_bound()
        self.controller.open_gripper(side)
        self._closed_gripper_sides.discard(side)
        self._post_close_tcp_z.pop(side, None)
        self.trace.append({"event": "gripper_opened", "side": side})

    def close_gripper(self, *, side: str) -> None:
        self._require_bound()
        self.controller.close_gripper(side)
        self._closed_gripper_sides.add(side)
        if getattr(self, "_post_close_z_policy", {}).get("enabled") is True:
            close_z = float(self._get_fresh_tool_pose(side)[0][2])
            self._post_close_tcp_z[side] = close_z
            self.trace.append(
                {
                    "event": "post_close_tcp_z_guard_armed",
                    "side": side,
                    "tcp_z_base_m": close_z,
                }
            )
        self.trace.append({"event": "gripper_closed", "side": side})
        dwell_s = float(
            self.plan.get("safety", {}).get("post_grasp_close_dwell_s", 0.0)
        )
        if (
            dwell_s > 0.0
            and bool(self._completed)
            and self._completed[-1] == "grasp"
        ):
            time.sleep(dwell_s)
            feedback_deadline = time.monotonic() + 0.5
            while True:
                self.controller.refresh_feedback(timeout_s=0.05)
                try:
                    self.controller.get_feedback_pair()
                    break
                except FeedbackStaleError:
                    if time.monotonic() >= feedback_deadline:
                        raise
            self.trace.append(
                {
                    "event": "post_grasp_close_dwell_completed",
                    "side": side,
                    "duration_s": dwell_s,
                }
            )

    def execute_single_arm_stage(self, *, side: str, stage_id: str) -> None:
        """Execute one certified arm while the other holds session initial joints."""
        self._require_bound()
        expected_index = len(self._completed)
        stage_order = [stage["id"] for stage in self.bundle["stages"]]
        if expected_index >= len(stage_order) or stage_id != stage_order[expected_index]:
            raise ValueError("single-arm stages must execute in immutable order")
        plan_stage = self.plan_stages[stage_id]
        if side not in ("left", "right") or plan_stage.get("active_side") != side:
            raise ValueError("single-arm stage side does not match approved plan")
        inactive = "right" if side == "left" else "left"
        initial = {
            arm: np.asarray(
                self.bundle["initial_state"][f"{arm}_joints_rad"],
                dtype=float,
            )
            for arm in ("left", "right")
        }
        stage = self.stages[stage_id]
        try:
            if stage_id == "home":
                current = self.controller.get_feedback_pair()
                self._validate_pair_segment(current, initial)
                self.controller.execute_joint_motion(
                    initial["left"],
                    initial["right"],
                    progress_steps=1,
                )
                self._record_actual_tcp(
                    side=side,
                    stage_id=stage_id,
                    phase="home",
                    sample_index=1,
                )
                arrived = self.controller.get_feedback_pair()
                home_error = max(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(arrived[arm], dtype=float)
                                - initial[arm]
                            )
                        )
                    )
                    for arm in ("left", "right")
                )
                if home_error > float(
                    self.plan["safety"]["home_joint_tolerance_rad"]
                ):
                    raise RuntimeError("session initial-state arrival gate failed")
                self._completed.append(stage_id)
                self.trace.append(
                    {
                        "event": "home_completed",
                        "stage_id": stage_id,
                        "maximum_joint_error_rad": home_error,
                    }
                )
                return

            active_waypoints = stage[side]["joint_waypoints_rad"]
            inactive_waypoints = stage[inactive]["joint_waypoints_rad"]
            overridden = self._post_grasp_joint_override.get(stage_id)
            if overridden is not None:
                if self._is_eighth or stage_id not in ("lift", "place_center"):
                    raise ValueError("post-grasp joint override stage is not allowed")
                active_waypoints = overridden
                inactive_waypoints = [
                    initial[inactive].tolist() for _ in active_waypoints
                ]
                self.trace.append(
                    {
                        "event": "pipeline7_post_grasp_joint_path_overridden",
                        "stage_id": stage_id,
                        "waypoint_count": len(active_waypoints),
                    }
                )
            if len(active_waypoints) != len(inactive_waypoints):
                raise ValueError("active/inactive waypoint counts differ")
            for waypoint_index, (active_target, inactive_target) in enumerate(
                zip(
                active_waypoints,
                inactive_waypoints,
                ),
                start=1,
            ):
                if not np.allclose(
                    np.asarray(inactive_target, dtype=float),
                    initial[inactive],
                    atol=1e-9,
                ):
                    raise ValueError("inactive arm waypoint changed from session initial")
                pair = {
                    side: np.asarray(active_target, dtype=float),
                    inactive: initial[inactive],
                }
                current = self.controller.get_feedback_pair()
                self._validate_pair_segment(current, pair, stage_id=stage_id)
                self._validate_post_close_tcp_z_segment(
                    side=side,
                    start_joints=current[side],
                    target_joints=pair[side],
                    stage_id=stage_id,
                    phase="certified_waypoint",
                    sample_index=waypoint_index,
                )
                self.controller.execute_single_arm_motion(
                    side=side,
                    target=pair[side],
                    inactive_hold=initial[inactive],
                    progress_steps=1,
                )
                self._record_actual_tcp(
                    side=side,
                    stage_id=stage_id,
                    phase="certified_waypoint",
                    sample_index=waypoint_index,
                )
                self._verify_post_close_tcp_z(
                    side=side,
                    stage_id=stage_id,
                    phase="certified_waypoint",
                    sample_index=waypoint_index,
                )

            if overridden is not None:
                error_components = self._cartesian_error_components_side(
                    stage_id, side
                )
                self._completed.append(stage_id)
                self.trace.append(
                    {
                        "event": "pipeline7_post_grasp_override_completed",
                        "stage_id": stage_id,
                        "side": side,
                        "waypoint_count": len(active_waypoints),
                        "cartesian_refinement_skipped": True,
                        "error_m": error_components["norm_m"],
                        "error_components_m": error_components,
                    }
                )
                return

            error_components = self._cartesian_error_components_side(
                stage_id, side
            )
            error = error_components["norm_m"]
            best_error = error
            stall_count = 0
            effort_reference = (
                self._sample_median_joint_telemetry(side)
                if stage_id == "grasp"
                and self.effort_contact_policy.get("enabled") is True
                and self.post_arrival_probe_policy.get("enabled") is not True
                else None
            )
            effort_contact = False
            for iteration in range(1, int(self.policy["max_iterations"]) + 1):
                if self._single_arm_arrived(stage_id, error_components):
                    break
                current = self.controller.get_feedback_pair()
                candidates = self.solver.solve_candidates(
                    side=side,
                    target_position_base_m=stage[side]["target_position_base_m"],
                    target_quaternion_xyzw=stage[side][
                        "target_quaternion_xyzw"
                    ],
                    reference_joints_rad=current[side],
                    probes=int(self.policy["ik_probes_per_iteration"]),
                )
                selected = self._select_single_arm_candidate(
                    side=side,
                    inactive=inactive,
                    stage_id=stage_id,
                    current=current,
                    inactive_hold=initial[inactive],
                    candidates=candidates,
                )
                self._validate_post_close_tcp_z_segment(
                    side=side,
                    start_joints=current[side],
                    target_joints=selected,
                    stage_id=stage_id,
                    phase="cartesian_refinement",
                    sample_index=iteration,
                )
                self.controller.execute_single_arm_motion(
                    side=side,
                    target=selected,
                    inactive_hold=initial[inactive],
                    progress_steps=1,
                )
                self._record_actual_tcp(
                    side=side,
                    stage_id=stage_id,
                    phase="cartesian_refinement",
                    sample_index=iteration,
                )
                self._verify_post_close_tcp_z(
                    side=side,
                    stage_id=stage_id,
                    phase="cartesian_refinement",
                    sample_index=iteration,
                )
                error_components = self._cartesian_error_components_side(
                    stage_id, side
                )
                error = error_components["norm_m"]
                if effort_reference is not None:
                    effort_contact, contact_evidence = (
                        self._grasp_effort_contact_detected(
                            side=side,
                            reference_effort=effort_reference["effort"],
                            error_components=error_components,
                        )
                    )
                    self.trace.append(
                        {
                            "event": "grasp_effort_contact_check",
                            "stage_id": stage_id,
                            "side": side,
                            "iteration": iteration,
                            **contact_evidence,
                        }
                    )
                improvement = best_error - error
                stall_count = (
                    stall_count + 1
                    if improvement < float(self.policy["minimum_improvement_m"])
                    else 0
                )
                self.trace.append(
                    {
                        "event": "single_arm_cartesian_refinement",
                        "stage_id": stage_id,
                        "side": side,
                        "iteration": iteration,
                        "error_m": error,
                        "error_components_m": error_components,
                        "improvement_m": improvement,
                        "stall_count": stall_count,
                    }
                )
                best_error = min(best_error, error)
                if effort_contact:
                    break
                if stage_id != "grasp" and stall_count >= int(
                    self.policy["local_optimum_stall_iterations"]
                ):
                    raise RuntimeError(
                        f"{stage_id} single-arm refinement reached local optimum"
                    )
            if not (
                self._single_arm_arrived(stage_id, error_components)
                or effort_contact
            ):
                raise RuntimeError(f"{stage_id} single-arm Cartesian arrival failed")
            post_arrival_probe = None
            if (
                stage_id == "grasp"
                and self.post_arrival_probe_policy.get("enabled") is True
            ):
                post_arrival_probe = self._execute_p7_post_arrival_grasp_probe(
                    side=side,
                    inactive=inactive,
                    inactive_hold=initial[inactive],
                )
            self._completed.append(stage_id)
            self.trace.append(
                {
                    "event": "single_arm_stage_completed",
                    "stage_id": stage_id,
                    "side": side,
                    "error_m": error,
                    "error_components_m": error_components,
                    "effort_contact_arrival": effort_contact,
                    "post_arrival_probe": post_arrival_probe,
                }
            )
        except Exception:
            self.controller.hold()
            raise

    def execute_bimanual_stage(self, *, stage_id: str) -> None:
        self._require_bound()
        expected_index = len(self._completed)
        stage_order = [stage["id"] for stage in self.bundle["stages"]]
        if expected_index >= len(stage_order) or stage_id != stage_order[expected_index]:
            raise ValueError("bimanual stages must execute in immutable order")
        stage = self.stages[stage_id]
        try:
            waypoint_count = len(stage["left"]["joint_waypoints_rad"])
            if len(stage["right"]["joint_waypoints_rad"]) != waypoint_count:
                raise ValueError("left/right stage waypoint counts differ")
            for waypoint_index in range(waypoint_count):
                waypoint_pair = {
                    side: stage[side]["joint_waypoints_rad"][waypoint_index]
                    for side in ("left", "right")
                }
                current = self.controller.get_feedback_pair()
                self._validate_pair_segment(current, waypoint_pair)
                self.controller.execute_joint_motion(
                    waypoint_pair["left"],
                    waypoint_pair["right"],
                )
            errors = self._cartesian_errors(stage_id)
            best_error = max(errors.values())
            stall_count = 0
            maximum_iterations = int(self.policy["max_iterations"])
            for iteration in range(1, maximum_iterations + 1):
                if best_error <= float(self.policy["target_tolerance_m"]):
                    break
                current = self.controller.get_feedback_pair()
                candidates = {
                    side: self.solver.solve_candidates(
                        side=side,
                        target_position_base_m=stage[side][
                            "target_position_base_m"
                        ],
                        target_quaternion_xyzw=stage[side][
                            "target_quaternion_xyzw"
                        ],
                        reference_joints_rad=current[side],
                        probes=int(self.policy["ik_probes_per_iteration"]),
                    )
                    for side in ("left", "right")
                }
                selected = self._select_pair(
                    stage_id=stage_id,
                    current=current,
                    candidates=candidates,
                )
                self.controller.execute_joint_motion(
                    selected["left"],
                    selected["right"],
                )
                errors = self._cartesian_errors(stage_id)
                current_error = max(errors.values())
                improvement = best_error - current_error
                stall_count = (
                    stall_count + 1
                    if improvement < float(self.policy["minimum_improvement_m"])
                    else 0
                )
                self.trace.append(
                    {
                        "event": "cartesian_refinement",
                        "stage_id": stage_id,
                        "iteration": iteration,
                        "errors_m": errors,
                        "improvement_m": improvement,
                        "stall_count": stall_count,
                    }
                )
                best_error = min(best_error, current_error)
                if current_error <= float(self.policy["target_tolerance_m"]):
                    break
                if stall_count >= int(
                    self.policy["local_optimum_stall_iterations"]
                ):
                    raise RuntimeError(
                        f"{stage_id} Cartesian refinement reached local optimum"
                    )
            else:
                raise RuntimeError(
                    f"{stage_id} exceeded Cartesian refinement iteration limit"
                )
            if max(errors.values()) > float(self.policy["target_tolerance_m"]):
                raise RuntimeError(f"{stage_id} Cartesian arrival gate failed")
            self._completed.append(stage_id)
            self.trace.append(
                {"event": "stage_completed", "stage_id": stage_id, "errors_m": errors}
            )
        except Exception:
            self.controller.hold()
            raise

    def execute_eighth_stage(self, *, stage_id: str) -> None:
        """Execute one certified pipeline-eight stage without online IK changes."""
        self._require_bound()
        try:
            if not self._is_eighth:
                raise ValueError("execute_eighth_stage requires a pipeline-eight bundle")
            stage_order = self.bundle.get("stage_order")
            if not isinstance(stage_order, list):
                raise ValueError("eighth bundle must declare stage_order")
            expected_index = len(self._completed)
            if (
                expected_index >= len(stage_order)
                or stage_id != stage_order[expected_index]
            ):
                raise ValueError("eighth stages must execute in bundle stage_order")

            stage = self.stages[stage_id]
            waypoints = stage.get("waypoints")
            if not isinstance(waypoints, list) or not waypoints:
                raise ValueError("eighth stage has no certified shared waypoints")
            if (
                stage_id == "release_retract"
                and self.plan_stages[stage_id].get("motion_policy")
                == "skip_after_release_direct_right_takeover"
            ):
                poses = self._get_fresh_tool_pose_pair()
                actual_right = np.asarray(poses["right"][0], dtype=float)
                nominal_right = np.asarray(
                    waypoints[-1]["right"]["target_position_base_m"], dtype=float
                )
                deviation = float(
                    np.linalg.norm(actual_right[:2] - nominal_right[:2])
                )
                maximum_deviation = float(
                    self.plan["safety"]["right_takeover_policy"][
                        "maximum_runtime_release_xy_deviation_m"
                    ]
                )
                if deviation > maximum_deviation:
                    raise RuntimeError(
                        "post-release right TCP XY exceeds certified runtime envelope"
                    )
                self._post_release_right_tcp = actual_right.copy()
                self.trace.append(
                    {
                        "event": "post_release_right_reference_captured",
                        "actual_right_tcp_base_m": actual_right.tolist(),
                        "nominal_right_tcp_base_m": nominal_right.tolist(),
                        "xy_deviation_m": deviation,
                        "maximum_xy_deviation_m": maximum_deviation,
                    }
                )
                self._completed.append(stage_id)
                self.trace.append(
                    {
                        "event": "eighth_stage_completed",
                        "stage_id": stage_id,
                        "waypoint_count": 0,
                        "motion_skipped": True,
                        "next_stage": "left_home_right_hold",
                    }
                )
                return
            if stage_id == "home":
                for side in ("left", "right"):
                    final_joints = np.asarray(
                        waypoints[-1][side]["joints_rad"], dtype=float
                    )
                    initial_joints = np.asarray(
                        self.bundle["initial_state"][f"{side}_joints_rad"],
                        dtype=float,
                    )
                    if not np.allclose(
                        final_joints, initial_joints, atol=1e-9, rtol=0.0
                    ):
                        raise ValueError(
                            "eighth home waypoint does not restore bundle initial"
                        )
            if stage_id in {
                "right_regrasp",
                "right_lift",
                "right_translate_center_x",
            }:
                if self._post_release_right_tcp is None:
                    raise RuntimeError(
                        "right takeover requires captured post-release TCP"
                    )
                if not self._eighth_right_waypoint_gate(
                    stage_id=stage_id,
                    waypoint_index=0,
                    waypoint=waypoints[-1],
                ):
                    self._eighth_refine_right_takeover_stage(
                        stage_id=stage_id,
                        waypoint=waypoints[-1],
                    )
                self._completed.append(stage_id)
                self.trace.append(
                    {
                        "event": "eighth_stage_completed",
                        "stage_id": stage_id,
                        "waypoint_count": 0,
                        "runtime_bound_target": True,
                    }
                )
                return
            for waypoint_index, waypoint in enumerate(waypoints, start=1):
                if stage_id == "pullback" and waypoint_index == 1:
                    evidence = (
                        self.controller.close_both_grippers_stable_and_hold()
                    )
                    self.trace.append(
                        {
                            "event": (
                                "post_lift_grippers_reclosed_before_pullback"
                            ),
                            "stage_id": stage_id,
                            "waypoint_index": waypoint_index,
                            "wait_duration_s": 2.0,
                            **evidence,
                        }
                    )
                target = {
                    side: np.asarray(waypoint[side]["joints_rad"], dtype=float)
                    for side in ("left", "right")
                }
                current = self.controller.get_feedback_pair()
                self._validate_pair_segment(
                    current, target, stage_id=stage_id
                )
                self.controller.execute_joint_motion(
                    target["left"],
                    target["right"],
                    progress_steps=self._eighth_joint_command_progress_steps,
                    trace_context={
                        "stage_id": stage_id,
                        "phase": "eighth_certified_waypoint_progress",
                        "waypoint_index": waypoint_index,
                    },
                )
                if waypoint.get("ik_acceptance") == "joint_transition":
                    self.trace.append(
                        {
                            "event": "eighth_grasp_seed_joint_transition",
                            "stage_id": stage_id,
                            "waypoint_index": waypoint_index,
                            "cartesian_arrival_gate_checked": False,
                        }
                    )
                    continue
                right_only = stage_id in {
                    "right_regrasp",
                    "right_lift",
                    "right_translate_center_x",
                }
                passed = (
                    self._eighth_right_waypoint_gate(
                        stage_id=stage_id,
                        waypoint_index=waypoint_index,
                        waypoint=waypoint,
                    )
                    if right_only
                    else self._eighth_waypoint_geometry_gate(
                        stage_id=stage_id,
                        waypoint_index=waypoint_index,
                        waypoint=waypoint,
                        raise_on_failure=False,
                        enforce_grasp_reference=(
                            stage_id != "grasp"
                            or waypoint_index == len(waypoints)
                        ),
                    )
                )
                if waypoint_index == len(waypoints) and not passed:
                    if right_only:
                        self._eighth_refine_right_takeover_stage(
                            stage_id=stage_id, waypoint=waypoint
                        )
                    else:
                        self._eighth_refine_stage(
                            stage_id=stage_id, waypoint=waypoint
                        )

            if stage_id == "left_home_right_hold":
                self._eighth_left_home_arrival_gate()
            if stage_id == "home":
                self._eighth_home_arrival_gate()
            self._completed.append(stage_id)
            self.trace.append(
                {
                    "event": "eighth_stage_completed",
                    "stage_id": stage_id,
                    "waypoint_count": len(waypoints),
                }
            )
        except Exception:
            self.controller.hold()
            raise

    def _eighth_left_home_arrival_gate(self) -> None:
        actual = self.controller.get_feedback_pair()
        target = np.asarray(
            self.bundle["initial_state"]["left_joints_rad"], dtype=float
        )
        error = float(
            np.max(np.abs(np.asarray(actual["left"], dtype=float) - target))
        )
        passed = error <= 0.02
        self.trace.append(
            {
                "event": "left_home_right_hold_joint_gate",
                "maximum_left_joint_error_rad": error,
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError("left arm did not reach session initial joints")

    def _eighth_right_waypoint_gate(
        self,
        *,
        stage_id: str,
        waypoint_index: int,
        waypoint: dict[str, Any],
    ) -> bool:
        poses = self._get_fresh_tool_pose_pair()
        corrected, correction = self._offset_aware_tool_positions(poses)
        actual = corrected["right"]
        target = self._eighth_runtime_right_target(
            stage_id=stage_id, waypoint=waypoint
        )
        error = float(np.linalg.norm(actual - target))
        tolerance = min(
            0.015,
            float(self.bundle["motion_limits"]["fk_position_tolerance_m"]),
        )
        passed = error <= tolerance
        self.trace.append(
            {
                "event": "eighth_right_takeover_geometry_gate",
                "stage_id": stage_id,
                "waypoint_index": waypoint_index,
                "actual_right_tcp_base_m": actual.tolist(),
                "raw_actual_right_tcp_base_m": [
                    float(value) for value in poses["right"][0]
                ],
                "right_control_offset_tcp_correction_m": correction.tolist(),
                "target_right_tcp_base_m": target.tolist(),
                "right_tcp_error_m": error,
                "tolerance_m": tolerance,
                "passed": passed,
            }
        )
        return passed

    def _eighth_runtime_right_target(
        self, *, stage_id: str, waypoint: dict[str, Any]
    ) -> np.ndarray:
        if self._post_release_right_tcp is None:
            raise RuntimeError("post-release right TCP reference is unavailable")
        nominal = np.asarray(
            waypoint["right"]["target_position_base_m"], dtype=float
        )
        target = nominal.copy()
        if stage_id in {"right_regrasp", "right_lift"}:
            target[:2] = self._post_release_right_tcp[:2]
        elif stage_id == "right_translate_center_x":
            target[1] = self._post_release_right_tcp[1]
        else:
            raise ValueError("unexpected right takeover stage")
        return target

    def _eighth_refine_right_takeover_stage(
        self, *, stage_id: str, waypoint: dict[str, Any]
    ) -> None:
        policy = self.plan["safety"]["right_takeover_policy"]
        iterations = int(policy["maximum_refinement_iterations"])
        target = self._eighth_runtime_right_target(
            stage_id=stage_id, waypoint=waypoint
        )
        quaternion = waypoint["right"]["target_quaternion_xyzw"]
        initial_left = np.asarray(
            self.bundle["initial_state"]["left_joints_rad"], dtype=float
        )
        for iteration in range(1, iterations + 1):
            current = self.controller.get_feedback_pair()
            candidates = self.solver.solve_candidates(
                side="right",
                target_position_base_m=target,
                target_quaternion_xyzw=quaternion,
                reference_joints_rad=current["right"],
                probes=8,
            )
            selected = self._select_eighth_individual_candidate(
                current=current,
                active_side="right",
                candidates=candidates,
                waypoint=waypoint,
                stage_id=stage_id,
                inactive_hold=initial_left,
                target_position=target,
            )
            self.controller.execute_single_arm_motion(
                side="right",
                target=selected["right"],
                inactive_hold=initial_left,
                progress_steps=self._eighth_joint_command_progress_steps,
            )
            if self._eighth_right_waypoint_gate(
                stage_id=stage_id,
                waypoint_index=1000 + iteration,
                waypoint=waypoint,
            ):
                self.trace.append(
                    {
                        "event": "eighth_right_takeover_refinement_completed",
                        "stage_id": stage_id,
                        "iterations": iteration,
                    }
                )
                return
        raise RuntimeError(
            f"{stage_id} exceeded right-arm Cartesian refinement limit"
        )

    def _eighth_cartesian_targets(
        self, *, stage_id: str, waypoint: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        if stage_id != "translate_center":
            return {
                side: np.asarray(
                    waypoint[side]["target_position_base_m"], dtype=float
                )
                for side in ("left", "right")
            }
        control = self.plan["safety"]["translation_control_policy"]
        if control["mode"] != "midpoint_plus_fixed_grasp_relative_vector":
            raise ValueError("unsupported eighth translation control mode")
        midpoint = np.asarray(
            waypoint["expected_midpoint_base_m"], dtype=float
        )
        relative = np.asarray(
            self.plan["grasp_geometry"][
                "initial_full_relative_vector_right_minus_left_m"
            ],
            dtype=float,
        )
        targets = {
            "left": midpoint - relative / 2.0,
            "right": midpoint + relative / 2.0,
        }
        self.trace.append(
            {
                "event": "eighth_translation_midpoint_targets",
                "midpoint_target_base_m": midpoint.tolist(),
                "fixed_right_minus_left_m": relative.tolist(),
                "left_target_base_m": targets["left"].tolist(),
                "right_target_base_m": targets["right"].tolist(),
            }
        )
        return targets

    def _eighth_waypoint_geometry_gate(
        self,
        *,
        stage_id: str,
        waypoint_index: int,
        waypoint: dict[str, Any],
        raise_on_failure: bool = True,
        enforce_grasp_reference: bool = True,
    ) -> bool:
        wait = getattr(self.controller, "wait_until_ready", None)
        if wait is not None:
            wait(timeout_s=1.0, require_tool_pose=True)
        poses = self._get_fresh_tool_pose_pair()
        actual, right_correction = self._offset_aware_tool_positions(poses)
        for side in ("left", "right"):
            self.trace.append(
                {
                    "event": "actual_tcp_sample",
                    "side": side,
                    "stage_id": stage_id,
                    "phase": "eighth_certified_waypoint",
                    "sample_index": waypoint_index,
                    "position_base_m": actual[side].tolist(),
                    "quaternion_xyzw": [
                        float(value) for value in poses[side][1]
                    ],
                }
            )

        expected_targets = self._eighth_cartesian_targets(
            stage_id=stage_id, waypoint=waypoint
        )
        target_errors = {
            side: float(np.linalg.norm(actual[side] - expected_targets[side]))
            for side in ("left", "right")
        }
        grasp_x_errors = {
            side: float(actual[side][0] - expected_targets[side][0])
            for side in ("left", "right")
        }
        actual_midpoint = (actual["left"] + actual["right"]) / 2.0
        expected_midpoint = np.asarray(
            waypoint["expected_midpoint_base_m"], dtype=float
        )
        midpoint_error = float(
            np.linalg.norm(actual_midpoint - expected_midpoint)
        )
        actual_relative = actual["right"] - actual["left"]
        expected_relative = np.asarray(
            waypoint["expected_right_minus_left_m"], dtype=float
        )
        expected_relative_error = float(
            np.linalg.norm(actual_relative - expected_relative)
        )
        reference_relative_error: float | None = None
        if stage_id in {"lift", "translate_center"} or (
            stage_id == "grasp" and enforce_grasp_reference
        ):
            reference = np.asarray(
                self.plan["grasp_geometry"][
                    "initial_full_relative_vector_right_minus_left_m"
                ],
                dtype=float,
            )
            reference_relative_error = float(
                np.linalg.norm(actual_relative - reference)
            )

        limits = self.bundle["motion_limits"]
        position_tolerance = min(
            0.015, float(limits["fk_position_tolerance_m"])
        )
        relative_tolerance = float(limits["relative_geometry_tolerance_m"])
        translate_height_error: float | None = None
        if stage_id == "translate_center":
            translate_height_error = float(
                abs(actual_midpoint[2] - expected_midpoint[2])
            )
        pair_error_gates_enforced = bool(
            self.plan["safety"]["grasp_refinement_policy"].get(
                "final_relative_geometry_gate_required", True
            )
        )
        grasp_x_gate_passed = True
        if stage_id == "grasp":
            x_tolerance = float(
                self.plan["safety"]["grasp_x_tolerance_m"]
            )
            maximum_inward = float(
                self.plan["safety"]["grasp_maximum_inward_x_error_m"]
            )
            grasp_x_gate_passed = all(
                abs(error) <= x_tolerance and error <= maximum_inward
                for error in grasp_x_errors.values()
            )
        passed = all(
            error <= position_tolerance for error in target_errors.values()
        ) and grasp_x_gate_passed and (
            not pair_error_gates_enforced
            or (
                midpoint_error <= position_tolerance
                and expected_relative_error <= relative_tolerance
                and (
                    reference_relative_error is None
                    or reference_relative_error <= relative_tolerance
                )
                and (
                    translate_height_error is None
                    or translate_height_error <= position_tolerance
                )
            )
        )
        self.trace.append(
            {
                "event": "geometry_gate",
                "stage_id": stage_id,
                "waypoint_index": waypoint_index,
                "passed": passed,
                "actual_tcp": {
                    side: actual[side].tolist() for side in ("left", "right")
                },
                "raw_actual_tcp": {
                    side: [float(value) for value in poses[side][0]]
                    for side in ("left", "right")
                },
                "right_control_offset_tcp_correction_m": (
                    right_correction.tolist()
                ),
                "target_error_m": target_errors,
                "grasp_x_error_m": grasp_x_errors,
                "grasp_x_gate_passed": grasp_x_gate_passed,
                "position_tolerance_m": position_tolerance,
                "actual_midpoint_base_m": actual_midpoint.tolist(),
                "expected_midpoint_base_m": expected_midpoint.tolist(),
                "midpoint_error_m": midpoint_error,
                "actual_right_minus_left_m": actual_relative.tolist(),
                "expected_right_minus_left_m": expected_relative.tolist(),
                "expected_relative_vector_error_m": expected_relative_error,
                "reference_relative_vector_error_m": reference_relative_error,
                "relative_tolerance_m": relative_tolerance,
                "pair_error_gates_enforced": pair_error_gates_enforced,
                "translate_height_error_m": translate_height_error,
            }
        )
        if not passed and raise_on_failure:
            raise RuntimeError(
                f"{stage_id} waypoint {waypoint_index} geometry gate failed"
            )
        return passed

    def _replay_saved_eighth_grasp_refinement(
        self, *, waypoint: dict[str, Any]
    ) -> None:
        artifact = self.saved_eighth_grasp_refinement
        if artifact is None:
            raise RuntimeError("saved eighth grasp refinement is unavailable")
        steps = artifact.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("saved eighth grasp refinement steps are invalid")
        for index, step in enumerate(steps, start=1):
            left = np.asarray(step["left_joints_rad"], dtype=float)
            right = np.asarray(step["right_joints_rad"], dtype=float)
            if (
                left.shape != (6,)
                or right.shape != (6,)
                or not np.isfinite(left).all()
                or not np.isfinite(right).all()
            ):
                raise ValueError("saved eighth grasp refinement joints are invalid")
            self.controller.execute_joint_motion(
                left,
                right,
                progress_steps=5,
                trace_context={
                    "stage_id": "grasp",
                    "phase": "saved_eighth_grasp_refinement_replay",
                    "waypoint_index": 2000 + index,
                },
            )
            self._record_actual_tcp(
                side="left",
                stage_id="grasp",
                phase="saved_eighth_grasp_refinement_replay",
                sample_index=index,
            )
            self._record_actual_tcp(
                side="right",
                stage_id="grasp",
                phase="saved_eighth_grasp_refinement_replay",
                sample_index=index,
            )
            self.trace.append(
                {
                    "event": "saved_eighth_grasp_refinement_replayed",
                    "step_index": index,
                    "source_iteration": step.get("source_iteration"),
                    "left_joints_rad": left.tolist(),
                    "right_joints_rad": right.tolist(),
                }
            )
        self._eighth_waypoint_geometry_gate(
            stage_id="grasp",
            waypoint_index=2999,
            waypoint=waypoint,
            raise_on_failure=False,
        )
        self.trace.append(
            {
                "event": "saved_eighth_grasp_refinement_completed",
                "step_count": len(steps),
                "gripper_close_authorized": True,
                "online_refinement_enabled": False,
            }
        )

    def _eighth_refine_stage(
        self, *, stage_id: str, waypoint: dict[str, Any]
    ) -> None:
        if stage_id == "grasp":
            if self.saved_eighth_grasp_refinement is not None:
                self._replay_saved_eighth_grasp_refinement(waypoint=waypoint)
                return
            self._eighth_refine_open_grasp_individually(waypoint=waypoint)
            return
        policy = self.plan["safety"]["grasp_refinement_policy"]
        tcp_gate_required = bool(
            policy.get("post_refinement_tcp_error_gate_required", True)
        )
        relative_gate_required = bool(
            policy.get("final_relative_geometry_gate_required", True)
        )
        cartesian_targets = self._eighth_cartesian_targets(
            stage_id=stage_id, waypoint=waypoint
        )
        for iteration in range(1, int(policy["maximum_iterations"]) + 1):
            current = self.controller.get_feedback_pair()
            candidates = {
                side: self.solver.solve_candidates(
                    side=side,
                    target_position_base_m=cartesian_targets[side],
                    target_quaternion_xyzw=waypoint[side][
                        "target_quaternion_xyzw"
                    ],
                    reference_joints_rad=current[side],
                    probes=5,
                )
                for side in ("left", "right")
            }
            selected = self._select_eighth_refinement_pair(
                stage_id=stage_id,
                current=current,
                candidates=candidates,
                waypoint=waypoint,
            )
            self.controller.execute_joint_motion(
                selected["left"],
                selected["right"],
                progress_steps=self._eighth_joint_command_progress_steps,
                trace_context={
                    "stage_id": stage_id,
                    "phase": "eighth_refinement_progress",
                    "waypoint_index": 1000 + iteration,
                },
            )
            passed = self._eighth_waypoint_geometry_gate(
                stage_id=stage_id,
                waypoint_index=1000 + iteration,
                waypoint=waypoint,
                raise_on_failure=False,
            )
            self.trace.append(
                {
                    "event": "eighth_grasp_refinement",
                    "stage": stage_id,
                    "iteration": iteration,
                    "passed": passed,
                }
            )
            if passed:
                return
            if not tcp_gate_required and not relative_gate_required:
                self.trace.append(
                    {
                        "event": "eighth_refinement_force_continued",
                        "stage_id": stage_id,
                        "iterations": iteration,
                        "tcp_and_relative_error_gates_enforced": False,
                        "reason": "non_enforced_geometry_gates_single_attempt",
                    }
                )
                return
        self.trace.append(
            {
                "event": "eighth_refinement_force_continued",
                "stage_id": stage_id,
                "iterations": int(policy["maximum_iterations"]),
                "tcp_and_relative_error_gates_enforced": False,
            }
        )
        return

    def _eighth_refine_open_grasp_individually(
        self, *, waypoint: dict[str, Any]
    ) -> None:
        policy = self.plan["safety"]["grasp_refinement_policy"]
        tolerance = float(policy["individual_tcp_tolerance_m"])
        targets = {
            side: np.asarray(
                waypoint[side]["target_position_base_m"], dtype=float
            )
            for side in ("left", "right")
        }
        for iteration in range(1, int(policy["maximum_iterations"]) + 1):
            poses = self._get_fresh_tool_pose_pair()
            errors = {
                side: float(
                    np.linalg.norm(
                        np.asarray(poses[side][0], dtype=float)
                        - targets[side]
                    )
                )
                for side in ("left", "right")
            }
            current = self.controller.get_feedback_pair()
            if any(error > tolerance for error in errors.values()):
                active_order = sorted(
                    (
                        side
                        for side in ("left", "right")
                        if errors[side] > tolerance
                    ),
                    key=lambda side: errors[side],
                    reverse=True,
                )
            else:
                active_order = sorted(
                    ("left", "right"),
                    key=lambda side: errors[side],
                    reverse=True,
                )
            selected = None
            active_side = None
            rejection_reasons = []
            for side in active_order:
                candidates = self.solver.solve_candidates(
                    side=side,
                    target_position_base_m=waypoint[side][
                        "target_position_base_m"
                    ],
                    target_quaternion_xyzw=waypoint[side][
                        "target_quaternion_xyzw"
                    ],
                    reference_joints_rad=current[side],
                    probes=5,
                )
                try:
                    selected = self._select_eighth_individual_candidate(
                        current=current,
                        active_side=side,
                        candidates=candidates,
                        waypoint=waypoint,
                    )
                    active_side = side
                    break
                except RuntimeError as exc:
                    rejection_reasons.append(f"{side}: {exc}")
            if selected is None or active_side is None:
                raise RuntimeError(
                    "no independent open-grasp refinement candidate passed "
                    "joint and collision gates; "
                    + "; ".join(rejection_reasons)
                )
            inactive_side = "right" if active_side == "left" else "left"
            self.controller.execute_joint_motion(
                selected["left"],
                selected["right"],
                progress_steps=self._eighth_joint_command_progress_steps,
                trace_context={
                    "stage_id": "grasp",
                    "phase": "eighth_independent_grasp_refinement_progress",
                    "waypoint_index": 1000 + iteration,
                    "active_side": active_side,
                },
            )
            passed = self._eighth_waypoint_geometry_gate(
                stage_id="grasp",
                waypoint_index=1000 + iteration,
                waypoint=waypoint,
                raise_on_failure=False,
            )
            self.trace.append(
                {
                    "event": "eighth_independent_grasp_refinement",
                    "iteration": iteration,
                    "active_side": active_side,
                    "inactive_side": inactive_side,
                    "pre_move_target_errors_m": errors,
                    "passed": passed,
                }
            )
            if passed:
                return
        poses = self._get_fresh_tool_pose_pair()
        x_tolerance = float(self.plan["safety"]["grasp_x_tolerance_m"])
        maximum_inward = float(
            self.plan["safety"]["grasp_maximum_inward_x_error_m"]
        )
        final_x_errors = {
            side: float(
                np.asarray(poses[side][0], dtype=float)[0] - targets[side][0]
            )
            for side in ("left", "right")
        }
        x_gate_passed = all(
            abs(error) <= x_tolerance and error <= maximum_inward
            for error in final_x_errors.values()
        )
        self.trace.append(
            {
                "event": "eighth_refinement_force_continued",
                "stage_id": "grasp",
                "iterations": int(policy["maximum_iterations"]),
                "tcp_and_relative_error_gates_enforced": False,
                "grasp_x_error_m": final_x_errors,
                "grasp_x_gate_passed": x_gate_passed,
                "gripper_close_authorized": x_gate_passed,
            }
        )
        if not x_gate_passed:
            raise RuntimeError(
                "grasp refinement exhausted before the configured X gate passed"
            )
        return

    def _select_eighth_individual_candidate(
        self,
        *,
        current: dict[str, Sequence[float]],
        active_side: str,
        candidates: list[Sequence[float]],
        waypoint: dict[str, Any],
        stage_id: str = "grasp",
        inactive_hold: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        joint_limit = min(
            1.0,
            float(
                self.bundle["motion_limits"][
                    "maximum_waypoint_joint_delta_rad"
                ]
            ),
        )
        target = (
            np.asarray(waypoint[active_side]["target_position_base_m"], dtype=float)
            if target_position is None
            else np.asarray(target_position, dtype=float)
        )
        valid = []
        inactive_side = "right" if active_side == "left" else "left"
        for value in candidates:
            active = np.asarray(value, dtype=float)
            if (
                np.max(
                    np.abs(
                        active
                        - np.asarray(current[active_side], dtype=float)
                    )
                )
                > joint_limit
            ):
                continue
            pair = {
                active_side: active,
                inactive_side: (
                    np.asarray(current[inactive_side], dtype=float)
                    if inactive_hold is None
                    else np.asarray(inactive_hold, dtype=float)
                ),
            }
            try:
                if stage_id == "right_regrasp":
                    assert self.right_takeover_safety is not None
                    rotation = Rotation.from_quat(
                        waypoint["right"]["target_quaternion_xyzw"]
                    ).as_matrix()
                    offsets = (
                        R1LiteTableCollisionKinematics
                        .finger_minimum_z_offsets_from_tcp(
                            rotation, fingers_open=True
                        )
                    )
                    contact_link = ("finger_link1", "finger_link2")[
                        int(np.argmin(offsets))
                    ]
                    self.right_takeover_safety.validate_interpolated_segment(
                        np.asarray(current["left"], dtype=float),
                        np.asarray(current["right"], dtype=float),
                        pair["left"],
                        pair["right"],
                        self.torso,
                        samples=50,
                        fingers_open=True,
                        allowed_contact_finger_link=contact_link,
                        minimum_noncontact_clearance_m=0.001,
                        minimum_contact_clearance_m=-0.0002,
                    )
                else:
                    self._validate_pair_segment(
                        current,
                        pair,
                        fingers_open=stage_id != "right_lift",
                        stage_id=stage_id,
                    )
            except ValueError:
                continue
            fk_position = self.local_fk[active_side].fk(
                active, self.torso
            ).position_base_m
            valid.append(
                (
                    float(np.linalg.norm(fk_position - target)),
                    float(
                        np.max(
                            np.abs(
                                active
                                - np.asarray(
                                    current[active_side], dtype=float
                                )
                            )
                        )
                    ),
                    pair,
                )
            )
        if not valid:
            raise RuntimeError("no candidate passed joint and collision gates")
        valid.sort(key=lambda item: item[:2])
        return valid[0][2]

    def _eighth_refine_grasp(self, *, waypoint: dict[str, Any]) -> None:
        self._eighth_refine_stage(stage_id="grasp", waypoint=waypoint)

    def _select_eighth_refinement_pair(
        self,
        *,
        stage_id: str,
        current: dict[str, Sequence[float]],
        candidates: dict[str, list[Sequence[float]]],
        waypoint: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        joint_limit = float(
            self.bundle["motion_limits"]["maximum_waypoint_joint_delta_rad"]
        )
        joint_limit = min(1.0, joint_limit)
        expected_midpoint = np.asarray(
            waypoint["expected_midpoint_base_m"], dtype=float
        )
        expected_relative = np.asarray(
            self.plan["grasp_geometry"][
                "initial_full_relative_vector_right_minus_left_m"
            ]
            if stage_id == "translate_center"
            else waypoint["expected_right_minus_left_m"],
            dtype=float,
        )
        target = self._eighth_cartesian_targets(
            stage_id=stage_id, waypoint=waypoint
        )
        valid = []
        for left_value, right_value in product(
            candidates["left"], candidates["right"]
        ):
            pair = {
                "left": np.asarray(left_value, dtype=float),
                "right": np.asarray(right_value, dtype=float),
            }
            if any(
                np.max(
                    np.abs(
                        pair[side]
                        - np.asarray(current[side], dtype=float)
                    )
                )
                > joint_limit
                for side in ("left", "right")
            ):
                continue
            try:
                self._validate_pair_segment(
                    current, pair, fingers_open=False
                )
            except ValueError:
                continue
            fk_positions = {
                side: self.local_fk[side]
                .fk(pair[side], self.torso)
                .position_base_m
                for side in ("left", "right")
            }
            relative_error = float(
                np.linalg.norm(
                    (fk_positions["right"] - fk_positions["left"])
                    - expected_relative
                )
            )
            midpoint_error = float(
                np.linalg.norm(
                    (fk_positions["left"] + fk_positions["right"]) / 2.0
                    - expected_midpoint
                )
            )
            side_errors = {
                side: float(np.linalg.norm(fk_positions[side] - target[side]))
                for side in ("left", "right")
            }
            score = (
                (
                    midpoint_error,
                    max(side_errors.values()),
                    sum(side_errors.values()),
                )
                if stage_id == "translate_center"
                else (
                    max(side_errors.values()),
                    midpoint_error,
                    sum(side_errors.values()),
                )
            )
            valid.append((*score, pair))
        if not valid:
            raise RuntimeError(
                "no synchronized grasp refinement pair passed "
                "joint, collision, and relative-geometry gates"
            )
        valid.sort(key=lambda item: item[:3])
        return valid[0][3]

    def _eighth_home_arrival_gate(self) -> None:
        arrived = self.controller.get_feedback_pair()
        logical_arrived = self._logical_feedback_pair(arrived)
        initial = {
            side: np.asarray(
                self.bundle["initial_state"][f"{side}_joints_rad"], dtype=float
            )
            for side in ("left", "right")
        }
        errors = {
            side: float(
                np.max(
                    np.abs(logical_arrived[side] - initial[side])
                )
            )
            for side in ("left", "right")
        }
        tolerance = max(
            float(self.plan["safety"].get("home_joint_tolerance_rad", 0.01)),
            float(
                getattr(
                    getattr(self.controller, "config", None),
                    "arrival_soft_threshold_rad",
                    0.0,
                )
            ),
        )
        passed = all(error <= tolerance for error in errors.values())
        self.trace.append(
            {
                "event": "home_joint_gate",
                "stage_id": "home",
                "passed": passed,
                "joint_errors_rad": errors,
                "tolerance_rad": tolerance,
                "right_joint2_control_offset_rad": (
                    self._right_joint2_control_offset_rad()
                ),
            }
        )
        if not passed:
            raise RuntimeError("eighth home initial-joint arrival gate failed")

    def _select_pair(
        self,
        *,
        stage_id: str,
        current: dict[str, Sequence[float]],
        candidates: dict[str, list[Sequence[float]]],
    ) -> dict[str, np.ndarray]:
        joint_limit = float(self.policy["max_incremental_joint_delta_rad"])
        target_positions = {
            side: np.asarray(
                self.stages[stage_id][side]["target_position_base_m"],
                dtype=float,
            )
            for side in ("left", "right")
        }
        valid = []
        for left_value, right_value in product(
            candidates["left"], candidates["right"]
        ):
            pair = {
                "left": np.asarray(left_value, dtype=float),
                "right": np.asarray(right_value, dtype=float),
            }
            if any(
                np.max(np.abs(pair[side] - np.asarray(current[side])))
                > joint_limit
                for side in ("left", "right")
            ):
                continue
            try:
                self._validate_pair_segment(
                    current, pair, stage_id=stage_id
                )
            except ValueError:
                continue
            errors = {
                side: float(
                    np.linalg.norm(
                        self.local_fk[side]
                        .fk(pair[side], self.torso)
                        .position_base_m
                        - target_positions[side]
                    )
                )
                for side in ("left", "right")
            }
            valid.append((max(errors.values()), sum(errors.values()), pair))
        if not valid:
            raise RuntimeError("no paired IK refinement candidate passed gates")
        valid.sort(key=lambda item: (item[0], item[1]))
        return valid[0][2]

    def _select_single_arm_candidate(
        self,
        *,
        side: str,
        inactive: str,
        stage_id: str,
        current: dict[str, Sequence[float]],
        inactive_hold: np.ndarray,
        candidates: list[Sequence[float]],
        target_position_base_m: Sequence[float] | None = None,
        probe_reference_position_base_m: Sequence[float] | None = None,
        probe_require_z_decrease: bool = True,
        probe_skip_collision: bool | None = None,
    ) -> np.ndarray:
        target = np.asarray(
            (
                target_position_base_m
                if target_position_base_m is not None
                else self.stages[stage_id][side]["target_position_base_m"]
            ),
            dtype=float,
        )
        valid = []
        legacy_replay_refinement = bool(
            getattr(self, "_p7_legacy_replay_refinement", False)
        )
        for value in candidates:
            active = np.asarray(value, dtype=float)
            maximum_joint_delta = (
                0.25
                if stage_id == "grasp_contact_probe"
                and legacy_replay_refinement
                else float(self.policy["max_incremental_joint_delta_rad"])
            )
            if (
                np.max(np.abs(active - np.asarray(current[side], dtype=float)))
                > maximum_joint_delta
            ):
                continue
            pair = {side: active, inactive: inactive_hold}
            skip_probe_collision = (
                stage_id == "grasp_contact_probe"
                and (
                    legacy_replay_refinement
                    or (
                        bool(probe_skip_collision)
                        if probe_skip_collision is not None
                        else self.post_arrival_probe_policy.get(
                            "skip_collision_during_descent"
                        )
                        is True
                    )
                )
            )
            if not skip_probe_collision:
                try:
                    self._validate_pair_segment(
                        current, pair, stage_id=stage_id
                    )
                except ValueError:
                    continue
            candidate_position = np.asarray(
                self.local_fk[side].fk(active, self.torso).position_base_m,
                dtype=float,
            )
            difference = candidate_position - target
            error = float(np.linalg.norm(difference))
            if stage_id == "grasp_contact_probe":
                if probe_reference_position_base_m is None:
                    raise ValueError(
                        "grasp contact probe requires the current TCP position"
                    )
                reference_position = np.asarray(
                    probe_reference_position_base_m, dtype=float
                )
                step_difference = candidate_position - reference_position
                if legacy_replay_refinement:
                    if abs(float(difference[2])) > 0.015:
                        continue
                    valid.append(
                        (
                            abs(float(difference[2])),
                            float(np.linalg.norm(active - current[side])),
                            active,
                        )
                    )
                    continue
                forbid_upward = self.post_arrival_probe_policy.get(
                    "forbid_upward_tcp_motion"
                ) is True
                if probe_require_z_decrease or forbid_upward:
                    if step_difference[2] >= 0.0:
                        continue
                elif abs(float(difference[2])) >= abs(
                    float(reference_position[2] - target[2])
                ):
                    continue
                if forbid_upward:
                    interpolation = np.linspace(
                        np.asarray(current[side], dtype=float),
                        active,
                        20,
                    )
                    trajectory_z = [
                        float(
                            self.local_fk[side]
                            .fk(joints, self.torso)
                            .position_base_m[2]
                        )
                        for joints in interpolation
                    ]
                    if any(
                        next_z > previous_z + 1e-6
                        for previous_z, next_z in zip(
                            trajectory_z, trajectory_z[1:]
                        )
                    ):
                        continue
                xy_shift = float(np.linalg.norm(step_difference[:2]))
                maximum_xy_shift = float(
                    self.post_arrival_probe_policy[
                        "maximum_xy_shift_per_descent_m"
                    ]
                )
                if xy_shift > maximum_xy_shift:
                    continue
                score = (
                    abs(float(difference[2]))
                    / float(self.policy["grasp_z_tolerance_m"])
                    + xy_shift / maximum_xy_shift
                )
                valid.append((score, error, active))
            elif stage_id == "grasp":
                xy_error = float(np.linalg.norm(difference[:2]))
                z_error = float(abs(difference[2]))
                score = max(
                    xy_error / float(self.policy["grasp_xy_tolerance_m"]),
                    z_error / float(self.policy["grasp_z_tolerance_m"]),
                )
                valid.append((score, error, active))
            else:
                valid.append((error, error, active))
        if not valid:
            raise RuntimeError("no single-arm IK candidate passed collision gates")
        valid.sort(key=lambda item: (item[0], item[1]))
        return valid[0][2]

    def _execute_p7_post_arrival_grasp_probe(
        self,
        *,
        side: str,
        inactive: str,
        inactive_hold: np.ndarray,
    ) -> dict[str, Any]:
        policy = self.post_arrival_probe_policy
        strict_contact = bool(getattr(self, "_strict_p7_contact", False))
        legacy_replay_refinement = bool(
            getattr(self, "_p7_legacy_replay_refinement", False)
        )
        initial_descent = float(policy["initial_descent_m"])
        refinement_iterations = int(policy["absolute_height_refinement_iterations"])
        if initial_descent <= 0.0 or refinement_iterations <= 0:
            raise ValueError("post-arrival grasp probe configuration is invalid")
        baseline = self._sample_median_joint_telemetry(side)
        if baseline is None:
            raise RuntimeError("post-arrival grasp probe requires effort feedback")
        self.trace.append(
            {
                "event": "p7_post_arrival_contact_baseline",
                "side": side,
                "capture": "before_initial_descent",
                "effort_nm": baseline["effort"].tolist(),
            }
        )
        start_pose = self._get_fresh_tool_pose(side)
        start_position = np.asarray(start_pose[0], dtype=float)
        preprobe_joints = {
            arm: np.asarray(values, dtype=float)
            for arm, values in self.controller.get_feedback_pair().items()
        }
        direction = np.asarray(policy["direction_base"], dtype=float)
        target_quaternion = self.stages["grasp"][side][
            "target_quaternion_xyzw"
        ]
        contacted = False
        actual_descent = 0.0
        completed_descent_commands = 0
        completed_refinement_commands = 0
        descent_unavailable_reason = None
        absolute_contact_position = np.asarray(
            self.stages["grasp"][side]["target_position_base_m"], dtype=float
        )
        absolute_contact_position[2] = float(
            policy["absolute_contact_tcp_z_base_m"]
        )
        ignore_absolute_height = policy.get(
            "ignore_absolute_contact_height"
        ) is True
        candidate_descents = tuple(
            getattr(self, "_p7_candidate_descents_m", ())
        )
        candidate_reference_position: np.ndarray | None = None
        incremental_step = float(
            policy.get("incremental_refinement_step_m", 0.0005)
        )
        commands = [
            (
                "initial_10mm_descent",
                start_position + direction * initial_descent,
                True,
            )
        ]
        if candidate_descents:
            commands.extend(
                (
                    "candidate_downward_ik",
                    np.asarray([descent, 0.0, 0.0], dtype=float),
                    True,
                )
                for descent in candidate_descents
            )
        else:
            commands.extend(
                (
                    "absolute_height_refinement",
                    absolute_contact_position,
                    False,
                )
                for _ in range(refinement_iterations)
            )
        for index, (phase, target_position, require_z_decrease) in enumerate(
            commands, start=1
        ):
            probe_reference_position = np.asarray(
                self._get_fresh_tool_pose(side)[0], dtype=float
            )
            if phase == "candidate_downward_ik":
                if candidate_reference_position is None:
                    candidate_reference_position = (
                        probe_reference_position.copy()
                    )
                candidate_descent = float(target_position[0])
                target_position = (
                    candidate_reference_position
                    + direction * candidate_descent
                )
            if phase == "absolute_height_refinement" and ignore_absolute_height:
                target_position = (
                    probe_reference_position + direction * incremental_step
                )
                require_z_decrease = True
            if (
                phase == "absolute_height_refinement"
                and not ignore_absolute_height
                and abs(
                    float(
                        probe_reference_position[2]
                        - absolute_contact_position[2]
                    )
                )
                <= float(policy["absolute_contact_z_tolerance_m"])
            ):
                break
            wait_until_ready = getattr(
                self.controller, "wait_until_ready", None
            )
            if callable(wait_until_ready):
                wait_until_ready(timeout_s=1.0, require_tool_pose=False)
            current = self.controller.get_feedback_pair()
            candidates = self.solver.solve_candidates(
                side=side,
                target_position_base_m=target_position,
                target_quaternion_xyzw=target_quaternion,
                reference_joints_rad=current[side],
                probes=(
                    20
                    if legacy_replay_refinement
                    else int(self.policy["ik_probes_per_iteration"])
                ),
            )
            try:
                selected = self._select_single_arm_candidate(
                    side=side,
                    inactive=inactive,
                    stage_id="grasp_contact_probe",
                    current=current,
                    inactive_hold=inactive_hold,
                    candidates=candidates,
                    target_position_base_m=target_position,
                    probe_reference_position_base_m=probe_reference_position,
                    probe_require_z_decrease=require_z_decrease,
                    probe_skip_collision=(
                        phase == "initial_10mm_descent"
                        or (
                            phase == "absolute_height_refinement"
                            and self.post_arrival_probe_policy.get(
                                "check_collision_during_absolute_height_refinement"
                            )
                            is False
                        )
                    ),
                )
            except RuntimeError as exc:
                if str(exc) != (
                    "no single-arm IK candidate passed collision gates"
                ):
                    raise
                descent_unavailable_reason = str(exc)
                self.trace.append(
                    {
                        "event": "p7_post_arrival_grasp_probe_unavailable",
                        "side": side,
                        "step_index": index,
                        "phase": phase,
                        "target_position_base_m": target_position.tolist(),
                        "reason": descent_unavailable_reason,
                        "close_authorized": not strict_contact,
                    }
                )
                if phase == "candidate_downward_ik":
                    continue
                break
            contact_monitor_samples: list[dict[str, Any]] = []
            required_consecutive_samples = (
                3 if strict_contact else 1
            )
            contact_streak = 0

            def contact_stop_requested() -> bool:
                nonlocal contact_streak
                if baseline is None:
                    return False
                actual_position = np.asarray(
                    self._get_fresh_tool_pose(side)[0], dtype=float
                )
                difference = actual_position - target_position
                raw_detected, evidence = self._grasp_effort_contact_detected(
                    side=side,
                    reference_effort=baseline["effort"],
                    error_components={
                        "norm_m": float(np.linalg.norm(difference)),
                        "xy_m": float(np.linalg.norm(difference[:2])),
                        "z_m": float(abs(difference[2])),
                        "x_signed_m": float(difference[0]),
                        "y_signed_m": float(difference[1]),
                        "z_signed_m": float(difference[2]),
                    },
                    sample_count=5,
                    residual_threshold_nm=2.5,
                    require_settled_velocity=strict_contact,
                    require_near_table=strict_contact,
                )
                contact_streak = (
                    contact_streak + 1 if raw_detected else 0
                )
                detected = contact_streak >= required_consecutive_samples
                contact_monitor_samples.append(
                    {
                        "sample_index": len(contact_monitor_samples) + 1,
                        "tcp_position_base_m": actual_position.tolist(),
                        **evidence,
                        "raw_detected": raw_detected,
                        "detected": detected,
                        "contact_streak": contact_streak,
                        "required_consecutive_samples": (
                            required_consecutive_samples
                        ),
                    }
                )
                return detected

            stopped = self.controller.execute_single_arm_motion(
                side=side,
                target=selected,
                inactive_hold=inactive_hold,
                progress_steps=1,
                stop_requested=(
                    None
                    if legacy_replay_refinement
                    and phase == "initial_10mm_descent"
                    else contact_stop_requested
                ),
            )
            completed_descent_commands += 1
            if phase in (
                "absolute_height_refinement",
                "candidate_downward_ik",
            ):
                completed_refinement_commands += 1
            self._record_actual_tcp(
                side=side,
                stage_id="grasp",
                phase="post_arrival_contact_probe",
                sample_index=index,
            )
            actual_position = np.asarray(
                self._get_fresh_tool_pose(side)[0], dtype=float
            )
            difference = actual_position - target_position
            actual_descent = float(start_position[2] - actual_position[2])
            error_components = {
                "norm_m": float(np.linalg.norm(difference)),
                "xy_m": float(np.linalg.norm(difference[:2])),
                "z_m": float(abs(difference[2])),
                "x_signed_m": float(difference[0]),
                "y_signed_m": float(difference[1]),
                "z_signed_m": float(difference[2]),
            }
            contacted = bool(stopped) or any(
                sample.get("detected") is True
                for sample in contact_monitor_samples
            )
            if (
                not contacted
                and not (
                    legacy_replay_refinement
                    and phase == "initial_10mm_descent"
                )
            ):
                contacted = contact_stop_requested()
            if (
                legacy_replay_refinement
                and phase == "initial_10mm_descent"
            ):
                baseline = self._sample_median_joint_telemetry(side)
                if baseline is None:
                    raise RuntimeError(
                        "legacy replay refinement requires post-descent "
                        "effort feedback"
                    )
                self.trace.append(
                    {
                        "event": "p7_legacy_post_descent_contact_baseline",
                        "side": side,
                        "capture": "after_unsegmented_initial_10mm_descent",
                        "effort_nm": baseline["effort"].tolist(),
                    }
                )
            evidence = (
                contact_monitor_samples[-1]
                if contact_monitor_samples
                else {"available": False, "detected": False}
            )
            self.trace.append(
                {
                    "event": "p7_post_arrival_grasp_probe",
                    "side": side,
                    "step_index": index,
                    "phase": phase,
                    "initial_descent_m": initial_descent,
                    "actual_descent_m": actual_descent,
                    "target_position_base_m": target_position.tolist(),
                    "actual_position_base_m": actual_position.tolist(),
                    "continuous_contact_monitoring": True,
                    "contact_monitor_samples": contact_monitor_samples,
                    **evidence,
                }
            )
            if contacted:
                self.controller.hold()
                break
            if phase == "candidate_downward_ik":
                break
        if not contacted:
            result = {
                "contact_detected": False,
                "completed_descent_commands": completed_descent_commands,
                "initial_descent_m": initial_descent,
                "completed_refinement_commands": completed_refinement_commands,
                "absolute_contact_tcp_z_base_m": float(
                    absolute_contact_position[2]
                ),
                "absolute_contact_height_ignored": ignore_absolute_height,
                "incremental_refinement_step_m": incremental_step,
                "candidate_descents_m": list(candidate_descents),
                "actual_descent_m": actual_descent,
                "descent_unavailable_reason": descent_unavailable_reason,
                "close_authorized_by": (
                    None
                    if strict_contact
                    else (
                        "descent_condition_unavailable"
                        if descent_unavailable_reason is not None
                        else (
                            "maximum_incremental_refinements_completed_without_contact"
                            if ignore_absolute_height
                            else "absolute_height_refinement_completed_without_contact"
                        )
                    )
                ),
                "close_authorized": not strict_contact,
            }
            self.trace.append(
                {
                    "event": "p7_post_arrival_grasp_probe_completed",
                    "side": side,
                    **result,
                }
            )
            if strict_contact:
                self.controller.execute_single_arm_motion(
                    side=side,
                    target=preprobe_joints[side],
                    inactive_hold=preprobe_joints[inactive],
                    progress_steps=5,
                )
                self.trace.append(
                    {
                        "event": "p7_strict_contact_failure_retracted",
                        "side": side,
                    }
                )
                raise RuntimeError(
                    "strict P7 grasp probe found no stable near-table "
                    "effort contact; gripper close forbidden"
                )
            return result
        result = {
            "contact_detected": True,
            "completed_descent_commands": completed_descent_commands,
            "initial_descent_m": initial_descent,
            "completed_refinement_commands": completed_refinement_commands,
            "absolute_contact_tcp_z_base_m": float(
                absolute_contact_position[2]
            ),
            "absolute_contact_height_ignored": ignore_absolute_height,
            "incremental_refinement_step_m": incremental_step,
            "candidate_descents_m": list(candidate_descents),
            "actual_descent_m": actual_descent,
            "descent_unavailable_reason": descent_unavailable_reason,
            "close_authorized_by": "effort_contact",
        }
        self.trace.append(
            {
                "event": "p7_post_arrival_grasp_probe_completed",
                "side": side,
                **result,
            }
        )
        return result

    def _validate_post_close_tcp_z_segment(
        self,
        *,
        side: str,
        start_joints: Sequence[float],
        target_joints: Sequence[float],
        stage_id: str,
        phase: str,
        sample_index: int,
    ) -> None:
        if (
            self._post_close_z_policy.get("enabled") is not True
            or side not in self._closed_gripper_sides
        ):
            return
        if (
            stage_id == "place_center"
            and sample_index >= 2
            and getattr(self, "_p7_disable_place_z_guard", False)
        ):
            self.trace.append(
                {
                    "event": "post_close_tcp_z_segment_check_skipped",
                    "side": side,
                    "stage_id": stage_id,
                    "phase": phase,
                    "sample_index": sample_index,
                    "reason": "pipeline7_place_z_guard_disabled",
                }
            )
            return
        samples = int(self._post_close_z_policy.get("fk_samples_per_segment", 50))
        tolerance = float(
            self._post_close_z_policy.get("planned_decrease_tolerance_m", 0.0002)
        )
        start = np.asarray(start_joints, dtype=float)
        target = np.asarray(target_joints, dtype=float)
        z_values = [
            float(
                self.local_fk[side]
                .fk(start * (1.0 - alpha) + target * alpha, self.torso)
                .position_base_m[2]
            )
            for alpha in np.linspace(0.0, 1.0, samples + 1)
        ]
        minimum_delta = min(np.diff(z_values), default=0.0)
        total_delta = z_values[-1] - z_values[0]
        horizontal_tolerance = float(
            getattr(self, "_p7_horizontal_place_z_tolerance_m", 0.0)
        )
        horizontal_segment = (
            stage_id == "place_center"
            and sample_index >= 2
            and horizontal_tolerance > 0.0
        )
        if horizontal_segment:
            if self._horizontal_place_planned_z_reference is None:
                self._horizontal_place_planned_z_reference = z_values[0]
            horizontal_reference = (
                self._horizontal_place_planned_z_reference
            )
            maximum_horizontal_error = max(
                abs(value - horizontal_reference) for value in z_values
            )
            passed = maximum_horizontal_error <= horizontal_tolerance
        else:
            horizontal_reference = None
            maximum_horizontal_error = None
            passed = bool(
                minimum_delta >= -tolerance and total_delta >= -tolerance
            )
        evidence = {
            "event": "post_close_tcp_z_segment_checked",
            "side": side,
            "stage_id": stage_id,
            "phase": phase,
            "sample_index": sample_index,
            "start_tcp_z_base_m": z_values[0],
            "end_tcp_z_base_m": z_values[-1],
            "minimum_sample_delta_m": float(minimum_delta),
            "total_delta_m": float(total_delta),
            "horizontal_corridor_enabled": horizontal_segment,
            "horizontal_reference_z_m": horizontal_reference,
            "maximum_horizontal_error_m": maximum_horizontal_error,
            "horizontal_tolerance_m": (
                horizontal_tolerance if horizontal_segment else None
            ),
            "samples": samples,
            "passed": passed,
        }
        self.trace.append(evidence)
        if not evidence["passed"]:
            raise RuntimeError(
                "post-close TCP Z monotonic gate failed before motion: "
                f"{evidence}"
            )

    def _verify_post_close_tcp_z(
        self,
        *,
        side: str,
        stage_id: str,
        phase: str,
        sample_index: int,
    ) -> None:
        if (
            self._post_close_z_policy.get("enabled") is not True
            or side not in self._closed_gripper_sides
        ):
            return
        if (
            stage_id == "place_center"
            and sample_index >= 2
            and getattr(self, "_p7_disable_place_z_guard", False)
        ):
            current_z = float(self._get_fresh_tool_pose(side)[0][2])
            self.trace.append(
                {
                    "event": "post_close_tcp_z_feedback_check_skipped",
                    "side": side,
                    "stage_id": stage_id,
                    "phase": phase,
                    "sample_index": sample_index,
                    "current_tcp_z_base_m": current_z,
                    "reason": "pipeline7_place_z_guard_disabled",
                }
            )
            self._post_close_tcp_z[side] = current_z
            return
        current_z = float(self._get_fresh_tool_pose(side)[0][2])
        previous_z = self._post_close_tcp_z[side]
        tolerance = float(
            self._post_close_z_policy.get("measured_decrease_tolerance_m", 0.002)
        )
        horizontal_tolerance = float(
            getattr(self, "_p7_horizontal_place_z_tolerance_m", 0.0)
        )
        if (
            stage_id == "place_center"
            and sample_index == 1
            and horizontal_tolerance > 0.0
        ):
            self._horizontal_place_measured_z_reference = current_z
        horizontal_feedback = (
            stage_id == "place_center"
            and sample_index >= 2
            and horizontal_tolerance > 0.0
            and self._horizontal_place_measured_z_reference is not None
        )
        if horizontal_feedback:
            measured_reference = (
                self._horizontal_place_measured_z_reference
            )
            horizontal_error = abs(current_z - measured_reference)
            passed = horizontal_error <= horizontal_tolerance
        else:
            measured_reference = None
            horizontal_error = None
            passed = current_z >= previous_z - tolerance
        self.trace.append(
            {
                "event": "post_close_tcp_z_feedback_checked",
                "side": side,
                "stage_id": stage_id,
                "phase": phase,
                "sample_index": sample_index,
                "previous_tcp_z_base_m": previous_z,
                "current_tcp_z_base_m": current_z,
                "delta_m": current_z - previous_z,
                "tolerance_m": tolerance,
                "horizontal_corridor_enabled": horizontal_feedback,
                "horizontal_reference_z_m": measured_reference,
                "horizontal_error_m": horizontal_error,
                "horizontal_tolerance_m": (
                    horizontal_tolerance if horizontal_feedback else None
                ),
                "passed": passed,
            }
        )
        if not passed:
            self.controller.hold()
            raise RuntimeError("post-close TCP Z feedback decreased beyond tolerance")
        self._post_close_tcp_z[side] = max(previous_z, current_z)

    def _validate_pair_segment(
        self,
        start: dict[str, Sequence[float]],
        end: dict[str, Sequence[float] | np.ndarray],
        *,
        fingers_open: bool = True,
        stage_id: str | None = None,
    ) -> None:
        p7_closed_lift = (
            not self._is_eighth
            and stage_id == "lift"
            and self.plan["safety"].get(
                "post_grasp_closed_lift_collision_policy"
            )
            == "skip_table_collision_if_gripper_close_succeeded"
            and self.plan["phase_side"] in self._closed_gripper_sides
            and bool(self._completed)
            and self._completed[-1] == "grasp"
        )
        if p7_closed_lift:
            self.trace.append(
                {
                    "event": "collision_check_skipped",
                    "pipeline": 7,
                    "stage_id": "lift",
                    "reason": "gripper_close_succeeded",
                }
            )
            return
        effort_guarded_grasp = (
            self._is_eighth
            and stage_id == "grasp"
            and self.contact_probe_policy.get(
                "execution_grasp_table_gate_bypassed"
            )
            is True
        )
        if effort_guarded_grasp or not self._collision_checks_enabled or (
            self._is_eighth and stage_id not in self._collision_stage_ids
        ):
            self.trace.append(
                {
                    "event": "collision_check_skipped",
                    "policy": self.plan.get("safety", {}).get(
                        "collision_policy", "disabled"
                    ),
                    "pipeline": 8,
                    "stage_id": stage_id,
                    "effort_guarded_grasp": effort_guarded_grasp,
                }
            )
            return
        self.safety.validate_interpolated_segment(
            np.asarray(start["left"]),
            np.asarray(start["right"]),
            np.asarray(end["left"]),
            np.asarray(end["right"]),
            self.torso,
            samples=50,
            fingers_open=fingers_open,
            **(
                {
                    "allowed_contact_finger_link": self.plan["safety"][
                        "grasp_allowed_table_contact_link"
                    ],
                    "minimum_noncontact_clearance_m": float(
                        (
                            self.post_arrival_probe_policy[
                                "noncontact_min_clearance_m"
                            ]
                            if stage_id == "grasp_contact_probe"
                            else self.plan["safety"][
                                "grasp_noncontact_min_clearance_m"
                            ]
                        )
                    ),
                    "minimum_contact_clearance_m": float(
                        (
                            self.post_arrival_probe_policy[
                                "designated_contact_min_clearance_m"
                            ]
                            if stage_id == "grasp_contact_probe"
                            else self.plan["safety"][
                                "grasp_designated_contact_min_clearance_m"
                            ]
                        )
                    ),
                }
                if not self._is_eighth
                and stage_id in ("grasp", "grasp_contact_probe")
                else {}
            ),
        )
        if self._is_eighth:
            self.trace.append(
                {
                    "event": "table_collision_check_passed",
                    "policy": self.plan["safety"]["collision_policy"],
                    "pipeline": 8,
                    "stage_id": stage_id,
                    "sweep_samples": 50,
                    "fingers_open": fingers_open,
                }
            )

    def _cartesian_errors(self, stage_id: str) -> dict[str, float]:
        poses = self._get_fresh_tool_pose_pair()
        return {
            side: float(
                np.linalg.norm(
                    np.asarray(poses[side][0], dtype=float)
                    - np.asarray(
                        self.stages[stage_id][side]["target_position_base_m"],
                        dtype=float,
                    )
                )
            )
            for side in ("left", "right")
        }

    def _cartesian_error_side(self, stage_id: str, side: str) -> float:
        return self._cartesian_error_components_side(stage_id, side)["norm_m"]

    def _cartesian_error_components_side(
        self,
        stage_id: str,
        side: str,
    ) -> dict[str, float]:
        self.controller.wait_until_ready(
            timeout_s=1.0,
            require_tool_pose=True,
        )
        pose = self._get_fresh_tool_pose(side)[0]
        difference = np.asarray(pose, dtype=float) - np.asarray(
            self.stages[stage_id][side]["target_position_base_m"],
            dtype=float,
        )
        return {
            "norm_m": float(np.linalg.norm(difference)),
            "xy_m": float(np.linalg.norm(difference[:2])),
            "z_m": float(abs(difference[2])),
            "x_signed_m": float(difference[0]),
            "y_signed_m": float(difference[1]),
            "z_signed_m": float(difference[2]),
        }

    def _sample_median_joint_telemetry(
        self, side: str, *, sample_count: int | None = None
    ) -> dict[str, np.ndarray] | None:
        getter = getattr(self.controller, "get_joint_telemetry", None)
        if getter is None:
            return None
        efforts = []
        velocities = []
        refresh = getattr(self.controller, "refresh_feedback", None)
        try:
            count = (
                int(sample_count)
                if sample_count is not None
                else int(self.effort_contact_policy["sample_count"])
            )
            for _ in range(count):
                if refresh is not None:
                    refresh(0.01)
                sample = getter(side)
                efforts.append(np.asarray(sample["effort"], dtype=float))
                velocities.append(np.asarray(sample["velocity"], dtype=float))
        except Exception:
            return None
        return {
            "effort": np.median(np.asarray(efforts), axis=0),
            "median_abs_velocity": np.median(
                np.abs(np.asarray(velocities)), axis=0
            ),
        }

    def _grasp_effort_contact_detected(
        self,
        *,
        side: str,
        reference_effort: np.ndarray,
        error_components: dict[str, float],
        sample_count: int | None = None,
        residual_threshold_nm: float | None = None,
        require_settled_velocity: bool = True,
        require_near_table: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        telemetry = self._sample_median_joint_telemetry(
            side, sample_count=sample_count
        )
        if telemetry is None:
            return False, {"available": False, "detected": False}
        effort = telemetry["effort"]
        indices = [
            int(value)
            for value in self.effort_contact_policy["joint_indices_zero_based"]
        ]
        residual = effort - reference_effort
        score = float(np.sum(np.abs(residual[indices])))
        maximum_velocity = float(np.max(telemetry["median_abs_velocity"]))
        near_table = (
            error_components["z_signed_m"]
            <= float(
                self.effort_contact_policy["maximum_tcp_z_above_target_m"]
            )
        )
        settled = maximum_velocity <= float(
            self.effort_contact_policy[
                "maximum_median_abs_velocity_rad_s"
            ]
        )
        threshold = (
            float(self.effort_contact_policy["sum_abs_residual_threshold_nm"])
            if residual_threshold_nm is None
            else float(residual_threshold_nm)
        )
        detected = (
            (near_table or not require_near_table)
            and (settled or not require_settled_velocity)
            and score >= threshold
        )
        return detected, {
            "available": True,
            "detected": detected,
            "effort_nm": effort.tolist(),
            "effort_residual_nm": residual.tolist(),
            "joint_residual_score_nm": score,
            "residual_threshold_nm": threshold,
            "median_max_abs_velocity_rad_s": maximum_velocity,
            "near_table": near_table,
            "settled": settled,
            "near_table_required": require_near_table,
            "settled_velocity_required": require_settled_velocity,
        }

    def _single_arm_arrived(
        self,
        stage_id: str,
        error_components: dict[str, float],
    ) -> bool:
        if stage_id == "grasp":
            if self.post_arrival_probe_policy.get("enabled") is True:
                return (
                    error_components["xy_m"]
                    <= float(self.policy["grasp_xy_tolerance_m"])
                    and error_components["z_m"]
                    < float(
                        self.post_arrival_probe_policy[
                            "maximum_entry_z_error_m"
                        ]
                    )
                )
            return (
                error_components["xy_m"]
                <= float(self.policy["grasp_xy_tolerance_m"])
                and error_components["z_m"]
                <= float(self.policy["grasp_z_tolerance_m"])
            )
        return error_components["norm_m"] <= float(
            self.policy["target_tolerance_m"]
        )

    def _require_bound(self) -> None:
        if not self._bound:
            raise RuntimeError("bimanual policy must bind the approved plan first")
