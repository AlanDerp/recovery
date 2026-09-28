import numpy as np
import pytest
from types import SimpleNamespace

from capx.integrations.r1lite.bimanual_towel_executor import (
    BimanualTowelExecutor,
)
from capx.integrations.r1lite.execution_bundle import EIGHTH_STAGE_ORDER


class FakeController:
    def __init__(self):
        self.joints = {
            "left": np.full(6, 0.4),
            "right": np.full(6, -0.4),
        }
        self.pose = {
            "left": (np.array([0.0, 0.0, 0.7]), np.array([0, 0, 0, 1])),
            "right": (np.array([0.2, 0.0, 0.7]), np.array([0, 0, 0, 1])),
        }
        self.progress_steps = []
        self.holds = 0
        self.closed = 0
        self.gripper_hold_enabled = False

    def get_feedback_pair(self):
        return {side: joints.copy() for side, joints in self.joints.items()}

    def execute_joint_motion(
        self, left, right, *, progress_steps=None, trace_context=None
    ):
        self.joints = {
            "left": np.asarray(left, dtype=float),
            "right": np.asarray(right, dtype=float),
        }
        self.progress_steps.append(progress_steps)

    def get_tool_pose(self, side):
        return self.pose[side]

    def get_tool_pose_pair(self):
        return self.pose

    def execute_single_arm_motion(
        self,
        *,
        side,
        target,
        inactive_hold,
        progress_steps=None,
        trace_context=None,
    ):
        inactive = "right" if side == "left" else "left"
        self.joints[side] = np.asarray(target, dtype=float)
        self.joints[inactive] = np.asarray(inactive_hold, dtype=float)
        self.progress_steps.append(progress_steps)

    def close_both_grippers(self):
        self.closed += 1

    def close_both_grippers_stable_and_hold(self):
        self.closed += 1
        self.gripper_hold_enabled = True
        return {
            "initial_position": {"left": 40.0, "right": 40.0},
            "final_position": {"left": 5.0, "right": 6.0},
            "reached_close_target": True,
            "stable_contact_position": True,
            "validated_contact_position": True,
            "minimum_close_command_duration_s": 2.0,
            "continuous_close_hold_enabled": True,
        }

    def clear_gripper_hold(self):
        self.gripper_hold_enabled = False

    def wait_until_ready(self, **_kwargs):
        return None

    def hold(self):
        self.holds += 1


class FakeSafety:
    def __init__(self):
        self.samples = []

    def validate_interpolated_segment(self, *_args, **kwargs):
        self.samples.append(kwargs["samples"])
        return []


class UnusedSolver:
    def solve_candidates(self, **_kwargs):
        return []


def make_executor():
    stages = []
    previous_left = 0.4
    previous_right = -0.4
    for index, stage_id in enumerate(EIGHTH_STAGE_ORDER):
        if stage_id == "home":
            left_joint, right_joint = 0.0, 0.0
        else:
            left_joint = right_joint = 0.05 * (index + 1)
        waypoint = {
            "alpha": 1.0,
            "ik_acceptance": (
                "best_effort_seed"
                if stage_id == "grasp"
                else "joint_transition"
                if stage_id == "home"
                else "required"
            ),
            "expected_midpoint_base_m": [0.1, 0.0, 0.7],
            "expected_right_minus_left_m": [0.2, 0.0, 0.0],
            "left": {
                "target_position_base_m": [0.0, 0.0, 0.7],
                "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "joints_rad": [left_joint] * 6,
            },
            "right": {
                "target_position_base_m": [0.2, 0.0, 0.7],
                "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "joints_rad": [right_joint] * 6,
            },
        }
        stages.append({"id": stage_id, "waypoints": [waypoint]})
        previous_left, previous_right = left_joint, right_joint
    plan = {
        "schema": "capx.r1lite.eighth_bimanual_recenter.v33",
        "pipeline": 8,
        "stages": [
            {
                "id": stage_id,
                **(
                    {
                        "motion_policy": (
                            "skip_after_release_direct_home"
                        )
                    }
                    if stage_id == "release_retract"
                    else {}
                ),
            }
            for stage_id in EIGHTH_STAGE_ORDER
        ],
        "grasp_geometry": {
            "initial_full_relative_vector_right_minus_left_m": [0.2, 0.0, 0.0]
        },
        "safety": {
            "table_z_m": 0.0,
            "towel_thickness_m": 0.00002,
            "tcp_to_fingertip_m": 0.078,
            "joint_command_progress_steps": 5,
            "grasp_x_tolerance_m": 0.005,
            "grasp_maximum_inward_x_error_m": 0.005,
            "translation_control_policy": {
                "mode": "midpoint_plus_fixed_grasp_relative_vector",
                "reference_vector_source": (
                    "initial_full_relative_vector_right_minus_left_m"
                ),
            },
            "collision_policy": "first_phase_collision_disabled",
            "execution_collision_checks_enabled": False,
            "collision_checked_stage_ids": [],
            "grasp_refinement_policy": {
                "enabled": True,
                "maximum_iterations": 20,
                "individual_tcp_tolerance_m": 0.015,
                "initial_seed_policy": (
                    "best_joint_bounded_candidate_without_ik_fk_acceptance"
                ),
                "initial_seed_ik_fk_acceptance_required": False,
                "initial_seed_maximum_total_joint_delta_rad": 3.0,
                "initial_seed_maximum_substep_joint_delta_rad": 1.0,
                "gripper_close_requires_both_tcp_within_tolerance": False,
                "completion_policy": (
                    "force_continue_after_maximum_iterations"
                ),
                "post_refinement_tcp_error_gate_required": False,
                "no_local_optimum_early_exit": True,
                "strategy": "independent_preclose_worst_error_first",
                "synchronized_pair_only": False,
                "inactive_arm_holds": True,
                "final_relative_geometry_gate_required": False,
            },
            "contact_probe_policy": {
                "enabled": True,
                "probe_order": ["right", "left"],
                "axis_base": [0.0, 0.0, -1.0],
                "step_m": 0.0005,
                "maximum_descent_m": 0.04,
                "sample_count": 40,
                "joint_indices_zero_based": [1, 2],
                "residual_reference_mode": "probe_start_fixed",
                "sum_abs_residual_threshold_nm": 6.0,
                "single_joint_emergency_threshold_nm": 5.0,
                "maximum_median_abs_velocity_rad_s": 0.05,
                "confirmation_samples": 2,
                "freeze_on_first_contact_indication": True,
                "stationary_confirmation_maximum_attempts": 12,
                "ik_probes_per_step": 8,
                "maximum_fk_position_error_m": 0.015,
                "maximum_incremental_joint_delta_rad": 0.25,
                "execution_grasp_table_gate_bypassed": True,
                "lock_xy_to_grasp_target": True,
            },
        },
    }
    bundle = {
        "schema": "capx.r1lite.eighth_execution_bundle.v14",
        "artifact_bindings": {"plan_sha256": "approved"},
        "motion_limits": {
            "fk_position_tolerance_m": 0.03,
            "relative_geometry_tolerance_m": 0.005,
            "maximum_waypoint_joint_delta_rad": 1.0,
        },
        "initial_state": {
            "left_joints_rad": [0.0] * 6,
            "right_joints_rad": [0.0] * 6,
            "torso_joints_rad": [0.0] * 3,
        },
        "stage_order": list(EIGHTH_STAGE_ORDER),
        "stages": stages,
    }
    controller = FakeController()
    safety = FakeSafety()
    executor = BimanualTowelExecutor(
        plan=plan,
        plan_sha256="approved",
        bundle=bundle,
        controller=controller,
        solver=UnusedSolver(),
        safety=safety,
    )
    executor.bind_plan(sha256="approved")
    return executor, controller, safety


def test_eighth_executes_certified_shared_waypoints_without_refinement():
    executor, controller, safety = make_executor()

    for stage_id in EIGHTH_STAGE_ORDER:
        executor.execute_eighth_stage(stage_id=stage_id)

    assert controller.progress_steps == [5] * 6
    assert safety.samples == []
    assert len(
        [
            event
            for event in executor.trace
            if event["event"] == "collision_check_skipped"
        ]
    ) == 6
    assert any(
        event["event"] == "collision_check_skipped"
        and event["stage_id"] == "grasp"
        and event["effort_guarded_grasp"] is True
        for event in executor.trace
    )
    assert controller.holds == 0
    assert len(
        [event for event in executor.trace if event["event"] == "geometry_gate"]
    ) == 5
    assert not any(
        event["event"] == "eighth_right_takeover_geometry_gate"
        for event in executor.trace
    )
    assert executor.trace[-1]["event"] == "eighth_stage_completed"


def test_eighth_recloses_grippers_before_first_post_lift_waypoint():
    executor, controller, _safety = make_executor()
    executor.execute_eighth_stage(stage_id="grasp")
    executor.execute_eighth_stage(stage_id="lift")
    calls = []
    original_reclose = controller.close_both_grippers_stable_and_hold
    original_motion = controller.execute_joint_motion

    def reclose():
        calls.append("reclose")
        return original_reclose()

    def move(left, right, *, progress_steps=None, trace_context=None):
        calls.append("motion")
        return original_motion(
            left,
            right,
            progress_steps=progress_steps,
            trace_context=trace_context,
        )

    controller.close_both_grippers_stable_and_hold = reclose
    controller.execute_joint_motion = move

    executor.execute_eighth_stage(stage_id="pullback")

    assert calls[:2] == ["reclose", "motion"]
    event = next(
        item
        for item in executor.trace
        if item["event"] == "post_lift_grippers_reclosed_before_pullback"
    )
    assert event["waypoint_index"] == 1
    assert event["wait_duration_s"] == pytest.approx(2.0)


def test_translate_targets_are_derived_from_midpoint_and_grasp_vector():
    executor, _controller, _safety = make_executor()
    waypoint = executor.stages["translate_center"]["waypoints"][0]
    waypoint["left"]["target_position_base_m"] = [9.0, 9.0, 9.0]
    waypoint["right"]["target_position_base_m"] = [-9.0, -9.0, -9.0]

    targets = executor._eighth_cartesian_targets(
        stage_id="translate_center", waypoint=waypoint
    )

    assert targets["left"] == pytest.approx([0.0, 0.0, 0.7])
    assert targets["right"] == pytest.approx([0.2, 0.0, 0.7])


def test_second_phase_is_not_embedded_in_eighth_executor():
    executor, _controller, _safety = make_executor()
    assert "right_regrasp" not in executor.stages
    assert list(executor.stages) == list(EIGHTH_STAGE_ORDER)


def test_eighth_relative_vector_error_is_recorded_but_does_not_block():
    executor, controller, _safety = make_executor()
    controller.pose["right"] = (
        np.array([0.2, 0.01, 0.7]),
        np.array([0, 0, 0, 1]),
    )

    executor.execute_eighth_stage(stage_id="grasp")

    assert controller.holds == 0
    gate = [event for event in executor.trace if event["event"] == "geometry_gate"][-1]
    assert gate["passed"] is True
    assert gate["expected_relative_vector_error_m"] == pytest.approx(0.01)
    assert gate["pair_error_gates_enforced"] is False


def test_open_grasp_refines_only_worst_arm_while_other_holds():
    executor, controller, _safety = make_executor()
    controller.pose["left"] = (
        np.array([0.03, 0.0, 0.7]),
        np.array([0, 0, 0, 1]),
    )

    class Solver:
        def __init__(self):
            self.sides = []

        def solve_candidates(self, **kwargs):
            self.sides.append(kwargs["side"])
            return [np.zeros(6)]

    solver = Solver()
    executor.solver = solver
    original_execute = controller.execute_joint_motion
    calls = 0

    def execute(
        left, right, *, progress_steps=None, trace_context=None
    ):
        nonlocal calls
        calls += 1
        original_execute(left, right, progress_steps=progress_steps)
        if calls == 2:
            controller.pose["left"] = (
                np.array([0.0, 0.0, 0.7]),
                np.array([0, 0, 0, 1]),
            )

    controller.execute_joint_motion = execute
    executor.execute_eighth_stage(stage_id="grasp")

    events = [
        event
        for event in executor.trace
        if event["event"] == "eighth_independent_grasp_refinement"
    ]
    assert solver.sides == ["left"]
    assert events[0]["active_side"] == "left"
    assert events[0]["inactive_side"] == "right"
    assert events[0]["passed"] is True


def test_grasp_aborts_after_twenty_refinements_if_x_gate_still_fails():
    executor, controller, _safety = make_executor()
    controller.pose["left"] = (
        np.array([0.03, 0.0, 0.7]),
        np.array([0, 0, 0, 1]),
    )

    class Solver:
        def solve_candidates(self, **_kwargs):
            return [np.zeros(6)]

    executor.solver = Solver()
    with pytest.raises(RuntimeError, match="configured X gate"):
        executor.execute_eighth_stage(stage_id="grasp")

    forced = [
        event
        for event in executor.trace
        if event["event"] == "eighth_refinement_force_continued"
    ]
    assert forced[0]["grasp_x_gate_passed"] is False
    assert forced[0]["gripper_close_authorized"] is False
    assert controller.holds == 1


def test_eighth_rejects_out_of_order_stage_and_holds():
    executor, controller, safety = make_executor()

    with pytest.raises(ValueError, match="bundle stage_order"):
        executor.execute_eighth_stage(stage_id="lift")

    assert controller.holds == 1
    assert controller.progress_steps == []
    assert safety.samples == []


def test_eighth_home_targets_initial_joints_and_checks_arrival():
    executor, controller, safety = make_executor()
    executor._completed = list(EIGHTH_STAGE_ORDER[:-1])

    executor.execute_eighth_stage(stage_id="home")

    assert controller.joints["left"] == pytest.approx(np.zeros(6))
    assert controller.joints["right"] == pytest.approx(np.zeros(6))
    assert controller.progress_steps == [5]
    assert safety.samples == []
    gate = [event for event in executor.trace if event["event"] == "home_joint_gate"][-1]
    assert gate["passed"] is True


def test_eighth_home_gate_compares_offset_right_feedback_in_logical_space():
    executor, controller, _safety = make_executor()
    controller.right_joint2_control_offset_rad = 0.2
    controller.right_joint2_control_offset_enabled = True
    controller.joints = {
        "left": np.zeros(6),
        "right": np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
    }

    executor._eighth_home_arrival_gate()

    gate = [event for event in executor.trace if event["event"] == "home_joint_gate"][-1]
    assert gate["passed"] is True
    assert gate["joint_errors_rad"]["right"] == pytest.approx(0.0)
    assert gate["right_joint2_control_offset_rad"] == pytest.approx(0.2)


def test_eighth_geometry_gate_removes_right_control_offset_fk_delta():
    executor, controller, _safety = make_executor()
    controller.right_joint2_control_offset_rad = 0.2
    controller.right_joint2_control_offset_enabled = True
    controller.joints["right"] = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    controller.pose["right"] = (
        np.array([0.4, 0.0, 0.7]),
        np.array([0, 0, 0, 1]),
    )

    class OffsetFK:
        def fk(self, joints, _torso):
            return SimpleNamespace(
                position_base_m=np.array([0.2 + joints[1], 0.0, 0.7])
            )

    executor.local_fk["right"] = OffsetFK()
    waypoint = executor.stages["lift"]["waypoints"][0]

    passed = executor._eighth_waypoint_geometry_gate(
        stage_id="lift",
        waypoint_index=1,
        waypoint=waypoint,
        raise_on_failure=False,
    )

    assert passed is True
    gate = [event for event in executor.trace if event["event"] == "geometry_gate"][-1]
    assert gate["raw_actual_tcp"]["right"] == pytest.approx([0.4, 0.0, 0.7])
    assert gate["actual_tcp"]["right"] == pytest.approx([0.2, 0.0, 0.7])
    assert gate["right_control_offset_tcp_correction_m"] == pytest.approx(
        [-0.2, 0.0, 0.0]
    )


def test_non_enforced_post_grasp_geometry_gate_refines_only_once():
    executor, controller, _safety = make_executor()
    executor._completed = ["grasp"]
    controller.pose["left"] = (
        np.array([0.03, 0.0, 0.7]),
        np.array([0, 0, 0, 1]),
    )
    controller.pose["right"] = (
        np.array([0.23, 0.0, 0.7]),
        np.array([0, 0, 0, 1]),
    )
    executor._select_eighth_refinement_pair = lambda **_kwargs: {
        "left": np.zeros(6),
        "right": np.zeros(6),
    }

    executor.execute_eighth_stage(stage_id="lift")

    assert controller.progress_steps == [5, 5]
    forced = [
        event
        for event in executor.trace
        if event["event"] == "eighth_refinement_force_continued"
    ]
    assert forced[-1]["iterations"] == 1
    assert forced[-1]["reason"] == "non_enforced_geometry_gates_single_attempt"


class ProbeSolver:
    def __init__(self):
        self.targets = {}

    def solve_candidates(self, **kwargs):
        side = kwargs["side"]
        self.targets[side] = np.asarray(
            kwargs["target_position_base_m"], dtype=float
        )
        return [np.asarray(kwargs["reference_joints_rad"], dtype=float) + 0.001]


class ProbeFK:
    def __init__(self, side, solver):
        self.side = side
        self.solver = solver

    def fk(self, _joints, _torso):
        return SimpleNamespace(position_base_m=self.solver.targets[self.side])


def telemetry(effort, *, velocity=0.0):
    return {
        "effort": np.asarray(effort, dtype=float),
        "median_abs_velocity": np.full(6, velocity, dtype=float),
    }


def test_eighth_contact_probe_requires_confirmed_contact_on_both_arms():
    executor, controller, _safety = make_executor()
    controller.pose["left"] = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0, 0, 0, 1]),
    )
    controller.pose["right"] = (
        np.array([0.2, 0.0, 0.0]),
        np.array([0, 0, 0, 1]),
    )
    solver = ProbeSolver()
    executor.solver = solver
    executor.local_fk = {
        side: ProbeFK(side, solver) for side in ("left", "right")
    }
    samples = iter(
        [
            telemetry([0, 0, 0, 0, 0, 0]),
            telemetry([0, 3, 3, 0, 0, 0]),
            telemetry([0, 3, 3, 0, 0, 0]),
            telemetry([0, 0, 0, 0, 0, 0]),
            telemetry([0, 3, 3, 0, 0, 0]),
            telemetry([0, 3, 3, 0, 0, 0]),
        ]
    )
    executor._sample_median_joint_telemetry = lambda *_args, **_kwargs: next(
        samples
    )
    executor._completed = ["grasp"]

    executor.execute_eighth_contact_probe()
    executor.close_both_grippers()

    detections = [
        event["side"]
        for event in executor.trace
        if event["event"] == "eighth_contact_probe_detected"
    ]
    assert detections == ["right", "left"]
    assert controller.closed == 1
    assert controller.gripper_hold_enabled is True
    close_event = [
        event
        for event in executor.trace
        if event["event"] == "grippers_close_stabilized"
    ][-1]
    assert close_event["minimum_close_command_duration_s"] == pytest.approx(2.0)
    assert close_event["validated_contact_position"] is True
    assert executor._contact_probe_completed is True


def test_contact_indication_freezes_descent_until_velocity_settles():
    executor, controller, _safety = make_executor()
    controller.pose["left"] = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0, 0, 0, 1]),
    )
    controller.pose["right"] = (
        np.array([0.2, 0.0, 0.0]),
        np.array([0, 0, 0, 1]),
    )
    solver = ProbeSolver()
    executor.solver = solver
    executor.local_fk = {
        side: ProbeFK(side, solver) for side in ("left", "right")
    }
    samples = iter(
        [
            telemetry([0, 0, 0, 0, 0, 0]),
            telemetry([0, 3, 3, 0, 0, 0], velocity=0.06),
            telemetry([0, 3, 3, 0, 0, 0]),
            telemetry([0, 3, 3, 0, 0, 0]),
            telemetry([0, 0, 0, 0, 0, 0]),
            telemetry([0, 3, 3, 0, 0, 0]),
            telemetry([0, 3, 3, 0, 0, 0]),
        ]
    )
    executor._sample_median_joint_telemetry = lambda *_args, **_kwargs: next(
        samples
    )
    executor._completed = ["grasp"]

    executor.execute_eighth_contact_probe()

    frozen = [
        event
        for event in executor.trace
        if event["event"] == "eighth_contact_probe_descent_frozen"
    ]
    assert [event["side"] for event in frozen] == ["right", "left"]
    right_descent_steps = [
        context
        for context in controller.progress_steps
        if context == 1
    ]
    assert len(right_descent_steps) == 2
    assert executor._contact_probe_completed is True


def test_eighth_contact_probe_retracts_and_blocks_close_without_contact():
    executor, controller, _safety = make_executor()
    solver = ProbeSolver()
    executor.solver = solver
    executor.local_fk = {
        side: ProbeFK(side, solver) for side in ("left", "right")
    }
    executor._sample_median_joint_telemetry = (
        lambda *_args, **_kwargs: telemetry([0, 0, 0, 0, 0, 0])
    )
    executor._completed = ["grasp"]

    with pytest.raises(RuntimeError, match="not detected within 40.0mm"):
        executor.execute_eighth_contact_probe()
    with pytest.raises(RuntimeError, match="cannot close"):
        executor.close_both_grippers()

    assert controller.closed == 0
    assert controller.holds == 1
    assert any(
        event["event"] == "eighth_contact_probe_failure_retracted"
        for event in executor.trace
    )


def test_contact_probe_uses_fixed_probe_start_effort_reference():
    executor, _controller, _safety = make_executor()
    solver = ProbeSolver()
    executor.solver = solver
    executor.local_fk = {
        side: ProbeFK(side, solver) for side in ("left", "right")
    }
    executor.contact_probe_policy["maximum_descent_m"] = 0.001
    samples = iter(
        [
            telemetry([0, 0, 0, 0, 0, 0]),
            telemetry([0, 0.3, 0.3, 0, 0, 0]),
            telemetry([0, 0.6, 0.6, 0, 0, 0]),
        ]
    )
    executor._sample_median_joint_telemetry = lambda *_args, **_kwargs: next(
        samples
    )
    executor._completed = ["grasp"]

    with pytest.raises(RuntimeError, match="not detected within 1.0mm"):
        executor.execute_eighth_contact_probe()

    probe_samples = [
        event
        for event in executor.trace
        if event["event"] == "eighth_contact_probe_sample"
    ]
    assert [
        event["joint_residual_score_nm"] for event in probe_samples
    ] == pytest.approx([0.6, 1.2])
    assert not any(event["contact_indication"] for event in probe_samples)
    assert all(
        event["effort_residual_reference"] == "probe_start_fixed"
        for event in probe_samples
    )
