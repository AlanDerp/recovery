import numpy as np
import pytest

from capx.integrations.r1lite.bimanual_towel_executor import (
    BimanualTowelExecutor,
)


class FakeController:
    def __init__(self):
        self.joints = {"left": np.zeros(6), "right": np.zeros(6)}
        self.pose = {
            "left": (np.array([0.0, 0.0, 0.0]), np.array([0, 0, 0, 1])),
            "right": (np.array([0.0, 0.0, 0.0]), np.array([0, 0, 0, 1])),
        }
        self.moves = 0
        self.holds = 0
        self.grippers = []
        self.motion_progress_steps = []

    def get_feedback_pair(self):
        return {side: value.copy() for side, value in self.joints.items()}

    def wait_until_ready(self, **_kwargs):
        return None

    def refresh_feedback(self, timeout_s=0.02):
        return None

    def execute_joint_motion(self, left, right, **kwargs):
        self.motion_progress_steps.append(kwargs.get("progress_steps"))
        self.joints = {"left": np.asarray(left), "right": np.asarray(right)}
        self.moves += 1
        if self.moves == 1:
            self.pose["left"] = (np.array([0.07, 0.0, 0.0]), np.array([0, 0, 0, 1]))
            self.pose["right"] = (np.array([0.07, 0.0, 0.0]), np.array([0, 0, 0, 1]))
        else:
            self.pose["left"] = (np.array([0.10, 0.0, 0.0]), np.array([0, 0, 0, 1]))
            self.pose["right"] = (np.array([0.10, 0.0, 0.0]), np.array([0, 0, 0, 1]))

    def get_tool_pose_pair(self):
        return self.pose

    def hold(self):
        self.holds += 1

    def open_both_grippers(self):
        self.grippers.append("open")

    def close_both_grippers(self):
        self.grippers.append("close")

    def open_gripper(self, side):
        self.grippers.append(("open", side))

    def close_gripper(self, side):
        self.grippers.append(("close", side))

    def execute_single_arm_motion(
        self,
        *,
        side,
        target,
        inactive_hold,
        **kwargs,
    ):
        inactive = "right" if side == "left" else "left"
        values = {side: target, inactive: inactive_hold}
        self.execute_joint_motion(
            values["left"],
            values["right"],
            **kwargs,
        )


class FakeSolver:
    def solve_candidates(self, **kwargs):
        reference = np.asarray(kwargs["reference_joints_rad"])
        return [reference + 0.01]


class FakeSafety:
    def __init__(self):
        self.segments = 0

    def validate_interpolated_segment(self, *args, **kwargs):
        self.segments += 1
        return []


def test_p7_closed_gripper_lift_skips_table_collision_gate():
    safety = FakeSafety()
    executor = object.__new__(BimanualTowelExecutor)
    executor._is_eighth = False
    executor._completed = ["grasp"]
    executor._closed_gripper_sides = {"right"}
    executor.plan = {
        "phase_side": "right",
        "safety": {
            "post_grasp_closed_lift_collision_policy": (
                "skip_table_collision_if_gripper_close_succeeded"
            )
        },
    }
    executor.trace = []
    executor.safety = safety
    executor.torso = np.zeros(3)

    executor._validate_pair_segment(
        {"left": np.zeros(6), "right": np.zeros(6)},
        {"left": np.zeros(6), "right": np.ones(6)},
        stage_id="lift",
    )

    assert safety.segments == 0
    assert executor.trace[-1] == {
        "event": "collision_check_skipped",
        "pipeline": 7,
        "stage_id": "lift",
        "reason": "gripper_close_succeeded",
    }


def test_p7_close_gripper_waits_one_second_after_grasp(monkeypatch):
    sleeps = []
    executor = object.__new__(BimanualTowelExecutor)
    executor._bound = True
    executor._completed = ["grasp"]
    executor._closed_gripper_sides = set()
    executor.controller = FakeController()
    executor.plan = {"safety": {"post_grasp_close_dwell_s": 1.0}}
    executor.trace = []
    monkeypatch.setattr(
        "capx.integrations.r1lite.bimanual_towel_executor.time.sleep",
        sleeps.append,
    )

    executor.close_gripper(side="right")

    assert sleeps == [1.0]
    assert executor.trace == [
        {"event": "gripper_closed", "side": "right"},
        {
            "event": "post_grasp_close_dwell_completed",
            "side": "right",
            "duration_s": 1.0,
        },
    ]


def test_grasp_effort_contact_accepts_calibrated_settled_residual():
    class EffortController:
        def refresh_feedback(self, _timeout):
            return None

        def get_joint_telemetry(self, _side):
            return {
                "effort": [0.0, 3.2, 3.0, 0.0, 0.0, 0.0],
                "velocity": [0.0] * 6,
            }

    executor = object.__new__(BimanualTowelExecutor)
    executor.controller = EffortController()
    executor.effort_contact_policy = {
        "joint_indices_zero_based": [1, 2],
        "sum_abs_residual_threshold_nm": 6.0,
        "sample_count": 40,
        "maximum_median_abs_velocity_rad_s": 0.05,
        "maximum_tcp_z_above_target_m": 0.005,
    }

    detected, evidence = executor._grasp_effort_contact_detected(
        side="right",
        reference_effort=np.zeros(6),
        error_components={"z_signed_m": 0.001},
    )

    assert detected is True
    assert evidence["joint_residual_score_nm"] == pytest.approx(6.2)


def test_grasp_effort_contact_motion_monitor_uses_replay_threshold_without_velocity_gate():
    class EffortController:
        def refresh_feedback(self, _timeout):
            return None

        def get_joint_telemetry(self, _side):
            return {
                "effort": [0.0, 1.3, 1.3, 0.0, 0.0, 0.0],
                "velocity": [0.4] * 6,
            }

    executor = object.__new__(BimanualTowelExecutor)
    executor.controller = EffortController()
    executor.effort_contact_policy = {
        "joint_indices_zero_based": [1, 2],
        "sum_abs_residual_threshold_nm": 2.5,
        "sample_count": 40,
        "maximum_median_abs_velocity_rad_s": 0.05,
        "maximum_tcp_z_above_target_m": 0.005,
    }

    detected, evidence = executor._grasp_effort_contact_detected(
        side="right",
        reference_effort=np.zeros(6),
        error_components={"z_signed_m": 0.020},
        sample_count=5,
        residual_threshold_nm=2.5,
        require_settled_velocity=False,
        require_near_table=False,
    )

    assert detected is True
    assert evidence["joint_residual_score_nm"] == pytest.approx(2.6)
    assert evidence["settled"] is False
    assert evidence["settled_velocity_required"] is False
    assert evidence["near_table_required"] is False


@pytest.mark.parametrize(
    (
        "contact_step",
        "expected_steps",
        "authorization",
        "actual_step_m",
        "selector_available",
        "should_raise",
        "expected_refinements",
    ),
    [
        (1, 1, "effort_contact", 0.0005, True, False, 0),
        (3, 3, "effort_contact", 0.0005, True, False, 2),
        (
            None,
            21,
            "absolute_height_refinement_completed_without_contact",
            0.0005,
            True,
            False,
            20,
        ),
        (
            None,
            0,
            "descent_condition_unavailable",
            0.0005,
            False,
            False,
            0,
        ),
    ],
)
def test_p7_post_arrival_probe_prefers_contact_with_close_fallbacks(
    contact_step,
    expected_steps,
    authorization,
    actual_step_m,
    selector_available,
    should_raise,
    expected_refinements,
):
    class ProbeController:
        def __init__(self):
            self.joints = {"left": np.zeros(6), "right": np.zeros(6)}
            self.position = np.array([0.5, -0.2, 0.9])
            self.moves = 0
            self.holds = 0

        def get_feedback_pair(self):
            return {side: value.copy() for side, value in self.joints.items()}

        def execute_single_arm_motion(
            self, *, side, target, inactive_hold, **kwargs
        ):
            self.joints[side] = np.asarray(target)
            self.joints["left" if side == "right" else "right"] = np.asarray(
                inactive_hold
            )
            self.position[2] -= actual_step_m
            self.moves += 1
            stop_requested = kwargs.get("stop_requested")
            return bool(stop_requested is not None and stop_requested())

        def hold(self):
            self.holds += 1

    controller = ProbeController()
    executor = object.__new__(BimanualTowelExecutor)
    executor.controller = controller
    executor.solver = FakeSolver()
    executor.trace = []
    executor.policy = {
        "ik_probes_per_iteration": 3,
    }
    executor.post_arrival_probe_policy = {
        "direction_base": [0.0, 0.0, -1.0],
        "initial_descent_m": 0.010,
        "absolute_height_refinement_iterations": 20,
        "absolute_contact_tcp_z_base_m": 0.88,
        "absolute_contact_z_tolerance_m": 0.0001,
    }
    executor.stages = {
        "grasp": {
            "right": {
                "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "target_position_base_m": [0.5, -0.2, 0.89],
            }
        }
    }
    executor._sample_median_joint_telemetry = lambda _side: {
        "effort": np.zeros(6),
        "median_abs_velocity": np.zeros(6),
    }
    executor._get_fresh_tool_pose = lambda _side: (
        controller.position.copy(),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    def select(**kwargs):
        if not selector_available:
            raise RuntimeError(
                "no single-arm IK candidate passed collision gates"
            )
        return np.asarray(kwargs["candidates"][0])

    executor._select_single_arm_candidate = select
    executor._record_actual_tcp = lambda **_kwargs: None

    def detect(**_kwargs):
        detected = (
            contact_step is not None and controller.moves >= contact_step
        )
        return detected, {"available": True, "detected": detected}

    executor._grasp_effort_contact_detected = detect
    if should_raise:
        with pytest.raises(
            RuntimeError,
            match="stable table contact was not confirmed",
        ):
            executor._execute_p7_post_arrival_grasp_probe(
                side="right",
                inactive="left",
                inactive_hold=np.zeros(6),
            )
        result = executor.trace[-1]
    else:
        result = executor._execute_p7_post_arrival_grasp_probe(
            side="right",
            inactive="left",
            inactive_hold=np.zeros(6),
        )

    assert controller.moves == expected_steps
    assert result["close_authorized_by"] == authorization
    assert result["actual_descent_m"] == pytest.approx(
        expected_steps * actual_step_m
    )
    assert result["completed_descent_commands"] == expected_steps
    assert result["completed_refinement_commands"] == expected_refinements
    assert controller.holds == (1 if contact_step is not None else 0)
    if expected_steps:
        initial_probe = next(
            item
            for item in executor.trace
            if item.get("event") == "p7_post_arrival_grasp_probe"
            and item.get("phase") == "initial_10mm_descent"
        )
        assert initial_probe["continuous_contact_monitoring"] is True


def test_p7_probe_executes_first_solvable_candidate_descent_only():
    class ProbeController:
        def __init__(self):
            self.joints = {"left": np.zeros(6), "right": np.zeros(6)}
            self.position = np.array([0.5, -0.2, 0.9])
            self.moves = 0
            self.holds = 0

        def get_feedback_pair(self):
            return {side: value.copy() for side, value in self.joints.items()}

        def execute_single_arm_motion(
            self, *, side, target, inactive_hold, **kwargs
        ):
            self.joints[side] = np.asarray(target)
            self.position[2] -= 0.0005
            self.moves += 1
            stop_requested = kwargs.get("stop_requested")
            return bool(stop_requested is not None and stop_requested())

        def hold(self):
            self.holds += 1

    controller = ProbeController()
    executor = object.__new__(BimanualTowelExecutor)
    executor.controller = controller
    executor.solver = FakeSolver()
    executor.trace = []
    executor.policy = {"ik_probes_per_iteration": 3}
    executor.post_arrival_probe_policy = {
        "direction_base": [0.0, 0.0, -1.0],
        "initial_descent_m": 0.010,
        "absolute_height_refinement_iterations": 20,
        "absolute_contact_tcp_z_base_m": 0.88,
        "absolute_contact_z_tolerance_m": 0.0001,
    }
    executor._p7_candidate_descents_m = (
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.010,
    )
    executor.stages = {
        "grasp": {
            "right": {
                "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "target_position_base_m": [0.5, -0.2, 0.89],
            }
        }
    }
    executor._sample_median_joint_telemetry = lambda _side: {
        "effort": np.zeros(6),
        "median_abs_velocity": np.zeros(6),
    }
    executor._get_fresh_tool_pose = lambda _side: (
        controller.position.copy(),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    attempted_candidate_z = []
    select_calls = 0

    def select(**kwargs):
        nonlocal select_calls
        select_calls += 1
        target_z = float(kwargs["target_position_base_m"][2])
        if select_calls > 1:
            attempted_candidate_z.append(target_z)
            if len(attempted_candidate_z) < 3:
                raise RuntimeError(
                    "no single-arm IK candidate passed collision gates"
                )
        return np.asarray(kwargs["candidates"][0])

    executor._select_single_arm_candidate = select
    executor._record_actual_tcp = lambda **_kwargs: None
    executor._grasp_effort_contact_detected = lambda **_kwargs: (
        controller.moves >= 2,
        {"available": True, "detected": controller.moves >= 2},
    )

    result = executor._execute_p7_post_arrival_grasp_probe(
        side="right",
        inactive="left",
        inactive_hold=np.zeros(6),
    )

    assert attempted_candidate_z == pytest.approx([0.8945, 0.8935, 0.8925])
    assert controller.moves == 2
    assert controller.holds == 1
    assert result["completed_refinement_commands"] == 1
    assert result["candidate_descents_m"] == pytest.approx(
        [0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
    )


def test_post_close_tcp_z_segment_rejects_downward_fk_motion():
    class FK:
        def fk(self, joints, _torso):
            return type(
                "Pose",
                (),
                {"position_base_m": np.array([0.0, 0.0, float(joints[0])])},
            )()

    executor = object.__new__(BimanualTowelExecutor)
    executor._post_close_z_policy = {
        "enabled": True,
        "fk_samples_per_segment": 100,
        "planned_decrease_tolerance_m": 0.0002,
    }
    executor._closed_gripper_sides = {"right"}
    executor.local_fk = {"right": FK()}
    executor.torso = np.zeros(3)
    executor.trace = []

    with pytest.raises(RuntimeError, match="monotonic gate failed"):
        executor._validate_post_close_tcp_z_segment(
            side="right",
            start_joints=np.array([1.0, 0, 0, 0, 0, 0]),
            target_joints=np.array([0.99, 0, 0, 0, 0, 0]),
            stage_id="place_center",
            phase="certified_waypoint",
            sample_index=1,
        )
    assert executor.trace[-1]["passed"] is False


def test_post_close_horizontal_place_corridor_allows_small_z_variation():
    class FK:
        def fk(self, joints, _torso):
            return type(
                "Pose",
                (),
                {"position_base_m": np.array([0.0, 0.0, float(joints[0])])},
            )()

    executor = object.__new__(BimanualTowelExecutor)
    executor._post_close_z_policy = {
        "enabled": True,
        "fk_samples_per_segment": 100,
        "planned_decrease_tolerance_m": 0.0002,
    }
    executor._p7_horizontal_place_z_tolerance_m = 0.002
    executor._horizontal_place_planned_z_reference = None
    executor._closed_gripper_sides = {"right"}
    executor.local_fk = {"right": FK()}
    executor.torso = np.zeros(3)
    executor.trace = []

    executor._validate_post_close_tcp_z_segment(
        side="right",
        start_joints=np.array([1.0, 0, 0, 0, 0, 0]),
        target_joints=np.array([0.9993, 0, 0, 0, 0, 0]),
        stage_id="place_center",
        phase="certified_waypoint",
        sample_index=2,
    )

    evidence = executor.trace[-1]
    assert evidence["passed"] is True
    assert evidence["horizontal_corridor_enabled"] is True
    assert evidence["maximum_horizontal_error_m"] == pytest.approx(0.0007)


def test_post_close_place_z_guard_can_be_explicitly_disabled():
    executor = object.__new__(BimanualTowelExecutor)
    executor._post_close_z_policy = {"enabled": True}
    executor._closed_gripper_sides = {"right"}
    executor._p7_disable_place_z_guard = True
    executor.trace = []

    executor._validate_post_close_tcp_z_segment(
        side="right",
        start_joints=np.ones(6),
        target_joints=np.zeros(6),
        stage_id="place_center",
        phase="certified_waypoint",
        sample_index=2,
    )

    assert executor.trace[-1]["event"] == (
        "post_close_tcp_z_segment_check_skipped"
    )


def test_post_close_tcp_z_feedback_drop_holds_and_fails():
    class Controller:
        def __init__(self):
            self.holds = 0

        def hold(self):
            self.holds += 1

    executor = object.__new__(BimanualTowelExecutor)
    executor.controller = Controller()
    executor._post_close_z_policy = {
        "enabled": True,
        "measured_decrease_tolerance_m": 0.002,
    }
    executor._closed_gripper_sides = {"right"}
    executor._post_close_tcp_z = {"right": 1.0}
    executor._get_fresh_tool_pose = lambda _side: (
        np.array([0.0, 0.0, 0.997]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    executor.trace = []

    with pytest.raises(RuntimeError, match="feedback decreased"):
        executor._verify_post_close_tcp_z(
            side="right",
            stage_id="lift",
            phase="certified_waypoint",
            sample_index=1,
        )
    assert executor.controller.holds == 1


def test_p7_initial_descent_skips_but_refinement_checks_collision_gate():
    class RejectingSafety:
        def validate_interpolated_segment(self, *args, **kwargs):
            raise ValueError("collision")

    class FK:
        def fk(self, _joints, _torso):
            return type("Pose", (), {"position_base_m": np.zeros(3)})()

    executor = object.__new__(BimanualTowelExecutor)
    executor.policy = {
        "max_incremental_joint_delta_rad": 1.0,
        "grasp_xy_tolerance_m": 0.005,
        "grasp_z_tolerance_m": 0.0005,
    }
    executor.post_arrival_probe_policy = {
        "maximum_xy_shift_per_descent_m": 0.010,
        "forbid_upward_tcp_motion": True,
    }
    executor.local_fk = {"right": FK()}
    executor.torso = np.zeros(3)
    executor.safety = RejectingSafety()
    executor._is_eighth = False
    executor._collision_checks_enabled = True

    selected = executor._select_single_arm_candidate(
        side="right",
        inactive="left",
        stage_id="grasp_contact_probe",
        current={"left": np.zeros(6), "right": np.zeros(6)},
        inactive_hold=np.zeros(6),
        candidates=[np.full(6, 0.01)],
        target_position_base_m=np.zeros(3),
        probe_reference_position_base_m=[0.0, 0.0, 0.001],
        probe_skip_collision=True,
    )

    assert selected == pytest.approx(np.full(6, 0.01))
    def reject_collision(*_args, **_kwargs):
        raise ValueError("collision")

    executor._validate_pair_segment = reject_collision
    with pytest.raises(
        RuntimeError,
        match="no single-arm IK candidate passed collision gates",
    ):
        executor._select_single_arm_candidate(
            side="right",
            inactive="left",
            stage_id="grasp_contact_probe",
            current={"left": np.zeros(6), "right": np.zeros(6)},
            inactive_hold=np.zeros(6),
            candidates=[np.full(6, 0.01)],
            target_position_base_m=np.zeros(3),
            probe_reference_position_base_m=[0.0, 0.0, 0.001],
            probe_skip_collision=False,
        )


def test_stage_uses_feedback_refinement_and_rechecks_swept_collision():
    stages = []
    plan_stages = []
    for stage_id in ("pregrasp", "grasp", "lift", "fold_center"):
        stages.append(
            {
                "id": stage_id,
                "left": {
                    "joints_rad": [0.01] * 6,
                    "joint_waypoints_rad": [[0.01] * 6],
                    "target_position_base_m": [0.1, 0.0, 0.0],
                    "target_quaternion_xyzw": [0, 0, 0, 1],
                    "fk_position_base_m": [0.1, 0.0, 0.0],
                    "fk_quaternion_xyzw": [0, 0, 0, 1],
                },
                "right": {
                    "joints_rad": [-0.01] * 6,
                    "joint_waypoints_rad": [[-0.01] * 6],
                    "target_position_base_m": [0.1, 0.0, 0.0],
                    "target_quaternion_xyzw": [0, 0, 0, 1],
                    "fk_position_base_m": [0.1, 0.0, 0.0],
                    "fk_quaternion_xyzw": [0, 0, 0, 1],
                },
            }
        )
        plan_stages.append(
            {
                "id": stage_id,
                "left_tcp_position_base_m": [0.1, 0.0, 0.0],
                "right_tcp_position_base_m": [0.1, 0.0, 0.0],
            }
        )
    plan = {
        "stages": plan_stages,
        "safety": {
            "cartesian_refinement_policy": {
                "target_tolerance_m": 0.015,
                "grasp_xy_tolerance_m": 0.010,
                "grasp_z_tolerance_m": 0.002,
                "max_iterations": 20,
                "ik_probes_per_iteration": 3,
                "minimum_improvement_m": 0.001,
                "local_optimum_stall_iterations": 2,
                "max_incremental_joint_delta_rad": 1.0,
            }
        },
    }
    bundle = {
        "artifact_bindings": {"plan_sha256": "approved"},
        "initial_state": {"torso_joints_rad": [0.0, 0.0, 0.0]},
        "stages": stages,
    }
    controller = FakeController()
    safety = FakeSafety()
    executor = BimanualTowelExecutor(
        plan=plan,
        plan_sha256="approved",
        bundle=bundle,
        controller=controller,
        solver=FakeSolver(),
        safety=safety,
    )
    executor.bind_plan(sha256="approved")
    executor.execute_bimanual_stage(stage_id="pregrasp")

    assert controller.moves == 2
    assert safety.segments == 2
    assert controller.holds == 0
    assert executor.trace[-1]["event"] == "stage_completed"


def test_single_arm_stage_holds_inactive_arm_and_uses_active_cartesian_gate():
    stage = {
        "id": "pregrasp",
        "left": {
            "joints_rad": [0.0] * 6,
            "joint_waypoints_rad": [[0.0] * 6],
            "target_position_base_m": [0.0, 0.0, 0.0],
            "target_quaternion_xyzw": [0, 0, 0, 1],
            "fk_position_base_m": [0.0, 0.0, 0.0],
            "fk_quaternion_xyzw": [0, 0, 0, 1],
        },
        "right": {
            "joints_rad": [0.01] * 6,
            "joint_waypoints_rad": [[0.01] * 6],
            "target_position_base_m": [0.07, 0.0, 0.0],
            "target_quaternion_xyzw": [0, 0, 0, 1],
            "fk_position_base_m": [0.07, 0.0, 0.0],
            "fk_quaternion_xyzw": [0, 0, 0, 1],
        },
    }
    plan = {
        "stages": [
            {
                "id": "pregrasp",
                "active_side": "right",
                "active_tcp_position_base_m": [0.07, 0.0, 0.0],
            }
        ],
        "safety": {
            "home_joint_tolerance_rad": 0.01,
            "cartesian_refinement_policy": {
                "target_tolerance_m": 0.015,
                    "grasp_xy_tolerance_m": 0.010,
                "grasp_z_tolerance_m": 0.002,
                "max_iterations": 20,
                "ik_probes_per_iteration": 3,
                "minimum_improvement_m": 0.001,
                "local_optimum_stall_iterations": 2,
                "max_incremental_joint_delta_rad": 1.0,
            },
        },
    }
    bundle = {
        "artifact_bindings": {"plan_sha256": "approved"},
        "initial_state": {
            "left_joints_rad": [0.0] * 6,
            "right_joints_rad": [0.0] * 6,
            "torso_joints_rad": [0.0] * 3,
        },
        "stages": [stage],
    }
    controller = FakeController()
    executor = BimanualTowelExecutor(
        plan=plan,
        plan_sha256="approved",
        bundle=bundle,
        controller=controller,
        solver=FakeSolver(),
        safety=FakeSafety(),
    )
    executor.bind_plan(sha256="approved")
    executor.execute_single_arm_stage(side="right", stage_id="pregrasp")

    assert controller.joints["left"] == pytest.approx(np.zeros(6))
    assert controller.motion_progress_steps == [1]
    assert executor.trace[-1]["event"] == "single_arm_stage_completed"
    assert executor._single_arm_arrived(
        "grasp",
        {"norm_m": 0.010, "xy_m": 0.0099, "z_m": 0.0019},
    )
    assert not executor._single_arm_arrived(
        "grasp",
        {"norm_m": 0.011, "xy_m": 0.0101, "z_m": 0.001},
    )
    assert not executor._single_arm_arrived(
        "grasp",
        {"norm_m": 0.003, "xy_m": 0.001, "z_m": 0.0021},
    )
    executor.post_arrival_probe_policy = {
        "enabled": True,
        "maximum_entry_z_error_m": 0.005,
    }
    assert executor._single_arm_arrived(
        "grasp",
        {"norm_m": 0.010, "xy_m": 0.0099, "z_m": 0.0049},
    )
    assert not executor._single_arm_arrived(
        "grasp",
        {"norm_m": 0.010, "xy_m": 0.0099, "z_m": 0.005},
    )
