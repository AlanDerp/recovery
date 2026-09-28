from __future__ import annotations

from collections import deque
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from capx.envs.simulators import r1lite_real
from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel


def _fake_controller(*, response_sign: float = 1.0) -> R1LiteRealLowLevel:
    controller = object.__new__(R1LiteRealLowLevel)
    controller.safe_joint_home = np.array(
        [-0.005, 1.063, -1.541, 0.088, 0.003, 0.002], dtype=float
    )
    controller.safe_joint_home_feedback = np.array(
        [0.012766, 1.347447, -1.485500, 0.116383, 0.028511, 0.023830],
        dtype=float,
    )
    controller.adaptive_joint_tolerance_rad = 0.001
    controller.adaptive_correction_fraction = 0.15
    controller.adaptive_probe_step_rad = 0.006
    controller.adaptive_probe_steps_rad = np.array(
        [0.006, 0.006, 0.010, 0.006, 0.006, 0.006], dtype=float
    )
    controller.adaptive_max_command_excursion_rad = 0.02
    controller.adaptive_max_target_step_rad = 0.02
    controller.adaptive_max_measured_step_rad = 0.04
    controller.adaptive_settle_s = 0.01
    controller.adaptive_max_iterations = 36
    controller.adaptive_max_stalled_iterations = 14
    controller.adaptive_min_progress_rad = 0.0002
    controller.arm_min_motion_rad = 0.001
    controller.r1lite_motion_validation_enabled = True
    controller._lock = threading.Lock()
    controller._last_joint_command = controller.safe_joint_home.copy()
    controller._last_action = {}
    controller._test_measured = controller.safe_joint_home_feedback.copy()
    controller._test_records = []
    controller._test_publish_count = 0

    def get_arm_joints() -> np.ndarray:
        return controller._test_measured.copy()

    def publish(command: np.ndarray, _duration_s: float, **_kwargs) -> np.ndarray:
        controller._test_publish_count += 1
        command_delta = np.asarray(command) - controller._last_joint_command
        controller._test_measured += response_sign * command_delta
        controller._last_joint_command = np.asarray(command).copy()
        return controller._test_measured.copy()

    controller.get_arm_joints = get_arm_joints
    controller._publish_joint_command_for = publish
    controller._append_adaptive_record = controller._test_records.append
    return controller


def test_adaptive_joint_step_learns_command_feedback_offset() -> None:
    controller = _fake_controller()
    desired = controller.get_arm_joints()
    desired[1] -= 0.015

    result = controller.move_joints_adaptive(desired, context="unit_converge")

    assert result["success"] is True
    assert result["max_error_rad"] <= controller.adaptive_joint_tolerance_rad
    assert controller._test_records
    assert np.allclose(controller.get_arm_joints(), desired, atol=0.001)
    assert not np.allclose(result["final_command"], desired)


def test_small_j2_residual_at_lower_limit_counts_as_saturated() -> None:
    controller = _fake_controller()
    controller.arm_joint_tolerance_rad = 0.025
    controller.right_arm_j2_command_lower_limit_rad = 0.01745329252
    desired = controller.get_arm_joints()
    measured = desired.copy()
    measured[1] += 0.017
    command = controller._last_joint_command.copy()
    command[1] = controller.right_arm_j2_command_lower_limit_rad

    effective, saturated = controller._adaptive_effective_error(
        desired,
        measured,
        command,
    )

    assert effective[1] == 0.0
    assert saturated == [1]


def test_large_j2_residual_at_lower_limit_is_not_ignored() -> None:
    controller = _fake_controller()
    controller.arm_joint_tolerance_rad = 0.025
    controller.right_arm_j2_command_lower_limit_rad = 0.01745329252
    desired = controller.get_arm_joints()
    measured = desired.copy()
    measured[1] += 0.03
    command = controller._last_joint_command.copy()
    command[1] = controller.right_arm_j2_command_lower_limit_rad

    effective, saturated = controller._adaptive_effective_error(
        desired,
        measured,
        command,
    )

    assert effective[1] == pytest.approx(-0.03)
    assert saturated == []


def test_adaptive_joint_step_rejects_opposite_motion() -> None:
    controller = _fake_controller(response_sign=-1.0)
    desired = controller.get_arm_joints()
    desired[1] -= 0.015

    with pytest.raises(RuntimeError, match="moved away"):
        controller.move_joints_adaptive(desired, context="unit_reverse")


def test_large_feedback_jump_latches_fault_without_recovery_publish() -> None:
    controller = _fake_controller()

    def jump_feedback(
        command: np.ndarray,
        _duration_s: float,
        **_kwargs,
    ) -> np.ndarray:
        controller._test_publish_count += 1
        controller._last_joint_command = np.asarray(command).copy()
        controller._test_measured[1] -= 0.4428
        return controller._test_measured.copy()

    controller._publish_joint_command_for = jump_feedback
    desired = controller.get_arm_joints()
    desired[1] += 0.015

    with pytest.raises(RuntimeError, match="measured step"):
        controller.move_joints_adaptive(desired, context="unit_j2_jump")

    assert controller._test_publish_count == 1
    assert controller._motion_fault_latched is True
    assert controller.motion_fault_record()["reason"] == (
        "measured joint step exceeded safety limit"
    )

    with pytest.raises(RuntimeError, match="motion fault is latched"):
        controller.move_joints_adaptive(desired, context="unit_after_fault")
    assert controller._test_publish_count == 1


def test_configured_j2_feedback_and_command_limits_are_enforced() -> None:
    controller = _fake_controller()
    controller.right_arm_j2_feedback_upper_limit_rad = 2.8
    controller.right_arm_j2_command_lower_limit_rad = 0.01745329252
    controller.right_arm_j2_command_upper_limit_rad = 2.8

    feedback_target = controller.get_arm_joints()
    feedback_target[1] = 2.801
    with pytest.raises(ValueError, match="J2 feedback target"):
        controller.validate_arm_joint_feedback_target(
            feedback_target,
            context="unit_j2_feedback_limit",
        )

    command_target = controller._last_joint_command.copy()
    command_target[1] = 2.801
    with pytest.raises(ValueError, match="J2 MOCA command"):
        controller.validate_moca_joint_command(
            command_target,
            context="unit_j2_command_limit",
        )
    below_lower = controller._last_joint_command.copy()
    below_lower[1] = -0.02
    clamped = controller.validate_moca_joint_command(
        below_lower,
        context="unit_j2_command_lower_limit",
    )
    assert clamped[1] == pytest.approx(0.01745329252)


def test_moca_interpolator_output_corridor_latches_fault() -> None:
    controller = _fake_controller()
    controller._lock = threading.Lock()
    controller._active_moca_command_corridor = (
        7,
        np.array([-1.0, 1.39, -1.0, -2.0, -1.0, -1.0]),
        np.array([1.0, 1.45, 1.0, 1.0, 1.0, 1.0]),
        np.array([0.55, 1.4275, 0.0, -1.32, -0.53, 0.15]),
        np.array([0.55, 1.4215, 0.0, -1.32, -0.53, 0.15]),
    )
    controller._moca_output_fault_thread = None
    stopped = threading.Event()
    controller._run_moca_stop_command = stopped.set
    controller._wait_for_moca_to_stop = lambda: None
    controller._release_moca_lock = lambda: None
    controller._moca_started_by_env = True
    controller._moca_owned_by_env = True
    controller._pending_gripper_target = 100.0

    controller._on_arm_control(
        SimpleNamespace(
            p_des=[0.55, -0.6399, 0.0, -1.32, -0.53, 0.15],
        )
    )

    assert stopped.wait(timeout=1.0)
    assert controller._motion_fault_latched is True
    fault = controller.motion_fault_record()
    assert fault["reason"] == "MOCA interpolated command left safety corridor"
    assert fault["details"]["active_joint"] == 1
    assert fault["details"]["violation_rad"] > 2.0
    assert controller._active_moca_command_corridor is None
    assert controller._pending_gripper_target is None


def test_moca_startup_stability_rejects_cross_joint_drift() -> None:
    controller = object.__new__(R1LiteRealLowLevel)
    controller._lock = threading.Lock()
    reference = np.zeros(6)
    drifted = reference.copy()
    drifted[1] = 0.0917
    now = r1lite_real.time.monotonic()
    controller._arm_feedback_history = deque(
        [
            (1, now, reference.copy()),
            (2, now + 0.005, drifted),
        ],
        maxlen=512,
    )
    controller.r1lite_motion_validation_enabled = True
    controller.adaptive_feedback_settle_timeout_s = 1.0
    controller.adaptive_feedback_stability_s = 0.3
    controller.adaptive_feedback_stability_rad = 0.003
    controller.adaptive_feedback_min_samples = 4
    controller.adaptive_max_measured_step_rad = 0.06
    controller.publish_rate_hz = 20.0
    controller._assert_motion_ready = lambda: None
    faults = []
    controller._enter_motion_fault = lambda **kwargs: faults.append(kwargs)

    with pytest.raises(RuntimeError, match="before MOCA startup stabilized"):
        controller._wait_for_moca_startup_stability(
            reference=reference,
            after_sequence=0,
            maximum_drift_rad=0.06,
        )

    assert faults
    assert faults[0]["reason"] == (
        "right arm moved before MOCA startup stabilized"
    )


def test_motion_validation_disabled_skips_adaptive_safety_aborts() -> None:
    controller = _fake_controller(response_sign=-1.0)
    controller.r1lite_motion_validation_enabled = False
    desired = controller.get_arm_joints()
    desired[1] -= 0.015

    result = controller.move_joints_adaptive(desired, context="unit_no_validation")

    assert result["success"] is True
    assert result["motion_validation_enabled"] is False


def test_adaptive_correction_updates_multiple_joints_by_fraction() -> None:
    controller = _fake_controller()
    desired = controller.get_arm_joints()
    desired[1] -= 0.015
    desired[2] += 0.015

    result = controller.move_joints_adaptive(desired, context="unit_joint3")

    first_delta = np.asarray(result["history"][0]["command_delta"])
    assert first_delta[1] == pytest.approx(-0.00225)
    assert first_delta[2] == pytest.approx(0.00225)
    assert np.count_nonzero(first_delta) == 2
    assert result["success"] is True


def test_stalled_joint_cannot_accumulate_unbounded_command() -> None:
    controller = _fake_controller(response_sign=0.0)
    desired = controller.get_arm_joints()
    desired[2] += 0.015

    with pytest.raises(RuntimeError, match="no measurable progress"):
        controller.move_joints_adaptive(desired, context="unit_stalled_joint3")


def test_adaptive_joint_control_can_be_disabled() -> None:
    controller = _fake_controller()
    controller.r1lite_adaptive_joint_control_enabled = False
    controller.command_duration_s = 0.01
    desired = controller.get_arm_joints()
    desired[1] -= 0.015

    result = controller.move_joints_adaptive(desired, context="unit_feedforward")

    assert result["mode"] == "feedforward"
    assert result["adaptive_control_enabled"] is False
    assert controller._test_records == []


def test_adaptive_joint_path_can_be_disabled() -> None:
    controller = _fake_controller()
    controller.r1lite_adaptive_joint_control_enabled = False
    controller.max_joint_step = 0.03
    controller.command_duration_s = 0.01
    target = controller.get_arm_joints()
    target[1] -= 0.025

    result = controller.move_joint_waypoints_adaptive(
        target,
        waypoint_step_rad=0.015,
        context="unit_feedforward_path",
    )

    assert result["mode"] == "feedforward"
    assert result["adaptive_control_enabled"] is False
    assert result["waypoint_count"] >= 1
    assert all(
        waypoint["mode"] == "feedforward"
        for waypoint in result["waypoints"]
    )


def _feedback_wait_controller(
    feedback_values: list[np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> R1LiteRealLowLevel:
    controller = object.__new__(R1LiteRealLowLevel)
    controller._lock = threading.Lock()
    controller._arm_feedback_sequence = 0
    controller._arm_feedback_history = deque(maxlen=512)
    controller.publish_rate_hz = 20.0
    controller.adaptive_feedback_stability_s = 0.15
    controller.adaptive_feedback_stability_rad = 0.002
    controller.adaptive_feedback_min_samples = 3
    controller.adaptive_feedback_settle_timeout_s = 1.0
    controller._last_joint_command = None
    controller._pending_gripper_target = None
    controller._ensure_moca_for_joint_target = lambda _command: None
    controller._enable_arm_joint_control = lambda: None
    controller._test_publish_count = 0
    controller._test_published_commands = []
    clock = [0.0]

    class Publisher:
        def publish(self, _msg) -> None:
            controller._test_publish_count += 1
            controller._test_published_commands.append(np.asarray(_msg, dtype=float))

    def sleep(seconds: float) -> None:
        clock[0] += seconds
        if feedback_values:
            controller._arm_feedback_sequence += 1
            controller._arm_feedback_history.append(
                (
                    controller._arm_feedback_sequence,
                    clock[0],
                    np.asarray(feedback_values.pop(0), dtype=float),
                )
            )

    controller._arm_pub = Publisher()
    controller._make_joint_state = lambda _name, values: values
    controller.arm_name = "right_arm"
    monkeypatch.setattr(r1lite_real.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(r1lite_real.time, "sleep", sleep)
    return controller


def test_settled_feedback_ignores_transient_reverse_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transient = np.zeros(6)
    transient[2] = 0.04
    settled = []
    for offset in [0.0, 0.0004, -0.0003, 0.0002, 0.0, -0.0001]:
        sample = np.zeros(6)
        sample[2] = -0.01 + offset
        settled.append(sample)
    controller = _feedback_wait_controller(
        [transient, *settled],
        monkeypatch,
    )

    measured = controller._wait_for_settled_arm_feedback(
        after_sequence=0,
        timeout_s=1.0,
    )

    assert measured[2] == pytest.approx(-0.01, abs=0.0003)


def test_settled_feedback_requires_samples_newer_than_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _feedback_wait_controller([], monkeypatch)
    controller._arm_feedback_sequence = 4
    for sequence in range(1, 5):
        controller._arm_feedback_history.append(
            (sequence, -1.0 + sequence * 0.1, np.zeros(6))
        )

    with pytest.raises(TimeoutError, match="fresh stable arm feedback"):
        controller._wait_for_settled_arm_feedback(
            after_sequence=4,
            timeout_s=0.3,
        )


def test_joint_command_applies_j2_lower_limit_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settled = [np.zeros(6) for _ in range(8)]
    controller = _feedback_wait_controller(settled, monkeypatch)

    measured = controller._publish_joint_command_for(
        np.zeros(6),
        duration_s=0.1,
    )

    assert np.allclose(measured, np.zeros(6))
    assert controller._test_publish_count >= 1
    expected_command = np.zeros(6)
    expected_command[1] = 0.01745329252
    assert all(
        np.allclose(command, expected_command)
        for command in controller._test_published_commands
    )


def test_enable_seeds_preexisting_moca_with_calibrated_hold() -> None:
    controller = object.__new__(R1LiteRealLowLevel)
    measured = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    controller.safe_joint_home = np.array(
        [0.01, 0.22, -0.02, 0.01, 0.04, 0.01]
    )
    controller.safe_joint_home_feedback = np.array(
        [0.03, 0.45, -0.01, 0.03, 0.06, 0.03]
    )
    controller._lock = threading.Lock()
    controller._arm_feedback_sequence = 0
    controller.get_arm_joints = lambda: measured.copy()
    seeded = []

    class BrakePublisher:
        subscription_count = 1

        def get_subscription_count(self) -> int:
            return self.subscription_count

        def publish(self, _message) -> None:
            return None

    brake = BrakePublisher()
    controller._brake_mode_pub = brake

    def ensure_moca(command: np.ndarray) -> None:
        seeded.append(np.asarray(command, dtype=float))
        brake.subscription_count = 1

    controller._ensure_moca_for_joint_target = ensure_moca
    controller._wait_for_subscriber = lambda _publisher, _context: None
    controller._wait_for_moca_startup_stability = lambda **_kwargs: {}

    class Bool:
        data = False

    controller._Bool = Bool
    controller._enable_arm_joint_control(reset_function_frame=False)

    expected = (
        measured
        + controller.safe_joint_home
        - controller.safe_joint_home_feedback
    )
    expected[1] = 0.01745329252
    assert len(seeded) == 1
    assert np.allclose(seeded[0], expected)


def test_moca_session_restarts_unknown_instance_and_close_stops_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object.__new__(R1LiteRealLowLevel)
    controller._motion_fault_latched = False
    controller._motion_fault_record = None
    controller._moca_started_by_env = False
    controller._moca_owned_by_env = False
    controller._moca_lock_file = None
    controller._last_action = {}
    controller._pending_gripper_target = None
    controller._lock = threading.Lock()
    controller._moca_command_corridor_generation = 0
    controller._active_moca_command_corridor = None
    controller.lazy_moca_start_command = ("start-moca",)
    controller.lazy_moca_stop_command = ("stop-moca",)
    controller.feedback_timeout_s = 0.1
    controller._closed = False
    events: list[str] = []

    class Publisher:
        subscription_count = 1

        def get_subscription_count(self) -> int:
            return self.subscription_count

        def publish(self, _message) -> None:
            return None

    publisher = Publisher()
    controller._arm_pub = publisher
    controller.arm_name = "right_arm"
    controller.get_arm_joints = lambda: np.zeros(6)
    controller._make_joint_state = lambda _name, values: values
    controller._spin_once_or_sleep = lambda: None
    controller._acquire_moca_lock = lambda: events.append("lock")
    controller._release_moca_lock = lambda: events.append("unlock")

    def stop_moca() -> None:
        events.append("stop")
        publisher.subscription_count = 0

    controller._run_moca_stop_command = stop_moca

    def run(command, **_kwargs):
        assert tuple(command) == controller.lazy_moca_start_command
        events.append("start")
        publisher.subscription_count = 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(r1lite_real.subprocess, "run", run)

    controller._ensure_moca_for_joint_target(np.zeros(6))

    assert events[:3] == ["lock", "stop", "start"]
    assert controller._moca_owned_by_env is True

    controller.close()

    assert events.count("stop") == 2
    assert controller._moca_owned_by_env is False
