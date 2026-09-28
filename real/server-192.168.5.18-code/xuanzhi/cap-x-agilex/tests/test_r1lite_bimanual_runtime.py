from pathlib import Path

import pytest

from capx.integrations.r1lite.bimanual_runtime import (
    BimanualRuntimeConfig,
    FeedbackStaleError,
    JointArrivalError,
    LifecycleState,
    R1LiteBimanualRuntime,
)


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTransport:
    def __init__(self, clock: ManualClock, *, arrival: str = "none") -> None:
        self.clock = clock
        self.arrival = arrival
        self.pairs = []
        self.gripper_pairs = []
        self.events = []
        self._on_feedback = None
        self._on_control = None
        self._on_gripper = None
        self._pending = None
        self._spins_for_command = 0
        self.closed = False

    def start(self, on_feedback, on_control, on_gripper, on_pose) -> None:
        self._on_feedback = on_feedback
        self._on_control = on_control
        self._on_gripper = on_gripper
        self._on_pose = on_pose

    def seed(self, left=(0.0, 0.0), right=(0.0, 0.0)) -> None:
        for side, joints in (("left", left), ("right", right)):
            self._on_feedback(side, joints)
            self._on_control(side, joints)
            self._on_gripper(side, 100.0)

    def publish_pair(self, left, right) -> None:
        pair = (tuple(left), tuple(right))
        self.pairs.append(pair)
        self.events.append(("publish", pair))
        self._on_control("left", pair[0])
        self._on_control("right", pair[1])
        self._pending = pair
        self._spins_for_command = 0

    def spin_once(self, timeout_s: float) -> None:
        self.clock.advance(max(timeout_s, 0.001))
        self._spins_for_command += 1
        if self.arrival == "staggered" and self._pending is not None:
            left, right = self._pending
            if self._spins_for_command == 1:
                self._on_feedback("left", left)
                self.events.append(("feedback", "left", left))
            elif self._spins_for_command == 2:
                self._on_feedback("right", right)
                self.events.append(("feedback", "right", right))

    def publish_gripper_pair(self, left, right) -> None:
        self.gripper_pairs.append((left, right))
        self._on_gripper("left", left)
        self._on_gripper("right", right)

    def request_ik(self, **kwargs):
        return kwargs["seed_joints"]

    def close(self) -> None:
        self.closed = True


def runtime_config(tmp_path: Path, **overrides):
    values = {
        "feedback_max_age_s": 0.1,
        "arrival_timeout_s": 0.02,
        "arrival_poll_s": 0.01,
        "progress_steps": 2,
        "lock_path": tmp_path / "bimanual.lock",
    }
    values.update(overrides)
    return BimanualRuntimeConfig(**values)


def start_runtime(tmp_path, *, arrival="none", **config_overrides):
    clock = ManualClock()
    transport = FakeTransport(clock, arrival=arrival)
    runtime = R1LiteBimanualRuntime(
        transport,
        config=runtime_config(tmp_path, **config_overrides),
        clock=clock,
    )
    runtime.start()
    transport.seed()
    return runtime, transport, clock


def test_only_paired_commands_are_published(tmp_path):
    runtime, transport, _clock = start_runtime(tmp_path)

    runtime.publish_pair((0.1, 0.2), (-0.1, -0.2))

    assert transport.pairs == [((0.1, 0.2), (-0.1, -0.2))]
    assert runtime.state is LifecycleState.READY
    runtime.close()


def test_publish_joint_targets_preserves_live_moca_offset(tmp_path):
    runtime, transport, _clock = start_runtime(tmp_path)
    runtime.update_feedback("left", (0.1, 0.2))
    runtime.update_control("left", (1.0, 2.0))
    runtime.update_feedback("right", (-0.1, -0.2))
    runtime.update_control("right", (-1.0, -2.0))

    runtime.publish_joint_targets((0.3, 0.5), (-0.4, -0.6))

    assert transport.pairs[-1][0] == pytest.approx((1.2, 2.3))
    assert transport.pairs[-1][1] == pytest.approx((-1.3, -2.4))
    assert runtime.state is LifecycleState.READY
    runtime.close()


def test_publish_joint_targets_for_repeats_fixed_moca_command(tmp_path):
    runtime, transport, _clock = start_runtime(
        tmp_path,
        feedback_max_age_s=1.0,
    )
    runtime.update_feedback("left", (0.1, 0.2))
    runtime.update_control("left", (1.0, 2.0))
    runtime.update_feedback("right", (-0.1, -0.2))
    runtime.update_control("right", (-1.0, -2.0))

    runtime.publish_joint_targets_for(
        (0.3, 0.5),
        (-0.4, -0.6),
        duration_s=0.05,
        poll_s=0.02,
    )

    assert len(transport.pairs) >= 3
    assert all(
        left == pytest.approx((1.2, 2.3))
        and right == pytest.approx((-1.3, -2.4))
        for left, right in transport.pairs
    )
    runtime.close()


def test_stable_gripper_close_is_held_during_motion_until_open(tmp_path):
    runtime, transport, clock = start_runtime(
        tmp_path,
        feedback_max_age_s=10.0,
        gripper_timeout_s=3.0,
    )

    evidence = runtime.close_both_grippers_stable_and_hold()

    assert clock.now >= 2.0
    assert evidence["reached_close_target"] is True
    assert evidence["validated_contact_position"] is True
    assert evidence["continuous_close_hold_enabled"] is True
    assert len(transport.gripper_pairs) > 2
    transport.gripper_pairs.clear()
    runtime.publish_joint_targets((0.0, 0.0), (0.0, 0.0))
    assert transport.gripper_pairs[-1] == pytest.approx((0.0, 0.0))

    runtime.open_both_grippers()
    assert transport.gripper_pairs[-1] == pytest.approx((100.0, 100.0))
    transport.gripper_pairs.clear()
    runtime.publish_joint_targets((0.0, 0.0), (0.0, 0.0))
    assert transport.gripper_pairs == []
    runtime.close()


def test_joint_telemetry_exposes_velocity_and_effort(tmp_path):
    runtime, _transport, _clock = start_runtime(tmp_path)
    runtime.update_feedback(
        "right",
        (0.1, 0.2),
        velocity=(0.01, -0.02),
        effort=(1.5, -2.5),
    )

    telemetry = runtime.get_joint_telemetry("right")

    assert telemetry["position"] == pytest.approx((0.1, 0.2))
    assert telemetry["velocity"] == pytest.approx((0.01, -0.02))
    assert telemetry["effort"] == pytest.approx((1.5, -2.5))
    runtime.close()


def test_stale_feedback_interlock_holds_both_arms(tmp_path):
    runtime, transport, clock = start_runtime(tmp_path)
    clock.advance(0.11)

    with pytest.raises(FeedbackStaleError):
        runtime.publish_pair((0.1, 0.2), (-0.1, -0.2))

    assert transport.pairs == [((0.0, 0.0), (0.0, 0.0))]
    assert runtime.state is LifecycleState.FAULT
    assert "FeedbackStaleError" in runtime.fault_reason
    runtime.close()


def test_shared_progress_waits_for_both_arm_arrival(tmp_path):
    runtime, transport, _clock = start_runtime(tmp_path, arrival="staggered")

    runtime.execute_joint_motion((0.04, 0.02), (-0.04, -0.02))

    assert transport.pairs[0][0] == pytest.approx((0.02, 0.01))
    assert transport.pairs[0][1] == pytest.approx((-0.02, -0.01))
    assert transport.pairs[1][0] == pytest.approx((0.04, 0.02))
    assert transport.pairs[1][1] == pytest.approx((-0.04, -0.02))
    first_right_feedback = transport.events.index(
        ("feedback", "right", (-0.02, -0.01))
    )
    second_publish = [
        index
        for index, event in enumerate(transport.events)
        if event[0] == "publish"
    ][1]
    assert first_right_feedback < second_publish
    assert runtime.state is LifecycleState.READY
    runtime.close()


def test_motion_stop_callback_holds_current_feedback(tmp_path):
    runtime, transport, _clock = start_runtime(
        tmp_path,
        arrival="none",
        feedback_max_age_s=1.0,
    )
    callback_calls = 0

    def stop_requested():
        nonlocal callback_calls
        callback_calls += 1
        return True

    stopped = runtime.execute_single_arm_motion(
        side="right",
        target=(0.04, 0.02),
        inactive_hold=(0.0, 0.0),
        progress_steps=1,
        stop_requested=stop_requested,
    )

    assert stopped is True
    assert callback_calls == 1
    assert transport.pairs == [
        ((0.0, 0.0), (0.04, 0.02)),
        ((0.0, 0.0), (0.0, 0.0)),
    ]
    assert runtime.state is LifecycleState.HOLDING
    runtime.close()


def test_arrival_failure_retries_twice_then_holds_both(tmp_path):
    runtime, transport, _clock = start_runtime(
        tmp_path,
        arrival="none",
        feedback_max_age_s=1.0,
    )
    target = ((0.03, 0.0), (-0.03, 0.0))

    with pytest.raises(JointArrivalError):
        runtime.execute_joint_motion(*target, progress_steps=1)

    assert transport.pairs[:-1] == [
        target,
        ((0.06, 0.0), (-0.06, 0.0)),
        ((0.09, 0.0), (-0.09, 0.0)),
    ]
    assert transport.pairs[-1] == ((0.09, 0.0), (-0.09, 0.0))
    assert runtime.state is LifecycleState.FAULT
    assert "after 3 attempts" in runtime.fault_reason
    runtime.close()


def test_single_arm_motion_holds_other_arm_and_single_gripper(tmp_path):
    runtime, transport, _clock = start_runtime(tmp_path, arrival="staggered")

    runtime.execute_single_arm_motion(
        side="right",
        target=(0.04, 0.02),
        inactive_hold=(0.0, 0.0),
    )
    runtime.close_gripper("right")

    assert all(pair[0] == pytest.approx((0.0, 0.0)) for pair in transport.pairs)
    assert transport.pairs[-1][1] == pytest.approx((0.04, 0.02))
    assert transport.gripper_pairs[-1] == pytest.approx((100.0, 0.0))
    runtime.close()
