"""Fail-together command mux for the R1 Lite arms.

The runtime deliberately owns both arms.  There is no API for publishing a
single-arm command, so synchronization and interlocks cannot be bypassed by a
caller.  ROS imports are confined to :class:`RosBimanualTransport`, allowing
this module to be imported and unit-tested without a ROS installation.
"""

from __future__ import annotations

import enum
import fcntl
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Protocol


SIDES = ("left", "right")
JointVector = tuple[float, ...]
FeedbackCallback = Callable[..., None]
ControlCallback = Callable[[str, Sequence[float]], None]
GripperCallback = Callable[[str, float], None]
PoseCallback = Callable[[str, Sequence[float], Sequence[float]], None]


class BimanualTransport(Protocol):
    """Transport boundary used by the safety runtime and test fakes."""

    def start(
        self,
        on_feedback: FeedbackCallback,
        on_control: ControlCallback,
        on_gripper: GripperCallback,
        on_pose: PoseCallback,
    ) -> None: ...

    def publish_pair(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> None: ...

    def spin_once(self, timeout_s: float) -> None: ...

    def publish_gripper_pair(self, left: float, right: float) -> None: ...

    def request_ik(
        self,
        *,
        side: str,
        position: Sequence[float],
        quaternion_xyzw: Sequence[float],
        seed_joints: Sequence[float],
        timeout_s: float,
        reset: bool,
    ) -> Sequence[float]: ...

    def close(self) -> None: ...


class LifecycleState(enum.Enum):
    NEW = "new"
    READY = "ready"
    MOVING = "moving"
    HOLDING = "holding"
    FAULT = "fault"
    CLOSED = "closed"


class BimanualRuntimeError(RuntimeError):
    """Base error for bimanual runtime failures."""


class LifecycleError(BimanualRuntimeError):
    """An operation was attempted in an invalid lifecycle state."""


class FeedbackStaleError(BimanualRuntimeError):
    """At least one arm lacks fresh joint feedback."""


class JointArrivalError(BimanualRuntimeError):
    """The paired joint-arrival barrier was not satisfied."""


class ProcessLockError(BimanualRuntimeError):
    """Another process owns the bimanual command mux."""


@dataclass(frozen=True)
class BimanualRuntimeConfig:
    feedback_max_age_s: float = 0.20
    arrival_timeout_s: float = 1.0
    arrival_poll_s: float = 0.01
    arrival_soft_threshold_rad: float = 0.01
    arrival_velocity_threshold_rad_s: float | None = None
    arrival_settle_time_s: float = 0.0
    arrival_max_retries: int = 2
    arrival_hard_threshold_rad: float = 0.05
    progress_steps: int = 20
    record_joint_progress_tcp: bool = False
    gripper_timeout_s: float = 5.0
    gripper_tolerance: float = 5.0
    gripper_open_position: float = 100.0
    gripper_close_position: float = 0.0
    gripper_minimum_close_duration_s: float = 2.0
    gripper_stability_window_s: float = 0.5
    gripper_stability_tolerance: float = 0.5
    gripper_maximum_contact_position: float = 20.0
    lock_path: Path = Path("/tmp/capx-r1lite-bimanual-runtime.lock")

    def __post_init__(self) -> None:
        if self.feedback_max_age_s <= 0:
            raise ValueError("feedback_max_age_s must be positive")
        if self.arrival_timeout_s < 0 or self.arrival_poll_s < 0:
            raise ValueError("arrival timeouts must be non-negative")
        if self.arrival_soft_threshold_rad < 0:
            raise ValueError("arrival_soft_threshold_rad must be non-negative")
        if (
            self.arrival_velocity_threshold_rad_s is not None
            and self.arrival_velocity_threshold_rad_s < 0
        ):
            raise ValueError("arrival velocity threshold must be non-negative")
        if self.arrival_settle_time_s < 0:
            raise ValueError("arrival settle time must be non-negative")
        if self.arrival_hard_threshold_rad < self.arrival_soft_threshold_rad:
            raise ValueError("hard arrival threshold must not be below soft threshold")
        if self.arrival_max_retries < 0:
            raise ValueError("arrival_max_retries must be non-negative")
        if self.progress_steps <= 0:
            raise ValueError("progress_steps must be positive")
        if self.gripper_minimum_close_duration_s < 0:
            raise ValueError("minimum gripper close duration must be non-negative")
        if self.gripper_stability_window_s <= 0:
            raise ValueError("gripper stability window must be positive")
        if self.gripper_stability_tolerance < 0:
            raise ValueError("gripper stability tolerance must be non-negative")
        if self.gripper_maximum_contact_position < self.gripper_close_position:
            raise ValueError("maximum gripper contact position is below close target")


@dataclass(frozen=True)
class _Sample:
    joints: JointVector
    received_at: float


@dataclass(frozen=True)
class _PoseSample:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    received_at: float


def _joints(values: Sequence[float], *, label: str) -> JointVector:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{label} joint vector is empty")
    if any(value != value or value in (float("inf"), float("-inf")) for value in result):
        raise ValueError(f"{label} joint vector contains a non-finite value")
    return result


class _ProcessLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: Any | None = None

    def acquire(self) -> None:
        if self._file is not None:
            return
        try:
            handle = self.path.open("a+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if "handle" in locals():
                handle.close()
            raise ProcessLockError(
                f"bimanual command mux already locked by another process: {self.path}"
            ) from exc
        except Exception:
            if "handle" in locals():
                handle.close()
            raise
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._file = handle

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None


class R1LiteBimanualRuntime:
    """Lifecycle-controlled, fail-together dual-arm command mux."""

    def __init__(
        self,
        transport: BimanualTransport | None = None,
        *,
        config: BimanualRuntimeConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or BimanualRuntimeConfig()
        self.transport: BimanualTransport = transport or RosBimanualTransport()
        self._clock = clock
        self._process_lock = _ProcessLock(self.config.lock_path)
        self._mutex = RLock()
        self._feedback: dict[str, _Sample | None] = {side: None for side in SIDES}
        self._velocity: dict[str, _Sample | None] = {side: None for side in SIDES}
        self._effort: dict[str, _Sample | None] = {side: None for side in SIDES}
        self._control: dict[str, JointVector | None] = {side: None for side in SIDES}
        self._control_received_at: dict[str, float | None] = {
            side: None for side in SIDES
        }
        self._gripper: dict[str, _Sample | None] = {side: None for side in SIDES}
        self._gripper_hold_pair: tuple[float, float] | None = None
        self._pose: dict[str, _PoseSample | None] = {side: None for side in SIDES}
        self._state = LifecycleState.NEW
        self._fault_reason: str | None = None
        self._command_pair_count = 0
        self._joint_arrival_trace: list[dict[str, Any]] = []
        self._joint_progress_tcp_trace: list[dict[str, Any]] = []

    @property
    def state(self) -> LifecycleState:
        with self._mutex:
            return self._state

    @property
    def fault_reason(self) -> str | None:
        with self._mutex:
            return self._fault_reason

    @property
    def command_pair_count(self) -> int:
        with self._mutex:
            return self._command_pair_count

    @property
    def joint_arrival_trace(self) -> list[dict[str, Any]]:
        with self._mutex:
            return [dict(item) for item in self._joint_arrival_trace]

    @property
    def joint_progress_tcp_trace(self) -> list[dict[str, Any]]:
        with self._mutex:
            return [dict(item) for item in self._joint_progress_tcp_trace]

    def start(self) -> None:
        with self._mutex:
            if self._state is not LifecycleState.NEW:
                raise LifecycleError(f"cannot start runtime from {self._state.value}")
            self._process_lock.acquire()
            try:
                self.transport.start(
                    self.update_feedback,
                    self.update_control,
                    self.update_gripper,
                    self.update_gripper_pose,
                )
            except Exception:
                self._process_lock.release()
                raise
            self._state = LifecycleState.READY

    def update_feedback(
        self,
        side: str,
        joints: Sequence[float],
        velocity: Sequence[float] = (),
        effort: Sequence[float] = (),
    ) -> None:
        """Receive feedback; public so deterministic fake transports stay tiny."""
        self._validate_side(side)
        received_at = self._clock()
        sample = _Sample(_joints(joints, label=f"{side} feedback"), received_at)
        with self._mutex:
            self._feedback[side] = sample
            self._velocity[side] = (
                _Sample(_joints(velocity, label=f"{side} velocity"), received_at)
                if len(velocity) > 0
                else None
            )
            self._effort[side] = (
                _Sample(_joints(effort, label=f"{side} effort"), received_at)
                if len(effort) > 0
                else None
            )

    def update_control(self, side: str, joints: Sequence[float]) -> None:
        """Receive the current MOCA command (``MotorControl.p_des``)."""
        self._validate_side(side)
        command = _joints(joints, label=f"{side} control")
        with self._mutex:
            self._control[side] = command
            self._control_received_at[side] = self._clock()

    def update_gripper(self, side: str, position: float) -> None:
        self._validate_side(side)
        sample = _Sample((float(position),), self._clock())
        with self._mutex:
            self._gripper[side] = sample

    def update_gripper_pose(
        self,
        side: str,
        position: Sequence[float],
        quaternion_xyzw: Sequence[float],
    ) -> None:
        self._validate_side(side)
        xyz = _joints(position, label=f"{side} gripper position")
        quat = _joints(quaternion_xyzw, label=f"{side} gripper quaternion")
        if len(xyz) != 3 or len(quat) != 4:
            raise ValueError("gripper pose must contain 3D position and quaternion")
        norm = sum(value * value for value in quat) ** 0.5
        if norm <= 1e-12:
            raise ValueError("gripper quaternion cannot be zero")
        normalized = tuple(value / norm for value in quat)
        with self._mutex:
            self._pose[side] = _PoseSample(
                (xyz[0], xyz[1], xyz[2]),
                (normalized[0], normalized[1], normalized[2], normalized[3]),
                self._clock(),
            )

    def get_feedback_pair(self) -> dict[str, JointVector]:
        with self._mutex:
            self._require_fresh_feedback()
            return {
                side: self._feedback[side].joints  # type: ignore[union-attr]
                for side in SIDES
            }

    def get_gripper_feedback(self, side: str) -> float:
        """Return one fresh gripper position for grasp-retention gates."""
        self._validate_side(side)
        with self._mutex:
            sample = self._gripper[side]
            if (
                sample is None
                or self._clock() - sample.received_at
                > self.config.feedback_max_age_s
            ):
                raise FeedbackStaleError(f"{side} gripper feedback is stale")
            return float(sample.joints[0])

    def get_joint_telemetry(
        self, side: str
    ) -> dict[str, JointVector | float]:
        """Return fresh joint position, velocity, and motor effort feedback."""
        self._validate_side(side)
        with self._mutex:
            now = self._clock()
            position = self._feedback[side]
            velocity = self._velocity[side]
            effort = self._effort[side]
            if any(
                sample is None
                or now - sample.received_at > self.config.feedback_max_age_s
                for sample in (position, velocity, effort)
            ):
                raise FeedbackStaleError(f"{side} joint telemetry is stale")
            return {
                "position": position.joints,  # type: ignore[union-attr]
                "velocity": velocity.joints,  # type: ignore[union-attr]
                "effort": effort.joints,  # type: ignore[union-attr]
                "received_at": position.received_at,  # type: ignore[union-attr]
            }

    def get_tool_pose_pair(
        self,
    ) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
        """Return vendor gripper poses expressed in the shared tool-Z convention."""
        result = {}
        with self._mutex:
            now = self._clock()
            for side in SIDES:
                sample = self._pose[side]
                if (
                    sample is None
                    or now - sample.received_at > self.config.feedback_max_age_s
                ):
                    raise FeedbackStaleError(f"{side} physical EE pose is stale")
                result[side] = (
                    sample.position,
                    self._quat_multiply(
                        sample.quaternion_xyzw,
                        (0.0, 2.0**-0.5, 0.0, 2.0**-0.5),
                    ),
                )
        return result

    def get_tool_pose(
        self, side: str
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return one vendor gripper pose in the shared tool-Z convention."""
        self._validate_side(side)
        with self._mutex:
            sample = self._pose[side]
            if (
                sample is None
                or self._clock() - sample.received_at > self.config.feedback_max_age_s
            ):
                raise FeedbackStaleError(f"{side} physical EE pose is stale")
            return (
                sample.position,
                self._quat_multiply(
                    sample.quaternion_xyzw,
                    (0.0, 2.0**-0.5, 0.0, 2.0**-0.5),
                ),
            )

    def feedback_is_fresh(self) -> bool:
        with self._mutex:
            now = self._clock()
            return all(
                sample is not None
                and 0.0 <= now - sample.received_at <= self.config.feedback_max_age_s
                for sample in self._feedback.values()
            )

    def refresh_feedback(self, timeout_s: float = 0.02) -> None:
        """Spin the transport once so queued sensor callbacks can advance."""
        self.transport.spin_once(timeout_s)

    def tool_pose_is_fresh(self) -> bool:
        with self._mutex:
            now = self._clock()
            return all(
                sample is not None
                and 0.0 <= now - sample.received_at <= self.config.feedback_max_age_s
                for sample in self._pose.values()
            )

    def wait_until_ready(
        self,
        timeout_s: float = 5.0,
        *,
        require_control_reference: bool = True,
        require_tool_pose: bool = False,
    ) -> None:
        """Pump the sole executor until both arms have usable input snapshots."""
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        with self._mutex:
            self._require_operational()
        deadline = self._clock() + timeout_s
        while True:
            with self._mutex:
                ready = self.feedback_is_fresh() and (
                    not require_control_reference
                    or all(command is not None for command in self._control.values())
                ) and (
                    not require_tool_pose or self.tool_pose_is_fresh()
                )
            if ready:
                return
            remaining = deadline - self._clock()
            if remaining <= 0:
                error = FeedbackStaleError(
                    "timed out waiting for paired feedback and control references"
                )
                self._fault_and_hold(error)
                raise error
            self.transport.spin_once(min(self.config.arrival_poll_s, remaining))

    def publish_pair(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> None:
        """Publish one indivisible API-level pair after checking both feedbacks."""
        try:
            self.wait_until_ready(timeout_s=1.0)
            with self._mutex:
                self._require_operational()
                self._require_fresh_feedback()
                pair = self._validate_pair(left, right)
                self._publish_raw_pair(*pair)
        except Exception as exc:
            if self.state is not LifecycleState.FAULT:
                self._fault_and_hold(exc)
            raise

    def publish_joint_targets(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
    ) -> None:
        """Publish feedback-space joint targets without an arrival gate."""
        try:
            self.wait_until_ready(timeout_s=1.0)
            with self._mutex:
                self._require_operational()
                self._require_fresh_feedback()
                targets = self._validate_pair(left_target, right_target)
                commands = self._feedback_targets_to_commands(targets)
                self._publish_raw_pair(*commands)
        except Exception as exc:
            if self.state is not LifecycleState.FAULT:
                self._fault_and_hold(exc)
            raise

    def publish_joint_targets_for(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        *,
        duration_s: float,
        poll_s: float = 0.02,
    ) -> None:
        """Convert feedback targets once, then repeat the fixed MOCA command."""
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if poll_s <= 0:
            raise ValueError("poll_s must be positive")
        try:
            self.wait_until_ready(timeout_s=1.0)
            with self._mutex:
                self._require_operational()
                self._require_fresh_feedback()
                targets = self._validate_pair(left_target, right_target)
                commands = self._feedback_targets_to_commands(targets)
            deadline = self._clock() + duration_s
            while True:
                with self._mutex:
                    self._require_operational()
                    self._require_fresh_feedback()
                    self._publish_raw_pair(*commands)
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return
                self.transport.spin_once(min(poll_s, remaining))
        except Exception as exc:
            if self.state is not LifecycleState.FAULT:
                self._fault_and_hold(exc)
            raise

    def hold(self) -> None:
        """Command a common hold pair using latest commands, then feedback fallback."""
        try:
            with self._mutex:
                if self._state in (LifecycleState.NEW, LifecycleState.CLOSED):
                    raise LifecycleError(f"cannot hold runtime from {self._state.value}")
                if not self._publish_hold_pair():
                    raise BimanualRuntimeError(
                        "cannot hold both arms without paired command or feedback"
                    )
                if self._state is not LifecycleState.FAULT:
                    self._state = LifecycleState.HOLDING
        except Exception as exc:
            if self.state is not LifecycleState.FAULT:
                self._fault_and_hold(exc)
            raise

    def execute_joint_motion(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        *,
        progress_steps: int | None = None,
        trace_context: dict[str, Any] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> bool:
        """Move both arms through the same scalar progress and arrival barriers."""
        steps = self.config.progress_steps if progress_steps is None else progress_steps
        if steps <= 0:
            raise ValueError("progress_steps must be positive")
        try:
            self.wait_until_ready(timeout_s=1.0)
            with self._mutex:
                self._require_operational()
                self._require_fresh_feedback()
                targets = self._validate_pair(left_target, right_target)
                starts = self._motion_starts()
                self._state = LifecycleState.MOVING

            for index in range(1, steps + 1):
                progress = index / steps
                waypoint = (
                    self._interpolate(starts[0], targets[0], progress),
                    self._interpolate(starts[1], targets[1], progress),
                )
                progress_context = {
                    **(trace_context or {}),
                    "progress_step": index,
                    "progress_steps": steps,
                }
                stopped = self._await_paired_arrival(
                    waypoint,
                    trace_context=progress_context,
                    stop_requested=stop_requested,
                )
                if stopped:
                    with self._mutex:
                        self._state = LifecycleState.HOLDING
                    return True
                if self.config.record_joint_progress_tcp:
                    poses = self.get_tool_pose_pair()
                    with self._mutex:
                        self._joint_progress_tcp_trace.append(
                            {
                                **progress_context,
                                "sampled_at_monotonic_s": self._clock(),
                                "left_position_base_m": list(poses["left"][0]),
                                "left_quaternion_xyzw": list(poses["left"][1]),
                                "right_position_base_m": list(poses["right"][0]),
                                "right_quaternion_xyzw": list(poses["right"][1]),
                            }
                        )

            with self._mutex:
                self._state = LifecycleState.READY
            return False
        except Exception as exc:
            if self.state is not LifecycleState.FAULT:
                self._fault_and_hold(exc)
            raise

    def execute_single_arm_motion(
        self,
        *,
        side: str,
        target: Sequence[float],
        inactive_hold: Sequence[float],
        progress_steps: int | None = None,
        trace_context: dict[str, Any] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> bool:
        """Move one arm while commanding the other to an explicit hold target."""
        self._validate_side(side)
        if side == "left":
            return self.execute_joint_motion(
                target,
                inactive_hold,
                progress_steps=progress_steps,
                trace_context=trace_context,
                stop_requested=stop_requested,
            )
        return self.execute_joint_motion(
            inactive_hold,
            target,
            progress_steps=progress_steps,
            trace_context=trace_context,
            stop_requested=stop_requested,
        )

    def command_grippers(self, position: float) -> None:
        """Publish one synchronized gripper target and require paired feedback."""
        target = float(position)
        try:
            with self._mutex:
                self._require_operational()
                before = {
                    side: None
                    if self._gripper[side] is None
                    else self._gripper[side].joints[0]
                    for side in SIDES
                }
                self.transport.publish_gripper_pair(target, target)
                self._command_pair_count += 1
            deadline = self._clock() + self.config.gripper_timeout_s
            while self._clock() < deadline:
                self.transport.spin_once(self.config.arrival_poll_s)
                with self._mutex:
                    reached = all(
                        self._gripper[side] is not None
                        and abs(self._gripper[side].joints[0] - target)
                        <= self.config.gripper_tolerance
                        for side in SIDES
                    )
                    moved_for_contact = (
                        target == self.config.gripper_close_position
                        and all(
                            before[side] is not None
                            and self._gripper[side] is not None
                            and abs(
                                self._gripper[side].joints[0] - before[side]
                            )
                            >= 2.0
                            for side in SIDES
                        )
                    )
                if reached or moved_for_contact:
                    return
                self.transport.publish_gripper_pair(target, target)
                self._command_pair_count += 1
            raise BimanualRuntimeError(
                "paired gripper feedback did not reach or respond to target"
            )
        except Exception as exc:
            self._fault_and_hold(exc)
            raise

    def close_both_grippers_stable_and_hold(self) -> dict[str, Any]:
        """Close for at least two seconds, validate feedback, then keep holding."""
        target = float(self.config.gripper_close_position)
        try:
            self.wait_until_ready(timeout_s=1.0)
            feedback_deadline = self._clock() + 1.0
            while True:
                with self._mutex:
                    self._require_operational()
                    before = {
                        side: None
                        if self._gripper[side] is None
                        else float(self._gripper[side].joints[0])
                        for side in SIDES
                    }
                    if all(value is not None for value in before.values()):
                        self._gripper_hold_pair = (target, target)
                        break
                if self._clock() >= feedback_deadline:
                    raise FeedbackStaleError(
                        "paired gripper feedback unavailable before close"
                    )
                self.transport.spin_once(self.config.arrival_poll_s)
            started_at = self._clock()
            deadline = started_at + self.config.gripper_timeout_s
            samples: dict[str, list[tuple[float, float]]] = {
                side: [] for side in SIDES
            }
            while self._clock() < deadline:
                with self._mutex:
                    self._publish_active_gripper_hold()
                self.transport.spin_once(self.config.arrival_poll_s)
                now = self._clock()
                with self._mutex:
                    positions = {
                        side: (
                            None
                            if self._gripper[side] is None
                            else float(self._gripper[side].joints[0])
                        )
                        for side in SIDES
                    }
                if any(value is None for value in positions.values()):
                    continue
                for side in SIDES:
                    samples[side].append((now, float(positions[side])))
                    samples[side] = [
                        sample
                        for sample in samples[side]
                        if sample[0]
                        >= now - self.config.gripper_stability_window_s
                    ]
                reached = all(
                    abs(float(positions[side]) - target)
                    <= self.config.gripper_tolerance
                    for side in SIDES
                )
                moved_for_contact = all(
                    abs(float(positions[side]) - float(before[side])) >= 2.0
                    for side in SIDES
                )
                stable = all(
                    len(samples[side]) >= 5
                    and samples[side][-1][0] - samples[side][0][0]
                    >= self.config.gripper_stability_window_s * 0.8
                    and max(value for _, value in samples[side])
                    - min(value for _, value in samples[side])
                    <= self.config.gripper_stability_tolerance
                    for side in SIDES
                )
                validated_contact_position = all(
                    float(positions[side])
                    <= self.config.gripper_maximum_contact_position
                    for side in SIDES
                )
                preexisting_contact_position = all(
                    float(before[side])
                    <= self.config.gripper_maximum_contact_position
                    for side in SIDES
                )
                elapsed = now - started_at
                if (
                    elapsed >= self.config.gripper_minimum_close_duration_s
                    and (
                        reached
                        or (
                            stable
                            and validated_contact_position
                            and (
                                moved_for_contact
                                or preexisting_contact_position
                            )
                        )
                    )
                ):
                    return {
                        "initial_position": {
                            side: float(before[side]) for side in SIDES
                        },
                        "final_position": {
                            side: float(positions[side]) for side in SIDES
                        },
                        "reached_close_target": reached,
                        "stable_contact_position": stable,
                        "validated_contact_position": (
                            validated_contact_position
                        ),
                        "preexisting_contact_position": (
                            preexisting_contact_position
                        ),
                        "minimum_close_command_duration_s": (
                            self.config.gripper_minimum_close_duration_s
                        ),
                        "continuous_close_hold_enabled": True,
                    }
            with self._mutex:
                self._gripper_hold_pair = None
            raise BimanualRuntimeError(
                "paired grippers did not reach a stable validated close position"
            )
        except Exception as exc:
            with self._mutex:
                self._gripper_hold_pair = None
            self._fault_and_hold(exc)
            raise

    def clear_gripper_hold(self) -> None:
        with self._mutex:
            self._gripper_hold_pair = None

    def publish_gripper(self, side: str, position: float) -> None:
        """Publish one gripper target without applying an arrival gate."""
        self._validate_side(side)
        target = float(position)
        inactive = "right" if side == "left" else "left"
        try:
            self.wait_until_ready(timeout_s=1.0)
            with self._mutex:
                self._require_operational()
                inactive_sample = self._gripper[inactive]
                if inactive_sample is None:
                    raise FeedbackStaleError(
                        f"{inactive} gripper feedback is unavailable"
                    )
                pair = {
                    side: target,
                    inactive: inactive_sample.joints[0],
                }
                self.transport.publish_gripper_pair(pair["left"], pair["right"])
                self._command_pair_count += 1
        except Exception as exc:
            if self.state is not LifecycleState.FAULT:
                self._fault_and_hold(exc)
            raise

    def command_gripper(self, side: str, position: float) -> None:
        """Command one gripper while preserving the other gripper's feedback target."""
        self._validate_side(side)
        target = float(position)
        inactive = "right" if side == "left" else "left"
        try:
            with self._mutex:
                self._require_operational()
                active_before = (
                    None
                    if self._gripper[side] is None
                    else self._gripper[side].joints[0]
                )
                inactive_sample = self._gripper[inactive]
                if inactive_sample is None:
                    raise FeedbackStaleError(
                        f"{inactive} gripper feedback unavailable for hold"
                    )
                pair = {
                    side: target,
                    inactive: inactive_sample.joints[0],
                }
                inactive_hold = inactive_sample.joints[0]
                self.transport.publish_gripper_pair(pair["left"], pair["right"])
                self._command_pair_count += 1
            deadline = self._clock() + self.config.gripper_timeout_s
            while self._clock() < deadline:
                self.transport.spin_once(self.config.arrival_poll_s)
                with self._mutex:
                    sample = self._gripper[side]
                    reached = (
                        sample is not None
                        and abs(sample.joints[0] - target)
                        <= self.config.gripper_tolerance
                    )
                    contacted = (
                        target == self.config.gripper_close_position
                        and active_before is not None
                        and sample is not None
                        and abs(sample.joints[0] - active_before) >= 2.0
                    )
                    inactive_feedback = self._gripper[inactive]
                    inactive_held = (
                        inactive_feedback is not None
                        and abs(
                            inactive_feedback.joints[0] - inactive_hold
                        )
                        <= self.config.gripper_tolerance
                    )
                    if not inactive_held:
                        raise BimanualRuntimeError(
                            f"{inactive} gripper changed during {side} command"
                        )
                if reached or contacted:
                    return
                self.transport.publish_gripper_pair(pair["left"], pair["right"])
                self._command_pair_count += 1
            raise BimanualRuntimeError(
                f"{side} gripper feedback did not reach or respond to target"
            )
        except Exception as exc:
            self._fault_and_hold(exc)
            raise

    def open_gripper(self, side: str) -> None:
        self.command_gripper(side, self.config.gripper_open_position)

    def close_gripper(self, side: str) -> None:
        self.command_gripper(side, self.config.gripper_close_position)

    def open_both_grippers(self) -> None:
        self.clear_gripper_hold()
        self.command_grippers(self.config.gripper_open_position)

    def close_both_grippers(self) -> None:
        self.command_grippers(self.config.gripper_close_position)

    def request_ik(
        self,
        *,
        side: str,
        position: Sequence[float],
        quaternion_xyzw: Sequence[float],
        seed_joints: Sequence[float],
        timeout_s: float = 1.0,
        reset: bool = True,
    ) -> JointVector:
        self._validate_side(side)
        with self._mutex:
            self._require_operational()
        return _joints(
            self.transport.request_ik(
                side=side,
                position=position,
                quaternion_xyzw=quaternion_xyzw,
                seed_joints=seed_joints,
                timeout_s=timeout_s,
                reset=reset,
            ),
            label=f"{side} IK result",
        )

    def close(self) -> None:
        with self._mutex:
            if self._state is LifecycleState.CLOSED:
                return
            if self._state not in (LifecycleState.NEW, LifecycleState.FAULT):
                try:
                    self._publish_hold_pair()
                except Exception:
                    pass
            try:
                if self._state is not LifecycleState.NEW:
                    self.transport.close()
            finally:
                self._process_lock.release()
                self._state = LifecycleState.CLOSED

    @staticmethod
    def _quat_multiply(
        first: Sequence[float],
        second: Sequence[float],
    ) -> tuple[float, float, float, float]:
        x1, y1, z1, w1 = first
        x2, y2, z2, w2 = second
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    def __enter__(self) -> R1LiteBimanualRuntime:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _validate_side(side: str) -> None:
        if side not in SIDES:
            raise ValueError(f"unknown arm side {side!r}")

    def _require_operational(self) -> None:
        if self._state not in (
            LifecycleState.READY,
            LifecycleState.MOVING,
            LifecycleState.HOLDING,
        ):
            raise LifecycleError(f"runtime is not operational: {self._state.value}")

    def _require_fresh_feedback(self) -> None:
        now = self._clock()
        stale = []
        for side in SIDES:
            sample = self._feedback[side]
            if (
                sample is None
                or now < sample.received_at
                or now - sample.received_at > self.config.feedback_max_age_s
            ):
                stale.append(side)
        if stale:
            raise FeedbackStaleError(
                f"fresh joint feedback required for both arms; stale={','.join(stale)}"
            )

    def _validate_pair(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> tuple[JointVector, JointVector]:
        pair = (_joints(left, label="left command"), _joints(right, label="right command"))
        if len(pair[0]) != len(pair[1]):
            raise ValueError("left and right command dimensions differ")
        for side, command in zip(SIDES, pair):
            sample = self._feedback[side]
            if sample is not None and len(command) != len(sample.joints):
                raise ValueError(f"{side} command and feedback dimensions differ")
        return pair

    def _publish_raw_pair(self, left: JointVector, right: JointVector) -> None:
        self.transport.publish_pair(left, right)
        self._command_pair_count += 1
        self._publish_active_gripper_hold()

    def _publish_active_gripper_hold(self) -> None:
        if self._gripper_hold_pair is None:
            return
        self.transport.publish_gripper_pair(*self._gripper_hold_pair)
        self._command_pair_count += 1

    def _hold_pair(self) -> tuple[JointVector, JointVector] | None:
        commands: list[JointVector] = []
        for side in SIDES:
            command = self._control[side]
            if command is None:
                sample = self._feedback[side]
                command = None if sample is None else sample.joints
            if command is None:
                return None
            commands.append(command)
        if len(commands[0]) != len(commands[1]):
            return None
        return commands[0], commands[1]

    def _publish_hold_pair(self) -> bool:
        pair = self._hold_pair()
        if pair is None:
            return False
        self._publish_raw_pair(*pair)
        return True

    def _fault_and_hold(self, exc: BaseException) -> None:
        with self._mutex:
            if self._state in (LifecycleState.NEW, LifecycleState.CLOSED):
                return
            self._fault_reason = f"{type(exc).__name__}: {exc}"
            self._state = LifecycleState.FAULT
            try:
                self._publish_hold_pair()
            except Exception:
                pass

    def _motion_starts(self) -> tuple[JointVector, JointVector]:
        starts = []
        for side in SIDES:
            sample = self._feedback[side]
            assert sample is not None
            starts.append(sample.joints)
        return starts[0], starts[1]

    def _feedback_targets_to_commands(
        self,
        targets: tuple[JointVector, JointVector],
    ) -> tuple[JointVector, JointVector]:
        """Map desired feedback joints through the live MOCA command offset."""
        commands = []
        for side, target in zip(SIDES, targets):
            sample = self._feedback[side]
            control = self._control[side]
            if sample is None or control is None:
                raise BimanualRuntimeError(
                    f"{side} feedback/control reference unavailable"
                )
            commands.append(
                tuple(
                    commanded + desired - observed
                    for commanded, desired, observed in zip(
                        control,
                        target,
                        sample.joints,
                    )
                )
            )
        return commands[0], commands[1]

    @staticmethod
    def _interpolate(
        start: JointVector,
        target: JointVector,
        progress: float,
    ) -> JointVector:
        return tuple(a + (b - a) * progress for a, b in zip(start, target))

    def _arrival_errors(
        self,
        waypoint: tuple[JointVector, JointVector],
    ) -> dict[str, float]:
        errors = {}
        for side, target in zip(SIDES, waypoint):
            sample = self._feedback[side]
            assert sample is not None
            errors[side] = max(
                abs(observed - expected)
                for observed, expected in zip(sample.joints, target)
            )
        return errors

    def _await_paired_arrival(
        self,
        waypoint: tuple[JointVector, JointVector],
        *,
        trace_context: dict[str, Any] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> bool:
        last_errors: dict[str, float] = {}
        attempts = self.config.arrival_max_retries + 1
        for attempt in range(attempts):
            with self._mutex:
                self._require_fresh_feedback()
                commands = self._feedback_targets_to_commands(waypoint)
                before = {}
                command_offsets = []
                now = self._clock()
                for side, target, command in zip(SIDES, waypoint, commands):
                    feedback = self._feedback[side]
                    control = self._control[side]
                    assert feedback is not None and control is not None
                    command_offsets.append(
                        tuple(
                            commanded - observed
                            for commanded, observed in zip(
                                control,
                                feedback.joints,
                            )
                        )
                    )
                    before[side] = {
                        "target_rad": list(target),
                        "feedback_rad": list(feedback.joints),
                        "velocity_rad_s": (
                            None
                            if self._velocity[side] is None
                            else list(self._velocity[side].joints)
                        ),
                        "control_reference_rad": list(control),
                        "published_command_rad": list(command),
                        "target_minus_feedback_rad": [
                            desired - observed
                            for desired, observed in zip(target, feedback.joints)
                        ],
                        "feedback_age_s": now - feedback.received_at,
                        "control_age_s": (
                            None
                            if self._control_received_at[side] is None
                            else now - self._control_received_at[side]  # type: ignore[operator]
                        ),
                    }
                self._publish_raw_pair(*commands)
                trace_item = {
                    **(trace_context or {}),
                    "attempt": attempt + 1,
                    "published_at_monotonic_s": now,
                    "before_publish": before,
                }
                self._joint_arrival_trace.append(trace_item)
            deadline = self._clock() + self.config.arrival_timeout_s
            stable_since: float | None = None
            while True:
                self.transport.spin_once(self.config.arrival_poll_s)
                with self._mutex:
                    self._publish_active_gripper_hold()
                    self._require_fresh_feedback()
                    last_errors = self._arrival_errors(waypoint)
                if stop_requested is not None and stop_requested():
                    with self._mutex:
                        hold_commands = tuple(
                            tuple(
                                observed + offset
                                for observed, offset in zip(
                                    self._feedback[side].joints,  # type: ignore[union-attr]
                                    command_offsets[index],
                                )
                            )
                            for index, side in enumerate(SIDES)
                        )
                        self._publish_raw_pair(
                            hold_commands[0],
                            hold_commands[1],
                        )
                        trace_item["result"] = "stopped_by_callback"
                        trace_item["final_max_error_rad"] = dict(last_errors)
                        trace_item["final_feedback_rad"] = {
                            side: list(self._feedback[side].joints)  # type: ignore[union-attr]
                            for side in SIDES
                        }
                    return True
                if all(
                    error <= self.config.arrival_soft_threshold_rad
                    for error in last_errors.values()
                ):
                    velocity_passed = (
                        self.config.arrival_velocity_threshold_rad_s is None
                        or all(
                            self._velocity[side] is not None
                            and max(
                                abs(value)
                                for value in self._velocity[side].joints
                            )
                            <= self.config.arrival_velocity_threshold_rad_s
                            for side in SIDES
                        )
                    )
                    if velocity_passed:
                        if stable_since is None:
                            stable_since = self._clock()
                    else:
                        stable_since = None
                    if (
                        stable_since is None
                        or self._clock() - stable_since
                        < self.config.arrival_settle_time_s
                    ):
                        if self._clock() >= deadline:
                            break
                        continue
                    with self._mutex:
                        trace_item["result"] = "arrived"
                        trace_item["final_max_error_rad"] = dict(last_errors)
                        trace_item["final_feedback_rad"] = {
                            side: list(self._feedback[side].joints)  # type: ignore[union-attr]
                            for side in SIDES
                        }
                        trace_item["final_velocity_rad_s"] = {
                            side: (
                                None
                                if self._velocity[side] is None
                                else list(self._velocity[side].joints)
                            )
                            for side in SIDES
                        }
                    return False
                stable_since = None
                if self._clock() >= deadline:
                    break
            if any(
                error > self.config.arrival_hard_threshold_rad
                for error in last_errors.values()
            ):
                with self._mutex:
                    trace_item["result"] = "hard_limit"
                    trace_item["final_max_error_rad"] = dict(last_errors)
                    trace_item["final_feedback_rad"] = {
                        side: list(self._feedback[side].joints)  # type: ignore[union-attr]
                        for side in SIDES
                    }
                raise JointArrivalError(
                    f"hard joint-arrival limit exceeded: {last_errors}"
                )
            with self._mutex:
                trace_item["result"] = "retry_timeout"
                trace_item["final_max_error_rad"] = dict(last_errors)
                trace_item["final_feedback_rad"] = {
                    side: list(self._feedback[side].joints)  # type: ignore[union-attr]
                    for side in SIDES
                }
        with self._mutex:
            trace_item["result"] = "soft_timeout"
            trace_item["final_max_error_rad"] = dict(last_errors)
            trace_item["final_feedback_rad"] = {
                side: list(self._feedback[side].joints)  # type: ignore[union-attr]
                for side in SIDES
            }
        raise JointArrivalError(
            f"paired joint-arrival barrier failed after {attempts} attempts: {last_errors}"
        )


class RosBimanualTransport:
    """Production ROS 2 transport with one node and one executor."""

    def __init__(self, *, node_name: str = "capx_r1lite_bimanual_command_mux") -> None:
        self.node_name = node_name
        self._trusted_infrastructure_publishers = {"moca_adapter"}
        self._rclpy: Any | None = None
        self._node: Any | None = None
        self._executor: Any | None = None
        self._publishers: dict[str, Any] = {}
        self._gripper_publishers: dict[str, Any] = {}
        self._target_topics = {
            side: f"/motion_target/target_joint_state_arm_{side}"
            for side in SIDES
        }
        self._ik_pose_publishers: dict[str, Any] = {}
        self._ik_reset_publishers: dict[str, Any] = {}
        self._ik_results: dict[str, tuple[int, JointVector] | None] = {
            side: None for side in SIDES
        }
        self._ik_sequences = {side: 0 for side in SIDES}
        self._joint_state_type: Any | None = None
        self._pose_stamped_type: Any | None = None
        self._owns_context = False

    def start(
        self,
        on_feedback: FeedbackCallback,
        on_control: ControlCallback,
        on_gripper: GripperCallback,
        on_pose: PoseCallback,
    ) -> None:
        if self._node is not None:
            raise RuntimeError("ROS bimanual transport already started")
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from hdas_msg.msg import MotorControl
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
        except Exception as exc:  # pragma: no cover - requires robot ROS install
            raise RuntimeError(
                "RosBimanualTransport requires sourced ROS 2 and hdas_msg packages"
            ) from exc

        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        try:
            self._node = rclpy.create_node(self.node_name)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._joint_state_type = JointState
            self._pose_stamped_type = PoseStamped
            for side in SIDES:
                if self._node.get_publishers_info_by_topic(
                    self._target_topics[side]
                ):
                    raise RuntimeError(
                        f"foreign publisher already owns {self._target_topics[side]}"
                    )
                self._publishers[side] = self._node.create_publisher(
                    JointState,
                    self._target_topics[side],
                    10,
                )
                self._gripper_publishers[side] = self._node.create_publisher(
                    JointState,
                    f"/motion_target/target_position_gripper_{side}",
                    10,
                )
                self._ik_pose_publishers[side] = self._node.create_publisher(
                    PoseStamped,
                    f"/capx/relaxed_ik/target_pose_arm_{side}",
                    10,
                )
                self._ik_reset_publishers[side] = self._node.create_publisher(
                    JointState,
                    f"/relaxed_ik/reset_{side}",
                    10,
                )
                self._node.create_subscription(
                    JointState,
                    f"/hdas/feedback_arm_{side}",
                    lambda msg, arm=side: on_feedback(
                        arm,
                        msg.position,
                        msg.velocity,
                        msg.effort,
                    ),
                    100,
                )
                self._node.create_subscription(
                    MotorControl,
                    f"/motion_control/control_arm_{side}",
                    lambda msg, arm=side: on_control(arm, msg.p_des),
                    qos_profile_sensor_data,
                )
                self._node.create_subscription(
                    JointState,
                    f"/relaxed_ik/isolated_target_joint_state_arm_{side}",
                    lambda msg, arm=side: self._on_ik_result(
                        arm,
                        msg.position,
                    ),
                    10,
                )
                self._node.create_subscription(
                    PoseStamped,
                    f"/motion_control/pose_ee_arm_{side}",
                    lambda msg, arm=side: on_pose(
                        arm,
                        (
                            msg.pose.position.x,
                            msg.pose.position.y,
                            msg.pose.position.z,
                        ),
                        (
                            msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z,
                            msg.pose.orientation.w,
                        ),
                    ),
                    qos_profile_sensor_data,
                )
                self._node.create_subscription(
                    JointState,
                    f"/hdas/feedback_gripper_{side}",
                    lambda msg, arm=side: on_gripper(
                        arm,
                        float(msg.position[0]),
                    )
                    if msg.position
                    else None,
                    qos_profile_sensor_data,
                )
        except Exception:
            self.close()
            raise

    def publish_pair(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> None:
        if self._node is None or self._joint_state_type is None:
            raise RuntimeError("ROS bimanual transport is not started")
        messages = {}
        for side, command in zip(SIDES, (left, right)):
            message = self._joint_state_type()
            message.position = [float(value) for value in command]
            message.velocity = [1.6, 1.6, 1.6, 4.0, 4.0, 4.0]
            messages[side] = message
        self._publishers["left"].publish(messages["left"])
        self._publishers["right"].publish(messages["right"])

    def spin_once(self, timeout_s: float) -> None:
        if self._executor is None:
            raise RuntimeError("ROS bimanual transport is not started")
        for side, topic in self._target_topics.items():
            infos = self._node.get_publishers_info_by_topic(topic)
            # Discovery can briefly expose this node's freshly-created endpoint
            # without identity. start() already proved the topic had no publisher
            # before creating it, so only identified non-Mux nodes are foreign.
            foreign = [
                info
                for info in infos
                if info.node_name != "_NODE_NAME_UNKNOWN_"
                and info.node_name not in self._trusted_infrastructure_publishers
                and (
                    info.node_name != self.node_name
                    or info.node_namespace != self._node.get_namespace()
                )
            ]
            if foreign:
                identities = [
                    f"{info.node_namespace.rstrip('/')}/{info.node_name}"
                    for info in foreign
                ]
                raise RuntimeError(
                    f"foreign publisher detected on {side} target: {identities}"
                )
        self._executor.spin_once(timeout_sec=timeout_s)

    def publish_gripper_pair(self, left: float, right: float) -> None:
        if self._joint_state_type is None:
            raise RuntimeError("ROS bimanual transport is not started")
        for side, value in zip(SIDES, (left, right)):
            message = self._joint_state_type()
            message.name = [f"{side}_gripper"]
            message.position = [float(value)]
            self._gripper_publishers[side].publish(message)

    def _on_ik_result(self, side: str, values: Sequence[float]) -> None:
        if len(values) < 6:
            return
        self._ik_sequences[side] += 1
        self._ik_results[side] = (
            self._ik_sequences[side],
            tuple(float(value) for value in values[:6]),
        )

    def request_ik(
        self,
        *,
        side: str,
        position: Sequence[float],
        quaternion_xyzw: Sequence[float],
        seed_joints: Sequence[float],
        timeout_s: float,
        reset: bool,
    ) -> Sequence[float]:
        if self._executor is None or self._pose_stamped_type is None:
            raise RuntimeError("ROS bimanual transport is not started")
        if reset:
            reset_message = self._joint_state_type()
            reset_message.position = [float(value) for value in seed_joints]
            self._ik_reset_publishers[side].publish(reset_message)
            reset_deadline = time.monotonic() + 0.12
            while time.monotonic() < reset_deadline:
                self.spin_once(min(0.01, reset_deadline - time.monotonic()))
        sequence = self._ik_sequences[side]
        message = self._pose_stamped_type()
        message.header.frame_id = f"relaxed_ik_{side}"
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y, message.pose.position.z = [
            float(value) for value in position
        ]
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = [float(value) for value in quaternion_xyzw]
        self._ik_pose_publishers[side].publish(message)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.spin_once(min(0.02, deadline - time.monotonic()))
            result = self._ik_results[side]
            if result is not None and result[0] > sequence:
                return result[1]
        raise TimeoutError(f"{side} isolated RelaxedIK candidate timed out")

    def close(self) -> None:
        if self._executor is not None:
            if self._node is not None:
                self._executor.remove_node(self._node)
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
        self._publishers.clear()
        self._gripper_publishers.clear()
        self._ik_pose_publishers.clear()
        self._ik_reset_publishers.clear()
        self._executor = None
        self._node = None
        self._joint_state_type = None
        self._pose_stamped_type = None


# Short alias for callers that do not need the robot model in the class name.
BimanualRuntime = R1LiteBimanualRuntime


__all__ = [
    "BimanualRuntime",
    "BimanualRuntimeConfig",
    "BimanualRuntimeError",
    "BimanualTransport",
    "FeedbackStaleError",
    "JointArrivalError",
    "LifecycleError",
    "LifecycleState",
    "ProcessLockError",
    "R1LiteBimanualRuntime",
    "RosBimanualTransport",
]
