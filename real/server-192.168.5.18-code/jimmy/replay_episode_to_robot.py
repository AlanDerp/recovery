#!/usr/bin/env python3
"""Safely inspect, shadow, or replay R1 Lite arm targets from one MCAP.

This module never invokes ``ros2 bag play``. It opens the MCAP read-only and
whitelists only the two high-level arm joint target topics. Inspect mode is the
default and creates no ROS entities. Live mode is deliberately difficult to
arm and remains unvalidated on physical hardware until a guarded operator test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from typing import Iterable, Iterator, Mapping, Optional, Sequence


LEFT_ARM_TOPIC = "/motion_target/target_joint_state_arm_left"
RIGHT_ARM_TOPIC = "/motion_target/target_joint_state_arm_right"
ARM_TOPICS = (LEFT_ARM_TOPIC, RIGHT_ARM_TOPIC)
ARM_FEEDBACK_TOPICS = {
    LEFT_ARM_TOPIC: "/hdas/feedback_arm_left",
    RIGHT_ARM_TOPIC: "/hdas/feedback_arm_right",
}
LOW_LEVEL_ARM_TOPICS = {
    LEFT_ARM_TOPIC: "/motion_control/control_arm_left",
    RIGHT_ARM_TOPIC: "/motion_control/control_arm_right",
}
ARM_POSE_FEEDBACK_TOPICS = {
    LEFT_ARM_TOPIC: "/motion_control/pose_ee_arm_left",
    RIGHT_ARM_TOPIC: "/motion_control/pose_ee_arm_right",
}
EXPECTED_JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
EXPECTED_MOTOR_CONTROL_TYPE = "hdas_msg/msg/MotorControl"
EXPECTED_POSE_TYPE = "geometry_msgs/msg/PoseStamped"
REPLAYER_NODE_NAME = "jimmy_episode_replayer"
LIVE_ARM_TOKENS = {
    "left": "REPLAY_R1_LITE_LEFT_ARM",
    "right": "REPLAY_R1_LITE_RIGHT_ARM",
    "both": "REPLAY_R1_LITE_BOTH_ARMS",
}


def selected_arm_topics(selection: str) -> tuple[str, ...]:
    if selection == "left":
        return (LEFT_ARM_TOPIC,)
    if selection == "right":
        return (RIGHT_ARM_TOPIC,)
    if selection == "both":
        return ARM_TOPICS
    raise ReplaySafetyError(f"unsupported arm selection: {selection!r}")

# Any unexpected source on these inputs can move some part of the robot while
# an arm replay is running. moca_adapter's internal arm/torso target publishers
# are explicitly classified in graph checks instead of relying on raw counts.
WATCHED_COMMAND_TYPES = {
    "/controller": "hdas_msg/msg/ControllerSignalStamped",
    LEFT_ARM_TOPIC: EXPECTED_JOINT_STATE_TYPE,
    RIGHT_ARM_TOPIC: EXPECTED_JOINT_STATE_TYPE,
    "/motion_target/target_joint_state_torso": EXPECTED_JOINT_STATE_TYPE,
    "/motion_target/target_pose_arm_left": "geometry_msgs/msg/PoseStamped",
    "/motion_target/target_pose_arm_right": "geometry_msgs/msg/PoseStamped",
    "/motion_target/target_pose_torso": "geometry_msgs/msg/PoseStamped",
    "/motion_target/target_position_gripper_left": EXPECTED_JOINT_STATE_TYPE,
    "/motion_target/target_position_gripper_right": EXPECTED_JOINT_STATE_TYPE,
    "/motion_target/target_speed_chassis": "geometry_msgs/msg/TwistStamped",
    "/motion_target/chassis_acc_limit": "geometry_msgs/msg/TwistStamped",
    "/motion_target/brake_mode": "std_msgs/msg/Bool",
}
INTERNAL_MOCA_TARGET_TOPICS = {
    LEFT_ARM_TOPIC,
    RIGHT_ARM_TOPIC,
    "/motion_target/target_joint_state_torso",
}
EXPECTED_VENDOR_COMMAND_PUBLISHERS = {
    "/controller": {"HDAS"},
}


class ReplaySafetyError(RuntimeError):
    """A fail-closed validation or runtime safety failure."""


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ReplaySafetyError(f"{label} contains NaN or Inf")
    return converted


def _string_tuple(values: Sequence[object], label: str) -> tuple[str, ...]:
    converted = tuple(str(value) for value in values)
    if any(not value or any(character.isspace() for character in value) for value in converted):
        raise ReplaySafetyError(f"{label} contains an empty or whitespace joint name")
    if len(set(converted)) != len(converted):
        raise ReplaySafetyError(f"{label} contains duplicate joint names")
    return converted


@dataclass(frozen=True)
class TopicLimits:
    topic: str
    feedback_topic: str
    low_level_topic: str
    joint_names: tuple[str, ...]
    position_min: tuple[float, ...]
    position_max: tuple[float, ...]
    controller_max_velocity: tuple[float, ...]
    command_velocity: tuple[float, ...]
    position_margin_rad: float
    controller_input_clamp_tolerance_rad: float

    def __post_init__(self) -> None:
        dimension = len(self.joint_names)
        if self.topic not in ARM_TOPICS:
            raise ReplaySafetyError(f"unsupported command topic in limits: {self.topic}")
        if dimension != 6:
            raise ReplaySafetyError(f"{self.topic} must declare exactly 6 joints")
        vectors = (
            self.position_min,
            self.position_max,
            self.controller_max_velocity,
            self.command_velocity,
        )
        if any(len(vector) != dimension for vector in vectors):
            raise ReplaySafetyError(f"limit-vector length mismatch for {self.topic}")
        if self.position_margin_rad < 0.0:
            raise ReplaySafetyError("position_margin_rad must be non-negative")
        if not 0.0 <= self.controller_input_clamp_tolerance_rad <= 0.05:
            raise ReplaySafetyError(
                "controller_input_clamp_tolerance_rad must be between 0 and 0.05"
            )
        for index, (minimum, maximum, controller_velocity, command_velocity) in enumerate(
            zip(*vectors)
        ):
            if minimum + 2.0 * self.position_margin_rad >= maximum:
                raise ReplaySafetyError(
                    f"joint {index + 1} has no usable range after its position margin"
                )
            if controller_velocity <= 0.0 or command_velocity <= 0.0:
                raise ReplaySafetyError(f"joint {index + 1} velocity must be positive")
            if command_velocity > controller_velocity:
                raise ReplaySafetyError(
                    f"joint {index + 1} command velocity exceeds controller limit"
                )

    @property
    def dimension(self) -> int:
        return len(self.joint_names)

    def validate_position(self, position: Sequence[float], context: str) -> tuple[float, ...]:
        """Validate controller-space input and normalize it into the safe URDF range."""

        vector = _finite_vector(position, f"{context} position")
        if len(vector) != self.dimension:
            raise ReplaySafetyError(
                f"{context} has {len(vector)} positions; expected {self.dimension}"
            )
        normalized = []
        for index, (value, minimum, maximum) in enumerate(
            zip(vector, self.position_min, self.position_max)
        ):
            safe_minimum = minimum + self.position_margin_rad
            safe_maximum = maximum - self.position_margin_rad
            accepted_minimum = minimum - self.controller_input_clamp_tolerance_rad
            accepted_maximum = maximum + self.controller_input_clamp_tolerance_rad
            if not accepted_minimum <= value <= accepted_maximum:
                raise ReplaySafetyError(
                    f"{context} joint {index + 1}={value:.6f} is outside "
                    f"the reviewed controller-input envelope "
                    f"[{accepted_minimum:.6f}, {accepted_maximum:.6f}]"
                )
            normalized.append(min(max(value, safe_minimum), safe_maximum))
        return tuple(normalized)


@dataclass(frozen=True)
class SafetyConfig:
    path: Path
    schema_version: int
    robot_name: str
    robot_serial: str
    robot_description_path: Path
    robot_description_sha256: str
    controller_config_path: Path
    controller_config_sha256: str
    controller_command_rate_hz: float
    expected_self_collision_check: bool
    workspace_margin_m: float
    max_workspace_radius_m: float
    topics: Mapping[str, TopicLimits]


def _require_sha256(value: object, label: str) -> str:
    text = str(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ReplaySafetyError(f"{label} is not a SHA-256 digest")
    return text


def load_safety_config(path: Path) -> SafetyConfig:
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ReplaySafetyError(f"cannot read safety config {resolved}: {error}") from error

    if payload.get("schema_version") != 1:
        raise ReplaySafetyError("unsupported safety-config schema_version")
    robot = payload.get("robot")
    controller = payload.get("controller")
    topics_payload = payload.get("topics")
    if not isinstance(robot, dict) or not isinstance(controller, dict):
        raise ReplaySafetyError("safety config is missing robot/controller objects")
    if not isinstance(topics_payload, dict) or set(topics_payload) != set(ARM_TOPICS):
        raise ReplaySafetyError("safety config must contain exactly the two arm topics")

    topic_limits: dict[str, TopicLimits] = {}
    for topic in ARM_TOPICS:
        item = topics_payload[topic]
        if not isinstance(item, dict):
            raise ReplaySafetyError(f"limits for {topic} must be an object")
        limits = TopicLimits(
            topic=topic,
            feedback_topic=str(item.get("feedback_topic", "")),
            low_level_topic=str(item.get("low_level_topic", "")),
            joint_names=_string_tuple(item.get("joint_names", []), f"{topic} joint_names"),
            position_min=_finite_vector(item.get("position_min", []), f"{topic} minimum"),
            position_max=_finite_vector(item.get("position_max", []), f"{topic} maximum"),
            controller_max_velocity=_finite_vector(
                item.get("controller_max_velocity", []), f"{topic} controller velocity"
            ),
            command_velocity=_finite_vector(
                item.get("command_velocity", []), f"{topic} command velocity"
            ),
            position_margin_rad=float(item.get("position_margin_rad", -1.0)),
            controller_input_clamp_tolerance_rad=float(
                item.get("controller_input_clamp_tolerance_rad", -1.0)
            ),
        )
        if limits.feedback_topic != ARM_FEEDBACK_TOPICS[topic]:
            raise ReplaySafetyError(f"unexpected feedback topic for {topic}")
        if limits.low_level_topic != LOW_LEVEL_ARM_TOPICS[topic]:
            raise ReplaySafetyError(f"unexpected low-level topic for {topic}")
        topic_limits[topic] = limits

    config = SafetyConfig(
        path=resolved,
        schema_version=1,
        robot_name=str(robot.get("name", "")),
        robot_serial=str(robot.get("serial", "")),
        robot_description_path=Path(str(robot.get("description_path", ""))),
        robot_description_sha256=_require_sha256(
            robot.get("description_sha256", ""), "robot description digest"
        ),
        controller_config_path=Path(str(controller.get("config_path", ""))),
        controller_config_sha256=_require_sha256(
            controller.get("config_sha256", ""), "controller config digest"
        ),
        controller_command_rate_hz=float(controller.get("command_rate_hz", 0.0)),
        expected_self_collision_check=bool(controller.get("self_collision_check", False)),
        workspace_margin_m=float(controller.get("workspace_margin_m", -1.0)),
        max_workspace_radius_m=float(controller.get("max_workspace_radius_m", 0.0)),
        topics=topic_limits,
    )
    if not config.robot_name or not config.robot_serial:
        raise ReplaySafetyError("safety config robot name/serial must be non-empty")
    if not config.robot_description_path.is_absolute():
        raise ReplaySafetyError("robot description path must be absolute")
    if not config.controller_config_path.is_absolute():
        raise ReplaySafetyError("controller config path must be absolute")
    if (
        not math.isfinite(config.controller_command_rate_hz)
        or config.controller_command_rate_hz <= 0
    ):
        raise ReplaySafetyError("controller command rate must be finite and positive")
    if not 0.0 <= config.workspace_margin_m <= 0.20:
        raise ReplaySafetyError("workspace margin must be between 0 and 0.20 metres")
    if not 0.0 < config.max_workspace_radius_m <= 5.0:
        raise ReplaySafetyError("maximum workspace radius must be in (0, 5] metres")
    return config


@dataclass(frozen=True)
class RecordedCommand:
    topic: str
    record_time_ns: int
    joint_names: tuple[str, ...]
    source_position: tuple[float, ...]
    position: tuple[float, ...]
    recorded_velocity: tuple[float, ...]


@dataclass(frozen=True)
class RecordedPose:
    topic: str
    record_time_ns: int
    frame_id: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class WorkspaceEnvelope:
    pose_topic: str
    frame_id: str
    recorded_minimum: tuple[float, float, float]
    recorded_maximum: tuple[float, float, float]
    margin_m: float
    sample_count: int

    @property
    def minimum(self) -> tuple[float, float, float]:
        return tuple(value - self.margin_m for value in self.recorded_minimum)

    @property
    def maximum(self) -> tuple[float, float, float]:
        return tuple(value + self.margin_m for value in self.recorded_maximum)

    def validate(self, frame_id: str, position: Sequence[float], context: str) -> None:
        if frame_id != self.frame_id:
            raise ReplaySafetyError(
                f"{context} frame {frame_id!r} does not match recorded {self.frame_id!r}"
            )
        vector = _finite_vector(position, f"{context} Cartesian position")
        if len(vector) != 3:
            raise ReplaySafetyError(f"{context} Cartesian position must have 3 values")
        for axis, value, minimum, maximum in zip("xyz", vector, self.minimum, self.maximum):
            if not minimum <= value <= maximum:
                raise ReplaySafetyError(
                    f"{context} {axis}={value:.6f} m is outside demonstrated "
                    f"workspace [{minimum:.6f}, {maximum:.6f}]"
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "pose_topic": self.pose_topic,
            "frame_id": self.frame_id,
            "recorded_minimum": list(self.recorded_minimum),
            "recorded_maximum": list(self.recorded_maximum),
            "margin_m": self.margin_m,
            "enforced_minimum": list(self.minimum),
            "enforced_maximum": list(self.maximum),
            "sample_count": self.sample_count,
        }


def make_recorded_pose(
    *,
    topic: str,
    record_time_ns: int,
    frame_id: str,
    position: Sequence[float],
    orientation: Sequence[float],
    max_workspace_radius_m: float,
    context: str,
) -> RecordedPose:
    if topic not in ARM_POSE_FEEDBACK_TOPICS.values():
        raise ReplaySafetyError(f"unsupported pose topic: {topic}")
    if not frame_id or any(character.isspace() for character in frame_id):
        raise ReplaySafetyError(f"{context} has an empty or invalid frame_id")
    position_vector = _finite_vector(position, f"{context} position")
    orientation_vector = _finite_vector(orientation, f"{context} orientation")
    if len(position_vector) != 3 or len(orientation_vector) != 4:
        raise ReplaySafetyError(f"{context} has an invalid pose shape")
    quaternion_norm = math.sqrt(sum(value * value for value in orientation_vector))
    if not 0.95 <= quaternion_norm <= 1.05:
        raise ReplaySafetyError(
            f"{context} quaternion norm {quaternion_norm:.6f} is not near one"
        )
    radius = math.sqrt(sum(value * value for value in position_vector))
    if radius > max_workspace_radius_m:
        raise ReplaySafetyError(
            f"{context} radius {radius:.6f} m exceeds {max_workspace_radius_m:.6f} m"
        )
    return RecordedPose(
        topic=topic,
        record_time_ns=int(record_time_ns),
        frame_id=frame_id,
        position=(position_vector[0], position_vector[1], position_vector[2]),
    )


def build_workspace_envelope(
    poses: Sequence[RecordedPose], margin_m: float
) -> WorkspaceEnvelope:
    if not poses:
        raise ReplaySafetyError("cannot build a workspace envelope without poses")
    frames = {pose.frame_id for pose in poses}
    if len(frames) != 1:
        raise ReplaySafetyError(f"recorded pose frames changed: {sorted(frames)}")
    minimum = list(poses[0].position)
    maximum = list(poses[0].position)
    previous_time_ns = poses[0].record_time_ns
    for pose in poses:
        if pose.record_time_ns < previous_time_ns:
            raise ReplaySafetyError(f"timestamps move backwards on {pose.topic}")
        previous_time_ns = pose.record_time_ns
        for index, value in enumerate(pose.position):
            minimum[index] = min(minimum[index], value)
            maximum[index] = max(maximum[index], value)
    return WorkspaceEnvelope(
        pose_topic=poses[0].topic,
        frame_id=poses[0].frame_id,
        recorded_minimum=(minimum[0], minimum[1], minimum[2]),
        recorded_maximum=(maximum[0], maximum[1], maximum[2]),
        margin_m=margin_m,
        sample_count=len(poses),
    )


def make_recorded_command(
    *,
    topic: str,
    record_time_ns: int,
    joint_names: Sequence[object],
    position: Sequence[float],
    velocity: Sequence[float],
    limits: TopicLimits,
    context: str,
) -> RecordedCommand:
    if topic != limits.topic:
        raise ReplaySafetyError(f"{context} topic does not match its limits")
    if int(record_time_ns) < 0:
        raise ReplaySafetyError(f"{context} has a negative record timestamp")
    names = tuple(str(value) for value in joint_names)
    if names:
        names = _string_tuple(names, f"{context} names")
        if names != limits.joint_names:
            raise ReplaySafetyError(
                f"{context} joint order {names!r} does not match {limits.joint_names!r}"
            )
    source_positions = _finite_vector(position, f"{context} position")
    positions = limits.validate_position(source_positions, context)
    velocities = _finite_vector(velocity, f"{context} velocity")
    if velocities and len(velocities) != limits.dimension:
        raise ReplaySafetyError(
            f"{context} has {len(velocities)} velocities; expected 0 or {limits.dimension}"
        )
    return RecordedCommand(
        topic=topic,
        record_time_ns=int(record_time_ns),
        joint_names=names,
        source_position=source_positions,
        position=positions,
        recorded_velocity=velocities,
    )


@dataclass(frozen=True)
class TopicStats:
    count: int
    first_time_ns: int
    last_time_ns: int
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    source_minimum: tuple[float, ...]
    source_maximum: tuple[float, ...]
    normalized_sample_count: int
    max_normalization_rad: tuple[float, ...]
    max_adjacent_step: tuple[float, ...]
    max_gap_seconds: float
    messages_with_names: int
    messages_with_velocity: int

    def as_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "first_time_ns": self.first_time_ns,
            "last_time_ns": self.last_time_ns,
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "source_minimum": list(self.source_minimum),
            "source_maximum": list(self.source_maximum),
            "normalized_sample_count": self.normalized_sample_count,
            "max_normalization_rad": list(self.max_normalization_rad),
            "max_adjacent_step": list(self.max_adjacent_step),
            "max_gap_seconds": self.max_gap_seconds,
            "messages_with_names": self.messages_with_names,
            "messages_with_velocity": self.messages_with_velocity,
        }


def build_topic_stats(commands: Sequence[RecordedCommand]) -> TopicStats:
    if not commands:
        raise ReplaySafetyError("cannot build statistics for an empty command stream")
    dimension = len(commands[0].position)
    minimum = list(commands[0].position)
    maximum = list(commands[0].position)
    source_minimum = list(commands[0].source_position)
    source_maximum = list(commands[0].source_position)
    normalized_count = 0
    max_normalization = [0.0] * dimension
    max_step = [0.0] * dimension
    max_gap_ns = 0
    named = 0
    with_velocity = 0
    previous = commands[0]
    for index, command in enumerate(commands):
        if index and command.record_time_ns < previous.record_time_ns:
            raise ReplaySafetyError(f"timestamps move backwards on {command.topic}")
        if command.joint_names:
            named += 1
        if command.recorded_velocity:
            with_velocity += 1
        for joint, value in enumerate(command.position):
            minimum[joint] = min(minimum[joint], value)
            maximum[joint] = max(maximum[joint], value)
            source_value = command.source_position[joint]
            source_minimum[joint] = min(source_minimum[joint], source_value)
            source_maximum[joint] = max(source_maximum[joint], source_value)
            max_normalization[joint] = max(
                max_normalization[joint], abs(value - source_value)
            )
            if index:
                max_step[joint] = max(
                    max_step[joint], abs(value - previous.position[joint])
                )
        if index:
            max_gap_ns = max(max_gap_ns, command.record_time_ns - previous.record_time_ns)
        if command.position != command.source_position:
            normalized_count += 1
        previous = command
    return TopicStats(
        count=len(commands),
        first_time_ns=commands[0].record_time_ns,
        last_time_ns=commands[-1].record_time_ns,
        minimum=tuple(minimum),
        maximum=tuple(maximum),
        source_minimum=tuple(source_minimum),
        source_maximum=tuple(source_maximum),
        normalized_sample_count=normalized_count,
        max_normalization_rad=tuple(max_normalization),
        max_adjacent_step=tuple(max_step),
        max_gap_seconds=max_gap_ns / 1_000_000_000,
        messages_with_names=named,
        messages_with_velocity=with_velocity,
    )


@dataclass(frozen=True)
class ReplayPlan:
    source: Path
    source_size_bytes: int
    topics: tuple[str, ...]
    events: Mapping[str, tuple[RecordedCommand, ...]]
    stats: Mapping[str, TopicStats]
    workspace: Mapping[str, WorkspaceEnvelope]
    start_time_ns: int
    end_time_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns

    @property
    def duration_seconds(self) -> float:
        return self.duration_ns / 1_000_000_000

    @property
    def first_commands(self) -> Mapping[str, RecordedCommand]:
        return {topic: self.events[topic][0] for topic in self.topics}


def validate_source_path(path: Path) -> Path:
    source = path.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".mcap":
        raise ReplaySafetyError(f"source is not an MCAP file: {source}")
    return source


def load_replay_plan(
    source: Path, safety: SafetyConfig, topics: Sequence[str] = ARM_TOPICS
) -> ReplayPlan:
    """Read and validate only whitelisted arm targets from the MCAP."""

    selected = tuple(topics)
    if not selected or len(set(selected)) != len(selected):
        raise ReplaySafetyError("selected arm topics must be non-empty and unique")
    if any(topic not in ARM_TOPICS for topic in selected):
        raise ReplaySafetyError("selected arm topics contain an unsupported topic")

    try:
        import rosbag2_py
        from geometry_msgs.msg import PoseStamped
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import JointState
    except ImportError as error:
        raise ReplaySafetyError(
            "ROS 2 Python modules are required to inspect an MCAP; "
            "source the R1 ROS 2 and Galaxea setup files"
        ) from error

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    for topic in selected:
        if topic_types.get(topic) != EXPECTED_JOINT_STATE_TYPE:
            raise ReplaySafetyError(
                f"expected {EXPECTED_JOINT_STATE_TYPE} on {topic}; "
                f"found {topic_types.get(topic)!r}"
            )
        pose_topic = ARM_POSE_FEEDBACK_TOPICS[topic]
        if topic_types.get(pose_topic) != EXPECTED_POSE_TYPE:
            raise ReplaySafetyError(
                f"expected {EXPECTED_POSE_TYPE} on {pose_topic}; "
                f"found {topic_types.get(pose_topic)!r}"
            )
    selected_topics = list(selected) + [ARM_POSE_FEEDBACK_TOPICS[topic] for topic in selected]
    reader.set_filter(rosbag2_py.StorageFilter(topics=selected_topics))

    mutable_events: dict[str, list[RecordedCommand]] = {topic: [] for topic in selected}
    mutable_poses: dict[str, list[RecordedPose]] = {
        ARM_POSE_FEEDBACK_TOPICS[topic]: [] for topic in selected
    }
    while reader.has_next():
        topic, serialized, record_time_ns = reader.read_next()
        if topic in mutable_events:
            message = deserialize_message(serialized, JointState)
            index = len(mutable_events[topic])
            command = make_recorded_command(
                topic=topic,
                record_time_ns=record_time_ns,
                joint_names=message.name,
                position=message.position,
                velocity=message.velocity,
                limits=safety.topics[topic],
                context=f"{topic} message {index}",
            )
            mutable_events[topic].append(command)
        elif topic in mutable_poses:
            message = deserialize_message(serialized, PoseStamped)
            index = len(mutable_poses[topic])
            pose = make_recorded_pose(
                topic=topic,
                record_time_ns=record_time_ns,
                frame_id=message.header.frame_id,
                position=(
                    message.pose.position.x,
                    message.pose.position.y,
                    message.pose.position.z,
                ),
                orientation=(
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                    message.pose.orientation.w,
                ),
                max_workspace_radius_m=safety.max_workspace_radius_m,
                context=f"{topic} message {index}",
            )
            mutable_poses[topic].append(pose)

    for topic, commands in mutable_events.items():
        if not commands:
            raise ReplaySafetyError(f"MCAP has no replayable messages on {topic}")
    for topic, poses in mutable_poses.items():
        if not poses:
            raise ReplaySafetyError(f"MCAP has no workspace evidence on {topic}")
    events = {topic: tuple(commands) for topic, commands in mutable_events.items()}
    stats = {topic: build_topic_stats(commands) for topic, commands in events.items()}
    workspace = {
        command_topic: build_workspace_envelope(
            mutable_poses[ARM_POSE_FEEDBACK_TOPICS[command_topic]],
            safety.workspace_margin_m,
        )
        for command_topic in selected
    }
    start_time_ns = min(stat.first_time_ns for stat in stats.values())
    end_time_ns = max(stat.last_time_ns for stat in stats.values())
    if end_time_ns <= start_time_ns:
        raise ReplaySafetyError("replay duration is zero or negative")
    return ReplayPlan(
        source=source,
        source_size_bytes=source.stat().st_size,
        topics=selected,
        events=events,
        stats=stats,
        workspace=workspace,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
    )


class ReplayCursor:
    """Linearly resample selected topic streams without extrapolation."""

    def __init__(self, plan: ReplayPlan):
        self.plan = plan
        self.indices = {topic: 1 for topic in plan.topics}

    def advance(self, virtual_elapsed_ns: int) -> Mapping[str, RecordedCommand]:
        absolute_time_ns = self.plan.start_time_ns + max(0, int(virtual_elapsed_ns))
        current: dict[str, RecordedCommand] = {}
        for topic in self.plan.topics:
            events = self.plan.events[topic]
            index = self.indices[topic]
            while index < len(events) and events[index].record_time_ns <= absolute_time_ns:
                index += 1
            self.indices[topic] = index
            if index == 1 and absolute_time_ns <= events[0].record_time_ns:
                current[topic] = events[0]
                continue
            if index >= len(events):
                current[topic] = events[-1]
                continue
            previous = events[index - 1]
            following = events[index]
            interval_ns = following.record_time_ns - previous.record_time_ns
            if interval_ns <= 0:
                current[topic] = following
                continue
            fraction = min(
                1.0,
                max(0.0, (absolute_time_ns - previous.record_time_ns) / interval_ns),
            )

            def interpolate(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
                return tuple(
                    left_value + fraction * (right_value - left_value)
                    for left_value, right_value in zip(left, right)
                )

            current[topic] = RecordedCommand(
                topic=topic,
                record_time_ns=absolute_time_ns,
                joint_names=previous.joint_names or following.joint_names,
                source_position=interpolate(
                    previous.source_position, following.source_position
                ),
                position=interpolate(previous.position, following.position),
                recorded_velocity=(),
            )
        return current


def iter_schedule(
    plan: ReplayPlan, *, rate_hz: float, speed: float, start_offset_seconds: float = 0.0
) -> Iterator[tuple[float, int, Mapping[str, RecordedCommand]]]:
    if rate_hz <= 0.0 or speed <= 0.0:
        raise ReplaySafetyError("rate and speed must be positive")
    if not 0.0 <= start_offset_seconds < plan.duration_seconds:
        raise ReplaySafetyError("start offset must be within the replay duration")
    start_offset_ns = int(round(start_offset_seconds * 1_000_000_000))
    remaining_duration_ns = plan.duration_ns - start_offset_ns
    wall_duration = (remaining_duration_ns / 1_000_000_000) / speed
    tick_count = int(math.ceil(wall_duration * rate_hz))
    if tick_count > 10_000_000:
        raise ReplaySafetyError("requested replay schedule is unreasonably large")
    cursor = ReplayCursor(plan)
    for tick in range(tick_count + 1):
        wall_elapsed = min(wall_duration, tick / rate_hz)
        virtual_elapsed_ns = min(
            plan.duration_ns,
            start_offset_ns + int(round(wall_elapsed * speed * 1_000_000_000)),
        )
        yield wall_elapsed, virtual_elapsed_ns, cursor.advance(virtual_elapsed_ns)


def validate_resampled_schedule(
    plan: ReplayPlan,
    safety: SafetyConfig,
    *,
    rate_hz: float,
    speed: float,
    step_slack_rad: float,
    start_offset_seconds: float = 0.0,
) -> dict[str, object]:
    if rate_hz > safety.controller_command_rate_hz:
        raise ReplaySafetyError(
            f"replay rate {rate_hz:g} exceeds controller rate "
            f"{safety.controller_command_rate_hz:g} Hz"
        )
    if step_slack_rad < 0.0:
        raise ReplaySafetyError("step slack must be non-negative")

    previous: Optional[Mapping[str, RecordedCommand]] = None
    max_observed = {topic: [0.0] * 6 for topic in plan.topics}
    ticks = 0
    last_wall_elapsed = 0.0
    for wall_elapsed, _, commands in iter_schedule(
        plan,
        rate_hz=rate_hz,
        speed=speed,
        start_offset_seconds=start_offset_seconds,
    ):
        if previous is not None:
            for topic in plan.topics:
                limits = safety.topics[topic]
                for index, (current_value, previous_value, velocity) in enumerate(
                    zip(
                        commands[topic].position,
                        previous[topic].position,
                        limits.command_velocity,
                    )
                ):
                    step = abs(current_value - previous_value)
                    max_observed[topic][index] = max(max_observed[topic][index], step)
                    allowed = velocity / rate_hz + step_slack_rad
                    if step > allowed:
                        raise ReplaySafetyError(
                            f"resampled {topic} joint {index + 1} step {step:.6f} rad "
                            f"exceeds {allowed:.6f} rad at speed={speed:g}, rate={rate_hz:g}"
                        )
        previous = commands
        ticks += 1
        last_wall_elapsed = wall_elapsed
    return {
        "tick_count": ticks,
        "wall_duration_seconds": last_wall_elapsed,
        "rate_hz": rate_hz,
        "speed": speed,
        "start_offset_seconds": start_offset_seconds,
        "max_resampled_step": {
            topic: values for topic, values in max_observed.items()
        },
    }


def maximum_position_error(
    desired: Sequence[float], actual: Sequence[float]
) -> float:
    if len(desired) != len(actual):
        raise ReplaySafetyError("cannot compare position vectors with different lengths")
    desired_vector = _finite_vector(desired, "desired position")
    actual_vector = _finite_vector(actual, "feedback position")
    return max(abs(left - right) for left, right in zip(desired_vector, actual_vector))


def extract_feedback_position(
    names: Sequence[object],
    positions: Sequence[float],
    expected_names: Sequence[str],
) -> tuple[float, ...]:
    position_vector = _finite_vector(positions, "feedback position")
    observed_names = tuple(str(name) for name in names)
    if observed_names:
        expected_group_name = "left_arm" if expected_names[0].startswith("left_") else "right_arm"
        if observed_names == (expected_group_name,) and len(position_vector) == len(
            expected_names
        ):
            return position_vector
        if len(observed_names) != len(position_vector):
            raise ReplaySafetyError("feedback names/position lengths differ")
        by_name = dict(zip(observed_names, position_vector))
        if all(name in by_name for name in expected_names):
            return tuple(by_name[name] for name in expected_names)
        raise ReplaySafetyError("feedback joint names do not contain the configured arm order")
    if len(position_vector) < len(expected_names):
        raise ReplaySafetyError("feedback position vector is shorter than the arm dimension")
    return tuple(position_vector[: len(expected_names)])


class AtomicManifest:
    def __init__(self, path: Path, initial: Mapping[str, object]):
        self.path = path.expanduser().resolve()
        self.data = dict(initial)
        self.write()

    def update(self, **changes: object) -> None:
        self.data.update(changes)
        self.data["updated_at"] = utc_timestamp()
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def _endpoint_node_name(info: object) -> str:
    return str(getattr(info, "node_name", "")).lstrip("/")


def _endpoint_types(infos: Iterable[object]) -> set[str]:
    return {
        str(getattr(info, "topic_type", ""))
        for info in infos
        if getattr(info, "topic_type", "")
    }


def verify_runtime_file_identities(safety: SafetyConfig) -> None:
    identities = (
        (
            safety.robot_description_path,
            safety.robot_description_sha256,
            "robot description",
        ),
        (safety.controller_config_path, safety.controller_config_sha256, "controller config"),
    )
    for path, expected_digest, label in identities:
        if not path.is_file():
            raise ReplaySafetyError(f"{label} is missing: {path}")
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            raise ReplaySafetyError(
                f"{label} checksum mismatch for {path}: "
                f"expected {expected_digest}, found {actual_digest}"
            )


def run_ros_mode(
    args: argparse.Namespace,
    plan: ReplayPlan,
    safety: SafetyConfig,
    source_sha256: Optional[str],
    manifest: AtomicManifest,
) -> int:
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rcl_interfaces.srv import GetParameters
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
    except ImportError as error:
        raise ReplaySafetyError(
            "ROS 2 Python modules are required for shadow/live mode"
        ) from error

    start_offset_ns = int(round(args.start_offset_seconds * 1_000_000_000))
    start_commands = ReplayCursor(plan).advance(start_offset_ns)

    class ReplaySafetyNode(Node):
        def __init__(self) -> None:
            super().__init__(REPLAYER_NODE_NAME)
            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            qos.history = HistoryPolicy.KEEP_LAST
            self.feedback: dict[str, tuple[float, tuple[float, ...]]] = {}
            self.ee_pose: dict[
                str, tuple[float, str, tuple[float, float, float]]
            ] = {}
            self.feedback_lock = threading.Lock()
            self.fatal_error: Optional[str] = None
            self.arm_command_publishers: dict[str, object] = {}
            self.last_graph_check = 0.0
            self.tracking_violation_since: Optional[float] = None
            self.parameter_client = self.create_client(
                GetParameters, "/moca_adapter/get_parameters"
            )
            self.feedback_subscriptions = []
            for command_topic in plan.topics:
                feedback_topic = safety.topics[command_topic].feedback_topic

                def callback(message: object, topic: str = command_topic) -> None:
                    try:
                        position = extract_feedback_position(
                            getattr(message, "name"),
                            getattr(message, "position"),
                            safety.topics[topic].joint_names,
                        )
                        safety.topics[topic].validate_position(
                            position, f"live {topic} feedback"
                        )
                        with self.feedback_lock:
                            self.feedback[topic] = (time.monotonic(), position)
                    except Exception as error:
                        self.fatal_error = f"invalid {topic} feedback: {error}"

                self.feedback_subscriptions.append(
                    self.create_subscription(JointState, feedback_topic, callback, qos)
                )

                pose_topic = ARM_POSE_FEEDBACK_TOPICS[command_topic]

                def pose_callback(message: object, topic: str = command_topic) -> None:
                    try:
                        frame_id = str(getattr(getattr(message, "header"), "frame_id"))
                        pose = getattr(message, "pose")
                        position = _finite_vector(
                            (pose.position.x, pose.position.y, pose.position.z),
                            f"live {topic} end-effector position",
                        )
                        orientation = _finite_vector(
                            (
                                pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w,
                            ),
                            f"live {topic} end-effector orientation",
                        )
                        quaternion_norm = math.sqrt(
                            sum(value * value for value in orientation)
                        )
                        if not 0.95 <= quaternion_norm <= 1.05:
                            raise ReplaySafetyError(
                                f"quaternion norm {quaternion_norm:.6f} is not near one"
                            )
                        with self.feedback_lock:
                            self.ee_pose[topic] = (
                                time.monotonic(),
                                frame_id,
                                (position[0], position[1], position[2]),
                            )
                    except Exception as error:
                        self.fatal_error = f"invalid {topic} end-effector feedback: {error}"

                self.feedback_subscriptions.append(
                    self.create_subscription(PoseStamped, pose_topic, pose_callback, qos)
                )

        def runtime_parameters(self) -> Mapping[str, str]:
            if not self.parameter_client.wait_for_service(timeout_sec=args.discovery_timeout):
                raise ReplaySafetyError("missing /moca_adapter/get_parameters service")
            request = GetParameters.Request()
            request.names = ["robot_name", "robot_description_file", "config_file"]
            future = self.parameter_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=args.discovery_timeout)
            response = future.result()
            if response is None or len(response.values) != len(request.names):
                raise ReplaySafetyError("moca_adapter parameter query failed")
            return {
                name: value.string_value for name, value in zip(request.names, response.values)
            }

        def graph_problems(self, include_self: bool) -> list[str]:
            problems: list[str] = []
            for topic, expected_type in WATCHED_COMMAND_TYPES.items():
                allowed_publishers = set(EXPECTED_VENDOR_COMMAND_PUBLISHERS.get(topic, set()))
                if topic in INTERNAL_MOCA_TARGET_TOPICS:
                    allowed_publishers.add("moca_adapter")
                if include_self and topic in plan.topics:
                    allowed_publishers.add(REPLAYER_NODE_NAME)
                publishers = self.get_publishers_info_by_topic(topic)
                unexpected_nodes = sorted(
                    {
                        _endpoint_node_name(info)
                        for info in publishers
                        if _endpoint_node_name(info) not in allowed_publishers
                    }
                )
                wrong_types = _endpoint_types(publishers) - {expected_type}
                if unexpected_nodes:
                    problems.append(
                        f"unexpected publisher(s) on {topic}: {unexpected_nodes}"
                    )
                if wrong_types:
                    problems.append(f"unexpected type(s) on {topic}: {sorted(wrong_types)}")

            for topic in plan.topics:
                subscriptions = self.get_subscriptions_info_by_topic(topic)
                subscriber_nodes = {_endpoint_node_name(info) for info in subscriptions}
                if "moca_adapter" not in subscriber_nodes:
                    problems.append(f"moca_adapter is not subscribed to {topic}")
                wrong_types = _endpoint_types(subscriptions) - {EXPECTED_JOINT_STATE_TYPE}
                if wrong_types:
                    problems.append(
                        f"unexpected subscriber type(s) on {topic}: {sorted(wrong_types)}"
                    )

                low_level_topic = safety.topics[topic].low_level_topic
                low_level_publishers = self.get_publishers_info_by_topic(low_level_topic)
                low_level_nodes = [_endpoint_node_name(info) for info in low_level_publishers]
                if low_level_nodes != ["moca_adapter"]:
                    problems.append(
                        f"expected exactly one moca_adapter publisher on {low_level_topic}; "
                        f"found {low_level_nodes}"
                    )
                wrong_low_level_types = _endpoint_types(low_level_publishers) - {
                    EXPECTED_MOTOR_CONTROL_TYPE
                }
                if wrong_low_level_types:
                    problems.append(
                        f"unexpected type(s) on {low_level_topic}: "
                        f"{sorted(wrong_low_level_types)}"
                    )

                feedback_topic = safety.topics[topic].feedback_topic
                feedback_publishers = self.get_publishers_info_by_topic(feedback_topic)
                feedback_nodes = [_endpoint_node_name(info) for info in feedback_publishers]
                if feedback_nodes != ["HDAS"]:
                    problems.append(
                        f"expected exactly one HDAS publisher on {feedback_topic}; "
                        f"found {feedback_nodes}"
                    )
                wrong_feedback_types = _endpoint_types(feedback_publishers) - {
                    EXPECTED_JOINT_STATE_TYPE
                }
                if wrong_feedback_types:
                    problems.append(
                        f"unexpected type(s) on {feedback_topic}: "
                        f"{sorted(wrong_feedback_types)}"
                    )

                pose_topic = ARM_POSE_FEEDBACK_TOPICS[topic]
                pose_publishers = self.get_publishers_info_by_topic(pose_topic)
                pose_nodes = [_endpoint_node_name(info) for info in pose_publishers]
                if pose_nodes != ["moca_adapter"]:
                    problems.append(
                        f"expected exactly one moca_adapter publisher on {pose_topic}; "
                        f"found {pose_nodes}"
                    )
                wrong_pose_types = _endpoint_types(pose_publishers) - {
                    EXPECTED_POSE_TYPE
                }
                if wrong_pose_types:
                    problems.append(
                        f"unexpected type(s) on {pose_topic}: {sorted(wrong_pose_types)}"
                    )
            return problems

        def feedback_snapshot(self) -> Mapping[str, tuple[float, tuple[float, ...]]]:
            with self.feedback_lock:
                return dict(self.feedback)

        def pose_snapshot(
            self,
        ) -> Mapping[str, tuple[float, str, tuple[float, float, float]]]:
            with self.feedback_lock:
                return dict(self.ee_pose)

        def feedback_problems(self) -> list[str]:
            now = time.monotonic()
            snapshot = self.feedback_snapshot()
            pose_snapshot = self.pose_snapshot()
            problems = []
            for topic in plan.topics:
                item = snapshot.get(topic)
                if item is None:
                    problems.append(f"no feedback received for {topic}")
                elif now - item[0] > args.feedback_timeout:
                    problems.append(
                        f"stale feedback for {topic}: {now - item[0]:.3f} seconds"
                    )
                pose_item = pose_snapshot.get(topic)
                if pose_item is None:
                    problems.append(f"no end-effector feedback received for {topic}")
                elif now - pose_item[0] > args.feedback_timeout:
                    problems.append(
                        f"stale end-effector feedback for {topic}: "
                        f"{now - pose_item[0]:.3f} seconds"
                    )
                else:
                    try:
                        plan.workspace[topic].validate(
                            pose_item[1], pose_item[2], f"live {topic} end effector"
                        )
                    except ReplaySafetyError as error:
                        problems.append(str(error))
            if self.fatal_error:
                problems.append(self.fatal_error)
            return problems

        def preflight(self) -> None:
            parameters = self.runtime_parameters()
            expected = {
                "robot_name": safety.robot_name,
                "robot_description_file": str(safety.robot_description_path),
                "config_file": str(safety.controller_config_path),
            }
            for name, expected_value in expected.items():
                if parameters.get(name) != expected_value:
                    raise ReplaySafetyError(
                        f"moca_adapter {name} mismatch: expected {expected_value!r}, "
                        f"found {parameters.get(name)!r}"
                    )
            verify_runtime_file_identities(safety)

            deadline = time.monotonic() + args.discovery_timeout
            problems: list[str] = ["ROS discovery not complete"]
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                problems = self.graph_problems(include_self=False) + self.feedback_problems()
                if not problems:
                    break
            if problems:
                raise ReplaySafetyError("live preflight failed: " + "; ".join(problems))

            snapshot = self.feedback_snapshot()
            start_errors = {
                topic: maximum_position_error(
                    start_commands[topic].position, snapshot[topic][1]
                )
                for topic in plan.topics
            }
            excessive = {
                topic: error
                for topic, error in start_errors.items()
                if error > args.start_tolerance_rad
            }
            if excessive:
                details = ", ".join(
                    f"{topic}={error:.4f} rad" for topic, error in excessive.items()
                )
                raise ReplaySafetyError(
                    f"recorded start pose does not match live feedback: {details}"
                )
            manifest.update(
                status="preflight_passed",
                runtime_parameters=parameters,
                start_pose_max_error_rad=start_errors,
            )

        def start_pose_errors(self) -> Mapping[str, float]:
            snapshot = self.feedback_snapshot()
            return {
                topic: maximum_position_error(
                    start_commands[topic].position, snapshot[topic][1]
                )
                for topic in plan.topics
            }

        def arm_publishers(self) -> None:
            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            qos.history = HistoryPolicy.KEEP_LAST
            self.arm_command_publishers = {
                topic: self.create_publisher(JointState, topic, qos)
                for topic in plan.topics
            }

        def check_runtime_safety(
            self, commands: Optional[Mapping[str, RecordedCommand]] = None
        ) -> None:
            feedback_problems = self.feedback_problems()
            if feedback_problems:
                raise ReplaySafetyError("; ".join(feedback_problems))
            now = time.monotonic()
            if now - self.last_graph_check >= 0.5:
                self.last_graph_check = now
                graph_problems = self.graph_problems(
                    include_self=bool(self.arm_command_publishers)
                )
                if graph_problems:
                    raise ReplaySafetyError("; ".join(graph_problems))

            if commands is None:
                return
            snapshot = self.feedback_snapshot()
            largest_error = max(
                maximum_position_error(commands[topic].position, snapshot[topic][1])
                for topic in plan.topics
            )
            if largest_error <= args.tracking_tolerance_rad:
                self.tracking_violation_since = None
                return
            if self.tracking_violation_since is None:
                self.tracking_violation_since = now
                return
            if now - self.tracking_violation_since >= args.tracking_grace:
                raise ReplaySafetyError(
                    f"arm tracking error {largest_error:.4f} rad exceeded "
                    f"{args.tracking_tolerance_rad:.4f} rad for "
                    f"{args.tracking_grace:.3f} seconds"
                )

        def publish(self, commands: Mapping[str, RecordedCommand]) -> None:
            stamp = self.get_clock().now().to_msg()
            for topic in plan.topics:
                limits = safety.topics[topic]
                message = JointState()
                message.header.stamp = stamp
                message.name = list(limits.joint_names)
                message.position = list(commands[topic].position)
                # The recorded velocity field is not trusted as a safety limit.
                # Publish the robot-specific reviewed ceiling scaled with time.
                message.velocity = list(limits.command_velocity)
                message.effort = []
                self.arm_command_publishers[topic].publish(message)

    rclpy.init()
    node = ReplaySafetyNode()
    spin_thread: Optional[threading.Thread] = None
    try:
        node.get_logger().info("Running replay graph, identity, feedback, and start-pose preflight")
        node.preflight()

        if args.mode == "live":
            if not sys.stdin.isatty():
                raise ReplaySafetyError("live replay requires an interactive terminal")
            expected_phrase = f"REPLAY {args.arms.upper()} {source_sha256[:12]}"
            entered = input(
                "No motion has been sent. Verify operator, workspace, E-stop, and control "
                f"ownership now. Type '{expected_phrase}' to create the selected "
                "arm publisher(s): "
            )
            if entered != expected_phrase:
                raise ReplaySafetyError("interactive arming phrase did not match")
            node.arm_publishers()
            manifest.update(status="armed", armed_at=utc_timestamp())

        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

        start_convergence_count = 0
        if args.mode == "live":
            convergence_deadline = time.monotonic() + args.start_convergence_timeout
            convergence_period = 1.0 / args.rate
            while True:
                node.check_runtime_safety()
                start_errors = node.start_pose_errors()
                if max(start_errors.values()) <= args.start_convergence_tolerance_rad:
                    break
                if time.monotonic() >= convergence_deadline:
                    details = ", ".join(
                        f"{topic}={error:.4f} rad"
                        for topic, error in start_errors.items()
                    )
                    raise ReplaySafetyError(
                        "start convergence timed out with errors: " + details
                    )
                node.publish(start_commands)
                start_convergence_count += 1
                time.sleep(convergence_period)
            manifest.update(
                status="start_converged",
                start_convergence_published_messages=start_convergence_count,
                start_convergence_final_error_rad=start_errors,
            )

        start_wall = time.monotonic()
        last_manifest_update = start_wall
        command_count = {topic: 0 for topic in plan.topics}
        final_commands: Optional[Mapping[str, RecordedCommand]] = None
        for wall_elapsed, virtual_elapsed_ns, commands in iter_schedule(
            plan,
            rate_hz=args.rate,
            speed=args.speed,
            start_offset_seconds=args.start_offset_seconds,
        ):
            deadline = start_wall + wall_elapsed
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, 0.01))
            node.check_runtime_safety(commands if args.mode == "live" else None)
            if args.mode == "live":
                node.publish(commands)
                for topic in plan.topics:
                    command_count[topic] += 1
            final_commands = commands
            now = time.monotonic()
            if now - last_manifest_update >= 5.0:
                last_manifest_update = now
                manifest.update(
                    status="running_live" if args.mode == "live" else "running_shadow",
                    elapsed_wall_seconds=wall_elapsed,
                    elapsed_recorded_seconds=virtual_elapsed_ns / 1_000_000_000,
                    published_messages=command_count,
                )

        if args.mode == "live" and final_commands is not None and args.hold_seconds > 0.0:
            hold_end = time.monotonic() + args.hold_seconds
            period = 1.0 / args.rate
            while time.monotonic() < hold_end:
                node.check_runtime_safety(final_commands)
                node.publish(final_commands)
                for topic in plan.topics:
                    command_count[topic] += 1
                time.sleep(period)

        manifest.update(
            status="completed_live" if args.mode == "live" else "completed_shadow",
            completed_at=utc_timestamp(),
            published_messages=command_count,
            start_convergence_published_messages=start_convergence_count,
        )
        return 0
    finally:
        # Destroying the publishers does not prove that the robot has stopped.
        # The responsible operator and emergency stop remain the stop authority.
        node.destroy_node()
        rclpy.try_shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


def plan_summary(
    plan: ReplayPlan,
    safety: SafetyConfig,
    schedule: Mapping[str, object],
    source_sha256: Optional[str],
) -> dict[str, object]:
    return {
        "source_mcap": str(plan.source),
        "source_size_bytes": plan.source_size_bytes,
        "source_sha256": source_sha256,
        "duration_seconds": plan.duration_seconds,
        "whitelisted_topics": list(plan.topics),
        "rejected_topic_classes": [
            "/motion_control/*",
            "/controller",
            "feedback",
            "camera",
            "service/action traffic",
        ],
        "safety_config": str(safety.path),
        "robot_name": safety.robot_name,
        "robot_serial": safety.robot_serial,
        "topic_stats": {topic: plan.stats[topic].as_dict() for topic in plan.topics},
        "workspace_envelopes": {
            topic: plan.workspace[topic].as_dict() for topic in plan.topics
        },
        "schedule": dict(schedule),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", type=Path, help="immutable source episode MCAP")
    parser.add_argument(
        "--mode",
        choices=("inspect", "shadow", "live"),
        default="inspect",
        help="inspect is default and creates no ROS entities",
    )
    parser.add_argument(
        "--arms",
        choices=("left", "right", "both"),
        default="both",
        help="arm command streams to inspect or replay; unselected arms get no publisher",
    )
    parser.add_argument(
        "--limits",
        type=Path,
        default=script_dir / "r1_episode_replay_r1lite_2026_limits.json",
        help="robot-specific replay safety configuration",
    )
    parser.add_argument("--manifest", type=Path, help="atomic run-manifest output path")
    parser.add_argument("--source-sha256", help="required source digest for live mode")
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="calculate source SHA-256 in inspect/shadow mode",
    )
    parser.add_argument("--rate", type=float, default=50.0, help="bounded publish/tick rate")
    parser.add_argument(
        "--speed", type=float, default=0.25, help="recorded-time scale in (0, 1]"
    )
    parser.add_argument(
        "--start-offset-seconds",
        type=float,
        default=0.0,
        help="resume from this recorded-time offset without replaying earlier commands",
    )
    parser.add_argument("--step-slack-rad", type=float, default=0.002)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.10)
    parser.add_argument("--start-convergence-tolerance-rad", type=float, default=0.10)
    parser.add_argument("--start-convergence-timeout", type=float, default=15.0)
    parser.add_argument("--tracking-tolerance-rad", type=float, default=0.25)
    parser.add_argument("--tracking-grace", type=float, default=0.50)
    parser.add_argument("--feedback-timeout", type=float, default=0.25)
    parser.add_argument("--discovery-timeout", type=float, default=8.0)
    parser.add_argument("--hold-seconds", type=float, default=0.50)

    # Live mode requires both machine-checkable identity and explicit operator
    # attestations. These flags do not replace the final interactive phrase.
    parser.add_argument("--arm-token")
    parser.add_argument("--source-robot-id")
    parser.add_argument("--operator-confirmed", action="store_true")
    parser.add_argument("--workspace-clear", action="store_true")
    parser.add_argument("--estop-ready", action="store_true")
    parser.add_argument("--control-owned", action="store_true")
    parser.add_argument("--recorder-inactive", action="store_true")
    parser.add_argument(
        "--acknowledge-self-collision-check-disabled", action="store_true"
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace, source: Path, safety: SafetyConfig) -> None:
    positive_fields = {
        "rate": args.rate,
        "speed": args.speed,
        "start tolerance": args.start_tolerance_rad,
        "start convergence tolerance": args.start_convergence_tolerance_rad,
        "start convergence timeout": args.start_convergence_timeout,
        "tracking tolerance": args.tracking_tolerance_rad,
        "tracking grace": args.tracking_grace,
        "feedback timeout": args.feedback_timeout,
        "discovery timeout": args.discovery_timeout,
    }
    for label, value in positive_fields.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ReplaySafetyError(f"{label} must be finite and greater than zero")
    if args.speed > 1.0:
        raise ReplaySafetyError("speed above 1.0 is prohibited")
    if not math.isfinite(args.start_offset_seconds) or args.start_offset_seconds < 0.0:
        raise ReplaySafetyError("start offset must be finite and non-negative")
    if args.rate > safety.controller_command_rate_hz:
        raise ReplaySafetyError("rate exceeds the configured controller command rate")
    if args.step_slack_rad < 0.0 or not math.isfinite(args.step_slack_rad):
        raise ReplaySafetyError("step slack must be finite and non-negative")
    if not 0.0 <= args.hold_seconds <= 5.0:
        raise ReplaySafetyError("hold seconds must be between 0 and 5")
    if args.start_convergence_tolerance_rad > args.tracking_tolerance_rad:
        raise ReplaySafetyError(
            "start convergence tolerance cannot exceed tracking tolerance"
        )

    if args.manifest is not None:
        manifest = args.manifest.expanduser().resolve()
        if is_within(manifest, source.parent):
            raise ReplaySafetyError("manifest must not be written inside the raw episode")
    if args.mode in ("shadow", "live") and args.manifest is None:
        raise ReplaySafetyError(f"--manifest is required for {args.mode} mode")
    if args.mode == "shadow" and not (args.checksum or args.source_sha256):
        raise ReplaySafetyError(
            "shadow mode requires --checksum or a verified --source-sha256"
        )

    if args.source_sha256 is not None:
        _require_sha256(args.source_sha256, "source digest")
    if args.mode != "live":
        return
    if args.source_sha256 is None:
        raise ReplaySafetyError("--source-sha256 is required for live mode")
    expected_token = LIVE_ARM_TOKENS[args.arms]
    if args.arm_token != expected_token:
        raise ReplaySafetyError(f"live mode requires --arm-token {expected_token}")
    if args.source_robot_id != safety.robot_serial:
        raise ReplaySafetyError(
            f"live mode requires --source-robot-id {safety.robot_serial}"
        )
    attestations = {
        "--operator-confirmed": args.operator_confirmed,
        "--workspace-clear": args.workspace_clear,
        "--estop-ready": args.estop_ready,
        "--control-owned": args.control_owned,
        "--recorder-inactive": args.recorder_inactive,
        "--acknowledge-self-collision-check-disabled": (
            safety.expected_self_collision_check
            or args.acknowledge_self_collision_check_disabled
        ),
    }
    missing = [flag for flag, present in attestations.items() if not present]
    if missing:
        raise ReplaySafetyError("live mode is missing: " + ", ".join(missing))


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    manifest: Optional[AtomicManifest] = None
    try:
        source = validate_source_path(args.mcap)
        safety = load_safety_config(args.limits)
        validate_args(args, source, safety)
        replay_topics = selected_arm_topics(args.arms)

        manifest_path = args.manifest.expanduser().resolve() if args.manifest else None
        if manifest_path is not None:
            manifest = AtomicManifest(
                manifest_path,
                {
                    "schema_version": 1,
                    "created_at": utc_timestamp(),
                    "updated_at": utc_timestamp(),
                    "status": "started",
                    "mode": args.mode,
                    "arm_selection": args.arms,
                    "source_mcap": str(source),
                    "safety_config": str(safety.path),
                },
            )

        source_digest: Optional[str] = None
        if args.checksum or args.source_sha256 is not None or args.mode == "live":
            source_digest = file_sha256(source)
            if args.source_sha256 is not None and source_digest != args.source_sha256.lower():
                raise ReplaySafetyError(
                    f"source checksum mismatch: expected {args.source_sha256.lower()}, "
                    f"found {source_digest}"
                )

        plan = load_replay_plan(source, safety, replay_topics)
        if args.start_offset_seconds >= plan.duration_seconds:
            raise ReplaySafetyError("start offset must be less than the replay duration")
        schedule = validate_resampled_schedule(
            plan,
            safety,
            rate_hz=args.rate,
            speed=args.speed,
            step_slack_rad=args.step_slack_rad,
            start_offset_seconds=args.start_offset_seconds,
        )
        summary = plan_summary(plan, safety, schedule, source_digest)
        if manifest is not None:
            manifest.update(status="trajectory_validated", summary=summary)

        if args.mode == "inspect":
            print(json.dumps(summary, indent=2, sort_keys=True))
            if manifest is not None:
                manifest.update(status="completed_inspect", completed_at=utc_timestamp())
            return 0

        assert manifest is not None
        return run_ros_mode(args, plan, safety, source_digest, manifest)
    except KeyboardInterrupt:
        if manifest is not None:
            manifest.update(status="aborted", error="operator interrupt")
        print("ABORTED: operator interrupt; no stop state is inferred", file=sys.stderr)
        return 130
    except Exception as error:
        if manifest is not None:
            manifest.update(status="failed", error=str(error))
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
