#!/usr/bin/env python3
"""Keyboard-only gripper control for a Galaxea R1 Lite.

This custom node intentionally publishes only gripper position targets. It does
not publish arm, torso, chassis, brake, or generic controller messages.

Normal mode uses the observed Galaxea contract:

* ``sensor_msgs/msg/JointState``
* 0.0 = fully closed
* 100.0 = fully open
* best-effort, keep-last QoS

Dry-run mode imports no ROS modules, creates no ROS entities, changes no
parameters, and publishes nothing.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import select
import stat
import struct
import sys
import termios
import threading
import time
import tty
from typing import Deque, Dict, Optional


LEFT_TOPIC = "/motion_target/target_position_gripper_left"
RIGHT_TOPIC = "/motion_target/target_position_gripper_right"
EXPECTED_TOPIC_TYPE = "sensor_msgs/msg/JointState"

GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 100.0

USB_TOGGLE_KEY = "<f12>"
EV_KEY = 0x01
KEY_DOWN = 1
EVIOCGRAB = 0x40044590
INPUT_EVENT_STRUCT = struct.Struct("@llHHI")

LINUX_KEY_CODES = {
    1: "\x1b",  # KEY_ESC
    11: "0",  # KEY_0
    16: "q",  # KEY_Q
    18: "e",  # KEY_E
    23: "i",  # KEY_I
    25: "p",  # KEY_P
    88: USB_TOGGLE_KEY,  # KEY_F12
}


@dataclass(frozen=True)
class GripperCommand:
    """One absolute gripper target requested by a keyboard key."""

    side: str
    position: float
    label: str


KEY_BINDINGS = {
    "q": GripperCommand("left", GRIPPER_OPEN, "OPEN LEFT"),
    "e": GripperCommand("left", GRIPPER_CLOSED, "CLOSE LEFT"),
    "i": GripperCommand("right", GRIPPER_OPEN, "OPEN RIGHT"),
    "p": GripperCommand("right", GRIPPER_CLOSED, "CLOSE RIGHT"),
}


def command_for_key(key: str) -> Optional[GripperCommand]:
    """Return the command for a single key, case-insensitively."""

    if not key:
        return None
    return KEY_BINDINGS.get(key[0].lower())


def clamp_position(position: float) -> float:
    """Clamp a gripper percentage target to the observed valid range."""

    return max(GRIPPER_CLOSED, min(GRIPPER_OPEN, float(position)))


@dataclass(frozen=True)
class ActiveBurst:
    """A bounded publish burst for one gripper."""

    position: float
    deadline: float
    label: str


class BurstBook:
    """Pure state machine used by ROS mode and unit tests."""

    def __init__(self, duration_sec: float):
        if duration_sec <= 0:
            raise ValueError("duration_sec must be greater than zero")
        self.duration_sec = float(duration_sec)
        self._active: Dict[str, ActiveBurst] = {}

    def activate(self, command: GripperCommand, now: float) -> None:
        if command.side not in ("left", "right"):
            raise ValueError(f"unknown gripper side: {command.side}")
        self._active[command.side] = ActiveBurst(
            position=clamp_position(command.position),
            deadline=float(now) + self.duration_sec,
            label=command.label,
        )

    def cancel_all(self) -> None:
        self._active.clear()

    def snapshot(self, now: float) -> Dict[str, ActiveBurst]:
        expired = [side for side, burst in self._active.items() if burst.deadline <= now]
        for side in expired:
            del self._active[side]
        return dict(self._active)


class RawTerminal:
    """Restore terminal settings even when the controller exits on error."""

    def __init__(self) -> None:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise RuntimeError("keyboard control requires an interactive terminal")
        self.fd = sys.stdin.fileno()
        self.previous = termios.tcgetattr(self.fd)

    def __enter__(self) -> "RawTerminal":
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_: object) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)

    def read_key(self, timeout_sec: float = 0.2) -> Optional[str]:
        readable, _, _ = select.select([self.fd], [], [], timeout_sec)
        if not readable:
            return None
        return sys.stdin.read(1)


def validate_input_device_path(path: str) -> str:
    """Accept only a stable keyboard symlink in Linux's by-id directory."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise argparse.ArgumentTypeError("--input-device must be an absolute path")
    try:
        candidate.relative_to("/dev/input/by-id")
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--input-device must be under /dev/input/by-id"
        ) from error
    if candidate.parent != Path("/dev/input/by-id"):
        raise argparse.ArgumentTypeError(
            "--input-device must be a direct entry in /dev/input/by-id"
        )
    if not candidate.name.endswith("-event-kbd"):
        raise argparse.ArgumentTypeError(
            "--input-device must name a stable *-event-kbd device"
        )
    return str(candidate)


class LinuxInputEventDecoder:
    """Decode complete Linux input_event records from arbitrary read chunks."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[str]:
        self.buffer.extend(data)
        keys = []
        size = INPUT_EVENT_STRUCT.size
        while len(self.buffer) >= size:
            record = bytes(self.buffer[:size])
            del self.buffer[:size]
            _, _, event_type, code, value = INPUT_EVENT_STRUCT.unpack(record)
            if event_type != EV_KEY or value != KEY_DOWN:
                continue
            key = LINUX_KEY_CODES.get(code)
            if key is not None:
                keys.append(key)
        return keys

    def finish(self) -> None:
        if self.buffer:
            raise RuntimeError("USB keyboard disconnected with a partial input event")


class LinuxInputDevice:
    """Read and exclusively own one physical Linux keyboard event device."""

    def __init__(self, path: str) -> None:
        self.path = validate_input_device_path(path)
        self.fd: Optional[int] = None
        self.decoder = LinuxInputEventDecoder()
        self.pending: Deque[str] = deque()

    def __enter__(self) -> "LinuxInputDevice":
        if sys.platform != "linux":
            raise RuntimeError("--input-device is supported only on Linux")
        try:
            device_stat = os.stat(self.path)
            if not stat.S_ISCHR(device_stat.st_mode):
                raise RuntimeError(f"input device is not a character device: {self.path}")
            self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            fcntl.ioctl(self.fd, EVIOCGRAB, struct.pack("i", 1))
        except PermissionError as error:
            self._close()
            raise RuntimeError(
                f"cannot read USB keyboard {self.path}; grant temporary read access first"
            ) from error
        except OSError as error:
            self._close()
            raise RuntimeError(
                f"cannot exclusively open USB keyboard {self.path}: {error}"
            ) from error
        return self

    def _close(self) -> None:
        if self.fd is None:
            return
        try:
            fcntl.ioctl(self.fd, EVIOCGRAB, struct.pack("i", 0))
        except OSError:
            pass
        os.close(self.fd)
        self.fd = None

    def __exit__(self, *_: object) -> None:
        self._close()

    def read_key(self, timeout_sec: float = 0.2) -> Optional[str]:
        if self.pending:
            return self.pending.popleft()
        if self.fd is None:
            raise RuntimeError("USB keyboard is not open")
        readable, _, _ = select.select([self.fd], [], [], timeout_sec)
        if not readable:
            return None
        try:
            data = os.read(self.fd, INPUT_EVENT_STRUCT.size * 32)
        except BlockingIOError:
            return None
        except OSError as error:
            raise RuntimeError(f"USB keyboard read failed: {error}") from error
        if not data:
            self.decoder.finish()
            raise RuntimeError("USB keyboard disconnected")
        self.pending.extend(self.decoder.feed(data))
        return self.pending.popleft() if self.pending else None


@dataclass(frozen=True)
class InputDecision:
    """A safety-gated interpretation of one terminal or USB key."""

    action: str
    command: Optional[GripperCommand] = None
    message: str = ""


class UsbSafetyGate:
    """Require an explicit F12 transition before accepting USB motion keys."""

    def __init__(self) -> None:
        self.unlocked = False

    def interpret(self, key: str) -> InputDecision:
        if is_exit_key(key):
            return InputDecision("exit")
        if key == USB_TOGGLE_KEY:
            self.unlocked = not self.unlocked
            state = "UNLOCKED" if self.unlocked else "LOCKED"
            return InputDecision("toggle", message=f"USB INPUT {state} — no target sent")

        command = command_for_key(key)
        is_control_key = command is not None or key == "0"
        if not self.unlocked:
            if is_control_key:
                return InputDecision(
                    "ignored", message="IGNORED — USB INPUT LOCKED; press F12"
                )
            return InputDecision("noop")
        if key == "0":
            return InputDecision("cancel", message="CANCEL — no new target sent")
        if command is not None:
            return InputDecision("command", command=command)
        return InputDecision("noop")

    def status(self) -> str:
        return "USB UNLOCKED" if self.unlocked else "USB LOCKED — press F12"


def interpret_terminal_key(key: str) -> InputDecision:
    if is_exit_key(key):
        return InputDecision("exit")
    if key == "0":
        return InputDecision("cancel", message="CANCEL — no new target sent")
    command = command_for_key(key)
    if command is not None:
        return InputDecision("command", command=command)
    return InputDecision("noop")


TERMINAL_HELP = """\
============================================================
  Galaxea R1 Lite Keyboard Gripper Control
============================================================

  [Q] OPEN left       [E] CLOSE left
  [I] OPEN right      [P] CLOSE right

  [0] cancel active publish bursts (does not move a gripper)
  [Esc / Ctrl-C] exit (does not send a new position)

  Keep this terminal focused while operating the grippers.
============================================================"""

USB_HELP = """\
============================================================
  Galaxea R1 Lite Direct-USB Gripper Control
============================================================

  [F12] lock / unlock physical USB input (starts locked)
  [Q] OPEN left       [E] CLOSE left
  [I] OPEN right      [P] CLOSE right

  [0] cancel active publish bursts (does not move a gripper)
  [Esc] exit (does not send a new position)

  Only key-down events are accepted; held-key repeats are ignored.
============================================================"""


def raw_write(text: str) -> None:
    sys.stdout.write(text.replace("\n", "\r\n"))
    sys.stdout.flush()


def render(help_text: str, mode: str, status: str, last_action: str, active: str) -> None:
    sys.stdout.write("\033[2J\033[H")
    raw_write(
        f"{help_text}\n\n"
        f"  Mode:        {mode}\n"
        f"  ROS status:  {status}\n"
        f"  Last action: {last_action}\n"
        f"  Active:      {active}\n"
    )


def is_exit_key(key: str) -> bool:
    return key in ("\x03", "\x1b")


def input_source(args: argparse.Namespace) -> RawTerminal | LinuxInputDevice:
    if args.input_device:
        return LinuxInputDevice(args.input_device)
    return RawTerminal()


def run_dry_mode(args: argparse.Namespace) -> int:
    """Exercise keyboard handling without importing or initializing ROS."""

    last_action = "none"
    usb_gate = UsbSafetyGate() if args.input_device else None
    help_text = USB_HELP if usb_gate else TERMINAL_HELP
    mode = "DRY RUN — DIRECT USB; NO ROS ENTITIES" if usb_gate else "DRY RUN — NO ROS ENTITIES"
    with input_source(args) as keyboard:
        status = f"disabled by design; {usb_gate.status()}" if usb_gate else "disabled by design"
        render(help_text, mode, status, last_action, "none")
        while True:
            key = keyboard.read_key()
            if key is None:
                continue
            decision = usb_gate.interpret(key) if usb_gate else interpret_terminal_key(key)
            if decision.action == "exit":
                break
            if decision.action == "command" and decision.command is not None:
                last_action = (
                    f"{decision.command.label} -> {decision.command.position:.1f} "
                    "(display only)"
                )
            elif decision.action in ("cancel", "toggle", "ignored"):
                last_action = decision.message + " (display only)"
            status = (
                f"disabled by design; {usb_gate.status()}"
                if usb_gate
                else "disabled by design"
            )
            render(help_text, mode, status, last_action, "none")
    sys.stdout.write("\nDry run exited; no ROS entities were created.\n")
    return 0


def run_ros_mode(args: argparse.Namespace) -> int:
    """Run the fail-closed ROS controller after lazy-importing ROS packages."""

    import rclpy
    from rcl_interfaces.msg import Parameter as ParameterMsg
    from rcl_interfaces.msg import ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    topics = {"left": LEFT_TOPIC, "right": RIGHT_TOPIC}

    class KeyboardGripperNode(Node):
        PARAM_SERVICE = "/moca_adapter/set_parameters"

        def __init__(self) -> None:
            super().__init__("jimmy_keyboard_gripper_teleop")
            self.parameter_client = self.create_client(SetParameters, self.PARAM_SERVICE)
            self.gripper_publishers = {}
            self.lock = threading.Lock()
            self.bursts = BurstBook(args.duration)
            self.last_action = "none"
            self.fatal_error: Optional[str] = None
            self.last_conflict_check = 0.0
            self.timer = None

        @staticmethod
        def _types(endpoint_infos: object) -> set[str]:
            return {
                getattr(info, "topic_type", "")
                for info in endpoint_infos
                if getattr(info, "topic_type", "")
            }

        def _interface_state(self) -> tuple[dict[str, int], dict[str, int], set[str]]:
            subscriber_counts = {}
            publisher_counts = {}
            wrong_types: set[str] = set()
            for side, topic in topics.items():
                subscriptions = self.get_subscriptions_info_by_topic(topic)
                publishers = self.get_publishers_info_by_topic(topic)
                subscriber_counts[side] = len(subscriptions)
                publisher_counts[side] = len(publishers)
                endpoint_types = self._types(subscriptions) | self._types(publishers)
                wrong_types.update(t for t in endpoint_types if t != EXPECTED_TOPIC_TYPE)
            return subscriber_counts, publisher_counts, wrong_types

        def preflight(self) -> None:
            deadline = time.monotonic() + args.discovery_timeout
            last_state = ({"left": 0, "right": 0}, {"left": 0, "right": 0}, set())

            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                last_state = self._interface_state()
                subscribers, publishers, wrong_types = last_state
                service_ready = self.parameter_client.service_is_ready()
                if (
                    service_ready
                    and all(subscribers[side] >= 1 for side in topics)
                    and all(publishers[side] == 0 for side in topics)
                    and not wrong_types
                ):
                    return

            subscribers, publishers, wrong_types = last_state
            problems = []
            if not self.parameter_client.service_is_ready():
                problems.append(f"missing service {self.PARAM_SERVICE}")
            for side, topic in topics.items():
                if subscribers[side] < 1:
                    problems.append(f"no subscriber for {topic}")
                if publishers[side] != 0:
                    problems.append(
                        f"{publishers[side]} existing publisher(s) on {topic}; "
                        "stop Bluetooth/joy gripper authority first"
                    )
            if wrong_types:
                problems.append(f"unexpected topic type(s): {sorted(wrong_types)}")
            raise RuntimeError("preflight failed: " + "; ".join(problems))

        @staticmethod
        def _bool_parameter(name: str, value: bool) -> ParameterMsg:
            return ParameterMsg(
                name=name,
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=value,
                ),
            )

        def enable_end_effector_control(self) -> None:
            request = SetParameters.Request()
            request.parameters = [
                self._bool_parameter("end_effector_control.enable", True),
                self._bool_parameter("end_effector_control.enable_left_ee", True),
                self._bool_parameter("end_effector_control.enable_right_ee", True),
                self._bool_parameter("end_effector_control.use_interpolation", True),
            ]
            future = self.parameter_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            response = future.result()
            if response is None:
                raise RuntimeError("moca_adapter parameter request timed out")
            failures = [result.reason for result in response.results if not result.successful]
            if failures:
                raise RuntimeError("moca_adapter rejected parameters: " + "; ".join(failures))

        def arm_publishers(self) -> None:
            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            qos.history = HistoryPolicy.KEEP_LAST
            self.gripper_publishers = {
                side: self.create_publisher(JointState, topic, qos)
                for side, topic in topics.items()
            }
            self.timer = self.create_timer(1.0 / args.rate, self._publish_tick)

        def request(self, command: GripperCommand) -> None:
            with self.lock:
                self.bursts.activate(command, time.monotonic())
                self.last_action = f"{command.label} -> {command.position:.1f}"

        def cancel(self, message: str = "CANCEL — no new target sent") -> None:
            with self.lock:
                self.bursts.cancel_all()
                self.last_action = message

        def set_last_action(self, message: str) -> None:
            with self.lock:
                self.last_action = message

        def status(self) -> tuple[str, str]:
            with self.lock:
                snapshot = self.bursts.snapshot(time.monotonic())
                active = ", ".join(
                    f"{side}={burst.position:.1f}" for side, burst in sorted(snapshot.items())
                ) or "none"
                return self.last_action, active

        def _detect_new_publisher(self, now: float) -> bool:
            if now - self.last_conflict_check < 0.5:
                return False
            self.last_conflict_check = now
            conflicts = []
            for side, topic in topics.items():
                # This node owns one publisher after arm_publishers().
                count = self.count_publishers(topic)
                if count > 1:
                    conflicts.append(f"{side}={count}")
            if not conflicts:
                return False
            with self.lock:
                self.bursts.cancel_all()
                self.fatal_error = (
                    "competing gripper publisher detected (" + ", ".join(conflicts) + ")"
                )
            self.get_logger().error(self.fatal_error)
            return True

        def _publish_tick(self) -> None:
            now = time.monotonic()
            if self._detect_new_publisher(now):
                return
            with self.lock:
                snapshot = self.bursts.snapshot(now)
            for side, burst in snapshot.items():
                message = JointState()
                message.header.stamp = self.get_clock().now().to_msg()
                message.position = [clamp_position(burst.position)]
                self.gripper_publishers[side].publish(message)

    usb_gate = UsbSafetyGate() if args.input_device else None
    help_text = USB_HELP if usb_gate else TERMINAL_HELP
    mode = "LIVE ROS — GRIPPER ONLY; DIRECT USB" if usb_gate else "LIVE ROS — GRIPPER ONLY"

    with ExitStack() as stack:
        keyboard = input_source(args)
        # Open and exclusively grab a USB device before creating any ROS entity.
        # Terminal raw mode is delayed until after preflight as before.
        if usb_gate:
            keyboard = stack.enter_context(keyboard)

        rclpy.init()
        node = KeyboardGripperNode()
        spin_thread = None
        try:
            node.get_logger().info("Running fail-closed gripper interface preflight")
            node.preflight()
            node.enable_end_effector_control()
            node.arm_publishers()
            node.get_logger().info("Keyboard gripper control is armed; no position sent yet")

            spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
            spin_thread.start()

            if not usb_gate:
                keyboard = stack.enter_context(keyboard)
            last_action, active = node.status()
            ros_status = f"armed; {usb_gate.status()}" if usb_gate else "armed"
            render(help_text, mode, ros_status, last_action, active)
            displayed_status = (ros_status, last_action, active)
            while True:
                if node.fatal_error:
                    raise RuntimeError(node.fatal_error)
                key = keyboard.read_key()
                if key is None:
                    last_action, active = node.status()
                    ros_status = f"armed; {usb_gate.status()}" if usb_gate else "armed"
                    current_status = (ros_status, last_action, active)
                    if current_status != displayed_status:
                        render(help_text, mode, *current_status)
                        displayed_status = current_status
                    continue
                decision = usb_gate.interpret(key) if usb_gate else interpret_terminal_key(key)
                if decision.action == "exit":
                    break
                if decision.action == "command" and decision.command is not None:
                    node.request(decision.command)
                elif decision.action in ("cancel", "toggle"):
                    node.cancel(decision.message)
                elif decision.action == "ignored":
                    node.set_last_action(decision.message)
                last_action, active = node.status()
                ros_status = f"armed; {usb_gate.status()}" if usb_gate else "armed"
                render(help_text, mode, ros_status, last_action, active)
                displayed_status = (ros_status, last_action, active)
            return 0
        finally:
            node.cancel()
            node.destroy_node()
            rclpy.try_shutdown()
            if spin_thread is not None:
                spin_thread.join(timeout=1.0)
            sys.stdout.write("\nKeyboard gripper controller exited; no shutdown target was sent.\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Galaxea R1 Lite keyboard gripper control")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read keys and display actions without importing or initializing ROS",
    )
    parser.add_argument(
        "--input-device",
        type=validate_input_device_path,
        help=(
            "read a directly connected R1 keyboard from an explicit "
            "/dev/input/by-id/*-event-kbd path; USB input starts locked"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="bounded publish duration after each keypress in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="publish rate during an active burst in Hz (default: 20)",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=6.0,
        help="seconds to wait for required ROS endpoints (default: 6)",
    )
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.rate <= 0:
        parser.error("--rate must be greater than zero")
    if args.discovery_timeout <= 0:
        parser.error("--discovery-timeout must be greater than zero")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.dry_run:
            return run_dry_mode(args)
        return run_ros_mode(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
