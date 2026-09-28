#!/usr/bin/env python3
"""Interactive reference-photo overlay for Galaxea head-camera reset guidance."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage


DEFAULT_TOPIC = (
    "/hdas/camera_head/left_raw/image_raw_color/compressed"
)
WINDOW_NAME = "Jimmy object-reset guide - corrected head left"


def tint_reference(reference: np.ndarray) -> np.ndarray:
    """Make the reference magenta-biased so alignment appears as color ghosts."""
    adjusted = cv2.convertScaleAbs(reference, alpha=1.12, beta=12)
    blue, green, red = cv2.split(adjusted)
    blue = np.clip(blue.astype(np.float32) * 1.18, 0, 255).astype(np.uint8)
    green = np.clip(green.astype(np.float32) * 0.48, 0, 255).astype(np.uint8)
    red = np.clip(red.astype(np.float32) * 1.30, 0, 255).astype(np.uint8)
    return cv2.merge((blue, green, red))


def compose_overlay(
    live: np.ndarray,
    reference: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if reference.shape[:2] != live.shape[:2]:
        reference = cv2.resize(
            reference,
            (live.shape[1], live.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    tinted = tint_reference(reference)
    output = cv2.addWeighted(live, 1.0 - alpha, tinted, alpha, 0.0)

    # A thin yellow outline from the reference makes small placement errors
    # easier to see while retaining the semi-transparent color overlay.
    edges = cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 70, 150)
    output[edges > 0] = (0, 220, 255)
    return output


def draw_banner(
    image: np.ndarray,
    title: str,
    instruction: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    output = image.copy()
    height = 92
    cv2.rectangle(output, (0, 0), (output.shape[1], height), (18, 18, 18), -1)
    cv2.rectangle(output, (0, 0), (12, height), color, -1)
    cv2.putText(
        output,
        title,
        (28, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        instruction,
        (28, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return output


class LatestCompressedFrame(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("jimmy_head_reset_overlay")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self.create_subscription(CompressedImage, topic, self._callback, qos)

    def _callback(self, message: CompressedImage) -> None:
        with self._lock:
            self._jpeg = bytes(message.data)
            self._sequence += 1

    def snapshot(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._sequence


def write_ready_file(path: Path, reference_path: Path, topic: str) -> None:
    payload = {
        "ready": True,
        "reference_path": str(reference_path),
        "topic": topic,
        "accepted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_reference(reference: np.ndarray, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_corrected_head_left_reset_reference.png"
    )
    if not cv2.imwrite(str(path), reference):
        raise RuntimeError(f"Could not save reference image: {path}")
    return path


def self_test() -> int:
    live = np.full((240, 320, 3), (80, 100, 120), dtype=np.uint8)
    reference = live.copy()
    cv2.rectangle(reference, (80, 60), (240, 180), (20, 220, 60), -1)
    result = compose_overlay(live, reference, 0.42)
    if result.shape != live.shape or result.dtype != np.uint8:
        print("self-test failed", file=sys.stderr)
        return 1
    print("head_reset_overlay self-test passed")
    return 0


def run(args: argparse.Namespace) -> int:
    stop_event = threading.Event()

    def request_stop(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    rclpy.init()
    node = LatestCompressedFrame(args.topic)

    def spin_node() -> None:
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass

    ros_thread = threading.Thread(
        target=spin_node,
        name="reset-overlay-ros",
        daemon=True,
    )
    ros_thread.start()

    mode = "capture"
    reference: np.ndarray | None = None
    candidate: np.ndarray | None = None
    live: np.ndarray | None = None
    last_sequence = -1
    accepted_once = False
    last_frame_time = time.monotonic()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.window_width, args.window_height)

        while not stop_event.is_set():
            jpeg, sequence = node.snapshot()
            if jpeg is not None and sequence != last_sequence:
                decoded = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if decoded is not None:
                    live = decoded
                    last_sequence = sequence
                    last_frame_time = time.monotonic()

            if live is None:
                display = np.zeros((720, 960, 3), dtype=np.uint8)
                display = draw_banner(
                    display,
                    "Waiting for corrected head-left camera...",
                    f"Topic: {args.topic} | Q: quit",
                    (0, 180, 255),
                )
            elif mode == "capture":
                display = draw_banner(
                    live,
                    "Capture the object-reset reference",
                    "SPACE/C: take photo | Q: cancel",
                    (0, 180, 255),
                )
            elif mode == "preview" and candidate is not None:
                display = compose_overlay(live, candidate, args.alpha)
                display = draw_banner(
                    display,
                    "Check the reset-reference overlay",
                    "Y/ENTER: accept | R: retake | Q: cancel",
                    (70, 210, 70),
                )
            elif reference is not None:
                display = compose_overlay(live, reference, args.alpha)
                age = time.monotonic() - last_frame_time
                warning = " | CAMERA FRAME STALE" if age > 2.0 else ""
                display = draw_banner(
                    display,
                    "Live object-reset alignment guide",
                    "Match the colored ghost | R: retake reference | Q: close"
                    + warning,
                    (210, 80, 210),
                )
            else:
                mode = "capture"
                continue

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if live is None:
                continue
            if mode == "capture" and key in (ord("c"), ord(" ")):
                candidate = live.copy()
                mode = "preview"
            elif mode == "preview" and key in (ord("r"),):
                candidate = None
                mode = "capture"
            elif mode == "preview" and key in (ord("y"), 10, 13):
                reference = candidate.copy()
                reference_path = save_reference(reference, args.reference_dir)
                write_ready_file(args.ready_file, reference_path, args.topic)
                print(f"Accepted reset reference: {reference_path}", flush=True)
                accepted_once = True
                candidate = None
                mode = "overlay"
            elif mode == "overlay" and key == ord("r"):
                candidate = None
                mode = "capture"
    except cv2.error as exc:
        print(
            "ERROR: Could not open the OpenCV window. "
            "Check DISPLAY/XAUTHORITY.\n"
            f"{exc}",
            file=sys.stderr,
        )
        return 2
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=2.0)

    if not accepted_once:
        print("No reset reference was accepted.", file=sys.stderr)
        return 3
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/home/nvidia/jimmy/reset_references"),
    )
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--window-width", type=int, default=1200)
    parser.add_argument("--window-height", type=int, default=960)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 0.05 <= args.alpha <= 0.90:
        parser.error("--alpha must be between 0.05 and 0.90")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
