#!/usr/bin/env python3
"""Browser-rendered reset overlay controlled only by the collector terminal."""

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

from head_camera_web_preview import LatestJpeg, LOOPBACK_ADDRESS, PreviewHTTPServer
from head_reset_overlay import (
    DEFAULT_TOPIC,
    LatestCompressedFrame,
    compose_overlay,
    draw_banner,
    save_reference,
    write_ready_file,
)


VALID_COMMANDS = frozenset({"capture", "accept", "retake", "quit"})


class CommandFifo:
    """Non-blocking terminal-command channel owned by the collector shell."""

    def __init__(self, path: Path) -> None:
        if not path.exists() or not path.is_fifo():
            raise RuntimeError(f"command FIFO is missing or invalid: {path}")
        self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self._buffer = b""

    def read(self) -> list[str]:
        try:
            chunk = os.read(self._fd, 4096)
        except BlockingIOError:
            return []
        if chunk:
            self._buffer += chunk
        commands: list[str] = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            command = line.decode("ascii", errors="ignore").strip().lower()
            if command in VALID_COMMANDS:
                commands.append(command)
        return commands

    def close(self) -> None:
        os.close(self._fd)


def write_status_file(path: Path, topic: str, port: int, fps: float) -> None:
    payload = {
        "ready": True,
        "mode": "web",
        "topic": topic,
        "url": f"http://{LOOPBACK_ADDRESS}:{port}",
        "fps": fps,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def encode_preview(image: np.ndarray, max_width: int, quality: int) -> bytes:
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(
            image,
            (max_width, max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        (cv2.IMWRITE_JPEG_QUALITY, quality),
    )
    if not success:
        raise RuntimeError("could not encode reset preview as JPEG")
    return encoded.tobytes()


def self_test() -> int:
    live = np.full((240, 480, 3), (80, 100, 120), dtype=np.uint8)
    reference = live.copy()
    cv2.rectangle(reference, (120, 60), (360, 180), (20, 220, 60), -1)
    composed = compose_overlay(live, reference, 0.42)
    jpeg = encode_preview(composed, 320, 65)
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        print("web reset-overlay self-test failed", file=sys.stderr)
        return 1
    print("head_reset_web_overlay self-test passed")
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
        name="reset-web-ros",
        daemon=True,
    )
    ros_thread.start()

    frame_store = LatestJpeg()
    server = PreviewHTTPServer(
        (LOOPBACK_ADDRESS, args.port),
        frame_store,
        topic=args.topic,
        fps=args.fps,
        stale_after=args.stale_after,
        page_title="R1 object-reset guide",
        heading="Object-reset alignment guide",
    )
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="reset-web-http",
        daemon=True,
    )
    command_fifo = CommandFifo(args.command_fifo)
    server_thread.start()

    mode = "capture"
    reference: np.ndarray | None = None
    candidate: np.ndarray | None = None
    live: np.ndarray | None = None
    last_sequence = -1
    last_frame_time = time.monotonic()
    next_encode_at = 0.0
    accepted_once = False
    status_written = False

    try:
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

            for command in command_fifo.read():
                if command == "quit":
                    stop_event.set()
                elif command == "capture" and live is not None:
                    candidate = live.copy()
                    mode = "preview"
                elif command == "retake":
                    candidate = None
                    mode = "capture"
                elif command == "accept" and mode == "preview" and candidate is not None:
                    reference = candidate.copy()
                    reference_path = save_reference(reference, args.reference_dir)
                    write_ready_file(args.ready_file, reference_path, args.topic)
                    print(f"Accepted reset reference: {reference_path}", flush=True)
                    accepted_once = True
                    candidate = None
                    mode = "overlay"

            if live is None:
                display = np.zeros((720, 960, 3), dtype=np.uint8)
                display = draw_banner(
                    display,
                    "Waiting for corrected head-left camera...",
                    f"Topic: {args.topic} | controls remain in SSH terminal",
                    (0, 180, 255),
                )
            elif mode == "capture":
                display = draw_banner(
                    live,
                    "Capture the object-reset reference",
                    "Press ENTER in the collector terminal to take the photo",
                    (0, 180, 255),
                )
            elif mode == "preview" and candidate is not None:
                display = compose_overlay(live, candidate, args.alpha)
                display = draw_banner(
                    display,
                    "Check the reset-reference overlay",
                    "Accept, retake, or cancel in the collector terminal",
                    (70, 210, 70),
                )
            elif reference is not None:
                display = compose_overlay(live, reference, args.alpha)
                age = time.monotonic() - last_frame_time
                warning = " | CAMERA FRAME STALE" if age > args.stale_after else ""
                display = draw_banner(
                    display,
                    "Live object-reset alignment guide",
                    "Match the colored ghost | browser is display-only" + warning,
                    (210, 80, 210),
                )
            else:
                mode = "capture"
                continue

            now = time.monotonic()
            if now >= next_encode_at:
                frame_store.update(encode_preview(display, args.max_width, args.jpeg_quality))
                next_encode_at = now + (1.0 / args.fps)
                if live is not None and not status_written:
                    write_status_file(args.status_file, args.topic, args.port, args.fps)
                    status_written = True
            stop_event.wait(0.02)
    finally:
        stop_event.set()
        command_fifo.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=2.0)
        args.status_file.unlink(missing_ok=True)

    if not accepted_once:
        print("No reset reference was accepted.", file=sys.stderr)
        return 3
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--command-fifo", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/home/nvidia/jimmy/reset_references"),
    )
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--stale-after", type=float, default=2.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=65)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not 0.05 <= args.alpha <= 0.90:
        parser.error("--alpha must be between 0.05 and 0.90")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0 < args.fps <= 10:
        parser.error("--fps must be greater than 0 and at most 10")
    if args.stale_after <= 0:
        parser.error("--stale-after must be positive")
    if not 320 <= args.max_width <= 1920:
        parser.error("--max-width must be between 320 and 1920")
    if not 30 <= args.jpeg_quality <= 90:
        parser.error("--jpeg-quality must be between 30 and 90")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
