#!/usr/bin/env python3
"""Read-only web preview for one R1 Lite compressed head-camera topic.

The HTTP surface intentionally exposes only GET endpoints.  Collection,
reference capture, and episode decisions remain owned by the foreground
collector terminal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from typing import Callable
from urllib.parse import urlsplit


DEFAULT_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
DEFAULT_PORT = 8765
DEFAULT_FPS = 10.0
MAX_FPS = 10.0
MAX_JPEG_BYTES = 5 * 1024 * 1024
LOOPBACK_ADDRESS = "127.0.0.1"
MJPEG_BOUNDARY = "jimmy-r1-frame"


def _is_jpeg(data: bytes) -> bool:
    return len(data) >= 4 and data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


@dataclass(frozen=True)
class FrameSnapshot:
    jpeg: bytes
    sequence: int
    received_at: float


class LatestJpeg:
    """Thread-safe, single-frame buffer; raw frames are never written to disk."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._condition = threading.Condition()
        self._snapshot: FrameSnapshot | None = None
        self._sequence = 0

    def update(self, jpeg: bytes) -> bool:
        frame = bytes(jpeg)
        if len(frame) > MAX_JPEG_BYTES or not _is_jpeg(frame):
            return False
        with self._condition:
            self._sequence += 1
            self._snapshot = FrameSnapshot(frame, self._sequence, self._clock())
            self._condition.notify_all()
        return True

    def snapshot(self) -> FrameSnapshot | None:
        with self._condition:
            return self._snapshot

    def wait_after(self, sequence: int, timeout: float) -> FrameSnapshot | None:
        deadline = self._clock() + timeout
        with self._condition:
            while self._snapshot is None or self._snapshot.sequence <= sequence:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._snapshot


class PreviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        frame_store: LatestJpeg,
        *,
        topic: str,
        fps: float,
        stale_after: float,
        page_title: str = "R1 left-head preview",
        heading: str = "Corrected left-head camera",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if server_address[0] != LOOPBACK_ADDRESS:
            raise ValueError("preview server must bind only to 127.0.0.1")
        if not math.isfinite(fps) or not 0 < fps <= MAX_FPS:
            raise ValueError(f"fps must be greater than 0 and at most {MAX_FPS:g}")
        if not math.isfinite(stale_after) or stale_after <= 0:
            raise ValueError("stale_after must be positive")
        self.frame_store = frame_store
        self.topic = topic
        self.fps = fps
        self.stale_after = stale_after
        self.page_title = page_title
        self.heading = heading
        self.clock = clock
        super().__init__(server_address, PreviewRequestHandler)

    def fresh_snapshot(self) -> tuple[FrameSnapshot | None, float | None]:
        snapshot = self.frame_store.snapshot()
        if snapshot is None:
            return None, None
        age = max(0.0, self.clock() - snapshot.received_at)
        if age > self.stale_after:
            return None, age
        return snapshot, age


class PreviewRequestHandler(BaseHTTPRequestHandler):
    server: PreviewHTTPServer

    def log_message(self, _format: str, *_args: object) -> None:
        # Avoid logging client addresses and repeated MJPEG reconnects.
        return

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; img-src 'self'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; connect-src 'self'",
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        extra_headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        self._send_bytes(
            status,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/":
            self._send_bytes(HTTPStatus.OK, self._index_html(), "text/html; charset=utf-8")
        elif path == "/healthz":
            self._serve_health()
        elif path == "/snapshot.jpg":
            self._serve_snapshot()
        elif path == "/stream.mjpg":
            if self.command == "HEAD":
                self._send_bytes(
                    HTTPStatus.OK,
                    b"",
                    f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
                )
            else:
                self._serve_stream()
        else:
            self._send_bytes(HTTPStatus.NOT_FOUND, b"Not found\n", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self._send_bytes(
            HTTPStatus.METHOD_NOT_ALLOWED,
            b"Read-only preview: only GET and HEAD are allowed.\n",
            "text/plain; charset=utf-8",
            extra_headers=(("Allow", "GET, HEAD"),),
        )

    def _index_html(self) -> bytes:
        def escape(value: str) -> str:
            return (
                value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            )

        topic = escape(self.server.topic)
        page_title = escape(self.server.page_title)
        heading = escape(self.server.heading)
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title}</title>
<style>
html,body{{margin:0;background:#101214;color:#edf0f2;font:15px system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:18px}}
header{{display:flex;justify-content:space-between;gap:12px;align-items:baseline}}
h1{{font-size:18px;margin:0 0 5px}} p{{margin:0;color:#aeb7bd}}
#status{{font:13px ui-monospace,monospace;color:#f1c75b}}
img{{display:block;width:100%;height:auto;margin-top:14px;background:#050607;border:1px solid #30363a}}
code{{word-break:break-all}}
</style>
</head>
<body><main>
<header><div><h1>{heading}</h1><p><code>{topic}</code></p></div>
<div id="status">waiting for frames</div></header>
<img src="/stream.mjpg" alt="Corrected left-head camera preview">
</main>
<script>
const statusNode=document.getElementById('status');
async function refresh(){{
  try{{
    const response=await fetch('/healthz',{{cache:'no-store'}});
    const state=await response.json();
    statusNode.textContent=state.ok ? `${{state.fps}} Hz limit · frame age ${{state.age_seconds.toFixed(2)}} s` : 'waiting for fresh frames';
    statusNode.style.color=state.ok ? '#79d29d' : '#f1c75b';
  }}catch(_error){{statusNode.textContent='preview disconnected';statusNode.style.color='#ff7b72'}}
}}
refresh();setInterval(refresh,1000);
</script></body></html>"""
        return html.encode("utf-8")

    def _serve_health(self) -> None:
        snapshot, age = self.server.fresh_snapshot()
        ok = snapshot is not None
        payload: dict[str, object] = {
            "ok": ok,
            "topic": self.server.topic,
            "fps": self.server.fps,
            "age_seconds": age,
        }
        self._send_json(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE, payload)

    def _serve_snapshot(self) -> None:
        snapshot, age = self.server.fresh_snapshot()
        if snapshot is None:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": "no fresh JPEG frame", "age_seconds": age},
            )
            return
        self._send_bytes(
            HTTPStatus.OK,
            snapshot.jpeg,
            "image/jpeg",
            extra_headers=(("X-Frame-Age-Seconds", f"{age:.3f}"),),
        )

    def _serve_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}"
        )
        self.end_headers()

        last_sequence = 0
        next_send_at = 0.0
        interval = 1.0 / self.server.fps
        try:
            while True:
                snapshot = self.server.frame_store.wait_after(last_sequence, timeout=1.0)
                if snapshot is None:
                    continue
                remaining = next_send_at - self.server.clock()
                if remaining > 0:
                    time.sleep(remaining)
                latest, _age = self.server.fresh_snapshot()
                if latest is None:
                    last_sequence = snapshot.sequence
                    continue
                header = (
                    f"--{MJPEG_BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(latest.jpeg)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(latest.jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_sequence = latest.sequence
                next_send_at = self.server.clock() + interval
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a read-only corrected left-head camera preview over an SSH tunnel."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--stale-after", type=float, default=2.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.topic.startswith("/") or any(character.isspace() for character in args.topic):
        raise ValueError("topic must be an absolute ROS topic without whitespace")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not math.isfinite(args.fps) or not 0 < args.fps <= MAX_FPS:
        raise ValueError(f"fps must be greater than 0 and at most {MAX_FPS:g}")
    if not math.isfinite(args.stale_after) or args.stale_after <= 0:
        raise ValueError("stale-after must be positive")


def run_ros_preview(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable; source the R1 ROS environments first"
        ) from exc

    rclpy.init(args=None)

    frame_store = LatestJpeg()

    class HeadPreviewNode(Node):
        def __init__(self) -> None:
            super().__init__("jimmy_head_camera_web_preview")
            self.invalid_frames = 0
            self.subscription = self.create_subscription(
                CompressedImage, args.topic, self._on_frame, qos_profile_sensor_data
            )

        def _on_frame(self, message: CompressedImage) -> None:
            if not frame_store.update(message.data):
                self.invalid_frames += 1
                if self.invalid_frames == 1:
                    self.get_logger().error(
                        "received a non-JPEG or oversized frame; preview remains unavailable"
                    )

    node: HeadPreviewNode | None = None
    server: PreviewHTTPServer | None = None
    server_thread: threading.Thread | None = None
    try:
        server = PreviewHTTPServer(
            (LOOPBACK_ADDRESS, args.port),
            frame_store,
            topic=args.topic,
            fps=args.fps,
            stale_after=args.stale_after,
        )
        node = HeadPreviewNode()
        server_thread = threading.Thread(
            target=server.serve_forever, name="preview-http"
        )
        server_thread.start()
        print(
            f"Read-only preview listening on http://{LOOPBACK_ADDRESS}:{args.port}\n"
            f"Topic: {args.topic}\n"
            f"Tunnel rate limit: {args.fps:g} Hz\n"
            "No collection or robot-control HTTP endpoints are enabled.",
            flush=True,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if server is not None:
            if server_thread is not None:
                server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=3.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        return run_ros_preview(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
