#!/usr/bin/env python3
"""Serve the R1 Lite right-wrist ROS image as a temporary MJPEG webpage."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from typing import Any

import cv2
import numpy as np


class FrameStore:
    def __init__(self) -> None:
        self.jpeg: bytes | None = None
        self.condition = threading.Condition()

    def update(self, jpeg: bytes) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.condition.notify_all()


def image_to_bgr(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")
    channels = 4 if "a8" in encoding else 3
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        int(msg.height),
        int(msg.step),
    )
    image = raw[:, : int(msg.width) * channels].reshape(
        int(msg.height),
        int(msg.width),
        channels,
    )
    image = image[:, :, :3]
    if encoding.startswith("rgb"):
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def make_handler(store: FrameStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                body = (
                    b"<!doctype html><title>R1 right wrist</title>"
                    b"<style>body{margin:0;background:#111;text-align:center}"
                    b"img{max-width:100vw;max-height:100vh}</style>"
                    b'<img src="/stream.mjpg">'
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/stream.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with store.condition:
                        store.condition.wait(timeout=1.0)
                        jpeg = store.jpeg
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        default="/hdas/camera_wrist_right/color/image_raw",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    store = FrameStore()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rclpy.init()
    node = Node("capx_r1lite_wrist_web_preview")

    def callback(msg: Any) -> None:
        image = image_to_bgr(msg)
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, int(args.quality)],
        )
        if ok:
            store.update(encoded.tobytes())

    node.create_subscription(Image, args.topic, callback, qos_profile_sensor_data)
    print(f"Open http://127.0.0.1:{args.port}/")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
