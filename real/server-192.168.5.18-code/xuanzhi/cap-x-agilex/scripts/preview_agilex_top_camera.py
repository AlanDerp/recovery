#!/usr/bin/env python3
"""Preview the fixed top camera (/camera_h) for Agilex pipeline startup.

Saves the latest RGB frame to disk and optionally opens an OpenCV window when a
display is available.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

DEFAULT_TOPIC = "/camera_h/color/image_raw"
DEFAULT_OUTPUT = "outputs/agilex_top_camera/latest_rgb.png"
WINDOW_NAME = "camera_h top view (q=quit)"


def _should_show(show_flag: str) -> bool:
    normalized = show_flag.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _ros_image_to_rgb(msg: Any) -> np.ndarray:
    try:
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise RuntimeError("cv_bridge is required to read ROS Image messages") from exc

    bridge = CvBridge()
    arr = bridge.imgmsg_to_cv2(msg, "passthrough")
    arr = np.asarray(arr)
    encoding = getattr(msg, "encoding", "").lower()

    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported image shape from ROS: {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return np.asarray(arr, dtype=np.uint8)


def _init_ros() -> None:
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    try:
        import rospy
    except ImportError as exc:
        raise RuntimeError("rospy is required for camera preview") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_top_camera_preview", anonymous=True, disable_signals=True)


def wait_for_ros_image(topic: str, timeout_s: float) -> np.ndarray:
    _init_ros()
    import rospy
    from sensor_msgs.msg import Image

    print(f"[preview] waiting for one frame on {topic} timeout={timeout_s}s")
    msg = rospy.wait_for_message(topic, Image, timeout=timeout_s)
    rgb = _ros_image_to_rgb(msg)
    print(
        "[preview] received frame "
        f"shape={rgb.shape} encoding={getattr(msg, 'encoding', 'unknown')} "
        f"stamp={msg.header.stamp.to_sec():.6f}"
    )
    return rgb


def save_frame(rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(rgb).save(output_path)


def show_frame(rgb: np.ndarray) -> bool:
    try:
        import cv2
    except ImportError:
        print("[preview] OpenCV not installed; skipping window preview")
        return False

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, rgb[:, :, ::-1])
        return True
    except cv2.error as exc:
        print(f"[preview] OpenCV window unavailable: {exc}")
        return False


def run_once(args: argparse.Namespace) -> int:
    rgb = wait_for_ros_image(args.topic, args.timeout)
    output_path = Path(args.output)
    save_frame(rgb, output_path)
    print(f"[preview] saved {output_path.resolve()}")

    if not _should_show(args.show):
        return 0

    if not show_frame(rgb):
        return 0

    import cv2

    if args.hold_s > 0:
        print(f"[preview] showing window for {args.hold_s:.1f}s")
        deadline = time.time() + args.hold_s
        while time.time() < deadline:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), 27):
                break
    else:
        print("[preview] press q in the preview window to close")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


def run_watch(args: argparse.Namespace) -> int:
    _init_ros()
    import rospy
    from sensor_msgs.msg import Image

    latest: dict[str, np.ndarray | None] = {"rgb": None}

    def callback(msg: Any) -> None:
        latest["rgb"] = _ros_image_to_rgb(msg)

    rospy.Subscriber(args.topic, Image, callback, queue_size=1)
    output_path = Path(args.output)
    show_enabled = _should_show(args.show)
    window_ok = False

    print(
        f"[preview] watching {args.topic}, updating {output_path.resolve()} "
        f"every {args.interval:.1f}s"
    )
    while not rospy.is_shutdown():
        rgb = latest["rgb"]
        if rgb is not None:
            save_frame(rgb, output_path)
            if show_enabled and not window_ok:
                window_ok = show_frame(rgb)
            if window_ok:
                import cv2

                cv2.imshow(WINDOW_NAME, rgb[:, :, ::-1])
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[preview] quit requested from preview window")
                    break
        time.sleep(max(args.interval, 0.1))

    if window_ok:
        import cv2

        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview Agilex top camera_h RGB stream.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("once", "watch"), default="once")
    parser.add_argument("--interval", type=float, default=1.0, help="watch mode refresh seconds")
    parser.add_argument("--timeout", type=float, default=30.0, help="once mode ROS wait seconds")
    parser.add_argument(
        "--show",
        default="auto",
        help="Show OpenCV window: auto|1|0 (auto uses DISPLAY/WAYLAND_DISPLAY)",
    )
    parser.add_argument(
        "--hold-s",
        type=float,
        default=0.0,
        help="once mode: auto-close preview window after N seconds (0=wait for q)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "watch":
        raise SystemExit(run_watch(args))
    raise SystemExit(run_once(args))


if __name__ == "__main__":
    main()
