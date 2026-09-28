"""Capture frames from the side USB camera (ROS Image topic)."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage


def _ros_image_to_rgb(msg: Any) -> np.ndarray:
    try:
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise RuntimeError("cv_bridge is required for side camera capture") from exc

    bridge = CvBridge()
    arr = bridge.imgmsg_to_cv2(msg, "passthrough")
    arr = np.asarray(arr)
    encoding = getattr(msg, "encoding", "").lower()
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return np.asarray(arr, dtype=np.uint8)


def _ensure_ros() -> None:
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import rospy
    except ImportError as exc:
        raise RuntimeError("rospy is required for side camera capture") from exc
    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_side_camera_capture", anonymous=True, disable_signals=True)


def capture_rgb(topic: str, *, timeout_s: float = 1.5) -> np.ndarray | None:
    """Read one RGB frame from a ROS image topic."""
    if not topic:
        return None
    try:
        _ensure_ros()
        import rospy
        from sensor_msgs.msg import Image

        msg = rospy.wait_for_message(topic, Image, timeout=timeout_s)
        return _ros_image_to_rgb(msg)
    except Exception:
        return None


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip())
    return cleaned[:80] or "snapshot"


def save_snapshot(
    topic: str,
    output_dir: str | Path,
    *,
    label: str,
    step: int,
    latest_name: str = "latest_rgb.png",
    timeout_s: float = 1.5,
) -> Path | None:
    """Save one side-camera frame; also refresh latest_rgb.png in parent folder."""
    rgb = capture_rgb(topic, timeout_s=timeout_s)
    if rgb is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _sanitize_label(label)
    ts = time.strftime("%H%M%S")
    path = output_dir / f"{step:04d}_{ts}_{safe_label}.png"
    PILImage.fromarray(rgb).save(path)

    latest_path = output_dir.parent / latest_name
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(rgb).save(latest_path)
    return path
