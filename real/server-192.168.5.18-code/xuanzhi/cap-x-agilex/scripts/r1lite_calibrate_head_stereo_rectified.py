#!/usr/bin/env python3
"""Convert the saved raw head-camera plane calibration to stereo rectified pixels."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capx.integrations.r1lite.head_stereo_calibration import (  # noqa: E402
    build_rectified_plane_calibration,
)
from r1lite_head_stereo_depth import (  # noqa: E402
    DEFAULT_LEFT_TO_RIGHT,
    DEFAULT_LEFT_TO_RIGHT_ROTATION,
    LEFT_INFO_TOPIC,
    RIGHT_INFO_TOPIC,
)


DEFAULT_INPUT = REPO_ROOT / "outputs/r1lite/head_plane_calibration.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/r1lite/head_stereo_rectified_plane_calibration.json"
)


class CameraInfoPair(Node):
    def __init__(self, left_topic: str, right_topic: str) -> None:
        super().__init__("r1lite_rectified_calibration_camera_info")
        self.left: CameraInfo | None = None
        self.right: CameraInfo | None = None
        self.create_subscription(
            CameraInfo, left_topic, self._on_left, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, right_topic, self._on_right, qos_profile_sensor_data
        )

    def _on_left(self, message: CameraInfo) -> None:
        self.left = message

    def _on_right(self, message: CameraInfo) -> None:
        self.right = message


def _camera_payload(message: CameraInfo) -> dict[str, Any]:
    return {
        "camera_matrix": list(message.k),
        "distortion": list(message.d),
        "size": [int(message.width), int(message.height)],
        "distortion_model": str(message.distortion_model),
    }


def wait_for_camera_info(
    left_topic: str, right_topic: str, timeout_s: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    rclpy.init()
    node = CameraInfoPair(left_topic, right_topic)
    deadline = time.monotonic() + float(timeout_s)
    try:
        while (
            (node.left is None or node.right is None)
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.left is None or node.right is None:
            raise TimeoutError(
                "timed out waiting for raw left/right CameraInfo topics"
            )
        return _camera_payload(node.left), _camera_payload(node.right)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--left-info", default=LEFT_INFO_TOPIC)
    parser.add_argument("--right-info", default=RIGHT_INFO_TOPIC)
    parser.add_argument("--processing-width", type=int, default=640)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--left-to-right",
        nargs=3,
        type=float,
        default=DEFAULT_LEFT_TO_RIGHT,
    )
    parser.add_argument(
        "--left-to-right-rotation",
        nargs=9,
        type=float,
        default=DEFAULT_LEFT_TO_RIGHT_ROTATION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_calibration = json.loads(args.input.read_text())
    raw_width, raw_height = [
        int(value) for value in raw_calibration["image_size"]
    ]
    target_width = int(args.processing_width)
    target_height = int(round(raw_height * target_width / raw_width))
    left, right = wait_for_camera_info(
        args.left_info, args.right_info, args.timeout
    )
    calibration = build_rectified_plane_calibration(
        raw_calibration,
        left_camera_matrix=left["camera_matrix"],
        left_distortion=left["distortion"],
        left_info_size=left["size"],
        right_camera_matrix=right["camera_matrix"],
        right_distortion=right["distortion"],
        right_info_size=right["size"],
        left_to_right_rotation=args.left_to_right_rotation,
        left_to_right_translation_m=args.left_to_right,
        target_size=(target_width, target_height),
        alpha=args.alpha,
    )
    calibration["camera_info"] = {"left": left, "right": right}
    calibration["source_calibration_path"] = str(args.input.resolve())
    calibration["rectification_implementation"] = (
        "cv2.stereoRectify_plus_undistortPoints_matching_"
        "r1lite_head_stereo_depth"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = args.output.with_name(
            f"{args.output.stem}_backup_{stamp}{args.output.suffix}"
        )
        shutil.copy2(args.output, backup)
        print(f"backed up previous calibration: {backup}")
    args.output.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote rectified calibration: {args.output}")
    print(
        "reprojection: "
        f"rmse={calibration['reprojection_rmse_px']:.4f}px, "
        f"max={calibration['maximum_reprojection_error_px']:.4f}px"
    )


if __name__ == "__main__":
    main()
