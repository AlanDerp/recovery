#!/usr/bin/env python3
"""Convert a Galaxea MCAP episode into a labeled four-camera MP4."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo, CompressedImage
except ImportError as exc:
    raise SystemExit(
        "ROS 2 Python packages are unavailable. Run:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source /home/nvidia/galaxea/install/setup.bash"
    ) from exc


HEAD_LEFT_RECT = "/capx/head_stereo/left/image_rect/compressed"
HEAD_RIGHT_RECT = "/capx/head_stereo/right/image_rect/compressed"
HEAD_LEFT_RAW = "/hdas/camera_head/left_raw/image_raw_color/compressed"
HEAD_RIGHT_RAW = "/hdas/camera_head/right_raw/image_raw_color/compressed"
HEAD_LEFT_INFO = "/calib/head_left/camera_info"
HEAD_RIGHT_INFO = "/calib/head_right/camera_info"

WRIST_LEFT_RECT = "/hdas/camera_wrist_left/color/image_rect_raw/compressed"
WRIST_RIGHT_RECT = "/hdas/camera_wrist_right/color/image_rect_raw/compressed"
WRIST_LEFT_RAW = "/hdas/camera_wrist_left/color/image_raw/compressed"
WRIST_RIGHT_RAW = "/hdas/camera_wrist_right/color/image_raw/compressed"
DISTORTION_FIX_MARKER = "/dataset/distortion_fix"

LEFT_TO_RIGHT = np.array(
    (-0.0649659127, 0.0007217971, -0.0019771703), dtype=np.float64
)
LEFT_TO_RIGHT_ROTATION = np.array(
    (
        0.9999213396,
        0.0121167089,
        0.0032403765,
        -0.0121361831,
        0.9999079948,
        0.0060592902,
        -0.0031666597,
        -0.0060981393,
        0.9999763922,
    ),
    dtype=np.float64,
).reshape(3, 3)

CAMERAS = ("head_left", "head_right", "wrist_left", "wrist_right")
LABELS = {
    "head_left": "HEAD - LEFT - RECTIFIED",
    "head_right": "HEAD - RIGHT - RECTIFIED",
    "wrist_left": "WRIST - LEFT",
    "wrist_right": "WRIST - RIGHT",
}


def make_reader(uri: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return reader


def resolve_bag(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix == ".mcap":
        return path
    if path.is_dir():
        files = sorted(path.glob("*.mcap"))
        if len(files) == 1:
            return files[0]
        if not files:
            raise SystemExit(f"No .mcap file found in {path}")
        raise SystemExit(f"More than one .mcap file found in {path}; specify one")
    raise SystemExit(f"MCAP input does not exist: {path}")


def scan_bag(uri: Path) -> tuple[Counter, dict[str, CameraInfo], bool]:
    candidates = {
        HEAD_LEFT_RECT,
        HEAD_RIGHT_RECT,
        HEAD_LEFT_RAW,
        HEAD_RIGHT_RAW,
        WRIST_LEFT_RECT,
        WRIST_RIGHT_RECT,
        WRIST_LEFT_RAW,
        WRIST_RIGHT_RAW,
    }
    info_topics = {HEAD_LEFT_INFO, HEAD_RIGHT_INFO}
    counts: Counter = Counter()
    camera_info: dict[str, CameraInfo] = {}
    distortion_fixed = False
    reader = make_reader(uri)
    while reader.has_next():
        topic, data, _timestamp = reader.read_next()
        if topic in candidates:
            counts[topic] += 1
        elif topic in info_topics and topic not in camera_info:
            camera_info[topic] = deserialize_message(data, CameraInfo)
        elif topic == DISTORTION_FIX_MARKER:
            distortion_fixed = True
    return counts, camera_info, distortion_fixed


def choose_topic(counts: Counter, preferred: str, fallback: str) -> Optional[str]:
    if counts[preferred]:
        return preferred
    if counts[fallback]:
        return fallback
    return None


def build_head_maps(
    camera_info: dict[str, CameraInfo],
    alpha: float,
) -> Optional[dict[str, tuple[np.ndarray, np.ndarray]]]:
    left = camera_info.get(HEAD_LEFT_INFO)
    right = camera_info.get(HEAD_RIGHT_INFO)
    if left is None or right is None:
        return None
    if not left.width or not left.height:
        return None

    image_size = (int(left.width), int(left.height))
    k1 = np.asarray(left.k, dtype=np.float64).reshape(3, 3)
    k2 = np.asarray(right.k, dtype=np.float64).reshape(3, 3)
    d1 = np.asarray(left.d, dtype=np.float64)
    d2 = np.asarray(right.d, dtype=np.float64)
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        k1,
        d1,
        k2,
        d2,
        image_size,
        LEFT_TO_RIGHT_ROTATION,
        LEFT_TO_RIGHT,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    left_maps = cv2.initUndistortRectifyMap(
        k1, d1, r1, p1, image_size, cv2.CV_16SC2
    )
    right_maps = cv2.initUndistortRectifyMap(
        k2, d2, r2, p2, image_size, cv2.CV_16SC2
    )
    return {"head_left": left_maps, "head_right": right_maps}


def decode_compressed(data: bytes) -> Optional[np.ndarray]:
    msg = deserialize_message(data, CompressedImage)
    encoded = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def fit_to_cell(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    target_w = max(1, int(round(source_w * scale)))
    target_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    cell = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - target_w) // 2
    y = (height - target_h) // 2
    cell[y : y + target_h, x : x + target_w] = resized
    return cell


def draw_label(cell: np.ndarray, label: str) -> None:
    cv2.rectangle(cell, (0, 0), (420, 42), (20, 20, 20), thickness=-1)
    cv2.rectangle(cell, (0, 0), (6, 42), (155, 225, 65), thickness=-1)
    cv2.putText(
        cell,
        label,
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def placeholder(width: int, height: int, label: str) -> np.ndarray:
    cell = np.full((height, width, 3), 24, dtype=np.uint8)
    text = f"{label}: NO DATA"
    size, _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = max(10, (width - size[0]) // 2)
    y = height // 2
    cv2.putText(
        cell,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    return cell


def render_mosaic(
    frames: dict[str, np.ndarray],
    output_size: tuple[int, int],
) -> np.ndarray:
    output_w, output_h = output_size
    cell_w, cell_h = output_w // 2, output_h // 2
    cells = []
    for camera in CAMERAS:
        image = frames.get(camera)
        cell = (
            fit_to_cell(image, cell_w, cell_h)
            if image is not None
            else placeholder(cell_w, cell_h, LABELS[camera])
        )
        draw_label(cell, LABELS[camera])
        cells.append(cell)
    return np.vstack((np.hstack(cells[:2]), np.hstack(cells[2:])))


def process_episode(args: argparse.Namespace) -> None:
    bag = resolve_bag(args.bag)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    counts, camera_info, distortion_fixed = scan_bag(bag)
    selected = {
        "head_left": choose_topic(counts, HEAD_LEFT_RECT, HEAD_LEFT_RAW),
        "head_right": choose_topic(counts, HEAD_RIGHT_RECT, HEAD_RIGHT_RAW),
        "wrist_left": choose_topic(counts, WRIST_LEFT_RECT, WRIST_LEFT_RAW),
        "wrist_right": choose_topic(counts, WRIST_RIGHT_RECT, WRIST_RIGHT_RAW),
    }
    topic_to_camera = {topic: camera for camera, topic in selected.items() if topic}
    if not topic_to_camera:
        raise SystemExit("The bag contains none of the supported camera topics")

    raw_head_selected = any(
        selected[camera] in {HEAD_LEFT_RAW, HEAD_RIGHT_RAW}
        for camera in ("head_left", "head_right")
    )
    head_maps = (
        build_head_maps(camera_info, args.alpha)
        if raw_head_selected and not distortion_fixed
        else None
    )
    if raw_head_selected and not distortion_fixed and head_maps is None:
        print("WARNING: raw head images will remain distorted; CameraInfo is missing")
    if distortion_fixed:
        print("distortion  : fixed packet marker found; no second correction applied")

    for camera in CAMERAS:
        topic = selected[camera]
        count = counts[topic] if topic else 0
        print(f"{camera:12s}: {topic or 'MISSING'} ({count} messages)")

    output_size = (args.width, args.height)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        output_size,
    )
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open MP4 writer: {output}")

    frame_period_ns = int(1_000_000_000 / args.fps)
    max_duration_ns = (
        int(args.max_seconds * 1_000_000_000) if args.max_seconds > 0 else None
    )
    frames: dict[str, np.ndarray] = {}
    next_decode: dict[str, int] = {}
    next_output_ns: Optional[int] = None
    video_start_ns: Optional[int] = None
    written = 0

    reader = make_reader(bag)
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        camera = topic_to_camera.get(topic)
        if camera is None:
            continue
        if timestamp < next_decode.get(camera, 0):
            continue

        image = decode_compressed(data)
        if image is None:
            continue
        if camera in {"head_left", "head_right"} and selected[camera] in {
            HEAD_LEFT_RAW,
            HEAD_RIGHT_RAW,
        }:
            if head_maps is not None:
                map_x, map_y = head_maps[camera]
                image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        frames[camera] = image
        next_decode[camera] = timestamp + frame_period_ns
        if video_start_ns is None:
            video_start_ns = timestamp
            next_output_ns = timestamp

        if max_duration_ns is not None and timestamp - video_start_ns > max_duration_ns:
            break
        if next_output_ns is None or timestamp < next_output_ns:
            continue

        writer.write(render_mosaic(frames, output_size))
        written += 1
        next_output_ns += frame_period_ns

    writer.release()
    if not written:
        output.unlink(missing_ok=True)
        raise SystemExit("No MP4 frames were written")

    duration = written / args.fps
    print(f"Wrote {written} frames ({duration:.2f}s) to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="episode .mcap file or its directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/home/nvidia/jimmy/four_camera.mp4"),
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="trim output to this duration; 0 keeps the full episode",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="raw-head undistortion crop: 0 crops borders, 1 keeps full FOV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    process_episode(parse_args())
