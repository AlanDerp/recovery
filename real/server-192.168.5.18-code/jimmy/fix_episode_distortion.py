#!/usr/bin/env python3
"""Create distortion-corrected copies of Galaxea ROS 2 MCAP episodes."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
import yaml

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import CameraInfo, CompressedImage
    from std_msgs.msg import String
except ImportError as exc:
    raise SystemExit(
        "ROS 2 Python packages are unavailable. Run:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source /home/nvidia/galaxea/install/setup.bash"
    ) from exc


LEFT_IMAGE_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
RIGHT_IMAGE_TOPIC = "/hdas/camera_head/right_raw/image_raw_color/compressed"
LEFT_INFO_TOPIC = "/calib/head_left/camera_info"
RIGHT_INFO_TOPIC = "/calib/head_right/camera_info"
MARKER_TOPIC = "/dataset/distortion_fix"

# Calibrated from 1,305 inlier correspondences on 2026-07-18.
LEFT_TO_RIGHT = np.array(
    (-0.0649659127, 0.0007217971, -0.0019771703), dtype=np.float64
).reshape(3, 1)
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


@dataclass(frozen=True)
class Rectification:
    image_size: tuple[int, int]
    left_maps: tuple[np.ndarray, np.ndarray]
    right_maps: tuple[np.ndarray, np.ndarray]
    left_projection: np.ndarray
    right_projection: np.ndarray


@dataclass
class QueuedMessage:
    topic: str
    data: bytes | Future[bytes]
    timestamp: int
    is_future: bool = False


def make_reader(uri: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return reader


def discover_bags(input_path: Path) -> list[Path]:
    source = input_path.expanduser().resolve()
    if source.is_file():
        if source.suffix != ".mcap":
            raise ValueError(f"Input file is not an MCAP: {source}")
        bags = [source]
    elif source.is_dir():
        bags = sorted(path for path in source.rglob("*.mcap") if path.is_file())
    else:
        raise ValueError(f"Input path does not exist: {source}")

    bags = [
        path
        for path in bags
        if not any("DISTORTION_FIXED" in part.upper() for part in path.parts)
    ]
    if not bags:
        raise ValueError(f"No raw MCAP files found under: {source}")
    return bags


def output_path_for(
    bag: Path,
    output_root: Optional[Path],
) -> tuple[Path, Path]:
    date_dir = next(
        (parent for parent in bag.parents if re.fullmatch(r"\d{8}", parent.name)),
        None,
    )
    date_text = date_dir.name if date_dir else datetime.now().strftime("%Y%m%d")
    base = (
        output_root.expanduser().resolve()
        if output_root is not None
        else (date_dir.parent if date_dir else bag.parent)
    )
    fixed_date_dir = base / f"{date_text}_DISTORTION_FIXED"

    episode_name = bag.parent.name
    if date_dir is not None and bag.parent == date_dir:
        episode_name = bag.stem
    fixed_episode = fixed_date_dir / f"{episode_name}_DISTORTION_FIXED"
    fixed_mcap = fixed_episode / f"{bag.stem}_DISTORTION_FIXED.mcap"
    return fixed_episode, fixed_mcap


def first_camera_info(bag: Path) -> tuple[CameraInfo, CameraInfo]:
    left: Optional[CameraInfo] = None
    right: Optional[CameraInfo] = None
    reader = make_reader(bag)
    while reader.has_next() and (left is None or right is None):
        topic, data, _timestamp = reader.read_next()
        if topic == LEFT_INFO_TOPIC and left is None:
            left = deserialize_message(data, CameraInfo)
        elif topic == RIGHT_INFO_TOPIC and right is None:
            right = deserialize_message(data, CameraInfo)
    if left is None or right is None:
        missing = []
        if left is None:
            missing.append(LEFT_INFO_TOPIC)
        if right is None:
            missing.append(RIGHT_INFO_TOPIC)
        raise ValueError("Missing required CameraInfo: " + ", ".join(missing))
    if not left.width or not left.height:
        raise ValueError("Head CameraInfo has an invalid zero image size")
    return left, right


def scaled_k(info: CameraInfo, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    k = np.asarray(info.k, dtype=np.float64).reshape(3, 3).copy()
    if info.width and info.height and (info.width, info.height) != image_size:
        sx = width / float(info.width)
        sy = height / float(info.height)
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
    return k


def build_rectification(
    left: CameraInfo,
    right: CameraInfo,
    alpha: float,
) -> Rectification:
    image_size = (int(left.width), int(left.height))
    k1 = scaled_k(left, image_size)
    k2 = scaled_k(right, image_size)
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
    return Rectification(image_size, left_maps, right_maps, p1, p2)


def rectified_camera_info(
    original: CameraInfo,
    projection: np.ndarray,
    image_size: tuple[int, int],
) -> CameraInfo:
    width, height = image_size
    output = CameraInfo()
    output.header = original.header
    output.height = height
    output.width = width
    output.distortion_model = "plumb_bob"
    output.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    output.k = [
        float(projection[0, 0]),
        float(projection[0, 1]),
        float(projection[0, 2]),
        0.0,
        float(projection[1, 1]),
        float(projection[1, 2]),
        0.0,
        0.0,
        1.0,
    ]
    output.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    output.p = [float(value) for value in projection.reshape(-1)]
    output.binning_x = original.binning_x
    output.binning_y = original.binning_y
    output.roi = original.roi
    return output


def rectify_compressed(
    serialized: bytes,
    maps: tuple[np.ndarray, np.ndarray],
    image_size: tuple[int, int],
    jpeg_quality: int,
) -> bytes:
    message = deserialize_message(serialized, CompressedImage)
    encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode a head-camera JPEG")
    actual_size = (image.shape[1], image.shape[0])
    if actual_size != image_size:
        raise ValueError(
            f"Head image size changed: expected {image_size}, received {actual_size}"
        )
    rectified = cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)
    ok, output = cv2.imencode(
        ".jpg",
        rectified,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise ValueError("OpenCV could not encode a corrected head-camera JPEG")
    message.format = "jpeg"
    message.data = output.tobytes()
    return serialize_message(message)


def rewrite_metadata(
    bag_dir: Path,
    output_mcap_name: str,
    annotation: dict,
) -> None:
    metadata_path = bag_dir / "metadata.yaml"
    document = yaml.safe_load(metadata_path.read_text())
    info = document["rosbag2_bagfile_information"]
    old_paths = list(info.get("relative_file_paths", []))
    old_name = old_paths[0] if old_paths else None
    info["relative_file_paths"] = [output_mcap_name]
    for file_info in info.get("files", []):
        if old_name is None or file_info.get("path") == old_name:
            file_info["path"] = output_mcap_name
    document["distortion_fix"] = annotation
    metadata_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def copy_episode_sidecars(source_bag: Path, output_dir: Path) -> None:
    for source in source_bag.parent.iterdir():
        if not source.is_file():
            continue
        if source == source_bag or source.name == "metadata.yaml":
            continue
        shutil.copy2(source, output_dir / source.name)


def install_output(
    staged_bag_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        os.replace(staged_bag_dir, output_dir)
        return
    if not overwrite:
        raise FileExistsError(
            f"Corrected output already exists: {output_dir}\n"
            "Use --overwrite to replace the corrected copy."
        )
    if "DISTORTION_FIXED" not in output_dir.name.upper():
        raise ValueError(f"Refusing to overwrite an unsafe path: {output_dir}")

    backup = output_dir.parent / (
        f".{output_dir.name}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.replace(output_dir, backup)
    try:
        os.replace(staged_bag_dir, output_dir)
    except Exception:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def process_bag(
    source_bag: Path,
    output_dir: Path,
    output_mcap: Path,
    args: argparse.Namespace,
) -> dict:
    topic_reader = make_reader(source_bag)
    topics = topic_reader.get_all_topics_and_types()
    topic_names = {topic.name for topic in topics}
    missing = {
        LEFT_IMAGE_TOPIC,
        RIGHT_IMAGE_TOPIC,
        LEFT_INFO_TOPIC,
        RIGHT_INFO_TOPIC,
    } - topic_names
    if missing:
        raise ValueError(
            "Bag does not contain the original raw head-camera packet: "
            + ", ".join(sorted(missing))
        )

    left_info, right_info = first_camera_info(source_bag)
    rectification = build_rectification(left_info, right_info, args.alpha)
    processed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    annotation = {
        "distortion_fixed": True,
        "processed_at": processed_at,
        "source_mcap": str(source_bag),
        "algorithm": "OpenCV stereoRectify + remap",
        "calibration": "Galaxea R1 Lite stereo calibration 2026-07-18",
        "alpha": args.alpha,
        "jpeg_quality": args.jpeg_quality,
        "image_size": {
            "width": rectification.image_size[0],
            "height": rectification.image_size[1],
        },
        "rewritten_topics": [LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC],
        "camera_info_topics": [LEFT_INFO_TOPIC, RIGHT_INFO_TOPIC],
        "original_preserved": True,
    }
    marker = String()
    marker.data = json.dumps(annotation, separators=(",", ":"), sort_keys=True)
    marker_data = serialize_message(marker)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".distortion_fix-", dir=output_dir.parent
    ) as temporary:
        staging_root = Path(temporary)
        writer_uri = staging_root / "bag"
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(writer_uri), storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        for topic in topics:
            writer.create_topic(topic)
        if MARKER_TOPIC not in topic_names:
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=MARKER_TOPIC,
                    type="std_msgs/msg/String",
                    serialization_format="cdr",
                )
            )

        cpu_count = max(1, os.cpu_count() or 1)
        cv2.setNumThreads(max(1, cpu_count // args.threads))
        reader = make_reader(source_bag)
        queue: deque[QueuedMessage] = deque()
        pending_images = 0
        corrected_left = 0
        corrected_right = 0
        total_messages = 0
        marker_written = False
        next_progress = 100
        started = time.monotonic()

        def flush_one() -> None:
            nonlocal pending_images
            queued = queue.popleft()
            payload = queued.data.result() if queued.is_future else queued.data
            if queued.is_future:
                pending_images -= 1
            writer.write(queued.topic, payload, queued.timestamp)

        with ThreadPoolExecutor(
            max_workers=args.threads,
            thread_name_prefix="head-rectify",
        ) as executor:
            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                total_messages += 1
                if not marker_written:
                    writer.write(MARKER_TOPIC, marker_data, timestamp)
                    marker_written = True

                if topic == LEFT_IMAGE_TOPIC:
                    future = executor.submit(
                        rectify_compressed,
                        data,
                        rectification.left_maps,
                        rectification.image_size,
                        args.jpeg_quality,
                    )
                    queue.append(QueuedMessage(topic, future, timestamp, True))
                    pending_images += 1
                    corrected_left += 1
                elif topic == RIGHT_IMAGE_TOPIC:
                    future = executor.submit(
                        rectify_compressed,
                        data,
                        rectification.right_maps,
                        rectification.image_size,
                        args.jpeg_quality,
                    )
                    queue.append(QueuedMessage(topic, future, timestamp, True))
                    pending_images += 1
                    corrected_right += 1
                elif topic == LEFT_INFO_TOPIC:
                    info = deserialize_message(data, CameraInfo)
                    fixed = rectified_camera_info(
                        info,
                        rectification.left_projection,
                        rectification.image_size,
                    )
                    queue.append(
                        QueuedMessage(topic, serialize_message(fixed), timestamp)
                    )
                elif topic == RIGHT_INFO_TOPIC:
                    info = deserialize_message(data, CameraInfo)
                    fixed = rectified_camera_info(
                        info,
                        rectification.right_projection,
                        rectification.image_size,
                    )
                    queue.append(
                        QueuedMessage(topic, serialize_message(fixed), timestamp)
                    )
                else:
                    queue.append(QueuedMessage(topic, data, timestamp))

                if pending_images >= args.threads * 2:
                    before = pending_images
                    while queue and pending_images == before:
                        flush_one()

                corrected = corrected_left + corrected_right
                if corrected >= next_progress:
                    elapsed = max(0.001, time.monotonic() - started)
                    print(
                        f"  corrected {corrected} head frames "
                        f"({corrected / elapsed:.1f} frames/s)",
                        end="\r",
                        flush=True,
                    )
                    next_progress += 100

            while queue:
                flush_one()
        writer.close()
        print(" " * 80, end="\r")

        generated_mcaps = sorted(writer_uri.glob("*.mcap"))
        if len(generated_mcaps) != 1:
            raise RuntimeError(
                f"Expected one generated MCAP, found {len(generated_mcaps)}"
            )
        generated_mcap = generated_mcaps[0]
        target_inside_stage = writer_uri / output_mcap.name
        generated_mcap.rename(target_inside_stage)
        rewrite_metadata(writer_uri, output_mcap.name, annotation)
        (writer_uri / "distortion_fix.json").write_text(
            json.dumps(annotation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        copy_episode_sidecars(source_bag, writer_uri)
        install_output(writer_uri, output_dir, args.overwrite)

    elapsed = time.monotonic() - started
    return {
        "source": str(source_bag),
        "output": str(output_mcap),
        "left_frames": corrected_left,
        "right_frames": corrected_right,
        "messages": total_messages,
        "seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create corrected MCAP copies beside a Galaxea dataset. The original "
            "recordings are never modified."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="an MCAP file, episode directory, date directory, or dataset root",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        help="optional output root; default is beside the original date folder",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        choices=range(1, 101),
        metavar="1..100",
        default=95,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        choices=(0.0, 1.0),
        default=0.0,
        help="0 crops invalid borders; 1 retains the full corrected field of view",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 1) // 2)),
        help="parallel head-frame workers (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing DISTORTION_FIXED copy; raw input is untouched",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    try:
        bags = discover_bags(args.path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Found {len(bags)} raw MCAP file(s).")
    failures = []
    results = []
    for index, source in enumerate(bags, start=1):
        output_dir, output_mcap = output_path_for(source, args.output_root)
        print(f"[{index}/{len(bags)}] {source}")
        print(f"  -> {output_mcap}")
        try:
            result = process_bag(source, output_dir, output_mcap, args)
        except Exception as exc:
            failures.append((source, str(exc)))
            print(f"  FAILED: {exc}", file=sys.stderr)
        else:
            results.append(result)
            print(
                "  complete: "
                f"{result['left_frames']} left + {result['right_frames']} right "
                f"frames in {result['seconds']:.1f}s"
            )

    print(f"Completed {len(results)} of {len(bags)} MCAP file(s).")
    if failures:
        for source, error in failures:
            print(f"FAILED {source}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
