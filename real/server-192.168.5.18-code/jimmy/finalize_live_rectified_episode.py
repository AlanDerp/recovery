#!/usr/bin/env python3
"""Package an already-rectified Galaxea episode without re-encoding images."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
import yaml


LEFT_IMAGE = "/hdas/camera_head/left_raw/image_raw_color/compressed"
RIGHT_IMAGE = "/hdas/camera_head/right_raw/image_raw_color/compressed"
LEFT_INFO = "/calib/head_left/camera_info"
RIGHT_INFO = "/calib/head_right/camera_info"
MARKER_TOPIC = "/dataset/distortion_fix"
CALIBRATION = Path("/opt/galaxea/body/stereo_ros.yaml")


def reader_for(path: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return reader


def find_mcap(source: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_file() and source.suffix == ".mcap":
        return source
    if source.is_dir():
        files = sorted(source.glob("*.mcap"))
        if len(files) == 1:
            return files[0]
    raise ValueError("Input must be an episode directory or one MCAP file")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rectified_input(mcap: Path) -> tuple[list, int]:
    reader = reader_for(mcap)
    topics = reader.get_all_topics_and_types()
    topic_names = {topic.name for topic in topics}
    required = {LEFT_IMAGE, RIGHT_IMAGE, LEFT_INFO, RIGHT_INFO}
    missing = required - topic_names
    if missing:
        raise ValueError("Missing required topics: " + ", ".join(sorted(missing)))

    camera_info: dict[str, CameraInfo] = {}
    first_timestamp = 0
    while reader.has_next() and len(camera_info) < 2:
        topic, data, timestamp = reader.read_next()
        if first_timestamp == 0:
            first_timestamp = timestamp
        if topic in {LEFT_INFO, RIGHT_INFO} and topic not in camera_info:
            camera_info[topic] = deserialize_message(data, CameraInfo)

    for topic in (LEFT_INFO, RIGHT_INFO):
        info = camera_info.get(topic)
        if info is None:
            raise ValueError(f"No CameraInfo messages found on {topic}")
        if (info.width, info.height) != (1920, 1536):
            raise ValueError(f"Unexpected head image size on {topic}")
        if not info.d or any(abs(value) > 1e-12 for value in info.d):
            raise ValueError(
                f"{topic} is not marked rectified: distortion coefficients are nonzero"
            )
    if abs(camera_info[RIGHT_INFO].p[3]) < 1e-9:
        raise ValueError("Right CameraInfo has no stereo baseline in P[3]")
    return topics, first_timestamp


def output_paths(mcap: Path, output_root: Path) -> tuple[Path, Path]:
    date = next(
        (parent.name for parent in mcap.parents if re.fullmatch(r"\d{8}", parent.name)),
        datetime.now().strftime("%Y%m%d"),
    )
    episode = mcap.parent.name
    output_dir = (
        output_root.expanduser().resolve()
        / f"{date}_DISTORTION_FIXED"
        / f"{episode}_DISTORTION_FIXED"
    )
    output_mcap = output_dir / f"{mcap.stem}_DISTORTION_FIXED.mcap"
    return output_dir, output_mcap


def rewrite_metadata(
    metadata_path: Path,
    output_mcap_name: str,
    annotation: dict,
) -> None:
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = document["rosbag2_bagfile_information"]
    info["relative_file_paths"] = [output_mcap_name]
    for item in info.get("files", []):
        item["path"] = output_mcap_name
    document["distortion_fix"] = annotation
    metadata_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def package(source: Path, output_root: Path) -> Path:
    mcap = find_mcap(source)
    topics, first_timestamp = validate_rectified_input(mcap)
    output_dir, output_mcap = output_paths(mcap, output_root)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    annotation = {
        "distortion_fixed": True,
        "correction_stage": "live_before_rosbag_recording",
        "head_camera_only": True,
        "wrist_cameras_modified": False,
        "image_reencoded_during_finalization": False,
        "all_original_serialized_messages_preserved": True,
        "backend": "NVIDIA VPI 3.1 CUDA remap",
        "calibration_file": str(CALIBRATION),
        "calibration_sha256": sha256(CALIBRATION),
        "head_resolution": [1920, 1536],
        "jpeg_quality": 90,
        "processed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_mcap": str(mcap),
        "head_image_topics": [LEFT_IMAGE, RIGHT_IMAGE],
        "head_camera_info_topics": [LEFT_INFO, RIGHT_INFO],
    }
    marker = String()
    marker.data = json.dumps(annotation, separators=(",", ":"), sort_keys=True)
    marker_data = serialize_message(marker)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".live_rectified-", dir=output_dir.parent
    ) as temp:
        staged = Path(temp) / "episode"
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(staged), storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        topic_names = {topic.name for topic in topics}
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

        reader = reader_for(mcap)
        marker_written = False
        messages = 0
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if not marker_written:
                writer.write(
                    MARKER_TOPIC,
                    marker_data,
                    first_timestamp or timestamp,
                )
                marker_written = True
            writer.write(topic, data, timestamp)
            messages += 1
        writer.close()

        generated = sorted(staged.glob("*.mcap"))
        if len(generated) != 1:
            raise RuntimeError(f"Expected one generated MCAP, found {len(generated)}")
        staged_mcap = staged / output_mcap.name
        generated[0].rename(staged_mcap)
        rewrite_metadata(staged / "metadata.yaml", output_mcap.name, annotation)
        (staged / "distortion_fix.json").write_text(
            json.dumps(annotation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for sidecar in mcap.parent.iterdir():
            if (
                sidecar.is_file()
                and sidecar != mcap
                and sidecar.name != "metadata.yaml"
            ):
                shutil.copy2(sidecar, staged / sidecar.name)
        os.replace(staged, output_dir)

    print(f"Packaged {messages} original messages plus one marker.")
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/nvidia/jimmy"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        package(args.source, args.output_root)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
