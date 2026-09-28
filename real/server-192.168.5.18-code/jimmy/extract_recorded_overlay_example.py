#!/usr/bin/env python3
"""Extract a reference, changed frame, and reset overlay from one MCAP.

The source bag is opened read-only through rosbag2_py.  This tool never plays
the bag, publishes ROS messages, or writes inside the source episode directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

from head_reset_overlay import DEFAULT_TOPIC, compose_overlay, draw_banner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", type=Path, help="Source ROS 2 MCAP file")
    parser.add_argument("output_dir", type=Path, help="New derivative directory")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument(
        "--minimum-separation",
        type=float,
        default=2.0,
        help="Ignore candidates this many seconds after the reference",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Score every Nth valid topic frame",
    )
    return parser


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    source = args.mcap.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".mcap":
        raise ValueError(f"source is not an MCAP file: {source}")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    if is_within(output, source.parent):
        raise ValueError("output directory must not be inside the raw episode directory")
    if not args.topic.startswith("/") or any(character.isspace() for character in args.topic):
        raise ValueError("topic must be an absolute ROS topic without whitespace")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if args.minimum_separation < 0.0:
        raise ValueError("minimum-separation must be non-negative")
    if args.sample_stride < 1:
        raise ValueError("sample-stride must be at least 1")
    return source, output


def decode_frame(message: CompressedImage) -> tuple[bytes, np.ndarray] | None:
    jpeg = bytes(message.data)
    if len(jpeg) < 4 or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        return None
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return jpeg, image


def thumbnail(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(reduced, (5, 5), 0)


def difference_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    pixel_difference = float(cv2.mean(cv2.absdiff(reference, candidate))[0])
    reference_edges = cv2.Canny(reference, 70, 150)
    candidate_edges = cv2.Canny(candidate, 70, 150)
    edge_difference = float(cv2.mean(cv2.bitwise_xor(reference_edges, candidate_edges))[0])
    return pixel_difference + 0.25 * edge_difference


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write image: {path}")


def comparison_card(image: np.ndarray, title: str, instruction: str) -> np.ndarray:
    banner = draw_banner(image, title, instruction, (40, 210, 120))
    width = 640
    height = round(banner.shape[0] * width / banner.shape[1])
    return cv2.resize(banner, (width, height), interpolation=cv2.INTER_AREA)


def extract(args: argparse.Namespace) -> dict[str, object]:
    source, output = validate_args(args)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    expected_type = "sensor_msgs/msg/CompressedImage"
    if topic_types.get(args.topic) != expected_type:
        raise ValueError(
            f"expected {expected_type} on {args.topic}, found {topic_types.get(args.topic)!r}"
        )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    reference_jpeg: bytes | None = None
    reference_image: np.ndarray | None = None
    reference_thumb: np.ndarray | None = None
    reference_time_ns = 0
    reference_index = 0
    changed_jpeg: bytes | None = None
    changed_image: np.ndarray | None = None
    changed_time_ns = 0
    changed_index = 0
    best_score = -1.0
    valid_frames = 0
    scored_candidates = 0

    while reader.has_next():
        topic, serialized, log_time_ns = reader.read_next()
        if topic != args.topic:
            continue
        decoded = decode_frame(deserialize_message(serialized, CompressedImage))
        if decoded is None:
            continue
        jpeg, image = decoded
        current_index = valid_frames
        valid_frames += 1

        if reference_image is None:
            reference_jpeg = jpeg
            reference_image = image
            reference_thumb = thumbnail(image)
            reference_time_ns = log_time_ns
            reference_index = current_index
            continue

        elapsed = (log_time_ns - reference_time_ns) / 1_000_000_000
        if elapsed < args.minimum_separation or current_index % args.sample_stride:
            continue
        scored_candidates += 1
        score = difference_score(reference_thumb, thumbnail(image))
        if score > best_score:
            best_score = score
            changed_jpeg = jpeg
            changed_image = image
            changed_time_ns = log_time_ns
            changed_index = current_index

    if reference_image is None or reference_jpeg is None:
        raise RuntimeError(f"no decodable JPEG frames found on {args.topic}")
    if changed_image is None or changed_jpeg is None:
        raise RuntimeError("no changed-frame candidate passed the selection constraints")

    elapsed_seconds = (changed_time_ns - reference_time_ns) / 1_000_000_000
    overlay = compose_overlay(changed_image, reference_image, args.alpha)
    reference_card = comparison_card(
        reference_image,
        "REFERENCE FRAME",
        "Early frame used as the magenta/yellow ghost",
    )
    changed_card = comparison_card(
        changed_image,
        "CHANGED LIVE FRAME",
        f"Most different sampled view, {elapsed_seconds:.2f} s later",
    )
    overlay_card = comparison_card(
        overlay,
        "RESET OVERLAY",
        "Mismatch appears as separated live and ghost features",
    )
    comparison = cv2.hconcat((reference_card, changed_card, overlay_card))

    output.mkdir(parents=True, exist_ok=False)
    reference_path = output / "reference_frame.jpg"
    changed_path = output / "changed_frame.jpg"
    overlay_path = output / "overlay.png"
    comparison_path = output / "overlay_comparison.png"
    reference_path.write_bytes(reference_jpeg)
    changed_path.write_bytes(changed_jpeg)
    write_png(overlay_path, overlay)
    write_png(comparison_path, comparison)

    artifact_paths = (reference_path, changed_path, overlay_path, comparison_path)
    manifest: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_episode": source.parent.name,
        "source_mcap": str(source),
        "source_mcap_size_bytes": source.stat().st_size,
        "source_mcap_sha256": file_sha256(source),
        "source_topic": args.topic,
        "selection": {
            "method": "maximum thumbnail pixel-plus-edge difference from first valid frame",
            "sample_stride": args.sample_stride,
            "minimum_separation_seconds": args.minimum_separation,
            "valid_topic_frames": valid_frames,
            "scored_candidates": scored_candidates,
            "reference_frame_index": reference_index,
            "reference_log_time_ns": reference_time_ns,
            "changed_frame_index": changed_index,
            "changed_log_time_ns": changed_time_ns,
            "elapsed_seconds": elapsed_seconds,
            "difference_score": best_score,
        },
        "overlay": {
            "alpha": args.alpha,
            "implementation": "head_reset_overlay.compose_overlay",
        },
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_paths
        },
    }
    manifest_path = output / "manifest.json"
    temporary_manifest = output / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = extract(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    selection = manifest["selection"]
    print(
        f"Created overlay example from {manifest['source_episode']}\n"
        f"Elapsed: {selection['elapsed_seconds']:.3f} s\n"
        f"Difference score: {selection['difference_score']:.3f}\n"
        f"Output: {args.output_dir.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
