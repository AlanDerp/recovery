#!/usr/bin/env python3
"""Capture one head-left image and overlay fixed cup table coordinates."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-scene", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    return parser.parse_args()


def capture_head_left(timeout_s: float) -> tuple[Image.Image, int]:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init(args=None)
    node = rclpy.create_node("capx_fixed_cup_overlay_capture")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    received: dict[str, object] = {}

    def callback(message: CompressedImage) -> None:
        received["image"] = Image.open(io.BytesIO(bytes(message.data))).convert("RGB")
        received["stamp_ns"] = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    subscription = node.create_subscription(
        CompressedImage,
        "/hdas/camera_head/left_raw/image_raw_color/compressed",
        callback,
        qos_profile_sensor_data,
    )
    try:
        deadline = time.monotonic() + float(timeout_s)
        while "image" not in received and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if "image" not in received:
            raise TimeoutError("timed out waiting for head-left image")
        return received["image"], int(received["stamp_ns"])
    finally:
        node.destroy_subscription(subscription)
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if owns_context:
            rclpy.shutdown()


def main() -> None:
    args = parse_args()
    scene = json.loads(args.fixed_scene.read_text())
    calibration = json.loads(args.calibration.read_text())
    image, stamp_ns = capture_head_left(args.timeout_s)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.raw_output)

    homography = np.asarray(
        calibration["image_to_base_xy_homography"], dtype=float
    ).reshape(3, 3)
    base_to_image = np.linalg.inv(homography)
    calibration_width, calibration_height = calibration["image_size"]
    scale_u = image.width / float(calibration_width)
    scale_v = image.height / float(calibration_height)
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 24)
    title_font = ImageFont.truetype(font_path, 28)
    projected = {}
    colors = {
        "orange cup": (249, 115, 22),
        "purple cup": (168, 85, 247),
        "green cup": (34, 197, 94),
    }
    for name in ("orange cup", "purple cup", "green cup"):
        rim = np.asarray(
            scene["cups"][name]["rim_center_base_m"], dtype=float
        )
        pixel_h = base_to_image @ np.array([rim[0], rim[1], 1.0])
        u = float(pixel_h[0] / pixel_h[2]) * scale_u
        v = float(pixel_h[1] / pixel_h[2]) * scale_v
        projected[name] = {
            "pixel_uv": [u, v],
            "fixed_rim_center_base_m": rim.tolist(),
        }
        color = colors[name]
        radius = max(24, int(round(image.width / 50)))
        draw.ellipse(
            (u - radius, v - radius, u + radius, v + radius),
            outline=color,
            width=max(6, radius // 5),
        )
        draw.line((u - radius, v, u + radius, v), fill=color, width=3)
        draw.line((u, v - radius, u, v + radius), fill=color, width=3)
        label = name.replace(" cup", "").upper()
        box = draw.textbbox((u - radius, v - radius - 34), label, font=font)
        draw.rectangle(
            (box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4),
            fill=(15, 23, 42),
            outline=color,
            width=2,
        )
        draw.text((u - radius, v - radius - 34), label, fill=color, font=font)

    title = "FIXED CUP COORDINATES (table-plane projection; not live detection)"
    title_box = draw.textbbox((18, 18), title, font=title_font)
    draw.rectangle(
        (10, 10, title_box[2] + 8, title_box[3] + 8),
        fill=(15, 23, 42),
        outline=(226, 232, 240),
        width=2,
    )
    draw.text((18, 18), title, fill=(226, 232, 240), font=title_font)
    legend_lines = []
    for name in ("orange cup", "purple cup", "green cup"):
        rim = projected[name]["fixed_rim_center_base_m"]
        legend_lines.append(
            f"{name.upper()}: ({rim[0]:.3f}, {rim[1]:.3f}, {rim[2]:.3f}) m"
        )
    legend = "\n".join(legend_lines)
    legend_box = draw.multiline_textbbox((18, 68), legend, font=font, spacing=6)
    draw.rectangle(
        (10, 60, legend_box[2] + 8, legend_box[3] + 8),
        fill=(15, 23, 42),
        outline=(226, 232, 240),
        width=2,
    )
    draw.multiline_text(
        (18, 68),
        legend,
        fill=(226, 232, 240),
        font=font,
        spacing=6,
    )
    image.save(args.output)
    metadata = {
        "schema": "capx.r1lite.fixed_cup_head_overlay.v1",
        "captured_stamp_ns": stamp_ns,
        "image_size": [image.width, image.height],
        "calibration_plane_z_m": calibration["plane_z"],
        "fixed_table_height_m": scene.get("table_height_m"),
        "projection": "inverse_image_to_base_xy_homography",
        "projected_cups": projected,
        "raw_image": str(args.raw_output.resolve()),
        "annotated_image": str(args.output.resolve()),
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
