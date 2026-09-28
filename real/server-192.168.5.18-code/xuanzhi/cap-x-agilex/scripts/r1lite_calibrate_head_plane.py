#!/usr/bin/env python3
"""Create a head-camera tabletop homography for R1 Lite.

The calibration maps head-left image pixels ``[u, v]`` to base-frame plane
coordinates ``[x, y]``.  It is intended for flat work surfaces such as the case
top or tabletop visible in the head camera.

Example points JSON:

{
  "plane_z": 0.85,
  "points": [
    {"pixel": [420, 1100], "base_xy": [0.35, -0.25]},
    {"pixel": [1520, 1100], "base_xy": [0.35, 0.25]},
    {"pixel": [1550, 1450], "base_xy": [0.65, 0.25]},
    {"pixel": [390, 1450], "base_xy": [0.65, -0.25]}
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_OUTPUT = "outputs/r1lite/head_plane_calibration.json"


def _compute_homography(src_uv: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Compute image-to-plane homography with normalized DLT."""
    if src_uv.shape[0] < 4:
        raise ValueError("At least four point correspondences are required")
    rows = []
    for (u, v), (x, y) in zip(src_uv, dst_xy, strict=True):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("Degenerate homography; check calibration points")
    return h / h[2, 2]


def _load_points(path: str) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    pts = data.get("points", [])
    if len(pts) < 4:
        raise ValueError("points JSON must contain at least four points")
    src = []
    dst = []
    for p in pts:
        src.append(p["pixel"])
        dst.append(p["base_xy"])
    plane_z = float(data.get("plane_z", 0.0))
    return (
        np.asarray(src, dtype=np.float64).reshape(-1, 2),
        np.asarray(dst, dtype=np.float64).reshape(-1, 2),
        plane_z,
        data,
    )


def _save_template(image_path: str | None, output: str) -> None:
    template = {
        "plane_z": 0.0,
        "points": [
            {"pixel": [0, 0], "base_xy": [0.0, 0.0]},
            {"pixel": [100, 0], "base_xy": [0.0, 0.1]},
            {"pixel": [100, 100], "base_xy": [0.1, 0.1]},
            {"pixel": [0, 100], "base_xy": [0.1, 0.0]},
        ],
    }
    if image_path:
        img = Image.open(image_path)
        template["image_size"] = list(img.size)
        template["image_path"] = image_path
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, indent=2))
    print(f"Wrote calibration template to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-json", help="JSON file with pixel/base correspondences")
    parser.add_argument("--image", help="Optional image path used for template metadata")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write a point JSON template instead of computing calibration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        _save_template(args.image, args.output)
        return
    if not args.points_json:
        raise SystemExit("--points-json is required unless --write-template is used")

    src_uv, dst_xy, plane_z, raw = _load_points(args.points_json)
    h = _compute_homography(src_uv, dst_xy)
    calibration = {
        "type": "image_to_base_plane_homography",
        "source": "r1lite_head_left",
        "plane_z": plane_z,
        "image_to_base_xy_homography": h.tolist(),
        "num_points": int(src_uv.shape[0]),
        "points": raw.get("points", []),
    }
    if args.image:
        img = Image.open(args.image)
        calibration["image_size"] = list(img.size)
        calibration["image_path"] = args.image

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calibration, indent=2))
    print(f"Wrote head-plane calibration to {out}")
    print("H image->base_xy:")
    print(h)


if __name__ == "__main__":
    main()
