#!/usr/bin/env python3
"""Validate rectified head-stereo extrinsics against the physical tabletop."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

from cv_bridge import CvBridge
import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CALIBRATION = (
    REPO_ROOT / "outputs/r1lite/head_stereo_rectified_plane_calibration.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/r1lite/head_stereo_table_validation.json"


class DepthSample(Node):
    def __init__(self, depth_topic: str, info_topic: str, rgb_topic: str) -> None:
        super().__init__("r1lite_head_stereo_table_validation")
        self.bridge = CvBridge()
        self.depth_samples: list[np.ndarray] = []
        self.rgb: np.ndarray | None = None
        self.intrinsics: np.ndarray | None = None
        self.image_size: tuple[int, int] | None = None
        self.create_subscription(
            Image, depth_topic, self._on_depth, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, info_topic, self._on_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, rgb_topic, self._on_rgb, qos_profile_sensor_data
        )

    def _on_depth(self, message: Image) -> None:
        depth = np.asarray(
            self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"),
            dtype=np.float64,
        )
        if message.encoding in {"16UC1", "mono16"}:
            depth *= 0.001
        self.depth_samples.append(depth)
        if len(self.depth_samples) > 30:
            self.depth_samples.pop(0)

    def _on_rgb(self, message: Image) -> None:
        bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        self.rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)

    def _on_info(self, message: CameraInfo) -> None:
        self.intrinsics = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        self.image_size = (int(message.width), int(message.height))


def fit_plane_ransac(
    points: np.ndarray,
    *,
    threshold_m: float,
    iterations: int = 1200,
    maximum_candidate_tilt_deg: float = 25.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    if len(points) < 200:
        raise ValueError("at least 200 table candidates are required")
    rng = np.random.default_rng(20260830)
    best = np.zeros(len(points), dtype=bool)
    minimum_normal_z = float(
        np.cos(np.radians(float(maximum_candidate_tilt_deg)))
    )
    for _ in range(int(iterations)):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        if abs(float(normal[2])) < minimum_normal_z:
            continue
        if normal[2] < 0.0:
            normal *= -1.0
        offset = -float(np.dot(normal, sample[0]))
        inliers = np.abs(points @ normal + offset) <= float(threshold_m)
        if int(np.count_nonzero(inliers)) > int(np.count_nonzero(best)):
            best = inliers
    if int(np.count_nonzero(best)) < 200:
        raise RuntimeError("table RANSAC found fewer than 200 inliers")
    inlier_points = points[best]
    center = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal *= -1.0
    offset = -float(np.dot(normal, center))
    refined = np.abs(points @ normal + offset) <= float(threshold_m)
    return normal, offset, refined


def segment_table(rgb: np.ndarray, service_url: str) -> tuple[np.ndarray, float]:
    ok, encoded = cv2.imencode(
        ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    )
    if not ok:
        raise RuntimeError("failed to encode table image")
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        f"{service_url.rstrip('/')}/segment",
        json={
            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "text_prompt": "flat tabletop surface",
        },
        timeout=60.0,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise RuntimeError("SAM3 returned no tabletop mask")
    best = max(results, key=lambda item: float(item.get("score", 0.0)))
    score = float(best.get("score", 0.0))
    if score < 0.80:
        raise RuntimeError(f"SAM3 tabletop score is too low: {score:.3f}")
    height, width = (int(value) for value in best["shape"])
    mask = np.frombuffer(
        base64.b64decode(best["mask_base64"]), dtype=np.uint8
    ).reshape(height, width).astype(bool)
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask, score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--depth-topic",
        default="/capx/head_stereo/aligned_depth_to_left/image_raw",
    )
    parser.add_argument(
        "--info-topic",
        default="/capx/head_stereo/aligned_depth_to_left/camera_info",
    )
    parser.add_argument(
        "--rgb-topic", default="/capx/head_stereo/left/image_rect"
    )
    parser.add_argument("--sam3-service-url", default="http://127.0.0.1:8114")
    parser.add_argument("--sample-count", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.012)
    parser.add_argument("--maximum-tilt-deg", type=float, default=1.0)
    parser.add_argument("--maximum-rmse-m", type=float, default=0.012)
    parser.add_argument("--maximum-height-span-m", type=float, default=0.010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("image_domain") != "head_left_stereo_rectified":
        raise SystemExit("calibration is not in the rectified head-left domain")
    base_from_camera = np.asarray(
        calibration["base_from_rectified_left_camera"], dtype=np.float64
    ).reshape(4, 4)
    plane_z = float(calibration["plane_z"])

    rclpy.init()
    node = DepthSample(args.depth_topic, args.info_topic, args.rgb_topic)
    deadline = time.monotonic() + float(args.timeout)
    try:
        while (
            (
                len(node.depth_samples) < int(args.sample_count)
                or node.intrinsics is None
                or node.rgb is None
            )
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.2)
        if (
            len(node.depth_samples) < int(args.sample_count)
            or node.intrinsics is None
            or node.image_size is None
            or node.rgb is None
        ):
            raise TimeoutError(
                "timed out waiting for rectified RGB, depth samples and CameraInfo"
            )
        depth_stack = np.stack(node.depth_samples[-int(args.sample_count) :])
        valid_depth = np.isfinite(depth_stack) & (depth_stack > 0.0)
        depth_stack = np.where(valid_depth, depth_stack, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            depth = np.nanmedian(depth_stack, axis=0)
        rgb = node.rgb.copy()
        intrinsics = node.intrinsics.copy()
        image_size = node.image_size
    finally:
        node.destroy_node()
        rclpy.shutdown()

    expected_size = tuple(int(value) for value in calibration["image_size"])
    if image_size != expected_size or depth.shape != (
        expected_size[1],
        expected_size[0],
    ):
        raise RuntimeError(
            f"depth domain mismatch: info={image_size}, depth={depth.shape}, "
            f"calibration={expected_size}"
        )
    expected_k = np.asarray(
        calibration["rectified_intrinsics"], dtype=np.float64
    ).reshape(3, 3)
    if not np.allclose(intrinsics, expected_k, rtol=0.0, atol=1e-3):
        raise RuntimeError("live rectified intrinsics differ from calibration")

    table_mask, sam3_score = segment_table(rgb, args.sam3_service_url)
    rows, columns = np.indices(depth.shape)
    stride = 2
    z = depth[::stride, ::stride].reshape(-1)
    u = columns[::stride, ::stride].reshape(-1)
    v = rows[::stride, ::stride].reshape(-1)
    inside_coverage = table_mask[::stride, ::stride].reshape(-1)
    valid = (
        inside_coverage & np.isfinite(z) & (z >= 0.2) & (z <= 3.0)
    )
    z, u, v = z[valid], u[valid], v[valid]
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    camera_xyz = np.column_stack([x, y, z])
    camera_points = np.column_stack([camera_xyz, np.ones_like(z)])
    base_points = (base_from_camera @ camera_points.T).T[:, :3]
    candidates = base_points[
        np.isfinite(base_points).all(axis=1)
        & (base_points[:, 2] >= plane_z - 0.35)
        & (base_points[:, 2] <= plane_z + 0.35)
    ]
    normal_camera, camera_offset, camera_inliers = fit_plane_ransac(
        camera_xyz,
        threshold_m=args.ransac_threshold_m,
        maximum_candidate_tilt_deg=89.0,
    )
    camera_center = np.median(camera_xyz[camera_inliers], axis=0)
    normal = base_from_camera[:3, :3] @ normal_camera
    if normal[2] < 0.0:
        normal *= -1.0
    transformed_center = (
        base_from_camera[:3, :3] @ camera_center
        + base_from_camera[:3, 3]
    )
    offset = -float(np.dot(normal, transformed_center))
    inliers = (
        np.abs(candidates @ normal + offset) <= float(args.ransac_threshold_m)
    )
    inlier_points = candidates[inliers]
    residuals = inlier_points @ normal + offset
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    tilt_deg = float(
        np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
    )
    xy_min = np.percentile(inlier_points[:, :2], 2.0, axis=0)
    xy_max = np.percentile(inlier_points[:, :2], 98.0, axis=0)
    corners = np.asarray(
        [
            [xy_min[0], xy_min[1]],
            [xy_min[0], xy_max[1]],
            [xy_max[0], xy_min[1]],
            [xy_max[0], xy_max[1]],
        ]
    )
    corner_z = -(corners @ normal[:2] + offset) / normal[2]
    height_span = float(np.max(corner_z) - np.min(corner_z))
    passed = (
        tilt_deg <= float(args.maximum_tilt_deg)
        and rmse <= float(args.maximum_rmse_m)
        and height_span <= float(args.maximum_height_span_m)
    )
    audit = {
        "schema": "capx.r1lite.head_stereo_table_validation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "robot_motion_commands_sent": False,
        "calibration_path": str(args.calibration.resolve()),
        "candidate_count": int(len(candidates)),
        "inlier_count": int(len(inlier_points)),
        "depth_sample_count": int(args.sample_count),
        "sam3_table_score": sam3_score,
        "normal_camera": normal_camera.tolist(),
        "plane_offset_camera": camera_offset,
        "plane_center_camera_m": camera_center.tolist(),
        "normal_base": normal.tolist(),
        "plane_offset": offset,
        "tilt_deg": tilt_deg,
        "rmse_m": rmse,
        "workspace_xy_min_m": xy_min.tolist(),
        "workspace_xy_max_m": xy_max.tolist(),
        "workspace_corner_z_m": corner_z.tolist(),
        "workspace_height_span_m": height_span,
        "thresholds": {
            "maximum_tilt_deg": float(args.maximum_tilt_deg),
            "maximum_rmse_m": float(args.maximum_rmse_m),
            "maximum_height_span_m": float(args.maximum_height_span_m),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
