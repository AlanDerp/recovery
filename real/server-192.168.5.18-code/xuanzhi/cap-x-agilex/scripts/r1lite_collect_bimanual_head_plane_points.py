#!/usr/bin/env python3
"""Collect and solve a dual-arm R1 Lite head-camera table calibration.

The script never sends robot motion commands.  An operator manually moves the
requested arm so its fingertip center touches each clicked table point.  Left-
only and right-only points expand coverage; overlap points are touched by both
arms and are used to estimate a zero-mean left/right kinematic/TCP bias before
fitting one image-to-base homography.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_IMAGE = REPO_ROOT / "outputs/r1lite/bimanual_head_plane_rgb.png"
DEFAULT_POINTS = REPO_ROOT / "outputs/r1lite/bimanual_head_plane_points.json"
DEFAULT_CALIBRATION = REPO_ROOT / "outputs/r1lite/head_plane_calibration.json"
DEFAULT_BOARD_POINTS = REPO_ROOT / "outputs/r1lite/bimanual_chessboard_points.json"
DEFAULT_BOARD_PREVIEW = REPO_ROOT / "outputs/r1lite/bimanual_chessboard_preview.png"

HEAD_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
POSE_TOPICS = {
    "left": "/motion_control/pose_ee_arm_left",
    "right": "/motion_control/pose_ee_arm_right",
}
TF_CANDIDATES = {
    "left": ("left_gripper_link",),
    "right": ("right_gripper_link",),
}
TABLE_Z_SANITY_RANGE_M = (0.45, 1.35)
BOARD_SQUARE_ROWS = 14
BOARD_SQUARE_COLUMNS = 18
BOARD_INNER_ROWS = BOARD_SQUARE_ROWS - 1
BOARD_INNER_COLUMNS = BOARD_SQUARE_COLUMNS - 1
BOARD_SQUARE_SIZE_M = 0.020
BOARD_PATTERN_SIZE = (BOARD_INNER_COLUMNS, BOARD_INNER_ROWS)
OPEN_RIGHT_FINGERTIP_LINK = "finger_link2"
OPEN_RIGHT_FINGERTIP_CENTER_GRIPPER_M = np.array(
    [0.07835500, -0.03325007, -0.004453985],
    dtype=np.float64,
)


def contact_position_from_gripper_pose(
    origin_base_m: Sequence[float],
    quaternion_xyzw: Sequence[float],
    contact_offset_gripper_m: Sequence[float],
) -> np.ndarray:
    """Transform one URDF gripper-local contact point into base_link."""
    origin = np.asarray(origin_base_m, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    offset = np.asarray(contact_offset_gripper_m, dtype=np.float64).reshape(3)
    if not np.isfinite(origin).all() or not np.isfinite(quaternion).all():
        raise ValueError("gripper pose must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("gripper quaternion is invalid")
    return origin + Rotation.from_quat(quaternion / norm).apply(offset)


def _setup_ros_env() -> None:
    os.environ.setdefault("ROS_DOMAIN_ID", "1")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    os.environ.setdefault(
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "/opt/galaxea/find_server/super_client_configuration_file.xml",
    )


def _ensure_display() -> None:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    if Path("/tmp/.X11-unix/X0").exists():
        os.environ["DISPLAY"] = ":0"
        return
    raise SystemExit(
        "No GUI display is available. Run on the robot desktop or set DISPLAY=:0."
    )


def _wait_for_head_rgb(timeout_s: float) -> np.ndarray:
    import io

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    class Capture(Node):
        def __init__(self) -> None:
            super().__init__("capx_bimanual_table_calibration_camera")
            self.rgb: np.ndarray | None = None
            self.create_subscription(
                CompressedImage,
                HEAD_TOPIC,
                self._on_image,
                qos_profile_sensor_data,
            )

        def _on_image(self, msg: CompressedImage) -> None:
            image = Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
            self.rgb = np.asarray(image, dtype=np.uint8)

    _setup_ros_env()
    if not rclpy.ok():
        rclpy.init()
    node = Capture()
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.rgb is not None:
                return node.rgb.copy()
        raise TimeoutError(f"timed out waiting for {HEAD_TOPIC}")
    finally:
        node.destroy_node()


def _sample_gripper_link_pose(
    side: str,
    *,
    timeout_s: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, str, float, float]:
    """Return median gripper-link pose and position/orientation spread."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    if side not in POSE_TOPICS:
        raise ValueError(f"invalid arm side: {side}")

    class PoseReader(Node):
        def __init__(self) -> None:
            super().__init__(f"capx_bimanual_table_pose_{side}")
            self.positions: list[np.ndarray] = []
            self.quaternions: list[np.ndarray] = []
            self.create_subscription(PoseStamped, POSE_TOPICS[side], self._on_pose, 20)

        def _on_pose(self, msg: PoseStamped) -> None:
            if str(msg.header.frame_id).strip() != "base_link":
                return
            p = msg.pose.position
            q = msg.pose.orientation
            self.positions.append(np.array([p.x, p.y, p.z], dtype=np.float64))
            self.quaternions.append(
                np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            )

    _setup_ros_env()
    if not rclpy.ok():
        rclpy.init()
    node = PoseReader()
    deadline = time.time() + timeout_s
    source = POSE_TOPICS[side]
    try:
        while time.time() < deadline and len(node.positions) < sample_count:
            rclpy.spin_once(node, timeout_sec=0.1)
        positions = node.positions[-sample_count:]
        quaternions = node.quaternions[-sample_count:]
        if not positions:
            # Some installations expose only TF.  tool_z and gripper_link have
            # the same origin, so either is valid for XY contact calibration.
            try:
                from tf2_ros import Buffer, TransformListener
            except ImportError as exc:
                raise TimeoutError(f"no pose samples on {POSE_TOPICS[side]}") from exc
            buffer = Buffer()
            listener = TransformListener(buffer, node)
            for child in TF_CANDIDATES[side]:
                tf_deadline = time.time() + min(5.0, timeout_s)
                while time.time() < tf_deadline and len(positions) < sample_count:
                    rclpy.spin_once(node, timeout_sec=0.05)
                    try:
                        transform = buffer.lookup_transform(
                            "base_link", child, rclpy.time.Time()
                        )
                    except Exception:
                        continue
                    p = transform.transform.translation
                    q = transform.transform.rotation
                    positions.append(np.array([p.x, p.y, p.z], dtype=np.float64))
                    quaternions.append(
                        np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
                    )
                if positions:
                    source = f"tf:base_link->{child}"
                    break
        if not positions:
            raise TimeoutError(
                f"no physical {side} EE pose on {POSE_TOPICS[side]} or TF"
            )
        values = np.vstack(positions)
        median = np.median(values, axis=0)
        spread = float(np.max(np.linalg.norm(values - median, axis=1)))
        quaternion_values = np.vstack(quaternions)
        reference = quaternion_values[0]
        quaternion_values = np.asarray(
            [
                -value if float(np.dot(value, reference)) < 0.0 else value
                for value in quaternion_values
            ]
        )
        mean_quaternion = np.mean(quaternion_values, axis=0)
        mean_quaternion /= np.linalg.norm(mean_quaternion)
        orientation_spread = float(
            np.max(
                (
                    Rotation.from_quat(quaternion_values)
                    * Rotation.from_quat(mean_quaternion).inv()
                ).magnitude()
            )
        )
        return median, mean_quaternion, source, spread, orientation_spread
    finally:
        node.destroy_node()


def _pick_pixels(
    image_path: Path,
    *,
    left_only: int,
    right_only: int,
    overlap: int,
) -> dict[str, list[list[int]]]:
    _ensure_display()
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    groups = [
        ("left_only", left_only, "lime"),
        ("right_only", right_only, "cyan"),
        ("overlap", overlap, "yellow"),
    ]
    sequence = [
        (group, index, color)
        for group, count, color in groups
        for index in range(1, count + 1)
    ]
    image = np.asarray(Image.open(image_path).convert("RGB"))
    selected: dict[str, list[list[int]]] = {name: [] for name, _, _ in groups}
    cursor = 0

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(image)

    def update_title() -> None:
        if cursor >= len(sequence):
            ax.set_title("采点完成")
            return
        group, index, _ = sequence[cursor]
        descriptions = {
            "left_only": "点击左臂可达区域点",
            "right_only": "点击右臂可达区域点",
            "overlap": "点击左右臂都能触碰的重叠点",
        }
        ax.set_title(f"{descriptions[group]}：{index}")

    def onclick(event: Any) -> None:
        nonlocal cursor
        if cursor >= len(sequence) or event.xdata is None or event.ydata is None:
            return
        group, index, color = sequence[cursor]
        pixel = [int(round(event.xdata)), int(round(event.ydata))]
        selected[group].append(pixel)
        ax.plot(pixel[0], pixel[1], "o", color=color, markersize=7)
        ax.text(
            pixel[0] + 8,
            pixel[1] + 8,
            f"{group[0].upper()}{index}",
            color=color,
        )
        print(f"[pick] {group}[{index}]={pixel}")
        cursor += 1
        update_title()
        fig.canvas.draw_idle()
        if cursor >= len(sequence):
            plt.close(fig)

    update_title()
    connection = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(connection)
    if cursor != len(sequence):
        raise SystemExit("point picking was cancelled before all points were selected")
    return selected


def detect_chessboard(
    image: np.ndarray,
    *,
    pattern_size: tuple[int, int] = BOARD_PATTERN_SIZE,
    square_size_m: float = BOARD_SQUARE_SIZE_M,
) -> list[dict[str, Any]]:
    """Detect all ordered inner corners of the 14x18-square board."""
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("chessboard image must be RGB")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    expected = int(pattern_size[0] * pattern_size[1])
    if not found or corners is None or len(corners) != expected:
        raise RuntimeError(
            f"expected {pattern_size[0]}x{pattern_size[1]}={expected} inner "
            "corners; chessboard detection failed"
        )
    values = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    result = []
    for row in range(pattern_size[1]):
        for column in range(pattern_size[0]):
            index = row * pattern_size[0] + column
            result.append(
                {
                    "corner_id": f"r{row:02d}_c{column:02d}",
                    "row": row,
                    "column": column,
                    "pixel": values[index].tolist(),
                    "board_xy_m": [
                        float(column * square_size_m),
                        float(row * square_size_m),
                    ],
                }
            )
    return result


def _farthest_pixel_subset(
    corners: Sequence[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0 or len(corners) < count:
        raise ValueError("invalid chessboard touch-point subset size")
    points = np.asarray([item["pixel"] for item in corners], dtype=np.float64)
    scale = np.maximum(np.ptp(points, axis=0), 1.0)
    normalized = (points - np.mean(points, axis=0)) / scale
    selected = [int(np.argmax(np.linalg.norm(normalized, axis=1)))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(
                normalized[:, None, :] - normalized[selected][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return [dict(corners[index]) for index in selected]


def select_chessboard_touch_points(
    corners: Sequence[dict[str, Any]],
    *,
    left_only_count: int = 5,
    right_only_count: int = 5,
    overlap_count: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    """Select spatially spread left/right/overlap corners by image horizontal position."""
    if len(corners) != BOARD_INNER_ROWS * BOARD_INNER_COLUMNS:
        raise ValueError("complete 13x17 inner-corner detection is required")
    u = np.asarray([item["pixel"][0] for item in corners], dtype=np.float64)
    left_limit, overlap_low, overlap_high, right_limit = np.quantile(
        u, [0.35, 0.40, 0.60, 0.65]
    )
    candidates = {
        "left_only": [item for item in corners if item["pixel"][0] <= left_limit],
        "right_only": [item for item in corners if item["pixel"][0] >= right_limit],
        "overlap": [
            item
            for item in corners
            if overlap_low <= item["pixel"][0] <= overlap_high
        ],
    }
    selected = {
        "left_only": _farthest_pixel_subset(
            candidates["left_only"], count=left_only_count
        ),
        "right_only": _farthest_pixel_subset(
            candidates["right_only"], count=right_only_count
        ),
        "overlap": _farthest_pixel_subset(
            candidates["overlap"], count=overlap_count
        ),
    }
    ids = [
        item["corner_id"]
        for group in selected.values()
        for item in group
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError("automatic chessboard touch-point groups overlap")
    return selected


def render_chessboard_preview(
    image: np.ndarray,
    corners: Sequence[dict[str, Any]],
    selected: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    canvas = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    for item in corners:
        u, v = [int(round(value)) for value in item["pixel"]]
        cv2.circle(canvas, (u, v), 2, (120, 120, 120), -1)
    colors = {
        "left_only": (0, 255, 0),
        "right_only": (255, 255, 0),
        "overlap": (0, 255, 255),
    }
    prefixes = {"left_only": "L", "right_only": "R", "overlap": "O"}
    for group, items in selected.items():
        for index, item in enumerate(items, start=1):
            u, v = [int(round(value)) for value in item["pixel"]]
            cv2.circle(canvas, (u, v), 9, colors[group], 2)
            cv2.putText(
                canvas,
                f"{prefixes[group]}{index}",
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colors[group],
                2,
                cv2.LINE_AA,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def _record_contact(
    *,
    side: str,
    group: str,
    point_id: str,
    pixel: Sequence[int],
    timeout_s: float,
    pose_samples: int,
    maximum_pose_spread_m: float,
    contact_offset_gripper_m: np.ndarray | None = None,
    contact_definition: str = "tool_origin",
    maximum_tool_z_tilt_deg: float | None = None,
) -> dict[str, Any]:
    print(
        f"\n[{point_id}] 请手动移动{side}臂，使指尖中心轻触 pixel={list(pixel)} "
        "对应的桌面点。"
    )
    answer = input("到位后按 Enter 记录；输入 q 退出：").strip().lower()
    if answer == "q":
        raise SystemExit("collection cancelled")
    position, quaternion, source, spread, orientation_spread = (
        _sample_gripper_link_pose(
            side,
            timeout_s=timeout_s,
            sample_count=pose_samples,
        )
    )
    if spread > maximum_pose_spread_m:
        raise RuntimeError(
            f"{point_id} pose was not stationary: {spread*1000:.2f}mm > "
            f"{maximum_pose_spread_m*1000:.2f}mm"
        )
    contact_offset = (
        np.zeros(3, dtype=np.float64)
        if contact_offset_gripper_m is None
        else np.asarray(contact_offset_gripper_m, dtype=np.float64).reshape(3)
    )
    contact_position = contact_position_from_gripper_pose(
        position,
        quaternion,
        contact_offset,
    )
    tool_z_base = Rotation.from_quat(quaternion).apply([1.0, 0.0, 0.0])
    tool_z_tilt_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    float(np.dot(tool_z_base, np.array([0.0, 0.0, -1.0]))),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    if (
        maximum_tool_z_tilt_deg is not None
        and tool_z_tilt_deg > maximum_tool_z_tilt_deg
    ):
        raise RuntimeError(
            f"{point_id} tool_z tilt is {tool_z_tilt_deg:.2f}deg; "
            f"required <= {maximum_tool_z_tilt_deg:.2f}deg"
        )
    print(
        f"[record] {point_id} {side} contact_base_xyz={contact_position.tolist()} "
        f"spread={spread*1000:.2f}mm source={source}"
    )
    return {
        "point_id": point_id,
        "group": group,
        "side": side,
        "pixel": [int(pixel[0]), int(pixel[1])],
        "base_xyz": contact_position.tolist(),
        "gripper_link_origin_base_xyz": position.tolist(),
        "gripper_link_quaternion_xyzw": quaternion.tolist(),
        "contact_offset_gripper_m": contact_offset.tolist(),
        "contact_definition": contact_definition,
        "pose_source": source,
        "pose_spread_m": spread,
        "orientation_spread_rad": orientation_spread,
        "tool_z_base": tool_z_base.tolist(),
        "tool_z_tilt_from_vertical_down_deg": tool_z_tilt_deg,
    }


def _overlap_pairs(records: Sequence[dict[str, Any]]) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for record in records:
        if record["group"] != "overlap":
            continue
        grouped[record["point_id"]][record["side"]] = np.asarray(
            record["base_xyz"], dtype=np.float64
        )
    pairs = []
    for point_id, by_side in sorted(grouped.items()):
        if set(by_side) != {"left", "right"}:
            raise ValueError(f"overlap point {point_id} lacks one arm measurement")
        pairs.append((by_side["left"], by_side["right"]))
    if len(pairs) < 3:
        raise ValueError("at least three dual-arm overlap points are required")
    return pairs


def solve_bimanual_calibration(
    payload: dict[str, Any],
    *,
    ransac_threshold_m: float = 0.012,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Estimate zero-mean arm biases, then robustly fit one homography."""
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < 10:
        raise ValueError("insufficient dual-arm calibration records")
    pairs = _overlap_pairs(records)
    differences = np.vstack([left - right for left, right in pairs])
    median_difference = np.median(differences, axis=0)
    arm_bias = {
        "left": median_difference / 2.0,
        "right": -median_difference / 2.0,
    }

    pixels = []
    corrected_xy = []
    corrected_z = []
    sides = []
    corrected_records = []
    for record in records:
        side = str(record["side"])
        raw = np.asarray(record["base_xyz"], dtype=np.float64).reshape(3)
        corrected = raw - arm_bias[side]
        pixels.append(record["pixel"])
        corrected_xy.append(corrected[:2])
        corrected_z.append(float(corrected[2]))
        sides.append(side)
        corrected_records.append(
            {
                **record,
                "raw_base_xyz": raw.tolist(),
                "base_xyz": corrected.tolist(),
                "base_xy": corrected[:2].tolist(),
                "applied_arm_bias_xyz_m": arm_bias[side].tolist(),
            }
        )

    src = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(corrected_xy, dtype=np.float64).reshape(-1, 2)
    homography, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_m),
        maxIters=5000,
        confidence=0.999,
    )
    if homography is None or mask is None:
        raise ValueError("homography solve failed")
    inliers = mask.reshape(-1).astype(bool)
    if int(np.count_nonzero(inliers)) < 8:
        raise ValueError("fewer than eight calibration measurements survived RANSAC")
    homography, _ = cv2.findHomography(src[inliers], dst[inliers], method=0)
    homogeneous = np.column_stack([src, np.ones(len(src))])
    projected = (homography @ homogeneous.T).T
    predictions = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(predictions - dst, axis=1)

    per_arm = {}
    for side in ("left", "right"):
        indices = np.asarray([value == side for value in sides], dtype=bool)
        side_errors = errors[indices]
        per_arm[side] = {
            "count": int(np.count_nonzero(indices)),
            "inlier_count": int(np.count_nonzero(inliers & indices)),
            "rmse_m": float(np.sqrt(np.mean(np.square(side_errors)))),
            "maximum_error_m": float(np.max(side_errors)),
        }
    overlap_raw_distances = np.linalg.norm(differences, axis=1)
    corrected_overlap_distances = np.linalg.norm(
        differences - median_difference, axis=1
    )
    unique_pixels = np.unique(src, axis=0).astype(np.float32)
    hull = cv2.convexHull(unique_pixels).reshape(-1, 2)
    image_size = payload.get("image_size")
    image_area = (
        float(image_size[0] * image_size[1])
        if isinstance(image_size, list) and len(image_size) == 2
        else float("nan")
    )
    hull_area = float(cv2.contourArea(hull.astype(np.float32)))

    audit = {
        "schema": "capx.r1lite.bimanual_head_plane_audit.v1",
        "arm_bias_xyz_m": {
            side: value.tolist() for side, value in arm_bias.items()
        },
        "zero_mean_bias_constraint": True,
        "overlap_point_count": len(pairs),
        "overlap_raw_median_disagreement_m": float(
            np.median(overlap_raw_distances)
        ),
        "overlap_corrected_rmse_m": float(
            np.sqrt(np.mean(np.square(corrected_overlap_distances)))
        ),
        "ransac_threshold_m": float(ransac_threshold_m),
        "inlier_count": int(np.count_nonzero(inliers)),
        "measurement_count": len(records),
        "overall_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "maximum_error_m": float(np.max(errors)),
        "per_arm": per_arm,
        "coverage_hull_pixel_uv": hull.astype(float).tolist(),
        "coverage_hull_area_px2": hull_area,
        "coverage_fraction_of_image": hull_area / image_area,
        "measurement_errors_m": errors.tolist(),
        "inlier_mask": inliers.tolist(),
    }
    calibration = {
        "type": "image_to_base_plane_homography",
        "source": "r1lite_head_left_bimanual",
        "plane_z": float(np.mean(corrected_z)),
        "image_to_base_xy_homography": homography.tolist(),
        "num_points": len(records),
        "points": [
            {"pixel": item["pixel"], "base_xy": item["base_xy"]}
            for item in corrected_records
        ],
        "image_size": image_size,
        "image_path": payload.get("image_path"),
        "bimanual_calibration": {
            "schema": "capx.r1lite.bimanual_head_plane.v1",
            "left_tcp": "left_gripper_tool_z_nominal_urdf",
            "right_tcp": "right_gripper_tool_z_nominal_urdf",
            "arm_bias_xyz_m": audit["arm_bias_xyz_m"],
            "overlap_point_count": len(pairs),
            "coverage_hull_pixel_uv": audit["coverage_hull_pixel_uv"],
            "quality": {
                "overall_rmse_m": audit["overall_rmse_m"],
                "maximum_error_m": audit["maximum_error_m"],
                "per_arm": per_arm,
                "overlap_corrected_rmse_m": audit[
                    "overlap_corrected_rmse_m"
                ],
            },
        },
        "raw_records": corrected_records,
    }
    return calibration, audit


def solve_chessboard_calibration(
    payload: dict[str, Any],
    *,
    ransac_threshold_m: float = 0.006,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Register the metric chessboard to base, then use all 221 visual corners."""
    board = payload.get("board", {})
    if (
        board.get("square_rows") != BOARD_SQUARE_ROWS
        or board.get("square_columns") != BOARD_SQUARE_COLUMNS
        or not np.isclose(
            float(board.get("square_size_m", 0.0)),
            BOARD_SQUARE_SIZE_M,
            atol=1e-12,
        )
    ):
        raise ValueError("unexpected chessboard specification")
    records = payload.get("records")
    corners = payload.get("detected_corners")
    if not isinstance(records, list) or len(records) < 10:
        raise ValueError("insufficient chessboard touch records")
    if not isinstance(corners, list) or len(corners) != (
        BOARD_INNER_ROWS * BOARD_INNER_COLUMNS
    ):
        raise ValueError("all 221 detected inner corners are required")

    pairs = _overlap_pairs(records)
    differences = np.vstack([left - right for left, right in pairs])
    median_difference = np.median(differences, axis=0)
    arm_bias = {
        "left": median_difference / 2.0,
        "right": -median_difference / 2.0,
    }
    board_xy = []
    corrected_xyz = []
    sides = []
    corrected_records = []
    for record in records:
        side = str(record["side"])
        raw = np.asarray(record["base_xyz"], dtype=np.float64).reshape(3)
        corrected = raw - arm_bias[side]
        metric = np.asarray(record["board_xy_m"], dtype=np.float64).reshape(2)
        board_xy.append(metric)
        corrected_xyz.append(corrected)
        sides.append(side)
        corrected_records.append(
            {
                **record,
                "raw_base_xyz": raw.tolist(),
                "base_xyz": corrected.tolist(),
                "base_xy": corrected[:2].tolist(),
                "applied_arm_bias_xyz_m": arm_bias[side].tolist(),
            }
        )

    source = np.asarray(board_xy, dtype=np.float64)
    measured = np.asarray(corrected_xyz, dtype=np.float64)
    source_center = np.mean(source, axis=0)
    measured_center = np.mean(measured[:, :2], axis=0)
    centered_source = source - source_center
    centered_measured = measured[:, :2] - measured_center
    u, _, vt = np.linalg.svd(centered_source.T @ centered_measured)
    board_to_base_row_rotation = u @ vt
    board_to_base_translation = (
        measured_center - source_center @ board_to_base_row_rotation
    )
    predicted_touch_xy = (
        source @ board_to_base_row_rotation + board_to_base_translation
    )
    touch_errors = np.linalg.norm(predicted_touch_xy - measured[:, :2], axis=1)

    all_board_xy = np.asarray(
        [item["board_xy_m"] for item in corners], dtype=np.float64
    )
    all_pixels = np.asarray(
        [item["pixel"] for item in corners], dtype=np.float64
    )
    all_base_xy = (
        all_board_xy @ board_to_base_row_rotation
        + board_to_base_translation
    )
    homography, mask = cv2.findHomography(
        all_pixels,
        all_base_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_m),
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None or mask is None:
        raise ValueError("dense chessboard homography solve failed")
    inliers = mask.reshape(-1).astype(bool)
    if int(np.count_nonzero(inliers)) < int(0.85 * len(corners)):
        raise ValueError("fewer than 85% of chessboard corners survived RANSAC")
    homography, _ = cv2.findHomography(
        all_pixels[inliers], all_base_xy[inliers], method=0
    )
    homogeneous = np.column_stack([all_pixels, np.ones(len(all_pixels))])
    projected = (homography @ homogeneous.T).T
    dense_predictions = projected[:, :2] / projected[:, 2:3]
    dense_errors = np.linalg.norm(dense_predictions - all_base_xy, axis=1)

    per_arm = {}
    for side in ("left", "right"):
        indices = np.asarray([value == side for value in sides], dtype=bool)
        errors = touch_errors[indices]
        per_arm[side] = {
            "count": int(np.count_nonzero(indices)),
            "rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
            "maximum_error_m": float(np.max(errors)),
        }
    corrected_overlap = differences - median_difference
    hull = cv2.convexHull(all_pixels.astype(np.float32)).reshape(-1, 2)
    image_size = payload.get("image_size")
    image_area = float(image_size[0] * image_size[1])
    audit = {
        "schema": "capx.r1lite.bimanual_chessboard_audit.v1",
        "board_specification": {
            "square_rows": BOARD_SQUARE_ROWS,
            "square_columns": BOARD_SQUARE_COLUMNS,
            "inner_rows": BOARD_INNER_ROWS,
            "inner_columns": BOARD_INNER_COLUMNS,
            "square_size_m": BOARD_SQUARE_SIZE_M,
            "physical_size_m": [
                BOARD_SQUARE_COLUMNS * BOARD_SQUARE_SIZE_M,
                BOARD_SQUARE_ROWS * BOARD_SQUARE_SIZE_M,
            ],
        },
        "arm_bias_xyz_m": {
            side: value.tolist() for side, value in arm_bias.items()
        },
        "zero_mean_bias_constraint": True,
        "overlap_point_count": len(pairs),
        "overlap_raw_median_disagreement_m": float(
            np.median(np.linalg.norm(differences, axis=1))
        ),
        "overlap_corrected_rmse_m": float(
            np.sqrt(
                np.mean(np.square(np.linalg.norm(corrected_overlap, axis=1)))
            )
        ),
        "board_to_base_row_rotation_2x2": (
            board_to_base_row_rotation.tolist()
        ),
        "board_to_base_translation_xy_m": (
            board_to_base_translation.tolist()
        ),
        "board_coordinate_handedness": float(
            np.linalg.det(board_to_base_row_rotation)
        ),
        "overall_rmse_m": float(
            np.sqrt(np.mean(np.square(touch_errors)))
        ),
        "maximum_error_m": float(np.max(touch_errors)),
        "per_arm": per_arm,
        "dense_corner_count": len(corners),
        "dense_homography_inlier_count": int(np.count_nonzero(inliers)),
        "dense_homography_rmse_m": float(
            np.sqrt(np.mean(np.square(dense_errors)))
        ),
        "dense_homography_maximum_error_m": float(np.max(dense_errors)),
        "coverage_hull_pixel_uv": hull.astype(float).tolist(),
        "coverage_fraction_of_image": (
            float(cv2.contourArea(hull.astype(np.float32))) / image_area
        ),
        "touch_errors_m": touch_errors.tolist(),
        "dense_corner_errors_m": dense_errors.tolist(),
    }
    calibration = {
        "type": "image_to_base_plane_homography",
        "source": "r1lite_head_left_bimanual_chessboard",
        "plane_z": float(np.mean(measured[:, 2])),
        "image_to_base_xy_homography": homography.tolist(),
        "num_points": len(corners),
        "points": [
            {
                "pixel": item["pixel"],
                "base_xy": base_xy.tolist(),
                "board_xy_m": item["board_xy_m"],
                "corner_id": item["corner_id"],
            }
            for item, base_xy in zip(corners, all_base_xy, strict=True)
        ],
        "image_size": image_size,
        "image_path": payload.get("image_path"),
        "bimanual_calibration": {
            "schema": "capx.r1lite.bimanual_chessboard.v1",
            "left_tcp": "left_gripper_tool_z_nominal_urdf",
            "right_tcp": "right_gripper_tool_z_nominal_urdf",
            "contact_definition": (
                "fully_open_finger_link2_local_minus_y_front_center"
            ),
            "contact_offset_gripper_m": (
                OPEN_RIGHT_FINGERTIP_CENTER_GRIPPER_M.tolist()
            ),
            "arm_bias_xyz_m": audit["arm_bias_xyz_m"],
            "overlap_point_count": len(pairs),
            "coverage_hull_pixel_uv": audit["coverage_hull_pixel_uv"],
            "board_specification": audit["board_specification"],
            "board_to_base_row_rotation_2x2": (
                board_to_base_row_rotation.tolist()
            ),
            "board_to_base_translation_xy_m": (
                board_to_base_translation.tolist()
            ),
            "quality": {
                "overall_rmse_m": audit["overall_rmse_m"],
                "maximum_error_m": audit["maximum_error_m"],
                "per_arm": per_arm,
                "overlap_corrected_rmse_m": audit[
                    "overlap_corrected_rmse_m"
                ],
                "dense_homography_rmse_m": audit[
                    "dense_homography_rmse_m"
                ],
            },
        },
        "raw_records": corrected_records,
    }
    return calibration, audit


def _quality_gate(
    audit: dict[str, Any],
    *,
    maximum_rmse_m: float,
    maximum_error_m: float,
    maximum_overlap_rmse_m: float,
) -> None:
    failures = []
    if float(audit["overall_rmse_m"]) > maximum_rmse_m:
        failures.append("overall RMSE")
    if float(audit["maximum_error_m"]) > maximum_error_m:
        failures.append("maximum point error")
    if float(audit["overlap_corrected_rmse_m"]) > maximum_overlap_rmse_m:
        failures.append("corrected overlap RMSE")
    for side, metrics in audit["per_arm"].items():
        if float(metrics["rmse_m"]) > maximum_rmse_m:
            failures.append(f"{side} RMSE")
    if failures:
        raise RuntimeError("calibration quality gate failed: " + ", ".join(failures))


def cmd_capture(args: argparse.Namespace) -> None:
    rgb = _wait_for_head_rgb(args.timeout)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)
    print(f"saved {output} ({rgb.shape[1]}x{rgb.shape[0]})")


def cmd_collect(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    if not args.no_recapture:
        cmd_capture(argparse.Namespace(output=str(image_path), timeout=args.timeout))
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    pixels = _pick_pixels(
        image_path,
        left_only=args.left_only_points,
        right_only=args.right_only_points,
        overlap=args.overlap_points,
    )
    records = []
    for group, side in (("left_only", "left"), ("right_only", "right")):
        for index, pixel in enumerate(pixels[group], start=1):
            records.append(
                _record_contact(
                    side=side,
                    group=group,
                    point_id=f"{group}_{index:02d}",
                    pixel=pixel,
                    timeout_s=args.timeout,
                    pose_samples=args.pose_samples,
                    maximum_pose_spread_m=args.maximum_pose_spread_m,
                )
            )
    for index, pixel in enumerate(pixels["overlap"], start=1):
        point_id = f"overlap_{index:02d}"
        for side in ("left", "right"):
            records.append(
                _record_contact(
                    side=side,
                    group="overlap",
                    point_id=point_id,
                    pixel=pixel,
                    timeout_s=args.timeout,
                    pose_samples=args.pose_samples,
                    maximum_pose_spread_m=args.maximum_pose_spread_m,
                )
            )
    payload = {
        "schema": "capx.r1lite.bimanual_head_plane_points.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "robot_motion_commands_sent": False,
        "operator_moved_arms_manually": True,
        "image_path": str(image_path),
        "image_size": list(Image.open(image_path).size),
        "head_topic": HEAD_TOPIC,
        "tcp_contract": {
            "left": "left_gripper_tool_z_nominal_urdf",
            "right": "right_gripper_tool_z_nominal_urdf",
            "contact_definition": "fingertip_center_lightly_touching_table",
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {output} with {len(records)} measurements")


def _capture_detect_select_board(
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    image_path = Path(args.image)
    if not args.no_recapture:
        cmd_capture(argparse.Namespace(output=str(image_path), timeout=args.timeout))
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    corners = detect_chessboard(image)
    selected = select_chessboard_touch_points(
        corners,
        left_only_count=args.left_only_points,
        right_only_count=args.right_only_points,
        overlap_count=args.overlap_points,
    )
    render_chessboard_preview(image, corners, selected, Path(args.preview))
    print(
        f"[board] detected {len(corners)} inner corners; "
        f"preview={args.preview}"
    )
    return image, corners, selected


def cmd_board_detect(args: argparse.Namespace) -> None:
    _, corners, selected = _capture_detect_select_board(args)
    print(
        json.dumps(
            {
                "square_grid": [BOARD_SQUARE_COLUMNS, BOARD_SQUARE_ROWS],
                "inner_corners": [BOARD_INNER_COLUMNS, BOARD_INNER_ROWS],
                "square_size_m": BOARD_SQUARE_SIZE_M,
                "detected_corner_count": len(corners),
                "selected": {
                    group: [item["corner_id"] for item in items]
                    for group, items in selected.items()
                },
                "robot_motion_commands_sent": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_board_collect(args: argparse.Namespace) -> None:
    image, corners, selected = _capture_detect_select_board(args)
    if not args.no_preview_window:
        _ensure_display()
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(13, 9))
        plt.imshow(Image.open(args.preview).convert("RGB"))
        plt.title("确认自动选择的 L/R/O 触碰点；关闭窗口后开始采集")
        plt.show()

    records = []
    for group, side in (("left_only", "left"), ("right_only", "right")):
        for index, corner in enumerate(selected[group], start=1):
            record = _record_contact(
                side=side,
                group=group,
                point_id=f"{group}_{index:02d}",
                pixel=corner["pixel"],
                timeout_s=args.timeout,
                pose_samples=args.pose_samples,
                maximum_pose_spread_m=args.maximum_pose_spread_m,
                contact_offset_gripper_m=OPEN_RIGHT_FINGERTIP_CENTER_GRIPPER_M,
                contact_definition=(
                    "fully_open_finger_link2_local_minus_y_front_center"
                ),
                maximum_tool_z_tilt_deg=args.maximum_tool_z_tilt_deg,
            )
            record.update(
                {
                    "corner_id": corner["corner_id"],
                    "board_xy_m": corner["board_xy_m"],
                }
            )
            records.append(record)
    for index, corner in enumerate(selected["overlap"], start=1):
        point_id = f"overlap_{index:02d}"
        for side in ("left", "right"):
            record = _record_contact(
                side=side,
                group="overlap",
                point_id=point_id,
                pixel=corner["pixel"],
                timeout_s=args.timeout,
                pose_samples=args.pose_samples,
                maximum_pose_spread_m=args.maximum_pose_spread_m,
                contact_offset_gripper_m=OPEN_RIGHT_FINGERTIP_CENTER_GRIPPER_M,
                contact_definition=(
                    "fully_open_finger_link2_local_minus_y_front_center"
                ),
                maximum_tool_z_tilt_deg=args.maximum_tool_z_tilt_deg,
            )
            record.update(
                {
                    "corner_id": corner["corner_id"],
                    "board_xy_m": corner["board_xy_m"],
                }
            )
            records.append(record)
    payload = {
        "schema": "capx.r1lite.bimanual_chessboard_points.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "robot_motion_commands_sent": False,
        "operator_moved_arms_manually": True,
        "image_path": str(args.image),
        "preview_path": str(args.preview),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "head_topic": HEAD_TOPIC,
        "board": {
            "square_rows": BOARD_SQUARE_ROWS,
            "square_columns": BOARD_SQUARE_COLUMNS,
            "inner_rows": BOARD_INNER_ROWS,
            "inner_columns": BOARD_INNER_COLUMNS,
            "square_size_m": BOARD_SQUARE_SIZE_M,
            "physical_width_m": BOARD_SQUARE_COLUMNS * BOARD_SQUARE_SIZE_M,
            "physical_height_m": BOARD_SQUARE_ROWS * BOARD_SQUARE_SIZE_M,
        },
        "tcp_contract": {
            "left": "left_gripper_tool_z_nominal_urdf",
            "right": "right_gripper_tool_z_nominal_urdf",
            "contact_definition": (
                "fully_open_finger_link2_local_minus_y_front_center"
            ),
            "contact_offset_gripper_m": (
                OPEN_RIGHT_FINGERTIP_CENTER_GRIPPER_M.tolist()
            ),
            "required_orientation": "unconstrained_pose_compensated_from_quaternion",
            "required_gripper_state": "fully_open",
        },
        "detected_corners": corners,
        "selected_touch_corners": selected,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[board] wrote {output} with {len(records)} touch measurements")


def cmd_board_solve(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.points_json).read_text())
    excluded_point_ids = sorted(set(args.exclude_point_id))
    if excluded_point_ids:
        original_records = payload.get("records", [])
        payload["records"] = [
            record
            for record in original_records
            if record.get("point_id") not in excluded_point_ids
        ]
        removed_count = len(original_records) - len(payload["records"])
        if removed_count == 0:
            raise ValueError(
                "none of the requested excluded point IDs exist: "
                + ", ".join(excluded_point_ids)
            )
        print(
            f"[board] excluded {removed_count} measurements from point IDs: "
            + ", ".join(excluded_point_ids)
        )
    calibration, audit = solve_chessboard_calibration(
        payload,
        ransac_threshold_m=args.ransac_threshold_m,
    )
    audit["excluded_touch_point_ids"] = excluded_point_ids
    audit["excluded_measurement_count"] = (
        len(json.loads(Path(args.points_json).read_text()).get("records", []))
        - len(payload["records"])
    )
    calibration["bimanual_calibration"]["excluded_touch_point_ids"] = (
        excluded_point_ids
    )
    _quality_gate(
        audit,
        maximum_rmse_m=args.maximum_rmse_m,
        maximum_error_m=args.maximum_error_m,
        maximum_overlap_rmse_m=args.maximum_overlap_rmse_m,
    )
    if float(audit["dense_homography_rmse_m"]) > args.maximum_dense_rmse_m:
        raise RuntimeError(
            "calibration quality gate failed: dense chessboard homography RMSE"
        )
    plane_z = float(calibration["plane_z"])
    if not TABLE_Z_SANITY_RANGE_M[0] <= plane_z <= TABLE_Z_SANITY_RANGE_M[1]:
        raise RuntimeError(f"calibrated contact plane z is implausible: {plane_z}")
    output = Path(args.output)
    audit_output = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(f"{output.stem}_backup_{stamp}{output.suffix}")
        shutil.copy2(output, backup)
        print(f"backed up existing calibration to {backup}")
    output.write_text(json.dumps(calibration, indent=2, ensure_ascii=False))
    audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"wrote chessboard calibration: {output}")
    print(f"wrote chessboard audit: {audit_output}")


def cmd_solve(args: argparse.Namespace) -> None:
    points_path = Path(args.points_json)
    payload = json.loads(points_path.read_text())
    calibration, audit = solve_bimanual_calibration(
        payload,
        ransac_threshold_m=args.ransac_threshold_m,
    )
    _quality_gate(
        audit,
        maximum_rmse_m=args.maximum_rmse_m,
        maximum_error_m=args.maximum_error_m,
        maximum_overlap_rmse_m=args.maximum_overlap_rmse_m,
    )
    output = Path(args.output)
    audit_output = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(f"{output.stem}_backup_{stamp}{output.suffix}")
        shutil.copy2(output, backup)
        print(f"backed up existing calibration to {backup}")
    output.write_text(json.dumps(calibration, indent=2, ensure_ascii=False))
    audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"wrote calibration: {output}")
    print(f"wrote audit: {audit_output}")


def cmd_check(args: argparse.Namespace) -> None:
    rgb = _wait_for_head_rgb(args.timeout)
    print(f"[OK] head camera: {rgb.shape[1]}x{rgb.shape[0]}")
    for side in ("left", "right"):
        position, quaternion, source, spread, orientation_spread = (
            _sample_gripper_link_pose(
            side,
            timeout_s=args.timeout,
            sample_count=min(5, args.pose_samples),
            )
        )
        print(
            f"[OK] {side}: {position.tolist()} source={source} "
            f"spread={spread*1000:.2f}mm "
            f"orientation_spread={np.degrees(orientation_spread):.2f}deg "
            f"quaternion={quaternion.tolist()}"
        )


def cmd_verify(args: argparse.Namespace) -> None:
    calibration = json.loads(Path(args.calibration).read_text())
    quality = calibration.get("bimanual_calibration", {}).get("quality")
    if not isinstance(quality, dict):
        raise SystemExit("calibration is not a bimanual calibration artifact")
    print(json.dumps(quality, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=20.0)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check")
    check.add_argument("--timeout", type=float, default=12.0)
    check.add_argument("--pose-samples", type=int, default=5)
    check.set_defaults(func=cmd_check)

    capture = sub.add_parser("capture")
    capture.add_argument("--output", default=str(DEFAULT_IMAGE))
    capture.add_argument("--timeout", type=float, default=20.0)
    capture.set_defaults(func=cmd_capture)

    collect = sub.add_parser("collect")
    collect.add_argument("--image", default=str(DEFAULT_IMAGE))
    collect.add_argument("--output", default=str(DEFAULT_POINTS))
    collect.add_argument("--left-only-points", type=int, default=5)
    collect.add_argument("--right-only-points", type=int, default=5)
    collect.add_argument("--overlap-points", type=int, default=4)
    collect.add_argument("--pose-samples", type=int, default=15)
    collect.add_argument("--maximum-pose-spread-m", type=float, default=0.002)
    collect.add_argument("--timeout", type=float, default=30.0)
    collect.add_argument("--no-recapture", action="store_true")
    collect.set_defaults(func=cmd_collect)

    board_detect = sub.add_parser(
        "board-detect",
        help="Detect the 13x17 inner corners and render selected touch points",
    )
    board_detect.add_argument("--image", default=str(DEFAULT_IMAGE))
    board_detect.add_argument("--preview", default=str(DEFAULT_BOARD_PREVIEW))
    board_detect.add_argument("--left-only-points", type=int, default=5)
    board_detect.add_argument("--right-only-points", type=int, default=5)
    board_detect.add_argument("--overlap-points", type=int, default=4)
    board_detect.add_argument("--timeout", type=float, default=20.0)
    board_detect.add_argument("--no-recapture", action="store_true")
    board_detect.set_defaults(func=cmd_board_detect)

    board_collect = sub.add_parser(
        "board-collect",
        help="Detect the chessboard and collect dual-arm corner contacts",
    )
    board_collect.add_argument("--image", default=str(DEFAULT_IMAGE))
    board_collect.add_argument("--preview", default=str(DEFAULT_BOARD_PREVIEW))
    board_collect.add_argument("--output", default=str(DEFAULT_BOARD_POINTS))
    board_collect.add_argument("--left-only-points", type=int, default=5)
    board_collect.add_argument("--right-only-points", type=int, default=5)
    board_collect.add_argument("--overlap-points", type=int, default=4)
    board_collect.add_argument("--pose-samples", type=int, default=15)
    board_collect.add_argument("--maximum-pose-spread-m", type=float, default=0.002)
    board_collect.add_argument(
        "--maximum-tool-z-tilt-deg",
        type=float,
        default=None,
        help=(
            "可选姿态硬门；默认不限制tool_z倾角，因为接触点会根据实时四元数"
            "从gripper_link转换到base_link"
        ),
    )
    board_collect.add_argument("--timeout", type=float, default=30.0)
    board_collect.add_argument("--no-recapture", action="store_true")
    board_collect.add_argument("--no-preview-window", action="store_true")
    board_collect.set_defaults(func=cmd_board_collect)

    board_solve = sub.add_parser(
        "board-solve",
        help="Register the 20mm chessboard in base and solve dense homography",
    )
    board_solve.add_argument("--points-json", default=str(DEFAULT_BOARD_POINTS))
    board_solve.add_argument("--output", default=str(DEFAULT_CALIBRATION))
    board_solve.add_argument(
        "--audit-output",
        default=str(REPO_ROOT / "outputs/r1lite/bimanual_chessboard_audit.json"),
    )
    board_solve.add_argument("--ransac-threshold-m", type=float, default=0.006)
    board_solve.add_argument(
        "--exclude-point-id",
        action="append",
        default=[],
        help=(
            "排除指定point_id的全部左右臂测量；可重复传入，原始采集文件不变"
        ),
    )
    board_solve.add_argument("--maximum-rmse-m", type=float, default=0.015)
    board_solve.add_argument("--maximum-error-m", type=float, default=0.030)
    board_solve.add_argument("--maximum-overlap-rmse-m", type=float, default=0.015)
    board_solve.add_argument("--maximum-dense-rmse-m", type=float, default=0.006)
    board_solve.set_defaults(func=cmd_board_solve)

    solve = sub.add_parser("solve")
    solve.add_argument("--points-json", default=str(DEFAULT_POINTS))
    solve.add_argument("--output", default=str(DEFAULT_CALIBRATION))
    solve.add_argument(
        "--audit-output",
        default=str(REPO_ROOT / "outputs/r1lite/bimanual_head_plane_audit.json"),
    )
    solve.add_argument("--ransac-threshold-m", type=float, default=0.012)
    solve.add_argument("--maximum-rmse-m", type=float, default=0.015)
    solve.add_argument("--maximum-error-m", type=float, default=0.030)
    solve.add_argument("--maximum-overlap-rmse-m", type=float, default=0.015)
    solve.set_defaults(func=cmd_solve)

    verify = sub.add_parser("verify")
    verify.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    verify.set_defaults(func=cmd_verify)
    return parser.parse_args()


def main() -> None:
    _setup_ros_env()
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
