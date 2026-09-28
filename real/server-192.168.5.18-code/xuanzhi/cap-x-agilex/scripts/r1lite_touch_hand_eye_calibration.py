#!/usr/bin/env python3
"""Touch-assisted hand-eye calibration for the R1 Lite right wrist camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciRotation
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from r1lite_eye_in_hand_calibration import (  # noqa: E402
    _capture_messages,
    _detect_board,
    _image_to_bgr,
    _load_intrinsics,
    _matrix_to_list,
    _pose_to_matrix,
)

DEFAULT_OUTPUT = Path("outputs/r1lite/right_touch_hand_eye_calibration")


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else default


def _capture_stable_ee_pose(
    topic: str,
    *,
    timeout_s: float,
    duration_s: float = 0.8,
) -> tuple[np.ndarray, dict[str, float]]:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    if not rclpy.ok():
        rclpy.init()
    node = Node("capx_r1lite_touch_pose_capture")
    messages: list[Any] = []
    node.create_subscription(
        PoseStamped,
        topic,
        messages.append,
        qos_profile_sensor_data,
    )
    deadline = time.time() + timeout_s
    started: float | None = None
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if messages and started is None:
                started = time.time()
            if started is not None and time.time() - started >= duration_s:
                break
        if len(messages) < 5:
            raise TimeoutError(
                f"Only received {len(messages)} EE poses from {topic}"
            )
    finally:
        node.destroy_node()

    matrices = [_pose_to_matrix(message) for message in messages]
    positions = np.asarray([matrix[:3, 3] for matrix in matrices])
    quaternions = np.asarray(
        [SciRotation.from_matrix(matrix[:3, :3]).as_quat() for matrix in matrices]
    )
    reference = quaternions[0]
    quaternions[np.sum(quaternions * reference, axis=1) < 0] *= -1
    mean_quaternion = quaternions.mean(axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    mean_position = positions.mean(axis=0)
    position_std = float(np.linalg.norm(positions.std(axis=0)))
    rotation_deviations = [
        2.0
        * np.arccos(
            np.clip(abs(np.dot(quaternion, mean_quaternion)), 0.0, 1.0)
        )
        for quaternion in quaternions
    ]
    rotation_max_deg = float(np.degrees(max(rotation_deviations)))
    if position_std > 0.0015 or rotation_max_deg > 0.5:
        raise RuntimeError(
            "Arm is not stationary enough for touch capture: "
            f"position_std={position_std:.4f}m, "
            f"rotation_max={rotation_max_deg:.2f}deg"
        )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = SciRotation.from_quat(mean_quaternion).as_matrix()
    matrix[:3, 3] = mean_position
    return matrix, {
        "position_std_m": position_std,
        "rotation_max_deg": rotation_max_deg,
        "num_pose_messages": len(messages),
    }


def touch(args: argparse.Namespace) -> None:
    if not (0 <= args.col <= 8 and 0 <= args.row <= 5):
        raise ValueError("For a 9x6 inner-corner board, col=0..8 and row=0..5")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "touches.json"
    payload = _load_json(
        path,
        {
            "square_size_m": args.square_size,
            "touches": [],
        },
    )
    if not np.isclose(float(payload["square_size_m"]), args.square_size):
        raise RuntimeError("Existing touch data uses another square size")
    base_from_ee, stability = _capture_stable_ee_pose(
        args.pose_topic,
        timeout_s=args.timeout,
    )
    sample = {
        "col": args.col,
        "row": args.row,
        "point_board": [
            args.col * args.square_size,
            args.row * args.square_size,
            0.0,
        ],
        "T_base_ee": _matrix_to_list(base_from_ee),
        "stability": stability,
    }
    payload["touches"].append(sample)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "captured_touch": len(payload["touches"]),
                "board_corner": [args.col, args.row],
                "point_board_m": sample["point_board"],
                "ee_position": base_from_ee[:3, 3].tolist(),
                "stability": stability,
            },
            indent=2,
        )
    )


def _initial_board_pose(
    board_points: np.ndarray,
    ee_origins: np.ndarray,
) -> np.ndarray:
    board_center = board_points.mean(axis=0)
    base_center = ee_origins.mean(axis=0)
    source = board_points - board_center
    target = ee_origins - base_center
    covariance = source.T @ target
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = base_center - rotation @ board_center
    return matrix


def solve_board(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    payload = json.loads((output_dir / "touches.json").read_text())
    touches = payload["touches"]
    if len(touches) < args.min_touches:
        raise RuntimeError(
            f"Only {len(touches)} touches; need at least {args.min_touches}"
        )
    unique_points = {
        (int(touch["col"]), int(touch["row"])) for touch in touches
    }
    required = {(0, 0), (8, 0), (8, 5), (0, 5)}
    if not required.issubset(unique_points):
        raise RuntimeError(
            f"Touches must include all four outer inner corners: missing "
            f"{sorted(required - unique_points)}"
        )
    board_points = np.asarray(
        [touch["point_board"] for touch in touches],
        dtype=np.float64,
    )
    base_from_ee = [
        np.asarray(touch["T_base_ee"], dtype=np.float64).reshape(4, 4)
        for touch in touches
    ]
    initial_board = _initial_board_pose(
        board_points,
        np.asarray([matrix[:3, 3] for matrix in base_from_ee]),
    )
    initial = np.concatenate(
        [
            SciRotation.from_matrix(initial_board[:3, :3]).as_rotvec(),
            initial_board[:3, 3],
            np.zeros(3),
        ]
    )

    def residual(params: np.ndarray) -> np.ndarray:
        board_rotation = SciRotation.from_rotvec(params[:3]).as_matrix()
        board_translation = params[3:6]
        tip_in_ee = params[6:9]
        errors = []
        for point_board, matrix in zip(
            board_points,
            base_from_ee,
            strict=True,
        ):
            point_from_robot = (
                matrix[:3, :3] @ tip_in_ee + matrix[:3, 3]
            )
            point_from_board = board_rotation @ point_board + board_translation
            errors.append(point_from_robot - point_from_board)
        return np.concatenate(errors)

    solved = least_squares(
        residual,
        initial,
        loss="soft_l1",
        f_scale=0.002,
        max_nfev=5000,
    )
    if not solved.success:
        raise RuntimeError(f"Touch solve failed: {solved.message}")
    errors = residual(solved.x).reshape(-1, 3)
    norms = np.linalg.norm(errors, axis=1)
    rmse = float(np.sqrt(np.mean(norms**2)))
    max_error = float(norms.max())
    if rmse > args.max_rmse or max_error > args.max_error:
        raise RuntimeError(
            f"Touch calibration rejected: rmse={rmse:.4f}m, "
            f"max_error={max_error:.4f}m"
        )
    base_from_board = np.eye(4, dtype=np.float64)
    base_from_board[:3, :3] = SciRotation.from_rotvec(
        solved.x[:3]
    ).as_matrix()
    base_from_board[:3, 3] = solved.x[3:6]
    result = {
        "T_base_board": _matrix_to_list(base_from_board),
        "tip_in_ee": solved.x[6:9].tolist(),
        "num_touches": len(touches),
        "touch_rmse_m": rmse,
        "touch_max_error_m": max_error,
        "touch_errors_m": norms.tolist(),
    }
    (output_dir / "touch_result.yaml").write_text(
        yaml.safe_dump(result, sort_keys=False)
    )
    print(yaml.safe_dump(result, sort_keys=False))


def camera(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    result = yaml.safe_load((output_dir / "touch_result.yaml").read_text())
    camera_matrix, distortion, calibrated_size = _load_intrinsics(
        Path(args.intrinsics)
    )
    image_msg, pose_msg = _capture_messages(
        args.image_topic,
        args.pose_topic,
        args.timeout,
        args.max_sync_skew,
    )
    bgr = _image_to_bgr(image_msg)
    if bgr.shape[:2] != calibrated_size:
        raise RuntimeError(
            f"Image size {bgr.shape[:2]} != calibrated size {calibrated_size}"
        )
    camera_from_board, corners, _ = _detect_board(
        bgr,
        board_cols=9,
        board_rows=6,
        square_size_m=args.square_size,
        camera_matrix=camera_matrix,
        distortion=distortion,
    )
    base_from_ee = _pose_to_matrix(pose_msg)
    path = output_dir / "camera_samples.json"
    payload = _load_json(path, {"samples": []})
    payload["samples"].append(
        {
            "T_base_ee": _matrix_to_list(base_from_ee),
            "T_camera_board": _matrix_to_list(camera_from_board),
        }
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    index = len(payload["samples"])
    cv2.drawChessboardCorners(bgr, (9, 6), corners, True)
    cv2.imwrite(str(output_dir / f"camera_{index:02d}.jpg"), bgr)
    print(
        json.dumps(
            {
                "captured_camera": index,
                "touch_rmse_m": result["touch_rmse_m"],
            },
            indent=2,
        )
    )


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    translations = np.asarray([matrix[:3, 3] for matrix in transforms])
    quaternions = np.asarray(
        [SciRotation.from_matrix(matrix[:3, :3]).as_quat() for matrix in transforms]
    )
    reference = quaternions[0]
    quaternions[np.sum(quaternions * reference, axis=1) < 0] *= -1
    quaternion = quaternions.mean(axis=0)
    quaternion /= np.linalg.norm(quaternion)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = SciRotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translations.mean(axis=0)
    return matrix


def compute(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    touch_result = yaml.safe_load(
        (output_dir / "touch_result.yaml").read_text()
    )
    base_from_board = np.asarray(
        touch_result["T_base_board"],
        dtype=np.float64,
    ).reshape(4, 4)
    camera_payload = json.loads(
        (output_dir / "camera_samples.json").read_text()
    )
    samples = camera_payload["samples"]
    if len(samples) < args.min_camera_samples:
        raise RuntimeError(
            f"Only {len(samples)} camera samples; "
            f"need {args.min_camera_samples}"
        )
    ee_from_camera_samples = []
    for sample in samples:
        base_from_ee = np.asarray(
            sample["T_base_ee"],
            dtype=np.float64,
        ).reshape(4, 4)
        camera_from_board = np.asarray(
            sample["T_camera_board"],
            dtype=np.float64,
        ).reshape(4, 4)
        ee_from_camera_samples.append(
            np.linalg.inv(base_from_ee)
            @ base_from_board
            @ np.linalg.inv(camera_from_board)
        )
    ee_from_camera = _average_transforms(ee_from_camera_samples)
    translation_errors = [
        float(np.linalg.norm(matrix[:3, 3] - ee_from_camera[:3, 3]))
        for matrix in ee_from_camera_samples
    ]
    rotation_errors_deg = [
        float(
            np.degrees(
                SciRotation.from_matrix(
                    ee_from_camera[:3, :3].T @ matrix[:3, :3]
                ).magnitude()
            )
        )
        for matrix in ee_from_camera_samples
    ]
    if max(translation_errors) > args.max_translation_spread:
        raise RuntimeError(
            f"Camera translation spread too large: {translation_errors}"
        )
    if max(rotation_errors_deg) > args.max_rotation_spread_deg:
        raise RuntimeError(
            f"Camera rotation spread too large: {rotation_errors_deg}"
        )
    result = {
        "calibration_type": "r1lite_touch_assisted_eye_in_hand",
        "T_ee_camera": _matrix_to_list(ee_from_camera),
        "num_touches": touch_result["num_touches"],
        "num_camera_samples": len(samples),
        "touch_rmse_m": touch_result["touch_rmse_m"],
        "translation_spread_m": translation_errors,
        "rotation_spread_deg": rotation_errors_deg,
    }
    (output_dir / "T_ee_camera.yaml").write_text(
        yaml.safe_dump(result, sort_keys=False)
    )
    print(yaml.safe_dump(result, sort_keys=False))


def reset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "touches.json",
        "touch_result.yaml",
        "camera_samples.json",
        "T_ee_camera.yaml",
        "camera_*.jpg",
    ):
        for path in output_dir.glob(pattern):
            path.unlink()
    print(f"reset {output_dir}")


def status(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    touches = _load_json(output_dir / "touches.json", {"touches": []})
    cameras = _load_json(
        output_dir / "camera_samples.json",
        {"samples": []},
    )
    print(
        json.dumps(
            {
                "touches": len(touches["touches"]),
                "camera_samples": len(cameras["samples"]),
                "board_solved": (output_dir / "touch_result.yaml").exists(),
                "calibration_solved": (output_dir / "T_ee_camera.yaml").exists(),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    commands = parser.add_subparsers(dest="command", required=True)
    touch_parser = commands.add_parser("touch")
    touch_parser.add_argument("--col", type=int, required=True)
    touch_parser.add_argument("--row", type=int, required=True)
    touch_parser.add_argument("--square-size", type=float, default=0.020)
    touch_parser.add_argument(
        "--pose-topic",
        default="/motion_control/pose_ee_arm_right",
    )
    touch_parser.add_argument("--timeout", type=float, default=10.0)
    touch_parser.set_defaults(func=touch)
    board_parser = commands.add_parser("solve-board")
    board_parser.add_argument("--min-touches", type=int, default=8)
    board_parser.add_argument("--max-rmse", type=float, default=0.004)
    board_parser.add_argument("--max-error", type=float, default=0.008)
    board_parser.set_defaults(func=solve_board)
    camera_parser = commands.add_parser("camera")
    camera_parser.add_argument(
        "--image-topic",
        default="/hdas/camera_wrist_right/color/image_raw",
    )
    camera_parser.add_argument(
        "--pose-topic",
        default="/motion_control/pose_ee_arm_right",
    )
    camera_parser.add_argument(
        "--intrinsics",
        default="/home/nvidia/wrist_right.yaml",
    )
    camera_parser.add_argument("--square-size", type=float, default=0.020)
    camera_parser.add_argument("--timeout", type=float, default=15.0)
    camera_parser.add_argument("--max-sync-skew", type=float, default=0.20)
    camera_parser.set_defaults(func=camera)
    compute_parser = commands.add_parser("compute")
    compute_parser.add_argument("--min-camera-samples", type=int, default=3)
    compute_parser.add_argument(
        "--max-translation-spread",
        type=float,
        default=0.012,
    )
    compute_parser.add_argument(
        "--max-rotation-spread-deg",
        type=float,
        default=5.0,
    )
    compute_parser.set_defaults(func=compute)
    reset_parser = commands.add_parser("reset")
    reset_parser.set_defaults(func=reset)
    status_parser = commands.add_parser("status")
    status_parser.set_defaults(func=status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
