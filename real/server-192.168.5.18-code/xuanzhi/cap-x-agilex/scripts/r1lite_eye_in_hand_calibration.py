#!/usr/bin/env python3
"""Capture and solve ROS 2 eye-in-hand calibration for the R1 Lite right wrist."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4).tolist()


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = yaml.safe_load(path.read_text())
    k = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(
        data["distortion_coefficients"]["data"],
        dtype=np.float64,
    ).reshape(-1)
    size = (int(data["image_height"]), int(data["image_width"]))
    return k, d, size


def _stamp_ns(msg: Any) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(
        msg.header.stamp.nanosec
    )


def _image_to_bgr(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        raise RuntimeError(f"Unsupported wrist image encoding: {msg.encoding}")
    channels = 4 if "a8" in encoding else 3
    row_bytes = int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), row_bytes)
    image = raw[:, : int(msg.width) * channels].reshape(
        int(msg.height),
        int(msg.width),
        channels,
    )
    if channels == 4:
        image = image[:, :, :3]
    if encoding.startswith("rgb"):
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def _pose_to_matrix(msg: Any) -> np.ndarray:
    if str(msg.header.frame_id).strip() != "base_link":
        raise RuntimeError(
            f"Expected EE pose in base_link, got {msg.header.frame_id!r}"
        )
    p = msg.pose.position
    q = msg.pose.orientation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        raise RuntimeError("EE pose has an invalid quaternion")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = SciRotation.from_quat(quat / norm).as_matrix()
    matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix


def _capture_messages(
    image_topic: str,
    pose_topic: str,
    timeout_s: float,
    max_skew_s: float,
) -> tuple[Any, Any]:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    if not rclpy.ok():
        rclpy.init()
    node = Node("capx_r1lite_hand_eye_capture")
    images: deque[Any] = deque(maxlen=60)
    poses: deque[Any] = deque(maxlen=120)
    node.create_subscription(
        Image,
        image_topic,
        images.append,
        qos_profile_sensor_data,
    )
    node.create_subscription(
        PoseStamped,
        pose_topic,
        poses.append,
        qos_profile_sensor_data,
    )
    deadline = time.time() + timeout_s
    best_skew_s = float("inf")
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if images and poses:
                image = images[-1]
                pose = min(
                    poses,
                    key=lambda item: abs(_stamp_ns(image) - _stamp_ns(item)),
                )
                skew_s = abs(_stamp_ns(image) - _stamp_ns(pose)) / 1e9
                best_skew_s = min(best_skew_s, skew_s)
                if skew_s <= max_skew_s:
                    return image, pose
        raise TimeoutError(
            "Timed out waiting for synchronized wrist image and EE pose: "
            f"images={len(images)}, poses={len(poses)}, "
            f"best_skew_s={best_skew_s:.4f}, limit={max_skew_s:.4f}"
        )
    finally:
        node.destroy_node()


def _detect_board(
    bgr: np.ndarray,
    *,
    board_cols: int,
    board_rows: int,
    square_size_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pattern = (board_cols, board_rows)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
    if not found or corners is None:
        raise RuntimeError(
            f"Could not detect {board_cols}x{board_rows} checkerboard"
        )
    object_points = np.zeros((board_cols * board_rows, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[
        0:board_cols,
        0:board_rows,
    ].T.reshape(-1, 2)
    object_points *= float(square_size_m)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for checkerboard")
    camera_from_board = np.eye(4, dtype=np.float64)
    camera_from_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_from_board[:3, 3] = np.asarray(tvec).reshape(3)
    return camera_from_board, corners, object_points


def capture(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.json"
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
            f"Image size {bgr.shape[:2]} does not match calibration "
            f"{calibrated_size}"
        )
    try:
        camera_from_board, corners, object_points = _detect_board(
            bgr,
            board_cols=args.board_cols,
            board_rows=args.board_rows,
            square_size_m=args.square_size,
            camera_matrix=camera_matrix,
            distortion=distortion,
        )
    except Exception:
        failure_path = output_dir / "detection_failed.jpg"
        cv2.imwrite(str(failure_path), bgr)
        print(f"Saved failed detection frame: {failure_path}", file=sys.stderr)
        raise
    base_from_ee = _pose_to_matrix(pose_msg)
    skew_s = abs(_stamp_ns(image_msg) - _stamp_ns(pose_msg)) / 1e9
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    corner_xy = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    hull_area = float(cv2.contourArea(cv2.convexHull(corner_xy.astype(np.float32))))
    coverage = hull_area / float(bgr.shape[0] * bgr.shape[1])
    border_margin_px = float(
        min(
            corner_xy[:, 0].min(),
            corner_xy[:, 1].min(),
            bgr.shape[1] - 1 - corner_xy[:, 0].max(),
            bgr.shape[0] - 1 - corner_xy[:, 1].max(),
        )
    )
    rvec = cv2.Rodrigues(camera_from_board[:3, :3])[0]
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        camera_from_board[:3, 3],
        camera_matrix,
        distortion,
    )
    reprojection_rmse_px = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        projected.reshape(-1, 2)
                        - corner_xy
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )
    failures = []
    if blur_score < args.min_blur_score:
        failures.append(
            f"blur score {blur_score:.1f} < {args.min_blur_score:.1f}"
        )
    if not (args.min_coverage <= coverage <= args.max_coverage):
        failures.append(
            f"board coverage {coverage:.3f} outside "
            f"[{args.min_coverage:.3f}, {args.max_coverage:.3f}]"
        )
    if border_margin_px < args.min_border_margin:
        failures.append(
            f"border margin {border_margin_px:.1f}px < "
            f"{args.min_border_margin:.1f}px"
        )
    if reprojection_rmse_px > args.max_reprojection_error:
        failures.append(
            f"reprojection RMSE {reprojection_rmse_px:.3f}px > "
            f"{args.max_reprojection_error:.3f}px"
        )
    payload = {
        "board_cols": args.board_cols,
        "board_rows": args.board_rows,
        "square_size_m": args.square_size,
        "camera_frame_id": str(image_msg.header.frame_id),
        "samples": [],
    }
    if samples_path.exists():
        payload = json.loads(samples_path.read_text())
        expected = (
            int(payload["board_cols"]),
            int(payload["board_rows"]),
            float(payload["square_size_m"]),
        )
        actual = (args.board_cols, args.board_rows, args.square_size)
        if expected != actual:
            raise RuntimeError(
                f"Existing sample board {expected} differs from requested {actual}"
            )
        for index, previous in enumerate(payload.get("samples", []), start=1):
            previous_pose = np.asarray(
                previous["T_base_ee"],
                dtype=np.float64,
            ).reshape(4, 4)
            translation_delta = float(
                np.linalg.norm(base_from_ee[:3, 3] - previous_pose[:3, 3])
            )
            rotation_delta_deg = float(
                np.degrees(
                    SciRotation.from_matrix(
                        previous_pose[:3, :3].T @ base_from_ee[:3, :3]
                    ).magnitude()
                )
            )
            if (
                translation_delta < args.min_pose_translation
                and rotation_delta_deg < args.min_pose_rotation_deg
            ):
                failures.append(
                    f"pose duplicates sample {index}: "
                    f"translation={translation_delta:.3f}m, "
                    f"rotation={rotation_delta_deg:.1f}deg"
                )
                break
    if failures:
        rejected_path = output_dir / "quality_rejected.jpg"
        cv2.imwrite(str(rejected_path), bgr)
        raise RuntimeError(
            "Calibration sample rejected: " + "; ".join(failures)
        )
    sample = {
        "image_stamp_ns": _stamp_ns(image_msg),
        "pose_stamp_ns": _stamp_ns(pose_msg),
        "skew_s": skew_s,
        "T_base_ee": _matrix_to_list(base_from_ee),
        "T_camera_board": _matrix_to_list(camera_from_board),
        "quality": {
            "blur_score": blur_score,
            "coverage": coverage,
            "border_margin_px": border_margin_px,
            "reprojection_rmse_px": reprojection_rmse_px,
        },
    }
    payload["samples"].append(sample)
    samples_path.write_text(json.dumps(payload, indent=2) + "\n")
    index = len(payload["samples"])
    annotated = bgr.copy()
    cv2.drawChessboardCorners(
        annotated,
        (args.board_cols, args.board_rows),
        corners,
        True,
    )
    image_path = output_dir / f"sample_{index:02d}.jpg"
    cv2.imwrite(str(image_path), annotated)
    print(
        json.dumps(
            {
                "captured": index,
                "image": str(image_path),
                "skew_s": skew_s,
                "board_distance_m": float(
                    np.linalg.norm(camera_from_board[:3, 3])
                ),
                "ee_position": base_from_ee[:3, 3].tolist(),
                "quality": sample["quality"],
            },
            indent=2,
        )
    )


def _rotation_spread_rad(rotations: list[np.ndarray]) -> float:
    quaternions = np.asarray(
        [SciRotation.from_matrix(rotation).as_quat() for rotation in rotations]
    )
    reference = quaternions[0]
    quaternions[np.sum(quaternions * reference, axis=1) < 0] *= -1
    mean = quaternions.mean(axis=0)
    mean /= np.linalg.norm(mean)
    return float(
        max(
            2.0 * np.arccos(np.clip(abs(np.dot(quaternion, mean)), 0.0, 1.0))
            for quaternion in quaternions
        )
    )


def compute(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    samples_path = output_dir / "samples.json"
    payload = json.loads(samples_path.read_text())
    samples = payload["samples"]
    if len(samples) < args.min_samples:
        raise RuntimeError(
            f"Only {len(samples)} samples; need at least {args.min_samples}"
        )
    base_from_ee = [
        np.asarray(sample["T_base_ee"], dtype=np.float64) for sample in samples
    ]
    camera_from_board = [
        np.asarray(sample["T_camera_board"], dtype=np.float64)
        for sample in samples
    ]
    reference_rotation = base_from_ee[0][:3, :3]
    motion_rotvecs = np.asarray(
        [
            SciRotation.from_matrix(
                reference_rotation.T @ matrix[:3, :3]
            ).as_rotvec()
            for matrix in base_from_ee
        ]
    )
    motion_angles_deg = np.degrees(np.linalg.norm(motion_rotvecs, axis=1))
    singular_values = np.linalg.svd(motion_rotvecs, compute_uv=False)
    if float(motion_angles_deg.max()) < args.min_rotation_span_deg:
        raise RuntimeError(
            f"Insufficient rotation span: {motion_angles_deg.max():.1f}deg < "
            f"{args.min_rotation_span_deg:.1f}deg"
        )
    if len(singular_values) < 2 or float(singular_values[1]) < args.min_second_axis_rad:
        raise RuntimeError(
            "Insufficient rotation diversity across two axes: "
            f"singular_values={singular_values.tolist()}"
        )
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    results: dict[str, Any] = {}
    for name, method in methods.items():
        rotation, translation = cv2.calibrateHandEye(
            [matrix[:3, :3] for matrix in base_from_ee],
            [matrix[:3, 3].reshape(3, 1) for matrix in base_from_ee],
            [matrix[:3, :3] for matrix in camera_from_board],
            [matrix[:3, 3].reshape(3, 1) for matrix in camera_from_board],
            method=method,
        )
        ee_from_camera = np.eye(4, dtype=np.float64)
        ee_from_camera[:3, :3] = rotation
        ee_from_camera[:3, 3] = np.asarray(translation).reshape(3)
        board_poses = [
            base_ee @ ee_from_camera @ camera_board
            for base_ee, camera_board in zip(
                base_from_ee,
                camera_from_board,
                strict=True,
            )
        ]
        positions = np.asarray([pose[:3, 3] for pose in board_poses])
        center = positions.mean(axis=0)
        errors = np.linalg.norm(positions - center, axis=1)
        translation_rmse = float(np.sqrt(np.mean(errors**2)))
        rotation_spread = _rotation_spread_rad(
            [pose[:3, :3] for pose in board_poses]
        )
        score = translation_rmse + 0.05 * rotation_spread
        results[name] = {
            "T_ee_camera": _matrix_to_list(ee_from_camera),
            "translation_rmse_m": translation_rmse,
            "rotation_spread_rad": rotation_spread,
            "score": score,
        }
    finite = {
        name: result
        for name, result in results.items()
        if np.isfinite(result["score"])
    }
    if not finite:
        raise RuntimeError("All hand-eye calibration methods returned invalid results")
    best_name = min(finite, key=lambda name: finite[name]["score"])
    best = finite[best_name]
    rotation_spread_deg = float(np.degrees(best["rotation_spread_rad"]))
    if best["translation_rmse_m"] > args.max_translation_rmse:
        raise RuntimeError(
            f"Calibration translation RMSE {best['translation_rmse_m']:.4f}m "
            f"exceeds {args.max_translation_rmse:.4f}m"
        )
    if rotation_spread_deg > args.max_rotation_spread_deg:
        raise RuntimeError(
            f"Calibration rotation spread {rotation_spread_deg:.2f}deg "
            f"exceeds {args.max_rotation_spread_deg:.2f}deg"
        )
    result = {
        "calibration_type": "r1lite_right_eye_in_hand",
        "method": best_name,
        "camera_frame_id": payload["camera_frame_id"],
        "num_samples": len(samples),
        "T_ee_camera": best["T_ee_camera"],
        "translation_rmse_m": best["translation_rmse_m"],
        "rotation_spread_rad": best["rotation_spread_rad"],
        "rotation_spread_deg": rotation_spread_deg,
        "motion_rotation_span_deg": float(motion_angles_deg.max()),
        "motion_rotation_singular_values": singular_values.tolist(),
        "all_methods": results,
    }
    result_path = output_dir / "T_ee_camera.yaml"
    result_path.write_text(yaml.safe_dump(result, sort_keys=False))
    print(yaml.safe_dump(result, sort_keys=False))


def status(args: argparse.Namespace) -> None:
    samples_path = Path(args.output_dir) / "samples.json"
    if not samples_path.exists():
        print(json.dumps({"samples": 0}))
        return
    payload = json.loads(samples_path.read_text())
    print(json.dumps({"samples": len(payload.get("samples", []))}, indent=2))


def reset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        return
    for path in output_dir.glob("sample_*.jpg"):
        path.unlink()
    for filename in ("samples.json", "T_ee_camera.yaml"):
        path = output_dir / filename
        if path.exists():
            path.unlink()
    print(f"reset {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/r1lite/right_eye_in_hand_calibration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument(
        "--image-topic",
        default="/hdas/camera_wrist_right/color/image_raw",
    )
    capture_parser.add_argument(
        "--pose-topic",
        default="/motion_control/pose_ee_arm_right",
    )
    capture_parser.add_argument(
        "--intrinsics",
        default="/home/nvidia/wrist_right.yaml",
    )
    capture_parser.add_argument("--board-cols", type=int, default=9)
    capture_parser.add_argument("--board-rows", type=int, default=6)
    capture_parser.add_argument("--square-size", type=float, default=0.020)
    capture_parser.add_argument("--timeout", type=float, default=12.0)
    capture_parser.add_argument("--max-sync-skew", type=float, default=0.20)
    capture_parser.add_argument("--min-blur-score", type=float, default=60.0)
    capture_parser.add_argument("--min-coverage", type=float, default=0.04)
    capture_parser.add_argument("--max-coverage", type=float, default=0.65)
    capture_parser.add_argument("--min-border-margin", type=float, default=20.0)
    capture_parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=0.8,
    )
    capture_parser.add_argument(
        "--min-pose-translation",
        type=float,
        default=0.025,
    )
    capture_parser.add_argument(
        "--min-pose-rotation-deg",
        type=float,
        default=6.0,
    )
    capture_parser.set_defaults(func=capture)
    compute_parser = subparsers.add_parser("compute")
    compute_parser.add_argument("--min-samples", type=int, default=9)
    compute_parser.add_argument(
        "--min-rotation-span-deg",
        type=float,
        default=25.0,
    )
    compute_parser.add_argument(
        "--min-second-axis-rad",
        type=float,
        default=0.12,
    )
    compute_parser.add_argument(
        "--max-translation-rmse",
        type=float,
        default=0.012,
    )
    compute_parser.add_argument(
        "--max-rotation-spread-deg",
        type=float,
        default=5.0,
    )
    compute_parser.set_defaults(func=compute)
    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=status)
    reset_parser = subparsers.add_parser("reset")
    reset_parser.set_defaults(func=reset)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
