#!/usr/bin/env python3
"""Eye-to-Hand hand-eye calibration for a fixed overhead camera (ROS Noetic).

The calibration board is mounted on the robot end-effector. The external camera
stays fixed while the arm moves. This script estimates T_base_camera using
checkerboard detection and cv2.calibrateHandEye.

End-effector pose defaults to pyAgxArm FK (same as point-correspondence
calibration). Use --pose-source tf when ROS TF is available.

Controls (click the image window first; keys do not work in the terminal):
  s  capture sample (checkerboard + arm pose)
  c  compute calibration (requires >= 15 samples)
  q  quit

Offline compute from saved samples.json:
  uv run scripts/eye_to_hand_calibration.py compute --write-config ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from agilex_camera_calib_io import (
    CALIB_TYPE_EYE_TO_HAND,
    DEFAULT_AGILEX_CONFIG,
    DEFAULT_EYE_TO_HAND_OUTPUT_DIR,
    apply_calibration_to_config,
    print_yaml_matrix,
    read_current_arm_joints,
    resolve_arm_config,
    T_base_ee_from_joints,
    validate_sample_format,
    write_calibration_result_json,
)


MIN_SAMPLES_FOR_COMPUTE = 15
RECOMMENDED_SAMPLES = (25, 40)
WINDOW_NAME = "eye-to-hand calibration (s=sample, c=compute, q=quit)"

HAND_EYE_METHODS: dict[str, int] = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _normalize_rgb(image: np.ndarray, encoding: str | None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if encoding and encoding.lower() in ("bgr8", "bgra8"):
        arr = arr[:, :, :3]
    return np.asarray(arr, dtype=np.uint8)


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4).tolist()


def _list_to_matrix(values: list[list[float]]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _transform_to_matrix(translation: Any, rotation_xyzw: Any) -> np.ndarray:
    trans = np.array(
        [translation.x, translation.y, translation.z],
        dtype=np.float64,
    )
    quat = np.array(
        [rotation_xyzw.x, rotation_xyzw.y, rotation_xyzw.z, rotation_xyzw.w],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError("Quaternion norm is zero")
    rot = Rotation.from_quat(quat / norm).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform


def _matrix_to_pose_dict(transform: np.ndarray) -> dict[str, Any]:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rot = transform[:3, :3]
    quat: list[float] | None
    try:
        quat = Rotation.from_matrix(rot).as_quat().tolist() if np.linalg.det(rot) > 0 else None
    except ValueError:
        quat = None
    return {
        "translation": transform[:3, 3].tolist(),
        "rotation_quaternion": quat,
        "rotation_matrix": rot.tolist(),
        "homogeneous_matrix": transform.tolist(),
    }


def _make_object_points(board_width: int, board_height: int, square_size: float) -> np.ndarray:
    cols, rows = board_width, board_height
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= float(square_size)
    return grid


def _detect_checkerboard_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board_width: int,
    board_height: int,
    square_size: float,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pattern_size = (board_width, board_height)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return False, None, None, None, float("inf")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    object_points = _make_object_points(board_width, board_height, square_size)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return False, corners, None, None, float("inf")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    observed = corners.reshape(-1, 2)
    reproj_error = float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))

    rot, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = tvec.reshape(3)
    return True, corners, transform, rvec, reproj_error


def _draw_detection(
    image_bgr: np.ndarray,
    corners: np.ndarray | None,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray | None,
    board_width: int,
    board_height: int,
    square_size: float,
    sample_count: int,
    status: str,
) -> np.ndarray:
    display = image_bgr.copy()
    if corners is not None:
        cv2.drawChessboardCorners(
            display,
            (board_width, board_height),
            corners,
            True,
        )
    if rvec is not None:
        axis_len = square_size * max(board_width, board_height) * 0.35
        cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, np.zeros((3, 1)), axis_len)

    cv2.putText(
        display,
        f"samples={sample_count}  min={MIN_SAMPLES_FOR_COMPUTE}  rec={RECOMMENDED_SAMPLES[0]}~{RECOMMENDED_SAMPLES[1]}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        status,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def _pose_error(T_pred: np.ndarray, T_true: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(T_true) @ T_pred
    trans_err = float(np.linalg.norm(delta[:3, 3]))
    rot = np.asarray(delta[:3, :3], dtype=np.float64)
    if np.linalg.det(rot) <= 0:
        rot_err = float("inf")
    else:
        try:
            rot_err = float(Rotation.from_matrix(rot).magnitude())
        except ValueError:
            rot_err = float("inf")
    return trans_err, rot_err


def _evaluate_calibration(
    T_base_camera: np.ndarray,
    samples: list["CalibrationSample"],
) -> dict[str, float]:
    mount_trans_errors: list[float] = []
    mount_rot_errors: list[float] = []
    axb_rot_errors: list[float] = []
    axb_trans_errors: list[float] = []

    # Board corner is not at the FK frame origin; check mount consistency instead.
    for sample in samples:
        T_base_board_est = T_base_camera @ sample.T_camera_board
        T_ee_board = np.linalg.inv(sample.T_base_ee) @ T_base_board_est
        mount_trans_errors.append(float(np.linalg.norm(T_ee_board[:3, 3])))
        try:
            mount_rot_errors.append(float(Rotation.from_matrix(T_ee_board[:3, :3]).magnitude()))
        except ValueError:
            mount_rot_errors.append(float("inf"))

    mount_trans = np.asarray(mount_trans_errors, dtype=np.float64)
    mount_rot = np.asarray(mount_rot_errors, dtype=np.float64)
    mount_trans_std = float(np.std(mount_trans))
    mount_rot_std = float(np.std(mount_rot))

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            A = samples[i].T_base_ee @ np.linalg.inv(samples[j].T_base_ee)
            B = samples[i].T_camera_board @ np.linalg.inv(samples[j].T_camera_board)
            lhs = A @ T_base_camera
            rhs = T_base_camera @ B
            delta = np.linalg.inv(rhs) @ lhs
            axb_trans_errors.append(float(np.linalg.norm(delta[:3, 3])))
            _, axb_r = _pose_error(lhs, rhs)
            axb_rot_errors.append(axb_r)

    return {
        "mount_translation_std_m": mount_trans_std,
        "mount_rotation_std_rad": mount_rot_std,
        "board_translation_rmse_m": mount_trans_std,
        "board_rotation_rmse_rad": mount_rot_std,
        "board_translation_mean_m": float(np.mean(mount_trans)),
        "board_rotation_mean_rad": float(np.mean(mount_rot)),
        "axb_translation_rmse_m": float(np.sqrt(np.mean(np.square(axb_trans_errors)))),
        "axb_rotation_rmse_rad": float(np.sqrt(np.mean(np.square(axb_rot_errors)))),
        "score": float(
            mount_trans_std
            + mount_rot_std
            + np.sqrt(np.mean(np.square(axb_trans_errors)))
            + np.sqrt(np.mean(np.square(axb_rot_errors)))
        ),
    }


def _run_hand_eye(samples: list["CalibrationSample"], method: int) -> np.ndarray:
    # Eye-to-hand: OpenCV expects base->gripper motion (R_base2gripper), not FK T_base_ee.
    # See OpenCV calibrateHandEye eye-to-hand AX=XB formulation and RM SDK examples.
    R_base2gripper: list[np.ndarray] = []
    t_base2gripper: list[np.ndarray] = []
    for sample in samples:
        T_base2gripper = np.linalg.inv(sample.T_base_ee)
        R_base2gripper.append(T_base2gripper[:3, :3])
        t_base2gripper.append(T_base2gripper[:3, 3].reshape(3, 1))
    R_target2cam = [sample.T_camera_board[:3, :3] for sample in samples]
    t_target2cam = [sample.T_camera_board[:3, 3].reshape(3, 1) for sample in samples]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper,
        t_base2gripper,
        R_target2cam,
        t_target2cam,
        method=method,
    )
    T_base_camera = np.eye(4, dtype=np.float64)
    T_base_camera[:3, :3] = R_cam2base
    T_base_camera[:3, 3] = t_cam2base.reshape(3)
    return T_base_camera


@dataclass
class CalibrationSample:
    index: int
    timestamp: float
    image_filename: str
    reprojection_error_px: float
    T_base_ee: np.ndarray
    T_camera_board: np.ndarray
    base_frame: str
    ee_frame: str
    pose_source: str = "pyagxarm"
    stamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "stamp": self.stamp,
            "image_filename": self.image_filename,
            "reprojection_error_px": self.reprojection_error_px,
            "base_frame": self.base_frame,
            "ee_frame": self.ee_frame,
            "pose_source": self.pose_source,
            "T_base_ee": _matrix_to_list(self.T_base_ee),
            "T_camera_board": _matrix_to_list(self.T_camera_board),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationSample":
        return cls(
            index=int(data["index"]),
            timestamp=float(data.get("timestamp", time.time())),
            image_filename=str(data["image_filename"]),
            reprojection_error_px=float(data["reprojection_error_px"]),
            T_base_ee=_list_to_matrix(data["T_base_ee"]),
            T_camera_board=_list_to_matrix(data["T_camera_board"]),
            base_frame=str(data.get("base_frame", "link0")),
            ee_frame=str(data.get("ee_frame", "link6")),
            pose_source=str(data.get("pose_source", "pyagxarm")),
            stamp=data.get("stamp"),
        )


def load_eye_to_hand_samples(path: Path) -> tuple[list[CalibrationSample], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    validate_sample_format(payload, CALIB_TYPE_EYE_TO_HAND, path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not an eye_to_hand samples file")
    samples = [CalibrationSample.from_dict(item) for item in payload.get("samples", [])]
    return samples, payload


def compute_eye_to_hand_calibration(
    samples: list[CalibrationSample],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    camera_frame_id: str,
) -> np.ndarray:
    count = len(samples)
    if count < MIN_SAMPLES_FOR_COMPUTE:
        raise RuntimeError(f"Only {count} samples in file; need at least {MIN_SAMPLES_FOR_COMPUTE}")

    output_dir = Path(output_dir)
    result_path = output_dir / "T_base_camera.yaml"
    result_json_path = output_dir / "calibration_result.json"

    method_results: dict[str, dict[str, Any]] = {}
    for method_name, method_id in HAND_EYE_METHODS.items():
        T_base_camera = _run_hand_eye(samples, method_id)
        metrics = _evaluate_calibration(T_base_camera, samples)
        method_results[method_name] = {
            "T_base_camera": T_base_camera,
            "metrics": metrics,
        }
        print(
            f"[compute] {method_name}: score={metrics['score']:.6f} "
            f"board_t_rmse={metrics['board_translation_rmse_m']:.5f} m "
            f"board_r_rmse={metrics['board_rotation_rmse_rad']:.5f} rad "
            f"axb_t_rmse={metrics['axb_translation_rmse_m']:.5f} m "
            f"axb_r_rmse={metrics['axb_rotation_rmse_rad']:.5f} rad"
        )

    finite_methods = {
        name: data
        for name, data in method_results.items()
        if np.isfinite(data["metrics"]["score"])
    }
    if not finite_methods:
        raise RuntimeError("All hand-eye methods failed; check sample quality and board mount")
    best_name = min(finite_methods, key=lambda name: finite_methods[name]["metrics"]["score"])
    best_T = method_results[best_name]["T_base_camera"]
    best_metrics = method_results[best_name]["metrics"]

    result_doc: dict[str, Any] = {
        "calibration_type": CALIB_TYPE_EYE_TO_HAND,
        "parent_frame": args.base_frame,
        "child_frame": camera_frame_id,
        "num_samples": count,
        "recommended_method": best_name,
        "pose_source": args.pose_source,
        "board": {
            "board_width": args.board_width,
            "board_height": args.board_height,
            "square_size_m": args.square_size,
        },
        "metrics": best_metrics,
        "all_methods": {
            name: {
                "metrics": data["metrics"],
                **_matrix_to_pose_dict(data["T_base_camera"]),
            }
            for name, data in method_results.items()
        },
        **_matrix_to_pose_dict(best_T),
    }
    result_path.write_text(yaml.safe_dump(result_doc, sort_keys=False))
    write_calibration_result_json(
        result_json_path,
        calibration_type=CALIB_TYPE_EYE_TO_HAND,
        transform=best_T,
        metrics=best_metrics,
        extra={
            "recommended_method": best_name,
            "num_samples": count,
            "pose_source": args.pose_source,
        },
    )

    print(f"[compute] recommended method: {best_name}")
    print(f"[compute] wrote {result_path}")
    print("[compute] T_base_camera / camera_pose_mats.main:")
    print(best_T)
    print_yaml_matrix(best_T)
    print(
        f"[compute] board_translation_rmse_m={best_metrics['board_translation_rmse_m']:.5f} "
        f"board_rotation_rmse_rad={best_metrics['board_rotation_rmse_rad']:.5f}"
    )

    if args.write_config:
        apply_calibration_to_config(
            Path(args.write_config),
            best_T,
            reset_graspnet_offsets=bool(args.reset_graspnet_offsets),
        )
        print(f"[compute] wrote camera_pose_mats.main to {args.write_config}")

    return best_T


@dataclass
class CalibrationState:
    args: argparse.Namespace
    samples: list[CalibrationSample] = field(default_factory=list)
    latest_bgr: np.ndarray | None = None
    latest_stamp: float | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    camera_frame_id: str = "camera_h_color_optical_frame"
    status: str = "waiting for image..."
    last_corners: np.ndarray | None = None
    last_rvec: np.ndarray | None = None
    last_reproj_error: float = float("inf")


class EyeToHandCalibrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.debug_dir = self.output_dir / "debug_images"
        self.samples_path = self.output_dir / "samples.json"
        self.result_path = self.output_dir / "T_base_camera.yaml"
        self.result_json_path = self.output_dir / "calibration_result.json"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.urdf_path, self.target_link_name, self.tcp_offset = resolve_arm_config(
            config_path=Path(args.config) if args.config else None,
            urdf=args.urdf,
            target_link=args.target_link,
            tcp_offset=args.tcp_offset,
        )

        self.state = CalibrationState(args=args)
        self._load_existing_samples()

        os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
        Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image

        if not rospy.core.is_initialized():
            rospy.init_node("capx_eye_to_hand_calibration", anonymous=True, disable_signals=True)

        self.rospy = rospy
        self.bridge = CvBridge()
        self.tf_buffer = None
        self.tf_listener = None
        if args.pose_source == "tf":
            import tf2_ros

            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber(args.image_topic, Image, self._image_callback, queue_size=1)
        rospy.Subscriber(args.camera_info_topic, CameraInfo, self._camera_info_callback, queue_size=1)

        print(f"[ros] image topic: {args.image_topic}")
        print(f"[ros] camera_info topic: {args.camera_info_topic}")
        print(f"[pose] source: {args.pose_source}")
        if args.pose_source == "tf":
            print(f"[ros] TF: {args.base_frame} -> {args.ee_frame}")
        else:
            print(
                f"[arm] FK urdf={self.urdf_path} link={self.target_link_name} "
                f"tcp_offset={np.array2string(self.tcp_offset, precision=4)}"
            )
        print(f"[ros] board inner corners: {args.board_width}x{args.board_height}, square={args.square_size} m")
        print(f"[io] output dir: {self.output_dir}")
        if args.write_config:
            print(f"[io] will write config on compute: {args.write_config}")
        print("[help] click this image window, then press s/c/q (terminal typing does not work)")

    def _load_existing_samples(self) -> None:
        if not self.samples_path.exists():
            return
        try:
            payload = json.loads(self.samples_path.read_text())
        except json.JSONDecodeError:
            print(f"[warn] could not parse {self.samples_path}; starting with empty samples")
            return
        try:
            validate_sample_format(payload, CALIB_TYPE_EYE_TO_HAND, self.samples_path)
        except ValueError as exc:
            print(f"[warn] {exc}; starting fresh in memory")
            return
        raw_samples = payload.get("samples", [])
        self.state.samples = [CalibrationSample.from_dict(item) for item in raw_samples]
        print(f"[io] loaded {len(self.state.samples)} existing eye_to_hand samples")

    def _image_callback(self, msg: Any) -> None:
        try:
            encoding = getattr(msg, "encoding", "bgr8")
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.state.latest_bgr = _normalize_rgb(image, encoding)
            if encoding.lower() == "rgb8":
                self.state.latest_bgr = cv2.cvtColor(self.state.latest_bgr, cv2.COLOR_RGB2BGR)
            self.state.latest_stamp = msg.header.stamp.to_sec()
            if self.state.status == "waiting for image...":
                self.state.status = "ready"
        except Exception as exc:
            self.state.status = f"image error: {exc}"

    def _camera_info_callback(self, msg: Any) -> None:
        self.state.camera_matrix = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
        self.state.dist_coeffs = np.asarray(msg.D, dtype=np.float64).reshape(-1)
        if msg.distortion_model == "plumb_bob" and self.state.dist_coeffs.size < 5:
            padded = np.zeros(5, dtype=np.float64)
            padded[: self.state.dist_coeffs.size] = self.state.dist_coeffs
            self.state.dist_coeffs = padded
        if msg.header.frame_id:
            self.state.camera_frame_id = msg.header.frame_id.strip("/")

    def _lookup_base_ee_tf(self) -> np.ndarray:
        if self.tf_buffer is None:
            raise RuntimeError("TF buffer not initialized; use --pose-source tf")
        timeout = self.rospy.Duration(self.args.tf_timeout)
        transform = self.tf_buffer.lookup_transform(
            self.args.base_frame,
            self.args.ee_frame,
            self.rospy.Time(0),
            timeout=timeout,
        )
        return _transform_to_matrix(transform.transform.translation, transform.transform.rotation)

    def _lookup_base_ee_pyagxarm(self) -> np.ndarray:
        joints = read_current_arm_joints(
            arm_channel=self.args.arm_channel,
            bitrate=self.args.bitrate,
            robot=self.args.robot,
        )
        return T_base_ee_from_joints(
            urdf_path=self.urdf_path,
            target_link_name=self.target_link_name,
            joints=joints,
            tcp_offset=self.tcp_offset,
        )

    def _lookup_base_ee(self) -> np.ndarray:
        if self.args.pose_source == "tf":
            return self._lookup_base_ee_tf()
        return self._lookup_base_ee_pyagxarm()

    def _save_samples(self) -> None:
        payload = {
            "calibration_type": CALIB_TYPE_EYE_TO_HAND,
            "pose_source": self.args.pose_source,
            "image_topic": self.args.image_topic,
            "camera_info_topic": self.args.camera_info_topic,
            "base_frame": self.args.base_frame,
            "ee_frame": self.args.ee_frame,
            "target_link": self.target_link_name,
            "tcp_offset": self.tcp_offset.tolist(),
            "board_width": self.args.board_width,
            "board_height": self.args.board_height,
            "square_size": self.args.square_size,
            "child_frame": self.state.camera_frame_id,
            "samples": [sample.to_dict() for sample in self.state.samples],
        }
        self.samples_path.write_text(json.dumps(payload, indent=2))

    def _capture_sample(self) -> None:
        if self.state.latest_bgr is None:
            self.state.status = "sample failed: no image"
            print("[sample] failed: no image received yet")
            return
        if self.state.camera_matrix is None or self.state.dist_coeffs is None:
            self.state.status = "sample failed: no camera_info"
            print("[sample] failed: camera_info not received yet")
            return

        found, corners, T_camera_board, rvec, reproj_error = _detect_checkerboard_pose(
            self.state.latest_bgr,
            self.state.camera_matrix,
            self.state.dist_coeffs,
            self.args.board_width,
            self.args.board_height,
            self.args.square_size,
        )
        self.state.last_corners = corners
        self.state.last_rvec = rvec
        self.state.last_reproj_error = reproj_error

        if not found or T_camera_board is None:
            self.state.status = "sample failed: checkerboard not detected"
            print("[sample] failed: checkerboard not detected")
            return

        try:
            T_base_ee = self._lookup_base_ee()
        except Exception as exc:
            self.state.status = f"sample failed: pose error ({exc})"
            print(f"[sample] failed: pose lookup ({self.args.pose_source}): {exc}")
            return

        sample_idx = len(self.state.samples)
        image_name = f"sample_{sample_idx:03d}.png"
        image_path = self.debug_dir / image_name
        annotated = _draw_detection(
            self.state.latest_bgr,
            corners,
            self.state.camera_matrix,
            self.state.dist_coeffs,
            rvec,
            self.args.board_width,
            self.args.board_height,
            self.args.square_size,
            sample_idx + 1,
            f"captured sample #{sample_idx}",
        )
        cv2.imwrite(str(image_path), annotated)

        sample = CalibrationSample(
            index=sample_idx,
            timestamp=time.time(),
            image_filename=str(Path("debug_images") / image_name),
            reprojection_error_px=reproj_error,
            T_base_ee=T_base_ee,
            T_camera_board=T_camera_board,
            base_frame=self.args.base_frame,
            ee_frame=self.args.ee_frame,
            pose_source=self.args.pose_source,
            stamp=self.state.latest_stamp,
        )
        self.state.samples.append(sample)
        self._save_samples()

        self.state.status = f"saved sample #{sample_idx} (reproj={reproj_error:.3f}px)"
        print(
            f"[sample] saved #{sample_idx}: reproj={reproj_error:.4f}px "
            f"pose_source={self.args.pose_source} total={len(self.state.samples)} image={image_path}"
        )
        if len(self.state.samples) < MIN_SAMPLES_FOR_COMPUTE:
            need = MIN_SAMPLES_FOR_COMPUTE - len(self.state.samples)
            print(f"[sample] need {need} more samples before compute")
        elif len(self.state.samples) < RECOMMENDED_SAMPLES[0]:
            print(f"[sample] recommended {RECOMMENDED_SAMPLES[0]}~{RECOMMENDED_SAMPLES[1]} samples")

    def _compute_calibration(self) -> None:
        count = len(self.state.samples)
        if count < MIN_SAMPLES_FOR_COMPUTE:
            self.state.status = f"compute failed: need >= {MIN_SAMPLES_FOR_COMPUTE} samples"
            print(f"[compute] failed: only {count} samples; need at least {MIN_SAMPLES_FOR_COMPUTE}")
            return
        try:
            compute_eye_to_hand_calibration(
                self.state.samples,
                self.args,
                output_dir=self.output_dir,
                camera_frame_id=self.state.camera_frame_id,
            )
        except Exception as exc:
            self.state.status = f"compute failed: {exc}"
            print(f"[compute] failed: {exc}")
            return
        self._save_samples()
        self.state.status = f"computed; saved {self.result_path.name}"

    def _render_preview(self) -> np.ndarray | None:
        if self.state.latest_bgr is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "waiting for image...", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return blank

        corners = self.state.last_corners
        rvec = self.state.last_rvec
        reproj = self.state.last_reproj_error
        status = self.state.status

        if self.state.camera_matrix is not None and self.state.dist_coeffs is not None:
            found, corners, _, rvec, reproj = _detect_checkerboard_pose(
                self.state.latest_bgr,
                self.state.camera_matrix,
                self.state.dist_coeffs,
                self.args.board_width,
                self.args.board_height,
                self.args.square_size,
            )
            if found:
                status = f"board detected, reproj={reproj:.3f}px"
            else:
                status = "board not detected"
            self.state.last_corners = corners
            self.state.last_rvec = rvec
            self.state.last_reproj_error = reproj

        return _draw_detection(
            self.state.latest_bgr,
            corners,
            self.state.camera_matrix if self.state.camera_matrix is not None else np.eye(3),
            self.state.dist_coeffs if self.state.dist_coeffs is not None else np.zeros(5),
            rvec,
            self.args.board_width,
            self.args.board_height,
            self.args.square_size,
            len(self.state.samples),
            status,
        )

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        rate = self.rospy.Rate(30)
        try:
            while not self.rospy.is_shutdown():
                preview = self._render_preview()
                if preview is not None:
                    cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    self._capture_sample()
                elif key == ord("c"):
                    self._compute_calibration()
                elif key in (ord("q"), 27):
                    break
                rate.sleep()
        finally:
            cv2.destroyAllWindows()


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=str(DEFAULT_EYE_TO_HAND_OUTPUT_DIR))
    parser.add_argument("--config", default=str(DEFAULT_AGILEX_CONFIG), help="Agilex env config for URDF/TCP defaults.")
    parser.add_argument("--base-frame", default="link0", help="Robot base frame for TF lookup.")
    parser.add_argument("--ee-frame", default="link6", help="End-effector frame where the board is mounted.")
    parser.add_argument("--board-width", type=int, default=9, help="Checkerboard inner corner columns.")
    parser.add_argument("--board-height", type=int, default=6, help="Checkerboard inner corner rows.")
    parser.add_argument("--square-size", type=float, default=0.025, help="Checkerboard square size in meters.")
    parser.add_argument("--pose-source", choices=("pyagxarm", "tf"), default="pyagxarm")
    parser.add_argument(
        "--write-config",
        default=None,
        help="On successful compute, write T_base_camera to camera_pose_mats.main in this yaml.",
    )
    parser.add_argument(
        "--reset-graspnet-offsets",
        action="store_true",
        help="When writing config, zero graspnet_world_offset and graspnet_rotation_correction_rpy.",
    )


def run_compute_offline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    samples_path = output_dir / "samples.json"
    samples, payload = load_eye_to_hand_samples(samples_path)
    if not samples:
        raise RuntimeError(f"No samples found in {samples_path}")
    print(f"[compute] loaded {len(samples)} samples from {samples_path}")
    camera_frame_id = str(payload.get("child_frame", "camera_h_color_optical_frame"))
    compute_eye_to_hand_calibration(
        samples,
        args,
        output_dir=output_dir,
        camera_frame_id=camera_frame_id,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eye-to-Hand checkerboard calibration for Agilex overhead camera.")
    subparsers = parser.add_subparsers(dest="cmd")

    run_parser = subparsers.add_parser("run", help="Interactive capture and calibration (default).")
    _add_shared_args(run_parser)
    run_parser.add_argument("--image-topic", default="/camera_h/color/image_raw")
    run_parser.add_argument("--camera-info-topic", default="/camera_h/color/camera_info")
    run_parser.add_argument("--tf-timeout", type=float, default=2.0, help="TF lookup timeout in seconds.")
    run_parser.add_argument("--urdf", default=None)
    run_parser.add_argument("--target-link", default=None)
    run_parser.add_argument("--tcp-offset", nargs=3, type=float, default=None)
    run_parser.add_argument("--arm-channel", default="can_left")
    run_parser.add_argument("--bitrate", type=int, default=1_000_000)
    run_parser.add_argument("--robot", default="piper")
    run_parser.set_defaults(func="run")

    compute_parser = subparsers.add_parser("compute", help="Compute from saved samples.json (no GUI).")
    _add_shared_args(compute_parser)
    compute_parser.set_defaults(func="compute")

    # Backward compatibility: flags without an explicit subcommand still run interactively.
    _add_shared_args(parser)
    parser.add_argument("--image-topic", default="/camera_h/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera_h/color/camera_info")
    parser.add_argument("--tf-timeout", type=float, default=2.0, help="TF lookup timeout in seconds.")
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--target-link", default=None)
    parser.add_argument("--tcp-offset", nargs=3, type=float, default=None)
    parser.add_argument("--arm-channel", default="can_left")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--robot", default="piper")
    parser.set_defaults(func="run")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if getattr(args, "func", "run") == "compute":
        run_compute_offline(args)
        return

    try:
        import rospy  # noqa: F401
    except ImportError as exc:
        print("rospy is required. Source ROS Noetic and retry.", file=sys.stderr)
        raise SystemExit(1) from exc

    calibrator = EyeToHandCalibrator(args)
    calibrator.run()


if __name__ == "__main__":
    main()
