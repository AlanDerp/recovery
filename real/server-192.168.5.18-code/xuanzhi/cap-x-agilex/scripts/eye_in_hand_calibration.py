#!/usr/bin/env python3
"""Eye-in-Hand checkerboard calibration for a wrist-mounted camera (ROS Noetic).

The checkerboard stays fixed on the table. The arm moves so the wrist camera sees
the board from many poses. This estimates T_ee_camera (camera relative to the EE).

Controls (click the image window first):
  s  capture sample (checkerboard + arm pose)
  c  compute calibration (requires >= 15 samples)
  q  quit

Example (left wrist camera):
  uv run --no-sync --active scripts/eye_in_hand_calibration.py run \\
    --image-topic /camera_l/color/image_raw \\
    --camera-info-topic /camera_l/color/camera_info \\
    --target-link link6 --tcp-offset 0 0 0 \\
    --board-width 7 --board-height 5 --square-size 0.020
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from agilex_camera_calib_io import (
    DEFAULT_AGILEX_CONFIG,
    T_base_ee_from_joints,
    read_current_arm_joints,
    read_joints_from_ros,
    resolve_arm_config,
    write_calibration_result_json,
)
from eye_to_hand_calibration import (
    HAND_EYE_METHODS,
    MIN_SAMPLES_FOR_COMPUTE,
    RECOMMENDED_SAMPLES,
    CalibrationSample,
    EyeToHandCalibrator,
    _draw_detection,
    _detect_checkerboard_pose,
)

CALIB_TYPE_EYE_IN_HAND = "eye_in_hand"
DEFAULT_EYE_IN_HAND_OUTPUT_DIR = Path("outputs/agilex_eye_in_hand_calibration")
WINDOW_NAME = "eye-in-hand calibration (s=sample, c=compute, q=quit)"


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4).tolist()


def _run_hand_eye_in_hand(samples: list[CalibrationSample], method: int) -> np.ndarray:
    """OpenCV eye-in-hand: solve T_ee_camera (camera in EE frame)."""
    R_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    for sample in samples:
        T = sample.T_base_ee
        R_gripper2base.append(T[:3, :3])
        t_gripper2base.append(T[:3, 3].reshape(3, 1))
    R_target2cam = [sample.T_camera_board[:3, :3] for sample in samples]
    t_target2cam = [sample.T_camera_board[:3, 3].reshape(3, 1) for sample in samples]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=method,
    )
    T_ee_camera = np.eye(4, dtype=np.float64)
    T_ee_camera[:3, :3] = R_cam2gripper
    T_ee_camera[:3, 3] = t_cam2gripper.reshape(3)
    return T_ee_camera


def _evaluate_eye_in_hand(T_ee_camera: np.ndarray, samples: list[CalibrationSample]) -> dict[str, float]:
    """Fixed board on table -> T_base_board should be stable across samples."""
    board_origins: list[np.ndarray] = []
    for sample in samples:
        T_base_board = sample.T_base_ee @ T_ee_camera @ sample.T_camera_board
        board_origins.append(T_base_board[:3, 3])
    origins = np.asarray(board_origins, dtype=np.float64)
    centroid = origins.mean(axis=0)
    errors = np.linalg.norm(origins - centroid, axis=1)
    return {
        "board_origin_std_m": float(np.std(origins, axis=0).max()),
        "board_origin_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "board_origin_mean_m": float(np.mean(errors)),
    }


def compute_eye_in_hand_calibration(
    samples: list[CalibrationSample],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    camera_frame_id: str,
) -> np.ndarray:
    count = len(samples)
    if count < MIN_SAMPLES_FOR_COMPUTE:
        raise RuntimeError(f"Only {count} samples; need at least {MIN_SAMPLES_FOR_COMPUTE}")

    output_dir = Path(output_dir)
    result_path = output_dir / "T_ee_camera.yaml"
    result_json_path = output_dir / "calibration_result.json"

    method_results: dict[str, dict[str, Any]] = {}
    for method_name, method_id in HAND_EYE_METHODS.items():
        T_ee_camera = _run_hand_eye_in_hand(samples, method_id)
        metrics = _evaluate_eye_in_hand(T_ee_camera, samples)
        method_results[method_name] = {
            "T_ee_camera": T_ee_camera,
            "metrics": metrics,
        }
        print(
            f"[compute] {method_name}: board_origin_rmse={metrics['board_origin_rmse_m']:.5f} m "
            f"board_origin_std={metrics['board_origin_std_m']:.5f} m"
        )

    best_name = min(method_results, key=lambda k: method_results[k]["metrics"]["board_origin_rmse_m"])
    best = method_results[best_name]
    T_ee_camera = best["T_ee_camera"]
    best_metrics = best["metrics"]

    result_path.write_text(
        yaml.safe_dump(
            {
                "calibration_type": CALIB_TYPE_EYE_IN_HAND,
                "method": best_name,
                "camera_frame_id": camera_frame_id,
                "T_ee_camera": _matrix_to_list(T_ee_camera),
            },
            sort_keys=False,
        )
    )
    write_calibration_result_json(
        result_json_path,
        calibration_type=CALIB_TYPE_EYE_IN_HAND,
        transform=T_ee_camera,
        metrics=best_metrics,
        extra={
            "method": best_name,
            "camera_frame_id": camera_frame_id,
            "num_samples": count,
            "all_methods": {
                name: {
                    "metrics": data["metrics"],
                    "T_ee_camera": _matrix_to_list(data["T_ee_camera"]),
                }
                for name, data in method_results.items()
            },
        },
    )

    print(f"[compute] best method={best_name}")
    print(f"[compute] saved {result_path}")
    print(f"[compute] saved {result_json_path}")
    print("[compute] T_ee_camera (camera in EE frame):")
    print(T_ee_camera)
    print("\n# Paste into yaml if needed, e.g. wrist_camera_pose_mats.left:")
    for row in T_ee_camera.reshape(4, 4):
        values = ", ".join(f"{float(v):.9f}" for v in row)
        print(f"    - [{values}]")
    return T_ee_camera


class EyeInHandCalibrator(EyeToHandCalibrator):
    def __init__(self, args: argparse.Namespace) -> None:
        if not hasattr(args, "write_config"):
            args.write_config = None
        if not hasattr(args, "reset_graspnet_offsets"):
            args.reset_graspnet_offsets = False
        super().__init__(args)
        self.result_path = self.output_dir / "T_ee_camera.yaml"

    def _load_existing_samples(self) -> None:
        if not self.samples_path.exists():
            return
        try:
            payload = json.loads(self.samples_path.read_text())
        except json.JSONDecodeError:
            print(f"[warn] could not parse {self.samples_path}; starting empty")
            return
        if payload.get("calibration_type") not in (CALIB_TYPE_EYE_IN_HAND, None):
            print(
                f"[warn] {self.samples_path} is not eye_in_hand "
                f"(type={payload.get('calibration_type')}); starting fresh"
            )
            return
        raw_samples = payload.get("samples", [])
        self.state.samples = [CalibrationSample.from_dict(item) for item in raw_samples]
        print(f"[io] loaded {len(self.state.samples)} existing eye_in_hand samples")

    def _lookup_base_ee_pyagxarm(self) -> np.ndarray:
        try:
            joints = read_current_arm_joints(
                arm_channel=self.args.arm_channel,
                bitrate=self.args.bitrate,
                robot=self.args.robot,
            )
        except RuntimeError as exc:
            if "pyAgxArm" not in str(exc):
                raise
            joint_topic = getattr(self.args, "joint_topic", "/puppet/joint_left")
            print(f"[pose] pyAgxArm unavailable; using ROS joints {joint_topic}")
            joints = read_joints_from_ros(topic=joint_topic, timeout=self.args.timeout)
        return T_base_ee_from_joints(
            urdf_path=self.urdf_path,
            target_link_name=self.target_link_name,
            joints=joints,
            tcp_offset=self.tcp_offset,
        )

    def _save_samples(self) -> None:
        payload = {
            "calibration_type": CALIB_TYPE_EYE_IN_HAND,
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

    def _compute_calibration(self) -> None:
        count = len(self.state.samples)
        if count < MIN_SAMPLES_FOR_COMPUTE:
            self.state.status = f"compute failed: need >= {MIN_SAMPLES_FOR_COMPUTE} samples"
            print(f"[compute] failed: only {count} samples; need at least {MIN_SAMPLES_FOR_COMPUTE}")
            return
        try:
            compute_eye_in_hand_calibration(
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

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("[help] board must stay fixed on the table; move the arm, not the board")
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


def load_eye_in_hand_samples(path: Path) -> tuple[list[CalibrationSample], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a valid samples file")
    samples = [CalibrationSample.from_dict(item) for item in payload.get("samples", [])]
    return samples, payload


def run_compute_offline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    samples_path = output_dir / "samples.json"
    samples, payload = load_eye_in_hand_samples(samples_path)
    if payload.get("calibration_type") not in (CALIB_TYPE_EYE_IN_HAND, None):
        raise RuntimeError(f"{samples_path} is not an eye_in_hand samples file")
    if not samples:
        raise RuntimeError(f"No samples found in {samples_path}")
    camera_frame_id = str(payload.get("child_frame", "camera_l_color_optical_frame"))
    compute_eye_in_hand_calibration(
        samples,
        args,
        output_dir=output_dir,
        camera_frame_id=camera_frame_id,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eye-in-Hand wrist camera calibration (board on table).")
    subparsers = parser.add_subparsers(dest="cmd")

    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", default=str(DEFAULT_EYE_IN_HAND_OUTPUT_DIR))
        p.add_argument("--config", default=str(DEFAULT_AGILEX_CONFIG))
        p.add_argument("--base-frame", default="link0")
        p.add_argument("--ee-frame", default="link6", help="EE frame where the wrist camera is mounted.")
        p.add_argument("--board-width", type=int, default=7)
        p.add_argument("--board-height", type=int, default=5)
        p.add_argument("--square-size", type=float, default=0.020)
        p.add_argument("--pose-source", choices=("pyagxarm", "tf"), default="pyagxarm")
        p.add_argument("--timeout", type=float, default=10.0)

    run_parser = subparsers.add_parser("run")
    add_shared(run_parser)
    run_parser.add_argument("--image-topic", default="/camera_l/color/image_raw")
    run_parser.add_argument("--camera-info-topic", default="/camera_l/color/camera_info")
    run_parser.add_argument("--tf-timeout", type=float, default=2.0)
    run_parser.add_argument("--urdf", default=None)
    run_parser.add_argument("--target-link", default="link6")
    run_parser.add_argument("--tcp-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    run_parser.add_argument("--arm-channel", default="can_left")
    run_parser.add_argument("--bitrate", type=int, default=1_000_000)
    run_parser.add_argument("--robot", default="piper")
    run_parser.add_argument("--joint-topic", default="/puppet/joint_left")
    run_parser.set_defaults(func="run")

    compute_parser = subparsers.add_parser("compute")
    add_shared(compute_parser)
    compute_parser.set_defaults(func="compute")

    add_shared(parser)
    parser.add_argument("--image-topic", default="/camera_l/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera_l/color/camera_info")
    parser.add_argument("--tf-timeout", type=float, default=2.0)
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--target-link", default="link6")
    parser.add_argument("--tcp-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--arm-channel", default="can_left")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--robot", default="piper")
    parser.add_argument("--joint-topic", default="/puppet/joint_left")
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

    calibrator = EyeInHandCalibrator(args)
    calibrator.run()


if __name__ == "__main__":
    main()
