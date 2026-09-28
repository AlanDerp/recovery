#!/usr/bin/env python3
"""Visualize Agilex camera calibration accuracy.

Offline (from saved samples):
  uv run --no-sync --active scripts/visualize_agilex_camera_calibration.py \\
    --mode samples --samples-file outputs/agilex_point_calibration/samples.json

Live (project current TCP onto the camera image + 3D alignment check):
  uv run --no-sync --active scripts/visualize_agilex_camera_calibration.py \\
    --mode live --config env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agilex_camera_calib_io import (  # noqa: E402
    CALIB_TYPE_EYE_TO_HAND,
    CALIB_TYPE_POINT_CORRESPONDENCE,
    DEFAULT_AGILEX_CONFIG,
    DEFAULT_EYE_TO_HAND_OUTPUT_DIR,
    DEFAULT_POINT_OUTPUT_DIR,
    base_point_from_joints,
    estimate_camera_to_base,
    load_point_correspondence_samples,
    read_T_base_camera,
    read_current_arm_joints,
    read_joints_from_ros,
    read_tcp_from_ros_end_pose,
    resolve_arm_config,
    validate_sample_format,
)
from eye_to_hand_calibration import (  # noqa: E402
    CalibrationSample,
    _evaluate_calibration,
    load_eye_to_hand_samples,
)
from visualize_agilex_urdf import _add_frame, _compute_fk, _parse_urdf  # noqa: E402

DEFAULT_CONFIG = DEFAULT_AGILEX_CONFIG


def _load_low_level_config(config_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    low_level = dict(OmegaConf.to_container(cfg.env.cfg.low_level, resolve=True))
    low_level.pop("_target_", None)
    return low_level


def _load_T_base_camera(
    *,
    config_path: Path | None,
    calibration_file: Path | None,
) -> np.ndarray:
    if calibration_file is not None:
        return read_T_base_camera(calibration_file)
    if config_path is None:
        raise ValueError("Provide --calibration-file or --config with camera_pose_mats.main")
    low_level = _load_low_level_config(config_path)
    pose = low_level.get("camera_pose_mats", {}).get("main")
    if pose is None:
        raise ValueError(f"No camera_pose_mats.main in {config_path}")
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def _normalize_rgb(image: np.ndarray, encoding: str | None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if encoding is not None and encoding.lower() == "bgr8":
        arr = arr[:, :, ::-1]
    return np.asarray(arr, dtype=np.uint8)


def _normalize_depth(image: np.ndarray, depth_scale: float) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) * depth_scale
    else:
        arr = arr.astype(np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _wait_for_rgbd(low_level: dict[str, Any], timeout_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError("rospy, cv_bridge, and sensor_msgs are required") from exc

    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_camera_calib_viz", anonymous=True, disable_signals=True)

    rgb_topic = low_level.get("img_front_topic", "/camera_h/color/image_raw")
    depth_topic = low_level.get("img_front_depth_topic", "/camera_h/depth/image_raw")
    camera_info_topic = low_level.get("img_front_camera_info_topic", "/camera_h/color/camera_info")
    bridge = CvBridge()

    print(f"[ros] waiting RGB: {rgb_topic}")
    rgb_msg = rospy.wait_for_message(rgb_topic, Image, timeout=timeout_s)
    print(f"[ros] waiting depth: {depth_topic}")
    depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=timeout_s)
    print(f"[ros] waiting camera_info: {camera_info_topic}")
    info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=timeout_s)

    rgb = _normalize_rgb(
        bridge.imgmsg_to_cv2(rgb_msg, "passthrough"),
        getattr(rgb_msg, "encoding", None),
    )
    depth = _normalize_depth(
        bridge.imgmsg_to_cv2(depth_msg, "passthrough"),
        float(low_level.get("depth_scale", 0.001)),
    )
    intrinsics = np.asarray(info_msg.K, dtype=np.float64).reshape(3, 3)
    if depth.shape[:2] != rgb.shape[:2]:
        from capx.utils.camera_utils import scale_intrinsics_for_image

        intrinsics = scale_intrinsics_for_image(
            intrinsics,
            depth.shape[:2],
            (int(info_msg.height), int(info_msg.width)),
        )
    return rgb, depth, intrinsics


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=1)
    out = (T @ homo.T).T[:, :3]
    return out


def _project_base_to_pixel(
    point_base: np.ndarray,
    *,
    T_base_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[float, float] | None:
    T_camera_base = np.linalg.inv(T_base_camera)
    p_cam = T_camera_base @ np.append(np.asarray(point_base, dtype=np.float64).reshape(3), 1.0)
    if p_cam[2] <= 1e-6:
        return None
    u = intrinsics[0, 0] * p_cam[0] / p_cam[2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * p_cam[1] / p_cam[2] + intrinsics[1, 2]
    return float(u), float(v)


def _points_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
    *,
    depth_range: tuple[float, float],
    max_points: int = 12000,
) -> np.ndarray:
    near, far = depth_range
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[ys, xs]
    x = (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_cam = np.stack([x, y, z], axis=1)
    points_h = np.concatenate(
        [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_world = (pose_mat @ points_h.T).T[:, :3]
    points_world = points_world[np.isfinite(points_world).all(axis=1)]
    if len(points_world) <= max_points:
        return points_world
    idx = np.random.default_rng(0).choice(len(points_world), max_points, replace=False)
    return points_world[idx]


def _annotate_overlay(
    rgb: np.ndarray,
    *,
    title: str,
    markers: list[dict[str, Any]],
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for 2D overlay output") from exc

    display = rgb[:, :, ::-1].copy()
    for marker in markers:
        u, v = marker["pixel"]
        color = marker.get("color", (0, 255, 0))
        label = marker.get("label", "")
        err_mm = marker.get("error_mm")
        err_px = marker.get("error_px")
        cv2.drawMarker(
            display,
            (int(round(u)), int(round(v))),
            color,
            cv2.MARKER_CROSS,
            24,
            2,
        )
        text = label
        if err_mm is not None:
            text = f"{label} {err_mm:.1f}mm"
        elif err_px is not None:
            text = f"{label} {err_px:.1f}px"
        if text:
            cv2.putText(
                display,
                text,
                (int(round(u)) + 8, int(round(v)) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        display,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display[:, :, ::-1]


def _print_point_metrics(
    *,
    samples: list[dict[str, Any]],
    T_base_camera: np.ndarray,
) -> tuple[float, np.ndarray]:
    camera_pts = np.asarray([s["camera_point"] for s in samples], dtype=np.float64)
    base_pts = np.asarray([s["base_point"] for s in samples], dtype=np.float64)
    pred_pts = _transform_points(T_base_camera, camera_pts)
    errors_m = np.linalg.norm(pred_pts - base_pts, axis=1)
    rms_m = float(np.sqrt(np.mean(errors_m**2)))
    print(f"[metrics] point_correspondence samples={len(samples)}")
    print(f"[metrics] rms_error_mm={rms_m * 1000.0:.2f}")
    print(f"[metrics] max_error_mm={float(np.max(errors_m)) * 1000.0:.2f}")
    print(f"[metrics] per_sample_error_mm={np.round(errors_m * 1000.0, 2).tolist()}")
    return rms_m, errors_m


def _print_eye_to_hand_metrics(
    *,
    samples: list[CalibrationSample],
    T_base_camera: np.ndarray,
) -> dict[str, float]:
    metrics = _evaluate_calibration(T_base_camera, samples)
    print(f"[metrics] eye_to_hand samples={len(samples)}")
    print(
        "[metrics] board_translation_rmse_mm="
        f"{metrics['board_translation_rmse_m'] * 1000.0:.2f} "
        f"board_rotation_rmse_deg={np.degrees(metrics['board_rotation_rmse_rad']):.2f}"
    )
    print(
        "[metrics] axb_translation_rmse_mm="
        f"{metrics['axb_translation_rmse_m'] * 1000.0:.2f} "
        f"axb_rotation_rmse_deg={np.degrees(metrics['axb_rotation_rmse_rad']):.2f}"
    )
    print(f"[metrics] score={metrics['score']:.6f}")
    return metrics


def _build_point_correspondence_figure(
    *,
    samples: list[dict[str, Any]],
    T_base_camera: np.ndarray,
    errors_m: np.ndarray,
) -> go.Figure:
    camera_pts = np.asarray([s["camera_point"] for s in samples], dtype=np.float64)
    base_pts = np.asarray([s["base_point"] for s in samples], dtype=np.float64)
    pred_pts = _transform_points(T_base_camera, camera_pts)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=base_pts[:, 0],
            y=base_pts[:, 1],
            z=base_pts[:, 2],
            mode="markers+text",
            marker={"size": 5, "color": "#2ca02c"},
            text=[f"gt#{i}" for i in range(len(base_pts))],
            name="ground truth base points",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=pred_pts[:, 0],
            y=pred_pts[:, 1],
            z=pred_pts[:, 2],
            mode="markers+text",
            marker={"size": 5, "color": "#d62728"},
            text=[f"pred#{i}" for i in range(len(pred_pts))],
            name="predicted from camera",
        )
    )
    for idx, (gt, pred) in enumerate(zip(base_pts, pred_pts)):
        fig.add_trace(
            go.Scatter3d(
                x=[gt[0], pred[0]],
                y=[gt[1], pred[1]],
                z=[gt[2], pred[2]],
                mode="lines",
                line={"color": "#ff7f0e", "width": 4},
                name=f"error#{idx}",
                showlegend=False,
                hovertext=[f"{errors_m[idx] * 1000.0:.1f} mm"],
            )
        )
    _add_frame(fig, np.eye(4), "base", axis_len=0.08)
    _add_frame(fig, T_base_camera, "camera_main", axis_len=0.08)
    fig.update_layout(
        title="Point-correspondence calibration accuracy",
        scene={
            "xaxis": {"title": "base X (m)"},
            "yaxis": {"title": "base Y (m)"},
            "zaxis": {"title": "base Z (m)"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def _build_eye_to_hand_figure(
    *,
    samples: list[CalibrationSample],
    T_base_camera: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    board_positions: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []
    for sample in samples:
        T_base_board = T_base_camera @ sample.T_camera_board
        board_positions.append(T_base_board[:3, 3])
        ee_positions.append(sample.T_base_ee[:3, 3])
    board_positions_arr = np.asarray(board_positions, dtype=np.float64)
    ee_positions_arr = np.asarray(ee_positions, dtype=np.float64)

    fig.add_trace(
        go.Scatter3d(
            x=ee_positions_arr[:, 0],
            y=ee_positions_arr[:, 1],
            z=ee_positions_arr[:, 2],
            mode="markers",
            marker={"size": 4, "color": "#1f77b4"},
            name="end-effector positions",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=board_positions_arr[:, 0],
            y=board_positions_arr[:, 1],
            z=board_positions_arr[:, 2],
            mode="markers",
            marker={"size": 4, "color": "#ff4fa3"},
            name="estimated board origins",
        )
    )
    _add_frame(fig, np.eye(4), "base", axis_len=0.08)
    _add_frame(fig, T_base_camera, "camera_main", axis_len=0.08)
    fig.update_layout(
        title="Eye-to-hand calibration consistency",
        scene={
            "xaxis": {"title": "base X (m)"},
            "yaxis": {"title": "base Y (m)"},
            "zaxis": {"title": "base Z (m)"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def _build_live_figure(
    *,
    T_base_camera: np.ndarray,
    scene_points: np.ndarray,
    tcp_base: np.ndarray,
    urdf_path: Path,
    joints: np.ndarray,
    target_link_name: str,
    tcp_offset: np.ndarray,
) -> go.Figure:
    _, urdf_joints = _parse_urdf(urdf_path)
    joint_values = {f"joint{idx + 1}": float(value) for idx, value in enumerate(joints)}
    link_poses = _compute_fk(urdf_joints, joint_values)

    fig = go.Figure()
    if len(scene_points) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=scene_points[:, 0],
                y=scene_points[:, 1],
                z=scene_points[:, 2],
                mode="markers",
                marker={"size": 1.2, "color": "rgba(120,120,120,0.25)"},
                name="depth point cloud (base frame)",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[tcp_base[0]],
            y=[tcp_base[1]],
            z=[tcp_base[2]],
            mode="markers+text",
            marker={"size": 6, "color": "#ff4fa3"},
            text=["TCP"],
            name="current TCP",
        )
    )
    for joint in urdf_joints:
        if joint.parent not in link_poses or joint.child not in link_poses:
            continue
        parent_pos = link_poses[joint.parent][:3, 3]
        child_pos = link_poses[joint.child][:3, 3]
        fig.add_trace(
            go.Scatter3d(
                x=[parent_pos[0], child_pos[0]],
                y=[parent_pos[1], child_pos[1]],
                z=[parent_pos[2], child_pos[2]],
                mode="lines",
                line={"color": "#444444", "width": 6},
                showlegend=False,
            )
        )
    if target_link_name in link_poses:
        _add_frame(fig, link_poses[target_link_name], target_link_name, axis_len=0.05)
    _add_frame(fig, np.eye(4), "base", axis_len=0.08)
    _add_frame(fig, T_base_camera, "camera_main", axis_len=0.08)
    fig.update_layout(
        title="Live calibration check: point cloud + TCP in base frame",
        scene={
            "xaxis": {"title": "base X (m)"},
            "yaxis": {"title": "base Y (m)"},
            "zaxis": {"title": "base Z (m)"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def _detect_sample_type(samples_path: Path) -> str:
    payload = json.loads(samples_path.read_text())
    if isinstance(payload, list):
        return CALIB_TYPE_POINT_CORRESPONDENCE
    if isinstance(payload, dict) and payload.get("calibration_type") == CALIB_TYPE_EYE_TO_HAND:
        return CALIB_TYPE_EYE_TO_HAND
    validate_sample_format(payload, CALIB_TYPE_POINT_CORRESPONDENCE, samples_path)
    return CALIB_TYPE_POINT_CORRESPONDENCE


def run_samples_mode(args: argparse.Namespace) -> None:
    samples_path = Path(args.samples_file)
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    T_base_camera = _load_T_base_camera(
        config_path=Path(args.config) if args.config else None,
        calibration_file=Path(args.calibration_file) if args.calibration_file else None,
    )
    sample_type = _detect_sample_type(samples_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markers: list[dict[str, Any]] = []
    rgb: np.ndarray | None = None
    intrinsics: np.ndarray | None = None

    if sample_type == CALIB_TYPE_POINT_CORRESPONDENCE:
        samples = load_point_correspondence_samples(samples_path)
        if not samples:
            raise RuntimeError(f"No samples in {samples_path}")
        _, errors_m = _print_point_metrics(samples=samples, T_base_camera=T_base_camera)
        fig = _build_point_correspondence_figure(
            samples=samples,
            T_base_camera=T_base_camera,
            errors_m=errors_m,
        )
        if args.config:
            low_level = _load_low_level_config(Path(args.config))
            rgb, _, intrinsics = _wait_for_rgbd(low_level, args.timeout)
            camera_pts = np.asarray([s["camera_point"] for s in samples], dtype=np.float64)
            pred_pts = _transform_points(T_base_camera, camera_pts)
            for idx, sample in enumerate(samples):
                u, v = sample["pixel"]
                pred = _project_base_to_pixel(
                    pred_pts[idx],
                    T_base_camera=T_base_camera,
                    intrinsics=intrinsics,
                )
                err_px = None
                if pred is not None:
                    err_px = float(np.hypot(pred[0] - u, pred[1] - v))
                markers.append(
                    {
                        "pixel": (u, v),
                        "color": (0, 255, 0),
                        "label": f"click#{idx}",
                    }
                )
                if pred is not None:
                    markers.append(
                        {
                            "pixel": pred,
                            "color": (255, 0, 0),
                            "label": f"reproj#{idx}",
                            "error_px": err_px,
                        }
                    )
            reproj_px = [
                float(np.hypot(pred[0] - sample["pixel"][0], pred[1] - sample["pixel"][1]))
                for sample, pred in zip(samples, [_project_base_to_pixel(p, T_base_camera=T_base_camera, intrinsics=intrinsics) for p in pred_pts])
                if pred is not None
            ]
            if reproj_px:
                reproj_rms = float(np.sqrt(np.mean(np.square(reproj_px))))
                print(f"[metrics] reproj_rms_px={reproj_rms:.2f}")
                print(f"[metrics] reproj_max_px={float(np.max(reproj_px)):.2f}")
    else:
        samples, _meta = load_eye_to_hand_samples(samples_path)
        if not samples:
            raise RuntimeError(f"No eye-to-hand samples in {samples_path}")
        _print_eye_to_hand_metrics(samples=samples, T_base_camera=T_base_camera)
        fig = _build_eye_to_hand_figure(samples=samples, T_base_camera=T_base_camera)

    html_path = output_dir / "calibration_accuracy_3d.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"[output] wrote {html_path}")

    if rgb is not None and intrinsics is not None and markers:
        overlay = _annotate_overlay(
            rgb,
            title="Green=clicked pixel, Blue=calibrated 3D point reprojected",
            markers=markers,
        )
        png_path = output_dir / "calibration_reprojection_overlay.png"
        PILImage.fromarray(overlay).save(png_path)
        print(f"[output] wrote {png_path}")


def _resolve_live_arm_state(
    args: argparse.Namespace,
    *,
    low_level: dict[str, Any],
    config_path: Path,
) -> tuple[np.ndarray | None, np.ndarray, str]:
    urdf_path, target_link_name, tcp_offset = resolve_arm_config(
        config_path=config_path,
        urdf=args.urdf,
        target_link=args.target_link,
        tcp_offset=args.tcp_offset,
    )
    pose_source = args.pose_source
    joint_topic = args.joint_topic or low_level.get(
        "puppet_joint_topic",
        "/puppet/joint_left",
    )

    if pose_source == "auto":
        try:
            import pyAgxArm  # noqa: F401

            pose_source = "pyagxarm"
        except ImportError:
            pose_source = "ros_joints"
            print("[pose] pyAgxArm not installed; falling back to ros_joints")

    if pose_source == "pyagxarm":
        joints = read_current_arm_joints(
            arm_channel=args.arm_channel,
            bitrate=args.bitrate,
            robot=args.robot,
        )
        tcp_base = base_point_from_joints(
            urdf_path=urdf_path,
            target_link_name=target_link_name,
            joints=joints,
            tcp_offset=tcp_offset,
        )
        print(f"[pose] source=pyagxarm joints={np.array2string(joints, precision=4)}")
        return joints, tcp_base, pose_source

    if pose_source == "ros_joints":
        joints = read_joints_from_ros(topic=str(joint_topic), timeout=args.timeout)
        tcp_base = base_point_from_joints(
            urdf_path=urdf_path,
            target_link_name=target_link_name,
            joints=joints,
            tcp_offset=tcp_offset,
        )
        print(
            f"[pose] source=ros_joints topic={joint_topic} "
            f"joints={np.array2string(joints, precision=4)}"
        )
        return joints, tcp_base, pose_source

    if pose_source == "ros_end_pose":
        tcp_base = read_tcp_from_ros_end_pose(
            topic=args.end_pose_topic,
            marker_offset=tcp_offset,
            timeout=args.timeout,
        )
        print(f"[pose] source=ros_end_pose topic={args.end_pose_topic}")
        return None, tcp_base, pose_source

    raise ValueError(f"Unknown pose source: {pose_source}")


def run_live_mode(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    low_level = _load_low_level_config(config_path)
    T_base_camera = _load_T_base_camera(
        config_path=config_path,
        calibration_file=Path(args.calibration_file) if args.calibration_file else None,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb, depth, intrinsics = _wait_for_rgbd(low_level, args.timeout)
    urdf_path, target_link_name, tcp_offset = resolve_arm_config(
        config_path=config_path,
        urdf=args.urdf,
        target_link=args.target_link,
        tcp_offset=args.tcp_offset,
    )
    joints, tcp_base, pose_source = _resolve_live_arm_state(
        args,
        low_level=low_level,
        config_path=config_path,
    )
    projected = _project_base_to_pixel(
        tcp_base,
        T_base_camera=T_base_camera,
        intrinsics=intrinsics,
    )
    if joints is not None:
        print(f"[live] joints={np.array2string(joints, precision=4)}")
    print(f"[live] pose_source={pose_source}")
    print(f"[live] tcp_base_m={np.array2string(tcp_base, precision=5)}")
    if projected is None:
        print("[live] TCP projects behind the camera; move arm into view")
    else:
        print(f"[live] tcp_projected_pixel=({projected[0]:.1f}, {projected[1]:.1f})")

    depth_range = tuple(float(v) for v in args.depth_range)
    scene_points = _points_from_depth(
        depth,
        intrinsics,
        T_base_camera,
        depth_range=depth_range,
        max_points=args.max_points,
    )
    if joints is not None:
        fig = _build_live_figure(
            T_base_camera=T_base_camera,
            scene_points=scene_points,
            tcp_base=tcp_base,
            urdf_path=urdf_path,
            joints=joints,
            target_link_name=target_link_name,
            tcp_offset=tcp_offset,
        )
    else:
        fig = go.Figure()
        if len(scene_points) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=scene_points[:, 0],
                    y=scene_points[:, 1],
                    z=scene_points[:, 2],
                    mode="markers",
                    marker={"size": 1.2, "color": "rgba(120,120,120,0.25)"},
                    name="depth point cloud (base frame)",
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[tcp_base[0]],
                y=[tcp_base[1]],
                z=[tcp_base[2]],
                mode="markers+text",
                marker={"size": 6, "color": "#ff4fa3"},
                text=["TCP"],
                name="current TCP",
            )
        )
        _add_frame(fig, np.eye(4), "base", axis_len=0.08)
        _add_frame(fig, T_base_camera, "camera_main", axis_len=0.08)
        fig.update_layout(
            title="Live calibration check: point cloud + TCP in base frame",
            scene={
                "xaxis": {"title": "base X (m)"},
                "yaxis": {"title": "base Y (m)"},
                "zaxis": {"title": "base Z (m)"},
                "aspectmode": "data",
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 50},
        )
    html_path = output_dir / "calibration_live_check_3d.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"[output] wrote {html_path}")

    markers: list[dict[str, Any]] = []
    if projected is not None:
        markers.append(
            {
                "pixel": projected,
                "color": (255, 0, 0),
                "label": "TCP projected",
            }
        )
    overlay = _annotate_overlay(
        rgb,
        title="Blue cross = current TCP projected with camera_pose_mats.main",
        markers=markers,
    )
    png_path = output_dir / "calibration_live_tcp_overlay.png"
    PILImage.fromarray(overlay).save(png_path)
    print(f"[output] wrote {png_path}")
    print(
        "[hint] If the blue TCP cross does not land on the real gripper tip in the image, "
        "calibration or tcp_offset needs adjustment."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Agilex camera calibration accuracy.")
    parser.add_argument(
        "--mode",
        choices=("samples", "live"),
        default="live",
        help="samples=offline from saved calibration samples; live=project current TCP now",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="Optional T_base_camera yaml/json; otherwise read camera_pose_mats.main from --config",
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=None,
        help="Point-correspondence or eye-to-hand samples.json (samples mode)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/agilex_calibration_viz"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--depth-range", nargs=2, type=float, default=(0.05, 2.0))
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--target-link", default=None)
    parser.add_argument("--tcp-offset", nargs=3, type=float, default=None)
    parser.add_argument(
        "--pose-source",
        choices=("auto", "pyagxarm", "ros_joints", "ros_end_pose"),
        default="auto",
        help="auto tries pyAgxArm, then falls back to ROS /puppet/joint_left",
    )
    parser.add_argument("--joint-topic", default=None)
    parser.add_argument("--end-pose-topic", default="/puppet/end_pose_left")
    parser.add_argument("--arm-channel", default="can_left")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--robot", default="piper")
    args = parser.parse_args()

    if args.mode == "samples":
        if args.samples_file is None:
            point_default = DEFAULT_POINT_OUTPUT_DIR / "samples.json"
            eye_default = DEFAULT_EYE_TO_HAND_OUTPUT_DIR / "samples.json"
            if point_default.exists():
                args.samples_file = point_default
            elif eye_default.exists():
                args.samples_file = eye_default
            else:
                raise SystemExit(
                    "samples mode requires --samples-file "
                    f"(checked {point_default} and {eye_default})"
                )
        run_samples_mode(args)
        return

    run_live_mode(args)


if __name__ == "__main__":
    main()
