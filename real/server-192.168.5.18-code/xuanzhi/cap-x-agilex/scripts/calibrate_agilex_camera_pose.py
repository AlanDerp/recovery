#!/usr/bin/env python3
"""Calibrate Agilex main RGB-D camera pose from 3D point correspondences.

This estimates T_base_camera, the 4x4 camera-to-robot-base transform used by
capx as camera_pose_mats.main.

Recommended workflow (matches visualize_agilex_camera_calibration.py live mode):
  uv run --no-sync --active scripts/calibrate_agilex_camera_pose.py collect-current-tcp \\
    --config env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml
  # left-click the same physical point (e.g. gripper tip), then press q
  # repeat for 6-10 arm poses, then:
  uv run --no-sync --active scripts/calibrate_agilex_camera_pose.py solve \\
    --samples-file outputs/agilex_point_calibration/samples.json \\
    --write-config env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml \\
    --reset-graspnet-offsets

Note: collect-end-pose does NOT take --config. It uses /puppet/end_pose_left, which
may disagree with the URDF FK used by live visualization.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from agilex_camera_calib_io import (
    DEFAULT_AGILEX_CONFIG,
    DEFAULT_POINT_OUTPUT_DIR,
    apply_calibration_to_config,
    base_point_from_joints,
    estimate_camera_to_base,
    load_point_correspondence_samples,
    print_yaml_matrix,
    read_current_arm_joints,
    read_joints_from_ros,
    resolve_arm_config,
    save_point_correspondence_samples,
    solve_point_correspondence_from_file,
)


def _normalize_rgb(image: np.ndarray, encoding: str | None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if encoding and encoding.lower() == "bgr8":
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


def _decode_depth_msg(depth_msg: Any, depth_scale: float) -> np.ndarray:
    encoding = getattr(depth_msg, "encoding", "") or ""
    data = np.frombuffer(depth_msg.data, dtype=np.uint8)
    if encoding in ("16UC1", "mono16"):
        arr = data.view(np.uint16).reshape(depth_msg.height, depth_msg.width)
        return _normalize_depth(arr, depth_scale)
    if encoding in ("32FC1",):
        arr = data.view(np.float32).reshape(depth_msg.height, depth_msg.width)
        return arr.astype(np.float64)

    try:
        from cv_bridge import CvBridge

        bridge = CvBridge()
        return _normalize_depth(
            bridge.imgmsg_to_cv2(depth_msg, "passthrough"),
            depth_scale,
        )
    except Exception:
        if data.size == depth_msg.height * depth_msg.width * 2:
            arr = data.view(np.uint16).reshape(depth_msg.height, depth_msg.width)
            return _normalize_depth(arr, depth_scale)
        raise


def _wait_for_rgbd(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    try:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError("rospy, cv_bridge, and sensor_msgs are required") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_camera_pose_calibration", anonymous=True, disable_signals=True)

    bridge = CvBridge()
    print(f"[ros] waiting rgb: {args.rgb_topic}")
    rgb_msg = rospy.wait_for_message(args.rgb_topic, Image, timeout=args.timeout)
    print(f"[ros] waiting depth: {args.depth_topic}")
    depth_msg = rospy.wait_for_message(args.depth_topic, Image, timeout=args.timeout)
    print(f"[ros] waiting camera_info: {args.camera_info_topic}")
    info_msg = rospy.wait_for_message(args.camera_info_topic, CameraInfo, timeout=args.timeout)

    try:
        rgb = _normalize_rgb(
            bridge.imgmsg_to_cv2(rgb_msg, "passthrough"),
            getattr(rgb_msg, "encoding", None),
        )
    except Exception:
        rgb = _normalize_rgb(
            np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
                rgb_msg.height, rgb_msg.width, -1
            ),
            getattr(rgb_msg, "encoding", None),
        )
    depth = _decode_depth_msg(depth_msg, args.depth_scale)
    intrinsics = np.asarray(info_msg.K, dtype=np.float64).reshape(3, 3)
    if depth.shape[:2] != (int(info_msg.height), int(info_msg.width)):
        from capx.utils.camera_utils import scale_intrinsics_for_image

        intrinsics = scale_intrinsics_for_image(
            intrinsics,
            depth.shape[:2],
            (int(info_msg.height), int(info_msg.width)),
        )
        print(
            f"[intrinsics] scaled K for depth {depth.shape[:2]} "
            f"from camera_info {(info_msg.height, info_msg.width)}"
        )
    return rgb, depth, intrinsics, rgb_msg.header.stamp.to_sec()


def _prompt_pixel_from_terminal(
    output_path: Path | None,
    width: int,
    height: int,
) -> tuple[int, int]:
    if output_path is not None:
        abs_path = output_path.resolve()
        print("[click] OpenCV GUI is unavailable in this environment (opencv-python-headless).")
        print(
            f"[click] Open the saved image, move the mouse to the gripper tip, "
            f"and read pixel coordinates (u=x column, v=y row; image size {width}x{height}):"
        )
        print(f"        {abs_path}")
        try:
            import subprocess

            subprocess.Popen(
                ["xdg-open", str(abs_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass
    else:
        print(
            f"[click] OpenCV GUI is unavailable; enter gripper-tip pixel "
            f"(u=x column, v=y row; image size {width}x{height})."
        )

    while True:
        raw = input("[click] Enter 'u v' for the gripper tip (or 'q' to abort): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise RuntimeError("Calibration click aborted")
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("[click] Please enter two integers, for example: 360 250")
            continue
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:
            print("[click] Invalid integers")
            continue
        if not (0 <= u < width and 0 <= v < height):
            print(f"[click] Out of range; image is {width}x{height}")
            continue
        return u, v


def _click_pixel(rgb: np.ndarray, output_path: Path | None = None) -> tuple[int, int]:
    if output_path is not None:
        from PIL import Image

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(output_path)
        print(f"[image] saved click reference image: {output_path}")

    height, width = rgb.shape[:2]
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for interactive clicking; use --pixel U V") from exc

    clicked: list[tuple[int, int]] = []
    display = rgb[:, :, ::-1].copy()

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((int(x), int(y)))
            marker = display.copy()
            cv2.drawMarker(marker, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.imshow("click calibration point, then press q", marker)

    try:
        cv2.namedWindow("click calibration point, then press q", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("click calibration point, then press q", on_mouse)
        cv2.imshow("click calibration point, then press q", display)
        print("[click] left-click the calibration point, then press q")
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 13, 27) and clicked:
                break
        cv2.destroyAllWindows()
        return clicked[0]
    except cv2.error as exc:
        if "not implemented" not in str(exc).lower():
            raise
        return _prompt_pixel_from_terminal(output_path, width, height)


def _map_rgb_pixel_to_depth_pixel(
    u_rgb: int,
    v_rgb: int,
    rgb_shape: tuple[int, ...],
    depth_shape: tuple[int, ...],
) -> tuple[int, int]:
    rgb_h, rgb_w = int(rgb_shape[0]), int(rgb_shape[1])
    depth_h, depth_w = int(depth_shape[0]), int(depth_shape[1])
    u = int(round(float(u_rgb) * depth_w / rgb_w))
    v = int(round(float(v_rgb) * depth_h / rgb_h))
    return max(0, min(depth_w - 1, u)), max(0, min(depth_h - 1, v))


def _depth_at(depth: np.ndarray, u: int, v: int, radius: int) -> float:
    h, w = depth.shape
    u0 = max(0, u - radius)
    u1 = min(w, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(h, v + radius + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise RuntimeError(
            f"No valid depth near pixel ({u}, {v}) with radius {radius}. "
            "Matte-black gripper tips often have no depth. Options: "
            "(1) lower the arm slightly and retry, "
            "(2) use --depth-m <meters> from a nearby valid reading, "
            "(3) change pose. Avoid bare metal and image edges."
        )
    return float(np.median(valid))


def _pixel_to_camera_point(u: int, v: int, z: float, intrinsics: np.ndarray) -> np.ndarray:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def _wait_for_end_pose(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
    except ImportError as exc:
        raise RuntimeError("rospy and geometry_msgs are required for collect-end-pose") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_camera_pose_calibration", anonymous=True, disable_signals=True)

    print(f"[ros] waiting end pose: {args.end_pose_topic}")
    msg = rospy.wait_for_message(args.end_pose_topic, PoseStamped, timeout=args.timeout)
    position = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
        dtype=np.float64,
    )
    quaternion_xyzw = np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    stamp = msg.header.stamp.to_sec()
    return position, quaternion_xyzw, stamp


def _base_point_from_pose(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    marker_offset: np.ndarray,
) -> np.ndarray:
    quat = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise RuntimeError("End pose orientation quaternion is zero")
    rot = Rotation.from_quat(quat / norm)
    return np.asarray(position, dtype=np.float64).reshape(3) + rot.apply(marker_offset)


def _append_sample(
    *,
    args: argparse.Namespace,
    base_point: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    rgb, depth, intrinsics, stamp = _wait_for_rgbd(args)
    if args.pixel is None:
        image_path = Path(args.output_image) if args.output_image else None
        u_rgb, v_rgb = _click_pixel(rgb, output_path=image_path)
    else:
        u_rgb, v_rgb = int(args.pixel[0]), int(args.pixel[1])

    u_depth, v_depth = _map_rgb_pixel_to_depth_pixel(u_rgb, v_rgb, rgb.shape, depth.shape)
    if rgb.shape[:2] != depth.shape[:2]:
        print(
            f"[pixel] mapped rgb ({u_rgb}, {v_rgb}) {rgb.shape[1]}x{rgb.shape[0]} "
            f"-> depth ({u_depth}, {v_depth}) {depth.shape[1]}x{depth.shape[0]}"
        )

    if args.depth_m is not None:
        z = float(args.depth_m)
        print(f"[depth] using manual depth_m={z:.4f} at pixel_depth=({u_depth}, {v_depth})")
    else:
        z = _depth_at(depth, u_depth, v_depth, args.depth_radius)
    camera_point = _pixel_to_camera_point(u_depth, v_depth, z, intrinsics)
    base_point = np.asarray(base_point, dtype=np.float64).reshape(3)

    sample = {
        "time": time.time(),
        "stamp": stamp,
        "pixel": [int(u_rgb), int(v_rgb)],
        "depth_pixel": [int(u_depth), int(v_depth)],
        "depth_m": float(z),
        "camera_point": camera_point.tolist(),
        "base_point": base_point.tolist(),
    }
    if metadata:
        sample.update(metadata)
    path = Path(args.samples_file)
    samples = load_point_correspondence_samples(path)
    samples.append(sample)
    save_point_correspondence_samples(path, samples)

    print(f"[sample] appended #{len(samples)} to {path}")
    print(
        f"[sample] pixel_rgb=({u_rgb}, {v_rgb}) pixel_depth=({u_depth}, {v_depth}) "
        f"depth={z:.4f} camera={camera_point} base={base_point}"
    )
    if len(samples) >= 4:
        transform, rms, errors = estimate_camera_to_base(samples)
        print(f"[solve] samples={len(samples)} rms_error_m={rms:.5f}")
        print(f"[solve] per_sample_error_m={np.round(errors, 5).tolist()}")
        print_yaml_matrix(transform)


def collect(args: argparse.Namespace) -> None:
    base_point = np.asarray(args.base_point, dtype=np.float64).reshape(3)
    _append_sample(args=args, base_point=base_point)


def collect_current_tcp(args: argparse.Namespace) -> None:
    urdf_path, target_link_name, tcp_offset = resolve_arm_config(
        config_path=Path(args.config) if args.config else None,
        urdf=args.urdf,
        target_link=args.target_link,
        tcp_offset=args.tcp_offset,
    )

    pose_source = "pyagxarm"
    try:
        joints = read_current_arm_joints(
            arm_channel=args.arm_channel,
            bitrate=args.bitrate,
            robot=args.robot,
        )
    except RuntimeError as exc:
        if "pyAgxArm" not in str(exc):
            raise
        joint_topic = args.joint_topic or "/puppet/joint_left"
        print(f"[pose] pyAgxArm not installed; falling back to ros_joints ({joint_topic})")
        joints = read_joints_from_ros(topic=joint_topic, timeout=args.timeout)
        pose_source = "ros_joints"

    base_point = base_point_from_joints(
        urdf_path=urdf_path,
        target_link_name=target_link_name,
        joints=joints,
        tcp_offset=tcp_offset,
    )
    print(f"[arm] pose_source={pose_source}")
    print(f"[arm] joints={np.array2string(joints, precision=5)}")
    print(f"[arm] base_point({target_link_name}+tcp_offset)={np.array2string(base_point, precision=5)}")
    _append_sample(
        args=args,
        base_point=base_point,
        metadata={
            "base_point_source": pose_source,
            "arm_joints": joints.tolist(),
            "target_link": target_link_name,
            "tcp_offset": tcp_offset.tolist(),
            "joint_topic": args.joint_topic,
        },
    )


def collect_end_pose(args: argparse.Namespace) -> None:
    marker_offset = (
        np.zeros(3, dtype=np.float64)
        if args.marker_offset is None
        else np.asarray(args.marker_offset, dtype=np.float64).reshape(3)
    )
    position, quaternion_xyzw, end_pose_stamp = _wait_for_end_pose(args)
    base_point = _base_point_from_pose(position, quaternion_xyzw, marker_offset)
    print(f"[pose] position={np.array2string(position, precision=6)}")
    print(f"[pose] orientation_xyzw={np.array2string(quaternion_xyzw, precision=6)}")
    print(f"[pose] marker_offset={np.array2string(marker_offset, precision=5)}")
    print(f"[pose] base_point={np.array2string(base_point, precision=6)}")
    _append_sample(
        args=args,
        base_point=base_point,
        metadata={
            "base_point_source": "ros_end_pose",
            "end_pose_topic": args.end_pose_topic,
            "end_pose_stamp": end_pose_stamp,
            "end_pose_position": position.tolist(),
            "end_pose_orientation_xyzw": quaternion_xyzw.tolist(),
            "marker_offset": marker_offset.tolist(),
        },
    )


def solve(args: argparse.Namespace) -> None:
    transform, rms, errors = solve_point_correspondence_from_file(Path(args.samples_file))
    print(f"[solve] samples file={args.samples_file}")
    print(f"[solve] rms_error_m={rms:.5f}")
    print(f"[solve] per_sample_error_m={np.round(errors, 5).tolist()}")
    print("[solve] T_base_camera / camera_pose_mats.main:")
    print(transform)
    print_yaml_matrix(transform)
    if args.write_config:
        apply_calibration_to_config(
            Path(args.write_config),
            transform,
            reset_graspnet_offsets=bool(args.reset_graspnet_offsets),
        )


def _add_common_collect_args(parser: argparse.ArgumentParser) -> None:
    default_samples = str(DEFAULT_POINT_OUTPUT_DIR / "samples.json")
    default_image = str(DEFAULT_POINT_OUTPUT_DIR / "latest_rgb.png")
    parser.add_argument("--samples-file", default=default_samples)
    parser.add_argument("--rgb-topic", default="/camera_h/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera_h/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera_h/color/camera_info")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--depth-radius", type=int, default=2)
    parser.add_argument(
        "--depth-m",
        type=float,
        default=None,
        help="Manual depth in meters when the gripper tip has no valid depth (black metal).",
    )
    parser.add_argument("--pixel", nargs=2, type=int, default=None)
    parser.add_argument("--output-image", default=default_image)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    collect_parser = subparsers.add_parser("collect")
    _add_common_collect_args(collect_parser)
    collect_parser.add_argument("--base-point", nargs=3, type=float, required=True)
    collect_parser.set_defaults(func=collect)

    current_parser = subparsers.add_parser("collect-current-tcp")
    _add_common_collect_args(current_parser)
    current_parser.add_argument("--config", default=str(DEFAULT_AGILEX_CONFIG))
    current_parser.add_argument("--urdf", default=None)
    current_parser.add_argument("--target-link", default=None)
    current_parser.add_argument("--tcp-offset", nargs=3, type=float, default=None)
    current_parser.add_argument("--arm-channel", default="can_left")
    current_parser.add_argument("--bitrate", type=int, default=1000000)
    current_parser.add_argument("--robot", default="piper")
    current_parser.add_argument(
        "--joint-topic",
        default="/puppet/joint_left",
        help="Fallback ROS JointState topic when pyAgxArm is unavailable.",
    )
    current_parser.set_defaults(func=collect_current_tcp)

    pose_parser = subparsers.add_parser(
        "collect-end-pose",
        help="Use /puppet/end_pose_left. Prefer collect-current-tcp to match live viz FK.",
    )
    _add_common_collect_args(pose_parser)
    pose_parser.add_argument("--end-pose-topic", default="/puppet/end_pose_left")
    pose_parser.add_argument(
        "--marker-offset",
        nargs=3,
        type=float,
        default=None,
        help="Marker offset in the end-pose frame, meters. Default is the end-pose origin.",
    )
    pose_parser.set_defaults(func=collect_end_pose)

    solve_parser = subparsers.add_parser("solve")
    solve_parser.add_argument("--samples-file", default=str(DEFAULT_POINT_OUTPUT_DIR / "samples.json"))
    solve_parser.add_argument("--write-config", default=None)
    solve_parser.add_argument(
        "--reset-graspnet-offsets",
        action="store_true",
        help="Zero graspnet_world_offset and graspnet_rotation_correction_rpy in the target config.",
    )
    solve_parser.set_defaults(func=solve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
