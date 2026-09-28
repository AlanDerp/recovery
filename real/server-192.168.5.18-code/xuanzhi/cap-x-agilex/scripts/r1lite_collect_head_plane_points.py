#!/usr/bin/env python3
"""Collect head-camera tabletop calibration points for R1 Lite.

Workflow:
  1. capture  - save a head camera RGB image
  2. collect  - click 4 tabletop corners, then record arm EE pose for each
  3. solve    - compute homography (delegates to r1lite_calibrate_head_plane.py)

The calibration maps head-left image pixels [u, v] to base-frame plane coords
[x, y] on a flat work surface (case top / tabletop).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_IMAGE = REPO_ROOT / "outputs/r1lite/latest_head_left_rgb.png"
DEFAULT_POINTS = REPO_ROOT / "outputs/r1lite/head_plane_points.json"
DEFAULT_CALIB = REPO_ROOT / "outputs/r1lite/head_plane_calibration.json"

HEAD_TOPICS = {
    "left": "/hdas/camera_head/left_raw/image_raw_color/compressed",
    "right": "/hdas/camera_head/right_raw/image_raw_color/compressed",
}
HEAD_TOPIC_FALLBACKS = {
    "left": [],
    "right": [
        "/calib/head_right/image_raw",
    ],
}
# Table calibration must use the physical end-effector pose in base_link.
# Do not use /relaxed_ik/motion_control/pose_ee_arm_right here: it is a
# planning/solver pose and can be expressed in the current tool-frame convention.
EE_POSE_TOPICS = [
    "/motion_control/pose_ee_arm_right",
]
ARM_FEEDBACK_TOPIC = "/hdas/feedback_arm_right"
EE_TF_CANDIDATES = [
    ("base_link", "right_gripper_link"),
    ("base_link", "right_arm_link6"),
    ("base_link", "right_ee"),
]
TABLE_Z_SANITY_RANGE_M = (0.45, 1.35)


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _ensure_display() -> str:
    """Use the local X session when collect is launched over SSH/tmux."""
    if _has_display():
        return os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or ""
    if Path("/tmp/.X11-unix/X0").exists():
        os.environ["DISPLAY"] = ":0"
        print("[collect] DISPLAY was unset; using local monitor DISPLAY=:0")
        return ":0"
    return ""


def _display_help() -> str:
    return (
        "No GUI display is available for point picking.\n"
        "Fix options:\n"
        "  1) Run on the robot with a monitor attached:\n"
        "       export DISPLAY=:0\n"
        "       bash scripts/run_r1lite_head_calibration.sh collect\n"
        "  2) From SSH, forward X11 or set DISPLAY to the robot desktop:\n"
        "       ssh -X nvidia@<robot-ip>\n"
        "       export DISPLAY=:0\n"
        "  3) Skip the GUI and type pixel coordinates manually:\n"
        "       python3 scripts/r1lite_collect_head_plane_points.py collect \\\n"
        "         --pixels 960,1100 1200,900 1500,600 700,600\n"
        "     (open outputs/r1lite/latest_head_left_rgb.png to read u,v values)\n"
    )


def _setup_ros_env() -> None:
    os.environ.setdefault("ROS_DOMAIN_ID", "1")
    os.environ.setdefault(
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "/opt/galaxea/find_server/super_client_configuration_file.xml",
    )


def _compressed_to_rgb(data: bytes) -> np.ndarray:
    import io

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _resolve_head_topic(camera: str) -> str:
    import rclpy
    from rclpy.node import Node

    candidates = [HEAD_TOPICS[camera], *HEAD_TOPIC_FALLBACKS.get(camera, [])]
    _setup_ros_env()
    if not rclpy.ok():
        rclpy.init()
    node = Node("capx_r1lite_head_topic_resolve")
    try:
        deadline = time.time() + 8.0
        while time.time() < deadline:
            names = {name for name, _ in node.get_topic_names_and_types()}
            for topic in candidates:
                if topic in names:
                    return topic
            time.sleep(0.2)
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
    return HEAD_TOPICS[camera]


def _wait_for_head_rgb(topic: str, timeout_s: float = 15.0) -> np.ndarray:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image

    class _Capture(Node):
        def __init__(self) -> None:
            super().__init__("capx_r1lite_head_capture")
            self._rgb: np.ndarray | None = None
            if "compressed" in topic:
                self.create_subscription(
                    CompressedImage, topic, self._on_compressed, qos_profile_sensor_data
                )
            else:
                self.create_subscription(Image, topic, self._on_raw, qos_profile_sensor_data)

        def _on_compressed(self, msg: CompressedImage) -> None:
            self._rgb = _compressed_to_rgb(bytes(msg.data))

        def _on_raw(self, msg: Image) -> None:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            channels = max(1, int(msg.step) // max(1, int(msg.width)))
            rgb = arr.reshape(int(msg.height), int(msg.width), channels)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            encoding = str(msg.encoding).lower()
            if encoding in {"bgr8", "bgra8"}:
                rgb = rgb[..., ::-1]
            if rgb.shape[-1] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            self._rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    _setup_ros_env()
    if not rclpy.ok():
        rclpy.init()
    node = _Capture()
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node._rgb is not None:
                return node._rgb.copy()
        raise TimeoutError(f"Timed out waiting for head image on {topic}")
    finally:
        node.destroy_node()


def _wait_for_ee_pose(timeout_s: float = 30.0) -> tuple[np.ndarray, str]:
    """Return right-arm position [x,y,z] and the source used."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    class _PoseReader(Node):
        def __init__(self, topics: list[str]) -> None:
            super().__init__("capx_r1lite_ee_pose_reader")
            self._position: np.ndarray | None = None
            self._source = ""
            for topic in topics:
                self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, t=topic: self._on_pose(msg, t),
                    10,
                )

        def _on_pose(self, msg: PoseStamped, topic: str) -> None:
            if self._position is not None:
                return
            if str(msg.header.frame_id).strip() != "base_link":
                self.get_logger().warn(
                    f"Ignoring {topic} pose in frame {msg.header.frame_id!r}; "
                    "expected 'base_link'"
                )
                return
            p = msg.pose.position
            self._position = np.array([p.x, p.y, p.z], dtype=np.float64)
            self._source = topic

    def _try_tf(node: Node) -> tuple[np.ndarray, str] | None:
        try:
            from tf2_ros import Buffer, TransformListener
        except ImportError:
            return None
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        deadline = time.time() + min(timeout_s, 5.0)
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            for parent, child in EE_TF_CANDIDATES:
                try:
                    if not buffer.can_transform(parent, child, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.2)):
                        continue
                    tf = buffer.lookup_transform(parent, child, rclpy.time.Time())
                    p = tf.transform.translation
                    return np.array([p.x, p.y, p.z], dtype=np.float64), f"tf:{parent}->{child}"
                except Exception:
                    continue
        return None

    _setup_ros_env()
    if not rclpy.ok():
        rclpy.init()
    node = _PoseReader(EE_POSE_TOPICS)
    deadline = time.time() + timeout_s
    try:
        tf_result = _try_tf(node)
        if tf_result is not None:
            return tf_result
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node._position is not None:
                return node._position.copy(), node._source
        raise TimeoutError(_pose_timeout_help())
    finally:
        node.destroy_node()


def _pose_timeout_help() -> str:
    return (
        "Timed out waiting for right-arm EE pose.\n"
        "Tried pose topics:\n"
        + "\n".join(f"  - {t}" for t in EE_POSE_TOPICS)
        + "\n"
        "And TF frames:\n"
        + "\n".join(f"  - {a} -> {b}" for a, b in EE_TF_CANDIDATES)
        + "\n\n"
        "Please restart robot stack, then start EE pose publisher:\n"
        "  bash ~/start_r1_teleop_ready.sh\n"
        "  cd ~/galaxea/install/startup_config/share/startup_config/script/boot/modules/mobiman\n"
        "  ./start_r1lite_ee_pose.sh\n\n"
        "Then verify one pose message is available:\n"
        "  ros2 topic echo /motion_control/pose_ee_arm_right --once"
    )


def _probe_ee_pose(timeout_s: float = 8.0) -> tuple[bool, str]:
    try:
        _, source = _wait_for_ee_pose(timeout_s=timeout_s)
        return True, source
    except TimeoutError as exc:
        return False, str(exc)


def _discover_topics(required: list[str], wait_s: float = 8.0) -> dict[str, bool]:
    import rclpy
    from rclpy.node import Node

    _setup_ros_env()
    if not rclpy.ok():
        rclpy.init()
    node = Node("capx_r1lite_topic_check")
    try:
        deadline = time.time() + wait_s
        found = {topic: False for topic in required}
        while time.time() < deadline:
            names = {name for name, _ in node.get_topic_names_and_types()}
            for topic in required:
                if topic in names:
                    found[topic] = True
            if all(found.values()):
                break
            time.sleep(0.2)
            rclpy.spin_once(node, timeout_sec=0.05)
        return found
    finally:
        node.destroy_node()


def _check_ros_topics(required: list[str]) -> dict[str, bool]:
    return _discover_topics(required)


def cmd_check(_: argparse.Namespace) -> None:
    camera_topics = [
        HEAD_TOPICS["left"],
        HEAD_TOPICS["right"],
        *HEAD_TOPIC_FALLBACKS["right"],
    ]
    arm_topics = [
        *EE_POSE_TOPICS,
        ARM_FEEDBACK_TOPIC,
        "/hdas/feedback_gripper_right",
    ]
    status = _check_ros_topics(camera_topics + arm_topics)
    print("ROS topic check (ROS_DOMAIN_ID=%s):" % os.environ.get("ROS_DOMAIN_ID", "?"))
    has_camera = any(status[t] for t in camera_topics)
    has_arm_feedback = status[ARM_FEEDBACK_TOPIC]
    has_pose_topic = any(status[t] for t in EE_POSE_TOPICS)
    for topic in camera_topics + arm_topics:
        ok = status[topic]
        print(f"  [{'OK' if ok else 'MISSING'}] {topic}")
    if not has_camera:
        print("\nNo head camera topic found. Start the robot stack:")
        print("  bash ~/start_r1_teleop_ready.sh")
        raise SystemExit(1)
    if not has_arm_feedback and not has_pose_topic:
        print("\nCamera OK, but arm topics are missing.")
        print("You can run `capture` now. For `collect`, start the robot stack:")
        print("  bash ~/start_r1_teleop_ready.sh")
        raise SystemExit(2)
    print("\nProbing live EE pose stream ...")
    ok_pose, detail = _probe_ee_pose(timeout_s=8.0)
    if ok_pose:
        print(f"  [OK] EE pose available from {detail}")
        print("\nAll required streams are available.")
        return
    print("  [MISSING] No live EE pose yet.")
    print(detail)
    raise SystemExit(3)


def cmd_capture(args: argparse.Namespace) -> None:
    topic = _resolve_head_topic(args.camera)
    print(f"[capture] waiting for {topic} ...")
    rgb = _wait_for_head_rgb(topic, timeout_s=args.timeout)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)
    print(f"[capture] saved {out} ({rgb.shape[1]}x{rgb.shape[0]})")


def _parse_pixels_arg(values: list[str], num_points: int) -> list[list[int]]:
    pixels: list[list[int]] = []
    for raw in values:
        parts = raw.replace(";", ",").split(",")
        if len(parts) != 2:
            raise SystemExit(f"Invalid --pixels value {raw!r}; expected u,v")
        pixels.append([int(round(float(parts[0]))), int(round(float(parts[1])))])
    if len(pixels) != num_points:
        raise SystemExit(f"--pixels needs exactly {num_points} values, got {len(pixels)}")
    return pixels


def _pick_pixels(image_path: Path, num_points: int) -> list[list[int]]:
    display = _ensure_display()
    if not display:
        raise SystemExit(_display_help())

    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    img = np.asarray(Image.open(image_path).convert("RGB"))
    clicks: list[tuple[int, int]] = []

    def onclick(event: Any) -> None:
        if event.xdata is None or event.ydata is None:
            return
        u, v = int(round(event.xdata)), int(round(event.ydata))
        clicks.append((u, v))
        ax.plot(u, v, "o", color="lime", markersize=8)
        ax.text(u + 12, v + 12, f"P{len(clicks)}", color="lime", fontsize=12)
        fig.canvas.draw_idle()
        print(f"  clicked P{len(clicks)}: pixel=[{u}, {v}]")
        if len(clicks) >= num_points:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img)
    ax.set_title(
        "静态参考图：点击工作面 4 个角点（画面不会随机械臂移动更新）"
    )
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if len(clicks) < num_points:
        raise SystemExit(
            f"Need {num_points} pixel clicks, got {len(clicks)}. Re-run collect."
        )
    return [[u, v] for u, v in clicks]


def cmd_collect(args: argparse.Namespace) -> None:
    if not args.no_recapture:
        print("[collect] refreshing head camera image before point picking ...")
        cap_args = argparse.Namespace(
            camera=args.camera,
            output=args.image,
            timeout=args.timeout,
        )
        cmd_capture(cap_args)

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}. Run capture first.")

    print("=" * 60)
    print("Step 1/2: Pick pixel coordinates on the flat work surface")
    print("=" * 60)
    print(f"Image: {image_path}")
    print("注意: 下面弹出的是【静态参考图】，不会随机械臂移动而更新。")
    print("      头部相机固定在机器人头上，只动右臂时画面几乎不变，这是正常的。")
    print("      动机械臂是为了让夹爪去碰桌面角点，用来记录位姿，不是为了刷新图像。")
    print("In the image window, click 4 corners of the tabletop/case top.")
    print("Suggested order: near-left, near-right, far-right, far-left")
    if args.pixels:
        pixels = _parse_pixels_arg(args.pixels, args.num_points)
        print("Using manual pixel coordinates:")
        for i, pixel in enumerate(pixels, start=1):
            print(f"  P{i}: pixel={pixel}")
    else:
        pixels = _pick_pixels(image_path, args.num_points)

    print("\nChecking EE pose stream before arm pose collection ...")
    ok_pose, detail = _probe_ee_pose(timeout_s=10.0)
    if not ok_pose:
        print(detail)
        raise SystemExit(3)
    print(f"EE pose ready: {detail}")

    points: list[dict[str, Any]] = []
    z_values: list[float] = []

    print("\n" + "=" * 60)
    print("Step 2/2: Record arm poses for each corner")
    print("=" * 60)

    for idx, pixel in enumerate(pixels, start=1):
        print(f"\n--- Point {idx}/{len(pixels)}: pixel={pixel} ---")
        print(">>> 请移动右臂，让夹爪尖端轻触该角点对应的桌面位置 <<<")
        print("    （图像不会变化是正常的；这里只记录机械臂位姿）")
        print("    到位后按 Enter 记录位姿（输入 q 退出）")
        resp = input("    按 Enter 继续... ").strip().lower()
        if resp == "q":
            raise SystemExit("Collection cancelled.")

        pose, pose_source = _wait_for_ee_pose(timeout_s=args.timeout)
        base_xy = pose[:2].tolist()
        z_values.append(float(pose[2]))
        point = {
            "pixel": pixel,
            "base_xy": base_xy,
            "base_xyz": pose.tolist(),
            "pose_source": pose_source,
        }
        points.append(point)
        print(f"    recorded base_xy={base_xy}, z={pose[2]:.4f} (from {pose_source})")

    plane_z = float(np.mean(z_values))
    z_min, z_max = TABLE_Z_SANITY_RANGE_M
    if not (z_min <= plane_z <= z_max):
        raise SystemExit(
            "Refusing to write head-plane calibration points because the recorded "
            f"mean table z={plane_z:.4f}m is outside the sanity range "
            f"[{z_min:.2f}, {z_max:.2f}]m. This usually means the EE pose source "
            "is not the physical gripper pose in base_link."
        )
    payload = {
        "plane_z": plane_z,
        "points": [{"pixel": p["pixel"], "base_xy": p["base_xy"]} for p in points],
        "image_size": list(Image.open(image_path).size),
        "image_path": str(image_path),
        "camera": args.camera,
        "ee_pose_topics": EE_POSE_TOPICS,
        "raw_records": points,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[collect] wrote {out}")
    print(f"[collect] plane_z (mean EE z) = {plane_z:.4f}")


def cmd_solve(args: argparse.Namespace) -> None:
    points_json = Path(args.points_json)
    if not points_json.exists():
        raise SystemExit(f"Points file not found: {points_json}")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "r1lite_calibrate_head_plane.py"),
        "--points-json",
        str(points_json),
        "--output",
        str(args.output),
    ]
    image = args.image or json.loads(points_json.read_text()).get("image_path")
    if image and Path(image).exists():
        cmd.extend(["--image", str(image)])
    print("[solve]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[solve] calibration saved to {args.output}")


def cmd_verify(args: argparse.Namespace) -> None:
    calib_path = Path(args.calibration)
    if not calib_path.exists():
        raise SystemExit(f"Calibration not found: {calib_path}")

    data = json.loads(calib_path.read_text())
    h = np.asarray(data["image_to_base_xy_homography"], dtype=np.float64).reshape(3, 3)
    plane_z = float(data.get("plane_z", 0.0))
    points = data.get("points", [])

    print(f"Calibration: {calib_path}")
    print(f"plane_z = {plane_z:.4f}")
    print("Point reprojection errors (base_xy):")
    errors = []
    for i, pt in enumerate(points, start=1):
        u, v = pt["pixel"]
        q = h @ np.array([u, v, 1.0], dtype=np.float64)
        pred = (q[:2] / q[2]).tolist()
        gt = pt["base_xy"]
        err = float(np.linalg.norm(np.asarray(pred) - np.asarray(gt)))
        errors.append(err)
        print(
            f"  P{i}: gt={gt} pred=[{pred[0]:.4f}, {pred[1]:.4f}] err={err*1000:.1f}mm"
        )
    print(f"RMSE = {float(np.sqrt(np.mean(np.square(errors))))*1000:.1f} mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=15.0)
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check", help="Verify ROS topics are available")
    check_p.set_defaults(func=cmd_check)

    cap_p = sub.add_parser("capture", help="Save head camera RGB image")
    cap_p.add_argument("--camera", choices=("left", "right"), default="left")
    cap_p.add_argument("--output", default=str(DEFAULT_IMAGE))
    cap_p.set_defaults(func=cmd_capture)

    col_p = sub.add_parser("collect", help="Interactive pixel + arm pose collection")
    col_p.add_argument("--image", default=str(DEFAULT_IMAGE))
    col_p.add_argument("--output", default=str(DEFAULT_POINTS))
    col_p.add_argument("--camera", choices=("left", "right"), default="left")
    col_p.add_argument("--num-points", type=int, default=4)
    col_p.add_argument("--timeout", type=float, default=30.0)
    col_p.add_argument(
        "--no-recapture",
        action="store_true",
        help="Do not refresh the head image before point picking",
    )
    col_p.add_argument(
        "--pixels",
        nargs="+",
        default=None,
        metavar="U,V",
        help="Manual pixel coordinates (skip GUI), e.g. --pixels 960,1100 1200,900 ...",
    )
    col_p.set_defaults(func=cmd_collect)

    sol_p = sub.add_parser("solve", help="Compute homography from collected points")
    sol_p.add_argument("--points-json", default=str(DEFAULT_POINTS))
    sol_p.add_argument("--output", default=str(DEFAULT_CALIB))
    sol_p.add_argument("--image", default=None)
    sol_p.set_defaults(func=cmd_solve)

    ver_p = sub.add_parser("verify", help="Check calibration reprojection error")
    ver_p.add_argument("--calibration", default=str(DEFAULT_CALIB))
    ver_p.set_defaults(func=cmd_verify)

    return parser.parse_args()


def main() -> None:
    _setup_ros_env()
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
