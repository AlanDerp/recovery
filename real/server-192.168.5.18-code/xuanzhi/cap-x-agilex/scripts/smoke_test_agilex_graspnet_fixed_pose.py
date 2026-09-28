#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str)
    with io.BytesIO(data) as f:
        return np.load(f)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _quat_wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            quat = np.array(
                [
                    (r[2, 1] - r[1, 2]) / s,
                    0.25 * s,
                    (r[0, 1] + r[1, 0]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            quat = np.array(
                [
                    (r[0, 2] - r[2, 0]) / s,
                    (r[0, 1] + r[1, 0]) / s,
                    0.25 * s,
                    (r[1, 2] + r[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            quat = np.array(
                [
                    (r[1, 0] - r[0, 1]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                    (r[1, 2] + r[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _transform_graspnet_pose(raw_grasp: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Apply the same local gripper-frame conversion used by Franka point-cloud GraspNet."""
    raw = np.asarray(raw_grasp, dtype=np.float64).reshape(4, 4)
    correction = _rotation_matrix_from_rpy(args.graspnet_rotation_correction_rpy)
    offset = np.asarray(args.grasp_frame_offset, dtype=np.float64).reshape(3)

    transformed = raw.copy()
    transformed[:3, 3] = raw[:3, 3] + raw[:3, :3] @ offset
    transformed[:3, :3] = raw[:3, :3] @ correction
    return transformed


def _axis_vector(rot: np.ndarray, axis_name: str) -> np.ndarray:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    name = axis_name[1:] if axis_name.startswith("-") else axis_name
    axis_idx = {"x": 0, "y": 1, "z": 2}[name]
    return sign * np.asarray(rot, dtype=np.float64).reshape(3, 3)[:, axis_idx]


def _select_grasp_index(
    grasps: np.ndarray,
    scores: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores_arr) != len(grasps):
        scores_arr = np.zeros(len(grasps), dtype=np.float64)

    order = np.argsort(-scores_arr)
    if not args.prefer_top_down:
        idx = int(order[0])
        return idx, {
            "selection": "best_score",
            "top_down_alignment": None,
            "top_down_threshold": args.top_down_threshold,
        }

    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    best_fallback_idx = int(order[0])
    best_fallback_alignment = -np.inf

    for idx_np in order:
        idx = int(idx_np)
        transformed = _transform_graspnet_pose(grasps[idx], args)
        approach = _axis_vector(transformed[:3, :3], args.approach_axis)
        norm = np.linalg.norm(approach)
        if norm < 1e-9:
            continue
        approach = approach / norm
        alignment = float(-np.dot(approach, world_z))
        if alignment > best_fallback_alignment:
            best_fallback_alignment = alignment
            best_fallback_idx = idx
        if alignment >= args.top_down_threshold:
            return idx, {
                "selection": "top_down_best_score",
                "top_down_alignment": alignment,
                "top_down_threshold": args.top_down_threshold,
                "approach_axis": args.approach_axis,
            }

    return best_fallback_idx, {
        "selection": "fallback_best_alignment",
        "top_down_alignment": best_fallback_alignment,
        "top_down_threshold": args.top_down_threshold,
        "approach_axis": args.approach_axis,
    }


def _make_cup_scene(
    center: np.ndarray,
    radius: float,
    height: float,
    object_points: int,
    table_points: int,
    table_z: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    n_side = max(1, int(object_points * 0.70))
    n_rim = max(1, int(object_points * 0.18))
    n_bottom = max(1, object_points - n_side - n_rim)

    theta = rng.uniform(0.0, 2.0 * np.pi, n_side)
    z_side = rng.uniform(-height / 2.0, height / 2.0, n_side)
    r_side = radius + rng.normal(0.0, radius * 0.03, n_side)
    side = np.column_stack(
        [
            center[0] + r_side * np.cos(theta),
            center[1] + r_side * np.sin(theta),
            center[2] + z_side,
        ]
    )

    theta = rng.uniform(0.0, 2.0 * np.pi, n_rim)
    r_rim = radius + rng.normal(0.0, radius * 0.04, n_rim)
    z_rim = center[2] + height / 2.0 + rng.normal(0.0, height * 0.01, n_rim)
    rim = np.column_stack(
        [
            center[0] + r_rim * np.cos(theta),
            center[1] + r_rim * np.sin(theta),
            z_rim,
        ]
    )

    theta = rng.uniform(0.0, 2.0 * np.pi, n_bottom)
    r_bottom = radius * np.sqrt(rng.uniform(0.0, 1.0, n_bottom))
    z_bottom = center[2] - height / 2.0 + rng.normal(0.0, height * 0.01, n_bottom)
    bottom = np.column_stack(
        [
            center[0] + r_bottom * np.cos(theta),
            center[1] + r_bottom * np.sin(theta),
            z_bottom,
        ]
    )

    pc_segment = np.vstack([side, rim, bottom]).astype(np.float32)

    table_extent = max(0.35, radius * 6.0)
    table_xy = rng.uniform(-table_extent, table_extent, size=(table_points, 2))
    table = np.column_stack(
        [
            center[0] + table_xy[:, 0],
            center[1] + table_xy[:, 1],
            np.full(table_points, table_z, dtype=np.float64),
        ]
    )
    pc_full = np.vstack([pc_segment, table.astype(np.float32)]).astype(np.float32)
    return pc_full, pc_segment


def _service_ready(service_url: str) -> bool:
    try:
        resp = requests.get(f"{service_url.rstrip('/')}/docs", timeout=2.0)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _wait_for_service(
    service_url: str,
    timeout_s: float,
    proc: subprocess.Popen[str] | None,
    log_path: Path | None,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _service_ready(service_url):
            return
        if proc is not None and proc.poll() is not None:
            details = ""
            if log_path is not None and log_path.exists():
                details = "\nLast server log lines:\n" + "".join(log_path.read_text().splitlines(True)[-40:])
            raise RuntimeError(f"GraspNet server exited before becoming ready.{details}")
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for GraspNet service at {service_url}")


def _start_server(args: argparse.Namespace, output_dir: Path) -> subprocess.Popen[str] | None:
    if _service_ready(args.service_url):
        print(f"GraspNet service already ready at {args.service_url}")
        return None

    parsed = urlparse(args.service_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8115
    log_path = output_dir / "graspnet_server.log"
    log_file = log_path.open("w")
    cmd = [
        sys.executable,
        "-m",
        "capx.serving.launch_contact_graspnet_server",
        "--device",
        args.device,
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"Starting GraspNet service: {' '.join(cmd)}")
    print(f"Server log: {log_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _wait_for_service(args.service_url, args.startup_timeout, proc, log_path)
    return proc


def _call_plan_point_clouds(
    service_url: str,
    pc_full: np.ndarray,
    pc_segment: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = {
        "pc_full_base64": _numpy_to_base64(pc_full),
        "pc_segment_base64": _numpy_to_base64(pc_segment),
        "segmap_id": 1,
        "local_regions": args.local_regions,
        "filter_grasps": args.filter_grasps,
        "forward_passes": args.forward_passes,
        "max_retries": args.max_retries,
    }
    resp = requests.post(
        f"{service_url.rstrip('/')}/plan_point_clouds",
        json=payload,
        timeout=args.request_timeout,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"GraspNet API failed: {resp.status_code} {resp.text}") from exc
    data = resp.json()
    return (
        _base64_to_numpy(data["grasps_base64"]),
        _base64_to_numpy(data["scores_base64"]),
        _base64_to_numpy(data["contact_pts_base64"]),
    )


def _load_low_level_cfg(config_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    low_level = cfg.env.cfg.low_level
    low_level_cfg = dict(OmegaConf.to_container(low_level, resolve=True))
    low_level_cfg.pop("_target_", None)
    return low_level_cfg


def _make_motion_pose(
    motion_grasp: np.ndarray,
    object_center: np.ndarray,
    args: argparse.Namespace,
    low_level_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.motion_position_source == "cup-center":
        position = np.asarray(object_center, dtype=np.float64).reshape(3).copy()
        position[2] += args.cup_center_z_offset
    else:
        position = motion_grasp[:3, 3].copy()

    if args.motion_quaternion is not None:
        quaternion = np.asarray(args.motion_quaternion, dtype=np.float64).reshape(4)
    elif args.motion_orientation == "config":
        quaternion = np.asarray(
            low_level_cfg.get("default_grasp_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(4)
    else:
        quaternion = _quat_wxyz_from_matrix(motion_grasp[:3, :3])

    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("Motion quaternion must be non-zero")
    quaternion = quaternion / norm

    metadata = {
        "motion_position_source": args.motion_position_source,
        "motion_orientation": args.motion_orientation,
        "grasp_frame_offset": np.asarray(args.grasp_frame_offset, dtype=np.float64),
        "graspnet_rotation_correction_rpy": np.asarray(
            args.graspnet_rotation_correction_rpy,
            dtype=np.float64,
        ),
        "cup_center_z_offset": args.cup_center_z_offset,
    }
    return position, quaternion, metadata


def _execute_interpolated(
    *,
    env: Any,
    target_joints: np.ndarray,
    gripper_width: float,
    steps: int,
    label: str,
    sleep_s: float,
) -> None:
    if env.hardware is None:
        raise RuntimeError("Hardware did not initialize; check direct Agilex control settings.")

    start_joints, start_width = env.hardware.get_state()
    start_joints = np.asarray(start_joints, dtype=np.float64).reshape(6)
    target_joints = env._clip_joint_targets(np.asarray(target_joints, dtype=np.float64))
    start_width = float(np.clip(start_width, 0.0, env.gripper_max_width))
    target_width = float(np.clip(gripper_width, 0.0, env.gripper_max_width))
    steps = max(1, int(steps))

    print(f"\nExecuting {label}")
    print("  start joints:", np.array2string(start_joints, precision=5))
    print("  target joints:", np.array2string(target_joints, precision=5))
    print(f"  start gripper width: {start_width:.4f} m")
    print(f"  target gripper width: {target_width:.4f} m")
    print(f"  interpolation steps: {steps}")

    for idx in range(1, steps + 1):
        alpha = idx / steps
        joints = start_joints + (target_joints - start_joints) * alpha
        width = start_width + (target_width - start_width) * alpha
        env.hardware.move(joints=joints, gripper_width=width)
        if idx == 1 or idx == steps or idx % 10 == 0:
            print(f"    step {idx:03d}/{steps}")
        if sleep_s > 0:
            time.sleep(sleep_s)


def _execute_motion(
    *,
    motion_grasp: np.ndarray,
    object_center: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from capx.envs.simulators.agilex_real import AgilexRealLowLevel
    from capx.integrations.agilex.control import AgilexControlApi

    low_level_cfg = _load_low_level_cfg(Path(args.config))
    low_level_cfg.update(
        {
            "control_backend": "direct",
            "camera_backend": "none",
            "wait_for_observation": False,
            "start_server": False,
            "reset_on_env_reset": False,
        }
    )
    env = AgilexRealLowLevel(**low_level_cfg)
    if env.hardware is None:
        raise RuntimeError("Hardware did not initialize; direct Agilex control is unavailable.")

    if args.reset_before_motion:
        print("\nResetting arm with configured interpolated reset path...")
        env.reset_on_env_reset = True
        env._reset_hardware_to_home()
        env.reset_on_env_reset = False

    api = AgilexControlApi(env, initialize_motion=True, debug=True)
    grasp_pos, grasp_quat, motion_metadata = _make_motion_pose(
        motion_grasp,
        object_center,
        args,
        low_level_cfg,
    )
    pregrasp_pos = grasp_pos + np.array([0.0, 0.0, args.pregrasp_height], dtype=np.float64)
    lift_pos = grasp_pos + np.array([0.0, 0.0, args.lift_height], dtype=np.float64)

    targets = [("pregrasp", pregrasp_pos, args.open_width)]
    if args.move_to in {"grasp", "lift", "full"}:
        targets.append(("grasp", grasp_pos, args.open_width))
    if args.move_to in {"lift", "full"}:
        targets.append(("lift", lift_pos, args.close_width))

    solved: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    print("\nSolving IK for planned hardware motion before executing...")
    for label, position, width in targets:
        joints = api.solve_ik(position, grasp_quat)
        solved.append((label, position, joints, width))
        print(f"  {label} position:", np.array2string(position, precision=5))
        print(f"  {label} joints:", np.array2string(joints, precision=5))

    if args.dry_run_motion:
        print("\n--dry-run-motion set; IK passed but hardware motion was skipped.")
        return {
            **motion_metadata,
            "grasp_position_m": grasp_pos,
            "grasp_quaternion_wxyz": grasp_quat,
            "pregrasp_position_m": pregrasp_pos,
            "lift_position_m": lift_pos,
            "executed": False,
        }

    for label, _position, joints, width in solved:
        _execute_interpolated(
            env=env,
            target_joints=joints,
            gripper_width=width,
            steps=args.steps_per_waypoint,
            label=label,
            sleep_s=args.command_sleep,
        )
        if args.pause_between_waypoints > 0:
            time.sleep(args.pause_between_waypoints)

        if label == "grasp" and args.move_to in {"lift", "full"}:
            current_joints, _ = env.hardware.get_state()
            _execute_interpolated(
                env=env,
                target_joints=current_joints,
                gripper_width=args.close_width,
                steps=args.gripper_steps,
                label="close gripper",
                sleep_s=args.command_sleep,
            )
            if args.pause_between_waypoints > 0:
                time.sleep(args.pause_between_waypoints)

    final_joints, final_width = env.hardware.get_state()
    print("\nFinal observed joints:", np.array2string(final_joints, precision=5))
    print(f"Final observed gripper width: {final_width:.4f} m")

    return {
        **motion_metadata,
        "grasp_position_m": grasp_pos,
        "grasp_quaternion_wxyz": grasp_quat,
        "pregrasp_position_m": pregrasp_pos,
        "lift_position_m": lift_pos,
        "final_joints": np.asarray(final_joints, dtype=np.float64),
        "final_gripper_width": float(final_width),
        "executed": True,
    }


def _save_plot(
    output_path: Path,
    pc_full: np.ndarray,
    pc_segment: np.ndarray,
    top_grasp: np.ndarray | None,
    object_center: np.ndarray,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot because matplotlib is unavailable: {exc}")
        return

    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    full_idx = rng.choice(len(pc_full), size=min(2500, len(pc_full)), replace=False)
    seg_idx = rng.choice(len(pc_segment), size=min(2500, len(pc_segment)), replace=False)
    ax.scatter(
        pc_full[full_idx, 0],
        pc_full[full_idx, 1],
        pc_full[full_idx, 2],
        s=1,
        c="#808080",
        alpha=0.18,
        label="scene",
    )
    ax.scatter(
        pc_segment[seg_idx, 0],
        pc_segment[seg_idx, 1],
        pc_segment[seg_idx, 2],
        s=4,
        c="#ff5aa5",
        alpha=0.75,
        label="cup segment",
    )
    ax.scatter(
        [object_center[0]],
        [object_center[1]],
        [object_center[2]],
        s=40,
        c="black",
        label="object center",
    )

    if top_grasp is not None:
        origin = top_grasp[:3, 3]
        rot = top_grasp[:3, :3]
        axis_len = 0.06
        colors = ["red", "green", "blue"]
        labels = ["grasp x", "grasp y", "grasp z"]
        for axis, color, label in zip(rot.T, colors, labels):
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                axis[0],
                axis[1],
                axis[2],
                length=axis_len,
                color=color,
                label=label,
            )

    mins = np.min(pc_full[:, :3], axis=0)
    maxs = np.max(pc_full[:, :3], axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_xlabel("x base/world (m)")
    ax.set_ylabel("y base/world (m)")
    ax.set_zlabel("z base/world (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test Contact-GraspNet /plan_point_clouds using a synthetic cup "
            "at a fixed robot base/world coordinate. This does not move hardware."
        )
    )
    parser.add_argument("--service-url", default="http://127.0.0.1:8115")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--position", type=float, nargs=3, default=[0.30, -0.40, 0.10])
    parser.add_argument("--position-is", choices=["center", "bottom"], default="center")
    parser.add_argument("--cup-radius", type=float, default=0.035)
    parser.add_argument("--cup-height", type=float, default=0.10)
    parser.add_argument("--object-points", type=int, default=4096)
    parser.add_argument("--table-points", type=int, default=4096)
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-passes", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=7)
    parser.add_argument("--local-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-grasps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--prefer-top-down",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer the highest-scoring grasp whose approach axis points downward.",
    )
    parser.add_argument("--top-down-threshold", type=float, default=0.65)
    parser.add_argument(
        "--approach-axis",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        default="z",
        help="Grasp frame axis treated as the approach direction for top-down filtering.",
    )
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect to the real Agilex arm and execute the planned motion.",
    )
    parser.add_argument(
        "--config",
        default="env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml",
        help="Agilex config used for direct hardware settings and IK parameters.",
    )
    parser.add_argument(
        "--move-to",
        choices=["pregrasp", "grasp", "lift", "full"],
        default="pregrasp",
        help="How far to execute after GraspNet succeeds.",
    )
    parser.add_argument(
        "--motion-position-source",
        choices=["graspnet", "cup-center"],
        default="graspnet",
        help="Use Contact-GraspNet pose position or the fixed cup coordinate.",
    )
    parser.add_argument(
        "--motion-orientation",
        choices=["graspnet", "config"],
        default="config",
        help="Use Contact-GraspNet orientation or config default_grasp_quaternion_wxyz.",
    )
    parser.add_argument("--motion-quaternion", type=float, nargs=4, default=None)
    parser.add_argument("--grasp-frame-offset", type=float, nargs=3, default=[0.0, 0.0, 0.12])
    parser.add_argument(
        "--graspnet-rotation-correction-rpy",
        type=float,
        nargs=3,
        default=[0.0, 0.0, math.pi / 2.0],
        help="Local RPY correction applied after GraspNet, matching Franka's point-cloud path.",
    )
    parser.add_argument("--cup-center-z-offset", type=float, default=0.03)
    parser.add_argument("--pregrasp-height", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--open-width", type=float, default=0.07)
    parser.add_argument("--close-width", type=float, default=0.0)
    parser.add_argument("--steps-per-waypoint", type=int, default=50)
    parser.add_argument("--gripper-steps", type=int, default=25)
    parser.add_argument("--command-sleep", type=float, default=0.01)
    parser.add_argument("--pause-between-waypoints", type=float, default=0.2)
    parser.add_argument("--reset-before-motion", action="store_true")
    parser.add_argument(
        "--dry-run-motion",
        action="store_true",
        help="Solve IK and print planned motion, but do not send hardware commands.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/agilex_graspnet_fixed_pose_smoke",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    object_position = np.asarray(args.position, dtype=np.float32)
    if args.position_is == "bottom":
        object_center = object_position.copy()
        object_center[2] += args.cup_height / 2.0
    else:
        object_center = object_position

    pc_full, pc_segment = _make_cup_scene(
        center=object_center,
        radius=args.cup_radius,
        height=args.cup_height,
        object_points=args.object_points,
        table_points=args.table_points,
        table_z=args.table_z,
        seed=args.seed,
    )

    proc: subprocess.Popen[str] | None = None
    try:
        if args.start_server:
            proc = _start_server(args, output_dir)
        else:
            _wait_for_service(args.service_url, timeout_s=3.0, proc=None, log_path=None)

        print(f"Testing GraspNet service: {args.service_url}")
        print(f"Cup {args.position_is} position: {object_position.tolist()} m")
        print(f"Synthetic cup center used for point cloud: {object_center.tolist()} m")
        print(f"pc_full shape: {pc_full.shape}, pc_segment shape: {pc_segment.shape}")

        grasps, scores, contact_pts = _call_plan_point_clouds(
            args.service_url,
            pc_full,
            pc_segment,
            args,
        )

        raw_top_grasp = None
        motion_grasp = None
        summary: dict[str, Any] = {
            "service_url": args.service_url,
            "position_arg_m": object_position,
            "position_is": args.position_is,
            "cup_center_m": object_center,
            "cup_radius_m": args.cup_radius,
            "cup_height_m": args.cup_height,
            "pc_full_shape": pc_full.shape,
            "pc_segment_shape": pc_segment.shape,
            "num_grasps": int(len(grasps)),
        }

        if len(grasps) > 0:
            scores = np.asarray(scores).reshape(-1)
            top_idx, selection_info = _select_grasp_index(grasps, scores, args)
            raw_top_grasp = np.asarray(grasps[top_idx], dtype=np.float64)
            motion_grasp = _transform_graspnet_pose(raw_top_grasp, args)
            top_score = float(scores[top_idx]) if len(scores) > top_idx else float("nan")
            raw_top_pos = raw_top_grasp[:3, 3]
            raw_top_quat = _quat_wxyz_from_matrix(raw_top_grasp[:3, :3])
            top_pos = motion_grasp[:3, 3]
            top_quat = _quat_wxyz_from_matrix(motion_grasp[:3, :3])
            top_contact = (
                np.asarray(contact_pts[top_idx], dtype=np.float64)
                if len(contact_pts) > top_idx
                else np.full(3, np.nan)
            )
            summary.update(
                {
                    "top_index": top_idx,
                    "top_score": top_score,
                    "grasp_selection": selection_info,
                    "raw_top_grasp_pose_matrix": raw_top_grasp,
                    "raw_top_grasp_position_m": raw_top_pos,
                    "raw_top_grasp_quaternion_wxyz": raw_top_quat,
                    "top_grasp_pose_matrix": motion_grasp,
                    "top_grasp_position_m": top_pos,
                    "top_grasp_quaternion_wxyz": top_quat,
                    "top_contact_point_m": top_contact,
                    "top_grasp_distance_to_cup_center_m": float(
                        np.linalg.norm(top_pos - object_center)
                    ),
                }
            )
            print(f"Returned grasps: {len(grasps)}")
            print(f"Selection: {selection_info}")
            print(f"Top score: {top_score:.6f}")
            print(f"Raw GraspNet position: {np.array2string(raw_top_pos, precision=4)}")
            print(f"Raw GraspNet quaternion_wxyz: {np.array2string(raw_top_quat, precision=4)}")
            print(f"Top grasp position: {np.array2string(top_pos, precision=4)}")
            print(f"Top grasp quaternion_wxyz: {np.array2string(top_quat, precision=4)}")
            print(f"Top contact point: {np.array2string(top_contact, precision=4)}")
            if args.execute:
                motion_summary = _execute_motion(
                    motion_grasp=motion_grasp,
                    object_center=object_center,
                    args=args,
                )
                summary["motion"] = motion_summary
        else:
            print("Returned grasps: 0")
            if not args.allow_empty:
                print("No grasp candidates found. Use --allow-empty only if you only want service liveness.")

        npz_path = output_dir / "graspnet_fixed_pose_smoke.npz"
        json_path = output_dir / "summary.json"
        plot_path = output_dir / "graspnet_fixed_pose_smoke.png"
        np.savez(
            npz_path,
            pc_full=pc_full,
            pc_segment=pc_segment,
            grasps=grasps,
            scores=scores,
            contact_pts=contact_pts,
            object_center=object_center,
        )
        json_path.write_text(json.dumps(_jsonify(summary), indent=2) + "\n")
        _save_plot(plot_path, pc_full, pc_segment, motion_grasp, object_center)

        print(f"Saved arrays: {npz_path}")
        print(f"Saved summary: {json_path}")
        print(f"Saved plot: {plot_path}")

        if len(grasps) == 0 and not args.allow_empty:
            return 2
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
