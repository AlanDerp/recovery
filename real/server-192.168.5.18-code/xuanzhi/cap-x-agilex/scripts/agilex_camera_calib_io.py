"""Shared IO utilities for Agilex camera calibration workflows."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf

CALIB_TYPE_POINT_CORRESPONDENCE = "point_correspondence"
CALIB_TYPE_EYE_TO_HAND = "eye_to_hand"

DEFAULT_POINT_OUTPUT_DIR = Path("outputs/agilex_point_calibration")
DEFAULT_EYE_TO_HAND_OUTPUT_DIR = Path("outputs/agilex_eye_to_hand_calibration")
LEGACY_OUTPUT_DIR = Path("outputs/agilex_camera_calibration")

DEFAULT_AGILEX_CONFIG = Path("env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml")


def format_pose_matrix_for_config(transform: np.ndarray, indent: str = "          ") -> str:
    rows = []
    for row in np.asarray(transform, dtype=np.float64).reshape(4, 4):
        values = ", ".join(f"{float(v):.9f}" for v in row)
        rows.append(f"{indent}- [{values}]")
    return "\n".join(rows)


def print_yaml_matrix(transform: np.ndarray) -> None:
    print("\ncamera_pose_mats:")
    print("  main:")
    for row in np.asarray(transform, dtype=np.float64).reshape(4, 4):
        values = ", ".join(f"{float(v): .9f}" for v in row)
        print(f"    - [{values}]")


def write_camera_pose_to_config(config_path: Path, transform: np.ndarray) -> None:
    text = config_path.read_text()
    lines = text.splitlines()
    output: list[str] = []
    idx = 0
    replaced = False
    while idx < len(lines):
        line = lines[idx]
        output.append(line)
        if line.strip() == "main:" and idx > 0 and lines[idx - 1].strip() == "camera_pose_mats:":
            base_indent = line[: len(line) - len(line.lstrip())] + "  "
            idx += 1
            while idx < len(lines):
                stripped = lines[idx].lstrip()
                current_indent = len(lines[idx]) - len(stripped)
                if stripped.startswith("- ["):
                    idx += 1
                    continue
                if stripped and current_indent <= len(base_indent) - 2:
                    break
                if not stripped:
                    break
                break
            output.extend(format_pose_matrix_for_config(transform, indent=base_indent).splitlines())
            replaced = True
            continue
        idx += 1

    if not replaced:
        raise RuntimeError(
            f"Could not find active camera_pose_mats.main block in {config_path}; "
            "write the printed YAML snippet manually or add an active block first."
        )
    config_path.write_text("\n".join(output) + "\n")
    print(f"[write] updated camera_pose_mats.main in {config_path}")


def reset_graspnet_offsets_in_config(config_path: Path) -> None:
    text = config_path.read_text()
    replacements = {
        r"(^(\s*)graspnet_world_offset:\s*)\[[^\]]*\]": r"\1[0.0, 0.0, 0.0]",
        r"(^(\s*)graspnet_rotation_correction_rpy:\s*)\[[^\]]*\]": r"\1[0.0, 0.0, 0.0]",
        r"(^(\s*)graspnet_frame_offset:\s*)\[[^\]]*\]": r"\1[0.0, 0.0, 0.0]",
    }
    updated = text
    for pattern, repl in replacements.items():
        updated = re.sub(pattern, repl, updated, count=1, flags=re.MULTILINE)
    if updated != text:
        config_path.write_text(updated)
        print(f"[write] reset graspnet offsets in {config_path}")
    else:
        print(f"[write] no graspnet offset fields updated in {config_path}")


def apply_calibration_to_config(
    config_path: Path,
    transform: np.ndarray,
    *,
    reset_graspnet_offsets: bool = False,
) -> None:
    write_camera_pose_to_config(config_path, transform)
    if reset_graspnet_offsets:
        reset_graspnet_offsets_in_config(config_path)


def _matrix_from_nested(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    raise ValueError(f"Expected a 4x4 matrix, got shape {arr.shape}")


def read_T_base_camera(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        doc = yaml.safe_load(path.read_text())
        if not isinstance(doc, dict):
            raise ValueError(f"{path} does not contain a YAML mapping")
        if "homogeneous_matrix" in doc:
            return _matrix_from_nested(doc["homogeneous_matrix"])
        if "T_base_camera" in doc:
            return _matrix_from_nested(doc["T_base_camera"])
        raise ValueError(f"{path} missing homogeneous_matrix or T_base_camera")

    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "T_base_camera" in payload:
            return _matrix_from_nested(payload["T_base_camera"])
        if "homogeneous_matrix" in payload:
            return _matrix_from_nested(payload["homogeneous_matrix"])
    raise ValueError(f"{path} is not a recognized calibration result file")


def validate_sample_format(payload: Any, expected_type: str, path: Path | None = None) -> None:
    label = f" ({path})" if path is not None else ""
    if expected_type == CALIB_TYPE_POINT_CORRESPONDENCE:
        if isinstance(payload, list):
            return
        if isinstance(payload, dict) and payload.get("calibration_type") in (
            None,
            CALIB_TYPE_POINT_CORRESPONDENCE,
        ):
            return
        actual = payload.get("calibration_type") if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(
            f"Expected point_correspondence samples{label}, got {actual!r}. "
            f"Use {DEFAULT_POINT_OUTPUT_DIR}/samples.json for point calibration, "
            f"or {DEFAULT_EYE_TO_HAND_OUTPUT_DIR}/samples.json for eye-to-hand."
        )

    if expected_type == CALIB_TYPE_EYE_TO_HAND:
        if isinstance(payload, dict) and payload.get("calibration_type") == CALIB_TYPE_EYE_TO_HAND:
            return
        actual = payload.get("calibration_type") if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(
            f"Expected eye_to_hand samples{label}, got {actual!r}. "
            f"Do not mix point-correspondence and eye-to-hand sample files."
        )
    raise ValueError(f"Unknown calibration type: {expected_type}")


def load_point_correspondence_samples(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    validate_sample_format(payload, CALIB_TYPE_POINT_CORRESPONDENCE, path)
    if isinstance(payload, list):
        if path.parent == LEGACY_OUTPUT_DIR.resolve() or path.parent == LEGACY_OUTPUT_DIR:
            print(
                f"[warn] reading legacy samples from {path}; "
                f"consider migrating to {DEFAULT_POINT_OUTPUT_DIR / 'samples.json'}"
            )
        return payload
    return list(payload.get("samples", []))


def save_point_correspondence_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibration_type": CALIB_TYPE_POINT_CORRESPONDENCE,
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2))


def estimate_camera_to_base(samples: list[dict[str, Any]]) -> tuple[np.ndarray, float, np.ndarray]:
    camera_pts = np.asarray([s["camera_point"] for s in samples], dtype=np.float64)
    base_pts = np.asarray([s["base_point"] for s in samples], dtype=np.float64)
    if camera_pts.shape[0] < 4:
        raise RuntimeError("Need at least 4 samples; 6-10 spread-out samples are better")

    cam_centroid = camera_pts.mean(axis=0)
    base_centroid = base_pts.mean(axis=0)
    cam_centered = camera_pts - cam_centroid
    base_centered = base_pts - base_centroid

    h = cam_centered.T @ base_centered
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = base_centroid - r @ cam_centroid

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r
    transform[:3, 3] = t

    pred = (r @ camera_pts.T).T + t
    errors = np.linalg.norm(pred - base_pts, axis=1)
    return transform, float(np.sqrt(np.mean(errors**2))), errors


def solve_point_correspondence_from_file(samples_path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    samples = load_point_correspondence_samples(samples_path)
    if not samples:
        raise RuntimeError(f"No point-correspondence samples found in {samples_path}")
    return estimate_camera_to_base(samples)


def write_calibration_result_json(
    path: Path,
    *,
    calibration_type: str,
    transform: np.ndarray,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "calibration_type": calibration_type,
        "timestamp": time.time(),
        "T_base_camera": np.asarray(transform, dtype=np.float64).reshape(4, 4).tolist(),
        "homogeneous_matrix": np.asarray(transform, dtype=np.float64).reshape(4, 4).tolist(),
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[io] wrote {path}")


def load_low_level_cfg(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    cfg = OmegaConf.load(config_path)
    return dict(OmegaConf.to_container(cfg.env.cfg.low_level, resolve=True))


def resolve_arm_config(
    *,
    config_path: Path | None,
    urdf: str | Path | None = None,
    target_link: str | None = None,
    tcp_offset: list[float] | np.ndarray | None = None,
) -> tuple[Path, str, np.ndarray]:
    low_level_cfg = load_low_level_cfg(config_path)
    urdf_path = Path(
        urdf or low_level_cfg.get("agilex_ik_urdf_path") or "capx/envs/assets/agilex/piper_ik.urdf"
    )
    target_link_name = target_link or low_level_cfg.get("agilex_ik_target_link_name", "link6")
    tcp_offset_value = tcp_offset if tcp_offset is not None else low_level_cfg.get("agilex_tcp_offset")
    tcp_offset_arr = (
        np.zeros(3, dtype=np.float64)
        if tcp_offset_value is None
        else np.asarray(tcp_offset_value, dtype=np.float64).reshape(3)
    )
    return urdf_path, str(target_link_name), tcp_offset_arr


def read_joints_from_ros(
    *,
    topic: str = "/puppet/joint_left",
    timeout: float = 10.0,
) -> np.ndarray:
    """Read the first six arm joints from a ROS JointState topic."""
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    try:
        import rospy
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise RuntimeError("rospy and sensor_msgs are required for ros joint pose source") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_read_joints", anonymous=True, disable_signals=True)

    print(f"[ros] waiting joints: {topic}")
    msg = rospy.wait_for_message(topic, JointState, timeout=timeout)
    if len(msg.position) < 6:
        raise RuntimeError(
            f"JointState on {topic!r} has {len(msg.position)} positions; expected at least 6"
        )
    return np.asarray(msg.position[:6], dtype=np.float64).reshape(6)


def read_tcp_from_ros_end_pose(
    *,
    topic: str = "/puppet/end_pose_left",
    marker_offset: np.ndarray | list[float] | None = None,
    timeout: float = 10.0,
) -> np.ndarray:
    """Read TCP position from a PoseStamped topic, with optional marker offset in EE frame."""
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("rospy and geometry_msgs are required for ros end-pose source") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_read_end_pose", anonymous=True, disable_signals=True)

    print(f"[ros] waiting end pose: {topic}")
    msg = rospy.wait_for_message(topic, PoseStamped, timeout=timeout)
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
    offset = (
        np.zeros(3, dtype=np.float64)
        if marker_offset is None
        else np.asarray(marker_offset, dtype=np.float64).reshape(3)
    )
    if np.allclose(offset, 0.0):
        return position
    quat = quaternion_xyzw / max(np.linalg.norm(quaternion_xyzw), 1e-12)
    return position + Rotation.from_quat(quat).apply(offset)


def read_current_arm_joints(
    *,
    arm_channel: str = "can_left",
    bitrate: int = 1_000_000,
    robot: str = "piper",
) -> np.ndarray:
    import time as time_mod

    try:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
    except ImportError as exc:
        raise RuntimeError("pyAgxArm is required for pyagxarm pose source") from exc

    cfg = create_agx_arm_config(
        robot=robot,
        comm="can",
        channel=arm_channel,
        bitrate=bitrate,
    )
    arm = AgxArmFactory.create_arm(cfg)
    arm.connect()
    time_mod.sleep(0.2)
    joint_msg = arm.get_joint_angles()
    if joint_msg is None:
        raise RuntimeError("Failed to read current Agilex joint angles")
    return np.asarray(joint_msg.msg, dtype=np.float64).reshape(6)


def T_base_ee_from_joints(
    *,
    urdf_path: Path,
    target_link_name: str,
    joints: np.ndarray,
    tcp_offset: np.ndarray,
) -> np.ndarray:
    from visualize_agilex_urdf import _compute_fk, _parse_urdf

    _links, urdf_joints = _parse_urdf(urdf_path)
    joint_values = {f"joint{idx + 1}": float(value) for idx, value in enumerate(joints)}
    link_poses = _compute_fk(urdf_joints, joint_values)
    if target_link_name not in link_poses:
        raise RuntimeError(f"Target link {target_link_name!r} not found in {urdf_path}")
    target_pose = np.asarray(link_poses[target_link_name], dtype=np.float64).reshape(4, 4).copy()
    offset = np.asarray(tcp_offset, dtype=np.float64).reshape(3)
    if np.any(offset != 0):
        T_offset = np.eye(4, dtype=np.float64)
        T_offset[:3, 3] = offset
        target_pose = target_pose @ T_offset
    return target_pose


def base_point_from_joints(
    *,
    urdf_path: Path,
    target_link_name: str,
    joints: np.ndarray,
    tcp_offset: np.ndarray,
) -> np.ndarray:
    pose = T_base_ee_from_joints(
        urdf_path=urdf_path,
        target_link_name=target_link_name,
        joints=joints,
        tcp_offset=tcp_offset,
    )
    return pose[:3, 3].copy()
