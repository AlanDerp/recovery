#!/usr/bin/env python3
"""Generate a local HTML visualization for the Agilex/Piper IK URDF."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


DEFAULT_URDF = (
    Path(__file__).resolve().parents[1] / "capx" / "envs" / "assets" / "agilex" / "piper_ik.urdf"
)


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def _parse_vector(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def _origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    mat[:3, 3] = xyz
    return mat


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        return np.eye(4, dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_rotvec(axis / axis_norm * angle).as_matrix()
    return mat


def _parse_urdf(path: Path) -> tuple[list[str], list[Joint]]:
    root = ET.parse(path).getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    joints: list[Joint] = []
    for joint_el in root.findall("joint"):
        origin_el = joint_el.find("origin")
        axis_el = joint_el.find("axis")
        limit_el = joint_el.find("limit")
        joints.append(
            Joint(
                name=joint_el.attrib["name"],
                joint_type=joint_el.attrib.get("type", "fixed"),
                parent=joint_el.find("parent").attrib["link"],  # type: ignore[union-attr]
                child=joint_el.find("child").attrib["link"],  # type: ignore[union-attr]
                xyz=_parse_vector(origin_el.attrib.get("xyz") if origin_el is not None else None, (0, 0, 0)),
                rpy=_parse_vector(origin_el.attrib.get("rpy") if origin_el is not None else None, (0, 0, 0)),
                axis=_parse_vector(axis_el.attrib.get("xyz") if axis_el is not None else None, (0, 0, 1)),
                lower=float(limit_el.attrib["lower"]) if limit_el is not None and "lower" in limit_el.attrib else None,
                upper=float(limit_el.attrib["upper"]) if limit_el is not None and "upper" in limit_el.attrib else None,
            )
        )
    return links, joints


def _compute_fk(joints: list[Joint], joint_values: dict[str, float]) -> dict[str, np.ndarray]:
    children_by_parent: dict[str, list[Joint]] = {}
    children = set()
    for joint in joints:
        children_by_parent.setdefault(joint.parent, []).append(joint)
        children.add(joint.child)

    root_links = sorted({joint.parent for joint in joints} - children)
    root_link = root_links[0] if root_links else "world"
    poses = {root_link: np.eye(4, dtype=np.float64)}

    stack = [root_link]
    while stack:
        parent = stack.pop()
        parent_pose = poses[parent]
        for joint in children_by_parent.get(parent, []):
            transform = _origin_matrix(joint.xyz, joint.rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                transform = transform @ _axis_rotation(joint.axis, joint_values.get(joint.name, 0.0))
            poses[joint.child] = parent_pose @ transform
            stack.append(joint.child)
    return poses


def _load_low_level_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    try:
        import yaml
    except ImportError:
        return {}

    data = yaml.safe_load(config_path.read_text())
    return data.get("env", {}).get("cfg", {}).get("low_level", {})


def _load_camera_pose(config_path: Path | None) -> np.ndarray | None:
    low_level = _load_low_level_config(config_path)
    pose = low_level.get("camera_pose_mats", {}).get("main")
    if pose is None:
        return None
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def _load_ik_target_link(config_path: Path | None) -> str | None:
    low_level = _load_low_level_config(config_path)
    target = low_level.get("agilex_ik_target_link_name")
    if target is None:
        return None
    return str(target)


def _add_line(fig: go.Figure, start: np.ndarray, end: np.ndarray, *, color: str, width: int, name: str) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line={"color": color, "width": width},
            name=name,
            showlegend=False,
        )
    )


def _add_frame(fig: go.Figure, pose: np.ndarray, name: str, *, axis_len: float = 0.055) -> None:
    origin = pose[:3, 3]
    axes = [
        ("x", "#d62728", pose[:3, 0]),
        ("y", "#2ca02c", pose[:3, 1]),
        ("z", "#1f77b4", pose[:3, 2]),
    ]
    for suffix, color, direction in axes:
        _add_line(fig, origin, origin + direction * axis_len, color=color, width=5, name=f"{name}_{suffix}")
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode="markers+text",
            marker={"size": 4, "color": "#111111"},
            text=[name],
            textposition="top center",
            name=name,
            showlegend=False,
        )
    )


def _joint_values_from_args(joints: list[Joint], values: list[float] | None) -> dict[str, float]:
    movable = [joint for joint in joints if joint.joint_type in {"revolute", "continuous"}]
    if values is not None and len(values) != len(movable):
        raise ValueError(f"--joints expects {len(movable)} values, got {len(values)}")
    result: dict[str, float] = {}
    for idx, joint in enumerate(movable):
        value = 0.0 if values is None else float(values[idx])
        if joint.lower is not None and joint.upper is not None:
            value = float(np.clip(value, joint.lower, joint.upper))
        result[joint.name] = value
    return result


def build_figure(urdf_path: Path, config_path: Path | None, joint_args: list[float] | None) -> go.Figure:
    _, joints = _parse_urdf(urdf_path)
    joint_values = _joint_values_from_args(joints, joint_args)
    poses = _compute_fk(joints, joint_values)

    fig = go.Figure()
    for joint in joints:
        if joint.parent not in poses or joint.child not in poses:
            continue
        parent_pos = poses[joint.parent][:3, 3]
        child_pos = poses[joint.child][:3, 3]
        _add_line(fig, parent_pos, child_pos, color="#444444", width=7, name=joint.name)

    for link_name, pose in poses.items():
        _add_frame(fig, pose, link_name)

    camera_pose = _load_camera_pose(config_path)
    if camera_pose is not None:
        _add_frame(fig, camera_pose, "camera_main", axis_len=0.09)
        camera_origin = camera_pose[:3, 3]
        forward = camera_pose[:3, 2]
        _add_line(
            fig,
            camera_origin,
            camera_origin + forward * 0.18,
            color="#9467bd",
            width=6,
            name="camera_forward",
        )

    target_link = _load_ik_target_link(config_path)
    if target_link in poses:
        target_pose = poses[target_link]
        _add_frame(fig, target_pose, f"ik_target_{target_link}", axis_len=0.085)
        target_origin = target_pose[:3, 3]
        target_forward = target_pose[:3, 2]
        _add_line(
            fig,
            target_origin,
            target_origin + target_forward * 0.14,
            color="#ff7f0e",
            width=8,
            name=f"{target_link}_forward",
        )

    joint_text = "<br>".join(f"{name}: {value:.4f} rad" for name, value in joint_values.items())
    target_text = f"<br>IK target: {target_link}" if target_link else ""
    fig.update_layout(
        title=f"Agilex/Piper URDF skeleton<br><sup>{urdf_path}{target_text}<br>{joint_text}</sup>",
        scene={
            "xaxis": {"title": "base X (m)"},
            "yaxis": {"title": "base Y (m)"},
            "zaxis": {"title": "base Z (m)"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 90},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=None, help="Optional Agilex YAML to draw camera_pose_mats.main.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/agilex_piper_urdf.html"))
    parser.add_argument("--joints", type=float, nargs="*", default=None, help="Six joint values in radians.")
    args = parser.parse_args()

    fig = build_figure(args.urdf, args.config, args.joints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
