#!/usr/bin/env python3
"""Audit R1 Lite TCP, frame and quaternion conventions.

Default operation is read-only: it never publishes arm or gripper commands.
Solver probing must be explicitly requested with ``--solver-only`` and still
refuses to run unless the RelaxedIK pose topic is isolated from MOCA.
Minimal robot motion requires the additional ``--allow-motion`` flag.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

REPO = Path("/home/nvidia/xuanzhi/cap-x-agilex")
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel


CONFIG = REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
RELAXED_URDF = Path(
    "/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/"
    "r1_lite_floating_right.urdf"
)
FULL_URDF = Path(
    "/home/nvidia/galaxea/install/r1lite_urdf/share/r1lite_urdf/urdf/"
    "r1lite_2026.urdf"
)
OUTPUT_DIR = REPO / "outputs/r1lite_tennis_ball_in_box"


def xyz(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(value) for value in text.split()], dtype=float)


def rigid(
    translation: np.ndarray | tuple[float, float, float],
    rotation: Rotation | None = None,
) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, 3] = np.asarray(translation, dtype=float)
    if rotation is not None:
        result[:3, :3] = rotation.as_matrix()
    return result


def pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return rigid(position, Rotation.from_quat(quaternion))


def matrix_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    return matrix[:3, 3].copy(), Rotation.from_matrix(matrix[:3, :3]).as_quat()


def rotation_error_rad(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=float)[:3, :3].T @ np.asarray(b, dtype=float)[:3, :3]
    return float(Rotation.from_matrix(relative).magnitude())


def residual(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    """Return B minus A translation and A-to-B relative rotation."""
    a = np.asarray(a, dtype=float).reshape(4, 4)
    b = np.asarray(b, dtype=float).reshape(4, 4)
    relative = np.linalg.inv(a) @ b
    return {
        "translation_norm_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "translation_b_minus_a_base_m": (b[:3, 3] - a[:3, 3]).tolist(),
        "rotation_error_rad": rotation_error_rad(a, b),
        "relative_transform_a_to_b": relative.tolist(),
    }


class UrdfChain:
    def __init__(self, path: Path) -> None:
        self.path = path
        root = ET.parse(path).getroot()
        self.by_child: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                self.by_child[child.attrib["link"]] = joint

    def joints_from_root(self, root_link: str, target_link: str) -> list[ET.Element]:
        reverse: list[ET.Element] = []
        current = target_link
        while current != root_link:
            if current not in self.by_child:
                raise ValueError(f"No URDF chain from {root_link} to {target_link}")
            joint = self.by_child[current]
            reverse.append(joint)
            parent = joint.find("parent")
            assert parent is not None
            current = parent.attrib["link"]
        return list(reversed(reverse))

    def fk(
        self,
        root_link: str,
        target_link: str,
        joint_positions: dict[str, float],
    ) -> np.ndarray:
        result = np.eye(4, dtype=float)
        for joint in self.joints_from_root(root_link, target_link):
            origin = joint.find("origin")
            origin_xyz = xyz(None if origin is None else origin.attrib.get("xyz"), (0, 0, 0))
            origin_rpy = xyz(None if origin is None else origin.attrib.get("rpy"), (0, 0, 0))
            result = result @ rigid(origin_xyz, Rotation.from_euler("xyz", origin_rpy))
            if joint.attrib.get("type") in {"revolute", "continuous"}:
                axis_node = joint.find("axis")
                axis = xyz(
                    None if axis_node is None else axis_node.attrib.get("xyz"),
                    (1, 0, 0),
                )
                axis /= np.linalg.norm(axis)
                angle = float(joint_positions.get(joint.attrib["name"], 0.0))
                result = result @ rigid((0, 0, 0), Rotation.from_rotvec(axis * angle))
            elif joint.attrib.get("type") == "prismatic":
                axis_node = joint.find("axis")
                axis = xyz(
                    None if axis_node is None else axis_node.attrib.get("xyz"),
                    (1, 0, 0),
                )
                position = float(joint_positions.get(joint.attrib["name"], 0.0))
                result = result @ rigid(axis * position)
        return result


def load_low_level(*, motion_validation: bool = False) -> R1LiteRealLowLevel:
    config = yaml.safe_load(CONFIG.read_text())
    values = dict(config["env"]["cfg"]["low_level"])
    values.pop("_target_", None)
    values["r1lite_motion_validation_enabled"] = motion_validation
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {key: value for key, value in values.items() if key in signature.parameters}
    return R1LiteRealLowLevel(**kwargs)


def tool_transform_from_urdf() -> np.ndarray:
    chain = UrdfChain(RELAXED_URDF)
    return chain.fk("right_gripper_link", "right_gripper_tool_z", {})


def arm_joint_map(arm: np.ndarray) -> dict[str, float]:
    return {
        f"right_arm_joint{index + 1}": float(value)
        for index, value in enumerate(np.asarray(arm, dtype=float).reshape(6))
    }


def full_joint_map(arm: np.ndarray, torso: np.ndarray) -> dict[str, float]:
    result = arm_joint_map(arm)
    result.update(
        {
            f"torso_joint{index + 1}": float(value)
            for index, value in enumerate(np.asarray(torso, dtype=float).reshape(3))
        }
    )
    return result


def roundtrip_audit(base_from_torso: np.ndarray) -> dict[str, object]:
    samples = [
        pose_matrix(np.array([0.2, -0.3, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])),
        pose_matrix(np.array([0.5, -0.1, 0.9]), np.array([1.0, 0.0, 0.0, 0.0])),
        pose_matrix(
            np.array([0.35, -0.25, 1.15]),
            Rotation.from_euler("xyz", [0.3, -0.5, 1.2]).as_quat(),
        ),
    ]
    rows = []
    for target_base in samples:
        target_torso = np.linalg.inv(base_from_torso) @ target_base
        recovered_base = base_from_torso @ target_torso
        rows.append(residual(target_base, recovered_base))

    q = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_quat()
    q_neg = -q
    q_equivalence = rotation_error_rad(
        pose_matrix(np.zeros(3), q),
        pose_matrix(np.zeros(3), q_neg),
    )
    expected_tool = rigid(
        (0, 0, 0),
        Rotation.from_euler("y", math.pi / 2),
    )
    actual_tool = tool_transform_from_urdf()
    return {
        "base_torso_base_samples": rows,
        "max_position_error_m": max(row["translation_norm_m"] for row in rows),
        "max_rotation_error_rad": max(row["rotation_error_rad"] for row in rows),
        "q_neg_q_rotation_error_rad": q_equivalence,
        "tool_transform_urdf": actual_tool.tolist(),
        "tool_transform_expected": expected_tool.tolist(),
        "tool_transform_residual": residual(expected_tool, actual_tool),
    }


def static_alignment(r1: R1LiteRealLowLevel) -> dict[str, object]:
    arm = r1.get_arm_joints()
    physical_position, physical_quaternion = r1.get_ee_pose()
    planning_position, planning_quaternion = r1.get_planning_ee_pose()
    base_from_torso, torso = r1._base_from_torso_link3()

    relaxed_chain = UrdfChain(RELAXED_URDF)
    torso_from_gripper = relaxed_chain.fk(
        "torso_link3",
        "right_gripper_link",
        arm_joint_map(arm),
    )
    torso_from_tool = relaxed_chain.fk(
        "torso_link3",
        "right_gripper_tool_z",
        arm_joint_map(arm),
    )
    model_gripper = base_from_torso @ torso_from_gripper
    model_tool = base_from_torso @ torso_from_tool
    full_chain = UrdfChain(FULL_URDF)
    full_gripper = full_chain.fk(
        "base_link",
        "right_gripper_link",
        full_joint_map(arm, torso),
    )
    full_tool = full_chain.fk(
        "base_link",
        "right_gripper_tool_z",
        full_joint_map(arm, torso),
    )
    physical = pose_matrix(physical_position, physical_quaternion)
    planning = pose_matrix(planning_position, planning_quaternion)
    physical_as_tool = physical @ tool_transform_from_urdf()

    return {
        "timestamp_unix": time.time(),
        "semantics": {
            "task_target": "T_base_right_gripper_tool_z",
            "physical_topic_observed": "T_base_right_gripper_link",
            "planning_topic_observed": "T_base_right_gripper_tool_z",
            "quaternion_order": "xyzw",
        },
        "sources": {
            "relaxed_urdf": str(RELAXED_URDF),
            "full_urdf": str(FULL_URDF),
            "physical_topic": "/motion_control/pose_ee_arm_right",
            "planning_topic": "/relaxed_ik/motion_control/pose_ee_arm_right",
        },
        "arm_joints_rad": arm.tolist(),
        "torso_joints_rad": torso.tolist(),
        "matrices": {
            "T_base_torso": base_from_torso.tolist(),
            "T_base_physical_topic": physical.tolist(),
            "T_base_planning_topic": planning.tolist(),
            "T_base_model_gripper_link": model_gripper.tolist(),
            "T_base_model_tool_z": model_tool.tolist(),
            "T_base_full_urdf_gripper_link": full_gripper.tolist(),
            "T_base_full_urdf_tool_z": full_tool.tolist(),
            "T_base_physical_topic_converted_to_tool_z": physical_as_tool.tolist(),
        },
        "residuals": {
            "physical_vs_model_gripper": residual(physical, model_gripper),
            "physical_vs_full_urdf_gripper": residual(physical, full_gripper),
            "physical_vs_model_tool": residual(physical, model_tool),
            "physical_as_tool_vs_model_tool": residual(physical_as_tool, model_tool),
            "physical_as_tool_vs_full_urdf_tool": residual(
                physical_as_tool,
                full_tool,
            ),
            "relaxed_model_tool_vs_full_urdf_tool": residual(model_tool, full_tool),
            "planning_vs_model_tool": residual(planning, model_tool),
            "physical_as_tool_vs_planning": residual(physical_as_tool, planning),
        },
    }


def subscriber_nodes(r1: R1LiteRealLowLevel) -> list[str]:
    infos = r1._node.get_subscriptions_info_by_topic(
        r1._relaxed_ik_pose_pub.topic_name
    )
    return sorted({str(info.node_name) for info in infos})


def solver_audit(r1: R1LiteRealLowLevel) -> dict[str, object]:
    subscribers = subscriber_nodes(r1)
    unexpected = [
        name
        for name in subscribers
        if "relaxed_ik_right" not in name
    ]
    if unexpected:
        raise RuntimeError(
            "Refusing solver audit because pose target has non-solver subscribers: "
            f"{unexpected}"
        )

    arm = r1.get_arm_joints()
    physical_position, physical_quaternion = r1.get_ee_pose()
    tool_quaternion = (
        Rotation.from_quat(physical_quaternion)
        * Rotation.from_euler("y", math.pi / 2)
    ).as_quat()

    rows = []
    for semantic, quaternion in [
        ("physical_gripper_quaternion_as_tool_target", physical_quaternion),
        ("physical_gripper_converted_to_tool_z", tool_quaternion),
    ]:
        r1._relaxed_ik_initialized = False
        r1._reset_relaxed_ik_to_current()
        candidate = r1.get_relaxed_ik_candidate(
            physical_position,
            quaternion,
            timeout_s=3.0,
            reference_joints=arm,
        )
        fk_position, fk_quaternion = r1.get_relaxed_ik_fk_pose_base()
        target = pose_matrix(physical_position, quaternion)
        fk = pose_matrix(fk_position, fk_quaternion)
        rows.append(
            {
                "semantic": semantic,
                "target_position_base_m": physical_position.tolist(),
                "target_quaternion_xyzw": quaternion.tolist(),
                "seed_measured_joints_rad": arm.tolist(),
                "solution_rad": candidate.tolist(),
                "max_joint_delta_from_seed_rad": float(
                    np.max(np.abs(candidate - arm))
                ),
                "fk_position_base_m": fk_position.tolist(),
                "fk_quaternion_xyzw": fk_quaternion.tolist(),
                "fk_target_residual": residual(target, fk),
            }
        )
    return {
        "timestamp_unix": time.time(),
        "pose_target_subscribers": subscribers,
        "rows": rows,
    }


def motion_audit(
    r1: R1LiteRealLowLevel,
    *,
    joint_delta_rad: float,
) -> dict[str, object]:
    initial = r1.get_arm_joints()
    static_before = static_alignment(r1)
    rows = [{"name": "initial", "alignment": static_before}]
    try:
        for joint_index in [5, 4, 3]:
            target = initial.copy()
            target[joint_index] += joint_delta_rad
            r1.move_joints_safe(target, max_step=0.01, duration_s=0.8)
            time.sleep(0.3)
            rows.append(
                {
                    "name": f"joint_{joint_index + 1}_plus",
                    "commanded_joints_rad": target.tolist(),
                    "measured_joints_rad": r1.get_arm_joints().tolist(),
                    "alignment": static_alignment(r1),
                }
            )
            r1.move_joints_safe(initial, max_step=0.01, duration_s=0.8)
            time.sleep(0.3)
    finally:
        r1.move_joints_safe(initial, max_step=0.01, duration_s=0.8)
    return {
        "timestamp_unix": time.time(),
        "initial_joints_rad": initial.tolist(),
        "joint_delta_rad": joint_delta_rad,
        "rows": rows,
        "final_joints_rad": r1.get_arm_joints().tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-only", action="store_true")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--joint-delta-rad", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.allow_motion and args.solver_only:
        parser.error("--allow-motion and --solver-only are separate audit modes")

    r1 = load_low_level(motion_validation=True)
    try:
        base_from_torso, torso = r1._base_from_torso_link3()
        result: dict[str, object] = {
            "contract": str(REPO / "docs/r1lite_tcp_frame_contract.md"),
            "roundtrip": roundtrip_audit(base_from_torso),
            "torso_joints_rad": torso.tolist(),
        }
        if args.solver_only:
            result["solver"] = solver_audit(r1)
            default_name = f"tcp_frame_solver_audit_{time.strftime('%Y%m%d_%H%M%S')}.json"
        elif args.allow_motion:
            result["motion"] = motion_audit(
                r1,
                joint_delta_rad=float(args.joint_delta_rad),
            )
            default_name = f"tcp_frame_motion_audit_{time.strftime('%Y%m%d_%H%M%S')}.json"
        else:
            result["static"] = static_alignment(r1)
            default_name = f"tcp_frame_static_audit_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output = args.output or OUTPUT_DIR / default_name
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(output)
        print(json.dumps(result, indent=2))
    finally:
        r1.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
