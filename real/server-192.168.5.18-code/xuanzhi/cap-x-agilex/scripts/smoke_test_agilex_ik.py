#!/usr/bin/env python3
"""Smoke test Agilex/Piper IK, with optional interpolated hardware execution."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from capx.integrations.agilex.control import AgilexControlApi

from visualize_agilex_urdf import _compute_fk, _parse_urdf


DEFAULT_CONFIG = Path("env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml")
DEFAULT_CASES = np.array(
    [
        [0.0, 0.6, -0.7, 0.0, 0.35, 0.0],
        [0.25, 1.0, -1.2, 0.25, 0.25, -0.35],
        [-0.35, 0.85, -1.05, -0.2, -0.2, 0.45],
    ],
    dtype=np.float64,
)


class OfflineAgilexEnv:
    """Minimal env object used by AgilexControlApi for offline IK checks."""

    def __init__(self, low_level_cfg: dict[str, Any], initial_joints: np.ndarray) -> None:
        for key, value in low_level_cfg.items():
            setattr(self, key, value)
        self._obs = {
            "robot_joint_pos": np.concatenate(
                [np.asarray(initial_joints, dtype=np.float32), np.array([1.0], dtype=np.float32)]
            )
        }

    def get_observation(self) -> dict[str, Any]:
        return self._obs


def _as_wxyz(rot: Rotation) -> np.ndarray:
    quat_xyzw = rot.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _wxyz_to_rotation(quat_wxyz: np.ndarray) -> Rotation:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])


def _pose_to_matrix_wxyz_xyz(pose: list[float] | np.ndarray | None) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if pose is None:
        return mat
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(7)
    mat[:3, :3] = _wxyz_to_rotation(pose_arr[:4]).as_matrix()
    mat[:3, 3] = pose_arr[4:]
    return mat


def _target_pose_from_joints(
    *,
    urdf_path: Path,
    target_link_name: str,
    joints: np.ndarray,
    base_pose_wxyz_xyz: list[float] | np.ndarray | None,
    tcp_offset: list[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, urdf_joints = _parse_urdf(urdf_path)
    joint_values = {f"joint{idx + 1}": float(value) for idx, value in enumerate(joints)}
    link_poses_base = _compute_fk(urdf_joints, joint_values)
    if target_link_name not in link_poses_base:
        raise ValueError(f"Target link {target_link_name!r} is not in {urdf_path}")

    world_base = _pose_to_matrix_wxyz_xyz(base_pose_wxyz_xyz)
    target_link_world = world_base @ link_poses_base[target_link_name]
    target_rot_world = Rotation.from_matrix(target_link_world[:3, :3])
    target_quat_world = _as_wxyz(target_rot_world)
    offset = (
        np.zeros(3, dtype=np.float64)
        if tcp_offset is None
        else np.asarray(tcp_offset, dtype=np.float64).reshape(3)
    )
    requested_pos_world = target_link_world[:3, 3] - target_rot_world.apply(offset)
    return requested_pos_world, target_quat_world, target_link_world


def _fk_pose(
    *,
    urdf_path: Path,
    target_link_name: str,
    joints: np.ndarray,
    base_pose_wxyz_xyz: list[float] | np.ndarray | None,
) -> np.ndarray:
    _, urdf_joints = _parse_urdf(urdf_path)
    joint_values = {f"joint{idx + 1}": float(value) for idx, value in enumerate(joints)}
    link_poses_base = _compute_fk(urdf_joints, joint_values)
    return _pose_to_matrix_wxyz_xyz(base_pose_wxyz_xyz) @ link_poses_base[target_link_name]


def _orientation_error_deg(target: np.ndarray, actual: np.ndarray) -> float:
    target_rot = Rotation.from_matrix(target[:3, :3])
    actual_rot = Rotation.from_matrix(actual[:3, :3])
    return math.degrees((target_rot.inv() * actual_rot).magnitude())


def _load_low_level_cfg(config_path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    low_level = cfg.env.cfg.low_level
    return dict(OmegaConf.to_container(low_level, resolve=True))


def _initial_joints_from_cfg(low_level_cfg: dict[str, Any]) -> np.ndarray:
    joints = low_level_cfg.get("reset_joint_positions", [0.0] * 6)
    return np.asarray(joints, dtype=np.float64).reshape(6)


def _run_offline_cases(
    *,
    config_path: Path,
    case_selector: str,
    pos_tol: float,
    rot_tol_deg: float,
    custom_position: list[float] | None,
    custom_quaternion: list[float] | None,
) -> list[dict[str, Any]]:
    low_level_cfg = _load_low_level_cfg(config_path)
    urdf_path = Path(
        low_level_cfg.get("agilex_ik_urdf_path")
        or "capx/envs/assets/agilex/piper_ik.urdf"
    )
    target_link_name = low_level_cfg.get("agilex_ik_target_link_name", "link6")
    base_pose = low_level_cfg.get("base_pose_wxyz_xyz")
    tcp_offset = low_level_cfg.get("agilex_tcp_offset")
    env = OfflineAgilexEnv(low_level_cfg, _initial_joints_from_cfg(low_level_cfg))
    api = AgilexControlApi(env, initialize_motion=True, debug=True)

    if custom_position is not None:
        if custom_quaternion is None:
            raise ValueError("--position requires --quaternion")
        requested_targets = [
            {
                "name": "custom",
                "source_joints": None,
                "position": np.asarray(custom_position, dtype=np.float64).reshape(3),
                "quaternion": np.asarray(custom_quaternion, dtype=np.float64).reshape(4),
                "target_pose": None,
            }
        ]
    else:
        cases = DEFAULT_CASES
        if case_selector != "all":
            idx = int(case_selector)
            cases = cases[idx : idx + 1]
        requested_targets = []
        for idx, joints in enumerate(cases):
            position, quat, target_pose = _target_pose_from_joints(
                urdf_path=urdf_path,
                target_link_name=target_link_name,
                joints=joints,
                base_pose_wxyz_xyz=base_pose,
                tcp_offset=tcp_offset,
            )
            requested_targets.append(
                {
                    "name": f"case_{idx if case_selector == 'all' else case_selector}",
                    "source_joints": joints,
                    "position": position,
                    "quaternion": quat,
                    "target_pose": target_pose,
                }
            )

    results: list[dict[str, Any]] = []
    for target in requested_targets:
        print(f"\n=== {target['name']} ===")
        print("target position:", np.array2string(target["position"], precision=5))
        print("target quat wxyz:", np.array2string(target["quaternion"], precision=5))
        solved = api.solve_ik(target["position"], target["quaternion"])
        solved_pose = _fk_pose(
            urdf_path=urdf_path,
            target_link_name=target_link_name,
            joints=solved,
            base_pose_wxyz_xyz=base_pose,
        )
        if target["target_pose"] is None:
            desired_pose = np.eye(4, dtype=np.float64)
            desired_pose[:3, :3] = _wxyz_to_rotation(target["quaternion"]).as_matrix()
            desired_pose[:3, 3] = target["position"]
        else:
            desired_pose = target["target_pose"]

        pos_err = float(np.linalg.norm(solved_pose[:3, 3] - desired_pose[:3, 3]))
        rot_err = _orientation_error_deg(desired_pose, solved_pose)
        passed = pos_err <= pos_tol and rot_err <= rot_tol_deg
        print("solved joints:", np.array2string(solved, precision=5))
        if target["source_joints"] is not None:
            print("source joints:", np.array2string(target["source_joints"], precision=5))
        print("solved FK position:", np.array2string(solved_pose[:3, 3], precision=5))
        print(f"position error: {pos_err:.5f} m")
        print(f"orientation error: {rot_err:.3f} deg")
        print("status:", "PASS" if passed else "FAIL")
        results.append(
            {
                "name": target["name"],
                "position": target["position"],
                "quaternion": target["quaternion"],
                "solved_joints": solved,
                "pos_err": pos_err,
                "rot_err_deg": rot_err,
                "passed": passed,
            }
        )
    return results


def _execute_interpolated(
    *,
    low_level_cfg: dict[str, Any],
    target_joints: np.ndarray,
    gripper_width: float,
    steps: int,
) -> None:
    from capx.envs.simulators.agilex_real import AgilexRealLowLevel

    kwargs = dict(low_level_cfg)
    kwargs.update(
        {
            "camera_backend": "none",
            "wait_for_observation": False,
            "start_server": False,
            "reset_on_env_reset": False,
        }
    )
    env = AgilexRealLowLevel(**kwargs)
    if env.hardware is None:
        raise RuntimeError("Hardware did not initialize; check control_backend/direct settings.")

    start_joints, start_width = env.hardware.get_state()
    start_joints = np.asarray(start_joints, dtype=np.float64).reshape(6)
    target_joints = env._clip_joint_targets(target_joints)
    start_width = float(np.clip(start_width, 0.0, env.gripper_max_width))
    gripper_width = float(np.clip(gripper_width, 0.0, env.gripper_max_width))
    steps = max(1, int(steps))

    print("\nExecuting interpolated hardware move")
    print("start joints:", np.array2string(start_joints, precision=5))
    print("target joints:", np.array2string(target_joints, precision=5))
    print(f"start gripper width: {start_width:.4f} m")
    print(f"target gripper width: {gripper_width:.4f} m")
    print(f"steps: {steps}")

    for idx in range(1, steps + 1):
        alpha = idx / steps
        joints = start_joints + (target_joints - start_joints) * alpha
        width = start_width + (gripper_width - start_width) * alpha
        env.hardware.move(joints=joints, gripper_width=width)
        if idx == 1 or idx == steps or idx % 10 == 0:
            print(f"  step {idx:03d}/{steps}")
        time.sleep(0.01)

    final_joints, final_width = env.hardware.get_state()
    print("final observed joints:", np.array2string(final_joints, precision=5))
    print(f"final observed gripper width: {final_width:.4f} m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case", default="all", help="'all' or case index 0..2.")
    parser.add_argument("--pos-tol", type=float, default=0.03)
    parser.add_argument("--rot-tol-deg", type=float, default=10.0)
    parser.add_argument("--position", type=float, nargs=3, default=None)
    parser.add_argument("--quaternion", type=float, nargs=4, default=None, help="WXYZ quaternion.")
    parser.add_argument("--execute", action="store_true", help="Connect to hardware and execute selected IK result.")
    parser.add_argument("--steps", type=int, default=50, help="Execution interpolation steps.")
    parser.add_argument("--gripper-width", type=float, default=0.05)
    args = parser.parse_args()

    if args.execute and args.case == "all" and args.position is None:
        raise SystemExit("Use --case 0, --case 1, or --case 2 with --execute.")

    results = _run_offline_cases(
        config_path=args.config,
        case_selector=args.case,
        pos_tol=args.pos_tol,
        rot_tol_deg=args.rot_tol_deg,
        custom_position=args.position,
        custom_quaternion=args.quaternion,
    )
    failed = [result for result in results if not result["passed"]]
    if failed:
        print("\nIK smoke test failed.")
        raise SystemExit(1)

    print("\nIK smoke test passed.")
    if args.execute:
        low_level_cfg = _load_low_level_cfg(args.config)
        _execute_interpolated(
            low_level_cfg=low_level_cfg,
            target_joints=results[0]["solved_joints"],
            gripper_width=args.gripper_width,
            steps=args.steps,
        )


if __name__ == "__main__":
    main()
