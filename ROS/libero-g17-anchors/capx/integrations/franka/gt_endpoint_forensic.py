"""Evaluator-only MuJoCo endpoint contact witness for acquisition probes.

The helper temporarily places only the seven Panda arm qpos at an already
solved IK candidate, runs ``forward`` without advancing physics, records raw
MuJoCo contacts, and restores the complete simulator state.  It is diagnostic
GT: no result is exposed to or consumed by the robot policy.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np


def enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V300A_GT_ENDPOINT_FORENSIC", "0") == "1"


def _geom_name(model: Any, geom_id: int) -> str:
    return str(model.geom_id2name(int(geom_id)) or f"geom:{int(geom_id)}")


def _pyroki_hand_pose(api: Any, candidate_cfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from capx.integrations.franka.tip_xy_only import fk_panda_hand

    return fk_panda_hand(api, candidate_cfg)


def audit_endpoint(api: Any, candidate_cfg: np.ndarray) -> dict[str, Any]:
    """Return raw endpoint contacts while proving exact state restoration."""
    env = api._env
    sim = env.handle.env.sim
    addresses = np.asarray(env._panda_joint_qpos_addrs, dtype=np.int64)
    candidate = np.asarray(candidate_cfg, dtype=np.float64).reshape(-1)
    if len(addresses) != 7 or candidate.size < 7:
        raise ValueError("GT endpoint forensic requires seven Panda arm joints")

    state = copy.deepcopy(sim.get_state())
    qpos_before = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    contacts: list[dict[str, Any]] = []
    endpoint_q_error = float("inf")
    fk_crosscheck: dict[str, Any] = {}
    try:
        sim.data.qpos[addresses] = candidate[:7]
        sim.forward()
        endpoint_q_error = float(
            np.max(np.abs(np.asarray(sim.data.qpos)[addresses] - candidate[:7]))
        )
        try:
            import viser.transforms as vtf

            base = vtf.SE3(
                wxyz_xyz=np.concatenate(
                    [
                        np.asarray(sim.data.xquat[env.base_link_idx]),
                        np.asarray(sim.data.xpos[env.base_link_idx]),
                    ]
                )
            )
            eef = vtf.SE3(
                wxyz_xyz=np.concatenate(
                    [
                        np.asarray(sim.data.xquat[env.gripper_link_idx]),
                        np.asarray(sim.data.xpos[env.gripper_link_idx]),
                    ]
                )
            )
            mujoco_hand = base.inverse() @ eef
            mujoco_pos = np.asarray(mujoco_hand.translation(), dtype=np.float64)
            mujoco_quat = np.asarray(mujoco_hand.rotation().wxyz, dtype=np.float64)
            pyroki_pos, pyroki_quat = _pyroki_hand_pose(api, candidate)
            quat_dot = float(
                abs(
                    np.dot(
                        mujoco_quat / np.linalg.norm(mujoco_quat),
                        pyroki_quat / np.linalg.norm(pyroki_quat),
                    )
                )
            )
            pyroki_hand = vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3(np.asarray(pyroki_quat, dtype=np.float64)),
                translation=np.asarray(pyroki_pos, dtype=np.float64),
            )
            # robosuite attaches PandaGripper.right_gripper/eef below
            # Panda.right_hand.  At runtime the resulting fixed transform from
            # PyRoKi panda_hand to MuJoCo gripper0_eef is Rz(-pi/2), +96.5 mm Z.
            predicted_mujoco_grip = pyroki_hand @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, -np.pi / 2.0),
                translation=np.array([0.0, 0.0, 0.0965]),
            )
            aligned_pos = np.asarray(predicted_mujoco_grip.translation())
            aligned_quat = np.asarray(predicted_mujoco_grip.rotation().wxyz)
            aligned_dot = float(
                abs(
                    np.dot(
                        mujoco_quat / np.linalg.norm(mujoco_quat),
                        aligned_quat / np.linalg.norm(aligned_quat),
                    )
                )
            )
            fk_crosscheck = {
                "mujoco_hand_xyz": mujoco_pos.round(7).tolist(),
                "mujoco_hand_wxyz": mujoco_quat.round(7).tolist(),
                "pyroki_hand_xyz": np.asarray(pyroki_pos).round(7).tolist(),
                "pyroki_hand_wxyz": np.asarray(pyroki_quat).round(7).tolist(),
                "position_error_m": float(np.linalg.norm(mujoco_pos - pyroki_pos)),
                "orientation_error_rad": float(
                    2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))
                ),
                "model_aligned_mujoco_grip_xyz": aligned_pos.round(7).tolist(),
                "model_aligned_mujoco_grip_wxyz": aligned_quat.round(7).tolist(),
                "model_aligned_position_error_m": float(
                    np.linalg.norm(mujoco_pos - aligned_pos)
                ),
                "model_aligned_orientation_error_rad": float(
                    2.0 * np.arccos(np.clip(aligned_dot, -1.0, 1.0))
                ),
            }
        except Exception as exc:
            fk_crosscheck = {"error": repr(exc)}
        for index in range(int(sim.data.ncon)):
            contact = sim.data.contact[index]
            name1 = _geom_name(sim.model, int(contact.geom1))
            name2 = _geom_name(sim.model, int(contact.geom2))
            pair = sorted((name1, name2))
            if not any(
                token in name.lower()
                for name in pair
                for token in ("robot0", "gripper0", "panda")
            ):
                continue
            row: dict[str, Any] = {
                "geom_pair": pair,
                "distance_m": float(getattr(contact, "dist", float("nan"))),
            }
            position = getattr(contact, "pos", None)
            if position is not None:
                row["position_xyz"] = (
                    np.asarray(position, dtype=np.float64).round(7).tolist()
                )
            contacts.append(row)
    finally:
        sim.set_state(state)
        sim.forward()

    qpos_after = np.asarray(sim.data.qpos, dtype=np.float64)
    restore_error = float(np.max(np.abs(qpos_after - qpos_before)))
    contacts.sort(key=lambda row: (row["distance_m"], row["geom_pair"]))
    target_name = str(getattr(api, "_capx_v293_active_object", ""))
    target_geom_token = target_name.lower().replace(" ", "_")
    cabinet_contacts = [
        row
        for row in contacts
        if any("wooden_cabinet" in name for name in row["geom_pair"])
    ]
    target_contacts = [
        row
        for row in contacts
        if target_geom_token
        and any(target_geom_token in name.lower() for name in row["geom_pair"])
    ]
    return {
        "schema": "capx_libero_v300a_gt_endpoint_forensic_v1",
        "source": "mujoco_geom_contacts_at_teleported_endpoint",
        "behavior_effect": "audit_only_state_restored_no_physics_step",
        "endpoint_joint_error_max_rad": endpoint_q_error,
        "restored_qpos_error_max": restore_error,
        "fk_crosscheck": fk_crosscheck,
        "target_geom_token": target_geom_token,
        "robot_contact_count": len(contacts),
        "cabinet_contact_count": len(cabinet_contacts),
        "target_contact_count": len(target_contacts),
        "contacts": contacts,
    }
