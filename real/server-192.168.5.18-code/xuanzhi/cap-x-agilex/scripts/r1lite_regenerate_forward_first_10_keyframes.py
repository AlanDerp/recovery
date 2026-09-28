#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

REPO = Path("/home/nvidia/xuanzhi/cap-x-agilex")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel


CONFIG = REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
PLAN = REPO / "outputs/r1lite_tennis_ball_in_box/offline_ik_pick_place_joint_poses_3d.json"
SCENE = REPO / "outputs/r1lite_tennis_ball_in_box/current_scene_3d.json"


def rigid_transform(xyz, rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def axis_rotation(axis, angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr = axis_arr / np.linalg.norm(axis_arr)
    transform[:3, :3] = Rotation.from_rotvec(axis_arr * float(angle)).as_matrix()
    return transform


def arm_joint_points_base(r1: R1LiteRealLowLevel, joints: np.ndarray) -> list[list[float]]:
    base_from_torso, _ = r1._base_from_torso_link3()
    transform = base_from_torso @ rigid_transform((0.0, -0.335, 0.12306))
    points = [transform[:3, 3].tolist()]
    chain = [
        ((0.0, 0.0, 0.08605), (0.0, 0.0, 1.0)),
        ((0.0, 0.03075, 0.04925), (0.0, 1.0, 0.0)),
        ((-0.3, 0.00025004, 0.0), (0.0, 1.0, 0.0)),
        ((0.1747, 0.00049739, 0.075485), (0.0, 1.0, 0.0)),
        ((0.08, -0.031498, 0.0405), (0.0, 0.0, 1.0)),
        ((0.022503, 0.0, -0.0405), (1.0, 0.0, 0.0)),
    ]
    for joint_value, (origin_xyz, axis_xyz) in zip(joints.tolist(), chain):
        transform = transform @ rigid_transform(origin_xyz)
        points.append(transform[:3, 3].tolist())
        transform = transform @ axis_rotation(axis_xyz, joint_value)
    return points


def physical_ee_pose(r1: R1LiteRealLowLevel) -> tuple[np.ndarray, np.ndarray]:
    """Return the physical pose using RelaxedIK's tool-Z TCP semantic."""
    return r1.get_physical_tool_z_pose()


def quat_error_rad(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    a = np.asarray(a_xyzw, dtype=float).reshape(4)
    b = np.asarray(b_xyzw, dtype=float).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * math.acos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def slerp_quats(start_xyzw: np.ndarray, end_xyzw: np.ndarray, fractions: list[float]) -> list[np.ndarray]:
    start = np.asarray(start_xyzw, dtype=float).reshape(4)
    end = np.asarray(end_xyzw, dtype=float).reshape(4)
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    if float(np.dot(start, end)) < 0:
        end = -end
    slerp = Slerp([0.0, 1.0], Rotation.from_quat([start, end]))
    return [slerp([float(f)]).as_quat()[0] for f in fractions]


def solve_candidate(r1, position, quat, reference_joints):
    joints = r1.get_relaxed_ik_candidate(
        position,
        quat,
        timeout_s=3.0,
        reference_joints=reference_joints,
        start_joints=reference_joints,
    )
    time.sleep(0.08)
    fk_pos, fk_quat = r1.get_relaxed_ik_fk_pose_base()
    return joints, fk_pos, fk_quat


def waypoint(r1, index, wid, label, position, quat, joints, status, fk_pos=None, fk_quat=None):
    if fk_pos is None:
        fk_pos = np.asarray(position, dtype=float)
    if fk_quat is None:
        fk_quat = np.asarray(quat, dtype=float)
    position = np.asarray(position, dtype=float).reshape(3)
    quat = np.asarray(quat, dtype=float).reshape(4)
    joints = np.asarray(joints, dtype=float).reshape(6)
    points = arm_joint_points_base(r1, joints)
    return {
        "index": index,
        "id": wid,
        "label": label,
        "mode": status,
        "status": status,
        "target_ee_position_base_m": position.tolist(),
        "target_quaternion_xyzw": (quat / np.linalg.norm(quat)).tolist(),
        "joint_feedback_solution_rad": joints.tolist(),
        "joint_points_base_m": points,
        "joint_points_y_base_m": [float(p[1]) for p in points],
        "fk_position_base_m": np.asarray(fk_pos, dtype=float).reshape(3).tolist(),
        "fk_quaternion_xyzw_base": np.asarray(fk_quat, dtype=float).reshape(4).tolist(),
        "fk_position_error_m": float(np.linalg.norm(np.asarray(fk_pos, dtype=float).reshape(3) - position)),
        "fk_orientation_error_rad": quat_error_rad(quat, np.asarray(fk_quat, dtype=float).reshape(4)),
    }


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    low_cfg = dict(cfg["env"]["cfg"]["low_level"])
    low_cfg.pop("_target_", None)
    low_cfg["r1lite_motion_validation_enabled"] = True
    sig = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {k: v for k, v in low_cfg.items() if k in sig.parameters}
    r1 = R1LiteRealLowLevel(**kwargs)

    prev = json.loads(PLAN.read_text())
    scene = json.loads(SCENE.read_text())
    objects = dict(prev["objects"])
    ball = np.asarray(scene["tennis_ball"]["position_base_m"], dtype=float)
    drop = np.asarray(objects["drop_position_base_m"], dtype=float)
    box = np.asarray(objects["box_center_base_m"], dtype=float)

    current_pos, current_quat = physical_ee_pose(r1)
    current_joints = r1.get_arm_joints()
    top_down = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    clear_z = max(float(current_pos[2] + 0.10), float(ball[2] + 0.276), float(drop[2] + 0.16))

    waypoints = [
        waypoint(
            r1,
            0,
            "pick_00_current",
            "当前初始末端位姿",
            current_pos,
            current_quat,
            current_joints,
            "current_feedback",
            fk_pos=current_pos,
            fk_quat=current_quat,
        )
    ]

    def choose_segment(
        segment_name: str,
        reference_pos: np.ndarray,
        reference_joints: np.ndarray,
        target_pos: np.ndarray,
        quat_fractions: list[float],
        min_x_progress: float,
        max_joint_delta_preferred: float,
    ) -> dict:
        candidate_quats = slerp_quats(current_quat, top_down, quat_fractions)
        candidates = []
        idx = 0
        for fraction in [0.35, 0.50, 0.65, 0.80, 1.0]:
            probe = reference_pos + fraction * (target_pos - reference_pos)
            for quat in candidate_quats:
                try:
                    joints, fk_pos, fk_quat = solve_candidate(r1, probe, quat, reference_joints)
                except Exception as exc:
                    print(f"[warn] {segment_name} candidate failed idx={idx}: {exc!r}")
                    idx += 1
                    continue
                x_progress = float(fk_pos[0] - reference_pos[0])
                z_drop = max(0.0, float(reference_pos[2] - fk_pos[2]))
                joint_delta = float(np.max(np.abs(joints - reference_joints)))
                candidates.append(
                    {
                        "idx": idx,
                        "joints": joints,
                        "fk_pos": fk_pos,
                        "fk_quat": fk_quat,
                        "x_progress": x_progress,
                        "z_drop": z_drop,
                        "joint_delta": joint_delta,
                        "probe_fk_error": float(np.linalg.norm(fk_pos - probe)),
                    }
                )
                idx += 1
        if not candidates:
            raise RuntimeError(f"No candidate for {segment_name}")
        forward = [c for c in candidates if c["x_progress"] >= min_x_progress and c["z_drop"] < 0.05]
        small_joint = [c for c in forward if c["joint_delta"] <= max_joint_delta_preferred]
        pool = small_joint or forward or candidates
        pool.sort(
            key=lambda c: (
                c["joint_delta"] > max_joint_delta_preferred,
                c["joint_delta"],
                -c["x_progress"],
                c["probe_fk_error"],
            )
        )
        selected = pool[0]
        print(f"[{segment_name} candidates]")
        for c in sorted(candidates, key=lambda item: item["joint_delta"])[:8]:
            print(
                f"idx={c['idx']} x_progress={c['x_progress']:.4f} "
                f"fk={c['fk_pos'].round(4).tolist()} joint_delta={c['joint_delta']:.4f} "
                f"probe_fk_error={c['probe_fk_error']:.4f}"
            )
        print(
            f"[selected {segment_name}]",
            f"idx={selected['idx']}",
            f"x_progress={selected['x_progress']:.4f}",
            f"fk={selected['fk_pos'].round(4).tolist()}",
            f"joint_delta={selected['joint_delta']:.4f}",
        )
        return selected

    direction = ball - current_pos
    ref_pos = current_pos.copy()
    ref_joints = current_joints.copy()
    forward_targets = [
        ("pick_01_forward_small", "抓取段1/9: 小步向前伸出", current_pos + np.array([0.06, 0.015, 0.02]), [0.0, 0.1], 0.015, 0.65),
        ("pick_02_forward_mid", "抓取段2/9: 继续向前伸出", current_pos + np.array([0.12, 0.035, 0.05]), [0.0, 0.15, 0.3], 0.020, 0.75),
        ("pick_03_forward_high", "抓取段3/9: 前伸并抬高", np.array([current_pos[0] + 0.18, current_pos[1] + 0.055, min(clear_z, current_pos[2] + 0.09)]), [0.1, 0.25, 0.4], 0.020, 0.85),
        ("pick_04_above_transition", "抓取段4/9: 过渡到球上方方向", np.array([ball[0] - 0.07, ball[1] - 0.03, clear_z]), [0.25, 0.4, 0.55], 0.020, 1.00),
    ]
    for wid, label, target, quat_fracs, min_x, max_delta in forward_targets:
        selected = choose_segment(wid, ref_pos, ref_joints, target, quat_fracs, min_x, max_delta)
        waypoints.append(
            waypoint(
                r1,
                len(waypoints),
                wid,
                label,
                selected["fk_pos"],
                selected["fk_quat"],
                selected["joints"],
                "ik_solution",
                fk_pos=selected["fk_pos"],
                fk_quat=selected["fk_quat"],
            )
        )
        ref_pos = selected["fk_pos"]
        ref_joints = selected["joints"]

    targets = [
        ("pick_05_above_ball_high", "抓取段5/9: 移到球上方高位", np.array([ball[0], ball[1], clear_z]), top_down),
        ("pick_06_pregrasp_high", "抓取段6/9: 预抓取高位", np.array([ball[0], ball[1], ball[2] + 0.18]), top_down),
        ("pick_07_pregrasp", "抓取段7/9: 预抓取高度", np.array([ball[0], ball[1], ball[2] + 0.12]), top_down),
        ("pick_08_grasp", "抓取段8/9: 抓取位姿", np.array([ball[0], ball[1], ball[2] + 0.045]), top_down),
        ("place_01_lift_with_ball", "放置段1/4: 抓后抬起", np.array([ball[0], ball[1], clear_z]), top_down),
        ("place_02_above_box", "放置段2/4: 移到盒子上方", np.array([drop[0], drop[1], clear_z]), top_down),
        ("place_03_release", "放置段3/4: 释放高度", drop, top_down),
        ("place_04_retreat", "放置段4/4: 释放后撤离", np.array([drop[0], drop[1], drop[2] + 0.15]), top_down),
    ]
    for wid, label, pos, quat in targets:
        joints, fk_pos, fk_quat = solve_candidate(r1, pos, quat, ref_joints)
        waypoints.append(
            waypoint(r1, len(waypoints), wid, label, pos, quat, joints, "ik_solution", fk_pos=fk_pos, fk_quat=fk_quat)
        )
        ref_joints = joints

    out = dict(prev)
    out["generated_at_unix"] = time.time()
    out["execution"] = "offline_ik_only_no_robot_motion_commands_sent"
    out["notes"] = [
        "Forward-first IK strategy regenerated from current physical pose.",
        "Pick segment uses 9 frames before grasp completion: current, four forward transitions, above ball, pregrasp high, pregrasp, grasp.",
        "Forward transition targets are selected RelaxedIK FK poses with smaller per-frame joint deltas.",
        "This script only probes RelaxedIK and reads feedback; it does not command robot motion.",
    ]
    out["objects"] = {
        **objects,
        "tennis_ball_center_base_m": ball.tolist(),
        "box_center_base_m": box.tolist(),
        "drop_position_base_m": drop.tolist(),
    }
    out["waypoints"] = waypoints

    backup = PLAN.with_name(f"{PLAN.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{PLAN.suffix}")
    shutil.copy2(PLAN, backup)
    PLAN.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[backup] {backup}")
    print(f"[wrote] {PLAN}")
    print("[summary]")
    for wp in waypoints:
        print(wp["index"], wp["id"], "pos=", [round(float(x), 4) for x in wp["target_ee_position_base_m"]], "fk_err=", round(float(wp.get("fk_position_error_m", 0.0)), 4))


if __name__ == "__main__":
    main()
