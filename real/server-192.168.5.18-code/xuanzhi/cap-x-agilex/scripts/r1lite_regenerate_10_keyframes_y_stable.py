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
PICK01_Y_SOFT_LIMIT_M = 0.08
PICK01_FK_CLOSE_BAND_M = 0.03
PICK01_LIFT_M = 0.05
PICK01_PROBE_LIFTS_M = (0.02, 0.05, 0.08, 0.10)
PICK01_MAX_JOINT_DELTA_RAD = 0.35


def rigid_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr = axis_arr / np.linalg.norm(axis_arr)
    transform[:3, :3] = Rotation.from_rotvec(axis_arr * float(angle)).as_matrix()
    return transform


def arm_joint_points_base(r1: R1LiteRealLowLevel, joints: np.ndarray) -> list[list[float]]:
    """Return right-arm base and six joint-origin points in base_link."""
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


def unique_quaternions(quaternions: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for quat in quaternions:
        q = np.asarray(quat, dtype=float).reshape(4)
        q = q / np.linalg.norm(q)
        if not any(abs(float(np.dot(q, old))) > 0.999 for old in unique):
            unique.append(q)
    return unique


def slerp_quaternions(start_xyzw: np.ndarray, end_xyzw: np.ndarray, fractions: list[float]) -> list[np.ndarray]:
    start = np.asarray(start_xyzw, dtype=float).reshape(4)
    end = np.asarray(end_xyzw, dtype=float).reshape(4)
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    if float(np.dot(start, end)) < 0.0:
        end = -end
    slerp = Slerp([0.0, 1.0], Rotation.from_quat([start, end]))
    return [slerp([float(f)]).as_quat()[0] for f in fractions]


def solve_candidate(
    r1: R1LiteRealLowLevel,
    position: np.ndarray,
    quaternion: np.ndarray,
    reference_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = r1.get_relaxed_ik_candidate(
        position,
        quaternion,
        timeout_s=3.0,
        reference_joints=reference_joints,
        start_joints=reference_joints,
    )
    time.sleep(0.1)
    fk_pos, fk_quat = r1.get_relaxed_ik_fk_pose_base()
    return joints, fk_pos, fk_quat


def make_waypoint(
    r1: R1LiteRealLowLevel,
    index: int,
    wid: str,
    label: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    joints: np.ndarray,
    status: str,
    fk_position: np.ndarray | None = None,
    fk_quaternion: np.ndarray | None = None,
) -> dict:
    points = arm_joint_points_base(r1, joints)
    if fk_position is not None and fk_quaternion is not None:
        fk_pos = np.asarray(fk_position, dtype=float).reshape(3)
        fk_quat = np.asarray(fk_quaternion, dtype=float).reshape(4)
        fk_position_error_m = float(np.linalg.norm(fk_pos - position))
        fk_orientation_error_rad = quat_error_rad(quaternion, fk_quat)
    else:
        fk_pos = position.copy()
        fk_quat = quaternion.copy()
        fk_position_error_m = 0.0
        fk_orientation_error_rad = 0.0

    return {
        "index": index,
        "id": wid,
        "label": label,
        "mode": status,
        "status": status,
        "target_ee_position_base_m": position.tolist(),
        "target_quaternion_xyzw": (quaternion / np.linalg.norm(quaternion)).tolist(),
        "joint_feedback_solution_rad": joints.tolist(),
        "joint_points_base_m": points,
        "joint_points_y_base_m": [float(p[1]) for p in points],
        "fk_position_base_m": fk_pos.tolist(),
        "fk_quaternion_xyzw_base": fk_quat.tolist(),
        "fk_position_error_m": fk_position_error_m,
        "fk_orientation_error_rad": fk_orientation_error_rad,
    }


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    low_cfg = dict(cfg["env"]["cfg"]["low_level"])
    low_cfg.pop("_target_", None)
    # For offline plan generation, require RelaxedIK's FK echo so the JSON records
    # whether a candidate really matches the target pose.
    low_cfg["r1lite_motion_validation_enabled"] = True
    sig = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {k: v for k, v in low_cfg.items() if k in sig.parameters}
    r1 = R1LiteRealLowLevel(**kwargs)

    previous_plan = json.loads(PLAN.read_text())
    scene = json.loads(SCENE.read_text())
    objects = previous_plan["objects"]
    ball = np.asarray(objects["tennis_ball_center_base_m"], dtype=float)
    drop = np.asarray(objects["drop_position_base_m"], dtype=float)
    box = np.asarray(objects["box_center_base_m"], dtype=float)
    table_z = float(scene.get("table_plane_z_m", objects["table_boundary_base_m"][0][2]))

    current_pos, current_quat = physical_ee_pose(r1)
    current_joints = r1.get_arm_joints()
    # Keep the first lift in the current orientation to avoid an immediate IK branch flip.
    lift_quat = current_quat / np.linalg.norm(current_quat)
    top_down_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    clear_z = max(float(current_pos[2] + 0.10), float(ball[2] + 0.276), float(drop[2] + 0.16))

    targets = [
        ("pick_00_current", "当前初始末端位姿", current_pos, current_quat, current_joints),
        ("pick_01_lift_clear", "抓取段1/6: 可达稳定过渡帧, 关节点Y小范围变化", np.array([current_pos[0], current_pos[1], current_pos[2] + PICK01_LIFT_M]), lift_quat, None),
        ("pick_02_above_ball_high", "抓取段2/6: 移到球上方高位", np.array([ball[0], ball[1], clear_z]), top_down_quat, None),
        ("pick_03_pregrasp", "抓取段3/6: 预抓取高度", np.array([ball[0], ball[1], ball[2] + 0.14]), top_down_quat, None),
        ("pick_04_approach", "抓取段4/6: 接近球心", np.array([ball[0], ball[1], ball[2] + 0.08]), top_down_quat, None),
        ("pick_05_grasp", "抓取段5/6: 抓取位姿", np.array([ball[0], ball[1], ball[2] + 0.045]), top_down_quat, None),
        ("place_01_lift_with_ball", "放置段1/4: 抓后抬起", np.array([ball[0], ball[1], clear_z]), top_down_quat, None),
        ("place_02_above_box", "放置段2/4: 移到盒子上方", np.array([drop[0], drop[1], clear_z]), top_down_quat, None),
        ("place_03_release", "放置段3/4: 释放高度", drop, top_down_quat, None),
        ("place_04_retreat", "放置段4/4: 释放后撤离", np.array([drop[0], drop[1], drop[2] + 0.15]), top_down_quat, None),
    ]

    waypoints: list[dict] = []
    reference = current_joints.copy()
    frame0_y = np.asarray(arm_joint_points_base(r1, current_joints), dtype=float)[:, 1]

    # Regenerate only frame 0 and frame 1. Frames 2-9 stay from the existing
    # IK plan so the total pick segment remains 6 frames and the later grasp /
    # place targets are not perturbed by this transition-frame search.
    for index, (wid, label, position, quat, fixed_joints) in enumerate(targets[:2]):
        fk_pos = None
        fk_quat = None
        if fixed_joints is not None:
            joints = fixed_joints
        elif wid == "pick_01_lift_clear":
            # Frame 1 is a stable transition, not a hard Cartesian lift. Probe
            # several nearby pose targets, then write the selected candidate's
            # actual RelaxedIK FK as the waypoint target. This keeps the pick
            # segment at 6 frames while avoiding a large side swing at frame 1.
            candidate_quats = unique_quaternions(
                slerp_quaternions(lift_quat, top_down_quat, [0.0, 0.25, 0.5, 0.75, 1.0])
                + [
                    lift_quat,
                    top_down_quat,
                    np.asarray(previous_plan["waypoints"][1]["target_quaternion_xyzw"], dtype=float),
                ]
            )
            candidates = []
            candidate_index = 0
            for lift_m in PICK01_PROBE_LIFTS_M:
                probe_position = np.array([current_pos[0], current_pos[1], current_pos[2] + lift_m])
                for candidate_quat in candidate_quats:
                    try:
                        joints, fk_pos, fk_quat = solve_candidate(r1, probe_position, candidate_quat, reference)
                        points_y = np.asarray(arm_joint_points_base(r1, joints), dtype=float)[:, 1]
                        y_l2 = float(np.linalg.norm(points_y - frame0_y))
                        y_max = float(np.max(np.abs(points_y - frame0_y)))
                        joint_delta = float(np.max(np.abs(joints - reference)))
                        probe_fk_error = float(np.linalg.norm(fk_pos - probe_position))
                        z_delta = float(fk_pos[2] - current_pos[2])
                        candidates.append(
                            {
                                "candidate_index": candidate_index,
                                "probe_lift_m": float(lift_m),
                                "z_delta": z_delta,
                                "y_l2": y_l2,
                                "y_max": y_max,
                                "joint_delta": joint_delta,
                                "probe_fk_error": probe_fk_error,
                                "joints": joints,
                                "quat": fk_quat,
                                "fk_pos": fk_pos,
                                "fk_quat": fk_quat,
                            }
                        )
                    except Exception as exc:
                        print(f"[warn] pick_01 candidate failed: {exc!r}")
                    candidate_index += 1
            if not candidates:
                raise RuntimeError("No IK candidate for pick_01_lift_clear")

            pool = [
                c
                for c in candidates
                if c["y_max"] <= PICK01_Y_SOFT_LIMIT_M
                and c["joint_delta"] <= PICK01_MAX_JOINT_DELTA_RAD
                and c["probe_fk_error"] <= PICK01_FK_CLOSE_BAND_M
                and c["z_delta"] > 0.0
            ]
            if not pool:
                raise RuntimeError(
                    "No safe pick_01 candidate satisfies TCP/FK, joint continuity, "
                    "positive lift and Y-drift constraints"
                )
            pool.sort(
                key=lambda c: (
                    abs(c["z_delta"] - PICK01_LIFT_M),
                    c["joint_delta"],
                    c["probe_fk_error"],
                    c["y_max"],
                )
            )
            selected = pool[0]
            for c in sorted(candidates, key=lambda item: (item["y_max"], abs(item["z_delta"] - PICK01_LIFT_M)))[:10]:
                print(
                    "[pick_01 candidate]",
                    f"idx={c['candidate_index']}",
                    f"probe_lift={c['probe_lift_m']:.3f}",
                    f"z_delta={c['z_delta']:.4f}",
                    f"probe_fk_error={c['probe_fk_error']:.4f}",
                    f"y_max={c['y_max']:.4f}",
                    f"y_l2={c['y_l2']:.4f}",
                    f"joint_delta={c['joint_delta']:.4f}",
                )

            y_l2 = selected["y_l2"]
            y_max = selected["y_max"]
            joint_delta = selected["joint_delta"]
            probe_fk_error = selected["probe_fk_error"]
            joints = selected["joints"]
            quat = selected["quat"]
            fk_pos = selected["fk_pos"]
            fk_quat = selected["fk_quat"]
            position = fk_pos.copy()
            print(
                "[pick_01 selection]",
                f"idx={selected['candidate_index']}",
                f"probe_lift={selected['probe_lift_m']:.3f}",
                f"z_delta={selected['z_delta']:.4f}",
                f"y_l2={y_l2:.4f}",
                f"y_max={y_max:.4f}",
                f"joint_delta={joint_delta:.4f}",
                f"probe_fk_error={probe_fk_error:.4f}",
            )
        else:
            joints, fk_pos, fk_quat = solve_candidate(r1, position, quat, reference)

        wp = make_waypoint(
            r1,
            index,
            wid,
            label,
            position,
            quat,
            joints,
            "current_feedback" if index == 0 else "ik_solution",
            fk_position=fk_pos,
            fk_quaternion=fk_quat,
        )
        if wid == "pick_01_lift_clear":
            y = np.asarray(wp["joint_points_y_base_m"], dtype=float)
            wp["y_stability_reference_frame"] = "pick_00_current"
            wp["joint_points_y_delta_from_pick_00_m"] = (y - frame0_y).tolist()
            wp["joint_points_y_l2_delta_from_pick_00_m"] = float(np.linalg.norm(y - frame0_y))
            wp["joint_points_y_max_delta_from_pick_00_m"] = float(np.max(np.abs(y - frame0_y)))
        waypoints.append(wp)
        reference = joints.copy()

    for wp in previous_plan["waypoints"][2:]:
        copied = dict(wp)
        copied["index"] = len(waypoints)
        waypoints.append(copied)

    out = dict(previous_plan)
    out["generated_at_unix"] = time.time()
    out["execution"] = "offline_ik_only_no_robot_motion_commands_sent"
    out["notes"] = [
        "Regenerated from current physical frame 0.",
        "All stored waypoint quaternions use right_gripper_tool_z in XYZW order.",
        f"Frame 1 probes nearby small lifts up to {max(PICK01_PROBE_LIFTS_M):.3f}m and selects a reachable FK transition with small right-arm joint-point Y drift from frame 0.",
        f"Frame 1 must satisfy max joint delta {PICK01_MAX_JOINT_DELTA_RAD:.3f}rad and FK position error {PICK01_FK_CLOSE_BAND_M:.3f}m.",
        "Frame 1 target pose is the selected candidate's actual RelaxedIK FK pose, so it is a reachable transition frame.",
        "Frames 2-9 are preserved from the existing IK plan.",
        "This script only probes RelaxedIK and reads feedback; it does not command arm or gripper motion.",
    ]
    out["objects"] = {
        **objects,
        "tennis_ball_center_base_m": ball.tolist(),
        "box_center_base_m": box.tolist(),
        "drop_position_base_m": drop.tolist(),
        "table_plane_z_m": table_z,
    }
    out["waypoints"] = waypoints

    backup = PLAN.with_name(f"{PLAN.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{PLAN.suffix}")
    shutil.copy2(PLAN, backup)
    PLAN.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[backup] {backup}")
    print(f"[wrote] {PLAN}")
    print("[summary]")
    for wp in waypoints:
        print(wp["index"], wp["id"], "pos=", [round(x, 4) for x in wp["target_ee_position_base_m"]])
    r1.close()


if __name__ == "__main__":
    main()
