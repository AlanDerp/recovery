"""Audit-only collision-aware grasp-orientation and IK-branch probe.

The probe asks a narrow feasibility question before implementing a planner:
does the same perceptually grounded GraspNet contact admit an accurate IK
branch whose arm endpoint and swept path avoid the observed non-target scene?
It never changes the grasp returned to the policy or sends a robot command.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import viser.transforms as vtf

from capx.integrations.franka.common import tip_pose_to_hand_ik_target
from capx.integrations.franka.measured_seed_ik import (
    _candidate_metrics,
    _fallback_seeds,
    _quat_multiply_wxyz,
    _validate_candidate,
)


# Shared GraspReady→ContactReached interface.  Candidate qualification and
# command verification must use the same Cartesian contract.
ACQUISITION_POSITION_TOLERANCE_M = 0.010
ACQUISITION_ORIENTATION_TOLERANCE_RAD = 0.100


def enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE", "0") == "1"


def nullspace_probe_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V299A_NULLSPACE_PROBE", "0") == "1"


def diagonal_entry_probe_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V299B_DIAGONAL_ENTRY_PROBE", "0") == "1"


def correct_grasp_frame_probe_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V301A_CORRECT_GRASP_FRAME_PROBE", "0") == "1"


def topdown_grasp_probe_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V302A_TOPDOWN_GRASP_PROBE", "0") == "1"


def topdown_grasp_behavior_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V303_TOPDOWN_GRASP_BEHAVIOR", "0") == "1"


def multi_contact_topdown_enabled() -> bool:
    return (
        os.environ.get(
            "CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN", "0"
        )
        == "1"
    )


def _recommended_topdown_pose(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select a contact-height pose without simulator state.

    Before v312, ``clear`` meant that a joint-linear interpolation from the
    *current* arm state all the way to contact was collision-free.  That is
    not the path executed by a top-down acquisition (pregrasp, then ingress),
    and rejected valid endpoints whenever the imaginary shortcut crossed the
    cabinet.  v312 therefore treats endpoint clearance as the contact-pose
    qualification.  The actual approach and ingress remain command-level
    obligations of the acquisition transaction.
    """
    def qualified(item: dict[str, Any]) -> bool:
        path = item["pt_swept_path"]
        if multi_contact_topdown_enabled():
            transaction = item.get("topdown_transaction") or {}
            return bool(
                float(path["scene_arm_endpoint_min_distance_m"]) > 0.0
                and transaction.get("pregrasp_path_clear")
                and transaction.get("ingress_path_clear")
            )
        return bool(path["predicted_arm_obstacle_path_clear"])

    eligible = [
        item
        for item in candidates
        if abs(float(item.get("tip_z_offset_m", np.inf))) <= 1e-8
        and qualified(item)
    ]
    if not eligible:
        return None
    if multi_contact_topdown_enabled():
        selected = min(
            eligible,
            key=lambda item: (
                float(
                    item["topdown_transaction"]["total_joint_path_norm_rad"]
                ),
                -float(
                    item["pt_swept_path"][
                        "scene_arm_endpoint_min_distance_m"
                    ]
                ),
            ),
        )
    else:
        selected = max(
            eligible,
            key=lambda item: (
                item["pt_swept_path"]["scene_arm_endpoint_min_distance_m"],
                item["pt_swept_path"]["scene_arm_min_distance_m"],
                -float(item["joint_delta_norm_rad"]),
            ),
        )
    return {
        "source": "robot_rgbd_plus_measured_joints_plus_urdf",
        "selection": (
            "contact_height_endpoint_clear_then_maximum_PT_arm_clearance"
            if multi_contact_topdown_enabled()
            else "contact_height_then_maximum_PT_arm_clearance"
        ),
        "orientation": selected["orientation"],
        "seed": selected["seed"],
        "tip_xyz": selected["tip_xyz"],
        "tip_wxyz": selected["tip_wxyz"],
        "tip_z_offset_m": selected["tip_z_offset_m"],
        "pt_swept_path": selected["pt_swept_path"],
        "topdown_transaction": selected.get("topdown_transaction"),
    }


def probe_topdown_contacts(
    api: Any,
    grasp_matrices: np.ndarray,
    grasp_scores: np.ndarray,
    measured_cfg: np.ndarray,
    perceptual_scene: Any,
    *,
    max_contacts: int = 4,
    min_tip_separation_m: float = 0.012,
) -> dict[str, Any]:
    """Search several distinct GraspNet contacts for a top-down endpoint.

    The legacy probe rotates only the single GraspNet argmax.  If that XYZ is
    in the bowl interior or against the cabinet, eight yaw samples still test
    the same bad contact.  This wrapper preserves GraspNet as the contact
    generator while searching a small, score-ordered set of geometrically
    distinct contacts.  All inputs are robot-visible point clouds, measured
    joints and the URDF; no simulator pose or contact truth is consumed.
    """
    matrices = np.asarray(grasp_matrices, dtype=np.float64).reshape(-1, 4, 4)
    scores = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
    selected_indices: list[int] = []
    selected_tips: list[np.ndarray] = []
    for raw_index in np.argsort(-scores):
        index = int(raw_index)
        tip = np.asarray(
            vtf.SE3.from_matrix(matrices[index]).wxyz_xyz[-3:], dtype=np.float64
        )
        if any(
            np.linalg.norm(tip - prior) < float(min_tip_separation_m)
            for prior in selected_tips
        ):
            continue
        selected_indices.append(index)
        selected_tips.append(tip)
        if len(selected_indices) >= int(max_contacts):
            break

    reports: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    for index in selected_indices:
        report = probe(
            api,
            matrices[index : index + 1],
            scores[index : index + 1],
            measured_cfg,
            perceptual_scene,
        )
        report["source_graspnet_index"] = int(index)
        report["source_graspnet_score"] = float(scores[index])
        reports.append(report)
        recommended = report.get("recommended_pose")
        if recommended is not None:
            recommendation = dict(recommended)
            recommendation["graspnet_index"] = int(index)
            recommendation["graspnet_score"] = float(scores[index])
            recommendations.append(recommendation)
            # Score-ordered cascade: a complete pregrasp→contact proof is
            # sufficient.  Searching lower-scored contacts after that only
            # multiplies IK/collision cost and cannot improve correctness.
            break

    recommended_pose = None
    if recommendations:
        recommended_pose = min(
            recommendations,
            key=lambda item: (
                item["topdown_transaction"]["total_joint_path_norm_rad"],
                -item["graspnet_score"],
                -item["pt_swept_path"]["scene_arm_endpoint_min_distance_m"],
            ),
        )
        recommended_pose["selection"] = (
            "distinct_GraspNet_contacts_then_topdown_endpoint_clearance"
        )

    return {
        "schema": "capx_libero_v312_multi_contact_topdown_probe_v1",
        "source": "graspnet_contacts_plus_robot_rgbd_plus_measured_joints_plus_urdf",
        "behavior_effect": "select_topdown_contact",
        "probe_mode": "distinct_GraspNet_contacts_x_topdown_yaw",
        "contact_indices": [
            int(row["source_graspnet_index"]) for row in reports
        ],
        "contact_count": len(reports),
        "early_stop": bool(recommendations),
        "early_stop_reason": (
            "first_complete_topdown_transaction"
            if recommendations
            else "contact_budget_exhausted"
        ),
        "solve_count": int(sum(row.get("solve_count", 0) for row in reports)),
        "accurate_unique_branch_count": int(
            sum(row.get("accurate_unique_branch_count", 0) for row in reports)
        ),
        "recommended_pose": recommended_pose,
        "contact_reports": reports,
    }


def proximal_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V297A_PROXIMAL_CONTACT_PROBE", "0") == "1"


def proximal_target_points(
    target_points: np.ndarray,
    robot_tip_xyz: np.ndarray,
    *,
    keep_fraction: float = 0.55,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the target surface half facing the current robot tip.

    This uses only the robot's point cloud and proprioceptive tip position.  It
    is a contact-generation proposal, not a simulator identity or collision
    signal.
    """
    points = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not 0.0 < float(keep_fraction) <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if len(points) < 32 or keep_fraction == 1.0:
        return points.copy(), {
            "input_points": int(len(points)),
            "kept_points": int(len(points)),
            "keep_fraction": float(keep_fraction),
            "mode": "all_points_small_cloud",
        }
    center = np.median(points, axis=0)
    toward_robot = np.asarray(robot_tip_xyz, dtype=np.float64).reshape(3) - center
    toward_robot[2] = 0.0
    norm = float(np.linalg.norm(toward_robot))
    if norm <= 1e-8:
        raise ValueError("robot tip and target center have identical XY")
    toward_robot /= norm
    projection = (points - center) @ toward_robot
    cutoff = float(np.quantile(projection, 1.0 - float(keep_fraction)))
    selected = points[projection >= cutoff]
    return selected, {
        "input_points": int(len(points)),
        "kept_points": int(len(selected)),
        "keep_fraction": float(keep_fraction),
        "target_median_xyz": center.round(7).tolist(),
        "robot_tip_xyz": np.asarray(robot_tip_xyz, dtype=np.float64).round(7).tolist(),
        "toward_robot_xy": toward_robot[:2].round(7).tolist(),
        "projection_cutoff_m": cutoff,
        "projection_min_m": float(np.min(projection)),
        "projection_max_m": float(np.max(projection)),
        "mode": "proximal_horizontal_surface",
    }


def _drawer_entry_orientations() -> tuple[tuple[str, np.ndarray], ...]:
    """Sweep horizontal entry azimuth and finger roll independently.

    ``tip_pose_to_hand_ik_target`` places the hand origin along tip-local +Z.
    Hence local +Z is the outward direction from contact to wrist.  The drawer
    opening normal is not assumed to point toward the robot: this PT-only probe
    enumerates it and lets observed swept clearance identify the opening.
    """
    local_x_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rows = []
    # v298d found its least-bad endpoint clearances in the contiguous opening
    # sector 4..8.  Restrict the Z audit to that observed sector so Z is the
    # only new variable.
    azimuth_indices = (5,) if nullspace_probe_enabled() else range(4, 9)
    for azimuth_index in azimuth_indices:
        azimuth = azimuth_index * 2.0 * np.pi / 12.0
        local_z_world = np.array(
            [np.cos(azimuth), np.sin(azimuth), 0.0], dtype=np.float64
        )
        local_y_world = np.cross(local_z_world, local_x_world)
        canonical = np.asarray(
            vtf.SO3.from_matrix(
                np.column_stack(
                    [local_x_world, local_y_world, local_z_world]
                )
            ).wxyz,
            dtype=np.float64,
        )
        rolls = tuple(enumerate(np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)))
        if nullspace_probe_enabled():
            rolls = (rolls[1],)
        for roll_index, roll in rolls:
            local_z_roll = np.array(
                [np.cos(roll / 2.0), 0.0, 0.0, np.sin(roll / 2.0)],
                dtype=np.float64,
            )
            rows.append(
                (
                    f"drawer_entry_azimuth_{azimuth_index:02d}_roll_{roll_index:02d}",
                    _quat_multiply_wxyz(canonical, local_z_roll),
                )
            )
    return tuple(rows)


def _diagonal_entry_orientations() -> tuple[tuple[str, np.ndarray], ...]:
    """Keep the contact in the shelf while lifting the wrist out of it."""
    rows = []
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for azimuth_index in range(4, 9):
        azimuth = azimuth_index * 2.0 * np.pi / 12.0
        for elevation_deg in (15, 30, 45, 60, 75):
            elevation = np.deg2rad(elevation_deg)
            local_z_world = np.array(
                [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ],
                dtype=np.float64,
            )
            local_x_world = world_up - np.dot(world_up, local_z_world) * local_z_world
            local_x_world /= np.linalg.norm(local_x_world)
            local_y_world = np.cross(local_z_world, local_x_world)
            canonical = np.asarray(
                vtf.SO3.from_matrix(
                    np.column_stack([local_x_world, local_y_world, local_z_world])
                ).wxyz,
                dtype=np.float64,
            )
            for roll_index, roll in enumerate(
                np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
            ):
                local_z_roll = np.array(
                    [np.cos(roll / 2.0), 0.0, 0.0, np.sin(roll / 2.0)],
                    dtype=np.float64,
                )
                rows.append(
                    (
                        "diagonal_entry_"
                        f"azimuth_{azimuth_index:02d}_"
                        f"elevation_{elevation_deg:02d}_roll_{roll_index:02d}",
                        _quat_multiply_wxyz(canonical, local_z_roll),
                    )
                )
    return tuple(rows)


def _correct_grasp_frame_orientations() -> tuple[tuple[str, np.ndarray], ...]:
    """Sweep opening directions with local +Z pointing from hand into shelf.

    v286 maps the rotated GraspNet pose to ``panda_hand`` by translating
    10 cm along pose-local -Z.  Therefore local +Z is hand→contact (inward),
    the opposite of the v298-v300 probe assumption.
    """
    rows = []
    local_x_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for outward_index in range(12):
        azimuth = outward_index * 2.0 * np.pi / 12.0
        outward_world = np.array(
            [np.cos(azimuth), np.sin(azimuth), 0.0], dtype=np.float64
        )
        local_z_world = -outward_world
        local_y_world = np.cross(local_z_world, local_x_world)
        canonical = np.asarray(
            vtf.SO3.from_matrix(
                np.column_stack([local_x_world, local_y_world, local_z_world])
            ).wxyz,
            dtype=np.float64,
        )
        for roll_index, roll in enumerate(
            np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
        ):
            local_z_roll = np.array(
                [np.cos(roll / 2.0), 0.0, 0.0, np.sin(roll / 2.0)],
                dtype=np.float64,
            )
            rows.append(
                (
                    f"correct_frame_outward_{outward_index:02d}_roll_{roll_index:02d}",
                    _quat_multiply_wxyz(canonical, local_z_roll),
                )
            )
    return tuple(rows)


def _v286_topdown_orientations() -> tuple[tuple[str, np.ndarray], ...]:
    """Top-down grasps in the frozen v286 GraspNet pose convention.

    v286 places ``panda_hand`` 10 cm along pose-local -Z.  Therefore a wrist
    above the observed contact requires pose-local +Z to point down.  The
    remaining degree of freedom is finger yaw about world Z.
    """
    tool_z_down = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    rows = []
    for yaw_index, yaw in enumerate(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)):
        world_yaw = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
            dtype=np.float64,
        )
        rows.append(
            (
                f"v286_topdown_z_down_yaw_{yaw_index:02d}",
                _quat_multiply_wxyz(world_yaw, tool_z_down),
            )
        )
    return tuple(rows)


def _orientation_family(
    base_wxyz: np.ndarray,
    tip_xyz: np.ndarray | None = None,
) -> tuple[tuple[str, np.ndarray], ...]:
    base = np.asarray(base_wxyz, dtype=np.float64).reshape(4)
    base /= np.linalg.norm(base)
    if topdown_grasp_probe_enabled():
        return _v286_topdown_orientations()
    if correct_grasp_frame_probe_enabled():
        return _correct_grasp_frame_orientations()
    if diagonal_entry_probe_enabled():
        return _diagonal_entry_orientations()
    rows: list[tuple[str, np.ndarray]] = [
        ("graspnet", base),
        (
            "graspnet_tool_z_pi",
            _quat_multiply_wxyz(base, np.array([0.0, 0.0, 0.0, 1.0])),
        ),
    ]
    # ``goto_pose`` now consumes the observation-tip frame exactly.  Its hand
    # origin is +0.107 m along tip-local +Z, so a top-down grasp must have
    # local +Z pointing upward (hand above tip).  The legacy [0,0,1,0]
    # convention points local +Z downward and puts the wrist below the object.
    canonical = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for index, yaw in enumerate(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)):
        world_yaw = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
            dtype=np.float64,
        )
        rows.append(
            (
                f"top_down_tip_z_up_world_yaw_{index:02d}",
                _quat_multiply_wxyz(world_yaw, canonical),
            )
        )
    if tip_xyz is not None:
        rows.extend(_drawer_entry_orientations())
    unique: list[tuple[str, np.ndarray]] = []
    for name, quat in rows:
        if not any(abs(float(np.dot(quat, prior))) >= 1.0 - 1e-7 for _, prior in unique):
            unique.append((name, quat))
    return tuple(unique)


def _tip_position_family(
    tip_position: np.ndarray,
    orientation_name: str,
    orientation_wxyz: np.ndarray,
) -> tuple[tuple[float, np.ndarray], ...]:
    """Probe observation-grounded near-side contacts for drawer entry.

    v298c established that 6 cm is the first accurate horizontal IK depth.  For
    every candidate opening azimuth, move the distal GraspNet tip 6 cm along
    local +Z to a near-side contact while keeping wrist and approach collinear.
    """
    tip = np.asarray(tip_position, dtype=np.float64).reshape(3)
    if orientation_name.startswith("v286_topdown_z_down_"):
        offsets = (0.0, 0.05, 0.10) if multi_contact_topdown_enabled() else (
            0.0,
            0.02,
            0.04,
        )
        return tuple(
            (float(z_offset), tip + np.array([0.0, 0.0, z_offset]))
            for z_offset in offsets
        )
    if orientation_name.startswith("correct_frame_outward_"):
        rotation = vtf.SO3(
            np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
        )
        inward = np.asarray(rotation @ np.array([0.0, 0.0, 1.0]))
        outward = -inward
        outward[2] = 0.0
        outward /= np.linalg.norm(outward)
        contact = tip + 0.06 * outward + np.array([0.0, 0.0, -0.02])
        return ((0.06, contact),)
    if orientation_name.startswith("diagonal_entry_"):
        rotation = vtf.SO3(
            np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
        )
        outward = np.asarray(rotation @ np.array([0.0, 0.0, 1.0]))
        outward[2] = 0.0
        outward /= np.linalg.norm(outward)
        # The contact location is held at the PT-observed near bowl surface;
        # elevation changes only the wrist position, not the contact height.
        contact = tip + 0.06 * outward + np.array([0.0, 0.0, -0.02])
        return ((0.06, contact),)
    if not orientation_name.startswith("drawer_entry_"):
        return ((0.0, tip.copy()),)
    rotation = vtf.SO3(
        np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
    )
    outward = np.asarray(rotation @ np.array([0.0, 0.0, 1.0]))
    near_side = tip + 0.06 * outward
    rows = tuple(
        (0.06, near_side + np.array([0.0, 0.0, z_offset]))
        for z_offset in (0.0, -0.02, -0.04)
    )
    if nullspace_probe_enabled():
        return (rows[1],)
    return rows


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, digit = divmod(index, base)
        denominator *= base
        value += digit / denominator
    return value


def _nullspace_seeds(
    api: Any, measured_cfg: np.ndarray, *, sample_count: int = 64
) -> tuple[tuple[str, np.ndarray], ...]:
    """Deterministically cover Panda joint space to expose redundant IK branches.

    This changes optimizer initialization only.  Every returned solution must
    still meet the exact same Cartesian target and is subsequently scored by
    the observation-only collision audit.
    """
    measured = np.asarray(measured_cfg, dtype=np.float64).reshape(-1)
    lower = np.asarray(api._robot.joints.lower_limits, dtype=np.float64).reshape(-1)
    upper = np.asarray(api._robot.joints.upper_limits, dtype=np.float64).reshape(-1)
    arm_n = min(7, measured.size, lower.size, upper.size)
    if arm_n < 7:
        raise ValueError("Panda nullspace probe requires seven arm joints")
    # Stay away from the hard boundary, where a seed itself can make the
    # nonlinear solve needlessly ill-conditioned.
    lo = lower[:arm_n] + 0.02
    hi = upper[:arm_n] - 0.02
    primes = (2, 3, 5, 7, 11, 13, 17)
    rows = [
        ("measured", measured.copy()),
        *_fallback_seeds(measured),
    ]
    for sample_index in range(1, int(sample_count) + 1):
        unit = np.array(
            [_van_der_corput(sample_index, prime) for prime in primes],
            dtype=np.float64,
        )
        seed = measured.copy()
        seed[:arm_n] = lo + unit * (hi - lo)
        rows.append((f"halton_{sample_index:03d}", seed))
    return tuple(rows)


def _v286_grasp_pose_to_hand_target(
    grasp_position: np.ndarray, grasp_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the frozen v286 legacy GraspNet→PyRoKi hand mapping."""
    quat = np.asarray(grasp_wxyz, dtype=np.float64).reshape(4)
    quat /= np.linalg.norm(quat)
    rotation = vtf.SO3(quat)
    hand_position = np.asarray(grasp_position, dtype=np.float64).reshape(3)
    hand_position = hand_position + np.asarray(
        rotation @ np.array([0.0, 0.0, -0.1], dtype=np.float64)
    )
    return hand_position, quat


def probe(
    api: Any,
    grasp_matrices: np.ndarray,
    grasp_scores: np.ndarray,
    measured_cfg: np.ndarray,
    perceptual_scene: Any,
) -> dict[str, Any]:
    """Probe one best GraspNet contact across orientations and IK seeds."""
    from capx.integrations.franka import perceptual_swept_path as swept

    matrices = np.asarray(grasp_matrices, dtype=np.float64).reshape(-1, 4, 4)
    scores = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
    index = int(np.argmax(scores))
    grasp = vtf.SE3.from_matrix(matrices[index])
    tip_position = np.asarray(grasp.wxyz_xyz[-3:], dtype=np.float64)
    base_quat = np.asarray(grasp.wxyz_xyz[:4], dtype=np.float64)
    measured_cfg = np.asarray(measured_cfg, dtype=np.float64)
    if nullspace_probe_enabled():
        seeds = _nullspace_seeds(api, measured_cfg)
    else:
        seeds = (("measured", measured_cfg.copy()),) + _fallback_seeds(measured_cfg)
    variants = []
    forensic_candidates: list[dict[str, Any]] = []
    accurate_count = 0
    solve_count = 0
    for orientation_name, tip_quat in _orientation_family(
        base_quat, tip_position
    ):
        for tip_offset, candidate_tip_position in _tip_position_family(
            tip_position, orientation_name, tip_quat
        ):
            if correct_grasp_frame_probe_enabled() or topdown_grasp_probe_enabled():
                hand_position, hand_quat = _v286_grasp_pose_to_hand_target(
                    candidate_tip_position, tip_quat
                )
            else:
                hand_position, hand_quat = tip_pose_to_hand_ik_target(
                    candidate_tip_position, tip_quat
                )
            branch_rows = []
            seen_cfgs: list[np.ndarray] = []
            for seed_name, seed in seeds:
                solve_count += 1
                try:
                    cfg = np.asarray(
                        api._pks.solve_ik_seeded(
                            robot=api._robot,
                            target_link_name=api._target_link_name,
                            target_position=hand_position,
                            target_wxyz=hand_quat,
                            initial_cfg=seed,
                        ),
                        dtype=np.float64,
                    ).reshape(-1)
                    _validate_candidate(api, cfg)
                    pos_e, ori_e, max_dq, norm_dq = _candidate_metrics(
                        api,
                        cfg,
                        np.asarray(measured_cfg, dtype=np.float64),
                        hand_position,
                        hand_quat,
                    )
                    accurate = bool(
                        pos_e <= ACQUISITION_POSITION_TOLERANCE_M
                        and ori_e <= ACQUISITION_ORIENTATION_TOLERANCE_RAD
                    )
                    row: dict[str, Any] = {
                        "seed": seed_name,
                        "accurate": accurate,
                        "position_error_m": pos_e,
                        "orientation_error_rad": ori_e,
                        "max_joint_delta_rad": max_dq,
                        "joint_delta_norm_rad": norm_dq,
                    }
                    duplicate = any(
                        np.max(np.abs(cfg[:7] - prior[:7])) <= 1e-4
                        for prior in seen_cfgs
                    )
                    row["duplicate_branch"] = bool(duplicate)
                    if accurate and not duplicate:
                        accurate_count += 1
                        seen_cfgs.append(cfg.copy())
                        if nullspace_probe_enabled():
                            row["arm_cfg_rad"] = cfg[:7].round(7).tolist()
                        row["pt_swept_path"] = swept.audit_candidate_path(
                            api, perceptual_scene, measured_cfg, cfg
                        )
                        forensic_candidates.append(
                            {
                                "orientation": orientation_name,
                                "seed": seed_name,
                                "tip_xyz": np.asarray(candidate_tip_position)
                                .round(7)
                                .tolist(),
                                "tip_z_offset_m": float(
                                    candidate_tip_position[2] - tip_position[2]
                                ),
                                "tip_wxyz": np.asarray(tip_quat).round(7).tolist(),
                                "arm_cfg_rad": cfg[:7].round(7).tolist(),
                                "candidate_cfg": cfg.copy(),
                                "joint_delta_norm_rad": float(norm_dq),
                                "pt_swept_path": row["pt_swept_path"],
                            }
                        )
                    branch_rows.append(row)
                except Exception as exc:
                    branch_rows.append({"seed": seed_name, "error": repr(exc)})
            variants.append(
                {
                    "orientation": orientation_name,
                    "tip_offset_toward_robot_m": float(tip_offset),
                    "tip_z_offset_m": float(
                        candidate_tip_position[2] - tip_position[2]
                    ),
                    "tip_xyz": np.asarray(candidate_tip_position).round(7).tolist(),
                    "tip_wxyz": np.asarray(tip_quat).round(7).tolist(),
                    "tip_target_nearest_distance_m": float(
                        np.min(
                            np.linalg.norm(
                                perceptual_scene.target_points[
                                    : perceptual_scene.target_valid
                                ]
                                - candidate_tip_position,
                                axis=1,
                            )
                        )
                    ),
                    "branches": branch_rows,
                }
            )
    if multi_contact_topdown_enabled() and topdown_grasp_probe_enabled():
        # Bind an exact IK branch for both halves of the motion actually sent
        # by goto_pose(z_approach=0.1): current→pregrasp and
        # pregrasp→contact.  This prevents selection on one IK branch followed
        # by execution on a newly solved elbow branch.
        contacts = [
            item
            for item in forensic_candidates
            if abs(float(item["tip_z_offset_m"])) <= 1e-8
        ]
        pregrasps = [
            item
            for item in forensic_candidates
            if abs(float(item["tip_z_offset_m"]) - 0.10) <= 1e-8
        ]
        for contact in contacts:
            matching = [
                item
                for item in pregrasps
                if item["orientation"] == contact["orientation"]
            ]
            if matching:
                matching = [
                    max(
                        matching,
                        key=lambda item: (
                            item["pt_swept_path"][
                                "predicted_arm_obstacle_path_clear"
                            ],
                            item["pt_swept_path"][
                                "scene_arm_min_distance_m"
                            ],
                            -float(item["joint_delta_norm_rad"]),
                        ),
                    )
                ]
            transactions = []
            for pregrasp in matching:
                ingress = swept.audit_candidate_path(
                    api,
                    perceptual_scene,
                    np.asarray(pregrasp["candidate_cfg"], dtype=np.float64),
                    np.asarray(contact["candidate_cfg"], dtype=np.float64),
                )
                pregrasp_path = pregrasp["pt_swept_path"]
                transactions.append(
                    {
                        "pregrasp_path_clear": bool(
                            pregrasp_path["predicted_arm_obstacle_path_clear"]
                        ),
                        "ingress_path_clear": bool(
                            ingress["predicted_arm_obstacle_path_clear"]
                        ),
                        "pregrasp_tip_xyz": pregrasp["tip_xyz"],
                        "pregrasp_arm_cfg_rad": np.asarray(
                            pregrasp["candidate_cfg"], dtype=np.float64
                        )[:7]
                        .round(8)
                        .tolist(),
                        "contact_arm_cfg_rad": np.asarray(
                            contact["candidate_cfg"], dtype=np.float64
                        )[:7]
                        .round(8)
                        .tolist(),
                        "pregrasp_joint_path_norm_rad": float(
                            pregrasp["joint_delta_norm_rad"]
                        ),
                        "ingress_joint_path_norm_rad": float(
                            np.linalg.norm(
                                np.asarray(
                                    contact["candidate_cfg"], dtype=np.float64
                                )[:7]
                                - np.asarray(
                                    pregrasp["candidate_cfg"], dtype=np.float64
                                )[:7]
                            )
                        ),
                        "total_joint_path_norm_rad": float(
                            pregrasp["joint_delta_norm_rad"]
                            + np.linalg.norm(
                                np.asarray(
                                    contact["candidate_cfg"], dtype=np.float64
                                )[:7]
                                - np.asarray(
                                    pregrasp["candidate_cfg"], dtype=np.float64
                                )[:7]
                            )
                        ),
                        "pregrasp_pt_swept_path": pregrasp_path,
                        "ingress_pt_swept_path": ingress,
                        "pregrasp_seed": pregrasp["seed"],
                        "contact_seed": contact["seed"],
                    }
                )
            if transactions:
                contact["topdown_transaction"] = max(
                    transactions,
                    key=lambda item: (
                        item["pregrasp_path_clear"]
                        and item["ingress_path_clear"],
                        item["ingress_pt_swept_path"][
                            "scene_arm_min_distance_m"
                        ],
                        item["pregrasp_pt_swept_path"][
                            "scene_arm_min_distance_m"
                        ],
                    ),
                )
    gt_endpoint_forensic = None
    from capx.integrations.franka import gt_endpoint_forensic as _gt_endpoint

    if _gt_endpoint.enabled() and forensic_candidates:
        if topdown_grasp_probe_enabled():
            # Validate a contact-preserving endpoint, not the artificially
            # safest candidate obtained by lifting the grasp above the bowl.
            selected = max(
                forensic_candidates,
                key=lambda item: (
                    -abs(item["tip_z_offset_m"]),
                    item["pt_swept_path"]["scene_arm_endpoint_min_distance_m"],
                    item["pt_swept_path"]["scene_arm_min_distance_m"],
                ),
            )
        else:
            selected = max(
                forensic_candidates,
                key=lambda item: (
                    item["pt_swept_path"]["scene_arm_endpoint_min_distance_m"],
                    item["pt_swept_path"]["scene_arm_min_distance_m"],
                ),
            )
        gt_endpoint_forensic = {
            key: value
            for key, value in selected.items()
            if key != "candidate_cfg"
        }
        gt_endpoint_forensic["mujoco_endpoint"] = _gt_endpoint.audit_endpoint(
            api, selected["candidate_cfg"]
        )

    recommended_pose = (
        _recommended_topdown_pose(forensic_candidates)
        if topdown_grasp_probe_enabled()
        else None
    )
    return {
        "schema": "capx_libero_v296b_perceptual_grasp_family_probe_v1",
        "source": "graspnet_contact_plus_robot_rgbd_plus_measured_joints_plus_urdf",
        "behavior_effect": "audit_only",
        "probe_mode": (
            "v286_topdown_grasp_height_yaw_sweep"
            if topdown_grasp_probe_enabled()
            else (
                "v286_correct_grasp_frame_opening_sweep"
                if correct_grasp_frame_probe_enabled()
                else (
                    "diagonal_entry_elevation_family"
                    if diagonal_entry_probe_enabled()
                    else (
                        "fixed_target_global_ik_multistart"
                        if nullspace_probe_enabled()
                        else "orientation_position_family"
                    )
                )
            )
        ),
        "graspnet_index": index,
        "graspnet_score": float(scores[index]),
        "tip_xyz": tip_position.round(7).tolist(),
        "orientation_count": len(variants),
        "solve_count": solve_count,
        "accurate_unique_branch_count": accurate_count,
        "gt_endpoint_forensic": gt_endpoint_forensic,
        "recommended_pose": recommended_pose,
        "variants": variants,
    }
