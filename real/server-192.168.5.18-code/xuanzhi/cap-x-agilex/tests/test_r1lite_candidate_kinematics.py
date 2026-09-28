import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciRotation

from capx.integrations.r1lite.kinematics import (
    R1LiteCandidateKinematics,
    candidate_joint_delta_limit,
    max_joint_delta_rad,
    rank_fk_qualified_candidates,
    solve_probes_with_reference_seed,
)


def candidate(
    yaw_deg: float,
    *,
    fk_error_m: float,
    joint_delta_rad: float,
    dual_position_error_m: float = 0.0,
    dual_orientation_error_rad: float = 0.0,
) -> dict:
    return {
        "yaw_deg": yaw_deg,
        "fk_error_m": fk_error_m,
        "max_joint_delta_from_start_rad": joint_delta_rad,
        "dual_urdf_position_error_m": dual_position_error_m,
        "dual_urdf_orientation_error_rad": dual_orientation_error_rad,
    }


def load_coupled_grasp_geometry():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()
    embedded = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(embedded)
    names = {
        "normalized_quat",
        "tool_approach_axis_base",
        "coupled_tcp_target",
        "fingertip_center_from_tcp",
        "require_finite",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "np": np,
        "SciRotation": SciRotation,
        "coupled_grasp_tcp_to_fingertip_m": 0.078,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<geometry>", "exec"),
        namespace,
    )
    return namespace


def test_coupled_grasp_geometry_offsets_tcp_along_selected_tool_axis():
    geometry = load_coupled_grasp_geometry()
    target = geometry["coupled_tcp_target"]
    fingertip = geometry["fingertip_center_from_tcp"]
    axis = geometry["tool_approach_axis_base"]
    ball = np.array([0.4, -0.2, 0.85])

    topdown = np.array([1.0, 0.0, 0.0, 0.0])
    topdown_tcp = target(ball, topdown, 0.0)
    assert topdown_tcp == pytest.approx(ball + np.array([0.0, 0.0, 0.078]))
    assert fingertip(topdown_tcp, topdown) == pytest.approx(ball)

    tilted = (
        SciRotation.from_quat(topdown)
        * SciRotation.from_euler("y", np.deg2rad(45.0))
    ).as_quat()
    tilted_tcp = target(ball, tilted, 0.0)
    assert abs(float(tilted_tcp[0] - ball[0])) > 0.04
    assert fingertip(tilted_tcp, tilted) == pytest.approx(ball)
    assert np.linalg.norm(axis(tilted)) == pytest.approx(1.0)


def test_coupled_grasp_rejects_nonfinite_kinematic_values():
    geometry = load_coupled_grasp_geometry()
    require_finite = geometry["require_finite"]

    with pytest.raises(RuntimeError, match="NaN or Inf"):
        require_finite("candidate", np.array([0.0, np.nan]))
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        require_finite("candidate", np.array([0.0, np.inf]))


def test_local_fk_dual_urdf_agrees_for_candidate_joints():
    kinematics = R1LiteCandidateKinematics()
    result = kinematics.fk(
        np.array([0.31, -0.52, 0.27, 0.19, -0.41, 0.73]),
        np.array([0.08, -0.14, 0.21]),
    )

    assert result.position_base_m.shape == (3,)
    assert result.quaternion_xyzw.shape == (4,)
    assert result.dual_position_error_m < 1e-12
    assert result.dual_orientation_error_rad < 1e-12


def test_async_old_fk_cannot_admit_bad_local_fk_candidate():
    stale_topic_would_appear_good = candidate(
        0.0,
        fk_error_m=0.2345,
        joint_delta_rad=0.1,
    )
    stale_topic_would_appear_good["async_topic_fk_error_m"] = 0.001

    qualified, rejected = rank_fk_qualified_candidates(
        [stale_topic_would_appear_good],
        fk_tolerance_m=0.06,
    )

    assert qualified == []
    assert rejected[0]["reasons"] == ["fk_position_error"]


def test_cup_pregrasp_local_fk_rejects_historically_mispaired_solution():
    kinematics = R1LiteCandidateKinematics()
    torso = np.array(
        [-0.7249000072479248, 1.3738000392913818, 0.9301999807357788]
    )
    target = np.array([0.369125986605484, -0.3517798552469621, 1.058])
    stale_paired_joints = np.array(
        [
            -0.03976469439464763,
            1.8853932453148747,
            -1.0512435207923065,
            0.5571896223294371,
            0.01574952547322336,
            -0.07739952338783798,
        ]
    )
    correlated_joints = np.array(
        [
            -0.036878361047262276,
            2.4950309230137377,
            -2.114889062676666,
            1.458016391036812,
            -0.010050417131555862,
            -0.07592026393322411,
        ]
    )

    stale_fk = kinematics.fk(stale_paired_joints, torso).position_base_m
    correlated_fk = kinematics.fk(correlated_joints, torso).position_base_m

    assert np.linalg.norm(stale_fk - target) > 0.08
    assert np.linalg.norm(correlated_fk - target) < 0.01


def test_continuity_is_ranked_before_fk_error_after_hard_gate():
    close_branch = candidate(45.0, fk_error_m=0.050, joint_delta_rad=0.20)
    distant_branch = candidate(90.0, fk_error_m=0.001, joint_delta_rad=0.30)

    qualified, rejected = rank_fk_qualified_candidates(
        [distant_branch, close_branch],
        fk_tolerance_m=0.06,
    )

    assert rejected == []
    assert [item["yaw_deg"] for item in qualified] == [45.0, 90.0]


def test_dual_urdf_disagreement_is_rejected():
    mismatched = candidate(
        135.0,
        fk_error_m=0.01,
        joint_delta_rad=0.1,
        dual_position_error_m=0.002,
    )

    qualified, rejected = rank_fk_qualified_candidates(
        [mismatched],
        fk_tolerance_m=0.06,
    )

    assert qualified == []
    assert "dual_urdf_position_disagreement" in rejected[0]["reasons"]


def test_every_orientation_probe_receives_identical_reference_seed():
    reference = np.array([0.2, -0.1, 0.4, -0.3, 0.6, -0.5])
    observed_seeds = []

    results, errors = solve_probes_with_reference_seed(
        [0.0, 45.0, 90.0],
        reference,
        seed_solver=lambda seed: observed_seeds.append(seed),
        solve_probe=lambda yaw: yaw,
    )

    assert errors == []
    assert results == [0.0, 45.0, 90.0]
    assert len(observed_seeds) == 3
    assert all(np.array_equal(seed, reference) for seed in observed_seeds)
    assert all(seed is not reference for seed in observed_seeds)


def test_f6_uses_one_radian_limit_and_rejects_incident_branch_jump():
    start = np.array([0.44, -0.2, 0.1, 0.0, 0.2, -0.3])
    incident = start.copy()
    incident[0] = 2.88

    assert candidate_joint_delta_limit(2) == 2.5
    assert candidate_joint_delta_limit(3) == 2.5
    assert candidate_joint_delta_limit(6) == 1.0
    assert candidate_joint_delta_limit(5) is None
    assert max_joint_delta_rad(incident, start) > candidate_joint_delta_limit(6)


def test_pipeline_keeps_f6_strict_xy_completion_condition():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert 'F6_BOX_XY_TOL_M="${F6_BOX_XY_TOL_M:-0.13}"' in script
    assert "box_xy_error_m < f6_box_xy_tol_m" in script


def test_f0_f5_use_four_point_five_cm_but_f6_keeps_three_point_five_cm():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert 'POS_TOL_M="${POS_TOL_M:-0.045}"' in script
    assert 'F6_POS_TOL_M="${F6_POS_TOL_M:-0.035}"' in script
    assert "f6_pos_tol_m = float(sys.argv[18])" in script
    assert "place_waypoint = is_place_waypoint(frame_index, wp)" in script
    assert "f6_pos_tol_m" in script
    assert "if place_waypoint" in script
    assert "else pos_tol_m" in script
    assert "coupled_grasp_fingertip_tol_m" in script
    assert "pos_error_m <= frame_position_tolerance_m" in script


def test_pipeline_uses_eight_cm_local_fk_hard_gate():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert 'CARTESIAN_IK_FK_TOL_M="${CARTESIAN_IK_FK_TOL_M:-0.08}"' in script
    assert "active_fk_tolerance_m if validate_fk else float(\"inf\")" in script
    assert "coupled_grasp_ik_fk_tol_m" in script


def test_pipeline_uses_local_fk_globally_but_f6_only_deterministic_policy():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert "if not place_policy:" in script
    assert '"fix_scope": "legacy_solver_candidate_local_fk"' in script
    assert '"fix_scope": "f6_or_semantic_above_box_only"' in script
    assert '"candidate_local_fk_frame_indices": "all_cartesian_frames"' in script
    assert '{"frame_indices": [6]}' in script
    assert '{"semantic_roles": ["above_box"]}' in script
    assert "get_relaxed_ik_fk_pose_base()" not in script


def test_only_grasp_is_strict_topdown_while_other_frames_keep_tilt_search():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert "if not place_policy:" in script
    assert "def topdown_orientation_probes()" in script
    assert "def topdown_tilt_degrees()" in script
    assert "for probe in active_orientation_probes:" in script
    assert "for tilt_deg in topdown_tilt_degrees():" in script
    assert "if not topdown_only or abs(float(probe[\"tilt_deg\"])) <= 1e-9" in script
    assert (
        'CARTESIAN_TOPDOWN_MAX_TILT_DEG="${CARTESIAN_TOPDOWN_MAX_TILT_DEG:-60}"'
        in script
    )
    assert (
        'CARTESIAN_TOPDOWN_TILT_STEP_DEG="${CARTESIAN_TOPDOWN_TILT_STEP_DEG:-15}"'
        in script
    )
    assert "topdown_only=grasp_waypoint and not coupled_pick_waypoint" in script
    assert '"strict_topdown_yaw_search"' in script
    assert 'else "topdown_yaw_tilt_search"' in script
    assert (
        'else "topdown_yaw_tilt_search_deterministic_seed"'
        in script
    )
    assert (
        '"semantic_grasp": "strict_topdown_yaw_search_tilt_0_deg"'
        in script
    )
    assert (
        '"other_cartesian_frames": ('
        in script
    )


def test_f7_returns_to_runtime_captured_f1_pose_in_joint_space():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert '"event": "first_f1_pose_captured"' in script
    assert '"event": "return_to_first_f1_after_release_begin"' in script
    assert '"mode": "measured_f1_joint_space"' in script
    assert "r1.move_joints_safe(" in script
    assert "first_f1_joints," in script
    assert '"id": "return_to_first_f1_after_release"' in script
    assert "return_to_f0_after_release" not in script


def test_pipeline_defaults_to_two_rounds_but_supports_one_round():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert 'R1LITE_ROUND_COUNT="${R1LITE_ROUND_COUNT:-2}"' in script
    assert '"round_count": round_count' in script
    assert '{"1": "F0-F7", "2": "F1-F7"}' in script
    assert "for round_index in range(1, round_count + 1):" in script
    assert "start_frame_index = 0 if round_index == 1 else 1" in script
    assert '"event": "round_begin"' in script
    assert '"event": "round_finished"' in script
    assert "return_to_first_f1_after_release(frame_index)" in script


def test_round_one_uses_frame_specific_correction_limits_only():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert "first_round_correction_limits = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}" in script
    assert "first_round_correction_limits.get(frame_index)" in script
    assert "for attempt in range(frame_max_corrections + 1):" in script
    assert '"event": "frame_force_accepted_after_correction_limit"' in script
    assert '"force_accept_after_corrections_by_frame": {' in script
    assert '"2": 0,' in script
    assert '"3": 0,' in script
    assert '"4": 0,' in script
    assert '"5": 0,' in script
    assert '"6": 0,' in script
    assert '"preserved_frame_indices": [0, 1, 7]' in script
    assert '"2": "unchanged"' in script


def test_round_one_cartesian_step_is_35cm_while_round_two_stays_20cm():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert 'MAX_COMPENSATION_NORM_M="${MAX_COMPENSATION_NORM_M:-0.20}"' in script
    assert "and use_cartesian_online_ik" in script
    assert "and not use_llm_absolute_20_policy" in script
    assert "planned_frame_delta, frame_cartesian_step_cap_m" in script
    assert "cartesian_delta_to_desired, frame_cartesian_step_cap_m" in script
    assert "correction_gain * pos_error, correction_cap_m" in script
    assert 'else {"1_F2-F6": 0.35, "2": max_compensation_norm_m}' in script


def test_round_one_replays_plan_deltas_while_round_two_stays_absolute():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert 'frame_step_source = "planned_inter_frame_delta"' in script
    assert "planned_frame_delta = desired_position - previous_planned_position" in script
    assert "compensated_target = frame_start_pos + commanded_frame_delta" in script
    assert 'frame_step_source = "next_plan_target_minus_current_actual"' in script
    assert '"1_F2-F6": "current_actual_plus_planned_inter_frame_delta"' in script
    assert '"2": "current_actual_toward_next_planned_absolute_target"' in script


def test_fourth_pipeline_uses_absolute_targets_with_twenty_corrections():
    repo = Path(__file__).resolve().parents[1]
    executor = (
        repo / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()
    launcher = (
        repo / "scripts/run_r1lite_fourth_llm_plan_pipeline.sh"
    ).read_text()

    assert "R1LITE_SINGLE_ROUND_POLICY=llm_absolute_20" in launcher
    assert "MAX_CORRECTIONS=20" in launcher
    assert "CORRECTION_GAIN=0.5" in launcher
    assert "CORRECTION_STEP_CAP_M=0.02" in launcher
    assert 'single_round_policy == "llm_absolute_20"' in executor
    assert "active_round_index == 1 and not use_llm_absolute_20_policy" in executor
    assert '"1_F2-F6": "current_actual_toward_llm_absolute_target"' in executor
    assert '"force_accept": False' in executor
    assert "if use_llm_absolute_20_policy and use_cartesian_online_ik:" in executor
    assert "frame_cartesian_step_cap_m = None" in executor
    assert '{"1_F2-F6": None}' in executor
    assert "capped_for_active_policy(" in executor
    assert "correction_delta_xyz_m" in executor
    assert "correction_delta_norm_m" in executor


def test_pipeline_requires_joint_arrival_before_tcp_error_correction():
    repo = Path(__file__).resolve().parents[1]
    executor = (
        repo / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()
    low_level = (
        repo / "capx/envs/simulators/r1lite_real.py"
    ).read_text()

    assert 'JOINT_ARRIVAL_TOL_RAD="${JOINT_ARRIVAL_TOL_RAD:-0.01}"' in executor
    assert 'JOINT_ARRIVAL_MAX_RETRIES="${JOINT_ARRIVAL_MAX_RETRIES:-3}"' in executor
    assert "settled_joints = r1.wait_for_settled_arm_joints()" in executor
    assert '"event": "frame_joint_arrival_abort"' in executor
    assert '"kinematic_execution_decomposition"' in executor
    assert '"joint_tracking_position_error_m"' in executor
    assert '"vendor_vs_actual_joint_fk_error_m"' in executor
    assert "def wait_for_settled_arm_joints(" in low_level


def test_both_rounds_use_original_planned_target_positions():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert "def planned_position_for_active_round(frame_index: int, wp: dict)" in script
    assert 'return np.asarray(wp["target_ee_position_base_m"]' in script
    assert "position[2] +=" not in script
    assert "previous_planned_position = planned_position_for_active_round(" in script
    assert "frame_index - 1, prev_wp" in script
    assert '"round_target_overrides": {"1": "unchanged", "2": "unchanged"}' in script


def test_round_one_fk_failures_rollback_to_f1_and_retry_at_most_three_times():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert "round1_max_fk_retries = 3" in script
    assert "def is_fk_gate_failure(exc: Exception)" in script
    assert '"event": "round1_fk_failure_rollback_begin"' in script
    assert "return_to_first_f1_after_release(failed_frame_index)" in script
    assert "attempt_start_frame_index = 2" in script
    assert '"event": "round1_fk_retry_begin"' in script
    assert '"event": "round1_fk_retries_exhausted"' in script


def test_post_grasp_fk_failure_waits_for_manual_ball_reset_at_f1():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
    ).read_text()

    assert '"/tmp/capx_r1lite_round1_manual_reset_ready"' in script
    assert '"event": "round1_fk_failure_gripper_open_at_f1"' in script
    assert '"event": "round1_manual_reset_required"' in script
    assert "while not round1_manual_reset_signal_path.exists():" in script
    assert '"event": "round1_manual_reset_confirmed"' in script


def test_pipeline_explicitly_cleans_up_ros_before_interpreter_exit():
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh").read_text()
    low_level = (repo / "capx/envs/simulators/r1lite_real.py").read_text()

    assert "atexit.register(close_r1_at_exit)" in script
    assert '"event": "ros_cleanup_begin"' in script
    assert "r1.close()" in script
    assert "atexit.unregister(close_r1_at_exit)" in script
    assert '"event": "ros_cleanup_finished"' in script
    assert "self._owns_rclpy_context = not rclpy.ok()" in low_level
    assert 'if getattr(self, "_closed", False):' in low_level
    assert "executor.shutdown(timeout_sec=2.0)" in low_level
