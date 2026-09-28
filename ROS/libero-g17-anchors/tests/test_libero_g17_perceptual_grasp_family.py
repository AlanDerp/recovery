import numpy as np
import pytest

from capx.integrations.franka import perceptual_grasp_family as family


def test_orientation_family_is_deterministic_and_spans_top_down_yaw() -> None:
    base = np.array([-0.37749348, 0.57099235, 0.72859167, 0.02491174])
    first = family._orientation_family(base)
    second = family._orientation_family(base)
    assert [name for name, _ in first] == [name for name, _ in second]
    top_down = [
        quat for name, quat in first if name.startswith("top_down_tip_z_up_")
    ]
    assert len(top_down) == 8
    for w, x, y, z in top_down:
        local_z = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )
        np.testing.assert_allclose(local_z, [0.0, 0.0, 1.0], atol=1e-7)
    for (_, lhs), (_, rhs) in zip(first, second, strict=True):
        np.testing.assert_allclose(lhs, rhs)
        assert np.linalg.norm(lhs) == pytest.approx(1.0)


def test_probe_is_opt_in_and_declared_audit_only(monkeypatch) -> None:
    monkeypatch.delenv("CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE", raising=False)
    assert not family.enabled()
    monkeypatch.setenv("CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE", "1")
    assert family.enabled()
    assert "never changes" in (family.__doc__ or "")


def test_drawer_entry_family_sweeps_horizontal_azimuth_and_roll() -> None:
    rows = family._drawer_entry_orientations()
    drawer = [quat for name, quat in rows if name.startswith("drawer_entry_")]
    assert len(drawer) == 20
    directions = []
    for w, x, y, z in drawer:
        local_z = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )
        assert local_z[2] == pytest.approx(0.0, abs=1e-7)
        directions.append(tuple(np.round(local_z, 6)))
    assert len(set(directions)) == 5


def test_drawer_entry_tip_position_moves_along_local_z() -> None:
    tip = np.array([0.77, -0.13, 0.19])
    name, quat = family._drawer_entry_orientations()[0]
    rows = family._tip_position_family(tip, name, quat)
    assert len(rows) == 3
    assert [row[0] for row in rows] == pytest.approx([0.06, 0.06, 0.06])
    assert [row[1][2] - tip[2] for row in rows] == pytest.approx(
        [0.0, -0.02, -0.04]
    )
    np.testing.assert_allclose(
        family._tip_position_family(
            tip, "graspnet", np.array([1.0, 0.0, 0.0, 0.0])
        )[0][1],
        tip,
    )


def test_nullspace_probe_fixes_target_and_covers_joint_limits(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V299A_NULLSPACE_PROBE", "1")
    rows = family._drawer_entry_orientations()
    assert [name for name, _ in rows] == ["drawer_entry_azimuth_05_roll_01"]
    tip = np.array([0.77, -0.13, 0.19])
    position_rows = family._tip_position_family(tip, *rows[0])
    assert len(position_rows) == 1
    assert position_rows[0][0] == pytest.approx(0.06)
    assert position_rows[0][1][2] == pytest.approx(0.17)

    class Joints:
        lower_limits = np.array([-2.9] * 7 + [0.0])
        upper_limits = np.array([2.9] * 7 + [0.08])

    class Robot:
        joints = Joints()

    class Api:
        _robot = Robot()

    seeds = family._nullspace_seeds(Api(), np.zeros(8), sample_count=64)
    assert len(seeds) == 68
    halton = np.stack([seed[:7] for name, seed in seeds if name.startswith("halton")])
    assert np.all(halton > -2.9)
    assert np.all(halton < 2.9)
    # Every joint receives broad deterministic coverage, not merely joint 7.
    assert np.all(np.ptp(halton, axis=0) > 4.5)


def test_diagonal_entry_lifts_wrist_without_lifting_contact(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V299B_DIAGONAL_ENTRY_PROBE", "1")
    rows = family._orientation_family(np.array([1.0, 0.0, 0.0, 0.0]))
    assert len(rows) == 5 * 5 * 4
    tip = np.array([0.773, -0.126, 0.191])
    elevations = set()
    for name, quat in rows:
        local_z = family.vtf.SO3(quat) @ np.array([0.0, 0.0, 1.0])
        elevations.add(round(float(np.rad2deg(np.arcsin(local_z[2])))))
        position_rows = family._tip_position_family(tip, name, quat)
        assert len(position_rows) == 1
        contact = position_rows[0][1]
        assert contact[2] == pytest.approx(tip[2] - 0.02)
        assert np.linalg.norm((contact - tip)[:2]) == pytest.approx(0.06)
    assert elevations == {15, 30, 45, 60, 75}


def test_correct_frame_places_hand_outside_contact(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V301A_CORRECT_GRASP_FRAME_PROBE", "1")
    rows = family._orientation_family(np.array([1.0, 0.0, 0.0, 0.0]))
    assert len(rows) == 12 * 4
    center = np.array([0.77, -0.13, 0.19])
    for name, quat in rows:
        [(offset, contact)] = family._tip_position_family(center, name, quat)
        hand, hand_quat = family._v286_grasp_pose_to_hand_target(contact, quat)
        inward = family.vtf.SO3(quat) @ np.array([0.0, 0.0, 1.0])
        outward = -inward
        assert offset == pytest.approx(0.06)
        assert np.dot((contact - center)[:2], outward[:2]) > 0.059
        # The hand lies behind the contact, toward the opening/outside.
        np.testing.assert_allclose(hand - contact, 0.1 * outward, atol=1e-7)
        np.testing.assert_allclose(hand_quat, quat)


def test_v286_topdown_places_hand_above_contact(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V302A_TOPDOWN_GRASP_PROBE", "1")
    rows = family._orientation_family(np.array([1.0, 0.0, 0.0, 0.0]))
    assert len(rows) == 8
    center = np.array([0.77, -0.13, 0.19])
    for name, quat in rows:
        local_z = family.vtf.SO3(quat) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(local_z, [0.0, 0.0, -1.0], atol=1e-7)
        position_rows = family._tip_position_family(center, name, quat)
        assert [offset for offset, _ in position_rows] == pytest.approx(
            [0.0, 0.02, 0.04]
        )
        for _, contact in position_rows:
            hand, hand_quat = family._v286_grasp_pose_to_hand_target(contact, quat)
            np.testing.assert_allclose(hand - contact, [0.0, 0.0, 0.1], atol=1e-7)
            np.testing.assert_allclose(hand_quat, quat)


def test_recommended_topdown_pose_requires_contact_height_and_clear_path() -> None:
    def candidate(z, endpoint, clear, dq, name):
        return {
            "orientation": name,
            "seed": "measured",
            "tip_xyz": [0.77, -0.13, 0.19 + z],
            "tip_wxyz": [0.0, 1.0, 0.0, 0.0],
            "tip_z_offset_m": z,
            "joint_delta_norm_rad": dq,
            "pt_swept_path": {
                "predicted_arm_obstacle_path_clear": clear,
                "scene_arm_endpoint_min_distance_m": endpoint,
                "scene_arm_min_distance_m": 0.01,
            },
        }

    result = family._recommended_topdown_pose(
        [
            candidate(0.04, 0.09, True, 1.0, "lifted"),
            candidate(0.0, 0.04, False, 1.0, "blocked"),
            candidate(0.0, 0.02, True, 1.0, "clear_low"),
            candidate(0.0, 0.03, True, 2.0, "clear_best"),
        ]
    )
    assert result is not None
    assert result["orientation"] == "clear_best"
    assert result["tip_z_offset_m"] == pytest.approx(0.0)
    assert result["source"] == "robot_rgbd_plus_measured_joints_plus_urdf"


def test_v312_qualifies_endpoint_not_imaginary_current_to_contact_path(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN", "1")
    candidate = {
        "orientation": "topdown",
        "seed": "measured",
        "tip_xyz": [0.77, -0.13, 0.19],
        "tip_wxyz": [0.0, 1.0, 0.0, 0.0],
        "tip_z_offset_m": 0.0,
        "joint_delta_norm_rad": 1.0,
        "pt_swept_path": {
            "predicted_arm_obstacle_path_clear": False,
            "scene_arm_endpoint_min_distance_m": 0.012,
            "scene_arm_min_distance_m": -0.04,
        },
        "topdown_transaction": {
            "pregrasp_path_clear": True,
            "ingress_path_clear": True,
            "total_joint_path_norm_rad": 1.0,
        },
    }
    result = family._recommended_topdown_pose([candidate])
    assert result is not None
    assert result["selection"].startswith("contact_height_endpoint_clear")


def test_v312_searches_distinct_graspnet_contacts(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN", "1")
    matrices = np.repeat(np.eye(4)[None, ...], 5, axis=0)
    matrices[:, :3, 3] = np.array(
        [
            [0.70, 0.00, 0.20],
            [0.705, 0.00, 0.20],  # clustered with argmax
            [0.74, 0.00, 0.20],
            [0.78, 0.00, 0.20],
            [0.82, 0.00, 0.20],
        ]
    )
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6])

    def fake_probe(api, one_matrix, one_score, measured, scene):
        x = float(one_matrix[0, 0, 3])
        recommendation = None
        if x >= 0.78:
            recommendation = {
                "tip_xyz": [x, 0.0, 0.2],
                "tip_wxyz": [0.0, 1.0, 0.0, 0.0],
                "pt_swept_path": {
                    "scene_arm_endpoint_min_distance_m": x - 0.75,
                    "scene_arm_min_distance_m": -0.01,
                },
                "topdown_transaction": {
                    "total_joint_path_norm_rad": 1.0 - x,
                },
            }
        return {
            "solve_count": 96,
            "accurate_unique_branch_count": 8,
            "recommended_pose": recommendation,
        }

    monkeypatch.setattr(family, "probe", fake_probe)
    report = family.probe_topdown_contacts(
        object(), matrices, scores, np.zeros(8), object(), max_contacts=4
    )
    assert report["contact_indices"] == [0, 2, 3]
    assert report["recommended_pose"]["graspnet_index"] == 3
    assert report["solve_count"] == 3 * 96
    assert report["early_stop"] is True
    assert report["early_stop_reason"] == "first_complete_topdown_transaction"


def test_proximal_target_half_faces_robot_tip() -> None:
    x = np.linspace(0.60, 0.80, 101)
    points = np.column_stack([x, np.zeros_like(x), np.full_like(x, 0.2)])
    selected, report = family.proximal_target_points(
        points, np.array([0.30, 0.0, 0.4]), keep_fraction=0.50
    )
    assert len(selected) == 51
    assert selected[:, 0].max() <= 0.70 + 1e-9
    assert report["toward_robot_xy"] == [-1.0, 0.0]
    assert report["mode"] == "proximal_horizontal_surface"
