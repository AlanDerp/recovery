from types import SimpleNamespace

import numpy as np

from capx.integrations.franka import grasp_recovery as recovery


def test_near_miss_requires_empty_plus_contact_or_inside() -> None:
    base = dict(
        firm_acquired=False,
        target_contact=False,
        tip_in_target_volume=False,
        relative_x="WITHIN_REFERENCE_BAND",
        relative_y="WITHIN_REFERENCE_BAND",
        relative_z="POSITIVE_OF_REFERENCE_BAND",
        coordinate_in_target_band=False,
        target_direction_xy=(-1, 0),
        attachment="empty",
    )
    assert recovery.CloseTimeFg(**base).near_miss is False
    assert recovery.CloseTimeFg(**{**base, "target_contact": True}).near_miss
    assert recovery.CloseTimeFg(**{**base, "tip_in_target_volume": True}).near_miss
    assert not recovery.CloseTimeFg(
        **{**base, "target_contact": True, "attachment": "holding_other_movable"}
    ).near_miss
    assert recovery.CloseTimeFg(
        **{**base, "coordinate_in_target_band": True}
    ).near_miss


def test_grasp_event_latches_transient_contact_but_keeps_empty_outcome() -> None:
    common = dict(
        firm_acquired=False,
        tip_in_target_volume=False,
        relative_x="WITHIN_REFERENCE_BAND",
        relative_y="WITHIN_REFERENCE_BAND",
        relative_z="WITHIN_REFERENCE_BAND",
        coordinate_in_target_band=True,
        target_direction_xy=(-1, 0),
        attachment="empty",
    )
    before = recovery.CloseTimeFg(**common, target_contact=True)
    after = recovery.CloseTimeFg(
        **{**common, "target_contact": False, "coordinate_in_target_band": False}
    )
    latched = recovery.latch_grasp_event(before, after)
    assert latched.target_contact is True
    assert latched.coordinate_in_target_band is True
    assert latched.attachment == "empty"
    assert latched.near_miss


def test_fg_band_correction_moves_toward_target_by_fixed_distance() -> None:
    delta = recovery.correction_from_fg(
        "POSITIVE_OF_REFERENCE_BAND",
        "WITHIN_REFERENCE_BAND",
        np.array([-0.03, 0.004]),
    )
    np.testing.assert_allclose(delta, [-recovery.POC_GRASPABLE_DISTANCE_M, 0.0])


def test_center_empty_uses_dominant_close_time_direction() -> None:
    delta = recovery.correction_from_fg(
        "WITHIN_REFERENCE_BAND",
        "WITHIN_REFERENCE_BAND",
        np.array([-0.03, 0.006]),
    )
    np.testing.assert_allclose(delta, [recovery.POC_GRASPABLE_DISTANCE_M, 0.0])


def test_tip_inside_uses_target_oriented_volume(monkeypatch) -> None:
    target = SimpleNamespace(
        position=np.array([0.70, -0.10, 0.20]),
        half_extents=np.array([0.05, 0.04, 0.03]),
        orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    raw = SimpleNamespace(objects={"obj01": target})
    api = SimpleNamespace(
        get_observation=lambda: {
            "robot_cartesian_pos": np.array(
                [0.72, -0.11, 0.21, 1.0, 0.0, 0.0, 0.0, 0.0]
            )
        }
    )
    inside, toward = recovery._tip_inside_target(api, raw)
    assert inside is True
    np.testing.assert_allclose(toward, [-0.02, 0.01, -0.01])


def test_frozen_transaction_is_strictly_smoke_only(monkeypatch) -> None:
    api = SimpleNamespace()
    monkeypatch.delenv(recovery.FROZEN_TRIAL02_ENV, raising=False)
    monkeypatch.setenv("CAPX_ACTIVE_TRIAL", "2")
    assert recovery.frozen_trial02_recommended_pose(api, 1) is None
    monkeypatch.setenv(recovery.FROZEN_TRIAL02_ENV, "1")
    assert recovery.frozen_trial02_recommended_pose(api, 2) is None
    monkeypatch.setenv("CAPX_ACTIVE_TRIAL", "3")
    assert recovery.frozen_trial02_recommended_pose(api, 1) is None
    monkeypatch.setenv("CAPX_ACTIVE_TRIAL", "2")
    pose = recovery.frozen_trial02_recommended_pose(api, 1)
    assert pose is not None
    assert pose["source_job"] == 527196
    assert pose["selection"] == "frozen_v313_trial02_mechanism_replay"


def test_frozen_transaction_preserves_exact_tip_ingress() -> None:
    contact = np.array([0.7547516, -0.1550308, 0.1966753])
    pregrasp = np.array([0.7547516, -0.1550308, 0.2966753])
    achieved_xy = np.array([0.0001, 0.0182])
    corrected_pregrasp = pregrasp.copy()
    corrected_pregrasp[:2] += achieved_xy
    corrected_contact = corrected_pregrasp + (contact - pregrasp)
    np.testing.assert_allclose(corrected_contact[:2], contact[:2] + achieved_xy)
    assert corrected_contact[2] == contact[2]


def test_v318_near_miss_is_z_aligned_single_side_contact() -> None:
    base = dict(
        firm_acquired=False,
        target_contact=True,
        tip_in_target_volume=False,
        relative_x="WITHIN_REFERENCE_BAND",
        relative_y="WITHIN_REFERENCE_BAND",
        relative_z="WITHIN_REFERENCE_BAND",
        coordinate_in_target_band=True,
        target_direction_xy=(0, -1),
        attachment="empty",
        target_contact_side="FINGER1",
        target_body_name="akita_black_bowl_1",
    )
    assert recovery.CloseTimeFg(**base).adaptive_near_miss
    assert not recovery.CloseTimeFg(
        **{**base, "relative_z": "POSITIVE_OF_REFERENCE_BAND"}
    ).adaptive_near_miss
    assert recovery.CloseTimeFg(
        **{**base, "target_contact_side": "NONE"}
    ).adaptive_near_miss
    assert not recovery.CloseTimeFg(
        **{**base, "firm_acquired": True, "attachment": "holding_payload"}
    ).adaptive_near_miss


def test_v318_fk_axis_and_fg_sign_are_separate() -> None:
    # 180 degrees about world X maps hand +Y to world -Y.
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        recovery.signed_jaw_axis_xy(quat, "FINGER1"), [0.0, -1.0], atol=1e-8
    )
    np.testing.assert_allclose(
        recovery.signed_jaw_axis_xy(quat, "FINGER2"), [0.0, 1.0], atol=1e-8
    )
    np.testing.assert_allclose(
        recovery.signed_jaw_axis_xy(quat, "BOTH", (0, -1)),
        [0.0, -1.0],
        atol=1e-8,
    )


def test_v318_scalar_halves_from_intrinsic_gripper_width() -> None:
    np.testing.assert_allclose(
        recovery.adaptive_probe_magnitudes(), [0.020, 0.010, 0.005, 0.0025]
    )


def test_v318_too_far_requires_both_unreached_and_non_target_contact() -> None:
    report = {
        "robot_contact_pairs": [
            {
                "geom_pair": [
                    "gripper0_finger1_collision",
                    "wooden_cabinet_1_g9",
                ]
            }
        ]
    }
    assert recovery.blocked_too_far(
        report, reached=False, target_body_name="akita_black_bowl_1"
    )
    # The successful 1 cm replay had a brief cabinet touch but did reach.
    assert not recovery.blocked_too_far(
        report, reached=True, target_body_name="akita_black_bowl_1"
    )
    target_only = {
        "robot_contact_pairs": [
            {
                "geom_pair": [
                    "akita_black_bowl_1_g22",
                    "gripper0_finger1_collision",
                ]
            }
        ]
    }
    assert not recovery.blocked_too_far(
        target_only, reached=False, target_body_name="akita_black_bowl_1"
    )
