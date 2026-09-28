"""Algebraic identity: obs TCP forward ↔ tip→hand IK inverse."""

from __future__ import annotations

import numpy as np
import viser.transforms as vtf

from capx.integrations.franka.common import (
    OBS_TCP_TRANSLATION,
    OBS_TCP_YAW_RAD,
    hand_pose_to_obs_tip,
    tip_pose_to_hand_ik_target,
)


def test_tip_hand_roundtrip_matches_obs_tcp() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        hand_pos = rng.normal(size=3)
        # random unit quat wxyz
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        tip_pos, tip_quat = hand_pose_to_obs_tip(hand_pos, q)
        back_pos, back_quat = tip_pose_to_hand_ik_target(tip_pos, tip_quat)
        assert np.allclose(back_pos, hand_pos, atol=1e-9)
        # quaternions are equivalent up to sign
        assert np.allclose(back_quat, q, atol=1e-9) or np.allclose(
            back_quat, -q, atol=1e-9
        )


def test_obs_tcp_constants_match_simulator() -> None:
    assert OBS_TCP_YAW_RAD == np.pi / 2.0
    assert np.allclose(OBS_TCP_TRANSLATION, [0.0, 0.0, -0.107])


def test_legacy_double_offset_lifts_when_tip_points_down() -> None:
    """Document why legacy goto lifts: tip + R@[0,0,-0.1] ≠ inv(obs TCP)."""
    # Hand at origin, pointing world -Z (gripper down): R maps local +Z → world -Z
    # Use a 180° about X: local Z → -world Z
    hand = vtf.SE3.from_rotation_and_translation(
        rotation=vtf.SO3.from_rpy_radians(np.pi, 0.0, 0.0),
        translation=np.zeros(3),
    )
    tip_pos, tip_quat = hand_pose_to_obs_tip(
        hand.translation(), hand.rotation().wxyz
    )
    # Correct IK target ≈ hand
    ik_pos, _ = tip_pose_to_hand_ik_target(tip_pos, tip_quat)
    assert np.allclose(ik_pos, hand.translation(), atol=1e-9)

    # Legacy: tip + R_tip @ [0,0,-0.1]
    tip_xyzw = np.array(
        [tip_quat[1], tip_quat[2], tip_quat[3], tip_quat[0]], dtype=np.float64
    )
    from scipy.spatial.transform import Rotation as SciRotation

    rot = SciRotation.from_quat(tip_xyzw)
    legacy = tip_pos + rot.apply(np.array([0.0, 0.0, -0.1]))
    # Gripper down → legacy IK target is ~0.2 m above the true hand (lift).
    assert legacy[2] - ik_pos[2] > 0.15
