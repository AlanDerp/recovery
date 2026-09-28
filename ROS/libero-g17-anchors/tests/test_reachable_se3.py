from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from capx.integrations.franka import reachable_se3 as reach
from capx.integrations.franka.trusted_se3 import TrustedSe3Plan, TrustedSe3Result


def _result(
    *,
    ok: bool,
    status: str,
    pos_err: float,
    ori_err: float,
) -> TrustedSe3Result:
    return TrustedSe3Result(
        ok=ok,
        status=status,
        target_position=(0.0, 0.0, 0.0),
        achieved_position=(0.0, 0.0, 0.0),
        final_position_error_m=pos_err,
        final_orientation_error_rad=ori_err,
        executed_waypoints=1,
        preflight_rejections=0,
        max_candidate_arm_joint_delta_rad=0.1,
        max_executed_arm_joint_delta_rad=0.1,
        nonprogress_events=0,
        message=status,
    )


def _plan(*, ok: bool, status: str, waypoints: int = 1) -> TrustedSe3Plan:
    cfg = np.zeros(8)
    return TrustedSe3Plan(
        ok=ok,
        status=status,
        start_cfg=cfg,
        final_cfg=cfg,
        waypoint_cfgs=tuple(cfg.copy() for _ in range(waypoints)),
        preflight_rejections=0,
        max_arm_joint_delta_rad=0.1,
        final_position_error_m=0.0 if ok else 0.05,
        final_orientation_error_rad=0.0 if ok else 0.7,
        message=status,
    )


def test_controller_floor_is_classified_separately_without_recovery(monkeypatch) -> None:
    api = SimpleNamespace(
        position=np.array([0.6, 0.2, 0.1]),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    calls = []

    monkeypatch.setattr(reach, "_live_actuated_cfg", lambda api: np.zeros(8))
    monkeypatch.setattr(
        reach, "fk_panda_hand", lambda api, cfg: (api.position.copy(), api.wxyz.copy())
    )

    def move(api, position, wxyz, *, log):
        calls.append((position.copy(), wxyz.copy()))
        return _result(
            ok=False,
            status="execution_nonprogress",
            pos_err=0.0055,
            ori_err=0.008,
        )

    monkeypatch.setattr(reach, "move_hand_pose_trusted", move)
    result = reach.move_hand_pose_reachable(
        api, np.array([0.6, 0.2, 0.1]), np.array([1.0, 0, 0, 0]), log=False
    )
    assert result.ok
    assert result.status == "controller_floor_accepted"
    assert result.selected_route is None
    assert len(calls) == 1


def test_joint_limit_uses_high_translate_then_rotate_route(monkeypatch) -> None:
    api = SimpleNamespace(
        position=np.array([0.55, 0.10, 0.20]),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    scripted = [
        _result(ok=False, status="local_unreachable", pos_err=0.06, ori_err=0.7),
        *[
            _result(ok=True, status="completed", pos_err=0.001, ori_err=0.005)
            for _ in range(4)
        ],
    ]

    monkeypatch.setattr(reach, "_live_actuated_cfg", lambda api: np.zeros(8))
    monkeypatch.setattr(
        reach, "fk_panda_hand", lambda api, cfg: (api.position.copy(), api.wxyz.copy())
    )

    def move(api, position, wxyz, *, log):
        calls.append((position.copy(), wxyz.copy()))
        result = scripted.pop(0)
        if result.ok:
            api.position = position.copy()
            api.wxyz = wxyz.copy()
        return result

    monkeypatch.setattr(reach, "move_hand_pose_trusted", move)
    target_position = np.array([0.72, 0.21, 0.10])
    target_wxyz = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])
    result = reach.move_hand_pose_reachable(
        api, target_position, target_wxyz, log=False
    )
    assert result.ok and result.status == "recovered"
    assert result.selected_route == "transit_a"
    assert len(calls) == 5
    transit_z = 0.30
    # lift, translate at old orientation, rotate in place, descend
    assert np.allclose(calls[1][0], [0.55, 0.10, transit_z])
    assert np.allclose(calls[2][0], [0.72, 0.21, transit_z])
    assert np.allclose(calls[2][1], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(calls[3][0], calls[2][0])
    assert np.allclose(calls[3][1], target_wxyz)
    assert np.allclose(calls[4][0], target_position)


def test_failed_first_route_replans_second_route_from_measured_state(monkeypatch) -> None:
    api = SimpleNamespace(
        position=np.array([0.50, 0.00, 0.20]),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    calls = []
    # primary fail; route A lift succeeds then translate fails; route B's four
    # legs succeed.  Route B must start from the measured lifted state.
    scripted = [
        _result(ok=False, status="local_unreachable", pos_err=0.08, ori_err=0.8),
        _result(ok=True, status="completed", pos_err=0.001, ori_err=0.001),
        _result(ok=False, status="local_unreachable", pos_err=0.05, ori_err=0.7),
        *[
            _result(ok=True, status="completed", pos_err=0.001, ori_err=0.001)
            for _ in range(4)
        ],
    ]
    monkeypatch.setattr(reach, "_live_actuated_cfg", lambda api: np.zeros(8))
    monkeypatch.setattr(
        reach, "fk_panda_hand", lambda api, cfg: (api.position.copy(), api.wxyz.copy())
    )

    def move(api, position, wxyz, *, log):
        calls.append((position.copy(), wxyz.copy()))
        result = scripted.pop(0)
        if result.ok:
            api.position = position.copy()
            api.wxyz = wxyz.copy()
        return result

    monkeypatch.setattr(reach, "move_hand_pose_trusted", move)
    result = reach.move_hand_pose_reachable(
        api,
        np.array([0.70, 0.20, 0.10]),
        np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)]),
        log=False,
    )
    assert result.ok and result.selected_route == "transit_b"
    labels = [label for label, _ in result.recovery_stages]
    assert labels == [
        "transit_a:lift",
        "transit_a:translate",
        "transit_b:lift",
        "transit_b:rotate",
        "transit_b:translate",
        "transit_b:goal",
    ]
    assert calls[3][0][2] > calls[1][0][2]


def test_contract_preserves_trusted_backend() -> None:
    contract = reach.reachable_se3_contract()
    assert contract["schema"] == "capx_libero_reachable_se3_contract_v1"
    assert contract["trusted_backend"] == "capx_libero_trusted_se3_contract_v1"
    assert contract["max_transit_routes"] == 2


def test_full_path_failure_selects_preflighted_neutral_before_execution(
    monkeypatch,
) -> None:
    api = SimpleNamespace(
        position=np.array([0.65, 0.10, 0.22]),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    calls = []
    primary_plan = _plan(ok=False, status="local_unreachable", waypoints=12)
    leg_a = _plan(ok=True, status="completed", waypoints=16)
    leg_b = _plan(ok=True, status="completed", waypoints=25)
    neutral_pose = np.array([0.46, 0.0, 0.36, 1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(reach, "_live_actuated_cfg", lambda api: np.zeros(8))
    monkeypatch.setattr(
        reach, "fk_panda_hand", lambda api, cfg: (api.position.copy(), api.wxyz.copy())
    )
    monkeypatch.setattr(reach, "plan_hand_pose_trusted", lambda *a, **k: primary_plan)
    monkeypatch.setattr(
        reach,
        "_preflight_neutral_route",
        lambda *a, **k: ("neutral_0", neutral_pose, leg_a, leg_b),
    )

    def move(api, position, wxyz, *, log):
        calls.append((position.copy(), wxyz.copy()))
        api.position = position.copy()
        api.wxyz = wxyz.copy()
        return _result(ok=True, status="completed", pos_err=0.001, ori_err=0.002)

    monkeypatch.setattr(reach, "move_hand_pose_trusted", move)
    target_position = np.array([0.72, 0.20, 0.10])
    target_wxyz = np.array([0.0, 0.0, 1.0, 0.0])
    result = reach.move_hand_pose_reachable(
        api, target_position, target_wxyz, log=False
    )
    assert result.ok and result.selected_route == "neutral_0"
    assert result.primary is None
    assert result.primary_plan is primary_plan
    assert len(calls) == 2
    assert np.allclose(calls[0][0], neutral_pose[:3])
    assert np.allclose(calls[1][0], target_position)
