from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from capx.integrations.franka import planned_se3 as planned
from capx.integrations.franka.trusted_se3 import TrustedSe3Plan


def _plan(start: np.ndarray, final: np.ndarray, *, ok: bool = True) -> TrustedSe3Plan:
    return TrustedSe3Plan(
        ok=ok,
        status="completed" if ok else "local_unreachable",
        start_cfg=start.copy(),
        final_cfg=final.copy() if ok else start.copy(),
        waypoint_cfgs=(final.copy(),) if ok else (),
        preflight_rejections=0 if ok else 4,
        max_arm_joint_delta_rad=float(np.max(np.abs(final[:7] - start[:7]))) if ok else 0.2,
        final_position_error_m=0.0 if ok else 0.1,
        final_orientation_error_rad=0.0 if ok else 0.7,
        message="ok" if ok else "dead end",
    )


class _Env:
    def __init__(self, api: SimpleNamespace) -> None:
        self.api = api
        self.commands: list[np.ndarray] = []

    def move_to_joints_blocking(self, joints: np.ndarray, *, max_steps: int) -> None:
        assert max_steps == 40
        self.api.live[:7] = np.asarray(joints)
        self.commands.append(self.api.live.copy())


def _api() -> SimpleNamespace:
    api = SimpleNamespace(
        live=np.zeros(8),
        cfg=np.zeros(8),
        _robot=SimpleNamespace(
            joints=SimpleNamespace(num_actuated_joints=8)
        ),
    )
    api._env = _Env(api)
    api.get_observation = lambda: {"robot_joint_pos": api.live[:7].copy()}
    return api


def _fk(api: SimpleNamespace, cfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(cfg)
    return q[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0])


def test_executor_uses_proved_joint_waypoints_with_bounded_commands(monkeypatch) -> None:
    api = _api()
    monkeypatch.setattr(planned, "fk_panda_hand", _fk)
    final = np.array([0.50, -0.30, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
    plan = _plan(api.live, final)
    result = planned.execute_trusted_plan(
        api, plan, final[:3], np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )
    assert result.ok, result.message
    assert np.allclose(api.live, final)
    path = np.vstack([np.zeros(8), api._env.commands])
    assert np.max(np.abs(np.diff(path[:, :7], axis=0))) <= 0.180001


def test_neutral_second_leg_is_replanned_from_measured_joint_state(monkeypatch) -> None:
    api = _api()
    monkeypatch.setattr(planned, "fk_panda_hand", _fk)
    original = api.live.copy()
    neutral = planned._neutral_cfg_candidates(original)[0]
    target_cfg = neutral.copy()
    target_cfg[0] = 0.40
    target_position = target_cfg[:3].copy()
    calls: list[np.ndarray] = []

    def planner(api, position, wxyz, *, start_cfg):
        start = np.asarray(start_cfg).copy()
        calls.append(start)
        if len(calls) == 1:
            return _plan(start, start, ok=False)  # direct path
        if len(calls) == 2:
            return _plan(start, neutral)  # first neutral leg
        if len(calls) == 3:
            return _plan(neutral, target_cfg)  # ideal route screening
        assert np.allclose(start, api.live)
        assert np.allclose(start, neutral)
        return _plan(start, target_cfg)  # measured-state proof

    monkeypatch.setattr(planned, "plan_hand_pose_trusted", planner)
    result = planned.move_hand_pose_planned(
        api, target_position, np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )
    assert result.ok and result.status == "recovered"
    assert result.selected_route == "neutral_0"
    assert len(calls) == 4
    assert np.allclose(api.live, target_cfg)


def test_failed_measured_second_leg_rolls_back_without_target_execution(
    monkeypatch,
) -> None:
    api = _api()
    monkeypatch.setattr(planned, "fk_panda_hand", _fk)
    original = api.live.copy()
    neutral = planned._neutral_cfg_candidates(original)[0]
    call_count = 0

    def planner(api, position, wxyz, *, start_cfg):
        nonlocal call_count
        call_count += 1
        start = np.asarray(start_cfg).copy()
        if call_count == 1:
            return _plan(start, start, ok=False)
        if call_count == 2:
            return _plan(start, neutral)
        if call_count == 3:
            target = neutral.copy()
            target[0] = 0.4
            return _plan(neutral, target)
        return _plan(start, start, ok=False)

    monkeypatch.setattr(planned, "plan_hand_pose_trusted", planner)
    result = planned.move_hand_pose_planned(
        api, np.array([0.4, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )
    assert not result.ok
    assert result.rollback_performed
    assert np.allclose(api.live, original)
    assert result.final_position_error_m == 0.4


def test_v281_joins_measured_state_to_preflighted_second_joint_path(
    monkeypatch,
) -> None:
    api = _api()
    monkeypatch.setattr(planned, "fk_panda_hand", _fk)
    monkeypatch.setenv("CAPX_LIBERO_G17_V281_JOINT_JOIN", "1")
    original = api.live.copy()
    neutral = planned._neutral_cfg_candidates(original)[0]
    target_cfg = neutral.copy()
    target_cfg[0] = 0.40
    calls = 0

    def planner(api, position, wxyz, *, start_cfg):
        nonlocal calls
        calls += 1
        start = np.asarray(start_cfg).copy()
        if calls == 1:
            return _plan(start, start, ok=False)
        if calls == 2:
            return _plan(start, neutral)
        if calls == 3:
            return _plan(neutral, target_cfg)  # proved second joint path
        return _plan(start, start, ok=False)  # measured IK false negative

    monkeypatch.setattr(planned, "plan_hand_pose_trusted", planner)
    result = planned.move_hand_pose_planned(
        api,
        target_cfg[:3],
        np.array([1.0, 0.0, 0.0, 0.0]),
        log=False,
    )
    assert result.ok and result.status == "recovered"
    assert result.stages[-1][0] == "neutral_0:goal_joint_join"
    assert "measured_replan=local_unreachable" in result.message
    assert not result.rollback_performed
    assert np.allclose(api.live, target_cfg)
