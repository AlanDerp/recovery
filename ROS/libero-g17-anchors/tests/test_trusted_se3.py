from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from capx.integrations.franka import trusted_se3 as se3


class _Env:
    def __init__(self, api: "_Api") -> None:
        self.api = api
        self.history: list[np.ndarray] = []

    def move_to_joints_blocking(self, joints: np.ndarray, *, max_steps: int) -> None:
        assert max_steps == 40
        self.api.live[:7] = np.asarray(joints)
        self.history.append(self.api.live.copy())


class _Solver:
    def __init__(self, api: "_Api", reject_large_translation: bool = False) -> None:
        self.api = api
        self.reject_large_translation = reject_large_translation

    def solve_ik_local(self, **kwargs: object) -> np.ndarray:
        prev = np.asarray(kwargs["prev_cfg"], dtype=np.float64).copy()
        pos = np.asarray(kwargs["target_position"], dtype=np.float64)
        quat = np.asarray(kwargs["target_wxyz"], dtype=np.float64)
        candidate = prev.copy()
        candidate[:3] = pos
        candidate[3:7] = quat
        if self.reject_large_translation and np.linalg.norm(pos - prev[:3]) > 0.011:
            candidate[0] = prev[0] + 0.5
        return candidate


class _Api:
    def __init__(self, *, reject_large_translation: bool = False) -> None:
        self.live = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.02])
        joints = SimpleNamespace(
            num_actuated_joints=8,
            lower_limits=np.full(8, -10.0),
            upper_limits=np.full(8, 10.0),
        )
        self._robot = SimpleNamespace(joints=joints)
        self._target_link_name = "panda_hand"
        self.cfg = self.live.copy()
        self._pks = _Solver(self, reject_large_translation)
        self._env = _Env(self)

    def get_observation(self) -> dict[str, np.ndarray]:
        return {"robot_joint_pos": self.live[:7].copy()}


@pytest.fixture(autouse=True)
def _fake_fk(monkeypatch: pytest.MonkeyPatch) -> None:
    def fk(api: _Api, cfg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = api.live if cfg is None else np.asarray(cfg)
        return q[:3].copy(), q[3:7].copy()

    monkeypatch.setattr(se3, "fk_panda_hand", fk)


def test_shortest_slerp_treats_antipodes_as_same_rotation() -> None:
    q = np.array([0.5, 0.5, 0.5, 0.5])
    mid = se3._slerp_shortest(q, -q, 0.5)
    assert se3._quaternion_angle_rad(mid, q) < 1e-9


def test_far_v272_like_target_is_monotonic_and_branch_local() -> None:
    api = _Api()
    target_position = np.array([0.19280362, -0.01606171, -0.20683716])
    target_wxyz = np.array([-0.34075639, 0.60183579, -0.62495569, -0.36208998])
    result = se3.move_hand_pose_trusted(
        api, target_position, target_wxyz, log=False
    )
    assert result.ok, result.message
    assert result.executed_waypoints > 10
    assert result.max_executed_arm_joint_delta_rad <= 0.2002
    points = np.array([row[:3] for row in api._env.history])
    start = np.zeros(3)
    path = np.vstack([start, points])
    remaining = np.linalg.norm(target_position[None, :] - path, axis=1)
    assert np.all(np.diff(remaining) < 1e-9)
    first = path[1] - path[0]
    assert float(np.dot(first, target_position - start)) > 0.0


def test_preflight_rejection_backs_off_without_executing_bad_candidate() -> None:
    api = _Api(reject_large_translation=True)
    result = se3.move_hand_pose_trusted(
        api, np.array([0.040, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )
    assert result.ok, result.message
    assert result.preflight_rejections >= 1
    assert api._env.history
    assert max(abs(np.diff([0.0] + [float(q[0]) for q in api._env.history]))) <= 0.011


def test_full_path_plan_is_nonmutating_and_matches_trusted_endpoint() -> None:
    api = _Api()
    target_position = np.array([0.12, -0.04, 0.08])
    target_wxyz = np.array([np.cos(0.35), 0.0, 0.0, np.sin(0.35)])
    start = api.live.copy()
    plan = se3.plan_hand_pose_trusted(
        api, target_position, target_wxyz, start_cfg=start
    )
    assert plan.ok, plan.message
    assert len(plan.waypoint_cfgs) > 1
    assert np.array_equal(api.live, start)
    assert not api._env.history
    final_position, final_wxyz = se3.fk_panda_hand(api, plan.final_cfg)
    assert np.linalg.norm(final_position - target_position) <= 0.003
    assert se3._quaternion_angle_rad(final_wxyz, target_wxyz) <= 0.025


def test_full_path_plan_rejection_executes_nothing() -> None:
    api = _Api()

    def impossible(**kwargs: object) -> np.ndarray:
        prev = np.asarray(kwargs["prev_cfg"], dtype=np.float64).copy()
        prev[0] += 0.5
        return prev

    api._pks.solve_ik_local = impossible
    plan = se3.plan_hand_pose_trusted(
        api, np.array([0.04, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
    )
    assert not plan.ok
    assert plan.status == "local_unreachable"
    assert plan.preflight_rejections > 0
    assert not api._env.history


def test_invalid_quaternion_executes_nothing() -> None:
    api = _Api()
    with pytest.raises(ValueError, match="quaternion"):
        se3.move_hand_pose_trusted(api, np.zeros(3), np.zeros(4), log=False)
    assert not api._env.history


def test_contract_names_hard_limits() -> None:
    contract = se3.trusted_se3_contract()
    assert contract["schema"] == "capx_libero_trusted_se3_contract_v1"
    assert contract["max_arm_joint_delta_rad"] == 0.2
    assert contract["max_translation_step_m"] == 0.02
