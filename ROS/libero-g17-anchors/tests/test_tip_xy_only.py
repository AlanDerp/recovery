"""Unit tests for tip XY-only primitive (no sim required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from capx.integrations.franka import tip_xy_only as txy


class _FakeJoints:
    num_actuated_joints = 8


class _FakeLinks:
    names = ["panda_link0", "panda_hand"]


class _FakeRobot:
    joints = _FakeJoints()
    links = _FakeLinks()

    def forward_kinematics(self, cfg):
        # Return identity SE3 params for two links; hand at cfg-derived xy.
        import jax.numpy as jnp

        # pyroki/jaxlie SE3 often stored as wxyz_xyz — we only need translation
        # via jaxlie.SE3 in fk_panda_hand. Provide a minimal stub by patching
        # fk in the test instead if this is too heavy.
        raise NotImplementedError


def test_tip_xy_result_acceptance_thresholds() -> None:
    good = txy.TipXyMoveResult(
        ok=True,
        commanded_xy=(0.03, 0.0),
        achieved_xy=(0.029, 0.001),
        tip_z0=0.1,
        tip_z1=0.101,
        d_tip_z=0.001,
        iters=2,
        final_xy_err=0.001,
        message="ok",
    )
    assert good.acceptance_ok
    bad_z = txy.TipXyMoveResult(
        ok=False,
        commanded_xy=(0.03, 0.0),
        achieved_xy=(0.03, 0.0),
        tip_z0=0.1,
        tip_z1=0.15,
        d_tip_z=0.05,
        iters=1,
        final_xy_err=0.0,
        message="lift",
    )
    assert not bad_z.acceptance_ok


def test_local_rejection_preserves_measured_partial_progress() -> None:
    partial = txy.TipXyMoveResult(
        ok=False,
        commanded_xy=(0.0, -0.0081624),
        achieved_xy=(-0.0000136, -0.0060521),
        tip_z0=0.40552,
        tip_z1=0.40548,
        d_tip_z=-0.00004,
        iters=4,
        final_xy_err=0.0021104,
        message="local joint-limit rejection after three committed steps",
        rejection_stage="local_unreachable",
        executed_steps=3,
        status="partial",
    )
    assert not partial.acceptance_ok
    assert partial.usable_partial_progress
    assert partial.measured_forward_progress_m > 0.006
    np.testing.assert_allclose(
        partial.measured_remaining_xy,
        (0.0000136, -0.0021103),
        atol=1e-7,
    )


def test_execution_failure_is_not_usable_partial_progress() -> None:
    failed = txy.TipXyMoveResult(
        ok=False,
        commanded_xy=(0.01, 0.0),
        achieved_xy=(0.004, 0.0),
        tip_z0=0.2,
        tip_z1=0.2,
        d_tip_z=0.0,
        iters=2,
        final_xy_err=0.006,
        message="controller violated the committed step contract",
        rejection_stage="execution",
        executed_steps=1,
        status="partial",
    )
    assert not failed.usable_partial_progress


def test_move_tip_xy_closed_loop_with_mock_api() -> None:
    """Hand XY steps update a fake tip 1:1 (TCP const); loop must converge."""
    tip = np.array([0.5, 0.1, 0.12], dtype=np.float64)
    hand = tip.copy()  # tip==hand for mock (const 0)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cfg = np.zeros(8, dtype=np.float64)

    class FakeApi:
        def __init__(self) -> None:
            self.cfg = cfg.copy()
            self._robot = SimpleNamespace(
                joints=SimpleNamespace(num_actuated_joints=8),
                links=SimpleNamespace(names=["panda_hand"]),
            )
            self._target_link_name = "panda_hand"
            self._env = SimpleNamespace(
                move_to_joints_blocking=lambda j, **_kwargs: None
            )
            self._pks = SimpleNamespace()

            def solve_ik_vel_cost(**kwargs):
                nonlocal hand
                assert np.allclose(kwargs["initial_cfg"], kwargs["prev_cfg"])
                target = np.asarray(kwargs["target_position"], dtype=np.float64)
                hand = target.copy()
                tip[:] = hand  # 1:1
                out = cfg.copy()
                return out

            self._pks.solve_ik_vel_cost = solve_ik_vel_cost

        def get_observation(self):
            return {
                "robot_cartesian_pos": np.concatenate([tip, quat, [0.0]]),
                "robot_joint_pos": cfg[:8].copy(),
            }

    # Patch fk_panda_hand to read mock hand without jaxlie
    real_fk = txy.fk_panda_hand

    def fake_fk(api, cfg=None):
        return hand.copy(), quat.copy()

    txy.fk_panda_hand = fake_fk  # type: ignore
    try:
        api = FakeApi()
        res = txy.move_tip_xy(api, 0.03, 0.0, tol_m=0.001, max_iters=5, log=False)
        assert res.acceptance_ok
        assert abs(res.achieved_xy[0] - 0.03) < 0.002
        assert abs(res.d_tip_z) < 0.001
    finally:
        txy.fk_panda_hand = real_fk  # type: ignore


def test_move_tip_xy_stops_on_discrete_predicate_before_full_command(
    monkeypatch,
) -> None:
    """A legal FG event may terminate a long axis command after one micro-step."""
    live_cfg = np.zeros(8, dtype=np.float64)
    tip = np.array([0.5, 0.1, 0.12], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state: dict[str, np.ndarray] = {}

    class FakeApi:
        cfg = live_cfg.copy()
        _robot = SimpleNamespace(
            joints=SimpleNamespace(num_actuated_joints=8),
            links=SimpleNamespace(names=["panda_hand"]),
        )
        _target_link_name = "panda_hand"

        def __init__(self) -> None:
            self._env = SimpleNamespace(move_to_joints_blocking=self._move)
            self._pks = SimpleNamespace(solve_ik_local=self._solve)

        def get_observation(self):
            return {
                "robot_cartesian_pos": np.concatenate([tip, quat, [0.0]]),
                "robot_joint_pos": live_cfg.copy(),
            }

        def _solve(self, **kwargs):
            target = np.asarray(kwargs["target_position"]).copy()
            candidate = live_cfg.copy()
            candidate[0] += 0.05
            state["target"] = target
            state["candidate"] = candidate
            return candidate

        def _move(self, _joints, **_kwargs):
            tip[:2] = state["target"][:2]
            live_cfg[:] = state["candidate"]

    def fake_fk(_api, cfg=None):
        q = live_cfg if cfg is None else np.asarray(cfg)
        if "candidate" in state and np.allclose(q, state["candidate"]):
            return state["target"].copy(), quat.copy()
        return tip.copy(), quat.copy()

    monkeypatch.setattr(txy, "fk_panda_hand", fake_fk)
    predicate_calls = 0

    def reached() -> bool:
        nonlocal predicate_calls
        predicate_calls += 1
        return True

    result = txy.move_tip_xy(
        FakeApi(),
        0.15,
        0.0,
        max_step_m=0.03,
        stop_predicate=reached,
        log=False,
    )
    assert result.ok
    assert result.status == "predicate_reached"
    assert result.predicate_reached
    assert result.executed_steps == 1
    assert predicate_calls == 1
    assert result.achieved_xy[0] == pytest.approx(0.03, abs=0.002)
    assert not result.acceptance_ok


def test_preflight_uses_live_seed_and_rejects_joint_branch_before_execution(
    monkeypatch,
) -> None:
    """A 1-rad IK branch jump must cause exactly zero simulator motion."""
    live_cfg = np.zeros(8, dtype=np.float64)
    stale_cfg = np.full(8, 2.0, dtype=np.float64)
    tip = np.array([0.5, 0.1, 0.12], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    solved: dict[str, np.ndarray] = {}
    move_calls: list[np.ndarray] = []

    class FakeApi:
        cfg = stale_cfg.copy()
        _robot = SimpleNamespace(
            joints=SimpleNamespace(num_actuated_joints=8),
            links=SimpleNamespace(names=["panda_hand"]),
        )
        _target_link_name = "panda_hand"

        def __init__(self) -> None:
            self._env = SimpleNamespace(move_to_joints_blocking=self._move)
            self._pks = SimpleNamespace(solve_ik_vel_cost=self._solve)

        def get_observation(self):
            return {
                "robot_cartesian_pos": np.concatenate([tip, quat, [0.0]]),
                "robot_joint_pos": live_cfg.copy(),
            }

        def _solve(self, **kwargs):
            solved["prev"] = np.asarray(kwargs["prev_cfg"]).copy()
            solved["initial"] = np.asarray(kwargs["initial_cfg"]).copy()
            solved["target"] = np.asarray(kwargs["target_position"]).copy()
            candidate = live_cfg.copy()
            candidate[0] = 1.0
            solved["candidate"] = candidate
            return candidate

        def _move(self, joints, **_kwargs):
            move_calls.append(np.asarray(joints).copy())

    def fake_fk(_api, cfg=None):
        q = live_cfg if cfg is None else np.asarray(cfg)
        if "candidate" in solved and np.allclose(q, solved["candidate"]):
            return solved["target"].copy(), quat.copy()
        return tip.copy(), quat.copy()

    monkeypatch.setattr(txy, "fk_panda_hand", fake_fk)
    result = txy.move_tip_xy(FakeApi(), 0.03, 0.0, max_iters=2, log=False)

    assert result.rejection_stage == "local_unreachable"
    assert not result.ok
    assert result.executed_steps == 0
    assert not move_calls
    # The seven arm coordinates come from the simulator, not stale api.cfg.
    assert np.allclose(solved["prev"][:7], live_cfg[:7])
    assert np.allclose(solved["initial"], solved["prev"])
    assert result.max_arm_joint_delta == 1.0


def test_z_drift_is_diagnostic_and_partial_xy_completes(monkeypatch) -> None:
    live_cfg = np.zeros(8, dtype=np.float64)
    tip = np.array([0.5, 0.1, 0.12], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state: dict[str, np.ndarray] = {}
    move_calls = 0

    class FakeApi:
        cfg = live_cfg.copy()
        _robot = SimpleNamespace(
            joints=SimpleNamespace(num_actuated_joints=8),
            links=SimpleNamespace(names=["panda_hand"]),
        )
        _target_link_name = "panda_hand"

        def __init__(self) -> None:
            self._env = SimpleNamespace(move_to_joints_blocking=self._move)
            self._pks = SimpleNamespace(solve_ik_vel_cost=self._solve)

        def get_observation(self):
            return {
                "robot_cartesian_pos": np.concatenate([tip, quat, [0.0]]),
                "robot_joint_pos": live_cfg.copy(),
            }

        def _solve(self, **kwargs):
            state["target"] = np.asarray(kwargs["target_position"]).copy()
            candidate = live_cfg.copy()
            candidate[0] += 0.05
            state["candidate"] = candidate
            return candidate

        def _move(self, _joints, **_kwargs):
            nonlocal move_calls
            move_calls += 1
            # Only half the requested XY step arrives, while Z drifts 7 mm.
            # v276 must seek the measured remainder, never replay the original
            # full command merely because Z did not track.
            target = state["target"]
            tip[:2] += 0.5 * (target[:2] - tip[:2])
            tip[2] += 0.007
            live_cfg[:] += 0.5 * (state["candidate"] - live_cfg)

    def fake_fk(_api, cfg=None):
        q = live_cfg if cfg is None else np.asarray(cfg)
        if "candidate" in state and np.allclose(q, state["candidate"]):
            return state["target"].copy(), quat.copy()
        return tip.copy(), quat.copy()

    monkeypatch.setattr(txy, "fk_panda_hand", fake_fk)
    result = txy.move_tip_xy(FakeApi(), 0.03, 0.0, max_iters=5, log=False)
    assert result.rejection_stage is None
    assert result.ok and result.acceptance_ok
    assert result.status == "completed"
    assert result.d_tip_z > 0.005
    assert result.executed_steps > 1
    assert move_calls == result.executed_steps


def test_local_solver_backoff_recovers_without_executing_branch_jump(
    monkeypatch,
) -> None:
    """An infeasible 2 cm solve shrinks to 1 cm and still completes the goal."""
    live_cfg = np.zeros(8, dtype=np.float64)
    tip = np.array([0.5, 0.1, 0.12], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state: dict[str, np.ndarray] = {}
    attempted_steps: list[float] = []
    executed_steps: list[float] = []

    class FakeApi:
        cfg = live_cfg.copy()
        _robot = SimpleNamespace(
            joints=SimpleNamespace(num_actuated_joints=8),
            links=SimpleNamespace(names=["panda_hand"]),
        )
        _target_link_name = "panda_hand"

        def __init__(self) -> None:
            self._env = SimpleNamespace(move_to_joints_blocking=self._move)
            self._pks = SimpleNamespace(solve_ik_local=self._solve_local)

        def get_observation(self):
            return {
                "robot_cartesian_pos": np.concatenate([tip, quat, [0.0]]),
                "robot_joint_pos": live_cfg.copy(),
            }

        def _solve_local(self, **kwargs):
            assert kwargs["max_arm_delta"] == 0.20
            target = np.asarray(kwargs["target_position"]).copy()
            step = float(np.linalg.norm(target[:2] - tip[:2]))
            attempted_steps.append(step)
            candidate = live_cfg.copy()
            candidate[0] += 1.0 if step > 0.011 else 0.05
            state["target"] = target
            state["candidate"] = candidate
            return candidate

        def _move(self, _joints, **_kwargs):
            executed_steps.append(float(np.linalg.norm(state["target"][:2] - tip[:2])))
            tip[:2] = state["target"][:2]
            live_cfg[:] = state["candidate"]

    def fake_fk(_api, cfg=None):
        q = live_cfg if cfg is None else np.asarray(cfg)
        if "candidate" in state and np.allclose(q, state["candidate"]):
            return state["target"].copy(), quat.copy()
        return tip.copy(), quat.copy()

    monkeypatch.setattr(txy, "fk_panda_hand", fake_fk)
    result = txy.move_tip_xy(FakeApi(), 0.03, 0.0, max_step_m=0.02, log=False)

    assert result.ok and result.status == "completed"
    assert result.preflight_rejections >= 2
    assert attempted_steps[0] > 0.011
    assert all(step <= 0.011 for step in executed_steps)
    assert result.executed_steps == len(executed_steps)


def test_student_exposure_flag_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CAPX_LIBERO_TIP_XY_ONLY", raising=False)
    assert txy.tip_xy_only_student_exposed() is False
    monkeypatch.setenv("CAPX_LIBERO_TIP_XY_ONLY", "1")
    assert txy.tip_xy_only_student_exposed() is True
