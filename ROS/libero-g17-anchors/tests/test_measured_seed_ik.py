from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from capx.integrations.franka import measured_seed_ik as measured


class _Env:
    def __init__(self, api: SimpleNamespace, settle: float = 1.0) -> None:
        self.api = api
        self.settle = settle
        self.commands: list[np.ndarray] = []

    def move_to_joints_blocking(self, joints: np.ndarray, *, max_steps: int) -> None:
        assert max_steps == 480
        target = np.asarray(joints, dtype=np.float64)
        self.api.live[:7] += self.settle * (target - self.api.live[:7])
        self.commands.append(target.copy())


class _Solver:
    def __init__(self, deltas: list[np.ndarray]) -> None:
        self.deltas = [np.asarray(delta, dtype=np.float64) for delta in deltas]
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def solve_ik_seeded(self, **kwargs) -> np.ndarray:
        initial = np.asarray(kwargs["initial_cfg"], dtype=np.float64)
        self.calls.append((initial.copy(), initial.copy()))
        return initial + self.deltas[len(self.calls) - 1]


def _api(*, settle: float = 1.0, deltas: list[np.ndarray]) -> SimpleNamespace:
    live = np.array([0.3, -0.4, 0.2, -1.8, 0.1, 1.7, 0.6, 0.0])
    limits = np.full(8, 10.0)
    api = SimpleNamespace(
        live=live.copy(),
        cfg=None,
        _target_link_name="panda_hand",
        _robot=SimpleNamespace(
            joints=SimpleNamespace(
                num_actuated_joints=8,
                lower_limits=-limits,
                upper_limits=limits,
            )
        ),
        _pks=_Solver(deltas),
    )
    api._env = _Env(api, settle=settle)
    api.get_observation = lambda: {"robot_joint_pos": api.live[:7].copy()}
    return api


def _fk(_api: SimpleNamespace, cfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(cfg)
    return q[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0])


def test_first_solve_is_initialized_and_referenced_from_measured_q(monkeypatch) -> None:
    delta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    api = _api(deltas=[delta])
    start = api.live.copy()
    monkeypatch.setattr(measured, "fk_panda_hand", _fk)

    result = measured.move_hand_pose_measured_seeded(
        api, start[:3] + delta[:3], np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )

    prev, initial = api._pks.calls[0]
    assert np.allclose(prev, start)
    assert np.allclose(initial, start)
    assert result.ok and result.execution_commands == 1
    assert len(api._env.commands) == 1


def test_next_stage_reseeds_from_controller_measurement_not_prior_target(
    monkeypatch,
) -> None:
    delta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    api = _api(settle=0.8, deltas=[delta, delta])
    start = api.live.copy()
    monkeypatch.setattr(measured, "fk_panda_hand", _fk)

    measured.move_hand_pose_measured_seeded(
        api, start[:3] + delta[:3], np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )
    after_first = api.live.copy()
    measured.move_hand_pose_measured_seeded(
        api, after_first[:3] + delta[:3], np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )

    second_prev, second_initial = api._pks.calls[1]
    assert np.allclose(after_first[0], start[0] + 0.08)
    assert np.allclose(second_prev, after_first)
    assert np.allclose(second_initial, after_first)
    assert np.allclose(api.cfg[:7], api.live[:7])
    assert len(api._env.commands) == 2


def test_large_but_valid_nearest_branch_is_not_rejected(monkeypatch) -> None:
    delta = np.array([0.65, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    api = _api(deltas=[delta])
    start = api.live.copy()
    monkeypatch.setattr(measured, "fk_panda_hand", _fk)

    result = measured.move_hand_pose_measured_seeded(
        api, start[:3] + delta[:3], np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )

    assert result.ok
    assert result.candidate_arm_delta_rad == pytest.approx(0.65)
    assert len(api._env.commands) == 1


def test_invalid_solver_candidate_is_never_executed(monkeypatch) -> None:
    delta = np.zeros(8)
    delta[2] = np.nan
    api = _api(deltas=[delta])
    monkeypatch.setattr(measured, "fk_panda_hand", _fk)

    with pytest.raises(RuntimeError, match="invalid cfg"):
        measured.move_hand_pose_measured_seeded(
            api, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), log=False
        )
    assert not api._env.commands


def test_preview_reports_unreachable_without_executing(monkeypatch) -> None:
    delta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    api = _api(deltas=[delta, delta, delta, delta])
    monkeypatch.setattr(measured, "fk_panda_hand", _fk)

    result = measured.move_hand_pose_measured_seeded(
        api,
        api.live[:3] + np.array([0.25, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        log=False,
        execute=False,
    )

    assert not result.ok
    assert result.status == "preview_inaccurate"
    assert result.execution_commands == 0
    assert not api._env.commands


def test_contract_has_no_motion_or_residual_gate() -> None:
    contract = measured.measured_seed_ik_contract()
    assert contract["native_blocking_commands_per_stage"] == 1
    assert contract["cartesian_waypoints"] == 0
    assert contract["joint_delta_gate"] == 0
    assert contract["terminal_residual_gate"] == 0
    assert contract["neutral_route"] == 0
    assert contract["rollback"] == 0


def test_tool_pi_symmetry_selects_nearest_physical_orientation(monkeypatch) -> None:
    requested_delta = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    symmetric_delta = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    api = _api(deltas=[requested_delta, symmetric_delta])
    start = api.live.copy()
    target_position = np.array([0.6, 0.1, 0.2])

    def symmetry_fk(_api, cfg):
        moved = float(np.asarray(cfg)[0] - start[0])
        wxyz = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if moved > 1.0
            else np.array([0.0, 0.0, 0.0, 1.0])
        )
        return target_position.copy(), wxyz

    monkeypatch.setenv("CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY", "1")
    monkeypatch.setattr(measured, "fk_panda_hand", symmetry_fk)
    result = measured.move_hand_pose_measured_seeded(
        api,
        target_position,
        np.array([1.0, 0.0, 0.0, 0.0]),
        log=False,
    )

    assert result.ok
    assert result.selected_orientation == "tool_z_pi"
    assert result.candidate_arm_delta_rad == pytest.approx(0.2)
    assert result.candidate_count == 2
    assert len(api._env.commands) == 1


def test_pt_audit_uses_exact_native_command_branch(monkeypatch) -> None:
    from capx.integrations.franka import perceptual_swept_path as swept

    delta = np.array([0.17, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    api = _api(deltas=[delta])
    start = api.live.copy()
    api._capx_v296a_pt_scene = object()
    api._capx_v296a_pt_target_points = np.ones((3, 3))
    live_scene = object()
    observed = {}

    def audit_candidate_path(_api, scene, measured_cfg, candidate_cfg):
        observed["scene"] = scene
        observed["start"] = np.asarray(measured_cfg).copy()
        observed["goal"] = np.asarray(candidate_cfg).copy()
        return {"predicted_native_path_clear": True}

    monkeypatch.setenv("CAPX_LIBERO_G17_V296A_PT_SWEPT_AUDIT", "1")
    monkeypatch.setattr(measured, "fk_panda_hand", _fk)
    monkeypatch.setattr(
        swept,
        "prepare_scene_from_observation",
        lambda _api, target, measured_cfg: live_scene,
    )
    monkeypatch.setattr(swept, "audit_candidate_path", audit_candidate_path)

    result = measured.move_hand_pose_measured_seeded(
        api, start[:3] + delta[:3], np.array([1.0, 0.0, 0.0, 0.0]), log=False
    )

    assert np.allclose(observed["start"], start)
    assert observed["scene"] is live_scene
    assert np.allclose(observed["goal"][:7], api._env.commands[0])
    assert result.pt_swept_path["audit_subject"] == "exact_native_command_q_goal"
    assert result.pt_swept_path["scene_snapshot"] == "exact_command_live_rgbd"
    assert np.allclose(result.pt_swept_path["command_goal_arm_q"], api._env.commands[0])
