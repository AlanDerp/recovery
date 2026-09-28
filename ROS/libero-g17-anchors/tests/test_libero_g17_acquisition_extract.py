from types import SimpleNamespace

import numpy as np

from capx.integrations.franka import acquisition_extract as extract


class FakeEnv:
    def __init__(self) -> None:
        self.q = np.zeros(8)
        self.commands = []
        self.max_steps = []
        self._last_joint_move_report = {"exit_reason": "converged"}

    def move_to_joints_blocking(self, joints, *, max_steps=None) -> None:
        arm = np.asarray(joints, dtype=np.float64).reshape(7)
        self.commands.append(arm.copy())
        self.max_steps.append(None if max_steps is None else int(max_steps))
        self.q[:7] = arm


class FakeApi:
    def __init__(self) -> None:
        self._env = FakeEnv()
        self._capx_v293_attempt = 4
        self._capx_v293_active_object = "bowl"


def test_retreat_reuses_recorded_ingress_without_distance_constant(monkeypatch) -> None:
    monkeypatch.setenv(extract.ENV, "1")
    monkeypatch.setenv(extract.COMPLETE_INVERSE_ENV, "1")
    monkeypatch.setattr(
        "capx.integrations.franka.tip_xy_only._live_actuated_cfg",
        lambda api: api._env.q.copy(),
    )
    monkeypatch.setattr(
        "capx.integrations.franka.acquisition_audit.audit", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "capx.integrations.franka.acquisition_audit.audit_attachment",
        lambda *a, **k: "holding_payload",
    )
    api = FakeApi()
    ingress_q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.02])
    extract.record_ingress(
        api,
        attempt=4,
        object_name="bowl",
        arm_cfg=ingress_q,
        hand_target_xyz=np.array([0.7, -0.1, 0.3]),
        hand_target_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    assert extract.has_current_ingress(api)
    api._env.q[:7] = -1.0
    report = extract.retreat_after_close(api)
    assert report is not None
    np.testing.assert_allclose(api._env.commands, [ingress_q[:7]])
    assert api._env.max_steps == [480]
    assert api._capx_v304_ingress is None
    assert api._capx_v304_extraction_complete is True
    assert not extract.has_current_ingress(api)


def test_retreat_rejects_stale_attempt(monkeypatch) -> None:
    monkeypatch.setenv(extract.ENV, "1")
    monkeypatch.setattr(
        "capx.integrations.franka.acquisition_audit.audit", lambda *a, **k: None
    )
    api = FakeApi()
    extract.record_ingress(
        api,
        attempt=3,
        object_name="old bowl",
        arm_cfg=np.zeros(7),
        hand_target_xyz=np.zeros(3),
        hand_target_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert not extract.has_current_ingress(api)
    assert extract.retreat_after_close(api) is None
    assert api._env.commands == []
