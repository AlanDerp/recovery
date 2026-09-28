from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from capx.integrations.franka import acquisition_audit as acq


def test_cloud_summary_is_robust_and_serializable() -> None:
    points = np.array(
        [[0.5, 0.1, 0.2], [0.5, 0.1, 0.3], [0.5, 0.1, np.nan]]
    )
    summary = acq.cloud_summary(points)
    assert summary["point_count"] == 2
    assert summary["median_xyz"] == [0.5, 0.1, 0.25]
    json.dumps(summary)


def test_transaction_trace_has_attempt_and_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V293_ACQUISITION", "1")
    monkeypatch.setenv("CAPX_REPRO_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("CAPX_ACTIVE_TRIAL", "4")
    api = SimpleNamespace()
    assert acq.begin_attempt(api, "akita black bowl") == 1
    acq.audit(api, "target_grounded", mode="point_correct")
    assert acq.begin_attempt(api, "akita black bowl") == 2
    path = tmp_path / "libero_g17_v293_acquisition" / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event_idx"] for row in rows] == [0, 1, 2]
    assert [row["event"] for row in rows] == [
        "acquisition_needed",
        "target_grounded",
        "acquisition_needed",
    ]
    assert rows[0]["trial"] == 4


def test_v312_new_binding_invalidates_stale_motion_transaction(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V293_ACQUISITION", "1")
    monkeypatch.setenv("CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION", "1")
    monkeypatch.setenv("CAPX_REPRO_AUDIT_DIR", str(tmp_path))
    api = SimpleNamespace(
        _capx_v312_acquisition_transaction={"attempt": 7},
        _capx_v304_ingress={"attempt": 7},
    )
    acq.begin_attempt(api, "bowl")
    assert api._capx_v312_acquisition_transaction is None
    assert api._capx_v304_ingress is None


def test_topk_selection_skips_high_score_inaccurate_ik(monkeypatch) -> None:
    from capx.integrations.franka import measured_seed_ik as measured
    from capx.integrations.franka.libero import FrankaLiberoApi

    monkeypatch.delenv("CAPX_LIBERO_G17_GOTO_TIP_IDENTITY", raising=False)
    monkeypatch.delenv("CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY", raising=False)
    monkeypatch.setattr(measured, "_live_actuated_cfg", lambda _api: np.zeros(8))
    monkeypatch.setattr(measured, "_validate_candidate", lambda _api, _cfg: None)

    def metrics(_api, _cfg, _measured, target, _orientation):
        # The GraspNet argmax at x=.10 has no accurate solution; x=.20 does.
        pos_error = 0.20 if target[0] < 0.15 else 0.001
        return pos_error, 0.01, 0.2, 0.3

    monkeypatch.setattr(measured, "_candidate_metrics", metrics)
    api = SimpleNamespace(
        _TCP_OFFSET=np.zeros(3),
        _robot=object(),
        _target_link_name="panda_hand",
        _pks=SimpleNamespace(
            solve_ik_seeded=lambda **_kwargs: np.zeros(8),
        ),
    )
    transforms = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    transforms[0, 0, 3] = 0.10
    transforms[1, 0, 3] = 0.20
    selected, rows = FrankaLiberoApi._select_v293_grasp_candidate(
        api, transforms, np.array([0.9, 0.8])
    )
    assert selected == 1
    assert rows[0]["ik_feasible"] is False
    assert rows[1]["ik_feasible"] is True


def test_sample_grasp_initializes_private_audit_scope() -> None:
    import inspect

    from capx.integrations.franka.libero import FrankaLiberoApi

    source = inspect.getsource(FrankaLiberoApi.sample_grasp_pose)
    assert "acquisition_audit as _acq" in source
    assert "attempt = _acq.begin_attempt" in source


def test_goal_pose_observation_preserves_acquisition_binding() -> None:
    from capx.integrations.franka.libero import FrankaLiberoApi

    points = np.array(
        [[0.70, 0.20, 0.01], [0.72, 0.20, 0.01], [0.71, 0.22, 0.01]]
    )
    transaction = {"attempt": 3, "object_name": "bowl"}
    api = SimpleNamespace(
        _capx_v293_attempt=3,
        _capx_v293_active_object="bowl",
        _capx_v293_capture_state="acquisition_needed",
        _capx_v312_acquisition_transaction=transaction,
        _capx_v304_ingress={"attempt": 3},
        _resolve_typed_instance=lambda *args, **kwargs: {"points_3d": points},
        filter_noise=lambda value: (value, None),
        get_oriented_bounding_box_from_3d_points=lambda value: {
            "center": np.mean(value, axis=0),
            "R": np.eye(3),
        },
    )
    position, _ = FrankaLiberoApi.get_object_pose(api, "plate")
    np.testing.assert_allclose(position, np.mean(points, axis=0))
    assert api._capx_v293_attempt == 3
    assert api._capx_v293_active_object == "bowl"
    assert api._capx_v293_capture_state == "acquisition_needed"
    assert api._capx_v312_acquisition_transaction is transaction
    assert api._capx_v304_ingress == {"attempt": 3}
