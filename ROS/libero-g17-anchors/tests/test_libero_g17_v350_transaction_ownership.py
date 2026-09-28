from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from capx.integrations.franka import acquisition_ownership as ownership


def _api(*, transaction: dict | None = None, attempt: int | None = 4):
    return SimpleNamespace(
        _capx_v293_attempt=attempt,
        _capx_v293_capture_state="acquisition_needed",
        _capx_v312_acquisition_transaction=transaction,
    )


def _transaction(api):
    return ownership.bind_transaction(
        api,
        attempt=api._capx_v293_attempt,
        object_name="akita black bowl",
        tip_xyz=np.array([0.70, -0.20, 0.16]),
        tip_wxyz=np.array([0.72, -0.12, 0.50, 0.45]),
        source="test",
        topdown_transaction={
            "pregrasp_tip_xyz": [0.70, -0.20, 0.26],
            "pregrasp_arm_cfg_rad": np.zeros(7),
            "contact_arm_cfg_rad": np.ones(7) * 0.1,
        },
    )


def test_changed_policy_pose_is_canonicalized_not_executed(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    api = _api()
    transaction = _transaction(api)
    decision = ownership.route_goto(
        api,
        np.array([0.83, -0.15, 0.41]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        0.0,
    )
    assert decision.mode == "canonical_transaction"
    np.testing.assert_allclose(decision.position, transaction["tip_xyz"])
    np.testing.assert_allclose(
        decision.quaternion_wxyz, transaction["tip_wxyz"]
    )
    assert decision.position_divergence_m > 0.1
    assert decision.orientation_divergence_rad > 0.1


def test_quaternion_sign_is_not_a_divergence(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    api = _api()
    transaction = _transaction(api)
    decision = ownership.route_goto(
        api,
        transaction["tip_xyz"],
        -np.asarray(transaction["tip_wxyz"]),
        0.15,
    )
    assert decision.mode == "canonical_transaction"
    assert decision.orientation_divergence_rad == pytest.approx(0.0)


def test_second_goto_cannot_take_control_after_ingress(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    api = _api()
    transaction = _transaction(api)
    transaction["contact_reached"] = True
    transaction["executed"] = True
    decision = ownership.route_goto(
        api,
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        0.0,
    )
    assert decision.mode == "suppress_after_ingress"


def test_stale_or_missing_transaction_fails_before_motion(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    api = _api(transaction={"attempt": 3})
    with pytest.raises(ownership.AcquisitionOwnershipError):
        ownership.route_goto(
            api,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            0.0,
        )
    with pytest.raises(ownership.AcquisitionOwnershipError):
        ownership.require_close_transaction(api)


def test_ownership_starts_at_first_bound_proposal(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    api = _api(transaction=None, attempt=None)
    decision = ownership.route_goto(
        api,
        np.array([0.5, 0.0, 0.4]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        0.0,
    )
    assert decision.mode == "passthrough"


def test_v350g_rejects_raw_pose_motion_before_first_proposal(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    monkeypatch.setenv(
        "CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP", "1"
    )
    api = _api(transaction=None, attempt=None)
    with pytest.raises(
        ownership.AcquisitionOwnershipError,
        match="no typed grasp transaction",
    ):
        ownership.route_goto(
            api,
            np.array([0.5, 0.0, 0.4]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            0.0,
        )


def test_v350g_rejects_raw_pose_close_before_first_proposal(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    monkeypatch.setenv(
        "CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP", "1"
    )
    api = _api(transaction=None, attempt=None)
    with pytest.raises(
        ownership.AcquisitionOwnershipError,
        match="no typed grasp transaction",
    ):
        ownership.require_close_transaction(api)


def test_v350g_does_not_own_transport_or_placement(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    monkeypatch.setenv(
        "CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP", "1"
    )
    api = _api(transaction=None, attempt=None)
    api._capx_v293_capture_state = "holding_payload"
    decision = ownership.route_goto(
        api,
        np.array([0.5, 0.0, 0.4]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        0.0,
    )
    assert decision.mode == "passthrough"
    assert ownership.require_close_transaction(api) is None


def test_contact_is_not_a_grasp_precondition(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    api = _api()
    transaction = _transaction(api)
    assert transaction["physical_contact_required"] is False
    # The transaction records execution of the ingress endpoint, not a
    # contact predicate.  A subsequent close may be FirmAcquired even when no
    # pre-close contact was observed.
    transaction["contact_reached"] = True
    assert ownership.require_close_transaction(api) is transaction


def test_native_fallback_gets_the_same_owned_shape(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    api = _api()
    transaction = ownership.bind_transaction(
        api,
        attempt=4,
        object_name="bowl",
        tip_xyz=np.array([0.7, -0.2, 0.16]),
        tip_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        source="native_graspnet_postclose_observed",
    )
    decision = ownership.route_goto(
        api,
        np.array([9.0, 9.0, 9.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        0.0,
    )
    assert decision.mode == "canonical_transaction"
    assert decision.z_approach == pytest.approx(0.10)
    assert transaction["pregrasp_arm_cfg_rad"] is None


def test_v350k_pose_only_native_attempt_closes_on_outer_fg_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    monkeypatch.setenv(
        "CAPX_LIBERO_G17_V350K_TRANSPARENT_NATIVE_TRANSACTION", "1"
    )
    api = _api()
    transaction = ownership.bind_transaction(
        api,
        attempt=4,
        object_name="bowl",
        tip_xyz=np.array([0.7, -0.2, 0.16]),
        tip_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        source="native_graspnet_postclose_observed",
    )
    assert ownership.uses_transparent_native_executor(transaction)
    assert not ownership.ingress_attempt_completed(transaction)
    transaction["ingress_attempted"] = True
    transaction["executed"] = True
    transaction["native_motion_ok"] = False
    assert transaction["contact_reached"] is False
    assert ownership.ingress_attempt_completed(transaction)

    # Once the native primitive has physically attempted ingress, generated
    # code cannot issue a second acquisition motion before close.
    decision = ownership.route_goto(
        api,
        transaction["tip_xyz"],
        transaction["tip_wxyz"],
        0.0,
    )
    assert decision.mode == "suppress_after_ingress"


def test_v350k_certified_transaction_still_requires_contact_endpoint(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    monkeypatch.setenv(
        "CAPX_LIBERO_G17_V350K_TRANSPARENT_NATIVE_TRANSACTION", "1"
    )
    api = _api()
    transaction = _transaction(api)
    transaction["ingress_attempted"] = True
    transaction["executed"] = True
    assert not ownership.uses_transparent_native_executor(transaction)
    assert not ownership.ingress_attempt_completed(transaction)
    transaction["contact_reached"] = True
    assert ownership.ingress_attempt_completed(transaction)


def test_v350k_native_goto_bypasses_q1_q2_completion() -> None:
    from capx.integrations.franka.libero import FrankaLiberoApi

    source = inspect.getsource(FrankaLiberoApi.goto_pose)
    completion = source.index("complete_native_transaction")
    transparent_guard = source.rindex(
        "not _ownership.uses_transparent_native_executor", 0, completion
    )
    assert transparent_guard < completion
    assert "transparent_native_ingress_completed" in source


def test_v350l_failed_writer_releases_to_fg_without_forcing_escape(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    monkeypatch.setenv(
        "CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP", "1"
    )
    monkeypatch.setenv(
        "CAPX_LIBERO_G17_V350L_FG_REGENERATION_ON_EXECUTION_FAILURE",
        "1",
    )
    api = _api()
    api._capx_v293_active_object = "akita black bowl"
    api._capx_v304_ingress = {"attempt": 4}
    transaction = _transaction(api)

    outcome = ownership.release_for_fg_regeneration(
        api,
        transaction,
        stage="pregrasp",
        failure="endpoint_unreached",
    )

    assert outcome["next_state"] == "fresh_FG_observation_then_new_proposal"
    assert transaction["ownership_state"] == "released_to_fg_regeneration"
    assert api._capx_v312_acquisition_transaction is None
    assert api._capx_v304_ingress is None
    assert api._capx_v293_capture_state == "acquisition_needed"
    assert api._capx_v293_active_object == "akita black bowl"
    with pytest.raises(
        ownership.AcquisitionOwnershipError,
        match="fresh sample_grasp_pose",
    ):
        ownership.route_goto(
            api,
            transaction["tip_xyz"],
            transaction["tip_wxyz"],
            0.1,
        )


def test_v350m_no_grasp_candidate_is_typed_proposal_failure() -> None:
    from capx.integrations.franka.libero import (
        FrankaLiberoApi,
        GraspProposalUnavailable,
    )

    api = SimpleNamespace(
        grasp_net_plan_point_clouds_fn=lambda *_args, **_kwargs: (
            np.empty((0, 4, 4)),
            np.empty((0,)),
            np.empty((0, 3)),
        )
    )
    with pytest.raises(GraspProposalUnavailable, match="No grasp candidates"):
        FrankaLiberoApi.plan_grasp_from_point_clouds(
            api,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        )


def test_v350m_sample_routes_proposal_failure_to_fg_breakpoint() -> None:
    from capx.integrations.franka.libero import FrankaLiberoApi

    source = inspect.getsource(FrankaLiberoApi.sample_grasp_pose)
    assert "acquisition_proposal_unavailable_fg_regeneration" in source
    assert "physical_command_executed=False" in source
    assert "escape_executed=False" in source


def test_native_fallback_is_completed_on_one_connected_branch(
    monkeypatch,
) -> None:
    from capx.integrations.franka import measured_seed_ik
    from capx.integrations.franka import perceptual_grasp_family
    from capx.integrations.franka import perceptual_swept_path
    from capx.integrations.franka import tip_xy_only

    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    monkeypatch.setattr(
        measured_seed_ik, "tool_pi_symmetry_enabled", lambda: False
    )
    monkeypatch.setattr(measured_seed_ik, "_fallback_seeds", lambda _q: ())
    monkeypatch.setattr(measured_seed_ik, "_validate_candidate", lambda *_: None)
    monkeypatch.setattr(
        measured_seed_ik,
        "_candidate_metrics",
        lambda _api, q, initial, *_: (
            0.0,
            0.0,
            float(np.max(np.abs(q[:7] - initial[:7]))),
            float(np.linalg.norm(q[:7] - initial[:7])),
        ),
    )
    monkeypatch.setattr(
        perceptual_grasp_family,
        "_v286_grasp_pose_to_hand_target",
        lambda xyz, quat: (np.asarray(xyz), np.asarray(quat)),
    )
    monkeypatch.setattr(
        tip_xy_only, "_live_actuated_cfg", lambda _api: np.zeros(8)
    )
    monkeypatch.setattr(
        perceptual_swept_path,
        "audit_candidate_path",
        lambda *_: {"predicted_arm_obstacle_path_clear": True},
    )
    seeds = []

    def solve(**kwargs):
        seed = np.asarray(kwargs["initial_cfg"], dtype=np.float64).copy()
        seeds.append(seed)
        result = seed.copy()
        result[:7] += 0.1
        return result

    api = _api()
    api._pks = SimpleNamespace(solve_ik_seeded=solve)
    api._robot = object()
    api._target_link_name = "panda_hand"
    api._capx_v296a_pt_scene = object()
    transaction = ownership.bind_transaction(
        api,
        attempt=4,
        object_name="bowl",
        tip_xyz=np.array([0.7, -0.2, 0.16]),
        tip_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        source="native",
    )
    completed = ownership.complete_native_transaction(api, transaction)
    np.testing.assert_allclose(completed["pregrasp_arm_cfg_rad"], 0.1)
    np.testing.assert_allclose(completed["contact_arm_cfg_rad"], 0.2)
    # The contact solve is seeded from the selected pregrasp solution, not
    # measured/reset/neutral state.
    np.testing.assert_allclose(seeds[1][:7], 0.1)
    assert completed["ownership_state"] == "proposed_complete"


def test_failed_transaction_is_terminal_not_replayed(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "1")
    api = _api()
    transaction = _transaction(api)
    transaction["ownership_state"] = "execution_failed"
    with pytest.raises(ownership.AcquisitionOwnershipError):
        ownership.route_goto(
            api,
            transaction["tip_xyz"],
            transaction["tip_wxyz"],
            0.1,
        )
