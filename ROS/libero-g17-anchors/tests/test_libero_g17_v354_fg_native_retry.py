from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest

from capx.integrations.franka import fg_native_retry as retry


@pytest.fixture(autouse=True)
def _enable_v354(monkeypatch):
    monkeypatch.setenv(retry.ENV, "1")
    monkeypatch.delenv(retry.CLEAN_C_ENV, raising=False)
    monkeypatch.delenv(retry.DISABLE_SOFT_GAIN_ENV, raising=False)


def _fg(**changes):
    values = {
        "firm_acquired": False,
        "attachment": "empty",
        "target_contact": False,
        "tip_in_target_volume": False,
        "coordinate_in_target_band": False,
        "relative_x": "POSITIVE_OF_REFERENCE_BAND",
        "relative_y": "WITHIN_REFERENCE_BAND",
        "relative_z": "WITHIN_REFERENCE_BAND",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _tx(**changes):
    values = {
        "attempt": 2,
        "object_name": "akita black bowl",
        "source": "native_graspnet_postclose_observed",
        "native_motion_ok": True,
        "ingress_attempted": True,
        "executed": True,
    }
    values.update(changes)
    return values


@pytest.mark.parametrize(
    ("fg", "tx", "expected"),
    [
        (
            _fg(firm_acquired=True, attachment="holding_payload"),
            _tx(),
            retry.EndpointKind.TARGET_HELD,
        ),
        (
            _fg(attachment="holding_other_movable"),
            _tx(),
            retry.EndpointKind.WRONG_OBJECT_HELD,
        ),
        (
            _fg(target_contact=True),
            _tx(),
            retry.EndpointKind.EMPTY_TARGET_NEAR,
        ),
        (
            _fg(tip_in_target_volume=True),
            _tx(),
            retry.EndpointKind.EMPTY_TARGET_NEAR,
        ),
        (
            _fg(),
            _tx(native_motion_ok=False),
            retry.EndpointKind.EMPTY_BLOCKED,
        ),
        (
            _fg(),
            _tx(),
            retry.EndpointKind.EMPTY_CLEAR_OFFSET,
        ),
    ],
)
def test_endpoint_partition_is_total_and_positive(fg, tx, expected) -> None:
    event = retry.classify_endpoint(
        fg, tx, inverse_ingress_available=True
    )
    assert event.kind == expected


@pytest.mark.parametrize("kind", list(retry.EndpointKind))
@pytest.mark.parametrize("inverse", [False, True])
def test_every_endpoint_has_one_nonterminal_route(kind, inverse) -> None:
    event = retry.EndpointEvent(
        attempt=1,
        object_name="bowl",
        kind=kind,
        attachment=("holding_payload" if kind == retry.EndpointKind.TARGET_HELD else "empty"),
        target_contact=False,
        tip_in_target_volume=False,
        relative_x="WITHIN_REFERENCE_BAND",
        relative_y="WITHIN_REFERENCE_BAND",
        relative_z="WITHIN_REFERENCE_BAND",
        motion_outcome=retry.MotionOutcome.REACHED,
        proposal_family="NATIVE_GRASPNET",
        inverse_ingress_available=inverse,
    )
    route = retry.route_endpoint(event)
    assert route.execute_inverse is inverse
    if kind == retry.EndpointKind.TARGET_HELD and inverse:
        assert route.admit_transport
        assert not route.regenerate
        assert not route.open_gripper
        assert not route.execute_gr
    else:
        assert not route.admit_transport
        assert route.regenerate
        assert not route.execute_gr


def test_clean_c_nearmiss_has_one_gr_owner_then_native_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv(retry.CLEAN_C_ENV, "1")
    event = retry.classify_endpoint(
        _fg(target_contact=True), _tx(), inverse_ingress_available=True
    )
    route = retry.route_endpoint(event)
    assert route.open_gripper
    assert route.execute_inverse
    assert route.execute_gr
    assert not route.regenerate
    assert route.next_state == "inverse_escape_then_adaptive_GR"

    fallback = retry.route_endpoint(event, gr_already_attempted=True)
    assert not fallback.execute_gr
    assert fallback.regenerate
    assert not fallback.admit_transport


def test_clean_c_nearmiss_without_inverse_degrades_to_native_retry(
    monkeypatch,
) -> None:
    monkeypatch.setenv(retry.CLEAN_C_ENV, "1")
    event = retry.classify_endpoint(
        _fg(coordinate_in_target_band=True),
        _tx(),
        inverse_ingress_available=False,
    )
    route = retry.route_endpoint(event)
    assert not route.execute_inverse
    assert not route.execute_gr
    assert route.regenerate


def test_v359_has_no_hard_connected_retry_motor_directive() -> None:
    from capx.integrations.franka.libero import FrankaLiberoApi

    retry_source = inspect.getsource(retry)
    goto_source = inspect.getsource(FrankaLiberoApi.goto_pose)
    assert "consume_connected_retry_directive" not in retry_source
    assert "consume_connected_retry_directive" not in goto_source


def test_wrong_payload_is_released_before_retry() -> None:
    event = retry.classify_endpoint(
        _fg(attachment="holding_other_movable"),
        _tx(),
        inverse_ingress_available=True,
    )
    route = retry.route_endpoint(event)
    assert route.open_gripper
    assert route.execute_inverse
    assert route.regenerate


def test_soft_projection_is_discrete_and_consumed_once() -> None:
    retry.reset_trial(7)
    event = retry.classify_endpoint(
        _fg(target_contact=True),
        _tx(),
        inverse_ingress_available=True,
    )
    event = retry.finalize_recovery(event, inverse_completed=True)
    retry.publish(event, trial=7)
    report, meta = retry.consume_student_report(7)
    assert "locked target" in report
    assert "recorded ingress was reversed" in report
    assert meta["hard_endpoint_kind"] == "empty_target_near"
    assert meta["continuous_privileged_world_input"] is False
    assert retry.consume_student_report(7) is None
    assert not re.search(r"(?<![A-Za-z_])[-+]?\d", report)
    for forbidden in ("reward", "success", "failed", "next action"):
        assert forbidden not in report.lower()


def test_soft_projection_reports_same_family_information_gain() -> None:
    retry.reset_trial(3)
    event = retry.classify_endpoint(
        _fg(), _tx(), inverse_ingress_available=False
    )
    retry.publish(event, trial=3)
    retry.consume_student_report(3)
    retry.publish(event, trial=3)
    report, meta = retry.consume_student_report(3)
    assert meta["same_family_retry_state_repeated"] is True
    assert "already produced an empty close" in report


def test_soft_gain_ablation_consumes_mailbox_without_student_projection(
    monkeypatch,
) -> None:
    monkeypatch.setenv(retry.ENV, "1")
    monkeypatch.setenv(retry.DISABLE_SOFT_GAIN_ENV, "1")
    retry.reset_trial(11)
    event = retry.classify_endpoint(
        _fg(target_contact=True), _tx(), inverse_ingress_available=True
    )
    retry.publish(event, trial=11)
    assert retry.consume_student_report(11) is None
    assert retry._HISTORY[11] == [event]
    monkeypatch.delenv(retry.DISABLE_SOFT_GAIN_ENV)
    assert retry.consume_student_report(11) is None


def test_no_candidate_is_a_typed_soft_retry_not_terminal() -> None:
    retry.reset_trial(8)
    event = retry.proposal_unavailable_event(
        attempt=3, object_name="akita black bowl"
    )
    retry.publish(event, trial=8)
    report, meta = retry.consume_student_report(8)
    assert "produced no grasp proposal" in report
    assert "no physical close" in report
    assert meta["retry_event_kind"] == "proposal_unavailable"
    assert meta["physical_command_executed"] is False


def test_unreached_ingress_preserves_proposal_family_for_regeneration() -> None:
    event = retry.ingress_unreached_event(
        _tx(), physical_command_executed=True
    )
    assert event.kind == retry.PrecloseKind.INGRESS_UNREACHED
    assert event.proposal_family == "NATIVE_GRASPNET"
    assert event.physical_command_executed is True


def test_libero_close_supports_v354_ablation_and_v355_clean_c_gr() -> None:
    from capx.integrations.franka.libero import FrankaLiberoApi

    source = inspect.getsource(FrankaLiberoApi.close_gripper)
    assert "fg_native_retry_endpoint" in source
    assert "fg_native_retry_recovery_complete" in source
    assert "fg_clean_c_gr_transition" in source
    assert "native_retry_route.execute_gr" in source
    assert "closed_endpoint_retry_boundary" in source
    assert "and not _native_retry.enabled()" in source
    assert source.count("and not _native_retry.enabled()") >= 4


def test_payload_loss_releases_stale_firm_transaction() -> None:
    from capx.integrations.franka.libero import FrankaLiberoApi

    source = inspect.getsource(FrankaLiberoApi.goto_pose)
    assert "payload_lost_after_firm_transport" in source
    assert "fg_clean_c_transaction_released" in source


def test_preclose_audit_does_not_collide_with_event_parameter() -> None:
    from capx.integrations.franka.libero import FrankaLiberoApi

    source = inspect.getsource(FrankaLiberoApi.sample_grasp_pose)
    assert '"fg_native_retry_preclose_event"' in source
    assert "retry_event=_native_retry.event_dict(proposal_event)" in source
    assert not re.search(
        r"(?m)^\s+event=_native_retry\.event_dict\(proposal_event\)",
        source,
    )


def test_takeover_appends_soft_projection_to_fg_carrier() -> None:
    from capx.verification import libero_g17_takeover

    source = inspect.getsource(libero_g17_takeover.run_takeover)
    assert "consume_student_report" in source
    assert 'replacement = f"{replacement.rstrip()}\\n{retry_report}"' in source
