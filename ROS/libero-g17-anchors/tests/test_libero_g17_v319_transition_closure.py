from types import SimpleNamespace
from pathlib import Path

from capx.integrations.franka import transition_closure


def test_transition_closure_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(transition_closure.ENV, raising=False)
    monkeypatch.delenv(transition_closure.SEMANTIC_ENV, raising=False)
    assert not transition_closure.enabled()
    monkeypatch.setenv(transition_closure.ENV, "1")
    assert transition_closure.enabled()
    monkeypatch.delenv(transition_closure.ENV)
    monkeypatch.setenv(transition_closure.SEMANTIC_ENV, "1")
    assert transition_closure.enabled()


def test_post_capture_transition_preserves_native_skill_ownership() -> None:
    assert transition_closure.post_capture_transition(
        "holding_payload", True
    ) == "holding_native_goto_complete"
    assert transition_closure.post_capture_transition(
        "holding_payload", False
    ) == "holding_native_plan_continue"


def test_payload_loss_returns_to_start_even_when_motion_reached() -> None:
    assert transition_closure.post_capture_transition(
        "empty", True
    ) == "payload_lost_return_start"
    assert transition_closure.post_capture_transition(
        "holding_other_movable", False
    ) == "payload_lost_return_start"


def test_transport_epoch_is_bound_only_to_firm_acquisition() -> None:
    api = SimpleNamespace()
    assert not transition_closure.arm_transport(api, "empty", escaped=True)
    assert not transition_closure.transport_armed(api)
    assert not transition_closure.arm_transport(
        api, "holding_payload", escaped=False
    )
    assert transition_closure.arm_transport(
        api, "holding_payload", escaped=True
    )
    assert transition_closure.transport_armed(api)
    assert (
        transition_closure.symbolic_obligation(api)
        == transition_closure.PLACEMENT_OBLIGATION
    )
    transition_closure.disarm_transport(api)
    assert not transition_closure.transport_armed(api)
    assert transition_closure.symbolic_obligation(api) is None


def test_frozen_trial01_miss_sequence_preserves_native_plan_until_loss() -> None:
    # v319 job 528852 trial 1: three numerical misses retained the payload;
    # only the fourth observation was a real transition failure.
    attachments = ["holding_payload"] * 3 + ["empty"]
    transitions = [
        transition_closure.post_capture_transition(value, False)
        for value in attachments
    ]
    assert transitions[:3] == ["holding_native_plan_continue"] * 3
    assert transitions[3] == "payload_lost_return_start"


def test_transition_gates_are_attached_to_the_correct_boundaries() -> None:
    source = (
        Path(__file__).parents[1]
        / "capx/integrations/franka/libero.py"
    ).read_text()
    contact = source.index('"transition_contact_endpoint_fg"')
    transport = source.index('"transition_native_transport_boundary"')
    assert "if _v319.enabled():" in source[contact - 700 : contact]
    assert "transport_armed" not in source[contact - 700 : contact]
    assert "if _v319.enabled() and _v319.transport_armed(self):" in source[
        transport - 700 : transport
    ]
    transport_suffix = source[transport : transport + 1800]
    assert 'transition == "payload_lost_return_start"' in transport_suffix
    assert "native_transport_needs_regeneration" not in transport_suffix
