from capx.integrations.franka import postclose_acquisition as state_machine


def test_v331_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(state_machine.ENV, raising=False)
    assert not state_machine.enabled()
    monkeypatch.setenv(state_machine.ENV, "1")
    assert state_machine.enabled()


def test_v332_native_retry_ablation_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(
        state_machine.NATIVE_RETRY_ABLATION_ENV, raising=False
    )
    assert not state_machine.native_retry_ablation_enabled()
    assert state_machine.near_miss_next_state() == "inverse_escape_then_gr"
    monkeypatch.setenv(state_machine.NATIVE_RETRY_ABLATION_ENV, "1")
    assert state_machine.native_retry_ablation_enabled()
    assert (
        state_machine.near_miss_next_state()
        == "inverse_escape_then_native_regen"
    )


def test_close_certificate_never_denies_native_close() -> None:
    assert (
        state_machine.close_gate_decision(certificate_ready=False)
        == "allow_postclose_fg_classification"
    )
    assert (
        state_machine.close_gate_decision(certificate_ready=True)
        == "allow_certificate_ready"
    )


def test_postclose_partition_is_total_and_ordered() -> None:
    assert state_machine.classify_close(
        firm_acquired=True, near_miss=True
    ) == state_machine.CloseOutcome.FIRM_ACQUIRED
    assert state_machine.classify_close(
        firm_acquired=False, near_miss=True
    ) == state_machine.CloseOutcome.NEAR_MISS
    assert state_machine.classify_close(
        firm_acquired=False, near_miss=False
    ) == state_machine.CloseOutcome.DEFINITE_MISS


def test_v351_native_retry_regenerates_without_mandatory_escape(
    monkeypatch,
) -> None:
    monkeypatch.setenv(state_machine.NATIVE_RETRY_ABLATION_ENV, "1")
    for outcome in (
        state_machine.CloseOutcome.NEAR_MISS,
        state_machine.CloseOutcome.DEFINITE_MISS,
    ):
        route = state_machine.route_close(
            outcome, inverse_ingress_available=True
        )
        assert route.regenerate
        assert not route.execute_escape
        assert not route.execute_gr
        assert route.next_state == "fresh_FG_regeneration"


def test_v351_gr_uses_escape_only_as_its_available_precondition(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        state_machine.NATIVE_RETRY_ABLATION_ENV, raising=False
    )
    connected = state_machine.route_close(
        state_machine.CloseOutcome.NEAR_MISS,
        inverse_ingress_available=True,
    )
    assert connected.execute_escape
    assert connected.execute_gr
    assert not connected.regenerate

    disconnected = state_machine.route_close(
        state_machine.CloseOutcome.NEAR_MISS,
        inverse_ingress_available=False,
    )
    assert not disconnected.execute_escape
    assert not disconnected.execute_gr
    assert disconnected.regenerate


def test_v351_missing_payload_escape_degrades_to_observation() -> None:
    route = state_machine.route_close(
        state_machine.CloseOutcome.FIRM_ACQUIRED,
        inverse_ingress_available=False,
    )
    assert not route.execute_escape
    assert route.next_state == "holding_payload_observe_in_place"


def test_v352_definite_miss_uses_available_inverse_for_observation_recovery(
    monkeypatch,
) -> None:
    monkeypatch.setenv(state_machine.NATIVE_RETRY_ABLATION_ENV, "1")
    monkeypatch.setenv(state_machine.FG_OPTIONAL_RETRY_ESCAPE_ENV, "1")
    route = state_machine.route_close(
        state_machine.CloseOutcome.DEFINITE_MISS,
        inverse_ingress_available=True,
    )
    assert route.execute_escape
    assert route.regenerate
    assert not route.execute_gr
    assert (
        route.escape_use
        == state_machine.EscapeUse.RETRY_OBSERVATION_RECOVERY
    )
    assert route.next_state == "inverse_escape_then_fresh_FG_regeneration"


def test_v352_retry_escape_is_optional_not_a_retry_precondition(
    monkeypatch,
) -> None:
    monkeypatch.setenv(state_machine.NATIVE_RETRY_ABLATION_ENV, "1")
    monkeypatch.setenv(state_machine.FG_OPTIONAL_RETRY_ESCAPE_ENV, "1")

    no_inverse = state_machine.route_close(
        state_machine.CloseOutcome.DEFINITE_MISS,
        inverse_ingress_available=False,
    )
    assert not no_inverse.execute_escape
    assert no_inverse.regenerate
    assert no_inverse.next_state == "fresh_FG_regeneration"

    near_miss = state_machine.route_close(
        state_machine.CloseOutcome.NEAR_MISS,
        inverse_ingress_available=True,
    )
    assert not near_miss.execute_escape
    assert near_miss.regenerate
    assert near_miss.next_state == "fresh_FG_regeneration"
