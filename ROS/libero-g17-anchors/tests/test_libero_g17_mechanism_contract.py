"""Permanent experiment-class contract (no MuJoCo, no live LLM)."""

from __future__ import annotations

import pytest

from capx.verification import libero_g17_mechanism_contract as contract


def test_unset_class_is_exploratory_and_cannot_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(contract.EXPERIMENT_CLASS_ENV, raising=False)
    assert contract.experiment_class() == "exploratory_live"
    assert contract.mechanism_claim_allowed() is False
    assert contract.live_student_llm_forbidden() is False


def test_illegal_class_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(contract.EXPERIMENT_CLASS_ENV, "fair_k5")
    with pytest.raises(contract.MechanismContractError, match="not a legal"):
        contract.experiment_class()


@pytest.mark.parametrize("cls", ["mechanism_verify", "replay_confirm"])
def test_mechanism_classes_forbid_live_llm(
    cls: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(contract.EXPERIMENT_CLASS_ENV, cls)
    with pytest.raises(contract.MechanismContractError, match="forbids live student LLM"):
        contract.forbid_live_student_llm(trial=1, turn=1, reason="prefix miss")


def test_register_live_allows_live_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(contract.EXPERIMENT_CLASS_ENV, "register_live")
    contract.forbid_live_student_llm(trial=1, turn=1, reason="registration")


def test_result_cell_rejects_invalid_and_unknown() -> None:
    with pytest.raises(contract.MechanismContractError, match="validity"):
        contract.assert_result_cell(
            {"validity": "invalid_infra", "outcome": "success"},
            where="cell",
        )
    with pytest.raises(contract.MechanismContractError, match="outcome"):
        contract.assert_result_cell({"outcome": "unknown"}, where="cell")
    contract.assert_result_cell({"outcome": "fail"}, where="cell")


def test_public_text_rejects_incomplete_as_third_value() -> None:
    with pytest.raises(contract.MechanismContractError, match="incomplete"):
        contract.assert_definite_truth_text(
            "- note: bound placement evidence is incomplete relative to On.",
            where="public report",
        )
    contract.assert_definite_truth_text(
        "- robot gripper is not holding the acorn bowl.",
        where="public report",
    )


def test_schema_audit_must_ban_unknown() -> None:
    contract.assert_schema_unknown_free(
        {"unknown_is_not_a_gt_value": True}, where="g17"
    )
    with pytest.raises(contract.MechanismContractError, match="UNKNOWN"):
        contract.assert_schema_unknown_free(
            {"unknown_is_not_a_gt_value": False}, where="g17"
        )


def test_fair_live_summary_cannot_be_a_mechanism_claim() -> None:
    stamped = contract.classify_fair_live_summary(
        {"validity": "valid_prefix_arm", "gain_result": "positive"}
    )
    assert stamped["experiment_class"] == "exploratory_live"
    assert stamped["mechanism_claim_allowed"] is False
    with pytest.raises(contract.MechanismContractError, match="cannot make a mechanism"):
        contract.assert_mechanism_payload(
            {
                **stamped,
                "cells": {"t1": {"outcome": "success"}},
            },
            where="fair-k",
        )


def test_mechanism_payload_accepts_only_closed_cells() -> None:
    contract.assert_mechanism_payload(
        {
            "experiment_class": "mechanism_verify",
            "mechanism_claim_allowed": True,
            "cells": {
                "spatial_0_on": {"outcome": "success"},
                "object_1_in": {"outcome": "fail"},
            },
        },
        where="iff",
    )
