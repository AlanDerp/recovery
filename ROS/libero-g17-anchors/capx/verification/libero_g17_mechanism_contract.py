"""Permanent LIBERO G17 experiment-class contract.

This module is the fail-closed law. New FG schemas may change atoms. They may
not invent a fourth experiment class, a third outcome, or a third truth value.

Experiment classes
------------------
``mechanism_verify``
    Offline IFF or full-trajectory replay. Live student LLM is forbidden.
    Result cells are only ``success`` / ``fail``.

``register_live``
    Exactly one live treatment registration. Output is a replay package, not a
    scored comparison. Remote Qwen variance is accepted once, then frozen.

``replay_confirm``
    Complete request/response replay required. Any miss aborts before a result
    file is written.

``exploratory_live``
    Default if unset. Includes fair-K live-after-π0. May not write a mechanism
    claim. Residual remote-Qwen variance is the measurement, not the method.

Forbidden vocabulary
--------------------
- Outcome: no ``invalid``, ``invalid_infra``, ``unknown``.
- Ground truth: no ``UNKNOWN``, ``UNDECIDABLE``, ``AMBIGUOUS``, ``incomplete``.
  If a predicate cannot be decided, it is not in the schema.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Mapping

EXPERIMENT_CLASS_ENV = "CAPX_LIBERO_G17_EXPERIMENT_CLASS"

ExperimentClass = Literal[
    "mechanism_verify",
    "register_live",
    "replay_confirm",
    "exploratory_live",
]

LEGAL_CLASSES: frozenset[str] = frozenset(
    ("mechanism_verify", "register_live", "replay_confirm", "exploratory_live")
)
LIVE_STUDENT_LLM_FORBIDDEN: frozenset[str] = frozenset(
    ("mechanism_verify", "replay_confirm")
)
MECHANISM_CLAIM_ALLOWED: frozenset[str] = frozenset(
    ("mechanism_verify", "replay_confirm")
)

LEGAL_OUTCOMES: frozenset[str] = frozenset(("success", "fail"))

_UNKNOWN_SHAPED = re.compile(
    r"\b(unknown|undecidable|ambiguous|incomplete)\b",
    re.IGNORECASE,
)


class MechanismContractError(RuntimeError):
    """Raised when a run tries to leave the permanent contract."""


def experiment_class(raw: str | None = None) -> ExperimentClass:
    value = (raw if raw is not None else os.environ.get(EXPERIMENT_CLASS_ENV, "")).strip()
    if not value:
        return "exploratory_live"
    if value not in LEGAL_CLASSES:
        raise MechanismContractError(
            f"{EXPERIMENT_CLASS_ENV}={value!r} is not a legal experiment class; "
            f"legal={sorted(LEGAL_CLASSES)}"
        )
    return value  # type: ignore[return-value]


def live_student_llm_forbidden(cls: ExperimentClass | None = None) -> bool:
    return (experiment_class() if cls is None else cls) in LIVE_STUDENT_LLM_FORBIDDEN


def mechanism_claim_allowed(cls: ExperimentClass | None = None) -> bool:
    return (experiment_class() if cls is None else cls) in MECHANISM_CLAIM_ALLOWED


def forbid_live_student_llm(*, trial: int | None, turn: int | None, reason: str) -> None:
    """Abort instead of calling remote Qwen when the class forbids it."""
    cls = experiment_class()
    if cls not in LIVE_STUDENT_LLM_FORBIDDEN:
        return
    raise MechanismContractError(
        f"{cls} forbids live student LLM "
        f"(trial={trial} turn={turn}): {reason}. "
        "Register one live trajectory, then replay_confirm; "
        "or run offline IFF under mechanism_verify."
    )


def assert_definite_truth_text(text: str, *, where: str) -> None:
    """Public / GT text may only name declared states. No third-value hedges."""
    if not text:
        return
    hit = _UNKNOWN_SHAPED.search(text)
    if hit is None:
        return
    raise MechanismContractError(
        f"{where} used unknown-shaped token {hit.group(0)!r}. "
        "Truth values are declared states only; drop the predicate if it cannot close."
    )


def assert_schema_unknown_free(audit: Mapping[str, Any], *, where: str) -> None:
    if audit.get("unknown_is_not_a_gt_value") is not True:
        raise MechanismContractError(
            f"{where} schema admits UNKNOWN as a GT value"
        )


def assert_result_cell(cell: Mapping[str, Any], *, where: str) -> None:
    """A scored cell is only success or fail. Infra failure is not a cell."""
    if "validity" in cell:
        raise MechanismContractError(
            f"{where} used validity={cell.get('validity')!r}; "
            "experiment cells are success/fail only. "
            "Infra failure must abort before writing a result."
        )
    outcome = cell.get("outcome")
    if outcome not in LEGAL_OUTCOMES:
        raise MechanismContractError(
            f"{where} outcome={outcome!r} is not success|fail"
        )


def assert_mechanism_payload(payload: Mapping[str, Any], *, where: str) -> None:
    """A mechanism table cannot carry exploratory-live or invalid rows."""
    cls = payload.get("experiment_class")
    if cls not in MECHANISM_CLAIM_ALLOWED:
        raise MechanismContractError(
            f"{where} experiment_class={cls!r} cannot make a mechanism claim"
        )
    if payload.get("mechanism_claim_allowed") is not True:
        raise MechanismContractError(f"{where} mechanism_claim_allowed is not true")
    for key in ("validity", "n_valid", "invalid_infra"):
        if key in payload:
            raise MechanismContractError(
                f"{where} still carries {key}={payload.get(key)!r}"
            )
    cells = payload.get("cells")
    if not isinstance(cells, Mapping):
        raise MechanismContractError(f"{where} missing cells mapping")
    for name, cell in cells.items():
        if not isinstance(cell, Mapping):
            raise MechanismContractError(f"{where} cell {name!r} is not a mapping")
        assert_result_cell(cell, where=f"{where}.cells.{name}")


def classify_fair_live_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Stamp an existing fair-K summary so it cannot be read as mechanism."""
    return {
        "experiment_class": "exploratory_live",
        "mechanism_claim_allowed": False,
        "reason": (
            "Fair K-seed freezes π0 only; post-fork remote Qwen remains live. "
            "Frozen-A0 already proved temperature=0 is not determinism. "
            "This file is variance, not IFF."
        ),
        "source_validity": summary.get("validity"),
        "source_gain_result": summary.get("gain_result"),
    }
