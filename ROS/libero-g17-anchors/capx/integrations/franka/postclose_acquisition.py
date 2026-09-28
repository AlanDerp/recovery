"""Total post-close acquisition policy for the v320 GR stack.

The transaction is evidence about an executed native attempt. It is never a
precondition for executing the first close. Closed FG after that close owns
the only three controller transitions in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os


ENV = "CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE"
NATIVE_RETRY_ABLATION_ENV = (
    "CAPX_LIBERO_G17_V332_NATIVE_RETRY_ABLATION"
)
CLOSED_ROUTER_ENV = "CAPX_LIBERO_G17_V351_CLOSED_TRANSACTION_ROUTER"
FG_OPTIONAL_RETRY_ESCAPE_ENV = (
    "CAPX_LIBERO_G17_V352_FG_OPTIONAL_RETRY_ESCAPE"
)


class CloseOutcome(str, Enum):
    FIRM_ACQUIRED = "firm_acquired"
    NEAR_MISS = "near_miss"
    DEFINITE_MISS = "definite_miss"


class EscapeUse(str, Enum):
    PAYLOAD_EXTRACTION = "payload_extraction"
    GR_PRECONDITION = "gr_precondition"
    OPTIONAL_RETRY_SKIPPED = "optional_retry_skipped"
    RETRY_OBSERVATION_RECOVERY = "retry_observation_recovery"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CloseRoute:
    """One exhaustive controller route after endpoint FG classification."""

    outcome: CloseOutcome
    execute_escape: bool
    escape_use: EscapeUse
    execute_gr: bool
    regenerate: bool
    next_state: str


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def native_retry_ablation_enabled() -> bool:
    """Return whether NearMiss bypasses GR after the same inverse escape."""
    return os.environ.get(NATIVE_RETRY_ABLATION_ENV, "0") == "1"


def closed_router_enabled() -> bool:
    return os.environ.get(CLOSED_ROUTER_ENV, "0") == "1"


def fg_optional_retry_escape_enabled() -> bool:
    return os.environ.get(FG_OPTIONAL_RETRY_ESCAPE_ENV, "0") == "1"


def route_close(
    outcome: CloseOutcome,
    *,
    inverse_ingress_available: bool,
) -> CloseRoute:
    """Return the unique post-close route without continuous world GT.

    Payload extraction and GR use the recorded inverse ingress when it exists.
    A native retry never requires escape: its monotone fallback is observation
    of the actual current world followed by FG regeneration.  Missing ingress
    therefore degrades to regeneration rather than an interface exception.
    """

    if outcome == CloseOutcome.FIRM_ACQUIRED:
        return CloseRoute(
            outcome=outcome,
            execute_escape=bool(inverse_ingress_available),
            escape_use=(
                EscapeUse.PAYLOAD_EXTRACTION
                if inverse_ingress_available
                else EscapeUse.UNAVAILABLE
            ),
            execute_gr=False,
            regenerate=False,
            next_state=(
                "escaped_holding"
                if inverse_ingress_available
                else "holding_payload_observe_in_place"
            ),
        )
    if outcome == CloseOutcome.NEAR_MISS and not native_retry_ablation_enabled():
        return CloseRoute(
            outcome=outcome,
            execute_escape=bool(inverse_ingress_available),
            escape_use=(
                EscapeUse.GR_PRECONDITION
                if inverse_ingress_available
                else EscapeUse.UNAVAILABLE
            ),
            execute_gr=bool(inverse_ingress_available),
            regenerate=not inverse_ingress_available,
            next_state=(
                "inverse_escape_then_gr"
                if inverse_ingress_available
                else "fresh_FG_regeneration"
            ),
        )
    # DefiniteMiss and the v332 NearMiss ablation both use native retry.  A
    # NearMiss deliberately preserves its useful endpoint for the retry/GR
    # policy.  A DefiniteMiss has neither payload nor a useful close endpoint;
    # v352 may therefore invoke the measured inverse ingress as an optional
    # observation-recovery skill before fresh FG.  Missing ingress still
    # degrades to in-place FG rather than becoming a retry precondition.
    if (
        outcome == CloseOutcome.DEFINITE_MISS
        and fg_optional_retry_escape_enabled()
        and inverse_ingress_available
    ):
        return CloseRoute(
            outcome=outcome,
            execute_escape=True,
            escape_use=EscapeUse.RETRY_OBSERVATION_RECOVERY,
            execute_gr=False,
            regenerate=True,
            next_state="inverse_escape_then_fresh_FG_regeneration",
        )
    return CloseRoute(
        outcome=outcome,
        execute_escape=False,
        escape_use=EscapeUse.OPTIONAL_RETRY_SKIPPED,
        execute_gr=False,
        regenerate=True,
        next_state="fresh_FG_regeneration",
    )


def near_miss_next_state() -> str:
    """Keep the v332 ablation decision explicit and independently testable."""
    if native_retry_ablation_enabled():
        return "inverse_escape_then_native_regen"
    return "inverse_escape_then_gr"


def classify_close(*, firm_acquired: bool, near_miss: bool) -> CloseOutcome:
    """Return the exhaustive controller state after one physical close."""
    if firm_acquired:
        return CloseOutcome.FIRM_ACQUIRED
    if near_miss:
        return CloseOutcome.NEAR_MISS
    return CloseOutcome.DEFINITE_MISS


def close_gate_decision(*, certificate_ready: bool) -> str:
    """Describe the certificate without granting or denying motor authority."""
    if certificate_ready:
        return "allow_certificate_ready"
    return "allow_postclose_fg_classification"
