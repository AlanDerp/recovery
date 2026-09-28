"""Closed FG outer retry policy for LIBERO acquisition.

This module deliberately separates two consumers of the same endpoint event:

* :func:`route_endpoint` is the hard SL projection.  It owns only recovery,
  observation boundaries, and transport admission; it never proposes a grasp.
* :func:`consume_student_report` is the soft projection.  It publishes a
  finite, non-numeric account of the previous acquisition to the next native
  regeneration turn.

The policy is total over the post-close attachment/contact partition.  A miss
is therefore a retry state, never evidence that acquisition is impossible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import os
from typing import Any


ENV = "CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY"
CLEAN_C_ENV = "CAPX_LIBERO_G17_V355_CLEAN_C"
DISABLE_SOFT_GAIN_ENV = "CAPX_LIBERO_G17_V354_DISABLE_SOFT_GAIN"


class EndpointKind(str, Enum):
    TARGET_HELD = "target_held"
    WRONG_OBJECT_HELD = "wrong_object_held"
    EMPTY_TARGET_NEAR = "empty_target_near"
    EMPTY_BLOCKED = "empty_blocked"
    EMPTY_CLEAR_OFFSET = "empty_clear_offset"


class MotionOutcome(str, Enum):
    REACHED = "reached"
    PARTIAL_OR_BLOCKED = "partial_or_blocked"
    NOT_EXECUTED = "not_executed"


class RecoveryStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    INVERSE_COMPLETED = "inverse_completed"
    INVERSE_UNAVAILABLE = "inverse_unavailable"


class PrecloseKind(str, Enum):
    PROPOSAL_UNAVAILABLE = "proposal_unavailable"
    INGRESS_UNREACHED = "ingress_unreached"


@dataclass(frozen=True)
class EndpointEvent:
    """Finite controller-visible account of one physical close."""

    attempt: int | None
    object_name: str
    kind: EndpointKind
    attachment: str
    target_contact: bool
    tip_in_target_volume: bool
    relative_x: str
    relative_y: str
    relative_z: str
    motion_outcome: MotionOutcome
    proposal_family: str
    inverse_ingress_available: bool
    recovery_status: RecoveryStatus = RecoveryStatus.NOT_REQUIRED


@dataclass(frozen=True)
class PrecloseEvent:
    """Typed retry event produced before a physical close."""

    attempt: int | None
    object_name: str
    kind: PrecloseKind
    proposal_family: str
    physical_command_executed: bool


@dataclass(frozen=True)
class HardRoute:
    """Unique motor/control successor for an endpoint event."""

    open_gripper: bool
    execute_inverse: bool
    execute_gr: bool
    regenerate: bool
    admit_transport: bool
    next_state: str


# Trials are process-isolated in the fair-live harness.  Keeping this mailbox
# in-process avoids exposing private continuous state or depending on audit
# files as a control channel.
RetryEvent = EndpointEvent | PrecloseEvent

_PENDING: dict[int, RetryEvent] = {}
_HISTORY: dict[int, list[RetryEvent]] = {}


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def clean_c_enabled() -> bool:
    return enabled() and os.environ.get(CLEAN_C_ENV, "0") == "1"


def soft_gain_enabled() -> bool:
    """Whether endpoint FG is projected into the next student turn."""

    return enabled() and os.environ.get(DISABLE_SOFT_GAIN_ENV, "0") != "1"


def reset_trial(trial: int) -> None:
    _PENDING.pop(int(trial), None)
    _HISTORY.pop(int(trial), None)


def _proposal_family(transaction: dict[str, Any] | None) -> str:
    source = str((transaction or {}).get("source", "")).lower()
    if "certified_topdown" in source:
        return "CERTIFIED_TOPDOWN"
    if "graspnet" in source:
        return "NATIVE_GRASPNET"
    if source:
        return "NATIVE_OTHER"
    return "UNBOUND_NATIVE"


def _motion_outcome(transaction: dict[str, Any] | None) -> MotionOutcome:
    if not isinstance(transaction, dict):
        return MotionOutcome.NOT_EXECUTED
    if transaction.get("native_motion_ok") is False:
        return MotionOutcome.PARTIAL_OR_BLOCKED
    if transaction.get("contact_reached") is True:
        return MotionOutcome.REACHED
    if transaction.get("ingress_attempted") and transaction.get("executed"):
        return MotionOutcome.REACHED
    return MotionOutcome.NOT_EXECUTED


def classify_endpoint(
    close_fg: Any,
    transaction: dict[str, Any] | None,
    *,
    inverse_ingress_available: bool,
) -> EndpointEvent:
    """Compile the exhaustive post-close partition from closed predicates."""

    attachment = str(close_fg.attachment)
    firm = bool(close_fg.firm_acquired)
    target_near = bool(
        close_fg.target_contact
        or close_fg.tip_in_target_volume
        or close_fg.coordinate_in_target_band
    )
    motion = _motion_outcome(transaction)
    if firm and attachment == "holding_payload":
        kind = EndpointKind.TARGET_HELD
    elif attachment != "empty":
        kind = EndpointKind.WRONG_OBJECT_HELD
    elif target_near:
        kind = EndpointKind.EMPTY_TARGET_NEAR
    elif motion == MotionOutcome.PARTIAL_OR_BLOCKED:
        kind = EndpointKind.EMPTY_BLOCKED
    else:
        kind = EndpointKind.EMPTY_CLEAR_OFFSET
    return EndpointEvent(
        attempt=(transaction or {}).get("attempt"),
        object_name=str((transaction or {}).get("object_name", "")),
        kind=kind,
        attachment=attachment,
        target_contact=bool(close_fg.target_contact),
        tip_in_target_volume=bool(close_fg.tip_in_target_volume),
        relative_x=str(close_fg.relative_x),
        relative_y=str(close_fg.relative_y),
        relative_z=str(close_fg.relative_z),
        motion_outcome=motion,
        proposal_family=_proposal_family(transaction),
        inverse_ingress_available=bool(inverse_ingress_available),
    )


def route_endpoint(
    event: EndpointEvent, *, gr_already_attempted: bool = False
) -> HardRoute:
    """Return the unique successor of one closed endpoint FG event.

    Clean C inserts adaptive GR only for ``EmptyTargetNear``.  All remaining
    non-target endpoints, and a NearMiss after its one bounded GR call, return
    to the full-domain native retry loop.
    """

    inverse = bool(event.inverse_ingress_available)
    if event.kind == EndpointKind.TARGET_HELD:
        return HardRoute(
            open_gripper=False,
            execute_inverse=inverse,
            execute_gr=False,
            regenerate=not inverse,
            admit_transport=inverse,
            next_state=(
                "escaped_holding" if inverse else "observe_holding_in_place"
            ),
        )
    if (
        clean_c_enabled()
        and event.kind == EndpointKind.EMPTY_TARGET_NEAR
        and inverse
        and not gr_already_attempted
    ):
        return HardRoute(
            open_gripper=True,
            execute_inverse=True,
            execute_gr=True,
            regenerate=False,
            admit_transport=False,
            next_state="inverse_escape_then_adaptive_GR",
        )
    return HardRoute(
        open_gripper=True,
        execute_inverse=inverse,
        execute_gr=False,
        regenerate=True,
        admit_transport=False,
        next_state=(
            "inverse_escape_then_FG_regeneration"
            if inverse
            else "current_world_FG_regeneration"
        ),
    )


def finalize_recovery(
    event: EndpointEvent, *, inverse_completed: bool
) -> EndpointEvent:
    """Attach recovery truth without changing the close-time classification."""

    if event.inverse_ingress_available:
        status = (
            RecoveryStatus.INVERSE_COMPLETED
            if inverse_completed
            else RecoveryStatus.INVERSE_UNAVAILABLE
        )
    else:
        status = RecoveryStatus.INVERSE_UNAVAILABLE
    return replace(event, recovery_status=status)


def publish(event: RetryEvent, *, trial: int | None = None) -> None:
    """Publish exactly one endpoint event to the next student FG turn."""

    key = int(
        trial
        if trial is not None
        else os.environ.get("CAPX_ACTIVE_TRIAL", "1")
    )
    _PENDING[key] = event
    _HISTORY.setdefault(key, []).append(event)
def event_dict(event: RetryEvent) -> dict[str, Any]:
    return asdict(event)


def proposal_unavailable_event(
    *, attempt: int | None, object_name: str
) -> PrecloseEvent:
    return PrecloseEvent(
        attempt=attempt,
        object_name=str(object_name),
        kind=PrecloseKind.PROPOSAL_UNAVAILABLE,
        proposal_family="NATIVE_GRASPNET",
        physical_command_executed=False,
    )


def ingress_unreached_event(
    transaction: dict[str, Any], *, physical_command_executed: bool
) -> PrecloseEvent:
    return PrecloseEvent(
        attempt=transaction.get("attempt"),
        object_name=str(transaction.get("object_name", "")),
        kind=PrecloseKind.INGRESS_UNREACHED,
        proposal_family=_proposal_family(transaction),
        physical_command_executed=bool(physical_command_executed),
    )


def _relation_line(axis: str, value: str) -> str:
    labels = {
        "NEGATIVE_OF_REFERENCE_BAND": "negative of the target band",
        "WITHIN_REFERENCE_BAND": "within the target band",
        "POSITIVE_OF_REFERENCE_BAND": "positive of the target band",
    }
    return f"- fact: gripper {axis} is {labels.get(value, value.lower())}."


def render_student_report(event: RetryEvent, *, repeated: bool) -> str:
    """Render the soft projection; no continuous values or private labels."""

    lines = ["Verified acquisition retry evidence:"]
    if isinstance(event, PrecloseEvent):
        if event.kind == PrecloseKind.PROPOSAL_UNAVAILABLE:
            lines.append(
                "- fact: the previous perception call produced no grasp proposal."
            )
        else:
            lines.append(
                "- fact: the previous acquisition ingress did not reach its requested endpoint."
            )
        lines.extend(
            [
                "- fact: no physical close belongs to that acquisition attempt.",
                "- fact: acquisition of the locked target remains incomplete.",
                "- fact: reason from the current endpoint evidence and preserve the locked target identity.",
            ]
        )
        if repeated:
            lines.append(
                "- fact: this proposal family has already produced the same retry state in this episode."
            )
        return "\n".join(lines) + "\n"
    if event.kind == EndpointKind.TARGET_HELD:
        lines.append("- fact: the locked target is held by the gripper.")
    elif event.kind == EndpointKind.WRONG_OBJECT_HELD:
        lines.append("- fact: the gripper held a movable other than the locked target.")
    else:
        lines.append("- fact: the gripper is empty after the most recent close.")
    lines.extend(
        [
            f"- fact: target contact at close: {'present' if event.target_contact else 'absent'}.",
            _relation_line("x", event.relative_x),
            _relation_line("y", event.relative_y),
            _relation_line("z", event.relative_z),
        ]
    )
    if event.tip_in_target_volume:
        lines.append("- fact: the gripper tip was inside the target volume at close.")
    if event.motion_outcome == MotionOutcome.PARTIAL_OR_BLOCKED:
        lines.append("- fact: the commanded ingress was only partially realized.")
    if event.recovery_status == RecoveryStatus.INVERSE_COMPLETED:
        lines.append("- fact: the recorded ingress was reversed before this observation.")
    elif event.recovery_status == RecoveryStatus.INVERSE_UNAVAILABLE:
        lines.append("- fact: no completed inverse ingress precedes this observation.")
    if repeated:
        lines.append(
            "- fact: this proposal family has already produced an empty close in this episode."
        )
    if event.kind != EndpointKind.TARGET_HELD:
        lines.extend(
            [
                "- fact: acquisition of the locked target remains incomplete.",
                "- fact: the previous grasp pose is not established as reusable.",
                "- fact: reason from the current endpoint evidence and preserve the locked target identity.",
            ]
        )
    return "\n".join(lines) + "\n"


def consume_student_report(trial: int) -> tuple[str, dict[str, Any]] | None:
    """Pop the pending soft projection for one turn and return audit metadata."""

    key = int(trial)
    event = _PENDING.pop(key, None)
    # The control arm still records the exact same endpoint history and uses
    # the exact same hard recovery router.  It only withholds that evidence
    # from proposal generation.  Pop first so toggling the ablation cannot
    # leak a stale event into a later turn.
    if event is None or not soft_gain_enabled():
        return None
    prior = _HISTORY.get(key, [])[:-1]
    repeated = bool(
        not (
            isinstance(event, EndpointEvent)
            and event.kind == EndpointKind.TARGET_HELD
        )
        and any(
            old.proposal_family == event.proposal_family
            and old.kind == event.kind
            for old in prior
        )
    )
    metadata = {
        "schema": "capx_libero_g17_v354_fg_native_retry_projection_v1",
        "retry_event_kind": event.kind.value,
        "proposal_family": event.proposal_family,
        "same_family_retry_state_repeated": repeated,
        "continuous_privileged_world_input": False,
    }
    if isinstance(event, EndpointEvent):
        metadata.update(
            {
                "hard_endpoint_kind": event.kind.value,
                "motion_outcome": event.motion_outcome.value,
                "recovery_status": event.recovery_status.value,
            }
        )
    else:
        metadata["physical_command_executed"] = (
            event.physical_command_executed
        )
    return render_student_report(event, repeated=repeated), metadata
