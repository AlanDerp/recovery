"""Semantic closure around the frozen v318 adaptive grasp recovery.

This module owns neither grasp generation nor near-miss correction.  It only
closes the two native transition boundaries adjacent to adaptive GR:

* an executed contact endpoint is offered to closed FG before a numerical IK
  tolerance can discard it and regenerate;
* payload loss ends the stale native code block immediately;
* a numerical goto miss while still holding preserves native ownership and
  the symbolic obligation to reach PlacementNearMiss.
"""

from __future__ import annotations

import os
from typing import Any


ENV = "CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE"
SEMANTIC_ENV = "CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION"
PLACEMENT_OBLIGATION = "escaped_holding_to_placement_near_miss"


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1" or semantic_enabled()


def semantic_enabled() -> bool:
    return os.environ.get(SEMANTIC_ENV, "0") == "1"


def contact_endpoint_near_miss(api: Any) -> tuple[bool, Any]:
    """Return whether exact FG admits this executed endpoint as NearMiss."""
    from capx.integrations.franka import grasp_recovery

    fg = grasp_recovery.capture_close_time_fg(api)
    return bool(fg.adaptive_near_miss), fg


def post_capture_transition(attachment: str | None, motion_ok: bool) -> str:
    """Classify one native goto without confusing tolerance with task state."""
    if attachment != "holding_payload":
        return "payload_lost_return_start"
    if not motion_ok:
        return "holding_native_plan_continue"
    return "holding_native_goto_complete"


def arm_transport(
    api: Any, attachment: str | None, *, escaped: bool = False
) -> bool:
    """Bind native ownership to a verified FirmAcquired+Escaped event."""
    armed = attachment == "holding_payload" and escaped
    api._capx_v319_transport_armed = armed
    api._capx_v320_transport_state = (
        "escaped_holding" if armed else "inactive"
    )
    api._capx_v320_symbolic_obligation = (
        PLACEMENT_OBLIGATION if armed else None
    )
    return armed


def transport_armed(api: Any) -> bool:
    """Return whether a captured payload is still inside its transport epoch."""
    return bool(getattr(api, "_capx_v319_transport_armed", False))


def disarm_transport(api: Any) -> None:
    """Close the transport epoch after payload loss or a physical release."""
    api._capx_v319_transport_armed = False
    api._capx_v320_transport_state = "inactive"
    api._capx_v320_symbolic_obligation = None


def symbolic_obligation(api: Any) -> str | None:
    """Return the discrete FGSL post-acquisition obligation, if active."""
    return getattr(api, "_capx_v320_symbolic_obligation", None)
