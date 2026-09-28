"""Post-miss clear-cache: drop prior code history after a wrong release near-miss.

When enabled, a placement_xy near-miss with empty hand (typically after a bad
release) triggers a prompt-side reset: only the frozen code prefix remains as
``executed_code``, prior regenerate thrash is omitted, and the student continues
from current FG + goal (with between→ordinal lock applied in takeover).
"""

from __future__ import annotations

import os
from typing import Any


def post_miss_clear_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE", "0") == "1"


def goal_lock_between_to_ordinal_enabled() -> bool:
    """between-clause → locked ordinal (right/left/…) without full Goal takeover env."""
    if post_miss_clear_enabled():
        return True
    return os.environ.get("CAPX_LIBERO_G17_GOAL_LOCK_BETWEEN_TO_ORDINAL", "0") == "1"


def should_clear_after_projection(proj: Any) -> bool:
    """True when endpoint is empty-hand placement_xy near-miss (post bad release)."""
    if not post_miss_clear_enabled():
        return False
    if getattr(proj, "task_gap", None) != "placement_xy":
        return False
    if getattr(proj, "executor_attachment", None) != "empty":
        return False
    report = (getattr(proj, "combined_report", None) or "").lower()
    near = "placement near-miss" in report or (
        "not center-to-center aligned" in report and "center_on_surface: on" in report
    )
    if not near:
        return False
    if getattr(proj, "released_goal_after_transport", False):
        return True
    gain = getattr(proj, "gain", None)
    if gain is not None and getattr(gain, "bound_release", False):
        return True
    # Already thrashing in near-miss with empty hand — keep clearing.
    return True


def context_reset_note_lines() -> list[str]:
    return [
        "Verified context reset:",
        "- fact: prior code attempts are omitted from this prompt.",
        "- fact: continue from the current endpoint evidence and goal only.",
    ]
