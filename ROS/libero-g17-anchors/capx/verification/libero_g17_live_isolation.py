"""v270 live-policy isolation guards for LIBERO ReleaseReady.

The live student retains normal grasp and transport control.  At a release
intent, the existing v269 controller runs synchronously and owns the robot
until it either authorizes or refuses the release.  Once LIBERO reports a
terminal state, that state is absorbing for every remaining mutating call in
the current generated code block.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ENV = "CAPX_LIBERO_G17_V270_LIVE_ISOLATION"
FAIR_LIVE_ENV = "CAPX_LIBERO_G17_V272_FAIR_LIVE"
RELEASE_COMMIT_ENV = "CAPX_LIBERO_G17_V284_RELEASE_COMMIT_LATCH"
FULL_RELEASE_ENV = "CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW"


def enabled() -> bool:
    return (
        os.environ.get(ENV, "0") == "1"
        or os.environ.get(FAIR_LIVE_ENV, "0") == "1"
    )


def release_commit_latch_enabled() -> bool:
    return bool(
        os.environ.get(RELEASE_COMMIT_ENV, "0") == "1"
        or os.environ.get(FULL_RELEASE_ENV, "0") == "1"
    )


def audit(owner: Any, event: str, **fields: Any) -> None:
    if not enabled():
        return
    root = os.environ.get("CAPX_REPRO_AUDIT_DIR", "").strip()
    if not root:
        raise RuntimeError("v270 requires CAPX_REPRO_AUDIT_DIR")
    path = Path(root) / "libero_g17_v270_live_isolation" / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    index = int(getattr(owner, "_capx_v270_event_idx", 0))
    owner._capx_v270_event_idx = index + 1
    row = {
        "schema": "capx_libero_g17_v270_live_isolation_trace_v1",
        "event_index": index,
        "event": str(event),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def reset_terminal_latch(env: Any) -> None:
    env._capx_v270_terminal_latched = False
    env._capx_v270_terminal_reason = None
    # A placement predicate may become transiently true while the gripper is
    # still holding the payload over / against the receptacle.  Such a state is
    # not a completed ReleaseReady placement and must not make control
    # absorbing before open_gripper() actually executes.
    env._capx_v270_success_latch_armed = not release_commit_latch_enabled()
    env._capx_v284_release_window_active = False
    env._capx_v284_release_success_deferred = False


def arm_success_latch_for_release(env: Any) -> None:
    """Allow task-success latching for the physical gripper-open interval."""
    if not enabled() or not release_commit_latch_enabled():
        return
    env._capx_v270_success_latch_armed = True
    env._capx_v284_release_window_active = True
    env._capx_v284_release_success_deferred = False
    audit(env, "success_latch_armed", reason="release_committed")


def disarm_success_latch_after_release(env: Any) -> None:
    """Finish all open steps, then recognize success or close the window."""
    if not enabled() or not release_commit_latch_enabled():
        return
    if bool(getattr(env, "_capx_v270_terminal_latched", False)):
        return
    was_active = bool(getattr(env, "_capx_v284_release_window_active", False))
    env._capx_v284_release_window_active = False
    if was_active and refresh_terminal_latch(
        env, source="release_window_complete"
    ):
        return
    env._capx_v270_success_latch_armed = False
    audit(env, "success_latch_disarmed", reason="release_window_complete")


def refresh_terminal_latch(env: Any, *, source: str) -> bool:
    """Latch success/done after a low-level step; never clear until reset."""
    if not enabled():
        return False
    if bool(getattr(env, "_capx_v270_terminal_latched", False)):
        return True

    success = False
    try:
        success = bool(env.task_completed())
    except Exception as exc:  # observation must not make control fail open
        audit(env, "terminal_probe_error", source=source, error=repr(exc))
    done = bool(getattr(env, "_current_done", False))
    success_armed = bool(getattr(env, "_capx_v270_success_latch_armed", True))
    release_window_active = bool(
        getattr(env, "_capx_v284_release_window_active", False)
    )
    # LIBERO commonly reports done together with check_success().  Ignore that
    # pair until release is committed; still preserve genuine non-success
    # terminal conditions such as a horizon timeout.
    latch_success = success and success_armed and not release_window_active
    latch_failure = done and not success
    if success and release_window_active:
        if not bool(
            getattr(env, "_capx_v284_release_success_deferred", False)
        ):
            env._capx_v284_release_success_deferred = True
            audit(
                env,
                "release_success_deferred",
                source=source,
                sim_step_count=int(getattr(env, "_sim_step_count", -1)),
            )
    elif success and not success_armed:
        audit(
            env,
            "premature_success_ignored",
            source=source,
            sim_step_count=int(getattr(env, "_sim_step_count", -1)),
        )
    if latch_success or latch_failure:
        reason = "task_completed" if success else "environment_done"
        env._capx_v270_terminal_latched = True
        env._capx_v270_terminal_reason = reason
        audit(
            env,
            "terminal_latched",
            source=source,
            reason=reason,
            sim_step_count=int(getattr(env, "_sim_step_count", -1)),
        )
        return True
    return False


def action_blocked(api: Any, action: str) -> bool:
    """Return True after terminal latch and record the suppressed mutation."""
    if not enabled():
        return False
    env = getattr(api, "_env", api)
    if not bool(getattr(env, "_capx_v270_terminal_latched", False)):
        return False
    audit(
        env,
        "action_suppressed",
        action=str(action),
        reason=getattr(env, "_capx_v270_terminal_reason", "terminal"),
    )
    return True
