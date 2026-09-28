"""POC acquisition extraction by reversing the observed ingress endpoint.

No retreat distance is defined here.  ``goto_pose(..., z_approach=...)`` owns
the approach geometry; this module records the reached pregrasp configuration
and, immediately after close, commands that same configuration again.  The
behavior therefore inherits the caller's approach vector and magnitude and
works for non-vertical approaches without inventing another trajectory.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


ENV = "CAPX_LIBERO_G17_V304_INGRESS_RETREAT"
COMPLETE_INVERSE_ENV = "CAPX_LIBERO_G17_V353_COMPLETE_INVERSE_ESCAPE"


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def record_ingress(
    api: Any,
    *,
    attempt: int | None,
    object_name: str | None,
    arm_cfg: np.ndarray,
    hand_target_xyz: np.ndarray,
    hand_target_wxyz: np.ndarray,
) -> None:
    """Cache the actually reached pregrasp endpoint for this attempt."""
    if not enabled():
        return
    api._capx_v304_ingress = {
        "attempt": attempt,
        "object_name": object_name,
        "arm_cfg": np.asarray(arm_cfg, dtype=np.float64).reshape(-1)[:7].copy(),
        "hand_target_xyz": np.asarray(hand_target_xyz, dtype=np.float64)
        .reshape(3)
        .copy(),
        "hand_target_wxyz": np.asarray(hand_target_wxyz, dtype=np.float64)
        .reshape(4)
        .copy(),
    }


def has_current_ingress(api: Any) -> bool:
    """Whether this attempt owns one unused measured inverse endpoint."""

    if not enabled():
        return False
    ingress = getattr(api, "_capx_v304_ingress", None)
    if not isinstance(ingress, dict):
        return False
    current_attempt = getattr(api, "_capx_v293_attempt", None)
    return ingress.get("attempt") == current_attempt


def retreat_after_close(api: Any) -> dict[str, Any] | None:
    """Return to the cached ingress configuration once, without GT gating."""
    if not enabled():
        return None
    ingress = getattr(api, "_capx_v304_ingress", None)
    api._capx_v304_ingress = None
    if not isinstance(ingress, dict):
        return None

    from capx.integrations.franka import acquisition_audit as _acq
    from capx.integrations.franka.tip_xy_only import _live_actuated_cfg

    current_attempt, current_object = _acq.active_attempt(api)
    if ingress["attempt"] != current_attempt:
        _acq.audit(
            api,
            "ingress_retreat_skipped",
            attempt=current_attempt,
            object_name=current_object,
            reason="stale_attempt",
            ingress_attempt=ingress["attempt"],
        )
        return None

    start = _live_actuated_cfg(api)
    target = np.asarray(ingress["arm_cfg"], dtype=np.float64).reshape(7)
    _acq.audit(
        api,
        "ingress_retreat_command",
        attempt=current_attempt,
        object_name=current_object,
        source="inverse_of_reached_approach_endpoint",
        target_arm_q=target.round(7).tolist(),
        start_arm_delta_max_rad=float(np.max(np.abs(start[:7] - target))),
        hand_target_xyz=np.asarray(ingress["hand_target_xyz"]).round(7).tolist(),
        hand_target_wxyz=np.asarray(ingress["hand_target_wxyz"]).round(7).tolist(),
    )
    # Reuse the same command-level execution budget as native goto.  The
    # simulator's legacy 120-step default can return halfway through a valid
    # inverse (observed residuals 0.84--1.16 rad), leaving the next FG view in
    # the failed contact configuration.  This changes only the maximum: the
    # controller still exits immediately when it reaches tolerance.
    from capx.integrations.franka.measured_seed_ik import (
        NATIVE_CONTROLLER_MAX_STEPS,
    )

    if os.environ.get(COMPLETE_INVERSE_ENV, "0") == "1":
        api._env.move_to_joints_blocking(
            target, max_steps=NATIVE_CONTROLLER_MAX_STEPS
        )
    else:
        api._env.move_to_joints_blocking(target)
    final = _live_actuated_cfg(api)
    report = {
        "source": "inverse_of_reached_approach_endpoint",
        "final_arm_error_max_rad": float(np.max(np.abs(final[:7] - target))),
        "measured_arm_delta_max_rad": float(np.max(np.abs(final[:7] - start[:7]))),
        "joint_move_report": getattr(api._env, "_last_joint_move_report", None),
    }
    _acq.audit(
        api,
        "ingress_retreat_result",
        attempt=current_attempt,
        object_name=current_object,
        **report,
    )
    # Evaluator-only witness.  It is recorded after the command and never
    # gates whether the retreat command is issued.
    _acq.audit_attachment(api, "extraction_capture_result")
    api._capx_v304_extraction_complete = True
    return report
