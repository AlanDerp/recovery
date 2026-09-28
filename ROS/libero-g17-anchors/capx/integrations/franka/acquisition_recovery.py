"""Proprioceptive stall detection and reversible acquisition rollback.

No simulator contacts or object poses are controller inputs.  Detection uses
only commanded-vs-measured joint residual history.  The rollback path is the
reverse of configurations the robot actually traversed during this command.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


_SAMPLE_EVERY_STEPS = 20
_MIN_DETECTION_STEP = 200
_WINDOW_STEPS = 80
_MIN_RESIDUAL_NORM_RAD = 0.25
_MIN_WINDOW_IMPROVEMENT_RAD = 0.08


def enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V297B_ACQUISITION_RECOVERY", "0") == "1"


def detect_strong_stall(progress: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return evidence only for a sustained, large-residual plateau."""
    if not progress:
        return None
    current = progress[-1]
    step = int(current["step"])
    residual = float(current["joint_error_norm_rad"])
    if step < _MIN_DETECTION_STEP or residual <= _MIN_RESIDUAL_NORM_RAD:
        return None
    prior = None
    for row in reversed(progress[:-1]):
        if int(row["step"]) <= step - _WINDOW_STEPS:
            prior = row
            break
    if prior is None:
        return None
    improvement = float(prior["joint_error_norm_rad"]) - residual
    if improvement >= _MIN_WINDOW_IMPROVEMENT_RAD:
        return None
    return {
        "schema": "capx_libero_v297b_acquisition_stall_v1",
        "source": "commanded_vs_measured_joint_residual",
        "step": step,
        "window_start_step": int(prior["step"]),
        "window_start_residual_norm_rad": float(prior["joint_error_norm_rad"]),
        "current_residual_norm_rad": residual,
        "window_improvement_rad": improvement,
        "minimum_required_improvement_rad": _MIN_WINDOW_IMPROVEMENT_RAD,
    }


def rollback_checkpoints(
    measured_history: list[tuple[int, np.ndarray]],
    *,
    minimum_joint_spacing_rad: float = 0.025,
) -> list[np.ndarray]:
    """Return a deduplicated reverse path ending at the measured start q."""
    if not measured_history:
        return []
    reverse: list[np.ndarray] = []
    for _, cfg in reversed(measured_history):
        q = np.asarray(cfg, dtype=np.float64).reshape(7)
        if not reverse or np.max(np.abs(q - reverse[-1])) >= minimum_joint_spacing_rad:
            reverse.append(q.copy())
    start = np.asarray(measured_history[0][1], dtype=np.float64).reshape(7)
    if not reverse or np.max(np.abs(start - reverse[-1])) > 1e-8:
        reverse.append(start.copy())
    return reverse
