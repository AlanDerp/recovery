from __future__ import annotations

import numpy as np
import pytest

from scripts.r1lite_identify_joint_response import (
    JOINT_INDEX,
    SafetyAbort,
    SafetyLimits,
    analyze_phase,
    classify_sweep,
    validate_feedback_sample,
)


def _samples(target_values: list[float], *, inactive_delta: float = 0.0):
    rows = []
    for index, target in enumerate(target_values):
        feedback = np.zeros(6, dtype=float)
        feedback[JOINT_INDEX] = target
        feedback[0] = inactive_delta * index / max(1, len(target_values) - 1)
        rows.append(
            {
                "monotonic_s": index * 0.05,
                "feedback": feedback.tolist(),
            }
        )
    return rows


def _probe(delta: float, values: list[float]) -> dict:
    return {
        "run_id": f"probe_{delta:+.3f}",
        "phase": "probe",
        "command_delta_rad": delta,
        "analysis": analyze_phase(_samples(values), command_delta_rad=delta),
    }


def test_analyze_phase_reports_normal_gain_and_latency() -> None:
    result = analyze_phase(
        _samples([0.0, 0.0, 0.0015, 0.003, 0.004, 0.004]),
        command_delta_rad=0.004,
    )

    assert result["classification"] == "responsive"
    assert result["gain"] == pytest.approx(1.0)
    assert result["latency_s"] == pytest.approx(0.1)
    assert result["cross_coupling_rad"] == 0.0


def test_sweep_reports_deadband_and_bidirectional_response() -> None:
    summaries = [
        _probe(0.002, [0.0] * 6),
        _probe(-0.002, [0.0] * 6),
        _probe(0.006, [0.0, 0.002, 0.005, 0.006, 0.006]),
        _probe(-0.006, [0.0, -0.002, -0.005, -0.006, -0.006]),
    ]

    result = classify_sweep(summaries)

    assert result["classification"] == "normal_local_response"
    assert result["deadband_upper_bound_rad"] == pytest.approx(0.006)
    assert result["median_local_gain"] == pytest.approx(1.0)


def test_sweep_reports_zero_response() -> None:
    result = classify_sweep(
        [_probe(0.010, [0.0] * 8), _probe(-0.010, [0.0] * 8)]
    )

    assert result["classification"] == "no_response_or_control_path_fault"
    assert result["deadband_upper_bound_rad"] is None


def test_sweep_reports_reversed_response() -> None:
    result = classify_sweep(
        [_probe(0.004, [0.0, -0.002, -0.004, -0.004])]
    )

    assert result["classification"] == "direction_reversed"


def test_sample_gate_rejects_cross_coupling() -> None:
    limits = SafetyLimits(cross_coupling_rad=0.01)
    start = np.zeros(6)
    current = start.copy()
    current[0] = 0.011

    with pytest.raises(SafetyAbort, match="Inactive-joint motion"):
        validate_feedback_sample(
            current,
            start,
            start,
            phase="probe",
            command_delta_rad=0.004,
            limits=limits,
        )


def test_sample_gate_rejects_jump_and_reverse_motion() -> None:
    start = np.zeros(6)
    jump = start.copy()
    jump[JOINT_INDEX] = 0.021
    with pytest.raises(SafetyAbort, match="sample jump"):
        validate_feedback_sample(
            jump,
            start,
            start,
            phase="probe",
            command_delta_rad=0.004,
            limits=SafetyLimits(sample_jump_rad=0.02),
        )

    reverse = start.copy()
    reverse[JOINT_INDEX] = -0.002
    with pytest.raises(SafetyAbort, match="opposite"):
        validate_feedback_sample(
            reverse,
            start,
            start,
            phase="probe",
            command_delta_rad=0.004,
            limits=SafetyLimits(),
        )
