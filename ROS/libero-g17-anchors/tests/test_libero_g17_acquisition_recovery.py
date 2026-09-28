import numpy as np

from capx.integrations.franka import acquisition_recovery as recovery


def _row(step: int, residual: float) -> dict[str, float | int]:
    return {"step": step, "joint_error_norm_rad": residual}


def test_strong_stall_matches_task4_plateau() -> None:
    progress = [
        _row(0, 2.92),
        _row(120, 1.18),
        _row(200, 0.987),
        _row(220, 0.965),
        _row(240, 0.945),
        _row(260, 0.926),
        _row(280, 0.909),
    ]
    evidence = recovery.detect_strong_stall(progress)
    assert evidence is not None
    assert evidence["step"] == 280
    assert evidence["window_improvement_rad"] < 0.08


def test_normal_progress_and_small_terminal_residual_do_not_stall() -> None:
    normal = [_row(step, 1.0 - step / 300.0) for step in range(0, 241, 20)]
    assert recovery.detect_strong_stall(normal) is None
    small = [_row(120, 0.20), _row(200, 0.19), _row(280, 0.18)]
    assert recovery.detect_strong_stall(small) is None


def test_rollback_reverses_measured_path_and_ends_at_start() -> None:
    history = [
        (0, np.zeros(7)),
        (20, np.full(7, 0.10)),
        (40, np.full(7, 0.11)),
        (60, np.full(7, 0.30)),
    ]
    checkpoints = recovery.rollback_checkpoints(history)
    np.testing.assert_allclose(checkpoints[0], 0.30)
    np.testing.assert_allclose(checkpoints[-1], 0.0)
    assert len(checkpoints) == 3
