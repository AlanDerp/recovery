import numpy as np
import pytest

from scheduler.action_safety import validate_action


SAFETY = {
    "allowed_action_parts": ["right_arm", "right_gripper"],
    "max_right_arm_delta_rad": 0.015,
    "right_gripper_min": 0.0,
    "right_gripper_max": 100.0,
}


def test_accepts_bounded_right_arm_action():
    state = {"right_arm": np.zeros(6, dtype=np.float32)}
    action = {"right_arm": np.full(6, 0.01, dtype=np.float32)}

    validated = validate_action(action, state, SAFETY)

    np.testing.assert_allclose(validated["right_arm"], action["right_arm"])


@pytest.mark.parametrize(
    "action",
    [
        {"left_arm": np.zeros(6)},
        {"right_arm": np.full(6, 0.02)},
        {"right_arm": np.array([0, 0, 0, 0, 0, np.nan])},
        {"right_gripper": np.array([101.0])},
    ],
)
def test_rejects_unsafe_action(action):
    with pytest.raises(ValueError):
        validate_action(action, {"right_arm": np.zeros(6)}, SAFETY)
