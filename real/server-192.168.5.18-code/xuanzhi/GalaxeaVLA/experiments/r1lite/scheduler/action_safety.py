import numpy as np


def validate_action(action: dict, state: dict, safety: dict) -> dict:
    """Return a finite, allow-listed action bounded against robot feedback."""
    allowed = set(safety.get("allowed_action_parts", ()))
    unexpected = set(action) - allowed
    if unexpected:
        raise ValueError(f"model returned disallowed action parts: {sorted(unexpected)}")

    validated = {}
    for part, values in action.items():
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{part} action contains non-finite values")
        if part == "right_arm":
            feedback = np.asarray(state.get(part), dtype=np.float32).reshape(-1)
            if array.shape != feedback.shape:
                raise ValueError(
                    f"right_arm shape mismatch: action={array.shape}, state={feedback.shape}"
                )
            max_delta = float(np.max(np.abs(array - feedback)))
            limit = float(safety.get("max_right_arm_delta_rad", 0.015))
            if max_delta > limit:
                raise ValueError(
                    f"right_arm delta {max_delta:.4f}rad exceeds {limit:.4f}rad"
                )
        elif part == "right_gripper":
            low = float(safety.get("right_gripper_min", 0.0))
            high = float(safety.get("right_gripper_max", 100.0))
            if np.any(array < low) or np.any(array > high):
                raise ValueError(
                    f"right_gripper action outside [{low:.3f}, {high:.3f}]"
                )
        validated[part] = array
    return validated
