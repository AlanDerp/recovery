from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with the API described below.
Goal: pick up the tennis ball and put it into the open box, then release it.

Key rules:
- Use "tennis ball" as the grasp target.
- Use "open box" or "box" to locate the container.
- The extent from get_object_pose(..., return_bbox_extent=True) is the FULL side length. Use half the relevant extent for height/radius calculations.
- The quaternion from get_object_pose is unreliable for placement. Reuse the grasp quaternion from sample_grasp_pose for carrying and placing the ball.
- Always use z_approach=0.1 when approaching the ball or the box opening.
- After grasping, lift the ball by at least 0.15 m before moving laterally.
- Place at the center of the box opening. Compute a low release height from the box top and the tennis ball radius so the ball is not dropped from a height.
- Retract upward after opening the gripper.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_CODE = """
import numpy as np

BALL_QUERY = "tennis ball"
BOX_QUERIES = ["open box", "box"]


def locate_first(query_names, return_bbox_extent=False):
    last_error = None
    for query_name in query_names:
        try:
            return get_object_pose(query_name, return_bbox_extent=return_bbox_extent)
        except Exception as exc:
            last_error = exc
            print(f"Could not locate {query_name!r}: {exc}")
    raise last_error


# Measure both objects before grasping so placement can be computed relative to the box.
_, _, ball_ext = get_object_pose(BALL_QUERY, return_bbox_extent=True)
box_pos, _, box_ext = locate_first(BOX_QUERIES, return_bbox_extent=True)

# Contact-GraspNet plans the actual tennis-ball grasp.
open_gripper()
pick_pos, pick_quat = sample_grasp_pose(BALL_QUERY)
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()

# Lift clear of the table before any lateral motion.
carry_pos = pick_pos.copy()
carry_pos[2] += 0.18
goto_pose(carry_pos, pick_quat)

# Release just above the box opening.  Use the largest ball extent as diameter because
# a tennis ball may have a slightly anisotropic reconstructed OBB.
ball_radius = float(np.max(ball_ext) / 2.0)
box_top_z = float(box_pos[2] + box_ext[2] / 2.0)

place_pos = box_pos.copy()
place_pos[2] = box_top_z + ball_radius + 0.02

pre_place_pos = place_pos.copy()
pre_place_pos[2] += 0.10
goto_pose(pre_place_pos, pick_quat)
goto_pose(place_pos, pick_quat, z_approach=0.1)
open_gripper()

# Retract without pushing the ball or box.
retract_pos = place_pos.copy()
retract_pos[2] += 0.12
goto_pose(retract_pos, pick_quat)
"""


class FrankaTennisBallToBoxCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for putting a tennis ball into a box."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "FrankaTennisBallToBoxCodeEnv",
]
