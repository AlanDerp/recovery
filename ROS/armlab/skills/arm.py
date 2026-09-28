"""Hand-feel library for the Panda arm under robosuite's BASIC/OSC_POSE controller.

Everything here was measured, not assumed. See MEASUREMENTS below.

MEASUREMENTS (robosuite 1.5.2, Panda, Stack env, control_freq=20)
----------------------------------------------------------------
frames      robot0_base xmat == identity, so OSC "base" delta axes coincide
            with world axes. Action dims 0..2 map to world x,y,z directly.
scale       output_max = 0.05 m and 0.5 rad per step, action in [-1, 1].
lag         a sustained +0.5 on dx for 30 steps yields only 0.183 m of travel
            (~24% of the 0.75 m commanded) -> OSC is a soft impedance target,
            never assume the commanded delta is achieved.
gains       kp=20, kr=1.0 converge in ~25 steps to ~1 mm with no overshoot.
            kp=80 oscillates. kr=2.0 drags position by 4 mm.
gripper     fingers separate along the eef frame's Y axis. Open span 4.17 cm
            vs a 4 cm cubeA -> ~1 mm clearance per side, so yaw alignment is
            mandatory, not cosmetic.
geometry    eef_pos sits 1.14 cm above the fingertip midpoint along eef +Z.
            table surface z = 0.80, but a resting cube's center sits at 0.83
            (half-extent 0.02 + 1 cm of contact softening).
            cubeA half 0.020, cubeB half 0.025 -> stacked A center at 0.880.
success     Stack needs r_stack > 0: A lifted >4 cm above table_offset AND
            touching B AND the gripper NOT grasping. Releasing is part of the
            task, so a controller that never lets go always scores 0.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# Tuned in 02_probe_gripper_axis.py
KP_POS = 20.0
KR_ROT = 1.0

EEF_TO_FINGERTIP = 0.0114  # eef_pos is this far above the fingertip midpoint
TABLE_Z = 0.80
RESTING_CUBE_CENTER_Z = 0.83  # where a 4 cm cube actually settles

OPEN = -1.0
CLOSE = 1.0


def topdown_rotmat(yaw: float) -> np.ndarray:
    """Gripper pointing straight down (-Z approach), rotated by `yaw` about world Z."""
    return R.from_euler("z", yaw).as_matrix() @ R.from_euler("x", np.pi).as_matrix()


def yaw_of(quat) -> float:
    """Yaw about world Z from a robosuite (xyzw) quaternion."""
    return R.from_quat(quat).as_euler("xyz")[2]


def grasp_yaw_for_cube(cube_quat) -> float:
    """Pick the finger yaw that lines up with a square cube's faces.

    A cube has 90 deg symmetry, so wrap the cube's yaw into (-45, 45] and take
    the nearest face normal. Keeps the wrist from taking the long way around.
    """
    y = yaw_of(cube_quat)
    return (y + np.pi / 4) % (np.pi / 2) - np.pi / 4


# Measured gripper footprint in the eef frame, fully open (see 05/06 probes):
#   along the finger-open axis (eef Y): outer edge at 0.0512 + 0.0136 = 6.5 cm
#   perpendicular to it       (eef X): outer edge at only 1.1 cm
# So the hand is ~6x wider across the opening axis. Which yaw you pick decides
# whether a fingertip clips the neighbouring cube on the way down.
FINGER_HALF_SPAN = 0.065
FINGER_HALF_THICK = 0.011


def grasp_yaw_avoiding(cube_quat, cube_pos, obstacle_pos) -> float:
    """Face-aligned yaw that keeps the wide finger axis away from an obstacle.

    All four face-aligned yaws (base + k*90 deg) grasp the cube equally well, so
    among them pick the one whose finger-opening axis is most perpendicular to
    the direction of the neighbour. Discovered the hard way: descending with the
    opening axis pointed at a cube 7 cm away makes a finger shell hit it, which
    shoves the hand ~15 mm off target and the grasp closes on air.
    """
    base = grasp_yaw_for_cube(cube_quat)
    d = np.asarray(obstacle_pos)[:2] - np.asarray(cube_pos)[:2]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return base

    d = d / n
    # Fingers are symmetric, so yaw and yaw+180 are the same grasp: only two
    # distinct options exist, `base` and `base` turned 90 deg. Keep both at their
    # own angle (do NOT wrap them together - that collapses the choice and throws
    # the avoidance information away). base is already in (-45, 45], so its
    # alternative is at most 135 deg of wrist travel; fold that by 180 to stay
    # short without changing where the fingers end up.
    alt = base + np.pi / 2
    if alt > np.pi / 2:
        alt -= np.pi
    cands = [base, alt]

    def score(yaw):
        open_axis = np.array([-np.sin(yaw), np.cos(yaw)])  # eef Y in world
        return (abs(float(open_axis @ d)), abs(yaw))        # lower is better

    return min(cands, key=score)


def rot_error(current_quat, target_rotmat) -> np.ndarray:
    """Rotation vector taking the current eef orientation to the target."""
    Rc = R.from_quat(current_quat).as_matrix()
    return R.from_matrix(target_rotmat @ Rc.T).as_rotvec()


def action(pos_err, rot_err, grip: float, kp=KP_POS, kr=KR_ROT) -> np.ndarray:
    """Assemble a clipped 7-dim OSC action from world-frame errors."""
    a = np.zeros(7)
    a[:3] = np.clip(kp * np.asarray(pos_err), -1.0, 1.0)
    a[3:6] = np.clip(kr * np.asarray(rot_err), -1.0, 1.0)
    a[6] = grip
    return a
