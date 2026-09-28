"""Goal: stack cubeA on cubeB and let go, so the tower stands on its own.

Run headless (fast, many trials):   python 03_stack.py --trials 20
Run with the live window:           mjpython 03_stack.py --view

Phase machine: hover -> align -> descend -> close -> lift -> carry -> place
-> release -> retreat -> settle. Each phase has its own tolerance and step
budget; a phase that runs out of budget advances anyway so a stuck attempt
still produces a diagnosable trace instead of hanging.
"""
import argparse
import platform
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--view", action="store_true", help="open the interactive window")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

if args.view and platform.system() == "Darwin":
    import mac_gl_patch
    mac_gl_patch.apply()

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

sys.path.insert(0, "skills")
import arm

np.set_printoptions(precision=4, suppress=True)

HOVER_H = 0.10       # how high above a cube to stage the approach
PLACE_GAP = 0.004    # release cubeA this far above cubeB's top face
CLEAR_H = 0.16       # carry height, well clear of cubeB


def run_episode(env, verbose=True):
    obs = env.reset()

    A = obs["cubeA_pos"].copy()
    B = obs["cubeB_pos"].copy()
    yaw = arm.grasp_yaw_for_cube(obs["cubeA_quat"])
    Rt = arm.topdown_rotmat(yaw)

    # Fingertips must reach the cube's mid-height, so command eef_pos a little
    # higher than the cube center.
    grasp_z = A[2] + arm.EEF_TO_FINGERTIP
    # Stacked cubeA center = cubeB top + cubeA half-extent, plus a release gap.
    place_z = B[2] + 0.025 + 0.020 + PLACE_GAP + arm.EEF_TO_FINGERTIP

    plan = [
        # name        target                                      grip        budget  tol
        ("hover",    lambda o: np.r_[A[:2], grasp_z + HOVER_H],    arm.OPEN,   60,  0.004),
        ("descend",  lambda o: np.r_[A[:2], grasp_z],              arm.OPEN,   60,  0.003),
        ("close",    None,                                         arm.CLOSE,  15,  None),
        ("lift",     lambda o: np.r_[A[:2], CLEAR_H + arm.TABLE_Z], arm.CLOSE, 40,  0.006),
        ("carry",    lambda o: np.r_[B[:2], CLEAR_H + arm.TABLE_Z], arm.CLOSE, 60,  0.004),
        ("place",    lambda o: np.r_[B[:2], place_z],               arm.CLOSE, 60,  0.003),
        ("release",  None,                                         arm.OPEN,   15,  None),
        ("retreat",  lambda o: np.r_[B[:2], place_z + 0.12],        arm.OPEN,   50,  0.010),
    ]

    log = []
    for name, target_fn, grip, budget, tol in plan:
        used = budget
        for k in range(budget):
            o = env._get_observations()
            if target_fn is None:
                # gripper-only phase: hold station, just squeeze or release
                pos_err = np.zeros(3)
            else:
                pos_err = target_fn(o) - o["robot0_eef_pos"]
            re = arm.rot_error(o["robot0_eef_quat"], Rt)
            env.step(arm.action(pos_err, re, grip))
            if env.viewer is not None:
                env.render()
            if tol is not None and np.linalg.norm(pos_err) < tol and np.linalg.norm(re) < 0.06:
                used = k + 1
                break
        o = env._get_observations()
        err = 0.0 if target_fn is None else np.linalg.norm(target_fn(o) - o["robot0_eef_pos"])
        log.append((name, used, budget, err))
        if verbose:
            print(f"  {name:8s} {used:3d}/{budget:3d} steps  residual {err*1000:6.2f} mm  "
                  f"A={o['cubeA_pos']}")

    # Let physics settle with the arm out of the way: does the tower survive?
    for _ in range(40):
        o = env._get_observations()
        re = arm.rot_error(o["robot0_eef_quat"], Rt)
        env.step(arm.action(np.zeros(3), re, arm.OPEN))
        if env.viewer is not None:
            env.render()

    o = env._get_observations()
    r_reach, r_lift, r_stack = env.staged_rewards()
    success = r_stack > 0
    A_final = o["cubeA_pos"]
    offset = np.linalg.norm(A_final[:2] - o["cubeB_pos"][:2])
    if verbose:
        print(f"  -> stacked={success}  A_final={A_final}  "
              f"height={A_final[2]:.4f} (ideal 0.880)  lateral offset={offset*1000:.1f} mm")
        print(f"     staged: reach={r_reach:.3f} lift={r_lift:.3f} stack={r_stack:.1f}")
    return success, A_final, offset, log


def main():
    cc = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = suite.make(
        "Stack", robots="Panda", controller_configs=cc,
        has_renderer=args.view, renderer="mjviewer",
        has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20,
        ignore_done=True, horizon=100000,
    )
    np.random.seed(args.seed)

    wins = 0
    for t in range(args.trials):
        if not args.quiet:
            print(f"\n--- trial {t} ---")
        ok, _, off, _ = run_episode(env, verbose=not args.quiet)
        wins += ok
    print(f"\n=== {wins}/{args.trials} stacked ({100*wins/args.trials:.0f}%) ===")
    env.close()


main()
