"""The stacking policy itself, importable so scripts can share one implementation."""
import numpy as np

import arm

HOVER_H = 0.10       # staging height above a cube
PLACE_GAP = 0.004    # release cubeA this far above cubeB's top face
CLEAR_H = 0.16       # carry height above the table, clear of cubeB


def build_plan(obs):
    """Phase list for the current layout. Targets are captured at reset time."""
    A = obs["cubeA_pos"].copy()
    B = obs["cubeB_pos"].copy()
    # Turn the wide finger axis away from cubeB, otherwise on tight layouts
    # (AB < ~9 cm) a finger shell clips cubeB during the descent and knocks the
    # hand off target. See logs/failures.json and the note in arm.py.
    yaw = arm.grasp_yaw_avoiding(obs["cubeA_quat"], A, B)

    grasp_z = A[2] + arm.EEF_TO_FINGERTIP
    place_z = B[2] + 0.025 + 0.020 + PLACE_GAP + arm.EEF_TO_FINGERTIP
    carry_z = CLEAR_H + arm.TABLE_Z

    plan = [
        ("hover",   np.r_[A[:2], grasp_z + HOVER_H], arm.OPEN,  60, 0.004),
        ("descend", np.r_[A[:2], grasp_z],           arm.OPEN,  60, 0.003),
        ("close",   None,                            arm.CLOSE, 15, None),
        ("lift",    np.r_[A[:2], carry_z],           arm.CLOSE, 40, 0.006),
        ("carry",   np.r_[B[:2], carry_z],           arm.CLOSE, 60, 0.004),
        ("place",   np.r_[B[:2], place_z],           arm.CLOSE, 60, 0.003),
        ("release", None,                            arm.OPEN,  15, None),
        ("retreat", np.r_[B[:2], place_z + 0.12],    arm.OPEN,  50, 0.010),
    ]
    return plan, yaw


def run_episode(env, render=False, verbose=False, on_step=None):
    """One full attempt. Returns a result dict with a per-phase trace."""
    obs = env.reset()
    init = {k: np.array(v) for k, v in obs.items()
            if k in ("cubeA_pos", "cubeB_pos", "cubeA_quat", "cubeB_quat")}
    plan, yaw = build_plan(obs)
    Rt = arm.topdown_rotmat(yaw)

    trace = []
    for name, target, grip, budget, tol in plan:
        used = budget
        for k in range(budget):
            o = env._get_observations()
            pos_err = np.zeros(3) if target is None else target - o["robot0_eef_pos"]
            re = arm.rot_error(o["robot0_eef_quat"], Rt)
            env.step(arm.action(pos_err, re, grip))
            if render:
                env.render()
            if on_step is not None:
                on_step(name, env)
            if tol is not None and np.linalg.norm(pos_err) < tol and np.linalg.norm(re) < 0.06:
                used = k + 1
                break
        o = env._get_observations()
        err = 0.0 if target is None else float(np.linalg.norm(target - o["robot0_eef_pos"]))
        grasped = bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA))
        trace.append(dict(phase=name, used=used, budget=budget, err=err,
                          cubeA=o["cubeA_pos"].copy(), grasped=grasped))
        if verbose:
            print(f"  {name:8s} {used:3d}/{budget:3d}  residual {err*1000:6.2f} mm  "
                  f"grasp={grasped}  A={o['cubeA_pos']}")

    # Settle with the arm clear: does the tower stand on its own?
    for _ in range(40):
        o = env._get_observations()
        re = arm.rot_error(o["robot0_eef_quat"], Rt)
        env.step(arm.action(np.zeros(3), re, arm.OPEN))
        if render:
            env.render()
        if on_step is not None:
            on_step("settle", env)

    o = env._get_observations()
    r_reach, r_lift, r_stack = env.staged_rewards()
    A_final = o["cubeA_pos"].copy()
    return dict(
        success=bool(r_stack > 0),
        A_final=A_final,
        height=float(A_final[2]),
        offset=float(np.linalg.norm(A_final[:2] - o["cubeB_pos"][:2])),
        init=init,
        trace=trace,
        staged=(float(r_reach), float(r_lift), float(r_stack)),
        yaw=float(yaw),
    )
