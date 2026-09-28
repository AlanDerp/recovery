"""Why does the grasp miss on close-together layouts?

The failure trace says cubeA never moves and grasp==False throughout. Candidate
causes, and what would distinguish them:

  (a) fingertips stop too high -> compare fingertip z against cube center z
  (b) lateral miss -> compare fingertip midpoint xy against cube center xy
  (c) fingers straddle the wrong thing (hit cubeB) -> log contacts during close
  (d) yaw misaligned so the 1 mm/side clearance is gone -> log yaw residual
  (e) cube pushed away during descent -> log cube displacement per phase

This replays only the failing seeds and dumps all five.
"""
import sys

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "skills")
import arm
import stack_task

np.set_printoptions(precision=4, suppress=True)

cc = load_composite_controller_config(controller="BASIC", robot="Panda")
env = suite.make(
    "Stack", robots="Panda", controller_configs=cc,
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, control_freq=20, ignore_done=True, horizon=100000,
)

TIP1 = env.sim.model.body_name2id("gripper0_right_finger_joint1_tip")
TIP2 = env.sim.model.body_name2id("gripper0_right_finger_joint2_tip")


def fingertip_mid():
    return (env.sim.data.body_xpos[TIP1] + env.sim.data.body_xpos[TIP2]) / 2.0


def contacts_with(geom_substr):
    """Names of geoms currently touching anything matching geom_substr."""
    out = set()
    for i in range(env.sim.data.ncon):
        c = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(c.geom1) or ""
        g2 = env.sim.model.geom_id2name(c.geom2) or ""
        if geom_substr in g1:
            out.add(g2)
        elif geom_substr in g2:
            out.add(g1)
    return out


FAIL_SEEDS = [5, 6, 7, 11, 19, 21, 28, 58]

# Reproduce the same layout sequence as the seed-7 run, capturing the failures.
np.random.seed(7)
for t in range(max(FAIL_SEEDS) + 1):
    if t not in FAIL_SEEDS:
        stack_task.run_episode(env)
        continue

    obs = env.reset()
    A0 = obs["cubeA_pos"].copy()
    B0 = obs["cubeB_pos"].copy()
    plan, yaw = stack_task.build_plan(obs)
    Rt = arm.topdown_rotmat(yaw)

    print(f"\n===== trial {t} =====")
    print(f"A0={A0}  B0={B0}  AB={np.linalg.norm(A0[:2]-B0[:2]):.4f}  "
          f"grasp_yaw={np.degrees(yaw):.1f} deg")
    print(f"cubeA yaw={np.degrees(arm.yaw_of(obs['cubeA_quat'])):.1f}  "
          f"cubeB yaw={np.degrees(arm.yaw_of(obs['cubeB_quat'])):.1f}")

    for name, target, grip, budget, tol in plan:
        if name not in ("hover", "descend", "close", "lift"):
            continue
        for k in range(budget):
            o = env._get_observations()
            pos_err = np.zeros(3) if target is None else target - o["robot0_eef_pos"]
            re = arm.rot_error(o["robot0_eef_quat"], Rt)
            env.step(arm.action(pos_err, re, grip))
            if tol is not None and np.linalg.norm(pos_err) < tol and np.linalg.norm(re) < 0.06:
                break

        o = env._get_observations()
        tip = fingertip_mid()
        Ac = o["cubeA_pos"]
        re = arm.rot_error(o["robot0_eef_quat"], Rt)
        print(f"  [{name}]")
        print(f"    eef      {o['robot0_eef_pos']}   target {target}")
        print(f"    tip_mid  {tip}")
        print(f"    cubeA    {Ac}  (moved {np.linalg.norm(Ac-A0)*1000:.1f} mm from start)")
        print(f"    tip-cubeA: lateral {np.linalg.norm(tip[:2]-Ac[:2])*1000:6.2f} mm   "
              f"vertical {(tip[2]-Ac[2])*1000:+7.2f} mm")
        print(f"    yaw residual {np.degrees(np.linalg.norm(re)):.2f} deg   "
              f"gripper qpos {o['robot0_gripper_qpos']}")
        touch = contacts_with("gripper0_right")
        print(f"    gripper touching: {sorted(touch) if touch else '(nothing)'}")
        if name == "close":
            print(f"    _check_grasp = "
                  f"{env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)}")

env.close()
