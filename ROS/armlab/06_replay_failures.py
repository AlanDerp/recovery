"""Reproduce failures reliably by recording the exact layout, not the RNG path.

Lesson from 05: replaying "seed 7, trial 5" by re-running episodes in sequence
does NOT reproduce the layout, because run_episode consumes a variable number of
random draws (per-phase early exits differ). The fix is to capture the initial
object placement of each failing episode and force it back with sim state, so a
failure is a first-class, replayable artifact.
"""
import argparse
import json
import sys

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

sys.path.insert(0, "skills")
import arm
import stack_task

np.set_printoptions(precision=4, suppress=True)

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=60)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

cc = load_composite_controller_config(controller="BASIC", robot="Panda")
env = suite.make(
    "Stack", robots="Panda", controller_configs=cc,
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, control_freq=20, ignore_done=True, horizon=100000,
)

TIP1 = env.sim.model.body_name2id("gripper0_right_finger_joint1_tip")
TIP2 = env.sim.model.body_name2id("gripper0_right_finger_joint2_tip")
QA = env.sim.model.get_joint_qpos_addr("cubeA_joint0")
QB = env.sim.model.get_joint_qpos_addr("cubeB_joint0")


def tip_mid():
    return (env.sim.data.body_xpos[TIP1] + env.sim.data.body_xpos[TIP2]) / 2.0


def gripper_contacts():
    out = set()
    for i in range(env.sim.data.ncon):
        c = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(c.geom1) or ""
        g2 = env.sim.model.geom_id2name(c.geom2) or ""
        if "gripper0_right" in g1:
            out.add(g2)
        elif "gripper0_right" in g2:
            out.add(g1)
    return sorted(x for x in out if x)


def capture_layout():
    """The 7-dof free-joint qpos (pos + quat) of both cubes."""
    return dict(
        A=np.array(env.sim.data.qpos[QA[0]:QA[0] + 7]).tolist(),
        B=np.array(env.sim.data.qpos[QB[0]:QB[0] + 7]).tolist(),
    )


def restore_layout(layout):
    env.reset()
    env.sim.data.qpos[QA[0]:QA[0] + 7] = layout["A"]
    env.sim.data.qpos[QB[0]:QB[0] + 7] = layout["B"]
    env.sim.forward()
    # let them settle onto the table so contacts are consistent
    for _ in range(10):
        env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    return env._get_observations()


# ------------------------------------------------------------------ collect
np.random.seed(args.seed)
failures = []
for t in range(args.trials):
    env.reset()
    layout = capture_layout()  # captured BEFORE the policy runs
    obs = restore_layout(layout)

    plan, yaw = stack_task.build_plan(obs)
    Rt = arm.topdown_rotmat(yaw)
    for name, target, grip, budget, tol in plan:
        for k in range(budget):
            o = env._get_observations()
            pos_err = np.zeros(3) if target is None else target - o["robot0_eef_pos"]
            re = arm.rot_error(o["robot0_eef_quat"], Rt)
            env.step(arm.action(pos_err, re, grip))
            if tol is not None and np.linalg.norm(pos_err) < tol and np.linalg.norm(re) < 0.06:
                break
    for _ in range(40):
        o = env._get_observations()
        env.step(arm.action(np.zeros(3), arm.rot_error(o["robot0_eef_quat"], Rt), arm.OPEN))

    _, _, r_stack = env.staged_rewards()
    if r_stack <= 0:
        failures.append(layout)
        print(f"FAIL t{t:2d} captured (A={np.round(layout['A'][:3],4)} "
              f"B={np.round(layout['B'][:3],4)})")

print(f"\n{args.trials - len(failures)}/{args.trials} ok, "
      f"{len(failures)} failures captured")
with open("logs/failures.json", "w") as f:
    json.dump(failures, f, indent=1)

# ------------------------------------------------------------------ dissect
print("\n=== dissecting captured failures (now exactly reproducible) ===")
for idx, layout in enumerate(failures):
    obs = restore_layout(layout)
    A0 = obs["cubeA_pos"].copy()
    B0 = obs["cubeB_pos"].copy()
    plan, yaw = stack_task.build_plan(obs)
    Rt = arm.topdown_rotmat(yaw)
    print(f"\n--- failure {idx}: A0={A0} B0={B0} AB={np.linalg.norm(A0[:2]-B0[:2]):.4f} "
          f"grasp_yaw={np.degrees(yaw):.1f} cubeA_yaw={np.degrees(arm.yaw_of(obs['cubeA_quat'])):.1f}")
    for name, target, grip, budget, tol in plan:
        for k in range(budget):
            o = env._get_observations()
            pos_err = np.zeros(3) if target is None else target - o["robot0_eef_pos"]
            re = arm.rot_error(o["robot0_eef_quat"], Rt)
            env.step(arm.action(pos_err, re, grip))
            if tol is not None and np.linalg.norm(pos_err) < tol and np.linalg.norm(re) < 0.06:
                break
        o = env._get_observations()
        tip, Ac = tip_mid(), o["cubeA_pos"]
        print(f"    {name:8s} tip-A lat {np.linalg.norm(tip[:2]-Ac[:2])*1000:6.2f}mm "
              f"vert {(tip[2]-Ac[2])*1000:+7.2f}mm  A_moved {np.linalg.norm(Ac-A0)*1000:6.1f}mm  "
              f"grasp={env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)!s:5s} "
              f"touch={gripper_contacts()}")
    o = env._get_observations()
    print(f"    RESULT A={o['cubeA_pos']} staged={np.round(env.staged_rewards(),3)}")

env.close()
