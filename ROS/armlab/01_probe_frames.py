"""Probe: what do the 7 action dims actually DO in world coordinates?

Answers, empirically rather than from docs:
  1. Is the OSC "base" reference frame the same as the world frame for Panda?
  2. Which action dim moves the eef along world x / y / z, and with what sign?
  3. What does the eef quaternion look like at reset, and which rotation dim
     controls yaw about the world z axis?
  4. How wide does the gripper open, and how many steps does a grasp take?
"""
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=4, suppress=True)


def make_env():
    cc = load_composite_controller_config(controller="BASIC", robot="Panda")
    return suite.make(
        "Stack", robots="Panda", controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20,
        ignore_done=True, horizon=100000,
    )


env = make_env()
obs = env.reset()

print("=== base frame vs world frame ===")
base_pos = env.robots[0].base_pos if hasattr(env.robots[0], "base_pos") else None
print("robot base_pos attr:", base_pos)
bid = env.sim.model.body_name2id("robot0_base")
print("robot0_base xpos:", env.sim.data.body_xpos[bid])
print("robot0_base xmat:\n", env.sim.data.body_xmat[bid].reshape(3, 3))
print("table_offset:", env.table_offset)

print("\n=== eef pose at reset ===")
print("eef_pos       :", obs["robot0_eef_pos"])
print("eef_quat      :", obs["robot0_eef_quat"])
rot = R.from_quat(obs["robot0_eef_quat"])  # robosuite uses xyzw
print("eef rotmat    :\n", rot.as_matrix())
print("eef euler xyz :", np.degrees(rot.as_euler("xyz")))
print("gripper z-axis in world (approach dir):", rot.as_matrix()[:, 2])
print("gripper qpos  :", obs["robot0_gripper_qpos"])

print("\n=== per-dim response test (30 steps of +0.5 on each dim) ===")
for dim, label in enumerate(["dx", "dy", "dz", "drx", "dry", "drz"]):
    env.reset()
    p0 = env._get_observations()["robot0_eef_pos"].copy()
    q0 = env._get_observations()["robot0_eef_quat"].copy()
    a = np.zeros(7)
    a[dim] = 0.5
    a[6] = -1
    for _ in range(30):
        env.step(a)
    o = env._get_observations()
    dp = o["robot0_eef_pos"] - p0
    dr = (R.from_quat(o["robot0_eef_quat"]) * R.from_quat(q0).inv()).as_rotvec()
    print(f"  {label:4s} -> world dpos {dp}  |dpos|={np.linalg.norm(dp):.4f}   "
          f"world drotvec {dr} (deg {np.degrees(np.linalg.norm(dr)):.1f})")

print("\n=== gripper open/close ===")
env.reset()
for _ in range(20):
    env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
print("fully open qpos:", env._get_observations()["robot0_gripper_qpos"])
for _ in range(20):
    env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
print("fully closed qpos:", env._get_observations()["robot0_gripper_qpos"])

print("\n=== cube geometry & scene layout over 5 resets ===")
for i in range(5):
    o = env.reset()
    yaw_a = R.from_quat(o["cubeA_quat"]).as_euler("xyz")[2]
    yaw_b = R.from_quat(o["cubeB_quat"]).as_euler("xyz")[2]
    print(f"  reset {i}: A={o['cubeA_pos']} yaw={np.degrees(yaw_a):7.1f}  "
          f"B={o['cubeB_pos']} yaw={np.degrees(yaw_b):7.1f}  "
          f"A-B horiz dist={np.linalg.norm(o['cubeA_to_cubeB'][:2]):.3f}")

print("\ncubeA geom size:", env.sim.model.geom_size[env.sim.model.geom_name2id("cubeA_g0")])
print("cubeB geom size:", env.sim.model.geom_size[env.sim.model.geom_name2id("cubeB_g0")])

env.close()
