"""Probe: which way do the fingers open, and what P-gain converges?

Two questions the frame probe left open:
  1. Fingers separate along which axis of the eef frame? (needed to align yaw
     with a square cube's faces)
  2. What proportional gain on position error actually converges without
     oscillating, given the OSC lag measured earlier (~24% of commanded)?
"""
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=4, suppress=True)

cc = load_composite_controller_config(controller="BASIC", robot="Panda")
env = suite.make(
    "Stack", robots="Panda", controller_configs=cc,
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, control_freq=20,
    ignore_done=True, horizon=100000,
)
obs = env.reset()

print("=== finger geometry ===")
for name in env.sim.model.body_names:
    if "finger" in name or "hand" in name:
        i = env.sim.model.body_name2id(name)
        print(f"  {name:28s} xpos={env.sim.data.body_xpos[i]}")

eef_rot = R.from_quat(obs["robot0_eef_quat"]).as_matrix()
i0 = env.sim.model.body_name2id("gripper0_right_leftfinger")
i1 = env.sim.model.body_name2id("gripper0_right_rightfinger")
sep_world = env.sim.data.body_xpos[i1] - env.sim.data.body_xpos[i0]
print("\nfinger separation vector (world):", sep_world, "norm", np.linalg.norm(sep_world))
print("same vector in eef frame        :", eef_rot.T @ sep_world)
print("-> fingers open along eef axis  :", np.argmax(np.abs(eef_rot.T @ sep_world)),
      "(0=x, 1=y, 2=z)")

# ---------------------------------------------------------------- gain sweep
def topdown(yaw):
    """Rotation matrix: gripper pointing straight down, rotated by yaw about z."""
    return R.from_euler("z", yaw).as_matrix() @ R.from_euler("x", np.pi).as_matrix()


def servo(env, target_pos, target_yaw, kp, kr, steps, grip=-1.0, tol=0.002):
    """P-servo toward a world position + top-down orientation. Returns trace."""
    trace = []
    Rt = topdown(target_yaw)
    for k in range(steps):
        o = env._get_observations()
        p = o["robot0_eef_pos"]
        err = target_pos - p
        Rc = R.from_quat(o["robot0_eef_quat"]).as_matrix()
        rot_err = R.from_matrix(Rt @ Rc.T).as_rotvec()
        a = np.zeros(7)
        a[:3] = np.clip(kp * err, -1, 1)
        a[3:6] = np.clip(kr * rot_err, -1, 1)
        a[6] = grip
        env.step(a)
        trace.append(np.linalg.norm(err))
        if np.linalg.norm(err) < tol and np.linalg.norm(rot_err) < 0.05:
            break
    return np.array(trace)


print("\n=== position gain sweep (target: 12cm move, 150 step budget) ===")
for kp in [5, 10, 20, 40, 80]:
    env.reset()
    p0 = env._get_observations()["robot0_eef_pos"].copy()
    tgt = p0 + np.array([0.10, 0.06, -0.03])
    tr = servo(env, tgt, 0.0, kp, 2.0, 150)
    final = tr[-1]
    # detect oscillation: does the error increase anywhere after step 10?
    osc = np.any(np.diff(tr[10:]) > 0.002) if len(tr) > 12 else False
    print(f"  kp={kp:3d}: {len(tr):3d} steps, final err {final*1000:6.2f} mm, "
          f"overshoot/oscillation: {osc}")

print("\n=== rotation gain sweep (target: 60 deg yaw) ===")
for kr in [0.5, 1.0, 2.0, 4.0]:
    env.reset()
    p0 = env._get_observations()["robot0_eef_pos"].copy()
    n = 0
    Rt = topdown(np.radians(60))
    for k in range(150):
        o = env._get_observations()
        Rc = R.from_quat(o["robot0_eef_quat"]).as_matrix()
        rot_err = R.from_matrix(Rt @ Rc.T).as_rotvec()
        a = np.zeros(7)
        a[:3] = np.clip(20.0 * (p0 - o["robot0_eef_pos"]), -1, 1)
        a[3:6] = np.clip(kr * rot_err, -1, 1)
        a[6] = -1
        env.step(a)
        n = k
        if np.linalg.norm(rot_err) < 0.02:
            break
    o = env._get_observations()
    Rc = R.from_quat(o["robot0_eef_quat"]).as_matrix()
    res = np.degrees(np.linalg.norm(R.from_matrix(Rt @ Rc.T).as_rotvec()))
    drift = np.linalg.norm(o["robot0_eef_pos"] - p0)
    print(f"  kr={kr:4.1f}: {n:3d} steps, residual {res:5.2f} deg, pos drift {drift*1000:5.1f} mm")

env.close()
