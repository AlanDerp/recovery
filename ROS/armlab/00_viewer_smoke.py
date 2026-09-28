"""Smoke test: does an interactive MuJoCo window actually open on this Mac?

Run with:  mjpython 00_viewer_smoke.py
"""
import time

import mac_gl_patch

mac_gl_patch.apply()  # must precede robosuite import

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

cc = load_composite_controller_config(controller="BASIC", robot="Panda")

env = suite.make(
    "Lift",
    robots="Panda",
    controller_configs=cc,
    has_renderer=True,
    renderer="mjviewer",
    has_offscreen_renderer=True,   # extra: lets me grab frames to verify
    use_camera_obs=False,
    camera_names="frontview",
    camera_heights=480,
    camera_widths=640,
    control_freq=20,
    ignore_done=True,              # do not terminate on horizon
    horizon=100000,
)

env.reset()
print("[smoke] env built, action_dim =", env.action_dim, flush=True)

t0 = time.time()
N = 160  # 8 s at 20 Hz
for k in range(N):
    a = np.zeros(env.action_dim)
    a[0] = 0.3 * np.sin(k / 15.0)
    a[2] = 0.3 * np.cos(k / 15.0)
    a[-1] = np.sin(k / 25.0)
    env.step(a)
    env.render()

dt = time.time() - t0
print(f"[smoke] stepped {N} times in {dt:.1f}s -> {N/dt:.1f} fps", flush=True)

# Prove the pixels exist, independent of whether the window is visible.
import cv2

frame = env.sim.render(camera_name="frontview", width=640, height=480)
cv2.imwrite("media/smoke_frame.png", cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR))
print("[smoke] wrote media/smoke_frame.png", flush=True)

env.close()
