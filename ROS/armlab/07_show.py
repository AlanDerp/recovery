"""Live demo: open the window, stack cubes repeatedly, and record video.

    mjpython 07_show.py --episodes 3

Records media/stack_demo.mp4 from the offscreen camera at the same time, which
only works because mac_gl_patch swaps GLFW for CGL (GLFW cannot create a
context off the main thread, and mjpython puts Python on a secondary thread).
"""
import argparse
import platform
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--camera", default="frontview")
args = ap.parse_args()

if platform.system() == "Darwin":
    import mac_gl_patch
    mac_gl_patch.apply()

import cv2
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

sys.path.insert(0, "skills")
import arm
import stack_task

W, H, FPS = 960, 720, 20
record = not args.no_video

cc = load_composite_controller_config(controller="BASIC", robot="Panda")
env = suite.make(
    "Stack", robots="Panda", controller_configs=cc,
    has_renderer=True, renderer="mjviewer",
    has_offscreen_renderer=record,
    use_camera_obs=False,
    camera_names=args.camera, camera_heights=H, camera_widths=W,
    control_freq=FPS, ignore_done=True, horizon=100000,
)

writer = None
if record:
    writer = cv2.VideoWriter("media/stack_demo.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))


def grab(_phase, e):
    if writer is not None:
        frame = e.sim.render(camera_name=args.camera, width=W, height=H)
        writer.write(cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR))


np.random.seed(args.seed)
wins = 0
t0 = time.time()
for ep in range(args.episodes):
    print(f"\n--- episode {ep} ---", flush=True)
    r = stack_task.run_episode(env, render=True, verbose=True,
                              on_step=grab if record else None)
    wins += r["success"]
    print(f"  => {'STACKED' if r['success'] else 'MISSED'}  "
          f"height {r['height']:.4f}  lateral offset {r['offset']*1000:.1f} mm",
          flush=True)

print(f"\n=== {wins}/{args.episodes} stacked in {time.time()-t0:.1f}s ===")
if writer is not None:
    writer.release()
    print("wrote media/stack_demo.mp4")
env.close()
