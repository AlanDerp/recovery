"""Run N trials and dissect every failure. Headless, no window needed."""
import argparse
import sys

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

sys.path.insert(0, "skills")
import arm
import stack_task

np.set_printoptions(precision=4, suppress=True)

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cc = load_composite_controller_config(controller="BASIC", robot="Panda")
env = suite.make(
    "Stack", robots="Panda", controller_configs=cc,
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, control_freq=20, ignore_done=True, horizon=100000,
)
np.random.seed(args.seed)

results = []
for t in range(args.trials):
    r = stack_task.run_episode(env)
    results.append(r)
    A0, B0 = r["init"]["cubeA_pos"], r["init"]["cubeB_pos"]
    ab = np.linalg.norm(A0[:2] - B0[:2])
    print(f"{'OK  ' if r['success'] else 'FAIL'} t{t:2d} "
          f"AB={ab:.3f} yaw={np.degrees(r['yaw']):6.1f} "
          f"h={r['height']:.3f} off={r['offset']*1000:5.1f}mm "
          f"stack={r['staged'][2]:.0f}")

wins = sum(r["success"] for r in results)
print(f"\n=== {wins}/{len(results)} stacked ({100*wins/len(results):.0f}%) ===")

fails = [(t, r) for t, r in enumerate(results) if not r["success"]]
if not fails:
    sys.exit(0)

print("\n=== FAILURE DETAIL ===")
for t, r in fails:
    A0, B0 = r["init"]["cubeA_pos"], r["init"]["cubeB_pos"]
    print(f"\n-- trial {t}: AB dist={np.linalg.norm(A0[:2]-B0[:2]):.3f}  "
          f"A0={A0} B0={B0}  grasp_yaw={np.degrees(r['yaw']):.1f} deg")
    print(f"   final A={r['A_final']} height={r['height']:.4f} "
          f"offset={r['offset']*1000:.1f}mm staged={r['staged']}")
    for s in r["trace"]:
        flag = "  <-- HIT BUDGET" if s["used"] >= s["budget"] and s["budget"] > 15 else ""
        print(f"     {s['phase']:8s} {s['used']:3d}/{s['budget']:3d} "
              f"err={s['err']*1000:6.2f}mm grasp={str(s['grasped']):5s} "
              f"A={s['cubeA']}{flag}")

print("\n=== aggregate over successes ===")
oks = [r for r in results if r["success"]]
if oks:
    print(f"height  mean {np.mean([r['height'] for r in oks]):.4f} "
          f"min {np.min([r['height'] for r in oks]):.4f} (ideal 0.880)")
    print(f"offset  mean {np.mean([r['offset'] for r in oks])*1000:.1f} mm "
          f"max {np.max([r['offset'] for r in oks])*1000:.1f} mm")
env.close()
