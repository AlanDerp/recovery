#!/usr/bin/env python3
"""LIBERO sim probe: tip XY-only primitive vs legacy goto_pose.

Boots ``FrankaLiberoEnv`` + pyroki only (no SAM3/GraspNet). Writes JSON under
``outputs/libero_tip_xy_only_probe_*``.

Acceptance (primitive, dx=0.03):
  |ach_x-0.03|<0.005, |ach_y|<0.005, |d_tip_z|<0.005
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

G17 = Path("/home/qshou/capx_g17_observation_integrity")
CAPX = Path("/project/peilab/qshou/cap-x-agilex")
OUT = Path(
    os.environ.get(
        "TIP_XY_PROBE_OUT",
        str(CAPX / "outputs" / f"libero_tip_xy_only_probe_{time.strftime('%Y%m%d')}"),
    )
)
STEP = float(os.environ.get("TIP_XY_PROBE_STEP_M", "0.03"))


def main() -> int:
    sys.path.insert(0, str(G17))
    os.environ["LIBERO_CONFIG_PATH"] = str(CAPX / ".libero")

    from capx.envs.simulators.libero import FrankaLiberoEnv
    from capx.integrations.franka import tip_xy_only as txy
    from capx.integrations.motion import pyroki_snippets as pks
    from capx.integrations.motion.pyroki_context import get_pyroki_context

    OUT.mkdir(parents=True, exist_ok=True)
    env = FrankaLiberoEnv(
        suite_name="libero_spatial",
        task_id=0,
        privileged=False,
        seed=0,
        enable_render=False,
    )
    env.reset(seed=0)
    ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
    api = SimpleNamespace(
        _env=env,
        _robot=ctx.robot,
        _target_link_name=ctx.target_link_name,
        _pks=pks,
        cfg=None,
        get_observation=env.get_observation,
    )

    tip0, _ = txy.read_obs_tip(api)
    print(f"[probe] start tip={tip0}", flush=True)

    # --- primitive ---
    prim = txy.move_tip_xy(api, STEP, 0.0, log=True)
    tip_p, _ = txy.read_obs_tip(api)

    # restore toward start with primitive
    txy.move_tip_xy(
        api,
        float(tip0[0] - tip_p[0]),
        float(tip0[1] - tip_p[1]),
        log=True,
    )
    tip_mid, quat_mid = txy.read_obs_tip(api)

    # --- legacy goto A/B (same commanded tip delta) ---
    legacy_target = tip_mid.copy()
    legacy_target[0] += STEP
    # Need goto_pose on a real API — use FrankaLiberoApi.goto_pose unbound via
    # a thin wrapper that only implements what goto needs.
    from capx.integrations.franka.libero import FrankaLiberoApi

    full = FrankaLiberoApi.__new__(FrankaLiberoApi)
    full._env = env
    full._robot = ctx.robot
    full._target_link_name = ctx.target_link_name
    full._pks = pks
    full.cfg = api.cfg
    full._TCP_OFFSET = FrankaLiberoApi._TCP_OFFSET
    # Disable tip-identity flag for true legacy comparison.
    os.environ["CAPX_LIBERO_G17_GOTO_TIP_IDENTITY"] = "0"
    tip_before_leg = tip_mid.copy()
    FrankaLiberoApi.goto_pose(full, legacy_target, quat_mid, z_approach=0.0)
    tip_leg, _ = txy.read_obs_tip(api)

    report = {
        "step_m": STEP,
        "start_tip": tip0.tolist(),
        "primitive": {
            "acceptance_ok": prim.acceptance_ok,
            "ok": prim.ok,
            "commanded_xy": prim.commanded_xy,
            "achieved_xy": prim.achieved_xy,
            "d_tip_z": prim.d_tip_z,
            "iters": prim.iters,
            "final_xy_err": prim.final_xy_err,
            "message": prim.message,
        },
        "legacy_goto": {
            "achieved_xy": (
                float(tip_leg[0] - tip_before_leg[0]),
                float(tip_leg[1] - tip_before_leg[1]),
            ),
            "d_tip_z": float(tip_leg[2] - tip_before_leg[2]),
        },
    }
    out_path = OUT / "probe_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[probe] wrote {out_path}", flush=True)

    if not prim.acceptance_ok:
        print("PROBE_FAIL primitive acceptance", flush=True)
        return 1
    print("PROBE_OK primitive acceptance", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
