#!/usr/bin/env python3
"""Live LIBERO G17-goal IFF smoke (no LLM, no FGSL/FGPP).

Cells:
  - libero_spatial task 0: On(bowl, plate) — reset + success init states
  - libero_object  task 1: In(cream cheese, basket)
  - libero_goal    task 7: TurnOn(stove) — must refuse bind

Exit code 0 only when every expected cell passes its contract.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

PROJECT = Path("/home/qshou/capx_g17_observation_integrity")
CAPX_AGILEX = Path("/project/peilab/qshou/cap-x-agilex")
OUT_DIR = Path(
    os.environ.get(
        "LIBERO_G17_SMOKE_OUT",
        str(CAPX_AGILEX / "outputs" / "libero_g17_iff_smoke_20260814"),
    )
)
SEED_TABLE = CAPX_AGILEX / "outputs" / "libero_m3_3x10x10_20260813" / "SEED_TABLE.json"

# Known M3 success trial indices (1-based) from 517993 run summaries/dirs.
SUCCESS_TRIALS = {
    ("libero_spatial", 0): [2, 4],
    ("libero_object", 1): [4, 9, 10],
}
FAIL_TRIALS = {
    ("libero_spatial", 0): [1],
    ("libero_object", 1): [1],
}


def _setup_paths() -> None:
    sys.path.insert(0, str(PROJECT))
    os.environ["LIBERO_CONFIG_PATH"] = str(CAPX_AGILEX / ".libero")


def _load_env(suite: str, task_id: int):
    from capx.envs.simulators.libero import FrankaLiberoEnv

    # Prefer the Cap-X agilex package for the simulator (has init_state fix),
    # while closed_gt / adapter live in the G17 worktree.
    agilex = str(CAPX_AGILEX)
    if agilex not in sys.path:
        sys.path.insert(0, agilex)
    env = FrankaLiberoEnv(
        suite_name=suite,
        task_id=task_id,
        privileged=True,
        max_steps=400,
        seed=1,
        enable_render=False,
    )
    return env


def _audit_cell(suite: str, task_id: int, trial: int, label: str) -> dict:
    from capx.verification import closed_gt
    from capx.verification import libero_g17_adapter as adapter

    env = _load_env(suite, task_id)
    try:
        binding = adapter.bind_libero_g17_env(env, suite_name=suite, task_id=task_id)
        # trial k -> init_states[k-1]
        init_states = env.handle.init_states
        idx = int(trial) - 1
        if idx < 0 or idx >= len(init_states):
            raise IndexError(f"init state idx {idx} out of range ({len(init_states)})")
        env.handle.env.reset()
        env.handle.env.set_init_state(init_states[idx])
        # settle like FrankaLiberoEnv.reset
        for _ in range(10):
            env._step_once()

        raw = adapter.extract_libero_raw_scene(env, binding)
        snapshot = closed_gt.compile_snapshot(binding.scene, raw)
        schema = closed_gt.schema_audit(binding.scene)
        audit = adapter.g17_versus_bddl_audit(snapshot, env, binding)
        row = {
            "ok": True,
            "label": label,
            "suite": suite,
            "task_id": task_id,
            "trial": trial,
            "binding": binding.name,
            "relation": binding.relation,
            "payload": binding.payload_name,
            "receptacle_eval": binding.receptacle_eval_name,
            "receptacle_body": binding.receptacle_body_name,
            "schema_field_count": schema["state_field_count"],
            "schema_unknown_free": schema["unknown_is_not_a_gt_value"],
            **audit,
            "g17_relative_z": snapshot.values[
                "object_object.obj01.obj02.relative_z"
            ],
            "g17_support_region": snapshot.values[
                "object_object.obj01.obj02.support_region"
            ],
            "g17_holding_payload": snapshot.values[
                "executor_object.exe01.obj01.holding"
            ],
        }
        return row
    except Exception as exc:
        return {
            "ok": False,
            "label": label,
            "suite": suite,
            "task_id": task_id,
            "trial": trial,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            env.handle.env.close()
        except Exception:
            pass


def _reject_cell(suite: str, task_id: int) -> dict:
    from capx.verification import libero_g17_adapter as adapter

    env = _load_env(suite, task_id)
    try:
        try:
            adapter.bind_libero_g17_env(env, suite_name=suite, task_id=task_id)
            return {
                "ok": False,
                "label": "negative_control_should_reject",
                "suite": suite,
                "task_id": task_id,
                "error": "bind unexpectedly succeeded",
            }
        except adapter.UnsupportedLiberoG17Goal as exc:
            return {
                "ok": True,
                "label": "negative_control_rejected",
                "suite": suite,
                "task_id": task_id,
                "rejected": True,
                "reason": str(exc),
            }
    finally:
        try:
            env.handle.env.close()
        except Exception:
            pass


def main() -> int:
    _setup_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Positive pick-place cells
    for (suite, task_id), trials in SUCCESS_TRIALS.items():
        for trial in trials[:2]:  # cap at 2 success inits for smoke budget
            rows.append(
                _audit_cell(suite, task_id, trial, label="success_init")
            )
        for trial in FAIL_TRIALS.get((suite, task_id), []):
            rows.append(_audit_cell(suite, task_id, trial, label="fail_or_reset_init"))

    # Also audit official init_states[0] after bind for each suite (reset-like)
    for suite, task_id in (("libero_spatial", 0), ("libero_object", 1)):
        rows.append(_audit_cell(suite, task_id, 1, label="init_state_0"))

    # Negative control
    rows.append(_reject_cell("libero_goal", 7))

    out_path = OUT_DIR / "IFF_SMOKE.json"
    summary = {
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_fail": sum(1 for r in rows if not r.get("ok")),
        "bddl_iff_task_ok": sum(
            1 for r in rows if r.get("bddl_iff_task_completed") is True
        ),
        "g17_iff_bddl_ok": sum(1 for r in rows if r.get("g17_iff_bddl") is True),
        "g17_iff_bddl_holes": [
            {
                "suite": r.get("suite"),
                "task_id": r.get("task_id"),
                "trial": r.get("trial"),
                "label": r.get("label"),
                "relation": r.get("relation"),
                "bddl": r.get("bddl_atom_true"),
                "proxy": r.get("geometric_proxy"),
                "task_completed": r.get("private_task_completed"),
            }
            for r in rows
            if r.get("ok")
            and "g17_iff_bddl" in r
            and r.get("g17_iff_bddl") is False
        ],
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    print(f"wrote {out_path}")

    # Contract:
    # 1) every row ok
    # 2) negative control rejected
    # 3) BDDL atom IFF task_completed on all positive cells
    # Geometric proxy IFF is reported but does NOT fail the smoke — that is the
    # hole signal that decides whether BDDL_IN/ON wrappers are needed later.
    if summary["n_fail"]:
        return 1
    if not any(r.get("rejected") for r in rows):
        print("missing negative-control rejection", file=sys.stderr)
        return 1
    positive = [r for r in rows if "bddl_iff_task_completed" in r]
    if not positive or not all(r.get("bddl_iff_task_completed") for r in positive):
        print("BDDL atom not IFF task_completed on all positive cells", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
