#!/usr/bin/env python3
"""LIBERO G17 FG projection smoke — isolated from the M3 baseline.

Reads SEED_TABLE / agilex FrankaLiberoEnv for live states only.
Writes ONLY under outputs/libero_g17_projection_smoke_* .
Does not set CAPX_IDEAL_TK_*, does not patch trial.py, does not rewrite
sbatch_libero_m3_3x10x10.sh / franka_libero_m3.yaml / SEED_TABLE.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
# Hard isolation: never activate cube Ideal-TK / takeover on this smoke.
for _flag in (
    "CAPX_IDEAL_TK_V4",
    "CAPX_IDEAL_TK",
    "CAPX_CLOSED_GT_TAKEOVER",
    "CAPX_IDEAL_TK_V4_SEMANTIC_GAIN_GATE",
    "CAPX_IDEAL_TK_V4_RELEASE_BOUND_PLACEMENT",
):
    os.environ[_flag] = "0"

PROJECT = Path("/home/qshou/capx_g17_observation_integrity")
CAPX_AGILEX = Path("/project/peilab/qshou/cap-x-agilex")
OUT_DIR = Path(
    os.environ.get(
        "LIBERO_G17_PROJECTION_SMOKE_OUT",
        str(CAPX_AGILEX / "outputs" / "libero_g17_projection_smoke_20260814"),
    )
)
BASELINE_ROOT = CAPX_AGILEX / "outputs" / "libero_m3_3x10x10_20260813"
BASELINE_FINGERPRINTS = {
    "sbatch_libero_m3_3x10x10.sh": (
        CAPX_AGILEX / "scripts" / "sbatch_libero_m3_3x10x10.sh",
        "1728358749e5e2fe916011e4578e81b15c1ffda55ec07c116c36f0a2b1794ef1",
    ),
    "franka_libero_m3.yaml": (
        CAPX_AGILEX / "env_configs" / "libero" / "franka_libero_m3.yaml",
        "befb7ff390ad619ac4c6a0201c470ae884376206de666dd7cfba2bb3787a4939",
    ),
    "SEED_TABLE.json": (
        BASELINE_ROOT / "SEED_TABLE.json",
        "4fec7669be41d3c6fbff096f1ca587052a7253c7af6a9e0a619eb0854f65c24d",
    ),
}


def _setup_paths() -> None:
    sys.path.insert(0, str(PROJECT))
    os.environ["LIBERO_CONFIG_PATH"] = str(CAPX_AGILEX / ".libero")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_baseline_untouched() -> dict:
    rows = {}
    for name, (path, expected) in BASELINE_FINGERPRINTS.items():
        got = _sha256_file(path)
        rows[name] = {
            "path": str(path),
            "expected": expected,
            "got": got,
            "ok": got == expected,
        }
    if not all(r["ok"] for r in rows.values()):
        raise RuntimeError(f"baseline fingerprint drift: {rows}")
    return rows


def _load_env(suite: str, task_id: int):
    from capx.envs.simulators.libero import FrankaLiberoEnv

    agilex = str(CAPX_AGILEX)
    if agilex not in sys.path:
        sys.path.insert(0, agilex)
    return FrankaLiberoEnv(
        suite_name=suite,
        task_id=task_id,
        privileged=True,
        max_steps=400,
        seed=1,
        enable_render=False,
    )


def _force_on_success(env, binding) -> None:
    from capx.verification import libero_g17_adapter as adapter

    domain = adapter._libero_domain(env)
    payload_id = adapter._body_id(domain, binding.payload_name)
    receptacle_id = adapter._body_id(domain, binding.receptacle_body_name)
    sim = domain.sim
    plate_pos = sim.data.body_xpos[receptacle_id].copy()
    plate_quat = sim.data.body_xquat[receptacle_id].copy()
    target = plate_pos.copy()
    target[2] += 0.05
    joint_name = None
    for j in range(sim.model.njnt):
        if int(sim.model.jnt_bodyid[j]) == int(payload_id):
            joint_name = sim.model.joint_id2name(j)
            break
    if joint_name is None:
        raise RuntimeError(f"no joint for payload {binding.payload_name}")
    addr = sim.model.get_joint_qpos_addr(joint_name)
    start, end = addr if isinstance(addr, tuple) else (int(addr), int(addr) + 7)
    if end - start < 7:
        raise RuntimeError(f"payload joint {joint_name} is not a free joint")
    qpos = sim.data.qpos.copy()
    qpos[start : start + 3] = target
    qpos[start + 3 : start + 7] = plate_quat
    sim.data.qpos[:] = qpos
    sim.forward()
    for _ in range(30):
        env._step_once()


def _force_in_success(env, binding) -> None:
    from capx.verification import libero_g17_adapter as adapter

    domain = adapter._libero_domain(env)
    payload_id = adapter._body_id(domain, binding.payload_name)
    sim = domain.sim
    try:
        site_pos = sim.data.get_site_xpos(binding.receptacle_eval_name).copy()
    except Exception:
        body_id = adapter._body_id(domain, binding.receptacle_body_name)
        site_pos = sim.data.body_xpos[body_id].copy()
        site_pos[2] += 0.05
    joint_name = None
    for j in range(sim.model.njnt):
        if int(sim.model.jnt_bodyid[j]) == int(payload_id):
            joint_name = sim.model.joint_id2name(j)
            break
    if joint_name is None:
        raise RuntimeError(f"no joint for payload {binding.payload_name}")
    addr = sim.model.get_joint_qpos_addr(joint_name)
    start, end = addr if isinstance(addr, tuple) else (int(addr), int(addr) + 7)
    if end - start < 7:
        raise RuntimeError(f"payload joint {joint_name} is not a free joint")
    qpos = sim.data.qpos.copy()
    qpos[start : start + 3] = site_pos
    qpos[start + 3 : start + 7] = [1, 0, 0, 0]
    sim.data.qpos[:] = qpos
    sim.forward()
    for _ in range(30):
        env._step_once()


def _snapshot_at_init(env, binding, trial: int):
    from capx.verification import closed_gt
    from capx.verification import libero_g17_adapter as adapter

    init_states = env.handle.init_states
    idx = int(trial) - 1
    env.handle.env.reset()
    env.handle.env.set_init_state(init_states[idx])
    for _ in range(10):
        env._step_once()
    raw = adapter.extract_libero_raw_scene(env, binding)
    return closed_gt.compile_snapshot(binding.scene, raw)


def _project_cell(suite: str, task_id: int) -> dict:
    from capx.verification import libero_g17_adapter as adapter
    from capx.verification import libero_g17_fg_projection as projection

    env = _load_env(suite, task_id)
    try:
        binding = adapter.bind_libero_g17_env(env, suite_name=suite, task_id=task_id)
        before = _snapshot_at_init(env, binding, trial=1)
        # Forced placement endpoint = after (release-time proxy without LLM).
        if binding.relation == "on":
            _force_on_success(env, binding)
        else:
            _force_in_success(env, binding)
        after_binding, after = adapter.extract_libero_snapshot(env, binding)
        assert after_binding.name == binding.name
        proj = projection.project_libero_fg(binding, before, after)
        audit = adapter.g17_versus_bddl_audit(after, env, binding)
        return {
            "ok": True,
            "label": "projection_pregoal_to_forced_success",
            "suite": suite,
            "task_id": task_id,
            "relation": binding.relation,
            "payload_public": binding.scene.objects[0].public_name,
            "receptacle_public": binding.scene.objects[1].public_name,
            "bddl_atom_true": audit["bddl_atom_true"],
            "private_task_completed": audit["private_task_completed"],
            "geometric_proxy": audit["geometric_proxy"],
            "gain_status": proj.gain.status,
            "novel_fact_count": len(proj.gain.novel_fact_ids),
            "novel_fact_ids": list(proj.gain.novel_fact_ids),
            "changed_field_count": proj.changed_field_count,
            "parent_sha256": proj.parent_sha256,
            "combined_sha256": proj.combined_sha256,
            "parent_byte_identity_ok": projection.parent_byte_identity_preserved(
                proj
            ),
            "parent_report": proj.parent_report,
            "release_report": proj.release_report,
            "combined_report": proj.combined_report,
            "ideal_tk_flags_forced_off": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "label": "projection_pregoal_to_forced_success",
            "suite": suite,
            "task_id": task_id,
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
    baseline = _assert_baseline_untouched()
    rows: list[dict] = [
        _project_cell("libero_spatial", 0),
        _project_cell("libero_object", 1),
        _reject_cell("libero_goal", 7),
    ]
    baseline_after = _assert_baseline_untouched()

    summary = {
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_fail": sum(1 for r in rows if not r.get("ok")),
        "baseline_untouched_before": baseline,
        "baseline_untouched_after": baseline_after,
        "gain_cells": [
            {
                "suite": r.get("suite"),
                "task_id": r.get("task_id"),
                "relation": r.get("relation"),
                "gain_status": r.get("gain_status"),
                "novel_fact_count": r.get("novel_fact_count"),
                "bddl_atom_true": r.get("bddl_atom_true"),
                "parent_byte_identity_ok": r.get("parent_byte_identity_ok"),
            }
            for r in rows
            if "gain_status" in r
        ],
        "rows": rows,
    }
    out_path = OUT_DIR / "PROJECTION_SMOKE.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    print(f"wrote {out_path}")

    if summary["n_fail"]:
        return 1
    if not any(r.get("rejected") for r in rows):
        print("missing negative-control rejection", file=sys.stderr)
        return 1
    gain_rows = [r for r in rows if r.get("gain_status")]
    if len(gain_rows) < 2:
        print("expected two projection cells", file=sys.stderr)
        return 1
    if not all(r.get("bddl_atom_true") for r in gain_rows):
        print("forced-success endpoint not BDDL-true", file=sys.stderr)
        return 1
    if not all(r.get("parent_byte_identity_ok") for r in gain_rows):
        print("parent/combined identity contract broken", file=sys.stderr)
        return 1
    # Projection smoke certifies the gate machinery. Novel facts are expected
    # when release candidates stay unchanged while other state moves; either
    # novel_release_facts or parent_complete is a valid certified outcome.
    if not all(
        r.get("gain_status") in {"novel_release_facts", "parent_complete"}
        for r in gain_rows
    ):
        print("unexpected gain status", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
