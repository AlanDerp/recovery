#!/usr/bin/env python3
"""Class-level LIBERO G17 public-IFF audit (no LLM, no rescue smoke).

Hard contract (in-quotient On/In cells):
  bddl_atom ⇔ task_completed
  libero_public_atom ⇔ bddl_atom   (On: check_ontop replica; In: in_box replica)

Informational only:
  g17 geometric proxies (supported_by / volume) may disagree with BDDL —
  that is the known non-IFF hole; it must not gate this audit.

Coverage: libero_spatial 0–9 and libero_object 0–9 (bind or reject).
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
        str(CAPX_AGILEX / "outputs" / "libero_g17_public_iff_audit_20260814"),
    )
)


def _setup_paths() -> None:
    sys.path.insert(0, str(PROJECT))
    os.environ["LIBERO_CONFIG_PATH"] = str(CAPX_AGILEX / ".libero")


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


def _settle(env, init_idx: int = 0) -> None:
    env.handle.env.reset()
    env.handle.env.set_init_state(env.handle.init_states[init_idx])
    for _ in range(10):
        env._step_once()


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


def _row_from_env(env, binding, *, suite: str, task_id: int, label: str, trial) -> dict:
    from capx.verification import closed_gt
    from capx.verification import libero_g17_adapter as adapter
    from capx.verification import libero_g17_fg_projection as projection

    binding, snapshot = adapter.extract_libero_snapshot(env, binding)
    schema = closed_gt.schema_audit(binding.scene)
    audit = adapter.g17_versus_bddl_audit(snapshot, env, binding)
    finish = projection.public_finish_authorized(binding, snapshot)
    return {
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
        "containment_kind": (
            None
            if binding.containment_geom is None
            else binding.containment_geom.kind
        ),
        "schema_field_count": schema["state_field_count"],
        "finish_authorized": finish,
        "finish_matches_public_atom": finish == bool(audit.get("libero_public_atom")),
        **audit,
    }


def _audit_pregoal(suite: str, task_id: int, trial: int = 1) -> dict:
    from capx.verification import libero_g17_adapter as adapter

    env = _load_env(suite, task_id)
    try:
        try:
            binding = adapter.bind_libero_g17_env(
                env, suite_name=suite, task_id=task_id
            )
        except adapter.UnsupportedLiberoG17Goal as exc:
            return {
                "ok": True,
                "label": "out_of_quotient",
                "suite": suite,
                "task_id": task_id,
                "trial": trial,
                "rejected": True,
                "reason": str(exc),
            }
        _settle(env, trial - 1)
        return _row_from_env(
            env, binding, suite=suite, task_id=task_id, label="pregoal_init", trial=trial
        )
    except Exception as exc:
        return {
            "ok": False,
            "label": "pregoal_init",
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


def _audit_forced(suite: str, task_id: int) -> dict:
    from capx.verification import libero_g17_adapter as adapter

    env = _load_env(suite, task_id)
    try:
        try:
            binding = adapter.bind_libero_g17_env(
                env, suite_name=suite, task_id=task_id
            )
        except adapter.UnsupportedLiberoG17Goal as exc:
            return {
                "ok": True,
                "label": "out_of_quotient",
                "suite": suite,
                "task_id": task_id,
                "rejected": True,
                "reason": str(exc),
            }
        _settle(env, 0)
        if binding.relation == "on":
            _force_on_success(env, binding)
        else:
            _force_in_success(env, binding)
        return _row_from_env(
            env,
            binding,
            suite=suite,
            task_id=task_id,
            label="forced_success",
            trial=None,
        )
    except Exception as exc:
        return {
            "ok": False,
            "label": "forced_success",
            "suite": suite,
            "task_id": task_id,
            "trial": None,
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
        except Exception as exc:
            return {
                "ok": False,
                "label": "negative_control_error",
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


def main() -> int:
    _setup_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for suite in ("libero_spatial", "libero_object"):
        for task_id in range(10):
            rows.append(_audit_pregoal(suite, task_id, trial=1))
            rows.append(_audit_forced(suite, task_id))

    rows.append(_reject_cell("libero_goal", 7))

    in_quotient = [
        r
        for r in rows
        if r.get("ok") and r.get("label") in {"pregoal_init", "forced_success"}
    ]
    public_holes = [
        r
        for r in in_quotient
        if r.get("libero_public_iff_bddl") is not True
    ]
    proxy_holes = [
        r for r in in_quotient if r.get("g17_iff_bddl") is False
    ]
    summary = {
        "schema_note": (
            "No closed_gt predicate deleted. Finish uses libero_public_atom "
            "(On/In BDDL replicas). G17 supported_by/volume remain observational."
        ),
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_fail": sum(1 for r in rows if not r.get("ok")),
        "n_in_quotient": len(in_quotient),
        "n_out_of_quotient": sum(1 for r in rows if r.get("label") == "out_of_quotient"),
        "bddl_iff_task_ok": sum(
            1 for r in in_quotient if r.get("bddl_iff_task_completed") is True
        ),
        "public_iff_bddl_ok": sum(
            1 for r in in_quotient if r.get("libero_public_iff_bddl") is True
        ),
        "public_iff_bddl_holes": [
            {
                "suite": r.get("suite"),
                "task_id": r.get("task_id"),
                "label": r.get("label"),
                "relation": r.get("relation"),
                "bddl": r.get("bddl_atom_true"),
                "public_atom": r.get("libero_public_atom"),
                "finish_authorized": r.get("finish_authorized"),
                "containment_kind": r.get("containment_kind"),
            }
            for r in public_holes
        ],
        "g17_proxy_holes_informational": [
            {
                "suite": r.get("suite"),
                "task_id": r.get("task_id"),
                "label": r.get("label"),
                "relation": r.get("relation"),
                "bddl": r.get("bddl_atom_true"),
                "proxy": r.get("geometric_proxy"),
                "public_atom": r.get("libero_public_atom"),
            }
            for r in proxy_holes
        ],
        "forced_success": [
            {
                "suite": r.get("suite"),
                "task_id": r.get("task_id"),
                "relation": r.get("relation"),
                "bddl": r.get("bddl_atom_true"),
                "public_atom": r.get("libero_public_atom"),
                "public_iff": r.get("libero_public_iff_bddl"),
                "finish_authorized": r.get("finish_authorized"),
            }
            for r in rows
            if r.get("label") == "forced_success" and not r.get("rejected")
        ],
        "rows": rows,
    }
    out_path = OUT_DIR / "PUBLIC_IFF_AUDIT.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    print(f"wrote {out_path}")

    if summary["n_fail"]:
        print("runtime failures present", file=sys.stderr)
        return 1
    if not any(r.get("label") == "negative_control_rejected" for r in rows):
        print("missing negative-control rejection", file=sys.stderr)
        return 1
    if not in_quotient:
        print("no in-quotient cells", file=sys.stderr)
        return 1
    if summary["bddl_iff_task_ok"] != len(in_quotient):
        print("BDDL atom not IFF task_completed on all in-quotient cells", file=sys.stderr)
        return 1
    if summary["public_iff_bddl_ok"] != len(in_quotient):
        print(
            f"public IFF holes: {len(public_holes)} "
            f"(see public_iff_bddl_holes)",
            file=sys.stderr,
        )
        return 1
    forced = [r for r in in_quotient if r.get("label") == "forced_success"]
    if not forced or not all(r.get("bddl_atom_true") for r in forced):
        print("forced_success cells did not achieve BDDL true", file=sys.stderr)
        return 1
    if not all(r.get("finish_authorized") for r in forced):
        print("forced_success missing finish_authorized", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
