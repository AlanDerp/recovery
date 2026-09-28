"""Runtime-only reproducibility helpers for CaP-X Robosuite evaluations.

This module never changes task configuration or model prompts.  It only binds
the RNG already exposed by ``reset(seed=...)`` to the Robosuite objects that
actually consume randomness and writes hashes for reproducibility audits.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def resolve_episode_seed(
    requested_seed: int | None,
    *,
    fixed_episode_seed: str | None = None,
    episode_seed_base: str | None = None,
) -> int | None:
    """Resolve an explicit physical reset seed without coupling it to API seeds."""
    fixed = fixed_episode_seed not in (None, "")
    based = episode_seed_base not in (None, "")
    if fixed and based:
        raise ValueError(
            "CAPX_FIXED_EPISODE_SEED and CAPX_EPISODE_SEED_BASE are mutually exclusive"
        )
    if fixed:
        return int(fixed_episode_seed)
    if based:
        if requested_seed is None:
            raise ValueError("CAPX_EPISODE_SEED_BASE requires a requested trial seed")
        return int(episode_seed_base) + int(requested_seed)
    return None if requested_seed is None else int(requested_seed)


def _walk_sampler_tree(root: Any) -> list[Any]:
    """Return sampler-like objects reachable from a placement initializer."""
    if root is None:
        return []
    pending = [root]
    seen: set[int] = set()
    result: list[Any] = []
    while pending:
        obj = pending.pop()
        identity = id(obj)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(obj)
        for attr in ("samplers", "_samplers"):
            children = getattr(obj, attr, None)
            if isinstance(children, dict):
                pending.extend(children.values())
            elif isinstance(children, (list, tuple)):
                pending.extend(children)
    return result


def bind_robosuite_seed(wrapper: Any, seed: int) -> dict[str, Any]:
    """Bind one seed to every RNG used during a Robosuite wrapper reset.

    Several CaP-X wrappers recreate ``wrapper._rng`` but their placement
    sampler retains a reference to ``wrapper.robosuite_env.rng`` created at
    construction time.  This function repairs that disconnected seed path.
    """
    import numpy as np

    seed = int(seed)
    seeded_rng = np.random.default_rng(seed)
    bound: list[str] = []

    if hasattr(wrapper, "_rng"):
        wrapper._rng = seeded_rng
        bound.append("wrapper._rng")

    robosuite_env = getattr(wrapper, "robosuite_env", None)
    if robosuite_env is None:
        return {"seed": seed, "bound": bound, "has_robosuite_env": False}

    robosuite_env.seed = seed
    robosuite_env.rng = seeded_rng
    bound.extend(("robosuite_env.seed", "robosuite_env.rng"))

    initializer = getattr(robosuite_env, "placement_initializer", None)
    for index, sampler in enumerate(_walk_sampler_tree(initializer)):
        if hasattr(sampler, "rng"):
            sampler.rng = seeded_rng
            bound.append(f"placement_sampler[{index}].rng")

    # Randomized model objects may retain the environment generator too.
    for attr_name, value in vars(robosuite_env).items():
        if attr_name in {"rng", "placement_initializer"}:
            continue
        if hasattr(value, "rng") and value.__class__.__module__.startswith("robosuite"):
            value.rng = seeded_rng
            bound.append(f"robosuite_env.{attr_name}.rng")

    return {"seed": seed, "bound": sorted(set(bound)), "has_robosuite_env": True}


def _hash_array(value: Any) -> str | None:
    try:
        import numpy as np

        array = np.asarray(value)
        if array.dtype.hasobject:
            return None
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()
    except Exception:
        return None


def initial_state_record(wrapper: Any, observation: Any, seed: int) -> dict[str, Any]:
    """Build a hash-only initial-state record; nothing is exposed to the model."""
    record: dict[str, Any] = {"seed": int(seed)}
    robosuite_env = getattr(wrapper, "robosuite_env", None)
    sim = getattr(robosuite_env, "sim", None)
    data = getattr(sim, "data", None)
    if data is not None:
        for name in ("qpos", "qvel", "xpos", "xquat"):
            digest = _hash_array(getattr(data, name, None))
            if digest:
                record[f"sim_{name}_sha256"] = digest

    observation_hashes: dict[str, str] = {}
    if isinstance(observation, dict):
        for key in sorted(observation):
            # Hash numeric observations, including RGB/depth, but not prompts.
            digest = _hash_array(observation[key])
            if digest:
                observation_hashes[str(key)] = digest
    record["observation_sha256"] = observation_hashes

    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["record_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return record


def append_audit_record(record: dict[str, Any]) -> None:
    audit_dir = os.environ.get("CAPX_REPRO_AUDIT_DIR")
    if not audit_dir:
        return
    path = Path(audit_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "initial_states.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
