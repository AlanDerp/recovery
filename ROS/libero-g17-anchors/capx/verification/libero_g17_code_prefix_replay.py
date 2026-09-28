"""LIBERO G17 code-prefix replay (π0 freeze until first FG fork).

Package schema: ``capx_libero_g17_code_prefix_replay_v1``.

Enabled by env ``CAPX_LIBERO_G17_CODE_PREFIX_REPLAY=/path/to/package.json``.
When active, registered code blocks replace LLM outputs for
``turn < n_prefix`` (turn 0 = initial code). After the prefix, the normal
multi-turn path runs (FG takeover or native VLM).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "capx_libero_g17_code_prefix_replay_v1"
ENV_PATH = "CAPX_LIBERO_G17_CODE_PREFIX_REPLAY"


@dataclass(frozen=True)
class TrialPrefix:
    trial: int
    init_state_idx: int
    init_state_sha256: str
    n_prefix: int
    code_blocks: tuple[str, ...]
    source_job: str | None = None


@dataclass(frozen=True)
class CodePrefixPackage:
    path: Path
    sha256: str
    suite_name: str
    task_id: int
    global_seed: int
    trials: dict[int, TrialPrefix]

    def trial_entry(self, trial: int) -> TrialPrefix:
        try:
            return self.trials[int(trial)]
        except KeyError as exc:
            raise ValueError(
                f"code-prefix replay package lacks trial {trial}: {self.path}"
            ) from exc


def package_configured() -> bool:
    return bool(os.environ.get(ENV_PATH, "").strip())


def package_path() -> Path:
    raw = os.environ.get(ENV_PATH, "").strip()
    if not raw:
        raise RuntimeError(f"{ENV_PATH} is not set")
    return Path(raw)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_package(path: Path | None = None) -> CodePrefixPackage:
    pkg_path = Path(path) if path is not None else package_path()
    if not pkg_path.is_file():
        raise FileNotFoundError(f"code-prefix replay package missing: {pkg_path}")
    raw = json.loads(pkg_path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(
            f"unsupported code-prefix schema {raw.get('schema')!r} in {pkg_path}"
        )
    trials_raw = raw.get("trials")
    if not isinstance(trials_raw, dict) or not trials_raw:
        raise ValueError(f"code-prefix package has no trials: {pkg_path}")
    trials: dict[int, TrialPrefix] = {}
    for key, entry in trials_raw.items():
        trial = int(key)
        blocks = entry.get("code_blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(f"trial {trial} missing code_blocks in {pkg_path}")
        cleaned = tuple(str(block) for block in blocks)
        if any(not block.strip() for block in cleaned):
            raise ValueError(f"trial {trial} has empty code block in {pkg_path}")
        n_prefix = int(entry.get("n_prefix", 1))
        if n_prefix < 1:
            raise ValueError(f"trial {trial} n_prefix must be >= 1")
        if n_prefix > len(cleaned):
            raise ValueError(
                f"trial {trial} n_prefix={n_prefix} exceeds code_blocks={len(cleaned)}"
            )
        trials[trial] = TrialPrefix(
            trial=trial,
            init_state_idx=int(entry["init_state_idx"]),
            init_state_sha256=str(entry["init_state_sha256"]),
            n_prefix=n_prefix,
            code_blocks=cleaned,
            source_job=(
                None
                if entry.get("source_job") is None
                else str(entry.get("source_job"))
            ),
        )
    return CodePrefixPackage(
        path=pkg_path,
        sha256=sha256_file(pkg_path),
        suite_name=str(raw.get("suite_name") or ""),
        task_id=int(raw.get("task_id", 0)),
        global_seed=int(raw.get("global_seed", 0)),
        trials=trials,
    )


def wrap_python_fence(code: str) -> str:
    body = code.strip()
    if body.startswith("```"):
        return body
    return f"```python\n{body}\n```"


def wrap_regenerate(code: str) -> str:
    body = code.strip()
    if body.lstrip().upper().startswith("REGENERATE"):
        return body
    return "REGENERATE\n" + wrap_python_fence(body)


def assert_init_matches(env: Any, trial: int, package: CodePrefixPackage) -> None:
    entry = package.trial_entry(trial)
    low = getattr(env, "low_level_env", None)
    record = getattr(env, "_last_reset_record", None)
    if not isinstance(record, dict) and low is not None:
        # CodeExecutionEnvBase keeps the simulator on low_level_env.
        record = getattr(low, "_last_reset_record", None)
    if not isinstance(record, dict):
        # Fall back through wrapped libero env handles.
        handle = getattr(env, "handle", None)
        record = getattr(handle, "_last_reset_record", None) if handle else None
    if not isinstance(record, dict):
        base = getattr(env, "env", None) or getattr(env, "_env", None)
        record = getattr(base, "_last_reset_record", None) if base else None
    if not isinstance(record, dict) and low is not None:
        handle = getattr(low, "handle", None)
        record = getattr(handle, "_last_reset_record", None) if handle else None
    if not isinstance(record, dict):
        # Disk fallback written by FrankaLiberoEnv.reset when CAPX_REPRO_AUDIT_DIR is set.
        import json
        import os
        from pathlib import Path

        audit_dir = os.environ.get("CAPX_REPRO_AUDIT_DIR", "").strip()
        if audit_dir:
            path = Path(audit_dir) / f"trial_{int(trial):02d}_init_state.json"
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise RuntimeError(
            "code-prefix replay requires env reset record with init_state_sha256"
        )
    got_sha = record.get("init_state_sha256")
    got_idx = record.get("init_state_idx")
    if got_sha != entry.init_state_sha256:
        raise RuntimeError(
            f"code-prefix init sha mismatch trial={trial}: "
            f"got {got_sha} want {entry.init_state_sha256}"
        )
    if got_idx is not None and int(got_idx) != int(entry.init_state_idx):
        raise RuntimeError(
            f"code-prefix init idx mismatch trial={trial}: "
            f"got {got_idx} want {entry.init_state_idx}"
        )


def prefix_initial_raw_code(package: CodePrefixPackage, trial: int) -> str | None:
    """Return fenced initial model content if turn 0 is inside the prefix."""
    entry = package.trial_entry(trial)
    if entry.n_prefix < 1:
        return None
    return wrap_python_fence(entry.code_blocks[0])


def prefix_regenerate_raw_content(
    package: CodePrefixPackage, trial: int, *, executed_block_count: int
) -> str | None:
    """Return regenerate content for the next block if still inside prefix.

    ``executed_block_count`` is the number of blocks already executed
    (``code_block_idx`` after finishing a block, before appending the new one).
    The next forced block index is ``executed_block_count`` (0-based in package).
    Initial block is index 0; first regenerate forces index 1 when n_prefix > 1.
    """
    entry = package.trial_entry(trial)
    next_idx = int(executed_block_count)
    if next_idx < 1:
        return None
    if next_idx >= entry.n_prefix:
        return None
    return wrap_regenerate(entry.code_blocks[next_idx])


def write_prefix_hit_audit(
    output_dir: str | Path,
    *,
    trial: int,
    turn: int,
    code: str,
    package: CodePrefixPackage,
) -> None:
    root = Path(output_dir) / "libero_g17_code_prefix_replay"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"trial_{int(trial):02d}_turn_{int(turn):02d}.json"
    payload = {
        "schema": SCHEMA,
        "source": "code_prefix_replay",
        "trial": int(trial),
        "turn": int(turn),
        "package_path": str(package.path),
        "package_sha256": package.sha256,
        "code_sha256": sha256_text(code.strip()),
        "code": code,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_package_dict(
    *,
    suite_name: str,
    task_id: int,
    global_seed: int,
    trials: dict[str, dict[str, Any]],
    notes: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "suite_name": suite_name,
        "task_id": int(task_id),
        "global_seed": int(global_seed),
        "trials": trials,
    }
    if notes:
        payload["notes"] = notes
    return payload
