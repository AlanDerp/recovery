#!/usr/bin/env python3
"""Export LIBERO G17 code-prefix replay package (π0 freeze).

Default sources (plan):
  - trial 1 CB0 from rescue success job-519050
  - trial 2 CB0 from M3 baseline success job-517994

Writes package JSON and prints sha256.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from capx.verification.libero_g17_code_prefix_replay import (
    SCHEMA,
    build_package_dict,
    sha256_file,
    sha256_text,
)

_BLOCK_SPLIT = re.compile(r"(?m)^# Code block \d+\s*$")


def _split_code_py(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    parts = _BLOCK_SPLIT.split(text)
    blocks = [part.strip() for part in parts[1:] if part.strip()]
    if blocks:
        return blocks
    # Fallback: all_responses.json initial entry
    responses = path.with_name("all_responses.json")
    if responses.is_file():
        data = json.loads(responses.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            cbs = data[0].get("code_blocks") or []
            return [str(x).strip() for x in cbs if str(x).strip()]
    raise ValueError(f"no code blocks found in {path}")


def _init_meta(repro_audit: Path, trial: int) -> dict:
    path = repro_audit / f"trial_{trial:02d}_init_state.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "init_state_idx": int(data["init_state_idx"]),
        "init_state_sha256": str(data["init_state_sha256"]),
    }


def _find_trial_code(eval_root: Path, trial: int, *, prefer_success: bool) -> Path:
    matches = sorted(eval_root.glob(f"trial_{trial:02d}_*"))
    dirs = [p for p in matches if p.is_dir() and (p / "code.py").is_file()]
    if not dirs:
        raise FileNotFoundError(f"no trial_{trial:02d}_* with code.py under {eval_root}")
    if prefer_success:
        success = [p for p in dirs if "taskcompleted_1" in p.name]
        if success:
            return success[-1] / "code.py"
    return dirs[-1] / "code.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/project/peilab/qshou/cap-x-agilex/outputs/"
            "libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json"
        ),
    )
    parser.add_argument("--n-prefix", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=20260718)
    parser.add_argument("--suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()

    rescue_eval = Path(
        "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_rescue_smoke_20260814/"
        "runs/libero_spatial/job-519050/libero_spatial_0/qwen3.7-plus/eval"
    )
    rescue_repro = Path(
        "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_rescue_smoke_20260814/"
        "runs/libero_spatial/job-519050/libero_spatial_0/repro_audit"
    )
    baseline_eval = Path(
        "/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/"
        "runs/libero_spatial/job-517994/libero_spatial_0/qwen3.7-plus/eval"
    )
    baseline_repro = Path(
        "/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/"
        "runs/libero_spatial/job-517994/libero_spatial_0/repro_audit"
    )

    sources = {
        1: (
            "job-519050",
            _find_trial_code(rescue_eval, 1, prefer_success=True),
            rescue_repro,
        ),
        2: (
            "job-517994",
            _find_trial_code(baseline_eval, 2, prefer_success=True),
            baseline_repro,
        ),
    }

    trials: dict[str, dict] = {}
    for trial, (source_job, code_path, repro) in sources.items():
        blocks = _split_code_py(code_path)
        meta = _init_meta(repro, trial)
        n_prefix = min(int(args.n_prefix), len(blocks))
        prefix_blocks = blocks[:n_prefix]
        trials[str(trial)] = {
            "source_job": source_job,
            "source_code_py": str(code_path),
            "init_state_idx": meta["init_state_idx"],
            "init_state_sha256": meta["init_state_sha256"],
            "n_prefix": n_prefix,
            "code_blocks": prefix_blocks,
            "code_block_sha256": [sha256_text(b) for b in prefix_blocks],
        }

    payload = build_package_dict(
        suite_name=args.suite_name,
        task_id=args.task_id,
        global_seed=args.global_seed,
        trials=trials,
        notes=(
            "π0 freeze only (n_prefix=1): trial1 from 519050 success, "
            "trial2 from 517994 success. First FG fork is live."
        ),
    )
    assert payload["schema"] == SCHEMA
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256_file(args.out)
    print(json.dumps({"wrote": str(args.out), "sha256": digest, "trials": list(trials)}, indent=2))
    # Sidecar fingerprint for sbatch gates
    (args.out.parent / "PACKAGE_SHA256.txt").write_text(digest + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
