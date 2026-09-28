#!/usr/bin/env python3
"""Freeze the v354 paired-treatment Task-4 failures for clean-C replay."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from capx.verification.libero_g17_code_prefix_replay import (
    build_package_dict,
    sha256_file,
    sha256_text,
)


_BLOCK = re.compile(r"(?m)^# Code block \d+\s*$")


def _blocks(path: Path) -> list[str]:
    rows = [part.strip() for part in _BLOCK.split(path.read_text())[1:]]
    rows = [row for row in rows if row]
    if not rows:
        raise ValueError(f"no code blocks in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-task-root",
        type=Path,
        default=Path(
            "/home/qshou/capx_g17_results/"
            "libero_g17_v354_task4_fg_gain_paired_k10_20260828/runs/"
            "libero_spatial/job-535684/libero_spatial_4"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "experiments/libero_g17_v355/assets/"
            "v354_job535684_task4_failures_full_prefix.json"
        ),
    )
    parser.add_argument(
        "--trials", type=int, nargs="+", default=[3, 4, 5, 6, 9]
    )
    args = parser.parse_args()

    eval_root = args.source_task_root / "qwen3.7-plus" / "eval"
    trials = {}
    for trial in args.trials:
        matches = sorted(eval_root.glob(f"trial_{trial:02d}_*/code.py"))
        if not matches:
            raise ValueError(
                f"expected trial-{trial:02d} code.py, got {matches}"
            )
        # The evaluator snapshots the accumulated program under more than one
        # sandbox return-code directory.  The longest file is the complete
        # terminal history; shorter siblings are intermediate snapshots.
        code_path = max(matches, key=lambda path: path.stat().st_size)
        init_path = (
            args.source_task_root
            / "repro_audit"
            / f"trial_{trial:02d}_init_state.json"
        )
        init = json.loads(init_path.read_text())
        blocks = _blocks(code_path)
        trials[str(trial)] = {
            "source_job": "535684",
            "source_code_py": str(code_path),
            "init_state_idx": int(init["init_state_idx"]),
            "init_state_sha256": str(init["init_state_sha256"]),
            "n_prefix": len(blocks),
            "code_blocks": blocks,
            "code_block_sha256": [sha256_text(block) for block in blocks],
        }

    payload = build_package_dict(
        suite_name="libero_spatial",
        task_id=4,
        global_seed=20260718,
        trials=trials,
        notes=(
            "Mechanism-only replay of v354 treatment job 535684 failures. "
            "Trials 3/5/6/9 reached EmptyTargetNear; trial 4 audits identity "
            "and transaction liveness. Never score as fair live performance."
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(args.out), "sha256": sha256_file(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
