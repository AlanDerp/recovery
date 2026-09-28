#!/usr/bin/env python3
"""Freeze outcome-blind CB0 programs from one completed LIBERO live task."""

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


def _first_block(path: Path) -> str:
    parts = _BLOCK_SPLIT.split(path.read_text(encoding="utf-8"))
    blocks = [part.strip() for part in parts[1:] if part.strip()]
    if not blocks:
        raise ValueError(f"no code block in {path}")
    return blocks[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--global-seed", type=int, default=20260718)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    eval_root = args.task_root / "qwen3.7-plus" / "eval"
    repro = args.task_root / "repro_audit"
    trials: dict[str, dict[str, object]] = {}
    for trial in range(1, args.trials + 1):
        # sandboxrc_0 is the original episode execution.  Never select using
        # reward/taskcompleted; require a unique source to prevent leakage.
        candidates = sorted(
            eval_root.glob(f"trial_{trial:02d}_sandboxrc_0_*/code.py")
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"trial {trial}: expected exactly one sandboxrc_0 source, "
                f"found {len(candidates)}: {candidates}"
            )
        code_path = candidates[0]
        block = _first_block(code_path)
        init_path = repro / f"trial_{trial:02d}_init_state.json"
        init = json.loads(init_path.read_text(encoding="utf-8"))
        trials[str(trial)] = {
            "source_job": args.source_job,
            "source_code_py": str(code_path),
            "init_state_idx": int(init["init_state_idx"]),
            "init_state_sha256": str(init["init_state_sha256"]),
            "n_prefix": 1,
            "code_blocks": [block],
            "code_block_sha256": [sha256_text(block)],
        }

    payload = build_package_dict(
        suite_name=args.suite_name,
        task_id=args.task_id,
        global_seed=args.global_seed,
        trials=trials,
        notes=(
            "Outcome-blind sandboxrc_0 CB0 freeze from the formal v305 live "
            f"task; source {args.source_job}; all trials 1..{args.trials}."
        ),
    )
    if payload["schema"] != SCHEMA:
        raise AssertionError(payload["schema"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256_file(args.out)
    args.out.with_name("PACKAGE_SHA256.txt").write_text(
        digest + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(args.out), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
