#!/usr/bin/env python3
"""Build task-local CB0 packages for LIBERO-Spatial held-out evaluation.

The source is the registered native M3 job.  We freeze only code block zero
for trials 1 and 2.  A cell may have multiple result directories (for example
after an infrastructure retry); all copies must agree on CB0 or construction
fails closed.  Success is never used as a selection criterion.
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


def _first_block(path: Path) -> str:
    parts = _BLOCK_SPLIT.split(path.read_text(encoding="utf-8"))
    blocks = [part.strip() for part in parts[1:] if part.strip()]
    if not blocks:
        raise ValueError(f"no code blocks in {path}")
    return blocks[0]


def _source_cb0(eval_root: Path, trial: int) -> tuple[str, list[str]]:
    paths = sorted(
        eval_root.glob(f"trial_{trial:02d}_sandboxrc_0_*/code.py")
    )
    if not paths:
        raise FileNotFoundError(
            f"missing sandboxrc_0 trial {trial} under {eval_root}"
        )
    by_digest: dict[str, tuple[str, list[str]]] = {}
    for path in paths:
        block = _first_block(path)
        digest = sha256_text(block)
        if digest not in by_digest:
            by_digest[digest] = (block, [])
        by_digest[digest][1].append(str(path))
    if len(by_digest) != 1:
        detail = {digest: rows[1] for digest, rows in by_digest.items()}
        raise RuntimeError(
            f"ambiguous CB0 for trial {trial}; refusing outcome-based choice: "
            f"{json.dumps(detail, indent=2)}"
        )
    return next(iter(by_digest.values()))


def _init_meta(task_root: Path, trial: int) -> dict[str, object]:
    path = task_root / "repro_audit" / f"trial_{trial:02d}_init_state.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "init_state_idx": int(raw["init_state_idx"]),
        "init_state_sha256": str(raw["init_state_sha256"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(
            "/project/peilab/qshou/cap-x-agilex/outputs/"
            "libero_m3_3x10x10_20260813/runs/libero_spatial/job-517994"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/project/peilab/qshou/cap-x-agilex/outputs/"
            "libero_g17_spatial_prefix_replay_20260818"
        ),
    )
    parser.add_argument("--task-ids", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--global-seed", type=int, default=20260718)
    args = parser.parse_args()

    task_ids = [int(token) for token in args.task_ids.split(",") if token.strip()]
    if not task_ids or any(task_id < 1 or task_id > 9 for task_id in task_ids):
        raise ValueError("held-out task ids must be a non-empty subset of 1..9")

    index: dict[str, object] = {
        "schema": "capx_libero_g17_spatial_prefix_index_v1",
        "source_job": "job-517994",
        "selection": "sandboxrc_0, CB0 consensus, outcome-blind",
        "tasks": {},
    }
    for task_id in task_ids:
        task_root = args.baseline_root / f"libero_spatial_{task_id}"
        eval_root = task_root / "qwen3.7-plus" / "eval"
        trials: dict[str, dict[str, object]] = {}
        for trial in (1, 2):
            block, source_paths = _source_cb0(eval_root, trial)
            trials[str(trial)] = {
                **_init_meta(task_root, trial),
                "n_prefix": 1,
                "code_blocks": [block],
                "code_block_sha256": [sha256_text(block)],
                "source_job": "job-517994",
                "source_code_py_candidates": source_paths,
            }
        payload = build_package_dict(
            suite_name="libero_spatial",
            task_id=task_id,
            global_seed=args.global_seed,
            trials=trials,
            notes=(
                "Held-out breadth CB0 freeze; trials 1-2 from native M3 "
                "job-517994 sandboxrc_0; consensus and outcome-blind."
            ),
        )
        if payload["schema"] != SCHEMA:
            raise AssertionError(payload["schema"])
        out_dir = args.out_root / f"task_{task_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "code_prefix_replay_v1.json"
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        digest = sha256_file(out)
        (out_dir / "PACKAGE_SHA256.txt").write_text(
            digest + "\n", encoding="utf-8"
        )
        index["tasks"][str(task_id)] = {
            "package": str(out),
            "sha256": digest,
            "trials": {
                trial: {
                    "init_state_idx": entry["init_state_idx"],
                    "init_state_sha256": entry["init_state_sha256"],
                    "code_block_sha256": entry["code_block_sha256"][0],
                }
                for trial, entry in trials.items()
            },
        }
        print(f"task={task_id} package={out} sha256={digest}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    index_path = args.out_root / "INDEX.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"index={index_path} sha256={sha256_file(index_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
