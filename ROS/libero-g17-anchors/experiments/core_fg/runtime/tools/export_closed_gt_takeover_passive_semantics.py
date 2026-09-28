#!/usr/bin/env python3
"""Export an action-invisible closed-GT transition reference for M3 replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = "capx-closed-gt-takeover-v2.1"
TRACE_DIR = "closed_gt_takeover_passive_traces"


def export(args: argparse.Namespace) -> dict:
    roots = sorted(args.run_root.glob(f"**/{TRACE_DIR}/trial_*"))
    if len(roots) != args.total_trials:
        raise ValueError(
            f"Expected {args.total_trials} passive traces, found {len(roots)}"
        )
    turn_semantics: dict[str, dict[str, str]] = {}
    for root in roots:
        trial = str(int(root.name.split("_")[-1]))
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("mode") != "passive_action_invisible"
            or state.get("task") != args.task
            or state.get("trial") != int(trial)
            or state.get("student_prompt_changed") is not False
            or state.get("sim_step_observer_installed") is not True
        ):
            raise ValueError(f"Invalid passive closed-GT state: {root}")
        turn_paths = sorted(root.glob("turn_*/transition.json"))
        if state.get("turns_completed") != len(turn_paths):
            raise ValueError(f"Passive closed-GT turn count mismatch: {root}")
        turn_semantics[trial] = {}
        for expected_turn, path in enumerate(turn_paths, 1):
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("schema_version") != SCHEMA_VERSION
                or record.get("task") != args.task
                or record.get("trial") != int(trial)
                or record.get("turn") != expected_turn
                or record.get("student_prompt_changed") is not False
                or record.get("student_action_request_changed") is not False
                or len(record.get("public_interval_grasp_outcomes", {})) != 2
                or not isinstance(record.get("public_turn_semantics_sha256"), str)
            ):
                raise ValueError(f"Invalid passive closed-GT transition: {path}")
            turn_semantics[trial][str(expected_turn)] = record[
                "public_turn_semantics_sha256"
            ]
    return {
        "schema": "capx_closed_gt_takeover_passive_reference_v2.1",
        "task": args.task,
        "native_mode": "M3",
        "global_seed": args.global_seed,
        "total_trials": args.total_trials,
        "capture_mode": "passive_action_invisible",
        "student_prompt_changed": False,
        "student_action_request_changed": False,
        "source_run_root": str(args.run_root.resolve()),
        "closed_gt_turn_semantics_sha256": turn_semantics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("cube_stack", "cube_restack"), required=True)
    parser.add_argument("--total-trials", type=int, required=True)
    parser.add_argument("--global-seed", type=int, required=True)
    args = parser.parse_args()
    result = export(args)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
