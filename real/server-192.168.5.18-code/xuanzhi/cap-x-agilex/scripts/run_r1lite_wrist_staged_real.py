#!/usr/bin/env python3
"""Run deterministic R1 Lite wrist tennis staged execution without LLM planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader
from capx.envs.launch import LaunchArgs
from capx.envs.tasks.base import CodeExecutionEnvBase
import capx.envs.trial_r1lite_staged as trial_r1lite_staged
from capx.envs.trial_r1lite_staged import R1_STAGES, run_r1_staged_trial
import capx.llm.client as llm_client


def _fixed_stage_plan(_args: LaunchArgs, _messages: list[dict]) -> dict[str, str]:
    return {
        "content": json.dumps(
            {"stages": [{"id": stage} for stage in R1_STAGES]},
            ensure_ascii=False,
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="env_configs/r1lite/r1lite_tennis_ball_in_box.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/r1lite_wrist_real_test",
    )
    args = parser.parse_args()

    config = DictLoader.load([str(REPO_ROOT / args.config_path)])
    config["output_dir"] = args.output_dir
    config["r1lite_staged_tennis_execution"] = True
    config["use_wrist_camera"] = True

    launch_args = LaunchArgs(
        config_path=str(REPO_ROOT / args.config_path),
        output_dir=args.output_dir,
        model="gpt-5.5",
        server_url="http://127.0.0.1:8110/chat/completions",
        api_key=None,
        max_tokens=4096,
        temperature=0.0,
        reasoning_effort=None,
        debug=True,
        use_visual_feedback=False,
        use_oracle_code=False,
    )

    env_cfg = config["env"]["cfg"]
    env = CodeExecutionEnvBase(instantiate(env_cfg))
    original_query_model = llm_client.query_model
    original_staged_query_model = trial_r1lite_staged.query_model
    llm_client.query_model = _fixed_stage_plan
    trial_r1lite_staged.query_model = _fixed_stage_plan
    try:
        summary = run_r1_staged_trial(env, trial=1, args=launch_args, config=config)
    finally:
        llm_client.query_model = original_query_model
        trial_r1lite_staged.query_model = original_staged_query_model
        env.close()

    print(
        json.dumps(
            {
                "task_completed": summary.task_completed,
                "reward": summary.reward,
                "sandbox_rc": summary.sandbox_rc,
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not summary.task_completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
