"""Agilex multi-turn CaP-X entry point (copy of launch.py).

Uses runner_agilex_multiturn / trial_agilex_multiturn so execution failures
are fed back to the LLM for REGENERATE retries within one trial.
"""

from __future__ import annotations

import os
import time

import tyro

from capx.envs.launch import LaunchArgs, _run_web_ui
from capx.utils.launch_utils import _load_config

os.environ.setdefault("MUJOCO_GL", "egl")


def main(args: LaunchArgs) -> None:
    from capx.envs.runner_agilex_multiturn import (
        _run_headless_trials,
        _start_api_servers,
        _stop_api_servers,
    )

    start_time = time.time()
    env_factory, config, api_servers = _load_config(args)
    server_procs = _start_api_servers(api_servers)

    try:
        if config.get("web_ui", False):
            _run_web_ui(args, config)
        else:
            _run_headless_trials(args, env_factory, config, start_time)
    finally:
        try:
            from capx.envs.simulators.agilex_reset import reset_agilex_from_config_path

            if not reset_agilex_from_config_path(args.config_path):
                pass
        except Exception as exc:
            print(f"[reset] Post-task arm homing failed: {exc}")
        try:
            _stop_api_servers(server_procs)
        except KeyboardInterrupt:
            import sys

            sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(LaunchArgs))
