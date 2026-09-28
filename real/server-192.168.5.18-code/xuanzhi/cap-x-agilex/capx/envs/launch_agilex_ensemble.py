"""Agilex dual-model ensemble CaP-X entry point (gpt-5.5 + Codex 5.3).

Uses runner_agilex_ensemble / trial_agilex_ensemble.
"""

from __future__ import annotations

import os
import time

import tyro

from capx.envs.launch import LaunchArgs, _run_web_ui
from capx.envs.configs.loader import DictLoader
from capx.utils.launch_utils import _load_config

os.environ.setdefault("MUJOCO_GL", "egl")


def _merge_ensemble_config(args: LaunchArgs, config: dict) -> None:
    """Preserve ensemble-only YAML fields filtered by the shared loader."""
    configs_dict = DictLoader.load([args.config_path])
    for key, value in configs_dict.items():
        if key.startswith("agilex_dual_model_") or key in {
            "use_parallel_ensemble",
            "use_multimodel",
            "agilex_reset_after_code_block",
        }:
            config[key] = value


def main(args: LaunchArgs) -> None:
    from capx.envs.runner_agilex_ensemble import (
        _run_headless_trials,
        _start_api_servers,
        _stop_api_servers,
    )

    start_time = time.time()
    env_factory, config, api_servers = _load_config(args)
    _merge_ensemble_config(args, config)
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
