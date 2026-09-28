"""Post-task Agilex arm homing used by capx/envs/launch.py."""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig, OmegaConf

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader


def _extract_low_level_cfg(configs_dict: dict[str, Any]) -> dict[str, Any] | None:
    env_cfg = configs_dict.get("env")
    if not isinstance(env_cfg, dict):
        return None
    cfg = env_cfg.get("cfg")
    if not isinstance(cfg, dict):
        return None
    low_level = cfg.get("low_level")
    if not isinstance(low_level, dict):
        return None
    if isinstance(low_level, DictConfig):
        return OmegaConf.to_container(low_level, resolve=True)  # type: ignore[return-value]
    return dict(low_level)


def reset_agilex_from_config_path(config_path: str) -> bool:
    """Home the Agilex arm after a trial when configured.

    Returns True if homing ran, False if the config is not Agilex or homing
    is disabled via ``reset_on_task_end: false``.
    """
    config_path = os.path.expanduser(config_path)
    configs_dict = DictLoader.load([config_path])
    low_level_cfg = _extract_low_level_cfg(configs_dict)
    if not isinstance(low_level_cfg, dict):
        return False

    target = str(low_level_cfg.get("_target_", ""))
    if "agilex_real" not in target and "AgilexRealLowLevel" not in target:
        return False

    if not bool(low_level_cfg.get("reset_on_task_end", True)):
        return False

    homing_cfg = dict(low_level_cfg)
    homing_cfg.pop("reset_on_task_end", None)
    homing_cfg["wait_for_observation"] = False
    homing_cfg["start_server"] = False
    homing_cfg["camera_backend"] = "none"

    env = instantiate(homing_cfg)
    if env.hardware is None:
        print("[reset] Agilex hardware not connected; skipping post-task homing.")
        return False

    print("[reset] Interpolating Agilex arm to configured home pose ...")
    env._reset_hardware_to_home()
    print("[reset] Post-task arm homing complete.")
    return True


__all__ = ["reset_agilex_from_config_path"]
