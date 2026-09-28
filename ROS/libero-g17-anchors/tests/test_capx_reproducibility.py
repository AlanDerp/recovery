from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/core_fg/runtime/capx_reproducibility.py"
)
SPEC = importlib.util.spec_from_file_location("capx_reproducibility_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_episode_seed_defaults_to_requested_trial_seed() -> None:
    assert MODULE.resolve_episode_seed(7) == 7


def test_episode_seed_base_changes_only_physical_seed_mapping() -> None:
    assert MODULE.resolve_episode_seed(7, episode_seed_base="1000") == 1007


def test_fixed_episode_seed_is_supported_without_base() -> None:
    assert MODULE.resolve_episode_seed(7, fixed_episode_seed="19") == 19


def test_episode_seed_overrides_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        MODULE.resolve_episode_seed(
            7,
            fixed_episode_seed="19",
            episode_seed_base="1000",
        )


def test_episode_seed_base_requires_requested_trial_seed() -> None:
    with pytest.raises(ValueError, match="requires a requested trial seed"):
        MODULE.resolve_episode_seed(None, episode_seed_base="1000")
