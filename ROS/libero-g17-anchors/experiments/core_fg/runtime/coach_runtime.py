"""No-op compatibility surface for the reproducibility sitecustomize module."""

from __future__ import annotations

from typing import Any


def reset_episode_state(trial: int, task_name: str | None = None) -> dict[str, Any]:
    return {"trial": int(trial), "task_name": str(task_name or "unknown_task")}


def fallback_guidance() -> None:
    return None
