from __future__ import annotations

from types import SimpleNamespace

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.execution_control import breakpoint_code_block


def _bare_exec_env() -> CodeExecutionEnvBase:
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = SimpleNamespace(get_observation=lambda: {})
    env._apis = {}
    env._full_prompt = []
    env._exec_globals = {
        "__name__": "__main__",
        "breakpoint_code_block": breakpoint_code_block,
    }
    return env


def test_breakpoint_stops_suffix_without_sandbox_error() -> None:
    env = _bare_exec_env()
    result = env._exec_user_code(
        "before = True\n"
        "breakpoint_code_block('firm_acquired')\n"
        "after = True\n"
    )

    assert result["ok"] is True
    assert result["stderr"] == ""
    assert result["breakpoint_reason"] == "firm_acquired"
    assert env._exec_globals["before"] is True
    assert "after" not in env._exec_globals
    assert "[code-block-breakpoint] firm_acquired" in result["stdout"]


def test_generated_exception_handler_cannot_consume_breakpoint() -> None:
    env = _bare_exec_env()
    result = env._exec_user_code(
        "try:\n"
        "    breakpoint_code_block('atomic_transition')\n"
        "except Exception:\n"
        "    consumed = True\n"
    )

    assert result["ok"] is True
    assert result["breakpoint_reason"] == "atomic_transition"
    assert "consumed" not in env._exec_globals
