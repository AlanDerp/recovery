"""No-sim contract tests for the v270 live-policy isolation boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

from capx.verification import libero_g17_live_isolation as isolation
from capx.verification import libero_g17_takeover as takeover


class _FakeEnv:
    def __init__(self) -> None:
        self._current_done = False
        self._sim_step_count = 12
        self.success = False

    def task_completed(self) -> bool:
        return self.success


def test_terminal_success_is_absorbing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(isolation.ENV, "1")
    monkeypatch.setenv("CAPX_REPRO_AUDIT_DIR", str(tmp_path))
    env = _FakeEnv()
    isolation.reset_terminal_latch(env)
    assert isolation.refresh_terminal_latch(env, source="before") is False

    env.success = True
    assert isolation.refresh_terminal_latch(env, source="release") is True
    env.success = False
    assert isolation.refresh_terminal_latch(env, source="later") is True
    assert isolation.action_blocked(SimpleNamespace(_env=env), "close_gripper")

    path = tmp_path / "libero_g17_v270_live_isolation" / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "terminal_latched",
        "action_suppressed",
    ]


def test_success_while_still_holding_does_not_latch(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(isolation.ENV, "1")
    monkeypatch.setenv(isolation.RELEASE_COMMIT_ENV, "1")
    monkeypatch.setenv("CAPX_REPRO_AUDIT_DIR", str(tmp_path))
    env = _FakeEnv()
    isolation.reset_terminal_latch(env)

    # LIBERO can satisfy the spatial predicate while the payload is still in
    # the closed gripper.  That must not suppress the subsequent release.
    env.success = True
    env._current_done = True
    assert isolation.refresh_terminal_latch(env, source="held-placement") is False
    assert not env._capx_v270_terminal_latched
    assert not isolation.action_blocked(SimpleNamespace(_env=env), "open_gripper")

    isolation.arm_success_latch_for_release(env)
    assert isolation.refresh_terminal_latch(env, source="open-step") is False
    isolation.disarm_success_latch_after_release(env)
    assert env._capx_v270_terminal_latched


def test_unsuccessful_release_window_is_disarmed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(isolation.ENV, "1")
    monkeypatch.setenv(isolation.RELEASE_COMMIT_ENV, "1")
    monkeypatch.setenv("CAPX_REPRO_AUDIT_DIR", str(tmp_path))
    env = _FakeEnv()
    isolation.reset_terminal_latch(env)

    isolation.arm_success_latch_for_release(env)
    isolation.disarm_success_latch_after_release(env)
    env.success = True
    env._current_done = True
    assert isolation.refresh_terminal_latch(env, source="later-held-motion") is False
    assert not env._capx_v270_terminal_latched


def test_success_is_deferred_until_full_release_window_finishes(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(isolation.ENV, "1")
    monkeypatch.setenv(isolation.RELEASE_COMMIT_ENV, "1")
    monkeypatch.setenv("CAPX_REPRO_AUDIT_DIR", str(tmp_path))
    env = _FakeEnv()
    isolation.reset_terminal_latch(env)
    isolation.arm_success_latch_for_release(env)

    # Every low-level open step must remain executable even if LIBERO's
    # spatial predicate becomes true on the first one.
    env.success = True
    env._current_done = True
    for step in range(40):
        env._sim_step_count = step
        assert isolation.refresh_terminal_latch(env, source="open-step") is False
        assert not env._capx_v270_terminal_latched

    isolation.disarm_success_latch_after_release(env)
    assert env._capx_v270_terminal_latched
    assert env._capx_v270_terminal_reason == "task_completed"


def test_done_without_success_is_also_absorbing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(isolation.ENV, "1")
    monkeypatch.setenv("CAPX_REPRO_AUDIT_DIR", str(tmp_path))
    env = _FakeEnv()
    isolation.reset_terminal_latch(env)
    env._current_done = True
    assert isolation.refresh_terminal_latch(env, source="horizon") is True
    assert env._capx_v270_terminal_reason == "environment_done"


def test_v270_schema_keeps_v269_mechanism(monkeypatch) -> None:
    flags = {
        "CAPX_LIBERO_G17_XY_LOCK": "1",
        "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL": "1",
        "CAPX_LIBERO_G17_RELEASE_READY": "1",
        "CAPX_LIBERO_G17_NEARMISS_OSC": "1",
        "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW": "1",
        "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
        "CAPX_LIBERO_G17_V269_MECH": "1",
        "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
        isolation.ENV: "1",
    }
    for key, value in flags.items():
        monkeypatch.setenv(key, value)
    schema = takeover.schema_version()
    assert "v2.70" in schema
    assert "v256-shared-mech-fg-source-live-isolation" in schema


def test_v272_fair_live_uses_guards_without_v270_flag(monkeypatch) -> None:
    monkeypatch.delenv(isolation.ENV, raising=False)
    monkeypatch.setenv(isolation.FAIR_LIVE_ENV, "1")
    for key in (
        "CAPX_LIBERO_G17_XY_LOCK",
        "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
        "CAPX_LIBERO_G17_RELEASE_READY",
        "CAPX_LIBERO_G17_NEARMISS_OSC",
        "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
        "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH",
        "CAPX_LIBERO_G17_V269_MECH",
    ):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CAPX_LIBERO_G17_V269_SYMBOL_SOURCE", "fg")
    assert isolation.enabled()
    schema = takeover.schema_version()
    assert "v2.72" in schema
    assert "fair-live-no-replay" in schema
