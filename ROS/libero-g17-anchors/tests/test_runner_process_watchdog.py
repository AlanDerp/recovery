from __future__ import annotations

import queue
import signal

from capx.envs import runner
from capx.envs import trial as trial_module


class _FakeQueue:
    def __init__(self, messages):
        self.messages = list(messages)
        self.closed = False

    def get(self, timeout):
        if not self.messages:
            raise queue.Empty
        return self.messages.pop(0)

    def get_nowait(self):
        if not self.messages:
            raise queue.Empty
        return self.messages.pop(0)

    def close(self):
        self.closed = True

    def join_thread(self):
        assert self.closed


class _FakeProcess:
    exitcode = 1

    def start(self):
        return None

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


class _FakeContext:
    def __init__(self, queues):
        self.queues = queues

    def Queue(self):
        return self.queues.pop(0)

    def Process(self, **kwargs):
        return _FakeProcess()


def test_process_watchdog_retries_only_prebootstrap_failure(monkeypatch) -> None:
    expected = object()
    queues = [
        _FakeQueue([]),
        _FakeQueue([("bootstrap_started", None), ("success", expected)]),
    ]
    context = _FakeContext(queues)
    monkeypatch.setattr(runner.multiprocessing, "get_context", lambda method: context)
    monkeypatch.setenv("CAPX_PROCESS_BOOTSTRAP_RETRIES", "1")

    actual = runner._run_trial_in_process_with_watchdog(
        trial=1,
        args=None,
        env_factory={},
        config={},
        timeout_s=10,
    )

    assert actual is expected
    assert not queues


def test_reset_alarm_uses_configured_trial_timeout(monkeypatch) -> None:
    calls: list[int] = []

    def fake_alarm(seconds: int) -> int:
        calls.append(seconds)
        return 731 if seconds == 0 else 0

    monkeypatch.setattr(signal, "alarm", fake_alarm)
    monkeypatch.setenv("CAPX_TRIAL_TIMEOUT_SECONDS", "1800")

    assert trial_module._restart_trial_alarm_after_reset() == 1800
    assert calls == [0, 1800]


def test_reset_alarm_preserves_remaining_time_without_override(monkeypatch) -> None:
    calls: list[int] = []

    def fake_alarm(seconds: int) -> int:
        calls.append(seconds)
        return 731 if seconds == 0 else 0

    monkeypatch.setattr(signal, "alarm", fake_alarm)
    monkeypatch.delenv("CAPX_TRIAL_TIMEOUT_SECONDS", raising=False)

    assert trial_module._restart_trial_alarm_after_reset() == 731
    assert calls == [0, 731]
