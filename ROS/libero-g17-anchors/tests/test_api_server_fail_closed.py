from __future__ import annotations

import socket

import pytest

from capx.envs import runner


class _ExitedProcess:
    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass


def test_required_api_server_exit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    monkeypatch.setenv("CAPX_REQUIRE_API_SERVERS_READY", "1")
    monkeypatch.setenv("CAPX_API_SERVER_WAIT_TIMEOUT_SECONDS", "2")
    monkeypatch.setattr(runner, "run_server_proc", lambda _cfg: _ExitedProcess())

    with pytest.raises(RuntimeError, match="exited before ready"):
        runner._start_api_servers(
            [{"_target_": "unused", "host": "127.0.0.1", "port": port}]
        )
