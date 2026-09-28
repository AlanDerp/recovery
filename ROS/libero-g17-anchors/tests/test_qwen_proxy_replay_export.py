from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
import httpx
from openai import APIConnectionError, APIStatusError
from PIL import Image

REPO_ROOT = Path(__file__).parents[1]
PROXY_PATH = REPO_ROOT / "experiments/core_fg/runtime/qwen_proxy.py"
EXPORT_PATH = (
    REPO_ROOT
    / "experiments/core_fg/runtime/tools/export_request_hash_replay.py"
)
REGISTER_PATH = (
    REPO_ROOT
    / "experiments/core_fg/runtime/tools/register_replay_visual_fingerprints.py"
)
RESTACK_PI0_PATH = (
    REPO_ROOT
    / "experiments/pgsvp_v9_verify/replays"
    / "cube_restack_qwen37_m2_pi0_seed20260718_10trial_v1.json"
)
STACK_PI0_PATH = (
    REPO_ROOT
    / "experiments/pgsvp_v9_verify/replays"
    / "cube_stack_qwen37_m2_pi0_seed20260718_10trial_v1.json"
)
RESTACK_A0_PATH = (
    REPO_ROOT
    / "experiments/pgsvp_v9_verify/replays"
    / "cube_restack_qwen37_m2_a0_full_seed20260718_10trial_v1.json"
)
STACK_A0_PATH = (
    REPO_ROOT
    / "experiments/pgsvp_v9_verify/replays"
    / "cube_stack_qwen37_m2_a0_full_seed20260718_10trial_v1.json"
)
RESTACK_A0_REFERENCE = (
    REPO_ROOT
    / "experiments/pgsvp_v9_verify/replays"
    / "cube_restack_a0_reference_seed20260718_10trial_v1.json"
)
STACK_A0_REFERENCE = (
    REPO_ROOT
    / "experiments/pgsvp_v9_verify/replays"
    / "cube_stack_a0_reference_seed20260718_10trial_v1.json"
)
RESTACK_M3_SMOKE_REPLAY = (
    REPO_ROOT
    / "experiments/fgsrp/replays"
    / "cube_restack_qwen37_m3_a1_full_seed6000_1trial_job500319_v1.json"
)
RESTACK_M3_SMOKE_REFERENCE = (
    REPO_ROOT
    / "experiments/fgsrp/replays"
    / "cube_restack_m3_a1_reference_seed6000_1trial_job500319_v1.json"
)
SPEC = importlib.util.spec_from_file_location("qwen_proxy", PROXY_PATH)
assert SPEC is not None and SPEC.loader is not None
PROXY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROXY)
EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_request_hash_replay_for_test", EXPORT_PATH
)
assert EXPORT_SPEC is not None and EXPORT_SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(EXPORT)
REGISTER_SPEC = importlib.util.spec_from_file_location(
    "register_replay_visual_fingerprints", REGISTER_PATH
)
assert REGISTER_SPEC is not None and REGISTER_SPEC.loader is not None
REGISTER = importlib.util.module_from_spec(REGISTER_SPEC)
REGISTER_SPEC.loader.exec_module(REGISTER)


def _transient_media_error() -> APIStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    body = {
        "error": {
            "code": "invalid_parameter_error",
            "message": "Download multimodal file timed out",
        }
    }
    response = httpx.Response(400, request=request, json=body)
    return APIStatusError(
        "Download multimodal file timed out", response=response, body=body
    )


def test_only_provider_multimodal_timeout_is_retryable() -> None:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(400, request=request)
    auth_error = APIStatusError(
        "Invalid API key", response=response, body={"error": "invalid key"}
    )

    assert PROXY.retryable_transient_media_error(_transient_media_error()) is True
    assert PROXY.retryable_transient_media_error(auth_error) is False


def test_replay_normalization_removes_internal_verifier_wrapper_frame() -> None:
    source = {
        "model": "qwen3.7-plus",
        "messages": [{
            "role": "user",
            "content": (
                "Traceback:\n"
                '  File "<CAPX_RUNTIME>/capx/envs/tasks/base.py", line 174, '
                "in _exec_user_code\n"
                '  File "<CAPX_RUNTIME>/capx/verification/ideal_tk_v4.py", '
                "line 1021, in wrapped\n"
                "    return __original(*args, **kwargs)\n"
                '  File "<CAPX_RUNTIME>/capx/integrations/franka/control.py", '
                "line 441, in sample_grasp_pose\n"
                "RuntimeError: no executable grasp\n"
            ),
        }],
    }
    current = json.loads(json.dumps(source))
    current["messages"][0]["content"] = current["messages"][0]["content"].replace(
        '  File "<CAPX_RUNTIME>/capx/verification/ideal_tk_v4.py", '
        "line 1021, in wrapped\n"
        "    return __original(*args, **kwargs)\n",
        "",
    )

    _, source_raw, _, source_normalized = PROXY.request_hashes(source)
    current_canonical, current_raw, _, current_normalized = PROXY.request_hashes(
        current
    )

    assert source_raw != current_raw
    assert source_normalized == current_normalized
    assert (
        EXPORT.normalized_request_hash_from_canonical(current_canonical)
        == current_normalized
    )
    assert PROXY.NORMALIZATION_VERSION == EXPORT.NORMALIZATION_VERSION


def test_transient_multimodal_timeout_retries_identical_upstream_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "qwen.key"
    key_file.write_text("test-key\n", encoding="utf-8")
    audit_log = tmp_path / "qwen_proxy_audit.jsonl"
    calls: list[dict] = []

    class Response:
        def model_dump(self) -> dict:
            return {
                "id": "response",
                "choices": [
                    {
                        "message": {"content": "FINISH"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _transient_media_error()
            return Response()

    class Client:
        def __init__(self, **_kwargs) -> None:
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(PROXY, "AsyncOpenAI", Client)
    monkeypatch.setattr(PROXY, "TRANSIENT_MEDIA_RETRY_DELAY_SECONDS", 0)
    app = PROXY.make_app(
        key_file=str(key_file),
        base_url="https://example.invalid/v1",
        audit_log=str(audit_log),
        upstream_concurrency=1,
        enable_thinking=False,
        request_seed=None,
        replay_file=None,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/chat/completions"
    )
    request_body = {
        "model": "qwen3.7-plus",
        "messages": [{"role": "user", "content": "state"}],
        "temperature": 0,
        "seed": 123,
    }

    class Request:
        async def json(self) -> dict:
            return request_body

    response = asyncio.run(endpoint(Request()))
    record = json.loads(audit_log.read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert record["ok"] is True
    assert record["upstream_attempt_count"] == 2
    assert len(record["transient_upstream_errors"]) == 1


def test_transport_failure_retries_identical_upstream_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "qwen.key"
    key_file.write_text("test-key\n", encoding="utf-8")
    audit_log = tmp_path / "qwen_proxy_audit.jsonl"
    calls: list[dict] = []

    class Response:
        def model_dump(self) -> dict:
            return {
                "id": "response",
                "choices": [
                    {
                        "message": {"content": "FINISH"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                request = httpx.Request(
                    "POST", "https://example.invalid/v1/chat/completions"
                )
                raise APIConnectionError(request=request)
            return Response()

    class Client:
        def __init__(self, **kwargs) -> None:
            assert kwargs["timeout"] == PROXY.UPSTREAM_TIMEOUT_SECONDS
            assert kwargs["max_retries"] == 0
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(PROXY, "AsyncOpenAI", Client)
    monkeypatch.setattr(PROXY, "TRANSIENT_MEDIA_RETRY_DELAY_SECONDS", 0)
    app = PROXY.make_app(
        key_file=str(key_file),
        base_url="https://example.invalid/v1",
        audit_log=str(audit_log),
        upstream_concurrency=1,
        enable_thinking=False,
        request_seed=None,
        replay_file=None,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/chat/completions"
    )
    request_body = {
        "model": "qwen3.7-plus",
        "messages": [{"role": "user", "content": "state"}],
        "temperature": 0,
        "seed": 123,
    }

    class Request:
        async def json(self) -> dict:
            return request_body

    response = asyncio.run(endpoint(Request()))
    record = json.loads(audit_log.read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert record["upstream_attempt_count"] == 2
    assert record["transient_upstream_errors"] == [
        {
            "attempt": 1,
            "status_code": None,
            "error_type": "APIConnectionError",
            "error": "Connection error.",
        }
    ]


def _image_data_url(value: int, *, compress_level: int = 6) -> str:
    image = Image.new("RGB", (2, 2), color=(value, value, value))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _write_frame(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(value, value, value)).save(path)
    return REGISTER.rgb_fingerprint(path)


def test_m3_visual_fingerprints_follow_report_action_alternation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    trial_root = (
        run_root
        / "qwen3.7-plus/cube_restack/trial_01_sandboxrc_0_reward_0_taskcompleted_0"
    )
    frames = [
        _write_frame(trial_root / f"visual_feedback_{index:02d}.png", value)
        for index, value in enumerate((0, 64, 128))
    ]
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "source_run_root": str(run_root),
                "entries": {
                    str(seed): {"effective_request_seed": seed}
                    for seed in range(7000, 7006)
                },
            }
        ),
        encoding="utf-8",
    )

    REGISTER.register(
        replay,
        "cube_restack",
        6000,
        native_mode="M3",
        total_trials=1,
    )
    package = json.loads(replay.read_text(encoding="utf-8"))
    observed = {
        int(entry["effective_request_seed"]): entry["request_image_rgb_sha256"]
        for entry in package["entries"].values()
    }
    assert observed == {
        7000: [frames[0]],
        7001: [],
        7002: [frames[0], frames[1]],
        7003: [],
        7004: [frames[1], frames[2]],
        7005: [],
    }


def test_m3_visual_fingerprints_support_fixed_episode_smoke_seed(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    trial_root = (
        run_root
        / "qwen3.7-plus/cube_restack/trial_01_sandboxrc_0_reward_0_taskcompleted_0"
    )
    frames = [
        _write_frame(trial_root / f"visual_feedback_{index:02d}.png", value)
        for index, value in enumerate((0, 64))
    ]
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "source_run_root": str(run_root),
                "entries": {
                    str(seed): {"effective_request_seed": seed}
                    for seed in range(4000, 4004)
                },
            }
        ),
        encoding="utf-8",
    )

    REGISTER.register(
        replay,
        "cube_restack",
        20260718,
        native_mode="M3",
        total_trials=1,
        fixed_episode_seed=4,
    )
    package = json.loads(replay.read_text(encoding="utf-8"))
    observed = {
        int(entry["effective_request_seed"]): (
            entry["trial"], entry["turn"], entry["request_image_rgb_sha256"]
        )
        for entry in package["entries"].values()
    }
    assert observed == {
        4000: (1, 0, [frames[0]]),
        4001: (1, 1, []),
        4002: (1, 2, [frames[0], frames[1]]),
        4003: (1, 3, []),
    }
    assert package["visual_fingerprint_fixed_episode_seed"] == 4


def _assert_frozen_pi0(path: Path, task: str) -> None:
    package = json.loads(path.read_text(encoding="utf-8"))
    expected_seeds = [20260718 + trial * 1000 for trial in range(1, 11)]

    assert package["schema"] == "capx_request_hash_replay_v1"
    assert package["canonical_scope"] == "initial_pi0_only"
    assert package["canonical_task"] == task
    assert package["missing_count"] == 0
    assert package["entries_count"] == 10
    assert sorted(
        entry["effective_request_seed"] for entry in package["entries"].values()
    ) == expected_seeds
    assert len(PROXY.load_request_hash_replay(str(path))) == 10


def test_frozen_restack_pi0_contains_only_initial_requests() -> None:
    _assert_frozen_pi0(RESTACK_PI0_PATH, "cube_restack")


def test_frozen_stack_pi0_contains_only_initial_requests() -> None:
    _assert_frozen_pi0(STACK_PI0_PATH, "cube_stack")


def test_registered_m3_smoke_reference_binds_complete_failed_a1() -> None:
    replay_bytes = RESTACK_M3_SMOKE_REPLAY.read_bytes()
    replay = json.loads(replay_bytes)
    reference = json.loads(RESTACK_M3_SMOKE_REFERENCE.read_text(encoding="utf-8"))

    assert reference["native_mode"] == "M3"
    assert reference["global_seed"] == 6000
    assert reference["total_trials"] == 1
    assert reference["expected_summary"] == "1.000/0.025/0"
    assert reference["full_replay_entries"] == replay["entries_count"] == 12
    assert reference["full_replay_sha256"] == hashlib.sha256(replay_bytes).hexdigest()
    assert replay["visual_fingerprint_native_mode"] == "M3"
    assert replay["visual_fingerprint_total_trials"] == 1
    assert sorted(reference["m3_semantic_transition_sha256"]["1"]) == [
        "1", "2", "3", "4", "5"
    ]
    assert reference["m3_semantic_source_repair_count"] == 0
    for entry in replay["entries"].values():
        turn = entry["turn"]
        expected_image_count = 1 if turn == 0 else (0 if turn % 2 else 2)
        assert len(entry["request_image_rgb_sha256"]) == expected_image_count


@pytest.mark.parametrize(
    ("path", "expected_entries"),
    ((RESTACK_A0_PATH, 41), (STACK_A0_PATH, 48)),
)
def test_frozen_a0_contains_complete_multiturn_requests(
    path: Path, expected_entries: int
) -> None:
    package = json.loads(path.read_text(encoding="utf-8"))

    assert package["schema"] == "capx_request_hash_replay_v1"
    assert package["missing_count"] == 0
    assert package["audit_records"] == expected_entries
    assert package["entries_count"] == expected_entries
    assert len(package["entries"]) == expected_entries
    assert package["visual_fingerprint_schema"] == "rgb_shape_bytes_sha256_v1"
    assert all(
        entry["request_image_rgb_sha256"]
        and entry["trial"] in range(1, 11)
        and entry["turn"] >= 0
        for entry in package["entries"].values()
    )
    assert len(PROXY.load_request_hash_replay(str(path))) == expected_entries


@pytest.mark.parametrize(
    ("task", "replay_path", "pi0_path", "reference_path"),
    (
        (
            "cube_restack",
            RESTACK_A0_PATH,
            RESTACK_PI0_PATH,
            RESTACK_A0_REFERENCE,
        ),
        ("cube_stack", STACK_A0_PATH, STACK_PI0_PATH, STACK_A0_REFERENCE),
    ),
)
def test_frozen_a0_reference_binds_replay(
    task: str, replay_path: Path, pi0_path: Path, reference_path: Path
) -> None:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    pi0 = json.loads(pi0_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    assert reference["schema"] == "capx_frozen_a0_reference_v1"
    assert reference["task"] == task
    assert reference["global_seed"] == 20260718
    assert reference["total_trials"] == 10
    assert reference["full_replay_entries"] == replay["entries_count"]
    assert reference["full_replay_sha256"] == hashlib.sha256(
        replay_path.read_bytes()
    ).hexdigest()
    assert len(reference["initial_state_record_sha256"]) == 10
    assert sorted(map(int, reference["trials"])) == list(range(1, 11))
    full_seed_responses = {
        (entry["effective_request_seed"], entry["response_sha256"])
        for entry in replay["entries"].values()
    }
    pi0_seed_responses = {
        (entry["effective_request_seed"], entry["response_sha256"])
        for entry in pi0["entries"].values()
    }
    assert len(pi0_seed_responses) == 10
    assert pi0_seed_responses <= full_seed_responses


def test_replayed_response_can_be_exported_again(tmp_path: Path) -> None:
    key_file = tmp_path / "qwen.key"
    key_file.write_text("test-key\n", encoding="utf-8")
    audit_log = tmp_path / "run" / "qwen_proxy_audit.jsonl"
    replay_file = tmp_path / "source_replay.json"
    exported_file = tmp_path / "exported_replay.json"
    request = {
        "model": "qwen3.7-plus",
        "messages": [{"role": "user", "content": "state"}],
        "temperature": 0,
        "seed": 123,
    }
    _, request_sha, _, normalized_sha = PROXY.request_hashes(request)
    content = "REGENERATE\n```python\npass\n```"
    response_sha = hashlib.sha256(content.encode()).hexdigest()
    replay_file.write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "entries": {
                    request_sha: {
                        "content": content,
                        "response_sha256": response_sha,
                        "request_normalized_sha256": normalized_sha,
                        "role": "student",
                        "trial": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    app = PROXY.make_app(
        key_file=str(key_file),
        base_url="https://example.invalid/v1",
        audit_log=str(audit_log),
        upstream_concurrency=1,
        enable_thinking=False,
        request_seed=None,
        replay_file=None,
        request_hash_replay_file=str(replay_file),
        audit_raw_content=True,
    )

    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/chat/completions"
    )

    class ReplayRequest:
        async def json(self) -> dict:
            return request

    response = asyncio.run(endpoint(ReplayRequest()))
    assert response.status_code == 200
    response_body = json.loads(response.body)
    assert response_body["choices"][0]["message"]["content"] == content
    audit = json.loads(audit_log.read_text(encoding="utf-8"))
    assert audit["source"] == "request_hash_replay"
    assert audit["response_content"] == content

    subprocess.run(
        [
            sys.executable,
            str(EXPORT_PATH),
            "--run-root",
            str(audit_log.parent),
            "--output",
            str(exported_file),
        ],
        check=True,
    )
    exported = json.loads(exported_file.read_text(encoding="utf-8"))
    assert exported["missing_count"] == 0
    assert exported["entries"][request_sha]["content"] == content


def test_export_recomputes_stale_normalized_hash_from_raw_request(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    content = "REGENERATE\n```python\npass\n```"
    response_sha = hashlib.sha256(content.encode()).hexdigest()
    request = {
        "model": "qwen3.7-plus",
        "messages": [{
            "role": "user",
            "content": (
                '  File "<CAPX_RUNTIME>/capx/verification/ideal_tk_v4.py", '
                "line 1021, in wrapped\n"
                "    return __original(*args, **kwargs)\n"
                "RuntimeError: no executable grasp\n"
            ),
        }],
    }
    request_canonical, request_sha, _, expected_normalized = PROXY.request_hashes(
        request
    )
    (run_root / "qwen_proxy_audit.jsonl").write_text(
        json.dumps({
            "ok": True,
            "request_sha256": request_sha,
            "request_normalized_sha256": "d" * 64,
            "request_normalization_version": "stale_v1",
            "request_canonical": request_canonical,
            "response_content_sha256": response_sha,
            "response_content": content,
        })
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "replay.json"

    subprocess.run(
        [
            sys.executable,
            str(EXPORT_PATH),
            "--run-root",
            str(run_root),
            "--output",
            str(output),
        ],
        check=True,
    )

    entry = json.loads(output.read_text(encoding="utf-8"))["entries"][request_sha]
    assert entry["request_normalized_sha256"] == expected_normalized
    assert entry["request_normalization_version"] == PROXY.NORMALIZATION_VERSION


def test_export_preserves_audited_visual_fingerprints_without_raw_request(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    content = "visual report"
    response_sha = hashlib.sha256(content.encode()).hexdigest()
    request_sha = "a" * 64
    fingerprints = ["b" * 64, "c" * 64]
    (run_root / "qwen_proxy_audit.jsonl").write_text(
        json.dumps(
            {
                "ok": True,
                "request_sha256": request_sha,
                "request_normalized_sha256": "d" * 64,
                "response_content_sha256": response_sha,
                "response_content": content,
                "request_image_rgb_sha256": fingerprints,
                "effective_request_seed": 7002,
            }
        )
        + "\n"
    )
    output = tmp_path / "replay.json"

    subprocess.run(
        [
            sys.executable,
            str(EXPORT_PATH),
            "--run-root",
            str(run_root),
            "--output",
            str(output),
        ],
        check=True,
    )

    entry = json.loads(output.read_text())["entries"][request_sha]
    assert entry["request_image_rgb_sha256"] == fingerprints


def test_normalized_replay_matches_only_registered_legacy_runtime_path(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "qwen.key"
    key_file.write_text("test-key\n", encoding="utf-8")
    audit_log = tmp_path / "qwen_proxy_audit.jsonl"
    replay_file = tmp_path / "source_replay.json"
    legacy_root = "/home/qshou/capx_pgsvp_v9_verify_clean"
    source_request = {
        "model": "qwen3.7-plus",
        "messages": [{
            "role": "user",
            "content": f'File "{legacy_root}/capx/envs/tasks/base.py"',
        }],
        "seed": 123,
    }
    current_request = json.loads(json.dumps(source_request))
    current_request["messages"][0]["content"] = (
        'File "<CAPX_RUNTIME>/capx/envs/tasks/base.py"'
    )
    _, source_sha, _, source_normalized_sha = PROXY.request_hashes(source_request)
    content = "REGENERATE\n```python\npass\n```"
    replay_file.write_text(
        json.dumps({
            "schema": "capx_request_hash_replay_v1",
            "entries": {
                source_sha: {
                    "content": content,
                    "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "request_normalized_sha256": source_normalized_sha,
                }
            },
        }),
        encoding="utf-8",
    )
    app = PROXY.make_app(
        key_file=str(key_file),
        base_url="https://example.invalid/v1",
        audit_log=str(audit_log),
        upstream_concurrency=1,
        enable_thinking=False,
        request_seed=None,
        replay_file=None,
        request_hash_replay_file=str(replay_file),
        enable_normalized_replay=True,
        replay_legacy_runtime_roots=legacy_root,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/chat/completions"
    )

    class ReplayRequest:
        async def json(self) -> dict:
            return current_request

    response = asyncio.run(endpoint(ReplayRequest()))
    assert response.status_code == 200
    audit = json.loads(audit_log.read_text(encoding="utf-8"))
    assert audit["source"] == "request_legacy_runtime_normalized_hash_replay"
    assert audit["replay_legacy_runtime_root"] == legacy_root


def test_normalized_replay_fails_before_response_on_visual_mismatch(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "qwen.key"
    key_file.write_text("test-key\n", encoding="utf-8")
    audit_log = tmp_path / "qwen_proxy_audit.jsonl"
    replay_file = tmp_path / "source_replay.json"
    source_request = {
        "model": "qwen3.7-plus",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": _image_data_url(0)},
            }],
        }],
        "seed": 123,
    }
    current_request = json.loads(json.dumps(source_request))
    current_request["messages"][0]["content"][0]["image_url"]["url"] = (
        _image_data_url(255)
    )
    _, source_sha, _, normalized_sha = PROXY.request_hashes(source_request)
    content = "FINISH"
    replay_file.write_text(
        json.dumps({
            "schema": "capx_request_hash_replay_v1",
            "entries": {
                source_sha: {
                    "content": content,
                    "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "request_normalized_sha256": normalized_sha,
                    "request_image_rgb_sha256": (
                        PROXY.request_image_rgb_fingerprints(source_request)
                    ),
                }
            },
        }),
        encoding="utf-8",
    )
    app = PROXY.make_app(
        key_file=str(key_file),
        base_url="https://example.invalid/v1",
        audit_log=str(audit_log),
        upstream_concurrency=1,
        enable_thinking=False,
        request_seed=None,
        replay_file=None,
        request_hash_replay_file=str(replay_file),
        enable_normalized_replay=True,
        require_normalized_replay_visual_match=True,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/chat/completions"
    )

    class ReplayRequest:
        async def json(self) -> dict:
            return current_request

    with pytest.raises(HTTPException, match="source image mismatch"):
        asyncio.run(endpoint(ReplayRequest()))
    audit = json.loads(audit_log.read_text(encoding="utf-8"))
    assert audit["source"] == "normalized_replay_visual_mismatch"


def test_normalized_replay_records_verified_visual_match(tmp_path: Path) -> None:
    key_file = tmp_path / "qwen.key"
    key_file.write_text("test-key\n", encoding="utf-8")
    audit_log = tmp_path / "qwen_proxy_audit.jsonl"
    replay_file = tmp_path / "source_replay.json"
    source_request = {
        "model": "qwen3.7-plus",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": _image_data_url(32)},
            }],
        }],
        "seed": 123,
    }
    current_request = json.loads(json.dumps(source_request))
    current_request["messages"][0]["content"][0]["image_url"]["url"] = (
        _image_data_url(32, compress_level=9)
    )
    _, source_sha, _, normalized_sha = PROXY.request_hashes(source_request)
    content = "FINISH"
    fingerprints = PROXY.request_image_rgb_fingerprints(source_request)
    replay_file.write_text(
        json.dumps({
            "schema": "capx_request_hash_replay_v1",
            "entries": {
                source_sha: {
                    "content": content,
                    "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "request_normalized_sha256": normalized_sha,
                    "request_image_rgb_sha256": fingerprints,
                }
            },
        }),
        encoding="utf-8",
    )
    app = PROXY.make_app(
        key_file=str(key_file),
        base_url="https://example.invalid/v1",
        audit_log=str(audit_log),
        upstream_concurrency=1,
        enable_thinking=False,
        request_seed=None,
        replay_file=None,
        request_hash_replay_file=str(replay_file),
        enable_normalized_replay=True,
        require_normalized_replay_visual_match=True,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/chat/completions"
    )

    class ReplayRequest:
        async def json(self) -> dict:
            return current_request

    response = asyncio.run(endpoint(ReplayRequest()))
    assert response.status_code == 200
    audit = json.loads(audit_log.read_text(encoding="utf-8"))
    assert audit["source"] == "request_normalized_hash_replay"
    assert audit["normalized_replay_visual_match_verified"] is True
    assert audit["expected_request_image_rgb_sha256"] == fingerprints
    assert audit["request_image_rgb_sha256"] == fingerprints


def test_m3_prefix_replay_stops_after_action_divergence(tmp_path: Path) -> None:
    key_file = tmp_path / "qwen.key"
    key_file.write_text("test-key\n", encoding="utf-8")
    audit_log = tmp_path / "qwen_proxy_audit.jsonl"
    replay_file = tmp_path / "source_replay.json"
    fallback_file = tmp_path / "fallback.json"
    fallback_file.write_text("[]", encoding="utf-8")

    report_2 = {
        "model": "qwen3.7-plus",
        "messages": [{"role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": _image_data_url(0)},
        }]}],
        "seed": 7002,
    }
    action_3 = {
        "model": "qwen3.7-plus",
        "messages": [{"role": "user", "content": "source report"}],
        "seed": 7003,
    }
    report_4 = {**json.loads(json.dumps(report_2)), "seed": 7004}
    entries = {}
    for request, turn, content in (
        (report_2, 2, "source report"),
        (action_3, 3, "source action"),
        (report_4, 4, "later source report"),
    ):
        _, raw_sha, _, normalized_sha = PROXY.request_hashes(request)
        entries[raw_sha] = {
            "content": content,
            "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "request_normalized_sha256": normalized_sha,
            "request_image_rgb_sha256": (
                PROXY.request_image_rgb_fingerprints(request)
            ),
            "effective_request_seed": request["seed"],
            "trial": 1,
            "turn": turn,
        }
    replay_file.write_text(
        json.dumps({"schema": "capx_request_hash_replay_v1", "entries": entries}),
        encoding="utf-8",
    )
    app = PROXY.make_app(
        key_file=str(key_file),
        base_url="https://example.invalid/v1",
        audit_log=str(audit_log),
        upstream_concurrency=1,
        enable_thinking=False,
        request_seed=None,
        replay_file=str(fallback_file),
        request_hash_replay_file=str(replay_file),
        enable_normalized_replay=True,
        require_normalized_replay_visual_match=True,
        m3_prefix_replay=True,
        allow_deferred_semantic_visual_match=True,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/chat/completions"
    )

    class ReplayRequest:
        def __init__(self, body: dict):
            self.body = body

        async def json(self) -> dict:
            return self.body

    current_report_2 = json.loads(json.dumps(report_2))
    current_report_2["messages"][0]["content"][0]["image_url"]["url"] = (
        _image_data_url(255)
    )
    changed_action_3 = json.loads(json.dumps(action_3))
    changed_action_3["messages"][0]["content"] = "corrected report"
    for request in (current_report_2, changed_action_3, report_4):
        response = asyncio.run(endpoint(ReplayRequest(request)))
        assert response.status_code == 200

    records = [json.loads(line) for line in audit_log.read_text().splitlines()]
    assert records[0]["source"] == "request_normalized_hash_replay"
    assert records[0]["normalized_replay_visual_match_deferred_to_semantics"] is True
    assert records[1]["source"] == "replay"
    assert records[1]["m3_prefix_diverged_on_request"] is True
    assert records[2]["source"] == "replay"
    assert records[2]["m3_prefix_replay_blocked"] is True


def test_require_all_replay_fails_closed_on_any_miss(tmp_path: Path) -> None:
    key_file = tmp_path / "qwen.key"
    key_file.write_text("test-key\n", encoding="utf-8")
    audit_log = tmp_path / "qwen_proxy_audit.jsonl"
    replay_file = tmp_path / "empty_replay.json"
    replay_file.write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "entries": {},
            }
        ),
        encoding="utf-8",
    )
    request = {
        "model": "qwen3.7-plus",
        "messages": [{"role": "user", "content": "unfrozen request"}],
        "temperature": 0,
        "seed": 999,
    }
    app = PROXY.make_app(
        key_file=str(key_file),
        base_url="https://example.invalid/v1",
        audit_log=str(audit_log),
        upstream_concurrency=1,
        enable_thinking=False,
        request_seed=None,
        replay_file=None,
        request_hash_replay_file=str(replay_file),
        require_all_request_hash_replay=True,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/chat/completions"
    )

    class ReplayRequest:
        async def json(self) -> dict:
            return request

    with pytest.raises(HTTPException, match="Required request-hash replay miss"):
        asyncio.run(endpoint(ReplayRequest()))

    audit = json.loads(audit_log.read_text(encoding="utf-8"))
    assert audit["source"] == "request_hash_replay_required_miss"
    assert audit["require_all_request_hash_replay"] is True
