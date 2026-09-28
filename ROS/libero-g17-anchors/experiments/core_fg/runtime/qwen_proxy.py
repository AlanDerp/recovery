#!/usr/bin/env python3
"""Small OpenAI-compatible proxy for Qwen CAP-X runs.

This intentionally does less than capx.serving.openrouter_server:
- preserves upstream HTTP status codes for rate-limit/billing failures;
- writes JSONL audit records per request;
- returns OpenAI-compatible chat completion JSON to CAP-X.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from PIL import Image
import uvicorn


NORMALIZATION_VERSION = "image_url_redacted_verifier_frame_removed_v2"
_VERIFIER_WRAPPER_FRAME_RE = re.compile(
    r'(?m)^[ \t]*File "(?:<CAPX_RUNTIME>|[^"\n]*)/capx/verification/'
    r'[^"\n]+\.py", line \d+, in wrapped\n(?:[ \t]+[^\n]*(?:\n|$))?'
)
TRANSIENT_MEDIA_RETRY_ATTEMPTS = 2
TRANSIENT_MEDIA_RETRY_DELAY_SECONDS = 1.0
TRANSIENT_TRANSPORT_RETRY_ATTEMPTS = 2
UPSTREAM_TIMEOUT_SECONDS = 240.0


def retryable_transient_media_error(error: APIStatusError) -> bool:
    """Recognize the provider's transient image-fetch failure, not generic 4xx."""
    status = int(getattr(error, "status_code", 500) or 500)
    return status == 400 and "download multimodal file timed out" in str(error).lower()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def normalize_request_for_replay(value: Any) -> Any:
    """Normalize non-semantic request fields for conservative replay matching.

    Raw image bytes are removed from OpenAI-compatible multimodal messages.
    Internal verifier wrapper frames are also removed because their source line
    changes when instrumentation code moves while the propagated error does not.
    """
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if key == "image_url" and isinstance(child, dict) and "url" in child:
                image_child = dict(child)
                image_child["url"] = "<CAPX_IMAGE_URL_REDACTED_FOR_REPLAY>"
                normalized[key] = image_child
            else:
                normalized[key] = normalize_request_for_replay(child)
        return normalized
    if isinstance(value, list):
        return [normalize_request_for_replay(item) for item in value]
    if isinstance(value, str):
        return _VERIFIER_WRAPPER_FRAME_RE.sub("", value)
    return value


def request_hashes(body: Any) -> tuple[str, str, str, str]:
    raw_canonical = canonical_json(body)
    raw_sha = hashlib.sha256(raw_canonical.encode()).hexdigest()
    normalized_canonical = canonical_json(normalize_request_for_replay(body))
    normalized_sha = hashlib.sha256(normalized_canonical.encode()).hexdigest()
    return raw_canonical, raw_sha, normalized_canonical, normalized_sha


def request_image_rgb_fingerprints(value: Any) -> list[str]:
    """Hash decoded RGB pixels in request order, including image dimensions."""
    fingerprints: list[str] = []

    def visit(child: Any) -> None:
        if isinstance(child, dict):
            image = child.get("image_url")
            if isinstance(image, dict):
                url = image.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    _, separator, encoded = url.partition(",")
                    if not separator:
                        raise ValueError("malformed request image data URL")
                    with Image.open(io.BytesIO(base64.b64decode(encoded))) as source:
                        rgb = source.convert("RGB")
                        payload = (
                            rgb.width.to_bytes(4, "big")
                            + rgb.height.to_bytes(4, "big")
                            + rgb.tobytes()
                        )
                    fingerprints.append(hashlib.sha256(payload).hexdigest())
            for key, item in child.items():
                if key != "image_url":
                    visit(item)
        elif isinstance(child, list):
            for item in child:
                visit(item)

    visit(value)
    return fingerprints


def load_key(path: str) -> str:
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise RuntimeError(f"No API key found in {path}")


def parse_required_replay_seeds(value: str | None) -> set[int]:
    if not value:
        return set()
    seeds: set[int] = set()
    for item in value.replace("+", ",").split(","):
        item = item.strip()
        if item:
            seeds.add(int(item))
    return seeds


def parse_legacy_runtime_roots(value: str | None) -> list[str]:
    if not value:
        return []
    roots = []
    for item in value.split("+"):
        root = item.strip().rstrip("/")
        if root and root not in roots:
            roots.append(root)
    return roots


def load_request_hash_replay(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = json.loads(Path(path).read_text())
    if source.get("schema") != "capx_request_hash_replay_v1":
        raise RuntimeError(f"Unsupported request-hash replay schema in {path}")
    entries = source.get("entries") or {}
    if not isinstance(entries, dict):
        raise RuntimeError(f"request-hash replay entries must be an object in {path}")
    loaded: dict[str, dict[str, Any]] = {}
    for request_hash, entry in entries.items():
        content = str(entry.get("content") or "")
        if not content:
            raise RuntimeError(f"Empty replay content for request hash {request_hash}")
        expected = str(entry.get("response_sha256") or "")
        actual = hashlib.sha256(content.encode()).hexdigest()
        if expected and expected != actual:
            raise RuntimeError(
                f"Replay content hash mismatch for {request_hash}: {actual} != {expected}"
            )
        loaded[str(request_hash)] = dict(entry, content=content, response_sha256=actual)
    return loaded


def make_app(
    key_file: str,
    base_url: str,
    audit_log: str,
    upstream_concurrency: int,
    enable_thinking: bool | None,
    request_seed: int | None,
    replay_file: str | None,
    request_hash_replay_file: str | None = None,
    required_replay_seeds: str | None = None,
    audit_raw_content: bool = False,
    audit_raw_request: bool = False,
    audit_raw_request_on_miss: bool = False,
    enable_normalized_replay: bool = False,
    require_all_request_hash_replay: bool = False,
    require_normalized_replay_visual_match: bool = False,
    replay_legacy_runtime_roots: str | None = None,
    m3_prefix_replay: bool = False,
    allow_deferred_semantic_visual_match: bool = False,
) -> FastAPI:
    # Bound each upstream attempt below the CAP-X client/trial watchdogs. The
    # proxy performs the retries itself so the exact payload and request seed
    # remain unchanged and only one local request is in flight.
    client = AsyncOpenAI(
        api_key=load_key(key_file),
        base_url=base_url,
        timeout=UPSTREAM_TIMEOUT_SECONDS,
        max_retries=0,
    )
    app = FastAPI(title="Qwen CAP-X Proxy")
    audit_path = Path(audit_log)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    import asyncio

    sem = asyncio.Semaphore(upstream_concurrency) if upstream_concurrency > 0 else None
    request_hash_replay = load_request_hash_replay(request_hash_replay_file)
    normalized_request_hash_replay: dict[str, dict[str, Any]] = {}
    if enable_normalized_replay:
        for request_hash, entry in request_hash_replay.items():
            normalized_hash = str(entry.get("request_normalized_sha256") or "")
            if not normalized_hash:
                continue
            existing = normalized_request_hash_replay.get(normalized_hash)
            if existing and existing.get("response_sha256") != entry.get("response_sha256"):
                raise RuntimeError(
                    "Conflicting normalized replay entries for "
                    f"{normalized_hash}: {existing.get('response_sha256')} != "
                    f"{entry.get('response_sha256')}"
                )
            normalized_request_hash_replay[normalized_hash] = dict(
                entry,
                request_sha256=request_hash,
            )
    required_replay_seed_set = parse_required_replay_seeds(required_replay_seeds)
    legacy_runtime_roots = parse_legacy_runtime_roots(replay_legacy_runtime_roots)
    replay_seed_positions: dict[int, tuple[int, int]] = {}
    for entry in request_hash_replay.values():
        seed = entry.get("effective_request_seed")
        trial = entry.get("trial")
        turn = entry.get("turn")
        if all(isinstance(value, int) for value in (seed, trial, turn)):
            replay_seed_positions[int(seed)] = (int(trial), int(turn))
    diverged_m3_trials: set[int] = set()
    replay_responses: list[str] | None = None
    if replay_file:
        source = json.loads(Path(replay_file).read_text())
        replay_responses = []
        for entry in source:
            decision = str(entry.get("decision", "")).lower()
            blocks = entry.get("code_blocks") or []
            if decision == "initial" and blocks:
                replay_responses.append(str(blocks[0]))
            elif decision == "regenerate" and blocks:
                replay_responses.append(
                    "REGENERATE\n```python\n" + str(blocks[-1]) + "\n```"
                )

    def audit(record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat/completions")
    async def chat_completions(req: Request) -> JSONResponse:
        body = await req.json()
        (
            request_canonical,
            request_sha256,
            request_normalized_canonical,
            request_normalized_sha256,
        ) = request_hashes(body)
        start = time.time()
        model = body.get("model")
        effective_seed = body.get("seed", request_seed)
        upstream_attempt_count = 0
        transient_upstream_errors: list[dict[str, Any]] = []
        replay_position = replay_seed_positions.get(effective_seed)
        replay_trial = replay_position[0] if replay_position else None
        replay_turn = replay_position[1] if replay_position else None
        prefix_replay_blocked = bool(
            m3_prefix_replay
            and replay_trial is not None
            and replay_trial in diverged_m3_trials
        )
        m3_diverged_on_request = False
        try:
            replay_entry = (
                None if prefix_replay_blocked
                else request_hash_replay.get(request_sha256)
            )
            replay_source = "request_hash_replay"
            replay_legacy_runtime_root = None
            replay_expected_images = None
            replay_current_images = None
            replay_visual_match_verified = False
            if (
                replay_entry is None
                and enable_normalized_replay
                and not prefix_replay_blocked
            ):
                replay_entry = normalized_request_hash_replay.get(request_normalized_sha256)
                replay_source = "request_normalized_hash_replay"
                if (
                    replay_entry is None
                    and "<CAPX_RUNTIME>" in request_normalized_canonical
                ):
                    for legacy_root in legacy_runtime_roots:
                        legacy_canonical = request_normalized_canonical.replace(
                            "<CAPX_RUNTIME>", legacy_root
                        )
                        legacy_hash = hashlib.sha256(
                            legacy_canonical.encode()
                        ).hexdigest()
                        replay_entry = normalized_request_hash_replay.get(legacy_hash)
                        if replay_entry is not None:
                            replay_source = (
                                "request_legacy_runtime_normalized_hash_replay"
                            )
                            replay_legacy_runtime_root = legacy_root
                            break
                if replay_entry is not None and require_normalized_replay_visual_match:
                    replay_expected_images = replay_entry.get(
                        "request_image_rgb_sha256"
                    )
                    replay_current_images = request_image_rgb_fingerprints(body)
                    if (
                        not isinstance(replay_expected_images, list)
                        or replay_current_images != replay_expected_images
                    ):
                        can_defer_to_semantics = bool(
                            allow_deferred_semantic_visual_match
                            and m3_prefix_replay
                            and replay_trial is not None
                            and isinstance(replay_turn, int)
                            and replay_turn > 0
                            and replay_turn % 2 == 0
                        )
                        if not can_defer_to_semantics:
                            audit({
                                "ok": False,
                                "status_code": 409,
                                "source": "normalized_replay_visual_mismatch",
                                "model": model,
                                "effective_request_seed": effective_seed,
                                "request_sha256": request_sha256,
                                "request_normalized_sha256": request_normalized_sha256,
                                "expected_request_image_rgb_sha256": replay_expected_images,
                                "request_image_rgb_sha256": replay_current_images,
                                "request_hash_replay_file": request_hash_replay_file,
                                **(
                                    {"request_canonical": request_canonical}
                                    if (audit_raw_request or audit_raw_request_on_miss)
                                    else {}
                                ),
                            })
                            raise HTTPException(
                                status_code=409,
                                detail="Normalized replay source image mismatch",
                            )
                    else:
                        replay_visual_match_verified = True
            if replay_entry is not None:
                content = str(replay_entry["content"])
                data = {
                    "id": f"capx-request-hash-replay-{request_sha256[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                audit({
                    "ok": True,
                    "status_code": 200,
                    "source": replay_source,
                    "model": model,
                    "configured_request_seed": request_seed,
                    "effective_request_seed": effective_seed,
                    "request_sha256": request_sha256,
                    "request_normalized_sha256": request_normalized_sha256,
                    "request_normalization_version": NORMALIZATION_VERSION,
                    "response_content_sha256": replay_entry["response_sha256"],
                    "request_hash_replay_file": request_hash_replay_file,
                    "replay_request_sha256": replay_entry.get("request_sha256", request_sha256),
                    "replay_request_normalized_sha256": replay_entry.get("request_normalized_sha256"),
                    "replay_role": replay_entry.get("role"),
                    "m3_prefix_replay": m3_prefix_replay,
                    "m3_prefix_replay_blocked": prefix_replay_blocked,
                    "replay_trial": replay_trial,
                    "replay_turn": replay_turn,
                    "replay_legacy_runtime_root": replay_legacy_runtime_root,
                    "normalized_replay_visual_match_required": (
                        require_normalized_replay_visual_match
                    ),
                    "normalized_replay_visual_match_verified": (
                        replay_visual_match_verified
                    ),
                    "normalized_replay_visual_match_deferred_to_semantics": bool(
                        replay_source != "request_hash_replay"
                        and require_normalized_replay_visual_match
                        and not replay_visual_match_verified
                    ),
                    "expected_request_image_rgb_sha256": replay_expected_images,
                    "request_image_rgb_sha256": replay_current_images,
                    **({"response_content": content} if audit_raw_content else {}),
                    **({"request_canonical": request_canonical} if audit_raw_request else {}),
                })
                return JSONResponse(data, status_code=200)

            if (
                require_all_request_hash_replay
                or effective_seed in required_replay_seed_set
            ):
                audit({
                    "ok": False,
                    "status_code": 409,
                    "source": "request_hash_replay_required_miss",
                    "model": model,
                    "configured_request_seed": request_seed,
                    "effective_request_seed": effective_seed,
                    "request_sha256": request_sha256,
                    "request_normalized_sha256": request_normalized_sha256,
                    "request_normalization_version": NORMALIZATION_VERSION,
                    "request_hash_replay_file": request_hash_replay_file,
                    "normalized_replay_enabled": enable_normalized_replay,
                    "latency_s": round(time.time() - start, 3),
                    "require_all_request_hash_replay": require_all_request_hash_replay,
                    "error": (
                        "required request did not match any frozen request hash"
                        if require_all_request_hash_replay
                        else "required replay seed did not match any frozen request hash"
                    ),
                    **(
                        {"request_canonical": request_canonical}
                        if (audit_raw_request or audit_raw_request_on_miss)
                        else {}
                    ),
                })
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Required request-hash replay miss for "
                        f"seed={effective_seed} sha={request_sha256} "
                        f"require_all={require_all_request_hash_replay}"
                    ),
                )

            if (
                m3_prefix_replay
                and replay_trial is not None
                and isinstance(replay_turn, int)
                and replay_turn % 2 == 1
                and replay_trial not in diverged_m3_trials
            ):
                diverged_m3_trials.add(replay_trial)
                m3_diverged_on_request = True

            if replay_responses is not None:
                query_index = int(effective_seed or 0) % 1000
                content = (
                    replay_responses[query_index]
                    if query_index < len(replay_responses)
                    else "FINISH"
                )
                data = {
                    "id": f"capx-replay-{query_index}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                audit({
                    "ok": True,
                    "status_code": 200,
                    "source": "replay",
                    "model": model,
                    "configured_request_seed": request_seed,
                    "effective_request_seed": effective_seed,
                    "query_index": query_index,
                    "request_sha256": request_sha256,
                    "request_normalized_sha256": request_normalized_sha256,
                    "request_normalization_version": NORMALIZATION_VERSION,
                    "request_image_rgb_sha256": request_image_rgb_fingerprints(body),
                    "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "m3_prefix_replay": m3_prefix_replay,
                    "m3_prefix_replay_blocked": prefix_replay_blocked,
                    "replay_trial": replay_trial,
                    "replay_turn": replay_turn,
                    "m3_prefix_diverged_on_request": m3_diverged_on_request,
                    "replay_file": replay_file,
                    **({"request_canonical": request_canonical} if audit_raw_request else {}),
                })
                return JSONResponse(data, status_code=200)

            kwargs = dict(body)
            kwargs["stream"] = False
            if enable_thinking is not None:
                extra_body = dict(kwargs.get("extra_body") or {})
                extra_body.setdefault("enable_thinking", enable_thinking)
                kwargs["extra_body"] = extra_body
            if request_seed is not None:
                kwargs.setdefault("seed", request_seed)
            upstream_seed = kwargs.get("seed")
            retry_attempts = max(
                TRANSIENT_MEDIA_RETRY_ATTEMPTS,
                TRANSIENT_TRANSPORT_RETRY_ATTEMPTS,
            )
            for upstream_attempt_count in range(1, retry_attempts + 2):
                try:
                    if sem is None:
                        response = await client.chat.completions.create(**kwargs)
                    else:
                        async with sem:
                            response = await client.chat.completions.create(**kwargs)
                    break
                except APIStatusError as exc:
                    if (
                        not retryable_transient_media_error(exc)
                        or upstream_attempt_count > TRANSIENT_MEDIA_RETRY_ATTEMPTS
                    ):
                        raise
                    transient_upstream_errors.append(
                        {
                            "attempt": upstream_attempt_count,
                            "status_code": int(
                                getattr(exc, "status_code", 500) or 500
                            ),
                            "error": str(exc)[:1000],
                        }
                    )
                    await asyncio.sleep(TRANSIENT_MEDIA_RETRY_DELAY_SECONDS)
                except (APITimeoutError, APIConnectionError) as exc:
                    if upstream_attempt_count > TRANSIENT_TRANSPORT_RETRY_ATTEMPTS:
                        raise
                    transient_upstream_errors.append(
                        {
                            "attempt": upstream_attempt_count,
                            "status_code": None,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                        }
                    )
                    await asyncio.sleep(TRANSIENT_MEDIA_RETRY_DELAY_SECONDS)

            data = response.model_dump()
            choices = data.get("choices") or []
            first = choices[0] if choices else {}
            msg = first.get("message") or {}
            usage = data.get("usage")
            content = msg.get("content") or ""
            audit(
                {
                    "ok": True,
                    "status_code": 200,
                    "model": model,
                    "configured_request_seed": request_seed,
                    "effective_request_seed": effective_seed,
                    "upstream_seed": upstream_seed,
                    "upstream_attempt_count": upstream_attempt_count,
                    "transient_upstream_errors": transient_upstream_errors,
                    "request_sha256": request_sha256,
                    "request_normalized_sha256": request_normalized_sha256,
                    "request_normalization_version": NORMALIZATION_VERSION,
                    "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "latency_s": round(time.time() - start, 3),
                    "m3_prefix_replay": m3_prefix_replay,
                    "m3_prefix_replay_blocked": prefix_replay_blocked,
                    "replay_trial": replay_trial,
                    "replay_turn": replay_turn,
                    "m3_prefix_diverged_on_request": m3_diverged_on_request,
                    "finish_reason": first.get("finish_reason"),
                    "content_is_none": msg.get("content") is None,
                    "content_len": len(msg.get("content") or ""),
                    "usage": usage,
                    "id": data.get("id"),
                    **({"response_content": content} if audit_raw_content else {}),
                    **({"request_canonical": request_canonical} if audit_raw_request else {}),
                }
            )
            return JSONResponse(data, status_code=200)
        except HTTPException:
            raise
        except APIStatusError as e:
            status = int(getattr(e, "status_code", 500) or 500)
            detail = str(e)
            audit(
                {
                    "ok": False,
                    "status_code": status,
                    "model": model,
                    "configured_request_seed": request_seed,
                    "effective_request_seed": effective_seed,
                    "request_sha256": request_sha256,
                    "request_normalized_sha256": request_normalized_sha256,
                    "request_normalization_version": NORMALIZATION_VERSION,
                    "latency_s": round(time.time() - start, 3),
                    "error_type": type(e).__name__,
                    "error": detail[:4000],
                    "upstream_attempt_count": upstream_attempt_count,
                    "transient_upstream_errors": transient_upstream_errors,
                    **(
                        {"request_canonical": request_canonical}
                        if (audit_raw_request or audit_raw_request_on_miss)
                        else {}
                    ),
                }
            )
            # Preserve 4xx so CAP-X does not retry quota/billing/auth failures as 500.
            raise HTTPException(status_code=status, detail=detail)
        except Exception as e:
            audit(
                {
                    "ok": False,
                    "status_code": 500,
                    "model": model,
                    "configured_request_seed": request_seed,
                    "effective_request_seed": effective_seed,
                    "request_sha256": request_sha256,
                    "request_normalized_sha256": request_normalized_sha256,
                    "request_normalization_version": NORMALIZATION_VERSION,
                    "latency_s": round(time.time() - start, 3),
                    "error_type": type(e).__name__,
                    "error": str(e)[:4000],
                    "upstream_attempt_count": upstream_attempt_count,
                    "transient_upstream_errors": transient_upstream_errors,
                    **(
                        {"request_canonical": request_canonical}
                        if (audit_raw_request or audit_raw_request_on_miss)
                        else {}
                    ),
                }
            )
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--audit-log", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--upstream-concurrency", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false", "unset"],
        default="false",
        help="Inject Qwen extra_body.enable_thinking. Default false for DeepSeek-style codegen comparability.",
    )
    parser.add_argument(
        "--m3-prefix-replay",
        action="store_true",
        help="Disable all later native replay after the first M3 action miss.",
    )
    parser.add_argument(
        "--allow-deferred-semantic-visual-match",
        action="store_true",
        help=(
            "Allow a pre-divergence M3 report replay with different RGB only "
            "when the experiment validator will certify its GT predicate vector."
        ),
    )
    parser.add_argument(
        "--request-seed",
        type=int,
        default=None,
        help="Optional request-level seed forwarded to the OpenAI-compatible upstream if supported.",
    )
    parser.add_argument(
        "--replay-file",
        default=None,
        help="Replay a saved CaP-X all_responses.json instead of calling upstream.",
    )
    parser.add_argument(
        "--request-hash-replay-file",
        default=None,
        help="Replay selected responses by exact request SHA-256, while other requests call upstream.",
    )
    parser.add_argument(
        "--required-replay-seeds",
        default=None,
        help="Comma- or plus-separated effective request seeds that must hit request-hash replay.",
    )
    parser.add_argument(
        "--audit-raw-content",
        action="store_true",
        help=(
            "Write assistant response content into the audit JSONL. "
            "Use this only for reproducibility runs; it can substantially increase log size."
        ),
    )
    parser.add_argument(
        "--audit-raw-request",
        action="store_true",
        help="Write canonical request JSON into every audit record. Use only for reproducibility debugging.",
    )
    parser.add_argument(
        "--audit-raw-request-on-miss",
        action="store_true",
        help="Write canonical request JSON only for replay misses or upstream errors.",
    )
    parser.add_argument(
        "--enable-normalized-replay",
        action="store_true",
        help=(
            "Allow replay by normalized request hash after exact raw hash misses. "
            "Current normalization redacts image_url.url only; text remains exact."
        ),
    )
    parser.add_argument(
        "--require-all-request-hash-replay",
        action="store_true",
        help=(
            "Fail closed when any chat request misses request-hash replay. "
            "This freezes a complete multi-turn model trajectory."
        ),
    )
    parser.add_argument(
        "--require-normalized-replay-visual-match",
        action="store_true",
        help=(
            "Require decoded source/current image fingerprints to match before "
            "serving a normalized-hash replay response."
        ),
    )
    parser.add_argument(
        "--replay-legacy-runtime-roots",
        default=None,
        help=(
            "Plus-separated legacy source roots used only to match traceback "
            "paths that are now normalized to <CAPX_RUNTIME>."
        ),
    )
    args = parser.parse_args()
    enable_thinking = (
        True if args.enable_thinking == "true" else False if args.enable_thinking == "false" else None
    )
    request_seed = args.request_seed
    if request_seed is not None and request_seed < 0:
        request_seed = None
    uvicorn.run(
        make_app(
            args.key_file,
            args.base_url,
            args.audit_log,
            args.upstream_concurrency,
            enable_thinking,
            request_seed,
            args.replay_file,
            args.request_hash_replay_file,
            args.required_replay_seeds,
            args.audit_raw_content,
            args.audit_raw_request,
            args.audit_raw_request_on_miss,
            args.enable_normalized_replay,
            args.require_all_request_hash_replay,
            args.require_normalized_replay_visual_match,
            args.replay_legacy_runtime_roots,
            args.m3_prefix_replay,
            args.allow_deferred_semantic_visual_match,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
