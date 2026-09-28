#!/usr/bin/env python3
"""Minimal fail-closed OpenAI-compatible proxy for v269 mechanism replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn


SCHEMA = "capx_libero_g17_v269_effective_seed_replay_v1"


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def make_app(package_path: Path, audit_path: Path) -> FastAPI:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get("schema") != SCHEMA:
        raise RuntimeError(f"unsupported replay schema: {package.get('schema')!r}")
    entries = package.get("entries") or {}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Qwen v269 effective-seed replay")

    def audit(row: dict[str, Any]) -> None:
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**row, "ts": time.time()}, sort_keys=True) + "\n")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        seed = body.get("seed")
        entry = entries.get(str(seed))
        request_sha = canonical_sha(body)
        if entry is None:
            audit(
                {
                    "ok": False,
                    "seed": seed,
                    "request_sha256": request_sha,
                    "source": "required_miss",
                }
            )
            raise HTTPException(status_code=409, detail=f"v269 replay miss for seed={seed}")
        content = str(entry["content"])
        actual = hashlib.sha256(content.encode()).hexdigest()
        if actual != entry["content_sha256"]:
            raise HTTPException(
                status_code=500,
                detail=f"v269 replay content drift for seed={seed}",
            )
        audit(
            {
                "ok": True,
                "seed": seed,
                "request_sha256": request_sha,
                "response_content_sha256": actual,
                "historical_request_sha256": entry.get("historical_request_sha256"),
                "historical_response_sha256": entry.get("historical_response_sha256"),
                "role": entry.get("role"),
                "source": "v269_effective_seed_replay",
            }
        )
        return JSONResponse(
            {
                "id": f"capx-v269-replay-{seed}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--audit-log", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(
        make_app(args.package, args.audit_log),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
