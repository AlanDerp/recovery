import itertools
import json
import logging
import time
from pathlib import Path
from typing import List, Literal, Optional, Union

import httpx
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIStatusError, AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentItem] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "openrouter/google/gemini-2.5-pro-preview"
    messages: list[Message]
    temperature: float | None = 0.2
    max_tokens: int | None = 256
    stream: bool = False
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]


def _http_status_for_openai_error(exc: APIStatusError) -> int:
    status = int(getattr(exc, "status_code", 500) or 500)
    if status in {400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}:
        return status
    return 500


def _log_openai_error(exc: APIStatusError) -> None:
    logger.error(
        "OpenRouter upstream error status=%s body=%s",
        getattr(exc, "status_code", None),
        getattr(exc, "body", None) or str(exc),
    )


def _openai_error_response(exc: APIStatusError) -> JSONResponse:
    _log_openai_error(exc)
    status = _http_status_for_openai_error(exc)
    detail = str(getattr(exc, "body", None) or exc)
    return JSONResponse(status_code=status, content={"detail": detail})


def _load_api_keys(key_file: str) -> list[str]:
    """Load API keys from a file, one key per line. Ignores blank lines and comments."""
    path = Path(key_file)
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {key_file}")
    keys = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    if not keys:
        raise ValueError(f"No API keys found in {key_file}")
    return keys


def create_app(api_key: str, base_url: str, async_client: bool = True) -> FastAPI:
    default_headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
        "X-Title": "CaP-X",
    }

    if async_client:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )
    else:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    app = FastAPI(title="OpenRouter Proxy", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if async_client:

        @app.post("/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                client_kwargs = request.model_dump(exclude_none=True)

                # Strip the "openrouter/" prefix if present so OpenRouter sees the
                # native model identifier (e.g. "google/gemini-2.5-pro-preview").
                model = client_kwargs.get("model", "")
                if model.startswith("openrouter/"):
                    client_kwargs["model"] = model[len("openrouter/"):]

                url = base_url.rstrip("/") + "/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    **default_headers,
                }
                async with httpx.AsyncClient(timeout=200.0) as http_client:
                    if request.stream:
                        upstream = await http_client.post(
                            url, headers=headers, json=client_kwargs
                        )
                    else:
                        client_kwargs["stream"] = False
                        upstream = await http_client.post(
                            url, headers=headers, json=client_kwargs
                        )
                if upstream.status_code >= 400:
                    return JSONResponse(
                        status_code=upstream.status_code,
                        content={"detail": upstream.text},
                    )
                return JSONResponse(content=upstream.json())

            except APIStatusError as e:
                return _openai_error_response(e)
            except Exception as e:
                logger.exception("OpenRouter proxy failed")
                raise HTTPException(status_code=500, detail=str(e)) from e

    else:

        @app.post("/chat/completions", response_model=ChatCompletionResponse)
        def chat_completions(request: ChatCompletionRequest):
            try:
                client_kwargs = request.model_dump(exclude_none=True)

                model = client_kwargs.get("model", "")
                if model.startswith("openrouter/"):
                    client_kwargs["model"] = model[len("openrouter/"):]

                client_kwargs["stream"] = False

                response = client.chat.completions.create(**client_kwargs)

                choices = [
                    ChatCompletionResponseChoice(
                        index=c.index,
                        message=Message(role=c.message.role, content=c.message.content),
                        finish_reason=c.finish_reason,
                    )
                    for c in response.choices
                ]

                return ChatCompletionResponse(
                    id=response.id, created=response.created, model=response.model, choices=choices
                )

            except APIStatusError as e:
                return _openai_error_response(e)
            except Exception as e:
                logger.exception("OpenRouter proxy failed")
                raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def main(
    key_file: str = ".openrouterkey",
    api_key: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8111,
    base_url: str = "https://openrouter.ai/api/v1/",
    async_client: bool = True,
):
    """
    Start the OpenRouter Proxy Server.

    Reads an API key from --api-key or from the key file (one key per line).
    """
    if api_key is None:
        keys = _load_api_keys(key_file)
        api_key = keys[0]
        logger.info(f"Loaded API key from {key_file}")

    app = create_app(api_key=api_key, base_url=base_url, async_client=async_client)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
