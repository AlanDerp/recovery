from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import PIL
import requests
from PIL import Image

_PROC: Any | None = None  # kept for backward compatibility; unused with vLLM HTTP API
_MODEL: Any | None = None  # kept for backward compatibility; unused with vLLM HTTP API

# SERVICE_URL = "https://openrouter.ai/api/" # OpenRouter
# Must honor sbatch MOLMO_SERVICE_URL (dynamic port). Hardcoding 8122 breaks
# multiview when Molmo listens elsewhere — G17 overlay previously did that.
SERVICE_URL = os.environ.get("MOLMO_SERVICE_URL", "http://127.0.0.1:8122/v1")

_AUDIT_LOCK = threading.Lock()
_REPLAY_LOCK = threading.Lock()
_REPLAY_CACHE: dict[str, deque[dict[str, Any]]] | None = None

MOLMO_AUDIT_SCHEMA = "capx_molmo_response_audit_v1"
MOLMO_REPLAY_SCHEMA = "capx_molmo_response_replay_v1"


def reset_molmo_replay_cache() -> None:
    """Test helper: force reload of CAPX_MOLMO_RESPONSE_REPLAY_FILE."""
    global _REPLAY_CACHE
    _REPLAY_CACHE = None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _png_bytes(image: PIL.Image.Image) -> bytes:
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        return buf.getvalue()


def _molmo_request_envelope(
    *,
    model_name: str,
    prompt: str,
    image_png_sha256: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "endpoint": "chat_completions_point",
        "model": model_name,
        "prompt": prompt,
        "image_png_sha256": image_png_sha256,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def _append_molmo_audit(record: dict[str, Any]) -> None:
    path = os.environ.get("CAPX_MOLMO_AUDIT_FILE", "")
    if not path:
        return
    with _AUDIT_LOCK:
        audit = Path(path)
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()


def _load_molmo_replay() -> dict[str, deque[dict[str, Any]]]:
    global _REPLAY_CACHE
    if _REPLAY_CACHE is not None:
        return _REPLAY_CACHE
    path = os.environ.get("CAPX_MOLMO_RESPONSE_REPLAY_FILE", "")
    replay: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    if not path:
        _REPLAY_CACHE = replay
        return replay
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != MOLMO_REPLAY_SCHEMA:
        raise ValueError(f"Unsupported Molmo replay schema in {path}")
    for entry in payload.get("entries", []):
        request = entry["request"]
        digest = _canonical_sha256(request)
        if digest != entry["request_sha256"]:
            raise ValueError(f"Corrupt Molmo replay request hash in {path}")
        replay[digest].append(
            {
                "response": entry["response"],
                "response_sha256": entry["response_sha256"],
            }
        )
    _REPLAY_CACHE = replay
    return replay


def _resolve_molmo_content(
    *,
    request: dict[str, Any],
    live_call: Callable[[], str],
) -> str:
    """Return generated text; audit + optional fail-closed replay."""
    request_sha256 = _canonical_sha256(request)
    replay = _load_molmo_replay()
    with _REPLAY_LOCK:
        queued = replay.get(request_sha256)
        hit = queued.popleft() if queued else None
    if hit is not None:
        content = str(hit["response"].get("content", ""))
        source = "registered_replay"
        response = {"content": content}
    else:
        if os.environ.get("CAPX_MOLMO_REQUIRE_ALL_REPLAY", "0") == "1":
            raise RuntimeError(
                f"Unregistered Molmo request {request_sha256}"
            )
        content = live_call()
        source = "live_server"
        response = {"content": content}
    record = {
        "schema": MOLMO_AUDIT_SCHEMA,
        "request": request,
        "request_sha256": request_sha256,
        "response": response,
        "response_sha256": _canonical_sha256(response),
        "source": source,
    }
    _append_molmo_audit(record)
    print(
        "[capx Molmo record/replay] "
        f"request={request_sha256} response={record['response_sha256']} "
        f"source={source}",
        flush=True,
    )
    return content


def _parse_points(text: str) -> tuple[list[tuple[float, float]], float]:
    """Parse point coordinates from model text output.

    Supports multiple formats:
    - Molmo2: <points coords="obj_idx x y ...">label</points> (normalized 0-1000)
    - Molmo1: <point x="X" y="Y"> (normalized 0-100)
    - Legacy: <points x1=".." y1=".."> (normalized 0-100)
    - Fallback: plain "x, y" pairs (normalized 0-100)

    Args:
        text: Generated text potentially containing point tags.

    Returns:
        Tuple of (points, norm_scale) where points is a list of (x, y) tuples
        and norm_scale is the normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    # 1) Molmo2 format: <points coords="type obj_idx x y ...">label</points>
    # Format: type indicator, then triplets of (obj_idx, x, y) where x,y are in [0, 1000]
    coords_match = re.search(r'<points\s+coords\s*=\s*["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
    if coords_match:
        nums = [float(n) for n in coords_match.group(1).split()]
        points: list[tuple[float, float]] = []
        # Skip first number (type indicator), then parse triplets: (obj_idx, x, y)
        i = 1
        while i + 2 <= len(nums):
            x, y = nums[i + 1], nums[i + 2]
            points.append((x, y))
            i += 3
        return points, 1000.0  # Molmo2 uses 0-1000 normalization

    # 2) Parse one or more <point x=".." y=".."> tags (Molmo1 format)
    point_tags = re.findall(r"<point\b[^>]*>", text, flags=re.IGNORECASE)
    points = []
    for tag in point_tags:
        mx = re.search(r"\bx\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, flags=re.IGNORECASE)
        my = re.search(r"\by\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, flags=re.IGNORECASE)
        if mx and my:
            points.append((float(mx.group(1)), float(my.group(1))))
    if points:
        return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0

    # 3) Legacy <points x1=".." y1=".." ...> format
    tag_match = re.search(r"<points\b[^>]*>", text, flags=re.IGNORECASE)
    if tag_match:
        source = tag_match.group(0)
        xs = {
            int(i): float(v)
            for i, v in re.findall(r"x(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source)
        }
        ys = {
            int(i): float(v)
            for i, v in re.findall(r"y(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source)
        }
        idxs = sorted(set(xs) & set(ys))
        points = [(xs[i], ys[i]) for i in idxs]
        if points:
            return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0

    # 4) Fallback: parse plain "x, y" pairs anywhere in text
    pairs = re.findall(r"([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)", text)
    points = [(float(x), float(y)) for x, y in pairs]
    return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0


def _overlay_and_save(
    image: Image.Image, points: list[tuple[float, float]], save_path: str, *, norm_scale: float = 100.0
) -> None:
    """Overlay points on image and save figure.

    Args:
        image: PIL image the points refer to.
        points: Normalized coordinates ordered top-down, left-to-right.
        save_path: Path to save the visualization.
        norm_scale: Normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    if not points:
        return
    width, height = image.size
    xs = [x / norm_scale * width for x, _ in points]
    ys = [y / norm_scale * height for _, y in points]

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(xs, ys, s=60, c="red", edgecolors="white", linewidths=1.5, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x + 3, y + 3, str(i), color="yellow", fontsize=10, weight="bold", zorder=4)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def convert_to_pixel_coordinates(
    points: list[tuple[float, float]], image: PIL.Image.Image, norm_scale: float = 100.0
) -> list[tuple[int, int]]:
    """Convert normalized point coordinates to pixel coordinates.
    
    Args:
        points: List of (x, y) tuples in normalized coordinates.
        image: PIL image to get dimensions from.
        norm_scale: Normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    width, height = image.size
    xs = [int(x / norm_scale * width) for x, _ in points]
    ys = [int(y / norm_scale * height) for _, y in points]
    return list(zip(xs, ys, strict=True))


def _image_to_data_url(image: PIL.Image.Image) -> str:
    """Encode a PIL image as a PNG data URL suitable for OpenAI-compatible APIs."""
    b64 = base64.b64encode(_png_bytes(image)).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def init_molmo(
    # model_name: str = "allenai/moldmo-2-8b:free", # OpenRouter
    # model_name: str = "allenai/Molmo2-O-7B",
    model_name: str = "allenai/Molmo2-8B",
    base_url: str | None = None,
    api_key: str | None = None,
) -> Callable[[PIL.Image.Image, list[str] | None], dict[str, tuple[int | None, int | None]]]:
    """Initialize a detector that queries a vLLM OpenAI-compatible server.

    Supports optional client-side record/replay (PyRoki-shaped):

    - ``CAPX_MOLMO_AUDIT_FILE``: append-only JSONL audit
    - ``CAPX_MOLMO_RESPONSE_REPLAY_FILE``: frozen ``capx_molmo_response_replay_v1``
    - ``CAPX_MOLMO_REQUIRE_ALL_REPLAY=1``: fail closed on miss

    Args:
        model_name: The served model name (as registered in vLLM).
        base_url: Base URL of the OpenAI-compatible endpoint (no trailing slash beyond /v1).
        api_key: Optional API key; omit if the server does not require auth.

    Returns:
        A callable that takes an image and a list of object strings, and returns a
        mapping from object name to a pixel coordinate tuple (x, y). If a point
        could not be parsed, the value is (None, None).
    """
    resolved_base = (
        base_url
        if base_url is not None
        else os.environ.get("MOLMO_SERVICE_URL", SERVICE_URL)
    )
    # chat_url = f"{resolved_base.rstrip('/')}/v1/chat/completions" # SGLANG
    chat_url = f"{resolved_base.rstrip('/')}/chat/completions"  # vLLM
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def det_fn(
        image: PIL.Image.Image, objects: list[str] | None = None
    ) -> dict[str, tuple[int | None, int | None]]:
        """
        Args:
            image: PIL.Image.Image: The RGB image to process.
            objects: list[str]: The list of object queries to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """

        if not objects:
            return {}

        png = _png_bytes(image)
        image_sha = hashlib.sha256(png).hexdigest()
        img_url = f"data:image/png;base64,{base64.b64encode(png).decode('utf-8')}"
        all_points: dict[str, tuple[int | None, int | None]] = {}
        for obj in objects:
            prompt = f"Point at {obj}"
            max_tokens = 1024
            temperature = 0.0
            request = _molmo_request_envelope(
                model_name=model_name,
                prompt=prompt,
                image_png_sha256=image_sha,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_url}},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["<|endoftext|>"],
            }

            def _live_call() -> str:
                max_retries, backoff = 3, 1.0
                generated = ""
                for attempt in range(max_retries):
                    try:
                        resp = session.post(
                            chat_url, json=payload, headers=headers, timeout=120
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        generated = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        break
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"Request failed for '{obj}' "
                            f"(attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        if attempt < max_retries - 1:
                            time.sleep(backoff * (2**attempt))
                        else:
                            generated = ""
                return generated

            generated_text = _resolve_molmo_content(
                request=request, live_call=_live_call
            )
            print(f"Generated text for '{obj}': {generated_text}")

            points, norm_scale = _parse_points(generated_text or "")
            if points:
                abs_coords = convert_to_pixel_coordinates(points, image, norm_scale)[0]
            else:
                abs_coords = (None, None)
                print("No points parsed from model output; skipping visualization.")
            all_points[obj] = abs_coords

        return all_points

    # register_object_detector(det_fn)
    return det_fn
