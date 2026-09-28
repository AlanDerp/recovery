"""Agilex multimodel ensemble — mirrors capx.llm.client.query_model_ensemble.

Two models (gpt-5.5 via UseZyla + qwen via OpenRouter), parallel candidates,
then synthesis with gpt-5.5 (same pattern as original Cap-X).
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import time
from typing import TYPE_CHECKING, Any

import requests

from capx.llm.client import (
    ModelQueryArgs,
    OPENROUTER_SERVER_URL,
    query_model,
)

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

# Two temps per model (low + high diversity).
AGILEX_ENSEMBLE_CONFIGS = [
    ("gpt-5.5", [0.2, 0.7]),
    ("qwen/qwen3.7-max", [0.2, 0.7]),
]


def _resolve_secondary_api_key(config: dict[str, Any]) -> str | None:
    env_name = config.get("agilex_dual_model_secondary_api_key_env", "OPENROUTER_API_KEY")
    api_key = os.getenv(env_name, "").strip()
    if api_key:
        return api_key
    key_file = config.get("agilex_dual_model_secondary_api_key_file", ".openrouterkey")
    if os.path.isfile(key_file):
        with open(key_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    return None


def _response_indicates_unsupported_images(response: requests.Response) -> bool:
    if response.status_code != 404:
        return False
    try:
        body = response.json()
    except Exception:
        return False
    detail = body.get("detail", body)
    message = json.dumps(detail, ensure_ascii=False).lower()
    return "image input" in message or (
        "image" in message and ("not support" in message or "no endpoints found" in message)
    )


def _should_fail_fast(response: requests.Response) -> bool:
    if _response_indicates_unsupported_images(response):
        return False
    if response.status_code not in {400, 403, 404, 503}:
        return False
    try:
        body = response.json()
    except Exception:
        body = {}
    detail = body.get("detail", body)
    message = json.dumps(detail, ensure_ascii=False).lower()
    unsupported_markers = (
        "不支持",
        "not support",
        "unsupported",
        "invalid model",
        "model_not_found",
        "no endpoints found",
    )
    return any(marker in message for marker in unsupported_markers)


def _prompt_has_images(prompt: list[dict]) -> bool:
    for msg in prompt:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                return True
    return False


def _strip_images_from_prompt(prompt: list[dict]) -> list[dict]:
    stripped: list[dict] = []
    for msg in prompt:
        msg_copy = copy.deepcopy(msg)
        content = msg_copy.get("content")
        if isinstance(content, list):
            text_parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            image_count = sum(
                1
                for item in content
                if isinstance(item, dict) and item.get("type") == "image_url"
            )
            note = ""
            if image_count:
                note = (
                    f"\n[{image_count} camera frame(s) omitted for this text-only model; "
                    "use the execution logs and code context above.]"
                )
            msg_copy["content"] = "".join(text_parts) + note
        stripped.append(msg_copy)
    return stripped


def _route_model(
    model: str,
    config: dict[str, Any],
    args: "LaunchArgs | ModelQueryArgs",
) -> tuple[str, str | None]:
    primary = config.get("agilex_dual_model_primary", args.model)
    if model == primary:
        return args.server_url, args.api_key
    secondary_url = config.get(
        "agilex_dual_model_secondary_server_url",
        OPENROUTER_SERVER_URL,
    )
    return secondary_url, _resolve_secondary_api_key(config)


def query_model_agilex(args: ModelQueryArgs, prompt: list[dict]) -> dict[str, Any]:
    """Query OpenRouter proxy (8110). Key is held by the proxy process."""
    prompt_to_send = copy.deepcopy(prompt)
    images_stripped = False
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": prompt_to_send,
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    start_time = time.time()
    response = requests.post(
        args.server_url, headers=headers, data=json.dumps(payload), timeout=200
    )
    retry = 1
    max_retries = int(os.environ.get("LLM_QUERY_MAX_RETRIES", "8"))
    while response.status_code in [404, 500, 502, 503, 504] and retry <= max_retries:
        if (
            _response_indicates_unsupported_images(response)
            and not images_stripped
            and _prompt_has_images(prompt_to_send)
        ):
            print(
                f"[agilex_llm] {args.model} returned 404 for image input; "
                "retrying once without camera frames."
            )
            prompt_to_send = _strip_images_from_prompt(prompt_to_send)
            images_stripped = True
            payload["messages"] = prompt_to_send
            response = requests.post(
                args.server_url, headers=headers, data=json.dumps(payload), timeout=200
            )
            if response.status_code == 200:
                break
        if _should_fail_fast(response):
            response.raise_for_status()
        sleep_time = min(30.0 * retry, 120.0) + random.uniform(0, 5)
        print(
            f"Retry {retry}/{max_retries}. Model query failed with status code "
            f"{response.status_code}. Error: {response.text}. Retrying in {sleep_time:.1f} seconds..."
        )
        time.sleep(sleep_time)
        response = requests.post(
            args.server_url, headers=headers, data=json.dumps(payload), timeout=200
        )
        retry += 1
    if response.status_code in [404, 500, 502, 503, 504]:
        print(
            f"Model query failed after {max_retries} retries with status code "
            f"{response.status_code}. Error: {response.text}"
        )

    print(f"Time taken to query model {args.model}: {time.time() - start_time:.2f} seconds")
    response.raise_for_status()
    body = response.json()
    message = body["choices"][0]["message"]
    return {
        "content": message["content"],
        "reasoning": message.get("reasoning"),
    }


def query_agilex_dual_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    config: dict[str, Any],
    is_multiturn: bool = False,
) -> dict[str, Any]:
    """Query gpt-5.5 + qwen in parallel, synthesize with qwen/qwen3.7-max."""
    synthesis_model = config.get("agilex_dual_model_synthesis", "qwen/qwen3.7-max")
    print(f"[Multimodel Ensemble] Synthesizing with {synthesis_model}")
    temps = config.get("agilex_dual_model_temps", [0.2, 0.7])
    primary = config.get("agilex_dual_model_primary", args.model)
    secondary = config.get("agilex_dual_model_secondary", "qwen/qwen3.7-max")
    ensemble_configs = [(primary, temps), (secondary, temps)]

    def query_single(model: str, temp: float) -> dict:
        server_url, api_key = _route_model(model, config, args)
        query_args = ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            primary_model = config.get("agilex_dual_model_primary", args.model)
            if model == primary_model:
                result = query_model(query_args, copy.deepcopy(prompt))
            else:
                result = query_model_agilex(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Multimodel Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    tasks = [(m, t) for m, temps in ensemble_configs for t in temps]
    responses: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(tasks), 1)) as executor:
        futures = {executor.submit(query_single, m, t): (m, t) for m, t in tasks}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp["ok"]:
                print(
                    f"[Multimodel Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}"
                )

    successful = [r for r in responses if r["ok"]]
    if not successful:
        print("\n=== All ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All ensemble queries failed")

    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    regenerate_count = sum(
        1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"]
    )
    finish_count = sum(
        1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"]
    )

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {"role": "system", "content": synthesis_system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": synthesis_user_prompt}],
        },
    ]

    synth_url, synth_key = _route_model(synthesis_model, config, args)
    synth_args = ModelQueryArgs(
        model=synthesis_model,
        server_url=synth_url,
        api_key=synth_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    primary_model = config.get("agilex_dual_model_primary", args.model)
    if synthesis_model == primary_model:
        final = query_model(synth_args, synthesis_prompt)
    else:
        final = query_model_agilex(synth_args, synthesis_prompt)

    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {synthesis_model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }
