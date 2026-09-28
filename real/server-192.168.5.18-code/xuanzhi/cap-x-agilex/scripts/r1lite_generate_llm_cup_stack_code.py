#!/usr/bin/env python3
"""Generate hash-bound executable code for a validated LLM cup-stack plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.cup_stack_llm import (  # noqa: E402
    execute_generated_code_dry_run,
    expected_operations,
    validate_generated_code,
)
from capx.llm.client import ModelQueryArgs, query_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-code", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8110/chat/completions",
    )
    return parser.parse_args()


def render_program(operations: list[dict[str, object]]) -> str:
    lines = []
    for operation in operations:
        kwargs = ", ".join(
            f"{key}={value!r}"
            for key, value in operation.items()
            if key != "call"
        )
        lines.append(f"{operation['call']}({kwargs})")
    return "\n".join(lines)


def strip_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("incomplete LLM code fence")
    return "\n".join(lines[1:-1]).strip()


def main() -> None:
    args = parse_args()
    plan_bytes = args.plan.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    expected = expected_operations(plan, plan_sha256=plan_sha256)
    reference = render_program(expected)
    api_key = args.api_key_file.read_text().strip()
    if not api_key:
        raise RuntimeError("LLM API key file is empty")
    prompt = f"""
Generate the complete Python control program for the validated R1 Lite
three-cup plan. The only available functions are:
bind_cup_plan, open_gripper, execute_cup_stage, close_gripper, verify_grasp,
verify_cup_stack, and return_home.

Only literal function calls are allowed. No imports, variables, loops,
conditionals, attributes, comments, Markdown, or extra statements.
The exact hash-bound safe program is:

{reference}

Plan SHA-256: {plan_sha256}
Return Python code only.
""".strip()
    response = query_model(
        ModelQueryArgs(
            model=args.model,
            server_url=args.server_url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=4096,
            reasoning_effort="medium",
        ),
        [{"role": "user", "content": prompt}],
    )
    raw_content = str(response["content"])
    code = strip_code_fence(raw_content)
    validate_generated_code(code, expected=expected)
    trace = execute_generated_code_dry_run(code, expected=expected)
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != plan_sha256:
        raise RuntimeError("validated cup-stack plan changed during generation")
    args.output_code.parent.mkdir(parents=True, exist_ok=True)
    args.output_code.write_text(code + "\n")
    code_sha256 = hashlib.sha256(args.output_code.read_bytes()).hexdigest()
    audit = {
        "schema": "capx.r1lite.three_cup_stack.llm_code_audit.v1",
        "model": args.model,
        "generated_at_unix": time.time(),
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": plan_sha256,
        "generated_code_path": str(args.output_code.resolve()),
        "generated_code_sha256": code_sha256,
        "ast_whitelist_passed": True,
        "restricted_dry_run_passed": True,
        "dry_run_trace": trace,
        "raw_llm_response": raw_content,
        "robot_motion_commands_sent": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
