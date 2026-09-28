#!/usr/bin/env python3
"""Generate and validate a constrained task program for an LLM-created plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.generated_policy import (  # noqa: E402
    execute_dry_run,
    expected_operations,
    validate_program,
)
from capx.llm.client import ModelQueryArgs, query_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-code", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8110/chat/completions"
    )
    parser.add_argument("--api-key-file", type=Path, required=True)
    return parser.parse_args()


def strip_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("incomplete Markdown code fence")
    return "\n".join(lines[1:-1]).strip()


def render_reference_program(operations: list[dict[str, object]]) -> str:
    lines = []
    for operation in operations:
        name = str(operation["call"])
        kwargs = [
            f"{key}={value!r}"
            for key, value in operation.items()
            if key != "call"
        ]
        lines.append(f"{name}({', '.join(kwargs)})")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    plan_bytes = args.plan.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    expected = expected_operations(plan, plan_sha256=plan_sha256)
    reference_program = render_reference_program(expected)
    key = args.api_key_file.read_text().strip()
    if not key:
        raise RuntimeError("API key file is empty")

    prompt = f"""
Generate the complete task-level Python program for one R1 Lite pick-and-place
run using the already validated immutable plan below.

The only available functions are:
- bind_plan(sha256=<literal string>)
- execute_frame(frame_id=<literal string>)
- close_gripper()
- open_gripper()
- return_to_f1()

The runtime behind execute_frame owns all feedback reads, bounded Cartesian
corrections, online IK/FK checks, joint-arrival checks and motion safety gates.
The grasp frame is forced to strict top-down by that runtime. You must express
the complete frame/action orchestration and may not bypass these functions.

No imports, variables, loops, conditionals, attributes, comments, Markdown or
extra statements are allowed. Return Python code only. The exact safe program
required by this plan is:

{reference_program}

Immutable plan SHA-256: {plan_sha256}
""".strip()
    response = query_model(
        ModelQueryArgs(
            model=args.model,
            server_url=args.server_url,
            api_key=key,
            temperature=0.0,
            max_tokens=4096,
            reasoning_effort="medium",
        ),
        [{"role": "user", "content": prompt}],
    )
    raw_content = str(response["content"])
    code = strip_code_fence(raw_content)
    validate_program(code, expected=expected)
    dry_run_trace = execute_dry_run(code, expected=expected)
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != plan_sha256:
        raise RuntimeError("validated plan changed during code generation")

    args.output_code.parent.mkdir(parents=True, exist_ok=True)
    args.output_code.write_text(code + "\n")
    generated_code_sha256 = hashlib.sha256(
        args.output_code.read_bytes()
    ).hexdigest()
    audit = {
        "pipeline": 4,
        "stage": "llm_task_code_generation",
        "model": args.model,
        "generated_at_unix": time.time(),
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": plan_sha256,
        "generated_code_path": str(args.output_code.resolve()),
        "generated_code_sha256": generated_code_sha256,
        "ast_whitelist_passed": True,
        "restricted_dry_run_passed": True,
        "dry_run_trace": dry_run_trace,
        "robot_motion_commands_sent": False,
        "raw_response": raw_content,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
