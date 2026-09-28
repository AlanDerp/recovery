#!/usr/bin/env python3
"""Generate the complete constrained task program for pipeline seven."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.generated_bimanual_policy import (  # noqa: E402
    execute_dry_run,
    expected_operations,
    validate_program,
)
from capx.integrations.r1lite.towel_plan import (  # noqa: E402
    SEQUENTIAL_SINGLE_ARM_SCHEMA,
)
from capx.llm.client import ModelQueryArgs, query_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-code", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8110/chat/completions",
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
    return "\n".join(
        f"{operation['call']}("
        + ", ".join(
            f"{key}={value!r}"
            for key, value in operation.items()
            if key != "call"
        )
        + ")"
        for operation in operations
    )


def main() -> None:
    args = parse_args()
    plan_bytes = args.plan.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    expected = expected_operations(plan, plan_sha256=plan_sha256)
    reference = render_reference_program(expected)
    if plan.get("schema") == SEQUENTIAL_SINGLE_ARM_SCHEMA:
        available_functions = """- bind_plan(sha256=<literal string>)
- open_gripper(side=<literal string>)
- execute_single_arm_stage(side=<literal string>, stage_id=<literal string>)
- close_gripper(side=<literal string>)"""
        task_description = "one certified sequential single-arm towel-fold phase"
    else:
        available_functions = """- bind_plan(sha256=<literal string>)
- open_both_grippers()
- execute_bimanual_stage(stage_id=<literal string>)
- close_both_grippers()"""
        task_description = "one R1 Lite dual-arm towel fold run"
    api_key = args.api_key_file.read_text().strip()
    if not api_key:
        raise RuntimeError("API key file is empty")
    prompt = f"""
Generate the complete task-level Python program for {task_description}.
The validated plan is immutable.

The only available functions are:
{available_functions}

The runtime owns head-camera perception, top-down orientation, dual-arm IK/FK,
bounded interpolation, collision gates, synchronized publishing, feedback
barriers and stop behavior. Do not bypass it.

No imports, variables, loops, conditionals, attributes, comments, Markdown or
extra statements are allowed. Return Python code only. The exact complete
program required by this immutable plan is:

{reference}

Immutable plan SHA-256: {plan_sha256}
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
    validate_program(code, expected=expected)
    dry_run_trace = execute_dry_run(code, expected=expected)
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != plan_sha256:
        raise RuntimeError("validated towel plan changed during code generation")
    args.output_code.parent.mkdir(parents=True, exist_ok=True)
    args.output_code.write_text(code + "\n")
    code_sha256 = hashlib.sha256(args.output_code.read_bytes()).hexdigest()
    audit = {
        "pipeline": 7,
        "stage": "llm_complete_bimanual_task_code_generation",
        "model": args.model,
        "generated_at_unix": time.time(),
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": plan_sha256,
        "generated_code_path": str(args.output_code.resolve()),
        "generated_code_sha256": code_sha256,
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
