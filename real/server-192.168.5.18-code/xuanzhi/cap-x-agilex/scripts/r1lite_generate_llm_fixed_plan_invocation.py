#!/usr/bin/env python3
"""Ask an LLM for a tightly constrained pipeline-2 invocation program."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.llm.client import ModelQueryArgs, query_model


ALLOWED_FUNCTION = "execute_pipeline_2_single_round"


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


def validate_program(
    code: str,
    *,
    expected_plan_path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("LLM program must contain exactly one call expression")
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        raise ValueError("LLM program must contain exactly one function call")
    if not isinstance(call.func, ast.Name) or call.func.id != ALLOWED_FUNCTION:
        raise ValueError(f"only {ALLOWED_FUNCTION} may be called")
    if call.args:
        raise ValueError("positional arguments are forbidden")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("dictionary expansion is forbidden")

    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg in values:
            raise ValueError(f"duplicate keyword: {keyword.arg}")
        values[str(keyword.arg)] = ast.literal_eval(keyword.value)
    required = {"plan_path", "expected_sha256", "round_count"}
    if set(values) != required:
        raise ValueError(f"call keywords must be exactly {sorted(required)}")
    if values["plan_path"] != expected_plan_path:
        raise ValueError("LLM changed the fixed plan path")
    if values["expected_sha256"] != expected_sha256:
        raise ValueError("LLM changed the fixed plan SHA-256")
    if values["round_count"] != 1:
        raise ValueError("LLM changed the required single-round count")
    return values


def execute_validated_program(
    code: str,
    *,
    expected_plan_path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    invocation: dict[str, Any] = {}

    def execute_pipeline_2_single_round(
        *,
        plan_path: str,
        expected_sha256: str,
        round_count: int,
    ) -> None:
        invocation.update(
            {
                "plan_path": plan_path,
                "expected_sha256": expected_sha256,
                "round_count": round_count,
            }
        )

    exec(  # noqa: S102 - AST validation above permits one known call only.
        compile(code, "<llm-fixed-plan-invocation>", "exec"),
        {"__builtins__": {}, ALLOWED_FUNCTION: execute_pipeline_2_single_round},
        {},
    )
    if invocation != {
        "plan_path": expected_plan_path,
        "expected_sha256": expected_sha256,
        "round_count": 1,
    }:
        raise RuntimeError("validated LLM program did not produce the expected invocation")
    return invocation


def main() -> None:
    args = parse_args()
    plan_path = args.plan.resolve()
    plan_bytes = plan_path.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    key = args.api_key_file.read_text().strip()
    if not key:
        raise RuntimeError("API key file is empty")

    expected_code = (
        f"{ALLOWED_FUNCTION}(\n"
        f"    plan_path={str(plan_path)!r},\n"
        f"    expected_sha256={plan_sha256!r},\n"
        "    round_count=1,\n"
        ")"
    )
    prompt = f"""
Generate a Python program that requests one execution of an already validated
R1 Lite pipeline-2 plan. The F0-F6 plan and execution policy are immutable.
Return Python code only: no Markdown, imports, variables, comments, extra calls,
or additional statements. Your program must be exactly one call with keyword
arguments equivalent to:

{expected_code}
""".strip()
    response = query_model(
        ModelQueryArgs(
            model=args.model,
            server_url=args.server_url,
            api_key=key,
            temperature=0.0,
            max_tokens=1024,
            reasoning_effort="medium",
        ),
        [{"role": "user", "content": prompt}],
    )
    raw_content = str(response["content"])
    code = strip_code_fence(raw_content)
    validate_program(
        code,
        expected_plan_path=str(plan_path),
        expected_sha256=plan_sha256,
    )
    invocation = execute_validated_program(
        code,
        expected_plan_path=str(plan_path),
        expected_sha256=plan_sha256,
    )
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha256:
        raise RuntimeError("fixed plan changed during LLM invocation generation")

    args.output_code.parent.mkdir(parents=True, exist_ok=True)
    args.output_code.write_text(code + "\n")
    audit = {
        "pipeline": 5,
        "model": args.model,
        "code_source": "llm",
        "oracle_used": False,
        "generated_at_unix": time.time(),
        "fixed_plan_path": str(plan_path),
        "fixed_plan_sha256": plan_sha256,
        "validated_invocation": invocation,
        "llm_code_executed_in_restricted_namespace": True,
        "robot_motion_commands_sent": False,
        "raw_response": raw_content,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
