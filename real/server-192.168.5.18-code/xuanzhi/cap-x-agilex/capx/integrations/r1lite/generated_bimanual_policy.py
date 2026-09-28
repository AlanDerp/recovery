"""AST validation and dry-run for seventh-pipeline LLM programs."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any, Protocol

from capx.integrations.r1lite.towel_plan import (
    SEQUENTIAL_SINGLE_ARM_SCHEMA,
    STAGE_ORDER,
    validate_towel_fold_plan,
)


ALLOWED_CALLS = {
    "bind_plan",
    "open_both_grippers",
    "execute_bimanual_stage",
    "close_both_grippers",
    "open_gripper",
    "execute_single_arm_stage",
    "close_gripper",
}


class BimanualPolicyRuntime(Protocol):
    """Narrow runtime surface available to an approved LLM program."""

    def bind_plan(self, *, sha256: str) -> None: ...

    def open_both_grippers(self) -> None: ...

    def execute_bimanual_stage(self, *, stage_id: str) -> None: ...

    def close_both_grippers(self) -> None: ...

    def open_gripper(self, *, side: str) -> None: ...

    def execute_single_arm_stage(self, *, side: str, stage_id: str) -> None: ...

    def close_gripper(self, *, side: str) -> None: ...


def expected_operations(
    plan: dict[str, Any],
    *,
    plan_sha256: str,
) -> list[dict[str, Any]]:
    validate_towel_fold_plan(plan)
    if plan.get("schema") == SEQUENTIAL_SINGLE_ARM_SCHEMA:
        side = str(plan["phase_side"])
        operations = [
            {"call": "bind_plan", "sha256": plan_sha256},
            {"call": "open_gripper", "side": side},
        ]
        for stage in plan["stages"]:
            stage_id = str(stage["id"])
            operations.append(
                {
                    "call": "execute_single_arm_stage",
                    "side": side,
                    "stage_id": stage_id,
                }
            )
            if stage_id == "grasp":
                operations.append({"call": "close_gripper", "side": side})
            if stage_id == "place_center":
                operations.append({"call": "open_gripper", "side": side})
        return operations
    operations: list[dict[str, Any]] = [
        {"call": "bind_plan", "sha256": plan_sha256},
        {"call": "open_both_grippers"},
    ]
    for stage_id in STAGE_ORDER:
        operations.append(
            {"call": "execute_bimanual_stage", "stage_id": stage_id}
        )
        if stage_id == "grasp":
            operations.append({"call": "close_both_grippers"})
        if stage_id == "fold_center":
            operations.append({"call": "open_both_grippers"})
    return operations


def _literal_keywords(call: ast.Call) -> dict[str, Any]:
    if call.args:
        raise ValueError("generated bimanual policy positional arguments are forbidden")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("generated bimanual policy dictionary expansion is forbidden")
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        assert keyword.arg is not None
        if keyword.arg in values:
            raise ValueError(f"duplicate keyword {keyword.arg!r}")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return values


def validate_program(
    code: str,
    *,
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != len(expected):
        raise ValueError(
            "generated bimanual program must contain exactly the required calls"
        )
    observed = []
    for index, statement in enumerate(tree.body):
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value,
            ast.Call,
        ):
            raise ValueError(
                f"generated bimanual statement {index} must be one call expression"
            )
        call = statement.value
        if not isinstance(call.func, ast.Name):
            raise ValueError("attribute and dynamic calls are forbidden")
        name = call.func.id
        if name not in ALLOWED_CALLS:
            raise ValueError(f"generated bimanual call {name!r} is forbidden")
        observed.append({"call": name, **_literal_keywords(call)})
    if observed != expected:
        raise ValueError(
            "generated bimanual operations do not exactly match the immutable plan"
        )
    return observed


def execute_dry_run(
    code: str,
    *,
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validate_program(code, expected=expected)
    trace: list[dict[str, Any]] = []

    def recorder(name: str) -> Callable[..., None]:
        def record(**kwargs: Any) -> None:
            trace.append({"call": name, **kwargs})

        return record

    namespace = {name: recorder(name) for name in ALLOWED_CALLS}
    exec(  # noqa: S102 - AST validation permits only literal safe calls.
        compile(code, "<r1lite-seventh-generated-policy>", "exec"),
        {"__builtins__": {}, **namespace},
        {},
    )
    if trace != expected:
        raise RuntimeError("generated bimanual dry-run trace mismatch")
    return trace


def dispatch_program(
    code: str,
    *,
    expected: list[dict[str, Any]],
    runtime: BimanualPolicyRuntime,
) -> list[dict[str, Any]]:
    """Dispatch an approved program without evaluating generated Python."""
    operations = validate_program(code, expected=expected)
    trace: list[dict[str, Any]] = []
    for operation in operations:
        name = str(operation["call"])
        kwargs = {
            key: value for key, value in operation.items() if key != "call"
        }
        method = getattr(runtime, name, None)
        if method is None or not callable(method):
            raise TypeError(f"runtime does not implement approved call {name!r}")
        method(**kwargs)
        trace.append(dict(operation))
    return trace
