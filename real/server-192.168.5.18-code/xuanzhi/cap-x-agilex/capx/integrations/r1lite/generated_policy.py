"""Validation and dry-run support for constrained LLM task policies."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any


ALLOWED_CALLS = {
    "bind_plan",
    "execute_frame",
    "close_gripper",
    "open_gripper",
    "return_to_f1",
}


def expected_operations(
    plan: dict[str, Any], *, plan_sha256: str
) -> list[dict[str, Any]]:
    waypoints = plan.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("plan must contain a non-empty waypoints list")
    operations: list[dict[str, Any]] = [
        {"call": "bind_plan", "sha256": plan_sha256}
    ]
    grasp_count = 0
    place_count = 0
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, dict):
            raise ValueError(f"waypoint {index} must be an object")
        frame_id = str(waypoint.get("id", "")).strip()
        if not frame_id:
            raise ValueError(f"waypoint {index} has no id")
        role = str(waypoint.get("semantic_role", "")).strip()
        operations.append({"call": "execute_frame", "frame_id": frame_id})
        if role == "grasp":
            grasp_count += 1
            operations.append({"call": "close_gripper"})
        if role == "above_box":
            place_count += 1
            operations.append({"call": "open_gripper"})
            operations.append({"call": "return_to_f1"})
    if grasp_count != 1 or place_count != 1:
        raise ValueError(
            "single-round generated policy requires exactly one grasp and "
            f"one above_box; got grasp={grasp_count}, above_box={place_count}"
        )
    return operations


def _literal_keywords(call: ast.Call) -> dict[str, Any]:
    if call.args:
        raise ValueError("generated policy positional arguments are forbidden")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("generated policy dictionary expansion is forbidden")
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        assert keyword.arg is not None
        if keyword.arg in values:
            raise ValueError(f"duplicate keyword {keyword.arg!r}")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return values


def validate_program(
    code: str, *, expected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != len(expected):
        raise ValueError(
            "generated policy must contain exactly the required calls: "
            f"expected {len(expected)}, got {len(tree.body)} statements"
        )
    observed: list[dict[str, Any]] = []
    for index, statement in enumerate(tree.body):
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            raise ValueError(
                f"generated policy statement {index} must be one call expression"
            )
        call = statement.value
        if not isinstance(call.func, ast.Name):
            raise ValueError("attribute and dynamic calls are forbidden")
        name = call.func.id
        if name not in ALLOWED_CALLS:
            raise ValueError(f"generated policy call {name!r} is forbidden")
        operation = {"call": name, **_literal_keywords(call)}
        observed.append(operation)
    if observed != expected:
        raise ValueError(
            "generated policy operations do not exactly match the validated "
            f"plan; expected={expected!r}, observed={observed!r}"
        )
    return observed


def execute_dry_run(
    code: str, *, expected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    validate_program(code, expected=expected)
    trace: list[dict[str, Any]] = []

    def recorder(name: str) -> Callable[..., None]:
        def record(**kwargs: Any) -> None:
            trace.append({"call": name, **kwargs})

        return record

    namespace = {name: recorder(name) for name in ALLOWED_CALLS}
    exec(  # noqa: S102 - validation admits only literal calls to known recorders.
        compile(code, "<r1lite-llm-generated-policy>", "exec"),
        {"__builtins__": {}, **namespace},
        {},
    )
    if trace != expected:
        raise RuntimeError(
            f"generated policy dry-run mismatch: expected={expected!r}, trace={trace!r}"
        )
    return trace
