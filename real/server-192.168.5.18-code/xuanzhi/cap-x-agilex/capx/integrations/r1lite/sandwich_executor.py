"""Deterministic dispatcher and grasp gates for pipeline nine."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from capx.integrations.r1lite.sandwich_plan import (
    validate_policy,
    validate_sandwich_plan,
)


class SequentialRuntime(Protocol):
    trace: list[dict[str, Any]]
    controller: Any

    def bind_plan(self, *, sha256: str) -> None: ...

    def open_gripper(self, *, side: str) -> None: ...

    def close_gripper(self, *, side: str) -> None: ...

    def execute_single_arm_stage(self, *, side: str, stage_id: str) -> None: ...


class SandwichExecutionRuntime:
    """Wrap the shared executor with a mandatory post-lift retention gate."""

    def __init__(self, delegate: SequentialRuntime, plan: Mapping[str, Any]) -> None:
        validate_sandwich_plan(plan)
        self.delegate = delegate
        self.plan = dict(plan)
        self.trace = delegate.trace

    def bind_plan(self, *, sha256: str) -> None:
        self.delegate.bind_plan(sha256=sha256)

    def open_gripper(self, *, side: str) -> None:
        self.delegate.open_gripper(side=side)

    def close_gripper(self, *, side: str) -> None:
        self.delegate.close_gripper(side=side)

    def execute_single_arm_stage(self, *, side: str, stage_id: str) -> None:
        self.delegate.execute_single_arm_stage(side=side, stage_id=stage_id)
        if stage_id != "lift":
            return
        policy = self.plan["safety"]["grasp_retention_policy"]
        feedback = float(self.delegate.controller.get_gripper_feedback(side))
        threshold = float(policy["empty_close_feedback"]) + float(
            policy["minimum_gap_above_empty_close"]
        )
        passed = feedback >= threshold
        self.trace.append(
            {
                "event": "sandwich_grasp_retention_gate",
                "side": side,
                "stage_id": stage_id,
                "gripper_feedback": feedback,
                "required_minimum_feedback": threshold,
                "passed": passed,
            }
        )
        if not passed:
            self.delegate.controller.hold()
            raise RuntimeError(
                f"{side} gripper retention gate failed: "
                f"{feedback:.3f} < {threshold:.3f}"
            )


def dispatch_policy(
    policy: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    runtime: SandwichExecutionRuntime,
) -> list[dict[str, Any]]:
    """Dispatch only the exact deterministic plan-bound operation list."""
    validate_policy(policy, plan=plan, plan_sha256=plan_sha256)
    trace: list[dict[str, Any]] = []
    allowed = {
        "bind_plan",
        "open_gripper",
        "close_gripper",
        "execute_single_arm_stage",
    }
    for operation in policy["operations"]:
        name = str(operation["call"])
        if name not in allowed:
            raise ValueError(f"pipeline-nine policy call {name!r} is forbidden")
        method = getattr(runtime, name)
        kwargs = {
            key: value for key, value in operation.items() if key != "call"
        }
        method(**kwargs)
        trace.append(dict(operation))
    return trace
