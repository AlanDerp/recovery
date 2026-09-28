"""Scope-correct, compact closed-GT takeover for multiturn regeneration."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capx.verification import closed_gt
from capx.verification.closed_gt_takeover import (
    SemanticEventBatch,
    _action_visible_specs,
    _entity_names,
    _interval_grasp_outcomes,
    _non_text_structure,
    _public_label,
    _public_value,
    _temporal_specs,
)

SCHEMA_VERSION = "capx-ideal-tk-v3.3-public-lossless"
TRACE_DIR = "ideal_tk_traces"
PUBLIC_WORD_AUDIT_TARGET = 500
MAX_LEDGER_ENTRIES = 2

_DERIVED_STATE_HEADER = "Canonical derived state (authoritative composite status):"
_PRIMITIVE_STATE_HEADER = "Canonical primitive factual basis:"
_TRANSITION_HEADER = "Verified endpoint transition from the previous endpoint:"
_EVENT_HEADER = "Verified ordered factual events during the latest code block:"
_HISTORY_HEADER = "Earlier verified block outcomes (historical, not current state):"

_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
_FORBIDDEN_PUBLIC_TERMS = (
    "reward",
    "task completed",
    "task success",
    "next move",
    "next action",
    "repair plan",
    "oracle",
    "move_to_pose",
    "execute_trajectory",
)


@dataclass(frozen=True)
class ApiCallTrace:
    name: str
    target_ids: tuple[str, ...]
    status: str
    before: closed_gt.CompiledSnapshot
    after: closed_gt.CompiledSnapshot
    events: tuple[SemanticEventBatch, ...]


@dataclass
class _Runtime:
    scene: closed_gt.SceneSpec
    latest: closed_gt.CompiledSnapshot
    block_before: closed_gt.CompiledSnapshot
    pending: list[SemanticEventBatch] = field(default_factory=list)
    calls: list[ApiCallTrace] = field(default_factory=list)
    error: str | None = None

    def observe(self, snapshot: closed_gt.CompiledSnapshot) -> None:
        closed_gt.validate_snapshot(self.scene, snapshot)
        changes = tuple(
            (spec.key, self.latest.values[spec.key], snapshot.values[spec.key])
            for spec in _temporal_specs(self.scene)
            if self.latest.values[spec.key] != snapshot.values[spec.key]
        )
        if changes:
            self.pending.append(SemanticEventBatch(changes))
        self.latest = snapshot


_RUNTIMES: dict[tuple[str, int], _Runtime] = {}


def _cache_key(output_dir: str, trial: int) -> tuple[str, int]:
    return str(Path(output_dir).resolve()), trial


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_text_once(
    prompt: list[dict[str, Any]], *, original: str, replacement: str, carrier: str
) -> list[dict[str, Any]]:
    if not original.strip() or original == replacement:
        raise ValueError(f"Ideal TK requires a distinct non-empty {carrier}")
    updated = copy.deepcopy(prompt)
    matches = 0
    for message in updated:
        content = message.get("content")
        text_items: list[dict[str, Any]] = []
        if isinstance(content, str):
            count = content.count(original)
            if count:
                message["content"] = content.replace(original, replacement)
                matches += count
        elif isinstance(content, list):
            text_items = [
                item for item in content if isinstance(item, dict) and item.get("type") == "text"
            ]
        for item in text_items:
            text = item.get("text")
            if not isinstance(text, str):
                continue
            count = text.count(original)
            if count:
                item["text"] = text.replace(original, replacement)
                matches += count
    if matches != 1:
        raise ValueError(f"Ideal TK expected one {carrier} carrier, found {matches}")
    return updated


def render_scope_correct_execution_context(
    executed_code_blocks: list[str], console_stdout: str, console_stderr: str
) -> tuple[str, str, int]:
    """Bind native console evidence to the unmodified latest code block only."""
    if not executed_code_blocks:
        raise ValueError("Ideal TK requires at least one executed code block")
    action_code = executed_code_blocks[-1]
    sections = (
        "Latest executed code block (unmodified; all earlier code blocks omitted):",
        "```python",
        action_code,
        "```",
        "The console evidence below belongs only to that action code.",
        "Latest block stdout:",
        "```text",
        console_stdout,
        "```",
        "Latest block stderr:",
        "```text",
        console_stderr,
        "```",
    )
    return "\n".join(sections), action_code, 0


def _restore_task_contract(
    prompt: list[dict[str, Any]], *, task_description: str, execution_context: str
) -> tuple[list[dict[str, Any]], str, bool]:
    """Remove the initial visual/PT suffix while preserving the task/API contract."""
    if not task_description.strip():
        raise ValueError("Ideal TK requires the original task/API contract")
    updated = copy.deepcopy(prompt)
    matches: list[dict[str, Any]] = []
    for message in updated:
        content = message.get("content")
        if isinstance(content, str):
            if content.startswith(task_description):
                matches.append(message)
        elif isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    and item["text"].startswith(task_description)
                ):
                    matches.append(item)
    if len(matches) != 1:
        raise ValueError(f"Ideal TK expected one task/API contract carrier, found {len(matches)}")
    carrier = matches[0]
    key = "content" if "content" in carrier else "text"
    native = carrier[key]
    context_index = native.find(execution_context, len(task_description))
    if context_index >= 0:
        suffix = native[len(task_description) : context_index]
        carrier[key] = f"{task_description}\n\n{native[context_index:]}"
    else:
        suffix = native[len(task_description) :]
        carrier[key] = task_description
    removed = bool(suffix.strip())
    return updated, native, removed


def _target_ids(
    scene: closed_gt.SceneSpec, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, ...]:
    strings = [value for value in (*args, *kwargs.values()) if isinstance(value, str)]
    targets = []
    for obj in scene.objects:
        if any(obj.public_name.lower() in value.lower() for value in strings):
            targets.append(obj.entity_id)
    return tuple(targets)


def _restore_and_wrap_apis(env: Any, runtime: _Runtime) -> tuple[str, ...]:
    wrapped_names: list[str] = []
    for api in getattr(env, "_apis", {}).values():
        originals = getattr(api, "_capx_ideal_tk_originals", None)
        if originals is not None:
            for name, original in originals.items():
                setattr(api, name, original)
        originals = dict(api.functions())
        api._capx_ideal_tk_originals = originals
        for name, original in originals.items():
            if not hasattr(api, name):
                continue

            @functools.wraps(original)
            def wrapped(
                *args: Any,
                __name: str = name,
                __original: Callable[..., Any] = original,
                **kwargs: Any,
            ) -> Any:
                if runtime.error is not None:
                    return __original(*args, **kwargs)
                try:
                    before = closed_gt.extract_cube_snapshot(env)
                    runtime.observe(before)
                    event_start = len(runtime.pending)
                    targets = _target_ids(runtime.scene, args, kwargs)
                except Exception as exc:
                    runtime.error = f"{type(exc).__name__}: {exc}"
                    return __original(*args, **kwargs)
                status = "returned"
                try:
                    return __original(*args, **kwargs)
                except BaseException:
                    status = "raised"
                    raise
                finally:
                    try:
                        after = closed_gt.extract_cube_snapshot(env)
                        runtime.observe(after)
                        runtime.calls.append(
                            ApiCallTrace(
                                __name,
                                targets,
                                status,
                                before,
                                after,
                                tuple(runtime.pending[event_start:]),
                            )
                        )
                    except Exception as exc:
                        runtime.error = f"{type(exc).__name__}: {exc}"

            setattr(api, name, wrapped)
            wrapped_names.append(name)
    for api in getattr(env, "_apis", {}).values():
        for name, function in api.functions().items():
            env._exec_globals[name] = function
    return tuple(sorted(set(wrapped_names)))


def _install_observer(env: Any, runtime: _Runtime) -> bool:
    low_level = getattr(env, "low_level_env", None)
    if low_level is None or not hasattr(low_level, "_notify_post_sim_step_observer"):
        return False

    def observe_after_step() -> None:
        if runtime.error is not None:
            return
        try:
            runtime.observe(closed_gt.extract_cube_snapshot(env))
        except Exception as exc:
            runtime.error = f"{type(exc).__name__}: {exc}"

    low_level._capx_post_sim_step_observer = observe_after_step
    return True


def initialize_trial_trace(env: Any, *, output_dir: str, trial: int, task: str) -> None:
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Ideal TK has no audited adapter for {task!r}")
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    key = _cache_key(output_dir, trial)
    if key in _RUNTIMES:
        raise RuntimeError("Ideal TK private trial cache already exists")
    initial = closed_gt.extract_cube_snapshot(env)
    runtime = _Runtime(closed_gt.cube_scene_spec(), initial, initial)
    _RUNTIMES[key] = runtime
    observer_installed = _install_observer(env, runtime)
    wrapped_names = _restore_and_wrap_apis(env, runtime)
    _save_json(
        root / "state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "closed_gt_schema_version": closed_gt.SCHEMA_VERSION,
            "task": task,
            "trial": trial,
            "turns_completed": 0,
            "evidence_ledger": [],
            "initial_public_state_sha256": _sha256_json(initial.values),
            "sim_step_observer_installed": observer_installed,
            "wrapped_api_names": list(wrapped_names),
            "continuous_state_persisted": False,
        },
    )


def _state_line(scene: closed_gt.SceneSpec, spec: closed_gt.PredicateSpec, value: str) -> str:
    return (
        f"- {_public_label(scene, spec)} = {_public_enum(scene, value)} "
        f"({_public_value(scene, value)})."
    )


def _public_enum(scene: closed_gt.SceneSpec, value: str) -> str:
    """Replace private entity tokens inside finite-domain values bijectively."""
    rendered = value
    for entity in (*scene.executors, *scene.objects, scene.table):
        public_token = re.sub(r"[^A-Z]+", "_", entity.public_name.upper()).strip("_")
        rendered = rendered.replace(entity.entity_id.upper(), public_token)
    return rendered


def _enum_decoder(scene: closed_gt.SceneSpec, spec: closed_gt.PredicateSpec) -> dict[str, str]:
    decoder = {_public_enum(scene, value): value for value in spec.values}
    if len(decoder) != len(spec.values):
        raise ValueError(f"Public enum mapping is not bijective: {spec.key}")
    return decoder


def _section_lines(report: str, start: str, end: str) -> list[str]:
    lines = report.splitlines()
    if lines.count(start) != 1 or lines.count(end) != 1:
        raise ValueError(f"Canonical public report section is absent or ambiguous: {start}")
    begin = lines.index(start) + 1
    finish = lines.index(end)
    if finish <= begin:
        raise ValueError(f"Canonical public report section is empty: {start}")
    return [line for line in lines[begin:finish] if line]


def parse_canonical_current_state(scene: closed_gt.SceneSpec, report: str) -> dict[str, str]:
    """Decode the complete student-visible state without privileged context."""
    specs = closed_gt.predicate_specs(scene)
    labels = {_public_label(scene, spec): spec for spec in specs}
    if len(labels) != len(specs):
        raise ValueError("Canonical public labels are not unique")
    raw_lines = [
        *_section_lines(report, _DERIVED_STATE_HEADER, _PRIMITIVE_STATE_HEADER),
        *_section_lines(report, _PRIMITIVE_STATE_HEADER, _TRANSITION_HEADER),
    ]
    decoded: dict[str, str] = {}
    for line in raw_lines:
        if not line.startswith("- ") or " = " not in line or " (" not in line:
            raise ValueError(f"Malformed canonical state line: {line}")
        label, encoded = line[2:].split(" = ", 1)
        public_enum, rendered = encoded.split(" (", 1)
        spec = labels.get(label)
        if spec is None or spec.key in decoded:
            raise ValueError(f"Unknown or duplicated canonical state slot: {label}")
        value = _enum_decoder(scene, spec).get(public_enum)
        if value is None:
            raise ValueError(f"Canonical state value is outside its domain: {label}")
        if rendered != f"{_public_value(scene, value)}).":
            raise ValueError(f"Canonical state rendering does not match its enum: {label}")
        decoded[spec.key] = value
    expected = [spec.key for spec in specs]
    if set(decoded) != set(expected):
        missing = sorted(set(expected) - set(decoded))
        extra = sorted(set(decoded) - set(expected))
        raise ValueError(
            f"Canonical public state coverage mismatch missing={missing} extra={extra}"
        )
    return {key: decoded[key] for key in expected}


def parse_canonical_transition(
    scene: closed_gt.SceneSpec,
    report: str,
    current_values: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Recover the prior endpoint and complete transition from public text."""
    specs = closed_gt.predicate_specs(scene)
    labels = {_public_label(scene, spec): spec for spec in specs}
    lines = _section_lines(report, _TRANSITION_HEADER, _EVENT_HEADER)
    unchanged_all = "- Every canonical current-state slot was unchanged."
    unchanged_rest = "- Every canonical current-state slot not listed above was unchanged."
    markers = [index for index, line in enumerate(lines) if line in {unchanged_all, unchanged_rest}]
    if len(markers) != 1:
        raise ValueError("Canonical transition has no unique unchanged-state marker")
    marker = markers[0]
    changed_lines = lines[:marker]
    metadata_lines = lines[marker + 1 :]
    if (not changed_lines) != (lines[marker] == unchanged_all):
        raise ValueError("Canonical transition unchanged-state marker is inconsistent")

    before_values = dict(current_values)
    transition_values = {f"transition.{spec.key}": "UNCHANGED" for spec in specs}
    changed_keys: set[str] = set()
    for line in changed_lines:
        if not line.startswith("- ") or " = " not in line or " (" not in line:
            raise ValueError(f"Malformed canonical transition line: {line}")
        label, encoded = line[2:].split(" = ", 1)
        enum_path, rendered = encoded.split(" (", 1)
        if " -> " not in enum_path:
            raise ValueError(f"Malformed canonical transition enums: {line}")
        old_public, new_public = enum_path.split(" -> ", 1)
        spec = labels.get(label)
        if spec is None or spec.key in changed_keys:
            raise ValueError(f"Unknown or duplicated transition slot: {label}")
        decoder = _enum_decoder(scene, spec)
        old = decoder.get(old_public)
        new = decoder.get(new_public)
        if old is None or new is None or old == new:
            raise ValueError(f"Invalid canonical transition domain: {label}")
        expected_rendered = f"{_public_value(scene, old)} -> {_public_value(scene, new)})."
        if rendered != expected_rendered or current_values.get(spec.key) != new:
            raise ValueError(f"Canonical transition conflicts with current state: {label}")
        before_values[spec.key] = old
        transition_values[f"transition.{spec.key}"] = "CHANGED"
        changed_keys.add(spec.key)

    names = _entity_names(scene)
    expected_metadata: dict[str, tuple[str, tuple[str, ...]]] = {}
    for entity in (*scene.executors, *scene.objects):
        label = f"pose change for {names[entity.entity_id]}"
        key = f"transition.{entity.kind}.{entity.entity_id}.pose"
        expected_metadata[label] = (key, closed_gt.MOTION_VALUES)
    for executor in scene.executors:
        for obj in scene.objects:
            label = f"holding outcome for {names[executor.entity_id]} and {names[obj.entity_id]}"
            key = f"transition.executor_object.{executor.entity_id}.{obj.entity_id}.holding_outcome"
            expected_metadata[label] = (key, closed_gt.HOLDING_OUTCOME_VALUES)
    expected_metadata["aggregate relevant state change"] = (
        "transition.any_relevant_state",
        closed_gt.CHANGE_VALUES,
    )
    seen_metadata: set[str] = set()
    for line in metadata_lines:
        if not line.startswith("- ") or " = " not in line or " (" not in line:
            raise ValueError(f"Malformed canonical transition metadata: {line}")
        label, encoded = line[2:].split(" = ", 1)
        public_enum, rendered = encoded.split(" (", 1)
        entry = expected_metadata.get(label)
        if entry is None or label in seen_metadata:
            raise ValueError(f"Unknown or duplicated transition metadata: {label}")
        key, domain = entry
        value = next(
            (candidate for candidate in domain if _public_enum(scene, candidate) == public_enum),
            None,
        )
        if value is None or rendered != f"{_public_value(scene, value)}).":
            raise ValueError(f"Invalid canonical transition metadata: {label}")
        transition_values[key] = value
        seen_metadata.add(label)
    if seen_metadata != set(expected_metadata):
        raise ValueError("Canonical transition metadata coverage mismatch")
    return before_values, transition_values


def _interval_outcome_text(outcome: str) -> str:
    values = {
        "NO_EXECUTOR_CONTACT_OBSERVED": "no executor contact occurred during the block",
        "CONTACT_WITHOUT_BILATERAL_GRASP_OBSERVED": (
            "executor contact occurred, but bilateral grasp contact never occurred"
        ),
        "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST": (
            "bilateral grasp contact occurred without confirmed lift and is absent at the endpoint"
        ),
        "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED": (
            "bilateral grasp contact occurred without confirmed lift and contact is present at the endpoint"
        ),
        "LIFT_CONFIRMED_THEN_LOST": (
            "lift-confirmed bilateral grasp occurred and bilateral contact is absent at the endpoint"
        ),
        "LIFT_CONFIRMED_AND_RETAINED": (
            "lift-confirmed bilateral grasp occurred and bilateral contact, not necessarily lift, is present at the endpoint"
        ),
    }
    try:
        return values[outcome]
    except KeyError as exc:
        raise ValueError(f"Unknown interval grasp outcome: {outcome}") from exc


def _validate_public_report(scene: closed_gt.SceneSpec, report: str) -> None:
    lowered = report.lower()
    if any(term in lowered for term in _FORBIDDEN_PUBLIC_TERMS):
        raise ValueError("Ideal TK report contains prohibited content")
    private_ids = [
        entity.entity_id.lower() for entity in (*scene.executors, *scene.objects, scene.table)
    ]
    if any(entity_id in lowered for entity_id in private_ids):
        raise ValueError("Ideal TK report exposes a private entity ID")
    if _NUMERIC_TOKEN.search(report):
        raise ValueError("Ideal TK report exposes a numeric value")


def _transition_lines(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    transition: closed_gt.CompiledTransition,
) -> list[str]:
    names = _entity_names(scene)
    lines: list[str] = []
    changed = [
        spec
        for spec in closed_gt.predicate_specs(scene)
        if before.values[spec.key] != after.values[spec.key]
    ]
    if changed:
        for spec in changed:
            old = before.values[spec.key]
            new = after.values[spec.key]
            lines.append(
                f"- {_public_label(scene, spec)} = {_public_enum(scene, old)} -> "
                f"{_public_enum(scene, new)} "
                f"({_public_value(scene, old)} -> {_public_value(scene, new)})."
            )
        lines.append("- Every canonical current-state slot not listed above was unchanged.")
    else:
        lines.append("- Every canonical current-state slot was unchanged.")
    for entity in (*scene.executors, *scene.objects):
        key = f"transition.{entity.kind}.{entity.entity_id}.pose"
        value = transition.values[key]
        lines.append(
            f"- pose change for {names[entity.entity_id]} = {_public_enum(scene, value)} "
            f"({_public_value(scene, value)})."
        )
    for executor in scene.executors:
        for obj in scene.objects:
            key = f"transition.executor_object.{executor.entity_id}.{obj.entity_id}.holding_outcome"
            value = transition.values[key]
            lines.append(
                f"- holding outcome for {names[executor.entity_id]} and "
                f"{names[obj.entity_id]} = {_public_enum(scene, value)} "
                f"({_public_value(scene, value)})."
            )
    aggregate = transition.values["transition.any_relevant_state"]
    lines.append(
        f"- aggregate relevant state change = {_public_enum(scene, aggregate)} "
        f"({_public_value(scene, aggregate)})."
    )
    return lines


def _event_lines(
    scene: closed_gt.SceneSpec, event_batches: tuple[SemanticEventBatch, ...]
) -> list[str]:
    specs = {spec.key: spec for spec in closed_gt.predicate_specs(scene)}
    if not event_batches:
        return ["- No tracked semantic predicate changed during the block."]
    lines: list[str] = []
    for batch in event_batches:
        changes: list[str] = []
        for key, old, new in batch.changes:
            spec = specs.get(key)
            if spec is None:
                raise ValueError(f"Event references an unknown canonical state slot: {key}")
            changes.append(
                f"{_public_label(scene, spec)} = {_public_enum(scene, old)} -> "
                f"{_public_enum(scene, new)} "
                f"({_public_value(scene, old)} -> {_public_value(scene, new)})"
            )
        lines.append("- In one simulator step: " + "; ".join(changes) + ".")
    return lines


def render_takeover_report(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    calls: tuple[ApiCallTrace, ...],
    event_batches: tuple[SemanticEventBatch, ...],
    prior_ledger: tuple[str, ...],
) -> tuple[str, tuple[str, ...], str]:
    """Render a reversible complete state and latest-block factual transition."""
    del calls  # Call boundaries remain in the private audit; event order is public.
    closed_gt.validate_snapshot(scene, before)
    closed_gt.validate_snapshot(scene, after)
    transition = closed_gt.compile_transition(scene, before, after)
    specs = closed_gt.predicate_specs(scene)
    derived = tuple(spec for spec in specs if spec.layer == "derived")
    primitive = tuple(spec for spec in specs if spec.layer == "primitive")
    lines = [
        "Authoritative factual reconstruction:",
        "The facts below replace the native visual report and apply only to the latest executed code block.",
        "Each canonical slot has exactly one authoritative value. Derived composite slots are the authoritative interpretation of their primitive evidence; a primitive fact alone must not be promoted into a conflicting composite state.",
        "",
        _DERIVED_STATE_HEADER,
    ]
    lines.extend(_state_line(scene, spec, after.values[spec.key]) for spec in derived)
    lines.extend(("", _PRIMITIVE_STATE_HEADER))
    lines.extend(_state_line(scene, spec, after.values[spec.key]) for spec in primitive)
    lines.extend(("", _TRANSITION_HEADER))
    lines.extend(_transition_lines(scene, before, after, transition))
    lines.extend(("", _EVENT_HEADER))
    lines.extend(_event_lines(scene, event_batches))

    outcomes = _interval_grasp_outcomes(scene, before, after, event_batches)
    names = _entity_names(scene)
    spec_by_key = {spec.key: spec for spec in closed_gt.predicate_specs(scene)}
    meaningful_outcomes: list[str] = []
    for key, outcome in outcomes:
        executor, obj = (names[item] for item in spec_by_key[key].arguments)
        meaningful_outcomes.append(
            f"{executor} and {obj} = {outcome} ({_interval_outcome_text(outcome)})"
        )
    for outcome in meaningful_outcomes:
        lines.append(f"- interval grasp outcome for {outcome}.")

    if prior_ledger:
        lines.extend(("", _HISTORY_HEADER))
        lines.extend(f"- Historical block outcome only: {entry}." for entry in prior_ledger)

    changed_labels = [
        _public_label(scene, spec)
        for spec in specs
        if before.values[spec.key] != after.values[spec.key]
    ]
    nonempty_outcomes = [
        outcome for outcome in meaningful_outcomes if "NO_EXECUTOR_CONTACT_OBSERVED" not in outcome
    ]
    if nonempty_outcomes:
        summary = "; ".join(nonempty_outcomes)
    elif changed_labels:
        summary = "Changed: " + ", ".join(changed_labels[:4])
    else:
        summary = "No canonical physical predicate changed"
    report = "\n".join(lines)
    _validate_public_report(scene, report)
    decoded = parse_canonical_current_state(scene, report)
    if decoded != after.values:
        raise RuntimeError("Canonical public current-state round trip failed")
    return report, tuple(spec.key for spec in specs), summary


def run_takeover(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    turn: int,
    task: str,
    task_description: str,
    executed_code_blocks: list[str],
    console_stdout: str,
    console_stderr: str,
    native_execution_context: str,
    native_report: str | None,
    native_decision_prompt: list[dict[str, Any]],
    private_task_completed: bool,
) -> list[dict[str, Any]]:
    """Replace every regeneration PT carrier with verified factual state."""
    if not isinstance(native_report, str) or not native_report.strip():
        raise ValueError("Ideal TK requires a native M3 report")
    key = _cache_key(output_dir, trial)
    runtime = _RUNTIMES.get(key)
    if runtime is None:
        raise RuntimeError("Ideal TK runtime is missing")
    if runtime.error is not None:
        raise RuntimeError(f"Ideal TK capture failed: {runtime.error}")
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("task") != task
        or state.get("trial") != trial
        or state.get("turns_completed") != turn - 1
        or len(executed_code_blocks) != turn
    ):
        raise ValueError("Ideal TK trace provenance or turn synchronization mismatch")

    after = closed_gt.extract_cube_snapshot(env)
    runtime.observe(after)
    before = runtime.block_before
    event_batches = tuple(runtime.pending)
    calls = tuple(runtime.calls)
    transition = closed_gt.compile_transition(runtime.scene, before, after)
    interval_outcomes = _interval_grasp_outcomes(runtime.scene, before, after, event_batches)
    prior_ledger = tuple(state.get("evidence_ledger", []))
    report, canonical_fields, summary = render_takeover_report(
        runtime.scene, before, after, calls, event_batches, prior_ledger
    )
    decoded_current = parse_canonical_current_state(runtime.scene, report)
    decoded_before, decoded_transition = parse_canonical_transition(
        runtime.scene, report, decoded_current
    )
    if decoded_current != after.values:
        raise RuntimeError("Public current state is not lossless")
    if decoded_before != before.values:
        raise RuntimeError("Public prior endpoint is not lossless")
    if decoded_transition != transition.values:
        raise RuntimeError("Public endpoint transition is not lossless")
    scoped_context, action_code, removed_comments = render_scope_correct_execution_context(
        executed_code_blocks, console_stdout, console_stderr
    )
    corrected = _replace_text_once(
        native_decision_prompt,
        original=native_execution_context,
        replacement=scoped_context,
        carrier="execution-context",
    )
    corrected = _replace_text_once(
        corrected,
        original=native_report,
        replacement=report,
        carrier="visual-report",
    )
    corrected, native_initial_context, initial_suffix_removed = _restore_task_contract(
        corrected,
        task_description=task_description,
        execution_context=scoped_context,
    )
    if _non_text_structure(native_decision_prompt) != _non_text_structure(corrected):
        raise RuntimeError("Ideal TK changed request roles or media")

    goal = closed_gt.cube_goal_definition(task)
    goal_audit = closed_gt.goal_iff_audit(
        after,
        goal,
        private_task_completed=private_task_completed,
    )
    if goal_audit["iff_holds"] is not True:
        raise RuntimeError("Ideal TK benchmark-goal IFF audit failed")
    public_goal_satisfied = closed_gt.goal_satisfied(
        closed_gt.CompiledSnapshot(
            closed_gt.SCHEMA_VERSION,
            decoded_current,
            {},
        ),
        goal,
    )
    public_goal_audit = {
        "semantic_goal_satisfied": public_goal_satisfied,
        "private_semantic_goal_satisfied": goal_audit["semantic_goal_satisfied"],
        "private_task_completed": bool(private_task_completed),
        "iff_holds": (
            public_goal_satisfied
            == goal_audit["semantic_goal_satisfied"]
            == bool(private_task_completed)
        ),
        "goal_definition_visible": False,
        "private_poses_used": False,
    }
    if public_goal_audit["iff_holds"] is not True:
        raise RuntimeError("Public canonical state benchmark-goal IFF audit failed")

    public_events = [
        {"changes": [list(change) for change in batch.changes]} for batch in event_batches
    ]
    turn_semantics = {
        "public_current_state": dict(decoded_current),
        "public_transition": dict(decoded_transition),
        "public_interval_grasp_outcomes": dict(interval_outcomes),
        "public_temporal_event_batches": public_events,
    }
    public_calls = [
        {
            "name": call.name,
            "target_ids": list(call.target_ids),
            "status": call.status,
            "before_state_sha256": _sha256_json(call.before.values),
            "after_state_sha256": _sha256_json(call.after.values),
            "event_batch_count": len(call.events),
        }
        for call in calls
    ]
    turn_root = root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    (turn_root / "native_execution_context.txt").write_text(
        native_execution_context, encoding="utf-8"
    )
    (turn_root / "scope_correct_execution_context.txt").write_text(scoped_context, encoding="utf-8")
    (turn_root / "native_report.txt").write_text(native_report, encoding="utf-8")
    (turn_root / "takeover_report.txt").write_text(report, encoding="utf-8")
    _save_json(
        turn_root / "takeover.json",
        {
            "schema_version": SCHEMA_VERSION,
            "closed_gt_schema_version": closed_gt.SCHEMA_VERSION,
            "task": task,
            "trial": trial,
            "turn": turn,
            "carriers": [
                "initial_visual_pt_suffix",
                "execution_context",
                "m3_visual_difference_report",
            ],
            "task_contract_sha256": _sha256_text(task_description),
            "native_initial_context_sha256": _sha256_text(native_initial_context),
            "initial_visual_pt_suffix_removed": initial_suffix_removed,
            "initial_visual_pt_suffix_absent": True,
            "native_execution_context_sha256": _sha256_text(native_execution_context),
            "scope_correct_execution_context_sha256": _sha256_text(scoped_context),
            "latest_executed_code_sha256": _sha256_text(executed_code_blocks[-1]),
            "latest_action_code_sha256": _sha256_text(action_code),
            "student_comment_count_removed": removed_comments,
            "earlier_code_block_count_omitted": len(executed_code_blocks) - 1,
            "native_report_sha256": _sha256_text(native_report),
            "takeover_report_sha256": _sha256_text(report),
            "native_prompt_sha256": _sha256_json(native_decision_prompt),
            "takeover_prompt_sha256": _sha256_json(corrected),
            "non_text_structure_identical": True,
            "latest_block_only_binding": True,
            "canonical_current_fields": list(canonical_fields),
            "canonical_current_field_count": len(canonical_fields),
            "public_current_state": dict(decoded_current),
            "public_current_state_field_count": len(decoded_current),
            "public_current_state_roundtrip_equal": True,
            "public_prior_state": dict(decoded_before),
            "public_prior_state_roundtrip_equal": True,
            "public_transition": dict(decoded_transition),
            "public_transition_field_count": len(decoded_transition),
            "public_transition_roundtrip_equal": True,
            "public_full_closure_visible": True,
            "full_prior_state": dict(before.values),
            "full_current_state": dict(after.values),
            "full_transition": dict(transition.values),
            "public_interval_grasp_outcomes": dict(interval_outcomes),
            "public_temporal_event_batches": public_events,
            "public_turn_semantics_sha256": _sha256_json(turn_semantics),
            "api_calls": public_calls,
            "private_goal_iff_audit": goal_audit,
            "public_goal_iff_audit": public_goal_audit,
            "report_word_count": len(report.split()),
            "report_attention_target": PUBLIC_WORD_AUDIT_TARGET,
            "report_attention_target_exceeded": (len(report.split()) > PUBLIC_WORD_AUDIT_TARGET),
            "continuous_state_persisted": False,
            "raw_image_replaced": False,
            "executed_code_execution_changed": False,
            "executed_code_prompt_representation_changed": (
                removed_comments > 0 or len(executed_code_blocks) > 1
            ),
            "console_stdout_content_changed": False,
            "console_stderr_content_changed": False,
            "execution_scope_labels_changed": True,
            "task_or_api_text_replaced": False,
            "completion_or_reward_visible": False,
            "next_move_visible": False,
            "numeric_state_visible": False,
        },
    )

    ledger = list(prior_ledger)
    if not ledger or ledger[-1] != summary:
        ledger.append(summary)
    state["evidence_ledger"] = ledger[-MAX_LEDGER_ENTRIES:]
    state["turns_completed"] = turn
    state["latest_public_state_sha256"] = _sha256_json(after.values)
    state["latest_public_transition_sha256"] = _sha256_json(transition.values)
    _save_json(state_path, state)
    runtime.block_before = after
    runtime.pending.clear()
    runtime.calls.clear()
    return corrected
