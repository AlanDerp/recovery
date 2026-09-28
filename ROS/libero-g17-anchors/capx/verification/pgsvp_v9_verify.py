"""Materialized PT-GT verification over the historical v9 factual kernel.

The action student first externalizes a perceived-truth worksheet from the same
decision context. A privileged verifier corrects only fields that differ from
GT. The host applies that delta, fails closed unless the result equals the
complete verified state, and exposes only the resulting worksheet to the action
student. Raw PT and the correction protocol remain private audit artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from capx.verification.core_fg import extract_core_facts
from capx.utils.launch_utils import MULTI_TURN_DECISION_INSTRUCTION


SCHEMA_VERSION = "capx-pgsvp-v9-as-materialized-ptgt-v1"
VERIFIER_SCHEMA_VERSION = "pgsvp_v9_pt_gt_verifier_v2"
PT_REQUEST_SEED_OFFSET = 500
VERIFIER_REQUEST_SEED_OFFSET = 700
SCHEMA_RETRY_SEED_STRIDE = 100
PUBLIC_HEADER = "GT_BACKED_VERIFIER_FACTS:"
DELTA_HEADER = "PGSVP_PT_GT_CORRECTIONS:"
PUBLIC_BRIDGE_MARKER = "PGSVP MATERIALIZED STATE CONTRACT"
PUBLIC_FIELDS = (
    "TASK_RELATION",
    "RUNTIME",
    "ATTACHMENT",
    "RELATIVE_X",
    "RELATIVE_Y",
    "RELATIVE_Z",
    "FOOTPRINT",
    "CONTACT_SUPPORT",
    "STAGE_POSTCONDITIONS",
    "EARLIEST_ABSENT_POSTCONDITION",
    "DOMINANT_ABSENT_FACT",
    "FALSE_BELIEFS",
)
GT_PUBLIC_FIELDS = (
    "ATTACHMENT",
    "RELATIVE_X",
    "RELATIVE_Y",
    "RELATIVE_Z",
    "FOOTPRINT",
    "CONTACT_SUPPORT",
)
LLM_PUBLIC_FIELDS = tuple(
    field for field in PUBLIC_FIELDS if field not in GT_PUBLIC_FIELDS
)
PT_GT_ENUMS = {
    "ATTACHMENT": frozenset(("empty", "holding_target", "holding_other")),
    "RELATIVE_X": frozenset(
        ("aligned", "target_offset_negative_x", "target_offset_positive_x")
    ),
    "RELATIVE_Y": frozenset(
        ("aligned", "target_offset_negative_y", "target_offset_positive_y")
    ),
    "RELATIVE_Z": frozenset(("same_level", "target_below_goal", "target_above_goal")),
    "FOOTPRINT": frozenset(
        (
            "target_center_inside_goal_footprint",
            "target_center_outside_goal_footprint",
        )
    ),
    "CONTACT_SUPPORT": frozenset(
        ("target_contacting_goal", "target_separated_from_goal")
    ),
}

_EXACT_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")
_FORBIDDEN_PUBLIC = (
    re.compile(
        r"\b(?:reward|task_completed|task\s+(?:completed?|incomplete|failed)|"
        r"task\s+postcondition\s+(?:not\s+)?met|success(?:ful)?(?:\s+label)?|"
        r"trial|seed|run_id)\b",
        re.I,
    ),
    re.compile(r"\b(?:cubeA|cubeB|body_xpos|body_xquat|sim\.data|robosuite_env)\b"),
    re.compile(r"\b(?:goto_pose|open_gripper|close_gripper|sample_grasp_pose)\b"),
    re.compile(
        r"\b(?:should|must|recommend|please|need(?:s)?\s+to|ought\s+to|"
        r"next\s+action|repair\s+plan)\b",
        re.I,
    ),
)
_IMPERATIVE_START = re.compile(
    r"^(?:move|place|release|open|close|call|use|retry|regrasp|lift|lower|execute)\b",
    re.I,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _public_bridge_path() -> Path:
    return Path(
        os.environ.get(
            "CAPX_PGSVP_V9_BRIDGE_PATH",
            Path(__file__).with_name("pgsvp_v9_public_bridge.txt"),
        )
    )


def _load_public_bridge() -> tuple[str, str]:
    text = _public_bridge_path().read_text(encoding="utf-8").strip()
    if text.count(PUBLIC_BRIDGE_MARKER) != 1:
        raise ValueError("PGSVP bridge marker must occur exactly once")
    if PUBLIC_HEADER in text or DELTA_HEADER in text:
        raise ValueError("PGSVP bridge must not contain a factual packet")
    if len(text) > 1200 or _EXACT_NUMBER.search(text):
        raise ValueError("PGSVP bridge failed length or numeric audit")
    for pattern in _FORBIDDEN_PUBLIC[1:3]:
        if pattern.search(text):
            raise ValueError(f"PGSVP bridge matched unsafe pattern {pattern.pattern!r}")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_text_occurrences(value: Any, needle: str) -> int:
    if isinstance(value, str):
        return value.count(needle)
    if isinstance(value, dict):
        return sum(_count_text_occurrences(item, needle) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_text_occurrences(item, needle) for item in value)
    return 0


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _unwrap_cube_environment(env: Any) -> tuple[Any, Any]:
    low_level = getattr(env, "low_level_env", None)
    simulator = getattr(low_level, "robosuite_env", None)
    required = (
        "cubeA",
        "cubeB",
        "cubeA_body_id",
        "cubeB_body_id",
        "robots",
        "sim",
    )
    if low_level is None or simulator is None or not all(
        hasattr(simulator, name) for name in required
    ):
        raise TypeError(
            "PGSVP-v9 verifier requires a CaP-X Robosuite Stack environment"
        )
    return low_level, simulator


def _privileged_packet(env: Any, *, task: str) -> dict[str, Any]:
    """Build private evidence without reward or completion labels."""
    low_level, simulator = _unwrap_cube_environment(env)
    data = simulator.sim.data
    core = extract_core_facts(env)
    raw = {
        "red_cube": {
            "center_world": data.body_xpos[simulator.cubeA_body_id],
            "orientation_wxyz": data.body_xquat[simulator.cubeA_body_id],
            "half_extents": simulator.cubeA.size,
            "held_by_gripper": simulator._check_grasp(
                gripper=simulator.robots[0].gripper,
                object_geoms=simulator.cubeA,
            ),
            "contacting_table": simulator.check_contact(
                simulator.cubeA, "table_collision"
            ),
        },
        "green_cube": {
            "center_world": data.body_xpos[simulator.cubeB_body_id],
            "orientation_wxyz": data.body_xquat[simulator.cubeB_body_id],
            "half_extents": simulator.cubeB.size,
            "held_by_gripper": simulator._check_grasp(
                gripper=simulator.robots[0].gripper,
                object_geoms=simulator.cubeB,
            ),
            "contacting_table": simulator.check_contact(
                simulator.cubeB, "table_collision"
            ),
        },
        "red_green_contact": simulator.check_contact(
            simulator.cubeA, simulator.cubeB
        ),
    }
    task_pose_input = {
        "cubeA_pos": np.asarray(raw["red_cube"]["center_world"]),
        "cubeB_pos": np.asarray(raw["green_cube"]["center_world"]),
        "cubeA_quat": np.roll(
            np.asarray(raw["red_cube"]["orientation_wxyz"]), -1
        ),
        "cubeB_quat": np.roll(
            np.asarray(raw["green_cube"]["orientation_wxyz"]), -1
        ),
    }
    task_poses = low_level._cube_pose_dict(task_pose_input)
    raw["task_frame_centers"] = {
        "red_cube": np.asarray(task_poses["primary"][:3]),
        "green_cube": np.asarray(task_poses["secondary"][:3]),
    }
    return {
        "task": task,
        "public_object_mapping": {
            "target": "red cube",
            "goal_or_support": "green cube",
            "executor": "robot gripper",
        },
        "deterministic_cross_check": {
            "target_grasp": core.target_grasp,
            "relative_x": core.relative_x,
            "relative_y": core.relative_y,
            "relative_z": core.relative_z,
            "footprint": core.footprint,
            "target_goal_contact": core.target_goal_contact,
            "target_table_contact": core.target_table_contact,
        },
        "raw_private_geometry_and_contact": _to_jsonable(raw),
        "reward_or_completion_oracle_provided": False,
    }


def initialize_trial_trace(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    """Record that the treatment hook was active before any action executes."""
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Unsupported PGSVP-v9 task: {task}")
    root = (
        Path(output_dir)
        / "pgsvp_v9_verify_traces"
        / f"trial_{trial:02d}"
    )
    root.mkdir(parents=True, exist_ok=False)
    initial_state = {field: "unknown" for field in PUBLIC_FIELDS}
    record = {
        "schema_version": SCHEMA_VERSION,
        "trial": trial,
        "task": task,
        "stage": "INITIALIZED_BEFORE_ACTION",
        "initial_verified_state": initial_state,
        "initial_verified_state_sha256": _sha256(initial_state),
        "public_bridge_injected": False,
    }
    (root / "initialization.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_prior_verified_state(
    *,
    output_dir: str,
    trial: int,
    turn: int,
) -> dict[str, str]:
    if turn <= 1:
        return {field: "unknown" for field in PUBLIC_FIELDS}
    path = (
        Path(output_dir)
        / "pgsvp_v9_verify_traces"
        / f"trial_{trial:02d}"
        / f"turn_{turn - 1:02d}"
        / "verification.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    state = record.get("verified_state")
    if not isinstance(state, dict) or set(state) != set(PUBLIC_FIELDS):
        raise ValueError(f"Malformed prior PGSVP verified state: {path}")
    if not all(isinstance(state[field], str) for field in PUBLIC_FIELDS):
        raise ValueError(f"Non-string prior PGSVP verified state: {path}")
    return {field: state[field] for field in PUBLIC_FIELDS}


def record_student_prompt_audit(
    *,
    output_dir: str,
    trial: int,
    turn: int,
    decision_prompt: Any,
) -> None:
    """Prove that action sees one materialized state and no PT/delta protocol."""
    path = (
        Path(output_dir)
        / "pgsvp_v9_verify_traces"
        / f"trial_{trial:02d}"
        / f"turn_{turn:02d}"
        / "verification.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    prompt_value = _to_jsonable(decision_prompt)
    prompt_sha = _sha256(prompt_value)
    if prompt_sha != record.get("student_action_prompt_sha256"):
        raise RuntimeError(f"PGSVP action prompt differs from recorded prompt: {path}")

    canonical_pt = record.get("pt_canonical_response")
    materialized_state = record.get("materialized_state_response")
    delta = record.get("delta_message")
    corrections = record.get("fact_corrections")
    if (
        not isinstance(canonical_pt, str)
        or not isinstance(materialized_state, str)
        or not isinstance(delta, str)
        or not isinstance(corrections, dict)
    ):
        raise RuntimeError(f"Incomplete PGSVP prompt audit record: {path}")
    counts = {
        "pt_request": _count_text_occurrences(
            prompt_value, "PGSVP PERCEIVED-TRUTH WORKSHEET"
        ),
        "prior_verified_worksheet": _count_text_occurrences(
            prompt_value, "PRIOR VERIFIED WORKSHEET"
        ),
        "closed_physical_values": _count_text_occurrences(
            prompt_value, "CLOSED PHYSICAL VALUES"
        ),
        "materialized_state": _count_text_occurrences(
            prompt_value, materialized_state
        ),
        "bridge": _count_text_occurrences(prompt_value, PUBLIC_BRIDGE_MARKER),
        "delta": _count_text_occurrences(prompt_value, DELTA_HEADER),
        "full_gt_report": _count_text_occurrences(prompt_value, PUBLIC_HEADER),
        "action_instruction": _count_text_occurrences(
            prompt_value, MULTI_TURN_DECISION_INSTRUCTION
        ),
    }
    expected = {
        "pt_request": 0,
        "prior_verified_worksheet": 0,
        "closed_physical_values": 0,
        "materialized_state": 1,
        "bridge": 1,
        "delta": 0,
        "full_gt_report": 0,
        "action_instruction": 1,
    }
    if counts != expected:
        raise RuntimeError(
            f"PGSVP action prompt violates PT-GT ownership contract: {counts}"
        )
    materialized_assistant_matches = sum(
        1
        for message in decision_prompt
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("content") == materialized_state
    )
    if materialized_assistant_matches != 1:
        raise RuntimeError(
            "Materialized PT-GT state must occur once as an assistant message"
        )
    raw_pt_assistant_matches = sum(
        1
        for message in decision_prompt
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("content") == canonical_pt
    )
    expected_raw_pt_matches = 0 if corrections else 1
    if raw_pt_assistant_matches != expected_raw_pt_matches:
        raise RuntimeError(
            "Uncorrected PT leaked into the materialized action prompt"
        )
    record.update(
        {
            "student_prompt_audit_passed": True,
            "student_prompt_occurrences": counts,
            "materialized_state_visible_as_assistant_message": True,
            "materialized_state_assistant_message_occurrences": (
                materialized_assistant_matches
            ),
            "uncorrected_pt_visible_to_action": False,
            "raw_pt_assistant_message_occurrences": raw_pt_assistant_matches,
        }
    )
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _request(
    args: Any,
    messages: list[dict[str, Any]],
    *,
    seed: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    headers = {"Content-Type": "application/json"}
    if getattr(args, "api_key", None):
        headers["Authorization"] = f"Bearer {args.api_key}"
    timeout = float(os.environ.get("CAPX_MODEL_QUERY_TIMEOUT_SECONDS", "200"))
    attempts = int(os.environ.get("CAPX_PGSVP_V9_TRANSPORT_ATTEMPTS", "2"))
    if attempts < 1:
        raise ValueError("CAPX_PGSVP_V9_TRANSPORT_ATTEMPTS must be positive")
    audit: list[dict[str, Any]] = []
    for index in range(attempts):
        try:
            response = requests.post(
                args.server_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
        except requests.RequestException as exc:
            audit.append({"attempt": index, "error": type(exc).__name__})
            if index + 1 == attempts:
                raise
            time.sleep(1)
            continue
        retriable = response.status_code in {408, 429} or response.status_code >= 500
        audit.append(
            {
                "attempt": index,
                "status_code": response.status_code,
                "retriable": retriable,
            }
        )
        if response.ok:
            body = response.json()
            try:
                content = str(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("Unexpected Qwen response structure") from exc
            return content, {
                "seed": seed,
                "request_sha256": _sha256(payload),
                "response_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "transport": audit,
            }
        if not retriable or index + 1 == attempts:
            response.raise_for_status()
        time.sleep(1)
    raise RuntimeError("PGSVP-v9 request exhausted without a response")


def _extract_json_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Response contains no JSON object")


def _visible_images(images: list[str]) -> list[dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": image}}
        for image in images
        if isinstance(image, str) and image.startswith("data:image/")
    ]


def _factual_execution_evidence(execution_evidence: str) -> str:
    """Remove the M2 action-response instruction from quoted evidence."""
    marker = "Please respond with EXACTLY ONE of the following:"
    evidence = str(execution_evidence)
    if marker in evidence:
        evidence = evidence.split(marker, 1)[0]
    return evidence.rstrip()


def _pt_request_instruction(prior_verified_state: dict[str, str]) -> str:
    if tuple(prior_verified_state) != PUBLIC_FIELDS:
        raise ValueError("Prior PT-GT state does not match the v9 field order")
    closed_values = {
        field: ["UNKNOWN", *sorted(values)]
        for field, values in PT_GT_ENUMS.items()
    }
    return (
        "PGSVP PERCEIVED-TRUTH WORKSHEET\n"
        "Before deciding whether to regenerate, externalize your current factual "
        "beliefs after the latest execution. This worksheet is your own persistent "
        "state and will remain in this same conversation. Update stale prior values. "
        "Use UNKNOWN when the visible evidence does not support a belief. Do not "
        "plan, advise, or write code. Return exactly one JSON object with every "
        "worksheet key and no other text.\n"
        "PRIOR VERIFIED WORKSHEET\n"
        + json.dumps(prior_verified_state, ensure_ascii=True)
        + "\nCLOSED PHYSICAL VALUES\n"
        + json.dumps(closed_values, ensure_ascii=True, sort_keys=True)
    )


def _build_student_pt_prompt(
    decision_prompt: list[dict[str, Any]],
    prior_verified_state: dict[str, str],
) -> list[dict[str, Any]]:
    """Remove all action directives and insert one PT request in their place."""
    prompt = copy.deepcopy(decision_prompt)
    if not prompt or prompt[-1].get("role") != "user":
        raise ValueError("Student decision prompt must end with a user message")
    content = prompt[-1].get("content")
    replacement = _pt_request_instruction(prior_verified_state)
    sentinel = "__CAPX_PGSVP_PT_REQUEST_SENTINEL__"
    response_marker = "Please respond with EXACTLY ONE of the following:"
    replaced = 0

    def transform(text: str) -> str:
        nonlocal replaced
        count = text.count(MULTI_TURN_DECISION_INSTRUCTION)
        replaced += count
        value = text.replace(MULTI_TURN_DECISION_INSTRUCTION, sentinel)
        if response_marker in value:
            prefix, _ = value.split(response_marker, 1)
            value = prefix.rstrip()
            if sentinel in text.replace(MULTI_TURN_DECISION_INSTRUCTION, sentinel):
                value = f"{value}\n{sentinel}"
        return value.replace(sentinel, replacement)

    if isinstance(content, str):
        prompt[-1]["content"] = transform(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            item["text"] = transform(text)
    else:
        raise ValueError("Student decision prompt has unsupported content")
    if replaced != 1:
        raise ValueError(
            f"Expected one action instruction in the student context, found {replaced}"
        )
    if _count_text_occurrences(prompt, MULTI_TURN_DECISION_INSTRUCTION) != 0:
        raise ValueError("Action instruction leaked into the PT request")
    if _count_text_occurrences(prompt, response_marker) != 0:
        raise ValueError("An embedded M2 action directive leaked into the PT request")
    if _count_text_occurrences(prompt, sentinel) != 0:
        raise ValueError("PT request sentinel leaked into the model prompt")
    return prompt


def _canonical_pt_response(pt: dict[str, str]) -> str:
    if tuple(pt) != PUBLIC_FIELDS:
        raise ValueError("Canonical PT does not match the v9 field order")
    return json.dumps(pt, ensure_ascii=True, separators=(",", ":"))


def _normalize_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip() or "unknown"
    if isinstance(value, (bool, int, float)):
        return _canonical_json(value)
    if isinstance(value, (list, dict)):
        return _canonical_json(value)
    return "unknown"


def _normalize_pt(content: str) -> tuple[dict[str, str], dict[str, Any]]:
    value = _extract_json_object(content)
    missing = sorted(set(PUBLIC_FIELDS) - set(value))
    extra = sorted(set(value) - set(PUBLIC_FIELDS))
    coerced = sorted(
        field
        for field in PUBLIC_FIELDS
        if field in value and not isinstance(value[field], str)
    )
    normalized = {
        field: _normalize_value(value.get(field))
        for field in PUBLIC_FIELDS
    }
    invalid_closed: list[str] = []
    for field, allowed in PT_GT_ENUMS.items():
        candidate = normalized[field].casefold()
        if candidate == "unknown":
            normalized[field] = "unknown"
        elif candidate in allowed:
            normalized[field] = candidate
        else:
            normalized[field] = "unknown"
            invalid_closed.append(field)
    return normalized, {
        "missing_fields_filled_unknown": missing,
        "extra_fields_ignored": extra,
        "non_string_fields_canonicalized": coerced,
        "invalid_closed_fields_filled_unknown": invalid_closed,
        "canonical_projection_applied": bool(
            missing or extra or coerced or invalid_closed
        ),
    }


def _validate_pt(content: str) -> dict[str, str]:
    """Compatibility wrapper returning the canonical closed PT worksheet."""
    return _normalize_pt(content)[0]


def _verifier_messages(
    *,
    prompt_text: str,
    schema: dict[str, Any],
    task_description: str,
    execution_evidence: str,
    pt: dict[str, str],
    privileged: dict[str, Any],
) -> list[dict[str, Any]]:
    factual_evidence = _factual_execution_evidence(execution_evidence)
    packet = {
        "task_description": task_description,
        "student_visible_execution_evidence": factual_evidence,
        "student_perceived_truth_candidate": pt,
        "privileged_simulator_evidence": privileged,
        "required_schema": schema,
    }
    return [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": (
                "Verify the candidate perceived-truth worksheet field by field "
                "against privileged evidence. Correct every contradicted or missing "
                "fact. Return strict JSON only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        },
    ]


def _expected_gt_fields(
    privileged: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    deterministic = privileged["deterministic_cross_check"]
    raw = privileged["raw_private_geometry_and_contact"]
    expected_private = {
        "robot_target_attachment": (
            "holding_target"
            if deterministic["target_grasp"] == "target_grasped_by_robot"
            else (
                "holding_other"
                if raw["green_cube"]["held_by_gripper"]
                else "empty"
            )
        ),
        "target_goal_contact": (
            "contact"
            if deterministic["target_goal_contact"] == "target_contacting_goal"
            else "separated"
        ),
        "relative_x": {
            "target_center_negative_x_of_goal_band": "target_offset_negative_x",
            "target_center_within_goal_x_band": "aligned",
            "target_center_positive_x_of_goal_band": "target_offset_positive_x",
        }[deterministic["relative_x"]],
        "relative_y": {
            "target_center_negative_y_of_goal_band": "target_offset_negative_y",
            "target_center_within_goal_y_band": "aligned",
            "target_center_positive_y_of_goal_band": "target_offset_positive_y",
        }[deterministic["relative_y"]],
        "footprint": {
            "target_center_inside_goal_footprint": "target_center_inside_goal_footprint",
            "target_center_outside_goal_footprint": "target_center_outside_goal_footprint",
        }[deterministic["footprint"]],
        "contact_support": (
            "target_contacting_goal"
            if deterministic["target_goal_contact"] == "target_contacting_goal"
            else "target_separated_from_goal"
        ),
    }
    expected_public = {
        "ATTACHMENT": expected_private["robot_target_attachment"],
        "RELATIVE_X": expected_private["relative_x"],
        "RELATIVE_Y": expected_private["relative_y"],
        "RELATIVE_Z": {
            "target_below_goal_height_band": "target_below_goal",
            "target_within_goal_height_band": "same_level",
            "target_above_goal_height_band": "target_above_goal",
        }[deterministic["relative_z"]],
        "FOOTPRINT": expected_private["footprint"],
        "CONTACT_SUPPORT": expected_private["contact_support"],
    }
    return expected_private, expected_public


def _legacy_message_fields(message: str) -> dict[str, str]:
    normalized = re.sub(r"\s+", " ", message).strip()
    if normalized.startswith(PUBLIC_HEADER):
        normalized = normalized[len(PUBLIC_HEADER) :].strip()
    starts: list[tuple[int, str, int]] = []
    for field in PUBLIC_FIELDS:
        match = re.search(rf"(?:^|\s){re.escape(field)}\s*:", normalized)
        if match:
            starts.append((match.start(), field, match.end()))
    starts.sort()
    values: dict[str, str] = {}
    for index, (_, field, value_start) in enumerate(starts):
        value_end = starts[index + 1][0] if index + 1 < len(starts) else len(normalized)
        values[field] = normalized[value_start:value_end].strip(" .;")
    return values


def _candidate_public_fields(parsed: dict[str, Any]) -> tuple[dict[str, Any], str]:
    public = parsed.get("public_student_guidance")
    if not isinstance(public, dict):
        return {}, "missing"
    fields = public.get("fields")
    if isinstance(fields, dict):
        return fields, "structured_fields"
    message = public.get("message")
    if isinstance(message, str):
        return _legacy_message_fields(message), "legacy_message"
    return {}, "missing"


def _safe_candidate_field(field: str, value: Any) -> tuple[str, str | None]:
    text = _normalize_value(value)
    if len(text) > 180:
        return "unknown", "too_long"
    if _IMPERATIVE_START.search(text):
        return "unknown", "imperative_start"
    if _EXACT_NUMBER.search(text):
        return "unknown", "exact_number"
    for pattern in _FORBIDDEN_PUBLIC:
        if pattern.search(text):
            return "unknown", f"forbidden:{pattern.pattern}"
    if any(
        re.search(rf"\b{re.escape(label)}\s*:", text)
        for label in PUBLIC_FIELDS
        if label != field
    ):
        return "unknown", "embedded_field_label"
    return text, None


def _render_public_fields(fields: dict[str, str]) -> str:
    return " ".join(
        [PUBLIC_HEADER]
        + [f"{field}: {fields[field]};" for field in PUBLIC_FIELDS]
    )


def _pt_gt_corrections(
    pt: dict[str, str],
    verified_state: dict[str, str],
) -> dict[str, dict[str, str]]:
    if tuple(pt) != PUBLIC_FIELDS or tuple(verified_state) != PUBLIC_FIELDS:
        raise ValueError("PT-GT comparison requires the complete ordered v9 state")
    return {
        field: {
            "perceived": pt[field],
            "verified": verified_state[field],
        }
        for field in PUBLIC_FIELDS
        if pt[field] != verified_state[field]
    }


def _apply_pt_gt_delta(
    pt: dict[str, str],
    corrections: dict[str, dict[str, str]],
) -> dict[str, str]:
    if tuple(pt) != PUBLIC_FIELDS:
        raise ValueError("PT-GT merge requires the complete ordered v9 PT")
    expected_order = tuple(field for field in PUBLIC_FIELDS if field in corrections)
    if tuple(corrections) != expected_order:
        raise ValueError("PT-GT corrections are not in canonical field order")
    merged = dict(pt)
    for field, correction in corrections.items():
        if set(correction) != {"perceived", "verified"}:
            raise ValueError(f"Malformed correction for {field}")
        if correction["perceived"] != pt[field]:
            raise ValueError(f"Correction base disagrees with PT for {field}")
        merged[field] = correction["verified"]
    return merged


def _render_delta_message(corrections: dict[str, dict[str, str]]) -> str:
    if not corrections:
        return f"{DELTA_HEADER} NONE;"
    expected_order = tuple(field for field in PUBLIC_FIELDS if field in corrections)
    if tuple(corrections) != expected_order:
        raise ValueError("PT-GT corrections are not in canonical field order")
    return " ".join(
        [DELTA_HEADER]
        + [
            f"{field}: {corrections[field]['verified']};"
            for field in corrections
        ]
    )


def _validate_delta_message(
    message: str,
    corrections: dict[str, dict[str, str]],
) -> None:
    if message != _render_delta_message(corrections):
        raise ValueError("PT-GT delta is not the canonical correction rendering")
    if len(message) > 2600:
        raise ValueError("PT-GT delta exceeds 2600 characters")
    if _EXACT_NUMBER.search(message):
        raise ValueError("PT-GT delta contains an exact number")
    for pattern in _FORBIDDEN_PUBLIC:
        if pattern.search(message):
            raise ValueError(f"PT-GT delta matched forbidden {pattern.pattern}")


def _compose_student_action_prompt(
    *,
    student_decision_prompt: list[dict[str, Any]],
    materialized_state_response: str,
    bridge: str,
) -> list[dict[str, Any]]:
    if bridge.count(PUBLIC_BRIDGE_MARKER) != 1:
        raise ValueError("PT-GT action prompt requires one bridge marker")
    materialized_state = _validate_pt(materialized_state_response)
    if _canonical_pt_response(materialized_state) != materialized_state_response:
        raise ValueError("Materialized PT-GT state is not canonical")
    prompt = _without_action_instruction(student_decision_prompt)
    prompt.append(
        {"role": "assistant", "content": materialized_state_response}
    )
    prompt.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{bridge}\n\n{MULTI_TURN_DECISION_INSTRUCTION}"
                    ),
                }
            ],
        }
    )
    return prompt


def _without_action_instruction(
    decision_prompt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain M2 evidence while removing its old action-response requests."""
    prompt = copy.deepcopy(decision_prompt)
    if not prompt or prompt[-1].get("role") != "user":
        raise ValueError("Student decision prompt must end with a user message")
    content = prompt[-1].get("content")
    response_marker = "Please respond with EXACTLY ONE of the following:"
    removed_action_instructions = 0
    removed_response_markers = 0

    def transform(text: str) -> str:
        nonlocal removed_action_instructions, removed_response_markers
        removed_action_instructions += text.count(MULTI_TURN_DECISION_INSTRUCTION)
        value = text.replace(MULTI_TURN_DECISION_INSTRUCTION, "")
        if response_marker in value:
            removed_response_markers += value.count(response_marker)
            value = value.split(response_marker, 1)[0]
        return value.rstrip()

    if isinstance(content, str):
        prompt[-1]["content"] = transform(content)
    elif isinstance(content, list):
        cleaned = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                cleaned.append(item)
                continue
            text = item.get("text")
            if not isinstance(text, str):
                cleaned.append(item)
                continue
            updated = copy.deepcopy(item)
            updated["text"] = transform(text)
            if updated["text"]:
                cleaned.append(updated)
        prompt[-1]["content"] = cleaned
    else:
        raise ValueError("Student decision prompt has unsupported content")
    if removed_action_instructions != 1 or removed_response_markers != 1:
        raise ValueError(
            "Expected one canonical and one embedded M2 action request, found "
            f"{removed_action_instructions} and {removed_response_markers}"
        )
    if _count_text_occurrences(prompt, MULTI_TURN_DECISION_INSTRUCTION) != 0:
        raise ValueError("Canonical M2 action instruction leaked into action evidence")
    if _count_text_occurrences(prompt, response_marker) != 0:
        raise ValueError("Embedded M2 action request leaked into action evidence")
    return prompt


def _validate_public_message(message: str) -> None:
    if not message.startswith(PUBLIC_HEADER):
        raise ValueError("Public verifier message has the wrong header")
    if len(message) > 1700:
        raise ValueError("Public verifier message exceeds 1700 characters")
    if _EXACT_NUMBER.search(message):
        raise ValueError("Public verifier message contains an exact number")
    for pattern in _FORBIDDEN_PUBLIC:
        if pattern.search(message):
            raise ValueError(f"Public verifier message matched forbidden {pattern.pattern}")


def _canonicalize_verifier_output(
    content: str,
    privileged: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    candidate = _extract_json_object(content)
    candidate_fields, candidate_format = _candidate_public_fields(candidate)
    expected_private, expected_public = _expected_gt_fields(privileged)

    final_fields: dict[str, str] = {}
    rejected_fields: dict[str, str] = {}
    for field in LLM_PUBLIC_FIELDS:
        value, rejection = _safe_candidate_field(field, candidate_fields.get(field))
        final_fields[field] = value
        if rejection:
            rejected_fields[field] = rejection
    final_fields.update(expected_public)
    final_fields = {field: final_fields[field] for field in PUBLIC_FIELDS}

    private_candidate = candidate.get("private_verifier_audit")
    private = copy.deepcopy(private_candidate) if isinstance(private_candidate, dict) else {}
    attachment = private.get("attachment_facts")
    if not isinstance(attachment, dict):
        attachment = {}
        private["attachment_facts"] = attachment
    spatial = private.get("axis_grounded_spatial_facts")
    if not isinstance(spatial, dict):
        spatial = {}
        private["axis_grounded_spatial_facts"] = spatial
    observed = {
        "robot_target_attachment": attachment.get("robot_target_attachment"),
        "target_goal_contact": attachment.get("target_goal_contact"),
        "relative_x": spatial.get("relative_x"),
        "relative_y": spatial.get("relative_y"),
        "footprint": spatial.get("footprint"),
        "contact_support": spatial.get("contact_support"),
    }
    attachment["robot_target_attachment"] = expected_private[
        "robot_target_attachment"
    ]
    attachment["target_goal_contact"] = expected_private["target_goal_contact"]
    for field in ("relative_x", "relative_y", "footprint", "contact_support"):
        spatial[field] = expected_private[field]

    mismatches = {
        field: {"candidate": observed[field], "canonical_gt": expected_private[field]}
        for field in expected_private
        if observed[field] != expected_private[field]
    }
    message = _render_public_fields(final_fields)
    _validate_public_message(message)
    leakage = {
        "safe_public_guidance": True,
        "contains_exact_coordinates": False,
        "contains_numeric_constants": False,
        "contains_reward_value": False,
        "contains_trial_id_or_seed": False,
        "contains_hidden_object_id": False,
        "contains_simulator_internal_api": False,
        "contains_oracle_code": False,
        "contains_action_advice": False,
        "contains_repair_plan": False,
        "contains_imperative_language": False,
    }
    parsed = {
        "schema_version": VERIFIER_SCHEMA_VERSION,
        "private_verifier_audit": private,
        "public_student_guidance": {
            "fields": final_fields,
            "message": message,
        },
        "leakage_audit": leakage,
    }
    interface_audit = {
        "candidate_format": candidate_format,
        "candidate_top_level_extra_keys": sorted(
            set(candidate)
            - {
                "schema_version",
                "private_verifier_audit",
                "public_student_guidance",
                "leakage_audit",
            }
        ),
        "candidate_missing_public_fields": sorted(
            set(PUBLIC_FIELDS) - set(candidate_fields)
        ),
        "candidate_rejected_public_fields": rejected_fields,
        "deterministic_gt_overrides": mismatches,
        "final_renderer": "fixed_order_local_v1",
    }
    return message, parsed, interface_audit


def _sanitize_verifier_output(
    content: str,
    privileged: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Canonicalize and sanitize a verifier candidate for compatibility."""
    message, parsed, _ = _canonicalize_verifier_output(content, privileged)
    return message, parsed


def _validate_private_gt_alignment(
    parsed: dict[str, Any],
    privileged: dict[str, Any],
) -> dict[str, str]:
    """Require the verifier's core private enums to match deterministic GT."""
    expected, _ = _expected_gt_fields(privileged)
    private = parsed.get("private_verifier_audit")
    if not isinstance(private, dict):
        raise ValueError("Verifier output is missing private_verifier_audit")
    attachment = private.get("attachment_facts")
    spatial = private.get("axis_grounded_spatial_facts")
    if not isinstance(attachment, dict) or not isinstance(spatial, dict):
        raise ValueError("Verifier output is missing private attachment/spatial facts")
    observed = {
        "robot_target_attachment": attachment.get("robot_target_attachment"),
        "target_goal_contact": attachment.get("target_goal_contact"),
        "relative_x": spatial.get("relative_x"),
        "relative_y": spatial.get("relative_y"),
        "footprint": spatial.get("footprint"),
        "contact_support": spatial.get("contact_support"),
    }
    mismatches = {
        key: f"expected={expected[key]!r}, observed={observed[key]!r}"
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"Verifier private facts disagree with deterministic GT: {mismatches}")
    return expected


def _request_until_valid(
    args: Any,
    messages: list[dict[str, Any]],
    *,
    seed: int,
    max_tokens: int,
    protocol_name: str,
    parser: Any,
) -> tuple[str, Any, dict[str, Any], list[dict[str, Any]]]:
    """Retry malformed responses and return the exact accepted request context."""
    attempts = int(os.environ.get("CAPX_PGSVP_V9_SCHEMA_ATTEMPTS", "2"))
    if attempts < 1:
        raise ValueError("CAPX_PGSVP_V9_SCHEMA_ATTEMPTS must be positive")
    current_messages = copy.deepcopy(messages)
    protocol_audit: list[dict[str, Any]] = []
    for attempt in range(attempts):
        attempt_seed = seed + attempt * SCHEMA_RETRY_SEED_STRIDE
        content, request_audit = _request(
            args,
            current_messages,
            seed=attempt_seed,
            max_tokens=max_tokens,
        )
        try:
            parsed = parser(content)
        except (KeyError, TypeError, ValueError) as exc:
            protocol_audit.append(
                {
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "request": request_audit,
                    "response_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
            )
            if attempt + 1 == attempts:
                raise ValueError(
                    f"{protocol_name} remained malformed after {attempts} attempts"
                ) from exc
            current_messages.extend(
                (
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"The previous {protocol_name} response was not a "
                            "parseable protocol object. Return exactly one corrected "
                            "JSON object matching the provided worksheet/schema. "
                            "Do not add explanation, code, or action advice."
                        ),
                    },
                )
            )
            continue
        protocol_audit.append(
            {
                "attempt": attempt,
                "seed": attempt_seed,
                "request": request_audit,
                "response_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "parse_error": None,
            }
        )
        combined_audit = copy.deepcopy(request_audit)
        combined_audit.update(
            {
                "base_seed": seed,
                "seed": attempt_seed,
                "protocol_name": protocol_name,
                "schema_attempt_count": attempt + 1,
                "schema_attempts": protocol_audit,
            }
        )
        return content, parsed, combined_audit, copy.deepcopy(current_messages)
    raise RuntimeError(f"{protocol_name} request loop exhausted")


def run_verification_round(
    env: Any,
    args: Any,
    *,
    output_dir: str,
    trial: int,
    turn: int,
    task: str,
    task_description: str,
    execution_evidence: str,
    runtime_stderr: str,
    visible_images: list[str],
    student_decision_prompt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the action prompt after action-owned PT and GT-delta verification."""
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Unsupported PGSVP-v9 task: {task}")
    root = (
        Path(output_dir)
        / "pgsvp_v9_verify_traces"
        / f"trial_{trial:02d}"
        / f"turn_{turn:02d}"
    )
    root.mkdir(parents=True, exist_ok=False)
    global_seed = int(os.environ.get("CAPX_GLOBAL_SEED", "0"))
    prior_verified_state = _load_prior_verified_state(
        output_dir=output_dir,
        trial=trial,
        turn=turn,
    )
    base_prompt_sha256 = _sha256(_to_jsonable(student_decision_prompt))
    pt_request_prompt = _build_student_pt_prompt(
        student_decision_prompt,
        prior_verified_state,
    )

    pt_content, pt_result, pt_audit, accepted_pt_request = _request_until_valid(
        args,
        pt_request_prompt,
        seed=global_seed + trial * 1000 + PT_REQUEST_SEED_OFFSET + turn,
        max_tokens=1800,
        protocol_name="PT worksheet",
        parser=_normalize_pt,
    )
    pt, pt_normalization = pt_result
    canonical_pt_response = _canonical_pt_response(pt)
    privileged = _privileged_packet(env, task=task)

    prompt_path = Path(
        os.environ.get(
            "CAPX_PGSVP_V9_PROMPT_PATH",
            Path(__file__).with_name("pgsvp_v9_verifier_prompt.md"),
        )
    )
    schema_path = Path(
        os.environ.get(
            "CAPX_PGSVP_V9_SCHEMA_PATH",
            Path(__file__).with_name("pgsvp_v9_verifier_schema.json"),
        )
    )
    prompt_text = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    verifier_content, verifier_result, verifier_audit, _ = _request_until_valid(
        args,
        _verifier_messages(
            prompt_text=prompt_text,
            schema=schema,
            task_description=task_description,
            execution_evidence=execution_evidence,
            pt=pt,
            privileged=privileged,
        ),
        seed=global_seed + trial * 1000 + VERIFIER_REQUEST_SEED_OFFSET + turn,
        max_tokens=1800,
        protocol_name="GT verifier",
        parser=lambda content: _canonicalize_verifier_output(
            content, privileged
        ),
    )
    public_message, parsed, interface_audit = verifier_result
    verified_candidate = parsed["public_student_guidance"]["fields"]
    verified_state = {
        field: verified_candidate[field] for field in PUBLIC_FIELDS
    }
    corrections = _pt_gt_corrections(pt, verified_state)
    reconstructed_state = _apply_pt_gt_delta(pt, corrections)
    if reconstructed_state != verified_state:
        raise RuntimeError("PT plus verifier delta did not reconstruct verified GT")
    delta_message = _render_delta_message(corrections)
    _validate_delta_message(delta_message, corrections)
    public_bridge, public_bridge_sha256 = _load_public_bridge()
    materialized_state_response = _canonical_pt_response(reconstructed_state)
    if materialized_state_response != _canonical_pt_response(verified_state):
        raise RuntimeError("Materialized state rendering differs from direct GT")
    action_prompt = _compose_student_action_prompt(
        student_decision_prompt=student_decision_prompt,
        materialized_state_response=materialized_state_response,
        bridge=public_bridge,
    )
    deterministic_alignment = _validate_private_gt_alignment(parsed, privileged)
    record = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "trial": trial,
        "turn": turn,
        "pt": pt,
        "prior_verified_state": prior_verified_state,
        "prior_verified_state_sha256": _sha256(prior_verified_state),
        "pt_normalization": pt_normalization,
        "pt_raw_response": pt_content,
        "pt_canonical_response": canonical_pt_response,
        "pt_canonical_response_sha256": hashlib.sha256(
            canonical_pt_response.encode("utf-8")
        ).hexdigest(),
        "pt_audit": pt_audit,
        "base_student_decision_prompt_sha256": base_prompt_sha256,
        "accepted_pt_request_sha256": _sha256(
            _to_jsonable(accepted_pt_request)
        ),
        "pt_extraction_request_visible_to_action": False,
        "prior_verified_state_visible_to_action": False,
        "closed_value_enums_visible_to_action": False,
        "pt_request_embedded_action_directive_occurrences": 0,
        "schema_retry_history_visible_to_action": False,
        "privileged_packet": privileged,
        "privileged_packet_sha256": _sha256(privileged),
        "verifier_raw_response": verifier_content,
        "verifier_parsed": parsed,
        "verifier_interface_audit": interface_audit,
        "verifier_audit": verifier_audit,
        "deterministic_private_alignment": deterministic_alignment,
        "deterministic_private_alignment_passed": True,
        "verified_state": verified_state,
        "verified_state_sha256": _sha256(verified_state),
        "fact_corrections": corrections,
        "fact_correction_count": len(corrections),
        "delta_message": delta_message,
        "delta_message_sha256": hashlib.sha256(
            delta_message.encode("utf-8")
        ).hexdigest(),
        "pt_plus_delta_equals_verified_state": reconstructed_state
        == verified_state,
        "materialized_state": reconstructed_state,
        "materialized_state_sha256": _sha256(reconstructed_state),
        "materialized_state_response": materialized_state_response,
        "materialized_state_response_sha256": hashlib.sha256(
            materialized_state_response.encode("utf-8")
        ).hexdigest(),
        "materialized_state_equals_verified_state": reconstructed_state
        == verified_state,
        "direct_gt_render_equals_materialized_render": (
            materialized_state_response == _canonical_pt_response(verified_state)
        ),
        "public_message": public_message,
        "public_message_sha256": hashlib.sha256(
            public_message.encode("utf-8")
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "public_bridge_text": public_bridge,
        "public_bridge_sha256": public_bridge_sha256,
        "student_action_prompt_sha256": _sha256(_to_jsonable(action_prompt)),
        "pt_extraction_context_isolated_from_action_history": True,
        "pt_owned_by_action_student": True,
        "pt_persisted_as_assistant_message": False,
        "raw_pt_visible_to_action": False,
        "delta_visible_to_action": False,
        "verifier_delta_retained_in_private_audit": True,
        "action_delivery_is_materialized_state_only": True,
        "materialized_state_persisted_as_assistant_message": True,
        "full_gt_report_injected": False,
        "materialized_verified_state_injected": True,
        "runtime_stderr_sha256": hashlib.sha256(
            runtime_stderr.encode("utf-8")
        ).hexdigest(),
        "visible_image_count_in_action_context": len(visible_images),
        "public_bridge_is_materialized_factual_interface": True,
        "contains_coordinates_public": False,
        "contains_reward_or_completion_label_public": False,
        "contains_action_advice_public": False,
        "public_local_sanitizer": {
            "version": "pgsvp_v9_public_sanitizer_v2",
            "passed": True,
        },
    }
    (root / "verification.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return action_prompt
