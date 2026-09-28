"""Agilex multi-turn trial execution with LLM error-feedback retries.

Copy of capx.envs.trial with a stronger multi-turn loop: when execution
fails (non-empty stderr / sandbox failure), stderr is sent back to the LLM
and the model is prompted to REGENERATE fixed code until success or limits.
"""

from __future__ import annotations

import base64
import copy
import gc
import glob
import io
import json
import os
import re
import time
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.agilex_tennis_success import evaluate_tennis_task_success
from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.integrations.agilex.control import AgilexControlApi

from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = int(os.environ.get("AGILEX_MULTITURN_LIMIT", "15"))
AGILEX_ERROR_RETRY_LIMIT = int(os.environ.get("AGILEX_ERROR_RETRY_LIMIT", "8"))

_AGILEX_ERROR_RETRY_SUFFIX = (
    "\n\nCRITICAL: The last execution FAILED (non-empty stderr and/or sandbox failure). "
    "Common Agilex failures include RuntimeError from goto_pose joint jumps, IK failures, "
    "TimeoutError from joint convergence, and perception errors. "
    "Because this is a real robot, the failed motion may have disturbed the tennis ball "
    "or box; treat all previously computed object poses and grasp poses as stale, "
    "re-perceive the current scene, and recompute box_pos/drop_pos/grasp_pos/grasp_quat. "
    "Read stderr carefully and respond with REGENERATE plus a complete corrected Python "
    "program in a ```python ... ``` fenced block. Do NOT respond FINISH while errors remain."
)

_AGILEX_TASK_RETRY_SUFFIX = (
    "\n\nCRITICAL: The code executed without Python errors, but the task still FAILED "
    "(tennis ball not confirmed in the box). Treat this as failure and regenerate a better "
    "plan. Re-perceive current ball/box state, then correct grasp Z/XY alignment, increase "
    "descent granularity, and use safer transfer waypoints."
)

_AGILEX_STRATEGY_SHIFT_SUFFIX = (
    "\n\nSTRATEGY SHIFT REQUIRED: Multiple previous execution attempts have failed. "
    "Do not make a tiny local edit. Change the plan more substantially while staying safe. "
    "Examples: choose a different grasp yaw/tilt candidate through sample_grasp_pose, increase "
    "pre-grasp height and safe_z clearance, split XY/Z motion into more conservative waypoints, "
    "avoid reusing a risky diagonal transfer, re-perceive the current ball/box, and rebuild the "
    "pick-place sequence from the current scene. Preserve safety guards and prefer failing safely "
    "over forcing a large IK jump."
)


def _agilex_execution_failed(info_step: dict[str, Any]) -> bool:
    return info_step.get("sandbox_rc", 1) != 0 or bool(
        str(info_step.get("stderr", "")).strip()
    )


def _agilex_task_failed(info_step: dict[str, Any]) -> bool:
    return info_step.get("task_completed", None) is False


def _agilex_needs_retry(info_step: dict[str, Any]) -> bool:
    return _agilex_execution_failed(info_step) or _agilex_task_failed(info_step)


def _looks_like_python_start(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    starters = (
        "import ",
        "from ",
        "def ",
        "class ",
        "#",
        "open_gripper(",
        "close_gripper(",
        "goto_pose(",
        "move_to_joints(",
        "get_object_pose(",
        "sample_grasp_pose(",
        "time.sleep(",
        "for ",
        "while ",
        "if ",
        "try:",
    )
    if stripped.startswith(starters):
        return True
    if "=" in stripped and not stripped.lower().startswith(("i ", "i'", "i\u2019")):
        return True
    return False


def _extract_agilex_code(content: str | None) -> list[str]:
    """Extract executable Python and drop model prose from Agilex responses."""
    blocks = _extract_code(content or "")
    cleaned_blocks: list[str] = []
    for block in blocks:
        token_match = re.search(
            r"(?m)(import\s+|from\s+|def\s+|class\s+|#|open_gripper\(|close_gripper\(|goto_pose\(|move_to_joints\(|get_object_pose\(|sample_grasp_pose\(|time\.sleep\(|for\s+|while\s+|if\s+|try:)",
            block,
        )
        if token_match is not None:
            token_cleaned = block[token_match.start():].strip()
            if token_cleaned != block.strip():
                print("[agilex_multiturn] Stripped non-code prefix from model response")
            cleaned_blocks.append(token_cleaned)
            continue

        lines = block.splitlines()
        start_idx = 0
        for idx, line in enumerate(lines):
            if _looks_like_python_start(line):
                start_idx = idx
                break
        else:
            start_idx = 0
        cleaned = "\n".join(lines[start_idx:]).strip()
        if cleaned != block.strip():
            print("[agilex_multiturn] Stripped non-code prefix from model response")
        cleaned_blocks.append(cleaned)
    return cleaned_blocks


def _as_jsonable_vector(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _capture_agilex_initial_scene(env: CodeExecutionEnvBase) -> dict[str, Any]:
    """Perceive current initial ball/box positions before code generation."""
    scene: dict[str, Any] = {"ok": False, "objects": {}, "errors": {}}
    api = getattr(env, "_apis", {}).get("AgilexControlApi")
    if api is None:
        scene["errors"]["api"] = "AgilexControlApi is not available"
        return scene

    for object_name in ("tennis ball", "cardboard box"):
        try:
            position, quaternion = api.get_object_pose(object_name)
            scene["objects"][object_name] = {
                "position": _as_jsonable_vector(position),
                "quaternion_wxyz": _as_jsonable_vector(quaternion),
            }
        except Exception as exc:
            scene["errors"][object_name] = repr(exc)

    scene["ok"] = bool(scene["objects"])
    return scene


def _agilex_success_memory_path(config: dict[str, Any]) -> str:
    return str(
        config.get(
            "agilex_success_memory_path",
            "./outputs/agilex_tennis_ball_top_camera_success_memory.json",
        )
    )


def _agilex_success_memory_enabled(config: dict[str, Any]) -> bool:
    env_value = os.environ.get("AGILEX_SUCCESS_MEMORY")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(config.get("agilex_success_memory_enabled", False))


def _agilex_failure_vision_enabled(config: dict[str, Any]) -> bool:
    env_value = os.environ.get("AGILEX_FAILURE_VISION")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(config.get("agilex_failure_vision_enabled", False))


def _agilex_failure_vision_max_frames(config: dict[str, Any]) -> int:
    env_value = os.environ.get("AGILEX_FAILURE_VISION_MAX_FRAMES")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return max(1, int(config.get("agilex_failure_vision_max_frames", 4)))


def _agilex_strategy_shift_enabled(config: dict[str, Any]) -> bool:
    env_value = os.environ.get("AGILEX_STRATEGY_SHIFT")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(config.get("agilex_strategy_shift_enabled", False))


def _agilex_strategy_shift_after_failures(config: dict[str, Any]) -> int:
    env_value = os.environ.get("AGILEX_STRATEGY_SHIFT_AFTER_FAILURES")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return max(1, int(config.get("agilex_strategy_shift_after_failures", 2)))


def _agilex_strategy_shift_base_failures(config: dict[str, Any]) -> int:
    env_value = os.environ.get("AGILEX_STRATEGY_SHIFT_BASE_FAILURES")
    if env_value is not None:
        try:
            return max(0, int(env_value))
        except ValueError:
            pass
    return max(0, int(config.get("agilex_strategy_shift_base_failures", 0)))


def _agilex_strategy_shift_active(config: dict[str, Any], failures: int) -> bool:
    return (
        _agilex_strategy_shift_enabled(config)
        and failures >= _agilex_strategy_shift_after_failures(config)
    )


def _agilex_strategy_shift_prompt(config: dict[str, Any], failures: int) -> str:
    if not _agilex_strategy_shift_active(config, failures):
        return ""
    return (
        f"\n\nThe last {failures} execution attempt(s) failed. "
        f"{_AGILEX_STRATEGY_SHIFT_SUFFIX}"
    )


def _load_agilex_success_memory(config: dict[str, Any]) -> dict[str, Any] | None:
    path = _agilex_success_memory_path(config)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"[agilex_memory] Failed to load success memory {path}: {exc}")
        return None


def _vector_delta(current: dict[str, Any], previous: dict[str, Any], name: str) -> list[float] | None:
    try:
        cur = np.asarray(current["objects"][name]["position"], dtype=np.float64)
        prev = np.asarray(previous["initial_scene"]["objects"][name]["position"], dtype=np.float64)
        return (cur - prev).reshape(-1).tolist()
    except Exception:
        return None


def _append_agilex_success_memory_prompt(
    obs: dict[str, Any],
    config: dict[str, Any],
    current_scene: dict[str, Any],
) -> dict[str, Any] | None:
    """Append previous-success pose delta and code to the initial LLM prompt."""
    memory = _load_agilex_success_memory(config)
    if memory is None:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nNo previous successful Agilex tennis-ball trajectory memory is available yet. "
            "Use the current perception from get_object_pose/sample_grasp_pose and obey all safety constraints."
        )
        return None

    ball_delta = _vector_delta(current_scene, memory, "tennis ball")
    box_delta = _vector_delta(current_scene, memory, "cardboard box")
    current_objects = current_scene.get("objects", {})
    previous_objects = memory.get("initial_scene", {}).get("objects", {})
    previous_code = str(memory.get("final_code", "")).strip()
    if len(previous_code) > 6000:
        previous_code = previous_code[-6000:]

    obs["full_prompt"][-1]["content"][0]["text"] += f"""

Previous successful Agilex run memory is available. Use it as guidance, not as a blind replay.

Previous successful initial scene:
{json.dumps(previous_objects, indent=2)}

Current initial scene:
{json.dumps(current_objects, indent=2)}

Current minus previous initial position deltas:
- tennis ball delta xyz: {ball_delta}
- cardboard box delta xyz: {box_delta}

How to use this memory:
- The previous successful code below contains useful path structure and key parameters.
- Adapt the previous successful path by considering the tennis-ball and box initial position deltas.
- Still call get_object_pose("cardboard box") and sample_grasp_pose("tennis ball") in the new code; do not hard-code stale positions.
- If the ball delta is large, trust current perception more than the old trajectory.
- Preserve the safety pattern from the successful run: pre-grasp, vertical descent, close, lift, safe_z transfer, vertical drop, open, retreat.

Previous successful final code:
```python
{previous_code}
```
"""
    print(
        "[agilex_memory] Added previous success memory to prompt "
        f"(ball_delta={ball_delta}, box_delta={box_delta})"
    )
    return memory


def _save_agilex_success_memory(
    config: dict[str, Any],
    trial: int,
    initial_scene: dict[str, Any],
    final_code: str,
    raw_code: str,
    info_step: dict[str, Any],
) -> None:
    if not initial_scene.get("ok"):
        return
    path = _agilex_success_memory_path(config)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    memory = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trial": trial,
        "initial_scene": initial_scene,
        "final_code": final_code,
        "raw_code": raw_code,
        "sandbox_rc": info_step.get("sandbox_rc"),
        "task_completed": info_step.get("task_completed"),
    }
    with open(path, "w") as f:
        json.dump(memory, f, indent=2)
    print(f"[agilex_memory] Saved successful trajectory memory to {path}")


def _encode_frame_base64(frame: np.ndarray) -> str:
    pil_img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _latest_side_snapshot_base64(env: CodeExecutionEnvBase) -> str | None:
    """Return side-camera latest snapshot as data URL, if available."""
    low_level = getattr(env, "low_level_env", None)
    candidates: list[str] = []
    if low_level is not None:
        side_output_dir = getattr(low_level, "side_camera_output_dir", None)
        if side_output_dir:
            candidates.append(os.path.join(os.path.dirname(str(side_output_dir)), "latest_rgb.png"))
    # Fallback to preview watcher output used by pipeline scripts.
    candidates.append("outputs/agilex_side_camera/latest_rgb.png")

    for latest_path in candidates:
        if not os.path.exists(latest_path):
            continue
        try:
            with open(latest_path, "rb") as f:
                raw = f.read()
            if not raw:
                continue
        except Exception:
            continue
        return f"data:image/png;base64,{base64.b64encode(raw).decode('utf-8')}"
    return None


def _side_snapshot_index(path: str) -> int:
    name = os.path.basename(path)
    try:
        return int(name.split("_", 1)[0])
    except Exception:
        return -1


def _collect_side_failure_snapshots(
    env: CodeExecutionEnvBase,
    *,
    min_images: int = 5,
    max_images: int = 6,
) -> list[tuple[str, str]]:
    """Collect latest + contact-related side snapshots for failure diagnosis."""
    low_level = getattr(env, "low_level_env", None)
    candidate_dirs: list[str] = []
    if low_level is not None:
        side_output_dir = getattr(low_level, "side_camera_output_dir", None)
        if side_output_dir:
            candidate_dirs.append(str(side_output_dir))
    candidate_dirs.append("outputs/agilex_side_camera/task")

    all_paths: list[str] = []
    for directory in candidate_dirs:
        all_paths.extend(glob.glob(os.path.join(directory, "*.png")))
    if not all_paths:
        return []
    all_paths = sorted(set(all_paths), key=lambda p: os.path.getmtime(p))

    latest_by_tag: dict[str, str] = {}
    for tag in ("sample_grasp_pose", "close_gripper", "open_gripper"):
        tagged = [p for p in all_paths if f"_{tag}.png" in os.path.basename(p)]
        if tagged:
            latest_by_tag[tag] = tagged[-1]

    selected: list[tuple[str, str]] = []
    selected_paths: set[str] = set()

    def add(path: str, label: str) -> None:
        if path in selected_paths:
            return
        try:
            with open(path, "rb") as f:
                raw = f.read()
            if not raw:
                return
        except Exception:
            return
        selected.append((label, f"data:image/png;base64,{base64.b64encode(raw).decode('utf-8')}"))
        selected_paths.add(path)

    # 1) Always include latest side frame.
    add(all_paths[-1], "latest_side")
    # 2) Key semantic frames.
    if "sample_grasp_pose" in latest_by_tag:
        add(latest_by_tag["sample_grasp_pose"], "sample_grasp_pose")
    if "close_gripper" in latest_by_tag:
        close_path = latest_by_tag["close_gripper"]
        add(close_path, "close_gripper_contact")
        close_idx = _side_snapshot_index(close_path)
        if close_idx >= 0:
            by_idx = { _side_snapshot_index(p): p for p in all_paths if _side_snapshot_index(p) >= 0 }
            for delta, label in [(-2, "pre_contact_2"), (-1, "pre_contact_1"), (1, "post_contact_1")]:
                cand = by_idx.get(close_idx + delta)
                if cand is not None and "_goto_pose.png" in os.path.basename(cand):
                    add(cand, label)
    if "open_gripper" in latest_by_tag:
        add(latest_by_tag["open_gripper"], "release_phase")

    if len(selected) < min_images:
        for path in reversed(all_paths):
            if len(selected) >= min_images:
                break
            if path in selected_paths:
                continue
            add(path, f"recent_extra_{len(selected) + 1}")

    return selected[:max_images]


def _query_failure_referee_feedback(
    env: CodeExecutionEnvBase,
    visual_differencing_args: ModelQueryArgs,
    info_step: dict[str, Any],
    turn_frames: list[np.ndarray] | None,
) -> str | None:
    """Use a VLM referee to analyze failure frames and return text guidance."""
    if visual_differencing_args.model not in VLM_MODELS:
        print(
            "[agilex_referee] visual_differencing_model does not support images: "
            f"{visual_differencing_args.model}"
        )
        return None

    top_sampled = _sample_turn_frames(turn_frames or [], 4)
    side_frames = _collect_side_failure_snapshots(env, min_images=5, max_images=6)
    if not top_sampled and not side_frames:
        print("[agilex_referee] no failure frames available for referee model")
        return None

    stderr_text = str(info_step.get("stderr", "")).strip()
    stdout_text = str(info_step.get("stdout", "")).strip()
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Analyze this failed Agilex tennis-ball-in-box attempt and return concise guidance.\n"
                f"- task_completed={info_step.get('task_completed')}\n"
                f"- sandbox_rc={info_step.get('sandbox_rc')}\n"
                f"- stderr (possibly empty):\n{stderr_text or '(empty)'}\n"
                f"- stdout excerpt:\n{stdout_text[-2000:] if stdout_text else '(empty)'}\n\n"
                "Output ONLY plain text with exactly two sections:\n"
                "1) ROOT_CAUSE: 1-3 bullet points\n"
                "2) FIX_PLAN: 3-6 actionable bullet points for next regenerated code\n"
                "Focus on grasp contact quality, grasp height/alignment, and safe transfer waypoints."
            ),
        }
    ]

    for ordinal, (_idx, frame) in enumerate(top_sampled, start=1):
        user_content.append({"type": "text", "text": f"Top failure frame {ordinal}/{len(top_sampled)}"})
        user_content.append({"type": "image_url", "image_url": {"url": _encode_frame_base64(frame)}})

    for label, image_url in side_frames:
        user_content.append({"type": "text", "text": f"Side keyframe: {label}"})
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a strict robot failure referee. "
                "Use visual evidence and logs to diagnose why grasp/place failed. "
                "Prioritize concrete, safe, code-level fixes."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    try:
        referee_out = _query_model(visual_differencing_args, prompt)
        text = str(referee_out.get("content", "")).strip()
        if not text:
            return None
        print(
            "[agilex_referee] generated visual failure diagnosis "
            f"using {visual_differencing_args.model}"
        )
        print("[agilex_referee] turn failure analysis + fix plan:")
        print("-" * 80)
        print(text)
        print("-" * 80)
        return text
    except Exception as exc:
        print(f"[agilex_referee] referee query failed: {exc}")
        return None


def _append_side_snapshot_for_failure(
    env: CodeExecutionEnvBase,
    decision_prompt: list[dict[str, Any]],
    info_step: dict[str, Any],
    *,
    allow_image_input: bool = True,
) -> None:
    """Attach latest side-camera image for failed turns."""
    if not _agilex_needs_retry(info_step):
        return
    side_frames = _collect_side_failure_snapshots(env, min_images=5, max_images=6)
    if not side_frames:
        side_base64 = _latest_side_snapshot_base64(env)
        if side_base64 is not None:
            side_frames = [("latest_side", side_base64)]
    if not side_frames:
        return
    content = decision_prompt[-1].setdefault("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
        decision_prompt[-1]["content"] = content
    content.append({
        "type": "text",
        "text": (
            "Side camera keyframes near this failure are attached below (latest + contact-related frames). "
            "Use them explicitly to correct grasp height/alignment before regenerating code: "
            "if gripper is above ball, lower grasp Z and increase descent steps; "
            "if gripper is off-center, add a small XY correction toward current perceived ball; "
            "if near table collision risk, increase safe_z and split motion into smaller waypoints."
        ),
    })
    if allow_image_input:
        for label, image_url in side_frames:
            content.append({
                "type": "text",
                "text": f"Side keyframe: {label}",
            })
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        print(f"[agilex_vision] Attached {len(side_frames)} side keyframes to failure prompt")
    else:
        labels = ", ".join(label for label, _ in side_frames)
        content.append({
            "type": "text",
            "text": (
                "Side keyframes were captured but model endpoint is text-only (no image input). "
                f"Captured labels: {labels}. "
                "Use this contact-stage sequence for correction: adjust grasp Z/XY alignment and descent steps."
            ),
        })
        print(
            "[agilex_vision] Model is text-only; side keyframes summarized as text "
            f"({len(side_frames)} captured)"
        )


def _sample_turn_frames(
    frames: list[np.ndarray],
    max_frames: int,
) -> list[tuple[int, np.ndarray]]:
    if not frames:
        return []
    count = min(max_frames, len(frames))
    if count == 1:
        indices = [len(frames) - 1]
    else:
        indices = np.linspace(0, len(frames) - 1, count, dtype=int).tolist()
    # np.linspace can duplicate indices for very short videos.
    unique_indices = list(dict.fromkeys(indices))
    return [(idx, frames[idx]) for idx in unique_indices]


def _append_agilex_failure_vision_frames(
    env: CodeExecutionEnvBase,
    decision_prompt: list[dict[str, Any]],
    config: dict[str, Any],
    info_step: dict[str, Any],
    turn_frames: list[np.ndarray] | None,
    *,
    allow_image_input: bool = True,
) -> None:
    """Append sampled failure frames to the multiturn prompt when enabled."""
    if not _agilex_failure_vision_enabled(config):
        return
    if not _agilex_needs_retry(info_step):
        return
    if not turn_frames:
        print("[agilex_vision] Failure vision enabled but no turn frames were recorded")
        return

    content = decision_prompt[-1].setdefault("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
        decision_prompt[-1]["content"] = content

    sampled = _sample_turn_frames(
        turn_frames, _agilex_failure_vision_max_frames(config)
    )
    if not sampled:
        return

    content.append({
        "type": "text",
        "text": (
            "\n\nFailure-vision frames from the top camera are attached below. "
            "Use them together with stderr to diagnose why the turn failed. "
            "Consider visual causes such as wrong approach angle, object moved by the arm, "
            "depth error, calibration offset, occlusion, gripper not aligned with the ball, "
            "or an unsafe/too-large motion. Then regenerate safer code if needed."
        ),
    })
    if allow_image_input:
        for ordinal, (frame_idx, frame) in enumerate(sampled, start=1):
            content.append({
                "type": "text",
                "text": f"Top camera failure frame {ordinal}/{len(sampled)} (turn frame index {frame_idx}):",
            })
            content.append({"type": "image_url", "image_url": {"url": _encode_frame_base64(frame)}})
        print(f"[agilex_vision] Attached {len(sampled)} failure frames to LLM prompt")
    else:
        content.append({
            "type": "text",
            "text": (
                "Top failure frames were captured but model endpoint is text-only (no image input). "
                f"Captured {len(sampled)} frame(s); rely on stderr and side keyframe sequence labels."
            ),
        })
        print(
            "[agilex_vision] Model is text-only; top failure frames summarized as text "
            f"({len(sampled)} captured)"
        )


def _get_agilex_control_api(env: CodeExecutionEnvBase) -> AgilexControlApi | None:
    apis = getattr(env, "_apis", {})
    for api in apis.values():
        if isinstance(api, AgilexControlApi):
            return api
    return None


def _apply_agilex_visual_success_check(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    info_step: dict[str, Any],
    reward: float,
) -> tuple[dict[str, Any], float]:
    if not config.get("agilex_visual_success_check", False):
        return info_step, reward
    control_api = _get_agilex_control_api(env)
    if control_api is None:
        print("[agilex_success] AgilexControlApi unavailable; skipping geometric success check")
        return info_step, reward

    ok, details = evaluate_tennis_task_success(control_api, config)
    info_step = dict(info_step)
    info_step["task_completed"] = ok
    info_step["agilex_visual_success"] = details
    reward = 1.0 if ok else 0.0
    geometric = details.get("geometric", details)
    xy = geometric.get("xy_distance_m")
    z_delta = geometric.get("z_delta_m")
    reason = details.get("reason", geometric.get("reason", "unknown"))
    if isinstance(xy, (int, float)) and isinstance(z_delta, (int, float)):
        print(
            f"[agilex_success] perception+geometry: {ok} "
            f"({reason}, xy={xy:.3f}m, z_delta={z_delta:.3f}m)"
        )
    else:
        print(f"[agilex_success] perception+geometry: {ok} ({details})")
    return info_step, reward


def _write_agilex_visual_success_report(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> None:
    details = info_step.get("agilex_visual_success")
    if not details or not config.get("output_dir"):
        return
    trial_dir = _trial_video_dir(config, trial, info_step, reward)
    os.makedirs(trial_dir, exist_ok=True)
    report_path = os.path.join(trial_dir, "agilex_visual_success.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    print(f"[agilex_success] wrote {report_path}")

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        description = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
        )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
) -> tuple[str, str | None, dict | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data)
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(str(obs["full_prompt"]))

    ensemble_data = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            out = _query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        out = _query_model(args, obs["full_prompt"])

    return out["content"], out["reasoning"], ensemble_data


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step["stdout"],
        console_stderr=info_step["stderr"],
    )

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    # Build decision prompt
    allow_image_input = args.model in VLM_MODELS
    if not allow_image_input:
        print(f"[agilex_vision] {args.model} is text-only; using text fallback for failure vision")

    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    _append_agilex_failure_vision_frames(
        env,
        decision_prompt,
        config,
        info_step,
        turn_frames,
        allow_image_input=allow_image_input,
    )
    _append_side_snapshot_for_failure(
        env,
        decision_prompt,
        info_step,
        allow_image_input=allow_image_input,
    )
    if (not allow_image_input) and _agilex_needs_retry(info_step):
        referee_feedback = _query_failure_referee_feedback(
            env,
            visual_differencing_args,
            info_step,
            turn_frames,
        )
        if referee_feedback:
            content = decision_prompt[-1].setdefault("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
                decision_prompt[-1]["content"] = content
            content.append({
                "type": "text",
                "text": (
                    "\n\nReferee model visual diagnosis (from top + side failure frames):\n"
                    f"{referee_feedback}"
                ),
            })

    # Query model
    multiturn_ensemble_entry = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            content = _query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        content = _query_model(args, decision_prompt)

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)

    # --- 1. Reset environment ---
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    # Reset the SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation.
    import signal
    remaining = signal.alarm(0)  # cancel current alarm
    if remaining > 0:
        signal.alarm(1000)  # restart fresh 1000s from now
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif use_video_diff and hasattr(env, "enable_video_capture"):
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    execution_failure_count = _agilex_strategy_shift_base_failures(config)
    if execution_failure_count > 0:
        print(
            "[agilex_multiturn] Loaded base failure count from outer attempts: "
            f"{execution_failure_count}"
        )
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model must be in the list of VLM models"
        )

    # --- 2. Capture initial visual feedback ---
    visual_feedback_imgs, visual_feedback_base64_history, task_description = (
        _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
    )
    agilex_initial_scene: dict[str, Any] = {"ok": False, "objects": {}, "errors": {}}
    if _agilex_success_memory_enabled(config):
        agilex_initial_scene = _capture_agilex_initial_scene(env)
        if agilex_initial_scene.get("ok"):
            print(
                "[agilex_memory] Captured current initial scene: "
                f"{json.dumps(agilex_initial_scene.get('objects', {}))}"
            )
        elif agilex_initial_scene.get("errors"):
            print(f"[agilex_memory] Initial scene capture warnings: {agilex_initial_scene['errors']}")
        _append_agilex_success_memory_prompt(obs, config, agilex_initial_scene)
    else:
        print("[agilex_memory] Success memory disabled; skipping initial scene memory capture")

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
    else:
        raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Parse initial code into blocks
    initial_blocks = _extract_agilex_code(raw_code)
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
    })

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff or _agilex_failure_vision_enabled(config))
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT:
        code = code_blocks[code_block_idx]
        code_block_idx += 1

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        obs_next, reward, terminated, truncated, info_step = env.step(code)
        if _agilex_needs_retry(info_step):
            execution_failure_count += 1
            print(
                "[agilex_multiturn] Execution failure count: "
                f"{execution_failure_count}"
            )
            if _agilex_strategy_shift_active(config, execution_failure_count):
                print(
                    "[agilex_multiturn] Strategy-shift mode active for next regeneration"
                )

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next

        # Multi-turn decision (Agilex: retry on execution errors)
        if multi_turn_prompt:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if (use_video_diff or _agilex_failure_vision_enabled(config)) and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            active_prompt = (
                multi_turn_prompt
                + _agilex_strategy_shift_prompt(config, execution_failure_count)
            )
            error_retry_attempt = 0
            while True:
                decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = (
                    _handle_multi_turn_step(
                        env, obs, args, config, visual_differencing_args,
                        active_prompt, code_blocks, code_block_idx, info_step,
                        task_description, visual_feedback_imgs, visual_feedback_base64_history,
                        stderr_history,
                        turn_frames=turn_frames,
                        wrist_turn_frames=wrist_turn_frames,
                        wrist_base64_history=wrist_base64_history,
                    )
                )

                if mt_ensemble is not None:
                    mt_ensemble["regeneration"] = num_regenerations + 1
                    multiturn_ensemble_data.append(mt_ensemble)

                execution_failed = _agilex_execution_failed(info_step)
                task_failed = _agilex_task_failed(info_step)

                if decision == "regenerate":
                    print("Model chose to regenerate code")
                    new_blocks = _extract_agilex_code(new_code)
                    all_responses.append({
                        "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                        "block_idx": [code_block_idx],
                        "code_blocks": new_blocks,
                        "decision": "regenerate",
                        "reasoning": mt_reasoning if mt_reasoning is not None else "",
                    })
                    del code_blocks[code_block_idx:]
                    del code_block_metadata[code_block_idx:]
                    code_blocks.extend(new_blocks)
                    code_block_metadata.extend(
                        [{"generation": num_regenerations + 1, "regenerated": True,
                          "regenerated_at_idx": code_block_idx}]
                        * len(new_blocks)
                    )
                    num_regenerations += 1
                    if partial_artifacts is not None:
                        partial_artifacts["num_regenerations"] = num_regenerations
                    break

                if decision == "finish" and not execution_failed and not task_failed:
                    all_responses.append({
                        "decision": "finish",
                        "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                    })
                    print("Model chose to finish")
                    num_finishes += 1
                    if partial_artifacts is not None:
                        partial_artifacts["num_finishes"] = num_finishes
                    break

                if (execution_failed or task_failed) and error_retry_attempt < AGILEX_ERROR_RETRY_LIMIT:
                    error_retry_attempt += 1
                    if execution_failed:
                        print(
                            "[agilex_multiturn] Execution failed; sending stderr back to LLM "
                            f"({error_retry_attempt}/{AGILEX_ERROR_RETRY_LIMIT})"
                        )
                    else:
                        print(
                            "[agilex_multiturn] Task failed without stderr; forcing regenerate "
                            f"({error_retry_attempt}/{AGILEX_ERROR_RETRY_LIMIT})"
                        )
                    all_responses.append({
                        "decision": "error_retry" if execution_failed else "task_retry",
                        "reasoning": mt_reasoning if mt_reasoning is not None else "",
                        "stderr": info_step.get("stderr", ""),
                        "attempt": error_retry_attempt,
                        "task_completed": info_step.get("task_completed"),
                    })
                    active_prompt = (
                        multi_turn_prompt
                        + (_AGILEX_ERROR_RETRY_SUFFIX if execution_failed else _AGILEX_TASK_RETRY_SUFFIX)
                        + _agilex_strategy_shift_prompt(
                            config, execution_failure_count
                        )
                    )
                    continue

                if decision == "finish" and execution_failed:
                    print(
                        "[agilex_multiturn] Stopping after FINISH with remaining execution errors"
                    )
                    all_responses.append({
                        "decision": "finish_with_errors",
                        "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                    })
                    num_finishes += 1
                    if partial_artifacts is not None:
                        partial_artifacts["num_finishes"] = num_finishes
                    break

                break

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    print("Code blocks done")

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    info_step, reward = _apply_agilex_visual_success_check(env, config, info_step, reward)
    _write_agilex_visual_success_report(config, trial, info_step, reward)

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)

    success = info_step["sandbox_rc"] == 0
    if (
        _agilex_success_memory_enabled(config)
        and success
        and info_step.get("task_completed", False)
    ):
        _save_agilex_success_memory(
            config, trial, agilex_initial_scene, final_code, raw_code, info_step
        )

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    if not hasattr(env.low_level_env, "handle"):
        return
    handle = env.low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = (
            obs["full_prompt"][-1]["content"][0]["text"].format(
                libero_environment_goal=goal
            )
        )
