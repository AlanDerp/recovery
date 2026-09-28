"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import base64
import copy
import gc
import io
import json
import os
import time
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase

def _restart_trial_alarm_after_reset() -> int | None:
    """Restart an active trial alarm without replacing its configured limit."""
    import signal

    remaining = signal.alarm(0)
    if remaining <= 0:
        return None
    configured_timeout = os.environ.get("CAPX_TRIAL_TIMEOUT_SECONDS")
    execution_timeout = (
        max(1, int(float(configured_timeout)))
        if configured_timeout
        else remaining
    )
    signal.alarm(execution_timeout)
    return execution_timeout


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


MULTITURN_LIMIT = 10

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
        obs["_capx_initial_native_report"] = feedback
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


def _replace_native_report_in_prompt(
    prompt: list[dict[str, Any]],
    *,
    native_report: str,
    replacement: str,
) -> int:
    """Replace native M3 report carrier; returns match count."""
    matches = 0
    for message in prompt:
        content = message.get("content")
        if isinstance(content, str):
            count = content.count(native_report)
            if count:
                message["content"] = content.replace(native_report, replacement)
                matches += count
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                count = text.count(native_report)
                if count:
                    item["text"] = text.replace(native_report, replacement)
                    matches += count
    return matches


def _run_ideal_stuck_outer_regen(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
    env: CodeExecutionEnvBase,
    *,
    ideal_rt: Any,
    trial: int,
) -> tuple[str | None, dict[str, Any]]:
    """L3 truth-anchored regen after Ideal L1 Stuck (no continuous GT in L1)."""
    from capx.verification import closed_gt
    from capx.verification.backtrack_retry_replan import (
        build_stuck_outer_regen_feedback,
        ideal_stuck_outer_regen_enabled,
        stuck_report_from_runtime_acceptance,
    )
    from capx.verification.ideal_tk_v4 import render_verified_initial_state

    meta: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "protocol": "backtrack_retry_replan_v1",
    }
    if not ideal_stuck_outer_regen_enabled():
        meta["skip_reason"] = "CAPX_IDEAL_STUCK_OUTER_REGEN=0"
        return None, meta
    if config.get("use_oracle_code"):
        meta["skip_reason"] = "oracle_code"
        return None, meta
    if config.get("use_parallel_ensemble"):
        meta["skip_reason"] = "parallel_ensemble"
        return None, meta
    native = obs.get("_capx_initial_native_report")
    if not isinstance(native, str) or not native.strip():
        meta["skip_reason"] = "no_native_report"
        return None, meta

    meta["attempted"] = True
    scene = closed_gt.cube_scene_spec()
    snap = closed_gt.extract_cube_snapshot(env)
    verified = render_verified_initial_state(scene, snap)
    stuck = stuck_report_from_runtime_acceptance(
        ideal_rt.acceptance, message=ideal_rt.message
    )
    meta["stuck"] = stuck.as_dict()
    feedback = build_stuck_outer_regen_feedback(
        verified_report=verified,
        stuck=stuck,
        ideal_message=ideal_rt.message,
    )
    prompt = copy.deepcopy(obs["full_prompt"])
    matches = _replace_native_report_in_prompt(
        prompt, native_report=native, replacement=feedback
    )
    if matches != 1:
        # Fallback: append diagnosis on the terminal user turn.
        if not prompt or prompt[-1].get("role") != "user":
            meta["skip_reason"] = f"prompt_carrier_matches={matches}"
            return None, meta
        content = prompt[-1].get("content")
        if isinstance(content, str):
            prompt[-1]["content"] = f"{content}\n\n{feedback}"
        elif isinstance(content, list):
            content.append({"type": "text", "text": feedback})
        else:
            meta["skip_reason"] = "unextendable_prompt"
            return None, meta
        meta["carrier_mode"] = "append"
    else:
        meta["carrier_mode"] = "replace_native_report"

    out_dir = config.get("output_dir")
    if out_dir:
        stuck_dir = os.path.join(
            out_dir, "ideal_tk_v4_traces", f"trial_{trial:02d}", "stuck_outer_regen"
        )
        os.makedirs(stuck_dir, exist_ok=True)
        with open(os.path.join(stuck_dir, "stuck_regen_prompt.txt"), "w") as f:
            f.write(str(prompt))
        with open(os.path.join(stuck_dir, "stuck_report.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)

    print("Ideal L1 Stuck → outer regen (truth-anchored L3)")
    out = _query_model(args, prompt)
    raw = out.get("content") or ""
    if out_dir:
        with open(
            os.path.join(
                config["output_dir"],
                "ideal_tk_v4_traces",
                f"trial_{trial:02d}",
                "stuck_outer_regen",
                "stuck_regen_response.txt",
            ),
            "w",
        ) as f:
            f.write(raw)
    if not str(raw).strip():
        meta["skip_reason"] = "empty_regen"
        return None, meta
    meta["ok"] = True
    meta["reasoning"] = out.get("reasoning") or ""
    return str(raw), meta


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
    trial: int | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None, bool]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt,
         post_miss_clear)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    n_prefix = 1
    if trial is not None and os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY", "").strip():
        from capx.verification import libero_g17_code_prefix_replay as code_prefix

        try:
            n_prefix = int(code_prefix.load_package().trial_entry(trial).n_prefix)
        except Exception:
            n_prefix = 1

    # Default: full executed history. Post-miss clear may shrink to prefix only.
    history_end = code_block_idx
    executed_code = "\n".join(code_blocks[:history_end])
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

    if os.environ.get("CAPX_LIBERO_G17_FG", "0") == "1":
        if trial is None:
            raise RuntimeError("LIBERO G17 FG requires a trial identifier")
        from capx.verification.libero_g17_takeover import (
            consume_takeover_bundle,
            run_takeover as run_libero_g17_fg,
            _apply_goal_referent_takeover,
            _replace_report_once,
        )
        from capx.verification import libero_g17_post_miss_clear as post_miss

        decision_prompt = run_libero_g17_fg(
            env,
            output_dir=config["output_dir"],
            trial=trial,
            turn=code_block_idx,
            task=os.environ.get("CAPX_CURRENT_TASK_NAME", ""),
            native_report=differencing_feedback,
            native_decision_prompt=decision_prompt,
            private_task_completed=bool(info_step.get("task_completed", False)),
        )
        bundle = consume_takeover_bundle(
            config["output_dir"], trial, code_block_idx
        )
        post_miss_clear = bool(bundle and bundle.get("post_miss_clear"))
        if post_miss_clear and bundle is not None:
            # Drop thrash history from the student prompt; world state stays.
            if len(visual_feedback_base64_history) > 1:
                del visual_feedback_base64_history[:-1]
            stderr_history.clear()
            executed_code = "\n".join(code_blocks[: max(0, min(n_prefix, code_block_idx))])
            complete_multi_turn_prompt = multi_turn_prompt.format(
                executed_code=executed_code,
                console_stdout=info_step.get("stdout", ""),
                console_stderr="",
            )
            visual_feedback_base64 = (
                visual_feedback_base64_history[-1]
                if visual_feedback_base64_history
                else None
            )
            if not config["use_visual_feedback"]:
                visual_feedback_base64 = None
            if args.use_legacy_multi_turn_decision_prompt:
                decision_prompt = _build_multi_turn_decision_prompt_legacy(
                    obs,
                    complete_multi_turn_prompt,
                    visual_feedback_base64,
                    differencing_feedback,
                    is_video_feedback=is_video_feedback,
                )
            else:
                decision_prompt = _build_multi_turn_decision_prompt(
                    obs,
                    complete_multi_turn_prompt,
                    visual_feedback_base64,
                    differencing_feedback,
                    is_video_feedback=is_video_feedback,
                )
            decision_prompt = _replace_report_once(
                decision_prompt,
                original=differencing_feedback,
                replacement=str(bundle["replacement"]),
            )
            if post_miss.goal_lock_between_to_ordinal_enabled():
                from capx.verification import libero_g17_takeover as g17_takeover

                binding = g17_takeover._PRIVATE_BINDINGS.get(
                    g17_takeover._cache_key(config["output_dir"], trial)
                )
                attachments = g17_takeover._PRIVATE_ATTACHMENTS.get(
                    g17_takeover._cache_key(config["output_dir"], trial)
                )
                attachment = attachments[-1] if attachments else None
                if binding is not None and attachment is not None:
                    decision_prompt, _goal_meta = _apply_goal_referent_takeover(
                        decision_prompt,
                        binding=binding,
                        attachment=attachment,
                        env=env,
                    )
            print(
                f"[libero-g17-fg] post-miss-clear rebuilt prompt "
                f"trial={trial} turn={code_block_idx} n_prefix={n_prefix}",
                flush=True,
            )
    else:
        post_miss_clear = False

    # Query model
    multiturn_ensemble_entry = None
    prefix_forced = None
    if trial is not None and os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY", "").strip():
        from capx.verification import libero_g17_code_prefix_replay as code_prefix

        pkg = code_prefix.load_package()
        prefix_forced = code_prefix.prefix_regenerate_raw_content(
            pkg, trial, executed_block_count=code_block_idx
        )
        if prefix_forced is not None:
            code_prefix.write_prefix_hit_audit(
                config["output_dir"],
                trial=trial,
                turn=code_block_idx,
                code=pkg.trial_entry(trial).code_blocks[code_block_idx],
                package=pkg,
            )
            print(
                f"[libero-g17-prefix] trial={trial} turn={code_block_idx} "
                f"source=code_prefix_replay package={pkg.sha256[:12]}"
            )
            content = {"content": prefix_forced, "reasoning": None}
            reasoning = None
            decision, new_code = _parse_multi_turn_decision(content["content"])
            return decision, new_code, reasoning, None, decision_prompt, post_miss_clear
        from capx.verification.libero_g17_mechanism_contract import (
            forbid_live_student_llm,
        )

        forbid_live_student_llm(
            trial=trial,
            turn=code_block_idx,
            reason="code-prefix miss; refusing live regenerate",
        )

    from capx.verification.libero_g17_mechanism_contract import (
        forbid_live_student_llm as _forbid_live_student_llm,
    )

    _forbid_live_student_llm(
        trial=trial,
        turn=code_block_idx,
        reason="live regenerate is not a mechanism-class path",
    )

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

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt, post_miss_clear



def _pending_block_is_code_prefix(trial: int | None, block_idx: int) -> bool:
    if trial is None:
        return False
    if not os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY", "").strip():
        return False
    from capx.verification import libero_g17_code_prefix_replay as code_prefix

    pkg = code_prefix.load_package()
    entry = pkg.trial_entry(trial)
    return int(block_idx) < len(entry.code_blocks)


def _run_identity_verify_regen(
    env: CodeExecutionEnvBase,
    *,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    obs: dict[str, Any],
    verdict: dict[str, Any],
    trial: int,
    visual_feedback_base64_history: list[str],
) -> tuple[str, str | None, list]:
    """FG verify mismatch: ask for regenerate with identity-lock report (no env step)."""
    from capx.verification import libero_g17_takeover as g17

    verify_report = g17.render_identity_verify_report(verdict)
    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step.get("stdout", ""),
        console_stderr=info_step.get("stderr", ""),
    )
    visual_feedback_base64 = None
    if config.get("use_visual_feedback") and visual_feedback_base64_history:
        visual_feedback_base64 = visual_feedback_base64_history[-1]
    if args.use_legacy_multi_turn_decision_prompt:
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs,
            complete_multi_turn_prompt,
            visual_feedback_base64,
            verify_report,
            is_video_feedback=False,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs,
            complete_multi_turn_prompt,
            visual_feedback_base64,
            verify_report,
            is_video_feedback=False,
        )
    decision_prompt, lock_meta = g17.apply_cached_goal_identity_lock(
        decision_prompt,
        output_dir=config["output_dir"],
        trial=trial,
        env=env,
    )
    print(
        "[libero-g17-fg] goal-identity-verify mismatch → regen "
        f"locked={verdict.get('locked_identity')!r} "
        f"args={verdict.get('sgp_args')!r} "
        f"lock_applied={int(bool(lock_meta.get('applied')))}",
        flush=True,
    )
    content = _query_model(args, decision_prompt)
    decision, new_code = _parse_multi_turn_decision(content["content"])
    return decision, new_code, decision_prompt


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
    _restart_trial_alarm_after_reset()
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

    if os.environ.get("CAPX_LIBERO_G17_FG", "0") == "1":
        from capx.verification.libero_g17_takeover import (
            initialize_trial_trace as initialize_libero_g17_fg,
        )

        initialize_libero_g17_fg(
            env,
            output_dir=config["output_dir"],
            trial=trial,
            task=os.environ.get("CAPX_CURRENT_TASK_NAME", ""),
        )

    # --- 3. Initial code generation ---
    prefix_package = None
    if os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY", "").strip():
        from capx.verification import libero_g17_code_prefix_replay as code_prefix

        prefix_package = code_prefix.load_package()
        code_prefix.assert_init_matches(env, trial, prefix_package)

    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
    elif prefix_package is not None:
        from capx.verification import libero_g17_code_prefix_replay as code_prefix

        forced = code_prefix.prefix_initial_raw_code(prefix_package, trial)
        if forced is None:
            from capx.verification.libero_g17_mechanism_contract import (
                forbid_live_student_llm,
            )

            forbid_live_student_llm(
                trial=trial,
                turn=0,
                reason="code-prefix miss; refusing live initial code",
            )
            raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)
        else:
            raw_code = forced
            reasoning = None
            ensemble_data = None
            code_prefix.write_prefix_hit_audit(
                config["output_dir"],
                trial=trial,
                turn=0,
                code=prefix_package.trial_entry(trial).code_blocks[0],
                package=prefix_package,
            )
            print(
                f"[libero-g17-prefix] trial={trial} turn=0 "
                f"source=code_prefix_replay package={prefix_package.sha256[:12]}"
            )
            with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
                f.write(str(obs["full_prompt"]))
    else:
        from capx.verification.libero_g17_mechanism_contract import (
            forbid_live_student_llm,
        )

        forbid_live_student_llm(
            trial=trial,
            turn=0,
            reason="no code-prefix package; refusing live initial code",
        )
        raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)

    initial_action_prompt = copy.deepcopy(obs["full_prompt"])
    initial_rebound_enabled = (
        os.environ.get("CAPX_IDEAL_TK_V4_INITIAL_REBOUND", "0") == "1"
    )
    # Ideal-TK outer-loop flag; stays False for G17 FG arms (Ideal-TK off).
    skip_multiturn_after_ideal_fail = False
    if initial_rebound_enabled:
        if os.environ.get("CAPX_IDEAL_TK_V4", "0") != "1":
            raise RuntimeError("Initial rebound requires Ideal TK v4")
        if config["use_oracle_code"]:
            raise RuntimeError("Initial rebound cannot validate oracle code")
        if config["use_parallel_ensemble"]:
            raise RuntimeError("Initial rebound does not support ensemble queries")
        native_initial_report = obs.get("_capx_initial_native_report")
        if not isinstance(native_initial_report, str) or not native_initial_report.strip():
            raise RuntimeError("Initial rebound requires the native M3 initial report")

        from capx.verification import closed_gt
        from capx.verification.candidate_rebound import (
            build_initial_clean_rebound_prompt,
            record_initial_rebound_request,
            record_initial_validation,
            render_endpoint_pre_facts,
            validate_initial_candidate,
        )

        scene = closed_gt.cube_scene_spec()
        initial_snapshot = closed_gt.extract_cube_snapshot(env)
        verified_initial_report = render_endpoint_pre_facts(initial_snapshot.values)
        object_ids_by_name = {
            item.public_name.casefold(): item.entity_id for item in scene.objects
        }
        max_rewrites = int(
            os.environ.get("CAPX_IDEAL_TK_V4_INITIAL_REBOUND_ATTEMPTS", "2")
        )
        if max_rewrites < 1:
            raise ValueError("Initial rebound attempts must be at least one")
        attempt = 0
        while True:
            validation = validate_initial_candidate(
                initial_snapshot.values, object_ids_by_name, raw_code
            )
            record_initial_validation(
                output_dir=config["output_dir"],
                trial=trial,
                attempt=attempt,
                response=raw_code,
                validation=validation,
            )
            if validation.accepted:
                break
            if attempt >= max_rewrites:
                raise RuntimeError(
                    "Initial candidate remains inconsistent after bounded rebounds"
                )
            rebound_prompt = build_initial_clean_rebound_prompt(
                obs["full_prompt"],
                native_report=native_initial_report,
                verified_initial_report=verified_initial_report,
                validation=validation,
            )
            record_initial_rebound_request(
                output_dir=config["output_dir"],
                trial=trial,
                attempt=attempt,
                rejected_response=raw_code,
                rebound_prompt=rebound_prompt,
            )
            out = _query_model(args, rebound_prompt)
            raw_code = out["content"]
            reasoning = out["reasoning"]
            initial_action_prompt = rebound_prompt
            attempt += 1

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
    initial_blocks = _extract_code(raw_code)
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": initial_action_prompt,
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
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    if skip_multiturn_after_ideal_fail:
        multi_turn_prompt = None

    identity_verify_audit: list[dict[str, Any]] = []
    while code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT:
        code = code_blocks[code_block_idx]

        # Pre-exec goal-identity verify (live blocks only; skip frozen prefix).
        if (
            os.environ.get("CAPX_LIBERO_G17_FG", "0") == "1"
            and trial is not None
            and multi_turn_prompt
            and not _pending_block_is_code_prefix(trial, code_block_idx)
        ):
            from capx.verification import libero_g17_takeover as g17

            if g17.goal_identity_verify_enabled():
                max_verify = int(
                    os.environ.get("CAPX_LIBERO_G17_GOAL_IDENTITY_VERIFY_MAX", "2")
                )
                attempt = 0
                while attempt <= max_verify:
                    verdict = g17.verify_pending_acquire_code(
                        code,
                        output_dir=config["output_dir"],
                        trial=trial,
                    )
                    if not verdict.get("applicable") or verdict.get("ok"):
                        if attempt or verdict.get("applicable"):
                            identity_verify_audit.append(
                                {
                                    "block_idx": code_block_idx,
                                    "attempt": attempt,
                                    "verdict": verdict,
                                    "action": "pass",
                                }
                            )
                        break
                    attempt += 1
                    if attempt > max_verify:
                        locked = str(verdict.get("locked_identity") or "")
                        code2, n_rw = g17.rewrite_sgp_args_to_locked_identity(
                            code, locked
                        )
                        identity_verify_audit.append(
                            {
                                "block_idx": code_block_idx,
                                "attempt": attempt,
                                "verdict": verdict,
                                "action": "fallback_rewrite",
                                "rewrite_hits": n_rw,
                            }
                        )
                        print(
                            "[libero-g17-fg] goal-identity-verify fallback rewrite "
                            f"hits={n_rw} locked={locked!r}",
                            flush=True,
                        )
                        code = code2
                        code_blocks[code_block_idx] = code
                        break
                    decision_v, new_code_v, decision_prompt_v = _run_identity_verify_regen(
                        env,
                        args=args,
                        config=config,
                        multi_turn_prompt=multi_turn_prompt,
                        code_blocks=code_blocks,
                        code_block_idx=code_block_idx,
                        info_step=info_step,
                        obs=obs,
                        verdict=verdict,
                        trial=trial,
                        visual_feedback_base64_history=visual_feedback_base64_history,
                    )
                    identity_verify_audit.append(
                        {
                            "block_idx": code_block_idx,
                            "attempt": attempt,
                            "verdict": verdict,
                            "action": "regen",
                            "decision": decision_v,
                        }
                    )
                    if decision_v != "regenerate" or not new_code_v:
                        print(
                            "[libero-g17-fg] goal-identity-verify regen did not "
                            f"return code (decision={decision_v!r}); fallback rewrite",
                            flush=True,
                        )
                        locked = str(verdict.get("locked_identity") or "")
                        code2, n_rw = g17.rewrite_sgp_args_to_locked_identity(
                            code, locked
                        )
                        code = code2
                        code_blocks[code_block_idx] = code
                        identity_verify_audit[-1]["action"] = "regen_then_fallback"
                        identity_verify_audit[-1]["rewrite_hits"] = n_rw
                        break
                    new_blocks = _extract_code(new_code_v)
                    all_responses.append(
                        {
                            "multi_turn_prompt": (
                                decision_prompt_v
                                if config.get("save_multiturn_prompts", False)
                                else None
                            ),
                            "block_idx": [code_block_idx],
                            "code_blocks": new_blocks,
                            "decision": "regenerate",
                            "reasoning": "",
                            "identity_verify": True,
                        }
                    )
                    del code_blocks[code_block_idx:]
                    del code_block_metadata[code_block_idx:]
                    code_blocks.extend(new_blocks)
                    code_block_metadata.extend(
                        [
                            {
                                "generation": num_regenerations + 1,
                                "regenerated": True,
                                "regenerated_at_idx": code_block_idx,
                                "identity_verify": True,
                            }
                        ]
                        * len(new_blocks)
                    )
                    num_regenerations += 1
                    if partial_artifacts is not None:
                        partial_artifacts["num_regenerations"] = num_regenerations
                    code = code_blocks[code_block_idx]

        code_block_idx += 1

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        if os.environ.get("CAPX_LIBERO_G17_FG", "0") == "1" and trial is not None:
            from capx.verification.libero_g17_takeover import (
                begin_turn_interval_sampling,
            )

            begin_turn_interval_sampling(config["output_dir"], trial)

        obs_next, reward, terminated, truncated, info_step = env.step(code)

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

        if bool(terminated) or bool(truncated) or bool(
            info_step.get("task_completed", False)
        ):
            stop_reason = (
                "task_completed"
                if info_step.get("task_completed", False)
                else ("terminated" if terminated else "truncated")
            )
            print(
                "Environment ended the episode after code block "
                f"{code_block_idx}: {stop_reason}"
            )
            if os.environ.get("CAPX_LIBERO_G17_FG", "0") == "1" and trial is not None:
                from capx.verification.libero_g17_takeover import (
                    capture_terminal_witness,
                )

                capture_terminal_witness(
                    env,
                    output_dir=config["output_dir"],
                    trial=trial,
                    turn=code_block_idx,
                    task=os.environ.get("CAPX_CURRENT_TASK_NAME", ""),
                    private_task_completed=bool(
                        info_step.get("task_completed", False)
                    ),
                    stop_reason=stop_reason,
                )
            break

        # Multi-turn decision
        if multi_turn_prompt:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt, post_miss_clear = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                turn_frames=turn_frames,
                wrist_turn_frames=wrist_turn_frames,
                wrist_base64_history=wrist_base64_history,
                trial=trial,
            )

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                print("Model chose to regenerate code")
                new_blocks = _extract_code(new_code)
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                    "post_miss_clear": bool(post_miss_clear),
                })
                if post_miss_clear:
                    n_prefix = 1
                    if os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY", "").strip():
                        from capx.verification import (
                            libero_g17_code_prefix_replay as code_prefix,
                        )

                        try:
                            n_prefix = int(
                                code_prefix.load_package().trial_entry(trial).n_prefix
                            )
                        except Exception:
                            n_prefix = 1
                    n_prefix = max(0, min(n_prefix, len(code_blocks), code_block_idx))
                    keep_blocks = code_blocks[:n_prefix]
                    keep_meta = code_block_metadata[:n_prefix]
                    code_blocks[:] = keep_blocks
                    code_block_metadata[:] = keep_meta
                    code_blocks.extend(new_blocks)
                    code_block_metadata.extend(
                        [{"generation": num_regenerations + 1, "regenerated": True,
                          "regenerated_at_idx": n_prefix, "post_miss_clear": True}]
                        * len(new_blocks)
                    )
                    code_block_idx = n_prefix
                    print(
                        f"[libero-g17-fg] post-miss-clear reset code history "
                        f"n_prefix={n_prefix} new_blocks={len(new_blocks)}",
                        flush=True,
                    )
                else:
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

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
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
    if identity_verify_audit:
        verify_path = os.path.join(
            config["output_dir"], "libero_g17_identity_verify_audit.json"
        )
        with open(verify_path, "w", encoding="utf-8") as f:
            json.dump(identity_verify_audit, f, indent=2)
        print(
            f"[libero-g17-fg] wrote identity verify audit n={len(identity_verify_audit)}",
            flush=True,
        )

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
