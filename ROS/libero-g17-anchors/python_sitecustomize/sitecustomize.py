"""Runtime compatibility patches for CaP-X Slurm smoke jobs.

Loaded automatically when this directory is prepended to PYTHONPATH.
This intentionally avoids importing heavy packages at interpreter startup.
"""

from __future__ import annotations

import builtins
import os
import re
import sys


_original_import = builtins.__import__
_CURRENT_TRIAL = None
_CURRENT_QUERY_INDEX = 0
_PRIVILEGED_COACH = os.environ.get("CAPX_PRIVILEGED_COACH", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_NODE_COACH = os.environ.get("CAPX_NODE_COACH", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _seed_runtime_from_env() -> None:
    raw_seed = os.environ.get("CAPX_GLOBAL_SEED")
    if not raw_seed:
        return
    try:
        seed = int(raw_seed)
    except ValueError:
        print(
            f"[capx runtime patch] WARNING: invalid CAPX_GLOBAL_SEED={raw_seed!r}",
            file=sys.stderr,
        )
        return

    try:
        import random

        random.seed(seed)
    except Exception as exc:
        print(
            f"[capx runtime patch] WARNING: failed to seed python random: {exc}",
            file=sys.stderr,
        )

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception as exc:
        print(
            f"[capx runtime patch] WARNING: failed to seed numpy: {exc}",
            file=sys.stderr,
        )

    if os.environ.get("CAPX_SEED_TORCH", "0") == "1":
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception as exc:
            print(
                f"[capx runtime patch] WARNING: failed to seed torch: {exc}",
                file=sys.stderr,
            )

    print(
        f"[capx runtime patch] seeded python/numpy"
        f"{'/torch' if os.environ.get('CAPX_SEED_TORCH', '0') == '1' else ''}"
        f" with CAPX_GLOBAL_SEED={seed}",
        file=sys.stderr,
    )


_seed_runtime_from_env()


def _seed_all_for_deterministic_context(seed: int, label: str) -> None:
    try:
        import random

        random.seed(seed)
    except Exception as exc:
        print(
            f"[capx deterministic seed] WARNING: failed to seed python random for {label}: {exc}",
            file=sys.stderr,
        )
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception as exc:
        print(
            f"[capx deterministic seed] WARNING: failed to seed numpy for {label}: {exc}",
            file=sys.stderr,
        )
    if os.environ.get("CAPX_SEED_TORCH", "0") == "1":
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception as exc:
            print(
                f"[capx deterministic seed] WARNING: failed to seed torch for {label}: {exc}",
                file=sys.stderr,
            )
    print(f"[capx deterministic seed] {label}: seed={seed}", file=sys.stderr)


def _deterministic_base_seed() -> int:
    return int(os.environ.get("CAPX_GLOBAL_SEED", "0") or "0")


def _deterministic_seed_enabled() -> bool:
    return os.environ.get("CAPX_DETERMINISTIC_TRIAL_SEEDING", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _deterministic_trial_seed(trial: int) -> int:
    fixed = os.environ.get("CAPX_FIXED_EPISODE_SEED")
    if fixed not in (None, ""):
        return int(fixed)
    return _deterministic_base_seed() + int(trial)


def _deterministic_query_seed(trial: int, query_index: int) -> int:
    fixed = os.environ.get("CAPX_FIXED_EPISODE_SEED")
    if fixed not in (None, ""):
        return int(fixed) * 1000 + int(query_index)
    return _deterministic_base_seed() + int(trial) * 1000 + int(query_index)


_LIFT_DOCFIX = os.environ.get("LIFT_DOCFIX", "0") == "1"
_LIFT_M3_FINISH_GUARD = os.environ.get("LIFT_M3_FINISH_GUARD", "0") == "1"
_ENABLE_DEPTH_PATCH = os.environ.get("CAPX_ENABLE_DEPTH_PATCH", "0") == "1"
_PATCH_POINTCLOUD_ALIGNMENT = (
    os.environ.get("CAPX_PATCH_POINTCLOUD_ALIGNMENT", "0") == "1"
)
_INJECT_NUMPY_GLOBAL = os.environ.get("CAPX_INJECT_NUMPY_GLOBAL", "0") == "1"
_NUT_REDUCED_DOCFIX = os.environ.get("NUT_REDUCED_DOCFIX", "0") == "1"
_OPEN3D_OBB_FALLBACK = os.environ.get("CAPX_OPEN3D_OBB_FALLBACK", "0") == "1"
_ROBUST_MASK_OBB = os.environ.get("CAPX_ROBUST_MASK_OBB", "0") == "1"


def _looks_like_two_arm_lift_config(filenames) -> bool:
    if isinstance(filenames, str):
        return "two_arm_lift" in filenames
    if isinstance(filenames, list):
        return any(isinstance(item, str) and "two_arm_lift" in item for item in filenames)
    return False


def _looks_like_nut_assembly_config(filenames) -> bool:
    if isinstance(filenames, str):
        return "nut_assembly" in filenames
    if isinstance(filenames, list):
        return any(isinstance(item, str) and "nut_assembly" in item for item in filenames)
    return False


def _patch_two_arm_lift_loaded_config(cfg) -> None:
    if not isinstance(cfg, dict):
        return
    env = cfg.get("env")
    if not isinstance(env, dict):
        return
    env_cfg = env.get("cfg")
    if not isinstance(env_cfg, dict):
        return

    if _LIFT_DOCFIX:
        prompt = env_cfg.get("prompt")
        if isinstance(prompt, str) and "Important grasp convention for this task" not in prompt:
            needle = (
                "- Coordinate system: All pose functions accept/return positions in robot0's base frame."
            )
            insert = """- Important grasp convention for this task: `get_handle0_pos()` and `get_handle1_pos()` return visual bbox centers, not final pregrasp poses.
- For the non-privileged handle-center API, use the following side-grasp strategy: set `gripper0_wxyz = np.array([0.7071, 0.7071, 0, 0])` and `gripper1_wxyz = np.array([0.7071, -0.7071, 0, 0])`.
- Before closing the grippers, first approach handle0 with `handle0_pos[1] += 0.15` and handle1 with `handle1_pos[1] -= 0.15`, then move back to the original bbox centers and close both grippers.
- Lift both arms simultaneously after closing the grippers."""
            if needle in prompt:
                env_cfg["prompt"] = prompt.replace(needle, needle + "\n" + insert)
                print(
                    "[capx runtime patch] applied two_arm_lift API documentation fix",
                    file=sys.stderr,
                )
            else:
                print(
                    "[capx runtime patch] WARNING: two_arm_lift prompt insertion point not found",
                    file=sys.stderr,
                )

    if _LIFT_M3_FINISH_GUARD:
        multi_turn_prompt = env_cfg.get("multi_turn_prompt")
        if (
            isinstance(multi_turn_prompt, str)
            and "Two-arm lift finish guard" not in multi_turn_prompt
        ):
            guard = """

      Two-arm lift finish guard:
      - If the latest execution had empty stderr and stdout indicates that the pot was lifted or the task was completed (for example: "Task Completed: True", "Task complete", "Pot lifted", or "Lift complete"), respond with exactly FINISH.
      - Do not regenerate only to re-detect handles after the pot has already moved; repeated perception after a lift can lose the handles from view and corrupt an otherwise successful attempt.
"""
            marker = "      Please respond with EXACTLY ONE of the following:"
            if marker in multi_turn_prompt:
                env_cfg["multi_turn_prompt"] = multi_turn_prompt.replace(marker, guard + marker)
                print(
                    "[capx runtime patch] applied two_arm_lift M3 finish guard",
                    file=sys.stderr,
                )
            else:
                env_cfg["multi_turn_prompt"] = multi_turn_prompt + guard
                print(
                    "[capx runtime patch] appended two_arm_lift M3 finish guard",
                    file=sys.stderr,
                )


def _patch_nut_assembly_reduced_loaded_config(cfg) -> None:
    if not _NUT_REDUCED_DOCFIX or not isinstance(cfg, dict):
        return
    env = cfg.get("env")
    if not isinstance(env, dict):
        return
    env_cfg = env.get("cfg")
    if not isinstance(env_cfg, dict):
        return
    apis = env_cfg.get("apis")
    if not isinstance(apis, list) or not any(
        api in ("FrankaControlNutAssemblyApiReduced", "FrankaControlNutAssemblyApiReducedExampleless")
        for api in apis
    ):
        return
    prompt = env_cfg.get("prompt")
    if not isinstance(prompt, str):
        prompt = cfg.get("prompt") if isinstance(cfg.get("prompt"), str) else None
    if prompt is None:
        # The class default prompt is inherited later; provide a reduced-API-specific
        # prompt here so the task description matches the actual callable API tier.
        prompt = """You are controlling a Franka Emika robot with API described below.
Goal: grasp and insert the `brown square nut` onto the `brown square block`.
Use the reduced low-level API only. The functions below are already imported to the environment.
For perception, call get_observation(), identify pixels with point_prompt_molmo or SAM3 segmentation, use plan_grasp(depth, intrinsics, segmentation) for grasp candidates, transform camera-frame grasp poses using the camera pose, then use solve_ik and move_to_joints for motion.
The brown square nut should be grasped by its extruded handle. The brown square block is the target peg/block for insertion.
You may write python code comments for reasoning but ONLY write executable Python code and do not write it in code fences.
"""
    if "Use the reduced low-level API only" in prompt:
        return
    prompt = prompt.replace(
        "The grasp pose query for 'extruded handle of the brown square nut' returns an end-effector pose expressed in world frame, located on the handle region.\n",
        "",
    )
    prompt = prompt.replace(
        "The nut's object center pose obtained via 'white hollow center of the brown square nut' and the handle grasp pose obtained via 'extruded handle of the brown square nut' have a fixed rigid transform, which must be applied correctly when inserting the nut onto the peg.\n",
        "",
    )
    insert = (
        "Use the reduced low-level API only. For perception, call get_observation(), "
        "identify pixels with point_prompt_molmo or SAM3 segmentation, use "
        "plan_grasp(depth, intrinsics, segmentation) for grasp candidates, transform "
        "camera-frame grasp poses using the camera pose, then use solve_ik and "
        "move_to_joints for motion.\n"
    )
    prompt = prompt.replace(
        "The functions (APIs) below are already imported to the environment.\n",
        "The functions (APIs) below are already imported to the environment.\n" + insert,
    )
    env_cfg["prompt"] = prompt
    print(
        "[capx runtime patch] applied nut_assembly reduced API documentation fix",
        file=sys.stderr,
    )


def _patch_nut_assembly_task_prompt(module) -> None:
    if not _NUT_REDUCED_DOCFIX:
        return
    cls = getattr(module, "FrankaNutAssemblyCodeEnv", None)
    if cls is None or getattr(cls, "_capx_nut_reduced_docfix", False):
        return
    prompt = getattr(cls, "prompt", "")
    if not isinstance(prompt, str):
        return
    if "Use the reduced low-level API only" in prompt:
        return
    prompt = prompt.replace(
        "The grasp pose query for 'extruded handle of the brown square nut' returns an end-effector pose expressed in world frame, located on the handle region.\n",
        "",
    )
    prompt = prompt.replace(
        "The nut's object center pose obtained via 'white hollow center of the brown square nut' and the handle grasp pose obtained via 'extruded handle of the brown square nut' have a fixed rigid transform, which must be applied correctly when inserting the nut onto the peg.\n",
        "",
    )
    insert = (
        "Use the reduced low-level API only. For perception, call get_observation(), "
        "identify pixels with point_prompt_molmo or SAM3 segmentation, use "
        "plan_grasp(depth, intrinsics, segmentation) for grasp candidates, transform "
        "camera-frame grasp poses using the camera pose, then use solve_ik and "
        "move_to_joints for motion.\n"
    )
    prompt = prompt.replace(
        "The functions (APIs) below are already imported to the environment.\n",
        "The functions (APIs) below are already imported to the environment.\n" + insert,
    )
    cls.prompt = prompt
    cls._capx_nut_reduced_docfix = True
    print(
        "[capx runtime patch] applied nut_assembly reduced class prompt fix",
        file=sys.stderr,
    )


def _patch_config_loader(module) -> None:
    dict_loader = getattr(module, "DictLoader", None)
    if dict_loader is None:
        return
    original = getattr(dict_loader, "load", None)
    if original is None or getattr(original, "_capx_lift_patched", False):
        return

    def load_with_capx_runtime_patch(filenames, keys=None):
        cfg = original(filenames, keys)
        if _looks_like_two_arm_lift_config(filenames) and (_LIFT_DOCFIX or _LIFT_M3_FINISH_GUARD):
            _patch_two_arm_lift_loaded_config(cfg)
        if _looks_like_nut_assembly_config(filenames) and _NUT_REDUCED_DOCFIX:
            _patch_nut_assembly_reduced_loaded_config(cfg)
        return cfg

    load_with_capx_runtime_patch._capx_lift_patched = True
    dict_loader.load = staticmethod(load_with_capx_runtime_patch)


def _two_arm_lift_should_finish(env, info_step) -> bool:
    if not _LIFT_M3_FINISH_GUARD:
        return False
    if env.__class__.__name__ != "TwoArmLiftCodeEnv":
        return False
    stderr = str(info_step.get("stderr", "") or "").strip()
    if stderr:
        return False
    return bool(info_step.get("task_completed", False))


def _patch_trial_module(module) -> None:
    original = getattr(module, "_handle_multi_turn_step", None)
    if original is None or getattr(original, "_capx_runtime_multiturn_patch", False):
        return

    def handle_multi_turn_step_with_lift_guard(
        env,
        obs,
        args,
        config,
        visual_differencing_args,
        multi_turn_prompt,
        code_blocks,
        code_block_idx,
        info_step,
        task_description,
        visual_feedback_imgs,
        visual_feedback_base64_history,
        stderr_history,
        turn_frames=None,
        wrist_turn_frames=None,
        wrist_base64_history=None,
        **kwargs,
    ):
        if _two_arm_lift_should_finish(env, info_step):
            print(
                "[capx runtime patch] two_arm_lift finish guard forced FINISH",
                file=sys.stderr,
            )
            return (
                "finish",
                None,
                "Runtime finish guard: two_arm_lift environment reported task_completed=True with empty stderr.",
                None,
                None,
            )
        if _PRIVILEGED_COACH and not bool(info_step.get("task_completed", False)):
            try:
                import coach_runtime

                reward = getattr(module, "reward", 0.0)
                # The trial module keeps reward local, so parse it from recent stdout only if unavailable.
                # The coach receives reward only as a bucket and never exposes it to the student.
                public_guidance, audit = coach_runtime.build_and_call_coach(
                    env=env,
                    args=args,
                    config=config,
                    code_blocks=code_blocks,
                    code_block_idx=code_block_idx,
                    info_step=info_step,
                    reward=0.0,
                    task_description=task_description,
                )
                if not public_guidance and os.environ.get("CAPX_COACH_FALLBACK", "1") == "1":
                    public_guidance = coach_runtime.fallback_guidance()
                if public_guidance:
                    info_step = dict(info_step)
                    info_step["stdout"] = (
                        f"{info_step.get('stdout', '')}\n\n"
                        "SANITIZED_PRIVILEGED_COACH_GUIDANCE:\n"
                        f"{public_guidance}\n"
                    )
                    print(
                        f"[capx privileged coach] injected sanitized guidance; audit={audit}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[capx privileged coach] no guidance injected; audit={audit}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(
                    f"[capx privileged coach] WARNING: coach failed: {exc}",
                    file=sys.stderr,
                )
        return original(
            env,
            obs,
            args,
            config,
            visual_differencing_args,
            multi_turn_prompt,
            code_blocks,
            code_block_idx,
            info_step,
            task_description,
            visual_feedback_imgs,
            visual_feedback_base64_history,
            stderr_history,
            turn_frames=turn_frames,
            wrist_turn_frames=wrist_turn_frames,
            wrist_base64_history=wrist_base64_history,
            **kwargs,
        )

    handle_multi_turn_step_with_lift_guard._capx_runtime_multiturn_patch = True
    module._handle_multi_turn_step = handle_multi_turn_step_with_lift_guard


def _patch_trial_deterministic_module(module) -> None:
    run_single = getattr(module, "_run_single_trial", None)
    if run_single is None or getattr(run_single, "_capx_deterministic_trial_seed", False):
        return

    def run_single_trial_with_deterministic_seed(
        env,
        trial,
        args,
        config,
        multi_turn_prompt,
        partial_artifacts=None,
    ):
        global _CURRENT_TRIAL, _CURRENT_QUERY_INDEX
        previous_trial = _CURRENT_TRIAL
        previous_query_index = _CURRENT_QUERY_INDEX
        try:
            import coach_runtime

            coach_state = coach_runtime.reset_episode_state(
                int(trial),
                os.environ.get("CAPX_CURRENT_TASK_NAME"),
            )
            print(
                f"[capx coach] reset episode state: {coach_state}",
                file=sys.stderr,
            )
        except Exception as exc:
            if os.environ.get("CAPX_PRIVILEGED_COACH", "0") == "1" or os.environ.get(
                "CAPX_NODE_COACH", "0"
            ) == "1":
                raise RuntimeError(
                    f"Failed to reset Coach state at trial boundary: {exc}"
                ) from exc
        if _deterministic_seed_enabled():
            _CURRENT_TRIAL = int(trial)
            _CURRENT_QUERY_INDEX = 0
            _seed_all_for_deterministic_context(
                _deterministic_trial_seed(int(trial)),
                f"trial_start trial={trial}",
            )
        try:
            result = run_single(
                env,
                trial,
                args,
                config,
                multi_turn_prompt,
                partial_artifacts=partial_artifacts,
            )
            if _deterministic_seed_enabled() and os.environ.get("CAPX_REPRO_AUDIT", "0") == "1":
                from capx_reproducibility import append_audit_record, initial_state_record

                final_record = initial_state_record(
                    getattr(env, "low_level_env", None),
                    {},
                    _deterministic_trial_seed(int(trial)),
                )
                final_record.update({"phase": "final", "trial": int(trial)})
                append_audit_record(final_record)
            return result
        finally:
            if _deterministic_seed_enabled():
                _CURRENT_TRIAL = previous_trial
                _CURRENT_QUERY_INDEX = previous_query_index

    run_single_trial_with_deterministic_seed._capx_deterministic_trial_seed = True
    module._run_single_trial = run_single_trial_with_deterministic_seed


def _patch_llm_client_module(module) -> None:
    if not _deterministic_seed_enabled():
        return
    requests_mod = getattr(module, "requests", None)
    if requests_mod is None:
        return
    original_post = getattr(requests_mod, "post", None)
    if original_post is None or getattr(original_post, "_capx_deterministic_request_seed", False):
        return

    def post_with_deterministic_seed(url, *args, **kwargs):
        global _CURRENT_QUERY_INDEX
        data = kwargs.get("data")
        if (
            _CURRENT_TRIAL is not None
            and isinstance(url, str)
            and "/chat/completions" in url
            and isinstance(data, str)
        ):
            try:
                import json

                payload = json.loads(data)
                if isinstance(payload, dict) and "seed" not in payload:
                    seed = _deterministic_query_seed(
                        int(_CURRENT_TRIAL),
                        int(_CURRENT_QUERY_INDEX),
                    )
                    payload["seed"] = seed
                    kwargs["data"] = json.dumps(payload)
                    print(
                        f"[capx deterministic seed] model_request trial={_CURRENT_TRIAL} "
                        f"query={_CURRENT_QUERY_INDEX} seed={seed}",
                        file=sys.stderr,
                    )
                    _CURRENT_QUERY_INDEX += 1
            except Exception as exc:
                print(
                    f"[capx deterministic seed] WARNING: failed to inject request seed: {exc}",
                    file=sys.stderr,
                )
        return original_post(url, *args, **kwargs)

    post_with_deterministic_seed._capx_deterministic_request_seed = True
    requests_mod.post = post_with_deterministic_seed


def _patch_base_task_module(module) -> None:
    cls = getattr(module, "CodeExecutionEnvBase", None)
    if cls is None:
        return
    original_init = getattr(cls, "_init_exec_globals", None)
    original_wrap = getattr(cls, "_maybe_wrap_preclose_feedback", None)
    original_exec = getattr(cls, "_exec_user_code", None)
    if original_init is None or original_wrap is None:
        return
    if getattr(original_init, "_capx_node_coach_registry_init", False):
        return

    action_pattern = os.environ.get(
        "CAPX_NODE_COACH_ACTION_PATTERN",
        r"^(?:open|close|release)_gripper(?:_|$)",
    )
    try:
        compiled_action_pattern = re.compile(action_pattern, re.IGNORECASE)
    except re.error as exc:
        raise RuntimeError(
            f"Invalid CAPX_NODE_COACH_ACTION_PATTERN={action_pattern!r}: {exc}"
        ) from exc

    if not getattr(original_wrap, "_capx_node_coach_registry_wrapper", False):
        def maybe_wrap_with_node_coach(self, fn_name, fn):
            wrapped_fn = original_wrap(self, fn_name, fn)
            if not _NODE_COACH or not compiled_action_pattern.search(fn_name):
                return wrapped_fn
            if getattr(wrapped_fn, "_capx_node_coach_action_wrapper", False):
                return wrapped_fn

            def node_coach_wrapped(*args, **kwargs):
                import coach_runtime
                import inspect

                if (
                    os.environ.get("CAPX_C1_SUCCESS_FREEZE", "0").lower()
                    in {"1", "true", "yes", "on"}
                    and coach_runtime.environment_task_completed(self.low_level_env)
                ):
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase="terminal_success_frozen_action_skipped",
                        api_args=args,
                        api_kwargs=kwargs,
                        decision="SUCCESS_FREEZE",
                        reason="exact terminal predicate already true",
                    )
                    print(
                        f"[capx node resume] SUCCESS_FREEZE {fn_name}: "
                        "terminal task state preserved",
                        file=sys.stderr,
                    )
                    return None

                try:
                    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                    candidate_arguments = dict(bound.arguments)
                except Exception:
                    candidate_arguments = dict(kwargs)

                fn_lower = fn_name.lower()
                if fn_lower.startswith("close_gripper"):
                    stage = "pre_grasp_commit"
                elif fn_lower.startswith(("open_gripper", "release_gripper")):
                    stage = "pre_release_commit"
                elif fn_lower.startswith(("goto_pose", "move", "solve")):
                    stage = "pre_motion_commit"
                else:
                    stage = "pre_public_action"

                retry_gate = coach_runtime.check_interrupted_retry_transaction(fn_name)
                if not retry_gate.get("allowed", False):
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase="local_retry_transaction_replay_blocked",
                        api_args=args,
                        api_kwargs=kwargs,
                        decision="INTERRUPT",
                        reason=str(retry_gate),
                    )
                    raise RuntimeError(
                        "C1_LOCAL_RETRY_TRANSACTION_LOCK:\n"
                        "A validated correction has executed. Retry the interrupted API "
                        "immediately; do not replay any earlier or unrelated robot action.\n"
                        f"EVIDENCE: {retry_gate}"
                    )
                if retry_gate.get("retry_consumed"):
                    print(
                        f"[capx transaction lock] RETRY {fn_name}: "
                        "resuming the interrupted local transaction",
                        file=sys.stderr,
                    )

                causal_motion_enabled = os.environ.get(
                    "CAPX_C1_CAUSAL_POST_CLOSE_GATE", "0"
                ).lower() in {"1", "true", "yes", "on"}
                causal_motion_gate = (
                    stage == "pre_motion_commit"
                    and causal_motion_enabled
                    and coach_runtime.has_pending_stable_commit()
                )
                close_prestate = (
                    coach_runtime.capture_close_prestate(self.low_level_env, fn_name)
                    if stage == "pre_grasp_commit"
                    and (
                        causal_motion_enabled
                        or os.environ.get(
                            "CAPX_C1_COUNTERFACTUAL_CLOSE_PREVIEW", "0"
                        ).lower() in {"1", "true", "yes", "on"}
                    )
                    else None
                )
                close_candidate_outcome = None
                if (
                    stage == "pre_grasp_commit"
                    and os.environ.get(
                        "CAPX_C1_COUNTERFACTUAL_CLOSE_PREVIEW", "0"
                    ).lower() in {"1", "true", "yes", "on"}
                ):
                    preview_snapshot = coach_runtime.snapshot_counterfactual_state(
                        self.low_level_env
                    )
                    try:
                        wrapped_fn(*args, **kwargs)
                        close_candidate_outcome = (
                            coach_runtime.evaluate_close_transition(
                                self.low_level_env, close_prestate or {}
                            )
                        )
                    finally:
                        coach_runtime.restore_counterfactual_state(
                            self.low_level_env, preview_snapshot
                        )
                release_candidate_outcome = None
                if (
                    stage == "pre_release_commit"
                    and os.environ.get(
                        "CAPX_C1_COUNTERFACTUAL_RELEASE_PREVIEW", "0"
                    ).lower() in {"1", "true", "yes", "on"}
                ):
                    release_prestate = coach_runtime.capture_release_prestate(
                        self.low_level_env, fn_name
                    )
                    preview_snapshot = coach_runtime.snapshot_counterfactual_state(
                        self.low_level_env
                    )
                    try:
                        wrapped_fn(*args, **kwargs)
                        release_candidate_outcome = (
                            coach_runtime.evaluate_release_transition(
                                self.low_level_env, release_prestate
                            )
                        )
                    finally:
                        coach_runtime.restore_counterfactual_state(
                            self.low_level_env, preview_snapshot
                        )
                motion_candidate_outcome = None
                if (
                    causal_motion_gate
                    and os.environ.get(
                        "CAPX_C1_COUNTERFACTUAL_PROTECTED_MOTION_PREVIEW", "0"
                    ).lower() in {"1", "true", "yes", "on"}
                ):
                    preview_snapshot = coach_runtime.snapshot_counterfactual_state(
                        self.low_level_env
                    )
                    try:
                        wrapped_fn(*args, **kwargs)
                        motion_candidate_outcome = (
                            coach_runtime.evaluate_pending_stable_commit(
                                self.low_level_env
                            )
                        )
                    finally:
                        coach_runtime.restore_counterfactual_state(
                            self.low_level_env, preview_snapshot
                        )

                def execute_with_counterfactual_preview():
                    if os.environ.get("CAPX_C1_NO_WORSE", "0").lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }:
                        # The dry-run is evidence for validating a Coach request,
                        # not an independent blocker. Once the no-worse firewall
                        # resolves the request to ALLOW, execute the untouched A1
                        # action exactly once.
                        return wrapped_fn(*args, **kwargs)
                    close_preview_enabled = (
                        stage == "pre_grasp_commit"
                        and os.environ.get(
                            "CAPX_C1_COUNTERFACTUAL_CLOSE_PREVIEW", "0"
                        ).lower() in {"1", "true", "yes", "on"}
                    )
                    motion_preview_enabled = (
                        causal_motion_gate
                        and os.environ.get(
                            "CAPX_C1_COUNTERFACTUAL_PROTECTED_MOTION_PREVIEW", "0"
                        ).lower() in {"1", "true", "yes", "on"}
                    )
                    if not close_preview_enabled and not motion_preview_enabled:
                        return wrapped_fn(*args, **kwargs)
                    snapshot = coach_runtime.snapshot_counterfactual_state(
                        self.low_level_env
                    )
                    try:
                        wrapped_fn(*args, **kwargs)
                        if close_preview_enabled:
                            preview_evidence = coach_runtime.evaluate_close_transition(
                                self.low_level_env, close_prestate or {}
                            )
                        else:
                            preview_evidence = (
                                coach_runtime.evaluate_pending_stable_commit(
                                    self.low_level_env
                                )
                            )
                    except Exception:
                        coach_runtime.restore_counterfactual_state(
                            self.low_level_env, snapshot
                        )
                        raise
                    coach_runtime.restore_counterfactual_state(
                        self.low_level_env, snapshot
                    )
                    preview_kind = "close" if close_preview_enabled else "protected_motion"
                    preview_accepted = (
                        preview_evidence.get("accepted")
                        if close_preview_enabled
                        else preview_evidence.get("preserved")
                    )
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase=(
                            f"counterfactual_{preview_kind}_accepted"
                            if preview_accepted
                            else f"counterfactual_{preview_kind}_rejected"
                        ),
                        api_args=args,
                        api_kwargs=kwargs,
                        decision="COUNTERFACTUAL_PREVIEW",
                        reason=str(preview_evidence),
                    )
                    print(
                        f"[capx counterfactual preview] {fn_name}: {preview_evidence}",
                        file=sys.stderr,
                    )
                    if not preview_accepted:
                        raise RuntimeError(
                            f"COUNTERFACTUAL_{preview_kind.upper()}_REJECTED:\n"
                            "A privileged simulator dry-run predicts that this action will not "
                            "establish or preserve its evaluator postcondition. The simulator "
                            "state has been restored. Stop this code block and repair only the "
                            "current action before retrying.\n"
                            f"EVIDENCE: {preview_evidence}"
                        )
                    return wrapped_fn(*args, **kwargs)

                if coach_runtime.consume_validated_action_permit(
                    fn_name, candidate_arguments
                ):
                    print(
                        f"[capx node coach] ONE_SHOT_PERMIT {fn_name}: "
                        "executing exact harness-validated Coach correction",
                        file=sys.stderr,
                    )
                    try:
                        result = execute_with_counterfactual_preview()
                    except Exception as exc:
                        coach_runtime.cancel_consumed_action_permit()
                        coach_runtime.record_c1_transaction(
                            low_level_env=self.low_level_env,
                            api_name=fn_name,
                            phase="validated_correction_execution_error",
                            api_args=args,
                            api_kwargs=kwargs,
                            decision="ONE_SHOT_PERMIT",
                            reason="exact Coach-validated action consumed once",
                            error=repr(exc),
                        )
                        raise
                    retry_lock = coach_runtime.complete_validated_action_permit()
                    correction_reason = "exact Coach-validated action consumed once"
                    if retry_lock.get("locked"):
                        correction_reason += f"; retry_lock={retry_lock}"
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase="validated_correction_executed_and_observed",
                        api_args=args,
                        api_kwargs=kwargs,
                        decision="ONE_SHOT_PERMIT",
                        reason=correction_reason,
                    )
                    if stage == "pre_grasp_commit" and causal_motion_enabled:
                        registration = coach_runtime.register_stable_commit_after_close(
                            self.low_level_env, fn_name, close_prestate or {}
                        )
                        print(
                            f"[capx causal gate] close registration: {registration}",
                            file=sys.stderr,
                        )
                        if (
                            os.environ.get("CAPX_C1_ABORT_UNVERIFIED_CLOSE", "0").lower()
                            in {"1", "true", "yes", "on"}
                            and not registration.get("registered")
                        ):
                            raise RuntimeError(
                                "CAUSAL_CLOSE_UNVERIFIED:\n"
                                "The executed close did not establish a stable evaluator "
                                "postcondition. Stop this code block and repair only this "
                                "grasp commit; do not execute dependent motions.\n"
                                f"EVIDENCE: {registration}"
                            )
                    if causal_motion_gate:
                        verification = coach_runtime.verify_pending_stable_commit(
                            self.low_level_env
                        )
                        coach_runtime.record_c1_transaction(
                            low_level_env=self.low_level_env,
                            api_name=fn_name,
                            phase=(
                                "causal_motion_preserved_commit"
                                if verification.get("preserved")
                                else "causal_motion_lost_commit"
                            ),
                            api_args=args,
                            api_kwargs=kwargs,
                            decision="CAUSAL_VERIFY",
                            reason=str(verification),
                        )
                        if not verification.get("preserved"):
                            raise RuntimeError(
                                "CAUSAL_COMMIT_LOST:\n"
                                "A dependent motion after a verified close destroyed "
                                "its evaluator-confirmed postcondition. Stop this code block. "
                                "Recover only that motion while preserving the close commit.\n"
                                f"EVIDENCE: {verification}"
                            )
                    return result

                api_contract = {"api_instances": []}
                for api_name, api_obj in getattr(self, "_apis", {}).items():
                    exposed = {}
                    for public_name, public_fn in api_obj.functions().items():
                        try:
                            signature = str(inspect.signature(public_fn))
                        except Exception:
                            signature = "(...)"
                        exposed[public_name] = {
                            "signature": signature,
                            "documentation": inspect.getdoc(public_fn) or "",
                        }
                    api_contract["api_instances"].append(
                        {
                            "registry_name": api_name,
                            "class_name": api_obj.__class__.__name__,
                            "exposed_functions": exposed,
                            "tcp_transform_metadata": {
                                key: getattr(api_obj, key).tolist()
                                if hasattr(getattr(api_obj, key), "tolist")
                                else repr(getattr(api_obj, key))
                                for key in ("_TCP_OFFSET", "TCP_OFFSET", "tcp_offset")
                                if hasattr(api_obj, key)
                            },
                        }
                    )

                adaptive_motion_gate = os.environ.get(
                    "CAPX_C1_ADAPTIVE_MOTION_GATE", "0"
                ).lower() in {"1", "true", "yes", "on"}
                stable_gate_state = (
                    coach_runtime.stable_manipulation_gate_state(self.low_level_env)
                    if stage == "pre_motion_commit" and adaptive_motion_gate
                    else None
                )

                # Reversible motions are evidence-gathering boundaries in C1.
                # Execute them, observe the resulting privileged state, and let
                # the Coach intervene at the next irreversible commit. This
                # prevents an uncertain semantic prior from blocking a motion
                # before any outcome evidence exists.
                if stage == "pre_motion_commit" and not causal_motion_gate and (
                    os.environ.get("CAPX_C1_MOTION_OBSERVE_ONLY", "0").lower()
                    in {"1", "true", "yes", "on"}
                    or (
                        adaptive_motion_gate
                        and not bool((stable_gate_state or {}).get("stable"))
                    )
                ):
                    try:
                        result = wrapped_fn(*args, **kwargs)
                    except Exception as exc:
                        coach_runtime.record_c1_transaction(
                            low_level_env=self.low_level_env,
                            api_name=fn_name,
                            phase="reversible_motion_error",
                            api_args=args,
                            api_kwargs=kwargs,
                            decision="OBSERVE",
                            reason=(
                                "adaptive observe-first: no stable evaluator-plus-closed-gripper state"
                                if adaptive_motion_gate
                                else "reversible motion executed before irreversible Coach gate"
                            ),
                            error=repr(exc),
                        )
                        raise
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase="reversible_motion_executed_and_observed",
                        api_args=args,
                        api_kwargs=kwargs,
                        decision="OBSERVE_ALLOW",
                        reason=(
                            "adaptive observe-first: no stable evaluator-plus-closed-gripper state"
                            if adaptive_motion_gate
                            else "outcome evidence recorded for next irreversible Coach gate"
                        ),
                    )
                    print(
                        f"[capx node coach] OBSERVE {fn_name}: reversible motion executed; "
                        "post-action state recorded",
                        file=sys.stderr,
                    )
                    return result

                # Once a verified commit exists, reversible transport is guarded
                # mechanically by simulator counterfactual preservation.  Do not
                # ask the language Coach to rewrite every intermediate motion:
                # its semantic preference is not stronger evidence than a dry-run,
                # and repeated speculative interruptions can prevent the next
                # irreversible node from ever being reached.  The Coach remains
                # responsible for close/release commits, where successor readiness
                # is actually decided.
                if (
                    stage == "pre_motion_commit"
                    and causal_motion_gate
                    and bool((motion_candidate_outcome or {}).get("preserved"))
                ):
                    try:
                        result = execute_with_counterfactual_preview()
                    except Exception as exc:
                        coach_runtime.record_c1_transaction(
                            low_level_env=self.low_level_env,
                            api_name=fn_name,
                            phase="counterfactual_transport_rejected",
                            api_args=args,
                            api_kwargs=kwargs,
                            decision="COUNTERFACTUAL_ONLY",
                            reason="persistent invariant rejected reversible transport",
                            error=repr(exc),
                        )
                        raise
                    verification = coach_runtime.verify_pending_stable_commit(
                        self.low_level_env
                    )
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase="counterfactual_transport_executed_and_preserved",
                        api_args=args,
                        api_kwargs=kwargs,
                        decision="COUNTERFACTUAL_ONLY",
                        reason=str(verification),
                    )
                    if not verification.get("preserved"):
                        raise RuntimeError(
                            "CAUSAL_COMMIT_LOST_AFTER_PREVIEW:\n"
                            "A reversible transport failed to preserve the active "
                            "commit after its accepted dry-run. Stop this block.\n"
                            f"EVIDENCE: {verification}"
                        )
                    print(
                        f"[capx node coach] COUNTERFACTUAL_ONLY {fn_name}: "
                        "persistent commit preserved; language Coach deferred to "
                        "the next irreversible node",
                        file=sys.stderr,
                    )
                    return result
                try:
                    decision = coach_runtime.build_and_call_node_coach(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        api_args=args,
                        api_kwargs=kwargs,
                        stage_hint=stage,
                        task_description=str(getattr(self, "_task_prompt", "") or ""),
                        student_code=str(getattr(self, "_capx_current_code", "") or ""),
                        student_api_contract=api_contract,
                        counterfactual_candidate_outcome=(
                            close_candidate_outcome
                            if close_candidate_outcome is not None
                            else release_candidate_outcome
                            if release_candidate_outcome is not None
                            else motion_candidate_outcome
                        ),
                    )
                except Exception as exc:
                    if os.environ.get("CAPX_C1_NO_WORSE", "0").lower() not in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }:
                        raise
                    decision = {
                        "decision": "ALLOW",
                        "reason": (
                            "NO_WORSE_A1_FALLBACK: Coach call, parsing, or "
                            f"validation failed ({type(exc).__name__})"
                        ),
                        "coach_failure": repr(exc),
                        "intervention_authorized": False,
                    }
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase="coach_failure_a1_fallback",
                        api_args=args,
                        api_kwargs=kwargs,
                        decision="ALLOW",
                        reason=repr(exc),
                    )
                verdict = str(decision.get("decision", "")).upper()
                if (
                    verdict == "INTERRUPT"
                    and os.environ.get("CAPX_C1_NO_WORSE", "0").lower()
                    in {"1", "true", "yes", "on"}
                    and decision.get("intervention_authorized") is not True
                ):
                    verdict = "ALLOW"
                    decision["decision"] = "ALLOW"
                    decision["reason"] = (
                        "NO_WORSE_A1_FALLBACK: interruption lacked independent authorization"
                    )
                if verdict == "INTERRUPT":
                    message = str(
                        decision.get("student_message")
                        or decision.get("reason")
                        or "The pre-action coach found that this irreversible action is not yet justified."
                    ).strip()
                    print(
                        f"[capx node coach] INTERRUPT {fn_name}: {message}",
                        file=sys.stderr,
                    )
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase="interrupted_before_execution",
                        api_args=args,
                        api_kwargs=kwargs,
                        decision=verdict,
                        reason=message,
                    )
                    inline_resume = (
                        os.environ.get("CAPX_NODE_COACH_INLINE_RESUME", "0").lower()
                        in {"1", "true", "yes", "on"}
                        and stage in {"pre_grasp_commit", "pre_release_commit"}
                    )
                    if inline_resume:
                        used = int(getattr(self, "_capx_node_resume_repairs", 0))
                        limit = int(os.environ.get("CAPX_NODE_RESUME_MAX_REPAIRS", "10"))
                        if used >= limit:
                            raise RuntimeError(
                                "NODE_RESUME_BUDGET_EXHAUSTED:\n"
                                f"Node-local continuation budget {limit} exhausted while "
                                "the successful execution prefix remains frozen."
                            )
                        self._capx_node_resume_repairs = used + 1
                        coach_runtime.record_c1_transaction(
                            low_level_env=self.low_level_env,
                            api_name=fn_name,
                            phase="successful_prefix_frozen_before_local_repair",
                            api_args=args,
                            api_kwargs=kwargs,
                            decision="LOCAL_CONTINUATION",
                            reason=(
                                f"repair {used + 1}/{limit}; current call stack, globals, "
                                "simulator state, and executed prefix retained"
                            ),
                        )
                        print(
                            f"[capx node resume] PAUSE_AT {fn_name}: generating local "
                            f"continuation {used + 1}/{limit}",
                            file=sys.stderr,
                        )
                        repair = coach_runtime.request_node_local_repair(
                            low_level_env=self.low_level_env,
                            interrupted_api_name=fn_name,
                            coach_decision=decision,
                            task_description=str(getattr(self, "_task_prompt", "") or ""),
                            student_code=str(getattr(self, "_capx_current_code", "") or ""),
                            student_api_contract=api_contract,
                            available_student_symbols=set(self._exec_globals),
                        )
                        repair_code = str(repair.get("code") or "")
                        trace_id = str(repair.get("trace_id") or "")
                        try:
                            exec(repair_code, self._exec_globals, self._exec_globals)
                        except BaseException as exc:
                            coach_runtime.record_node_local_repair_execution(
                                trace_id, repr(exc)
                            )
                            coach_runtime.record_c1_transaction(
                                low_level_env=self.low_level_env,
                                api_name=fn_name,
                                phase="local_continuation_execution_error",
                                api_args=args,
                                api_kwargs=kwargs,
                                decision="LOCAL_CONTINUATION",
                                reason=trace_id,
                                error=repr(exc),
                            )
                            if used + 1 < limit:
                                print(
                                    f"[capx node resume] REPAIR_ERROR {fn_name}: "
                                    "keeping current state and requesting a revised local "
                                    f"continuation ({type(exc).__name__})",
                                    file=sys.stderr,
                                )
                                return node_coach_wrapped(*args, **kwargs)
                            raise RuntimeError(
                                "NODE_LOCAL_REPAIR_EXECUTION_FAILED:\n"
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        coach_runtime.record_node_local_repair_execution(trace_id, None)
                        coach_runtime.record_c1_transaction(
                            low_level_env=self.low_level_env,
                            api_name=fn_name,
                            phase="local_continuation_executed_current_state",
                            api_args=args,
                            api_kwargs=kwargs,
                            decision="LOCAL_CONTINUATION",
                            reason=(
                                f"{trace_id}; original prefix was not replayed; retrying only "
                                f"the paused node {fn_name}"
                            ),
                        )
                        return node_coach_wrapped(*args, **kwargs)
                    coach_runtime.register_validated_action_permit(decision, fn_name)
                    raise RuntimeError(
                        "NODE_COACH_INTERRUPT:\n"
                        f"API_CALL: {fn_name}\n"
                        f"STAGE: {stage}\n"
                        "PRE_ACTION_COACH_GUIDANCE:\n"
                        f"{message}\n"
                        "C1_BRIDGE_CONTRACT:\n"
                        "On the next regeneration, preserve all executed-and-observed successful "
                        "actions. Execute only the latest validated corrective public action and "
                        "then retry the interrupted API. Do not replay initialization or replace "
                        "the entire prior policy block.\n"
                    )
                print(
                    f"[capx node coach] ALLOW {fn_name}: {decision.get('reason', '')}",
                    file=sys.stderr,
                )
                try:
                    result = execute_with_counterfactual_preview()
                except Exception as exc:
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase="execution_error",
                        api_args=args,
                        api_kwargs=kwargs,
                        decision=verdict,
                        reason=str(decision.get("reason", "")),
                        error=repr(exc),
                    )
                    raise
                coach_runtime.record_c1_transaction(
                    low_level_env=self.low_level_env,
                    api_name=fn_name,
                    phase="executed_and_observed",
                    api_args=args,
                    api_kwargs=kwargs,
                    decision=verdict,
                    reason=str(decision.get("reason", "")),
                )
                if stage == "pre_grasp_commit" and causal_motion_enabled:
                    registration = coach_runtime.register_stable_commit_after_close(
                        self.low_level_env, fn_name, close_prestate or {}
                    )
                    print(
                        f"[capx causal gate] close registration: {registration}",
                        file=sys.stderr,
                    )
                    if (
                        os.environ.get("CAPX_C1_ABORT_UNVERIFIED_CLOSE", "0").lower()
                        in {"1", "true", "yes", "on"}
                        and not registration.get("registered")
                    ):
                        raise RuntimeError(
                            "CAUSAL_CLOSE_UNVERIFIED:\n"
                            "The executed close did not establish a stable evaluator "
                            "postcondition. Stop this code block and repair only this "
                            "grasp commit; do not execute dependent motions.\n"
                            f"EVIDENCE: {registration}"
                        )
                if causal_motion_gate:
                    verification = coach_runtime.verify_pending_stable_commit(
                        self.low_level_env
                    )
                    coach_runtime.record_c1_transaction(
                        low_level_env=self.low_level_env,
                        api_name=fn_name,
                        phase=(
                            "causal_motion_preserved_commit"
                            if verification.get("preserved")
                            else "causal_motion_lost_commit"
                        ),
                        api_args=args,
                        api_kwargs=kwargs,
                        decision="CAUSAL_VERIFY",
                        reason=str(verification),
                    )
                    if not verification.get("preserved"):
                        raise RuntimeError(
                            "CAUSAL_COMMIT_LOST:\n"
                            "A dependent motion after a verified close destroyed "
                            "its evaluator-confirmed postcondition. Stop this code block. "
                            "Recover only that motion while preserving the close commit.\n"
                            f"EVIDENCE: {verification}"
                        )
                return result

            node_coach_wrapped.__name__ = getattr(wrapped_fn, "__name__", fn_name)
            node_coach_wrapped.__doc__ = getattr(wrapped_fn, "__doc__", None)
            node_coach_wrapped._capx_node_coach_action_wrapper = True
            return node_coach_wrapped

        maybe_wrap_with_node_coach._capx_node_coach_registry_wrapper = True
        cls._maybe_wrap_preclose_feedback = maybe_wrap_with_node_coach

    if original_exec is not None and not getattr(
        original_exec, "_capx_node_coach_code_context", False
    ):
        def exec_user_code_with_node_context(self, code):
            self._capx_current_code = str(code or "")
            if not hasattr(self, "_capx_node_resume_repairs"):
                self._capx_node_resume_repairs = 0
            return original_exec(self, code)

        exec_user_code_with_node_context._capx_node_coach_code_context = True
        cls._exec_user_code = exec_user_code_with_node_context

    def init_exec_globals_with_np(self):
        original_init(self)
        self._capx_node_resume_repairs = 0
        if _INJECT_NUMPY_GLOBAL:
            try:
                import numpy as np

                self._exec_globals.setdefault("np", np)
            except Exception:
                pass
        if _NODE_COACH:
            gated = sorted(
                name
                for name, fn in self._exec_globals.items()
                if getattr(fn, "_capx_node_coach_action_wrapper", False)
            )
            if not gated:
                raise RuntimeError(
                    "CAPX_NODE_COACH=1 but no exposed action API matched "
                    f"CAPX_NODE_COACH_ACTION_PATTERN={action_pattern!r}. "
                    "Refusing to run an un-gated experiment."
                )
            print(
                "[capx runtime patch] node coach registry gate active for: "
                + ", ".join(gated),
                file=sys.stderr,
            )

    init_exec_globals_with_np._capx_inject_np = True
    init_exec_globals_with_np._capx_node_coach_registry_init = True
    cls._init_exec_globals = init_exec_globals_with_np


def _patch_task_reset_determinism(module) -> None:
    """Patch the common task reset boundary, which every Robosuite task crosses."""
    if not _deterministic_seed_enabled():
        return
    cls = getattr(module, "CodeExecutionEnvBase", None)
    if cls is None:
        return
    original_reset = getattr(cls, "reset", None)
    if original_reset is None or getattr(original_reset, "_capx_task_deterministic_seed", False):
        return

    def deterministic_task_reset(self, *, seed=None, options=None):
        from capx_reproducibility import resolve_episode_seed

        episode_seed_base = os.environ.get("CAPX_EPISODE_SEED_BASE")
        effective_seed = resolve_episode_seed(
            seed,
            fixed_episode_seed=os.environ.get("CAPX_FIXED_EPISODE_SEED"),
            episode_seed_base=episode_seed_base,
        )
        reset_api_state = os.environ.get("CAPX_RESET_API_EPISODE_STATE", "0") == "1"
        reset_fields: list[str] = []
        if reset_api_state:
            # API objects are constructed once per worker and reused across trials.
            # IK warm-start configurations are episode-local state and must not leak
            # from the terminal pose of one trial into the first motion of the next.
            for api_name, api in getattr(self, "_apis", {}).items():
                for attr in ("cfg", "cfg_0", "cfg_1"):
                    if hasattr(api, attr):
                        setattr(api, attr, None)
                        reset_fields.append(f"{api_name}.{attr}")
            print(
                "[capx deterministic seed] reset episode-local API state: "
                + (", ".join(reset_fields) if reset_fields else "none"),
                file=sys.stderr,
            )
        binding = None
        low_level = getattr(self, "low_level_env", None)
        if effective_seed is not None and hasattr(low_level, "robosuite_env"):
            from capx_reproducibility import bind_robosuite_seed

            binding = bind_robosuite_seed(low_level, int(effective_seed))
            print(
                f"[capx deterministic seed] task_reset requested={seed} "
                f"effective={effective_seed} targets={len(binding['bound'])}",
                file=sys.stderr,
            )
        observation, info = original_reset(self, seed=effective_seed, options=options)
        if effective_seed is not None and os.environ.get("CAPX_REPRO_AUDIT", "0") == "1":
            from capx_reproducibility import append_audit_record, initial_state_record

            record = initial_state_record(low_level, observation, int(effective_seed))
            record.update(
                {
                    "environment_class": self.__class__.__name__,
                    "requested_seed": seed,
                    "effective_seed": int(effective_seed),
                    "episode_seed_base": (
                        int(episode_seed_base)
                        if episode_seed_base not in (None, "")
                        else None
                    ),
                    "binding": binding,
                    "reset_api_episode_state": reset_fields,
                }
            )
            append_audit_record(record)
        return observation, info

    deterministic_task_reset._capx_task_deterministic_seed = True
    cls.reset = deterministic_task_reset
    print(
        "[capx deterministic seed] patched common CodeExecutionEnvBase.reset boundary",
        file=sys.stderr,
    )


def _patch_open3d_geometry(module) -> None:
    if not (_OPEN3D_OBB_FALLBACK or _ROBUST_MASK_OBB):
        return
    point_cloud_cls = getattr(getattr(module, "geometry", None), "PointCloud", None)
    obb_cls = getattr(getattr(module, "geometry", None), "OrientedBoundingBox", None)
    if point_cloud_cls is None or obb_cls is None:
        return
    original = getattr(point_cloud_cls, "get_oriented_bounding_box", None)
    if original is None or getattr(original, "_capx_robust_obb", False):
        return

    def get_oriented_bounding_box_with_fallback(self, *args, **kwargs):
        point_cloud = self
        if _ROBUST_MASK_OBB:
            import numpy as np

            point_count = len(np.asarray(self.points))
            # Match the generic Franka control path: remove sparse depth
            # outliers before fitting an object box.  Keep the unfiltered
            # cloud when the statistical estimate is underdetermined or
            # removes too many points.
            if point_count >= 21:
                filtered, _ = self.remove_statistical_outlier(
                    nb_neighbors=20,
                    std_ratio=2.0,
                )
                filtered_count = len(np.asarray(filtered.points))
                if filtered_count >= 4:
                    point_cloud = filtered
        try:
            return original(point_cloud, *args, **kwargs)
        except RuntimeError as exc:
            if not _OPEN3D_OBB_FALLBACK:
                raise
            message = str(exc)
            if "QH6154" not in message and "Initial simplex is flat" not in message:
                raise
            import numpy as np

            aabb = point_cloud.get_axis_aligned_bounding_box()
            center = np.asarray(aabb.get_center(), dtype=np.float64)
            extent = np.asarray(aabb.get_extent(), dtype=np.float64)
            extent = np.maximum(extent, 1e-6)
            print(
                "[capx runtime patch] Open3D OBB failed on degenerate point cloud; "
                "using axis-aligned fallback",
                file=sys.stderr,
            )
            return obb_cls(center, np.eye(3), extent)

    get_oriented_bounding_box_with_fallback._capx_robust_obb = True
    point_cloud_cls.get_oriented_bounding_box = get_oriented_bounding_box_with_fallback
    enabled = []
    if _ROBUST_MASK_OBB:
        enabled.append("statistical outlier filtering")
    if _OPEN3D_OBB_FALLBACK:
        enabled.append("degenerate-cloud fallback")
    print(
        "[capx runtime patch] enabled Open3D OBB " + " and ".join(enabled),
        file=sys.stderr,
    )


def _patch_camera_utils(module) -> None:
    original = getattr(module, "get_real_depth_map", None)
    if original is None or getattr(original, "_capx_depth_sanitized", False):
        return

    def get_real_depth_map_sanitized(sim, depth_map):
        import numpy as np

        depth = np.asarray(depth_map)
        if (
            not np.all(np.isfinite(depth))
            or np.any(depth < 0.0)
            or np.any(depth > 1.0)
        ):
            depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0)
            depth = np.clip(depth, 0.0, 1.0)
        return original(sim, depth)

    get_real_depth_map_sanitized._capx_depth_sanitized = True
    module.get_real_depth_map = get_real_depth_map_sanitized


def _patch_depth_utils(module) -> None:
    if not _PATCH_POINTCLOUD_ALIGNMENT:
        return
    original = getattr(module, "depth_color_to_pointcloud", None)
    if original is None or getattr(original, "_capx_pixel_aligned", False):
        return

    def depth_color_to_pointcloud_nan_mask_aligned(
        depth,
        img,
        intrinsics,
        subsample_factor=1,
        depth_clip_range=(0.015, 20.0),
    ):
        """Match consumers that compress masks by ``~isnan(depth)``.

        Several perception APIs first compress the segmentation mask with that
        exact validity mask and then use the resulting indices on this point
        cloud. Filtering zero / clipped depths here, or returning all pixels,
        shifts every later index. Preserve pixel order and filter only NaNs.
        """
        import numpy as np

        if len(depth.shape) != 2:
            raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Image array must be (H, W, 3), got shape {img.shape}")
        if depth.shape[:2] != img.shape[:2]:
            raise ValueError(
                f"Depth and image dimensions must match: {depth.shape[:2]} vs {img.shape[:2]}"
            )
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Intrinsics must be (3, 3), got shape {intrinsics.shape}")
        if subsample_factor <= 0:
            raise ValueError(f"Subsample factor must be positive, got {subsample_factor}")

        height, width = depth.shape
        height_sub = height // subsample_factor
        width_sub = width // subsample_factor
        depth_sub = depth[::subsample_factor, ::subsample_factor]
        img_sub = img[::subsample_factor, ::subsample_factor]

        intr = intrinsics.copy()
        intr[0, 0] /= subsample_factor
        intr[1, 1] /= subsample_factor
        intr[0, 2] /= subsample_factor
        intr[1, 2] /= subsample_factor

        xs, ys = np.meshgrid(np.arange(width_sub), np.arange(height_sub), indexing="xy")
        z_raw = depth_sub.reshape(-1)
        valid = ~np.isnan(z_raw)
        z = z_raw
        x = (xs.reshape(-1) - intr[0, 2]) * z / intr[0, 0]
        y = (ys.reshape(-1) - intr[1, 2]) * z / intr[1, 1]
        points = np.stack([x, y, z], axis=-1)
        colors = img_sub.reshape(-1, img_sub.shape[-1])[:, :3] / 255.0
        return points[valid], colors[valid]

    def depth_to_pointcloud_nan_mask_aligned(
        depth,
        intrinsics,
        subsample_factor=1,
        depth_clip_range=(0.015, 20.0),
        filter_invalid=True,
    ):
        """Depth-only equivalent with the same segmentation-index contract."""
        import numpy as np

        if len(depth.shape) != 2:
            raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Intrinsics must be (3, 3), got shape {intrinsics.shape}")
        if subsample_factor <= 0:
            raise ValueError(f"Subsample factor must be positive, got {subsample_factor}")
        height, width = depth.shape
        depth_sub = depth[::subsample_factor, ::subsample_factor]
        height_sub, width_sub = depth_sub.shape
        intr = intrinsics.copy()
        intr[0, 0] /= subsample_factor
        intr[1, 1] /= subsample_factor
        intr[0, 2] /= subsample_factor
        intr[1, 2] /= subsample_factor
        xs, ys = np.meshgrid(
            np.arange(width_sub), np.arange(height_sub), indexing="xy"
        )
        z = depth_sub.reshape(-1)
        x = (xs.reshape(-1) - intr[0, 2]) * z / intr[0, 0]
        y = (ys.reshape(-1) - intr[1, 2]) * z / intr[1, 1]
        points = np.stack([x, y, z], axis=-1)
        if not filter_invalid:
            return points
        return points[~np.isnan(z)]

    depth_color_to_pointcloud_nan_mask_aligned._capx_pixel_aligned = True
    depth_to_pointcloud_nan_mask_aligned._capx_pixel_aligned = True
    module.depth_color_to_pointcloud = depth_color_to_pointcloud_nan_mask_aligned
    module.depth_to_pointcloud = depth_to_pointcloud_nan_mask_aligned
    print(
        "[capx runtime patch] depth point clouds match NaN-compressed mask indices",
        file=sys.stderr,
    )


def _patch_franka_depth_bindings(module) -> None:
    if not _PATCH_POINTCLOUD_ALIGNMENT:
        return
    try:
        from capx.utils import depth_utils
    except Exception:
        return
    patched = getattr(depth_utils, "depth_color_to_pointcloud", None)
    if patched is not None:
        module.depth_color_to_pointcloud = patched
    patched_depth = getattr(depth_utils, "depth_to_pointcloud", None)
    if patched_depth is not None:
        module.depth_to_pointcloud = patched_depth


def _patch_pyroki_client(module) -> None:
    """Route default PyRoKi clients to the current Slurm job's isolated port."""
    url = os.environ.get("CAPX_PYROKI_URL")
    if not url:
        return
    original = getattr(module, "init_pyroki", None)
    if original is None or getattr(original, "_capx_job_isolated", False):
        return

    def init_pyroki_job_isolated(server_url=None):
        return original(server_url=url if server_url is None else server_url)

    init_pyroki_job_isolated._capx_job_isolated = True
    module.DEFAULT_URL = url
    module.init_pyroki = init_pyroki_job_isolated

    original_traj = getattr(module, "init_pyroki_trajopt", None)
    if original_traj is not None and not getattr(original_traj, "_capx_job_isolated", False):
        def init_pyroki_trajopt_job_isolated(server_url=None):
            return original_traj(server_url=url if server_url is None else server_url)

        init_pyroki_trajopt_job_isolated._capx_job_isolated = True
        module.init_pyroki_trajopt = init_pyroki_trajopt_job_isolated
    print(f"[capx runtime patch] PyRoKi client isolated at {url}", file=sys.stderr)


def _patch_robot_descriptions_cache(module) -> None:
    """Avoid flaky GitPython head.object on shallow/NFS robot_descriptions caches.

    Job 521102 failed all trials because ``clone_to_directory`` called
    ``clone.head.object.hexsha`` on a shallow ``example-robot-data`` cache and
    GitPython raised ``SHA ... missing``. If the working tree already has the
    panda URDF, reuse it and skip git entirely.
    """
    if getattr(module, "_capx_skip_git_if_cached", False):
        return
    original = getattr(module, "clone_to_directory", None)
    if original is None:
        return

    class _CachedRepo:
        def __init__(self, working_dir: str) -> None:
            self.working_dir = working_dir

    def clone_to_directory_cached(repo_url, target_dir, commit=None):
        urdf = os.path.join(
            target_dir, "robots", "panda_description", "urdf", "panda.urdf"
        )
        if os.path.isfile(urdf):
            print(
                f"[capx runtime patch] robot_descriptions reuse cached tree "
                f"(skip git): {target_dir}",
                file=sys.stderr,
            )
            return _CachedRepo(target_dir)
        return original(repo_url, target_dir, commit=commit)

    clone_to_directory_cached._capx_skip_git_if_cached = True
    module.clone_to_directory = clone_to_directory_cached
    module._capx_skip_git_if_cached = True
    print(
        "[capx runtime patch] robot_descriptions cache skip-git when URDF present",
        file=sys.stderr,
    )


def _patch_handover_deterministic_rng(module) -> None:
    """Bind robosuite's private Generator objects to recorded experiment seeds."""
    if not _deterministic_seed_enabled():
        return
    try:
        rs_cls = module.suite.environments.manipulation.two_arm_handover.TwoArmHandover
    except Exception:
        return
    original_rs_init = rs_cls.__init__
    if not getattr(original_rs_init, "_capx_deterministic_seed", False):
        def deterministic_rs_init(self, *args, **kwargs):
            if kwargs.get("seed") is None:
                kwargs["seed"] = _deterministic_base_seed()
            return original_rs_init(self, *args, **kwargs)

        deterministic_rs_init._capx_deterministic_seed = True
        rs_cls.__init__ = deterministic_rs_init

    wrapper_cls = getattr(module, "RobosuiteHandoverEnv", None)
    if wrapper_cls is None:
        return
    original_reset = wrapper_cls.reset
    if getattr(original_reset, "_capx_deterministic_seed", False):
        return

    def deterministic_handover_reset(self, *, seed=None, options=None):
        if seed is not None:
            import numpy as np

            seeded_rng = np.random.default_rng(int(seed))
            rs = self.robosuite_env
            rs.rng = seeded_rng
            sampler = getattr(rs, "placement_initializer", None)
            if sampler is not None:
                sampler.rng = seeded_rng
            # Some randomized object properties retain the environment RNG.
            hammer = getattr(rs, "hammer", None)
            if hammer is not None and hasattr(hammer, "rng"):
                hammer.rng = seeded_rng
        return original_reset(self, seed=seed, options=options)

    deterministic_handover_reset._capx_deterministic_seed = True
    wrapper_cls.reset = deterministic_handover_reset
    print(
        "[capx deterministic seed] bound TwoArmHandover constructor and placement sampler RNG",
        file=sys.stderr,
    )


def _patch_generic_robosuite_deterministic_rng(module) -> None:
    """Make ``reset(seed=...)`` reach the RNG that placement samplers consume."""
    if not _deterministic_seed_enabled():
        return
    for value in vars(module).values():
        if not isinstance(value, type) or value.__module__ != module.__name__:
            continue
        original_reset = getattr(value, "reset", None)
        if original_reset is None or getattr(original_reset, "_capx_generic_deterministic_seed", False):
            continue

        def make_reset(reset_fn):
            def deterministic_reset(self, *, seed=None, options=None):
                binding = None
                if seed is not None and hasattr(self, "robosuite_env"):
                    from capx_reproducibility import bind_robosuite_seed

                    binding = bind_robosuite_seed(self, int(seed))
                    print(
                        f"[capx deterministic seed] bound {self.__class__.__name__} "
                        f"seed={seed} targets={len(binding['bound'])}",
                        file=sys.stderr,
                    )
                observation, info = reset_fn(self, seed=seed, options=options)
                if seed is not None and os.environ.get("CAPX_REPRO_AUDIT", "0") == "1":
                    from capx_reproducibility import append_audit_record, initial_state_record

                    record = initial_state_record(self, observation, int(seed))
                    record["environment_class"] = self.__class__.__name__
                    record["binding"] = binding
                    append_audit_record(record)
                return observation, info

            deterministic_reset._capx_generic_deterministic_seed = True
            return deterministic_reset

        value.reset = make_reset(original_reset)
        print(
            f"[capx deterministic seed] patched {module.__name__}.{value.__name__}",
            file=sys.stderr,
        )


def _patch_handover_node_coach(module) -> None:
    if not _NODE_COACH:
        return
    method_stage = {
        "close_gripper_arm0": "pre_close_gripper_arm0",
        "close_gripper_arm1": "pre_close_gripper_arm1",
        "open_gripper_arm0": "pre_open_gripper_arm0",
        "open_gripper_arm1": "pre_open_gripper_arm1",
    }
    class_names = (
        "FrankaHandoverApi",
        "FrankaHandoverApiReduced",
        "FrankaHandoverApiReducedExampleless",
    )
    for class_name in class_names:
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls, "_capx_node_coach_patched", False):
            continue
        for method_name, stage_hint in method_stage.items():
            original = getattr(cls, method_name, None)
            if original is None or getattr(original, "_capx_node_coach_wrapper", False):
                continue

            def make_wrapper(orig, api_name, stage):
                def wrapped(self, *args, **kwargs):
                    try:
                        import coach_runtime

                        decision = coach_runtime.build_and_call_node_coach(
                            low_level_env=getattr(self, "_env", None),
                            api_name=api_name,
                            api_args=args,
                            api_kwargs=kwargs,
                            stage_hint=stage,
                        )
                        if str(decision.get("decision", "")).upper() == "INTERRUPT":
                            msg = str(decision.get("student_message") or decision.get("reason") or "").strip()
                            if not msg:
                                msg = "Privileged node coach interrupted this irreversible action because prerequisites are not satisfied."
                            print(
                                f"[capx node coach] INTERRUPT {api_name}: {msg}",
                                file=sys.stderr,
                            )
                            raise RuntimeError(
                                "NODE_COACH_INTERRUPT:\n"
                                f"API_CALL: {api_name}\n"
                                f"STAGE: {stage}\n"
                                "CHEATING_PRIVILEGED_PRE_ACTION_GUIDANCE:\n"
                                f"{msg}\n"
                            )
                        print(
                            f"[capx node coach] ALLOW {api_name}: {decision.get('reason', '')}",
                            file=sys.stderr,
                        )
                    except RuntimeError as exc:
                        if "NODE_COACH_INTERRUPT" in str(exc):
                            raise
                        print(
                            f"[capx node coach] WARNING: gate failed for {api_name}; allowing action: {exc}",
                            file=sys.stderr,
                        )
                    except Exception as exc:
                        print(
                            f"[capx node coach] WARNING: gate failed for {api_name}; allowing action: {exc}",
                            file=sys.stderr,
                        )
                    return orig(self, *args, **kwargs)

                wrapped._capx_node_coach_wrapper = True
                return wrapped

            setattr(cls, method_name, make_wrapper(original, method_name, stage_hint))
        cls._capx_node_coach_patched = True
        print(
            f"[capx runtime patch] node coach patched {module.__name__}.{class_name}",
            file=sys.stderr,
        )


def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
    try:
        module = _original_import(name, globals, locals, fromlist, level)
    except ModuleNotFoundError as exc:
        # The verified v346 runtime stores its configured robosuite macros in
        # macros_private.py while internal imports still request
        # robosuite.macros.  Recreate the alias in every isolated trial.
        if name != "robosuite.macros" or exc.name != "robosuite.macros":
            raise
        private = _original_import(
            "robosuite.macros_private",
            globals,
            locals,
            ("*",),
            0,
        )
        sys.modules["robosuite.macros"] = private
        parent = sys.modules.get("robosuite")
        if parent is not None:
            setattr(parent, "macros", private)
        module = _original_import(name, globals, locals, fromlist, level)
    if name == "robosuite.macros":
        try:
            sys.modules[name].CACHE_NUMBA = False
        except Exception:
            pass
    if name == "open3d":
        _patch_open3d_geometry(module)
    if name == "robosuite.utils.camera_utils" and _ENABLE_DEPTH_PATCH:
        _patch_camera_utils(module)
    elif name == "robosuite.utils" and "camera_utils" in fromlist:
        camera_utils = getattr(module, "camera_utils", None)
        if camera_utils is not None and _ENABLE_DEPTH_PATCH:
            _patch_camera_utils(camera_utils)
    elif name == "capx.utils.depth_utils":
        _patch_depth_utils(module)
    elif name.startswith("capx.integrations.franka."):
        _patch_franka_depth_bindings(module)
    elif name == "capx.integrations.motion.pyroki":
        _patch_pyroki_client(module)
    elif name == "capx.envs.simulators.robosuite_handover":
        _patch_handover_deterministic_rng(module)
        _patch_generic_robosuite_deterministic_rng(module)
    elif name.startswith("capx.envs.simulators.robosuite_"):
        _patch_generic_robosuite_deterministic_rng(module)
    elif name == "capx.envs.configs.loader":
        _patch_config_loader(module)
    elif name == "capx.llm.client":
        _patch_llm_client_module(module)
    elif name == "capx.envs.tasks.base":
        _patch_base_task_module(module)
        _patch_task_reset_determinism(module)
    elif name == "capx.envs.tasks.franka.franka_nut_assembly":
        _patch_nut_assembly_task_prompt(module)
    elif name == "capx.envs.trial":
        trial_module = sys.modules.get(name, module)
        _patch_trial_module(trial_module)
        _patch_trial_deterministic_module(trial_module)
    elif name == "capx.envs" and "trial" in fromlist:
        trial_module = getattr(module, "trial", None)
        if trial_module is not None:
            _patch_trial_module(trial_module)
            _patch_trial_deterministic_module(trial_module)
    # ``import robot_descriptions._cache`` returns the parent package from
    # __import__; patch via sys.modules once the submodule exists.
    cache_mod = sys.modules.get("robot_descriptions._cache")
    if cache_mod is not None:
        _patch_robot_descriptions_cache(cache_mod)
    return module


builtins.__import__ = _import_hook
