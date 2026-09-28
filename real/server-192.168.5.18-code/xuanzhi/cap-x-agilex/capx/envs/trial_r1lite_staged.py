"""Bounded ROS2 R1 Lite staged execution for tennis-ball-in-box."""

from __future__ import annotations

import copy
import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.trial import _annotate_code_blocks, _build_log_lines
from capx.llm.client import query_model
from capx.utils.launch_utils import TrialSummary, _extract_code, _save_trial_artifacts

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


R1_STAGES = (
    "perceive",
    "graspnet_plan",
    "approach",
    "grasp_gate",
    "transfer",
    "place",
    "verify",
)

R1_STEER = {
    "perceive": (
        "STATE: arm is at the R1 safe home pose. TARGET: detect and cache the box and tennis "
        "ball using the fixed head camera before any arm motion. EVIDENCE: finite base-frame "
        "positions. EXECUTION: perception only. RISK: do not invent coordinates."
    ),
    "graspnet_plan": (
        "STATE: head perception is cached and a constrained dynamic wrist-view path is validated. "
        "TARGET: move with the fixed safe camera orientation, require post-motion RGB-D, refine "
        "the ball position, then require a calibrated Contact-GraspNet plan with full approach IK. "
        "EVIDENCE: head/wrist agreement and a safe candidate. RISK: no fixed-pose fallback."
    ),
    "approach": (
        "STATE: a validated GraspNet grasp and pregrasp exist. TARGET: goto the selected "
        "pregrasp, confirm the ball has not moved, then advance in bounded steps along the "
        "selected GraspNet approach axis. EVIDENCE: each waypoint is logged. EXECUTION: open "
        "gripper only. RISK: do not alter the GraspNet orientation or close early."
    ),
    "grasp_gate": (
        "STATE: gripper is aligned at grasp_pos. TARGET: close_gripper(), wait one second, "
        "call verify_grasp(), and set RESULT={'grasp_verified': bool(ok), 'details': details}. "
        "EVIDENCE: true only when verify_grasp reports retained_gripper_gap. "
        "EXECUTION: gripper only. RISK: if false, open_gripper() then raise RuntimeError; "
        "never move toward the box."
    ),
    "transfer": (
        "STATE: grasp_verified is True. TARGET: lift vertically, move to a safe height, then "
        "move XY above the box while retaining the closed gripper. EVIDENCE: define lift_pos, safe_z, "
        "safe_drop_xy_pos. RISK: no diagonal carry and no release."
    ),
    "place": (
        "STATE: retained ball is above the box. TARGET: descend vertically to drop_pos, open "
        "the gripper, then retreat vertically. EVIDENCE: RESULT={'stage_complete': True}. "
        "RISK: keep XY fixed while descending and do not re-perceive."
    ),
    "verify": (
        "STATE: arm has retreated. TARGET: call verify_tennis_in_box() and set "
        "RESULT={'success': bool(ok), 'details': details}. EXECUTION: verification only. "
        "RISK: no arm or gripper motion; an unsuccessful verification is failure."
    ),
}


def parse_r1_stage_plan(content: str | None) -> tuple[str, ...]:
    """Validate planner output so an LLM cannot skip the grasp gate."""
    payload = json.loads((content or "").strip())
    stages = payload.get("stages") if isinstance(payload, dict) else payload
    ids = tuple(item.get("id") if isinstance(item, dict) else item for item in stages)
    if ids != R1_STAGES:
        raise ValueError(f"expected fixed R1 stage order {R1_STAGES}, got {ids}")
    return ids


def build_r1_steer_prompt(stage: str) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [{"type": "text", "text": (
            "Generate only executable Python for this single ROS2 Galaxea/Xinghaitu R1 Lite "
            f"subtask: {stage}.\n{R1_STEER[stage]}\n"
            "Persistent variables from earlier stages remain available. Available APIs: "
            "open_gripper, close_gripper, get_object_pose, plan_tennis_graspnet, "
            "plan_wrist_observation_pose, execute_wrist_observation_plan, "
            "refine_tennis_position_wrist, "
            "goto_pose, descend_z_in_steps, verify_grasp, verify_tennis_in_box. "
            "Import numpy as np and time if needed. "
            "Do not include Markdown, prose, later stages, or alternative plans."
        )}],
    }]


def build_r1_plan_prompt() -> list[dict[str, Any]]:
    """Ask the LLM for a high-level plan without exposing motor commands."""
    return [{
        "role": "user",
        "content": [{"type": "text", "text": (
            "Plan a real Galaxea R1 Lite task: pick up one tennis ball and place it "
            "in the cardboard box. Return JSON only with key `stages`. The stage IDs "
            "must be exactly perceive, graspnet_plan, approach, grasp_gate, transfer, place, verify "
            "in that order. You may add goal and gate descriptions, but do not output "
            "coordinates, joint values, Python, ROS topics, or motor commands."
        )}],
    }]


def _stage_ok(stage: str, info: dict[str, Any], env: CodeExecutionEnvBase) -> bool:
    if info.get("sandbox_rc", 1) != 0 or str(info.get("stderr", "")).strip():
        return False
    g = env._exec_globals
    if stage == "perceive":
        return all(
            name in g and np.isfinite(np.asarray(g[name], dtype=float)).all()
            for name in ("box_pos", "drop_pos", "initial_ball_pos")
        )
    if stage == "graspnet_plan":
        return (
            isinstance(g.get("graspnet_plan"), dict)
            and g["graspnet_plan"].get("mandatory_graspnet") is True
            and g["graspnet_plan"].get("fallback_used") is False
        )
    if stage in {"approach", "place"}:
        return g.get("RESULT") == {"stage_complete": True}
    if stage == "grasp_gate":
        return isinstance(g.get("RESULT"), dict) and g["RESULT"].get("grasp_verified") is True
    if stage == "transfer":
        return all(name in g for name in ("lift_pos", "safe_z", "safe_drop_xy_pos"))
    return isinstance(g.get("RESULT"), dict) and g["RESULT"].get("success") is True


def _validate_stage_code(stage: str, code: str) -> None:
    """Reject generated code that attempts to cross a stage safety boundary."""
    forbidden = {
        "perceive": ("goto_pose(", "descend_z_in_steps(", "close_gripper("),
        "graspnet_plan": ("goto_pose(", "descend_z_in_steps(", "close_gripper("),
        "approach": ("close_gripper(", "verify_grasp(", "verify_tennis_in_box("),
        "grasp_gate": ("goto_pose(", "descend_z_in_steps("),
        "transfer": ("open_gripper(", "verify_tennis_in_box("),
        "place": ("get_object_pose(", "verify_tennis_in_box("),
        "verify": ("goto_pose(", "descend_z_in_steps(", "open_gripper(", "close_gripper("),
    }[stage]
    violations = [token for token in forbidden if token in code]
    if violations:
        raise ValueError(f"{stage} code crossed safety boundary: {violations}")


def _return_to_initial_pose(api: Any) -> None:
    """Recover with servo power retained, then leave the gripper open."""
    api.move_joints_safe(
        api._r1.safe_joint_home,
        expected_feedback=api._r1.safe_joint_home_feedback,
    )
    api.open_gripper()


def _abort(env: CodeExecutionEnvBase, reason: str) -> None:
    for api in env._apis.values():
        if hasattr(api, "move_joints_safe") and hasattr(api, "open_gripper"):
            try:
                _return_to_initial_pose(api)
                print("[r1_staged] recovered to initial pose with servo power retained")
            except Exception as recovery_exc:
                print(f"[r1_staged] initial-pose recovery failed: {recovery_exc!r}")
            break
    print(f"[r1_staged] ABORT: {reason}")


def _require_wrist_tennis(
    api: Any,
    label: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Hard-gate motion on a fresh wrist frame containing the tennis ball."""
    ok, details = api.check_wrist_tennis(
        label=label,
        min_pixels=int(config.get("r1lite_wrist_min_tennis_pixels", 30)),
        max_age_s=float(config.get("r1lite_wrist_max_age_s", 1.0)),
    )
    if not ok:
        raise RuntimeError(f"{label} wrist gate rejected: {details}")
    return details


def run_r1_staged_trial(
    env: CodeExecutionEnvBase, trial: int, args: LaunchArgs, config: dict[str, Any]
) -> TrialSummary:
    """Execute fixed R1 stages; retries are permitted only before arm motion."""
    started = time.time()
    if not bool(config.get("r1lite_graspnet_mandatory", False)):
        raise RuntimeError("R1 staged tennis execution requires mandatory GraspNet")
    if bool(config.get("r1lite_graspnet_allow_fallback", False)):
        raise RuntimeError("R1 staged tennis execution forbids grasp-pose fallback")
    env.reset(options={"trial": trial}, seed=trial)
    api = next(
        (candidate for candidate in env._apis.values() if hasattr(candidate, "goto_pose")),
        None,
    )
    if api is None:
        raise RuntimeError("R1 control API is unavailable")

    if bool(config.get("r1lite_staged_use_fixed_plan", False)):
        fixed_plan = json.dumps(
            {"stages": [{"id": stage} for stage in R1_STAGES]},
            ensure_ascii=False,
        )
        planner_response = {"content": fixed_plan}
    else:
        planner_response = query_model(args, build_r1_plan_prompt())
    planned_stages = parse_r1_stage_plan(planner_response.get("content"))

    retries = max(1, int(config.get("r1lite_perception_attempts", 3)))
    margin = float(config.get("r1lite_workspace_margin_m", 0.03))
    grasp_gap = float(config.get("r1lite_min_grasp_gap", 0.35))
    responses: list[dict[str, Any]] = [{
        "decision": (
            "fixed_stage_plan"
            if config.get("r1lite_staged_use_fixed_plan", False)
            else "llm_gated_plan"
        ),
        "model": args.model,
        "stages": planned_stages,
        "planner_response": planner_response.get("content"),
    }]
    code_blocks: list[str] = []
    info: dict[str, Any] = {"sandbox_rc": 1, "stdout": "", "stderr": ""}

    # Head perception is the only retryable phase. No arm command is issued here.
    perceived: dict[str, Any] | None = None
    try:
        api.open_gripper()
        validation_enabled = bool(
            getattr(api._r1, "r1lite_motion_validation_enabled", True)
        )
        for attempt in range(1, retries + 1):
            try:
                api.save_head_rgb(f"outputs/r1lite/staged_perceive_{attempt:02d}.png")
                box_pos, _box_quat = api.get_object_pose("cardboard box")
                initial_ball_pos, _head_quat = api.get_object_pose("tennis ball")
                box_pos = np.asarray(box_pos, dtype=float).reshape(3)
                initial_ball_pos = np.asarray(initial_ball_pos, dtype=float).reshape(3)
                x_limits = getattr(api._r1, "workspace_x_limits")
                y_limits = getattr(api._r1, "workspace_y_limits")
                for name, pos in (
                    ("box", box_pos),
                    ("head_ball", initial_ball_pos),
                ):
                    if not np.isfinite(pos).all():
                        raise RuntimeError(f"{name} pose is non-finite")
                    if validation_enabled and not (
                        x_limits[0] + margin <= pos[0] <= x_limits[1] - margin
                    ):
                        raise RuntimeError(f"{name} x={pos[0]:.3f} violates workspace margin")
                    if validation_enabled and not (
                        y_limits[0] + margin <= pos[1] <= y_limits[1] - margin
                    ):
                        raise RuntimeError(f"{name} y={pos[1]:.3f} violates workspace margin")
                if (
                    validation_enabled
                    and np.linalg.norm(box_pos[:2] - initial_ball_pos[:2]) < 0.10
                ):
                    raise RuntimeError("ball and box are too close for a safe vertical approach")
                perceived = {
                    "box_pos": box_pos,
                    "initial_ball_pos": initial_ball_pos,
                }
                responses.append({
                    "stage": "head_perceive",
                    "attempt": attempt,
                    "success": True,
                })
                break
            except Exception as exc:
                responses.append({"stage": "perceive", "attempt": attempt, "success": False, "error": repr(exc)})
                if attempt == retries:
                    raise
                time.sleep(0.5)

        assert perceived is not None
        box_pos = perceived["box_pos"]
        initial_ball_pos = perceived["initial_ball_pos"]
        drop_pos = box_pos + np.array([0.0, 0.0, 0.10])

        fast_motion = not validation_enabled
        pose_hold_s = 0.25 if fast_motion else 1.2
        approach_hold_s = 0.20 if fast_motion else 0.8
        descend_steps = 4 if fast_motion else 12
        grasp_quat = np.asarray(api._default_tool_down_quat_xyzw(), dtype=float)
        head_pregrasp_height_m = float(
            config.get(
                "r1lite_head_pregrasp_height_m",
                config.get("r1lite_topdown_pregrasp_height_m", 0.10),
            )
        )
        head_pre_grasp_pos = initial_ball_pos + np.array(
            [0.0, 0.0, head_pregrasp_height_m],
            dtype=float,
        )
        head_pregrasp_feedback = api.goto_pose(
            head_pre_grasp_pos,
            grasp_quat,
            gripper=100.0,
            hold_sec=pose_hold_s,
        )
        head_pregrasp_barrier_ns = api._r1.wait_for_arm_stable()
        responses.append({
            "stage": "head_pregrasp_approach",
            "target_position": head_pre_grasp_pos.tolist(),
            "head_ball_position": initial_ball_pos.tolist(),
            "height_above_head_ball_m": head_pregrasp_height_m,
            "arm_feedback": head_pregrasp_feedback,
            "post_motion_barrier_ns": head_pregrasp_barrier_ns,
        })
        refined_ball = api.refine_tennis_position_wrist(
            initial_ball_pos,
            newer_than_ns=head_pregrasp_barrier_ns,
            save_debug=True,
        )
        responses.append({
            "stage": "wrist_tennis_refine",
            "result": refined_ball,
        })
        if not refined_ball.get("center_margin_ok", False):
            raise RuntimeError(
                "wrist refine rejected after head pregrasp: tennis ball is not "
                f"centered in wrist RGB-D ({refined_ball})"
            )
        if not refined_ball.get("coarse_error_ok", False):
            responses.append({
                "stage": "wrist_refine_override",
                "reason": "head_wrist_position_disagreement",
                "policy": "use_wrist_refine_position",
                "coarse_error_m": refined_ball.get("coarse_error_m"),
                "head_position_base": refined_ball.get("expected_position_base"),
                "wrist_position_base": refined_ball.get("position_base"),
            })

        # Physical motion has started. GraspNet/TF/IK failure is terminal and
        # is never retried with the fixed observation pose or a pose fallback.
        graspnet_plan = api.plan_tennis_graspnet(
            save_debug=True,
            confirm_head=False,
            expected_ball_position_base=refined_ball["position_base"],
        )
        if not graspnet_plan.get("ok") or not graspnet_plan.get(
            "mandatory_graspnet"
        ):
            raise RuntimeError(f"mandatory GraspNet planning failed: {graspnet_plan}")
        selected_grasp = graspnet_plan["selected"]
        grasp_pos = np.asarray(
            selected_grasp["grasp_position"],
            dtype=float,
        ).reshape(3)
        grasp_quat = np.asarray(
            selected_grasp["grasp_quaternion_xyzw"],
            dtype=float,
        ).reshape(4)
        pre_grasp_pos = np.asarray(
            selected_grasp["pregrasp_position"],
            dtype=float,
        ).reshape(3)
        approach_axis = np.asarray(
            selected_grasp["approach_axis_base"],
            dtype=float,
        ).reshape(3)
        if not validation_enabled:
            ball_pos = np.asarray(refined_ball["position_base"], dtype=float).reshape(3)
            forward_offset_m = float(
                config.get("r1lite_topdown_grasp_forward_offset_m", 0.0)
            )
            z_offset_m = float(config.get("r1lite_topdown_grasp_z_offset_m", 0.0))
            pregrasp_height_m = float(
                config.get("r1lite_topdown_pregrasp_height_m", 0.10)
            )
            original_grasp_pos = grasp_pos.copy()
            grasp_pos = np.array(
                [
                    ball_pos[0] + forward_offset_m,
                    ball_pos[1],
                    ball_pos[2] + z_offset_m,
                ],
                dtype=float,
            )
            pre_grasp_pos = grasp_pos + np.array([0.0, 0.0, pregrasp_height_m])
            grasp_quat = np.asarray(api._default_tool_down_quat_xyzw(), dtype=float)
            approach_axis = np.array([0.0, 0.0, -1.0], dtype=float)
            selected_grasp = {
                **selected_grasp,
                "capx_topdown_override": True,
                "original_grasp_position": original_grasp_pos.tolist(),
                "grasp_position": grasp_pos.tolist(),
                "pregrasp_position": pre_grasp_pos.tolist(),
                "grasp_quaternion_xyzw": grasp_quat.tolist(),
                "approach_axis_base": approach_axis.tolist(),
                "topdown_forward_offset_m": forward_offset_m,
                "topdown_z_offset_m": z_offset_m,
                "topdown_pregrasp_height_m": pregrasp_height_m,
            }
        x_limits = getattr(api._r1, "workspace_x_limits")
        y_limits = getattr(api._r1, "workspace_y_limits")
        for name, pos in (
            ("graspnet_grasp", grasp_pos),
            ("graspnet_pregrasp", pre_grasp_pos),
        ):
            if not np.isfinite(pos).all():
                raise RuntimeError(f"{name} pose is non-finite")
            if validation_enabled and not (
                x_limits[0] + margin <= pos[0] <= x_limits[1] - margin
            ):
                raise RuntimeError(f"{name} x={pos[0]:.3f} violates workspace margin")
            if validation_enabled and not (
                y_limits[0] + margin <= pos[1] <= y_limits[1] - margin
            ):
                raise RuntimeError(f"{name} y={pos[1]:.3f} violates workspace margin")
        responses.append({"stage": "graspnet_plan", "graspnet": graspnet_plan})

        # From here on, any error is terminal: do not repeat a physical grasp.
        approach_feedback = api.goto_pose(
            pre_grasp_pos,
            grasp_quat,
            gripper=100.0,
            hold_sec=pose_hold_s,
        )
        # The R1 Lite wrist camera faces away from the fingertips in the
        # tool-down pose, so it cannot see a ball below the gripper.  Recheck
        # alignment with the fixed head camera instead.
        pre_descent_ball_pos, _ = api.get_object_pose(
            "tennis ball", save_debug=True
        )
        pre_descent_ball_pos = np.asarray(pre_descent_ball_pos, dtype=float).reshape(3)
        pre_descent_xy_error = float(
            np.linalg.norm(pre_descent_ball_pos[:2] - initial_ball_pos[:2])
        )
        if pre_descent_xy_error > 0.05:
            raise RuntimeError(
                "tennis ball moved before descent: "
                f"xy_error={pre_descent_xy_error:.3f}m"
            )
        responses.append({
            "stage": "approach",
            "arm_feedback": approach_feedback,
            "head_ball_position": pre_descent_ball_pos.tolist(),
            "head_ball_xy_error_m": pre_descent_xy_error,
        })
        approach_steps = max(
            2,
            int(config.get("r1lite_graspnet_approach_steps", 12)),
        )
        if fast_motion:
            approach_steps = 1
        approach_waypoints: list[dict[str, Any]] = []
        for step in range(1, approach_steps + 1):
            fraction = step / approach_steps
            waypoint = pre_grasp_pos + fraction * (grasp_pos - pre_grasp_pos)
            feedback = api.goto_pose(
                waypoint,
                grasp_quat,
                gripper=100.0,
                hold_sec=approach_hold_s,
            )
            approach_waypoints.append({
                "step": step,
                "position": waypoint.tolist(),
                "feedback": feedback,
            })
        responses.append({
            "stage": "graspnet_approach",
            "approach_axis_base": approach_axis.tolist(),
            "waypoints": approach_waypoints,
        })
        close_feedback = api.close_gripper()
        time.sleep(1.0)
        grasp_ok, grasp_details = api.verify_grasp(min_gap_above_close=grasp_gap)
        responses.append({
            "stage": "grasp_gate",
            "success": grasp_ok,
            "close_feedback": close_feedback,
            "details": grasp_details,
        })
        if not grasp_ok:
            raise RuntimeError(f"grasp gate rejected: {grasp_details}")

        lift_pos = grasp_pos + np.array([0.0, 0.0, 0.18])
        safe_z = max(float(lift_pos[2]), float(drop_pos[2] + 0.12))
        lift_feedback = api.goto_pose(lift_pos, grasp_quat, hold_sec=pose_hold_s)
        retained_ok, retained_details = api.verify_grasp(
            min_gap_above_close=grasp_gap,
            save_debug=False,
        )
        post_lift_ball: dict[str, Any]
        try:
            detected_pos, _ = api.get_object_pose("tennis ball", save_debug=True)
            detected_pos = np.asarray(detected_pos, dtype=float).reshape(3)
            table_xy_error = float(
                np.linalg.norm(detected_pos[:2] - initial_ball_pos[:2])
            )
            ball_removed = table_xy_error > 0.08
            post_lift_ball = {
                "detected": True,
                "position": detected_pos.tolist(),
                "distance_from_original_xy_m": table_xy_error,
                "removed_from_original": ball_removed,
            }
        except Exception as exc:
            # A ball held below the gripper is commonly occluded in the fixed
            # head view.  Accept absence only together with the gripper gap.
            ball_removed = True
            post_lift_ball = {
                "detected": False,
                "removed_from_original": True,
                "perception_error": repr(exc),
            }
        responses.append({
            "stage": "post_lift_gate",
            "arm_feedback": lift_feedback,
            "head_ball": post_lift_ball,
            "grasp": retained_details,
        })
        if validation_enabled and (not ball_removed or not retained_ok):
            raise RuntimeError(
                f"post-lift gate rejected: head={post_lift_ball}, grasp={retained_details}"
            )
        api.goto_pose(
            np.array([lift_pos[0], lift_pos[1], safe_z]),
            grasp_quat,
            hold_sec=pose_hold_s,
        )
        api.goto_pose(
            np.array([drop_pos[0], drop_pos[1], safe_z]),
            grasp_quat,
            hold_sec=pose_hold_s,
        )
        responses.append({"stage": "transfer", "steer": R1_STEER["transfer"]})

        pre_drop_pos = drop_pos + np.array([0.0, 0.0, 0.10])
        api.goto_pose(pre_drop_pos, grasp_quat, hold_sec=pose_hold_s)
        api.descend_z_in_steps(
            drop_pos,
            grasp_quat,
            start_z=float(pre_drop_pos[2]),
            num_steps=descend_steps,
            hold_sec_per_step=approach_hold_s,
        )
        api.open_gripper()
        time.sleep(0.5)
        api.goto_pose(
            drop_pos + np.array([0.0, 0.0, 0.15]),
            grasp_quat,
            hold_sec=pose_hold_s,
        )
        responses.append({"stage": "place", "steer": R1_STEER["place"]})
        success, details = api.verify_tennis_in_box()
        responses.append({"stage": "verify", "success": bool(success), "details": details})
        if not success:
            raise RuntimeError(f"final verification failed: {details}")
        _return_to_initial_pose(api)
        responses.append({
            "stage": "return_to_initial_pose",
            "servo_power_retained": True,
        })
        info = {"sandbox_rc": 0, "stdout": "deterministic staged execution complete", "stderr": ""}
    except Exception as exc:
        info = {"sandbox_rc": 1, "stdout": "", "stderr": repr(exc)}
        _abort(env, info["stderr"])

    success = info["sandbox_rc"] == 0
    info = dict(info)
    info["task_completed"] = success
    reward = 1.0 if success else 0.0
    final_code = "# LLM-planned, deterministic gated R1 stage execution."
    logs = _build_log_lines(final_code, info, reward, success, False, 0, 0, len(code_blocks), prefix="R1 staged STEER execution.")
    code_path = _save_trial_artifacts(config, trial, info["sandbox_rc"], reward, success, final_code, "\n\n".join(code_blocks), responses, logs, [])
    print(f"R1 staged trial {trial} took {time.time() - started:.2f}s")
    return TrialSummary(trial, success, reward, success, False, info["sandbox_rc"], "\n".join(logs), success, code_path, 0, 0, len(code_blocks))
