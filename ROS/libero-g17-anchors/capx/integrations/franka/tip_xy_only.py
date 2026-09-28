"""Trustworthy tip XY-only motion for Cap-X LIBERO Franka.

Problem
-------
``robot_cartesian_pos`` (obs tip) → ``goto_pose`` is not an identity. Small
tip-space XY commands couple into Z and break contact/grasp. Success trajectories
never needed this primitive; near-miss closed-loop placement does.

Primitive
---------
``move_tip_xy`` does **not** call ``goto_pose``. It:

1. Reads the student tip from ``get_observation()["robot_cartesian_pos"]``.
2. FK ``panda_hand`` (IK link), commands a **world-frame XY** translation inside
   a hard joint-space trust region around the live arm state.
3. Closed-loop: recomputes the remaining absolute XY error after every executed
   micro-step.  An infeasible candidate is automatically retried at a smaller
   Cartesian step; the rejected candidate is never executed.
4. Anchors hand Z and quaternion to the primitive's initial pose so incidental
   controller drift is corrected on the next local solve, not compounded.

With orientation held fixed, tip = hand ⊕ const_world(TCP), so Δtip ≈ Δhand and
locking hand Z locks tip Z. Tip feedback corrects residual IK error.

Completion is defined in commanded XY.  Z is measured and reported, but is not
allowed to turn an achieved placement correction into a replay of the original
full command.  Holding/task validity remains the responsibility of the RR gate.

Legacy ``goto_pose`` is untouched. Enable student exposure with
``CAPX_LIBERO_TIP_XY_ONLY=1`` (optional); controllers may call the API method
directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

_DEFAULT_TOL_M = 0.002
_DEFAULT_MAX_ITERS = 12
_DEFAULT_MAX_STEP_M = 0.02
_ACCEPT_XY_M = 0.005
_PREFLIGHT_MAX_ARM_JOINT_DELTA_RAD = 0.20
_PREFLIGHT_MAX_POSITION_ERR_M = 0.004
_PREFLIGHT_MAX_ORIENTATION_ERR_RAD = 0.05
_EXEC_MAX_REVERSE_PROGRESS_M = 0.002
_EXEC_MAX_LATERAL_DRIFT_M = 0.012
_EXEC_MAX_OVERSHOOT_M = 0.015
_MIN_LOCAL_STEP_M = 0.005
_MAX_PREFLIGHT_BACKOFFS = 3
_MAX_JOINT_CONTROLLER_STEPS = 40


def tip_xy_only_student_exposed() -> bool:
    return os.environ.get("CAPX_LIBERO_TIP_XY_ONLY", "0") == "1"


def trusted_tip_xy_contract() -> dict[str, float | int | str]:
    """Machine-checkable execution contract for the transactional primitive."""
    return {
        "schema": "capx_completion_tip_xy_contract_v2",
        "max_arm_joint_delta_rad": _PREFLIGHT_MAX_ARM_JOINT_DELTA_RAD,
        "max_candidate_fk_position_err_m": _PREFLIGHT_MAX_POSITION_ERR_M,
        "max_candidate_fk_orientation_err_rad": _PREFLIGHT_MAX_ORIENTATION_ERR_RAD,
        "max_reverse_progress_m": _EXEC_MAX_REVERSE_PROGRESS_M,
        "max_lateral_drift_m": _EXEC_MAX_LATERAL_DRIFT_M,
        "max_overshoot_m": _EXEC_MAX_OVERSHOOT_M,
        "min_local_step_m": _MIN_LOCAL_STEP_M,
        "max_preflight_backoffs": _MAX_PREFLIGHT_BACKOFFS,
        "max_joint_controller_steps": _MAX_JOINT_CONTROLLER_STEPS,
    }


@dataclass(frozen=True)
class TipXyMoveResult:
    ok: bool
    commanded_xy: tuple[float, float]
    achieved_xy: tuple[float, float]
    tip_z0: float
    tip_z1: float
    d_tip_z: float
    iters: int
    final_xy_err: float
    message: str
    rejection_stage: str | None = None
    preflight_rejections: int = 0
    executed_steps: int = 0
    max_arm_joint_delta: float = 0.0
    max_executed_arm_joint_delta: float = 0.0
    status: str = "legacy"
    predicate_reached: bool = False

    @property
    def acceptance_ok(self) -> bool:
        dx_c, dy_c = self.commanded_xy
        dx_a, dy_a = self.achieved_xy
        return bool(
            self.ok
            and abs(dx_a - dx_c) < _ACCEPT_XY_M
            and abs(dy_a - dy_c) < _ACCEPT_XY_M
        )

    @property
    def measured_remaining_xy(self) -> tuple[float, float]:
        """Command remainder after consuming robot-measured tip displacement."""
        dx_c, dy_c = self.commanded_xy
        dx_a, dy_a = self.achieved_xy
        return (float(dx_c - dx_a), float(dy_c - dy_a))

    @property
    def measured_forward_progress_m(self) -> float:
        """Signed progress along the command, using proprioceptive tip motion only."""
        command = np.asarray(self.commanded_xy, dtype=np.float64)
        achieved = np.asarray(self.achieved_xy, dtype=np.float64)
        norm = float(np.linalg.norm(command))
        if norm <= 1e-12:
            return 0.0
        return float(np.dot(achieved, command / norm))

    @property
    def usable_partial_progress(self) -> bool:
        """Whether a locally rejected command still produced safe forward motion.

        ``move_tip_xy`` validates every committed micro-step before execution.
        Therefore a later local IK rejection must not erase earlier measured
        progress.  Execution-contract violations remain fatal; only a local
        preflight exhaustion after at least one committed forward step is
        exposed as a usable intermediate state.

        This consumes robot proprioception/FK only.  It does not inspect scene
        geometry, object pose, or any continuous privileged world value.
        """
        return bool(
            not self.ok
            and self.status == "partial"
            and self.rejection_stage == "local_unreachable"
            and self.executed_steps > 0
            and self.measured_forward_progress_m > 1e-6
        )


def read_obs_tip(api: Any) -> tuple[np.ndarray, np.ndarray]:
    obs = api.get_observation()
    cart = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64).reshape(-1)
    return cart[:3].copy(), cart[3:7].copy()


def _live_actuated_cfg(api: Any) -> np.ndarray:
    """Read the arm seed from the simulator, never from commanded ``api.cfg``.

    PyRoKi's Panda model has one extra actuated gripper coordinate.  LIBERO's
    observation provides the seven physical arm joints plus a normalized
    gripper value; preserve the solver's cached extra coordinate when present,
    but always replace the seven arm coordinates with live measurements.
    """
    n = int(api._robot.joints.num_actuated_joints)
    obs = api.get_observation()
    joints = np.asarray(obs["robot_joint_pos"], dtype=np.float64).reshape(-1)
    if joints.size < min(7, n):
        raise RuntimeError(f"live joint state too short: {joints.size} < {min(7, n)}")
    cfg = np.zeros(n, dtype=np.float64)
    arm_n = min(7, n)
    cfg[:arm_n] = joints[:arm_n]
    cached = np.asarray(getattr(api, "cfg", []), dtype=np.float64).reshape(-1)
    if n > arm_n and cached.size >= n and np.all(np.isfinite(cached[arm_n:n])):
        cfg[arm_n:n] = cached[arm_n:n]
    elif n > arm_n and joints.size >= n:
        cfg[arm_n:n] = joints[arm_n:n]
    return cfg


def fk_panda_hand(api: Any, cfg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    import jax.numpy as jnp
    import jaxlie

    if cfg is None:
        cfg = _live_actuated_cfg(api)
    robot = api._robot
    link_i = robot.links.names.index(api._target_link_name)
    Ts = robot.forward_kinematics(jnp.asarray(cfg, dtype=jnp.float32))
    se3 = jaxlie.SE3(Ts[link_i])
    pos = np.asarray(se3.translation(), dtype=np.float64).reshape(3)
    quat = np.asarray(se3.rotation().wxyz, dtype=np.float64).reshape(4)
    return pos, quat


def _quaternion_angle_rad(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64).reshape(4)
    qb = np.asarray(b, dtype=np.float64).reshape(4)
    qa /= max(float(np.linalg.norm(qa)), 1e-12)
    qb /= max(float(np.linalg.norm(qb)), 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)))


@dataclass(frozen=True)
class _PreflightCandidate:
    ok: bool
    cfg_live: np.ndarray
    cfg_candidate: np.ndarray | None
    target_position: np.ndarray
    target_wxyz: np.ndarray
    max_arm_joint_delta: float
    position_err_m: float
    orientation_err_rad: float
    reason: str


def _preflight_hand_xy_step(
    api: Any,
    *,
    delta_xy: np.ndarray,
    anchor_hand_z: float,
    anchor_wxyz: np.ndarray,
) -> _PreflightCandidate:
    """Solve and validate one local IK candidate without moving the robot."""
    cfg = _live_actuated_cfg(api)
    pos, _quat = fk_panda_hand(api, cfg)
    target = pos.copy()
    target[0] += float(delta_xy[0])
    target[1] += float(delta_xy[1])
    # Correct back toward the primitive-wide Z/orientation anchor.  Using the
    # current pose here would silently ratchet controller drift at every step.
    target[2] = float(anchor_hand_z)
    target_quat = np.asarray(anchor_wxyz, dtype=np.float64).reshape(4)
    try:
        local_solver = getattr(api._pks, "solve_ik_local", None)
        if local_solver is not None:
            raw_candidate = local_solver(
                robot=api._robot,
                target_link_name=api._target_link_name,
                target_position=target,
                target_wxyz=target_quat,
                prev_cfg=cfg,
                max_arm_delta=_PREFLIGHT_MAX_ARM_JOINT_DELTA_RAD,
            )
        else:
            # Test/back-compat path.  Production v276 requires solve_ik_local
            # in its source manifest and preflight, so this is never the live
            # branch used by the experiment.
            raw_candidate = api._pks.solve_ik_vel_cost(
                robot=api._robot,
                target_link_name=api._target_link_name,
                target_position=target,
                target_wxyz=target_quat,
                prev_cfg=cfg,
                initial_cfg=cfg,
            )
        candidate = np.asarray(raw_candidate, dtype=np.float64).reshape(-1)
    except Exception as exc:
        return _PreflightCandidate(
            False, cfg, None, target, target_quat, float("inf"), float("inf"),
            float("inf"), f"ik_exception:{exc!r}"
        )

    if candidate.size != cfg.size or not np.all(np.isfinite(candidate)):
        return _PreflightCandidate(
            False, cfg, None, target, target_quat, float("inf"), float("inf"),
            float("inf"), "candidate_nonfinite_or_wrong_shape"
        )
    arm_n = min(7, cfg.size)
    max_dq = float(np.max(np.abs(candidate[:arm_n] - cfg[:arm_n])))
    try:
        predicted_pos, predicted_quat = fk_panda_hand(api, candidate)
        pos_err = float(np.linalg.norm(predicted_pos - target))
        ori_err = _quaternion_angle_rad(predicted_quat, target_quat)
    except Exception as exc:
        return _PreflightCandidate(
            False, cfg, candidate, target, target_quat, max_dq, float("inf"),
            float("inf"), f"candidate_fk_exception:{exc!r}"
        )

    reasons: list[str] = []
    lower = np.asarray(
        getattr(api._robot.joints, "lower_limits", []), dtype=np.float64
    ).reshape(-1)
    upper = np.asarray(
        getattr(api._robot.joints, "upper_limits", []), dtype=np.float64
    ).reshape(-1)
    if lower.size == candidate.size and upper.size == candidate.size:
        if np.any(candidate < lower - 1e-8) or np.any(candidate > upper + 1e-8):
            reasons.append("joint_limit_violation")
    if max_dq > _PREFLIGHT_MAX_ARM_JOINT_DELTA_RAD:
        reasons.append(f"joint_jump:{max_dq:.4f}")
    if pos_err > _PREFLIGHT_MAX_POSITION_ERR_M:
        reasons.append(f"fk_position_err:{pos_err:.4f}")
    if ori_err > _PREFLIGHT_MAX_ORIENTATION_ERR_RAD:
        reasons.append(f"fk_orientation_err:{ori_err:.4f}")
    return _PreflightCandidate(
        not reasons,
        cfg,
        candidate,
        target,
        target_quat,
        max_dq,
        pos_err,
        ori_err,
        "ok" if not reasons else ",".join(reasons),
    )


def _execute_preflighted_step(
    api: Any,
    candidate: _PreflightCandidate,
    *,
    tip_before: np.ndarray,
    commanded_step_xy: np.ndarray,
) -> tuple[bool, str, np.ndarray, float]:
    """Execute a local candidate and validate directional Cartesian progress.

    Under-travel is partial progress, not a failure: the outer loop recomputes
    the absolute remaining XY goal.  Z and joint tracking remain diagnostics;
    they cannot cause the original full relative command to be replayed.
    """
    assert candidate.ok and candidate.cfg_candidate is not None
    arm_n = min(7, candidate.cfg_live.size)
    joints = candidate.cfg_candidate[:arm_n]
    api._env.move_to_joints_blocking(
        joints,
        max_steps=_MAX_JOINT_CONTROLLER_STEPS,
    )
    tip_after, _ = read_obs_tip(api)
    cfg_after = _live_actuated_cfg(api)
    # Synchronize the solver cache to what the simulator actually reached.
    api.cfg = cfg_after.copy()

    achieved_step = tip_after[:2] - tip_before[:2]
    command_norm = float(np.linalg.norm(commanded_step_xy))
    direction = commanded_step_xy / max(command_norm, 1e-12)
    progress = float(np.dot(achieved_step, direction))
    lateral = float(np.linalg.norm(achieved_step - progress * direction))
    actual_max_dq = float(
        np.max(np.abs(cfg_after[:arm_n] - candidate.cfg_live[:arm_n]))
    )
    reasons: list[str] = []
    if progress < -_EXEC_MAX_REVERSE_PROGRESS_M:
        reasons.append(f"reverse_progress:{progress:.4f}")
    if lateral > _EXEC_MAX_LATERAL_DRIFT_M:
        reasons.append(f"lateral_drift:{lateral:.4f}")
    if progress > command_norm + _EXEC_MAX_OVERSHOOT_M:
        reasons.append(f"overshoot:{progress - command_norm:.4f}")
    detail = (
        f"progress:{progress:.4f},lateral:{lateral:.4f},"
        f"d_z:{float(tip_after[2] - tip_before[2]):+.4f},"
        f"actual_dq:{actual_max_dq:.4f}"
    )
    return (
        not reasons,
        detail if not reasons else ",".join(reasons + [detail]),
        tip_after,
        actual_max_dq,
    )


def move_tip_xy(
    api: Any,
    dx: float,
    dy: float,
    *,
    tol_m: float = _DEFAULT_TOL_M,
    max_iters: int = _DEFAULT_MAX_ITERS,
    max_step_m: float = _DEFAULT_MAX_STEP_M,
    log: bool = True,
    stop_predicate: Callable[[], bool] | None = None,
) -> TipXyMoveResult:
    """Transactional closed-loop world-frame tip XY move.

    The requested relative command is converted once to an absolute XY goal.
    Every solve is hard-constrained around live joints; infeasible candidates
    are shrunk before execution.  After any partial progress, only the measured
    remainder is commanded, so a completed command can never be replayed merely
    because Z or joint tracking was imperfect.
    """
    tip0, _ = read_obs_tip(api)
    goal_xy = np.array(
        [float(tip0[0]) + float(dx), float(tip0[1]) + float(dy)], dtype=np.float64
    )
    z0 = float(tip0[2])
    cfg0 = _live_actuated_cfg(api)
    hand0, hand_quat0 = fk_panda_hand(api, cfg0)
    iters = 0
    final_err = float("inf")
    rejection_stage: str | None = None
    rejection_reason = ""
    preflight_rejections = 0
    executed_steps = 0
    max_arm_joint_delta = 0.0
    max_executed_arm_joint_delta = 0.0
    predicate_reached = False
    command_norm = float(np.linalg.norm(np.array([dx, dy], dtype=np.float64)))
    # A caller-provided max_iters remains a floor.  Reserve enough attempts for
    # a command that must shrink all the way to the minimum local step.
    iteration_budget = max(
        int(max_iters),
        int(np.ceil(command_norm / max(_MIN_LOCAL_STEP_M, 1e-12))) + 4,
    )

    for iters in range(1, iteration_budget + 1):
        tip, _ = read_obs_tip(api)
        err = goal_xy - tip[:2]
        final_err = float(np.linalg.norm(err))
        if final_err < float(tol_m):
            break
        # Cap step to keep IK local / contact-safe.
        step = err.copy()
        n = float(np.linalg.norm(step))
        if n > float(max_step_m):
            step *= float(max_step_m) / n
        # Back off locally before giving up.  Rejected candidates are never
        # committed, so this search cannot perturb the scene.
        candidate: _PreflightCandidate | None = None
        proposed = step.copy()
        for backoff in range(_MAX_PREFLIGHT_BACKOFFS + 1):
            candidate = _preflight_hand_xy_step(
                api,
                delta_xy=proposed,
                anchor_hand_z=float(hand0[2]),
                anchor_wxyz=hand_quat0,
            )
            max_arm_joint_delta = max(
                max_arm_joint_delta, float(candidate.max_arm_joint_delta)
            )
            if candidate.ok:
                break
            preflight_rejections += 1
            if float(np.linalg.norm(proposed)) <= _MIN_LOCAL_STEP_M + 1e-12:
                break
            proposed *= 0.5
            if float(np.linalg.norm(proposed)) < _MIN_LOCAL_STEP_M:
                proposed *= _MIN_LOCAL_STEP_M / max(
                    float(np.linalg.norm(proposed)), 1e-12
                )
            if log:
                print(
                    f"[tip-xy-only] preflight reject backoff={backoff} "
                    f"reason={candidate.reason} next_step={proposed.tolist()}",
                    flush=True,
                )
        assert candidate is not None
        if not candidate.ok:
            rejection_stage = "local_unreachable"
            rejection_reason = candidate.reason
            break

        executed_ok, executed_reason, tip_a, actual_max_dq = _execute_preflighted_step(
            api,
            candidate,
            tip_before=tip,
            commanded_step_xy=proposed,
        )
        executed_steps += 1
        max_executed_arm_joint_delta = max(
            max_executed_arm_joint_delta, actual_max_dq
        )
        if not executed_ok:
            rejection_stage = "execution"
            rejection_reason = executed_reason
            break
        if log:
            print(
                f"[tip-xy-only] iter={iters} err={final_err:.4f}m "
                f"cmd_dxy=({proposed[0]:+.4f},{proposed[1]:+.4f}) "
                f"tip_z={z0:.4f}->{float(tip_a[2]):.4f} "
                f"d_tip_z={float(tip_a[2]) - z0:+.5f}",
                flush=True,
            )
        if stop_predicate is not None:
            try:
                predicate_reached = bool(stop_predicate())
            except Exception as exc:
                if log:
                    print(
                        f"[tip-xy-only] stop predicate error ({exc!r}); continue",
                        flush=True,
                    )
                predicate_reached = False
            if predicate_reached:
                break

    tip1, _ = read_obs_tip(api)
    achieved = (float(tip1[0] - tip0[0]), float(tip1[1] - tip0[1]))
    d_z = float(tip1[2] - tip0[2])
    final_err = float(np.linalg.norm(goal_xy - tip1[:2]))
    if (
        not predicate_reached
        and rejection_stage is None
        and final_err >= float(tol_m)
    ):
        rejection_stage = "max_iters"
        rejection_reason = f"remaining_xy:{final_err:.4f}"
    ok = bool(
        predicate_reached
        or (rejection_stage is None and final_err < float(tol_m))
    )
    status = "predicate_reached" if predicate_reached else (
        "completed" if ok else (
            "partial" if executed_steps else "local_unreachable"
        )
    )
    msg = (
        f"cmd=({dx:+.4f},{dy:+.4f}) ach={achieved} "
        f"d_tip_z={d_z:+.5f} err={final_err:.4f} iters={iters} "
        f"reject={rejection_stage}:{rejection_reason} "
        f"preflight_rejections={preflight_rejections} executed={executed_steps} "
        f"status={status} budget={iteration_budget}"
    )
    result = TipXyMoveResult(
        ok=ok,
        commanded_xy=(float(dx), float(dy)),
        achieved_xy=achieved,
        tip_z0=z0,
        tip_z1=float(tip1[2]),
        d_tip_z=d_z,
        iters=int(iters),
        final_xy_err=final_err,
        message=msg,
        rejection_stage=rejection_stage,
        preflight_rejections=preflight_rejections,
        executed_steps=executed_steps,
        max_arm_joint_delta=max_arm_joint_delta,
        max_executed_arm_joint_delta=max_executed_arm_joint_delta,
        status=status,
        predicate_reached=predicate_reached,
    )
    if log:
        print(
            f"[tip-xy-only] done ok={result.ok} accept={result.acceptance_ok} {msg}",
            flush=True,
        )
    return result


def probe_tip_xy_primitive(
    api: Any,
    step_m: float = 0.03,
    *,
    restore: bool = True,
) -> dict[str, Any]:
    """+X then optional restore −X; compare primitive vs legacy goto on |d_tip_z|."""
    tip_before, quat_before = read_obs_tip(api)

    prim = move_tip_xy(api, float(step_m), 0.0, log=True)
    out: dict[str, Any] = {
        "primitive": {
            "acceptance_ok": prim.acceptance_ok,
            "ok": prim.ok,
            "achieved_xy": prim.achieved_xy,
            "d_tip_z": prim.d_tip_z,
            "iters": prim.iters,
            "message": prim.message,
        }
    }

    if restore:
        back = move_tip_xy(api, -float(step_m), 0.0, log=True)
        out["primitive_restore"] = {
            "acceptance_ok": back.acceptance_ok,
            "d_tip_z": back.d_tip_z,
            "achieved_xy": back.achieved_xy,
            "message": back.message,
        }

    # Legacy A/B from the same start pose (re-seed tip if restore ran).
    tip_now, quat_now = read_obs_tip(api)
    legacy_target = tip_now.copy()
    legacy_target[0] += float(step_m)
    api.goto_pose(legacy_target, quat_now, z_approach=0.0)
    tip_leg, _ = read_obs_tip(api)
    out["legacy_goto"] = {
        "achieved_xy": (
            float(tip_leg[0] - tip_now[0]),
            float(tip_leg[1] - tip_now[1]),
        ),
        "d_tip_z": float(tip_leg[2] - tip_now[2]),
    }
    # Best-effort restore via primitive (not goto).
    if restore:
        move_tip_xy(
            api,
            float(tip_before[0] - read_obs_tip(api)[0][0]),
            float(tip_before[1] - read_obs_tip(api)[0][1]),
            log=False,
        )
        out["start_tip"] = tip_before.tolist()
        out["end_tip"] = read_obs_tip(api)[0].tolist()
    return out
