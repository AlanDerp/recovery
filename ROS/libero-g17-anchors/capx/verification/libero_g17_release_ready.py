"""Minimal ReleaseReady: full-conjunct gate + XY nudge (no LLM).

Holding On payload: ``open_gripper`` only when placement ready.

Modes
-----
- **tipxy** (``CAPX_LIBERO_G17_RELEASE_READY_TIPXY=1``, v251/v254):
  At ``open_gripper``, **before** release judgment completes: if holding and
  out of public On XY band, slide tip XY toward c2c=0 with band-sized steps
  (``LIBERO_OBJECT_ONTOP_XY_M`` = 0.03 m), not mm micro-steps.
  Default residual is GT ``private_poses`` c2c. With
  ``CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL=1`` (v255), residual /
  tipxy ``xy_within`` come from student ``get_object_pose`` only.
- **goto** (``CAPX_LIBERO_G17_RELEASE_READY_GOTO=1``, v248): legacy tip→goto.
- **hand** (default RELEASE_READY without above, v247): panda_hand FK/IK.

Near-miss osc (``NEARMISS_OSC=1`` without tipxy mode) dispatches
``libero_g17_nearmiss_osc`` instead.

Silent-allow close (v266): soft-held promotes into the release path instead
of silent empty-hand open. Plate contact is **not** required for tipxy
*open after within* (v265 hold-guard rolled back).

Near-miss gate (``RELEASE_READY_NEARMISS_GATE=1``):
  Release = hold open → always hits RR.
  Near-miss = xy_distance ≤ outer (default **0.15 m**; XY-only, no contact/Z).
  v268: drop v267 plate-contact conjunct (single-variable).
  * near-miss ∧ ¬within → tipxy / wash only
  * holding ∧ ¬near-miss ∧ ¬within → **refuse open** (student goto)
  * within ∧ hold → allow open (Z/contact not required)
  * empty → allow open
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from capx.verification.libero_g17_public_iff import (
    LIBERO_OBJECT_ONTOP_XY_M,
    derive_libero_object_ontop_basis,
)

# Legacy hand/goto micro-step law (v246-style). Tipxy must NOT use these.
_DEFAULT_STEP_M = 0.005
_MIN_STEP_M = 0.002
_MAX_STEP_M = 0.010
_MAX_ITERS = 25
_GAIN = 0.7
_DIST_ABORT_M = 0.20
_DIST_JUMP_ABORT_M = 0.05
_TIPXY_Z_ABORT_M = 0.01
# Contract: tipxy step == public On XY band (3 cm). Tol must be ≪ step.
_TIPXY_STEP_M = float(LIBERO_OBJECT_ONTOP_XY_M)
_TIPXY_TOL_M = 0.0015
_TIPXY_MAX_ITERS = 8


def release_ready_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_RELEASE_READY", "0") == "1"


def release_ready_goto_mode() -> bool:
    """Student tip↔goto_pose convention (v248)."""
    return os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_GOTO", "0") == "1"


def release_ready_tipxy_mode() -> bool:
    """v246 gate + tip XY-only primitive (v251)."""
    return os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_TIPXY", "0") == "1"


def release_ready_tipxy_fair_residual() -> bool:
    """Ablation: tipxy residual/within from student get_object_pose (v255)."""
    return (
        os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL", "0")
        == "1"
    )


def release_ready_no_silent_allow() -> bool:
    """v266: soft-held without firm acquire must not silent-open."""
    return os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_NO_SILENT_ALLOW", "0") == "1"


def release_ready_nearmiss_gate_enabled() -> bool:
    """v267: tipxy/wash only in placement near-miss; else refuse hold-open."""
    return os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_NEARMISS_GATE", "0") == "1"


def nearmiss_outer_m() -> float:
    """Outer XY envelope for near-miss (default = wash proof start 0.15 m)."""
    raw = os.environ.get("CAPX_LIBERO_G17_NEARMISS_OUTER_M", "").strip()
    if raw:
        return float(raw)
    return 0.15


def is_placement_near_miss(gate: ReleaseReadyGate) -> bool:
    """XY near-miss: c2c within outer envelope (default 0.15 m).

    Single-variable vs v267: **no plate-contact / Z requirement** — contact
    must not block tipxy entry (v266 success path is often contact=False).
    Far XY (dist > outer) while holding → not near-miss → refuse open.
    """
    if gate.xy_distance_m is None:
        return False
    return float(gate.xy_distance_m) <= float(nearmiss_outer_m())



@dataclass(frozen=True)
class ReleaseReadyGate:
    relation: str | None
    acquired: bool
    contact: bool
    surface_on: bool
    xy_within: bool
    xy_distance_m: float | None
    residual_xy: tuple[float, float] | None

    @property
    def release_ok(self) -> bool:
        return bool(
            self.acquired and self.contact and self.surface_on and self.xy_within
        )

    @property
    def release_ok_tipxy(self) -> bool:
        """Public On XY band met while holding (plate contact not required)."""
        return bool(self.acquired and self.xy_within)

    @property
    def needs_align(self) -> bool:
        return bool(self.acquired and not self.release_ok)

    @property
    def needs_align_tipxy(self) -> bool:
        """Holding but not tipxy-release-ok → must align (or refuse) before open."""
        return bool(self.acquired and not self.release_ok_tipxy)

    @property
    def active(self) -> bool:
        return self.needs_align and (not self.xy_within)


def _payload_acquired(snapshot: Any) -> bool:
    holding = snapshot.values.get("executor_object.exe01.obj01.holding") == "TRUE"
    payload = snapshot.values.get("executor.exe01.payload")
    grasp_ev = snapshot.values.get("executor_object.exe01.obj01.grasp_evidence")
    contact = snapshot.values.get("executor_object.exe01.obj01.contact") == "TRUE"
    return holding and payload == "OBJ01" and grasp_ev != "NOT_HELD" and contact


def _payload_soft_held(snapshot: Any) -> bool:
    """Holding payload without requiring gripper contact (silent-allow close)."""
    holding = snapshot.values.get("executor_object.exe01.obj01.holding") == "TRUE"
    payload = snapshot.values.get("executor.exe01.payload")
    grasp_ev = snapshot.values.get("executor_object.exe01.obj01.grasp_evidence")
    return holding and payload == "OBJ01" and grasp_ev != "NOT_HELD"


def evaluate_gate(binding: Any, snapshot: Any) -> ReleaseReadyGate:
    relation = getattr(binding, "relation", None)
    if relation != "on":
        return ReleaseReadyGate(
            relation=relation,
            acquired=False,
            contact=False,
            surface_on=False,
            xy_within=False,
            xy_distance_m=None,
            residual_xy=None,
        )
    acquired = _payload_acquired(snapshot)
    basis = derive_libero_object_ontop_basis(snapshot)
    footprint = snapshot.values.get(
        "object_object.obj01.obj02.center_in_reference_footprint"
    )
    surface_on = footprint == "INSIDE"
    xy_within = basis.xy_band == "WITHIN_ONTOP_XY_BAND"
    payload = np.asarray(snapshot.private_poses["obj01"][0], dtype=np.float64)
    receptacle = np.asarray(snapshot.private_poses["obj02"][0], dtype=np.float64)
    residual = payload[:2] - receptacle[:2]
    return ReleaseReadyGate(
        relation=relation,
        acquired=acquired,
        contact=bool(basis.contact),
        surface_on=surface_on,
        xy_within=xy_within,
        xy_distance_m=float(basis.xy_distance_m),
        residual_xy=(float(residual[0]), float(residual[1])),
    )


def promote_soft_held(
    gate: ReleaseReadyGate, snapshot: Any
) -> tuple[ReleaseReadyGate, bool]:
    """If firm-acquire false but soft-held, promote acquired for tipxy gating.

    No-op unless ``CAPX_LIBERO_G17_RELEASE_READY_NO_SILENT_ALLOW=1`` (v266),
    so v264 remains bit-reproducible. Returns (gate, promoted).
    """
    if not release_ready_no_silent_allow():
        return gate, False
    if gate.acquired:
        return gate, False
    if not _payload_soft_held(snapshot):
        return gate, False
    return replace(gate, acquired=True), True


def student_c2c_residual_xy(
    api: Any,
) -> tuple[np.ndarray | None, float | None, str]:
    """Student c2c residual in GT sign: payload_xy − plate_xy.

    Uses the same ``get_object_pose`` path as fair-place-servo (Molmo/SAM).
    ``fair_xy_residual`` returns plate−payload; we negate for tipxy step law.
    """
    from capx.verification.libero_g17_fair_place_servo import fair_xy_residual

    fair_r, src = fair_xy_residual(api)
    if fair_r is None:
        return None, None, src
    residual = -np.asarray(fair_r, dtype=np.float64).reshape(2)
    dist = float(np.linalg.norm(residual))
    return residual, dist, src


def overlay_gate_student_residual(
    gate: ReleaseReadyGate,
    api: Any,
    *,
    band_m: float = LIBERO_OBJECT_ONTOP_XY_M,
) -> tuple[ReleaseReadyGate | None, str]:
    """Replace tipxy residual/within with student poses; keep acquired/contact."""
    residual, dist, src = student_c2c_residual_xy(api)
    if residual is None or dist is None:
        return None, src
    within = bool(dist < float(band_m))
    return (
        replace(
            gate,
            residual_xy=(float(residual[0]), float(residual[1])),
            xy_distance_m=float(dist),
            xy_within=within,
        ),
        src,
    )


def next_xy_step(
    residual_xy: np.ndarray,
    *,
    prev_distance_m: float | None,
    step_m: float,
    band_m: float = LIBERO_OBJECT_ONTOP_XY_M,
) -> tuple[np.ndarray, float, float]:
    """Legacy adaptive micro-step (hand/goto). Do not use for tipxy."""
    residual = np.asarray(residual_xy, dtype=np.float64).reshape(2)
    distance = float(np.linalg.norm(residual))
    if distance < band_m or distance < 1e-9:
        return np.zeros(2, dtype=np.float64), step_m, distance
    step = float(step_m)
    if prev_distance_m is not None:
        if distance < prev_distance_m:
            step = min(_MAX_STEP_M, step * 1.1)
        else:
            step = max(_MIN_STEP_M, step * 0.5)
    unit = -residual / distance
    move_mag = min(step, distance * _GAIN)
    return unit * move_mag, step, distance


def next_xy_step_band(
    residual_xy: np.ndarray,
    *,
    step_m: float = _TIPXY_STEP_M,
    band_m: float = LIBERO_OBJECT_ONTOP_XY_M,
) -> tuple[np.ndarray, float, float]:
    """Tipxy contract: one band-sized (3 cm) step toward c2c=0, or residual if smaller."""
    residual = np.asarray(residual_xy, dtype=np.float64).reshape(2)
    distance = float(np.linalg.norm(residual))
    if distance < band_m or distance < 1e-9:
        return np.zeros(2, dtype=np.float64), float(step_m), distance
    unit = -residual / distance
    move_mag = min(float(step_m), distance)
    return unit * move_mag, float(step_m), distance


# ----- v248: student tip ↔ goto_pose -----


def _tip_pose(api: Any) -> tuple[np.ndarray, np.ndarray]:
    obs = api.get_observation()
    cart = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64).reshape(-1)
    return cart[:3].copy(), cart[3:7].copy()


def _nudge_xy_goto(api: Any, delta_xy: np.ndarray) -> str:
    tip, quat = _tip_pose(api)
    target = tip.copy()
    target[0] += float(delta_xy[0])
    target[1] += float(delta_xy[1])
    api.goto_pose(target, quat, z_approach=0.0)
    tip2, _ = _tip_pose(api)
    return (
        f"tip_z={float(tip[2]):.4f}->{float(tip2[2]):.4f} "
        f"d_tip_z={float(tip2[2] - tip[2]):+.5f}"
    )


def _nudge_xy_tipxy(api: Any, delta_xy: np.ndarray) -> str:
    """Band-step tip XY-only; tol ≪ 3 cm so commands are never no-op'd away."""
    from capx.integrations.franka.tip_xy_only import move_tip_xy

    res = move_tip_xy(
        api,
        float(delta_xy[0]),
        float(delta_xy[1]),
        tol_m=_TIPXY_TOL_M,
        max_step_m=_TIPXY_STEP_M,
        log=True,
    )
    return (
        f"tip_z={res.tip_z0:.4f}->{res.tip_z1:.4f} "
        f"d_tip_z={res.d_tip_z:+.5f} accept={res.acceptance_ok} "
        f"ach={res.achieved_xy}"
    )


# ----- v247: panda_hand FK / IK -----


def _actuated_cfg(api: Any) -> np.ndarray:
    n = int(api._robot.joints.num_actuated_joints)
    if getattr(api, "cfg", None) is not None:
        cfg = np.asarray(api.cfg, dtype=np.float64).reshape(-1)
        if cfg.size >= n:
            return cfg[:n].copy()
    obs = api.get_observation()
    joints = np.asarray(obs["robot_joint_pos"], dtype=np.float64).reshape(-1)
    cfg = np.zeros(n, dtype=np.float64)
    cfg[: min(7, n, joints.size)] = joints[: min(7, n, joints.size)]
    return cfg


def _fk_panda_hand(api: Any, cfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import jax.numpy as jnp
    import jaxlie

    robot = api._robot
    link_i = robot.links.names.index(api._target_link_name)
    Ts = robot.forward_kinematics(jnp.asarray(cfg, dtype=jnp.float32))
    se3 = jaxlie.SE3(Ts[link_i])
    pos = np.asarray(se3.translation(), dtype=np.float64).reshape(3)
    quat = np.asarray(se3.rotation().wxyz, dtype=np.float64).reshape(4)
    return pos, quat


def _nudge_hand_xy_lock_z(api: Any, delta_xy: np.ndarray) -> str:
    cfg = _actuated_cfg(api)
    pos, quat = _fk_panda_hand(api, cfg)
    z0 = float(pos[2])
    target = pos.copy()
    target[0] += float(delta_xy[0])
    target[1] += float(delta_xy[1])
    target[2] = z0
    api.cfg = api._pks.solve_ik_vel_cost(
        robot=api._robot,
        target_link_name=api._target_link_name,
        target_position=target,
        target_wxyz=quat,
        prev_cfg=cfg,
    )
    joints = np.asarray(api.cfg, dtype=np.float64).reshape(-1)[:7]
    api._env.move_to_joints_blocking(joints)
    pos2, _ = _fk_panda_hand(api, _actuated_cfg(api))
    z1 = float(pos2[2])
    return f"z0={z0:.4f} z1={z1:.4f} dz={z1 - z0:+.5f}"


def _read_live(api: Any) -> tuple[Any, Any, ReleaseReadyGate]:
    from capx.verification import libero_g17_adapter as adapter

    env = api._env
    binding = adapter.bind_libero_g17_env(env)
    binding, snapshot, _attachment = adapter.extract_libero_snapshot(env, binding)
    return binding, snapshot, evaluate_gate(binding, snapshot)


def _apply_tipxy_residual_source(
    api: Any, gate: ReleaseReadyGate
) -> tuple[ReleaseReadyGate | None, str]:
    """Tipxy residual source: GT (default) or student get_object_pose (ablation)."""
    if not (release_ready_tipxy_mode() and release_ready_tipxy_fair_residual()):
        return gate, "private_poses"
    overlaid, src = overlay_gate_student_residual(gate, api)
    if overlaid is None:
        return None, src
    return overlaid, f"get_object_pose:{src}"


def maybe_xy_align_before_release(api: Any) -> bool:
    """True ⇒ allow open_gripper; False ⇒ refuse."""
    if not release_ready_enabled():
        return True
    if getattr(api, "_capx_release_ready_busy", False):
        print("[release-ready] busy re-entry; open ok (nested)", flush=True)
        return True

    # Independent near-miss start/step oscillator (v250); tipxy mode wins if both set.
    from capx.verification import libero_g17_nearmiss_osc as osc

    if osc.nearmiss_osc_enabled() and not release_ready_tipxy_mode():
        return osc.run_nearmiss_osc(api)

    try:
        _binding, _snapshot, gate = _read_live(api)
    except Exception as exc:
        print(
            f"[release-ready] snapshot failed ({exc!r}); refuse open (fail closed)",
            flush=True,
        )
        return False

    if gate.relation != "on":
        print(
            f"[release-ready] relation={gate.relation!r}; open ok (not On gate)",
            flush=True,
        )
        return True
    # Close silent-allow: soft-held without firm contact must enter tipxy/refuse.
    if not gate.acquired:
        gate, promoted = promote_soft_held(gate, _snapshot)
        if promoted:
            print(
                "[release-ready] soft-held without firm acquire; "
                "promote for tipxy (close silent-allow)",
                flush=True,
            )
        else:
            print("[release-ready] not acquired; open ok (empty)", flush=True)
            return True

    # v267 near-miss gate: only tipxy when in placement neighborhood.
    # Holding + not near-miss + not already within → refuse (student goto).
    if (
        release_ready_nearmiss_gate_enabled()
        and not gate.xy_within
        and not is_placement_near_miss(gate)
    ):
        print(
            "[release-ready] holding but not near-miss "
            f"(contact={gate.contact} dist={gate.xy_distance_m} "
            f"outer={nearmiss_outer_m():.4f}m); refuse open "
            "(student goto/approach)",
            flush=True,
        )
        return False

    tipxy = release_ready_tipxy_mode()
    residual_src = "private_poses"
    if tipxy:
        gate, residual_src = _apply_tipxy_residual_source(api, gate)
        if gate is None:
            print(
                f"[release-ready] tipxy fair residual failed ({residual_src}); "
                "refuse open",
                flush=True,
            )
            return False
        if gate.release_ok_tipxy:
            print(
                f"[release-ready] tipxy release_ok dist={gate.xy_distance_m:.4f}m "
                f"contact={gate.contact} residual_src={residual_src}; open ok",
                flush=True,
            )
            return True
        # Window fix: holding + placement_xy out-of-band ⇒ always enter tip_xy
        # with residual *before* refusing. Do NOT require plate contact to slide.
        if not gate.needs_align_tipxy:
            print(
                f"[release-ready] tipxy no-align-needed "
                f"contact={gate.contact} within={gate.xy_within} "
                f"residual_src={residual_src}; open",
                flush=True,
            )
            return True
    elif gate.release_ok:
        print(
            f"[release-ready] release_ok dist={gate.xy_distance_m:.4f}m "
            f"contact={gate.contact} surface={gate.surface_on}; open ok",
            flush=True,
        )
        return True

    if gate.residual_xy is None:
        print("[release-ready] holding but no residual; refuse open", flush=True)
        return False

    if tipxy:
        mode = "tipxy"
    elif release_ready_goto_mode():
        mode = "goto"
    else:
        mode = "hand"
    print(
        f"[release-ready] holding block open until full ready "
        f"within={gate.xy_within} dist={gate.xy_distance_m} "
        f"residual={gate.residual_xy} contact={gate.contact} "
        f"residual_src={residual_src}; nudging mode={mode}",
        flush=True,
    )
    api._capx_release_ready_busy = True
    step = _TIPXY_STEP_M if tipxy else _DEFAULT_STEP_M
    prev_dist: float | None = gate.xy_distance_m
    max_iters = _TIPXY_MAX_ITERS if tipxy else _MAX_ITERS
    try:
        for i in range(max_iters):
            _b, _s, gate = _read_live(api)
            if not gate.acquired:
                gate, promoted = promote_soft_held(gate, _s)
                if promoted:
                    print(
                        f"[release-ready] soft-held without firm acquire iter={i}; "
                        "continue tipxy (close silent-allow)",
                        flush=True,
                    )
                else:
                    print(
                        f"[release-ready] lost grasp iter={i}; refuse open",
                        flush=True,
                    )
                    return False
            if tipxy:
                gate, residual_src = _apply_tipxy_residual_source(api, gate)
                if gate is None:
                    print(
                        f"[release-ready] tipxy fair residual failed iter={i} "
                        f"({residual_src}); refuse open",
                        flush=True,
                    )
                    return False
            if tipxy and gate.release_ok_tipxy:
                print(
                    f"[release-ready] tipxy converged iter={i} "
                    f"dist={gate.xy_distance_m:.4f}m contact={gate.contact} "
                    f"residual_src={residual_src}",
                    flush=True,
                )
                return True
            if (not tipxy) and gate.release_ok:
                print(
                    f"[release-ready] converged iter={i} dist={gate.xy_distance_m:.4f}m",
                    flush=True,
                )
                return True

            dist = float(gate.xy_distance_m or 0.0)
            if dist > _DIST_ABORT_M:
                print(
                    f"[release-ready] dist abort iter={i} dist={dist:.4f}m; refuse open",
                    flush=True,
                )
                return False
            if prev_dist is not None and dist > prev_dist + _DIST_JUMP_ABORT_M:
                print(
                    f"[release-ready] dist jump abort iter={i} "
                    f"{prev_dist:.4f}->{dist:.4f}m; refuse open",
                    flush=True,
                )
                return False
            # Tipxy: flat-plate place — do not refuse on contact flicker mid-slide.
            if tipxy and not gate.contact:
                print(
                    f"[release-ready] tipxy contact=False iter={i} "
                    f"(continue band-step; flat plate)",
                    flush=True,
                )
            if gate.residual_xy is None or gate.xy_within:
                if tipxy and gate.xy_within:
                    print(
                        f"[release-ready] tipxy within iter={i} "
                        f"contact={gate.contact}; open",
                        flush=True,
                    )
                    return True
                print(
                    f"[release-ready] xy settled but not release_ok iter={i} "
                    f"contact={gate.contact} surface={gate.surface_on}; refuse open",
                    flush=True,
                )
                return False

            if tipxy:
                delta_xy, step, dist = next_xy_step_band(
                    np.asarray(gate.residual_xy, dtype=np.float64),
                    step_m=_TIPXY_STEP_M,
                )
            else:
                delta_xy, step, dist = next_xy_step(
                    np.asarray(gate.residual_xy, dtype=np.float64),
                    prev_distance_m=prev_dist,
                    step_m=step,
                )
            prev_dist = dist
            if float(np.linalg.norm(delta_xy)) < 1e-6:
                return bool(gate.release_ok_tipxy if tipxy else gate.release_ok)

            try:
                if tipxy:
                    detail = _nudge_xy_tipxy(api, delta_xy)
                    if "d_tip_z=" in detail:
                        dz = float(detail.split("d_tip_z=")[1].split()[0])
                        if abs(dz) > _TIPXY_Z_ABORT_M:
                            print(
                                f"[release-ready] tipxy Z abort {detail}; refuse open",
                                flush=True,
                            )
                            return False
                elif release_ready_goto_mode():
                    detail = _nudge_xy_goto(api, delta_xy)
                else:
                    detail = _nudge_hand_xy_lock_z(api, delta_xy)
                    if "dz=" in detail:
                        dz = float(detail.split("dz=")[1].split()[0])
                        if abs(dz) > 0.01:
                            print(
                                f"[release-ready] Z lock violated {detail}; refuse open",
                                flush=True,
                            )
                            return False
            except Exception as exc:
                print(
                    f"[release-ready] nudge failed iter={i} ({exc!r}); refuse open",
                    flush=True,
                )
                return False
            print(
                f"[release-ready] iter={i} dist={dist:.4f}m step={step:.4f}m "
                f"dxy=({delta_xy[0]:+.4f},{delta_xy[1]:+.4f}) "
                f"residual_src={residual_src} {detail}",
                flush=True,
            )

        _b, _s, gate = _read_live(api)
        if tipxy:
            gate, residual_src = _apply_tipxy_residual_source(api, gate)
            if gate is None:
                print(
                    f"[release-ready] tipxy fair residual failed after max_iters "
                    f"({residual_src}); refuse open",
                    flush=True,
                )
                return False
        ok = bool(gate.release_ok_tipxy if tipxy else gate.release_ok)
        print(
            f"[release-ready] max_iters={max_iters} release_ok={ok} "
            f"dist={gate.xy_distance_m} contact={gate.contact} "
            f"residual_src={residual_src}",
            flush=True,
        )
        return ok
    finally:
        api._capx_release_ready_busy = False
