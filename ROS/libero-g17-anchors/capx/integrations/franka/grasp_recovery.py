"""FG-guided recovery for a completed, empty top-down grasp.

The native v313 acquisition remains untouched through its first physical
close.  Only a bound target near miss is intercepted: close-time FG says the
target was contacted or the observation tip was inside its oriented volume,
while FirmAcquired is false.  Recovery opens, reverses to the reached
pregrasp, translates one hard-coded bowl-rim width toward the target in XY,
and repeats the same ingress vector with the same orientation.

v318 replaces the v317 privileged radial direction and fixed scalar with a
closed predicate / robot-model controller.  A z-aligned target contact with an
empty gripper is the near-miss event.  The jaw axis comes from Panda FK, its
sign comes from the contacted finger predicate, and the scalar follows the
sign comes from discrete FG (contact side or RelXY direction), and the scalar
follows the robot's intrinsic finger travel: 20, 10, 5, 2.5 mm. A probe is too far only
when it both fails to reach its endpoint and contacts a non-target object.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


ENV = "CAPX_LIBERO_G17_V317_GRASP_RECOVERY_POC"
ADAPTIVE_ENV = "CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY"
FROZEN_TRIAL02_ENV = "CAPX_LIBERO_G17_V317_FROZEN_TRIAL02_TRANSACTION"
POC_GRASPABLE_DISTANCE_M = 0.010
PANDA_FINGER_TRAVEL_M = 0.040
ADAPTIVE_START_M = PANDA_FINGER_TRAVEL_M / 2.0
ADAPTIVE_PROBE_COUNT = 4


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1" or adaptive_enabled()


def adaptive_enabled() -> bool:
    return os.environ.get(ADAPTIVE_ENV, "0") == "1"


def frozen_trial02_recommended_pose(api: Any, attempt: int) -> dict[str, Any] | None:
    """Return the recorded v313 trial-02 transaction for a mechanism smoke.

    This is an explicitly test-only fixture.  It removes live GraspNet
    candidate drift from the one-trial causal test and must remain disabled
    for K=10/live evaluation.
    """

    if os.environ.get(FROZEN_TRIAL02_ENV, "0") != "1":
        return None
    trial = int(os.environ.get("CAPX_ACTIVE_TRIAL", "-1"))
    if trial != 2 or int(attempt) != 1:
        return None
    return {
        "tip_xyz": [0.7547516, -0.1550308, 0.1966753],
        "tip_wxyz": [0.0, 1.0, 0.0, 0.0],
        "selection": "frozen_v313_trial02_mechanism_replay",
        "source_job": 527196,
        "topdown_transaction": {
            "pregrasp_tip_xyz": [0.7547516, -0.1550308, 0.2966753],
            "pregrasp_arm_cfg_rad": [
                -0.16026011,
                0.84098589,
                -0.08205885,
                -0.88883048,
                0.06189067,
                1.72784507,
                0.56066686,
            ],
            "contact_arm_cfg_rad": [
                -0.15054511,
                0.87866533,
                -0.08616009,
                -1.07922792,
                0.07152712,
                1.95517564,
                0.55292588,
            ],
        },
    }


@dataclass(frozen=True)
class CloseTimeFg:
    firm_acquired: bool
    target_contact: bool
    tip_in_target_volume: bool
    relative_x: str
    relative_y: str
    relative_z: str
    coordinate_in_target_band: bool
    target_direction_xy: tuple[int, int]
    attachment: str
    target_contact_side: str = "NONE"
    target_body_name: str = ""

    @property
    def near_miss(self) -> bool:
        return bool(
            not self.firm_acquired
            and self.attachment == "empty"
            and (
                self.target_contact
                or self.coordinate_in_target_band
                or self.tip_in_target_volume
            )
        )

    @property
    def adaptive_near_miss(self) -> bool:
        """v318 event: z-aligned target contact, but no acquisition."""
        return bool(
            not self.firm_acquired
            and self.attachment == "empty"
            and self.target_contact
            and self.relative_z == "WITHIN_REFERENCE_BAND"
        )


def latch_grasp_event(before: CloseTimeFg, after: CloseTimeFg) -> CloseTimeFg:
    """Latch transient grasp evidence while keeping the post-close outcome."""

    return CloseTimeFg(
        firm_acquired=after.firm_acquired,
        target_contact=before.target_contact or after.target_contact,
        tip_in_target_volume=(
            before.tip_in_target_volume or after.tip_in_target_volume
        ),
        relative_x=after.relative_x,
        relative_y=after.relative_y,
        relative_z=after.relative_z,
        coordinate_in_target_band=(
            before.coordinate_in_target_band
            or after.coordinate_in_target_band
        ),
        target_direction_xy=after.target_direction_xy,
        attachment=after.attachment,
        target_contact_side=_latch_contact_side(
            before.target_contact_side, after.target_contact_side
        ),
        target_body_name=after.target_body_name or before.target_body_name,
    )


def _latch_contact_side(before: str, after: str) -> str:
    sides = {str(before), str(after)} - {"NONE"}
    if not sides:
        return "NONE"
    if "BOTH" in sides or sides == {"FINGER1", "FINGER2"}:
        return "BOTH"
    return next(iter(sides))


def _live_target_contact_side(api: Any, target_body_name: str) -> str:
    """Compile exact simulator contact into a finite FG side predicate."""
    sim = api._env.handle.env.sim
    token = str(target_body_name).lower().replace(" ", "_")
    fingers: set[str] = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        names = (
            str(sim.model.geom_id2name(int(contact.geom1)) or ""),
            str(sim.model.geom_id2name(int(contact.geom2)) or ""),
        )
        lowered = tuple(name.lower() for name in names)
        if not token or not any(token in name for name in lowered):
            continue
        if any("gripper0_finger1" in name for name in lowered):
            fingers.add("FINGER1")
        if any("gripper0_finger2" in name for name in lowered):
            fingers.add("FINGER2")
    if len(fingers) == 2:
        return "BOTH"
    return next(iter(fingers), "NONE")


def jaw_axis_xy_from_fk(hand_wxyz: np.ndarray) -> np.ndarray:
    """Return the undirected Panda opening axis projected into world XY."""
    axis = _rotation_matrix_wxyz(hand_wxyz) @ np.array([0.0, 1.0, 0.0])
    xy = np.asarray(axis[:2], dtype=np.float64)
    norm = float(np.linalg.norm(xy))
    if norm < 0.90:
        raise RuntimeError(f"Panda jaw axis is not approximately horizontal: {axis}")
    return xy / norm


def signed_jaw_axis_xy(
    hand_wxyz: np.ndarray,
    contact_side: str,
    fg_direction_xy: tuple[int, int] | np.ndarray | None = None,
) -> np.ndarray:
    """Orient the FK axis using discrete FG, never a continuous displacement.

    A unique contacted finger is already an unambiguous signed predicate.  A
    hollow-object center miss can contact both fingers, so in that case the
    closed RelXY-derived direction predicate supplies the sign projected onto
    the robot-model jaw axis.
    """
    axis = jaw_axis_xy_from_fk(hand_wxyz)
    if contact_side == "FINGER1":
        return axis
    if contact_side == "FINGER2":
        return -axis
    direction = np.asarray(fg_direction_xy, dtype=np.float64).reshape(2)
    projection = float(np.dot(direction, axis))
    if abs(projection) <= 1e-9:
        raise RuntimeError(
            "FG direction has no component on the FK jaw axis: "
            f"side={contact_side}, direction={direction}, axis={axis}"
        )
    return axis * float(np.sign(projection))


def adaptive_probe_magnitudes() -> tuple[float, ...]:
    return tuple(ADAPTIVE_START_M / (2**i) for i in range(ADAPTIVE_PROBE_COUNT))


def _non_target_collision(report: dict[str, Any] | None, target: str) -> bool:
    token = str(target).lower().replace(" ", "_")
    for row in (report or {}).get("robot_contact_pairs", []):
        pair = [str(name).lower() for name in row.get("geom_pair", [])]
        # Self-contact is not an obstacle observation.
        external = [
            name for name in pair
            if not any(tag in name for tag in ("robot0", "gripper0", "panda"))
        ]
        if external and not all(token and token in name for name in external):
            return True
    return False


def blocked_too_far(
    report: dict[str, Any] | None,
    *,
    reached: bool,
    target_body_name: str,
) -> bool:
    return bool(not reached and _non_target_collision(report, target_body_name))


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    import viser.transforms as vtf

    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return np.asarray(vtf.SO3(quat).as_matrix(), dtype=np.float64)


def _tip_inside_target(api: Any, raw: Any) -> tuple[bool, np.ndarray]:
    tip = np.asarray(
        api.get_observation()["robot_cartesian_pos"][:3], dtype=np.float64
    )
    target = raw.objects["obj01"]
    center = np.asarray(target.position, dtype=np.float64).reshape(3)
    half = np.asarray(target.half_extents, dtype=np.float64).reshape(3)
    rotation = _rotation_matrix_wxyz(target.orientation_wxyz)
    local = rotation.T @ (tip - center)
    return bool(np.all(np.abs(local) <= half)), center - tip


def correction_from_fg(
    relative_x: str,
    relative_y: str,
    target_delta_xy: np.ndarray,
    *,
    distance_m: float = POC_GRASPABLE_DISTANCE_M,
) -> np.ndarray:
    """One fixed-magnitude XY correction toward the bound target.

    Existing RelXY bands decide every axis outside the target band.  When the
    tip is already inside both coarse bands (the center-empty case), moving
    toward the object center would deepen the miss.  The close-time FG instead
    chooses the dominant radial axis away from the center, back toward a
    graspable rim. Continuous displacement magnitude is never consumed by the
    controller.
    """

    bands = (str(relative_x), str(relative_y))
    raw = np.asarray(target_delta_xy, dtype=np.float64).reshape(2)
    direction = np.zeros(2, dtype=np.float64)
    for axis, band in enumerate(bands):
        if band == "NEGATIVE_OF_REFERENCE_BAND":
            direction[axis] = 1.0
        elif band == "POSITIVE_OF_REFERENCE_BAND":
            direction[axis] = -1.0
    if not np.any(direction):
        # Empty while inside a hollow target means center-void, so move
        # radially away from its center toward the contacted rim.
        dominant = int(np.argmax(np.abs(raw)))
        direction[dominant] = -float(np.sign(raw[dominant]))
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        return np.zeros(2, dtype=np.float64)
    return direction * (float(distance_m) / norm)


def capture_close_time_fg(api: Any) -> CloseTimeFg:
    """Compile the exact closed FG at the physical close boundary.

    Target geometry is compiled here into predicates only.  The recovery
    controller receives a discrete direction and never the continuous target
    displacement or privileged motion magnitude.
    """

    from capx.verification.libero_g17_adapter import (
        extract_libero_raw_scene,
        extract_libero_snapshot,
    )

    binding, snapshot, attachment = extract_libero_snapshot(api._env)
    raw = extract_libero_raw_scene(api._env, binding)
    prefix = "executor_object.exe01.obj01"
    tip_inside, target_minus_tip = _tip_inside_target(api, raw)
    relative_x = snapshot.values[f"{prefix}.relative_x"]
    relative_y = snapshot.values[f"{prefix}.relative_y"]
    relative_z = snapshot.values[f"{prefix}.relative_z"]
    correction = correction_from_fg(relative_x, relative_y, target_minus_tip[:2])
    direction = tuple(int(np.sign(value)) for value in correction)
    target_body_name = binding.payload_name
    return CloseTimeFg(
        firm_acquired=bool(
            snapshot.values[f"{prefix}.holding"] == "TRUE"
            and attachment.kind == "holding_payload"
        ),
        target_contact=snapshot.values[f"{prefix}.contact"] == "TRUE",
        tip_in_target_volume=tip_inside,
        relative_x=relative_x,
        relative_y=relative_y,
        relative_z=relative_z,
        coordinate_in_target_band=all(
            value == "WITHIN_REFERENCE_BAND"
            for value in (relative_x, relative_y, relative_z)
        ),
        target_direction_xy=direction,
        attachment=attachment.kind,
        target_contact_side=_live_target_contact_side(api, target_body_name),
        target_body_name=target_body_name,
    )


def execute_retry_after_escape(
    api: Any,
    transaction: dict[str, Any],
    close_fg: CloseTimeFg,
) -> dict[str, Any]:
    """Repeat the bound ingress, using v318 inward probes when enabled."""

    from capx.integrations.franka import acquisition_audit as _acq
    from capx.integrations.franka import acquisition_extract as _extract
    from capx.integrations.franka.measured_seed_ik import (
        _candidate_metrics,
        _validate_candidate,
    )
    from capx.integrations.franka.perceptual_grasp_family import (
        ACQUISITION_ORIENTATION_TOLERANCE_RAD,
        ACQUISITION_POSITION_TOLERANCE_M,
        _v286_grasp_pose_to_hand_target,
    )
    from capx.integrations.franka.tip_xy_only import (
        _live_actuated_cfg,
        fk_panda_hand,
        move_tip_xy,
    )

    attempt, object_name = _acq.active_attempt(api)
    contact_tip = np.asarray(transaction["tip_xyz"], dtype=np.float64)
    original_pregrasp_tip = np.asarray(
        transaction["pregrasp_tip_xyz"], dtype=np.float64
    )
    tip_wxyz = np.asarray(transaction["tip_wxyz"], dtype=np.float64)
    if adaptive_enabled():
        _hand_xyz, hand_wxyz0 = fk_panda_hand(api, _live_actuated_cfg(api))
        direction = signed_jaw_axis_xy(
            hand_wxyz0,
            close_fg.target_contact_side,
            close_fg.target_direction_xy,
        )
        magnitudes = adaptive_probe_magnitudes()
    else:
        direction = np.asarray(close_fg.target_direction_xy, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        magnitudes = (POC_GRASPABLE_DISTANCE_M,)

    cumulative_xy = np.zeros(2, dtype=np.float64)
    retry_fg = close_fg
    last_retreat: dict[str, Any] | None = None
    for probe_index, magnitude in enumerate(magnitudes, start=1):
        desired_xy = direction * float(magnitude)
        command_xy = desired_xy - cumulative_xy
        move = move_tip_xy(
            api, float(command_xy[0]), float(command_xy[1]), log=True
        )
        # A later local IK rejection does not undo already committed, measured
        # tip motion.  Probe the actual reached pregrasp rather than throwing
        # away progress or replaying the stale full command.  Only the trusted
        # tip primitive's robot-proprioceptive result is consumed here.
        partial_progress = bool(move.usable_partial_progress)
        if not (move.acceptance_ok or partial_progress):
            raise RuntimeError(
                f"GR pregrasp XY probe {probe_index} failed: {move}"
            )
        achieved_step = np.asarray(move.achieved_xy, dtype=np.float64)
        cumulative_xy += achieved_step
        corrected_pregrasp_tip = original_pregrasp_tip.copy()
        corrected_pregrasp_tip[:2] += cumulative_xy
        corrected_contact_tip = corrected_pregrasp_tip + (
            contact_tip - original_pregrasp_tip
        )
        hand_xyz, hand_wxyz = _v286_grasp_pose_to_hand_target(
            corrected_contact_tip, tip_wxyz
        )
        live = _live_actuated_cfg(api)
        candidate = np.asarray(
            api._pks.solve_ik_seeded(
                robot=api._robot,
                target_link_name=api._target_link_name,
                target_position=hand_xyz,
                target_wxyz=hand_wxyz,
                initial_cfg=live,
            ),
            dtype=np.float64,
        ).reshape(-1)
        _validate_candidate(api, candidate)
        pos_e, ori_e, max_dq, _norm_dq = _candidate_metrics(
            api, candidate, live, hand_xyz, hand_wxyz
        )
        if (
            pos_e > ACQUISITION_POSITION_TOLERANCE_M
            or ori_e > ACQUISITION_ORIENTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                "GR corrected contact IK inaccurate: "
                f"position={pos_e:.5f}, orientation={ori_e:.5f}"
            )
        _extract.record_ingress(
            api,
            attempt=attempt,
            object_name=object_name,
            arm_cfg=live,
            hand_target_xyz=corrected_pregrasp_tip,
            hand_target_wxyz=tip_wxyz,
        )
        _acq.audit(
            api,
            "gr_recovery_retry_command",
            attempt=attempt,
            object_name=object_name,
            probe_index=probe_index,
            probe_magnitude_m=float(magnitude),
            close_fg=close_fg.__dict__,
            fk_jaw_direction_xy=direction.round(7).tolist(),
            delta_xy_m=command_xy.round(7).tolist(),
            cumulative_delta_xy_m=cumulative_xy.round(7).tolist(),
            achieved_delta_xy_m=achieved_step.round(7).tolist(),
            tip_xy_z0_m=float(move.tip_z0),
            tip_xy_z1_m=float(move.tip_z1),
            tip_xy_delta_z_m=float(move.d_tip_z),
            tip_xy_acceptance_ok=bool(move.acceptance_ok),
            tip_xy_partial_progress_used=partial_progress,
            tip_xy_remaining_xy_m=list(move.measured_remaining_xy),
            tip_xy_forward_progress_m=float(move.measured_forward_progress_m),
            corrected_pregrasp_tip_xyz=corrected_pregrasp_tip.round(7).tolist(),
            corrected_contact_tip_xyz=corrected_contact_tip.round(7).tolist(),
            position_error_m=float(pos_e),
            orientation_error_rad=float(ori_e),
            max_joint_delta_rad=float(max_dq),
        )
        api._env.move_to_joints_blocking(candidate[:7], max_steps=600)
        reached_cfg = _live_actuated_cfg(api)
        reached_hand_xyz, reached_hand_wxyz = fk_panda_hand(api, reached_cfg)
        reached_obs_tip = np.asarray(
            api.get_observation()["robot_cartesian_pos"][:3], dtype=np.float64
        )
        ingress_report = getattr(api._env, "_last_joint_move_report", None)
        reached_position_error = float(
            np.linalg.norm(reached_hand_xyz - np.asarray(hand_xyz))
        )
        reached_orientation_error = float(
            _candidate_metrics(
                api, reached_cfg, reached_cfg, hand_xyz, hand_wxyz
            )[1]
        )
        reached_joint_error = float(
            np.max(np.abs(reached_cfg[:7] - candidate[:7]))
        )
        endpoint_reached = bool(
            reached_position_error <= ACQUISITION_POSITION_TOLERANCE_M
            and reached_orientation_error <= ACQUISITION_ORIENTATION_TOLERANCE_RAD
            and reached_joint_error <= 0.02
        )
        too_far = blocked_too_far(
            ingress_report,
            reached=endpoint_reached,
            target_body_name=close_fg.target_body_name,
        )
        _acq.audit(
            api,
            "gr_recovery_retry_ingress_result",
            attempt=attempt,
            object_name=object_name,
            probe_index=probe_index,
            probe_magnitude_m=float(magnitude),
            endpoint_reached=endpoint_reached,
            blocked_too_far=too_far,
            desired_tip_xyz=corrected_contact_tip.round(7).tolist(),
            desired_hand_xyz=np.asarray(hand_xyz).round(7).tolist(),
            reached_observation_tip_xyz=reached_obs_tip.round(7).tolist(),
            reached_hand_xyz=reached_hand_xyz.round(7).tolist(),
            reached_hand_position_error_m=reached_position_error,
            reached_hand_orientation_error_rad=reached_orientation_error,
            reached_joint_error_max_rad=reached_joint_error,
            desired_arm_q=np.asarray(candidate[:7]).round(8).tolist(),
            reached_arm_q=np.asarray(reached_cfg[:7]).round(8).tolist(),
            joint_move_report=ingress_report,
        )

        # Never close at a physically unreached endpoint.  Reverse the exact
        # attempted ingress before changing the scalar at the pregrasp plane.
        if not endpoint_reached:
            last_retreat = _extract.retreat_after_close(api)
            if last_retreat is None:
                raise RuntimeError("GR blocked probe has no inverse ingress")
            _acq.audit(
                api,
                "gr_recovery_probe_update",
                attempt=attempt,
                object_name=object_name,
                probe_index=probe_index,
                outcome=("blocked_too_far" if too_far else "unreached"),
                next_magnitude_m=(
                    float(magnitudes[probe_index])
                    if probe_index < len(magnitudes)
                    else None
                ),
            )
            continue

        api._env._set_gripper(0.0)
        for _ in range(60):
            api._env._step_once()
        retry_fg = capture_close_time_fg(api)
        _acq.audit(
            api,
            "gr_recovery_retry_close_fg",
            attempt=attempt,
            object_name=object_name,
            probe_index=probe_index,
            probe_magnitude_m=float(magnitude),
            **retry_fg.__dict__,
        )
        if not retry_fg.firm_acquired:
            api._env._set_gripper(1.0)
            for _ in range(40):
                api._env._step_once()
        last_retreat = _extract.retreat_after_close(api)
        if last_retreat is None:
            raise RuntimeError("GR retry completed without inverse ingress")
        if retry_fg.firm_acquired:
            return {
                "retry_fg": retry_fg,
                "retreat": last_retreat,
                "delta_xy_m": cumulative_xy,
                "probe_index": probe_index,
            }

    return {
        "retry_fg": retry_fg,
        "retreat": last_retreat,
        "delta_xy_m": cumulative_xy,
        "probe_index": len(magnitudes),
    }
