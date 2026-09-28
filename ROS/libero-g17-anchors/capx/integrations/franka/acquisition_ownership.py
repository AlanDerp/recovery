"""Single-writer ownership for an active acquisition transaction.

v320-v332 protected the exact pose returned by ``sample_grasp_pose`` but
silently fell back to the generic Cartesian API when generated code copied,
changed, or reconstructed that pose.  The fallback gave two controllers write
access to one physical transition: the bound SL transaction and arbitrary
policy-authored ``goto_pose`` calls.

v350 makes ownership total while an acquisition attempt is active.  The
policy may still choose a target and request a new native proposal, but the
bound transaction owns its continuous pose and IK realization until close or
explicit regeneration.  This module deliberately does not add a grasp-ready
predicate: observed contact is evidence for recovery, never a prerequisite
for a successful close.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


class AcquisitionOwnershipError(RuntimeError):
    """Recoverable API-contract error raised before any physical command."""


@dataclass(frozen=True)
class GotoOwnershipDecision:
    mode: str
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    z_approach: float
    position_divergence_m: float
    orientation_divergence_rad: float


def enabled() -> bool:
    return (
        os.environ.get(
            "CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP", "0"
        )
        == "1"
    )


def total_acquisition_phase_enabled() -> bool:
    """Whether ownership starts at the symbolic acquisition phase boundary.

    v350 started ownership when ``sample_grasp_pose`` bound a proposal.  That
    left a provenance hole before the first proposal: policy code could turn
    raw type-level point clouds or ``get_object_pose`` into a handcrafted pose,
    execute ``goto_pose``, and physically close without ever creating a typed
    grasp transaction.  v350g closes that hole.  Observation remains free and
    ``goto_home_joint_position`` remains an explicit recovery action, but an
    acquisition ``goto_pose`` or close must have one current bound proposal.
    """

    return (
        enabled()
        and os.environ.get(
            "CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP", "0"
        )
        == "1"
    )


def transparent_native_transaction_enabled() -> bool:
    """Wrap pose-only native acquisition without replacing its executor."""

    return (
        enabled()
        and os.environ.get(
            "CAPX_LIBERO_G17_V350K_TRANSPARENT_NATIVE_TRANSACTION", "0"
        )
        == "1"
    )


def fg_regeneration_on_execution_failure_enabled() -> bool:
    """Release a failed ingress writer to fresh FG observation.

    A transaction owns commands only while it can realize its promised
    transition.  If the selected branch is not physically reached before
    close, retaining ownership turns a recoverable execution miss into a
    dead-end.  v350l ends that transaction without prescribing an escape:
    the next code block observes the world as it actually is and must bind a
    fresh proposal before another acquisition motion or close.
    """

    return (
        enabled()
        and os.environ.get(
            "CAPX_LIBERO_G17_V350L_FG_REGENERATION_ON_EXECUTION_FAILURE",
            "0",
        )
        == "1"
    )


def release_for_fg_regeneration(
    api: Any,
    transaction: dict[str, Any],
    *,
    stage: str,
    failure: str,
) -> dict[str, Any]:
    """End one failed writer while preserving the acquisition obligation.

    This deliberately does not move the robot.  Inverse escape is a separate
    recovery skill and is only valid when an ingress path has been recorded
    and remains reversible.  A failed pre-close transaction instead exposes
    the measured post-failure world to the outer FG regeneration loop.
    """

    current = current_transaction(api)
    if current is not transaction:
        raise AcquisitionOwnershipError(
            "Cannot release a stale acquisition transaction"
        )
    outcome = {
        "transaction_id": transaction.get("transaction_id"),
        "attempt": transaction.get("attempt"),
        "object_name": transaction.get("object_name"),
        "stage": str(stage),
        "failure": str(failure),
        "closed": False,
        "next_state": "fresh_FG_observation_then_new_proposal",
    }
    transaction["ownership_state"] = "released_to_fg_regeneration"
    transaction["release_outcome"] = outcome.copy()
    api._capx_v350l_last_transaction_failure = outcome.copy()
    api._capx_v312_acquisition_transaction = None
    api._capx_v304_ingress = None
    # The task obligation remains acquisition.  Total ownership therefore
    # rejects any raw goto/close until sample_grasp_pose binds a fresh writer.
    api._capx_v293_capture_state = "acquisition_needed"
    return outcome


def uses_transparent_native_executor(
    transaction: dict[str, Any] | None,
) -> bool:
    """True for a bound pose-only proposal owned by the native primitive."""

    return bool(
        transparent_native_transaction_enabled()
        and isinstance(transaction, dict)
        and transaction.get("pregrasp_arm_cfg_rad") is None
        and transaction.get("contact_arm_cfg_rad") is None
    )


def ingress_attempt_completed(transaction: dict[str, Any] | None) -> bool:
    """Whether close may hand this attempt to the outer FG classifier."""

    if not isinstance(transaction, dict):
        return False
    if transaction.get("contact_reached") is True:
        return True
    return bool(
        uses_transparent_native_executor(transaction)
        and transaction.get("ingress_attempted") is True
        and transaction.get("executed") is True
    )


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64).reshape(4)
    qb = np.asarray(b, dtype=np.float64).reshape(4)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    # q and -q encode the same rotation.
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), 0.0, 1.0)))


def bind_transaction(
    api: Any,
    *,
    attempt: int | None,
    object_name: str,
    tip_xyz: np.ndarray,
    tip_wxyz: np.ndarray,
    source: str,
    topdown_transaction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the one canonical transaction shape used by every proposal."""

    geometry = topdown_transaction or {}
    serial = int(getattr(api, "_capx_v350_transaction_serial", 0)) + 1
    api._capx_v350_transaction_serial = serial
    transaction = {
        "transaction_id": f"{attempt}:{serial}",
        "attempt": attempt,
        "object_name": str(object_name),
        "source": str(source),
        "tip_xyz": np.asarray(tip_xyz, dtype=np.float64).reshape(3).copy(),
        "tip_wxyz": np.asarray(tip_wxyz, dtype=np.float64).reshape(4).copy(),
        "pregrasp_tip_xyz": geometry.get("pregrasp_tip_xyz"),
        "pregrasp_arm_cfg_rad": geometry.get("pregrasp_arm_cfg_rad"),
        "contact_arm_cfg_rad": geometry.get("contact_arm_cfg_rad"),
        "pregrasp_recorded": False,
        "ingress_attempted": False,
        "native_motion_ok": None,
        # This means that the commanded ingress endpoint was reached.  It is
        # not a physical-contact predicate and is not a grasp precondition.
        "contact_reached": False,
        "executed": False,
        "ownership_state": "proposed",
        "physical_contact_required": False,
    }
    api._capx_v312_acquisition_transaction = transaction
    return transaction


def complete_native_transaction(
    api: Any,
    transaction: dict[str, Any],
    *,
    approach_distance_m: float = 0.10,
) -> dict[str, Any]:
    """Turn a pose-only native proposal into one connected IK transaction.

    This is kinematic transaction completion, not a new proposal ranker or a
    collision-prevention policy.  The contact solve is seeded only from the
    selected pregrasp branch, so execution cannot re-solve onto another elbow
    branch.  An incomplete proposal terminates before motion and is returned
    to outer native regeneration.
    """

    if (
        transaction.get("pregrasp_arm_cfg_rad") is not None
        and transaction.get("contact_arm_cfg_rad") is not None
    ):
        return transaction
    if transaction.get("ownership_state") in {
        "execution_failed",
        "proposal_incomplete",
    }:
        raise AcquisitionOwnershipError(
            "Terminal acquisition transaction cannot be completed again; "
            "request a fresh sample_grasp_pose"
        )

    import viser.transforms as vtf

    from capx.integrations.franka.measured_seed_ik import (
        _candidate_metrics,
        _fallback_seeds,
        _quat_multiply_wxyz,
        _validate_candidate,
        tool_pi_symmetry_enabled,
    )
    from capx.integrations.franka.perceptual_grasp_family import (
        ACQUISITION_ORIENTATION_TOLERANCE_RAD,
        ACQUISITION_POSITION_TOLERANCE_M,
        _v286_grasp_pose_to_hand_target,
    )
    from capx.integrations.franka import perceptual_swept_path
    from capx.integrations.franka.tip_xy_only import _live_actuated_cfg

    measured = _live_actuated_cfg(api)
    tip_xyz = np.asarray(transaction["tip_xyz"], dtype=np.float64).reshape(3)
    base_wxyz = np.asarray(
        transaction["tip_wxyz"], dtype=np.float64
    ).reshape(4)
    base_wxyz /= np.linalg.norm(base_wxyz)
    orientations = [("requested", base_wxyz)]
    if tool_pi_symmetry_enabled():
        orientations.append(
            (
                "tool_z_pi",
                _quat_multiply_wxyz(
                    base_wxyz, np.array([0.0, 0.0, 0.0, 1.0])
                ),
            )
        )

    complete: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for orientation_name, tip_wxyz in orientations:
        rotation = vtf.SO3(tip_wxyz)
        pregrasp_tip = tip_xyz + np.asarray(
            rotation
            @ np.array([0.0, 0.0, -float(approach_distance_m)])
        )
        pre_hand_xyz, pre_hand_wxyz = _v286_grasp_pose_to_hand_target(
            pregrasp_tip, tip_wxyz
        )
        contact_hand_xyz, contact_hand_wxyz = _v286_grasp_pose_to_hand_target(
            tip_xyz, tip_wxyz
        )
        pregrasp_candidates = []
        for seed_name, seed in (
            ("measured", measured.copy()),
            *_fallback_seeds(measured),
        ):
            try:
                candidate = np.asarray(
                    api._pks.solve_ik_seeded(
                        robot=api._robot,
                        target_link_name=api._target_link_name,
                        target_position=pre_hand_xyz,
                        target_wxyz=pre_hand_wxyz,
                        initial_cfg=seed,
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                _validate_candidate(api, candidate)
                metrics = _candidate_metrics(
                    api,
                    candidate,
                    measured,
                    pre_hand_xyz,
                    pre_hand_wxyz,
                )
                if (
                    metrics[0] <= ACQUISITION_POSITION_TOLERANCE_M
                    and metrics[1] <= ACQUISITION_ORIENTATION_TOLERANCE_RAD
                ):
                    pregrasp_candidates.append((seed_name, candidate, metrics))
            except Exception as exc:
                failures.append(
                    {
                        "orientation": orientation_name,
                        "stage": "pregrasp",
                        "seed": seed_name,
                        "error": repr(exc),
                    }
                )
        if not pregrasp_candidates:
            continue
        pre_seed, pregrasp_q, pre_metrics = min(
            pregrasp_candidates, key=lambda row: (row[2][3], row[2][2])
        )
        try:
            # The only contact seed is the selected pregrasp q.  This is the
            # branch-continuity invariant missing from the pose-only fallback.
            contact_q = np.asarray(
                api._pks.solve_ik_seeded(
                    robot=api._robot,
                    target_link_name=api._target_link_name,
                    target_position=contact_hand_xyz,
                    target_wxyz=contact_hand_wxyz,
                    initial_cfg=pregrasp_q,
                ),
                dtype=np.float64,
            ).reshape(-1)
            _validate_candidate(api, contact_q)
            contact_metrics = _candidate_metrics(
                api,
                contact_q,
                pregrasp_q,
                contact_hand_xyz,
                contact_hand_wxyz,
            )
            if not (
                contact_metrics[0] <= ACQUISITION_POSITION_TOLERANCE_M
                and contact_metrics[1]
                <= ACQUISITION_ORIENTATION_TOLERANCE_RAD
            ):
                raise AcquisitionOwnershipError(
                    "native contact solve is not Cartesian-accurate"
                )
            perceptual_scene = getattr(api, "_capx_v296a_pt_scene", None)
            if perceptual_scene is None:
                raise AcquisitionOwnershipError(
                    "native transaction has no robot-visible scene binding"
                )
            pregrasp_path = perceptual_swept_path.audit_candidate_path(
                api, perceptual_scene, measured, pregrasp_q
            )
            ingress_path = perceptual_swept_path.audit_candidate_path(
                api, perceptual_scene, pregrasp_q, contact_q
            )
            if not (
                pregrasp_path["predicted_arm_obstacle_path_clear"]
                and ingress_path["predicted_arm_obstacle_path_clear"]
            ):
                raise AcquisitionOwnershipError(
                    "native pregrasp-contact transaction is not connected "
                    "through the robot-visible scene"
                )
            complete.append(
                {
                    "orientation_name": orientation_name,
                    "tip_wxyz": tip_wxyz.copy(),
                    "pregrasp_tip_xyz": pregrasp_tip.copy(),
                    "pregrasp_q": pregrasp_q.copy(),
                    "contact_q": contact_q.copy(),
                    "pregrasp_seed": pre_seed,
                    "pregrasp_norm": float(pre_metrics[3]),
                    "ingress_norm": float(
                        np.linalg.norm(contact_q[:7] - pregrasp_q[:7])
                    ),
                    "pregrasp_pt_swept_path": pregrasp_path,
                    "ingress_pt_swept_path": ingress_path,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "orientation": orientation_name,
                    "stage": "contact",
                    "seed": "selected_pregrasp",
                    "error": repr(exc),
                }
            )

    if not complete:
        transaction["ownership_state"] = "proposal_incomplete"
        transaction["completion_failures"] = failures
        raise AcquisitionOwnershipError(
            "Native grasp proposal has no connected accurate "
            "pregrasp-to-contact IK transaction; request a fresh proposal"
        )
    selected = min(
        complete,
        key=lambda row: (row["pregrasp_norm"] + row["ingress_norm"]),
    )
    transaction.update(
        {
            "tip_wxyz": selected["tip_wxyz"].copy(),
            "pregrasp_tip_xyz": selected["pregrasp_tip_xyz"].copy(),
            "pregrasp_arm_cfg_rad": selected["pregrasp_q"][:7].copy(),
            "contact_arm_cfg_rad": selected["contact_q"][:7].copy(),
            "completion_source": "measured_pregrasp_then_seeded_contact",
            "completion_orientation": selected["orientation_name"],
            "pregrasp_seed": selected["pregrasp_seed"],
            "pregrasp_joint_path_norm_rad": selected["pregrasp_norm"],
            "ingress_joint_path_norm_rad": selected["ingress_norm"],
            "pregrasp_pt_swept_path": selected[
                "pregrasp_pt_swept_path"
            ],
            "ingress_pt_swept_path": selected["ingress_pt_swept_path"],
            "ownership_state": "proposed_complete",
        }
    )
    return transaction


def current_transaction(api: Any) -> dict[str, Any] | None:
    transaction = getattr(api, "_capx_v312_acquisition_transaction", None)
    attempt = getattr(api, "_capx_v293_attempt", None)
    if not isinstance(transaction, dict):
        return None
    if transaction.get("attempt") != attempt:
        return None
    return transaction


def route_goto(
    api: Any,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    z_approach: float,
) -> GotoOwnershipDecision:
    """Canonicalize or suppress a goto owned by an active acquisition.

    Any first goto after a bound proposal is interpreted as a request to
    execute that proposal.  Generated approach offsets and quaternion edits
    cannot replace its continuous realization.  Once the ingress endpoint has
    been reached, further gotos before close are redundant and are suppressed.
    """

    requested_position = np.asarray(position, dtype=np.float64).reshape(3)
    requested_quaternion = np.asarray(
        quaternion_wxyz, dtype=np.float64
    ).reshape(4)
    if not enabled() or getattr(api, "_capx_v293_capture_state", None) != (
        "acquisition_needed"
    ):
        return GotoOwnershipDecision(
            "passthrough",
            requested_position.copy(),
            requested_quaternion.copy(),
            float(z_approach),
            0.0,
            0.0,
        )

    attempt = getattr(api, "_capx_v293_attempt", None)
    transaction = current_transaction(api)
    if transaction is None:
        if attempt is None and not total_acquisition_phase_enabled():
            # Before the first proposal the baseline remains free to observe
            # and position the robot.  Ownership starts at proposal binding.
            return GotoOwnershipDecision(
                "passthrough",
                requested_position.copy(),
                requested_quaternion.copy(),
                float(z_approach),
                0.0,
                0.0,
            )
        if attempt is None:
            raise AcquisitionOwnershipError(
                "Acquisition goto has no typed grasp transaction; request "
                "sample_grasp_pose instead of converting raw perception or "
                "get_object_pose into an acquisition motion"
            )
        raise AcquisitionOwnershipError(
            "Acquisition goto has no current bound transaction; request a "
            "fresh sample_grasp_pose instead of executing an unowned pose"
        )

    canonical_position = np.asarray(
        transaction["tip_xyz"], dtype=np.float64
    ).reshape(3)
    canonical_quaternion = np.asarray(
        transaction["tip_wxyz"], dtype=np.float64
    ).reshape(4)
    position_divergence = float(
        np.linalg.norm(requested_position - canonical_position)
    )
    orientation_divergence = _quat_angle(
        requested_quaternion, canonical_quaternion
    )
    if transaction.get("contact_reached") or transaction.get("executed"):
        return GotoOwnershipDecision(
            "suppress_after_ingress",
            canonical_position.copy(),
            canonical_quaternion.copy(),
            0.0,
            position_divergence,
            orientation_divergence,
        )
    if transaction.get("ownership_state") in {
        "execution_failed",
        "proposal_incomplete",
    }:
        raise AcquisitionOwnershipError(
            "Terminal acquisition transaction cannot be replayed; request a "
            "fresh sample_grasp_pose"
        )

    # A precomputed transaction ignores z_approach and owns both halves.  A
    # native fallback still uses the legacy insertion primitive, but always at
    # the canonical pose.  Preserve a positive caller approach when present;
    # otherwise use the already-established generic insertion distance.
    has_bound_branch = (
        transaction.get("pregrasp_arm_cfg_rad") is not None
        and transaction.get("contact_arm_cfg_rad") is not None
    )
    canonical_z_approach = (
        0.0
        if has_bound_branch
        else (abs(float(z_approach)) if float(z_approach) != 0.0 else 0.10)
    )
    return GotoOwnershipDecision(
        "canonical_transaction",
        canonical_position.copy(),
        canonical_quaternion.copy(),
        canonical_z_approach,
        position_divergence,
        orientation_divergence,
    )


def require_close_transaction(api: Any) -> dict[str, Any] | None:
    """Return the active transaction or reject an unowned physical close."""

    if not enabled() or getattr(api, "_capx_v293_capture_state", None) != (
        "acquisition_needed"
    ):
        return None
    attempt = getattr(api, "_capx_v293_attempt", None)
    transaction = current_transaction(api)
    if transaction is None:
        if attempt is None and total_acquisition_phase_enabled():
            raise AcquisitionOwnershipError(
                "Acquisition close has no typed grasp transaction; request "
                "sample_grasp_pose before physically closing"
            )
        if attempt is not None:
            raise AcquisitionOwnershipError(
                "Acquisition close has no current bound transaction; request "
                "a fresh sample_grasp_pose before closing"
            )
    return transaction
