"""Immutable plans and manifests for the ninth R1 Lite sandwich pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.r1lite.bimanual_kinematics import (
    strict_topdown_edge_rotation,
    tcp_position_for_fingertip,
)
from capx.integrations.r1lite.execution_bundle import canonical_sha256
from capx.integrations.r1lite.sandwich_perception import SCENE_SCHEMA


INTENT_SCHEMA = "capx.r1lite.ninth_sandwich_intent.v1"
PLAN_SCHEMA = "capx.r1lite.ninth_sandwich_sequential.v1"
POLICY_SCHEMA = "capx.r1lite.ninth_sandwich_policy.v1"
MANIFEST_SCHEMA = "capx.r1lite.ninth_sandwich_session.v1"
STAGE_ORDER = (
    "pregrasp",
    "align_grasp",
    "grasp",
    "lift",
    "place_on_target",
    "release_retract",
    "home",
)
SOURCE_ORIENTATION_STAGES = ("pregrasp", "align_grasp", "grasp", "lift")
TARGET_ORIENTATION_STAGES = ("place_on_target", "release_retract")
PHASES = {
    "donut_on_brown": {
        "source": "donut",
        "support": "brown_bread",
        "target_orientation": "support",
    },
    "mixed_bread_on_stack": {
        "source": "mixed_bread",
        "support": "donut",
        "target_orientation": "brown_bread",
    },
}


def _vector(value: Any, size: int, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must contain {size} finite values")
    return result


def _unit_xy(value: Any, *, field: str) -> np.ndarray:
    result = _vector(value, 2, field=field)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{field} cannot be zero")
    return result / norm


def _topdown_quaternion(jaw_axis_xy: np.ndarray) -> np.ndarray:
    rotation = strict_topdown_edge_rotation(jaw_axis_xy)
    return Rotation.from_matrix(rotation).as_quat()


def _stage(
    stage_id: str,
    fingertip: np.ndarray,
    quaternion: np.ndarray,
    jaw_axis_xy: np.ndarray,
    *,
    tcp_to_fingertip_m: float,
    speed: str,
) -> dict[str, Any]:
    tcp = tcp_position_for_fingertip(
        fingertip,
        quaternion,
        tcp_to_fingertip_m,
    )
    return {
        "id": stage_id,
        "active_side": "auto",
        "active_tcp_position_base_m": tcp.tolist(),
        "active_fingertip_position_base_m": fingertip.tolist(),
        "target_quaternion_xyzw": quaternion.tolist(),
        "tool_z_base": [0.0, 0.0, -1.0],
        "jaw_closing_axis_tool": "tool_y",
        "jaw_closing_axis_base_xy": jaw_axis_xy.tolist(),
        "orientation": "strict_topdown_object_axis",
        "tilt_search_deg": [0.0],
        "inactive_arm_policy": "hold_session_initial_joints",
        "speed": speed,
    }


def build_sandwich_intent(
    scene: Mapping[str, Any],
    *,
    phase: str,
    first_image_sha256: str,
    second_image_sha256: str,
    repeatability_m: float,
    tcp_to_fingertip_m: float = 0.078,
    gripper_max_span_m: float = 0.090,
    fingertip_bottom_margin_m: float = 0.004,
    pregrasp_clearance_m: float = 0.100,
    alignment_clearance_m: float = 0.050,
    lift_clearance_m: float = 0.120,
) -> dict[str, Any]:
    """Compile arm-neutral geometry which certification resolves to one side."""
    if scene.get("schema") != SCENE_SCHEMA:
        raise ValueError("sandwich intent requires a pipeline-nine RGB-D scene")
    if phase not in PHASES:
        raise ValueError(f"unsupported sandwich phase {phase!r}")
    objects = scene["objects"]
    definition = PHASES[phase]
    source = objects[definition["source"]]
    support = objects[definition["support"]]
    brown = objects["brown_bread"]
    source_center = _vector(source["center_base_m"], 3, field="source center")
    target_center = _vector(support["center_base_m"], 3, field="support center")
    source_minor = _unit_xy(
        source["minor_axis_base_xy"], field="source minor axis"
    )
    target_reference = brown if definition["target_orientation"] == "brown_bread" else support
    target_minor = _unit_xy(
        target_reference["minor_axis_base_xy"], field="target minor axis"
    )
    source_span = float(source["width_m"])
    if source_span <= 0.0 or source_span > gripper_max_span_m:
        raise ValueError(
            f"{definition['source']} grasp span {source_span:.4f}m exceeds "
            f"{gripper_max_span_m:.4f}m"
        )
    source_table_z = float(source["table_z_at_center_m"])
    conservative_table_z = max(
        float(item["table_z_at_center_m"]) for item in objects.values()
    )
    support_top_z = float(support["top_z_base_m"])
    grasp_fingertip = np.array(
        [
            source_center[0],
            source_center[1],
            source_table_z + fingertip_bottom_margin_m,
        ]
    )
    place_fingertip = np.array(
        [
            target_center[0],
            target_center[1],
            support_top_z + fingertip_bottom_margin_m,
        ]
    )
    source_quaternion = _topdown_quaternion(source_minor)
    target_quaternion = _topdown_quaternion(target_minor)
    stages = [
        _stage(
            "pregrasp",
            grasp_fingertip + np.array([0.0, 0.0, pregrasp_clearance_m]),
            source_quaternion,
            source_minor,
            tcp_to_fingertip_m=tcp_to_fingertip_m,
            speed="slow",
        ),
        _stage(
            "align_grasp",
            grasp_fingertip + np.array([0.0, 0.0, alignment_clearance_m]),
            source_quaternion,
            source_minor,
            tcp_to_fingertip_m=tcp_to_fingertip_m,
            speed="very_slow",
        ),
        _stage(
            "grasp",
            grasp_fingertip,
            source_quaternion,
            source_minor,
            tcp_to_fingertip_m=tcp_to_fingertip_m,
            speed="very_slow",
        ),
        _stage(
            "lift",
            grasp_fingertip + np.array([0.0, 0.0, lift_clearance_m]),
            source_quaternion,
            source_minor,
            tcp_to_fingertip_m=tcp_to_fingertip_m,
            speed="very_slow",
        ),
        _stage(
            "place_on_target",
            place_fingertip,
            target_quaternion,
            target_minor,
            tcp_to_fingertip_m=tcp_to_fingertip_m,
            speed="very_slow",
        ),
        _stage(
            "release_retract",
            place_fingertip + np.array([0.0, 0.0, pregrasp_clearance_m]),
            target_quaternion,
            target_minor,
            tcp_to_fingertip_m=tcp_to_fingertip_m,
            speed="very_slow",
        ),
        {
            "id": "home",
            "active_side": "auto",
            "inactive_arm_policy": "hold_session_initial_joints",
            "target": "session_initial_joint_state",
            "speed": "slow",
        },
    ]
    intent = {
        "schema": INTENT_SCHEMA,
        "pipeline": 9,
        "task": "two_layer_sandwich_stack",
        "phase": phase,
        "phase_definition": deepcopy(definition),
        "arm_selection": "automatic_during_offline_certification",
        "perception": {
            "method": (
                "head_left_sam3_plus_calibrated_plane_homography"
                if scene.get("perception_method")
                == "sam3_plus_calibrated_plane_homography"
                else "head_left_sam3_plus_aligned_stereo_depth_and_tf"
            ),
            "scene": deepcopy(dict(scene)),
            "first_image_sha256": first_image_sha256,
            "second_image_sha256": second_image_sha256,
            "two_frame_maximum_center_drift_m": float(repeatability_m),
        },
        "grasp_geometry": {
            "source_object": definition["source"],
            "support_object": definition["support"],
            "source_span_m": source_span,
            "gripper_max_span_m": float(gripper_max_span_m),
            "source_jaw_axis_base_xy": source_minor.tolist(),
            "target_jaw_axis_base_xy": target_minor.tolist(),
            "target_center_base_xy_m": target_center[:2].tolist(),
            "support_top_z_base_m": support_top_z,
        },
        "tcp": {
            "left": "left_gripper_tool_z",
            "right": "right_gripper_tool_z",
        },
        "stages": stages,
        "safety": {
            "tcp_to_fingertip_m": float(tcp_to_fingertip_m),
            "table_z_m": float(conservative_table_z),
            "fingertip_bottom_margin_m": float(fingertip_bottom_margin_m),
            "inactive_arm_policy": "hold_session_initial_joints",
            "collision_policy": "combined_interarm_table_and_open_gripper",
            "minimum_table_clearance_m": 0.0,
            "minimum_link_clearance_m": 0.005,
            "minimum_gripper_envelope_separation_m": 0.12552056969962836,
            "home_joint_tolerance_rad": 0.01,
            "cartesian_refinement_policy": {
                "enabled": True,
                "feedback_source": "physical_tool_z_pose_after_each_motion",
                "target_tolerance_m": 0.012,
                "grasp_xy_tolerance_m": 0.005,
                "grasp_z_tolerance_m": 0.003,
                "max_iterations": 20,
                "ik_probes_per_iteration": 3,
                "minimum_improvement_m": 0.001,
                "local_optimum_stall_iterations": 2,
                "max_incremental_joint_delta_rad": 1.0,
                "joint_arrival_tolerance_rad": 0.01,
                "joint_arrival_max_retries": 2,
                "joint_hard_safety_limit_rad": 0.05,
                "joint_command_progress_steps": 1,
                "paired_swept_collision_gate_each_iteration": True,
                "force_continue_after_maximum_iterations": False,
            },
            "grasp_retention_policy": {
                "feedback_source": "gripper_position",
                "empty_close_feedback": 5.5,
                "minimum_gap_above_empty_close": 2.0,
                "verify_after_lift": True,
            },
            "failure_policy": "hold_both_and_abort",
            "automatic_resume_from_saved_grasp": False,
        },
    }
    validate_sandwich_intent(intent)
    return intent


def validate_sandwich_intent(intent: Mapping[str, Any]) -> None:
    if intent.get("schema") != INTENT_SCHEMA or intent.get("pipeline") != 9:
        raise ValueError("unexpected pipeline-nine sandwich intent schema")
    phase = str(intent.get("phase"))
    if phase not in PHASES or intent.get("phase_definition") != PHASES[phase]:
        raise ValueError("sandwich phase definition mismatch")
    if intent.get("arm_selection") != "automatic_during_offline_certification":
        raise ValueError("sandwich intent must defer arm selection to certification")
    if intent.get("perception", {}).get("method") not in {
        "head_left_sam3_plus_aligned_stereo_depth_and_tf",
        "head_left_sam3_plus_calibrated_plane_homography",
    }:
        raise ValueError("sandwich intent has an unsupported perception backend")
    stages = intent.get("stages")
    if not isinstance(stages, list) or tuple(
        stage.get("id") for stage in stages
    ) != STAGE_ORDER:
        raise ValueError("sandwich stage order is immutable")
    for stage in stages:
        if stage.get("active_side") != "auto":
            raise ValueError("unselected intent stage must remain arm-neutral")
        if stage.get("inactive_arm_policy") != "hold_session_initial_joints":
            raise ValueError("inactive arm must hold the certified initial state")
        if stage["id"] == "home":
            if stage.get("target") != "session_initial_joint_state":
                raise ValueError("home must restore the certified initial state")
            continue
        _vector(stage.get("active_tcp_position_base_m"), 3, field="stage TCP")
        _vector(
            stage.get("active_fingertip_position_base_m"),
            3,
            field="stage fingertip",
        )
        quaternion = _vector(
            stage.get("target_quaternion_xyzw"), 4, field="stage quaternion"
        )
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6):
            raise ValueError("stage quaternion must be normalized")
        if stage.get("orientation") != "strict_topdown_object_axis":
            raise ValueError("sandwich motions must remain strict top-down")
        if stage.get("tool_z_base") != [0.0, 0.0, -1.0]:
            raise ValueError("sandwich tool Z must point down")
        if stage.get("tilt_search_deg") != [0.0]:
            raise ValueError("sandwich intent prohibits tilted IK")
    geometry = intent.get("grasp_geometry", {})
    span = float(geometry.get("source_span_m"))
    limit = float(geometry.get("gripper_max_span_m"))
    if not 0.0 < span <= limit:
        raise ValueError("sandwich source span exceeds gripper opening")
    safety = intent.get("safety", {})
    if safety.get("collision_policy") != (
        "combined_interarm_table_and_open_gripper"
    ):
        raise ValueError("pipeline nine requires all collision gates")
    refinement = safety.get("cartesian_refinement_policy", {})
    if refinement.get("force_continue_after_maximum_iterations") is not False:
        raise ValueError("pipeline nine must never force-continue refinement")
    if safety.get("automatic_resume_from_saved_grasp") is not False:
        raise ValueError("pipeline nine cannot automatically resume a grasp")


def select_arm(intent: Mapping[str, Any], side: str) -> dict[str, Any]:
    validate_sandwich_intent(intent)
    if side not in ("left", "right"):
        raise ValueError("selected sandwich arm must be left or right")
    plan = deepcopy(dict(intent))
    plan["schema"] = PLAN_SCHEMA
    plan["selected_side"] = side
    plan["arm_selection"] = "offline_ik_and_swept_collision_best_safe_candidate"
    for stage in plan["stages"]:
        stage["active_side"] = side
    validate_sandwich_plan(plan)
    return plan


def apply_grasp_orientation_branch(
    plan: Mapping[str, Any],
    *,
    source_yaw_offset_deg: float,
    target_yaw_offset_deg: float,
) -> dict[str, Any]:
    """Apply one sampled top-down grasp yaw branch to a selected plan."""
    validate_sandwich_plan(plan)
    branched = deepcopy(dict(plan))
    offsets = {
        **{
            stage_id: float(source_yaw_offset_deg)
            for stage_id in SOURCE_ORIENTATION_STAGES
        },
        **{
            stage_id: float(target_yaw_offset_deg)
            for stage_id in TARGET_ORIENTATION_STAGES
        },
    }
    for stage in branched["stages"]:
        offset_deg = offsets.get(stage["id"])
        if offset_deg is None:
            continue
        angle = np.deg2rad(offset_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        jaw_axis = _unit_xy(
            stage["jaw_closing_axis_base_xy"],
            field=f"{stage['id']} jaw axis",
        )
        rotated_axis = rotation @ jaw_axis
        stage["jaw_closing_axis_base_xy"] = rotated_axis.tolist()
        stage["target_quaternion_xyzw"] = _topdown_quaternion(
            rotated_axis
        ).tolist()
    branched["grasp_orientation_branch"] = {
        "method": "offline_reachable_topdown_yaw_sampling",
        "source_yaw_offset_deg": float(source_yaw_offset_deg),
        "target_yaw_offset_deg": float(target_yaw_offset_deg),
    }
    validate_sandwich_plan(branched)
    return branched


def validate_sandwich_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unexpected pipeline-nine sandwich plan schema")
    side = plan.get("selected_side")
    if side not in ("left", "right"):
        raise ValueError("sandwich plan must bind one selected arm")
    neutral = deepcopy(dict(plan))
    neutral["schema"] = INTENT_SCHEMA
    neutral.pop("selected_side", None)
    neutral["arm_selection"] = "automatic_during_offline_certification"
    for stage in neutral["stages"]:
        if stage.get("active_side") != side:
            raise ValueError("all sandwich stages must bind the selected arm")
        stage["active_side"] = "auto"
    validate_sandwich_intent(neutral)


def build_policy(plan: Mapping[str, Any], *, plan_sha256: str) -> dict[str, Any]:
    validate_sandwich_plan(plan)
    side = str(plan["selected_side"])
    operations: list[dict[str, Any]] = [
        {"call": "bind_plan", "sha256": plan_sha256},
        {"call": "open_gripper", "side": side},
    ]
    for stage_id in STAGE_ORDER:
        operations.append(
            {"call": "execute_single_arm_stage", "side": side, "stage_id": stage_id}
        )
        if stage_id == "grasp":
            operations.append({"call": "close_gripper", "side": side})
        if stage_id == "place_on_target":
            operations.append({"call": "open_gripper", "side": side})
    return {
        "schema": POLICY_SCHEMA,
        "pipeline": 9,
        "plan_sha256": plan_sha256,
        "operations": operations,
        "failure_policy": "hold_both_and_abort",
    }


def validate_policy(
    policy: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
) -> None:
    expected = build_policy(plan, plan_sha256=plan_sha256)
    if dict(policy) != expected:
        raise ValueError("pipeline-nine policy differs from deterministic operations")


def build_session_manifest(
    *,
    donut_phase: Mapping[str, Any],
    top_phase: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "pipeline": 9,
        "phase_order": ["donut_on_brown", "mixed_bread_on_stack"],
        "donut_on_brown": deepcopy(dict(donut_phase)),
        "mixed_bread_on_stack": (
            None if top_phase is None else deepcopy(dict(top_phase))
        ),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_session_manifest(manifest)
    return manifest


def validate_session_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("pipeline") != 9:
        raise ValueError("unexpected pipeline-nine session manifest")
    if manifest.get("phase_order") != [
        "donut_on_brown",
        "mixed_bread_on_stack",
    ]:
        raise ValueError("pipeline-nine phase order changed")
    expected = dict(manifest)
    observed_hash = expected.pop("manifest_sha256", None)
    if observed_hash != canonical_sha256(expected):
        raise ValueError("pipeline-nine session manifest hash mismatch")
