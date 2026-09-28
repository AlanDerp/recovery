#!/usr/bin/env python3
"""Preview and audit the seventh R1 Lite dual-arm towel-fold pipeline.

This first implementation is deliberately preview-only. Execute mode validates
the approved artifacts and then fails closed until the dual-arm execution stack
has an explicit capability certificate.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import io
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel  # noqa: E402
from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteBimanualKinematics,
    R1LiteTableCollisionKinematics,
)
from capx.integrations.r1lite.bimanual_relaxed_ik import (  # noqa: E402
    IsolatedBimanualRelaxedIK,
)
from capx.integrations.r1lite.bimanual_runtime import (  # noqa: E402
    BimanualRuntimeConfig,
    R1LiteBimanualRuntime,
)
from capx.integrations.r1lite.bimanual_towel_executor import (  # noqa: E402
    BimanualTowelExecutor,
)
from capx.integrations.r1lite.capability_certificate import (  # noqa: E402
    load_capability_certificate,
)
from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    load_execution_bundle,
    sha256_file,
)
from capx.integrations.r1lite.generated_bimanual_policy import (  # noqa: E402
    dispatch_program,
    execute_dry_run,
    expected_operations,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    FULL_URDF,
    LEFT_RELAXED_URDF,
    RELAXED_URDF,
)
from capx.integrations.r1lite.towel_perception import (  # noqa: E402
    calibration_polygon,
    extract_towel_geometry,
    keypoint_repeatability_error_m,
)
from capx.integrations.r1lite.towel_plan import (  # noqa: E402
    SEQUENTIAL_SINGLE_ARM_SCHEMA,
    build_sequential_single_arm_plan,
    validate_towel_fold_plan,
)


MAXIMUM_KEYPOINT_EXTRAPOLATION_M = 0.15
MAXIMUM_TWO_FRAME_KEYPOINT_DRIFT_M = 0.015
# Keep the handoff perception identical to pipeline eight.  Requiring the
# weaker "cloth towel" prompt as a second hard gate caused fresh pipeline-seven
# handoffs to fail even when the approved "towel" mask was confident.
SAM3_PROMPTS = ("towel",)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_camera_robot(config_path: Path) -> R1LiteRealLowLevel:
    cfg = yaml.safe_load(config_path.read_text())
    low_cfg = dict(cfg["env"]["cfg"]["low_level"])
    low_cfg.pop("_target_", None)
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {
        name: value
        for name, value in low_cfg.items()
        if name in signature.parameters
    }
    kwargs["wrist_camera_enabled"] = False
    return R1LiteRealLowLevel(**kwargs)


def encode_png_base64(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        buffer,
        format="PNG",
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def segment_towel(
    rgb: np.ndarray,
    *,
    service_url: str,
    timeout_s: float = 60.0,
    minimum_score: float = 0.15,
    prompts: tuple[str, ...] = SAM3_PROMPTS,
) -> dict[str, Any]:
    """Select a towel mask using one or more explicitly approved prompts."""
    if not prompts:
        raise ValueError("at least one SAM3 prompt is required")
    best_by_prompt: dict[str, dict[str, Any]] = {}
    session = requests.Session()
    session.trust_env = False
    image_base64 = encode_png_base64(rgb)
    for prompt in prompts:
        response = session.post(
            f"{service_url.rstrip('/')}/segment",
            json={"image_base64": image_base64, "text_prompt": prompt},
            timeout=timeout_s,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            raise RuntimeError(f"SAM3 returned no towel candidate for {prompt!r}")
        best_by_prompt[prompt] = max(
            (dict(item) for item in results),
            key=lambda item: float(item.get("score", 0.0)),
        )
    decoded = {}
    for prompt, item in best_by_prompt.items():
        score = float(item.get("score", 0.0))
        if score < minimum_score:
            raise RuntimeError(
                f"SAM3 {prompt!r} score below threshold: "
                f"{score:.3f} < {minimum_score:.3f}"
            )
        height, width = [int(value) for value in item["shape"]]
        decoded[prompt] = np.frombuffer(
            base64.b64decode(item["mask_base64"]),
            dtype=np.uint8,
        ).reshape(height, width).astype(bool)
    if len(prompts) == 1:
        prompt_iou = 1.0
    else:
        prompt_masks = [decoded[prompt] for prompt in prompts]
        pair_ious = []
        for index, first_mask in enumerate(prompt_masks):
            for second_mask in prompt_masks[index + 1 :]:
                intersection = int(np.count_nonzero(first_mask & second_mask))
                union = int(np.count_nonzero(first_mask | second_mask))
                pair_ious.append(0.0 if union == 0 else intersection / union)
        prompt_iou = min(pair_ious)
        if prompt_iou < 0.70:
            raise RuntimeError(
                "SAM3 towel prompts disagree: "
                f"minimum mask IoU={prompt_iou:.3f} < 0.700"
            )
    best_prompt, best = max(
        best_by_prompt.items(),
        key=lambda item: float(item[1].get("score", 0.0)),
    )
    score = float(best.get("score", 0.0))
    mask = decoded[best_prompt]
    if mask.shape != rgb.shape[:2]:
        raise RuntimeError(
            f"SAM3 towel mask/image shape mismatch: {mask.shape} != {rgb.shape[:2]}"
        )
    return {
        "mask": mask,
        "score": score,
        "prompt": best_prompt,
        "prompt_mask_iou": prompt_iou,
        "box": best.get("box"),
        "candidate_count": len(best_by_prompt),
        "selection_policy": (
            "single_approved_prompt" if len(prompts) == 1 else "multi_prompt_consensus"
        ),
    }


def render_preview(
    rgb: np.ndarray,
    mask: np.ndarray,
    geometry: dict[str, Any],
    calibration: dict[str, Any],
    *,
    output_path: Path,
    mask_path: Path,
) -> None:
    image_array = np.asarray(rgb, dtype=np.uint8).copy()
    overlay = image_array.copy()
    overlay[np.asarray(mask, dtype=bool)] = np.array([255, 80, 80], dtype=np.uint8)
    blended = (0.65 * image_array + 0.35 * overlay).astype(np.uint8)
    image = Image.fromarray(blended, mode="RGB")
    draw = ImageDraw.Draw(image)
    polygon = calibration_polygon(calibration)
    draw.line(
        [tuple(point) for point in polygon] + [tuple(polygon[0])],
        fill=(0, 220, 255),
        width=4,
    )
    corners = [tuple(point) for point in geometry["corners_pixel_uv"]]
    draw.line(corners + [corners[0]], fill=(255, 220, 0), width=5)
    colors = {
        "center": (255, 255, 255),
        "left_edge_midpoint": (0, 255, 0),
        "right_edge_midpoint": (80, 120, 255),
    }
    for name, item in geometry["keypoints"].items():
        u, v = item["pixel_uv"]
        color = colors[name]
        radius = 10
        draw.ellipse(
            [u - radius, v - radius, u + radius, v + radius],
            outline=color,
            width=5,
        )
        draw.text(
            (u + 12, v - 12),
            f"{name} ext={item['extrapolation_m'] * 100:.1f}cm",
            fill=color,
        )
    for candidate in geometry.get("right_side_grasp_candidates", []):
        u, v = candidate["pixel_uv"]
        radius = 5
        draw.ellipse(
            [u - radius, v - radius, u + radius, v + radius],
            fill=(255, 0, 255),
        )
        draw.text((u + 6, v + 4), candidate["id"], fill=(255, 0, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(mask_path)


def capture_geometry(
    robot: R1LiteRealLowLevel,
    calibration: dict[str, Any],
    *,
    raw_path: Path | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    rgb = robot.get_head_rgb("left")
    expected_size = list(calibration.get("image_size", []))
    if expected_size != [rgb.shape[1], rgb.shape[0]]:
        raise RuntimeError(
            "head image does not match homography resolution: "
            f"image={[rgb.shape[1], rgb.shape[0]]}, calibration={expected_size}"
        )
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(raw_path)
    segmentation = segment_towel(
        rgb,
        service_url=str(getattr(robot, "sam3_service_url", "http://127.0.0.1:8114")),
    )
    geometry = extract_towel_geometry(
        segmentation["mask"],
        calibration,
        maximum_extrapolation_m=MAXIMUM_KEYPOINT_EXTRAPOLATION_M,
    )
    return rgb, segmentation, geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "execute"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--raw-image", type=Path)
    parser.add_argument("--preview-image", type=Path)
    parser.add_argument("--mask-image", type=Path)
    parser.add_argument("--approved-plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--generated-code", type=Path)
    parser.add_argument("--expected-code-sha256")
    parser.add_argument("--capability-certificate", type=Path)
    parser.add_argument("--expected-capability-certificate-sha256")
    parser.add_argument("--execution-bundle", type=Path)
    parser.add_argument("--expected-execution-bundle-sha256")
    parser.add_argument("--sequential-side", choices=["left", "right"])
    parser.add_argument("--original-center-json")
    return parser.parse_args()


def validate_approved_artifacts(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    if (
        args.approved_plan is None
        or not args.expected_plan_sha256
        or args.generated_code is None
        or not args.expected_code_sha256
    ):
        raise ValueError(
            "execute requires approved plan/code paths and both expected SHA-256 values"
        )
    plan_bytes = args.approved_plan.read_bytes()
    code_bytes = args.generated_code.read_bytes()
    if sha256_bytes(plan_bytes) != args.expected_plan_sha256:
        raise RuntimeError("approved seventh-pipeline plan SHA-256 mismatch")
    if sha256_bytes(code_bytes) != args.expected_code_sha256:
        raise RuntimeError("approved seventh-pipeline code SHA-256 mismatch")
    plan = json.loads(plan_bytes)
    validate_towel_fold_plan(plan)
    if (
        plan.get("schema") != SEQUENTIAL_SINGLE_ARM_SCHEMA
        or plan.get("phase_side") != "right"
    ):
        raise ValueError("pipeline 7 executes right-arm-only plans")
    expected = expected_operations(plan, plan_sha256=args.expected_plan_sha256)
    execute_dry_run(code_bytes.decode(), expected=expected)
    return plan, code_bytes.decode(), expected


def execute_approved_bimanual_program(
    args: argparse.Namespace,
    *,
    plan: dict[str, Any],
    code: str,
    expected: list[dict[str, Any]],
) -> dict[str, Any]:
    required = (
        args.config,
        args.execution_bundle,
        args.expected_execution_bundle_sha256,
        args.capability_certificate,
        args.expected_capability_certificate_sha256,
    )
    if any(value is None for value in required):
        raise ValueError(
            "execute requires config, execution bundle/hash and capability "
            "certificate/hash"
        )
    assert args.config is not None
    assert args.execution_bundle is not None
    assert args.expected_execution_bundle_sha256 is not None
    assert args.capability_certificate is not None
    assert args.expected_capability_certificate_sha256 is not None
    if plan.get("schema") == SEQUENTIAL_SINGLE_ARM_SCHEMA:
        safety = R1LiteTableCollisionKinematics(
            active_side=str(plan["phase_side"]),
            table_z_m=float(plan["safety"]["table_z_m"]),
            tcp_to_fingertip_m=float(
                plan["safety"]["tcp_to_fingertip_m"]
            ),
            distal_capsule_radius_m=float(
                plan["safety"]["distal_link_capsule_radius_m"]
            ),
            minimum_table_clearance_m=float(
                plan["safety"]["minimum_table_clearance_m"]
            ),
        )
        collision_model = {
            "name": (
                "active_arm_distal_links_and_two_open_finger_mesh_aabbs"
                "_vs_table_plane"
            ),
            "active_side": plan["phase_side"],
            "table_z_m": safety.table_z_m,
            "tcp_to_fingertip_m": safety.tcp_to_fingertip_m,
            "link_capsule_radius_m": safety.link_capsule_radius_m,
            "gripper_finger_open_travel_m": safety.FINGER_MAX_TRAVEL_M,
            "gripper_finger_joint_origins_m": [
                value.tolist() for value in safety.FINGER_JOINT_ORIGINS_M
            ],
            "gripper_finger_mesh_bounds_m": [
                [lower.tolist(), upper.tolist()]
                for lower, upper in safety.FINGER_MESH_BOUNDS_M
            ],
            "minimum_link_clearance_m": safety.minimum_link_clearance_m,
            "minimum_tcp_separation_m": 0.0,
            "sweep_interpolation": "linear_joint_space",
            "sweep_samples_per_segment": 50,
        }
    else:
        safety = R1LiteBimanualKinematics(
            minimum_tcp_separation_m=0.12552056969962836
        )
        collision_model = {
            "name": "arm_link_capsules_plus_gripper_TCP_mesh_envelope",
            "link_capsule_radius_m": safety.link_capsule_radius_m,
            "minimum_link_clearance_m": safety.minimum_link_clearance_m,
            "minimum_tcp_separation_m": safety.minimum_tcp_separation_m,
            "sweep_interpolation": "linear_joint_space",
            "sweep_samples_per_segment": 50,
        }
    bundle = load_execution_bundle(
        args.execution_bundle,
        expected_file_sha256=args.expected_execution_bundle_sha256,
        plan_sha256=args.expected_plan_sha256,
        config_sha256=sha256_file(args.config),
        full_urdf_sha256=sha256_file(FULL_URDF),
        left_relaxed_urdf_sha256=sha256_file(LEFT_RELAXED_URDF),
        right_relaxed_urdf_sha256=sha256_file(RELAXED_URDF),
        tcp_definition=plan["tcp"],
        collision_model=collision_model,
    )
    bundle_file_sha = sha256_file(args.execution_bundle)
    load_capability_certificate(
        args.capability_certificate,
        expected_sha256=args.expected_capability_certificate_sha256,
        plan_sha256=args.expected_plan_sha256,
        code_sha256=args.expected_code_sha256,
        execution_bundle_sha256=bundle_file_sha,
    )
    audit: dict[str, Any] = {
        "pipeline": 7,
        "mode": "execute",
        "plan_sha256": args.expected_plan_sha256,
        "code_sha256": args.expected_code_sha256,
        "execution_bundle_sha256": bundle_file_sha,
        "robot_motion_commands_sent": False,
        "passed": False,
        "events": [],
    }
    arrival_hard_threshold = float(
        os.environ.get("R1LITE_ARRIVAL_HARD_THRESHOLD_RAD", "0.05")
    )
    arrival_soft_threshold = float(
        os.environ.get("R1LITE_ARRIVAL_SOFT_THRESHOLD_RAD", "0.01")
    )
    initial_state_tolerance = float(
        os.environ.get("R1LITE_INITIAL_STATE_TOLERANCE_RAD", "0.01")
    )
    audit["joint_arrival_hard_gate"] = {
        "enabled": bool(np.isfinite(arrival_hard_threshold)),
        "threshold_rad": (
            arrival_hard_threshold
            if np.isfinite(arrival_hard_threshold)
            else None
        ),
    }
    audit["joint_arrival_soft_gate"] = {
        "enabled": bool(np.isfinite(arrival_soft_threshold)),
        "threshold_rad": (
            arrival_soft_threshold
            if np.isfinite(arrival_soft_threshold)
            else None
        ),
    }
    # A direct (progress_steps=1) waypoint can legitimately need more than the
    # one-second runtime default. Give the physical arm enough time to settle;
    # individual launchers may explicitly disable only the hard-error gate.
    controller = R1LiteBimanualRuntime(
        config=BimanualRuntimeConfig(
            arrival_timeout_s=5.0,
            arrival_soft_threshold_rad=arrival_soft_threshold,
            arrival_velocity_threshold_rad_s=(
                0.1 if not np.isfinite(arrival_soft_threshold) else None
            ),
            arrival_settle_time_s=(
                0.25 if not np.isfinite(arrival_soft_threshold) else 0.0
            ),
            arrival_hard_threshold_rad=arrival_hard_threshold,
        )
    )
    runtime: BimanualTowelExecutor | None = None
    try:
        controller.start()
        controller.wait_until_ready(require_tool_pose=True)
        current = controller.get_feedback_pair()
        maximum_initial_error = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(current[side], dtype=float)
                        - np.asarray(
                            bundle["initial_state"][f"{side}_joints_rad"],
                            dtype=float,
                        )
                    )
                )
            )
            for side in ("left", "right")
        )
        initial_tolerance = initial_state_tolerance
        prealign_initial_state = (
            os.environ.get("R1LITE_PREALIGN_INITIAL_STATE", "0") == "1"
        )
        prealign_safety_limit = min(arrival_hard_threshold, 0.05)
        if (
            maximum_initial_error > initial_tolerance
            and prealign_initial_state
            and maximum_initial_error <= prealign_safety_limit
        ):
            motion_config = controller.config
            controller.config = replace(
                motion_config,
                arrival_soft_threshold_rad=initial_tolerance,
            )
            try:
                controller.execute_joint_motion(
                    np.asarray(
                        bundle["initial_state"]["left_joints_rad"], dtype=float
                    ),
                    np.asarray(
                        bundle["initial_state"]["right_joints_rad"], dtype=float
                    ),
                    progress_steps=30,
                )
            finally:
                controller.config = motion_config
            audit["robot_motion_commands_sent"] = True
            current = controller.get_feedback_pair()
            maximum_initial_error = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(current[side], dtype=float)
                            - np.asarray(
                                bundle["initial_state"][
                                    f"{side}_joints_rad"
                                ],
                                dtype=float,
                            )
                        )
                    )
                )
                for side in ("left", "right")
            )
            audit["events"].append(
                {
                    "event": "initial_state_prealigned",
                    "maximum_error_rad": maximum_initial_error,
                }
            )
        if maximum_initial_error > initial_tolerance:
            raise RuntimeError(
                "online dual-arm initial state differs from execution bundle: "
                f"{maximum_initial_error:.6f}rad > "
                f"{initial_tolerance:.6f}rad"
            )
        poses = controller.get_tool_pose_pair()
        solver = IsolatedBimanualRelaxedIK(
            controller,
            bundle["initial_state"]["torso_joints_rad"],
        )
        side = plan["phase_side"]
        if side != "right":
            raise RuntimeError("pipeline 7 is right-arm-only")
        probe = np.asarray(
            solver.solve_candidates(
                side=side,
                target_position_base_m=poses[side][0],
                target_quaternion_xyzw=poses[side][1],
                reference_joints_rad=current[side],
                probes=1,
            )[0],
            dtype=float,
        )
        roundtrip_delta = float(
            np.max(np.abs(probe - np.asarray(current[side], dtype=float)))
        )
        if roundtrip_delta > 0.05:
            raise RuntimeError(
                f"{side} isolated IK roundtrip preflight failed: "
                f"{roundtrip_delta:.6f}rad"
            )
        runtime = BimanualTowelExecutor(
            plan=plan,
            plan_sha256=args.expected_plan_sha256,
            bundle=bundle,
            controller=controller,
            solver=solver,
            safety=safety,
        )
        trace = dispatch_program(code, expected=expected, runtime=runtime)
        audit["events"] = runtime.trace
        audit["program_trace"] = trace
        audit["passed"] = True
        return audit
    except Exception as exc:
        audit["error"] = repr(exc)
        if runtime is not None:
            audit["events"] = runtime.trace
        try:
            controller.hold()
            audit["common_hold_attempted"] = True
        except Exception as hold_exc:
            audit["common_hold_error"] = repr(hold_exc)
        raise
    finally:
        controller.close()
        audit["robot_motion_commands_sent"] = controller.command_pair_count > 0
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    calibration_bytes = args.calibration.read_bytes()
    calibration = json.loads(calibration_bytes)
    calibration_sha256 = sha256_bytes(calibration_bytes)
    if args.mode == "execute":
        plan, code, expected = validate_approved_artifacts(args)
        audit = execute_approved_bimanual_program(
            args,
            plan=plan,
            code=code,
            expected=expected,
        )
        print(json.dumps(audit, ensure_ascii=False))
        return

    if (
        args.config is None
        or args.output_plan is None
        or args.preview_image is None
        or args.mask_image is None
    ):
        raise ValueError(
            "preview requires config, output plan, preview image and mask image"
        )
    if args.sequential_side != "right":
        raise ValueError("pipeline 7 preview requires --sequential-side right")
    robot = load_camera_robot(args.config)
    try:
        first_rgb, first_segmentation, first_geometry = capture_geometry(
            robot,
            calibration,
            raw_path=args.raw_image,
        )
        time.sleep(0.5)
        second_rgb, second_segmentation, second_geometry = capture_geometry(
            robot,
            calibration,
            raw_path=None,
        )
    finally:
        robot.close()
    repeatability = keypoint_repeatability_error_m(
        first_geometry,
        second_geometry,
    )
    if repeatability > MAXIMUM_TWO_FRAME_KEYPOINT_DRIFT_M:
        raise RuntimeError(
            "two-frame towel keypoint repeatability gate failed: "
            f"{repeatability:.6f}m > {MAXIMUM_TWO_FRAME_KEYPOINT_DRIFT_M:.6f}m"
        )
    first_candidates = first_geometry["right_side_grasp_candidates"]
    stable_candidates = []
    for candidate in second_geometry["right_side_grasp_candidates"]:
        point = np.asarray(candidate["base_xyz_m"], dtype=float)
        nearest = min(
            first_candidates,
            key=lambda item: float(
                np.linalg.norm(
                    np.asarray(item["base_xyz_m"], dtype=float) - point
                )
            ),
        )
        drift = float(
            np.linalg.norm(np.asarray(nearest["base_xyz_m"], dtype=float) - point)
        )
        if drift <= MAXIMUM_TWO_FRAME_KEYPOINT_DRIFT_M:
            stable_candidates.append(
                {
                    **candidate,
                    "two_frame_repeatability_m": drift,
                    "matched_first_candidate_id": nearest["id"],
                }
            )
    if not stable_candidates:
        raise RuntimeError("no stable right-side contour grasp candidate")
    second_geometry["right_side_grasp_candidates"] = stable_candidates
    if args.original_center_json is None:
        original_center = second_geometry["keypoints"]["center"]["base_xyz_m"]
    else:
        original_center = json.loads(args.original_center_json)
    plan = build_sequential_single_arm_plan(
        second_geometry,
        side="right",
        original_center_base_m=original_center,
        calibration_sha256=calibration_sha256,
        first_image_sha256=sha256_bytes(first_rgb.tobytes()),
        second_image_sha256=sha256_bytes(second_rgb.tobytes()),
        keypoint_repeatability_m=repeatability,
        table_z_m=float(calibration["plane_z"]),
    )
    plan["perception"]["sam3"] = {
        "first_prompt": first_segmentation["prompt"],
        "first_score": first_segmentation["score"],
        "second_prompt": second_segmentation["prompt"],
        "second_score": second_segmentation["score"],
    }
    plan["execution_readiness"] = {
        "ready": False,
        "reason": "right-arm IK and swept collision execution are not yet certified",
        "left_tcp_model_accepted": True,
        "robot_motion_commands_sent": False,
    }
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    plan_sha256 = sha256_bytes(args.output_plan.read_bytes())
    render_preview(
        second_rgb,
        second_segmentation["mask"],
        second_geometry,
        calibration,
        output_path=args.preview_image,
        mask_path=args.mask_image,
    )
    audit = {
        "pipeline": 7,
        "mode": "preview",
        "plan_path": str(args.output_plan.resolve()),
        "plan_sha256": plan_sha256,
        "calibration_sha256": calibration_sha256,
        "maximum_keypoint_extrapolation_m": MAXIMUM_KEYPOINT_EXTRAPOLATION_M,
        "keypoint_repeatability_m": repeatability,
        "phase_side": args.sequential_side,
        "original_center_base_m": plan.get("original_center_base_m"),
        "execution_ready": False,
        "robot_motion_commands_sent": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
