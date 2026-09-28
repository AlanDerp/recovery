#!/usr/bin/env python3
"""Preview or execute one certified R1 Lite sandwich stacking phase."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel  # noqa: E402
from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteCombinedCollisionKinematics,
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
    canonical_json_bytes,
    load_execution_bundle,
    sha256_file,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    FULL_URDF,
    LEFT_RELAXED_URDF,
    RELAXED_URDF,
)
from capx.integrations.r1lite.sandwich_executor import (  # noqa: E402
    SandwichExecutionRuntime,
    dispatch_policy,
)
from capx.integrations.r1lite.sandwich_perception import (  # noqa: E402
    OBJECT_ORDER,
    perceive_sandwich_scene_homography,
    require_repeatable_scene,
    segment_sandwich_objects,
    support_contains_center,
)
from capx.integrations.r1lite.sandwich_plan import (  # noqa: E402
    build_sandwich_intent,
    validate_policy,
    validate_sandwich_plan,
)


COLORS = {
    "brown_bread": (80, 180, 255),
    "donut": (255, 80, 80),
    "mixed_bread": (80, 255, 120),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "execute"], required=True)
    parser.add_argument(
        "--phase",
        choices=["donut_on_brown", "mixed_bread_on_stack"],
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("outputs/r1lite/head_plane_calibration.json"),
    )
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--output-intent", type=Path)
    parser.add_argument("--raw-image", type=Path)
    parser.add_argument("--preview-image", type=Path)
    parser.add_argument("--depth-output", type=Path)
    parser.add_argument("--mask-output-dir", type=Path)
    parser.add_argument("--approved-plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--approved-policy", type=Path)
    parser.add_argument("--expected-policy-sha256")
    parser.add_argument("--execution-bundle", type=Path)
    parser.add_argument("--expected-execution-bundle-sha256")
    parser.add_argument("--capability-certificate", type=Path)
    parser.add_argument("--expected-capability-certificate-sha256")
    parser.add_argument("--verify-after-execution", action="store_true")
    parser.add_argument("--verification-image", type=Path)
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _allowed_overlap_pairs(phase: str, *, final: bool = False) -> set[frozenset[str]]:
    if final:
        return {
            frozenset((first, second))
            for index, first in enumerate(OBJECT_ORDER)
            for second in OBJECT_ORDER[index + 1 :]
        }
    if phase == "mixed_bread_on_stack":
        return {frozenset(("brown_bread", "donut"))}
    return set()


def capture_scene(
    robot: R1LiteRealLowLevel,
    *,
    phase: str,
    calibration: dict[str, Any],
    newer_than_ns: int | None = None,
    final: bool = False,
    raw_debug_path: Path | None = None,
) -> tuple[np.ndarray, None, dict[str, dict[str, Any]], dict[str, Any]]:
    overlap = _allowed_overlap_pairs(phase, final=final)
    del newer_than_ns
    rgb = np.asarray(robot.get_head_rgb("left"), dtype=np.uint8)
    if raw_debug_path is not None:
        raw_debug_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(raw_debug_path)
    segmentations = segment_sandwich_objects(
        rgb,
        service_url=str(
            getattr(robot, "sam3_service_url", "http://127.0.0.1:8114")
        ),
        allowed_overlap_pairs=overlap,
    )
    if raw_debug_path is not None:
        debug_dir = raw_debug_path.parent / f"{raw_debug_path.stem}_debug_masks"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name, item in segmentations.items():
            Image.fromarray(
                np.asarray(item["mask"], dtype=np.uint8) * 255,
                mode="L",
            ).save(debug_dir / f"{name}.png")
    scene = perceive_sandwich_scene_homography(
        rgb,
        segmentations,
        calibration,
        phase=phase,
        allowed_overlap_pairs=overlap,
    )
    return rgb, None, segmentations, scene


def render_preview(
    rgb: np.ndarray,
    segmentations: dict[str, dict[str, Any]],
    scene: dict[str, Any],
    *,
    output: Path,
    title: str,
) -> None:
    image = np.asarray(rgb, dtype=np.uint8).copy()
    overlay = image.copy()
    for name in OBJECT_ORDER:
        overlay[np.asarray(segmentations[name]["mask"], dtype=bool)] = COLORS[name]
    blended = (0.70 * image + 0.30 * overlay).astype(np.uint8)
    canvas = Image.fromarray(blended, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((8, 8, 620, 42), fill=(0, 0, 0))
    draw.text((16, 16), title, fill=(255, 255, 255))
    for name in OBJECT_ORDER:
        geometry = scene["objects"][name]
        u, v = geometry["pixel_geometry"]["centroid_px"]
        color = COLORS[name]
        radius = 10
        draw.ellipse(
            (u - radius, v - radius, u + radius, v + radius),
            outline=color,
            width=4,
        )
        draw.text(
            (u + 12, v - 12),
            (
                f"{name} {geometry['width_m'] * 100:.1f}x"
                f"{geometry['length_m'] * 100:.1f}cm "
                f"h={geometry['height_above_table_m'] * 100:.1f}cm"
            ),
            fill=color,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_failure_preview(
    *,
    raw_image: Path,
    output: Path,
    error: str,
) -> None:
    image = np.asarray(Image.open(raw_image).convert("RGB"), dtype=np.uint8)
    overlay = image.copy()
    debug_dir = raw_image.parent / f"{raw_image.stem}_debug_masks"
    for name in OBJECT_ORDER:
        mask_path = debug_dir / f"{name}.png"
        if mask_path.is_file():
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            overlay[mask] = COLORS[name]
    blended = (0.70 * image + 0.30 * overlay).astype(np.uint8)
    canvas = Image.fromarray(blended, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((8, 8, canvas.width - 8, 72), fill=(120, 0, 0))
    draw.text((16, 16), "FAIL-CLOSED: no executable intent generated", fill="white")
    draw.text((16, 42), error[:110], fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _save_preview_artifacts(
    *,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    segmentations: dict[str, dict[str, Any]],
    scene: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.raw_image is not None:
        args.raw_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(args.raw_image)
    if args.depth_output is not None and depth is not None:
        args.depth_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.depth_output, depth)
    if args.mask_output_dir is not None:
        args.mask_output_dir.mkdir(parents=True, exist_ok=True)
        for name, item in segmentations.items():
            Image.fromarray(
                np.asarray(item["mask"], dtype=np.uint8) * 255,
                mode="L",
            ).save(args.mask_output_dir / f"{name}.png")
    if args.preview_image is not None:
        render_preview(
            rgb,
            segmentations,
            scene,
            output=args.preview_image,
            title=f"Pipeline 9: {args.phase}",
        )


def preview(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_intent is None:
        raise ValueError("preview requires --output-intent")
    calibration = json.loads(args.calibration.read_text())
    robot = load_camera_robot(args.config)
    try:
        first_rgb, _, _, first_scene = capture_scene(
            robot,
            phase=args.phase,
            calibration=calibration,
            raw_debug_path=args.raw_image,
        )
        time.sleep(0.5)
        second_rgb, depth, segmentations, second_scene = capture_scene(
            robot,
            phase=args.phase,
            calibration=calibration,
        )
    finally:
        robot.close()
    repeatability = require_repeatable_scene(first_scene, second_scene)
    phase_verification = None
    if args.phase == "mixed_bread_on_stack":
        phase_verification = support_contains_center(
            second_scene["objects"]["brown_bread"],
            second_scene["objects"]["donut"],
        )
        if not phase_verification["passed"]:
            raise RuntimeError(
                "phase-one visual verification failed; donut is not supported "
                "by the brown bread"
            )
    intent = build_sandwich_intent(
        second_scene,
        phase=args.phase,
        first_image_sha256=_sha256_bytes(first_rgb.tobytes()),
        second_image_sha256=_sha256_bytes(second_rgb.tobytes()),
        repeatability_m=float(repeatability["maximum_m"]),
    )
    args.output_intent.parent.mkdir(parents=True, exist_ok=True)
    args.output_intent.write_bytes(canonical_json_bytes(intent))
    _save_preview_artifacts(
        rgb=second_rgb,
        depth=depth,
        segmentations=segmentations,
        scene=second_scene,
        args=args,
    )
    return {
        "schema": "capx.r1lite.ninth_sandwich_preview_audit.v1",
        "pipeline": 9,
        "mode": "preview",
        "phase": args.phase,
        "passed": True,
        "intent_path": str(args.output_intent.resolve()),
        "intent_sha256": sha256_file(args.output_intent),
        "config_sha256": sha256_file(args.config),
        "calibration_sha256": sha256_file(args.calibration),
        "repeatability": repeatability,
        "previous_phase_verification": phase_verification,
        "robot_motion_commands_sent": False,
    }


def _require_hash(path: Path | None, expected: str | None, *, label: str) -> str:
    if path is None or expected is None:
        raise ValueError(f"execute requires {label} path and expected SHA-256")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def _load_approved_execution(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    plan_sha = _require_hash(
        args.approved_plan, args.expected_plan_sha256, label="plan"
    )
    policy_sha = _require_hash(
        args.approved_policy, args.expected_policy_sha256, label="policy"
    )
    bundle_sha = _require_hash(
        args.execution_bundle,
        args.expected_execution_bundle_sha256,
        label="execution bundle",
    )
    certificate_sha = _require_hash(
        args.capability_certificate,
        args.expected_capability_certificate_sha256,
        label="capability certificate",
    )
    assert args.approved_plan is not None
    assert args.approved_policy is not None
    assert args.execution_bundle is not None
    assert args.capability_certificate is not None
    plan = json.loads(args.approved_plan.read_text())
    validate_sandwich_plan(plan)
    if plan["phase"] != args.phase:
        raise ValueError("approved plan phase differs from requested execution")
    policy = json.loads(args.approved_policy.read_text())
    validate_policy(policy, plan=plan, plan_sha256=plan_sha)
    bundle = load_execution_bundle(
        args.execution_bundle,
        expected_file_sha256=bundle_sha,
        plan_sha256=plan_sha,
        config_sha256=sha256_file(args.config),
        full_urdf_sha256=sha256_file(FULL_URDF),
        left_relaxed_urdf_sha256=sha256_file(LEFT_RELAXED_URDF),
        right_relaxed_urdf_sha256=sha256_file(RELAXED_URDF),
    )
    load_capability_certificate(
        args.capability_certificate,
        expected_sha256=certificate_sha,
        plan_sha256=plan_sha,
        code_sha256=policy_sha,
        execution_bundle_sha256=bundle_sha,
    )
    return plan, policy, bundle, {
        "plan_sha256": plan_sha,
        "policy_sha256": policy_sha,
        "execution_bundle_sha256": bundle_sha,
        "capability_certificate_sha256": certificate_sha,
    }


def _final_visual_verification(
    args: argparse.Namespace,
) -> dict[str, Any]:
    robot = load_camera_robot(args.config)
    calibration = json.loads(args.calibration.read_text())
    try:
        rgb, _, segmentations, scene = capture_scene(
            robot,
            phase=args.phase,
            calibration=calibration,
            final=True,
        )
    finally:
        robot.close()
    result = support_contains_center(
        scene["objects"]["brown_bread"],
        scene["objects"]["mixed_bread"],
    )
    if args.verification_image is not None:
        render_preview(
            rgb,
            segmentations,
            scene,
            output=args.verification_image,
            title="Pipeline 9: final stack verification",
        )
    if not result["passed"]:
        raise RuntimeError("final mixed-bread stack verification failed")
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan, policy, bundle, hashes = _load_approved_execution(args)
    audit: dict[str, Any] = {
        "schema": "capx.r1lite.ninth_sandwich_execution_audit.v1",
        "pipeline": 9,
        "mode": "execute",
        "phase": args.phase,
        "passed": False,
        "home_completed": False,
        "robot_motion_commands_sent": False,
        **hashes,
        "events": [],
    }
    controller = R1LiteBimanualRuntime(
        config=BimanualRuntimeConfig(arrival_timeout_s=5.0)
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
        if maximum_initial_error > 0.01:
            raise RuntimeError(
                "online initial state differs from certified bundle: "
                f"{maximum_initial_error:.6f}rad"
            )
        solver = IsolatedBimanualRelaxedIK(
            controller,
            bundle["initial_state"]["torso_joints_rad"],
        )
        side = str(plan["selected_side"])
        poses = controller.get_tool_pose_pair()
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
        if float(np.max(np.abs(probe - np.asarray(current[side])))) > 0.05:
            raise RuntimeError(f"{side} isolated IK roundtrip preflight failed")
        safety = R1LiteCombinedCollisionKinematics(
            table_z_m=float(plan["safety"]["table_z_m"]),
            tcp_to_fingertip_m=float(
                plan["safety"]["tcp_to_fingertip_m"]
            ),
            minimum_link_clearance_m=float(
                plan["safety"]["minimum_link_clearance_m"]
            ),
            minimum_table_clearance_m=float(
                plan["safety"]["minimum_table_clearance_m"]
            ),
            minimum_gripper_envelope_separation_m=float(
                plan["safety"]["minimum_gripper_envelope_separation_m"]
            ),
        )
        runtime = BimanualTowelExecutor(
            plan=plan,
            plan_sha256=hashes["plan_sha256"],
            bundle=bundle,
            controller=controller,
            solver=solver,
            safety=safety,
        )
        wrapper = SandwichExecutionRuntime(runtime, plan)
        audit["program_trace"] = dispatch_policy(
            policy,
            plan=plan,
            plan_sha256=hashes["plan_sha256"],
            runtime=wrapper,
        )
        audit["events"] = runtime.trace
        audit["home_completed"] = any(
            event.get("event") == "home_completed" for event in runtime.trace
        )
        if not audit["home_completed"]:
            raise RuntimeError("pipeline-nine execution did not complete home")
        audit["passed"] = True
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
        audit["robot_motion_commands_sent"] = controller.command_pair_count > 0
        controller.close()
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.verify_after_execution:
        try:
            audit["final_visual_verification"] = _final_visual_verification(args)
        except Exception as exc:
            audit["passed"] = False
            audit["final_visual_verification_error"] = repr(exc)
            args.audit_output.write_text(
                json.dumps(audit, indent=2, ensure_ascii=False)
            )
            raise
        args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


def main() -> None:
    args = parse_args()
    try:
        audit = preview(args) if args.mode == "preview" else execute(args)
    except Exception as exc:
        if args.mode == "preview":
            failure = {
                "schema": "capx.r1lite.ninth_sandwich_preview_audit.v1",
                "pipeline": 9,
                "mode": "preview",
                "phase": args.phase,
                "passed": False,
                "robot_motion_commands_sent": False,
                "error": repr(exc),
            }
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            args.audit_output.write_text(
                json.dumps(failure, indent=2, ensure_ascii=False)
            )
            if (
                args.raw_image is not None
                and args.raw_image.is_file()
                and args.preview_image is not None
            ):
                render_failure_preview(
                    raw_image=args.raw_image,
                    output=args.preview_image,
                    error=str(exc),
                )
        raise
    if args.mode == "preview":
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
