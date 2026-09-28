#!/usr/bin/env python3
"""Pre-grasp depth alignment check for Agilex top-camera + optional USB side camera.

Uses the main RGB-D camera (camera_h) to estimate object/grasp height relative to the
fitted table plane, then optionally cross-checks with a side USB camera (UGREEN).

Side camera mount (tennis setup): lens ~20 cm above table surface.
Record in config as side_camera_height_above_table_m when available.

Typical usage (tennis pipeline):
  uv run --no-sync --active scripts/verify_agilex_pregrasp_depth.py \\
    --config env_configs/agilex/agilex_real_tennis_ball_in_box_top_camera_single_turn.yaml
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image as PILImage, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from refine_agilex_camera_pose_with_table_plane import (  # noqa: E402
    _depth_to_base_points,
    _fit_plane_svd,
    _plane_tilt_deg,
    _select_table_candidates,
)
from visualize_agilex_camera_calibration import (  # noqa: E402
    _load_low_level_config,
    _load_T_base_camera,
    _normalize_rgb,
    _project_base_to_pixel,
    _wait_for_rgbd,
)

DEFAULT_OUTPUT = Path("outputs/agilex_pregrasp_depth_check")
DEFAULT_SAM3_URL = "http://127.0.0.1:8114"
TENNIS_BALL_RADIUS_M = 0.033


def _encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    PILImage.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _sam3_segment(rgb: np.ndarray, prompt: str, sam3_url: str) -> list[dict[str, Any]]:
    payload = {"image_base64": _encode_png(rgb), "text_prompt": prompt}
    session = requests.Session()
    session.trust_env = False
    resp = session.post(f"{sam3_url.rstrip('/')}/segment", json=payload, timeout=120.0)
    resp.raise_for_status()
    results: list[dict[str, Any]] = []
    for item in resp.json().get("results", []):
        raw = base64.b64decode(item["mask_base64"])
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(item["shape"])).astype(bool)
        results.append(
            {
                "mask": mask,
                "box": item["box"],
                "score": float(item["score"]),
                "label": item.get("label", prompt),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty mask")
    return float(np.mean(xs)), float(np.mean(ys))


def _resize_mask_to_depth(mask: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == depth_shape:
        return mask
    from PIL import Image

    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    resized = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (depth_shape[1], depth_shape[0]), resample=nearest
        ),
        dtype=np.uint8,
    )
    return resized > 0


def _mask_to_world_points(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
    *,
    depth_clip: tuple[float, float],
) -> np.ndarray:
    mask = _resize_mask_to_depth(mask, depth.shape[:2])
    near, far = depth_clip
    valid = mask & np.isfinite(depth) & (depth >= near) & (depth <= far)
    ys, xs = np.nonzero(valid)
    if len(xs) < 20:
        raise RuntimeError(f"Too few valid depth points in mask ({len(xs)})")
    z = depth[ys, xs]
    x = (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_cam = np.stack([x, y, z], axis=1)
    homo = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    points_world = (pose_mat @ homo.T).T[:, :3]
    return points_world[np.isfinite(points_world).all(axis=1)]


def _estimate_ball_center(
    object_points: np.ndarray,
    *,
    z_percentile: float,
) -> tuple[np.ndarray, float, float]:
    """XY from median; Z from upper percentile to avoid table bleed in mask."""
    xy = np.median(object_points[:, :2], axis=0)
    z_median = float(np.median(object_points[:, 2]))
    z_top = float(np.percentile(object_points[:, 2], z_percentile))
    center = np.array([xy[0], xy[1], z_top], dtype=np.float64)
    return center, z_median, z_top


def _table_z_at_xy(normal: np.ndarray, offset: float, xy: np.ndarray) -> float:
    n = normal / max(np.linalg.norm(normal), 1e-12)
    x, y = float(xy[0]), float(xy[1])
    if abs(n[2]) < 1e-6:
        raise RuntimeError("Table plane is vertical; cannot solve Z")
    return float(-(n[0] * x + n[1] * y + offset) / n[2])
    n = normal / max(np.linalg.norm(normal), 1e-12)
    x, y = float(xy[0]), float(xy[1])
    if abs(n[2]) < 1e-6:
        raise RuntimeError("Table plane is vertical; cannot solve Z")
    return float(-(n[0] * x + n[1] * y + offset) / n[2])


def _wait_for_side_rgb(topic: str, timeout_s: float) -> np.ndarray:
    try:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError("rospy and cv_bridge are required for side camera") from exc

    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_pregrasp_depth_check", anonymous=True, disable_signals=True)
    bridge = CvBridge()
    print(f"[ros] waiting side RGB: {topic}")
    msg = rospy.wait_for_message(topic, Image, timeout=timeout_s)
    return _normalize_rgb(bridge.imgmsg_to_cv2(msg, "passthrough"), getattr(msg, "encoding", None))


def _draw_marker(
    image: np.ndarray,
    pixel: tuple[float, float],
    *,
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    out = image.copy()
    draw = ImageDraw.Draw(PILImage.fromarray(out))
    u, v = pixel
    r = 10
    draw.ellipse((u - r, v - r, u + r, v + r), outline=color, width=3)
    draw.line((u - 14, v, u + 14, v), fill=color, width=2)
    draw.line((u, v - 14, u, v + 14), fill=color, width=2)
    draw.text((u + 12, v + 12), label, fill=color)
    return np.asarray(PILImage.fromarray(out))


def _annotate_and_save(
    path: Path,
    image: np.ndarray,
    *,
    title: str,
    lines: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = PILImage.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    y = 8
    draw.text((8, y), title, fill=(255, 255, 0))
    y += 18
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255))
        y += 16
    canvas.save(path)


def run_check(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    low_level = _load_low_level_config(config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T_main = _load_T_base_camera(config_path=config_path, calibration_file=None)
    rgb, depth, intrinsics = _wait_for_rgbd(low_level, args.timeout)
    depth_clip = tuple(float(v) for v in low_level.get("object_depth_clip_range", [0.05, 2.0]))

    print(f"[sam3] segment prompt={args.object_prompt!r}")
    results = _sam3_segment(rgb, args.object_prompt, args.sam3_url)
    if not results:
        raise RuntimeError(f"SAM3 found no mask for {args.object_prompt!r}")
    best = results[0]
    mask = np.asarray(best["mask"], dtype=bool)
    object_points = _mask_to_world_points(mask, depth, intrinsics, T_main, depth_clip=depth_clip)
    object_pos, ball_z_median, ball_z_top = _estimate_ball_center(
        object_points,
        z_percentile=float(args.ball_z_percentile),
    )
    world_offset = np.asarray(low_level.get("object_world_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    object_pos = object_pos + world_offset

    grasp_z_offset = float(low_level.get("grasp_z_offset", 0.04))
    grasp_pos = object_pos.copy()
    grasp_pos[2] += grasp_z_offset

    scene_points = _depth_to_base_points(
        depth,
        intrinsics,
        T_main,
        depth_range=(0.15, 1.5),
        max_points=20000,
    )
    table_points = _select_table_candidates(scene_points, lower_z_percentile=60.0)
    table_normal, table_offset = _fit_plane_svd(table_points)
    if table_normal[2] < 0.0:
        table_normal = -table_normal
        table_offset = -table_offset
    table_tilt_deg = _plane_tilt_deg(table_normal)
    table_z_at_ball = _table_z_at_xy(table_normal, table_offset, object_pos[:2])
    ball_clearance_m = float(object_pos[2] - table_z_at_ball)
    ball_clearance_median_m = float(ball_z_median - table_z_at_ball)
    grasp_clearance_m = float(grasp_pos[2] - table_z_at_ball)

    projected_grasp = _project_base_to_pixel(grasp_pos, T_base_camera=T_main, intrinsics=intrinsics)
    top_overlay = rgb.copy()
    if projected_grasp is not None:
        top_overlay = _draw_marker(top_overlay, projected_grasp, color=(0, 128, 255), label="grasp")
    cx, cy = _mask_centroid(mask)
    top_overlay = _draw_marker(top_overlay, (cx, cy), color=(0, 255, 0), label="ball")

    top_lines = [
        f"table_tilt={table_tilt_deg:.2f}deg",
        f"ball_clearance_top={ball_clearance_m*100:.1f}cm (med={ball_clearance_median_m*100:.1f})",
        f"grasp_clearance={grasp_clearance_m*100:.1f}cm (target~{grasp_z_offset*100:.0f})",
        f"object_z={object_pos[2]:.3f}m grasp_z={grasp_pos[2]:.3f}m",
    ]
    _annotate_and_save(output_dir / "top_depth_check.png", top_overlay, title="Top camera", lines=top_lines)

    issues: list[str] = []
    if table_tilt_deg > float(args.max_table_tilt_deg):
        issues.append(f"table tilt {table_tilt_deg:.2f}deg > {args.max_table_tilt_deg:.2f}deg")
    if ball_clearance_m < float(args.min_ball_clearance_m):
        issues.append(
            f"ball top too low vs table ({ball_clearance_m*100:.1f}cm < {args.min_ball_clearance_m*100:.1f}cm)"
        )
    if ball_clearance_m > float(args.max_ball_clearance_m):
        issues.append(
            f"ball top too high vs table ({ball_clearance_m*100:.1f}cm > {args.max_ball_clearance_m*100:.1f}cm)"
        )
    grasp_err_m = abs((grasp_clearance_m - ball_clearance_m) - grasp_z_offset)
    if grasp_err_m > float(args.max_grasp_clearance_error_m):
        issues.append(
            f"grasp offset mismatch (grasp-table={grasp_clearance_m*100:.1f}cm, "
            f"ball-table={ball_clearance_m*100:.1f}cm, "
            f"delta={(grasp_clearance_m-ball_clearance_m)*100:.1f}cm vs target {grasp_z_offset*100:.1f}cm)"
        )

    side_topic = args.side_topic or low_level.get("img_side_topic")
    side_pose = None
    side_mats = low_level.get("camera_pose_mats", {})
    if isinstance(side_mats, dict):
        side_pose = side_mats.get("side")
    side_report: dict[str, Any] = {"enabled": False}
    side_height_m = low_level.get("side_camera_height_above_table_m")
    if side_height_m is not None:
        side_report["camera_height_above_table_m"] = float(side_height_m)

    if side_topic:
        try:
            side_rgb = _wait_for_side_rgb(str(side_topic), args.timeout)
            side_results = _sam3_segment(side_rgb, args.object_prompt, args.sam3_url)
            side_overlay = side_rgb.copy()
            side_lines = [f"topic={side_topic}"]
            if side_results:
                side_mask = np.asarray(side_results[0]["mask"], dtype=bool)
                side_centroid = _mask_centroid(side_mask)
                side_overlay = _draw_marker(
                    side_overlay, side_centroid, color=(0, 255, 0), label="ball(side)"
                )
                side_report["ball_centroid_px"] = [side_centroid[0], side_centroid[1]]
            if side_pose is not None:
                T_side = np.asarray(side_pose, dtype=np.float64).reshape(4, 4)
                side_intrinsics = np.asarray(
                    low_level.get("side_camera_intrinsics", _default_side_intrinsics(side_rgb)),
                    dtype=np.float64,
                ).reshape(3, 3)
                projected_object = _project_base_to_pixel(
                    object_pos, T_base_camera=T_side, intrinsics=side_intrinsics
                )
                projected_grasp_side = _project_base_to_pixel(
                    grasp_pos, T_base_camera=T_side, intrinsics=side_intrinsics
                )
                if projected_object is not None:
                    side_overlay = _draw_marker(
                        side_overlay, projected_object, color=(255, 128, 0), label="obj3d"
                    )
                if projected_grasp_side is not None:
                    side_overlay = _draw_marker(
                        side_overlay, projected_grasp_side, color=(0, 128, 255), label="grasp3d"
                    )
                if side_results and projected_object is not None:
                    err_px = float(np.hypot(side_centroid[0] - projected_object[0], side_centroid[1] - projected_object[1]))
                    side_report["projection_error_px"] = err_px
                    side_lines.append(f"side_proj_err={err_px:.1f}px")
                    if err_px > float(args.max_side_projection_px):
                        issues.append(f"side projection error {err_px:.1f}px > {args.max_side_projection_px:.1f}px")
            else:
                side_lines.append("side_pose=missing (visual only)")
            _annotate_and_save(
                output_dir / "side_depth_check.png",
                side_overlay,
                title="Side USB camera",
                lines=side_lines,
            )
            side_path = output_dir / "side_latest_rgb.png"
            PILImage.fromarray(side_rgb).save(side_path)
            side_report["enabled"] = True
            side_report["image"] = str(side_path)
        except Exception as exc:
            issues.append(f"side camera check failed: {exc}")
            print(f"[warn] side camera: {exc}")

    report = {
        "ok": len(issues) == 0,
        "issues": issues,
        "object_pos_m": object_pos.tolist(),
        "grasp_pos_m": grasp_pos.tolist(),
        "table_tilt_deg": table_tilt_deg,
        "ball_clearance_m": ball_clearance_m,
        "ball_clearance_median_m": ball_clearance_median_m,
        "grasp_clearance_m": grasp_clearance_m,
        "grasp_z_offset_m": grasp_z_offset,
        "expected_ball_radius_m": TENNIS_BALL_RADIUS_M,
        "side": side_report,
        "outputs": {
            "top": str(output_dir / "top_depth_check.png"),
            "report": str(output_dir / "report.json"),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("[check] summary:")
    print(f"  table_tilt_deg={table_tilt_deg:.3f}")
    print(f"  ball_clearance_top_cm={ball_clearance_m*100:.2f} (median={ball_clearance_median_m*100:.2f})")
    print(f"  grasp_clearance_cm={grasp_clearance_m*100:.2f} (target={grasp_z_offset*100:.0f})")
    if side_height_m is not None:
        print(f"  side_camera_height_above_table_cm={float(side_height_m)*100:.0f}")
    print(f"  grasp_pos_m={np.array2string(grasp_pos, precision=4)}")
    print(f"[output] wrote {output_dir / 'top_depth_check.png'}")
    if side_report.get("enabled"):
        print(f"[output] wrote {output_dir / 'side_depth_check.png'}")
    print(f"[output] wrote {output_dir / 'report.json'}")

    if issues:
        print("[check] FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("[check] PASSED")
    return 0


def _default_side_intrinsics(rgb: np.ndarray) -> list[float]:
    h, w = rgb.shape[:2]
    fx = float(w)
    fy = float(h)
    cx = w / 2.0
    cy = h / 2.0
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify pre-grasp depth alignment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--object-prompt", default="tennis ball")
    parser.add_argument("--sam3-url", default=DEFAULT_SAM3_URL)
    parser.add_argument("--side-topic", default=None, help="Override side RGB topic")
    parser.add_argument("--ball-z-percentile", type=float, default=85.0)
    parser.add_argument("--max-table-tilt-deg", type=float, default=2.0)
    parser.add_argument("--min-ball-clearance-m", type=float, default=0.010)
    parser.add_argument("--max-ball-clearance-m", type=float, default=0.070)
    parser.add_argument("--max-grasp-clearance-error-m", type=float, default=0.025)
    parser.add_argument("--max-side-projection-px", type=float, default=80.0)
    return parser


def main() -> None:
    raise SystemExit(run_check(build_parser().parse_args()))


if __name__ == "__main__":
    main()
