#!/usr/bin/env python3
"""Refine Agilex top-camera extrinsics using a flat table plane.

For a downward-looking camera, a small pitch/roll error makes the table look
like a sloped plane (near vs far height bias). This script:

1. Captures one RGB-D frame from the main camera.
2. Builds a base-frame point cloud with the current T_base_camera.
3. RANSAC-fits the table plane.
4. Applies a small base-frame rotation so the plane normal aligns with +Z.
5. Optionally shifts Z so the plane matches --table-height.

Usage (empty table, arm out of view):
  uv run --no-sync --active scripts/refine_agilex_camera_pose_with_table_plane.py \\
    --config env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml \\
    --write-config

Then verify:
  uv run --no-sync --active scripts/visualize_agilex_camera_calibration.py --mode live \\
    --config env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import yaml
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agilex_camera_calib_io import (  # noqa: E402
    DEFAULT_AGILEX_CONFIG,
    apply_calibration_to_config,
    print_yaml_matrix,
    read_T_base_camera,
    write_calibration_result_json,
)
from visualize_agilex_urdf import _add_frame  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/agilex_table_plane_refine")
DEFAULT_CONFIG = DEFAULT_AGILEX_CONFIG
BASE_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _load_low_level_config(config_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    low_level = dict(OmegaConf.to_container(cfg.env.cfg.low_level, resolve=True))
    low_level.pop("_target_", None)
    return low_level


def _load_T_base_camera(
    *,
    config_path: Path | None,
    calibration_file: Path | None,
) -> np.ndarray:
    if calibration_file is not None:
        return read_T_base_camera(calibration_file)
    if config_path is None:
        raise ValueError("Provide --calibration-file or --config with camera_pose_mats.main")
    low_level = _load_low_level_config(config_path)
    pose = low_level.get("camera_pose_mats", {}).get("main")
    if pose is None:
        raise ValueError(f"No camera_pose_mats.main in {config_path}")
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def _normalize_depth(image: np.ndarray, depth_scale: float) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) * depth_scale
    else:
        arr = arr.astype(np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _decode_depth_msg(depth_msg: Any, depth_scale: float) -> np.ndarray:
    encoding = getattr(depth_msg, "encoding", "") or ""
    data = np.frombuffer(depth_msg.data, dtype=np.uint8)
    if encoding in ("16UC1", "mono16"):
        arr = data.view(np.uint16).reshape(depth_msg.height, depth_msg.width)
        return arr.astype(np.float64) * depth_scale
    if encoding in ("32FC1",):
        arr = data.view(np.float32).reshape(depth_msg.height, depth_msg.width)
        return arr.astype(np.float64)

    try:
        from cv_bridge import CvBridge

        bridge = CvBridge()
        return _normalize_depth(
            bridge.imgmsg_to_cv2(depth_msg, "passthrough"),
            depth_scale,
        )
    except Exception:
        if data.size == depth_msg.height * depth_msg.width * 2:
            arr = data.view(np.uint16).reshape(depth_msg.height, depth_msg.width)
            return arr.astype(np.float64) * depth_scale
        raise


def _wait_for_rgbd(low_level: dict[str, Any], timeout_s: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import rospy
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError("rospy and sensor_msgs are required") from exc

    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_table_plane_refine", anonymous=True, disable_signals=True)

    depth_topic = low_level.get("img_front_depth_topic", "/camera_h/depth/image_raw")
    camera_info_topic = low_level.get("img_front_camera_info_topic", "/camera_h/color/camera_info")

    print(f"[ros] waiting depth: {depth_topic}")
    depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=timeout_s)
    print(f"[ros] waiting camera_info: {camera_info_topic}")
    info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=timeout_s)

    depth = _decode_depth_msg(depth_msg, float(low_level.get("depth_scale", 0.001)))
    intrinsics = np.asarray(info_msg.K, dtype=np.float64).reshape(3, 3)
    if depth.shape[:2] != (int(info_msg.height), int(info_msg.width)):
        from capx.utils.camera_utils import scale_intrinsics_for_image

        intrinsics = scale_intrinsics_for_image(
            intrinsics,
            depth.shape[:2],
            (int(info_msg.height), int(info_msg.width)),
        )
    return depth, intrinsics


def _depth_to_base_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
    *,
    depth_range: tuple[float, float],
    max_points: int,
) -> np.ndarray:
    near, far = depth_range
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    ys, xs = np.nonzero(valid)
    if len(xs) > max_points:
        pick = np.random.default_rng(0).choice(len(xs), max_points, replace=False)
        ys = ys[pick]
        xs = xs[pick]
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    z = depth[ys, xs]
    x = (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_cam = np.stack([x, y, z], axis=1)
    points_h = np.concatenate(
        [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_base = (T_base_camera @ points_h.T).T[:, :3]
    points_base = points_base[np.isfinite(points_base).all(axis=1)]
    return points_base


def _select_table_candidates(points_base: np.ndarray, *, lower_z_percentile: float) -> np.ndarray:
    if len(points_base) == 0:
        return points_base
    cutoff = np.percentile(points_base[:, 2], lower_z_percentile)
    return points_base[points_base[:, 2] <= cutoff]


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def _plane_distances(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    return np.abs(points @ normal + offset)


def _ransac_plane(
    points: np.ndarray,
    *,
    threshold: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    if len(points) < 50:
        raise RuntimeError(f"Need at least 50 valid depth points, got {len(points)}")

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_count = 0
    n_pts = len(points)

    for _ in range(iterations):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, p0))
        inliers = _plane_distances(points, normal, offset) < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 50:
        raise RuntimeError("RANSAC plane fit failed; clear the table and widen --depth-range")

    inlier_pts = points[best_inliers]
    normal, offset = _fit_plane_svd(inlier_pts)
    if normal[2] < 0.0:
        normal = -normal
        offset = -offset
    return normal, offset, best_inliers


def _plane_tilt_deg(normal: np.ndarray) -> float:
    n = normal / max(np.linalg.norm(normal), 1e-12)
    cos_angle = float(np.clip(np.dot(n, BASE_UP), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _plane_metrics(points: np.ndarray, normal: np.ndarray, offset: float) -> dict[str, float]:
    dist = _plane_distances(points, normal, offset)
    return {
        "tilt_deg": _plane_tilt_deg(normal),
        "mean_abs_dist_m": float(np.mean(dist)),
        "rmse_m": float(np.sqrt(np.mean(dist**2))),
        "max_abs_dist_m": float(np.max(dist)),
    }


def _rotation_align_normal_to_base_up(normal: np.ndarray) -> np.ndarray:
    n = normal / max(np.linalg.norm(normal), 1e-12)
    if n[2] < 0.0:
        n = -n
    # scipy: R.apply(a) ~= b. With a=+Z and b=n, applying R to the plane flattens it (R.apply(n) ~= +Z).
    rot, _ = Rotation.align_vectors([BASE_UP], [n])
    return rot.as_matrix()


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _refine_pose_with_table_plane(
    T_base_camera: np.ndarray,
    points_base: np.ndarray,
    *,
    ransac_threshold: float,
    ransac_iterations: int,
    table_height: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    normal, offset, inliers = _ransac_plane(
        points_base,
        threshold=ransac_threshold,
        iterations=ransac_iterations,
        seed=0,
    )
    inlier_pts = points_base[inliers]
    before = _plane_metrics(inlier_pts, normal, offset)

    R_fix = _rotation_align_normal_to_base_up(normal)
    T_rot = _make_transform(R_fix, np.zeros(3, dtype=np.float64))
    T_stage = T_rot @ T_base_camera

    stage_pts = (T_rot @ np.concatenate([inlier_pts, np.ones((len(inlier_pts), 1))], axis=1).T).T[:, :3]
    stage_normal, stage_offset = _fit_plane_svd(stage_pts)
    if stage_normal[2] < 0.0:
        stage_normal = -stage_normal
        stage_offset = -stage_offset
    after_rot = _plane_metrics(stage_pts, stage_normal, stage_offset)

    T_height = np.eye(4, dtype=np.float64)
    height_shift = 0.0
    if table_height is not None:
        mean_z = float(np.mean(stage_pts[:, 2]))
        height_shift = float(table_height) - mean_z
        T_height[:3, 3] = np.array([0.0, 0.0, height_shift], dtype=np.float64)

    T_refined = T_height @ T_stage
    final_pts = (T_height @ np.concatenate([stage_pts, np.ones((len(stage_pts), 1))], axis=1).T).T[:, :3]
    final_normal, final_offset = _fit_plane_svd(final_pts)
    if final_normal[2] < 0.0:
        final_normal = -final_normal
        final_offset = -final_offset
    after = _plane_metrics(final_pts, final_normal, final_offset)

    info = {
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "before_tilt_deg": before["tilt_deg"],
        "after_tilt_deg": after["tilt_deg"],
        "before_plane_rmse_mm": before["rmse_m"] * 1000.0,
        "after_plane_rmse_mm": after["rmse_m"] * 1000.0,
        "after_rot_tilt_deg": after_rot["tilt_deg"],
        "height_shift_m": height_shift,
        "table_height_target_m": table_height,
        "T_rot_fix": T_rot.tolist(),
        "T_height_fix": T_height.tolist(),
    }
    return T_refined, info


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.random.default_rng(1).choice(len(points), max_points, replace=False)
    return points[idx]


def _build_plane_figure(
    *,
    points_before: np.ndarray,
    points_after: np.ndarray,
    T_before: np.ndarray,
    T_after: np.ndarray,
) -> go.Figure:
    before_show = _subsample(points_before, 5000)
    after_show = _subsample(points_after, 5000)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=before_show[:, 0],
            y=before_show[:, 1],
            z=before_show[:, 2],
            mode="markers",
            marker={"size": 1.5, "color": before_show[:, 2], "colorscale": "Viridis", "opacity": 0.45},
            name="before refine",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=after_show[:, 0],
            y=after_show[:, 1],
            z=after_show[:, 2],
            mode="markers",
            marker={"size": 1.5, "color": after_show[:, 2], "colorscale": "Plasma", "opacity": 0.45},
            name="after refine",
        )
    )
    _add_frame(fig, np.eye(4), "base", axis_len=0.08)
    _add_frame(fig, T_before, "camera_before", axis_len=0.07)
    _add_frame(fig, T_after, "camera_after", axis_len=0.07)
    fig.update_layout(
        title="Table plane refinement (before vs after point cloud)",
        scene={
            "xaxis": {"title": "base X (m)"},
            "yaxis": {"title": "base Y (m)"},
            "zaxis": {"title": "base Z (m)"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine camera_pose_mats.main using a flat table plane."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration-file", type=Path, default=None)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--depth-range", nargs=2, type=float, default=(0.15, 1.5))
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--ransac-threshold", type=float, default=0.008)
    parser.add_argument("--ransac-iterations", type=int, default=800)
    parser.add_argument(
        "--table-z-percentile",
        type=float,
        default=60.0,
        help="Use the lowest Z points in base frame (likely table) before RANSAC. 0 disables.",
    )
    parser.add_argument("--no-html", action="store_true", help="Skip plotly HTML output.")
    parser.add_argument(
        "--table-height",
        type=float,
        default=None,
        help="Optional target table height in base frame (m). Default: only fix tilt, not absolute Z.",
    )
    parser.add_argument(
        "--reset-graspnet-offsets",
        action="store_true",
        help="Zero graspnet offsets when writing --write-config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    low_level = _load_low_level_config(config_path)
    T_before = _load_T_base_camera(
        config_path=config_path,
        calibration_file=args.calibration_file,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth, intrinsics = _wait_for_rgbd(low_level, args.timeout)
    depth_range = (float(args.depth_range[0]), float(args.depth_range[1]))
    points_before = _depth_to_base_points(
        depth,
        intrinsics,
        T_before,
        depth_range=depth_range,
        max_points=args.max_points,
    )
    if len(points_before) < 1000:
        raise RuntimeError(
            f"Too few depth points ({len(points_before)}). "
            "Check camera stream, depth_range, and that the table is visible."
        )

    table_points = points_before
    if float(args.table_z_percentile) > 0.0:
        table_points = _select_table_candidates(
            points_before,
            lower_z_percentile=float(args.table_z_percentile),
        )
        print(
            f"[refine] table candidates: {len(table_points)} / {len(points_before)} "
            f"(lowest {args.table_z_percentile:.0f}% Z)"
        )
        if len(table_points) < 500:
            raise RuntimeError(
                f"Too few table candidate points ({len(table_points)}). "
                "Try --table-z-percentile 0 or adjust --depth-range."
            )

    T_after, info = _refine_pose_with_table_plane(
        T_before,
        table_points,
        ransac_threshold=float(args.ransac_threshold),
        ransac_iterations=int(args.ransac_iterations),
        table_height=args.table_height,
    )
    points_after = _depth_to_base_points(
        depth,
        intrinsics,
        T_after,
        depth_range=depth_range,
        max_points=args.max_points,
    )

    print("[refine] plane fit summary:")
    print(
        f"  inliers={info['inlier_count']} ({info['inlier_ratio'] * 100:.1f}% of sampled points)"
    )
    print(
        f"  tilt: {info['before_tilt_deg']:.3f} deg -> {info['after_tilt_deg']:.3f} deg"
    )
    print(
        f"  plane rmse: {info['before_plane_rmse_mm']:.2f} mm -> "
        f"{info['after_plane_rmse_mm']:.2f} mm"
    )
    if args.table_height is not None:
        print(f"  applied base Z shift: {info['height_shift_m'] * 1000.0:.2f} mm")

    result_doc = {
        "refinement_type": "table_plane",
        "T_base_camera_before": T_before.tolist(),
        "T_base_camera_after": T_after.tolist(),
        "metrics": info,
        "depth_range_m": list(depth_range),
        "ransac_threshold_m": float(args.ransac_threshold),
    }
    result_path = output_dir / "T_base_camera_refined.yaml"
    result_path.write_text(
        yaml.safe_dump(
            {
                "T_base_camera": T_after.tolist(),
                "homogeneous_matrix": T_after.tolist(),
                **result_doc,
            },
            sort_keys=False,
        )
    )
    write_calibration_result_json(
        output_dir / "calibration_result.json",
        calibration_type="table_plane_refine",
        transform=T_after,
        metrics={
            "before_tilt_deg": info["before_tilt_deg"],
            "after_tilt_deg": info["after_tilt_deg"],
            "before_plane_rmse_mm": info["before_plane_rmse_mm"],
            "after_plane_rmse_mm": info["after_plane_rmse_mm"],
        },
        extra=result_doc,
    )

    fig = _build_plane_figure(
        points_before=points_before,
        points_after=points_after,
        T_before=T_before,
        T_after=T_after,
    )
    html_path = output_dir / "table_plane_refine.html"
    if not args.no_html:
        fig.write_html(str(html_path), include_plotlyjs="cdn")

    print("[output] wrote:")
    print(f"  {result_path}")
    if not args.no_html:
        print(f"  {html_path}")
    print("[output] refined T_base_camera / camera_pose_mats.main:")
    print(T_after)
    print_yaml_matrix(T_after)

    if args.write_config:
        apply_calibration_to_config(
            config_path,
            T_after,
            reset_graspnet_offsets=bool(args.reset_graspnet_offsets),
        )
        print(f"[write] updated {config_path}")
    else:
        print("[hint] add --write-config to update camera_pose_mats.main in the yaml")


if __name__ == "__main__":
    main()
