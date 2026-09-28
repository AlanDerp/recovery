#!/usr/bin/env python3
"""Refine rectified head-stereo roll/pitch from a SAM3 tabletop point cloud."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = (
    REPO_ROOT / "outputs/r1lite/head_stereo_rectified_plane_calibration.json"
)


def dominant_plane(
    points: np.ndarray,
    *,
    threshold_m: float = 0.010,
    iterations: int = 30000,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(20260830)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(int(iterations)):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(np.dot(normal, sample[0]))
        inliers = np.abs(points @ normal + offset) <= float(threshold_m)
        if int(np.count_nonzero(inliers)) > int(np.count_nonzero(best)):
            best = inliers
    if int(np.count_nonzero(best)) < 200:
        raise RuntimeError("table point cloud has fewer than 200 planar inliers")
    center = np.median(points[best], axis=0)
    _, _, vh = np.linalg.svd(points[best] - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal *= -1.0
    offset = -float(np.dot(normal, center))
    refined = np.abs(points @ normal + offset) <= float(threshold_m)
    return normal, refined, offset


def rotation_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(first, second))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    rotation_vector = cross / sine * np.arctan2(sine, cosine)
    return cv2.Rodrigues(rotation_vector)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--table-audit", type=Path)
    source.add_argument("--validation-audit", type=Path)
    parser.add_argument("--threshold-m", type=float, default=0.010)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--maximum-correction-deg", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration.read_text())
    transform = np.asarray(
        calibration["base_from_rectified_left_camera"], dtype=np.float64
    ).reshape(4, 4)
    if args.validation_audit is not None:
        audit = json.loads(args.validation_audit.read_text())
        if float(audit.get("sam3_table_score", 0.0)) < 0.80:
            raise RuntimeError("a high-confidence SAM3 table validation is required")
        normal = np.asarray(audit["normal_base"], dtype=np.float64)
        camera_center = np.asarray(
            audit["plane_center_camera_m"], dtype=np.float64
        )
        pivot = (
            transform[:3, :3] @ camera_center + transform[:3, 3]
        )
        ratio = float(audit["inlier_count"]) / float(audit["candidate_count"])
        offset = float(audit["plane_offset"])
        point_count = int(audit["candidate_count"])
        inlier_count = int(audit["inlier_count"])
        sam3_score = float(audit["sam3_table_score"])
        source_path = args.validation_audit
    else:
        audit = json.loads(args.table_audit.read_text())
        detections = audit["events"][0]["detections"]
        table = detections.get("table")
        if not isinstance(table, dict) or float(table.get("score", 0.0)) < 0.80:
            raise RuntimeError("a high-confidence SAM3 table detection is required")
        base_points = np.asarray(table["point_cloud_base_m"], dtype=np.float64)
        if len(base_points) < 200 or not np.isfinite(base_points).all():
            raise RuntimeError("table audit has insufficient finite points")
        normal, inliers, offset = dominant_plane(
            base_points, threshold_m=args.threshold_m
        )
        ratio = float(np.mean(inliers))
        pivot = np.median(base_points[inliers], axis=0)
        point_count = int(len(base_points))
        inlier_count = int(np.count_nonzero(inliers))
        sam3_score = float(table["score"])
        source_path = args.table_audit
    if ratio < float(args.minimum_inlier_ratio):
        raise RuntimeError(
            f"table plane inlier ratio {ratio:.3f} is below "
            f"{args.minimum_inlier_ratio:.3f}"
        )
    correction_deg = float(
        np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
    )
    if correction_deg > float(args.maximum_correction_deg):
        raise RuntimeError(
            f"required table correction {correction_deg:.2f}deg exceeds limit"
        )

    correction = rotation_between(normal, np.array([0.0, 0.0, 1.0]))
    target_z = float(calibration["plane_z"])
    z_shift = target_z - float(pivot[2])

    refined = np.eye(4)
    refined[:3, :3] = correction @ transform[:3, :3]
    refined[:3, 3] = (
        correction @ (transform[:3, 3] - pivot)
        + pivot
        + np.array([0.0, 0.0, z_shift])
    )
    calibration["base_from_rectified_left_camera"] = refined.tolist()
    calibration["table_plane_refinement"] = {
        "schema": "capx.r1lite.head_stereo_table_plane_refinement.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_audit": str(source_path.resolve()),
        "sam3_score": sam3_score,
        "point_count": point_count,
        "inlier_count": inlier_count,
        "inlier_ratio": ratio,
        "normal_base_before": normal.tolist(),
        "plane_offset_before": offset,
        "correction_rotation": correction.tolist(),
        "correction_angle_deg": correction_deg,
        "pivot_base_m": pivot.tolist(),
        "z_shift_m": z_shift,
        "post_refinement_z_rmse_m": (
            float(audit.get("rmse_m", 0.0))
            if args.validation_audit is not None
            else None
        ),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.calibration.with_name(
        f"{args.calibration.stem}_backup_{stamp}{args.calibration.suffix}"
    )
    shutil.copy2(args.calibration, backup)
    args.calibration.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"backed up: {backup}")
    print(f"refined calibration: {args.calibration}")
    print(
        f"table correction={correction_deg:.3f}deg, "
        f"z_shift={z_shift:.4f}m, inlier_ratio={ratio:.3f}"
    )


if __name__ == "__main__":
    main()
