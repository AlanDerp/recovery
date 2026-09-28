"""Head-camera towel geometry for the seventh R1 Lite pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ProjectedKeypoint:
    name: str
    pixel_uv: list[float]
    base_xyz_m: list[float]
    extrapolation_m: float


def pixel_to_table(
    homography: np.ndarray,
    pixel_uv: np.ndarray,
    table_z_m: float,
) -> np.ndarray:
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    pixel = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
    projected = matrix @ np.array([pixel[0], pixel[1], 1.0])
    if not np.isfinite(projected).all() or abs(float(projected[2])) < 1e-12:
        raise ValueError("homography projection is non-finite or degenerate")
    xy = projected[:2] / projected[2]
    return np.array([xy[0], xy[1], float(table_z_m)], dtype=np.float64)


def calibration_polygon(calibration: dict[str, Any]) -> np.ndarray:
    pixels = np.asarray(
        [item["pixel"] for item in calibration.get("points", [])],
        dtype=np.float64,
    ).reshape(-1, 2)
    if len(pixels) < 4 or not np.isfinite(pixels).all():
        raise ValueError("head-plane calibration needs at least four finite points")
    return cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2).astype(float)


def _nearest_polygon_point(point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    query = np.asarray(point, dtype=np.float64).reshape(2)
    vertices = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    best = None
    best_distance = float("inf")
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        direction = end - start
        denominator = float(np.dot(direction, direction))
        alpha = (
            0.0
            if denominator < 1e-12
            else float(np.clip(np.dot(query - start, direction) / denominator, 0.0, 1.0))
        )
        candidate = start + alpha * direction
        distance = float(np.linalg.norm(query - candidate))
        if distance < best_distance:
            best = candidate
            best_distance = distance
    assert best is not None
    return best


def extrapolation_distance_m(
    pixel_uv: np.ndarray,
    calibration: dict[str, Any],
) -> float:
    """Measure physical XY distance beyond the calibrated convex hull."""
    pixel = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
    polygon = calibration_polygon(calibration)
    inside = cv2.pointPolygonTest(
        polygon.astype(np.float32),
        (float(pixel[0]), float(pixel[1])),
        False,
    )
    if inside >= 0:
        return 0.0
    boundary_pixel = _nearest_polygon_point(pixel, polygon)
    homography = np.asarray(
        calibration["image_to_base_xy_homography"],
        dtype=np.float64,
    )
    table_z = float(calibration["plane_z"])
    point_base = pixel_to_table(homography, pixel, table_z)
    boundary_base = pixel_to_table(homography, boundary_pixel, table_z)
    return float(np.linalg.norm(point_base[:2] - boundary_base[:2]))


def extract_towel_geometry(
    mask: np.ndarray,
    calibration: dict[str, Any],
    *,
    maximum_extrapolation_m: float = 0.15,
    minimum_mask_pixels: int = 2000,
) -> dict[str, Any]:
    """Extract center and camera-left/right side midpoints from a towel mask.

    The full mask may extend arbitrarily beyond the calibration convex hull.
    Only the three projected keypoints are subject to the bounded 15 cm
    extrapolation gate.
    """
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError("towel mask must be two-dimensional")
    if int(np.count_nonzero(binary)) < int(minimum_mask_pixels):
        raise ValueError("towel mask is too small")
    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    if components <= 1:
        raise ValueError("towel mask has no connected foreground")
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest = labels == largest_label
    if float(areas[largest_label - 1]) / float(np.count_nonzero(binary)) < 0.85:
        raise ValueError("towel mask is fragmented or ambiguous")

    contours, _ = cv2.findContours(
        largest.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("towel contour extraction failed")
    contour = max(contours, key=cv2.contourArea)
    rectangle = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rectangle).astype(np.float64)
    area = float(cv2.contourArea(contour))
    rectangle_area = float(rectangle[1][0] * rectangle[1][1])
    if rectangle_area <= 1.0 or area / rectangle_area < 0.50:
        raise ValueError("towel mask is not sufficiently rectangular")

    # cv2.boxPoints is cyclic. Opposite edge midpoint pairs are (0, 2) and
    # (1, 3); select the pair with the strongest camera-horizontal separation.
    edge_midpoints = (corners + np.roll(corners, -1, axis=0)) / 2.0
    opposite_pairs = [(edge_midpoints[0], edge_midpoints[2]), (edge_midpoints[1], edge_midpoints[3])]
    first, second = max(
        opposite_pairs,
        key=lambda pair: abs(float(pair[0][0] - pair[1][0])),
    )
    rectangle_left, rectangle_right = sorted(
        [first, second],
        key=lambda point: float(point[0]),
    )
    right_edge_index = int(
        np.argmin(np.linalg.norm(edge_midpoints - rectangle_right, axis=1))
    )
    right_edge_start = corners[right_edge_index]
    right_edge_end = corners[(right_edge_index + 1) % 4]

    contour_points = contour.reshape(-1, 2).astype(np.float64)
    contour_next = np.roll(contour_points, -1, axis=0)
    contour_segments = contour_next - contour_points
    segment_norm_sq = np.sum(contour_segments * contour_segments, axis=1)

    def project_to_contour(point: np.ndarray) -> tuple[np.ndarray, float]:
        relative = point - contour_points
        parameters = np.divide(
            np.sum(relative * contour_segments, axis=1),
            segment_norm_sq,
            out=np.zeros_like(segment_norm_sq),
            where=segment_norm_sq > 1e-12,
        )
        parameters = np.clip(parameters, 0.0, 1.0)
        projections = contour_points + parameters[:, None] * contour_segments
        distances = np.linalg.norm(projections - point, axis=1)
        index = int(np.argmin(distances))
        return projections[index], float(distances[index])

    left_pixel, left_projection_px = project_to_contour(rectangle_left)
    right_pixel, right_projection_px = project_to_contour(rectangle_right)
    center_pixel = corners.mean(axis=0)

    homography = np.asarray(
        calibration["image_to_base_xy_homography"],
        dtype=np.float64,
    )
    table_z = float(calibration["plane_z"])

    def contour_path(start_index: int, end_index: int, step: int) -> np.ndarray:
        indices = [start_index]
        current = start_index
        while current != end_index:
            current = (current + step) % len(contour_points)
            indices.append(current)
        return contour_points[np.asarray(indices, dtype=int)]

    def edge_distance_score(points: np.ndarray) -> float:
        direction = right_edge_end - right_edge_start
        denominator = float(np.dot(direction, direction))
        relative = points - right_edge_start
        alpha = np.clip(
            np.sum(relative * direction, axis=1) / max(denominator, 1e-12),
            0.0,
            1.0,
        )
        projections = right_edge_start + alpha[:, None] * direction
        return float(np.mean(np.linalg.norm(points - projections, axis=1)))

    start_index = int(
        np.argmin(np.linalg.norm(contour_points - right_edge_start, axis=1))
    )
    end_index = int(
        np.argmin(np.linalg.norm(contour_points - right_edge_end, axis=1))
    )
    right_side_paths = (
        contour_path(start_index, end_index, 1),
        contour_path(start_index, end_index, -1),
    )
    right_side_contour = min(right_side_paths, key=edge_distance_score)
    right_side_base = np.asarray(
        [pixel_to_table(homography, point, table_z) for point in right_side_contour],
        dtype=np.float64,
    )
    right_side_lengths = np.concatenate(
        [
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(right_side_base[:, :2], axis=0), axis=1)),
        ]
    )
    right_side_total_length_m = float(right_side_lengths[-1])
    corner_exclusion_m = 0.01
    finger_contact_width_m = 0.01
    contact_center_inset_m = finger_contact_width_m / 2.0
    center_base = pixel_to_table(homography, center_pixel, table_z)
    if right_side_total_length_m <= 2.0 * corner_exclusion_m:
        raise ValueError("towel right side is too short after corner exclusions")
    sample_distances = np.arange(
        corner_exclusion_m,
        right_side_total_length_m - corner_exclusion_m + 1e-9,
        0.01,
    )
    if len(sample_distances) > 24:
        sample_distances = np.linspace(
            corner_exclusion_m,
            right_side_total_length_m - corner_exclusion_m,
            24,
        )
    right_side_candidates = []
    for candidate_index, distance_m in enumerate(sample_distances):
        segment = min(
            int(np.searchsorted(right_side_lengths, distance_m, side="right") - 1),
            len(right_side_lengths) - 2,
        )
        segment_length = right_side_lengths[segment + 1] - right_side_lengths[segment]
        alpha = (
            0.0
            if segment_length <= 1e-12
            else float((distance_m - right_side_lengths[segment]) / segment_length)
        )
        pixel = (
            right_side_contour[segment] * (1.0 - alpha)
            + right_side_contour[segment + 1] * alpha
        )
        contour_base = pixel_to_table(homography, pixel, table_z)
        tangent_xy = (
            right_side_base[segment + 1, :2] - right_side_base[segment, :2]
        )
        tangent_norm = float(np.linalg.norm(tangent_xy))
        if tangent_norm <= 1e-9:
            continue
        tangent_xy /= tangent_norm
        inward_normal_xy = np.array([-tangent_xy[1], tangent_xy[0]])
        if float(
            np.dot(inward_normal_xy, center_base[:2] - contour_base[:2])
        ) < 0.0:
            inward_normal_xy *= -1.0
        base = contour_base.copy()
        base[:2] += inward_normal_xy * contact_center_inset_m
        extrapolation = extrapolation_distance_m(pixel, calibration)
        if extrapolation > float(maximum_extrapolation_m) + 1e-9:
            continue
        right_side_candidates.append(
            {
                "id": f"right_side_{candidate_index:02d}",
                "pixel_uv": pixel.tolist(),
                "base_xyz_m": base.tolist(),
                "contour_base_xyz_m": contour_base.tolist(),
                "inward_normal_base_xy": inward_normal_xy.tolist(),
                "contact_center_inset_m": contact_center_inset_m,
                "arc_distance_from_first_corner_m": float(distance_m),
                "extrapolation_m": extrapolation,
            }
        )
    if not right_side_candidates:
        raise ValueError("towel right side has no bounded contour candidates")

    keypoints: dict[str, ProjectedKeypoint] = {}
    for name, pixel in {
        "center": center_pixel,
        "left_edge_midpoint": left_pixel,
        "right_edge_midpoint": right_pixel,
    }.items():
        extrapolation = extrapolation_distance_m(pixel, calibration)
        if extrapolation > float(maximum_extrapolation_m) + 1e-9:
            raise ValueError(
                f"{name} exceeds bounded homography extrapolation: "
                f"{extrapolation:.6f}m > {maximum_extrapolation_m:.6f}m"
            )
        keypoints[name] = ProjectedKeypoint(
            name=name,
            pixel_uv=pixel.tolist(),
            base_xyz_m=pixel_to_table(homography, pixel, table_z).tolist(),
            extrapolation_m=extrapolation,
        )
    return {
        "mask_pixels": int(np.count_nonzero(binary)),
        "selected_component_pixels": int(np.count_nonzero(largest)),
        "rectangularity": area / rectangle_area,
        "corners_pixel_uv": corners.tolist(),
        "edge_midpoint_method": "rectangle_direction_then_nearest_contour_projection",
        "rectangle_edge_midpoints_pixel_uv": {
            "left": rectangle_left.tolist(),
            "right": rectangle_right.tolist(),
        },
        "contour_projection_distance_px": {
            "left": left_projection_px,
            "right": right_projection_px,
        },
        "right_side_contour_policy": {
            "definition": (
                "fitted_right_side_contour_with_inset_finger_contact_center"
            ),
            "corner_exclusion_m": corner_exclusion_m,
            "sample_spacing_m": 0.01,
            "maximum_candidates": 24,
            "side_length_m": right_side_total_length_m,
            "finger_contact_width_m": finger_contact_width_m,
            "contact_center_inset_m": contact_center_inset_m,
        },
        "right_side_contour_pixel_uv": right_side_contour.tolist(),
        "right_side_grasp_candidates": right_side_candidates,
        "keypoints": {
            name: {
                "pixel_uv": item.pixel_uv,
                "base_xyz_m": item.base_xyz_m,
                "extrapolation_m": item.extrapolation_m,
            }
            for name, item in keypoints.items()
        },
        "maximum_extrapolation_m": float(maximum_extrapolation_m),
        "mask_must_be_inside_calibration_polygon": False,
        "selected_mask": largest,
    }


def keypoint_repeatability_error_m(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    errors = []
    for name in ("center", "left_edge_midpoint", "right_edge_midpoint"):
        first_point = np.asarray(first["keypoints"][name]["base_xyz_m"], dtype=float)
        second_point = np.asarray(second["keypoints"][name]["base_xyz_m"], dtype=float)
        errors.append(float(np.linalg.norm(first_point - second_point)))
    return max(errors)


def extract_nearest_towel_corner_grasp_points(
    mask: np.ndarray,
    calibration: dict[str, Any],
    *,
    maximum_extrapolation_m: float = 0.15,
    minimum_mask_pixels: int = 2000,
    minimum_pair_separation_px: float = 40.0,
) -> dict[str, Any]:
    """Select the two towel corners nearest the robot base.

    Example: for an axis-aligned towel in front of the robot, select its two
    near-edge corners, not edge midpoints. For a rotated towel, rank all four
    projected corners by XY distance to the robot base and select the nearest
    two. Fitted rectangle corners are projected onto the real external contour
    before they are used as grasp targets.
    """
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError("towel mask must be two-dimensional")
    foreground_pixels = int(np.count_nonzero(binary))
    if foreground_pixels < int(minimum_mask_pixels):
        raise ValueError("towel mask is too small")

    components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    if components <= 1:
        raise ValueError("towel mask has no connected foreground")
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest = labels == largest_label
    selected_pixels = int(areas[largest_label - 1])
    if float(selected_pixels) / float(foreground_pixels) < 0.85:
        raise ValueError("towel mask is fragmented or ambiguous")

    contours, _ = cv2.findContours(
        largest.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("towel contour extraction failed")
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    if len(contour_points) < 4:
        raise ValueError("towel external contour is degenerate")

    center_pixel = np.asarray(centroids[largest_label], dtype=np.float64)
    homography = np.asarray(
        calibration["image_to_base_xy_homography"],
        dtype=np.float64,
    )
    table_z = float(calibration["plane_z"])

    rectangle = cv2.minAreaRect(contour)
    rectangle_area = float(rectangle[1][0] * rectangle[1][1])
    contour_area = float(cv2.contourArea(contour))
    if rectangle_area <= 1.0:
        raise ValueError("towel fitted rectangle is degenerate")
    rectangularity = contour_area / rectangle_area
    if rectangularity < 0.50:
        raise ValueError("towel mask is not sufficiently rectangular")

    fitted_corners = cv2.boxPoints(rectangle).astype(np.float64)
    contour_next = np.roll(contour_points, -1, axis=0)
    contour_segments = contour_next - contour_points
    segment_norm_sq = np.sum(contour_segments * contour_segments, axis=1)

    def project_to_contour(point: np.ndarray) -> tuple[np.ndarray, float]:
        relative = point - contour_points
        parameters = np.divide(
            np.sum(relative * contour_segments, axis=1),
            segment_norm_sq,
            out=np.zeros_like(segment_norm_sq),
            where=segment_norm_sq > 1e-12,
        )
        parameters = np.clip(parameters, 0.0, 1.0)
        projections = contour_points + parameters[:, None] * contour_segments
        distances = np.linalg.norm(projections - point, axis=1)
        index = int(np.argmin(distances))
        return projections[index], float(distances[index])

    projected_corners: list[np.ndarray] = []
    projection_distances_px: list[float] = []
    corner_base_points: list[np.ndarray] = []
    for fitted_corner in fitted_corners:
        projected, projection_distance = project_to_contour(fitted_corner)
        projected_corners.append(projected)
        projection_distances_px.append(projection_distance)
        corner_base_points.append(pixel_to_table(homography, projected, table_z))

    robot_xy = np.zeros(2, dtype=np.float64)
    robot_distances_m = [
        float(np.linalg.norm(point[:2] - robot_xy)) for point in corner_base_points
    ]
    ranked_indices = sorted(
        range(4),
        key=lambda index: (robot_distances_m[index], index),
    )
    selected_indices = ranked_indices[:2]
    selected_corner_pixels = [
        projected_corners[index] for index in selected_indices
    ]
    left_pixel, right_pixel = sorted(
        selected_corner_pixels,
        key=lambda point: float(point[0]),
    )
    pair_separation_px = float(np.linalg.norm(right_pixel - left_pixel))
    if (
        pair_separation_px < float(minimum_pair_separation_px)
        or not left_pixel[0] < right_pixel[0]
    ):
        raise ValueError("nearest towel corner pair is not safely separated")

    keypoints: dict[str, ProjectedKeypoint] = {}
    for name, pixel in {
        "center": center_pixel,
        "left_grasp": left_pixel,
        "right_grasp": right_pixel,
    }.items():
        extrapolation = extrapolation_distance_m(pixel, calibration)
        if extrapolation > float(maximum_extrapolation_m) + 1e-9:
            raise ValueError(
                f"{name} exceeds bounded homography extrapolation: "
                f"{extrapolation:.6f}m > {maximum_extrapolation_m:.6f}m"
            )
        keypoints[name] = ProjectedKeypoint(
            name=name,
            pixel_uv=pixel.tolist(),
            base_xyz_m=pixel_to_table(homography, pixel, table_z).tolist(),
            extrapolation_m=extrapolation,
        )

    contour_stride = max(1, int(np.ceil(len(contour_points) / 500.0)))
    sampled_contour_pixels = contour_points[::contour_stride]
    sampled_contour_base = [
        pixel_to_table(homography, pixel, table_z).tolist()
        for pixel in sampled_contour_pixels
    ]
    selection_margin_m = float(
        robot_distances_m[ranked_indices[2]] - robot_distances_m[ranked_indices[1]]
    )
    quality_score = float(np.clip(rectangularity, 0.0, 1.0))
    return {
        "source": "sam3_largest_external_contour",
        "grasp_point_method": "nearest_robot_two_projected_rectangle_corners",
        "mask_pixels": foreground_pixels,
        "selected_component_pixels": selected_pixels,
        "keypoints": {
            name: {
                "pixel_uv": item.pixel_uv,
                "base_xyz_m": item.base_xyz_m,
                "extrapolation_m": item.extrapolation_m,
            }
            for name, item in keypoints.items()
        },
        "quality": {
            "pair_separation_px": pair_separation_px,
            "rectangularity": rectangularity,
            "selected_corner_robot_distances_m": [
                robot_distances_m[index] for index in selected_indices
            ],
            "selection_margin_to_third_corner_m": selection_margin_m,
            "maximum_corner_projection_distance_px": max(
                projection_distances_px
            ),
            "left_contour_distance_px": 0.0,
            "right_contour_distance_px": 0.0,
            "quality_score": quality_score,
        },
        "corner_candidates": [
            {
                "pixel_uv": projected_corners[index].tolist(),
                "base_xyz_m": corner_base_points[index].tolist(),
                "robot_xy_distance_m": robot_distances_m[index],
                "selected": index in selected_indices,
            }
            for index in range(4)
        ],
        "robot_reference_base_xyz_m": [0.0, 0.0, 0.0],
        "pixel_order": "camera_left_to_camera_right",
        "same_scanline": bool(
            np.isclose(left_pixel[1], right_pixel[1], atol=1e-9)
        ),
        "points_on_real_external_contour": True,
        "external_contour_pixel_uv": sampled_contour_pixels.tolist(),
        "external_contour_base_xyz_m": sampled_contour_base,
        "external_contour_sample_stride": contour_stride,
        "maximum_extrapolation_m": float(maximum_extrapolation_m),
        "selected_mask": largest,
    }
