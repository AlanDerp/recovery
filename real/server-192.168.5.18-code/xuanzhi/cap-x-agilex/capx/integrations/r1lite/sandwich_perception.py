"""RGB-D perception and fail-closed geometry gates for pipeline nine."""

from __future__ import annotations

import base64
import io
from itertools import combinations, product
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import requests
from PIL import Image

from capx.integrations.r1lite.towel_perception import (
    calibration_polygon,
    extrapolation_distance_m,
    pixel_to_table,
)


SCENE_SCHEMA = "capx.r1lite.ninth_sandwich_scene.v1"
OBJECT_ORDER = ("brown_bread", "donut", "mixed_bread")
DEFAULT_PROMPTS: dict[str, tuple[str, ...]] = {
    "brown_bread": ("brown bread slice", "dark brown toast"),
    "donut": ("bagel", "donut", "bread ring", "ring-shaped pastry"),
    "mixed_bread": ("toasted white bread slice", "white bread toast"),
}
NOMINAL_HEIGHTS_M = {
    "brown_bread": 0.015,
    "donut": 0.025,
    "mixed_bread": 0.015,
}


@dataclass(frozen=True)
class PlaneModel:
    normal_base: np.ndarray
    d: float
    rmse_m: float
    inlier_count: int
    candidate_count: int

    def height(self, points_base: np.ndarray) -> np.ndarray:
        points = np.asarray(points_base, dtype=float).reshape(-1, 3)
        return points @ self.normal_base + self.d

    def z_at_xy(self, xy: Sequence[float]) -> float:
        x, y = np.asarray(xy, dtype=float).reshape(2)
        if abs(float(self.normal_base[2])) < 1e-8:
            raise ValueError("table plane is vertical")
        return float(
            -(self.normal_base[0] * x + self.normal_base[1] * y + self.d)
            / self.normal_base[2]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal_base": self.normal_base.tolist(),
            "d": float(self.d),
            "rmse_m": float(self.rmse_m),
            "inlier_count": int(self.inlier_count),
            "candidate_count": int(self.candidate_count),
        }


def _encode_png_base64(rgb: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_mask(item: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(str(item["mask_base64"]))
    mask = np.frombuffer(raw, dtype=np.uint8)
    if mask.size != int(shape[0] * shape[1]):
        raise ValueError("SAM3 mask byte count does not match RGB shape")
    return mask.reshape(shape).astype(bool)


def segment_sandwich_objects(
    rgb: np.ndarray,
    *,
    service_url: str,
    prompts: Mapping[str, Sequence[str]] | None = None,
    minimum_score: float = 0.15,
    timeout_s: float = 120.0,
    allowed_overlap_pairs: set[frozenset[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Segment all three objects using prompt agreement and unique masks."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("sandwich RGB image must have shape HxWx3")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    prompt_map = prompts or DEFAULT_PROMPTS
    payload_image = _encode_png_base64(image)
    session = requests.Session()
    session.trust_env = False
    selected: dict[str, dict[str, Any]] = {}
    for object_name in OBJECT_ORDER:
        candidates_by_prompt: list[list[dict[str, Any]]] = []
        prompt_failures: list[str] = []
        object_minimum_score = (
            min(minimum_score, 0.08) if object_name == "donut" else minimum_score
        )
        object_prompts = tuple(prompt_map.get(object_name, ()))
        if len(object_prompts) < 2:
            raise ValueError(f"{object_name} requires at least two SAM3 prompts")
        for prompt in object_prompts:
            response = session.post(
                f"{service_url.rstrip('/')}/segment",
                json={"image_base64": payload_image, "text_prompt": prompt},
                timeout=timeout_s,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                prompt_failures.append(f"{prompt!r}: no candidate")
                continue
            decoded = [
                {
                    "prompt": prompt,
                    "score": float(item.get("score", 0.0)),
                    "mask": _decode_mask(item, image.shape[:2]),
                }
                for item in results
                if float(item.get("score", 0.0)) >= object_minimum_score
            ]
            if not decoded:
                best_score = max(float(item.get("score", 0.0)) for item in results)
                prompt_failures.append(
                    f"{prompt!r}: {best_score:.3f} < "
                    f"{object_minimum_score:.3f}"
                )
                continue
            candidates_by_prompt.append(
                sorted(decoded, key=lambda item: item["score"], reverse=True)[:8]
            )
        if len(candidates_by_prompt) < 2:
            raise RuntimeError(
                f"SAM3 has fewer than two confident prompts for {object_name}: "
                + "; ".join(prompt_failures)
            )
        matched: list[dict[str, Any]] = []
        for prompt_pair in combinations(candidates_by_prompt, 2):
            for combination in product(*prompt_pair):
                intersection = np.logical_and.reduce(
                    [item["mask"] for item in combination]
                )
                union = np.logical_or.reduce([item["mask"] for item in combination])
                iou = float(np.count_nonzero(intersection)) / max(
                    1, int(np.count_nonzero(union))
                )
                candidate = max(combination, key=lambda item: item["score"])
                geometry = mask_geometry_2d(candidate["mask"])
                centroid_px = geometry["centroid_px"]
                if (
                    float(geometry["area_px"]) < 300.0
                    or float(geometry["area_px"])
                    > image.shape[0] * image.shape[1] * 0.15
                    or float(centroid_px[1]) < image.shape[0] * 0.50
                ):
                    continue
                median_value = float(
                    np.median(
                        hsv_image[..., 2][np.asarray(candidate["mask"], dtype=bool)]
                    )
                )
                if object_name == "brown_bread":
                    semantic_key = (
                        -median_value,
                        -abs(float(geometry["aspect_ratio"]) - 1.3),
                    )
                elif object_name == "mixed_bread":
                    semantic_key = (
                        median_value,
                        -abs(float(geometry["aspect_ratio"]) - 1.3),
                    )
                else:
                    semantic_key = (
                        float(geometry["circularity"]),
                        -abs(float(geometry["aspect_ratio"]) - 1.0),
                    )
                matched.append(
                    {
                        "iou": iou,
                        "minimum_score": min(
                            float(item["score"]) for item in combination
                        ),
                        "combination": combination,
                        "semantic_key": semantic_key,
                    }
                )
        eligible = [item for item in matched if item["iou"] >= 0.35]
        if not eligible:
            best_iou = max(
                (item["iou"] for item in matched),
                default=0.0,
            )
            raise RuntimeError(
                f"SAM3 prompts disagree for {object_name}: IoU={best_iou:.3f}"
            )
        match = max(
            eligible,
            key=lambda item: (
                item["semantic_key"],
                item["iou"],
                item["minimum_score"],
            ),
        )
        prompt_iou = float(match["iou"])
        candidates = match["combination"]
        if prompt_iou < 0.35:
            raise RuntimeError(
                f"SAM3 prompts disagree for {object_name}: IoU={prompt_iou:.3f}"
            )
        best = max(candidates, key=lambda item: item["score"])
        selected[object_name] = {
            "mask": np.asarray(best["mask"], dtype=bool),
            "prompt": best["prompt"],
            "score": float(best["score"]),
            "prompt_iou": prompt_iou,
            "prompt_scores": {
                str(item["prompt"]): float(item["score"]) for item in candidates
            },
        }
    validate_segmentation_masks(
        image,
        selected,
        allowed_overlap_pairs=allowed_overlap_pairs,
    )
    return selected


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("segmentation mask has no external contour")
    return max(contours, key=cv2.contourArea)


def _hole_ratio(mask: np.ndarray) -> float:
    contours, hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours or hierarchy is None:
        return 0.0
    hierarchy = hierarchy[0]
    outer_area = sum(
        cv2.contourArea(contour)
        for contour, node in zip(contours, hierarchy)
        if int(node[3]) < 0
    )
    hole_area = sum(
        cv2.contourArea(contour)
        for contour, node in zip(contours, hierarchy)
        if int(node[3]) >= 0
    )
    return 0.0 if outer_area <= 0 else float(hole_area / outer_area)


def mask_geometry_2d(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    contour = _largest_contour(binary)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    (center, size, angle_deg) = cv2.minAreaRect(contour)
    width_px, length_px = sorted((float(size[0]), float(size[1])))
    circularity = (
        0.0 if perimeter <= 0 else float(4.0 * np.pi * area / perimeter**2)
    )
    return {
        "centroid_px": [float(center[0]), float(center[1])],
        "area_px": area,
        "width_px": width_px,
        "length_px": length_px,
        "aspect_ratio": length_px / max(width_px, 1e-9),
        "angle_deg": float(angle_deg),
        "circularity": circularity,
        "hole_ratio": _hole_ratio(binary),
        "contour_px": contour.reshape(-1, 2).astype(float).tolist(),
    }


def validate_segmentation_masks(
    rgb: np.ndarray,
    segmentations: Mapping[str, Mapping[str, Any]],
    *,
    minimum_area_px: int = 300,
    maximum_pair_iou: float = 0.15,
    border_margin_px: int = 4,
    allowed_overlap_pairs: set[frozenset[str]] | None = None,
) -> None:
    """Reject missing, clipped, overlapping, or implausible object masks."""
    image = np.asarray(rgb)
    if set(segmentations) != set(OBJECT_ORDER):
        raise ValueError("sandwich scene must contain exactly three object masks")
    masks: dict[str, np.ndarray] = {}
    geometries: dict[str, dict[str, Any]] = {}
    for name in OBJECT_ORDER:
        mask = np.asarray(segmentations[name]["mask"], dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{name} mask shape does not match RGB")
        area = int(np.count_nonzero(mask))
        if area < minimum_area_px:
            raise ValueError(f"{name} mask is too small: {area}px")
        ys, xs = np.nonzero(mask)
        if (
            int(xs.min()) < border_margin_px
            or int(ys.min()) < border_margin_px
            or int(xs.max()) >= image.shape[1] - border_margin_px
            or int(ys.max()) >= image.shape[0] - border_margin_px
        ):
            raise ValueError(f"{name} mask touches the image boundary")
        masks[name] = mask
        geometries[name] = mask_geometry_2d(mask)
    for index, first in enumerate(OBJECT_ORDER):
        for second in OBJECT_ORDER[index + 1 :]:
            intersection = int(np.count_nonzero(masks[first] & masks[second]))
            union = int(np.count_nonzero(masks[first] | masks[second]))
            iou = intersection / max(1, union)
            pair = frozenset((first, second))
            if iou > maximum_pair_iou and pair not in (
                allowed_overlap_pairs or set()
            ):
                raise ValueError(
                    f"{first}/{second} masks overlap ambiguously: IoU={iou:.3f}"
                )
    donut = geometries["donut"]
    if donut["circularity"] < 0.45 or donut["aspect_ratio"] > 1.8:
        raise ValueError("donut mask is not sufficiently ring-like")
    for name in ("brown_bread", "mixed_bread"):
        geometry = geometries[name]
        if not 1.0 <= geometry["aspect_ratio"] <= 2.2:
            raise ValueError(f"{name} mask has implausible bread aspect ratio")
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    brown_value = float(np.median(hsv[..., 2][masks["brown_bread"]]))
    mixed_value = float(np.median(hsv[..., 2][masks["mixed_bread"]]))
    if brown_value >= mixed_value - 4.0:
        raise ValueError(
            "brown and mixed bread color ordering is ambiguous "
            f"(V={brown_value:.1f}/{mixed_value:.1f})"
        )


def _depth_points_camera(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pixel_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(depth, dtype=float)
    intrinsics = np.asarray(intrinsics, dtype=float).reshape(3, 3)
    mask = np.asarray(pixel_mask, dtype=bool)
    valid = mask & np.isfinite(values) & (values >= 0.20) & (values <= 3.0)
    ys, xs = np.nonzero(valid)
    z = values[ys, xs]
    x = (xs - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.column_stack((x, y, z)), np.column_stack((xs, ys))


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    homogeneous = np.column_stack((values, np.ones(len(values))))
    return (matrix @ homogeneous.T).T[:, :3]


def fit_table_plane(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    *,
    excluded_mask: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    workspace_x: tuple[float, float] = (0.15, 0.85),
    workspace_y: tuple[float, float] = (-0.65, 0.65),
    workspace_z: tuple[float, float] = (0.55, 1.20),
    ransac_threshold_m: float = 0.008,
) -> PlaneModel:
    """Fit the current tabletop from depth while excluding segmented objects."""
    values = np.asarray(depth, dtype=float)
    sample = np.zeros(values.shape, dtype=bool)
    step = max(4, int(max(values.shape) // 160))
    sample[::step, ::step] = True
    if excluded_mask is not None:
        sample &= ~np.asarray(excluded_mask, dtype=bool)
    if roi_mask is not None:
        sample &= np.asarray(roi_mask, dtype=bool)
    camera_points, _ = _depth_points_camera(values, intrinsics, sample)
    base_points = _transform_points(camera_points, base_from_camera)
    candidates = base_points[
        (base_points[:, 0] >= workspace_x[0])
        & (base_points[:, 0] <= workspace_x[1])
        & (base_points[:, 1] >= workspace_y[0])
        & (base_points[:, 1] <= workspace_y[1])
        & (base_points[:, 2] >= workspace_z[0])
        & (base_points[:, 2] <= workspace_z[1])
    ]
    if len(candidates) < 200:
        raise ValueError(f"insufficient table depth points: {len(candidates)}")
    rng = np.random.default_rng(42)
    best: tuple[int, np.ndarray] | None = None
    for _ in range(800):
        points = candidates[rng.choice(len(candidates), size=3, replace=False)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(float(normal[2])) < 0.80:
            continue
        d = -float(np.dot(normal, points[0]))
        inliers = np.abs(candidates @ normal + d) <= ransac_threshold_m
        score = int(np.count_nonzero(inliers))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0] < 150:
        raise ValueError("could not fit a stable near-horizontal table plane")
    inlier_points = candidates[best[1]]
    centroid = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    residual = inlier_points @ normal + d
    rmse = float(np.sqrt(np.mean(residual**2)))
    if rmse > 0.008 or float(normal[2]) < 0.90:
        raise ValueError(
            f"table plane quality failed: rmse={rmse:.4f}, nz={normal[2]:.3f}"
        )
    return PlaneModel(normal, d, rmse, len(inlier_points), len(candidates))


def _principal_geometry(
    points_base: np.ndarray,
    plane: PlaneModel,
) -> dict[str, Any]:
    points = np.asarray(points_base, dtype=float).reshape(-1, 3)
    tangent_x = np.array([1.0, 0.0, 0.0])
    tangent_x -= plane.normal_base * float(np.dot(tangent_x, plane.normal_base))
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(plane.normal_base, tangent_x)
    uv = np.column_stack((points @ tangent_x, points @ tangent_y))
    center_uv = np.median(uv, axis=0)
    _, _, vh = np.linalg.svd(uv - center_uv, full_matrices=False)
    major_uv = vh[0]
    major_base = tangent_x * major_uv[0] + tangent_y * major_uv[1]
    major_base[2] = 0.0
    major_base /= np.linalg.norm(major_base)
    if major_base[0] < 0 or (
        abs(float(major_base[0])) < 1e-9 and major_base[1] < 0
    ):
        major_base = -major_base
    minor_base = np.array([-major_base[1], major_base[0], 0.0])
    major_projection = points @ major_base
    minor_projection = points @ minor_base
    length = float(np.percentile(major_projection, 98) - np.percentile(major_projection, 2))
    width = float(np.percentile(minor_projection, 98) - np.percentile(minor_projection, 2))
    if width > length:
        length, width = width, length
        major_base, minor_base = minor_base, -major_base
    return {
        "major_axis_base_xy": major_base[:2].tolist(),
        "minor_axis_base_xy": minor_base[:2].tolist(),
        "length_m": length,
        "width_m": width,
        "yaw_rad": float(np.arctan2(major_base[1], major_base[0])),
    }


def extract_object_geometry(
    name: str,
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    plane: PlaneModel,
    *,
    minimum_foreground_height_m: float = 0.002,
) -> dict[str, Any]:
    camera_points, pixels = _depth_points_camera(depth, intrinsics, mask)
    if len(camera_points) < 80:
        raise ValueError(f"{name} has insufficient valid depth pixels")
    base_points = _transform_points(camera_points, base_from_camera)
    heights = plane.height(base_points)
    foreground = heights >= minimum_foreground_height_m
    depth_hole_ratio = None
    if name == "donut":
        centroid = np.asarray(mask_geometry_2d(mask)["centroid_px"], dtype=float)
        radial = np.linalg.norm(pixels.astype(float) - centroid, axis=1)
        outer_radius = float(np.percentile(radial, 95.0))
        ys, xs = np.ogrid[: mask.shape[0], : mask.shape[1]]
        central_mask = (
            (xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2
            <= (0.35 * outer_radius) ** 2
        )
        central_camera, _ = _depth_points_camera(
            depth,
            intrinsics,
            central_mask,
        )
        if len(central_camera) < 20:
            raise ValueError("donut has insufficient center depth evidence")
        central_base = _transform_points(central_camera, base_from_camera)
        central_foreground = (
            plane.height(central_base) >= minimum_foreground_height_m
        )
        depth_hole_ratio = float(np.mean(~central_foreground))
        if depth_hole_ratio < 0.10:
            raise ValueError(
                "donut depth does not show a reliable center hole: "
                f"{depth_hole_ratio:.3f}"
            )
    if int(np.count_nonzero(foreground)) < 50:
        raise ValueError(f"{name} has insufficient points above the table")
    points = base_points[foreground]
    object_heights = heights[foreground]
    top_height = float(np.percentile(object_heights, 85.0))
    bottom_height = max(0.0, float(np.percentile(object_heights, 5.0)))
    thickness = top_height - bottom_height
    if not 0.003 <= top_height <= 0.080:
        raise ValueError(f"{name} measured height is implausible: {top_height:.4f}m")
    xy = np.median(points[:, :2], axis=0)
    table_z = plane.z_at_xy(xy)
    center = np.array([xy[0], xy[1], table_z + top_height / 2.0])
    geometry = _principal_geometry(points, plane)
    if not 0.025 <= geometry["width_m"] <= 0.15:
        raise ValueError(f"{name} measured width is implausible")
    if not geometry["width_m"] <= geometry["length_m"] <= 0.20:
        raise ValueError(f"{name} measured length is implausible")
    pixel_geometry = mask_geometry_2d(mask)
    return {
        "name": name,
        "center_base_m": center.tolist(),
        "table_z_at_center_m": table_z,
        "top_z_base_m": table_z + top_height,
        "height_above_table_m": top_height,
        "thickness_m": max(thickness, 0.003),
        "valid_depth_points": int(len(points)),
        "depth_hole_ratio": depth_hole_ratio,
        "pixel_geometry": pixel_geometry,
        **geometry,
        "depth_pixel_bounds": [
            [int(np.min(pixels[:, 0])), int(np.min(pixels[:, 1]))],
            [int(np.max(pixels[:, 0])), int(np.max(pixels[:, 1]))],
        ],
    }


def perceive_sandwich_scene(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    segmentations: Mapping[str, Mapping[str, Any]],
    *,
    frame_id: str,
    stamp_ns: int,
    allowed_overlap_pairs: set[frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Build one immutable RGB-D scene observation."""
    image = np.asarray(rgb, dtype=np.uint8)
    validate_segmentation_masks(
        image,
        segmentations,
        allowed_overlap_pairs=allowed_overlap_pairs,
    )
    union = np.logical_or.reduce(
        [np.asarray(segmentations[name]["mask"], dtype=bool) for name in OBJECT_ORDER]
    )
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        raise ValueError("sandwich masks contain no pixels")
    padding = max(40, int(max(image.shape[:2]) * 0.05))
    roi = np.zeros(image.shape[:2], dtype=bool)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(image.shape[1], int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(image.shape[0], int(ys.max()) + padding + 1)
    roi[y0:y1, x0:x1] = True
    plane = fit_table_plane(
        depth,
        intrinsics,
        base_from_camera,
        excluded_mask=union,
        roi_mask=roi,
    )
    objects = {
        name: {
            **extract_object_geometry(
                name,
                np.asarray(segmentations[name]["mask"], dtype=bool),
                depth,
                intrinsics,
                base_from_camera,
                plane,
            ),
            "sam3": {
                key: value
                for key, value in segmentations[name].items()
                if key != "mask"
            },
        }
        for name in OBJECT_ORDER
    }
    return {
        "schema": SCENE_SCHEMA,
        "frame_id": str(frame_id),
        "stamp_ns": int(stamp_ns),
        "image_shape": list(image.shape[:2]),
        "table_plane": plane.to_dict(),
        "objects": objects,
    }


def validate_four_point_homography_calibration(
    calibration: Mapping[str, Any],
    *,
    maximum_control_point_error_m: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Validate an exact four-correspondence image-to-table homography."""
    value = dict(calibration)
    if value.get("type") != "image_to_base_plane_homography":
        raise ValueError("pipeline nine requires an image-to-base plane homography")
    points = value.get("points")
    if (
        value.get("num_points") != 4
        or not isinstance(points, list)
        or len(points) != 4
    ):
        raise ValueError(
            "pipeline nine requires exactly four homography correspondences"
        )
    try:
        source = np.asarray(
            [point["pixel"] for point in points], dtype=float
        ).reshape(4, 2)
        destination = np.asarray(
            [point["base_xy"] for point in points], dtype=float
        ).reshape(4, 2)
        homography = np.asarray(
            value["image_to_base_xy_homography"], dtype=float
        ).reshape(3, 3)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("four-point homography calibration is malformed") from exc
    if (
        not np.isfinite(source).all()
        or not np.isfinite(destination).all()
        or not np.isfinite(homography).all()
    ):
        raise ValueError("four-point homography calibration must be finite")
    source_area = float(cv2.contourArea(cv2.convexHull(source.astype(np.float32))))
    destination_area = float(
        cv2.contourArea(cv2.convexHull(destination.astype(np.float32)))
    )
    if source_area <= 1.0 or destination_area <= 1e-8:
        raise ValueError("four-point homography correspondences are degenerate")
    if np.linalg.matrix_rank(homography) != 3:
        raise ValueError("four-point homography matrix is singular")
    homogeneous = np.column_stack((source, np.ones(4))) @ homography.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("four-point homography projects a control point to infinity")
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    maximum_error = float(
        np.max(np.linalg.norm(projected - destination, axis=1))
    )
    if maximum_error > maximum_control_point_error_m:
        raise ValueError(
            "four-point homography does not reproduce its control points: "
            f"{maximum_error:.6f}m > {maximum_control_point_error_m:.6f}m"
        )
    return homography, maximum_error


def perceive_sandwich_scene_homography(
    rgb: np.ndarray,
    segmentations: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
    *,
    phase: str,
    maximum_required_extrapolation_m: float = 0.020,
    allowed_overlap_pairs: set[frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Recover planar sandwich geometry from the calibrated head homography.

    Heights intentionally use conservative, explicit prop dimensions; a planar
    homography cannot measure object thickness.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    validate_segmentation_masks(
        image,
        segmentations,
        allowed_overlap_pairs=allowed_overlap_pairs,
    )
    calibration_value = dict(calibration)
    image_size = np.asarray(
        calibration_value.get("image_size"), dtype=float
    ).reshape(2)
    if np.any(image_size <= 0):
        raise ValueError("homography calibration image size is invalid")
    scale = image_size / np.array([image.shape[1], image.shape[0]], dtype=float)
    if not np.isclose(scale[0], scale[1], rtol=0.02):
        raise ValueError("camera image and homography have inconsistent aspect ratios")
    homography, maximum_control_point_error = (
        validate_four_point_homography_calibration(calibration_value)
    )
    plane_z = float(calibration_value["plane_z"])
    required_objects = (
        {"brown_bread", "donut"}
        if phase == "donut_on_brown"
        else set(OBJECT_ORDER)
    )
    objects: dict[str, dict[str, Any]] = {}
    for name in OBJECT_ORDER:
        pixel_geometry = mask_geometry_2d(segmentations[name]["mask"])
        center_pixel = np.asarray(
            pixel_geometry["centroid_px"], dtype=float
        ) * scale
        contour_pixel = np.asarray(
            pixel_geometry["contour_px"], dtype=float
        ) * scale
        center_base = pixel_to_table(homography, center_pixel, plane_z)
        projected = np.asarray(
            [
                pixel_to_table(homography, point, plane_z)[:2]
                for point in contour_pixel
            ],
            dtype=float,
        )
        centered = projected - np.median(projected, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        major = vh[0]
        if major[0] < 0 or (
            abs(float(major[0])) < 1e-9 and major[1] < 0
        ):
            major = -major
        minor = np.array([-major[1], major[0]])
        major_values = projected @ major
        minor_values = projected @ minor
        length = float(
            np.percentile(major_values, 98)
            - np.percentile(major_values, 2)
        )
        width = float(
            np.percentile(minor_values, 98)
            - np.percentile(minor_values, 2)
        )
        if width > length:
            length, width = width, length
            major, minor = minor, -major
        extrapolation = extrapolation_distance_m(
            center_pixel,
            calibration_value,
        )
        if (
            name in required_objects
            and extrapolation > maximum_required_extrapolation_m
        ):
            raise ValueError(
                f"{name} lies {extrapolation:.4f}m outside the calibrated "
                f"homography hull (limit {maximum_required_extrapolation_m:.4f}m)"
            )
        height = float(NOMINAL_HEIGHTS_M[name])
        support_offset = (
            NOMINAL_HEIGHTS_M["brown_bread"]
            if phase == "mixed_bread_on_stack" and name == "donut"
            else 0.0
        )
        bottom_z = plane_z + float(support_offset)
        object_center = center_base.copy()
        object_center[2] = bottom_z + height / 2.0
        objects[name] = {
            "name": name,
            "center_base_m": object_center.tolist(),
            "table_z_at_center_m": plane_z,
            "top_z_base_m": bottom_z + height,
            "height_above_table_m": height + float(support_offset),
            "thickness_m": height,
            "major_axis_base_xy": major.tolist(),
            "minor_axis_base_xy": minor.tolist(),
            "length_m": length,
            "width_m": width,
            "yaw_rad": float(np.arctan2(major[1], major[0])),
            "pixel_geometry": pixel_geometry,
            "calibration_pixel_uv": center_pixel.tolist(),
            "calibration_extrapolation_m": extrapolation,
            "height_source": "configured_prop_dimension",
            "sam3": {
                key: value
                for key, value in segmentations[name].items()
                if key != "mask"
            },
        }
    return {
        "schema": SCENE_SCHEMA,
        "frame_id": str(calibration_value.get("source", "r1lite_head_left")),
        "stamp_ns": 0,
        "image_shape": list(image.shape[:2]),
        "perception_method": "sam3_plus_calibrated_plane_homography",
        "homography_model": {
            "type": "exact_four_point",
            "correspondence_count": 4,
            "maximum_control_point_error_m": maximum_control_point_error,
        },
        "table_plane": {
            "normal_base": [0.0, 0.0, 1.0],
            "d": -plane_z,
            "rmse_m": None,
            "inlier_count": 0,
            "candidate_count": 0,
            "plane_z_m": plane_z,
            "calibration_polygon_px": calibration_polygon(
                calibration_value
            ).tolist(),
            "maximum_required_extrapolation_m": (
                maximum_required_extrapolation_m
            ),
        },
        "objects": objects,
    }


def scene_repeatability_error(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    if first.get("schema") != SCENE_SCHEMA or second.get("schema") != SCENE_SCHEMA:
        raise ValueError("repeatability requires pipeline-nine scene objects")
    per_object: dict[str, float] = {}
    for name in OBJECT_ORDER:
        first_center = np.asarray(first["objects"][name]["center_base_m"], dtype=float)
        second_center = np.asarray(second["objects"][name]["center_base_m"], dtype=float)
        per_object[name] = float(np.linalg.norm(first_center - second_center))
    maximum = max(per_object.values())
    return {"per_object_m": per_object, "maximum_m": maximum}


def require_repeatable_scene(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    maximum_center_drift_m: float = 0.012,
) -> dict[str, Any]:
    result = scene_repeatability_error(first, second)
    if result["maximum_m"] > maximum_center_drift_m:
        raise ValueError(
            "two-frame sandwich perception drift exceeded limit: "
            f"{result['maximum_m']:.6f}m > {maximum_center_drift_m:.6f}m"
        )
    return result


def support_contains_center(
    support: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    margin_m: float = 0.008,
) -> dict[str, Any]:
    """Check that an item's center lies inside the support's oriented footprint."""
    support_center = np.asarray(support["center_base_m"], dtype=float)[:2]
    item_center = np.asarray(item["center_base_m"], dtype=float)[:2]
    major = np.asarray(support["major_axis_base_xy"], dtype=float)
    minor = np.asarray(support["minor_axis_base_xy"], dtype=float)
    delta = item_center - support_center
    major_error = abs(float(np.dot(delta, major)))
    minor_error = abs(float(np.dot(delta, minor)))
    major_limit = float(support["length_m"]) / 2.0 - margin_m
    minor_limit = float(support["width_m"]) / 2.0 - margin_m
    passed = major_error <= major_limit and minor_error <= minor_limit
    return {
        "passed": passed,
        "center_error_m": float(np.linalg.norm(delta)),
        "major_error_m": major_error,
        "minor_error_m": minor_error,
        "major_limit_m": major_limit,
        "minor_limit_m": minor_limit,
    }
