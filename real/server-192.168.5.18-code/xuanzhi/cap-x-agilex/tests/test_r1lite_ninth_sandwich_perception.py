import numpy as np
import pytest

from capx.integrations.r1lite.sandwich_perception import (
    OBJECT_ORDER,
    perceive_sandwich_scene,
    perceive_sandwich_scene_homography,
    require_repeatable_scene,
    support_contains_center,
    validate_four_point_homography_calibration,
    validate_segmentation_masks,
)


def synthetic_observation():
    height, width = 240, 320
    rgb = np.full((height, width, 3), 220, dtype=np.uint8)
    masks = {name: np.zeros((height, width), dtype=bool) for name in OBJECT_ORDER}
    masks["brown_bread"][70:120, 45:105] = True
    rgb[masks["brown_bread"]] = [85, 50, 25]
    yy, xx = np.ogrid[:height, :width]
    outer = (xx - 160) ** 2 + (yy - 95) ** 2 <= 23**2
    inner = (xx - 160) ** 2 + (yy - 95) ** 2 < 9**2
    masks["donut"] = outer & ~inner
    rgb[masks["donut"]] = [190, 130, 55]
    masks["mixed_bread"][145:195, 205:265] = True
    rgb[masks["mixed_bread"]] = [245, 220, 95]
    depth = np.full((height, width), 0.700, dtype=float)
    depth[masks["brown_bread"]] = 0.684
    depth[masks["donut"]] = 0.675
    depth[masks["mixed_bread"]] = 0.681
    intrinsics = np.array(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    )
    base_from_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.45],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.50],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    segmentations = {
        name: {
            "mask": mask,
            "prompt": name,
            "score": 0.9,
            "prompt_iou": 0.8,
        }
        for name, mask in masks.items()
    }
    return rgb, depth, intrinsics, base_from_camera, segmentations


def four_point_calibration():
    points = [
        {"pixel": [0.0, 0.0], "base_xy": [0.30, 0.20]},
        {"pixel": [320.0, 0.0], "base_xy": [0.30, -0.12]},
        {"pixel": [320.0, 240.0], "base_xy": [0.54, -0.12]},
        {"pixel": [0.0, 240.0], "base_xy": [0.54, 0.20]},
    ]
    return {
        "type": "image_to_base_plane_homography",
        "source": "r1lite_head_left",
        "plane_z": 0.80,
        "image_size": [320, 240],
        "num_points": 4,
        "points": points,
        "image_to_base_xy_homography": [
            [0.0, 0.001, 0.30],
            [-0.001, 0.0, 0.20],
            [0.0, 0.0, 1.0],
        ],
    }


def test_rgbd_scene_recovers_three_objects_and_table():
    rgb, depth, intrinsics, transform, segmentations = synthetic_observation()
    scene = perceive_sandwich_scene(
        rgb,
        depth,
        intrinsics,
        transform,
        segmentations,
        frame_id="head_left",
        stamp_ns=123,
    )

    assert set(scene["objects"]) == set(OBJECT_ORDER)
    assert scene["table_plane"]["normal_base"][2] == pytest.approx(1.0, abs=0.02)
    assert scene["table_plane"]["rmse_m"] < 0.002
    assert scene["objects"]["donut"]["height_above_table_m"] == pytest.approx(
        0.025, abs=0.004
    )
    assert scene["objects"]["donut"]["pixel_geometry"]["hole_ratio"] > 0.05


def test_four_point_homography_recovers_planar_scene():
    rgb, _, _, _, segmentations = synthetic_observation()
    scene = perceive_sandwich_scene_homography(
        rgb,
        segmentations,
        four_point_calibration(),
        phase="donut_on_brown",
    )

    assert scene["perception_method"] == "sam3_plus_calibrated_plane_homography"
    assert scene["homography_model"]["type"] == "exact_four_point"
    assert scene["homography_model"]["correspondence_count"] == 4
    assert scene["homography_model"]["maximum_control_point_error_m"] < 1e-9
    assert scene["objects"]["donut"]["center_base_m"][:2] == pytest.approx(
        [0.395, 0.04], abs=0.002
    )


def test_four_point_homography_rejects_extra_or_inconsistent_points():
    calibration = four_point_calibration()
    calibration["num_points"] = 5
    calibration["points"].append(
        {"pixel": [160.0, 120.0], "base_xy": [0.42, 0.04]}
    )
    with pytest.raises(ValueError, match="exactly four"):
        validate_four_point_homography_calibration(calibration)

    calibration = four_point_calibration()
    calibration["image_to_base_xy_homography"][0][2] += 0.01
    with pytest.raises(ValueError, match="does not reproduce"):
        validate_four_point_homography_calibration(calibration)


def test_masks_fail_closed_on_missing_hole_overlap_and_color_ambiguity():
    rgb, depth, intrinsics, transform, segmentations = synthetic_observation()
    no_hole = {
        name: {**item, "mask": item["mask"].copy()}
        for name, item in segmentations.items()
    }
    yy, xx = np.ogrid[: rgb.shape[0], : rgb.shape[1]]
    filled_donut = (xx - 160) ** 2 + (yy - 95) ** 2 <= 23**2
    no_hole["donut"]["mask"] = filled_donut
    no_hole_depth = depth.copy()
    no_hole_depth[filled_donut] = 0.675
    with pytest.raises(ValueError, match="center hole"):
        perceive_sandwich_scene(
            rgb,
            no_hole_depth,
            intrinsics,
            transform,
            no_hole,
            frame_id="head_left",
            stamp_ns=123,
        )

    overlapping = {
        name: {**item, "mask": item["mask"].copy()}
        for name, item in segmentations.items()
    }
    overlapping["mixed_bread"]["mask"] = overlapping["brown_bread"]["mask"].copy()
    with pytest.raises(ValueError, match="overlap"):
        validate_segmentation_masks(rgb, overlapping)

    ambiguous_rgb = rgb.copy()
    ambiguous_rgb[segmentations["brown_bread"]["mask"]] = [245, 220, 95]
    with pytest.raises(ValueError, match="color ordering"):
        validate_segmentation_masks(ambiguous_rgb, segmentations)


def test_repeatability_and_support_gates_are_fail_closed():
    rgb, depth, intrinsics, transform, segmentations = synthetic_observation()
    scene = perceive_sandwich_scene(
        rgb,
        depth,
        intrinsics,
        transform,
        segmentations,
        frame_id="head_left",
        stamp_ns=123,
    )
    repeated = {**scene, "objects": {k: dict(v) for k, v in scene["objects"].items()}}
    repeated["objects"]["donut"]["center_base_m"] = list(
        repeated["objects"]["donut"]["center_base_m"]
    )
    repeated["objects"]["donut"]["center_base_m"][0] += 0.020
    with pytest.raises(ValueError, match="drift exceeded"):
        require_repeatable_scene(scene, repeated)

    brown = dict(scene["objects"]["brown_bread"])
    donut = dict(scene["objects"]["donut"])
    donut["center_base_m"] = list(brown["center_base_m"])
    assert support_contains_center(brown, donut)["passed"] is True
    donut["center_base_m"][0] += 0.20
    assert support_contains_center(brown, donut)["passed"] is False
