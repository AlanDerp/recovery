import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from r1lite_collect_bimanual_head_plane_points import (  # noqa: E402
    BOARD_INNER_COLUMNS,
    BOARD_INNER_ROWS,
    BOARD_SQUARE_COLUMNS,
    BOARD_SQUARE_ROWS,
    BOARD_SQUARE_SIZE_M,
    OPEN_RIGHT_FINGERTIP_CENTER_GRIPPER_M,
    _quality_gate,
    contact_position_from_gripper_pose,
    detect_chessboard,
    solve_bimanual_calibration,
    solve_chessboard_calibration,
)


def _true_xy(pixel):
    u, v = pixel
    return np.array([0.20 + 0.0004 * u, -0.35 + 0.0003 * v])


def _record(point_id, group, side, pixel, bias):
    xy = _true_xy(pixel) + bias[:2]
    return {
        "point_id": point_id,
        "group": group,
        "side": side,
        "pixel": list(pixel),
        "base_xyz": [float(xy[0]), float(xy[1]), float(0.84 + bias[2])],
        "pose_source": "synthetic",
        "pose_spread_m": 0.0,
    }


def synthetic_payload():
    left_bias = np.array([0.010, -0.006, 0.004])
    right_bias = -left_bias
    records = []
    for index, pixel in enumerate(
        [(150, 1100), (250, 800), (400, 500), (600, 300)], start=1
    ):
        records.append(
            _record(f"left_only_{index:02d}", "left_only", "left", pixel, left_bias)
        )
    for index, pixel in enumerate(
        [(1750, 1100), (1650, 800), (1500, 500), (1300, 300)], start=1
    ):
        records.append(
            _record(
                f"right_only_{index:02d}",
                "right_only",
                "right",
                pixel,
                right_bias,
            )
        )
    for index, pixel in enumerate(
        [(800, 1000), (960, 700), (1120, 400)], start=1
    ):
        point_id = f"overlap_{index:02d}"
        records.append(_record(point_id, "overlap", "left", pixel, left_bias))
        records.append(_record(point_id, "overlap", "right", pixel, right_bias))
    return {
        "image_size": [1920, 1536],
        "image_path": "synthetic.png",
        "records": records,
    }


def test_solver_recovers_zero_mean_arm_bias_and_wide_homography():
    calibration, audit = solve_bimanual_calibration(
        synthetic_payload(),
        ransac_threshold_m=0.001,
    )

    assert audit["arm_bias_xyz_m"]["left"] == pytest.approx(
        [0.010, -0.006, 0.004], abs=1e-9
    )
    assert audit["arm_bias_xyz_m"]["right"] == pytest.approx(
        [-0.010, 0.006, -0.004], abs=1e-9
    )
    assert audit["overall_rmse_m"] < 1e-7
    assert audit["overlap_corrected_rmse_m"] < 1e-8
    assert audit["coverage_fraction_of_image"] > 0.20
    assert calibration["plane_z"] == pytest.approx(0.84)

    h = np.asarray(calibration["image_to_base_xy_homography"])
    query = np.array([1200.0, 900.0, 1.0])
    predicted = h @ query
    predicted = predicted[:2] / predicted[2]
    assert predicted == pytest.approx(_true_xy(query[:2]), abs=1e-7)


def test_quality_gate_rejects_bad_calibration():
    _, audit = solve_bimanual_calibration(
        synthetic_payload(),
        ransac_threshold_m=0.001,
    )
    audit["per_arm"]["left"]["rmse_m"] = 0.020

    with pytest.raises(RuntimeError, match="left RMSE"):
        _quality_gate(
            audit,
            maximum_rmse_m=0.015,
            maximum_error_m=0.030,
            maximum_overlap_rmse_m=0.015,
        )


def test_detects_14_by_18_square_board_as_13_by_17_inner_corners():
    square_pixels = 40
    board = np.zeros(
        (
            BOARD_SQUARE_ROWS * square_pixels,
            BOARD_SQUARE_COLUMNS * square_pixels,
            3,
        ),
        dtype=np.uint8,
    )
    for row in range(BOARD_SQUARE_ROWS):
        for column in range(BOARD_SQUARE_COLUMNS):
            value = 255 if (row + column) % 2 == 0 else 0
            board[
                row * square_pixels : (row + 1) * square_pixels,
                column * square_pixels : (column + 1) * square_pixels,
            ] = value
    image = np.full((700, 900, 3), 127, dtype=np.uint8)
    image[60 : 60 + board.shape[0], 80 : 80 + board.shape[1]] = board

    corners = detect_chessboard(image)

    assert len(corners) == BOARD_INNER_ROWS * BOARD_INNER_COLUMNS
    assert corners[0]["board_xy_m"] == pytest.approx([0.0, 0.0])
    assert corners[-1]["board_xy_m"] == pytest.approx(
        [
            (BOARD_INNER_COLUMNS - 1) * BOARD_SQUARE_SIZE_M,
            (BOARD_INNER_ROWS - 1) * BOARD_SQUARE_SIZE_M,
        ]
    )


def test_open_finger_link2_contact_offset_is_transformed_by_gripper_pose():
    identity_contact = contact_position_from_gripper_pose(
        [0.4, -0.2, 0.9],
        [0.0, 0.0, 0.0, 1.0],
        OPEN_RIGHT_FINGERTIP_CENTER_GRIPPER_M,
    )
    assert identity_contact == pytest.approx(
        np.array([0.4, -0.2, 0.9])
        + np.array([0.078355, -0.03325007, -0.004453985])
    )

    quarter_turn_z = np.sqrt(0.5)
    rotated_contact = contact_position_from_gripper_pose(
        [0.4, -0.2, 0.9],
        [0.0, 0.0, quarter_turn_z, quarter_turn_z],
        OPEN_RIGHT_FINGERTIP_CENTER_GRIPPER_M,
    )
    assert rotated_contact == pytest.approx(
        [0.43325007, -0.121645, 0.895546015],
        abs=1e-9,
    )


def test_chessboard_solver_uses_touch_points_and_all_dense_corners():
    angle = np.radians(24.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    translation = np.array([0.35, -0.28])
    left_bias = np.array([0.008, -0.004, 0.003])
    right_bias = -left_bias
    corners = []
    for row in range(BOARD_INNER_ROWS):
        for column in range(BOARD_INNER_COLUMNS):
            board_xy = np.array(
                [column * BOARD_SQUARE_SIZE_M, row * BOARD_SQUARE_SIZE_M]
            )
            corners.append(
                {
                    "corner_id": f"r{row:02d}_c{column:02d}",
                    "row": row,
                    "column": column,
                    "pixel": [120.0 + column * 35.0, 90.0 + row * 34.0],
                    "board_xy_m": board_xy.tolist(),
                }
            )

    def touch(point_id, group, side, row, column, bias):
        board_xy = np.array(
            [column * BOARD_SQUARE_SIZE_M, row * BOARD_SQUARE_SIZE_M]
        )
        base_xy = board_xy @ rotation + translation + bias[:2]
        return {
            "point_id": point_id,
            "group": group,
            "side": side,
            "pixel": [120.0 + column * 35.0, 90.0 + row * 34.0],
            "board_xy_m": board_xy.tolist(),
            "base_xyz": [base_xy[0], base_xy[1], 0.84 + bias[2]],
            "pose_source": "synthetic",
            "pose_spread_m": 0.0,
        }

    records = [
        touch("left_01", "left_only", "left", 0, 0, left_bias),
        touch("left_02", "left_only", "left", 6, 1, left_bias),
        touch("left_03", "left_only", "left", 12, 2, left_bias),
        touch("right_01", "right_only", "right", 0, 16, right_bias),
        touch("right_02", "right_only", "right", 6, 15, right_bias),
        touch("right_03", "right_only", "right", 12, 14, right_bias),
    ]
    for index, (row, column) in enumerate([(1, 8), (6, 8), (11, 8)], start=1):
        point_id = f"overlap_{index:02d}"
        records.append(
            touch(point_id, "overlap", "left", row, column, left_bias)
        )
        records.append(
            touch(point_id, "overlap", "right", row, column, right_bias)
        )
    payload = {
        "image_size": [900, 700],
        "image_path": "synthetic_board.png",
        "board": {
            "square_rows": BOARD_SQUARE_ROWS,
            "square_columns": BOARD_SQUARE_COLUMNS,
            "square_size_m": BOARD_SQUARE_SIZE_M,
        },
        "detected_corners": corners,
        "records": records,
    }

    calibration, audit = solve_chessboard_calibration(
        payload,
        ransac_threshold_m=0.001,
    )

    assert calibration["num_points"] == 221
    assert calibration["plane_z"] == pytest.approx(0.84)
    assert audit["arm_bias_xyz_m"]["left"] == pytest.approx(left_bias)
    assert audit["overall_rmse_m"] < 1e-8
    assert audit["dense_homography_rmse_m"] < 1e-7
