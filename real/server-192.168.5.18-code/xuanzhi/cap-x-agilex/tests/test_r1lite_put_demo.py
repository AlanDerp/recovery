from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/r1lite_execute_put_demo.py"
REFERENCE_PLAN_PATH = (
    ROOT
    / "outputs/r1lite_three_cup_stack_llm"
    / "cup_stack_llm_plan_20260830_beizi2_xy_early_drop.json"
)
SPEC = importlib.util.spec_from_file_location("put_demo", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PUT_DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUT_DEMO)


@pytest.mark.parametrize(
    "order",
    list(itertools.permutations(("peach", "tennis", "apple"))),
)
def test_every_put_demo_order_is_supported(
    order: tuple[str, str, str],
) -> None:
    normalized = PUT_DEMO.normalize_object_order(order)
    assert set(normalized) == {"peach", "tennis ball", "apple"}


def test_put_demo_accepts_chinese_object_names() -> None:
    assert PUT_DEMO.normalize_object_order(
        ("桃子", "网球", "苹果")
    ) == ("peach", "tennis ball", "apple")


@pytest.mark.parametrize(
    "order",
    [
        ("peach", "peach", "apple"),
        ("peach", "tennis"),
        ("peach", "tennis", "orange"),
    ],
)
def test_invalid_put_demo_orders_are_rejected(
    order: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="unique permutation"):
        PUT_DEMO.normalize_object_order(order)


def test_fixed_object_centers_match_beizi2_left_to_right() -> None:
    reference_plan = json.loads(REFERENCE_PLAN_PATH.read_text())
    centers = PUT_DEMO.load_fixed_object_centers(reference_plan)

    assert centers["peach"].tolist() == pytest.approx(
        [0.369125986605484, -0.3517798552469621, 0.94]
    )
    assert centers["tennis ball"].tolist() == pytest.approx(
        [0.369125986605484, -0.2690440808570822, 0.94]
    )
    assert centers["apple"].tolist() == pytest.approx(
        [0.369125986605484, -0.18576186433675607, 0.94]
    )
    assert (
        centers["peach"][1]
        < centers["tennis ball"][1]
        < centers["apple"][1]
    )


def test_put_demo_targets_use_topdown_lift_and_plus_x_release() -> None:
    center = np.array([0.369125986605484, -0.2690440808570822, 0.94])
    topdown = np.array([0.0, 1.0, 0.0, 0.0])
    targets = PUT_DEMO.build_put_demo_targets(center, topdown)

    assert targets["grasp_point"].tolist() == pytest.approx(
        [
            center[0],
            center[1],
            center[2] - PUT_DEMO.GRASP_BELOW_OBJECT_CENTER_M,
        ]
    )
    assert targets["grasp_tcp"].tolist() == pytest.approx(
        [
            center[0],
            center[1],
            center[2]
            - PUT_DEMO.GRASP_BELOW_OBJECT_CENTER_M
            + PUT_DEMO.TCP_TO_FINGERTIP_M,
        ]
    )
    assert (
        targets["pregrasp_tcp"] - targets["grasp_tcp"]
    ).tolist() == pytest.approx([0.0, 0.0, 0.10])
    assert (
        targets["lift_tcp"] - targets["grasp_tcp"]
    ).tolist() == pytest.approx([0.0, 0.0, 0.10])
    assert (
        targets["release_tcp"] - targets["lift_tcp"]
    ).tolist() == pytest.approx([0.15, 0.0, 0.0])
    assert (
        targets["box_center"] - center
    ).tolist() == pytest.approx([0.20, 0.0, 0.0])


def test_release_policy_has_no_height_gate_and_opens_from_100() -> None:
    center = np.array([0.4, -0.2, 123.0])
    targets = PUT_DEMO.build_put_demo_targets(
        center,
        np.array([0.0, 1.0, 0.0, 0.0]),
    )

    assert PUT_DEMO.OPEN_GRIPPER_COMMAND == 100.0
    assert targets["release_tcp"][2] == pytest.approx(
        targets["lift_tcp"][2]
    )
    assert not hasattr(PUT_DEMO, "object_center_is_over_box")


def test_beizi_entry_goes_direct_to_pregrasp_then_descends() -> None:
    calls: list[tuple[str, object, dict[str, object]]] = []

    class Api:
        def goto_pose(
            self,
            target: np.ndarray,
            quaternion: np.ndarray,
            **kwargs: object,
        ) -> dict[str, bool]:
            calls.append(("goto", target.copy(), kwargs))
            return {"success": True}

        def descend_z_in_steps(
            self,
            target: np.ndarray,
            quaternion: np.ndarray,
            **kwargs: object,
        ) -> dict[str, bool]:
            calls.append(("descend", target.copy(), kwargs))
            return {"success": True}

    quaternion = np.array([0.0, 1.0, 0.0, 0.0])
    targets = PUT_DEMO.build_put_demo_targets(
        np.array([0.369125986605484, -0.3517798552469621, 0.94]),
        quaternion,
    )
    result = PUT_DEMO.execute_beizi_grasp_entry(
        Api(),
        targets,
        quaternion,
    )

    assert [call[0] for call in calls] == ["goto", "descend"]
    assert np.allclose(calls[0][1], targets["pregrasp_tcp"])
    assert calls[0][2]["translate_then_orient"] is False
    assert calls[0][2]["max_cartesian_step_m"] == pytest.approx(0.06)
    assert np.allclose(calls[1][1], targets["grasp_tcp"])
    assert calls[1][2]["start_z"] == pytest.approx(
        targets["pregrasp_tcp"][2]
    )
    assert calls[1][2]["num_steps"] == 12
    assert result["entry_policy"].startswith("beizi_")


def test_beizi_motion_parameters_override_steps_without_replacing_config() -> None:
    class Robot:
        lazy_moca_start_command = ["required-start-command"]
        arm_target_topic = "/required/arm/topic"

    robot = Robot()
    PUT_DEMO.apply_beizi_motion_parameters(robot)

    for name, value in PUT_DEMO.BEIZI_MOTION_PARAMETERS.items():
        if isinstance(value, bool):
            assert getattr(robot, name) is value
        else:
            assert getattr(robot, name) == pytest.approx(value)
    assert robot.lazy_moca_start_command == ["required-start-command"]
    assert robot.arm_target_topic == "/required/arm/topic"
