import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from capx.integrations.r1lite.cup_rgbd import reconstruct_inverted_cup_pose
from capx.integrations.r1lite.cup_stack_plan import (
    PHASE_ORDER,
    build_cup_phase_plan,
    validate_cup_phase_plan,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/r1lite_prepare_three_cup_stack_config.py"
LAUNCHER = ROOT / "scripts/run_r1lite_three_cup_stack_pipeline.sh"
LLM_LAUNCHER = ROOT / "scripts/run_r1lite_three_cup_stack_llm_pipeline.sh"
CONTROL = ROOT / "capx/integrations/r1lite/control.py"


def build(phase="orange_on_purple", support_rim=0.80):
    return build_cup_phase_plan(
        phase=phase,
        source_table_position_base_m=[0.42, -0.10, 0.80],
        support_position_base_m=[0.50, 0.08, 0.80],
        source_height_m=0.10,
        support_rim_z_m=support_rim,
        cup_diameter_m=0.07,
    )


def test_phase_order_and_vertical_motion_are_immutable():
    assert PHASE_ORDER == ("orange_on_purple", "green_on_orange")
    plan = build()
    validate_cup_phase_plan(plan)
    waypoints = {item["id"]: item for item in plan["waypoints"]}

    assert plan["source"] == "orange cup"
    assert plan["support"] == "purple cup"
    assert waypoints["pregrasp"]["tcp_position_base_m"][:2] == pytest.approx(
        waypoints["grasp"]["tcp_position_base_m"][:2]
    )
    assert waypoints["above_support"]["tcp_position_base_m"][:2] == pytest.approx(
        waypoints["place"]["tcp_position_base_m"][:2]
    )
    assert waypoints["lift"]["tcp_position_base_m"][2] == pytest.approx(
        waypoints["above_support"]["tcp_position_base_m"][2]
    )


def test_second_phase_nests_green_at_the_same_rim_height():
    plan = build("green_on_orange", support_rim=0.80)
    place = next(item for item in plan["waypoints"] if item["id"] == "place")
    expected_jaw_center_z = 0.80 + 0.10 * 0.55
    assert place["fingertip_position_base_m"][2] == pytest.approx(
        expected_jaw_center_z
    )
    assert plan["stacking_mode"] == "same_orientation_nested_inverted_cups"
    assert plan["nested_height_increment_m"] == 0.0
    assert plan["source"] == "green cup"
    assert plan["support"] == "orange cup"


def test_rgbd_reconstructs_inverted_cup_top_body_and_rim_centers():
    depth = np.full((80, 100), 0.60, dtype=float)
    mask = np.zeros_like(depth, dtype=bool)
    mask[20:60, 30:70] = True
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    base_from_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.45],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.50],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    pose = reconstruct_inverted_cup_pose(
        depth,
        intrinsics,
        base_from_camera,
        mask,
        cup_height_m=0.10,
    )

    assert pose["top_center_base_m"][2] == pytest.approx(0.90)
    assert pose["body_center_base_m"][2] == pytest.approx(0.85)
    assert pose["rim_center_base_m"][2] == pytest.approx(0.80)


@pytest.mark.parametrize(
    ("height", "diameter"),
    [(0.01, 0.07), (0.10, 0.10)],
)
def test_unsafe_cup_dimensions_are_rejected(height, diameter):
    with pytest.raises(ValueError):
        build_cup_phase_plan(
            phase="orange_on_purple",
            source_table_position_base_m=[0.4, 0.0, 0.8],
            support_position_base_m=[0.5, 0.0, 0.8],
            source_height_m=height,
            support_rim_z_m=0.8,
            cup_diameter_m=diameter,
        )


def test_generated_config_is_deterministic_and_uses_color_gates():
    spec = importlib.util.spec_from_file_location("cup_config", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = yaml.safe_load(
        (
            ROOT / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
        ).read_text()
    )
    config = module.build_config(
        source,
        cup_height_m=0.10,
        cup_diameter_m=0.07,
        gripper_close_position=20.0,
    )
    code = config["env"]["cfg"]["oracle_code"]

    assert code.index('"orange_on_purple"') < code.index('"green_on_orange"')
    assert "locate_colored_cups_rgbd" in code
    assert 'nested["orange cup"]["rim_center_base_m"]' in code
    assert "nested_occlusion_expected=True" in code
    assert "expected_top_center_base_m" in code
    assert "success, details = verify_cup_stack(" in code
    assert config["r1lite_staged_tennis_execution"] is False
    assert config["use_oracle_code"] is True
    assert config["env"]["cfg"]["low_level"]["object_prompts"] == {
        "orange cup": "orange plastic cup",
        "purple cup": "purple plastic cup",
        "green cup": "green plastic cup",
    }


def test_launcher_is_no_motion_by_default_and_execution_requires_measurements():
    launcher = LAUNCHER.read_text()
    llm_launcher = LLM_LAUNCHER.read_text()
    control = CONTROL.read_text()

    assert 'MODE="${1:-generate}"' in launcher
    assert 'CUP_STACK_LLM_EXECUTE_ROBOT="${CUP_STACK_LLM_EXECUTE_ROBOT:-0}"' in (
        llm_launcher
    )
    assert ': "${CUP_HEIGHT_M:?Set measured CUP_HEIGHT_M}"' in llm_launcher
    assert ': "${CUP_DIAMETER_M:?Set measured CUP_DIAMETER_M}"' in llm_launcher
    assert ': "${CUP_GRIPPER_CLOSE_POSITION:?' in llm_launcher
    assert 'if [[ "${CUP_STACK_LLM_EXECUTE_ROBOT}" != "1" ]]' in llm_launcher
    assert "SKIP_RIGHT_ARM_MIT_MODE_RESET=1" in llm_launcher
    assert "scripts/r1lite_hold_torso.py" in llm_launcher
    assert '--torso-hold-status "${TORSO_HOLD_STATUS}"' in llm_launcher
    assert '--fixed-scene "${FIXED_CUP_SCENE_PATH}"' in llm_launcher
    assert '"locate_colored_cups": self.locate_colored_cups' in control
    assert '"verify_cup_stack": self.verify_cup_stack' in control
    assert '"median_hsv"' in control
