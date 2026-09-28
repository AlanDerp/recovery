import importlib.util
from pathlib import Path

import pytest

from capx.integrations.r1lite.sandwich_executor import (
    SandwichExecutionRuntime,
    dispatch_policy,
)
from capx.integrations.r1lite.sandwich_perception import SCENE_SCHEMA
from capx.integrations.r1lite.sandwich_plan import (
    build_policy,
    build_sandwich_intent,
    select_arm,
)


def plan():
    def item(center, width, length, height, major=(1.0, 0.0)):
        return {
            "center_base_m": list(center),
            "table_z_at_center_m": 0.80,
            "top_z_base_m": 0.80 + height,
            "height_above_table_m": height,
            "thickness_m": height,
            "major_axis_base_xy": list(major),
            "minor_axis_base_xy": [-major[1], major[0]],
            "width_m": width,
            "length_m": length,
            "yaw_rad": 0.0,
        }

    scene = {
        "schema": SCENE_SCHEMA,
        "objects": {
            "brown_bread": item((0.45, 0.0, 0.81), 0.07, 0.09, 0.02),
            "donut": item((0.40, 0.12, 0.815), 0.06, 0.07, 0.03),
            "mixed_bread": item((0.52, 0.12, 0.81), 0.07, 0.09, 0.02),
        },
        "table_plane": {
            "normal_base": [0, 0, 1],
            "d": -0.8,
            "rmse_m": 0.001,
            "inlier_count": 100,
            "candidate_count": 120,
        },
    }
    intent = build_sandwich_intent(
        scene,
        phase="donut_on_brown",
        first_image_sha256="1" * 64,
        second_image_sha256="2" * 64,
        repeatability_m=0.001,
    )
    return select_arm(intent, "right")


class FakeController:
    def __init__(self, feedback=20.0):
        self.feedback = feedback
        self.hold_count = 0

    def get_gripper_feedback(self, side):
        assert side == "right"
        return self.feedback

    def hold(self):
        self.hold_count += 1


class FakeDelegate:
    def __init__(self, feedback=20.0):
        self.trace = []
        self.controller = FakeController(feedback)
        self.calls = []

    def bind_plan(self, **kwargs):
        self.calls.append(("bind_plan", kwargs))

    def open_gripper(self, **kwargs):
        self.calls.append(("open_gripper", kwargs))

    def close_gripper(self, **kwargs):
        self.calls.append(("close_gripper", kwargs))

    def execute_single_arm_stage(self, **kwargs):
        self.calls.append(("execute_single_arm_stage", kwargs))


def test_deterministic_dispatch_requires_home_and_checks_retention():
    value = plan()
    policy = build_policy(value, plan_sha256="a" * 64)
    delegate = FakeDelegate(feedback=20.0)
    runtime = SandwichExecutionRuntime(delegate, value)

    trace = dispatch_policy(
        policy,
        plan=value,
        plan_sha256="a" * 64,
        runtime=runtime,
    )

    assert trace[-1]["stage_id"] == "home"
    assert delegate.trace[-1]["event"] == "sandwich_grasp_retention_gate"
    assert delegate.trace[-1]["passed"] is True


def test_empty_gripper_after_lift_holds_and_aborts():
    value = plan()
    policy = build_policy(value, plan_sha256="a" * 64)
    delegate = FakeDelegate(feedback=6.0)
    runtime = SandwichExecutionRuntime(delegate, value)

    with pytest.raises(RuntimeError, match="retention gate failed"):
        dispatch_policy(
            policy,
            plan=value,
            plan_sha256="a" * 64,
            runtime=runtime,
        )
    assert delegate.controller.hold_count == 1
    assert delegate.trace[-1]["passed"] is False


def test_launcher_has_three_approval_stages_and_no_pipeline_eight_escape_hatch():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/run_r1lite_ninth_sandwich_pipeline.sh").read_text()
    validator = (
        root / "scripts/r1lite_validate_ninth_sandwich_phase.py"
    ).read_text()
    main = (root / "scripts/r1lite_ninth_sandwich.py").read_text()

    assert "prepare-donut)" in launcher
    assert "execute-donut-and-prepare-top)" in launcher
    assert "execute-top)" in launcher
    assert "DONUT_CAPABILITY_CERTIFICATE_PATH" in launcher
    assert "TOP_CAPABILITY_CERTIFICATE_PATH" in launcher
    assert "R1LiteCombinedCollisionKinematics" in validator
    assert '"robot_motion_commands_sent": False' in validator
    assert "force_continue_after_maximum_iterations" not in main
    assert "verify-after-execution" in main


def test_renderer_embeds_preview_and_arm_scores(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/render_r1lite_ninth_sandwich_html.py"
    spec = importlib.util.spec_from_file_location("ninth_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = plan()
    value["perception"]["scene"]["objects"]["brown_bread"]["pixel_geometry"] = {
        "centroid_px": [10, 10]
    }
    value["perception"]["scene"]["objects"]["donut"]["pixel_geometry"] = {
        "centroid_px": [20, 20]
    }
    value["perception"]["scene"]["objects"]["mixed_bread"]["pixel_geometry"] = {
        "centroid_px": [30, 30]
    }

    html = module.render(
        plan=value,
        bundle=None,
        certification={
            "arm_candidates": {
                "left": {"passed": False},
                "right": {
                    "passed": True,
                    "score": 0.2,
                    "maximum_fk_position_error_m": 0.001,
                    "minimum_swept_link_clearance_m": 0.01,
                },
            }
        },
        execution=None,
        preview_uri="data:image/png;base64,AAAA",
    )

    assert "data:image/png;base64,AAAA" in html
    assert "armCandidates" in html
    assert "计划 TCP" in html
