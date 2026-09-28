import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from capx.integrations.r1lite.generated_policy import (
    execute_dry_run,
    expected_operations,
    validate_program,
)

REPO = Path(__file__).resolve().parents[1]
THIRD_PIPELINE = REPO / "scripts/run_r1lite_tennis_ball_in_box_llm_pipeline.sh"
FOURTH_PIPELINE = REPO / "scripts/run_r1lite_fourth_llm_plan_pipeline.sh"
FOURTH_PLANNER = REPO / "scripts/r1lite_generate_llm_f2_f6_plan.py"
FOURTH_CODE_GENERATOR = (
    REPO / "scripts/r1lite_generate_llm_plan_execution_code.py"
)
FIFTH_PIPELINE = REPO / "scripts/run_r1lite_fifth_llm_fixed_plan_pipeline.sh"
FIFTH_GENERATOR = (
    REPO / "scripts/r1lite_generate_llm_fixed_plan_invocation.py"
)
SECOND_PIPELINE = REPO / "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh"
TRAJECTORY_RENDERER = (
    REPO / "scripts/render_r1lite_ik_actual_trajectory_html.py"
)
SHARED_LAUNCHER = REPO / "scripts/run_r1lite_tennis_ball_in_box_pipeline.sh"
R1_CONTROL = REPO / "capx/integrations/r1lite/control.py"
R1_LOW_LEVEL = REPO / "capx/envs/simulators/r1lite_real.py"


def test_third_pipeline_uses_llm_code_generation_with_separate_outputs():
    script = THIRD_PIPELINE.read_text()

    assert 'export USE_ORACLE_CODE=0' in script
    assert 'MODEL="${MODEL:-gpt-5.5}"' in script
    assert (
        'SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/'
        'r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"'
    ) in script
    assert 'cfg["use_oracle_code"] = False' in script
    assert 'cfg["r1lite_staged_tennis_execution"] = False' in script
    assert 'cfg["r1lite_staged_use_fixed_plan"] = False' in script
    assert 'cfg["use_wrist_camera"] = False' in script
    assert 'low_level["wrist_camera_enabled"] = False' in script
    assert 'low_level["max_joint_step"] = 0.15' in script
    assert 'low_level["goto_pose_joint_waypoint_step_rad"] = 0.15' in script
    assert 'low_level["r1lite_motion_validation_enabled"] = True' in script
    assert 'low_level["r1lite_require_pose_arrival"] = True' in script
    assert (
        'low_level["arm_target_topic"] = '
        '"/motion_target/target_joint_state_arm_right"'
    ) in script
    assert (
        'low_level["gripper_target_topic"] = '
        '"/motion_target/target_position_gripper_right"'
    ) in script
    assert (
        'low_level["relaxed_ik_target_topic"] = '
        '"/capx/relaxed_ik/target_pose_arm_right"'
    ) in script
    assert "outputs/r1lite_tennis_ball_in_box_llm" in script
    assert "run_r1lite_tennis_ball_in_box_pipeline.sh" in script


def test_third_pipeline_requires_key_file_without_embedding_a_secret():
    script = THIRD_PIPELINE.read_text()

    assert 'if [[ ! -r "${API_KEY_FILE}" ]]' in script
    assert "API_KEY=" not in script
    assert "--api-key" not in script


def test_shared_launcher_does_not_print_the_api_key():
    script = SHARED_LAUNCHER.read_text()

    assert "api_key=[redacted]" in script
    assert 'echo "[run] ${cmd[*]}"' not in script
    assert "local use_wrist_camera=False" in script
    assert '--use-wrist-camera "${use_wrist_camera}"' in script
    assert "--use-wrist-camera True)" not in script
    assert script.rindex("ensure_isolated_relaxed_ik") < script.rindex(
        "ensure_single_right_arm_input_source"
    )


def test_second_pipeline_remains_independent_of_the_llm_launcher():
    script = SECOND_PIPELINE.read_text()
    source_config = (
        REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
    ).read_text()

    assert "run_r1lite_tennis_ball_in_box_llm_pipeline" not in script
    assert "USE_ORACLE_CODE" not in script
    assert "r1lite_staged_tennis_execution: true" in source_config
    assert "r1lite_staged_use_fixed_plan: true" in source_config


def test_goto_pose_uses_configured_joint_step_and_hard_arrival_gate():
    control = R1_CONTROL.read_text()
    low_level = R1_LOW_LEVEL.read_text()

    assert '"goto_pose_joint_waypoint_step_rad"' in control
    assert "waypoint_step_rad=0.015" not in control
    assert '"r1lite_require_pose_arrival"' in control
    assert "goto_pose_joint_waypoint_step_rad: float = 0.015" in low_level
    assert "r1lite_require_pose_arrival: bool = False" in low_level
    assert "def get_torso_joints(self) -> np.ndarray:" in low_level


def test_fourth_pipeline_uses_llm_plan_then_constrained_llm_code():
    launcher = FOURTH_PIPELINE.read_text()
    planner = FOURTH_PLANNER.read_text()

    assert "r1lite_generate_llm_f2_f6_plan.py" in launcher
    assert "r1lite_generate_llm_plan_execution_code.py" in launcher
    assert 'LLM_GENERATED_POLICY_PATH="${GENERATED_CODE}"' in launcher
    assert 'FOURTH_PIPELINE_EXECUTE_ROBOT="${FOURTH_PIPELINE_EXECUTE_ROBOT:-0}"' in launcher
    assert 'if [[ "${FOURTH_PIPELINE_EXECUTE_ROBOT}" != "1" ]]' in launcher
    assert "generation and restricted dry-run passed" in launcher
    assert "R1LITE_ROUND_COUNT=1" in launcher
    assert "R1LITE_SINGLE_ROUND_POLICY=llm_absolute_20" in launcher
    assert "MAX_CORRECTIONS=20" in launcher
    assert "r1lite_execute_10_keyframes_xyz_corrected.sh" in launcher
    assert "cartesian_absolute" in launcher
    assert "R1LITE_PIPELINE_SETUP_ONLY=1" in launcher
    assert 'LLM_MAX_CARTESIAN_FRAME_STEP_M="${LLM_MAX_CARTESIAN_FRAME_STEP_M:-0.10}"' in launcher
    assert (
        'LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD="${LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD:-1.0}"'
        in launcher
    )
    assert '--max-cartesian-step-m "${LLM_MAX_CARTESIAN_FRAME_STEP_M}"' in launcher
    assert '"robot_motion_commands_sent": False' in planner
    assert '"output_plan_sha256": output_plan_sha256' in planner
    assert "REQUIRED_ROLES" in planner
    assert '"variable_frame_plan": True' in planner
    assert "def validate_anchors(" in planner
    assert "def densify_anchors(" in planner
    assert '"target_source"] = str(item["target_source"])' in planner
    assert '"coupled_grasp_enabled": False' in planner
    assert '"planner": "llm_variable_semantic_cartesian_targets"' in planner
    assert "runtime_torso_joints = r1.get_torso_joints()" in planner
    assert (
        'waypoint["torso_joints_rad"] = '
        "runtime_torso_joints.tolist()"
    ) in planner
    assert 'waypoint["torso_joints_source"] = "runtime_feedback"' in planner
    assert '"runtime_torso_joints_rad": runtime_torso_joints.tolist()' in planner
    assert '"f0_f1_source": "template_preserved"' in planner
    assert '"generated_orientation_source": "template_preserved"' in planner
    generator = FOURTH_CODE_GENERATOR.read_text()
    assert "expected_operations" in generator
    assert "validate_program" in generator
    assert "execute_dry_run" in generator
    assert '"robot_motion_commands_sent": False' in generator
    assert '"generated_code_sha256": generated_code_sha256' in generator


def _load_variable_plan_functions():
    tree = ast.parse(FOURTH_PLANNER.read_text())
    selected = []
    names = {
        "REQUIRED_ROLES",
        "ALLOWED_ROLES",
        "DEFAULT_MAX_CARTESIAN_STEP_M",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "validate_anchors",
            "densify_anchors",
        }:
            selected.append(node)
    namespace: dict[str, Any] = {"np": np, "Any": Any}
    exec(compile(ast.Module(selected, type_ignores=[]), "<planner>", "exec"), namespace)
    return (
        namespace["validate_anchors"],
        namespace["densify_anchors"],
    )


def _valid_variable_anchor_payload():
    roles_and_positions = [
        ("above_ball", [0.0, 0.0, 0.45]),
        ("pregrasp", [0.0, 0.0, 0.32]),
        ("approach", [0.0, 0.0, 0.26]),
        ("grasp", [0.0, 0.0, 0.23]),
        ("transit", [0.20, 0.0, 0.40]),
        ("transfer", [0.30, 0.0, 0.40]),
        ("above_box", [0.40, 0.0, 0.25]),
    ]
    return {
        "frames": [
            {
                "id": f"anchor_{index}",
                "role": role,
                "position_base_m": position,
            }
            for index, (role, position) in enumerate(roles_and_positions)
        ]
    }


def test_variable_plan_accepts_optional_transit_and_densifies_to_ten_cm():
    validate_anchors, densify_anchors = _load_variable_plan_functions()
    anchors = validate_anchors(
        _valid_variable_anchor_payload(),
        np.array([0.0, 0.0, 0.20]),
        np.array([0.40, 0.0, 0.10]),
        ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)),
    )
    expanded = densify_anchors(
        np.array([-0.30, 0.0, 0.50]),
        anchors,
        max_step_m=0.10,
    )
    points = [np.array([-0.30, 0.0, 0.50])] + [
        item["position"] for item in expanded
    ]
    distances = [
        float(np.linalg.norm(right - left))
        for left, right in zip(points, points[1:])
    ]

    assert len(anchors) == 7
    assert len(expanded) > len(anchors)
    assert max(distances) <= 0.10 + 1e-9
    assert any(item["target_source"] == "system_interpolation" for item in expanded)
    assert [item["role"] for item in expanded if not item["is_interpolated"]] == [
        "above_ball",
        "pregrasp",
        "approach",
        "grasp",
        "transit",
        "transfer",
        "above_box",
    ]


@pytest.mark.parametrize("mutation", ["missing", "out_of_order", "duplicate_id"])
def test_variable_plan_rejects_missing_disordered_roles_and_duplicate_ids(mutation):
    validate_anchors, _ = _load_variable_plan_functions()
    payload = _valid_variable_anchor_payload()
    if mutation == "missing":
        payload["frames"] = [
            frame for frame in payload["frames"] if frame["role"] != "transfer"
        ]
    elif mutation == "out_of_order":
        payload["frames"][1], payload["frames"][2] = (
            payload["frames"][2],
            payload["frames"][1],
        )
    else:
        payload["frames"][1]["id"] = payload["frames"][0]["id"]

    with pytest.raises(ValueError):
        validate_anchors(
            payload,
            np.array([0.0, 0.0, 0.20]),
            np.array([0.40, 0.0, 0.10]),
            ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)),
        )


def _generated_policy_fixture():
    plan = {
        "waypoints": [
            {"id": "f0", "semantic_role": "initial"},
            {"id": "f1", "semantic_role": "f1_anchor"},
            {"id": "pick", "semantic_role": "grasp"},
            {"id": "place", "semantic_role": "above_box"},
        ]
    }
    expected = expected_operations(plan, plan_sha256="a" * 64)
    code = "\n".join(
        [
            "bind_plan(sha256=" + repr("a" * 64) + ")",
            "execute_frame(frame_id='f0')",
            "execute_frame(frame_id='f1')",
            "execute_frame(frame_id='pick')",
            "close_gripper()",
            "execute_frame(frame_id='place')",
            "open_gripper()",
            "return_to_f1()",
        ]
    )
    return expected, code


def test_generated_policy_is_plan_bound_and_dry_runs_without_motion():
    expected, code = _generated_policy_fixture()

    assert validate_program(code, expected=expected) == expected
    assert execute_dry_run(code, expected=expected) == expected


def test_fourth_code_generator_writes_hash_bound_audit_without_motion(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "fourth_code_generator", FOURTH_CODE_GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = {
        "waypoints": [
            {"id": "f0", "semantic_role": "initial"},
            {"id": "f1", "semantic_role": "f1_anchor"},
            {"id": "pick", "semantic_role": "grasp"},
            {"id": "place", "semantic_role": "above_box"},
        ]
    }
    plan_path = tmp_path / "plan.json"
    code_path = tmp_path / "policy.py"
    audit_path = tmp_path / "audit.json"
    key_path = tmp_path / "key"
    plan_path.write_text(json.dumps(plan))
    key_path.write_text("not-a-real-key")
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    operations = expected_operations(plan, plan_sha256=plan_sha256)
    generated_code = module.render_reference_program(operations)
    monkeypatch.setattr(
        module,
        "query_model",
        lambda *_args, **_kwargs: {"content": generated_code},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(FOURTH_CODE_GENERATOR),
            "--plan",
            str(plan_path),
            "--output-code",
            str(code_path),
            "--audit-output",
            str(audit_path),
            "--api-key-file",
            str(key_path),
        ],
    )

    module.main()

    audit = json.loads(audit_path.read_text())
    assert audit["plan_sha256"] == plan_sha256
    assert audit["restricted_dry_run_passed"] is True
    assert audit["robot_motion_commands_sent"] is False
    assert audit["dry_run_trace"] == operations
    assert audit["generated_code_sha256"] == hashlib.sha256(
        code_path.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "invalid_code",
    [
        "import os",
        "open('/tmp/unsafe', 'w')",
        "while True:\n    execute_frame(frame_id='f0')",
        "execute_frame(frame_id='place')",
        "bind_plan(sha256='wrong')",
    ],
)
def test_generated_policy_rejects_unsafe_or_plan_mismatched_code(invalid_code):
    expected, _ = _generated_policy_fixture()

    with pytest.raises((ValueError, SyntaxError)):
        validate_program(invalid_code, expected=expected)


def test_fourth_executor_is_role_driven_and_runtime_subdivides_only_moves():
    executor = SECOND_PIPELINE.read_text()

    assert "def waypoint_role(wp: dict) -> str:" in executor
    assert "waypoint_role(wp) == \"grasp\"" in executor
    assert "waypoint_role(wp) == \"above_box\"" in executor
    assert "def execute_runtime_subdivisions(" in executor
    assert '"event": "runtime_cartesian_subdivision_inserted"' in executor
    assert 'runtime_wp["semantic_role"] = "transit"' in executor
    assert 'runtime_wp["semantic_action"] = "move"' in executor
    assert "frame_max_corrections = 0" in executor
    assert (
        "distance <= llm_max_cartesian_frame_step_m + 1e-9"
        in executor
    )
    assert "return llm_max_incremental_joint_delta_rad" in executor
    assert "def resolve_coupled_grasp_chain(" in executor
    assert "def coupled_tcp_target(" in executor
    assert "def fingertip_center_from_tcp(" in executor
    assert '"event": "coupled_grasp_chain_resolved"' in executor
    assert '"event": "gripper_close_safety_check"' in executor
    assert 'locked_quaternion_xyzw=(' in executor
    assert '"system_coupled_grasp_resolved"' in executor
    assert executor.index('"event": "gripper_close_safety_check"') < executor.index(
        "feedback = r1.command_gripper("
    )
    assert "close_tip_error_m" in executor
    assert "<= coupled_grasp_fingertip_tol_m" in executor
    assert "non-coupled variable plan cannot contain deferred grasp stages" in executor
    assert "refusing unresolved coupled-grasp waypoint" in executor
    assert (
        "if (\n        coupled_grasp_enabled\n        and waypoint_role(wp)"
        in executor
    )
    assert (
        "if coupled_grasp_enabled and not bool(\n"
        '                    wp.get("coupled_grasp_locked", False)'
        in executor
    )
    assert "coupled grasp candidate FK fingertip misses" in executor
    assert "require_finite(" in executor
    assert "place_xy_ok and place_z_ok" in executor
    assert "consume_generated_policy_operation(" in executor
    assert '"llm_generated_policy_bound"' in executor
    assert 'consume_generated_policy_operation("close_gripper")' in executor
    assert 'consume_generated_policy_operation("open_gripper")' in executor
    assert 'consume_generated_policy_operation("return_to_f1")' in executor
    assert "nearby_probes + [None] + global_fallback_probes" in executor
    assert 'selected_search_mode = "llm_neighborhood"' in executor
    assert 'selected_search_mode = "global_fallback"' in executor
    assert '"global_fallback_used": fallback_used' in executor
    assert (
        'LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD="${LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD:-1.0}"'
        in executor
    )


def test_trajectory_renderer_distinguishes_dynamic_anchors_and_interpolation():
    renderer = TRAJECTORY_RENDERER.read_text()

    assert '"LLM语义锚点"' in renderer
    assert '"系统/运行时10cm插值点"' in renderer
    assert '"llm_runtime_anchor"' in renderer
    assert '"system_interpolation"' in renderer
    assert '"runtime_interpolation"' in renderer
    assert "error_segment_points" in renderer
    assert '"IK关节解FK轨迹：FK(q_IK)"' in renderer
    assert '"实测关节FK轨迹：FK(q_actual)"' in renderer
    assert '"系统姿态耦合TCP目标"' in renderer
    assert '"目标指尖抓取中心轨迹"' in renderer
    assert '"实际指尖抓取中心轨迹"' in renderer
    assert '"锁定接近轴：TCP→指尖/球心"' in renderer


def test_trajectory_renderer_builds_dynamic_plan_traces(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "trajectory_renderer", TRAJECTORY_RENDERER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan_path = tmp_path / "variable_plan.json"
    plan_path.write_text(
        '{"fourth_pipeline":{"planner":"llm_variable_semantic_cartesian_targets"},'
        '"objects":{},"waypoints":[]}'
    )
    records = [
        {
            "event": "start",
            "plan": str(plan_path),
            "initial_physical_position_base_m": [0.0, 0.0, 0.0],
        },
        {
            "event": "frame_begin",
            "frame_index": 2,
            "id": "anchor_above_ball",
            "semantic_role": "above_ball",
            "target_source": "llm_runtime_anchor",
            "desired_position_base_m": [0.1, 0.0, 0.2],
        },
        {
            "event": "frame_begin",
            "frame_index": 3,
            "id": "interp_00_01",
            "semantic_role": "transit",
            "target_source": "system_interpolation",
            "is_interpolated": True,
            "desired_position_base_m": [0.2, 0.0, 0.2],
        },
        {
            "event": "coupled_grasp_chain_resolved",
            "ball_center_base_m": [0.3, 0.0, 0.2],
            "tcp_to_fingertip_m": 0.078,
            "llm_preferred_approach_side": "left",
            "llm_preferred_yaw_deg": 45.0,
            "llm_preferred_tilt_deg": 30.0,
            "selected_yaw_deg": 45.0,
            "selected_tilt_deg": 15.0,
            "selected_search_mode": "llm_neighborhood",
            "global_fallback_used": False,
            "stage_tcp_targets_base_m": {
                "grasp": [0.3, 0.0, 0.278],
            },
        },
        {
            "event": "frame_begin",
            "frame_index": 4,
            "id": "system_grasp_deferred",
            "semantic_role": "grasp",
            "target_source": "system_coupled_grasp_resolved",
            "desired_position_base_m": [0.3, 0.0, 0.278],
            "desired_fingertip_center_base_m": [0.3, 0.0, 0.2],
        },
        {
            "event": "frame_attempt",
            "frame_index": 4,
            "id": "system_grasp_deferred",
            "attempt": 0,
            "a0_pipeline": "cartesian_absolute_step",
            "compensated_target_position_base_m": [0.3, 0.0, 0.278],
            "actual_position_base_m": [0.3, 0.0, 0.279],
            "desired_position_base_m": [0.3, 0.0, 0.278],
            "position_error_m": 0.001,
            "actual_fingertip_center_base_m": [0.3, 0.0, 0.201],
            "fingertip_error_m": 0.001,
            "before_joints": [0.0] * 6,
            "actual_joints": [0.0] * 6,
            "command_result": {},
        },
        {"event": "finished"},
    ]

    traces, summary = module.build_page(records, tmp_path / "run.jsonl")
    names = {item["name"] for item in traces}

    assert "LLM语义锚点" in names
    assert "系统/运行时10cm插值点" in names
    assert "系统姿态耦合TCP目标" in names
    assert "目标指尖抓取中心轨迹" in names
    assert "实际指尖抓取中心轨迹" in names
    assert "锁定接近轴：TCP→指尖/球心" in names
    assert "LLM锚点 1" in summary
    assert "插值点 1" in summary


def test_trajectory_renderer_escapes_inline_script_terminators():
    spec = importlib.util.spec_from_file_location(
        "trajectory_renderer_xss", TRAJECTORY_RENDERER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    encoded = module.json_for_inline_script(
        [{"text": "</script><script>alert(1)</script>&"}]
    )

    assert "</script>" not in encoded
    assert "<script>" not in encoded
    assert "\\u003c/script\\u003e" in encoded
    assert "\\u0026" in encoded


def test_shared_launcher_has_explicit_setup_only_mode():
    script = SHARED_LAUNCHER.read_text()

    assert 'R1LITE_PIPELINE_SETUP_ONLY:-0' in script
    assert "ensure_llm_proxy" in script
    assert "run_launch" in script


def test_fifth_pipeline_keeps_fixed_plan_and_uses_llm_for_one_round():
    launcher = FIFTH_PIPELINE.read_text()
    generator = FIFTH_GENERATOR.read_text()

    assert "offline_ik_pick_place_joint_poses_3d.json" in launcher
    assert 'cfg["use_oracle_code"] = False' in launcher
    assert "r1lite_generate_llm_fixed_plan_invocation.py" in launcher
    assert "R1LITE_ROUND_COUNT=1" in launcher
    assert '"${FIXED_PLAN}"' in launcher
    assert "r1lite_generate_llm_f2_f6_plan.py" not in launcher
    assert "sha256sum" in launcher
    assert 'ALLOWED_FUNCTION = "execute_pipeline_2_single_round"' in generator
    assert '"oracle_used": False' in generator
    assert '"robot_motion_commands_sent": False' in generator


def test_fifth_pipeline_llm_program_is_restricted_to_exact_fixed_plan_call():
    spec = importlib.util.spec_from_file_location("fifth_generator", FIFTH_GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected_path = "/tmp/fixed-plan.json"
    expected_hash = "a" * 64
    valid = (
        "execute_pipeline_2_single_round("
        f"plan_path={expected_path!r}, "
        f"expected_sha256={expected_hash!r}, "
        "round_count=1)"
    )
    assert module.validate_program(
        valid,
        expected_plan_path=expected_path,
        expected_sha256=expected_hash,
    )["round_count"] == 1

    with pytest.raises(ValueError, match="exactly one call expression"):
        module.validate_program(
            valid + "\nprint('unsafe')",
            expected_plan_path=expected_path,
            expected_sha256=expected_hash,
        )
    with pytest.raises(ValueError, match="changed the fixed plan path"):
        module.validate_program(
            valid.replace(expected_path, "/tmp/other.json"),
            expected_plan_path=expected_path,
            expected_sha256=expected_hash,
        )
