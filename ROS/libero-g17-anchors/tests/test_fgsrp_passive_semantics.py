from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from capx.verification import fgsrp

REPO_ROOT = Path(__file__).parents[1]
EXPORT_PATH = (
    REPO_ROOT
    / "experiments/core_fg/runtime/tools/export_fgsrp_passive_semantics.py"
)
SPEC = importlib.util.spec_from_file_location(
    "export_fgsrp_passive_semantics", EXPORT_PATH
)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def _fake_env(*, grasped: bool = False) -> SimpleNamespace:
    class FakeSimulator:
        cubeA = SimpleNamespace(size=np.array([0.02, 0.02, 0.02]))
        cubeB = SimpleNamespace(size=np.array([0.025, 0.025, 0.025]))
        cubeA_body_id = 0
        cubeB_body_id = 1
        robots = [SimpleNamespace(gripper=object())]
        sim = SimpleNamespace(
            data=SimpleNamespace(
                body_xpos=np.array([[0.04, 0.0, 0.10], [0.0, 0.0, 0.05]]),
                body_xquat=np.array(
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                ),
            )
        )

        def _check_grasp(self, *, gripper, object_geoms) -> bool:
            return grasped and object_geoms is self.cubeA

        def check_contact(self, first, second) -> bool:
            return False

    class FakeLowLevel:
        robosuite_env = FakeSimulator()

        def _cube_pose_dict(self, observation):
            return {
                "primary": [*observation["cubeA_pos"], 1.0, 0.0, 0.0, 0.0],
                "secondary": [*observation["cubeB_pos"], 1.0, 0.0, 0.0, 0.0],
            }

    return SimpleNamespace(low_level_env=FakeLowLevel())


def test_passive_capture_exports_action_invisible_semantic_reference(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    trial_output = run_root / "qwen3.7-plus/cube_stack/trial_01"
    trial_output.mkdir(parents=True)

    fgsrp.initialize_passive_trial_trace(
        _fake_env(grasped=False),
        output_dir=str(trial_output),
        trial=1,
        task="cube_stack",
    )
    fgsrp.capture_passive_transition(
        _fake_env(grasped=True),
        output_dir=str(trial_output),
        trial=1,
        turn=1,
        task="cube_stack",
    )

    transition_path = (
        trial_output
        / fgsrp.PASSIVE_TRACE_DIR
        / "trial_01/turn_01/transition.json"
    )
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    assert transition["action_visible_gt_packet_injected"] is False
    assert transition["action_visible_bridge_injected"] is False
    assert transition["action_request_changed"] is False

    output = tmp_path / "semantic-reference.json"
    package = EXPORT.export(
        run_root=run_root,
        output=output,
        task="cube_stack",
        total_trials=1,
        global_seed=6100,
    )

    assert package["capture_mode"] == "passive_action_invisible"
    assert package["student_prompt_changed"] is False
    assert package["student_action_request_changed"] is False
    assert package["transition_count"] == 1
    assert package["m3_semantic_transition_sha256"]["1"]["1"] == (
        EXPORT.semantic_evidence_sha256(transition)
    )


def test_passive_capture_rejects_skipped_turn(tmp_path: Path) -> None:
    trial_output = tmp_path / "trial"
    fgsrp.initialize_passive_trial_trace(
        _fake_env(),
        output_dir=str(trial_output),
        trial=1,
        task="cube_restack",
    )

    try:
        fgsrp.capture_passive_transition(
            _fake_env(),
            output_dir=str(trial_output),
            trial=1,
            turn=2,
            task="cube_restack",
        )
    except ValueError as exc:
        assert "turn-synchronous" in str(exc)
    else:
        raise AssertionError("Skipped passive turn was accepted")
