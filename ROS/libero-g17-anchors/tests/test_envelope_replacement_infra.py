from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from capx.envs.trial import _sanitize_runtime_stderr_for_prompt
from capx.utils.launch_utils import _get_visual_feedback


TOOLS = (
    Path(__file__).parents[1]
    / "experiments"
    / "core_fg"
    / "runtime"
    / "tools"
)
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "validate_envelope_replacement_run_for_infra_test",
    TOOLS / "validate_envelope_replacement_run.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_runtime_traceback_paths_are_prompt_canonical() -> None:
    first = (
        'Traceback:\n  File "/home/run-a/workdir/capx/envs/tasks/base.py", '
        "line 174, in _exec_user_code\nNameError: name 'np' is not defined\n"
    )
    second = first.replace("/home/run-a/workdir", "/scratch/run-b/source")

    expected = (
        'Traceback:\n  File "<CAPX_RUNTIME>/capx/envs/tasks/base.py", '
        "line 174, in _exec_user_code\nNameError: name 'np' is not defined\n"
    )
    assert _sanitize_runtime_stderr_for_prompt(first) == expected
    assert _sanitize_runtime_stderr_for_prompt(second) == expected


def test_runtime_traceback_omits_internal_verifier_wrapper_frame() -> None:
    def traceback_with_wrapper_line(line: int) -> str:
        return (
            'Traceback:\n  File "/home/run/workdir/capx/envs/tasks/base.py", '
            "line 174, in _exec_user_code\n"
            '  File "/home/run/workdir/capx/verification/ideal_tk_v4.py", '
            f"line {line}, in wrapped\n"
            "    return __original(*args, **kwargs)\n"
            '  File "/home/run/workdir/capx/integrations/franka/control.py", '
            "line 441, in sample_grasp_pose\n"
            "RuntimeError: no executable grasp\n"
        )

    expected = (
        'Traceback:\n  File "<CAPX_RUNTIME>/capx/envs/tasks/base.py", '
        "line 174, in _exec_user_code\n"
        '  File "<CAPX_RUNTIME>/capx/integrations/franka/control.py", '
        "line 441, in sample_grasp_pose\n"
        "RuntimeError: no executable grasp\n"
    )
    assert (
        _sanitize_runtime_stderr_for_prompt(traceback_with_wrapper_line(1021))
        == expected
    )
    assert (
        _sanitize_runtime_stderr_for_prompt(traceback_with_wrapper_line(1211))
        == expected
    )


def test_visual_feedback_owns_render_pixels() -> None:
    shared = np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeEnv:
        def render(self):
            return shared

    _, image = _get_visual_feedback(FakeEnv())
    shared[:] = 255

    assert image is not None
    assert np.asarray(image).tolist() == np.zeros((2, 2, 3), dtype=np.uint8).tolist()


def test_visual_feedback_renders_once_before_owning_pixels() -> None:
    class FakeEnv:
        calls = 0

        def render(self):
            self.calls += 1
            return np.full((2, 2, 3), self.calls, dtype=np.uint8)

    env = FakeEnv()
    _, image = _get_visual_feedback(env)

    assert env.calls == 1
    assert image is not None
    assert np.array_equal(np.asarray(image), np.ones((2, 2, 3), dtype=np.uint8))


def test_cube_obb_validator_rejects_silent_semantic_corruption(
    tmp_path: Path,
) -> None:
    (tmp_path / "cube_stack.log").write_text(
        "Object extent for green cube: [0.051 0.048 0.000056]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Silent cube OBB semantic corruption"):
        VALIDATOR._validate_cube_extent_semantics(tmp_path, "cube_stack")


def test_cube_obb_validator_accepts_broad_valid_domain(tmp_path: Path) -> None:
    (tmp_path / "cube_restack.log").write_text(
        "Object extent for red cube: [0.055 0.041 0.013]\n",
        encoding="utf-8",
    )

    assert VALIDATOR._validate_cube_extent_semantics(
        tmp_path,
        "cube_restack",
    ) == 1


def test_action_request_seed_lookup_is_unique() -> None:
    records = [
        {"effective_request_seed": 10, "source": "request_hash_replay"},
        {"effective_request_seed": 20, "source": "upstream"},
    ]

    assert VALIDATOR._request_record_for_seed(records, 10)["source"] == (
        "request_hash_replay"
    )
    with pytest.raises(ValueError, match="Expected one action request"):
        VALIDATOR._request_record_for_seed(records + [records[0]], 10)


def test_action_request_message_hash_is_bound_to_canonical_request() -> None:
    messages = [{"role": "user", "content": "prompt"}]
    record = {
        "request_canonical": json.dumps(
            {"model": "model", "messages": messages},
            sort_keys=True,
        )
    }

    assert VALIDATOR._message_hash_for_request_record(record) == (
        VALIDATOR._sha256_json(messages)
    )


def test_runner_snapshot_must_be_archived_and_read_only(tmp_path: Path) -> None:
    runner = tmp_path / "SUBMISSION_RUNNER.sbatch"
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    runner.chmod(0o444)
    digest = VALIDATOR._sha256_bytes(runner.read_bytes())
    manifest = {
        "git_rev": "abc123",
        "submission_git_rev": "abc123",
        "submission_runner_git_rev": "abc123",
        "submission_runner_snapshot": str(runner),
        "submission_runner_sha256": digest,
        "submission_runner_read_only": "true",
    }

    VALIDATOR._validate_runner_snapshot(
        pair_root=tmp_path,
        pair_manifest=manifest,
    )

    runner.chmod(0o644)
    with pytest.raises(ValueError, match="runner snapshot provenance"):
        VALIDATOR._validate_runner_snapshot(
            pair_root=tmp_path,
            pair_manifest=manifest,
        )


def test_silent_normalized_replay_requires_bounded_pixel_difference(
    tmp_path: Path,
) -> None:
    a1 = tmp_path / "a1" / "qwen" / "cube_stack" / "trial_01_result"
    treatment = (
        tmp_path / "treatment" / "qwen" / "cube_stack" / "trial_01_result"
    )
    a1.mkdir(parents=True)
    treatment.mkdir(parents=True)
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    second = first.copy()
    second.reshape(-1)[:12] = 255
    Image.fromarray(first).save(a1 / "visual_feedback_01.png")
    Image.fromarray(second).save(treatment / "visual_feedback_01.png")

    audit = VALIDATOR._validate_silent_visual_equivalence(
        a1_run_root=tmp_path / "a1",
        treatment_run_root=tmp_path / "treatment",
        task="cube_stack",
        trial=1,
        turn=1,
    )

    assert audit["changed_channels"] == 12

    second.reshape(-1)[:33] = 255
    Image.fromarray(second).save(treatment / "visual_feedback_01.png")
    with pytest.raises(ValueError, match="visual-equivalence bound"):
        VALIDATOR._validate_silent_visual_equivalence(
            a1_run_root=tmp_path / "a1",
            treatment_run_root=tmp_path / "treatment",
            task="cube_stack",
            trial=1,
            turn=1,
        )
