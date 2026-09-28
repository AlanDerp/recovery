from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).parents[1] / "experiments" / "core_fg" / "runtime" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "validate_fgsrp_treatment_freeze_for_test",
    TOOLS / "validate_fgsrp_run.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _fixture(tmp_path: Path) -> dict:
    run_root = tmp_path / "run"
    output = run_root / "model" / "cube_restack"
    output.mkdir(parents=True)
    (output / "summaries.txt").write_text(
        "Code generation success rate / Average reward / Task completed: \n"
        "1.000/0.500/1\n"
    )
    (run_root / "cube_restack.log").write_text(
        "Code saved to /x/trial_1_sandboxrc_0_reward_0.500_taskcompleted_1/code.py\n"
    )
    states = run_root / "repro_audit"
    states.mkdir()
    (states / "initial_states.jsonl").write_text(
        json.dumps({"binding": {}, "record_sha256": "state"}) + "\n"
    )
    trace_root = output / "fgsrp_traces" / "trial_01"
    turn = trace_root / "turn_01"
    turn.mkdir(parents=True)
    replacement = {
        "turn": 1,
        "native_prompt_sha256": "native",
        "corrected_prompt_sha256": "corrected",
        "repair_count": 1,
        "before_evidence_sha256": "before",
        "after_evidence_sha256": "after",
        "temporal_evidence_sha256": "temporal",
    }
    (turn / "replacement.json").write_text(json.dumps(replacement))
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "missing_count": 0,
                "entries": {"request": {"content": "response"}},
            }
        )
    )
    manifest = {
        "task": "cube_restack",
        "fgsrp_carrier": "m3_visual_difference",
        "native_mode": "M3",
        "global_seed": "6000",
        "fixed_episode_seed": "none",
        "trials": "1",
        "git_rev": "commit",
        "model": "qwen3.7-plus",
        "qwen_base_url": "https://example.invalid/v1",
        "frozen_a0_reference_sha256": "a0",
        "treatment_implementation_sha256": "implementation",
        "verifier_prompt_sha256": "vp",
        "verifier_schema_sha256": "vs",
        "adjudicator_prompt_sha256": "ap",
        "adjudicator_schema_sha256": "as",
    }
    audit = [
        {
            "ok": True,
            "source": "request_hash_replay",
            "replay_request_sha256": "request",
            "replay_request_normalized_sha256": "normalized",
            "response_content_sha256": "response",
            "effective_request_seed": 7001,
        }
    ]
    reference = {
        "schema": "capx_fgsrp_treatment_reference_v1",
        "task": "cube_restack",
        "carrier": "m3_visual_difference",
        "native_mode": "M3",
        "global_seed": 6000,
        "fixed_episode_seed": "none",
        "total_trials": 1,
        "source_git_rev": "commit",
        "model": "qwen3.7-plus",
        "qwen_base_url": "https://example.invalid/v1",
        "materialized_config_semantic_sha256": "config",
        "frozen_a0_reference_sha256": "a0",
        "infra_bundle_sha256": "infra",
        "python_packages_sha256": "packages",
        "source_pair_job_id": "1",
        "treatment_implementation_sha256": "implementation",
        "verifier_prompt_sha256": "vp",
        "verifier_schema_sha256": "vs",
        "adjudicator_prompt_sha256": "ap",
        "adjudicator_schema_sha256": "as",
        "full_replay_sha256": VALIDATOR._sha256_file(replay_path),
        "full_replay_entries": 1,
        "request_response_sequence": [
            {
                "request_sha256": "request",
                "request_normalized_sha256": "normalized",
                "response_content_sha256": "response",
                "effective_request_seed": 7001,
            }
        ],
        "expected_summary": "1.000/0.500/1",
        "trials": {
            "1": {"sandbox_rc": 0, "reward": 0.5, "task_completed": True}
        },
        "initial_state_record_sha256": ["state"],
        "trace_signatures": VALIDATOR._trace_signatures([trace_root]),
    }
    reference_path = tmp_path / "reference.json"
    source_validation_path = tmp_path / "source-validation.json"
    source_validation_path.write_text(json.dumps({"status": "valid"}))
    reference["source_pair_validation_sha256"] = VALIDATOR._sha256_file(
        source_validation_path
    )
    reference_path.write_text(json.dumps(reference))
    return {
        "reference_path": reference_path,
        "replay_path": replay_path,
        "source_validation_path": source_validation_path,
        "manifest": manifest,
        "treatment_run_root": run_root,
        "audit": audit,
        "trace_roots": [trace_root],
        "base_validation": {
            "materialized_config_semantic_sha256": "config",
            "infra_bundle_sha256": "infra",
            "python_packages_sha256": "packages",
        },
    }


def test_frozen_treatment_accepts_exact_registered_trajectory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = VALIDATOR._validate_frozen_treatment(
        **fixture,
        task="cube_restack",
        trials=1,
        global_seed=6000,
    )

    assert result["frozen_treatment_no_live_model_calls"] is True
    assert result["frozen_treatment_per_trial_results_equal"] is True


def test_frozen_treatment_rejects_live_model_call(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["audit"][0]["source"] = None

    with pytest.raises(ValueError, match="live or failed"):
        VALIDATOR._validate_frozen_treatment(
            **fixture,
            task="cube_restack",
            trials=1,
            global_seed=6000,
        )
