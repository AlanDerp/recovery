from __future__ import annotations

import importlib.util
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "core_fg"
    / "runtime"
    / "tools"
    / "validate_paired_run.py"
)
SPEC = importlib.util.spec_from_file_location("validate_paired_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _paired_fixture(tmp_path: Path) -> Namespace:
    pair_root = tmp_path / "pair"
    a1_root = pair_root / "a1-run"
    core_root = pair_root / "core-run"
    seed = 2000
    response_hash = "response-sha"

    for root in (a1_root, core_root):
        (root / "model" / "cube_stack").mkdir(parents=True)
        (root / "model" / "cube_stack" / "summaries.txt").write_text(
            "summary\n", encoding="utf-8"
        )
        config = (
            root
            / "materialized_configs"
            / "env_configs"
            / "cube_stack"
            / "franka_robosuite_cube_stack_multiturn_vf.yaml"
        )
        config.parent.mkdir(parents=True)
        config.write_text("same: config\n", encoding="utf-8")
        (root / "cube_stack.log").write_text("healthy\n", encoding="utf-8")
        _write_jsonl(
            root / "repro_audit" / "initial_states.jsonl",
            [
                {
                    "binding": {"seed": 1},
                    "requested_seed": 1,
                    "effective_seed": 1,
                    "episode_seed_base": None,
                    "seed": 1,
                    "record_sha256": "state-sha",
                }
            ],
        )

    _write_jsonl(
        a1_root / "qwen_proxy_audit.jsonl",
        [
            {
                "ok": True,
                "source": "upstream",
                "effective_request_seed": seed,
                "response_content_sha256": response_hash,
            }
        ],
    )
    _write_jsonl(
        core_root / "qwen_proxy_audit.jsonl",
        [
            {
                "ok": True,
                "source": "request_hash_replay",
                "effective_request_seed": seed,
                "response_content_sha256": response_hash,
            }
        ],
    )
    (pair_root / "a1_initial_request_hash_replay.json").write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "missing_count": 0,
                "entries": {},
            }
        ),
        encoding="utf-8",
    )
    (pair_root / "INFRA_FINGERPRINT.json").write_text(
        json.dumps(
            {
                "schema": "capx_cube_infra_fingerprint_v1",
                "infra_bundle_sha256": "infra-sha",
                "python_packages_sha256": "packages-sha",
                "lock_matched": True,
                "lock_mismatches": {},
            }
        ),
        encoding="utf-8",
    )

    return Namespace(
        pair_root=pair_root,
        a1_run_root=a1_root,
        core_run_root=core_root,
        task="cube_stack",
        trials=1,
        global_seed=1000,
    )


def test_validate_accepts_complete_pair(tmp_path: Path) -> None:
    result = MODULE.validate(_paired_fixture(tmp_path))

    assert result["status"] == "valid"
    assert result["replay_hits"] == 1
    assert result["initial_state_hashes_equal"] is True


def test_validate_supports_m3_initial_action_seed_offset(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    args.initial_query_seed_offset = 1
    args.config_suffix = "vdm"
    expected_seed = args.global_seed + 1001
    for root in (args.a1_run_root, args.core_run_root):
        audit = root / "qwen_proxy_audit.jsonl"
        records = [json.loads(line) for line in audit.read_text().splitlines()]
        records[0]["effective_request_seed"] = expected_seed
        _write_jsonl(audit, records)
        old_config = next((root / "materialized_configs").rglob("*_vf.yaml"))
        new_name = old_config.name.replace("_vf.yaml", "_vdm.yaml")
        old_config.rename(old_config.with_name(new_name))

    result = MODULE.validate(args)

    assert result["required_initial_query_seeds"] == [expected_seed]
    assert result["initial_query_seed_offset"] == 1
    assert result["config_suffix"] == "vdm"


def test_validate_accepts_episode_seed_base(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    args.episode_seed_base = 9000
    for root in (args.a1_run_root, args.core_run_root):
        path = root / "repro_audit" / "initial_states.jsonl"
        record = json.loads(path.read_text())
        record.update(
            {
                "binding": {"seed": 9001},
                "effective_seed": 9001,
                "episode_seed_base": 9000,
                "seed": 9001,
            }
        )
        _write_jsonl(path, [record])

    result = MODULE.validate(args)

    assert result["episode_seed_base"] == 9000
    assert result["effective_episode_seeds"] == [9001]
    assert result["episode_seed_mapping_verified"] is True


def test_validate_rejects_wrong_episode_seed_mapping(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    args.episode_seed_base = 9000

    with pytest.raises(ValueError, match="effective seed mismatch"):
        MODULE.validate(args)


def test_validate_rejects_fixed_and_base_episode_seeds(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    args.fixed_episode_seed = 17
    args.episode_seed_base = 9000

    with pytest.raises(ValueError, match="mutually exclusive"):
        MODULE.validate(args)


def test_validate_rejects_replay_response_mismatch(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    audit = args.core_run_root / "qwen_proxy_audit.jsonl"
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    records[0]["response_content_sha256"] = "mutated-response-sha"
    _write_jsonl(audit, records)

    with pytest.raises(ValueError, match="Initial response hash mismatch"):
        MODULE.validate(args)


def test_validate_rejects_noninitial_failed_model_call(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    audit = args.a1_run_root / "qwen_proxy_audit.jsonl"
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    records.append(
        {
            "ok": False,
            "status_code": 409,
            "source": "request_hash_replay_required_miss",
            "effective_request_seed": 2001,
            "error": "required request did not match frozen request hash",
        }
    )
    _write_jsonl(audit, records)

    with pytest.raises(ValueError, match="A1 Qwen audit contains failed calls"):
        MODULE.validate(args)


def test_validate_rejects_trial_child_process_failure(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    (args.core_run_root / "cube_stack.log").write_text(
        "Trial 1 child process failed:\nTraceback\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="trial_child_process_failed"):
        MODULE.validate(args)


def test_validate_accepts_complete_frozen_a0_trajectory(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    replay_path = args.pair_root / "a1_initial_request_hash_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "missing_count": 0,
                "entries": {"request-sha": {"content": "response"}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        args.a1_run_root / "qwen_proxy_audit.jsonl",
        [
            {
                "ok": True,
                "source": "request_hash_replay",
                "effective_request_seed": 2000,
                "response_content_sha256": "response-sha",
                "replay_request_sha256": "request-sha",
            }
        ],
    )
    summary = args.a1_run_root / "model" / "cube_stack" / "summaries.txt"
    summary.write_text(
        "Code generation success rate / Average reward / Task completed:\n"
        "1.000/1.000/1\n",
        encoding="utf-8",
    )
    (args.a1_run_root / "cube_stack.log").write_text(
        "Code saved to /tmp/trial_01_sandboxrc_0_reward_1.000_"
        "taskcompleted_1/code.py\n",
        encoding="utf-8",
    )
    replay_sha = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema": "capx_frozen_a0_reference_v1",
                "task": "cube_stack",
                "global_seed": 1000,
                "total_trials": 1,
                "source_git_rev": "source-rev",
                "source_pair_job_id": "source-job",
                "full_replay_sha256": replay_sha,
                "full_replay_entries": 1,
                "expected_summary": "1.000/1.000/1",
                "initial_state_record_sha256": ["state-sha"],
                "trials": {
                    "1": {
                        "sandbox_rc": 0,
                        "reward": 1.0,
                        "task_completed": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args.frozen_a0_reference = reference

    result = MODULE.validate(args)

    assert result["frozen_a0"] is True
    assert result["frozen_a0_no_live_model_calls"] is True
    assert result["frozen_a0_replay_entries_consumed"] == 1
    assert result["frozen_a0_reexecuted_in_current_run"] is True

    _write_jsonl(
        args.a1_run_root / "qwen_proxy_audit.jsonl",
        [
            {
                "ok": True,
                "source": "upstream",
                "effective_request_seed": 2000,
                "response_content_sha256": "response-sha",
            }
        ],
    )
    args.frozen_a0_reference_only = True
    result = MODULE.validate(args)
    assert result["frozen_a0_reference_only"] is True
    assert result["frozen_a0_replay_entries_consumed"] is None
    assert result["frozen_a0_current_run_model_calls"] == 0
    assert result["frozen_a0_reexecuted_in_current_run"] is False


def test_validate_rejects_live_call_in_frozen_a0(tmp_path: Path) -> None:
    args = _paired_fixture(tmp_path)
    args.frozen_a0_reference = tmp_path / "reference.json"
    args.frozen_a0_reference.write_text(
        json.dumps(
            {
                "schema": "capx_frozen_a0_reference_v1",
                "task": "cube_stack",
                "global_seed": 1000,
                "total_trials": 1,
                "full_replay_sha256": hashlib.sha256(
                    (args.pair_root / "a1_initial_request_hash_replay.json").read_bytes()
                ).hexdigest(),
                "full_replay_entries": 0,
                "initial_state_record_sha256": ["state-sha"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="live or failed model call"):
        MODULE.validate(args)
