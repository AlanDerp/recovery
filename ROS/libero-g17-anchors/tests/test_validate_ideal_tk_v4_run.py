from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from capx.verification import closed_gt, ideal_tk_v4
from tests.test_closed_gt_takeover import _failed_after, _initial, _native_prompt


TOOLS = Path(__file__).parents[1] / "experiments" / "core_fg" / "runtime" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "validate_ideal_tk_v4_run_for_test",
    TOOLS / "validate_ideal_tk_v4_run.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class _LowLevel:
    def _notify_post_sim_step_observer(self) -> None:
        pass


class _Api:
    def do_work(self) -> None:
        pass

    def functions(self) -> dict:
        return {"do_work": self.do_work}


class _Env:
    def __init__(self) -> None:
        self.low_level_env = _LowLevel()
        self._apis = {"control": _Api()}
        self._exec_globals = {}


def _write_v4_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    report = "native visual report"
    env = _Env()
    ideal_tk_v4.initialize_trial_trace(
        env, output_dir=output_dir, trial=1, task="cube_stack"
    )
    ideal_tk_v4.run_takeover(
        env,
        output_dir=output_dir,
        trial=1,
        turn=1,
        task="cube_stack",
        native_report=report,
        native_decision_prompt=_native_prompt(report),
        private_task_completed=False,
    )
    root = tmp_path / "run" / ideal_tk_v4.TRACE_DIR / "trial_01"
    record = json.loads((root / "turn_01" / "takeover.json").read_text())
    return root, record["takeover_prompt_sha256"]


def test_v4_trace_validator_accepts_canonical_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, prompt_hash = _write_v4_trace(tmp_path, monkeypatch)

    count, hashes = VALIDATOR._validate_trace(
        root, action_message_hashes=Counter({prompt_hash: 1})
    )

    assert count == 1
    assert list(hashes) == ["1"]


def test_semantic_gain_noop_preserves_parent_prompt_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPX_IDEAL_TK_V4_RELEASE_BOUND_PLACEMENT", "1")
    monkeypatch.setenv("CAPX_IDEAL_TK_V4_API_TRANSITION_LEDGER", "1")
    monkeypatch.setenv(
        "CAPX_IDEAL_TK_V4_API_TRANSITION_LEDGER_MODE",
        "phase_bound_sparse",
    )
    monkeypatch.setenv("CAPX_IDEAL_TK_V4_SEMANTIC_GAIN_GATE", "1")

    root, prompt_hash = _write_v4_trace(tmp_path, monkeypatch)
    record = json.loads((root / "turn_01" / "takeover.json").read_text())

    assert record["semantic_gain_status"] == "no_bound_release"
    assert record["semantic_gain_extension_applied"] is False
    assert record["parent_prompt_preserved"] is True
    assert record["parent_prompt_sha256"] == record["takeover_prompt_sha256"]
    count, _hashes = VALIDATOR._validate_trace(
        root, action_message_hashes=Counter({prompt_hash: 1})
    )
    assert count == 1


def test_ledger_projection_keeps_full_events_audit_only(tmp_path: Path) -> None:
    path = tmp_path / "takeover.json"

    VALIDATOR._validate_visible_temporal_fields(
        path=path,
        audited_event_fields=["object.obj01.table_contact"],
        visible_event_fields=[],
        ledger_applied=True,
    )
    with pytest.raises(ValueError, match="leaked ordered-event fields"):
        VALIDATOR._validate_visible_temporal_fields(
            path=path,
            audited_event_fields=["object.obj01.table_contact"],
            visible_event_fields=["object.obj01.table_contact"],
            ledger_applied=True,
        )


def test_v4_trace_validator_rejects_projection_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, prompt_hash = _write_v4_trace(tmp_path, monkeypatch)
    report_path = root / "turn_01" / "takeover_report.txt"
    report_path.write_text(
        report_path.read_text().replace("Verified physical endpoint", "Physical endpoint", 1)
    )
    record_path = root / "turn_01" / "takeover.json"
    record = json.loads(record_path.read_text())
    tampered = report_path.read_text()
    record["takeover_report_sha256"] = VALIDATOR._sha256_text(tampered)
    record["canonical_a_star_report_sha256"] = VALIDATOR._sha256_text(tampered)
    record["parent_report_sha256"] = VALIDATOR._sha256_text(tampered)
    record_path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="exact v2.1 report projection"):
        VALIDATOR._validate_trace(
            root, action_message_hashes=Counter({prompt_hash: 1})
        )


def test_v4_trace_validator_accepts_audited_effect_contradiction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _prompt_hash = _write_v4_trace(tmp_path, monkeypatch)
    turn_root = root / "turn_01"
    report_path = turn_root / "takeover_report.txt"
    canonical = report_path.read_text()
    report = (
        canonical
        + VALIDATOR.EFFECT_ALIGNMENT_SEPARATOR
        + "- CONTRADICTED: bilateral grasp of red cube was lost before release."
    )
    report_path.write_text(report)
    record_path = turn_root / "takeover.json"
    record = json.loads(record_path.read_text())
    record.update(
        {
            "takeover_report_sha256": VALIDATOR._sha256_text(report),
            "parent_report_sha256": VALIDATOR._sha256_text(report),
            "effect_alignment_enabled": True,
            "effect_alignment_status": "mismatch",
            "effect_alignment_expected_effects": [
                "GRASP_RETAINED_DURING_POST_GRASP_MOTION"
            ],
            "effect_alignment_contradicted_effects": [
                "GRASP_RETAINED_DURING_POST_GRASP_MOTION"
            ],
            "standalone_api_effect_section_visible": True,
            "canonical_a_star_prompt_unchanged": False,
            "v2_1_projection_equal": False,
            "takeover_prompt_sha256": "aligned-prompt",
            "parent_prompt_sha256": "aligned-prompt",
        }
    )
    record_path.write_text(json.dumps(record))

    count, _hashes = VALIDATOR._validate_trace(
        root, action_message_hashes=Counter({"aligned-prompt": 1})
    )

    assert count == 1


def test_observation_integrity_parser_preserves_exact_g17_parent(
    tmp_path: Path,
) -> None:
    g17 = "Verified canonical G17 report."
    objects = ["obj01"]
    text = VALIDATOR._expected_observation_integrity_text(objects)
    report = g17 + "\n\n" + text

    stripped = VALIDATOR._strip_observation_integrity(
        report,
        applied=True,
        object_ids=objects,
        expected_text_sha256=VALIDATOR._sha256_json(text),
        path=tmp_path / "takeover.json",
    )

    assert stripped == g17


def test_observation_integrity_parser_rejects_silent_or_tampered_suffix(
    tmp_path: Path,
) -> None:
    objects = ["obj01"]
    text = VALIDATOR._expected_observation_integrity_text(objects)
    report = "G17\n\n" + text

    with pytest.raises(ValueError, match="Silent takeover"):
        VALIDATOR._strip_observation_integrity(
            report,
            applied=False,
            object_ids=[],
            expected_text_sha256=VALIDATOR._sha256_json(None),
            path=tmp_path / "takeover.json",
        )
    with pytest.raises(ValueError, match="appendix is invalid"):
        VALIDATOR._strip_observation_integrity(
            report,
            applied=True,
            object_ids=objects,
            expected_text_sha256="wrong-hash",
            path=tmp_path / "takeover.json",
        )


def test_v4_trace_validator_accepts_instance_bound_observation_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _prompt_hash = _write_v4_trace(tmp_path, monkeypatch)
    turn_root = root / "turn_01"
    report_path = turn_root / "takeover_report.txt"
    g17_report = report_path.read_text()
    objects = ["obj01"]
    text = VALIDATOR._expected_observation_integrity_text(objects)
    report = g17_report + "\n\n" + text
    report_path.write_text(report)

    record_path = turn_root / "takeover.json"
    record = json.loads(record_path.read_text())
    g17_prompt_hash = record["takeover_prompt_sha256"]
    audit = [
        {
            "api_call_index": 0,
            "object_id": "obj01",
            "status": "INCONSISTENT",
            "observed_signature_sha256": "a" * 64,
            "expected_signature_sha256": "b" * 64,
        }
    ]
    record.update(
        {
            "takeover_report_sha256": VALIDATOR._sha256_text(report),
            "g17_parent_report_sha256": VALIDATOR._sha256_text(g17_report),
            "g17_parent_prompt_sha256": g17_prompt_hash,
            "g17_parent_prompt_preserved": False,
            "takeover_prompt_sha256": "integrity-prompt",
            "observation_integrity_enabled": True,
            "observation_integrity_applied": True,
            "observation_integrity_inconsistent_objects": objects,
            "observation_integrity_text_sha256": VALIDATOR._sha256_json(text),
            "private_extent_integrity_audit": audit,
            "private_extent_integrity_audit_sha256": VALIDATOR._sha256_json(
                audit
            ),
            "public_executed_api_calls": ["get_object_pose"],
        }
    )
    record_path.write_text(json.dumps(record))

    count, _hashes = VALIDATOR._validate_trace(
        root, action_message_hashes=Counter({"integrity-prompt": 1})
    )

    assert count == 1


def test_v4_trace_validator_accepts_audited_release_bound_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _prompt_hash = _write_v4_trace(tmp_path, monkeypatch)
    turn_root = root / "turn_01"
    report_path = turn_root / "takeover_report.txt"
    report = report_path.read_text().replace(
        "\nVerified physical endpoint after the executed code:",
        "\nVerified release-bound placement result:\n"
        "- red cube relative to green cube when bilateral grasp contact ended "
        "during the executed release: x, y, and z were respectively within "
        "the reference band, within the reference band, and on the positive "
        "side of the reference band; the support region was in the center "
        "region and the footprint was inside; green cube had local z aligned "
        "with the table normal and table support; support was present when "
        "grasp contact ended and was present at the endpoint.\n\n"
        "Verified physical endpoint after the executed code:",
        1,
    )
    report_path.write_text(report)
    record_path = turn_root / "takeover.json"
    record = json.loads(record_path.read_text())
    release = {
        "subject_id": "obj01",
        "reference_id": "obj02",
        "release_call_index": 0,
        "relative_x": "WITHIN_REFERENCE_BAND",
        "relative_y": "WITHIN_REFERENCE_BAND",
        "relative_z": "POSITIVE_OF_REFERENCE_BAND",
        "support_region": "CENTER_REGION",
        "footprint": "INSIDE",
        "reference_orientation": "LOCAL_Z_ALIGNED_WITH_TABLE_NORMAL",
        "reference_stability": "TABLE_SUPPORT",
        "support_at_contact_loss": True,
        "support_first_established": "AT_CONTACT_LOSS",
        "support_at_endpoint": True,
    }
    placements = {
        "placement_outcome.obj01.obj02": "CENTER_SUPPORT_AT_ENDPOINT"
    }
    record.update(
        {
            "takeover_report_sha256": VALIDATOR._sha256_text(report),
            "canonical_a_star_report_sha256": VALIDATOR._sha256_text(report),
            "canonical_v2_1_projection_equal": False,
            "v2_1_projection_equal": False,
            "release_bound_placement_enabled": True,
            "release_bound_placement_applied": True,
            "release_bound_placement": release,
            "release_bound_placement_sha256": VALIDATOR._sha256_json(release),
            "release_bound_placement_candidate": release,
            "release_bound_placement_candidate_sha256": (
                VALIDATOR._sha256_json(release)
            ),
            "semantic_gain_extension_applied": True,
            "parent_prompt_preserved": False,
            "public_interval_placement_outcomes": placements,
            "public_executed_api_calls": ["open_gripper"],
            "takeover_prompt_sha256": "release-bound-prompt",
        }
    )
    semantics = {
        "public_transition": record["public_transition"],
        "public_interval_grasp_outcomes": record[
            "public_interval_grasp_outcomes"
        ],
        "public_interval_placement_outcomes": placements,
        "public_temporal_event_batches": record["public_temporal_event_batches"],
    }
    record["public_turn_semantics_sha256"] = VALIDATOR._sha256_json(semantics)
    record_path.write_text(json.dumps(record))

    count, _hashes = VALIDATOR._validate_trace(
        root, action_message_hashes=Counter({"release-bound-prompt": 1})
    )

    assert count == 1


def test_v4_validator_rejects_runner_level_timeout(tmp_path: Path) -> None:
    summary = tmp_path / "trial_07_sandboxrc_1_reward_0.011_taskcompleted_0" / "summary.txt"
    summary.parent.mkdir()
    summary.write_text(
        "-" * 100
        + "\nTrial 7 timed out after 1800 seconds.\n"
        + "Generated program:\nprint('valid student text')\n"
    )

    with pytest.raises(ValueError, match="runner-level trial chain failures"):
        VALIDATOR._assert_no_trial_chain_failures(tmp_path)


def test_v4_validator_does_not_interpret_student_timeout_text(tmp_path: Path) -> None:
    summary = tmp_path / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1" / "summary.txt"
    summary.parent.mkdir()
    summary.write_text(
        "Generated program:\n"
        "print('Trial 7 timed out after 1800 seconds.')\n"
        "Reward: 1.0\n"
    )

    assert VALIDATOR._assert_no_trial_chain_failures(tmp_path) == 1


def test_v4_validator_accepts_response_frozen_audit(tmp_path: Path) -> None:
    request_hash = "request-hash"
    response_hash = "response-hash"
    replay_path = tmp_path / "response_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "source_run_root": "/source/v4",
                "missing_count": 0,
                "duplicate_request_conflicts": [],
                "entries": {
                    request_hash: {"response_sha256": response_hash}
                },
            }
        )
    )
    manifest = {
        "treatment_replay_mode": "response_frozen_audit",
        "treatment_request_hash_replay_file": str(replay_path),
        "treatment_request_hash_replay_sha256": hashlib.sha256(
            replay_path.read_bytes()
        ).hexdigest(),
    }
    audit = [
        {
            "ok": True,
            "source": "request_hash_replay",
            "replay_request_sha256": request_hash,
            "response_content_sha256": response_hash,
        }
    ]

    result = VALIDATOR._validate_response_frozen_audit(manifest, audit)

    assert result["response_frozen_no_live_model_calls"] is True
    assert result["response_frozen_replay_entries_consumed"] == 1


def test_v4_validator_rejects_live_call_in_response_frozen_audit(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "response_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema": "capx_request_hash_replay_v1",
                "missing_count": 0,
                "duplicate_request_conflicts": [],
                "entries": {"request-hash": {"response_sha256": "response-hash"}},
            }
        )
    )
    manifest = {
        "treatment_replay_mode": "response_frozen_audit",
        "treatment_request_hash_replay_file": str(replay_path),
        "treatment_request_hash_replay_sha256": hashlib.sha256(
            replay_path.read_bytes()
        ).hexdigest(),
    }

    with pytest.raises(ValueError, match="live or failed model call"):
        VALIDATOR._validate_response_frozen_audit(
            manifest,
            [
                {
                    "ok": True,
                    "source": None,
                    "replay_request_sha256": "request-hash",
                    "response_content_sha256": "response-hash",
                }
            ],
        )


def test_semantic_reference_decoder_accepts_direct_and_frozen_carriers(
    tmp_path: Path,
) -> None:
    hashes = {"1": {"1": "semantic-hash"}}
    identity = {
        "task": "cube_stack",
        "native_mode": "M3",
        "global_seed": 6100,
        "total_trials": 1,
    }
    direct = {
        **identity,
        "schema": "capx_closed_gt_takeover_passive_reference_v2.1",
        "capture_mode": "passive_action_invisible",
        "student_prompt_changed": False,
        "student_action_request_changed": False,
        "source_run_root": str(tmp_path),
        "closed_gt_turn_semantics_sha256": hashes,
    }
    frozen = {
        **identity,
        "schema": "capx_frozen_a0_reference_v1",
        "m3_semantic_source_capture_mode": "passive_action_invisible",
        "m3_semantic_source_reference_sha256": "source-hash",
        "source_pair_validation_sha256": "validation-hash",
        "m3_semantic_transition_sha256": hashes,
    }

    for reference in (direct, frozen):
        assert VALIDATOR._expected_semantic_hashes(
            reference,
            task="cube_stack",
            global_seed=6100,
            trials=1,
            a1_run_root=tmp_path,
        ) == hashes
