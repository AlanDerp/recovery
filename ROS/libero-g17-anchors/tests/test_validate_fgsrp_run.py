from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).parents[1] / "experiments" / "core_fg" / "runtime" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "validate_fgsrp_run_for_test",
    TOOLS / "validate_fgsrp_run.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
RUNNER = Path(__file__).parents[1] / "experiments" / "core_fg" / "run_paired_task.sbatch"


def _verifier_record(*, contradicted: bool = False) -> dict:
    disposition = (
        "replace_contradicted"
        if contradicted
        else "preserve_compatible_or_undecidable"
    )
    proposal = {
        "original_claim": "claim",
        "replacement_claim": "correction",
    }
    return {
        "protocol": {"sentence_count": 1, "claim_count": 1},
        "proposals": [proposal] if contradicted else [],
        "claim_ledger": [
            {
                "sentence_index": 0,
                "sentence_text": "claim",
                "claims": [
                    {
                        "original_claim": "claim",
                        "disposition": disposition,
                    }
                ],
            }
        ],
    }


def _write_trace(root: Path, *, changed: bool = False) -> str:
    root.mkdir(parents=True)
    state = {
        "schema_version": VALIDATOR.SCHEMA_VERSION,
        "snapshots": [
            {"turn": 0, "evidence": {}},
            {"turn": 1, "evidence": {}},
        ],
    }
    (root / "fgsrp_state.json").write_text(json.dumps(state))
    (root / "initialization.json").write_text(
        json.dumps(
            {
                "schema_version": VALIDATOR.SCHEMA_VERSION,
                "action_visible_gt_packet_injected": False,
                "action_visible_bridge_injected": False,
            }
        )
    )
    prompt = [{"role": "user", "content": "corrected"}]
    prompt_hash = VALIDATOR._sha256_json(prompt)
    turn = root / "turn_01"
    turn.mkdir()
    record = {
        "schema_version": VALIDATOR.SCHEMA_VERSION,
        "turn": 1,
        "carrier": "m3_visual_difference_report",
        "non_text_structure_identical": True,
        "action_visible_gt_packet_injected": False,
        "action_visible_bridge_injected": False,
        "action_visible_verifier_protocol_injected": False,
        "raw_image_replaced": False,
        "console_stdout_replaced": False,
        "console_stderr_replaced": False,
        "task_or_api_text_replaced": changed,
        "corrected_prompt_sha256": prompt_hash,
        "native_prompt_sha256": prompt_hash,
        "repairs": [],
        "applied_patches": [],
        "repair_count": 0,
        "active_repair_count": 0,
        "no_op_prompt_byte_equivalent": True,
        "verification_protocol": {
            "verifier": _verifier_record(),
            "adjudicator": None,
            "known_false_claims": [],
        },
    }
    (turn / "replacement.json").write_text(json.dumps(record))
    return prompt_hash


def test_trace_validator_accepts_audited_no_op(tmp_path: Path) -> None:
    root = tmp_path / "trial_01"
    prompt_hash = _write_trace(root)

    result = VALIDATOR._validate_trace(
        root,
        carrier="m3_visual_difference_report",
        action_message_hashes=VALIDATOR.Counter({prompt_hash: 1}),
    )

    assert result == {"turn_count": 1, "repair_count": 0, "no_op_count": 1}


def test_runner_tracks_embedded_m3_semantics_by_source_not_copied_path() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert "frozen_a0_semantic_reference_embedded=1" in runner
    assert '&& "$frozen_a0_semantic_reference_embedded" != "1"' in runner
    assert '&& "$m3_semantic_reference" != "$frozen_a0_reference"' not in runner


def test_trace_validator_rejects_immutable_surface_change(tmp_path: Path) -> None:
    root = tmp_path / "trial_01"
    prompt_hash = _write_trace(root, changed=True)

    with pytest.raises(ValueError, match="task_or_api_text_replaced"):
        VALIDATOR._validate_trace(
            root,
            carrier="m3_visual_difference_report",
            action_message_hashes=VALIDATOR.Counter({prompt_hash: 1}),
        )


def test_trace_validator_requires_corrected_request_to_reach_api(tmp_path: Path) -> None:
    root = tmp_path / "trial_01"
    _write_trace(root)

    with pytest.raises(ValueError, match="was not sent"):
        VALIDATOR._validate_trace(
            root,
            carrier="m3_visual_difference_report",
            action_message_hashes=VALIDATOR.Counter(),
        )


def test_trace_validator_requires_exhaustive_claim_ledger(tmp_path: Path) -> None:
    root = tmp_path / "trial_01"
    prompt_hash = _write_trace(root)
    path = root / "turn_01" / "replacement.json"
    record = json.loads(path.read_text())
    record["verification_protocol"]["verifier"].pop("claim_ledger")
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="exhaustive claim ledger"):
        VALIDATOR._validate_trace(
            root,
            carrier="m3_visual_difference_report",
            action_message_hashes=VALIDATOR.Counter({prompt_hash: 1}),
        )


def test_trace_validator_rejects_unresolved_known_false_claim(tmp_path: Path) -> None:
    root = tmp_path / "trial_01"
    prompt_hash = _write_trace(root)
    path = root / "turn_01" / "replacement.json"
    record = json.loads(path.read_text())
    record["verification_protocol"]["known_false_claims"] = [
        {
            "original_claim": "the red cube is on the green cube",
            "resolved": False,
        }
    ]
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="known-false claim unresolved"):
        VALIDATOR._validate_trace(
            root,
            carrier="m3_visual_difference_report",
            action_message_hashes=VALIDATOR.Counter({prompt_hash: 1}),
        )


def test_trace_validator_accepts_resolved_atomic_refinement(tmp_path: Path) -> None:
    root = tmp_path / "trial_01"
    _write_trace(root)
    path = root / "turn_01" / "replacement.json"
    record = json.loads(path.read_text())
    corrected_hash = VALIDATOR._sha256_json(
        [{"role": "user", "content": "atomically corrected"}]
    )
    record.update(
        {
            "corrected_prompt_sha256": corrected_hash,
            "repairs": [{"original_claim": "false", "replacement_claim": "true"}],
            "applied_patches": [
                {"original_claim": "false", "replacement_claim": "true"}
            ],
            "repair_count": 1,
            "active_repair_count": 1,
            "no_op_prompt_byte_equivalent": None,
            "verification_protocol": {
                "verifier": _verifier_record(contradicted=True),
                "adjudicator": {
                    "decisions": [
                        {
                            "repair_index": 0,
                            "relation": "contradicted",
                            "accepted": False,
                        }
                    ]
                },
                "deterministic_gate": [
                    {
                        "repair_index": 0,
                        "accepted": True,
                        "llm_gate_accepted": False,
                    }
                ],
                "known_false_claims": [
                    {"original_claim": "broad false", "resolved": True}
                ],
                "atomic_refinement": {
                    "status": "resolved",
                    "host_certificates": [{"accepted": True}],
                    "gate": {
                        "decisions": [{"repair_index": 0, "accepted": True}]
                    },
                },
            },
        }
    )
    path.write_text(json.dumps(record))

    result = VALIDATOR._validate_trace(
        root,
        carrier="m3_visual_difference_report",
        action_message_hashes=VALIDATOR.Counter({corrected_hash: 1}),
    )

    assert result == {"turn_count": 1, "repair_count": 1, "no_op_count": 0}
