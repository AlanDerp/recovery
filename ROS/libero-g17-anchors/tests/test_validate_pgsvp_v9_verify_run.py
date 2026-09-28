from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


TOOLS = (
    Path(__file__).parents[1]
    / "experiments"
    / "core_fg"
    / "runtime"
    / "tools"
)
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "validate_pgsvp_v9_verify_run",
    TOOLS / "validate_pgsvp_v9_verify_run.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_responses(root: Path, *, initial: str, multiturn: str) -> None:
    path = root / "model" / "task" / "trial" / "all_responses.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                {"decision": "initial", "initial_prompt": initial},
                {"decision": "regenerate", "multi_turn_prompt": multiturn},
            ]
        ),
        encoding="utf-8",
    )


def _grounded_prompt() -> str:
    return f"{MODULE.PUBLIC_BRIDGE_MARKER}\ncontract\n\ncanonical state"


def test_saved_prompt_audit_accepts_isolated_a1_and_grounded_treatment(
    tmp_path: Path,
) -> None:
    a1 = tmp_path / "a1"
    treatment = tmp_path / "treatment"
    _write_responses(a1, initial="initial", multiturn="ordinary evidence")
    _write_responses(treatment, initial="initial", multiturn=_grounded_prompt())

    MODULE._validate_saved_prompt_isolation(run_root=a1, treatment=False)
    MODULE._validate_saved_prompt_isolation(
        run_root=treatment,
        treatment=True,
    )


def test_saved_prompt_audit_rejects_a1_bridge_leak(tmp_path: Path) -> None:
    root = tmp_path / "a1"
    _write_responses(root, initial="initial", multiturn=_grounded_prompt())

    with pytest.raises(ValueError, match="occurrence mismatch"):
        MODULE._validate_saved_prompt_isolation(run_root=root, treatment=False)


@pytest.mark.parametrize(
    "multiturn",
    [
        "ordinary evidence",
        f"{_grounded_prompt()}\n\n{_grounded_prompt()}",
    ],
)
def test_saved_prompt_audit_rejects_missing_or_duplicate_treatment_bridge(
    tmp_path: Path,
    multiturn: str,
) -> None:
    root = tmp_path / "treatment"
    _write_responses(root, initial="initial", multiturn=multiturn)

    with pytest.raises(ValueError, match="occurrence mismatch"):
        MODULE._validate_saved_prompt_isolation(run_root=root, treatment=True)


@pytest.mark.parametrize(
    "private_marker",
    [
        MODULE.PT_REQUEST_MARKER,
        MODULE.DELTA_HEADER,
        MODULE.PUBLIC_HEADER,
    ],
)
def test_saved_prompt_audit_rejects_private_protocol_in_action(
    tmp_path: Path,
    private_marker: str,
) -> None:
    root = tmp_path / "treatment"
    _write_responses(
        root,
        initial="initial",
        multiturn=f"{_grounded_prompt()}\n{private_marker}",
    )

    with pytest.raises(ValueError, match="occurrence mismatch"):
        MODULE._validate_saved_prompt_isolation(run_root=root, treatment=True)


def test_frozen_reference_only_provenance_is_hash_bound(tmp_path: Path) -> None:
    source_validation = tmp_path / "source_validation.json"
    source_validation.write_text(
        json.dumps({"status": "valid", "task": "cube_stack", "trials": 10})
        + "\n"
    )
    source_sha = hashlib.sha256(source_validation.read_bytes()).hexdigest()
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"source_pair_validation_sha256": source_sha}) + "\n"
    )

    mode = MODULE._validate_a0_execution_provenance(
        pair_manifest={
            "a0_execution_mode": "frozen_reference_only",
            "a1_reexecuted": "false",
            "same_slurm_allocation": "false",
            "frozen_a0_source_pair_validation": str(source_validation),
            "frozen_a0_source_pair_validation_sha256": source_sha,
        },
        frozen_a0_reference=reference,
        task="cube_stack",
        trials=10,
    )

    assert mode == "frozen_reference_only"


def test_frozen_reference_only_rejects_unregistered_source(
    tmp_path: Path,
) -> None:
    source_validation = tmp_path / "source_validation.json"
    source_validation.write_text(
        json.dumps({"status": "valid", "task": "cube_stack", "trials": 10})
        + "\n"
    )
    source_sha = hashlib.sha256(source_validation.read_bytes()).hexdigest()
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"source_pair_validation_sha256": "0" * 64}) + "\n"
    )

    with pytest.raises(ValueError, match="not registered"):
        MODULE._validate_a0_execution_provenance(
            pair_manifest={
                "a0_execution_mode": "frozen_reference_only",
                "a1_reexecuted": "false",
                "same_slurm_allocation": "false",
                "frozen_a0_source_pair_validation": str(source_validation),
                "frozen_a0_source_pair_validation_sha256": source_sha,
            },
            frozen_a0_reference=reference,
            task="cube_stack",
            trials=10,
        )
