"""Unit tests for LIBERO G17 terminal success witness (Layer-1)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from capx.verification import libero_g17_takeover as takeover
from capx.verification import libero_g17_fg_projection as projection


def test_schema_is_interaction_referents_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAPX_LIBERO_G17_WORKGAP_ME", raising=False)
    monkeypatch.delenv("CAPX_LIBERO_G17_L1_SPARSE", raising=False)
    assert "same-name-xy-logic" in takeover.schema_version()


def test_schema_sparse_interaction_referents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAPX_LIBERO_G17_WORKGAP_ME", raising=False)
    monkeypatch.setenv("CAPX_LIBERO_G17_L1_SPARSE", "1")
    assert "same-name-xy-logic" in takeover.schema_version()


def test_placement_xy_facts_in_endpoint_report() -> None:
    """v2.23: v213 placement + same-name ordinal w.r.t. robot."""
    src = Path(projection.__file__).read_text(encoding="utf-8")
    assert "wrong-grasp same-name acquire" in src
    assert "first {name} from the robot's right" in src
    assert 'sample_grasp_pose("<str>") maps to:' not in src
    assert "center_to_center offset:" in src
    assert "Active placement work" not in src
    assert "goal_object acquire language target:" not in src


def test_capture_terminal_witness_writes_finish_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CAPX_LIBERO_G17_WORKGAP_ME", raising=False)
    monkeypatch.delenv("CAPX_LIBERO_G17_L1_SPARSE", raising=False)
    if "success-witness" not in takeover.SCHEMA_VERSION:
        pytest.skip("schema is not success-witness")

    trial = 1
    turn = 1
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    root = output_dir / takeover.TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True)

    # Minimal private cache + state as initialize_trial_trace would leave them.
    before_values = {"sentinel": "before"}
    after_values = {"sentinel": "after"}
    before = SimpleNamespace(values=before_values)
    after = SimpleNamespace(values=after_values)
    key = takeover._cache_key(str(output_dir), trial)
    takeover._PRIVATE_SNAPSHOTS[key] = [before]
    binding = SimpleNamespace(
        relation="on",
        suite_name="libero_spatial",
        task_id=0,
        containment_geom=None,
        scene=SimpleNamespace(objects=[SimpleNamespace(public_name="bowl"), SimpleNamespace(public_name="plate")]),
        name="mock",
    )
    takeover._PRIVATE_BINDINGS[key] = binding
    (root / "state.json").write_text(
        json.dumps(
            {
                "schema_version": takeover.SCHEMA_VERSION,
                "trial": trial,
                "turns_completed": 0,
            }
        ),
        encoding="utf-8",
    )

    fake_proj = SimpleNamespace(
        combined_report=(
            "Verified ordered state changes over the executed interval:\n"
            "- no eye-checkable state changes.\n\n"
            "Verified physical endpoint for the bound placement pair:\n"
            "Verified acquisition evidence at the endpoint:\n"
            "- robot gripper aperture: open.\n"
            "Verified placement evidence at the endpoint:\n"
            "- contact between bowl and plate: true.\n"
            "- note: public evidence authorizes finishing the task."
        ),
        parent_report="Verified ordered state changes over the executed interval:\n- no eye-checkable state changes.",
        endpoint_report="Verified physical endpoint for the bound placement pair:\n- note: public evidence authorizes finishing the task.",
        release_report="",
        gain=SimpleNamespace(status="no_bound_release", bound_release=False, novel_fact_ids=()),
        world_fg_complete=True,
        endpoint_gap="placement_incomplete",
        finish_authorized=True,
        goal_evidence_keys=("a", "b"),
        parent_sha256="p" * 64,
        combined_sha256="c" * 64,
        executor_attachment="empty",
        interaction_grasp="none",
    )

    monkeypatch.setattr(
        takeover.adapter,
        "extract_libero_snapshot",
        lambda env, binding: (
            binding,
            after,
            takeover.adapter.LiberoAttachmentFact("empty"),
        ),
    )
    monkeypatch.setattr(
        takeover.projection,
        "project_libero_fg",
        lambda binding, before, after, attachment=None: fake_proj,
    )
    monkeypatch.setattr(
        takeover.projection,
        "parent_byte_identity_preserved",
        lambda proj: True,
    )
    monkeypatch.setattr(
        takeover.projection,
        "acquire_evidence_keys",
        lambda binding: ("acq",),
    )
    monkeypatch.setattr(
        takeover.projection,
        "place_evidence_keys",
        lambda binding: ("plc",),
    )
    monkeypatch.setattr(
        takeover.adapter,
        "g17_versus_bddl_audit",
        lambda after, env, binding: {
            "bddl_atom_true": True,
            "private_task_completed": True,
            "bddl_iff_task_completed": True,
            "geometric_proxy": True,
            "g17_iff_bddl": True,
        },
    )

    record = takeover.capture_terminal_witness(
        MagicMock(),
        output_dir=str(output_dir),
        trial=trial,
        turn=turn,
        task="libero_spatial_0",
        private_task_completed=True,
        stop_reason="task_completed",
    )
    assert record["terminal_witness"] is True
    assert record["finish_authorized"] is True
    assert record["bddl_atom_true"] is True
    assert record["carrier"] == "terminal_success_witness_libero_g17_fg"
    turn_rec = json.loads((root / "turn_01" / "record.json").read_text())
    assert turn_rec["finish_authorized"] is True
    state = json.loads((root / "state.json").read_text())
    assert state["terminal_witness"]["finish_authorized"] is True
    assert len(takeover._PRIVATE_SNAPSHOTS[key]) == 2

    # cleanup module cache for other tests
    del takeover._PRIVATE_SNAPSHOTS[key]
    del takeover._PRIVATE_BINDINGS[key]


def test_capture_terminal_witness_fail_closed_on_iff_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if "success-witness" not in takeover.SCHEMA_VERSION:
        pytest.skip("schema is not success-witness")

    trial = 2
    turn = 1
    output_dir = tmp_path / "eval2"
    output_dir.mkdir()
    root = output_dir / takeover.TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True)
    before = SimpleNamespace(values={})
    after = SimpleNamespace(values={})
    key = takeover._cache_key(str(output_dir), trial)
    takeover._PRIVATE_SNAPSHOTS[key] = [before]
    takeover._PRIVATE_BINDINGS[key] = SimpleNamespace(relation="on")
    (root / "state.json").write_text(
        json.dumps(
            {
                "schema_version": takeover.SCHEMA_VERSION,
                "trial": trial,
                "turns_completed": 0,
            }
        ),
        encoding="utf-8",
    )
    fake_proj = SimpleNamespace(
        combined_report="Verified ordered state changes over the executed interval:\n- no eye-checkable state changes.\n\nendpoint",
        parent_report="Verified ordered state changes over the executed interval:\n- no eye-checkable state changes.",
        endpoint_report="endpoint",
        release_report="",
        gain=SimpleNamespace(status="x", bound_release=False, novel_fact_ids=()),
        world_fg_complete=True,
        endpoint_gap="placement_incomplete",
        finish_authorized=False,
        goal_evidence_keys=(),
        parent_sha256="p" * 64,
        combined_sha256="c" * 64,
        executor_attachment="empty",
        interaction_grasp="none",
    )
    monkeypatch.setattr(
        takeover.adapter,
        "extract_libero_snapshot",
        lambda e, b: (b, after, takeover.adapter.LiberoAttachmentFact("empty")),
    )
    monkeypatch.setattr(
        takeover.projection, "project_libero_fg", lambda *a, **k: fake_proj
    )
    monkeypatch.setattr(
        takeover.adapter,
        "g17_versus_bddl_audit",
        lambda *a, **k: {
            "bddl_atom_true": True,
            "private_task_completed": True,
            "bddl_iff_task_completed": True,
            "geometric_proxy": False,
            "g17_iff_bddl": True,
        },
    )
    with pytest.raises(RuntimeError, match="finish_authorized=False"):
        takeover.capture_terminal_witness(
            MagicMock(),
            output_dir=str(output_dir),
            trial=trial,
            turn=turn,
            task="t",
            private_task_completed=True,
            stop_reason="task_completed",
        )
    del takeover._PRIVATE_SNAPSHOTS[key]
    del takeover._PRIVATE_BINDINGS[key]
