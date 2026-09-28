from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).parents[1] / "experiments" / "core_fg" / "runtime" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "register_frozen_a0_for_test",
    TOOLS / "register_frozen_a0.py",
)
assert SPEC is not None and SPEC.loader is not None
REGISTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGISTER)


@pytest.mark.parametrize(
    ("schema", "key"),
    [
        (
            "capx_fgsrp_passive_semantic_reference_v1",
            "m3_semantic_transition_sha256",
        ),
        (
            "capx_closed_gt_takeover_passive_reference_v2.1",
            "closed_gt_turn_semantics_sha256",
        ),
    ],
)
def test_semantic_hashes_preserve_supported_source_schema(
    schema: str, key: str
) -> None:
    hashes = {"1": {"1": "digest"}}

    actual_hashes, actual_key = REGISTER._semantic_hashes(
        {"schema": schema, key: hashes}
    )

    assert actual_hashes is hashes
    assert actual_key == key


def test_semantic_hashes_reject_unknown_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported M3 passive semantic schema"):
        REGISTER._semantic_hashes({"schema": "unknown"})
