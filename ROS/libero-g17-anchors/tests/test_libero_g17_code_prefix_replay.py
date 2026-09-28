"""Unit tests for LIBERO G17 code-prefix replay (no MuJoCo)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from capx.verification import libero_g17_code_prefix_replay as prefix
from capx.verification import libero_g17_fg_projection as projection
from capx.verification import closed_gt
from capx.verification import libero_g17_adapter as adapter
from tests.test_libero_g17_fg_projection import _raw, _spatial_problem


def _write_pkg(tmp_path: Path, *, n_prefix: int = 1) -> Path:
    trials = {
        "1": {
            "source_job": "job-test",
            "init_state_idx": 0,
            "init_state_sha256": "aaa",
            "n_prefix": n_prefix,
            "code_blocks": ["print('cb0')", "print('cb1')"][: max(n_prefix, 1)],
        },
        "2": {
            "source_job": "job-test",
            "init_state_idx": 1,
            "init_state_sha256": "bbb",
            "n_prefix": 1,
            "code_blocks": ["print('t2cb0')"],
        },
    }
    if n_prefix > 1:
        trials["1"]["code_blocks"] = ["print('cb0')", "print('cb1')"]
    path = tmp_path / "pkg.json"
    path.write_text(
        json.dumps(
            prefix.build_package_dict(
                suite_name="libero_spatial",
                task_id=0,
                global_seed=20260718,
                trials=trials,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_load_package_and_identical_pi0_across_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_pkg(tmp_path)
    monkeypatch.setenv(prefix.ENV_PATH, str(path))
    a = prefix.load_package()
    b = prefix.load_package()
    assert a.sha256 == b.sha256
    ca = prefix.prefix_initial_raw_code(a, 1)
    cb = prefix.prefix_initial_raw_code(b, 1)
    assert ca == cb
    assert "print('cb0')" in (ca or "")


def test_init_sha_gate(tmp_path: Path) -> None:
    path = _write_pkg(tmp_path)
    pkg = prefix.load_package(path)

    class Env:
        _last_reset_record = {
            "init_state_idx": 0,
            "init_state_sha256": "aaa",
        }

    prefix.assert_init_matches(Env(), 1, pkg)
    Env._last_reset_record["init_state_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="init sha mismatch"):
        prefix.assert_init_matches(Env(), 1, pkg)


def test_prefix_exhaust_then_live(tmp_path: Path) -> None:
    path = _write_pkg(tmp_path, n_prefix=1)
    pkg = prefix.load_package(path)
    assert prefix.prefix_initial_raw_code(pkg, 1) is not None
    # After CB0 executed (count=1), n_prefix=1 ⇒ no more forced regenerates
    assert prefix.prefix_regenerate_raw_content(pkg, 1, executed_block_count=1) is None


def test_workgap_me_flag_changes_near_miss_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(
        binding.scene,
        _raw(contact_payload_plate=False, z_payload=0.02, payload_xy=(0.05, 0.0)),
    )
    monkeypatch.delenv("CAPX_LIBERO_G17_WORKGAP_ME", raising=False)
    assert projection.classify_endpoint_gap(binding, snap) == "acquisition_incomplete"
    monkeypatch.setenv("CAPX_LIBERO_G17_WORKGAP_ME", "1")
    assert projection.classify_endpoint_gap(binding, snap) == "placement_incomplete"
