"""Unit tests for near-miss start/step oscillator (no sim)."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np

from capx.verification import libero_g17_nearmiss_osc as osc
from capx.verification import libero_g17_takeover as takeover
from capx.verification.libero_g17_public_iff import LIBERO_OBJECT_ONTOP_XY_M


def test_axis_magnitude_schedule() -> None:
    start, step = 1.0, 0.1
    mags = []
    t = 0
    while True:
        m = osc.axis_magnitude(start, step, t)
        if m is None:
            break
        mags.append(m)
        t += 1
    assert mags[0] == 1.0
    assert abs(mags[-1] - 0.1) < 1e-12
    assert len(mags) == 10


def test_example_residual_converges_like_user_math() -> None:
    """User example: r0=0.5, start=1, step=0.1 → oscillate into ±0.1."""
    start, step = 1.0, 0.1
    r = 0.5
    t = 0
    hist = [r]
    while abs(r) >= step:
        mag = osc.axis_magnitude(start, step, t)
        assert mag is not None
        d = osc.axis_delta(r, mag, step)
        r = r + d
        hist.append(r)
        t += 1
        if t > 20:
            break
    assert abs(r) < step
    # First step overshoots: 0.5 - 1.0 = -0.5
    assert abs(hist[1] + 0.5) < 1e-9


def test_axis_delta_skips_inside_band() -> None:
    assert osc.axis_delta(0.01, 0.08, 0.03) == 0.0
    assert osc.axis_delta(0.05, 0.08, 0.03) == -0.08
    assert osc.axis_delta(-0.05, 0.08, 0.03) == 0.08


def test_schema_v248_nearmiss_osc() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_NEARMISS_OSC",
            "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
            "CAPX_LIBERO_G17_RELEASE_READY_GOTO",
            "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
            "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC"] = "1"
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
        "CAPX_LIBERO_G17_RELEASE_READY_GOTO",
        "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
        "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.48" in s and "nearmiss-osc-tipxy" in s
        assert "bandwindow" not in s
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_schema_v254_nearmiss_osc_bandwindow() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_NEARMISS_OSC",
            "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_RELEASE_READY_GOTO",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW"] = "1"
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
        "CAPX_LIBERO_G17_RELEASE_READY_GOTO",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.54" in s and "nearmiss-osc-tipxy-bandwindow" in s
        assert "fair-residual" not in s
        assert osc.nearmiss_osc_bandwindow_enabled()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_schema_v255_nearmiss_osc_fair_residual() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_NEARMISS_OSC",
            "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
            "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL"] = "1"
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.55" in s and "bandwindow-fair-residual" in s
        assert osc.nearmiss_osc_fair_residual()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_schema_v256_xy_decouple_predicate() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_NEARMISS_OSC",
            "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
            "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
            "CAPX_LIBERO_G17_NEARMISS_OSC_STUDENT_SIGN",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_STUDENT_SIGN"] = "1"
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.56" in s and "student-sign" in s
        assert osc.nearmiss_osc_student_sign()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_schema_v257_xy_decouple_fg_sign() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_NEARMISS_OSC",
            "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
            "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
            "CAPX_LIBERO_G17_NEARMISS_OSC_STUDENT_SIGN",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE"] = "1"
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_STUDENT_SIGN",
        "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.57" in s and "fg-sign" in s
        assert not osc.nearmiss_osc_student_sign()
        assert osc.nearmiss_osc_xy_decouple_enabled()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_xy_decouple_independent_schedules() -> None:
    """X and Y advance independent t; mag never equals residual (no vector anchor)."""
    start, step = 0.15, 0.03
    # Axis X at t=0, Y at t=2 → different mags in one logical round.
    assert osc.axis_magnitude(start, step, 0) == 0.15
    assert abs(osc.axis_magnitude(start, step, 2) - 0.09) < 1e-12
    # Sign-only: |d| == mag, not |r|
    d = osc.axis_delta(0.04, 0.15, 0.03)
    assert abs(abs(d) - 0.15) < 1e-12


def test_proof_schedule_15_minus_3t() -> None:
    """Fixed start=0.15, step=0.03 → 15,12,9,6,3 cm then exhausted."""
    start, step = 0.15, 0.03
    mags = []
    t = 0
    while True:
        m = osc.axis_magnitude(start, step, t)
        if m is None:
            break
        mags.append(round(m, 6))
        t += 1
    assert mags == [0.15, 0.12, 0.09, 0.06, 0.03]
    # sign-only: magnitude never equals residual
    d = osc.axis_delta(0.04, 0.15, 0.03)
    assert abs(d + 0.15) < 1e-12


def test_step_matches_ontop_band() -> None:
    assert LIBERO_OBJECT_ONTOP_XY_M == 0.03


def test_fg_predicate_wash_isomorphic_to_axis_delta() -> None:
    """Band → delta must match -sign(r)·mag with |r|<step skip (v256 wash)."""
    from capx.verification.libero_g17_band_obligations import residual_to_band

    step, mag = 0.03, 0.15
    for r in (-0.12, -0.03, -0.029, -1e-9, 0.0, 1e-9, 0.029, 0.03, 0.12, 0.5):
        band = residual_to_band(axis="x", residual=r, band_m=step)
        d_band = osc.axis_delta_from_fg_band(band, mag)
        d_float = osc.axis_delta(r, mag, step)
        assert abs(d_band - d_float) < 1e-12, (r, band, d_band, d_float)


def test_v269_mech_materializes_equal_gt_fg_bands() -> None:
    """The selected source changes provenance only, never the discrete action."""
    from types import SimpleNamespace

    snapshot = SimpleNamespace(
        private_poses={
            "obj01": (np.array([0.10, -0.01, 0.0]), None),
            "obj02": (np.array([0.00, 0.00, 0.0]), None),
        }
    )
    gate = SimpleNamespace(residual_xy=(0.10, -0.01))
    previous = os.environ.get("CAPX_LIBERO_G17_V269_SYMBOL_SOURCE")
    try:
        for source in ("gt", "fg"):
            os.environ["CAPX_LIBERO_G17_V269_SYMBOL_SOURCE"] = source
            selected, gt, fg, actual_source = osc._v269_bands(
                snapshot, gate, band_m=0.03
            )
            assert selected == gt == fg
            assert actual_source == source
            deltas = tuple(osc.axis_delta_from_fg_band(band, 0.15) for band in selected)
            assert deltas == (-0.15, 0.0)
    finally:
        if previous is None:
            os.environ.pop("CAPX_LIBERO_G17_V269_SYMBOL_SOURCE", None)
        else:
            os.environ["CAPX_LIBERO_G17_V269_SYMBOL_SOURCE"] = previous


def test_v274_keeps_15_minus_3_schedule_but_closes_square_circle_gap() -> None:
    from capx.verification.libero_g17_band_obligations import residual_to_band

    assert osc.axis_magnitude(0.15, 0.03, 0) == 0.15
    assert abs(osc.axis_magnitude(0.15, 0.03, 4) - 0.03) < 1e-12
    band_m = osc.v274_axis_skip_band_m()
    assert abs(band_m - 0.03 / math.sqrt(2.0)) < 1e-12
    # The v273 deadlock corner was skipped by each 3 cm axis predicate even
    # though it was outside the 3 cm radial release disk.
    assert residual_to_band(axis="x", residual=0.025, band_m=0.03) == "WITHIN_REFERENCE_BAND"
    assert residual_to_band(axis="x", residual=0.025, band_m=band_m) == "POSITIVE_OF_REFERENCE_BAND"
    assert math.hypot(band_m, band_m) <= 0.03 + 1e-12


def test_v274_retries_rejected_motion_without_advancing_t_axis(monkeypatch) -> None:
    from capx.integrations.franka import tip_xy_only
    from capx.verification import libero_g17_release_ready as release_ready

    state = {"done": False, "move_calls": 0}
    events: list[dict] = []

    def read_live(_api):
        residual = 0.05
        snapshot = SimpleNamespace(
            private_poses={
                "obj01": (np.array([residual, 0.0, 0.0]), None),
                "obj02": (np.array([0.0, 0.0, 0.0]), None),
            }
        )
        gate = SimpleNamespace(
            relation="on",
            acquired=True,
            release_ok_tipxy=state["done"],
            needs_align_tipxy=not state["done"],
            residual_xy=(residual, 0.0),
        )
        return object(), snapshot, gate

    def move(_api, dx, dy, **_kwargs):
        state["move_calls"] += 1
        accepted = state["move_calls"] == 2
        if accepted:
            state["done"] = True
        return SimpleNamespace(
            ok=accepted,
            acceptance_ok=accepted,
            d_tip_z=0.0,
            achieved_xy=(dx if accepted else 0.0, dy),
            final_xy_err=0.0 if accepted else abs(dx),
        )

    def audit(_api, event, **payload):
        events.append({"event": event, **payload})

    monkeypatch.setattr(release_ready, "_read_live", read_live)
    monkeypatch.setattr(tip_xy_only, "move_tip_xy", move)
    monkeypatch.setattr(osc, "_v269_audit", audit)
    monkeypatch.setenv("CAPX_LIBERO_G17_V269_SYMBOL_SOURCE", "fg")
    api = SimpleNamespace(_capx_release_ready_busy=False)

    assert osc._run_v274_mech_contract(api)
    assert state["move_calls"] == 2
    decisions = [row for row in events if row["event"] == "axis_decision"]
    assert [(row["t"], row["axis"], row["attempt"]) for row in decisions] == [
        (0, "x", 0),
        (0, "x", 1),
    ]
    assert any(row["event"] == "axis_rejected_no_advance" for row in events)
    assert not any(row["event"] == "axis_retry_exhausted" for row in events)


def test_v286_reobserves_final_fg_axes_before_schedule_refusal(monkeypatch) -> None:
    from capx.verification import libero_g17_release_ready as release_ready

    state = {"reads": 0}
    events: list[dict] = []

    def read_live(_api):
        state["reads"] += 1
        # Entry is outside; the schedule-exhaustion observation is centered.
        residual = 0.05 if state["reads"] == 1 else 0.0
        snapshot = SimpleNamespace(
            private_poses={
                "obj01": (np.array([residual, residual, 0.0]), None),
                "obj02": (np.array([0.0, 0.0, 0.0]), None),
            }
        )
        gate = SimpleNamespace(
            relation="on",
            acquired=True,
            release_ok_tipxy=False,
            needs_align_tipxy=True,
            residual_xy=(residual, residual),
        )
        return object(), snapshot, gate

    monkeypatch.setattr(release_ready, "_read_live", read_live)
    monkeypatch.setattr(osc, "axis_magnitude", lambda *_args: None)
    monkeypatch.setattr(
        osc,
        "_v269_audit",
        lambda _api, event, **data: events.append({"event": event, **data}),
    )
    monkeypatch.setenv("CAPX_LIBERO_G17_V269_SYMBOL_SOURCE", "fg")
    monkeypatch.setenv("CAPX_LIBERO_G17_V286_FINAL_FG_CHECK", "1")

    assert osc._run_v274_mech_contract(
        SimpleNamespace(_capx_release_ready_busy=False)
    )
    final = next(row for row in events if row["event"] == "final_fg_check")
    assert final["decision"] == "allow"
    assert final["x_band"] == final["y_band"] == "WITHIN_REFERENCE_BAND"


def test_schema_v274_mechanical_contract() -> None:
    keys = (
        "CAPX_LIBERO_G17_XY_LOCK",
        "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
        "CAPX_LIBERO_G17_RELEASE_READY",
        "CAPX_LIBERO_G17_NEARMISS_OSC",
        "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
        "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH",
        "CAPX_LIBERO_G17_V269_MECH",
        "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE",
        "CAPX_LIBERO_G17_V272_FAIR_LIVE",
        "CAPX_LIBERO_G17_V274_MECH_CONTRACT",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys[:7]:
            os.environ[key] = "1"
        os.environ["CAPX_LIBERO_G17_V269_SYMBOL_SOURCE"] = "fg"
        os.environ["CAPX_LIBERO_G17_V272_FAIR_LIVE"] = "1"
        os.environ["CAPX_LIBERO_G17_V274_MECH_CONTRACT"] = "1"
        schema = takeover.schema_version()
        assert "v2.74" in schema
        assert "fair-live-inscribed-acceptance-gated" in schema
        assert osc.nearmiss_osc_v274_mech_contract_enabled()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_schema_v275_trusted_tipxy() -> None:
    keys = (
        "CAPX_LIBERO_G17_XY_LOCK",
        "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
        "CAPX_LIBERO_G17_RELEASE_READY",
        "CAPX_LIBERO_G17_NEARMISS_OSC",
        "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
        "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH",
        "CAPX_LIBERO_G17_V269_MECH",
        "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE",
        "CAPX_LIBERO_G17_V272_FAIR_LIVE",
        "CAPX_LIBERO_G17_V274_MECH_CONTRACT",
        "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys[:7]:
            os.environ[key] = "1"
        os.environ["CAPX_LIBERO_G17_V269_SYMBOL_SOURCE"] = "fg"
        os.environ["CAPX_LIBERO_G17_V272_FAIR_LIVE"] = "1"
        os.environ["CAPX_LIBERO_G17_V274_MECH_CONTRACT"] = "1"
        os.environ["CAPX_LIBERO_G17_V275_TRUSTED_TIPXY"] = "1"
        schema = takeover.schema_version()
        assert "v2.75" in schema
        assert "transactional-tipxy" in schema
        assert osc.nearmiss_osc_v275_trusted_tipxy_enabled()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_schema_v276_completion_local_ik() -> None:
    keys = (
        "CAPX_LIBERO_G17_XY_LOCK",
        "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
        "CAPX_LIBERO_G17_RELEASE_READY",
        "CAPX_LIBERO_G17_NEARMISS_OSC",
        "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
        "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH",
        "CAPX_LIBERO_G17_V269_MECH",
        "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE",
        "CAPX_LIBERO_G17_V272_FAIR_LIVE",
        "CAPX_LIBERO_G17_V274_MECH_CONTRACT",
        "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY",
        "CAPX_LIBERO_G17_V276_COMPLETION_IK",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys[:7]:
            os.environ[key] = "1"
        os.environ["CAPX_LIBERO_G17_V269_SYMBOL_SOURCE"] = "fg"
        os.environ["CAPX_LIBERO_G17_V272_FAIR_LIVE"] = "1"
        os.environ["CAPX_LIBERO_G17_V274_MECH_CONTRACT"] = "1"
        os.environ.pop("CAPX_LIBERO_G17_V275_TRUSTED_TIPXY", None)
        os.environ["CAPX_LIBERO_G17_V276_COMPLETION_IK"] = "1"
        schema = takeover.schema_version()
        assert "v2.76" in schema
        assert "completion-local-ik" in schema
        assert osc.nearmiss_osc_v276_completion_ik_enabled()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_v276_large_z_is_diagnostic_when_xy_completed_and_still_held(
    monkeypatch,
) -> None:
    from capx.integrations.franka import tip_xy_only
    from capx.verification import libero_g17_release_ready as release_ready

    state = {"done": False}
    events: list[dict] = []

    def read_live(_api):
        residual = 0.05
        snapshot = SimpleNamespace(
            private_poses={
                "obj01": (np.array([residual, 0.0, 0.0]), None),
                "obj02": (np.array([0.0, 0.0, 0.0]), None),
            }
        )
        gate = SimpleNamespace(
            relation="on",
            acquired=True,
            release_ok_tipxy=state["done"],
            needs_align_tipxy=not state["done"],
            residual_xy=(residual, 0.0),
        )
        return object(), snapshot, gate

    def move(_api, dx, dy, **_kwargs):
        state["done"] = True
        return SimpleNamespace(
            ok=True,
            acceptance_ok=True,
            d_tip_z=0.02,
            achieved_xy=(dx, dy),
            final_xy_err=0.0,
        )

    monkeypatch.setattr(release_ready, "_read_live", read_live)
    monkeypatch.setattr(tip_xy_only, "move_tip_xy", move)
    monkeypatch.setattr(
        osc, "_v269_audit", lambda _api, event, **data: events.append(
            {"event": event, **data}
        )
    )
    monkeypatch.setenv("CAPX_LIBERO_G17_V269_SYMBOL_SOURCE", "fg")
    monkeypatch.setenv("CAPX_LIBERO_G17_V276_COMPLETION_IK", "1")

    assert osc._run_v274_mech_contract(
        SimpleNamespace(_capx_release_ready_busy=False)
    )
    assert any(row["event"] == "z_diagnostic_continue" for row in events)
    assert not any(row["event"] == "z_abort" for row in events)


def test_schema_v269_shared_mech_has_no_later_guards() -> None:
    keys = (
        "CAPX_LIBERO_G17_XY_LOCK",
        "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
        "CAPX_LIBERO_G17_RELEASE_READY",
        "CAPX_LIBERO_G17_NEARMISS_OSC",
        "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
        "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH",
        "CAPX_LIBERO_G17_V269_MECH",
        "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE",
        "CAPX_LIBERO_G17_RELEASE_READY_NO_SILENT_ALLOW",
        "CAPX_LIBERO_G17_RELEASE_READY_NEARMISS_GATE",
        "CAPX_LIBERO_G17_RELEASE_READY_HOLD_GUARD",
        "CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE",
        "CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT",
        "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys[:6]:
            os.environ[key] = "1"
        os.environ["CAPX_LIBERO_G17_V269_MECH"] = "1"
        os.environ["CAPX_LIBERO_G17_V269_SYMBOL_SOURCE"] = "fg"
        for key in keys[8:]:
            os.environ.pop(key, None)
        schema = takeover.schema_version()
        assert "v2.69" in schema
        assert "v256-shared-mech-fg-source" in schema
        assert "hold-guard" not in schema
        assert "nearmiss-gate" not in schema
        assert "inscribed" not in schema
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_schema_v258_fg_predicate_wash() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_NEARMISS_OSC",
            "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH",
            "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
            "CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE",
            "CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH"] = "1"
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
        "CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL",
        "CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE",
        "CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT",
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.58" in s and "fg-predicate-wash" in s
        assert "inscribed" not in s
        assert osc.nearmiss_osc_fg_predicate_wash_enabled()
        assert osc.nearmiss_osc_bandwindow_enabled()
        assert not osc.nearmiss_osc_xy_decouple_enabled()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_inscribed_square_scalars_and_deadlock_gone() -> None:
    """圆中方: skip band = open/√2; corner of old square now still moves."""
    from capx.verification.libero_g17_band_obligations import residual_to_band

    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE",
            "CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT",
        )
    }
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT", None)
    try:
        start, step = osc.schedule_scalars_m()
        assert abs(start - 0.15 / math.sqrt(2)) < 1e-12
        assert abs(step - 0.03 / math.sqrt(2)) < 1e-12
        band = osc.axis_skip_band_m()
        assert abs(band - step) < 1e-12
        # Old deadlock corner (0.025, 0.025): L2>0.03 but was L∞ WITHIN.
        # With inscribed band ≈0.0212, residual 0.025 is POS → still moves.
        b = residual_to_band(axis="x", residual=0.025, band_m=band)
        d = osc.axis_delta_from_fg_band(b, start)
        assert abs(d + start) < 1e-12
        # Inside inscribed square ⇒ WITHIN ⇒ skip, and L2 < open disk.
        b2 = residual_to_band(axis="x", residual=0.015, band_m=band)
        assert osc.axis_delta_from_fg_band(b2, start) == 0.0
        assert math.hypot(0.015, 0.015) < 0.03
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_schema_v259_inscribed_no_z() -> None:
    """Historical v261 schema name retired: inscribed+no_z now v2.60 with guards."""
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_NEARMISS_OSC",
            "CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH",
            "CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE",
            "CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT",
            "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT"] = "1"
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.60" in s and "inscribed-no-z-guard" in s
        assert osc.nearmiss_osc_inscribed_square_enabled()
        assert osc.nearmiss_osc_no_z_abort_enabled()
        # Budget constant is optional in current tipxy loop; schema gate is enough.
        assert "v2.60" in s
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
