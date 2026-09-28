"""Tests for near-miss fail-open tipxy (no sim)."""

from __future__ import annotations

import os
from types import SimpleNamespace

from capx.verification import libero_g17_nearmiss_failopen as nm
from capx.verification import libero_g17_takeover as takeover


def test_is_nearmiss_band() -> None:
    g = SimpleNamespace(
        relation="on",
        acquired=True,
        contact=True,
        xy_within=False,
        xy_distance_m=0.05,
        residual_xy=(0.04, 0.03),
    )
    assert nm.is_nearmiss(g, r_m=0.12, band_m=0.03)
    g.xy_distance_m = 0.02
    assert not nm.is_nearmiss(g, r_m=0.12, band_m=0.03)
    g.xy_distance_m = 0.05
    g.contact = False
    assert not nm.is_nearmiss(g, r_m=0.12, band_m=0.03)
    g.contact = True
    g.xy_distance_m = 0.20
    assert not nm.is_nearmiss(g, r_m=0.12, band_m=0.03)


def test_schema_v250_failopen() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_NEARMISS_FAILOPEN",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
            "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_FAILOPEN"] = "1"
    for k in (
        "CAPX_LIBERO_G17_RELEASE_READY",
        "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
        "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.50" in s and "nearmiss-failopen-tipxy" in s
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_run_always_true_without_env(monkeypatch) -> None:
    """Fail-open: snapshot failure still allows open."""

    def boom(api):
        raise RuntimeError("no env")

    monkeypatch.setattr(nm, "_read_gate", boom)
    assert nm.run_nearmiss_tipxy_failopen(SimpleNamespace()) is True
