"""Tests for post-miss clear-cache gate."""

from __future__ import annotations

import os
from types import SimpleNamespace

from capx.verification import libero_g17_post_miss_clear as pmc
from capx.verification import libero_g17_takeover as takeover


def test_should_clear_on_placement_xy_near_miss_empty() -> None:
    prev = os.environ.get("CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE")
    os.environ["CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE"] = "1"
    try:
        proj = SimpleNamespace(
            task_gap="placement_xy",
            executor_attachment="empty",
            released_goal_after_transport=True,
            gain=SimpleNamespace(bound_release=True),
            combined_report=(
                "placement near-miss: center_on_surface is on while "
                "center_to_center_xy is outside"
            ),
        )
        assert pmc.should_clear_after_projection(proj) is True
        proj.task_gap = "acquisition"
        assert pmc.should_clear_after_projection(proj) is False
    finally:
        if prev is None:
            os.environ.pop("CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE", None)
        else:
            os.environ["CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE"] = prev


def test_schema_post_miss_clear_on_v238_stack() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
            "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_BAND_OBLIGATIONS", None)
    os.environ.pop("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", None)
    try:
        schema = takeover.schema_version()
        assert "post-miss-clear" in schema
        assert "v2.41" in schema
        assert pmc.goal_lock_between_to_ordinal_enabled() is True
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
