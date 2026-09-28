"""Tests for fair place servo (no sim, no GT)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from capx.verification import libero_g17_fair_place_servo as fps
from capx.verification import libero_g17_takeover as takeover


def test_module_has_no_closed_gt() -> None:
    src = open(fps.__file__, encoding="utf-8").read()
    assert "import closed_gt" not in src
    assert "from capx.verification import closed_gt" not in src
    assert "libero_g17_adapter" not in src
    assert "libero_g17_release_ready" not in src
    assert "get_object_pose" in src
    assert "move_tip_xy" in src


def test_should_servo_tight_band() -> None:
    assert fps.should_servo(0.04, band_m=0.03, max_near_m=0.05)
    assert not fps.should_servo(0.02, band_m=0.03, max_near_m=0.05)
    assert not fps.should_servo(0.08, band_m=0.03, max_near_m=0.05)


def test_schema_v251_fair_place() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_FAIR_PLACE_SERVO",
            "CAPX_LIBERO_G17_NEARMISS_FAILOPEN",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
            "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_FAIR_PLACE_SERVO"] = "1"
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_FAILOPEN",
        "CAPX_LIBERO_G17_RELEASE_READY",
        "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
        "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.51" in s and "fair-place-servo-tipxy" in s
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_run_failopen_on_pose_error(monkeypatch) -> None:
    class BoomApi:
        def get_observation(self):
            return {"robot_cartesian_pos": np.array([0, 0, 0, 1, 0, 0, 0, 0.0])}

        def get_object_pose(self, name, use_multiview=True):
            raise RuntimeError("vision down")

    monkeypatch.setattr(
        fps,
        "_read_tip_and_gripper",
        lambda api: (np.zeros(3), 0.0),
    )
    assert fps.run_fair_place_servo(BoomApi()) is True


def test_skip_when_too_far(monkeypatch) -> None:
    class Api:
        pass

    monkeypatch.setattr(
        fps,
        "_read_tip_and_gripper",
        lambda api: (np.zeros(3), 0.0),
    )
    monkeypatch.setattr(
        fps,
        "fair_xy_residual",
        lambda api, **kw: (np.array([0.08, 0.0]), "get_object_pose"),
    )
    assert fps.run_fair_place_servo(Api()) is True
