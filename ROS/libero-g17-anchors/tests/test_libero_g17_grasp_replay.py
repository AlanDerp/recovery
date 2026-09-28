"""Unit tests for v256 GraspNet freeze/replay (no sim)."""

from __future__ import annotations

import os
from pathlib import Path

from capx.verification import libero_g17_grasp_replay as gr
from capx.verification import libero_g17_takeover as takeover


def _pkg_path() -> Path:
    return Path(
        "/project/peilab/qshou/cap-x-agilex/outputs/"
        "libero_g17_grasp_replay_v256_20260817/grasp_replay_v1.json"
    )


def test_grasp_replay_package_loads_and_sequences() -> None:
    pkg = _pkg_path()
    assert pkg.is_file()
    prev = {
        k: os.environ.get(k)
        for k in (
            gr.ENV_PATH,
            gr.ENV_STRICT,
            gr.ENV_OVERFLOW,
            "REPLICATE",
            "CAPX_ACTIVE_TRIAL",
        )
    }
    os.environ[gr.ENV_PATH] = str(pkg)
    os.environ[gr.ENV_STRICT] = "1"
    os.environ[gr.ENV_OVERFLOW] = "miss"
    os.environ["REPLICATE"] = "0"
    os.environ["CAPX_ACTIVE_TRIAL"] = "2"
    gr._PACKAGE = None
    gr.reset_cursor()
    try:
        assert gr.enabled()
        p0 = gr.next_pose(object_name="akita black bowl")
        assert p0 is not None and p0.call_idx == 0
        assert p0.position.shape == (3,)
        p1 = gr.next_pose(object_name="akita black bowl")
        assert p1 is not None and p1.call_idx == 1
        try:
            gr.next_pose()
            raise AssertionError("expected GRASP_REPLAY_MISS")
        except RuntimeError as exc:
            assert "GRASP_REPLAY_MISS" in str(exc)
    finally:
        gr._PACKAGE = None
        gr.reset_cursor()
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_grasp_replay_overflow_repeat_last(capsys) -> None:
    pkg = _pkg_path()
    prev = {
        k: os.environ.get(k)
        for k in (
            gr.ENV_PATH,
            gr.ENV_STRICT,
            gr.ENV_OVERFLOW,
            "REPLICATE",
            "CAPX_ACTIVE_TRIAL",
        )
    }
    os.environ[gr.ENV_PATH] = str(pkg)
    os.environ[gr.ENV_STRICT] = "1"
    os.environ[gr.ENV_OVERFLOW] = "repeat_last"
    os.environ["REPLICATE"] = "0"
    os.environ["CAPX_ACTIVE_TRIAL"] = "2"
    gr._PACKAGE = None
    gr.reset_cursor()
    try:
        p0 = gr.next_pose()
        p1 = gr.next_pose()
        assert p0 is not None and p1 is not None
        p2 = gr.next_pose()
        assert p2 is not None and p2.call_idx == 2
        assert (p2.position == p1.position).all()
        p3 = gr.next_pose()
        assert p3 is not None and (p3.position == p1.position).all()
        assert "[grasp-replay]" not in capsys.readouterr().out
    finally:
        gr._PACKAGE = None
        gr.reset_cursor()
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_schema_v262_v260_plus_grasp_replay() -> None:
    pkg = _pkg_path()
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
            "CAPX_LIBERO_G17_GRASP_REPLAY",
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
    os.environ["CAPX_LIBERO_G17_GRASP_REPLAY"] = str(pkg)
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE",
        "CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT",
        "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.62" in s and "fg-predicate-wash" in s and "grasp-replay" in s
        assert "inscribed" not in s
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_schema_v261_grasp_replay() -> None:
    pkg = _pkg_path()
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
            "CAPX_LIBERO_G17_GRASP_REPLAY",
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
    os.environ["CAPX_LIBERO_G17_GRASP_REPLAY"] = str(pkg)
    for k in (
        "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.61" in s and "grasp-replay" in s
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
