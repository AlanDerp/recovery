"""Tests for ReleaseReady full-conjunct gate + control law."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from capx.verification import libero_g17_release_ready as rr
from capx.verification import libero_g17_takeover as takeover


def test_next_xy_step_moves_toward_zero() -> None:
    residual = np.array([0.05, 0.0], dtype=np.float64)
    delta, step, dist = rr.next_xy_step(residual, prev_distance_m=None, step_m=0.008)
    assert dist == 0.05
    assert delta[0] < 0.0
    assert abs(delta[1]) < 1e-12


def test_next_xy_step_shrinks_when_farther() -> None:
    residual = np.array([0.04, 0.0], dtype=np.float64)
    _d1, step1, _ = rr.next_xy_step(residual, prev_distance_m=0.03, step_m=0.008)
    assert step1 < 0.008


def test_release_ok_requires_all_conjuncts() -> None:
    poses = {
        "obj01": (np.array([0.01, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0])),
        "obj02": (np.array([0.0, 0.0, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])),
        "exe01": (np.array([0.01, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    base_values = {
        "executor_object.exe01.obj01.holding": "TRUE",
        "executor.exe01.payload": "OBJ01",
        "executor_object.exe01.obj01.grasp_evidence": "STABLE_GRASP",
        "executor_object.exe01.obj01.contact": "TRUE",
        "object_object.obj01.obj02.contact": "TRUE",
        "object_object.obj01.obj02.center_in_reference_footprint": "INSIDE",
    }
    binding = SimpleNamespace(relation="on")
    gate = rr.evaluate_gate(
        binding, SimpleNamespace(values=base_values, private_poses=poses)
    )
    assert gate.release_ok is True
    assert gate.needs_align is False

    # XY outside → not release_ok even with contact+surface
    poses_out = {
        "obj01": (np.array([0.05, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0])),
        "obj02": poses["obj02"],
        "exe01": poses["exe01"],
    }
    gate_out = rr.evaluate_gate(
        binding, SimpleNamespace(values=base_values, private_poses=poses_out)
    )
    assert gate_out.xy_within is False
    assert gate_out.release_ok is False
    assert gate_out.needs_align is True

    # Within XY but no plate contact → still not release_ok
    values_nc = dict(base_values)
    values_nc["object_object.obj01.obj02.contact"] = "FALSE"
    gate_nc = rr.evaluate_gate(
        binding, SimpleNamespace(values=values_nc, private_poses=poses)
    )
    assert gate_nc.xy_within is True
    assert gate_nc.contact is False
    assert gate_nc.release_ok is False


def test_schema_release_ready_hand_and_goto() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
            "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_RELEASE_READY_GOTO",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_GOTO_TIP_IDENTITY",
            "CAPX_LIBERO_G17_NEARMISS_OSC",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_BAND_OBLIGATIONS", None)
    os.environ.pop("CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE", None)
    os.environ.pop("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", None)
    os.environ.pop("CAPX_LIBERO_G17_GOTO_TIP_IDENTITY", None)
    os.environ.pop("CAPX_LIBERO_G17_NEARMISS_OSC", None)
    os.environ.pop("CAPX_LIBERO_G17_RELEASE_READY_TIPXY", None)
    try:
        os.environ.pop("CAPX_LIBERO_G17_RELEASE_READY_GOTO", None)
        schema = takeover.schema_version()
        assert "release-ready-xyhand" in schema and "v2.45" in schema
        os.environ["CAPX_LIBERO_G17_RELEASE_READY_GOTO"] = "1"
        schema_g = takeover.schema_version()
        assert "release-ready-goto" in schema_g and "v2.46" in schema_g
        os.environ.pop("CAPX_LIBERO_G17_RELEASE_READY_GOTO", None)
        os.environ["CAPX_LIBERO_G17_RELEASE_READY_TIPXY"] = "1"
        schema_t = takeover.schema_version()
        assert "release-ready-tipxy" in schema_t and "v2.49" in schema_t
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_release_ok_tipxy_drops_surface() -> None:
    poses = {
        "obj01": (np.array([0.01, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0])),
        "obj02": (np.array([0.0, 0.0, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])),
        "exe01": (np.array([0.01, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    values = {
        "executor_object.exe01.obj01.holding": "TRUE",
        "executor.exe01.payload": "OBJ01",
        "executor_object.exe01.obj01.grasp_evidence": "STABLE_GRASP",
        "executor_object.exe01.obj01.contact": "TRUE",
        "object_object.obj01.obj02.contact": "TRUE",
        "object_object.obj01.obj02.center_in_reference_footprint": "OUTSIDE",
    }
    gate = rr.evaluate_gate(
        SimpleNamespace(relation="on"),
        SimpleNamespace(values=values, private_poses=poses),
    )
    assert gate.surface_on is False
    assert gate.release_ok is False
    assert gate.release_ok_tipxy is True
    assert gate.needs_align_tipxy is False


def test_next_xy_step_band_is_3cm() -> None:
    residual = np.array([0.05, 0.0], dtype=np.float64)
    delta, step, dist = rr.next_xy_step_band(residual)
    assert abs(step - 0.03) < 1e-12
    assert abs(dist - 0.05) < 1e-12
    assert abs(delta[0] + 0.03) < 1e-12  # toward zero, capped at band
    # residual smaller than band step → full residual
    residual2 = np.array([0.02, 0.0], dtype=np.float64)
    d2, s2, dist2 = rr.next_xy_step_band(residual2)
    assert dist2 < 0.03
    assert abs(np.linalg.norm(d2)) < 1e-12  # already within band → zero


def test_needs_align_tipxy_without_plate_contact() -> None:
    """Window: holding + out-of-band must slide even if plate contact is false."""
    poses = {
        "obj01": (np.array([0.05, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0])),
        "obj02": (np.array([0.0, 0.0, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])),
        "exe01": (np.array([0.05, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    values = {
        "executor_object.exe01.obj01.holding": "TRUE",
        "executor.exe01.payload": "OBJ01",
        "executor_object.exe01.obj01.grasp_evidence": "STABLE_GRASP",
        "executor_object.exe01.obj01.contact": "TRUE",
        "object_object.obj01.obj02.contact": "FALSE",
        "object_object.obj01.obj02.center_in_reference_footprint": "OUTSIDE",
    }
    gate = rr.evaluate_gate(
        SimpleNamespace(relation="on"),
        SimpleNamespace(values=values, private_poses=poses),
    )
    assert gate.contact is False
    assert gate.xy_within is False
    assert gate.acquired is True
    assert gate.needs_align_tipxy is True
    assert gate.release_ok_tipxy is False


def test_schema_v252_tipxy_bandstep() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL",
            "CAPX_LIBERO_G17_FAIR_PLACE_SERVO",
            "CAPX_LIBERO_G17_NEARMISS_FAILOPEN",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY_TIPXY"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP"] = "1"
    for k in (
        "CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL",
        "CAPX_LIBERO_G17_FAIR_PLACE_SERVO",
        "CAPX_LIBERO_G17_NEARMISS_FAILOPEN",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.52" in s and "tipxy-bandstep" in s
        assert "fair-residual" not in s
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_schema_v253_tipxy_bandstep_fair_residual() -> None:
    prev = {
        k: os.environ.get(k)
        for k in (
            "CAPX_LIBERO_G17_XY_LOCK",
            "CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL",
            "CAPX_LIBERO_G17_RELEASE_READY",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP",
            "CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL",
            "CAPX_LIBERO_G17_FAIR_PLACE_SERVO",
            "CAPX_LIBERO_G17_NEARMISS_FAILOPEN",
            "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
        )
    }
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY_TIPXY"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL"] = "1"
    for k in (
        "CAPX_LIBERO_G17_FAIR_PLACE_SERVO",
        "CAPX_LIBERO_G17_NEARMISS_FAILOPEN",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)
    try:
        s = takeover.schema_version()
        assert "v2.53" in s and "tipxy-bandstep-fair-residual" in s
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_overlay_gate_student_residual_sign_and_within() -> None:
    """Student residual uses GT sign (payload−plate); within from student dist."""
    poses = {
        "obj01": (np.array([0.05, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0])),
        "obj02": (np.array([0.0, 0.0, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])),
        "exe01": (np.array([0.05, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    values = {
        "executor_object.exe01.obj01.holding": "TRUE",
        "executor.exe01.payload": "OBJ01",
        "executor_object.exe01.obj01.grasp_evidence": "STABLE_GRASP",
        "executor_object.exe01.obj01.contact": "TRUE",
        "object_object.obj01.obj02.contact": "FALSE",
        "object_object.obj01.obj02.center_in_reference_footprint": "OUTSIDE",
    }
    gate = rr.evaluate_gate(
        SimpleNamespace(relation="on"),
        SimpleNamespace(values=values, private_poses=poses),
    )
    assert gate.residual_xy is not None
    assert abs(gate.residual_xy[0] - 0.05) < 1e-9

    class _Api:
        def get_object_pose(self, name: str):
            # plate at origin; bowl at (+0.04, 0) — student near-miss
            if name == "plate":
                return np.array([0.0, 0.0, 0.05]), np.array([1.0, 0, 0, 0])
            if name == "akita_black_bowl":
                return np.array([0.04, 0.0, 0.1]), np.array([1.0, 0, 0, 0])
            return None, None

        def get_observation(self):
            return {
                "robot_cartesian_pos": np.array(
                    [0.04, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0]
                )
            }

    overlaid, src = rr.overlay_gate_student_residual(gate, _Api())
    assert overlaid is not None
    assert "get_object_pose" in src or src == "get_object_pose"
    assert overlaid.acquired is True  # GT acquired preserved
    assert overlaid.contact is False
    assert abs(overlaid.residual_xy[0] - 0.04) < 1e-9  # payload − plate
    assert abs(overlaid.xy_distance_m - 0.04) < 1e-9
    assert overlaid.xy_within is False
    assert overlaid.needs_align_tipxy is True


def test_schema_release_ready_full() -> None:
    # Alias kept for older callers.
    test_schema_release_ready_hand_and_goto()
