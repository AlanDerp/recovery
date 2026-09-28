"""Unit tests for generic band obligation mini predicate contracts."""

from __future__ import annotations

import os

from capx.verification import closed_gt
from capx.verification import libero_g17_adapter as adapter
from capx.verification import libero_g17_band_obligations as obl
from capx.verification import libero_g17_fg_projection as projection
from capx.verification import libero_g17_takeover as takeover


def _problem() -> dict:
    return {
        "fixtures": {"table": ["main_table"]},
        "objects": {
            "akita_black_bowl": ["akita_black_bowl_1", "akita_black_bowl_2"],
            "plate": ["plate_1"],
        },
        "goal_state": [["On", "akita_black_bowl_1", "plate_1"]],
    }


def _raw(
    *,
    payload_xy: tuple[float, float],
    contact: bool,
    holding_payload: bool = False,
) -> closed_gt.RawSceneState:
    import numpy as np

    object_contacts = (
        frozenset({frozenset({"obj01", "obj02"})}) if contact else frozenset()
    )
    holding = frozenset({("exe01", "obj01")}) if holding_payload else frozenset()
    executor_contacts = (
        frozenset({("exe01", "obj01")}) if holding_payload else frozenset()
    )
    return closed_gt.RawSceneState(
        executors={
            "exe01": closed_gt.BodyState(
                position=np.array([0.0, 0.0, 0.30]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.02, 0.02, 0.02]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        },
        objects={
            "obj01": closed_gt.BodyState(
                position=np.array([payload_xy[0], payload_xy[1], 0.04]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.04, 0.04, 0.02]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            ),
            "obj02": closed_gt.BodyState(
                position=np.array([0.0, 0.0, 0.01]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.08, 0.08, 0.01]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            ),
        },
        executor_aperture={"exe01": 0.0 if holding_payload else 1.0},
        holding=holding,
        executor_contacts=executor_contacts,
        object_contacts=object_contacts,
        table_contacts=frozenset({"obj01", "obj02"}),
        table_height=0.0,
    )


def test_sign_to_band_and_invert_algebra() -> None:
    assert obl.sign_to_band(axis="x", sign="closer") == obl._NEG
    assert obl.invert_band(obl._NEG) == obl._POS
    assert "farther" in obl.cnl_for_band("x", obl._POS, ref="plate center")
    assert "same depth" in obl.cnl_for_band("x", obl._WITHIN, ref="plate center")


def test_contract_lists_both_axes_and_satisfied_sibling() -> None:
    # x outside band, y within band → one open, one satisfied.
    obls = obl.open_c2c_band_contract(
        payload_label="akita black bowl",
        receptacle_label="plate",
        dx=0.05,
        dy=0.01,
        center_within=False,
        band_m=0.03,
    )
    assert len(obls) == 2
    by_axis = {o.axis: o for o in obls}
    assert by_axis["x"].satisfied is False
    assert by_axis["y"].satisfied is True
    lines = obl.render_band_obligation_lines(
        obls, payload_label="akita black bowl", holding_payload=True
    )
    text = "\n".join(lines).lower()
    assert "instantiated predicate contract:" in text
    assert "current" in text and "required" in text
    assert "not satisfied." in text
    assert "satisfied." in text
    assert "terminal factual conditions:" in text
    assert "gripper holding akita black bowl" in text
    assert "target true" in text
    assert "→" not in "\n".join(lines)
    assert "before other placement" not in text
    assert "sample_grasp_pose" not in text
    assert "which=" not in text


def test_primary_obligation_prefers_lateral_open() -> None:
    obls = obl.open_c2c_band_obligations(
        payload_label="akita black bowl",
        receptacle_label="plate",
        depth_sign="closer",
        lateral_sign="right",
        center_within=False,
    )
    assert len(obls) == 2
    primary = obl.pick_primary_obligation(obls)
    assert primary is not None
    assert primary.axis == "y"


def test_no_obligation_when_center_within() -> None:
    obls = obl.open_c2c_band_contract(
        payload_label="akita black bowl",
        receptacle_label="plate",
        dx=0.05,
        dy=0.04,
        center_within=True,
    )
    assert obls == []


def test_projection_emits_contract_when_env_on() -> None:
    prev = os.environ.get("CAPX_LIBERO_G17_BAND_OBLIGATIONS")
    os.environ["CAPX_LIBERO_G17_BAND_OBLIGATIONS"] = "1"
    try:
        binding = adapter.bind_libero_g17_goal(
            _problem(), suite_name="libero_spatial", task_id=0
        )
        after = closed_gt.compile_snapshot(
            binding.scene,
            _raw(payload_xy=(0.05, 0.04), contact=True, holding_payload=True),
        )
        peak = adapter.LiberoAttachmentFact("holding_payload", ("akita_black_bowl_1",))
        holding = adapter.LiberoAttachmentFact(
            "holding_payload", ("akita_black_bowl_1",)
        )
        proj = projection.project_libero_fg(
            binding,
            after,
            after,
            attachment=holding,
            attachment_before=peak,
            attachment_peak=peak,
        )
        text = proj.combined_report.lower()
        assert proj.task_gap == "placement_xy"
        assert "instantiated predicate contract:" in text
        assert "current" in text and "required" in text and "not satisfied" in text
        assert "terminal factual conditions:" in text
        assert "→" not in proj.combined_report
        assert "before other placement" not in text
        assert "sample_grasp_pose" not in text
    finally:
        if prev is None:
            os.environ.pop("CAPX_LIBERO_G17_BAND_OBLIGATIONS", None)
        else:
            os.environ["CAPX_LIBERO_G17_BAND_OBLIGATIONS"] = prev


def test_schema_band_contract_on_v238_stack() -> None:
    prev_xy = os.environ.get("CAPX_LIBERO_G17_XY_LOCK")
    prev_ord = os.environ.get("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL")
    prev_obl = os.environ.get("CAPX_LIBERO_G17_BAND_OBLIGATIONS")
    prev_goal = os.environ.get("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER")
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_BAND_OBLIGATIONS"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", None)
    try:
        schema = takeover.schema_version()
        assert "band-contract" in schema
        assert "v2.39" in schema
    finally:
        for key, prev in (
            ("CAPX_LIBERO_G17_XY_LOCK", prev_xy),
            ("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", prev_ord),
            ("CAPX_LIBERO_G17_BAND_OBLIGATIONS", prev_obl),
            ("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", prev_goal),
        ):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
