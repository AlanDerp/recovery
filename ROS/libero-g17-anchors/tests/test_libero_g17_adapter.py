"""Unit tests for the LIBERO G17-goal adapter.

These tests stay off GPU / MuJoCo when possible: binding and rejection are pure
BDDL-structure checks. Extraction uses a lightweight fake domain that mirrors
the closed_gt cube adapter surface.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from capx.verification import closed_gt, libero_g17_adapter as adapter


def _spatial_problem() -> dict:
    return {
        "fixtures": {"table": ["main_table"], "wooden_cabinet": ["wooden_cabinet_1"]},
        "objects": {
            "akita_black_bowl": ["akita_black_bowl_1", "akita_black_bowl_2"],
            "plate": ["plate_1"],
        },
        "obj_of_interest": ["akita_black_bowl_1", "plate_1"],
        "goal_state": [["On", "akita_black_bowl_1", "plate_1"]],
    }


def _object_problem() -> dict:
    return {
        "fixtures": {"table": ["main_table"]},
        "objects": {
            "cream_cheese": ["cream_cheese_1"],
            "basket": ["basket_1"],
        },
        "obj_of_interest": ["cream_cheese_1", "basket_1"],
        "goal_state": [["In", "cream_cheese_1", "basket_1"]],
    }


def _stove_problem() -> dict:
    return {
        "fixtures": {"table": ["main_table"], "flat_stove": ["flat_stove_1"]},
        "objects": {},
        "obj_of_interest": ["flat_stove_1"],
        "goal_state": [["Turnon", "flat_stove_1"]],
    }


def test_bind_on_keeps_two_object_g17_schema() -> None:
    binding = adapter.bind_libero_g17_goal(
        _spatial_problem(), suite_name="libero_spatial", task_id=0
    )
    assert binding.relation == "on"
    assert binding.payload_name == "akita_black_bowl_1"
    assert binding.receptacle_eval_name == "plate_1"
    assert binding.receptacle_body_name == "plate_1"
    assert binding.table_name == "main_table"
    assert binding.scene.objects[0].entity_id == "obj01"
    assert binding.scene.objects[1].entity_id == "obj02"
    audit = closed_gt.schema_audit(binding.scene)
    assert audit["state_field_count"] == 51
    assert audit["derived_field_count"] == 17
    assert audit["unknown_is_not_a_gt_value"] is True


def test_bind_in_resolves_site_receptacle_to_parent_body() -> None:
    states = {
        "cream_cheese_1": SimpleNamespace(object_state_type="object"),
        "basket_1_contain_region": SimpleNamespace(
            object_state_type="site", parent_name="basket_1"
        ),
    }
    problem = _object_problem()
    problem["goal_state"] = [["In", "cream_cheese_1", "basket_1_contain_region"]]
    binding = adapter.bind_libero_g17_goal(problem, object_states=states)
    assert binding.relation == "in"
    assert binding.receptacle_eval_name == "basket_1_contain_region"
    assert binding.receptacle_body_name == "basket_1"


def test_bind_rejects_turnon_and_multi_atom_goals() -> None:
    with pytest.raises(adapter.UnsupportedLiberoG17Goal, match="excludes turnon"):
        adapter.bind_libero_g17_goal(_stove_problem())
    with pytest.raises(adapter.UnsupportedLiberoG17Goal, match="exactly one"):
        adapter.bind_libero_g17_goal(
            {
                "fixtures": {"table": ["main_table"]},
                "objects": {},
                "goal_state": [
                    ["On", "bowl_1", "plate_1"],
                    ["On", "bowl_2", "plate_1"],
                ],
            }
        )


def test_and_wrapped_single_atom_is_accepted() -> None:
    problem = _spatial_problem()
    problem["goal_state"] = ["And", ["On", "akita_black_bowl_1", "plate_1"]]
    binding = adapter.bind_libero_g17_goal(problem)
    assert binding.relation == "on"


class _FakeModel:
    def __init__(self, size=(0.03, 0.03, 0.02)):
        self.size = np.asarray(size, dtype=np.float64)


class _FakeGripper:
    pass


class _FakeRobot:
    def __init__(self):
        self.gripper = _FakeGripper()
        self._ref_gripper_joint_pos_indexes = {"right": (7, 8)}


class _FakeSimData:
    def __init__(self):
        self.body_xpos = np.zeros((10, 3), dtype=np.float64)
        self.body_xquat = np.zeros((10, 4), dtype=np.float64)
        self.body_xquat[:, 0] = 1.0
        self.body_xvelp = np.zeros((10, 3), dtype=np.float64)
        self.body_xvelr = np.zeros((10, 3), dtype=np.float64)
        self.qpos = np.zeros(20, dtype=np.float64)
        # open gripper: 0.04 / 0.04 = 1.0
        self.qpos[7] = 0.04
        self.qpos[8] = -0.04
        # base / gripper / objects
        self.body_xpos[1] = (0.0, 0.0, 0.0)  # base
        self.body_xpos[2] = (0.0, 0.0, 0.30)  # gripper
        self.body_xpos[3] = (0.0, 0.0, 0.061)  # payload
        self.body_xpos[4] = (0.0, 0.0, 0.020)  # receptacle
        self.body_xpos[5] = (0.0, 0.0, 0.0)  # table


class _FakeSimModel:
    def body_name2id(self, name: str) -> int:
        return {
            "robot0_base": 1,
            "gripper0_eef": 2,
        }[name]


class _FakeSim:
    def __init__(self):
        self.data = _FakeSimData()
        self.model = _FakeSimModel()


class _FakeDomain:
    def __init__(self, *, on_true: bool = True, in_true: bool = False):
        self.parsed_problem = _spatial_problem() if on_true or not in_true else _object_problem()
        if in_true:
            self.parsed_problem = _object_problem()
        self.sim = _FakeSim()
        self.robots = [_FakeRobot()]
        self.obj_body_id = {
            "akita_black_bowl_1": 3,
            "plate_1": 4,
            "cream_cheese_1": 3,
            "basket_1": 4,
            "main_table": 5,
        }
        self.table_offset = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self._models = {
            name: _FakeModel() for name in self.obj_body_id
        }
        self._holding = set()
        self._contacts = {("akita_black_bowl_1", "plate_1"), ("cream_cheese_1", "basket_1")}
        self._table_contacts = {"plate_1", "basket_1"}
        self._on_true = on_true
        self._in_true = in_true
        self.object_states_dict = {
            "akita_black_bowl_1": SimpleNamespace(
                object_state_type="object",
                check_ontop=lambda other: False,
                check_contain=lambda other: False,
                check_contact=lambda other: False,
            ),
            "plate_1": SimpleNamespace(
                object_state_type="object",
                check_ontop=lambda other: self._on_true,
                check_contain=lambda other: False,
                check_contact=lambda other: True,
            ),
            "cream_cheese_1": SimpleNamespace(
                object_state_type="object",
                check_ontop=lambda other: False,
                check_contain=lambda other: False,
                check_contact=lambda other: False,
            ),
            "basket_1": SimpleNamespace(
                object_state_type="object",
                check_ontop=lambda other: False,
                check_contain=lambda other: self._in_true,
                check_contact=lambda other: True,
            ),
        }

    def get_object(self, name: str):
        return self._models[name]

    def _check_grasp(self, *, gripper, object_geoms):
        for name, model in self._models.items():
            if model is object_geoms and name in self._holding:
                return True
        return False

    def check_contact(self, first, second):
        if second == "table_collision":
            return False
        name_a = self._name_of(first)
        name_b = self._name_of(second)
        if name_b == "main_table":
            return name_a in self._table_contacts
        if name_a == "main_table":
            return name_b in self._table_contacts
        return (name_a, name_b) in self._contacts or (name_b, name_a) in self._contacts

    def _check_success(self):
        if self.parsed_problem["goal_state"][0][0].lower() == "in":
            return self._in_true
        return self._on_true

    def _name_of(self, obj):
        if isinstance(obj, str):
            return obj
        for name, model in self._models.items():
            if model is obj:
                return name
        if obj is self.robots[0].gripper:
            return "gripper"
        raise KeyError(obj)


def test_extract_compiles_g17_snapshot_and_bddl_audit() -> None:
    domain = _FakeDomain(on_true=True)
    binding = adapter.bind_libero_g17_env(
        domain, suite_name="libero_spatial", task_id=0
    )
    raw = adapter.extract_libero_raw_scene(domain, binding)
    snapshot = closed_gt.compile_snapshot(binding.scene, raw)
    assert snapshot.values["object_object.obj01.obj02.contact"] == "TRUE"
    assert snapshot.values["executor.exe01.aperture"] == "OPEN"
    assert snapshot.values["object.obj02.table_contact"] == "TRUE"
    audit = adapter.g17_versus_bddl_audit(snapshot, domain, binding)
    assert audit["bddl_atom_true"] is True
    assert audit["private_task_completed"] is True
    assert audit["bddl_iff_task_completed"] is True
    assert audit["schema_expanded"] is False
    assert audit["action_visible"] is False


def test_in_binding_reports_volume_proxy_separately_from_bddl() -> None:
    domain = _FakeDomain(on_true=False, in_true=True)
    binding = adapter.bind_libero_g17_env(
        domain, suite_name="libero_object", task_id=1
    )
    assert binding.relation == "in"
    raw = adapter.extract_libero_raw_scene(domain, binding)
    snapshot = closed_gt.compile_snapshot(binding.scene, raw)
    audit = adapter.g17_versus_bddl_audit(snapshot, domain, binding)
    assert audit["bddl_atom_true"] is True
    assert audit["relation"] == "in"
    # Geometric In proxy uses volume+contact; fake bodies are stacked, so the
    # proxy may disagree with BDDL. That disagreement is the IFF hole signal.
    assert "g17_iff_bddl" in audit
    assert audit["bddl_iff_task_completed"] is True
