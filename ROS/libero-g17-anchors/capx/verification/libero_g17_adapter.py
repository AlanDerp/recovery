"""LIBERO adapter for the G17 closed-GT algebra.

G17 itself is unchanged: one executor, two objects, one table surface, the
existing 51-field schema, and no Open/TurnOn/push primitives. This module only
binds a LIBERO pick-place goal onto that algebra.

Entity map:
    exe01  robot gripper
    obj01  payload (BDDL On/In first argument)
    obj02  receptacle body (BDDL On/In second argument, or its parent if a site)
    table  the unique table fixture

Public entity names are humanized for eye-checkable FG
(``libero_g17_fg_lexicon``). BDDL ``On`` / ``In`` stay outside the G17 schema.
See ``docs/LIBERO_G17_CLOSED_SET_SPEC.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from capx.verification.closed_gt import (
    BodyState,
    EntitySpec,
    RawSceneState,
    SceneSpec,
    compile_snapshot,
    schema_audit,
    _array,
    _axis_value,
    _body_velocity,
    _normalized_quaternion,
    _projected_half_extents,
    _rotation_matrix_wxyz,
    _world_pose_to_frame,
)
from capx.verification.libero_g17_fg_lexicon import (
    assert_fg_eye_catalog_complete,
    humanize_libero_name,
)

Relation = Literal["on", "in"]
ContainKind = Literal["site_box", "composite_aabb"]
ExecutorAttachmentKind = Literal[
    "empty",
    "holding_payload",
    "holding_receptacle",
    "holding_other_movable",
]

PAYLOAD_ID = "obj01"
RECEPTACLE_ID = "obj02"
EXECUTOR_ID = "exe01"
TABLE_ID = "surface.table"
ATTACHMENT_KINDS: tuple[ExecutorAttachmentKind, ...] = (
    "empty",
    "holding_payload",
    "holding_receptacle",
    "holding_other_movable",
)

_ATTACHMENT_RANK: dict[str, int] = {
    "empty": 0,
    "holding_receptacle": 1,
    "holding_other_movable": 2,
    "holding_payload": 3,
}

_UNSUPPORTED_PREDICATES = frozenset({"open", "close", "turnon", "turnoff"})
_SUPPORTED_RELATIONS = frozenset({"on", "in"})
_DEFAULT_GRIPPER_METRIC_LENGTH = 0.04


class UnsupportedLiberoG17Goal(ValueError):
    """The BDDL goal is outside the G17 pick-place quotient."""


@dataclass(frozen=True)
class LiberoContainmentGeom:
    """Frame-aligned containment geometry for public In IFF (robot/base frame)."""

    kind: ContainKind
    position: np.ndarray  # (3,)
    size: np.ndarray  # (3,)
    rotation_mat: np.ndarray | None  # (3,3) for site_box; None for composite
    contact_required: bool


@dataclass(frozen=True)
class LiberoG17Binding:
    """Declared G17-goal quotient for one LIBERO pick-place task."""

    relation: Relation
    payload_name: str
    receptacle_eval_name: str
    receptacle_body_name: str
    table_name: str
    scene: SceneSpec
    suite_name: str | None = None
    task_id: int | None = None
    containment_geom: LiberoContainmentGeom | None = None

    @property
    def name(self) -> str:
        suite = self.suite_name or "libero"
        task = "na" if self.task_id is None else str(self.task_id)
        return f"{suite}_{task}_{self.relation}_{self.payload_name}_{self.receptacle_eval_name}"


@dataclass(frozen=True)
class LiberoAttachmentFact:
    """Executor attachment over **all** LIBERO movables (not only bound pair).

    Closes the hole where bound ``payload: empty`` was read as gripper-empty
    while a distractor was grasped. Fixtures never enter holding_*.

    Optional ``held_rel_xyz`` / ``held_vs_goal_rel_xyz`` carry robot-frame
    axis bands for the primary held movable (interaction completeness when
    the held body is outside the two-object G17 scene).
    """

    kind: ExecutorAttachmentKind
    held_movable_names: tuple[str, ...] = ()
    held_rel_xyz: tuple[str, str, str] | None = None
    held_vs_goal_rel_xyz: tuple[str, str, str] | None = None
    # Same-name sibling for R(goal): other body + goal-relative-to-other bands.
    same_name_other_raw: str | None = None
    goal_vs_same_name_other_rel_xyz: tuple[str, str, str] | None = None

    def assert_mutex_with_snapshot(self, snapshot: Any) -> None:
        """Fail closed if attachment disagrees with bound holding/payload."""
        values = snapshot.values
        holding_payload = (
            values["executor_object.exe01.obj01.holding"] == "TRUE"
        )
        holding_receptacle = (
            values["executor_object.exe01.obj02.holding"] == "TRUE"
        )
        payload = values["executor.exe01.payload"]
        if self.kind == "empty":
            if holding_payload or holding_receptacle or payload != "EMPTY":
                raise ValueError(
                    "attachment empty conflicts with bound holding/payload"
                )
            if self.held_movable_names:
                raise ValueError("attachment empty has held movables")
            return
        if self.kind == "holding_payload":
            if not holding_payload or payload != "OBJ01":
                raise ValueError(
                    "attachment holding_payload conflicts with bound snapshot"
                )
            if holding_receptacle:
                raise ValueError(
                    "attachment holding_payload conflicts with receptacle hold"
                )
            return
        if self.kind == "holding_receptacle":
            if not holding_receptacle or holding_payload or payload != "EMPTY":
                raise ValueError(
                    "attachment holding_receptacle conflicts with bound snapshot"
                )
            return
        if self.kind == "holding_other_movable":
            if holding_payload or payload != "EMPTY":
                raise ValueError(
                    "attachment holding_other_movable conflicts with payload hold"
                )
            if not self.held_movable_names:
                raise ValueError(
                    "attachment holding_other_movable requires held movables"
                )
            return
        raise ValueError(f"unknown attachment kind: {self.kind}")


def merge_attachment_peak(
    current: LiberoAttachmentFact | None, sample: LiberoAttachmentFact
) -> LiberoAttachmentFact:
    """Keep the strongest executor attachment observed in an interval."""
    if current is None:
        return sample
    cur_rank = _ATTACHMENT_RANK.get(current.kind, -1)
    samp_rank = _ATTACHMENT_RANK.get(sample.kind, -1)
    if samp_rank > cur_rank:
        return sample
    if samp_rank < cur_rank:
        return current
    if len(sample.held_movable_names) > len(current.held_movable_names):
        return sample
    # Prefer the sample that carries interaction RelXYZ bands.
    if sample.held_rel_xyz is not None and current.held_rel_xyz is None:
        return sample
    return current


def _frame_body_state(
    domain: Any,
    object_name: str,
    frame_position: np.ndarray,
    frame_orientation: np.ndarray,
) -> BodyState:
    """Body pose in the robot-base frame used by closed-GT RelXYZ."""
    sim_data = domain.sim.data
    body_id = _body_id(domain, object_name)
    model = domain.get_object(object_name)
    world_position = _array(sim_data.body_xpos[body_id], (3,), f"{object_name} position")
    world_orientation = _normalized_quaternion(
        sim_data.body_xquat[body_id], f"{object_name} orientation"
    )
    position, orientation = _world_pose_to_frame(
        world_position,
        world_orientation,
        frame_position,
        frame_orientation,
    )
    linear, angular = _body_velocity(sim_data, body_id)
    return BodyState(
        position,
        orientation,
        _half_extents(domain, model, object_name, body_id),
        linear,
        angular,
    )


def _rel_xyz_bands(subject: BodyState, reference: BodyState) -> tuple[str, str, str]:
    """Match closed_gt executor↔object / object↔object relative axis bands."""
    delta = subject.position - reference.position
    bands = _projected_half_extents(reference)
    return (
        _axis_value(float(delta[0]), float(bands[0])),
        _axis_value(float(delta[1]), float(bands[1])),
        _axis_value(float(delta[2]), float(bands[2])),
    )


def compute_held_interaction_bands(
    domain: Any,
    binding: LiberoG17Binding,
    held_movable_names: tuple[str, ...],
    *,
    wrapper: Any,
) -> tuple[tuple[str, str, str] | None, tuple[str, str, str] | None]:
    """Gripper↔held and held↔goal RelXYZ bands for interaction completeness."""
    if not held_movable_names:
        return None, None
    try:
        frame_position, frame_orientation = _robot_base_frame(domain, wrapper)
        gripper_body_id = _gripper_body_id(domain, wrapper)
        sim_data = domain.sim.data
        grip_pos, grip_ori = _world_pose_to_frame(
            _array(sim_data.body_xpos[gripper_body_id], (3,), "gripper position"),
            _normalized_quaternion(
                sim_data.body_xquat[gripper_body_id], "gripper orientation"
            ),
            frame_position,
            frame_orientation,
        )
        grip_lin, grip_ang = _body_velocity(sim_data, gripper_body_id)
        gripper = BodyState(grip_pos, grip_ori, None, grip_lin, grip_ang)
        held = _frame_body_state(
            domain, held_movable_names[0], frame_position, frame_orientation
        )
        held_rel = _rel_xyz_bands(gripper, held)
        if held_movable_names[0] == binding.payload_name:
            return held_rel, None
        goal = _frame_body_state(
            domain, binding.payload_name, frame_position, frame_orientation
        )
        vs_goal = _rel_xyz_bands(held, goal)
        return held_rel, vs_goal
    except Exception:
        return None, None


def enrich_same_name_goal_referent(
    domain: Any,
    binding: LiberoG17Binding,
    attachment: LiberoAttachmentFact,
    *,
    wrapper: Any,
) -> LiberoAttachmentFact:
    """Attach GT same-name sibling + goal↔other Rel for R(goal) API string.

    Fires whenever another graspable body shares the payload public name.
    ``goal_vs_same_name_other_rel_xyz`` is **goal relative to other** (directly
    usable in ``sample_grasp_pose`` language without band inversion).
    """
    payload = binding.payload_name
    pub = humanize_libero_name(payload)
    siblings = [
        name
        for name in _graspable_movable_names(domain)
        if humanize_libero_name(name) == pub and name != payload
    ]
    if not siblings:
        return attachment
    other: str | None = None
    if (
        attachment.kind == "holding_other_movable"
        and attachment.held_movable_names
        and humanize_libero_name(attachment.held_movable_names[0]) == pub
    ):
        other = attachment.held_movable_names[0]
    else:
        other = siblings[0]
    bands: tuple[str, str, str] | None = None
    try:
        frame_position, frame_orientation = _robot_base_frame(domain, wrapper)
        goal = _frame_body_state(
            domain, payload, frame_position, frame_orientation
        )
        other_body = _frame_body_state(
            domain, other, frame_position, frame_orientation
        )
        bands = _rel_xyz_bands(goal, other_body)
    except Exception:
        bands = None
    return replace(
        attachment,
        same_name_other_raw=other,
        goal_vs_same_name_other_rel_xyz=bands,
    )


def bind_libero_g17_goal(
    parsed_problem: dict[str, Any],
    *,
    suite_name: str | None = None,
    task_id: int | None = None,
    object_states: dict[str, Any] | None = None,
) -> LiberoG17Binding:
    """Bind exactly one BDDL On/In atom onto the G17 two-object scene."""
    atoms = _goal_atoms(parsed_problem.get("goal_state"))
    if not atoms:
        raise UnsupportedLiberoG17Goal("LIBERO goal has no atoms")
    unsupported = [
        atom for atom in atoms if _predicate_name(atom[0]) in _UNSUPPORTED_PREDICATES
    ]
    if unsupported:
        names = ", ".join(sorted({_predicate_name(atom[0]) for atom in unsupported}))
        raise UnsupportedLiberoG17Goal(
            f"G17 pick-place quotient excludes {names}"
        )
    relations = [atom for atom in atoms if _predicate_name(atom[0]) in _SUPPORTED_RELATIONS]
    other = [
        atom
        for atom in atoms
        if _predicate_name(atom[0]) not in _SUPPORTED_RELATIONS
    ]
    if other:
        names = ", ".join(sorted({_predicate_name(atom[0]) for atom in other}))
        raise UnsupportedLiberoG17Goal(
            f"G17 pick-place quotient cannot compile {names}"
        )
    if len(relations) != 1:
        raise UnsupportedLiberoG17Goal(
            "G17-goal requires exactly one On/In atom, "
            f"found {len(relations)}"
        )
    atom = relations[0]
    if len(atom) != 3:
        raise UnsupportedLiberoG17Goal("On/In atom must be binary")
    relation: Relation = "in" if _predicate_name(atom[0]) == "in" else "on"
    payload_name = str(atom[1])
    receptacle_eval_name = str(atom[2])
    receptacle_body_name = _body_name_for(
        receptacle_eval_name, object_states=object_states
    )
    table_name = _table_fixture_name(parsed_problem.get("fixtures") or {})
    scene = SceneSpec(
        executors=(EntitySpec(EXECUTOR_ID, "robot gripper", "executor"),),
        objects=(
            EntitySpec(PAYLOAD_ID, humanize_libero_name(payload_name), "object"),
            EntitySpec(
                RECEPTACLE_ID, humanize_libero_name(receptacle_body_name), "object"
            ),
        ),
        table=EntitySpec(TABLE_ID, humanize_libero_name(table_name), "surface"),
    )
    audit = schema_audit(scene)
    if audit["state_field_count"] != 51:
        raise RuntimeError(
            "LIBERO G17-goal scene must keep the two-object 51-field closure, "
            f"got {audit['state_field_count']}"
        )
    if not audit["unknown_is_not_a_gt_value"]:
        raise RuntimeError("LIBERO G17-goal scene introduced UNKNOWN")
    assert_fg_eye_catalog_complete(scene)
    return LiberoG17Binding(
        relation=relation,
        payload_name=payload_name,
        receptacle_eval_name=receptacle_eval_name,
        receptacle_body_name=receptacle_body_name,
        table_name=table_name,
        scene=scene,
        suite_name=suite_name,
        task_id=task_id,
    )


def bind_libero_g17_env(
    env: Any,
    *,
    suite_name: str | None = None,
    task_id: int | None = None,
) -> LiberoG17Binding:
    """Bind from a live LIBERO / CaP-X wrapper environment."""
    domain = _libero_domain(env)
    parsed = getattr(domain, "parsed_problem", None)
    if not isinstance(parsed, dict):
        raise TypeError("LIBERO G17 adapter requires parsed_problem")
    return bind_libero_g17_goal(
        parsed,
        suite_name=suite_name or _optional_suite(env),
        task_id=task_id if task_id is not None else _optional_task_id(env),
        object_states=getattr(domain, "object_states_dict", None),
    )


def extract_libero_raw_scene(env: Any, binding: LiberoG17Binding) -> RawSceneState:
    """Read the bound payload/receptacle into the G17 private interface."""
    domain = _libero_domain(env)
    wrapper = _capx_wrapper(env)
    required = ("sim", "robots", "obj_body_id", "get_object")
    if not all(hasattr(domain, name) for name in required):
        raise TypeError("LIBERO G17 adapter requires a LIBERO BDDL domain")

    sim = domain.sim
    sim_data = sim.data
    frame_position, frame_orientation = _robot_base_frame(domain, wrapper)
    object_names = {
        PAYLOAD_ID: binding.payload_name,
        RECEPTACLE_ID: binding.receptacle_body_name,
    }
    objects: dict[str, BodyState] = {}
    models: dict[str, Any] = {}
    body_ids: dict[str, int] = {}
    for object_id, object_name in object_names.items():
        body_id = _body_id(domain, object_name)
        model = domain.get_object(object_name)
        world_position = _array(sim_data.body_xpos[body_id], (3,), f"{object_id} position")
        world_orientation = _normalized_quaternion(
            sim_data.body_xquat[body_id], f"{object_id} orientation"
        )
        position, orientation = _world_pose_to_frame(
            world_position,
            world_orientation,
            frame_position,
            frame_orientation,
        )
        linear, angular = _body_velocity(sim_data, body_id)
        objects[object_id] = BodyState(
            position,
            orientation,
            _half_extents(domain, model, object_name, body_id),
            linear,
            angular,
        )
        models[object_id] = model
        body_ids[object_id] = body_id

    gripper_body_id = _gripper_body_id(domain, wrapper)
    gripper_position, gripper_orientation = _world_pose_to_frame(
        _array(sim_data.body_xpos[gripper_body_id], (3,), "gripper position"),
        _normalized_quaternion(
            sim_data.body_xquat[gripper_body_id], "gripper orientation"
        ),
        frame_position,
        frame_orientation,
    )
    gripper_linear, gripper_angular = _body_velocity(sim_data, gripper_body_id)
    executor = BodyState(
        gripper_position,
        gripper_orientation,
        None,
        gripper_linear,
        gripper_angular,
    )
    aperture = _gripper_aperture(domain, wrapper, sim_data)

    robot = domain.robots[0]
    gripper = robot.gripper
    holding = frozenset(
        (EXECUTOR_ID, object_id)
        for object_id, model in models.items()
        if bool(domain._check_grasp(gripper=gripper, object_geoms=model))
    )
    executor_contacts = frozenset(
        (EXECUTOR_ID, object_id)
        for object_id, model in models.items()
        if (EXECUTOR_ID, object_id) in holding
        or bool(domain.check_contact(gripper, model))
    )
    payload_receptacle_contact = bool(
        domain.check_contact(models[PAYLOAD_ID], models[RECEPTACLE_ID])
    )
    contacts = frozenset(
        {frozenset((PAYLOAD_ID, RECEPTACLE_ID))}
        if payload_receptacle_contact
        else set()
    )
    table_model = _table_model(domain, binding.table_name)
    table_contacts = frozenset(
        object_id
        for object_id, model in models.items()
        if _table_contact(domain, model, table_model)
    )
    table_world = _table_world_position(domain, binding.table_name, table_model)
    world_to_frame = _rotation_matrix_wxyz(frame_orientation).T
    table_normal_frame = world_to_frame @ np.array((0.0, 0.0, 1.0))
    if not np.allclose(
        table_normal_frame,
        np.array((0.0, 0.0, 1.0)),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Closed-GT LIBERO frame +Z is not the table normal")
    table_frame = world_to_frame @ (table_world - frame_position)
    return RawSceneState(
        executors={EXECUTOR_ID: executor},
        objects=objects,
        executor_aperture={EXECUTOR_ID: aperture},
        holding=holding,
        executor_contacts=executor_contacts,
        object_contacts=contacts,
        table_contacts=table_contacts,
        table_height=float(table_frame[2]),
    )


def extract_libero_containment_geom(
    env: Any, binding: LiberoG17Binding
) -> LiberoContainmentGeom | None:
    """Extract BDDL-aligned containment geometry in the robot/base frame."""
    if binding.relation != "in":
        return None
    domain = _libero_domain(env)
    wrapper = _capx_wrapper(env)
    states = getattr(domain, "object_states_dict", None)
    if not isinstance(states, dict):
        raise TypeError("LIBERO G17 In IFF requires object_states_dict")
    eval_name = binding.receptacle_eval_name
    if eval_name not in states:
        raise KeyError(f"missing receptacle state {eval_name}")
    state = states[eval_name]
    frame_position, frame_orientation = _robot_base_frame(domain, wrapper)
    world_to_frame = _rotation_matrix_wxyz(frame_orientation).T
    is_site = str(getattr(state, "object_state_type", "object")) == "site"
    if is_site:
        sites = getattr(domain, "object_sites_dict", None)
        if not isinstance(sites, dict) or eval_name not in sites:
            raise TypeError(f"site receptacle {eval_name} missing from object_sites_dict")
        site_obj = sites[eval_name]
        site_size = np.asarray(site_obj.size, dtype=np.float64).reshape(3)
        site_pos_world = _array(
            domain.sim.data.get_site_xpos(eval_name), (3,), "site position"
        )
        site_mat_world = np.asarray(
            domain.sim.data.get_site_xmat(eval_name), dtype=np.float64
        ).reshape(3, 3)
        site_pos_frame = world_to_frame @ (site_pos_world - frame_position)
        site_mat_frame = world_to_frame @ site_mat_world
        return LiberoContainmentGeom(
            kind="site_box",
            position=np.asarray(site_pos_frame, dtype=np.float64).reshape(3),
            size=site_size,
            rotation_mat=np.asarray(site_mat_frame, dtype=np.float64).reshape(3, 3),
            contact_required=False,
        )

    model = domain.get_object(binding.receptacle_body_name)
    body_id = _body_id(domain, binding.receptacle_body_name)
    body_pos_world = _array(
        domain.sim.data.body_xpos[body_id], (3,), "receptacle position"
    )
    body_pos_frame, _ = _world_pose_to_frame(
        body_pos_world,
        _normalized_quaternion(
            domain.sim.data.body_xquat[body_id], "receptacle orientation"
        ),
        frame_position,
        frame_orientation,
    )
    if not hasattr(model, "total_size"):
        raise TypeError(
            f"In receptacle {binding.receptacle_body_name} lacks total_size "
            "(expected CompositeObject.in_box surface)"
        )
    total_size = np.asarray(model.total_size, dtype=np.float64).reshape(3)
    return LiberoContainmentGeom(
        kind="composite_aabb",
        position=np.asarray(body_pos_frame, dtype=np.float64).reshape(3),
        size=total_size,
        rotation_mat=None,
        contact_required=True,
    )


def extract_libero_snapshot(env: Any, binding: LiberoG17Binding | None = None):
    """Compile the G17 snapshot for a bound LIBERO pick-place env.

    Returns ``(binding, snapshot, attachment)``. ``attachment`` classifies
    gripper grasp over **all** movables (executor occupancy), while the
    snapshot holding/payload fields remain bound-pair only.
    """
    if binding is None:
        binding = bind_libero_g17_env(env)
    raw = extract_libero_raw_scene(env, binding)
    snapshot = compile_snapshot(binding.scene, raw)
    geom = extract_libero_containment_geom(env, binding)
    binding = replace(binding, containment_geom=geom)
    domain = _libero_domain(env)
    gripper = domain.robots[0].gripper
    attachment = classify_executor_attachment(domain, gripper, binding)
    if attachment.kind != "empty" and attachment.held_movable_names:
        held_rel, vs_goal = compute_held_interaction_bands(
            domain,
            binding,
            attachment.held_movable_names,
            wrapper=_capx_wrapper(env),
        )
        attachment = replace(
            attachment,
            held_rel_xyz=held_rel,
            held_vs_goal_rel_xyz=vs_goal,
        )
    attachment = enrich_same_name_goal_referent(
        domain,
        binding,
        attachment,
        wrapper=_capx_wrapper(env),
    )
    attachment.assert_mutex_with_snapshot(snapshot)
    return binding, snapshot, attachment


def _graspable_movable_names(domain: Any) -> list[str]:
    """Body-level movables only (skip sites / fixtures / non-geom models)."""
    objects_dict = getattr(domain, "objects_dict", None)
    states = getattr(domain, "object_states_dict", None)
    if isinstance(objects_dict, dict) and objects_dict:
        candidate_names = sorted(str(name) for name in objects_dict.keys())
    elif isinstance(states, dict) and states:
        candidate_names = sorted(str(name) for name in states.keys())
    else:
        raise TypeError(
            "LIBERO executor attachment requires objects_dict or object_states_dict"
        )
    names: list[str] = []
    for name in candidate_names:
        model = domain.get_object(name)
        geoms = getattr(model, "contact_geoms", None)
        if isinstance(geoms, (list, tuple)):
            names.append(str(name))
    if not names:
        raise TypeError(
            "LIBERO executor attachment found no graspable movable bodies"
        )
    return names


def classify_executor_attachment(
    domain: Any,
    gripper: Any,
    binding: LiberoG17Binding,
) -> LiberoAttachmentFact:
    """ME partition of gripper bilateral grasp over all domain movables."""
    held: list[str] = []
    for name in _graspable_movable_names(domain):
        model = domain.get_object(name)
        geoms = getattr(model, "contact_geoms", None)
        if not isinstance(geoms, (list, tuple)):
            # SiteObject / fixtures are not bilateral-grasp targets.
            continue
        if bool(domain._check_grasp(gripper=gripper, object_geoms=model)):
            held.append(str(name))
    held_t = tuple(held)
    payload = binding.payload_name
    receptacle = binding.receptacle_body_name
    held_set = set(held_t)
    if not held_set:
        return LiberoAttachmentFact("empty", ())
    if held_set == {payload}:
        return LiberoAttachmentFact("holding_payload", held_t)
    if held_set == {receptacle}:
        return LiberoAttachmentFact("holding_receptacle", held_t)
    # Distractor-only, multiple objects, or payload+other → other occupancy.
    return LiberoAttachmentFact("holding_other_movable", held_t)


def attachment_from_bound_snapshot(snapshot: Any) -> LiberoAttachmentFact:
    """Infer attachment from bound fields only (no distractor visibility).

    Used by offline unit tests that lack a live domain. Cannot encode
    ``holding_other_movable``; callers that need distractor coverage must
    pass an explicit ``LiberoAttachmentFact``.
    """
    values = snapshot.values
    if values["executor_object.exe01.obj01.holding"] == "TRUE":
        return LiberoAttachmentFact("holding_payload", ("payload",))
    if values["executor_object.exe01.obj02.holding"] == "TRUE":
        return LiberoAttachmentFact("holding_receptacle", ("receptacle",))
    return LiberoAttachmentFact("empty", ())


def libero_bddl_atom_truth(env: Any, binding: LiberoG17Binding) -> bool:
    """Evaluate the unmodified LIBERO On/In atom. Not a G17 predicate."""
    domain = _libero_domain(env)
    states = getattr(domain, "object_states_dict", None)
    if not isinstance(states, dict):
        raise TypeError("LIBERO G17 adapter requires object_states_dict")
    payload = states[binding.payload_name]
    receptacle = states[binding.receptacle_eval_name]
    if binding.relation == "on":
        return bool(receptacle.check_ontop(payload))
    return bool(receptacle.check_contain(payload) and receptacle.check_contact(payload))


def libero_task_completed(env: Any) -> bool:
    """Private benchmark label. Never action-visible."""
    domain = _libero_domain(env)
    checker = getattr(domain, "_check_success", None)
    if checker is None:
        raise TypeError("LIBERO G17 adapter requires _check_success")
    return bool(checker())


def g17_versus_bddl_audit(
    snapshot,
    env: Any,
    binding: LiberoG17Binding,
) -> dict[str, Any]:
    """Compare G17 geometric proxies to the BDDL atom and task label."""
    bddl_true = libero_bddl_atom_truth(env, binding)
    task_completed = libero_task_completed(env)
    supported = (
        snapshot.values["object_object.obj01.obj02.supported_by_reference"] == "TRUE"
    )
    contact = snapshot.values["object_object.obj01.obj02.contact"] == "TRUE"
    footprint = snapshot.values["object_object.obj01.obj02.center_in_reference_footprint"]
    volume = snapshot.values["object_object.obj01.obj02.center_in_reference_volume"]
    holding = snapshot.values["executor_object.exe01.obj01.holding"] == "TRUE"
    geometric_on = supported
    geometric_in = volume == "INSIDE" and contact
    geometric_proxy = geometric_in if binding.relation == "in" else geometric_on
    from capx.verification.libero_g17_public_iff import (
        derive_libero_object_in_basis,
        derive_libero_object_ontop_basis,
    )

    ontop_basis = derive_libero_object_ontop_basis(snapshot)
    in_basis = None
    if binding.relation == "in" and binding.containment_geom is not None:
        in_basis = derive_libero_object_in_basis(snapshot, binding.containment_geom)
    public_atom = (
        ontop_basis.libero_object_ontop
        if binding.relation == "on"
        else (in_basis.libero_object_in if in_basis is not None else False)
    )
    return {
        "binding": binding.name,
        "relation": binding.relation,
        "payload_name": binding.payload_name,
        "receptacle_eval_name": binding.receptacle_eval_name,
        "bddl_atom_true": bddl_true,
        "private_task_completed": task_completed,
        "bddl_iff_task_completed": bddl_true == task_completed,
        "g17_supported_by": supported,
        "g17_contact": contact,
        "g17_footprint": footprint,
        "g17_volume": volume,
        "g17_payload_holding": holding,
        "libero_object_ontop": ontop_basis.libero_object_ontop,
        "libero_object_in": None if in_basis is None else in_basis.libero_object_in,
        "libero_public_atom": public_atom,
        "libero_public_iff_bddl": public_atom == bddl_true,
        "libero_ontop_iff_bddl": (
            ontop_basis.libero_object_ontop == bddl_true
            if binding.relation == "on"
            else None
        ),
        "geometric_proxy": geometric_proxy,
        "g17_iff_bddl": geometric_proxy == bddl_true,
        "action_visible": False,
        "schema_expanded": False,
    }


def _predicate_name(raw: Any) -> str:
    return str(raw).strip().lower().replace("_", "")


def _goal_atoms(goal_state: Any) -> list[list[Any]]:
    if goal_state is None:
        return []
    if not isinstance(goal_state, (list, tuple)):
        raise UnsupportedLiberoG17Goal("LIBERO goal_state is not a list")
    items = list(goal_state)
    if items and _predicate_name(items[0]) == "and":
        items = items[1:]
    atoms: list[list[Any]] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or not item:
            raise UnsupportedLiberoG17Goal("LIBERO goal atom is malformed")
        if _predicate_name(item[0]) == "and":
            atoms.extend(_goal_atoms(item))
            continue
        atoms.append(list(item))
    return atoms


def _table_fixture_name(fixtures: dict[str, Any]) -> str:
    named: list[tuple[str, str]] = []
    for category, instances in fixtures.items():
        if not isinstance(instances, (list, tuple)):
            continue
        for instance in instances:
            named.append((str(category), str(instance)))
    for _category, instance in named:
        if instance == "main_table":
            return instance
    tableish = [
        instance
        for category, instance in named
        if "table" in category.lower() or "table" in instance.lower()
    ]
    unique_table = sorted(set(tableish))
    if len(unique_table) == 1:
        return unique_table[0]
    floors = [
        instance
        for category, instance in named
        if category.lower() == "floor" or instance.lower() == "floor"
    ]
    unique_floor = sorted(set(floors))
    if len(unique_floor) == 1:
        return unique_floor[0]
    raise UnsupportedLiberoG17Goal(
        "G17-goal requires exactly one table/floor fixture; "
        f"found tables={unique_table or []} floors={unique_floor or []} "
        f"all={[name for _c, name in named]}"
    )


def _body_name_for(
    eval_name: str, *, object_states: dict[str, Any] | None
) -> str:
    if not object_states or eval_name not in object_states:
        return eval_name
    state = object_states[eval_name]
    if str(getattr(state, "object_state_type", "object")) != "site":
        return eval_name
    parent = getattr(state, "parent_name", None)
    if not parent:
        raise UnsupportedLiberoG17Goal(
            f"Site receptacle {eval_name} has no parent body"
        )
    return str(parent)


def _libero_domain(env: Any) -> Any:
    """Unwrap Cap-X / OffScreenRenderEnv layers to the BDDL domain."""
    candidates: list[Any] = [env]
    low = getattr(env, "low_level_env", None)
    if low is not None:
        candidates.append(low)
    for root in (env, low):
        if root is None:
            continue
        handle = getattr(root, "handle", None)
        if handle is None:
            continue
        candidates.append(handle)
        outer = getattr(handle, "env", None)
        if outer is not None:
            candidates.append(outer)
            # ControlEnv/OffScreenRenderEnv stores the BDDL problem on .env
            candidates.append(getattr(outer, "env", None))
        candidates.append(getattr(root, "env", None))
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        if hasattr(candidate, "parsed_problem") and hasattr(candidate, "sim"):
            return candidate
    raise TypeError("Closed-GT LIBERO adapter requires a LIBERO BDDL domain")


def _capx_wrapper(env: Any) -> Any | None:
    for root in (env, getattr(env, "low_level_env", None)):
        if root is None:
            continue
        if hasattr(root, "handle") and hasattr(root, "gripper_link_idx"):
            return root
    return None


def _optional_suite(env: Any) -> str | None:
    for root in (env, getattr(env, "low_level_env", None)):
        if root is None:
            continue
        handle = getattr(root, "handle", None)
        suite = getattr(handle, "suite_name", None) or getattr(root, "suite_name", None)
        if suite is not None:
            return suite
    return None


def _optional_task_id(env: Any) -> int | None:
    for root in (env, getattr(env, "low_level_env", None)):
        if root is None:
            continue
        handle = getattr(root, "handle", None)
        value = getattr(handle, "task_id", None)
        if value is None:
            value = getattr(root, "task_id", None)
        if value is not None:
            return int(value)
    return None


def _body_id(domain: Any, object_name: str) -> int:
    body_ids = domain.obj_body_id
    if object_name not in body_ids:
        raise ValueError(f"LIBERO object {object_name} has no body id")
    return int(body_ids[object_name])


def _robot_base_frame(domain: Any, wrapper: Any | None) -> tuple[np.ndarray, np.ndarray]:
    if wrapper is not None:
        pose = _array(wrapper.base_link_wxyz_xyz, (7,), "base pose")
        return pose[4:], _normalized_quaternion(pose[:4], "base orientation")
    body_id = domain.sim.model.body_name2id("robot0_base")
    return (
        _array(domain.sim.data.body_xpos[body_id], (3,), "base position"),
        _normalized_quaternion(
            domain.sim.data.body_xquat[body_id], "base orientation"
        ),
    )


def _gripper_body_id(domain: Any, wrapper: Any | None) -> int:
    if wrapper is not None:
        return int(wrapper.gripper_link_idx)
    return int(domain.sim.model.body_name2id("gripper0_eef"))


def _gripper_aperture(domain: Any, wrapper: Any | None, sim_data: Any) -> float:
    robot = domain.robots[0]
    indexes = _gripper_qpos_indexes(robot)
    gripper_qpos = _array(
        sim_data.qpos[np.asarray(indexes, dtype=np.int64)],
        (len(indexes),),
        "gripper qpos",
    )
    metric = float(
        getattr(wrapper, "gripper_metric_length", _DEFAULT_GRIPPER_METRIC_LENGTH)
        if wrapper is not None
        else _DEFAULT_GRIPPER_METRIC_LENGTH
    )
    if not np.isfinite(metric) or metric <= 0:
        raise ValueError("Invalid gripper metric length")
    # LIBERO/Panda gripper qpos[0] is the signed finger joint; normalize like Cap-X.
    return float(np.clip(float(gripper_qpos[0]) / metric, 0.0, 1.0))


def _gripper_qpos_indexes(robot: Any) -> tuple[int, ...]:
    raw = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    if isinstance(raw, dict):
        groups = [tuple(indexes) for indexes in raw.values() if indexes]
        if len(groups) != 1:
            raise ValueError("Closed-GT LIBERO adapter requires one two-finger gripper")
        indexes = tuple(int(x) for x in groups[0])
    elif isinstance(raw, (list, tuple)) and raw:
        indexes = tuple(int(x) for x in raw)
    else:
        joints = getattr(robot, "gripper_joints", None)
        if not joints:
            raise ValueError("Simulator exposes no gripper joint position indexes")
        sim = robot.sim
        indexes = tuple(int(sim.model.get_joint_qpos_addr(name)) for name in joints)
    if len(indexes) < 1:
        raise ValueError("Closed-GT LIBERO adapter requires gripper joint indexes")
    return indexes


def _half_extents(
    domain: Any, model: Any, object_name: str, body_id: int
) -> np.ndarray:
    size = getattr(model, "size", None)
    if size is not None:
        arr = np.asarray(size, dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])) and np.all(arr[:3] > 0):
            return arr[:3].copy()
    radius = getattr(model, "horizontal_radius", None)
    height = getattr(model, "height", None)
    if radius is not None and height is not None:
        radius_f = float(radius)
        height_f = float(np.abs(height))
        if radius_f > 0 and height_f > 0:
            return np.array((radius_f, radius_f, height_f / 2.0), dtype=np.float64)
    return _half_extents_from_mujoco(domain.sim, body_id, object_name)


def _half_extents_from_mujoco(sim: Any, body_id: int, object_name: str) -> np.ndarray:
    """Axis-aligned half extents of body geoms in the body frame."""
    model = sim.model
    data = sim.data
    body_pos = _array(data.body_xpos[body_id], (3,), f"{object_name} body pos")
    body_mat = np.asarray(data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    found = False
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom_id]) != int(body_id):
            continue
        # Skip visual-only / non-colliding geoms when a collision group exists.
        # LIBERO scanned objects often put massless visual meshes on the body.
        geom_pos = _array(data.geom_xpos[geom_id], (3,), f"{object_name} geom pos")
        geom_mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64).reshape(-1)
        gtype = int(model.geom_type[geom_id])
        # MuJoCo geom types: 0 plane, 1 heightfield, 2 sphere, 3 capsule,
        # 4 ellipsoid, 5 cylinder, 6 box, 7 mesh
        if gtype == 2:  # sphere
            local_corners = size[0] * np.array(
                [
                    [1, 1, 1],
                    [1, 1, -1],
                    [1, -1, 1],
                    [1, -1, -1],
                    [-1, 1, 1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [-1, -1, -1],
                ],
                dtype=np.float64,
            )
        elif gtype in (3, 5):  # capsule / cylinder — size=(radius, half-length)
            r, h = float(size[0]), float(size[1])
            local_corners = np.array(
                [
                    [r, r, h],
                    [r, r, -h],
                    [r, -r, h],
                    [r, -r, -h],
                    [-r, r, h],
                    [-r, r, -h],
                    [-r, -r, h],
                    [-r, -r, -h],
                ],
                dtype=np.float64,
            )
        elif gtype == 6:  # box
            local_corners = np.array(
                [
                    [size[0], size[1], size[2]],
                    [size[0], size[1], -size[2]],
                    [size[0], -size[1], size[2]],
                    [size[0], -size[1], -size[2]],
                    [-size[0], size[1], size[2]],
                    [-size[0], size[1], -size[2]],
                    [-size[0], -size[1], size[2]],
                    [-size[0], -size[1], -size[2]],
                ],
                dtype=np.float64,
            )
        elif gtype == 7:  # mesh — use geom AABB via size approximation from xpos only
            # Fall back to a small sphere around geom origin using geom_rbound.
            rbound = float(model.geom_rbound[geom_id])
            if not np.isfinite(rbound) or rbound <= 0:
                continue
            local_corners = rbound * np.array(
                [
                    [1, 1, 1],
                    [1, 1, -1],
                    [1, -1, 1],
                    [1, -1, -1],
                    [-1, 1, 1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [-1, -1, -1],
                ],
                dtype=np.float64,
            )
        else:
            continue
        world_corners = (geom_mat @ local_corners.T).T + geom_pos
        body_local = (body_mat.T @ (world_corners - body_pos).T).T
        mins = np.minimum(mins, body_local.min(axis=0))
        maxs = np.maximum(maxs, body_local.max(axis=0))
        found = True
    if not found:
        raise ValueError(f"LIBERO object {object_name} has no usable half extents")
    half = 0.5 * (maxs - mins)
    if not np.all(np.isfinite(half)) or not np.all(half > 0):
        raise ValueError(f"LIBERO object {object_name} has invalid AABB half extents")
    return half.astype(np.float64)


def _table_model(domain: Any, table_name: str) -> Any:
    getter = getattr(domain, "get_object", None)
    if getter is None:
        raise TypeError("LIBERO domain has no get_object")
    try:
        return getter(table_name)
    except Exception as exc:
        raise ValueError(f"LIBERO table fixture {table_name} is missing") from exc


def _table_contact(domain: Any, model: Any, table_model: Any) -> bool:
    if domain.check_contact(model, table_model):
        return True
    try:
        return bool(domain.check_contact(model, "table_collision"))
    except Exception:
        return False


def _table_world_position(domain: Any, table_name: str, table_model: Any) -> np.ndarray:
    offset = getattr(domain, "table_offset", None)
    if offset is not None:
        return _array(offset, (3,), "table offset")
    workspace = getattr(domain, "workspace_offset", None)
    if workspace is not None:
        return _array(workspace, (3,), "workspace offset")
    if table_name in getattr(domain, "obj_body_id", {}):
        body_id = int(domain.obj_body_id[table_name])
        return _array(domain.sim.data.body_xpos[body_id], (3,), "table position")
    # Floor arenas often have no body pose; use world origin as the support plane.
    if table_name.lower() == "floor":
        return np.zeros(3, dtype=np.float64)
    raise ValueError("LIBERO domain exposes no table position")
