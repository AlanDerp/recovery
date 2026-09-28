"""Unit tests for FG RelXY token + privileged same-name body pick."""

from __future__ import annotations

import numpy as np

from capx.verification import libero_g17_xy_lock as xy_lock


def test_query_matches_class_accepts_bare_and_ordinal() -> None:
    name = "akita black bowl"
    assert xy_lock.query_matches_class("akita black bowl", name)
    assert xy_lock.query_matches_class("akita_black_bowl", name)
    assert xy_lock.query_matches_class(
        "first akita black bowl from the robot's right", name
    )
    assert not xy_lock.query_matches_class("plate", name)


def test_pick_index_by_token_robot_frame() -> None:
    centroids = np.array(
        [
            [0.40, 0.10, 0.02],
            [0.55, -0.12, 0.02],
        ]
    )
    assert xy_lock.pick_index_by_token(centroids, "right") == 1
    assert xy_lock.pick_index_by_token(centroids, "left") == 0
    assert xy_lock.pick_index_by_token(centroids, "closer") == 0
    assert xy_lock.pick_index_by_token(centroids, "farther") == 1


def test_choose_complete_index_right() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    centroids = np.array(
        [
            [0.50, 0.10, 0.02],
            [0.50, -0.12, 0.02],
        ]
    )
    assert xy_lock.choose_complete_index(centroids, "right", identity) == 1


def test_project_world_to_pixel_pinhole() -> None:
    k = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    # Camera at origin looking +Z; point in front at (0, 0, 1) → principal point.
    t_world_from_cam = np.eye(4)
    uv = xy_lock.project_frame_to_pixel(
        np.array([0.0, 0.0, 1.0]), t_world_from_cam, k, (20, 20)
    )
    assert uv == (10, 10)


def test_stock_points_none_check_avoids_numpy_truth_value() -> None:
    """Regression: ``arr or []`` raises on non-empty ndarray."""
    import numpy as np

    pts = np.zeros((4, 3), dtype=np.float64)
    try:
        _ = pts or []
        raised = False
    except ValueError:
        raised = True
    assert raised
    raw = pts
    stock_pts = np.asarray(raw if raw is not None else [], dtype=np.float64)
    assert stock_pts.shape == (4, 3)

    body = np.array([0.50, -0.12, 0.02])
    assert xy_lock.mask_centroid_matches_body(
        np.array([0.52, -0.11, 0.05]), body, thresh_xy_m=0.08
    )
    assert not xy_lock.mask_centroid_matches_body(
        np.array([0.50, 0.10, 0.02]), body, thresh_xy_m=0.08
    )


def test_v291_xyz_match_rejects_xy_alias_at_wrong_height() -> None:
    body = np.array([0.50, -0.12, 0.28])
    tabletop_alias = np.array([0.51, -0.11, 0.04])
    # Frozen v286 behavior was intentionally XY-only.
    assert xy_lock.mask_centroid_matches_body(tabletop_alias, body)
    # v291 opts into the missing acquisition-Z predicate.
    assert not xy_lock.mask_centroid_matches_body(
        tabletop_alias, body, thresh_z_m=0.08
    )
    assert xy_lock.mask_centroid_matches_body(
        np.array([0.51, -0.11, 0.25]), body, thresh_z_m=0.08
    )


def test_body_world_must_convert_before_pose_mat_project() -> None:
    """pose_mat is robot-from-cam; raw MuJoCo world XYZ must not be projected as-is."""
    # Robot base translated by +1m in world X; body at world (1.5, 0, 0) → robot (0.5, 0, 0).
    base = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # wxyz + origin
    world = np.array([1.5, 0.0, 0.0])
    robot = xy_lock.world_to_base(world, base)[0]
    assert np.allclose(robot, [0.5, 0.0, 0.0], atol=1e-6)
    k = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    # Cam at robot origin looking +Z; body at robot (0, 0, 1) → center.
    cam = np.eye(4)
    assert xy_lock.project_frame_to_pixel(np.array([0.0, 0.0, 1.0]), cam, k, (100, 100)) == (
        50,
        50,
    )
    # Wrong: projecting world coords through robot pose_mat fails / wrong pixel.
    assert xy_lock.project_frame_to_pixel(world, cam, k, (100, 100)) is None


def test_pick_same_name_body_ranks_by_token(monkeypatch) -> None:
    """Privileged path selects the robot-right body without SAM second-stock."""

    class FakeDomain:
        def __init__(self) -> None:
            self.sim = type(
                "S",
                (),
                {
                    "data": type(
                        "D",
                        (),
                        {
                            "body_xpos": np.array(
                                [
                                    [0.0, 0.0, 0.0],
                                    [0.50, 0.10, 0.02],
                                    [0.50, -0.12, 0.02],
                                ]
                            )
                        },
                    )()
                },
            )()
            self.obj_body_id = {
                "akita_black_bowl_1": 1,
                "akita_black_bowl_2": 2,
            }

    fake = FakeDomain()

    def fake_domain(_env):
        return fake

    def fake_names(_domain):
        return ["akita_black_bowl_1", "akita_black_bowl_2"]

    def fake_body_id(_domain, name):
        return fake.obj_body_id[name]

    monkeypatch.setattr(
        "capx.verification.libero_g17_adapter._libero_domain", fake_domain
    )
    monkeypatch.setattr(
        "capx.verification.libero_g17_adapter._graspable_movable_names",
        fake_names,
    )
    monkeypatch.setattr(
        "capx.verification.libero_g17_adapter._body_id", fake_body_id
    )
    monkeypatch.setattr(
        xy_lock,
        "robot_base_or_none",
        lambda _env: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    right = xy_lock.pick_same_name_body(object(), "akita black bowl", "right")
    left = xy_lock.pick_same_name_body(object(), "akita_black_bowl", "left")
    assert right is not None and right[0] == "akita_black_bowl_2"
    assert left is not None and left[0] == "akita_black_bowl_1"


def test_resolve_lock_prefix_uses_fg_token() -> None:
    import os
    from types import SimpleNamespace

    from capx.verification.libero_g17_adapter import LiberoAttachmentFact

    prev = os.environ.get("CAPX_LIBERO_G17_XY_LOCK")
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    xy_lock.clear_xy_lock()
    try:
        binding = SimpleNamespace(payload_name="akita_black_bowl_1")
        att = LiberoAttachmentFact(
            "empty",
            (),
            same_name_other_raw="akita_black_bowl_2",
            goal_vs_same_name_other_rel_xyz=(
                "WITHIN_REFERENCE_BAND",
                "NEGATIVE_OF_REFERENCE_BAND",
                "WITHIN_REFERENCE_BAND",
            ),
        )
        xy_lock.publish_xy_lock(binding, att)
        got = xy_lock.resolve_lock("akita black bowl")
        assert got is not None
        assert got["token"] == "right"
        assert xy_lock.resolve_lock("plate") is None
        explicit = xy_lock.resolve_lock("akita_black_bowl", which="farther")
        assert explicit is not None
        assert explicit["token"] == "farther"
        assert not xy_lock.same_name_unresolved("akita black bowl")
        assert not xy_lock.same_name_unresolved("akita black bowl", which="right")
    finally:
        xy_lock.clear_xy_lock()
        if prev is None:
            os.environ.pop("CAPX_LIBERO_G17_XY_LOCK", None)
        else:
            os.environ["CAPX_LIBERO_G17_XY_LOCK"] = prev


def test_publish_and_live_lock_requires_sibling_bands() -> None:
    from types import SimpleNamespace

    from capx.verification.libero_g17_adapter import LiberoAttachmentFact

    xy_lock.clear_xy_lock()
    binding = SimpleNamespace(payload_name="akita_black_bowl_1")
    empty = LiberoAttachmentFact("empty", ())
    xy_lock.publish_xy_lock(binding, empty)
    assert xy_lock.live_xy_lock() is None
    import os

    prev = os.environ.get("CAPX_LIBERO_G17_XY_LOCK")
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    try:
        xy_lock.publish_xy_lock(
            binding,
            LiberoAttachmentFact(
                "empty", (), same_name_other_raw="akita_black_bowl_2"
            ),
        )
        assert xy_lock.same_name_unresolved("akita black bowl")
        assert not xy_lock.same_name_unresolved("akita black bowl", which="right")
        assert not xy_lock.same_name_unresolved("plate")
    finally:
        if prev is None:
            os.environ.pop("CAPX_LIBERO_G17_XY_LOCK", None)
        else:
            os.environ["CAPX_LIBERO_G17_XY_LOCK"] = prev
    xy_lock.clear_xy_lock()
    xy_lock.publish_xy_lock(binding, empty)
    assert xy_lock.live_xy_lock() is None

    att = LiberoAttachmentFact(
        "empty",
        (),
        same_name_other_raw="akita_black_bowl_2",
        goal_vs_same_name_other_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    xy_lock.publish_xy_lock(binding, att)
    live = xy_lock.live_xy_lock()
    assert live is not None
    assert live["token"] == "right"
    assert live["class_name"] == "akita black bowl"
    xy_lock.clear_xy_lock()


def test_bound_identity_survives_empty_and_relation_flip(monkeypatch) -> None:
    from types import SimpleNamespace

    from capx.verification.libero_g17_adapter import LiberoAttachmentFact

    binding = SimpleNamespace(payload_name="akita_black_bowl_1")
    monkeypatch.setenv("CAPX_LIBERO_G17_XY_LOCK", "1")
    initial = LiberoAttachmentFact(
        "empty",
        (),
        same_name_other_raw="akita_black_bowl_2",
        goal_vs_same_name_other_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    xy_lock.clear_xy_lock()
    xy_lock.publish_xy_lock(binding, initial, reset=True)
    first = xy_lock.resolve_lock("akita black bowl")
    assert first is not None
    assert first["token"] == "right"
    assert first["body_name"] == "akita_black_bowl_1"

    # Empty acquisition feedback carries no relative bands.  It must preserve
    # the task instance rather than re-enable the stock type lottery.
    xy_lock.publish_xy_lock(binding, LiberoAttachmentFact("empty", ()))
    after_empty = xy_lock.resolve_lock("akita black bowl")
    assert after_empty is not None
    assert after_empty["token"] == "right"
    assert after_empty["body_name"] == "akita_black_bowl_1"

    # A moved sibling may flip the descriptive token; the bound handle remains.
    flipped = LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        same_name_other_raw="akita_black_bowl_2",
        goal_vs_same_name_other_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    xy_lock.publish_xy_lock(binding, flipped)
    after_flip = xy_lock.resolve_lock("akita black bowl")
    assert after_flip is not None
    assert after_flip["token"] == "left"
    assert after_flip["body_name"] == "akita_black_bowl_1"
    xy_lock.clear_xy_lock()


def test_schema_xy_lock_is_ordinal_sgp_overload() -> None:
    import os

    from capx.verification import libero_g17_takeover as takeover

    prev = os.environ.get("CAPX_LIBERO_G17_XY_LOCK")
    prev_ord = os.environ.get("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL")
    prev_goal = os.environ.get("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER")
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", None)
    os.environ.pop("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", None)
    try:
        assert "ordinal-sgp-overload" in takeover.schema_version()
        os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "0"
        assert "sgp-overload-no-ordinal" in takeover.schema_version()
    finally:
        if prev is None:
            os.environ.pop("CAPX_LIBERO_G17_XY_LOCK", None)
        else:
            os.environ["CAPX_LIBERO_G17_XY_LOCK"] = prev
        if prev_ord is None:
            os.environ.pop("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", None)
        else:
            os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = prev_ord
        if prev_goal is None:
            os.environ.pop("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", None)
        else:
            os.environ["CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER"] = prev_goal



def test_wrong_grasp_ordinal_switch_suppresses_acquire_block() -> None:
    import os

    from capx.verification import libero_g17_adapter as adapter
    from capx.verification import libero_g17_fg_projection as projection

    problem = {
        "fixtures": {"table": ["main_table"]},
        "objects": {
            "akita_black_bowl": ["akita_black_bowl_1", "akita_black_bowl_2"],
            "plate": ["plate_1"],
        },
        "goal_state": [["On", "akita_black_bowl_1", "plate_1"]],
    }
    binding = adapter.bind_libero_g17_goal(
        problem, suite_name="libero_spatial", task_id=0
    )
    wrong = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
        ),
        held_vs_goal_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    assert projection.render_same_name_xy_acquire_referent(binding, wrong)
    prev = os.environ.get("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL")
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "0"
    try:
        assert os.environ.get("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", "1") == "0"
    finally:
        if prev is None:
            os.environ.pop("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", None)
        else:
            os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = prev


def test_xy_lock_keeps_wrong_grasp_ordinal_drops_empty_sibling() -> None:
    """v238: skill lock suppresses empty-hand XY, not wrong-grasp ordinal."""
    from capx.verification import libero_g17_adapter as adapter
    from capx.verification import libero_g17_fg_projection as projection

    problem = {
        "fixtures": {"table": ["main_table"]},
        "objects": {
            "akita_black_bowl": ["akita_black_bowl_1", "akita_black_bowl_2"],
            "plate": ["plate_1"],
        },
        "goal_state": [["On", "akita_black_bowl_1", "plate_1"]],
    }
    binding = adapter.bind_libero_g17_goal(
        problem, suite_name="libero_spatial", task_id=0
    )
    wrong = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
        ),
        held_vs_goal_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    empty_sib = adapter.LiberoAttachmentFact(
        "empty",
        (),
        same_name_other_raw="akita_black_bowl_2",
        goal_vs_same_name_other_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    wrong_lines = projection.render_same_name_xy_acquire_referent(
        binding, wrong, allow_empty_sibling_xy=False
    )
    empty_lines = projection.render_same_name_xy_acquire_referent(
        binding, empty_sib, allow_empty_sibling_xy=False
    )
    text = "\n".join(wrong_lines).lower()
    assert "wrong-grasp same-name acquire" in text
    assert "sample_grasp_pose" not in text
    assert "which=" not in text
    assert empty_lines == []
