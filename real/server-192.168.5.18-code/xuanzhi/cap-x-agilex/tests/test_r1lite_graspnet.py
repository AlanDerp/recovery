from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel
from capx.integrations.r1lite.control import (
    R1LITE_DEPLOYMENT_URDF,
    R1_LITE_IK_HARNESS,
    R1LiteControlApi,
    R1LiteIkHarness,
    _dynamic_wrist_observation_transform,
    _grasp_axis,
    _transform_graspnet_pose,
    derive_r1lite_ik_joint_limits_from_urdf,
)


def _valid_ik_waypoint(total_joint_delta_rad: float) -> dict[str, float]:
    return {
        "incremental_joint_delta_rad": 0.01,
        "total_joint_delta_rad": total_joint_delta_rad,
        "fk_position_error_m": 0.0,
        "fk_orientation_error_rad": 0.0,
        "forward_step_m": 0.0,
        "actual_forward_progress_m": 0.0,
        "expected_forward_progress_m": 0.0,
    }


def test_r1lite_harness_ignores_total_joint_overrun_below_tolerance() -> None:
    harness = R1LiteIkHarness(
        max_total_joint_delta_rad=1.85,
        max_total_joint_delta_overrun_rad=0.02,
    )

    harness.validate_waypoint(_valid_ik_waypoint(1.869))


def test_r1lite_harness_blocks_total_joint_overrun_at_tolerance() -> None:
    harness = R1LiteIkHarness(
        max_total_joint_delta_rad=1.85,
        max_total_joint_delta_overrun_rad=0.02,
    )

    with pytest.raises(RuntimeError, match=r"allowed_overrun<0.020"):
        harness.validate_waypoint(_valid_ik_waypoint(1.87))


def test_r1lite_harness_uses_soft_preference_below_hard_total_limit() -> None:
    harness = R1LiteIkHarness()
    urdf_limits = derive_r1lite_ik_joint_limits_from_urdf()

    harness.validate_waypoint(
        _valid_ik_waypoint(urdf_limits["preferred_total_joint_delta_rad"] - 0.01)
    )

    assert harness.preferred_total_joint_delta_rad == pytest.approx(
        urdf_limits["preferred_total_joint_delta_rad"]
    )
    assert harness.max_total_joint_delta_rad == pytest.approx(
        urdf_limits["max_total_joint_delta_rad"]
    )


def test_r1lite_harness_blocks_at_hard_total_limit() -> None:
    harness = R1LiteIkHarness()
    hard_limit = harness.max_total_joint_delta_rad

    with pytest.raises(RuntimeError, match=rf"limit={hard_limit:.3f}"):
        harness.validate_waypoint(_valid_ik_waypoint(hard_limit))


def test_r1lite_ik_joint_limits_match_deployment_urdf() -> None:
    limits = derive_r1lite_ik_joint_limits_from_urdf(R1LITE_DEPLOYMENT_URDF)

    assert limits["max_total_joint_delta_rad"] == pytest.approx(5.72469341496)
    assert limits["max_incremental_joint_delta_rad"] == pytest.approx(5.72469341496)
    assert limits["preferred_total_joint_delta_rad"] == pytest.approx(3.27587470748)
    assert R1_LITE_IK_HARNESS.max_total_joint_delta_rad == pytest.approx(
        limits["max_total_joint_delta_rad"]
    )


def test_transform_graspnet_pose_composes_camera_tf_and_local_tool_offset() -> None:
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = SciRotation.from_euler("z", 90, degrees=True).as_matrix()
    base_from_camera[:3, 3] = [1.0, 2.0, 3.0]
    camera_from_grasp = np.eye(4)
    camera_from_grasp[:3, 3] = [0.2, 0.0, 0.1]
    ee_from_grasp = np.eye(4)
    ee_from_grasp[:3, 3] = [0.0, 0.0, -0.05]

    transformed = _transform_graspnet_pose(
        camera_from_grasp,
        base_from_camera,
        ee_from_grasp=ee_from_grasp,
    )

    assert np.allclose(transformed[:3, 3], [1.0, 2.2, 3.15])
    assert np.allclose(transformed[:3, :3], base_from_camera[:3, :3])


def test_lookup_wrist_camera_to_base_builds_homogeneous_matrix() -> None:
    controller = object.__new__(R1LiteRealLowLevel)
    controller.wrist_tf_base_frame = "base_link"
    controller.wrist_tf_timeout_s = 1.0
    controller._Time = lambda *, nanoseconds: ("time", nanoseconds)
    controller._Duration = lambda *, seconds: ("duration", seconds)
    transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.1, y=-0.2, z=0.3),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    class Buffer:
        def lookup_transform(self, target, source, stamp, *, timeout):
            assert (target, source) == ("base_link", "wrist_optical")
            assert stamp == ("time", 123)
            assert timeout == ("duration", 1.0)
            return transform

    controller._tf_buffer = Buffer()

    matrix = controller.lookup_wrist_camera_to_base(
        frame_id="wrist_optical",
        stamp_ns=123,
    )

    assert np.allclose(matrix[:3, :3], np.eye(3))
    assert np.allclose(matrix[:3, 3], [0.1, -0.2, 0.3])


def test_dynamic_wrist_observation_keeps_orientation_and_centers_ball() -> None:
    ee_from_camera = np.eye(4)
    ee_from_camera[:3, 3] = [0.02, 0.0, 0.05]
    ball = np.array([0.40, -0.20, 0.90])
    reference_position = np.array([0.40, -0.20, 0.65])

    base_from_ee, details = _dynamic_wrist_observation_transform(
        ball,
        ee_from_camera=ee_from_camera,
        reference_ee_position=reference_position,
        reference_ee_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        distance_m=0.30,
        max_translation_m=0.20,
    )

    base_from_camera = base_from_ee @ ee_from_camera
    assert np.allclose(base_from_ee[:3, :3], np.eye(3))
    assert np.allclose(base_from_camera[:3, 3], [0.40, -0.20, 0.60])
    assert np.allclose(ball - base_from_camera[:3, 3], [0.0, 0.0, 0.30])
    assert details["optical_axis_error_m"] == pytest.approx(0.0)


def test_dynamic_wrist_observation_rejects_large_translation() -> None:
    with pytest.raises(ValueError, match="dynamic wrist translation"):
        _dynamic_wrist_observation_transform(
            np.array([0.80, 0.20, 0.90]),
            ee_from_camera=np.eye(4),
            reference_ee_position=np.array([0.20, -0.20, 0.60]),
            reference_ee_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            distance_m=0.30,
            max_translation_m=0.10,
        )


def _fake_tennis_api(*, raw_rotation: np.ndarray | None = None) -> R1LiteControlApi:
    api = object.__new__(R1LiteControlApi)
    downward_rotation = np.diag([1.0, -1.0, -1.0])
    raw_grasp = np.eye(4)
    raw_grasp[:3, :3] = (
        downward_rotation if raw_rotation is None else raw_rotation
    )
    raw_grasp[:3, 3] = [0.40, -0.20, 0.50]

    class FakeR1:
        graspnet_min_segment_points = 3
        graspnet_max_full_points = 100
        graspnet_max_segment_points = 100
        graspnet_max_candidates = 8
        graspnet_min_score = 0.1
        graspnet_prefer_top_down = True
        graspnet_top_down_threshold = 0.65
        graspnet_approach_axis = "z"
        graspnet_pregrasp_distance_m = 0.12
        graspnet_ee_from_grasp = np.eye(4)
        graspnet_max_joint_delta_rad = 1.2
        graspnet_max_approach_joint_delta_rad = 0.35
        graspnet_approach_validation_steps = 4
        wrist_ball_coarse_max_error_m = 0.12
        graspnet_table_clearance_m = 0.02

        def __init__(self):
            self.ik_position = np.zeros(3)
            self.ik_quaternion = np.array([0.0, 0.0, 0.0, 1.0])

        def object_prompt(self, _name):
            return "yellow tennis ball"

        def get_wrist_rgbd(self, *, require_depth):
            assert require_depth is True
            return {
                "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "depth": np.ones((4, 4), dtype=np.float32) * 0.5,
                "intrinsics": np.eye(3),
                "frame_id": "wrist_optical",
                "stamp_ns": 123,
            }

        def lookup_wrist_camera_to_base(self, *, frame_id, stamp_ns):
            assert (frame_id, stamp_ns) == ("wrist_optical", 123)
            return np.eye(4)

        def lookup_camera_to_base(self, *, frame_id, stamp_ns, camera_label):
            assert (frame_id, stamp_ns, camera_label) == (
                "wrist_optical",
                123,
                "right_wrist",
            )
            return np.eye(4)

        def get_arm_joints(self):
            return np.zeros(6)

        def validate_workspace_point(self, point):
            point = np.asarray(point)
            if not (
                0.0 <= point[0] <= 0.8
                and -0.7 <= point[1] <= 0.4
                and 0.3 <= point[2] <= 1.3
            ):
                raise ValueError("outside workspace")

        def load_head_plane_calibration(self):
            return {"plane_z": 0.4}

        def get_relaxed_ik_candidate(
            self,
            position,
            quaternion,
            *,
            timeout_s,
            **_constraints,
        ):
            assert np.isfinite(position).all()
            assert np.isfinite(quaternion).all()
            assert timeout_s == 2.0
            self.ik_position = np.asarray(position, dtype=float)
            self.ik_quaternion = np.asarray(quaternion, dtype=float)
            return np.zeros(6)

        def reset_relaxed_ik_probe(self):
            return None

        def get_planning_ee_pose(self):
            return (
                np.array([0.4, -0.2, 0.5]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            )

        def get_relaxed_ik_fk_pose_base(self):
            return self.ik_position.copy(), self.ik_quaternion.copy()

    api._r1 = FakeR1()
    api.locate_object_head = lambda *args, **kwargs: {
        "ok": True,
        "pixel": [1.0, 1.0],
    }
    api._segment_tennis_mask = lambda *args, **kwargs: {
        "ok": True,
        "mask": np.ones((4, 4), dtype=bool),
        "method": "test",
    }
    api._wrist_point_clouds = lambda *args, **kwargs: (
        np.ones((8, 3), dtype=np.float64),
        np.ones((6, 3), dtype=np.float64),
    )
    api._call_graspnet_point_clouds = lambda *args, **kwargs: (
        raw_grasp[None, ...],
        np.array([0.9]),
        np.array([[0.4, -0.2, 0.5]]),
    )
    api.save_wrist_rgb = lambda _path: _path
    return api


def test_tennis_graspnet_plan_uses_candidate_pose_and_approach_axis() -> None:
    api = _fake_tennis_api()

    plan = api.plan_tennis_graspnet(
        camera="wrist",
        use_sam3=False,
        save_debug=False,
    )

    selected = plan["selected"]
    assert plan["mandatory_graspnet"] is True
    assert plan["fallback_used"] is False
    assert selected["score"] == pytest.approx(0.9)
    assert np.allclose(selected["grasp_position"], [0.4, -0.2, 0.5])
    assert np.allclose(selected["approach_axis_base"], [0.0, 0.0, -1.0])
    assert selected["top_down_alignment"] == pytest.approx(1.0)
    assert np.allclose(selected["pregrasp_position"], [0.4, -0.2, 0.62])


def test_tennis_graspnet_rejects_bottom_up_candidate() -> None:
    api = _fake_tennis_api(raw_rotation=np.eye(3))

    with pytest.raises(RuntimeError, match="approach is not top-down"):
        api.plan_tennis_graspnet(
            camera="wrist",
            use_sam3=False,
            save_debug=False,
        )


def test_tennis_graspnet_rejects_grasp_below_table_before_ik() -> None:
    api = _fake_tennis_api()
    grasp = np.eye(4)
    grasp[:3, :3] = np.diag([1.0, -1.0, -1.0])
    grasp[:3, 3] = [0.4, -0.2, 0.39]
    api._call_graspnet_point_clouds = lambda *args, **kwargs: (
        grasp[None, ...],
        np.array([0.9]),
        np.array([[0.4, -0.2, 0.39]]),
    )

    with pytest.raises(RuntimeError, match="below the calibrated table plane"):
        api.plan_tennis_graspnet(
            camera="wrist",
            use_sam3=False,
            save_debug=False,
        )


def test_tennis_graspnet_prefers_lower_motion_executable_candidate() -> None:
    api = _fake_tennis_api()
    lower_score_grasp = np.eye(4)
    lower_score_grasp[:3, :3] = np.diag([1.0, -1.0, -1.0])
    lower_score_grasp[:3, 3] = [0.5, -0.2, 0.5]
    higher_score_grasp = lower_score_grasp.copy()
    higher_score_grasp[:3, 3] = [0.4, -0.2, 0.5]
    api._call_graspnet_point_clouds = lambda *args, **kwargs: (
        np.stack([higher_score_grasp, lower_score_grasp]),
        np.array([0.9, 0.8]),
        np.array([[0.4, -0.2, 0.5], [0.5, -0.2, 0.5]]),
    )

    def fake_plan(position, _quaternion, **_kwargs):
        total_delta = 1.0 if np.asarray(position)[0] < 0.45 else 0.5
        waypoint = {
            "incremental_joint_delta_rad": 0.01,
            "total_joint_delta_rad": total_delta,
            "fk_position_error_m": 0.01,
        }
        return {
            "waypoint_count": 1,
            "harness": R1_LITE_IK_HARNESS.to_dict(),
            "waypoints": [waypoint],
        }

    api.plan_pose_waypoints = fake_plan

    plan = api.plan_tennis_graspnet(
        camera="wrist",
        use_sam3=False,
        save_debug=False,
    )

    assert plan["selected"]["index"] == 1
    assert plan["selected"]["score"] == pytest.approx(0.8)
    assert [candidate["index"] for candidate in plan["accepted_candidates"]] == [1, 0]
    assert plan["selection_policy"].startswith("executable_then_min_total_joint_delta")


def test_tennis_graspnet_has_no_fixed_pose_fallback() -> None:
    api = _fake_tennis_api(raw_rotation=np.eye(3))
    api._r1.get_relaxed_ik_candidate = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(RuntimeError("IK rejected"))
    )

    with pytest.raises(RuntimeError, match="No Contact-GraspNet candidate"):
        api.plan_tennis_graspnet(
            camera="wrist",
            use_sam3=False,
            save_debug=False,
        )


def test_tennis_graspnet_rejects_candidate_far_from_refined_ball() -> None:
    api = _fake_tennis_api()

    with pytest.raises(RuntimeError, match="No Contact-GraspNet candidate"):
        api.plan_tennis_graspnet(
            camera="wrist",
            use_sam3=False,
            save_debug=False,
            expected_ball_position_base=[0.1, 0.1, 0.9],
        )


def test_tennis_graspnet_requires_calibrated_tool_transform() -> None:
    api = _fake_tennis_api()
    api._r1.graspnet_ee_from_grasp = None

    with pytest.raises(RuntimeError, match="calibrated graspnet_ee_from_grasp"):
        api.plan_tennis_graspnet(
            camera="wrist",
            use_sam3=False,
            save_debug=False,
        )


def test_grasp_axis_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="approach_axis"):
        _grasp_axis(np.eye(3), "forward")


class _FakeWaypointR1:
    def __init__(self, joint_scale: float) -> None:
        self.joint_scale = float(joint_scale)
        self.target = np.zeros(3)
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self.reset_count = 0
        self.constraint_calls = []

    def validate_workspace_point(self, point) -> None:
        assert np.isfinite(point).all()

    def reset_relaxed_ik_probe(self) -> None:
        self.reset_count += 1

    def get_planning_ee_pose(self):
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])

    def get_arm_joints(self):
        return np.zeros(6)

    def get_relaxed_ik_candidate(
        self,
        position,
        quaternion,
        *,
        timeout_s,
        **constraints,
    ):
        assert timeout_s == 2.0
        self.constraint_calls.append(constraints)
        self.target = np.asarray(position, dtype=float)
        self.quaternion = np.asarray(quaternion, dtype=float)
        joints = np.zeros(6)
        joints[0] = self.joint_scale * self.target[0]
        return joints

    def get_relaxed_ik_fk_pose_base(self):
        return self.target.copy(), self.quaternion.copy()


def test_pose_waypoint_plan_validates_complete_path_before_execution() -> None:
    api = object.__new__(R1LiteControlApi)
    api._r1 = _FakeWaypointR1(joint_scale=5.0)

    plan = api.plan_pose_waypoints([0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    assert plan["waypoint_count"] == 8
    assert plan["waypoints"][-1]["total_joint_delta_rad"] == pytest.approx(0.5)
    assert plan["harness"] == R1_LITE_IK_HARNESS.to_dict()
    assert api._r1.reset_count == 1
    assert api._r1.constraint_calls
    assert api._r1.constraint_calls[0]["max_incremental_joint_delta_rad"] == (
        R1_LITE_IK_HARNESS.max_incremental_joint_delta_rad
    )
    assert api._r1.constraint_calls[0]["max_total_joint_delta_rad"] == (
        R1_LITE_IK_HARNESS.max_total_joint_delta_rad
    )


def test_pose_waypoint_plan_rejects_below_table_before_ik() -> None:
    api = object.__new__(R1LiteControlApi)
    api._r1 = _FakeWaypointR1(joint_scale=5.0)

    with pytest.raises(ValueError, match="below the active table constraint"):
        api.plan_pose_waypoints(
            [0.1, 0.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
            min_z_m=0.5,
        )

    assert api._r1.reset_count == 0
    assert not api._r1.constraint_calls


def test_r1lite_harness_accepts_smooth_path_above_old_total_limit() -> None:
    api = object.__new__(R1LiteControlApi)
    api._r1 = _FakeWaypointR1(joint_scale=13.0)

    plan = api.plan_pose_waypoints(
        [0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )

    assert plan["waypoints"][-1]["total_joint_delta_rad"] == pytest.approx(1.3)
    assert max(
        point["incremental_joint_delta_rad"] for point in plan["waypoints"]
    ) < R1_LITE_IK_HARNESS.max_incremental_joint_delta_rad


def test_pose_waypoint_plan_rejects_large_total_joint_branch() -> None:
    api = object.__new__(R1LiteControlApi)
    api._r1 = _FakeWaypointR1(joint_scale=15.0)

    with pytest.raises(RuntimeError, match="total joint displacement"):
        api.plan_pose_waypoints(
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            max_cartesian_step_m=0.01,
            max_incremental_joint_delta_rad=0.18,
            max_total_joint_delta_rad=1.20,
        )


def test_deferred_gripper_failure_does_not_block_arm_target() -> None:
    r1 = object.__new__(R1LiteRealLowLevel)
    r1._pending_gripper_target = 14.5
    r1._last_action = {}
    r1._ensure_moca_for_joint_target = lambda _command: None
    r1._enable_arm_joint_control = lambda: None
    r1.command_gripper = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("no gripper response")
    )
    r1._make_joint_state = lambda _name, _position: object()
    r1.arm_name = "right_arm"
    r1._lock = Lock()
    r1._arm_feedback_sequence = 1
    r1.publish_rate_hz = 20.0
    r1.adaptive_feedback_settle_timeout_s = 0.1
    r1._arm_pub = SimpleNamespace(publish=lambda _message: None)
    r1._wait_for_settled_arm_feedback = lambda **_kwargs: np.zeros(6)

    measured = r1._publish_joint_command_for(np.zeros(6), duration_s=0.0)

    assert np.allclose(measured, np.zeros(6))
    assert r1._pending_gripper_target is None
    assert r1._last_action["deferred_gripper_warning"]["arm_motion_continued"] is True
