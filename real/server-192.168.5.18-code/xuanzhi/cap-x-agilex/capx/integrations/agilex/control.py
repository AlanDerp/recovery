"""Control API scaffold for an Agilex/Piper single-arm robot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.utils.agilex_side_camera import save_snapshot
from capx.utils.camera_utils import scale_intrinsics_for_image

DEFAULT_PIPER_URDF = (
    Path(__file__).resolve().parents[2] / "envs" / "assets" / "agilex" / "piper_ik.urdf"
)
IDENTITY_WXYZ_XYZ = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
PIPER_JOINT_LOWER = np.array(
    [-2.6179938, 0.0, -2.9670597, -1.7453292, -1.2217304, -2.0943951],
    dtype=np.float64,
)
PIPER_JOINT_UPPER = np.array(
    [2.6179938, 3.1415926, 0.0, 1.7453292, 1.2217304, 2.0943951],
    dtype=np.float64,
)


def _as_wxyz_xyz(pose: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray:
    if pose is None:
        return IDENTITY_WXYZ_XYZ.copy()
    arr = np.asarray(pose, dtype=np.float64).reshape(7)
    norm = np.linalg.norm(arr[:4])
    if norm == 0:
        raise ValueError("Pose quaternion must be non-zero")
    arr[:4] /= norm
    return arr


def _as_quat_wxyz(quat: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Quaternion must be non-zero")
    return arr / norm


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _pose_to_matrix(pose_wxyz_xyz: np.ndarray) -> np.ndarray:
    pose = _as_wxyz_xyz(pose_wxyz_xyz)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = SciRotation.from_quat(_wxyz_to_xyzw(pose[:4])).as_matrix()
    mat[:3, 3] = pose[4:]
    return mat


def _matrix_to_pose(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    quat_wxyz = _xyzw_to_wxyz(SciRotation.from_matrix(mat[:3, :3]).as_quat())
    pos = mat[:3, 3].copy()
    return pos, quat_wxyz


def _apply_local_grasp_transform(
    grasp_pose: np.ndarray,
    local_offset: np.ndarray,
    local_rpy: np.ndarray,
    world_offset: np.ndarray,
) -> np.ndarray:
    """Convert Contact-GraspNet grasp frame to the robot gripper target frame."""
    raw = np.asarray(grasp_pose, dtype=np.float64).reshape(4, 4)
    transformed = raw.copy()
    transformed[:3, 3] = (
        raw[:3, 3]
        + raw[:3, :3] @ np.asarray(local_offset, dtype=np.float64)
        + np.asarray(world_offset, dtype=np.float64)
    )
    transformed[:3, :3] = raw[:3, :3] @ SciRotation.from_euler(
        "xyz", np.asarray(local_rpy, dtype=np.float64)
    ).as_matrix()
    return transformed


def _grasp_axis(rotation: np.ndarray, axis_name: str) -> np.ndarray:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    name = axis_name[1:] if axis_name.startswith("-") else axis_name
    if name not in {"x", "y", "z"}:
        raise ValueError("graspnet_approach_axis must be one of x, y, z, -x, -y, -z")
    axis_idx = {"x": 0, "y": 1, "z": 2}[name]
    axis = sign * np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, axis_idx]
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return axis
    return axis / norm


def _target_world_to_arm_base(
    position_world: np.ndarray,
    quaternion_world_wxyz: np.ndarray,
    base_pose_wxyz_xyz: np.ndarray,
    tcp_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position_world, dtype=np.float64).reshape(3)
    quat_wxyz = np.asarray(quaternion_world_wxyz, dtype=np.float64).reshape(4)
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)

    world_target = np.eye(4, dtype=np.float64)
    rot_world_target = SciRotation.from_quat(_wxyz_to_xyzw(quat_wxyz))
    world_target[:3, :3] = rot_world_target.as_matrix()
    world_target[:3, 3] = pos + rot_world_target.apply(np.asarray(tcp_offset, dtype=np.float64))

    world_base = _pose_to_matrix(base_pose_wxyz_xyz)
    base_target = np.linalg.inv(world_base) @ world_target
    return _matrix_to_pose(base_target)


class AgilexControlApi(ApiBase):
    """LLM-facing control helpers for an Agilex/Piper single-arm robot."""

    def __init__(
        self,
        env: BaseEnv,
        *,
        initialize_perception: bool = False,
        initialize_motion: bool = False,
        robot_urdf_path: str | None = None,
        target_link_name: str | None = None,
        base_pose_wxyz_xyz: list[float] | np.ndarray | None = None,
        tcp_offset: list[float] | np.ndarray | None = None,
        sam3_service_url: str | None = None,
        sam3_checkpoint_path: str | None = None,
        sam3_device: str | None = None,
        graspnet_service_url: str | None = None,
        use_graspnet_grasp_pose: bool | None = None,
        graspnet_frame_offset: list[float] | np.ndarray | None = None,
        graspnet_world_offset: list[float] | np.ndarray | None = None,
        graspnet_rotation_correction_rpy: list[float] | np.ndarray | None = None,
        graspnet_prefer_top_down: bool | None = None,
        graspnet_approach_axis: str | None = None,
        graspnet_top_down_threshold: float | None = None,
        graspnet_forward_passes: int | None = None,
        graspnet_max_retries: int | None = None,
        graspnet_local_regions: bool | None = None,
        graspnet_filter_grasps: bool | None = None,
        graspnet_fallback_to_topdown: bool | None = None,
        graspnet_max_full_points: int | None = None,
        graspnet_max_segment_points: int | None = None,
        object_depth_clip_range: tuple[float, float] | list[float] | None = None,
        object_min_points: int | None = None,
        object_world_offset: list[float] | np.ndarray | None = None,
        default_grasp_quaternion_wxyz: list[float] | np.ndarray | None = None,
        grasp_z_offset: float | None = None,
        grasp_ik_z_offsets: list[float] | tuple[float, ...] | None = None,
        grasp_quaternion_yaw_offsets: list[float] | tuple[float, ...] | None = None,
        grasp_quaternion_tilt_offsets_rpy: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        ik_rest_cost_weight: float | None = None,
        ik_soft_limit_margin_rad: float | None = None,
        ik_soft_limit_weight: float | None = None,
        joint_limit_margin: float | None = None,
        max_single_joint_delta: float | None = None,
        safe_z_clearance: float | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        self.debug = debug
        self.initialize_perception = initialize_perception
        self.initialize_motion = initialize_motion
        self._perception_models: dict[str, Any] = {}
        self.robot_urdf_path = str(
            robot_urdf_path
            or getattr(env, "agilex_ik_urdf_path", None)
            or DEFAULT_PIPER_URDF
        )
        self.target_link_name = target_link_name or getattr(
            env, "agilex_ik_target_link_name", "link6"
        )
        self._base_pose = _as_wxyz_xyz(
            base_pose_wxyz_xyz
            if base_pose_wxyz_xyz is not None
            else getattr(env, "base_pose_wxyz_xyz", None)
        )
        tcp_offset_value = (
            tcp_offset if tcp_offset is not None else getattr(env, "agilex_tcp_offset", None)
        )
        self._tcp_offset = (
            np.zeros(3, dtype=np.float64)
            if tcp_offset_value is None
            else np.asarray(tcp_offset_value, dtype=np.float64).reshape(3)
        )
        self.sam3_service_url = sam3_service_url or getattr(
            env, "sam3_service_url", "http://127.0.0.1:8114"
        )
        self.sam3_checkpoint_path = sam3_checkpoint_path or getattr(
            env, "sam3_checkpoint_path", "/home/agilex/checkpoints/sam3"
        )
        self.sam3_device = sam3_device or getattr(env, "sam3_device", "cuda")
        self.graspnet_service_url = graspnet_service_url or getattr(
            env, "graspnet_service_url", "http://127.0.0.1:8115"
        )
        self._use_graspnet_grasp_pose = bool(
            use_graspnet_grasp_pose
            if use_graspnet_grasp_pose is not None
            else getattr(env, "use_graspnet_grasp_pose", False)
        )
        self._graspnet_frame_offset = np.asarray(
            graspnet_frame_offset
            if graspnet_frame_offset is not None
            else getattr(env, "graspnet_frame_offset", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        self._graspnet_world_offset = np.asarray(
            graspnet_world_offset
            if graspnet_world_offset is not None
            else getattr(env, "graspnet_world_offset", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        self._graspnet_rotation_correction_rpy = np.asarray(
            graspnet_rotation_correction_rpy
            if graspnet_rotation_correction_rpy is not None
            else getattr(env, "graspnet_rotation_correction_rpy", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        self._graspnet_prefer_top_down = bool(
            graspnet_prefer_top_down
            if graspnet_prefer_top_down is not None
            else getattr(env, "graspnet_prefer_top_down", False)
        )
        self._graspnet_approach_axis = (
            graspnet_approach_axis
            if graspnet_approach_axis is not None
            else getattr(env, "graspnet_approach_axis", "z")
        )
        self._graspnet_top_down_threshold = float(
            graspnet_top_down_threshold
            if graspnet_top_down_threshold is not None
            else getattr(env, "graspnet_top_down_threshold", 0.65)
        )
        self._graspnet_forward_passes = int(
            graspnet_forward_passes
            if graspnet_forward_passes is not None
            else getattr(env, "graspnet_forward_passes", 1)
        )
        self._graspnet_max_retries = int(
            graspnet_max_retries
            if graspnet_max_retries is not None
            else getattr(env, "graspnet_max_retries", 7)
        )
        self._graspnet_local_regions = bool(
            graspnet_local_regions
            if graspnet_local_regions is not None
            else getattr(env, "graspnet_local_regions", True)
        )
        self._graspnet_filter_grasps = bool(
            graspnet_filter_grasps
            if graspnet_filter_grasps is not None
            else getattr(env, "graspnet_filter_grasps", False)
        )
        self._graspnet_fallback_to_topdown = bool(
            graspnet_fallback_to_topdown
            if graspnet_fallback_to_topdown is not None
            else getattr(env, "graspnet_fallback_to_topdown", False)
        )
        self._graspnet_max_full_points = int(
            graspnet_max_full_points
            if graspnet_max_full_points is not None
            else getattr(env, "graspnet_max_full_points", 20000)
        )
        self._graspnet_max_segment_points = int(
            graspnet_max_segment_points
            if graspnet_max_segment_points is not None
            else getattr(env, "graspnet_max_segment_points", 10000)
        )
        depth_clip_value = object_depth_clip_range or getattr(
            env, "object_depth_clip_range", (0.05, 2.0)
        )
        self._object_depth_clip_range = tuple(
            float(v) for v in np.asarray(depth_clip_value, dtype=np.float64).reshape(2)
        )
        self._object_min_points = int(
            object_min_points
            if object_min_points is not None
            else getattr(env, "object_min_points", 50)
        )
        self._object_world_offset = np.asarray(
            object_world_offset
            if object_world_offset is not None
            else getattr(env, "object_world_offset", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        grasp_quat_value = (
            default_grasp_quaternion_wxyz
            if default_grasp_quaternion_wxyz is not None
            else getattr(env, "default_grasp_quaternion_wxyz", None)
        )
        self._default_grasp_quaternion_wxyz = _as_quat_wxyz(
            grasp_quat_value if grasp_quat_value is not None else [1.0, 0.0, 0.0, 0.0]
        )
        self._grasp_z_offset = float(
            grasp_z_offset
            if grasp_z_offset is not None
            else getattr(env, "grasp_z_offset", 0.03)
        )
        ik_offsets_value = (
            grasp_ik_z_offsets
            if grasp_ik_z_offsets is not None
            else getattr(env, "grasp_ik_z_offsets", [0.0, 0.03, 0.06, 0.1])
        )
        self._grasp_ik_z_offsets = [
            float(v) for v in np.asarray(ik_offsets_value, dtype=np.float64).reshape(-1)
        ]
        yaw_offsets_value = (
            grasp_quaternion_yaw_offsets
            if grasp_quaternion_yaw_offsets is not None
            else getattr(env, "grasp_quaternion_yaw_offsets", [0.0])
        )
        self._grasp_quaternion_yaw_offsets = [
            float(v) for v in np.asarray(yaw_offsets_value, dtype=np.float64).reshape(-1)
        ]
        tilt_offsets_value = (
            grasp_quaternion_tilt_offsets_rpy
            if grasp_quaternion_tilt_offsets_rpy is not None
            else getattr(env, "grasp_quaternion_tilt_offsets_rpy", [[0.0, 0.0, 0.0]])
        )
        self._grasp_quaternion_tilt_offsets_rpy = np.asarray(
            tilt_offsets_value, dtype=np.float64
        ).reshape(-1, 3)
        self._ik_rest_cost_weight = float(
            ik_rest_cost_weight
            if ik_rest_cost_weight is not None
            else getattr(env, "ik_rest_cost_weight", 0.0)
        )
        self._ik_soft_limit_margin_rad = float(
            max(
                0.0,
                ik_soft_limit_margin_rad
                if ik_soft_limit_margin_rad is not None
                else getattr(env, "ik_soft_limit_margin_rad", 0.0),
            )
        )
        self._ik_soft_limit_weight = float(
            max(
                0.0,
                ik_soft_limit_weight
                if ik_soft_limit_weight is not None
                else getattr(env, "ik_soft_limit_weight", 0.0),
            )
        )
        self._joint_limit_margin = float(
            max(
                0.0,
                joint_limit_margin
                if joint_limit_margin is not None
                else getattr(env, "joint_limit_margin", 0.005),
            )
        )
        self._max_single_joint_delta = float(
            max(
                0.0,
                max_single_joint_delta
                if max_single_joint_delta is not None
                else getattr(env, "max_single_joint_delta", 0.0),
            )
        )
        self._safe_z_clearance = float(
            max(
                0.0,
                safe_z_clearance
                if safe_z_clearance is not None
                else getattr(env, "safe_z_clearance", 0.0),
            )
        )
        side_topic = getattr(env, "img_side_topic", None)
        self._side_camera_topic = str(side_topic) if side_topic else None
        self._side_camera_save = bool(getattr(env, "side_camera_save_during_task", False))
        self._side_camera_output_dir = Path(
            getattr(env, "side_camera_output_dir", "outputs/agilex_side_camera/task")
        )
        self._side_snapshot_idx = 0
        self._motion_initialized = False
        self._motion_ctx: Any | None = None
        self._pks: Any | None = None
        self._last_cfg = np.zeros(6, dtype=np.float64)
        self._joint_lower = PIPER_JOINT_LOWER.copy()
        self._joint_upper = PIPER_JOINT_UPPER.copy()

        if initialize_motion:
            self._initialize_motion_stack()
        if initialize_perception:
            self._initialize_perception_stack()

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_camera_observation": self.get_camera_observation,
            "move_to_joints": self.move_to_joints,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "solve_ik": self.solve_ik,
            "goto_pose": self.goto_pose,
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "verify_grasp": self.verify_grasp,
            "verify_tennis_in_box": self.verify_tennis_in_box,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the latest robot and camera observation.

        Returns:
            observation:
                Dictionary with:
                - ``robot_joint_pos``: shape (7,), six joints plus gripper.
                - ``robot0_robotview``: main camera RGB/depth/intrinsics/extrinsics.
        """
        self._log_step("get_observation", "Capturing Agilex observation ...")
        obs = self._env.get_observation()
        main_rgb = obs.get("robot0_robotview", {}).get("images", {}).get("rgb")
        self._log_step_update(images=main_rgb)
        return obs

    def get_camera_observation(self, camera: str = "main") -> dict[str, Any]:
        """Get one camera observation.

        Args:
            camera: Must be ``"main"``.

        Returns:
            Camera dictionary with ``images``, ``intrinsics``, ``pose``, and
            ``pose_mat`` when provided by the real robot client.
        """
        key = self._camera_key(camera)
        return self._env.get_observation().get(key, {})

    def _joint_bounds_with_margin(self) -> tuple[np.ndarray, np.ndarray]:
        margin = min(
            self._joint_limit_margin,
            float(np.min((self._joint_upper - self._joint_lower) / 2.0)) - 1e-6,
        )
        return self._joint_lower + margin, self._joint_upper - margin

    def _joint_limit_distances(self, joints: np.ndarray) -> np.ndarray:
        lower, upper = self._joint_bounds_with_margin()
        joints_arr = np.asarray(joints, dtype=np.float64).reshape(6)
        return np.minimum(joints_arr - lower, upper - joints_arr)

    def _joint_diagnostics(
        self, current: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_arr = np.asarray(current, dtype=np.float64).reshape(6)
        target_arr = np.asarray(target, dtype=np.float64).reshape(6)
        delta = target_arr - current_arr
        limit_dist = self._joint_limit_distances(target_arr)
        return current_arr, delta, limit_dist

    def _log_joint_diagnostics(
        self, *, context: str, current: np.ndarray, target: np.ndarray
    ) -> None:
        if not self.debug:
            return
        current_arr, delta, limit_dist = self._joint_diagnostics(current, target)
        print(
            f"[agilex:{context}] current={np.array2string(current_arr, precision=3)} "
            f"target={np.array2string(np.asarray(target), precision=3)} "
            f"max_delta={float(np.max(np.abs(delta))):.3f} "
            f"min_limit_margin={float(np.min(limit_dist)):.3f}"
        )

    def _validate_joint_target(
        self,
        target: np.ndarray,
        *,
        current: np.ndarray,
        context: str,
        allow_large_delta: bool = False,
    ) -> None:
        target_arr = np.asarray(target, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(target_arr)):
            raise RuntimeError(f"{context}: IK produced non-finite joint target {target_arr}")

        _current_arr, delta, limit_dist = self._joint_diagnostics(current, target_arr)
        max_delta = float(np.max(np.abs(delta)))
        min_limit_dist = float(np.min(limit_dist))
        self._log_joint_diagnostics(context=context, current=current, target=target_arr)

        if (
            self._max_single_joint_delta > 0.0
            and not allow_large_delta
            and max_delta > self._max_single_joint_delta
        ):
            raise RuntimeError(
                f"{context}: refusing large Agilex joint jump "
                f"(max_delta={max_delta:.3f} rad > {self._max_single_joint_delta:.3f}). "
                f"current={np.array2string(np.asarray(current), precision=3)} "
                f"target={np.array2string(target_arr, precision=3)}"
            )

    def _joint_target_is_safe(
        self,
        target: np.ndarray,
        *,
        current: np.ndarray,
        context: str,
        allow_large_delta: bool = False,
    ) -> bool:
        try:
            self._validate_joint_target(
                target,
                current=current,
                context=context,
                allow_large_delta=allow_large_delta,
            )
        except RuntimeError as exc:
            if self.debug:
                print(f"[agilex:{context}] rejected candidate: {exc}")
            return False
        return True

    def _capture_side_camera(self, label: str) -> None:
        if not self._side_camera_save or not self._side_camera_topic:
            return
        self._side_snapshot_idx += 1
        path = save_snapshot(
            self._side_camera_topic,
            self._side_camera_output_dir,
            label=label,
            step=self._side_snapshot_idx,
        )
        if path is not None:
            print(f"[side_camera] saved {path.resolve()}")

    def _move_to_joints_direct(
        self,
        joints: np.ndarray,
        *,
        gripper: float | None = None,
        context: str = "move_to_joints",
        allow_large_delta: bool = False,
    ) -> None:
        target = self._clip_joint_targets(joints)
        current = self._get_current_joint_positions()
        self._validate_joint_target(
            target,
            current=current,
            context=context,
            allow_large_delta=allow_large_delta,
        )
        self._env.move_to_joints_blocking(target, gripper=gripper)
        self._capture_side_camera(context)

    def move_to_joints(self, joints: np.ndarray, gripper: float | None = None) -> None:
        """Move to six joint positions.

        Args:
            joints: Shape (6,), joint targets in radians.
            gripper: Optional gripper fraction, 0.0 closed and 1.0 open.
        """
        self._move_to_joints_direct(joints, gripper=gripper)

    def open_gripper(self) -> None:
        """Open the gripper."""
        self._env._set_gripper(1.0)
        self._env._step_once()
        self._capture_side_camera("open_gripper")

    def close_gripper(self) -> None:
        """Close the gripper."""
        self._env._set_gripper(0.0)
        self._env._step_once()
        self._capture_side_camera("close_gripper")

    def verify_grasp(
        self, min_open_fraction: float = 0.08
    ) -> tuple[bool, dict[str, float | bool | str]]:
        """Check whether a closed gripper retained a non-zero opening.

        This is a conservative grasp gate, not a proof of object identity: a
        fully closed gripper means the ball was not retained and blocks any
        carrying motion. A side-camera snapshot is also saved when configured
        for operator review.
        """
        obs = self._env.get_observation()
        joint_pos = obs.get("robot_joint_pos")
        fraction = 0.0
        if joint_pos is not None:
            values = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
            if len(values) >= 7 and np.isfinite(values[6]):
                fraction = float(np.clip(values[6], 0.0, 1.0))
        threshold = float(np.clip(min_open_fraction, 0.0, 1.0))
        retained = fraction >= threshold
        self._capture_side_camera("verify_grasp_retained" if retained else "verify_grasp_empty")
        details: dict[str, float | bool | str] = {
            "success": retained,
            "reason": "retained_gripper_gap" if retained else "gripper_closed_to_empty_gap",
            "gripper_fraction": fraction,
            "min_open_fraction": threshold,
        }
        self._log_step_update(text=f"grasp gate: {details}")
        return retained, details

    def verify_tennis_in_box(self) -> tuple[bool, dict[str, Any]]:
        """Run the geometric tennis-ball-in-box verifier on the current scene."""
        from capx.envs.agilex_tennis_success import verify_tennis_ball_in_box_geometric

        return verify_tennis_ball_in_box_geometric(self)

    def solve_ik(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve IK for the 6-DoF arm.

        Args:
            position: Shape (3,), target XYZ in the robot world frame, meters.
            quaternion_wxyz: Shape (4,), target orientation as WXYZ quaternion.

        Returns:
            Shape (6,), joint targets in radians.
        """
        return self._solve_ik(position, quaternion_wxyz)

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, gripper: float | None = None
    ) -> None:
        """Move the end effector to a Cartesian pose via IK."""
        current = self._get_current_joint_positions()
        joints = self._solve_ik(position, quaternion_wxyz, initial_q=current)
        _current_arr, delta, _limit_dist = self._joint_diagnostics(current, joints)
        if self._should_use_safe_goto_path(gripper=gripper, delta=delta):
            self._goto_pose_via_safe_z(position, quaternion_wxyz, gripper=gripper)
            return
        self._move_to_joints_direct(
            joints,
            gripper=gripper,
            context="goto_pose",
        )

    def _should_use_safe_goto_path(
        self, *, gripper: float | None, delta: np.ndarray
    ) -> bool:
        if self._safe_z_clearance <= 0.0 or self._max_single_joint_delta <= 0.0:
            return False
        if float(np.max(np.abs(delta))) <= self._max_single_joint_delta:
            return False
        # Only auto-detour while the gripper is open or unspecified. Carrying moves
        # should fail fast instead of silently routing through a ready pose.
        return gripper is None or float(gripper) >= 0.5

    def _goto_pose_via_safe_z(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        gripper: float | None,
    ) -> None:
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        ready = np.asarray(
            getattr(self._env, "reset_joint_positions", np.zeros(6, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(6)
        if self.debug:
            print(
                "[agilex:goto_pose] large initial joint jump; routing via "
                "configured ready pose and safe-z waypoint"
            )

        self._move_to_joints_direct(
            ready,
            gripper=gripper,
            context="goto_pose_ready",
            allow_large_delta=True,
        )

        safe_position = target_position.copy()
        safe_position[2] = max(safe_position[2], self._safe_z_clearance)
        current = self._get_current_joint_positions()
        safe_joints = self._solve_ik(
            safe_position,
            quaternion_wxyz,
            initial_q=current,
        )
        self._move_to_joints_direct(
            safe_joints,
            gripper=gripper,
            context="goto_pose_safe_z",
        )

        current = self._get_current_joint_positions()
        final_joints = self._solve_ik(
            target_position,
            quaternion_wxyz,
            initial_q=current,
        )
        self._move_to_joints_direct(
            final_joints,
            gripper=gripper,
            context="goto_pose_final",
        )

    def get_object_pose(
        self, object_name: str, camera: str = "main"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate an object pose from one camera.

        Args:
            object_name: Natural language target, for example ``"red cup"``.
            camera: Must be ``"main"``.

        Returns:
            ``(position, quaternion_wxyz)`` in the robot world frame.
        """
        detection, points_world, _camera_data = self._observe_object_points(
            object_name,
            camera,
        )
        position = np.median(points_world, axis=0) + self._object_world_offset
        quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self._perception_models["last_object_detection"] = detection
        self._perception_models["last_object_points_world"] = points_world
        self._log_step_update(
            text=(
                f"{object_name} pose from {camera}: "
                f"pos={np.array2string(position, precision=4)}, "
                f"mask_pixels={int(np.count_nonzero(detection['mask']))}, "
                f"points={len(points_world)}"
            )
        )
        return position, quaternion_wxyz

    def sample_grasp_pose(
        self, object_name: str, camera: str = "main"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a reachable grasp pose for an object from RGB-D observations."""
        detection, object_points_world, camera_data = self._observe_object_points(
            object_name,
            camera,
        )
        object_pos = np.median(object_points_world, axis=0) + self._object_world_offset

        if self._use_graspnet_grasp_pose:
            try:
                return self._sample_graspnet_pose(
                    object_name=object_name,
                    camera=camera,
                    detection=detection,
                    object_points_world=object_points_world,
                    camera_data=camera_data,
                )
            except Exception as exc:
                if not self._graspnet_fallback_to_topdown:
                    raise
                if self.debug:
                    print(f"Contact-GraspNet grasp failed, falling back to top-down: {exc}")

        return self._sample_top_down_grasp_pose(object_name, camera, object_pos)

    def _sample_top_down_grasp_pose(
        self,
        object_name: str,
        camera: str,
        object_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        base_quat = self._default_grasp_quaternion_wxyz.copy()
        base_rot = SciRotation.from_quat(_wxyz_to_xyzw(base_quat))
        candidates: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []

        for extra_z in self._grasp_ik_z_offsets:
            grasp_pos = object_pos.copy()
            grasp_pos[2] += self._grasp_z_offset + extra_z
            pre_grasp_pos = grasp_pos + np.array([0.0, 0.0, 0.12], dtype=np.float64)
            for tilt_rpy in self._grasp_quaternion_tilt_offsets_rpy:
                tilt_rot = SciRotation.from_euler("xyz", tilt_rpy)
                for yaw_offset in self._grasp_quaternion_yaw_offsets:
                    yaw_rot = SciRotation.from_euler("z", yaw_offset)
                    quat = _xyzw_to_wxyz((base_rot * tilt_rot * yaw_rot).as_quat())
                    try:
                        current = self._get_current_joint_positions()
                        pre_joints = self._solve_ik(pre_grasp_pos, quat, initial_q=current)
                        if not self._joint_target_is_safe(
                            pre_joints,
                            current=current,
                            context=(
                                f"sample_pre_grasp_z_{extra_z:.3f}_"
                                f"tilt_{np.linalg.norm(tilt_rpy):.2f}_yaw_{yaw_offset:.2f}"
                            ),
                        ):
                            continue

                        joints = self._solve_ik(grasp_pos, quat, initial_q=pre_joints)
                        if not self._joint_target_is_safe(
                            joints,
                            current=pre_joints,
                            context=(
                                f"sample_grasp_z_{extra_z:.3f}_"
                                f"tilt_{np.linalg.norm(tilt_rpy):.2f}_yaw_{yaw_offset:.2f}"
                            ),
                        ):
                            continue
                    except Exception as exc:
                        if self.debug:
                            print(
                                f"IK rejected grasp at z offset {extra_z:.3f}, "
                                f"tilt {np.array2string(tilt_rpy, precision=2)}, "
                                f"yaw offset {yaw_offset:.2f}: {exc}"
                            )
                        continue
                    if np.all(np.isfinite(joints)):
                        max_delta = float(np.max(np.abs(pre_joints - current)))
                        min_limit_dist = float(
                            min(
                                np.min(self._joint_limit_distances(pre_joints)),
                                np.min(self._joint_limit_distances(joints)),
                            )
                        )
                        tilt_penalty = 0.05 * float(np.linalg.norm(tilt_rpy))
                        score = max_delta - 0.1 * min_limit_dist + tilt_penalty
                        candidates.append(
                            (score, extra_z, grasp_pos.copy(), quat.copy(), tilt_rpy.copy())
                        )

        if candidates:
            _score, extra_z, grasp_pos, quat, tilt_rpy = min(
                candidates, key=lambda item: item[0]
            )
            self._log_step_update(
                text=(
                    f"{object_name} grasp from {camera}: "
                    f"pos={np.array2string(grasp_pos, precision=4)}, "
                    f"quat={np.array2string(quat, precision=4)}, "
                    f"z_offset={extra_z:.3f}, "
                    f"tilt_rpy={np.array2string(tilt_rpy, precision=3)}"
                )
            )
            self._capture_side_camera("sample_grasp_pose")
            return grasp_pos, quat

        raise RuntimeError(
            f"No IK-reachable top-down grasp found for {object_name!r} "
            f"using z offsets {self._grasp_ik_z_offsets} and yaw offsets "
            f"{self._grasp_quaternion_yaw_offsets}; tilt offsets "
            f"{self._grasp_quaternion_tilt_offsets_rpy.tolist()}"
        )

    def _sample_graspnet_pose(
        self,
        *,
        object_name: str,
        camera: str,
        detection: dict[str, Any],
        object_points_world: np.ndarray,
        camera_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        plan_point_clouds = self._perception_models.get("contact_graspnet_point_clouds")
        if plan_point_clouds is None:
            raise RuntimeError("Contact-GraspNet point-cloud planner is not initialized")

        _rgb, depth, intrinsics, pose_mat = camera_data
        pc_full = self._depth_to_world_points(depth, intrinsics, pose_mat)
        pc_full = self._subsample_points(pc_full, self._graspnet_max_full_points)
        pc_segment = self._subsample_points(
            object_points_world,
            self._graspnet_max_segment_points,
        )
        if len(pc_segment) < self._object_min_points:
            raise RuntimeError(
                f"Only {len(pc_segment)} segment points available for Contact-GraspNet"
            )

        raw_grasps, scores, contact_pts = plan_point_clouds(
            pc_full.astype(np.float32),
            pc_segment.astype(np.float32),
            segmap_id=1,
            local_regions=self._graspnet_local_regions,
            filter_grasps=self._graspnet_filter_grasps,
            forward_passes=self._graspnet_forward_passes,
            max_retries=self._graspnet_max_retries,
        )
        raw_grasps = np.asarray(raw_grasps, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if raw_grasps.size == 0 or len(raw_grasps) == 0:
            raise RuntimeError("Contact-GraspNet returned no grasp candidates")
        if len(scores) != len(raw_grasps):
            scores = np.zeros(len(raw_grasps), dtype=np.float64)

        candidates = self._rank_graspnet_candidates(raw_grasps, scores)
        last_error: Exception | None = None
        for idx, transformed, alignment in candidates:
            grasp_pos, grasp_quat = _matrix_to_pose(transformed)
            try:
                joints = self.solve_ik(grasp_pos, grasp_quat)
            except Exception as exc:
                last_error = exc
                if self.debug:
                    print(
                        f"IK rejected GraspNet candidate {idx} "
                        f"(score={scores[idx]:.4f}, alignment={alignment:.3f}): {exc}"
                    )
                continue
            if np.all(np.isfinite(joints)):
                self._perception_models["last_graspnet_raw_grasps"] = raw_grasps
                self._perception_models["last_graspnet_scores"] = scores
                self._perception_models["last_graspnet_contact_pts"] = contact_pts
                self._perception_models["last_graspnet_selected_index"] = idx
                self._log_step_update(
                    text=(
                        f"{object_name} Contact-GraspNet grasp from {camera}: "
                        f"pos={np.array2string(grasp_pos, precision=4)}, "
                        f"quat={np.array2string(grasp_quat, precision=4)}, "
                        f"score={scores[idx]:.4f}, alignment={alignment:.3f}, "
                        f"mask_pixels={int(np.count_nonzero(detection['mask']))}"
                    )
                )
                if self.debug:
                    print(
                        f"Selected GraspNet candidate {idx}: "
                        f"score={scores[idx]:.4f}, alignment={alignment:.3f}, "
                        f"pos={np.array2string(grasp_pos, precision=4)}, "
                        f"quat={np.array2string(grasp_quat, precision=4)}"
                    )
                self._capture_side_camera("sample_grasp_pose_graspnet")
                return grasp_pos, grasp_quat

        raise RuntimeError(
            "No IK-reachable Contact-GraspNet grasp candidate found"
            + (f": {last_error}" if last_error is not None else "")
        )

    def _rank_graspnet_candidates(
        self,
        raw_grasps: np.ndarray,
        scores: np.ndarray,
    ) -> list[tuple[int, np.ndarray, float]]:
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        scored_candidates: list[tuple[int, np.ndarray, float, float]] = []
        for idx, raw_grasp in enumerate(raw_grasps):
            transformed = _apply_local_grasp_transform(
                raw_grasp,
                self._graspnet_frame_offset,
                self._graspnet_rotation_correction_rpy,
                self._graspnet_world_offset,
            )
            approach = _grasp_axis(transformed[:3, :3], self._graspnet_approach_axis)
            alignment = float(-np.dot(approach, world_z))
            scored_candidates.append((idx, transformed, float(scores[idx]), alignment))

        if not self._graspnet_prefer_top_down:
            ordered = sorted(scored_candidates, key=lambda item: item[2], reverse=True)
            return [
                (idx, transformed, alignment)
                for idx, transformed, _score, alignment in ordered
            ]

        top_down = [
            item for item in scored_candidates if item[3] >= self._graspnet_top_down_threshold
        ]
        if top_down:
            ordered = sorted(top_down, key=lambda item: item[2], reverse=True)
        else:
            ordered = sorted(
                scored_candidates,
                key=lambda item: (item[3], item[2]),
                reverse=True,
            )
        return [(idx, transformed, alignment) for idx, transformed, _score, alignment in ordered]

    def _observe_object_points(
        self,
        object_name: str,
        camera: str,
    ) -> tuple[dict[str, Any], np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        self._initialize_perception_stack()
        camera_obs = self.get_camera_observation(camera)
        rgb, depth, intrinsics, pose_mat = self._require_rgbd_camera(camera_obs, camera)
        detection = self._segment_object(rgb, object_name)
        points_world = self._mask_to_world_points(
            detection["mask"],
            depth,
            intrinsics,
            pose_mat,
        )
        return detection, points_world, (rgb, depth, intrinsics, pose_mat)

    def _depth_to_world_points(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_mat: np.ndarray,
    ) -> np.ndarray:
        near, far = self._object_depth_clip_range
        valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            raise RuntimeError("Depth image has no valid points for Contact-GraspNet")

        z = depth[ys, xs]
        x = (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
        points_cam = np.stack([x, y, z], axis=1)
        points_h = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        points_world = (pose_mat @ points_h.T).T[:, :3]
        return points_world[np.isfinite(points_world).all(axis=1)]

    def _subsample_points(self, points: np.ndarray, max_points: int) -> np.ndarray:
        points_arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        max_points = int(max_points)
        if max_points <= 0 or len(points_arr) <= max_points:
            return points_arr
        idx = np.random.default_rng(0).choice(len(points_arr), max_points, replace=False)
        return points_arr[idx]

    def _initialize_motion_stack(self) -> None:
        if self._motion_initialized:
            return
        from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        from capx.integrations.motion.pyroki_context import get_pyroki_context

        if not Path(self.robot_urdf_path).exists():
            raise FileNotFoundError(f"Agilex/Piper IK URDF not found: {self.robot_urdf_path}")

        self._motion_ctx = get_pyroki_context(
            self.robot_urdf_path,
            target_link_name=self.target_link_name,
        )
        self._pks = pks
        self._motion_initialized = True

    def _solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        initial_q: np.ndarray | None = None,
    ) -> np.ndarray:
        self._initialize_motion_stack()
        if self._motion_ctx is None or self._pks is None:
            raise RuntimeError("Agilex motion stack failed to initialize")

        pos_str = np.array2string(np.asarray(position, dtype=np.float64), precision=4)
        self._log_step("IK Solver", f"Solving IK for target {pos_str} ...")
        target_pos_base, target_quat_base = _target_world_to_arm_base(
            position,
            quaternion_wxyz,
            self._base_pose,
            self._tcp_offset,
        )
        initial_q_arr = (
            self._get_current_joint_positions()
            if initial_q is None
            else self._clip_joint_targets(initial_q)
        )

        joints = self._pks.solve_ik_rest(
            self._motion_ctx.robot,
            self._motion_ctx.target_link_name,
            target_quat_base,
            target_pos_base,
            rest_cost_weights=self._ik_rest_cost_weight,
            initial_q=initial_q_arr,
            soft_limit_margin=self._ik_soft_limit_margin_rad,
            soft_limit_weight=self._ik_soft_limit_weight,
            joint_lower=self._joint_lower,
            joint_upper=self._joint_upper,
        )
        joints = np.asarray(joints, dtype=np.float64).reshape(6)
        joints = self._clip_joint_targets(joints)
        self._last_cfg = joints.copy()

        self._log_joint_diagnostics(
            context="solve_ik",
            current=initial_q_arr,
            target=joints,
        )
        self._log_step_update(text=f"IK target {pos_str} -> joints solved")
        return joints

    def _clip_joint_targets(self, joints: np.ndarray) -> np.ndarray:
        joints_arr = np.asarray(joints, dtype=np.float64).reshape(6)
        lower, upper = self._joint_bounds_with_margin()
        return np.clip(joints_arr, lower, upper)

    def _get_current_joint_positions(self) -> np.ndarray:
        obs = self._env.get_observation()
        joint_obs = obs.get("robot_joint_pos")
        if isinstance(joint_obs, dict):
            for key in ("arm", "robot", "left"):
                if key in joint_obs:
                    joint_obs = joint_obs[key]
                    break
            else:
                joint_obs = None
        if joint_obs is not None:
            joints = np.asarray(joint_obs[:6], dtype=np.float64).reshape(6)
            if np.all(np.isfinite(joints)):
                clipped_joints = self._clip_joint_targets(joints)
                self._last_cfg = clipped_joints.copy()
                return clipped_joints
        return self._last_cfg.copy()

    def _initialize_perception_stack(self) -> None:
        if "sam3_segment" not in self._perception_models:
            from capx.integrations.vision import sam3 as sam3_module

            sam3_module.SERVICE_URL = self.sam3_service_url
            self._perception_models["sam3_segment"] = sam3_module.init_sam3(
                checkpoint_path=self.sam3_checkpoint_path,
                device=self.sam3_device,
            )

        if (
            self._use_graspnet_grasp_pose
            and "contact_graspnet_point_clouds" not in self._perception_models
        ):
            from capx.integrations.vision import graspnet as graspnet_module

            graspnet_module.SERVICE_URL = self.graspnet_service_url
            self._perception_models["contact_graspnet_point_clouds"] = (
                graspnet_module.init_contact_graspnet_point_clouds()
            )

    def _require_rgbd_camera(
        self, camera_obs: dict[str, Any], camera: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        images = camera_obs.get("images", {})
        rgb = images.get("rgb")
        depth = images.get("depth")
        intrinsics = camera_obs.get("intrinsics")
        pose_mat = camera_obs.get("pose_mat")

        if pose_mat is None and "pose" in camera_obs:
            pose = np.asarray(camera_obs["pose"], dtype=np.float64)
            if pose.shape == (4, 4):
                pose_mat = pose
            elif pose.size == 7:
                pose_mat = _pose_to_matrix(pose.reshape(7))

        missing = []
        if rgb is None:
            missing.append("images.rgb")
        if depth is None:
            missing.append("images.depth")
        if intrinsics is None:
            missing.append("intrinsics")
        if pose_mat is None:
            missing.append("pose_mat")
        if missing:
            raise RuntimeError(
                f"Camera {camera!r} is missing {', '.join(missing)}. "
                "get_object_pose() needs RGB, depth, intrinsics, and camera-to-world pose_mat."
            )

        rgb_arr = np.asarray(rgb, dtype=np.uint8)
        depth_arr = np.asarray(depth, dtype=np.float64)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.ndim != 2:
            raise ValueError(f"Depth image must be HxW or HxWx1, got {depth_arr.shape}")

        intrinsics_arr = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        if depth_arr.shape[:2] != rgb_arr.shape[:2]:
            intrinsics_arr = scale_intrinsics_for_image(
                intrinsics_arr,
                depth_arr.shape[:2],
                rgb_arr.shape[:2],
            )
            if self.debug:
                print(
                    f"[intrinsics] scaled K for depth {depth_arr.shape[:2]} "
                    f"from rgb ref {rgb_arr.shape[:2]}"
                )
        pose_mat_arr = np.asarray(pose_mat, dtype=np.float64).reshape(4, 4)
        return rgb_arr, depth_arr, intrinsics_arr, pose_mat_arr

    def _segment_object(self, rgb: np.ndarray, object_name: str) -> dict[str, Any]:
        segment = self._perception_models.get("sam3_segment")
        if segment is None:
            raise RuntimeError("SAM3 perception model is not initialized")

        results = segment(rgb, object_name)
        if not results:
            raise RuntimeError(f"SAM3 did not find any mask for {object_name!r}")

        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        mask = np.asarray(best["mask"], dtype=bool)
        if int(np.count_nonzero(mask)) <= 0:
            raise RuntimeError(f"SAM3 returned an empty mask for {object_name!r}")

        return {
            "mask": mask,
            "box": best.get("box"),
            "score": float(best.get("score", 0.0)),
            "label": best.get("label", object_name),
        }

    def _mask_to_world_points(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_mat: np.ndarray,
    ) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            from PIL import Image

            nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (depth.shape[1], depth.shape[0]), resample=nearest
                ),
                dtype=np.uint8,
            ) > 0

        near, far = self._object_depth_clip_range
        valid = mask & np.isfinite(depth) & (depth >= near) & (depth <= far)
        ys, xs = np.nonzero(valid)
        if len(xs) < self._object_min_points:
            raise RuntimeError(
                f"Object mask has only {len(xs)} valid depth point(s); "
                f"need at least {self._object_min_points}. Check depth image, mask, "
                f"and depth clip range {self._object_depth_clip_range}."
            )

        z = depth[ys, xs]
        x = (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
        points_cam = np.stack([x, y, z], axis=1)
        points_h = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float64)], axis=1
        )
        points_world = (pose_mat @ points_h.T).T[:, :3]
        finite = np.isfinite(points_world).all(axis=1)
        points_world = points_world[finite]
        if len(points_world) < self._object_min_points:
            raise RuntimeError(
                f"Only {len(points_world)} finite world point(s) after projection; "
                f"need at least {self._object_min_points}."
            )
        return points_world

    def _camera_key(self, camera: str) -> str:
        camera_keys = {
            "main": "robot0_robotview",
        }
        if camera not in camera_keys:
            raise ValueError("camera must be 'main'")
        return camera_keys[camera]
