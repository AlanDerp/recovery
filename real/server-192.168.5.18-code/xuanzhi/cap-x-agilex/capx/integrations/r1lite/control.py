"""Safety-first Cap-X API for a real Galaxea/Xinghaitu R1 Lite."""

from __future__ import annotations

import base64
import colorsys
import io
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel
from capx.integrations.base_api import ApiBase
from capx.integrations.r1lite.cup_rgbd import reconstruct_inverted_cup_pose

R1LITE_DEPLOYMENT_URDF = Path(
    "/home/nvidia/galaxea/install/r1lite_urdf/share/r1lite_urdf/urdf/r1lite_2026.urdf"
)
R1LITE_DEFAULT_COMMAND_HOME = (
    0.010470,
    0.222985,
    -0.022772,
    0.008466,
    0.035299,
    0.006095,
)


def _parse_r1lite_right_arm_limits(urdf_path: Path) -> list[tuple[float, float]]:
    root = ET.parse(urdf_path).getroot()
    limits: list[tuple[float, float]] = []
    for index in range(1, 7):
        joint = root.find(f".//joint[@name='right_arm_joint{index}']/limit")
        if joint is None:
            raise ValueError(
                f"Missing right_arm_joint{index} limits in {urdf_path}"
            )
        limits.append((float(joint.get("lower")), float(joint.get("upper"))))
    return limits


def derive_r1lite_ik_joint_limits_from_urdf(
    urdf_path: Path | None = None,
    command_home: tuple[float, ...] | list[float] | None = None,
) -> dict[str, float]:
    """Derive joint-displacement IK limits from the deployed R1 Lite URDF."""
    path = Path(urdf_path or R1LITE_DEPLOYMENT_URDF)
    home = tuple(command_home or R1LITE_DEFAULT_COMMAND_HOME)
    if len(home) != 6:
        raise ValueError("command_home must contain six right-arm joints")

    joint_limits = _parse_r1lite_right_arm_limits(path)
    ranges = [upper - lower for lower, upper in joint_limits]
    max_range = float(max(ranges))
    max_from_home = float(
        max(
            max(abs(home[index] - lower), abs(home[index] - upper))
            for index, (lower, upper) in enumerate(joint_limits)
        )
    )
    return {
        "max_incremental_joint_delta_rad": max_range,
        "preferred_total_joint_delta_rad": max_from_home,
        "max_total_joint_delta_rad": max_range,
        "max_total_joint_delta_overrun_rad": 0.0,
    }


def _load_urdf_ik_joint_limits() -> dict[str, float]:
    if not R1LITE_DEPLOYMENT_URDF.exists():
        return {
            "max_incremental_joint_delta_rad": 0.18,
            "preferred_total_joint_delta_rad": 1.85,
            "max_total_joint_delta_rad": 3.0,
            "max_total_joint_delta_overrun_rad": 0.0,
        }
    return derive_r1lite_ik_joint_limits_from_urdf()


_URDF_IK_JOINT_LIMITS = _load_urdf_ik_joint_limits()


@dataclass(frozen=True)
class R1LiteIkHarness:
    """Motion-planning thresholds validated for the R1 Lite adaptation."""

    max_cartesian_step_m: float = 0.0125
    max_orientation_step_rad: float = 0.03
    max_incremental_joint_delta_rad: float = (
        _URDF_IK_JOINT_LIMITS["max_incremental_joint_delta_rad"]
    )
    preferred_total_joint_delta_rad: float = (
        _URDF_IK_JOINT_LIMITS["preferred_total_joint_delta_rad"]
    )
    max_total_joint_delta_rad: float = (
        _URDF_IK_JOINT_LIMITS["max_total_joint_delta_rad"]
    )
    max_total_joint_delta_overrun_rad: float = (
        _URDF_IK_JOINT_LIMITS["max_total_joint_delta_overrun_rad"]
    )
    max_fk_position_error_m: float = 0.10
    max_fk_orientation_error_rad: float = 0.30
    max_backward_step_m: float = 0.005
    max_forward_progress_error_m: float = 0.03

    def validate_waypoint(
        self,
        waypoint: dict[str, Any],
        *,
        check_cartesian_error: bool = True,
    ) -> None:
        if (
            waypoint["incremental_joint_delta_rad"]
            > self.max_incremental_joint_delta_rad
        ):
            raise RuntimeError(
                "IK waypoint has a discontinuous joint jump: "
                f"limit={self.max_incremental_joint_delta_rad:.3f}, "
                f"waypoint={waypoint}"
            )
        total_joint_overrun = (
            waypoint["total_joint_delta_rad"] - self.max_total_joint_delta_rad
        )
        if total_joint_overrun >= self.max_total_joint_delta_overrun_rad:
            raise RuntimeError(
                "IK path exceeds total joint displacement limit: "
                f"limit={self.max_total_joint_delta_rad:.3f}, "
                f"allowed_overrun<{self.max_total_joint_delta_overrun_rad:.3f}, "
                f"actual_overrun={total_joint_overrun:.3f}, waypoint={waypoint}"
            )
        if (
            check_cartesian_error
            and waypoint["fk_position_error_m"] > self.max_fk_position_error_m
        ):
            raise RuntimeError(
                "RelaxedIK FK does not reproduce the base_link waypoint: "
                f"limit={self.max_fk_position_error_m:.3f}m, waypoint={waypoint}"
            )
        if (
            check_cartesian_error
            and waypoint["fk_orientation_error_rad"]
            > self.max_fk_orientation_error_rad
        ):
            raise RuntimeError(
                "RelaxedIK FK orientation does not reproduce the waypoint: "
                f"limit={self.max_fk_orientation_error_rad:.3f}rad, "
                f"waypoint={waypoint}"
            )
        if (
            check_cartesian_error
            and waypoint["forward_step_m"] < -self.max_backward_step_m
        ):
            raise RuntimeError(
                "RelaxedIK FK moves opposite the Cartesian path: "
                f"limit={-self.max_backward_step_m:.3f}m, waypoint={waypoint}"
            )
        progress_error = abs(
            waypoint["actual_forward_progress_m"]
            - waypoint["expected_forward_progress_m"]
        )
        if (
            check_cartesian_error
            and progress_error > self.max_forward_progress_error_m
        ):
            raise RuntimeError(
                "RelaxedIK FK forward progress disagrees with the path: "
                f"limit={self.max_forward_progress_error_m:.3f}m, "
                f"waypoint={waypoint}"
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


R1_LITE_IK_HARNESS = R1LiteIkHarness()

_RIGHT_GRIPPER_LINK_FROM_TOOL_Z = np.eye(4, dtype=np.float64)
_RIGHT_GRIPPER_LINK_FROM_TOOL_Z[:3, :3] = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _native_gripper_pose_to_tool_z(base_from_native: np.ndarray) -> np.ndarray:
    return (
        _validate_rigid_transform(base_from_native, "base_from_native_gripper")
        @ _RIGHT_GRIPPER_LINK_FROM_TOOL_Z
    )


def _grasp_axis(rotation: np.ndarray, axis_name: str) -> np.ndarray:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    name = axis_name[1:] if axis_name.startswith("-") else axis_name
    if name not in {"x", "y", "z"}:
        raise ValueError("graspnet_approach_axis must be one of x, y, z, -x, -y, -z")
    axis = sign * np.asarray(rotation, dtype=np.float64).reshape(3, 3)[
        :, {"x": 0, "y": 1, "z": 2}[name]
    ]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("GraspNet approach axis has zero length")
    return axis / norm


def _transform_graspnet_pose(
    camera_from_grasp: np.ndarray,
    base_from_camera: np.ndarray,
    *,
    ee_from_grasp: np.ndarray,
) -> np.ndarray:
    """Convert a Contact-GraspNet camera-frame pose to an R1 tool target."""
    raw = _validate_rigid_transform(camera_from_grasp, "camera_from_grasp")
    base_camera = _validate_rigid_transform(base_from_camera, "base_from_camera")
    ee_grasp = _validate_rigid_transform(ee_from_grasp, "ee_from_grasp")
    transformed = base_camera @ raw @ np.linalg.inv(ee_grasp)
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed GraspNet pose is non-finite")
    return transformed


def _validate_rigid_transform(matrix: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} is non-finite")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not +1")
    return transform


def _matrix_to_pose_xyzw(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    position = transform[:3, 3].copy()
    quaternion = SciRotation.from_matrix(transform[:3, :3]).as_quat()
    return position, quaternion


def _dynamic_wrist_observation_transform(
    ball_position_base: np.ndarray,
    *,
    ee_from_camera: np.ndarray,
    reference_ee_position: np.ndarray,
    reference_ee_quaternion_xyzw: np.ndarray,
    distance_m: float,
    max_translation_m: float,
    camera_roll_rad: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the calibrated safe camera orientation and translate it onto the ball."""
    ball = np.asarray(ball_position_base, dtype=np.float64).reshape(3)
    reference_position = np.asarray(
        reference_ee_position,
        dtype=np.float64,
    ).reshape(3)
    reference_quaternion = np.asarray(
        reference_ee_quaternion_xyzw,
        dtype=np.float64,
    ).reshape(4)
    if not np.isfinite(ball).all():
        raise ValueError("Head tennis-ball position is non-finite")
    reference_quaternion /= np.linalg.norm(reference_quaternion)
    base_from_reference_ee = np.eye(4, dtype=np.float64)
    base_from_reference_ee[:3, :3] = SciRotation.from_quat(
        reference_quaternion
    ).as_matrix()
    base_from_reference_ee[:3, 3] = reference_position
    ee_camera = _validate_rigid_transform(ee_from_camera, "ee_from_camera")
    reference_camera = base_from_reference_ee @ ee_camera
    optical_z = reference_camera[:3, 2]
    optical_z /= np.linalg.norm(optical_z)
    base_from_camera = reference_camera.copy()
    base_from_camera[:3, :3] = (
        reference_camera[:3, :3]
        @ SciRotation.from_rotvec(
            np.array([0.0, 0.0, float(camera_roll_rad)])
        ).as_matrix()
    )
    base_from_camera[:3, 3] = ball - float(distance_m) * optical_z
    base_from_ee = base_from_camera @ np.linalg.inv(ee_camera)
    translation = float(
        np.linalg.norm(base_from_ee[:3, 3] - reference_position)
    )
    if translation > float(max_translation_m):
        raise ValueError(
            f"dynamic wrist translation {translation:.3f}m exceeds "
            f"{float(max_translation_m):.3f}m"
        )
    line_error = float(
        np.linalg.norm(
            np.cross(
                ball - base_from_camera[:3, 3],
                optical_z,
            )
        )
    )
    return base_from_ee, {
        "ball_position_base": ball.tolist(),
        "camera_position_base": base_from_camera[:3, 3].tolist(),
        "camera_optical_z_base": optical_z.tolist(),
        "distance_m": float(distance_m),
        "camera_roll_deg": float(np.degrees(camera_roll_rad)),
        "reference_translation_m": translation,
        "optical_axis_error_m": line_error,
    }


class R1LiteControlApi(ApiBase):
    """Right-arm R1 Lite helpers exposed to generated code.

    The API deliberately avoids direct end-effector pose commands.  The current
    R1 Lite controller was validated in joint-space, while direct target-pose
    publishing caused unexpected large motion.
    """

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        if not isinstance(env, R1LiteRealLowLevel):
            raise TypeError(
                "R1LiteControlApi requires R1LiteRealLowLevel; "
                f"got {type(env).__name__}"
            )
        self._r1: R1LiteRealLowLevel = env
        self._last_object_detection: dict[str, Any] | None = None
        self._last_object_pose: np.ndarray | None = None
        self._active_min_ee_z_m: float | None = None

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_observation": self.get_observation,
            "get_joint_state": self.get_joint_state,
            "return_home": self.return_home,
            "move_joints_safe": self.move_joints_safe,
            "move_joints_adaptive": self.move_joints_adaptive,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "command_gripper": self.command_gripper,
            "get_head_rgb": self.get_head_rgb,
            "get_head_rgbd": self.get_head_rgbd,
            "save_head_rgb": self.save_head_rgb,
            "save_head_stereo_rgb": self.save_head_stereo_rgb,
            "get_wrist_rgbd": self.get_wrist_rgbd,
            "save_wrist_rgb": self.save_wrist_rgb,
            "check_wrist_tennis": self.check_wrist_tennis,
            "locate_object_head": self.locate_object_head,
            "locate_colored_cups": self.locate_colored_cups,
            "locate_colored_cups_rgbd": self.locate_colored_cups_rgbd,
            "get_object_pose": self.get_object_pose,
            "get_tennis_position_head_sam3": self.get_tennis_position_head_sam3,
            "locate_put_objects_head_rgbd": (
                self.locate_put_objects_head_rgbd
            ),
            "plan_wrist_observation_pose": self.plan_wrist_observation_pose,
            "execute_wrist_observation_plan": self.execute_wrist_observation_plan,
            "refine_tennis_position_wrist": self.refine_tennis_position_wrist,
            "plan_pose_waypoints": self.plan_pose_waypoints,
            "goto_pose": self.goto_pose,
            "descend_z_in_steps": self.descend_z_in_steps,
            "verify_grasp": self.verify_grasp,
            "verify_tennis_in_box": self.verify_tennis_in_box,
            "verify_cup_stack": self.verify_cup_stack,
            "locate_pen_head": self.locate_pen_head,
            "pixel_to_base_on_plane": self.pixel_to_base_on_plane,
            "plan_pen_pregrasp": self.plan_pen_pregrasp,
            "plan_pen_graspnet_candidates": self.plan_pen_graspnet_candidates,
            "plan_tennis_graspnet": self.plan_tennis_graspnet,
            "execute_grasp_sequence": self.execute_grasp_sequence,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return current R1 Lite right-arm and right-gripper feedback.

        Returns:
            A dictionary with ``right_arm_joints`` as a 6D numpy array and
            ``right_gripper_position`` as a float.
        """
        return self._r1.get_observation()

    def get_joint_state(self) -> np.ndarray:
        """Return the current right-arm joint state.

        Returns:
            A numpy array of shape ``(6,)`` containing right-arm joint positions
            in radians.
        """
        return self._r1.get_arm_joints()

    def return_home(self) -> dict[str, Any]:
        """Return the right arm to its configured power-on joint position."""
        return self._r1.return_home()

    def move_joints_safe(
        self,
        target_joints: list[float] | tuple[float, ...] | np.ndarray,
        max_step: float = 0.03,
        duration_s: float = 3.0,
        expected_feedback: (
            list[float] | tuple[float, ...] | np.ndarray | None
        ) = None,
    ) -> np.ndarray:
        """Move the right arm to a joint target using small safe increments.

        Args:
            target_joints: Six right-arm joint targets in radians.
            max_step: Maximum per-joint interpolation step in radians. Keep this
                small; the default is approximately 1.7 degrees.
            duration_s: Seconds to keep publishing the final target.

        Returns:
            Latest right-arm joint feedback after the command.
        """
        return self._r1.move_joints_safe(
            target_joints,
            max_step=max_step,
            duration_s=duration_s,
            expected_feedback=expected_feedback,
        )

    def move_joints_adaptive(
        self,
        desired_feedback: list[float] | tuple[float, ...] | np.ndarray,
        waypoint_step_rad: float = 0.015,
        context: str = "api_adaptive_joint_path",
    ) -> dict[str, Any]:
        """Reach desired measured joints with online residual correction."""
        return self._r1.move_joint_waypoints_adaptive(
            desired_feedback,
            waypoint_step_rad=waypoint_step_rad,
            context=context,
        )

    def open_gripper(self, position: float | None = None) -> float:
        """Open the right gripper.

        Args:
            position: Optional open command. If omitted, uses the configured
                open value, currently around ``14.5`` for this R1 Lite.

        Returns:
            Latest gripper feedback after publishing.
        """
        if position is None:
            return self._r1.open_gripper()
        return self._r1.command_gripper(float(position))

    def close_gripper(self, position: float | None = None) -> float:
        """Close the right gripper.

        Args:
            position: Optional close command. If omitted, uses the configured
                close value, currently around ``5.5``. Use smaller values only
                after confirming they are safe for the object.

        Returns:
            Latest gripper feedback after publishing.
        """
        if position is None:
            return self._r1.close_gripper()
        return self._r1.command_gripper(float(position))

    def command_gripper(
        self,
        position: float,
        duration_s: float = 3.0,
        *,
        allow_contact: bool = False,
    ) -> float:
        """Command the right gripper to an explicit position.

        Args:
            position: Right gripper command value. Larger values open more.
            duration_s: Seconds to continuously publish the command.

        Returns:
            Latest gripper feedback after publishing.
        """
        return self._r1.command_gripper(
            float(position),
            duration_s=duration_s,
            allow_contact=bool(allow_contact),
        )

    def verify_grasp(
        self,
        min_gap_above_close: float = 0.35,
        min_gap_below_open: float = 10.0,
        save_debug: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        """Conservatively gate carrying motion after closing on a tennis ball.

        A retained ball must keep the R1 gripper feedback measurably above the
        configured fully-closed position. This rejects an empty fully closed
        gripper; it intentionally does not claim object identity from gripper
        feedback alone. A wrist frame is saved when available for review.
        """
        feedback = float(self._r1.get_gripper_position())
        closed = float(self._r1.gripper_close_position)
        empty_close = float(
            getattr(self._r1, "gripper_empty_close_feedback", closed)
        )
        threshold = empty_close + max(0.0, float(min_gap_above_close))
        open_position = float(self._r1.gripper_open_position)
        upper_threshold = open_position - max(0.0, float(min_gap_below_open))
        retained = threshold <= feedback <= upper_threshold
        validation_enabled = bool(
            getattr(self._r1, "r1lite_motion_validation_enabled", True)
        )
        success = bool(retained or not validation_enabled)
        details: dict[str, Any] = {
            "success": success,
            "reason": (
                "retained_gripper_gap"
                if retained
                else "gripper_validation_skipped"
                if not validation_enabled
                else (
                    "gripper_still_open"
                    if feedback > upper_threshold
                    else "empty_gripper_gap"
                )
            ),
            "retained_gripper_gap": bool(retained),
            "validation_skipped": not validation_enabled,
            "gripper_feedback": feedback,
            "closed_position": closed,
            "empty_close_feedback": empty_close,
            "threshold": threshold,
            "open_position": open_position,
            "upper_threshold": upper_threshold,
        }
        if save_debug:
            try:
                details["wrist_debug_image"] = self.save_wrist_rgb(
                    "outputs/r1lite/grasp_gate_wrist_rgb.png"
                )
            except Exception as exc:
                details["wrist_debug_error"] = repr(exc)
        return success, details

    def get_wrist_rgbd(self, require_depth: bool = True) -> dict[str, np.ndarray | None]:
        """Return the latest right-wrist RGB-D camera observation.

        Args:
            require_depth: If true, wait until a depth frame has been received.

        Returns:
            A dictionary with:
            - ``rgb``: uint8 RGB image, shape ``(H, W, 3)``.
            - ``depth``: float32 depth image in meters, shape ``(H, W)`` if available.
            - ``intrinsics``: 3x3 camera matrix if camera info has arrived.
        """
        return self._r1.get_wrist_rgbd(require_depth=require_depth)

    def get_head_rgbd(
        self,
        camera: str = "left",
        require_depth: bool = True,
    ) -> dict[str, np.ndarray | None]:
        """Return the latest rectified head stereo RGB-D observation."""
        return self._r1.get_head_rgbd(
            camera=camera,
            require_depth=require_depth,
        )

    def get_head_rgb(self, camera: str = "left") -> np.ndarray:
        """Return the latest head camera RGB image.

        Args:
            camera: ``"left"`` or ``"right"``. The left head camera is used as
                the Cap-X primary visual feedback image by default.

        Returns:
            uint8 RGB image, shape ``(H, W, 3)``.
        """
        return self._r1.get_head_rgb(camera=camera)

    def save_head_rgb(
        self,
        path: str = "outputs/r1lite/latest_head_left_rgb.png",
        camera: str = "left",
    ) -> str:
        """Save the latest head camera RGB image for visual inspection.

        Args:
            path: Output PNG path.
            camera: ``"left"`` or ``"right"``.

        Returns:
            The saved path.
        """
        rgb = self._r1.get_head_rgb(camera=camera)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out)
        return str(out)

    def save_head_stereo_rgb(
        self,
        path: str = "outputs/r1lite/latest_head_stereo_left_rgb.png",
    ) -> str:
        """Save the rectified head-left RGB frame used with stereo depth."""
        obs = self._r1.get_head_rgbd(camera="left", require_depth=False)
        rgb = obs["rgb"]
        if rgb is None:
            raise RuntimeError("No rectified head-left RGB frame is available yet")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out)
        return str(out)

    def save_wrist_rgb(self, path: str = "outputs/r1lite/latest_wrist_rgb.png") -> str:
        """Save the latest right-wrist RGB image for visual inspection.

        Args:
            path: Output PNG path.

        Returns:
            The saved path.
        """
        obs = self._r1.get_wrist_rgbd(require_depth=False)
        rgb = obs["rgb"]
        if rgb is None:
            raise RuntimeError("No wrist RGB frame is available yet")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out)
        return str(out)

    def check_wrist_tennis(
        self,
        *,
        label: str = "wrist_tennis",
        min_pixels: int = 30,
        max_age_s: float = 1.0,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate a fresh wrist frame and look for the yellow tennis ball."""
        obs = self._r1.get_wrist_rgbd(require_depth=False)
        rgb = obs.get("rgb")
        if rgb is None:
            return False, {"success": False, "reason": "missing_wrist_rgb"}
        rgb_arr = np.asarray(rgb, dtype=np.uint8)
        age_s = float(self._r1.get_wrist_rgb_age_s())
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3 or rgb_arr.size == 0:
            return False, {
                "success": False,
                "reason": "invalid_wrist_rgb",
                "shape": list(rgb_arr.shape),
            }
        mask = self._yellow_tennis_ball_mask(rgb_arr[:, :, :3])
        pixel_count = int(np.count_nonzero(mask))
        centroid, component_pixels = self._best_blob_centroid(
            mask,
            min_pixels=max(1, int(min_pixels)),
            max_pixels=int(mask.size),
            prefer_largest=True,
        )
        out = Path(f"outputs/r1lite/{label}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_arr[:, :, :3]).save(out)
        success = (
            age_s <= float(max_age_s)
            and centroid is not None
            and component_pixels >= int(min_pixels)
        )
        return success, {
            "success": success,
            "reason": (
                "wrist_tennis_visible"
                if success
                else "stale_wrist_rgb"
                if age_s > float(max_age_s)
                else "tennis_not_visible_in_wrist"
            ),
            "age_s": age_s,
            "shape": list(rgb_arr.shape),
            "mask_pixels": pixel_count,
            "component_pixels": int(component_pixels),
            "centroid": None if centroid is None else centroid.tolist(),
            "image": str(out),
        }

    @staticmethod
    def _default_tool_down_quat_xyzw() -> tuple[float, float, float, float]:
        """Gripper-down orientation used for top-down pick-place on R1 Lite."""
        # In the tool-z frame, local +Z is the approach axis.  A true
        # top-down grasp must therefore map tool +Z to base -Z.
        return (1.0, 0.0, 0.0, 0.0)

    def _object_z_offset(self, object_name: str) -> float:
        key = str(object_name).strip().lower()
        if "tennis" in key and "ball" in key:
            return float(getattr(self._r1, "tennis_ball_radius_m", 0.034))
        if "box" in key:
            return 0.05
        return 0.0

    def _sam3_available(self, timeout_s: float = 0.5) -> bool:
        url = getattr(self._r1, "sam3_service_url", "http://127.0.0.1:8114")
        try:
            from urllib.parse import urlparse

            parsed = urlparse(str(url))
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 8114
            import socket

            with socket.create_connection((host, int(port)), timeout=timeout_s):
                return True
        except Exception:
            return False

    def locate_object_head(
        self,
        object_name: str,
        *,
        text_prompt: str | None = None,
        use_sam3: bool = True,
        sam3_timeout_s: float = 30.0,
        min_score: float = 0.15,
        min_pixels: int = 40,
        save_debug: bool = True,
    ) -> dict[str, Any]:
        """Locate an object candidate in the head-left camera image.

        Args:
            object_name: Canonical object name, for example ``"tennis ball"``.
            text_prompt: Optional SAM3 prompt override.
            use_sam3: Try SAM3 text-prompt segmentation first.
            sam3_timeout_s: HTTP timeout for local SAM3 segmentation.
            min_score: Minimum SAM3 mask score.
            min_pixels: Minimum mask pixel count.
            save_debug: Save debug images when true.

        Returns:
            Dictionary with ``ok``, ``pixel``, ``method``, and optional debug paths.
        """
        rgb = self._r1.get_head_rgb("left")
        slug = str(object_name).strip().lower().replace(" ", "_")
        result: dict[str, Any] = {"object_name": object_name}
        if save_debug:
            result["debug_image"] = self.save_head_rgb(
                f"outputs/r1lite/locate_{slug}_head.png",
                camera="left",
            )
        prompt = text_prompt or self._r1.object_prompt(object_name)
        if use_sam3 and not self._sam3_available():
            return {
                **result,
                "ok": False,
                "method": "sam3",
                "reason": "SAM3 service unavailable",
            }
        if use_sam3:
            sam3 = self._locate_object_head_sam3(
                rgb,
                prompt=prompt,
                timeout_s=sam3_timeout_s,
                min_score=min_score,
                min_pixels=min_pixels,
                save_debug=save_debug,
                slug=slug,
            )
            if sam3["ok"]:
                sam3.update(result)
                return sam3
            result["sam3_reason"] = sam3.get("reason", "SAM3 did not return a valid mask")
            return {
                **result,
                "ok": False,
                "method": "sam3",
                "reason": result["sam3_reason"],
            }

        if "pen" in slug:
            mask = self._blue_pen_mask(rgb)
            method = "color_fallback"
        elif "tennis" in slug or "ball" in slug:
            mask = self._yellow_tennis_ball_mask(rgb)
            method = "color_fallback"
        elif "box" in slug:
            mask = self._cardboard_box_mask(rgb)
            method = "color_fallback"
        else:
            return {
                **result,
                "ok": False,
                "method": "sam3_only",
                "reason": result.get("sam3_reason", f"SAM3 failed for {object_name!r}"),
            }

        ys, xs = np.nonzero(mask)
        result.update(
            {
                "ok": False,
                "method": method,
                "count": int(len(xs)),
                "pixel": None,
            }
        )
        if save_debug:
            self._save_mask_debug(mask, f"outputs/r1lite/locate_{slug}_color_mask.png")
        if "box" in slug:
            h, w = mask.shape
            mask = mask.copy()
            # The box may be placed in the lower half of the tabletop.  The old
            # upper-image-only ROI rejected valid boxes in the normal right-arm
            # workspace, including the current tennis task scene.
            yy, xx = np.mgrid[0:h, 0:w]
            mask &= (
                (xx > int(w * 0.40))
                & (xx < int(w * 0.85))
                & (yy > int(h * 0.15))
                & (yy < int(h * 0.70))
            )
            blob_min_pixels = max(min_pixels, 800)
            blob_max_pixels = max(120000, int(mask.size * 0.12))
            effective_min_pixels = blob_min_pixels
            prefer_lowest = False
        elif "tennis" in slug or "ball" in slug:
            h, w = mask.shape
            mask = mask.copy()
            yy, xx = np.mgrid[0:h, 0:w]
            roi = (
                (xx > int(0.20 * w))
                & (xx < int(0.92 * w))
                & (yy > int(0.25 * h))
                & (yy < int(0.92 * h))
            )
            mask &= roi
            blob_min_pixels = max(20, min_pixels // 2)
            # Head images are 1920x1536 on the current R1 Lite.  A nearby
            # tennis ball spans roughly 30k pixels, so a fixed 8k cap rejects
            # the correct connected component.
            blob_max_pixels = max(8000, int(mask.size * 0.03))
            effective_min_pixels = blob_min_pixels
            prefer_lowest = False
        else:
            blob_min_pixels = min_pixels
            blob_max_pixels = 250000
            effective_min_pixels = min_pixels
            prefer_lowest = False
        pixel, count = self._best_blob_centroid(
            mask,
            min_pixels=blob_min_pixels,
            max_pixels=blob_max_pixels,
            prefer_largest=(
                "box" in slug or "tennis" in slug or "ball" in slug
            ),
            prefer_lowest=prefer_lowest,
        )
        result["count"] = count
        if pixel is None or count < effective_min_pixels:
            result["reason"] = f"mask pixels below threshold: {count} < {effective_min_pixels}"
            return result
        result.update(
            {
                "ok": True,
                "pixel": pixel.tolist(),
                "centroid_u": float(pixel[0]),
                "centroid_v": float(pixel[1]),
            }
        )
        return result

    def get_object_pose(
        self,
        object_name: str,
        *,
        text_prompt: str | None = None,
        save_debug: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate an object pose on the calibrated table plane.

        Args:
            object_name: Canonical object name, for example ``"tennis ball"``.
            text_prompt: Optional SAM3 prompt override.
            save_debug: Save debug images when true.

        Returns:
            ``(position_xyz, quaternion_xyzw)`` in the robot base frame.
        """
        detection = self.locate_object_head(
            object_name,
            text_prompt=text_prompt,
            save_debug=save_debug,
        )
        if not detection.get("ok"):
            raise RuntimeError(
                f"Could not locate {object_name!r}: {detection.get('reason', detection)}"
            )
        try:
            plane_point = self._r1.pixel_to_base_on_plane(detection["pixel"])
        except ValueError as exc:
            slug = str(object_name).strip().lower().replace(" ", "_")
            raise ValueError(
                f"{object_name!r} pixel {detection.get('pixel')} maps outside workspace: {exc}. "
                f"See outputs/r1lite/locate_{slug}_head.png"
            ) from exc
        position = plane_point.copy()
        position[2] += self._object_z_offset(object_name)
        # Preserve the measured near-tool-down wrist orientation.  The exact
        # analytic tool-down quaternion can push RelaxedIK just beyond the R1
        # Lite wrist limit and cause the Cartesian controller to reject the
        # whole target.
        try:
            _ee_position, quat = self._r1.get_planning_ee_pose()
        except Exception:
            quat = np.array(self._default_tool_down_quat_xyzw(), dtype=np.float64)
        self._last_object_detection = detection
        self._last_object_pose = position.copy()
        return position, np.asarray(quat, dtype=np.float64).reshape(4)

    def locate_colored_cups(
        self,
        *,
        save_debug: bool = True,
        require_separation: bool = True,
        minimum_pixel_separation: float = 25.0,
    ) -> dict[str, dict[str, Any]]:
        """Locate orange, purple, and green cups with semantic and HSV gates."""
        specifications = {
            "orange cup": (10.0, 48.0),
            "purple cup": (245.0, 335.0),
            "green cup": (65.0, 175.0),
        }
        detections: dict[str, dict[str, Any]] = {}
        for name, (minimum_hue, maximum_hue) in specifications.items():
            detection = self.locate_object_head(name, save_debug=save_debug)
            if not detection.get("ok"):
                raise RuntimeError(
                    f"Could not locate {name!r}: "
                    f"{detection.get('reason', detection)}"
                )
            hue = float(detection.get("median_hsv", [math.nan])[0])
            saturation = float(
                detection.get("median_hsv", [math.nan, math.nan])[1]
            )
            if (
                not np.isfinite(hue)
                or not np.isfinite(saturation)
                or not minimum_hue <= hue <= maximum_hue
                or saturation < 0.30
            ):
                raise RuntimeError(
                    f"{name} failed HSV identity gate: "
                    f"hue={hue:.1f}, saturation={saturation:.3f}"
                )
            position = self._r1.pixel_to_base_on_plane(detection["pixel"])
            detections[name] = {
                **detection,
                "position_base_m": np.asarray(position, dtype=float).tolist(),
                "expected_hue_range_deg": [minimum_hue, maximum_hue],
            }
        if require_separation:
            names = tuple(specifications)
            for index, first in enumerate(names):
                for second in names[index + 1 :]:
                    first_pixel = np.asarray(
                        detections[first]["pixel"], dtype=float
                    )
                    second_pixel = np.asarray(
                        detections[second]["pixel"], dtype=float
                    )
                    separation = float(np.linalg.norm(first_pixel - second_pixel))
                    if separation < float(minimum_pixel_separation):
                        raise RuntimeError(
                            "Colored-cup detections are not unique: "
                            f"{first} and {second} are {separation:.1f}px apart"
                        )
        return detections

    def locate_colored_cups_rgbd(
        self,
        *,
        cup_height_m: float,
        object_names: tuple[str, ...] | list[str] | None = None,
        save_debug: bool = True,
        require_separation: bool = True,
        minimum_xy_separation_m: float = 0.025,
    ) -> dict[str, dict[str, Any]]:
        """Recover colored inverted-cup poses from one synchronized head RGB-D frame."""
        specifications = {
            "orange cup": (10.0, 48.0),
            "purple cup": (245.0, 335.0),
            "green cup": (65.0, 175.0),
        }
        requested = tuple(object_names or specifications)
        unknown = set(requested) - set(specifications)
        if unknown:
            raise ValueError(f"unsupported colored cups: {sorted(unknown)}")
        observation = self._r1.get_head_rgbd_median(
            camera="left",
        )
        rgb = np.asarray(observation["rgb"], dtype=np.uint8)
        depth = np.asarray(observation["depth"], dtype=float)
        intrinsics = np.asarray(
            observation["intrinsics"], dtype=float
        ).reshape(3, 3)
        frame_id = str(observation.get("frame_id") or "")
        stamp_ns = int(observation.get("stamp_ns") or 0)
        if not frame_id or stamp_ns <= 0:
            raise RuntimeError("head RGB-D frame is missing frame_id or timestamp")
        # The RGB-D image is rectified against the fixed head-left plane
        # calibration. Do not alternate between that calibrated optical pose
        # and an intermittently discoverable ROS frame with different axes.
        base_from_camera = (
            self._r1._head_left_camera_to_base_from_calibration()
        )
        table_z = float(self._r1.load_head_plane_calibration()["plane_z"])
        base_from_camera, table_leveling = self._level_head_rgbd_to_table(
            rgb,
            depth,
            intrinsics,
            base_from_camera,
            table_z=table_z,
            save_debug=save_debug,
        )
        detections: dict[str, dict[str, Any]] = {}
        for name in requested:
            minimum_hue, maximum_hue = specifications[name]
            slug = name.replace(" ", "_")
            segmentation = self._locate_object_head_sam3(
                rgb,
                prompt=self._r1.object_prompt(name),
                timeout_s=30.0,
                min_score=0.15,
                min_pixels=40,
                save_debug=save_debug,
                slug=f"{slug}_rgbd",
                include_mask=True,
            )
            if not segmentation.get("ok"):
                raise RuntimeError(
                    f"RGB-D SAM3 failed for {name}: "
                    f"{segmentation.get('reason', segmentation)}"
                )
            hue, saturation, _value = np.asarray(
                segmentation["median_hsv"], dtype=float
            ).reshape(3)
            if (
                not minimum_hue <= hue <= maximum_hue
                or saturation < 0.30
            ):
                raise RuntimeError(
                    f"{name} failed RGB-D HSV identity gate: "
                    f"hue={hue:.1f}, saturation={saturation:.3f}"
                )
            mask = self._resize_mask(
                np.asarray(segmentation.pop("mask"), dtype=bool),
                depth.shape,
            )
            pose = reconstruct_inverted_cup_pose(
                depth,
                intrinsics,
                base_from_camera,
                mask,
                cup_height_m=float(cup_height_m),
            )
            detections[name] = {
                **segmentation,
                **pose,
                "position_base_m": pose["body_center_base_m"],
                "expected_hue_range_deg": [minimum_hue, maximum_hue],
                "rgbd_frame_id": frame_id,
                "rgbd_stamp_ns": stamp_ns,
                "depth_sample_count": int(
                    observation.get("depth_sample_count", 1)
                ),
                "table_leveling": table_leveling,
            }
        if require_separation:
            for index, first in enumerate(requested):
                for second in requested[index + 1 :]:
                    first_xy = np.asarray(
                        detections[first]["body_center_base_m"], dtype=float
                    )[:2]
                    second_xy = np.asarray(
                        detections[second]["body_center_base_m"], dtype=float
                    )[:2]
                    separation = float(np.linalg.norm(first_xy - second_xy))
                    if separation < float(minimum_xy_separation_m):
                        raise RuntimeError(
                            "RGB-D colored-cup detections are not unique: "
                            f"{first} and {second} are {separation:.4f}m apart"
                        )
        return detections

    def get_tennis_position_head_sam3(
        self,
        *,
        save_debug: bool = True,
    ) -> dict[str, Any]:
        """Estimate the tennis-ball center from SAM3 and head stereo depth."""
        obs = self._r1.get_head_rgbd(camera="left", require_depth=True)
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        depth = np.asarray(obs["depth"], dtype=np.float64)
        intrinsics = np.asarray(obs["intrinsics"], dtype=np.float64).reshape(3, 3)
        frame_id = str(obs.get("frame_id") or "")
        stamp_ns = int(obs.get("stamp_ns") or 0)
        segmentation = self._segment_tennis_mask(
            rgb,
            prompt=self._r1.object_prompt("tennis ball"),
            use_sam3=True,
            save_debug=save_debug,
            camera_name="head_stereo_position",
        )
        if not segmentation.get("ok"):
            raise RuntimeError(
                "SAM3 head tennis segmentation failed: "
                f"{segmentation.get('reason', segmentation)}"
            )
        mask = self._resize_mask(
            np.asarray(segmentation["mask"], dtype=bool),
            depth.shape,
        )
        ys, xs = np.nonzero(mask)
        if len(xs) < int(getattr(self._r1, "graspnet_min_segment_points", 30)):
            raise RuntimeError(
                f"SAM3 tennis mask has too few pixels after resize: {len(xs)}"
            )
        centroid = np.array([float(np.median(xs)), float(np.median(ys))])
        radial = np.hypot(xs - centroid[0], ys - centroid[1])
        central_limit = float(np.percentile(radial, 35.0))
        central = radial <= max(central_limit, 2.0)
        central_x = xs[central]
        central_y = ys[central]
        central_depth = depth[central_y, central_x]
        near, far = getattr(self._r1, "graspnet_depth_range", (0.2, 2.0))
        valid = (
            np.isfinite(central_depth)
            & (central_depth >= float(near))
            & (central_depth <= float(far))
        )
        if int(np.count_nonzero(valid)) < 10:
            raise RuntimeError(
                "SAM3 tennis center has insufficient valid stereo depth: "
                f"{int(np.count_nonzero(valid))}"
            )
        surface_depth = float(np.median(central_depth[valid]))
        surface_camera = np.array(
            [
                (centroid[0] - intrinsics[0, 2])
                * surface_depth
                / intrinsics[0, 0],
                (centroid[1] - intrinsics[1, 2])
                * surface_depth
                / intrinsics[1, 1],
                surface_depth,
            ],
            dtype=np.float64,
        )
        ray = surface_camera / np.linalg.norm(surface_camera)
        center_camera = surface_camera + ray * float(self._r1.tennis_ball_radius_m)
        base_from_camera = self._r1.lookup_camera_to_base(
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            camera_label="head_stereo_left",
        )
        depth_center_base = (
            base_from_camera[:3, :3] @ center_camera
            + base_from_camera[:3, 3]
        )
        center_base = depth_center_base.copy()
        z_offset = float(getattr(self._r1, "head_stereo_tennis_z_offset_m", 0.0))
        center_base[2] += z_offset
        self._r1.validate_workspace_point(center_base)
        result = {
            "ok": True,
            "method": "sam3_head_stereo_depth",
            "position_base": center_base.tolist(),
            "position_base_before_z_offset": depth_center_base.tolist(),
            "head_stereo_tennis_z_offset_m": z_offset,
            "surface_position_camera": surface_camera.tolist(),
            "center_position_camera": center_camera.tolist(),
            "centroid_px": centroid.tolist(),
            "surface_depth_m": surface_depth,
            "valid_center_depth_pixels": int(np.count_nonzero(valid)),
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
            "segmentation": self._strip_mask(segmentation),
        }
        self._last_object_detection = result
        self._last_object_pose = center_base.copy()
        return result

    def locate_put_objects_head_rgbd(
        self,
        *,
        object_names: tuple[str, ...] | list[str] = (
            "tennis ball",
            "peach",
            "apple",
            "cardboard box",
        ),
        save_debug: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Locate put-task objects from one synchronized head RGB-D frame."""
        supported = {
            "tennis ball": "yellow tennis ball on the table",
            "peach": "red fruit",
            "apple": "green apple",
            "cardboard box": "open cardboard box on the table",
        }
        requested = tuple(str(name).strip().lower() for name in object_names)
        if len(requested) != len(set(requested)):
            raise ValueError("put-task object names must be unique")
        unknown = set(requested) - set(supported)
        if unknown:
            raise ValueError(f"unsupported put-task objects: {sorted(unknown)}")
        observation = self._r1.get_head_rgbd_median(
            camera="left",
        )
        rgb = np.asarray(observation["rgb"], dtype=np.uint8)
        depth = np.asarray(observation["depth"], dtype=np.float64)
        intrinsics = np.asarray(
            observation["intrinsics"], dtype=np.float64
        ).reshape(3, 3)
        frame_id = str(observation.get("frame_id") or "")
        stamp_ns = int(observation.get("stamp_ns") or 0)
        if not frame_id or stamp_ns <= 0:
            raise RuntimeError("head RGB-D frame is missing frame_id or timestamp")
        base_from_camera = (
            self._r1._head_left_camera_to_base_from_calibration()
        )
        table_z = float(
            self._r1.load_head_plane_calibration()["plane_z"]
        )
        base_from_camera, table_leveling = self._level_head_rgbd_to_table(
            rgb,
            depth,
            intrinsics,
            base_from_camera,
            table_z=table_z,
            save_debug=save_debug,
        )
        nominal_radii = {
            "tennis ball": 0.034,
            "peach": 0.040,
            "apple": 0.040,
        }
        detections: dict[str, dict[str, Any]] = {}
        for name in requested:
            slug = name.replace(" ", "_")
            segmentation = self._locate_object_head_sam3(
                rgb,
                prompt=self._r1.object_prompt(name)
                if name in self._r1.object_prompts
                else supported[name],
                timeout_s=30.0,
                min_score=0.15,
                min_pixels=80,
                save_debug=save_debug,
                slug=f"put_general_{slug}",
                include_mask=True,
            )
            if not segmentation.get("ok"):
                raise RuntimeError(
                    f"put-task RGB-D SAM3 failed for {name}: "
                    f"{segmentation.get('reason', segmentation)}"
                )
            mask = self._resize_mask(
                np.asarray(segmentation.pop("mask"), dtype=bool),
                depth.shape,
            )
            _full, segment_camera = self._rgbd_point_clouds(
                depth,
                intrinsics,
                mask,
                camera_label=f"head_{slug}",
            )
            if len(segment_camera) < 30:
                raise RuntimeError(
                    f"{name} has insufficient valid RGB-D points"
                )
            segment_base = (
                base_from_camera[:3, :3] @ segment_camera.T
            ).T + base_from_camera[:3, 3]
            finite = np.isfinite(segment_base).all(axis=1)
            segment_base = segment_base[finite]
            if len(segment_base) < 30:
                raise RuntimeError(f"{name} has insufficient finite 3D points")
            plausible = segment_base[
                (segment_base[:, 2] >= table_z - 0.03)
                & (segment_base[:, 2] <= table_z + 0.30)
            ]
            if len(plausible) < 30:
                z_percentiles = np.percentile(
                    segment_base[:, 2], [1.0, 5.0, 50.0, 95.0, 99.0]
                ).tolist()
                raise RuntimeError(
                    f"{name} has insufficient points near the table: "
                    f"count={len(plausible)}, total={len(segment_base)}, "
                    f"z_percentiles_1_5_50_95_99={z_percentiles}, "
                    f"table_z={table_z:.6f}"
                )
            sample_indices = np.linspace(
                0,
                len(plausible) - 1,
                min(len(plausible), 1500),
            ).astype(int)
            sampled_points = plausible[sample_indices]
            center_xy = np.median(plausible[:, :2], axis=0)
            if name == "cardboard box":
                outer_min = np.percentile(
                    plausible[:, :2], 5.0, axis=0
                )
                outer_max = np.percentile(
                    plausible[:, :2], 95.0, axis=0
                )
                inset = 0.025
                opening_min = outer_min + inset
                opening_max = outer_max - inset
                if np.any(opening_max - opening_min < 0.06):
                    raise RuntimeError(
                        "detected box opening is too small after safety inset"
                    )
                elevated = segment_base[
                    (segment_base[:, 2] >= table_z - 0.02)
                    & (segment_base[:, 2] <= table_z + 0.30)
                ]
                measured_rim_z = (
                    float(np.percentile(elevated[:, 2], 85.0))
                    if len(elevated) >= 30
                    else table_z + 0.08
                )
                rim_z = float(
                    np.clip(measured_rim_z, table_z + 0.03, table_z + 0.25)
                )
                detections[name] = {
                    **segmentation,
                    "position_base_m": [
                        float(center_xy[0]),
                        float(center_xy[1]),
                        rim_z,
                    ],
                    "opening_xy_min_base_m": opening_min.tolist(),
                    "opening_xy_max_base_m": opening_max.tolist(),
                    "outer_xy_min_base_m": outer_min.tolist(),
                    "outer_xy_max_base_m": outer_max.tolist(),
                    "rim_z_base_m": rim_z,
                    "safety_inset_m": inset,
                    "point_cloud_base_m": sampled_points.tolist(),
                    "depth_sample_count": int(
                        observation.get("depth_sample_count", 1)
                    ),
                    "table_leveling": table_leveling,
                }
                continue
            elevated = segment_base[
                (segment_base[:, 2] >= table_z + 0.005)
                & (segment_base[:, 2] <= table_z + 0.16)
            ]
            measured_top_z = (
                float(np.percentile(elevated[:, 2], 90.0))
                if len(elevated) >= 30
                else table_z + 2.0 * nominal_radii[name]
            )
            measured_radius = 0.5 * (measured_top_z - table_z)
            radius = float(
                np.clip(
                    measured_radius,
                    max(0.025, nominal_radii[name] - 0.012),
                    min(0.055, nominal_radii[name] + 0.012),
                )
            )
            center = np.array(
                [center_xy[0], center_xy[1], table_z + radius],
                dtype=float,
            )
            self._r1.validate_workspace_point(center)
            detections[name] = {
                **segmentation,
                "position_base_m": center.tolist(),
                "estimated_radius_m": radius,
                "measured_top_z_base_m": measured_top_z,
                "table_z_base_m": table_z,
                "point_cloud_base_m": sampled_points.tolist(),
                "depth_sample_count": int(
                    observation.get("depth_sample_count", 1)
                ),
                "table_leveling": table_leveling,
            }
        object_centers = [
            (
                name,
                np.asarray(detections[name]["position_base_m"], dtype=float),
            )
            for name in requested
            if name != "cardboard box"
        ]
        for index, (first_name, first) in enumerate(object_centers):
            for second_name, second in object_centers[index + 1 :]:
                separation = float(np.linalg.norm(first[:2] - second[:2]))
                if separation < 0.06:
                    raise RuntimeError(
                        "put-task detections are not unique: "
                        f"{first_name} and {second_name} are "
                        f"{separation:.3f}m apart"
                    )
        return detections

    def plan_wrist_observation_pose(
        self,
        ball_position_base: list[float] | tuple[float, ...] | np.ndarray,
    ) -> dict[str, Any]:
        """Plan an isolated-IK path that centers the coarse head ball position."""
        current_position, _ = self._r1.get_planning_ee_pose()
        accepted = []
        rejected = []
        candidates = [
            (roll_deg, clearance_m)
            for roll_deg in self._r1.wrist_observation_roll_candidates_deg
            for clearance_m in (
                self._r1.wrist_observation_transit_clearance_candidates_m
            )
        ]
        for roll_deg, clearance_m in candidates:
            try:
                self._r1.reset_relaxed_ik_probe()
                target_transform, geometry = _dynamic_wrist_observation_transform(
                    np.asarray(ball_position_base, dtype=np.float64),
                    ee_from_camera=self._r1.load_wrist_hand_eye_calibration(),
                    reference_ee_position=(
                        self._r1.wrist_observation_reference_ee_position
                    ),
                    reference_ee_quaternion_xyzw=(
                        self._r1.wrist_observation_reference_ee_quaternion_xyzw
                    ),
                    distance_m=self._r1.wrist_observation_distance_m,
                    max_translation_m=(
                        self._r1.wrist_observation_max_translation_m
                    ),
                    camera_roll_rad=float(np.radians(roll_deg)),
                )
                target_transform = _native_gripper_pose_to_tool_z(target_transform)
                target_position, target_quaternion = _matrix_to_pose_xyzw(
                    target_transform
                )
                self._r1.validate_workspace_point(target_position)
                transit_z = min(
                    self._r1.workspace_z_limits[1] - 0.05,
                    max(
                        float(current_position[2]),
                        float(target_position[2]),
                    )
                    + float(clearance_m),
                )
                via_positions = [
                    [
                        float(current_position[0]),
                        float(current_position[1]),
                        transit_z,
                    ],
                    [
                        float(target_position[0]),
                        float(target_position[1]),
                        transit_z,
                    ],
                ]
                ik_plan = self.plan_pose_waypoints(
                    target_position,
                    target_quaternion,
                    max_cartesian_step_m=(
                        0.005
                        if getattr(
                            self._r1,
                            "r1lite_motion_validation_enabled",
                            True,
                        )
                        else 0.020
                    ),
                    max_orientation_step_rad=(
                        0.03
                        if getattr(
                            self._r1,
                            "r1lite_motion_validation_enabled",
                            True,
                        )
                        else 0.08
                    ),
                    max_incremental_joint_delta_rad=0.18,
                    max_total_joint_delta_rad=(
                        self._r1.wrist_observation_max_joint_delta_rad
                    ),
                    via_positions=via_positions,
                )
                max_total_delta = max(
                    waypoint["total_joint_delta_rad"]
                    for waypoint in ik_plan["waypoints"]
                )
                accepted.append({
                    "roll_deg": float(roll_deg),
                    "transit_clearance_m": float(clearance_m),
                    "target_ee_position": target_position.tolist(),
                    "target_ee_quaternion_xyzw": target_quaternion.tolist(),
                    "geometry": geometry,
                    "ik_plan": ik_plan,
                    "max_total_joint_delta_rad": float(max_total_delta),
                })
                if not getattr(
                    self._r1,
                    "r1lite_motion_validation_enabled",
                    True,
                ):
                    break
            except Exception as exc:
                rejected.append({
                    "roll_deg": float(roll_deg),
                    "transit_clearance_m": float(clearance_m),
                    "reason": str(exc),
                })
        if not accepted:
            raise RuntimeError(
                f"No continuous dynamic wrist observation IK candidate: {rejected}"
            )
        selected = min(
            accepted,
            key=lambda item: (
                item["max_total_joint_delta_rad"],
                abs(item["roll_deg"]),
            ),
        )
        return {
            "success": True,
            "mode": "bounded_camera_roll_dynamic_translation",
            **selected,
            "rejected_candidates": rejected,
        }

    def execute_wrist_observation_plan(
        self,
        observation_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a previously validated dynamic observation IK plan."""
        if observation_plan.get("mode") != "bounded_camera_roll_dynamic_translation":
            raise ValueError("Unrecognized wrist observation plan")
        ik_plan = observation_plan.get("ik_plan")
        if not isinstance(ik_plan, dict) or not ik_plan.get("waypoints"):
            raise ValueError("Wrist observation plan has no validated IK waypoints")
        execution = []
        validation_enabled = getattr(
            self._r1,
            "r1lite_motion_validation_enabled",
            True,
        )
        waypoints = list(ik_plan["waypoints"])
        if not validation_enabled:
            waypoint = waypoints[-1]
            result = self._r1.move_joints_feedforward(
                waypoint["joint_feedback"],
                duration_s=0.25,
                context=f"dynamic_wrist_observation_fast_{waypoint['index']:03d}",
            )
            execution.append({"index": waypoint["index"], "joint_result": result})
        else:
            for waypoint in waypoints:
                result = self._r1.move_joint_waypoints_adaptive(
                    waypoint["joint_feedback"],
                    waypoint_step_rad=0.020,
                    context=f"dynamic_wrist_observation_{waypoint['index']:03d}",
                    skip_feedback_settle=bool(
                        getattr(
                            self._r1,
                            "r1lite_wrist_observation_skip_feedback_settle",
                            False,
                        )
                    ),
                )
                execution.append(
                    {
                        "index": waypoint["index"],
                        "joint_result": result,
                    }
                )
        target = np.asarray(
            observation_plan["target_ee_position"],
            dtype=np.float64,
        )
        local_fk = getattr(self._r1, "get_local_planning_fk_pose", None)
        if callable(local_fk):
            measured_position, measured_quaternion = local_fk()
            measured_pose_source = "joint_feedback_local_urdf_fk"
        else:
            measured_position, measured_quaternion = (
                self._r1.get_planning_ee_pose()
            )
            measured_pose_source = "relaxed_ik_async_ee_topic"
        position_error = float(np.linalg.norm(measured_position - target))
        if (
            validation_enabled
            and position_error > self._r1.pose_position_tolerance_m
        ):
            raise RuntimeError(
                f"Dynamic wrist observation position error {position_error:.4f}m "
                f"exceeds {self._r1.pose_position_tolerance_m:.4f}m"
            )
        barrier_ns = self._r1.wait_for_arm_stable()
        return {
            "success": True,
            "execution": execution,
            "measured_position": measured_position.tolist(),
            "measured_pose_source": measured_pose_source,
            "measured_quaternion_xyzw": measured_quaternion.tolist(),
            "position_error_m": position_error,
            "post_motion_barrier_ns": barrier_ns,
        }

    def refine_tennis_position_wrist(
        self,
        expected_ball_position_base: (
            list[float] | tuple[float, ...] | np.ndarray
        ),
        *,
        newer_than_ns: int | None = None,
        save_debug: bool = True,
    ) -> dict[str, Any]:
        """Refine the ball center from fresh wrist RGB-D and cross-check the head pose."""
        expected = np.asarray(
            expected_ball_position_base,
            dtype=np.float64,
        ).reshape(3)
        obs = self._r1.get_wrist_rgbd(
            require_depth=True,
            newer_than_ns=newer_than_ns,
        )
        segmentation = self._segment_tennis_mask(
            np.asarray(obs["rgb"], dtype=np.uint8),
            prompt=self._r1.object_prompt("tennis ball"),
            use_sam3=True,
            save_debug=save_debug,
        )
        if not segmentation.get("ok"):
            raise RuntimeError(f"Wrist tennis refinement failed: {segmentation}")
        mask = np.asarray(segmentation["mask"], dtype=bool)
        depth = np.asarray(obs["depth"], dtype=np.float64)
        intrinsics = np.asarray(obs["intrinsics"], dtype=np.float64)
        valid = (
            mask
            & np.isfinite(depth)
            & (depth >= self._r1.graspnet_depth_range[0])
            & (depth <= self._r1.graspnet_depth_range[1])
        )
        ys, xs = np.nonzero(valid)
        if len(xs) < self._r1.graspnet_min_segment_points:
            raise RuntimeError(
                f"Wrist tennis has only {len(xs)} valid depth pixels"
            )
        h, w = depth.shape
        centroid = np.array([float(np.median(xs)), float(np.median(ys))])
        margin = self._r1.wrist_ball_center_margin_ratio
        center_margin_ok = (
            margin * w <= centroid[0] <= (1.0 - margin) * w
            and margin * h <= centroid[1] <= (1.0 - margin) * h
        )
        if not center_margin_ok and getattr(
            self._r1,
            "r1lite_motion_validation_enabled",
            True,
        ):
            raise RuntimeError(
                f"Wrist tennis centroid {centroid.tolist()} is outside center margin"
            )
        z = float(np.median(depth[ys, xs]))
        u, v = centroid
        camera_point = np.array(
            [
                (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (v - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
                1.0,
            ],
            dtype=np.float64,
        )
        base_from_camera = self._r1.lookup_wrist_camera_to_base(
            frame_id=str(obs["frame_id"]),
            stamp_ns=int(obs["stamp_ns"]),
        )
        surface_point = (base_from_camera @ camera_point)[:3]
        camera_ray = camera_point[:3]
        ray_norm = float(np.linalg.norm(camera_ray))
        if ray_norm < 1e-8:
            raise RuntimeError("Wrist tennis surface ray is degenerate")
        center_camera = camera_point.copy()
        center_camera[:3] += (
            float(self._r1.tennis_ball_radius_m) * camera_ray / ray_norm
        )
        refined = (base_from_camera @ center_camera)[:3]
        coarse_error = float(np.linalg.norm(refined - expected))
        coarse_error_ok = coarse_error <= self._r1.wrist_ball_coarse_max_error_m
        if not coarse_error_ok and getattr(
            self._r1,
            "r1lite_motion_validation_enabled",
            True,
        ):
            raise RuntimeError(
                f"Head/wrist tennis disagreement {coarse_error:.3f}m exceeds "
                f"{self._r1.wrist_ball_coarse_max_error_m:.3f}m"
            )
        return {
            "ok": True,
            "position_base": refined.tolist(),
            "surface_position_base": surface_point.tolist(),
            "expected_position_base": expected.tolist(),
            "coarse_error_m": coarse_error,
            "coarse_error_ok": bool(coarse_error_ok),
            "centroid_px": centroid.tolist(),
            "center_margin_ok": bool(center_margin_ok),
            "depth_m": z,
            "frame_id": obs["frame_id"],
            "stamp_ns": int(obs["stamp_ns"]),
            "segmentation": self._strip_mask(segmentation),
        }

    def goto_pose(
        self,
        position: list[float] | tuple[float, ...] | np.ndarray,
        quaternion_xyzw: list[float] | tuple[float, ...] | np.ndarray | None = None,
        *,
        gripper: float | None = None,
        hold_sec: float = 1.2,
        verify_motion: bool = True,
        translate_then_orient: bool = True,
        max_cartesian_step_m: float | None = None,
        max_orientation_step_rad: float | None = None,
        check_cartesian_ik_error: bool = True,
        require_pose_arrival: bool | None = None,
        position_arrival_tolerance_m: float | None = None,
        via_positions: list[list[float]] | None = None,
        via_quaternions_xyzw: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Plan and validate a complete IK waypoint path, then execute it.

        Args:
            position: Target XYZ in ``base_link``.
            quaternion_xyzw: Target orientation. Defaults to tool-down.
            gripper: Optional gripper command published before the pose hold.
            hold_sec: Minimum time to let the low-level target settle.
            translate_then_orient: If true, finish translation before rotating.
            max_cartesian_step_m: Optional task-specific IK path spacing.
            max_orientation_step_rad: Optional task-specific rotation spacing.
            check_cartesian_ik_error: Validate IK/FK Cartesian path agreement.
            require_pose_arrival: Override the configured real-pose arrival gate.
            position_arrival_tolerance_m: Optional task-specific XYZ tolerance.
            via_positions: Optional Cartesian route points before the target.
            via_quaternions_xyzw: Optional orientation for each route point.
        """
        if quaternion_xyzw is None:
            quaternion_xyzw = self._default_tool_down_quat_xyzw()
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        target_quaternion = np.asarray(
            quaternion_xyzw, dtype=np.float64
        ).reshape(4)
        cartesian_step = (
            R1_LITE_IK_HARNESS.max_cartesian_step_m
            if max_cartesian_step_m is None
            else float(max_cartesian_step_m)
        )
        orientation_step = (
            R1_LITE_IK_HARNESS.max_orientation_step_rad
            if max_orientation_step_rad is None
            else float(max_orientation_step_rad)
        )
        if not 0.0 < cartesian_step <= 0.06:
            raise ValueError("Cartesian IK step must be in (0m, 0.06m]")
        if not 0.0 < orientation_step <= 0.25:
            raise ValueError("orientation IK step must be in (0rad, 0.25rad]")
        plan = self.plan_pose_waypoints(
            target_position,
            target_quaternion,
            max_cartesian_step_m=cartesian_step,
            max_orientation_step_rad=orientation_step,
            max_incremental_joint_delta_rad=(
                R1_LITE_IK_HARNESS.max_incremental_joint_delta_rad
            ),
            max_total_joint_delta_rad=(
                R1_LITE_IK_HARNESS.max_total_joint_delta_rad
            ),
            max_fk_position_error_m=(
                R1_LITE_IK_HARNESS.max_fk_position_error_m
            ),
            max_fk_orientation_error_rad=(
                R1_LITE_IK_HARNESS.max_fk_orientation_error_rad
            ),
            max_backward_step_m=R1_LITE_IK_HARNESS.max_backward_step_m,
            max_forward_progress_error_m=(
                R1_LITE_IK_HARNESS.max_forward_progress_error_m
            ),
            min_z_m=self._active_min_ee_z_m,
            translate_then_orient=bool(translate_then_orient),
            check_cartesian_ik_error=bool(check_cartesian_ik_error),
            via_positions=via_positions,
            via_quaternions_xyzw=via_quaternions_xyzw,
        )
        if gripper is not None:
            self.command_gripper(float(gripper), duration_s=min(0.8, hold_sec))
        execution = []
        for index, waypoint in enumerate(plan["waypoints"]):
            result = self._r1.move_joint_waypoints_adaptive(
                waypoint["joint_feedback"],
                waypoint_step_rad=float(
                    getattr(
                        self._r1,
                        "goto_pose_joint_waypoint_step_rad",
                        0.015,
                    )
                ),
                context=(
                    f"goto_pose_waypoint_{index + 1:03d}_of_"
                    f"{len(plan['waypoints']):03d}"
                ),
            )
            execution.append(result)
        local_fk = getattr(self._r1, "get_local_planning_fk_pose", None)
        if callable(local_fk):
            measured_position, measured_quaternion = local_fk()
            measured_pose_source = "joint_feedback_local_urdf_fk"
        else:
            measured_position, measured_quaternion = (
                self._r1.get_planning_ee_pose()
            )
            measured_pose_source = "relaxed_ik_async_ee_topic"
        position_error = float(np.linalg.norm(measured_position - target_position))
        target_quaternion = target_quaternion / np.linalg.norm(target_quaternion)
        measured_quaternion = measured_quaternion / np.linalg.norm(measured_quaternion)
        orientation_error = float(
            2.0
            * np.arccos(
                np.clip(
                    abs(np.dot(target_quaternion, measured_quaternion)),
                    0.0,
                    1.0,
                )
            )
        )
        position_tolerance = (
            self._r1.pose_position_tolerance_m
            if position_arrival_tolerance_m is None
            else float(position_arrival_tolerance_m)
        )
        pose_within_tolerance = (
            position_error <= position_tolerance
            and orientation_error <= self._r1.pose_orientation_tolerance_rad
        )
        if (
            (
                verify_motion
                or bool(
                    getattr(self._r1, "r1lite_require_pose_arrival", False)
                    if require_pose_arrival is None
                    else bool(require_pose_arrival)
                )
            )
            and getattr(self._r1, "r1lite_motion_validation_enabled", True)
            and not pose_within_tolerance
        ):
            raise RuntimeError(
                "planned goto_pose execution missed its target: "
                f"position_error_m={position_error:.4f}, "
                f"orientation_error_rad={orientation_error:.4f}"
            )
        result = {
            "success": True,
            "plan": plan,
            "target_joints": plan["waypoints"][-1]["joint_feedback"],
            "execution": execution,
            "target_position": target_position.tolist(),
            "measured_position": measured_position.tolist(),
            "measured_pose_source": measured_pose_source,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "pose_within_tolerance": pose_within_tolerance,
        }
        return result

    @staticmethod
    def _slerp_quaternion(
        start_xyzw: np.ndarray,
        end_xyzw: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        start = np.asarray(start_xyzw, dtype=np.float64).reshape(4)
        end = np.asarray(end_xyzw, dtype=np.float64).reshape(4)
        start = start / np.linalg.norm(start)
        end = end / np.linalg.norm(end)
        dot = float(np.dot(start, end))
        if dot < 0.0:
            end = -end
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            result = start + float(fraction) * (end - start)
            return result / np.linalg.norm(result)
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        a = math.sin((1.0 - float(fraction)) * theta) / sin_theta
        b = math.sin(float(fraction) * theta) / sin_theta
        return a * start + b * end

    def plan_pose_waypoints(
        self,
        position: list[float] | tuple[float, ...] | np.ndarray,
        quaternion_xyzw: list[float] | tuple[float, ...] | np.ndarray,
        *,
        max_cartesian_step_m: float = R1_LITE_IK_HARNESS.max_cartesian_step_m,
        max_orientation_step_rad: float = R1_LITE_IK_HARNESS.max_orientation_step_rad,
        max_incremental_joint_delta_rad: float = (
            R1_LITE_IK_HARNESS.max_incremental_joint_delta_rad
        ),
        max_total_joint_delta_rad: float = (
            R1_LITE_IK_HARNESS.max_total_joint_delta_rad
        ),
        max_total_joint_delta_overrun_rad: float = (
            R1_LITE_IK_HARNESS.max_total_joint_delta_overrun_rad
        ),
        max_fk_position_error_m: float = (
            R1_LITE_IK_HARNESS.max_fk_position_error_m
        ),
        max_fk_orientation_error_rad: float = (
            R1_LITE_IK_HARNESS.max_fk_orientation_error_rad
        ),
        max_backward_step_m: float = R1_LITE_IK_HARNESS.max_backward_step_m,
        max_forward_progress_error_m: float = (
            R1_LITE_IK_HARNESS.max_forward_progress_error_m
        ),
        min_z_m: float | None = None,
        via_positions: list[list[float]] | None = None,
        via_quaternions_xyzw: list[list[float]] | None = None,
        ik_seed_joints: list[float] | np.ndarray | None = None,
        translate_then_orient: bool = False,
        check_cartesian_ik_error: bool = True,
    ) -> dict[str, Any]:
        """Plan isolated IK solutions along a bounded Cartesian path."""
        harness_enabled = bool(
            getattr(self._r1, "r1lite_ik_harness_enabled", True)
        )
        harness = R1LiteIkHarness(
            max_cartesian_step_m=float(max_cartesian_step_m),
            max_orientation_step_rad=float(max_orientation_step_rad),
            max_incremental_joint_delta_rad=float(
                max_incremental_joint_delta_rad
            ),
            max_total_joint_delta_rad=float(max_total_joint_delta_rad),
            max_total_joint_delta_overrun_rad=float(
                max_total_joint_delta_overrun_rad
            ),
            max_fk_position_error_m=float(max_fk_position_error_m),
            max_fk_orientation_error_rad=float(max_fk_orientation_error_rad),
            max_backward_step_m=float(max_backward_step_m),
            max_forward_progress_error_m=float(max_forward_progress_error_m),
        )
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        target_quaternion = np.asarray(
            quaternion_xyzw, dtype=np.float64
        ).reshape(4)
        validation_enabled = getattr(
            self._r1,
            "r1lite_motion_validation_enabled",
            True,
        )
        if validation_enabled:
            self._r1.validate_workspace_point(target_position)
        start_position, start_quaternion = self._r1.get_planning_ee_pose()
        start_joints = self._r1.get_arm_joints()
        start_quaternion = start_quaternion / np.linalg.norm(start_quaternion)
        target_quaternion = target_quaternion / np.linalg.norm(target_quaternion)
        via = [
            np.asarray(point, dtype=np.float64).reshape(3)
            for point in (via_positions or [])
        ]
        via_quaternions = [
            np.asarray(value, dtype=np.float64).reshape(4)
            for value in (via_quaternions_xyzw or [])
        ]
        if via_quaternions and len(via_quaternions) != len(via):
            raise ValueError(
                "via_quaternions_xyzw must match via_positions length"
            )
        via_quaternions = [
            value / np.linalg.norm(value) for value in via_quaternions
        ]
        if validation_enabled:
            for point in via:
                self._r1.validate_workspace_point(point)
        if translate_then_orient:
            if via:
                raise ValueError(
                    "translate_then_orient cannot be combined with via_positions"
                )
            path_positions = [start_position, target_position, target_position]
            path_quaternions = [
                start_quaternion,
                start_quaternion,
                target_quaternion,
            ]
        else:
            path_positions = [start_position, *via, target_position]
            path_quaternions = [
                start_quaternion,
                *(
                    via_quaternions
                    if via_quaternions
                    else [target_quaternion] * len(via)
                ),
                target_quaternion,
            ]
        for point in path_positions:
            if validation_enabled:
                self._r1.validate_workspace_point(point)
            if (
                validation_enabled
                and min_z_m is not None
                and float(point[2]) < float(min_z_m)
            ):
                raise ValueError(
                    "IK path endpoint is below the active table constraint: "
                    f"z={point[2]:.3f}, minimum={min_z_m:.3f}"
                )
        if ik_seed_joints is None:
            self._r1.reset_relaxed_ik_probe()
        else:
            self._r1.seed_relaxed_ik_probe(
                np.asarray(ik_seed_joints, dtype=np.float64).reshape(6),
            )
        if not getattr(
            self._r1,
            "r1lite_motion_validation_enabled",
            True,
        ):
            target_joints = self._r1.get_relaxed_ik_candidate(
                target_position,
                target_quaternion,
                timeout_s=2.0,
            )
            total_delta = float(
                np.max(np.abs(target_joints - start_joints))
            )
            waypoint = {
                "index": 1,
                "segment": 1,
                "position": target_position.tolist(),
                "quaternion_xyzw": target_quaternion.tolist(),
                "joint_feedback": target_joints.tolist(),
                "fk_position_base": target_position.tolist(),
                "fk_quaternion_xyzw_base": target_quaternion.tolist(),
                "fk_position_error_m": 0.0,
                "fk_orientation_error_rad": 0.0,
                "expected_forward_progress_m": float(
                    np.linalg.norm(target_position - start_position)
                ),
                "actual_forward_progress_m": float(
                    np.linalg.norm(target_position - start_position)
                ),
                "forward_step_m": float(
                    np.linalg.norm(target_position - start_position)
                ),
                "incremental_joint_delta_rad": total_delta,
                "total_joint_delta_rad": total_delta,
            }
            return {
                "success": True,
                "harness_enabled": False,
                "motion_validation_enabled": False,
                "start_position": start_position.tolist(),
                "target_position": target_position.tolist(),
                "position_distance_m": float(
                    np.linalg.norm(target_position - start_position)
                ),
                "orientation_distance_rad": float(
                    2.0
                    * np.arccos(
                        np.clip(
                            abs(
                                np.dot(
                                    start_quaternion,
                                    target_quaternion,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    )
                ),
                "harness": harness.to_dict(),
                "via_positions": [],
                "waypoint_count": 1,
                "waypoints": [waypoint],
            }
        position_distance = float(
            sum(
                np.linalg.norm(end - begin)
                for begin, end in zip(path_positions, path_positions[1:])
            )
        )
        orientation_distance = float(
            2.0
            * np.arccos(
                np.clip(
                    abs(np.dot(start_quaternion, target_quaternion)),
                    0.0,
                    1.0,
                )
            )
        )
        previous_joints = start_joints.copy()
        waypoints = []
        waypoint_index = 0
        previous_fk_position = start_position.copy()
        for segment_index, (
            segment_start,
            segment_end,
            quaternion_start,
            quaternion_end,
        ) in enumerate(
            zip(
                path_positions,
                path_positions[1:],
                path_quaternions,
                path_quaternions[1:],
            ),
            start=1,
        ):
            segment_distance = float(np.linalg.norm(segment_end - segment_start))
            segment_direction = (
                np.zeros(3, dtype=np.float64)
                if segment_distance < 1e-9
                else (segment_end - segment_start) / segment_distance
            )
            segment_orientation = float(
                2.0
                * np.arccos(
                    np.clip(
                        abs(np.dot(quaternion_start, quaternion_end)),
                        0.0,
                        1.0,
                    )
                )
            )
            segment_steps = max(
                1,
                int(np.ceil(segment_distance / max_cartesian_step_m)),
                int(np.ceil(segment_orientation / max_orientation_step_rad)),
            )
            for segment_step in range(1, segment_steps + 1):
                waypoint_index += 1
                fraction = segment_step / segment_steps
                waypoint_position = segment_start + fraction * (
                    segment_end - segment_start
                )
                waypoint_quaternion = self._slerp_quaternion(
                    quaternion_start,
                    quaternion_end,
                    fraction,
                )
                waypoint_joints = self._r1.get_relaxed_ik_candidate(
                    waypoint_position,
                    waypoint_quaternion,
                    timeout_s=2.0,
                    reference_joints=previous_joints,
                    start_joints=start_joints,
                    max_incremental_joint_delta_rad=(
                        harness.max_incremental_joint_delta_rad
                        if harness_enabled
                        else None
                    ),
                    max_total_joint_delta_rad=(
                        harness.max_total_joint_delta_rad
                        if harness_enabled
                        else None
                    ),
                    max_total_joint_delta_overrun_rad=(
                        harness.max_total_joint_delta_overrun_rad
                        if harness_enabled
                        else 0.0
                    ),
                )
                local_fk = getattr(
                    self._r1, "get_local_planning_fk_pose", None
                )
                refined_from_current_seed = False
                if callable(local_fk):
                    torso_joints = self._r1.get_torso_joints()
                    fk_position, fk_quaternion = local_fk(
                        waypoint_joints,
                        torso_joints,
                    )
                    if (
                        float(
                            np.linalg.norm(
                                fk_position - waypoint_position
                            )
                        )
                        > 0.015
                    ):
                        self._r1.reset_relaxed_ik_probe()
                        refined_joints = (
                            self._r1.get_relaxed_ik_candidate(
                                waypoint_position,
                                waypoint_quaternion,
                                timeout_s=2.0,
                                reference_joints=previous_joints,
                                start_joints=start_joints,
                                max_incremental_joint_delta_rad=(
                                    harness.max_incremental_joint_delta_rad
                                    if harness_enabled
                                    else None
                                ),
                                max_total_joint_delta_rad=(
                                    harness.max_total_joint_delta_rad
                                    if harness_enabled
                                    else None
                                ),
                                max_total_joint_delta_overrun_rad=(
                                    harness.max_total_joint_delta_overrun_rad
                                    if harness_enabled
                                    else 0.0
                                ),
                            )
                        )
                        refined_position, refined_quaternion = local_fk(
                            refined_joints,
                            torso_joints,
                        )
                        refined_error = float(
                            np.linalg.norm(
                                refined_position - waypoint_position
                            )
                        )
                        original_error = float(
                            np.linalg.norm(
                                fk_position - waypoint_position
                            )
                        )
                        refined_incremental_delta = float(
                            np.max(
                                np.abs(
                                    refined_joints - previous_joints
                                )
                            )
                        )
                        if (
                            refined_error < original_error
                            and refined_incremental_delta <= 0.60
                        ):
                            waypoint_joints = refined_joints
                            fk_position = refined_position
                            fk_quaternion = refined_quaternion
                            refined_from_current_seed = True
                else:
                    fk_position, fk_quaternion = (
                        self._r1.get_relaxed_ik_fk_pose_base()
                    )
                incremental_delta = float(
                    np.max(np.abs(waypoint_joints - previous_joints))
                )
                total_delta = float(
                    np.max(np.abs(waypoint_joints - start_joints))
                )
                fk_position_error = float(
                    np.linalg.norm(fk_position - waypoint_position)
                )
                fk_quaternion = fk_quaternion / np.linalg.norm(fk_quaternion)
                fk_orientation_error = float(
                    2.0
                    * np.arccos(
                        np.clip(
                            abs(np.dot(fk_quaternion, waypoint_quaternion)),
                            0.0,
                            1.0,
                        )
                    )
                )
                expected_progress = float(fraction * segment_distance)
                actual_progress = float(
                    np.dot(fk_position - segment_start, segment_direction)
                )
                forward_step = float(
                    np.dot(
                        fk_position - previous_fk_position,
                        segment_direction,
                    )
                )
                waypoint = {
                    "index": waypoint_index,
                    "segment": segment_index,
                    "position": waypoint_position.tolist(),
                    "quaternion_xyzw": waypoint_quaternion.tolist(),
                    "joint_feedback": waypoint_joints.tolist(),
                    "fk_position_base": fk_position.tolist(),
                    "fk_quaternion_xyzw_base": fk_quaternion.tolist(),
                    "fk_position_error_m": fk_position_error,
                    "fk_orientation_error_rad": fk_orientation_error,
                    "expected_forward_progress_m": expected_progress,
                    "actual_forward_progress_m": actual_progress,
                    "forward_step_m": forward_step,
                    "incremental_joint_delta_rad": incremental_delta,
                    "total_joint_delta_rad": total_delta,
                    "refined_from_current_seed": (
                        refined_from_current_seed
                    ),
                }
                if harness_enabled:
                    harness.validate_waypoint(
                        waypoint,
                        check_cartesian_error=bool(
                            check_cartesian_ik_error
                        ),
                    )
                waypoints.append(waypoint)
                previous_joints = waypoint_joints
                previous_fk_position = fk_position
        return {
            "success": True,
            "harness_enabled": harness_enabled,
            "start_position": start_position.tolist(),
            "target_position": target_position.tolist(),
            "position_distance_m": position_distance,
            "orientation_distance_rad": orientation_distance,
            "harness": harness.to_dict(),
            "via_positions": [point.tolist() for point in via],
            "via_quaternions_xyzw": [
                value.tolist() for value in via_quaternions
            ],
            "waypoint_count": len(waypoints),
            "waypoints": waypoints,
        }

    def descend_z_in_steps(
        self,
        target_position: list[float] | tuple[float, ...] | np.ndarray,
        quaternion_xyzw: list[float] | tuple[float, ...] | np.ndarray | None = None,
        *,
        start_z: float | None = None,
        num_steps: int = 10,
        gripper: float | None = None,
        hold_sec_per_step: float = 0.8,
        check_final_cartesian_error_only: bool = False,
    ) -> dict[str, Any]:
        """Descend vertically from a start height to ``target_position[2]``.

        Args:
            target_position: Final XYZ target. Only Z changes during descent.
            quaternion_xyzw: Orientation held constant during descent.
            start_z: Optional starting Z. Defaults to target Z + 0.12 m.
            num_steps: Number of vertical interpolation steps.
            gripper: Optional constant gripper command.
            hold_sec_per_step: Hold duration for each vertical step.
            check_final_cartesian_error_only: Check only the final grasp point.
        """
        if quaternion_xyzw is None:
            quaternion_xyzw = self._default_tool_down_quat_xyzw()
        target = np.asarray(target_position, dtype=np.float64).reshape(3)
        before = self._r1.get_arm_joints()
        z_start = float(target[2] + 0.12 if start_z is None else start_z)
        steps = max(2, int(num_steps))
        for i in range(1, steps + 1):
            z = z_start + (target[2] - z_start) * (i / steps)
            waypoint = np.array([target[0], target[1], z], dtype=np.float64)
            self.goto_pose(
                waypoint,
                quaternion_xyzw,
                gripper=gripper,
                hold_sec=hold_sec_per_step,
                verify_motion=(
                    check_final_cartesian_error_only and i == steps
                ),
                check_cartesian_ik_error=(
                    not check_final_cartesian_error_only or i == steps
                ),
                require_pose_arrival=(
                    (i == steps)
                    if check_final_cartesian_error_only
                    else None
                ),
            )
        return self._r1.verify_arm_motion(
            before,
            context="descend_z_in_steps",
            require_motion=True,
        )

    def verify_tennis_in_box(
        self,
        *,
        xy_margin: float = 0.12,
        max_z_above_box: float = 0.15,
        min_z_below_box: float = -0.10,
    ) -> tuple[bool, dict[str, Any]]:
        """Check whether the tennis ball is geometrically inside the cardboard box.

        Args:
            xy_margin: Maximum XY distance between ball and box centers.
            max_z_above_box: Maximum allowed ball height above box center.
            min_z_below_box: Minimum allowed ball height relative to box center.

        Returns:
            ``(success, details)`` with poses and threshold diagnostics.
        """
        details: dict[str, Any] = {"method": "geometric", "success": False}
        try:
            box_pos, _ = self.get_object_pose("cardboard box", save_debug=True)
            ball_pos, _ = self.get_object_pose("tennis ball", save_debug=True)
        except Exception as exc:
            details["error"] = str(exc)
            details["reason"] = "perception_failed"
            details["failed_object"] = (
                "cardboard box" if "cardboard box" in str(exc) or "box" in str(exc).lower() else
                "tennis ball" if "tennis ball" in str(exc) or "ball" in str(exc).lower() else
                "unknown"
            )
            return False, details

        box_pos = np.asarray(box_pos, dtype=float).reshape(3)
        ball_pos = np.asarray(ball_pos, dtype=float).reshape(3)
        xy_dist = float(np.linalg.norm(ball_pos[:2] - box_pos[:2]))
        z_delta = float(ball_pos[2] - box_pos[2])
        in_box_xy = xy_dist <= xy_margin
        in_box_z = min_z_below_box <= z_delta <= max_z_above_box
        success = in_box_xy and in_box_z
        if success:
            reason = "ball_in_box"
        elif not in_box_xy:
            reason = "ball_outside_box_xy"
        else:
            reason = "ball_not_at_box_height"
        details.update(
            {
                "success": success,
                "reason": reason,
                "box_pos": box_pos.tolist(),
                "ball_pos": ball_pos.tolist(),
                "xy_distance_m": xy_dist,
                "z_delta_m": z_delta,
                "thresholds": {
                    "xy_margin": xy_margin,
                    "max_z_above_box": max_z_above_box,
                    "min_z_below_box": min_z_below_box,
                },
            }
        )
        return success, details

    def verify_cup_stack(
        self,
        *,
        expected_top_center_base_m: (
            list[float] | tuple[float, ...] | np.ndarray | None
        ) = None,
        cup_height_m: float | None = None,
        nested_occlusion_expected: bool = False,
        nested_top_object_name: str = "green cup",
        orange_on_purple_xy_tolerance_m: float = 0.055,
        green_on_orange_xy_tolerance_m: float = 0.055,
        top_z_tolerance_m: float = 0.025,
    ) -> tuple[bool, dict[str, Any]]:
        """Verify top-view alignment while accounting for nested-cup occlusion.

        Fully nested inverted cups can hide the lower two colors. In that mode,
        the final green cup is checked against the immutable initial support
        center instead of requiring three independently visible masks.
        """
        details: dict[str, Any] = {
            "method": "sam3_hsv_plus_head_homography_xy",
            "success": False,
            "z_order_verified": False,
        }
        if nested_occlusion_expected:
            if expected_top_center_base_m is None or cup_height_m is None:
                raise ValueError(
                    "nested RGB-D verification requires expected top center "
                    "and cup height"
                )
            expected = np.asarray(
                expected_top_center_base_m, dtype=float
            ).reshape(3)
            top_object = str(nested_top_object_name)
            if top_object not in {"orange cup", "purple cup", "green cup"}:
                raise ValueError("unsupported nested top cup color")
            try:
                detection = self.locate_colored_cups_rgbd(
                    cup_height_m=float(cup_height_m),
                    object_names=(top_object,),
                    save_debug=True,
                    require_separation=False,
                )
                observed = np.asarray(
                    detection[top_object]["top_center_base_m"],
                    dtype=float,
                ).reshape(3)
            except Exception as exc:
                details.update(reason="perception_failed", error=str(exc))
                return False, details
            xy_error = float(np.linalg.norm(observed[:2] - expected[:2]))
            z_error = abs(float(observed[2] - expected[2]))
            success = bool(
                xy_error <= float(green_on_orange_xy_tolerance_m)
                and z_error <= float(top_z_tolerance_m)
            )
            details.update(
                {
                    "success": success,
                    "reason": (
                        "nested_top_cup_aligned_to_initial_support"
                        if success
                        else "nested_top_cup_alignment_failed"
                    ),
                    "top_cup": top_object,
                    "expected_top_center_base_m": expected.tolist(),
                    "observed_top_center_base_m": observed.tolist(),
                    "top_cup_xy_error_m": xy_error,
                    "top_cup_top_z_error_m": z_error,
                    "top_cup_xy_tolerance_m": float(
                        green_on_orange_xy_tolerance_m
                    ),
                    "top_cup_z_tolerance_m": float(top_z_tolerance_m),
                    "lower_cup_masks_required": False,
                    "limitations": (
                        "Nested lower cups may be occluded; success also relies "
                        "on the ordered execution and grasp-retention gates."
                    ),
                }
            )
            if top_object == "green cup":
                details.update(
                    {
                        "observed_green_top_center_base_m": observed.tolist(),
                        "green_xy_error_m": xy_error,
                        "green_top_z_error_m": z_error,
                        "green_xy_tolerance_m": float(
                            green_on_orange_xy_tolerance_m
                        ),
                        "green_top_z_tolerance_m": float(top_z_tolerance_m),
                    }
                )
            return success, details
        try:
            detections = self.locate_colored_cups(
                save_debug=True,
                require_separation=False,
            )
        except Exception as exc:
            details.update(reason="perception_failed", error=str(exc))
            return False, details
        positions = {
            name: np.asarray(item["position_base_m"], dtype=float).reshape(3)
            for name, item in detections.items()
        }
        orange_purple = float(
            np.linalg.norm(
                positions["orange cup"][:2] - positions["purple cup"][:2]
            )
        )
        green_orange = float(
            np.linalg.norm(
                positions["green cup"][:2] - positions["orange cup"][:2]
            )
        )
        orange_ok = orange_purple <= float(orange_on_purple_xy_tolerance_m)
        green_ok = green_orange <= float(green_on_orange_xy_tolerance_m)
        success = bool(orange_ok and green_ok)
        details.update(
            {
                "success": success,
                "reason": (
                    "cup_stack_xy_aligned"
                    if success
                    else "cup_stack_xy_alignment_failed"
                ),
                "positions_base_m": {
                    name: value.tolist() for name, value in positions.items()
                },
                "orange_on_purple_xy_error_m": orange_purple,
                "green_on_orange_xy_error_m": green_orange,
                "thresholds_m": {
                    "orange_on_purple": float(
                        orange_on_purple_xy_tolerance_m
                    ),
                    "green_on_orange": float(
                        green_on_orange_xy_tolerance_m
                    ),
                },
                "limitations": (
                    "The head-plane homography cannot prove vertical order; "
                    "final preview must be reviewed or checked with RGB-D."
                ),
            }
        )
        return success, details

    def locate_pen_head(
        self,
        *,
        use_sam3: bool = True,
        text_prompt: str | None = None,
        sam3_timeout_s: float = 30.0,
        min_score: float = 0.15,
        min_pixels: int = 40,
        save_debug: bool = True,
    ) -> dict[str, Any]:
        """Locate a pen candidate in the head-left camera image."""
        return self.locate_object_head(
            "blue pen",
            text_prompt=text_prompt or getattr(self._r1, "sam3_pen_prompt", "blue pen"),
            use_sam3=use_sam3,
            sam3_timeout_s=sam3_timeout_s,
            min_score=min_score,
            min_pixels=min_pixels,
            save_debug=save_debug,
        )

    def _locate_object_head_sam3(
        self,
        rgb: np.ndarray,
        *,
        prompt: str,
        timeout_s: float,
        min_score: float,
        min_pixels: int,
        save_debug: bool,
        slug: str = "object",
        include_mask: bool = False,
    ) -> dict[str, Any]:
        url = getattr(self._r1, "sam3_service_url", "http://127.0.0.1:8114")
        try:
            import requests

            payload = {
                "image_base64": self._encode_png_base64(rgb),
                "text_prompt": prompt,
            }
            session = requests.Session()
            session.trust_env = False
            resp = session.post(
                f"{str(url).rstrip('/')}/segment",
                json=payload,
                timeout=float(timeout_s),
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as exc:
            return {
                "ok": False,
                "method": "sam3",
                "reason": f"SAM3 unavailable at {url}: {exc}",
            }
        if not results:
            return {"ok": False, "method": "sam3", "reason": f"SAM3 returned no mask for {prompt!r}"}

        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        score = float(best.get("score", 0.0))
        shape = best.get("shape")
        if score < min_score:
            return {
                "ok": False,
                "method": "sam3",
                "reason": f"SAM3 score below threshold: {score:.3f} < {min_score:.3f}",
                "score": score,
            }
        try:
            mask = self._decode_sam3_mask(best["mask_base64"], shape)
        except Exception as exc:
            return {"ok": False, "method": "sam3", "reason": f"Could not decode SAM3 mask: {exc}"}

        ys, xs = np.nonzero(mask)
        if len(xs) < min_pixels:
            return {
                "ok": False,
                "method": "sam3",
                "reason": f"SAM3 mask pixels below threshold: {len(xs)} < {min_pixels}",
                "score": score,
                "count": int(len(xs)),
            }
        if save_debug:
            self._save_mask_debug(mask, f"outputs/r1lite/locate_{slug}_sam3_mask.png")
            self._save_overlay_debug(
                rgb,
                mask,
                f"outputs/r1lite/locate_{slug}_sam3_overlay.png",
            )
        pixel = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
        median_rgb = np.median(
            np.asarray(rgb, dtype=np.uint8)[mask],
            axis=0,
        )
        hue, saturation, value = colorsys.rgb_to_hsv(
            *(np.asarray(median_rgb, dtype=float) / 255.0)
        )
        result = {
            "ok": True,
            "method": "sam3",
            "prompt": prompt,
            "score": score,
            "box": best.get("box"),
            "count": int(len(xs)),
            "pixel": pixel.tolist(),
            "centroid_u": float(pixel[0]),
            "centroid_v": float(pixel[1]),
            "median_rgb": np.asarray(median_rgb, dtype=float).tolist(),
            "median_hsv": [
                float(hue * 360.0),
                float(saturation),
                float(value),
            ],
        }
        if include_mask:
            result["mask"] = mask
        return result

    def pixel_to_base_on_plane(
        self,
        pixel: list[float] | tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """Convert a head-left image pixel to base-frame point on the calibrated plane.

        Args:
            pixel: Image pixel ``[u, v]`` in the original head-left image.

        Returns:
            A numpy array ``[x, y, z]`` in the robot base plane frame.
        """
        return self._r1.pixel_to_base_on_plane(pixel)

    def plan_pen_pregrasp(
        self,
        *,
        pixel: list[float] | tuple[float, float] | np.ndarray | None = None,
        approach_height: float = 0.12,
        quaternion_xyzw: list[float] | tuple[float, ...] = (0.0, -0.2, 0.0, 0.98),
        save_debug: bool = True,
    ) -> dict[str, Any]:
        """Plan a conservative pregrasp candidate for the pen without moving.

        The planner maps a detected head pixel to base-plane coordinates and
        attempts a guarded RelaxedIK candidate only if enabled in the low-level
        config.  If no safe IK candidate is available, the result is diagnostic
        only and should not be executed.

        Args:
            pixel: Optional head-left image pixel. If omitted, locate_pen_head()
                is used.
            approach_height: Height above calibrated plane for pregrasp.
            quaternion_xyzw: Candidate gripper orientation for IK probing.
            save_debug: Save debug camera images.

        Returns:
            A plan dictionary with ``ok``, ``target_xyz``, ``pregrasp_xyz``,
            and optionally ``pregrasp_joints``.
        """
        if pixel is None:
            detection = self.locate_pen_head(save_debug=save_debug)
            if not detection["ok"]:
                return {"ok": False, "stage": "detect", **detection}
            pixel = detection["pixel"]
        try:
            target = self._r1.pixel_to_base_on_plane(pixel)
            pregrasp = target.copy()
            pregrasp[2] += float(approach_height)
            self._r1.validate_workspace_point(pregrasp)
        except Exception as exc:
            plan: dict[str, Any] = {
                "ok": False,
                "stage": "calibration",
                "pixel": np.asarray(pixel, dtype=float).reshape(2).tolist(),
                "reason": str(exc),
                "safe_to_execute": False,
            }
            if save_debug:
                plan["head_debug_image"] = self.save_head_rgb(
                    "outputs/r1lite/plan_pen_head_left.png",
                    camera="left",
                )
                plan["wrist_debug_image"] = self.save_wrist_rgb(
                    "outputs/r1lite/plan_pen_wrist.png"
                )
            return plan
        plan: dict[str, Any] = {
            "ok": False,
            "stage": "ik",
            "pixel": np.asarray(pixel, dtype=float).reshape(2).tolist(),
            "target_xyz": target.tolist(),
            "pregrasp_xyz": pregrasp.tolist(),
        }
        if save_debug:
            plan["head_debug_image"] = self.save_head_rgb(
                "outputs/r1lite/plan_pen_head_left.png",
                camera="left",
            )
            plan["wrist_debug_image"] = self.save_wrist_rgb(
                "outputs/r1lite/plan_pen_wrist.png"
            )
        try:
            q = self._r1.get_relaxed_ik_candidate(pregrasp, quaternion_xyzw)
        except Exception as exc:
            plan["reason"] = str(exc)
            plan["safe_to_execute"] = False
            return plan
        current = self._r1.get_arm_joints()
        max_delta = float(np.max(np.abs(np.asarray(q) - current)))
        plan["pregrasp_joints"] = np.asarray(q, dtype=float).reshape(6).tolist()
        plan["max_joint_delta"] = max_delta
        plan["safe_to_execute"] = max_delta <= 0.35
        plan["ok"] = bool(plan["safe_to_execute"])
        if not plan["ok"]:
            plan["reason"] = f"IK joint delta too large: {max_delta:.3f} rad"
        return plan

    def plan_pen_graspnet_candidates(
        self,
        *,
        text_prompt: str | None = None,
        use_sam3: bool = True,
        save_debug: bool = True,
        max_candidates: int = 8,
    ) -> dict[str, Any]:
        """Generate Contact-GraspNet candidates from head context and wrist RGB-D.

        Head camera detection is used for global confirmation only because the
        head stream has no depth.  Contact-GraspNet is run on right-wrist RGB-D
        point clouds, so returned grasp poses are in the wrist camera frame.
        This helper never moves the robot and does not claim candidates are
        executable.
        """
        head_detection = self.locate_pen_head(
            use_sam3=use_sam3,
            text_prompt=text_prompt,
            save_debug=save_debug,
        )
        if not head_detection.get("ok"):
            return {
                "ok": False,
                "stage": "head_detection",
                "reason": head_detection.get("reason", "head detection failed"),
                "head_detection": head_detection,
                "safe_to_execute": False,
            }

        try:
            wrist_obs = self._r1.get_wrist_rgbd(require_depth=True)
            wrist_rgb = wrist_obs["rgb"]
            wrist_depth = wrist_obs["depth"]
            wrist_k = wrist_obs["intrinsics"]
            if wrist_rgb is None or wrist_depth is None or wrist_k is None:
                raise RuntimeError("Wrist RGB-D or intrinsics are not available")
        except Exception as exc:
            return {
                "ok": False,
                "stage": "wrist_rgbd",
                "reason": str(exc),
                "head_detection": head_detection,
                "safe_to_execute": False,
            }

        prompt = text_prompt or getattr(self._r1, "sam3_pen_prompt", "blue pen")
        wrist_mask_result = self._segment_pen_mask(
            np.asarray(wrist_rgb, dtype=np.uint8),
            prompt=prompt,
            use_sam3=use_sam3,
            camera_name="wrist",
            save_debug=save_debug,
        )
        if not wrist_mask_result.get("ok"):
            return {
                "ok": False,
                "stage": "wrist_segmentation",
                "reason": wrist_mask_result.get("reason", "wrist segmentation failed"),
                "head_detection": head_detection,
                "wrist_segmentation": self._strip_mask(wrist_mask_result),
                "safe_to_execute": False,
            }

        mask = wrist_mask_result["mask"]
        pc_full, pc_segment = self._wrist_point_clouds(
            np.asarray(wrist_depth, dtype=np.float32),
            np.asarray(wrist_k, dtype=np.float64),
            np.asarray(mask, dtype=bool),
        )
        min_points = int(getattr(self._r1, "graspnet_min_segment_points", 30))
        if len(pc_segment) < min_points:
            return {
                "ok": False,
                "stage": "point_cloud",
                "reason": f"Only {len(pc_segment)} wrist segment points; need {min_points}",
                "head_detection": head_detection,
                "wrist_segmentation": self._strip_mask(wrist_mask_result),
                "pc_full_points": int(pc_full_points),
                "pc_segment_points": int(pc_segment_points),
                "safe_to_execute": False,
            }

        pc_full = self._subsample_points(
            pc_full,
            int(getattr(self._r1, "graspnet_max_full_points", 20000)),
        )
        pc_segment = self._subsample_points(
            pc_segment,
            int(getattr(self._r1, "graspnet_max_segment_points", 10000)),
        )
        try:
            grasps, scores, contact_pts = self._call_graspnet_point_clouds(
                pc_full.astype(np.float32),
                pc_segment.astype(np.float32),
            )
        except Exception as exc:
            return {
                "ok": False,
                "stage": "graspnet",
                "reason": str(exc),
                "head_detection": head_detection,
                "wrist_segmentation": self._strip_mask(wrist_mask_result),
                "pc_full_points": int(len(pc_full)),
                "pc_segment_points": int(len(pc_segment)),
                "safe_to_execute": False,
            }

        grasps = np.asarray(grasps, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        contact_pts = np.asarray(contact_pts, dtype=np.float64)
        if grasps.size == 0 or len(grasps) == 0:
            return {
                "ok": False,
                "stage": "graspnet",
                "reason": "Contact-GraspNet returned no grasp candidates",
                "head_detection": head_detection,
                "wrist_segmentation": self._strip_mask(wrist_mask_result),
                "pc_full_points": int(len(pc_full)),
                "pc_segment_points": int(len(pc_segment)),
                "safe_to_execute": False,
            }
        if len(scores) != len(grasps):
            scores = np.zeros(len(grasps), dtype=np.float64)

        order = np.argsort(scores)[::-1]
        keep = order[: max(1, int(max_candidates))]
        candidates = []
        for idx in keep:
            candidate = {
                "index": int(idx),
                "score": float(scores[idx]),
                "grasp_pose_wrist_camera": grasps[idx].tolist(),
            }
            if contact_pts.ndim >= 2 and len(contact_pts) > idx:
                candidate["contact_point_wrist_camera"] = contact_pts[idx].tolist()
            candidates.append(candidate)

        if save_debug:
            self.save_wrist_rgb("outputs/r1lite/graspnet_wrist_rgb.png")
        return {
            "ok": True,
            "stage": "graspnet",
            "frame": "right_wrist_camera",
            "safe_to_execute": False,
            "reason": (
                "Generated wrist-camera-frame GraspNet candidates. "
                "Execution still requires wrist-to-base calibration and IK safety validation."
            ),
            "head_detection": head_detection,
            "wrist_segmentation": self._strip_mask(wrist_mask_result),
            "pc_full_points": int(pc_full_points),
            "pc_segment_points": int(pc_segment_points),
            "num_candidates": int(len(grasps)),
            "candidates": candidates,
        }

    def plan_tennis_graspnet(
        self,
        *,
        text_prompt: str | None = None,
        use_sam3: bool = True,
        save_debug: bool = True,
        max_candidates: int | None = None,
        confirm_head: bool = True,
        camera_only: bool = False,
        camera: str = "head",
        expected_ball_position_base: (
            list[float] | tuple[float, ...] | np.ndarray | None
        ) = None,
    ) -> dict[str, Any]:
        """Select the safest executable Contact-GraspNet tennis grasp.

        All retained candidates are checked by the R1 Lite IK harness. Selection
        minimizes total joint displacement and FK error before using GraspNet
        score as a tie-breaker.
        """
        camera_key = str(camera).strip().lower()
        if camera_key in {"head", "head_left", "head-stereo", "head_stereo"}:
            camera_key = "head"
        elif camera_key in {"wrist", "right_wrist"}:
            camera_key = "wrist"
        else:
            raise ValueError("camera must be 'head' or 'wrist'")
        expected_ball = (
            None
            if expected_ball_position_base is None
            else np.asarray(
                expected_ball_position_base,
                dtype=np.float64,
            ).reshape(3)
        )
        if confirm_head:
            head_detection = self.locate_object_head(
                "tennis ball",
                text_prompt=text_prompt,
                use_sam3=use_sam3,
                save_debug=save_debug,
            )
            if not head_detection.get("ok"):
                raise RuntimeError(
                    "Tennis-ball head confirmation failed before GraspNet: "
                    f"{head_detection.get('reason', head_detection)}"
                )
        else:
            head_detection = {
                "ok": True,
                "skipped": True,
                "reason": "confirmed before moving to wrist observation pose",
            }

        if camera_key == "head":
            rgbd_obs = self._r1.get_head_rgbd(camera="left", require_depth=True)
            camera_label = "head_stereo_left"
            debug_prefix = "head_stereo"
        else:
            rgbd_obs = self._r1.get_wrist_rgbd(require_depth=True)
            camera_label = "right_wrist"
            debug_prefix = "wrist"
        camera_rgb = rgbd_obs.get("rgb")
        camera_depth = rgbd_obs.get("depth")
        camera_k = rgbd_obs.get("intrinsics")
        frame_id = str(rgbd_obs.get("frame_id") or "")
        stamp_ns = int(rgbd_obs.get("stamp_ns") or 0)
        if camera_rgb is None or camera_depth is None or camera_k is None:
            raise RuntimeError(
                f"Fresh {camera_label} RGB-D and intrinsics are required for GraspNet"
            )
        base_from_camera = None
        if not camera_only:
            # Lock the observation transform before GPU inference outlives the TF cache.
            base_from_camera = self._r1.lookup_camera_to_base(
                frame_id=frame_id,
                stamp_ns=stamp_ns,
                camera_label=camera_label,
            )
        prompt = text_prompt or self._r1.object_prompt("tennis ball")
        camera_segmentation = self._segment_tennis_mask(
            np.asarray(camera_rgb, dtype=np.uint8),
            prompt=prompt,
            use_sam3=use_sam3,
            save_debug=save_debug,
            camera_name=debug_prefix,
        )
        if not camera_segmentation.get("ok"):
            raise RuntimeError(
                f"{camera_label} tennis segmentation failed before GraspNet: "
                f"{camera_segmentation.get('reason', camera_segmentation)}"
            )
        segment_mask = np.asarray(camera_segmentation["mask"], dtype=bool)
        min_points = int(getattr(self._r1, "graspnet_min_segment_points", 30))
        input_mode = str(
            getattr(self._r1, "graspnet_input_mode", "point_clouds")
        ).strip().lower()
        if input_mode in {"depth", "depth_segmap", "rgbd", "plan"}:
            depth_array = np.asarray(camera_depth, dtype=np.float32)
            near, far = getattr(self._r1, "graspnet_depth_range", (0.2, 2.0))
            valid_full = np.isfinite(depth_array) & (depth_array > near) & (depth_array < far)
            valid_segment = valid_full & segment_mask
            pc_full_points = int(np.count_nonzero(valid_full))
            pc_segment_points = int(np.count_nonzero(valid_segment))
            if pc_segment_points < min_points:
                raise RuntimeError(
                    f"Only {pc_segment_points} {camera_label} tennis depth pixels are "
                    f"available; need {min_points}"
                )
            segmap = np.zeros(np.asarray(camera_depth).shape, dtype=np.int32)
            segmap[segment_mask] = 1
            raw_grasps, scores, contact_pts = self._call_graspnet_depth_segmap(
                depth_array,
                np.asarray(camera_k, dtype=np.float32),
                segmap,
            )
            graspnet_endpoint = "plan"
            segment_depth_filter = {
                "enabled": False,
                "reason": "Cap-X depth+segmap endpoint uses the SAM mask directly",
                "mask_pixels_before": int(np.count_nonzero(segment_mask)),
                "mask_pixels_after": int(np.count_nonzero(segment_mask)),
            }
        elif input_mode in {"point_cloud", "point_clouds", "pc"}:
            segment_mask, segment_depth_filter = self._filter_graspnet_segment_by_depth(
                np.asarray(camera_depth, dtype=np.float32),
                segment_mask,
                camera_label=camera_label,
            )
            pc_full, pc_segment = self._rgbd_point_clouds(
                np.asarray(camera_depth, dtype=np.float32),
                np.asarray(camera_k, dtype=np.float64),
                segment_mask,
                camera_label=camera_label,
            )
            if len(pc_segment) < min_points:
                raise RuntimeError(
                    f"Only {len(pc_segment)} {camera_label} tennis points are available; "
                    f"need {min_points}"
                )
            pc_full = self._subsample_points(
                pc_full,
                int(getattr(self._r1, "graspnet_max_full_points", 20000)),
            )
            pc_segment = self._subsample_points(
                pc_segment,
                int(getattr(self._r1, "graspnet_max_segment_points", 10000)),
            )
            pc_full_points = len(pc_full)
            pc_segment_points = len(pc_segment)
            raw_grasps, scores, contact_pts = self._call_graspnet_point_clouds(
                pc_full.astype(np.float32),
                pc_segment.astype(np.float32),
            )
            graspnet_endpoint = "plan_point_clouds"
        else:
            raise ValueError(
                "graspnet_input_mode must be 'depth_segmap' or 'point_clouds'"
            )
        raw_grasps = np.asarray(raw_grasps, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        contact_pts = np.asarray(contact_pts, dtype=np.float64)
        if raw_grasps.size == 0:
            raise RuntimeError(
                "Contact-GraspNet returned no tennis grasp candidates "
                f"for {camera_label}: pc_full={pc_full_points}, "
                f"pc_segment={pc_segment_points}, "
                f"endpoint={graspnet_endpoint}, "
                f"segmentation={self._strip_mask(camera_segmentation)}, "
                f"segment_depth_filter={segment_depth_filter}"
            )
        if raw_grasps.ndim != 3 or raw_grasps.shape[1:] != (4, 4):
            raise RuntimeError(
                f"Contact-GraspNet returned invalid grasp shape {raw_grasps.shape}"
            )
        if len(scores) != len(raw_grasps):
            raise RuntimeError(
                "Contact-GraspNet score count does not match its grasp candidates"
            )
        if camera_only:
            if save_debug:
                if camera_key == "head":
                    self.save_head_stereo_rgb(
                        "outputs/r1lite/graspnet_tennis_head_stereo_rgb.png"
                    )
                else:
                    self.save_wrist_rgb(
                        "outputs/r1lite/graspnet_tennis_wrist_rgb.png"
                    )
            return {
                "ok": True,
                "camera_only": True,
                "safe_to_execute": False,
                "reason": "Camera-frame GraspNet validation only; TF and IK were not evaluated",
                "camera": camera_key,
                "frame_id": frame_id,
                "stamp_ns": stamp_ns,
                "pc_full_points": int(pc_full_points),
                "pc_segment_points": int(pc_segment_points),
                "graspnet_endpoint": graspnet_endpoint,
                "head_detection": head_detection,
                "camera_segmentation": self._strip_mask(camera_segmentation),
                "segment_depth_filter": segment_depth_filter,
                "num_candidates": int(len(raw_grasps)),
                "scores": scores.tolist(),
                "grasps_camera": raw_grasps.tolist(),
                "contact_points_camera": contact_pts.tolist(),
            }

        if base_from_camera is None:
            raise RuntimeError("Camera transform was not captured for motion planning")

        approach_axis_name = str(getattr(self._r1, "graspnet_approach_axis", "z"))
        approach_distance = float(
            getattr(self._r1, "graspnet_pregrasp_distance_m", 0.12)
        )
        prefer_top_down = bool(
            getattr(self._r1, "graspnet_prefer_top_down", False)
        )
        top_down_threshold = float(
            getattr(self._r1, "graspnet_top_down_threshold", 0.5)
        )
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ee_from_grasp = getattr(self._r1, "graspnet_ee_from_grasp", None)
        if ee_from_grasp is None:
            raise RuntimeError(
                "Mandatory GraspNet execution requires a calibrated "
                "graspnet_ee_from_grasp 4x4 transform"
            )
        ee_from_grasp = _validate_rigid_transform(
            np.asarray(ee_from_grasp, dtype=np.float64),
            "graspnet_ee_from_grasp",
        )
        candidate_limit = min(
            len(raw_grasps),
            int(
                max_candidates
                if max_candidates is not None
                else getattr(self._r1, "graspnet_max_candidates", 16)
            ),
        )
        min_score = float(getattr(self._r1, "graspnet_min_score", 0.0))
        ordered_indices = np.argsort(scores)[::-1][:candidate_limit]
        base_camera = _validate_rigid_transform(
            base_from_camera,
            "base_from_camera",
        )
        accepted_candidates: list[dict[str, Any]] = []
        rejected_candidates: list[dict[str, Any]] = []
        table_calibration = self._r1.load_head_plane_calibration()
        table_z = float(table_calibration["plane_z"])
        table_clearance = float(self._r1.graspnet_table_clearance_m)
        validation_enabled = getattr(
            self._r1,
            "r1lite_motion_validation_enabled",
            True,
        )
        for raw_index in ordered_indices:
            index = int(raw_index)
            score = float(scores[index])
            try:
                if score < min_score:
                    raise ValueError(
                        f"score {score:.4f} is below {min_score:.4f}"
                    )
                base_from_contact_grasp = base_camera @ _validate_rigid_transform(
                    raw_grasps[index],
                    "camera_from_grasp",
                )
                base_from_ee = _transform_graspnet_pose(
                    raw_grasps[index],
                    base_camera,
                    ee_from_grasp=ee_from_grasp,
                )
                grasp_position, grasp_quaternion = _matrix_to_pose_xyzw(
                    base_from_ee
                )
                approach_axis = _grasp_axis(
                    base_from_contact_grasp[:3, :3],
                    approach_axis_name,
                )
                top_down_alignment = float(-np.dot(approach_axis, world_z))
                if (
                    validation_enabled
                    and prefer_top_down
                    and top_down_alignment < top_down_threshold
                ):
                    raise ValueError(
                        "approach is not top-down: alignment "
                        f"{top_down_alignment:.3f} is below "
                        f"{top_down_threshold:.3f}"
                    )
                pregrasp_position = (
                    grasp_position - approach_axis * approach_distance
                )
                if validation_enabled:
                    self._r1.validate_workspace_point(pregrasp_position)
                    self._r1.validate_workspace_point(grasp_position)
                if validation_enabled and grasp_position[2] < table_z:
                    raise ValueError(
                        "grasp target is below the calibrated table plane: "
                        f"z={grasp_position[2]:.3f}, table={table_z:.3f}"
                    )
                if validation_enabled and pregrasp_position[2] < table_z + table_clearance:
                    raise ValueError(
                        "pregrasp target violates table clearance: "
                        f"z={pregrasp_position[2]:.3f}, "
                        f"minimum={table_z + table_clearance:.3f}"
                    )
                if expected_ball is not None:
                    coarse_error = float(
                        np.linalg.norm(grasp_position - expected_ball)
                    )
                    if validation_enabled and coarse_error > float(
                        getattr(self._r1, "wrist_ball_coarse_max_error_m", 0.12)
                    ):
                        raise ValueError(
                            f"grasp is {coarse_error:.3f}m from the expected ball"
                        )
                pregrasp_plan = self.plan_pose_waypoints(
                    pregrasp_position,
                    grasp_quaternion,
                    min_z_m=table_z if validation_enabled else None,
                    translate_then_orient=True,
                )
                candidate = {
                    "index": index,
                    "score": score,
                    "grasp_pose_camera": raw_grasps[index].tolist(),
                    "contact_grasp_pose_base": base_from_contact_grasp.tolist(),
                    "ee_pose_base": base_from_ee.tolist(),
                    "grasp_position": grasp_position.tolist(),
                    "grasp_quaternion_xyzw": grasp_quaternion.tolist(),
                    "approach_axis_base": approach_axis.tolist(),
                    "top_down_alignment": top_down_alignment,
                    "pregrasp_position": pregrasp_position.tolist(),
                    "r1lite_ik_harness": {
                        "waypoint_count": pregrasp_plan["waypoint_count"],
                        "harness": pregrasp_plan["harness"],
                        "max_incremental_joint_delta_rad": max(
                            point["incremental_joint_delta_rad"]
                            for point in pregrasp_plan["waypoints"]
                        ),
                        "max_total_joint_delta_rad": max(
                            point["total_joint_delta_rad"]
                            for point in pregrasp_plan["waypoints"]
                        ),
                        "max_fk_position_error_m": max(
                            point["fk_position_error_m"]
                            for point in pregrasp_plan["waypoints"]
                        ),
                    },
                    "ik_waypoints": pregrasp_plan["waypoints"],
                }
                if contact_pts.ndim >= 2 and len(contact_pts) > index:
                    candidate["contact_point_camera"] = contact_pts[index].tolist()
                accepted_candidates.append(candidate)
            except Exception as exc:
                rejected_candidates.append(
                    {
                        "index": index,
                        "score": score,
                        "reason": str(exc),
                    }
                )
        if not accepted_candidates:
            raise RuntimeError(
                "No Contact-GraspNet candidate passed the R1 Lite IK harness: "
                f"{rejected_candidates}"
            )
        if prefer_top_down:
            accepted_candidates.sort(
                key=lambda candidate: (
                    -float(candidate["top_down_alignment"]),
                    candidate["r1lite_ik_harness"]["max_total_joint_delta_rad"],
                    candidate["r1lite_ik_harness"]["max_fk_position_error_m"],
                    candidate["r1lite_ik_harness"]["max_incremental_joint_delta_rad"],
                    -candidate["score"],
                )
            )
            selection_policy = "max_top_down_alignment_then_min_joint_delta_then_fk_error_then_score"
        else:
            accepted_candidates.sort(
                key=lambda candidate: (
                    candidate["r1lite_ik_harness"]["max_total_joint_delta_rad"],
                    candidate["r1lite_ik_harness"]["max_fk_position_error_m"],
                    candidate["r1lite_ik_harness"]["max_incremental_joint_delta_rad"],
                    -candidate["score"],
                )
            )
            selection_policy = "executable_then_min_total_joint_delta_then_fk_error_then_score"
        selected = accepted_candidates[0]
        self._active_min_ee_z_m = table_z
        if save_debug:
            if camera_key == "head":
                self.save_head_stereo_rgb(
                    "outputs/r1lite/graspnet_tennis_head_stereo_rgb.png"
                )
            else:
                self.save_wrist_rgb("outputs/r1lite/graspnet_tennis_wrist_rgb.png")
        return {
            "ok": True,
            "mandatory_graspnet": True,
            "fallback_used": False,
            **selected,
            "camera": camera_key,
            "camera_label": camera_label,
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
            "base_from_camera": base_from_camera.tolist(),
            "pc_full_points": int(pc_full_points),
            "pc_segment_points": int(pc_segment_points),
            "graspnet_endpoint": graspnet_endpoint,
            "head_detection": head_detection,
            "camera_segmentation": self._strip_mask(camera_segmentation),
            "segment_depth_filter": segment_depth_filter,
            "expected_ball_position_base": (
                None if expected_ball is None else expected_ball.tolist()
            ),
            "selected": selected,
            "ik_constraints": {
                "table_z_m": table_z,
                "pregrasp_clearance_m": table_clearance,
                "minimum_ee_z_m": table_z,
                **R1_LITE_IK_HARNESS.to_dict(),
            },
            "selection_policy": selection_policy,
            "accepted_candidates": accepted_candidates,
            "rejected_candidates": rejected_candidates,
        }

    def execute_grasp_sequence(
        self,
        plan: dict[str, Any],
        *,
        execute: bool = False,
        close_position: float | None = None,
        lift_joint_delta: float = -0.04,
    ) -> dict[str, Any]:
        """Execute a validated pregrasp/close/lift sequence.

        Args:
            plan: Output from plan_pen_pregrasp().
            execute: Must be true to move. If false, this only validates.
            close_position: Optional gripper close command.
            lift_joint_delta: Small delta applied to joint 2 after closing.

        Returns:
            Execution result dictionary.
        """
        if not plan.get("ok") or not plan.get("safe_to_execute"):
            return {
                "executed": False,
                "reason": plan.get("reason", "plan is not safe_to_execute"),
                "plan": plan,
            }
        if "pregrasp_joints" not in plan:
            return {
                "executed": False,
                "reason": "plan has no pregrasp_joints",
                "plan": plan,
            }
        if not execute:
            return {"executed": False, "reason": "execute=False", "plan": plan}

        q_pre = np.asarray(plan["pregrasp_joints"], dtype=float).reshape(6)
        q_after_pre = self._r1.move_joints_safe(q_pre, max_step=0.02, duration_s=2.0)
        self.open_gripper()
        grip = self.close_gripper(position=close_position)
        q_lift = np.asarray(q_after_pre, dtype=float).copy()
        q_lift[1] += float(lift_joint_delta)
        q_after_lift = self._r1.move_joints_safe(q_lift, max_step=0.01, duration_s=1.5)
        return {
            "executed": True,
            "pregrasp_feedback": q_after_pre,
            "gripper_feedback": grip,
            "lift_feedback": q_after_lift,
        }

    def _encode_png_base64(self, rgb: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _numpy_to_base64(self, arr: np.ndarray) -> str:
        with io.BytesIO() as f:
            np.save(f, np.asarray(arr))
            return base64.b64encode(f.getvalue()).decode("utf-8")

    def _base64_to_numpy(self, b64_str: str) -> np.ndarray:
        with io.BytesIO(base64.b64decode(b64_str)) as f:
            return np.load(f)

    def _decode_sam3_mask(self, mask_base64: str, shape: Any) -> np.ndarray:
        h, w = [int(v) for v in shape]
        raw = base64.b64decode(mask_base64)
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
        return mask.astype(bool)

    def _save_mask_debug(self, mask: np.ndarray, path: str) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(out)
        return str(out)

    def _save_overlay_debug(self, rgb: np.ndarray, mask: np.ndarray, path: str) -> str:
        img = np.asarray(rgb, dtype=np.uint8).copy()
        if mask.shape != img.shape[:2]:
            mask = self._resize_mask(mask, img.shape[:2])
        overlay = img.copy()
        overlay[mask] = np.array([255, 80, 80], dtype=np.uint8)
        blended = (0.65 * img + 0.35 * overlay).astype(np.uint8)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(blended).save(out)
        return str(out)

    def _segment_pen_mask(
        self,
        rgb: np.ndarray,
        *,
        prompt: str,
        use_sam3: bool,
        camera_name: str,
        save_debug: bool,
    ) -> dict[str, Any]:
        if use_sam3:
            sam3 = self._sam3_segment_mask(
                rgb,
                prompt=prompt,
                min_score=0.15,
                min_pixels=10,
            )
            if sam3["ok"]:
                if save_debug:
                    self._save_mask_debug(
                        sam3["mask"],
                        f"outputs/r1lite/{camera_name}_sam3_mask.png",
                    )
                    self._save_overlay_debug(
                        rgb,
                        sam3["mask"],
                        f"outputs/r1lite/{camera_name}_sam3_overlay.png",
                    )
                return sam3
            return sam3
        mask = self._blue_pen_mask(rgb)
        count = int(np.count_nonzero(mask))
        if save_debug:
            self._save_mask_debug(mask, f"outputs/r1lite/{camera_name}_color_mask.png")
        if count < 10:
            return {
                "ok": False,
                "method": "color_fallback",
                "reason": f"color fallback mask too small: {count}",
                "sam3_reason": None if not use_sam3 else sam3.get("reason"),
            }
        ys, xs = np.nonzero(mask)
        return {
            "ok": True,
            "method": "color_fallback",
            "mask": mask,
            "count": count,
            "pixel": [float(xs.mean()), float(ys.mean())],
            "sam3_reason": None if not use_sam3 else sam3.get("reason"),
        }

    def _segment_tennis_mask(
        self,
        rgb: np.ndarray,
        *,
        prompt: str,
        use_sam3: bool,
        save_debug: bool,
        camera_name: str = "wrist",
    ) -> dict[str, Any]:
        sam3_reason: str | None = None
        if use_sam3 and self._sam3_available():
            sam3 = self._sam3_segment_mask(
                rgb,
                prompt=prompt,
                min_score=0.15,
                min_pixels=30,
            )
            if sam3["ok"]:
                if save_debug:
                    self._save_mask_debug(
                        sam3["mask"],
                        f"outputs/r1lite/{camera_name}_tennis_sam3_mask.png",
                    )
                    self._save_overlay_debug(
                        rgb,
                        sam3["mask"],
                        f"outputs/r1lite/{camera_name}_tennis_sam3_overlay.png",
                    )
                return sam3
            sam3_reason = str(sam3.get("reason", "SAM3 tennis segmentation failed"))
        elif use_sam3:
            sam3_reason = "SAM3 service unavailable"
        if use_sam3:
            return {
                "ok": False,
                "method": "sam3",
                "reason": sam3_reason,
            }

        mask = self._yellow_tennis_ball_mask(np.asarray(rgb, dtype=np.uint8))
        raw_count = int(np.count_nonzero(mask))
        try:
            import cv2

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8),
                connectivity=8,
            )
            best_label: int | None = None
            best_area = 0
            for label in range(1, num_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                width = int(stats[label, cv2.CC_STAT_WIDTH])
                height = int(stats[label, cv2.CC_STAT_HEIGHT])
                if area < 30 or width <= 0 or height <= 0:
                    continue
                aspect = max(width, height) / max(1, min(width, height))
                compactness = area / float(width * height)
                if aspect > 1.6 or compactness < 0.60:
                    continue
                if area > best_area:
                    best_label = label
                    best_area = area
            mask = (
                np.zeros_like(mask, dtype=bool)
                if best_label is None
                else labels == best_label
            )
        except Exception:
            pass
        count = int(np.count_nonzero(mask))
        if save_debug:
            self._save_mask_debug(
                mask,
                f"outputs/r1lite/{camera_name}_tennis_color_mask.png",
            )
            self._save_overlay_debug(
                rgb,
                mask,
                f"outputs/r1lite/{camera_name}_tennis_color_overlay.png",
            )
        if count < 30:
            return {
                "ok": False,
                "method": "color_fallback",
                "reason": (
                    "no compact round tennis component: "
                    f"selected={count}, raw_color_pixels={raw_count}"
                ),
                "sam3_reason": sam3_reason,
            }
        ys, xs = np.nonzero(mask)
        return {
            "ok": True,
            "method": "color_fallback",
            "mask": mask,
            "count": count,
            "pixel": [float(xs.mean()), float(ys.mean())],
            "sam3_reason": sam3_reason,
        }

    def _sam3_segment_mask(
        self,
        rgb: np.ndarray,
        *,
        prompt: str,
        min_score: float,
        min_pixels: int,
    ) -> dict[str, Any]:
        url = getattr(self._r1, "sam3_service_url", "http://127.0.0.1:8114")
        try:
            import requests

            payload = {
                "image_base64": self._encode_png_base64(rgb),
                "text_prompt": prompt,
            }
            session = requests.Session()
            session.trust_env = False
            resp = session.post(f"{str(url).rstrip('/')}/segment", json=payload, timeout=30.0)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as exc:
            return {"ok": False, "method": "sam3", "reason": f"SAM3 unavailable: {exc}"}
        if not results:
            return {"ok": False, "method": "sam3", "reason": f"SAM3 returned no mask for {prompt!r}"}
        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        score = float(best.get("score", 0.0))
        if score < min_score:
            return {
                "ok": False,
                "method": "sam3",
                "reason": f"SAM3 score below threshold: {score:.3f} < {min_score:.3f}",
                "score": score,
            }
        mask = self._decode_sam3_mask(best["mask_base64"], best.get("shape"))
        count = int(np.count_nonzero(mask))
        if count < min_pixels:
            return {
                "ok": False,
                "method": "sam3",
                "reason": f"SAM3 mask too small: {count} < {min_pixels}",
                "score": score,
                "count": count,
            }
        ys, xs = np.nonzero(mask)
        return {
            "ok": True,
            "method": "sam3",
            "prompt": prompt,
            "score": score,
            "box": best.get("box"),
            "mask": mask,
            "count": count,
            "pixel": [float(xs.mean()), float(ys.mean())],
        }

    def _resize_mask(self, mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        if mask.shape == target_hw:
            return mask.astype(bool)
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (target_hw[1], target_hw[0]),
            resample=Image.Resampling.NEAREST,
        )
        return np.asarray(resized, dtype=np.uint8) > 0

    def _wrist_point_clouds(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._rgbd_point_clouds(
            depth,
            intrinsics,
            mask,
            camera_label="wrist",
        )

    def _rgbd_point_clouds(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
        *,
        camera_label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if depth.ndim != 2:
            raise ValueError(
                f"Expected 2D {camera_label} depth, got shape {depth.shape}"
            )
        mask = self._resize_mask(mask, depth.shape)
        near, far = getattr(self._r1, "graspnet_depth_range", (0.05, 2.0))
        valid = np.isfinite(depth) & (depth >= float(near)) & (depth <= float(far))
        h, w = depth.shape
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            raise RuntimeError(f"{camera_label} depth has no valid points")
        z = depth[ys, xs].astype(np.float64)
        x = (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
        pc_full = np.stack([x, y, z], axis=1)

        seg_valid = valid & mask
        seg_y, seg_x = np.nonzero(seg_valid)
        if len(seg_x) == 0:
            return pc_full, np.zeros((0, 3), dtype=np.float64)
        seg_z = depth[seg_y, seg_x].astype(np.float64)
        seg_pc_x = (seg_x.astype(np.float64) - intrinsics[0, 2]) * seg_z / intrinsics[0, 0]
        seg_pc_y = (seg_y.astype(np.float64) - intrinsics[1, 2]) * seg_z / intrinsics[1, 1]
        pc_segment = np.stack([seg_pc_x, seg_pc_y, seg_z], axis=1)
        return pc_full, pc_segment

    def _level_head_rgbd_to_table(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        base_from_camera: np.ndarray,
        *,
        table_z: float,
        save_debug: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Level one temporally filtered RGB-D observation to the known table."""
        import cv2

        segmentation = self._locate_object_head_sam3(
            rgb,
            prompt="flat tabletop surface",
            timeout_s=60.0,
            min_score=0.80,
            min_pixels=1000,
            save_debug=save_debug,
            slug="head_rgbd_table_level",
            include_mask=True,
        )
        if not segmentation.get("ok"):
            raise RuntimeError(
                "head RGB-D table leveling failed: "
                f"{segmentation.get('reason', segmentation)}"
            )
        table_mask = self._resize_mask(
            np.asarray(segmentation["mask"], dtype=bool), depth.shape
        )
        _full, camera_points = self._rgbd_point_clouds(
            depth,
            intrinsics,
            table_mask,
            camera_label="head_table_level",
        )
        if len(camera_points) > 12000:
            indices = np.linspace(0, len(camera_points) - 1, 12000).astype(int)
            camera_points = camera_points[indices]
        base_points = (
            base_from_camera[:3, :3] @ camera_points.T
        ).T + base_from_camera[:3, 3]
        base_points = base_points[
            np.isfinite(base_points).all(axis=1)
            & (base_points[:, 2] >= float(table_z) - 0.40)
            & (base_points[:, 2] <= float(table_z) + 0.40)
        ]
        if len(base_points) < 300:
            raise RuntimeError("table leveling has fewer than 300 valid points")

        rng = np.random.default_rng(20260830)
        best = np.zeros(len(base_points), dtype=bool)
        threshold_m = 0.012
        for _ in range(2500):
            sample = base_points[rng.choice(len(base_points), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal /= norm
            offset = -float(np.dot(normal, sample[0]))
            inliers = np.abs(base_points @ normal + offset) <= threshold_m
            if int(np.count_nonzero(inliers)) > int(np.count_nonzero(best)):
                best = inliers
        inlier_count = int(np.count_nonzero(best))
        if inlier_count < 300 or inlier_count / len(base_points) < 0.20:
            raise RuntimeError("table leveling did not find a dominant plane")
        pivot = np.median(base_points[best], axis=0)
        _, _, vh = np.linalg.svd(base_points[best] - pivot, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0.0:
            normal *= -1.0
        tilt_deg = float(
            np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
        )
        if tilt_deg > 60.0:
            raise RuntimeError(
                f"table leveling correction is unsafe: {tilt_deg:.2f}deg"
            )

        target = np.array([0.0, 0.0, 1.0])
        axis = np.cross(normal, target)
        sine = float(np.linalg.norm(axis))
        cosine = float(np.dot(normal, target))
        if sine < 1e-12:
            correction = np.eye(3)
        else:
            correction = cv2.Rodrigues(
                axis / sine * np.arctan2(sine, cosine)
            )[0]
        z_shift = float(table_z) - float(pivot[2])
        leveled = np.eye(4, dtype=np.float64)
        leveled[:3, :3] = correction @ base_from_camera[:3, :3]
        leveled[:3, 3] = (
            correction @ (base_from_camera[:3, 3] - pivot)
            + pivot
            + np.array([0.0, 0.0, z_shift])
        )
        return leveled, {
            "method": "sam3_table_plane_rigid_leveling",
            "sam3_score": float(segmentation["score"]),
            "point_count": int(len(base_points)),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_count / len(base_points),
            "normal_base_before": normal.tolist(),
            "correction_angle_deg": tilt_deg,
            "pivot_base_m": pivot.tolist(),
            "z_shift_m": z_shift,
        }

    def _filter_graspnet_segment_by_depth(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        *,
        camera_label: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if depth.ndim != 2:
            raise ValueError(
                f"Expected 2D {camera_label} depth, got shape {depth.shape}"
            )
        mask = self._resize_mask(np.asarray(mask, dtype=bool), depth.shape)
        enabled = bool(
            getattr(self._r1, "graspnet_segment_depth_filter_enabled", True)
        )
        near, far = getattr(self._r1, "graspnet_depth_range", (0.05, 2.0))
        valid = np.isfinite(depth) & (depth >= float(near)) & (depth <= float(far))
        values = np.asarray(depth[valid & mask], dtype=np.float32)
        info: dict[str, Any] = {
            "enabled": enabled,
            "camera_label": camera_label,
            "mask_pixels_before": int(np.count_nonzero(mask)),
            "valid_depth_pixels_before": int(values.size),
        }
        if not enabled or values.size == 0:
            info["mask_pixels_after"] = int(np.count_nonzero(mask))
            info["reason"] = "disabled" if not enabled else "no_valid_depth"
            return mask, info

        anchor_percentile = float(
            getattr(self._r1, "graspnet_segment_depth_anchor_percentile", 10.0)
        )
        lower_margin = float(
            getattr(self._r1, "graspnet_segment_depth_lower_margin_m", 0.02)
        )
        band = float(getattr(self._r1, "graspnet_segment_depth_band_m", 0.12))
        anchor = float(np.percentile(values, anchor_percentile))
        low = max(float(near), anchor - lower_margin)
        high = min(float(far), anchor + band)
        filtered = mask & valid & (depth >= low) & (depth <= high)
        filtered_count = int(np.count_nonzero(filtered))
        min_points = int(getattr(self._r1, "graspnet_min_segment_points", 30))
        info.update(
            {
                "anchor_percentile": anchor_percentile,
                "anchor_depth_m": anchor,
                "low_m": low,
                "high_m": high,
                "mask_pixels_after": filtered_count,
                "depth_min_before": float(np.min(values)),
                "depth_median_before": float(np.median(values)),
                "depth_max_before": float(np.max(values)),
            }
        )
        if filtered_count < min_points:
            info["reason"] = (
                f"filtered depth band left only {filtered_count} points; "
                f"need {min_points}, using original mask"
            )
            info["mask_pixels_after"] = int(np.count_nonzero(mask))
            return mask, info
        info["reason"] = "filtered"
        return filtered, info

    def _subsample_points(self, points: np.ndarray, max_points: int) -> np.ndarray:
        points = np.asarray(points)
        if max_points <= 0 or len(points) <= max_points:
            return points
        idx = np.linspace(0, len(points) - 1, int(max_points)).astype(int)
        return points[idx]

    def _call_graspnet_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import requests

        url = getattr(self._r1, "graspnet_service_url", "http://127.0.0.1:8115")
        payload = {
            "pc_full_base64": self._numpy_to_base64(pc_full),
            "pc_segment_base64": self._numpy_to_base64(pc_segment),
            "segmap_id": 1,
            "local_regions": bool(getattr(self._r1, "graspnet_local_regions", True)),
            "filter_grasps": bool(getattr(self._r1, "graspnet_filter_grasps", False)),
            "forward_passes": int(getattr(self._r1, "graspnet_forward_passes", 1)),
            "max_retries": int(getattr(self._r1, "graspnet_max_retries", 7)),
        }
        session = requests.Session()
        session.trust_env = False
        timeout_s = float(getattr(self._r1, "graspnet_request_timeout_s", 420.0))
        resp = session.post(
            f"{str(url).rstrip('/')}/plan_point_clouds",
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            self._base64_to_numpy(data["grasps_base64"]),
            self._base64_to_numpy(data["scores_base64"]),
            self._base64_to_numpy(data["contact_pts_base64"]),
        )

    def _call_graspnet_depth_segmap(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import requests

        url = getattr(self._r1, "graspnet_service_url", "http://127.0.0.1:8115")
        payload = {
            "depth_base64": self._numpy_to_base64(np.asarray(depth, dtype=np.float32)),
            "cam_K_base64": self._numpy_to_base64(
                np.asarray(intrinsics, dtype=np.float32)
            ),
            "segmap_base64": self._numpy_to_base64(np.asarray(segmap, dtype=np.int32)),
            "segmap_id": 1,
            "local_regions": bool(getattr(self._r1, "graspnet_local_regions", True)),
            "filter_grasps": bool(getattr(self._r1, "graspnet_filter_grasps", False)),
            "skip_border_objects": False,
            "z_range": list(getattr(self._r1, "graspnet_depth_range", (0.05, 2.0))),
            "forward_passes": int(getattr(self._r1, "graspnet_forward_passes", 1)),
            "max_retries": int(getattr(self._r1, "graspnet_max_retries", 7)),
        }
        session = requests.Session()
        session.trust_env = False
        timeout_s = float(getattr(self._r1, "graspnet_request_timeout_s", 420.0))
        resp = session.post(
            f"{str(url).rstrip('/')}/plan",
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            self._base64_to_numpy(data["grasps_base64"]),
            self._base64_to_numpy(data["scores_base64"]),
            self._base64_to_numpy(data["contact_pts_base64"]),
        )

    def _strip_mask(self, result: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in result.items() if k != "mask"}

    def _mask_centroid(self, mask: np.ndarray) -> tuple[np.ndarray | None, int]:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None, 0
        pixel = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
        return pixel, int(len(xs))

    def _best_blob_centroid(
        self,
        mask: np.ndarray,
        *,
        min_pixels: int,
        max_pixels: int,
        prefer_largest: bool = False,
        prefer_lowest: bool = False,
    ) -> tuple[np.ndarray | None, int]:
        try:
            import cv2
        except Exception:
            return self._mask_centroid(mask)

        mask_u8 = np.asarray(mask, dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        h, w = mask.shape[:2]
        center_x = w * 0.5
        best = None
        best_score = -1.0
        best_count = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_pixels or area > max_pixels:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if width <= 0 or height <= 0:
                continue
            aspect = max(width, height) / max(1, min(width, height))
            if aspect > 2.0:
                continue
            compactness = area / float(width * height)
            cx, cy = centroids[label]
            if prefer_largest:
                score = float(area)
            elif prefer_lowest:
                x_penalty = abs(cx - center_x) / max(1.0, w * 0.5)
                y_bonus = cy / max(1.0, h)
                score = compactness + 0.8 * y_bonus - 0.25 * x_penalty - 0.05 * aspect
            else:
                score = compactness - 0.05 * aspect
            if score > best_score:
                best_score = score
                best = np.array(centroids[label], dtype=np.float64)
                best_count = area
        if best is None:
            return None, 0
        return best, best_count

    def _blue_pen_mask(self, rgb: np.ndarray) -> np.ndarray:
        img = np.asarray(rgb, dtype=np.int16)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        return (b > 70) & (b > r + 25) & (b > g + 15)

    def _yellow_tennis_ball_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Detect yellow-green tennis balls under varied desk lighting."""
        img = np.asarray(rgb, dtype=np.float32)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        strict = (
            (g > 105)
            & (r > 70)
            & (b < 135)
            & ((g - b) > 45)
            & ((r - b) > 25)
            & (g > 0.80 * r)
        )
        loose = (
            (g > 90)
            & (r > 55)
            & (b < 150)
            & ((g - b) > 30)
            & ((r + g) > 2 * b + 60)
        )
        mask = strict | loose
        try:
            import cv2

            hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
            hsv_mask = cv2.inRange(hsv, np.array([18, 70, 70]), np.array([48, 255, 255]))
            mask = mask | hsv_mask.astype(bool)
            mask_u8 = mask.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            mask = mask_u8.astype(bool)
        except Exception:
            pass
        return mask

    def _cardboard_box_mask(self, rgb: np.ndarray) -> np.ndarray:
        img = np.asarray(rgb, dtype=np.float32)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        brightness = (r + g + b) / 3.0
        neutral = np.abs(r - g) < 35.0
        brownish = (r > 70.0) & (g > 50.0) & (b < 120.0) & (r >= b) & (g >= b * 0.7)
        return (brightness > 55.0) & (brightness < 210.0) & neutral & brownish
