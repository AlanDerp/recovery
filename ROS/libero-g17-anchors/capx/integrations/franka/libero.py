import os
import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import tip_pose_to_hand_ik_target
from capx.integrations.vision.graspnet import init_contact_graspnet, init_contact_graspnet_point_clouds
from capx.integrations.vision.molmo import init_molmo
from capx.integrations.vision.sam2 import init_sam2_point_prompt
from capx.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_color_to_pointcloud,
    depth_to_pointcloud,
    depth_to_rgb,
)
from capx.integrations.motion.pyroki import init_pyroki

_curobo_api = None


class GraspProposalUnavailable(RuntimeError):
    """The perceptual proposer returned no candidate for this observation."""


def _get_curobo_api():
    """Lazy import of cuRobo API to avoid warp init before Isaac Sim."""
    global _curobo_api
    if _curobo_api is None:
        from capx.integrations.motion import curobo_api as _mod
        _curobo_api = _mod
    return _curobo_api
from sklearn.cluster import DBSCAN

# ------------------------------- Control API ------------------------------
class FrankaLiberoApi(ApiBase):
    """Robot control helpers for Franka.
    """

    # Legacy LIBERO-only offset (diverges from Cap-X DEFAULT_TCP_OFFSET=-0.107).
    # Used only when CAPX_LIBERO_G17_GOTO_TIP_IDENTITY is off.
    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def __init__(self, env: BaseEnv, use_sam3: bool = True) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self._robot = ctx.robot
        self._robot_coll = ctx.robot_coll
        self._target_link_name = ctx.target_link_name
        self._pks = pks

        # Initialize perception models for non-privileged API
        self.use_sam3 = use_sam3
        if self.use_sam3:
            self.sam3_seg_fn = init_sam3()
            self.sam3_point_prompt_fn = init_sam3_point_prompt()
        else:
            self.sam2_point_prompt_fn = init_sam2_point_prompt()
        self.molmo_point_fn = init_molmo()
        self.grasp_net_plan_fn = init_contact_graspnet()
        # used for multiview grasps
        self.grasp_net_plan_point_clouds_fn = init_contact_graspnet_point_clouds()
        self.ik_solve_fn = init_pyroki()
        self.camera_name = "agentview"
        self.wrist_camera_name = "robot0_eye_in_hand"
        self.cfg = None
        self._curobo_world_config = None

    def functions(self) -> dict[str, Any]:
        fns =  {
            "get_observation": self.get_observation,
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_oriented_bounding_box_from_3d_points": self.get_oriented_bounding_box_from_3d_points,
            "get_object_3d_points_and_masks_from_language": self.get_object_3d_points_and_masks_from_language,
            "goto_home_joint_position": self.goto_home_joint_position,
            # # CuRobo, uncomment these for the coding agent to use them!
            # "plan_grasp_trajectory": self.plan_grasp_trajectory,
            # "plan_with_grasped_object": self.plan_with_grasped_object,
            # "execute_joint_trajectory": self.execute_joint_trajectory,
        }
        # Opt-in student exposure. Controllers may call ``self.move_tip_xy``
        # directly even when CAPX_LIBERO_TIP_XY_ONLY is off.
        from capx.integrations.franka.tip_xy_only import tip_xy_only_student_exposed

        if tip_xy_only_student_exposed():
            fns["move_tip_xy"] = self.move_tip_xy
        return fns

    def move_tip_xy(self, dx: float, dy: float) -> None:
        """Move obs tip in world XY only (closed-loop; does not use goto_pose).

        Args:
            dx: World-frame X displacement in meters.
            dy: World-frame Y displacement in meters.
        """
        from capx.verification.libero_g17_live_isolation import action_blocked

        if action_blocked(self, "move_tip_xy"):
            return
        from capx.integrations.franka.tip_xy_only import move_tip_xy as _move

        _move(self, float(dx), float(dy), log=True)

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["agentview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["agentview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32.
                - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["agentview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot0_eye_in_hand"]["images"]["rgb"]: Current wrist camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_eye_in_hand"]["images"]["depth"]: Current wrist camera depth image as a numpy array of shape (H, W), dtype float32.
                - ["robot0_eye_in_hand"]["intrinsics"]: Wrist camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_eye_in_hand"]["pose_mat"]: Wrist camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot_cartesian_pos"]: Current end-effector (panda_hand) pose in the robot/world frame as a numpy array of shape (8,), dtype float64. The first 3 elements are the robot's end-effector XYZ, the next 4 elements are the quaternion wxyz, and the last element is the gripper position normalized, 0 (closed) to 1 (open).
                - ["robot_joint_pos"]: Current joint positions as a numpy array of shape (7,), dtype float64. The last element is the gripper position normalized, 0 (closed) to 1 (open).
        """
        obs = self._env.get_observation()
        obs[self.camera_name]["images"]["depth"] = obs[self.camera_name]["images"]["depth"].squeeze(-1)
        obs[self.wrist_camera_name]["images"]["depth"] = obs[self.wrist_camera_name]["images"]["depth"].squeeze(-1)
        return obs

    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image, optionally conditioned on an image coordinate point prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            point_coords:
                (x, y) pixel coordinates of the point prompt.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3_point_prompt(rgb, (100, 100))
        """
        return self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)

    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image conditioned on a text prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt:
                Text prompt for SAM3 segmentation.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "box": list [x1, y1, x2, y2] in pixel coordinates.
                  - "score": float confidence score.
        Note:
            Returns an empty list if no results are found.

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3(rgb, text_prompt="red mug")
        """
        results = self.sam3_seg_fn(rgb, text_prompt=text_prompt)
        if len(results) == 0:
            print(f"[segment_sam3_text_prompt] SAM3 returned no results for prompt: '{text_prompt}'")
            return []
        return results

    # --------------------------------------------------------------------- #
    # Molmo point prompt
    # --------------------------------------------------------------------- #
    def point_prompt_molmo(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Use Molmo to point to a coordinate in the image based on a text prompt.

        Args:
            image: np.ndarray: The RGB image to process. Shape: (H, W, 3), dtype uint8.
            text_prompt: str: The text prompt to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """
        return self.molmo_point_fn(Image.fromarray(image), objects=[text_prompt])

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """Get the oriented bounding box from 3D points.

        Args:
            points: np.ndarray: The 3D points to get the oriented bounding box from.
                Shape: (N, 3), dtype float64.

        Returns:
            dict[str, Any]: The oriented bounding box. The dictionary contains the following keys:
                - "center": np.ndarray: The center of the oriented bounding box in point cloud frame.
                - "extent": np.ndarray: The extent of the oriented bounding box.
                - "R": np.ndarray: The rotation matrix of the oriented bounding box in point cloud frame.

        Example:
            >>> points = np.random.randn((100, 3))
            >>> obb = get_oriented_bounding_box_from_3d_points(points)
        """
        return _get_obb(points)

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """

        from capx.verification.libero_g17_live_isolation import action_blocked

        if action_blocked(self, "goto_pose"):
            return
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        requested_pos = pos.copy()
        requested_quat_wxyz = quat_wxyz.copy()
        from capx.integrations.franka import acquisition_ownership as _ownership

        pending_transaction = _ownership.current_transaction(self)
        if (
            _ownership.enabled()
            and getattr(self, "_capx_v293_capture_state", None)
            == "acquisition_needed"
            and isinstance(pending_transaction, dict)
            and (
                pending_transaction.get("pregrasp_arm_cfg_rad") is None
                or pending_transaction.get("contact_arm_cfg_rad") is None
            )
            and not _ownership.uses_transparent_native_executor(
                pending_transaction
            )
        ):
            pending_transaction = _ownership.complete_native_transaction(
                self, pending_transaction
            )
            from capx.integrations.franka import acquisition_audit as _acq

            attempt, active_object = _acq.active_attempt(self)
            _acq.audit(
                self,
                "transaction_ownership_native_completed",
                attempt=attempt,
                object_name=active_object,
                transaction_id=pending_transaction.get("transaction_id"),
                completion_source=pending_transaction.get(
                    "completion_source"
                ),
                completion_orientation=pending_transaction.get(
                    "completion_orientation"
                ),
                pregrasp_seed=pending_transaction.get("pregrasp_seed"),
                pregrasp_joint_path_norm_rad=pending_transaction.get(
                    "pregrasp_joint_path_norm_rad"
                ),
                ingress_joint_path_norm_rad=pending_transaction.get(
                    "ingress_joint_path_norm_rad"
                ),
            )

        ownership = _ownership.route_goto(
            self, requested_pos, requested_quat_wxyz, z_approach
        )
        if ownership.mode != "passthrough":
            from capx.integrations.franka import acquisition_audit as _acq

            attempt, active_object = _acq.active_attempt(self)
            _acq.audit(
                self,
                "transaction_ownership_goto",
                attempt=attempt,
                object_name=active_object,
                decision=ownership.mode,
                caller_position=requested_pos.round(7).tolist(),
                caller_quaternion_wxyz=requested_quat_wxyz.round(7).tolist(),
                caller_z_approach_m=float(z_approach),
                canonical_position=ownership.position.round(7).tolist(),
                canonical_quaternion_wxyz=(
                    ownership.quaternion_wxyz.round(7).tolist()
                ),
                position_divergence_m=ownership.position_divergence_m,
                orientation_divergence_rad=(
                    ownership.orientation_divergence_rad
                ),
                physical_command=(
                    ownership.mode == "canonical_transaction"
                ),
            )
        if ownership.mode == "suppress_after_ingress":
            print(
                "[acquisition-ownership] suppress goto after bound ingress; "
                "close must classify the attempt",
                flush=True,
            )
            return
        if ownership.mode == "canonical_transaction":
            pos = ownership.position.copy()
            quat_wxyz = ownership.quaternion_wxyz.copy()
            z_approach = float(ownership.z_approach)
            # Matching and audit below must describe what will physically be
            # executed, not the policy-authored pose that was canonicalized.
            requested_pos = pos.copy()
            requested_quat_wxyz = quat_wxyz.copy()
        v277_canonical_grasp = (
            os.environ.get("CAPX_LIBERO_G17_V277_CANONICAL_GRASP", "0") == "1"
        )
        # Tip→hand for IK:
        # - CAPX_LIBERO_G17_GOTO_TIP_IDENTITY=1: exact inverse of obs TCP
        #   (Rz(π/2), [0,0,-0.107]) so goto(current tip) is a no-op under perfect IK.
        # - else (legacy): tip + R_tip@[0,0,-0.1] with tip quat unchanged (non-identity).
        if os.environ.get("CAPX_LIBERO_G17_GOTO_TIP_IDENTITY", "0") == "1":
            offset_pos, quat_wxyz = tip_pose_to_hand_ik_target(pos, quat_wxyz)
            quat_xyzw = np.array(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                dtype=np.float64,
            )
            rot = SciRotation.from_quat(quat_xyzw)
        else:
            quat_xyzw = np.array(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                dtype=np.float64,
            )
            rot = SciRotation.from_quat(quat_xyzw)
            offset_pos = pos + rot.apply(self._TCP_OFFSET)

        # v278 changes only execution, never the requested grasp/place pose or
        # its legacy tip→hand / z_approach semantics.  Each absolute hand target
        # is reached through live-seeded, FK-validated local SE(3) waypoints.
        from capx.integrations.franka.trusted_se3 import (
            move_hand_pose_trusted,
            trusted_se3_enabled,
        )
        from capx.integrations.franka.reachable_se3 import (
            move_hand_pose_reachable,
            reachable_se3_enabled,
        )
        from capx.integrations.franka.planned_se3 import (
            move_hand_pose_planned,
            planned_se3_enabled,
        )
        from capx.integrations.franka.measured_seed_ik import (
            measured_seed_ik_enabled,
            move_hand_pose_measured_seeded,
            tool_pi_symmetry_enabled,
        )

        if measured_seed_ik_enabled():
            from capx.integrations.franka import acquisition_audit as _acq
            from capx.integrations.franka.measured_seed_ik import (
                _quat_multiply_wxyz,
            )
            from capx.integrations.franka.tip_xy_only import (
                _live_actuated_cfg,
                _quaternion_angle_rad,
                fk_panda_hand,
            )

            attempt, active_object = _acq.active_attempt(self)
            transaction = getattr(
                self, "_capx_v312_acquisition_transaction", None
            )
            transaction_matches = bool(
                isinstance(transaction, dict)
                and transaction.get("attempt") == attempt
                and np.linalg.norm(
                    requested_pos
                    - np.asarray(transaction.get("tip_xyz"), dtype=np.float64)
                )
                <= 0.005
                and abs(
                    float(
                        np.dot(
                            requested_quat_wxyz,
                            np.asarray(
                                transaction.get("tip_wxyz"), dtype=np.float64
                            ),
                        )
                    )
                )
                >= 1.0 - 1e-5
            )
            if (
                transaction_matches
                and transaction.get("pregrasp_arm_cfg_rad") is not None
                and transaction.get("contact_arm_cfg_rad") is not None
                and not transaction.get("executed")
            ):
                # Execute the exact PT-audited IK branches selected with the
                # grasp contact.  Re-solving here was the branch-identity hole
                # that allowed an endpoint-clear top-down recommendation to
                # turn into an elbow-first native command.
                pregrasp_q = np.asarray(
                    transaction["pregrasp_arm_cfg_rad"], dtype=np.float64
                ).reshape(7)
                contact_q = np.asarray(
                    transaction["contact_arm_cfg_rad"], dtype=np.float64
                ).reshape(7)
                execution_count = int(
                    transaction.get("execution_count", 0)
                ) + 1
                if execution_count != 1:
                    transaction["ownership_state"] = "execution_failed"
                    raise _ownership.AcquisitionOwnershipError(
                        "Acquisition transaction is single-execution and "
                        "cannot be replayed"
                    )
                transaction["execution_count"] = execution_count
                transaction["ownership_state"] = "executing"
                transaction_start = _live_actuated_cfg(self)
                transaction["execution_start_arm_cfg_rad"] = (
                    transaction_start[:7].copy()
                )

                def _owned_joint_move(
                    stage_name: str,
                    target_q: np.ndarray,
                    rollback_q: np.ndarray,
                ) -> None:
                    try:
                        self._env.move_to_joints_blocking(
                            target_q, max_steps=600
                        )
                    except Exception:
                        transaction["ownership_state"] = "execution_failed"
                        try:
                            self._env.move_to_joints_blocking(
                                rollback_q, max_steps=600
                            )
                        except Exception as rollback_exc:
                            transaction["rollback_error"] = repr(rollback_exc)
                        _acq.audit(
                            self,
                            "transaction_ownership_execution_failed",
                            attempt=attempt,
                            object_name=active_object,
                            transaction_id=transaction.get("transaction_id"),
                            stage=stage_name,
                            failure="joint_command_exception",
                            rollback_error=transaction.get("rollback_error"),
                        )
                        raise
                _acq.audit(
                    self,
                    "topdown_transaction_command",
                    attempt=attempt,
                    object_name=active_object,
                    caller_z_approach_m=float(z_approach),
                    selected_pregrasp_tip_xyz=transaction.get(
                        "pregrasp_tip_xyz"
                    ),
                    pregrasp_arm_q=pregrasp_q.round(8).tolist(),
                    contact_arm_q=contact_q.round(8).tolist(),
                )
                _owned_joint_move(
                    "pregrasp", pregrasp_q, transaction_start[:7]
                )
                reached_pregrasp = _live_actuated_cfg(self)
                pregrasp_error = float(
                    np.max(np.abs(reached_pregrasp[:7] - pregrasp_q))
                )
                if pregrasp_error > 0.02:
                    transaction["ownership_state"] = "execution_failed"
                    _acq.audit(
                        self,
                        "transaction_ownership_execution_failed",
                        attempt=attempt,
                        object_name=active_object,
                        transaction_id=transaction.get("transaction_id"),
                        stage="pregrasp",
                        failure="endpoint_unreached",
                        joint_error_max_rad=pregrasp_error,
                    )
                    if (
                        _ownership.fg_regeneration_on_execution_failure_enabled()
                    ):
                        release = _ownership.release_for_fg_regeneration(
                            self,
                            transaction,
                            stage="pregrasp",
                            failure="endpoint_unreached",
                        )
                        _acq.audit(
                            self,
                            "transaction_ownership_released_to_fg_regeneration",
                            **release,
                            escape_executed=False,
                            escape_policy="separate_optional_skill",
                        )
                        from capx.integrations.franka import fg_native_retry as _native_retry

                        if _native_retry.enabled():
                            retry_event = _native_retry.ingress_unreached_event(
                                transaction,
                                physical_command_executed=True,
                            )
                            _native_retry.publish(retry_event)
                            _acq.audit(
                                self,
                                "fg_native_retry_preclose_event",
                                retry_event=_native_retry.event_dict(retry_event),
                                hard_route="current_world_FG_regeneration",
                                soft_projection_pending=True,
                            )
                        from capx.execution_control import breakpoint_code_block

                        breakpoint_code_block(
                            "transaction_pregrasp_unreached_fg_regeneration"
                        )
                    self._env.move_to_joints_blocking(
                        transaction_start[:7], max_steps=600
                    )
                    raise _ownership.AcquisitionOwnershipError(
                        "Selected top-down pregrasp branch was not reached"
                    )
                from capx.integrations.franka import acquisition_extract as _extract

                _extract.record_ingress(
                    self,
                    attempt=attempt,
                    object_name=active_object,
                    arm_cfg=reached_pregrasp,
                    hand_target_xyz=np.asarray(
                        transaction["pregrasp_tip_xyz"], dtype=np.float64
                    ),
                    hand_target_wxyz=requested_quat_wxyz,
                )
                transaction["pregrasp_recorded"] = True
                transaction["pregrasp_arm_cfg"] = reached_pregrasp[:7].copy()
                _owned_joint_move(
                    "contact", contact_q, reached_pregrasp[:7]
                )
                reached_contact = _live_actuated_cfg(self)
                contact_joint_error = float(
                    np.max(np.abs(reached_contact[:7] - contact_q))
                )
                actual_pos, actual_quat = fk_panda_hand(self, reached_contact)
                contact_position_error = float(
                    np.linalg.norm(actual_pos - offset_pos)
                )
                contact_orientation_error = _quaternion_angle_rad(
                    actual_quat, quat_wxyz
                )
                from capx.integrations.franka import perceptual_grasp_family as _family

                aligned = bool(
                    contact_joint_error <= 0.02
                    and contact_position_error
                    <= _family.ACQUISITION_POSITION_TOLERANCE_M
                    and contact_orientation_error
                    <= _family.ACQUISITION_ORIENTATION_TOLERANCE_RAD
                )
                _acq.audit(
                    self,
                    "topdown_transaction_contact_result",
                    attempt=attempt,
                    object_name=active_object,
                    ok=aligned,
                    joint_error_max_rad=contact_joint_error,
                    position_error_m=contact_position_error,
                    orientation_error_rad=contact_orientation_error,
                )
                if not aligned:
                    from capx.integrations.franka import transition_closure as _v319

                    if _v319.enabled():
                        near_miss, endpoint_fg = _v319.contact_endpoint_near_miss(
                            self
                        )
                        _acq.audit(
                            self,
                            "transition_contact_endpoint_fg",
                            attempt=attempt,
                            object_name=active_object,
                            numerical_contact_aligned=False,
                            near_miss=near_miss,
                            **endpoint_fg.__dict__,
                        )
                        if near_miss:
                            # The native skill has reached the exact symbolic
                            # boundary required by adaptive GR.  Preserve its
                            # recorded ingress and let close_gripper hand the
                            # event to the frozen v318 controller.
                            transaction["contact_reached"] = True
                            transaction["executed"] = True
                            transaction["ownership_state"] = "ingress_endpoint"
                            transaction["contact_admitted_by_fg"] = True
                            self._capx_v293_last_goal = (
                                np.asarray(offset_pos, dtype=np.float64).copy(),
                                np.asarray(quat_wxyz, dtype=np.float64).copy(),
                            )
                            return
                    transaction["ownership_state"] = "execution_failed"
                    _acq.audit(
                        self,
                        "transaction_ownership_execution_failed",
                        attempt=attempt,
                        object_name=active_object,
                        transaction_id=transaction.get("transaction_id"),
                        stage="contact",
                        failure="endpoint_unreached",
                        joint_error_max_rad=contact_joint_error,
                        position_error_m=contact_position_error,
                        orientation_error_rad=contact_orientation_error,
                    )
                    if (
                        _ownership.fg_regeneration_on_execution_failure_enabled()
                    ):
                        release = _ownership.release_for_fg_regeneration(
                            self,
                            transaction,
                            stage="contact",
                            failure="endpoint_unreached_not_near_miss",
                        )
                        _acq.audit(
                            self,
                            "transaction_ownership_released_to_fg_regeneration",
                            **release,
                            escape_executed=False,
                            escape_policy="separate_optional_skill",
                        )
                        from capx.integrations.franka import fg_native_retry as _native_retry

                        if _native_retry.enabled():
                            retry_event = _native_retry.ingress_unreached_event(
                                transaction,
                                physical_command_executed=True,
                            )
                            _native_retry.publish(retry_event)
                            _acq.audit(
                                self,
                                "fg_native_retry_preclose_event",
                                retry_event=_native_retry.event_dict(retry_event),
                                hard_route="current_world_FG_regeneration",
                                soft_projection_pending=True,
                            )
                        from capx.execution_control import breakpoint_code_block

                        breakpoint_code_block(
                            "transaction_contact_unreached_fg_regeneration"
                        )
                    # Legacy behavior: return to the known pregrasp branch.
                    # v350l intentionally leaves this optional rather than
                    # making escape a prerequisite for retry.
                    self._env.move_to_joints_blocking(
                        reached_pregrasp[:7], max_steps=600
                    )
                    raise _ownership.AcquisitionOwnershipError(
                        "Selected top-down contact branch was not reached"
                    )
                transaction["contact_reached"] = True
                transaction["executed"] = True
                transaction["ownership_state"] = "ingress_endpoint"
                self._capx_v293_last_goal = (
                    np.asarray(offset_pos, dtype=np.float64).copy(),
                    np.asarray(quat_wxyz, dtype=np.float64).copy(),
                )
                return
            motion_phase = (
                "post_capture"
                if getattr(self, "_capx_v293_capture_state", None) == "holding_payload"
                else "approach"
            )
            prior = getattr(self, "_capx_v293_last_goal", None)
            if prior is not None:
                prior_pos, prior_quat = prior
                same_position = bool(np.linalg.norm(offset_pos - prior_pos) <= 1e-6)
                same_orientation = bool(
                    _quaternion_angle_rad(quat_wxyz, prior_quat) <= 1e-6
                )
                if same_position and same_orientation:
                    measured = _live_actuated_cfg(self)
                    actual_pos, actual_quat = fk_panda_hand(self, measured)
                    orientations = [quat_wxyz]
                    if tool_pi_symmetry_enabled():
                        orientations.append(
                            _quat_multiply_wxyz(
                                quat_wxyz, np.array([0.0, 0.0, 0.0, 1.0])
                            )
                        )
                    pos_error = float(np.linalg.norm(actual_pos - offset_pos))
                    ori_error = min(
                        _quaternion_angle_rad(actual_quat, q) for q in orientations
                    )
                    if pos_error <= 0.005 and ori_error <= 0.030:
                        _acq.audit(
                            self,
                            "idempotent_goal_skip",
                            attempt=attempt,
                            object_name=active_object,
                            final_position_error_m=pos_error,
                            final_orientation_error_rad=ori_error,
                        )
                        print(
                            "[acquisition-contract] same measured-aligned goal; "
                            "skip redundant goto_pose",
                            flush=True,
                        )
                        return
            targets: list[tuple[str, np.ndarray]] = []
            if z_approach != 0.0:
                targets.append(
                    (
                        "approach",
                        offset_pos
                        + rot.apply(np.array([0.0, 0.0, -float(z_approach)])),
                    )
                )
            targets.append(("goal", offset_pos))
            results = []
            for stage, hand_target in targets:
                if (
                    stage == "goal"
                    and motion_phase == "approach"
                    and isinstance(transaction, dict)
                    and transaction_matches
                    and not transaction.get("pregrasp_recorded")
                ):
                    # Explicit two-goto and zero-z_approach programs still
                    # need an exact inverse boundary. Record the measured
                    # state immediately before the physical contact command.
                    from capx.integrations.franka import acquisition_extract as _extract

                    measured_pregrasp = _live_actuated_cfg(self)
                    observed_pregrasp_tip = np.asarray(
                        self.get_observation()["robot_cartesian_pos"][:3],
                        dtype=np.float64,
                    )
                    _extract.record_ingress(
                        self,
                        attempt=attempt,
                        object_name=active_object,
                        arm_cfg=measured_pregrasp,
                        hand_target_xyz=observed_pregrasp_tip,
                        hand_target_wxyz=requested_quat_wxyz,
                    )
                    transaction["pregrasp_recorded"] = True
                    transaction["pregrasp_arm_cfg"] = measured_pregrasp[:7].copy()
                    transaction["pregrasp_tip_xyz"] = observed_pregrasp_tip.copy()
                _acq.audit(
                    self,
                    f"{motion_phase}_command",
                    attempt=attempt,
                    object_name=active_object,
                    stage=stage,
                    target_xyz=np.asarray(hand_target).round(6).tolist(),
                )
                if stage == "approach":
                    preview = move_hand_pose_measured_seeded(
                        self, hand_target, quat_wxyz, log=True, execute=False
                    )
                    if not preview.ok:
                        results.append(("approach_preview", preview))
                        _acq.audit(
                            self,
                            "approach_skipped",
                            attempt=attempt,
                            object_name=active_object,
                            reason="predicted_endpoint_inaccurate",
                            predicted_position_error_m=preview.final_position_error_m,
                            predicted_orientation_error_rad=preview.final_orientation_error_rad,
                        )
                        continue
                recovery_armed = bool(
                    getattr(self, "_capx_v293_capture_state", None)
                    == "acquisition_needed"
                )
                self._env._capx_v297b_acquisition_recovery_armed = recovery_armed
                try:
                    result = move_hand_pose_measured_seeded(
                        self, hand_target, quat_wxyz, log=True
                    )
                finally:
                    self._env._capx_v297b_acquisition_recovery_armed = False
                results.append((stage, result))
                _acq.audit(
                    self,
                    f"{motion_phase}_result",
                    attempt=attempt,
                    object_name=active_object,
                    stage=stage,
                    ok=bool(result.ok),
                    status=result.status,
                    final_position_error_m=result.final_position_error_m,
                    final_orientation_error_rad=result.final_orientation_error_rad,
                    candidate_arm_delta_rad=result.candidate_arm_delta_rad,
                    measured_arm_delta_rad=result.measured_arm_delta_rad,
                    pt_swept_path=result.pt_swept_path,
                    joint_move_report=getattr(
                        self._env, "_last_joint_move_report", None
                    ),
                )
                if motion_phase == "post_capture":
                    attachment_after = _acq.audit_attachment(
                        self, "lift_verification", stage=stage
                    )
                    from capx.integrations.franka import transition_closure as _v319

                    if _v319.enabled() and _v319.transport_armed(self):
                        transition = _v319.post_capture_transition(
                            attachment_after, bool(result.ok)
                        )
                        _acq.audit(
                            self,
                            "transition_native_transport_boundary",
                            attempt=attempt,
                            object_name=active_object,
                            stage=stage,
                            attachment=attachment_after,
                            motion_ok=bool(result.ok),
                            transition=transition,
                            transport_state=getattr(
                                self, "_capx_v320_transport_state", None
                            ),
                            symbolic_obligation=_v319.symbolic_obligation(
                                self
                            ),
                        )
                        if transition == "payload_lost_return_start":
                            _v319.disarm_transport(self)
                            from capx.integrations.franka import fg_native_retry as _native_retry

                            stale = _ownership.current_transaction(self)
                            if (
                                _native_retry.clean_c_enabled()
                                and isinstance(stale, dict)
                            ):
                                release = _ownership.release_for_fg_regeneration(
                                    self,
                                    stale,
                                    stage="payload_lost_after_firm_transport",
                                    failure="payload_lost",
                                )
                                _acq.audit(
                                    self,
                                    "fg_clean_c_transaction_released",
                                    attempt=attempt,
                                    object_name=active_object,
                                    release=release,
                                    next_state="fresh_FG_regeneration",
                                )
                            else:
                                self._capx_v293_capture_state = "acquisition_needed"
                            from capx.execution_control import breakpoint_code_block

                            breakpoint_code_block(
                                "payload_lost_during_native_transport"
                            )
                        # A measured endpoint miss is not an FG failure.  If
                        # attachment remains FirmAcquired, preserve the native
                        # program and its symbolic PlacementNearMiss obligation.
                        # The following native goto(s) and RR own the rest.
                if (
                    stage == "approach"
                    and result.ok
                    and getattr(self, "_capx_v293_capture_state", None)
                    == "acquisition_needed"
                ):
                    from capx.integrations.franka import acquisition_extract as _extract

                    _extract.record_ingress(
                        self,
                        attempt=attempt,
                        object_name=active_object,
                        arm_cfg=_live_actuated_cfg(self),
                        hand_target_xyz=hand_target,
                        hand_target_wxyz=quat_wxyz,
                    )
                    if transaction_matches:
                        transaction["pregrasp_recorded"] = True
                        transaction["pregrasp_arm_cfg"] = _live_actuated_cfg(
                            self
                        )[:7].copy()
                        transaction["pregrasp_tip_xyz"] = np.asarray(
                            self.get_observation()["robot_cartesian_pos"][:3],
                            dtype=np.float64,
                        ).copy()
                move_report = getattr(self._env, "_last_joint_move_report", None)
                if (
                    isinstance(move_report, dict)
                    and move_report.get("exit_reason")
                    == "acquisition_strong_stall"
                ):
                    raise RuntimeError(
                        "Acquisition motion stalled and was rolled back along "
                        "measured joint history; regenerate a different grasp"
                    )
                if stage == "goal":
                    self._capx_v293_last_goal = (
                        np.asarray(offset_pos, dtype=np.float64).copy(),
                        np.asarray(quat_wxyz, dtype=np.float64).copy(),
                    )
                    if transaction_matches and (
                        result.ok
                        or _ownership.uses_transparent_native_executor(
                            transaction
                        )
                    ):
                        transaction["ingress_attempted"] = True
                        transaction["native_motion_ok"] = bool(result.ok)
                        # Certified execution uses numerical endpoint reach as
                        # its command postcondition.  A transparent native
                        # attempt instead proceeds to close even when its
                        # measured endpoint is imperfect; the outer FG
                        # observation owns Firm/NearMiss/Miss classification.
                        transaction["contact_reached"] = bool(result.ok)
                        transaction["executed"] = True
                        transaction["ownership_state"] = (
                            "ingress_endpoint"
                            if result.ok
                            else "native_ingress_attempted"
                        )
                        _acq.audit(
                            self,
                            "transparent_native_ingress_completed",
                            attempt=attempt,
                            object_name=active_object,
                            transaction_id=transaction.get(
                                "transaction_id"
                            ),
                            native_motion_ok=bool(result.ok),
                            final_position_error_m=(
                                result.final_position_error_m
                            ),
                            final_orientation_error_rad=(
                                result.final_orientation_error_rad
                            ),
                            next_state="close_then_FG_classification",
                        )
            self._capx_last_measured_seed_ik = results
            return

        if (
            trusted_se3_enabled()
            or reachable_se3_enabled()
            or planned_se3_enabled()
        ):
            if planned_se3_enabled():
                mover = move_hand_pose_planned
            elif reachable_se3_enabled():
                mover = move_hand_pose_reachable
            else:
                mover = move_hand_pose_trusted
            targets: list[tuple[str, np.ndarray]] = []
            if z_approach != 0.0:
                targets.append(
                    (
                        "approach",
                        offset_pos
                        + rot.apply(np.array([0.0, 0.0, -float(z_approach)])),
                    )
                )
            targets.append(("goal", offset_pos))
            results = []
            for stage, hand_target in targets:
                result = mover(
                    self, hand_target, quat_wxyz, log=True
                )
                results.append((stage, result))
                if not result.ok:
                    print(
                        f"[trusted-se3] goto_pose stopped stage={stage} "
                        f"{result.message}",
                        flush=True,
                    )
                    break
            self._capx_last_trusted_se3 = results
            return

        if (
            z_approach != 0.0
        ):  # If z_approach is not 0.0, approach the object from above by z_approach meters
            if v277_canonical_grasp:
                # v277 counterfactual: canonical top-down grasp and world-Z pregrasp.
                z_offset_pos = offset_pos + np.array([0.0, 0.0, z_approach])
            else:
                z_offset_pos = offset_pos + rot.apply(
                    np.array([0, 0, -z_approach])
                )

            if self.cfg is None:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                )
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg,
                )
            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking(joints_z_offset)

        if self.cfg is None:
            self.cfg = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
            )
        else:
            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        from capx.verification.libero_g17_live_isolation import (
            action_blocked,
            arm_success_latch_for_release,
            audit as _v270_audit,
            disarm_success_latch_after_release,
            enabled as _v270_enabled,
        )

        if action_blocked(self, "open_gripper"):
            return
        if _v270_enabled():
            _v270_audit(self._env, "release_intent", decision="pending")
        # Fair place servo (v253): get_object_pose + tip_xy; never refuse; no GT.
        try:
            from capx.verification import libero_g17_fair_place_servo as _fps

            if _fps.fair_place_servo_enabled():
                _fps.run_fair_place_servo(self)
                arm_success_latch_for_release(self._env)
                self._env._set_gripper(1.0)
                for _ in range(40):
                    self._env._step_once()
                from capx.integrations.franka import transition_closure as _v319

                if _v319.enabled():
                    _v319.disarm_transport(self)
                disarm_success_latch_after_release(self._env)
                return
        except Exception as _fps_exc:
            print(f"[fair-place] gate error ({_fps_exc!r}); opening anyway", flush=True)

        # Near-miss fail-open (v252): tip_xy nudge only on clear near-miss; never refuse.
        try:
            from capx.verification import libero_g17_nearmiss_failopen as _nm

            if _nm.nearmiss_failopen_enabled():
                _nm.run_nearmiss_tipxy_failopen(self)
                arm_success_latch_for_release(self._env)
                self._env._set_gripper(1.0)
                for _ in range(40):
                    self._env._step_once()
                from capx.integrations.franka import transition_closure as _v319

                if _v319.enabled():
                    _v319.disarm_transport(self)
                disarm_success_latch_after_release(self._env)
                return
        except Exception as _nm_exc:
            print(f"[nearmiss-fo] gate error ({_nm_exc!r}); opening anyway", flush=True)

        # ReleaseReady (v244+): may refuse open when not release_ok.
        try:
            from capx.verification import libero_g17_release_ready as _rr

            if _rr.release_ready_enabled():
                if not _rr.maybe_xy_align_before_release(self):
                    if _v270_enabled():
                        _v270_audit(
                            self._env, "release_gate_result", decision="refuse"
                        )
                    print(
                        "[release-ready] open_gripper refused: not fully release_ok",
                        flush=True,
                    )
                    return
        except Exception as _rr_exc:
            if _v270_enabled():
                _v270_audit(
                    self._env,
                    "release_gate_error",
                    error=repr(_rr_exc),
                    decision="refuse",
                )
                print(
                    f"[release-ready] gate error ({_rr_exc!r}); "
                    "refuse open (v270 fail-closed)",
                    flush=True,
                )
                return
            # Pre-existing fail-open on any gate error. v261 no-Z burns horizon and
            # hits "terminated episode" more often — differential: refuse open only
            # under NO_Z_ABORT so v260 Z-abort path keeps historical fail-open.
            exc_s = repr(_rr_exc).lower()
            no_z = (
                __import__("os").environ.get(
                    "CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT", "0"
                )
                == "1"
            )
            if no_z and "terminated episode" in exc_s:
                print(
                    f"[release-ready] gate error ({_rr_exc!r}); "
                    "refuse open (v262 no-Z differential; no fail-open)",
                    flush=True,
                )
                return
            print(f"[release-ready] gate error ({_rr_exc!r}); opening anyway", flush=True)
        if _v270_enabled():
            _v270_audit(self._env, "release_gate_result", decision="allow")
        arm_success_latch_for_release(self._env)
        self._env._set_gripper(1.0)
        for _ in range(40):
            self._env._step_once()
        from capx.integrations.franka import transition_closure as _v319

        if _v319.enabled():
            _v319.disarm_transport(self)
        disarm_success_latch_after_release(self._env)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        from capx.verification.libero_g17_live_isolation import (
            action_blocked,
            disarm_success_latch_after_release,
        )

        if action_blocked(self, "close_gripper"):
            return
        disarm_success_latch_after_release(self._env)
        from capx.integrations.franka import acquisition_audit as _acq

        transaction_enabled = (
            os.environ.get(
                "CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION", "0"
            )
            == "1"
        )
        transaction = getattr(self, "_capx_v312_acquisition_transaction", None)
        from capx.integrations.franka import acquisition_ownership as _ownership

        owned_transaction = _ownership.require_close_transaction(self)
        transparent_native_attempted = bool(
            _ownership.uses_transparent_native_executor(owned_transaction)
            and _ownership.ingress_attempt_completed(owned_transaction)
        )
        if (
            owned_transaction is not None
            and not owned_transaction.get("contact_reached")
            and not transparent_native_attempted
        ):
            # ``close`` is the semantic acquisition boundary.  Generated code
            # is allowed to omit or mangle its preceding goto, but it cannot
            # take continuous control away from the bound SL transaction.
            # Finish the canonical ingress before closing.  This is endpoint
            # execution, not a requirement for observed physical contact.
            _acq.audit(
                self,
                "transaction_ownership_close_completes_ingress",
                attempt=owned_transaction.get("attempt"),
                object_name=owned_transaction.get("object_name"),
                transaction_id=owned_transaction.get("transaction_id"),
                physical_contact_required=False,
            )
            self.goto_pose(
                np.asarray(owned_transaction["tip_xyz"], dtype=np.float64),
                np.asarray(owned_transaction["tip_wxyz"], dtype=np.float64),
                z_approach=0.10,
            )
            transaction = getattr(
                self, "_capx_v312_acquisition_transaction", None
            )
            if not (
                isinstance(transaction, dict)
                and _ownership.ingress_attempt_completed(transaction)
            ):
                raise _ownership.AcquisitionOwnershipError(
                    "Bound acquisition ingress did not reach its endpoint; "
                    "close was not physically executed"
                )
        if transaction_enabled and _acq.enabled():
            from capx.integrations.franka import postclose_acquisition as _v331

            attempt, object_name = _acq.active_attempt(self)
            ingress = getattr(self, "_capx_v304_ingress", None)
            ready = bool(
                isinstance(transaction, dict)
                and transaction.get("attempt") == attempt
                and transaction.get("pregrasp_recorded") is True
                and _ownership.ingress_attempt_completed(transaction)
                and isinstance(ingress, dict)
                and ingress.get("attempt") == attempt
            )
            _acq.audit(
                self,
                "topdown_transaction_close_gate",
                attempt=attempt,
                object_name=object_name,
                decision=(
                    _v331.close_gate_decision(certificate_ready=ready)
                    if _v331.enabled()
                    else ("allow" if ready else "regenerate_before_close")
                ),
                pregrasp_recorded=(
                    transaction.get("pregrasp_recorded")
                    if isinstance(transaction, dict)
                    else False
                ),
                contact_reached=(
                    transaction.get("contact_reached")
                    if isinstance(transaction, dict)
                    else False
                ),
                ingress_attempted=(
                    transaction.get("ingress_attempted")
                    if isinstance(transaction, dict)
                    else False
                ),
                ingress_present=isinstance(ingress, dict),
                physical_contact_required=False,
            )
            if not ready and _ownership.enabled():
                raise _ownership.AcquisitionOwnershipError(
                    "Owned acquisition transaction is incomplete; close was "
                    "not physically executed"
                )
            if not ready and not _v331.enabled():
                raise RuntimeError(
                    "Top-down acquisition transaction is incomplete; "
                    "regenerate a grasp and execute pregrasp→contact before close"
                )
        from capx.integrations.franka import grasp_recovery as _gr_recovery
        from capx.integrations.franka import fg_native_retry as _native_retry

        before_close_fg = None
        if (
            (_gr_recovery.enabled() or _native_retry.enabled())
            and isinstance(transaction, dict)
        ):
            before_close_fg = _gr_recovery.capture_close_time_fg(self)
        self._env._set_gripper(0.0)
        for _ in range(60):
            self._env._step_once()
        if isinstance(transaction, dict):
            transaction["ownership_state"] = "close_observed"

        close_fg = None
        gr_near_miss = False
        close_outcome = None
        native_retry_event = None
        native_retry_route = None
        if (
            (_gr_recovery.enabled() or _native_retry.enabled())
            and isinstance(transaction, dict)
        ):
            close_fg = _gr_recovery.capture_close_time_fg(self)
            if before_close_fg is not None:
                close_fg = _gr_recovery.latch_grasp_event(
                    before_close_fg, close_fg
                )
            attempt, object_name = _acq.active_attempt(self)
            gr_near_miss = (
                close_fg.adaptive_near_miss
                if _gr_recovery.adaptive_enabled()
                else close_fg.near_miss
            )
            _acq.audit(
                self,
                "gr_recovery_close_time_fg",
                attempt=attempt,
                object_name=object_name,
                before_close_fg=before_close_fg.__dict__,
                **close_fg.__dict__,
                near_miss=gr_near_miss,
                near_miss_definition=(
                    "z_aligned_target_contact_empty"
                    if _gr_recovery.adaptive_enabled()
                    else "v317_contact_or_inside_empty"
                ),
            )

            from capx.integrations.franka import postclose_acquisition as _v331

            if _v331.enabled() and not _native_retry.enabled():
                close_outcome = _v331.classify_close(
                    firm_acquired=close_fg.firm_acquired,
                    near_miss=gr_near_miss,
                )
                _acq.audit(
                    self,
                    "postclose_acquisition_transition",
                    attempt=attempt,
                    object_name=object_name,
                    outcome=close_outcome.value,
                    next_state=(
                        "closed_transaction_router"
                        if _v331.closed_router_enabled()
                        else {
                            _v331.CloseOutcome.FIRM_ACQUIRED: (
                                "inverse_escape_holding"
                            ),
                            _v331.CloseOutcome.NEAR_MISS: (
                                _v331.near_miss_next_state()
                            ),
                            _v331.CloseOutcome.DEFINITE_MISS: (
                                "inverse_escape_then_regen"
                            ),
                        }[close_outcome]
                    ),
                )

            if _native_retry.enabled():
                from capx.integrations.franka import acquisition_extract as _extract

                native_retry_event = _native_retry.classify_endpoint(
                    close_fg,
                    transaction,
                    inverse_ingress_available=_extract.has_current_ingress(self),
                )
                native_retry_route = _native_retry.route_endpoint(
                    native_retry_event
                )
                _acq.audit(
                    self,
                    "fg_native_retry_endpoint",
                    attempt=attempt,
                    object_name=object_name,
                    endpoint=_native_retry.event_dict(native_retry_event),
                    route={
                        "open_gripper": native_retry_route.open_gripper,
                        "execute_inverse": native_retry_route.execute_inverse,
                        "execute_gr": native_retry_route.execute_gr,
                        "regenerate": native_retry_route.regenerate,
                        "admit_transport": native_retry_route.admit_transport,
                        "next_state": native_retry_route.next_state,
                    },
                    hard_projection="closed_endpoint_FG_to_total_retry_route",
                    gr_executed=False,
                    continuous_privileged_world_input=False,
                )

        if _acq.enabled():
            attachment = _acq.audit_attachment(self, "capture_result")
            self._capx_v293_capture_state = attachment or "capture_unknown"
        from capx.integrations.franka import acquisition_extract as _extract

        # Any non-payload close must not drag an object while the exact ingress
        # is reversed. This covers NearMiss, empty miss, and wrong payload.
        if (
            native_retry_route is not None
            and native_retry_route.open_gripper
        ) or (
            native_retry_route is None
            and close_fg is not None
            and not close_fg.firm_acquired
        ):
            self._env._set_gripper(1.0)
            for _ in range(40):
                self._env._step_once()
        close_route = None
        if (
            _v331.enabled()
            and _v331.closed_router_enabled()
            and close_outcome is not None
            and not _native_retry.enabled()
        ):
            close_route = _v331.route_close(
                close_outcome,
                inverse_ingress_available=_extract.has_current_ingress(self),
            )
            _acq.audit(
                self,
                "closed_acquisition_transaction_route",
                attempt=attempt,
                object_name=object_name,
                outcome=close_route.outcome.value,
                execute_escape=close_route.execute_escape,
                escape_use=close_route.escape_use.value,
                execute_gr=close_route.execute_gr,
                regenerate=close_route.regenerate,
                next_state=close_route.next_state,
                route_inputs=(
                    "endpoint_FG_outcome+inverse_ingress_available_bool"
                ),
                continuous_privileged_world_input=False,
            )
        if native_retry_route is not None:
            retreat = (
                _extract.retreat_after_close(self)
                if native_retry_route.execute_inverse
                else None
            )
        else:
            retreat = (
                _extract.retreat_after_close(self)
                if close_route is None or close_route.execute_escape
                else None
            )
        recovery_result = None
        clean_c_gr_attempted = False
        clean_c_gr_error = None
        from capx.integrations.franka import postclose_acquisition as _v331

        native_retry_ablation = _v331.native_retry_ablation_enabled()
        if (
            close_fg is not None
            and native_retry_route is not None
            and native_retry_route.execute_gr
            and retreat is not None
        ):
            clean_c_gr_attempted = True
            try:
                recovery_result = _gr_recovery.execute_retry_after_escape(
                    self, transaction, close_fg
                )
                retry_fg = recovery_result["retry_fg"]
                retry_attachment = _acq.audit_attachment(
                    self, "gr_recovery_capture_result"
                )
                self._capx_v293_capture_state = (
                    retry_attachment or "capture_unknown"
                )
                native_retry_event = _native_retry.classify_endpoint(
                    retry_fg,
                    transaction,
                    inverse_ingress_available=(
                        recovery_result.get("retreat") is not None
                    ),
                )
                native_retry_route = _native_retry.route_endpoint(
                    native_retry_event, gr_already_attempted=True
                )
                retreat = recovery_result.get("retreat")
            except Exception as exc:
                # GR is a bounded NearMiss consumer, never the full-domain
                # fallback.  An unavailable correction returns control to
                # native retry from the actual post-inverse world.
                clean_c_gr_error = f"{type(exc).__name__}: {exc}"
                gr_failure_retreat = (
                    _extract.retreat_after_close(self)
                    if _extract.has_current_ingress(self)
                    else None
                )
                retreat = gr_failure_retreat
                recovered_fg = _gr_recovery.capture_close_time_fg(self)
                native_retry_event = _native_retry.classify_endpoint(
                    recovered_fg,
                    transaction,
                    inverse_ingress_available=gr_failure_retreat is not None,
                )
                native_retry_route = _native_retry.route_endpoint(
                    native_retry_event, gr_already_attempted=True
                )
            _acq.audit(
                self,
                "fg_clean_c_gr_transition",
                attempt=attempt,
                object_name=object_name,
                source_endpoint="empty_target_near",
                gr_attempted=True,
                gr_error=clean_c_gr_error,
                result_endpoint=(
                    _native_retry.event_dict(native_retry_event)
                    if native_retry_event is not None
                    else None
                ),
                next_state=native_retry_route.next_state,
                native_retry_fallback=native_retry_route.regenerate,
            )
        if (
            close_fg is not None
            and gr_near_miss
            and retreat is not None
            and not native_retry_ablation
            and not _native_retry.enabled()
        ):
            recovery_result = _gr_recovery.execute_retry_after_escape(
                self, transaction, close_fg
            )
            retry_attachment = _acq.audit_attachment(
                self, "gr_recovery_capture_result"
            )
            self._capx_v293_capture_state = (
                retry_attachment or "capture_unknown"
            )
            from capx.integrations.franka import postclose_acquisition as _v331

            if _v331.enabled():
                retry_fg = recovery_result["retry_fg"]
                close_outcome = _v331.classify_close(
                    firm_acquired=bool(retry_fg.firm_acquired),
                    near_miss=bool(retry_fg.adaptive_near_miss),
                )
                _acq.audit(
                    self,
                    "postclose_gr_transition",
                    attempt=attempt,
                    object_name=object_name,
                    outcome=close_outcome.value,
                    next_state=(
                        "escaped_holding"
                        if close_outcome == _v331.CloseOutcome.FIRM_ACQUIRED
                        else "native_regeneration"
                    ),
                )

        if (
            close_fg is not None
            and gr_near_miss
            and retreat is not None
            and native_retry_ablation
            and not _native_retry.enabled()
        ):
            _acq.audit(
                self,
                "postclose_near_miss_native_regeneration",
                attempt=attempt,
                object_name=object_name,
                terminal_outcome=close_outcome.value,
                inverse_escape_completed=True,
                gr_executed=False,
                next_observation_source=(
                    "fresh_rgbd_segmentation_graspnet_qwen_regeneration"
                ),
                discarded_close_time_control_signal=(
                    "close_time_contact_side_fk_axis_local_gr_correction"
                ),
            )
        elif (
            close_fg is not None
            and gr_near_miss
            and native_retry_ablation
            and close_route is not None
            and close_route.regenerate
            and not _native_retry.enabled()
        ):
            _acq.audit(
                self,
                "postclose_near_miss_native_regeneration",
                attempt=attempt,
                object_name=object_name,
                terminal_outcome=close_outcome.value,
                inverse_escape_completed=False,
                gr_executed=False,
                next_observation_source=(
                    "current_world_FG_then_native_retry"
                ),
                discarded_close_time_control_signal=(
                    "close_time_contact_side_fk_axis_local_gr_correction"
                ),
            )

        from capx.integrations.franka import postclose_acquisition as _v331

        if native_retry_event is not None:
            native_retry_event = _native_retry.finalize_recovery(
                native_retry_event,
                inverse_completed=retreat is not None,
            )
            _native_retry.publish(native_retry_event)
            _acq.audit(
                self,
                "fg_native_retry_recovery_complete",
                attempt=attempt,
                object_name=object_name,
                endpoint=_native_retry.event_dict(native_retry_event),
                next_observation="fresh_closed_FG",
                soft_projection_pending=True,
                clean_c_gr_attempted=clean_c_gr_attempted,
                clean_c_gr_error=clean_c_gr_error,
            )

        if (
            _v331.enabled()
            and close_outcome is not None
            and close_outcome != _v331.CloseOutcome.FIRM_ACQUIRED
            and (
                not gr_near_miss
                or recovery_result is not None
                or (native_retry_ablation and retreat is not None)
                or (close_route is not None and close_route.regenerate)
            )
        ):
            if _ownership.enabled():
                # Keep ownership active across regeneration.  v332 stored the
                # attachment label (usually ``empty``) here, which reopened
                # generic goto_pose in the next code block before a fresh
                # sample_grasp_pose.  A failed close is still AcquireNeeded.
                self._capx_v293_capture_state = "acquisition_needed"
            _acq.audit(
                self,
                "postclose_definite_miss_regeneration",
                attempt=attempt,
                object_name=object_name,
                terminal_outcome=close_outcome.value,
                inverse_escape_completed=retreat is not None,
            )
            from capx.execution_control import breakpoint_code_block

            breakpoint_code_block(
                "fg_native_retry_regeneration"
                if native_retry_route is not None
                else "acquisition_endpoint_FG_regeneration"
                if close_route is not None and close_route.regenerate
                else (
                    "near_miss_after_inverse_escape_native_retry_ablation"
                    if gr_near_miss and native_retry_ablation
                    else "definite_miss_after_inverse_escape"
                )
            )
        elif (
            native_retry_route is not None
            and native_retry_route.regenerate
        ):
            if _ownership.enabled():
                self._capx_v293_capture_state = "acquisition_needed"
                if _native_retry.clean_c_enabled():
                    stale = _ownership.current_transaction(self)
                    if isinstance(stale, dict):
                        release = _ownership.release_for_fg_regeneration(
                            self,
                            stale,
                            stage="closed_endpoint_retry_boundary",
                            failure=native_retry_event.kind.value,
                        )
                        _acq.audit(
                            self,
                            "fg_clean_c_transaction_released",
                            attempt=attempt,
                            object_name=object_name,
                            release=release,
                            next_state="fresh_FG_regeneration",
                        )
            from capx.execution_control import breakpoint_code_block

            breakpoint_code_block("fg_native_retry_regeneration")
        from capx.integrations.franka import transition_closure as _v319

        if _v319.enabled():
            transport_armed = _v319.arm_transport(
                self,
                getattr(self, "_capx_v293_capture_state", None),
                escaped=retreat is not None,
            )
            if _v319.semantic_enabled():
                attempt, object_name = _acq.active_attempt(self)
                _acq.audit(
                    self,
                    "transition_firm_acquired_escaped",
                    attempt=attempt,
                    object_name=object_name,
                    attachment=getattr(
                        self, "_capx_v293_capture_state", None
                    ),
                    inverse_ingress_completed=retreat is not None,
                    transport_armed=transport_armed,
                    transport_state=getattr(
                        self, "_capx_v320_transport_state", None
                    ),
                    symbolic_obligation=_v319.symbolic_obligation(self),
                )
        if transaction_enabled:
            attempt, object_name = _acq.active_attempt(self)
            _acq.audit(
                self,
                "topdown_transaction_escape",
                attempt=attempt,
                object_name=object_name,
                completed=retreat is not None,
                recovery_attempted=recovery_result is not None,
            )
            self._capx_v312_acquisition_transaction = None
            if retreat is None and not _v331.closed_router_enabled():
                raise RuntimeError(
                    "Top-down close completed without its paired inverse ingress"
                )
        # v309: closing the gripper completes one acquisition transaction.
        # Always return control to the FG loop after the physical close and
        # optional inverse-ingress extraction.  The next snapshot, rather
        # than stale code generated before contact, decides whether the result
        # is FirmAcquired, wrong-object, empty, or lost-during-extraction.
        # No private attachment value gates this boundary.
        if (
            os.environ.get(
                "CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY", "0"
            )
            == "1"
            and _acq.enabled()
            and _acq.active_attempt(self)[0] is not None
        ):
            attempt, object_name = _acq.active_attempt(self)
            _acq.audit(
                self,
                "acquisition_event_boundary",
                attempt=attempt,
                object_name=object_name,
                source="physical_close_plus_optional_inverse_ingress",
            )
            from capx.execution_control import breakpoint_code_block

            breakpoint_code_block("acquisition_transition_complete")

    def get_object_pose(
        self,
        object_name: str,
        use_multiview: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp
        pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for
        placement position.
        It is possible that get_object_pose is sometimes unreliable and will return None for both
        position and quaternion.

        Args:
            object_name: The name of the object to get the pose of, in underscore separated lowercase words.
            use_multiview: If True, uses the wrist camera as well as the main camera for segmentation. If False, only uses the main camera for segmentation.

        Returns:
            position: (3,) XYZ in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
        """
        start_time = time.time()

        # Same-name disambiguation is skill-internal (FG live RelXY token); not a public arg.
        result = self._resolve_typed_instance(
            object_name, use_multiview=use_multiview, which=None
        )
        points_3d = result["points_3d"]

        if len(points_3d) == 0:
            return None, None

        points_3d, _ = self.filter_noise(points_3d)
        if len(points_3d) == 0:
            return None, None

        obb = self.get_oriented_bounding_box_from_3d_points(points_3d)

        position = np.array(obb["center"])
        R = np.array(obb["R"])

        # Ensure z-axis points down for gripper frame consistency
        z_axis_world = R[:, 2]
        if z_axis_world[2] > 0:
            R = R @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        quaternion_wxyz = vtf.SO3.from_matrix(R).wxyz

        print(f"get_object_pose in {time.time() - start_time} seconds")
        return position, quaternion_wxyz

    def sample_grasp_pose(
        self,
        object_name: str,
        use_multiview: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Uses multiview point clouds and plan_grasp_from_point_clouds for more reliable grasp planning.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for, in underscore separated lowercase words.
            use_multiview: If True, uses the wrist camera as well as the main camera for segmentation. If False, only uses the main camera for segmentation.

        Returns:
            position: (3,) XYZ in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
        """
        start_time = time.time()

        from capx.integrations.franka import acquisition_audit as _acq

        attempt = _acq.begin_attempt(self, object_name) if _acq.enabled() else None

        # Same-name disambiguation is skill-internal (FG live RelXY token); not a public arg.
        result = self._resolve_typed_instance(
            object_name, use_multiview=use_multiview, which=None
        )

        # Optional: freeze GraspNet outputs from a donor arm (v256) by call index.
        try:
            from capx.verification import libero_g17_grasp_replay as _grasp_replay

            if _grasp_replay.enabled():
                frozen = _grasp_replay.next_pose(object_name=object_name)
                if frozen is not None:
                    frozen_quat = frozen.quaternion_wxyz
                    if (
                        os.environ.get(
                            "CAPX_LIBERO_G17_V277_CANONICAL_GRASP", "0"
                        )
                        == "1"
                    ):
                        frozen_quat = np.array(
                            [0.0, 0.0, 1.0, 0.0], dtype=np.float64
                        )
                    # Match live GraspNet student-visible lines (no replay metadata).
                    print(
                        f"sample_grasp_pose in {time.time() - start_time} seconds",
                        flush=True,
                    )
                    print(
                        f"Grasp sample position for {object_name}: {frozen.position}",
                        flush=True,
                    )
                    print(
                        f"Grasp sample quaternion wxyz for {object_name}: "
                        f"{frozen_quat}",
                        flush=True,
                    )
                    from capx.integrations.franka import (
                        acquisition_ownership as _ownership,
                    )

                    if (
                        _ownership.enabled()
                        and os.environ.get(
                            "CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION", "0"
                        )
                        == "1"
                    ):
                        transaction = _ownership.bind_transaction(
                            self,
                            attempt=attempt,
                            object_name=object_name,
                            tip_xyz=frozen.position,
                            tip_wxyz=frozen_quat,
                            source="frozen_graspnet_replay",
                        )
                        _acq.audit(
                            self,
                            "transaction_ownership_bound",
                            attempt=attempt,
                            object_name=object_name,
                            transaction_id=transaction["transaction_id"],
                            source=transaction["source"],
                            has_precomputed_branch=False,
                        )
                    return frozen.position, frozen_quat
        except RuntimeError as _gr_exc:
            # Strict miss: surface clearly for scoring (invalid_grasp_replay_miss).
            if "GRASP_REPLAY_MISS" in str(_gr_exc):
                print(f"[grasp-replay] {_gr_exc}", flush=True)
                raise
            raise

        pc_segment = result["points_3d"]

        if len(pc_segment) == 0:
            raise ValueError(f"Could not segment object '{object_name}'")

        # Build full scene point cloud in world frame from both cameras
        obs = self.get_observation()
        pc_full_parts = []
        for cam_name in [self.camera_name, self.wrist_camera_name]:
            depth = obs[cam_name]["images"]["depth"]
            intrinsics = obs[cam_name]["intrinsics"]
            extrinsics = obs[cam_name]["pose_mat"]
            pts_camera = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
            pts_homogeneous = np.concatenate(
                [pts_camera, np.ones((len(pts_camera), 1))], axis=1
            )
            pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]
            pc_full_parts.append(pts_world)
        pc_full = np.concatenate(pc_full_parts)

        pc_segment, _ = self.filter_noise(pc_segment)
        if len(pc_segment) == 0:
            raise ValueError(f"No valid points after filtering for '{object_name}'")

        # pc_full = self.subsample_point_cloud(pc_full, max_points=20000)
        # pc_segment = self.subsample_point_cloud(pc_segment, max_points=10000)

        try:
            grasp_sample_tf, grasp_scores, grasp_contacts = (
                self.plan_grasp_from_point_clouds(pc_full, pc_segment)
            )
        except GraspProposalUnavailable:
            if os.environ.get(
                "CAPX_LIBERO_G17_V350M_PROPOSAL_FAILURE_FG_REGENERATION",
                "0",
            ) == "1":
                _acq.audit(
                    self,
                    "acquisition_proposal_unavailable",
                    attempt=attempt,
                    object_name=object_name,
                    proposal_source="contact_graspnet",
                    physical_command_executed=False,
                    escape_executed=False,
                    next_state="fresh_FG_observation_then_new_proposal",
                )
                # No transaction was bound and no physical ingress happened.
                # Keep the symbolic acquisition obligation, clear any stale
                # ingress, and let the outer FG loop observe the current world.
                self._capx_v293_capture_state = "acquisition_needed"
                self._capx_v312_acquisition_transaction = None
                self._capx_v304_ingress = None
                from capx.integrations.franka import fg_native_retry as _native_retry

                if _native_retry.enabled():
                    proposal_event = _native_retry.proposal_unavailable_event(
                        attempt=attempt,
                        object_name=object_name,
                    )
                    _native_retry.publish(proposal_event)
                    _acq.audit(
                        self,
                        "fg_native_retry_preclose_event",
                        retry_event=_native_retry.event_dict(proposal_event),
                        hard_route="current_world_FG_regeneration",
                        soft_projection_pending=True,
                    )
                from capx.execution_control import breakpoint_code_block

                breakpoint_code_block(
                    "acquisition_proposal_unavailable_fg_regeneration"
                )
            raise

        rotated_grasps = np.stack(
            [
                (
                    vtf.SE3.from_matrix(matrix)
                    @ vtf.SE3.from_rotation(
                        rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2)
                    )
                ).as_matrix()
                for matrix in np.asarray(grasp_sample_tf)
            ],
            axis=0,
        )
        if _acq.enabled():
            best_idx, candidate_audit = self._select_v293_grasp_candidate(
                rotated_grasps,
                np.asarray(grasp_scores, dtype=np.float64),
                pc_full=np.asarray(pc_full, dtype=np.float64),
                pc_target=np.asarray(pc_segment, dtype=np.float64),
                contact_points=np.asarray(grasp_contacts, dtype=np.float64),
            )
            _acq.audit(
                self,
                "grasp_candidates_ranked",
                attempt=attempt,
                object_name=object_name,
                selected_index=int(best_idx),
                segment=_acq.cloud_summary(pc_segment),
                candidates=candidate_audit,
            )
            probe_report = getattr(self, "_capx_v296b_probe_report", None)
            if probe_report is not None:
                _acq.audit(
                    self,
                    "perceptual_grasp_family_probe",
                    attempt=attempt,
                    object_name=object_name,
                    probe=probe_report,
                )
            from capx.integrations.franka import perceptual_grasp_family as _family

            if _family.proximal_enabled():
                try:
                    live_tip = np.asarray(
                        self.get_observation()["robot_cartesian_pos"][:3],
                        dtype=np.float64,
                    )
                    proximal_pc, proximal_summary = _family.proximal_target_points(
                        pc_segment, live_tip
                    )
                    prox_tfs, prox_scores, _ = self.plan_grasp_from_point_clouds(
                        pc_full, proximal_pc
                    )
                    prox_rotated = np.stack(
                        [
                            (
                                vtf.SE3.from_matrix(matrix)
                                @ vtf.SE3.from_rotation(
                                    rotation=vtf.SO3.from_rpy_radians(
                                        0.0, 0.0, np.pi / 2
                                    )
                                )
                            ).as_matrix()
                            for matrix in np.asarray(prox_tfs)
                        ],
                        axis=0,
                    )
                    from capx.integrations.franka.tip_xy_only import (
                        _live_actuated_cfg,
                    )

                    proximal_probe = _family.probe(
                        self,
                        prox_rotated,
                        np.asarray(prox_scores, dtype=np.float64),
                        _live_actuated_cfg(self),
                        getattr(self, "_capx_v296a_pt_scene"),
                    )
                    proximal_probe["contact_generation"] = proximal_summary
                except Exception as exc:
                    proximal_probe = {
                        "schema": "capx_libero_v297a_proximal_contact_probe_v1",
                        "behavior_effect": "audit_only",
                        "error": repr(exc),
                    }
                _acq.audit(
                    self,
                    "proximal_contact_grasp_family_probe",
                    attempt=attempt,
                    object_name=object_name,
                    probe=proximal_probe,
                )
            if (
                candidate_audit
                and candidate_audit[-1].get("selection")
                == "fail_open_original_argmax"
                and os.environ.get(
                    "CAPX_LIBERO_G17_V293_REQUIRE_FEASIBLE_GRASP", "0"
                )
                == "1"
            ):
                _acq.audit(
                    self,
                    "acquire_ready_failed",
                    attempt=attempt,
                    object_name=object_name,
                    reason="no_accurate_measured_branch_candidate",
                )
                raise ValueError(
                    f"No measured-branch-feasible grasp candidate for '{object_name}'"
                )
        else:
            best_idx = int(np.asarray(grasp_scores).argmax())
        best_grasp = vtf.SE3.from_matrix(rotated_grasps[best_idx])

        grasp_position = best_grasp.wxyz_xyz[-3:]
        grasp_quat = best_grasp.wxyz_xyz[:4]
        from capx.integrations.franka import perceptual_grasp_family as _family

        if _family.topdown_grasp_behavior_enabled():
            report = getattr(self, "_capx_v296b_probe_report", None) or {}
            recommended = report.get("recommended_pose")
            from capx.integrations.franka import grasp_recovery as _gr_recovery

            frozen_recommended = _gr_recovery.frozen_trial02_recommended_pose(
                self, attempt
            )
            if frozen_recommended is not None:
                recommended = frozen_recommended
                if _acq.enabled():
                    _acq.audit(
                        self,
                        "gr_recovery_frozen_transaction_replay",
                        attempt=attempt,
                        object_name=object_name,
                        source_job=527196,
                    )
            if recommended is None:
                from capx.integrations.franka import postclose_acquisition as _v331

                if (
                    os.environ.get(
                        "CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION", "0"
                    )
                    == "1"
                    and not _v331.enabled()
                ):
                    # A missing recommendation means that the robot-visible
                    # candidate search is incomplete, not that an arbitrary
                    # oblique GraspNet command is safe to execute.  Return to
                    # regeneration without moving; v312 has already searched
                    # multiple distinct contacts inside this API call.
                    if _acq.enabled():
                        _acq.audit(
                            self,
                            "topdown_transaction_unavailable",
                            attempt=attempt,
                            object_name=object_name,
                            reason="no_endpoint_clear_multi_contact_topdown",
                        )
                    self._capx_v312_acquisition_transaction = None
                    raise ValueError(
                        f"No endpoint-clear multi-contact top-down grasp for "
                        f"'{object_name}'"
                    )
                if (
                    os.environ.get(
                        "CAPX_LIBERO_G17_V309_TOPDOWN_SOFT_FALLBACK", "0"
                    )
                    == "1"
                    or _v331.enabled()
                ):
                    # PT-clear is a ranking preference, not proof that all
                    # other perceptual contacts are impossible.  Preserve the
                    # baseline GraspNet candidate instead of converting an
                    # incomplete PT model into a deterministic acquisition
                    # failure.
                    if _acq.enabled():
                        _acq.audit(
                            self,
                            "topdown_grasp_fallback",
                            attempt=attempt,
                            object_name=object_name,
                            selection="measured_branch_GraspNet_candidate",
                            reason=(
                                "v331_postclose_native_fallback"
                                if _v331.enabled()
                                else "no_PT_clear_contact_height_candidate"
                            ),
                        )
                else:
                    raise ValueError(
                        f"No PT-clear contact-height top-down grasp for '{object_name}'"
                    )
            else:
                grasp_position = np.asarray(recommended["tip_xyz"], dtype=np.float64)
                grasp_quat = np.asarray(recommended["tip_wxyz"], dtype=np.float64)
                if (
                    os.environ.get(
                        "CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION", "0"
                    )
                    == "1"
                ):
                    from capx.integrations.franka import (
                        acquisition_ownership as _ownership,
                    )

                    if _ownership.enabled():
                        transaction = _ownership.bind_transaction(
                            self,
                            attempt=attempt,
                            object_name=object_name,
                            tip_xyz=grasp_position,
                            tip_wxyz=grasp_quat,
                            source="certified_topdown_transaction",
                            topdown_transaction=recommended.get(
                                "topdown_transaction"
                            ),
                        )
                        _acq.audit(
                            self,
                            "transaction_ownership_bound",
                            attempt=attempt,
                            object_name=object_name,
                            transaction_id=transaction["transaction_id"],
                            source=transaction["source"],
                            has_precomputed_branch=True,
                        )
                    else:
                        self._capx_v312_acquisition_transaction = {
                            "attempt": attempt,
                            "object_name": object_name,
                            "tip_xyz": grasp_position.copy(),
                            "tip_wxyz": grasp_quat.copy(),
                            "pregrasp_tip_xyz": (
                                recommended.get("topdown_transaction") or {}
                            ).get("pregrasp_tip_xyz"),
                            "pregrasp_arm_cfg_rad": (
                                recommended.get("topdown_transaction") or {}
                            ).get("pregrasp_arm_cfg_rad"),
                            "contact_arm_cfg_rad": (
                                recommended.get("topdown_transaction") or {}
                            ).get("contact_arm_cfg_rad"),
                            "pregrasp_recorded": False,
                            "contact_reached": False,
                            "executed": False,
                        }
                if _acq.enabled():
                    _acq.audit(
                        self,
                        "topdown_grasp_selected",
                        attempt=attempt,
                        object_name=object_name,
                        selection=recommended,
                    )
        if (
            os.environ.get("CAPX_LIBERO_G17_V277_CANONICAL_GRASP", "0") == "1"
        ):
            grasp_quat = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        transaction = getattr(self, "_capx_v312_acquisition_transaction", None)
        from capx.integrations.franka import postclose_acquisition as _v331

        if (
            _v331.enabled()
            and os.environ.get(
                "CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION", "0"
            )
            == "1"
            and not isinstance(transaction, dict)
        ):
            # Bind the native GraspNet fallback so its physical close can be
            # classified after execution. goto_pose fills the measured
            # pregrasp boundary; no path geometry is guessed here.
            from capx.integrations.franka import (
                acquisition_ownership as _ownership,
            )

            if _ownership.enabled():
                transaction = _ownership.bind_transaction(
                    self,
                    attempt=attempt,
                    object_name=object_name,
                    tip_xyz=grasp_position,
                    tip_wxyz=grasp_quat,
                    source="native_graspnet_postclose_observed",
                )
            else:
                transaction = {
                    "attempt": attempt,
                    "object_name": object_name,
                    "source": "native_graspnet_postclose_observed",
                    "tip_xyz": np.asarray(grasp_position).copy(),
                    "tip_wxyz": np.asarray(grasp_quat).copy(),
                    "pregrasp_tip_xyz": None,
                    "pregrasp_arm_cfg_rad": None,
                    "contact_arm_cfg_rad": None,
                    "pregrasp_recorded": False,
                    "contact_reached": False,
                    "executed": False,
                }
                self._capx_v312_acquisition_transaction = transaction
            if _acq.enabled():
                _acq.audit(
                    self,
                    "native_candidate_bound_for_postclose",
                    attempt=attempt,
                    object_name=object_name,
                    source=transaction["source"],
                )
                if _ownership.enabled():
                    _acq.audit(
                        self,
                        "transaction_ownership_bound",
                        attempt=attempt,
                        object_name=object_name,
                        transaction_id=transaction["transaction_id"],
                        source=transaction["source"],
                        has_precomputed_branch=False,
                    )
        if isinstance(transaction, dict):
            transaction["tip_xyz"] = np.asarray(grasp_position).copy()
            transaction["tip_wxyz"] = np.asarray(grasp_quat).copy()
        print(f"sample_grasp_pose in {time.time() - start_time} seconds")
        print(f"Grasp sample position for {object_name}: {grasp_position}")
        print(f"Grasp sample quaternion wxyz for {object_name}: {grasp_quat}")
        return grasp_position, grasp_quat

    def _select_v293_grasp_candidate(
        self,
        grasp_tfs: np.ndarray,
        grasp_scores: np.ndarray,
        *,
        top_k: int = 12,
        pc_full: np.ndarray | None = None,
        pc_target: np.ndarray | None = None,
        contact_points: np.ndarray | None = None,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Choose a GraspNet candidate that is accurate on the measured branch.

        This is a pre-execution ranking operation, not a global motion gate.  If
        no top-k candidate has an accurate measured-seed solution, it fails open
        to the original GraspNet argmax while making that fact explicit in the
        private audit.
        """
        from capx.integrations.franka.measured_seed_ik import (
            _candidate_metrics,
            _live_actuated_cfg,
            _quat_multiply_wxyz,
            _validate_candidate,
            tool_pi_symmetry_enabled,
        )

        matrices = np.asarray(grasp_tfs, dtype=np.float64).reshape(-1, 4, 4)
        scores = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
        order = np.argsort(-scores)[: min(int(top_k), len(scores))]
        measured = _live_actuated_cfg(self)
        self._capx_v296b_probe_report = None
        pt_scene = None
        pt_scene_error = None
        from capx.integrations.franka import perceptual_swept_path as _swept

        if _swept.enabled() and pc_full is not None and pc_target is not None:
            try:
                pt_scene = _swept.prepare_scene(
                    self, pc_full, pc_target, measured
                )
                # The command-level audit consumes this immutable observation
                # snapshot only after the executor has selected its exact IK
                # branch.  Candidate reports below remain useful diagnostics,
                # but are not mistaken for the commanded branch.
                self._capx_v296a_pt_scene = pt_scene
                self._capx_v296a_pt_target_points = np.asarray(
                    pc_target, dtype=np.float64
                ).copy()
            except Exception as exc:
                pt_scene_error = repr(exc)
        rows: list[dict[str, Any]] = []
        feasible: list[tuple[float, float, int]] = []
        for rank, idx_raw in enumerate(order):
            idx = int(idx_raw)
            grasp = vtf.SE3.from_matrix(matrices[idx])
            tip_pos = np.asarray(grasp.wxyz_xyz[-3:], dtype=np.float64)
            tip_quat = np.asarray(grasp.wxyz_xyz[:4], dtype=np.float64)
            if os.environ.get("CAPX_LIBERO_G17_GOTO_TIP_IDENTITY", "0") == "1":
                hand_pos, hand_quat = tip_pose_to_hand_ik_target(tip_pos, tip_quat)
            else:
                rot = SciRotation.from_quat(
                    [tip_quat[1], tip_quat[2], tip_quat[3], tip_quat[0]]
                )
                hand_pos = tip_pos + rot.apply(self._TCP_OFFSET)
                hand_quat = tip_quat
            orientations = [("requested", hand_quat)]
            if tool_pi_symmetry_enabled():
                orientations.append(
                    (
                        "tool_z_pi",
                        _quat_multiply_wxyz(
                            hand_quat, np.array([0.0, 0.0, 0.0, 1.0])
                        ),
                    )
                )
            best: tuple[float, float, float, float, str] | None = None
            best_cfg: np.ndarray | None = None
            error: str | None = None
            for orientation_name, orientation in orientations:
                try:
                    cfg = np.asarray(
                        self._pks.solve_ik_seeded(
                            robot=self._robot,
                            target_link_name=self._target_link_name,
                            target_position=hand_pos,
                            target_wxyz=orientation,
                            initial_cfg=measured,
                        ),
                        dtype=np.float64,
                    ).reshape(-1)
                    _validate_candidate(self, cfg)
                    pos_e, ori_e, max_dq, norm_dq = _candidate_metrics(
                        self, cfg, measured, hand_pos, orientation
                    )
                    values = (pos_e, ori_e, max_dq, norm_dq, orientation_name)
                    if best is None or values[:4] < best[:4]:
                        best = values
                        best_cfg = cfg.copy()
                except Exception as exc:
                    error = repr(exc)
            row: dict[str, Any] = {
                "index": idx,
                "score_rank": rank,
                "graspnet_score": float(scores[idx]),
                "tip_xyz": tip_pos.round(6).tolist(),
            }
            if contact_points is not None and idx < len(contact_points):
                contact = np.asarray(contact_points[idx], dtype=np.float64).reshape(3)
                row["contact_xyz"] = contact.round(6).tolist()
                row["tip_contact_distance_m"] = float(
                    np.linalg.norm(tip_pos - contact)
                )
            if best is None:
                row.update({"ik_feasible": False, "error": error})
            else:
                pos_e, ori_e, max_dq, norm_dq, orientation_name = best
                accurate = pos_e <= 0.010 and ori_e <= 0.100
                row.update(
                    {
                        "ik_feasible": bool(accurate),
                        "position_error_m": pos_e,
                        "orientation_error_rad": ori_e,
                        "max_joint_delta_rad": max_dq,
                        "joint_delta_norm_rad": norm_dq,
                        "orientation": orientation_name,
                    }
                )
                if accurate:
                    # GraspNet quality remains primary within the accurate set;
                    # measured-branch motion breaks near-score ties.
                    utility = float(scores[idx]) - 0.02 * max_dq
                    feasible.append((-utility, max_dq, idx))
                if pt_scene is not None and best_cfg is not None:
                    try:
                        row["pt_swept_path"] = _swept.audit_candidate_path(
                            self, pt_scene, measured, best_cfg
                        )
                    except Exception as exc:
                        row["pt_swept_path_error"] = repr(exc)
                elif pt_scene_error is not None:
                    row["pt_swept_path_error"] = pt_scene_error
            rows.append(row)
        from capx.integrations.franka import perceptual_grasp_family as _family

        if _family.enabled() and pt_scene is not None:
            try:
                if (
                    _family.multi_contact_topdown_enabled()
                    and _family.topdown_grasp_probe_enabled()
                ):
                    self._capx_v296b_probe_report = (
                        _family.probe_topdown_contacts(
                            self, matrices, scores, measured, pt_scene
                        )
                    )
                else:
                    self._capx_v296b_probe_report = _family.probe(
                        self, matrices, scores, measured, pt_scene
                    )
            except Exception as exc:
                self._capx_v296b_probe_report = {
                    "schema": "capx_libero_v296b_perceptual_grasp_family_probe_v1",
                    "behavior_effect": "audit_only",
                    "error": repr(exc),
                }
        if feasible:
            return min(feasible)[2], rows
        original = int(np.argmax(scores))
        rows.append(
            {
                "selection": "fail_open_original_argmax",
                "index": original,
                "reason": "no_accurate_measured_branch_candidate",
            }
        )
        return original, rows

    def _segment_object_from_language(
        self, image: Image.Image, object_name: str
    ) -> tuple[np.ndarray, tuple[int, int] | None, list[float] | None]:
        """Use SAM3 (or Molmo + SAM2) to return a binary mask for a language-described object."""
        if self.use_sam3:
            # SAM3: unified detection + segmentation from text prompt
            results = self.sam3_seg_fn(image, text_prompt=object_name)
            if len(results) == 0:
                # try molmo if language sam3 fails
                dets = self.molmo_point_fn(image, objects=[object_name])
                point = dets.get(object_name)
                if point is None or any(coord is None for coord in point):
                    return None, None, None
                point_coords = (float(point[0]), float(point[1]))
                results = self.sam3_point_prompt_fn(image, point_coords=point_coords)
            
            if len(results) == 0:
                return None, None, None

            scores = [result["score"] for result in results]
            best_result = results[np.argmax(scores)]
            mask_bool = best_result["mask"].astype(bool)

            # Get center point of mask for visualization
            ys, xs = np.where(mask_bool)
            if len(xs) > 0 and len(ys) > 0:
                point_xy = (int(xs.mean()), int(ys.mean()))
            else:
                point_xy = None

            return mask_bool, point_xy, scores
        else:
            # Molmo + SAM2: separate detection and segmentation
            dets = self.molmo_point_fn(image, objects=[object_name])
            point = dets.get(object_name)
            if point is None or any(coord is None for coord in point):
                return None, None, None
            point_coords = (float(point[0]), float(point[1]))
            scores, masks = self.sam2_point_prompt_fn(image, point_coords=point_coords)
            if len(masks) == 0:
                raise ValueError(f"SAM2 returned no masks for '{object_name}'")

            best_mask = np.asarray(masks[0])
            best_mask = np.squeeze(best_mask)
            if best_mask.ndim != 2:
                raise ValueError(f"SAM2 mask must be 2D, got shape {best_mask.shape}")

            mask_bool = best_mask.astype(bool)
            point_xy = (int(round(point_coords[0])), int(round(point_coords[1])))
            return mask_bool, point_xy, scores

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """Get the oriented bounding box from 3D points.

        Args:
            points: np.ndarray: The 3D points to get the oriented bounding box from.
                Shape: (N, 3), dtype float64.

        Returns:
            dict[str, Any]: The oriented bounding box. The dictionary contains the following keys:
                - "center": np.ndarray: The center of the oriented bounding box in point cloud frame.
                - "extent": np.ndarray: The extent of the oriented bounding box.
                - "R": np.ndarray: The rotation matrix of the oriented bounding box in point cloud frame.

        Example:
            >>> points = np.random.randn((100, 3))
            >>> obb = get_oriented_bounding_box_from_3d_points(points)
        """
        # inject some noise to the points
        points = points + np.random.normal(0, 0.0001, points.shape)
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)
        o3d_points, ind = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        obb = o3d_points.get_oriented_bounding_box()
        return {
            "center": obb.center,
            "extent": obb.extent,
            "R": obb.R,
            "quaternion_wxyz": vtf.SO3.from_matrix(obb.R).wxyz,
        }

    def _stock_best_mask(
        self,
        rgb: np.ndarray,
        text_prompt: str,
        point: tuple[int, int] | None = None,
        *,
        allow_text_fallback: bool = True,
    ) -> dict[str, Any] | None:
        """Stock keep-one: optional point → Molmo point → SAM3 text, then max(score)."""
        masks: list[dict[str, Any]] = []
        if point is not None and point[0] is not None:
            masks = self.segment_sam3_point_prompt(rgb, point)
        if not masks and allow_text_fallback:
            points = self.point_prompt_molmo(rgb, text_prompt)
            molmo_point = points[text_prompt]
            if molmo_point[0] is not None:
                masks = self.segment_sam3_point_prompt(rgb, molmo_point)
            if not masks:
                masks = self.segment_sam3_text_prompt(rgb, text_prompt)
        if not masks:
            return None
        return max(masks, key=lambda item: item["score"])

    def _world_points_for_mask(
        self,
        mask: np.ndarray,
        pts_world: np.ndarray,
        cam_name: str,
    ) -> np.ndarray:
        mask_flat = np.asarray(mask).reshape(-1)
        if len(pts_world) != len(mask_flat):
            print(
                f"Warning: Point cloud size ({len(pts_world)}) doesn't match "
                f"mask size ({len(mask_flat)}) for {cam_name}"
            )
            min_len = min(len(pts_world), len(mask_flat))
            return pts_world[:min_len][mask_flat[:min_len]]
        return pts_world[mask_flat]

    def _agentview_world_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = self.get_observation()
        cam = self.camera_name
        depth = obs[cam]["images"]["depth"]
        intrinsics = obs[cam]["intrinsics"]
        extrinsics = obs[cam]["pose_mat"]
        rgb = obs[cam]["images"]["rgb"]
        pts_camera = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
        pts_homogeneous = np.concatenate(
            [pts_camera, np.ones((len(pts_camera), 1))], axis=1
        )
        pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]
        return rgb, pts_world, extrinsics, intrinsics

    def _lock_privileged_aim(
        self,
        object_name: str,
        *,
        token: str,
        body_name: str | None = None,
        use_multiview: bool = True,
    ) -> dict[str, Any] | None:
        """RelXY-pick body; keep stock Molmo/SAM if it matches, else point-correct."""
        from capx.integrations.franka import acquisition_audit as _acq
        from capx.verification.libero_g17_xy_lock import (
            body_world_to_robot_xyz,
            mask_centroid_matches_body,
            pick_bound_body,
            pick_same_name_body,
            project_frame_to_pixel,
        )

        picked = (
            pick_bound_body(self._env, object_name, body_name)
            if body_name
            else pick_same_name_body(self._env, object_name, token)
        )
        if picked is None:
            return None
        body_name, world_xyz = picked
        robot_xyz = body_world_to_robot_xyz(self._env, world_xyz)
        if robot_xyz is None:
            print(
                f"[xy-lock] skip: token={token} body={body_name} no robot base",
                flush=True,
            )
            return None

        # Prefer stock Molmo+SAM quality when the lottery already hit this body.
        try:
            stock = self.get_object_3d_points_and_masks_from_language(
                object_name, use_multiview=use_multiview
            )
        except Exception as exc:
            stock = None
            print(f"[xy-lock] stock failed ({exc!r}); point-correct", flush=True)
        if stock is not None:
            raw_pts = stock.get("points_3d")
            if raw_pts is None:
                stock_pts = np.zeros((0, 3), dtype=np.float64)
            else:
                stock_pts = np.asarray(raw_pts, dtype=np.float64)
            if stock_pts.ndim == 2 and len(stock_pts) > 0:
                centroid = np.mean(stock_pts, axis=0)
                dist = float(np.linalg.norm(centroid[:2] - robot_xyz[:2]))
                dist_z = abs(float(centroid[2] - robot_xyz[2]))
                matches = mask_centroid_matches_body(
                    centroid,
                    robot_xyz,
                    thresh_z_m=0.08 if _acq.enabled() else None,
                )
                if matches:
                    print(
                        f"[xy-lock] keep-stock token={token} body={body_name} "
                        f"dist_xy={dist:.3f} dist_z={dist_z:.3f}",
                        flush=True,
                    )
                    _acq.audit(
                        self,
                        "target_grounded",
                        object_name=object_name,
                        mode="keep_stock",
                        token=token,
                        body_name=body_name,
                        dist_xy_m=dist,
                        dist_z_m=dist_z,
                        cloud=_acq.cloud_summary(stock_pts),
                    )
                    chosen = dict(stock)
                    chosen["xy_lock_body"] = body_name
                    chosen["xy_lock_token"] = token
                    chosen["xy_lock_mode"] = "keep_stock"
                    return chosen
                print(
                    f"[xy-lock] stock-mismatch token={token} body={body_name} "
                    f"dist_xy={dist:.3f} dist_z={dist_z:.3f}; point-correct",
                    flush=True,
                )
                _acq.audit(
                    self,
                    "grounding_rejected",
                    object_name=object_name,
                    mode="stock",
                    token=token,
                    body_name=body_name,
                    dist_xy_m=dist,
                    dist_z_m=dist_z,
                    cloud=_acq.cloud_summary(stock_pts),
                )

        rgb, pts_world, extrinsics, intrinsics = self._agentview_world_points()
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        pixel = project_frame_to_pixel(robot_xyz, extrinsics, intrinsics, (h, w))
        if pixel is None:
            print(
                f"[xy-lock] skip: token={token} body={body_name} "
                f"centroid not in agentview robot_xyz={robot_xyz.tolist()}",
                flush=True,
            )
            return None
        if _acq.enabled():
            point_masks = self.segment_sam3_point_prompt(rgb, pixel)
            valid_point_masks: list[tuple[float, dict[str, Any], np.ndarray]] = []
            for candidate_index, candidate_mask in enumerate(point_masks):
                mask = candidate_mask.get("mask")
                if mask is None:
                    continue
                candidate_pts = self._world_points_for_mask(
                    mask, pts_world, self.camera_name
                )
                if len(candidate_pts) == 0:
                    continue
                candidate_centroid = np.mean(candidate_pts, axis=0)
                candidate_dist_xy = float(
                    np.linalg.norm(candidate_centroid[:2] - robot_xyz[:2])
                )
                candidate_dist_z = abs(
                    float(candidate_centroid[2] - robot_xyz[2])
                )
                candidate_matches = mask_centroid_matches_body(
                    candidate_centroid, robot_xyz, thresh_z_m=0.08
                )
                _acq.audit(
                    self,
                    "point_mask_candidate",
                    object_name=object_name,
                    candidate_index=candidate_index,
                    score=float(candidate_mask.get("score", 0.0)),
                    matches_bound_target=bool(candidate_matches),
                    dist_xy_m=candidate_dist_xy,
                    dist_z_m=candidate_dist_z,
                    cloud=_acq.cloud_summary(candidate_pts),
                )
                if candidate_matches:
                    valid_point_masks.append(
                        (
                            float(candidate_mask.get("score", 0.0)),
                            candidate_mask,
                            candidate_pts,
                        )
                    )
            if valid_point_masks:
                _, mask_data, precomputed_pts = max(
                    valid_point_masks, key=lambda row: row[0]
                )
            else:
                mask_data, precomputed_pts = None, None
        else:
            mask_data = self._stock_best_mask(
                rgb, object_name, point=pixel, allow_text_fallback=False
            )
            precomputed_pts = None
        if mask_data is None or mask_data.get("mask") is None:
            print(
                f"[xy-lock] skip: token={token} body={body_name} "
                f"pixel={pixel} SAM point empty",
                flush=True,
            )
            return None
        pts = (
            np.asarray(precomputed_pts)
            if precomputed_pts is not None
            else self._world_points_for_mask(
                mask_data["mask"], pts_world, self.camera_name
            )
        )
        if len(pts) == 0:
            return None
        if _acq.enabled():
            centroid = np.mean(np.asarray(pts, dtype=np.float64), axis=0)
            dist_xy = float(np.linalg.norm(centroid[:2] - robot_xyz[:2]))
            dist_z = abs(float(centroid[2] - robot_xyz[2]))
            if not mask_centroid_matches_body(
                centroid, robot_xyz, thresh_z_m=0.08
            ):
                _acq.audit(
                    self,
                    "grounding_rejected",
                    object_name=object_name,
                    mode="point_correct",
                    token=token,
                    body_name=body_name,
                    pixel=pixel,
                    dist_xy_m=dist_xy,
                    dist_z_m=dist_z,
                    cloud=_acq.cloud_summary(pts),
                )
                print(
                    f"[xy-lock] point-reject token={token} body={body_name} "
                    f"dist_xy={dist_xy:.3f} dist_z={dist_z:.3f}",
                    flush=True,
                )
                return None
        print(
            f"[xy-lock] point-correct token={token} body={body_name} "
            f"pixel={pixel} n_pts={len(pts)}",
            flush=True,
        )
        _acq.audit(
            self,
            "target_grounded",
            object_name=object_name,
            mode="point_correct",
            token=token,
            body_name=body_name,
            cloud=_acq.cloud_summary(pts),
        )
        result: dict[str, Any] = {
            "agentview_mask": mask_data["mask"],
            "agentview_score": mask_data.get("score"),
            "agentview_points_3d": pts,
            "wrist_mask": None,
            "wrist_score": None,
            "wrist_points_3d": None,
            "points_3d": np.asarray(pts),
            "xy_lock_body": body_name,
            "xy_lock_token": token,
            "xy_lock_mode": "point_correct",
        }
        if not use_multiview:
            return result
        obs = self.get_observation()
        wrist = self.wrist_camera_name
        if wrist not in obs:
            return result
        w_rgb = obs[wrist]["images"]["rgb"]
        w_depth = obs[wrist]["images"]["depth"]
        w_intr = obs[wrist]["intrinsics"]
        w_extr = obs[wrist]["pose_mat"]
        wh, ww = int(w_rgb.shape[0]), int(w_rgb.shape[1])
        w_pixel = project_frame_to_pixel(robot_xyz, w_extr, w_intr, (wh, ww))
        w_mask = self._stock_best_mask(
            w_rgb,
            object_name,
            point=w_pixel,
            allow_text_fallback=w_pixel is None,
        )
        if w_mask is None or w_mask.get("mask") is None:
            return result
        w_cam = depth_to_pointcloud(w_depth, w_intr, subsample_factor=1)
        w_hom = np.concatenate(
            [w_cam, np.ones((len(w_cam), 1))], axis=1
        )
        w_world = (w_extr @ w_hom.T).T[:, :3]
        w_pts = self._world_points_for_mask(w_mask["mask"], w_world, wrist)
        if _acq.enabled() and len(w_pts) > 0:
            w_centroid = np.mean(np.asarray(w_pts, dtype=np.float64), axis=0)
            w_match = mask_centroid_matches_body(
                w_centroid, robot_xyz, thresh_z_m=0.08
            )
            _acq.audit(
                self,
                "wrist_mask_candidate",
                object_name=object_name,
                matches_bound_target=bool(w_match),
                dist_xy_m=float(
                    np.linalg.norm(w_centroid[:2] - robot_xyz[:2])
                ),
                dist_z_m=abs(float(w_centroid[2] - robot_xyz[2])),
                cloud=_acq.cloud_summary(w_pts),
            )
            if not w_match:
                return result
        result["wrist_mask"] = w_mask["mask"]
        result["wrist_score"] = w_mask.get("score")
        result["wrist_points_3d"] = w_pts
        if len(w_pts) > 0:
            result["points_3d"] = np.concatenate(
                [np.asarray(pts), np.asarray(w_pts)], axis=0
            )
        return result

    def _resolve_typed_instance(
        self,
        object_name: str,
        use_multiview: bool = True,
        which: str | None = None,
    ) -> dict[str, Any]:
        """Skill-side type→instance: RelXY privileged aim, else stock type lottery."""
        from capx.verification.libero_g17_xy_lock import (
            resolve_lock,
            same_name_unresolved,
        )

        # No public which= coaching: missing live token → stock type path.
        if same_name_unresolved(object_name, which):
            return self.get_object_3d_points_and_masks_from_language(
                object_name, use_multiview=use_multiview
            )
        lock = resolve_lock(object_name, which)
        if lock is not None:
            aimed = self._lock_privileged_aim(
                object_name,
                token=str(lock["token"]),
                body_name=(
                    str(lock["body_name"])
                    if lock.get("body_name") is not None
                    else None
                ),
                use_multiview=use_multiview,
            )
            if aimed is not None:
                return aimed
            from capx.integrations.franka import acquisition_audit as _acq

            if _acq.enabled():
                _acq.audit(
                    self,
                    "grounding_failed",
                    object_name=object_name,
                    token=str(lock["token"]),
                    decision="reject_attempt",
                )
                raise ValueError(
                    f"No XYZ-consistent bound-target mask for '{object_name}'"
                )
            # Lock miss: fall back to stock rather than leaking RelXY tokens.
            print(
                f"[xy-lock] resolve-miss object={object_name!r}; stock fallback",
                flush=True,
            )
            return self.get_object_3d_points_and_masks_from_language(
                object_name, use_multiview=use_multiview
            )
        return self.get_object_3d_points_and_masks_from_language(
            object_name, use_multiview=use_multiview
        )

    def get_object_3d_points_and_masks_from_language(
        self,
        text_prompt: str,
        use_multiview: bool = True,
    ) -> dict[str, Any]:
        """Segment an object using text prompt across multiple views (agentview and wrist).
        
        Uses Molmo to locate the object in both camera views, then SAM3 for
        segmentation. Returns the masks and 3D points that appear in both views.
        This function is type-level only: one complete object via max(score).
        
        Args:
            text_prompt: Text description of the object to segment.
            use_multiview: If True, uses the wrist camera as well as the main camera for segmentation.
            
        Returns:
            dict containing:
                - "agentview_mask": np.ndarray of shape (H, W), dtype bool
                - "wrist_mask": np.ndarray of shape (H, W), dtype bool (None if single view)
                - "points_3d": np.ndarray of shape (N, 3), 3D points in world frame
                - "agentview_score": float, SAM3 confidence score
                - "wrist_score": float, SAM3 confidence score (None if single view)
        """
        obs = self.get_observation()
        
        cameras = [self.camera_name]
        if use_multiview:
            cameras.append(self.wrist_camera_name)
            
        camera_data = {}
        
        for cam_name in cameras:
            # get images from camera
            rgb = obs[cam_name]["images"]["rgb"]
            depth = obs[cam_name]["images"]["depth"]
            intrinsics = obs[cam_name]["intrinsics"]
            extrinsics = obs[cam_name]["pose_mat"]
            
            # Stock path: Molmo one point, then SAM3 text if that yields nothing.
            mask_data = self._stock_best_mask(rgb, text_prompt)
            if mask_data is None:
                raise ValueError(
                    f"SAM3 segmentation failed for '{text_prompt}' on {cam_name}. "
                    f"No masks returned from either point or text prompt."
                )
            
            # Get 3D points in camera frame
            pts_camera = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
            
            # Transform to world frame
            pts_homogeneous = np.concatenate([pts_camera, np.ones((len(pts_camera), 1))], axis=1) # add column of ones
            pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]

            mask = mask_data["mask"]
            score = mask_data["score"]
            pts_3d = self._world_points_for_mask(mask, pts_world, cam_name)
            
            camera_data[cam_name] = {
                "mask": mask,
                "score": score,
                "points_3d": pts_3d
            }
        
        agent_data = camera_data[self.camera_name]
        agent_pts_3d = agent_data["points_3d"]
        
        points_3d = agent_pts_3d
        
        wrist_mask = None
        wrist_score = None
        wrist_pts_3d = None
        
        if use_multiview and self.wrist_camera_name in camera_data:
            # Takes the union of the points from the wrist and agent view if they are close enough, otherwise takes the view with the higher score.
            wrist_data = camera_data[self.wrist_camera_name]
            wrist_pts_3d = wrist_data["points_3d"]
            wrist_mask = wrist_data["mask"]
            wrist_score = wrist_data["score"]
            
            # Find intersection using numpy broadcasting
            if len(wrist_pts_3d) > 0 and len(agent_pts_3d) > 0:
                # Compute pairwise distances between all agent and wrist points
                distances = np.linalg.norm(
                    agent_pts_3d[:, np.newaxis, :] - wrist_pts_3d[np.newaxis, :, :], 
                    axis=2
                )
                # Find minimum distance for each agent point
                min_distances = np.min(distances, axis=1)
                # Keep agent points within threshold
                threshold = 0.01  # 1cm
                if min_distances.min() < threshold:
                    points_3d = np.concatenate([agent_pts_3d, wrist_pts_3d])
                else:
                    if wrist_score > agent_data["score"]:
                        points_3d = wrist_pts_3d
                    else:
                        points_3d = agent_pts_3d
            elif len(wrist_pts_3d) > 0:
                points_3d = wrist_pts_3d
            elif len(agent_pts_3d) > 0:
                points_3d = agent_pts_3d
            else:
                print(f"Warning: No points found for {text_prompt}")
                points_3d = np.array([]).reshape(0, 3)
        
        return {
            "agentview_mask": agent_data["mask"],
            "wrist_mask": wrist_mask,
            "points_3d": np.asarray(points_3d),
            "agentview_points_3d": np.asarray(agent_pts_3d),
            "wrist_points_3d": np.asarray(wrist_pts_3d),
            "agentview_score": agent_data["score"],
            "wrist_score": wrist_score,
        }
    
    def goto_home_joint_position(self) -> None:
        """Return the arm to its reset joint configuration with high manipulability"""
        from capx.verification.libero_g17_live_isolation import action_blocked

        if action_blocked(self, "goto_home_joint_position"):
            return
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            raise RuntimeError("Home joint position is unavailable in the current environment.")
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def subsample_point_cloud(self, pc: np.ndarray, max_points: int = 10000) -> np.ndarray:
        """Randomly subsample a point cloud to a maximum number of points.
        
        Args:
            pc: (N, 3) array of points.
            max_points: The maximum number of points to subsample to. Default is 10000.
        Returns:
            subsampled_pc: (M, 3) array of subsampled points where M <= max_points.
        """
        if len(pc) > max_points:
            return pc[np.random.choice(len(pc), max_points, replace=False)]
        return pc
    
    def filter_noise(self, points: np.ndarray, colors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Filter noise from the point cloud.
        Args:
            points: (N, 3) array of points.
            colors: (N, 3) array of colors. Optional, default is None.
        Returns:
            (M, 3) array of points.
            (M, 3) array of colors. Optional, default is None.
        """
        eps = 0.005  # Maximum distance between samples to be neighbors
        min_samples = 10
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        filtered_pointcloud = points[labels != -1]
        if colors is not None:
            filtered_colors = colors[labels != -1]
        else:
            filtered_colors = None
        return filtered_pointcloud, filtered_colors
    
    def plan_grasp_from_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plan grasp candidates using Contact-GraspNet given a full point cloud and a segmented point cloud of the object wanting to be grasped. These point clouds should be composed from multiple viewpoints.
        NOTE: This function is best for generating grasps on objects with complex shapes. For regular objects like boxes, cylinders, etc..., use get_top_down_grasp_from_obb.

        Args:
            pc_full:
                Point cloud of the full environment, including the object to be grasped.
                Shape: (N, 3), dtype float32.
            pc_segment:
                Point cloud of the segmented object to be grasped.
                Shape: (N, 3), dtype float32.
        Returns:
            grasp_sample_tf: (4, 4) homogeneous transform for the grasp pose in THE POINT CLOUD FRAME.
            grasp_scores: (N,) array of grasp scores.
        """
        grasp_sample, grasp_scores, contact_points = self.grasp_net_plan_point_clouds_fn(
            pc_full, pc_segment, segmap_id=1
        )
        
        if len(grasp_sample) == 0:
            raise GraspProposalUnavailable("No grasp candidates found")

        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
        ).as_matrix()
        return grasp_sample_tf, grasp_scores, np.asarray(contact_points)
    
    def parse_grasp_poses_for_curobo(
        self, grasp_poses_world: np.ndarray, grasp_scores: np.ndarray, top_k: int = 15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parses grasp poses from the world frame into a format compatible with CuRobo.

        Args:
            grasp_poses_world: (N, 4, 4) array of grasp poses in world frame
            grasp_scores: (N,) array of grasp scores
            top_k: number of top grasp poses to return. Default is 15.
        Returns:
            positions: (N, 3) array of grasp positions
            quaternions: (N, 4) array of grasp quaternions
            scores: (N,) array of grasp scores sorted by score
        
        Example:
            >>> grasp_poses_cam, grasp_scores = plan_grasp_from_point_clouds(...)
            >>> positions, quaternions, scores = parse_grasp_poses_for_curobo(grasp_poses_cam, grasp_scores, top_k=15)
        """
        grasp_poses_world = np.asarray(grasp_poses_world)
        positions = grasp_poses_world[..., 3][:, :3]
        rotations = grasp_poses_world[..., :3][:, :3, :3]
        quaternions = vtf.SO3.from_matrix(rotations).wxyz
        scores = grasp_scores
        k = min(top_k, len(positions))
        order = np.argsort(-scores)[:k]
        return positions[order], quaternions[order], scores[order]
    
    ### CuRobo-related functions ###
    def create_curobo_world_from_depth(
        self,
        depth_image: np.ndarray,
        object_mask: np.ndarray,
        intrinsics: np.ndarray,
        camera_pose: np.ndarray | None = None,
        **kwargs: Any,
    ):
        """Create a CuRobo WorldConfig from a depth image and object mask. Stores the result for use by plan_grasp_trajectory."""
        world = _get_curobo_api().create_curobo_world_from_depth(
            depth_image, object_mask, intrinsics, camera_pose=camera_pose, **kwargs
        )
        self._curobo_world_config = world
        return world

    def create_curobo_world_from_pointcloud(
        self, point_cloud: np.ndarray, object_mask: np.ndarray, **kwargs: Any
    ):
        """Create a CuRobo WorldConfig from a point cloud and per-point object mask. Stores the result for use by plan_grasp_trajectory."""
        world = _get_curobo_api().create_curobo_world_from_pointcloud(point_cloud, object_mask, **kwargs)
        self._curobo_world_config = world
        return world

    def create_curobo_world_from_observation(
        self,
        object_mask: np.ndarray,
        *,
        camera_name: str | None = None,
        object_name: str = "object",
        scene_name: str = "scene",
        **kwargs: Any,
    ):
        """Create a CuRobo WorldConfig from the current observation: uses this camera's depth, intrinsics, and pose to build the world, split by object_mask. Stores the result for use by plan_grasp_trajectory.
        
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        intrinsics = obs[cam]["intrinsics"]
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=pose[:3],
        ).as_matrix()
        world = _get_curobo_api().create_curobo_world_from_depth(
            depth,
            object_mask,
            intrinsics,
            camera_pose=camera_pose_4x4,
            object_name=object_name,
            scene_name=scene_name,
            **kwargs,
        )
        self._curobo_world_config = world
        return world

    def update_curobo_world(
        self,
        *,
        camera_name: str | None = None,
        robot_distance_threshold: float = 0.15,
        robot_file: str = "franka.yml",
        **kwargs: Any,
    ) -> Any:
        """Build CuRobo world from current observation (full depth, robot excluded) and store it.

        Uses create_curobo_world_from_depth_full: single scene mesh with points near the
        robot removed (robot_distance_threshold) so start/IK configs are not in collision.
        Same logic as curobo_test script. Use before plan_grasp_trajectory when using
        collision checking.

        Args:
            camera_name: Camera to use; default is self.camera_name.
            robot_distance_threshold: Distance (m) to exclude points near the robot. Default is 0.15.
            robot_file: CuRobo robot config for segmenter. Default is 'franka.yml'.
            **kwargs: Passed to create_curobo_world_from_depth_full.

        Returns:
            The WorldConfig that was stored in self._curobo_world_config.
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = np.asarray(depth[:, :, 0], dtype=np.float64)
        else:
            depth = np.asarray(depth, dtype=np.float64)
        intrinsics = np.asarray(obs[cam]["intrinsics"], dtype=np.float64)
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=np.asarray(pose[:3], dtype=np.float64),
        ).as_matrix()
        robot_joint_pos = np.asarray(obs["robot_joint_pos"], dtype=np.float64)
        world = _get_curobo_api().create_curobo_world_from_depth_full(
            depth,
            intrinsics,
            camera_pose=camera_pose_4x4,
            robot_joint_position=robot_joint_pos,
            robot_file=robot_file,
            robot_distance_threshold=robot_distance_threshold,
            **kwargs,
        )
        self._curobo_world_config = world
        return world

    def plan_grasp_trajectory(
        self,
        object_name: str,
        *,
        object_mask: np.ndarray, #| None = None,
        grasp_poses: list[tuple[np.ndarray, np.ndarray]], #| None = None,
        top_k_grasps: int = 15,
        use_world_collision: bool = True,
        robot_distance_threshold: float = 0.15,
        robot_collision_sphere_buffer: float | None = -0.01,
        collision_activation_distance: float | None = 0.001,
        world_config: Any = None,
        # use_multiview: bool = False,
        **kwargs: Any,
    ) -> tuple[bool, np.ndarray | None, int | None]:
        """Plan a collision-free trajectory to one of the top-k grasp poses.

        Builds a CuRobo world with object mesh + scene mesh (robot excluded via segmenter).
        Collision between the object and the rest of the world is disabled so the robot
        can approach the object. If grasp_poses is None, samples grasps for object_name.

        Args:
            object_name: Name of the object (used for mask and grasps if not provided).
            object_mask: precomputed mask (H, W) of the object.
            grasp_poses: list of (position (3,), quaternion_wxyz (4,)) in world frame.
            top_k_grasps: Number of top grasps to try (by score). Default 15.
            use_world_collision: If True, use world for collision (scene only; object ignored for grasp). Default is True.
            robot_distance_threshold: Passed when building world. Default is 0.15.
            robot_collision_sphere_buffer: Passed to plan_to_grasp_poses. Default is -0.01.
            collision_activation_distance: Passed to plan_to_grasp_poses. Default is 0.001.
            world_config: Optional WorldConfig. If None, builds via update_curobo_world_with_object. Default is None.
            **kwargs: Passed through to plan_to_grasp_poses.

        Returns:
            (success, joint_trajectory, goalset_index): joint_trajectory is (T, 7) or None.
        """
        if grasp_poses is None:
            # positions, quaternions, scores = self._sample_grasp_poses_for_object(object_name, use_multiview=use_multiview)
            # if len(positions) == 0:
            #     return False, None, None
            # k = min(top_k_grasps, len(positions))
            # order = np.argsort(-scores)[:k] if scores is not None and len(scores) else np.arange(k)
            # grasps_to_try = [(positions[i], quaternions[i]) for i in order]
            assert grasp_poses is not None, "grasp_poses must be provided"
        else:
            grasps_to_try = [(np.asarray(p), np.asarray(q)) for p, q in grasp_poses]

        world = world_config
        if world is None:
            self.update_curobo_world_with_object(
                object_name,
                object_mask=object_mask,
                robot_distance_threshold=robot_distance_threshold,
            )
            world = self._curobo_world_config
            # Disable collision with the object so the robot can reach into it
            ignore_obstacle_names = [getattr(self, "_curobo_world_object_name", object_name.replace(" ", "_"))]
        else:
            ignore_obstacle_names = []

        obs = self._env.get_observation()
        start_joint_position = obs["robot_joint_pos"]

        success, joint_traj, goalset_idx = _get_curobo_api().plan_to_grasp_poses(
            world,
            start_joint_position,
            grasps_to_try,
            use_world_collision=use_world_collision,
            robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            collision_activation_distance=collision_activation_distance,
            ignore_obstacle_names=ignore_obstacle_names if use_world_collision else None,
            **kwargs,
        )
        return success, joint_traj, goalset_idx

    def execute_joint_trajectory(
        self,
        joint_trajectory: np.ndarray,
        *,
        subsample: int = 1,
        tolerance: float = 0.01,
        max_steps: int = 120,
    ) -> None:
        """Execute a joint-space trajectory (T, 7) by moving to each waypoint with move_to_joints_blocking.

        Args:
            joint_trajectory: (T, 7) joint positions in radians.
            subsample: Use every Nth waypoint (1 = all). Larger values speed up execution. Default is 1.
            tolerance: Passed to move_to_joints_blocking. Default is 0.01.
            max_steps: Passed to move_to_joints_blocking. Default is 120.
        """
        traj = np.asarray(joint_trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 7:
            raise ValueError(f"joint_trajectory must be (T, 7), got shape {traj.shape}")
        indices = list(range(0, len(traj), subsample))
        if indices and indices[-1] != len(traj) - 1:
            indices.append(len(traj) - 1)
        for i in indices:
            joints = traj[i, :7]
            self._env.move_to_joints_blocking(
                joints, tolerance=tolerance, max_steps=max_steps
            )
    
    def update_curobo_world_with_object(
        self,
        object_name: str,
        *,
        object_mask: np.ndarray | None = None,
        camera_name: str | None = None,
        robot_distance_threshold: float = 0.15,
        robot_file: str = "franka.yml",
        object_name_in_world: str | None = None,
        scene_name: str = "scene",
        **kwargs: Any,
    ) -> Any:
        """Build CuRobo world from current observation with object/scene split (robot excluded) and store it.

        Uses create_curobo_world_from_depth_with_object: creates two separate meshes:
        - Object mesh (from object_mask)
        - Scene mesh (everything else)

        Robot points are excluded so the start configuration is not in collision. Use this
        before plan_with_grasped_object to attach the object and plan motion.

        Args:
            object_name: Name of the object (used to get object_mask if object_mask is None).
            object_mask: Optional precomputed mask (H, W). If None, get_object_mask(object_name) is used.
            camera_name: Camera to use; default is self.camera_name.
            robot_distance_threshold: Distance (m) to exclude points near the robot. Default is 0.15.
            robot_file: CuRobo robot config for segmenter. Default is 'franka.yml'.
            object_name_in_world: Name for the object in WorldConfig (default: object_name, spaces → underscores).
            scene_name: Name for the scene mesh in WorldConfig. Default is 'scene'.
            **kwargs: Passed to create_curobo_world_from_depth_with_object.

        Returns:
            The WorldConfig that was stored in self._curobo_world_config.
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        elif depth.ndim == 2:
            pass
        else:
            raise ValueError(f"Depth image has invalid shape: {depth.shape}")
        depth = np.asarray(depth, dtype=np.float64)
        intrinsics = np.asarray(obs[cam]["intrinsics"], dtype=np.float64)
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=np.asarray(pose[:3], dtype=np.float64),
        ).as_matrix()
        robot_joint_pos = np.asarray(obs["robot_joint_pos"], dtype=np.float64)

        if object_mask is None:
            object_mask = self.get_object_mask(object_name)
        if object_mask is None:
            raise ValueError(f"Could not get object mask for '{object_name}' to create world with object/scene split")
        mask_bool = np.asarray(object_mask, dtype=bool)
        if getattr(self._env, "output_dir", None):
            rgb = np.asarray(obs[cam]["images"]["rgb"])
            self._save_rgb_with_object_mask(rgb, mask_bool, object_name, suffix="world")

        # World-safe name (no spaces) for CuRobo lookup
        world_object_name = object_name_in_world if object_name_in_world is not None else object_name
        mesh_name = world_object_name.replace(" ", "_")
        # Place object mesh at current EE pose so the world matches the grasped location (fixes wrong position in planning/debug)
        ee_pos, ee_quat_wxyz = self.get_ee_pose()
        world = _get_curobo_api().create_curobo_world_from_depth_with_object(
            depth,
            object_mask,
            intrinsics,
            camera_pose=camera_pose_4x4,
            robot_joint_position=robot_joint_pos,
            robot_file=robot_file,
            robot_distance_threshold=robot_distance_threshold,
            object_name=mesh_name,
            scene_name=scene_name,
            object_pose_override=(ee_pos, ee_quat_wxyz),
            **kwargs,
        )
        self._curobo_world_config = world
        self._curobo_world_object_name = mesh_name
        return world

    def plan_with_grasped_object(
        self,
        target_pose: tuple[np.ndarray, np.ndarray],
        object_name: str,
        *,
        object_pose: tuple[np.ndarray, np.ndarray] | None = None,
        object_mask: np.ndarray | None = None,
        world_config: Any = None,
        robot_collision_sphere_buffer: float | None = -0.01,
        collision_activation_distance: float | None = 0.01,
        **kwargs: Any,
    ) -> tuple[bool, np.ndarray | None]:
        """Plan a collision-free trajectory to move a grasped object to a target pose.

        NOTE: the grasped object must not be in collision with the scene before calling this function. First lift the object then plan the trajectory to the target pose.

        If object_mask is provided, the CuRobo world is rebuilt from the current observation
        so the object mesh is at its current (e.g. post-lift) position; collision between
        object and scene is enabled. The object is then attached to the robot and motion
        is planned to the target pose.

        Args:
            target_pose: (position (3,), quaternion_wxyz (4,)) target pose in world frame (e.g. basket).
            object_name: Name of the object to attach (used for world mesh name: spaces → underscores).
            object_pose: Optional (position, quat_wxyz) of object; unused for now, for API consistency. Default is None.
            object_mask: If provided, rebuild world from current observation with this mask before planning. Default is None.
            world_config: Optional WorldConfig. If None, uses stored or rebuilds when object_mask given. Default is None.
            robot_collision_sphere_buffer: Override robot collision_sphere_buffer (m). Default is -0.01m.
            collision_activation_distance: Distance (m) to activate collision cost. Default is 0.01m.
            **kwargs: Passed through to plan_with_grasped_object.

        Returns:
            (success, joint_trajectory): joint_trajectory is (T, 7) or None.
        """
        if object_mask is not None:
            self.update_curobo_world_with_object(
                object_name,
                object_mask=object_mask,
                robot_distance_threshold=0.15,
            )
        world = world_config if world_config is not None else self._curobo_world_config
        if world is None:
            raise ValueError(
                "No world_config available. Call update_curobo_world_with_object (or pass object_mask) first."
            )

        obs = self._env.get_observation()
        start_joint_position = obs["robot_joint_pos"]
        object_name_for_attach = getattr(
            self, "_curobo_world_object_name", None
        ) or object_name.replace(" ", "_")

        debug_out_dir = getattr(self._env, "output_dir", None)
        success, joint_traj = _get_curobo_api().plan_with_grasped_object(
            world,
            start_joint_position,
            target_pose,
            object_name_for_attach,
            robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            collision_activation_distance=collision_activation_distance,
            debug_out_dir=debug_out_dir,
            **kwargs,
        )
        return success, joint_traj
