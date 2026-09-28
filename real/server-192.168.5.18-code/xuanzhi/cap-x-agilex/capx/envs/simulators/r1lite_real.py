"""ROS 2 low-level wrapper for a real Galaxea/Xinghaitu R1 Lite right arm.

This module intentionally exposes only joint-space arm control and gripper
position control.  End-effector target-pose control is not used because the
current R1 Lite setup has EE interpolation disabled and direct pose targets can
produce large unexpected motions.
"""

from __future__ import annotations

from collections import deque
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, SupportsFloat

import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv


class R1LiteRealLowLevel(BaseEnv):
    """Low-level ROS 2 interface for the R1 Lite right arm."""

    def __init__(
        self,
        *,
        privileged: bool = False,
        enable_render: bool = False,
        viser_debug: bool = False,
        arm_name: str = "right_arm",
        gripper_name: str = "right_gripper",
        arm_feedback_topic: str = "/hdas/feedback_arm_right",
        torso_feedback_topic: str = "/hdas/feedback_torso",
        gripper_feedback_topic: str = "/hdas/feedback_gripper_right",
        arm_target_topic: str = "/motion_target/target_joint_state_arm_right",
        arm_control_topic: str = "/motion_control/control_arm_right",
        brake_mode_topic: str = "/motion_target/brake_mode",
        gripper_target_topic: str = "/motion_target/target_position_gripper_right",
        lazy_moca_start_command: list[str] | tuple[str, ...] | None = None,
        lazy_moca_stop_command: list[str] | tuple[str, ...] | None = None,
        reset_motion_enabled: bool = True,
        wrist_rgb_topic: str = "/hdas/camera_wrist_right/color/image_raw",
        wrist_depth_topic: str = "/hdas/camera_wrist_right/aligned_depth_to_color/image_raw",
        wrist_camera_info_topic: str = "/hdas/camera_wrist_right/color/camera_info",
        head_left_rgb_topic: str = "/hdas/camera_head/left_raw/image_raw_color/compressed",
        head_right_rgb_topic: str = "/hdas/camera_head/right_raw/image_raw_color/compressed",
        head_left_rect_rgb_topic: str = "/capx/head_stereo/left/image_rect",
        head_left_depth_topic: str = "/capx/head_stereo/aligned_depth_to_left/image_raw",
        head_left_camera_info_topic: str = "/capx/head_stereo/aligned_depth_to_left/camera_info",
        wrist_camera_enabled: bool = True,
        visual_feedback_max_width: int = 640,
        sam3_service_url: str = "http://127.0.0.1:8114",
        sam3_pen_prompt: str = "blue pen",
        object_prompts: dict[str, str] | None = None,
        tennis_ball_radius_m: float = 0.034,
        head_stereo_tennis_z_offset_m: float = 0.0,
        box_drop_z_offset_m: float = 0.10,
        ee_pose_publish_hz: float = 30.0,
        moca_param_node: str = "/moca_adapter",
        graspnet_service_url: str = "http://127.0.0.1:8115",
        graspnet_input_mode: str = "point_clouds",
        graspnet_forward_passes: int = 1,
        graspnet_max_retries: int = 7,
        graspnet_request_timeout_s: float = 420.0,
        graspnet_local_regions: bool = True,
        graspnet_filter_grasps: bool = False,
        graspnet_max_full_points: int = 20000,
        graspnet_max_segment_points: int = 10000,
        graspnet_min_segment_points: int = 30,
        graspnet_depth_range: list[float] | tuple[float, float] = (0.05, 2.0),
        graspnet_segment_depth_filter_enabled: bool = True,
        graspnet_segment_depth_anchor_percentile: float = 10.0,
        graspnet_segment_depth_lower_margin_m: float = 0.02,
        graspnet_segment_depth_band_m: float = 0.12,
        graspnet_max_candidates: int = 16,
        graspnet_min_score: float = 0.0,
        graspnet_prefer_top_down: bool = True,
        graspnet_approach_axis: str = "z",
        graspnet_top_down_threshold: float = 0.65,
        graspnet_pregrasp_distance_m: float = 0.12,
        graspnet_ee_from_grasp: list[list[float]] | None = None,
        graspnet_observation_joints: list[float] | tuple[float, ...] | None = None,
        graspnet_observation_feedback_joints: (
            list[float] | tuple[float, ...] | None
        ) = None,
        graspnet_max_joint_delta_rad: float = 1.20,
        graspnet_max_approach_joint_delta_rad: float = 0.35,
        graspnet_approach_validation_steps: int = 12,
        graspnet_table_clearance_m: float = 0.02,
        wrist_tf_base_frame: str = "base_link",
        wrist_tf_timeout_s: float = 2.0,
        head_tf_max_skew_s: float = 3.0,
        wrist_rgbd_max_age_s: float = 1.0,
        wrist_rgbd_max_skew_s: float = 0.15,
        wrist_hand_eye_calibration_path: str = (
            "outputs/r1lite/right_touch_hand_eye_calibration/T_ee_camera.yaml"
        ),
        wrist_observation_reference_ee_position: (
            list[float] | tuple[float, ...]
        ) = (0.352568, -0.332416, 0.968292),
        wrist_observation_reference_ee_quaternion_xyzw: (
            list[float] | tuple[float, ...]
        ) = (0.019904, 0.406737, 0.015123, 0.913203),
        wrist_observation_distance_m: float = 0.30,
        wrist_observation_max_translation_m: float = 0.18,
        wrist_observation_max_joint_delta_rad: float = 1.30,
        wrist_observation_roll_candidates_deg: (
            list[float] | tuple[float, ...]
        ) = (0.0, -20.0, 20.0, -40.0, 40.0),
        wrist_observation_transit_clearance_candidates_m: (
            list[float] | tuple[float, ...]
        ) = (0.05, 0.08, 0.12, 0.16),
        wrist_ball_coarse_max_error_m: float = 0.12,
        wrist_ball_center_margin_ratio: float = 0.25,
        wrist_observation_settle_s: float = 0.6,
        r1lite_wrist_observation_skip_feedback_settle: bool = False,
        head_plane_calibration_path: str = "outputs/r1lite/head_plane_calibration.json",
        head_stereo_rectified_calibration_path: str = (
            "outputs/r1lite/head_stereo_rectified_plane_calibration.json"
        ),
        head_rgbd_median_sample_count: int = 30,
        r1lite_use_current_head_rgbd_table_plane: bool = False,
        workspace_x_limits: list[float] | tuple[float, float] = (0.05, 0.75),
        workspace_y_limits: list[float] | tuple[float, float] = (-0.65, 0.05),
        workspace_z_limits: list[float] | tuple[float, float] = (0.4, 1.6),
        r1lite_ik_harness_enabled: bool = True,
        r1lite_adaptive_joint_control_enabled: bool = True,
        r1lite_motion_validation_enabled: bool = True,
        right_arm_j2_feedback_upper_limit_rad: float = 3.05,
        right_arm_j2_command_lower_limit_rad: float = 0.01745329252,
        right_arm_j2_command_upper_limit_rad: float = 3.05,
        moca_output_corridor_margin_rad: float = 0.03,
        r1lite_require_pose_arrival: bool = False,
        relaxed_ik_probe_enabled: bool = False,
        relaxed_ik_target_topic: str = (
            "/capx/relaxed_ik/target_pose_arm_right"
        ),
        relaxed_ik_solution_topic: str = "/relaxed_ik/joint_angle_solutions_right",
        relaxed_ik_reset_topic: str = "/relaxed_ik/reset_right",
        relaxed_ik_fk_topic: str = "/relaxed_ik/ik_ee_pose_right",
        relaxed_ik_target_echo_topic: str = "/relaxed_ik/target_ee_pose_right",
        use_moca_ee_control: bool = False,
        ee_feedback_topic: str = "/motion_control/pose_ee_arm_right",
        relaxed_ik_ee_feedback_topic: str = "/relaxed_ik/motion_control/pose_ee_arm_right",
        pose_position_tolerance_m: float = 0.008,
        pose_orientation_tolerance_rad: float = 0.20,
        pose_command_timeout_s: float = 10.0,
        depth_scale: float = 0.001,
        safe_joint_home: list[float] | tuple[float, ...] = (
            0.010470,
            0.222985,
            -0.022772,
            0.008466,
            0.035299,
            0.006095,
        ),
        safe_joint_home_feedback: list[float] | tuple[float, ...] = (
            0.027234,
            0.448298,
            -0.005745,
            0.028936,
            0.058085,
            0.032340,
        ),
        gripper_open_position: float = 100.0,
        gripper_close_position: float = 0.0,
        gripper_feedback_tolerance: float = 5.0,
        gripper_min_response: float = 2.0,
        gripper_empty_close_feedback: float = 5.5,
        arm_min_motion_rad: float = 0.005,
        arm_motion_timeout_s: float = 2.0,
        arm_joint_tolerance_rad: float = 0.02,
        arm_divergence_limit_rad: float = 0.04,
        adaptive_joint_tolerance_rad: float = 0.004,
        adaptive_correction_fraction: float = 0.15,
        adaptive_probe_step_rad: float = 0.006,
        adaptive_probe_steps_rad: list[float] | tuple[float, ...] | None = None,
        adaptive_max_command_excursion_rad: float = 0.02,
        adaptive_max_target_step_rad: float = 0.02,
        adaptive_max_measured_step_rad: float = 0.04,
        adaptive_settle_s: float = 2.5,
        adaptive_feedback_stability_s: float = 0.25,
        adaptive_feedback_stability_rad: float = 0.002,
        adaptive_feedback_settle_timeout_s: float = 3.0,
        moca_startup_stability_timeout_s: float = 18.0,
        moca_startup_command_tolerance_rad: float = 0.03,
        adaptive_feedback_min_samples: int = 4,
        adaptive_max_iterations: int = 36,
        adaptive_max_stalled_iterations: int = 14,
        adaptive_min_progress_rad: float = 0.0005,
        adaptive_log_path: str = "outputs/r1lite/adaptive_joint_history.jsonl",
        max_joint_step: float = 0.03,
        goto_pose_joint_waypoint_step_rad: float = 0.015,
        publish_rate_hz: float = 20.0,
        command_duration_s: float = 3.0,
        feedback_timeout_s: float = 5.0,
        startup_timeout_s: float = 20.0,
        startup_required_feedback: (
            list[str] | tuple[str, ...]
        ) = ("arm", "gripper"),
        startup_required_command_subscribers: (
            list[str] | tuple[str, ...]
        ) = (),
        auto_spin: bool = True,
    ) -> None:
        super().__init__()
        self.privileged = privileged
        self.enable_render = enable_render
        self.viser_debug = viser_debug
        self.wrist_camera_enabled = bool(wrist_camera_enabled)
        self.arm_name = arm_name
        self.gripper_name = gripper_name
        self.safe_joint_home = np.asarray(safe_joint_home, dtype=np.float64).reshape(6)
        self.safe_joint_home_feedback = np.asarray(
            safe_joint_home_feedback, dtype=np.float64
        ).reshape(6)
        self.gripper_open_position = float(gripper_open_position)
        self.gripper_close_position = float(gripper_close_position)
        self.gripper_feedback_tolerance = float(max(0.0, gripper_feedback_tolerance))
        self.gripper_min_response = float(max(0.0, gripper_min_response))
        self.gripper_empty_close_feedback = float(gripper_empty_close_feedback)
        self.arm_min_motion_rad = float(max(0.0, arm_min_motion_rad))
        self.arm_motion_timeout_s = float(max(0.0, arm_motion_timeout_s))
        self.arm_joint_tolerance_rad = float(max(0.001, arm_joint_tolerance_rad))
        self.arm_divergence_limit_rad = float(max(0.001, arm_divergence_limit_rad))
        self.adaptive_joint_tolerance_rad = float(
            max(0.0005, adaptive_joint_tolerance_rad)
        )
        self.adaptive_correction_fraction = float(
            np.clip(adaptive_correction_fraction, 0.01, 1.0)
        )
        self.adaptive_probe_step_rad = float(max(0.001, adaptive_probe_step_rad))
        self.adaptive_probe_steps_rad = np.asarray(
            (
                [self.adaptive_probe_step_rad] * 6
                if adaptive_probe_steps_rad is None
                else adaptive_probe_steps_rad
            ),
            dtype=np.float64,
        ).reshape(6)
        if not np.isfinite(self.adaptive_probe_steps_rad).all():
            raise ValueError("adaptive_probe_steps_rad must be finite")
        self.adaptive_probe_steps_rad = np.maximum(
            self.adaptive_probe_steps_rad, 0.001
        )
        self.adaptive_max_command_excursion_rad = float(
            max(
                float(np.max(self.adaptive_probe_steps_rad)),
                adaptive_max_command_excursion_rad,
            )
        )
        self.adaptive_max_target_step_rad = float(
            max(self.adaptive_probe_step_rad, adaptive_max_target_step_rad)
        )
        self.adaptive_max_measured_step_rad = float(
            max(self.adaptive_probe_step_rad, adaptive_max_measured_step_rad)
        )
        self.adaptive_settle_s = float(max(0.1, adaptive_settle_s))
        self.adaptive_feedback_stability_s = float(
            max(0.05, adaptive_feedback_stability_s)
        )
        self.adaptive_feedback_stability_rad = float(
            max(0.0001, adaptive_feedback_stability_rad)
        )
        self.adaptive_feedback_settle_timeout_s = float(
            max(
                self.adaptive_feedback_stability_s,
                adaptive_feedback_settle_timeout_s,
            )
        )
        self.moca_startup_stability_timeout_s = float(
            max(
                self.adaptive_feedback_stability_s,
                moca_startup_stability_timeout_s,
            )
        )
        self.moca_startup_command_tolerance_rad = float(
            max(0.0005, moca_startup_command_tolerance_rad)
        )
        self.adaptive_feedback_min_samples = int(
            max(2, adaptive_feedback_min_samples)
        )
        self.adaptive_max_iterations = int(max(1, adaptive_max_iterations))
        self.adaptive_max_stalled_iterations = int(
            max(1, min(adaptive_max_iterations, adaptive_max_stalled_iterations))
        )
        self.adaptive_min_progress_rad = float(
            max(0.0, adaptive_min_progress_rad)
        )
        self.adaptive_log_path = str(adaptive_log_path)
        self.max_joint_step = float(max_joint_step)
        self.goto_pose_joint_waypoint_step_rad = float(
            goto_pose_joint_waypoint_step_rad
        )
        if not 0.0 < self.goto_pose_joint_waypoint_step_rad <= self.max_joint_step:
            raise ValueError(
                "goto_pose_joint_waypoint_step_rad must be positive and no greater "
                "than max_joint_step"
            )
        self.publish_rate_hz = float(publish_rate_hz)
        self.command_duration_s = float(command_duration_s)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.startup_timeout_s = float(max(feedback_timeout_s, startup_timeout_s))
        self.startup_required_feedback = tuple(
            str(name) for name in startup_required_feedback
        )
        self.startup_required_command_subscribers = tuple(
            str(name) for name in startup_required_command_subscribers
        )
        self.lazy_moca_start_command = (
            tuple(str(part) for part in lazy_moca_start_command)
            if lazy_moca_start_command
            else None
        )
        self.lazy_moca_stop_command = (
            tuple(str(part) for part in lazy_moca_stop_command)
            if lazy_moca_stop_command
            else None
        )
        self.reset_motion_enabled = bool(reset_motion_enabled)
        self.visual_feedback_max_width = int(visual_feedback_max_width)
        self.sam3_service_url = str(sam3_service_url)
        self.sam3_pen_prompt = str(sam3_pen_prompt)
        self.object_prompts = {
            "tennis ball": "yellow tennis ball",
            "cardboard box": "cardboard box",
            "blue pen": "blue pen",
        }
        if object_prompts:
            self.object_prompts.update({str(k): str(v) for k, v in object_prompts.items()})
        self.tennis_ball_radius_m = float(tennis_ball_radius_m)
        self.head_stereo_tennis_z_offset_m = float(head_stereo_tennis_z_offset_m)
        self.box_drop_z_offset_m = float(box_drop_z_offset_m)
        self.ee_pose_publish_hz = float(ee_pose_publish_hz)
        self.moca_param_node = str(moca_param_node)
        self._ee_control_ready = False
        self.graspnet_service_url = str(graspnet_service_url)
        self.graspnet_input_mode = str(graspnet_input_mode)
        self.graspnet_forward_passes = int(graspnet_forward_passes)
        self.graspnet_max_retries = int(graspnet_max_retries)
        self.graspnet_request_timeout_s = float(graspnet_request_timeout_s)
        self.graspnet_local_regions = bool(graspnet_local_regions)
        self.graspnet_filter_grasps = bool(graspnet_filter_grasps)
        self.graspnet_max_full_points = int(graspnet_max_full_points)
        self.graspnet_max_segment_points = int(graspnet_max_segment_points)
        self.graspnet_min_segment_points = int(graspnet_min_segment_points)
        self.graspnet_depth_range = tuple(float(x) for x in graspnet_depth_range)
        self.graspnet_segment_depth_filter_enabled = bool(
            graspnet_segment_depth_filter_enabled
        )
        self.graspnet_segment_depth_anchor_percentile = float(
            np.clip(graspnet_segment_depth_anchor_percentile, 0.0, 50.0)
        )
        self.graspnet_segment_depth_lower_margin_m = float(
            max(0.0, graspnet_segment_depth_lower_margin_m)
        )
        self.graspnet_segment_depth_band_m = float(
            max(0.01, graspnet_segment_depth_band_m)
        )
        self.graspnet_max_candidates = int(max(1, graspnet_max_candidates))
        self.graspnet_min_score = float(graspnet_min_score)
        self.graspnet_prefer_top_down = bool(graspnet_prefer_top_down)
        self.graspnet_approach_axis = str(graspnet_approach_axis)
        self.graspnet_top_down_threshold = float(graspnet_top_down_threshold)
        self.graspnet_pregrasp_distance_m = float(
            max(0.02, graspnet_pregrasp_distance_m)
        )
        self.graspnet_ee_from_grasp = (
            None
            if graspnet_ee_from_grasp is None
            else np.asarray(graspnet_ee_from_grasp, dtype=np.float64).reshape(4, 4)
        )
        self.graspnet_observation_joints = (
            None
            if graspnet_observation_joints is None
            else np.asarray(graspnet_observation_joints, dtype=np.float64).reshape(6)
        )
        self.graspnet_observation_feedback_joints = (
            None
            if graspnet_observation_feedback_joints is None
            else np.asarray(
                graspnet_observation_feedback_joints,
                dtype=np.float64,
            ).reshape(6)
        )
        self.graspnet_max_joint_delta_rad = float(
            max(0.01, graspnet_max_joint_delta_rad)
        )
        self.graspnet_max_approach_joint_delta_rad = float(
            max(0.01, graspnet_max_approach_joint_delta_rad)
        )
        self.graspnet_approach_validation_steps = int(
            max(2, graspnet_approach_validation_steps)
        )
        self.wrist_tf_base_frame = str(wrist_tf_base_frame)
        self.wrist_tf_timeout_s = float(max(0.1, wrist_tf_timeout_s))
        self.head_tf_max_skew_s = float(max(0.1, head_tf_max_skew_s))
        self.wrist_rgbd_max_age_s = float(max(0.05, wrist_rgbd_max_age_s))
        self.wrist_rgbd_max_skew_s = float(max(0.0, wrist_rgbd_max_skew_s))
        self.head_rgbd_max_age_s = self.wrist_rgbd_max_age_s
        self.head_rgbd_max_skew_s = self.wrist_rgbd_max_skew_s
        self.wrist_hand_eye_calibration_path = str(wrist_hand_eye_calibration_path)
        self.wrist_observation_reference_ee_position = np.asarray(
            wrist_observation_reference_ee_position,
            dtype=np.float64,
        ).reshape(3)
        reference_quaternion = np.asarray(
            wrist_observation_reference_ee_quaternion_xyzw,
            dtype=np.float64,
        ).reshape(4)
        reference_quaternion_norm = float(np.linalg.norm(reference_quaternion))
        if not np.isfinite(reference_quaternion_norm) or reference_quaternion_norm < 1e-8:
            raise ValueError("wrist observation reference quaternion is invalid")
        self.wrist_observation_reference_ee_quaternion_xyzw = (
            reference_quaternion / reference_quaternion_norm
        )
        self.wrist_observation_distance_m = float(
            max(0.15, wrist_observation_distance_m)
        )
        self.wrist_observation_max_translation_m = float(
            max(0.01, wrist_observation_max_translation_m)
        )
        self.wrist_observation_max_joint_delta_rad = float(
            max(0.10, wrist_observation_max_joint_delta_rad)
        )
        self.wrist_observation_roll_candidates_deg = tuple(
            float(value) for value in wrist_observation_roll_candidates_deg
        )
        if not self.wrist_observation_roll_candidates_deg:
            raise ValueError("wrist observation roll candidates cannot be empty")
        if any(
            abs(value) > 45.0
            for value in self.wrist_observation_roll_candidates_deg
        ):
            raise ValueError("wrist observation roll candidates exceed 45 degrees")
        self.wrist_observation_transit_clearance_candidates_m = tuple(
            float(value)
            for value in wrist_observation_transit_clearance_candidates_m
        )
        if not self.wrist_observation_transit_clearance_candidates_m or any(
            value < 0.03 or value > 0.20
            for value in self.wrist_observation_transit_clearance_candidates_m
        ):
            raise ValueError("invalid wrist observation transit clearances")
        self.wrist_ball_coarse_max_error_m = float(
            max(0.01, wrist_ball_coarse_max_error_m)
        )
        self.graspnet_table_clearance_m = float(
            max(0.0, graspnet_table_clearance_m)
        )
        self.wrist_ball_center_margin_ratio = float(
            np.clip(wrist_ball_center_margin_ratio, 0.05, 0.45)
        )
        self.wrist_observation_settle_s = float(
            max(0.1, wrist_observation_settle_s)
        )
        self.r1lite_wrist_observation_skip_feedback_settle = bool(
            r1lite_wrist_observation_skip_feedback_settle
        )
        self.head_plane_calibration_path = str(head_plane_calibration_path)
        self.head_stereo_rectified_calibration_path = str(
            head_stereo_rectified_calibration_path
        )
        self.head_rgbd_median_sample_count = int(head_rgbd_median_sample_count)
        if not 15 <= self.head_rgbd_median_sample_count <= 30:
            raise ValueError("head_rgbd_median_sample_count must be in [15, 30]")
        self.r1lite_use_current_head_rgbd_table_plane = bool(
            r1lite_use_current_head_rgbd_table_plane
        )
        self.workspace_x_limits = tuple(float(x) for x in workspace_x_limits)
        self.workspace_y_limits = tuple(float(y) for y in workspace_y_limits)
        self.workspace_z_limits = tuple(float(z) for z in workspace_z_limits)
        self.r1lite_ik_harness_enabled = bool(r1lite_ik_harness_enabled)
        self.r1lite_adaptive_joint_control_enabled = bool(
            r1lite_adaptive_joint_control_enabled
        )
        self.r1lite_motion_validation_enabled = bool(r1lite_motion_validation_enabled)
        self.right_arm_j2_feedback_upper_limit_rad = float(
            right_arm_j2_feedback_upper_limit_rad
        )
        self.right_arm_j2_command_lower_limit_rad = float(
            right_arm_j2_command_lower_limit_rad
        )
        self.right_arm_j2_command_upper_limit_rad = float(
            right_arm_j2_command_upper_limit_rad
        )
        self.moca_output_corridor_margin_rad = float(
            max(0.005, moca_output_corridor_margin_rad)
        )
        self.r1lite_require_pose_arrival = bool(r1lite_require_pose_arrival)
        self.relaxed_ik_probe_enabled = bool(relaxed_ik_probe_enabled)
        self.use_moca_ee_control = bool(use_moca_ee_control)
        self.pose_position_tolerance_m = float(max(0.001, pose_position_tolerance_m))
        self.pose_orientation_tolerance_rad = float(
            max(0.01, pose_orientation_tolerance_rad)
        )
        self.pose_command_timeout_s = float(max(1.0, pose_command_timeout_s))
        self.depth_scale = float(depth_scale)

        try:
            import rclpy
            from rclpy.duration import Duration
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.time import Time
            from tf2_ros import Buffer, TransformListener
            from geometry_msgs.msg import PoseStamped
            from rcl_interfaces.msg import Parameter as ParameterMsg
            from rcl_interfaces.msg import ParameterType, ParameterValue
            from rcl_interfaces.srv import SetParameters
            from sensor_msgs.msg import CameraInfo
            from sensor_msgs.msg import CompressedImage
            from sensor_msgs.msg import Image
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Bool
            from hdas_msg.msg import MotorControl
            from hdas_msg.srv import FunctionFrame
        except Exception as exc:  # pragma: no cover - depends on ROS 2 install
            raise RuntimeError(
                "R1LiteRealLowLevel requires ROS 2 Python packages. "
                "Run `source /opt/ros/humble/setup.bash` first."
            ) from exc

        self._rclpy = rclpy
        self._Duration = Duration
        self._Time = Time
        self._camera_qos = qos_profile_sensor_data
        self._ee_qos = QoSProfile(depth=1)
        self._ee_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._ee_qos.history = HistoryPolicy.KEEP_LAST
        self._command_qos = QoSProfile(depth=10)
        self._command_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._command_qos.history = HistoryPolicy.KEEP_LAST
        self._command_qos.durability = DurabilityPolicy.VOLATILE
        self._PoseStamped = PoseStamped
        self._SetParameters = SetParameters
        self._ParameterMsg = ParameterMsg
        self._ParameterValue = ParameterValue
        self._ParameterType = ParameterType
        self._CompressedImage = CompressedImage
        self._Image = Image
        self._CameraInfo = CameraInfo
        self._JointState = JointState
        self._Bool = Bool
        self._MotorControl = MotorControl
        self._FunctionFrame = FunctionFrame
        self._owns_rclpy_context = not rclpy.ok()
        self._closed = False
        if self._owns_rclpy_context:
            rclpy.init(args=None)
        self._node = rclpy.create_node(
            f"capx_r1lite_real_low_level_{os.getpid()}_{id(self):x}"
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self._node,
            spin_thread=False,
        )
        self._arm_pub = self._node.create_publisher(
            JointState, arm_target_topic, self._command_qos
        )
        self._brake_mode_pub = self._node.create_publisher(
            Bool, brake_mode_topic, self._command_qos
        )
        self._gripper_pub = self._node.create_publisher(
            JointState, gripper_target_topic, self._command_qos
        )
        self._relaxed_ik_pose_pub = self._node.create_publisher(
            PoseStamped, relaxed_ik_target_topic, self._ee_qos
        )
        self._relaxed_ik_reset_pub = self._node.create_publisher(
            JointState, relaxed_ik_reset_topic, 10
        )
        self._moca_param_client = self._node.create_client(
            SetParameters, f"{self.moca_param_node}/set_parameters"
        )
        self._right_arm_function_client = self._node.create_client(
            FunctionFrame, "/function_frame_right_arm"
        )
        self._arm_sub = self._node.create_subscription(
            JointState, arm_feedback_topic, self._on_arm_feedback, 10
        )
        self._torso_sub = self._node.create_subscription(
            JointState, torso_feedback_topic, self._on_torso_feedback, 10
        )
        self._arm_control_sub = self._node.create_subscription(
            MotorControl, arm_control_topic, self._on_arm_control, self._camera_qos
        )
        self._gripper_sub = self._node.create_subscription(
            JointState,
            gripper_feedback_topic,
            self._on_gripper_feedback,
            self._camera_qos,
        )
        self._rgb_sub = self._node.create_subscription(
            Image, wrist_rgb_topic, self._on_wrist_rgb, self._camera_qos
        )
        self._depth_sub = self._node.create_subscription(
            Image, wrist_depth_topic, self._on_wrist_depth, self._camera_qos
        )
        self._camera_info_sub = self._node.create_subscription(
            CameraInfo, wrist_camera_info_topic, self._on_wrist_camera_info, self._camera_qos
        )
        self._head_left_sub = self._node.create_subscription(
            CompressedImage,
            head_left_rgb_topic,
            self._on_head_left_rgb,
            self._camera_qos,
        )
        self._head_right_sub = self._node.create_subscription(
            CompressedImage,
            head_right_rgb_topic,
            self._on_head_right_rgb,
            self._camera_qos,
        )
        self._head_left_rect_sub = self._node.create_subscription(
            Image,
            head_left_rect_rgb_topic,
            self._on_head_left_rect_rgb,
            self._camera_qos,
        )
        self._head_left_depth_sub = self._node.create_subscription(
            Image,
            head_left_depth_topic,
            self._on_head_left_depth,
            self._camera_qos,
        )
        self._head_left_info_sub = self._node.create_subscription(
            CameraInfo,
            head_left_camera_info_topic,
            self._on_head_left_camera_info,
            self._camera_qos,
        )
        self._relaxed_ik_solution_sub = self._node.create_subscription(
            JointState,
            relaxed_ik_solution_topic,
            self._on_relaxed_ik_solution,
            10,
        )
        self._relaxed_ik_fk_sub = self._node.create_subscription(
            PoseStamped,
            relaxed_ik_fk_topic,
            self._on_relaxed_ik_fk_pose,
            self._ee_qos,
        )
        self._relaxed_ik_target_echo_sub = self._node.create_subscription(
            PoseStamped,
            relaxed_ik_target_echo_topic,
            self._on_relaxed_ik_target_echo,
            self._ee_qos,
        )
        self._ee_pose_sub = self._node.create_subscription(
            PoseStamped,
            ee_feedback_topic,
            self._on_ee_pose,
            self._camera_qos,
        )
        self._relaxed_ik_ee_pose_sub = self._node.create_subscription(
            PoseStamped,
            relaxed_ik_ee_feedback_topic,
            self._on_relaxed_ik_ee_pose,
            self._camera_qos,
        )
        self._feedback_subscriptions = {
            "arm": self._arm_sub,
            "gripper": self._gripper_sub,
            "torso": self._torso_sub,
            "rgb": self._rgb_sub,
            "depth": self._depth_sub,
            "camera_info": self._camera_info_sub,
            "head_left_rgb": self._head_left_sub,
            "head_right_rgb": self._head_right_sub,
            "head_left_rect_rgb": self._head_left_rect_sub,
            "head_left_depth": self._head_left_depth_sub,
            "head_left_camera_info": self._head_left_info_sub,
            "relaxed_ik_solution": self._relaxed_ik_solution_sub,
            "relaxed_ik_fk": self._relaxed_ik_fk_sub,
            "relaxed_ik_target_echo": self._relaxed_ik_target_echo_sub,
            "ee_pose": self._ee_pose_sub,
            "relaxed_ik_ee_pose": self._relaxed_ik_ee_pose_sub,
        }
        self._command_publishers = {
            "arm": self._arm_pub,
            "gripper": self._gripper_pub,
            "brake_mode": self._brake_mode_pub,
            "relaxed_ik_pose": self._relaxed_ik_pose_pub,
            "relaxed_ik_reset": self._relaxed_ik_reset_pub,
        }

        self._lock = threading.Lock()
        self._latest_arm: np.ndarray | None = None
        self._latest_torso: np.ndarray | None = None
        self._latest_arm_monotonic: float | None = None
        self._arm_feedback_sequence = 0
        self._arm_feedback_history: deque[
            tuple[int, float, np.ndarray]
        ] = deque(maxlen=512)
        self._latest_gripper: float | None = None
        self._latest_wrist_rgb: np.ndarray | None = None
        self._latest_wrist_rgb_monotonic: float | None = None
        self._latest_wrist_rgb_stamp_ns: int | None = None
        self._latest_wrist_rgb_frame_id: str = ""
        self._latest_wrist_depth: np.ndarray | None = None
        self._latest_wrist_depth_monotonic: float | None = None
        self._latest_wrist_depth_stamp_ns: int | None = None
        self._latest_wrist_depth_frame_id: str = ""
        self._latest_wrist_intrinsics: np.ndarray | None = None
        self._latest_wrist_info_monotonic: float | None = None
        self._latest_wrist_info_stamp_ns: int | None = None
        self._latest_wrist_info_frame_id: str = ""
        self._latest_wrist_info_size: tuple[int, int] | None = None
        self._latest_head_left_rgb: np.ndarray | None = None
        self._latest_head_right_rgb: np.ndarray | None = None
        self._latest_head_left_rect_rgb: np.ndarray | None = None
        self._latest_head_left_rect_rgb_monotonic: float | None = None
        self._latest_head_left_rect_rgb_stamp_ns: int | None = None
        self._latest_head_left_rect_rgb_frame_id: str = ""
        self._latest_head_left_depth: np.ndarray | None = None
        self._latest_head_left_depth_monotonic: float | None = None
        self._latest_head_left_depth_stamp_ns: int | None = None
        self._latest_head_left_depth_frame_id: str = ""
        self._latest_head_left_intrinsics: np.ndarray | None = None
        self._latest_head_left_info_monotonic: float | None = None
        self._latest_head_left_info_stamp_ns: int | None = None
        self._latest_head_left_info_frame_id: str = ""
        self._latest_head_left_info_size: tuple[int, int] | None = None
        self._last_motion_barrier_ns: int | None = None
        self._last_motion_barrier_monotonic: float | None = None
        self._latest_relaxed_ik_solution: np.ndarray | None = None
        self._latest_relaxed_ik_stamp: float = 0.0
        self._latest_relaxed_ik_solution_monotonic: float = 0.0
        self._latest_relaxed_ik_solution_stamp_ns: int = 0
        self._latest_relaxed_ik_fk_position: np.ndarray | None = None
        self._latest_relaxed_ik_fk_quaternion_xyzw: np.ndarray | None = None
        self._latest_relaxed_ik_fk_monotonic: float = 0.0
        self._latest_relaxed_ik_fk_stamp_ns: int = 0
        self._latest_relaxed_ik_target_position: np.ndarray | None = None
        self._latest_relaxed_ik_target_quaternion_xyzw: np.ndarray | None = None
        self._latest_relaxed_ik_target_stamp_ns: int = 0
        self._latest_relaxed_ik_target_monotonic: float = 0.0
        self._latest_ee_position: np.ndarray | None = None
        self._latest_ee_quaternion_xyzw: np.ndarray | None = None
        self._latest_ee_pose_monotonic: float | None = None
        self._latest_relaxed_ik_ee_position: np.ndarray | None = None
        self._latest_relaxed_ik_ee_quaternion_xyzw: np.ndarray | None = None
        self._local_planning_kinematics: Any | None = None
        self._relaxed_ik_initialized = False
        self._head_plane_calibration: dict[str, Any] | None = None
        self._head_stereo_rectified_calibration: dict[str, Any] | None = None
        self._wrist_hand_eye_calibration: np.ndarray | None = None
        self._last_action: dict[str, Any] = {}
        self._last_joint_command: np.ndarray | None = None
        self._moca_started_by_env = False
        self._moca_owned_by_env = False
        self._moca_startup_stabilized = False
        self._moca_lock_file: Any | None = None
        self._motion_fault_latched = False
        self._motion_fault_record: dict[str, Any] | None = None
        self._active_moca_command_corridor: (
            tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ) = None
        self._moca_command_corridor_generation = 0
        self._moca_output_fault_thread: threading.Thread | None = None
        self._pending_gripper_target: float | None = None
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread: threading.Thread | None = None
        if auto_spin:
            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                daemon=True,
                name="capx_r1lite_spin",
            )
            self._spin_thread.start()

    def _on_arm_feedback(self, msg: Any) -> None:
        if msg.name and self.arm_name not in msg.name:
            return
        if len(msg.position) < 6:
            return
        feedback = np.asarray(msg.position[:6], dtype=np.float64)
        if not np.isfinite(feedback).all():
            return
        received_at = time.monotonic()
        with self._lock:
            self._latest_arm = feedback
            self._latest_arm_monotonic = received_at
            self._arm_feedback_sequence += 1
            self._arm_feedback_history.append(
                (
                    self._arm_feedback_sequence,
                    received_at,
                    feedback.copy(),
                )
            )

    def _on_torso_feedback(self, msg: Any) -> None:
        if len(msg.position) < 3:
            return
        feedback = np.asarray(msg.position[:3], dtype=np.float64)
        if not np.isfinite(feedback).all():
            return
        with self._lock:
            self._latest_torso = feedback

    def _on_arm_control(self, msg: Any) -> None:
        """Track the command MOCA actually sends instead of guessing its offset."""
        if len(msg.p_des) < 6:
            return
        command = np.asarray(msg.p_des[:6], dtype=np.float64)
        if not np.isfinite(command).all():
            return
        fault_record = None
        with self._lock:
            self._last_joint_command = command
            corridor = getattr(self, "_active_moca_command_corridor", None)
            if (
                corridor is not None
                and not getattr(self, "_motion_fault_latched", False)
            ):
                generation, lower, upper, requested, start = corridor
                violation = np.maximum(lower - command, command - upper)
                max_violation = float(np.max(violation))
                if max_violation > 0.0:
                    active_joint = int(np.argmax(violation))
                    fault_record = {
                        "timestamp": time.time(),
                        "reason": "MOCA interpolated command left safety corridor",
                        "details": {
                            "generation": generation,
                            "active_joint": active_joint,
                            "moca_command": command.tolist(),
                            "requested_command": requested.tolist(),
                            "corridor_start": start.tolist(),
                            "corridor_lower": lower.tolist(),
                            "corridor_upper": upper.tolist(),
                            "violation_rad": max_violation,
                        },
                    }
                    self._motion_fault_latched = True
                    self._motion_fault_record = fault_record
                    self._last_action["motion_fault"] = fault_record
                    self._pending_gripper_target = None
                    self._active_moca_command_corridor = None
        if fault_record is not None:
            # The callback sees MOCA's 200 Hz output.  Stop it from a separate
            # thread so the ROS executor remains responsive while the vendor
            # process is terminated.
            self._moca_output_fault_thread = threading.Thread(
                target=self._stop_moca_after_output_fault,
                args=(fault_record,),
                daemon=True,
                name="capx_moca_output_fault_stop",
            )
            self._moca_output_fault_thread.start()

    def _stop_moca_after_output_fault(self, record: dict[str, Any]) -> None:
        try:
            # Startup output can violate the corridor before ownership has
            # been marked, so stop the vendor process unconditionally.
            self._run_moca_stop_command()
            self._wait_for_moca_to_stop()
        except Exception as exc:
            record["moca_stop_error"] = repr(exc)
        finally:
            self._moca_started_by_env = False
            self._moca_owned_by_env = False
            self._moca_startup_stabilized = False
            self._release_moca_lock()

    def _on_gripper_feedback(self, msg: Any) -> None:
        if msg.name and self.gripper_name in msg.name:
            idx = list(msg.name).index(self.gripper_name)
        else:
            idx = 0
        if len(msg.position) <= idx:
            return
        with self._lock:
            self._latest_gripper = float(msg.position[idx])

    def _on_wrist_rgb(self, msg: Any) -> None:
        rgb = self._image_to_numpy(msg)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        encoding = str(getattr(msg, "encoding", "")).lower()
        if encoding in {"bgr8", "bgra8"}:
            rgb = rgb[..., ::-1]
        with self._lock:
            self._latest_wrist_rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            self._latest_wrist_rgb_monotonic = time.monotonic()
            self._latest_wrist_rgb_stamp_ns = self._message_stamp_ns(msg)
            self._latest_wrist_rgb_frame_id = str(msg.header.frame_id).strip()

    def _on_wrist_depth(self, msg: Any) -> None:
        depth = self._image_to_numpy(msg)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * self.depth_scale
        else:
            depth = depth.astype(np.float32)
        with self._lock:
            self._latest_wrist_depth = depth
            self._latest_wrist_depth_monotonic = time.monotonic()
            self._latest_wrist_depth_stamp_ns = self._message_stamp_ns(msg)
            self._latest_wrist_depth_frame_id = str(msg.header.frame_id).strip()

    def _on_wrist_camera_info(self, msg: Any) -> None:
        with self._lock:
            self._latest_wrist_intrinsics = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            self._latest_wrist_info_monotonic = time.monotonic()
            self._latest_wrist_info_stamp_ns = self._message_stamp_ns(msg)
            self._latest_wrist_info_frame_id = str(msg.header.frame_id).strip()
            self._latest_wrist_info_size = (int(msg.height), int(msg.width))

    def _on_head_left_rgb(self, msg: Any) -> None:
        with self._lock:
            self._latest_head_left_rgb = self._compressed_image_to_rgb(msg)

    def _on_head_right_rgb(self, msg: Any) -> None:
        with self._lock:
            self._latest_head_right_rgb = self._compressed_image_to_rgb(msg)

    def _on_head_left_rect_rgb(self, msg: Any) -> None:
        rgb = self._image_to_numpy(msg)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        encoding = str(getattr(msg, "encoding", "")).lower()
        if encoding in {"bgr8", "bgra8"}:
            rgb = rgb[..., ::-1]
        with self._lock:
            self._latest_head_left_rect_rgb = np.ascontiguousarray(
                rgb,
                dtype=np.uint8,
            )
            self._latest_head_left_rect_rgb_monotonic = time.monotonic()
            self._latest_head_left_rect_rgb_stamp_ns = self._message_stamp_ns(msg)
            self._latest_head_left_rect_rgb_frame_id = str(
                msg.header.frame_id
            ).strip()

    def _on_head_left_depth(self, msg: Any) -> None:
        depth = self._image_to_numpy(msg)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * self.depth_scale
        else:
            depth = depth.astype(np.float32)
        with self._lock:
            self._latest_head_left_depth = depth
            self._latest_head_left_depth_monotonic = time.monotonic()
            self._latest_head_left_depth_stamp_ns = self._message_stamp_ns(msg)
            self._latest_head_left_depth_frame_id = str(msg.header.frame_id).strip()

    def _on_head_left_camera_info(self, msg: Any) -> None:
        with self._lock:
            self._latest_head_left_intrinsics = np.asarray(
                msg.k,
                dtype=np.float64,
            ).reshape(3, 3)
            self._latest_head_left_info_monotonic = time.monotonic()
            self._latest_head_left_info_stamp_ns = self._message_stamp_ns(msg)
            self._latest_head_left_info_frame_id = str(msg.header.frame_id).strip()
            self._latest_head_left_info_size = (int(msg.height), int(msg.width))

    def _on_relaxed_ik_solution(self, msg: Any) -> None:
        if len(msg.position) < 6:
            return
        with self._lock:
            self._latest_relaxed_ik_solution = np.asarray(
                msg.position[:6], dtype=np.float64
            )
            self._latest_relaxed_ik_stamp = time.time()
            self._latest_relaxed_ik_solution_monotonic = time.monotonic()
            self._latest_relaxed_ik_solution_stamp_ns = (
                self._message_stamp_ns(msg)
            )

    def _on_relaxed_ik_fk_pose(self, msg: Any) -> None:
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            return
        with self._lock:
            self._latest_relaxed_ik_fk_position = position
            self._latest_relaxed_ik_fk_quaternion_xyzw = quaternion
            self._latest_relaxed_ik_fk_monotonic = time.monotonic()
            self._latest_relaxed_ik_fk_stamp_ns = self._message_stamp_ns(msg)

    def _on_relaxed_ik_target_echo(self, msg: Any) -> None:
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            return
        with self._lock:
            self._latest_relaxed_ik_target_position = position
            self._latest_relaxed_ik_target_quaternion_xyzw = quaternion
            self._latest_relaxed_ik_target_stamp_ns = self._message_stamp_ns(msg)
            self._latest_relaxed_ik_target_monotonic = time.monotonic()

    def _on_ee_pose(self, msg: Any) -> None:
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            return
        with self._lock:
            self._latest_ee_position = position
            self._latest_ee_quaternion_xyzw = quaternion
            self._latest_ee_pose_monotonic = time.monotonic()

    def _on_relaxed_ik_ee_pose(self, msg: Any) -> None:
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            return
        with self._lock:
            self._latest_relaxed_ik_ee_position = position
            self._latest_relaxed_ik_ee_quaternion_xyzw = quaternion

    def _compressed_image_to_rgb(self, msg: Any) -> np.ndarray:
        import io

        img = PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def _image_to_numpy(self, msg: Any) -> np.ndarray:
        encoding = str(getattr(msg, "encoding", "")).lower()
        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
            "8uc1": 1,
            "16uc1": 1,
            "32fc1": 1,
        }.get(encoding)
        dtype = {
            "rgb8": np.uint8,
            "bgr8": np.uint8,
            "rgba8": np.uint8,
            "bgra8": np.uint8,
            "mono8": np.uint8,
            "8uc1": np.uint8,
            "16uc1": np.uint16,
            "32fc1": np.float32,
        }.get(encoding)
        if channels is None or dtype is None:
            # Best effort fallback for common color streams that omit encoding.
            dtype = np.uint8
            channels = max(1, int(msg.step) // max(1, int(msg.width)))
        arr = np.frombuffer(msg.data, dtype=dtype)
        height = int(msg.height)
        width = int(msg.width)
        if channels == 1:
            return arr.reshape(height, width)
        return arr.reshape(height, width, channels)

    @staticmethod
    def _message_stamp_ns(msg: Any) -> int:
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _make_joint_state(self, name: str, position: list[float]) -> Any:
        msg = self._JointState()
        msg.position = [float(x) for x in position]
        # R1 Lite target topics are positional; the vendor examples leave names empty.
        msg.name = []
        if name == self.arm_name:
            # Match Galaxea's R1 Lite joint command example exactly.
            msg.velocity = [1.6, 1.6, 1.6, 4.0, 4.0, 4.0]
        return msg

    def _enable_arm_joint_control(
        self,
        *,
        reset_function_frame: bool = False,
        maximum_reset_drift_rad: float = 0.03,
    ) -> None:
        startup_reference = self.get_arm_joints()
        with self._lock:
            startup_feedback_sequence = self._arm_feedback_sequence
        if reset_function_frame:
            local_home_bias = self.safe_joint_home - self.safe_joint_home_feedback
            self._ensure_moca_for_joint_target(
                startup_reference + local_home_bias
            )
            deadline = time.monotonic() + self.feedback_timeout_s
            while time.monotonic() < deadline:
                with self._lock:
                    has_control_command = self._last_joint_command is not None
                if has_control_command:
                    break
                self._spin_once_or_sleep()
            else:
                self._stop_lazy_moca()
                raise TimeoutError(
                    "MOCA did not publish p_des before FunctionFrame reset"
                )
            time.sleep(0.5)
            startup_measured = self.get_arm_joints()
            startup_drift = float(
                np.max(np.abs(startup_measured - startup_reference))
            )
            if startup_drift > float(maximum_reset_drift_rad):
                self._enter_motion_fault(
                    reason="right arm moved during MOCA startup hold",
                    details={
                        "measured_before": startup_reference.tolist(),
                        "measured_after": startup_measured.tolist(),
                        "drift_rad": startup_drift,
                        "limit_rad": maximum_reset_drift_rad,
                    },
                )
                raise RuntimeError(
                    "right arm moved during MOCA startup hold: "
                    f"drift={startup_drift:.4f}rad, "
                    f"limit={maximum_reset_drift_rad:.4f}rad"
                )
            reference = self.get_arm_joints()
            with self._lock:
                hold_command = (
                    None
                    if self._last_joint_command is None
                    else self._last_joint_command.copy()
                )
            if hold_command is None:
                raise RuntimeError(
                    "MOCA p_des is unavailable; refusing FunctionFrame reset"
                )
            hold_message = self._make_joint_state(
                self.arm_name, hold_command.tolist()
            )
            hold_stop = threading.Event()

            def publish_latched_hold() -> None:
                while not hold_stop.wait(0.02):
                    self._arm_pub.publish(hold_message)

            hold_thread = threading.Thread(
                target=publish_latched_hold,
                name="r1lite-function-frame-hold",
                daemon=True,
            )
            hold_thread.start()
            if not self._right_arm_function_client.wait_for_service(timeout_sec=5.0):
                hold_stop.set()
                hold_thread.join(timeout=1.0)
                raise RuntimeError("right-arm FunctionFrame service is unavailable")
            try:
                for command in (2, 5, 1):
                    request = self._FunctionFrame.Request()
                    request.command = command
                    future = self._right_arm_function_client.call_async(request)
                    deadline = time.monotonic() + 8.0
                    while not future.done() and time.monotonic() < deadline:
                        self._spin_once_or_sleep()
                    response = future.result() if future.done() else None
                    if response is None or not response.success:
                        message = "timeout" if response is None else response.message
                        raise RuntimeError(
                            f"right-arm FunctionFrame command {command} failed: "
                            f"{message}"
                        )
                    drift = float(
                        np.max(np.abs(self.get_arm_joints() - reference))
                    )
                    if drift > float(maximum_reset_drift_rad):
                        raise RuntimeError(
                            "right arm moved during FunctionFrame reset: "
                            f"drift={drift:.4f}rad, "
                            f"limit={maximum_reset_drift_rad:.4f}rad"
                        )
            finally:
                hold_stop.set()
                hold_thread.join(timeout=1.0)
                for _ in range(10):
                    self._arm_pub.publish(hold_message)
                    time.sleep(0.02)
        # The isolated MOCA process is intentionally started lazily.  It owns
        # both the joint-target and brake-mode subscriptions.  HDAS feedback
        # and MOCA targets retain the calibrated power-on offset even after a
        # FunctionFrame reset, so convert the measured hold to command space.
        # Always establish an explicit finite hold target.  A pre-existing
        # vendor MOCA subscriber is not proof that this process owns or seeded
        # the session; skipping the seed leaves the startup command corridor
        # undefined and makes stability validation impossible.
        measured_hold = self.get_arm_joints()
        local_home_bias = self.safe_joint_home - self.safe_joint_home_feedback
        startup_hold_command = measured_hold + local_home_bias
        startup_hold_command[1] = max(
            float(startup_hold_command[1]),
            float(
                getattr(
                    self,
                    "right_arm_j2_command_lower_limit_rad",
                    0.01745329252,
                )
            ),
        )
        self._ensure_moca_for_joint_target(startup_hold_command)
        self._wait_for_subscriber(self._brake_mode_pub, "R1 brake mode")
        msg = self._Bool()
        msg.data = True
        self._brake_mode_pub.publish(msg)
        if (
            reset_function_frame
            or not getattr(self, "_moca_startup_stabilized", False)
        ):
            self._wait_for_moca_startup_stability(
                reference=startup_reference,
                after_sequence=startup_feedback_sequence,
                maximum_drift_rad=(
                    float(maximum_reset_drift_rad)
                    if reset_function_frame
                    else max(
                        float(
                            getattr(self, "adaptive_max_measured_step_rad", 0.04)
                        ),
                        float(
                            np.max(
                                np.abs(
                                    self.safe_joint_home_feedback
                                    - self.safe_joint_home
                                )
                            )
                        )
                        + float(
                            getattr(
                                self,
                                "moca_output_corridor_margin_rad",
                                0.03,
                            )
                        ),
                    )
                ),
            )
            self._moca_startup_stabilized = True

    def enable_arm_joint_control_safely(
        self, *, maximum_reset_drift_rad: float = 0.03
    ) -> None:
        """Reset arm mode while continuously replaying the live MOCA hold."""
        self._enable_arm_joint_control(
            reset_function_frame=True,
            maximum_reset_drift_rad=maximum_reset_drift_rad,
        )

    def enable_arm_joint_control_without_function_reset(self) -> None:
        """Enable an existing arm mode only after MOCA's hold becomes stable."""
        self._enable_arm_joint_control(reset_function_frame=False)

    def _wait_for_moca_startup_stability(
        self,
        *,
        reference: np.ndarray,
        after_sequence: int,
        maximum_drift_rad: float,
    ) -> dict[str, float]:
        """Require a fresh stable feedback window before the first motion."""
        if not bool(getattr(self, "r1lite_motion_validation_enabled", True)):
            return {"maximum_step_rad": 0.0, "maximum_drift_rad": 0.0}
        reference_arr = np.asarray(reference, dtype=np.float64).reshape(6)
        startup_timeout_s = float(
            getattr(
                self,
                "moca_startup_stability_timeout_s",
                max(10.0, self.adaptive_feedback_settle_timeout_s),
            )
        )
        deadline = time.monotonic() + startup_timeout_s
        last_sequence = int(after_sequence)
        previous = reference_arr.copy()
        maximum_step = 0.0
        maximum_drift = 0.0
        latest_command_error = float("inf")
        minimum_command_error = float("inf")
        latest_moca_command: np.ndarray | None = None
        latest_requested_command: np.ndarray | None = None
        samples: deque[tuple[float, np.ndarray]] = deque()
        while time.monotonic() < deadline:
            self._assert_motion_ready()
            with self._lock:
                fresh = [
                    (sequence, received_at, feedback.copy())
                    for sequence, received_at, feedback in self._arm_feedback_history
                    if sequence > last_sequence
                ]
            for sequence, received_at, feedback in fresh:
                step = float(np.max(np.abs(feedback - previous)))
                drift = float(np.max(np.abs(feedback - reference_arr)))
                maximum_step = max(maximum_step, step)
                maximum_drift = max(maximum_drift, drift)
                previous = feedback
                last_sequence = max(last_sequence, sequence)
                samples.append((received_at, feedback))
                if (
                    step > self.adaptive_max_measured_step_rad
                    or drift > float(maximum_drift_rad)
                ):
                    details = {
                        "reference": reference_arr.tolist(),
                        "measured": feedback.tolist(),
                        "maximum_step_rad": maximum_step,
                        "maximum_drift_rad": maximum_drift,
                        "step_limit_rad": self.adaptive_max_measured_step_rad,
                        "drift_limit_rad": float(maximum_drift_rad),
                    }
                    self._enter_motion_fault(
                        reason="right arm moved before MOCA startup stabilized",
                        details=details,
                    )
                    raise RuntimeError(
                        "right arm moved before MOCA startup stabilized: "
                        f"step={maximum_step:.4f}rad, "
                        f"drift={maximum_drift:.4f}rad"
                    )
            if samples:
                newest_at = samples[-1][0]
                window_start = newest_at - self.adaptive_feedback_stability_s
                while len(samples) > 1 and samples[1][0] <= window_start:
                    samples.popleft()
                if (
                    len(samples) >= self.adaptive_feedback_min_samples
                    and newest_at - samples[0][0]
                    >= self.adaptive_feedback_stability_s
                ):
                    values = np.asarray(
                        [feedback for _, feedback in samples],
                        dtype=np.float64,
                    )
                    settled_feedback = np.median(values, axis=0)
                    settled_drift = float(
                        np.max(np.abs(settled_feedback - reference_arr))
                    )
                    with self._lock:
                        moca_command = (
                            None
                            if self._last_joint_command is None
                            else self._last_joint_command.copy()
                        )
                        corridor = self._active_moca_command_corridor
                        requested_command = (
                            None if corridor is None else corridor[3].copy()
                        )
                    command_error = (
                        float("inf")
                        if moca_command is None or requested_command is None
                        else float(
                            np.max(
                                np.abs(moca_command - requested_command)
                            )
                        )
                    )
                    latest_command_error = command_error
                    minimum_command_error = min(
                        minimum_command_error,
                        command_error,
                    )
                    latest_moca_command = moca_command
                    latest_requested_command = requested_command
                    settled_drift_limit = min(
                        float(maximum_drift_rad),
                        float(self.adaptive_max_measured_step_rad),
                    )
                    if (
                        float(np.max(np.ptp(values, axis=0)))
                        <= self.adaptive_feedback_stability_rad
                        and settled_drift <= settled_drift_limit
                        and command_error
                        <= self.moca_startup_command_tolerance_rad
                    ):
                        details = {
                            "maximum_step_rad": maximum_step,
                            "maximum_drift_rad": maximum_drift,
                            "settled_drift_rad": settled_drift,
                            "settled_drift_limit_rad": settled_drift_limit,
                            "moca_command_error_rad": command_error,
                            "moca_command_tolerance_rad": (
                                self.moca_startup_command_tolerance_rad
                            ),
                        }
                        self._last_action["moca_startup_stability"] = details
                        return details
            time.sleep(1.0 / self.publish_rate_hz)
        details = {
            "reference": reference_arr.tolist(),
            "maximum_step_rad": maximum_step,
            "maximum_drift_rad": maximum_drift,
            "latest_moca_command_error_rad": latest_command_error,
            "minimum_moca_command_error_rad": minimum_command_error,
            "moca_command_tolerance_rad": (
                self.moca_startup_command_tolerance_rad
            ),
            "latest_moca_command": (
                None
                if latest_moca_command is None
                else latest_moca_command.tolist()
            ),
            "latest_requested_command": (
                None
                if latest_requested_command is None
                else latest_requested_command.tolist()
            ),
            "timeout_s": startup_timeout_s,
        }
        self._enter_motion_fault(
            reason="MOCA startup feedback did not stabilize",
            details=details,
        )
        raise TimeoutError(
            "MOCA startup feedback did not stabilize before motion"
        )

    def _wait_for_subscriber(self, publisher: Any, context: str) -> None:
        """Do not send a finite command before DDS endpoint matching completes."""
        timeout_s = (
            self.startup_timeout_s
            if "joint target" in context or "gripper" in context
            else self.feedback_timeout_s
        )
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if int(publisher.get_subscription_count()) > 0:
                return
            self._spin_once_or_sleep()
        raise TimeoutError(f"Timed out waiting for a subscriber to {context}")

    def _assert_motion_ready(self) -> None:
        if getattr(self, "_motion_fault_latched", False):
            raise RuntimeError(
                "right-arm motion fault is latched; create a new robot session "
                f"after inspection: {self._motion_fault_record}"
            )

    def motion_fault_record(self) -> dict[str, Any] | None:
        record = getattr(self, "_motion_fault_record", None)
        return None if record is None else dict(record)

    def _acquire_moca_lock(self) -> None:
        if getattr(self, "_moca_lock_file", None) is not None:
            return
        lock_file = open("/tmp/capx_r1lite_moca.lock", "a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError(
                "another Cap-X process owns the right-arm MOCA session"
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        self._moca_lock_file = lock_file

    def _release_moca_lock(self) -> None:
        lock_file = getattr(self, "_moca_lock_file", None)
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            self._moca_lock_file = None

    def _run_moca_stop_command(self) -> None:
        if self.lazy_moca_stop_command is None:
            return
        try:
            subprocess.run(
                self.lazy_moca_stop_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=25.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _wait_for_moca_to_stop(self, timeout_s: float = 15.0) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            if int(self._arm_pub.get_subscription_count()) == 0:
                return
            self._spin_once_or_sleep()
        raise TimeoutError("right-arm MOCA subscriber remained after stop")

    def _ensure_moca_for_joint_target(self, command: np.ndarray) -> None:
        """Start MOCA only after a complete finite arm target is available."""
        self._assert_motion_ready()
        target = np.asarray(command, dtype=np.float64).reshape(6)
        if not np.isfinite(target).all():
            raise ValueError("MOCA joint target must contain six finite values")
        if (
            getattr(self, "_moca_owned_by_env", False)
            and int(self._arm_pub.get_subscription_count()) > 0
        ):
            return
        if self.lazy_moca_start_command is None:
            self._wait_for_subscriber(self._arm_pub, "right arm joint target")
            return
        self._acquire_moca_lock()
        self._moca_startup_stabilized = False
        try:
            if int(self._arm_pub.get_subscription_count()) > 0:
                self._run_moca_stop_command()
                self._wait_for_moca_to_stop()
        except Exception:
            self._release_moca_lock()
            raise

        # Publish the finite hold target throughout MOCA startup.  Waiting for
        # the launcher to return before publishing leaves a race in which MOCA
        # can emit its default p_des and move the arm away from its measured
        # pose before the first Cap-X target arrives.
        msg = self._make_joint_state(self.arm_name, target.tolist())
        startup_command = self.get_arm_joints()
        with self._lock:
            margin = float(
                getattr(self, "moca_output_corridor_margin_rad", 0.03)
            )
            self._moca_command_corridor_generation = (
                int(getattr(self, "_moca_command_corridor_generation", 0)) + 1
            )
            generation = self._moca_command_corridor_generation
            self._active_moca_command_corridor = (
                generation,
                np.minimum(startup_command, target) - margin,
                np.maximum(startup_command, target) + margin,
                target.copy(),
                startup_command,
            )
        seed_stop = threading.Event()

        def publish_startup_seed() -> None:
            while not seed_stop.wait(0.02):
                self._arm_pub.publish(msg)

        seed_thread = threading.Thread(
            target=publish_startup_seed,
            name="r1lite-moca-startup-seed",
            daemon=True,
        )
        seed_thread.start()
        try:
            result = subprocess.run(
                self.lazy_moca_start_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(10.0, self.feedback_timeout_s),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            seed_stop.set()
            seed_thread.join(timeout=1.0)
            self._run_moca_stop_command()
            self._release_moca_lock()
            raise RuntimeError(f"Failed to start MOCA after IK: {exc}") from exc
        if result.returncode != 0:
            seed_stop.set()
            seed_thread.join(timeout=1.0)
            reason = result.stderr.strip() or result.stdout.strip() or "unknown error"
            self._run_moca_stop_command()
            self._release_moca_lock()
            raise RuntimeError(f"Failed to start MOCA after IK: {reason}")
        self._moca_started_by_env = True
        self._moca_owned_by_env = True

        deadline = time.monotonic() + max(10.0, self.feedback_timeout_s)
        while time.monotonic() < deadline:
            if int(self._arm_pub.get_subscription_count()) > 0:
                # Seed MOCA immediately; it must never run with an empty right-arm target.
                for _ in range(10):
                    self._arm_pub.publish(msg)
                    time.sleep(0.02)
                self._last_action["lazy_moca_start"] = {
                    "target": target.tolist(),
                    "started_after_ik": True,
                }
                seed_stop.set()
                seed_thread.join(timeout=1.0)
                return
            self._spin_once_or_sleep()
        seed_stop.set()
        seed_thread.join(timeout=1.0)
        self._stop_lazy_moca()
        raise TimeoutError("MOCA started after IK but did not subscribe to the joint target")

    def _stop_lazy_moca(self) -> None:
        if not getattr(self, "_moca_owned_by_env", False):
            return
        try:
            self._run_moca_stop_command()
            self._wait_for_moca_to_stop()
        finally:
            self._moca_started_by_env = False
            self._moca_owned_by_env = False
            self._moca_startup_stabilized = False
            self._release_moca_lock()

    def _enter_motion_fault(
        self,
        *,
        reason: str,
        details: dict[str, Any],
    ) -> None:
        if getattr(self, "_motion_fault_latched", False):
            return
        record = {
            "timestamp": time.time(),
            "reason": str(reason),
            "details": dict(details),
        }
        self._motion_fault_latched = True
        self._motion_fault_record = record
        self._last_action["motion_fault"] = record
        self._pending_gripper_target = None
        try:
            self._stop_lazy_moca()
        except Exception as exc:
            record["moca_stop_error"] = repr(exc)

    def _feedback_ready_locked(self, kind: str) -> bool:
        if kind == "arm":
            return self._latest_arm is not None
        if kind == "gripper":
            return self._latest_gripper is not None
        if kind == "torso":
            return self._latest_torso is not None
        if kind == "rgb":
            return self._latest_wrist_rgb is not None
        if kind == "depth":
            return self._latest_wrist_depth is not None
        if kind == "camera_info":
            return self._latest_wrist_intrinsics is not None
        if kind == "head_left_rgb":
            return self._latest_head_left_rgb is not None
        if kind == "head_right_rgb":
            return self._latest_head_right_rgb is not None
        if kind == "head_left_rect_rgb":
            return self._latest_head_left_rect_rgb is not None
        if kind == "head_left_depth":
            return self._latest_head_left_depth is not None
        if kind == "head_left_camera_info":
            return self._latest_head_left_intrinsics is not None
        if kind == "relaxed_ik_solution":
            return self._latest_relaxed_ik_solution is not None
        if kind == "relaxed_ik_fk":
            return (
                self._latest_relaxed_ik_fk_position is not None
                and self._latest_relaxed_ik_fk_quaternion_xyzw is not None
            )
        if kind == "ee_pose":
            return (
                self._latest_ee_position is not None
                and self._latest_ee_quaternion_xyzw is not None
            )
        if kind == "relaxed_ik_ee_pose":
            return (
                self._latest_relaxed_ik_ee_position is not None
                and self._latest_relaxed_ik_ee_quaternion_xyzw is not None
            )
        raise ValueError(f"Unknown feedback kind: {kind}")

    def wait_until_ready(self) -> dict[str, Any]:
        """Require DDS endpoint matching and initial feedback before any task action."""
        unknown_feedback = set(self.startup_required_feedback) - set(
            self._feedback_subscriptions
        )
        unknown_commands = set(self.startup_required_command_subscribers) - set(
            self._command_publishers
        )
        if unknown_feedback or unknown_commands:
            raise ValueError(
                "Invalid R1 Lite startup requirements: "
                f"feedback={sorted(unknown_feedback)}, "
                f"commands={sorted(unknown_commands)}"
            )

        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            missing_publishers = [
                name
                for name in self.startup_required_feedback
                if int(
                    self._node.count_publishers(
                        self._feedback_subscriptions[name].topic_name
                    )
                ) == 0
            ]
            missing_subscribers = [
                name
                for name in self.startup_required_command_subscribers
                if int(
                    self._command_publishers[name].get_subscription_count()
                ) == 0
            ]
            with self._lock:
                missing_feedback = [
                    name
                    for name in self.startup_required_feedback
                    if not self._feedback_ready_locked(name)
                ]
            if (
                not missing_publishers
                and not missing_subscribers
                and not missing_feedback
            ):
                readiness = {
                    "feedback": list(self.startup_required_feedback),
                    "command_subscribers": list(
                        self.startup_required_command_subscribers
                    ),
                }
                self._last_action["startup_readiness"] = readiness
                return readiness
            self._spin_once_or_sleep()

        raise TimeoutError(
            "R1 Lite startup readiness failed: "
            f"feedback_publishers_missing={missing_publishers}, "
            f"feedback_samples_missing={missing_feedback}, "
            f"command_subscribers_missing={missing_subscribers}"
        )

    def _wait_for_feedback(self, kind: str) -> None:
        deadline = time.time() + self.feedback_timeout_s
        while time.time() < deadline:
            with self._lock:
                ready = self._feedback_ready_locked(kind)
            if ready:
                return
            self._spin_once_or_sleep()
        raise TimeoutError(f"Timed out waiting for R1 Lite {kind} feedback")

    def _spin_once_or_sleep(self) -> None:
        """Avoid spinning one node concurrently in two ROS executors."""
        spin_thread = getattr(self, "_spin_thread", None)
        if spin_thread is not None and spin_thread.is_alive():
            time.sleep(0.05)
        else:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)

    def get_arm_joints(self) -> np.ndarray:
        """Return the latest right-arm joint feedback as a 6-vector."""
        self._wait_for_feedback("arm")
        with self._lock:
            assert self._latest_arm is not None
            return self._latest_arm.copy()

    def get_torso_joints(self) -> np.ndarray:
        """Return the latest torso joint feedback as a 3-vector."""
        self._wait_for_feedback("torso")
        with self._lock:
            assert self._latest_torso is not None
            return self._latest_torso.copy()

    def wait_for_settled_arm_joints(
        self, *, timeout_s: float | None = None
    ) -> np.ndarray:
        """Wait for a fresh stable arm-feedback window without commanding motion."""
        with self._lock:
            after_sequence = self._arm_feedback_sequence
        return self._wait_for_settled_arm_feedback(
            after_sequence=after_sequence,
            not_before_monotonic=time.monotonic(),
            timeout_s=(
                self.adaptive_feedback_settle_timeout_s
                if timeout_s is None
                else float(timeout_s)
            ),
        )

    def get_gripper_position(self) -> float:
        """Return the latest right-gripper position feedback."""
        self._wait_for_feedback("gripper")
        with self._lock:
            assert self._latest_gripper is not None
            return float(self._latest_gripper)

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the vendor physical ``right_gripper_link`` pose in base_link.

        Prefer :meth:`get_physical_gripper_pose` or
        :meth:`get_physical_tool_z_pose` in new code so the TCP semantic is
        explicit.
        """
        self._wait_for_feedback("ee_pose")
        with self._lock:
            assert self._latest_ee_position is not None
            assert self._latest_ee_quaternion_xyzw is not None
            return (
                self._latest_ee_position.copy(),
                self._latest_ee_quaternion_xyzw.copy(),
            )

    @staticmethod
    def gripper_link_to_tool_z_quaternion(
        gripper_quaternion_xyzw: (
            list[float] | tuple[float, ...] | np.ndarray
        ),
    ) -> np.ndarray:
        """Convert a gripper-link orientation to RelaxedIK's tool-Z TCP."""
        gripper = np.asarray(gripper_quaternion_xyzw, dtype=np.float64).reshape(4)
        gripper /= np.linalg.norm(gripper)
        return (
            SciRotation.from_quat(gripper)
            * SciRotation.from_euler("y", np.pi / 2.0)
        ).as_quat()

    @staticmethod
    def tool_z_to_gripper_link_quaternion(
        tool_z_quaternion_xyzw: (
            list[float] | tuple[float, ...] | np.ndarray
        ),
    ) -> np.ndarray:
        """Convert RelaxedIK's tool-Z orientation to the vendor gripper link."""
        tool = np.asarray(tool_z_quaternion_xyzw, dtype=np.float64).reshape(4)
        tool /= np.linalg.norm(tool)
        return (
            SciRotation.from_quat(tool)
            * SciRotation.from_euler("y", -np.pi / 2.0)
        ).as_quat()

    def get_physical_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``T_base_right_gripper_link`` from the vendor feedback."""
        return self.get_ee_pose()

    def get_physical_tool_z_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return physical feedback expressed as ``T_base_right_gripper_tool_z``."""
        position, gripper_quaternion = self.get_physical_gripper_pose()
        return (
            position,
            self.gripper_link_to_tool_z_quaternion(gripper_quaternion),
        )

    def get_relaxed_ik_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the same measured pose expressed in RelaxedIK's URDF frame."""
        self._wait_for_feedback("relaxed_ik_ee_pose")
        with self._lock:
            assert self._latest_relaxed_ik_ee_position is not None
            assert self._latest_relaxed_ik_ee_quaternion_xyzw is not None
            return (
                self._latest_relaxed_ik_ee_position.copy(),
                self._latest_relaxed_ik_ee_quaternion_xyzw.copy(),
            )

    def _relaxed_ik_pose_to_base(
        self,
        position: np.ndarray,
        quaternion_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform a ``torso_link3`` RelaxedIK pose into ``base_link``."""
        base_from_ik, _ = self._base_from_torso_link3()
        ik_rotation = SciRotation.from_quat(quaternion_xyzw).as_matrix()
        base_position = (
            base_from_ik[:3, :3] @ np.asarray(position, dtype=np.float64).reshape(3)
            + base_from_ik[:3, 3]
        )
        base_rotation = base_from_ik[:3, :3] @ ik_rotation
        return base_position, SciRotation.from_matrix(base_rotation).as_quat()

    def get_planning_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return RelaxedIK model ``T_base_right_gripper_tool_z``.

        This is a planning/model pose, not the physical vendor EE topic.
        """
        position, quaternion = self.get_relaxed_ik_ee_pose()
        return self._relaxed_ik_pose_to_base(position, quaternion)

    def get_local_planning_fk_pose(
        self,
        arm_joints: np.ndarray | None = None,
        torso_joints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the planning TCP from joint feedback using local URDF FK."""
        if self._local_planning_kinematics is None:
            from capx.integrations.r1lite.kinematics import (
                R1LiteCandidateKinematics,
            )

            self._local_planning_kinematics = R1LiteCandidateKinematics(
                side="right"
            )
        arm = (
            self.get_arm_joints()
            if arm_joints is None
            else np.asarray(arm_joints, dtype=np.float64).reshape(6)
        )
        torso = (
            self.get_torso_joints()
            if torso_joints is None
            else np.asarray(torso_joints, dtype=np.float64).reshape(3)
        )
        result = self._local_planning_kinematics.fk(arm, torso)
        return (
            result.position_base_m.copy(),
            result.quaternion_xyzw.copy(),
        )

    def get_relaxed_ik_fk_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the latest solved RelaxedIK FK pose in ``base_link``."""
        with self._lock:
            if (
                self._latest_relaxed_ik_fk_position is None
                or self._latest_relaxed_ik_fk_quaternion_xyzw is None
            ):
                raise RuntimeError("RelaxedIK has not published solved-pose FK")
            position = self._latest_relaxed_ik_fk_position.copy()
            quaternion = self._latest_relaxed_ik_fk_quaternion_xyzw.copy()
        return self._relaxed_ik_pose_to_base(position, quaternion)

    def get_wrist_rgbd(
        self,
        *,
        require_depth: bool = True,
        newer_than_ns: int | None = None,
    ) -> dict[str, Any]:
        """Return the latest right-wrist RGB-D observation."""
        if not self.wrist_camera_enabled:
            raise RuntimeError("Wrist camera use is disabled for this task")
        self._wait_for_feedback("rgb")
        if require_depth:
            self._wait_for_feedback("depth")
            self._wait_for_feedback("camera_info")
        fresh_deadline = time.monotonic() + self.feedback_timeout_s
        while True:
            with self._lock:
                rgb = (
                    None
                    if self._latest_wrist_rgb is None
                    else self._latest_wrist_rgb.copy()
                )
                depth = (
                    None
                    if self._latest_wrist_depth is None
                    else self._latest_wrist_depth.copy()
                )
                intrinsics = (
                    None
                    if self._latest_wrist_intrinsics is None
                    else self._latest_wrist_intrinsics.copy()
                )
                rgb_stamp_ns = self._latest_wrist_rgb_stamp_ns
                depth_stamp_ns = self._latest_wrist_depth_stamp_ns
                rgb_monotonic = self._latest_wrist_rgb_monotonic
                depth_monotonic = self._latest_wrist_depth_monotonic
                barrier_monotonic = (
                    self._last_motion_barrier_monotonic
                    if (
                        newer_than_ns is not None
                        and self._last_motion_barrier_ns == int(newer_than_ns)
                    )
                    else None
                )
                info_size = self._latest_wrist_info_size
                frame_ids = (
                    self._latest_wrist_rgb_frame_id,
                    self._latest_wrist_depth_frame_id,
                    self._latest_wrist_info_frame_id,
                )
            if not require_depth:
                break
            now_ns = int(self._node.get_clock().now().nanoseconds)
            ages = {
                "rgb": (
                    float("inf")
                    if rgb_stamp_ns is None
                    else max(0.0, (now_ns - rgb_stamp_ns) / 1e9)
                ),
                "depth": (
                    float("inf")
                    if depth_stamp_ns is None
                    else max(0.0, (now_ns - depth_stamp_ns) / 1e9)
                ),
            }
            stale = {name: age for name, age in ages.items() if age > self.wrist_rgbd_max_age_s}
            ros_before_barrier = (
                newer_than_ns is not None
                and (
                    rgb_stamp_ns is None
                    or depth_stamp_ns is None
                    or min(rgb_stamp_ns, depth_stamp_ns) <= int(newer_than_ns)
                )
            )
            receive_before_barrier = (
                barrier_monotonic is not None
                and (
                    rgb_monotonic is None
                    or depth_monotonic is None
                    or min(rgb_monotonic, depth_monotonic) <= barrier_monotonic
                )
            )
            # Camera image stamps can lag the local ROS clock used for the
            # motion barrier. If callback receive times show both streams
            # arrived after the barrier, treat the RGB-D pair as fresh.
            before_barrier = (
                bool(ros_before_barrier)
                if barrier_monotonic is None
                else bool(ros_before_barrier and receive_before_barrier)
            )
            if not stale and not before_barrier:
                break
            if time.monotonic() >= fresh_deadline:
                if before_barrier:
                    raise RuntimeError(
                        "Timed out waiting for wrist RGB-D captured after motion "
                        f"barrier {int(newer_than_ns)} "
                        f"(rgb_stamp={rgb_stamp_ns}, depth_stamp={depth_stamp_ns}, "
                        f"rgb_rx={rgb_monotonic}, depth_rx={depth_monotonic}, "
                        f"barrier_rx={barrier_monotonic})"
                    )
                raise RuntimeError(
                    f"Wrist RGB-D is stale (limit={self.wrist_rgbd_max_age_s:.3f}s): {stale}"
                )
            self._spin_once_or_sleep()
        if require_depth:
            stamps = [rgb_stamp_ns, depth_stamp_ns]
            if any(stamp is None or stamp <= 0 for stamp in stamps):
                raise RuntimeError("Wrist RGB/depth messages have missing timestamps")
            skew_s = (max(stamps) - min(stamps)) / 1e9
            if skew_s > self.wrist_rgbd_max_skew_s:
                raise RuntimeError(
                    f"Wrist RGB-D timestamp skew {skew_s:.3f}s exceeds "
                    f"{self.wrist_rgbd_max_skew_s:.3f}s"
                )
            nonempty_frames = {frame for frame in frame_ids if frame}
            if not nonempty_frames:
                raise RuntimeError("Wrist RGB-D messages have no camera frame_id")
            if len(nonempty_frames) != 1:
                raise RuntimeError(f"Wrist RGB-D frame mismatch: {frame_ids}")
            if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
                raise RuntimeError(
                    "Wrist RGB/depth resolution mismatch: "
                    f"rgb={None if rgb is None else rgb.shape[:2]}, "
                    f"depth={None if depth is None else depth.shape}"
                )
            if info_size is None or info_size != depth.shape:
                raise RuntimeError(
                    "Wrist CameraInfo resolution does not match aligned depth: "
                    f"camera_info={info_size}, depth={depth.shape}"
                )
            frame_id = next(iter(nonempty_frames))
            stamp_ns = int(depth_stamp_ns)
        else:
            frame_id = frame_ids[0]
            stamp_ns = rgb_stamp_ns
        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
        }

    def get_head_rgbd(
        self,
        *,
        camera: str = "left",
        require_depth: bool = True,
        newer_than_ns: int | None = None,
    ) -> dict[str, Any]:
        """Return the latest rectified head stereo RGB-D observation."""
        if camera != "left":
            raise ValueError("Only the left head stereo camera is RGB-D aligned")
        self._wait_for_feedback("head_left_rect_rgb")
        if require_depth:
            self._wait_for_feedback("head_left_depth")
            self._wait_for_feedback("head_left_camera_info")
        fresh_deadline = time.monotonic() + self.feedback_timeout_s
        while True:
            with self._lock:
                rgb = (
                    None
                    if self._latest_head_left_rect_rgb is None
                    else self._latest_head_left_rect_rgb.copy()
                )
                depth = (
                    None
                    if self._latest_head_left_depth is None
                    else self._latest_head_left_depth.copy()
                )
                intrinsics = (
                    None
                    if self._latest_head_left_intrinsics is None
                    else self._latest_head_left_intrinsics.copy()
                )
                rgb_stamp_ns = self._latest_head_left_rect_rgb_stamp_ns
                depth_stamp_ns = self._latest_head_left_depth_stamp_ns
                info_size = self._latest_head_left_info_size
                frame_ids = (
                    self._latest_head_left_rect_rgb_frame_id,
                    self._latest_head_left_depth_frame_id,
                    self._latest_head_left_info_frame_id,
                )
                monotonic_stamps = {
                    "rgb": self._latest_head_left_rect_rgb_monotonic,
                    "depth": self._latest_head_left_depth_monotonic,
                }
            if not require_depth:
                break
            now = time.monotonic()
            ages = {
                name: (
                    float("inf")
                    if stamp is None
                    else max(0.0, now - float(stamp))
                )
                for name, stamp in monotonic_stamps.items()
            }
            stale = {
                name: age
                for name, age in ages.items()
                if age > self.head_rgbd_max_age_s
            }
            before_barrier = (
                newer_than_ns is not None
                and (
                    rgb_stamp_ns is None
                    or depth_stamp_ns is None
                    or min(rgb_stamp_ns, depth_stamp_ns) <= int(newer_than_ns)
                )
            )
            skew_s = (
                float("inf")
                if rgb_stamp_ns is None or depth_stamp_ns is None
                else (max(rgb_stamp_ns, depth_stamp_ns) - min(rgb_stamp_ns, depth_stamp_ns))
                / 1e9
            )
            skew_too_high = skew_s > self.head_rgbd_max_skew_s
            if not stale and not before_barrier and not skew_too_high:
                break
            if time.monotonic() >= fresh_deadline:
                if before_barrier:
                    raise RuntimeError(
                        "Timed out waiting for head RGB-D captured after motion "
                        f"barrier {int(newer_than_ns)}"
                    )
                if skew_too_high:
                    raise RuntimeError(
                        f"Head RGB-D timestamp skew {skew_s:.3f}s exceeds "
                        f"{self.head_rgbd_max_skew_s:.3f}s"
                    )
                raise RuntimeError(
                    f"Head RGB-D is stale (limit={self.head_rgbd_max_age_s:.3f}s): {stale}"
                )
            self._spin_once_or_sleep()
        if require_depth:
            stamps = [rgb_stamp_ns, depth_stamp_ns]
            if any(stamp is None or stamp <= 0 for stamp in stamps):
                raise RuntimeError("Head RGB/depth messages have missing timestamps")
            skew_s = (max(stamps) - min(stamps)) / 1e9
            if skew_s > self.head_rgbd_max_skew_s:
                raise RuntimeError(
                    f"Head RGB-D timestamp skew {skew_s:.3f}s exceeds "
                    f"{self.head_rgbd_max_skew_s:.3f}s"
                )
            nonempty_frames = {frame for frame in frame_ids if frame}
            if not nonempty_frames:
                raise RuntimeError("Head RGB-D messages have no camera frame_id")
            if len(nonempty_frames) != 1:
                raise RuntimeError(f"Head RGB-D frame mismatch: {frame_ids}")
            if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
                raise RuntimeError(
                    "Head RGB/depth resolution mismatch: "
                    f"rgb={None if rgb is None else rgb.shape[:2]}, "
                    f"depth={None if depth is None else depth.shape}"
                )
            if info_size is None or info_size != depth.shape:
                raise RuntimeError(
                    "Head CameraInfo resolution does not match aligned depth: "
                    f"camera_info={info_size}, depth={depth.shape}"
                )
            frame_id = next(iter(nonempty_frames))
            stamp_ns = int(depth_stamp_ns)
        else:
            frame_id = frame_ids[0]
            stamp_ns = rgb_stamp_ns
        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
        }

    def get_head_rgbd_median(
        self,
        *,
        camera: str = "left",
        sample_count: int | None = None,
        minimum_valid_samples: int | None = None,
    ) -> dict[str, Any]:
        """Return one RGB frame with a temporal median of fresh stereo depths."""
        count = (
            self.head_rgbd_median_sample_count
            if sample_count is None
            else int(sample_count)
        )
        if not 15 <= count <= 30:
            raise ValueError("head RGB-D median sample_count must be in [15, 30]")
        required = (
            max(3, count // 3)
            if minimum_valid_samples is None
            else int(minimum_valid_samples)
        )
        if required <= 0 or required > count:
            raise ValueError("invalid minimum_valid_samples")

        observations: list[dict[str, Any]] = []
        barrier: int | None = None
        for _ in range(count):
            observation = self.get_head_rgbd(
                camera=camera,
                require_depth=True,
                newer_than_ns=barrier,
            )
            observations.append(observation)
            barrier = int(observation["stamp_ns"])
        depth_stack = np.stack(
            [np.asarray(item["depth"], dtype=np.float64) for item in observations]
        )
        valid = np.isfinite(depth_stack) & (depth_stack > 0.0)
        valid_counts = np.count_nonzero(valid, axis=0)
        masked_depth = np.ma.masked_where(~valid, depth_stack)
        depth = np.ma.median(masked_depth, axis=0).filled(np.nan)
        depth[valid_counts < required] = np.nan
        latest = dict(observations[-1])
        latest["depth"] = depth
        latest["depth_sample_count"] = count
        latest["minimum_valid_depth_samples"] = required
        latest["depth_stamp_range_ns"] = [
            int(observations[0]["stamp_ns"]),
            int(observations[-1]["stamp_ns"]),
        ]
        return latest

    def wait_for_arm_stable(
        self,
        *,
        settle_s: float | None = None,
        tolerance_rad: float = 0.008,
        timeout_s: float | None = None,
    ) -> int:
        """Wait for measured joints to remain stable and return a ROS-time barrier."""
        required = (
            self.wrist_observation_settle_s
            if settle_s is None
            else max(0.1, float(settle_s))
        )
        timeout = self.feedback_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + max(required, timeout)
        stable_since: float | None = None
        previous = self.get_arm_joints()
        while time.monotonic() < deadline:
            self._spin_once_or_sleep()
            current = self.get_arm_joints()
            delta = float(np.max(np.abs(current - previous)))
            now = time.monotonic()
            if delta <= float(tolerance_rad):
                stable_since = now if stable_since is None else stable_since
                if now - stable_since >= required:
                    barrier_ns = int(self._node.get_clock().now().nanoseconds)
                    with self._lock:
                        self._last_motion_barrier_ns = barrier_ns
                        self._last_motion_barrier_monotonic = time.monotonic()
                    return barrier_ns
            else:
                stable_since = None
            previous = current
        raise TimeoutError(
            "Timed out waiting for right arm to settle: "
            f"tolerance={float(tolerance_rad):.4f}rad, settle={required:.2f}s"
        )

    def lookup_camera_to_base(
        self,
        *,
        frame_id: str,
        stamp_ns: int,
        timeout_s: float | None = None,
        camera_label: str = "camera",
    ) -> np.ndarray:
        """Return ``base_link <- camera`` at the RGB-D sample time."""
        source_frame = str(frame_id).strip()
        if not source_frame:
            raise ValueError(f"{camera_label} frame_id must be non-empty")
        if int(stamp_ns) <= 0:
            raise ValueError(f"{camera_label} timestamp must be positive")
        timeout = self.wrist_tf_timeout_s if timeout_s is None else float(timeout_s)
        query_time = self._Time(nanoseconds=int(stamp_ns))
        duration = self._Duration(seconds=max(0.1, timeout))
        try:
            transform = self._tf_buffer.lookup_transform(
                self.wrist_tf_base_frame,
                source_frame,
                query_time,
                timeout=duration,
            )
        except Exception as exc:
            if source_frame in {"camera_head_left", "camera_head_left_link"}:
                try:
                    return self._head_left_camera_to_base_from_calibration()
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"TF lookup failed for {self.wrist_tf_base_frame} <- "
                        f"{source_frame} at {stamp_ns}: {exc}; "
                        f"head calibration fallback failed: {fallback_exc}"
                    ) from fallback_exc
            if "extrapolation" not in str(exc).lower():
                raise RuntimeError(
                    f"TF lookup failed for {self.wrist_tf_base_frame} <- "
                    f"{source_frame} at {stamp_ns}: {exc}"
                ) from exc
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.wrist_tf_base_frame,
                    source_frame,
                    self._Time(),
                    timeout=duration,
                )
                tf_stamp = transform.header.stamp
                tf_stamp_ns = int(tf_stamp.sec) * 1_000_000_000 + int(
                    tf_stamp.nanosec
                )
                skew_s = abs(int(stamp_ns) - tf_stamp_ns) / 1e9
                max_tf_skew = (
                    self.head_tf_max_skew_s
                    if "head" in str(camera_label).lower()
                    else self.wrist_rgbd_max_age_s
                )
                if skew_s > max_tf_skew:
                    raise RuntimeError(
                        f"latest {camera_label} TF is {skew_s:.3f}s from RGB-D sample"
                    )
            except Exception as latest_exc:
                raise RuntimeError(
                    f"TF lookup failed for {self.wrist_tf_base_frame} <- "
                    f"{source_frame} at {stamp_ns}: {exc}; "
                    f"latest-transform fallback failed: {latest_exc}"
                ) from latest_exc
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        quaternion_xyzw = np.array(
            [rotation.x, rotation.y, rotation.z, rotation.w],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion_xyzw))
        if not np.isfinite(quaternion_xyzw).all() or norm < 1e-9:
            raise RuntimeError(f"{camera_label} TF contains an invalid quaternion")
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = SciRotation.from_quat(quaternion_xyzw / norm).as_matrix()
        matrix[:3, 3] = [translation.x, translation.y, translation.z]
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"{camera_label} TF contains non-finite values")
        return matrix

    @staticmethod
    def _rigid_transform(
        xyz: tuple[float, float, float],
        rpy: tuple[float, float, float],
    ) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = SciRotation.from_euler("xyz", rpy).as_matrix()
        matrix[:3, 3] = xyz
        return matrix

    def _head_left_camera_to_base_from_urdf(self) -> np.ndarray:
        """Compute ``base_link <- camera_head_left`` from torso feedback and URDF."""
        base_from_torso, torso_joints = self._base_from_torso_link3()
        base_from_camera = base_from_torso @ self._rigid_transform(
                (0.054947, 0.03149, 0.63214),
                (-2.0952, 4.1972e-05, -1.5709),
            )
        self._last_action["head_camera_transform"] = {
            "source": "r1lite_2026.urdf_torso_fk",
            "torso_joints": torso_joints.tolist(),
            "base_from_camera": base_from_camera.tolist(),
        }
        return base_from_camera

    def _base_from_torso_link3(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the URDF transform and joints for ``base_link <- torso_link3``."""
        self._wait_for_feedback("torso")
        with self._lock:
            assert self._latest_torso is not None
            torso_joints = self._latest_torso.copy()
        q1, q2, q3 = torso_joints.tolist()

        def rotate_y(angle: float) -> np.ndarray:
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = SciRotation.from_rotvec(
                np.array([0.0, angle, 0.0], dtype=np.float64)
            ).as_matrix()
            return matrix

        transform = (
            self._rigid_transform((0.091831, 0.00036033, 0.14865), (0.0, 0.0, 0.0))
            @ rotate_y(q1)
            @ self._rigid_transform((0.0, -0.00185, 0.35), (0.0, 0.0, 0.0))
            @ rotate_y(q2)
            @ self._rigid_transform((9.6485e-05, 0.0, 0.35), (0.0, 0.0, 0.0))
            @ rotate_y(-q3)
        )
        return transform, torso_joints

    def _head_left_camera_to_base_from_calibration(self) -> np.ndarray:
        """Load ``base_link <- rectified head-left optical`` calibration."""
        self._wait_for_feedback("head_left_camera_info")
        calibration = self.load_head_stereo_rectified_calibration()
        with self._lock:
            if self._latest_head_left_intrinsics is None:
                raise RuntimeError("head-left camera intrinsics are unavailable")
            if self._latest_head_left_info_size is None:
                raise RuntimeError("head-left CameraInfo size is unavailable")
            intrinsics = self._latest_head_left_intrinsics.copy()
            current_height, current_width = self._latest_head_left_info_size

        expected_size = tuple(int(value) for value in calibration["image_size"])
        if expected_size != (current_width, current_height):
            raise RuntimeError(
                "rectified stereo calibration/image size mismatch: "
                f"calibration={expected_size}, current="
                f"{(current_width, current_height)}"
            )
        expected_intrinsics = np.asarray(
            calibration["rectified_intrinsics"], dtype=np.float64
        ).reshape(3, 3)
        if not np.allclose(intrinsics, expected_intrinsics, rtol=0.0, atol=1e-3):
            raise RuntimeError(
                "rectified stereo calibration/intrinsics mismatch; "
                "regenerate the rectified calibration"
            )
        base_from_camera = np.asarray(
            calibration["base_from_rectified_left_camera"],
            dtype=np.float64,
        ).reshape(4, 4)
        rotation = base_from_camera[:3, :3]
        if (
            not np.isfinite(base_from_camera).all()
            or not np.allclose(
                base_from_camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8
            )
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        ):
            raise RuntimeError("invalid rectified head-camera calibration pose")
        self._last_action["head_camera_transform"] = {
            "source": "head_stereo_rectified_plane_calibration",
            "calibration_path": self.head_stereo_rectified_calibration_path,
            "reprojection_rmse_px": calibration.get("reprojection_rmse_px"),
            "base_from_camera": base_from_camera.tolist(),
        }
        return base_from_camera.copy()

    def lookup_wrist_camera_to_base(
        self,
        *,
        frame_id: str,
        stamp_ns: int,
        timeout_s: float | None = None,
    ) -> np.ndarray:
        """Return ``base_link <- wrist camera`` at the RGB-D sample time."""
        return self.lookup_camera_to_base(
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            timeout_s=timeout_s,
            camera_label="wrist camera",
        )

    def get_wrist_rgb_age_s(self) -> float:
        """Return monotonic age of the latest wrist RGB frame."""
        self._wait_for_feedback("rgb")
        with self._lock:
            stamp = self._latest_wrist_rgb_monotonic
        if stamp is None:
            raise RuntimeError("No timestamp is available for wrist RGB")
        return max(0.0, time.monotonic() - stamp)

    def get_head_rgb(self, camera: str = "left") -> np.ndarray:
        """Return latest head camera RGB image.

        Args:
            camera: ``"left"`` or ``"right"``.
        """
        if camera not in {"left", "right"}:
            raise ValueError("camera must be 'left' or 'right'")
        kind = "head_left_rgb" if camera == "left" else "head_right_rgb"
        self._wait_for_feedback(kind)
        with self._lock:
            img = (
                self._latest_head_left_rgb
                if camera == "left"
                else self._latest_head_right_rgb
            )
            assert img is not None
            return img.copy()

    def load_head_plane_calibration(self) -> dict[str, Any]:
        """Load image-to-base-plane homography calibration from JSON."""
        if self._head_plane_calibration is not None:
            return self._head_plane_calibration
        path = Path(self.head_plane_calibration_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Head-plane calibration not found: {path}. "
                "Run scripts/r1lite_calibrate_head_plane.py first."
            )
        data = json.loads(path.read_text())
        h = np.asarray(data["image_to_base_xy_homography"], dtype=np.float64).reshape(3, 3)
        data["_homography_np"] = h
        self._head_plane_calibration = data
        return data

    def load_head_stereo_rectified_calibration(self) -> dict[str, Any]:
        """Load the calibration belonging to rectified head stereo RGB-D."""
        if self._head_stereo_rectified_calibration is not None:
            return self._head_stereo_rectified_calibration
        path = Path(self.head_stereo_rectified_calibration_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Rectified head-stereo calibration not found: {path}. "
                "Run scripts/r1lite_calibrate_head_stereo_rectified.py first."
            )
        data = json.loads(path.read_text())
        if data.get("image_domain") != "head_left_stereo_rectified":
            raise ValueError(
                f"Invalid rectified head-stereo calibration domain in {path}"
            )
        self._head_stereo_rectified_calibration = data
        return data

    def load_wrist_hand_eye_calibration(self) -> np.ndarray:
        """Load and validate ``end_effector <- wrist optical camera``."""
        if self._wrist_hand_eye_calibration is not None:
            return self._wrist_hand_eye_calibration.copy()
        path = Path(self.wrist_hand_eye_calibration_path)
        if not path.exists():
            raise FileNotFoundError(f"Wrist hand-eye calibration not found: {path}")
        import yaml

        payload = yaml.safe_load(path.read_text())
        transform = np.asarray(payload["T_ee_camera"], dtype=np.float64).reshape(4, 4)
        rotation = transform[:3, :3]
        if (
            not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        ):
            raise ValueError(f"Invalid rigid T_ee_camera in {path}")
        self._wrist_hand_eye_calibration = transform
        return transform.copy()

    def pixel_to_base_on_plane(
        self,
        pixel: list[float] | tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """Map a head-left image pixel to ``[x, y, z]`` on calibrated plane."""
        if self.r1lite_use_current_head_rgbd_table_plane:
            return self.pixel_to_base_on_current_head_rgbd_table_plane(pixel)
        calib = self.load_head_plane_calibration()
        h = calib["_homography_np"]
        u, v = np.asarray(pixel, dtype=np.float64).reshape(2)
        q = h @ np.array([u, v, 1.0], dtype=np.float64)
        if abs(q[2]) < 1e-12:
            raise ValueError("Degenerate homography projection")
        x, y = (q[:2] / q[2]).tolist()
        z = float(calib.get("plane_z", 0.0))
        point = np.array([x, y, z], dtype=np.float64)
        self.validate_workspace_point(point)
        return point

    def _fit_current_head_rgbd_table_plane(
        self,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
        """Fit the current tabletop from head stereo depth in ``base_link``."""
        obs = self.get_head_rgbd(camera="left", require_depth=True)
        depth = np.asarray(obs["depth"], dtype=np.float64)
        intrinsics = np.asarray(obs["intrinsics"], dtype=np.float64).reshape(3, 3)
        frame_id = str(obs["frame_id"])
        stamp_ns = int(obs["stamp_ns"])
        base_from_camera = self.lookup_camera_to_base(
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            camera_label="head-left",
        )
        height, width = depth.shape
        step = max(4, int(max(height, width) // 160))
        ys, xs = np.mgrid[0:height:step, 0:width:step]
        z = depth[ys, xs]
        valid = np.isfinite(z) & (z > 0.20) & (z < 3.0)
        xs = xs[valid].astype(np.float64)
        ys = ys[valid].astype(np.float64)
        z = z[valid]
        x = (xs - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys - intrinsics[1, 2]) * z / intrinsics[1, 1]
        camera_points = np.stack([x, y, z, np.ones_like(z)], axis=0)
        base_points = (base_from_camera @ camera_points)[:3].T
        candidates = base_points[
            (base_points[:, 0] >= self.workspace_x_limits[0])
            & (base_points[:, 0] <= self.workspace_x_limits[1])
            & (base_points[:, 1] >= self.workspace_y_limits[0])
            & (base_points[:, 1] <= self.workspace_y_limits[1])
            & (base_points[:, 2] >= 0.55)
            & (base_points[:, 2] <= 1.20)
        ]
        if len(candidates) < 200:
            raise RuntimeError(
                "not enough current head RGB-D points to fit tabletop plane: "
                f"{len(candidates)}"
            )
        rng = np.random.default_rng(42)
        best: tuple[int, np.ndarray, float, np.ndarray] | None = None
        threshold_m = 0.012
        for _ in range(800):
            sample = candidates[rng.choice(len(candidates), size=3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                continue
            normal = normal / norm
            if abs(float(normal[2])) < 0.75:
                continue
            d = -float(np.dot(normal, sample[0]))
            distances = np.abs(candidates @ normal + d)
            inliers = distances < threshold_m
            score = int(np.count_nonzero(inliers))
            if best is None or score > best[0]:
                best = (score, normal, d, inliers)
        if best is None:
            raise RuntimeError("failed to fit a near-horizontal current head RGB-D table plane")
        _, _normal, _d, inliers = best
        inlier_points = candidates[inliers]
        centroid = inlier_points.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        rmse_m = float(np.sqrt(np.mean((inlier_points @ normal + d) ** 2)))
        details = {
            "source": "current_head_rgbd_ransac_plane_fit",
            "frame_id": frame_id,
            "stamp_ns": stamp_ns,
            "normal_base": normal.tolist(),
            "d": d,
            "inlier_count": int(len(inlier_points)),
            "candidate_count": int(len(candidates)),
            "rmse_m": rmse_m,
        }
        return normal, d, base_from_camera, details

    def pixel_to_base_on_current_head_rgbd_table_plane(
        self,
        pixel: list[float] | tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """Project a head pixel onto the current head RGB-D fitted tabletop."""
        normal, d, base_from_camera, details = self._fit_current_head_rgbd_table_plane()
        obs = self.get_head_rgbd(camera="left", require_depth=True)
        intrinsics = np.asarray(obs["intrinsics"], dtype=np.float64).reshape(3, 3)
        u, v = np.asarray(pixel, dtype=np.float64).reshape(2)
        ray_camera = np.array(
            [(u - intrinsics[0, 2]) / intrinsics[0, 0], (v - intrinsics[1, 2]) / intrinsics[1, 1], 1.0],
            dtype=np.float64,
        )
        origin_base = base_from_camera[:3, 3]
        ray_base = base_from_camera[:3, :3] @ ray_camera
        denom = float(np.dot(normal, ray_base))
        if abs(denom) < 1e-8:
            raise ValueError("head pixel ray is parallel to current RGB-D table plane")
        scale = -float(np.dot(normal, origin_base) + d) / denom
        if scale <= 0:
            raise ValueError("head pixel ray intersects current RGB-D table plane behind camera")
        point = origin_base + scale * ray_base
        self.validate_workspace_point(point)
        self._last_action["current_head_rgbd_table_plane"] = {
            **details,
            "pixel": [float(u), float(v)],
            "projected_point_base": point.tolist(),
        }
        return point

    def validate_workspace_point(self, point_xyz: np.ndarray) -> None:
        """Reject points outside configured conservative workspace bounds."""
        if not self.r1lite_motion_validation_enabled:
            return
        x, y, z = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        if not (self.workspace_x_limits[0] <= x <= self.workspace_x_limits[1]):
            raise ValueError(f"x={x:.3f} outside workspace {self.workspace_x_limits}")
        if not (self.workspace_y_limits[0] <= y <= self.workspace_y_limits[1]):
            raise ValueError(f"y={y:.3f} outside workspace {self.workspace_y_limits}")
        if not (self.workspace_z_limits[0] <= z <= self.workspace_z_limits[1]):
            raise ValueError(f"z={z:.3f} outside workspace {self.workspace_z_limits}")

    def validate_arm_joint_feedback_target(
        self,
        joints: list[float] | tuple[float, ...] | np.ndarray,
        *,
        context: str,
    ) -> np.ndarray:
        """Reject the measured-joint region that triggered the MOCA J2 failure."""
        target = np.asarray(joints, dtype=np.float64).reshape(6)
        if not np.isfinite(target).all():
            raise ValueError(f"{context}: arm joint target must be finite")
        if not bool(getattr(self, "r1lite_motion_validation_enabled", True)):
            return target
        limit = float(
            getattr(self, "right_arm_j2_feedback_upper_limit_rad", 3.05)
        )
        if float(target[1]) > limit + 1e-9:
            raise ValueError(
                f"{context}: right-arm J2 feedback target {target[1]:.4f} rad "
                f"exceeds configured limit {limit:.4f} rad"
            )
        return target

    def validate_moca_joint_command(
        self,
        command: list[float] | tuple[float, ...] | np.ndarray,
        *,
        context: str,
    ) -> np.ndarray:
        """Reject command-space J2 targets near the reproduced OTG instability."""
        target = np.asarray(command, dtype=np.float64).reshape(6).copy()
        if not np.isfinite(target).all():
            raise ValueError(f"{context}: MOCA joint command must be finite")
        lower_limit = float(
            getattr(
                self,
                "right_arm_j2_command_lower_limit_rad",
                0.01745329252,
            )
        )
        target[1] = max(float(target[1]), lower_limit)
        if not bool(getattr(self, "r1lite_motion_validation_enabled", True)):
            return target
        upper_limit = float(
            getattr(self, "right_arm_j2_command_upper_limit_rad", 3.05)
        )
        if float(target[1]) > upper_limit + 1e-9:
            raise ValueError(
                f"{context}: right-arm J2 MOCA command {target[1]:.4f} rad "
                f"exceeds configured limit {upper_limit:.4f} rad"
            )
        return target

    def get_relaxed_ik_candidate(
        self,
        position_xyz: list[float] | tuple[float, ...] | np.ndarray,
        quaternion_xyzw: list[float] | tuple[float, ...] | np.ndarray,
        *,
        timeout_s: float = 2.0,
        reference_joints: np.ndarray | None = None,
        start_joints: np.ndarray | None = None,
        max_incremental_joint_delta_rad: float | None = None,
        max_total_joint_delta_rad: float | None = None,
        max_total_joint_delta_overrun_rad: float = 0.0,
        republish_interval_s: float | None = 0.05,
        validate_joint_target: bool = True,
    ) -> np.ndarray:
        """Request a RelaxedIK joint candidate without executing it.

        This is disabled by default because RelaxedIK shares the same target-pose
        topic as the live controller on the current R1 Lite graph.
        """
        if not self.relaxed_ik_probe_enabled:
            raise RuntimeError(
                "RelaxedIK probe is disabled because /motion_target/target_pose_arm_right "
                "can also trigger robot motion. Set relaxed_ik_probe_enabled only after "
                "confirming this is safe on the current controller graph."
            )
        subscriptions = self._node.get_subscriptions_info_by_topic(
            self._relaxed_ik_pose_pub.topic_name
        )
        if not subscriptions:
            raise RuntimeError(
                "RelaxedIK probe topic has no subscriber: "
                f"{self._relaxed_ik_pose_pub.topic_name}"
            )
        if (
            not self._relaxed_ik_initialized
            and self.r1lite_motion_validation_enabled
        ):
            self._reset_relaxed_ik_to_current()
        elif not self._relaxed_ik_initialized:
            self._relaxed_ik_initialized = True
        pos = np.asarray(position_xyz, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
        self.validate_workspace_point(pos)
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1e-8 or not np.isfinite(quat_norm):
            raise ValueError("quaternion_xyzw must be finite and non-zero")
        quat = quat / quat_norm
        command_pos, command_quat = self._pose_in_relaxed_ik_frame(pos, quat)
        msg = self._PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "relaxed_ik_right"
        msg.pose.position.x = float(command_pos[0])
        msg.pose.position.y = float(command_pos[1])
        msg.pose.position.z = float(command_pos[2])
        msg.pose.orientation.x = float(command_quat[0])
        msg.pose.orientation.y = float(command_quat[1])
        msg.pose.orientation.z = float(command_quat[2])
        msg.pose.orientation.w = float(command_quat[3])
        with self._lock:
            before = self._latest_relaxed_ik_stamp
            before_fk = self._latest_relaxed_ik_fk_monotonic
            before_target_echo = self._latest_relaxed_ik_target_monotonic
        self._relaxed_ik_pose_pub.publish(msg)
        last_publish = time.monotonic()
        target_acknowledged = False
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._spin_once_or_sleep()
            if (
                not target_acknowledged
                and republish_interval_s is not None
                and time.monotonic() - last_publish
                >= float(republish_interval_s)
            ):
                self._relaxed_ik_pose_pub.publish(msg)
                last_publish = time.monotonic()
            with self._lock:
                target_position = self._latest_relaxed_ik_target_position
                target_quaternion = (
                    self._latest_relaxed_ik_target_quaternion_xyzw
                )
                target_stamp_ns = self._latest_relaxed_ik_target_stamp_ns
                target_echo_monotonic = (
                    self._latest_relaxed_ik_target_monotonic
                )
                solution_monotonic = (
                    self._latest_relaxed_ik_solution_monotonic
                )
                fk_monotonic = self._latest_relaxed_ik_fk_monotonic
                solution_stamp_ns = (
                    self._latest_relaxed_ik_solution_stamp_ns
                )
                fk_stamp_ns = self._latest_relaxed_ik_fk_stamp_ns
                target_echo_is_current = (
                    target_position is not None
                    and target_quaternion is not None
                    and self._latest_relaxed_ik_target_monotonic
                    > before_target_echo
                    and float(np.linalg.norm(target_position - command_pos))
                    <= 1e-7
                    and abs(
                        float(
                            np.dot(
                                target_quaternion
                                / np.linalg.norm(target_quaternion),
                                command_quat
                                / np.linalg.norm(command_quat),
                            )
                        )
                    )
                    >= 1.0 - 1e-7
                )
                target_acknowledged = bool(target_echo_is_current)
                stamped_response_matches_target = (
                    target_echo_is_current
                    and target_stamp_ns > 0
                    and solution_stamp_ns > 0
                    and solution_stamp_ns >= target_stamp_ns
                    and fk_stamp_ns >= target_stamp_ns
                    and abs(solution_stamp_ns - fk_stamp_ns)
                    <= 10_000_000
                )
                unstamped_solution_matches_target = (
                    target_echo_is_current
                    and target_stamp_ns > 0
                    and solution_stamp_ns == 0
                    and fk_stamp_ns >= target_stamp_ns
                    and (
                        -0.01
                        <= solution_monotonic - target_echo_monotonic
                        <= 0.25
                    )
                )
                response_matches_target = (
                    stamped_response_matches_target
                    or unstamped_solution_matches_target
                )
                if (
                    self._latest_relaxed_ik_solution is not None
                    and self._latest_relaxed_ik_stamp > before
                    and (
                        not self.r1lite_motion_validation_enabled
                        or (
                            self._latest_relaxed_ik_fk_monotonic > before_fk
                            and response_matches_target
                        )
                    )
                ):
                    candidate = self._latest_relaxed_ik_solution.copy()
                    if validate_joint_target:
                        self.validate_arm_joint_feedback_target(
                            candidate,
                            context="RelaxedIK candidate",
                        )
                    if reference_joints is not None:
                        incremental_delta = float(
                            np.max(
                                np.abs(
                                    candidate
                                    - np.asarray(
                                        reference_joints,
                                        dtype=np.float64,
                                    ).reshape(6)
                                )
                            )
                        )
                        if (
                            max_incremental_joint_delta_rad is not None
                            and incremental_delta
                            > float(max_incremental_joint_delta_rad)
                        ):
                            raise RuntimeError(
                                "Constrained RelaxedIK candidate exceeds "
                                "incremental joint limit: "
                                f"limit={max_incremental_joint_delta_rad:.3f}, "
                                f"actual={incremental_delta:.3f}"
                            )
                    if (
                        start_joints is not None
                        and max_total_joint_delta_rad is not None
                    ):
                        total_delta = float(
                            np.max(
                                np.abs(
                                    candidate
                                    - np.asarray(
                                        start_joints,
                                        dtype=np.float64,
                                    ).reshape(6)
                                )
                            )
                        )
                        total_overrun = total_delta - float(max_total_joint_delta_rad)
                        if (
                            total_overrun >= float(max_total_joint_delta_overrun_rad)
                        ):
                            raise RuntimeError(
                                "Constrained RelaxedIK candidate exceeds total "
                                "joint budget: "
                                f"limit={max_total_joint_delta_rad:.3f}, "
                                "allowed_overrun"
                                f"<{max_total_joint_delta_overrun_rad:.3f}, "
                                f"actual={total_delta:.3f}"
                            )
                    return candidate
        with self._lock:
            timeout_details = {
                "requested_position_relaxed_ik": command_pos.tolist(),
                "latest_target_position_relaxed_ik": (
                    None
                    if self._latest_relaxed_ik_target_position is None
                    else self._latest_relaxed_ik_target_position.tolist()
                ),
                "target_stamp_ns": self._latest_relaxed_ik_target_stamp_ns,
                "fk_stamp_ns": self._latest_relaxed_ik_fk_stamp_ns,
                "solution_stamp_ns": (
                    self._latest_relaxed_ik_solution_stamp_ns
                ),
                "solution_fk_stamp_delta_ms": abs(
                    self._latest_relaxed_ik_solution_stamp_ns
                    - self._latest_relaxed_ik_fk_stamp_ns
                )
                / 1e6,
                "solution_target_receive_delta_ms": (
                    self._latest_relaxed_ik_solution_monotonic
                    - self._latest_relaxed_ik_target_monotonic
                )
                * 1e3,
                "solution_fk_receive_delta_ms": (
                    self._latest_relaxed_ik_solution_monotonic
                    - self._latest_relaxed_ik_fk_monotonic
                )
                * 1e3,
            }
        raise TimeoutError(
            "Timed out waiting for a RelaxedIK joint candidate and FK "
            f"correlated to the requested target: {timeout_details}"
        )

    def reset_relaxed_ik_probe(self) -> None:
        """Reset the isolated solver to measured joints before a new candidate."""
        if not self.r1lite_motion_validation_enabled:
            self._relaxed_ik_initialized = True
            return
        self._relaxed_ik_initialized = False
        self._reset_relaxed_ik_to_current()

    def relaxed_ik_probe_ready(self) -> bool:
        """Return whether both isolated RelaxedIK command inputs are connected."""
        pose_subscriptions = self._node.get_subscriptions_info_by_topic(
            self._relaxed_ik_pose_pub.topic_name
        )
        return bool(
            pose_subscriptions
            and int(self._relaxed_ik_reset_pub.get_subscription_count()) > 0
        )

    def _resize_for_visual_feedback(self, rgb: np.ndarray) -> np.ndarray:
        max_width = self.visual_feedback_max_width
        if max_width <= 0 or rgb.shape[1] <= max_width:
            return np.asarray(rgb, dtype=np.uint8)
        scale = max_width / float(rgb.shape[1])
        new_size = (max_width, max(1, int(round(rgb.shape[0] * scale))))
        pil = PILImage.fromarray(np.asarray(rgb, dtype=np.uint8))
        return np.asarray(pil.resize(new_size, PILImage.Resampling.BILINEAR), dtype=np.uint8)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Return latest head-left RGB image for Cap-X primary visual feedback."""
        if mode != "rgb_array":
            raise ValueError("R1LiteRealLowLevel only supports mode='rgb_array'")
        return self._resize_for_visual_feedback(self.get_head_rgb("left"))

    def render_wrist(self) -> np.ndarray:
        """Return the latest right-wrist RGB image."""
        rgbd = self.get_wrist_rgbd(require_depth=False)
        rgb = rgbd["rgb"]
        if rgb is None:
            raise RuntimeError("No wrist RGB frame is available yet")
        return self._resize_for_visual_feedback(np.asarray(rgb, dtype=np.uint8))

    def move_joints_safe(
        self,
        target: list[float] | tuple[float, ...] | np.ndarray,
        *,
        max_step: float | None = None,
        duration_s: float | None = None,
        expected_feedback: (
            list[float] | tuple[float, ...] | np.ndarray | None
        ) = None,
    ) -> np.ndarray:
        """Move the right arm in small joint-space increments.

        Args:
            target: Six right-arm joint positions in radians.
            max_step: Maximum per-joint step per interpolation waypoint.
            duration_s: Seconds to keep publishing the final target.

        Returns:
            Latest feedback after publishing the final target.
        """
        target_arr = np.asarray(target, dtype=np.float64).reshape(6)
        self._ensure_moca_for_joint_target(target_arr)
        self._enable_arm_joint_control()
        current = self.get_arm_joints()
        with self._lock:
            current_command = (
                None
                if self._last_joint_command is None
                else self._last_joint_command.copy()
            )
        if current_command is None:
            raise RuntimeError(
                "MOCA has not published an active right-arm p_des; refusing "
                "to guess a command/feedback offset"
            )
        expected_feedback_arr = (
            np.asarray(expected_feedback, dtype=np.float64).reshape(6)
            if expected_feedback is not None
            else target_arr
        )
        command_target = (
            target_arr
            if expected_feedback is not None
            else current_command + (target_arr - current)
        )
        initial_error = float(np.max(np.abs(expected_feedback_arr - current)))
        step = self.max_joint_step if max_step is None else float(max_step)
        if step <= 0:
            raise ValueError("max_step must be positive")
        n_steps = max(
            1,
            int(np.ceil(np.max(np.abs(command_target - current_command)) / step)),
        )
        rate_dt = 1.0 / self.publish_rate_hz
        for i in range(1, n_steps + 1):
            waypoint = current_command + (
                command_target - current_command
            ) * (i / n_steps)
            msg = self._make_joint_state(self.arm_name, waypoint.tolist())
            self._arm_pub.publish(msg)
            time.sleep(rate_dt)

        hold_s = self.command_duration_s if duration_s is None else float(duration_s)
        deadline = time.time() + max(0.0, hold_s)
        msg = self._make_joint_state(self.arm_name, command_target.tolist())
        while time.time() < deadline:
            self._arm_pub.publish(msg)
            time.sleep(rate_dt)
            measured = self.get_arm_joints()
            target_error = float(np.max(np.abs(expected_feedback_arr - measured)))
            measured_motion = float(np.max(np.abs(measured - current)))
            if (
                measured_motion >= self.arm_min_motion_rad
                and target_error > initial_error + self.arm_divergence_limit_rad
            ):
                raise RuntimeError(
                    "R1 arm feedback diverged from the joint target; stopping command "
                    "publication. The servo may be passive or another controller may "
                    "have ownership: "
                    f"before={current.tolist()}, desired_feedback="
                    f"{expected_feedback_arr.tolist()}, command_target="
                    f"{command_target.tolist()}, "
                    f"measured={measured.tolist()}, target_error={target_error:.4f}"
                )
        measured = self.get_arm_joints()
        target_error = float(np.max(np.abs(expected_feedback_arr - measured)))
        if target_error > self.arm_joint_tolerance_rad:
            raise TimeoutError(
                "R1 arm joint target did not converge: "
                f"target={target_arr.tolist()}, measured={measured.tolist()}, "
                f"max_error={target_error:.4f}, "
                f"tolerance={self.arm_joint_tolerance_rad:.4f}"
            )
        self._last_action["arm_joints"] = target_arr.copy()
        self._last_joint_command = command_target.copy()
        return measured

    def _publish_joint_command_for(
        self,
        command: np.ndarray,
        duration_s: float,
        *,
        skip_feedback_settle: bool = False,
    ) -> np.ndarray:
        """Hold one joint target without interpolation and return settled feedback."""
        self._assert_motion_ready()
        if not bool(getattr(self, "r1lite_motion_validation_enabled", True)):
            skip_feedback_settle = True
        command = self.validate_moca_joint_command(
            command,
            context="joint command publication",
        )
        self._ensure_moca_for_joint_target(command)
        self._enable_arm_joint_control()
        if self._pending_gripper_target is not None:
            pending_gripper = self._pending_gripper_target
            self._pending_gripper_target = None
            try:
                self.command_gripper(pending_gripper, duration_s=0.8)
            except (RuntimeError, TimeoutError) as exc:
                self._last_action["deferred_gripper_warning"] = {
                    "target": float(pending_gripper),
                    "error": str(exc),
                    "arm_motion_continued": True,
                }
        with self._lock:
            corridor_start = (
                command.copy()
                if self._last_joint_command is None
                else self._last_joint_command.copy()
            )
            margin = float(
                getattr(self, "moca_output_corridor_margin_rad", 0.03)
            )
            self._moca_command_corridor_generation = (
                int(getattr(self, "_moca_command_corridor_generation", 0)) + 1
            )
            corridor_generation = self._moca_command_corridor_generation
            self._active_moca_command_corridor = (
                corridor_generation,
                np.minimum(corridor_start, command) - margin,
                np.maximum(corridor_start, command) + margin,
                command.copy(),
                corridor_start,
            )
        msg = self._make_joint_state(self.arm_name, command.tolist())
        with self._lock:
            after_sequence = self._arm_feedback_sequence
        rate_dt = 1.0 / self.publish_rate_hz
        deadline = time.monotonic() + max(0.0, float(duration_s))
        last_publish = time.monotonic()
        try:
            while time.monotonic() < deadline:
                self._assert_motion_ready()
                self._arm_pub.publish(msg)
                last_publish = time.monotonic()
                time.sleep(rate_dt)
            self._assert_motion_ready()
            if skip_feedback_settle:
                measured = self.get_arm_joints()
            else:
                measured = self._wait_for_settled_arm_feedback(
                    after_sequence=after_sequence,
                    not_before_monotonic=last_publish,
                    timeout_s=self.adaptive_feedback_settle_timeout_s,
                )
            self._assert_motion_ready()
        finally:
            with self._lock:
                corridor = self._active_moca_command_corridor
                if (
                    corridor is not None
                    and corridor[0] == corridor_generation
                ):
                    self._active_moca_command_corridor = None
        self._last_joint_command = command.copy()
        return measured

    def move_joints_feedforward(
        self,
        desired_feedback: list[float] | tuple[float, ...] | np.ndarray,
        *,
        duration_s: float | None = None,
        context: str = "feedforward_joint_target",
        skip_feedback_settle: bool = False,
    ) -> dict[str, Any]:
        """Send one joint target estimated from the current command/feedback offset.

        RelaxedIK returns joints in the measured-feedback convention used by the
        robot state.  The command topic has a fixed local offset, so estimate that
        offset once and let MOCA execute the resulting target without online
        correction or waypoint subdivision.
        """
        desired = np.asarray(desired_feedback, dtype=np.float64).reshape(6)
        desired = self.validate_arm_joint_feedback_target(
            desired,
            context=context,
        )
        measured_before = self.get_arm_joints()
        if self._last_joint_command is None:
            command_offset = self.safe_joint_home - self.safe_joint_home_feedback
            command = desired + command_offset
        else:
            command = self._last_joint_command + (desired - measured_before)
        measured_after = self._publish_joint_command_for(
            command,
            self.command_duration_s if duration_s is None else float(duration_s),
            skip_feedback_settle=skip_feedback_settle,
        )
        requested_motion = float(np.max(np.abs(desired - measured_before)))
        measured_motion = float(np.max(np.abs(measured_after - measured_before)))
        details = {
            "context": context,
            "success": (
                requested_motion < self.arm_min_motion_rad
                or measured_motion >= self.arm_min_motion_rad
            ),
            "desired_feedback": desired.tolist(),
            "measured_before": measured_before.tolist(),
            "command": command.tolist(),
            "measured_after": measured_after.tolist(),
            "measured_motion_rad": measured_motion,
            "max_feedback_error_rad": float(np.max(np.abs(desired - measured_after))),
        }
        self._last_action["feedforward_joint_motion"] = details
        return details

    def _wait_for_settled_arm_feedback(
        self,
        *,
        after_sequence: int,
        not_before_monotonic: float = 0.0,
        timeout_s: float,
    ) -> np.ndarray:
        """Wait for a fresh stable feedback window without resetting MOCA."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        interval = 1.0 / self.publish_rate_hz
        last_sequence = int(after_sequence)
        samples: deque[tuple[float, np.ndarray]] = deque()
        while time.monotonic() < deadline:
            with self._lock:
                fresh = [
                    (sequence, received_at, feedback.copy())
                    for sequence, received_at, feedback in self._arm_feedback_history
                    if sequence > last_sequence
                ]
            for sequence, received_at, feedback in fresh:
                samples.append((received_at, feedback))
                last_sequence = max(last_sequence, sequence)
            if samples:
                newest_at = samples[-1][0]
                window_start = newest_at - self.adaptive_feedback_stability_s
                # Keep the final sample at or before the window boundary so
                # elapsed time can prove that the full window was observed.
                while len(samples) > 1 and samples[1][0] <= window_start:
                    samples.popleft()
                if (
                    newest_at >= not_before_monotonic
                    and len(samples) >= self.adaptive_feedback_min_samples
                    and newest_at - samples[0][0]
                    >= self.adaptive_feedback_stability_s
                ):
                    values = np.asarray(
                        [feedback for _, feedback in samples],
                        dtype=np.float64,
                    )
                    max_span = float(np.max(np.ptp(values, axis=0)))
                    if max_span <= self.adaptive_feedback_stability_rad:
                        return np.median(values, axis=0)
            time.sleep(interval)
        raise TimeoutError(
            "Timed out waiting for fresh stable arm feedback after command: "
            f"after_sequence={after_sequence}, "
            f"not_before_monotonic={not_before_monotonic:.3f}, "
            f"stability_window_s={self.adaptive_feedback_stability_s:.3f}, "
            f"stability_limit_rad={self.adaptive_feedback_stability_rad:.4f}"
        )

    def _append_adaptive_record(self, record: dict[str, Any]) -> None:
        path = Path(self.adaptive_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _adaptive_effective_error(
        self,
        desired: np.ndarray,
        measured: np.ndarray,
        command: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Ignore only a small J2 residual blocked by its physical lower limit."""
        error = np.asarray(desired, dtype=np.float64) - np.asarray(
            measured,
            dtype=np.float64,
        )
        effective_error = error.copy()
        saturated_joints: list[int] = []
        lower_limit = float(
            getattr(
                self,
                "right_arm_j2_command_lower_limit_rad",
                0.01745329252,
            )
        )
        command_arr = np.asarray(command, dtype=np.float64).reshape(6)
        saturation_tolerance = float(
            getattr(self, "arm_joint_tolerance_rad", 0.02)
        )
        if (
            float(command_arr[1]) <= lower_limit + 1e-6
            and float(error[1]) < 0.0
            and abs(float(error[1])) <= saturation_tolerance
        ):
            effective_error[1] = 0.0
            saturated_joints.append(1)
        return effective_error, saturated_joints

    def move_joints_adaptive(
        self,
        desired_feedback: list[float] | tuple[float, ...] | np.ndarray,
        *,
        max_target_step_rad: float | None = None,
        context: str = "adaptive_joint_step",
        skip_feedback_settle: bool = False,
    ) -> dict[str, Any]:
        """Reach a nearby measured joint state using online command correction.

        IK output is treated as desired measured feedback.  The command sent to
        MOCA is updated from observed residuals, so fixed command/feedback
        offsets are not assumed to be calibration constants.
        """
        self._assert_motion_ready()
        if not getattr(self, "r1lite_adaptive_joint_control_enabled", True):
            details = self.move_joints_feedforward(
                desired_feedback,
                context=context,
                skip_feedback_settle=skip_feedback_settle,
            )
            details["adaptive_control_enabled"] = False
            details["mode"] = "feedforward"
            self._last_action["adaptive_joint_motion"] = details
            return details

        desired = np.asarray(desired_feedback, dtype=np.float64).reshape(6)
        if not np.isfinite(desired).all():
            raise ValueError("desired_feedback must be finite")
        measured = self.get_arm_joints()
        max_target_step = (
            self.adaptive_max_target_step_rad
            if max_target_step_rad is None
            else float(max_target_step_rad)
        )
        requested_step = float(np.max(np.abs(desired - measured)))
        if requested_step > max_target_step + 1e-9:
            raise ValueError(
                f"adaptive target step {requested_step:.4f} rad exceeds "
                f"limit {max_target_step:.4f}; insert another waypoint"
            )

        if self._last_joint_command is None:
            local_home_bias = self.safe_joint_home - self.safe_joint_home_feedback
            command = measured + local_home_bias
        else:
            command = self._last_joint_command.copy()
        # A nearby feedback target normally requires the command to move by a
        # similar amount.  Bound only the extra online correction around that
        # unit-gain prediction, not the requested path motion itself.
        nominal_command = command + (desired - measured)

        history: list[dict[str, Any]] = []
        stalled_iterations = 0
        last_responsive_command = command.copy()
        for iteration in range(self.adaptive_max_iterations):
            error_before = desired - measured
            effective_error_before, saturated_joints = (
                self._adaptive_effective_error(
                    desired,
                    measured,
                    command,
                )
            )
            raw_max_error_before = float(np.max(np.abs(error_before)))
            max_error_before = float(
                np.max(np.abs(effective_error_before))
            )
            if max_error_before <= self.adaptive_joint_tolerance_rad:
                details = {
                    "context": context,
                    "success": True,
                    "adaptive_control_enabled": True,
                    "mode": "adaptive",
                    "desired_feedback": desired.tolist(),
                    "measured_feedback": measured.tolist(),
                    "final_command": command.tolist(),
                    "max_error_rad": max_error_before,
                    "max_raw_error_rad": raw_max_error_before,
                    "saturated_joints": saturated_joints,
                    "iterations": iteration,
                    "history": history,
                }
                self._last_action["adaptive_joint_motion"] = details
                return details

            # Correct all axes together so a multi-joint residual does not
            # consume the iteration budget one joint at a time.
            proposed_command_delta = (
                self.adaptive_correction_fraction * error_before
            )
            next_command = self.validate_moca_joint_command(
                command + proposed_command_delta,
                context=f"{context}: adaptive correction",
            )
            command_delta = next_command - command
            command_excursions = np.abs(next_command - nominal_command)
            active_joint = int(np.argmax(command_excursions))
            command_excursion = float(
                command_excursions[active_joint]
            )
            if (
                self.r1lite_motion_validation_enabled
                and command_excursion
                > self.adaptive_max_command_excursion_rad + 1e-9
            ):
                self._enter_motion_fault(
                    reason="adaptive command excursion exceeded safety limit",
                    details={
                        "context": context,
                        "active_joint": active_joint,
                        "command_excursion_rad": command_excursion,
                        "limit_rad": self.adaptive_max_command_excursion_rad,
                        "desired_feedback": desired.tolist(),
                        "measured_feedback": measured.tolist(),
                    },
                )
                raise RuntimeError(
                    f"{context}: joint {active_joint + 1} command excursion "
                    f"{command_excursion:.4f} rad exceeds "
                    f"{self.adaptive_max_command_excursion_rad:.4f}; "
                    "local response is too weak or stalled"
                )
            measured_before = measured.copy()
            try:
                measured = self._publish_joint_command_for(
                    next_command,
                    self.adaptive_settle_s,
                    skip_feedback_settle=skip_feedback_settle,
                )
            except Exception as exc:
                self._enter_motion_fault(
                    reason="joint command feedback did not settle",
                    details={
                        "context": context,
                        "iteration": iteration,
                        "active_joint": active_joint,
                        "command_before": command.tolist(),
                        "command_after": next_command.tolist(),
                        "measured_before": measured_before.tolist(),
                        "error": repr(exc),
                    },
                )
                raise
            measured_delta = measured - measured_before
            error_after = desired - measured
            effective_error_after, saturated_joints_after = (
                self._adaptive_effective_error(
                    desired,
                    measured,
                    next_command,
                )
            )
            raw_max_error_after = float(np.max(np.abs(error_after)))
            max_error_after = float(
                np.max(np.abs(effective_error_after))
            )
            progress = max_error_before - max_error_after
            max_measured_step = float(np.max(np.abs(measured_delta)))
            direction_score = float(np.dot(measured_delta, error_before))

            record = {
                "timestamp": time.time(),
                "context": context,
                "iteration": iteration,
                "active_joint": active_joint,
                "desired_feedback": desired.tolist(),
                "command_before": command.tolist(),
                "command_after": next_command.tolist(),
                "measured_before": measured_before.tolist(),
                "measured_after": measured.tolist(),
                "command_delta": command_delta.tolist(),
                "measured_delta": measured_delta.tolist(),
                "max_error_before": max_error_before,
                "max_error_after": max_error_after,
                "max_raw_error_before": raw_max_error_before,
                "max_raw_error_after": raw_max_error_after,
                "saturated_joints": saturated_joints_after,
                "progress": progress,
                "direction_score": direction_score,
            }
            history.append(record)
            self._append_adaptive_record(record)

            if (
                self.r1lite_motion_validation_enabled
                and max_measured_step > self.adaptive_max_measured_step_rad
            ):
                self._enter_motion_fault(
                    reason="measured joint step exceeded safety limit",
                    details=record,
                )
                raise RuntimeError(
                    f"{context}: measured step {max_measured_step:.4f} rad exceeds "
                    f"safety limit {self.adaptive_max_measured_step_rad:.4f}: {record}"
                )
            if self.r1lite_motion_validation_enabled and (
                max_measured_step >= self.arm_min_motion_rad
                and direction_score <= 0.0
            ):
                self._enter_motion_fault(
                    reason="joint correction moved away from desired feedback",
                    details=record,
                )
                raise RuntimeError(
                    f"{context}: joint correction moved away from desired feedback: "
                    f"{record}"
                )
            if progress < self.adaptive_min_progress_rad:
                stalled_iterations += 1
            else:
                stalled_iterations = 0
                last_responsive_command = next_command.copy()
            if (
                self.r1lite_motion_validation_enabled
                and stalled_iterations >= self.adaptive_max_stalled_iterations
            ):
                self._enter_motion_fault(
                    reason="adaptive correction made no measurable progress",
                    details=record,
                )
                raise RuntimeError(
                    f"{context}: online correction made no measurable progress: {record}"
                )
            command = next_command

        final_effective_error, final_saturated_joints = (
            self._adaptive_effective_error(
                desired,
                measured,
                command,
            )
        )
        final_error = float(np.max(np.abs(final_effective_error)))
        final_raw_error = float(np.max(np.abs(desired - measured)))
        if not self.r1lite_motion_validation_enabled:
            details = {
                "context": context,
                "success": True,
                "adaptive_control_enabled": True,
                "mode": "adaptive",
                "motion_validation_enabled": False,
                "desired_feedback": desired.tolist(),
                "measured_feedback": measured.tolist(),
                "final_command": command.tolist(),
                "max_error_rad": final_error,
                "max_raw_error_rad": final_raw_error,
                "saturated_joints": final_saturated_joints,
                "iterations": self.adaptive_max_iterations,
                "history": history,
            }
            self._last_action["adaptive_joint_motion"] = details
            return details
        self._enter_motion_fault(
            reason="adaptive correction did not converge",
            details={
                "context": context,
                "max_error_rad": final_error,
                "max_raw_error_rad": final_raw_error,
                "saturated_joints": final_saturated_joints,
                "desired_feedback": desired.tolist(),
                "measured_feedback": measured.tolist(),
                "iterations": self.adaptive_max_iterations,
            },
        )
        raise TimeoutError(
            f"{context}: adaptive correction did not converge after "
            f"{self.adaptive_max_iterations} iterations; "
            f"max_error={final_error:.4f}, desired={desired.tolist()}, "
            f"measured={measured.tolist()}"
        )

    def move_joint_waypoints_adaptive(
        self,
        target_feedback: list[float] | tuple[float, ...] | np.ndarray,
        *,
        waypoint_step_rad: float = 0.015,
        context: str = "adaptive_joint_path",
        skip_feedback_settle: bool = False,
    ) -> dict[str, Any]:
        """Follow a larger IK target through measured-joint waypoints."""
        adaptive_enabled = bool(
            getattr(self, "r1lite_adaptive_joint_control_enabled", True)
        )
        target = np.asarray(target_feedback, dtype=np.float64).reshape(6)
        target = self.validate_arm_joint_feedback_target(
            target,
            context=context,
        )
        start = self.get_arm_joints()
        step = float(waypoint_step_rad)
        max_step_limit = (
            self.adaptive_max_target_step_rad
            if adaptive_enabled
            else self.max_joint_step
        )
        if step <= 0.0 or step > max_step_limit:
            raise ValueError(
                "waypoint_step_rad must be positive and no greater than "
                f"{max_step_limit:.4f}"
            )
        initial_distance = float(np.max(np.abs(target - start)))
        minimum_progress = max(
            0.001,
            step - self.adaptive_joint_tolerance_rad,
        )
        max_segments = max(
            1,
            int(np.ceil(initial_distance / minimum_progress)) + 4,
        )
        results = []
        for index in range(1, max_segments + 1):
            current = self.get_arm_joints()
            delta = target - current
            with self._lock:
                convergence_command = (
                    None
                    if self._last_joint_command is None
                    else self._last_joint_command.copy()
                )
            if convergence_command is None:
                convergence_command = current.copy()
            effective_delta, _ = self._adaptive_effective_error(
                target,
                current,
                convergence_command,
            )
            remaining = float(np.max(np.abs(effective_delta)))
            if remaining <= self.adaptive_joint_tolerance_rad:
                break
            raw_remaining = float(np.max(np.abs(delta)))
            waypoint = current + delta * (
                min(step, raw_remaining) / raw_remaining
            )
            segment_context = f"{context}_{index:03d}_of_{max_segments:03d}"
            if adaptive_enabled:
                results.append(
                    self.move_joints_adaptive(
                        waypoint,
                        max_target_step_rad=self.adaptive_max_target_step_rad,
                        context=segment_context,
                        skip_feedback_settle=skip_feedback_settle,
                    )
                )
            else:
                segment = self.move_joints_feedforward(
                    waypoint,
                    context=segment_context,
                    skip_feedback_settle=skip_feedback_settle,
                )
                segment["adaptive_control_enabled"] = False
                segment["mode"] = "feedforward"
                results.append(segment)
        measured = self.get_arm_joints()
        with self._lock:
            final_command = (
                measured.copy()
                if self._last_joint_command is None
                else self._last_joint_command.copy()
            )
        final_effective_error, saturated_joints = (
            self._adaptive_effective_error(
                target,
                measured,
                final_command,
            )
        )
        final_error = float(np.max(np.abs(final_effective_error)))
        final_raw_error = float(np.max(np.abs(target - measured)))
        if (
            self.r1lite_motion_validation_enabled
            and adaptive_enabled
            and final_error > self.adaptive_joint_tolerance_rad
        ):
            raise TimeoutError(
                f"{context}: adaptive waypoint path did not converge; "
                f"max_error={final_error:.4f}, segments={len(results)}, "
                f"limit={max_segments}"
            )
        details = {
            "context": context,
            "success": True,
            "adaptive_control_enabled": adaptive_enabled,
            "skip_feedback_settle": skip_feedback_settle,
            "mode": "adaptive" if adaptive_enabled else "feedforward",
            "target_feedback": target.tolist(),
            "measured_feedback": measured.tolist(),
            "max_feedback_error_rad": final_error,
            "max_raw_feedback_error_rad": final_raw_error,
            "saturated_joints": saturated_joints,
            "waypoint_count": len(results),
            "waypoints": results,
        }
        self._last_action["adaptive_joint_path"] = details
        return details

    def command_gripper(
        self,
        position: float,
        *,
        duration_s: float | None = None,
        allow_contact: bool = False,
    ) -> float:
        """Command the gripper and require corresponding physical feedback."""
        target = float(np.clip(position, 0.0, 100.0))
        before = self.get_gripper_position()
        if (
            int(self._gripper_pub.get_subscription_count()) == 0
            and self.lazy_moca_start_command is not None
            and not self._moca_started_by_env
        ):
            self._pending_gripper_target = target
            self._last_action["deferred_gripper"] = {
                "target": target,
                "reason": "waiting_for_first_valid_arm_joint_target",
            }
            return before
        self._wait_for_subscriber(self._gripper_pub, "right gripper target")
        rate_dt = 1.0 / self.publish_rate_hz
        hold_s = self.command_duration_s if duration_s is None else float(duration_s)
        msg = self._make_joint_state(self.gripper_name, [target])
        deadline = time.time() + max(0.0, hold_s)
        while time.time() < deadline:
            self._gripper_pub.publish(msg)
            time.sleep(rate_dt)
        after = self.get_gripper_position()
        target_error = abs(after - target)
        response = abs(after - before)
        moved_toward_target = abs(after - target) < abs(before - target)
        reached = target_error <= self.gripper_feedback_tolerance
        contact_response = (
            allow_contact
            and response >= self.gripper_min_response
            and moved_toward_target
        )
        if (
            self.r1lite_motion_validation_enabled
            and not allow_contact
            and not (reached or contact_response)
        ):
            raise RuntimeError(
                "R1 gripper command produced no valid feedback response: "
                f"before={before:.3f}, target={target:.3f}, after={after:.3f}, "
                f"target_error={target_error:.3f}, response={response:.3f}"
            )
        self._last_action["gripper"] = target
        self._last_action["gripper_feedback"] = after
        return after

    def return_home(self, *, duration_s: float | None = None) -> dict[str, Any]:
        """Publish the calibrated power-on command and verify home feedback."""
        measured_before = self.get_arm_joints()
        command = self.safe_joint_home.copy()
        self._ensure_moca_for_joint_target(command)
        self._enable_arm_joint_control(reset_function_frame=True)
        hold_s = max(10.0, self.command_duration_s) if duration_s is None else float(duration_s)
        rate_dt = 1.0 / self.publish_rate_hz
        msg = self._make_joint_state(self.arm_name, command.tolist())
        with self._lock:
            after_sequence = self._arm_feedback_sequence
        deadline = time.monotonic() + max(0.0, hold_s)
        last_publish = time.monotonic()
        while time.monotonic() < deadline:
            self._arm_pub.publish(msg)
            last_publish = time.monotonic()
            time.sleep(rate_dt)
        if self.r1lite_motion_validation_enabled:
            measured_after = self._wait_for_settled_arm_feedback(
                after_sequence=after_sequence,
                not_before_monotonic=last_publish,
                timeout_s=self.adaptive_feedback_settle_timeout_s,
            )
        else:
            measured_after = self.get_arm_joints()
        self._last_joint_command = command.copy()
        max_error = float(
            np.max(np.abs(self.safe_joint_home_feedback - measured_after))
        )
        if (
            self.r1lite_motion_validation_enabled
            and max_error > self.arm_joint_tolerance_rad
        ):
            raise TimeoutError(
                "R1 arm did not reach the calibrated power-on position: "
                f"expected_feedback={self.safe_joint_home_feedback.tolist()}, "
                f"measured={measured_after.tolist()}, max_error={max_error:.4f}, "
                f"tolerance={self.arm_joint_tolerance_rad:.4f}"
            )
        details = {
            "context": "return_home",
            "success": True,
            "desired_feedback": self.safe_joint_home_feedback.tolist(),
            "measured_before": measured_before.tolist(),
            "command": command.tolist(),
            "measured_after": measured_after.tolist(),
            "max_feedback_error_rad": max_error,
        }
        self._last_action["return_home"] = details
        return details

    def open_gripper(self) -> float:
        """Open the right gripper to the configured open position."""
        return self.command_gripper(
            self.gripper_open_position,
            duration_s=max(5.0, self.command_duration_s),
        )

    def close_gripper(self) -> float:
        """Close the right gripper to the configured close position."""
        return self.command_gripper(self.gripper_close_position, allow_contact=True)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset task state, optionally restoring the configured home pose."""
        _ = seed, options
        self.wait_until_ready()
        if self.reset_motion_enabled:
            self.return_home()
            self.open_gripper()
        return self.get_observation(), {}

    def step(
        self,
        action: Any,
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply a dict action with optional ``arm_joints`` and ``gripper`` keys."""
        if isinstance(action, dict):
            if "arm_joints" in action:
                self.move_joints_safe(action["arm_joints"])
            if "gripper" in action:
                self.command_gripper(float(action["gripper"]))
        return self.get_observation(), self.compute_reward(), False, False, {}

    def get_observation(self) -> dict[str, Any]:
        """Return latest joint and gripper feedback."""
        wrist_rgbd = None
        if self.wrist_camera_enabled:
            try:
                wrist_rgbd = self.get_wrist_rgbd(require_depth=False)
            except TimeoutError:
                wrist_rgbd = None
        return {
            "right_arm_joints": self.get_arm_joints(),
            "right_gripper_position": self.get_gripper_position(),
            "head_left_rgb": self.get_head_rgb("left"),
            "wrist_rgbd": wrist_rgbd,
            "last_action": self._last_action.copy(),
        }

    def ensure_ee_control_enabled(self) -> None:
        """Enable moca_adapter right-arm control with adapter interpolation disabled."""
        if self._ee_control_ready:
            return
        if not self._moca_param_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(
                f"moca_adapter set_parameters unavailable at {self.moca_param_node}"
            )
        req = self._SetParameters.Request()
        req.parameters = [
            self._ParameterMsg(
                name="end_effector_control.enable",
                value=self._ParameterValue(
                    type=self._ParameterType.PARAMETER_BOOL, bool_value=True
                ),
            ),
            self._ParameterMsg(
                name="end_effector_control.enable_right_ee",
                value=self._ParameterValue(
                    type=self._ParameterType.PARAMETER_BOOL, bool_value=True
                ),
            ),
            self._ParameterMsg(
                name="end_effector_control.use_interpolation",
                value=self._ParameterValue(
                    type=self._ParameterType.PARAMETER_BOOL, bool_value=False
                ),
            ),
            self._ParameterMsg(
                name="end_effector_control.ee_control_mode",
                value=self._ParameterValue(
                    type=self._ParameterType.PARAMETER_STRING,
                    string_value="independent",
                ),
            ),
        ]
        fut = self._moca_param_client.call_async(req)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            self._spin_once_or_sleep()
            if fut.done():
                break
        if not fut.done() or fut.result() is None:
            raise RuntimeError("moca_adapter did not confirm EE control parameters")
        rejected = [
            result.reason
            for result in fut.result().results
            if not result.successful
        ]
        if rejected:
            raise RuntimeError(f"moca_adapter rejected EE control parameters: {rejected}")
        self._ee_control_ready = True

    def hold_arm_pose(
        self,
        position_xyz: list[float] | tuple[float, ...] | np.ndarray,
        quaternion_xyzw: list[float] | tuple[float, ...] | np.ndarray,
        *,
        hold_sec: float = 1.0,
        frame_id: str = "base_link",
        require_motion: bool = True,
    ) -> dict[str, Any]:
        """Publish a pose target until measured end-effector feedback converges."""
        if self.use_moca_ee_control:
            self.ensure_ee_control_enabled()
        elif (
            not self._relaxed_ik_initialized
            and self.r1lite_motion_validation_enabled
        ):
            self._reset_relaxed_ik_to_current()
        elif not self._relaxed_ik_initialized:
            self._relaxed_ik_initialized = True
        pos = np.asarray(position_xyz, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1e-8 or not np.isfinite(quat_norm):
            raise ValueError("quaternion_xyzw must be finite and non-zero")
        quat = quat / quat_norm
        self.validate_workspace_point(pos)
        if self.use_moca_ee_control:
            command_pos, command_quat = pos, quat
        else:
            command_pos, command_quat = self._pose_in_relaxed_ik_frame(pos, quat)
        self._wait_for_subscriber(self._relaxed_ik_pose_pub, "right arm pose target")
        interval = max(1.0 / self.ee_pose_publish_hz, 0.02)
        started = time.monotonic()
        minimum_hold_until = started + max(0.0, float(hold_sec))
        deadline = started + max(float(hold_sec), self.pose_command_timeout_s)
        last_position: np.ndarray | None = None
        last_quaternion: np.ndarray | None = None
        position_error = float("inf")
        orientation_error = float("inf")

        while time.monotonic() < deadline:
            msg = self._PoseStamped()
            msg.header.frame_id = frame_id
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.pose.position.x = float(command_pos[0])
            msg.pose.position.y = float(command_pos[1])
            msg.pose.position.z = float(command_pos[2])
            msg.pose.orientation.x = float(command_quat[0])
            msg.pose.orientation.y = float(command_quat[1])
            msg.pose.orientation.z = float(command_quat[2])
            msg.pose.orientation.w = float(command_quat[3])
            self._relaxed_ik_pose_pub.publish(msg)
            time.sleep(interval)

            with self._lock:
                if self._latest_ee_position is not None:
                    last_position = self._latest_ee_position.copy()
                if self._latest_ee_quaternion_xyzw is not None:
                    last_quaternion = self._latest_ee_quaternion_xyzw.copy()

            if last_position is None or last_quaternion is None:
                continue
            position_error = float(np.linalg.norm(last_position - pos))
            measured_norm = float(np.linalg.norm(last_quaternion))
            if measured_norm < 1e-8:
                continue
            measured_quat = last_quaternion / measured_norm
            quat_dot = float(np.clip(abs(np.dot(measured_quat, quat)), 0.0, 1.0))
            orientation_error = float(2.0 * np.arccos(quat_dot))
            if (
                time.monotonic() >= minimum_hold_until
                and position_error <= self.pose_position_tolerance_m
                and orientation_error <= self.pose_orientation_tolerance_rad
            ):
                details = {
                    "context": f"hold_arm_pose target={np.array2string(pos, precision=3)}",
                    "target_position": pos.tolist(),
                    "measured_position": last_position.tolist(),
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    "elapsed_s": time.monotonic() - started,
                }
                self._last_action["arm_motion"] = details
                return details

        details = {
            "target_position": pos.tolist(),
            "measured_position": (
                None if last_position is None else last_position.tolist()
            ),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "position_tolerance_m": self.pose_position_tolerance_m,
            "orientation_tolerance_rad": self.pose_orientation_tolerance_rad,
            "timeout_s": self.pose_command_timeout_s,
            "require_motion": require_motion,
        }
        raise TimeoutError(f"R1 end-effector pose did not converge: {details}")

    def _pose_in_relaxed_ik_frame(
        self,
        position_base: np.ndarray,
        quaternion_base_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a base_link target into RelaxedIK's floating-right-arm frame."""
        base_from_ik, torso_joints = self._base_from_torso_link3()
        base_from_ik_rot = base_from_ik[:3, :3]
        base_from_ik_translation = base_from_ik[:3, 3]
        target_base_rot = SciRotation.from_quat(quaternion_base_xyzw).as_matrix()
        target_ik_pos = base_from_ik_rot.T @ (
            position_base - base_from_ik_translation
        )
        target_ik_rot = base_from_ik_rot.T @ target_base_rot
        target_ik_quat = SciRotation.from_matrix(target_ik_rot).as_quat()
        self._last_action["relaxed_ik_frame_transform"] = {
            "source": "r1lite_2026.urdf_torso_fk",
            "torso_joints": torso_joints.tolist(),
            "base_from_relaxed_ik": base_from_ik.tolist(),
        }
        return target_ik_pos, target_ik_quat

    def _reset_relaxed_ik_to_current(self) -> None:
        """Seed RelaxedIK with measured joints to avoid a distant solution branch."""
        current = self.get_arm_joints()
        self.seed_relaxed_ik_probe(current, duration_s=0.5)

    def seed_relaxed_ik_probe(
        self,
        joints: list[float] | tuple[float, ...] | np.ndarray,
        *,
        duration_s: float = 0.2,
    ) -> None:
        """Explicitly seed a solver-only RelaxedIK probe.

        Unlike :meth:`reset_relaxed_ik_probe`, this method intentionally ignores
        ``r1lite_motion_validation_enabled``. Candidate search uses it to give
        every orientation probe the same measured-joint seed without changing
        the legacy automatic-reset semantics.
        """
        if not self.relaxed_ik_probe_enabled:
            raise RuntimeError("RelaxedIK probe is disabled")
        seed = np.asarray(joints, dtype=np.float64).reshape(6)
        if not np.isfinite(seed).all():
            raise ValueError("RelaxedIK seed joints must be finite")
        duration = float(duration_s)
        if duration <= 0.0:
            raise ValueError("duration_s must be positive")
        self._wait_for_subscriber(self._relaxed_ik_reset_pub, "RelaxedIK reset")
        msg = self._make_joint_state(self.arm_name, seed.tolist())
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self._relaxed_ik_reset_pub.publish(msg)
            time.sleep(0.05)
        self._relaxed_ik_initialized = True
        self._last_action["relaxed_ik_explicit_seed"] = {
            "joints": seed.tolist(),
            "duration_s": duration,
        }

    def verify_arm_motion(
        self,
        before: list[float] | tuple[float, ...] | np.ndarray,
        *,
        context: str,
        require_motion: bool = True,
    ) -> dict[str, Any]:
        """Require measurable feedback change after a Cartesian command."""
        before_arr = np.asarray(before, dtype=np.float64).reshape(6)
        deadline = time.time() + self.arm_motion_timeout_s
        after = self.get_arm_joints()
        max_delta = float(np.max(np.abs(after - before_arr)))
        while require_motion and max_delta < self.arm_min_motion_rad and time.time() < deadline:
            time.sleep(0.05)
            after = self.get_arm_joints()
            max_delta = float(np.max(np.abs(after - before_arr)))
        details = {
            "context": context,
            "before": before_arr.tolist(),
            "after": after.tolist(),
            "max_joint_delta": max_delta,
            "minimum_required": self.arm_min_motion_rad,
        }
        if require_motion and max_delta < self.arm_min_motion_rad:
            raise RuntimeError(f"R1 arm command was ignored or overridden: {details}")
        self._last_action["arm_motion"] = details
        return details

    def object_prompt(self, object_name: str) -> str:
        """Return the SAM3 text prompt for a canonical object name."""
        key = str(object_name).strip().lower()
        return self.object_prompts.get(key, str(object_name))

    def compute_reward(self) -> float:
        """No built-in task reward for the real R1 Lite safety wrapper."""
        return 0.0

    def task_completed(self) -> bool:
        """Real-world completion is not inferred by this low-level wrapper."""
        return False

    def close(self) -> None:
        """Shut down ROS 2 resources owned by this wrapper."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._stop_lazy_moca()
        except Exception:
            pass
        self._release_moca_lock()
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=2.0)
            except Exception:
                pass
        spin_thread = getattr(self, "_spin_thread", None)
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=3.0)
        node = getattr(self, "_node", None)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if (
            getattr(self, "_owns_rclpy_context", False)
            and getattr(self, "_rclpy", None) is not None
            and self._rclpy.ok()
        ):
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=2.0)
