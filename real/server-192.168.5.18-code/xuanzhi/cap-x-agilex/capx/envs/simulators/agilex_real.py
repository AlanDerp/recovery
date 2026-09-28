"""Real Agilex/Piper single-arm low-level environment scaffold.

This module follows the same low-level environment shape used by the real
Franka adapter, but leaves the hardware-specific transport details explicit.
Agilex is modeled as one 6-DoF arm plus one gripper scalar, with one main
camera.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.utils.camera_utils import obs_get_rgb, scale_intrinsics_for_image
from capx.utils.msgpack_server_client_utils import MsgpackNumpyServer
from capx.utils.video_utils import resize_with_pad

CAMERA_TO_OBS_KEY = {
    "main": "robot0_robotview",
}
CAMERA_MSG_ALIASES = {
    "main": ("camera_main", "main"),
}
DEFAULT_RGB_TOPICS = {
    "main": "/camera_h/color/image_raw",
}
DEFAULT_DEPTH_TOPICS = {
    "main": "/camera_h/depth/image_raw",
}
DEFAULT_CAMERA_INFO_TOPICS = {
    "main": "/camera_h/color/camera_info",
}
PIPER_JOINT_LOWER = np.array(
    [-2.6179938, 0.0, -2.9670597, -1.7453292, -1.2217304, -2.0943951],
    dtype=np.float64,
)
PIPER_JOINT_UPPER = np.array(
    [2.6179938, 3.1415926, 0.0, 1.7453292, 1.2217304, 2.0943951],
    dtype=np.float64,
)


def clip_piper_joint_targets(
    joints: np.ndarray | list[float],
    *,
    margin: float = 0.005,
) -> np.ndarray:
    """Clip joint commands to Piper/pyAgxArm limits (with optional margin)."""
    joints_arr = np.asarray(joints, dtype=np.float64).reshape(6)
    margin_f = max(0.0, float(margin))
    max_margin = float(np.min((PIPER_JOINT_UPPER - PIPER_JOINT_LOWER) / 2.0)) - 1e-6
    margin_f = min(margin_f, max_margin)
    lower = PIPER_JOINT_LOWER + margin_f
    upper = PIPER_JOINT_UPPER - margin_f
    return np.clip(joints_arr, lower, upper)


def _lookup(mapping: Mapping[Any, Any] | None, *keys: str) -> Any:
    if mapping is None:
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
        key_b = key.encode()
        if key_b in mapping:
            return mapping[key_b]
    return None


def _as_depth_image(depth: Any) -> np.ndarray | None:
    if depth is None:
        return None
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def _camera_obs_from_msg(cam: Mapping[Any, Any] | None) -> dict[str, Any]:
    obs: dict[str, Any] = {"images": {}}
    if cam is None:
        return obs

    images = _lookup(cam, "images") or {}
    rgb = _lookup(images, "rgb", "left_rgb", "color")
    if rgb is not None:
        obs["images"]["rgb"] = np.asarray(rgb, dtype=np.uint8)

    depth = _as_depth_image(_lookup(cam, "depth", "depth_data", "depth_linear"))
    if depth is not None:
        obs["images"]["depth"] = depth

    intrinsics = _lookup(cam, "intrinsics")
    if isinstance(intrinsics, Mapping):
        nested = _lookup(intrinsics, "left", "main", "camera")
        if isinstance(nested, Mapping):
            intrinsics = _lookup(nested, "intrinsics_matrix", "matrix")
    if intrinsics is not None:
        obs["intrinsics"] = np.asarray(intrinsics, dtype=np.float64)

    pose = _lookup(cam, "pose")
    if pose is not None:
        obs["pose"] = np.asarray(pose, dtype=np.float64)

    pose_mat = _lookup(cam, "pose_mat", "extrinsics", "extrinsics_matrix")
    if pose_mat is not None:
        obs["pose_mat"] = np.asarray(pose_mat, dtype=np.float64)

    return obs


def _is_ros_initialized(rospy: Any) -> bool:
    try:
        return bool(rospy.core.is_initialized())
    except Exception:
        return False


def _normalize_rgb_image(image: np.ndarray, encoding: str | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if encoding is not None and encoding.lower() == "bgr8":
        try:
            import cv2

            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except ImportError:
            arr = arr[:, :, ::-1]
    return np.asarray(arr, dtype=np.uint8)


def _normalize_depth_image(image: np.ndarray, *, depth_scale: float) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) * float(depth_scale)
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


class RosCameraAdapter:
    """Subscribe to Agilex RGB-D cameras and expose CaP-X camera observations."""

    def __init__(
        self,
        *,
        rgb_topics: Mapping[str, str] | None = None,
        depth_topics: Mapping[str, str] | None = None,
        camera_info_topics: Mapping[str, str] | None = None,
        use_depth_image: bool = False,
        depth_scale: float = 0.001,
        sync_wait_timeout: float = 0.2,
        queue_size: int = 2000,
        camera_intrinsics: Mapping[str, Any] | None = None,
        camera_poses: Mapping[str, Any] | None = None,
        camera_pose_mats: Mapping[str, Any] | None = None,
        node_name: str = "capx_agilex_low_level",
    ) -> None:
        try:
            import rospy
            from cv_bridge import CvBridge
            from sensor_msgs.msg import CameraInfo
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS camera backend requires rospy, cv_bridge, and sensor_msgs."
            ) from exc

        self.rospy = rospy
        self.bridge = CvBridge()
        self.Image = Image
        self.CameraInfo = CameraInfo
        self.use_depth_image = use_depth_image
        self.depth_scale = depth_scale
        self.sync_wait_timeout = float(sync_wait_timeout)
        self.camera_intrinsics = camera_intrinsics or {}
        self.camera_intrinsics_ref_shape: dict[str, tuple[int, int]] = {}
        self.camera_poses = camera_poses or {}
        self.camera_pose_mats = camera_pose_mats or {}
        self._lock = threading.Lock()
        self._rgb_buffers: dict[str, deque[Any]] = {
            camera: deque(maxlen=queue_size) for camera in CAMERA_TO_OBS_KEY
        }
        self._depth_buffers: dict[str, deque[Any]] = {
            camera: deque(maxlen=queue_size) for camera in CAMERA_TO_OBS_KEY
        }

        if not _is_ros_initialized(rospy):
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

        self.rgb_topics = {**DEFAULT_RGB_TOPICS, **dict(rgb_topics or {})}
        self.depth_topics = {**DEFAULT_DEPTH_TOPICS, **dict(depth_topics or {})}
        self.camera_info_topics = {
            **DEFAULT_CAMERA_INFO_TOPICS,
            **dict(camera_info_topics or {}),
        }

        for camera, topic in self.rgb_topics.items():
            if camera not in self._rgb_buffers:
                continue
            rospy.Subscriber(
                topic,
                Image,
                self._make_callback(self._rgb_buffers[camera]),
                queue_size=1000,
                tcp_nodelay=True,
            )

        for camera, topic in self.camera_info_topics.items():
            if camera not in self._rgb_buffers:
                continue
            rospy.Subscriber(
                topic,
                CameraInfo,
                self._make_camera_info_callback(camera),
                queue_size=10,
                tcp_nodelay=True,
            )

        if use_depth_image:
            for camera, topic in self.depth_topics.items():
                if camera not in self._depth_buffers:
                    continue
                rospy.Subscriber(
                    topic,
                    Image,
                    self._make_callback(self._depth_buffers[camera]),
                    queue_size=1000,
                    tcp_nodelay=True,
                )

    def _make_callback(self, buffer: deque[Any]):
        def callback(msg: Any) -> None:
            with self._lock:
                buffer.append(msg)

        return callback

    def _make_camera_info_callback(self, camera: str):
        def callback(msg: Any) -> None:
            with self._lock:
                self.camera_intrinsics[camera] = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
                self.camera_intrinsics_ref_shape[camera] = (int(msg.height), int(msg.width))

        return callback

    def debug_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "required_camera": "main",
                "rgb_buffers": {
                    camera: len(buffer) for camera, buffer in self._rgb_buffers.items()
                },
                "depth_buffers": {
                    camera: len(buffer) for camera, buffer in self._depth_buffers.items()
                },
                "intrinsics": sorted(self.camera_intrinsics.keys()),
                "pose_mats": sorted(self.camera_pose_mats.keys()),
            }

    def get_observation(self) -> dict[str, Any]:
        self._refresh_empty_buffers_from_ros()
        frame = self._get_synced_frame()
        if frame is None:
            return {}

        obs: dict[str, Any] = {}
        for camera_name, payload in frame.items():
            obs_key = CAMERA_TO_OBS_KEY[camera_name]
            cam_obs: dict[str, Any] = {"images": {}}

            rgb_msg = payload.get("rgb")
            if rgb_msg is not None:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "passthrough")
                cam_obs["images"]["rgb"] = _normalize_rgb_image(
                    rgb, getattr(rgb_msg, "encoding", None)
                )

            depth_msg = payload.get("depth")
            if depth_msg is not None:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
                cam_obs["images"]["depth"] = _normalize_depth_image(
                    depth, depth_scale=self.depth_scale
                )

            if camera_name in self.camera_intrinsics:
                intrinsics = np.asarray(
                    self.camera_intrinsics[camera_name], dtype=np.float64
                )
                depth_img = cam_obs["images"].get("depth")
                ref_shape = self.camera_intrinsics_ref_shape.get(camera_name)
                if depth_img is not None and ref_shape is not None:
                    intrinsics = scale_intrinsics_for_image(
                        intrinsics,
                        np.asarray(depth_img).shape[:2],
                        ref_shape,
                    )
                cam_obs["intrinsics"] = intrinsics
            if camera_name in self.camera_poses:
                cam_obs["pose"] = np.asarray(self.camera_poses[camera_name], dtype=np.float64)
            if camera_name in self.camera_pose_mats:
                cam_obs["pose_mat"] = np.asarray(
                    self.camera_pose_mats[camera_name], dtype=np.float64
                )
            obs[obs_key] = cam_obs

        return obs

    def _get_synced_frame(self) -> dict[str, dict[str, Any]] | None:
        with self._lock:
            if len(self._rgb_buffers["main"]) == 0:
                return None

            frame: dict[str, dict[str, Any]] = {}
            for camera in CAMERA_TO_OBS_KEY:
                rgb_buf = self._rgb_buffers[camera]
                if not rgb_buf:
                    continue
                rgb_msg = rgb_buf[-1]
                frame[camera] = {"rgb": rgb_msg}

                if self.use_depth_image:
                    depth_buf = self._depth_buffers[camera]
                    if depth_buf:
                        target_time = rgb_msg.header.stamp.to_sec()
                        frame[camera]["depth"] = min(
                            depth_buf,
                            key=lambda msg: abs(msg.header.stamp.to_sec() - target_time),
                        )

        return frame

    def _refresh_empty_buffers_from_ros(self) -> None:
        timeout = max(0.0, self.sync_wait_timeout)
        if timeout <= 0:
            return

        for camera, buffer in self._rgb_buffers.items():
            if len(buffer) > 0:
                continue
            topic = self.rgb_topics.get(camera)
            if topic is None:
                continue
            try:
                msg = self.rospy.wait_for_message(topic, self.Image, timeout=timeout)
            except Exception:
                continue
            with self._lock:
                buffer.append(msg)

        if self.use_depth_image:
            for camera, buffer in self._depth_buffers.items():
                if len(buffer) > 0:
                    continue
                topic = self.depth_topics.get(camera)
                if topic is None:
                    continue
                try:
                    msg = self.rospy.wait_for_message(topic, self.Image, timeout=0.05)
                except Exception:
                    continue
                with self._lock:
                    buffer.append(msg)

        for camera in CAMERA_TO_OBS_KEY:
            if camera in self.camera_intrinsics:
                continue
            topic = self.camera_info_topics.get(camera)
            if topic is None:
                continue
            try:
                msg = self.rospy.wait_for_message(topic, self.CameraInfo, timeout=0.05)
            except Exception:
                continue
            with self._lock:
                self.camera_intrinsics[camera] = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
                self.camera_intrinsics_ref_shape[camera] = (int(msg.height), int(msg.width))


class AgilexArmHardware:
    """Direct pyAgxArm backend for one Piper arm and AGX gripper."""

    def __init__(
        self,
        *,
        channel: str = "can_left",
        bitrate: int = 1000000,
        robot: str = "piper",
        speed_pct: int = 50,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 0.1,
        max_linear_acc: float = 0.4,
        max_angular_acc: float = 0.05,
        enable_retries: int = 5,
        gripper_force: float = 1.0,
        gripper_max_width: float = 0.1,
        command_sleep_s: float = 0.02,
        joint_limit_margin: float = 0.005,
    ) -> None:
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config
        except ImportError as exc:
            raise RuntimeError("Direct Agilex backend requires pyAgxArm.") from exc

        self.gripper_force = gripper_force
        self.gripper_max_width = gripper_max_width
        self.command_sleep_s = command_sleep_s
        self.joint_limit_margin = float(max(0.0, joint_limit_margin))
        cfg = create_agx_arm_config(
            robot=robot,
            comm="can",
            channel=channel,
            bitrate=bitrate,
        )
        self.arm = AgxArmFactory.create_arm(cfg)
        self.gripper = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.arm.connect()
        time.sleep(0.2)

        if not self.arm.is_ok():
            raise RuntimeError("Agilex arm connection check failed")

        self.arm.set_flange_vel_acc_limits(
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            max_linear_acc=max_linear_acc,
            max_angular_acc=max_angular_acc,
            timeout=1.0,
        )
        self.arm.set_speed_percent(speed_pct)

        enabled = False
        for _ in range(enable_retries):
            if self.arm.enable():
                enabled = True
                break
            time.sleep(0.5)
        if not enabled:
            raise RuntimeError("Agilex arm enable timed out")

    def get_state(self) -> tuple[np.ndarray, float]:
        joints = np.zeros(6, dtype=np.float64)
        width = 0.0
        joint_msg = self.arm.get_joint_angles()
        if joint_msg is not None:
            joints = np.asarray(joint_msg.msg, dtype=np.float64).reshape(6)
        if self.gripper is not None:
            gripper_msg = self.gripper.get_gripper_status()
            if gripper_msg is not None:
                width = float(gripper_msg.msg.width)
        return joints, width

    def move(
        self,
        *,
        joints: np.ndarray | None = None,
        gripper_width: float | None = None,
    ) -> None:
        if joints is not None:
            clipped = clip_piper_joint_targets(
                joints, margin=self.joint_limit_margin
            )
            self.arm.move_j(clipped.tolist())
        if self.gripper is not None and gripper_width is not None:
            width = float(np.clip(gripper_width, 0.0, self.gripper_max_width))
            self.gripper.move_gripper(width=width, force=self.gripper_force)
        if self.command_sleep_s > 0:
            time.sleep(self.command_sleep_s)


class AgilexRosArmHardware:
    """ROS JointState backend for Piper master/puppet teleop topics."""

    DEFAULT_MASTER_TOPIC = "/master/joint_left"
    DEFAULT_PUPPET_TOPIC = "/puppet/joint_left"

    def __init__(
        self,
        *,
        master_joint_topic: str = DEFAULT_MASTER_TOPIC,
        puppet_joint_topic: str = DEFAULT_PUPPET_TOPIC,
        enable_flag_topic: str = "/enable_flag",
        gripper_max_width: float = 0.1,
        command_sleep_s: float = 0.02,
        joint_limit_margin: float = 0.005,
        rospy_node_name: str = "agilex_capx_arm",
        puppet_wait_timeout: float = 5.0,
        auto_enable_on_connect: bool = True,
    ) -> None:
        try:
            import rospy
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Bool
        except ImportError as exc:
            raise RuntimeError(
                "ROS arm backend requires rospy and sensor_msgs."
            ) from exc

        self._rospy = rospy
        self._JointState = JointState
        self._Bool = Bool
        self._puppet_joint_topic = puppet_joint_topic
        self.gripper_max_width = float(gripper_max_width)
        self.command_sleep_s = float(command_sleep_s)
        self.joint_limit_margin = float(max(0.0, joint_limit_margin))
        self._lock = threading.Lock()
        self._latest_puppet: Any = None
        self._cmd_joints = np.zeros(6, dtype=np.float64)
        self._cmd_width = self.gripper_max_width

        if not rospy.core.is_initialized():
            rospy.init_node(rospy_node_name, anonymous=True, disable_signals=True)

        rospy.Subscriber(
            puppet_joint_topic,
            JointState,
            self._on_puppet,
            queue_size=50,
            tcp_nodelay=True,
        )
        self._pub = rospy.Publisher(master_joint_topic, JointState, queue_size=10)
        self._enable_pub = rospy.Publisher(enable_flag_topic, Bool, queue_size=1, latch=True)
        self._spin_thread = threading.Thread(
            target=rospy.spin,
            name="agilex_ros_arm_spin",
            daemon=True,
        )
        self._spin_thread.start()
        if auto_enable_on_connect:
            self._publish_enable_flag(True)
        deadline = time.time() + max(0.0, puppet_wait_timeout)
        while time.time() < deadline:
            if self._poll_puppet_feedback(timeout_s=0.05) is not None:
                break
        if self._latest_puppet is None:
            raise RuntimeError(
                f"Timed out waiting for puppet joint feedback on {puppet_joint_topic!r}. "
                "Start the Piper driver first, e.g. bash scripts/start_piper_left.sh"
            )

    def _on_puppet(self, msg: Any) -> None:
        with self._lock:
            self._latest_puppet = msg

    def _publish_enable_flag(self, enabled: bool) -> None:
        msg = self._Bool()
        msg.data = bool(enabled)
        self._enable_pub.publish(msg)
        self._rospy.sleep(0.05)

    def _poll_puppet_feedback(self, *, timeout_s: float) -> Any | None:
        try:
            msg = self._rospy.wait_for_message(
                self._puppet_joint_topic,
                self._JointState,
                timeout=max(0.0, timeout_s),
            )
        except self._rospy.ROSException:
            return None
        with self._lock:
            self._latest_puppet = msg
        return msg

    def get_state(self) -> tuple[np.ndarray, float]:
        self._poll_puppet_feedback(timeout_s=0.02)
        joints = np.zeros(6, dtype=np.float64)
        width = 0.0
        with self._lock:
            msg = self._latest_puppet
        if msg is not None and len(msg.position) >= 6:
            joints = np.asarray(msg.position[:6], dtype=np.float64).reshape(6)
            if len(msg.position) >= 7:
                width = float(msg.position[6])
        return joints, width

    def move(
        self,
        *,
        joints: np.ndarray | None = None,
        gripper_width: float | None = None,
    ) -> None:
        if joints is not None:
            self._cmd_joints = clip_piper_joint_targets(
                joints, margin=self.joint_limit_margin
            )
        if gripper_width is not None:
            self._cmd_width = float(
                np.clip(gripper_width, 0.0, self.gripper_max_width)
            )

        msg = self._JointState()
        msg.header.stamp = self._rospy.Time.now()
        msg.position = self._cmd_joints.tolist() + [self._cmd_width]
        self._pub.publish(msg)
        if self.command_sleep_s > 0:
            self._rospy.sleep(self.command_sleep_s)


class AgilexObsAdapter:
    """Convert Agilex client messages into the CaP-X observation schema.

    Expected client message shape:
        {
            "joint_pos": (6,),
            "gripper": float,
            "camera_main": {...},
        }

    Byte keys from msgpack are accepted as well.
    """

    def convert(self, msg: Mapping[Any, Any]) -> dict[str, Any]:
        obs: dict[str, Any] = {}

        joint_values = _lookup(msg, "joint_pos", "joints")
        gripper_value = _lookup(msg, "gripper", "gripper_pos")
        legacy_arm_msg = _lookup(msg, "left")
        if joint_values is None and isinstance(legacy_arm_msg, Mapping):
            joint_values = _lookup(legacy_arm_msg, "joint_pos", "joints")
            gripper_value = _lookup(legacy_arm_msg, "gripper", "gripper_pos")

        joints = (
            np.zeros(6, dtype=np.float32)
            if joint_values is None
            else np.asarray(joint_values, dtype=np.float32).reshape(6)
        )
        gripper = float(0.0 if gripper_value is None else gripper_value)
        obs["robot_joint_pos"] = np.concatenate(
            [joints, np.array([gripper], dtype=np.float32)]
        )

        for camera_name, obs_key in CAMERA_TO_OBS_KEY.items():
            cam_msg = _lookup(msg, *CAMERA_MSG_ALIASES[camera_name])
            obs[obs_key] = _camera_obs_from_msg(cam_msg)

        return obs


def start_msgpack_server_in_background(server: MsgpackNumpyServer):
    loop = asyncio.new_event_loop()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    return loop, thread


class AgilexRealLowLevel(BaseEnv):
    """Low-level scaffold for a real Agilex/Piper single-arm robot.

    The class stores and publishes normalized action dictionaries. If
    ``start_server=True``, actions are exposed through ``MsgpackNumpyServer`` in
    the same style as the existing real Franka adapter. Replace or extend
    ``_publish_action`` if your Agilex SDK uses another transport.
    """

    def __init__(
        self,
        seed: int | None = None,
        privileged: bool = False,
        enable_render: bool = False,
        viser_debug: bool = False,
        control_backend: str = "msgpack",
        camera_backend: str = "msgpack",
        host: str = "0.0.0.0",
        port: int = 9010,
        start_server: bool = True,
        wait_for_observation: bool = True,
        arm_channel: str = "can_left",
        bitrate: int = 1000000,
        robot: str = "piper",
        speed_pct: int = 50,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 0.1,
        max_linear_acc: float = 0.4,
        max_angular_acc: float = 0.05,
        gripper_force: float = 1.0,
        gripper_max_width: float = 0.1,
        command_sleep_s: float = 0.02,
        master_joint_topic: str = AgilexRosArmHardware.DEFAULT_MASTER_TOPIC,
        puppet_joint_topic: str = AgilexRosArmHardware.DEFAULT_PUPPET_TOPIC,
        enable_flag_topic: str = "/enable_flag",
        puppet_wait_timeout: float = 5.0,
        auto_enable_on_connect: bool = True,
        connect_on_init: bool = True,
        img_front_topic: str = DEFAULT_RGB_TOPICS["main"],
        img_front_depth_topic: str = DEFAULT_DEPTH_TOPICS["main"],
        img_front_camera_info_topic: str = DEFAULT_CAMERA_INFO_TOPICS["main"],
        img_side_topic: str | None = None,
        img_side_camera_info_topic: str | None = None,
        side_camera_height_above_table_m: float | None = None,
        side_camera_save_during_task: bool = False,
        side_camera_output_dir: str = "outputs/agilex_side_camera/task",
        use_depth_image: bool = False,
        depth_scale: float = 0.001,
        ros_sync_wait_timeout: float = 0.2,
        camera_intrinsics: Mapping[str, Any] | None = None,
        camera_poses: Mapping[str, Any] | None = None,
        camera_pose_mats: Mapping[str, Any] | None = None,
        agilex_ik_urdf_path: str | None = None,
        agilex_ik_target_link_name: str = "link6",
        base_pose_wxyz_xyz: list[float] | np.ndarray | None = None,
        agilex_tcp_offset: list[float] | np.ndarray | None = None,
        sam3_service_url: str = "http://127.0.0.1:8114",
        sam3_checkpoint_path: str = "/home/agilex/checkpoints/sam3",
        sam3_device: str = "cuda",
        graspnet_service_url: str = "http://127.0.0.1:8115",
        use_graspnet_grasp_pose: bool = False,
        graspnet_frame_offset: list[float] | np.ndarray | None = None,
        graspnet_world_offset: list[float] | np.ndarray | None = None,
        graspnet_rotation_correction_rpy: list[float] | np.ndarray | None = None,
        graspnet_prefer_top_down: bool = False,
        graspnet_approach_axis: str = "z",
        graspnet_top_down_threshold: float = 0.65,
        graspnet_forward_passes: int = 1,
        graspnet_max_retries: int = 7,
        graspnet_local_regions: bool = True,
        graspnet_filter_grasps: bool = False,
        graspnet_fallback_to_topdown: bool = False,
        graspnet_max_full_points: int = 20000,
        graspnet_max_segment_points: int = 10000,
        object_depth_clip_range: list[float] | tuple[float, float] = (0.05, 2.0),
        object_min_points: int = 50,
        object_world_offset: list[float] | np.ndarray | None = None,
        default_grasp_quaternion_wxyz: list[float] | np.ndarray | None = None,
        grasp_z_offset: float = 0.03,
        grasp_ik_z_offsets: list[float] | tuple[float, ...] = (0.0, 0.03, 0.06, 0.1),
        grasp_quaternion_yaw_offsets: list[float] | tuple[float, ...] = (0.0,),
        grasp_quaternion_tilt_offsets_rpy: list[list[float]] | tuple[tuple[float, ...], ...] = ((0.0, 0.0, 0.0),),
        ik_rest_cost_weight: float = 0.0,
        ik_soft_limit_margin_rad: float = 0.0,
        ik_soft_limit_weight: float = 0.0,
        joint_limit_margin: float = 0.005,
        max_single_joint_delta: float = 0.0,
        safe_z_clearance: float = 0.0,
        max_joint_step: float = 0.08,
        reset_on_env_reset: bool = True,
        reset_joint_positions: list[float] | np.ndarray | None = None,
        reset_gripper_width: float = 0.05,
        reset_interpolation_steps: int = 50,
        reset_joint_tolerance: float = 0.03,
        auto_complete_on_clean_run: bool = True,
    ) -> None:
        super().__init__()
        self.privileged = privileged
        self.enable_render = enable_render
        self.viser_debug = viser_debug
        self.wait_for_observation = wait_for_observation
        self.control_backend = control_backend
        self.camera_backend = camera_backend
        self.gripper_max_width = float(gripper_max_width)
        self.agilex_ik_urdf_path = agilex_ik_urdf_path
        self.agilex_ik_target_link_name = agilex_ik_target_link_name
        self.base_pose_wxyz_xyz = base_pose_wxyz_xyz
        self.agilex_tcp_offset = agilex_tcp_offset
        self.sam3_service_url = sam3_service_url
        self.sam3_checkpoint_path = sam3_checkpoint_path
        self.sam3_device = sam3_device
        self.graspnet_service_url = graspnet_service_url
        self.use_graspnet_grasp_pose = bool(use_graspnet_grasp_pose)
        self.graspnet_frame_offset = (
            [0.0, 0.0, 0.0] if graspnet_frame_offset is None else graspnet_frame_offset
        )
        self.graspnet_world_offset = (
            [0.0, 0.0, 0.0] if graspnet_world_offset is None else graspnet_world_offset
        )
        self.graspnet_rotation_correction_rpy = (
            [0.0, 0.0, 0.0]
            if graspnet_rotation_correction_rpy is None
            else graspnet_rotation_correction_rpy
        )
        self.graspnet_prefer_top_down = bool(graspnet_prefer_top_down)
        self.graspnet_approach_axis = graspnet_approach_axis
        self.graspnet_top_down_threshold = float(graspnet_top_down_threshold)
        self.graspnet_forward_passes = int(graspnet_forward_passes)
        self.graspnet_max_retries = int(graspnet_max_retries)
        self.graspnet_local_regions = bool(graspnet_local_regions)
        self.graspnet_filter_grasps = bool(graspnet_filter_grasps)
        self.graspnet_fallback_to_topdown = bool(graspnet_fallback_to_topdown)
        self.graspnet_max_full_points = int(graspnet_max_full_points)
        self.graspnet_max_segment_points = int(graspnet_max_segment_points)
        self.object_depth_clip_range = object_depth_clip_range
        self.object_min_points = object_min_points
        self.object_world_offset = (
            [0.0, 0.0, 0.0] if object_world_offset is None else object_world_offset
        )
        self.default_grasp_quaternion_wxyz = (
            [1.0, 0.0, 0.0, 0.0]
            if default_grasp_quaternion_wxyz is None
            else default_grasp_quaternion_wxyz
        )
        self.grasp_z_offset = grasp_z_offset
        self.grasp_ik_z_offsets = grasp_ik_z_offsets
        self.grasp_quaternion_yaw_offsets = grasp_quaternion_yaw_offsets
        self.grasp_quaternion_tilt_offsets_rpy = grasp_quaternion_tilt_offsets_rpy
        self.ik_rest_cost_weight = float(max(0.0, ik_rest_cost_weight))
        self.ik_soft_limit_margin_rad = float(max(0.0, ik_soft_limit_margin_rad))
        self.ik_soft_limit_weight = float(max(0.0, ik_soft_limit_weight))
        self.joint_limit_margin = float(max(0.0, joint_limit_margin))
        self.max_single_joint_delta = float(max(0.0, max_single_joint_delta))
        self.safe_z_clearance = float(max(0.0, safe_z_clearance))
        self.img_side_topic = img_side_topic
        self.img_side_camera_info_topic = img_side_camera_info_topic
        self.side_camera_height_above_table_m = side_camera_height_above_table_m
        self.side_camera_save_during_task = bool(side_camera_save_during_task)
        self.side_camera_output_dir = side_camera_output_dir
        self.max_joint_step = float(max(0.0, max_joint_step))
        self.reset_on_env_reset = bool(reset_on_env_reset)
        self.reset_joint_positions = (
            np.zeros(6, dtype=np.float64)
            if reset_joint_positions is None
            else np.asarray(reset_joint_positions, dtype=np.float64).reshape(6)
        )
        self.reset_gripper_width = float(
            np.clip(reset_gripper_width, 0.0, self.gripper_max_width)
        )
        self.reset_interpolation_steps = max(1, int(reset_interpolation_steps))
        self.reset_joint_tolerance = float(max(0.0, reset_joint_tolerance))
        self.puppet_wait_timeout = float(max(0.0, puppet_wait_timeout))
        self.auto_complete_on_clean_run = bool(auto_complete_on_clean_run)
        self._joint_lower = PIPER_JOINT_LOWER.copy()
        self._joint_upper = PIPER_JOINT_UPPER.copy()
        self.adapter = AgilexObsAdapter()
        self.hardware: AgilexArmHardware | None = None
        self.ros_camera_adapter: RosCameraAdapter | None = None
        self.low_level_server: MsgpackNumpyServer | None = None
        self._server_loop = None
        self._server_thread = None
        if start_server:
            self.low_level_server = MsgpackNumpyServer(host=host, port=port)
            self._server_loop, self._server_thread = start_msgpack_server_in_background(
                self.low_level_server
            )

        self.obs: dict[str, Any] = {}
        self._current_joints = np.zeros(6, dtype=np.float64)
        self._gripper_fraction = 1.0
        self._action_publish_period = 0.02
        self._target_state_initialized = False
        self.latest_action: dict[str, Any] = self._build_action()
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []

        if control_backend not in {"msgpack", "direct", "ros", "none"}:
            raise ValueError(
                "control_backend must be one of: msgpack, direct, ros, none"
            )
        if camera_backend not in {"msgpack", "ros", "none"}:
            raise ValueError("camera_backend must be one of: msgpack, ros, none")

        if connect_on_init:
            if control_backend == "direct":
                self.hardware = AgilexArmHardware(
                    channel=arm_channel,
                    bitrate=bitrate,
                    robot=robot,
                    speed_pct=speed_pct,
                    max_linear_vel=max_linear_vel,
                    max_angular_vel=max_angular_vel,
                    max_linear_acc=max_linear_acc,
                    max_angular_acc=max_angular_acc,
                    gripper_force=gripper_force,
                    gripper_max_width=gripper_max_width,
                    command_sleep_s=command_sleep_s,
                    joint_limit_margin=self.joint_limit_margin,
                )
            elif control_backend == "ros":
                self.hardware = AgilexRosArmHardware(
                    master_joint_topic=master_joint_topic,
                    puppet_joint_topic=puppet_joint_topic,
                    enable_flag_topic=enable_flag_topic,
                    gripper_max_width=gripper_max_width,
                    command_sleep_s=command_sleep_s,
                    joint_limit_margin=self.joint_limit_margin,
                    puppet_wait_timeout=self.puppet_wait_timeout,
                    auto_enable_on_connect=auto_enable_on_connect,
                )

        if camera_backend == "ros":
            self.ros_camera_adapter = RosCameraAdapter(
                rgb_topics={
                    "main": img_front_topic,
                },
                depth_topics={
                    "main": img_front_depth_topic,
                },
                camera_info_topics={
                    "main": img_front_camera_info_topic,
                },
                use_depth_image=use_depth_image,
                depth_scale=depth_scale,
                sync_wait_timeout=ros_sync_wait_timeout,
                camera_intrinsics=camera_intrinsics,
                camera_poses=camera_poses,
                camera_pose_mats=camera_pose_mats,
            )

    def _update_from_network(self) -> None:
        if self.low_level_server is not None:
            msg = self.low_level_server.latest_observation
            if msg is not None:
                new_obs = self.adapter.convert(msg)
                self._merge_observation(new_obs)
        self._update_from_hardware()
        self._update_from_ros_cameras()

    def _merge_observation(self, new_obs: Mapping[str, Any]) -> None:
        for key, value in new_obs.items():
            if key in CAMERA_TO_OBS_KEY.values():
                if value.get("images", {}).get("rgb") is not None:
                    self.obs[key] = value
                elif key not in self.obs:
                    self.obs[key] = value
            else:
                self.obs[key] = value

    def _update_from_hardware(self) -> None:
        if self.hardware is None:
            return
        joints, width = self.hardware.get_state()
        fraction = self._width_to_fraction(width)
        self.obs["robot_joint_pos"] = np.concatenate(
            [np.asarray(joints, dtype=np.float32), np.array([fraction], dtype=np.float32)]
        )
        self.obs["robot_gripper_width"] = width
        if not self._target_state_initialized:
            self._current_joints = np.asarray(joints, dtype=np.float64).reshape(6)
            self._gripper_fraction = fraction

    def _update_from_ros_cameras(self) -> None:
        if self.ros_camera_adapter is None:
            return
        camera_obs = self.ros_camera_adapter.get_observation()
        if camera_obs:
            self._merge_observation(camera_obs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.hardware is not None:
            self._target_state_initialized = False
            if self.reset_on_env_reset:
                self._reset_hardware_to_home()
        if self.wait_for_observation:
            while True:
                self._update_from_network()
                if self._observation_ready():
                    break
                detail = "; ".join(self._observation_missing_reasons())
                print(f"Waiting for observation from Agilex real environment... {detail}")
                time.sleep(1.0)
        else:
            self._update_from_network()
        return self.obs, {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if isinstance(action, Mapping):
            self.latest_action = dict(action)
            self._publish_action(self.latest_action)
        self._update_from_network()
        return self.obs, self.compute_reward(), self.task_completed(), False, {}

    def get_observation(self) -> dict[str, Any]:
        self._update_from_network()
        return self.obs

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        gripper: float | None = None,
        tolerance: float = 0.05,
        max_steps: int = 350,
    ) -> None:
        target_joints = self._clip_joint_targets(joints)
        start_joints = self._get_observed_joints_or_current()
        if gripper is not None:
            self._gripper_fraction = float(np.clip(gripper, 0.0, 1.0))
        self._target_state_initialized = True

        waypoints = self._joint_waypoints(start_joints, target_joints)
        for waypoint_idx, waypoint in enumerate(waypoints):
            self._current_joints = waypoint
            self._publish_action(self._build_action())
            if self._record_frames:
                self._record_frame()
            if waypoint_idx < len(waypoints) - 1:
                time.sleep(self._action_publish_period)

        if max_steps > 0:
            self._wait_for_joint_convergence(tolerance=tolerance, max_steps=max_steps)

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))
        self._target_state_initialized = True

    def _step_once(self) -> None:
        self._publish_action(self._build_action())
        if self._record_frames:
            self._record_frame()

    def _reset_hardware_to_home(self) -> None:
        if self.hardware is None:
            return
        start_joints, start_width = self.hardware.get_state()
        start_joints_arr = self._clip_joint_targets(
            np.asarray(start_joints, dtype=np.float64).reshape(6),
            for_reset=True,
        )
        start_width_f = float(np.clip(start_width, 0.0, self.gripper_max_width))
        target_joints = self._clip_joint_targets(
            self.reset_joint_positions.copy(), for_reset=True
        )
        target_width = self.reset_gripper_width

        joint_delta = float(np.max(np.abs(target_joints - start_joints_arr)))
        width_delta = abs(target_width - start_width_f)
        if (
            joint_delta <= self.reset_joint_tolerance
            and width_delta <= self.gripper_max_width * 0.05
        ):
            self._current_joints = target_joints.copy()
            self._gripper_fraction = self._width_to_fraction(target_width)
            self._target_state_initialized = True
            self.latest_action = self._build_action()
            if self.low_level_server is not None:
                self.low_level_server.latest_action = self.latest_action
            self.obs["robot_joint_pos"] = np.concatenate(
                [
                    target_joints.astype(np.float32),
                    np.array([self._width_to_fraction(target_width)], dtype=np.float32),
                ]
            )
            self.obs["robot_gripper_width"] = target_width
            return

        for idx in range(1, self.reset_interpolation_steps + 1):
            alpha = idx / self.reset_interpolation_steps
            joints = self._clip_joint_targets(
                start_joints_arr + (target_joints - start_joints_arr) * alpha,
                for_reset=True,
            )
            width = start_width_f + (target_width - start_width_f) * alpha
            self._current_joints = joints
            self._gripper_fraction = self._width_to_fraction(width)
            self._target_state_initialized = True
            self.latest_action = self._build_action()
            if self.low_level_server is not None:
                self.low_level_server.latest_action = self.latest_action
            self.hardware.move(joints=joints, gripper_width=width)

        self.obs["robot_joint_pos"] = np.concatenate(
            [
                target_joints.astype(np.float32),
                np.array([self._width_to_fraction(target_width)], dtype=np.float32),
            ]
        )
        self.obs["robot_gripper_width"] = target_width

    def _publish_action(self, command: dict[str, Any]) -> None:
        """Publish one normalized command.

        Commands use normalized gripper fractions: 0.0 closed, 1.0 fully open.
        The direct pyAgxArm backend converts those fractions to gripper width.
        """
        self._sync_target_from_command(command)
        self.latest_action = self._build_action()
        if self.low_level_server is not None:
            self.low_level_server.latest_action = self.latest_action
        if self.hardware is not None:
            self.hardware.move(
                joints=self._current_joints,
                gripper_width=self._fraction_to_width(self._gripper_fraction),
            )

    def _sync_target_from_command(self, command: Mapping[str, Any]) -> None:
        joints = _lookup(command, "joint_pos", "joints")
        gripper = _lookup(command, "gripper", "gripper_fraction")
        legacy_cmd = _lookup(command, "left")
        if joints is None and isinstance(legacy_cmd, Mapping):
            joints = _lookup(legacy_cmd, "joint_pos", "joints")
            gripper = _lookup(legacy_cmd, "gripper", "gripper_fraction")
        if joints is not None:
            self._current_joints = self._clip_joint_targets(joints)
        if gripper is not None:
            self._gripper_fraction = float(np.clip(float(gripper), 0.0, 1.0))
        self._target_state_initialized = True

    def _build_action(self) -> dict[str, Any]:
        return {
            "timestamp": time.time(),
            "joint_pos": self._current_joints.astype(np.float32).tolist(),
            "gripper": float(self._gripper_fraction),
        }

    def _clip_joint_targets(
        self, joints: np.ndarray, *, for_reset: bool = False
    ) -> np.ndarray:
        margin = 0.0 if for_reset else self.joint_limit_margin
        return clip_piper_joint_targets(joints, margin=margin)

    def _get_observed_joints_or_current(self) -> np.ndarray:
        observed = self.obs.get("robot_joint_pos")
        if observed is not None:
            joints = np.asarray(observed[:6], dtype=np.float64).reshape(6)
            if np.all(np.isfinite(joints)):
                return self._clip_joint_targets(joints)
        return self._current_joints.copy()

    def _joint_waypoints(self, start: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
        target_arr = self._clip_joint_targets(target)
        start_arr = self._clip_joint_targets(start)
        if self.max_joint_step <= 0:
            return [target_arr]
        max_delta = float(np.max(np.abs(target_arr - start_arr)))
        steps = max(1, int(np.ceil(max_delta / self.max_joint_step)))
        return [
            start_arr + (target_arr - start_arr) * (idx / steps)
            for idx in range(1, steps + 1)
        ]

    def _wait_for_joint_convergence(
        self, *, tolerance: float, max_steps: int
    ) -> None:
        last_publish = 0.0
        last_error: float | None = None
        use_rospy_sleep = isinstance(self.hardware, AgilexRosArmHardware)
        for step_idx in range(max_steps):
            self._update_from_network()
            observed = self.obs.get("robot_joint_pos")
            if observed is not None:
                current = np.asarray(observed[:6], dtype=np.float64)
                last_error = float(np.linalg.norm(current - self._current_joints))
                if last_error < tolerance:
                    break
            now = time.time()
            if now - last_publish >= self._action_publish_period:
                self._publish_action(self._build_action())
                last_publish = now
            if self._record_frames and step_idx % 5 == 0:
                self._record_frame()
            if use_rospy_sleep:
                self.hardware._rospy.sleep(0.01)
            else:
                time.sleep(0.01)
        else:
            if last_error is not None:
                target = self._current_joints.astype(np.float64).reshape(6)
                current = (
                    np.asarray(observed[:6], dtype=np.float64).reshape(6)
                    if observed is not None
                    else np.full(6, np.nan)
                )
                raise TimeoutError(
                    f"Agilex arm did not converge within {max_steps} steps "
                    f"(last joint error={last_error:.4f}, tolerance={tolerance:.4f}). "
                    f"current={np.array2string(current, precision=3)} "
                    f"target={np.array2string(target, precision=3)}. "
                    "If current stays near zero while target changes, check: "
                    "(1) only one `bash scripts/start_piper_left.sh` instance, "
                    "(2) Piper shows `使能状态: True`, "
                    "(3) `rostopic echo /puppet/joint_left` updates when you publish "
                    "to /master/joint_left."
                )

    def _fraction_to_width(self, fraction: Any) -> float | None:
        if fraction is None:
            return None
        return float(np.clip(float(fraction), 0.0, 1.0) * self.gripper_max_width)

    def _width_to_fraction(self, width: Any) -> float:
        if self.gripper_max_width <= 0:
            return 0.0
        return float(np.clip(float(width) / self.gripper_max_width, 0.0, 1.0))

    def _observation_ready(self) -> bool:
        if (
            self.control_backend in {"direct", "ros"}
            and "robot_joint_pos" not in self.obs
        ):
            return False
        if self.camera_backend in {"msgpack", "ros"}:
            main_rgb = self.obs.get("robot0_robotview", {}).get("images", {}).get("rgb")
            return main_rgb is not None
        return True

    def _observation_missing_reasons(self) -> list[str]:
        missing: list[str] = []
        if (
            self.control_backend in {"direct", "ros"}
            and "robot_joint_pos" not in self.obs
        ):
            missing.append("missing robot_joint_pos from arm state")
        if self.camera_backend in {"msgpack", "ros"}:
            main_rgb = self.obs.get("robot0_robotview", {}).get("images", {}).get("rgb")
            if main_rgb is None:
                missing.append("missing main RGB in obs['robot0_robotview']")
        if self.ros_camera_adapter is not None:
            status = self.ros_camera_adapter.debug_status()
            missing.append(f"ros_status={status}")
        return missing or ["observation is not ready"]

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def _record_frame(self) -> None:
        rgbs = obs_get_rgb(self.obs)
        if not rgbs:
            return
        self._frame_buffer.append(resize_with_pad(list(rgbs.values())[0], 480, 640))

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        rgb = self.obs.get("robot0_robotview", {}).get("images", {}).get("rgb")
        if rgb is not None:
            return rgb
        return np.zeros((480, 640, 3), dtype=np.uint8)


__all__ = [
    "AgilexArmHardware",
    "AgilexObsAdapter",
    "AgilexRealLowLevel",
    "AgilexRosArmHardware",
    "RosCameraAdapter",
]
