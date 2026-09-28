import numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from threading import Lock

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel


def rotation_error(a_xyzw, b_xyzw):
    return (
        Rotation.from_quat(a_xyzw).inv() * Rotation.from_quat(b_xyzw)
    ).magnitude()


def test_gripper_tool_z_quaternion_roundtrip():
    gripper = Rotation.from_euler("xyz", [0.3, -0.4, 1.2]).as_quat()
    tool = R1LiteRealLowLevel.gripper_link_to_tool_z_quaternion(gripper)
    recovered = R1LiteRealLowLevel.tool_z_to_gripper_link_quaternion(tool)

    assert rotation_error(gripper, recovered) < 1e-12


def test_gripper_to_tool_z_is_urdf_positive_y_quarter_turn():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    actual = R1LiteRealLowLevel.gripper_link_to_tool_z_quaternion(identity)
    expected = Rotation.from_euler("y", np.pi / 2.0).as_quat()

    assert rotation_error(actual, expected) < 1e-12


def test_quaternion_sign_does_not_change_tcp_conversion():
    gripper = Rotation.from_euler("xyz", [-0.2, 0.7, -1.1]).as_quat()
    tool_a = R1LiteRealLowLevel.gripper_link_to_tool_z_quaternion(gripper)
    tool_b = R1LiteRealLowLevel.gripper_link_to_tool_z_quaternion(-gripper)

    assert rotation_error(tool_a, tool_b) < 1e-12


def test_move_joints_safe_converts_feedback_delta_to_moca_command_delta():
    r1 = object.__new__(R1LiteRealLowLevel)
    measured = np.array([0.2, 0.3, -0.1, 0.0, 0.4, -0.2])
    active_command = np.array([0.1, 0.12, -0.08, -0.03, 0.35, -0.25])
    desired = measured.copy()
    desired[5] += 0.005
    feedback_rows = [measured.copy(), desired.copy()]
    published = []

    r1._lock = Lock()
    r1._last_joint_command = active_command.copy()
    r1._last_action = {}
    r1._ensure_moca_for_joint_target = lambda _target: None
    r1._enable_arm_joint_control = lambda: None
    r1.get_arm_joints = lambda: feedback_rows.pop(0) if feedback_rows else desired
    r1._make_joint_state = lambda _name, position: SimpleNamespace(
        position=position
    )
    r1._arm_pub = SimpleNamespace(
        publish=lambda message: published.append(np.asarray(message.position))
    )
    r1.arm_name = "right_arm"
    r1.max_joint_step = 0.03
    r1.publish_rate_hz = 1000.0
    r1.command_duration_s = 0.0
    r1.arm_min_motion_rad = 0.005
    r1.arm_divergence_limit_rad = 0.03
    r1.arm_joint_tolerance_rad = 0.025

    result = r1.move_joints_safe(desired, duration_s=0.0)

    expected_command = active_command + (desired - measured)
    assert np.allclose(published[-1], expected_command)
    assert np.allclose(r1._last_joint_command, expected_command)
    assert np.allclose(result, desired)
