#!/usr/bin/env python3
"""Run the vendor homogeneous controller without owning real gripper topics.

Only the two gripper publisher names are remapped. Arm, torso, chassis,
parameter, and data-collection interfaces retain their vendor names.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package = FindPackageShare("teleoperation_ros2")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "whole_body_urdf",
                default_value=PathJoinSubstitution(
                    [package, "config/urdf/r1_lite_whole_body.urdf"]
                ),
            ),
            DeclareLaunchArgument("gripper_mode", default_value="mode1"),
            DeclareLaunchArgument("default_chassis_vel", default_value="0.2"),
            Node(
                package="teleoperation_ros2",
                executable="homogeneous_controller_node",
                name="homogeneous_teleoperation",
                output="screen",
                parameters=[
                    {"robot_variant": "r1litet"},
                    {"whole_body_urdf": LaunchConfiguration("whole_body_urdf")},
                    {"gripper_mode": LaunchConfiguration("gripper_mode")},
                    {"default_chassis_vel": LaunchConfiguration("default_chassis_vel")},
                ],
                remappings=[
                    (
                        "/motion_target/target_position_gripper_left",
                        "/jimmy/disabled_homogeneous/gripper_left",
                    ),
                    (
                        "/motion_target/target_position_gripper_right",
                        "/jimmy/disabled_homogeneous/gripper_right",
                    ),
                ],
            ),
        ]
    )

