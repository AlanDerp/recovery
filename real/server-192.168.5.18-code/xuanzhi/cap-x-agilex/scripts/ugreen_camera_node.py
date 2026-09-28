#!/usr/bin/env python3
"""Publish UGREEN Camera 2K frames from a V4L2 device to ROS."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image


def main() -> int:
    rospy.init_node("ugreen_camera", anonymous=False)
    device = rospy.get_param("~device", "/dev/video2")
    image_topic = rospy.get_param("~image_topic", "/camera_ugreen/color/image_raw")
    info_topic = rospy.get_param("~camera_info_topic", "/camera_ugreen/color/camera_info")
    fps = float(rospy.get_param("~fps", 30.0))
    frame_id = rospy.get_param("~frame_id", "camera_ugreen_color_optical_frame")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        rospy.logerr("Failed to open UGREEN camera device: %s", device)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        rospy.logerr("Invalid camera resolution from device %s", device)
        return 1

    image_pub = rospy.Publisher(image_topic, Image, queue_size=1)
    info_pub = rospy.Publisher(info_topic, CameraInfo, queue_size=1)
    rate = rospy.Rate(fps if fps > 0 else 30.0)

    camera_info = CameraInfo()
    camera_info.width = width
    camera_info.height = height
    camera_info.distortion_model = "plumb_bob"
    camera_info.K = [float(width), 0.0, width / 2.0, 0.0, float(height), height / 2.0, 0.0, 0.0, 1.0]
    camera_info.P = [float(width), 0.0, width / 2.0, 0.0, 0.0, float(height), height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    rospy.loginfo(
        "UGREEN camera publishing %dx%d from %s to %s",
        width,
        height,
        device,
        image_topic,
    )

    while not rospy.is_shutdown():
        ok, frame = cap.read()
        if not ok:
            rospy.logwarn_throttle(5.0, "Failed to read frame from %s", device)
            rate.sleep()
            continue

        stamp = rospy.Time.now()
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        camera_info.header.stamp = stamp
        camera_info.header.frame_id = frame_id
        image_pub.publish(msg)
        info_pub.publish(camera_info)
        rate.sleep()

    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
