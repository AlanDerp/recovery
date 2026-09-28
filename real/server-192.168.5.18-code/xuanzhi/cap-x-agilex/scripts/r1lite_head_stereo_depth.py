#!/usr/bin/env python3
"""Rectify R1 Lite head cameras and optionally compute left-aligned depth."""

import argparse
from collections import deque
import math
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


LEFT_IMAGE_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
RIGHT_IMAGE_TOPIC = "/hdas/camera_head/right_raw/image_raw_color/compressed"
LEFT_INFO_TOPIC = "/calib/head_left/camera_info"
RIGHT_INFO_TOPIC = "/calib/head_right/camera_info"
DEFAULT_IGEV_ROOT = (
    Path(__file__).resolve().parents[1]
    / "capx"
    / "third_party"
    / "IGEV"
    / "IGEV-Stereo"
)
DEFAULT_IGEV_CHECKPOINT = (
    DEFAULT_IGEV_ROOT / "pretrained_models" / "middlebury" / "middlebury.pth"
)

# Calibrated online from 1,305 inlier head-camera feature correspondences on
# 2026-07-18. OpenCV stereoRectify expects camera1(left) -> camera2(right).
DEFAULT_LEFT_TO_RIGHT = (-0.0649659127, 0.0007217971, -0.0019771703)
DEFAULT_LEFT_TO_RIGHT_ROTATION = (
    0.9999213396,
    0.0121167089,
    0.0032403765,
    -0.0121361831,
    0.9999079948,
    0.0060592902,
    -0.0031666597,
    -0.0060981393,
    0.9999763922,
)


@dataclass
class Frame:
    msg: CompressedImage
    seq: int
    received_at: float


def stamp_seconds(msg) -> float:
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def scaled_k(info: CameraInfo, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    k = np.array(info.k, dtype=np.float64).reshape(3, 3).copy()
    if info.width and info.height and (info.width != width or info.height != height):
        sx = width / float(info.width)
        sy = height / float(info.height)
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
    return k


def camera_info_from_rect(
    template: CameraInfo,
    header,
    size: Tuple[int, int],
    p: np.ndarray,
) -> CameraInfo:
    width, height = size
    out = CameraInfo()
    out.header = header
    out.height = height
    out.width = width
    out.distortion_model = "plumb_bob"
    out.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    out.k = [
        float(p[0, 0]),
        float(p[0, 1]),
        float(p[0, 2]),
        0.0,
        float(p[1, 1]),
        float(p[1, 2]),
        0.0,
        0.0,
        1.0,
    ]
    out.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    out.p = [float(x) for x in p.reshape(-1)]
    out.binning_x = template.binning_x
    out.binning_y = template.binning_y
    return out


def compressed_image_from_cv(
    image: np.ndarray,
    header,
    jpeg_quality: int,
) -> CompressedImage:
    """Encode a BGR image using the same JPEG transport used by raw cameras."""
    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not encoded_ok:
        raise RuntimeError("OpenCV failed to JPEG-encode a rectified image")
    out = CompressedImage()
    out.header = header
    out.format = "jpeg"
    out.data = encoded.tobytes()
    return out


class IGEVMatcher:
    """Load IGEV-Stereo lazily and infer dense left-image disparity."""

    def __init__(self, args: argparse.Namespace):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "IGEV-Stereo requires PyTorch; run this node with "
                ".capx_r1lite_venv/bin/python"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("IGEV-Stereo requires the Jetson CUDA PyTorch build")

        root = Path(args.igev_root).expanduser().resolve()
        checkpoint = Path(args.igev_checkpoint).expanduser().resolve()
        if not (root / "core" / "igev_stereo.py").is_file():
            raise RuntimeError(f"IGEV-Stereo source is missing: {root}")
        if not checkpoint.is_file():
            raise RuntimeError(f"IGEV-Stereo checkpoint is missing: {checkpoint}")

        # The installed torchvision wheel lacks its optional NMS operator.
        # timm does not use NMS for MobileNetV2, but torchvision registers a
        # fake implementation at import time and expects the schema to exist.
        self._torchvision_library = None
        try:
            torch.ops.torchvision.nms
        except (AttributeError, RuntimeError):
            self._torchvision_library = torch.library.Library("torchvision", "DEF")
            self._torchvision_library.define(
                "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
            )

        for path in (str(root / "core"), str(root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from igev_stereo import IGEVStereo
        from utils.utils import InputPadder

        model_args = SimpleNamespace(
            hidden_dims=[128, 128, 128],
            corr_levels=2,
            corr_radius=4,
            n_downsample=2,
            n_gru_layers=3,
            max_disp=args.igev_max_disparity,
            mixed_precision=args.igev_mixed_precision,
            precision_dtype=args.igev_precision,
        )
        model = IGEVStereo(model_args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
        model.load_state_dict(state, strict=True)
        self.model = model.cuda().eval()
        self.torch = torch
        self.InputPadder = InputPadder
        self.valid_iters = args.igev_valid_iters

    def compute(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch

        def tensor_from_bgr(image: np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .cuda(non_blocking=True)
            )

        left = tensor_from_bgr(left_bgr)
        right = tensor_from_bgr(right_bgr)
        padder = self.InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        with torch.inference_mode():
            disparity = self.model(
                left,
                right,
                iters=self.valid_iters,
                test_mode=True,
            )
        disparity = padder.unpad(disparity)
        return disparity.squeeze().float().cpu().numpy()


class HeadStereoDepth(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("r1lite_head_stereo_depth")
        self.args = args
        self.bridge = CvBridge()
        self.left_info: Optional[CameraInfo] = None
        self.right_info: Optional[CameraInfo] = None
        self.left_frame: Optional[Frame] = None
        self.right_frame: Optional[Frame] = None
        self.left_frames = deque(maxlen=8)
        self.right_frames = deque(maxlen=8)
        self.left_seq = 0
        self.right_seq = 0
        self.last_pair = (-1, -1)
        self.last_left_rect_seq = -1
        self.last_right_rect_seq = -1
        self.last_log = 0.0
        self.rect_ready = False
        self.rect_image_size: Optional[Tuple[int, int]] = None
        self.map_lx = self.map_ly = None
        self.map_rx = self.map_ry = None
        self.p1 = self.p2 = None
        self.depth_info: Optional[CameraInfo] = None
        self.left_rect_info: Optional[CameraInfo] = None
        self.right_rect_info: Optional[CameraInfo] = None
        self.baseline_m = float(
            np.linalg.norm(np.array(args.left_to_right, dtype=np.float64))
        )

        sub_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        pub_qos = QoSProfile(depth=5)

        self.create_subscription(CompressedImage, args.left_image, self.left_cb, sub_qos)
        self.create_subscription(CompressedImage, args.right_image, self.right_cb, sub_qos)
        self.create_subscription(CameraInfo, args.left_info, self.left_info_cb, sub_qos)
        self.create_subscription(CameraInfo, args.right_info, self.right_info_cb, sub_qos)

        self.left_rect_pub = None
        self.right_rect_pub = None
        if not args.compressed_only:
            self.left_rect_pub = self.create_publisher(
                Image, f"{args.output_ns}/left/image_rect", pub_qos
            )
            self.right_rect_pub = self.create_publisher(
                Image, f"{args.output_ns}/right/image_rect", pub_qos
            )

        self.left_rect_compressed_pub = self.create_publisher(
            CompressedImage,
            f"{args.output_ns}/left/image_rect/compressed",
            pub_qos,
        )
        self.right_rect_compressed_pub = self.create_publisher(
            CompressedImage,
            f"{args.output_ns}/right/image_rect/compressed",
            pub_qos,
        )
        self.left_rect_info_pub = self.create_publisher(
            CameraInfo, f"{args.output_ns}/left/camera_info", pub_qos
        )
        self.right_rect_info_pub = self.create_publisher(
            CameraInfo, f"{args.output_ns}/right/camera_info", pub_qos
        )

        self.disp_pub = None
        self.depth_pub = None
        self.depth_info_pub = None
        self.matcher = None
        self.right_matcher = None
        self.wls_filter = None
        self.igev_matcher = None
        if not args.rectify_only:
            self.disp_pub = self.create_publisher(
                Image, f"{args.output_ns}/disparity/image_raw", pub_qos
            )
            self.depth_pub = self.create_publisher(
                Image,
                f"{args.output_ns}/aligned_depth_to_left/image_raw",
                pub_qos,
            )
            self.depth_info_pub = self.create_publisher(
                CameraInfo,
                f"{args.output_ns}/aligned_depth_to_left/camera_info",
                pub_qos,
            )

            if args.stereo_backend == "igev":
                self.igev_matcher = IGEVMatcher(args)
            else:
                num_disp = int(math.ceil(args.num_disparities / 16.0) * 16)
                block = (
                    args.block_size
                    if args.block_size % 2 == 1
                    else args.block_size + 1
                )
                channels = 1
                self.matcher = cv2.StereoSGBM_create(
                    minDisparity=args.min_disparity,
                    numDisparities=num_disp,
                    blockSize=block,
                    P1=8 * channels * block * block,
                    P2=32 * channels * block * block,
                    disp12MaxDiff=1,
                    uniquenessRatio=args.uniqueness_ratio,
                    speckleWindowSize=args.speckle_window_size,
                    speckleRange=args.speckle_range,
                    preFilterCap=31,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                )
                self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher)
                self.wls_filter.setLambda(args.wls_lambda)
                self.wls_filter.setSigmaColor(args.wls_sigma_color)

        self.create_timer(1.0 / args.rate, self.process_latest)
        self.get_logger().info(f"left image: {args.left_image}")
        self.get_logger().info(f"right image: {args.right_image}")
        self.get_logger().info(f"left CameraInfo: {args.left_info}")
        self.get_logger().info(f"right CameraInfo: {args.right_info}")
        mode = (
            "rectified JPEG only"
            if args.rectify_only
            else f"rectified JPEG and {args.stereo_backend} stereo depth"
        )
        self.get_logger().info(f"publishing {mode} under {args.output_ns}")

    def left_info_cb(self, msg: CameraInfo):
        self.left_info = msg

    def right_info_cb(self, msg: CameraInfo):
        self.right_info = msg

    def left_cb(self, msg: CompressedImage):
        self.left_seq += 1
        frame = Frame(msg=msg, seq=self.left_seq, received_at=time.monotonic())
        self.left_frame = frame
        self.left_frames.append(frame)

    def right_cb(self, msg: CompressedImage):
        self.right_seq += 1
        frame = Frame(msg=msg, seq=self.right_seq, received_at=time.monotonic())
        self.right_frame = frame
        self.right_frames.append(frame)

    def decode_image(self, msg: CompressedImage, side: str) -> Optional[np.ndarray]:
        try:
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"failed to decode {side} compressed image: {exc}")
            return None

    def init_rectification(self, image_size: Tuple[int, int]) -> bool:
        if self.left_info is None or self.right_info is None:
            return False

        k1 = scaled_k(self.left_info, image_size)
        k2 = scaled_k(self.right_info, image_size)
        d1 = np.array(self.left_info.d, dtype=np.float64).reshape(-1, 1)
        d2 = np.array(self.right_info.d, dtype=np.float64).reshape(-1, 1)
        r = np.array(
            self.args.left_to_right_rotation,
            dtype=np.float64,
        ).reshape(3, 3)
        t = np.array(self.args.left_to_right, dtype=np.float64).reshape(3, 1)

        r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            image_size,
            r,
            t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=self.args.alpha,
        )
        self.map_lx, self.map_ly = cv2.initUndistortRectifyMap(k1, d1, r1, p1, image_size, cv2.CV_16SC2)
        self.map_rx, self.map_ry = cv2.initUndistortRectifyMap(k2, d2, r2, p2, image_size, cv2.CV_16SC2)
        self.p1 = p1
        self.p2 = p2
        p_baseline = abs(float(p2[0, 3]) / float(p2[0, 0])) if p2[0, 0] else 0.0
        if p_baseline > 1e-6:
            self.baseline_m = p_baseline
        self.depth_info = camera_info_from_rect(self.left_info, self.left_info.header, image_size, p1)
        self.left_rect_info = camera_info_from_rect(
            self.left_info, self.left_info.header, image_size, p1
        )
        self.right_rect_info = camera_info_from_rect(
            self.right_info, self.right_info.header, image_size, p2
        )
        self.rect_image_size = image_size
        self.rect_ready = True
        self.get_logger().info(
            "rectification ready: size=%dx%d fx=%.3f baseline=%.6fm"
            % (image_size[0], image_size[1], p1[0, 0], self.baseline_m)
        )
        return True

    def synchronized(self, left: Frame, right: Frame) -> bool:
        return self.frame_skew_seconds(left, right) <= self.args.max_sync_ms * 1e-3

    @staticmethod
    def frame_skew_seconds(left: Frame, right: Frame) -> float:
        ls = stamp_seconds(left.msg)
        rs = stamp_seconds(right.msg)
        if ls > 0.0 and rs > 0.0:
            return abs(ls - rs)
        return abs(left.received_at - right.received_at)

    def nearest_synchronized_pair(self) -> Optional[Tuple[Frame, Frame]]:
        if not self.left_frames or not self.right_frames:
            return None
        candidates = [
            (self.frame_skew_seconds(left, right), left, right)
            for left in self.left_frames
            for right in self.right_frames
        ]
        skew_s, left, right = min(candidates, key=lambda item: item[0])
        if skew_s > self.args.max_sync_ms * 1e-3:
            return None
        while self.left_frames and self.left_frames[0].seq <= left.seq:
            self.left_frames.popleft()
        while self.right_frames and self.right_frames[0].seq <= right.seq:
            self.right_frames.popleft()
        return left, right

    def process_latest(self):
        if self.args.rectify_only:
            self.process_rectify_only()
            return

        if not self.left_frames or not self.right_frames:
            self.log_waiting("waiting for left/right compressed images")
            return
        pair = self.nearest_synchronized_pair()
        if pair is None:
            self.log_waiting("waiting for synchronized left/right frames")
            return
        left, right = pair
        if (left.seq, right.seq) == self.last_pair:
            return

        left_decoded = self.decode_image(left.msg, "left")
        right_decoded = self.decode_image(right.msg, "right")
        if left_decoded is None or right_decoded is None:
            return

        h, w = left_decoded.shape[:2]
        if right_decoded.shape[:2] != (h, w):
            self.get_logger().warn(
                "left/right image sizes differ: left=%s right=%s"
                % (left_decoded.shape[:2], right_decoded.shape[:2])
            )
            return

        if self.args.processing_width > 0 and w > self.args.processing_width:
            scale = self.args.processing_width / float(w)
            processing_size = (
                self.args.processing_width,
                max(1, int(round(h * scale))),
            )
            left_image = cv2.resize(left_decoded, processing_size, interpolation=cv2.INTER_AREA)
            right_image = cv2.resize(right_decoded, processing_size, interpolation=cv2.INTER_AREA)
            w, h = processing_size
        else:
            left_image = left_decoded
            right_image = right_decoded

        image_size = (w, h)
        if not self.rect_ready and not self.init_rectification(image_size):
            self.log_waiting("waiting for left/right CameraInfo")
            return

        left_rect = cv2.remap(left_image, self.map_lx, self.map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_image, self.map_rx, self.map_ry, cv2.INTER_LINEAR)

        left_header = left.msg.header
        right_header = right.msg.header
        if not left_header.frame_id:
            left_header.frame_id = self.args.frame_id
        if not right_header.frame_id:
            right_header.frame_id = "camera_head_right"

        try:
            left_compressed = compressed_image_from_cv(
                left_rect, left_header, self.args.jpeg_quality
            )
            right_compressed = compressed_image_from_cv(
                right_rect, right_header, self.args.jpeg_quality
            )
        except RuntimeError as exc:
            self.get_logger().warn(str(exc))
            return

        self.left_rect_compressed_pub.publish(left_compressed)
        self.right_rect_compressed_pub.publish(right_compressed)

        if self.left_rect_pub is not None and self.right_rect_pub is not None:
            left_msg = self.bridge.cv2_to_imgmsg(left_rect, encoding="bgr8")
            left_msg.header = left_header
            right_msg = self.bridge.cv2_to_imgmsg(right_rect, encoding="bgr8")
            right_msg.header = right_header
            self.left_rect_pub.publish(left_msg)
            self.right_rect_pub.publish(right_msg)

        if self.left_rect_info is not None:
            self.left_rect_info.header = left_header
            self.left_rect_info_pub.publish(self.left_rect_info)
        if self.right_rect_info is not None:
            self.right_rect_info.header = right_header
            self.right_rect_info_pub.publish(self.right_rect_info)

        if self.args.rectify_only:
            self.last_pair = (left.seq, right.seq)
            return

        if self.args.stereo_backend == "igev":
            disparity = self.igev_matcher.compute(left_rect, right_rect)
            confidence_valid = np.ones(disparity.shape, dtype=bool)
        else:
            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
            left_raw = self.matcher.compute(left_gray, right_gray)
            right_raw = self.right_matcher.compute(right_gray, left_gray)
            filtered_raw = self.wls_filter.filter(
                left_raw,
                left_gray,
                disparity_map_right=right_raw,
                right_view=right_gray,
            )
            confidence = self.wls_filter.getConfidenceMap()
            disparity = filtered_raw.astype(np.float32) / 16.0
            confidence_valid = confidence >= float(self.args.min_wls_confidence)
        min_valid = float(self.args.min_disparity) + 0.5
        valid = (
            np.isfinite(disparity)
            & (disparity > min_valid)
            & confidence_valid
        )
        depth = np.zeros(disparity.shape, dtype=np.float32)
        fx = float(self.p1[0, 0])
        depth[valid] = fx * self.baseline_m / disparity[valid]
        depth[(depth < self.args.min_depth) | (depth > self.args.max_depth)] = 0.0

        disp_msg = self.bridge.cv2_to_imgmsg(disparity, encoding="32FC1")
        disp_msg.header = left_header
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header = left_header

        self.disp_pub.publish(disp_msg)
        self.depth_pub.publish(depth_msg)
        if self.depth_info is not None:
            self.depth_info.header = left_header
            self.depth_info_pub.publish(self.depth_info)

        self.last_pair = (left.seq, right.seq)

    def process_rectify_only(self):
        """Rectify each head camera independently without stereo synchronization."""
        pending = (
            ("left", self.left_frame, self.last_left_rect_seq),
            ("right", self.right_frame, self.last_right_rect_seq),
        )
        if all(frame is None or frame.seq == last_seq for _, frame, last_seq in pending):
            self.log_waiting("waiting for new head camera frames")
            return

        for side, frame, last_seq in pending:
            if frame is None or frame.seq == last_seq:
                continue

            decoded = self.decode_image(frame.msg, side)
            if decoded is None:
                continue

            h, w = decoded.shape[:2]
            if self.args.processing_width > 0 and w > self.args.processing_width:
                scale = self.args.processing_width / float(w)
                processing_size = (
                    self.args.processing_width,
                    max(1, int(round(h * scale))),
                )
                image = cv2.resize(decoded, processing_size, interpolation=cv2.INTER_AREA)
                w, h = processing_size
            else:
                image = decoded

            image_size = (w, h)
            if not self.rect_ready and not self.init_rectification(image_size):
                self.log_waiting("waiting for left/right CameraInfo")
                return
            if image_size != self.rect_image_size:
                self.get_logger().warn(
                    "%s image size changed from %s to %s; skipping frame"
                    % (side, self.rect_image_size, image_size)
                )
                continue

            map_x, map_y = (
                (self.map_lx, self.map_ly)
                if side == "left"
                else (self.map_rx, self.map_ry)
            )
            rectified = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
            header = frame.msg.header
            if not header.frame_id:
                header.frame_id = (
                    self.args.frame_id if side == "left" else "camera_head_right"
                )

            try:
                compressed = compressed_image_from_cv(
                    rectified, header, self.args.jpeg_quality
                )
            except RuntimeError as exc:
                self.get_logger().warn(str(exc))
                continue

            if side == "left":
                self.left_rect_compressed_pub.publish(compressed)
                if self.left_rect_pub is not None:
                    raw = self.bridge.cv2_to_imgmsg(rectified, encoding="bgr8")
                    raw.header = header
                    self.left_rect_pub.publish(raw)
                if self.left_rect_info is not None:
                    self.left_rect_info.header = header
                    self.left_rect_info_pub.publish(self.left_rect_info)
                self.last_left_rect_seq = frame.seq
            else:
                self.right_rect_compressed_pub.publish(compressed)
                if self.right_rect_pub is not None:
                    raw = self.bridge.cv2_to_imgmsg(rectified, encoding="bgr8")
                    raw.header = header
                    self.right_rect_pub.publish(raw)
                if self.right_rect_info is not None:
                    self.right_rect_info.header = header
                    self.right_rect_info_pub.publish(self.right_rect_info)
                self.last_right_rect_seq = frame.seq

    def log_waiting(self, text: str):
        now = time.monotonic()
        if now - self.last_log > 2.0:
            self.get_logger().info(text)
            self.last_log = now


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-image", default=LEFT_IMAGE_TOPIC)
    parser.add_argument("--right-image", default=RIGHT_IMAGE_TOPIC)
    parser.add_argument("--left-info", default=LEFT_INFO_TOPIC)
    parser.add_argument("--right-info", default=RIGHT_INFO_TOPIC)
    parser.add_argument("--output-ns", default="/capx/head_stereo")
    parser.add_argument("--frame-id", default="camera_head_left")
    parser.add_argument("--processing-width", type=int, default=640)
    parser.add_argument(
        "--rectify-only",
        action="store_true",
        help="publish rectified images without computing stereo disparity/depth",
    )
    parser.add_argument(
        "--compressed-only",
        action="store_true",
        help="do not also publish uncompressed rectified Image topics",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        choices=range(1, 101),
        metavar="1..100",
        default=90,
        help="JPEG quality for rectified CompressedImage topics (default: 90)",
    )
    parser.add_argument(
        "--left-to-right",
        nargs=3,
        type=float,
        default=DEFAULT_LEFT_TO_RIGHT,
    )
    parser.add_argument(
        "--left-to-right-rotation",
        nargs=9,
        type=float,
        default=DEFAULT_LEFT_TO_RIGHT_ROTATION,
    )
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--max-sync-ms", type=float, default=25.0)
    parser.add_argument(
        "--stereo-backend",
        choices=("igev", "sgbm"),
        default=os.environ.get("HEAD_STEREO_BACKEND", "igev"),
        help="disparity backend (default: IGEV-Stereo)",
    )
    parser.add_argument("--igev-root", default=str(DEFAULT_IGEV_ROOT))
    parser.add_argument("--igev-checkpoint", default=str(DEFAULT_IGEV_CHECKPOINT))
    parser.add_argument("--igev-valid-iters", type=int, default=12)
    parser.add_argument("--igev-max-disparity", type=int, default=192)
    parser.add_argument(
        "--igev-mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--igev-precision",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=8)
    parser.add_argument("--speckle-window-size", type=int, default=80)
    parser.add_argument("--speckle-range", type=int, default=2)
    parser.add_argument("--wls-lambda", type=float, default=8000.0)
    parser.add_argument("--wls-sigma-color", type=float, default=1.5)
    parser.add_argument("--min-wls-confidence", type=float, default=64.0)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = HeadStereoDepth(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
