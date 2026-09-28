#!/usr/bin/env python3
"""Publish native, full-resolution rectified R1 Lite head camera JPEGs.

Pipeline:
    V4L2 YUYV capture -> OpenCV BGR capture conversion -> NVIDIA VPI CUDA
    remap -> JPEG -> ROS 2 CompressedImage

The node uses the company-provided K/D/R/P matrices from
``/opt/galaxea/body/stereo_ros.yaml`` and publishes the original Galaxea head
topic names so the unmodified data collector can record them.

It never sends robot motion or control commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import struct
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np
import yaml

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import CameraInfo, CompressedImage
    from std_msgs.msg import String
    import vpi
except ImportError as exc:
    raise SystemExit(
        "Required ROS 2 or NVIDIA VPI modules are unavailable. Run:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source /home/nvidia/galaxea/install/setup.bash"
    ) from exc


LEFT_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
RIGHT_TOPIC = "/hdas/camera_head/right_raw/image_raw_color/compressed"
LEFT_INFO_TOPIC = "/calib/head_left/camera_info"
RIGHT_INFO_TOPIC = "/calib/head_right/camera_info"
STATUS_TOPIC = "/jimmy/head_rectifier/status"


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    k: np.ndarray
    d: np.ndarray
    r: np.ndarray
    p: np.ndarray


@dataclass
class WorkerStats:
    captured: int = 0
    published: int = 0
    capture_seconds: float = 0.0
    remap_seconds: float = 0.0
    encode_seconds: float = 0.0
    publish_seconds: float = 0.0
    jpeg_bytes: int = 0
    started_at: float = field(default_factory=time.monotonic)
    error: Optional[str] = None

    def snapshot(self) -> dict:
        elapsed = max(1e-6, time.monotonic() - self.started_at)
        published = max(1, self.published)
        return {
            "captured": self.captured,
            "published": self.published,
            "fps": self.published / elapsed,
            "capture_ms": 1000.0 * self.capture_seconds / published,
            "remap_ms": 1000.0 * self.remap_seconds / published,
            "encode_ms": 1000.0 * self.encode_seconds / published,
            "publish_ms": 1000.0 * self.publish_seconds / published,
            "jpeg_kib": self.jpeg_bytes / published / 1024.0,
            "error": self.error,
        }


def load_opencv_yaml(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    cleaned = "\n".join(
        line
        for line in lines
        if not line.startswith("%YAML") and line.strip() != "---"
    )
    document = yaml.safe_load(cleaned)
    if not isinstance(document, dict):
        raise ValueError(f"Invalid calibration YAML: {path}")
    return document


def load_calibration(path: Path, side: str) -> Calibration:
    document = load_opencv_yaml(path)
    if side not in document:
        raise ValueError(f"Calibration has no '{side}' section: {path}")
    data = document[side]
    return Calibration(
        width=int(data["width"]),
        height=int(data["height"]),
        k=np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        d=np.asarray(data["D"], dtype=np.float64),
        r=np.asarray(data["R"], dtype=np.float64).reshape(3, 3),
        p=np.asarray(data["P"], dtype=np.float64).reshape(3, 4),
    )


def scaled_projection(
    calibration: Calibration,
    output_size: tuple[int, int],
) -> np.ndarray:
    output_width, output_height = output_size
    scale = np.diag(
        (
            output_width / float(calibration.width),
            output_height / float(calibration.height),
            1.0,
        )
    )
    return scale @ calibration.p


def create_vpi_warp(
    calibration: Calibration,
    output_size: tuple[int, int],
) -> tuple[vpi.WarpMap, np.ndarray]:
    projection = scaled_projection(calibration, output_size)
    map_x, map_y = cv2.initUndistortRectifyMap(
        calibration.k,
        calibration.d,
        calibration.r,
        projection,
        output_size,
        cv2.CV_32FC1,
    )
    warp = vpi.WarpMap(output_size)
    warp_array = np.asarray(warp)
    warp_array[:, :, 0] = map_x
    warp_array[:, :, 1] = map_y
    return warp, projection


def make_rectified_camera_info(
    calibration: Calibration,
    projection: np.ndarray,
    output_size: tuple[int, int],
) -> CameraInfo:
    width, height = output_size
    info = CameraInfo()
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.k = [
        float(projection[0, 0]),
        float(projection[0, 1]),
        float(projection[0, 2]),
        0.0,
        float(projection[1, 1]),
        float(projection[1, 2]),
        0.0,
        0.0,
        1.0,
    ]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [float(value) for value in projection.reshape(-1)]
    # The pixels have already been rectified by this node. Consumers must not
    # apply another remap.
    info.roi.do_rectify = False
    return info


def vendor_camera_running() -> list[str]:
    matches = []
    own_pid = os.getpid()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == own_pid:
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "signal_camera_node" in command:
            matches.append(f"{process.name}: {command.strip()}")
    return matches


class PythonImagePublisher:
    def __init__(self, node: Node, publisher):
        self.node = node
        self.publisher = publisher

    def publish(self, jpeg: bytes, timestamp_ns: int) -> None:
        message = CompressedImage()
        message.header.stamp.sec = timestamp_ns // 1_000_000_000
        message.header.stamp.nanosec = timestamp_ns % 1_000_000_000
        message.format = "jpeg"
        message.data = jpeg
        self.publisher.publish(message)

    def close(self) -> None:
        return


class CppImagePublisher:
    HEADER = struct.Struct("<QI")

    def __init__(self, helper: Path, topic: str, node_name: str):
        self.process = subprocess.Popen(
            [str(helper), topic, node_name],
            stdin=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        if self.process.stdin is None:
            raise RuntimeError("C++ publisher helper has no stdin pipe")
        self.fd = self.process.stdin.fileno()
        self.lock = threading.Lock()

    @staticmethod
    def write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        sent = 0
        while sent < len(view):
            written = os.write(fd, view[sent:])
            if written <= 0:
                raise BrokenPipeError("C++ publisher pipe closed")
            sent += written

    def publish(self, jpeg: bytes, timestamp_ns: int) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"C++ publisher exited with code {self.process.returncode}"
            )
        header = self.HEADER.pack(timestamp_ns, len(jpeg))
        with self.lock:
            self.write_all(self.fd, header)
            self.write_all(self.fd, jpeg)

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


class CameraWorker:
    def __init__(
        self,
        node: Node,
        side: str,
        device: str,
        publisher,
        warp: vpi.WarpMap,
        output_size: tuple[int, int],
        capture_size: tuple[int, int],
        jpeg_quality: int,
        warmup_frames: int,
        stop_event: threading.Event,
    ):
        self.node = node
        self.side = side
        self.device = device
        self.publisher = publisher
        self.warp = warp
        self.output_size = output_size
        self.capture_size = capture_size
        self.jpeg_quality = jpeg_quality
        self.warmup_frames = warmup_frames
        self.stop_event = stop_event
        self.stats = WorkerStats()
        self.stats_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self.run,
            name=f"head-{side}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout)

    def snapshot(self) -> dict:
        with self.stats_lock:
            return self.stats.snapshot()

    def fail(self, text: str) -> None:
        with self.stats_lock:
            self.stats.error = text
        self.node.get_logger().error(f"{self.side}: {text}")
        self.stop_event.set()

    def run(self) -> None:
        capture_width, capture_height = self.capture_size
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        if not cap.isOpened():
            self.fail(f"could not open {self.device}")
            return

        actual_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual_size != self.capture_size:
            cap.release()
            self.fail(
                f"capture size is {actual_size}, expected {self.capture_size}"
            )
            return

        try:
            for _ in range(self.warmup_frames):
                if self.stop_event.is_set():
                    return
                cap.read()

            self.node.get_logger().info(
                f"{self.side}: capturing {self.device} at "
                f"{capture_width}x{capture_height}"
            )
            while not self.stop_event.is_set():
                start_capture = time.perf_counter()
                ok, image = cap.read()
                capture_timestamp_ns = time.time_ns()
                capture_elapsed = time.perf_counter() - start_capture
                if not ok or image is None:
                    self.fail(f"failed to capture from {self.device}")
                    break

                start_remap = time.perf_counter()
                with vpi.Backend.CUDA:
                    remapped = vpi.asimage(image).remap(
                        self.warp,
                        size=self.output_size,
                        interp=vpi.Interp.LINEAR,
                    )
                corrected = remapped.cpu()
                remap_elapsed = time.perf_counter() - start_remap

                start_encode = time.perf_counter()
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    corrected,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                encode_elapsed = time.perf_counter() - start_encode
                if not encoded_ok:
                    self.fail("JPEG encoding failed")
                    break

                start_publish = time.perf_counter()
                jpeg = encoded.tobytes()
                self.publisher.publish(jpeg, capture_timestamp_ns)
                publish_elapsed = time.perf_counter() - start_publish

                with self.stats_lock:
                    self.stats.captured += 1
                    self.stats.published += 1
                    self.stats.capture_seconds += capture_elapsed
                    self.stats.remap_seconds += remap_elapsed
                    self.stats.encode_seconds += encode_elapsed
                    self.stats.publish_seconds += publish_elapsed
                    self.stats.jpeg_bytes += len(jpeg)
        except Exception as exc:
            self.fail(f"{type(exc).__name__}: {exc}")
        finally:
            cap.release()


class NativeHeadRectifier(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args.node_name)
        self.args = args
        self.stop_event = threading.Event()
        cv2.setNumThreads(1)

        conflicts = vendor_camera_running()
        if conflicts and not args.allow_vendor_conflict:
            details = "\n".join(conflicts)
            raise RuntimeError(
                "The vendor signal_camera_node is already running and owns the "
                f"head devices. Stop it before starting this node:\n{details}"
            )

        output_size = (args.output_width, args.output_height)
        capture_size = (args.capture_width, args.capture_height)
        left_calibration = load_calibration(args.calibration, "left")
        right_calibration = load_calibration(args.calibration, "right")
        left_warp, left_projection = create_vpi_warp(
            left_calibration, output_size
        )
        right_warp, right_projection = create_vpi_warp(
            right_calibration, output_size
        )

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_publishers = {}
        if args.side in {"both", "left"}:
            self.image_publishers["left"] = self.make_image_publisher(
                "left", args.left_topic, qos
            )
        if args.side in {"both", "right"}:
            self.image_publishers["right"] = self.make_image_publisher(
                "right", args.right_topic, qos
            )
        self.left_info_publisher = self.create_publisher(
            CameraInfo, args.left_info_topic, qos
        )
        self.right_info_publisher = self.create_publisher(
            CameraInfo, args.right_info_topic, qos
        )
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, qos)

        self.left_info = make_rectified_camera_info(
            left_calibration, left_projection, output_size
        )
        self.right_info = make_rectified_camera_info(
            right_calibration, right_projection, output_size
        )

        self.workers: dict[str, CameraWorker] = {}
        if args.side in {"both", "left"}:
            self.workers["left"] = CameraWorker(
                self,
                "left",
                args.left_device,
                self.image_publishers["left"],
                left_warp,
                output_size,
                capture_size,
                args.jpeg_quality,
                args.warmup_frames,
                self.stop_event,
            )
        if args.side in {"both", "right"}:
            self.workers["right"] = CameraWorker(
                self,
                "right",
                args.right_device,
                self.image_publishers["right"],
                right_warp,
                output_size,
                capture_size,
                args.jpeg_quality,
                args.warmup_frames,
                self.stop_event,
            )

        self.create_timer(1.0, self.publish_camera_info)
        self.create_timer(5.0, self.publish_status)
        self.get_logger().info(
            "official calibration: "
            f"{args.calibration} -> {output_size[0]}x{output_size[1]}, "
            f"JPEG quality {args.jpeg_quality}"
        )
        for worker in self.workers.values():
            worker.start()

    def make_image_publisher(self, side: str, topic: str, qos):
        if self.args.publisher_backend == "cpp":
            return CppImagePublisher(
                helper=self.args.publisher_helper,
                topic=topic,
                node_name=f"{self.args.node_name}_{side}_publisher",
            )
        publisher = self.create_publisher(CompressedImage, topic, qos)
        return PythonImagePublisher(self, publisher)

    def publish_camera_info(self) -> None:
        stamp = self.get_clock().now().to_msg()
        if "left" in self.workers:
            self.left_info.header.stamp = stamp
            self.left_info_publisher.publish(self.left_info)
        if "right" in self.workers:
            self.right_info.header.stamp = stamp
            self.right_info_publisher.publish(self.right_info)

    def publish_status(self) -> None:
        status = {
            "distortion_fixed": True,
            "backend": "NVIDIA VPI CUDA",
            "calibration": str(self.args.calibration),
            "output_size": [self.args.output_width, self.args.output_height],
            "jpeg_quality": self.args.jpeg_quality,
            "side": self.args.side,
        }
        for side, worker in self.workers.items():
            status[side] = worker.snapshot()
        message = String()
        message.data = json.dumps(status, separators=(",", ":"), sort_keys=True)
        self.status_publisher.publish(message)
        summary = []
        for side in ("left", "right"):
            if side not in status:
                continue
            item = status[side]
            summary.append(
                "%s %.2f FPS [capture %.1f, remap %.1f, encode %.1f, "
                "publish %.1f ms]"
                % (
                    side,
                    item["fps"],
                    item["capture_ms"],
                    item["remap_ms"],
                    item["encode_ms"],
                    item["publish_ms"],
                )
            )
        self.get_logger().info("; ".join(summary))

    def stop(self) -> None:
        self.stop_event.set()
        for worker in self.workers.values():
            worker.join()
        for publisher in self.image_publishers.values():
            publisher.close()

    def destroy_node(self):
        self.stop()
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("/opt/galaxea/body/stereo_ros.yaml"),
    )
    parser.add_argument("--left-device", default="/dev/video0")
    parser.add_argument("--right-device", default="/dev/video1")
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1536)
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--output-height", type=int, default=1536)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
        help="run both cameras or one camera per process",
    )
    parser.add_argument(
        "--node-name",
        default="jimmy_native_head_rectifier",
    )
    parser.add_argument("--left-topic", default=LEFT_TOPIC)
    parser.add_argument("--right-topic", default=RIGHT_TOPIC)
    parser.add_argument("--left-info-topic", default=LEFT_INFO_TOPIC)
    parser.add_argument("--right-info-topic", default=RIGHT_INFO_TOPIC)
    parser.add_argument(
        "--publisher-backend",
        choices=("cpp", "python"),
        default="cpp",
        help="C++ avoids Python serialization overhead; python is diagnostic",
    )
    parser.add_argument(
        "--publisher-helper",
        type=Path,
        default=Path(
            "/home/nvidia/jimmy/native_head_publisher/build/"
            "jimmy_compressed_publisher"
        ),
    )
    parser.add_argument(
        "--allow-vendor-conflict",
        action="store_true",
        help="unsafe diagnostic option; normally leave disabled",
    )
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if min(
        args.capture_width,
        args.capture_height,
        args.output_width,
        args.output_height,
    ) <= 0:
        parser.error("capture and output dimensions must be positive")
    if args.publisher_backend == "cpp" and not args.publisher_helper.is_file():
        parser.error(
            f"C++ publisher helper is missing: {args.publisher_helper}; "
            "run /home/nvidia/jimmy/build_native_head_publisher.sh"
        )
    return args


def main() -> int:
    args = parse_args()
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[NativeHeadRectifier] = None
    try:
        node = NativeHeadRectifier(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.stop()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
