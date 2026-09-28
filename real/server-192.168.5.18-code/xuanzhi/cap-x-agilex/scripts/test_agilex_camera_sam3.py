#!/usr/bin/env python3
"""Smoke-test an Agilex ROS RGB camera topic and SAM3 segmentation."""

from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image as PILImage


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3_URL = "http://127.0.0.1:8114"


def _encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    PILImage.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_mask(mask_b64: str, shape: list[int] | tuple[int, ...]) -> np.ndarray:
    raw = base64.b64decode(mask_b64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).astype(bool)


def _ros_image_to_rgb(msg: Any) -> np.ndarray:
    try:
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise RuntimeError("cv_bridge is required to read ROS Image messages") from exc

    bridge = CvBridge()
    arr = bridge.imgmsg_to_cv2(msg, "passthrough")
    arr = np.asarray(arr)
    encoding = getattr(msg, "encoding", "").lower()

    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported image shape from ROS: {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return np.asarray(arr, dtype=np.uint8)


def wait_for_ros_image(topic: str, timeout_s: float) -> tuple[np.ndarray, Any]:
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/capx_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    try:
        import rospy
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError("rospy and sensor_msgs are required for this test") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_camera_sam3_test", anonymous=True, disable_signals=True)

    print(f"[camera] waiting for one frame on {topic} timeout={timeout_s}s")
    msg = rospy.wait_for_message(topic, Image, timeout=timeout_s)
    rgb = _ros_image_to_rgb(msg)
    print(
        "[camera] received frame "
        f"shape={rgb.shape} encoding={getattr(msg, 'encoding', 'unknown')} "
        f"stamp={msg.header.stamp.to_sec():.6f}"
    )
    return rgb, msg


def _sam3_server_ready(sam3_url: str) -> bool:
    session = requests.Session()
    session.trust_env = False
    try:
        resp = session.get(f"{sam3_url.rstrip('/')}/openapi.json", timeout=1.0)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def start_sam3_server(
    *,
    sam3_url: str,
    checkpoint_path: str,
    device: str,
    startup_timeout_s: float,
    use_uv: bool,
) -> subprocess.Popen[str]:
    from urllib.parse import urlparse

    parsed = urlparse(sam3_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8114
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env["SAM3_CHECKPOINT_PATH"] = checkpoint_path
    vendor_sam3 = str(REPO_ROOT / "capx" / "third_party" / "sam3")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), vendor_sam3, env.get("PYTHONPATH", "")]
    )

    python_cmd = [sys.executable]
    if use_uv:
        python_cmd = ["uv", "run", "--no-sync", "python"]

    cmd = [
        *python_cmd,
        "-m",
        "capx.serving.launch_sam3_server",
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        device,
        "--checkpoint-path",
        checkpoint_path,
    ]
    print("[sam3] starting server:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout is not None else ""
            raise RuntimeError(f"SAM3 server exited early with code {proc.returncode}\n{output}")
        if _sam3_server_ready(sam3_url):
            print(f"[sam3] server ready at {sam3_url}")
            return proc
        time.sleep(1.0)

    raise TimeoutError(f"SAM3 server did not become ready within {startup_timeout_s}s")


def run_sam3(image_rgb: np.ndarray, *, prompt: str, sam3_url: str) -> list[dict[str, Any]]:
    payload = {
        "image_base64": _encode_png(image_rgb),
        "text_prompt": prompt,
    }
    print(f"[sam3] segment prompt={prompt!r} url={sam3_url}")
    session = requests.Session()
    session.trust_env = False
    resp = session.post(f"{sam3_url.rstrip('/')}/segment", json=payload, timeout=120.0)
    resp.raise_for_status()

    results: list[dict[str, Any]] = []
    for item in resp.json().get("results", []):
        mask = _decode_mask(item["mask_base64"], item["shape"])
        results.append(
            {
                "mask": mask,
                "box": item["box"],
                "score": float(item["score"]),
                "label": item.get("label", prompt),
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"[sam3] got {len(results)} result(s)")
    return results


def save_outputs(
    *,
    image_rgb: np.ndarray,
    results: list[dict[str, Any]],
    output_dir: Path,
    prompt: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "camera_frame.png"
    PILImage.fromarray(image_rgb).save(image_path)
    print(f"[output] saved frame: {image_path}")

    if not results:
        return

    best = results[0]
    mask = best["mask"]
    np.save(output_dir / "sam3_best_mask.npy", mask)

    mask_img = (mask.astype(np.uint8) * 255)
    PILImage.fromarray(mask_img).save(output_dir / "sam3_best_mask.png")

    overlay = image_rgb.copy()
    color = np.array([30, 144, 255], dtype=np.uint8)
    overlay[mask] = (overlay[mask].astype(np.float32) * 0.45 + color * 0.55).astype(np.uint8)
    PILImage.fromarray(overlay).save(output_dir / "sam3_overlay.png")

    print(
        "[output] saved SAM3 outputs: "
        f"score={best['score']:.4f} box={np.round(best['box'], 1).tolist()} "
        f"prompt={prompt!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera_h/color/image_raw")
    parser.add_argument("--prompt", default="cloth")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output-dir", default="outputs/agilex_camera_sam3_test")
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--sam3-url", default=DEFAULT_SAM3_URL)
    parser.add_argument("--checkpoint-path", default="/home/agilex/checkpoints/sam3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-sam3-server", action="store_true")
    parser.add_argument(
        "--sam3-server-launcher",
        choices=("uv", "python"),
        default="uv",
        help="Use uv run --no-sync for the SAM3 server by default.",
    )
    parser.add_argument("--sam3-startup-timeout", type=float, default=180.0)
    args = parser.parse_args()

    server_proc: subprocess.Popen[str] | None = None
    try:
        image_rgb, _ = wait_for_ros_image(args.topic, args.timeout)

        results: list[dict[str, Any]] = []
        if not args.skip_sam3:
            if not _sam3_server_ready(args.sam3_url):
                if not args.start_sam3_server:
                    raise RuntimeError(
                        f"SAM3 server is not reachable at {args.sam3_url}. "
                        "Start it first or rerun with --start-sam3-server."
                    )
                server_proc = start_sam3_server(
                    sam3_url=args.sam3_url,
                    checkpoint_path=args.checkpoint_path,
                    device=args.device,
                    startup_timeout_s=args.sam3_startup_timeout,
                    use_uv=args.sam3_server_launcher == "uv",
                )
            results = run_sam3(image_rgb, prompt=args.prompt, sam3_url=args.sam3_url)

        save_outputs(
            image_rgb=image_rgb,
            results=results,
            output_dir=Path(args.output_dir),
            prompt=args.prompt,
        )
        print("[done] camera/SAM3 smoke test finished")
    finally:
        if server_proc is not None:
            print("[sam3] stopping server started by this script")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
