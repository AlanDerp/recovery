#!/usr/bin/env python3
"""Visualize SAM3 object point cloud and Contact-GraspNet grasps for Agilex."""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import plotly.graph_objects as go
import requests
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation


DEFAULT_CONFIG = Path("env_configs/agilex/agilex_real_pink_cup_lift_single_turn.yaml")


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str)
    with io.BytesIO(data) as f:
        return np.load(f)


def _encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    PILImage.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_mask(mask_b64: str, shape: list[int] | tuple[int, ...]) -> np.ndarray:
    raw = base64.b64decode(mask_b64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).astype(bool)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _load_low_level_config(config_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    low_level = dict(OmegaConf.to_container(cfg.env.cfg.low_level, resolve=True))
    low_level.pop("_target_", None)
    return low_level


def _normalize_rgb(image: np.ndarray, encoding: str | None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if encoding is not None and encoding.lower() == "bgr8":
        arr = arr[:, :, ::-1]
    return np.asarray(arr, dtype=np.uint8)


def _normalize_depth(image: np.ndarray, depth_scale: float) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) * float(depth_scale)
    else:
        arr = arr.astype(np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _wait_for_rgbd(low_level: dict[str, Any], timeout_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError("rospy, cv_bridge, and sensor_msgs are required") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("capx_agilex_sam3_graspnet_viz", anonymous=True, disable_signals=True)

    rgb_topic = low_level.get("img_front_topic", "/camera_h/color/image_raw")
    depth_topic = low_level.get("img_front_depth_topic", "/camera_h/depth/image_raw")
    camera_info_topic = low_level.get("img_front_camera_info_topic", "/camera_h/color/camera_info")
    bridge = CvBridge()

    print(f"[ros] waiting RGB: {rgb_topic}")
    rgb_msg = rospy.wait_for_message(rgb_topic, Image, timeout=timeout_s)
    print(f"[ros] waiting depth: {depth_topic}")
    depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=timeout_s)
    print(f"[ros] waiting camera_info: {camera_info_topic}")
    info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=timeout_s)

    rgb = _normalize_rgb(
        bridge.imgmsg_to_cv2(rgb_msg, "passthrough"),
        getattr(rgb_msg, "encoding", None),
    )
    depth = _normalize_depth(
        bridge.imgmsg_to_cv2(depth_msg, "passthrough"),
        float(low_level.get("depth_scale", 0.001)),
    )
    intrinsics = np.asarray(info_msg.K, dtype=np.float64).reshape(3, 3)
    if depth.shape[:2] != rgb.shape[:2]:
        from capx.utils.camera_utils import scale_intrinsics_for_image

        intrinsics = scale_intrinsics_for_image(
            intrinsics,
            depth.shape[:2],
            (int(info_msg.height), int(info_msg.width)),
        )
        print(
            f"[intrinsics] scaled K for depth {depth.shape[:2]} "
            f"from camera_info {(info_msg.height, info_msg.width)}"
        )
    print(f"[ros] RGB shape={rgb.shape}, depth shape={depth.shape}")
    return rgb, depth, intrinsics


def _service_ready(url: str) -> bool:
    session = requests.Session()
    session.trust_env = False
    try:
        resp = session.get(f"{url.rstrip('/')}/openapi.json", timeout=1.0)
        if resp.status_code == 200:
            return True
        resp = session.get(f"{url.rstrip('/')}/docs", timeout=1.0)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _start_server(
    *,
    module: str,
    url: str,
    output_dir: Path,
    device: str,
    checkpoint_path: str | None = None,
    startup_timeout_s: float = 240.0,
) -> subprocess.Popen[str] | None:
    if _service_ready(url):
        print(f"[server] already ready: {url}")
        return None

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        raise ValueError(f"URL must include a port: {url}")

    log_path = output_dir / f"{module.rsplit('.', 1)[-1]}.log"
    log_file = log_path.open("w")
    cmd = [
        sys.executable,
        "-m",
        module,
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        device,
    ]
    if checkpoint_path is not None:
        cmd.extend(["--checkpoint-path", checkpoint_path])

    print("[server] starting:", " ".join(cmd))
    print(f"[server] log: {log_path}")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if _service_ready(url):
            print(f"[server] ready: {url}")
            return proc
        if proc.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            raise RuntimeError(f"Server exited before ready: {module}\n{tail}")
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {module} at {url}")


def _run_sam3(rgb: np.ndarray, prompt: str, sam3_url: str) -> dict[str, Any]:
    if not _service_ready(sam3_url):
        raise RuntimeError(f"SAM3 service is not reachable at {sam3_url}")

    payload = {"image_base64": _encode_png(rgb), "text_prompt": prompt}
    print(f"[sam3] segment prompt={prompt!r}")
    session = requests.Session()
    session.trust_env = False
    resp = session.post(f"{sam3_url.rstrip('/')}/segment", json=payload, timeout=180.0)
    resp.raise_for_status()

    results: list[dict[str, Any]] = []
    for item in resp.json().get("results", []):
        results.append(
            {
                "mask": _decode_mask(item["mask_base64"], item["shape"]),
                "box": item.get("box"),
                "score": float(item.get("score", 0.0)),
                "label": item.get("label", prompt),
            }
        )
    if not results:
        raise RuntimeError(f"SAM3 returned no mask for {prompt!r}")
    best = max(results, key=lambda item: item["score"])
    print(
        f"[sam3] best score={best['score']:.4f}, "
        f"mask_pixels={int(np.count_nonzero(best['mask']))}"
    )
    return best


def _resize_mask_to_depth(mask: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == depth_shape:
        return mask.astype(bool)
    nearest = getattr(getattr(PILImage, "Resampling", PILImage), "NEAREST")
    resized = PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
        (depth_shape[1], depth_shape[0]),
        resample=nearest,
    )
    return np.asarray(resized, dtype=np.uint8) > 0


def _points_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
    *,
    mask: np.ndarray | None,
    depth_range: tuple[float, float],
) -> np.ndarray:
    near, far = depth_range
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    if mask is not None:
        valid &= _resize_mask_to_depth(mask, depth.shape)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    z = depth[ys, xs]
    x = (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_cam = np.stack([x, y, z], axis=1)
    points_h = np.concatenate(
        [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_world = (pose_mat @ points_h.T).T[:, :3]
    return points_world[np.isfinite(points_world).all(axis=1)]


def _subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
    return points[idx]


def _call_graspnet(
    *,
    graspnet_url: str,
    pc_full: np.ndarray,
    pc_segment: np.ndarray,
    low_level: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not _service_ready(graspnet_url):
        raise RuntimeError(f"Contact-GraspNet service is not reachable at {graspnet_url}")

    payload = {
        "pc_full_base64": _numpy_to_base64(pc_full.astype(np.float32)),
        "pc_segment_base64": _numpy_to_base64(pc_segment.astype(np.float32)),
        "segmap_id": 1,
        "local_regions": bool(low_level.get("graspnet_local_regions", True)),
        "filter_grasps": bool(low_level.get("graspnet_filter_grasps", False)),
        "forward_passes": int(low_level.get("graspnet_forward_passes", 1)),
        "max_retries": int(low_level.get("graspnet_max_retries", 7)),
    }
    print(f"[graspnet] request pc_full={pc_full.shape}, pc_segment={pc_segment.shape}")
    session = requests.Session()
    session.trust_env = False
    resp = session.post(
        f"{graspnet_url.rstrip('/')}/plan_point_clouds",
        json=payload,
        timeout=300.0,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Contact-GraspNet failed: {resp.status_code} {resp.text}") from exc
    data = resp.json()
    grasps = _base64_to_numpy(data["grasps_base64"])
    scores = _base64_to_numpy(data["scores_base64"]).reshape(-1)
    contact_pts = _base64_to_numpy(data["contact_pts_base64"])
    print(f"[graspnet] returned grasps={len(grasps)}, scores={scores.shape}")
    return grasps, scores, contact_pts


def _apply_config_grasp_transform(grasp: np.ndarray, low_level: dict[str, Any]) -> np.ndarray:
    offset = np.asarray(low_level.get("graspnet_frame_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    world_offset = np.asarray(low_level.get("graspnet_world_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    rpy = np.asarray(low_level.get("graspnet_rotation_correction_rpy", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    transformed = np.asarray(grasp, dtype=np.float64).reshape(4, 4).copy()
    transformed[:3, 3] = transformed[:3, 3] + transformed[:3, :3] @ offset + world_offset
    transformed[:3, :3] = transformed[:3, :3] @ Rotation.from_euler("xyz", rpy).as_matrix()
    return transformed


def _save_2d_outputs(output_dir: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(rgb).save(output_dir / "rgb.png")
    PILImage.fromarray(mask.astype(np.uint8) * 255).save(output_dir / "sam3_mask.png")
    overlay = rgb.copy()
    if mask.shape != rgb.shape[:2]:
        mask = _resize_mask_to_depth(mask, rgb.shape[:2])
    color = np.array([255, 80, 160], dtype=np.uint8)
    overlay[mask] = (overlay[mask].astype(np.float32) * 0.45 + color * 0.55).astype(np.uint8)
    PILImage.fromarray(overlay).save(output_dir / "sam3_overlay.png")


def _add_line(
    fig: go.Figure,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    width: int,
    name: str,
    showlegend: bool = False,
) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line={"color": color, "width": width},
            name=name,
            showlegend=showlegend,
        )
    )


def _add_frame(fig: go.Figure, pose: np.ndarray, name: str, axis_len: float, width: int = 5) -> None:
    origin = pose[:3, 3]
    axes = [
        ("x", "#d62728", pose[:3, 0]),
        ("y", "#2ca02c", pose[:3, 1]),
        ("z", "#1f77b4", pose[:3, 2]),
    ]
    for suffix, color, direction in axes:
        _add_line(
            fig,
            origin,
            origin + direction * axis_len,
            color=color,
            width=width,
            name=f"{name}_{suffix}",
        )
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode="markers",
            marker={"size": 3, "color": "#111111"},
            name=name,
            hovertext=[name],
            showlegend=False,
        )
    )


def _build_figure(
    *,
    scene_points: np.ndarray,
    object_points: np.ndarray,
    grasps: np.ndarray,
    scores: np.ndarray,
    contact_pts: np.ndarray,
    pose_mat: np.ndarray,
    low_level: dict[str, Any],
    top_k: int,
) -> go.Figure:
    fig = go.Figure()
    scene_show = _subsample(scene_points, 12000, seed=1)
    obj_show = _subsample(object_points, 8000, seed=2)
    fig.add_trace(
        go.Scatter3d(
            x=scene_show[:, 0],
            y=scene_show[:, 1],
            z=scene_show[:, 2],
            mode="markers",
            marker={"size": 1.2, "color": "rgba(120,120,120,0.25)"},
            name="scene points",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=obj_show[:, 0],
            y=obj_show[:, 1],
            z=obj_show[:, 2],
            mode="markers",
            marker={"size": 2.2, "color": "#ff4fa3"},
            name="SAM3 object points",
        )
    )

    if len(contact_pts) > 0:
        pts = np.asarray(contact_pts, dtype=np.float64).reshape(-1, 3)
        pts = pts[: min(top_k, len(pts))]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 4, "color": "#000000"},
                name="GraspNet contact pts",
            )
        )

    order = np.argsort(-scores) if len(scores) == len(grasps) else np.arange(len(grasps))
    for rank, idx_np in enumerate(order[:top_k]):
        idx = int(idx_np)
        raw = np.asarray(grasps[idx], dtype=np.float64).reshape(4, 4)
        transformed = _apply_config_grasp_transform(raw, low_level)
        score = float(scores[idx]) if len(scores) > idx else float("nan")
        _add_frame(fig, raw, f"raw_grasp_{rank}_score_{score:.3f}", axis_len=0.045, width=5)
        if not np.allclose(raw, transformed):
            _add_frame(fig, transformed, f"configured_grasp_{rank}_score_{score:.3f}", axis_len=0.06, width=8)

    _add_frame(fig, np.eye(4), "base", axis_len=0.12, width=7)
    _add_frame(fig, pose_mat, "camera_main", axis_len=0.10, width=7)

    fig.update_layout(
        title="SAM3 object point cloud + Contact-GraspNet grasps",
        scene={
            "xaxis": {"title": "base X (m)"},
            "yaxis": {"title": "base Y (m)"},
            "zaxis": {"title": "base Z (m)"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
        legend={"itemsizing": "constant"},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", default="pink cup")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/agilex_sam3_graspnet_viz"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--start-servers", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    args = parser.parse_args()

    low_level = _load_low_level_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sam3_url = low_level.get("sam3_service_url", "http://127.0.0.1:8114")
    graspnet_url = low_level.get("graspnet_service_url", "http://127.0.0.1:8115")
    server_procs: list[subprocess.Popen[str]] = []
    try:
        if args.start_servers:
            proc = _start_server(
                module="capx.serving.launch_sam3_server",
                url=sam3_url,
                output_dir=args.output_dir,
                device=args.device,
                checkpoint_path=low_level.get("sam3_checkpoint_path", "/home/agilex/checkpoints/sam3"),
                startup_timeout_s=args.startup_timeout,
            )
            if proc is not None:
                server_procs.append(proc)
            proc = _start_server(
                module="capx.serving.launch_contact_graspnet_server",
                url=graspnet_url,
                output_dir=args.output_dir,
                device=args.device,
                startup_timeout_s=args.startup_timeout,
            )
            if proc is not None:
                server_procs.append(proc)

        rgb, depth, intrinsics = _wait_for_rgbd(low_level, args.timeout)
        detection = _run_sam3(rgb, args.prompt, sam3_url)
        mask = np.asarray(detection["mask"], dtype=bool)
        _save_2d_outputs(args.output_dir, rgb, mask)

        pose_mat = np.asarray(low_level["camera_pose_mats"]["main"], dtype=np.float64).reshape(4, 4)
        depth_range_values = low_level.get("object_depth_clip_range", [0.05, 2.0])
        depth_range = (float(depth_range_values[0]), float(depth_range_values[1]))
        scene_points = _points_from_depth(depth, intrinsics, pose_mat, mask=None, depth_range=depth_range)
        object_points = _points_from_depth(depth, intrinsics, pose_mat, mask=mask, depth_range=depth_range)
        if len(object_points) < int(low_level.get("object_min_points", 50)):
            raise RuntimeError(
                f"SAM3 mask has only {len(object_points)} valid depth points; "
                f"need at least {low_level.get('object_min_points', 50)}"
            )

        pc_full = _subsample(scene_points, int(low_level.get("graspnet_max_full_points", 20000)), seed=3)
        pc_segment = _subsample(object_points, int(low_level.get("graspnet_max_segment_points", 10000)), seed=4)
        grasps, scores, contact_pts = _call_graspnet(
            graspnet_url=graspnet_url,
            pc_full=pc_full,
            pc_segment=pc_segment,
            low_level=low_level,
        )

        npz_path = args.output_dir / "sam3_graspnet_outputs.npz"
        np.savez(
            npz_path,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            pose_mat=pose_mat,
            mask=mask,
            scene_points=scene_points,
            object_points=object_points,
            pc_full=pc_full,
            pc_segment=pc_segment,
            grasps=grasps,
            scores=scores,
            contact_pts=contact_pts,
        )
        summary = {
            "prompt": args.prompt,
            "sam3_score": detection["score"],
            "sam3_box": detection.get("box"),
            "mask_pixels": int(np.count_nonzero(mask)),
            "scene_points": len(scene_points),
            "object_points": len(object_points),
            "pc_full": len(pc_full),
            "pc_segment": len(pc_segment),
            "num_grasps": int(len(grasps)),
            "top_scores": np.asarray(scores[: min(10, len(scores))]),
            "camera_pose_mat": pose_mat,
            "graspnet_frame_offset": low_level.get("graspnet_frame_offset", [0.0, 0.0, 0.0]),
            "graspnet_world_offset": low_level.get("graspnet_world_offset", [0.0, 0.0, 0.0]),
            "graspnet_rotation_correction_rpy": low_level.get(
                "graspnet_rotation_correction_rpy",
                [0.0, 0.0, 0.0],
            ),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(_jsonify(summary), indent=2) + "\n")

        html_path = args.output_dir / "sam3_graspnet_3d.html"
        fig = _build_figure(
            scene_points=scene_points,
            object_points=object_points,
            grasps=grasps,
            scores=np.asarray(scores, dtype=np.float64).reshape(-1),
            contact_pts=contact_pts,
            pose_mat=pose_mat,
            low_level=low_level,
            top_k=args.top_k,
        )
        fig.write_html(html_path, include_plotlyjs=True, full_html=True)

        print(f"[output] saved overlay: {args.output_dir / 'sam3_overlay.png'}")
        print(f"[output] saved arrays: {npz_path}")
        print(f"[output] saved summary: {args.output_dir / 'summary.json'}")
        print(f"[output] saved 3D HTML: {html_path}")
    finally:
        for proc in server_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
