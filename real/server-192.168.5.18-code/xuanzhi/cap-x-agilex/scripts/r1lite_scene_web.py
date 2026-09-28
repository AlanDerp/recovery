#!/usr/bin/env python3
"""Read-only web dashboard for the R1 Lite head-camera scene."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
import requests
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "env_configs/r1lite/r1lite_tennis_ball_in_box_head_only_live.yaml"
)
PLOTLY_JS = (
    REPO_ROOT / ".venv/lib/python3.10/site-packages/plotly/package_data/plotly.min.js"
)


def _encode_png(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_mask(mask_base64: str, shape: list[int] | tuple[int, ...]) -> np.ndarray:
    raw = base64.b64decode(mask_base64)
    height, width = int(shape[0]), int(shape[1])
    if len(raw) == height * width:
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width) > 0
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("L").resize((width, height))) > 0


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _points_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    stride: int = 3,
    depth_range: tuple[float, float] = (0.2, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth)
    valid &= depth >= depth_range[0]
    valid &= depth <= depth_range[1]
    sample = np.zeros_like(valid)
    sample[:: max(1, stride), :: max(1, stride)] = True
    valid &= sample
    if mask is not None:
        valid &= _resize_mask(mask, depth.shape)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32)
    z = depth[ys, xs]
    x = (xs - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys - intrinsics[1, 2]) * z / intrinsics[1, 1]
    camera_points = np.column_stack((x, y, z, np.ones_like(z)))
    points = (base_from_camera @ camera_points.T).T[:, :3]
    return points, np.column_stack((ys, xs))


def _subsample(
    points: np.ndarray,
    colors: np.ndarray | None,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= limit:
        return points, colors
    indices = np.random.default_rng(seed).choice(len(points), limit, replace=False)
    return points[indices], None if colors is None else colors[indices]


def _pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    pose[:3, 3] = position
    return pose


def _native_gripper_frame(tool_z_pose: np.ndarray) -> np.ndarray:
    """Recover right_gripper_link from the controlled +Z-forward tool frame."""
    native_from_tool_z = np.eye(4)
    native_from_tool_z[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    return (
        np.asarray(tool_z_pose, dtype=float).reshape(4, 4)
        @ np.linalg.inv(native_from_tool_z)
    )


def _transform_from_message(transform: Any) -> np.ndarray:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return _pose_matrix(
        np.array([translation.x, translation.y, translation.z], dtype=float),
        np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=float),
    )


@dataclass
class SnapshotStore:
    lock: threading.Lock = field(default_factory=threading.Lock)
    payload: dict[str, Any] = field(
        default_factory=lambda: {
            "ok": False,
            "status": "waiting for first RGB-D frame",
            "updated_at": None,
        }
    )
    overlay_jpeg: bytes | None = None

    def update(self, payload: dict[str, Any], overlay_jpeg: bytes | None) -> None:
        with self.lock:
            self.payload = payload
            if overlay_jpeg is not None:
                self.overlay_jpeg = overlay_jpeg

    def read(self) -> tuple[dict[str, Any], bytes | None]:
        with self.lock:
            return dict(self.payload), self.overlay_jpeg


class SceneCollector:
    def __init__(
        self,
        low_level: Any,
        control_api: Any,
        *,
        sam3_url: str,
        max_scene_points: int,
        max_object_points: int,
        max_grasps: int,
        trajectory_path: Path | None,
        table_plane_z: float | None,
    ) -> None:
        self.low_level = low_level
        self.control_api = control_api
        self.sam3_url = sam3_url.rstrip("/")
        self.max_scene_points = max_scene_points
        self.max_object_points = max_object_points
        self.max_grasps = max_grasps
        self.trajectory_path = trajectory_path
        self.table_plane_z = table_plane_z
        self.store = SnapshotStore()
        self.refresh_lock = threading.Lock()
        self.session = requests.Session()
        self.session.trust_env = False

    def read_payload(self) -> dict[str, Any]:
        payload, _ = self.store.read()
        payload["table_plane_z_m"] = self.table_plane_z
        trajectory: list[dict[str, Any]] = []
        if self.trajectory_path is not None and self.trajectory_path.exists():
            try:
                loaded = json.loads(self.trajectory_path.read_text())
                if isinstance(loaded, list):
                    trajectory = loaded
            except (OSError, json.JSONDecodeError):
                pass
        payload["measured_wrist_trajectory"] = trajectory
        return payload

    def _segment(self, rgb: np.ndarray, prompt: str) -> dict[str, Any]:
        response = self.session.post(
            f"{self.sam3_url}/segment",
            json={"image_base64": _encode_png(rgb), "text_prompt": prompt},
            timeout=120,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            raise RuntimeError(f"SAM3 returned no result for {prompt!r}")
        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        return {
            "prompt": prompt,
            "score": float(best.get("score", 0.0)),
            "mask": _decode_mask(best["mask_base64"], best["shape"]),
            "box": best.get("box"),
        }

    @staticmethod
    def _translation(xyz: tuple[float, float, float]) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, 3] = xyz
        return transform

    @staticmethod
    def _axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec(
            np.asarray(axis, dtype=float) * float(angle)
        ).as_matrix()
        return transform

    def _arm_points(self, joints: np.ndarray, ee_position: np.ndarray) -> np.ndarray:
        base_from_torso, _ = self.low_level._base_from_torso_link3()
        transform = base_from_torso @ self._translation((0.0, -0.335, 0.12306))
        points = [transform[:3, 3].copy()]
        joint_chain = [
            ((0.0, 0.0, 0.08605), (0.0, 0.0, 1.0)),
            ((0.0, 0.03075, 0.04925), (0.0, 1.0, 0.0)),
            ((-0.3, 0.00025004, 0.0), (0.0, 1.0, 0.0)),
            ((0.1747, 0.00049739, 0.075485), (0.0, 1.0, 0.0)),
            ((0.08, -0.031498, 0.0405), (0.0, 0.0, 1.0)),
            ((0.022503, 0.0, -0.0405), (1.0, 0.0, 0.0)),
        ]
        for angle, (origin, axis) in zip(joints, joint_chain):
            transform = transform @ self._translation(origin)
            points.append(transform[:3, 3].copy())
            transform = transform @ self._axis_rotation(axis, float(angle))
        transform = transform @ self._translation((0.08165, 0.0, 0.0))
        points.append(transform[:3, 3].copy())
        points.append(np.asarray(ee_position, dtype=float))
        return np.asarray(points)

    def _graspnet(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        tennis_mask: np.ndarray,
        base_from_camera: np.ndarray,
    ) -> dict[str, Any]:
        segmap = np.zeros(depth.shape, dtype=np.int32)
        segmap[_resize_mask(tennis_mask, depth.shape)] = 1
        grasps, scores, contact_points = (
            self.control_api._call_graspnet_depth_segmap(
                depth,
                intrinsics,
                segmap,
            )
        )
        grasps = np.asarray(grasps, dtype=float).reshape(-1, 4, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        contact_points = np.asarray(contact_points, dtype=float).reshape(-1, 3)
        order = np.argsort(scores)[::-1][: self.max_grasps]
        selected_grasps = []
        for rank, raw_index in enumerate(order):
            index = int(raw_index)
            base_from_grasp = base_from_camera @ grasps[index]
            selected_grasps.append(
                {
                    "rank": rank + 1,
                    "index": index,
                    "score": float(scores[index]),
                    "matrix": base_from_grasp.tolist(),
                    "position": base_from_grasp[:3, 3].tolist(),
                }
            )
        if len(contact_points):
            contact_h = np.column_stack(
                (contact_points, np.ones(len(contact_points), dtype=float))
            )
            contacts_base = (base_from_camera @ contact_h.T).T[:, :3]
            contacts_base = contacts_base[order[order < len(contacts_base)]]
        else:
            contacts_base = np.zeros((0, 3))
        ik = None
        if len(order):
            ik = self._ik_visualization(
                grasps[int(order[0])],
                base_from_camera,
            )
        return {
            "candidate_count": int(len(grasps)),
            "grasps": selected_grasps,
            "contact_points": contacts_base.tolist(),
            "ik": ik,
        }

    def _ik_visualization(
        self,
        camera_from_grasp: np.ndarray,
        base_from_camera: np.ndarray,
    ) -> dict[str, Any]:
        ee_from_grasp = np.asarray(
            self.low_level.graspnet_ee_from_grasp,
            dtype=float,
        ).reshape(4, 4)
        base_from_contact_grasp = base_from_camera @ camera_from_grasp
        base_from_ee = (
            base_from_contact_grasp @ np.linalg.inv(ee_from_grasp)
        )
        target_position = base_from_ee[:3, 3]
        target_quaternion = Rotation.from_matrix(base_from_ee[:3, :3]).as_quat()
        axis_name = str(self.low_level.graspnet_approach_axis)
        sign = -1.0 if axis_name.startswith("-") else 1.0
        axis_index = {"x": 0, "y": 1, "z": 2}[axis_name.lstrip("-")]
        approach_axis = (
            sign * base_from_contact_grasp[:3, axis_index]
        )
        approach_axis /= np.linalg.norm(approach_axis)
        pregrasp_position = target_position - (
            approach_axis * float(self.low_level.graspnet_pregrasp_distance_m)
        )
        current_position, _ = self.low_level.get_planning_ee_pose()
        result: dict[str, Any] = {
            "status": "unvalidated",
            "current_position": current_position.tolist(),
            "pregrasp_target": pregrasp_position.tolist(),
            "grasp_target": target_position.tolist(),
            "target_quaternion_xyzw": target_quaternion.tolist(),
            "target_matrix": base_from_ee.tolist(),
            "native_target_matrix": _native_gripper_frame(base_from_ee).tolist(),
            "requested_points": [current_position.tolist(), pregrasp_position.tolist()],
            "fk_points": [],
            "target_joints": None,
        }
        try:
            plan = self.control_api.plan_pose_waypoints(
                pregrasp_position,
                target_quaternion,
                translate_then_orient=True,
            )
            result.update(
                {
                    "status": "harness passed",
                    "waypoint_count": int(plan["waypoint_count"]),
                    "requested_points": [
                        point["position"] for point in plan["waypoints"]
                    ],
                    "fk_points": [
                        point["fk_position_base"] for point in plan["waypoints"]
                    ],
                    "target_joints": plan["waypoints"][-1]["joint_feedback"],
                    "max_fk_error_m": max(
                        point["fk_position_error_m"]
                        for point in plan["waypoints"]
                    ),
                }
            )
        except Exception as exc:
            result["status"] = "harness rejected"
            result["error"] = str(exc)
            try:
                self.low_level.reset_relaxed_ik_probe()
                target_joints = self.low_level.get_relaxed_ik_candidate(
                    pregrasp_position,
                    target_quaternion,
                    timeout_s=2.0,
                )
                fk_position, _ = self.low_level.get_relaxed_ik_fk_pose_base()
                result["fk_points"] = [
                    current_position.tolist(),
                    fk_position.tolist(),
                ]
                result["target_joints"] = target_joints.tolist()
                result["direct_fk_position"] = fk_position.tolist()
            except Exception as direct_exc:
                result["direct_ik_error"] = str(direct_exc)
        return result

    def _collect_once(self) -> tuple[dict[str, Any], bytes]:
        observation = self.low_level.get_head_rgbd(camera="left", require_depth=True)
        rgb = np.asarray(observation["rgb"], dtype=np.uint8)
        depth = np.asarray(observation["depth"], dtype=np.float32)
        intrinsics = np.asarray(observation["intrinsics"], dtype=np.float64)
        base_from_camera = self.low_level.lookup_camera_to_base(
            frame_id=str(observation["frame_id"]),
            stamp_ns=int(observation["stamp_ns"]),
            camera_label="head_stereo_left",
        )

        detections: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for key, prompt in (
            ("tennis_ball", self.low_level.object_prompt("tennis ball")),
            ("cardboard_box", self.low_level.object_prompt("cardboard box")),
        ):
            try:
                detections[key] = self._segment(rgb, prompt)
            except Exception as exc:
                errors[key] = str(exc)

        scene_points, scene_pixels = _points_from_depth(
            depth,
            intrinsics,
            base_from_camera,
            stride=3,
        )
        scene_colors = rgb[scene_pixels[:, 0], scene_pixels[:, 1]]
        scene_points, scene_colors = _subsample(
            scene_points,
            scene_colors,
            self.max_scene_points,
            seed=1,
        )

        overlay = rgb.copy()
        objects: dict[str, Any] = {}
        object_colors = {
            "tennis_ball": np.array([255, 214, 10], dtype=np.uint8),
            "cardboard_box": np.array([50, 205, 130], dtype=np.uint8),
        }
        for seed, (key, detection) in enumerate(detections.items(), start=2):
            mask = _resize_mask(detection["mask"], depth.shape)
            points, _ = _points_from_depth(
                depth,
                intrinsics,
                base_from_camera,
                mask=mask,
                stride=1,
            )
            points, _ = _subsample(
                points,
                None,
                self.max_object_points,
                seed=seed,
            )
            color = object_colors[key]
            rgb_mask = _resize_mask(mask, rgb.shape[:2])
            overlay[rgb_mask] = (
                overlay[rgb_mask].astype(np.float32) * 0.45 + color * 0.55
            ).astype(np.uint8)
            center = np.median(points, axis=0).tolist() if len(points) else None
            objects[key] = {
                "prompt": detection["prompt"],
                "score": detection["score"],
                "mask_pixels": int(np.count_nonzero(mask)),
                "position": center,
                "points": points.round(5).tolist(),
                "color": color.tolist(),
            }

        graspnet: dict[str, Any] | None = None
        tennis_detection = detections.get("tennis_ball")
        if tennis_detection is not None:
            try:
                graspnet = self._graspnet(
                    depth,
                    intrinsics,
                    tennis_detection["mask"],
                    base_from_camera,
                )
            except Exception as exc:
                errors["graspnet"] = str(exc)

        ee_position, ee_quaternion = self.low_level.get_planning_ee_pose()
        tool_z_ee_matrix = _pose_matrix(ee_position, ee_quaternion)
        native_ee_matrix = _native_gripper_frame(tool_z_ee_matrix)
        arm_joints = self.low_level.get_arm_joints()
        arm_points = self._arm_points(arm_joints, ee_position)
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 86],
        )
        if not ok:
            raise RuntimeError("Failed to encode the SAM3 overlay")

        return {
            "ok": True,
            "status": "snapshot ready",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame_id": observation["frame_id"],
            "stamp_ns": int(observation["stamp_ns"]),
            "scene": {
                "points": scene_points.round(5).tolist(),
                "colors": scene_colors.tolist(),
            },
            "head_camera": {
                "position": base_from_camera[:3, 3].tolist(),
                "matrix": base_from_camera.tolist(),
            },
            "right_arm": {
                "joint_feedback": arm_joints.tolist(),
                "link_points": arm_points.tolist(),
                "ee_position": ee_position.tolist(),
                "ee_quaternion_xyzw": ee_quaternion.tolist(),
                "ee_matrix": tool_z_ee_matrix.tolist(),
                "native_ee_matrix": native_ee_matrix.tolist(),
                "visual_frame_convention": "gripper forward direction is +Z",
            },
            "objects": objects,
            "graspnet": graspnet,
            "errors": errors,
        }, encoded.tobytes()

    def refresh(self) -> None:
        with self.refresh_lock:
            try:
                payload, jpeg = self._collect_once()
                self.store.update(payload, jpeg)
            except Exception as exc:
                old, jpeg = self.store.read()
                old.update(
                    {
                        "ok": False,
                        "status": str(exc),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                self.store.update(old, jpeg)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>R1 Lite 场景坐标可视化</title>
  <script src="/plotly.min.js"></script>
  <style>
    :root{color-scheme:dark;--bg:#071019;--panel:#101c27;--line:#243746;--text:#e8f2f8;--muted:#91a7b5}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
    header{height:58px;display:flex;align-items:center;gap:18px;padding:0 20px;border-bottom:1px solid var(--line)}
    h1{font-size:18px;margin:0}.status{color:var(--muted)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#f0b429;margin-right:7px}
    main{display:grid;grid-template-columns:minmax(340px,34%) 1fr;gap:12px;padding:12px;height:calc(100vh - 58px)}
    .column{display:grid;grid-template-rows:minmax(260px,1fr) auto;gap:12px;min-height:0}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
    .plots{display:grid;grid-template-rows:1.25fr 1fr;gap:12px;min-height:0}
    .camera{width:100%;height:100%;object-fit:contain;background:#03080c}.plot{width:100%;height:100%}
    .cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:10px}.card{padding:10px;background:#0b1620;border:1px solid var(--line);border-radius:7px}
    .card b{display:block;color:#74d3ff;margin-bottom:5px}.xyz{font-family:ui-monospace,monospace;font-size:12px;white-space:nowrap}.error{color:#ff8f8f}
    @media(max-width:900px){main{grid-template-columns:1fr;height:auto}.column{min-height:620px}}
  </style>
</head>
<body>
<header><h1>R1 Lite 场景快照</h1><span class="status"><span class="dot" id="dot"></span><span id="status">采集中</span></span><span class="status" id="updated"></span><span class="status">按 F5 采集新快照</span></header>
<main>
  <section class="column">
    <div class="panel"><img class="camera" id="overlay" src="/overlay.jpg" alt="SAM3 overlay"></div>
    <div class="panel cards" id="cards"></div>
  </section>
  <section class="plots">
    <div class="panel"><div class="plot" id="plot"></div></div>
    <div class="panel"><div class="plot" id="ikplot"></div></div>
  </section>
</main>
<script>
const fmt=v=>v&&v.length===3?v.map(x=>Number(x).toFixed(3)).join(", "):"不可用";
const frameTraces=(m,name,scale)=>{
  if(!m)return[]; const o=[m[0][3],m[1][3],m[2][3]], cs=["#ff4d5a","#4ee28a","#42a5ff"], ns=["X","Y","Z"];
  return [0,1,2].map(i=>({type:"scatter3d",mode:"lines",name:name+" "+ns[i],showlegend:false,
    x:[o[0],o[0]+m[0][i]*scale],y:[o[1],o[1]+m[1][i]*scale],z:[o[2],o[2]+m[2][i]*scale],
    line:{color:cs[i],width:7},hovertemplate:name+" "+ns[i]+"<extra></extra>"}));
};
function objectTrace(obj,name,color){
  if(!obj||!obj.points||!obj.points.length)return null;
  return {type:"scatter3d",mode:"markers",name,x:obj.points.map(p=>p[0]),y:obj.points.map(p=>p[1]),z:obj.points.map(p=>p[2]),
    marker:{size:3,color,opacity:.95},hovertemplate:name+"<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"};
}
async function refresh(){
  try{
    const r=await fetch("/api/scene",{cache:"no-store"}),d=await r.json();
    document.getElementById("status").textContent=d.status||"unknown";
    document.getElementById("updated").textContent=d.updated_at||"";
    document.getElementById("dot").style.background=d.ok?"#35d07f":"#ff6262";
    document.getElementById("overlay").src="/overlay.jpg?t="+Date.now();
    if(!d.scene)return;
    const p=d.scene.points,c=d.scene.colors,traces=[{type:"scatter3d",mode:"markers",name:"头部 RGB-D 点云",
      x:p.map(q=>q[0]),y:p.map(q=>q[1]),z:p.map(q=>q[2]),marker:{size:1.4,color:c.map(q=>`rgb(${q[0]},${q[1]},${q[2]})`),opacity:.55},
      hoverinfo:"skip"}];
    const ball=d.objects&&d.objects.tennis_ball,box=d.objects&&d.objects.cardboard_box;
    const bt=objectTrace(ball,"SAM3 网球","#ffd60a"),xt=objectTrace(box,"SAM3 盒子","#35d07f"); if(bt)traces.push(bt);if(xt)traces.push(xt);
    if(d.table_plane_z_m!=null&&p.length){
      const xs=p.map(q=>q[0]),ys=p.map(q=>q[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),z=d.table_plane_z_m;
      traces.push({type:"surface",name:"标定桌面平面",showscale:false,opacity:.28,
        x:[[xmin,xmax],[xmin,xmax]],y:[[ymin,ymin],[ymax,ymax]],z:[[z,z],[z,z]],
        colorscale:[[0,"#4d8cff"],[1,"#4d8cff"]],hovertemplate:"桌面 z="+Number(z).toFixed(3)+" m<extra></extra>"});
    }
    const measured=d.measured_wrist_trajectory||[];
    if(measured.length)traces.push({type:"scatter3d",mode:"lines",name:"真机手腕实际轨迹",
      x:measured.map(q=>q.x_m),y:measured.map(q=>q.y_m),z:measured.map(q=>q.z_m),
      line:{color:"#ff2d95",width:9},hovertemplate:"实际手腕轨迹<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"});
    const arm=d.right_arm&&d.right_arm.link_points;if(arm)traces.push({type:"scatter3d",mode:"lines+markers",name:"右机械臂",
      x:arm.map(q=>q[0]),y:arm.map(q=>q[1]),z:arm.map(q=>q[2]),line:{color:"#ff7a45",width:12},marker:{size:5,color:"#fff0e8"},
      hovertemplate:"右臂关节点<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"});
    const gn=d.graspnet;if(gn&&gn.contact_points&&gn.contact_points.length)traces.push({type:"scatter3d",mode:"markers",name:"GraspNet 接触点",
      x:gn.contact_points.map(q=>q[0]),y:gn.contact_points.map(q=>q[1]),z:gn.contact_points.map(q=>q[2]),marker:{size:6,color:"#ff3bd4",symbol:"diamond"}});
    if(gn&&gn.grasps)gn.grasps.forEach((g,i)=>traces.push(...frameTraces(g.matrix,`GraspNet #${i+1} score ${g.score.toFixed(3)}`,.055)));
    traces.push(...frameTraces([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],"base_link",.12));
    traces.push(
      ...frameTraces(d.head_camera.matrix,"头部相机",.10),
      ...frameTraces(d.right_arm.native_ee_matrix,"right_gripper_link（原生 +X 前向）",.08),
      ...frameTraces(d.right_arm.ee_matrix,"right_gripper_tool_z（+Z 前向）",.08)
    );
    Plotly.react("plot",traces,{paper_bgcolor:"#101c27",plot_bgcolor:"#101c27",font:{color:"#dceaf2"},margin:{l:0,r:0,t:38,b:0},
      title:{text:"场景点云 / 机械臂 / GraspNet",font:{size:15}},
      legend:{orientation:"h"},scene:{aspectmode:"data",xaxis:{title:"base X (m)",gridcolor:"#2a3c49"},yaxis:{title:"base Y (m)",gridcolor:"#2a3c49"},
      zaxis:{title:"base Z (m)",gridcolor:"#2a3c49"},bgcolor:"#101c27"}},{responsive:true,displaylogo:false});
    const ik=gn&&gn.ik,iktraces=[];
    if(ik){
      const rq=ik.requested_points||[],fk=ik.fk_points||[];
      if(rq.length)iktraces.push({type:"scatter3d",mode:"lines+markers",name:"IK 请求目标点",
        x:rq.map(q=>q[0]),y:rq.map(q=>q[1]),z:rq.map(q=>q[2]),line:{color:"#55c7ff",width:7},marker:{size:3,color:"#bcecff"}});
      if(fk.length)iktraces.push({type:"scatter3d",mode:"lines+markers",name:"RelaxedIK FK 解算点",
        x:fk.map(q=>q[0]),y:fk.map(q=>q[1]),z:fk.map(q=>q[2]),line:{color:"#ff9d3d",width:7,dash:"dash"},marker:{size:4,color:"#ffd3a1"}});
      [["当前末端",ik.current_position,"#ffffff","circle"],["预抓取目标",ik.pregrasp_target,"#00e5ff","diamond"],["抓取目标",ik.grasp_target,"#ff3bd4","diamond-open"]].forEach(v=>{
        if(v[1])iktraces.push({type:"scatter3d",mode:"markers+text",name:v[0],x:[v[1][0]],y:[v[1][1]],z:[v[1][2]],
          marker:{size:8,color:v[2],symbol:v[3]},text:[v[0]],textposition:"top center"});
      });
      iktraces.push(...frameTraces(ik.target_matrix,"IK 目标夹爪（+Z 前向）",.065));
    }
    Plotly.react("ikplot",iktraces,{paper_bgcolor:"#101c27",plot_bgcolor:"#101c27",font:{color:"#dceaf2"},margin:{l:0,r:0,t:42,b:0},
      title:{text:ik?`IK 目标与解算结果 — ${ik.status}`:"IK 数据不可用",font:{size:15}},legend:{orientation:"h"},
      scene:{aspectmode:"data",xaxis:{title:"base X (m)",gridcolor:"#2a3c49"},yaxis:{title:"base Y (m)",gridcolor:"#2a3c49"},
      zaxis:{title:"base Z (m)",gridcolor:"#2a3c49"},bgcolor:"#101c27"}},{responsive:true,displaylogo:false});
    const cards=[
      ["头部相机",fmt(d.head_camera.position)],
      ["右臂末端",fmt(d.right_arm.ee_position)],
      ["SAM3 网球",fmt(ball&&ball.position)+"<br>score "+(ball?ball.score.toFixed(3):"-")],
      ["SAM3 盒子",fmt(box&&box.position)+"<br>score "+(box?box.score.toFixed(3):"-")],
      ["标定桌面平面",d.table_plane_z_m==null?"不可用":"z = "+Number(d.table_plane_z_m).toFixed(3)+" m"],
      ["真机手腕轨迹",`${measured.length} 个实测点`],
      ["GraspNet",gn?`${gn.candidate_count} candidates / 显示 ${gn.grasps.length}`:"不可用"],
      ["右臂关节",d.right_arm.joint_feedback.map(x=>Number(x).toFixed(3)).join(", ")],
      ["IK 预抓取目标",fmt(ik&&ik.pregrasp_target)],
      ["IK 目标关节",ik&&ik.target_joints?ik.target_joints.map(x=>Number(x).toFixed(3)).join(", "):"不可用"]
    ];
    document.getElementById("cards").innerHTML=cards.map(x=>`<div class="card"><b>${x[0]}</b><span class="xyz">${x[1]}</span></div>`).join("")+
      (d.errors&&Object.keys(d.errors).length?`<div class="card error">${JSON.stringify(d.errors)}</div>`:"");
  }catch(e){document.getElementById("status").textContent=e.toString();document.getElementById("dot").style.background="#ff6262"}
}
refresh();
</script></body></html>"""


def make_handler(collector: SceneCollector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                collector.refresh()
                body = HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif path == "/api/scene":
                payload = collector.read_payload()
                body = json.dumps(payload, allow_nan=False).encode()
                content_type = "application/json"
            elif path == "/overlay.jpg":
                _, jpeg = collector.store.read()
                if jpeg is None:
                    self.send_error(503, "No image is available yet")
                    return
                body = jpeg
                content_type = "image/jpeg"
            elif path == "/plotly.min.js":
                if not PLOTLY_JS.exists():
                    self.send_error(500, f"Plotly JavaScript not found: {PLOTLY_JS}")
                    return
                body = PLOTLY_JS.read_bytes()
                content_type = "application/javascript"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8114")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--max-scene-points", type=int, default=14000)
    parser.add_argument("--max-object-points", type=int, default=3500)
    parser.add_argument("--max-grasps", type=int, default=8)
    parser.add_argument("--trajectory-json", type=Path)
    parser.add_argument(
        "--head-plane-calibration",
        type=Path,
        default=REPO_ROOT / "outputs/r1lite/head_plane_calibration.json",
    )
    args = parser.parse_args()

    table_plane_z = None
    if args.head_plane_calibration.exists():
        calibration = json.loads(args.head_plane_calibration.read_text())
        if calibration.get("plane_z") is not None:
            table_plane_z = float(calibration["plane_z"])

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    config = DictLoader.load([str(args.config)])
    low_level = instantiate(config["env"]["cfg"]["low_level"])
    from capx.integrations.r1lite.control import R1LiteControlApi

    collector = SceneCollector(
        low_level,
        R1LiteControlApi(low_level),
        sam3_url=args.sam3_url,
        max_scene_points=args.max_scene_points,
        max_object_points=args.max_object_points,
        max_grasps=args.max_grasps,
        trajectory_path=args.trajectory_json,
        table_plane_z=table_plane_z,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(collector))
    print(f"R1 Lite scene dashboard: http://127.0.0.1:{args.port}/", flush=True)
    print("Read-only mode: no arm or gripper commands are sent.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        low_level.close()


if __name__ == "__main__":
    main()
