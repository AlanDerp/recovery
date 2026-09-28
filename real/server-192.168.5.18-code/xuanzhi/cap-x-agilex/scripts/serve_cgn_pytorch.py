"""Small Contact-GraspNet service compatible with the Cap-X point-cloud API."""

from __future__ import annotations

import argparse
import base64
import io
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial import cKDTree
from torch_geometric.nn import fps

import cgn_pytorch


MODEL = None
DEVICE = None


class PlanPointCloudsRequest(BaseModel):
    pc_full_base64: str
    pc_segment_base64: str
    segmap_id: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    forward_passes: int = 1
    max_retries: int = 7


def _decode_array(value: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(value)) as stream:
        return np.load(stream, allow_pickle=False)


def _encode_array(value: np.ndarray) -> str:
    with io.BytesIO() as stream:
        np.save(stream, np.asarray(value))
        return base64.b64encode(stream.getvalue()).decode("ascii")


def _empty_response() -> dict[str, str]:
    empty = np.empty((0,), dtype=np.float32)
    return {
        "grasps_base64": _encode_array(empty),
        "scores_base64": _encode_array(empty),
        "contact_pts_base64": _encode_array(empty),
    }


def _prepare_scene(points: np.ndarray, count: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 64:
        raise ValueError(f"point cloud has only {len(points)} finite points")
    center = np.mean(points, axis=0, dtype=np.float64).astype(np.float32)
    centered = points - center
    rng = np.random.default_rng(0)
    indices = rng.choice(len(centered), size=count, replace=len(centered) < count)
    return centered[indices], center


@torch.inference_mode()
def _infer(pc_full: np.ndarray, pc_segment: np.ndarray) -> dict[str, str]:
    scene, center = _prepare_scene(pc_full)
    segment = np.asarray(pc_segment, dtype=np.float32).reshape(-1, 3)
    segment = segment[np.isfinite(segment).all(axis=1)] - center
    if len(segment) < 30:
        raise ValueError(f"segment has only {len(segment)} finite points")

    scene_tensor = torch.as_tensor(scene, device=DEVICE)
    batch = torch.zeros(len(scene_tensor), dtype=torch.long, device=DEVICE)
    sampled_indices = fps(scene_tensor, batch, ratio=2048 / len(scene_tensor))
    features = torch.empty((len(scene_tensor), 0), dtype=torch.float32, device=DEVICE)
    points, grasps, logits, widths, _, _ = MODEL(
        features,
        scene_tensor,
        batch,
        sampled_indices,
    )

    points = points.reshape(-1, 3).detach().cpu().numpy()
    grasps = grasps.reshape(-1, 4, 4).detach().cpu().numpy()
    scores = torch.sigmoid(logits).reshape(-1).detach().cpu().numpy()
    widths = widths.reshape(-1).detach().cpu().numpy()

    distances, _ = cKDTree(segment).query(points, k=1)
    keep = np.isfinite(scores) & (distances <= 0.03) & (widths >= 0.045)
    indices = np.flatnonzero(keep)
    if not len(indices):
        return _empty_response()
    indices = indices[np.argsort(scores[indices])[::-1]][:128]

    selected_grasps = grasps[indices].astype(np.float32)
    selected_grasps[:, :3, 3] += center
    selected_contacts = (points[indices] + center).astype(np.float32)
    selected_scores = scores[indices].astype(np.float32)
    return {
        "grasps_base64": _encode_array(selected_grasps),
        "scores_base64": _encode_array(selected_scores),
        "contact_pts_base64": _encode_array(selected_contacts),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, DEVICE
    DEVICE = torch.device("cuda:0")
    MODEL, _, _ = cgn_pytorch.from_pretrained(device=0)
    MODEL.eval()
    yield
    MODEL = None
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": MODEL is not None, "device": str(DEVICE)}


@app.post("/plan_point_clouds")
def plan_point_clouds(request: PlanPointCloudsRequest) -> dict[str, str]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    try:
        return _infer(
            _decode_array(request.pc_full_base64),
            _decode_array(request.pc_segment_base64),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8115)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
