#!/usr/bin/env python3
"""Render a standalone Plotly HTML for R1 Lite scene + measured wrist trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_traces(scene: dict, trajectory: list[dict]) -> list[dict]:
    traces: list[dict] = []
    points = np.asarray(scene["scene"]["points"], dtype=float)
    if points.size:
        points = points.reshape(-1, 3)
    colors = scene["scene"]["colors"]
    if len(points):
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "头部 RGB-D 场景点云",
                "x": points[:, 0].tolist(),
                "y": points[:, 1].tolist(),
                "z": points[:, 2].tolist(),
                "marker": {
                    "size": 1.2,
                    "color": [f"rgb({r},{g},{b})" for r, g, b in colors],
                    "opacity": 0.42,
                },
                "hoverinfo": "skip",
            }
        )
    for key, name, color in (
        ("tennis_ball", "SAM3 网球点云", "#ffd60a"),
        ("cardboard_box", "SAM3 盒子点云", "#35d07f"),
    ):
        obj = scene.get("objects", {}).get(key, {})
        obj_points = np.asarray(obj.get("points") or [], dtype=float)
        if len(obj_points):
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "name": name,
                    "x": obj_points[:, 0].tolist(),
                    "y": obj_points[:, 1].tolist(),
                    "z": obj_points[:, 2].tolist(),
                    "marker": {"size": 3.5, "color": color, "opacity": 0.95},
                    "hovertemplate": (
                        f"{name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>"
                    ),
                }
            )
        position = obj.get("position")
        if position is not None:
            pos = np.asarray(position, dtype=float).reshape(3)
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers+text",
                    "name": f"{name} 中心",
                    "x": [float(pos[0])],
                    "y": [float(pos[1])],
                    "z": [float(pos[2])],
                    "text": [name.replace("SAM3 ", "")],
                    "textposition": "top center",
                    "marker": {"size": 8, "color": color, "symbol": "diamond"},
                    "hovertemplate": (
                        f"{name} 中心<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>"
                    ),
                }
            )

    table_z = scene.get("table_plane_z_m")
    object_positions = [
        np.asarray(obj.get("position"), dtype=float).reshape(3)
        for obj in scene.get("objects", {}).values()
        if obj.get("position") is not None
    ]
    if table_z is not None and (len(points) or object_positions):
        if len(points):
            xs = points[:, 0]
            ys = points[:, 1]
        else:
            stacked = np.asarray(object_positions, dtype=float)
            xs = stacked[:, 0]
            ys = stacked[:, 1]
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        z = float(table_z)
        traces.append(
            {
                "type": "surface",
                "name": "标定桌面平面",
                "showscale": False,
                "opacity": 0.28,
                "x": [[xmin, xmax], [xmin, xmax]],
                "y": [[ymin, ymin], [ymax, ymax]],
                "z": [[z, z], [z, z]],
                "colorscale": [[0, "#4d8cff"], [1, "#4d8cff"]],
                "hovertemplate": f"桌面 z={z:.3f} m<extra></extra>",
            }
        )

    if trajectory:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "真机手腕实际移动轨迹",
                "x": [sample["x_m"] for sample in trajectory],
                "y": [sample["y_m"] for sample in trajectory],
                "z": [sample["z_m"] for sample in trajectory],
                "line": {"color": "#ff2d95", "width": 9},
                "hovertemplate": (
                    "实际手腕轨迹<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
            }
        )
    return traces


def render_html(scene_path: Path, trajectory_path: Path, output_path: Path) -> None:
    scene = json.loads(scene_path.read_text())
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else []
    traces = build_traces(scene, trajectory)
    plotly_js = REPO_ROOT / ".venv/lib/python3.10/site-packages/plotly/package_data/plotly.min.js"
    if not plotly_js.exists():
        plotly_js = REPO_ROOT / ".capx_r1lite_venv/lib/python3.10/site-packages/plotly/package_data/plotly.min.js"
    plotly_name = "plotly.min.js"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plotly_js.exists():
        (output_path.parent / plotly_name).write_bytes(plotly_js.read_bytes())
    html = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>R1 Lite 腕部真机三维测试</title>"
        f"<script src='{plotly_name}'></script>"
        "<style>html,body,#plot{width:100%;height:100%;margin:0;background:#071019}</style>"
        "</head><body><div id='plot'></div><script>"
        f"const data={json.dumps(traces, ensure_ascii=False)};"
        "Plotly.newPlot('plot', data, {"
        "paper_bgcolor:'#101c27',plot_bgcolor:'#101c27',font:{color:'#dceaf2'},"
        "margin:{l:0,r:0,t:42,b:0},"
        "title:{text:'R1 Lite 腕部真机测试 — 网球 / 桌面 / 盒子 / 实测手腕轨迹',font:{size:16}},"
        "legend:{orientation:'h'},"
        "scene:{aspectmode:'data',"
        "xaxis:{title:'base X (m)',gridcolor:'#2a3c49'},"
        "yaxis:{title:'base Y (m)',gridcolor:'#2a3c49'},"
        "zaxis:{title:'base Z (m)',gridcolor:'#2a3c49'},"
        "bgcolor:'#101c27'}}"
        ", {responsive:true,displaylogo:false});"
        "</script></body></html>"
    )
    output_path.write_text(html, encoding="utf-8")


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    render_html(args.scene_json, args.trajectory_json, args.output_html)
    print(f"Wrote {args.output_html}", flush=True)


if __name__ == "__main__":
    main()
