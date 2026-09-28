#!/usr/bin/env python3
"""Render put-general head RGB-D detections as an interactive 3D page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    scene_event = next(
        event
        for event in audit["events"]
        if event["event"] == "initial_head_rgbd_scene"
    )
    detections = scene_event["detections"]
    colors = {
        "tennis ball": "#d9ef38",
        "peach": "#ff9b73",
        "apple": "#ef4444",
        "bread": "#d97706",
        "cup": "#8b5cf6",
        "cardboard box": "#a16207",
    }
    traces = []
    all_points = []
    for name, detection in detections.items():
        points = detection["point_cloud_base_m"]
        all_points.extend(points)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": name,
                "x": [point[0] for point in points],
                "y": [point[1] for point in points],
                "z": [point[2] for point in points],
                "marker": {
                    "size": 2.2,
                    "color": colors.get(name, "#22d3ee"),
                    "opacity": 0.72,
                },
            }
        )
        center = detection["position_base_m"]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": f"{name} center",
                "x": [center[0]],
                "y": [center[1]],
                "z": [center[2]],
                "text": [name],
                "textposition": "top center",
                "marker": {
                    "size": 7,
                    "color": colors.get(name, "#22d3ee"),
                    "symbol": "diamond",
                },
                "showlegend": False,
            }
        )
    box_name = next(
        (name for name in detections if "box" in name.lower()),
        None,
    )
    box = None if box_name is None else detections[box_name]
    if box is not None:
        minimum = box["opening_xy_min_base_m"]
        maximum = box["opening_xy_max_base_m"]
        rim_z = box["rim_z_base_m"]
        rectangle = [
            [minimum[0], minimum[1], rim_z],
            [maximum[0], minimum[1], rim_z],
            [maximum[0], maximum[1], rim_z],
            [minimum[0], maximum[1], rim_z],
            [minimum[0], minimum[1], rim_z],
        ]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "box safe opening",
                "x": [point[0] for point in rectangle],
                "y": [point[1] for point in rectangle],
                "z": [point[2] for point in rectangle],
                "line": {"width": 8, "color": "#fbbf24"},
            }
        )
    table_z = float(next(iter(detections.values()))["table_z_base_m"])
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values) - 0.04, max(x_values) + 0.04
    y_min, y_max = min(y_values) - 0.04, max(y_values) + 0.04
    traces.insert(
        0,
        {
            "type": "mesh3d",
            "name": f"table plane z={table_z:.3f}m",
            "x": [x_min, x_max, x_max, x_min],
            "y": [y_min, y_min, y_max, y_max],
            "z": [table_z, table_z, table_z, table_z],
            "i": [0, 0],
            "j": [1, 2],
            "k": [2, 3],
            "color": "#60a5fa",
            "opacity": 0.24,
            "flatshading": True,
            "hovertemplate": (
                f"calibrated table plane<br>z={table_z:.4f}m"
                "<extra></extra>"
            ),
        },
    )
    layout = {
        "title": "put-general head RGB-D detections",
        "scene": {
            "xaxis": {"title": "base X (m)"},
            "yaxis": {"title": "base Y (m)"},
            "zaxis": {"title": "base Z (m)"},
            "aspectmode": "data",
        },
        "legend": {"orientation": "h"},
        "margin": {"l": 0, "r": 0, "t": 55, "b": 0},
    }
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>put-general point cloud</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body style="margin:0;background:#111827;color:#f9fafb">
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
Plotly.newPlot("plot", {json.dumps(traces)}, {json.dumps(layout)},
  {{responsive: true}});
</script></body></html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
