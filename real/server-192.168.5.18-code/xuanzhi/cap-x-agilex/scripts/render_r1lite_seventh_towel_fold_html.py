#!/usr/bin/env python3
"""Render the approved seventh-pipeline Cartesian targets as offline 3D HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ik-audit", type=Path)
    parser.add_argument("--towel-mask", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--point-cloud-label", default="Towel mask projection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plotly-js",
        type=Path,
        default=Path("outputs/r1lite_tennis_ball_in_box/plotly.min.js"),
    )
    return parser.parse_args()


def project_mask(
    mask_path: Path,
    calibration_path: Path,
    *,
    maximum_points: int = 6000,
) -> list[list[float]]:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    pixels_vu = np.argwhere(mask)
    if len(pixels_vu) == 0:
        raise ValueError("towel mask is empty")
    stride = max(1, int(np.ceil(len(pixels_vu) / maximum_points)))
    sampled = pixels_vu[::stride]
    calibration = json.loads(calibration_path.read_text())
    homography = np.asarray(
        calibration["image_to_base_xy_homography"],
        dtype=float,
    )
    uv1 = np.column_stack(
        [sampled[:, 1], sampled[:, 0], np.ones(len(sampled))]
    )
    projected = (homography @ uv1.T).T
    valid = np.isfinite(projected).all(axis=1) & (
        np.abs(projected[:, 2]) > 1e-12
    )
    xy = projected[valid, :2] / projected[valid, 2, None]
    z = np.full((len(xy), 1), float(calibration["plane_z"]))
    return np.column_stack([xy, z]).tolist()


def ik_candidate_sequences(audit_path: Path) -> dict[str, list[dict]]:
    audit = json.loads(audit_path.read_text())
    sequences = {"left": [], "right": []}
    for stage in audit.get("stages", []):
        stage_id = stage["id"]
        for side in sequences:
            for item in stage.get("rejected", {}).get(side, []):
                if "fk_position_base_m" not in item:
                    continue
                sequences[side].append(
                    {
                        "position": item["fk_position_base_m"],
                        "label": f"{stage_id}/{item['label']}",
                        "position_error_mm": round(
                            float(item["fk_position_error_m"]) * 1000.0,
                            2,
                        ),
                        "orientation_error_deg": round(
                            float(item["fk_orientation_error_rad"])
                            * 180.0
                            / np.pi,
                            2,
                        ),
                        "ik_iteration": item.get("ik_iteration"),
                    }
                )
        selected = stage.get("selected_pair")
        if selected:
            for side in sequences:
                item = selected[side]
                sequences[side].append(
                    {
                        "position": item["fk_position_base_m"],
                        "label": f"{stage_id}/{item['label']}",
                        "position_error_mm": round(
                            float(item["fk_position_error_m"]) * 1000.0,
                            2,
                        ),
                        "orientation_error_deg": round(
                            float(item["fk_orientation_error_rad"])
                            * 180.0
                            / np.pi,
                            2,
                        ),
                        "ik_iteration": item.get("ik_iteration"),
                    }
                )
    return sequences


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    stages = plan["stages"]
    left = [item["left_tcp_position_base_m"] for item in stages]
    right = [item["right_tcp_position_base_m"] for item in stages]
    labels = [item["id"] for item in stages]
    keypoints = plan["perception"]["keypoints"]
    objects = {
        name: item["base_xyz_m"]
        for name, item in keypoints.items()
    }
    fold = stages[-1]
    fold_fingertips = {
        "left_fold_fingertip": fold["left_fingertip_position_base_m"],
        "right_fold_fingertip": fold["right_fingertip_position_base_m"],
    }
    ik_candidates = (
        ik_candidate_sequences(args.ik_audit)
        if args.ik_audit is not None
        else {"left": [], "right": []}
    )
    if (args.towel_mask is None) != (args.calibration is None):
        raise ValueError("--towel-mask and --calibration must be provided together")
    towel_cloud = (
        project_mask(args.towel_mask, args.calibration)
        if args.towel_mask is not None
        else []
    )
    plotly_source = args.plotly_js.read_text()
    payload = json.dumps(
        {
            "left": left,
            "right": right,
            "labels": labels,
            "objects": objects,
            "fold_fingertips": fold_fingertips,
            "ik_candidates": ik_candidates,
            "towel_cloud": towel_cloud,
            "point_cloud_label": args.point_cloud_label,
        },
        ensure_ascii=False,
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>R1 Lite Pipeline 7</title>
<script>{plotly_source}</script></head>
<body><div id="plot" style="width:100vw;height:100vh"></div>
<script>
const data = {payload};
function trajectory(name, points, color) {{
  return {{
    type: "scatter3d", mode: "lines+markers+text", name,
    x: points.map(p => p[0]), y: points.map(p => p[1]), z: points.map(p => p[2]),
    text: data.labels, textposition: "top center",
    line: {{color, width: 8}}, marker: {{color, size: 5}}
  }};
}}
function ikTrajectory(name, items, color) {{
  return {{
    type: "scatter3d", mode: "lines+markers", name,
    x: items.map(p => p.position[0]),
    y: items.map(p => p.position[1]),
    z: items.map(p => p.position[2]),
    text: items.map(p => `${{p.label}}<br>iteration=${{p.ik_iteration}}<br>position error=${{p.position_error_mm}} mm<br>orientation error=${{p.orientation_error_deg}} deg`),
    hovertemplate: "%{{text}}<extra></extra>",
    line: {{color, width: 5, dash: "dot"}},
    marker: {{color, size: 4, symbol: "diamond"}}
  }};
}}
const objectNames = Object.keys(data.objects);
const objects = {{
  type: "scatter3d", mode: "markers+text", name: "Towel geometry",
  x: objectNames.map(n => data.objects[n][0]),
  y: objectNames.map(n => data.objects[n][1]),
  z: objectNames.map(n => data.objects[n][2]),
  text: objectNames, textposition: "bottom center",
  marker: {{color: "black", size: 7}}
}};
const fingertipNames = Object.keys(data.fold_fingertips);
const fingertips = {{
  type: "scatter3d", mode: "markers+text", name: "2cm fold fingertips",
  x: fingertipNames.map(n => data.fold_fingertips[n][0]),
  y: fingertipNames.map(n => data.fold_fingertips[n][1]),
  z: fingertipNames.map(n => data.fold_fingertips[n][2]),
  text: fingertipNames, textposition: "top center",
  marker: {{color: "#ff8800", size: 8}}
}};
const towelCloud = {{
  type: "scatter3d", mode: "markers", name: data.point_cloud_label,
  x: data.towel_cloud.map(p => p[0]),
  y: data.towel_cloud.map(p => p[1]),
  z: data.towel_cloud.map(p => p[2]),
  hoverinfo: "skip",
  marker: {{color: "#b07d4f", size: 2, opacity: 0.45}}
}};
Plotly.newPlot("plot", [
  towelCloud,
  trajectory("Left TCP target", data.left, "#00aa55"),
  trajectory("Right TCP target", data.right, "#3366ff"),
  ikTrajectory("Left IK candidate FK sequence", data.ik_candidates.left, "#ff3355"),
  ikTrajectory("Right IK candidate FK sequence", data.ik_candidates.right, "#8a2be2"),
  objects,
  fingertips
], {{
  title: "Pipeline 7 IK candidate trajectories and towel point cloud (no robot motion)",
  scene: {{
    xaxis: {{title: "base X (m)"}},
    yaxis: {{title: "base Y (m)"}},
    zaxis: {{title: "base Z (m)"}},
    aspectmode: "data"
  }}
}}, {{responsive: true}});
</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)


if __name__ == "__main__":
    main()
