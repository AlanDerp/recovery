#!/usr/bin/env python3
"""Render sequential towel perception, certified IK, and measured TCP motion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.kinematics import (  # noqa: E402
    R1LiteCandidateKinematics,
)
from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    tool_z_axis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--execute-audit", type=Path, required=True)
    parser.add_argument("--towel-mask", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
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


def certified_ik_trajectory(
    bundle: dict[str, Any],
    side: str,
) -> list[dict[str, Any]]:
    torso = np.asarray(bundle["initial_state"]["torso_joints_rad"], dtype=float)
    fk = R1LiteCandidateKinematics(side=side)
    result = []
    initial = np.asarray(
        bundle["initial_state"][f"{side}_joints_rad"], dtype=float
    )
    initial_fk = fk.fk(initial, torso)
    result.append(
        {
            "stage_id": "initial",
            "waypoint_index": 0,
            "position_base_m": initial_fk.position_base_m.tolist(),
        }
    )
    for stage in bundle["stages"]:
        for index, joints in enumerate(
            stage[side]["joint_waypoints_rad"],
            start=1,
        ):
            pose = fk.fk(np.asarray(joints, dtype=float), torso)
            result.append(
                {
                    "stage_id": stage["id"],
                    "waypoint_index": index,
                    "position_base_m": pose.position_base_m.tolist(),
                }
            )
    return result


def actual_trajectory(
    audit: dict[str, Any],
    side: str,
) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": event["stage_id"],
            "phase": event["phase"],
            "sample_index": event["sample_index"],
            "position_base_m": event["position_base_m"],
            "quaternion_xyzw": event["quaternion_xyzw"],
        }
        for event in audit.get("events", [])
        if event.get("event") == "actual_tcp_sample"
        and event.get("side") == side
    ]


def calculated_fingertip_trajectory(
    actual: list[dict[str, Any]],
    tcp_to_fingertip_m: float,
) -> list[dict[str, Any]]:
    result = []
    for item in actual:
        position = np.asarray(item["position_base_m"], dtype=float)
        quaternion = np.asarray(item["quaternion_xyzw"], dtype=float)
        result.append(
            {
                **item,
                "position_base_m": (
                    position
                    + tool_z_axis(quaternion) * float(tcp_to_fingertip_m)
                ).tolist(),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    bundle = json.loads(args.bundle.read_text())
    audit = json.loads(args.execute_audit.read_text())
    side = str(plan["phase_side"])
    targets = [
        {
            "stage_id": stage["id"],
            "position_base_m": bundle_stage[side]["target_position_base_m"],
        }
        for stage, bundle_stage in zip(plan["stages"], bundle["stages"])
        if stage["id"] != "home"
    ]
    actual = actual_trajectory(audit, side)
    payload = {
        "side": side,
        "passed": bool(audit.get("passed")),
        "error": audit.get("error"),
        "towel_cloud": project_mask(args.towel_mask, args.calibration),
        "targets": targets,
        "ik": certified_ik_trajectory(bundle, side),
        "actual": actual,
        "calculated_fingertip": calculated_fingertip_trajectory(
            actual,
            float(plan["safety"]["tcp_to_fingertip_m"]),
        ),
        "towel_keypoints": {
            "detected_center": plan["perception"]["detected_center_base_m"],
            "selected_edge": plan["perception"]["selected_keypoint_base_m"],
            "original_center": plan["original_center_base_m"],
        },
    }
    plotly_source = args.plotly_js.read_text()
    data = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline 7 Towel Execution</title>
<script>{plotly_source}</script></head>
<body><div id="plot" style="width:100vw;height:100vh"></div>
<script>
const data = {data};
function pathTrace(name, items, color, dash) {{
  return {{
    type: "scatter3d", mode: "lines+markers", name,
    x: items.map(v => v.position_base_m[0]),
    y: items.map(v => v.position_base_m[1]),
    z: items.map(v => v.position_base_m[2]),
    text: items.map(v => `${{v.stage_id}} / ${{v.phase || "IK"}} / ${{v.sample_index ?? v.waypoint_index}}`),
    hovertemplate: "%{{text}}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra></extra>",
    line: {{color, width: 7, dash}}, marker: {{color, size: 4}}
  }};
}}
const cloud = {{
  type: "scatter3d", mode: "markers", name: "Towel mask projected to table plane",
  x: data.towel_cloud.map(v => v[0]), y: data.towel_cloud.map(v => v[1]),
  z: data.towel_cloud.map(v => v[2]), hoverinfo: "skip",
  marker: {{color: "#b07d4f", size: 2, opacity: 0.45}}
}};
const targets = {{
  type: "scatter3d", mode: "markers+text", name: "Certified stage targets",
  x: data.targets.map(v => v.position_base_m[0]),
  y: data.targets.map(v => v.position_base_m[1]),
  z: data.targets.map(v => v.position_base_m[2]),
  text: data.targets.map(v => v.stage_id), textposition: "top center",
  marker: {{color: "#111111", size: 7, symbol: "diamond"}}
}};
const names = Object.keys(data.towel_keypoints);
const keypoints = {{
  type: "scatter3d", mode: "markers+text", name: "Towel keypoints",
  x: names.map(n => data.towel_keypoints[n][0]),
  y: names.map(n => data.towel_keypoints[n][1]),
  z: names.map(n => data.towel_keypoints[n][2]),
  text: names, textposition: "bottom center",
  marker: {{color: "#ff8800", size: 8}}
}};
Plotly.newPlot("plot", [
  cloud,
  pathTrace("Certified IK waypoint FK trajectory", data.ik, "#8a2be2", "dot"),
  pathTrace("Measured physical TCP trajectory", data.actual, "#00a83b", "solid"),
  pathTrace("Calculated fingertip center (TCP + tool Z × 78 mm)", data.calculated_fingertip, "#ff3030", "dash"),
  targets,
  keypoints
], {{
  title: `Pipeline 7 ${{data.side}} phase: IK, measured TCP, and calculated fingertip (passed=${{data.passed}})`,
  scene: {{
    xaxis: {{title: "base X (m)"}}, yaxis: {{title: "base Y (m)"}},
    zaxis: {{title: "base Z (m)"}}, aspectmode: "data"
  }}
}}, {{responsive: true}});
</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "side": side,
                "actual_samples": len(payload["actual"]),
                "calculated_fingertip_samples": len(
                    payload["calculated_fingertip"]
                ),
                "ik_samples": len(payload["ik"]),
                "towel_points": len(payload["towel_cloud"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
