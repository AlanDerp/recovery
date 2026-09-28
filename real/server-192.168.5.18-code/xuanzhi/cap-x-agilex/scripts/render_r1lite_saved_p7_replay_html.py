#!/usr/bin/env python3
"""Render saved Pipeline-7 replay trajectories as an interactive 3D page."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.kinematics import R1LiteCandidateKinematics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plotly-js",
        type=Path,
        default=Path("outputs/r1lite_tennis_ball_in_box/plotly.min.js"),
    )
    return parser.parse_args()


def fk_point(
    fk: R1LiteCandidateKinematics,
    torso: np.ndarray,
    joints: list[float],
    **metadata: Any,
) -> dict[str, Any]:
    pose = fk.fk(np.asarray(joints, dtype=float), torso)
    return {
        "position_base_m": pose.position_base_m.tolist(),
        **metadata,
    }


def append_unique(
    items: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    tolerance_m: float = 1e-6,
) -> None:
    if items:
        previous = np.asarray(items[-1]["position_base_m"], dtype=float)
        current = np.asarray(item["position_base_m"], dtype=float)
        if float(np.linalg.norm(current - previous)) <= tolerance_m:
            return
    items.append(item)


def build_payload(
    plan: dict[str, Any],
    bundle: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    fk = R1LiteCandidateKinematics(side="right")
    torso = np.asarray(bundle["initial_state"]["torso_joints_rad"], dtype=float)
    desired = [
        fk_point(
            fk,
            torso,
            bundle["initial_state"]["right_joints_rad"],
            label="initial",
        )
    ]
    for stage in bundle["stages"]:
        for waypoint_index, joints in enumerate(
            stage["right"]["joint_waypoints_rad"],
            start=1,
        ):
            desired.append(
                fk_point(
                    fk,
                    torso,
                    joints,
                    label=f"{stage['id']} / waypoint {waypoint_index}",
                )
            )

    feedback_fk: list[dict[str, Any]] = []
    for trace_index, trace in enumerate(audit.get("joint_arrival_trace", []), start=1):
        final = trace.get("final_feedback_rad", {}).get("right")
        if not isinstance(final, list):
            continue
        append_unique(
            feedback_fk,
            fk_point(
                fk,
                torso,
                final,
                label=(
                    f"arrival {trace_index} / "
                    f"{trace.get('result', 'unknown')}"
                ),
            ),
        )

    vendor_tcp: list[dict[str, Any]] = []
    contact_points: list[dict[str, Any]] = []
    downward_targets: list[dict[str, Any]] = []
    for event in audit.get("events", []):
        if event.get("event") != "feedback_downward_table_refinement_completed":
            continue
        for refinement in event.get("refinements", []):
            before = refinement.get("before_tcp_position_base_m")
            target = refinement.get("target_tcp_position_base_m")
            after = refinement.get("after_tcp_position_base_m")
            if isinstance(before, list):
                append_unique(
                    vendor_tcp,
                    {
                        "position_base_m": before,
                        "label": "refinement start",
                    },
                )
            if isinstance(target, list):
                downward_targets.append(
                    {
                        "position_base_m": target,
                        "label": "requested Cartesian target",
                    }
                )
            for sample in refinement.get("contact_monitor_samples", []):
                position = sample.get("tcp_position_base_m")
                if not isinstance(position, list):
                    continue
                item = {
                    "position_base_m": position,
                    "label": (
                        f"contact sample {sample.get('sample_index')} / "
                        f"score={sample.get('joint_residual_score_nm', 0):.3f} Nm"
                    ),
                }
                append_unique(vendor_tcp, item, tolerance_m=1e-5)
                if sample.get("detected") is True:
                    contact_points.append(item)
            if isinstance(after, list):
                append_unique(
                    vendor_tcp,
                    {
                        "position_base_m": after,
                        "label": "refinement stop",
                    },
                )

    contour = []
    for candidate in plan.get("perception", {}).get(
        "right_side_grasp_candidates", []
    ):
        point = candidate.get("contour_base_xyz_m")
        if isinstance(point, list):
            contour.append(
                {
                    "position_base_m": point,
                    "label": candidate.get("id", "contour"),
                }
            )
    keypoints = [
        {
            "position_base_m": plan["perception"]["selected_keypoint_base_m"],
            "label": "selected grasp keypoint",
        },
        {
            "position_base_m": plan["perception"]["detected_center_base_m"],
            "label": "detected towel center",
        },
        {
            "position_base_m": plan["original_center_base_m"],
            "label": "original towel center",
        },
    ]
    return {
        "passed": audit.get("passed"),
        "desired": desired,
        "feedback_fk": feedback_fk,
        "vendor_tcp": vendor_tcp,
        "downward_targets": downward_targets,
        "contact_points": contact_points,
        "contour": contour,
        "keypoints": keypoints,
    }


def render(payload: dict[str, Any], plotly_source: str) -> str:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline 7 Replay Trajectory</title>
<script>{plotly_source}</script></head>
<body style="margin:0"><div id="plot" style="width:100vw;height:100vh"></div>
<script>
const data = {data};
function trace(name, items, color, dash, width=6, size=4) {{
  return {{
    type:"scatter3d", mode:"lines+markers", name,
    x:items.map(v=>v.position_base_m[0]),
    y:items.map(v=>v.position_base_m[1]),
    z:items.map(v=>v.position_base_m[2]),
    text:items.map(v=>v.label),
    hovertemplate:"%{{text}}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra></extra>",
    line:{{color,width,dash}}, marker:{{color,size}}
  }};
}}
const traces = [
  trace("Plan / local FK", data.desired, "#1f77b4", "dash"),
  trace("Feedback joints / local FK", data.feedback_fk, "#ff7f0e", "solid"),
  trace("Vendor measured TCP during descent", data.vendor_tcp, "#d62728", "solid", 8, 5),
  trace("Requested descent target", data.downward_targets, "#9467bd", "dot", 3, 7),
  trace("Towel right contour", data.contour, "#2ca02c", "solid", 4, 3),
  trace("Towel keypoints", data.keypoints, "#17becf", "dot", 2, 8),
  trace("Effort contact detected", data.contact_points, "#000000", "solid", 2, 10)
];
Plotly.newPlot("plot", traces, {{
  title:`Saved Pipeline-7 replay (passed=${{data.passed}})`,
  scene:{{
    xaxis:{{title:"base X (m)"}},
    yaxis:{{title:"base Y (m)"}},
    zaxis:{{title:"base Z (m)"}},
    aspectmode:"data"
  }},
  legend:{{orientation:"h"}},
  margin:{{l:0,r:0,b:0,t:60}}
}}, {{responsive:true}});
</script></body></html>"""


def main() -> None:
    args = parse_args()
    payload = build_payload(
        json.loads(args.plan.read_text()),
        json.loads(args.bundle.read_text()),
        json.loads(args.audit.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(payload, args.plotly_js.read_text()),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
