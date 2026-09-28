#!/usr/bin/env python3
"""Render reconstructed pose-1/lift/forward trajectories with towel cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from render_r1lite_sequential_towel_execution_html import project_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pose-1", type=Path, required=True)
    parser.add_argument("--towel-plan", type=Path, required=True)
    parser.add_argument("--towel-mask", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plotly-js",
        type=Path,
        default=Path("outputs/r1lite_tennis_ball_in_box/plotly.min.js"),
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    pose_1 = json.loads(args.pose_1.read_text())
    towel_plan = json.loads(args.towel_plan.read_text())
    lift = float(audit["lift_result"]["achieved_lift_m"])
    forward_events = [
        event
        for event in audit["events"]
        if str(event.get("id", "")).startswith("permissive_forward_")
    ]
    lift_ids = []
    for event in audit["events"]:
        stage_id = str(event.get("id", ""))
        if stage_id.startswith("lift_") and stage_id not in lift_ids:
            lift_ids.append(stage_id)

    paths = {}
    for side in ("left", "right"):
        origin = pose_1["arms"][side]["tcp_position_base_m"]
        points = [{"label": "pose_1 / grasp", "position": origin}]
        for index, stage_id in enumerate(lift_ids, start=1):
            points.append(
                {
                    "label": stage_id,
                    "position": [
                        origin[0],
                        origin[1],
                        origin[2] + lift * index / max(1, len(lift_ids)),
                    ],
                }
            )
        for event in forward_events:
            points.append(
                {
                    "label": event["id"],
                    "position": [
                        origin[0] + float(event["achieved_forward_m"]),
                        origin[1],
                        origin[2] + lift,
                    ],
                }
            )
        paths[side] = points

    payload = {
        "audit": {
            "passed": audit["passed"],
            "achieved_lift_m": lift,
            "achieved_forward_m": audit["forward_result"][
                "achieved_forward_m"
            ],
            "trajectory_kind": (
                "reconstructed from pose 1 plus measured scalar displacement"
            ),
        },
        "paths": paths,
        "towel_cloud": project_mask(args.towel_mask, args.calibration),
        "towel_center": towel_plan["original_center_base_m"],
    }
    plotly = args.plotly_js.read_text()
    data = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>R1 Lite Lift Forward Trajectory</title>
<script>{plotly}</script></head>
<body><div id="plot" style="width:100vw;height:100vh"></div><script>
const d={data};
function path(side,color) {{
 const p=d.paths[side]; return {{type:"scatter3d",mode:"lines+markers",
 name:`${{side}} TCP reconstructed trajectory`,x:p.map(v=>v.position[0]),
 y:p.map(v=>v.position[1]),z:p.map(v=>v.position[2]),text:p.map(v=>v.label),
 hovertemplate:"%{{text}}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra></extra>",
 line:{{color,width:7}},marker:{{color,size:4}}}}; }}
const cloud={{type:"scatter3d",mode:"markers",name:"Recognized towel point cloud",
 x:d.towel_cloud.map(v=>v[0]),y:d.towel_cloud.map(v=>v[1]),z:d.towel_cloud.map(v=>v[2]),
 marker:{{color:"#b07d4f",size:2,opacity:.45}},hoverinfo:"skip"}};
const c=d.towel_center;
const center={{type:"scatter3d",mode:"markers+text",name:"Recognized towel center",
 x:[c[0]],y:[c[1]],z:[c[2]],text:["towel center"],textposition:"top center",
 marker:{{color:"#ff8800",size:9,symbol:"diamond"}}}};
Plotly.newPlot("plot",[cloud,path("left","#0066ff"),path("right","#e53935"),center],{{
 title:`23:15 run — reconstructed TCP trajectories; lift=${{(d.audit.achieved_lift_m*100).toFixed(2)}}cm, forward=${{(d.audit.achieved_forward_m*100).toFixed(2)}}cm`,
 scene:{{xaxis:{{title:"base X (m)"}},yaxis:{{title:"base Y (m)"}},zaxis:{{title:"base Z (m)"}},aspectmode:"data"}},
 annotations:[{{text:d.audit.trajectory_kind,showarrow:false,x:0.5,y:1.02,xref:"paper",yref:"paper"}}]
}},{{responsive:true}});
</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "towel_points": len(payload["towel_cloud"]),
                "left_points": len(paths["left"]),
                "right_points": len(paths["right"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
