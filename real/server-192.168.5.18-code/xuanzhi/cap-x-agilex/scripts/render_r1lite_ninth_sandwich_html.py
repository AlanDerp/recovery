#!/usr/bin/env python3
"""Render a self-contained pipeline-nine perception and trajectory report."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-bundle", type=Path)
    parser.add_argument("--certification-audit", type=Path)
    parser.add_argument("--execution-audit", type=Path)
    parser.add_argument("--preview-image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _image_uri(path: Path | None) -> str | None:
    if path is None:
        return None
    suffix = path.suffix.lower().lstrip(".")
    media = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    return f"data:image/{media};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _plotly_source() -> str:
    candidates = (
        Path("outputs/r1lite_tennis_ball_in_box/plotly.min.js"),
        Path(__file__).resolve().parents[1]
        / "outputs/r1lite_tennis_ball_in_box/plotly.min.js",
    )
    for path in candidates:
        if path.is_file():
            return path.read_text()
    raise FileNotFoundError("local plotly.min.js is unavailable")


def _actual_tcp_points(audit: dict[str, Any] | None) -> dict[str, list[list[float]]]:
    result = {"left": [], "right": []}
    if audit is None:
        return result
    for event in audit.get("events", []):
        if event.get("event") != "actual_tcp_sample":
            continue
        side = event.get("side")
        position = event.get("position_base_m") or event.get("actual_tcp_position_base_m")
        if side in result and isinstance(position, list) and len(position) == 3:
            result[side].append([float(value) for value in position])
    return result


def render(
    *,
    plan: dict[str, Any],
    bundle: dict[str, Any] | None,
    certification: dict[str, Any] | None,
    execution: dict[str, Any] | None,
    preview_uri: str | None,
) -> str:
    scene = plan["perception"]["scene"]
    objects = scene["objects"]
    target_stages = [
        {
            "id": stage["id"],
            "point": stage.get("active_tcp_position_base_m"),
        }
        for stage in plan["stages"]
        if stage["id"] != "home"
    ]
    bundle_trajectories = {"left": [], "right": []}
    if bundle is not None:
        for stage in bundle.get("stages", []):
            for side in ("left", "right"):
                point = stage.get(side, {}).get("fk_position_base_m")
                if isinstance(point, list) and len(point) == 3:
                    bundle_trajectories[side].append(
                        {"stage": stage["id"], "point": point}
                    )
    payload = {
        "pipeline": 9,
        "phase": plan["phase"],
        "selectedSide": plan.get("selected_side", "not selected"),
        "objects": objects,
        "tablePlane": scene["table_plane"],
        "targetStages": target_stages,
        "bundleTrajectories": bundle_trajectories,
        "actualTcp": _actual_tcp_points(execution),
        "armCandidates": (
            {} if certification is None else certification.get("arm_candidates", {})
        ),
        "executionPassed": (
            None if execution is None else execution.get("passed")
        ),
        "homeCompleted": (
            None if execution is None else execution.get("home_completed")
        ),
    }
    preview_html = (
        '<p class="muted">未提供二维预览图。</p>'
        if preview_uri is None
        else f'<img class="preview" src="{preview_uri}" alt="RGB-D mask preview">'
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R1 Lite 第九三明治流水线</title>
<style>
body{{margin:0;background:#111827;color:#e5e7eb;font-family:system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:20px}}
.grid{{display:grid;grid-template-columns:minmax(360px,42%) 1fr;gap:18px}}
.card{{background:#1f2937;border:1px solid #374151;border-radius:10px;padding:14px}}
.preview{{width:100%;height:auto;border-radius:6px}}
#plot{{height:720px}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #374151;padding:7px;text-align:left}}
.muted{{color:#9ca3af}} code{{color:#93c5fd}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}#plot{{height:540px}}}}
</style>
<script>{_plotly_source()}</script>
</head>
<body><main>
<h1>R1 Lite 第九三明治流水线</h1>
<p>阶段：<code id="phase"></code>　选中机械臂：<code id="side"></code>　
执行通过：<code id="passed"></code>　回到认证初始姿态：<code id="home"></code></p>
<div class="grid">
  <section class="card"><h2>真实 RGB 与感知掩码</h2>{preview_html}</section>
  <section class="card"><h2>三维物体与 TCP 轨迹</h2><div id="plot"></div></section>
</div>
<section class="card"><h2>自动选臂证据</h2><table>
<thead><tr><th>机械臂</th><th>通过</th><th>评分</th><th>最大 FK 误差</th><th>最小扫掠间距</th></tr></thead>
<tbody id="scores"></tbody></table></section>
<script>
const data={json.dumps(payload, ensure_ascii=False)};
document.getElementById("phase").textContent=data.phase;
document.getElementById("side").textContent=data.selectedSide;
document.getElementById("passed").textContent=String(data.executionPassed);
document.getElementById("home").textContent=String(data.homeCompleted);
const colors={{brown_bread:"#b7791f",donut:"#ef4444",mixed_bread:"#22c55e"}};
const traces=[];
for(const [name,item] of Object.entries(data.objects)){{
  const p=item.center_base_m;
  traces.push({{type:"scatter3d",mode:"markers+text",name,
    x:[p[0]],y:[p[1]],z:[p[2]],text:[name],textposition:"top center",
    marker:{{size:9,color:colors[name]}}}});
  const a=item.major_axis_base_xy;
  const half=item.length_m/2;
  traces.push({{type:"scatter3d",mode:"lines",showlegend:false,
    x:[p[0]-a[0]*half,p[0]+a[0]*half],
    y:[p[1]-a[1]*half,p[1]+a[1]*half],z:[p[2],p[2]],
    line:{{width:7,color:colors[name]}}}});
}}
const targets=data.targetStages.filter(v=>v.point);
traces.push({{type:"scatter3d",mode:"lines+markers+text",name:"计划 TCP",
 x:targets.map(v=>v.point[0]),y:targets.map(v=>v.point[1]),z:targets.map(v=>v.point[2]),
 text:targets.map(v=>v.id),marker:{{size:5,color:"#facc15"}},line:{{color:"#facc15"}}}});
for(const side of ["left","right"]){{
 const bundle=data.bundleTrajectories[side];
 if(bundle.length)traces.push({{type:"scatter3d",mode:"lines+markers",name:`${{side}} 认证 FK`,
   x:bundle.map(v=>v.point[0]),y:bundle.map(v=>v.point[1]),z:bundle.map(v=>v.point[2]),
   text:bundle.map(v=>v.stage),line:{{dash:"dot"}}}});
 const actual=data.actualTcp[side];
 if(actual.length)traces.push({{type:"scatter3d",mode:"lines+markers",name:`${{side}} 实际 TCP`,
   x:actual.map(v=>v[0]),y:actual.map(v=>v[1]),z:actual.map(v=>v[2])}});
}}
Plotly.newPlot("plot",traces,{{paper_bgcolor:"#1f2937",plot_bgcolor:"#1f2937",
 font:{{color:"#e5e7eb"}},margin:{{l:0,r:0,t:25,b:0}},
 scene:{{aspectmode:"data",xaxis:{{title:"base X (m)"}},yaxis:{{title:"base Y (m)"}},
 zaxis:{{title:"base Z (m)"}}}}}},{{responsive:true}});
const tbody=document.getElementById("scores");
for(const side of ["left","right"]){{
 const v=data.armCandidates[side]||{{}};
 const row=document.createElement("tr");
 row.innerHTML=`<td>${{side}}</td><td>${{String(v.passed)}}</td>
 <td>${{v.score===undefined?"—":Number(v.score).toFixed(4)}}</td>
 <td>${{v.maximum_fk_position_error_m===undefined?"—":(v.maximum_fk_position_error_m*1000).toFixed(1)+" mm"}}</td>
 <td>${{v.minimum_swept_link_clearance_m===undefined?"—":(v.minimum_swept_link_clearance_m*1000).toFixed(1)+" mm"}}</td>`;
 tbody.appendChild(row);
}}
</script></main></body></html>"""


def main() -> None:
    args = parse_args()
    html = render(
        plan=_load(args.plan) or {},
        bundle=_load(args.execution_bundle),
        certification=_load(args.certification_audit),
        execution=_load(args.execution_audit),
        preview_uri=_image_uri(args.preview_image),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
