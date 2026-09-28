#!/usr/bin/env python3
"""Render left/right gripper_link and constructed tool-Z coordinate frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plotly-js",
        type=Path,
        default=Path("outputs/r1lite_tennis_ball_in_box/plotly.min.js"),
    )
    return parser.parse_args()


def frame_axes(rotation: np.ndarray) -> list[list[float]]:
    return [rotation[:, index].tolist() for index in range(3)]


def main() -> None:
    args = parse_args()
    gripper_rotation = np.eye(3)
    tool_rotation = Rotation.from_euler(
        "xyz",
        [0.0, np.pi / 2.0, 0.0],
    ).as_matrix()
    finger_origins = (
        np.array([0.03689, 0.013453, 0.00012059]),
        np.array([0.03689, -0.013453, -0.00012059]),
    )
    finger_open_translations = (
        np.array([0.0, 0.05, 0.0]),
        np.array([0.0, -0.05, 0.0]),
    )
    finger_bounds = (
        (
            np.array([-0.05049464, -0.06518286, -0.01420414]),
            np.array([0.04146500, 0.00477702, 0.02287093]),
        ),
        (
            np.array([-0.05049463, -0.00477701, -0.02287093]),
            np.array([0.04146500, 0.06518287, 0.01420414]),
        ),
    )

    def finger_model(index: int) -> dict[str, object]:
        lower, upper = finger_bounds[index]
        offset = finger_origins[index] + finger_open_translations[index]
        box_lower = lower + offset
        box_upper = upper + offset
        contact = offset + np.array(
            [
                upper[0],
                (lower[1] + upper[1]) / 2.0,
                (lower[2] + upper[2]) / 2.0,
            ]
        )
        return {
            "name": f"finger_link{index + 1}",
            "local_y_side": "+Y" if index == 0 else "-Y",
            "lower": box_lower.tolist(),
            "upper": box_upper.tolist(),
            "contact_center": contact.tolist(),
        }

    payload = {
        "axis_length_m": 0.10,
        "gripper_state": "fully_open",
        "fingers": [finger_model(0), finger_model(1)],
        "frames": [
            {
                "side": "left",
                "origin": [0.0, 0.15, 0.0],
                "gripper_link_axes": frame_axes(gripper_rotation),
                "tool_z_axes": frame_axes(tool_rotation),
            },
            {
                "side": "right",
                "origin": [0.0, -0.15, 0.0],
                "gripper_link_axes": frame_axes(gripper_rotation),
                "tool_z_axes": frame_axes(tool_rotation),
            },
        ],
        "transform": {
            "xyz_m": [0.0, 0.0, 0.0],
            "rpy_rad": [0.0, float(np.pi / 2.0), 0.0],
            "meaning": "tool +Z equals gripper-link +X physical forward",
        },
    }
    plotly_source = args.plotly_js.read_text()
    data = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>R1 Lite 左右夹爪坐标系</title>
<script>{plotly_source}</script>
<style>
body {{ margin:0; font-family:system-ui,sans-serif; }}
#plot {{ width:100vw; height:76vh; }}
#info {{ padding:12px 20px; line-height:1.55; }}
.x {{color:#e53935}} .y {{color:#43a047}} .z {{color:#1e88e5}}
code {{background:#eee;padding:2px 5px;border-radius:4px}}
</style></head><body>
<div id="plot"></div>
<div id="info">
<b>模型坐标系定义图，不代表当前真机在 base_link 中的实时位置。</b><br>
左右臂参数一致且几何对称。两侧均使用
<code>T_gripper_link_tool_z = xyz[0,0,0], rpy[0,+π/2,0]</code>。<br>
因此Tool的<span class="z">+Z蓝轴</span>与夹爪物理前向
（gripper_link的<span class="x">+X红轴</span>）重合。
实线为Tool-Z坐标系，灰色虚线为原始gripper_link坐标系。夹爪显示为全开状态。<br>
紫色为<code>finger_link1（局部+Y）</code>，橙色为
<code>finger_link2（局部-Y）</code>；球形标记是用于标定的指尖前端中心。
</div>
<script>
const model = {data};
const colors = ["#e53935", "#43a047", "#1e88e5"];
const axisNames = ["X", "Y", "Z"];
const traces = [];
const boxFaces = {{
  i:[0,0,0,1,1,2,3,4,4,5,6,6],
  j:[1,2,4,2,5,3,7,5,6,6,7,4],
  k:[2,4,1,5,2,7,4,6,5,7,4,7]
}};
function addFinger(origin, finger, color, side) {{
  const lo=finger.lower, hi=finger.upper;
  const vertices=[
    [lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],
    [lo[0],hi[1],lo[2]],[hi[0],hi[1],lo[2]],
    [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],
    [lo[0],hi[1],hi[2]],[hi[0],hi[1],hi[2]]
  ].map(v=>v.map((value,index)=>value+origin[index]));
  traces.push({{
    type:"mesh3d",name:`${{side}} ${{finger.name}} (${{finger.local_y_side}})`,
    x:vertices.map(v=>v[0]),y:vertices.map(v=>v[1]),z:vertices.map(v=>v[2]),
    i:boxFaces.i,j:boxFaces.j,k:boxFaces.k,color,opacity:0.55,
    flatshading:true
  }});
  const contact=finger.contact_center.map((value,index)=>value+origin[index]);
  traces.push({{
    type:"scatter3d",mode:"markers+text",
    name:`${{side}} ${{finger.name}} contact`,
    x:[contact[0]],y:[contact[1]],z:[contact[2]],
    text:[`${{finger.name}}<br>${{finger.local_y_side}}<br>${{
      (finger.contact_center[0]*1000).toFixed(1)}},${{
      (finger.contact_center[1]*1000).toFixed(1)}},${{
      (finger.contact_center[2]*1000).toFixed(1)}} mm`],
    textposition:"top center",marker:{{color,size:8,symbol:"circle"}}
  }});
}}
function addAxis(origin, vector, length, name, color, dash, width) {{
  const end = origin.map((value, index) => value + vector[index] * length);
  traces.push({{
    type:"scatter3d", mode:"lines+markers+text", name,
    x:[origin[0], end[0]], y:[origin[1], end[1]], z:[origin[2], end[2]],
    text:["", name], textposition:"top center",
    line:{{color, width, dash}}, marker:{{color, size:[2,5]}}
  }});
}}
for (const frame of model.frames) {{
  traces.push({{
    type:"scatter3d", mode:"markers+text", name:`${{frame.side}} origin`,
    x:[frame.origin[0]], y:[frame.origin[1]], z:[frame.origin[2]],
    text:[`${{frame.side}}_gripper_tool_z`], textposition:"bottom center",
    marker:{{color:"#111",size:7}}
  }});
  frame.gripper_link_axes.forEach((axis,index) =>
    addAxis(frame.origin, axis, model.axis_length_m * 0.75,
      `${{frame.side}} gripper_link +${{axisNames[index]}}`,
      "#888", "dash", 4));
  frame.tool_z_axes.forEach((axis,index) =>
    addAxis(frame.origin, axis, model.axis_length_m,
      `${{frame.side}} tool +${{axisNames[index]}}`,
      colors[index], "solid", 8));
  addFinger(frame.origin, model.fingers[0], "#8e24aa", frame.side);
  addFinger(frame.origin, model.fingers[1], "#fb8c00", frame.side);
}}
Plotly.newPlot("plot", traces, {{
  title:"R1 Lite 左右夹爪全开模型、指尖编号与 tool_z 坐标系",
  legend:{{orientation:"h"}},
  scene:{{
    xaxis:{{title:"示意 X (m)"}},
    yaxis:{{title:"示意 Y (m)"}},
    zaxis:{{title:"示意 Z (m)"}},
    aspectmode:"data",
    camera:{{eye:{{x:1.5,y:1.5,z:1.2}}}}
  }}
}}, {{responsive:true}});
</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)


if __name__ == "__main__":
    main()
