#!/usr/bin/env python3
"""Render a static Plotly page for one three-cup motion attempt."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def actual_stage_endpoints(
    audit: dict, home: list[float]
) -> tuple[list[list[float]], list[str]]:
    """Extract measured Cartesian stage endpoints recorded during execution."""
    all_events = audit.get("events", [])
    stage_events = [
        event for event in all_events if event.get("event") == "execute_cup_stage"
    ]
    points = [home]
    labels = ["Home（实测起点）"]
    for event in all_events:
        if event.get("event") == "resume_after_first_grasp_gate":
            measured = event.get("current_tcp_base_m")
            if isinstance(measured, list) and len(measured) == 3:
                points.append(measured)
                labels.append("续跑真实起点")
            continue
        if event.get("event") == "return_home":
            points.append(home)
            labels.append("返回 Home")
            continue
        if event.get("event") != "execute_cup_stage":
            continue
        result = event.get("result") or {}
        measured = result.get("measured_position")
        inferred = False
        if measured is None and event.get("stage_id") == "grasp":
            stage_index = stage_events.index(event)
            for later_event in stage_events[stage_index + 1 :]:
                later_plan = (later_event.get("result") or {}).get("plan") or {}
                measured = later_plan.get("start_position")
                if measured is not None:
                    inferred = True
                    break
        if not isinstance(measured, list) or len(measured) != 3:
            continue
        points.append(measured)
        label = f"{event.get('phase')} / {event.get('stage_id')}（实测）"
        if inferred:
            label += "，由下一阶段起点记录"
        labels.append(label)
    return points, labels


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    audits = [json.loads(path.read_text()) for path in args.audit]
    audit = dict(audits[-1])
    audit["events"] = [
        event for item in audits for event in item.get("events", [])
    ]
    cups = plan["rgbd_scene"]["cups"]
    phases = plan["phases"]
    home = plan["rgbd_scene"]["home_gripper_position_base_m"]

    traces = []
    for name, color in (
        ("orange cup", "#f97316"),
        ("purple cup", "#a855f7"),
        ("green cup", "#22c55e"),
    ):
        cup = cups[name]
        rim = cup["rim_center_base_m"]
        top = cup["top_center_base_m"]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers+text",
                "name": name,
                "x": [rim[0], top[0]],
                "y": [rim[1], top[1]],
                "z": [rim[2], top[2]],
                "text": ["rim", name],
                "textposition": "top center",
                "line": {"color": color, "width": 12},
                "marker": {"color": color, "size": 7},
            }
        )

    for phase, color in zip(phases, ("#fb923c", "#4ade80"), strict=True):
        points = [
            item["tcp_position_base_m"]
            for item in phase["waypoints"]
            if item["id"] != "place"
        ]
        release = phase["release_hover_tcp_position_base_m"]
        points.insert(4, release)
        labels = [
            "pregrasp",
            "grasp",
            "lift",
            "above support",
            "gravity release",
            "retract",
        ]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": f"planned only: {phase['phase']}",
                "x": [point[0] for point in points],
                "y": [point[1] for point in points],
                "z": [point[2] for point in points],
                "text": labels,
                "hovertemplate": "%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}"
                "<br>z=%{z:.3f}<extra></extra>",
                "line": {"color": color, "width": 5, "dash": "dash"},
                "marker": {"color": color, "size": 4},
                "visible": True,
            }
        )

    actual_points, actual_labels = actual_stage_endpoints(audit, home)
    if len(actual_points) > 1:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": "实际运动轨迹（实测阶段端点）",
                "x": [point[0] for point in actual_points],
                "y": [point[1] for point in actual_points],
                "z": [point[2] for point in actual_points],
                "text": actual_labels,
                "hovertemplate": "%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}"
                "<br>z=%{z:.3f}<extra></extra>",
                "line": {"color": "#22d3ee", "width": 9},
                "marker": {
                    "color": "#22d3ee",
                    "size": 7,
                    "symbol": "diamond",
                },
            }
        )

    if (
        audit.get("passed")
        and audit.get("completion_scope") == "first_grasp_only"
        and len(actual_points) > 1
    ):
        first = phases[0]
        second = phases[1]

        def phase_point(phase: dict, stage_id: str) -> list[float]:
            return next(
                item["tcp_position_base_m"]
                for item in phase["waypoints"]
                if item["id"] == stage_id
            )

        remaining_points = [
            actual_points[-1],
            phase_point(first, "above_support"),
            first["release_hover_tcp_position_base_m"],
            phase_point(first, "retract"),
            phase_point(second, "pregrasp"),
            phase_point(second, "grasp"),
            phase_point(second, "lift"),
            phase_point(second, "above_support"),
            second["release_hover_tcp_position_base_m"],
            phase_point(second, "retract"),
            home,
        ]
        remaining_labels = [
            "当前实际位置（橙杯已夹持）",
            "橙杯：目标杯上方",
            "橙杯：重力释放点",
            "橙杯：释放后撤回",
            "绿杯：pregrasp",
            "绿杯：grasp",
            "绿杯：lift",
            "绿杯：目标杯上方",
            "绿杯：重力释放点",
            "绿杯：释放后撤回",
            "返回本次 Home",
        ]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": "剩余未执行计划",
                "x": [point[0] for point in remaining_points],
                "y": [point[1] for point in remaining_points],
                "z": [point[2] for point in remaining_points],
                "text": remaining_labels,
                "hovertemplate": "%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}"
                "<br>z=%{z:.3f}<extra></extra>",
                "line": {"color": "#facc15", "width": 8, "dash": "dot"},
                "marker": {
                    "color": "#facc15",
                    "size": 6,
                    "symbol": "square",
                },
            }
        )

    error = audit.get("error", "unknown")
    payload = json.dumps(traces, ensure_ascii=False)
    title = "R1 Lite 三杯任务：规划轨迹与实际轨迹"
    if audit.get("passed") and audit.get("completion_scope") == "first_grasp_only":
        status = (
            "固定橙杯抓取、抬升和夹持确认已完成，并按要求停止。"
            " 青色实线为实际轨迹，黄色点线为剩余未执行计划，"
            "彩色虚线为完整任务规划轨迹。"
        )
    else:
        status = (
            "任务执行已中止。青色实线为实际记录的阶段端点，"
            "虚线为完整任务规划轨迹。"
        )
    output_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin:0; background:#0f172a; color:#e2e8f0; font-family:sans-serif; }}
    #status {{ padding:12px 18px; background:#7f1d1d; }}
    #plot {{ width:100vw; height:calc(100vh - 72px); }}
  </style>
</head>
<body>
  <div id="status"><b>{status}</b> {html.escape(str(error))}</div>
  <div id="plot"></div>
  <script>
    const traces = {payload};
    Plotly.newPlot("plot", traces, {{
      title: {{text: "{title}", font: {{color:"#e2e8f0"}}}},
      paper_bgcolor:"#0f172a", plot_bgcolor:"#0f172a",
      legend: {{font:{{color:"#e2e8f0"}}}},
      scene: {{
        aspectmode:"data",
        xaxis:{{title:"base X (m)", color:"#e2e8f0"}},
        yaxis:{{title:"base Y (m)", color:"#e2e8f0"}},
        zaxis:{{title:"base Z (m)", color:"#e2e8f0"}}
      }}
    }}, {{responsive:true}});
  </script>
</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_html)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
