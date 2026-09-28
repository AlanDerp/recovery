#!/usr/bin/env python3
"""Render an R1 Lite evaluation JSONL as a standalone 3D trajectory page."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import shutil
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def xyz(point: Any) -> list[float]:
    values = [float(value) for value in point]
    if len(values) != 3:
        raise ValueError(f"Expected XYZ point, got {point!r}")
    return values


def json_for_inline_script(value: Any) -> str:
    """Serialize JSON without permitting an embedded </script> terminator."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def trace(
    name: str,
    points: list[list[float]],
    *,
    color: str,
    mode: str = "lines+markers",
    width: int = 5,
    size: int = 5,
    dash: str = "solid",
    text: list[str] | None = None,
    symbol: str = "circle",
) -> dict[str, Any]:
    return {
        "type": "scatter3d",
        "mode": mode,
        "name": name,
        "x": [point[0] for point in points],
        "y": [point[1] for point in points],
        "z": [point[2] for point in points],
        "text": text,
        "line": {"color": color, "width": width, "dash": dash},
        "marker": {"color": color, "size": size, "symbol": symbol},
        "hovertemplate": (
            "%{text}<br>x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m"
            "<extra></extra>"
        ),
    }


def parse_rejected_fk(error: str) -> tuple[list[float] | None, list[float] | None]:
    target_match = re.search(r"target=(\[[^\]]+\])", error)
    fk_match = re.search(r"fk_position=(\[[^\]]+\])", error)
    target = json.loads(target_match.group(1)) if target_match else None
    fk = json.loads(fk_match.group(1)) if fk_match else None
    return (xyz(target) if target is not None else None, xyz(fk) if fk is not None else None)


def build_page(records: list[dict[str, Any]], source: Path) -> tuple[list[dict[str, Any]], str]:
    if not records or records[0].get("event") != "start":
        raise ValueError("Evaluation log must begin with a start event")
    start = records[0]
    plan_path = Path(start["plan"])
    plan = json.loads(plan_path.read_text())
    objects = plan.get("objects", {})

    desired: list[list[float]] = []
    desired_text: list[str] = []
    llm_anchors: list[list[float]] = []
    llm_anchor_text: list[str] = []
    interpolation_targets: list[list[float]] = []
    interpolation_text: list[str] = []
    coupled_targets: list[list[float]] = []
    coupled_target_text: list[str] = []
    desired_fingertips: list[list[float]] = []
    desired_fingertip_text: list[str] = []
    actual_fingertips: list[list[float]] = []
    actual_fingertip_text: list[str] = []
    coupled_axis_points: list[list[float]] = []
    coupled_axis_text: list[str] = []
    coupled_resolution: dict[str, Any] | None = None
    ik_requests: list[list[float]] = []
    ik_text: list[str] = []
    ik_solution_fk: list[list[float]] = []
    ik_solution_fk_text: list[str] = []
    actual_joint_fk: list[list[float]] = []
    actual_joint_fk_text: list[str] = []
    solver_fk: list[list[float]] = []
    solver_fk_text: list[str] = []
    actual = [xyz(start["initial_physical_position_base_m"])]
    actual_text = ["initial physical tool-Z TCP"]
    error_segment_points: list[list[float] | None] = []
    error_label_points: list[list[float]] = []
    error_label_text: list[str] = []
    error_label_hover: list[str] = []
    rejected_target: list[list[float]] = []
    rejected_target_text: list[str] = []
    rejected_fk: list[list[float]] = []
    rejected_fk_text: list[str] = []
    joint_rows: list[dict[str, Any]] = []
    reached: list[str] = []
    failure = "log ended without a finished event"

    for record in records[1:]:
        event = record.get("event")
        frame = record.get("frame_index")
        identifier = str(record.get("id", ""))
        if event == "frame_begin":
            point = xyz(record["desired_position_base_m"])
            role = str(record.get("semantic_role", "legacy"))
            source_name = str(record.get("target_source", "fixed_plan"))
            desired.append(point)
            desired_text.append(
                f"F{frame} {identifier} — 执行目标; role={role}; "
                f"source={source_name}"
            )
            if source_name == "llm_runtime_anchor":
                llm_anchors.append(point)
                llm_anchor_text.append(
                    f"F{frame} {identifier} — LLM语义锚点; role={role}"
                )
            elif source_name in {
                "system_interpolation",
                "runtime_interpolation",
            }:
                interpolation_targets.append(point)
                interpolation_text.append(
                    f"F{frame} {identifier} — {source_name}; role={role}"
                )
            elif source_name == "system_coupled_grasp_resolved":
                coupled_targets.append(point)
                coupled_target_text.append(
                    f"F{frame} {identifier} — 姿态耦合TCP目标; role={role}"
                )
            fingertip = record.get("desired_fingertip_center_base_m")
            if fingertip is not None:
                desired_fingertips.append(xyz(fingertip))
                desired_fingertip_text.append(
                    f"F{frame} {identifier} — 目标指尖抓取中心; role={role}"
                )
        elif event == "coupled_grasp_chain_resolved":
            coupled_resolution = record
            stage_targets = record.get("stage_tcp_targets_base_m", {})
            grasp_tcp = stage_targets.get("grasp")
            ball_center = record.get("ball_center_base_m")
            if grasp_tcp is not None and ball_center is not None:
                coupled_axis_points = [xyz(grasp_tcp), xyz(ball_center)]
                coupled_axis_text = [
                    "最终grasp TCP",
                    (
                        "沿锁定tool +Z前进"
                        f"{1000.0 * float(record['tcp_to_fingertip_m']):.1f}mm"
                        "后的指尖中心/球心"
                    ),
                ]
        elif event == "frame_a0_cartesian_absolute_step_target":
            solver_fk.append(xyz(record["candidate_fk_validation"]["fk_position_base_m"]))
            diag = record["candidate_fk_validation"]
            solver_fk_text.append(
                f"F{frame} {identifier} — solver FK; yaw={diag['selected_yaw_deg']}°; "
                f"residual={1000.0 * float(diag['fk_error_m']):.1f} mm"
            )
        elif event == "frame_attempt":
            attempt = int(record["attempt"])
            pipeline = str(record.get("a0_pipeline", ""))
            is_online_ik = pipeline in {
                "cartesian_absolute_step",
                "closed_loop_correction",
            }
            command = record.get("command_result", {})
            decomposition = command.get(
                "kinematic_execution_decomposition", {}
            )
            if is_online_ik:
                ik_requests.append(
                    xyz(record["compensated_target_position_base_m"])
                )
                ik_text.append(
                    f"F{frame} A{attempt} {identifier} — IK输入目标; "
                    f"pipeline={pipeline}"
                )
                command_fk = decomposition.get(
                    "ik_command_fk_position_base_m"
                )
                if command_fk is not None:
                    ik_solution_fk.append(xyz(command_fk))
                    ik_solution_fk_text.append(
                        f"F{frame} A{attempt} {identifier} — FK(q_IK); "
                        f"距厂商TCP="
                        f"{1000.0 * float(decomposition['ik_command_fk_to_vendor_error_m']):.1f} mm"
                    )
                measured_fk = decomposition.get(
                    "actual_joint_fk_position_base_m"
                )
                if measured_fk is not None:
                    actual_joint_fk.append(xyz(measured_fk))
                    actual_joint_fk_text.append(
                        f"F{frame} A{attempt} {identifier} — FK(q_actual); "
                        f"距厂商TCP="
                        f"{1000.0 * float(decomposition['vendor_vs_actual_joint_fk_error_m']):.1f} mm"
                    )
            actual.append(xyz(record["actual_position_base_m"]))
            actual_text.append(
                f"F{frame} A{attempt} {identifier} — 实际TCP; "
                f"目标误差={1000.0 * float(record['position_error_m']):.1f} mm"
            )
            actual_fingertip = record.get("actual_fingertip_center_base_m")
            if actual_fingertip is not None:
                actual_fingertips.append(xyz(actual_fingertip))
                tip_error = record.get("fingertip_error_m")
                actual_fingertip_text.append(
                    f"F{frame} A{attempt} {identifier} — 实际指尖中心; "
                    + (
                        f"指尖目标误差={1000.0 * float(tip_error):.1f} mm"
                        if tip_error is not None
                        else "指尖目标误差=—"
                    )
                )
            actual_point = xyz(record["actual_position_base_m"])
            target_point = xyz(record["desired_position_base_m"])
            error_xyz = [
                target_value - actual_value
                for target_value, actual_value in zip(
                    target_point, actual_point
                )
            ]
            error_mm = 1000.0 * float(record["position_error_m"])
            error_segment_points.extend(
                [actual_point, target_point, None]
            )
            error_label_points.append(
                [
                    0.5 * (actual_value + target_value)
                    for actual_value, target_value in zip(
                        actual_point, target_point
                    )
                ]
            )
            error_label_text.append(f"F{frame} A{attempt}: {error_mm:.1f} mm")
            error_label_hover.append(
                f"F{frame} A{attempt} {identifier}<br>"
                f"目标-实际 XYZ="
                f"[{1000.0 * error_xyz[0]:.1f}, "
                f"{1000.0 * error_xyz[1]:.1f}, "
                f"{1000.0 * error_xyz[2]:.1f}] mm<br>"
                f"三维误差={error_mm:.1f} mm"
            )
            before = command.get("measured_before", record.get("before_joints"))
            desired_joints = command.get("desired_feedback")
            after = record.get("actual_joints", command.get("measured_after"))
            if before is not None and desired_joints is not None and after is not None:
                requested_delta = max(
                    abs(float(target) - float(current))
                    for target, current in zip(desired_joints, before)
                )
                measured_delta = max(
                    abs(float(value) - float(current))
                    for value, current in zip(after, before)
                )
                joint_rows.append(
                    {
                        "frame": frame,
                        "attempt": attempt,
                        "id": identifier,
                        "pipeline": record.get("a0_pipeline", ""),
                        "requested_delta": requested_delta,
                        "measured_delta": measured_delta,
                        "tracking_error": command.get("max_feedback_error_rad"),
                        "desired_joints": desired_joints,
                        "actual_joints": after,
                    }
                )
        elif event == "return_home_after_release":
            actual.append(xyz(record["after_position_base_m"]))
            actual_text.append("F7 返回首次F1 — 实际tool-Z TCP")
        elif event == "frame_reached":
            reached.append(f"F{frame} {identifier}")
        elif event in {"frame_a0_cartesian_absolute_ik_error", "correction_ik_failed"}:
            target = record.get("compensated_target_position_base_m")
            parsed_target, parsed_fk = parse_rejected_fk(str(record.get("error", "")))
            target_point = xyz(target) if target is not None else parsed_target
            if target_point is not None:
                rejected_target.append(target_point)
                rejected_target_text.append(f"F{frame} {identifier} — 被拒绝IK目标")
            if parsed_fk is not None:
                rejected_fk.append(parsed_fk)
                rejected_fk_text.append(f"F{frame} {identifier} — 被拒绝solver FK")
            failure = f"F{frame} {identifier}: {record.get('error', event)}"
        elif event == "frame_joint_arrival_abort":
            failure = (
                f"F{frame} {identifier}: 关节到位失败，"
                f"最大关节误差="
                f"{float(record['max_joint_error_rad']):.4f} rad > "
                f"{float(record['joint_arrival_tolerance_rad']):.4f} rad"
            )
        elif event == "finished":
            failure = "finished"

    traces: list[dict[str, Any]] = []
    if desired:
        traces.append(
            trace(
                "完整执行目标轨迹",
                desired,
                color="#ffd60a",
                width=4,
                size=6,
                text=desired_text,
                dash="dash",
            )
        )
    if llm_anchors:
        traces.append(
            trace(
                "LLM语义锚点",
                llm_anchors,
                color="#ffd60a",
                mode="markers+text",
                size=9,
                text=llm_anchor_text,
                symbol="diamond",
            )
        )
    if interpolation_targets:
        traces.append(
            trace(
                "系统/运行时10cm插值点",
                interpolation_targets,
                color="#ff9f0a",
                mode="markers",
                size=5,
                text=interpolation_text,
            )
        )
    if coupled_targets:
        traces.append(
            trace(
                "系统姿态耦合TCP目标",
                coupled_targets,
                color="#ffcc00",
                width=5,
                size=8,
                text=coupled_target_text,
                symbol="diamond",
            )
        )
    if desired_fingertips:
        traces.append(
            trace(
                "目标指尖抓取中心轨迹",
                desired_fingertips,
                color="#64d2ff",
                width=4,
                size=7,
                text=desired_fingertip_text,
                dash="dash",
            )
        )
    if actual_fingertips:
        traces.append(
            trace(
                "实际指尖抓取中心轨迹",
                actual_fingertips,
                color="#bf5af2",
                width=6,
                size=6,
                text=actual_fingertip_text,
            )
        )
    if coupled_axis_points:
        traces.append(
            trace(
                "锁定接近轴：TCP→指尖/球心",
                coupled_axis_points,
                color="#ffffff",
                width=8,
                size=7,
                text=coupled_axis_text,
                dash="dot",
            )
        )
    if ik_requests:
        traces.append(
            trace(
                "在线IK输入目标轨迹",
                ik_requests,
                color="#00c2ff",
                width=5,
                size=5,
                text=ik_text,
            )
        )
    if ik_solution_fk:
        traces.append(
            trace(
                "IK关节解FK轨迹：FK(q_IK)",
                ik_solution_fk,
                color="#9d7cff",
                width=5,
                size=6,
                text=ik_solution_fk_text,
                symbol="diamond",
            )
        )
    if actual_joint_fk:
        traces.append(
            trace(
                "实测关节FK轨迹：FK(q_actual)",
                actual_joint_fk,
                color="#34c759",
                width=4,
                size=5,
                text=actual_joint_fk_text,
            )
        )
    if solver_fk:
        traces.append(
            trace(
                "候选筛选阶段FK（A0参考）",
                solver_fk,
                color="#9d7cff",
                mode="markers",
                size=7,
                text=solver_fk_text,
                symbol="diamond",
            )
        )
    if actual:
        traces.append(
            trace(
                "真机tool-Z TCP实际轨迹",
                actual,
                color="#ff2d95",
                width=8,
                size=5,
                text=actual_text,
            )
        )
    if error_segment_points:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "实际点→目标点误差向量",
                "x": [
                    None if point is None else point[0]
                    for point in error_segment_points
                ],
                "y": [
                    None if point is None else point[1]
                    for point in error_segment_points
                ],
                "z": [
                    None if point is None else point[2]
                    for point in error_segment_points
                ],
                "line": {"color": "#ff9500", "width": 3},
                "hoverinfo": "skip",
            }
        )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": "各实际点三维目标误差",
                "x": [point[0] for point in error_label_points],
                "y": [point[1] for point in error_label_points],
                "z": [point[2] for point in error_label_points],
                "text": error_label_text,
                "customdata": error_label_hover,
                "textposition": "top center",
                "textfont": {"color": "#ffcc66", "size": 10},
                "marker": {"color": "#ff9500", "size": 3},
                "hovertemplate": "%{customdata}<extra></extra>",
            }
        )
    if rejected_target:
        traces.append(
            trace(
                "被拒绝IK目标",
                rejected_target,
                color="#ff8c42",
                mode="markers",
                size=9,
                text=rejected_target_text,
                symbol="x",
            )
        )
    if rejected_fk:
        traces.append(
            trace(
                "被拒绝solver FK",
                rejected_fk,
                color="#ff3b30",
                mode="markers",
                size=10,
                text=rejected_fk_text,
                symbol="x",
            )
        )

    head_perception = (
        plan.get("fourth_pipeline", {}).get("planner")
        in {
            "llm_f2_f6_cartesian_targets",
            "llm_variable_semantic_cartesian_targets",
            "llm_free_space_with_system_coupled_grasp",
        }
    )
    for key, name, color in (
        (
            "tennis_ball_center_base_m",
            "头部相机定位：网球中心" if head_perception else "网球中心",
            "#dfff00",
        ),
        (
            "box_center_base_m",
            "头部相机定位：盒子中心" if head_perception else "盒子中心",
            "#35d07f",
        ),
    ):
        if key in objects:
            point = xyz(objects[key])
            traces.append(
                trace(
                    name,
                    [point],
                    color=color,
                    mode="markers+text",
                    size=10,
                    text=[name],
                    symbol="diamond",
                )
            )

    table_z = objects.get("table_plane_z_m")
    all_points = (
        desired
        + ik_requests
        + ik_solution_fk
        + actual_joint_fk
        + actual
        + actual_fingertips
        + desired_fingertips
        + rejected_fk
    )
    if table_z is not None and all_points:
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        margin = 0.05
        traces.append(
            {
                "type": "surface",
                "name": "桌面",
                "showscale": False,
                "opacity": 0.22,
                "x": [[min(xs) - margin, max(xs) + margin]] * 2,
                "y": [
                    [min(ys) - margin, min(ys) - margin],
                    [max(ys) + margin, max(ys) + margin],
                ],
                "z": [[float(table_z), float(table_z)]] * 2,
                "colorscale": [[0, "#4d8cff"], [1, "#4d8cff"]],
                "hovertemplate": f"table z={float(table_z):.3f} m<extra></extra>",
            }
        )

    rows = []
    for row in joint_rows:
        tracking = row["tracking_error"]
        tracking_cell = (
            f"<td>{float(tracking):.3f}</td>" if tracking is not None else "<td>—</td>"
        )
        rows.append(
            "<tr>"
            f"<td>F{row['frame']} A{row['attempt']}</td>"
            f"<td>{html.escape(row['id'])}</td>"
            f"<td>{html.escape(str(row['pipeline']))}</td>"
            f"<td>{row['requested_delta']:.3f}</td>"
            f"<td>{row['measured_delta']:.3f}</td>"
            f"{tracking_cell}"
            "</tr>"
        )
    max_requested = max((row["requested_delta"] for row in joint_rows), default=0.0)
    max_measured = max((row["measured_delta"] for row in joint_rows), default=0.0)
    ball = objects.get("tennis_ball_center_base_m")
    box = objects.get("box_center_base_m")
    perception_summary = ""
    if ball is not None:
        perception_summary += (
            f"<b>头部相机网球位置：</b>{html.escape(str(xyz(ball)))} m<br>"
        )
    if box is not None:
        perception_summary += (
            f"<b>头部相机盒子位置：</b>{html.escape(str(xyz(box)))} m<br>"
        )
    coupled_summary = ""
    if coupled_resolution is not None:
        coupled_summary = (
            "<b>LLM抓取建议：</b>"
            f"side={html.escape(str(coupled_resolution.get('llm_preferred_approach_side')))}, "
            f"yaw={float(coupled_resolution.get('llm_preferred_yaw_deg')):.1f}°, "
            f"tilt={float(coupled_resolution.get('llm_preferred_tilt_deg')):.1f}°<br>"
            "<b>系统选定抓取：</b>"
            f"yaw={float(coupled_resolution.get('selected_yaw_deg')):.1f}°, "
            f"tilt={float(coupled_resolution.get('selected_tilt_deg')):.1f}°, "
            f"mode={html.escape(str(coupled_resolution.get('selected_search_mode')))}, "
            f"fallback={bool(coupled_resolution.get('global_fallback_used'))}<br>"
        )
    summary = (
        f"<b>源日志：</b>{html.escape(str(source))}<br>"
        f"<b>Plan：</b>{html.escape(str(plan_path))}<br>"
        f"{perception_summary}"
        f"{coupled_summary}"
        f"<b>到达：</b>{html.escape(', '.join(reached) or 'none')}<br>"
        f"<b>结局：</b>{html.escape(failure)}<br>"
        f"<b>轨迹点：</b>LLM锚点 {len(llm_anchors)}，"
        f"插值点 {len(interpolation_targets)}，"
        f"耦合TCP目标 {len(coupled_targets)}，"
        f"目标/实际指尖 {len(desired_fingertips)}/{len(actual_fingertips)}，"
        f"IK输入 {len(ik_requests)}，"
        f"FK(q_IK) {len(ik_solution_fk)}，"
        f"FK(q_actual) {len(actual_joint_fk)}，"
        f"厂商TCP {len(actual)}，A0候选FK {len(solver_fk)}<br>"
        f"<b>最大单步：</b>请求 {max_requested:.3f} rad，实测 {max_measured:.3f} rad"
        "<table><thead><tr><th>步骤</th><th>ID</th><th>模式</th>"
        "<th>请求Δ(rad)</th><th>实测Δ(rad)</th><th>跟踪误差(rad)</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return traces, summary


def render(source: Path, output: Path) -> None:
    traces, summary = build_page(load_jsonl(source), source)
    serialized_traces = json_for_inline_script(traces)
    plotly = REPO / ".capx_r1lite_venv/lib/python3.10/site-packages/plotly/package_data/plotly.min.js"
    output.parent.mkdir(parents=True, exist_ok=True)
    if plotly.exists():
        shutil.copy2(plotly, output.parent / "plotly.min.js")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R1 Lite IK与实际轨迹</title>
<script src="plotly.min.js"></script>
<style>
html,body{{margin:0;background:#071019;color:#dceaf2;font:14px system-ui}}
#plot{{height:72vh;width:100%}} #summary{{padding:12px 18px}}
table{{border-collapse:collapse;margin-top:10px;width:100%}}
th,td{{border:1px solid #36505f;padding:5px;text-align:left}}
th{{background:#18303e}}
</style></head><body>
<div id="plot"></div><div id="summary">{summary}</div>
<script>
const data={serialized_traces};
Plotly.newPlot("plot",data,{{
  paper_bgcolor:"#101c27",plot_bgcolor:"#101c27",font:{{color:"#dceaf2"}},
  margin:{{l:0,r:0,t:48,b:0}},
  title:{{text:"R1 Lite：目标、IK/FK、实际TCP及逐点三维误差",font:{{size:17}}}},
  legend:{{orientation:"h"}},
  scene:{{aspectmode:"data",bgcolor:"#101c27",
    xaxis:{{title:"base X (m)",gridcolor:"#2a3c49"}},
    yaxis:{{title:"base Y (m)",gridcolor:"#2a3c49"}},
    zaxis:{{title:"base Z (m)",gridcolor:"#2a3c49"}}}}
}},{{responsive:true,displaylogo:false}});
</script></body></html>"""
    output.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--latest-html", type=Path)
    args = parser.parse_args()
    render(args.log_jsonl.resolve(), args.output_html.resolve())
    if args.latest_html is not None:
        args.latest_html.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output_html, args.latest_html)
    print(args.output_html)


if __name__ == "__main__":
    main()
