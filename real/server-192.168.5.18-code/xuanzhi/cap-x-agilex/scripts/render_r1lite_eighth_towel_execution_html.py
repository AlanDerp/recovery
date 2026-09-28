#!/usr/bin/env python3
"""Render pipeline-eight plan, certified waypoints, and measured TCP traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    validate_eighth_execution_bundle,
)
from capx.integrations.r1lite.towel_plan import (  # noqa: E402
    validate_eighth_bimanual_recenter_plan,
)

PLOTLY_JS = REPO / "outputs/r1lite_tennis_ball_in_box/plotly.min.js"
AUDIT_HASH_FIELDS = (
    "plan_sha256",
    "policy_sha256",
    "execution_bundle_sha256",
    "capability_certificate_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-bundle", type=Path)
    parser.add_argument("--execution-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_strict_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{path} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value, raw


def validate_bundle_binding(
    bundle: Mapping[str, Any], *, plan_file_sha256: str
) -> dict[str, Any]:
    """Strictly validate an eighth bundle and bind it to the exact plan file."""
    return validate_eighth_execution_bundle(
        bundle,
        plan_sha256=plan_file_sha256,
    )


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"execution audit {field} is not a lowercase SHA-256")
    return value


def validate_audit_bindings(
    audit: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    bundle_file_sha256: str | None = None,
) -> None:
    """Reject audits that do not identify and bind pipeline-eight artifacts."""
    if audit.get("pipeline") != 8 or audit.get("mode") != "execute":
        raise ValueError("execution audit is not a pipeline-eight execute audit")
    if not isinstance(audit.get("events"), list):
        raise ValueError("execution audit events must be a list")
    hashes = {
        field: _require_sha256(audit.get(field), field)
        for field in AUDIT_HASH_FIELDS
    }
    if hashes["plan_sha256"] != plan_file_sha256:
        raise ValueError("execution audit plan_sha256 binding mismatch")
    if (
        bundle_file_sha256 is not None
        and hashes["execution_bundle_sha256"] != bundle_file_sha256
    ):
        raise ValueError("execution audit execution_bundle_sha256 binding mismatch")


def fingertip_position(
    position_base_m: list[float],
    quaternion_xyzw: list[float],
    tcp_to_fingertip_m: float,
) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("fingertip trajectory quaternion cannot be zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    tool_z = [
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
    return [
        float(position_base_m[index])
        + tool_z[index] * float(tcp_to_fingertip_m)
        for index in range(3)
    ]


def extract_bundle_traces(
    bundle: Mapping[str, Any], *, tcp_to_fingertip_m: float
) -> dict[str, Any]:
    traces: dict[str, Any] = {
        "target": {"left": [], "right": []},
        "fk": {"left": [], "right": []},
        "target_fingertip": {"left": [], "right": []},
        "fk_fingertip": {"left": [], "right": []},
        "clearance": [],
    }
    sequence = 0
    for stage in bundle["stages"]:
        for waypoint_index, waypoint in enumerate(stage["waypoints"], start=1):
            sequence += 1
            label = f"{stage['id']} / waypoint {waypoint_index}"
            for side in ("left", "right"):
                traces["target"][side].append(
                    {
                        "label": label,
                        "position_base_m": waypoint[side]["target_position_base_m"],
                    }
                )
                traces["fk"][side].append(
                    {
                        "label": label,
                        "position_base_m": waypoint[side]["fk_position_base_m"],
                    }
                )
                traces["target_fingertip"][side].append(
                    {
                        "label": label,
                        "position_base_m": fingertip_position(
                            waypoint[side]["target_position_base_m"],
                            waypoint[side]["target_quaternion_xyzw"],
                            tcp_to_fingertip_m,
                        ),
                    }
                )
                traces["fk_fingertip"][side].append(
                    {
                        "label": label,
                        "position_base_m": fingertip_position(
                            waypoint[side]["fk_position_base_m"],
                            waypoint[side]["fk_quaternion_xyzw"],
                            tcp_to_fingertip_m,
                        ),
                    }
                )
            collision = waypoint["collision_summary"]
            if collision.get("checked") is False:
                continue
            clearance_values = [
                float(collision[field])
                for field in (
                    "estimated_link_clearance_m",
                    "left_table_clearance_m",
                    "right_table_clearance_m",
                    "gripper_envelope_separation_m",
                )
                if field in collision
            ]
            if not clearance_values:
                continue
            clearance = min(clearance_values)
            traces["clearance"].append(
                {
                    "sequence": sequence,
                    "label": label,
                    "clearance_m": clearance,
                }
            )
    return traces


def extract_actual_tcp_traces(
    audit: Mapping[str, Any], *, tcp_to_fingertip_m: float
) -> dict[str, list[dict[str, Any]]]:
    """Extract finite measured samples without coupling to runtime classes."""
    result: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    progress_samples = audit.get("joint_progress_tcp_trace", [])
    for event in audit.get("events", []):
        if not isinstance(event, dict) or event.get("event") != "actual_tcp_sample":
            continue
        if progress_samples and event.get("stage_id") != "initial":
            continue
        side = event.get("side")
        if side not in result:
            raise ValueError("actual_tcp_sample has an invalid side")
        position = event.get("position_base_m")
        quaternion = event.get("quaternion_xyzw")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in position
            )
        ):
            raise ValueError("actual_tcp_sample has an invalid position")
        if (
            not isinstance(quaternion, list)
            or len(quaternion) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in quaternion
            )
        ):
            raise ValueError("actual_tcp_sample has an invalid quaternion")
        result[side].append(
            {
                "stage_id": str(event.get("stage_id")),
                "phase": str(event.get("phase")),
                "sample_index": int(event.get("sample_index")),
                "position_base_m": [float(value) for value in position],
                "fingertip_position_base_m": fingertip_position(
                    position, quaternion, tcp_to_fingertip_m
                ),
            }
        )
    for sample in progress_samples:
        if not isinstance(sample, dict):
            raise ValueError("joint_progress_tcp_trace item must be an object")
        waypoint_index = int(sample.get("waypoint_index", 0))
        progress_step = int(sample["progress_step"])
        progress_steps = int(sample["progress_steps"])
        for side in ("left", "right"):
            position = sample[f"{side}_position_base_m"]
            quaternion = sample[f"{side}_quaternion_xyzw"]
            result[side].append(
                {
                    "stage_id": str(sample.get("stage_id")),
                    "phase": (
                        f"{sample.get('phase')} / waypoint {waypoint_index}"
                    ),
                    "sample_index": progress_step,
                    "position_base_m": [float(value) for value in position],
                    "fingertip_position_base_m": fingertip_position(
                        position, quaternion, tcp_to_fingertip_m
                    ),
                    "progress_step": progress_step,
                    "progress_steps": progress_steps,
                }
            )
    return result


def actual_pair_errors(
    actual: Mapping[str, list[dict[str, Any]]],
    plan: Mapping[str, Any],
    bundle: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Pair left/right samples and compute relative-vector and midpoint errors."""
    def key(item: Mapping[str, Any]) -> tuple[str, str, int]:
        return (
            str(item["stage_id"]),
            str(item["phase"]),
            int(item["sample_index"]),
        )

    left = {key(item): item for item in actual["left"]}
    right = {key(item): item for item in actual["right"]}
    expected: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
    if bundle is not None:
        for stage in bundle["stages"]:
            for index, waypoint in enumerate(stage["waypoints"], start=1):
                expected[(stage["id"], index)] = (
                    waypoint["expected_right_minus_left_m"],
                    waypoint["expected_midpoint_base_m"],
                )
    plan_stages = {stage["id"]: stage for stage in plan["stages"][:-1]}
    reference = plan["grasp_geometry"][
        "initial_full_relative_vector_right_minus_left_m"
    ]
    errors = []
    for pair_index, sample_key in enumerate(sorted(left.keys() & right.keys()), start=1):
        stage_id, phase, sample_index = sample_key
        left_position = left[sample_key]["position_base_m"]
        right_position = right[sample_key]["position_base_m"]
        expected_pair = expected.get((stage_id, sample_index))
        if expected_pair is not None:
            expected_relative, expected_midpoint = expected_pair
        elif stage_id in plan_stages:
            stage = plan_stages[stage_id]
            expected_left = stage["left_tcp_position_base_m"]
            expected_right = stage["right_tcp_position_base_m"]
            expected_relative = [
                right_value - left_value
                for left_value, right_value in zip(expected_left, expected_right)
            ]
            expected_midpoint = [
                (left_value + right_value) / 2.0
                for left_value, right_value in zip(expected_left, expected_right)
            ]
        else:
            continue
        actual_relative = [
            right_value - left_value
            for left_value, right_value in zip(left_position, right_position)
        ]
        actual_midpoint = [
            (left_value + right_value) / 2.0
            for left_value, right_value in zip(left_position, right_position)
        ]
        relative_reference = (
            reference
            if stage_id in {"grasp", "lift", "translate_center"}
            else expected_relative
        )
        errors.append(
            {
                "sequence": pair_index,
                "label": f"{stage_id} / {phase} / sample {sample_index}",
                "relative_vector_error_m": math.dist(
                    actual_relative, relative_reference
                ),
                "midpoint_error_m": math.dist(actual_midpoint, expected_midpoint),
            }
        )
    return errors


def build_payload(
    plan: Mapping[str, Any],
    bundle: Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    keypoints = plan["perception"]["keypoints"]
    tcp_to_fingertip_m = float(plan["safety"]["tcp_to_fingertip_m"])
    translate = next(
        stage for stage in plan["stages"] if stage["id"] == "translate_center"
    )
    translate_midpoint = [
        (
            translate["left_fingertip_position_base_m"][axis]
            + translate["right_fingertip_position_base_m"][axis]
        )
        / 2.0
        for axis in range(3)
    ]
    actual = (
        extract_actual_tcp_traces(
            audit, tcp_to_fingertip_m=tcp_to_fingertip_m
        )
        if audit is not None
        else {"left": [], "right": []}
    )
    bundle_traces = (
        extract_bundle_traces(
            bundle, tcp_to_fingertip_m=tcp_to_fingertip_m
        )
        if bundle is not None
        else {
            "target": {"left": [], "right": []},
            "fk": {"left": [], "right": []},
            "target_fingertip": {"left": [], "right": []},
            "fk_fingertip": {"left": [], "right": []},
            "clearance": [],
        }
    )
    actual_fingertip = {
        side: [
            {
                **item,
                "position_base_m": item["fingertip_position_base_m"],
            }
            for item in actual[side]
        ]
        for side in ("left", "right")
    }
    contact_probe = []
    if audit is not None:
        for event in audit.get("events", []):
            if (
                isinstance(event, dict)
                and event.get("event") == "eighth_contact_probe_sample"
            ):
                contact_probe.append(
                    {
                        "side": str(event["side"]),
                        "descent_m": float(event["descent_m"]),
                        "detected": bool(event["detected"]),
                        "joint_residual_score_nm": float(
                            event["joint_residual_score_nm"]
                        ),
                        "position_base_m": fingertip_position(
                            event["actual_tcp_position_base_m"],
                            event["actual_quaternion_xyzw"],
                            tcp_to_fingertip_m,
                        ),
                    }
                )
    return {
        "bundle": bundle_traces,
        "actual": actual,
        "actual_fingertip": actual_fingertip,
        "contact_probe": contact_probe,
        "errors": actual_pair_errors(actual, plan, bundle),
        "towel_contour_base_xyz_m": plan["perception"][
            "external_contour_base_xyz_m"
        ],
        "geometry": {
            "first_towel_center": plan["first_center_base_m"],
            "left_lower_contour_grasp": keypoints["left_grasp"]["base_xyz_m"],
            "right_lower_contour_grasp": keypoints["right_grasp"]["base_xyz_m"],
            "grasp_pair_midpoint": plan["grasp_geometry"][
                "grasp_pair_midpoint_base_m"
            ],
            "translate_center_endpoint": translate_midpoint,
        },
        "audit_passed": None if audit is None else audit.get("passed"),
    }


def render_html(payload: Mapping[str, Any], plotly_source: str) -> str:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>R1 Lite Pipeline 8 Execution</title>
<script>{plotly_source}</script>
<style>html,body{{margin:0;height:100%;font-family:sans-serif}}#trajectory{{height:68vh}}#errors{{height:32vh}}</style>
</head><body><div id="trajectory"></div><div id="errors"></div><script>
const data = {data};
function pathTrace(name, items, color, dash) {{
  return {{type:"scatter3d", mode:"lines+markers", name,
    x:items.map(v=>v.position_base_m[0]), y:items.map(v=>v.position_base_m[1]),
    z:items.map(v=>v.position_base_m[2]), text:items.map(v=>v.label || `${{v.stage_id}} / ${{v.phase}} / ${{v.sample_index}}`),
    hovertemplate:"%{{text}}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra></extra>",
    line:{{color,width:6,dash}}, marker:{{color,size:4}}}};
}}
const geometryNames = Object.keys(data.geometry);
const geometry = {{type:"scatter3d",mode:"markers+text",name:"Towel geometry",
  x:geometryNames.map(n=>data.geometry[n][0]), y:geometryNames.map(n=>data.geometry[n][1]),
  z:geometryNames.map(n=>data.geometry[n][2]), text:geometryNames,textposition:"top center",
  marker:{{color:"#111111",size:7,symbol:"diamond"}}}};
const towelContour = {{type:"scatter3d",mode:"lines+markers",name:"Perceived towel contour",
  x:data.towel_contour_base_xyz_m.map(v=>v[0]),
  y:data.towel_contour_base_xyz_m.map(v=>v[1]),
  z:data.towel_contour_base_xyz_m.map(v=>v[2]),
  line:{{color:"#8b6f47",width:5}},marker:{{color:"#8b6f47",size:2}}}};
const contactProbe = {{type:"scatter3d",mode:"markers",name:"Effort contact probe",
  x:data.contact_probe.map(v=>v.position_base_m[0]),
  y:data.contact_probe.map(v=>v.position_base_m[1]),
  z:data.contact_probe.map(v=>v.position_base_m[2]),
  text:data.contact_probe.map(v=>`${{v.side}} descent=${{(v.descent_m*1000).toFixed(1)}}mm residual=${{v.joint_residual_score_nm.toFixed(2)}}Nm detected=${{v.detected}}`),
  hovertemplate:"%{{text}}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra></extra>",
  marker:{{color:data.contact_probe.map(v=>v.detected ? "#ff0000" : "#ffaa00"),
    size:data.contact_probe.map(v=>v.detected ? 9 : 5),symbol:"diamond"}}}};
const centerConnection = {{type:"scatter3d",mode:"lines",name:"First center to translate-center endpoint",
  x:[data.geometry.first_towel_center[0],data.geometry.translate_center_endpoint[0]],
  y:[data.geometry.first_towel_center[1],data.geometry.translate_center_endpoint[1]],
  z:[data.geometry.first_towel_center[2],data.geometry.translate_center_endpoint[2]],
  line:{{color:"#ff8800",width:7,dash:"dash"}}}};
Plotly.newPlot("trajectory",[
  pathTrace("Left target fingertip",data.bundle.target_fingertip.left,"#00a65a","solid"),
  pathTrace("Right target fingertip",data.bundle.target_fingertip.right,"#2468ff","solid"),
  pathTrace("Left IK/FK fingertip",data.bundle.fk_fingertip.left,"#00a65a","dot"),
  pathTrace("Right IK/FK fingertip",data.bundle.fk_fingertip.right,"#2468ff","dot"),
  pathTrace("Left actual fingertip",data.actual_fingertip.left,"#ff3355","solid"),
  pathTrace("Right actual fingertip",data.actual_fingertip.right,"#8a2be2","solid"),
  contactProbe,towelContour,geometry,centerConnection
],{{title:`Pipeline 8 target / IK-FK / actual fingertip center (audit passed=${{data.audit_passed}})`,
  margin:{{l:0,r:0,b:0,t:45}},scene:{{xaxis:{{title:"base X (m)"}},yaxis:{{title:"base Y (m)"}},
  zaxis:{{title:"base Z (m)"}},aspectmode:"data"}}}},{{responsive:true}});
const errorTraces = [
  {{type:"scatter",mode:"lines+markers",name:"Actual pair relative-vector error",
    x:data.errors.map(v=>v.sequence),y:data.errors.map(v=>v.relative_vector_error_m),
    text:data.errors.map(v=>v.label),hovertemplate:"%{{text}}<br>%{{y:.5f}} m<extra></extra>"}},
  {{type:"scatter",mode:"lines+markers",name:"Actual pair midpoint error",
    x:data.errors.map(v=>v.sequence),y:data.errors.map(v=>v.midpoint_error_m),
    text:data.errors.map(v=>v.label),hovertemplate:"%{{text}}<br>%{{y:.5f}} m<extra></extra>"}},
  {{type:"scatter",mode:"lines+markers",name:"Bundle collision minimum clearance",
    x:data.bundle.clearance.map(v=>v.sequence),y:data.bundle.clearance.map(v=>v.clearance_m),
    text:data.bundle.clearance.map(v=>v.label),hovertemplate:"%{{text}}<br>%{{y:.5f}} m<extra></extra>"}}
];
Plotly.newPlot("errors",errorTraces,{{title:"Pair errors and certified collision clearance",
  xaxis:{{title:"paired sample / certified waypoint"}},yaxis:{{title:"distance (m)",rangemode:"tozero"}},
  margin:{{t:45}}}},{{responsive:true}});
</script></body></html>"""


def main() -> None:
    args = parse_args()
    plan, plan_raw = load_strict_object(args.plan)
    validate_eighth_bimanual_recenter_plan(plan)
    plan_hash = sha256_bytes(plan_raw)

    bundle = None
    bundle_hash = None
    if args.execution_bundle is not None:
        raw_bundle, bundle_raw = load_strict_object(args.execution_bundle)
        bundle = validate_bundle_binding(
            raw_bundle,
            plan_file_sha256=plan_hash,
        )
        bundle_hash = sha256_bytes(bundle_raw)

    audit = None
    if args.execution_audit is not None:
        audit, _ = load_strict_object(args.execution_audit)
        validate_audit_bindings(
            audit,
            plan_file_sha256=plan_hash,
            bundle_file_sha256=bundle_hash,
        )

    payload = build_payload(plan, bundle, audit)
    html = render_html(payload, PLOTLY_JS.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "target_waypoints": len(payload["bundle"]["target"]["left"]),
                "actual_left_samples": len(payload["actual"]["left"]),
                "actual_right_samples": len(payload["actual"]["right"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
