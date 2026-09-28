"""Merge real-robot, MuJoCo sim, and perception-based failure signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_mujoco_sim_report(result: dict[str, Any]) -> str:
    lines = [
        "MuJoCo proxy validation (robosuite PickPlaceCan analog):",
        f"  exec_ok: {result.get('exec_ok')}",
        f"  sim_success: {result.get('sim_success')}",
        f"  reason: {result.get('reason')}",
    ]
    stderr = str(result.get("stderr", "")).strip()
    if stderr:
        lines.append("  stderr excerpt:")
        lines.extend(f"    {line}" for line in stderr.splitlines()[:20])
    return "\n".join(lines)


def build_combined_correction_hints(
    *,
    geometric_report: dict[str, Any] | None = None,
    mujoco_report: dict[str, Any] | None = None,
    failure_analysis: str = "",
    next_fixes: str = "",
) -> str:
    parts: list[str] = []

    if geometric_report is not None:
        parts.append(
            "Real-robot perception+geometry check:\n"
            f"  success={geometric_report.get('success')} "
            f"reason={geometric_report.get('reason')} "
            f"xy={geometric_report.get('xy_distance_m')} "
            f"z_delta={geometric_report.get('z_delta_m')}"
        )

    if mujoco_report is not None:
        parts.append(format_mujoco_sim_report(mujoco_report))
        if mujoco_report.get("sim_success") and geometric_report and not geometric_report.get(
            "success"
        ):
            parts.append(
                "Diagnosis: code succeeds in MuJoCo proxy but fails on real hardware. "
                "Focus on perception drift, grasp height, safe_z, and real-arm joint limits."
            )
        elif not mujoco_report.get("exec_ok"):
            parts.append(
                "Diagnosis: code fails even in MuJoCo proxy. Fix motion sequence / IK jumps "
                "before changing perception-only details."
            )

    if failure_analysis.strip():
        parts.append(f"Failure analysis:\n{failure_analysis.strip()}")
    if next_fixes.strip():
        parts.append(f"Required fixes:\n{next_fixes.strip()}")

    return "\n\n".join(part for part in parts if part.strip())


def write_correction_bundle(
    path: str | Path,
    *,
    geometric_report: dict[str, Any] | None = None,
    mujoco_report: dict[str, Any] | None = None,
    failure_analysis: str = "",
    next_fixes: str = "",
) -> Path:
    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "geometric": geometric_report,
        "mujoco": mujoco_report,
        "failure_analysis": failure_analysis,
        "next_fixes": next_fixes,
        "combined_hints": build_combined_correction_hints(
            geometric_report=geometric_report,
            mujoco_report=mujoco_report,
            failure_analysis=failure_analysis,
            next_fixes=next_fixes,
        ),
    }
    bundle_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return bundle_path
