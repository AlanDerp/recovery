#!/usr/bin/env python3
"""Aggregate repeated head-stereo table validations using robust medians."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audits", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-tilt-deg", type=float, default=1.0)
    parser.add_argument("--maximum-rmse-m", type=float, default=0.012)
    parser.add_argument("--maximum-height-span-m", type=float, default=0.010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [json.loads(path.read_text()) for path in args.audits]
    if len(audits) < 3:
        raise RuntimeError("at least three table validations are required")
    if any(
        audit.get("schema") != "capx.r1lite.head_stereo_table_validation.v1"
        for audit in audits
    ):
        raise RuntimeError("unexpected validation audit schema")
    calibration_paths = {audit["calibration_path"] for audit in audits}
    if len(calibration_paths) != 1:
        raise RuntimeError("validation audits use different calibrations")
    normals = np.asarray(
        [audit["normal_base"] for audit in audits], dtype=np.float64
    )
    normal = np.median(normals, axis=0)
    normal /= np.linalg.norm(normal)
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    rmse = float(np.median([audit["rmse_m"] for audit in audits]))
    height_span = float(
        np.median([audit["workspace_height_span_m"] for audit in audits])
    )
    passed = (
        tilt <= float(args.maximum_tilt_deg)
        and rmse <= float(args.maximum_rmse_m)
        and height_span <= float(args.maximum_height_span_m)
    )
    result = {
        "schema": "capx.r1lite.head_stereo_table_validation_aggregate.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "robot_motion_commands_sent": False,
        "calibration_path": next(iter(calibration_paths)),
        "audit_paths": [str(path.resolve()) for path in args.audits],
        "sample_count": len(audits),
        "median_normal_base": normal.tolist(),
        "aggregate_tilt_deg": tilt,
        "median_rmse_m": rmse,
        "median_height_span_m": height_span,
        "individual_tilt_deg": [audit["tilt_deg"] for audit in audits],
        "individual_height_span_m": [
            audit["workspace_height_span_m"] for audit in audits
        ],
        "thresholds": {
            "maximum_tilt_deg": float(args.maximum_tilt_deg),
            "maximum_rmse_m": float(args.maximum_rmse_m),
            "maximum_height_span_m": float(args.maximum_height_span_m),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
