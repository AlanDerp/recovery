#!/usr/bin/env python3
"""Dry-run: near-miss osc schema + tip_xy wiring."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from capx.verification import libero_g17_nearmiss_osc as osc
from capx.verification import libero_g17_takeover as takeover


def main() -> int:
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ["CAPX_LIBERO_G17_NEARMISS_OSC"] = "1"
    for k in (
        "CAPX_LIBERO_G17_RELEASE_READY_GOTO",
        "CAPX_LIBERO_G17_GOTO_TIP_IDENTITY",
        "CAPX_LIBERO_G17_BAND_OBLIGATIONS",
        "CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE",
        "CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER",
    ):
        os.environ.pop(k, None)

    schema = takeover.schema_version()
    rr_src = (
        Path(__file__).resolve().parents[1]
        / "capx/verification/libero_g17_release_ready.py"
    ).read_text(encoding="utf-8")
    tip_src = (
        Path(__file__).resolve().parents[1]
        / "capx/integrations/franka/tip_xy_only.py"
    ).read_text(encoding="utf-8")
    checks = {
        "schema": "nearmiss-osc-tipxy" in schema and "v2.48" in schema,
        "flag": osc.nearmiss_osc_enabled(),
        "dispatch": "run_nearmiss_osc" in rr_src,
        "uses_tip_xy": "move_tip_xy" in open(osc.__file__, encoding="utf-8").read(),
        "no_goto_in_osc": "api.goto_pose" not in open(osc.__file__, encoding="utf-8").read()
        and "goto_pose(" not in open(osc.__file__, encoding="utf-8").read(),
        "tip_primitive": "def move_tip_xy" in tip_src,
        "math_schedule": osc.axis_magnitude(1.0, 0.1, 0) == 1.0,
    }
    print("schema", schema)
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL nearmiss_osc", file=sys.stderr)
        return 1
    print("DRY_RUN_OK nearmiss_osc tipxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
