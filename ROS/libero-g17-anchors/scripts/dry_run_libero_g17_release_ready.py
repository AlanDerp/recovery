#!/usr/bin/env python3
"""Dry-run: ReleaseReady student goto XY-only (v248) + tip-identity (v249) + hand path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from capx.verification import libero_g17_release_ready as rr
from capx.verification import libero_g17_takeover as takeover


def main() -> int:
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_RELEASE_READY"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_BAND_OBLIGATIONS", None)
    os.environ.pop("CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE", None)
    # Isolate legacy goto schema from tip-identity arm.
    os.environ.pop("CAPX_LIBERO_G17_GOTO_TIP_IDENTITY", None)

    os.environ.pop("CAPX_LIBERO_G17_RELEASE_READY_GOTO", None)
    schema_hand = takeover.schema_version()

    os.environ["CAPX_LIBERO_G17_RELEASE_READY_GOTO"] = "1"
    schema_goto = takeover.schema_version()

    os.environ["CAPX_LIBERO_G17_GOTO_TIP_IDENTITY"] = "1"
    schema_tip = takeover.schema_version()

    src = (
        Path(__file__).resolve().parents[1]
        / "capx/verification/libero_g17_release_ready.py"
    ).read_text(encoding="utf-8")
    libero_src = (
        Path(__file__).resolve().parents[1]
        / "capx/integrations/franka/libero.py"
    ).read_text(encoding="utf-8")
    checks = {
        "schema_hand": "release-ready-xyhand" in schema_hand and "v2.45" in schema_hand,
        "schema_goto": "release-ready-goto" in schema_goto and "v2.46" in schema_goto,
        "schema_tip_identity": (
            "tip-identity" in schema_tip and "v2.47" in schema_tip
        ),
        "goto_nudge": "def _nudge_xy_goto" in src and "api.goto_pose" in src,
        "hand_nudge": "def _nudge_hand_xy_lock_z" in src,
        "mode_switch": "release_ready_goto_mode" in src,
        "tip_ik_helper": "tip_pose_to_hand_ik_target" in libero_src,
    }
    print("schema_hand", schema_hand)
    print("schema_goto", schema_goto)
    print("schema_tip_identity", schema_tip)
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL", file=sys.stderr)
        return 1
    print("DRY_RUN_OK release-ready dual (hand+goto) + tip-identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
