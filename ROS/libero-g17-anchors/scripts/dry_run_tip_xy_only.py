#!/usr/bin/env python3
"""Dry-run: tip XY-only primitive is importable and acceptance helpers work.

Full sim probe (``probe_tip_xy_primitive``) needs a live FrankaLiberoApi; that is
a separate sbatch smoke, not this gate.
"""

from __future__ import annotations

import os
import sys

from capx.integrations.franka import tip_xy_only as txy
from capx.integrations.franka.libero import FrankaLiberoApi


def main() -> int:
    checks = {
        "module_move": callable(txy.move_tip_xy),
        "module_probe": callable(txy.probe_tip_xy_primitive),
        "api_method": hasattr(FrankaLiberoApi, "move_tip_xy"),
        "flag_default_off": not txy.tip_xy_only_student_exposed()
        if os.environ.get("CAPX_LIBERO_TIP_XY_ONLY", "0") != "1"
        else True,
        "goto_untouched_marker": "CAPX_LIBERO_G17_GOTO_TIP_IDENTITY"
        in open(
            __import__("capx.integrations.franka.libero", fromlist=["x"]).__file__,
            encoding="utf-8",
        ).read(),
    }
    # Ensure move_tip_xy source does not call goto_pose.
    src = open(txy.__file__, encoding="utf-8").read()
    checks["primitive_avoids_goto"] = (
        "api.goto_pose" not in src.split("def probe_tip_xy_primitive")[0]
    )
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL tip_xy_only", file=sys.stderr)
        return 1
    print("DRY_RUN_OK tip_xy_only primitive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
