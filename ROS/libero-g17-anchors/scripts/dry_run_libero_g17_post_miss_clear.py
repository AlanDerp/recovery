#!/usr/bin/env python3
"""Dry-run: post-miss clear gate + schema on v238 stack."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from capx.verification import libero_g17_post_miss_clear as pmc
from capx.verification import libero_g17_takeover as takeover


def main() -> int:
    os.environ["CAPX_LIBERO_G17_XY_LOCK"] = "1"
    os.environ["CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL"] = "1"
    os.environ["CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_BAND_OBLIGATIONS", None)
    os.environ.pop("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", None)
    schema = takeover.schema_version()
    proj = SimpleNamespace(
        task_gap="placement_xy",
        executor_attachment="empty",
        released_goal_after_transport=True,
        gain=SimpleNamespace(bound_release=True),
        combined_report="placement near-miss: center_on_surface is on",
    )
    checks = {
        "schema": "post-miss-clear" in schema and "v2.41" in schema,
        "clear_trigger": pmc.should_clear_after_projection(proj),
        "goal_lock": pmc.goal_lock_between_to_ordinal_enabled(),
        "note": "prior code attempts are omitted" in "\n".join(pmc.context_reset_note_lines()),
    }
    print("schema", schema)
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL", file=sys.stderr)
        return 1
    print("DRY_RUN_OK post-miss clear-cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
