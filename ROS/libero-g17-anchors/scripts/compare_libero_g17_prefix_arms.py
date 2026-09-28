#!/usr/bin/env python3
"""Compare baseline / v24 / next prefix-arm LATEST_SUMMARY.json files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_arms_20260815"
)
OUT = ROOT / "PREFIX_ARM_SUMMARY.json"
ARMS = ("baseline", "v24", "next")


def main() -> int:
    arms = {}
    for arm in ARMS:
        path = ROOT / "arms" / arm / "LATEST_SUMMARY.json"
        if not path.is_file():
            arms[arm] = {"missing": True, "path": str(path)}
            continue
        arms[arm] = json.loads(path.read_text(encoding="utf-8"))

    # Fairness: identical π0 sha across arms that have prefix hits
    pi0 = {}
    for arm, summary in arms.items():
        if summary.get("missing"):
            continue
        for hit in summary.get("prefix_hit_records") or []:
            key = f"trial{hit.get('trial')}_turn{hit.get('turn')}"
            pi0.setdefault(key, {})[arm] = hit.get("code_sha256")

    pi0_identical = True
    for key, by_arm in pi0.items():
        vals = set(by_arm.values())
        if len(vals) > 1:
            pi0_identical = False

    summary = {
        "ok": all(
            not a.get("missing") and a.get("validity") == "valid_prefix_arm"
            for a in arms.values()
        ),
        "pi0_identical_across_arms": pi0_identical,
        "pi0_by_arm": pi0,
        "arms": {
            arm: {
                "validity": data.get("validity"),
                "gain_result": data.get("gain_result"),
                "cells": data.get("cells"),
                "schemas": data.get("schemas"),
                "post_fork_carrier": data.get("post_fork_carrier"),
                "treatment_job": data.get("treatment_job"),
                "missing": data.get("missing", False),
            }
            for arm, data in arms.items()
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] and pi0_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
