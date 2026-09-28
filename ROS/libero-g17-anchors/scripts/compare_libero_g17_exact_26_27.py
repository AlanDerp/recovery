#!/usr/bin/env python3
"""Compare v2.6 vs v2.7 capture + exact-rerun for Layer-1 fair claim."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_exact_26_27_20260815"
)
OUT = EXPT / "EXACT_26_27_SUMMARY.json"
ARMS = ("v26", "v27")


def _load(arm: str, mode: str) -> dict | None:
    p = EXPT / "arms" / arm / mode / "rep-00" / "LATEST_SUMMARY.json"
    if p.is_file():
        return json.loads(p.read_text())
    cands = sorted((EXPT / "arms" / arm / mode).glob("**/LATEST_SUMMARY.json"))
    if not cands:
        return None
    return json.loads(cands[-1].read_text())


def _cells(row: dict) -> dict:
    c = row.get("cells") or {}
    return {
        "t1": (c.get("trial_1_rescue_target") or {}).get("treatment_completed"),
        "t2": (c.get("trial_2_success_guard") or {}).get("treatment_completed"),
        "gain": row.get("gain_result"),
        "validity": row.get("validity"),
        "schemas": row.get("schemas"),
        "job": row.get("treatment_job"),
        "suite_root": row.get("suite_root"),
    }


def _exact_match(capture: dict | None, exact: dict | None) -> dict:
    if not capture or not exact:
        return {"ok": False, "reason": "missing summary"}
    cc, ee = _cells(capture), _cells(exact)
    ok = (
        capture.get("validity") == "valid_prefix_arm"
        and exact.get("validity") == "valid_prefix_arm"
        and cc["t1"] == ee["t1"]
        and cc["t2"] == ee["t2"]
    )
    return {"ok": ok, "capture": cc, "exact": ee}


def main() -> int:
    by_arm = {}
    for arm in ARMS:
        capture = _load(arm, "capture")
        exact = _load(arm, "exact")
        by_arm[arm] = {
            "exact_rerun_matches_capture": _exact_match(capture, exact),
            "cells_capture": _cells(capture) if capture else None,
            "cells_exact": _cells(exact) if exact else None,
        }

    claim_ok = all(
        by_arm[a]["exact_rerun_matches_capture"]["ok"] for a in ARMS
    )
    summary = {
        "ok": claim_ok,
        "mode": "libero_g17_exact_26_27",
        "unique_delta": "v2.6-success-witness vs v2.7-sparse-nearmiss",
        "arms": {
            a: {
                "exact_rerun_ok": by_arm[a]["exact_rerun_matches_capture"],
                "capture_cells": by_arm[a]["cells_capture"],
                "exact_cells": by_arm[a]["cells_exact"],
            }
            for a in ARMS
        },
        "contrast_v27_minus_v26_on_proven": None,
    }
    if claim_ok:
        c26 = by_arm["v26"]["cells_capture"]
        c27 = by_arm["v27"]["cells_capture"]
        summary["contrast_v27_minus_v26_on_proven"] = {
            "t1": int(bool(c27["t1"])) - int(bool(c26["t1"])),
            "t2": int(bool(c27["t2"])) - int(bool(c26["t2"])),
        }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if claim_ok else 1


if __name__ == "__main__":
    sys.exit(main())
