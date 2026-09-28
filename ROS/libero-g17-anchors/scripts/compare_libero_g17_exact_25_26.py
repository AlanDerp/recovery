#!/usr/bin/env python3
"""Compare v2.5 vs v2.6 capture + exact-rerun for Layer-1 fair claim."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_exact_25_26_20260815"
)
OUT = EXPT / "EXACT_25_26_SUMMARY.json"
ARMS = ("v25", "v26")


def _load(arm: str, mode: str) -> dict | None:
    p = EXPT / "arms" / arm / "rep-00" / "LATEST_SUMMARY.json"
    # summaries may live under mode-specific trees only in runs/; arms/ is written by sbatch
    if not p.is_file():
        # try newest SUMMARY under arms
        cands = sorted((EXPT / "arms" / arm).glob("**/LATEST_SUMMARY.json"))
        if not cands:
            return None
        p = cands[-1]
    row = json.loads(p.read_text())
    if row.get("run_mode") and row.get("run_mode") != mode:
        # prefer matching mode if multiple
        for c in sorted((EXPT / "arms" / arm).glob("**/LATEST_SUMMARY.json")):
            r = json.loads(c.read_text())
            if r.get("run_mode") == mode:
                return r
    return row


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
        # Prefer summaries from latest submit; sbatch overwrites LATEST per arm.
        # Also scan runs for mode-capture / mode-exact.
        capture = None
        exact = None
        for mode in ("capture", "exact"):
            runs = sorted(
                (EXPT / "runs" / "libero_spatial" / f"mode-{mode}" / f"arm-{arm}").glob(
                    "**/LATEST_SUMMARY.json"
                )
            )
            # summaries are under arms/, not runs — reconstruct from arms + run_mode
            pass
        # Fall back: read all arm summaries and split by run_mode field written into them
        for p in sorted((EXPT / "arms" / arm).glob("**/SUMMARY_job-*.json")):
            row = json.loads(p.read_text())
            rm = row.get("run_mode")
            if rm == "capture":
                capture = row
            elif rm == "exact":
                exact = row
        # LATEST may be whichever finished last
        latest = _load(arm, "exact") or _load(arm, "capture")
        by_arm[arm] = {
            "capture": capture,
            "exact": exact,
            "latest": latest,
            "exact_rerun_matches_capture": _exact_match(capture, exact),
            "cells_capture": _cells(capture) if capture else None,
            "cells_exact": _cells(exact) if exact else None,
        }

    claim_ok = all(
        by_arm[a]["exact_rerun_matches_capture"]["ok"] for a in ARMS
    )
    summary = {
        "ok": claim_ok,
        "mode": "libero_g17_exact_25_26",
        "unique_delta": "v2.5-workgap-me vs v2.6-success-witness",
        "arms": {
            a: {
                "exact_rerun_ok": by_arm[a]["exact_rerun_matches_capture"],
                "capture_cells": by_arm[a]["cells_capture"],
                "exact_cells": by_arm[a]["cells_exact"],
            }
            for a in ARMS
        },
        "contrast_v26_minus_v25_on_proven": None,
    }
    if claim_ok:
        c25 = by_arm["v25"]["cells_exact"]
        c26 = by_arm["v26"]["cells_exact"]
        summary["contrast_v26_minus_v25_on_proven"] = {
            "t1": (c26["t1"] or 0) - (c25["t1"] or 0),
            "t2": (c26["t2"] or 0) - (c25["t2"] or 0),
            "v25": c25,
            "v26": c26,
        }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if claim_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
