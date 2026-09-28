#!/usr/bin/env python3
"""Compare frozen fair-FG K=5: baseline vs early-positive v24 vs v219."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE_FAIR = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815"
)
V219 = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v219_20260816"
)
# Optional: include v26 for side context (not the primary early-positive arm).
V26 = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v26_20260815"
)
OUT = V219 / "FAIR_FG_K5_BASELINE_V24_V219.json"


def _load_reps(arm_root: Path) -> list[dict]:
    rows = []
    for rep_dir in sorted(arm_root.glob("rep-*")):
        latest = rep_dir / "LATEST_SUMMARY.json"
        if latest.is_file():
            row = json.loads(latest.read_text(encoding="utf-8"))
            row["_path"] = str(latest)
            rows.append(row)
    return rows


def _rate(rows: list[dict], pred) -> dict:
    valid = [r for r in rows if r.get("validity") == "valid_prefix_arm"]
    n = len(valid)
    hits = sum(1 for r in valid if pred(r))
    return {"n_valid": n, "hits": hits, "rate": (hits / n) if n else None}


def _metrics(rows: list[dict]) -> dict:
    return {
        "n_summaries": len(rows),
        "n_valid": sum(1 for r in rows if r.get("validity") == "valid_prefix_arm"),
        "trial1_success": _rate(
            rows,
            lambda r: (r.get("cells") or {})
            .get("trial_1_rescue_target", {})
            .get("treatment_completed")
            == 1,
        ),
        "trial2_success": _rate(
            rows,
            lambda r: (r.get("cells") or {})
            .get("trial_2_success_guard", {})
            .get("treatment_completed")
            == 1,
        ),
        "joint_positive": _rate(rows, lambda r: r.get("gain_result") == "positive"),
        "replicates": [
            {
                "replicate": r.get("replicate"),
                "validity": r.get("validity"),
                "gain_result": r.get("gain_result"),
                "treatment_job": r.get("treatment_job"),
                "cells": r.get("cells"),
                "schemas": r.get("schemas"),
            }
            for r in rows
        ],
    }


def _delta(a: dict, b: dict, key: str):
    ra = a[key]["rate"]
    rb = b[key]["rate"]
    if ra is None or rb is None:
        return None
    return rb - ra


def main() -> int:
    baseline = _load_reps(BASELINE_FAIR / "arms" / "baseline")
    v24 = _load_reps(BASELINE_FAIR / "arms" / "v24")
    v219 = _load_reps(V219 / "arms" / "v219")
    v26 = _load_reps(V26 / "arms" / "v26") if (V26 / "arms" / "v26").is_dir() else []

    m_base = _metrics(baseline)
    m_v24 = _metrics(v24)
    m_v219 = _metrics(v219)
    m_v26 = _metrics(v26) if v26 else None

    keys = ("trial1_success", "trial2_success", "joint_positive")
    summary = {
        "ok": m_base["n_valid"] == 5
        and m_v24["n_valid"] == 5
        and m_v219["n_valid"] == 5,
        "mode": "libero_g17_fair_fg_k5_baseline_v24_v219",
        "frozen": {
            "baseline": str(BASELINE_FAIR / "arms" / "baseline"),
            "early_positive_v24": str(BASELINE_FAIR / "arms" / "v24"),
            "v219": str(V219 / "arms" / "v219"),
        },
        "note": (
            "v24 is the first fair-FG K=5 treatment with clear joint-positive "
            "gain vs baseline; v26 listed only as side context."
        ),
        "metrics": {
            "baseline": m_base,
            "v24": m_v24,
            "v219": m_v219,
            **({"v26_side": m_v26} if m_v26 is not None else {}),
        },
        "delta_vs_baseline": {
            "v24": {k: _delta(m_base, m_v24, k) for k in keys},
            "v219": {k: _delta(m_base, m_v219, k) for k in keys},
            **(
                {"v26_side": {k: _delta(m_base, m_v26, k) for k in keys}}
                if m_v26 is not None
                else {}
            ),
        },
        "delta_v219_minus_v24": {k: _delta(m_v24, m_v219, k) for k in keys},
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["ok"]:
        print(
            f"INCOMPLETE: baseline={m_base['n_valid']} v24={m_v24['n_valid']} "
            f"v219={m_v219['n_valid']} (need 5 each)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
