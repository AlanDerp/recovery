#!/usr/bin/env python3
"""Compare v26 success-witness K reps against reused fair-FG baseline K=5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

V26 = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v26_20260815"
)
BASELINE_FAIR = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815"
)
OUT = V26 / "FAIR_FG_V26_VS_BASELINE.json"


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


def _witness_stats(rows: list[dict]) -> dict:
    """For v26: how many OK cells witnessed finish_authorized in FG traces."""
    out = {"ok_cells": 0, "ok_with_finish_witness": 0, "details": []}
    for r in rows:
        if r.get("validity") != "valid_prefix_arm":
            continue
        suite = Path(r["suite_root"])
        for trial, key in (
            (1, "trial_1_rescue_target"),
            (2, "trial_2_success_guard"),
        ):
            ok = (r.get("cells") or {}).get(key, {}).get("treatment_completed") == 1
            if not ok:
                continue
            out["ok_cells"] += 1
            trace = (
                suite
                / "libero_spatial_0"
                / "qwen3.7-plus"
                / "eval"
                / "libero_g17_fg_traces"
                / f"trial_{trial:02d}"
            )
            witnessed = False
            terminal = False
            for td in sorted(trace.glob("turn_*")):
                rec = json.loads((td / "record.json").read_text())
                if rec.get("finish_authorized"):
                    witnessed = True
                if rec.get("terminal_witness"):
                    terminal = True
            if witnessed:
                out["ok_with_finish_witness"] += 1
            out["details"].append(
                {
                    "replicate": r.get("replicate"),
                    "trial": trial,
                    "finish_witnessed": witnessed,
                    "had_terminal_witness_turn": terminal,
                }
            )
    out["witness_rate"] = (
        out["ok_with_finish_witness"] / out["ok_cells"] if out["ok_cells"] else None
    )
    return out


def main() -> int:
    baseline = _load_reps(BASELINE_FAIR / "arms" / "baseline")
    v26 = _load_reps(V26 / "arms" / "v26")
    metrics = {"baseline_reused": _metrics(baseline), "v26": _metrics(v26)}
    summary = {
        "ok": metrics["baseline_reused"]["n_valid"] == 5
        and metrics["v26"]["n_valid"] > 0,
        "mode": "libero_g17_fair_fg_v26_vs_reused_baseline",
        "baseline_source": str(BASELINE_FAIR / "arms" / "baseline"),
        "metrics": metrics,
        "delta_v26_minus_baseline": {
            key: (
                None
                if metrics["v26"][key]["rate"] is None
                or metrics["baseline_reused"][key]["rate"] is None
                else metrics["v26"][key]["rate"]
                - metrics["baseline_reused"][key]["rate"]
            )
            for key in ("trial1_success", "trial2_success", "joint_positive")
        },
        "layer1_success_witness": _witness_stats(v26),
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
