#!/usr/bin/env python3
"""Aggregate max-freeze fair FG K-replicate summaries (baseline vs v24)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815"
)
OUT = ROOT / "FAIR_FG_KSEED_SUMMARY.json"
ARMS = ("baseline", "v24")


def _load_rep_summaries(arm: str) -> list[dict]:
    arm_root = ROOT / "arms" / arm
    if not arm_root.is_dir():
        return []
    rows: list[dict] = []
    for rep_dir in sorted(arm_root.glob("rep-*")):
        latest = rep_dir / "LATEST_SUMMARY.json"
        if not latest.is_file():
            continue
        row = json.loads(latest.read_text(encoding="utf-8"))
        row["_path"] = str(latest)
        rows.append(row)
    return rows


def _rate(rows: list[dict], pred) -> dict:
    valid = [r for r in rows if r.get("validity") == "valid_prefix_arm"]
    n = len(valid)
    hits = sum(1 for r in valid if pred(r))
    return {
        "n_valid": n,
        "hits": hits,
        "rate": (hits / n) if n else None,
    }


def main() -> int:
    by_arm: dict[str, list[dict]] = {arm: _load_rep_summaries(arm) for arm in ARMS}

    pi0: dict[str, dict[str, str]] = {}
    for arm, rows in by_arm.items():
        for row in rows:
            if row.get("validity") != "valid_prefix_arm":
                continue
            for hit in row.get("prefix_hit_records") or []:
                key = f"trial{hit.get('trial')}_turn{hit.get('turn')}"
                pi0.setdefault(key, {})[f"{arm}_r{int(row.get('replicate', -1)):02d}"] = (
                    hit.get("code_sha256")
                )

    pi0_identical = True
    for by_rep in pi0.values():
        if len(set(by_rep.values())) > 1:
            pi0_identical = False

    metrics = {}
    for arm, rows in by_arm.items():
        metrics[arm] = {
            "n_summaries": len(rows),
            "n_valid": sum(1 for r in rows if r.get("validity") == "valid_prefix_arm"),
            "trial1_success": _rate(
                rows,
                lambda r: (r.get("cells") or {})
                .get("trial_1_rescue_target", {})
                .get("treatment_success")
                is True
                or (r.get("cells") or {})
                .get("trial_1_rescue_target", {})
                .get("treatment_completed")
                == 1,
            ),
            "trial2_success": _rate(
                rows,
                lambda r: (r.get("cells") or {})
                .get("trial_2_success_guard", {})
                .get("treatment_success")
                is True
                or (r.get("cells") or {})
                .get("trial_2_success_guard", {})
                .get("treatment_completed")
                == 1,
            ),
            "joint_positive": _rate(
                rows,
                lambda r: r.get("gain_result") == "positive",
            ),
            "rescued": _rate(
                rows,
                lambda r: (r.get("cells") or {})
                .get("trial_1_rescue_target", {})
                .get("rescued")
                is True,
            ),
            "guard_held": _rate(
                rows,
                lambda r: (r.get("cells") or {})
                .get("trial_2_success_guard", {})
                .get("guard_held")
                is True,
            ),
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

    both_ready = all(metrics[a]["n_valid"] > 0 for a in ARMS)
    summary = {
        "ok": both_ready and pi0_identical,
        "mode": "libero_g17_fair_fg_max_freeze",
        "unique_delta": "CAPX_LIBERO_G17_FG",
        "workgap": False,
        "pi0_identical_across_valid_replicates": pi0_identical,
        "pi0_by_key": pi0,
        "metrics": metrics,
        "delta_v24_minus_baseline": {
            key: (
                None
                if metrics["v24"][key]["rate"] is None
                or metrics["baseline"][key]["rate"] is None
                else metrics["v24"][key]["rate"] - metrics["baseline"][key]["rate"]
            )
            for key in (
                "trial1_success",
                "trial2_success",
                "joint_positive",
                "rescued",
                "guard_held",
            )
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
