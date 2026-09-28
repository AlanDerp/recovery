#!/usr/bin/env python3
"""Dry-run analysis for Layer-1 success-witness fix before K=5 submit.

Checks:
1. Schema is v2.6-success-witness (workgap off).
2. trial.py hooks capture_terminal_witness on episode stop.
3. Unit tests for terminal witness pass.
4. Historical fair K=5 v24 OK trials never witnessed finish (precondition).
5. Baseline arm summaries from fair K=5 are reusable (no re-run needed).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

G17 = Path("/home/qshou/capx_g17_observation_integrity")
BASELINE_FAIR = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815"
)
OUT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v26_20260815"
)


def main() -> int:
    report: dict = {"ok": True, "checks": []}

    # 1. schema
    sys.path.insert(0, str(G17))
    from capx.verification import libero_g17_takeover as takeover

    schema = takeover.SCHEMA_VERSION
    c1 = {
        "name": "schema_success_witness",
        "schema": schema,
        "pass": "success-witness" in schema and "workgap" not in schema,
    }
    report["checks"].append(c1)

    # 2. trial hook
    trial_src = (G17 / "capx/envs/trial.py").read_text(encoding="utf-8")
    c2 = {
        "name": "trial_terminal_witness_hook",
        "pass": "capture_terminal_witness" in trial_src
        and "Layer-1: when the env stops" in trial_src,
    }
    report["checks"].append(c2)

    # 3. unit tests
    py = Path("/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python")
    if not py.is_file():
        py = Path(sys.executable)
    proc = subprocess.run(
        [
            str(py),
            "-m",
            "pytest",
            "-q",
            str(G17 / "tests/test_libero_g17_success_witness.py"),
            str(G17 / "tests/test_libero_g17_takeover.py"),
            str(G17 / "tests/test_libero_g17_public_iff.py"),
        ],
        cwd=str(G17),
        capture_output=True,
        text=True,
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "CAPX_LIBERO_G17_WORKGAP_ME": "0",
            "PYTHONPATH": f"{G17}:{__import__('os').environ.get('PYTHONPATH', '')}",
        },
    )
    c3 = {
        "name": "unit_tests",
        "pass": proc.returncode == 0,
        "stdout_tail": proc.stdout[-500:],
        "stderr_tail": proc.stderr[-500:],
    }
    report["checks"].append(c3)

    # 4. historical gap evidence
    hist = []
    for rep in range(5):
        p = BASELINE_FAIR / "arms" / "v24" / f"rep-{rep:02d}" / "LATEST_SUMMARY.json"
        if not p.is_file():
            continue
        s = json.loads(p.read_text())
        suite = Path(s["suite_root"])
        for trial in (1, 2):
            ok = (
                s["cells"][
                    "trial_1_rescue_target" if trial == 1 else "trial_2_success_guard"
                ].get("treatment_completed")
                == 1
            )
            trace = (
                suite
                / "libero_spatial_0"
                / "qwen3.7-plus"
                / "eval"
                / "libero_g17_fg_traces"
                / f"trial_{trial:02d}"
            )
            witnessed = False
            for td in trace.glob("turn_*"):
                rec = json.loads((td / "record.json").read_text())
                if rec.get("finish_authorized") or rec.get("bddl_atom_true"):
                    witnessed = True
            hist.append(
                {
                    "rep": rep,
                    "trial": trial,
                    "ok": ok,
                    "witnessed_finish_or_bddl": witnessed,
                }
            )
    ok_unwitnessed = [
        h for h in hist if h["ok"] and not h["witnessed_finish_or_bddl"]
    ]
    c4 = {
        "name": "historical_v24_ok_unwitnessed",
        "pass": len(ok_unwitnessed) > 0,
        "n_ok_unwitnessed": len(ok_unwitnessed),
        "detail": ok_unwitnessed,
        "note": "Precondition: v24 OK cells never saw finish/bddl in FG traces",
    }
    report["checks"].append(c4)

    # 5. reusable baseline
    baseline_reps = sorted((BASELINE_FAIR / "arms" / "baseline").glob("rep-*/LATEST_SUMMARY.json"))
    valid_baseline = 0
    for p in baseline_reps:
        s = json.loads(p.read_text())
        if s.get("validity") == "valid_prefix_arm":
            valid_baseline += 1
    c5 = {
        "name": "reusable_baseline_k5",
        "pass": valid_baseline == 5,
        "n_valid_baseline": valid_baseline,
        "source": str(BASELINE_FAIR / "arms" / "baseline"),
    }
    report["checks"].append(c5)

    report["ok"] = all(c["pass"] for c in report["checks"])
    report["next_experiment"] = {
        "mode": "treatment_only_k5",
        "arm": "v26",
        "schema": schema,
        "reuse_baseline_from": str(BASELINE_FAIR),
        "output": str(OUT),
        "prefix_pkg": (
            "/project/peilab/qshou/cap-x-agilex/outputs/"
            "libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json"
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "DRY_RUN_SUCCESS_WITNESS.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
