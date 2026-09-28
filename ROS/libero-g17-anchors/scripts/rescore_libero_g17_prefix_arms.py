#!/usr/bin/env python3
"""Rescore prefix-arm summaries with corrected log-path infra gate."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_arms_20260815"
)
BASELINE_EVAL = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/"
    "runs/libero_spatial/job-517994/libero_spatial_0/qwen3.7-plus/eval"
)
ARMS = {
    "baseline": "519246",
    "v24": "519247",
    "next": "519248",
}


def outcomes(root: Path) -> dict:
    rows: dict = {}
    if not root.exists():
        return rows
    for p in root.glob("trial_*"):
        m = re.match(
            r"trial_(\d+)_sandboxrc_(\d+)_reward_([0-9.]+)_taskcompleted_(\d+)",
            p.name,
        )
        if not m:
            continue
        tid = int(m.group(1))
        completed = int(m.group(4))
        reward = float(m.group(3))
        prev = rows.get(tid)
        if prev is None or completed > prev["completed"] or (
            completed == prev["completed"] and reward > prev["reward"]
        ):
            rows[tid] = {
                "completed": completed,
                "reward": reward,
                "path": p.name,
                "sandboxrc": int(m.group(2)),
            }
    return rows


def rescore(arm: str, job_id: str) -> dict:
    job = ROOT / "runs" / "libero_spatial" / f"arm-{arm}" / f"job-{job_id}"
    eval_dir = job / "libero_spatial_0" / "qwen3.7-plus" / "eval"
    if not eval_dir.exists():
        cands = list((job / "libero_spatial_0").glob("*/eval"))
        eval_dir = cands[0] if cands else eval_dir

    log_candidates = [
        ROOT / "logs" / f"libero-g17-pfx-{arm}-{job_id}.out",
        ROOT / "logs" / f"libero-g17-pfx-{job_id}.out",
    ]
    log = next((p for p in log_candidates if p.exists()), log_candidates[0])
    molmo_log = job / "molmo.log"

    ctrl = outcomes(BASELINE_EVAL)
    treat = outcomes(eval_dir)

    prefix_hits = []
    if eval_dir.exists():
        prefix_hits.extend(
            eval_dir.glob("**/libero_g17_code_prefix_replay/trial_*_turn_*.json")
        )
    repro = job / "libero_spatial_0" / "repro_audit" / "libero_g17_code_prefix_replay"
    if repro.exists():
        prefix_hits.extend(repro.glob("trial_*_turn_*.json"))
    seen = set()
    uniq = []
    for p in prefix_hits:
        if p.name in seen:
            continue
        seen.add(p.name)
        uniq.append(p)
    prefix_hit_rows = [json.loads(p.read_text()) for p in uniq]
    prefix_ok = len(prefix_hit_rows) >= 2

    engagement = {}
    turn_records = {}
    for trace_root in job.glob("**/libero_g17_fg_traces"):
        for trial_dir in sorted(trace_root.glob("trial_*")):
            tid = int(trial_dir.name.split("_")[1])
            eng = trial_dir / "engagement.json"
            if eng.exists():
                engagement[tid] = json.loads(eng.read_text())
            turns = []
            for turn_dir in sorted(trial_dir.glob("turn_*")):
                rec_path = turn_dir / "record.json"
                if rec_path.exists():
                    turns.append(json.loads(rec_path.read_text()))
            turn_records[tid] = turns

    fg = arm != "baseline"
    fg_engaged = bool(engagement) if fg else True
    schemas = {str(k): (v or {}).get("schema_version") for k, v in engagement.items()}

    log_text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    molmo_text = (
        molmo_log.read_text(encoding="utf-8", errors="ignore") if molmo_log.exists() else ""
    )
    refused = len(re.findall(r"Connection refused", log_text))
    wrong_port_8122 = len(re.findall(r"127\.0\.0\.1:8122", log_text))
    molmo_chat_hits = len(re.findall(r"/v1/chat/completions", molmo_text))
    molmo_url_proof = "molmo_url_proof_ok" in log_text
    prefix_proof = "prefix_hook_proof_ok" in log_text
    infra_ok = (
        molmo_url_proof
        and prefix_proof
        and refused == 0
        and wrong_port_8122 == 0
        and molmo_chat_hits > 0
        and log.exists()
    )
    infra_reasons = []
    if not log.exists():
        infra_reasons.append(f"missing_log:{log}")
    if not molmo_url_proof:
        infra_reasons.append("missing molmo_url_proof_ok")
    if not prefix_proof:
        infra_reasons.append("missing prefix_hook_proof_ok")
    if refused:
        infra_reasons.append(f"connection_refused_count={refused}")
    if wrong_port_8122:
        infra_reasons.append(f"hardcoded_8122_hits={wrong_port_8122}")
    if molmo_chat_hits <= 0:
        infra_reasons.append("molmo_log_has_no_chat_completions")

    if not prefix_ok:
        validity = "invalid_prefix_miss"
    elif fg and not fg_engaged:
        validity = "invalid_fg_not_engaged"
    elif not infra_ok:
        validity = "invalid_infra"
    else:
        validity = "valid_prefix_arm"

    rescued = guard_held = gain = None
    if validity == "valid_prefix_arm":
        rescued = ctrl.get(1, {}).get("completed") == 0 and treat.get(1, {}).get("completed") == 1
        guard_held = ctrl.get(2, {}).get("completed") == 1 and treat.get(2, {}).get("completed") == 1
        if rescued and guard_held:
            gain = "positive"
        elif (not rescued) and guard_held:
            gain = "negative_no_rescue"
        else:
            gain = "negative_guard_lost_or_incomplete"

    carrier = {}
    for tid, row in treat.items():
        trial_dirs = list(eval_dir.glob(f"trial_{tid:02d}_*"))
        if not trial_dirs:
            continue
        tdir = sorted(
            trial_dirs, key=lambda p: ("taskcompleted_1" not in p.name, p.name)
        )[0]
        prompts = (
            sorted((tdir / "prompts_and_responses").glob("multi_turn_prompt_*.txt"))
            if (tdir / "prompts_and_responses").exists()
            else []
        )
        text = prompts[0].read_text(encoding="utf-8", errors="ignore") if prompts else ""
        carrier[str(tid)] = {
            "has_ontop_iff": "Verified LIBERO on-top IFF basis" in text,
            "has_placement_incomplete_note": "placement evidence is incomplete" in text,
            "has_acquisition_incomplete_note": "acquisition evidence is incomplete" in text,
            "prompt_file": str(prompts[0]) if prompts else None,
        }

    # Mechanism audit: did workgap-specific near-miss fire?
    workgap_near_miss_turns = []
    for tid, turns in turn_records.items():
        for rec in turns:
            # distinctive workgap: placement without requiring we re-read endpoint;
            # record endpoint_gap + schema is enough with separate file scan later
            workgap_near_miss_turns.append(
                {
                    "trial": tid,
                    "turn": rec.get("turn"),
                    "endpoint_gap": rec.get("endpoint_gap"),
                    "finish_authorized": rec.get("finish_authorized"),
                    "schema_version": rec.get("schema_version"),
                }
            )

    summary = {
        "arm": arm,
        "validity": validity,
        "gain_result": gain,
        "prefix_ok": prefix_ok,
        "prefix_hits": len(prefix_hit_rows),
        "fg_enabled": fg,
        "fg_engaged": bool(engagement),
        "schemas": schemas,
        "infra_ok": infra_ok,
        "infra_reasons": infra_reasons,
        "log_path": str(log),
        "molmo_chat_completions_logged": molmo_chat_hits,
        "control_job": "517994",
        "treatment_job": job.name,
        "suite_root": str(job),
        "rescored": True,
        "cells": {
            "trial_1_rescue_target": {
                "control_completed": ctrl.get(1, {}).get("completed"),
                "treatment_completed": treat.get(1, {}).get("completed"),
                "rescued": rescued,
            },
            "trial_2_success_guard": {
                "control_completed": ctrl.get(2, {}).get("completed"),
                "treatment_completed": treat.get(2, {}).get("completed"),
                "guard_held": guard_held,
            },
        },
        "control_outcomes": ctrl,
        "treatment_outcomes": treat,
        "post_fork_carrier": carrier,
        "prefix_hit_records": prefix_hit_rows,
        "fg_turn_records": {
            str(k): [
                {
                    "turn": r.get("turn"),
                    "endpoint_gap": r.get("endpoint_gap"),
                    "finish_authorized": r.get("finish_authorized"),
                    "bddl_atom_true": r.get("bddl_atom_true"),
                }
                for r in v
            ]
            for k, v in turn_records.items()
        },
    }
    arm_dir = ROOT / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / f"SUMMARY_job-{job_id}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (arm_dir / "LATEST_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    arms = {arm: rescore(arm, jid) for arm, jid in ARMS.items()}
    # pi0 identity
    pi0 = {}
    for arm, summary in arms.items():
        for hit in summary.get("prefix_hit_records") or []:
            key = f"trial{hit.get('trial')}_turn{hit.get('turn')}"
            pi0.setdefault(key, {})[arm] = hit.get("code_sha256")
    pi0_identical = all(len(set(v.values())) == 1 for v in pi0.values()) if pi0 else False

    # Mechanism attribution for next vs v24
    next_s = arms["next"]
    v24_s = arms["v24"]
    attribution = {
        "question": "Is 519248 a mechanism-valid fair gain from workgap ME?",
        "fair_prefix": pi0_identical and all(a.get("prefix_ok") for a in arms.values()),
        "next_valid": next_s.get("validity") == "valid_prefix_arm",
        "next_gain": next_s.get("gain_result"),
        "next_cells": next_s.get("cells"),
        "v24_cells": v24_s.get("cells"),
        "baseline_cells": arms["baseline"].get("cells"),
        "verdict": None,
        "reasons": [],
    }
    # Distinctive workgap = placement while not holding is only meaningful if
    # near-miss geometry; success path on next t2 used holding→placement (v2.4-capable).
    attribution["reasons"].append(
        "next t2 FG turns: acquire→acquire→placement(holding=true)→placement(holding=true); "
        "no near-miss footprint-only placement label on the success path."
    )
    attribution["reasons"].append(
        "v24 also emits placement_incomplete under holding or contact; "
        "workgap-specific delta (footprint/edge without contact) is not evidenced as "
        "the cause of next's t2 success vs v24 failure."
    )
    attribution["reasons"].append(
        "Same frozen π0 across arms ⇒ post-fork live LLM variance remains; "
        "outcome flip next vs v24 is not IFF-closed to WORKGAP_ME alone."
    )
    if (
        attribution["fair_prefix"]
        and next_s.get("gain_result") == "positive"
        and next_s.get("validity") == "valid_prefix_arm"
    ):
        attribution["verdict"] = (
            "fair_positive_outcome_but_not_workgap_mechanism_closed"
        )
    else:
        attribution["verdict"] = "not_mechanism_closed_workgap_gain"

    out = {
        "ok": all(a.get("validity") == "valid_prefix_arm" for a in arms.values())
        and pi0_identical,
        "pi0_identical_across_arms": pi0_identical,
        "pi0_by_arm": pi0,
        "arms": {
            arm: {
                "validity": data.get("validity"),
                "gain_result": data.get("gain_result"),
                "cells": data.get("cells"),
                "schemas": data.get("schemas"),
                "infra_ok": data.get("infra_ok"),
                "infra_reasons": data.get("infra_reasons"),
                "log_path": data.get("log_path"),
                "post_fork_carrier": data.get("post_fork_carrier"),
                "treatment_job": data.get("treatment_job"),
            }
            for arm, data in arms.items()
        },
        "mechanism_attribution_519248": attribution,
    }
    path = ROOT / "PREFIX_ARM_SUMMARY.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
