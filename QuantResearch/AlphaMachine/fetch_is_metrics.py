#!/usr/bin/env python3
"""Read IS metrics for the alphas recorded in a sample run ledger and rank them.

Read-only: issues GET /alphas/{id} requests and writes a local report. Never submits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from sample_new_alphas import API_ROOT, BrainClient, BrainError, safe_message, utc_now

GATE = {"min_sharpe": 1.25, "min_fitness": 1.0, "turnover_range": (0.01, 0.7)}
METRIC_KEYS = (
    "sharpe",
    "fitness",
    "turnover",
    "returns",
    "margin",
    "drawdown",
    "longCount",
    "shortCount",
)
MAX_RATE_LIMIT_RETRIES = 5


def get_with_backoff(client: BrainClient, url: str) -> requests.Response:
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        response = client.request("GET", url)
        if response.status_code != 429:
            return response
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else 5.0
        except ValueError:
            delay = 5.0
        time.sleep(min(max(delay, 1.0), 60.0) * (attempt + 1))
    return response


def fetch_metrics(client: BrainClient, alpha_id: str) -> dict[str, Any]:
    response = get_with_backoff(client, f"{API_ROOT}/alphas/{alpha_id}")
    if not response.ok:
        return {"error": f"HTTP {response.status_code}: {safe_message(response)}"}
    body = response.json()
    is_block = body.get("is") or {}
    metrics = {key: is_block.get(key) for key in METRIC_KEYS}
    metrics["grade"] = body.get("grade")
    metrics["platform_status"] = body.get("status")
    checks = is_block.get("checks") or []
    metrics["failed_checks"] = [
        c.get("name") for c in checks if isinstance(c, dict) and c.get("result") == "FAIL"
    ]
    return metrics


def evaluate(metrics: dict[str, Any]) -> dict[str, Any]:
    sharpe = metrics.get("sharpe")
    fitness = metrics.get("fitness")
    turnover = metrics.get("turnover")
    if not isinstance(sharpe, (int, float)):
        return {"rankable": False, "effective_sharpe": None, "needs_inversion": False, "gate": "no_metrics"}

    needs_inversion = sharpe < 0
    effective_sharpe = abs(sharpe)
    effective_fitness = abs(fitness) if isinstance(fitness, (int, float)) else None

    reasons: list[str] = []
    if effective_sharpe < GATE["min_sharpe"]:
        reasons.append(f"sharpe<{GATE['min_sharpe']}")
    if effective_fitness is None or effective_fitness < GATE["min_fitness"]:
        reasons.append(f"fitness<{GATE['min_fitness']}")
    if not isinstance(turnover, (int, float)):
        reasons.append("turnover_missing")
    elif not GATE["turnover_range"][0] <= turnover <= GATE["turnover_range"][1]:
        reasons.append("turnover_out_of_range")

    return {
        "rankable": True,
        "effective_sharpe": effective_sharpe,
        "effective_fitness": effective_fitness,
        "needs_inversion": needs_inversion,
        "gate": "pass" if not reasons else "fail",
        "gate_reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, help="Path to a sample run ledger JSON file.")
    parser.add_argument("--out", type=Path, help="Where to write the ranked report JSON.")
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text())
    rows = [r for r in ledger["candidates"] if r.get("alpha_id")]
    if not rows:
        print("No alpha IDs found in ledger.")
        return 2

    client = BrainClient()
    report: list[dict[str, Any]] = []
    for row in rows:
        try:
            metrics = fetch_metrics(client, row["alpha_id"])
        except (BrainError, requests.RequestException, ValueError) as error:
            metrics = {"error": str(error)}
        record = {
            "ordinal": row["ordinal"],
            "family": row["family"],
            "alpha_id": row["alpha_id"],
            "sim_state": row["state"],
            "expression": row["expression"],
            **metrics,
            **evaluate(metrics),
        }
        report.append(record)

    ranked = sorted(
        report,
        key=lambda r: (r.get("effective_sharpe") is None, -(r.get("effective_sharpe") or 0.0)),
    )
    for position, record in enumerate(ranked, start=1):
        record["rank"] = position

    out_path = args.out or args.ledger.with_name(args.ledger.stem + "-is-metrics.json")
    out_path.write_text(
        json.dumps({"generated_at": utc_now(), "gate": GATE, "source_ledger": str(args.ledger), "alphas": ranked}, indent=2) + "\n"
    )

    errors = [r for r in ranked if r.get("error")]
    print(f"Fetched {len(ranked) - len(errors)}/{len(ranked)} alphas with IS metrics.")
    if errors:
        print(f"Fetch errors: {[r['ordinal'] for r in errors]}")
    print(f"Report: {out_path}\n")

    header = f"{'#':>2} {'ord':>3} {'family':18s} {'alpha':9s} {'sharpe':>7} {'fitness':>7} {'turnover':>8} {'ret':>7} {'gate':5s} {'inv':3s}"
    print(header)
    print("-" * len(header))
    for record in ranked:
        if record.get("error"):
            print(f"{record['rank']:>2} {record['ordinal']:>3} {record['family']:18s} {record['alpha_id']:9s}  error: {record['error'][:60]}")
            continue
        fmt = lambda v, p=2: f"{v:.{p}f}" if isinstance(v, (int, float)) else "-"
        print(
            f"{record['rank']:>2} {record['ordinal']:>3} {record['family']:18s} {record['alpha_id']:9s} "
            f"{fmt(record.get('sharpe')):>7} {fmt(record.get('fitness')):>7} {fmt(record.get('turnover'), 3):>8} "
            f"{fmt(record.get('returns'), 3):>7} {record.get('gate', '-'):5s} {'yes' if record.get('needs_inversion') else '-':3s}"
        )

    passing = [r for r in ranked if r.get("gate") == "pass"]
    print(f"\nGate pass ({GATE['min_sharpe']} sharpe / {GATE['min_fitness']} fitness / turnover {GATE['turnover_range']}): {len(passing)}")
    for record in passing:
        print(f"  #{record['rank']} ord{record['ordinal']:02d} {record['alpha_id']} {record['family']} sharpe={record.get('sharpe')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
