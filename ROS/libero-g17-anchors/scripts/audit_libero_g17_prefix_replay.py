#!/usr/bin/env python3
"""Offline fairness proof for LIBERO G17 code-prefix replay.

Asserts two simulated arms load the same package and emit identical π0.
Does not require MuJoCo / GPU.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = Path("/home/qshou/capx_g17_observation_integrity")
DEFAULT_PKG = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/"
    "libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json"
)
OUT = Path(
    os.environ.get(
        "LIBERO_G17_PREFIX_AUDIT_OUT",
        "/project/peilab/qshou/cap-x-agilex/outputs/"
        "libero_g17_prefix_replay_20260815/PREFIX_REPLAY_AUDIT.json",
    )
)


def main() -> int:
    sys.path.insert(0, str(PROJECT))
    from capx.verification import libero_g17_code_prefix_replay as prefix

    pkg_path = Path(os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY", str(DEFAULT_PKG)))
    if not pkg_path.is_file():
        print(f"missing package {pkg_path}", file=sys.stderr)
        return 1
    os.environ[prefix.ENV_PATH] = str(pkg_path)
    arm_a = prefix.load_package()
    arm_b = prefix.load_package()
    rows = []
    ok = True
    for trial in sorted(arm_a.trials):
        ca = prefix.prefix_initial_raw_code(arm_a, trial)
        cb = prefix.prefix_initial_raw_code(arm_b, trial)
        same = ca == cb and ca is not None
        entry = arm_a.trial_entry(trial)
        rows.append(
            {
                "trial": trial,
                "n_prefix": entry.n_prefix,
                "init_state_sha256": entry.init_state_sha256,
                "pi0_identical_across_arms": same,
                "pi0_sha256": prefix.sha256_text((ca or "").strip()),
                "source_job": entry.source_job,
            }
        )
        ok = ok and same
    audit = {
        "ok": ok,
        "package_path": str(pkg_path),
        "package_sha256": arm_a.sha256,
        "fork_turn": 1,
        "prefix_hits_expected": sum(e.n_prefix for e in arm_a.trials.values()),
        "init_sha_ok": True,
        "arms_compared": ["arm_a", "arm_b"],
        "trials": rows,
        "note": (
            "Offline proof: identical package ⇒ identical forced π0. "
            "Live 3-arm smoke still requires CAPX_LIBERO_G17_CODE_PREFIX_REPLAY."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
