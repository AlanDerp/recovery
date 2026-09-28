"""Build v256 grasp replay package from fair-FG logs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    trials: dict[int, list] = {}
    cur = None
    pending = None
    mfn = re.search(r"r(\d+)-(\d+)\.out$", path.name)
    rep = int(mfn.group(1)) if mfn else None
    job = mfn.group(2) if mfn else None
    for line in text.splitlines():
        m = re.search(r"trial_start trial=(\d+):", line)
        if m:
            cur = int(m.group(1))
            trials.setdefault(cur, [])
            pending = None
            continue
        if cur is None:
            continue
        mp = re.search(r"Grasp sample position for ([^:]+): \[([^\]]+)\]", line)
        if mp:
            pending = (
                mp.group(1).strip(),
                [float(x) for x in mp.group(2).split()],
            )
            continue
        mq = re.search(
            r"Grasp sample quaternion wxyz for ([^:]+): \[([^\]]+)\]", line
        )
        if mq and pending is not None:
            quat = [float(x) for x in mq.group(2).split()]
            trials[cur].append(
                {
                    "object": pending[0],
                    "position": pending[1],
                    "quaternion_wxyz": quat,
                }
            )
            pending = None
    return rep, job, trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--expt",
        default="/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v256_20260817",
    )
    ap.add_argument(
        "--out",
        default="/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817",
    )
    args = ap.parse_args()
    expt = Path(args.expt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reps = {}
    sources = []
    for path in sorted((expt / "logs").glob("libero-g17-v256-r*.out")):
        rep, job, trials = parse_log(path)
        assert rep is not None
        reps[str(rep)] = {
            "source_job": job,
            "source_log": path.name,
            "trials": {str(t): c for t, c in sorted(trials.items())},
        }
        sources.append(
            {
                "rep": rep,
                "job": job,
                "n_calls": sum(len(v) for v in trials.values()),
            }
        )
        print(
            f"rep-{rep:02d} job={job} "
            + "{"
            + ", ".join(f"{t}:{len(c)}" for t, c in sorted(trials.items()))
            + "}"
        )
    pkg = {
        "schema": "capx_libero_g17_grasp_replay_v1",
        "source_arm": "v256",
        "source_expt": str(expt),
        "suite_name": "libero_spatial",
        "task_id": 0,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": "sequential_per_trial_call_index; bypass GraspNet only",
        "overflow": "strict_raise_GRASP_REPLAY_MISS",
        "reps": reps,
    }
    pkg_path = out / "grasp_replay_v1.json"
    raw = json.dumps(pkg, indent=2) + "\n"
    pkg_path.write_text(raw)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    (out / "PACKAGE_SHA256.txt").write_text(sha + "\n")
    (out / "PACKAGE_META.json").write_text(
        json.dumps({"sha256": sha, "sources": sources, "path": str(pkg_path)}, indent=2)
        + "\n"
    )
    print("wrote", pkg_path)
    print("sha256", sha)


if __name__ == "__main__":
    main()
