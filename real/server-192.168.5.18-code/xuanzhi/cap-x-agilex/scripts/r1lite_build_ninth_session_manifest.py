#!/usr/bin/env python3
"""Bind both approved sandwich phases into one immutable session manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from capx.integrations.r1lite.sandwich_plan import (  # noqa: E402
    build_session_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for phase in ("donut", "top"):
        parser.add_argument(f"--{phase}-plan", type=Path, required=True)
        parser.add_argument(f"--{phase}-policy", type=Path, required=True)
        parser.add_argument(f"--{phase}-bundle", type=Path, required=True)
        parser.add_argument(f"--{phase}-certificate", type=Path, required=True)
        parser.add_argument(f"--{phase}-execution-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _phase(args: argparse.Namespace, name: str) -> dict[str, object]:
    paths = {
        "plan": getattr(args, f"{name}_plan"),
        "policy": getattr(args, f"{name}_policy"),
        "bundle": getattr(args, f"{name}_bundle"),
        "certificate": getattr(args, f"{name}_certificate"),
        "execution_audit": getattr(args, f"{name}_execution_audit"),
    }
    audit = json.loads(paths["execution_audit"].read_text())
    if audit.get("passed") is not True or audit.get("home_completed") is not True:
        raise ValueError(f"{name} phase did not pass execution and home gates")
    return {
        f"{label}_path": str(path.resolve())
        for label, path in paths.items()
    } | {
        f"{label}_sha256": sha256_file(path)
        for label, path in paths.items()
    }


def main() -> None:
    args = parse_args()
    manifest = build_session_manifest(
        donut_phase=_phase(args, "donut"),
        top_phase=_phase(args, "top"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(manifest))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
