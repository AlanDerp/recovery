#!/usr/bin/env python3
"""Fail if an experiment module would import outside its frozen source tree."""

from __future__ import annotations

import argparse
from importlib.machinery import PathFinder
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("modules", nargs="+")
    args = parser.parse_args()

    project = args.project.resolve()
    records = []
    failures = []
    for module in args.modules:
        parts = module.split(".")
        spec = PathFinder.find_spec(parts[0])
        search_path = None if spec is None else spec.submodule_search_locations
        for index in range(1, len(parts)):
            if not search_path:
                spec = None
                break
            spec = PathFinder.find_spec(".".join(parts[: index + 1]), search_path)
            search_path = None if spec is None else spec.submodule_search_locations
        origin = None if spec is None else spec.origin
        resolved = None
        within_project = False
        if origin and origin not in {"built-in", "frozen"}:
            resolved_path = Path(origin).resolve()
            resolved = str(resolved_path)
            within_project = resolved_path.is_relative_to(project)
        record = {
            "module": module,
            "origin": origin,
            "resolved_origin": resolved,
            "within_frozen_project": within_project,
        }
        records.append(record)
        if spec is None or not within_project:
            failures.append(record)

    payload = {
        "schema": "capx_runtime_import_audit_v1",
        "frozen_project": str(project),
        "status": "valid" if not failures else "invalid",
        "modules": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if failures:
        names = ", ".join(record["module"] for record in failures)
        raise SystemExit(f"runtime import isolation failed for: {names}")


if __name__ == "__main__":
    main()
