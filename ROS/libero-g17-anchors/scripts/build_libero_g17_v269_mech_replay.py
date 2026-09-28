#!/usr/bin/env python3
"""Build fail-closed, effective-seed Qwen replay assets from v256 jobs.

Unlike the generic request-hash replay, mechanism replay intentionally survives
prompt-only instrumentation changes.  Every full effective seed maps to the
historical v256 response.  Turn-0 student responses may be absent from saved
artifacts because the code-prefix hook replaced them; for those calls the
already-fingerprinted prefix block is used directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "capx_libero_g17_v269_effective_seed_replay_v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def student_content(entry: dict[str, Any]) -> str | None:
    blocks = entry.get("code_blocks") or []
    if not blocks:
        return None
    decision = str(entry.get("decision", "")).lower()
    if decision == "initial":
        return str(blocks[0])
    if decision == "regenerate":
        return "REGENERATE\n```python\n" + str(blocks[-1]) + "\n```"
    return str(blocks[-1])


def content_by_hash(run_root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(run_root.glob("**/all_responses.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            content = student_content(row)
            if content is not None:
                found[sha256_bytes(content.encode())] = content
    # Native VDM/FG reports are stored as ordinary text files, not coach_traces.
    for path in sorted(run_root.glob("**/*.txt")):
        try:
            payload = path.read_bytes()
            content = payload.decode("utf-8")
        except Exception:
            continue
        found[sha256_bytes(payload)] = content
    return found


def load_prefixes(path: Path) -> dict[int, str]:
    package = json.loads(path.read_text(encoding="utf-8"))
    prefixes: dict[int, str] = {}
    for trial, row in (package.get("trials") or {}).items():
        blocks = row.get("code_blocks") or []
        if len(blocks) != 1:
            raise RuntimeError(f"trial {trial}: v269 requires exactly one prefix block")
        content = str(blocks[0])
        expected = str((row.get("code_block_sha256") or [""])[0])
        actual = sha256_bytes(content.strip().encode())
        if expected != actual:
            raise RuntimeError(f"trial {trial}: prefix hash mismatch {actual} != {expected}")
        prefixes[int(trial)] = content
    return prefixes


def build_one(run_root: Path, prefix_path: Path, global_seed: int) -> dict[str, Any]:
    audit_path = run_root / "qwen_proxy_audit.jsonl"
    rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    saved = content_by_hash(run_root)
    prefixes = load_prefixes(prefix_path)
    entries: dict[str, dict[str, Any]] = {}
    for audit_index, row in enumerate(rows):
        if not row.get("ok", False):
            raise RuntimeError(f"audit row {audit_index} is not successful")
        seed = int(row["effective_request_seed"])
        response_hash = str(row["response_content_sha256"])
        content = saved.get(response_hash)
        role = "historical_v256"
        if content is None:
            delta = seed - int(global_seed)
            trial, turn = divmod(delta, 1000)
            if turn != 0 or trial not in prefixes:
                raise RuntimeError(
                    f"cannot recover seed={seed} hash={response_hash} "
                    f"(trial={trial}, turn={turn})"
                )
            content = prefixes[trial]
            role = "fingerprinted_prefix_substitute"
        key = str(seed)
        content_hash = sha256_bytes(content.encode())
        entry = {
            "content": content,
            "content_sha256": content_hash,
            "historical_response_sha256": response_hash,
            "historical_request_sha256": row.get("request_sha256"),
            "audit_index": audit_index,
            "role": role,
        }
        if key in entries and entries[key]["content_sha256"] != content_hash:
            raise RuntimeError(f"conflicting responses for effective seed {seed}")
        entries[key] = entry
    return {
        "schema": SCHEMA,
        "source_run_root": str(run_root.resolve()),
        "source_audit_sha256": sha256_bytes(audit_path.read_bytes()),
        "prefix_package": str(prefix_path.resolve()),
        "prefix_package_sha256": sha256_bytes(prefix_path.read_bytes()),
        "global_seed": int(global_seed),
        "entries_count": len(entries),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v256-root", type=Path, required=True)
    parser.add_argument("--prefix-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-seed", type=int, default=20260718)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = sorted(args.v256_root.glob("runs/libero_spatial/arm-v256/rep-*/job-*"))
    if len(jobs) != 5:
        raise SystemExit(f"expected five v256 jobs, found {len(jobs)}")
    manifest: dict[str, Any] = {"schema": SCHEMA, "replicates": {}}
    for run_root in jobs:
        match = re.search(r"/rep-(\d+)/", str(run_root))
        if match is None:
            raise RuntimeError(f"cannot parse replicate from {run_root}")
        rep = int(match.group(1))
        package = build_one(run_root, args.prefix_package, args.global_seed)
        output = args.output_dir / f"v256_rep_{rep:02d}.json"
        output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
        manifest["replicates"][str(rep)] = {
            "path": str(output.resolve()),
            "sha256": sha256_bytes(output.read_bytes()),
            "entries_count": package["entries_count"],
        }
    manifest_path = args.output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
