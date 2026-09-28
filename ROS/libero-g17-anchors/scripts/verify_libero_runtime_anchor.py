#!/usr/bin/env python3
"""Fail-closed provenance check for the shared LIBERO runtime anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("anchor", type=Path)
    args = parser.parse_args()
    anchor = args.anchor.resolve()
    manifest_path = anchor / "CRITICAL_SHA256.json"
    manifest = json.loads(manifest_path.read_text())
    if Path(manifest["anchor"]).resolve() != anchor:
        raise SystemExit("runtime anchor identity mismatch")
    for relative, expected in manifest["files"].items():
        path = anchor / relative
        if not path.is_file():
            raise SystemExit(f"runtime anchor missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(
                f"runtime anchor drift: {relative} {actual} != {expected}"
            )
    print(
        f"libero_runtime_anchor_ok path={anchor} "
        f"files={len(manifest['files'])} schema={manifest['schema']}"
    )


if __name__ == "__main__":
    main()
