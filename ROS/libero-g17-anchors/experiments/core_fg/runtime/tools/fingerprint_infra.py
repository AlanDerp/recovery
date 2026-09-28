#!/usr/bin/env python3
"""Compute and enforce the frozen cube-experiment infrastructure fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "capx_cube_infra_fingerprint_v1"
LOCK_SCHEMA = "capx_cube_infra_lock_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _infra_files(project: Path, file_list: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line_number, raw in enumerate(
        file_list.read_text(encoding="utf-8").splitlines(), 1
    ):
        relative = raw.strip()
        if not relative or relative.startswith("#"):
            continue
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                f"Unsafe infra path at {file_list}:{line_number}: {relative!r}"
            )
        path = project / candidate
        if not path.is_file():
            raise ValueError(f"Frozen infra file is missing: {path}")
        hashes[relative] = _sha256_file(path)
    if not hashes:
        raise ValueError("Frozen infra file list is empty")
    return hashes


def _python_packages(python: Path) -> list[dict[str, str]]:
    script = r"""
import importlib.metadata
import json

records = []
for distribution in importlib.metadata.distributions():
    direct_url = distribution.read_text("direct_url.json") or ""
    records.append(
        {
            "name": (distribution.metadata.get("Name") or "").lower(),
            "version": distribution.version or "",
            "direct_url": direct_url.strip(),
        }
    )
records.sort(key=lambda item: (item["name"], item["version"], item["direct_url"]))
print(json.dumps(records, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
"""
    output = subprocess.check_output(
        [str(python), "-c", script],
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    value = json.loads(output)
    if not isinstance(value, list):
        raise ValueError("Python package fingerprint is not a list")
    return value


def build_fingerprint(project: Path, file_list: Path, python: Path) -> dict[str, Any]:
    project = project.resolve()
    file_list = file_list.resolve()
    python_real = python.resolve()
    files = _infra_files(project, file_list)
    packages = _python_packages(python)
    python_version = subprocess.check_output(
        [str(python), "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    return {
        "schema": SCHEMA,
        "project": str(project),
        "file_list": str(file_list),
        "infra_files": files,
        "infra_bundle_sha256": _sha256_bytes(_canonical(files)),
        "python_executable": str(python_real),
        "python_executable_sha256": _sha256_file(python_real),
        "python_version": python_version,
        "python_packages": packages,
        "python_packages_sha256": _sha256_bytes(_canonical(packages)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--file-list", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fingerprint = build_fingerprint(args.project, args.file_list, args.python)
    fingerprint["lock_path"] = str(args.lock.resolve()) if args.lock else None
    fingerprint["lock_matched"] = None
    if args.lock:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        if lock.get("schema") != LOCK_SCHEMA:
            raise ValueError(f"Unexpected infra lock schema: {args.lock}")
        expected = {
            "infra_bundle_sha256": lock.get("infra_bundle_sha256"),
            "python_executable_sha256": lock.get("python_executable_sha256"),
            "python_version": lock.get("python_version"),
            "python_packages_sha256": lock.get("python_packages_sha256"),
        }
        actual = {key: fingerprint[key] for key in expected}
        mismatches = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        fingerprint["lock_expected"] = expected
        fingerprint["lock_mismatches"] = mismatches
        fingerprint["lock_matched"] = not mismatches

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if fingerprint["lock_matched"] is False:
        raise SystemExit(
            "Cube experiment infrastructure does not match the committed lock: "
            + json.dumps(fingerprint["lock_mismatches"], sort_keys=True)
        )
    print(
        json.dumps(
            {
                "infra_bundle_sha256": fingerprint["infra_bundle_sha256"],
                "lock_matched": fingerprint["lock_matched"],
                "python_packages_sha256": fingerprint["python_packages_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
