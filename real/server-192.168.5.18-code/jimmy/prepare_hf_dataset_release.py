#!/usr/bin/env python3
"""Prepare manifests and a private Hugging Face dataset card for raw episodes."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EPISODE_RE = re.compile(r"^R[0-9A-Z]+_(\d{17})_RAW$")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_episode(source_day: Path, episode_dir: Path, ordinal: int) -> dict:
    match = EPISODE_RE.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"Unexpected episode directory name: {episode_dir.name}")
    mcap = episode_dir / f"{episode_dir.name}.mcap"
    metadata = episode_dir / "metadata.yaml"
    sidecar = source_day / f"{episode_dir.name}.json"
    for path in (mcap, metadata, sidecar):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    bag_info = sidecar_data.get("bag_info", {})
    return {
        "ordinal": ordinal,
        "episode_id": episode_dir.name,
        "recorded_name_timestamp": match.group(1),
        "recorded_name_time_note": "Timestamp embedded by recorder; source timezone Asia/Hong_Kong",
        "source_relative_directory": episode_dir.name,
        "mcap_filename": mcap.name,
        "mcap_bytes": mcap.stat().st_size,
        "metadata_bytes": metadata.stat().st_size,
        "sidecar_filename": sidecar.name,
        "sidecar_bytes": sidecar.stat().st_size,
        "sidecar_bag_length_seconds": bag_info.get("bag_length_seconds"),
        "sidecar_digest": bag_info.get("digest"),
        "sidecar_digest_algorithm": "unverified source field",
        "sidecar_created_time": bag_info.get("created_time"),
        "manifest_generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def json_lines(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def build_readme(repo_id: str, summary: dict, excluded: list[dict]) -> str:
    exclusions = "\n".join(
        f"- Ordinal {row['ordinal']}: `{row['episode_id']}` — {row['exclusion_reason']}"
        for row in excluded
    )
    return f"""---
pretty_name: Galaxea R1 Lite demonstrations 2026-08-24
---

# Galaxea R1 Lite demonstrations — 2026-08-24

Private raw robotics dataset uploaded from the R1 Lite host for controlled
research use. Repository: `{repo_id}`.

## Contents

- Source date: 2026-08-24 (Asia/Hong_Kong)
- Included episodes: {summary['included_episode_count']}
- Excluded episodes: {summary['excluded_episode_count']}
- Included bytes: {summary['included_total_bytes']}
- Format: ROS 2 MCAP episode directory, `metadata.yaml`, and recorder JSON sidecar
- Raw source path: `/home/nvidia/GalaxeaDataset/20260824/`

Files are stored below `episodes/20260824/` with the original stable episode
identifiers. `included_manifest.jsonl` is the authoritative inclusion list.

## Exclusions

{exclusions}

Excluded recordings remain preserved on the robot host for provenance and
diagnosis; they are not uploaded to this repository.

## Important camera metadata limitation

Read-only inspection found conflicting zero-distortion and nonzero-distortion
`CameraInfo` packets interleaved on the head-camera calibration topics. Raw
recorded image pixels are preserved. Downstream tooling must not select the
first calibration packet nondeterministically; it should validate the camera
topic contract and fail closed on mixed calibration fingerprints.

## Privacy and access

This repository is private. Camera streams can include the laboratory and
people. Do not make it public or redistribute it without an explicit privacy,
consent, and license review.

## Integrity

The manifests record source sizes and the recorder sidecar digest field. The
digest algorithm is not asserted because it was not declared by the recorder.
Raw MCAPs were not modified during preparation or upload.
"""


def parse_exclusions(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        episode, separator, reason = value.partition("=")
        if not separator or not episode or not reason:
            raise ValueError("--exclude must be EPISODE_ID=REASON")
        if episode in result:
            raise ValueError(f"Duplicate exclusion: {episode}")
        result[episode] = reason
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-day", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    source_day = args.source_day.expanduser().resolve()
    if not source_day.is_dir():
        raise SystemExit(f"Source day does not exist: {source_day}")
    exclusions = parse_exclusions(args.exclude)
    episode_dirs = sorted(path for path in source_day.iterdir() if path.is_dir())
    if len(episode_dirs) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} episode directories, found {len(episode_dirs)}"
        )

    included = []
    excluded = []
    for ordinal, episode_dir in enumerate(episode_dirs, start=1):
        record = read_episode(source_day, episode_dir, ordinal)
        reason = exclusions.get(episode_dir.name)
        if reason is None:
            record["status"] = "included"
            included.append(record)
        else:
            record["status"] = "excluded"
            record["exclusion_reason"] = reason
            excluded.append(record)

    missing_exclusions = sorted(set(exclusions) - {row["episode_id"] for row in excluded})
    if missing_exclusions:
        raise SystemExit(f"Excluded episode IDs were not found: {missing_exclusions}")

    summary = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "private": True,
        "source_day": str(source_day),
        "source_episode_count": len(episode_dirs),
        "included_episode_count": len(included),
        "excluded_episode_count": len(excluded),
        "included_total_bytes": sum(
            row["mcap_bytes"] + row["metadata_bytes"] + row["sidecar_bytes"]
            for row in included
        ),
        "excluded_total_bytes": sum(
            row["mcap_bytes"] + row["metadata_bytes"] + row["sidecar_bytes"]
            for row in excluded
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if summary["included_episode_count"] != args.expected_count - len(exclusions):
        raise SystemExit("Included count does not match expected count minus exclusions")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "included_manifest.jsonl", json_lines(included))
    atomic_write(output / "excluded_manifest.jsonl", json_lines(excluded))
    atomic_write(output / "dataset_summary.json", json.dumps(summary, indent=2) + "\n")
    atomic_write(output / "README.md", build_readme(args.repo_id, summary, excluded))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
