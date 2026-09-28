#!/usr/bin/env python3
"""Resumably upload a manifest-selected raw MCAP dataset to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

METADATA_FILES = (
    "README.md",
    "included_manifest.jsonl",
    "excluded_manifest.jsonl",
    "dataset_summary.json",
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json_lines(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def episode_operations(
    source_day: Path,
    row: dict,
    commit_operation_add,
) -> tuple[list, list[str]]:
    episode = row["episode_id"]
    episode_dir = source_day / episode
    local_paths = (
        episode_dir / row["mcap_filename"],
        episode_dir / "metadata.yaml",
        source_day / row["sidecar_filename"],
    )
    expected_sizes = (
        row["mcap_bytes"],
        row["metadata_bytes"],
        row["sidecar_bytes"],
    )
    for path, expected in zip(local_paths, expected_sizes):
        if not path.is_file() or path.stat().st_size != expected:
            raise RuntimeError(f"Source changed or missing: {path}")
    remote_paths = [
        f"episodes/20260824/{episode}/{row['mcap_filename']}",
        f"episodes/20260824/{episode}/metadata.yaml",
        f"episodes/20260824/{row['sidecar_filename']}",
    ]
    operations = [
        commit_operation_add(path_in_repo=remote, path_or_fileobj=local)
        for remote, local in zip(remote_paths, local_paths)
    ]
    return operations, remote_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-day", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required on the upload host") from exc

    source_day = args.source_day.expanduser().resolve()
    release_dir = args.release_dir.expanduser().resolve()
    summary = json.loads((release_dir / "dataset_summary.json").read_text())
    included = load_json_lines(release_dir / "included_manifest.jsonl")
    excluded = load_json_lines(release_dir / "excluded_manifest.jsonl")
    if summary["repo_id"] != args.repo_id or not summary.get("private"):
        raise SystemExit("Summary repo identity/privacy does not match arguments")
    if len(included) != summary["included_episode_count"]:
        raise SystemExit("Included manifest count does not match summary")
    if len(excluded) != summary["excluded_episode_count"]:
        raise SystemExit("Excluded manifest count does not match summary")
    excluded_ids = {row["episode_id"] for row in excluded}
    if excluded_ids & {row["episode_id"] for row in included}:
        raise SystemExit("An episode is both included and excluded")

    selected = included[: args.max_episodes or None]
    for row in selected:
        episode_operations(source_day, row, CommitOperationAdd)
    print(
        f"Validated {len(selected)} included episodes; "
        f"excluded {len(excluded_ids)}; repo={args.repo_id}; private=true",
        flush=True,
    )
    if args.dry_run:
        return

    api = HfApi()
    identity = api.whoami().get("name")
    if not identity or args.repo_id.split("/", 1)[0] != identity:
        raise SystemExit(f"Authenticated user {identity!r} does not own {args.repo_id!r}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    metadata_operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=release_dir / name)
        for name in METADATA_FILES
    ]
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=metadata_operations,
        commit_message="Initialize private dataset card and manifests",
    )

    state_path = release_dir / "upload_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "repo_id": args.repo_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "uploaded": {},
            "failed": {},
        }
    if state.get("repo_id") != args.repo_id:
        raise SystemExit("Upload state belongs to a different repository")

    total = len(selected)
    for sequence, row in enumerate(selected, start=1):
        episode = row["episode_id"]
        if episode in excluded_ids:
            raise SystemExit(f"Refusing to upload excluded episode: {episode}")
        operations, remote_paths = episode_operations(
            source_day,
            row,
            CommitOperationAdd,
        )
        if episode in state["uploaded"]:
            print(f"[{sequence}/{total}] state skip {episode}", flush=True)
            continue
        if all(
            api.file_exists(args.repo_id, path, repo_type="dataset")
            for path in remote_paths
        ):
            state["uploaded"][episode] = {
                "status": "already_present",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(state_path, state)
            print(f"[{sequence}/{total}] Hub skip {episode}", flush=True)
            continue
        print(
            f"[{sequence}/{total}] uploading {episode} "
            f"({row['mcap_bytes']} MCAP bytes)",
            flush=True,
        )
        try:
            commit = api.create_commit(
                repo_id=args.repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Upload episode {sequence:03d} of {total}",
                commit_description=episode,
                num_threads=3,
            )
        except BaseException as exc:
            state["failed"][episode] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(state_path, state)
            raise
        state["uploaded"][episode] = {
            "status": "uploaded",
            "commit_oid": getattr(commit, "oid", None),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        state["failed"].pop(episode, None)
        atomic_json(state_path, state)
        print(f"[{sequence}/{total}] complete {episode}", flush=True)

    repo_files = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    expected_mcaps = {
        f"episodes/20260824/{row['episode_id']}/{row['mcap_filename']}"
        for row in selected
    }
    missing = sorted(expected_mcaps - repo_files)
    leaked = sorted(path for path in repo_files if any(value in path for value in excluded_ids))
    if missing or leaked:
        raise SystemExit(f"Hub verification failed: missing={missing}, excluded_present={leaked}")
    completion = {
        "repo_id": args.repo_id,
        "private": True,
        "verified_included_mcap_count": len(expected_mcaps),
        "verified_excluded_episode_ids_absent": sorted(excluded_ids),
        "repo_file_count": len(repo_files),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(release_dir / "UPLOAD_COMPLETE.json", completion)
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=release_dir / "UPLOAD_COMPLETE.json",
        path_in_repo="UPLOAD_COMPLETE.json",
        commit_message="Record verified upload completion",
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
