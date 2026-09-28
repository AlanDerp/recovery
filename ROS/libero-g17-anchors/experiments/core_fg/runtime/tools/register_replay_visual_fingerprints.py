#!/usr/bin/env python3
"""Bind a frozen request replay to its source visual-feedback frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def rgb_fingerprint(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        payload = (
            rgb.width.to_bytes(4, "big")
            + rgb.height.to_bytes(4, "big")
            + rgb.tobytes()
        )
    return hashlib.sha256(payload).hexdigest()


def unique_frame(run_root: Path, task: str, trial: int, turn: int) -> Path:
    paths = sorted(
        run_root.glob(
            f"**/{task}/trial_{trial:02d}_*/visual_feedback_{turn:02d}.png"
        )
    )
    if not paths:
        raise ValueError(
            f"missing source visual feedback task={task} trial={trial} turn={turn}"
        )
    fingerprints = {rgb_fingerprint(path) for path in paths}
    if len(fingerprints) != 1:
        raise ValueError(
            f"conflicting source visual feedback task={task} trial={trial} turn={turn}"
        )
    return paths[0]


def request_frames(native_mode: str, turn: int) -> list[int]:
    if native_mode == "M2":
        return [0] if turn == 0 else [0, turn]
    if native_mode != "M3":
        raise ValueError(f"unsupported native mode: {native_mode}")

    # M3 alternates a visual-difference request with a text-only policy
    # request. Query 0 describes frame 0; query 2k compares frames k-1 and k.
    if turn == 0:
        return [0]
    if turn % 2 == 1:
        return []
    current_frame = turn // 2
    return [current_frame - 1, current_frame]


def register(
    path: Path,
    task: str,
    global_seed: int,
    *,
    native_mode: str = "M2",
    total_trials: int = 10,
    fixed_episode_seed: int | None = None,
) -> None:
    package = json.loads(path.read_text(encoding="utf-8"))
    if package.get("schema") != "capx_request_hash_replay_v1":
        raise ValueError(f"unsupported replay schema: {path}")
    run_root = Path(package["source_run_root"])
    if fixed_episode_seed is not None and total_trials != 1:
        raise ValueError("a fixed episode seed requires exactly one trial")
    initial_fingerprints: dict[int, str] = {}
    for entry in package["entries"].values():
        request_seed = int(entry["effective_request_seed"])
        if fixed_episode_seed is None:
            trial = (request_seed - global_seed) // 1000
            turn = request_seed - (global_seed + trial * 1000)
        else:
            trial = 1
            turn = request_seed - fixed_episode_seed * 1000
        if trial not in range(1, total_trials + 1) or turn < 0:
            raise ValueError(f"invalid request seed {request_seed}")
        frame_indices = request_frames(native_mode, turn)
        fingerprints = []
        for frame_index in frame_indices:
            if frame_index == 0:
                if trial not in initial_fingerprints:
                    initial_fingerprints[trial] = rgb_fingerprint(
                        unique_frame(run_root, task, trial, 0)
                    )
                fingerprints.append(initial_fingerprints[trial])
            else:
                fingerprints.append(
                    rgb_fingerprint(
                        unique_frame(run_root, task, trial, frame_index)
                    )
                )
        entry["trial"] = trial
        entry["turn"] = turn
        entry["request_image_rgb_sha256"] = fingerprints
    package["visual_fingerprint_schema"] = "rgb_shape_bytes_sha256_v1"
    package["visual_fingerprint_native_mode"] = native_mode
    package["visual_fingerprint_global_seed"] = global_seed
    package["visual_fingerprint_total_trials"] = total_trials
    package["visual_fingerprint_fixed_episode_seed"] = fixed_episode_seed
    path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--global-seed", type=int, default=20260718)
    parser.add_argument("--native-mode", choices=("M2", "M3"), default="M2")
    parser.add_argument("--total-trials", type=int, default=10)
    parser.add_argument("--fixed-episode-seed", type=int)
    args = parser.parse_args()
    register(
        args.replay,
        args.task,
        args.global_seed,
        native_mode=args.native_mode,
        total_trials=args.total_trials,
        fixed_episode_seed=args.fixed_episode_seed,
    )


if __name__ == "__main__":
    main()
