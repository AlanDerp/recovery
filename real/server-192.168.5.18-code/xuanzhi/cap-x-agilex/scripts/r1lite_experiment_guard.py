#!/usr/bin/env python3
"""Create immutable, restorable snapshots around R1 Lite experiments.

This tool never publishes ROS commands. Use ``snapshot`` before an experiment,
``finalize`` afterward, and ``restore-files`` only with an explicit run ID.
Robot motion is deliberately not restored automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any


REPO = Path("/home/nvidia/xuanzhi/cap-x-agilex")
RUNS = REPO / "outputs/r1lite_audit_runs"
CRITICAL_FILES = [
    "capx/envs/simulators/r1lite_real.py",
    "capx/integrations/r1lite/control.py",
    "capx/integrations/r1lite/kinematics.py",
    "scripts/r1lite_experiment_guard.py",
    "scripts/start_r1lite_moca_on_target.sh",
    "scripts/run_r1lite_tennis_ball_in_box_pipeline.sh",
    "scripts/r1lite_audit_tcp_frames.py",
    "scripts/r1lite_audit_moca_joint_path.py",
    "scripts/r1lite_validate_moca_basis.py",
    "scripts/r1lite_validate_cross_configuration.py",
    "scripts/r1lite_validate_settling_decoupling.py",
    "scripts/r1lite_verify_nominal_plan.py",
    "scripts/r1lite_search_place_feasibility.py",
    "scripts/r1lite_experiment_guard.py",
    "scripts/r1lite_regenerate_10_keyframes_y_stable.py",
    "scripts/r1lite_regenerate_forward_first_10_keyframes.py",
    "scripts/r1lite_regenerate_tcp_consistent_plan.py",
    "scripts/r1lite_execute_10_keyframes_xyz_corrected.sh",
    "scripts/r1lite_execute_keyframes_pose_closed_loop.sh",
    "scripts/r1lite_servo_pick_place_position_only.sh",
    "docs/r1lite_tcp_frame_contract.md",
    "docs/r1lite_experiment_standard.md",
    "docs/r1lite_root_cause_ledger_20260724.md",
    "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml",
    "docs/r1lite_experiment_standard.md",
    "docs/r1lite_root_cause_ledger_20260724.md",
    "docs/r1lite_tcp_frame_contract.md",
    "tests/test_r1lite_tcp_frames.py",
    "tests/test_r1lite_candidate_kinematics.py",
    "outputs/r1lite_tennis_ball_in_box/offline_ik_pick_place_joint_poses_3d.json",
]
ROS_SETUP = (
    "set +u; source /opt/ros/humble/setup.bash; "
    "source /home/nvidia/galaxea/install/local_setup.bash; set -u; "
    "export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 "
    "FASTRTPS_DEFAULT_PROFILES_FILE="
    "/opt/galaxea/find_server/super_client_configuration_file.xml; "
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: str, timeout: float = 15.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout}s",
        }


def file_manifest() -> list[dict[str, Any]]:
    rows = []
    for relative in CRITICAL_FILES:
        path = REPO / relative
        rows.append(
            {
                "relative_path": relative,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else None,
            }
        )
    return rows


def capture_live_state(target: Path) -> dict[str, Any]:
    commands = {
        "nodes": ROS_SETUP + "ros2 node list",
        "arm_target_graph": (
            ROS_SETUP
            + "ros2 topic info -v "
            "/motion_target/target_joint_state_arm_right"
        ),
        "capx_target_graph": (
            ROS_SETUP
            + "ros2 topic info -v "
            "/capx/motion_target/target_joint_state_arm_right"
        ),
        "processes": (
            "pgrep -af '[m]oca_main_node|[h]omogeneous_teleoperation|"
            "[r]elaxed_ik.py 0 1 0|capx/envs/launch.py' || true"
        ),
        "tmux": "tmux ls 2>/dev/null || true",
    }
    captured = {
        key: command_output(command)
        for key, command in commands.items()
    }
    audit_path = target / "moca_readonly.json"
    audit = command_output(
        ROS_SETUP
        + f"{REPO}/.capx_r1lite_venv/bin/python "
        f"{REPO}/scripts/r1lite_audit_moca_joint_path.py "
        f"--duration-s 3 --output {audit_path}",
        timeout=20.0,
    )
    captured["readonly_audit"] = audit
    return captured


def create_snapshot(run_id: str, intent: str) -> Path:
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    before = run_dir / "before"
    files_dir = before / "files"
    files_dir.mkdir(parents=True)
    manifest = file_manifest()
    for row in manifest:
        if not row["exists"]:
            continue
        source = REPO / row["relative_path"]
        destination = files_dir / row["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "intent": intent,
        "created_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER"),
        "workspace": str(REPO),
        "git_available": (REPO / ".git").exists(),
        "critical_files": manifest,
        "live_state": capture_live_state(before),
        "restore_command": (
            f"python {REPO}/scripts/r1lite_experiment_guard.py "
            f"restore-files --run-id {run_id} --confirm {run_id}"
        ),
        "robot_state_restore_policy": (
            "Never automatic. Review before/moca_readonly.json and issue a "
            "separate safety-approved motion plan if physical restoration is needed."
        ),
    }
    (before / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return run_dir


def finalize(run_id: str, outcome: str) -> Path:
    run_dir = RUNS / run_id
    if not (run_dir / "before/manifest.json").exists():
        raise FileNotFoundError(f"Unknown snapshot run ID: {run_id}")
    after = run_dir / "after"
    after.mkdir(exist_ok=False)
    current = file_manifest()
    before = json.loads(
        (run_dir / "before/manifest.json").read_text()
    )
    before_hashes = {
        row["relative_path"]: row["sha256"]
        for row in before["critical_files"]
    }
    changes = [
        {
            **row,
            "before_sha256": before_hashes.get(row["relative_path"]),
            "changed": before_hashes.get(row["relative_path"])
            != row["sha256"],
        }
        for row in current
    ]
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "outcome": outcome,
        "finalized_at_unix": time.time(),
        "critical_files": changes,
        "live_state": capture_live_state(after),
    }
    (after / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return after


def restore_files(run_id: str, confirm: str) -> None:
    if confirm != run_id:
        raise ValueError("--confirm must exactly match --run-id")
    run_dir = RUNS / run_id
    manifest_path = run_dir / "before/manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Unknown snapshot run ID: {run_id}")
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["critical_files"]:
        source = run_dir / "before/files" / row["relative_path"]
        destination = REPO / row["relative_path"]
        if row["exists"]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        elif destination.exists():
            destination.unlink()
    print("Files restored. Robot/process state was not changed.")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--run-id", required=True)
    snapshot.add_argument("--intent", required=True)
    finish = subparsers.add_parser("finalize")
    finish.add_argument("--run-id", required=True)
    finish.add_argument("--outcome", required=True)
    restore = subparsers.add_parser("restore-files")
    restore.add_argument("--run-id", required=True)
    restore.add_argument("--confirm", required=True)
    args = parser.parse_args()

    if args.command == "snapshot":
        print(create_snapshot(args.run_id, args.intent))
    elif args.command == "finalize":
        print(finalize(args.run_id, args.outcome))
    else:
        restore_files(args.run_id, args.confirm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
