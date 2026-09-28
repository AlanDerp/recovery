#!/usr/bin/env python3
"""Validate MOCA mapping and TCP residual across reversible configurations."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml


REPO = Path("/home/nvidia/xuanzhi/cap-x-agilex")
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel
from scripts.r1lite_audit_tcp_frames import static_alignment


CONFIG = REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
MAX_CONFIG_DELTA_RAD = 0.02
MAX_PATH_STEP_RAD = 0.01
MAX_NON_TARGET_CONFIG_RAD = 0.003
MAX_CONFIG_ERROR_RAD = 0.005
MAX_COMMAND_EXCURSION_RAD = 0.08
MAX_ITERATIONS = 12
PROBE_DELTA_RAD = 0.01
MAX_PROBE_NON_TARGET_RAD = 0.002
MAX_PROBE_TOTAL_RAD = 0.015
MAX_TCP_POSITION_RESIDUAL_M = 0.01
MAX_TCP_ROTATION_RESIDUAL_RAD = 0.05


def load_low_level() -> R1LiteRealLowLevel:
    config = yaml.safe_load(CONFIG.read_text())
    values = dict(config["env"]["cfg"]["low_level"])
    values.pop("_target_", None)
    values["r1lite_motion_validation_enabled"] = True
    values["r1lite_adaptive_joint_control_enabled"] = True
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {key: value for key, value in values.items() if key in signature.parameters}
    return R1LiteRealLowLevel(**kwargs)


def active_command(r1: R1LiteRealLowLevel) -> np.ndarray:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with r1._lock:
            command = (
                None
                if r1._last_joint_command is None
                else r1._last_joint_command.copy()
            )
        if command is not None:
            return command
        r1._spin_once_or_sleep()
    raise RuntimeError("MOCA active p_des was not observed")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def guarded_reach(
    r1: R1LiteRealLowLevel,
    target: np.ndarray,
    *,
    baseline: np.ndarray,
    active_joint: int | None,
    command_origin: np.ndarray,
    context: str,
    events_path: Path,
) -> np.ndarray:
    for iteration in range(1, MAX_ITERATIONS + 1):
        before = r1.get_arm_joints()
        error = target - before
        if float(np.max(np.abs(error))) <= MAX_CONFIG_ERROR_RAD:
            return before
        remaining = float(np.max(np.abs(error)))
        step_target = before + error * (
            min(MAX_PATH_STEP_RAD, remaining) / remaining
        )
        after = r1.move_joints_safe(
            step_target,
            max_step=MAX_PATH_STEP_RAD,
            duration_s=0.6,
        )
        command = active_command(r1)
        event = {
            "event": "guarded_reach_iteration",
            "context": context,
            "iteration": iteration,
            "before_feedback_rad": before.tolist(),
            "step_target_feedback_rad": step_target.tolist(),
            "after_feedback_rad": after.tolist(),
            "active_moca_p_des_rad": command.tolist(),
            "target_error_rad": float(np.max(np.abs(target - after))),
            "command_excursion_rad": float(
                np.max(np.abs(command - command_origin))
            ),
        }
        append_event(events_path, event)
        if event["command_excursion_rad"] > MAX_COMMAND_EXCURSION_RAD:
            raise RuntimeError(
                f"{context}: command excursion exceeded safety limit"
            )
        if float(np.max(np.abs(after - baseline))) > (
            MAX_CONFIG_DELTA_RAD + MAX_CONFIG_ERROR_RAD
        ):
            raise RuntimeError(
                f"{context}: total motion exceeded safety envelope"
            )
        if active_joint is not None:
            non_target = np.delete(np.abs(after - baseline), active_joint)
            if float(np.max(non_target)) > MAX_NON_TARGET_CONFIG_RAD:
                raise RuntimeError(
                    f"{context}: non-target joint motion exceeded limit"
                )
    raise TimeoutError(f"{context}: did not converge in {MAX_ITERATIONS} steps")


def tcp_residual_check(alignment: dict[str, Any]) -> dict[str, float]:
    residual = alignment["residuals"]["physical_as_tool_vs_model_tool"]
    result = {
        "position_m": float(residual["translation_norm_m"]),
        "rotation_rad": float(residual["rotation_error_rad"]),
    }
    if (
        result["position_m"] > MAX_TCP_POSITION_RESIDUAL_M
        or result["rotation_rad"] > MAX_TCP_ROTATION_RESIDUAL_RAD
    ):
        raise RuntimeError(f"TCP residual exceeded limit: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events_path = args.output.with_suffix(".jsonl")
    if args.output.exists() or events_path.exists():
        raise FileExistsError("Experiment outputs already exist; use a new run ID")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_unix": time.time(),
        "thresholds": {
            "max_config_delta_rad": MAX_CONFIG_DELTA_RAD,
            "max_path_step_rad": MAX_PATH_STEP_RAD,
            "max_non_target_config_rad": MAX_NON_TARGET_CONFIG_RAD,
            "max_config_error_rad": MAX_CONFIG_ERROR_RAD,
            "max_command_excursion_rad": MAX_COMMAND_EXCURSION_RAD,
            "probe_delta_rad": PROBE_DELTA_RAD,
            "max_probe_non_target_rad": MAX_PROBE_NON_TARGET_RAD,
            "max_probe_total_rad": MAX_PROBE_TOTAL_RAD,
            "max_tcp_position_residual_m": MAX_TCP_POSITION_RESIDUAL_M,
            "max_tcp_rotation_residual_rad": MAX_TCP_ROTATION_RESIDUAL_RAD,
        },
        "configurations": [],
    }
    atomic_json(args.output, result)
    r1 = load_low_level()
    try:
        topic = r1._arm_pub.topic_name
        discovery_deadline = time.monotonic() + 5.0
        subscribers = []
        while time.monotonic() < discovery_deadline:
            subscribers = r1._node.get_subscriptions_info_by_topic(topic)
            if any(info.node_name == "moca_adapter" for info in subscribers):
                break
            r1._spin_once_or_sleep()
        publishers = r1._node.get_publishers_info_by_topic(topic)
        publisher_names = sorted({info.node_name for info in publishers})
        subscriber_names = sorted({info.node_name for info in subscribers})
        forbidden = {
            "homogeneous_teleoperation",
            "relaxed_ik_right",
        }.intersection(publisher_names)
        if forbidden:
            raise RuntimeError(f"Competing arm publishers: {sorted(forbidden)}")
        if "moca_adapter" not in subscriber_names:
            raise RuntimeError("MOCA is not subscribed to the arm target")

        baseline = r1.get_arm_joints()
        command_baseline = active_command(r1)
        time.sleep(0.5)
        stable = r1.get_arm_joints()
        baseline_drift = float(np.max(np.abs(stable - baseline)))
        if baseline_drift > 0.003:
            raise RuntimeError(
                f"Preflight feedback drift {baseline_drift:.4f}rad"
            )
        baseline = stable
        command_baseline = active_command(r1)
        result.update(
            {
                "arm_target_topic": topic,
                "publisher_names": publisher_names,
                "subscriber_names": subscriber_names,
                "baseline_feedback_rad": baseline.tolist(),
                "baseline_moca_p_des_rad": command_baseline.tolist(),
                "preflight_drift_rad": baseline_drift,
            }
        )
        append_event(
            events_path,
            {
                "event": "preflight_passed",
                "baseline_feedback_rad": baseline.tolist(),
                "baseline_moca_p_des_rad": command_baseline.tolist(),
                "publisher_names": publisher_names,
                "subscriber_names": subscriber_names,
            },
        )
        atomic_json(args.output, result)

        specifications = [
            ("joint1_plus", 0, 0.02, 5),
            ("joint2_plus", 1, 0.02, 5),
            ("joint3_minus", 2, -0.02, 5),
            ("joint4_plus", 3, 0.02, 5),
            ("joint6_plus", 5, 0.02, 4),
        ]
        for name, joint_index, offset, probe_joint in specifications:
            target = baseline.copy()
            target[joint_index] += offset
            configuration: dict[str, Any] = {
                "name": name,
                "active_joint": joint_index,
                "requested_offset_rad": offset,
                "target_feedback_rad": target.tolist(),
                "status": "running",
            }
            result["configurations"].append(configuration)
            atomic_json(args.output, result)

            reached = guarded_reach(
                r1,
                target,
                baseline=baseline,
                active_joint=joint_index,
                command_origin=command_baseline,
                context=f"{name}_outbound",
                events_path=events_path,
            )
            configuration["reached_feedback_rad"] = reached.tolist()
            configuration["reached_error_rad"] = float(
                np.max(np.abs(reached - target))
            )
            configuration["moca_p_des_rad"] = active_command(r1).tolist()
            alignment = static_alignment(r1)
            configuration["tcp_residual"] = tcp_residual_check(alignment)
            configuration["alignment"] = alignment

            probe_before = r1.get_arm_joints()
            probe_command_before = active_command(r1)
            probe_target = probe_before.copy()
            probe_target[probe_joint] += PROBE_DELTA_RAD
            probe_after = r1.move_joints_safe(
                probe_target,
                max_step=PROBE_DELTA_RAD,
                duration_s=0.7,
            )
            probe_command_after = active_command(r1)
            probe_delta = probe_after - probe_before
            command_delta = probe_command_after - probe_command_before
            non_target = np.delete(np.abs(probe_delta), probe_joint)
            probe = {
                "joint_index": probe_joint,
                "before_feedback_rad": probe_before.tolist(),
                "target_feedback_rad": probe_target.tolist(),
                "after_feedback_rad": probe_after.tolist(),
                "feedback_delta_rad": probe_delta.tolist(),
                "command_delta_rad": command_delta.tolist(),
                "target_gain": float(
                    probe_delta[probe_joint] / PROBE_DELTA_RAD
                ),
                "max_non_target_motion_rad": float(np.max(non_target)),
                "max_total_motion_rad": float(
                    np.max(np.abs(probe_delta))
                ),
            }
            configuration["local_probe"] = probe
            append_event(
                events_path,
                {"event": "local_probe", "configuration": name, **probe},
            )
            if (
                probe_delta[probe_joint] <= 0.0
                or probe["max_non_target_motion_rad"]
                > MAX_PROBE_NON_TARGET_RAD
                or probe["max_total_motion_rad"] > MAX_PROBE_TOTAL_RAD
            ):
                raise RuntimeError(f"{name}: local probe violated limits")

            probe_return = guarded_reach(
                r1,
                target,
                baseline=baseline,
                active_joint=None,
                command_origin=command_baseline,
                context=f"{name}_probe_return",
                events_path=events_path,
            )
            configuration["probe_return_error_rad"] = float(
                np.max(np.abs(probe_return - target))
            )
            returned = guarded_reach(
                r1,
                baseline,
                baseline=baseline,
                active_joint=None,
                command_origin=command_baseline,
                context=f"{name}_baseline_return",
                events_path=events_path,
            )
            return_error = float(np.max(np.abs(returned - baseline)))
            configuration["baseline_return_feedback_rad"] = returned.tolist()
            configuration["baseline_return_error_rad"] = return_error
            if return_error > MAX_CONFIG_ERROR_RAD:
                raise RuntimeError(f"{name}: baseline return failed")
            configuration["status"] = "passed"
            atomic_json(args.output, result)

        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "safety_aborted"
        result["error"] = repr(exc)
        append_event(
            events_path,
            {"event": "safety_aborted", "error": repr(exc)},
        )
    finally:
        result["finished_at_unix"] = time.time()
        result["final_feedback_rad"] = r1.get_arm_joints().tolist()
        result["final_moca_p_des_rad"] = active_command(r1).tolist()
        atomic_json(args.output, result)
        print(args.output)
        print(json.dumps(result, indent=2))
        r1.close()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
