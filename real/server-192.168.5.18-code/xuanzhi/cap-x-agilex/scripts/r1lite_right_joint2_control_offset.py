#!/usr/bin/env python3
"""Opt-in right-arm joint-2 command compensation.

This module only adjusts application-layer targets sent to MOCA. It never
modifies HDAS parameters, robot calibration files, URDFs, or encoder feedback.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import contextmanager

import numpy as np


DEFAULT_RIGHT_JOINT2_CONTROL_OFFSET_DEG = 9.0


def ensure_hdas_calibration_disabled(offset_rad: float) -> None:
    """Reject application compensation if HDAS calibration is also enabled."""
    if abs(float(offset_rad)) <= 1e-12:
        return
    result = subprocess.run(
        ["ros2", "param", "get", "/HDAS", "calib_enable"],
        capture_output=True,
        check=False,
        text=True,
        timeout=3.0,
    )
    if result.returncode == 0 and "enable" in result.stdout.lower():
        raise RuntimeError(
            "HDAS global joint calibration is enabled; refusing an "
            "additional right joint-2 control offset"
        )


class RightJoint2ControlOffsetMixin:
    """Add an explicit session-only offset to right-arm joint 2 targets."""

    def __init__(self, right_joint2_control_offset_rad: float, **kwargs) -> None:
        self.right_joint2_control_offset_rad = float(
            right_joint2_control_offset_rad
        )
        self.right_joint2_control_offset_enabled = True
        ensure_hdas_calibration_disabled(
            self.right_joint2_control_offset_rad
        )
        super().__init__(**kwargs)

    def _offset_right_target(self, target) -> np.ndarray:
        adjusted = np.asarray(target, dtype=float).copy()
        if adjusted.shape != (6,) or not np.isfinite(adjusted).all():
            raise ValueError("right control target must contain six finite joints")
        if self.right_joint2_control_offset_enabled:
            adjusted[1] += self.right_joint2_control_offset_rad
        return adjusted

    @contextmanager
    def suspend_right_joint2_control_offset(self):
        previous = self.right_joint2_control_offset_enabled
        self.right_joint2_control_offset_enabled = False
        try:
            yield
        finally:
            self.right_joint2_control_offset_enabled = previous

    def execute_joint_motion(self, left_target, right_target, **kwargs):
        return super().execute_joint_motion(
            left_target,
            self._offset_right_target(right_target),
            **kwargs,
        )

    def publish_joint_targets(self, left_target, right_target) -> None:
        super().publish_joint_targets(
            left_target,
            self._offset_right_target(right_target),
        )

    def publish_joint_targets_for(
        self,
        left_target,
        right_target,
        *,
        duration_s: float,
        poll_s: float = 0.02,
    ) -> None:
        super().publish_joint_targets_for(
            left_target,
            self._offset_right_target(right_target),
            duration_s=duration_s,
            poll_s=poll_s,
        )


class HistoricalMappingRightJoint2ControlOffsetMixin(
    RightJoint2ControlOffsetMixin
):
    """Keep historical lookup logical, then offset the raw MOCA command."""

    def _feedback_targets_to_commands(self, targets):
        logical_targets = [
            np.asarray(targets[0], dtype=float),
            np.asarray(targets[1], dtype=float).copy(),
        ]
        logical_targets[1][1] -= self.right_joint2_control_offset_rad
        left_command, right_command = super(
            RightJoint2ControlOffsetMixin, self
        )._feedback_targets_to_commands(logical_targets)
        adjusted_right = np.asarray(right_command, dtype=float)
        adjusted_right[1] += self.right_joint2_control_offset_rad
        if self.historical_mapping_trace:
            trace = self.historical_mapping_trace[-1]
            trace["logical_target_rad"] = trace["target_rad"]
            trace["target_rad"] = np.asarray(targets[1], dtype=float).tolist()
            trace["raw_command_rad"] = adjusted_right.tolist()
            trace["right_joint2_control_offset_rad"] = (
                self.right_joint2_control_offset_rad
            )
        return left_command, tuple(float(value) for value in adjusted_right)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the standalone right joint-2 control offset."
    )
    parser.add_argument(
        "--offset-deg",
        type=float,
        default=DEFAULT_RIGHT_JOINT2_CONTROL_OFFSET_DEG,
    )
    args = parser.parse_args()
    offset_rad = float(np.deg2rad(args.offset_deg))
    ensure_hdas_calibration_disabled(offset_rad)
    print(
        json.dumps(
            {
                "scope": "application_moca_command_layer_only",
                "right_joint2_control_offset_deg": args.offset_deg,
                "right_joint2_control_offset_rad": offset_rad,
                "persistent_machine_files_modified": False,
                "hdas_global_calibration_required": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
