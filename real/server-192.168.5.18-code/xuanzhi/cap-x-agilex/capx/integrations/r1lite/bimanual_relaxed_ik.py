"""Isolated RelaxedIK adapter for the paired command mux."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.r1lite.bimanual_runtime import R1LiteBimanualRuntime
from capx.integrations.r1lite.kinematics import FULL_URDF, UrdfChain


class IsolatedBimanualRelaxedIK:
    """Request IK only through remapped, non-commanding RelaxedIK nodes."""

    def __init__(
        self,
        runtime: R1LiteBimanualRuntime,
        torso_joints_rad: Sequence[float],
    ) -> None:
        self.runtime = runtime
        torso = np.asarray(torso_joints_rad, dtype=float).reshape(3)
        chain = UrdfChain(FULL_URDF)
        base_from_torso = chain.fk(
            "base_link",
            "torso_link3",
            {
                f"torso_joint{index + 1}": float(value)
                for index, value in enumerate(torso)
            },
        )
        self.base_from_torso_rotation = base_from_torso[:3, :3]
        self.base_from_torso_translation = base_from_torso[:3, 3]

    def solve_candidates(
        self,
        *,
        side: str,
        target_position_base_m: Sequence[float],
        target_quaternion_xyzw: Sequence[float],
        reference_joints_rad: Sequence[float],
        probes: int,
    ) -> list[Sequence[float]]:
        if side not in ("left", "right"):
            raise ValueError(f"unsupported arm side {side!r}")
        if probes <= 0:
            raise ValueError("IK probe count must be positive")
        position_base = np.asarray(target_position_base_m, dtype=float).reshape(3)
        rotation_base = Rotation.from_quat(
            np.asarray(target_quaternion_xyzw, dtype=float).reshape(4)
        ).as_matrix()
        torso_from_base_rotation = self.base_from_torso_rotation.T
        position_local = torso_from_base_rotation @ (
            position_base - self.base_from_torso_translation
        )
        rotation_local = torso_from_base_rotation @ rotation_base
        local_orientation = Rotation.from_matrix(rotation_local)
        if side == "left":
            local_orientation = local_orientation * Rotation.from_euler(
                "y", -np.pi / 2.0
            )
        quaternion_local = local_orientation.as_quat()
        seed = np.asarray(reference_joints_rad, dtype=float).reshape(6)
        candidates = []
        for probe_index in range(probes):
            candidate = np.asarray(
                self.runtime.request_ik(
                    side=side,
                    position=position_local,
                    quaternion_xyzw=quaternion_local,
                    seed_joints=seed,
                    timeout_s=1.0,
                    reset=probe_index == 0,
                ),
                dtype=float,
            ).reshape(6)
            if not np.isfinite(candidate).all():
                raise RuntimeError(f"{side} RelaxedIK returned non-finite joints")
            candidates.append(candidate.tolist())
        return candidates
