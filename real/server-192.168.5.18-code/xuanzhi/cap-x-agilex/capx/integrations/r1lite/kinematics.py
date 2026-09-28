"""Deterministic URDF forward kinematics for R1 Lite IK candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


RELAXED_URDF = Path(
    "/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/"
    "r1_lite_floating_right.urdf"
)
LEFT_RELAXED_URDF = Path(
    "/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/"
    "r1_lite_floating_left.urdf"
)
FULL_URDF = Path(
    "/home/nvidia/galaxea/install/r1lite_urdf/share/r1lite_urdf/urdf/"
    "r1lite_2026.urdf"
)
GRIPPER_LINK_TO_TOOL_Z_XYZ_M = np.zeros(3, dtype=np.float64)
GRIPPER_LINK_TO_TOOL_Z_RPY_RAD = np.array(
    [0.0, np.pi / 2.0, 0.0],
    dtype=np.float64,
)

ProbeT = TypeVar("ProbeT")
ResultT = TypeVar("ResultT")


def _xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def _rigid(
    translation: np.ndarray,
    rotation: Rotation | None = None,
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    if rotation is not None:
        result[:3, :3] = rotation.as_matrix()
    return result


class UrdfChain:
    """Minimal serial-chain URDF FK evaluator."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        root = ET.parse(self.path).getroot()
        self._by_child: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                self._by_child[child.attrib["link"]] = joint

    def _joints_from_root(
        self,
        root_link: str,
        target_link: str,
    ) -> list[ET.Element]:
        reverse: list[ET.Element] = []
        current = target_link
        while current != root_link:
            if current not in self._by_child:
                raise ValueError(
                    f"No URDF chain from {root_link} to {target_link} in {self.path}"
                )
            joint = self._by_child[current]
            reverse.append(joint)
            parent = joint.find("parent")
            assert parent is not None
            current = parent.attrib["link"]
        return list(reversed(reverse))

    def fk(
        self,
        root_link: str,
        target_link: str,
        joint_positions: dict[str, float],
    ) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        for joint in self._joints_from_root(root_link, target_link):
            origin = joint.find("origin")
            origin_xyz = _xyz(None if origin is None else origin.attrib.get("xyz"))
            origin_rpy = _xyz(None if origin is None else origin.attrib.get("rpy"))
            result = result @ _rigid(
                origin_xyz,
                Rotation.from_euler("xyz", origin_rpy),
            )
            joint_type = joint.attrib.get("type")
            if joint_type in {"revolute", "continuous"}:
                axis_node = joint.find("axis")
                axis = _xyz(
                    "1 0 0"
                    if axis_node is None
                    else axis_node.attrib.get("xyz", "1 0 0")
                )
                axis /= np.linalg.norm(axis)
                angle = float(joint_positions.get(joint.attrib["name"], 0.0))
                result = result @ _rigid(
                    np.zeros(3),
                    Rotation.from_rotvec(axis * angle),
                )
            elif joint_type == "prismatic":
                axis_node = joint.find("axis")
                axis = _xyz(
                    "1 0 0"
                    if axis_node is None
                    else axis_node.attrib.get("xyz", "1 0 0")
                )
                position = float(joint_positions.get(joint.attrib["name"], 0.0))
                result = result @ _rigid(axis * position)
        return result

    def joint_limits(self, joint_names: list[str]) -> dict[str, tuple[float, float]]:
        """Return finite lower/upper URDF limits for the requested joints."""
        root = ET.parse(self.path).getroot()
        requested = set(joint_names)
        limits: dict[str, tuple[float, float]] = {}
        for joint in root.findall("joint"):
            name = joint.attrib.get("name", "")
            if name not in requested:
                continue
            limit = joint.find("limit")
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                raise ValueError(f"Joint {name!r} has no finite URDF limits")
            limits[name] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
        missing = requested - limits.keys()
        if missing:
            raise ValueError(f"URDF is missing joint limits for {sorted(missing)}")
        return limits


def arm_joint_map(
    arm_joints: np.ndarray,
    *,
    side: str = "right",
) -> dict[str, float]:
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported R1 Lite arm side: {side!r}")
    arm = np.asarray(arm_joints, dtype=np.float64).reshape(6)
    return {
        f"{side}_arm_joint{index + 1}": float(value)
        for index, value in enumerate(arm)
    }


def full_joint_map(
    arm_joints: np.ndarray,
    torso_joints: np.ndarray,
    *,
    side: str = "right",
) -> dict[str, float]:
    result = arm_joint_map(arm_joints, side=side)
    torso = np.asarray(torso_joints, dtype=np.float64).reshape(3)
    result.update(
        {
            f"torso_joint{index + 1}": float(value)
            for index, value in enumerate(torso)
        }
    )
    return result


def _rotation_error_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(Rotation.from_matrix(relative).magnitude())


@dataclass(frozen=True)
class CandidateFk:
    """Candidate-specific FK evaluated independently from ROS FK topics."""

    position_base_m: np.ndarray
    quaternion_xyzw: np.ndarray
    relaxed_matrix: np.ndarray
    full_matrix: np.ndarray
    dual_position_error_m: float
    dual_orientation_error_rad: float


class R1LiteCandidateKinematics:
    """Evaluate one candidate with both RelaxedIK and full robot URDFs."""

    def __init__(
        self,
        *,
        side: str = "right",
        relaxed_urdf: Path | None = None,
        full_urdf: Path = FULL_URDF,
        enforce_joint_limits: bool = False,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported R1 Lite arm side: {side!r}")
        self.side = side
        self.enforce_joint_limits = bool(enforce_joint_limits)
        if relaxed_urdf is None:
            relaxed_urdf = (
                LEFT_RELAXED_URDF if side == "left" else RELAXED_URDF
            )
        self.relaxed_chain = UrdfChain(relaxed_urdf)
        self.full_chain = UrdfChain(full_urdf)
        self.relaxed_target_link = (
            "left_gripper_link"
            if side == "left"
            else "right_gripper_tool_z"
        )
        self.full_target_link = (
            "left_gripper_link"
            if side == "left"
            else "right_gripper_tool_z"
        )
        # The vendor left URDF omits left_gripper_tool_z.  Construct it with
        # the same transform as the right arm: zero translation and +90deg
        # about gripper-link Y.  Therefore tool +Z is gripper physical forward.
        self.tool_z_from_urdf_target = (
            _rigid(
                GRIPPER_LINK_TO_TOOL_Z_XYZ_M,
                Rotation.from_euler("xyz", GRIPPER_LINK_TO_TOOL_Z_RPY_RAD),
            )
            if side == "left"
            else np.eye(4, dtype=np.float64)
        )
        joint_names = [f"{side}_arm_joint{index}" for index in range(1, 7)]
        self.arm_limits = self.full_chain.joint_limits(joint_names)

    def validate_joint_limits(self, arm_joints: np.ndarray) -> None:
        """Fail closed when a candidate violates a full-URDF arm limit."""
        values = arm_joint_map(arm_joints, side=self.side)
        violations = []
        for name, value in values.items():
            lower, upper = self.arm_limits[name]
            if not np.isfinite(value) or value < lower or value > upper:
                violations.append(
                    {
                        "joint": name,
                        "value_rad": value,
                        "lower_rad": lower,
                        "upper_rad": upper,
                    }
                )
        if violations:
            raise ValueError(f"R1 Lite joint limit violation: {violations}")

    def fk(
        self,
        arm_joints: np.ndarray,
        torso_joints: np.ndarray,
    ) -> CandidateFk:
        arm = np.asarray(arm_joints, dtype=np.float64).reshape(6)
        torso = np.asarray(torso_joints, dtype=np.float64).reshape(3)
        if self.enforce_joint_limits:
            self.validate_joint_limits(arm)
        torso_map = {
            f"torso_joint{index + 1}": float(value)
            for index, value in enumerate(torso)
        }
        base_from_torso = self.full_chain.fk(
            "base_link",
            "torso_link3",
            torso_map,
        )
        relaxed_matrix = (
            base_from_torso
            @ self.relaxed_chain.fk(
                "torso_link3",
                self.relaxed_target_link,
                arm_joint_map(arm, side=self.side),
            )
            @ self.tool_z_from_urdf_target
        )
        full_matrix = (
            self.full_chain.fk(
                "base_link",
                self.full_target_link,
                full_joint_map(arm, torso, side=self.side),
            )
            @ self.tool_z_from_urdf_target
        )
        return CandidateFk(
            position_base_m=full_matrix[:3, 3].copy(),
            quaternion_xyzw=Rotation.from_matrix(
                full_matrix[:3, :3]
            ).as_quat(),
            relaxed_matrix=relaxed_matrix,
            full_matrix=full_matrix,
            dual_position_error_m=float(
                np.linalg.norm(
                    relaxed_matrix[:3, 3] - full_matrix[:3, 3]
                )
            ),
            dual_orientation_error_rad=_rotation_error_rad(
                relaxed_matrix,
                full_matrix,
            ),
        )


def rank_fk_qualified_candidates(
    candidates: list[dict],
    *,
    fk_tolerance_m: float,
    dual_position_tolerance_m: float = 1e-6,
    dual_orientation_tolerance_rad: float = 1e-6,
) -> tuple[list[dict], list[dict]]:
    """Filter by local FK, then rank candidates by joint continuity."""
    qualified: list[dict] = []
    rejected: list[dict] = []
    for candidate in candidates:
        reasons = []
        if float(candidate["fk_error_m"]) > float(fk_tolerance_m):
            reasons.append("fk_position_error")
        if (
            float(candidate["dual_urdf_position_error_m"])
            > float(dual_position_tolerance_m)
        ):
            reasons.append("dual_urdf_position_disagreement")
        if (
            float(candidate["dual_urdf_orientation_error_rad"])
            > float(dual_orientation_tolerance_rad)
        ):
            reasons.append("dual_urdf_orientation_disagreement")
        if reasons:
            rejected.append(
                {
                    "yaw_deg": candidate.get("yaw_deg"),
                    "reasons": reasons,
                    "fk_error_m": candidate.get("fk_error_m"),
                    "max_joint_delta_from_start_rad": candidate.get(
                        "max_joint_delta_from_start_rad"
                    ),
                }
            )
        else:
            qualified.append(candidate)
    qualified.sort(
        key=lambda item: (
            float(item["max_joint_delta_from_start_rad"]),
            float(item["fk_error_m"]),
        )
    )
    return qualified, rejected


def candidate_joint_delta_limit(
    frame_index: int,
    *,
    early_limit_rad: float = 2.5,
    f6_limit_rad: float = 1.0,
) -> float | None:
    """Return the configured continuity hard limit for a pipeline frame."""
    if int(frame_index) in {2, 3}:
        return float(early_limit_rad)
    if int(frame_index) == 6:
        return float(f6_limit_rad)
    return None


def max_joint_delta_rad(
    candidate_joints: np.ndarray,
    start_joints: np.ndarray,
) -> float:
    """Return the largest absolute single-joint branch displacement."""
    candidate = np.asarray(candidate_joints, dtype=np.float64).reshape(6)
    start = np.asarray(start_joints, dtype=np.float64).reshape(6)
    return float(np.max(np.abs(candidate - start)))


def solve_probes_with_reference_seed(
    probes: list[ProbeT],
    reference_joints: np.ndarray,
    *,
    seed_solver: Callable[[np.ndarray], None],
    solve_probe: Callable[[ProbeT], ResultT],
) -> tuple[list[ResultT], list[dict]]:
    """Solve independent probes after restoring the same seed each time."""
    reference = np.asarray(reference_joints, dtype=np.float64).reshape(6).copy()
    results: list[ResultT] = []
    errors: list[dict] = []
    for probe in probes:
        try:
            seed_solver(reference.copy())
            results.append(solve_probe(probe))
        except Exception as exc:
            errors.append({"probe": probe, "error": repr(exc)})
    return results, errors
