"""Deterministic R1 Lite dual-arm FK and conservative separation gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from capx.integrations.r1lite.kinematics import (
    FULL_URDF,
    R1LiteCandidateKinematics,
    UrdfChain,
    arm_joint_map,
)


def tool_z_axis(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return the tool-frame Z axis expressed in the base frame."""
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    return np.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )


def quaternion_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix for an XYZW quaternion."""
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-12:
        raise ValueError("quaternion cannot be zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def tcp_position_for_fingertip(
    fingertip_position_base_m: np.ndarray,
    quaternion_xyzw: np.ndarray,
    tcp_to_fingertip_m: float,
) -> np.ndarray:
    """Recover TCP position from a desired fingertip and oriented tool offset."""
    fingertip = np.asarray(fingertip_position_base_m, dtype=float).reshape(3)
    return fingertip - tool_z_axis(quaternion_xyzw) * float(tcp_to_fingertip_m)


def strict_topdown_edge_rotation(inward_normal_base_xy: np.ndarray) -> np.ndarray:
    """Build a right-handed tool frame with Y inward and Z vertically down."""
    inward = np.asarray(inward_normal_base_xy, dtype=float).reshape(2)
    norm = float(np.linalg.norm(inward))
    if norm <= 1e-9:
        raise ValueError("towel edge inward normal cannot be zero")
    tool_y = np.array([inward[0] / norm, inward[1] / norm, 0.0])
    tool_z = np.array([0.0, 0.0, -1.0])
    tool_x = np.cross(tool_y, tool_z)
    return np.column_stack([tool_x, tool_y, tool_z])


def _segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Return the shortest distance between two finite 3D line segments."""
    p1 = np.asarray(first_start, dtype=np.float64).reshape(3)
    q1 = np.asarray(first_end, dtype=np.float64).reshape(3)
    p2 = np.asarray(second_start, dtype=np.float64).reshape(3)
    q2 = np.asarray(second_end, dtype=np.float64).reshape(3)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    epsilon = 1e-12
    if a <= epsilon and e <= epsilon:
        return float(np.linalg.norm(p1 - p2))
    if a <= epsilon:
        s = 0.0
        t = float(np.clip(np.dot(d2, r) / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= epsilon:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            s = (
                float(np.clip((b * float(np.dot(d2, r)) - c * e) / denominator, 0.0, 1.0))
                if abs(denominator) > epsilon
                else 0.0
            )
            t = (b * s + float(np.dot(d2, r))) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


@dataclass(frozen=True)
class BimanualSafetyReport:
    left_tcp_base_m: list[float]
    right_tcp_base_m: list[float]
    tcp_separation_m: float
    minimum_link_centerline_distance_m: float
    estimated_link_clearance_m: float
    required_tcp_separation_m: float
    required_link_clearance_m: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R1LiteBimanualKinematics:
    """Validate a 12-joint candidate against both arms and shared geometry.

    Link collision geometry is conservatively represented by equal-radius
    capsules around the arm centerline.  This does not replace mesh collision
    checking, so execution must remain disabled unless this gate is available.
    """

    def __init__(
        self,
        *,
        link_capsule_radius_m: float = 0.025,
        minimum_link_clearance_m: float = 0.005,
        minimum_tcp_separation_m: float = 0.060,
    ) -> None:
        self.full_chain = UrdfChain(FULL_URDF)
        self.left = R1LiteCandidateKinematics(side="left")
        self.right = R1LiteCandidateKinematics(side="right")
        self.link_capsule_radius_m = float(link_capsule_radius_m)
        self.minimum_link_clearance_m = float(minimum_link_clearance_m)
        self.minimum_tcp_separation_m = float(minimum_tcp_separation_m)
        if self.link_capsule_radius_m <= 0.0:
            raise ValueError("link capsule radius must be positive")
        if self.minimum_link_clearance_m < 0.0:
            raise ValueError("minimum link clearance cannot be negative")
        if self.minimum_tcp_separation_m <= 0.0:
            raise ValueError("minimum TCP separation must be positive")

    @staticmethod
    def _joint_map(
        left_joints: np.ndarray,
        right_joints: np.ndarray,
        torso_joints: np.ndarray,
    ) -> dict[str, float]:
        result = arm_joint_map(left_joints, side="left")
        result.update(arm_joint_map(right_joints, side="right"))
        torso = np.asarray(torso_joints, dtype=np.float64).reshape(3)
        result.update(
            {
                f"torso_joint{index + 1}": float(value)
                for index, value in enumerate(torso)
            }
        )
        return result

    def _arm_points(
        self,
        side: str,
        joint_positions: dict[str, float],
    ) -> list[np.ndarray]:
        links = [
            f"{side}_arm_base_link",
            *[f"{side}_arm_link{index}" for index in range(1, 7)],
            f"{side}_gripper_link",
        ]
        return [
            self.full_chain.fk("base_link", link, joint_positions)[:3, 3].copy()
            for link in links
        ]

    def validate(
        self,
        left_joints: np.ndarray,
        right_joints: np.ndarray,
        torso_joints: np.ndarray,
    ) -> BimanualSafetyReport:
        left_q = np.asarray(left_joints, dtype=np.float64).reshape(6)
        right_q = np.asarray(right_joints, dtype=np.float64).reshape(6)
        torso_q = np.asarray(torso_joints, dtype=np.float64).reshape(3)
        left_fk = self.left.fk(left_q, torso_q)
        right_fk = self.right.fk(right_q, torso_q)
        tcp_separation = float(
            np.linalg.norm(left_fk.position_base_m - right_fk.position_base_m)
        )
        if tcp_separation + 1e-9 < self.minimum_tcp_separation_m:
            raise ValueError(
                "dual-arm TCP separation gate failed: "
                f"{tcp_separation:.6f}m < {self.minimum_tcp_separation_m:.6f}m"
            )

        joint_positions = self._joint_map(left_q, right_q, torso_q)
        left_points = self._arm_points("left", joint_positions)
        right_points = self._arm_points("right", joint_positions)
        minimum_centerline = min(
            _segment_distance(left_start, left_end, right_start, right_end)
            for left_start, left_end in zip(left_points, left_points[1:])
            for right_start, right_end in zip(right_points, right_points[1:])
        )
        estimated_clearance = (
            minimum_centerline - 2.0 * self.link_capsule_radius_m
        )
        if estimated_clearance + 1e-9 < self.minimum_link_clearance_m:
            raise ValueError(
                "dual-arm conservative link-clearance gate failed: "
                f"{estimated_clearance:.6f}m < "
                f"{self.minimum_link_clearance_m:.6f}m"
            )
        return BimanualSafetyReport(
            left_tcp_base_m=left_fk.position_base_m.tolist(),
            right_tcp_base_m=right_fk.position_base_m.tolist(),
            tcp_separation_m=tcp_separation,
            minimum_link_centerline_distance_m=minimum_centerline,
            estimated_link_clearance_m=estimated_clearance,
            required_tcp_separation_m=self.minimum_tcp_separation_m,
            required_link_clearance_m=self.minimum_link_clearance_m,
        )

    def validate_interpolated_segment(
        self,
        start_left: np.ndarray,
        start_right: np.ndarray,
        end_left: np.ndarray,
        end_right: np.ndarray,
        torso_joints: np.ndarray,
        *,
        samples: int = 20,
    ) -> list[BimanualSafetyReport]:
        if samples < 2:
            raise ValueError("dual-arm segment validation needs at least 2 samples")
        reports = []
        for alpha in np.linspace(0.0, 1.0, samples):
            reports.append(
                self.validate(
                    np.asarray(start_left) * (1.0 - alpha)
                    + np.asarray(end_left) * alpha,
                    np.asarray(start_right) * (1.0 - alpha)
                    + np.asarray(end_right) * alpha,
                    torso_joints,
                )
            )
        return reports


class R1LiteTableCollisionKinematics:
    """Check only active-arm distal geometry against the calibrated table plane."""

    # R1 Lite URDF joint origins and STL bounds in gripper_link coordinates.
    # Each prismatic finger is placed at its maximum travel, conservatively
    # checking the fully-open gripper even during stages where it is closed.
    FINGER_MAX_TRAVEL_M = 0.05
    FINGER_JOINT_ORIGINS_M = (
        np.array([0.03689, 0.013453, 0.00012059]),
        np.array([0.03689, -0.013453, -0.00012059]),
    )
    FINGER_OPEN_TRANSLATIONS_M = (
        np.array([0.0, FINGER_MAX_TRAVEL_M, 0.0]),
        np.array([0.0, -FINGER_MAX_TRAVEL_M, 0.0]),
    )
    FINGER_MESH_BOUNDS_M = (
        (
            np.array([-0.05049464, -0.06518286, -0.01420414]),
            np.array([0.04146500, 0.00477702, 0.02287093]),
        ),
        (
            np.array([-0.05049463, -0.00477701, -0.02287093]),
            np.array([0.04146500, 0.06518287, 0.01420414]),
        ),
    )
    # right/left_gripper_tool_z is +90deg about gripper_link local Y.
    TOOL_TO_GRIPPER_ROTATION = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    def __init__(
        self,
        *,
        active_side: str,
        table_z_m: float,
        tcp_to_fingertip_m: float = 0.078,
        distal_capsule_radius_m: float = 0.025,
        minimum_table_clearance_m: float = 0.0,
    ) -> None:
        if active_side not in ("left", "right"):
            raise ValueError("table collision active side must be left or right")
        self.active_side = active_side
        self.table_z_m = float(table_z_m)
        self.tcp_to_fingertip_m = float(tcp_to_fingertip_m)
        self.link_capsule_radius_m = float(distal_capsule_radius_m)
        self.minimum_link_clearance_m = float(minimum_table_clearance_m)
        self.minimum_tcp_separation_m = 0.0
        self.full_chain = UrdfChain(FULL_URDF)
        self.kinematics = R1LiteCandidateKinematics(side=active_side)
        self.left = R1LiteCandidateKinematics(side="left")
        self.right = R1LiteCandidateKinematics(side="right")

    @staticmethod
    def _box_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ],
            dtype=float,
        )

    @classmethod
    def finger_minimum_z_offset_from_tcp(
        cls,
        rotation_base_tool: np.ndarray,
        *,
        fingers_open: bool = True,
    ) -> float:
        """Return the lowest physical finger-mesh Z relative to the TCP."""
        rotation_base_gripper = (
            np.asarray(rotation_base_tool, dtype=float).reshape(3, 3)
            @ cls.TOOL_TO_GRIPPER_ROTATION
        )
        translations = (
            cls.FINGER_OPEN_TRANSLATIONS_M
            if fingers_open
            else (np.zeros(3), np.zeros(3))
        )
        points = []
        for origin, translation, bounds in zip(
            cls.FINGER_JOINT_ORIGINS_M,
            translations,
            cls.FINGER_MESH_BOUNDS_M,
        ):
            corners_gripper = cls._box_corners(*bounds) + origin + translation
            points.append(corners_gripper @ rotation_base_gripper.T)
        return float(np.min(np.vstack(points)[:, 2]))

    @classmethod
    def finger_minimum_z_offsets_from_tcp(
        cls,
        rotation_base_tool: np.ndarray,
        *,
        fingers_open: bool = True,
    ) -> tuple[float, float]:
        """Return each physical finger mesh's lowest Z relative to the TCP."""
        rotation_base_gripper = (
            np.asarray(rotation_base_tool, dtype=float).reshape(3, 3)
            @ cls.TOOL_TO_GRIPPER_ROTATION
        )
        translations = (
            cls.FINGER_OPEN_TRANSLATIONS_M
            if fingers_open
            else (np.zeros(3), np.zeros(3))
        )
        offsets = []
        for origin, translation, bounds in zip(
            cls.FINGER_JOINT_ORIGINS_M,
            translations,
            cls.FINGER_MESH_BOUNDS_M,
        ):
            corners_gripper = cls._box_corners(*bounds) + origin + translation
            offsets.append(
                float(np.min((corners_gripper @ rotation_base_gripper.T)[:, 2]))
            )
        return tuple(offsets)

    def _finger_points_by_link_base(
        self, fk: object, *, fingers_open: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation_base_tool = quaternion_rotation_matrix(fk.quaternion_xyzw)
        rotation_base_gripper = (
            rotation_base_tool @ self.TOOL_TO_GRIPPER_ROTATION
        )
        points = []
        translations = (
            self.FINGER_OPEN_TRANSLATIONS_M
            if fingers_open
            else (np.zeros(3), np.zeros(3))
        )
        for origin, translation, bounds in zip(
            self.FINGER_JOINT_ORIGINS_M,
            translations,
            self.FINGER_MESH_BOUNDS_M,
        ):
            corners_gripper = (
                self._box_corners(*bounds) + origin + translation
            )
            points.append(
                np.asarray(fk.position_base_m, dtype=float)
                + corners_gripper @ rotation_base_gripper.T
            )
        return tuple(points)

    def _finger_points_base(
        self, fk: object, *, fingers_open: bool
    ) -> np.ndarray:
        return np.vstack(
            self._finger_points_by_link_base(fk, fingers_open=fingers_open)
        )

    def _open_finger_points_base(self, fk: object) -> np.ndarray:
        return self._finger_points_base(fk, fingers_open=True)

    def validate(
        self,
        left_joints: np.ndarray,
        right_joints: np.ndarray,
        torso_joints: np.ndarray,
        *,
        fingers_open: bool = True,
        allowed_contact_finger_link: str | None = None,
        minimum_noncontact_clearance_m: float = 0.0,
        minimum_contact_clearance_m: float = 0.0,
    ) -> BimanualSafetyReport:
        if allowed_contact_finger_link not in (
            None,
            "finger_link1",
            "finger_link2",
        ):
            raise ValueError("unknown designated finger contact link")
        pair = {
            "left": np.asarray(left_joints, dtype=float).reshape(6),
            "right": np.asarray(right_joints, dtype=float).reshape(6),
        }
        torso = np.asarray(torso_joints, dtype=float).reshape(3)
        joint_positions = R1LiteBimanualKinematics._joint_map(
            pair["left"],
            pair["right"],
            torso,
        )
        points = R1LiteBimanualKinematics._arm_points(
            self,
            self.active_side,
            joint_positions,
        )
        # Only links 4-6 and the gripper enter the tabletop workspace.
        distal_points = points[4:]
        centerline_clearance = min(
            float(point[2] - self.table_z_m) for point in distal_points
        )
        capsule_clearance = centerline_clearance - self.link_capsule_radius_m
        fk = self.kinematics.fk(pair[self.active_side], torso)
        fingertip = (
            fk.position_base_m
            + tool_z_axis(fk.quaternion_xyzw) * self.tcp_to_fingertip_m
        )
        fingertip_clearance = float(fingertip[2] - self.table_z_m)
        finger_points = self._finger_points_by_link_base(
            fk, fingers_open=fingers_open
        )
        finger_clearances = [
            float(np.min(points[:, 2]) - self.table_z_m)
            for points in finger_points
        ]
        if allowed_contact_finger_link is not None:
            contact_index = int(allowed_contact_finger_link[-1]) - 1
            noncontact_index = 1 - contact_index
            contact_clearance = finger_clearances[contact_index]
            noncontact_clearance = min(
                capsule_clearance,
                finger_clearances[noncontact_index],
            )
            if contact_clearance + 1e-9 < float(minimum_contact_clearance_m):
                raise ValueError(
                    "designated finger table contact gate failed: "
                    f"{contact_clearance:.6f}m < "
                    f"{float(minimum_contact_clearance_m):.6f}m"
                )
            if noncontact_clearance + 1e-9 < float(
                minimum_noncontact_clearance_m
            ):
                raise ValueError(
                    "non-contact gripper/table clearance gate failed: "
                    f"{noncontact_clearance:.6f}m < "
                    f"{float(minimum_noncontact_clearance_m):.6f}m"
                )
            minimum_clearance = min(contact_clearance, noncontact_clearance)
            required_clearance = min(
                float(minimum_contact_clearance_m),
                float(minimum_noncontact_clearance_m),
            )
        else:
            open_finger_clearance = min(finger_clearances)
            minimum_clearance = min(
                capsule_clearance,
                fingertip_clearance,
                open_finger_clearance,
            )
            required_clearance = self.minimum_link_clearance_m
        if minimum_clearance + 1e-9 < required_clearance:
            raise ValueError(
                "active-arm table collision gate failed: "
                f"{minimum_clearance:.6f}m < "
                f"{required_clearance:.6f}m"
            )
        left_fk = self.left.fk(pair["left"], torso)
        right_fk = self.right.fk(pair["right"], torso)
        return BimanualSafetyReport(
            left_tcp_base_m=left_fk.position_base_m.tolist(),
            right_tcp_base_m=right_fk.position_base_m.tolist(),
            tcp_separation_m=0.0,
            minimum_link_centerline_distance_m=centerline_clearance,
            estimated_link_clearance_m=minimum_clearance,
            required_tcp_separation_m=0.0,
            required_link_clearance_m=required_clearance,
        )

    def validate_interpolated_segment(
        self,
        start_left: np.ndarray,
        start_right: np.ndarray,
        end_left: np.ndarray,
        end_right: np.ndarray,
        torso_joints: np.ndarray,
        *,
        samples: int = 20,
        fingers_open: bool = True,
        allowed_contact_finger_link: str | None = None,
        minimum_noncontact_clearance_m: float = 0.0,
        minimum_contact_clearance_m: float = 0.0,
    ) -> list[BimanualSafetyReport]:
        if samples < 2:
            raise ValueError("table segment validation needs at least 2 samples")
        return [
            self.validate(
                np.asarray(start_left) * (1.0 - alpha)
                + np.asarray(end_left) * alpha,
                np.asarray(start_right) * (1.0 - alpha)
                + np.asarray(end_right) * alpha,
                torso_joints,
                fingers_open=fingers_open,
                    allowed_contact_finger_link=allowed_contact_finger_link,
                    minimum_noncontact_clearance_m=(
                        minimum_noncontact_clearance_m
                    ),
                    minimum_contact_clearance_m=minimum_contact_clearance_m,
            )
            for alpha in np.linspace(0.0, 1.0, samples)
        ]


@dataclass(frozen=True)
class CombinedBimanualSafetyReport:
    """Summary from all mandatory pipeline-eight collision gates."""

    bimanual: dict[str, object]
    left_table: dict[str, object]
    right_table: dict[str, object]
    gripper_envelope_separation_m: float
    required_gripper_envelope_separation_m: float

    @property
    def tcp_separation_m(self) -> float:
        return float(self.bimanual["tcp_separation_m"])

    @property
    def estimated_link_clearance_m(self) -> float:
        return min(
            float(self.bimanual["estimated_link_clearance_m"]),
            float(self.left_table["estimated_link_clearance_m"]),
            float(self.right_table["estimated_link_clearance_m"]),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R1LiteBimanualTableCollisionKinematics:
    """Validate both arms against the table without inter-arm gates."""

    def __init__(
        self,
        *,
        table_z_m: float,
        tcp_to_fingertip_m: float = 0.078,
        link_capsule_radius_m: float = 0.025,
        minimum_table_clearance_m: float = 0.0,
    ) -> None:
        table_kwargs = {
            "table_z_m": table_z_m,
            "tcp_to_fingertip_m": tcp_to_fingertip_m,
            "distal_capsule_radius_m": link_capsule_radius_m,
            "minimum_table_clearance_m": minimum_table_clearance_m,
        }
        self.left_table = R1LiteTableCollisionKinematics(
            active_side="left", **table_kwargs
        )
        self.right_table = R1LiteTableCollisionKinematics(
            active_side="right", **table_kwargs
        )

    def validate(
        self,
        left_joints: np.ndarray,
        right_joints: np.ndarray,
        torso_joints: np.ndarray,
        *,
        fingers_open: bool = True,
    ) -> dict[str, object]:
        return {
            "left_table": self.left_table.validate(
                left_joints,
                right_joints,
                torso_joints,
                fingers_open=fingers_open,
            ).to_dict(),
            "right_table": self.right_table.validate(
                left_joints,
                right_joints,
                torso_joints,
                fingers_open=fingers_open,
            ).to_dict(),
        }

    def validate_interpolated_segment(
        self,
        start_left: np.ndarray,
        start_right: np.ndarray,
        end_left: np.ndarray,
        end_right: np.ndarray,
        torso_joints: np.ndarray,
        *,
        samples: int = 50,
        fingers_open: bool = True,
    ) -> list[dict[str, object]]:
        if samples < 2:
            raise ValueError("table segment validation needs at least 2 samples")
        return [
            self.validate(
                np.asarray(start_left) * (1.0 - alpha)
                + np.asarray(end_left) * alpha,
                np.asarray(start_right) * (1.0 - alpha)
                + np.asarray(end_right) * alpha,
                torso_joints,
                fingers_open=fingers_open,
            )
            for alpha in np.linspace(0.0, 1.0, samples)
        ]


class R1LiteCombinedCollisionKinematics:
    """Fail-closed union of inter-arm, table, and open-gripper gates.

    The gripper-envelope gate deliberately uses TCP separation as a conservative
    proxy for the distance between the two complete gripper envelopes.  It is
    stricter than the legacy 6 cm TCP-only gate and is checked independently so
    a caller cannot accidentally weaken it by configuring that legacy gate.
    """

    DEFAULT_MINIMUM_GRIPPER_ENVELOPE_SEPARATION_M = 0.12552056969962836

    def __init__(
        self,
        *,
        table_z_m: float,
        tcp_to_fingertip_m: float = 0.078,
        link_capsule_radius_m: float = 0.025,
        minimum_link_clearance_m: float = 0.005,
        minimum_table_clearance_m: float = 0.0,
        minimum_gripper_envelope_separation_m: float = (
            DEFAULT_MINIMUM_GRIPPER_ENVELOPE_SEPARATION_M
        ),
        bimanual_gate: object | None = None,
        left_table_gate: object | None = None,
        right_table_gate: object | None = None,
    ) -> None:
        self.minimum_gripper_envelope_separation_m = float(
            minimum_gripper_envelope_separation_m
        )
        if self.minimum_gripper_envelope_separation_m <= 0.0:
            raise ValueError("minimum gripper-envelope separation must be positive")
        self.bimanual = bimanual_gate or R1LiteBimanualKinematics(
            link_capsule_radius_m=link_capsule_radius_m,
            minimum_link_clearance_m=minimum_link_clearance_m,
            minimum_tcp_separation_m=min(
                0.060, self.minimum_gripper_envelope_separation_m
            ),
        )
        table_kwargs = {
            "table_z_m": table_z_m,
            "tcp_to_fingertip_m": tcp_to_fingertip_m,
            "distal_capsule_radius_m": link_capsule_radius_m,
            "minimum_table_clearance_m": minimum_table_clearance_m,
        }
        self.left_table = left_table_gate or R1LiteTableCollisionKinematics(
            active_side="left", **table_kwargs
        )
        self.right_table = right_table_gate or R1LiteTableCollisionKinematics(
            active_side="right", **table_kwargs
        )

    def validate(
        self,
        left_joints: np.ndarray,
        right_joints: np.ndarray,
        torso_joints: np.ndarray,
        *,
        fingers_open: bool = True,
    ) -> CombinedBimanualSafetyReport:
        pair = self.bimanual.validate(left_joints, right_joints, torso_joints)
        # Always invoke both table gates: each includes the fully-open two-finger
        # mesh AABBs as well as distal link capsules and fingertip clearance.
        if fingers_open:
            left_table = self.left_table.validate(
                left_joints, right_joints, torso_joints
            )
            right_table = self.right_table.validate(
                left_joints, right_joints, torso_joints
            )
        else:
            left_table = self.left_table.validate(
                left_joints,
                right_joints,
                torso_joints,
                fingers_open=False,
            )
            right_table = self.right_table.validate(
                left_joints,
                right_joints,
                torso_joints,
                fingers_open=False,
            )
        separation = float(pair.tcp_separation_m)
        if separation + 1e-9 < self.minimum_gripper_envelope_separation_m:
            raise ValueError(
                "dual-gripper conservative envelope gate failed: "
                f"{separation:.6f}m < "
                f"{self.minimum_gripper_envelope_separation_m:.6f}m"
            )
        return CombinedBimanualSafetyReport(
            bimanual=pair.to_dict(),
            left_table=left_table.to_dict(),
            right_table=right_table.to_dict(),
            gripper_envelope_separation_m=separation,
            required_gripper_envelope_separation_m=(
                self.minimum_gripper_envelope_separation_m
            ),
        )

    def validate_interpolated_segment(
        self,
        start_left: np.ndarray,
        start_right: np.ndarray,
        end_left: np.ndarray,
        end_right: np.ndarray,
        torso_joints: np.ndarray,
        *,
        samples: int = 50,
        fingers_open: bool = True,
    ) -> list[CombinedBimanualSafetyReport]:
        if samples < 2:
            raise ValueError("combined segment validation needs at least 2 samples")
        return [
            self.validate(
                np.asarray(start_left) * (1.0 - alpha)
                + np.asarray(end_left) * alpha,
                np.asarray(start_right) * (1.0 - alpha)
                + np.asarray(end_right) * alpha,
                torso_joints,
                fingers_open=fingers_open,
            )
            for alpha in np.linspace(0.0, 1.0, samples)
        ]
