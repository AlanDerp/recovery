"""Immutable, fail-closed execution bundles for the seventh R1 Lite pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "capx.r1lite.seventh_execution_bundle.v1"
SEQUENTIAL_SCHEMA = "capx.r1lite.seventh_execution_bundle.v2"
EIGHTH_SCHEMA = "capx.r1lite.eighth_execution_bundle.v14"
EIGHTH_STAGE_ORDER = (
    "grasp",
    "lift",
    "pullback",
    "translate_center",
    "release_retract",
    "home",
)
STAGE_ORDER = ("pregrasp", "grasp", "lift", "fold_center")
_TOP_LEVEL_KEYS = {
    "schema",
    "artifact_bindings",
    "motion_limits",
    "initial_state",
    "stages",
    "sweeps",
    "bundle_sha256",
}
_SEQUENTIAL_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS | {"execution_mode", "stage_order"}
_BINDING_KEYS = {
    "plan_sha256",
    "config_sha256",
    "full_urdf_sha256",
    "left_relaxed_urdf_sha256",
    "right_relaxed_urdf_sha256",
    "tcp_definition_sha256",
    "collision_model_sha256",
}
_MOTION_LIMIT_KEYS = {
    "maximum_stage_joint_delta_rad",
    "fk_position_tolerance_m",
    "fk_orientation_tolerance_rad",
}
_INITIAL_KEYS = {"left_joints_rad", "right_joints_rad", "torso_joints_rad"}
_STAGE_KEYS = {"id", "left", "right"}
_SIDE_KEYS = {
    "joints_rad",
    "joint_waypoints_rad",
    "target_position_base_m",
    "target_quaternion_xyzw",
    "fk_position_base_m",
    "fk_quaternion_xyzw",
    "fk_position_error_m",
    "fk_orientation_error_rad",
    "maximum_joint_delta_rad",
}
_SWEEP_KEYS = {
    "samples_per_segment",
    "segments",
    "minimum_swept_link_clearance_m",
    "minimum_swept_tcp_separation_m",
}
_SEGMENT_KEYS = {
    "to_stage",
    "samples",
    "minimum_estimated_link_clearance_m",
    "minimum_tcp_separation_m",
    "passed",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes, rejecting non-JSON and non-finite values."""
    _validate_json_value(value, path="$")
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported JSON type {type(value).__name__}")


def _exact_keys(value: Any, expected: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{field} fields mismatch: missing={missing}, extra={extra}")
    return value


def _number(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _vector(value: Any, size: int, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must contain exactly {size} numbers")
    return [_number(item, field=f"{field}[{index}]") for index, item in enumerate(value)]


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _without_bundle_hash(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key != "bundle_sha256"}


def _as_float_list(value: Sequence[float], *, field: str, size: int) -> list[float]:
    result = [float(item) for item in value]
    _vector(result, size, field=field)
    return result


def build_execution_bundle(
    *,
    plan_sha256: str,
    config_sha256: str,
    full_urdf_sha256: str,
    left_relaxed_urdf_sha256: str,
    right_relaxed_urdf_sha256: str,
    tcp_definition: Mapping[str, Any],
    collision_model: Mapping[str, Any],
    initial_left_joints_rad: Sequence[float],
    initial_right_joints_rad: Sequence[float],
    torso_joints_rad: Sequence[float],
    stages: Sequence[Mapping[str, Any]],
    sweeps: Mapping[str, Any],
    maximum_stage_joint_delta_rad: float,
    fk_position_tolerance_m: float,
    fk_orientation_tolerance_rad: float,
    execution_mode: str = "dual_arm_synchronous",
    stage_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build and self-hash a validated execution bundle.

    Inputs are copied into plain JSON values so caller mutations cannot alter the
    returned artifact accidentally without invalidating ``bundle_sha256``.
    """
    tcp_value = deepcopy(dict(tcp_definition))
    collision_value = deepcopy(dict(collision_model))
    sequential = execution_mode == "sequential_single_arm"
    if execution_mode not in ("dual_arm_synchronous", "sequential_single_arm"):
        raise ValueError("unknown execution bundle mode")
    bundle: dict[str, Any] = {
        "schema": SEQUENTIAL_SCHEMA if sequential else SCHEMA,
        "artifact_bindings": {
            "plan_sha256": plan_sha256,
            "config_sha256": config_sha256,
            "full_urdf_sha256": full_urdf_sha256,
            "left_relaxed_urdf_sha256": left_relaxed_urdf_sha256,
            "right_relaxed_urdf_sha256": right_relaxed_urdf_sha256,
            "tcp_definition_sha256": canonical_sha256(tcp_value),
            "collision_model_sha256": canonical_sha256(collision_value),
        },
        "motion_limits": {
            "maximum_stage_joint_delta_rad": float(maximum_stage_joint_delta_rad),
            "fk_position_tolerance_m": float(fk_position_tolerance_m),
            "fk_orientation_tolerance_rad": float(fk_orientation_tolerance_rad),
        },
        "initial_state": {
            "left_joints_rad": _as_float_list(
                initial_left_joints_rad, field="initial_left_joints_rad", size=6
            ),
            "right_joints_rad": _as_float_list(
                initial_right_joints_rad, field="initial_right_joints_rad", size=6
            ),
            "torso_joints_rad": _as_float_list(
                torso_joints_rad, field="torso_joints_rad", size=3
            ),
        },
        "stages": deepcopy(list(stages)),
        "sweeps": deepcopy(dict(sweeps)),
    }
    if sequential:
        declared_order = list(stage_order or [])
        if not declared_order:
            raise ValueError("sequential execution bundle requires stage order")
        bundle["execution_mode"] = execution_mode
        bundle["stage_order"] = declared_order
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return validate_execution_bundle(
        bundle,
        tcp_definition=tcp_value,
        collision_model=collision_value,
    )


def validate_execution_bundle(
    bundle: Mapping[str, Any],
    *,
    plan_sha256: str | None = None,
    config_sha256: str | None = None,
    full_urdf_sha256: str | None = None,
    left_relaxed_urdf_sha256: str | None = None,
    right_relaxed_urdf_sha256: str | None = None,
    tcp_definition: Mapping[str, Any] | None = None,
    collision_model: Mapping[str, Any] | None = None,
    initial_left_joints_rad: Sequence[float] | None = None,
    initial_right_joints_rad: Sequence[float] | None = None,
    torso_joints_rad: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Strictly validate schema, self-hash, cross-links and optional bindings."""
    if not isinstance(bundle, Mapping):
        raise ValueError("execution bundle must be an object")
    value = deepcopy(dict(bundle))
    _validate_json_value(value, path="$")
    schema = value.get("schema")
    if schema == SEQUENTIAL_SCHEMA:
        _exact_keys(
            value,
            _SEQUENTIAL_TOP_LEVEL_KEYS,
            field="sequential execution bundle",
        )
        if value["execution_mode"] != "sequential_single_arm":
            raise ValueError("unexpected sequential execution mode")
        declared_stage_order = value["stage_order"]
        if (
            not isinstance(declared_stage_order, list)
            or not declared_stage_order
            or any(
                not isinstance(item, str) or not item
                for item in declared_stage_order
            )
        ):
            raise ValueError("sequential stage order must contain stage IDs")
    else:
        _exact_keys(value, _TOP_LEVEL_KEYS, field="execution bundle")
        declared_stage_order = list(STAGE_ORDER)
    if schema not in (SCHEMA, SEQUENTIAL_SCHEMA):
        raise ValueError("unexpected execution bundle schema")

    bindings = _exact_keys(
        value["artifact_bindings"], _BINDING_KEYS, field="artifact_bindings"
    )
    for key in _BINDING_KEYS:
        _sha256(bindings[key], field=f"artifact_bindings.{key}")

    limits = _exact_keys(
        value["motion_limits"], _MOTION_LIMIT_KEYS, field="motion_limits"
    )
    maximum_delta = _number(
        limits["maximum_stage_joint_delta_rad"],
        field="motion_limits.maximum_stage_joint_delta_rad",
        minimum=0.0,
    )
    position_tolerance = _number(
        limits["fk_position_tolerance_m"],
        field="motion_limits.fk_position_tolerance_m",
        minimum=0.0,
    )
    orientation_tolerance = _number(
        limits["fk_orientation_tolerance_rad"],
        field="motion_limits.fk_orientation_tolerance_rad",
        minimum=0.0,
    )

    initial = _exact_keys(value["initial_state"], _INITIAL_KEYS, field="initial_state")
    _vector(initial["left_joints_rad"], 6, field="initial_state.left_joints_rad")
    _vector(initial["right_joints_rad"], 6, field="initial_state.right_joints_rad")
    _vector(initial["torso_joints_rad"], 3, field="initial_state.torso_joints_rad")

    stages = value["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")
    stage_ids: list[str] = []
    previous = {
        "left": initial["left_joints_rad"],
        "right": initial["right_joints_rad"],
    }
    for index, stage_value in enumerate(stages):
        stage = _exact_keys(stage_value, _STAGE_KEYS, field=f"stages[{index}]")
        stage_id = stage["id"]
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError(f"stages[{index}].id must be a non-empty string")
        if stage_id in stage_ids:
            raise ValueError(f"duplicate execution stage id {stage_id!r}")
        stage_ids.append(stage_id)
        for side in ("left", "right"):
            side_value = _exact_keys(
                stage[side], _SIDE_KEYS, field=f"stages[{index}].{side}"
            )
            joints = _vector(
                side_value["joints_rad"], 6, field=f"stages[{index}].{side}.joints_rad"
            )
            waypoint_values = side_value["joint_waypoints_rad"]
            if not isinstance(waypoint_values, list) or not waypoint_values:
                raise ValueError(
                    f"stages[{index}].{side}.joint_waypoints_rad must be non-empty"
                )
            waypoints = [
                _vector(
                    waypoint,
                    6,
                    field=(
                        f"stages[{index}].{side}.joint_waypoints_rad"
                        f"[{waypoint_index}]"
                    ),
                )
                for waypoint_index, waypoint in enumerate(waypoint_values)
            ]
            if waypoints[-1] != joints:
                raise ValueError(
                    f"stages[{index}].{side} final waypoint does not match joints"
                )
            target_position = _vector(
                side_value["target_position_base_m"],
                3,
                field=f"stages[{index}].{side}.target_position_base_m",
            )
            target_quaternion = _vector(
                side_value["target_quaternion_xyzw"],
                4,
                field=f"stages[{index}].{side}.target_quaternion_xyzw",
            )
            target_quaternion_norm = math.sqrt(
                sum(item * item for item in target_quaternion)
            )
            if abs(target_quaternion_norm - 1.0) > 1e-5:
                raise ValueError(
                    f"stages[{index}].{side} target quaternion is not unit"
                )
            fk_position = _vector(
                side_value["fk_position_base_m"],
                3,
                field=f"stages[{index}].{side}.fk_position_base_m",
            )
            quaternion = _vector(
                side_value["fk_quaternion_xyzw"],
                4,
                field=f"stages[{index}].{side}.fk_quaternion_xyzw",
            )
            quaternion_norm = math.sqrt(sum(item * item for item in quaternion))
            if abs(quaternion_norm - 1.0) > 1e-5:
                raise ValueError(f"stages[{index}].{side} FK quaternion is not unit")
            position_error = _number(
                side_value["fk_position_error_m"],
                field=f"stages[{index}].{side}.fk_position_error_m",
                minimum=0.0,
            )
            orientation_error = _number(
                side_value["fk_orientation_error_rad"],
                field=f"stages[{index}].{side}.fk_orientation_error_rad",
                minimum=0.0,
            )
            computed_position_error = math.sqrt(
                sum(
                    (actual - target) ** 2
                    for actual, target in zip(fk_position, target_position)
                )
            )
            quaternion_dot = abs(
                sum(
                    actual * target
                    for actual, target in zip(quaternion, target_quaternion)
                )
            )
            computed_orientation_error = 2.0 * math.acos(
                min(1.0, max(0.0, quaternion_dot))
            )
            if abs(position_error - computed_position_error) > 1e-9:
                raise ValueError(
                    f"stages[{index}].{side} FK position error mismatch"
                )
            if abs(orientation_error - computed_orientation_error) > 1e-9:
                raise ValueError(
                    f"stages[{index}].{side} FK orientation error mismatch"
                )
            if position_error > position_tolerance + 1e-12:
                raise ValueError(f"stages[{index}].{side} exceeds FK position limit")
            if orientation_error > orientation_tolerance + 1e-12:
                raise ValueError(
                    f"stages[{index}].{side} exceeds FK orientation limit"
                )
            recorded_delta = _number(
                side_value["maximum_joint_delta_rad"],
                field=f"stages[{index}].{side}.maximum_joint_delta_rad",
                minimum=0.0,
            )
            actual_delta = 0.0
            prior_waypoint = previous[side]
            for waypoint in waypoints:
                actual_delta = max(
                    actual_delta,
                    max(
                        abs(current - prior)
                        for current, prior in zip(waypoint, prior_waypoint)
                    ),
                )
                prior_waypoint = waypoint
            if abs(recorded_delta - actual_delta) > 1e-9:
                raise ValueError(
                    f"stages[{index}].{side} maximum joint delta mismatch"
                )
            if actual_delta > maximum_delta + 1e-9:
                raise ValueError(f"stages[{index}].{side} exceeds motion limit")
            previous[side] = joints
    if stage_ids != list(declared_stage_order):
        raise ValueError("seventh-pipeline execution stage order is immutable")

    sweeps = _exact_keys(value["sweeps"], _SWEEP_KEYS, field="sweeps")
    samples_per_segment = sweeps["samples_per_segment"]
    if (
        isinstance(samples_per_segment, bool)
        or not isinstance(samples_per_segment, int)
        or samples_per_segment < 2
    ):
        raise ValueError("sweeps.samples_per_segment must be an integer of at least 2")
    segments = sweeps["segments"]
    if not isinstance(segments, list) or len(segments) != len(stages):
        raise ValueError("sweep segments must bind one-to-one with execution stages")
    link_clearances: list[float] = []
    tcp_separations: list[float] = []
    for index, segment_value in enumerate(segments):
        segment = _exact_keys(
            segment_value, _SEGMENT_KEYS, field=f"sweeps.segments[{index}]"
        )
        if segment["to_stage"] != stage_ids[index]:
            raise ValueError("sweep segment stage binding mismatch")
        if segment["samples"] != samples_per_segment:
            raise ValueError("sweep segment sample count mismatch")
        if segment["passed"] is not True:
            raise ValueError("all execution bundle sweep segments must have passed")
        link_clearances.append(
            _number(
                segment["minimum_estimated_link_clearance_m"],
                field=(
                    f"sweeps.segments[{index}]."
                    "minimum_estimated_link_clearance_m"
                ),
            )
        )
        tcp_separations.append(
            _number(
                segment["minimum_tcp_separation_m"],
                field=f"sweeps.segments[{index}].minimum_tcp_separation_m",
                minimum=0.0,
            )
        )
    recorded_link_minimum = _number(
        sweeps["minimum_swept_link_clearance_m"],
        field="sweeps.minimum_swept_link_clearance_m",
    )
    recorded_tcp_minimum = _number(
        sweeps["minimum_swept_tcp_separation_m"],
        field="sweeps.minimum_swept_tcp_separation_m",
        minimum=0.0,
    )
    if abs(recorded_link_minimum - min(link_clearances)) > 1e-12:
        raise ValueError("minimum swept link clearance summary mismatch")
    if abs(recorded_tcp_minimum - min(tcp_separations)) > 1e-12:
        raise ValueError("minimum swept TCP separation summary mismatch")

    expected_bundle_hash = canonical_sha256(_without_bundle_hash(value))
    _sha256(value["bundle_sha256"], field="bundle_sha256")
    if value["bundle_sha256"] != expected_bundle_hash:
        raise ValueError("execution bundle canonical SHA-256 mismatch")

    expected_bindings = {
        "plan_sha256": plan_sha256,
        "config_sha256": config_sha256,
        "full_urdf_sha256": full_urdf_sha256,
        "left_relaxed_urdf_sha256": left_relaxed_urdf_sha256,
        "right_relaxed_urdf_sha256": right_relaxed_urdf_sha256,
        "tcp_definition_sha256": (
            None if tcp_definition is None else canonical_sha256(dict(tcp_definition))
        ),
        "collision_model_sha256": (
            None if collision_model is None else canonical_sha256(dict(collision_model))
        ),
    }
    for key, expected in expected_bindings.items():
        if expected is not None and bindings[key] != expected:
            raise ValueError(f"execution bundle {key} binding mismatch")

    expected_state = {
        "left_joints_rad": initial_left_joints_rad,
        "right_joints_rad": initial_right_joints_rad,
        "torso_joints_rad": torso_joints_rad,
    }
    for key, expected in expected_state.items():
        if expected is not None and initial[key] != [float(item) for item in expected]:
            raise ValueError(f"execution bundle {key} binding mismatch")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_execution_bundle(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_file_sha256: str | None = None,
    **bindings: Any,
) -> dict[str, Any]:
    """Load strict JSON and validate the complete immutable bundle."""
    raw = Path(path).read_bytes()
    if (
        expected_sha256 is not None
        and expected_file_sha256 is not None
        and expected_sha256 != expected_file_sha256
    ):
        raise ValueError("conflicting expected execution bundle file hashes")
    expected_file_hash = expected_sha256 or expected_file_sha256
    if expected_file_hash is not None:
        _sha256(expected_file_hash, field="expected_sha256")
        if sha256_bytes(raw) != expected_file_hash:
            raise ValueError("execution bundle file SHA-256 mismatch")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("execution bundle is not strict JSON") from exc
    return validate_execution_bundle(value, **bindings)


canonical_hash = canonical_sha256
build = build_execution_bundle
load = load_execution_bundle
validate = validate_execution_bundle


_EIGHTH_TOP_LEVEL_KEYS = {
    "schema",
    "artifact_bindings",
    "motion_limits",
    "initial_state",
    "stage_order",
    "stages",
    "bundle_sha256",
}
_EIGHTH_LIMIT_KEYS = {
    "maximum_shared_cartesian_step_m",
    "maximum_waypoint_joint_delta_rad",
    "fk_position_tolerance_m",
    "fk_orientation_tolerance_rad",
    "relative_geometry_tolerance_m",
    "sweep_samples_per_segment",
}
_EIGHTH_STAGE_KEYS = {"id", "waypoints"}
_EIGHTH_WAYPOINT_KEYS = {
    "alpha",
    "expected_midpoint_base_m",
    "expected_right_minus_left_m",
    "ik_acceptance",
    "left",
    "right",
    "collision_summary",
}
_EIGHTH_SIDE_KEYS = {
    "target_position_base_m",
    "target_quaternion_xyzw",
    "joints_rad",
    "fk_position_base_m",
    "fk_quaternion_xyzw",
    "fk_position_error_m",
    "fk_orientation_error_rad",
    "maximum_joint_delta_rad",
}
_EIGHTH_COLLISION_KEYS = {
    "mode",
    "checked",
    "sweep_samples",
}


def build_eighth_execution_bundle(
    *,
    plan_sha256: str,
    config_sha256: str,
    full_urdf_sha256: str,
    left_relaxed_urdf_sha256: str,
    right_relaxed_urdf_sha256: str,
    tcp_definition: Mapping[str, Any],
    collision_model: Mapping[str, Any],
    initial_left_joints_rad: Sequence[float],
    initial_right_joints_rad: Sequence[float],
    torso_joints_rad: Sequence[float],
    stages: Sequence[Mapping[str, Any]],
    maximum_shared_cartesian_step_m: float = 0.05,
    maximum_waypoint_joint_delta_rad: float = 1.0,
    fk_position_tolerance_m: float = 0.030,
    fk_orientation_tolerance_rad: float = 0.25,
    relative_geometry_tolerance_m: float = 0.005,
    sweep_samples_per_segment: int = 50,
) -> dict[str, Any]:
    """Build the immutable, self-hashed pipeline-eight offline certificate."""
    tcp_value = deepcopy(dict(tcp_definition))
    collision_value = deepcopy(dict(collision_model))
    bundle: dict[str, Any] = {
        "schema": EIGHTH_SCHEMA,
        "artifact_bindings": {
            "plan_sha256": plan_sha256,
            "config_sha256": config_sha256,
            "full_urdf_sha256": full_urdf_sha256,
            "left_relaxed_urdf_sha256": left_relaxed_urdf_sha256,
            "right_relaxed_urdf_sha256": right_relaxed_urdf_sha256,
            "tcp_definition_sha256": canonical_sha256(tcp_value),
            "collision_model_sha256": canonical_sha256(collision_value),
        },
        "motion_limits": {
            "maximum_shared_cartesian_step_m": float(
                maximum_shared_cartesian_step_m
            ),
            "maximum_waypoint_joint_delta_rad": float(
                maximum_waypoint_joint_delta_rad
            ),
            "fk_position_tolerance_m": float(fk_position_tolerance_m),
            "fk_orientation_tolerance_rad": float(fk_orientation_tolerance_rad),
            "relative_geometry_tolerance_m": float(relative_geometry_tolerance_m),
            "sweep_samples_per_segment": int(sweep_samples_per_segment),
        },
        "initial_state": {
            "left_joints_rad": _as_float_list(
                initial_left_joints_rad, field="initial_left_joints_rad", size=6
            ),
            "right_joints_rad": _as_float_list(
                initial_right_joints_rad, field="initial_right_joints_rad", size=6
            ),
            "torso_joints_rad": _as_float_list(
                torso_joints_rad, field="torso_joints_rad", size=3
            ),
        },
        "stage_order": list(EIGHTH_STAGE_ORDER),
        "stages": deepcopy(list(stages)),
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return validate_eighth_execution_bundle(
        bundle,
        tcp_definition=tcp_value,
        collision_model=collision_value,
    )


def validate_eighth_execution_bundle(
    bundle: Mapping[str, Any],
    *,
    plan_sha256: str | None = None,
    config_sha256: str | None = None,
    full_urdf_sha256: str | None = None,
    left_relaxed_urdf_sha256: str | None = None,
    right_relaxed_urdf_sha256: str | None = None,
    tcp_definition: Mapping[str, Any] | None = None,
    collision_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate every waypoint, binding, and the canonical self-hash."""
    value = deepcopy(dict(bundle))
    _validate_json_value(value, path="$")
    _exact_keys(value, _EIGHTH_TOP_LEVEL_KEYS, field="eighth execution bundle")
    if value["schema"] != EIGHTH_SCHEMA:
        raise ValueError("unexpected eighth execution bundle schema")
    if value["stage_order"] != list(EIGHTH_STAGE_ORDER):
        raise ValueError("eighth execution stage order is immutable")
    bindings = _exact_keys(
        value["artifact_bindings"], _BINDING_KEYS, field="artifact_bindings"
    )
    for key in _BINDING_KEYS:
        _sha256(bindings[key], field=f"artifact_bindings.{key}")
    limits = _exact_keys(
        value["motion_limits"], _EIGHTH_LIMIT_KEYS, field="motion_limits"
    )
    cart_step = _number(
        limits["maximum_shared_cartesian_step_m"],
        field="motion_limits.maximum_shared_cartesian_step_m",
        minimum=0.0,
    )
    if cart_step > 0.05 + 1e-12:
        raise ValueError("eighth Cartesian step may not exceed 0.05m")
    joint_limit = _number(
        limits["maximum_waypoint_joint_delta_rad"],
        field="motion_limits.maximum_waypoint_joint_delta_rad",
        minimum=0.0,
    )
    position_tol = _number(
        limits["fk_position_tolerance_m"],
        field="motion_limits.fk_position_tolerance_m",
        minimum=0.0,
    )
    orientation_tol = _number(
        limits["fk_orientation_tolerance_rad"],
        field="motion_limits.fk_orientation_tolerance_rad",
        minimum=0.0,
    )
    relative_tol = _number(
        limits["relative_geometry_tolerance_m"],
        field="motion_limits.relative_geometry_tolerance_m",
        minimum=0.0,
    )
    sweep_samples = limits["sweep_samples_per_segment"]
    if isinstance(sweep_samples, bool) or sweep_samples != 50:
        raise ValueError("eighth grasp table sweeps must use 50 samples")
    initial = _exact_keys(value["initial_state"], _INITIAL_KEYS, field="initial_state")
    prior_joints = {
        "left": _vector(initial["left_joints_rad"], 6, field="initial left joints"),
        "right": _vector(initial["right_joints_rad"], 6, field="initial right joints"),
    }
    _vector(initial["torso_joints_rad"], 3, field="initial torso joints")
    stages = value["stages"]
    if not isinstance(stages, list) or [
        stage.get("id") if isinstance(stage, dict) else None for stage in stages
    ] != list(EIGHTH_STAGE_ORDER):
        raise ValueError("eighth execution stages do not match fixed stage order")
    prior_targets: dict[str, list[float] | None] = {"left": None, "right": None}
    reference_relative: list[float] | None = None
    for stage_index, stage_value in enumerate(stages):
        stage = _exact_keys(
            stage_value, _EIGHTH_STAGE_KEYS, field=f"stages[{stage_index}]"
        )
        waypoints = stage["waypoints"]
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError(f"stages[{stage_index}].waypoints must be non-empty")
        previous_alpha = 0.0
        for waypoint_index, waypoint_value in enumerate(waypoints):
            field = f"stages[{stage_index}].waypoints[{waypoint_index}]"
            waypoint = _exact_keys(
                waypoint_value, _EIGHTH_WAYPOINT_KEYS, field=field
            )
            ik_acceptance = waypoint["ik_acceptance"]
            if ik_acceptance not in {
                "required",
                "joint_transition",
                "best_effort_seed",
            }:
                raise ValueError(f"{field}.ik_acceptance is invalid")
            if ik_acceptance == "best_effort_seed" and not (
                stage["id"] == "grasp"
                and waypoint_index == len(waypoints) - 1
            ):
                raise ValueError(
                    f"{field} best-effort IK is allowed only at final grasp seed"
                )
            if ik_acceptance == "joint_transition" and not (
                (
                    stage["id"] == "grasp"
                    and waypoint_index < len(waypoints) - 1
                )
                or stage["id"] == "home"
            ):
                raise ValueError(
                    f"{field} joint transition is allowed only before final grasp seed"
                )
            alpha = _number(waypoint["alpha"], field=f"{field}.alpha")
            if not previous_alpha < alpha <= 1.0:
                raise ValueError(f"{field}.alpha must increase in (0, 1]")
            previous_alpha = alpha
            sides: dict[str, dict[str, Any]] = {}
            for side in ("left", "right"):
                side_value = _exact_keys(
                    waypoint[side], _EIGHTH_SIDE_KEYS, field=f"{field}.{side}"
                )
                target = _vector(
                    side_value["target_position_base_m"], 3, field=f"{field}.{side}.target"
                )
                target_quaternion = _vector(
                    side_value["target_quaternion_xyzw"],
                    4,
                    field=f"{field}.{side}.target_quaternion",
                )
                fk_position = _vector(
                    side_value["fk_position_base_m"], 3, field=f"{field}.{side}.fk"
                )
                fk_quaternion = _vector(
                    side_value["fk_quaternion_xyzw"],
                    4,
                    field=f"{field}.{side}.fk_quaternion",
                )
                if abs(math.sqrt(sum(x * x for x in target_quaternion)) - 1.0) > 1e-5:
                    raise ValueError(f"{field}.{side} target quaternion is not unit")
                if abs(math.sqrt(sum(x * x for x in fk_quaternion)) - 1.0) > 1e-5:
                    raise ValueError(f"{field}.{side} FK quaternion is not unit")
                position_error = math.dist(target, fk_position)
                orientation_error = 2.0 * math.acos(
                    min(1.0, abs(sum(a * b for a, b in zip(
                        target_quaternion, fk_quaternion
                    ))))
                )
                if abs(position_error - _number(
                    side_value["fk_position_error_m"],
                    field=f"{field}.{side}.fk_position_error_m",
                )) > 1e-9:
                    raise ValueError(f"{field}.{side} FK position error mismatch")
                if abs(orientation_error - _number(
                    side_value["fk_orientation_error_rad"],
                    field=f"{field}.{side}.fk_orientation_error_rad",
                )) > 1e-9:
                    raise ValueError(f"{field}.{side} FK orientation error mismatch")
                if (
                    ik_acceptance == "required"
                    and (
                        position_error > position_tol + 1e-12
                        or orientation_error > orientation_tol + 1e-12
                    )
                ):
                    raise ValueError(f"{field}.{side} exceeds FK tolerance")
                joints = _vector(
                    side_value["joints_rad"], 6, field=f"{field}.{side}.joints"
                )
                actual_delta = max(
                    abs(current - previous)
                    for current, previous in zip(joints, prior_joints[side])
                )
                recorded_delta = _number(
                    side_value["maximum_joint_delta_rad"],
                    field=f"{field}.{side}.maximum_joint_delta_rad",
                )
                if abs(actual_delta - recorded_delta) > 1e-9:
                    raise ValueError(f"{field}.{side} joint delta mismatch")
                if actual_delta > joint_limit + 1e-9:
                    raise ValueError(f"{field}.{side} exceeds joint delta limit")
                if (
                    ik_acceptance in {"joint_transition", "best_effort_seed"}
                    and actual_delta > 1.0 + 1e-9
                ):
                    raise ValueError(
                        f"{field}.{side} grasp seed substep exceeds 1.0rad"
                    )
                if (
                    stage["id"] != "grasp"
                    and ik_acceptance
                    != "joint_transition"
                    and prior_targets[side] is not None
                    and math.dist(
                    target, prior_targets[side]
                    )
                    > cart_step + 1e-9
                ):
                    raise ValueError(f"{field}.{side} Cartesian step exceeds limit")
                prior_joints[side], prior_targets[side] = joints, target
                sides[side] = side_value
            midpoint = [
                (left + right) / 2.0
                for left, right in zip(
                    sides["left"]["target_position_base_m"],
                    sides["right"]["target_position_base_m"],
                )
            ]
            relative = [
                right - left
                for left, right in zip(
                    sides["left"]["target_position_base_m"],
                    sides["right"]["target_position_base_m"],
                )
            ]
            if _vector(
                waypoint["expected_midpoint_base_m"], 3, field=f"{field}.midpoint"
            ) != midpoint:
                raise ValueError(f"{field} midpoint mismatch")
            recorded_relative = _vector(
                waypoint["expected_right_minus_left_m"], 3, field=f"{field}.relative"
            )
            if recorded_relative != relative:
                raise ValueError(f"{field} right-minus-left mismatch")
            if (
                stage["id"] == "grasp"
                and waypoint_index == len(waypoints) - 1
                and reference_relative is None
            ):
                reference_relative = relative
            if reference_relative is not None and stage["id"] in {
                "grasp", "lift", "translate_center"
            } and math.dist(relative, reference_relative) > relative_tol + 1e-12:
                raise ValueError(f"{field} relative geometry exceeds tolerance")
            collision = _exact_keys(
                waypoint["collision_summary"],
                _EIGHTH_COLLISION_KEYS,
                field=f"{field}.collision_summary",
            )
            expected_collision = {
                "mode": "disabled_non_grasp_stage",
                "checked": False,
                "sweep_samples": 0,
            }
            if collision != expected_collision:
                raise ValueError(
                    f"{field} collision summary violates grasp-only table policy"
                )
        if abs(previous_alpha - 1.0) > 1e-12:
            raise ValueError(f"stages[{stage_index}] must end at alpha 1")
    for side in ("left", "right"):
        if prior_joints[side] != initial[f"{side}_joints_rad"]:
            raise ValueError("eighth home must restore initial joints")
    expected_hash = canonical_sha256(_without_bundle_hash(value))
    if value["bundle_sha256"] != expected_hash:
        raise ValueError("eighth execution bundle canonical SHA-256 mismatch")
    expected_bindings = {
        "plan_sha256": plan_sha256,
        "config_sha256": config_sha256,
        "full_urdf_sha256": full_urdf_sha256,
        "left_relaxed_urdf_sha256": left_relaxed_urdf_sha256,
        "right_relaxed_urdf_sha256": right_relaxed_urdf_sha256,
        "tcp_definition_sha256": None if tcp_definition is None else canonical_sha256(dict(tcp_definition)),
        "collision_model_sha256": None if collision_model is None else canonical_sha256(dict(collision_model)),
    }
    for key, expected in expected_bindings.items():
        if expected is not None and bindings[key] != expected:
            raise ValueError(f"eighth execution bundle {key} binding mismatch")
    return value


def load_eighth_execution_bundle(
    path: Path, *, expected_file_sha256: str | None = None, **bindings: Any
) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if expected_file_sha256 is not None and sha256_bytes(raw) != expected_file_sha256:
        raise ValueError("eighth execution bundle file SHA-256 mismatch")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("eighth execution bundle is not strict JSON") from exc
    return validate_eighth_execution_bundle(value, **bindings)
