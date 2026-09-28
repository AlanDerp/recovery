#!/usr/bin/env bash
set -euo pipefail

cd /home/nvidia/xuanzhi/cap-x-agilex

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

PLAN_PATH="${1:-/home/nvidia/xuanzhi/cap-x-agilex/outputs/r1lite_tennis_ball_in_box/offline_ik_pick_place_joint_poses_3d.json}"
FRAME_A0_PIPELINE="${2:-${FRAME_A0_PIPELINE:-nominal_joints}}"
R1LITE_CONFIG_PATH="${R1LITE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
R1LITE_ROUND_COUNT="${R1LITE_ROUND_COUNT:-2}"
R1LITE_SINGLE_ROUND_POLICY="${R1LITE_SINGLE_ROUND_POLICY:-default}"
if [[ "${R1LITE_ROUND_COUNT}" != "1" && "${R1LITE_ROUND_COUNT}" != "2" ]]; then
  echo "[error] R1LITE_ROUND_COUNT must be 1 or 2" >&2
  exit 1
fi
if [[ "${R1LITE_SINGLE_ROUND_POLICY}" != "default" && "${R1LITE_SINGLE_ROUND_POLICY}" != "llm_absolute_20" ]]; then
  echo "[error] R1LITE_SINGLE_ROUND_POLICY must be default or llm_absolute_20" >&2
  exit 1
fi
if [[ "${FRAME_A0_PIPELINE}" == "cartesian_delta" ]]; then
  FRAME_A0_PIPELINE="cartesian_absolute"
fi
if [[ "${FRAME_A0_PIPELINE}" == "cartesian_absolute" ]]; then
  POS_TOL_M="${POS_TOL_M:-0.045}"
  MAX_CORRECTIONS="${MAX_CORRECTIONS:-20}"
else
  POS_TOL_M="${POS_TOL_M:-0.045}"
  MAX_CORRECTIONS="${MAX_CORRECTIONS:-15}"
fi
CORRECTION_GAIN="${CORRECTION_GAIN:-1.0}"
CORRECTION_STEP_CAP_M="${CORRECTION_STEP_CAP_M:-}"
MAX_COMPENSATION_NORM_M="${MAX_COMPENSATION_NORM_M:-0.20}"
MIN_MOTION_RAD="${MIN_MOTION_RAD:-0.002}"
COMMAND_DURATION_S="${COMMAND_DURATION_S:-2.5}"
JOINT_ARRIVAL_TOL_RAD="${JOINT_ARRIVAL_TOL_RAD:-0.01}"
JOINT_ARRIVAL_MAX_RETRIES="${JOINT_ARRIVAL_MAX_RETRIES:-3}"
LLM_MAX_CARTESIAN_FRAME_STEP_M="${LLM_MAX_CARTESIAN_FRAME_STEP_M:-0.10}"
LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD="${LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD:-1.0}"
LLM_GENERATED_POLICY_PATH="${LLM_GENERATED_POLICY_PATH:-}"
SKIP_XYZ_WAYPOINT_STEP_RAD="${SKIP_XYZ_WAYPOINT_STEP_RAD:-0.02}"
RETURN_HOME_WAYPOINT_STEP_RAD="${RETURN_HOME_WAYPOINT_STEP_RAD:-0.03}"
BOX_ABOVE_XY_TOL_M="${BOX_ABOVE_XY_TOL_M:-0.12}"
BOX_ABOVE_Z_MARGIN_FROM_CENTER_M="${BOX_ABOVE_Z_MARGIN_FROM_CENTER_M:-0.05}"
F6_BOX_XY_TOL_M="${F6_BOX_XY_TOL_M:-0.13}"
F6_POS_TOL_M="${F6_POS_TOL_M:-0.035}"
CARTESIAN_IK_FK_TOL_M="${CARTESIAN_IK_FK_TOL_M:-0.08}"
CARTESIAN_EARLY_MAX_JOINT_DELTA_RAD="${CARTESIAN_EARLY_MAX_JOINT_DELTA_RAD:-2.5}"
CARTESIAN_F6_MAX_JOINT_DELTA_RAD="${CARTESIAN_F6_MAX_JOINT_DELTA_RAD:-1.0}"
CARTESIAN_TOPDOWN_YAW_STEP_DEG="${CARTESIAN_TOPDOWN_YAW_STEP_DEG:-45}"
CARTESIAN_TOPDOWN_MAX_TILT_DEG="${CARTESIAN_TOPDOWN_MAX_TILT_DEG:-60}"
CARTESIAN_TOPDOWN_TILT_STEP_DEG="${CARTESIAN_TOPDOWN_TILT_STEP_DEG:-15}"
export JOINT_ARRIVAL_TOL_RAD JOINT_ARRIVAL_MAX_RETRIES
export LLM_MAX_CARTESIAN_FRAME_STEP_M LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD
export LLM_GENERATED_POLICY_PATH
export CARTESIAN_TOPDOWN_MAX_TILT_DEG CARTESIAN_TOPDOWN_TILT_STEP_DEG

ARM_TOPIC="${ARM_TOPIC:-/capx/motion_target/target_joint_state_arm_right}"
GRIPPER_TOPIC="${GRIPPER_TOPIC:-/capx/motion_target/target_position_gripper_right}"

ensure_moca() {
  local info
  info="$(ros2 topic info -v "${ARM_TOPIC}" 2>/dev/null || true)"
  if grep -q "Node name: moca_adapter" <<<"${info}"; then
    echo "[ok] MOCA subscribes to ${ARM_TOPIC}"
    return
  fi

  echo "[error] MOCA is not subscribed to ${ARM_TOPIC}" >&2
  echo "Start isolated MOCA first, or run scripts/r1lite_return_home.sh once to initialize it." >&2
  exit 1
}

reset_mit_mode() {
  echo "[mit] reset right arm into MIT mode"
  for command in 2 5 1; do
    local response
    response="$(timeout 8 ros2 service call /function_frame_right_arm hdas_msg/srv/FunctionFrame "{command: ${command}}")"
    echo "${response}"
    if ! grep -q "success=True" <<<"${response}"; then
      echo "[error] MIT mode command ${command} failed" >&2
      exit 1
    fi
  done
}

echo "[plan] ${PLAN_PATH}"
echo "[params] POS_TOL_M=${POS_TOL_M} MAX_CORRECTIONS=${MAX_CORRECTIONS} CORRECTION_GAIN=${CORRECTION_GAIN} CORRECTION_STEP_CAP_M=${CORRECTION_STEP_CAP_M:-none} MAX_COMPENSATION_NORM_M=${MAX_COMPENSATION_NORM_M} MIN_MOTION_RAD=${MIN_MOTION_RAD} COMMAND_DURATION_S=${COMMAND_DURATION_S} JOINT_ARRIVAL_TOL_RAD=${JOINT_ARRIVAL_TOL_RAD} JOINT_ARRIVAL_MAX_RETRIES=${JOINT_ARRIVAL_MAX_RETRIES} LLM_MAX_CARTESIAN_FRAME_STEP_M=${LLM_MAX_CARTESIAN_FRAME_STEP_M} LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD=${LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD} SKIP_XYZ_WAYPOINT_STEP_RAD=${SKIP_XYZ_WAYPOINT_STEP_RAD} RETURN_HOME_WAYPOINT_STEP_RAD=${RETURN_HOME_WAYPOINT_STEP_RAD} BOX_ABOVE_XY_TOL_M=${BOX_ABOVE_XY_TOL_M} BOX_ABOVE_Z_MARGIN_FROM_CENTER_M=${BOX_ABOVE_Z_MARGIN_FROM_CENTER_M} F6_BOX_XY_TOL_M=${F6_BOX_XY_TOL_M} F6_POS_TOL_M=${F6_POS_TOL_M} FRAME_A0_PIPELINE=${FRAME_A0_PIPELINE} CARTESIAN_IK_FK_TOL_M=${CARTESIAN_IK_FK_TOL_M} CARTESIAN_EARLY_MAX_JOINT_DELTA_RAD=${CARTESIAN_EARLY_MAX_JOINT_DELTA_RAD} CARTESIAN_F6_MAX_JOINT_DELTA_RAD=${CARTESIAN_F6_MAX_JOINT_DELTA_RAD} CARTESIAN_TOPDOWN_YAW_STEP_DEG=${CARTESIAN_TOPDOWN_YAW_STEP_DEG} CARTESIAN_TOPDOWN_MAX_TILT_DEG=${CARTESIAN_TOPDOWN_MAX_TILT_DEG} CARTESIAN_TOPDOWN_TILT_STEP_DEG=${CARTESIAN_TOPDOWN_TILT_STEP_DEG} R1LITE_ROUND_COUNT=${R1LITE_ROUND_COUNT} R1LITE_SINGLE_ROUND_POLICY=${R1LITE_SINGLE_ROUND_POLICY} R1LITE_CONFIG_PATH=${R1LITE_CONFIG_PATH}"

ensure_moca
reset_mit_mode

/home/nvidia/xuanzhi/cap-x-agilex/.capx_r1lite_venv/bin/python - "${PLAN_PATH}" "${POS_TOL_M}" "${MAX_CORRECTIONS}" "${CORRECTION_GAIN}" "${MAX_COMPENSATION_NORM_M}" "${MIN_MOTION_RAD}" "${COMMAND_DURATION_S}" "${SKIP_XYZ_WAYPOINT_STEP_RAD}" "${RETURN_HOME_WAYPOINT_STEP_RAD}" "${BOX_ABOVE_XY_TOL_M}" "${BOX_ABOVE_Z_MARGIN_FROM_CENTER_M}" "${FRAME_A0_PIPELINE}" "${CARTESIAN_IK_FK_TOL_M}" "${CARTESIAN_EARLY_MAX_JOINT_DELTA_RAD}" "${CARTESIAN_F6_MAX_JOINT_DELTA_RAD}" "${CARTESIAN_TOPDOWN_YAW_STEP_DEG}" "${F6_BOX_XY_TOL_M}" "${F6_POS_TOL_M}" "${R1LITE_CONFIG_PATH}" "${R1LITE_ROUND_COUNT}" "${R1LITE_SINGLE_ROUND_POLICY}" "${CORRECTION_STEP_CAP_M}" <<'PY'
import atexit
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel
from capx.integrations.r1lite.kinematics import (
    R1LiteCandidateKinematics,
    candidate_joint_delta_limit as configured_candidate_joint_delta_limit,
    rank_fk_qualified_candidates,
    solve_probes_with_reference_seed,
)
from capx.integrations.r1lite.generated_policy import (
    execute_dry_run as execute_generated_policy_dry_run,
    expected_operations as expected_generated_policy_operations,
)

repo = Path("/home/nvidia/xuanzhi/cap-x-agilex")
plan_path = Path(sys.argv[1])
pos_tol_m = float(sys.argv[2])
max_corrections = int(sys.argv[3])
correction_gain = float(sys.argv[4])
max_compensation_norm_m = float(sys.argv[5])
min_motion_rad = float(sys.argv[6])
command_duration_s = float(sys.argv[7])
skip_xyz_waypoint_step_rad = float(sys.argv[8])
return_home_waypoint_step_rad = float(sys.argv[9])
box_above_xy_tol_m = float(sys.argv[10])
box_above_z_margin_from_center_m = float(sys.argv[11])
frame_a0_pipeline = sys.argv[12]
cartesian_ik_fk_tol_m = float(sys.argv[13])
cartesian_early_max_joint_delta_rad = float(sys.argv[14])
cartesian_f6_max_joint_delta_rad = float(sys.argv[15])
cartesian_topdown_yaw_step_deg = float(sys.argv[16])
f6_box_xy_tol_m = float(sys.argv[17])
f6_pos_tol_m = float(sys.argv[18])
config_path = Path(sys.argv[19])
if not config_path.is_absolute():
    config_path = repo / config_path
round_count = int(sys.argv[20])
single_round_policy = str(sys.argv[21])
use_llm_absolute_20_policy = (
    round_count == 1 and single_round_policy == "llm_absolute_20"
)
correction_step_cap_raw = str(sys.argv[22]).strip()
correction_step_cap_m = (
    None if not correction_step_cap_raw else float(correction_step_cap_raw)
)
if correction_step_cap_m is not None and correction_step_cap_m <= 0.0:
    raise ValueError("CORRECTION_STEP_CAP_M must be positive when configured")
joint_arrival_tol_rad = float(os.environ["JOINT_ARRIVAL_TOL_RAD"])
joint_arrival_max_retries = int(os.environ["JOINT_ARRIVAL_MAX_RETRIES"])
llm_max_cartesian_frame_step_m = float(
    os.environ["LLM_MAX_CARTESIAN_FRAME_STEP_M"]
)
llm_max_incremental_joint_delta_rad = float(
    os.environ["LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD"]
)
cartesian_topdown_max_tilt_deg = float(
    os.environ["CARTESIAN_TOPDOWN_MAX_TILT_DEG"]
)
cartesian_topdown_tilt_step_deg = float(
    os.environ["CARTESIAN_TOPDOWN_TILT_STEP_DEG"]
)
coupled_grasp_tcp_to_fingertip_m = float(
    os.environ.get("COUPLED_GRASP_TCP_TO_FINGERTIP_M", "0.078")
)
coupled_grasp_ik_fk_tol_m = float(
    os.environ.get("COUPLED_GRASP_IK_FK_TOL_M", "0.015")
)
coupled_grasp_orientation_tol_rad = float(
    os.environ.get("COUPLED_GRASP_ORIENTATION_TOL_RAD", "0.30")
)
coupled_grasp_fingertip_tol_m = float(
    os.environ.get("COUPLED_GRASP_FINGERTIP_TOL_M", "0.015")
)
if joint_arrival_tol_rad <= 0.0:
    raise ValueError("JOINT_ARRIVAL_TOL_RAD must be positive")
if joint_arrival_max_retries < 0:
    raise ValueError("JOINT_ARRIVAL_MAX_RETRIES must be non-negative")
if not 0.0 < llm_max_cartesian_frame_step_m <= 0.10:
    raise ValueError(
        "LLM_MAX_CARTESIAN_FRAME_STEP_M must be in (0, 0.10]"
    )
if not 0.0 < llm_max_incremental_joint_delta_rad <= 1.0:
    raise ValueError(
        "LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD must be in (0, 1.0]"
    )
if not 0.0 <= cartesian_topdown_max_tilt_deg <= 60.0:
    raise ValueError("CARTESIAN_TOPDOWN_MAX_TILT_DEG must be in [0, 60]")
if cartesian_topdown_tilt_step_deg <= 0.0:
    raise ValueError("CARTESIAN_TOPDOWN_TILT_STEP_DEG must be positive")
if not 0.05 <= coupled_grasp_tcp_to_fingertip_m <= 0.12:
    raise ValueError("COUPLED_GRASP_TCP_TO_FINGERTIP_M must be in [0.05, 0.12]")
if not 0.001 <= coupled_grasp_ik_fk_tol_m <= 0.03:
    raise ValueError("COUPLED_GRASP_IK_FK_TOL_M must be in [0.001, 0.03]")
if not 0.01 <= coupled_grasp_orientation_tol_rad <= 0.50:
    raise ValueError("COUPLED_GRASP_ORIENTATION_TOL_RAD must be in [0.01, 0.50]")
if not 0.001 <= coupled_grasp_fingertip_tol_m <= 0.03:
    raise ValueError("COUPLED_GRASP_FINGERTIP_TOL_M must be in [0.001, 0.03]")
return_f1_position_tolerance_m = min(pos_tol_m, 0.02)
return_f1_orientation_tolerance_rad = 0.10
if frame_a0_pipeline not in {"nominal_joints", "cartesian_absolute"}:
    raise RuntimeError(
        f"FRAME_A0_PIPELINE must be nominal_joints or cartesian_absolute, got {frame_a0_pipeline!r}"
    )

cfg = yaml.safe_load(
    config_path.read_text()
)
low_cfg = dict(cfg["env"]["cfg"]["low_level"])
low_cfg.pop("_target_", None)
sig = inspect.signature(R1LiteRealLowLevel.__init__)
kwargs = {k: v for k, v in low_cfg.items() if k in sig.parameters}

r1 = R1LiteRealLowLevel(**kwargs)


def close_r1_at_exit() -> None:
    r1.close()


atexit.register(close_r1_at_exit)
candidate_kinematics = R1LiteCandidateKinematics()
plan_bytes = plan_path.read_bytes()
plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
plan = json.loads(plan_bytes)
waypoints = plan["waypoints"]
if len(waypoints) < 2:
    raise RuntimeError(f"Expected at least 2 waypoints, got {len(waypoints)}")
generated_policy_path_raw = os.environ.get("LLM_GENERATED_POLICY_PATH", "").strip()
generated_policy_operations = None
generated_policy_sha256 = None
if generated_policy_path_raw:
    generated_policy_path = Path(generated_policy_path_raw)
    generated_policy_bytes = generated_policy_path.read_bytes()
    generated_policy_sha256 = hashlib.sha256(
        generated_policy_bytes
    ).hexdigest()
    generated_policy_expected = expected_generated_policy_operations(
        plan, plan_sha256=plan_sha256
    )
    generated_policy_operations = execute_generated_policy_dry_run(
        generated_policy_bytes.decode(), expected=generated_policy_expected
    )
    if round_count != 1:
        raise RuntimeError("LLM-generated task policy requires exactly one round")
generated_policy_cursor = 0


def consume_generated_policy_operation(
    call: str, **kwargs: object
) -> None:
    global generated_policy_cursor
    if generated_policy_operations is None:
        return
    observed = {"call": call, **kwargs}
    if generated_policy_cursor >= len(generated_policy_operations):
        raise RuntimeError(
            f"generated policy exhausted before runtime operation {observed!r}"
        )
    expected = generated_policy_operations[generated_policy_cursor]
    if observed != expected:
        raise RuntimeError(
            "generated policy/runtime operation mismatch: "
            f"expected={expected!r}, observed={observed!r}"
        )
    generated_policy_cursor += 1


fourth_pipeline_metadata = plan.get("fourth_pipeline", {})
variable_frame_plan = bool(
    fourth_pipeline_metadata.get("variable_frame_plan", False)
)
coupled_grasp_enabled = bool(
    fourth_pipeline_metadata.get("coupled_grasp_enabled", False)
)
if coupled_grasp_enabled and not math.isclose(
    float(
        fourth_pipeline_metadata.get(
            "tcp_to_fingertip_m", coupled_grasp_tcp_to_fingertip_m
        )
    ),
    coupled_grasp_tcp_to_fingertip_m,
    rel_tol=0.0,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "plan and executor disagree on coupled-grasp TCP-to-fingertip offset"
    )


def waypoint_role(wp: dict) -> str:
    role = str(wp.get("semantic_role", "")).strip()
    if role:
        return role
    waypoint_id = str(wp.get("id", ""))
    if waypoint_id == "pick_01_lift_clear":
        return "f1_anchor"
    if waypoint_id.endswith("_grasp"):
        return "grasp"
    if waypoint_id in {"place_02_above_box", "place_03_release"}:
        return "above_box"
    return "legacy"


def is_place_waypoint(frame_index: int, wp: dict) -> bool:
    if variable_frame_plan:
        return waypoint_role(wp) == "above_box"
    return frame_index == 6


if variable_frame_plan:
    if not use_llm_absolute_20_policy:
        raise RuntimeError(
            "variable-frame plans require llm_absolute_20 single-round policy"
        )
    startup_deferred_roles = [
        waypoint_role(wp)
        for wp in waypoints
        if bool(wp.get("coupled_grasp_deferred", False))
    ]
    if coupled_grasp_enabled and startup_deferred_roles != [
            "above_ball",
            "pregrasp",
            "approach",
            "grasp",
        ]:
        raise RuntimeError(
            "variable-frame plan must contain exactly four ordered deferred "
            f"coupled-grasp stages; got {startup_deferred_roles}"
        )
    if not coupled_grasp_enabled and startup_deferred_roles:
        raise RuntimeError(
            "non-coupled variable plan cannot contain deferred grasp stages"
        )
    if float(
        fourth_pipeline_metadata.get(
            "max_cartesian_step_m", llm_max_cartesian_frame_step_m
        )
    ) > llm_max_cartesian_frame_step_m + 1e-9:
        raise RuntimeError("plan Cartesian step bound exceeds executor hard limit")
    previous_target = np.asarray(
        waypoints[1]["target_ee_position_base_m"], dtype=float
    )
    previous_deferred = False
    for waypoint in waypoints[2:]:
        target = np.asarray(
            waypoint["target_ee_position_base_m"], dtype=float
        )
        distance = float(np.linalg.norm(target - previous_target))
        deferred = bool(waypoint.get("coupled_grasp_deferred", False))
        if (
            not deferred
            and not previous_deferred
            and distance > llm_max_cartesian_frame_step_m + 1e-9
        ):
            raise RuntimeError(
                "variable plan contains an oversized Cartesian segment: "
                f"{waypoint.get('id')} distance={distance:.6f}m > "
                f"{llm_max_cartesian_frame_step_m:.6f}m"
            )
        previous_target = target
        previous_deferred = deferred

objects = plan.get("objects", {})
box_center = np.asarray(objects.get("box_center_base_m", [np.nan, np.nan, np.nan]), dtype=float)
ball_center = np.asarray(
    objects.get("tennis_ball_center_base_m", [np.nan, np.nan, np.nan]),
    dtype=float,
)
if coupled_grasp_enabled and not np.all(np.isfinite(ball_center)):
    raise RuntimeError("coupled grasp requires a finite tennis-ball center")
llm_grasp_preference = fourth_pipeline_metadata.get(
    "llm_grasp_preference"
)
if coupled_grasp_enabled:
    if not isinstance(llm_grasp_preference, dict):
        raise RuntimeError(
            "coupled grasp requires an LLM grasp-preference object"
        )
    preferred_yaw_deg = float(
        llm_grasp_preference.get("preferred_yaw_deg", float("nan"))
    )
    preferred_tilt_deg = float(
        llm_grasp_preference.get("preferred_tilt_deg", float("nan"))
    )
    preferred_approach_side = str(
        llm_grasp_preference.get("approach_side", "")
    )
    if (
        not np.isfinite([preferred_yaw_deg, preferred_tilt_deg]).all()
        or not 0.0 <= preferred_yaw_deg < 360.0
        or not -60.0 <= preferred_tilt_deg <= 60.0
        or preferred_approach_side
        not in {"top", "front", "rear", "left", "right"}
    ):
        raise RuntimeError(
            f"invalid LLM grasp preference: {llm_grasp_preference}"
        )
else:
    preferred_yaw_deg = float("nan")
    preferred_tilt_deg = float("nan")
    preferred_approach_side = ""

log_path = (
    repo
    / "outputs/r1lite_tennis_ball_in_box"
    / f"ten_keyframe_xyz_corrected_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
)
active_round_index = None
last_waypoint_failure = None
round1_max_fk_retries = 3
round1_manual_reset_signal_path = Path(
    "/tmp/capx_r1lite_round1_manual_reset_ready"
)


def write(event: dict) -> None:
    if active_round_index is not None and "round_index" not in event:
        event["round_index"] = active_round_index
    event["t_unix"] = time.time()
    with log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def is_fk_gate_failure(exc: Exception) -> bool:
    message = repr(exc)
    return (
        "FK validation failed" in message
        or "no local-FK-qualified candidates" in message
    )


def physical_ee_pose() -> tuple[np.ndarray, np.ndarray]:
    """Read physical feedback converted to RelaxedIK's tool-Z TCP."""
    return r1.get_physical_tool_z_pose()


def quat_error_rad(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    a = np.asarray(a_xyzw, dtype=float).reshape(4)
    b = np.asarray(b_xyzw, dtype=float).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * math.acos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def capped(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


def capped_for_active_policy(
    vector: np.ndarray, max_norm: float | None
) -> np.ndarray:
    if max_norm is None:
        return np.asarray(vector, dtype=float).copy()
    return capped(vector, max_norm)


def normalized_quat(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8 or not np.isfinite(norm):
        raise ValueError("quaternion_xyzw must be finite and non-zero")
    return quat / norm


def require_finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains NaN or Inf: {array.tolist()}")
    return array


def gripper_link_quat_to_tool_z_quat(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Convert physical right_gripper_link orientation to RelaxedIK tool_z."""
    gripper_rot = SciRotation.from_quat(normalized_quat(quaternion_xyzw))
    tool_from_gripper = SciRotation.from_euler("y", math.pi / 2.0)
    return (gripper_rot * tool_from_gripper).as_quat()


def topdown_gripper_link_quat(yaw_rad: float) -> np.ndarray:
    """Return a right_gripper_link top-down orientation in base_link.

    The gripper_link +X axis is treated as the approach axis. Top-down means +X
    points toward base -Z; yaw selects the horizontal gripper roll about -Z.
    """
    x_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    y_axis = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=float)
    z_axis = np.cross(x_axis, y_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return SciRotation.from_matrix(rot).as_quat()


def topdown_tool_z_quat(yaw_rad: float) -> np.ndarray:
    return gripper_link_quat_to_tool_z_quat(topdown_gripper_link_quat(yaw_rad))


def topdown_yaw_degrees() -> list[float]:
    step = max(1.0, float(cartesian_topdown_yaw_step_deg))
    count = max(1, int(math.ceil(360.0 / step)))
    return [float((i * step) % 360.0) for i in range(count)]


def topdown_tilt_degrees() -> list[float]:
    maximum = float(cartesian_topdown_max_tilt_deg)
    if maximum <= 0.0:
        return [0.0]
    step = float(cartesian_topdown_tilt_step_deg)
    magnitudes = []
    value = step
    while value < maximum - 1e-9:
        magnitudes.append(value)
        value += step
    magnitudes.append(maximum)
    return [0.0] + [
        signed
        for magnitude in magnitudes
        for signed in (-float(magnitude), float(magnitude))
    ]


def topdown_orientation_probes() -> list[dict]:
    probes = []
    for yaw_deg in topdown_yaw_degrees():
        topdown_gripper = SciRotation.from_quat(
            topdown_gripper_link_quat(math.radians(yaw_deg))
        )
        for tilt_deg in topdown_tilt_degrees():
            tilted_gripper = topdown_gripper * SciRotation.from_euler(
                "y", math.radians(tilt_deg)
            )
            probes.append(
                {
                    "yaw_deg": yaw_deg,
                    "tilt_deg": tilt_deg,
                    "quaternion_xyzw": gripper_link_quat_to_tool_z_quat(
                        tilted_gripper.as_quat()
                    ),
                }
            )
    return probes


def tool_approach_axis_base(quaternion_xyzw: np.ndarray) -> np.ndarray:
    axis = SciRotation.from_quat(
        normalized_quat(quaternion_xyzw)
    ).apply(np.array([0.0, 0.0, 1.0], dtype=float))
    return axis / np.linalg.norm(axis)


def coupled_tcp_target(
    contact_center_base_m: np.ndarray,
    quaternion_xyzw: np.ndarray,
    clearance_m: float,
) -> np.ndarray:
    axis = tool_approach_axis_base(quaternion_xyzw)
    return np.asarray(contact_center_base_m, dtype=float).reshape(3) - (
        coupled_grasp_tcp_to_fingertip_m + float(clearance_m)
    ) * axis


def fingertip_center_from_tcp(
    tcp_position_base_m: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    return np.asarray(tcp_position_base_m, dtype=float).reshape(3) + (
        coupled_grasp_tcp_to_fingertip_m
        * tool_approach_axis_base(quaternion_xyzw)
    )


def candidate_joint_delta_limit(frame_index: int, wp: dict) -> float | None:
    if variable_frame_plan:
        return llm_max_incremental_joint_delta_rad
    return configured_candidate_joint_delta_limit(
        frame_index,
        early_limit_rad=cartesian_early_max_joint_delta_rad,
        f6_limit_rad=cartesian_f6_max_joint_delta_rad,
    )


def get_cartesian_pipeline_candidate(
    position: np.ndarray,
    *,
    frame_index: int,
    place_policy: bool,
    reference_joints: np.ndarray,
    start_joints: np.ndarray,
    max_incremental_joint_delta_rad: float | None = None,
    validate_fk: bool = True,
    locked_quaternion_xyzw: np.ndarray | None = None,
    fk_tolerance_m: float | None = None,
    topdown_only: bool = False,
) -> tuple[np.ndarray, dict]:
    _base_from_torso, torso_joints = r1._base_from_torso_link3()
    active_fk_tolerance_m = (
        cartesian_ik_fk_tol_m
        if fk_tolerance_m is None
        else float(fk_tolerance_m)
    )
    active_orientation_probes = [
        probe
        for probe in topdown_orientation_probes()
        if not topdown_only or abs(float(probe["tilt_deg"])) <= 1e-9
    ]
    if locked_quaternion_xyzw is not None:
        quat = normalized_quat(locked_quaternion_xyzw)
        r1.seed_relaxed_ik_probe(reference_joints)
        candidate = r1.get_relaxed_ik_candidate(
            position,
            quat,
            timeout_s=1.0,
            reference_joints=reference_joints,
            start_joints=start_joints,
            max_incremental_joint_delta_rad=max_incremental_joint_delta_rad,
        )
        candidate = require_finite("locked IK candidate joints", candidate)
        local_fk = candidate_kinematics.fk(candidate, torso_joints)
        fk_pos = require_finite(
            "locked candidate FK position", local_fk.position_base_m
        )
        fk_quat = require_finite(
            "locked candidate FK quaternion", local_fk.quaternion_xyzw
        )
        fk_error_m = float(np.linalg.norm(fk_pos - position))
        orientation_error_rad = quat_error_rad(
            quat, fk_quat
        )
        require_finite(
            "locked candidate diagnostics",
            np.array(
                [
                    fk_error_m,
                    orientation_error_rad,
                    local_fk.dual_position_error_m,
                    local_fk.dual_orientation_error_rad,
                ]
            ),
        )
        if (
            local_fk.dual_position_error_m > 1e-6
            or local_fk.dual_orientation_error_rad > 1e-6
        ):
            raise RuntimeError(
                "Locked candidate dual-URDF FK mismatch: "
                f"position={local_fk.dual_position_error_m:.6g}m, "
                f"orientation={local_fk.dual_orientation_error_rad:.6g}rad"
            )
        if validate_fk and fk_error_m > active_fk_tolerance_m:
            raise RuntimeError(
                "FK validation failed for locked coupled orientation: "
                f"fk_error_m={fk_error_m:.4f} > "
                f"tol={active_fk_tolerance_m:.4f}, "
                f"target={position.tolist()}, "
                f"fk_position={fk_pos.tolist()}"
            )
        if (
            validate_fk
            and orientation_error_rad > coupled_grasp_orientation_tol_rad
        ):
            raise RuntimeError(
                "FK orientation validation failed for locked coupled orientation: "
                f"error={orientation_error_rad:.4f}rad > "
                f"tol={coupled_grasp_orientation_tol_rad:.4f}rad"
            )
        return candidate, {
            "orientation_mode": "coupled_grasp_locked",
            "fix_scope": "coupled_pick_segment",
            "fk_source": "candidate_local_dual_urdf",
            "fk_validation_enabled": bool(validate_fk),
            "selected_yaw_deg": None,
            "selected_tilt_deg": None,
            "selected_quaternion_xyzw": quat.tolist(),
            "fk_position_base_m": fk_pos.tolist(),
            "fk_quaternion_xyzw": fk_quat.tolist(),
            "fk_error_m": fk_error_m,
            "fk_error_xyz_m": (fk_pos - position).tolist(),
            "fk_orientation_error_rad": orientation_error_rad,
            "fk_tol_m": active_fk_tolerance_m,
            "max_joint_delta_from_start_rad": float(
                np.max(np.abs(candidate - start_joints))
            ),
            "dual_urdf_position_error_m": local_fk.dual_position_error_m,
            "dual_urdf_orientation_error_rad": (
                local_fk.dual_orientation_error_rad
            ),
            "top_candidates": [],
            "candidate_errors": [],
        }
    if not place_policy:
        candidates: list[dict] = []
        errors: list[dict] = []
        for probe in active_orientation_probes:
            yaw_deg = float(probe["yaw_deg"])
            tilt_deg = float(probe["tilt_deg"])
            quat = np.asarray(probe["quaternion_xyzw"], dtype=float)
            try:
                candidate = r1.get_relaxed_ik_candidate(
                    position,
                    quat,
                    timeout_s=1.0,
                    reference_joints=reference_joints,
                    start_joints=start_joints,
                    max_incremental_joint_delta_rad=max_incremental_joint_delta_rad,
                )
                candidate = require_finite(
                    "orientation-search IK candidate joints", candidate
                )
                local_fk = candidate_kinematics.fk(candidate, torso_joints)
                fk_pos = require_finite(
                    "orientation-search FK position",
                    local_fk.position_base_m,
                )
                fk_quat = require_finite(
                    "orientation-search FK quaternion",
                    local_fk.quaternion_xyzw,
                )
                orientation_error_rad = quat_error_rad(quat, fk_quat)
                require_finite(
                    "orientation-search FK diagnostics",
                    np.array(
                        [
                            local_fk.dual_position_error_m,
                            local_fk.dual_orientation_error_rad,
                        ]
                    ),
                )
                if (
                    local_fk.dual_position_error_m > 1e-6
                    or local_fk.dual_orientation_error_rad > 1e-6
                ):
                    raise RuntimeError(
                        "Candidate dual-URDF FK mismatch: "
                        f"position={local_fk.dual_position_error_m:.6g}m, "
                        "orientation="
                        f"{local_fk.dual_orientation_error_rad:.6g}rad"
                    )
                if (
                    topdown_only
                    and orientation_error_rad
                    > coupled_grasp_orientation_tol_rad
                ):
                    raise RuntimeError(
                        "Strict top-down grasp candidate orientation error "
                        f"{orientation_error_rad:.4f}rad exceeds "
                        f"{coupled_grasp_orientation_tol_rad:.4f}rad"
                    )
                candidates.append(
                    {
                        "yaw_deg": yaw_deg,
                        "tilt_deg": tilt_deg,
                        "quaternion_xyzw": quat.tolist(),
                        "candidate": candidate,
                        "fk_position_base_m": fk_pos.tolist(),
                        "fk_error_m": float(np.linalg.norm(fk_pos - position)),
                        "fk_error_xyz_m": (fk_pos - position).tolist(),
                        "fk_orientation_error_rad": orientation_error_rad,
                        "max_joint_delta_from_start_rad": float(
                            np.max(np.abs(candidate - start_joints))
                        ),
                        "dual_urdf_position_error_m": (
                            local_fk.dual_position_error_m
                        ),
                        "dual_urdf_orientation_error_rad": (
                            local_fk.dual_orientation_error_rad
                        ),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "yaw_deg": yaw_deg,
                        "tilt_deg": tilt_deg,
                        "error": repr(exc),
                    }
                )
        if not candidates:
            raise RuntimeError(
                "Top-down yaw/tilt search produced no IK candidates: "
                f"{errors}"
            )
        candidates.sort(
            key=lambda item: (
                float(item["fk_error_m"]),
                float(item["max_joint_delta_from_start_rad"]),
            )
        )
        best = candidates[0]
        fk_diag = {
            "orientation_mode": (
                "strict_topdown_yaw_search"
                if topdown_only
                else "topdown_yaw_tilt_search"
            ),
            "fix_scope": "legacy_solver_candidate_local_fk",
            "fk_source": "candidate_local_dual_urdf",
            "fk_validation_enabled": bool(validate_fk),
            "selected_yaw_deg": best["yaw_deg"],
            "selected_tilt_deg": best["tilt_deg"],
            "selected_quaternion_xyzw": best["quaternion_xyzw"],
            "fk_position_base_m": best["fk_position_base_m"],
            "fk_error_m": best["fk_error_m"],
            "fk_error_xyz_m": best["fk_error_xyz_m"],
            "fk_orientation_error_rad": best[
                "fk_orientation_error_rad"
            ],
            "fk_tol_m": active_fk_tolerance_m,
            "max_joint_delta_from_start_rad": best[
                "max_joint_delta_from_start_rad"
            ],
            "dual_urdf_position_error_m": best[
                "dual_urdf_position_error_m"
            ],
            "dual_urdf_orientation_error_rad": best[
                "dual_urdf_orientation_error_rad"
            ],
            "top_candidates": [
                {
                    "yaw_deg": item["yaw_deg"],
                    "tilt_deg": item["tilt_deg"],
                    "fk_error_m": item["fk_error_m"],
                    "max_joint_delta_from_start_rad": item[
                        "max_joint_delta_from_start_rad"
                    ],
                    "fk_position_base_m": item["fk_position_base_m"],
                    "fk_orientation_error_rad": item[
                        "fk_orientation_error_rad"
                    ],
                    "dual_urdf_position_error_m": item[
                        "dual_urdf_position_error_m"
                    ],
                    "dual_urdf_orientation_error_rad": item[
                        "dual_urdf_orientation_error_rad"
                    ],
                }
                for item in candidates[:5]
            ],
            "candidate_errors": errors[:5],
        }
        if validate_fk and float(best["fk_error_m"]) > active_fk_tolerance_m:
            raise RuntimeError(
                "FK validation failed: "
                f"fk_error_m={float(best['fk_error_m']):.4f} > "
                f"tol={active_fk_tolerance_m:.4f}, "
                f"target={position.tolist()}, "
                f"fk_position={best['fk_position_base_m']}, "
                f"selected_yaw_deg={best['yaw_deg']}, "
                f"selected_tilt_deg={best['tilt_deg']}"
            )
        return np.asarray(best["candidate"], dtype=float), fk_diag

    def solve_orientation(probe: dict) -> dict:
        yaw_deg = float(probe["yaw_deg"])
        tilt_deg = float(probe["tilt_deg"])
        quat = np.asarray(probe["quaternion_xyzw"], dtype=float)
        candidate = r1.get_relaxed_ik_candidate(
            position,
            quat,
            timeout_s=1.0,
            reference_joints=reference_joints,
            start_joints=start_joints,
            max_incremental_joint_delta_rad=max_incremental_joint_delta_rad,
        )
        candidate = require_finite(
            "deterministic orientation-search IK candidate joints",
            candidate,
        )
        local_fk = candidate_kinematics.fk(candidate, torso_joints)
        fk_pos = require_finite(
            "deterministic orientation-search FK position",
            local_fk.position_base_m,
        )
        require_finite(
            "deterministic orientation-search FK quaternion",
            local_fk.quaternion_xyzw,
        )
        require_finite(
            "deterministic orientation-search FK diagnostics",
            np.array(
                [
                    local_fk.dual_position_error_m,
                    local_fk.dual_orientation_error_rad,
                ]
            ),
        )
        return {
            "yaw_deg": yaw_deg,
            "tilt_deg": tilt_deg,
            "quaternion_xyzw": quat.tolist(),
            "candidate": candidate,
            "fk_position_base_m": fk_pos.tolist(),
            "fk_error_m": float(np.linalg.norm(fk_pos - position)),
            "fk_error_xyz_m": (fk_pos - position).tolist(),
            "max_joint_delta_from_start_rad": float(
                np.max(np.abs(candidate - start_joints))
            ),
            "dual_urdf_position_error_m": local_fk.dual_position_error_m,
            "dual_urdf_orientation_error_rad": local_fk.dual_orientation_error_rad,
        }

    candidates, probe_errors = solve_probes_with_reference_seed(
        active_orientation_probes,
        reference_joints,
        seed_solver=r1.seed_relaxed_ik_probe,
        solve_probe=solve_orientation,
    )
    errors = [
        {
            "yaw_deg": item["probe"]["yaw_deg"],
            "tilt_deg": item["probe"]["tilt_deg"],
            "error": item["error"],
        }
        for item in probe_errors
    ]
    if not candidates:
        raise RuntimeError(
            f"F6 free-tilt orientation search produced no IK candidates: {errors}"
        )

    qualified, rejected = rank_fk_qualified_candidates(
        candidates,
        fk_tolerance_m=(
            active_fk_tolerance_m if validate_fk else float("inf")
        ),
    )
    if not qualified:
        raise RuntimeError(
            "F6 free-tilt orientation search produced no local-FK-qualified candidates: "
            f"target={position.tolist()}, rejected={rejected}, errors={errors}"
        )
    best = qualified[0]
    fk_diag = {
        "orientation_mode": (
            "strict_topdown_yaw_search_deterministic_seed"
            if topdown_only
            else "topdown_yaw_tilt_search_deterministic_seed"
        ),
        "fix_scope": "f6_or_semantic_above_box_only",
        "fk_source": "candidate_local_dual_urdf",
        "fk_validation_enabled": bool(validate_fk),
        "selected_yaw_deg": best["yaw_deg"],
        "selected_tilt_deg": best["tilt_deg"],
        "selected_quaternion_xyzw": best["quaternion_xyzw"],
        "fk_position_base_m": best["fk_position_base_m"],
        "fk_error_m": best["fk_error_m"],
        "fk_error_xyz_m": best["fk_error_xyz_m"],
        "fk_tol_m": active_fk_tolerance_m,
        "max_joint_delta_from_start_rad": best["max_joint_delta_from_start_rad"],
        "dual_urdf_position_error_m": best["dual_urdf_position_error_m"],
        "dual_urdf_orientation_error_rad": best[
            "dual_urdf_orientation_error_rad"
        ],
        "top_candidates": [
            {
                "yaw_deg": item["yaw_deg"],
                "tilt_deg": item["tilt_deg"],
                "fk_error_m": item["fk_error_m"],
                "max_joint_delta_from_start_rad": item["max_joint_delta_from_start_rad"],
                "fk_position_base_m": item["fk_position_base_m"],
                "dual_urdf_position_error_m": item[
                    "dual_urdf_position_error_m"
                ],
                "dual_urdf_orientation_error_rad": item[
                    "dual_urdf_orientation_error_rad"
                ],
            }
            for item in qualified[:5]
        ],
        "rejected_candidates": rejected,
        "candidate_errors": errors[:5],
    }
    return np.asarray(best["candidate"], dtype=float), fk_diag


def resolve_coupled_grasp_chain(
    current_position: np.ndarray,
    current_joints: np.ndarray,
) -> dict:
    deferred = [
        wp for wp in waypoints if bool(wp.get("coupled_grasp_deferred", False))
    ]
    expected_roles = ["above_ball", "pregrasp", "approach", "grasp"]
    if [waypoint_role(wp) for wp in deferred] != expected_roles:
        raise RuntimeError(
            "coupled grasp requires exactly the ordered deferred stages "
            f"{expected_roles}"
        )
    all_probes = topdown_orientation_probes()
    nearby_probes = [
        probe
        for probe in all_probes
        if min(
            abs(float(probe["yaw_deg"]) - preferred_yaw_deg),
            360.0 - abs(float(probe["yaw_deg"]) - preferred_yaw_deg),
        )
        <= cartesian_topdown_yaw_step_deg + 1e-9
        and abs(float(probe["tilt_deg"]) - preferred_tilt_deg)
        <= cartesian_topdown_tilt_step_deg + 1e-9
    ]
    if not nearby_probes:
        raise RuntimeError(
            "LLM grasp preference produced an empty nearby probe set"
        )
    nearby_keys = {
        (float(probe["yaw_deg"]), float(probe["tilt_deg"]))
        for probe in nearby_probes
    }
    global_fallback_probes = [
        probe
        for probe in all_probes
        if (float(probe["yaw_deg"]), float(probe["tilt_deg"]))
        not in nearby_keys
    ]
    accepted: list[dict] = []
    rejected: list[dict] = []
    selected_search_mode = "llm_neighborhood"
    fallback_used = False
    search_sequence: list[dict | None] = (
        nearby_probes + [None] + global_fallback_probes
    )
    for probe in search_sequence:
        if probe is None:
            if accepted:
                break
            selected_search_mode = "global_fallback"
            fallback_used = True
            continue
        quat = normalized_quat(probe["quaternion_xyzw"])
        axis = tool_approach_axis_base(quat)
        stage_targets = {
            waypoint_role(wp): coupled_tcp_target(
                ball_center,
                quat,
                float(wp["coupled_grasp_clearance_m"]),
            )
            for wp in deferred
        }
        try:
            for target in stage_targets.values():
                r1.validate_workspace_point(target)
            r1.seed_relaxed_ik_probe(current_joints)
            previous_position = np.asarray(current_position, dtype=float).copy()
            previous_joints = np.asarray(current_joints, dtype=float).copy()
            stage_solutions: dict[str, list[float]] = {}
            max_joint_delta = 0.0
            total_joint_delta = 0.0
            max_fk_error = 0.0
            max_orientation_error = 0.0
            grasp_fk_fingertip_error_m = None
            validation_steps = 0
            for wp in deferred:
                role = waypoint_role(wp)
                target = stage_targets[role]
                delta = target - previous_position
                segment_count = max(
                    1,
                    int(
                        math.ceil(
                            float(np.linalg.norm(delta))
                            / llm_max_cartesian_frame_step_m
                        )
                    ),
                )
                for segment_index in range(1, segment_count + 1):
                    subtarget = previous_position + delta * (
                        segment_index / segment_count
                    )
                    candidate, diag = get_cartesian_pipeline_candidate(
                        subtarget,
                        frame_index=int(wp["index"]),
                        place_policy=False,
                        reference_joints=previous_joints,
                        start_joints=previous_joints,
                        max_incremental_joint_delta_rad=(
                            llm_max_incremental_joint_delta_rad
                        ),
                        validate_fk=True,
                        locked_quaternion_xyzw=quat,
                        fk_tolerance_m=coupled_grasp_ik_fk_tol_m,
                    )
                    joint_delta = float(
                        np.max(np.abs(candidate - previous_joints))
                    )
                    max_joint_delta = max(max_joint_delta, joint_delta)
                    total_joint_delta += joint_delta
                    max_fk_error = max(
                        max_fk_error, float(diag["fk_error_m"])
                    )
                    max_orientation_error = max(
                        max_orientation_error,
                        float(diag["fk_orientation_error_rad"]),
                    )
                    if role == "grasp" and segment_index == segment_count:
                        grasp_fk_fingertip = fingertip_center_from_tcp(
                            np.asarray(
                                diag["fk_position_base_m"], dtype=float
                            ),
                            np.asarray(
                                diag["fk_quaternion_xyzw"], dtype=float
                            ),
                        )
                        grasp_fk_fingertip_error_m = float(
                            np.linalg.norm(
                                grasp_fk_fingertip - ball_center
                            )
                        )
                        if (
                            not math.isfinite(
                                grasp_fk_fingertip_error_m
                            )
                            or grasp_fk_fingertip_error_m
                            > coupled_grasp_fingertip_tol_m
                        ):
                            raise RuntimeError(
                                "coupled grasp candidate FK fingertip misses "
                                f"ball center: error="
                                f"{grasp_fk_fingertip_error_m:.6f}m > "
                                f"{coupled_grasp_fingertip_tol_m:.6f}m"
                            )
                    previous_joints = candidate
                    validation_steps += 1
                previous_position = target
                stage_solutions[role] = previous_joints.tolist()
            accepted.append(
                {
                    "yaw_deg": float(probe["yaw_deg"]),
                    "tilt_deg": float(probe["tilt_deg"]),
                    "quaternion_xyzw": quat.tolist(),
                    "approach_axis_base": axis.tolist(),
                    "stage_targets": {
                        role: target.tolist()
                        for role, target in stage_targets.items()
                    },
                    "stage_solutions": stage_solutions,
                    "max_joint_delta_rad": max_joint_delta,
                    "total_joint_delta_rad": total_joint_delta,
                    "max_fk_error_m": max_fk_error,
                    "max_orientation_error_rad": max_orientation_error,
                    "grasp_fk_fingertip_error_m": (
                        grasp_fk_fingertip_error_m
                    ),
                    "validation_steps": validation_steps,
                }
            )
        except Exception as exc:
            rejected.append(
                {
                    "yaw_deg": float(probe["yaw_deg"]),
                    "tilt_deg": float(probe["tilt_deg"]),
                    "approach_axis_base": axis.tolist(),
                    "error": repr(exc),
                }
            )
    if not accepted:
        raise RuntimeError(
            "coupled grasp search found no full-chain IK/FK-qualified direction: "
            f"{rejected}"
        )
    accepted.sort(
        key=lambda item: (
            float(item["max_joint_delta_rad"]),
            float(item["total_joint_delta_rad"]),
            float(item["max_fk_error_m"]),
        )
    )
    best = accepted[0]
    for wp in deferred:
        role = waypoint_role(wp)
        wp["target_ee_position_base_m"] = best["stage_targets"][role]
        wp["target_quaternion_xyzw"] = best["quaternion_xyzw"]
        wp["joint_feedback_solution_rad"] = best["stage_solutions"][role]
        wp["coupled_grasp_deferred"] = False
        wp["coupled_grasp_locked"] = True
        wp["selected_approach_axis_base"] = best["approach_axis_base"]
        wp["target_source"] = "system_coupled_grasp_resolved"
    diagnostic = {
        "event": "coupled_grasp_chain_resolved",
        "ball_center_base_m": ball_center.tolist(),
        "tcp_to_fingertip_m": coupled_grasp_tcp_to_fingertip_m,
        "tcp_to_fingertip_source": fourth_pipeline_metadata.get(
            "tcp_to_fingertip_source", "urdf_mesh_estimate"
        ),
        "selected_yaw_deg": best["yaw_deg"],
        "selected_tilt_deg": best["tilt_deg"],
        "llm_preferred_approach_side": preferred_approach_side,
        "llm_preferred_yaw_deg": preferred_yaw_deg,
        "llm_preferred_tilt_deg": preferred_tilt_deg,
        "selected_search_mode": selected_search_mode,
        "global_fallback_used": fallback_used,
        "nearby_probe_count": len(nearby_probes),
        "selected_quaternion_xyzw": best["quaternion_xyzw"],
        "selected_approach_axis_base": best["approach_axis_base"],
        "stage_tcp_targets_base_m": best["stage_targets"],
        "max_joint_delta_rad": best["max_joint_delta_rad"],
        "total_joint_delta_rad": best["total_joint_delta_rad"],
        "max_fk_error_m": best["max_fk_error_m"],
        "max_orientation_error_rad": best["max_orientation_error_rad"],
        "grasp_fk_fingertip_error_m": best[
            "grasp_fk_fingertip_error_m"
        ],
        "qualified_direction_count": len(accepted),
        "rejected_direction_count": len(rejected),
        "top_qualified_directions": [
            {
                key: item[key]
                for key in (
                    "yaw_deg",
                    "tilt_deg",
                    "approach_axis_base",
                    "max_joint_delta_rad",
                    "total_joint_delta_rad",
                    "max_fk_error_m",
                    "max_orientation_error_rad",
                    "grasp_fk_fingertip_error_m",
                )
            }
            for item in accepted[:5]
        ],
        "rejected_directions": rejected[:20],
    }
    write(diagnostic)
    return diagnostic


def box_xy_error_and_z_margin(position: np.ndarray) -> tuple[float, float]:
    if not np.all(np.isfinite(box_center)):
        return float("nan"), float("nan")
    xy_error_m = float(np.linalg.norm(position[:2] - box_center[:2]))
    z_margin_m = float(position[2] - box_center[2])
    return xy_error_m, z_margin_m


def is_above_box(position: np.ndarray) -> tuple[bool, float, float]:
    xy_error_m, z_margin_m = box_xy_error_and_z_margin(position)
    return (
        xy_error_m <= box_above_xy_tol_m
        and z_margin_m >= box_above_z_margin_from_center_m
    ), xy_error_m, z_margin_m


def command_frame(frame_index: int, wp: dict, command_joints: np.ndarray, attempt: int) -> tuple[dict, np.ndarray, np.ndarray]:
    before_pos, before_quat = physical_ee_pose()
    before_joints = r1.get_arm_joints()
    context = f"ten_keyframe_{frame_index:02d}_{wp['id']}_attempt_{attempt:02d}"
    if bool(wp.get("skip_xyz_correction", False)):
        result = r1.move_joint_waypoints_adaptive(
            command_joints,
            waypoint_step_rad=skip_xyz_waypoint_step_rad,
            context=context,
            skip_feedback_settle=True,
        )
    else:
        result = r1.move_joints_feedforward(
            command_joints,
            duration_s=command_duration_s,
            context=context,
            skip_feedback_settle=True,
        )

    arrival_history = []
    arrival_reached = False
    settled_joints = r1.wait_for_settled_arm_joints()
    for retry_index in range(joint_arrival_max_retries + 1):
        max_joint_error_rad = float(
            np.max(np.abs(command_joints - settled_joints))
        )
        arrival_record = {
            "retry_index": retry_index,
            "settled_joints": settled_joints.tolist(),
            "max_joint_error_rad": max_joint_error_rad,
        }
        arrival_history.append(arrival_record)
        if max_joint_error_rad <= joint_arrival_tol_rad:
            arrival_reached = True
            break
        if retry_index >= joint_arrival_max_retries:
            break
        retry_result = r1.move_joints_feedforward(
            command_joints,
            duration_s=command_duration_s,
            context=f"{context}_joint_arrival_retry_{retry_index + 1:02d}",
            skip_feedback_settle=True,
        )
        arrival_record["retry_command_result"] = retry_result
        settled_joints = r1.wait_for_settled_arm_joints()

    # Read the vendor TCP only after a fresh stable joint-feedback window.
    time.sleep(0.10)
    actual_pos, actual_quat = physical_ee_pose()
    actual_joints = settled_joints
    torso_joints = r1.get_torso_joints()
    command_fk = candidate_kinematics.fk(command_joints, torso_joints)
    actual_joint_fk = candidate_kinematics.fk(actual_joints, torso_joints)
    result = dict(result)
    result["success"] = arrival_reached
    result["max_feedback_error_rad"] = float(
        np.max(np.abs(command_joints - actual_joints))
    )
    result["joint_arrival"] = {
        "reached": arrival_reached,
        "tolerance_rad": joint_arrival_tol_rad,
        "max_retries": joint_arrival_max_retries,
        "history": arrival_history,
    }
    result["kinematic_execution_decomposition"] = {
        "torso_joints_rad": torso_joints.tolist(),
        "ik_command_fk_position_base_m": (
            command_fk.position_base_m.tolist()
        ),
        "actual_joint_fk_position_base_m": (
            actual_joint_fk.position_base_m.tolist()
        ),
        "vendor_actual_position_base_m": actual_pos.tolist(),
        "joint_tracking_position_error_m": float(
            np.linalg.norm(
                actual_joint_fk.position_base_m
                - command_fk.position_base_m
            )
        ),
        "vendor_vs_actual_joint_fk_error_m": float(
            np.linalg.norm(
                actual_pos - actual_joint_fk.position_base_m
            )
        ),
        "vendor_vs_actual_joint_fk_error_xyz_m": (
            actual_pos - actual_joint_fk.position_base_m
        ).tolist(),
        "ik_command_fk_to_vendor_error_m": float(
            np.linalg.norm(actual_pos - command_fk.position_base_m)
        ),
    }
    return result, actual_pos, actual_quat, before_pos, before_quat, before_joints, actual_joints


def planned_position_for_active_round(frame_index: int, wp: dict) -> np.ndarray:
    return np.asarray(wp["target_ee_position_base_m"], dtype=float).reshape(3).copy()


def execute_waypoint(frame_index: int, wp: dict, prev_wp: dict | None = None) -> bool:
    global last_waypoint_failure
    last_waypoint_failure = None
    wid = wp["id"]
    if (
        coupled_grasp_enabled
        and waypoint_role(wp)
        in {"above_ball", "pregrasp", "approach", "grasp"}
        and not bool(wp.get("coupled_grasp_locked", False))
    ):
        raise RuntimeError(
            f"refusing unresolved coupled-grasp waypoint {wid}"
        )
    skip_xyz_correction = bool(wp.get("skip_xyz_correction", False))
    first_round_correction_limits = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    first_round_force_accept_after_corrections = (
        first_round_correction_limits.get(frame_index)
        if active_round_index == 1 and not use_llm_absolute_20_policy
        else None
    )
    frame_max_corrections = (
        first_round_force_accept_after_corrections
        if first_round_force_accept_after_corrections is not None
        else max_corrections
    )
    if variable_frame_plan and waypoint_role(wp) == "transit":
        frame_max_corrections = 0
    desired_position = planned_position_for_active_round(frame_index, wp)
    desired_quat = np.asarray(wp["target_quaternion_xyzw"], dtype=float).reshape(4)
    nominal_joints = np.asarray(wp["joint_feedback_solution_rad"], dtype=float).reshape(6)
    place_waypoint = is_place_waypoint(frame_index, wp)
    coupled_pick_waypoint = bool(wp.get("coupled_grasp_locked", False))
    grasp_waypoint = waypoint_role(wp) == "grasp"
    desired_fingertip = (
        fingertip_center_from_tcp(desired_position, desired_quat)
        if coupled_pick_waypoint
        else None
    )
    frame_position_tolerance_m = (
        coupled_grasp_fingertip_tol_m
        if grasp_waypoint and coupled_pick_waypoint
        else f6_pos_tol_m
        if place_waypoint
        else pos_tol_m
    )
    use_cartesian_online_ik = (
        frame_a0_pipeline == "cartesian_absolute"
        and frame_index >= 2
        and not skip_xyz_correction
    )
    if use_llm_absolute_20_policy and use_cartesian_online_ik:
        frame_cartesian_step_cap_m = None
    else:
        frame_cartesian_step_cap_m = (
            0.35
            if active_round_index == 1 and use_cartesian_online_ik
            else max_compensation_norm_m
        )

    write(
        {
            "event": "frame_begin",
            "frame_index": frame_index,
            "id": wid,
            "label": wp["label"],
            "semantic_role": waypoint_role(wp),
            "target_source": wp.get("target_source"),
            "is_interpolated": bool(wp.get("is_interpolated", False)),
            "desired_position_base_m": desired_position.tolist(),
            "desired_quaternion_xyzw": desired_quat.tolist(),
            "coupled_grasp_locked": coupled_pick_waypoint,
            "selected_approach_axis_base": wp.get(
                "selected_approach_axis_base"
            ),
            "desired_fingertip_center_base_m": (
                desired_fingertip.tolist()
                if desired_fingertip is not None
                else None
            ),
            "ball_center_base_m": (
                ball_center.tolist() if coupled_pick_waypoint else None
            ),
            "position_tolerance_m": frame_position_tolerance_m,
            "max_corrections": frame_max_corrections,
            "force_accept_after_max_corrections": (
                first_round_force_accept_after_corrections is not None
            ),
        }
    )

    command_joints = nominal_joints
    compensated_target = desired_position.copy()
    a0_pipeline_used = "nominal_joints"
    if use_cartesian_online_ik:
        frame_start_pos, _frame_start_quat = physical_ee_pose()
        frame_start_joints = r1.get_arm_joints()
        cartesian_delta_to_desired = desired_position - frame_start_pos
        if active_round_index == 1 and not use_llm_absolute_20_policy:
            if prev_wp is None:
                raise RuntimeError(
                    f"Round-1 plan-delta replay requires previous waypoint for frame {frame_index}"
                )
            previous_planned_position = planned_position_for_active_round(
                frame_index - 1, prev_wp
            )
            planned_frame_delta = desired_position - previous_planned_position
            commanded_frame_delta = capped_for_active_policy(
                planned_frame_delta, frame_cartesian_step_cap_m
            )
            compensated_target = frame_start_pos + commanded_frame_delta
            frame_step_source = "planned_inter_frame_delta"
        else:
            previous_planned_position = None
            planned_frame_delta = None
            commanded_frame_delta = capped_for_active_policy(
                cartesian_delta_to_desired, frame_cartesian_step_cap_m
            )
            compensated_target = frame_start_pos + commanded_frame_delta
            frame_step_source = "next_plan_target_minus_current_actual"
        try:
            command_joints, fk_diag = get_cartesian_pipeline_candidate(
                compensated_target,
                frame_index=frame_index,
                place_policy=is_place_waypoint(frame_index, wp),
                reference_joints=frame_start_joints,
                start_joints=frame_start_joints,
                max_incremental_joint_delta_rad=candidate_joint_delta_limit(
                    frame_index, wp
                ),
                validate_fk=True,
                locked_quaternion_xyzw=(
                    desired_quat if coupled_pick_waypoint else None
                ),
                fk_tolerance_m=(
                    coupled_grasp_ik_fk_tol_m
                    if coupled_pick_waypoint
                    else None
                ),
                topdown_only=grasp_waypoint and not coupled_pick_waypoint,
            )
            a0_pipeline_used = "cartesian_absolute_step"
            write(
                {
                    "event": "frame_a0_cartesian_absolute_step_target",
                    "frame_index": frame_index,
                    "id": wid,
                    "frame_start_position_base_m": frame_start_pos.tolist(),
                    "desired_position_base_m": desired_position.tolist(),
                    "cartesian_delta_to_desired_m": cartesian_delta_to_desired.tolist(),
                    "frame_step_source": frame_step_source,
                    "previous_planned_position_base_m": (
                        previous_planned_position.tolist()
                        if previous_planned_position is not None
                        else None
                    ),
                    "planned_frame_delta_m": (
                        planned_frame_delta.tolist()
                        if planned_frame_delta is not None
                        else None
                    ),
                    "commanded_frame_delta_m": commanded_frame_delta.tolist(),
                    "max_step_m": frame_cartesian_step_cap_m,
                    "compensated_target_position_base_m": compensated_target.tolist(),
                    "online_ik_orientation_mode": (
                        "coupled_grasp_locked"
                        if coupled_pick_waypoint
                        else "strict_topdown_yaw_search"
                        if grasp_waypoint
                        else "topdown_yaw_tilt_search_deterministic_seed"
                        if place_waypoint
                        else "topdown_yaw_tilt_search"
                    ),
                    "candidate_fk_validation": fk_diag,
                }
            )
        except Exception as exc:
            last_waypoint_failure = {
                "category": "fk_gate" if is_fk_gate_failure(exc) else "ik_other",
                "frame_index": frame_index,
                "id": wid,
                "error": repr(exc),
            }
            write(
                {
                    "event": "frame_a0_cartesian_absolute_ik_error",
                    "frame_index": frame_index,
                    "id": wid,
                    "compensated_target_position_base_m": compensated_target.tolist(),
                    "error": repr(exc),
                }
            )
            return False

    # Attempt 0 uses the selected frame-entry pipeline. Attempts 1..frame_max_corrections
    # are immediate corrections at the same frame, computed from physical XYZ feedback.
    for attempt in range(frame_max_corrections + 1):
        (
            result,
            actual_pos,
            actual_quat,
            before_pos,
            before_quat,
            before_joints,
            actual_joints,
        ) = command_frame(frame_index, wp, command_joints, attempt)

        pos_error = desired_position - actual_pos
        pos_error_m = float(np.linalg.norm(pos_error))
        ori_error_rad = quat_error_rad(desired_quat, actual_quat)
        actual_fingertip = (
            fingertip_center_from_tcp(actual_pos, actual_quat)
            if coupled_pick_waypoint
            else None
        )
        fingertip_error = (
            desired_fingertip - actual_fingertip
            if desired_fingertip is not None and actual_fingertip is not None
            else None
        )
        fingertip_error_m = (
            float(np.linalg.norm(fingertip_error))
            if fingertip_error is not None
            else None
        )

        write(
            {
                "event": "frame_attempt",
                "frame_index": frame_index,
                "id": wid,
                "attempt": attempt,
                "compensated_target_position_base_m": compensated_target.tolist(),
                "a0_pipeline": a0_pipeline_used if attempt == 0 else "closed_loop_correction",
                "before_position_base_m": before_pos.tolist(),
                "actual_position_base_m": actual_pos.tolist(),
                "desired_position_base_m": desired_position.tolist(),
                "position_error_m": pos_error_m,
                "position_error_xyz_m": pos_error.tolist(),
                "orientation_error_rad": ori_error_rad,
                "desired_fingertip_center_base_m": (
                    desired_fingertip.tolist()
                    if desired_fingertip is not None
                    else None
                ),
                "actual_fingertip_center_base_m": (
                    actual_fingertip.tolist()
                    if actual_fingertip is not None
                    else None
                ),
                "fingertip_error_m": fingertip_error_m,
                "fingertip_error_xyz_m": (
                    fingertip_error.tolist()
                    if fingertip_error is not None
                    else None
                ),
                "fingertip_to_ball_center_error_m": (
                    float(np.linalg.norm(actual_fingertip - ball_center))
                    if actual_fingertip is not None and grasp_waypoint
                    else None
                ),
                "before_joints": before_joints.tolist(),
                "actual_joints": actual_joints.tolist(),
                "command_result": result,
            }
        )

        if not bool(result["joint_arrival"]["reached"]):
            write(
                {
                    "event": "frame_joint_arrival_abort",
                    "frame_index": frame_index,
                    "id": wid,
                    "attempt": attempt,
                    "max_joint_error_rad": float(
                        result["max_feedback_error_rad"]
                    ),
                    "joint_arrival_tolerance_rad": joint_arrival_tol_rad,
                    "joint_arrival_history": result["joint_arrival"]["history"],
                    "note": (
                        "IK joints did not converge after bounded joint-space "
                        "retries; Cartesian correction was not attempted."
                    ),
                }
            )
            return False

        reached_target = (
            fingertip_error_m <= coupled_grasp_fingertip_tol_m
            if grasp_waypoint
            and coupled_pick_waypoint
            and fingertip_error_m is not None
            else pos_error_m <= frame_position_tolerance_m
        )
        if reached_target:
            write({"event": "frame_reached", "frame_index": frame_index, "id": wid, "attempt": attempt})
            return True

        if frame_a0_pipeline == "cartesian_absolute" and place_waypoint:
            box_xy_error_m, box_z_margin_m = box_xy_error_and_z_margin(actual_pos)
            place_xy_ok = (
                math.isfinite(box_xy_error_m)
                and box_xy_error_m < f6_box_xy_tol_m
            )
            place_z_ok = (
                math.isfinite(box_z_margin_m)
                and box_z_margin_m
                >= box_above_z_margin_from_center_m
            )
            place_accept = (
                place_xy_ok and place_z_ok
                if variable_frame_plan
                else place_xy_ok
            )
            if place_accept:
                write(
                    {
                        "event": "frame_f6_xy_accepted",
                        "frame_index": frame_index,
                        "id": wid,
                        "attempt": attempt,
                        "position_error_m": pos_error_m,
                        "box_xy_error_m": box_xy_error_m,
                        "box_z_margin_m": box_z_margin_m,
                        "f6_box_xy_tol_m": f6_box_xy_tol_m,
                        "box_above_z_margin_from_center_m": (
                            box_above_z_margin_from_center_m
                        ),
                        "z_margin_required": variable_frame_plan,
                        "note": (
                            "Variable-frame placement requires both finite XY "
                            "and sufficient height above the box; legacy F6 "
                            "retains its historical XY-only completion rule."
                        ),
                    }
                )
                return True
        elif str(wid).startswith("place_"):
            above_box, box_xy_error_m, box_z_margin_m = is_above_box(actual_pos)
            if above_box:
                write(
                    {
                        "event": "frame_above_box_accepted",
                        "frame_index": frame_index,
                        "id": wid,
                        "attempt": attempt,
                        "position_error_m": pos_error_m,
                        "box_xy_error_m": box_xy_error_m,
                        "box_z_margin_m": box_z_margin_m,
                        "box_above_xy_tol_m": box_above_xy_tol_m,
                        "box_above_z_margin_from_center_m": box_above_z_margin_from_center_m,
                        "note": "Place frame accepted because EE is above the box; strict target error is ignored for release.",
                    }
                )
                return True

        if skip_xyz_correction:
            measured_motion = float(result.get("measured_motion_rad", 0.0))
            if bool(result.get("success", False)) or measured_motion >= min_motion_rad:
                write(
                    {
                        "event": "frame_joint_interpolation_done",
                        "frame_index": frame_index,
                        "id": wid,
                        "attempt": attempt,
                        "measured_motion_rad": measured_motion,
                        "note": "Joint-interpolated transition frame; XYZ correction intentionally skipped.",
                    }
                )
                return True
            write(
                {
                    "event": "frame_joint_interpolation_no_motion",
                    "frame_index": frame_index,
                    "id": wid,
                    "attempt": attempt,
                    "measured_motion_rad": measured_motion,
                }
            )
            return False

        if (
            attempt > 0
            and not bool(result.get("success", False))
            and float(result.get("measured_motion_rad", 0.0)) < min_motion_rad
        ):
            write(
                {
                    "event": "frame_no_motion_abort",
                    "frame_index": frame_index,
                    "id": wid,
                    "attempt": attempt,
                    "position_error_m": pos_error_m,
                    "measured_motion_rad": float(result.get("measured_motion_rad", 0.0)),
                    "max_feedback_error_rad": float(result.get("max_feedback_error_rad", 0.0)),
                    "note": "Command produced no measurable joint motion; stopping corrections to avoid compensation runaway.",
                }
            )
            return False

        if attempt >= frame_max_corrections:
            if first_round_force_accept_after_corrections is not None:
                write(
                    {
                        "event": "frame_force_accepted_after_correction_limit",
                        "frame_index": frame_index,
                        "id": wid,
                        "attempt": attempt,
                        "correction_count": attempt,
                        "final_position_error_m": pos_error_m,
                        "position_error_xyz_m": pos_error.tolist(),
                        "note": "Round-1 frame-specific policy: proceed after the configured correction count regardless of residual position error; execution safety gates remain active.",
                    }
                )
                return True
            write(
                {
                    "event": "frame_not_reached",
                    "frame_index": frame_index,
                    "id": wid,
                    "final_position_error_m": pos_error_m,
                }
            )
            return False

        # Local incremental XYZ correction: move from the physical EE pose
        # toward the desired pose by at most one capped Cartesian step.
        correction_cap_m = (
            correction_step_cap_m
            if correction_step_cap_m is not None
            else frame_cartesian_step_cap_m
        )
        compensation = capped_for_active_policy(
            correction_gain * pos_error, correction_cap_m
        )
        compensated_target = actual_pos + compensation
        try:
            if use_cartesian_online_ik:
                command_joints, _fk_diag = get_cartesian_pipeline_candidate(
                    compensated_target,
                    frame_index=frame_index,
                    place_policy=is_place_waypoint(frame_index, wp),
                    reference_joints=actual_joints,
                    start_joints=actual_joints,
                    max_incremental_joint_delta_rad=candidate_joint_delta_limit(
                        frame_index, wp
                    ),
                    validate_fk=True,
                    locked_quaternion_xyzw=(
                        desired_quat if coupled_pick_waypoint else None
                    ),
                    fk_tolerance_m=(
                        coupled_grasp_ik_fk_tol_m
                        if coupled_pick_waypoint
                        else None
                    ),
                    topdown_only=grasp_waypoint and not coupled_pick_waypoint,
                )
                write(
                    {
                        "event": "correction_ik_candidate_selected",
                        "frame_index": frame_index,
                        "id": wid,
                        "attempt": attempt,
                        "reference_joints": actual_joints.tolist(),
                        "position_error_xyz_m": pos_error.tolist(),
                        "correction_gain": correction_gain,
                        "correction_step_cap_m": correction_cap_m,
                        "correction_delta_xyz_m": compensation.tolist(),
                        "correction_delta_norm_m": float(
                            np.linalg.norm(compensation)
                        ),
                        "compensated_target_position_base_m": compensated_target.tolist(),
                        "command_joints": command_joints.tolist(),
                        "candidate_joint_delta_limit_rad": candidate_joint_delta_limit(
                            frame_index, wp
                        ),
                        "fk_validation": _fk_diag,
                    }
                )
            else:
                command_joints = r1.get_relaxed_ik_candidate(
                    compensated_target,
                    desired_quat,
                    timeout_s=3.0,
                    reference_joints=actual_joints,
                    start_joints=actual_joints,
                )
        except Exception as exc:
            last_waypoint_failure = {
                "category": "fk_gate" if is_fk_gate_failure(exc) else "ik_other",
                "frame_index": frame_index,
                "id": wid,
                "error": repr(exc),
            }
            write(
                {
                    "event": "correction_ik_failed",
                    "frame_index": frame_index,
                    "id": wid,
                    "attempt": attempt,
                    "compensated_target_position_base_m": compensated_target.tolist(),
                    "error": repr(exc),
                }
            )
            return False

    return False


def execute_runtime_subdivisions(
    frame_index: int, wp: dict, prev_wp: dict | None
) -> bool:
    if not variable_frame_plan or frame_index < 2:
        return True
    target = np.asarray(
        wp["target_ee_position_base_m"], dtype=float
    ).reshape(3)
    subdivision_index = 0
    while True:
        current, _ = physical_ee_pose()
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance <= llm_max_cartesian_frame_step_m + 1e-9:
            return True
        subdivision_index += 1
        subtarget = current + delta * (
            llm_max_cartesian_frame_step_m / distance
        )
        runtime_wp = copy.deepcopy(wp)
        runtime_wp["id"] = (
            f"runtime_interp_{frame_index:02d}_"
            f"{subdivision_index:02d}_toward_{wp['id']}"
        )
        runtime_wp["label"] = f"Runtime interpolation toward {wp['id']}"
        runtime_wp["target_ee_position_base_m"] = subtarget.tolist()
        runtime_wp["semantic_role"] = "transit"
        runtime_wp["semantic_action"] = "move"
        runtime_wp["target_source"] = "runtime_interpolation"
        runtime_wp["is_interpolated"] = True
        write(
            {
                "event": "runtime_cartesian_subdivision_inserted",
                "frame_index": frame_index,
                "id": runtime_wp["id"],
                "parent_waypoint_id": wp["id"],
                "current_position_base_m": current.tolist(),
                "parent_target_position_base_m": target.tolist(),
                "subtarget_position_base_m": subtarget.tolist(),
                "original_distance_m": distance,
                "max_step_m": llm_max_cartesian_frame_step_m,
            }
        )
        if not execute_waypoint(frame_index, runtime_wp, prev_wp):
            return False


initial_pos, initial_quat = physical_ee_pose()
initial_joints = r1.get_arm_joints()
write(
    {
        "event": "start",
        "mode": "keyframes_xyz_corrected_with_gripper",
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "llm_generated_policy_path": (
            generated_policy_path_raw or None
        ),
        "llm_generated_policy_validated": (
            generated_policy_operations is not None
        ),
        "llm_generated_policy_sha256": generated_policy_sha256,
        "log": str(log_path),
        "pos_tol_m": pos_tol_m,
        "max_corrections": max_corrections,
        "correction_gain": correction_gain,
        "correction_step_cap_m": correction_step_cap_m,
        "max_compensation_norm_m": max_compensation_norm_m,
        "min_motion_rad": min_motion_rad,
        "command_duration_s": command_duration_s,
        "joint_arrival_tolerance_rad": joint_arrival_tol_rad,
        "joint_arrival_max_retries": joint_arrival_max_retries,
        "tcp_sample_policy": "after_fresh_stable_joint_feedback",
        "kinematic_execution_decomposition": (
            "ik_command_fk_to_actual_joint_fk_to_vendor_tcp"
        ),
        "skip_xyz_waypoint_step_rad": skip_xyz_waypoint_step_rad,
        "return_home_waypoint_step_rad": return_home_waypoint_step_rad,
        "box_above_xy_tol_m": box_above_xy_tol_m,
        "box_above_z_margin_from_center_m": box_above_z_margin_from_center_m,
        "f6_box_xy_tol_m": f6_box_xy_tol_m,
        "f6_pos_tol_m": f6_pos_tol_m,
        "frame_a0_pipeline": frame_a0_pipeline,
        "round_count": round_count,
        "single_round_policy": single_round_policy,
        "variable_frame_plan": variable_frame_plan,
        "coupled_grasp_enabled": coupled_grasp_enabled,
        "coupled_grasp_tcp_to_fingertip_m": (
            coupled_grasp_tcp_to_fingertip_m
        ),
        "coupled_grasp_tcp_to_fingertip_source": (
            fourth_pipeline_metadata.get(
                "tcp_to_fingertip_source", "urdf_mesh_estimate"
            )
        ),
        "coupled_grasp_ik_fk_tolerance_m": coupled_grasp_ik_fk_tol_m,
        "coupled_grasp_orientation_tolerance_rad": (
            coupled_grasp_orientation_tol_rad
        ),
        "coupled_grasp_fingertip_tolerance_m": (
            coupled_grasp_fingertip_tol_m
        ),
        "llm_grasp_preference": llm_grasp_preference,
        "coupled_grasp_search_policy": (
            "llm_neighborhood_then_global_fallback"
        ),
        "llm_max_cartesian_frame_step_m": llm_max_cartesian_frame_step_m,
        "llm_max_incremental_joint_delta_rad": (
            llm_max_incremental_joint_delta_rad
        ),
        "variable_frame_runtime_subdivision": variable_frame_plan,
        "round_frame_ranges": (
            {"1": "F0-F7"}
            if round_count == 1
            else {"1": "F0-F7", "2": "F1-F7"}
        ),
        "round_correction_policy": (
            {
                "1": {
                    "F0_F1": "fixed_plan_targets",
                    "F2_F6": f"llm_absolute_targets_up_to_{max_corrections}_corrections",
                    "force_accept": False,
                }
            }
            if use_llm_absolute_20_policy
            else {
                "1": {
                    "force_accept_after_corrections_by_frame": {
                        "2": 0,
                        "3": 0,
                        "4": 0,
                        "5": 0,
                        "6": 0,
                    },
                    "preserved_frame_indices": [0, 1, 7],
                },
                "2": "unchanged",
            }
        ),
        "round_cartesian_step_caps_m": (
            {"1_F2-F6": None}
            if use_llm_absolute_20_policy
            else {"1_F2-F6": 0.35, "2": max_compensation_norm_m}
        ),
        "round_frame_step_policy": (
            {"1_F2-F6": "current_actual_toward_llm_absolute_target"}
            if use_llm_absolute_20_policy
            else {
                "1_F2-F6": "current_actual_plus_planned_inter_frame_delta",
                "2": "current_actual_toward_next_planned_absolute_target",
            }
        ),
        "round_target_overrides": {"1": "unchanged", "2": "unchanged"},
        "round1_fk_retry_policy": {
            "max_automatic_retries": round1_max_fk_retries,
            "retry_start_frame": 2,
            "return_target": "first_measured_F1",
            "post_grasp_failure": "return_F1_open_gripper_wait_manual_reset",
            "manual_reset_signal_path": str(round1_manual_reset_signal_path),
        },
        "cartesian_ik_fk_tol_m": cartesian_ik_fk_tol_m,
        "cartesian_early_max_joint_delta_rad": cartesian_early_max_joint_delta_rad,
        "cartesian_f6_max_joint_delta_rad": cartesian_f6_max_joint_delta_rad,
        "candidate_local_fk_frame_indices": "all_cartesian_frames",
        "deterministic_seed_continuity_fix": (
            {"semantic_roles": ["above_box"]}
            if variable_frame_plan
            else {"frame_indices": [6]}
        ),
        "orientation_policy": {
            "semantic_grasp": "strict_topdown_yaw_search_tilt_0_deg",
            "other_cartesian_frames": (
                "topdown_yaw_search_with_0_to_plus_minus_60_deg_tilt"
            ),
            "tilt_step_deg": cartesian_topdown_tilt_step_deg,
        },
        "cartesian_joint_delta_limits_rad": {
            "2": cartesian_early_max_joint_delta_rad,
            "3": cartesian_early_max_joint_delta_rad,
            "6": cartesian_f6_max_joint_delta_rad,
        },
        "box_center_base_m": box_center.tolist(),
        "initial_physical_position_base_m": initial_pos.tolist(),
        "initial_physical_quaternion_xyzw": initial_quat.tolist(),
        "initial_arm_joints": initial_joints.tolist(),
        "note": "Frame-level command plus local incremental XYZ corrections; skip_xyz_correction frames use segmented joint waypoints and no XYZ correction.",
    }
)
if generated_policy_operations is not None:
    consume_generated_policy_operation("bind_plan", sha256=plan_sha256)
    write(
        {
            "event": "llm_generated_policy_bound",
            "plan_sha256": plan_sha256,
            "operation_count": len(generated_policy_operations),
        }
    )

first_f1_position = None
first_f1_quaternion = None
first_f1_joints = None
coupled_grasp_resolved = False


def return_to_first_f1_after_release(frame_index: int) -> None:
    if (
        first_f1_position is None
        or first_f1_quaternion is None
        or first_f1_joints is None
    ):
        raise RuntimeError("F1 measured pose was not captured")
    before_home_pos, _before_home_quat = physical_ee_pose()
    before_home_joints = r1.get_arm_joints()
    return_frame_index = len(waypoints)
    write(
        {
            "event": "return_to_first_f1_after_release_begin",
            "after_frame_index": frame_index,
            "return_frame_index": return_frame_index,
            "target_f1_position_base_m": first_f1_position.tolist(),
            "target_f1_quaternion_xyzw": first_f1_quaternion.tolist(),
            "target_f1_joints": first_f1_joints.tolist(),
            "mode": "measured_f1_joint_space",
            "waypoint_step_rad": return_home_waypoint_step_rad,
            "note": "Return follows the first-round measured F1 joint target; Cartesian IK and XYZ micro-corrections are not used.",
        }
    )
    measured_return_joints = r1.move_joints_safe(
        first_f1_joints,
        max_step=return_home_waypoint_step_rad,
        duration_s=command_duration_s,
    )
    time.sleep(0.35)
    after_home_pos, after_home_quat = physical_ee_pose()
    after_home_joints = r1.get_arm_joints()
    joint_error_rad = float(np.max(np.abs(after_home_joints - first_f1_joints)))
    position_error_m = float(np.linalg.norm(after_home_pos - first_f1_position))
    orientation_error_rad = quat_error_rad(first_f1_quaternion, after_home_quat)
    write(
        {
            "event": "return_home_after_release",
            "after_frame_index": frame_index,
            "return_frame_index": return_frame_index,
            "return_target_id": "return_to_first_f1_after_release",
            "mode": "measured_f1_joint_space",
            "before_position_base_m": before_home_pos.tolist(),
            "after_position_base_m": after_home_pos.tolist(),
            "before_joints": before_home_joints.tolist(),
            "target_f1_position_base_m": first_f1_position.tolist(),
            "target_f1_quaternion_xyzw": first_f1_quaternion.tolist(),
            "target_f1_joints": first_f1_joints.tolist(),
            "after_joints": after_home_joints.tolist(),
            "measured_return_joints": measured_return_joints.tolist(),
            "max_joint_error_rad": joint_error_rad,
            "position_error_m": position_error_m,
            "orientation_error_rad": orientation_error_rad,
            "joint_tolerance_rad": r1.arm_joint_tolerance_rad,
            "position_tolerance_m": return_f1_position_tolerance_m,
            "orientation_tolerance_rad": return_f1_orientation_tolerance_rad,
        }
    )
    if position_error_m > return_f1_position_tolerance_m:
        raise RuntimeError(
            "F7 return-to-F1 TCP position validation failed: "
            f"error={position_error_m:.6f}m, "
            f"tolerance={return_f1_position_tolerance_m:.6f}m"
        )
    if orientation_error_rad > return_f1_orientation_tolerance_rad:
        raise RuntimeError(
            "F7 return-to-F1 TCP orientation validation failed: "
            f"error={orientation_error_rad:.6f}rad, "
            f"tolerance={return_f1_orientation_tolerance_rad:.6f}rad"
        )
    write(
        {
            "event": "frame_reached",
            "frame_index": return_frame_index,
            "id": "return_to_first_f1_after_release",
            "max_joint_error_rad": joint_error_rad,
            "position_error_m": position_error_m,
            "orientation_error_rad": orientation_error_rad,
        }
    )


for round_index in range(1, round_count + 1):
    active_round_index = round_index
    start_frame_index = 0 if round_index == 1 else 1
    round1_fk_retry_count = 0
    write(
        {
            "event": "round_begin",
            "start_frame_index": start_frame_index,
            "end_frame_index": len(waypoints),
            "note": (
                "Round 1 executes F0-F7."
                if round_count == 1
                else "Round 1 executes F0-F7; round 2 repeats F1-F7."
            ),
        }
    )
    attempt_start_frame_index = start_frame_index
    while True:
        failed_frame = None
        gripper_closed = False
        prev_wp = (
            None
            if attempt_start_frame_index == 0
            else waypoints[attempt_start_frame_index - 1]
        )
        for frame_index in range(attempt_start_frame_index, len(waypoints)):
            wp = waypoints[frame_index]
            consume_generated_policy_operation(
                "execute_frame", frame_id=str(wp["id"])
            )
            if not execute_runtime_subdivisions(frame_index, wp, prev_wp):
                failed_frame = (frame_index, wp)
                break
            ok = execute_waypoint(frame_index, wp, prev_wp)
            if not ok:
                failed_frame = (frame_index, wp)
                break

            if frame_index == 1 and first_f1_joints is None:
                time.sleep(0.35)
                first_f1_position, first_f1_quaternion = physical_ee_pose()
                first_f1_joints = r1.get_arm_joints()
                write(
                    {
                        "event": "first_f1_pose_captured",
                        "frame_index": frame_index,
                        "id": wp["id"],
                        "physical_position_base_m": first_f1_position.tolist(),
                        "physical_quaternion_xyzw": first_f1_quaternion.tolist(),
                        "arm_joints": first_f1_joints.tolist(),
                        "note": "Both F7 returns use this first-round measured F1 pose.",
                    }
                )
                if coupled_grasp_enabled and not coupled_grasp_resolved:
                    try:
                        resolve_coupled_grasp_chain(
                            first_f1_position, first_f1_joints
                        )
                        coupled_grasp_resolved = True
                    except Exception as exc:
                        write(
                            {
                                "event": "coupled_grasp_chain_resolution_failed",
                                "frame_index": frame_index,
                                "id": wp["id"],
                                "error": repr(exc),
                                "note": (
                                    "No robot motion was commanded after F1 "
                                    "because the full coupled pick chain failed."
                                ),
                            }
                        )
                        raise

            if (
                waypoint_role(wp) == "grasp"
                and not bool(wp.get("skip_xyz_correction", False))
            ):
                if coupled_grasp_enabled and not bool(
                    wp.get("coupled_grasp_locked", False)
                ):
                    raise RuntimeError(
                        "variable-frame grasp cannot close without a locked "
                        "coupled-grasp solution"
                    )
                consume_generated_policy_operation("close_gripper")
                if bool(wp.get("coupled_grasp_locked", False)):
                    close_tcp, close_quat = physical_ee_pose()
                    close_tip = fingertip_center_from_tcp(
                        close_tcp, close_quat
                    )
                    close_tip_error_m = float(
                        np.linalg.norm(close_tip - ball_center)
                    )
                    close_gate_passed = (
                        close_tip_error_m
                        <= coupled_grasp_fingertip_tol_m
                    )
                    write(
                        {
                            "event": "gripper_close_safety_check",
                            "after_frame_index": frame_index,
                            "tcp_position_base_m": close_tcp.tolist(),
                            "tcp_quaternion_xyzw": close_quat.tolist(),
                            "actual_fingertip_center_base_m": (
                                close_tip.tolist()
                            ),
                            "ball_center_base_m": ball_center.tolist(),
                            "fingertip_to_ball_center_error_m": (
                                close_tip_error_m
                            ),
                            "fingertip_tolerance_m": (
                                coupled_grasp_fingertip_tol_m
                            ),
                            "passed": close_gate_passed,
                        }
                    )
                    if not close_gate_passed:
                        raise RuntimeError(
                            "gripper-close safety gate rejected fingertip "
                            f"error {close_tip_error_m:.6f}m > "
                            f"{coupled_grasp_fingertip_tol_m:.6f}m"
                        )
                try:
                    feedback = r1.command_gripper(
                        float(r1.gripper_close_position), duration_s=1.2
                    )
                    gripper_closed = True
                    write(
                        {
                            "event": "gripper_close",
                            "after_frame_index": frame_index,
                            "feedback": float(feedback),
                        }
                    )
                except Exception as exc:
                    write(
                        {
                            "event": "gripper_close_error",
                            "after_frame_index": frame_index,
                            "error": repr(exc),
                        }
                    )
                    raise
                time.sleep(0.5)

            if waypoint_role(wp) == "above_box":
                consume_generated_policy_operation("open_gripper")
                try:
                    feedback = r1.command_gripper(
                        float(r1.gripper_open_position), duration_s=1.2
                    )
                    gripper_closed = False
                    write(
                        {
                            "event": "gripper_open_release",
                            "after_frame_index": frame_index,
                            "feedback": float(feedback),
                        }
                    )
                except Exception as exc:
                    write(
                        {
                            "event": "gripper_open_release_error",
                            "after_frame_index": frame_index,
                            "error": repr(exc),
                        }
                    )
                    raise
                time.sleep(0.5)
                try:
                    consume_generated_policy_operation("return_to_f1")
                    return_to_first_f1_after_release(frame_index)
                except Exception as exc:
                    write(
                        {
                            "event": "return_home_after_release_error",
                            "after_frame_index": frame_index,
                            "error": repr(exc),
                        }
                    )
                    raise

            prev_wp = wp

        if failed_frame is None:
            break

        failed_frame_index, failed_wp = failed_frame
        if (
            use_llm_absolute_20_policy
            or round_index != 1
            or last_waypoint_failure is None
            or last_waypoint_failure.get("category") != "fk_gate"
        ):
            raise SystemExit(
                f"failed round {round_index} frame {failed_frame_index} "
                f"{failed_wp['id']}"
            )

        post_grasp_failure = gripper_closed
        write(
            {
                "event": "round1_fk_failure_rollback_begin",
                "failed_frame_index": failed_frame_index,
                "failed_id": failed_wp["id"],
                "retry_count": round1_fk_retry_count,
                "max_retries": round1_max_fk_retries,
                "post_grasp_failure": post_grasp_failure,
                "failure": last_waypoint_failure,
            }
        )
        return_to_first_f1_after_release(failed_frame_index)
        if post_grasp_failure:
            feedback = r1.command_gripper(
                float(r1.gripper_open_position), duration_s=1.2
            )
            write(
                {
                    "event": "round1_fk_failure_gripper_open_at_f1",
                    "failed_frame_index": failed_frame_index,
                    "feedback": float(feedback),
                }
            )

        if round1_fk_retry_count >= round1_max_fk_retries:
            write(
                {
                    "event": "round1_fk_retries_exhausted",
                    "failed_frame_index": failed_frame_index,
                    "failed_id": failed_wp["id"],
                    "retry_count": round1_fk_retry_count,
                }
            )
            raise SystemExit(
                f"round 1 FK retries exhausted at frame {failed_frame_index} "
                f"{failed_wp['id']}"
            )

        round1_fk_retry_count += 1
        if post_grasp_failure:
            round1_manual_reset_signal_path.unlink(missing_ok=True)
            write(
                {
                    "event": "round1_manual_reset_required",
                    "failed_frame_index": failed_frame_index,
                    "retry_count": round1_fk_retry_count,
                    "signal_path": str(round1_manual_reset_signal_path),
                    "note": "Reset the ball, then create the signal file to continue.",
                }
            )
            while not round1_manual_reset_signal_path.exists():
                time.sleep(0.5)
            round1_manual_reset_signal_path.unlink(missing_ok=True)
            write(
                {
                    "event": "round1_manual_reset_confirmed",
                    "retry_count": round1_fk_retry_count,
                }
            )

        attempt_start_frame_index = 2
        write(
            {
                "event": "round1_fk_retry_begin",
                "retry_count": round1_fk_retry_count,
                "start_frame_index": attempt_start_frame_index,
            }
        )

    write(
        {
            "event": "round_finished",
            "start_frame_index": start_frame_index,
            "end_frame_index": len(waypoints),
            "round1_fk_retry_count": (
                round1_fk_retry_count if round_index == 1 else None
            ),
        }
    )

active_round_index = None
if (
    generated_policy_operations is not None
    and generated_policy_cursor != len(generated_policy_operations)
):
    raise RuntimeError(
        "generated policy did not complete: "
        f"consumed={generated_policy_cursor}, "
        f"total={len(generated_policy_operations)}"
    )

final_pos, final_quat = physical_ee_pose()
write(
    {
        "event": "finished",
        "final_physical_position_base_m": final_pos.tolist(),
        "final_physical_quaternion_xyzw": final_quat.tolist(),
        "final_arm_joints": r1.get_arm_joints().tolist(),
        "log": str(log_path),
    }
)
write({"event": "ros_cleanup_begin"})
r1.close()
atexit.unregister(close_r1_at_exit)
write({"event": "ros_cleanup_finished"})
PY
