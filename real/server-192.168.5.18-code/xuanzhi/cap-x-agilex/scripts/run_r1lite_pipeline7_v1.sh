#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Pipeline 7 V1:
# - original saved grasp target
# - legacy replay refinement: unmonitored initial 10 mm descent, then rebaseline
# - first solvable 5..10 mm IK, one-sample effort contact, legacy close fallback
# - after lift, keep right TCP near/above table + 0.12 m until center release
# - joint-arrival hard gate disabled; soft arrival and other safety gates remain
# - release and return to the certified session Home
export PLAN_PATH="${PLAN_PATH:-outputs/r1lite_dual_arm_towel_fold/towel_fold_plan_selected_pipeline7_v1_original_grasp_z12.json}"
export CODE_PATH="${CODE_PATH:-outputs/r1lite_dual_arm_towel_fold/towel_fold_llm_program_pipeline7_v1_original_grasp_z12.py}"
export BUNDLE_PATH="${BUNDLE_PATH:-outputs/r1lite_dual_arm_towel_fold/sequential_execution_bundle_pipeline7_v1_original_grasp_z12.json}"
export COLLISION_AUDIT_PATH="${COLLISION_AUDIT_PATH:-outputs/r1lite_dual_arm_towel_fold/towel_collision_audit_pipeline7_v1_original_grasp_z12.json}"
export CERTIFICATE_PATH="${CERTIFICATE_PATH:-outputs/r1lite_dual_arm_towel_fold/towel_capability_certificate_pipeline7_v1_original_grasp_z12.json}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_pipeline7_v1}"
export PIPELINE_LABEL="pipeline7-v1"
export EXECUTE_ENV_HINT="PIPELINE7_V1_EXECUTE_ROBOT=1"
export R1LITE_ARRIVAL_HARD_THRESHOLD_RAD=inf
export R1LITE_ARRIVAL_SOFT_THRESHOLD_RAD=inf
export R1LITE_INITIAL_STATE_TOLERANCE_RAD=0.01
export PIPELINE7_V1_MODE=1
export R1LITE_P7_HORIZONTAL_PLACE_Z_TOLERANCE_M=0
export R1LITE_P7_DISABLE_PLACE_Z_GUARD=1

# Fixed post-grasp path through the center release point. From place-center1,
# two intermediate points lead to the release point. The user explicitly
# disabled planned and measured TCP-Z gates for these final three segments.
export R1LITE_PIPELINE7_POST_GRASP_PATH_JSON='{
  "lift": [
    [0.44301495908939814, 2.667646784903786, -1.900157499685318, 0.4525241771068577, -0.1286791876486889, 0.36221802111850515],
    [0.4892974124057795, 2.6217233561025926, -1.7248575841759517, 0.20252916371811452, -0.2466713045157995, 0.4360313066413589]
  ],
  "place_center": [
    [0.6148968106850017, 2.631095114338888, -1.8651694067652398, 0.34443010339387, -0.2938268640655741, 0.5406794430317099],
    [0.6513068576990704, 2.7136547157258737, -2.0165118825674146, 0.3513481742270604, -0.35561886225812683, 0.5519028293118337],
    [0.6844987523330401, 2.8149550508889853, -2.2127151236176776, 0.382767343871362, -0.41815348804629066, 0.55467808738134],
    [0.7143374495495701, 2.920765178348358, -2.4145810449513854, 0.40835844872013427, -0.48016049460383486, 0.5465433216258844]
  ]
}'

if [[ "${PIPELINE7_V1_EXECUTE_ROBOT:-0}" == "1" ]]; then
  export PIPELINE7_V2_EXECUTE_ROBOT=1
else
  export PIPELINE7_V2_EXECUTE_ROBOT=0
fi

exec "${REPO_ROOT}/scripts/run_r1lite_pipeline7_v2.sh"
