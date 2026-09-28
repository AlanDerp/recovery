#!/usr/bin/env bash
#SBATCH --job-name=capx-core-fg-pair
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/qshou/capx_paper_verifier_results/logs/core-fg-pair-%j.out
#SBATCH --error=/home/qshou/capx_paper_verifier_results/logs/core-fg-pair-%j.out

set -euo pipefail

submission_project=${PROJECT_DIR:-/home/qshou/capx_core_fg_clean}
project="$submission_project"
helper="$project/experiments/core_fg/runtime"
python_bin=${CAPX_PYTHON:-/project/peilab/quanxin/cap-x/.venv-main/bin/python}
baseline_rev=${CAPX_BASELINE_REV:-345af89}
qwen_key_file=${QWEN_KEY_FILE:-/home/qshou/RoboTwin_Sandbox/cap-x/.qwenkey_sg_tmp}
qwen_base_url=${QWEN_BASE_URL:-https://ws-vpxr6jfor4t48uyo.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1}
canonical_initial_replay_file=${CAPX_CANONICAL_INITIAL_REPLAY_FILE:-}
frozen_a0_replay_file=${CAPX_FROZEN_A0_REPLAY_FILE:-}
frozen_a0_reference=${CAPX_FROZEN_A0_REFERENCE:-}
skip_a1=${CAPX_SKIP_A1:-0}
frozen_a0_run_root=${CAPX_FROZEN_A0_RUN_ROOT:-}
frozen_a0_pair_validation=${CAPX_FROZEN_A0_PAIR_VALIDATION:-}
m3_semantic_reference=${CAPX_M3_SEMANTIC_REFERENCE:-}
treatment_full_replay_file=${CAPX_FGSRP_TREATMENT_FULL_REPLAY_FILE:-}
treatment_reference=${CAPX_FGSRP_TREATMENT_REFERENCE:-}
treatment_source_pair_validation=${CAPX_FGSRP_TREATMENT_SOURCE_PAIR_VALIDATION:-}
treatment_response_replay_file=${CAPX_TREATMENT_RESPONSE_REPLAY_FILE:-}
infra_file_list="$project/experiments/pgsvp_v9_verify/infra_files.txt"
infra_lock="$project/experiments/pgsvp_v9_verify/infra_lock.json"
sam3_source_root=${CAPX_SAM3_SOURCE_ROOT:-/scratch/peilab/qshou/capx_paper_verifier/repo/capx/third_party/sam3}
graspnet_source_root=${CAPX_CONTACT_GRASPNET_SOURCE_ROOT:-/scratch/peilab/qshou/capx_paper_verifier/repo/capx/third_party/contact_graspnet_pytorch}
expected_sam3_rev=6fe87d64a5beb9084923d7a9e002741178635b09
expected_graspnet_rev=da3dcfb2f53e43b186083ee4a9d1e232f73efc98
expected_sam3_checkpoint_sha=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
expected_graspnet_checkpoint_sha=39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762
expected_graspnet_config_sha=7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5
task=${TASK_NAME:?Set TASK_NAME=cube_stack or cube_restack}
trials=${TOTAL_TRIALS:-1}
service_smoke_only=${SERVICE_SMOKE_ONLY:-0}
pair_smoke=${CAPX_PAIR_SMOKE:-0}
pair_local_a1_replay=${CAPX_PAIR_LOCAL_A1_REPLAY:-0}
ideal_tk_v4_effect_alignment=${CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT:-0}
ideal_tk_v4_effect_alignment_mode=${CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE:-contradictions}
global_seed=${CAPX_GLOBAL_SEED:-20260718}
fixed_episode_seed=${CAPX_FIXED_EPISODE_SEED:-}
episode_seed_base=${CAPX_EPISODE_SEED_BASE:-}
if [[ -n "$fixed_episode_seed" && -n "$episode_seed_base" ]]; then
  echo "CAPX_FIXED_EPISODE_SEED and CAPX_EPISODE_SEED_BASE are mutually exclusive" >&2
  exit 2
fi
treatment=${CAPX_TREATMENT:-core_fg}
fgsrp_carrier=${CAPX_FGSRP_CARRIER:-m2_comments}
native_mode=M2
initial_action_query_seed_offset=0
replacement_treatment=0
submission_git_rev=${CAPX_SUBMISSION_GIT_REV:-not_applicable}
submission_runner_git_rev=${CAPX_SUBMISSION_RUNNER_GIT_REV:-not_applicable}
submission_runner_snapshot=${CAPX_SUBMISSION_RUNNER_SNAPSHOT:-${BASH_SOURCE[0]}}
submission_runner_sha=${CAPX_SUBMISSION_RUNNER_SHA256:-not_applicable}
case "$treatment" in
  core_fg)
    default_result_root=/home/qshou/capx_paper_verifier_results/results/core_fg_cube_v1_20260728
    treatment_label=Core-FG
    treatment_module=capx/verification/core_fg.py
    treatment_spec=docs/CORE_FG_SPEC.md
    ;;
  pgsvp_v9_verify)
    default_result_root=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_as_materialized_ptgt_v1_20260731
    treatment_label=PGSVP-v9-AS-Materialized-PTGT
    treatment_module=capx/verification/pgsvp_v9_verify.py
    treatment_spec=docs/PGSVP_V9_VERIFY_SPEC.md
    ;;
  envelope_replacement)
    default_result_root=/home/qshou/capx_paper_verifier_results/results/envelope_replacement_v1_20260731
    treatment_label=Envelope-Replacement-v1
    treatment_module=capx/verification/envelope_replacement.py
    treatment_spec=docs/ENVELOPE_REPLACEMENT_SPEC.md
    replacement_treatment=1
    ;;
  fgsrp)
    default_result_root=/home/qshou/capx_paper_verifier_results/results/fgsrp_v1_3
    treatment_label=FGSRP-v2-semantic-ir-alignment
    treatment_module=capx/verification/fgsrp.py
    treatment_spec=docs/FGSRP_IR_ALIGNMENT_SPEC.md
    replacement_treatment=1
    case "$fgsrp_carrier" in
      m2_comments)
        native_mode=M2
        initial_action_query_seed_offset=0
        ;;
      m3_visual_difference)
        native_mode=M3
        initial_action_query_seed_offset=1
        ;;
      *)
        echo "Unsupported CAPX_FGSRP_CARRIER=$fgsrp_carrier" >&2
        exit 2
        ;;
    esac
    ;;
  fgsrp_nl)
    default_result_root=/home/qshou/capx_paper_verifier_results/results/fgsrp_nl_v1
    treatment_label=FGSRP-NL-v1
    treatment_module=capx/verification/fgsrp_nl.py
    treatment_spec=docs/FGSRP_NL_SPEC.md
    replacement_treatment=1
    case "$fgsrp_carrier" in
      m2_comments)
        native_mode=M2
        initial_action_query_seed_offset=0
        ;;
      m3_visual_difference)
        native_mode=M3
        initial_action_query_seed_offset=1
        ;;
      *)
        echo "Unsupported CAPX_FGSRP_CARRIER=$fgsrp_carrier" >&2
        exit 2
        ;;
    esac
    ;;
  closed_gt_takeover)
    default_result_root=/home/qshou/capx_paper_verifier_results/results/closed_gt_takeover_v2_1
    treatment_label=Closed-GT-Takeover-v2.1
    treatment_module=capx/verification/closed_gt_takeover.py
    treatment_spec=docs/CLOSED_GT_TAKEOVER_SPEC.md
    replacement_treatment=1
    native_mode=M3
    initial_action_query_seed_offset=1
    fgsrp_carrier=m3_visual_difference
    ;;
  ideal_tk)
    default_result_root=/home/qshou/capx_paper_verifier_results/results/ideal_tk_v3_3_public_lossless
    treatment_label=Ideal-TK-v3.3-public-lossless
    treatment_module=capx/verification/ideal_tk.py
    treatment_spec=docs/IDEAL_TK_V3_SPEC.md
    replacement_treatment=1
    native_mode=M3
    initial_action_query_seed_offset=1
    fgsrp_carrier=m3_visual_difference
    ;;
  ideal_tk_v4)
    default_result_root=/home/qshou/capx_paper_verifier_results/results/ideal_tk_v4_canonical_delta
    treatment_label=Ideal-TK-v4-canonical-delta
    treatment_module=capx/verification/ideal_tk_v4.py
    treatment_spec=docs/IDEAL_TK_V4_SPEC.md
    replacement_treatment=1
    native_mode=M3
    initial_action_query_seed_offset=1
    fgsrp_carrier=m3_visual_difference
    ;;
  *)
    echo "Unsupported CAPX_TREATMENT=$treatment" >&2
    exit 2
    ;;
esac
result_root=${RESULT_ROOT:-$default_result_root}
pair_root="$result_root/$task/${trials}trial_pair_run-${SLURM_JOB_ID}"
a1_base="$pair_root/a1"
treatment_base="$pair_root/$treatment"
replay_file="$pair_root/a1_initial_request_hash_replay.json"
current_a1_replay_file="$pair_root/a1_current_request_hash_replay.json"
treatment_replay_file="$replay_file"
treatment_replay_mode=registration_live
infra_fingerprint="$pair_root/INFRA_FINGERPRINT.json"
canonical_initial_replay_sha=not_applicable

case "$task" in
  cube_stack|cube_restack) ;;
  *)
    echo "Unsupported TASK_NAME=$task" >&2
    exit 2
    ;;
esac
if [[ "$ideal_tk_v4_effect_alignment" != "0" && "$ideal_tk_v4_effect_alignment" != "1" ]]; then
  echo "CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT must be 0 or 1" >&2
  exit 2
fi
case "$ideal_tk_v4_effect_alignment_mode" in
  contradictions|first_failure|verified_prefix|release_support|grasp_loss_only|instance_bound_grasp_loss_only|instance_bound_grasp_outcome_only|instance_bound_transport_failure_only|instance_bound_off_table_lift_regression_only|instance_bound_grasp_loss_plus_off_table_lift_regression|post_release_support_only|compact_outcome|effect_ledger) ;;
  *)
    echo "Unsupported CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE=$ideal_tk_v4_effect_alignment_mode" >&2
    exit 2
    ;;
esac

# A smoke run exercises the current M2 pair locally from canonical pi0, then
# exports that A1 request trajectory for the treatment arm. Formal ablations
# reference the registered deterministic A1 result and execute treatment only;
# the baseline must never be resampled after the ablation has been chosen.
if [[
  "$pair_smoke" == "1"
  && "$replacement_treatment" == "1"
  && "$native_mode" == "M2"
]]; then
  pair_local_a1_replay=1
fi
if [[
  "$replacement_treatment" == "1"
  && "$native_mode" == "M2"
  && "$service_smoke_only" != "1"
  && "$pair_smoke" != "1"
  && "$trials" == "10"
]]; then
  skip_a1=1
  case "$task" in
    cube_restack)
      frozen_a0_run_root=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_verify_pi0_replayfix_10trial_20260729/cube_restack/10trial_pair_run-496713/a1/capx-qwen37-m2-10trials-t0/run-496713
      frozen_a0_pair_validation=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_verify_pi0_replayfix_10trial_20260729/cube_restack/10trial_pair_run-496713/PAIR_VALIDATION.json
      ;;
    cube_stack)
      frozen_a0_run_root=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_verify_stack_pi0_10trial_20260729/cube_stack/10trial_pair_run-496763/a1/capx-qwen37-m2-10trials-t0/run-496763
      frozen_a0_pair_validation=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_verify_stack_pi0_10trial_20260729/cube_stack/10trial_pair_run-496763/PAIR_VALIDATION.json
      ;;
  esac
fi
mkdir -p "$pair_root" /home/qshou/capx_paper_verifier_results/logs
submission_runner_read_only=false
if [[ "$submission_runner_sha" != "not_applicable" ]]; then
  if [[
    "$submission_git_rev" != "$submission_runner_git_rev"
    || "$submission_runner_git_rev" != "$(git -C "$project" rev-parse HEAD)"
    || ! -f "$submission_runner_snapshot"
    || "$(sha256sum "$submission_runner_snapshot" | cut -d' ' -f1)" != "$submission_runner_sha"
  ]]; then
    echo "Frozen top-level runner provenance mismatch" >&2
    exit 4
  fi
  cp "$submission_runner_snapshot" "$pair_root/SUBMISSION_RUNNER.sbatch"
  chmod a-w "$pair_root/SUBMISSION_RUNNER.sbatch"
  submission_runner_snapshot="$pair_root/SUBMISSION_RUNNER.sbatch"
  submission_runner_read_only=true
fi

if [[ "$pair_local_a1_replay" == "1" ]]; then
  if [[
    "$skip_a1" == "1"
    || ! (
      "$replacement_treatment" == "1"
      && "$native_mode" == "M2"
    )
  ]]; then
    echo "Pair-local A1 replay requires live A1 plus an M2 replacement treatment" >&2
    exit 4
  fi
  if [[ -n "$frozen_a0_replay_file" || -n "$frozen_a0_reference" ]]; then
    echo "Pair-local A1 replay cannot consume a frozen full-A1 trajectory" >&2
    exit 4
  fi
  if [[ -z "$canonical_initial_replay_file" ]]; then
    canonical_initial_replay_file="$project/experiments/pgsvp_v9_verify/replays/${task}_qwen37_m2_pi0_seed20260718_10trial_v1.json"
  fi
fi
if [[ -n "$canonical_initial_replay_file" ]]; then
  if [[ ! -f "$canonical_initial_replay_file" ]]; then
    echo "Missing canonical initial replay: $canonical_initial_replay_file" >&2
    exit 4
  fi
  canonical_initial_replay_file=$(realpath "$canonical_initial_replay_file")
  canonical_initial_replay_sha=$(sha256sum "$canonical_initial_replay_file" | cut -d' ' -f1)
fi

if [[
  "$treatment" != "core_fg"
  && !( "$replacement_treatment" == "1" && "$native_mode" == "M3" )
  && "$pair_local_a1_replay" != "1"
  && -z "$frozen_a0_replay_file"
]]; then
  frozen_a0_replay_file="$project/experiments/pgsvp_v9_verify/replays/${task}_qwen37_m2_a0_full_seed20260718_10trial_v1.json"
fi
if [[
  "$treatment" != "core_fg"
  && !( "$replacement_treatment" == "1" && "$native_mode" == "M3" )
  && "$pair_local_a1_replay" != "1"
  && -z "$frozen_a0_reference"
]]; then
  frozen_a0_reference="$project/experiments/pgsvp_v9_verify/replays/${task}_a0_reference_seed20260718_10trial_v1.json"
fi
frozen_a0_replay_sha=not_applicable
frozen_a0_reference_sha=not_applicable
frozen_a0_pair_validation_sha=not_applicable
frozen_a0_has_semantic_reference=0
frozen_a0_semantic_reference_embedded=0
if [[ -n "$frozen_a0_replay_file" ]]; then
  if [[ ! -f "$frozen_a0_replay_file" ]]; then
    echo "Missing frozen A0 replay: $frozen_a0_replay_file" >&2
    exit 4
  fi
  if [[ ! -f "$frozen_a0_reference" ]]; then
    echo "Missing frozen A0 reference: $frozen_a0_reference" >&2
    exit 4
  fi
  frozen_a0_replay_file=$(realpath "$frozen_a0_replay_file")
  frozen_a0_reference=$(realpath "$frozen_a0_reference")
  reference_trials=$(
    "$python_bin" -c \
      'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["total_trials"])' \
      "$frozen_a0_reference"
  )
  reference_global_seed=$(
    "$python_bin" -c \
      'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["global_seed"])' \
      "$frozen_a0_reference"
  )
  if [[ "$trials" != "$reference_trials" || "$global_seed" != "$reference_global_seed" ]]; then
    echo "Frozen A1 reference requires TOTAL_TRIALS=$reference_trials and CAPX_GLOBAL_SEED=$reference_global_seed" >&2
    exit 4
  fi
  frozen_a0_replay_sha=$(sha256sum "$frozen_a0_replay_file" | cut -d' ' -f1)
  frozen_a0_reference_sha=$(sha256sum "$frozen_a0_reference" | cut -d' ' -f1)
  frozen_a0_has_semantic_reference=$(
    "$python_bin" -c \
      'import json,pathlib,sys; r=json.loads(pathlib.Path(sys.argv[1]).read_text()); print(int(isinstance(r.get("m3_semantic_transition_sha256"),dict)))' \
      "$frozen_a0_reference"
  )
  if [[ "$frozen_a0_has_semantic_reference" == "1" ]]; then
    m3_semantic_reference="$frozen_a0_reference"
    frozen_a0_semantic_reference_embedded=1
  fi
  cp "$frozen_a0_replay_file" "$replay_file"
  if [[ "$replacement_treatment" == "1" ]]; then
    # Exact A1 request hashes replay silently; a factual replacement changes
    # the hash and therefore branches to a live regeneration call.
    treatment_replay_file="$frozen_a0_replay_file"
    if [[ "$pair_smoke" == "1" && "$skip_a1" != "1" ]]; then
      treatment_replay_file="$replay_file"
    fi
  else
    treatment_replay_file="$project/experiments/pgsvp_v9_verify/replays/${task}_qwen37_m2_pi0_seed20260718_10trial_v1.json"
    if [[ ! -f "$treatment_replay_file" ]]; then
      echo "Missing treatment initial-policy replay: $treatment_replay_file" >&2
      exit 4
    fi
    treatment_replay_file=$(realpath "$treatment_replay_file")
  fi
fi
if [[
  -n "$treatment_response_replay_file"
  && ( -n "$treatment_full_replay_file" || -n "$treatment_reference" )
]]; then
  echo "Response-frozen audit cannot be combined with registered treatment replay" >&2
  exit 4
fi
if [[ -n "$treatment_full_replay_file" || -n "$treatment_reference" ]]; then
  if [[
    "$treatment" != "fgsrp"
    || -z "$treatment_full_replay_file"
    || -z "$treatment_reference"
  ]]; then
    echo "Exact treatment replay requires FGSRP plus both replay and reference files" >&2
    exit 4
  fi
  if [[ ! -f "$treatment_full_replay_file" || ! -f "$treatment_reference" ]]; then
    echo "Missing registered FGSRP treatment replay/reference" >&2
    exit 4
  fi
  treatment_full_replay_file=$(realpath "$treatment_full_replay_file")
  treatment_reference=$(realpath "$treatment_reference")
  treatment_replay_file="$treatment_full_replay_file"
  treatment_replay_mode=registered_exact
  if [[ -z "$treatment_source_pair_validation" ]]; then
    treatment_source_pair_validation=$(
      "$python_bin" -c \
        'import json,pathlib,sys; r=json.loads(pathlib.Path(sys.argv[1]).read_text()); print(pathlib.Path(r["source_pair_root"])/"PAIR_VALIDATION.json")' \
        "$treatment_reference"
    )
  fi
  if [[ ! -f "$treatment_source_pair_validation" ]]; then
    echo "Missing registered FGSRP source pair validation" >&2
    exit 4
  fi
  treatment_source_pair_validation=$(realpath "$treatment_source_pair_validation")
fi
if [[ -n "$treatment_response_replay_file" ]]; then
  if [[ "$treatment" != "ideal_tk_v4" || "$replacement_treatment" != "1" ]]; then
    echo "Response-frozen audit is currently restricted to Ideal-TK-v4" >&2
    exit 4
  fi
  if [[ ! -f "$treatment_response_replay_file" ]]; then
    echo "Missing treatment response replay: $treatment_response_replay_file" >&2
    exit 4
  fi
  treatment_response_replay_file=$(realpath "$treatment_response_replay_file")
  treatment_replay_file="$treatment_response_replay_file"
  treatment_replay_mode=response_frozen_audit
fi
if [[ "$skip_a1" == "1" ]]; then
  if [[
    "$treatment" == "core_fg"
    || "$service_smoke_only" == "1"
    || -z "$frozen_a0_replay_file"
    || -z "$frozen_a0_run_root"
    || -z "$frozen_a0_pair_validation"
  ]]; then
    echo "Frozen-reference-only A1 mode requires a formal verifier run and all frozen A1 inputs" >&2
    exit 4
  fi
  if [[ ! -d "$frozen_a0_run_root" ]]; then
    echo "Missing frozen A1 run root: $frozen_a0_run_root" >&2
    exit 4
  fi
  if [[ ! -f "$frozen_a0_pair_validation" ]]; then
    echo "Missing frozen A1 pair validation: $frozen_a0_pair_validation" >&2
    exit 4
  fi
  frozen_a0_run_root=$(realpath "$frozen_a0_run_root")
  frozen_a0_pair_validation=$(realpath "$frozen_a0_pair_validation")
  frozen_a0_pair_validation_sha=$(
    sha256sum "$frozen_a0_pair_validation" | cut -d' ' -f1
  )
  expected_a0_pair_validation_sha=$(
    "$python_bin" -c \
      'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["source_pair_validation_sha256"])' \
      "$frozen_a0_reference"
  )
  if [[ "$frozen_a0_pair_validation_sha" != "$expected_a0_pair_validation_sha" ]]; then
    echo "Frozen A1 pair validation hash does not match the registered reference" >&2
    exit 4
  fi
elif [[ -n "$frozen_a0_run_root" || -n "$frozen_a0_pair_validation" ]]; then
  echo "Frozen A1 source artifacts require CAPX_SKIP_A1=1" >&2
  exit 4
fi
treatment_replay_sha=generated_after_a1
if [[ -f "$treatment_replay_file" ]]; then
  treatment_replay_sha=$(sha256sum "$treatment_replay_file" | cut -d' ' -f1)
fi
if [[
  "$treatment" == "envelope_replacement"
  && "$pair_smoke" == "1"
  && "$skip_a1" != "1"
]]; then
  treatment_replay_sha=generated_after_a1
fi

required_seeds=""
required_replay_seeds=""
for trial in $(seq 1 "$trials"); do
  if [[ -n "$fixed_episode_seed" ]]; then
    query_seed_base=$((fixed_episode_seed * 1000))
  else
    query_seed_base=$((global_seed + trial * 1000))
  fi
  seed=$((query_seed_base + initial_action_query_seed_offset))
  if [[ -z "$required_seeds" ]]; then
    required_seeds="$seed"
  else
    required_seeds="${required_seeds}+${seed}"
  fi
  if [[ "$native_mode" == "M3" ]]; then
    report_seed=$query_seed_base
    replay_pair="${report_seed}+${seed}"
  else
    replay_pair="$seed"
  fi
  if [[ -z "$required_replay_seeds" ]]; then
    required_replay_seeds="$replay_pair"
  else
    required_replay_seeds="${required_replay_seeds}+${replay_pair}"
  fi
done

git -C "$project" diff --quiet
git -C "$project" diff --cached --quiet
if [[ -n "$(git -C "$project" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty worktree" >&2
  exit 3
fi
git -C "$project" cat-file -e "${baseline_rev}^{commit}"

"$python_bin" "$helper/tools/fingerprint_infra.py" \
  --project "$project" \
  --file-list "$infra_file_list" \
  --python "$python_bin" \
  --lock "$infra_lock" \
  --output "$infra_fingerprint"
infra_fingerprint_sha=$(sha256sum "$infra_fingerprint" | cut -d' ' -f1)
infra_lock_sha=$(sha256sum "$infra_lock" | cut -d' ' -f1)

git_rev=$(git -C "$project" rev-parse HEAD)
baseline_rev=$(git -C "$project" rev-parse "$baseline_rev")
patch_sha=$(
  git -C "$project" diff --binary "$baseline_rev..$git_rev" | sha256sum | cut -d' ' -f1
)
treatment_implementation_sha=$(sha256sum "$project/$treatment_module" | cut -d' ' -f1)
treatment_spec_sha=$(sha256sum "$project/$treatment_spec" | cut -d' ' -f1)
closed_gt_basis_sha=not_applicable
if [[
  "$treatment" == "closed_gt_takeover"
  || "$treatment" == "ideal_tk"
  || "$treatment" == "ideal_tk_v4"
]]; then
  closed_gt_basis_sha=$(sha256sum "$project/capx/verification/closed_gt.py" | cut -d' ' -f1)
fi
verifier_prompt_sha=not_applicable
verifier_schema_sha=not_applicable
adjudicator_prompt_sha=not_applicable
adjudicator_schema_sha=not_applicable
verifier_bridge_file_sha=not_applicable
verifier_bridge_text_sha=not_applicable
if [[ "$treatment" == "pgsvp_v9_verify" ]]; then
  verifier_prompt_sha=$(
    sha256sum "$project/capx/verification/pgsvp_v9_verifier_prompt.md" | cut -d' ' -f1
  )
  verifier_schema_sha=$(
    sha256sum "$project/capx/verification/pgsvp_v9_verifier_schema.json" | cut -d' ' -f1
  )
  verifier_bridge_file_sha=$(
    sha256sum "$project/capx/verification/pgsvp_v9_public_bridge.txt" | cut -d' ' -f1
  )
  verifier_bridge_text_sha=$(
    "$python_bin" -c \
      'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip().encode("utf-8")).hexdigest())' \
      "$project/capx/verification/pgsvp_v9_public_bridge.txt"
  )
elif [[ "$treatment" == "envelope_replacement" ]]; then
  verifier_prompt_sha=$(
    sha256sum "$project/capx/verification/envelope_replacement_verifier_prompt.md" | cut -d' ' -f1
  )
  verifier_schema_sha=$(
    sha256sum "$project/capx/verification/envelope_replacement_verifier_schema.json" | cut -d' ' -f1
  )
  adjudicator_prompt_sha=$(
    sha256sum "$project/capx/verification/envelope_replacement_adjudicator_prompt.md" | cut -d' ' -f1
  )
  adjudicator_schema_sha=$(
    sha256sum "$project/capx/verification/envelope_replacement_adjudicator_schema.json" | cut -d' ' -f1
  )
elif [[ "$treatment" == "fgsrp" ]]; then
  verifier_prompt_sha=$(
    sha256sum "$project/capx/verification/fgsrp_ir_parser_prompt.md" | cut -d' ' -f1
  )
  verifier_schema_sha=$(
    sha256sum "$project/capx/verification/fgsrp_ir_parser_schema.json" | cut -d' ' -f1
  )
  adjudicator_prompt_sha=$(
    sha256sum "$project/capx/verification/fgsrp_ir_rewriter_prompt.md" | cut -d' ' -f1
  )
  adjudicator_schema_sha=$(
    sha256sum "$project/capx/verification/fgsrp_ir_rewriter_schema.json" | cut -d' ' -f1
  )
elif [[ "$treatment" == "fgsrp_nl" ]]; then
  verifier_prompt_sha=$(
    sha256sum "$project/capx/verification/fgsrp_nl_prompt.md" | cut -d' ' -f1
  )
  verifier_schema_sha=$(
    sha256sum "$project/capx/verification/fgsrp_nl_schema.json" | cut -d' ' -f1
  )
fi
config_suffix=vf
if [[ "$native_mode" == "M3" ]]; then
  config_suffix=vdm
fi
source_config="$project/env_configs/$task/franka_robosuite_${task}_multiturn_${config_suffix}.yaml"
source_config_sha=$(sha256sum "$source_config" | cut -d' ' -f1)

sam3_rev=$(git -C "$sam3_source_root" rev-parse HEAD)
graspnet_rev=$(git -C "$graspnet_source_root" rev-parse HEAD)
if [[ "$sam3_rev" != "$expected_sam3_rev" ]]; then
  echo "Unexpected SAM3 revision: $sam3_rev" >&2
  exit 5
fi
if [[ "$graspnet_rev" != "$expected_graspnet_rev" ]]; then
  echo "Unexpected ContactGraspNet revision: $graspnet_rev" >&2
  exit 5
fi
if [[ -n "$(git -C "$sam3_source_root" status --porcelain)" ]]; then
  echo "Refusing dirty SAM3 dependency" >&2
  exit 5
fi
if [[ -n "$(git -C "$graspnet_source_root" status --porcelain)" ]]; then
  echo "Refusing dirty ContactGraspNet dependency" >&2
  exit 5
fi

sam3_checkpoint_sha=$(sha256sum /scratch/peilab/qshou/models/sam3/sam3.pt | cut -d' ' -f1)
graspnet_checkpoint_sha=$(sha256sum "$graspnet_source_root/checkpoints/contact_graspnet/checkpoints/model.pt" | cut -d' ' -f1)
graspnet_config_sha=$(sha256sum "$graspnet_source_root/checkpoints/contact_graspnet/config.yaml" | cut -d' ' -f1)
if [[ "$sam3_checkpoint_sha" != "$expected_sam3_checkpoint_sha" ]]; then
  echo "Unexpected SAM3 checkpoint hash: $sam3_checkpoint_sha" >&2
  exit 5
fi
if [[ "$graspnet_checkpoint_sha" != "$expected_graspnet_checkpoint_sha" ]]; then
  echo "Unexpected ContactGraspNet checkpoint hash: $graspnet_checkpoint_sha" >&2
  exit 5
fi
if [[ "$graspnet_config_sha" != "$expected_graspnet_config_sha" ]]; then
  echo "Unexpected ContactGraspNet config hash: $graspnet_config_sha" >&2
  exit 5
fi

# Run both arms from one immutable archive of the clean submission commit.
# No Python or helper-script import may resolve through the live worktree.
source_archive="$pair_root/SOURCE_SNAPSHOT.tar"
source_snapshot="$pair_root/source_snapshot"
source_checksums="$pair_root/SOURCE_SNAPSHOT_SHA256SUMS.txt"
git -C "$submission_project" archive --format=tar --output="$source_archive" "$git_rev"
source_archive_sha=$(sha256sum "$source_archive" | cut -d' ' -f1)
mkdir -p "$source_snapshot"
tar -xf "$source_archive" -C "$source_snapshot"
find "$source_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum > "$source_checksums"
chmod -R a-w "$source_snapshot"

snapshot_ready=1
verify_source_snapshot() {
  sha256sum -c "$source_checksums" > "$pair_root/SOURCE_SNAPSHOT_VERIFY.txt"
}
on_exit() {
  local rc=$?
  trap - EXIT
  if [[ "${snapshot_ready:-0}" == "1" ]] && ! verify_source_snapshot; then
    echo "Immutable source snapshot changed during paired run" >&2
    rc=6
  fi
  exit "$rc"
}
trap on_exit EXIT

# Copy all replay/reference inputs that remain live after snapshot creation.
if [[ -n "$canonical_initial_replay_file" ]]; then
  cp "$canonical_initial_replay_file" "$pair_root/canonical_initial_replay.json"
  canonical_initial_replay_file="$pair_root/canonical_initial_replay.json"
fi
if [[ -n "$frozen_a0_reference" ]]; then
  cp "$frozen_a0_reference" "$pair_root/frozen_a0_reference.json"
  frozen_a0_reference="$pair_root/frozen_a0_reference.json"
fi
if [[ "$treatment_replay_mode" == "registered_exact" ]]; then
  cp "$treatment_reference" "$pair_root/frozen_treatment_reference.json"
  treatment_reference="$pair_root/frozen_treatment_reference.json"
  cp "$treatment_source_pair_validation" \
    "$pair_root/frozen_treatment_source_pair_validation.json"
  treatment_source_pair_validation="$pair_root/frozen_treatment_source_pair_validation.json"
fi
if [[ "$skip_a1" == "1" ]]; then
  cp "$frozen_a0_pair_validation" "$pair_root/frozen_a0_source_pair_validation.json"
  frozen_a0_pair_validation="$pair_root/frozen_a0_source_pair_validation.json"
fi
if [[ "$treatment_replay_file" != "$replay_file" ]]; then
  if [[ "$replacement_treatment" == "1" ]]; then
    cp "$treatment_replay_file" "$pair_root/treatment_request_hash_replay.json"
    treatment_replay_file="$pair_root/treatment_request_hash_replay.json"
  else
    cp "$treatment_replay_file" "$pair_root/treatment_initial_replay.json"
    treatment_replay_file="$pair_root/treatment_initial_replay.json"
  fi
fi

project="$source_snapshot"
helper="$project/experiments/core_fg/runtime"
infra_file_list="$project/experiments/pgsvp_v9_verify/infra_files.txt"
infra_lock="$project/experiments/pgsvp_v9_verify/infra_lock.json"
if [[ "$skip_a1" == "1" ]]; then
  a1_reexecuted=false
  same_slurm_allocation=false
else
  a1_reexecuted=true
  same_slurm_allocation=true
fi

{
  echo "schema: capx_${treatment}_paired_run_v1"
  echo "slurm_job_id: $SLURM_JOB_ID"
  echo "node: $(hostname)"
  echo "task: $task"
  echo "trials: $trials"
  echo "service_smoke_only: $service_smoke_only"
  echo "pair_smoke: $pair_smoke"
  echo "pair_local_a1_replay: $pair_local_a1_replay"
  echo "git_rev: $git_rev"
  echo "git_status: clean"
  echo "control_runtime_git_rev: $git_rev"
  echo "control_runtime_kind: patched_clean_runtime"
  echo "baseline_reference_git_rev: $baseline_rev"
  echo "baseline_git_rev: $baseline_rev"
  echo "baseline_to_run_patch_sha256: $patch_sha"
  echo "treatment: $treatment"
  echo "treatment_label: $treatment_label"
  echo "ideal_tk_v4_effect_alignment: $ideal_tk_v4_effect_alignment"
  echo "ideal_tk_v4_effect_alignment_mode: $ideal_tk_v4_effect_alignment_mode"
  echo "treatment_replay_mode: $treatment_replay_mode"
  echo "frozen_treatment_reference: ${treatment_reference:-none}"
  echo "frozen_treatment_source_pair_validation: ${treatment_source_pair_validation:-none}"
  echo "native_mode: $native_mode"
  echo "fgsrp_carrier: $fgsrp_carrier"
  echo "treatment_module: $treatment_module"
  echo "treatment_implementation_sha256: $treatment_implementation_sha"
  echo "closed_gt_basis_sha256: $closed_gt_basis_sha"
  echo "treatment_spec: $treatment_spec"
  echo "treatment_spec_sha256: $treatment_spec_sha"
  echo "verifier_prompt_sha256: $verifier_prompt_sha"
  echo "verifier_schema_sha256: $verifier_schema_sha"
  echo "adjudicator_prompt_sha256: $adjudicator_prompt_sha"
  echo "adjudicator_schema_sha256: $adjudicator_schema_sha"
  echo "verifier_public_bridge_file_sha256: $verifier_bridge_file_sha"
  echo "verifier_public_bridge_text_sha256: $verifier_bridge_text_sha"
  echo "source_task_config: $source_config"
  echo "source_task_config_sha256: $source_config_sha"
  echo "model: qwen3.7-plus"
  echo "qwen_key_file: $qwen_key_file"
  echo "qwen_base_url: $qwen_base_url"
  echo "sam3_source_root: $sam3_source_root"
  echo "sam3_git_rev: $sam3_rev"
  echo "sam3_checkpoint_sha256: $sam3_checkpoint_sha"
  echo "contact_graspnet_source_root: $graspnet_source_root"
  echo "contact_graspnet_git_rev: $graspnet_rev"
  echo "contact_graspnet_checkpoint_sha256: $graspnet_checkpoint_sha"
  echo "contact_graspnet_config_sha256: $graspnet_config_sha"
  echo "temperature: 0"
  echo "num_workers: 1"
  echo "global_seed: $global_seed"
  echo "fixed_episode_seed: ${fixed_episode_seed:-none}"
  echo "episode_seed_base: ${episode_seed_base:-none}"
  echo "initial_action_query_seed_offset: $initial_action_query_seed_offset"
  echo "required_initial_query_seeds: $required_seeds"
  echo "required_native_replay_seeds: $required_replay_seeds"
  if [[ "$skip_a1" == "1" ]]; then
    echo "execution_order: frozen_A0_reference_then_${treatment}_only"
    echo "a0_model_trajectory_frozen: true"
    echo "a0_execution_mode: frozen_reference_only"
  elif [[ -n "$frozen_a0_replay_file" ]]; then
    echo "execution_order: frozen_A0_then_${treatment}"
    echo "a0_model_trajectory_frozen: true"
    echo "a0_execution_mode: replayed_in_current_allocation"
  else
    echo "execution_order: A1_then_export_then_${treatment}"
    echo "a0_model_trajectory_frozen: false"
    echo "a0_execution_mode: live_then_export"
  fi
  echo "a1_reexecuted: $a1_reexecuted"
  echo "same_slurm_allocation: $same_slurm_allocation"
  echo "submission_project: $submission_project"
  echo "submission_git_rev: $submission_git_rev"
  echo "submission_runner_git_rev: $submission_runner_git_rev"
  echo "submission_runner_snapshot: $submission_runner_snapshot"
  echo "submission_runner_sha256: $submission_runner_sha"
  echo "submission_runner_read_only: $submission_runner_read_only"
  echo "runtime_project: $project"
  echo "source_snapshot_archive: $source_archive"
  echo "source_snapshot_archive_sha256: $source_archive_sha"
  echo "source_snapshot_checksums: $source_checksums"
  echo "source_snapshot_read_only: true"
  echo "infra_freeze_lock: $infra_lock"
  echo "infra_freeze_lock_sha256: $infra_lock_sha"
  echo "infra_fingerprint: $infra_fingerprint"
  echo "infra_fingerprint_sha256: $infra_fingerprint_sha"
  echo "canonical_initial_replay_file: ${canonical_initial_replay_file:-none}"
  echo "canonical_initial_replay_sha256: $canonical_initial_replay_sha"
  echo "frozen_a0_replay_file: ${frozen_a0_replay_file:-none}"
  echo "frozen_a0_replay_sha256: $frozen_a0_replay_sha"
  echo "frozen_a0_reference: ${frozen_a0_reference:-none}"
  echo "frozen_a0_reference_sha256: $frozen_a0_reference_sha"
  echo "frozen_a0_run_root: ${frozen_a0_run_root:-none}"
  echo "frozen_a0_source_pair_validation: ${frozen_a0_pair_validation:-none}"
  echo "frozen_a0_source_pair_validation_sha256: $frozen_a0_pair_validation_sha"
  echo "m3_semantic_reference: ${m3_semantic_reference:-generated_after_a1}"
  echo "m3_semantic_reference_embedded_in_frozen_a0: $frozen_a0_semantic_reference_embedded"
  echo "a1_output_base: $a1_base"
  echo "treatment_output_base: $treatment_base"
  echo "replay_file: $replay_file"
  if [[ "$replacement_treatment" == "1" ]]; then
    echo "treatment_request_hash_replay_file: $treatment_replay_file"
    echo "treatment_request_hash_replay_sha256: $treatment_replay_sha"
  else
    echo "treatment_initial_replay_file: $treatment_replay_file"
    echo "treatment_initial_replay_sha256: $treatment_replay_sha"
  fi
} > "$pair_root/PAIR_MANIFEST.txt"

common_env=(
  PROJECT_DIR="$project"
  QWEN_HELPER_DIR="$helper"
  CAPX_FROZEN_GIT_REV="$git_rev"
  CAPX_RUNTIME_SOURCE_ROOT="$project"
  CAPX_PYTHON="$python_bin"
  QWEN_KEY_FILE="$qwen_key_file"
  QWEN_BASE_URL="$qwen_base_url"
  CAPX_SAM3_SOURCE_ROOT="$sam3_source_root"
  CAPX_CONTACT_GRASPNET_SOURCE_ROOT="$graspnet_source_root"
  MODE="$native_mode"
  MODEL=qwen3.7-plus
  TEMPERATURE=0
  TOTAL_TRIALS="$trials"
  NUM_WORKERS=1
  TASK_FILTER="$task"
  CAPX_GLOBAL_SEED="$global_seed"
  CAPX_EPISODE_SEED_BASE="$episode_seed_base"
  QWEN_REQUEST_SEED="$global_seed"
  QWEN_ENABLE_THINKING=false
  QWEN_UPSTREAM_CONCURRENCY=1
  CAPX_SEED_TORCH=1
  CAPX_DETERMINISTIC_TRIAL_SEEDING=1
  CAPX_DETERMINISTIC_GRASPNET=1
  CAPX_START_MOLMO=0
  CAPX_RESET_API_EPISODE_STATE=1
  CAPX_PROCESS_ISOLATED_TRIALS=1
  CAPX_PROCESS_BOOTSTRAP_RETRIES=1
  CAPX_MAX_TRIAL_RETRIES=0
  CAPX_TRIAL_TIMEOUT_SECONDS=1800
  CAPX_REPRO_AUDIT=1
  QWEN_AUDIT_RAW_CONTENT=1
  QWEN_AUDIT_RAW_REQUEST=1
  CAPX_PRIVILEGED_COACH=0
  CAPX_PRIVILEGED_COACH_BRIDGE=0
  CAPX_NODE_COACH=0
  CAPX_SEGMENTED_RECOVERY=0
  CAPX_CONTRACT_REPAIR=0
  CAPX_C1_NO_WORSE=0
  CAPX_MULTITURN_LIMIT=10
  CAPX_PGSVP_V9_VERIFY=0
  CAPX_ENVELOPE_REPLACEMENT=0
  CAPX_FGSRP=0
  CAPX_FGSRP_NL=0
  CAPX_FGSRP_PASSIVE_CAPTURE=0
  CAPX_CLOSED_GT_TAKEOVER=0
  CAPX_CLOSED_GT_PASSIVE_CAPTURE=0
  CAPX_IDEAL_TK=0
  CAPX_IDEAL_TK_V4=0
  CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT=0
  CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE=contradictions
)

a1_replay_env=()
if [[ -n "$frozen_a0_replay_file" ]]; then
  require_all_a0_replay=1
  a1_replay_env=(
    QWEN_REQUEST_HASH_REPLAY_FILE="$replay_file"
    QWEN_REQUIRE_ALL_REQUEST_HASH_REPLAY="$require_all_a0_replay"
    QWEN_ENABLE_NORMALIZED_REPLAY=1
    QWEN_AUDIT_RAW_REQUEST=1
    QWEN_AUDIT_RAW_REQUEST_ON_MISS=1
  )
elif [[ -n "$canonical_initial_replay_file" ]]; then
  a1_replay_env=(
    QWEN_REQUEST_HASH_REPLAY_FILE="$canonical_initial_replay_file"
    QWEN_REQUIRED_REPLAY_SEEDS="$required_replay_seeds"
    QWEN_ENABLE_NORMALIZED_REPLAY=1
    QWEN_AUDIT_RAW_REQUEST_ON_MISS=1
  )
fi

if [[ "$skip_a1" == "1" ]]; then
  a1_run_root="$frozen_a0_run_root"
  echo "[$(date -Is)] Referencing frozen A1 without re-execution: $a1_run_root"
else
  echo "[$(date -Is)] Starting paired A1 run task=$task trials=$trials"
  env "${common_env[@]}" "${a1_replay_env[@]}" \
    CAPX_CORE_FG=0 \
    CAPX_PGSVP_V9_VERIFY=0 \
    CAPX_ENVELOPE_REPLACEMENT=0 \
    CAPX_FGSRP=0 \
    CAPX_FGSRP_NL=0 \
    CAPX_FGSRP_PASSIVE_CAPTURE=$([[ "$native_mode" == "M3" && "$treatment" != "closed_gt_takeover" && "$treatment" != "ideal_tk" && "$treatment" != "ideal_tk_v4" ]] && echo 1 || echo 0) \
    CAPX_CLOSED_GT_PASSIVE_CAPTURE=$([[ "$treatment" == "closed_gt_takeover" || "$treatment" == "ideal_tk" || "$treatment" == "ideal_tk_v4" ]] && echo 1 || echo 0) \
    CAPX_IDEAL_TK=0 \
    CAPX_IDEAL_TK_V4=0 \
    QWEN_OUTPUT_BASE="$a1_base" \
    bash "$helper/run_qwen_multiturn.sbatch"
  native_mode_lc=${native_mode,,}
  a1_run_root="$a1_base/capx-qwen37-${native_mode_lc}-${trials}trials-t0/run-${SLURM_JOB_ID}"
fi
if [[ "$service_smoke_only" == "1" ]]; then
  echo "a1_run_root: $a1_run_root" >> "$pair_root/PAIR_MANIFEST.txt"
  echo "service_smoke_completed_at: $(date -Is)" >> "$pair_root/PAIR_MANIFEST.txt"
  touch "$pair_root/SERVICE_SMOKE_DONE"
  exit 0
fi
if [[ ! -f "$a1_run_root/qwen_proxy_audit.jsonl" ]]; then
  echo "Missing A1 Qwen audit: $a1_run_root/qwen_proxy_audit.jsonl" >&2
  exit 4
fi

if [[
  -z "$frozen_a0_replay_file"
  || (
    "$treatment" == "envelope_replacement"
    && "$pair_smoke" == "1"
    && "$skip_a1" != "1"
  )
]]; then
  "$python_bin" "$helper/tools/export_request_hash_replay.py" \
    --run-root "$a1_run_root" \
    --output "$replay_file"
fi
if [[
  "$replacement_treatment" == "1"
  && -n "$frozen_a0_replay_file"
  && "$pair_smoke" != "1"
  && "$skip_a1" != "1"
]]; then
  "$python_bin" "$helper/tools/export_request_hash_replay.py" \
    --run-root "$a1_run_root" \
    --output "$current_a1_replay_file"
  treatment_replay_file="$current_a1_replay_file"
  echo "treatment_request_hash_replay_file: $treatment_replay_file" \
    >> "$pair_root/PAIR_MANIFEST.txt"
fi
if [[ "$pair_local_a1_replay" == "1" ]]; then
  treatment_replay_file="$replay_file"
fi
if [[ "$native_mode" == "M3" && "$skip_a1" != "1" ]]; then
  visual_registration_args=(
    --replay "$treatment_replay_file"
    --task "$task"
    --global-seed "$global_seed"
    --native-mode M3
    --total-trials "$trials"
  )
  if [[ -n "$fixed_episode_seed" ]]; then
    visual_registration_args+=(--fixed-episode-seed "$fixed_episode_seed")
  fi
  "$python_bin" "$helper/tools/register_replay_visual_fingerprints.py" \
    "${visual_registration_args[@]}"
  m3_semantic_reference="$pair_root/a1_m3_semantic_reference.json"
  if [[
    "$treatment" == "closed_gt_takeover"
    || "$treatment" == "ideal_tk"
    || "$treatment" == "ideal_tk_v4"
  ]]; then
    "$python_bin" "$helper/tools/export_closed_gt_takeover_passive_semantics.py" \
      --run-root "$a1_run_root" \
      --output "$m3_semantic_reference" \
      --task "$task" \
      --total-trials "$trials" \
      --global-seed "$global_seed"
  else
    "$python_bin" "$helper/tools/export_fgsrp_passive_semantics.py" \
      --run-root "$a1_run_root" \
      --output "$m3_semantic_reference" \
      --task "$task" \
      --total-trials "$trials" \
      --global-seed "$global_seed"
  fi
  echo "m3_semantic_reference_runtime: $m3_semantic_reference" \
    >> "$pair_root/PAIR_MANIFEST.txt"
  echo "m3_semantic_reference_sha256: $(sha256sum "$m3_semantic_reference" | cut -d' ' -f1)" \
    >> "$pair_root/PAIR_MANIFEST.txt"
fi
if [[ "$replacement_treatment" == "1" ]]; then
  runtime_treatment_replay_sha=$(
    sha256sum "$treatment_replay_file" | cut -d' ' -f1
  )
  echo "treatment_request_hash_replay_runtime_sha256: $runtime_treatment_replay_sha" \
    >> "$pair_root/PAIR_MANIFEST.txt"
fi

treatment_env=(
  CAPX_CORE_FG=0
  CAPX_PGSVP_V9_VERIFY=0
  CAPX_ENVELOPE_REPLACEMENT=0
  CAPX_FGSRP=0
  CAPX_FGSRP_NL=0
  CAPX_CLOSED_GT_TAKEOVER=0
  CAPX_CLOSED_GT_PASSIVE_CAPTURE=0
  CAPX_IDEAL_TK=0
  CAPX_IDEAL_TK_V4=0
)
if [[ "$treatment" == "core_fg" ]]; then
  treatment_env=(
    CAPX_CORE_FG=1
    CAPX_PGSVP_V9_VERIFY=0
    CAPX_ENVELOPE_REPLACEMENT=0
  )
elif [[ "$treatment" == "pgsvp_v9_verify" ]]; then
  treatment_env=(
    CAPX_CORE_FG=0
    CAPX_PGSVP_V9_VERIFY=1
    CAPX_ENVELOPE_REPLACEMENT=0
    CAPX_PGSVP_V9_TRANSPORT_ATTEMPTS=2
    CAPX_PGSVP_V9_SCHEMA_ATTEMPTS=2
    CAPX_PGSVP_V9_PROMPT_PATH="$project/capx/verification/pgsvp_v9_verifier_prompt.md"
    CAPX_PGSVP_V9_SCHEMA_PATH="$project/capx/verification/pgsvp_v9_verifier_schema.json"
    CAPX_PGSVP_V9_BRIDGE_PATH="$project/capx/verification/pgsvp_v9_public_bridge.txt"
  )
elif [[ "$treatment" == "envelope_replacement" ]]; then
  treatment_env=(
    CAPX_CORE_FG=0
    CAPX_PGSVP_V9_VERIFY=0
    CAPX_ENVELOPE_REPLACEMENT=1
    CAPX_ENVELOPE_REPLACEMENT_TRANSPORT_ATTEMPTS=2
    CAPX_ENVELOPE_REPLACEMENT_SCHEMA_ATTEMPTS=2
    CAPX_ENVELOPE_ADJUDICATOR_SCHEMA_ATTEMPTS=2
    CAPX_ENVELOPE_REPLACEMENT_PROMPT_PATH="$project/capx/verification/envelope_replacement_verifier_prompt.md"
    CAPX_ENVELOPE_REPLACEMENT_SCHEMA_PATH="$project/capx/verification/envelope_replacement_verifier_schema.json"
    CAPX_ENVELOPE_ADJUDICATOR_PROMPT_PATH="$project/capx/verification/envelope_replacement_adjudicator_prompt.md"
    CAPX_ENVELOPE_ADJUDICATOR_SCHEMA_PATH="$project/capx/verification/envelope_replacement_adjudicator_schema.json"
  )
elif [[ "$treatment" == "fgsrp" ]]; then
  treatment_env=(
    CAPX_CORE_FG=0
    CAPX_PGSVP_V9_VERIFY=0
    CAPX_ENVELOPE_REPLACEMENT=0
    CAPX_FGSRP=1
    CAPX_FGSRP_PASSIVE_CAPTURE=0
    CAPX_FGSRP_CARRIER="$fgsrp_carrier"
    CAPX_FGSRP_SCHEMA_ATTEMPTS=2
    CAPX_FGSRP_IR_PARSER_PROMPT_PATH="$project/capx/verification/fgsrp_ir_parser_prompt.md"
    CAPX_FGSRP_IR_PARSER_SCHEMA_PATH="$project/capx/verification/fgsrp_ir_parser_schema.json"
    CAPX_FGSRP_IR_REWRITER_PROMPT_PATH="$project/capx/verification/fgsrp_ir_rewriter_prompt.md"
    CAPX_FGSRP_IR_REWRITER_SCHEMA_PATH="$project/capx/verification/fgsrp_ir_rewriter_schema.json"
  )
elif [[ "$treatment" == "fgsrp_nl" ]]; then
  treatment_env=(
    CAPX_CORE_FG=0
    CAPX_PGSVP_V9_VERIFY=0
    CAPX_ENVELOPE_REPLACEMENT=0
    CAPX_FGSRP=0
    CAPX_FGSRP_NL=1
    CAPX_FGSRP_PASSIVE_CAPTURE=0
    CAPX_FGSRP_CARRIER="$fgsrp_carrier"
    CAPX_FGSRP_NL_SCHEMA_ATTEMPTS=2
    CAPX_FGSRP_NL_PROMPT_PATH="$project/capx/verification/fgsrp_nl_prompt.md"
    CAPX_FGSRP_NL_SCHEMA_PATH="$project/capx/verification/fgsrp_nl_schema.json"
  )
elif [[ "$treatment" == "closed_gt_takeover" ]]; then
  treatment_env=(
    CAPX_CORE_FG=0
    CAPX_PGSVP_V9_VERIFY=0
    CAPX_ENVELOPE_REPLACEMENT=0
    CAPX_FGSRP=0
    CAPX_FGSRP_NL=0
    CAPX_FGSRP_PASSIVE_CAPTURE=0
    CAPX_CLOSED_GT_TAKEOVER=1
    CAPX_CLOSED_GT_PASSIVE_CAPTURE=0
  )
elif [[ "$treatment" == "ideal_tk" ]]; then
  treatment_env=(
    CAPX_CORE_FG=0
    CAPX_PGSVP_V9_VERIFY=0
    CAPX_ENVELOPE_REPLACEMENT=0
    CAPX_FGSRP=0
    CAPX_FGSRP_NL=0
    CAPX_FGSRP_PASSIVE_CAPTURE=0
    CAPX_CLOSED_GT_TAKEOVER=0
    CAPX_CLOSED_GT_PASSIVE_CAPTURE=0
    CAPX_IDEAL_TK=1
    CAPX_IDEAL_TK_V4=0
  )
elif [[ "$treatment" == "ideal_tk_v4" ]]; then
  treatment_env=(
    CAPX_CORE_FG=0
    CAPX_PGSVP_V9_VERIFY=0
    CAPX_ENVELOPE_REPLACEMENT=0
    CAPX_FGSRP=0
    CAPX_FGSRP_NL=0
    CAPX_FGSRP_PASSIVE_CAPTURE=0
    CAPX_CLOSED_GT_TAKEOVER=0
    CAPX_CLOSED_GT_PASSIVE_CAPTURE=0
    CAPX_IDEAL_TK=0
    CAPX_IDEAL_TK_V4=1
    CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT="$ideal_tk_v4_effect_alignment"
    CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE="$ideal_tk_v4_effect_alignment_mode"
  )
fi

echo "[$(date -Is)] Starting paired $treatment_label run task=$task trials=$trials"
if [[
  "$treatment_replay_mode" == "registered_exact"
  || "$treatment_replay_mode" == "response_frozen_audit"
]]; then
  require_all_treatment_replay=1
  m3_prefix_replay=0
  allow_deferred_semantic_visual_match=0
else
  require_all_treatment_replay=0
  m3_prefix_replay=$([[ "$native_mode" == "M3" ]] && echo 1 || echo 0)
  allow_deferred_semantic_visual_match=$(
    [[ "$native_mode" == "M3" && -n "$m3_semantic_reference" && -f "$m3_semantic_reference" ]] \
      && echo 1 || echo 0
  )
fi
env "${common_env[@]}" "${treatment_env[@]}" \
  QWEN_OUTPUT_BASE="$treatment_base" \
  QWEN_REQUEST_HASH_REPLAY_FILE="$treatment_replay_file" \
  QWEN_REQUIRED_REPLAY_SEEDS="$required_replay_seeds" \
  QWEN_ENABLE_NORMALIZED_REPLAY=1 \
  QWEN_REQUIRE_ALL_REQUEST_HASH_REPLAY="$require_all_treatment_replay" \
  QWEN_REQUIRE_NORMALIZED_REPLAY_VISUAL_MATCH=1 \
  QWEN_M3_PREFIX_REPLAY="$m3_prefix_replay" \
  QWEN_ALLOW_DEFERRED_SEMANTIC_VISUAL_MATCH="$allow_deferred_semantic_visual_match" \
  QWEN_REPLAY_LEGACY_RUNTIME_ROOTS=/home/qshou/capx_pgsvp_v9_verify_clean \
  QWEN_AUDIT_RAW_REQUEST_ON_MISS=1 \
  QWEN_AUDIT_RAW_REQUEST=1 \
  bash "$helper/run_qwen_multiturn.sbatch"

native_mode_lc=${native_mode,,}
treatment_run_root="$treatment_base/capx-qwen37-${native_mode_lc}-${trials}trials-t0/run-${SLURM_JOB_ID}"
echo "a1_run_root: $a1_run_root" >> "$pair_root/PAIR_MANIFEST.txt"
echo "treatment_run_root: $treatment_run_root" >> "$pair_root/PAIR_MANIFEST.txt"

verify_source_snapshot

if [[ "$treatment" == "core_fg" ]]; then
  "$python_bin" "$helper/tools/validate_paired_run.py" \
    --pair-root "$pair_root" \
    --a1-run-root "$a1_run_root" \
    --core-run-root "$treatment_run_root" \
    --task "$task" \
    --trials "$trials" \
    --global-seed "$global_seed" \
    --output "$pair_root/PAIR_VALIDATION.json"
elif [[ "$treatment" == "pgsvp_v9_verify" ]]; then
  validation_args=()
  if [[ -n "$frozen_a0_reference" ]]; then
    validation_args+=(--frozen-a0-reference "$frozen_a0_reference")
  fi
  "$python_bin" "$helper/tools/validate_pgsvp_v9_verify_run.py" \
    --pair-root "$pair_root" \
    --a1-run-root "$a1_run_root" \
    --treatment-run-root "$treatment_run_root" \
    --task "$task" \
    --trials "$trials" \
    --global-seed "$global_seed" \
    --bridge-file "$project/capx/verification/pgsvp_v9_public_bridge.txt" \
    "${validation_args[@]}" \
    --output "$pair_root/PAIR_VALIDATION.json"
elif [[ "$treatment" == "envelope_replacement" ]]; then
  validation_args=()
  if [[ -n "$frozen_a0_reference" ]]; then
    validation_args+=(--frozen-a0-reference "$frozen_a0_reference")
  fi
  "$python_bin" "$helper/tools/validate_envelope_replacement_run.py" \
    --pair-root "$pair_root" \
    --a1-run-root "$a1_run_root" \
    --treatment-run-root "$treatment_run_root" \
    --task "$task" \
    --trials "$trials" \
    --global-seed "$global_seed" \
    "${validation_args[@]}" \
    --output "$pair_root/PAIR_VALIDATION.json"
elif [[ "$treatment" == "fgsrp" ]]; then
  validation_args=()
  if [[ -n "$frozen_a0_reference" ]]; then
    validation_args+=(--frozen-a0-reference "$frozen_a0_reference")
  fi
  if [[
    -n "$m3_semantic_reference"
    && "$frozen_a0_semantic_reference_embedded" != "1"
  ]]; then
    validation_args+=(--m3-semantic-reference "$m3_semantic_reference")
  fi
  if [[ -n "$treatment_reference" ]]; then
    validation_args+=(
      --frozen-treatment-reference "$treatment_reference"
      --frozen-treatment-replay "$treatment_replay_file"
      --frozen-treatment-source-validation "$treatment_source_pair_validation"
    )
  fi
  "$python_bin" "$helper/tools/validate_fgsrp_run.py" \
    --pair-root "$pair_root" \
    --a1-run-root "$a1_run_root" \
    --treatment-run-root "$treatment_run_root" \
    --task "$task" \
    --trials "$trials" \
    --global-seed "$global_seed" \
    "${validation_args[@]}" \
    --output "$pair_root/PAIR_VALIDATION.json"
elif [[ "$treatment" == "fgsrp_nl" ]]; then
  validation_args=()
  if [[ -n "$frozen_a0_reference" ]]; then
    validation_args+=(--frozen-a0-reference "$frozen_a0_reference")
  fi
  if [[
    -n "$m3_semantic_reference"
    && "$frozen_a0_semantic_reference_embedded" != "1"
  ]]; then
    validation_args+=(--m3-semantic-reference "$m3_semantic_reference")
  fi
  "$python_bin" "$helper/tools/validate_fgsrp_nl_run.py" \
    --pair-root "$pair_root" \
    --a1-run-root "$a1_run_root" \
    --treatment-run-root "$treatment_run_root" \
    --task "$task" \
    --trials "$trials" \
    --global-seed "$global_seed" \
    "${validation_args[@]}" \
    --output "$pair_root/PAIR_VALIDATION.json"
elif [[ "$treatment" == "closed_gt_takeover" ]]; then
  validation_args=()
  if [[ -n "$frozen_a0_reference" ]]; then
    validation_args+=(--frozen-a0-reference "$frozen_a0_reference")
  fi
  if [[ -n "$m3_semantic_reference" ]]; then
    validation_args+=(--m3-semantic-reference "$m3_semantic_reference")
  fi
  if [[ -n "$fixed_episode_seed" ]]; then
    validation_args+=(--fixed-episode-seed "$fixed_episode_seed")
  fi
  "$python_bin" "$helper/tools/validate_closed_gt_takeover_run.py" \
    --pair-root "$pair_root" \
    --a1-run-root "$a1_run_root" \
    --treatment-run-root "$treatment_run_root" \
    --task "$task" \
    --trials "$trials" \
    --global-seed "$global_seed" \
    "${validation_args[@]}" \
    --output "$pair_root/PAIR_VALIDATION.json"
elif [[ "$treatment" == "ideal_tk" ]]; then
  validation_args=()
  if [[ -n "$frozen_a0_reference" ]]; then
    validation_args+=(--frozen-a0-reference "$frozen_a0_reference")
  fi
  if [[ -n "$m3_semantic_reference" ]]; then
    validation_args+=(--m3-semantic-reference "$m3_semantic_reference")
  fi
  if [[ -n "$fixed_episode_seed" ]]; then
    validation_args+=(--fixed-episode-seed "$fixed_episode_seed")
  fi
  "$python_bin" "$helper/tools/validate_ideal_tk_run.py" \
    --pair-root "$pair_root" \
    --a1-run-root "$a1_run_root" \
    --treatment-run-root "$treatment_run_root" \
    --task "$task" \
    --trials "$trials" \
    --global-seed "$global_seed" \
    "${validation_args[@]}" \
    --output "$pair_root/PAIR_VALIDATION.json"
elif [[ "$treatment" == "ideal_tk_v4" ]]; then
  validation_args=()
  if [[ -n "$frozen_a0_reference" ]]; then
    validation_args+=(--frozen-a0-reference "$frozen_a0_reference")
  fi
  if [[ -n "$m3_semantic_reference" ]]; then
    validation_args+=(--m3-semantic-reference "$m3_semantic_reference")
  fi
  if [[ -n "$fixed_episode_seed" ]]; then
    validation_args+=(--fixed-episode-seed "$fixed_episode_seed")
  fi
  if [[ -n "$episode_seed_base" ]]; then
    validation_args+=(--episode-seed-base "$episode_seed_base")
  fi
  "$python_bin" "$helper/tools/validate_ideal_tk_v4_run.py" \
    --pair-root "$pair_root" \
    --a1-run-root "$a1_run_root" \
    --treatment-run-root "$treatment_run_root" \
    --task "$task" \
    --trials "$trials" \
    --global-seed "$global_seed" \
    "${validation_args[@]}" \
    --output "$pair_root/PAIR_VALIDATION.json"
else
  echo "No validator configured for CAPX_TREATMENT=$treatment" >&2
  exit 4
fi

if [[ "$treatment" == "fgsrp" ]]; then
  "$python_bin" "$helper/tools/build_fgsrp_trial_evidence.py" \
    --pair-root "$pair_root" \
    --output "$pair_root/TRIAL_ATTRIBUTION_EVIDENCE.json"
  echo "trial_attribution_evidence: $pair_root/TRIAL_ATTRIBUTION_EVIDENCE.json" \
    >> "$pair_root/PAIR_MANIFEST.txt"
fi

echo "validation: valid" >> "$pair_root/PAIR_MANIFEST.txt"
echo "validation_file: $pair_root/PAIR_VALIDATION.json" >> "$pair_root/PAIR_MANIFEST.txt"
echo "completed_at: $(date -Is)" >> "$pair_root/PAIR_MANIFEST.txt"
if [[ "$pair_smoke" == "1" ]]; then
  touch "$pair_root/PAIR_SMOKE_DONE"
else
  touch "$pair_root/PAIR_DONE"
fi
