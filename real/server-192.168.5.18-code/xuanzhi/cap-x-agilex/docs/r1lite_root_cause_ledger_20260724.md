# R1 Lite Root-Cause Ledger — 2026-07-24

## Audit status

The work before this ledger was **not fully compliant** with the current hard
standard:

- `/home/nvidia/xuanzhi/cap-x-agilex` has no Git metadata, so there is no
  repository-level pre-change baseline.
- Several early diagnostic commands used fixed output names. Later runs
  overwrote those files, so exact byte-for-byte recovery of every intermediate
  artifact is impossible.
- Process topology changes were logged in terminal output but were not wrapped
  in a unified before/after snapshot.
- Physical rollback was attempted once after an unexpected movement and was
  stopped when divergence increased. The robot was not forced back to the old
  pose.

These facts must not be hidden or retroactively described as compliant.

## Recoverable evidence

### TCP/frame audit

- `tcp_frame_static_audit_20260724.json`
- `tcp_frame_solver_audit_20260724.json`
- `tcp_frame_solver_audit_after_fix_20260724.json`
- `tcp_frame_motion_audit_20260724.json`
- `tcp_frame_static_after_motion_abort_20260724.json`
- `nominal_plan_tcp_validation_20260724.json`

The unintended motion test records its requested target, first abort, failed
return attempt, final stationary feedback and pre/post frame residual.

### MOCA audit

- `moca_readonly_remap_20260724.json`
- `moca_readonly_vendor_20260724.json`
- `moca_zero_motion_active_pdes_20260724.json`
- `moca_functionframe_hold_20260724.json`
- `moca_basis_validation_20260724.json`
- `moca_basis_validation_0010_20260724.json`

The `0.005rad` basis test is retained as a safety-aborted/deadband result. The
`0.01rad` basis test and final joint-6 `0.02rad` probe passed their declared
non-target and return limits.

## Confirmed attribution

1. MOCA declares both a publisher and subscriber on its joint target topic,
   but neither remapped nor vendor topology produced MOCA target-topic samples
   during read-only observation. This is not evidence of an active self-loop.
2. MOCA continuously publishes `/motion_control/control_arm_right` and latches
   its last `p_des`.
3. The original `move_joints_safe()` started interpolation in command space
   but ended at a feedback-space value. The resulting command jump explains
   the large joint-2 motion.
4. The corrected relation is:

   ```text
   next_command_p_des =
       current_MOCA_p_des + desired_feedback - current_feedback
   ```

5. With this correction, the final joint-6 `0.02rad` probe produced
   `0.02043rad` target motion, `0.00085rad` maximum non-target motion and
   `0.00043rad` return error.

## Current code/config changes

- Explicit gripper-link/tool-Z quaternion conversion and physical/planning pose
  APIs in `r1lite_real.py`.
- RelaxedIK reset no longer depends on motion-validation enablement.
- MOCA command/feedback conversion fixed in `move_joints_safe()`.
- Vendor MOCA target topics restored; homogeneous teleoperation is stopped
  before Cap-X ownership.
- Motion validation and adaptive joint control enabled in the active YAML.
- TCP, MOCA and strict nominal-plan audit/generation scripts added.
- Focused TCP/MOCA mapping tests added.

## Available rollback points

- Nominal plan backups:
  - `offline_ik_pick_place_joint_poses_3d_backup_20260724_131207.json`
  - `offline_ik_pick_place_joint_poses_3d_backup_20260724_131307.json`
  - `offline_ik_pick_place_joint_poses_3d_before_remove_f7_20260724_1110.json`
- Future code/config snapshots:
  - `outputs/r1lite_audit_runs/<run_id>/before/files/`
- Future exact file restoration:
  - `scripts/r1lite_experiment_guard.py restore-files`
- Vendor live-stack restoration:
  - restart with `/home/nvidia/start_r1_teleop_ready.sh`, only after reviewing
    the target it will publish. This is not an automatic or motion-free
    rollback and must be treated as a new experiment.

## Prohibited next actions

- No real motion without a unique run ID and pre-snapshot.
- No overwrite of prior JSON/JSONL.
- No MOCA restart without first preserving the active `p_des`.
- No use of measured feedback directly as a MOCA command.
- No execution of the current nominal plan; its frames 2–6 remain rejected by
  offline FK validation.

