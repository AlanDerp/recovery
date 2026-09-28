# R1 Lite TCP / Frame / Quaternion Contract

This document defines the coordinate semantics used by the Cap-X R1 Lite
right-arm integration. Code and experiment logs must name poses according to
this contract instead of using the ambiguous term `ee_pose`.

## Frames

| Name | Meaning |
|---|---|
| `base_link` | Task/world frame used by Cap-X object and waypoint positions. |
| `torso_link3` | Root frame of the floating-right-arm RelaxedIK model. |
| `right_gripper_link` | Native vendor gripper frame, with the native forward axis. |
| `right_gripper_tool_z` | Controlled TCP frame used by RelaxedIK, with +Z as the forward/approach axis. |

The RelaxedIK configuration is:

```text
base link: torso_link3
EE link:   right_gripper_tool_z
```

The fixed URDF transform is:

```text
T_right_gripper_link_right_gripper_tool_z
  translation = [0, 0, 0] m
  rotation    = Ry(+pi/2)
```

Therefore:

```text
T_base_tool_z = T_base_gripper_link * T_gripper_link_tool_z
```

## Pose interfaces

| Interface | Required semantic |
|---|---|
| Cap-X task target | `T_base_tool_z` |
| RelaxedIK target | `T_torso_link3_tool_z` |
| RelaxedIK solved FK | `T_torso_link3_tool_z` |
| `/motion_control/pose_ee_arm_right` | Vendor physical EE report; its link semantic must be measured before conversion and must not be assumed. |
| `/relaxed_ik/motion_control/pose_ee_arm_right` | RelaxedIK-model pose in `torso_link3`; it is not the physical EE topic. |

## Quaternion convention

- All Python and ROS-facing arrays are `[x, y, z, w]`.
- Quaternions are normalized before use.
- Rotation error is computed from relative rotation matrices, never by direct
  quaternion subtraction.
- `q` and `-q` represent the same rotation.
- A target quaternion must be named for the link it orients, for example
  `tool_z_quaternion_xyzw`; unqualified `ee_quaternion` is not acceptable at
  planning/execution boundaries.

## Transform composition

`T_A_B` maps coordinates expressed in frame B into frame A:

```text
p_A = T_A_B * p_B
T_A_C = T_A_B * T_B_C
T_B_A = inverse(T_A_B)
```

The task-to-solver conversion is:

```text
T_torso_tool_z = inverse(T_base_torso) * T_base_tool_z
```

The solver-to-task conversion is:

```text
T_base_tool_z = T_base_torso * T_torso_tool_z
```

## Required experiment metadata

Every newly generated nominal plan must record:

- task pose semantic and frame;
- TCP link name;
- quaternion order;
- measured arm and torso seed;
- transformed RelaxedIK target;
- returned joint solution;
- solved FK in both `torso_link3` and `base_link`;
- position and orientation residual;
- source URDF and RelaxedIK settings paths.

## Nominal plan schema

A plan is data, not an execution authorization. It must use
`schema_version: "r1lite.nominal_plan.v2"` and contain:

- `plan_id`, `generated_at_unix`, `generation_run_id`;
- `status`, initially `candidate_unverified`;
- `pose_semantic: T_base_right_gripper_tool_z`;
- `quaternion_order: xyzw`;
- SHA-256 for the plan source targets, RelaxedIK settings, RelaxedIK URDF,
  full R1 Lite URDF and active low-level configuration;
- measured initial arm and torso feedback;
- immutable validation thresholds;
- ordered `anchors`.

Every anchor must contain:

- stable ID and task label;
- exact `target_position_base_m` and `target_tool_z_quaternion_xyzw`;
- `segment_seed_feedback_joints_rad`;
- a non-empty `microsteps` path, except the current-pose anchor;
- final `solution_feedback_joints_rad`;
- independently recomputable FK and residual fields.

Every microstep must contain:

- local index, fraction and route name;
- exact solver target in `T_base_right_gripper_tool_z`;
- solver seed and returned feedback-space solution;
- FK pose and SE(3) residual;
- incremental joint delta.

Command-space MOCA values must use names containing `command_p_des`; they must
never be stored in a field named `solution` or `feedback_joints`.

Only an independent verifier may produce `verified_for_single_step`. The plan
itself may never self-assert an executable status. A verifier report binds to
the exact plan SHA-256; modifying the plan invalidates the report.

