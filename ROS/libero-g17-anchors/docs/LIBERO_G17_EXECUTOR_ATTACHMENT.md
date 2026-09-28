# LIBERO G17 executor attachment (v2.8)

Minimal completeness fix: distinguish **bound payload interaction** from
**gripper occupancy** without expanding the 51-field G17 object algebra.

## Partition (ME)

Over all LIBERO `object_states_dict` movables via `_check_grasp`:

| kind | meaning |
|------|---------|
| `empty` | no movable bilaterally grasped |
| `holding_payload` | exactly the BDDL payload |
| `holding_receptacle` | exactly the receptacle body |
| `holding_other_movable` | distractor and/or multiple / mixed |

Fixtures never enter this partition.

## Public CNL

Always emitted in the acquire endpoint block:

- `robot gripper attachment: …`
- `robot gripper occupied by a non-goal movable object: true|false`

When `holding_other_movable`, gap note is:

- `gripper holds a non-goal movable object; bound payload is not held.`

and the bare `payload not firmly held by the gripper` wording is **not** used.

## Mutex

`LiberoAttachmentFact.assert_mutex_with_snapshot` fail-closes conflicts with
bound `holding` / `payload`. Schema:
`capx-libero-g17-fg-takeover-v2.8-executor-attachment` (sparse:
`…-v2.8-sparse-attachment`).
