# Core-FG Cube Placement Specification

## Scope

Core-FG v1 is a deterministic current-state verifier for the CaP-X cube
placement family. It is evaluated on `cube_stack` and `cube_restack`. In both
tasks, the public target is the primary/red cube and the public goal object is
the secondary/green cube.

The verifier runs only when constructing a multiturn regeneration prompt.
Initial student generation is unchanged and must be replayed exactly from the
paired A1 run.

## Public Predicate Basis

The public packet has seven fixed fields:

1. `TARGET_GRASP`: the Robosuite bilateral finger-pad grasp predicate for the
   target object.
2. `RELATIVE_X`: target center is below, within, or above the X interval of
   the goal object's horizontal OBB projection in the task API's robot-base
   frame.
3. `RELATIVE_Y`: the equivalent Y partition.
4. `RELATIVE_Z`: target center is below, within, or above a geometry-scaled
   height band around the goal center.
5. `FOOTPRINT`: target center is inside or outside the exact convex projection
   of the goal object's current OBB onto the robot-base horizontal plane. This
   remains a horizontal predicate when the goal object is tilted.
6. `TARGET_GOAL_CONTACT`: current MuJoCo collision contact between the two
   cube models.
7. `TARGET_TABLE_CONTACT`: current MuJoCo collision contact between the
   target cube and table.

All partitions include their boundaries and therefore produce exactly one
value for every finite valid cube state. `FOOTPRINT=inside` is valid only when
both axis predicates are within their corresponding projected intervals; an
inconsistent extractor result invalidates the verifier turn.

## Excluded Information

Core-FG v1 does not expose coordinates, quaternions, dimensions, thresholds,
simulator identifiers, reward, success, task completion, task stage, temporal
history, failure diagnosis, false-belief claims, repair plans, action advice,
code, or task-specific strategy.

`SUPPORT` is excluded because contact alone does not prove load-bearing
support. `RUNTIME` is excluded because stderr is already present in both A1
and Core-FG prompts.

## Registered Comparison

- Model: `qwen3.7-plus`
- Mode: CaP-X M2
- Temperature: `0`
- Workers: `1`
- Trials: `10` per task and condition
- Pairing: treatment replays each A1 initial response by request hash
- Seed schedule: one fixed global seed and trial-index-derived episode/query
  seeds, shared by both conditions
- Primary paired criterion: zero A1-success to Core-FG-failure harms and
  `(rescues - harms) > 1`
- Historical replication target for `cube_restack`: net gain `+5`

Runs with replay misses, state-hash mismatches, task YAML mutations, native
terminated-episode tracebacks, depth/mask failures, or verifier extraction
errors are invalid rather than negative verifier outcomes.
