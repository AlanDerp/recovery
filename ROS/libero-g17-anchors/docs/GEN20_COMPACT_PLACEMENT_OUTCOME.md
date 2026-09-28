# Gen20 Compact Placement Outcome

## Goal-bound novelty selection

The complete endpoint and ordered-event report remains unchanged. The compact
placement line is only a high-salience derived summary, so it is emitted when
it adds goal-relevant interval evidence:

- ordered object pairs are derived from relational atoms in the closed goal;
- reverse and non-goal pairs remain available through atomic GT but receive no
  duplicate compact summary;
- support established then lost, and region entered without support, are
  visible immediately for a goal pair;
- never entering the support region is visible only after the first recovery
  turn, when it becomes sequential negative evidence; and
- support at the endpoint is not duplicated because it is already explicit in
  the endpoint state.

These rules contain no task-name, seed, coordinate, action, or reward branch.

## Scope

This candidate is a single factual-grounding ablation over the exact historical
gen20 source commit `451159847372a20b3947a416805241faeeac67a7` used by jobs
`508857` and `508858`.

It does not change the task, reward, success validator, initial policy, replay,
continuous action parameters, code, API calls, effect-alignment mode, or endpoint
closed-GT basis. It adds one task-independent temporal projection for objects that
were bilaterally grasped during the executed code.

## Instance binding

An object is instantiated as a placement subject if and only if at least one
executor-object interval outcome proves bilateral grasp contact during that
interval. Each such subject is paired with every other scene object in canonical
scene order. No object name, task name, seed, target role, or desired answer is
hard-coded into the binding.

## Total partition

For each instantiated subject-reference pair, the simulator-step recorder tracks
the joint qualitative state:

`(support_region, supported_by_reference)`

The region is exactly one of `OUTSIDE`, `EDGE_REGION`, and `CENTER_REGION`; support
is exactly one of `FALSE` and `TRUE`. Closed-GT validation rejects the impossible
combination `(OUTSIDE, TRUE)`.

The complete interval is mapped to exactly one outcome, in this precedence order:

1. `CENTER_SUPPORT_AT_ENDPOINT`
2. `EDGE_SUPPORT_AT_ENDPOINT`
3. `CENTER_SUPPORT_ESTABLISHED_THEN_LOST`
4. `EDGE_SUPPORT_ESTABLISHED_THEN_LOST`
5. `CENTER_REGION_ENTERED_WITHOUT_SUPPORT`
6. `EDGE_REGION_ENTERED_WITHOUT_SUPPORT`
7. `NEVER_ENTERED_SUPPORT_REGION`

Endpoint support takes precedence over history. If support is absent at the
endpoint, established support takes precedence over region entry without support,
and center takes precedence over edge. This makes the mapping deterministic,
mutually exclusive, and exhaustive while retaining the strongest placement fact
observed during the interval.

Changes from the same simulator step are applied atomically before classification.
Region and support observations from different times are never joined into a
synthetic state.

## Public and audit carriers

The student sees one concise natural-language placement outcome per instantiated
pair. Raw support-region transitions are filtered from the ordered event section to
avoid symmetric event noise. Existing endpoint support-region predicates remain
unchanged.

The trace stores the complete qualitative event batches under
`audited_recorded_temporal_event_batches`. These audit events are not inserted into
the student prompt and contain no continuous poses or coordinates. Their hash makes
the compact mapping independently reproducible after the run.

## Validation

Synthetic tests cover all seven outcomes, atomic event reconstruction,
instance-bound rendering, raw-region filtering, sanitization, and exact Ideal-V4 to
closed-takeover report projection. The frozen `closed_gt.py` extractor is unchanged.

This candidate proceeds to a full trial only if its frozen-prefix replay and chain
health pass. Smoke success is interpreted at the seed level; partial reward or
changed reasoning alone is mechanism evidence, not a task rescue.
