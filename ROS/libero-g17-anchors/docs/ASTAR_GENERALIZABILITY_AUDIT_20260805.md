# A* Generalizability Audit (2026-08-05)

## Scope

This audit separates three claims that must not be conflated:

1. explicit anchor hard-coding;
2. implementation support for a task family; and
3. empirical generalization to unseen tasks and trials.

The frozen 20/20 result remains commit
`a99256d4e8c072c6bdcd92db70047fa962b7d81c`. This audit does not rewrite that
result.

## Findings

### Explicit hard-coding

The treatment selector contains no branch on task name, trial number, random
seed, public object name, cube color, reward, completion label, or oracle
answer. Objects are bound from the executed manipulation episode and rendered
through the scene entity registry. The reported facts are derived from GT
contact, grasp, support, and temporal event state.

Task names do occur in adapter admission checks and experiment launchers. They
define the evaluated scope and do not select per-trial feedback.

### Mechanism-level specialization

The 20/20 mode is not a complete task-general verifier. It was selected after
analysis of the frozen cube anchor and highlights only these transport events:

- grasp loss before release;
- loss of a previously lift-confirmed state at an API endpoint; and
- in-API grasp interruption followed by endpoint recovery.

The lift-regression event assumes a tabletop grasp-lift-transport lifecycle.
It is meaningful for the audited cube tasks but is not automatically valid for
sliding, insertion, articulated-object, or bimanual-transfer operators. The
event must therefore be advertised as a transport-family invariant, not a
universal manipulation invariant.

The recovered-loss wording at the frozen 20/20 commit omits endpoint recovery
from its short appended contradiction. Recovery is present in the full ordered
event report and audit record, but the shorter sentence was chosen after an
anchor-trial wording experiment. That is an evaluation-set optimization risk.

### Adapter coverage

The closed-GT compiler is structurally parameterized by executors, objects,
and predicate families, but the production extractor and goal-IFF adapter are
currently implemented only for `cube_stack` and `cube_restack`. Consequently:

- source-level task independence is supported only inside the two-cube
  adapter;
- cross-task implementation generality is not yet established; and
- empirical task generalization is not established by the frozen 20 trials.

## Alignment Increment

The post-audit increment keeps the frozen result intact and makes three
task-independent corrections:

1. public API names are normalized to close, release, and motion roles,
   including single-arm and `arm0`/`arm1` spellings;
2. a grasp episode records its actual executor instance, and lift regression
   uses that binding instead of executor slot zero; and
3. the concise recovered-loss fact states both interruption and endpoint
   recovery in the same sentence.

No task, trial, seed, object-name, strategy, coordinate, reward, success, or
oracle branch is introduced. The wording change deliberately requires a new
treatment smoke because it changes the regeneration request.

## Claim Boundary

The frozen result supports empirical dominance on the fixed response-frozen
20-trial cube suite. It does not support universal or unseen-task dominance.
The next generalization claim requires preregistered, response-frozen paired
tests on at least one new single-arm transport task and one new executor/task
family. Peg insertion and two-arm handover also require task adapters and clean
chain-health smoke tests before they can serve that role.

## Targeted Alignment Smoke

Job `506772` tested the aligned wording on the registered cube-restack trial 7,
the anchor case whose outcome had previously depended on recovered-loss
wording. The first 11 requests were exact request-hash replay hits. The aligned
request then diverged as intended and used three live model responses.

The trial completed with reward `1.000`, seven code blocks, and six
regenerations. The public contradiction stated both that grasp was interrupted
during the motion API and that it was re-established at that API's endpoint.
No replay miss or declared chain-health blocker occurred.

This is evidence that the truthful, executor-bound representation can retain
the local recovery. It is not an exact full-response replay result and does not
upgrade the cross-task claim boundary above.

## Frozen-Suite Alignment Result

The aligned implementation at commit
`28b9ed35964f4c94e6e0b9a82c2f05b0fcd45bf9` was evaluated on the complete
registered 10-trial replay suite for each frozen cube task. Both jobs were
pinned to the renderer-validated `dgx-10` node after runs `506790` and `506791`
were invalidated by pre-treatment visual replay mismatches on `dgx-29`.

Valid jobs `506804` (`cube_stack`) and `506805` (`cube_restack`) both completed
with Slurm exit code `0:0`:

| Task | Compile rate | Average reward | Completed | Average regenerations |
| --- | ---: | ---: | ---: | ---: |
| `cube_stack` | 1.000 | 1.000 | 10/10 | 3.300 |
| `cube_restack` | 1.000 | 1.000 | 10/10 | 2.800 |

Across both valid runs, 159 of 162 model requests were exact request-hash
replay hits. All 86 stack requests were exact hits. Restack used 73 exact hits;
trial 7 then made three live calls after the aligned recovered-loss wording
caused the intended request divergence. All requests returned HTTP 200. No
semantic visual fallback, replay miss, depth/mask/OBB failure, or terminated-
episode blocker occurred. Every task trial has a persisted verifier state
trace.

This establishes that the task-independent alignment increment preserves the
frozen-suite 20/20 result. It strengthens implementation cleanliness within the
audited two-cube transport scope; it still does not establish unseen-task
generalization.
