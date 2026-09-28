# A* Local Effect-Alignment Smoke, 2026-08-05

## Scope

- Anchor: Ideal-TK V4 A* (`3e287d7`), whose registered cube-stack failures
  are trial 2 and trial 5.
- Task: `cube_stack`, M3, Qwen `qwen3.7-plus`, temperature zero, one worker,
  global seed `20260718`, deterministic ContactGraspNet.
- Both trials use the registered A* M3 prefix replay with strict visual-hash
  validation. After an intervention changes a request, subsequent calls are
  live and prefix replay is blocked.
- No task YAML, reward, completion test, initial state, action API, physical
  threshold, or canonical A* report was changed.

## Local Directions

All directions share one deterministic GT extractor. They differ only in the
small appendix rendered when a public operator effect is GT-contradicted.

| Mode | Visible delta |
| --- | --- |
| `first_failure` | First contradicted effect only |
| `verified_prefix` | GT-confirmed prefix plus first contradiction |
| `release_support` | Support outcome after an unambiguous planned release |
| `compact_outcome` | All GT-proven confirmations plus first contradiction |
| `effect_ledger` | Every evaluable confirmed or contradicted effect |

Ambiguous object or release-reference binding is silent. None of the modes
exposes coordinates, simulator IDs, reward, completion, task answers, or a
repair action.

## Tri-State Correction

The first implementation (`1ef6219`) incorrectly treated an effect that was
not contradicted as confirmed. Trace review found one ledger containing both
"grasp was not established" and "grasp was retained." Results whose visible
prompt used positive confirmation at that commit are excluded.

Commit `5313dfe` fixes the verifier to use three states:

- `confirmed`: an API-bound effect has direct GT event evidence;
- `contradicted`: GT directly disproves the expected effect;
- `unevaluated`: neither conclusion is justified, so nothing is rendered.

Eighty relevant tests pass at the corrected commit.

## Corrected Results

`S` means task completed; `F` means a chain-valid task failure. HTTP 409 strict
visual-replay mismatches are invalid infrastructure attempts and are excluded,
then rerun without relaxing the gate.

| Mode | Trial 2 | Trial 5 | Clean total | Interpretation |
| --- | --- | --- | --- | --- |
| `first_failure` | S, F | S, S | 3/4 | Prompt matched the earlier raw contradiction at first divergence; mainly measures response variance |
| `release_support` | S | S | 2/2 | First visible mismatch fell back to the raw contradiction in these trials; mechanism not isolated |
| `verified_prefix` | S | F | 1/2 | Clean prompt intervention, no robust rescue |
| `compact_outcome` | S, S | S, F | 3/4 | Clean prompt intervention and strongest repeated local result |
| `effect_ledger` | S | F | 1/2 | Clean but too verbose; no repeated positive result |

Corrected tri-state jobs:

- Prefix: `506155`, `506156`.
- Compact: `506157`, `506158`; repeats `506164`, `506165`.
- Ledger: `506159`; trial-5 job `506160` was invalid and its strict rerun was
  `506162`.

## Determinism Boundary

The two corrected compact trial-5 runs had identical first-intervention data:

- takeover prompt SHA-256:
  `184fc27486fc16d9e5863545b01f8f03bd60620eee63a4cba86ca7151fbc18de`
- action request SHA-256:
  `bcc347e3912388d749ca6eacce421067e3c0bc47e6fb99d3ecc24e030812b54c`
- effective request seed: `20265725`

The upstream response hashes differed, and one episode succeeded while the
other failed. Therefore temperature zero plus a fixed request seed does not
provide response-level determinism for the current service. The verifier and
request are controlled; post-divergence model sampling is not fully controlled.

## Decision

`compact_outcome` is the only direction that is both a genuine prompt-level
intervention and net-positive under repetition on the fixed A* failure set.
It is the appropriate candidate for the next broader ablation. The local
result is a mechanism-screening POC, not evidence of no-worse or a population
success-rate claim.
