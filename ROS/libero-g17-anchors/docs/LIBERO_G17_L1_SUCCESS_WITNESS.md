# LIBERO G17 Layer-1 Success Witness (v2.6)

## Problem (from fair K=5 v24)

OK trials never had `bddl_atom_true` / `finish_authorized` in FG traces.
Cause: `trial.py` breaks on `task_completed` **before** multi-turn / FG takeover.

Near-miss fails (contact ∧ footprint inside ∧ XY outside band) are correctly
rejected by public On IFF; not an occlusion issue.

## Fix

1. `capture_terminal_witness` — project FG at episode stop without a decision prompt.
2. Hook before `break` when `CAPX_LIBERO_G17_FG=1`.
3. Fail-closed if `task_completed` but public `finish_authorized` is false.
4. Near-miss CNL note when On fails specifically on the XY alignment band.
5. Schema: `capx-libero-g17-fg-takeover-v2.6-success-witness`.

## Freeze gate (Layer-1)

On success stops: terminal witness must show
`finish_authorized ∧ bddl_atom_true ∧ private_task_completed`.

## Experiment

Treatment-only K=5 under the same code-prefix package; **reuse** baseline reps
from `libero_g17_fair_fg_20260815` (do not re-run baseline).
