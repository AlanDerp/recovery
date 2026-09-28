# LIBERO G17 v286 final-FG golden anchor

Date: 2026-08-19

## Scope

v286 is the live fair-FG champion after the measured-seed / tool-π / full-release
line. Single claimed change vs v285: after the last nonzero XY move, re-observe
both legal FG axis bands before schedule refusal (`CAPX_LIBERO_G17_V286_FINAL_FG_CHECK=1`).

This is a **fair-live** result (live Qwen / Molmo / GraspNet overflow), not a
v269-style Qwen+grasp replay mechanism table.

## Results

| Axis | Score | Notes |
|------|-------|-------|
| spatial task0 K=5×2 | **10/10** | all `valid_prefix_arm`; jobs `523874`,`523865`,`523875`,`523876`,`523877` |
| spatial tasks 1–9 held-out K=1×2 | **12/18** | +2 vs v273 same axis (10/18); jobs `523906`,`523908`–`523915` |

Schema (live):

`capx-libero-g17-fg-takeover-v2.86-...-fair-live-...-full-release-window-final-fg-axis-check`

Output root:

`/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v286_final_fg_20260819`

## Fair / clean gates (revalidated)

Fair:

- FG enabled + engaged on every scored cell
- `CAPX_LIBERO_G17_V272_FAIR_LIVE=1` (schema contains `fair-live`)
- Prefix package intact (`prefix_ok`, `prefix_hits=2`)
- Held-out tasks 1–3 had runner packaging `invalid_infra` false-negatives;
  immutable actual logs were aliased and revalidated (prefix / FG / Molmo /
  mech+release traces). Scores use evidence-valid cells, not packaging labels.

Clean (task0 K5):

- `max_candidate_arm_delta_rad = 1.21033` (acceptance ≤ 1.25)
- `goto_pose stopped` / `measured_joint_jump` / `rollback=True` = 0
- measured-seed contract: native blocking commands/stage = 1; cartesian
  waypoints / joint-delta gate / terminal-residual gate = 0
- 10/10 terminal latches only after full release window
  (`source=release_window_complete`, `released_goal=1`)

Held-out clean note: max measured seed Δq can exceed 1.25 on harder geometries;
freeze treats that as geometry, not branch-jump evidence, and still forbids
`goto_pose stopped` / jump / rollback markers.

## Failure taxonomy (held-out)

- Acquisition incomplete: 5 trials (tasks 3/4/4/6/7)
- Placement incomplete: 1 trial (task 5 trial 1)
- Visible small-step approach jitter remains **unquantified** and is not claimed solved

## Immutable fingerprints

| Artifact | SHA-256 |
|----------|---------|
| `SOURCE_SHA256.json` | `686c739eeacf16eae8cbdacdaf8d3693cff9c8ab22ce0723f42267bf6f03fbcc` |
| `SOURCE_SNAPSHOT.zip` | `d60260ab56a6ebe6ffef8f2ec6252d0bc2146d67750306ee78452a35e92f967d` |
| `FROZEN_FOCUSED_ANCHOR.json` | `be5ef742892a18cbf675a3c7040446a0fdd8c155b252d449fe370ab02e894ff2` |
| `FROZEN_K5_RESULT.json` | `7542ba650fe36c33a78b63fc0711812a3cbd272ef5009f017c60dd5a72c74b52` |
| `FROZEN_SPATIAL_HELDOUT_RESULT.json` | `ae4b15835b4fa989d1d0e0c07b0e35faa12d5a7812826ebebb12c15a9b9ba41b` |

Freeze validators (must exit 0 against the output root above):

- `scripts/freeze_libero_g17_v286.py`
- `scripts/freeze_libero_g17_v286_k5.py`
- `scripts/freeze_libero_g17_v286_spatial_heldout.py`

The completed output tree + source snapshot are the frozen anchor. This
worktree may drift after the freeze commit; restore from `SOURCE_SNAPSHOT.zip`
for exact bit-match reruns.

## Allowed interpretation

Under fair-live FG on `libero_spatial`, v286 reaches full task0 K5 and improves
held-out spatial breadth vs the v273/v272-controller smoke. Do not claim suite
completion, IK-jump solved in general, or jitter elimination.
