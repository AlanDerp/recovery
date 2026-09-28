# LIBERO G17 v269-mech mechanical golden anchor

Date: 2026-08-18

## Scope

v269-mech is the mechanism-level GT→FG wash anchor for the v256 controller.
It replays the complete v256 Qwen policy/VDM response stream and the frozen
v256 grasp package, preserves one shared v256 mechanical controller, and
selects FG ternary axis bands while recording GT bands as a prompt-silent
shadow.

This anchor proves action-level wash equivalence. It is not a live-policy
deployment result and does not claim that FG outperforms GT.

## Result

- Valid cells: 10/10 (`valid_prefix_arm`)
- Trial 1 rescue: 5/5
- Trial 2 guard: 5/5
- Gain result: positive for all five replicates
- Mechanism axis decisions: 11
- FG/GT band mismatches: 0
- Qwen effective-seed replay calls: 64/64 successful

Jobs: `522986`–`522990`

Output root:

`/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v269_mech_20260818`

Schema:

`capx-libero-g17-fg-takeover-v2.69-ordinal-sgp-nearmiss-osc-tipxy-bandwindow-v256-shared-mech-fg-source`

## Immutable fingerprints

- Prefix package SHA-256:
  `864604a1c0b39a7c0f8d188d89e9d6ed4f5b36ba1da717e1eee6faee919563cb`
- Grasp package SHA-256:
  `ce436389b6b64d614928db3a36e6b1e117ee590aa5ec62b8b1f4fc6025f655ef`
- Effective-seed replay manifest SHA-256:
  `012c89c381fce8e9a0825f3714616ef18e8a6b597e70d9c973501b61747cf314`
- Runtime source manifest SHA-256:
  `9bf0f2ff561c15b439dbda319dc316a72568fadf98661ffd02a29986f1763513`

The completed output, replay packages, per-job mechanism traces, and source
fingerprint manifest are the frozen anchor. The mutable development worktree
is not the anchor and may legitimately drift after submission.

## Allowed interpretation

Under a fixed v256 policy, grasp package, initial-state protocol, schedule,
tip-XY primitive, and release mechanism, replacing the GT-quantized axis
symbol source with the ideal FG ternary source caused no observed action-band
mismatch and no performance loss.

The historical v256 score was 9/10; v269-mech scored 10/10. The extra cell is
not attributed to FG. The defensible claim is non-degradation under the
mechanism-controlled replay.

## Next paired ablation

The PT ablation must keep the full v269 replay mechanism fixed and compare two
fresh arms under one source manifest:

- FG control: ideal FG ternary axis bands
- PT treatment: ternary axis bands derived from `get_object_pose`

Both arms must share effective-seed policy replay, grasp replay, controller,
initial states, release semantics, source tree, and prompt-silent audit. Only
the selected discrete symbol source may differ.
