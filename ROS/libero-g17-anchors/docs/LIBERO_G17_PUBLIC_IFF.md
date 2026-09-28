# LIBERO G17 Public Goal-IFF (Layer-1)

## Decision

Do **not** iterate rescue smoke until public finish IFF is representation-complete
for the goal family under test. Smoke measures control/rescue; it is not the
Layer-1 optimization target.

## Binary finish (no maybe)

\[
\texttt{finish\_authorized} \;\Leftrightarrow\; \texttt{BDDLAtom} \;\Leftrightarrow\; \texttt{task\_completed}
\]

There is no soft “may finish”. Public FG either authorizes finishing or it does not.

## On atom — public basis (complete)

LIBERO `ObjectState.check_ontop(receptacle, payload)` is exactly:

1. `receptacle.z <= payload.z`
2. `contact(receptacle, payload)`
3. `||payload.xy - receptacle.xy|| < 0.03`

Source: `capx/third_party/LIBERO-PRO/libero/libero/envs/object_states/base_object_states.py`
(`ObjectState.check_ontop`).

Public FG renders these three conjuncts plus their conjunction
(`libero_object_ontop`). By construction:

\[
\texttt{libero\_object\_ontop} \;\Leftrightarrow\; \texttt{ObjectState.check\_ontop}
\]

and therefore for single-atom On goals:

\[
\texttt{finish\_authorized} \;\Leftrightarrow\; \texttt{libero\_object\_ontop}
\]

Implementation: `capx/verification/libero_g17_public_iff.py`.

### Explicit non-basis

G17 `supported_by_reference` (= contact ∧ footprint-inside ∧ relative_z band)
is **not** IFF to On. Example: footprint INSIDE with XY center distance `0.05`
(`> 0.03`) ⇒ `supported_by` may be true while On is false.

## In atom — not yet complete

`check_contain` / site `in_box` is not yet mirrored by a proven public basis.
For `relation=in`, `finish_authorized` stays **False** (fail closed).

## Acquire / place gap notes

Work-gap labels (`acquisition_incomplete` | `placement_incomplete`) describe
remaining motor work. They do **not** authorize finish. When
`finish_authorized` is true, incomplete-work notes are omitted; only the
binary authorize note is emitted.

## Class-level acceptance (before smoke)

1. Unit grid: `tests/test_libero_g17_public_iff.py` — derived basis ≡ check_ontop replica.
2. Documented disagreement set: `supported_by` ⇏ On on the footprint-vs-3cm cases.
3. Projection always emits the On IFF block for On bindings.
4. No rescue smoke gate on this proposition.

## Out of scope here

- Rescue / guard scores on 2-trial smoke
- Expanding the cube 51-field closed-GT algebra
- SiteObjectState / In IFF (follow-on proposition)
