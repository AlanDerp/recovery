# LIBERO G17 Closed Set

## Status

Frozen relative to G17 parent `e1be3a5` (semantic-gain gate) and live IFF smoke
job `518410` (`outputs/libero_g17_iff_smoke_20260814/IFF_SMOKE.json`).

This document freezes the **pick-place closed set** used before any FGSL/FGPP
pairing. It does not invent Open/TurnOn predicates and does not expand the
51-field cube algebra.

## What “closed set” means here

Three layers, kept separate:

| Layer | Contents | Action-visible? |
|---|---|---|
| **G17 state closure** | 1 executor + 2 objects + 1 support surface → **51** current-state fields (34 primitive, 17 derived) | Yes, only after CNL projection |
| **BDDL goal bridge** | Exactly one unmodified `On` or `In` atom; private IFF to `_check_success` | Never (label + atom stay private) |
| **FG projection** | Ordered subset of the 51 fields rendered as eye-checkable English | Yes |

G17 itself is not “17 fields.” For this registry, derived count happens to be
17; the algebra is the entity expansion in `closed_gt.predicate_specs`.

## Coverage (v1)

**In:** Original LIBERO pick-place goals with a single `On`/`In` atom over a
payload and a receptacle body (or site resolved to its parent body). First
target: spatial+object set(20).

**Out (fail-closed bind):** `Open`, `Close`, `TurnOn`, `TurnOff`, multi-atom
goals, articulated/toggle-only tasks (e.g. `libero_goal` stove). Do not map
those into existing enums by approximation.

Support surface: unique `table` or `floor` fixture (object suite uses floor).

## Entity map

| Semantic ID | Role | Public name rule |
|---|---|---|
| `exe01` | Franka gripper | always `robot gripper` |
| `obj01` | BDDL On/In first argument (payload) | humanized instance name |
| `obj02` | receptacle **body** (site → parent) | humanized instance name |
| `surface.table` | table or floor fixture | humanized fixture name |

Simulator IDs (`akita_black_bowl_1`, `basket_1_contain_region`) stay private to
the adapter. FG text must use public names only.

Humanization (`humanize_libero_name`):

- strip a trailing `_` + digits instance suffix;
- strip a trailing `_contain_region` / `_region` site suffix when present;
- replace remaining `_` with spaces;
- lowercase;
- examples: `akita_black_bowl_1` → `akita black bowl`, `plate_1` → `plate`,
  `cream_cheese_1` → `cream cheese`, `main_table` → `main table`.

## Two IFF claims (do not conflate)

1. **State IFF:** completeness vs the declared 51-field quotient. No `UNKNOWN`.
   Derived fields recompute from declared primitives only.
2. **Goal IFF:** private `BDDL On/In atom ⇔ _check_success()`. Smoke `518410`
   passed this on pregoal and forced-success cells. The completion label is
   never action-visible.

### Known geometric hole (not a schema expansion)

On forced success (`libero_spatial` task 0): BDDL `On` and `_check_success`
are true, but G17 `supported_by_reference` can stay false (contact + footprint
inside, support derived false). Therefore:

- **Goal truth** uses the BDDL bridge, never G17 `supported_by` alone.
- **FG text** may report contact, footprint/region, and relative height; it
  must not claim “supported by the plate” unless `supported_by_reference` is
  true, and must not equate that claim with task success.

In (`libero_object` task 1) geometric proxy matched BDDL on forced success.

## Eye-check rule for every FG clause

An FG clause is allowed in the student-facing report only if a human watching
the agent camera / wrist camera video could reasonably agree without reading
coordinates, thresholds, rewards, or API dumps.

Operational tests (all required):

1. **Name test:** clause uses public names (`akita black bowl`), never sim IDs.
2. **Enum test:** value is one of the finite CNL phrases in the FG eye catalog
   (`libero_g17_fg_lexicon.FG_EYE_CATALOG`), not raw tokens like
   `NEGATIVE_OF_REFERENCE_BAND`.
3. **Viewpoint test:** spatial language is robot-base / table-up language
   (toward/away from robot, robot-left/right, above/below), never camera-pixel
   left/right unless a camera frame is separately audited.
4. **Instrumentation tag:** if the catalog marks a family `instrumented`, the
   clause must either (a) be omitted from FG until a visual corroborating
   family is also true, or (b) be worded as contact/grasp evidence, not as a
   pure “looks like” claim.
5. **No goal leakage:** FG never says task success/failure, reward, or BDDL
   atom truth.

Machine audit of the catalog: every `PredicateSpec.family` in the two-object
scene appears exactly once in `FG_EYE_CATALOG`, with non-empty `eye_test` and
CNL templates for every enum value.

## FG eye catalog (families)

### Executor

| Family | Eye test | CNL gist | Tag |
|---|---|---|---|
| `executor_aperture` | Fingers closed / partly open / open | gripper closed / partially open / open | visual |
| `executor_dynamic_state` | Gripper obviously moving vs still | moving / settled | visual-coarse |
| `executor_orientation_to_table` | Gripper upright vs tilted vs upside-down vs table | local z aligned / oblique / opposed to table up | visual-coarse |
| `executor_payload` | Which bound object is in the gripper, if any | empty / `<payload>` / `<receptacle>` / multiple | visual |

### Executor–object

| Family | Eye test | CNL gist | Tag |
|---|---|---|---|
| `executor_object_contact` | Fingers touch the named object | contact true/false | visual |
| `executor_object_holding` | Object rides with the gripper (lifted or pinched) | holding true/false | visual |
| `executor_object_relative_{x,y,z}` | Object vs gripper: toward/away robot, left/right of robot, above/below | see axis CNL below | visual |
| `executor_object_grasp_evidence` | Held+lifted vs contact-only | not held / contact only / lift confirmed | visual |
| `executor_object_grasp_contact_state` | Partition of contact/hold/support/lift | six exclusive CNL states | instrumented+visual |

### Object–object

| Family | Eye test | CNL gist | Tag |
|---|---|---|---|
| `object_object_relative_{x,y,z}` | Payload vs receptacle along robot axes | toward/away, left/right, below/level/above | visual |
| `object_object_support_region` | Overhead: payload center outside / on rim / in center of receptacle top | outside / edge / center of support area | visual-coarse |
| `object_object_volume` | Payload center inside receptacle interior | inside / outside the volume | visual |
| `object_object_footprint` | Same as support-region collapsed to inside/outside | inside / outside the footprint | visual |
| `object_object_contact` | Objects touch | contact true/false | visual |
| `object_object_support` | Payload rests on receptacle (G17 derived) | supported by … true/false | visual-coarse; **≠ BDDL On** |

### Object–surface / object self

| Family | Eye test | CNL gist | Tag |
|---|---|---|---|
| `object_table_contact` | Object touches table/floor | contact true/false | visual |
| `object_table_height` | Clearance above table looks lifted vs not | at/below lift band / above lift band | visual-coarse |
| `object_workspace_height` | Height vs robot workspace guard (restack legacy) | at/below / above workspace guard | instrumented |
| `object_dynamic_state` | Object moving vs still | moving / settled | visual-coarse |
| `object_orientation_to_table` | Upright / tilted / flipped | local z aligned / oblique / opposed | visual-coarse |
| `object_semantic_state` | Held / on table / on other object / airborne | CNL partition | visual |
| `object_top_access` | Something sitting on top blocking grasp from above | clear / blocked from above | visual |
| `object_load_bearing` | Whether this object currently supports another | not supporting / one / multiple | visual |
| `object_support_stability` | What stabilizes the object | table / object center or edge / gripper / multiple / none | visual-coarse |

### Axis CNL (robot base frame, +Z = table up)

Used for every `relative_x/y/z` FG clause; never emit raw band tokens.

| Axis | NEGATIVE | WITHIN | POSITIVE |
|---|---|---|---|
| x | closer to the robot than {ref} | at the same depth as {ref} | farther from the robot than {ref} |
| y | to the robot's right of {ref} | laterally aligned with {ref} | to the robot's left of {ref} |
| z | below {ref} | level with {ref} | above {ref} |

(+Y = robot left is the MuJoCo/robosuite convention used by the G17 frame.)

## Goal atoms (private bridge only)

```text
libero_on  := unmodified receptacle.check_ontop(payload)
libero_in  := unmodified receptacle.check_contain(payload)
              and receptacle.check_contact(payload)
```

These are **not** G17 schema fields. FG may describe geometric evidence; the
benchmark goal remains the BDDL bridge audited privately.

## Deliberate exclusions

Outside the closed set / FG:

- reward, `_check_success`, BDDL atom booleans;
- continuous poses, sizes, thresholds;
- action advice, next-move, code, API parameters;
- camera visibility / corruption epistemics;
- Open / TurnOn / articulation / liquid / deformable state.

## Advancement gate

Before FGSL or FGPP pairing on LIBERO:

1. Schema audit: 51 fields, no `UNKNOWN`, derived dependencies hold.
2. Bind rejects out-of-quotient goals.
3. Live Goal IFF smoke green (pregoal + forced success + TurnOn negative).
4. `FG_EYE_CATALOG` covers every family; unit test enforces completeness.
5. Public names on bound scenes are humanized (no trailing `_N` sim suffixes).

Only then project G17 FG onto set(10) spatial against M3 `517993`.
