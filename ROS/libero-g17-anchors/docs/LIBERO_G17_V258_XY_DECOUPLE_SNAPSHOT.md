# v258 XY-decouple ReleaseReady snapshot

**Schema:** `v2.56-…-nearmiss-osc-xy-decouple-predicate`  
**Flags:** `RELEASE_READY=1` `NEARMISS_OSC=1` `NEARMISS_OSC_BANDWINDOW=1` `NEARMISS_OSC_XY_DECOUPLE=1`  
**Not set:** `NEARMISS_OSC_FAIR_RESIDUAL` (within stays predicate FG)

## Law

| Piece | Source |
|---|---|
| mag | `0.15 − t_a·0.03` per axis (independent `t_x`, `t_y`) |
| sign | student `get_object_pose` c2c (± only; skip if \|r_a\| < 0.03) |
| open/enter | predicate `xy_within` / `acquired` (FG declared states) |
| forbidden | GT `private_poses` residual vector; `Δ=−r̂·min(band,\|r\|)` |

## Code (as submitted)

### `axis_magnitude` / `axis_delta`

```python
def axis_magnitude(start_m: float, step_m: float, t: int) -> float | None:
    """Return move magnitude for axis-local round ``t``, or None when exhausted."""
    mag = float(start_m) - float(t) * float(step_m)
    if mag <= 0.0:
        return None
    return mag



def axis_delta(residual_axis: float, magnitude: float, step_m: float) -> float:
    """Signed tip move for one axis; 0 if already inside ±step on that axis."""
    r = float(residual_axis)
    if abs(r) < float(step_m):
        return 0.0
    if abs(r) < 1e-12:
        return 0.0
    return float(-np.sign(r) * float(magnitude))


```

### `_run_xy_decouple_bandwindow`

```python
def _run_xy_decouple_bandwindow(api: Any) -> bool:
    """v258 law: independent 15−3t per axis; student sign; predicate within."""
    from capx.integrations.franka.tip_xy_only import move_tip_xy
    from capx.verification.libero_g17_release_ready import _read_live

    tag = "nearmiss-osc-xy-decouple"
    try:
        _binding, _snapshot, gate = _read_live(api)
    except Exception as exc:
        print(f"[{tag}] snapshot failed ({exc!r}); refuse open", flush=True)
        return False

    if gate.relation != "on":
        return True
    if not gate.acquired:
        return True
    # Predicate FG open/enter — never GT residual magnitude.
    if _predicate_within(gate):
        print(
            f"[{tag}] predicate xy_within; open ok (acquired={gate.acquired})",
            flush=True,
        )
        return True

    start = resolve_start_m(api)
    step = float(LIBERO_OBJECT_ONTOP_XY_M)
    print(
        f"[{tag}] takeover start={start:.4f}m step={step:.4f}m "
        f"law=mag(start-t*step) per-axis; sign=student; within=predicate FG",
        flush=True,
    )

    if getattr(api, "_capx_release_ready_busy", False):
        return True
    api._capx_release_ready_busy = True
    # Independent schedule indices — X and Y need not share a round.
    t_x = 0
    t_y = 0
    try:
        for round_i in range(_MAX_ROUNDS):
            _b, _s, gate = _read_live(api)
            if not gate.acquired:
                print(f"[{tag}] lost grasp round={round_i}; refuse open", flush=True)
                return False
            if _predicate_within(gate):
                print(
                    f"[{tag}] predicate within round={round_i} "
                    f"t_x={t_x} t_y={t_y}; open",
                    flush=True,
                )
                return True
            if not gate.contact:
                print(
                    f"[{tag}] contact=False round={round_i} "
                    "(continue schedule; flat plate)",
                    flush=True,
                )

            student_r, src = _student_axis_residual(api)
            if student_r is None:
                print(
                    f"[{tag}] student sign unavailable ({src}); refuse open",
                    flush=True,
                )
                return False

            moved_any = False
            exhausted_both = True
            for axis, t_axis, name in ((0, t_x, "x"), (1, t_y, "y")):
                mag = axis_magnitude(start, step, t_axis)
                if mag is None:
                    continue
                exhausted_both = False
                r_a = float(student_r[axis])
                d = axis_delta(r_a, mag, step)
                if abs(d) < 1e-9:
                    # Already inside ±step on this axis — do not advance t.
                    print(
                        f"[{tag}] round={round_i} axis={name} skip "
                        f"|r|={abs(r_a):.4f}<step t={t_axis} src={src}",
                        flush=True,
                    )
                    continue
                dx = d if axis == 0 else 0.0
                dy = d if axis == 1 else 0.0
                print(
                    f"[{tag}] round={round_i} axis={name} t={t_axis} "
                    f"mag={mag:.4f} sign={-int(np.sign(r_a)):+d} "
                    f"r_student={r_a:+.4f} d={d:+.4f} src={src}",
                    flush=True,
                )
                res = move_tip_xy(
                    api,
                    dx,
                    dy,
                    tol_m=0.0015,
                    max_step_m=float(step),
                    max_iters=20,
                    log=True,
                )
                if abs(res.d_tip_z) > _Z_ABORT_M:
                    print(
                        f"[{tag}] Z abort round={round_i} d_tip_z={res.d_tip_z:+.5f}",
                        flush=True,
                    )
                    return False
                moved_any = True
                if axis == 0:
                    t_x = t_axis + 1
                else:
                    t_y = t_axis + 1

                # Predicate check after each single-axis move (decoupled).
                _b, _s, gate = _read_live(api)
                if not gate.acquired:
                    print(
                        f"[{tag}] lost grasp mid-axis={name}; refuse open",
                        flush=True,
                    )
                    return False
                if _predicate_within(gate):
                    print(
                        f"[{tag}] predicate within after axis={name} "
                        f"t_x={t_x} t_y={t_y}; open",
                        flush=True,
                    )
                    return True
                # Refresh student residual for the other axis.
                student_r, src = _student_axis_residual(api)
                if student_r is None:
                    print(
                        f"[{tag}] student sign lost mid ({src}); refuse open",
                        flush=True,
                    )
                    return False

            if exhausted_both:
                ok = _predicate_within(gate)
                print(
                    f"[{tag}] both axis schedules exhausted "
                    f"predicate_within={ok} t_x={t_x} t_y={t_y} "
                    "(no GT residual snap)",
                    flush=True,
                )
                return ok
            if not moved_any:
                # Both axes already inside student half-band but predicate still out
                # (perception vs FG disagree) — refuse rather than GT-snap.
                print(
                    f"[{tag}] no axis move (student half-band) but "
                    f"predicate still out; refuse open t_x={t_x} t_y={t_y}",
                    flush=True,
                )
                return False

        _b, _s, gate = _read_live(api)
        ok = bool(gate.acquired and _predicate_within(gate))
        print(
            f"[{tag}] max_rounds predicate_within={ok} t_x={t_x} t_y={t_y}",
            flush=True,
        )
        return ok
    finally:
        api._capx_release_ready_busy = False


```
