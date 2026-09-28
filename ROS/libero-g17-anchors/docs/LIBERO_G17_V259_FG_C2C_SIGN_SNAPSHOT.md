# v259 FG c2c sign (method) snapshot

**Schema:** `v2.57-…-xy-decouple-fg-sign`  
**Flags:** `NEARMISS_OSC` + `BANDWINDOW` + `XY_DECOUPLE` (**no** `STUDENT_SIGN`)

| Piece | Source |
|---|---|
| mag | `0.15 − t_a·0.03` independent `t_x`/`t_y` |
| sign | **FG c2c** (same payload−receptacle FG uses for closer/farther & left/right) |
| within | predicate `WITHIN_ONTOP_XY_BAND` |
| not | student Molmo; not v254 vector anchoring |

v258 = same schedule + **student** sign (`STUDENT_SIGN=1`) ablation.

## `_fg_c2c_axis_residual`

```python
def _fg_c2c_axis_residual(snapshot: Any) -> tuple[np.ndarray | None, str]:
    """FG center-to-center axes: same payload−receptacle as FG placement facts.

    This is Factual Grounding c2c (closer/farther, left/right source), not a
    student pose and not a move-magnitude vector.
    """
    try:
        payload = np.asarray(snapshot.private_poses["obj01"][0], dtype=np.float64)
        receptacle = np.asarray(snapshot.private_poses["obj02"][0], dtype=np.float64)
    except Exception as exc:
        return None, f"fg_c2c_unavailable:{exc!r}"
    residual = payload[:2] - receptacle[:2]
    return residual.astype(np.float64), "fg_c2c"


```

## `_run_xy_decouple_bandwindow`

```python
def _run_xy_decouple_bandwindow(api: Any) -> bool:
    """XY-decoupled 15−3t; FG c2c sign (or student-sign ablation); predicate within."""
    from capx.integrations.franka.tip_xy_only import move_tip_xy
    from capx.verification.libero_g17_release_ready import _read_live

    student_sign = nearmiss_osc_student_sign()
    tag = (
        "nearmiss-osc-xy-decouple-student-sign"
        if student_sign
        else "nearmiss-osc-xy-decouple-fg"
    )
    try:
        _binding, snapshot, gate = _read_live(api)
    except Exception as exc:
        print(f"[{tag}] snapshot failed ({exc!r}); refuse open", flush=True)
        return False

    if gate.relation != "on":
        return True
    if not gate.acquired:
        return True
    if _predicate_within(gate):
        print(
            f"[{tag}] predicate xy_within; open ok (acquired={gate.acquired})",
            flush=True,
        )
        return True

    start = resolve_start_m(api)
    step = float(LIBERO_OBJECT_ONTOP_XY_M)
    sign_law = "student_get_object_pose" if student_sign else "fg_c2c"
    print(
        f"[{tag}] takeover start={start:.4f}m step={step:.4f}m "
        f"law=mag(start-t*step) per-axis; sign={sign_law}; within=predicate FG",
        flush=True,
    )

    if getattr(api, "_capx_release_ready_busy", False):
        return True
    api._capx_release_ready_busy = True
    t_x = 0
    t_y = 0
    try:
        for round_i in range(_MAX_ROUNDS):
            _b, snapshot, gate = _read_live(api)
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

            axis_r, src = _axis_sign_residual(api, snapshot)
            if axis_r is None:
                print(
                    f"[{tag}] sign residual unavailable ({src}); refuse open",
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
                r_a = float(axis_r[axis])
                d = axis_delta(r_a, mag, step)
                if abs(d) < 1e-9:
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
                    f"r={r_a:+.4f} d={d:+.4f} src={src}",
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

                _b, snapshot, gate = _read_live(api)
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
                axis_r, src = _axis_sign_residual(api, snapshot)
                if axis_r is None:
                    print(
                        f"[{tag}] sign residual lost mid ({src}); refuse open",
                        flush=True,
                    )
                    return False

            if exhausted_both:
                ok = _predicate_within(gate)
                print(
                    f"[{tag}] both axis schedules exhausted "
                    f"predicate_within={ok} t_x={t_x} t_y={t_y} "
                    "(no residual-vector snap)",
                    flush=True,
                )
                return ok
            if not moved_any:
                print(
                    f"[{tag}] no axis move (half-band) but predicate still out; "
                    f"refuse open t_x={t_x} t_y={t_y}",
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
