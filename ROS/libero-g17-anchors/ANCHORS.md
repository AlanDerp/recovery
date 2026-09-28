# LIBERO G17 performance freeze anchors

Public freeze repo for Cap-X / LIBERO G17 **scored method locks**.

## Current triad (primary)

**[`anchors/v332-x-86-pro-triad/`](anchors/v332-x-86-pro-triad/)** — method `e20e2ea`

| Canonical spatial | PRO Pos (`spatial_swap`) | PRO Task (`spatial_task`) |
|------------------:|-------------------------:|--------------------------:|
| **86/100** | **68/100** | **78/100** |

Mode: v332 + audited X through v354 FG-native-retry; **adaptive GR / Clean-C OFF**.

**Component code index:** [`anchors/v332-x-86-pro-triad/COMPONENTS.md`](anchors/v332-x-86-pro-triad/COMPONENTS.md)
(FG takeover · Soft-NR · ReleaseReady · identity · ownership · extract · topdown · `libero.py` wiring — all links pin `e20e2ea`).

```bash
git checkout freeze/v332-x-86-pro-triad
# or method-only:
git checkout e20e2ea   # tag: handoff/v332-x-86-method
```

## Older method tags

| Tag / branch | Score note |
|--------------|------------|
| `handoff/v332-82-method` / `freeze/v332-82` | immutable v332 live spatial **82/100** |
| `handoff/v332-x-86-method` / `freeze/v332-x-86` | method tree for **86** (pre-triad docs) |

Upstream Cap-X README remains below for environment install; **score claims** are only those listed under `anchors/`.
