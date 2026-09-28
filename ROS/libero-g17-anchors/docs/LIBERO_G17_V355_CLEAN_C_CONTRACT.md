# LIBERO G17 v355 Clean-C Contract

## Objective

v355 is the clean adaptive-acquisition state machine on the frozen v354
outer-loop implementation.  It does not replace the native first attempt,
add a grasp-ready prevention gate, or turn PT collision prediction into a
hard certificate.

## Total state transition

```text
AcquireNeeded
  -> native proposal and one owned physical attempt
  -> close-time FG
       TargetHeld
         -> exact inverse escape -> transport -> RR
       EmptyTargetNear
         -> exact inverse retreat -> one adaptive GR call
              TargetHeld -> exact GR inverse escape -> transport -> RR
              otherwise  -> current-world FG native retry
       WrongObjectHeld
         -> open -> exact inverse retreat -> current-world FG native retry
       EmptyBlocked / EmptyClearOffset
         -> open -> inverse when available -> current-world FG native retry
       proposal unavailable / ingress unreached
         -> no close -> current-world FG native retry
```

Native retry is the full-domain fallback.  Adaptive GR owns only
`EmptyTargetNear -> TargetHeld`; a GR exception, unavailable primitive, or
exhausted probe sequence is a typed fallback to native retry, not an episode
failure.

## Information and ownership

- Endpoint classification consumes the finite FG predicate catalog.
- GR consumes discrete RelXYZ/contact-side evidence and robot FK, plus the
  recorded transaction path.
- Student regeneration receives the finite endpoint FG projection without
  continuous privileged object values.
- One physical attempt has one target, proposal, transaction, writer, close,
  endpoint, and successor.
- A non-firm endpoint or payload loss retires the old transaction before the
  next observation.  The next motion requires a fresh `sample_grasp_pose`.

## Deliberately excluded

- v354 `EmptyBlocked -> forced connected retry` hard information gain;
- global PT collision hard veto;
- grasp-ready/contact preconditions before the native first close;
- transparent execution after a failed or partial ingress;
- recursive GR: one NearMiss receives one bounded GR call, then native retry.

## Measured partial-motion ownership

A locally rejected tip-XY command does not erase micro-steps that were already
validated, executed, and measured from robot proprioception. GR consumes that
forward progress from the actual reached pregrasp and repeats the bound ingress
there. Execution-contract violations remain fatal.

This rule consumes no continuous object/world ground truth: its inputs are the
command, measured robot tip displacement, the FK-held Z/orientation contract,
and the subsequent closed FG grasp endpoint.

## Acceptance

1. Native first-attempt command distribution is unchanged.
2. Every `EmptyTargetNear` with a recorded inverse is offered to GR exactly
   once.
3. GR FirmAcquired arms transport only after its recorded inverse completes.
4. GR non-firm/error returns to native retry without stale transaction state.
5. Hard connected-retry directives are zero under clean C.
6. Manual xyz/quaternion, wrong identity, and stale-transaction bypasses do
   not execute physical acquisition motion.
7. Final claims report conditional stages as well as task reward.
