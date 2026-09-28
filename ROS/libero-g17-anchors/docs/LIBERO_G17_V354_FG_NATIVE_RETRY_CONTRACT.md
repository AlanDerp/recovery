# LIBERO G17 v354 FG-native-retry contract

## Scope

v354 implements the acquisition outer loop only. Adaptive GR is not an owner
while `CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY=1`; GR may later be inserted solely
under `EmptyTargetNear -> FirmAcquired`.

## One physical attempt

1. A native proposal/executor owns motion through one physical close.
2. The close compiles one closed endpoint FG event.
3. Exactly one hard route owns recovery and transport admission.
4. Recovery ends at a fresh observation boundary.
5. The same event is projected, without continuous values, into the FG report
   that replaces the native visual-difference carrier.

Transaction ownership may record and delimit a native executor. It must not
rank, replace, or synthesize the grasp proposal.

## Total post-close partition

- `TargetHeld`
- `WrongObjectHeld`
- `EmptyTargetNear`
- `EmptyBlocked`
- `EmptyClearOffset`

`EmptyClearOffset` is a retry control state, not a claim that grasping is
impossible. Proposal-unavailable and ingress-unreached are typed pre-close
retry events and also return to fresh FG regeneration.

## Hard projection

- `TargetHeld + inverse available`: keep the gripper closed, reverse ingress,
  then admit transport.
- Every non-target endpoint: open the gripper, reverse ingress when available,
  and regenerate from the resulting current world.
- Missing inverse ingress never raises a terminal control exception; it returns
  to current-world FG regeneration.
- No v354 branch invokes adaptive GR.

## Soft projection

The next student turn receives only finite facts: attachment class, target
contact, RelXYZ bands, whether ingress was partially realized, whether inverse
ingress completed, proposal family, and whether the same family has already
produced the same retry state. No continuous simulator value or private success
label enters the action request.

## Acceptance invariants

- one close, one endpoint, one hard route, one next observation;
- no stale policy transport after a non-target endpoint;
- wrong payload never enters target transport;
- no terminal `NoCandidate` or missing-inverse state;
- GR does not alter the pre-GR command distribution;
- endpoint FG is consumed both by the hard router and the student-visible FG
  carrier, with distinct audited projections.
