# Phase 3: the force library

- **Dates:** 5 October 2026
- **Plan:** task 3.1; decision D24
- **Commits:** see the index
- **Written:** at the time

## Goal

Spring and damper with tabulated nonlinear curves, preload, bump and rebound
stops; a rotational spring-damper; forces on a joint coordinate (spring,
damper, friction, limit stop); a six-axis bushing, linear then nonlinear; a
user force. Each returns its derivatives with respect to position and
velocity. *Done when each derivative matches finite differences.*

## Decision first (D24)

Which derivatives? Statics, linearisation and implicit integration need the
derivatives of the generalized forces with respect to q and v. Two routes:
each element's tangent stiffness and damping in generalized coordinates
(geometric stiffness included), or each element's force law in its own
coordinates (a spring's length and rate, a joint's coordinate and rate) with
its two derivatives. The second was chosen: it is what can be verified
element by element now, and the assembly into generalized matrices belongs to
the tasks that use it. Tables are monotone cubics (PCHIP), because natural
splines overshoot between points: a monotone damper table could then give a
force of the wrong sign. Forces on joint coordinates became a kernel element
of their own, adding to the generalized forces, because the world-force
interface never sees a joint's coordinate.

## What was done

- `Curve`: a line, or a table through its points (Fritsch-Carlson slopes with
  Moler's end conditions), continued as a line with the end slopes. `LawValue`
  carries a force and its two derivatives.
- `SpringDamper`: spring curve against compression, preload, damper curve
  against the rate, bump and rebound stops beyond their clearances.
- `JointCoordinateForce` (kernel): spring, preload, damper, Coulomb friction
  regularised by `tanh(v / v_f)`, and limit stops whose spring and damper push
  but never pull. `System::joint_forces`; the simulator applies them after the
  world forces; constructor and `validate()` check their bodies.
- `RotationalSpringDamper`: twist of one marker relative to another about the
  first's Z axis, `2 atan2(z, w)` of their relative quaternion (the
  swing-twist split), so a small misalignment of the axes does not change it to
  first order.
- `Bushing`: in the first marker's axes, the rotation vector and the
  displacement of the second marker, their rates, and per-axis curves.
- `UserForce`: a function of the states.
- Message codes: F002 to F004, K026, K027, K042, K043, M035, M036, with
  catalogue entries; the catalogue test caught, before the tests ran, that a
  new source file not yet compiled already used a code (the test scans the
  source tree, not the build).

## What went wrong, and how it was found

- A check of a stop's engagement compared `spring.value(0.05)` with the law's
  own `spring.value(0.35 - 0.30)`; in floating point `0.35 - 0.30` is not 0.05,
  and at a table knot the two values differ. Found by reading the test before
  building; the expectation now uses the same arithmetic.
- A test needed two upward zero crossings of an oscillation whose period
  (0.23 s) exceeded what the run covered; found by computing the period before
  building, run lengthened.
- An end-slope index in the PCHIP code had a needless guard for a case that
  cannot reach that line; simplified.

## Verification

Derivatives: every element's `law()` against central differences (h = 1e-7,
points away from table knots and stop engagement), to 1e-6 relative.
Closed-form motions:

| Test | Measured | Bound and its reason |
|---|---|---|
| preloaded spring, static deflection `-(m g - P) / k` | 1e-17 m | 1e-9 |
| torsion-spring pendulum, period `2 pi sqrt(I / (k + m g L))` | 2.3e-6 relative | 1e-4; the measured value is the amplitude correction `theta0^2 / 16` times gravity's share of the stiffness, as predicted |
| Coulomb friction, stopping distance `m v0^2 / (2 F)` | 8.2e-6 m | 5e-5; the regularisation adds at most about `2 v_f m v_f / F` = 2e-5 |
| limit stop, rest at `m g / k` past the limit | 1e-17 m | 1e-9 |
| rotational spring against the joint torsion spring, same hinge | 8e-17 rad | 1e-12 |
| bushing, periods `sqrt(m / k)`, `sqrt(I / k_r)` | 2.9e-11 s, 1.3e-11 s | 1e-6 |
| user force, constant push | 2e-16 | 1e-12 |

385 tests pass.

## Open ends

- Generalized stiffness and damping matrices from these laws: tasks 3.5 and
  3.7, starting from finite differences.
- Bushings with coupling between axes (a full 6 x 6 matrix): add when a vehicle
  needs it.
- The rotational spring's twist is single-turn, in (-pi, pi]; a multi-turn
  torsion element would need the angle unwrapped over time.

## For the book

- Why tables should be monotone cubics, with a figure of a natural spline
  overshooting a damper table where PCHIP does not.
- Regularised friction: the trade between the regularisation velocity, the
  creep it allows, and the stiffness it adds to an explicit integrator.
- The derivatives of a force law as the bridge to statics and implicit
  integration (D24).
