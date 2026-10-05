# Phase 3: contact with a plane

- **Dates:** 5 October 2026
- **Plan:** task 3.8; finding F14; decisions D24 (laws with derivatives), D29
- **Commits:** this task's commit (the index gives the hash)
- **Written:** at the time, in a cloud session

## Goal

Point and sphere against a plane, with penalty stiffness, nonlinear damping
and regularised friction. *Done when a block on an incline slides at the
correct friction angle.*

## Starting point

The only contact in the engine was the tyres' vertical spring
(`TireContactForce`, `FullTireForce`), against a horizontal road at a fixed
height in the old Y-up frame. The force library of task 3.1 had set the form
of an element: a law in the element's own coordinates with its derivatives,
and a potential where there is one. The joint-coordinate friction of 3.1 was
already regularised by tanh. Statics (3.5) could now find resting states.

## What was done

1. `PlaneContact`: spheres of a body (radius 0 for points) against the XY
   plane of a frame on any body. For each sphere the depth, the contact
   point, the rate of penetration and the slip velocity from the velocities
   of the two material points at the contact point; the normal force
   `k d^e + c s(d / d_c) d_dot`, clamped at zero; friction
   `mu F_n tanh(|u| / v_s)` against the slip, written through `tanh(x) / x`
   so that it is smooth at zero slip; the force on the body and the
   reaction on the plane's body. `normal_law`, `friction_law`,
   `potential_energy`, and `contact(states, i)` for outputs.
2. Codes F040 (parameters) and F041 (contact points).
3. Six tests (`tests/forces/test_plane_contact.cpp`).

## Decisions

D29: penalty contact as a force element; the damper ramped from first touch
(continuous force) rather than Hunt and Crossley's law (an option for
impacts later) or a linear damper (a jump at contact); tanh friction as for
the joints, accepting creep in place of sticking, rather than a bristle
model, which needs a state per contact (task 4.1 makes that possible).

## What went wrong, and how it was found

- **A test asked the law for a pull.** The derivative test first used a
  separation rate of 0.05 m/s at a depth of 0.2 mm, where the damper's
  1.56 N exceeds the spring's 0.57 N: the force is (rightly) clamped at zero,
  and the test's own requirement that the force be positive failed. The rate
  became 0.005 m/s, and the comment says why.
- **A friction-derivative check too clever by half.** The first version
  handled zero slip with a one-sided difference in a single expression;
  rewritten as a plain central difference at slips away from zero, plus a
  separate check that friction vanishes at zero slip.

## Verification

| Check | Measured | Bound, and why |
|---|---|---|
| Normal law's derivatives and the spring's potential against central differences, depths in and beyond the ramp, rates both ways | within | 1e-6 relative: truncation and roundoff of the differences |
| Friction law's slope; at most `mu F_n`; `mu F_n` once sliding fast | within | 1e-6; 1e-15 |
| Incline one degree steeper than atan(0.5): acceleration against `g (sin th - mu cos th)` | 2.5e-13 relative (3 degrees: 6.3e-14) | 1e-6: Coulomb's law exact at 100 v_s, the pitch settled |
| Incline one degree shallower: creep speed against `v_s atanh(tan th / mu)` | 1.1e-16 relative | 1e-6: the steady state is exact |
| Hertzian sphere at rest, depth `(m g / k)^(2/3)` by statics | 9.6e-15 m | 1.7e-11 m: tolerance x m / contact stiffness |
| Undamped sphere from 0.1 m: rebound height | 3.6e-9 relative | 1e-5: the force's missing second derivative at first touch, across one step |
| Same, energy at the deepest point | 1.9e-6 relative | 1e-4: the deepest point is sampled once a step |
| Ball on a sprung plate (plane on a moving body): spring and contact by statics | 0 and 3.3e-16 m | 1e-9 m |

417 tests pass with GCC 13 (411 before).

## Measurements

[data/2026-10-05_contact.md](data/2026-10-05_contact.md). The incline runs
take 30 ms each in the container (5,000 RK4 steps). Statics reports the
Hertzian sphere's five free directions (it can roll, slide and spin).

## Lessons

- With regularised friction, "sticks below the friction angle" becomes "creeps
  at a speed set by the regularisation", and that speed has an exact formula
  on an incline. Testing the formula makes the regularisation part of the
  specification rather than an unmeasured approximation.
- The stability limit that friction's regularisation imposes on an explicit
  integrator (`dt < 2.8 v_s m / (mu F_n)`) belongs in the help, next to the
  parameter that causes it.

## Open ends

- Hunt and Crossley damping, for impacts with a known restitution (kerbs,
  Phase 9); a bristle (LuGre) friction state once the integrator interface
  takes element states (task 4.1).
- Contact against a height field or a road surface (Phase 7), and between
  bodies' shapes beyond a plane (not planned yet).
- The tyres still use their own vertical contact in the Y-up frame (task 7.1
  moves them to ISO 8855; they could then reuse this element's normal law).

## For the book

- The friction angle as a test, and creep as the regularised form of
  sticking: one closed-form line for each side of atan(mu). Figure: the
  block's speed over time at five angles around atan(mu), sliding above and
  creeping below.
- Why the damper is ramped from first touch: a linear damper's jump and its
  pull before separation, with a force-depth plot of one impact.
