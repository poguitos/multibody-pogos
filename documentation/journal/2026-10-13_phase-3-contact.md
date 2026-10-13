# Phase 3: simple contact

- **Dates:** 13 October 2026
- **Plan:** task 3.8; decision D28
- **Commits:** see the index
- **Written:** at the time

## Goal

Point and sphere against a plane, with penalty stiffness, nonlinear damping
and regularised friction. *Done when a block on an incline slides at the
correct friction angle.*

## Starting point

The vehicles' tyres had their own vertical contact (`TireContactForce`,
`FullTireForce`), and joint friction a tanh regularisation (task 3.1). There
was no general contact element.

## What was done

- `PlaneContact` (`forces/contact.hpp`): spheres or points on one body
  against a marker's plane on another; normal force by ADAMS's IMPACT law
  (power-law stiffness, damping ramped over a depth along a cubic step,
  never pulling); friction with a coefficient stepping from 0 to its static
  value and down to its dynamic value over two transition speeds; equal and
  opposite forces on both bodies at the contact point; `normal_law` and
  `friction` with their derivatives; `penetrations()`.
- Codes F040, F041; kernel document, help index, catalogue.

## Decisions

- **D28**: the IMPACT law, and a speed-only friction coefficient with
  static and dynamic levels, rather than a full-strength damper or a bristle
  model.

## What went wrong, and how it was found

Nothing failed. One bound was derived again before the first run: the
momentum check of the ball on the plate was first written at 1e-10, which
assumes RK4 conserves momentum exactly; it does so only to its truncation
error, because it combines body-frame velocities from stages at different
orientations. The estimate (w dt)^5 per contact step over 700 steps, about
2e-9, set the bound at 1e-8; measured 2e-11.

## Verification

Four test cases, 417 tests in all:

- **Laws:** both derivatives of the normal force at twelve points (inside
  and beyond the damping ramp, approaching and parting), and the friction
  coefficient's on both branches, against central differences; the normal
  force starts from zero (2e-13 N at a penetration of 1e-12 m moving at
  1 m/s).
- **Resting ball:** a Hertzian ball sinks (m g / k)^(2/3), found by statics,
  to 1.1e-14 m.
- **Ball on a free plate:** off-centre, sliding and bouncing; the pair's
  linear and angular momentum kept to 1.2e-10 and 1.8e-11.
- **The incline (done when):** 1 degree below the friction angle atan(0.5),
  the block creeps at the speed where the friction step reaches tan(th),
  0.874557 mm/s predicted by bisection on the cubic, to rounding (it never
  exceeds v_static); 1 degree above, it slides, and once past v_dynamic its
  acceleration is g (sin th - mu_dynamic cos th) = 1.06106 m/s^2 to 5e-15.

## Measurements

In [data/2026-10-13_contact.md](data/2026-10-13_contact.md).

## Lessons

- With regularised friction, "sticks" is a statement about a creep speed,
  and the creep speed is predictable. Testing that number, rather than
  "the block does not move much", makes the test sharp.
- Momentum is a good check of an element that acts on two bodies, as long as
  the bound accounts for the integrator.

## Open ends

- True sticking (a bristle model, or a stick-slip state with events):
  needs the extra states of Phase 4 and the events of task 3.9.
- Contact between general shapes (spheres on spheres, boxes, meshes), and
  the tyre on uneven ground, belong to later phases (7, and the terrain work
  after them).

## For the book

- The IMPACT law and why its damping ramps.
- Friction regularisation: the creep speed as a consequence one can predict
  and test, and the friction angle as the test of static against dynamic
  friction.
