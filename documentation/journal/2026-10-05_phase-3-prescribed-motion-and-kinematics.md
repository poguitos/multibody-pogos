# Phase 3: prescribed motion and kinematic analysis

- **Dates:** 5 October 2026
- **Plan:** tasks 3.2 and 3.6; decision D15
- **Commits:** see the index
- **Written:** at the time

## Goal

- **3.2** Drive a joint coordinate or a point as a function of time,
  consistent at position, velocity and acceleration level. *Done when a
  driven pendulum follows a sine exactly and reports the driving torque.*
- **3.6** Solve a mechanism with no degrees of freedom along a prescribed
  motion, replacing the trick of editing a constraint's target. *Done when the
  bump sweep runs through it with identical results.*

## Starting point

Both were largely done in task 2.7, when drivers became constraints with a
time-dependent target (D15): `JointDriver`, the primitives' target functions,
and `Kinematics` with t as the driving parameter. The driven pendulum test
already existed. What was missing: a driven *point* in dynamics, any check
that the new sweep reproduces the old method, and velocities and
accelerations in kinematic analysis.

## What was done

- **A point driven along a circle.** A free body's centre held on a
  horizontal circle by three `Dot2` drivers, one per coordinate. Expected in
  closed form: the point follows the circle, the body does not turn (the
  constraint force acts through the centre of mass and the drivers' Jacobian
  rows have no angular part), and the constraint force is
  `m a - m g = (-m w^2 R cos wt, -m w^2 R sin wt, m g)`.
- **The old method, rebuilt.** A second corner whose bump target is a
  variable edited before each solve, solved from the same starting points as
  the sweep. The position equations are identical, so the Newton iterations
  are too.
- **`Kinematics::velocities()` and `accelerations()`.** At the solved
  configuration, `J v = nu` and then `J a = gamma(q, v, t)`, solved by a
  complete orthogonal decomposition: exact when J has full column rank, even
  with redundant rows, least-norm otherwise. With t the bump travel these are
  the derivatives of the suspension's motion with respect to wheel travel,
  from which motion ratios (spring travel per wheel travel) follow.

## Verification

| Check | Measured | Bound and its reason |
|---|---|---|
| driven point on its circle over 1 s | 1.1e-13 m | 1e-10: the projection holds `|phi| <= 1e-10`, and phi is the error |
| its angular velocity | 0 exactly | no torque, no coupling in M |
| its constraint force against `m a - m g` | 2.5e-15 N | 1e-8 of `m g` |
| sweep against the edited target, 11 points | 0 (bitwise) | 1e-12 |
| wheel-centre vertical velocity against 1 | 1.1e-16 | 1e-9 |
| its acceleration against 0 | 2e-17 | 1e-8 |
| velocities against central differences, h = 1e-4 | 6.8e-8 (1.5e-8 relative) | 1e-6 relative: truncation of order h^2 times the third derivative, plus 1e-14 / (2 h) from the solves |

371 tests pass.

## Lessons

- A task that turns out to be done still needs its "done when" checked
  against evidence: here the evidence for 3.6 did not exist until the old
  method was rebuilt in a test.

## For the book

- t as a driving parameter: the same constraints give a kinematic sweep, its
  velocities and accelerations, and a driven dynamic simulation.
- The driven point as a textbook check of constraint forces.
