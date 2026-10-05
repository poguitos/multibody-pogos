# Phase 2: the simulator, and the vehicles on the kernel

- **Dates:** 4 October 2026
- **Plan:** task 2.7 and most of 2.7b; finding F10
- **Commits:** `24faade`
- **Written:** reconstructed on 5 October 2026 from the commit, the plan and
  the session record

## Goal

Delete the legacy single-body code (2.7), and move everything that still ran
on the legacy multibody system onto the kernel (most of 2.7b).

## What was done

- `dynamics_og.hpp` and `solvers/solver.hpp` deleted; their analytic tests
  became kernel cases (`test_kernel_analytic.cpp`): a free body, Euler's
  equations, the torque-free symmetric top, a pendulum rod's force, the
  elliptic-integral period of a large swing, a hinged bar's reaction.
- **The force bridge** (`kernel/forces`): force elements keep their interface,
  world forces at body origins; one backward pass turns them into
  generalized forces without forming Jacobians.
- **The simulator** (`kernel/simulator`): a `System` (model, constraints, force
  elements) and a `Simulator` with RK4 and semi-implicit Euler, three
  callbacks, and projection after every step.
- **Drivers.** `JointDriver` holds a revolute or prismatic coordinate at s(t);
  its multiplier is the force the drive needs. A driven pendulum's torque is
  checked against `I q'' + m g L sin q`.
- **Kinematic analysis with t as a parameter.** A suspension's bump driver
  holds the wheel centre at its nominal height plus t, so a bump sweep solves
  `phi(q, t) = 0` for successive t (the `Kinematics` helper) instead of
  editing a constraint's target between solves.
- **Steering as a body (F10).** A steered axle with linkage suspension gets a
  light rack sliding along the chassis's lateral axis, carrying the tie rods'
  inner points and driven to the commanded travel; each wheel's toe follows
  from the geometry. The rack travel per radian of toe is calibrated per
  corner by moving the rack 5 mm with the chassis pinned.
- Ported onto the kernel: suspension kinematics, the double-wishbone,
  McPherson and multilink corners, the vehicle template, the simple vehicle,
  the drivetrain, the suspension optimiser and the tests of the force
  elements, vehicles, steering, quarter car and lap validation.
- The projection gained a backtracking line search (up to 12 halvings), for
  starts far from the constraints.

## What went wrong, and how it was found

- **A build that could not succeed was stopped.** With MSVC's precompiled
  header, a header edited while a build is running is not seen by that build,
  so the tests compiled against the old header. Rule since then: do not edit
  sources that a running build has yet to compile.
- An ambiguous `Simulator` (legacy and kernel in one test) needed qualifying.

## Verification

The ported tests keep the same physical checks, and some became sharper: two
comparisons tightened from 0.05 degrees to 1e-8 rad, and an aerodynamic check
from "non-zero" to the exact moment. 516 tests passed.

## Measurements

The double-wishbone sedan: 292 us per step on the kernel against 739 us on the
legacy path in the same run (the target of task 2.10 was 250 us). This run was
not pinned to a core; see the performance entry of 5 October for why that
matters.

## For the book

- t as a driving parameter unifies kinematic sweeps and driven dynamics: the
  same constraint serves both.
- The steering rack: replacing an edit of the model between steps by a
  physical mechanism, so the steering rate enters the equations.
