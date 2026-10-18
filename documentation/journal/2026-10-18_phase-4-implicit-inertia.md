# Phase 4: the inertia's velocity products implicit in the Euler steps

- **Dates:** 18 October 2026
- **Plan:** task 4.7's open end (the author asked for it before task 4.8);
  decision D38
- **Commits:** see the index
- **Written:** at the time

## Goal

Task 4.7's long runs found semi-implicit Euler and the linearly implicit
step blowing up on a spinning top: both took the inertia's velocity
products (the Coriolis, centrifugal and gyroscopic terms) explicitly. The
linearly implicit step is the real-time integrator, and a vehicle with
spinning bodies (a motorcycle's wheels) would meet the same. Make both
steps stable on spinning bodies without spoiling anything else.

## Starting point

The steps solved one linear system for the velocities' correction delta
with J delta = 0 (the motions the constraints allow), the projection after
the step taking up the rest; the inertia's velocity products, part of the
explicit accelerations a, were not in it.

## What was done

- **The derivative.** `bias_force_velocity_derivative`: D = d b / dv by
  linearizing RNEA along each velocity over that joint's subtree and back to
  the root, exact because b is quadratic in v. Checked against unit-step
  central differences of RNEA on every joint type (5.6e-14 of 64).
- **First version:** D whole in both steps (W gains h D, r gains -h^2 D a),
  W solved by a new LU variant of the constrained solve, W's symmetric part
  checked for positive definiteness (K045). A free body's nutation then fell
  exactly as backward Euler's (a test with the exact value), and the whole
  suite passed.
- **The cost:** the linearly implicit step went from 130 to 186 us on the
  sedan. The derivative was made to skip the joint model where its motion
  subspace is constant (a new `JointModel::constant_motion_subspace`) and to
  walk each joint's subtree and path only, and the LU solve to work on all
  its right-hand sides at once.
- **Then a failure:** with those changes the ten-second constraint test
  failed for both steps on the four-bar, the energy running away at 9 s.
  The previous version had passed by luck: a probe of the energy over 100 s
  at 5, 2 and 1 ms showed the four-bar gaining energy until it blew up at
  every step size, where the explicit treatment had lost it.
- **Four more treatments** measured on the four-bar, the heavy top and two
  free bodies, one flat and one long (data, section 3): the skew part of D;
  a Coriolis matrix B frozen at the step's start (B v = b, B + B^T = dM/dt:
  a new algorithm, built from the bodies, verified, and removed again after
  the measurements); and each of B and D with the new velocities put on the
  constraints (J v+ = nu) instead of J delta = 0.
- **Kept** (D38): D, with the new velocities on the constraints. Tests: the
  nutation of flat and long bodies equals backward Euler's; the closed
  four-bar's energy never rises; the heavy top returns to the 1000-second
  test for both steps.

## Decisions

- **D38**: the inertia's velocity products linearly implicit through their
  exact derivative, and the new velocities on the constraints, in both
  Euler steps.

## What went wrong, and how it was found

- **A green suite hid an instability.** The first version passed all 463
  tests; a later change of rounding (the blocked solve) made the
  ten-second test fail, and only then did a probe show the energy growing
  at every step size. A test that passes once near an instability says
  little: the energy over a long run, at several steps, is what showed it.
  The four-bar test now checks that the energy never rises.
- **The energy argument was right for the wrong system.** For a tree, D's
  symmetric part (dM/dt in true coordinates) is what can feed energy in, and
  a frozen Coriolis matrix provably cannot: it seemed the cure. It made the
  four-bar gain energy too. The four-bar is closed: the correction kept J
  delta = 0, so the new velocities carried the explicit part of the
  constraints' own velocity terms, which the implicit inertial terms no
  longer balanced. Putting the new velocities on the constraints made both
  B and D stable there; the free bodies then decided between them.
- **The theoretically safe choice was wrong for free bodies.** With body-
  frame velocities, a frozen Coriolis matrix (energy-safe) damps a flat
  body's wobble 50 times more than backward Euler and makes a long body
  spinning about its axis tumble end over end, its spin from 30 to 0.002
  rad/s in 5 s: the minor-axis instability of a dissipating body, which a
  satellite shows and a simulation must not invent. The exact derivative
  damps both as backward Euler does and keeps the spin.
- **A test that subtracted two references.** The Coriolis matrix's test
  computed `rnea(...) - rnea(...)` and `crba(...) - crba(...)`; both return
  references to the same storage, so the differences were zero and every
  model failed. The derivative's test, written earlier, copied its results.
- **The cost of the first version** (+43 % on the real-time step) was
  measured before anything was committed; optimized, +23 % remains.

## Verification

Three test cases changed or added, in `tests/kernel/test_kernel_invariants.cpp`
(the derivative against RNEA), `tests/kernel/test_kernel_integrators.cpp`
(nutation, flat and long, both steps: 0.149716 against 0.149716 predicted,
w3 kept, energy falling every step) and `tests/kernel/test_kernel_drift.cpp`
(the closed four-bar for 20 s at 5 and 2 ms, both steps: energy never above
the start; and the 1000-second test with the top again for both steps);
464 tests, no compiler warning. Numbers in the data file
`2026-10-18_implicit-inertia.md`.

## Open ends

- The remaining cost: W's LU and the Cholesky of its symmetric part could
  share work, and the derivative could skip the columns of joints with no
  spinning body below them, at the price of an approximation.
- The two steps are first order and dissipative: the four-bar at 5 ms lost
  14.7 J of 17.8 in 100 s. That is their nature, not a defect; RK4 or Radau
  IIA where energy matters.

## For the book

- Linearization and energy: why the exact derivative, a frozen Coriolis
  matrix and the explicit terms behave differently, and what the
  constraints add.
- The minor-axis rule as a numerical artefact.
- A passing test near an instability, and the probe that showed it.
