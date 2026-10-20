# Phase 5: analytic cases and manufactured trajectories

- **Dates:** 20 October 2026
- **Plan:** task 5.1 (analytic cases, with the D42 additions); decisions D42,
  D43
- **Commits:** see the index
- **Written:** at the time, by a cloud agent (Claude) working from the
  repository alone, on a Linux machine; the Windows build and timings were
  not run in this session

## Goal

The plan's task 5.1: "Torque-free body with unequal inertias, heavy top,
pendulums, bead on rod, spring-mass systems. A rigid body on three bushings
against the frequencies and mode shapes of eig(K, M) in closed form, with the
modal energy distribution (each mode's kinetic energy by coordinate) added to
the linearisation and checked against the same closed form. Manufactured
trajectories: a smooth motion prescribed for random trees of every joint type
and for closed loops, its driving forces from inverse dynamics (with the
constraint forces for loops), and every integrator required to return that
motion at its order (D42). Done when each matches to a stated tolerance."

## Starting point

Several of the cases existed already, from Phases 2 and 4:

- pendulums: the revolute pendulum's period and statics, the spherical
  pendulum, the universal joint pendulum, the double pendulum's slow mode
  (`test_kernel_joints_physics.cpp`); the rod's pull and the elliptic-integral
  period of a large swing, the hinged bar (`test_kernel_analytic.cpp`);
- the bead on a spinning rod, `r = r0 cosh(w t)` (`test_kernel_invariants.cpp`);
- spring-mass systems: a slider on a spring, the mass on a spring in the
  simulator, the damped mass in the linearisation, the two masses of the
  integrator tests;
- the torque-free *symmetric* top, and Euler's equations at one instant.

Missing: the body with three unequal moments, the heavy top, the mount case,
the energy shares and the manufactured trajectories. RNEA, CRBA and ABA were
checked against each other at single states only.

## What was done

In this order:

1. **The energy shares in `linearize()`** (D43): `Mode::energy_fraction`,
   `Re(conj(phi_k) (M phi)_k) / Re(phi^H M phi)` with the whole mass matrix,
   and the three largest in `summary()`.
2. **Manufactured trajectories** (`tests/kernel/test_kernel_manufactured.cpp`,
   new). A motion class that prescribes every joint in closed form (sines for
   scalar coordinates, a product of three exponentials for each rotation),
   with its rates and their rates; four models: a free-root tree with
   revolute, prismatic, fixed and universal joints; a spherical-root tree
   with cylindrical, planar, free and spherical ones; two identical chains
   (universal, spherical, prismatic) side by side, tied at their tips by a
   distance constraint; a parallelogram four-bar closed by a revolute. On the
   loops only some joints are driven and the drives and multipliers come from
   one least-squares solve of `B tau_a + J^T lambda = RNEA`.
3. **The first order check failed**, over 1 s (see below); exploring with
   tables of five steps over three spans led to two checks: one step from the
   exact state (order p + 1) and a span of 0.5 s (order p).
4. **The mount case and a line of masses** in
   `tests/kernel/test_kernel_linearization.cpp`: a 68.5 kg engine on three
   tilted bushings against eig(K, M) with K and M written out; four masses
   between five springs against the textbook frequencies, shapes and shares.
5. **The torque-free body with unequal moments and the heavy top** in
   `tests/kernel/test_kernel_analytic.cpp`: Jacobi's elliptic functions
   (sn, cn, dn by the descending Landen transformation, written in the test)
   and their period; the top's steady precession at the smaller root of
   `m g l = Omega (C w3 - A Omega cos theta)`, and its nod, released with spin
   only, to the root of the energy cubic.
6. Documentation: `docs/kernel.md` (linearisation, tests), the how-to guide on
   analyses (its example and test together), the plan, this record.

## Decisions

D43: the energy shares from the whole mass matrix; the prescribed motion's
form; loops driven at some joints with the constraint forces solved with the
drives; orders measured by one step and over a 0.5 s span; generalised-alpha
without the one-step check and Dormand-Prince on one tree without the span
check, with their reasons; the mounts without gravity.

## What went wrong, and how it was found

- **Forced motions are unstable, so a 1 s span amplified the errors past
  any order.** The first version measured every method over 1 s and ten of
  its checks failed, some with errors of order one (linearly implicit
  Euler: 54 at 2 ms and 61 at 1 ms on the spherical-root tree, "order"
  -0.17). The forces are fixed functions of time: nothing pulls a deviation
  back, and on a model with a free body mid-tree and a spherical root the
  fifth-order methods stop converging at 2e-12 to 1e-11 over 0.5 s, about
  1e5 times the rounding of the state. Tables at 0.25, 0.5 and 1 s showed
  every method settling at its order at 0.5 s once the step was short
  enough, and the one-step error, which no amplification touches, settling
  at long steps. Both are now checked.
- **Dormand-Prince did not settle on one model.** Over 0.5 s on the
  spherical-root tree its ratios were 2^7.07, 2^4.39, 2^4.31 and 2^4.70 down
  to 1.6 ms, below the 2^5 expected, while Radau IIA, also of fifth order,
  gave 2^5.0 on the same model. A defect in its handling of quaternions was
  the first suspicion: run on each joint type alone (spherical, free,
  revolute then spherical, revolute then free, planar, cylindrical,
  universal, two revolutes), it settled at 5.0 on every one. Its single step from
  the exact state on the tree itself gave 6.80, 6.58, 6.36 and 6.20 between
  steps of 100 and 6.25 ms (6.50 to 6.09 between 40 and 2.5 ms): its local order is right, approaching 6 from above as a next term
  should. The span's error is a sum of local errors carried to the end by
  different growths, in many coordinates, which can cancel in the leading
  term down to short steps; that this happens for Dormand-Prince and not for
  Radau IIA is a property of their error constants on this model, not
  established further. The span check leaves that one pair out, with the
  numbers in the test (D43).
- **Generalised-alpha's single step is of order 2, not 3.** Expected once
  seen: its acceleration-like variable is started at the true acceleration,
  which is not the one its recurrence needs, and the first step's velocity
  error is O(h^2); later steps damp it (Arnold and Brüls 2007). Its span
  order is 2.00 on every model.
- **My own floor on the one-step error was too high.** A check that the
  smaller error is above 1e-14 failed at 8.4e-15 (Dormand-Prince on the
  parallelogram at 5 ms). The rounding floor seen there is 3.8e-16, so the
  bound became 2e-15, with that reasoning in the test.
- **Normalizing shapes by their largest entry is ambiguous.** The line of
  masses' modes are symmetric or antisymmetric, with two entries of the same
  largest size; `linearize()` and the closed form picked different ones and
  the comparison failed by 3.3 with the frequencies right. Shapes are now
  compared up to a scalar factor (the part of the unit shape not along the
  closed form's).
- **A header edited during a build left a stale object.** I changed
  `linearization.hpp` (adding `energy_fraction` to `Mode`) while the first
  full build was running. The quarter-car test object compiled against the
  old layout but was written after the edit, so Ninja saw it as up to date;
  two quarter-car tests then read `lm.modes.size()` as 6148914691236517208.
  The garbage size pointed straight at a layout mismatch; touching the header
  and rebuilding fixed it. `CLAUDE.md` forbids exactly this edit for this
  reason (on Windows, where it was first met); it holds on Linux too.

## Verification

Tests added (499 registered with ctest on Linux, all passing; 490 before):

- `test_kernel_manufactured.cpp`, four cases: the motion's rates against its
  own central differences (1e-7, from h^2 / 6 times the third derivative);
  the forward dynamics at three instants on every model returning the
  accelerations (seen 2.1e-13, allowed 1e-9 (1 + |a|)) and, on the loops, the
  constraint forces (9.0e-15 of 9 to 14, allowed 1e-9 relative), the inverse
  dynamics' residual (4.9e-16, allowed 1e-12), the loops' constraints met to
  rounding, and the constraints carrying at least a tenth of the load (seen
  0.27 to 2.6 times the drives); one step from the exact state at order p + 1
  for six methods on four models (within 0.17, allowed 0.25); every method
  over 0.5 s at order p (within 0.08, allowed 0.25, which keeps the next
  term's share below 19 %), Dormand-Prince on the spherical-root tree left
  out. A hidden fifth case prints the tables.
- `test_kernel_linearization.cpp`, two cases: the engine on three bushings
  (frequencies 2.0e-11 relative, allowed 1e-8 from the differences' rounding,
  2e-16 / 1e-6; shapes 2.8e-10 and shares 1.6e-10, allowed 1e-7, the
  rounding divided by the relative gaps of the squared frequencies, at least
  0.37); the line of masses (4.4e-12, 3.3e-11, 3.5e-11, same bounds).
- `test_kernel_analytic.cpp`, three cases: the body with unequal moments
  (k^2 = 0.0712; angular velocity within 1.7e-12 rad/s of Jacobi's functions
  over 5 s and back within 1.1e-12 after one period of 3.079 s, allowed 1e-9
  from RK4's (w h)^5 / 120 per step; world angular momentum fixed to 1.2e-13
  of 0.39); the steady top (Omega = 1.507 rad/s; precession angle after 1.5
  s within 2.0e-12 rad and tilt within 5.3e-14 rad, allowed 1e-8); the
  nodding top (lowest cos(tilt) within 2.6e-14 of the root, three nods,
  allowed 1e-7 from the sampling parabola's error estimate, which the result
  shows to be very pessimistic).

CI's sanitizer job was also run on this machine before the push, in its
own configuration (GCC, address and undefined-behaviour sanitizers,
`test_core`, `test_model` and `test_kernel`): 166 of 166 tests passed in
329 s with no sanitizer report; the span test took 32 s of it. Not run
here: the Windows build (no MSVC on this machine) and the Doxygen reference
(Doxygen not installed); CI runs both.

*Correction, added at task 5.2's close (D45):* "no sanitizer report" was
read from the ctest summary, which shows nothing of a test that passes. The
sanitizer then recovered from undefined behaviour without failing the test,
and in task 5.2's run the full log held two such reports, from code already
there (one in the span test above). This run's full log was overwritten, so
whether it had them cannot be checked. CI's no-allocation job (Debug,
`EIGEN_RUNTIME_NO_MALLOC`) was not run here either; CI ran it.

## Measurements

No timings. The error tables and the margins are in
`documentation/journal/data/2026-10-20_analytic-cases.md`, with the commands.
The engine's six frequencies are 11.0, 13.6, 18.0, 23.8, 27.8 and 33.4 Hz;
the purest mode (18.0 Hz) has 95 % of its energy in the translation along
the body's Z axis, the least pure (23.8 Hz) only 36 % in any coordinate.

## Lessons

- An exact reference is not enough; what carries the error to the point of
  comparison matters. A manufactured motion needs a span short against the
  motion's instability, or a check that does not depend on it (one step).
- When one method misbehaves on one model, run it on the parts of the model
  alone, and measure its local error on the whole: together they separate a
  defect from a property of the problem in a few minutes.
- Do not edit a header during a build, on any platform.

## Open ends

- Why Dormand-Prince's span error on the spherical-root tree settles so
  late, when Radau IIA's does not: not investigated beyond the local order.
- The mount case with gravity (the preload's stiffness) has no closed form
  here; the regression harness (5.5) could hold its frequencies once
  checked against another code (5.3).
- The Windows build (MSVC, through `scripts/build.ps1`) was not run in this
  session; CI's Windows job is the check.

## For the book

- Part VII (validation): the manufactured trajectory as the strongest single
  check of a dynamics code, with the product-of-exponentials motion and the
  loops' inverse dynamics written out; the instability of forced motions and
  the two measures of order (one step, a span) as a figure from the tables.
- Part VII: modal energy shares with the whole mass matrix against the
  diagonal-only form, with the engine on its mounts as the example.
- Part X (testing a numerical code): a failing check that pointed at the
  reference's conditioning, not the code; and the stale object from a header
  edited during a build.
