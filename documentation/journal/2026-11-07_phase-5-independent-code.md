# Phase 5: comparison with independent codes (Pinocchio and Project Chrono)

- **Dates:** 7 November 2026
- **Plan:** task 5.3 (independent code); decision D46
- **Commits:** see the index
- **Written:** at the time, on the author's laptop (Windows, MSVC), by
  Claude working with the author

## Goal

The plan's task 5.3: "Compare accelerations on random trees against
Pinocchio to 1e-8. Compare constrained mechanisms against Project Chrono.
Done when the comparison scripts run in CI."

Every check so far had been written inside this project: closed forms,
invariants, manufactured motions, the IFToMM problems. A code written by
other people, with its own conventions, catches what those cannot: a
convention misread the same way in the code and in its tests.

## Starting point

Tasks 5.1 and 5.2 had been done by a cloud agent on Linux (D43 to D45). Their
work built here with MSVC with one warning, C4458 in
`tests/kernel/iftomm.hpp` (a parameter `c` hiding the governor's member
`c`), which GCC does not report; renamed, 508 tests pass. No conda on the
laptop: Pinocchio's PyPI package (`pin`) has no Windows wheels, and PyPI's
`pychrono` is an unrelated library, so with the author's agreement Miniforge
26.9.4 was installed per user (`C:\Users\margk\miniforge3`, not on the
PATH, the installer's checksum checked after a first download came short)
and an environment `mbd-crosscheck` made with Pinocchio 4.1.0 and PyChrono
10.0.0 from conda-forge.

## What was done

1. **The exporter.** `mbd_crosscheck` (`crosscheck/`) writes, as JSON, random
   models and this library's answers, with 17 significant digits so that
   every number reads back exactly. It uses the tests' own generators
   (generic joint frames, tilted inertias, offset centres of mass). Built on
   request, like `mbd_bench`.
2. **Pinocchio** (`compare_pinocchio.py`). 40 random trees, 2 to 24 bodies
   each, every new body hung from a randomly chosen earlier one on a random
   joint of any of the eight kinds, gravity in a random direction; three
   random states each. The script rebuilds each tree in Pinocchio: our
   revolute and prismatic are its RZ and PZ, our universal, cylindrical and
   planar are composites of its single-axis joints, spherical and free are
   its own (the free joint's velocities in the other order); a fixed joint
   merges its body into the parent. The frames map through one rule: the
   pose of our body frame in Pinocchio's joint frame is the inverse of
   X_CJ. ABA, RNEA and the placements agreed at once (2e-13 and below), the
   mass matrix did not (see below). With the joints added to Pinocchio
   depth first, everything agrees within 2.1e-13 of Pinocchio's values.
3. **Chrono, first attempt.** The mechanisms: random trees of 3 to 8
   bodies, closed by one to three constraints of the six kinds Chrono has a
   link for (spherical, revolute, universal, cylindrical, prismatic,
   distance), each between a random body and another or the ground, its
   markers made to coincide at a random configuration; a draw with redundant
   equations or no motion left is drawn again (12 mechanisms took 44
   draws). Every body is a Chrono body; every joint of the tree and every
   closure is a link. The plan was: accelerations and joint reactions at a
   random state from Chrono's assembly at the acceleration level, and the
   motion against Chrono's implicit integrator.
4. **Measuring Chrono before trusting it**, on a pendulum with a closed
   form. Its assembly's accelerations left out the centripetal term
   altogether when the pendulum turned; at rest they were exact. Its
   integrators converged at first order, HHT included. The causes, one by
   one: the integrators start from the accelerations stored in the bodies,
   zero unless an assembly has set them; the assembly leaves the velocities
   advanced by its own small step; a probe that put them back did not,
   because PyChrono returns the body's own vector, not a copy. With the
   assembly done at rest and the velocities really put back, HHT converged
   at second order (ratios 3.66 to 3.95).
5. **The motion.** From rest under gravity for 0.5 s; ours by RK4 at 1e-4 s
   (its change from 2e-4 s at most 5e-10), Chrono's by HHT with alpha = 0
   at 2e-4 and 1e-4 s. On all 12 mechanisms Chrono's difference from ours
   falls fourfold when its step halves, and Richardson's extrapolation of
   its two runs agrees with ours to between 1.4e-12 and 6.2e-9: Chrono
   converges to our motion at its own order.
6. **At rest, second attempt.** Chrono's assembled accelerations and its
   links' reactions differed from ours by 1e-5 to 3e-4 relative, while the
   motions agreed to 1e-10. A pendulum whose joint was opened by 1e-15 m
   changed its angular acceleration by 5.3e-4 rad/s^2: the assembly holds
   the constraints through a step of 1e-6 s, so a residual of one rounding
   error, divided by the step squared, becomes 1e-4. Any configuration
   rebuilt in Chrono's own frames has such residuals. Chrono's integrators
   do not give better accelerations or reactions either (they oscillate
   with alpha = 0, converge at first order to about 1e-4 with alpha =
   -0.2). So the script now reads what the assembly sets up, Chrono's mass
   matrix, constraint Jacobian and forces (the system descriptor's matrix
   blocks, written to files with 13 significant digits), sets the
   constraints' right-hand side to its exact value at rest, zero, and
   solves those equations itself. Each link's rows give its wrench on its
   two bodies. Then accelerations, the closures' loads on each body and
   each joint's reaction agree with ours to 1.4e-12, the files' precision.
7. **A comparison that can fail.** One inertia entry raised by one part in
   a million in what Chrono builds: the at-rest part moved by 2e-9, which
   the first tolerance (1e-8) let through; at 1e-10 it fails, and so does
   the motion (its extrapolated difference 7.9e-10 against 5.3e-9 at h/2).
8. **CI.** A Linux job, `crosscheck`, installs the environment with
   micromamba from `crosscheck/environment.yml`, builds `mbd_crosscheck`
   with the tests off, and runs both scripts.

## Decisions

D46: the comparison as JSON written by the library and scripts in each
code's own language; Pinocchio's joints added depth first; Chrono's
equations at rest solved from its own matrices, and its motion checked by its
convergence to ours; the environment from conda-forge, pinned, NumPy on
OpenBLAS.

## What went wrong, and how it was found

- **Pinocchio's mass matrix was wrong on 37 trees of 40, and the cause was
  Pinocchio's assumption, not our mapping.** ABA and RNEA agreed to rounding,
  and both depend on the same mass properties, so the frames and inertias
  were right. The CRBA differences reached 0.66 relative. A probe built the
  mass matrix from Pinocchio's own RNEA (gravity off, v = 0, a = e_k): it
  differed from Pinocchio's own CRBA by up to 7.7, while ours matched it to
  2e-14. Tree 4 passed; small chains and stars of every joint, composite
  ones included, all agreed. The one difference left was the numbering:
  Pinocchio's CRBA writes each row of the matrix as one block over the
  subtree's velocities, which assumes each joint's subtree takes the ids
  just after it. Our generator hangs each new body from any earlier one.
  All 37 trees not numbered depth first differed, all 3 that were agreed.
  `addJoint` does not check it. The script now adds the joints depth first
  and maps the coordinates; our own CRBA works for any parents-first
  numbering, as it should.
- **Chrono's integrators at first order.** HHT, the trapezoidal rule and
  Newmark all converged at first order on the pendulum, with or without
  motion at the start, and on a mass on a spring with no constraint at all.
  So the cause was not the constraints. Setting the spring's initial
  acceleration by hand gave second order: the integrators start from the
  stored acceleration, zero by default.
- **The assembly fixed the order, but the error stopped falling at 1e-6.**
  A dump of the spring's state after the assembly showed the velocity at
  -7.5e-6 m/s: the acceleration times the assembly's step (1e-6 s), left in
  place. A first probe that saved and restored the velocities reported that
  the assembly had moved nothing, and the plateau stayed: PyChrono's
  `GetLinVel()` returns a reference to the body's own vector, so the saved
  value moved with it. Copying the numbers fixed it.
- **Chrono's accelerations left out the velocity products.** A pendulum at
  3 rad/s had its linear acceleration off by exactly omega^2 r. The
  assembly differentiates velocities that satisfy the constraints'
  velocity equations at the start, so the centripetal term never enters.
  This is why the comparison starts from rest.
- **The at-rest comparison failed by 1e-4 while the motion agreed to 1e-10.**
  Opening the pendulum's joint by 1e-16 to 1e-12 m made the error grow in
  proportion, 5.3e-4 per 1e-15 m: the residual over the step squared. The
  matrix blocks showed it directly: the right-hand side b held -1e-9, the
  1e-15 m gap over 1e-6 s.
- **NumPy stopped Python on a 3 x 3 product.** Exception 0xc06d007f (a
  delay-loaded DLL missing a function) at the first `@` after PyChrono was
  imported; then the same without PyChrono. Conda had chosen MKL 2026.1.0
  for NumPy's BLAS; `compare_pinocchio.py` had run only because it never
  reaches BLAS. Switched to OpenBLAS and pinned in `environment.yml`.
- **Chrono counts its equations only after `Setup()`**: the first count
  check reported 0 equations against 26.
- **The first at-rest tolerance was too loose to matter.** 1e-8, copied
  from the plan's figure for Pinocchio, let through a change of one part in
  a million in an inertia (2e-9). The floor is the files' 13 digits (1.4e-12
  seen); the tolerance is now 1e-10.

## Verification

- `compare_pinocchio.py`: 40 trees, 563 bodies (cylindrical 79, fixed 71,
  free 79, planar 63, prismatic 67, revolute 82, spherical 62, universal 60),
  120 states. Largest relative differences: ABA 2.1e-13, RNEA 1.4e-15, CRBA
  1.1e-15, placements 6.2e-16. Tolerance 1e-8, the plan's; the difference is
  measured as the largest entry of the difference over 1 + the largest of
  Pinocchio's value, relative for large values and absolute for small ones.
- `compare_chrono.py`: 12 mechanisms of 3 to 8 bodies, 4 to 31 degrees of
  freedom, every joint kind in the trees and every closure kind. At rest:
  angular accelerations within 1.0e-12, centre-of-mass accelerations
  4.8e-13, the closures' loads on each body 1.4e-12, joint reactions 3.6e-13
  (tolerance 1e-10: the floor of 13-digit files is near 1e-12, and a change
  of one part in a million in an inertia moved them by 2e-9). The motion:
  Chrono's difference from ours at h = 2e-4 s over that at h/2 must be
  between 3 and 5 (the trapezoidal rule's error falls as h^2; seen 3.98 to
  4.02 from the printed differences), and the extrapolated difference below
  a tenth of the difference at h/2 (seen 1/12000 to 1/88), unless it is
  below ten times our own motion's
  error estimate plus Chrono's Newton tolerance. The script also checks that
  Chrono has the number of equations we have, that its forces at rest are
  gravity's, and that each link's equations reach only its own bodies.
- Both scripts were shown to fail: the CRBA before the depth-first order,
  and the perturbed inertia (data, section 6).
- `scripts\build.ps1 -Test`: 508 of 508 pass; no warning.

## Measurements

All in `documentation/journal/data/2026-11-07_independent-code.md`: both
runs, the CRBA diagnosis, the pendulum and spring measurements of Chrono's
assembly and integrators, the accelerations and reactions along the motion,
the perturbed inertia, and the BLAS failure. `compare_chrono.py` takes
115 s on the laptop.

## Lessons

- Before comparing with another code, measure that code on a case with a
  closed form. Three of Chrono's behaviours (no velocity products in the
  assembly, a zero starting acceleration, the residual over the step
  squared) would each have been read as our error.
- When two algorithms of the other code agree with ours and a third does
  not, compare that code with itself first (its CRBA against its own RNEA).
- A comparison checked by convergence needs no guessed tolerance: the other
  code's own error, measured by halving its step, sets the scale.
- Copy values out of a wrapped object before changing the object.

## Open ends

- The comparison covers gravity and constraints only: no force elements, no
  driven joints, no contact. Force elements could follow when a task needs
  them (Chrono has springs and bushings).
- Pinocchio's depth-first assumption is stated nowhere we found; it may be
  worth reporting upstream, which is the author's call.
- CI's result on Linux is recorded when the run is seen (the index row's
  commits).

## For the book

Part VII, a section "Comparison with independent codes": mapping
conventions between codes (frames, joint coordinates, the order of
velocities); the trap of a numbering assumption in the other code; measuring
the other code before trusting it; the residual over the step squared as a
worked example; Richardson's extrapolation as the test that a second code
converges to the first. A figure: Chrono's difference from ours against its
step, falling as h^2, and the extrapolated difference below it.
