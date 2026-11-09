# Validation report

Version 1, 9 November 2026: the end of Phase 5, milestone M1 ("a trustworthy
core"). The evidence that the solver computes what it should, gathered from
the tests, the benchmarks and the comparisons with other codes, each with
how independent its reference is, the agreement measured, and where to find
and repeat it. What is not validated yet is listed at the end.

The kernel's design is in [the kernel](kernel.md), its integrators' own
verification in [the integrators, verified](integrators.md), the HMMWV's
costs in [the HMMWV benchmark](hmmwv.md). Every number below comes from a
test or a program in the repository; the decisions (D-numbers) and the
journal entries (`documentation/`) say how each was obtained.

## Summary

| Area | Reference | How independent | Agreement | Where |
|---|---|---|---|---|
| Tree dynamics | Pinocchio 4.1.0, 40 random trees, 563 bodies, all eight joint types | Another code, other authors | ABA 2.1e-13, RNEA 1.4e-15, mass matrix 1.1e-15, placements 6.2e-16 (relative) | `crosscheck/`, CI |
| Closed mechanisms | Project Chrono 10.0, 12 random closed mechanisms | Another code, other formulation (absolute coordinates) | At rest 1.4e-12; in motion Chrono converges to ours at its order, within 6.2e-9 m | `crosscheck/`, CI |
| Closed-form motions | Jacobi's elliptic functions, a heavy top's precession and nod, pendulum periods, a bead on a spinning rod, a spring-mass | Mathematics | 1e-8 to 1e-14 | `test_kernel_analytic.cpp`, `test_kernel_joints_physics.cpp` |
| Manufactured motions | Smooth motions prescribed for trees of every joint type and for loops | Inverse dynamics against forward dynamics | Accelerations 2e-13, constraint forces 1e-15; every integrator at its order | `test_kernel_manufactured.cpp` |
| Benchmark mechanisms | IFToMM A01 to A05: closed forms, the Test Set for IVP Solvers, the library's published results | Published references | 1e-8 to 1e-15 against closed forms; 1.4e-9 rad against the test set; 2e-6 to 3e-5 m against library results | `test_kernel_iftomm.cpp` |
| Invariants | Finite differences, energy, momentum, d'Alembert, gravity as a potential's gradient | Mathematics | 1e-10 to rounding | `test_kernel_invariants.cpp`, `test_kernel_constraints.cpp` |
| Integrators | Exact solutions of each method's recurrence; measured orders | Mathematics | Every method at its order; energies equal to their recurrences' to 3e-11 | [integrators.md](integrators.md) |
| Statics, assembly, linearisation, kinematics | Closed forms; eig(K, M) in closed form | Mathematics | 1e-11 to 1e-16 | `test_kernel_statics.cpp`, `test_kernel_assembly.cpp`, `test_kernel_linearization.cpp` |
| Force elements, contact, events | Finite differences of each law; closed-form motions | Mathematics | Derivatives 1e-6; motions 1e-5 to 1e-15 | `tests/forces/`, `test_kernel_events.cpp` |
| A vehicle-sized stiff model | The HMMWV of Haug et al. (1999): explicit against implicit methods | Two families of method, one model | With joints 2.2e-7 m; with bushings 2e-6 to 4e-6 m (explicit) and 2.4e-7 m (implicit) | `test_kernel_hmmwv.cpp`, [hmmwv.md](hmmwv.md) |
| Change detection | Stored references; deliberate mutations of the physics | The code's own past | A change of 1e-6 fails by 26 to 997 times the tolerance; 9 of 9 mutations caught | `tests/regression/`, nightly |

## 1. Tree dynamics against Pinocchio

Plan task 5.3, D46. `crosscheck/compare_pinocchio.py` rebuilds 40 random trees
of 2 to 24 bodies, every new body hung from a random earlier one on a random
joint of each of the eight kinds, gravity in a random direction, and
compares three random states each with Pinocchio 4.1.0 (conda-forge):

| Quantity | Largest relative difference |
|---|---|
| Accelerations, articulated-body algorithm | 2.1e-13 |
| Generalized forces, recursive Newton-Euler | 1.4e-15 |
| Mass matrix, composite-rigid-body algorithm | 1.1e-15 |
| Every body's placement | 6.2e-16 |

The plan asked for 1e-8. The comparison runs in CI (job `crosscheck`). It
needed Pinocchio's joints added depth first: its mass matrix assumes that
numbering, ours any parents-first one
([the note for its developers](../documentation/upstream/pinocchio_depth_first.md)).

## 2. Closed mechanisms against Project Chrono

Plan task 5.3, D46. `crosscheck/compare_chrono.py` rebuilds 12 random trees
of 3 to 8 bodies closed by one to three constraints of six kinds (spherical,
revolute, universal, cylindrical, prismatic, distance) in PyChrono 10.0.0,
every body on its own and every joint a link:

- **At rest**: accelerations, the closures' loads on each body and each
  joint's reaction within 1.4e-12 of Chrono's own equations (solved from the
  matrices its assembly sets up; its assembled accelerations carry the
  constraints' residual over its 1e-6 s step squared, about 1e-4).
- **In motion**, from rest for 0.5 s: Chrono's HHT at two steps converges to
  our motion at its second order (ratios 3.98 to 4.02), its Richardson
  extrapolation within 1.4e-12 to 6.2e-9 m of ours.

A change of one part in a million in one inertia fails both parts.

## 3. Closed-form motions

Tasks 2.7b and 5.1 (D43), and the analytic cases of Phase 1:

| Motion | Agreement |
|---|---|
| A torque-free body with three unequal moments (Jacobi's elliptic functions) | 1.7e-12 rad/s over 5 s; repeats with their period |
| A heavy top's steady precession (the smaller root of m g l = Omega (C w3 - A Omega cos theta)) | 2e-12 rad after 1.5 s |
| A heavy top's nod, released with spin only (the root of its energy cubic) | 2.6e-14 in cos theta |
| Pendulums on revolute, spherical and universal joints; the slow mode of a double pendulum | Periods to their closed forms |
| A pendulum's hinge reaction, m a - m g | 3e-14 N |
| A bead on a spinning rod; a slider on a spring; sliders on inclines | Closed forms |

## 4. Manufactured motions

Task 5.1 (D42, D43). A smooth motion prescribed in closed form for two
random trees using every joint type, two twin chains tied by a distance
constraint and a parallelogram four-bar, its driving forces from inverse
dynamics (with the constraint multipliers for the loops): the forward
dynamics returns the accelerations to 2e-13 and the constraint forces to
1e-15 relative; one step from the exact state is accurate to order p + 1 for
every one-step method, and every method returns the motion after 0.5 s at its
order.

## 5. The IFToMM benchmarks

Task 5.2 (D44), group A of the IFToMM library of computational benchmark
problems:

| Problem | Reference | Agreement |
|---|---|---|
| A01 simple pendulum | Closed form (Jacobi) | 1.1e-9 m over 10 s on a joint, 1.4e-8 m on a constraint |
| A02 N-four-bar | Closed form, one parallelogram | 2e-8 rad over 10 s and 19 singular positions; 40 loops 1.7e-7 to 1.6e-6 rad, the constraints' rounding amplified near the separatrix |
| A03 Andrews' mechanism | The Test Set for IVP Solvers at 0.03 s | 1.4e-9 rad, a time shift of 1e-12 s |
| A04 Bricard's mechanism (a redundant equation) | Its path in closed form (derived here); its dynamics reduced to one coordinate; three results from the library | 9e-15 rad; 2e-9 rad; 2.5e-5 m from the one that does not drift |
| A05 flyball governor (stiff) | Angular momentum and energy balance; three integrators; two results from the library | 1e-10; 1.4e-9; 2.0e-6 and 6.8e-6 m |

## 6. Invariants

Tasks 1.1, 2.2, 2.5. Every joint, as root and as child of every other and in
branching trees, passes one shared suite: velocities and Jacobians against
finite differences of the placements, d'Alembert's principle, the agreement
of the recursive algorithms, gravity as the gradient of the potential,
energy and momentum conserved, a bead on a spinning rod. Every constraint
primitive and closure passes finite-difference checks of its Jacobian and
both right-hand sides on random trees, and each closure allows exactly its
joint's motions. Before the Phase 1 fixes these checks failed 315 of 845
assertions, all where the review's findings F1 to F4 predicted.

## 7. Integrators

Tasks 4.1 to 4.9 ([the integrators, verified](integrators.md), D40, D41):
on an oscillator whose numerical solution is known per step, every method's
step equals its stability function to 2e-16 and its energy after 1000 s its
recurrence's to 3e-11; on the spatial models every method converges at its
order (RK4 4, Dormand-Prince 5, generalised-alpha 2, SDIRK4 4 on smooth
models, Radau IIA 5, the Euler steps 1), through the projection onto the
constraints; the explicit methods' step limits match the eigenvalues of each
model's linearisation; over 1000 s the constraints' residuals stay within
1e-10 for a four-bar and a heavy top (task 4.7, D38).

## 8. Statics, assembly, linearisation and kinematics

Tasks 3.4 to 3.7 and 5.1:

- Assembly: a four-bar given a loop open by 3.6 cm comes to the closed-form
  configuration and loop velocities to 1e-16; the least correction meets
  its optimality condition to 8e-14. Constraints that contradict each other
  are named, and only they, from each of 48 starting positions (D50).
- Statics: a pendulum, a spring, a four-bar against its energy minimum, a
  driven slider and a block landing on a stop give their closed-form
  equilibria; the sedan settles in 4 iterations to 3e-12; two springs a
  millionfold apart in stiffness settle in one Newton step (D47).
- Linearisation: the quarter car's two frequencies to 1e-11 relative and its
  shapes to 5e-10; a 68.5 kg engine on three tilted bushings against eig(K,
  M) in closed form to 2e-11 in frequency, 3e-10 in shape and 2e-10 in each
  mode's kinetic energy shares; a line of four masses to 4e-12 to 4e-11.
- Kinematics: a double wishbone's bump motion and its velocities against
  finite differences of the configurations to 1.5e-8 relative.

## 9. Force elements, contact and events

Tasks 3.1, 3.8, 3.9: every element's force law and its derivatives match
central differences to 1e-6, and each reproduces a closed-form motion: a
preloaded spring's static deflection exactly, a torsion-spring pendulum's
period to 2.3e-6, friction's stopping distance to 8e-6 m, a bushing's two
periods to 3e-11 s; a block slides down an incline with a = g (sin theta -
mu cos theta) to 5e-15; a bouncing ball meets the ground at the closed-form
times to 1.8e-14 s.

## 10. A vehicle-sized stiff model: the HMMWV

Task 5.4 (D47, D48; [the HMMWV benchmark](hmmwv.md)). The fourteen-body HMMWV
of Haug, Negrut, Serban and Solis (1999), its data from Project Chrono's
HMMWV with two corrections, run for 5 s over a half-cylinder bump by every
integrator against Radau IIA at 1e-10:

- With joints, Dormand-Prince at 1e-8 within 2.2e-7 m, Radau IIA at 1e-8
  within 6.4e-10 m; run again with events, and so with other steps, the
  reference moves by 1.6e-11 m.
- With the paper's bushings (fastest mode -2.06e6 s^-1), RK4 at 1.25e-6 s and
  Dormand-Prince within 3.6e-6 and 2.0e-6 m, taking 1,060 and 1,520 s, Radau
  IIA within 2.4e-7 m in 0.78 s.
- At rest the tyres carry the weight within 1e-3 N (the test's bound); the
  chassis rests at 0.6875 m (the paper: about 0.71 m) and settles after the
  bumps within about 3.5 s (the paper: about 4 s).

This is agreement between methods on one model, not a comparison with the
paper's numbers, which it does not give.

## 11. Change detection

Task 5.5 (D49; `tests/regression/README.md`). Nine deterministic scenarios
(the kernel, the force elements, the vehicle template with its tyres and
drivetrain, statics and linearisation, kinematics, the lap model) against
stored references, each channel within 1e-8 of its largest magnitude, on
every commit; a relative change of 1e-6 in one physical parameter fails each
by 26 to 997 times that tolerance, one of 1e-12 moves none by more than 5e-3
of it. Nightly, nine one-line mutations of the physics must each make a test
fail; all nine do.

## 12. Platforms and memory safety

Every push is built and tested on Windows (MSVC, the developer's build) and
on Linux (GCC, without the precompiled header) with the address and
undefined-behaviour sanitizers, any report failing its test (D45); a Debug
build checks that a step allocates nothing (Eigen's run-time check and an
allocation counter); the API reference is built with every Doxygen warning
an error. The regression scenarios' differences between the two platforms
are reported on every run. Nightly, the tests also run in GCC's Release
build (-O3), which found D50.

## What is not validated yet

- **Tyres.** The Magic Formula's curves against Pacejka's reference curves
  and an independent implementation are task 7.4; the tyre model in use has
  only its derivatives and closed-form contact cases checked.
- **The vehicle against measurements or another vehicle code.** Task 9.6
  (the bicycle model, the quarter car, Chrono's HMMWV in step steer and lane
  change). The HMMWV benchmark here compares methods, not vehicles.
- **The comparisons' coverage.** Pinocchio covers trees; Chrono covers
  gravity and constraints, not force elements, drivers or contact.
- **Aerodynamics, real time, flexible bodies**: later phases.
- **Known limits, documented where they apply:**
  - SDIRK4 loses its order beside stiff bushings and does not converge
    across force jumps without events.
  - The linearly implicit step gains energy on stiff springs between
    turning bodies when k h^2 / m exceeds about 25.
  - Generalised-alpha fails with events on the HMMWV.
  - Dormand-Prince locates events on its continuous extension, which
    assumes smooth dynamics across the crossing.
  - Statics counts a direction as stiff only above 1e-7 of the stiffest.
  - Events that go out and back within one step are not seen.
  - Long runs of chaotic models depend on rounding.

## Repeating it

```powershell
scripts\build.ps1 -Test                                  # every test (530), about a minute
build\tests\test_kernel.exe "[long]"                     # the long kernel tests (the drift over 1000 s), minutes
build\tests\test_regression.exe "[long]"                 # the long regression scenarios
build\tests\test_regression.exe "[sensitivity]" -s       # the regression tolerance's evidence
scripts\build.ps1 -Target mbd_crosscheck                 # then crosscheck\README.md: Pinocchio and Chrono
scripts\build.ps1 -Target mbd_verify                     # then build\bench\mbd_verify.exe docs\integrators
scripts\build.ps1 -Target mbd_hmmwv                      # then build\bench\mbd_hmmwv.exe (about 45 minutes)
python scripts\mutation_test.py --build "..." --test ... # the mutations (see the script's header)
```

## Versions of this report

| Version | Date | What |
|---|---|---|
| 1 | 9 November 2026 | The end of Phase 5 (milestone M1): sections 1 to 12 |
| 1.1 | 9 November 2026 | The nightly's first runs: A02's 40 loops measured on both platforms; contradicting constraints named from 48 starts (D50) |
