# Phase 3: static equilibrium

- **Dates:** 5 October 2026
- **Plan:** task 3.5; findings F7, F9, F14; decisions D24 (derivatives), D27
- **Commits:** this task's commit (the index gives the hash)
- **Written:** at the time, in a cloud session

## Goal

Newton iteration on the static residual with line search, and dynamic
relaxation as a fallback. *Done when the detailed sedan settles in under 20
iterations to accelerations below 1e-6.*

## Starting point

There was no static solver (F9, F14). `set_vehicle_equilibrium` placed a car
by a hand estimate: the review measured accelerations of up to 73 there, and
1.08 m/s^2 on the chassis. Task 3.4 had just given the constraint solver
held coordinates and a masked metric; the force library of task 3.1 returns
each element's derivatives in its own coordinates, and D24 had decided that
statics would start from finite differences of the generalized forces.

## What was done

1. **Forces at rest from the simulator itself.** `Simulator::acceleration`
   was split: its first half, everything applied (the force elements, joint
   forces, `tau`, the callbacks), became `Simulator::applied_forces`, which
   statics calls. A vehicle's tyres, springs and drivetrain hooks are then
   evaluated by the same code in statics and in a simulation.
2. **The residual as accelerations.**
   `ConstraintSolver::accelerations_at_rest(q, f, t, held)` gives
   `a = W^-1 (f + J^T lambda)` with `J a = 0`, held coordinates locked and
   time frozen (no velocity products, no driver rates): zero exactly at an
   equilibrium, in the units the plan's criterion uses. It reuses the masked
   metric and the range-space solve of task 3.4.
   `ConstraintSolver::project_positions` exposes assembly's first stage
   (projection with holds, without the refinement) for the trial points.
3. **Newton on the constraint surface** (`kernel::static_equilibrium`). The
   tangent stiffness K by central differences of `F = f + J^T lambda` at
   fixed lambda; N, the allowed motions, from an SVD of the constraint rows
   and the hold rows; the reduced system `(N^T K N) y = -N^T f`, solved by a
   complete orthogonal decomposition that drops directions without stiffness;
   the step halved until `a^T M a` decreases, each trial projected back.
4. **Dynamic relaxation** with kinetic damping when no step helps, until the
   accelerations have fallen a hundredfold; a blow-up returns to the last
   good point.
5. **The report**: iterations, the history of `|a|_inf`, the degrees of
   freedom, and from the symmetric part of `-N^T K N` at the end the neutral
   and the unstable directions; codes K070 to K074. The hold-list checks
   moved from `assembly.cpp` into `src/kernel/held.hpp`, shared by both.
6. **Tests** (`tests/kernel/test_kernel_statics.cpp`, 7 cases), each against
   a closed form or an independent search, and the sedan.

## Decisions

D27: the residual measured as accelerations; Newton in the reduced
(null-space) form rather than on the full KKT system in q and lambda, whose
mixed units make rank decisions unreliable; finite-difference stiffness, as
D24 planned; neutral directions dropped and reported rather than held;
kinetic damping rather than viscous relaxation, which would need a damping
coefficient per model; stability read from the reduced stiffness.

## What went wrong, and how it was found

- **Newton never took a step; relaxation did all the work.** The first run
  on the sedan converged, but through four fallbacks and 1,452 relaxation
  steps. Printing the reduced stiffness's singular values and the residual
  of its solve showed the solve returning steps of order 1e44: Eigen's
  `CompleteOrthogonalDecomposition` decides its rank when it decomposes, so a
  threshold set afterwards changes `rank()` but not `solve()`, which had kept
  the three directions without stiffness (singular values of 1e-9 to 1e-11
  against 1e6). Setting the threshold first gave three Newton iterations and
  no relaxation. The least-squares step written for task 3.4 had the same
  mistake; there the default threshold happened to give the right rank, and
  its measured result did not change with the fix. The lesson is in the code
  comments at both places.
- **A tolerance derived too tightly.** The linear spring test first asked
  for its position to 1e-14 m after one step, as if the central difference
  of a linear force were exact. Its roundoff, `eps |k z| / (2 h)`, is 5.5e-11
  of k; the bound became 2e-12 m with that derivation, and 2.4e-14 m was
  measured.

## Verification

| Check | Measured | Bound, and why |
|---|---|---|
| Linear spring: position after 1 iteration | 2.4e-14 m | 2e-12 m, the central difference's roundoff in K |
| Torsion-spring pendulum against the root of `k th + m g l cos th` (3 iterations) | 3.3e-12 rad | 1.3e-8 rad: tolerance x inertia / stiffness |
| Four-bar with crank spring against the least potential energy along the loop (golden section), 4 iterations | 9.7e-10 rad; on the loop to 1.2e-12 | 1e-7 rad: sqrt(eps) of the search, and tolerance x inertia / stiffness |
| Double pendulum, first hinge held: held value bit for bit, second link straight down (3 iterations) | 6.2e-13 rad | 1e-7 rad |
| Upright bob on a weak spring: found (2 iterations) and reported unstable, one direction | 6.5e-14 rad; K072 | 1e-7 rad |
| Body 5 cm above the ground: relaxation 135 steps, then Newton; `R - m g / k` | 1.7e-16 m | 1e-10 m: m tolerance / k |
| Sedan: iterations / `\|a\|_inf` | 3 / 1.8e-8 | under 20 / 1e-6 (the plan) |
| Sedan: tyre loads against the weight (14,724.81 N) | 1.1e-7 N | 0.05 N: accelerations at most 1.2e-5 m/s^2 per body over 1,501 kg |
| Sedan: moments of the loads about the centre of mass, pitch / roll | 1.2e-7 / 7e-12 N m | 0.1 N m, likewise with 2 m lever arms |
| Sedan: degrees of freedom / without stiffness / unstable | 10 / 3 / 0 | the position and heading on the road are free |

Every case converges quadratically (the histories are in the data file). 404
tests pass with GCC 13 (397 before). CI run 37335603045 on the pushed commit
passed (MSVC, sanitizers, no-allocation).

## Measurements

[data/2026-10-05_statics.md](data/2026-10-05_statics.md): every case's
report, the sedan's tyre loads, and a comparison with a simulation. The
sedan's statics takes 6 ms in the container. `set_vehicle_equilibrium` puts
its chassis 2.1 cm below the equilibrium height (0.3309 against 0.3520 m).
Simulated from the same start with RK4 at 1 ms, the car approaches the static
state, its chassis within 3e-5 m and its suspension angles within 2e-4 rad
after 8 s, but keeps rolling backwards at about 4 mm/s: its free-rolling
tyres resist nothing (F7), so the momentum of the first transient never dies
away. A simulation cannot give a resting state for this car; statics can.

## Lessons

- Read the documentation of a decomposition for when its parameters take
  effect. A rank threshold that only changes what is reported is a trap
  that no compiler catches; the symptom (steps of 1e44, rescued by a
  fallback that still converged) looked like an algorithmic weakness.
- A fallback that rescues every failure can hide that the main method never
  works. The relaxation count in the report is what showed it; the tests
  now assert that the sedan needs no relaxation.

## Open ends

- The vehicle builders should settle the car with `static_equilibrium`
  instead of `set_vehicle_equilibrium` (F9; the vehicle builder, Phase 8,
  where "settles to a computed static equilibrium" is the done-when).
- The analytic tangent stiffness from the elements' derivatives (D24), when
  a larger model makes `2 nv` force evaluations per iteration costly.
- Linearisation (task 3.7) needs the same K, and the damping matrix as well:
  the finite-difference stiffness here is the place to grow it from.

## For the book

- Statics as a root of the accelerations at rest; Newton on the constraint
  surface (the reduced stiffness and the geometric stiffness of constraint
  forces); why a car on a flat road has three free directions, and why that
  is physics, not a defect.
- Kinetic damping explained with the body falling onto the ground.
- The trap of the complete orthogonal decomposition: a fallback that hid a
  failing main method. A good example for the chapter on testing a
  numerical code.
- Figure: `|a|_inf` per iteration for the sedan and the four-bar (quadratic);
  the sedan's chassis height over 8 s of simulation against the static
  value.
