# Phase 3: static equilibrium

- **Dates:** 13 October 2026
- **Plan:** task 3.5; decision D26
- **Commits:** see the index
- **Written:** at the time

## Goal

Newton iteration on the static residual with line search, and dynamic
relaxation as a fallback. *Done when the detailed sedan settles in under 20
iterations to accelerations below 1e-6.*

## Starting point

`set_vehicle_equilibrium` placed a vehicle at an estimated height and the
projection closed its loops; finding F (section 3.2 of the plan) noted that
this "equilibrium" is not one. Task 3.4 had just added the least-change
machinery (holds, weights, Gauss-Newton projection) and shared the solver's
rank-revealing factorization.

## What was done

- `static_equilibrium(sim, options)` in `kernel/statics.hpp`: Newton on the
  accelerations at rest, Jacobian by central differences through
  projections, least-norm steps in the weights of the diagonal of the mass
  matrix, line search, dynamic relaxation with kinetic damping as fallback.
  `StaticReport`: residual history, relaxation used, degrees of freedom with
  and without stiffness, coordinate changes, loads, coded messages.
- `ConstraintSolver::quasi_static`: gamma taken as zero, which at rest
  freezes the drivers where they are.
- The least-change pieces of task 3.4 moved to an internal header
  (`least_change.hpp/.cpp`) shared by assembly and statics; the labels and
  number formats of the kernel's reports to another (`labels.hpp`), which
  `validate()` uses too.
- `RankRevealingLDLT` rewritten as a pivoted factorization that stops at the
  numerical rank (see below).
- Codes K070 to K073; kernel document, help index, catalogue.

## Decisions

- **D26**: the residual, the Jacobian through projections, least-norm steps
  that leave neutral directions alone, relaxation as fallback.

## What went wrong, and how it was found

- **A car that would not come down 1.6 cm.** The first build passed every
  kernel case and failed the sedan: after one good Newton step the residual
  was the chassis's vertical acceleration alone, 1.18 m/s^2, and Newton made
  no further progress; the report said 5 directions had stiffness where 7
  must. The first hypothesis was scaling: `da/dq = M^-1 K` is far from
  symmetric with the sedan's mass coupling, so its scaled singular values
  might spread past the factorization's cut. A temporary diagnostic test
  disproved it: both scalings had seven clear singular values, the seventh
  squared 1e-5 of the first against a cut of 1e-10. Printing inside the
  Newton loop showed the linear model itself leaving the heave residual
  untouched (|G step + a| = 1.178 before any line search), and computing G at
  that point outside statics gave the same matrix with seven clear
  directions. So G was right and its factorization wrong. Eigen's
  documentation says it plainly: its LDLT "is not rank-revealing". After the
  seven real pivots it went on eliminating a Schur complement made of
  rounding error (about 1e-12) and divided by it; the quotients passed for a
  large pivot, the relative cut rose above a real one, and the same entries
  spoiled the solve. The constraint solver had never met this because
  redundant constraints give exact zeros, which Eigen skips. The fix was to
  make `RankRevealingLDLT` what its name says, a diagonally pivoted
  factorization that stops at the numerical rank, as LAPACK's pstrf does. A
  unit test reproduces the failing case (a noisy Gram matrix of rank 7) and
  the sedan then converged in 3 iterations, 4 with the step limits.
- **Neutral is neutral to first order.** The check that the chassis keeps
  its forward position failed by 1.24e-5 m. The free joint moves by an
  exponential, so a rise of 2.1 cm with a pitch of 1.23e-3 rad moves the
  origin forward by half their product, 1.29e-5 m; the test and the message
  now say "but for second-order effects", and the test checks the bound.
- **A folded suspension.** Started 5 cm (then 3 cm) high, the sedan
  converged to an equilibrium 36 cm lower, pitched 15 degrees, its rear arms
  turned 85 degrees. Two things contributed. Relaxation was driven by the
  full dynamics, and below V_low a tyre's lateral force acts as a damper of
  about 45 kN s/m, unstable for an explicit 1 ms step on a 15 to 20 kg
  upright: relaxation now uses the accelerations at rest. And Newton's first
  step sent the hanging rear wheels past the end of their travel; ADAMS
  limits each iteration's increments (TLIMIT, ALIMIT) for this reason, and
  statics now does too (0.02 m, 10 degrees). The fold remained: with the
  limits the residual decreased steadily while the arm walked down 10
  degrees an iteration. It is a real equilibrium of the model, whose springs
  have no rebound stops and whose tyre contact point is fixed on the
  upright, so a turned upright lifts its "contact point" off the ground. The
  second sedan test starts 1 cm high, with every tyre loaded; the template's
  defects are open ends for the builders.
- **A test asserting what it had not derived.** The first version checked
  that the builder's starting state drifts by millimetres in a second; there
  was no derivation, only an expectation. It became a printed measurement
  (2.1 cm).

## Verification

Ten new test cases (407 tests, all passing):

- **Linear spring:** one Newton step lands on -m g / k to 2.3e-13 m when the
  step limit is raised above the deflection; with the default limit, two
  steps (0.02 m, then the rest), to 2.4e-14 m.
- **Torsion-spring pendulum:** the angle where -k (th - th0) = m g l cos th,
  found by bisection, to 4.2e-10 rad against a bound of tolerance x I / K =
  4.4e-9 rad, in 4 iterations.
- **Four-bar with a spring:** the crank angle of the least potential energy
  along the mechanism's closed-form configurations (scan and golden
  section), to 1.5e-8 rad, in 3 iterations from 0.2 rad away.
- **Frozen drivers:** a block on a driven slider sits at -m g / k (5.9e-14
  m), not at -m (g + s'') / k, 5.9e-3 m away; the solver's flag is given back.
- **Relaxation:** a block 0.1 m above a one-sided stop needs 193 relaxation
  steps, then 2 Newton steps, to -m g / k_stop exactly.
- **Failures reported:** a block with nothing to hold it (K070, K073, falls
  0.197 m in 0.2 s as g t^2 / 2 says), and positions that cannot reach the
  constraints (K071).
- **The factorization:** rank 7 of a noisy 27 x 27 Gram matrix shaped like the
  sedan's, solve exact to 6.6e-16 relative; rank 2 of the planar loop's
  exact 5 x 5.
- **The sedan (done when):** 4 Newton iterations, 73.4 to 3.1e-12, no
  relaxation; 10 degrees of freedom, 7 with stiffness, 3 neutral; the
  neutral components moved by at most 1.5e-6 against their second-order
  bound of 1.3e-5; the tyres carry the weight to 7e-12 N of 14725 N; a second
  of simulation from the equilibrium moves the chassis 4.7e-17 m (from the
  builder's state, 2.1 cm). From 1 cm higher, the same height to 6e-17 m.
- The earlier tests all pass with the new factorization inside the
  constraint solver, the allocation counts included.

## Measurements

In [data/2026-10-13_statics.md](data/2026-10-13_statics.md): the diagnosis
(singular values, the iterations as they ran), the folded suspension, every
margin above, and the benchmark. The factorization runs in every step, so
the sedan was timed pinned against the previous commit, alternately: one
RK4 step 148.9 and 148.5 us against 148.8 and 150.1 us, no difference.
Both are 149 us against 124 us on 5 October; whether that is the machine or
the code of tasks 3.1 to 3.4 is not established.

## Lessons

- Read the library's documentation of a decomposition before relying on a
  property it does not claim. "Rank-revealing" was in our class's name, not
  in Eigen's.
- When a solver stalls, test the pieces separately: the Jacobian against an
  independent computation, then the linear solve against the Jacobian. The
  first hypothesis (scaling) was plausible and wrong; one measurement
  settled it before any code was changed for it.
- A method that follows the model faithfully will find the model's
  equilibria, including absurd ones. The fold is a finding about the vehicle
  template.

## Open ends

- Analytic tangent stiffness (D24) instead of differences, when cost
  matters (Phase 4's implicit integrators need it anyway).
- `set_vehicle_equilibrium` could call statics and become what its name
  says; the vehicle builders are touched in task 7.1.
- Statics holds nothing; holds as in assembly would let a user fix, say, the
  chassis's height while the suspension settles. Not needed yet.
- The vehicle template: rebound stops on the suspension springs, and a tyre
  contact at the wheel's lowest point rather than a point fixed on the
  upright. Without them a vehicle started with its wheels in the air folds
  its rear suspension, in statics and presumably in dynamics. Tasks 7.x.
- The sedan step: 149 us today in both versions against 124 us on 5
  October. Build `9472d8d` and run it alternately to tell the machine from
  the code.

## For the book

- Statics as Newton on the accelerations at rest, and why the residual is
  the plan's criterion rather than a potential (tyres have none).
- Neutral directions: a vehicle at rest on tyres has three, and a least-norm
  Newton step leaves them where the user put them.
- "Not rank-revealing": what Eigen's documentation means, and how it showed
  up as a vehicle that would not come down by 1.6 cm.
