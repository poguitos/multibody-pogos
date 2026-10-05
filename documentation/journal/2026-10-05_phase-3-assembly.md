# Phase 3: assembly of initial conditions

- **Dates:** 5 October 2026
- **Plan:** task 3.4; decisions D25 (cloud sessions) and D26 (assembly)
- **Commits:** `b1b9db3` (record kept through git), then this task's commit
- **Written:** at the time, in the first cloud session

## Goal

Positions, velocities and accelerations solved onto the constraints with
chosen coordinates held, and the smallest correction to the rest. *Done when
a loop given inconsistent initial values is assembled and reported.*

## Starting point

This was the first session in the cloud. The clone's `main` stopped at task
2.10; the Phase 3 work so far (tasks 3.1, 3.2, 3.3, 3.6, H.1, H.2 and the
record) was on the branch `phase-3`, pushed from the laptop. The session's
branch was fast-forwarded to it, and the rules the laptop sessions kept in
their memory were written into `CLAUDE.md` first (D25), so that this and every
later session reads them from the repository. The suite was then built with
GCC 13 for the first time outside CI: 390 tests passed, with no warnings from
the project's own files (GCC 13 prints `-Warray-bounds` false positives from
inside its standard library, through spdlog and Catch2).

For assembly, the projection of task 2.6 already moved q onto the
constraints by Gauss-Newton steps in the kinetic-energy metric, and
`Simulator::initialize` called it. It held nothing, its correction was the
least only to first order, and it reported a residual and an iteration count.

## What was done

1. **Holding by masking.** A held coordinate k gets a zero column in J and an
   identity row and column in the metric. The correction `W^-1 J^T mu` is then
   exactly zero in k (the Cholesky factor of W keeps row and column k as those
   of the identity, so nothing reaches it), and the solver's existing
   range-space solve, with its rank-revealing factorization, works unchanged.
   The Gauss-Newton loop was taken out of `project()` into one function that
   both use.
2. **The least correction.** After Gauss-Newton reaches the constraints,
   refinements re-linearize at the current point and take the least-norm
   correction from the *given* configuration that satisfies the linearized
   constraints, `d = W^-1 J^T (J W^-1 J^T)^-1 (J d_k - phi)`, then project
   back. The fixed point satisfies `W d = J^T mu`, the optimality condition of
   the least correction.
3. **Rates.** Velocities and accelerations are linear problems:
   `v += W^-1 J_free^T A^-1 (nu - J v)`, with the held velocities unchanged,
   and the same for `J a = gamma` at the assembled `(q, v)`.
4. **The report.** `AssemblyReport` gives the counts (equations, rank,
   degrees of freedom) and, per level, the residual before and after, the
   iterations, the coordinate that changed most (named: "v[1], the revolute
   joint of body 2 (coupler)"), the rank left to the free coordinates and the
   freedom left. Warnings and errors carry codes K062 to K067. The helpers
   that name bodies and constraints and print numbers moved from
   `validate.cpp` into `src/kernel/labels.hpp`, so that both reports read
   alike.
5. **Holding rotations.** A rotation is held whole or not at all (K061): a
   correction applied about one body axis but not the others does not keep
   any one angle fixed. A free joint's translation, in its child-side axes,
   may be held in part only when its rotation is held, since those axes turn
   with the body otherwise.
6. **Tests** (`tests/kernel/test_kernel_assembly.cpp`, 7 cases), and the
   four-bar fixture moved into `tests/kernel/four_bar.hpp`, with its closure
   as a function of the crank angle, for the constrained-dynamics tests and
   these.

## Decisions

D26: masking rather than selecting the free columns into smaller matrices
(one solve, no index bookkeeping); the mass matrix as the metric rather than
unit weights (unit weights mix radians and metres) or user weights (can come
later as a diagonal scaling); exact holds rather than large weights;
refinement to the true least correction rather than stopping after
Gauss-Newton.

The K066 note ("freedom left, chosen by the least change") is given only for
a level that changed something: on the first run it also appeared for
velocities given as zero and left at zero, where it says nothing.

## What went wrong, and how it was found

- **The brute-force check was wrong, not the code.** With nothing held, the
  test compares the assembled four-bar against a golden-section search for
  the least correction along the loop. It failed by 0.12 rad. A probe
  printing the cost along the loop showed the search interval, 0.6 to 1.6
  rad around the given crank angle of 1.1, did not contain the minimum, at
  1.72 rad: the crank is the lightest link, so the least kinetic-energy
  correction turns it the most. The search now spans the cost's single valley
  (0.5 to 2.5 rad). The same surprise will meet users, so the help says it
  (K066) and recommends holding the coordinates that define the intended
  state.
- **Held values that contradict the constraints stalled the iteration at its
  first step.** With crank and rocker held 0.2 rad apart from any closed
  position, only the coupler is free: one column for two independent
  equations. The rank-revealing solve drops the dependent pivots, which is
  exact when the right-hand side is in the range of J, and then gives the same
  correction as the Moore-Penrose inverse. Here it is not: the step satisfied
  one equation and increased `|phi|`, and no halving helped. Two changes
  followed. A step that no halving makes decrease `|phi|` now ends the
  iteration at the best point; the old loop carried on from the last, worse,
  trial point. And when the equations are dependent and the plain step fails,
  the least-squares step `L^-T (Z^T)^+ phi` is tried, which reduces `|phi|`
  unless it is already at its least: here from 0.1198 to 0.1178, which the
  report gives with the reason (K065, K062).
- **An existing test caught the fallback where it did not belong.** The full
  suite then failed one simulator test: two contradictory drivers (one point
  held at heights 0 and 0.5), where the projection had kept the first
  equation and left the second, as the dynamics solve does. With the
  fallback the projection settled on the compromise, 0.25. Neither answer is
  right for contradictory constraints, but in a simulation the projection
  should drop the same equations as the dynamics, or the two pull the state
  in different directions every step; and the simulator's projection must not
  allocate, even while failing. So the least-squares step is used by assembly
  only, whose aim is the closest state it can reach and a report of it; the
  test stands unchanged. Stopping at the best point when no step helps
  applies to both, and changes nothing in that test (its step was zero).

## Verification

| Check | Measured | Bound, and why |
|---|---|---|
| Two sliders in series, linear constraint: positions, velocities, accelerations against the closed-form least correction | exact to roundoff | 1e-14: one linear step |
| Held coordinates unchanged (four-bar crank, slider, quaternion of a held rotation) | bit for bit; quaternion to 1e-16 | `==`; 1e-15 for the final normalization |
| Four-bar, crank held: positions against the analytic closure | 6.3e-12 rad | 3.5e-10 = 2 x tolerance / smallest singular value of the free columns |
| velocities against central differences of the closure | 1.6e-11 rad/s | 1e-8, from the truncation (h^2 \|q'''\|/6, \|q'''\| = 0.5) and roundoff |
| accelerations against second differences | 3.7e-7 rad/s^2 | 4e-6; the measured value is the truncation H^2 \|q''''\|/12 x 4 |
| Nothing held: optimality cosine \|n . M0 d\| / \|M0 d\| | 5.6e-11 (plain projection: 0.081) | 1e-8; and the projection at least 1e3 times worse |
| Nothing held: against the brute-force search | 2.4e-9 rad | 1e-7, golden section's sqrt(eps) in the crank angle times \|dq/dth\| |
| Contradicting holds, positions and velocities | K062, K063, K065 reported, held values untouched, the best state returned | |
| Malformed holds: index out of range, the ground, part of a rotation | K060, K061 thrown | |
| A driven pendulum assembled at t = 0.3 s by `assemble(sim)` | q, v equal s(t), s'(t) to 1e-12; kinematics refreshed | |
| A simulator that cannot be assembled | one K067 warning carrying K062 | |

397 tests pass with GCC 13 (390 before); the CI's no-allocation check passes
locally (`test_kernel "[alloc]"` and `test_alloc` in a Debug build with
`EIGEN_RUNTIME_NO_MALLOC`). Two unused helpers in `test_full_tire.cpp` and
`test_steering.cpp`, which GCC warns about and MSVC does not, were removed.
CI run 37332674074 on the pushed commit passed: the first MSVC build of this
work, the sanitizers, and the no-allocation job.

## Measurements

The margins above, with the program that printed them:
[data/2026-10-05_assembly-margins.md](data/2026-10-05_assembly-margins.md).
The least correction of the four-bar given all three angles wrong costs
0.0955 (d^T M0 d); a plain projection's costs 0.1077, 13 % more, and its
crank stops at 1.50 rad instead of 1.72.

## Lessons

- A brute-force check is only as good as its search region; print the cost
  along the search before trusting a failure in either direction.
- A rank-revealing solve that drops pivots is a solve of the independent
  equations, not a least-squares solve. The difference is invisible while the
  right-hand side is consistent, which is the normal case, and decisive when
  it is not.
- "Least correction in the kinetic-energy metric" sounds neutral and is not:
  light parts move most. Say so where users will look.

## Open ends

- The static equilibrium of task 3.5 will start from an assembled state; the
  vehicle builders still call `Simulator::initialize` (plain projection).
  Whether they should hold the chassis and assemble the corners is for
  Phase 8 (the vehicle builder), where the vehicle's design state is defined.
- User weights per coordinate (D26, alternatives) if a case needs them.

## For the book

- Chapter on constraints and their solution: assembly as a constrained
  least-squares problem; the masking trick (a held coordinate as an identity
  row in the metric and a zero column in J); why Gauss-Newton gives the least
  correction only to first order, and the re-linearized iteration whose fixed
  point is the optimum. Figure: the four-bar's cost along the loop, with the
  given point, the projection (1.50 rad) and the least correction (1.72 rad).
- The pivot-dropping solve versus least squares: a small example where they
  differ (one free coordinate, two equations).
