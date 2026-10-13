# Phase 3: initial-condition assembly

- **Dates:** 13 October 2026
- **Plan:** task 3.4; decision D25
- **Commits:** see the index
- **Written:** at the time

## Goal

Position, velocity and acceleration solved with chosen coordinates held, with
the smallest correction to the rest. *Done when a loop given inconsistent
initial values is assembled and reported.*

## Starting point

`Simulator::initialize()` projected the state onto the constraints with
`ConstraintSolver::project`: Gauss-Newton steps, each the least change in the
kinetic-energy metric (the full mass matrix). Nothing could be held, nothing
was reported beyond the residual, and the result was not the smallest
correction: each Gauss-Newton step is the least from where it starts, so the
sum of the steps is the least only to first order.

## What was done

- `kernel::assemble(sim, options)` in `kernel/assembly.hpp`, with
  `AssemblyOptions` (held velocity indices for positions and for velocities,
  `hold_joint`, tolerance, iterations) and `AssemblyReport` (residuals before
  and after, iterations, counts, given and assembled values, loads, warnings,
  `summary(model)`).
- Positions in two stages: Gauss-Newton onto the constraints, changing only
  the coordinates not held; then an iteration on the whole correction from
  the given values, which stops at the least one. Velocities by one linear
  solve. Accelerations and constraint forces from `compute_loads` at the
  assembled state.
- The least change of each kind as `W_f^-1/2 B^T mu` with `B = J_f W_f^-1/2`
  and `(B B^T) mu = r`, solved by the constraint solver's rank-revealing
  `L D L^T`, now a class of its own (`RankRevealingLDLT`) that both use. The
  ranks with and without the holds come from it, so held coordinates that
  over-determine the mechanism are found by comparing the two. The first
  version used a complete orthogonal decomposition instead (see below).
- Codes K060 to K064; the kernel document has a section on initial
  conditions; the help index points to the tests as worked examples.

## Decisions

- **D25**: holds as velocity indices, the diagonal of the mass matrix as the
  metric, "smallest" exact rather than to first order, no holds on
  accelerations, warnings in the report rather than printed.
- Holds could not be written as constraint equations, which was the first
  idea: a constraint sees the joint transforms in `Data`, not the
  coordinates (task 2.3 kept the coordinates out of `Data`), so a hold on an
  arbitrary coordinate has no place there. Removing the held columns from
  the least-change problem is simpler and exact.
- The catalogue test requires a code in every statement that warns. A helper
  that forwards collected warnings to `report_warning` would have needed a
  code in its own call, or an exemption; the report keeps them instead, as
  `validate()` does, and `[[nodiscard]]` makes ignoring it a compiler
  warning.

## What went wrong, and how it was found

- **Compile time.** Every test passed on the first build, but reading
  `build/.ninja_log` afterwards showed `assembly.cpp` at 39.5 s (most library
  files take 1 to 3 s, the heaviest, the kernel's algorithms and
  `validate()`, about 25 s) and the new kernel test
  at 58 s. Compiled alone with `/Bt+`, the library file spent 25.9 s in the
  front end, where MSVC instantiates templates; a copy with the decomposition
  stubbed out compiled in 5.6 s. Eigen's `CompleteOrthogonalDecomposition`
  costs about 31 s per file that instantiates it, and the full rebuild was
  heading past the ten minutes task 2.8 set. The fix also improved the
  design: the solver's rank-revealing `L D L^T` became a class compiled once,
  and the assembly uses it, so the assembly now counts rank by the solver's
  very rule instead of an equivalent threshold on another decomposition
  (the square root of the solver's 1e-10, equivalent only approximately).
  The test's reference computations were rewritten on the four-bar's planar
  loop equations with 2 x 2 inverses, which also makes them independent of
  the kernel's constraint code. Now 9.4 s and 10.4 s; full rebuild 8.3 min.
- **A sanity check with a thin margin.** In the pendulum test, "the loop was
  corrected" was checked as a change of more than 0.01 rad; the measured
  value was 0.0105. The bound had no derivation behind it. Replaced by the
  residuals: the loop open by 8.8e-3 before (bound 1e-3) and closed to
  2.2e-16 after (bound 1e-10).
- **A warning the test did not expect.** When a free body's position is held
  and only rotations may close a point coincidence, the three equations have
  rank 2 with the holds: the distance from the point to the body's origin
  cannot change. The assembly reported MBD-K063, correctly; the test had
  only checked `ok()`. It now checks the ranks and the warning, and the
  other case (orientation held) checks that there are no warnings.
- **Reading the report.** The sedan's summary lists four bodies called
  `dyn_LCA` and four `dyn_upright`: the builders do not put the corner in the
  names. The body index tells them apart; the names are an open end.

## Verification

Seven test cases, 397 tests in all, every one passing.

- **The "done when" case.** A four-bar (crank 0.3, coupler 0.8, rocker 0.6,
  ground 0.7) is given a loop open by 3.6 cm and velocities that would stretch
  it; the crank is held at 1 rad and 2 rad/s. With the crank held the loop
  has no freedom left, so the result must be the closed-form configuration
  (two circles meeting) and the loop velocities from the planar velocity
  equations: both to 1e-16, the crank bit for bit. The accelerations are
  checked without the solver: `J a = gamma`, and `v^T (M a + b) = 0` because
  the constraint forces do no work when `J v = 0`; with one degree of freedom
  along v, the two determine a. The bound on the second, 1e-10 times
  `|v| |M a + b|`, allows for rounding in a sum of a few terms of size 19.
  The summary names each coordinate and the counts.
- **Smallest.** With nothing held, the correction must be perpendicular, in
  the weights, to the one motion the mechanism allows at the assembled
  point: cosine 7.6e-14 (bound 1e-8, from the stage's step tolerance 1e-10
  against a correction of about 0.03). A golden-section search along the
  mechanism's configurations finds the same crank angle to 1.1e-9 (bound
  1e-6, the search's accuracy near a flat minimum being about sqrt(eps)
  relative), and no smaller distance. Gauss-Newton alone gives a cosine of
  6.1e-3: the second stage is what makes it the least.
- **Untouched coordinates.** A pendulum hung from the coupler keeps its angle
  and rate bit for bit; the solver's projection, with the full mass matrix,
  moves them by 0.022 rad and 1.35 rad/s. This is the evidence for D25.
- **Over-determination.** Crank and rocker held: rank 1 instead of 2, K063,
  and with inconsistent values K062 and the held values untouched; with
  consistent values, assembled with K063 only.
- **Free joint.** Orientation held: the body translates to `G - R0 r`
  (6e-17 m), its quaternion unchanged to 1.2e-16. Position held: position bit
  for bit, and the rotation is the smallest one, about the axis across
  `R0 r` and `G - p`, to 1.8e-12 rad (the iteration stops at a step of 1e-10;
  bound 1e-9).
- **Sedan.** The double-wishbone sedan, every coordinate but the chassis's
  moved by up to 0.02 and given random velocities, chassis held at 20 m/s:
  assembled in 3 + 4 iterations to 3e-16, chassis bit for bit, 10 degrees
  of freedom as `validate()` counts, and the result half as far from the given
  values as the configuration they were disturbed from (which has the same
  chassis and closes every loop, so it bounds the least correction).
- **Refused holds.** An index beyond the velocities and a hold on the ground
  or on a missing body throw MBD-K060.
- The constraint solver's change (the pivoting moved into
  `RankRevealingLDLT`, the code itself unchanged) leaves every earlier test
  passing, `test_alloc` included.

## Measurements

All in [data/2026-10-13_assembly.md](data/2026-10-13_assembly.md): the
margins of every check, the compile times of both versions with the `/Bt+`
split, and the full rebuild (85 objects, 8.3 min at one job, against 7.4 min
for 65 objects on 5 October).

## Lessons

- Read the build log after a build that passes. A file that compiles four
  times slower than its neighbours is a finding like a slow test, and here
  it pointed at a better design.
- In Eigen, which decomposition a file uses decides its compile time.
  Prefer the factorizations the kernel already compiles once, behind a
  function in a source file.
- A sanity check needs a derivation like any other bound, or it should test
  something with a clear margin (residuals before and after).

## Open ends

- The smallest correction is exact to second order only for spherical and
  free joints (the logarithm of a rotation does not add). Exact would need
  the derivative of the exponential map per joint; not worth it until a case
  needs it.
- `Simulator::initialize()` still only projects. Whether it should assemble
  is a question for Phase 4, when the integrators need consistent initial
  accelerations (generalised-alpha does).
- The vehicle builders place their vehicles near equilibrium and rely on the
  projection; they could assemble with the chassis held instead, and report.
- The vehicle builders give the bodies of all four corners the same names
  (`dyn_LCA`, `dyn_upright`, ...), which makes the assembly report, and later
  `describe(system)` (task H.4), harder to read. Prefix the corner (`FL_`
  and so on) when the builders are next touched (task 7.1 or H.4).
- The full rebuild is at 8.3 min of the 10 that task 2.8 set. The heaviest
  files are tests (`test_spatial` 51 s, `test_kernel_constraints` 34 s);
  worth a look before Phase 4 adds more.

## For the book

- Assembly as an optimisation problem: the least correction subject to the
  constraints, its optimality conditions, and why Gauss-Newton finds a point
  on the constraints but not the nearest one. The anchored iteration and its
  convergence rate (the derivative of the iteration map at the solution is
  of the order of the curvature times the correction).
- The choice of metric, with the pendulum outside the loop as the example: the
  full mass matrix turns it, the diagonal leaves it.
- Over-determination detected by ranks: the four-bar with crank and rocker
  held.
