# Phase 2: constraints and the constrained solve

- **Dates:** 3 to 4 October 2026
- **Plan:** tasks 2.5, 2.6
- **Commits:** `d33a128`
- **Written:** reconstructed on 5 October 2026 from the commit, `docs/kernel.md`
  and the session record

## Goal

Close loops on the kernel: every closure built from a few primitive equations
(2.5), and one place that forms and solves the constrained equations (2.6).

## What was done

- **Markers and primitives.** A marker is a frame fixed on a body. Five
  primitives give their value `phi`, Jacobian `J`, and the right-hand sides
  `nu` and `gamma` of the velocity and acceleration equations: point
  coincidence, dot-1, dot-2, distance and no-twist, each with a constant or
  time-dependent target (after Haug, and Shabana's *Computational Dynamics*,
  chapter 3). Point on a line and on a plane, and the spherical, revolute,
  universal, cylindrical, prismatic and constant-velocity closures, are
  compositions of them.
- **Jacobian rows as wrenches.** Every row is a sum of terms
  `c_w . w + c_p . v_P`, the power of a world wrench on the body, so a row is
  filled by walking up the tree once (`add_wrench_row`) without forming any
  body Jacobian.
- **The distance primitive** is `(d . d - L^2) / (2 L0)`: a length to first
  order, without the singularity of `|d|` at zero.
- **ConstraintSolver.** The range-space method: the Cholesky factor of `M`,
  then `J M^-1 J^T` factorised with diagonal pivoting; pivots below 1e-10 of
  the largest are dropped and their multipliers set to zero, so redundant
  constraints (a planar loop closed in 3D, a joint closed twice) are handled
  and reported through the rank. Optional Baumgarte stabilisation. Projection
  of q and v onto the constraints in the kinetic-energy metric.

## What went wrong, and how it was found

- **A hidden heap allocation in ABA.** A new hidden test, `[alloc]`, runs every
  kernel call in a Debug build with `EIGEN_RUNTIME_NO_MALLOC`, where any Eigen
  heap allocation fails an assertion. It caught one: Eigen evaluates
  `x = y - A b` through a temporary shaped like `y`, and when `y` is a block of
  a dynamic vector that temporary goes on the heap. Writing
  `x = y; x.noalias() -= A b;` fixed it. That became a rule in
  `docs/kernel.md`. A second change, adding `.noalias()` to products with
  transposes, was made with a wrong explanation and reverted once it was
  confirmed that Eigen already evaluates those without aliasing.
- **Catch2 3.6 cannot report a skipped test to CTest**, so a test that only
  makes sense in a special build would show as a failure. The allocation test
  is hidden (`[.]`), run by name in the CI job `kernel-no-malloc`, and fails
  loudly if run in a build without the check.
- **A tautological test.** A four-bar check compared a quantity with itself.
  It was replaced by one that can fail: the crank turning more than 2 pi.

## Verification

For every primitive and closure, with markers on every kind of body pair:
`J`, `nu` and `gamma` against first and second finite differences of `phi`.
Each closure allows exactly its joint's motions, and a body held by a closure
moves exactly like the same body on the matching tree joint, with constraint
forces equal to the tree's joint forces. A planar four-bar closed in 3D (rank 2
of 5) conserves energy to 1.6e-9 relative over 2 s while its crank turns
twice. Tolerances are derived: for instance the pendulum energy bound of 1e-8
from RK4's error at the step used. 509 tests passed.

## For the book

- Building every joint from five primitives is a small, complete design worth
  a section, with the table of closures from `docs/kernel.md`.
- Redundant constraints are where many codes fail; the pivoted factorisation
  and why `J^T lambda` is unique even when lambda is not.
- The allocation check: how a Debug-only Eigen switch turns "no allocation"
  from a claim into a test.
