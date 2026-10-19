# Phase 4: the mass matrix factorized along the tree

- **Dates:** 18 to 19 October 2026
- **Plan:** task 4.8 (sparse solve); decision D39
- **Commits:** see the index
- **Written:** at the time

## Goal

The plan's task 4.8: "sparse assembly and factorisation above a size
threshold. Done when dense and sparse results agree and the crossover size
is measured." Every evaluation of the constrained dynamics factorized the
mass matrix by a dense Cholesky, n^3 / 6 multiply-adds for n velocities,
though a tree's mass matrix is mostly zero.

## Starting point

The constraint solver's range-space method (task 2.6): M = L L^T by
Eigen's dense Cholesky, Z = L^-1 J^T, A = Z^T Z factorized with diagonal
pivoting. The mass matrix comes from the composite-rigid-body algorithm,
which computes only the entries where one velocity is the other's ancestor
(the rest are zero) into a dense nv x nv matrix. So the assembly was sparse
already; the factorization was not.

## What was done

- **The factorization.** `TreeCholesky`, Featherstone's LTL factorization
  ("Rigid Body Dynamics Algorithms", section 6.5): with each velocity
  numbered after its parent (the model's order), M = L^T L with L lower
  triangular has exactly M's pattern, no fill-in, eliminating from the last
  velocity to the first. Velocity-level parents: a joint's first velocity
  hangs from the last velocity of the nearest joint above it, its others
  from the one before. Work: each velocity k with d_k ancestors costs d_k
  (d_k + 1) / 2 + d_k + 1, so a chain costs n (n + 1) (n + 2) / 6 (the
  dense count) and a branched tree much less.
- **Into the solver.** M = L^T L is M = R R^T with R = L^T, so the
  range-space method only changes which triangular solves it calls: Z =
  L^-T J^T (skipping the zeros of J's sparse rows), M^-1 J^T x = L^-1 (Z x),
  and the free accelerations by both. `ConstraintSolver::mass_factorization`
  (Automatic, Dense, Tree); the projection and the forward dynamics share
  one `factorize_mass_matrix`.
- **Tests** (`tests/kernel/test_kernel_sparse.cpp`, new, and a sedan test):
  the pattern, L^T L = M, the solves and the solver's results against the
  dense ones on every joint type; the work counts against their closed
  forms; the sedan driving 500 steps with each factorization.
- **First measurement** (L stored as in the book, scalar loops along the
  parent pointers): the tree's factorization faster on every branched
  tree and on chains up to 16 velocities, the dense one from 32 on, by 3.7
  times on a chain of 64 (176 against 48 us; 78 against 21 the next day,
  the machine faster). The first rule, the tree's
  below half the dense work, chose dense for small models and short chains
  where the tree's had been measured faster.
- **The access pattern.** L's row k, walked along k's ancestors, is a stride
  of n doubles in a column-major matrix: on the chain of 64 each step a new
  cache line. L stored transposed puts
  that walk down a column, and a path to the root is a few runs of
  consecutively numbered velocities (a joint's, a chain numbered in order),
  each a contiguous stretch.
- **Runs as Eigen segments** (second version): the chain of 64 down to 34
  us, but every branched tree slower than before, the binary tree of 64
  from 2.0 to 13.1 us and the sedan slower than dense: a dynamic-size
  segment operation has a fixed cost that a run of one or two velocities
  does not repay.
- **Runs as plain loops** (kept): the chain of 64 at 23.0 us, the binary
  tree of 64 at 2.6, the sedan's constrained forward dynamics 19.7 us
  against 23.5 dense.
- **The crossover**, measured with a new sweep (a root body carrying 1, 2
  or 4 chains, 12 to 64 bodies): the tree's factorization faster on chains
  up to 32 velocities and on every branched tree; the dense one on chains of
  48 (by 2 to 4 %) and 64 (by 8 to 11 %), in two sessions. Automatic's rule
  follows: the tree's, unless the model has more than 40 velocities and the
  tree's work is at least 0.8 of the dense one's.
- **Eigen's general sparse Cholesky** (SimplicialLLT on M's pattern, not
  reordered, the pattern analysed once), measured as the alternative:
  slower than the dense Cholesky on every model (1.1 to 2.9 times), and
  than the tree's by 1.6 to 13.7 times. A general sparse code pays for its
  indexing; at a few dozen velocities the zeros do not repay it.
- **Against the previous commit** (`576df64`, built from an archive, run
  alternately): a step of the sedan takes 13 % less time with RK4, 12 % with
  generalised-alpha (11 % stabilized), 3 % with the linearly implicit step,
  which factorizes its own iteration matrix densely.

## Decisions

- **D39**: the mass matrix factorized along the tree, stored transposed and
  walked by runs; Automatic, the tree's except on long chains.

## What went wrong, and how it was found

- **The book's storage was slow on a computer.** RBDA's table 6.3 updates
  L's rows; stored as written, in a column-major matrix, each step along a
  path jumps n doubles. On a chain of 64 that made the tree's factorization
  3.7 times slower than the dense one for the same work. Found by comparing
  time per multiply-add: 3.9 ns for the tree's, 1.1 for the dense one's
  (the machine was slow that day).
- **Vectorizing made trees slower.** The obvious fix, an Eigen segment
  operation per run, sped the chain up and slowed every tree down, the
  binary tree of 64 by 6.5 times: the operation's fixed cost (alignment,
  sizes) is paid per run, and a tree's runs are one or two velocities long.
  Measured before anything was kept; plain loops, which the compiler
  vectorizes where a run is long and which cost nothing extra where it is
  short, kept both gains.
- **A threshold set before the measurement.** The first rule (the tree's
  below half the dense work) was a placeholder; the measurement showed the
  ratio alone is the wrong measure, because the dense factorization's speed
  per multiply-add grows with size (Eigen blocks it), so small models favour
  the tree's whatever the ratio.
- **The machine's speed changed by 2.3 times between sessions** (an RK4
  step of the sedan 282 us one day, 123 the next): only comparisons within
  one session, versions alternated, were used.

## Verification

`tests/kernel/test_kernel_sparse.cpp`, four test cases:

- On the joint types as roots of random trees and in chains, and a binary
  tree of 31: L zero off the tree's pattern (exactly), the runs listing
  each velocity's ancestors (exactly), L^T L = M to 1e-13 of |M|, the tree's
  solve and its two halves equal to the dense solve to 1e-12. Seen (`-s`):
  L^T L - M at most 1.9e-16 of |M|, the unit roundoff; the solves 5e-15 of
  the solution; the bounds leave a factor of 200 or more for worse
  conditioned models.
- The work counts: a chain of 16 exactly n (n + 1) (n + 2) / 6 = 816 and
  the dense count 16^3 / 6; a binary tree of 31 below a tenth of dense
  (351, 0.071).
- The solver with each factorization on all those models and on the
  four-bar closed in 3D off its constraints: accelerations and multipliers
  to 1e-11, the projection's positions to 1e-12 and velocities to 1e-11,
  in the same number of iterations. Seen: accelerations 1.9e-15 relative,
  the projected positions identical.
- Automatic: chains of 40 (tree, by size) and 41 (dense: ratio 1.07), a
  binary tree of 63 (work 1023) and a root with two chains of 32 (work
  13089), each work count against its closed form.

`tests/vehicle/test_vehicle_template_dynamic.cpp`: the double-wishbone
sedan, 500 steps of RK4 at 1 ms with throttle and steering ramps, with each
factorization: positions to 1e-9 relative, velocities to 1e-8, bounds
left for rounding that a run could grow. Seen: 1.0e-15 and 1.2e-14, no
growth over the run.

The whole suite: 469 tests, no compiler warning.

## Measurements

In the data file `data/2026-10-19_sparse-solve.md`: the three versions on
chains and binary trees, the crossover sweep with Eigen's sparse Cholesky,
the sedan's evaluation and step with each factorization, and every method's
step against the previous commit. All pinned to a performance core at high
priority, versions alternated within a session; the machine ran at
different speeds in the three sessions (the dense factorization of a chain
of 64: 47.6, 20.7 and 25.6 us), so only figures from one session are
compared.

## Lessons

- Time per multiply-add is the measure that showed both problems: a
  factorization doing the same work as another at three times the cost is
  an access-pattern problem, not an algorithm problem.
- A vector operation's fixed cost matters on the short runs that make a
  tree sparse. Measure the change on every shape, not just the one it was
  made for.
- Set a threshold after the measurement, not before.

## Open ends

- The rest of the constrained solve is dense: A = J M^-1 J^T = Z^T Z and
  its pivoted factorization, m^3 / 6 for m equations. A is dense even where
  J is not: M^-1 couples every corner through the chassis. Z is sparse
  (L^-T keeps a column of J^T on its velocities' paths to the root), which
  the products do not use yet.
- The iteration matrices of the implicit steps (the linearly implicit
  step's W = M - h C - h^2 K + h D) are factorized densely: K and C couple
  whatever the force elements join, and D whatever the velocity products
  do, so W need not have the tree's pattern.

## For the book

- Sparsity from topology: why a tree's mass matrix has the tree's pattern,
  and why numbering parents first leaves no fill-in.
- The same algorithm three times: the book's storage, a vectorized
  version, and plain loops, with their times on a chain and a tree; what a
  cache line and a vector operation's fixed cost do to an operation count.
- Chapter: outline item 32.
