# The mass matrix factorized along the tree: measurements (task 4.8, D39)

Machine: the development laptop (i7-13700H, 16 GB), Windows 11, MSVC
RelWithDebInfo (`/O2`), one compiler. `build\bench\mbd_bench.exe`, pinned to a
performance core at high priority, each figure the median of seven rounds;
two runs of each version, alternated, given as "run 1, run 2" where they
differ. Times in microseconds: one factorization and one solve of the mass
matrix (a vector of ones), the matrix already assembled. "Work ratio": the
tree factorization's multiply-adds, sum_k (d_k (d_k + 1) / 2 + d_k + 1), over
the dense n^3 / 6.

The machine's speed differed between sessions: one RK4 step of the sedan
(dense) took 282 us on 18 October and 123 us on the 19th; the dense
factorization of a chain of 64 took 20.7 us in session 2 and 25.6 in
session 3. Compare figures within a section only.

## 1. First version (session 1, 18 October, machine slow)

L stored as in RBDA's table 6.3 (lower triangle, column-major), scalar loops
along the parent pointers. Chains: dense against tree 0.11 / 0.03 (2
bodies), 1.28 / 0.67 (8), 3.21 / 2.66 (16), 12.4 / 16.9 (32), 47.6 / 176.1
and 17.5 / 59.7 (64, two runs); binary trees 3.21 / 0.69 (16), 12.7 / 1.87
(32), 49.9 / 5.44 (64). The sedan: constrained forward dynamics 53.3 / 45.4,
an RK4 step 282 / 244. Per multiply-add on the chain of 64: 3.9 ns for the
tree's, 1.1 ns for the dense one's.

## 2. Three versions of the same factorization (session 2, 19 October)

Scalar: as in section 1. Eigen segments: L^T stored (L's rows as columns),
each run of consecutively numbered ancestors one Eigen segment operation.
Plain loops (kept): L^T stored, each run a plain loop. Chains (each body a
child of the previous one) and binary trees (body i a child of body i / 2,
numbered breadth first, so most runs are one velocity long).

| model | work ratio | dense | scalar, L as in the book | Eigen segments | plain loops (kept) |
|---|---|---|---|---|---|
| revolute chain, 2 bodies | 3.00 | 0.09 | 0.02 | 0.05 | 0.03 |
| revolute chain, 4 bodies | 1.88 | 0.22 | 0.09 | 0.17 | 0.08 |
| revolute chain, 8 bodies | 1.41 | 0.58, 0.56 | 0.29 | 0.53 | 0.28 |
| revolute chain, 16 bodies | 1.20 | 1.41, 1.40 | 1.16, 1.17 | 1.84, 1.88 | 1.17 |
| revolute chain, 32 bodies | 1.10 | 5.38 | 7.39, 7.40 | 7.38, 7.39 | 4.76, 4.74 |
| revolute chain, 64 bodies | 1.05 | 20.68, 20.88 | 78.40, 77.23 | 34.31, 34.67 | 23.03, 23.05 |
| revolute tree, 2 bodies | 3.00 | 0.09 | 0.02 | 0.05 | 0.03 |
| revolute tree, 4 bodies | 1.22 | 0.22, 0.21 | 0.06 | 0.14 | 0.05 |
| revolute tree, 8 bodies | 0.48 | 0.57, 0.56 | 0.13 | 0.47 | 0.13 |
| revolute tree, 16 bodies | 0.18 | 1.40 | 0.30 | 1.53, 1.54 | 0.35 |
| revolute tree, 32 bodies | 0.07 | 5.36, 5.39 | 0.77, 0.78 | 4.69, 4.66 | 1.03, 1.05 |
| revolute tree, 64 bodies | 0.02 | 20.72, 20.92 | 1.99, 2.02 | 13.12 | 2.57, 2.55 |

The sedan (double wishbone, drivetrain; 27 velocities, work ratio 0.26):

| version | constrained forward dynamics, dense | the same, tree | one RK4 step of 1 ms, dense | the same, tree |
|---|---|---|---|---|
| scalar | 23.44, 23.17 | 20.09, 20.84 | 123.29, 122.83 | 107.23, 107.03 |
| Eigen segments | 24.05, 23.63 | 24.74, 24.33 | 123.37, 127.08 | 124.94, 128.31 |
| plain loops | 23.29, 23.65 | 19.67, 19.72 | 122.66, 125.38 | 105.11, 106.87 |

## 3. The crossover, and Eigen's sparse Cholesky (session 3, 19 October)

The kept version. A root body (on a revolute joint) carrying 1, 2 or 4
chains of revolute bodies numbered one chain after the other, n bodies in
all. Eigen sparse: `SimplicialLLT<SparseMatrix<double>, Lower,
NaturalOrdering<int>>` on M's lower triangle on the tree's pattern, the
pattern analysed once outside the timing, M's values copied in at each
call.

| model | work ratio | dense | tree | Eigen sparse |
|---|---|---|---|---|
| 1 chain, 12 bodies | 1.26 | 1.14, 1.18 | 0.80, 0.81 | 1.45 |
| 1 chain, 16 bodies | 1.20 | 1.68, 1.70 | 1.44, 1.48 | 2.41 |
| 1 chain, 24 bodies | 1.13 | 3.06, 3.13 | 3.11, 3.07 | 5.52, 5.36 |
| 1 chain, 32 bodies | 1.10 | 6.54, 6.46 | 5.67, 5.73 | 10.35, 10.45 |
| 1 chain, 48 bodies | 1.06 | 13.82, 13.96 | 14.37, 14.18 | 26.22, 27.08 |
| 1 chain, 64 bodies | 1.05 | 25.22, 25.63 | 27.60, 27.89 | 52.44, 53.78 |
| 2 chains, 12 bodies | 0.48 | 1.14, 1.15 | 0.56 | 1.36, 1.35 |
| 2 chains, 16 bodies | 0.42 | 1.61, 1.67 | 0.96, 1.00 | 2.21, 2.45 |
| 2 chains, 24 bodies | 0.36 | 3.07, 3.14 | 1.81, 1.84 | 4.96, 5.01 |
| 2 chains, 32 bodies | 0.33 | 6.41, 6.60 | 3.14, 3.04 | 9.57, 9.15 |
| 2 chains, 48 bodies | 0.30 | 13.47, 13.88 | 6.55, 6.57 | 23.80, 23.30 |
| 2 chains, 64 bodies | 0.29 | 24.57, 24.78 | 11.87, 11.89 | 46.32, 48.32 |
| 4 chains, 12 bodies | 0.23 | 1.13, 1.16 | 0.27, 0.29 | 1.28, 1.37 |
| 4 chains, 16 bodies | 0.18 | 1.65, 1.69 | 0.46 | 2.08, 2.12 |
| 4 chains, 24 bodies | 0.13 | 3.09, 3.05 | 1.31, 1.29 | 4.60, 4.70 |
| 4 chains, 32 bodies | 0.11 | 6.47, 6.41 | 1.90, 1.97 | 8.65, 8.73 |
| 4 chains, 48 bodies | 0.09 | 13.95, 13.87 | 3.84, 3.64 | 22.77, 22.50 |
| 4 chains, 64 bodies | 0.09 | 25.39, 25.55 | 6.17, 6.19 | 44.95, 47.27 |

The chains and binary trees of section 2, in this session:

| model | dense | tree | Eigen sparse |
|---|---|---|---|
| revolute chain, 2 bodies | 0.08, 0.09 | 0.03 | 0.23 |
| revolute chain, 4 bodies | 0.25, 0.26 | 0.10 | 0.41 |
| revolute chain, 8 bodies | 0.68 | 0.38, 0.37 | 0.82, 0.85 |
| revolute chain, 16 bodies | 1.72, 1.67 | 1.48, 1.44 | 2.45, 2.47 |
| revolute chain, 32 bodies | 6.66, 6.63 | 5.80, 5.67 | 10.60, 10.27 |
| revolute chain, 64 bodies | 25.64, 25.62 | 27.08, 27.86 | 53.89, 52.53 |
| revolute tree, 2 bodies | 0.10 | 0.03 | 0.27 |
| revolute tree, 4 bodies | 0.26 | 0.07 | 0.39 |
| revolute tree, 8 bodies | 0.68, 0.69 | 0.16 | 0.77, 0.76 |
| revolute tree, 16 bodies | 1.71, 1.66 | 0.46, 0.45 | 2.12, 2.13 |
| revolute tree, 32 bodies | 6.66, 6.65 | 1.35, 1.27 | 8.36, 8.58 |
| revolute tree, 64 bodies | 24.45, 24.95 | 4.16, 3.15 | 43.13, 43.08 |

Eigen's sparse Cholesky against the dense one: 1.10 to 2.88 times slower
over all the models above; against the tree's, 1.63 to 13.68 times slower.

## 4. The sedan against the previous commit (session 3)

`576df64` built from an archive, run alternately with this version
(new, old, new, old). The sedan rows (this version: dense, then tree; the
previous commit: dense):

| row | this version | previous commit |
|---|---|---|
| constrained forward dynamics, dense | 27.72, 28.79 | 27.88, 27.67 |
| constrained forward dynamics, tree | 23.37, 24.17 | |
| accelerations, with the forces (Automatic: tree) | 25.83, 26.73 | 29.55, 29.58 |
| one RK4 step of 1 ms, dense | 146.88, 151.04 | 145.35, 145.24 |
| one RK4 step of 1 ms, tree | 127.38, 134.34 | |

Every method on the sedan (Automatic: the tree's factorization in this
version), as in task 4.7's table: median of one 1 ms step; then single
steps over the drive.

| method | version | median of one step | single steps: min | median | 99th percentile |
|---|---|---|---|---|---|
| RK4 | this version | 126.1, 125.9 | 102.5, 103.6 | 127.5, 128.6 | 193.3, 205.4 |
| RK4 | 576df64 | 147.7, 143.6 | 119.1, 119.3 | 147.8, 148.0 | 182.6, 239.6 |
| generalised-alpha | this version | 87.9, 83.6 | 70.3, 70.3 | 87.4, 87.7 | 181.4, 210.5 |
| generalised-alpha | 576df64 | 99.0, 96.8 | 79.4, 79.3 | 98.8, 98.9 | 196.5, 227.9 |
| generalised-alpha, stabilized | this version | 126.0, 123.3 | 100.3, 101.2 | 125.1, 126.5 | 224.6, 249.2 |
| generalised-alpha, stabilized | 576df64 | 141.4, 140.1 | 113.9, 113.7 | 141.7, 141.7 | 249.0, 265.5 |
| linearly implicit Euler | this version | 166.1, 164.4 | 120.3, 121.1 | 155.3, 156.9 | 244.0, 234.3 |
| linearly implicit Euler | 576df64 | 173.5, 168.1 | 127.3, 126.5 | 164.5, 164.1 | 241.6, 215.3 |

Medians of one step, this version against the previous commit: RK4 126.0
against 145.7 (-13.5 %), generalised-alpha 85.8 against 97.9 (-12.4 %),
stabilized 124.7 against 140.8 (-11.4 %), linearly implicit Euler 165.3
against 170.8 (-3.2 %; its own iteration matrix W is factorized densely).

## 5. Agreement (the tests, with -s)

`test_kernel.exe "[sparse]" -s`, the worst case of each check relative to its
bound: L^T L - M 1.9e-16 of |M| (1.8e-15 for |M| 9.4); the tree's solve
against the dense one 5.1e-15 relative (9.8e-14 for a solution of norm 18);
the solver's accelerations 1.9e-15 relative (9.9e-14 for |a| 53); the
multipliers 1.1e-14; the four-bar's projected positions identical, its
velocities 9.9e-15 apart. The work counts: a chain of 16, 816; a binary tree
of 31, 351 (0.071 of 31^3 / 6); a binary tree of 63, 1023; a root with two
chains of 32, 13089.

`test_vehicle.exe "Dynamic template: the sedan drives the same with the
mass matrix factorized along its tree" -s`: after 500 steps of RK4 at 1 ms
(throttle and steering ramps) the positions differ by 1.0e-15 (|q| 2.3) and
the velocities by 1.2e-14.

`scripts\build.ps1 -Test`: 469 tests, no compiler warning.
