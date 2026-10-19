# Performance

How fast the dynamics algorithms are, measured with `bench/bench_dynamics.cpp` (plan task 2.10).

```powershell
scripts\build.ps1 -Target mbd_bench
build\bench\mbd_bench.exe
```

Each figure is the median of seven rounds, single thread, on the development laptop (Intel i7-13700H, Windows, MSVC 14.50, `/O2`). Compare runs on the same machine only. Timings on this laptop vary by up to a factor of two from one run to the next, so differences smaller than that need repeated runs.

The legacy code was removed in task 2.7b, so the benchmark now times the kernel only; the comparisons below were measured before that.

## Baseline: the Phase 1 kernel, 3 October 2026

The systems are chains and binary trees of revolute joints with generic axes and offsets, and the all-double-wishbone sedan from the vehicle template (26 coordinates, 16 constraint equations, moving at 0.2 s into a drive).

What the numbers show:

- **The mass matrix grows roughly with the cube of the body count**: 1.6 us for 2 bodies, 2650 us for a 64-body chain. Each body builds a dense Jacobian with one column per coordinate of the whole system. The composite-rigid-body algorithm of Phase 2 needs work proportional to the bodies times the depth of the tree.
- **Forward dynamics follows the mass matrix**: 1590 us for the 64-body chain, against 49 us for inverse dynamics. The articulated-body algorithm of Phase 2 grows linearly with the number of bodies.
- **The sedan** takes 547 us per RK4 step in this run (earlier runs measured 840 to 1430 us). The target of task 2.10 is under 250 us on average.

| Model | Operation | Time per call [us] |
|---|---|---|
| revolute chain, 2 bodies           | positions and velocities           |       0.69 |
| revolute chain, 2 bodies           | mass matrix                        |       1.57 |
| revolute chain, 2 bodies           | inverse dynamics                   |       1.59 |
| revolute chain, 2 bodies           | forward dynamics                   |       3.53 |
| revolute chain, 4 bodies           | positions and velocities           |       1.36 |
| revolute chain, 4 bodies           | mass matrix                        |       5.19 |
| revolute chain, 4 bodies           | inverse dynamics                   |       2.81 |
| revolute chain, 4 bodies           | forward dynamics                   |       8.67 |
| revolute chain, 8 bodies           | positions and velocities           |       2.64 |
| revolute chain, 8 bodies           | mass matrix                        |      21.86 |
| revolute chain, 8 bodies           | inverse dynamics                   |       5.24 |
| revolute chain, 8 bodies           | forward dynamics                   |      28.36 |
| revolute chain, 16 bodies          | positions and velocities           |       5.23 |
| revolute chain, 16 bodies          | mass matrix                        |      88.05 |
| revolute chain, 16 bodies          | inverse dynamics                   |      17.12 |
| revolute chain, 16 bodies          | forward dynamics                   |      93.07 |
| revolute chain, 32 bodies          | positions and velocities           |      13.41 |
| revolute chain, 32 bodies          | mass matrix                        |     377.98 |
| revolute chain, 32 bodies          | inverse dynamics                   |      24.12 |
| revolute chain, 32 bodies          | forward dynamics                   |     329.00 |
| revolute chain, 64 bodies          | positions and velocities           |      22.34 |
| revolute chain, 64 bodies          | mass matrix                        |    2650.47 |
| revolute chain, 64 bodies          | inverse dynamics                   |      49.27 |
| revolute chain, 64 bodies          | forward dynamics                   |    1589.54 |
| revolute tree, 2 bodies            | positions and velocities           |       0.75 |
| revolute tree, 2 bodies            | mass matrix                        |       1.58 |
| revolute tree, 2 bodies            | inverse dynamics                   |       1.59 |
| revolute tree, 2 bodies            | forward dynamics                   |       3.51 |
| revolute tree, 4 bodies            | positions and velocities           |       1.36 |
| revolute tree, 4 bodies            | mass matrix                        |       4.67 |
| revolute tree, 4 bodies            | inverse dynamics                   |       2.90 |
| revolute tree, 4 bodies            | forward dynamics                   |      10.13 |
| revolute tree, 8 bodies            | positions and velocities           |       2.68 |
| revolute tree, 8 bodies            | mass matrix                        |      17.59 |
| revolute tree, 8 bodies            | inverse dynamics                   |       5.28 |
| revolute tree, 8 bodies            | forward dynamics                   |      24.19 |
| revolute tree, 16 bodies           | positions and velocities           |       5.25 |
| revolute tree, 16 bodies           | mass matrix                        |      32.53 |
| revolute tree, 16 bodies           | inverse dynamics                   |      10.21 |
| revolute tree, 16 bodies           | forward dynamics                   |      44.57 |
| revolute tree, 32 bodies           | positions and velocities           |      10.38 |
| revolute tree, 32 bodies           | mass matrix                        |     122.98 |
| revolute tree, 32 bodies           | inverse dynamics                   |      19.82 |
| revolute tree, 32 bodies           | forward dynamics                   |     152.28 |
| revolute tree, 64 bodies           | positions and velocities           |      21.14 |
| revolute tree, 64 bodies           | mass matrix                        |     991.60 |
| revolute tree, 64 bodies           | inverse dynamics                   |      39.17 |
| revolute tree, 64 bodies           | forward dynamics                   |    1047.09 |
| double-wishbone sedan, 26 DOF      | mass matrix                        |      37.13 |
| double-wishbone sedan, 26 DOF      | constrained forward dynamics       |      94.88 |
| double-wishbone sedan, 26 DOF      | constraint projection              |      88.95 |
| double-wishbone sedan, 26 DOF      | one RK4 step of 1 ms, everything   |     547.33 |

## Phase 2 kernel against the legacy algorithms, 3 October 2026

The same models built in both, timed in the same run (task 2.4). The laptop
ran about 1.7 times slower in this run than in the baseline above (compare
the legacy columns), so read the ratio, which is legacy time over kernel time.

What the numbers show:

- **Forward dynamics**: the articulated-body algorithm grows linearly with the
  number of bodies, so the gain grows with size: 4 times at 2 bodies, 46
  times for the 64-body chain (97 us against 4.5 ms).
- **Mass matrix**: the composite-rigid-body algorithm is 4 to 36 times faster.
- **Kinematics and inverse dynamics** were already linear in the legacy code;
  the kernel is about 2 to 3 times faster.
- **The sedan** closes its suspension loops with constraints, which move onto
  the kernel with tasks 2.5 to 2.7; until then only its legacy timings exist.
  Its RK4 step took 1859 us in this slow run (547 us in the baseline run).

| Model | Operation | Legacy [us] | Kernel [us] | Ratio |
|---|---|---|---|---|
| revolute chain, 2 bodies       | positions and velocities         |       0.94 |       0.47 |    2.0 |
| revolute chain, 2 bodies       | mass matrix                      |       2.39 |       0.59 |    4.0 |
| revolute chain, 2 bodies       | inverse dynamics                 |       2.09 |       0.81 |    2.6 |
| revolute chain, 2 bodies       | forward dynamics                 |       4.69 |       1.19 |    4.0 |
| revolute chain, 4 bodies       | positions and velocities         |       1.76 |       0.96 |    1.8 |
| revolute chain, 4 bodies       | mass matrix                      |      17.36 |       3.26 |    5.3 |
| revolute chain, 4 bodies       | inverse dynamics                 |      11.63 |       3.55 |    3.3 |
| revolute chain, 4 bodies       | forward dynamics                 |      26.75 |       5.73 |    4.7 |
| revolute chain, 8 bodies       | positions and velocities         |       9.85 |       5.07 |    1.9 |
| revolute chain, 8 bodies       | mass matrix                      |      72.34 |       7.98 |    9.1 |
| revolute chain, 8 bodies       | inverse dynamics                 |      18.96 |       7.18 |    2.6 |
| revolute chain, 8 bodies       | forward dynamics                 |      84.38 |      11.91 |    7.1 |
| revolute chain, 16 bodies      | positions and velocities         |      25.70 |       9.60 |    2.7 |
| revolute chain, 16 bodies      | mass matrix                      |     217.91 |      31.36 |    6.9 |
| revolute chain, 16 bodies      | inverse dynamics                 |      35.62 |      14.09 |    2.5 |
| revolute chain, 16 bodies      | forward dynamics                 |     230.80 |      23.32 |    9.9 |
| revolute chain, 32 bodies      | positions and velocities         |      38.36 |      19.18 |    2.0 |
| revolute chain, 32 bodies      | mass matrix                      |     898.90 |      82.76 |   10.9 |
| revolute chain, 32 bodies      | inverse dynamics                 |      67.35 |      28.20 |    2.4 |
| revolute chain, 32 bodies      | forward dynamics                 |    1047.01 |      48.14 |   21.7 |
| revolute chain, 64 bodies      | positions and velocities         |      73.98 |      42.09 |    1.8 |
| revolute chain, 64 bodies      | mass matrix                      |    4612.52 |     316.86 |   14.6 |
| revolute chain, 64 bodies      | inverse dynamics                 |     134.82 |      71.09 |    1.9 |
| revolute chain, 64 bodies      | forward dynamics                 |    4509.70 |      97.42 |   46.3 |
| revolute tree, 2 bodies        | positions and velocities         |       2.41 |       1.22 |    2.0 |
| revolute tree, 2 bodies        | mass matrix                      |       5.17 |       1.31 |    4.0 |
| revolute tree, 2 bodies        | inverse dynamics                 |       5.22 |       1.80 |    2.9 |
| revolute tree, 2 bodies        | forward dynamics                 |      11.81 |       2.56 |    4.6 |
| revolute tree, 4 bodies        | positions and velocities         |       5.00 |       2.93 |    1.7 |
| revolute tree, 4 bodies        | mass matrix                      |      17.59 |       3.07 |    5.7 |
| revolute tree, 4 bodies        | inverse dynamics                 |       9.44 |       4.10 |    2.3 |
| revolute tree, 4 bodies        | forward dynamics                 |      27.43 |       7.50 |    3.7 |
| revolute tree, 8 bodies        | positions and velocities         |       9.34 |       4.93 |    1.9 |
| revolute tree, 8 bodies        | mass matrix                      |      55.22 |       6.44 |    8.6 |
| revolute tree, 8 bodies        | inverse dynamics                 |      18.50 |       6.99 |    2.6 |
| revolute tree, 8 bodies        | forward dynamics                 |      71.26 |      11.84 |    6.0 |
| revolute tree, 16 bodies       | positions and velocities         |      20.62 |       9.59 |    2.2 |
| revolute tree, 16 bodies       | mass matrix                      |     127.58 |      15.13 |    8.4 |
| revolute tree, 16 bodies       | inverse dynamics                 |      36.28 |      14.56 |    2.5 |
| revolute tree, 16 bodies       | forward dynamics                 |     175.09 |      24.33 |    7.2 |
| revolute tree, 32 bodies       | positions and velocities         |      37.66 |      20.10 |    1.9 |
| revolute tree, 32 bodies       | mass matrix                      |     404.33 |      35.17 |   11.5 |
| revolute tree, 32 bodies       | inverse dynamics                 |      66.48 |      28.46 |    2.3 |
| revolute tree, 32 bodies       | forward dynamics                 |     501.20 |      47.87 |   10.5 |
| revolute tree, 64 bodies       | positions and velocities         |      84.38 |      37.19 |    2.3 |
| revolute tree, 64 bodies       | mass matrix                      |    2722.55 |      75.44 |   36.1 |
| revolute tree, 64 bodies       | inverse dynamics                 |     133.45 |      65.04 |    2.1 |
| revolute tree, 64 bodies       | forward dynamics                 |    2860.38 |     104.09 |   27.5 |
| double-wishbone sedan, 26 DOF  | mass matrix                      |     120.35 |            |        |
| double-wishbone sedan, 26 DOF  | constrained forward dynamics     |     311.78 |            |        |
| double-wishbone sedan, 26 DOF  | constraint projection            |     291.80 |            |        |
| double-wishbone sedan, 26 DOF  | one RK4 step of 1 ms, everything |    1859.08 |            |        |

## The sedan on the kernel, 4 October 2026

The all-double-wishbone sedan, now built on the kernel (plan task 2.7b):
chassis on a free joint, a steering rack, four double-wishbone corners
closed by 17 constraint equations, springs, dampers and tyres, driven
through the drivetrain. Same run as the chain and tree timings above, in
which the legacy sedan took 739 us per step.

| Operation | Kernel [us] |
|---|---|
| mass matrix | 6.1 |
| constrained forward dynamics | 42.0 |
| constraint projection | 47.1 |
| one RK4 step of 1 ms, everything | 291.5 |

Task 2.10's target is 250 us. The step is four evaluations of about 42 us of
constrained dynamics each, one projection, and about 75 us for the force
elements and bookkeeping. Each evaluation still runs the kinematics four
times (the simulator, the constraints, the mass matrix and the bias forces
each do their own pass); sharing one pass is the obvious first saving.

## Task 2.10: the sedan step, 5 October 2026

**How to measure.** The i7-13700H mixes performance and efficiency cores,
and its clock follows temperature and the power plan. Unpinned, the same
benchmark varied by more than a factor of two within one hour, because a
single-threaded run lands on either kind of core. The figures below are
therefore taken pinned to logical processor 2, a performance core, at high
priority (PowerShell: start `mbd_bench.exe`, then set the process's
`ProcessorAffinity` to 4 and `PriorityClass` to `High`). Two runs pinned so
agree within 1 %. To compare two versions of the code, build both and run
them alternately in the same session.

**The step.** The commit of 4 October (`24faade`, built separately) against
this one, alternately, two rounds each, pinned:

| Operation | 4 October [us] | Now [us] |
|---|---|---|
| constrained forward dynamics | 30.4, 30.5 | 23.7, 23.6 |
| constraint projection (state already on the constraints) | 33.9, 34.2 | 24.1, 23.8 |
| one RK4 step of 1 ms, everything | 180.8, 182.3 | 124.4, 124.2 |

What changed:

- **One kinematics pass per evaluation instead of four.** The simulator,
  the constraint equations, the mass matrix and the bias forces each ran
  their own. Now `forward_kinematics(q, v, 0)` serves them all:
  `mass_matrix` and `bias_forces` take the placements, velocities and
  zero-acceleration body accelerations it leaves, and the simulator hands
  the pass it made for the forces to `forward_dynamics_from_kinematics`.
- **No `M^-1 J^T`.** The constraint matrix is `Z^T Z` with `Z = L^-1 J^T`,
  one triangular solve instead of two, and `M^-1 J^T lambda` is applied as
  `L^-T (Z lambda)`.
- **A cheaper projection.** It weighs by the mass matrix the step's last
  stage factorized instead of computing one more, and its velocity part
  uses the J and nu of the last position evaluation instead of evaluating
  again.

Where an evaluation's 25.8 us go now: kinematics 2.7, constraint equations
5.9, mass matrix 2.0 beyond the kinematics, bias forces 1.3, force elements
and body states about 2, and the linear algebra (Cholesky of the 27 x 27
mass matrix, `Z`, `Z^T Z`, its pivoted factorization and the solves) about
11. A step is four evaluations, one projection and a final kinematics pass.

**Against the target.** The step takes 124 us, half the 250 us of task 2.10.
Earlier on the same day the machine was slower for reasons outside the code:
the code of that morning, which runs as fast as the 4 October code, took 395
to 413 us pinned, 2.2 times its time above. At that factor the step would
take about 280 us. The target holds in the laptop's normal state, not in
that one.

**Allocations.** `test_alloc` counts every allocation through `operator new`
during 100 steps of this sedan, tyres and drivetrain included, and of a
four-bar with both integrators: none. The CI job `kernel-no-malloc` also runs
it with Eigen's own check on.

The whole benchmark, pinned:

| Model | Operation | Time [us] |
|---|---|---|
| revolute chain, 2 bodies       | positions and velocities         |       0.36 |
| revolute chain, 2 bodies       | mass matrix                      |       0.40 |
| revolute chain, 2 bodies       | inverse dynamics                 |       0.57 |
| revolute chain, 2 bodies       | forward dynamics                 |       0.82 |
| revolute chain, 4 bodies       | positions and velocities         |       0.73 |
| revolute chain, 4 bodies       | mass matrix                      |       1.02 |
| revolute chain, 4 bodies       | inverse dynamics                 |       1.10 |
| revolute chain, 4 bodies       | forward dynamics                 |       1.67 |
| revolute chain, 8 bodies       | positions and velocities         |       1.42 |
| revolute chain, 8 bodies       | mass matrix                      |       2.85 |
| revolute chain, 8 bodies       | inverse dynamics                 |       2.26 |
| revolute chain, 8 bodies       | forward dynamics                 |       3.46 |
| revolute chain, 16 bodies      | positions and velocities         |       2.82 |
| revolute chain, 16 bodies      | mass matrix                      |       8.23 |
| revolute chain, 16 bodies      | inverse dynamics                 |       4.55 |
| revolute chain, 16 bodies      | forward dynamics                 |       7.08 |
| revolute chain, 32 bodies      | positions and velocities         |       5.80 |
| revolute chain, 32 bodies      | mass matrix                      |      27.85 |
| revolute chain, 32 bodies      | inverse dynamics                 |       9.40 |
| revolute chain, 32 bodies      | forward dynamics                 |      14.23 |
| revolute chain, 64 bodies      | positions and velocities         |      11.49 |
| revolute chain, 64 bodies      | mass matrix                      |      97.71 |
| revolute chain, 64 bodies      | inverse dynamics                 |      18.38 |
| revolute chain, 64 bodies      | forward dynamics                 |      28.83 |
| revolute tree, 2 bodies        | positions and velocities         |       0.37 |
| revolute tree, 2 bodies        | mass matrix                      |       0.39 |
| revolute tree, 2 bodies        | inverse dynamics                 |       0.59 |
| revolute tree, 2 bodies        | forward dynamics                 |       0.79 |
| revolute tree, 4 bodies        | positions and velocities         |       0.74 |
| revolute tree, 4 bodies        | mass matrix                      |       0.94 |
| revolute tree, 4 bodies        | inverse dynamics                 |       1.10 |
| revolute tree, 4 bodies        | forward dynamics                 |       1.66 |
| revolute tree, 8 bodies        | positions and velocities         |       1.42 |
| revolute tree, 8 bodies        | mass matrix                      |       2.17 |
| revolute tree, 8 bodies        | inverse dynamics                 |       2.27 |
| revolute tree, 8 bodies        | forward dynamics                 |       3.54 |
| revolute tree, 16 bodies       | positions and velocities         |       2.94 |
| revolute tree, 16 bodies       | mass matrix                      |       4.87 |
| revolute tree, 16 bodies       | inverse dynamics                 |       4.27 |
| revolute tree, 16 bodies       | forward dynamics                 |       7.21 |
| revolute tree, 32 bodies       | positions and velocities         |       5.62 |
| revolute tree, 32 bodies       | mass matrix                      |      11.55 |
| revolute tree, 32 bodies       | inverse dynamics                 |       8.67 |
| revolute tree, 32 bodies       | forward dynamics                 |      13.99 |
| revolute tree, 64 bodies       | positions and velocities         |      11.17 |
| revolute tree, 64 bodies       | mass matrix                      |      25.48 |
| revolute tree, 64 bodies       | inverse dynamics                 |      17.42 |
| revolute tree, 64 bodies       | forward dynamics                 |      27.85 |
| double-wishbone sedan          | positions and velocities         |       2.65 |
| double-wishbone sedan          | constraint equations             |       8.53 |
| double-wishbone sedan          | mass matrix                      |       4.63 |
| double-wishbone sedan          | bias forces                      |       3.96 |
| double-wishbone sedan          | constrained forward dynamics     |      23.65 |
| double-wishbone sedan          | accelerations, with the forces   |      25.79 |
| double-wishbone sedan          | constraint projection            |      24.21 |
| double-wishbone sedan          | one RK4 step of 1 ms, everything |     127.79 |


## Task 4.5: the integrators on the sedan, 16 October 2026

`build\bench\mbd_bench.exe`, pinned to a performance core at high priority
(D21), two runs. The double-wishbone sedan with its drivetrain (27
velocities, 4 wheel spins). First column: the median time of one 1 ms step at
a moving state, as above. Then single steps over a drive (half throttle for
1 s, brake to a stop for 2 s, stand for 1 s), the first ten left out.

| Method | median of one step [us] | single steps: min | median | 99th percentile | 99.9th | max |
|---|---|---|---|---|---|---|
| RK4 | 122.2, 122.3 | 120.6, 119.9 | 121.7, 121.1 | 155.6, 144.9 | 264.2, 221.4 | 780.5, 262.8 |
| generalised-alpha | 81.3, 80.7 | 79.5, 79.2 | 80.8, 80.4 | 172.2, 160.0 | 322.5, 488.9 | 2337.4, 2103.9 |
| linearly implicit Euler | 108.1, 109.4 | 99.6, 99.5 | 106.3, 106.5 | 136.7, 141.5 | 228.2, 991.3 | 258.4, 2300.8 |

What the numbers show:

- **Every linearly implicit step does the same work** (one evaluation, one
  set of force derivatives, one constrained solve; a test checks it on
  every step of the drive), and its times spread as RK4's do, which also
  does fixed work: 99 % of the steps within 1.3 times the median.
- **Generalised-alpha's longest step is its own**: 2.1 to 2.3 ms in both
  runs, when it builds a new iteration matrix (2 nv + nx = 58 evaluations).
  It is cheaper per step than either other method, but a real-time budget
  must allow for that step.
- **The longest single steps of RK4 and the linearly implicit method differ
  between runs** (263 against 781 us, 258 against 2301 us): the operating
  system's, not the method's. Windows is not a real-time system (tasks 12.5
  to 12.8).
- The remaining variation within the method: the projection after the step
  takes a Gauss-Newton iteration or none, as the drift requires (task 4.7).

## Task 4.7: generalised-alpha on the constraints, 17 October 2026

`build\bench\mbd_bench.exe`, pinned to a performance core at high priority
(D21), against the previous commit's (`08099dd`, built from an archive with
the same two columns added), run alternately new, old, new, old. Columns as
in task 4.5, then the evaluations and Newton iterations per step over the
drive. The sedan has 27 velocities, 4 wheel spins and 17 constraint
equations.

| Method | median of one step [us] | single steps: min | median | 99th percentile | 99.9th | max | evaluations per step | Newton iterations per step |
|---|---|---|---|---|---|---|---|---|
| RK4, new | 155.1, 150.1 | 119.3, 119.3 | 148.2, 148.0 | 304.6, 293.0 | 1178.6, 686.5 | 2952.6, 894.3 | 4.00 | 0 |
| RK4, old | 151.5, 149.3 | 120.0, 119.7 | 149.0, 148.6 | 310.5, 319.3 | 1088.5, 1069.1 | 1226.8, 1507.4 | 4.00 | 0 |
| generalised-alpha, new | 100.3, 100.1 | 79.7, 79.8 | 99.9, 100.1 | 287.2, 282.2 | 1119.6, 469.1 | 2486.1, 2255.3 | 2.41 | 1.37 |
| generalised-alpha, old | 99.9, 100.7 | 79.2, 79.5 | 99.0, 99.1 | 273.0, 276.8 | 1020.0, 1297.5 | 2491.8, 2621.9 | 2.41 | 1.37 |
| generalised-alpha, stabilized (new only) | 148.9, 142.9 | 114.3, 112.3 | 146.6, 145.8 | 342.4, 357.2 | 612.0, 1019.5 | 2362.3, 2417.4 | 2.41 | 1.37 |
| linearly implicit Euler, new | 132.5, 138.8 | 99.3, 99.7 | 130.3, 130.5 | 282.7, 275.3 | 676.0, 501.1 | 1270.9, 1028.6 | 1.00 | 0 |
| linearly implicit Euler, old | 135.4, 133.7 | 100.1, 100.1 | 130.9, 130.8 | 277.4, 279.1 | 494.8, 710.8 | 1079.6, 1945.3 | 1.00 | 0 |

What the numbers show:

- **The default is unchanged**: every method still projects after its
  steps, and new and old agree within the run-to-run spread.
- **Generalised-alpha solved on the constraints costs 1.45 times as much a
  step** (146 against 100 us) for the same evaluations and Newton
  iterations: its Newton matrix has 31 + 17 rows instead of 31 (48, with
  the position correction eliminated; 169 us with it kept, 65 rows), and
  forming that matrix's blocks costs more than its LU. With the
  constraints not enforced at all the step takes 86 us, so the projection
  costs some 13 us and the stabilized form some 59. Its LU at 48 rows
  takes 27 us with our routine and 20 with Eigen's blocked one. Hence it is
  an option (D37).
- **The tails are wider than in task 4.5** (99th percentiles of 270 to 360
  us against 136 to 172) for old and new alike: the machine's state that
  day (about 1.4 GB of memory free), not the code.
- **The linearly implicit step still projects** (task 4.5's remaining
  variation: a Gauss-Newton iteration or none): the projection costs some
  13 us of a 130 us step. Holding the constraints inside the step instead
  was not tried for it; for generalised-alpha it cost four and a half times the
  projection.

## D38: the inertia's velocity products implicit in the Euler steps, 18 October 2026

The same benchmark and protocol as task 4.7 above, against `08099dd`, whose
linearly implicit step is the one before D38, run new, old, new, old,
pinned. The linearly implicit step's figures (the other methods unchanged):

| Version | median of one step [us] | single steps: min | median | 99th percentile |
|---|---|---|---|---|
| first version | 194.4, 193.8 | 144.3, 144.6 | 185.4, 187.1 | 492.2, 265.9 |
| kept, new | 165.6, 166.8 | 126.3, 126.3 | 159.1, 163.5 | 275.0, 272.1 |
| before D38, old | 131.4, 129.9 | 100.4, 99.9 | 131.2, 130.6 | 197.2, 183.6 |

What the numbers show:

- **The step costs about 30 us more** (a quarter): the derivative of the
  velocity products (D, a linearized RNEA per velocity over the joint's
  subtree and path), an LU of W instead of a Cholesky (D is not symmetric),
  with a Cholesky of W's symmetric part kept as the check behind MBD-K045,
  and the solves with all of W^-1 J^T's columns at once.
- **The first version cost twice that**: a joint-model call per velocity
  even where the motion subspace is constant, a backward pass over every
  body for every column, and the solves one column at a time.
- Every step still does the same work: one evaluation (1.00 a step over the
  drive).

## Task 4.8: the mass matrix along the tree, 19 October 2026

The benchmark's rows "mass matrix factor and solve" time one factorization
of an assembled mass matrix and one solve: Eigen's dense Cholesky, the
factorization along the tree (`TreeCholesky`, D39) and, for comparison,
Eigen's general sparse Cholesky (`SimplicialLLT`, natural ordering, pattern
analysed once). A sweep of a root body carrying 1, 2 or 4 chains locates the
crossover. Pinned, two runs alternated with the previous commit (`576df64`,
built from an archive); figures from one session (raw data:
`documentation/journal/data/2026-10-19_sparse-solve.md`).

| model (velocities) | work ratio | dense [us] | tree [us] | Eigen sparse [us] |
|---|---|---|---|---|
| chain (16) | 1.20 | 1.72, 1.67 | 1.48, 1.44 | 2.45, 2.47 |
| chain (32) | 1.10 | 6.66, 6.63 | 5.80, 5.67 | 10.60, 10.27 |
| chain (48) | 1.06 | 13.82, 13.96 | 14.37, 14.18 | 26.22, 27.08 |
| chain (64) | 1.05 | 25.64, 25.62 | 27.08, 27.86 | 53.89, 52.53 |
| root and 2 chains (64) | 0.29 | 24.57, 24.78 | 11.87, 11.89 | 46.32, 48.32 |
| root and 4 chains (64) | 0.09 | 25.39, 25.55 | 6.17, 6.19 | 44.95, 47.27 |
| binary tree (64) | 0.02 | 24.45, 24.95 | 4.16, 3.15 | 43.13, 43.08 |

The sedan (27 velocities, work ratio 0.26), every method, median of one 1 ms
step [us]:

| Method | before (576df64) | after |
|---|---|---|
| RK4 | 147.7, 143.6 | 126.1, 125.9 |
| generalised-alpha | 99.0, 96.8 | 87.9, 83.6 |
| generalised-alpha, stabilized | 141.4, 140.1 | 126.0, 123.3 |
| linearly implicit Euler | 173.5, 168.1 | 166.1, 164.4 |

What the numbers show:

- **The tree's factorization wins on every branched tree**, by 2 times with
  two chains and 4 to 8 with four or a binary tree, and on chains up to 32
  velocities; Eigen's blocked dense Cholesky wins on longer chains (2 to 4 %
  at 48, 8 to 11 % at 64 in two sessions). `ConstraintSolver`'s Automatic
  setting follows: dense only above 40 velocities with the tree's work at
  least 0.8 of the dense one's.
- **On the sedan** the constrained forward dynamics takes 16 % less time, an
  RK4 step 13 % less, generalised-alpha 11 to 12 %; the linearly implicit
  step only 3 %, since it factorizes its own iteration matrix W, which need
  not have the tree's pattern, densely.
- **A general sparse Cholesky does not pay at these sizes**: Eigen's is
  slower than the dense one on every model (1.1 to 2.9 times).
- **How the loops are written matters as much as the operation count.** The
  same factorization stored as in the book (L's rows walked in a
  column-major matrix) took 3.8 times the dense time on a chain of 64; each
  run of consecutive ancestors as an Eigen segment operation, 6.5 times its
  scalar time on a binary tree of 64; plain loops over runs of L^T's columns
  keep both cases fast (data, section 2).
