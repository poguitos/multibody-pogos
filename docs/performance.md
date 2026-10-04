# Performance

How fast the dynamics algorithms are, measured with `bench/bench_dynamics.cpp` (plan task 2.10).

```powershell
scripts\build.ps1 -Target mbd_bench
build\bench\mbd_bench.exe
```

Each figure is the median of seven rounds, single thread, on the development laptop (Intel i7-13700H, Windows, MSVC 14.50, `/O2`). Compare runs on the same machine only. Timings on this laptop vary by up to a factor of two from one run to the next, so differences smaller than that need repeated runs.

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
