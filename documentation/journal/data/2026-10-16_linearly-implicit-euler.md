# Linearly implicit Euler: checks and measurements (task 4.5)

Machine: the development laptop (i7-13700H, 16 GB), Windows 11, MSVC
RelWithDebInfo, one compiler. Counts of evaluations and derivative sets come
from `IntegratorStats`; times from the benchmark, pinned (section 5).

## 1. The tests (`scripts\build.ps1 -Test`, 454 tests, no compiler warning)

`test_kernel.exe "[linear]" -s` and `test_vehicle.exe "[linear]" -s`:

- **Linear oscillator** (1 kg, k = 1 N/m, c = 0.3 N s/m, both joint forces;
  h = 0.1 s, 100 steps): largest difference from backward Euler's recurrence
  (1 + h c + h^2 k) v+ = v - h k q, q+ = q + h v+: 1.54807e-13; 100
  evaluations, 100 derivative sets.
- **Spring 1000 times too stiff for RK4** (w h = 89.4): after one step q =
  1.24984e-5 m and v = -99.9875 m/s, as the recurrence; amplitude ratio
  0.0111796 = 1 / sqrt(1 + (w h)^2).
- **First order** on the bushed pendulum with soft bushings (1e4 N/m, 1e2 N
  m/rad), tip after 1 s against Dormand-Prince at 1e-12: 4.1197e-3 m at 2 ms,
  1.99134e-3 m at 1 ms, ratio 2.06881.
- **force_jacobian** on the bushed pendulum (1e7, 1e5) with a spring-damper
  (2e4 N/m, 300 N s/m) between points off the links' axes, against central
  differences (1e-6) of the generalized forces computed by the kernel's own
  functions: damping at a deflected, moving state, largest entry 2829.34,
  largest difference 3.27e-6 (1.2e-9 of it); stiffness at rest and
  undeflected (no load, so no geometric term), largest entry 2.002e7,
  largest difference 0.50 (2.5e-8 of it).
- **Negative stiffness** (-4e6 N/m on 1 kg at 1 ms): MBD-K045 once, the steps
  finite.
- **The car rolling at 0.5 m/s at 5 ms** (simple vehicle, drivetrain): speed
  0.500317 -> 0.500306 m/s over 2 s (RK4 at 0.5 ms: 0.500209), every wheel
  within 5.51e-5 of free rolling; 460 steps, 460 evaluations, 460 derivative
  sets.
- **The sedan through a drive** (double-wishbone, drivetrain; half throttle 1
  s, brake 0.6 to a stop for 2 s, stand 1 s; 1 ms): work per step
  (evaluations x 1000 + derivative sets) 1001 on every one of 4000 steps;
  top speed 1.46842 m/s against RK4's 1.46854; speed at the end -9.7e-7 m/s
  (RK4: -4.0e-4).
- No allocation while stepping the sedan or the four-bar; braking to a stop
  locks every wheel.

## 2. The first version: states implicit in themselves only

Temporary hidden test case in `tests/test_drivetrain.cpp` (removed after
use): the rolling car at steps of 1 to 10 ms, then the sedan through the
drive above with, per method, the range of evaluations, derivative sets and
projection iterations per step and unpinned times.

With only X = d x_dot / dx (no coupling of the spins to the bodies):

| run | result |
|---|---|
| rolling, 1 ms | 0.500331 -> 0.500319 m/s, wheels within 6.1e-5 |
| rolling, 5 ms | 0.500624 -> 0.500580, within 1.2e-4 |
| sedan, top speed after 1 s of half throttle | 1.168 m/s (RK4 1.469, generalised-alpha 1.468) |

The tyre's slip makes its force a steep function of both the hub's velocity
and the spin; implicit in each separately, the hub is pulled towards the
spin of the step's start every step, a strong damper that brakes the car.

## 3. Coupling the states, and the energy from Xq

With Tx = d tau / dx, Xv and Xq = d x_dot / dv and dq (Xq and Xv through the
element differences), the sedan's top speed came to 1.468 m/s, but the
coasting car gained speed. Speed after 2 s, worst wheel deviation from free
rolling, by what the coupling includes (temporary switches):

| step | Tx, Xv, Xq | Tx, Xv (kept) | none |
|---|---|---|---|
| 1 ms | 0.500218 -> 0.500229, 3.8e-5 | 0.500222 -> 0.500218, 3.9e-5 | 0.500331 -> 0.500319, 6.1e-5 |
| 2 ms | 0.500235 -> 0.500375, 3.7e-5 | 0.500237 -> 0.500232, 4.1e-5 | 0.500423 -> 0.500402, 7.9e-5 |
| 3 ms | 0.500286 -> 0.500819, 3.6e-5 | 0.500258 -> 0.500251, 4.5e-5 | 0.500499 -> 0.500470, 9.4e-5 |
| 5 ms | 0.500577 -> 0.503331, 3.2e-5 | 0.500317 -> 0.500306, 5.5e-5 | 0.500624 -> 0.500580, 1.2e-4 |
| 10 ms | 0.503468 -> 0.529576, 1.1e-4 | 0.500564 -> 0.500535, 9.6e-5 | 0.500868 -> 0.500793, 1.5e-4 |

Norms at 5 ms, every 200 steps (Frobenius): |Tx Xv| 5.88e8, its
antisymmetric part 0.05 to 32 (symmetric to 1e-7); |C| 6.2e5, antisymmetric
part 6e-5; |K| 2.09e6, antisymmetric part 4.27e5 (the tyres' forces turn
with the wheels); |X| 4.08e3; |Xq| 1.86e4. Xq comes from `FullTireForce`:
R_eff = R_free - deflection / 2, so at constant spin the wheel's height
changes the slip by about 0.5 / R per metre, times the slip stiffness and
r / I_w: 2e4, as measured. The suspicion that the symmetric part of Tx G was
to blame was wrong (it is symmetric); the gain came with Xq.

## 4. The sedan, final code (unpinned, from the temporary case)

nq 32, nv 27, nx 4. Over the drive: linearly implicit Euler 1 evaluation and
1 derivative set on every step, projection iterations 0 to 1; RK4 4 and 0,
0 to 1; generalised-alpha 2 to 76 evaluations and 0 to 1 matrices. Top
speeds 1.468, 1.469, 1.468 m/s.

## 5. Timing, pinned

`scripts\build.ps1 -Target mbd_bench`, then `build\bench\mbd_bench.exe`
started from PowerShell with `ProcessorAffinity = 1 -shl 2` and
`PriorityClass = High`, twice. The table is in `docs/performance.md` (task
4.5). The first run before the warm-up steps were left out gave maxima of
1042.6 (RK4), 2002.8 (generalised-alpha) and 1025.6 us (linearly implicit),
from the first step's sizing and cold caches; the two runs after:

```
| RK4 | 122.2 | 120.6 | 121.7 | 155.6 | 264.2 | 780.5 |
| generalised-alpha | 81.3 | 79.5 | 80.8 | 172.2 | 322.5 | 2337.4 |
| linearly implicit Euler | 108.1 | 99.6 | 106.3 | 136.7 | 228.2 | 258.4 |

| RK4 | 122.3 | 119.9 | 121.1 | 144.9 | 221.4 | 262.8 |
| generalised-alpha | 80.7 | 79.2 | 80.4 | 160.0 | 488.9 | 2103.9 |
| linearly implicit Euler | 109.4 | 99.5 | 106.5 | 141.5 | 991.3 | 2300.8 |
```

(median of one step at a moving state; single steps over the drive: min,
median, 99th, 99.9th percentile, max; microseconds). The same run's sedan
rows: positions and velocities 2.65, constraint equations 8.23, mass matrix
4.58, accelerations with the forces 25.80, constraint projection 23.46, one
RK4 step 123.20 us.
