# The inertia's velocity products implicit: measurements (task 4.7's open end, D38)

Machine: the development laptop (i7-13700H, 16 GB, about 1.4 GB free),
Windows 11, MSVC RelWithDebInfo, one compiler.

## 1. The defect

From task 4.7's long runs (`2026-10-17_constraints-per-integrator.md`,
section 5): the heavy top (spinning at 30 rad/s, held at its tip) with
semi-implicit Euler and with the linearly implicit step, at 5 ms: normal for
10 s (drift at most 8.5e-14), blown up within the first 100 s of 1000 (drift
2.4e56, 193186 projection failures in 200000 steps).

## 2. The derivative

`bias_force_velocity_derivative` against central differences of `rnea(q, v,
0)` with a unit step (exact for the quadratic terms but for rounding), on
the invariant tests' 66 models (every pair of joint types as root and child
followed by a revolute, and two trees): largest difference 5.6e-14 of a
derivative of norm 64 (`test_kernel.exe "[invariants]" -s`).

## 3. Six treatments, measured

A temporary hidden probe in `tests/kernel/test_kernel_drift.cpp` and
temporary switches in the steps (environment variables), all removed after
use. Semi-implicit Euler (the linearly implicit step is the same on these
models: no force elements).

**The four-bar with its pendulum** (no damping, energy 17.83 J), 100 s:
energy change at 100 s (the range over the last 10 s; the largest sampled
over the whole run in brackets where it rose).

| treatment | 5 ms | 2 ms | 1 ms |
|---|---|---|---|
| explicit (as before) | -11.9 to -11.4 | -7.1 to -6.6 | -4.0 to -3.3 |
| D, J delta = 0 | blew up before 10 s | blew up before 50 s (+400 J at 40 s) | blew up before 100 s (+34 J at 50 s) |
| skew part of D, J delta = 0 | -15.4 | -15.2 to -15.1 | -14.8 to -14.6 |
| Coriolis matrix B, J delta = 0 | +3900 to +6800 (growing) | +17 to +71 | +3.6 to +5.2 |
| D, J v+ = nu (kept) | -14.7 to -14.5 (largest -0.0015) | -12.4 to -12.3 (largest -0.00024) | -11.4 to -11.3 (largest -0.00006) |
| B, J v+ = nu | -14.0 to -13.8 | -13.9 to -13.7 | -12.5 to -11.8 |

**The heavy top** (energy 3.861 J, spin 30 rad/s), 300 s at 5 ms: energy
change and |v| at 300 s.

| treatment | energy | |v| |
|---|---|---|
| explicit | blew up before 50 s | |
| D, J delta = 0 | -1.842 (constant from 50 s) | 30.00 |
| skew part of D | -3.681 | 18.67 |
| B, J delta = 0 | -0.816 to -0.710 | 17.98 (falling: 24.5 at 100 s) |
| D, J v+ = nu (kept) | -1.842 | 30.00 |
| B, J v+ = nu | -0.816 to -0.710 | 17.98 |

The top released without precession nutates; the damped methods take the
nutation's energy (1.84 J) and leave a steady precession.

**Free bodies, no gravity** (1 kg boxes, w = (1, 0, 30) rad/s; flat: A =
0.0034667, C = 0.0066667; long: A = 0.0034667, C = 0.00026667 kg m^2), 5 s
at 5 ms: |(w1, w2)| at 1 s and w3 at 5 s. Backward Euler on Euler's
equations gives 0.149716 at 1 s for both (W = (C - A) / A w3, +-27.7 rad/s).

| treatment | flat | long |
|---|---|---|
| explicit | 6.68 at 1 s (growing) | not run |
| D, J delta = 0 | 0.149716, w3 30 | not run |
| skew part of D | 0.149837, w3 29.9915 | not run |
| B | 0.00312, w3 30.0059 | 5.23 at 1 s, 6.18 at 5 s; w3 16.2 at 1 s, 0.0021 at 5 s (tumbled) |
| D (kept) | 0.149716 (to 1e-10 of the prediction), w3 30 | the same |

The analysis behind them (journal): in true coordinates b = B v with B the
Christoffel matrix, D = 2 B and D + D^T = 2 dM/dt; an unconstrained,
unforced step with the positions moved by v+ changes the kinetic energy by
-|v+ - v|^2_M / 2 with B frozen, by h v+^T B (v+ - v) - |v+ - v|^2_M / 2
taken explicitly (a first-order term of either sign), and by h v+^T b(v+ -
v) - |v+ - v|^2_M / 2 through D, whose first term is of second order in the
velocities' change: dissipative while h |v| is small against the inertia's
rate of change. For the free
joint's body-frame velocities D / 2 is no such matrix, and B = (I w)x^
damps a nutation by sqrt((1 + (h w3)^2) / (1 + (h C w3 / A)^2)) per step,
which grows it for a long body (C < A): the tumbling. On a closed loop the
step's new velocities kept the explicit part of the constraints' velocity
terms while the inertia's were implicit, which no choice of B or D balanced
until the new velocities were put on the constraints.

## 4. The tests

`scripts\build.ps1 -Test`: 464 tests, no compiler warning. Then by name with
-s:

- **Nutation:** flat and long, both steps: |(w1, w2)| 0.149716 against
  0.149716 predicted, w3 30, the centre at rest, the energy falling at
  every step.
- **The closed loop:** the four-bar for 20 s, both steps: energy at most
  0.0015 J (5 ms) and 0.00024 J (2 ms) below the start, never above.

## 5. The cost on the sedan

The benchmark, pinned, new and old (`08099dd`, whose linearly implicit
step is the one before D38) alternately: median of one 1 ms step; then the
median and 99th percentile of single steps over the drive.

| version | linearly implicit Euler | RK4 | generalised-alpha |
|---|---|---|---|
| first version (D by a full backward pass, LU solve column by column) | 194.4, 193.8; 185.4, 187.1 | 144.9, 142.5 | 100.4 |
| old (two runs) | 129.1, 127.3; 130.4, 130.9 | 148.4, 144.9 | 98.5, 96.0 |
| kept (two runs) | 165.6, 166.8; 159.1, 163.5; 275.0, 272.1 | 145.3, 146.5 | 96.3, 95.4 |
| old (two runs) | 131.4, 129.9; 131.2, 130.6; 197.2, 183.6 | 147.2, 145.1 | 96.8, 98.3 |

The derivative skips the joint model where S is constant and walks each
joint's subtree and path only; the LU solve applies the row swaps and two
triangular solves to all of W^-1 J^T's columns at once and forms J W^-1
J^T in one product.

## 6. The 1000-second test

`test_kernel.exe "[long]" -s`, all 68 checks passed, the heavy top back in
for both steps. Their lines (the two steps alike: no force elements):

| model | drift, positions (first tenth, last tenth) | drift, velocities | residuals | evaluations; steps |
|---|---|---|---|---|
| four-bar | 6.19e-4 (6.19e-4, 9.13e-6) | 0.246 | 1.0e-10, 2.3e-15 | 200000; 200000 |
| heavy top | 1.69e-15 (1.67e-15, 1.69e-15) | 1.95e-15 | 1.7e-15, 2.4e-16 | 200000; 200000 |

The four-bar's velocity drift per step before the projection grew from
0.024 (task 4.7, section 5) to 0.25: the new velocities are put on the
constraints with J at the step's start, so at its end they are off by about
h |v|^2 times the loop's curvature, where the old choice (the explicit
step's J v + h gamma) was off by a term of order h^2. The projection takes
it off at every step, part of the steps' dissipation.
