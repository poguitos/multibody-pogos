# The IFToMM benchmarks: measurements (task 5.2, D44)

Machine: a cloud Linux machine (Intel Xeon at 2.10 GHz, 4 cores; Ubuntu
24.04, GCC 13.3.0), RelWithDebInfo, `MBD_PCH=OFF`. No timings are compared
here; where a duration is given it is wall-clock on this machine, for
orientation only.

Sections 1 to 5 come from scratch test cases, written into
`tests/kernel/test_kernel_iftomm.cpp` while the problems were explored and
removed before the commit (their output is copied here as printed); section 2
also needed a temporary edit of `RankRevealingLDLT::kRankTolerance`, undone
afterwards. Section 6 is the committed tests' own output. Section 7 is mixed:
the committed library cases print the size of each largest difference (the
table's last row, the governor's two), with the commands given there; the
rest comes from three more scratch cases of the same kind, also removed and
described there, except González's energy column, read with awk. The
commands for section 6:

```bash
build-linux/tests/test_kernel "[iftomm]~[long]" -s     # the five problems
build-linux/tests/test_kernel "IFToMM A02 with 40 four-bars" -s
```

## 1. The N-four-bar with the defaults: where it fails

N = 1, Dormand-Prince at 1e-10, `constraint_tolerance` 1e-10 (the default),
theta (the vertical bars' angle) every 10 ms against the closed form,
printed near each singular position (theta a multiple of pi) and every 0.1 s:

```
t 0.68 theta  0.136190 err -2.85e-11 coupler 2.05e-15 drift 4.65e-16
t 0.71 theta -0.007693 err -3.05e-11 coupler -1.45e-12 drift 1.11e-14   (first singular pass)
t 0.72 theta -0.058006 err -3.05e-11 coupler -3.38e-12 drift 1.96e-13
t 1.21 theta -3.115362 err -1.32e-11 coupler 7.44e-12 drift 1.95e-13
t 1.22 theta -3.164934 err  1.43e-07 coupler 2.50e-11 drift 5.84e-13   (second: an error enters)
t 1.50 theta -4.125129 err  9.34e-06
t 2.00 theta -4.789495 err  5.64e-05
t 3.00 theta -8.627246 err  6.49e-04
t 4.00 theta -11.148389 err 6.82e-03
t 4.56 theta -12.546571 err 4.03e-02
failed at 4.56402: MBD-K055: Dormand-Prince needs a step of [shorter than min_step] at t = 4.564016 s
```

("coupler" is the coupler's absolute angle, zero on the parallelogram;
"drift" the position residual each step leaves before the projection, never
projected away below the tolerance of 1e-10.) The same with N = 4: failed at
t = 4.631 s.

Inside the failing step at the second pass with the rank cut at 1e-13
(section 2), the coupler's angular velocity w2 (zero on the parallelogram)
as theta + pi approached zero, evaluation by evaluation:

```
theta+pi -2.698e-04  w2 0.000013
theta+pi -1.607e-04  w2 0.000036
theta+pi -1.031e-04  w2 0.000092
theta+pi -7.477e-05  w2 0.000174
```

An imperfect parallelogram, its loop open by delta, needs the coupler tilted
by about delta / (L theta), so w2 ~ delta theta' / (L theta^2): with delta =
2e-13 m (the drift above), theta' = 5 rad/s and theta = 7.5e-5 rad, 1.8e-4
rad/s.

## 2. The forward dynamics at the singular position

The accelerations of the constrained dynamics on the exact motion (theta at
the offset given from 0 or from -pi, its exact rate), theta'' against the
closed form -C cos(theta) / J, and the rank the solver used. Rank cut 1e-10
(as built):

```
N 1 theta +1.0e-05: a0 err -1.776e-15, rank 2 of 3
N 1 theta +3.0e-06: a0 err -1.473e-04, rank 1 of 3
N 1 theta +1.0e-06: a0 err -4.909e-05, rank 1 of 3
N 1 theta +1.0e-08: a0 err -4.909e-07, rank 1 of 3
N 1 theta  0      : a0 err -1.599e-14, rank 1 of 3
N 4 theta +1.0e-04: a0 err  3.552e-11, rank 8 of 12
N 4 theta +1.0e-05: a0 err -7.478e-01, rank 4 of 12
N 4 theta +1.0e-08: a0 err -7.457e-01, rank 4 of 12
N 4 theta  0      : a0 err -7.457e-01, rank 4 of 12
N 4 theta -1.0e-04: a0 err -2.842e-14, rank 8 of 12
```

Rank cut 1e-13 (temporary edit):

```
N 4 theta +1.0e-05: a0 err -3.030e-10, rank 8 of 12
N 4 theta +3.0e-06: a0 err  3.065e-09, rank 8 of 12
N 4 theta +1.0e-06: a0 err -5.225e-09, rank 8 of 12
N 4 theta +1.0e-07: a0 err -7.458e-01, rank 4 of 12
N 4 theta  0      : a0 err -7.457e-01, rank 4 of 12
```

The whole of `test_kernel` passed with the cut at 1e-13 (141 cases).

## 3. The N-four-bar's largest error over 10 s, by tolerances

Dormand-Prince at tolerance `tol`, the constraints held to `ctol`; largest
|theta - theta_exact| every 10 ms, or where it failed.

| rank cut | N | ctol | tol 1e-8 | tol 1e-10 |
|---|---|---|---|---|
| 1e-10 | 1 | 1e-10 | 17.8 | failed at 4.56 s |
| 1e-10 | 1 | 1e-12 | 1.6e-3 | 4.7e-3 |
| 1e-10 | 1 | 1e-13 | 2.5e-8 | 1.1e-5 |
| 1e-10 | 1 | 1e-14 | 2.4e-8 | 2.5e-8 |
| 1e-10 | 4 | 1e-10 | 13.9 | failed at 4.63 s |
| 1e-10 | 4 | 1e-12 | 1.8e-8 | 3.5e-3 |
| 1e-10 | 4 | 1e-13 | 2.1e-8 | 2.5e-6 |
| 1e-10 | 4 | 1e-14 | 2.2e-8 | 2.1e-8 |
| 1e-13 | 1 | 1e-10 | 31.9 | failed at 1.22 s |
| 1e-13 | 1 | 1e-13 | 2.5e-8 | failed at 2.63 s |
| 1e-13 | 1 | 1e-14 | 2.4e-8 | 2.5e-8 |
| 1e-13 | 4 | 1e-10 | 23.8 | failed at 1.24 s |
| 1e-13 | 4 | 1e-13 | 2.1e-8 | failed at 2.67 s |
| 1e-13 | 4 | 1e-14 | 2.2e-8 | 5.0e-4 |

With the cut at 1e-10 and ctol 1e-14, N = 4, the error printed at each
singular pass and every 0.5 s grew smoothly, largest in the slow passes over
the top (t = 2.0 to 2.7 s, 4.0 to 5.2 s, ...), with no step at any singular
pass: at tol 1e-10 from 2.8e-11 (t = 0.72) to 1.7e-8 (t = 9.0); at tol 1e-12
from 1.6e-12 to 2.5e-9.

N = 40, ctol 1e-14, tol 1e-10: largest error 1.8e-7 rad, 11001
evaluations, projections failing (MBD-K051 first at t = 0.06 s; residuals
after projection up to 8.8e-13; the number of failures was not printed),
52 s. With ctol 1e-12 (the committed hidden case): 3.4e-7 rad, 17 s.

## 4. Andrews' mechanism

First run: every coordinate off by 68 to 118 rad at 0.03 s. The test set's
start gave the accelerations to ten digits (14222.44392, -10666.83294, the
rest below 1e-10) but |phi| = (-0.0308, 0.0629) in every loop: point E on
body 3 was placed at (0, +ss) instead of (0, -ss). After the fix, the
largest error at 0.03 s against the reference:

| method | ctol | tol 1e-8 | tol 1e-10 | tol 1e-12 |
|---|---|---|---|---|
| Dormand-Prince | 1e-10 | 3.07e-8 | 1.21e-9 | 1.43e-9 |
| Dormand-Prince | 1e-14 | 4.36e-8 | 1.07e-9 | 1.43e-9 |
| Radau IIA | 1e-10 | 1.30e-9 | 1.43e-9 | 1.43e-9 |
| Radau IIA | 1e-14 | 1.56e-9 | 1.43e-9 | 1.43e-9 |

Radau IIA at 1e-12, ctol 1e-14, each coordinate's error at 0.03 s, divided
by its velocity:

```
beta    dq -1.1453e-09  v  1.139920e+03  dq/v -1.0047e-12   dv  1.08e-07 (rel  9.5e-11)
Theta   dq  1.4311e-09  v -1.424379e+03  dq/v -1.0047e-12   dv -1.83e-07 (rel  1.3e-10)
gamma   dq -1.1082e-11  v  1.103291e+01  dq/v -1.0044e-12   dv -3.09e-07 (rel -2.8e-08)
delta   dq -5.7965e-13  v  5.735699e-01  dq/v -1.0106e-12   dv -1.36e-08 (rel -2.4e-08)
Phi     dq -1.9380e-11  v  1.929337e+01  dq/v -1.0045e-12   dv -5.39e-07 (rel -2.8e-08)
epsilon dq -3.2863e-13  v  3.231791e-01  dq/v -1.0169e-12   dv -1.66e-08 (rel -5.1e-08)
Omega   dq  1.9380e-11  v -1.929337e+01  dq/v -1.0045e-12   dv  5.39e-07 (rel -2.8e-08)
```

## 5. Bricard's mechanism and the flyball governor

Bricard, Dormand-Prince at 1e-8 and 1e-10, ctol 1e-14, the joint angles
(joints 1 to 5) every second: joints 1, 3, 5 equal and joints 2, 4 equal to
the six digits printed at every time, e.g. t = 1: (1.035028, -0.480621,
1.035028, -0.480621, 1.035028); t = 5: 6.7e-5 (back near the start); the
same at both tolerances; energy change 1.5e-10 J over 10 s. An independent
closure solve in Python (numpy; Newton on psi for given phi, the chain as
turns about the cube's axes) gave psi = -0.480621028 for phi = 1.035028 and
-0.198554869 for phi = 2.893328; sympy reduced the six closure equations in
u = tan(phi / 2), w = tan(psi / 2) to factors of which one is common to all:
u w^2 + u^2 w + 2 u w + u + w = u w^2 + (u + 1)^2 w + u. Its small root
reproduced the Newton values to 1e-9 at phi = 0.5, 1.035, 2.0 and 2.893 (the
residuals of the Newton solutions: 3e-10 to 1.6e-9).

The flyball governor first built with both rods pointing inwards (a sign of
the hinge axes): the balls at x = -0.816 for the rod on +x. Fixed; then
Radau IIA at 1e-8 over 30 s, printed every 2 s: spindle rate 6.283 -> 5.10
(t = 20 s) -> 5.42 (t = 30 s), collar from 0.500 to 0.743 m and back to 0.648
m (as printed every 2 s), the rods from 2.094 to 1.831 rad and back; angular
momentum change 1.6e-8 N m s of 53.9. Read at the time as a slow oscillation
with a period near 40 s. **That reading was wrong** (found later the same day,
section 7): the motion swings about once a second (period 1.0247 s), and
printed every 2 s it looked slow. The row printed at 10 s (q: spindle angle,
collar, rods; v: their rates) already did not fit a slow motion of 0.12 m:

```
t 10.0 q 55.935410 0.619623 1.960870 1.960870 v 5.548204 -7.476e-01 8.085e-01 8.085e-01
```

The collar moves at -0.748 m/s there, downwards, while the samples every 2 s
rise.

## 6. The committed tests

```
A01 on a revolute joint: largest distance 1.08e-9 and 8.6e-11 m at tolerances 1e-10 and 1e-12;
    energy change 2.0e-9 and 1.6e-10 J
A01 on a distance constraint: 1.40e-8 and 1.52e-10 m; energy change 2.79e-8 and 2.96e-10 J
A02 (N = 4): largest error 2.15e-8 rad at 1e-10, 3.07e-9 at 1e-12
A02 (N = 40, hidden, ctol 1e-12): 3.4e-7 rad
A03: error at 0.03 s (beta, Theta, gamma, delta, Phi, epsilon, Omega)
     -1.145e-9, 1.431e-9, -1.12e-11, -5.8e-13, -1.95e-11, -3.3e-13, 1.95e-11;
     Dormand-Prince at 1e-12 minus Radau IIA at 1e-10: at most 6.4e-13
A04: largest phi 3.14159 rad; off the closed-form path by 9.3e-15 rad; energy change 1.5e-10 J;
     from the reduced model 2.0e-9 rad; validate: 1 degree of freedom, 1 redundant equation
A05: L(0) = 53.869 N m s, largest change 5.9e-9; E(0) = 238.885 J, dampers' work 3.12e-3 J,
     largest imbalance 2.3e-8 J; after 10 s from Radau IIA: Dormand-Prince 1.4e-9,
     generalised-alpha 2.78e-5 at 1 ms and 7.12e-6 at 0.5 ms (ratio 3.90)
```

## 7. A04 and A05 against the IFToMM library's results

Results from the library, downloaded by the author and given on 20 October
2026 (`tests/kernel/data/iftomm/`, its README for whose and what). The
comparison was first made by a scratch case reading them from the upload
folder, then by the committed cases, which print only the largest difference
from each result:

```bash
build-linux/tests/test_kernel "IFToMM A04: Bricard's mechanism against three results in the library" -s
build-linux/tests/test_kernel "IFToMM A05: the flyball governor against two results in the library" -s
```

Everything else here is from scratch cases (removed). The first, which read
the uploads, printed the largest distance of P2 from each result with its
time (the table's last row, the same at 1e-10 and 1e-12), and our P2 at
1e-12 at the five times given below the table. The second, written after a
review of this section, printed the rest but the governor's thresholds,
which a third gave (at the end of this section). The second used the
committed cases'
models and settings, ran each model again at a tighter tolerance (Bricard's
mechanism at 1e-12, the governor at 1e-10), and ran the governor once more
with Radau IIA at 1e-8, sampled every 1 ms to 30 s. It printed the values at
the times shown, the peaks, and the results' distances from each other (rows
matched by time).

Bricard's mechanism, point P2, ours (Dormand-Prince at 1e-12; at 1e-10 within
2.9e-11 m of it) against each result, every 0.5 s (and 0.05 s):

| t [s] | Tagliapietra (109) | González (72) | Pankiewicz (160) |
|---|---|---|---|
| 0.05 | 2.454e-05 | 1.950e-06 | 4.912e-06 |
| 0.50 | 3.862e-06 | 1.925e-05 | 3.862e-04 |
| 1.00 | 3.978e-06 | 2.902e-05 | 8.754e-04 |
| 1.50 | 3.995e-06 | 5.049e-05 | 1.203e-03 |
| 2.00 | 3.802e-06 | 7.061e-05 | 1.518e-03 |
| 2.50 | 8.739e-06 | 2.885e-05 | 2.554e-05 |
| 3.00 | 3.871e-06 | 4.877e-05 | 2.308e-03 |
| 3.50 | 3.984e-06 | 8.020e-05 | 2.951e-03 |
| 4.00 | 3.993e-06 | 1.140e-04 | 3.213e-03 |
| 4.50 | 3.791e-06 | 1.451e-04 | 3.334e-03 |
| 5.00 | 8.737e-06 | 5.471e-05 | 9.895e-05 |
| 5.50 | 3.881e-06 | 9.646e-05 | 4.180e-03 |
| 6.00 | 3.990e-06 | 1.497e-04 | 4.950e-03 |
| 6.50 | 3.992e-06 | 1.949e-04 | 5.302e-03 |
| 7.00 | 3.779e-06 | 2.333e-04 | 5.201e-03 |
| 7.50 | 8.735e-06 | 7.442e-05 | 2.247e-04 |
| 8.00 | 3.891e-06 | 1.635e-04 | 6.138e-03 |
| 8.50 | 3.997e-06 | 2.375e-04 | 7.017e-03 |
| 9.00 | 3.993e-06 | 2.966e-04 | 7.319e-03 |
| 9.50 | 3.766e-06 | 3.435e-04 | 6.977e-03 |
| 10.00 | 3.748e-08 | 1.012e-04 | 3.971e-04 |
| largest | 2.454e-05 (t = 0.05) | 3.494e-04 (t = 9.40) | 7.726e-03 (t = 9.25) |

Ours at t = 1, 2.5, 5, 7.5 and 10 s (tolerance 1e-12): P2 = (0.274481319,
-0.886707939, -0.462330003), (-0.999983185, -1.000000000, -0.000016815),
(0.999932742, -0.999999998, -0.000067255), (-0.999848674, -0.999999989,
-0.000151314), (0.999730987, -0.999999964, -0.000268977).

Pankiewicz's distance from ours, its local maxima at that result's 0.05 s
rows (t, distance, distance / t); small at the turning points (2.5, 5, 7.5 s,
in the table), where P2 stops, as a phase error is:

```
 1.05  8.9558e-04  8.529e-04
 1.90  1.5507e-03  8.162e-04
 3.85  3.1778e-03  8.254e-04
 4.30  3.5588e-03  8.276e-04
 6.00  4.9499e-03  8.250e-04
 6.80  5.6736e-03  8.344e-04
 8.35  7.0515e-03  8.445e-04
 8.85  7.1610e-03  8.092e-04
 9.25  7.7256e-03  8.352e-04
```

The results among themselves, largest distance of P2 at common times (every
0.01 s for the first pair, 0.05 s with Pankiewicz's): Tagliapietra and
González 3.48e-4 m (t = 9.39), Tagliapietra and Pankiewicz 7.73e-3 m
(t = 9.25), González and Pankiewicz 8.06e-3 m (t = 9.25). All three start at
(1, -1, 0). González's own fifth column (an energy change, probably) is not
monotonic: -4.589e-4 J at 4 s, -3.975e-4 J at 5 s, at most -8.249e-4 J at
8.84 s, -7.89e-4 J at 10 s (read with awk).

The flyball governor, s (the sleeve) every 0.1 s against González's two
results; ours from Radau IIA at 1e-8, within 8.2e-10 m of the same at 1e-10:
largest difference 2.03e-6 m from 116 (at 9.5 s) and 6.77e-6 m from 117 (at
10 s). Each whole second, the differences there and the largest of each in
the second before it:

```
t     ours (1e-8)  116-ours   117-ours   117-116 | largest |116-ours| (t)  |117-ours| (t)
 1.0  0.501420922  -9.22e-07  +1.31e-07  +1.05e-06 | 9.22e-07 (1.0)  5.96e-07 (0.8)
 2.0  0.505909294  -2.94e-07  +4.83e-07  +7.77e-07 | 7.71e-07 (1.2)  1.32e-06 (1.8)
 3.0  0.513246079  -7.85e-08  +1.03e-06  +1.11e-06 | 1.16e-06 (2.3)  1.99e-06 (2.8)
 4.0  0.523223371  -3.71e-07  +1.73e-06  +2.10e-06 | 1.22e-06 (3.3)  2.65e-06 (3.9)
 5.0  0.535563104  -1.04e-07  +2.56e-06  +2.66e-06 | 1.20e-06 (4.2)  3.41e-06 (4.9)
 6.0  0.549928666  +3.34e-07  +3.45e-06  +3.12e-06 | 1.13e-06 (5.5)  4.11e-06 (5.9)
 7.0  0.565938158  +8.42e-07  +4.36e-06  +3.52e-06 | 1.30e-06 (6.4)  4.71e-06 (6.9)
 8.0  0.583178337  +6.63e-07  +5.25e-06  +4.59e-06 | 1.40e-06 (7.4)  5.25e-06 (8.0)
 9.0  0.601218402  +5.98e-07  +6.07e-06  +5.47e-06 | 1.84e-06 (8.4)  6.07e-06 (9.0)
10.0  0.619622917  +1.08e-06  +6.77e-06  +5.69e-06 | 2.03e-06 (9.5)  6.77e-06 (10.0)
```

117 drifts from ours at about 7e-7 m per second ((6.77e-6 - 1.3e-7) / 9 s),
and from 116 at about 5e-7 m per second ((5.69e-6 - 1.05e-6) / 9 s), 116
moving from ours in the same direction (-9.2e-7 m at 1 s, +1.08e-6 m at
10 s); 116 stays within 2.03e-6 m, beyond its rounding of 5e-7 m from the
start.

The swing, from the run sampled every 1 ms (each extreme refined by a
parabola through three samples): s between 0.499911 m (t = 1.025 s) and
0.743571 m (t = 0.512 s) in the first 10 s; the largest and smallest of the
0.1 s samples 0.743362 and 0.499918 m. Ten upward crossings of the mean
(0.62620 m), at 0.2555 to 9.4775 s, ten maxima, at 0.512, 1.537, 2.562,
3.586, 4.611, 5.636, 6.660, 7.685, 8.710 and 9.734 s, and nine minima, at
1.0247 to 9.2220 s: each gives a mean period of 1.02467 s, and every single
interval rounds to 1.0247 s. The same run printed every 2 s to 30 s
reproduces the first run of section 5 (t, s [m], ds/dt [m/s]):

```
   0  0.5000  +0.0000
   2  0.5059  -0.2408
   4  0.5232  -0.4544
   6  0.5499  -0.6183
   8  0.5832  -0.7180
  10  0.6196  -0.7476
  12  0.6558  -0.7092
  14  0.6886  -0.6108
  16  0.7153  -0.4638
  18  0.7338  -0.2821
  20  0.7428  -0.0799
  22  0.7416  +0.1279
  24  0.7303  +0.3266
  26  0.7098  +0.5015
  28  0.6814  +0.6383
  30  0.6476  +0.7237
```

2 s is 2 / 1.02467 = 1.9518 swings, so each sample falls 0.0482 of a swing
earlier in it than the one before: the samples trace one swing every
2 / 0.0482 = 41.5 s, the "period near 40 s" of section 5.

The governor's thresholds, from the third scratch case, written after a
second check of this record had found the first estimate wrong. A model
whose period is longer by the fraction e moves as s(t / (1 + e)). Ours was
run with Radau IIA at 1e-8 every 1 ms to 10.01 s, s between samples taken by
cubic Hermite interpolation on (s, ds/dt) (against a separate run sampled
every 0.1 s: within 6.6e-10 m), and the largest |result - s(t / (1 + e))|
over the result's 101 rows found for each e; the limits by bisection:

```
e = 0:        116 2.035e-06 m,           117 6.771e-06 m
e = -7.0e-07: 116 6.894e-06 m (t 9.5)    117 1.200e-05 m (t 10.0)
e = +7.0e-07: 116 4.150e-06 m (t 10.0)   117 1.538e-06 m (t 10.0)
e = -1.0e-06: 116 8.977e-06 m (t 9.5)    117 1.425e-05 m (t 10.0)
e = +1.0e-06: 116 6.393e-06 m (t 10.0)   117 1.155e-06 m (t 9.5)
e = -3.0e-06: 116 2.351e-05 m (t 10.0)   117 2.920e-05 m (t 10.0)
e = +3.0e-06: 116 2.134e-05 m (t 10.0)   117 1.566e-05 m (t 10.0)
116 (bound 5e-6) fails for a period shorter by 4.271e-07 (first at t 9.5) or longer by 8.137e-07 (t 10.0)
117 (bound 2e-5) fails for a period shorter by 1.769e-06 (first at t 10.0) or longer by 3.581e-06 (t 10.0)
t |ds/dt| at 10 s: 7.4760 m, the largest over the 0.1 s samples
```

The limits differ by sign because both results lie above ours at 10 s, where
s falls at 0.748 m/s, and a longer period raises ours there. 117 comes within
1.2e-6 m of ours run 1e-6 slower: it behaves like a period about 1e-6 longer
than ours. The first estimate, 7.5 e m against each bound without the
results' own differences, gave 7e-7 and 3e-6 in either direction; it was
recorded so in the test's comment and the closing commit's message
(`85de229`).
