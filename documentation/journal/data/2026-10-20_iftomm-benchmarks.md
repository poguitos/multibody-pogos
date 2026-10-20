# The IFToMM benchmarks: measurements (task 5.2, D44)

Machine: a cloud Linux machine (Intel Xeon at 2.10 GHz, 4 cores; Ubuntu
24.04, GCC 13.3.0), RelWithDebInfo, `MBD_PCH=OFF`. No timings are compared
here; where a duration is given it is wall-clock on this machine, for
orientation only.

Sections 1 to 5 come from scratch test cases, written into
`tests/kernel/test_kernel_iftomm.cpp` while the problems were explored and
removed before the commit (their output is copied here as printed); section 2
also needed a temporary edit of `RankRevealingLDLT::kRankTolerance`, undone
afterwards. Section 6 is the committed tests' own output:

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
Radau IIA at 1e-8 over 30 s: spindle rate 6.283 -> 5.10 (t = 20 s) -> 5.42
(t = 30 s), collar from 0.500 to 0.743 m and back to 0.648 m, the rods from
2.094 to 1.831 rad and back: a slow oscillation, period near 40 s, barely
damped; angular momentum change 1.6e-8 N m s of 53.9.

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
