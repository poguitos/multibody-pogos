# Analytic cases and manufactured trajectories: measurements (task 5.1, D43)

Machine: a cloud Linux machine (Intel Xeon at 2.10 GHz, 4 cores; Ubuntu
24.04, GCC 13.3.0), RelWithDebInfo, `MBD_PCH=OFF`, built as `CLAUDE.md`
gives for Linux. None of the numbers below is a timing; all are errors, which
do not depend on the machine beyond rounding. Commit: the one that closes
task 5.1 (see the journal index).

## 1. Manufactured trajectories: the error tables

Command:

```bash
build-linux/tests/test_kernel "[manufactured][table]" 2>/dev/null
```

For every model and method, the error |q (-) q_ref| and |v - v_ref| together
(Euclidean norm), over 0.5 s from t = 0 ("0.5 s") and over one step from the
exact state at t = 0.2 s ("step"), at five steps each half the last; in
brackets log2 of the ratio to the previous error, the measured order.
Generalised-alpha has no "step" row (its first step carries an O(h^2)
velocity error by construction; see D43). "failed": Radau IIA's Newton did
not converge at that step (MBD-K058), a step of 0.1 s.

The steps compared by the tests (the longer of each pair; the order is read
in the column of the shorter one): over 0.5 s, RK4 10 ms, the two Euler
steps 1 ms, Dormand-Prince 12.5 ms, generalised-alpha 5 ms, SDIRK4 10 ms,
Radau IIA 25 ms; one step, 10 ms for all, Radau IIA 20 ms.

```
tree: free root; revolute, prismatic, fixed, universal
  RK4                      0.5 s   h 0.040000  1.688e-04          h 0.020000  9.787e-06 ( 4.11)  h 0.010000  6.666e-07 ( 3.88)  h 0.005000  4.342e-08 ( 3.94)  h 0.002500  2.770e-09 ( 3.97)
  RK4                      step    h 0.040000  1.058e-05          h 0.020000  3.205e-07 ( 5.04)  h 0.010000  9.853e-09 ( 5.02)  h 0.005000  3.053e-10 ( 5.01)  h 0.002500  9.500e-12 ( 5.01)
  semi-implicit Euler      0.5 s   h 0.004000  3.298e-02          h 0.002000  1.650e-02 ( 1.00)  h 0.001000  8.251e-03 ( 1.00)  h 0.000500  4.126e-03 ( 1.00)  h 0.000250  2.063e-03 ( 1.00)
  semi-implicit Euler      step    h 0.040000  5.925e-02          h 0.020000  1.490e-02 ( 1.99)  h 0.010000  3.729e-03 ( 2.00)  h 0.005000  9.321e-04 ( 2.00)  h 0.002500  2.330e-04 ( 2.00)
  Dormand-Prince           0.5 s   h 0.050000  1.015e-05          h 0.025000  2.974e-07 ( 5.09)  h 0.012500  9.193e-09 ( 5.02)  h 0.006250  2.876e-10 ( 5.00)  h 0.003125  8.938e-12 ( 5.01)
  Dormand-Prince           step    h 0.040000  4.854e-08          h 0.020000  7.241e-10 ( 6.07)  h 0.010000  1.111e-11 ( 6.03)  h 0.005000  1.722e-13 ( 6.01)  h 0.002500  2.585e-15 ( 6.06)
  generalised-alpha        0.5 s   h 0.020000  2.602e-03          h 0.010000  6.462e-04 ( 2.01)  h 0.005000  1.613e-04 ( 2.00)  h 0.002500  4.032e-05 ( 2.00)  h 0.001250  1.008e-05 ( 2.00)
  SDIRK4                   0.5 s   h 0.040000  7.738e-05          h 0.020000  4.018e-06 ( 4.27)  h 0.010000  2.535e-07 ( 3.99)  h 0.005000  1.591e-08 ( 3.99)  h 0.002500  9.986e-10 ( 3.99)
  SDIRK4                   step    h 0.040000  1.045e-06          h 0.020000  3.529e-08 ( 4.89)  h 0.010000  1.135e-09 ( 4.96)  h 0.005000  3.591e-11 ( 4.98)  h 0.002500  1.127e-12 ( 4.99)
  Radau IIA                0.5 s   h 0.100000  3.375e-05          h 0.050000  1.090e-06 ( 4.95)  h 0.025000  3.464e-08 ( 4.98)  h 0.012500  1.091e-09 ( 4.99)  h 0.006250  3.424e-11 ( 4.99)
  Radau IIA                step    h 0.080000  3.525e-07          h 0.040000  5.046e-09 ( 6.13)  h 0.020000  7.691e-11 ( 6.04)  h 0.010000  1.189e-12 ( 6.02)  h 0.005000  1.833e-14 ( 6.02)
  linearly implicit Euler  0.5 s   h 0.004000  3.298e-02          h 0.002000  1.650e-02 ( 1.00)  h 0.001000  8.251e-03 ( 1.00)  h 0.000500  4.126e-03 ( 1.00)  h 0.000250  2.063e-03 ( 1.00)
  linearly implicit Euler  step    h 0.040000  5.925e-02          h 0.020000  1.490e-02 ( 1.99)  h 0.010000  3.729e-03 ( 2.00)  h 0.005000  9.321e-04 ( 2.00)  h 0.002500  2.330e-04 ( 2.00)
tree: spherical root; cylindrical, planar, free, spherical
  RK4                      0.5 s   h 0.040000  2.625e-02          h 0.020000  1.539e-03 ( 4.09)  h 0.010000  1.032e-04 ( 3.90)  h 0.005000  6.676e-06 ( 3.95)  h 0.002500  4.244e-07 ( 3.98)
  RK4                      step    h 0.040000  3.692e-04          h 0.020000  1.081e-05 ( 5.09)  h 0.010000  3.263e-07 ( 5.05)  h 0.005000  1.002e-08 ( 5.03)  h 0.002500  3.101e-10 ( 5.01)
  semi-implicit Euler      0.5 s   h 0.004000  7.729e+00          h 0.002000  4.194e+00 ( 0.88)  h 0.001000  2.187e+00 ( 0.94)  h 0.000500  1.117e+00 ( 0.97)  h 0.000250  5.646e-01 ( 0.98)
  semi-implicit Euler      step    h 0.040000  1.214e-01          h 0.020000  3.119e-02 ( 1.96)  h 0.010000  7.848e-03 ( 1.99)  h 0.005000  1.965e-03 ( 2.00)  h 0.002500  4.912e-04 ( 2.00)
  Dormand-Prince           0.5 s   h 0.050000  1.642e-03          h 0.025000  2.228e-05 ( 6.20)  h 0.012500  1.655e-07 ( 7.07)  h 0.006250  7.878e-09 ( 4.39)  h 0.003125  3.977e-10 ( 4.31)
  Dormand-Prince           step    h 0.040000  5.592e-06          h 0.020000  6.157e-08 ( 6.50)  h 0.010000  7.794e-10 ( 6.30)  h 0.005000  1.084e-11 ( 6.17)  h 0.002500  1.589e-13 ( 6.09)
  generalised-alpha        0.5 s   h 0.020000  1.994e-01          h 0.010000  4.789e-02 ( 2.06)  h 0.005000  1.186e-02 ( 2.01)  h 0.002500  2.959e-03 ( 2.00)  h 0.001250  7.394e-04 ( 2.00)
  SDIRK4                   0.5 s   h 0.040000  6.660e-03          h 0.020000  3.113e-04 ( 4.42)  h 0.010000  1.895e-05 ( 4.04)  h 0.005000  1.172e-06 ( 4.01)  h 0.002500  7.303e-08 ( 4.00)
  SDIRK4                   step    h 0.040000  3.721e-05          h 0.020000  1.034e-06 ( 5.17)  h 0.010000  3.122e-08 ( 5.05)  h 0.005000  9.639e-10 ( 5.02)  h 0.002500  2.997e-11 ( 5.01)
  Radau IIA                0.5 s   h 0.100000  failed            h 0.050000  5.588e-05          h 0.025000  1.628e-06 ( 5.10)  h 0.012500  5.125e-08 ( 4.99)  h 0.006250  1.622e-09 ( 4.98)
  Radau IIA                step    h 0.080000  1.006e-05          h 0.040000  1.677e-07 ( 5.91)  h 0.020000  2.801e-09 ( 5.90)  h 0.010000  4.560e-11 ( 5.94)  h 0.005000  7.283e-13 ( 5.97)
  linearly implicit Euler  0.5 s   h 0.004000  7.729e+00          h 0.002000  4.194e+00 ( 0.88)  h 0.001000  2.187e+00 ( 0.94)  h 0.000500  1.117e+00 ( 0.97)  h 0.000250  5.646e-01 ( 0.98)
  linearly implicit Euler  step    h 0.040000  1.214e-01          h 0.020000  3.119e-02 ( 1.96)  h 0.010000  7.848e-03 ( 1.99)  h 0.005000  1.965e-03 ( 2.00)  h 0.002500  4.912e-04 ( 2.00)
twin chains tied at their tips (universal - spherical - prismatic)
  RK4                      0.5 s   h 0.040000  3.114e-04          h 0.020000  1.882e-05 ( 4.05)  h 0.010000  1.207e-06 ( 3.96)  h 0.005000  7.628e-08 ( 3.98)  h 0.002500  4.752e-09 ( 4.00)
  RK4                      step    h 0.040000  8.834e-06          h 0.020000  2.890e-07 ( 4.93)  h 0.010000  9.236e-09 ( 4.97)  h 0.005000  2.918e-10 ( 4.98)  h 0.002500  9.167e-12 ( 4.99)
  semi-implicit Euler      0.5 s   h 0.004000  1.104e-01          h 0.002000  5.575e-02 ( 0.99)  h 0.001000  2.802e-02 ( 0.99)  h 0.000500  1.404e-02 ( 1.00)  h 0.000250  7.031e-03 ( 1.00)
  semi-implicit Euler      step    h 0.040000  2.184e-02          h 0.020000  5.497e-03 ( 1.99)  h 0.010000  1.379e-03 ( 1.99)  h 0.005000  3.455e-04 ( 2.00)  h 0.002500  8.644e-05 ( 2.00)
  Dormand-Prince           0.5 s   h 0.050000  1.152e-05          h 0.025000  1.778e-07 ( 6.02)  h 0.012500  3.117e-09 ( 5.83)  h 0.006250  1.031e-10 ( 4.92)  h 0.003125  3.834e-12 ( 4.75)
  Dormand-Prince           step    h 0.040000  1.820e-07          h 0.020000  2.218e-09 ( 6.36)  h 0.010000  3.140e-11 ( 6.14)  h 0.005000  4.744e-13 ( 6.05)  h 0.002500  7.428e-15 ( 6.00)
  generalised-alpha        0.5 s   h 0.020000  1.075e-02          h 0.010000  2.688e-03 ( 2.00)  h 0.005000  6.720e-04 ( 2.00)  h 0.002500  1.680e-04 ( 2.00)  h 0.001250  4.201e-05 ( 2.00)
  SDIRK4                   0.5 s   h 0.040000  5.710e-05          h 0.020000  3.389e-06 ( 4.07)  h 0.010000  2.108e-07 ( 4.01)  h 0.005000  1.299e-08 ( 4.02)  h 0.002500  8.113e-10 ( 4.00)
  SDIRK4                   step    h 0.040000  5.971e-07          h 0.020000  1.888e-08 ( 4.98)  h 0.010000  5.894e-10 ( 5.00)  h 0.005000  1.841e-11 ( 5.00)  h 0.002500  5.739e-13 ( 5.00)
  Radau IIA                0.5 s   h 0.100000  1.064e-04          h 0.050000  3.190e-06 ( 5.06)  h 0.025000  9.753e-08 ( 5.03)  h 0.012500  3.014e-09 ( 5.02)  h 0.006250  9.364e-11 ( 5.01)
  Radau IIA                step    h 0.080000  3.437e-06          h 0.040000  4.438e-08 ( 6.28)  h 0.020000  6.259e-10 ( 6.15)  h 0.010000  9.286e-12 ( 6.07)  h 0.005000  1.416e-13 ( 6.04)
  linearly implicit Euler  0.5 s   h 0.004000  1.104e-01          h 0.002000  5.575e-02 ( 0.99)  h 0.001000  2.802e-02 ( 0.99)  h 0.000500  1.404e-02 ( 1.00)  h 0.000250  7.031e-03 ( 1.00)
  linearly implicit Euler  step    h 0.040000  2.184e-02          h 0.020000  5.497e-03 ( 1.99)  h 0.010000  1.379e-03 ( 1.99)  h 0.005000  3.455e-04 ( 2.00)  h 0.002500  8.644e-05 ( 2.00)
parallelogram four-bar, closed by a revolute
  RK4                      0.5 s   h 0.040000  2.780e-05          h 0.020000  1.668e-06 ( 4.06)  h 0.010000  1.069e-07 ( 3.96)  h 0.005000  6.765e-09 ( 3.98)  h 0.002500  4.255e-10 ( 3.99)
  RK4                      step    h 0.040000  9.320e-07          h 0.020000  2.926e-08 ( 4.99)  h 0.010000  9.162e-10 ( 5.00)  h 0.005000  2.866e-11 ( 5.00)  h 0.002500  8.964e-13 ( 5.00)
  semi-implicit Euler      0.5 s   h 0.004000  5.660e-03          h 0.002000  2.862e-03 ( 0.98)  h 0.001000  1.439e-03 ( 0.99)  h 0.000500  7.215e-04 ( 1.00)  h 0.000250  3.613e-04 ( 1.00)
  semi-implicit Euler      step    h 0.040000  3.349e-03          h 0.020000  8.467e-04 ( 1.98)  h 0.010000  2.128e-04 ( 1.99)  h 0.005000  5.335e-05 ( 2.00)  h 0.002500  1.336e-05 ( 2.00)
  Dormand-Prince           0.5 s   h 0.050000  3.041e-07          h 0.025000  1.128e-08 ( 4.75)  h 0.012500  3.824e-10 ( 4.88)  h 0.006250  1.239e-11 ( 4.95)  h 0.003125  3.567e-13 ( 5.12)
  Dormand-Prince           step    h 0.040000  1.821e-09          h 0.020000  3.249e-11 ( 5.81)  h 0.010000  5.401e-13 ( 5.91)  h 0.005000  8.435e-15 ( 6.00)  h 0.002500  3.846e-16 ( 4.45)
  generalised-alpha        0.5 s   h 0.020000  3.844e-04          h 0.010000  9.608e-05 ( 2.00)  h 0.005000  2.403e-05 ( 2.00)  h 0.002500  6.008e-06 ( 2.00)  h 0.001250  1.502e-06 ( 2.00)
  SDIRK4                   0.5 s   h 0.040000  3.665e-06          h 0.020000  2.033e-07 ( 4.17)  h 0.010000  1.255e-08 ( 4.02)  h 0.005000  7.799e-10 ( 4.01)  h 0.002500  4.887e-11 ( 4.00)
  SDIRK4                   step    h 0.040000  1.050e-07          h 0.020000  3.160e-09 ( 5.06)  h 0.010000  9.712e-11 ( 5.02)  h 0.005000  3.014e-12 ( 5.01)  h 0.002500  9.590e-14 ( 4.97)
  Radau IIA                0.5 s   h 0.100000  1.331e-06          h 0.050000  4.020e-08 ( 5.05)  h 0.025000  1.236e-09 ( 5.02)  h 0.012500  3.832e-11 ( 5.01)  h 0.006250  1.185e-12 ( 5.01)
  Radau IIA                step    h 0.080000  1.826e-08          h 0.040000  2.304e-10 ( 6.31)  h 0.020000  3.210e-12 ( 6.17)  h 0.010000  4.736e-14 ( 6.08)  h 0.005000  8.600e-16 ( 5.78)
  linearly implicit Euler  0.5 s   h 0.004000  5.660e-03          h 0.002000  2.862e-03 ( 0.98)  h 0.001000  1.439e-03 ( 0.99)  h 0.000500  7.215e-04 ( 1.00)  h 0.000250  3.613e-04 ( 1.00)
  linearly implicit Euler  step    h 0.040000  3.349e-03          h 0.020000  8.467e-04 ( 1.98)  h 0.010000  2.128e-04 ( 1.99)  h 0.005000  5.335e-05 ( 2.00)  h 0.002500  1.336e-05 ( 2.00)
assertions: - none -
```

Read from it:

- Every one-step order at the steps used is within 0.17 of p + 1 (the
  largest: Dormand-Prince on the spherical-root tree, 6.17).
- Every order over 0.5 s at the steps used is within 0.08 of p, but one:
  Dormand-Prince on the spherical-root tree, 7.07, 4.39, 4.31 from 25 ms to
  3.125 ms, and 4.70 at 1.56 ms (from a separate run of the same function at
  shorter steps, 12.5 ms to 0.78 ms: 4.39, 4.31, 4.70, 0.76), where the
  rounding floor of that model, 2e-12 to 1e-11 (Radau IIA there flattens at
  2.4e-12, Dormand-Prince at 9.1e-12), is reached.
- The semi-implicit and linearly implicit Euler steps give the same errors to
  the four digits printed on every model: with forces that do not depend on
  the state the linearly implicit step's force derivatives are zero, and
  both take the inertia's velocity products the same way (D38).
- At 1 s, an earlier run (not in the table) gave errors of the size of the
  motion for the first-order methods on the spherical-root tree (49 to 74 at
  every step from 8 to 0.5 ms) and irregular orders for Dormand-Prince and
  SDIRK4 on the twin chains; the span was cut to 0.5 s.

## 2. The rigid body on three bushings

Command:

```bash
build-linux/tests/test_kernel "Linearisation: a rigid body on three bushings has the modes of eig(K\, M)" -s
```

The 68.5 kg engine of the test, no gravity. Frequencies from `linearize()`
(equal to the closed form to the digits shown), and the three largest
energy shares of each mode. Coordinates of the free joint: 0 to 2 the body's
angular velocity, 3 to 5 its origin's velocity, in body axes.

| mode | frequency [Hz] | largest energy shares |
|---|---|---|
| 1 | 10.9678 | [4] 60.5 %, [0] 22.5 %, [3] 12.5 % |
| 2 | 13.6201 | [3] 77.6 %, [4] 13.7 %, [1] 8.2 % |
| 3 | 18.0024 | [5] 95.2 %, [4] 2.8 %, [0] 1.2 % |
| 4 | 23.7525 | [2] 35.6 %, [0] 28.6 %, [4] 18.0 % |
| 5 | 27.7797 | [2] 51.3 %, [0] 44.0 %, [4] 4.0 % |
| 6 | 33.3656 | [1] 76.7 %, [2] 9.3 %, [3] 8.8 % |

Largest errors against eig(K, M) in closed form: frequency 2.03e-11
relative, shape 2.78e-10 (the part of the unit shape not along the closed
form's), energy shares 1.59e-10; smallest relative gap between the squared
frequencies 0.368.

## 3. The other cases: margins seen

| test | quantity | seen | allowed |
|---|---|---|---|
| line of masses (N = 4) | frequency, relative | 4.4e-12 | 1e-8 |
| line of masses | shape | 3.3e-11 | 1e-7 |
| line of masses | energy shares | 3.5e-11 | 1e-7 |
| torque-free body, unequal moments (k^2 = 0.0712, period 3.079 s) | w against Jacobi's functions, over 5 s | 1.7e-12 rad/s | 1e-9 |
| same | world angular momentum | 1.2e-13 N m s of 0.393 | 3.9e-10 |
| same | w after one period | 1.1e-12 rad/s | 1e-9 |
| heavy top, steady precession (Omega = 1.50718 rad/s) | precession angle after 1.5 s | 2.0e-12 rad | 1e-8 |
| same | tilt | 5.3e-14 rad | 1e-8 |
| heavy top, released with spin | lowest cos(tilt) against u1 = 0.864251, three nods | 2.6e-14 | 1e-7 |
| manufactured, pointwise (4 models, 3 instants) | inverse dynamics residual, relative | 4.9e-16 | 1e-12 |
| same | accelerations from the forward dynamics | 2.1e-13 (of accelerations of 1.2 to 10.7) | 1e-9 (1 + \|a\|) |
| same, loops | constraint forces J^T lambda | 9.0e-15 (of 9.1 to 13.6) | 1e-9 relative |
| same, loops | constraint forces over applied forces | 0.27 to 2.6 | above 0.1 |

Commands: each test by name with `-s`, as in section 2.
