# The HMMWV benchmark: measurements (task 5.4, D47, D48)

Machine: the author's laptop (13th Gen Intel Core i7-13700H, Windows 11
Home), MSVC 19.50.35717, RelWithDebInfo. Its speed moved during the day:
the full test suite took 29.5 s in the morning and 85.9 s in the evening, and
RK4 at 1 ms on the model 0.44 s and later 0.80 s. Timings compare only within
one run; the tables of section 7 come from one run pinned to logical
processor 2 (a performance core) at high priority, started at 21:35.

Sections 1, 3 and 7 are the committed program's output (`mbd_hmmwv`,
`bench/hmmwv.cpp`), sections 5 and 6 the committed tests'. Sections 2 and 4
come from earlier versions of the program and a temporary probe in the
solver during the session, described where they differ from what was
committed.

## 1. The springs' free lengths

`mbd_hmmwv calibrate`: Newton on the two free lengths (front, rear), the
derivatives by differences of 1e-5 m, each evaluation a static equilibrium
of the model with joints at a tolerance of 1e-11:

```
free lengths 0.290000000 0.340000000: arm angles +2.164e-02 -3.196e-02 rad
free lengths 0.294783325 0.332977959: arm angles -2.232e-04 -1.197e-04 rad
free lengths 0.294734400 0.332951359: arm angles -5.101e-07 -7.147e-07 rad
free lengths 0.294734288 0.332951201: arm angles -2.980e-09 -1.631e-09 rad
free lengths 0.294734288 0.332951200: arm angles -6.815e-12 -9.506e-12 rad
free lengths 0.294734288 0.332951200: arm angles -3.935e-14 -2.237e-14 rad
free lengths 0.294734288 0.332951200: arm angles -6.596e-16 +2.612e-16 rad
free lengths 0.294734288 0.332951200: arm angles +6.540e-16 -2.321e-16 rad
```

With Chrono's free lengths (0.339, 0.382 m) and its arm inertias, the first
exploration printed:

```
design: chassis com height 0.7089 m
statics: converged 1, 8 iterations, residual 2.20e-12 (given 1.19e+03), dof 10 supported 7 neutral 3
settled: chassis com height 0.7935 m, pitch 0.00397 rad, front wheel centre z 0.4479, rear 0.4492
  corner 0: spring 0.2932 m, shock 0.4197 m      (front: spring stop at 0.30 m)
  corner 2: spring 0.3567 m, shock 0.4155 m      (rear: spring stop at 0.35 m)
```

and `validate()` reported MBD-M014 for every arm: principal moments (0.4,
0.4, 0.8938) and (0.03, 0.03, 0.06276) kg m^2 break the triangle inequality.

With the calibrated lengths and flat arms (the committed model):

```
statics: converged 1, 3 iterations, residual 2.69e-11 (given 2.19e+02), dof 10 supported 7 neutral 3
settled: chassis com height 0.6875 m, pitch 0.00041 rad, front wheel centre z 0.4479, rear 0.4492
  corner 0: spring 0.2463 m, shock 0.3728 m
  corner 2: spring 0.3058 m, shock 0.3640 m
RK4 at 1 ms for 5 s: spring ranges 0.2422 to 0.2564 m (front), 0.2997 to 0.3150 m (rear)
```

The springs stay inside Chrono's stops (0.15 to 0.30 m front, 0.15 to 0.35 m
rear); the front shocks, 0.3728 m at rest, inside their hydraulic stops
(0.324 to 0.419 m).

## 2. The bushed model's statics, before and after D47

From the design position, Newton alone (`max_relaxation_steps = 0`), with
the normal equations:

```
tolerance 1e-06: converged 0, 6 iterations, residual 2.37e+00, dof 18 supported 12 neutral 6
  history: 1.72e+03 1.85e+02 2.37e+00 2.37e+00 2.37e+00 2.37e+00
  chassis com height 0.70729 m
```

With relaxation (the defaults): 50 iterations, the history jumping from 2.37
to 2.18e+07 after a relaxation of 20000 steps, the chassis at 17.39 m.

The singular values of the scaled Jacobian Gs at each Newton iteration,
divided by the largest (a temporary probe in `newton_step`, removed):

```
bushed, first iteration (largest 2.07e+07):
 1.0 1.0 0.70 0.70 8.9e-03 8.8e-03 4.6e-03 4.5e-03 1.4e-04 1.3e-04 1.2e-04 1.1e-04 8.4e-06 7.3e-06 4.8e-06
 | 9.3e-12 7.6e-12 6.3e-12 5.1e-12 6.8e-13 ... 1.9e-18
bushed, later iterations: the same 15 down to 4.3e-06 to 4.9e-06, then 1.8e-13 and below
joints, first iteration (largest 3.04e+03):
 1.0 0.96 0.85 0.77 5.7e-02 5.0e-02 3.2e-02 | 7.0e-08 6.4e-08 4.7e-08 4.4e-08 9.9e-10 ... 3.4e-15
joints, later iterations: the same 7 down to 2.4e-02, then 2.2e-10 and below
```

With the QR step (the committed solver):

```
tolerance 1e-06: converged 1, 4 iterations, residual 7.46e-09, dof 18 supported 15 neutral 3
  history: 2.54e+01 1.84e+00 1.89e-06 7.46e-09
  chassis com height 0.68753 m
tolerance 1e-09: converged 0, 9 iterations, residual 3.44e-09 (the rounding floor)
```

## 3. The modes at rest

`mbd_hmmwv modes` (linearize at the static equilibrium):

```
with joints: statics converged in 3 iterations (residual 2.7e-11), 7 of 10 directions with stiffness; 12 modes, the fastest:
  -6.3493e+00 +1.5765e+01i  (|lambda| 1.6995e+01 rad/s, damping ratio 0.374)
  -5.0363e+01 +0.0000e+00i  (|lambda| 5.0363e+01 rad/s, damping ratio 1.000)
  -3.7288e+01 +4.6523e+01i  (|lambda| 5.9622e+01 rad/s, damping ratio 0.625)
  -3.6769e+01 +4.7876e+01i  (|lambda| 6.0366e+01 rad/s, damping ratio 0.609)
  -5.6810e+01 +2.1032e+01i  (|lambda| 6.0578e+01 rad/s, damping ratio 0.938)
  -6.6589e+01 +0.0000e+00i  (|lambda| 6.6589e+01 rad/s, damping ratio 1.000)
with bushings: statics converged in 4 iterations (residual 7.5e-09), 15 of 18 directions with stiffness; 27 modes, the fastest:
  -1.8022e+04 +0.0000e+00i  (|lambda| 1.8022e+04 rad/s, damping ratio 1.000)
  -1.8116e+04 +0.0000e+00i  (|lambda| 1.8116e+04 rad/s, damping ratio 1.000)
  -1.4393e+06 +0.0000e+00i  (|lambda| 1.4393e+06 rad/s, damping ratio 1.000)
  -1.4433e+06 +0.0000e+00i  (|lambda| 1.4433e+06 rad/s, damping ratio 1.000)
  -2.0590e+06 +0.0000e+00i  (|lambda| 2.0590e+06 rad/s, damping ratio 1.000)
  -2.0617e+06 +0.0000e+00i  (|lambda| 2.0617e+06 rad/s, damping ratio 1.000)
```

RK4 on the bushed model from the start (an exploration, Chrono-height bump
placement): at 2e-5, 1e-5 and 5e-6 s the projection failed at t = 0.000100,
0.000070 and 0.000045 s, and every position became NaN.

## 4. Exploratory runs and the events

The first benchmark (`quick`, no events, the bump then placed where the
unloaded tyre touches it, 10 ms before its force turns), not pinned:

| method | parameter | any body [m] | time [s] |
|---|---|---|---|
| RK4 | 4, 2, 1, 0.5, 0.25 ms | 4.28, 6.84e-3, 4.44e-3, 1.82e-3, 1.21e-3 | 0.13 to 1.75 |
| Dormand-Prince | tol 1e-2 ... 1e-8 | 5.38e-3, 5.38e-3, 1.57e-2, 4.31e-4, 5.61e-5, 3.10e-7 | 0.07 to 0.13 |
| generalised-alpha | 4, 2, 1, 0.5, 0.25 ms | 4.23, 1.52e-2, 1.33e-3, 3.66e-3, 4.50e-4 | 0.11 to 1.27 |
| Radau IIA | tol 1e-2 ... 1e-8 | 8.99e-3, 7.12e-4, 1.02e-4, 2.42e-6, 1.21e-6, 2.30e-9 | 0.07 to 0.37 |

The same with the tyres' events (the bump placed as committed), not
pinned, in a slower state of the machine: RK4 1.43e-3, 8.67e-4, 2.82e-4 m at
1, 0.5, 0.25 ms (chassis height 9.5e-6, 5.0e-6, 9.3e-7 m); Dormand-Prince
1.59e-2 m at 1e-2 to 1e-4; generalised-alpha failed at 4, 2, 1 and 0.5 ms
(and at every step on the bushed model); Radau IIA 1.04e-5 m at 1e-2,
2.76e-9 at 1e-8; the reference took 163 s for 7425 steps against 0.94 s for
2704 without events. The linearly implicit Euler step reported MBD-K045
at t = 0.510 s, where the front tyres turn to the bump.

The probe, back to back (`mbd_hmmwv probe`, since removed):

```
RK4 1 ms, events 0: 0.804 s, 5000 steps, 20000 evaluations, 0 events
RK4 1 ms, events 1: 0.865 s, 5283 steps, 21132 evaluations, 22 events
RK4 1 ms, events 0: 0.807 s, 5000 steps, 20000 evaluations, 0 events
RK4 1 ms, events 1: 1.679 s, 5283 steps, 21132 evaluations, 22 events
generalised-alpha 1 ms, events 0: ran after 5000 steps (1.505 s)
generalised-alpha 1 ms, events 1: FAILED after 746 steps (0.185 s): MBD-K058: kernel::Simulator: the
  generalised-alpha step of 0.000362 s from t = 0.606000 s did not converge within 10 Newton iterations,
  or its corrections stopped falling, even with an iteration matrix just built there. The forces change
  too fast across the step, or are not finite.
```

## 5. The tests

`build\tests\test_kernel.exe "[hmmwv]" -s`, the values behind the checks:

```
chassis height on joints 0.687549 m, on bushings 0.687526 m
lower arm angle (joints) 2.27956e-09, 9.17434e-10 rad; (bushings) 2.72474e-06, 7.88791e-06 rad
the front left tyre turned to the bump at 0.5 s
Dormand-Prince 1.2744e-05 m at 1e-6, 4.31242e-08 m at 1e-8; RK4 0.000764959 m at 1 ms, 0.000271668 m at 0.5 ms
after 1 ms at 2.5e-6 s: |q| -nan(ind), |v| nan, projection failures 381
RK4 at 1.25e-6 s against Radau IIA at 1e-9: 7.87128e-08 m over 15 ms; 0 projection failures
Radau IIA 6.68633e-08 m, SDIRK4 4.32563e-06 m at 1e-6; the chassis's height with bushings 3.41655e-05 m
  from its height with joints
```

Durations (`--durations yes`, evening): 0.03, 0.13, 0.35, 3.1, 6.6 and 4.9 s.
The statics tests: the soft spring's position within 2.3e-13 m of -g / k2.

## 6. The full test suite

`scripts\build.ps1 -Test`: 516 of 516 passed (85.9 s, evening; again on 8
November after the last edits, 43.3 s), no compiler warning.

## 7. The benchmark

`build\bench\mbd_hmmwv.exe`, pinned to logical processor 2 at high priority, the machine kept
from idle sleep (`SetThreadExecutionState`), 8 November 2026 from 12:43 to 13:28. A first run,
started at 21:35 the evening before, was suspended by the laptop's sleep from about 22:00 to
12:40 during the bushed model's second RK4 run and was stopped; its joint-model table without
events agrees with this one to the digits printed in differences and counts, and differs in
times (its event table was cut off in the output buffer). The program
then ran RK4 at 1.25e-6 and 1e-6 s and Dormand-Prince at 1e-3 and 1e-6 on the bushed model;
this run keeps one of each. Its output, as printed (its MBD-K045 warnings, one per linearly
implicit run, all at t = 0.500000 s, on standard error):


## The HMMWV with joints

Reference: Radau IIA at tol 1e-10, 1.4 s, 2746 steps; z1 from 0.679145 to 0.706081 m; speed at 5 s 4.250279 m/s.

### with joints

| method | step or tolerance | difference, any body [m] | difference, chassis height [m] | time [s] | steps (rejected) | evaluations | Newton iterations | Jacobians |
|---|---|---|---|---|---|---|---|---|
| RK4 | h 4 ms | 4.26e+00 | 2.37e-02 | 0.207 | 1500 (0) | 6000 | 0 | 0 |
| RK4 | h 2 ms | 4.75e-03 | 5.07e-06 | 0.342 | 2500 (0) | 10000 | 0 | 0 |
| RK4 | h 1 ms | 6.03e-03 | 4.15e-05 | 0.682 | 5000 (0) | 20000 | 0 | 0 |
| RK4 | h 0.5 ms | 1.31e-03 | 1.39e-05 | 2.24 | 10000 (0) | 40000 | 0 | 0 |
| RK4 | h 0.25 ms | 1.04e-04 | 1.14e-05 | 4.77 | 20000 (0) | 80000 | 0 | 0 |
| Dormand-Prince | tol 1e-02 | 3.93e-02 | 2.71e-04 | 0.197 | 500 (0) | 3500 | 0 | 0 |
| Dormand-Prince | tol 1e-03 | 1.50e-02 | 1.16e-04 | 0.201 | 505 (3) | 3553 | 0 | 0 |
| Dormand-Prince | tol 1e-04 | 1.20e-02 | 1.11e-04 | 0.204 | 513 (6) | 3627 | 0 | 0 |
| Dormand-Prince | tol 1e-05 | 4.33e-05 | 6.24e-06 | 0.229 | 554 (39) | 4112 | 0 | 0 |
| Dormand-Prince | tol 1e-06 | 9.35e-06 | 6.72e-07 | 0.253 | 608 (52) | 4568 | 0 | 0 |
| Dormand-Prince | tol 1e-08 | 2.22e-07 | 5.70e-09 | 0.353 | 843 (91) | 6447 | 0 | 0 |
| generalised-alpha | h 4 ms | 4.25e+00 | 2.37e-02 | 0.298 | 1500 (0) | 3871 | 2317 | 1 |
| generalised-alpha | h 2 ms | 1.81e-02 | 7.03e-05 | 0.488 | 2500 (0) | 6357 | 3803 | 1 |
| generalised-alpha | h 1 ms | 9.05e-03 | 5.74e-05 | 0.916 | 5000 (0) | 11911 | 6857 | 1 |
| generalised-alpha | h 0.5 ms | 8.27e-04 | 5.86e-05 | 1.74 | 10000 (0) | 22540 | 12486 | 1 |
| generalised-alpha | h 0.25 ms | 2.68e-03 | 1.24e-05 | 3.31 | 20000 (0) | 42361 | 22307 | 1 |
| SDIRK4 | tol 1e-02 | 9.64e-03 | 1.13e-04 | 0.195 | 513 (13) | 2931 | 2871 | 1 |
| SDIRK4 | tol 1e-03 | 9.80e-04 | 1.53e-04 | 0.244 | 529 (22) | 3703 | 3583 | 2 |
| SDIRK4 | tol 1e-04 | 2.94e-03 | 5.18e-05 | 0.311 | 557 (25) | 4964 | 4904 | 1 |
| SDIRK4 | tol 1e-05 | 2.63e-03 | 2.70e-05 | 0.551 | 610 (33) | 9234 | 9114 | 2 |
| SDIRK4 | tol 1e-06 | 2.66e-03 | 2.14e-05 | 0.594 | 717 (44) | 9781 | 9721 | 1 |
| SDIRK4 | tol 1e-08 | 7.96e-04 | 1.05e-05 | 1.4 | 1308 (49) | 22965 | 22905 | 1 |
| Radau IIA | tol 1e-02 | 6.47e-04 | 6.81e-05 | 0.168 | 508 (9) | 2359 | 576 | 2 |
| Radau IIA | tol 1e-03 | 5.24e-04 | 1.38e-05 | 0.192 | 520 (21) | 2668 | 654 | 3 |
| Radau IIA | tol 1e-04 | 1.51e-05 | 9.98e-07 | 0.235 | 552 (42) | 3159 | 842 | 1 |
| Radau IIA | tol 1e-05 | 5.36e-06 | 4.97e-08 | 0.329 | 618 (68) | 4307 | 1201 | 1 |
| Radau IIA | tol 1e-06 | 1.99e-07 | 1.01e-08 | 0.457 | 703 (79) | 5777 | 1663 | 1 |
| Radau IIA | tol 1e-08 | 6.35e-10 | 1.75e-10 | 0.957 | 1175 (101) | 11440 | 3392 | 1 |
| linearly implicit Euler | h 1 ms | 4.38e-02 | 5.31e-04 | 1.38 | 5000 (0) | 5000 | 0 | 5000 |
| linearly implicit Euler | h 0.5 ms | 1.95e-02 | 2.45e-04 | 2.71 | 10000 (0) | 10000 | 0 | 10000 |
| linearly implicit Euler | h 0.25 ms | 1.05e-02 | 1.33e-04 | 5.39 | 20000 (0) | 20000 | 0 | 20000 |
| linearly implicit Euler | h 0.125 ms | 5.95e-03 | 6.90e-05 | 10.7 | 40000 (0) | 40000 | 0 | 40000 |

The reference with the tyres' events (Radau IIA at tol 1e-10, 6.98 s, 6952 steps, 19 events) differs from it by 1.64e-11 m (any body), 1.55e-12 m (chassis height).

### with joints, with the tyres' events

| method | step or tolerance | difference, any body [m] | difference, chassis height [m] | time [s] | steps (rejected) | evaluations | Newton iterations | Jacobians |
|---|---|---|---|---|---|---|---|---|
| RK4 | h 2 ms | 7.52e-04 | 6.31e-06 | 0.685 | 2765 (0) | 11060 | 0 | 0 |
| RK4 | h 1 ms | 1.99e-03 | 8.26e-06 | 1.31 | 5283 (0) | 21132 | 0 | 0 |
| RK4 | h 0.5 ms | 1.09e-03 | 6.56e-06 | 2.55 | 10270 (0) | 41080 | 0 | 0 |
| RK4 | h 0.25 ms | 5.13e-04 | 3.00e-06 | 5.01 | 20268 (0) | 81072 | 0 | 0 |
| Dormand-Prince | tol 1e-03 | 1.42e-02 | 1.03e-04 | 0.217 | 521 (3) | 3665 | 0 | 0 |
| Dormand-Prince | tol 1e-06 | 3.99e-06 | 2.02e-07 | 0.272 | 622 (50) | 4654 | 0 | 0 |
| generalised-alpha | h 1 ms | failed: MBD-K058: kernel::Simulator: the generalised-alpha step of 0.000362 s from t = 0.606000 s | | 0.124 | 746 | | | |
| generalised-alpha | h 0.25 ms | failed: MBD-K058: kernel::Simulator: the generalised-alpha step of 0.000077 s from t = 0.606250 s | | 0.443 | 2558 | | | |
| SDIRK4 | tol 1e-03 | 1.82e-04 | 6.27e-06 | 1.11 | 1321 (199) | 17486 | 10826 | 111 |
| SDIRK4 | tol 1e-06 | 2.12e-07 | 8.30e-09 | 3.27 | 2426 (533) | 49831 | 38311 | 192 |
| Radau IIA | tol 1e-03 | 5.76e-05 | 5.65e-07 | 0.835 | 1056 (113) | 10312 | 2366 | 36 |
| Radau IIA | tol 1e-06 | 1.52e-07 | 1.31e-09 | 2.16 | 2253 (631) | 25981 | 6671 | 61 |
| linearly implicit Euler | h 1 ms | 2.18e-02 | 3.76e-04 | 1.5 | 5281 (0) | 5281 | 0 | 5281 |
| linearly implicit Euler | h 0.5 ms | 1.08e-02 | 1.92e-04 | 2.87 | 10245 (0) | 10245 | 0 | 10245 |

## The HMMWV with bushings

Reference: Radau IIA at tol 1e-10, 4.24 s, 2936 steps; z1 from 0.679123 to 0.706047 m; speed at 5 s 4.250286 m/s.

### with bushings

| method | step or tolerance | difference, any body [m] | difference, chassis height [m] | time [s] | steps (rejected) | evaluations | Newton iterations | Jacobians |
|---|---|---|---|---|---|---|---|---|
| RK4 | h 0.00125 ms | 3.55e-06 | 3.42e-08 | 1.06e+03 | 4000000 (0) | 16000000 | 0 | 0 |
| Dormand-Prince | tol 1e-06 | 2.03e-06 | 9.17e-09 | 1.52e+03 | 3113941 (515854) | 24892711 | 0 | 0 |
| generalised-alpha | h 4 ms | 4.25e+00 | 2.37e-02 | 0.349 | 1500 (0) | 3933 | 2363 | 1 |
| generalised-alpha | h 2 ms | 1.79e-02 | 6.82e-05 | 0.568 | 2500 (0) | 6348 | 3778 | 1 |
| generalised-alpha | h 1 ms | 9.28e-03 | 5.55e-05 | 1.07 | 5000 (0) | 11845 | 6775 | 1 |
| generalised-alpha | h 0.5 ms | 1.05e-03 | 5.66e-05 | 2.09 | 10000 (0) | 22526 | 12456 | 1 |
| generalised-alpha | h 0.25 ms | 2.90e-03 | 2.09e-05 | 3.88 | 20000 (0) | 42288 | 22218 | 1 |
| SDIRK4 | tol 1e-02 | 9.80e-04 | 1.58e-04 | 0.485 | 514 (13) | 6761 | 6521 | 3 |
| SDIRK4 | tol 1e-03 | 1.85e-03 | 1.39e-04 | 0.28 | 528 (19) | 3694 | 3614 | 1 |
| SDIRK4 | tol 1e-04 | 2.26e-04 | 4.21e-05 | 0.423 | 559 (29) | 5095 | 5015 | 1 |
| SDIRK4 | tol 1e-05 | 5.69e-06 | 2.55e-07 | 0.576 | 652 (62) | 7879 | 7799 | 1 |
| SDIRK4 | tol 1e-06 | 1.61e-04 | 1.36e-06 | 0.814 | 761 (67) | 11149 | 11069 | 1 |
| SDIRK4 | tol 1e-08 | 4.41e-07 | 3.74e-09 | 2.14 | 1495 (71) | 29087 | 29007 | 1 |
| Radau IIA | tol 1e-02 | 9.76e-03 | 7.75e-05 | 0.287 | 514 (8) | 3396 | 776 | 7 |
| Radau IIA | tol 1e-03 | 7.82e-04 | 1.33e-05 | 0.342 | 531 (29) | 3912 | 886 | 9 |
| Radau IIA | tol 1e-04 | 2.06e-05 | 9.61e-07 | 0.298 | 543 (34) | 3157 | 839 | 1 |
| Radau IIA | tol 1e-05 | 7.20e-06 | 2.32e-07 | 0.514 | 615 (57) | 5403 | 1300 | 11 |
| Radau IIA | tol 1e-06 | 2.41e-07 | 8.10e-09 | 0.78 | 706 (76) | 7116 | 1838 | 11 |
| Radau IIA | tol 1e-08 | 2.23e-09 | 1.44e-10 | 1.67 | 1228 (102) | 13829 | 3899 | 11 |
| linearly implicit Euler | h 1 ms | 4.36e-02 | 5.27e-04 | 1.92 | 5000 (0) | 5000 | 0 | 5000 |
| linearly implicit Euler | h 0.5 ms | 1.93e-02 | 2.43e-04 | 3.94 | 10000 (0) | 10000 | 0 | 10000 |
| linearly implicit Euler | h 0.25 ms | 1.02e-02 | 1.23e-04 | 7.74 | 20000 (0) | 20000 | 0 | 20000 |
| linearly implicit Euler | h 0.125 ms | 5.72e-03 | 6.79e-05 | 14.9 | 40000 (0) | 40000 | 0 | 40000 |
