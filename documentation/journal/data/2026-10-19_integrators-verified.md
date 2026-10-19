# The integrators verified: measurements (task 4.9, D40, D41)

Machine: the development laptop (i7-13700H, 16 GB), Windows 11, MSVC
RelWithDebInfo (`/O2`), one compiler.

## 1. The verification run

`build\bench\mbd_verify.exe docs\integrators`, pinned to a performance core
at high priority (a PowerShell `ProcessStartInfo`, `ProcessorAffinity` 4,
`PriorityClass` High), 19 October 2026, after D40: 2 min 2 s. Its standard
output follows, as printed (the report, `docs/integrators.md`, quotes it).
Standard error held only warnings: MBD-K050 once per four-bar simulator
(the loop's 5 equations of rank 2), and MBD-K045 four times, from
semi-implicit Euler's divergent runs on the pendulums (at t = 0.0008,
0.0011, 0.040 and 0.224 s). The CSV of every run is in `docs/integrators/`.


## The oscillator

Error |(q, v) - (cos t, -sin t)| at t = 10 s, released from q = 1 m:

| method | 200 ms | 100 ms | 50 ms | 25 ms | 12.5 ms | 6.25 ms | observed orders |
|---|---|---|---|---|---|---|---|
| RK4 | 0.000133 | 8.33e-06 | 5.21e-07 | 3.26e-08 | 2.03e-09 | 1.27e-10 | 4, 4, 4, 4, 4 |
| semi-implicit Euler | 0.0674 | 0.03 | 0.0142 | 0.00695 | 0.00344 | 0.00171 | 1.17, 1.07, 1.03, 1.02, 1.01 |
| Dormand-Prince | 9.01e-07 | 2.79e-08 | 8.69e-10 | 2.71e-11 | 8.46e-13 | 3.01e-13 | 5.01, 5, 5, 5, 1.49 |
| generalised-alpha | 0.035 | 0.00879 | 0.0022 | 0.00055 | 0.000138 | 3.44e-05 | 1.99, 2, 2, 2, 2 |
| SDIRK4 | 1.35e-05 | 8.46e-07 | 5.29e-08 | 3.31e-09 | 2.07e-10 | 1.32e-11 | 4, 4, 4, 4, 3.97 |
| Radau IIA | 4.44e-07 | 1.39e-08 | 4.34e-10 | 1.36e-11 | 4.26e-13 | 3.02e-13 | 5, 5, 5, 4.99, 0.499 |
| linearly implicit Euler | 0.63 | 0.393 | 0.221 | 0.117 | 0.0606 | 0.0308 | 0.681, 0.829, 0.912, 0.955, 0.978 |

Energy after 1000 s, E / E0, measured and predicted by the method's recurrence on the oscillator; and its range over the run:

| method | w h | steps | E / E0 measured | predicted | relative difference | least, largest E / E0 |
|---|---|---|---|---|---|---|
| RK4 | 0.1 | 10000 | 0.9998612943 | 0.9998612943 | 4.8e-13 | 0.999861, 1 |
| semi-implicit Euler | 0.1 | 10000 | 0.9872042584 | 0.9872042584 | 0 | 0.952381, 1.05263 |
| Dormand-Prince | 0.1 | 10000 | 0.9999945069 | 0.9999945069 | 1.2e-13 | 0.999995, 1 |
| generalised-alpha | 0.1 | 10000 | 0.998633078 | 0.998633078 | 2.2e-16 | 0.998605, 1 |
| SDIRK4 | 0.1 | 10000 | 0.9999994598 | 0.9999994598 | 1.1e-12 | 0.999999, 1 |
| Radau IIA | 0.1 | 10000 | 0.9999972239 | 0.9999972239 | 4.7e-13 | 0.999997, 1 |
| linearly implicit Euler | 0.1 | 10000 | 6.113109479e-44 | 6.113109479e-44 | 2.8e-11 | 6.11311e-44, 1 |
| RK4 | 0.5 | 2000 | 0.6567127219 | 0.6567127219 | 6.5e-14 | 0.656713, 1 |
| semi-implicit Euler | 0.5 | 2000 | 1.332483674 | 1.332483674 | 0 | 0.8, 1.33333 |
| Dormand-Prince | 0.5 | 2000 | 0.98746659 | 0.98746659 | 3.9e-13 | 0.987467, 1 |
| generalised-alpha | 0.5 | 2000 | 0.8519102662 | 0.8519102662 | 2.6e-15 | 0.85191, 1 |
| SDIRK4 | 0.5 | 2000 | 0.9984795071 | 0.9984795071 | 8.4e-13 | 0.99848, 1 |
| Radau IIA | 0.5 | 2000 | 0.9914855184 | 0.9914855184 | 1.3e-13 | 0.991486, 1 |
| linearly implicit Euler | 0.5 | 2000 | 1.513470582e-194 | 1.513470582e-194 | 1.1e-11 | 1.51347e-194, 1 |

## The soft bushed pendulum (smooth, spatial)

Reference (the tip at 1 s): Dormand-Prince at tolerance 1e-13 (40310 evaluations); Radau IIA at 1e-12 differs from it by 2.6e-14.

The fastest motions of the pendulum at the start (linearised): 1.447e+04 rad/s, damping ratio 1; 2073 rad/s, damping ratio 1; 297.1 rad/s, damping ratio 0.623; 296.8 rad/s, damping ratio 0.594.

Error of the tip at 1 s [m], fixed steps:

| method | 8 ms | 4 ms | 2 ms | 1 ms | 500 us | 250 us | observed orders |
|---|---|---|---|---|---|---|---|
| RK4 | 1.98e-05 | 1.1e-06 | 6.32e-08 | 3.77e-09 | 2.3e-10 | 1.42e-11 | 4.17, 4.12, 4.07, 4.04, 4.01 |
| semi-implicit Euler | diverged | diverged | 0.00636 | 0.00319 | 0.0016 | 0.000801 | 0.993, 0.997, 0.998 |
| Dormand-Prince | 1.54e-06 | 2.15e-08 | 4.17e-10 | 1.04e-11 | 5.13e-13 | 3.59e-13 | 6.16, 5.69, 5.33, 4.34, 0.514 |
| generalised-alpha | 0.000492 | 0.000123 | 3.09e-05 | 7.72e-06 | 1.93e-06 | 4.83e-07 | 2, 2, 2, 2, 2 |
| SDIRK4 | 5.57e-06 | 3.74e-07 | 2.29e-08 | 1.38e-09 | 8.45e-11 | 5.51e-12 | 3.9, 4.03, 4.05, 4.03, 3.94 |
| Radau IIA | 6.26e-09 | 1.92e-10 | 5.92e-12 | 1.54e-13 | 2.16e-13 | 2.33e-13 | 5.02, 5.02, 5.26, -0.488, -0.108 |
| linearly implicit Euler | 0.0584 | 0.0316 | 0.0165 | 0.0084 | 0.00424 | 0.00213 | 0.886, 0.942, 0.971, 0.985, 0.993 |

The cheapest run within each error:

| method | error <= 0.001 | error <= 1e-06 | error <= 1e-09 |
|---|---|---|---|
| RK4 | 1.01 ms (8 ms, 500 evaluations) | 4.04 ms (2 ms, 2000 evaluations) | 16.3 ms (500 us, 8000 evaluations) |
| semi-implicit Euler | 26.4 ms (250 us, 4000 evaluations) | - | - |
| Dormand-Prince | 1.8 ms (8 ms, 875 evaluations) | 3.58 ms (4 ms, 1750 evaluations) | 7.18 ms (2 ms, 3500 evaluations) |
| generalised-alpha | 3.8 ms (8 ms, 1640 evaluations) | 51.5 ms (250 us, 19661 evaluations) | - |
| SDIRK4 | 13.8 ms (8 ms, 5566 evaluations) | 25.4 ms (4 ms, 10119 evaluations) | 150 ms (500 us, 59949 evaluations) |
| Radau IIA | 9.46 ms (8 ms, 3702 evaluations) | 9.46 ms (8 ms, 3702 evaluations) | 15.2 ms (4 ms, 5880 evaluations) |
| linearly implicit Euler | - | - | - |
| Dormand-Prince, error control | 1.48 ms (tol 0.001, 720 evaluations) | 3.05 ms (tol 1e-05, 1518 evaluations) | 8.8 ms (tol 1e-08, 4447 evaluations) |
| SDIRK4, error control | 9.86 ms (tol 0.0001, 3570 evaluations) | 19.8 ms (tol 1e-05, 7284 evaluations) | 112 ms (tol 1e-08, 41242 evaluations) |
| Radau IIA, error control | 2.58 ms (tol 0.001, 932 evaluations) | 5.58 ms (tol 1e-05, 1983 evaluations) | 18.9 ms (tol 1e-07, 5506 evaluations) |

## The four-bar with a pendulum (closed loop)

Reference (the joint angles at 1 s): Dormand-Prince at tolerance 1e-13 (7887 evaluations); Radau IIA at 1e-12 differs from it by 2.2e-13.

The fastest motions of the four-bar at the start (linearised): 8.535 rad/s, damping ratio 1; 6.02 rad/s, damping ratio -1; 4.807 rad/s, damping ratio 1; 4.515 rad/s, damping ratio -1.

Error of the joint angles at 1 s [rad], fixed steps, projected after each (the stabilized form: on the constraints):

| method | 8 ms | 4 ms | 2 ms | 1 ms | 500 us | 250 us | observed orders |
|---|---|---|---|---|---|---|---|
| RK4 | 9.73e-05 | 3.63e-06 | 1.66e-07 | 9.47e-09 | 5.71e-10 | 3.53e-11 | 4.74, 4.45, 4.13, 4.05, 4.01 |
| semi-implicit Euler | 0.907 | 0.502 | 0.263 | 0.135 | 0.0681 | 0.0343 | 0.854, 0.93, 0.967, 0.984, 0.992 |
| Dormand-Prince | 1.11e-06 | 1.12e-08 | 2.88e-10 | 9.63e-12 | 8.85e-13 | 8.74e-13 | 6.62, 5.28, 4.9, 3.44, 0.017 |
| generalised-alpha | MBD-K058 | 0.00143 | 0.000287 | 6.27e-05 | 1.46e-05 | 3.5e-06 | 2.32, 2.19, 2.11, 2.06 |
| generalised-alpha, stabilized | 0.00402 | 0.000926 | 0.000223 | 5.48e-05 | 1.36e-05 | 3.38e-06 | 2.12, 2.05, 2.03, 2.01, 2.01 |
| SDIRK4 | 4.78e-06 | 2.17e-07 | 1.09e-08 | 3.06e-10 | 1.66e-11 | 7.16e-13 | 4.46, 4.31, 5.16, 4.2, 4.53 |
| Radau IIA | 1.61e-07 | 6.6e-09 | 3.78e-10 | 1.19e-11 | 9.31e-13 | 8.65e-13 | 4.61, 4.13, 4.98, 3.68, 0.106 |
| linearly implicit Euler | 0.907 | 0.502 | 0.263 | 0.135 | 0.0681 | 0.0343 | 0.854, 0.93, 0.967, 0.984, 0.992 |

Generalised-alpha at 8 ms with Newton at 1e-9: error 0.00794.

Energy over 100 s, relative to the start (17.83 J): the change at the end and the largest; error control at tolerance 1e-6 within each output step:

| method | 5 ms: at 100 s | largest | 1 ms: at 100 s | largest |
|---|---|---|---|---|
| RK4 | -5.78e-05 | -5.78e-05 | -3.33e-08 | -3.48e-08 |
| semi-implicit Euler | -0.822 | -0.822 | -0.528 | -0.567 |
| Dormand-Prince, error control | 1.11e-07 | 1.94e-07 | -2.65e-11 | -3.31e-11 |
| generalised-alpha | -0.0018 | -0.00434 | 9.97e-05 | -0.00012 |
| generalised-alpha, stabilized | -0.000424 | -0.00296 | -4.98e-05 | -0.000181 |
| SDIRK4, error control | 2.7e-07 | 6.96e-07 | 4.84e-08 | 5.19e-08 |
| Radau IIA, error control | -5.33e-08 | -6.06e-08 | -3.78e-09 | -3.78e-09 |
| linearly implicit Euler | -0.822 | -0.822 | -0.528 | -0.567 |

## The heavy top (gyroscopic, held at its tip)

Over 100 s (energy 3.861 J, spin 30 rad/s; no damping): energy change at the end and the largest, relative; the spin at the end [rad/s]:

| method | 5 ms: energy at 100 s | largest | spin | 1 ms: energy at 100 s | largest | spin |
|---|---|---|---|---|---|---|
| RK4 | -3.74e-05 | -3.74e-05 | 30 | -3.49e-08 | -3.49e-08 | 30 |
| semi-implicit Euler | -0.477 | -0.477 | 30 | -0.477 | -0.477 | 30 |
| Dormand-Prince, error control | -1.02e-06 | -1.02e-06 | 30 | -3.18e-10 | -3.18e-10 | 30 |
| generalised-alpha | -0.000715 | -0.001 | 30 | -5.33e-06 | -1.91e-05 | 30 |
| generalised-alpha, stabilized | -0.000715 | -0.001 | 30 | -5.33e-06 | -1.91e-05 | 30 |
| SDIRK4, error control | -3.59e-06 | -3.59e-06 | 30 | 5.62e-09 | 5.9e-09 | 30 |
| Radau IIA, error control | -1.3e-06 | -1.3e-06 | 30 | -3.25e-09 | -3.25e-09 | 30 |
| linearly implicit Euler | -0.477 | -0.477 | 30 | -0.477 | -0.477 | 30 |

Generalised-alpha projected and stabilized, in full:

- generalised-alpha: energy at 100 s 3.8581470925 J at 5 ms, 3.86088789693 J at 1 ms; Newton iterations 80000 and 300000; largest drift of a step 7.7e-15 and 4.7e-14 m
- generalised-alpha, stabilized: energy at 100 s 3.8581470925 J at 5 ms, 3.86088789693 J at 1 ms; Newton iterations 80000 and 300000; largest drift of a step 9.1e-17 and 9.7e-17 m

## The stiff bushed pendulum (k = 1e8 N/m)

Reference (the tip at 1 s): Dormand-Prince at tolerance 1e-13 (1538675 evaluations); Radau IIA at 1e-12 differs from it by 2.3e-13.

The fastest motions of the pendulum at the start (linearised): 1.447e+06 rad/s, damping ratio 1; 2.073e+05 rad/s, damping ratio 1; 2.971e+04 rad/s, damping ratio 0.623; 2.968e+04 rad/s, damping ratio 0.594.

Every run (error of the tip at 1 s [m]):

| method | step or tolerance | error | evaluations | steps | rejected | Newton iterations | time [ms] |
|---|---|---|---|---|---|---|---|
| RK4 | 100 us | diverged | 40000 | 10000 | 0 | 0 | 0 |
| RK4 | 50 us | 2.87e-10 | 80000 | 20000 | 0 | 0 | 162 |
| RK4 | 25 us | 1.6e-11 | 160000 | 40000 | 0 | 0 | 327 |
| semi-implicit Euler | 100 us | diverged | 10000 | 10000 | 0 | 0 | 0 |
| semi-implicit Euler | 50 us | diverged | 20000 | 20000 | 0 | 0 | 0 |
| semi-implicit Euler | 25 us | 8.52e-05 | 40000 | 40000 | 0 | 0 | 267 |
| generalised-alpha | 4 ms | MBD-K058 | 0 | 0 | 0 | 0 | 0 |
| generalised-alpha | 2 ms | 2.72e-05 | 12242 | 500 | 0 | 6126 | 26.9 |
| generalised-alpha | 1 ms | 6.81e-06 | 10488 | 1000 | 0 | 7592 | 24.9 |
| generalised-alpha | 500 us | 1.72e-06 | 15856 | 2000 | 0 | 13136 | 39.2 |
| generalised-alpha | 250 us | 4.59e-07 | 26062 | 4000 | 0 | 21750 | 65.9 |
| generalised-alpha | 125 us | 7.78e-08 | 48596 | 8000 | 0 | 40500 | 124 |
| linearly implicit Euler | 4 ms | 0.32 | 250 | 250 | 0 | 0 | 4.08 |
| linearly implicit Euler | 2 ms | 0.481 | 500 | 500 | 0 | 0 | 8.15 |
| linearly implicit Euler | 1 ms | 0.0422 | 1000 | 1000 | 0 | 0 | 16.5 |
| linearly implicit Euler | 500 us | 0.00799 | 2000 | 2000 | 0 | 0 | 32.7 |
| linearly implicit Euler | 250 us | 0.00195 | 4000 | 4000 | 0 | 0 | 65.9 |
| linearly implicit Euler | 125 us | 0.000963 | 8000 | 8000 | 0 | 0 | 131 |
| linearly implicit Euler | 62.5 us | 0.000516 | 16000 | 16000 | 0 | 0 | 264 |
| Dormand-Prince, error control | tol 0.0001 | 2.35e-06 | 77952 | 8796 | 2730 | 0 | 155 |
| Dormand-Prince, error control | tol 1e-06 | 5.61e-10 | 81956 | 11216 | 574 | 0 | 163 |
| Dormand-Prince, error control | tol 1e-08 | 5.34e-12 | 161885 | 23093 | 39 | 0 | 322 |
| Dormand-Prince, error control | tol 1e-10 | 1.56e-13 | 388369 | 55471 | 12 | 0 | 771 |
| SDIRK4, error control | tol 0.001 | 0.0548 | 14985 | 645 | 6 | 14040 | 41.4 |
| SDIRK4, error control | tol 0.0001 | 0.000647 | 47995 | 1910 | 21 | 47644 | 132 |
| SDIRK4, error control | tol 1e-05 | 6.59e-05 | 149559 | 5339 | 21 | 149370 | 409 |
| SDIRK4, error control | tol 1e-06 | 3.75e-06 | 339742 | 11805 | 19 | 339607 | 924 |
| SDIRK4, error control | tol 1e-07 | 4.55e-08 | 654372 | 22748 | 18 | 654291 | 1.78e+03 |
| SDIRK4, error control | tol 1e-08 | 3.37e-08 | 1134763 | 41629 | 9 | 1134709 | 3.11e+03 |
| Radau IIA, error control | tol 0.001 | 0.00039 | 6310 | 134 | 34 | 1166 | 17.4 |
| Radau IIA, error control | tol 0.0001 | 3.78e-06 | 8399 | 157 | 57 | 1794 | 23.6 |
| Radau IIA, error control | tol 1e-05 | 1.24e-06 | 10663 | 199 | 75 | 2482 | 30.1 |
| Radau IIA, error control | tol 1e-06 | 1.04e-07 | 11548 | 240 | 53 | 2716 | 33.6 |
| Radau IIA, error control | tol 1e-07 | 1.03e-07 | 11298 | 348 | 44 | 2942 | 35.1 |
| Radau IIA, error control | tol 1e-08 | 4.06e-09 | 13986 | 641 | 48 | 4098 | 46.5 |
| Radau IIA, error control | tol 1e-09 | 7.57e-09 | 23973 | 1277 | 35 | 7397 | 82.1 |
| Radau IIA, error control | tol 1e-10 | 7.54e-10 | 44941 | 2622 | 17 | 14026 | 157 |

The cheapest run within each error:

| method | error <= 0.001 | error <= 1e-05 | error <= 1e-07 |
|---|---|---|---|
| RK4 | 162 ms (50 us, 80000 evaluations) | 162 ms (50 us, 80000 evaluations) | 162 ms (50 us, 80000 evaluations) |
| semi-implicit Euler | 267 ms (25 us, 40000 evaluations) | - | - |
| generalised-alpha | 24.9 ms (1 ms, 10488 evaluations) | 24.9 ms (1 ms, 10488 evaluations) | 124 ms (125 us, 48596 evaluations) |
| linearly implicit Euler | 131 ms (125 us, 8000 evaluations) | - | - |
| Dormand-Prince, error control | 155 ms (tol 0.0001, 77952 evaluations) | 155 ms (tol 0.0001, 77952 evaluations) | 163 ms (tol 1e-06, 81956 evaluations) |
| SDIRK4, error control | 132 ms (tol 0.0001, 47995 evaluations) | 924 ms (tol 1e-06, 339742 evaluations) | 1.78e+03 ms (tol 1e-07, 654372 evaluations) |
| Radau IIA, error control | 17.4 ms (tol 0.001, 6310 evaluations) | 23.6 ms (tol 0.0001, 8399 evaluations) | 46.5 ms (tol 1e-08, 13986 evaluations) |

## The sedan

The fastest motions of the sedan at the start (linearised): 390.2 rad/s, damping ratio 1; 264.1 rad/s, damping ratio 1; 229.4 rad/s, damping ratio 1; 187.3 rad/s, damping ratio 1.

Reference: RK4 at 0.05 ms (4.2 s to run); at 0.1 ms it differs by 2.8e-12 m. The chassis at 2 s: (2.34457, 0.344827, 0.0438645) m from its start (6.00454e-09, 0.33087, -1.81944e-06); gear 1 at the start.

Every run (error of the chassis's position at 2 s [m]):

| method | step or tolerance | error | evaluations | steps | rejected | Newton iterations | time [ms] |
|---|---|---|---|---|---|---|---|
| RK4 | 1 ms | 8.61e-05 | 8000 | 2000 | 0 | 0 | 211 |
| RK4 | 500 us | 1.77e-11 | 16000 | 4000 | 0 | 0 | 414 |
| RK4 | 250 us | 2.05e-11 | 32000 | 8000 | 0 | 0 | 953 |
| RK4 | 100 us | 2.75e-12 | 80000 | 20000 | 0 | 0 | 2.09e+03 |
| multirate RK4 (fast steps 0.5 ms) | 4 ms | 0.00314 | 2000 | 500 | 0 | 0 | 130 |
| multirate RK4 (fast steps 0.5 ms) | 2 ms | 0.000528 | 4000 | 1000 | 0 | 0 | 180 |
| multirate RK4 (fast steps 0.5 ms) | 1 ms | 0.000108 | 8000 | 2000 | 0 | 0 | 280 |
| semi-implicit Euler | 1 ms | 0.00268 | 2000 | 2000 | 0 | 0 | 167 |
| semi-implicit Euler | 500 us | 0.000585 | 4000 | 4000 | 0 | 0 | 325 |
| semi-implicit Euler | 250 us | 0.000292 | 8000 | 8000 | 0 | 0 | 646 |
| semi-implicit Euler | 100 us | 0.000117 | 20000 | 20000 | 0 | 0 | 1.58e+03 |
| generalised-alpha | 10 ms | MBD-K058 | 0 | 0 | 0 | 0 | 0 |
| generalised-alpha | 5 ms | 3.29e-07 | 3728 | 400 | 0 | 3270 | 96.8 |
| generalised-alpha | 2 ms | 4.82e-08 | 7122 | 1000 | 0 | 6064 | 190 |
| generalised-alpha | 1 ms | 1.24e-08 | 11398 | 2000 | 0 | 9340 | 314 |
| generalised-alpha | 500 us | 2.63e-09 | 18941 | 4000 | 0 | 14883 | 541 |
| generalised-alpha | 250 us | 6.72e-10 | 34494 | 8000 | 0 | 26436 | 1e+03 |
| linearly implicit Euler | 5 ms | 0.0061 | 400 | 400 | 0 | 0 | 54 |
| linearly implicit Euler | 2 ms | 0.00244 | 1000 | 1000 | 0 | 0 | 134 |
| linearly implicit Euler | 1 ms | 0.00122 | 2000 | 2000 | 0 | 0 | 266 |
| linearly implicit Euler | 500 us | 0.000611 | 4000 | 4000 | 0 | 0 | 522 |
| linearly implicit Euler | 250 us | 0.000305 | 8000 | 8000 | 0 | 0 | 1.04e+03 |
| linearly implicit Euler | 100 us | 0.000122 | 20000 | 20000 | 0 | 0 | 2.58e+03 |
| Dormand-Prince, error control | tol 0.0001 | 3.57e-07 | 11915 | 1457 | 286 | 0 | 282 |
| Dormand-Prince, error control | tol 1e-05 | 3.34e-09 | 12008 | 1460 | 298 | 0 | 283 |
| Dormand-Prince, error control | tol 1e-06 | 2.99e-10 | 12019 | 1459 | 301 | 0 | 285 |
| Dormand-Prince, error control | tol 1e-07 | 3.15e-10 | 11975 | 1463 | 289 | 0 | 284 |
| Dormand-Prince, error control | tol 1e-08 | 2.75e-10 | 11872 | 1504 | 224 | 0 | 283 |
| SDIRK4, error control | tol 0.001 | 8.91e-07 | 1226 | 202 | 2 | 1162 | 36.8 |
| SDIRK4, error control | tol 0.0001 | 5.84e-07 | 1913 | 207 | 3 | 1849 | 53.8 |
| SDIRK4, error control | tol 1e-05 | 9.2e-09 | 3337 | 219 | 4 | 3273 | 87.1 |
| SDIRK4, error control | tol 1e-06 | 3.08e-09 | 4830 | 242 | 11 | 4766 | 124 |
| Radau IIA, error control | tol 0.001 | 1.47e-07 | 1461 | 202 | 1 | 398 | 43.7 |
| Radau IIA, error control | tol 0.0001 | 7.27e-09 | 1586 | 205 | 3 | 438 | 47.1 |
| Radau IIA, error control | tol 1e-05 | 7.76e-09 | 1897 | 213 | 3 | 539 | 54.6 |
| Radau IIA, error control | tol 1e-06 | 4.58e-10 | 2515 | 234 | 9 | 737 | 75.4 |
| Radau IIA, error control | tol 1e-07 | 8.95e-11 | 3345 | 270 | 12 | 1002 | 103 |

The cheapest run within each error:

| method | error <= 0.01 | error <= 0.001 | error <= 0.0001 | error <= 1e-05 |
|---|---|---|---|---|
| RK4 | 211 ms (1 ms, 8000 evaluations) | 211 ms (1 ms, 8000 evaluations) | 211 ms (1 ms, 8000 evaluations) | 414 ms (500 us, 16000 evaluations) |
| multirate RK4 (fast steps 0.5 ms) | 130 ms (4 ms, 2000 evaluations) | 180 ms (2 ms, 4000 evaluations) | - | - |
| semi-implicit Euler | 167 ms (1 ms, 2000 evaluations) | 325 ms (500 us, 4000 evaluations) | - | - |
| generalised-alpha | 96.8 ms (5 ms, 3728 evaluations) | 96.8 ms (5 ms, 3728 evaluations) | 96.8 ms (5 ms, 3728 evaluations) | 96.8 ms (5 ms, 3728 evaluations) |
| linearly implicit Euler | 54 ms (5 ms, 400 evaluations) | 522 ms (500 us, 4000 evaluations) | - | - |
| Dormand-Prince, error control | 282 ms (tol 0.0001, 11915 evaluations) | 282 ms (tol 0.0001, 11915 evaluations) | 282 ms (tol 0.0001, 11915 evaluations) | 282 ms (tol 0.0001, 11915 evaluations) |
| SDIRK4, error control | 36.8 ms (tol 0.001, 1226 evaluations) | 36.8 ms (tol 0.001, 1226 evaluations) | 36.8 ms (tol 0.001, 1226 evaluations) | 36.8 ms (tol 0.001, 1226 evaluations) |
| Radau IIA, error control | 43.7 ms (tol 0.001, 1461 evaluations) | 43.7 ms (tol 0.001, 1461 evaluations) | 43.7 ms (tol 0.001, 1461 evaluations) | 43.7 ms (tol 0.001, 1461 evaluations) |

## 2. The linearly implicit step on the stiff pendulum, before D40

A temporary probe in the verification program (removed). The bushed
pendulum, 1 s with the linearly implicit step, the step's matrix built from
the symmetric parts of K and C (as D35) and, with a temporary switch, from
K and C whole (D40). "Diverged": the tip farther than 10 m from the hinge.

| k [N/m], k_r [N m/rad] | 2 ms | 1 ms | 0.5 ms | 0.25 ms |
|---|---|---|---|---|
| 1e8, 1e6, symmetric parts | diverged at step 461 | at step 975 | at step 1998 | ran |
| 1e8, 1e2, symmetric parts | ran | ran | ran | ran |
| 1e6, 1e6, symmetric parts | diverged at step 488 | ran (tip up to 3.5 m: off its links) | ran | ran |
| 1e6, 1e2, symmetric parts | ran | ran | ran | ran |
| 1e8, 1e6, damping removed, symmetric parts | diverged at step 449 | at step 966 | ran | |
| every variant above, whole | ran | ran | ran | ran |

The derivatives against central differences (step 1e-7) of the whole
generalized forces of the force elements, along each velocity (q (+) e_j),
after RK4 at 25 us to the time given (Frobenius norms):

| damping | time | \|K\| | \|K - K_fd\| | \|K - K^T\| | \|C - C_fd\| |
|---|---|---|---|---|---|
| as the model | 0 s | 5.57e8 | 10.0 | 14.1 | 0 |
| as the model | 0.1 s | 5.57e8 | 1.44e4 | 2.10e5 | 6.3e-6 |
| as the model | 0.3 s | 5.57e8 | 3.49e4 | 1.46e6 | 3.2e-5 |
| none | 0 s | 5.57e8 | 10.0 | 14.1 | 0 |
| none | 0.1 s | 5.57e8 | 14.3 | 2.07e5 | 0 |
| none | 0.3 s | 5.57e8 | 25 | 1.46e6 | 0 |

Without damping K is the derivative to 5e-8 of its norm; with it, K lacks
about c |v| (the change of a damper's force when a pose turns at fixed joint
rates, the joints' geometry left out by D35). The antisymmetric part grows
with the links' relative rotation, damping or not.

## 3. The energy gain after D40

A temporary hidden test (removed). Released level, at rest and
undeflected: kinetic plus gravity's energy can never rise above zero
(section 5 of the report). Its largest value over 1 s [J]:

| k [N/m] | k_r [N m/rad] | 2 ms | 1 ms | 0.5 ms | 0.25 ms | 0.125 ms |
|---|---|---|---|---|---|---|
| 1e8 | 1e6 | +8.585 | +0.2091 | -2.069e-5 | -5.222e-6 | -1.340e-6 |
| 1e8 | 1e2 | +8.585 | +0.2091 | -2.069e-5 | -5.222e-6 | -1.340e-6 |
| 1e7 | 1e6 | +0.0485 | -8.318e-5 | -2.120e-5 | -5.520e-6 | -1.447e-6 |
| 1e7 | 1e2 | +0.0485 | -8.318e-5 | -2.120e-5 | -5.520e-6 | -1.447e-6 |
| 1e6 | 1e6 | -3.362e-4 | -8.692e-5 | -2.281e-5 | -5.904e-6 | -1.496e-6 |
| 1e6 | 1e2 | -3.362e-4 | -8.692e-5 | -2.281e-5 | -5.904e-6 | -1.496e-6 |

The rotational stiffness changes nothing; k h^2 decides: a gain at k h^2 =
400 and 100 (k = 1e8 at 2 and 1 ms) and 40 (1e7 at 2 ms), none at 25 and
below (per kilogram of link).

## 4. The four-bar's sensitivity to rounding

A temporary hidden test (removed): semi-implicit Euler on the four-bar with
its pendulum, released as in the report, with the mass matrix factorized
densely and along the tree (D39; they differ by rounding, 1e-15).
Relative energy change, and the crank angle q0 [rad], every 20 s:

| run | 20 s | 40 s | 60 s | 80 s | 100 s |
|---|---|---|---|---|---|
| 5 ms, dense | -0.666693 (-2.590686) | -0.760758 (-1.427185) | -0.791477 (-1.638537) | -0.809004 (-1.515712) | -0.821824 (-1.481250) |
| 5 ms, tree | -0.666693 (-2.590685) | -0.760758 (-1.427203) | -0.791477 (-1.638526) | -0.809003 (-1.515716) | -0.821824 (-1.481252) |
| 1 ms, dense | -0.293238 (-2.875531) | -0.466581 (-2.019397) | -0.523698 (-1.050077) | -0.606325 (-0.942676) | -0.637821 (-1.138507) |
| 1 ms, tree | -0.293239 (-2.875511) | -0.485752 (-3.236609) | -0.556492 (-0.986125) | -0.549621 (-1.284982) | -0.528168 (-1.952033) |

The dense run at 1 ms reproduces D38's figure (-11.4 J of 17.83 at 100 s).

## 5. The tests

`test_kernel.exe "Energy on the oscillator: every method keeps its own
recurrence's over many steps" -s`: every method's energy within 9.3e-14 of
its prediction (bound 1e-10, relative); semi-implicit Euler's invariant
changed by 1.2e-15 (bound 1e-13), its energy from 0.80003 to 1.33332 of the
start (bounds 0.8 and 1.3333).

`test_kernel.exe "Linearly implicit Euler: two links on stiff bushings swing
stably at long steps" -s`: the tip at most 2.0000000006, 2.0000052 and
2.0000032 m from the hinge at 2, 1 and 0.5 ms (bound 2.001); at 0.5 ms the
energy at most -2.07e-5 J (bound 0).

Before the second test was written, `scripts\build.ps1 -Test` with D40 in
place: 470 tests, no compiler warning, the vehicle tests of the linearly
implicit step unchanged.
