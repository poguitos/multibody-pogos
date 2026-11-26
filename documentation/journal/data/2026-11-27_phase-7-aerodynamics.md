# Data: Phase 7, the aerodynamic element (task 7.10)

The author's laptop (Windows 11, MSVC, RelWithDebInfo through
`scripts\build.ps1`, one compiler), on mains power
(Win32_Battery.BatteryStatus 2): the suite, 653
tests, all passing, 67.3 s.

## The synthetic map

Command: `build\tests\test_forces.exe "[aero]" -s`, the INFO lines (one
copy of each kept):

```
  road at 0 m: variables 0.07, 0.113298, 0.012, -0.017, 0.0797915; force -303.
  814 -383.265 -809.328 N; error 3.41061e-13 N, 5.95543e-14 N m
  road at 0.5 m: variables 0.07, 0.113298, 0.012, -0.017, 0.0797915; force -
  303.814 -383.265 -809.328 N; error 4.58286e-13 N, 6.07088e-14 N m
```

The force's magnitude: sqrt(303.814^2 + 383.265^2 + 809.328^2) = 946 N.

## The regression scenario sports-car-turn

Command, after building everything: `build\tests\test_regression.exe`
(before writing the reference again), the failing scenario's message:

```
  sports-car-turn: The double-wishbone sports car with anti-roll bars and
  downforce, accelerating, then turning left; RK4 at 1 ms for 2 s.
    chassis x [m]: 2.74e-05 (tolerance 3.88e-08, 705 of it)
    chassis y [m]: 5.26e-06 (tolerance 1.74e-09, 3.03e+03 of it)
    chassis z [m]: 4.65e-07 (tolerance 2.97e-09, 157 of it)
    chassis vx [m/s]: 7.62e-05 (tolerance 4.82e-08, 1.58e+03 of it)
    chassis vy [m/s]: 1.76e-05 (tolerance 4e-09, 4.42e+03 of it)
    chassis vz [m/s]: 9.71e-07 (tolerance 6.13e-11, 1.58e+04 of it)
    chassis wx [rad/s]: 2.28e-06 (tolerance 3.09e-10, 7.39e+03 of it)
    chassis wy [rad/s]: 1.39e-07 (tolerance 6.17e-10, 226 of it)
    chassis wz [rad/s]: 2.63e-06 (tolerance 8.88e-10, 2.96e+03 of it)
    FL Fz [N]: 0.0471 (tolerance 3.76e-05, 1.25e+03 of it)
    FL Fy [N]: 0.0748 (tolerance 7.18e-06, 1.04e+04 of it)
    FL toe [rad]: 4.38e-07 (tolerance 1.17e-09, 373 of it)
    FL camber [rad]: 3.48e-07 (tolerance 1.48e-11, 2.35e+04 of it)
    FR Fz [N]: 0.0178 (tolerance 3.76e-05, 473 of it)
    FR Fy [N]: 0.0532 (tolerance 3.34e-06, 1.59e+04 of it)
    FR toe [rad]: 1.26e-06 (tolerance 1.04e-09, 1.21e+03 of it)
    FR camber [rad]: 3.96e-07 (tolerance 2.39e-11, 1.66e+04 of it)
    RL Fz [N]: 0.0332 (tolerance 3.67e-05, 904 of it)
    RL Fy [N]: 0.0485 (tolerance 2.76e-06, 1.76e+04 of it)
    RL toe [rad]: 5.67e-07 (tolerance 4.94e-10, 1.15e+03 of it)
    RL camber [rad]: 3.13e-07 (tolerance 3.35e-11, 9.33e+03 of it)
    RR Fz [N]: 0.0131 (tolerance 3.7e-05, 353 of it)
    RR Fy [N]: 0.0346 (tolerance 4e-06, 8.67e+03 of it)
    RR toe [rad]: 1.11e-06 (tolerance 6.96e-10, 1.59e+03 of it)
    RR camber [rad]: 3.83e-07 (tolerance 1.79e-11, 2.14e+04 of it)
    state 0: 1.33e-08 (tolerance 2.84e-12, 4.68e+03 of it)
    state 1: 7.8e-07 (tolerance 6.57e-11, 1.19e+04 of it)
    state 2: 1.01e-08 (tolerance 3.03e-12, 3.33e+03 of it)
    state 3: 5.42e-07 (tolerance 2.65e-11, 2.04e+04 of it)
    state 4: 1.03e-07 (tolerance 9.97e-11, 1.03e+03 of it)
    state 5: 5.08e-07 (tolerance 4.87e-11, 1.04e+04 of it)
    state 6: 1.31e-07 (tolerance 9.86e-11, 1.33e+03 of it)
    state 7: 3.47e-07 (tolerance 6.48e-11, 5.36e+03 of it)
```

The old and new references compared by phase (a Python script in the
session's scratch folder, reading both CSV files; each difference over the
channel's largest magnitude in the old reference; the four largest):

```
at rest       FR camber 1.09e-15, FL camber 6.26e-16, RL camber 5.45e-16, chassis wx 4.00e-16
accelerating  chassis vz 2.21e-05, state 3 5.15e-06, FR Fy 4.22e-06, state 0 3.29e-06
turning       FL camber 2.35e-04, RR camber 2.14e-04, state 3 2.04e-04, RL Fy 1.76e-04
```

The forward speed, old and new:

```
t=0.3 s  old 0.000001385  new 0.000001385  diff -5.42e-19
t=0.6 s  old 0.791697876  new 0.791698206  diff  3.30e-07
t=1.0 s  old 1.842825611  new 1.842829345  diff  3.73e-06
t=1.5 s  old 3.185549525  new 3.185569847  diff  2.03e-05
t=2.0 s  old 4.821533660  new 4.821609846  diff  7.62e-05
```

Writing the references again (`MBD_REGRESSION_UPDATE=1`, the fast
scenarios) also rewrote `wishbone-sweep`, by at most 1.3e-15; that file was
restored.

The sensitivity probe after the rewrite (`build\tests\test_regression.exe
"[sensitivity]" -s`), the sports car's line:

```
sports-car-turn      gravity                                    0.0108     1.08e+04  chassis vz [m/s]
```

as in the regression README's table (1.1e-2 and 10800).
