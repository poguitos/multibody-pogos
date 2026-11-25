# Data: the wheel and its contact (task 7.3, 24 and 25 November 2026)

Machine: the development laptop (i7-13700H, 16 GB, Windows 11, MSVC, one
compiler), on mains power (`Win32_Battery.BatteryStatus` 2). Builds through
`scripts\build.ps1`; tests run from `build\tests`.

## The regression scenarios, before and after

The references at `5ee9e02` (before) against those written by this task
(`MBD_REGRESSION_UPDATE=1`, `test_regression.exe "[regression]~[sensitivity]"`
and `"[long]"`), compared on the channels they share by a scratch script
(`compare_refs.py`): each channel's largest difference over its largest
magnitude in the old reference, and the last row. The four `state k`
channels were the drivetrain's spins, which are no longer states.
`wishbone-sweep` was rewritten by the same run with differences of at most
5.5e-14 of its channels' scale (rounding: its builder's file was
recompiled); its old reference was restored, and it passes.

```
## sedan-drive: 47 rows before, 47 now; channels 14 before, 10 now
  gone: state 0, state 1, state 2, state 3
  largest changes (difference / channel's largest magnitude):
    chassis vy [m/s]          2.668e-01  (0.02843 of 0.1065)
    chassis y [m]             2.293e-01  (0.01254 of 0.0547)
    chassis wy [rad/s]        1.831e-01  (0.009518 of 0.05198)
    chassis vx [m/s]          1.606e-01  (0.4617 of 2.875)
    chassis wx [rad/s]        1.599e-01  (0.003037 of 0.01899)
    chassis wz [rad/s]        1.587e-01  (0.005028 of 0.03168)
    chassis x [m]             1.554e-01  (0.4464 of 2.873)
    chassis vz [m/s]          3.405e-03  (0.0003015 of 0.08856)
  at t = 2.3:
    chassis x [m]            before      2.87341  now      3.31986  (+15.54 %)
    chassis z [m]            before     0.533826  now     0.533888  (+0.01 %)
    chassis vx [m/s]         before      2.87462  now       3.3363  (+16.06 %)
    chassis vy [m/s]         before     0.106541  now     0.134969  (+26.68 %)
    chassis wz [rad/s]       before    0.0316835  now    0.0367117  (+15.87 %)

## sedan-slalom: 200 rows before, 200 now; channels 14 before, 10 now
  gone: state 0, state 1, state 2, state 3
  largest changes (difference / channel's largest magnitude):
    chassis wx [rad/s]        2.359e-01  (0.07394 of 0.3134)
    chassis vy [m/s]          2.336e-01  (0.402 of 1.721)
    chassis y [m]             2.328e-01  (0.1434 of 0.6159)
    chassis x [m]             1.892e-01  (38.42 of 203)
    chassis wy [rad/s]        1.713e-01  (0.01353 of 0.07898)
    chassis wz [rad/s]        1.485e-01  (0.04311 of 0.2902)
    chassis vx [m/s]          1.461e-01  (3.313 of 22.67)
    chassis vz [m/s]          1.070e-01  (0.009331 of 0.08724)
  at t = 20:
    chassis x [m]            before      203.002  now      241.418  (+18.92 %)
    chassis z [m]            before       0.5411  now     0.542522  (+0.26 %)
    chassis vx [m/s]         before      22.6745  now      25.5875  (+12.85 %)
    chassis vy [m/s]         before    -0.957784  now    -0.864221  (-9.77 %)
    chassis wz [rad/s]       before    -0.163204  now    -0.175526  (+7.55 %)

## sports-car-turn: 40 rows before, 40 now; channels 30 before, 26 now
  gone: state 0, state 1, state 2, state 3
  largest changes (difference / channel's largest magnitude):
    chassis wx [rad/s]        3.374e-01  (0.006433 of 0.01907)
    RL Fy [N]                 2.866e-01  (372.9 of 1301)
    RR Fy [N]                 2.866e-01  (372.9 of 1301)
    FL camber [rad]           2.815e-01  (0.0003789 of 0.001346)
    chassis wy [rad/s]        2.768e-01  (0.01236 of 0.04466)
    RL camber [rad]           2.540e-01  (0.0003762 of 0.001481)
    FR Fy [N]                 2.236e-01  (205.5 of 918.9)
    FL Fy [N]                 1.954e-01  (205.5 of 1052)
  at t = 2:
    chassis x [m]            before      3.21024  now      3.43856  (+7.11 %)
    chassis z [m]            before      0.35385  now     0.351572  (-0.64 %)
    chassis vx [m/s]         before      3.82843  now      4.13599  (+8.03 %)
    chassis vy [m/s]         before     0.286619  now       0.3208  (+11.93 %)
    chassis wz [rad/s]       before    0.0716638  now     0.077185  (+7.70 %)
    FL Fz [N]                before      3132.14  now      3410.79  (+8.90 %)
    FL Fy [N]                before      1051.83  now      1146.74  (+9.02 %)
    RL Fz [N]                before      3154.57  now      3537.44  (+12.14 %)

## hatchback-brake: 40 rows before, 40 now; channels 30 before, 26 now
  gone: state 0, state 1, state 2, state 3
  largest changes (difference / channel's largest magnitude):
    FR Fy [N]                 5.174e-01  (3697 of 7146)
    FL Fy [N]                 5.174e-01  (3697 of 7146)
    chassis wy [rad/s]        4.929e-01  (0.3787 of 0.7683)
    chassis vz [m/s]          4.927e-01  (0.1886 of 0.3828)
    FL toe [rad]              4.816e-01  (0.01817 of 0.03773)
    FL camber [rad]           4.079e-01  (0.04228 of 0.1036)
    FR camber [rad]           4.079e-01  (0.04228 of 0.1036)
    FL Fz [N]                 4.079e-01  (3448 of 8453)
  at t = 2:
    chassis x [m]            before     0.399341  now     0.445665  (+11.60 %)
    chassis z [m]            before     0.441189  now     0.441272  (+0.02 %)
    chassis vx [m/s]         before -1.33447e-05  now -0.000737802  (+5428.80 %)
    chassis vy [m/s]         before  2.89214e-05  now  2.69346e-05  (-6.87 %)
    chassis wz [rad/s]       before -6.37034e-06  now -6.38097e-06  (+0.17 %)
    FL Fz [N]                before      3541.09  now      3860.36  (+9.02 %)
    FL Fy [N]                before     -184.828  now     -89.0258  (-51.83 %)
    RL Fz [N]                before      2446.14  now      2521.75  (+3.09 %)

## sedan-rough-road: 40 rows before, 40 now; channels 30 before, 26 now
  gone: state 0, state 1, state 2, state 3
  largest changes (difference / channel's largest magnitude):
    RL Fz [N]                 3.124e-01  (1647 of 5272)
    FR Fz [N]                 2.646e-01  (1608 of 6077)
    chassis wx [rad/s]        2.502e-01  (0.03295 of 0.1317)
    FL Fz [N]                 2.467e-01  (1335 of 5414)
    RR Fz [N]                 2.195e-01  (1260 of 5742)
    chassis wy [rad/s]        2.033e-01  (0.03133 of 0.1541)
    RL Fy [N]                 1.456e-01  (187.3 of 1287)
    FL Fy [N]                 1.255e-01  (159.1 of 1268)
  at t = 2:
    chassis x [m]            before      31.8012  now      32.0498  (+0.78 %)
    chassis z [m]            before     0.532339  now     0.533093  (+0.14 %)
    chassis vx [m/s]         before      16.5264  now       16.789  (+1.59 %)
    chassis vy [m/s]         before      4.04166  now      4.11478  (+1.81 %)
    chassis wz [rad/s]       before     0.190082  now     0.189787  (-0.16 %)
    FL Fz [N]                before      2586.01  now      2083.03  (-19.45 %)
    FL Fy [N]                before         1215  now      1200.38  (-1.20 %)
    RL Fz [N]                before      4116.87  now      2812.95  (-31.67 %)

## sedan-modes: 9 rows before, 13 now; channels 3 before, 3 now
```

`sedan-modes` lists 13 rows instead of 9: the four wheels' spin modes at rest
(each a real, fast eigenvalue: the tyre's slip damping the spin) are new.

### Why sedan-drive moved: a one-dimensional estimate

A scratch script (`drive_1d.py`) integrates the scenario's straight-line
drive (rear-wheel drive, first gear, half throttle from 0.3 s to 2.3 s, from
rest; no slip, no load transfer) in the two forms. Before 7.3:
a = sum(T / R_eff) / (m + sum(I_eff / R_eff^2)), each driven wheel's I_eff
carrying half the engine's inertia (0.15 kg m^2) through the gears
(12.25); after: a = sum(T / r_l) / (m + sum(I / (R_eff r_l))), r_l = R - d the
arm of the force to the road.

```
d = 0.01912 m, R_eff = 0.34044 m, r_l = 0.33088 m
speed at 2.3 s, one-dimensional: before 2.8806 m/s, after 3.3208 m/s, ratio 1.1528
the scenario's references: before 2.87462 m/s, after 3.33630 m/s, ratio 1.16061
  of which the engine's inertia: 1.1181, the arm r_l for R_eff: 1.0289
```

The estimate gives both references within 0.5 %, and their ratio (1.153
against 1.161; the rest is the engine passing idle speed in the new run,
which raises its torque). The engine's inertia accounts for 11.8 %, the
force's arm for 2.9 %. The other driven scenarios gain speed for the same
reasons; the sports car and the hatchback also carry their wheels' mass now
(35 and 40 kg a corner on their linkage corners): the sports car's tyre loads
at rest rose by 337 to 339 N each (a 35 kg wheel weighs 343 N). On the rough
road the tyre loads agree to about 1 % over the first 0.3 s, while the two
cars are at the same place on the road, and differ more as the new one, 1.6 %
faster at 2 s, gets 0.25 m ahead.

## The gyroscope

`test_vehicle.exe "Wheel: a spinning wheel on a pivot precesses at m g l / (I_s omega_s)" -s`:
m = 10 kg, R = 0.3 m, l = 0.2 m, omega_s = 100 rad/s, Omega_p = 0.43585
rad/s, 3 s, RK4.

| Run | Precession error after 3 s [rad] | Axle elevation [rad] |
|---|---|---|
| 1 ms | 9.10e-15 | 3.33e-16 |
| 0.5 ms | 1.75e-14 | 3.33e-16 |
| 1 ms, spun the other way | -9.10e-15 | 3.33e-16 |

The spin at the end equal to 100 rad/s exactly. A hidden probe (removed
again) started the precession too fast: by 1e-6, the precession error -5.212e-9 rad
and the elevation 2.963e-9 rad after 3 s; by 1e-4, -5.212e-7 and 2.963e-7.

## The first lock, by forces only

`test_vehicle.exe` before the reset at the end of the step: "braking to a
stop" failed for every method. The slowest spin relative to the upright:
RK4 -0.00818, Dormand-Prince -0.00927, generalised-alpha -0.00818, SDIRK4
-0.00818, Radau IIA -0.1138 rad/s; at the end every locked wheel at
-1.74e-6 rad/s (and -1.35e-6), not zero. With the reset the test passes as
written; the car's speed at the end 2.3e-6 to 5.1e-6 m/s by method.

## Parked cars at RK4 1 ms

Before the settling tests moved to 0.5 ms: "Vehicle: settles to static
equilibrium from perturbed state" left a velocity of 0.0366 rad/s after 5 s
(allowed 0.02), "Template: DefaultSedan reaches static equilibrium"
0.0366; from the double-wishbone sedan's static equilibrium the chassis
moved 2.13e-5 m in 1 s (allowed 1e-6).

## The contact on a bump's flank

The road test's wheel (R = 0.35 m, centre at x = 1.125 m, z = 0.365 m) on
a bump of 0.04 m and 0.5 m starting at x = 1 m, its depth in the road by
three rules (a scratch computation with the bump's closed form; the nearest
point found by sampling x every 1e-6 m):

| Rule | Depth [m] | k depth [N] |
|---|---|---|
| The point below the centre (before task 7.3) | 0.004849 | 970 |
| The lowest point along the normal under the centre, the road asked there (task 7.3) | 0.011921 | 2384 |
| The circle's depth: R less the distance to the road's nearest point (x = 1.1854) | 0.013302 | 2660 |

## The regression sensitivity probe

`test_regression.exe "[sensitivity]" -s`, after the references were written
again (each scenario's largest channel change over its tolerance when gravity,
or the named parameter, changes by 1e-12 and by 1e-6):

```
scenario             perturbed                                   1e-12         1e-6  worst channel at 1e-12
four-bar             gravity                                  0.000693          693  rocker rate [rad/s]
heavy-top            gravity                                  0.000811          809  x [m]
bushed-pendulum      gravity                                  0.000168          177  link 2 x [m]
hmmwv-joints         gravity                                  5.95e-05         59.7  pitch [rad]
hmmwv-bushings       gravity                                  0.000238         59.7  pitch [rad]
sedan-drive          gravity                                  0.000991          996  chassis vz [m/s]
sedan-modes          gravity                                    0.0137         29.9  damping ratio
sports-car-turn      gravity                                  0.000472          471  FR Fy [N]
sedan-rough-road     gravity                                  0.000611          611  chassis vz [m/s]
hatchback-brake      gravity                                  0.000225          225  FL camber [rad]
wishbone-sweep       the upper ball joint's hardpoint         0.000684          682  camber [rad]
lap-oval             the friction                             4.16e-05         41.1  lap time [s]
hmmwv-joints-5s      gravity                                  6.56e-05         62.5  pitch [rad]
hmmwv-bushings-5s    gravity                                  7.17e-05         62.4  pitch [rad]
sedan-slalom         gravity                                  0.000995          996  chassis vz [m/s]
```

With the spin modes in the frequency channel (before they had one of their
own) `sedan-modes` gave 0.0137 and 8.47, and the probe failed (it asks for
more than 10 at 1e-6).

The modes of the sedan at rest (`sedan-modes`, natural frequency [rad/s] and
damping ratio before; now the same, and the spins' decay rates [1/s]):

| Mode | Before | Now |
|---|---|---|
| 0 to 2 | 7.663, 7.866, 12.075 | 7.812, 7.869, 12.289 |
| 3 to 6 | 71.72, 72.87, 73.72, 73.85 | 71.73, 72.86, 73.64, 73.86 |
| 7, 8 (real) | 187.96, 270.83 | 181.51, 259.22 |
| the spins (real) | - | 2758.6, 2758.6, 2852.7, 2919.8 |
