# Data: Phase 7, mass properties (task 7.11)

The author's laptop (Windows 11, MSVC, RelWithDebInfo through
`scripts\build.ps1`, one compiler), on mains power
(Win32_Battery.BatteryStatus 2 before and after the suite): the suite, 656
tests, all passing, 81.5 s.

## The tests' values

Command: `build\tests\test_vehicle.exe "[mass]" -s`; 46 assertions in 3
test cases, all passing. The two reports as printed:

```
Mass properties of mass_test (chassis frame: X forward, Y left, Z up)
  total mass                           1630.000 kg
  centre of mass                 x   -0.0799  y    0.0225  z    0.1671 m
  its height above the road              0.4615 m
  inertia about it, roll (Ixx)           667.39 kg m^2
                    pitch (Iyy)         2733.60 kg m^2
                    yaw (Izz)           3043.89 kg m^2
  products of inertia      Ixy   -13.024  Ixz  -100.205  Iyz    -2.040 kg m^2
  sprung mass                          1435.000 kg
  corner    unsprung      sprung       total   [kg]   contact x, y [m]
  FL          40.000     401.137     441.137            1.2002   0.8000
  FR          40.000     376.251     416.251            1.2002  -0.8000
  RL          57.500     339.885     397.385           -1.4998   0.7600
  RR          57.500     317.726     375.226           -1.4998  -0.7600
```

```
Mass properties of sports_car (chassis frame: X forward, Y left, Z up)
  total mass                           1441.000 kg
  centre of mass                 x   -0.0156  y   -0.0000  z    0.0000 m
  its height above the road              0.2966 m
  inertia about it, roll (Ixx)           382.65 kg m^2
                    pitch (Iyy)         1255.92 kg m^2
                    yaw (Izz)           1580.23 kg m^2
  products of inertia      Ixy    -0.000  Ixz     0.079  Iyz    -0.000 kg m^2
  sprung mass                          1221.873 kg
  corner    unsprung      sprung       total   [kg]   contact x, y [m]
  FL          54.782     327.987     382.769            1.2502   0.7800
  FR          54.782     327.987     382.769            1.2502  -0.7800
  RL          54.782     282.950     337.731           -1.4498   0.7800
  RR          54.782     282.950     337.731           -1.4498  -0.7800
  design loads per wheel over g: front 382.769, rear 337.731 kg
```

The differences between the first report and the inputs, as the test
prints them (one copy of each kept):

```
centre of mass: difference 4.16334e-17 m
inertia: largest difference 4.54747e-13 kg m^2 of 3043.89
corner 0: unsprung, difference 0 kg            (and corners 1 to 3: 0)
height of the centre of mass: difference 0 m; pitch 0.000733332 rad
corner 0: contact, difference 2.22045e-16 m    (and corners 1 to 3: the same)
corner 0: sprung, difference 5.68434e-14 kg
corner 1: sprung, difference 5.68434e-14 kg
corner 2: sprung, difference 0 kg
corner 3: sprung, difference -5.68434e-14 kg
corners: mass 0 kg, moments 5.96856e-13, 4.26326e-14 kg m
```

The double wishbone (the sports car's front corner):

```
unsprung: builder 54.7816 kg; from positions, difference 1.3617e-09 (delta
alone 3.27802e-07, 2 delta 1.30712e-06); a whole wheel and upright and half
the arms: 55
```

## The regression scenarios

Command: `build\tests\test_regression.exe -s`, each scenario's worst
channel (in the order of `tests/regression/test_regression.cpp`):

```
four-bar            crank [rad]: 0 (tolerance 3.65e-08, 0 of it)
heavy-top           x [m]: 0 (tolerance 9.59e-10, 0 of it)
bushed-pendulum     tip x [m]: 0 (tolerance 2e-08, 0 of it)
hmmwv-joints        chassis x [m]: 0 (tolerance 5.29e-08, 0 of it)
hmmwv-bushings      chassis x [m]: 0 (tolerance 5.29e-08, 0 of it)
sedan-drive         chassis x [m]: 0 (tolerance 3.59e-08, 0 of it)
sedan-modes         natural frequency [rad/s]: 0 (tolerance 1.67e-06, 0 of it)
sports-car-turn     chassis x [m]: 0 (tolerance 3.88e-08, 0 of it)
hatchback-brake     chassis wx [rad/s]: 1.2e-15 (tolerance 1.18e-10, 1.02e-05 of it)
sedan-rough-road    chassis x [m]: 0 (tolerance 3.12e-07, 0 of it)
wishbone-sweep      camber [rad]: 6.4e-16 (tolerance 1.35e-10, 4.74e-06 of it)
lap-oval            speed [m/s]: 0 (tolerance 3.72e-07, 0 of it)
```

(A zero worst channel means every channel is zero.) Of the long ones,
`build\tests\test_regression.exe "Regression: the sedan's slalom" -s`:
every channel 0, in 1.0 s. The HMMWV's 5 s scenarios were not run.

## The suite

Command: `scripts\build.ps1 -Test`, its summary:

```
100% tests passed, 0 tests failed out of 656
Total Test time (real) =  81.45 sec
```
