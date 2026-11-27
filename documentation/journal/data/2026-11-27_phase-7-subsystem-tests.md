# Data: Phase 7, subsystem tests (task 7.12)

The author's laptop (Windows 11, MSVC, RelWithDebInfo through
`scripts\build.ps1`, one compiler), on mains power
(Win32_Battery.BatteryStatus 2 before and after the suite): the suite, 660
tests, all passing, 69.4 s.

## The tests' values

Command: `build\tests\test_vehicle.exe "[subsystems]" -s`; 27 assertions in
4 test cases, all passing. The printed values (lines joined where the
console cut them):

```
double wishbone: instant centre (y, z) -6.94632  1.09474 m; camber rate
  0.129428 rad/m, predicted 0.129428 (difference -1.94289e-16); wheel centre
  sideways 0.141689, predicted 0.141689 (difference 2.77556e-17)
McPherson: instant centre (y, z) -0.0469767  0.0178295 m; camber rate
  1.23919 rad/m, predicted 1.23919 (difference 8.88178e-16); wheel centre
  sideways 0.0220941, predicted 0.0220941 (difference -8.67362e-17)
double wishbone: turn  0.129428 0.0904785 -0.904785 rad/m, wheel centre
  0.0814307  0.141689         1; five-link: differences 1.94289e-16
  5.09315e-15 1.44329e-15 and -4.71845e-16 -1.11022e-16            0
held: moved 0.000157415 m; shafts 368, 368 N m (by hand 368); brakes -368,
  -368 N m; tyres' moments -0.000118931, -0.000118931 N m
released: moved 0.0941992 m; brakes -294.4, -294.4 N m (by hand 294.4)
```

The brakes' case's comparisons, as Catch2 expands them:

```
368.00000024401435894 and 368.0 are within 0.0001% of each other   (each shaft)
367.99988107689290473 and 368.0 are within 0.1% of each other      (each brake, held)
294.40000030340172543 and 294.40000000000003411 are within 0.0001% of each other   (each brake, released)
```

## The first run of the brakes' case

With a hold of 1.5 s after the throttle's ramp, the case failed on the
shafts' torque:

```
CHECK_THAT( pt.shaft_torque[2], WithinRel(shaft, 1e-6) )
with expansion:
  367.9874226002323212 and 368.0 are within 0.0001% of each other
with message:
  held: moved 0.000155017 m; shafts 367.987, 367.987 N m (by hand 368); brakes
  -367.952, -367.952 N m; tyres' moments -0.0356959, -0.0356959 N m
```

The driveline's ringing, by hand: the carrier with the inertia reflected
through the ratio, (0.05 + 0.02) x 8^2 + 0.05 + 2 x 0.005 = 4.54 kg m^2, on
two shafts of 1e4 N m/rad and 20 N m s/rad: 66.4 rad/s, decay rate 40 /
(2 x 4.54) = 4.4 /s, exp(-6.6) = 1.4e-3 over 1.5 s. A ramp of 0.5 s (5.3
periods) leaves about 17 N m of ringing, undamped estimate: about 0.02 N m
after 1.5 s, against 0.013 seen. With a hold of 4 s, exp(-17.6).

## The suite

Command: `scripts\build.ps1 -Test`, its summary:

```
100% tests passed, 0 tests failed out of 660
Total Test time (real) =  69.36 sec
```

CI on the previous commit (task 7.11, `f20e6f7`), run 45966571409: every
job passed (Windows, Linux with sanitizers, no-malloc, docs, the comparison,
both Python packages).
