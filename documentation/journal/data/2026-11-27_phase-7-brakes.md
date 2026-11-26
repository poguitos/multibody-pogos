# Data: Phase 7, the brakes (task 7.9)

The author's laptop (Windows 11, MSVC, RelWithDebInfo through
`scripts\build.ps1`, one compiler), on mains power
(Win32_Battery.BatteryStatus 2): the suite, 648 tests, all passing, 67.2 s.

Command: `build\tests\test_vehicle.exe "[brakes]" -s`, the INFO lines
(each case prints its line once per check; one copy kept):

```
  stopped in 31.6219 m and 3.109 s; by hand 31.5954 m and 3.11036 s
  deceleration 2.94202 m/s^2, mu g 2.943
  without the controller 100.504 m (17880 of 20113 steps with a wheel locked
  above 2 m/s); with it 85.2558 m (0)
```

Relative differences: the distance (31.6219 - 31.5954) / 31.5954 = 8.4e-4,
the time (3.11036 - 3.109) / 3.11036 = 4.4e-4 (allowed 3e-3 each); the
slide (2.943 - 2.94202) / 2.943 = 3.3e-4 (allowed 1e-3).

The hand calculation of the stopping distance, with the test's numbers:
R = 0.3135 m (the linear tyre's free radius), m = 1560 kg (body 1400, four
wheels of 40), I = 0.5 * 40 * R^2 per wheel, so 4 I / R^2 = 80 kg; at 6 MPa
the brakes give 2 * (1.9e-4 + 0.9e-4) * 6e6 = 3360 N m in all, and
d_inf = 3360 / R / 1640 = 6.535 m/s^2; the stop comes near v0 / d_inf + tau
= 3.110 s.
