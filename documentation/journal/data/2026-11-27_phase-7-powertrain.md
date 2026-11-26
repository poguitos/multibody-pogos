# Data: Phase 7, the powertrain (task 7.8)

The author's laptop (Windows 11, MSVC, RelWithDebInfo through
`scripts\build.ps1`, one compiler), on mains power
(Win32_Battery.BatteryStatus 2): the suite, 644 tests, all passing, 66.5 s.

Command: `build\tests\test_vehicle.exe "[powertrain]" -s`, the INFO lines:

```
  inboard: anti-squat 40 % from the multipliers, 40 % from the geometry

  outboard: anti-squat 640 % from the multipliers, 640 % from the geometry
  engine 207.846 rad/s; shafts 3381, 3381 N m
  twist 0.3381 rad, Bouc-Wen z 0 rad
  twist 0.330241 rad, Bouc-Wen z 0.0196475 rad
  open: wheels accelerate at 141.219 and 70.8441 rad/s^2, the rigid limit 141.
  219 and 70.8441
  locked: wheels accelerate at 101.657 and 101.657 rad/s^2, the rigid limit
  101.657 and 101.657
  limited slip: the wheels' speeds differ by 4.57459 rad/s, the steady state 4.
  57459
  limited slip: wheels accelerate at 101.657 and 101.657 rad/s^2, the rigid
  limit 101.657 and 101.657
  engine after 0.5 s: 398.405 rad/s, closed form 398.405
  shift began at 2.9985 s, second gear at 3.1485 s; least gearbox torque in the
  shift 5.08913e-10 N m
  after 2 s: 7.04543 m/s, engine 586.454 rad/s in gear 4; shafts 311.13 and
  311.13 N m
```
