# Data: the simple tyres (task 7.5, 26 November 2026)

Machine: the development laptop (i7-13700H, 16 GB, Windows 11, MSVC, one
compiler), on battery power (`Win32_Battery.BatteryStatus` 1, 68 % charge)
for both the suite and the benchmark. Builds through `scripts\build.ps1`.

## The benchmark's tyre table

`build\bench\mbd_bench.exe` started from PowerShell, its process set to
`ProcessorAffinity` 4 (logical processor 2, a performance core) and
`PriorityClass` High; the whole program ran, this is its last table. One
evaluation at kappa = -0.05, alpha = 0.04 rad, gamma = 0.01 rad, Fz = 4000 N,
Vcx = 20 m/s (median of seven rounds of about 20 ms); one RK4 step of 1 ms of
the double-wishbone sedan on four of the tyre, after 0.2 s at 30 % throttle.

```
| Tyre | one evaluation [us] | the sedan's RK4 step of 1 ms [us] |
|---|---|---|
| Magic Formula | 1.345 | 325.4 |
| linear | 0.034 | 289.1 |
| Fiala | 0.100 | 290.6 |
| brush | 0.067 | 289.8 |
```

## The car held on its brakes on a 10 % slope, each simple tyre

`test_vehicle "Drivetrain: a car on a simple tyre\, held on its brakes on a
slope\, stays put too" -s -r xml`; by hand 3.37208 mm (the carcass's
358066 N/m and the pitch):

```
linear: rolled back 0.003488 m, means -0.003488 and -0.003488 m, speed at 10 s -5.66e-15 m/s
Fiala: rolled back 0.00348682 m, means -0.00348682 and -0.00348682 m, speed at 10 s -5.75e-15 m/s
brush: rolled back 0.00355235 m, means -0.00355235 and -0.00355235 m, speed at 10 s -3.44e-15 m/s
```

## The suite

`scripts\build.ps1 -Test`, on battery: 604 tests pass in 129.0 s (592 before
this task; about 70 s on mains).
