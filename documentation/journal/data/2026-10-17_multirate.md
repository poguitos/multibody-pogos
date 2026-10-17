# Multirate RK4: measurements (task 4.6)

Machine: the development laptop (i7-13700H, 16 GB), Windows 11, MSVC
RelWithDebInfo, one compiler. Wall times unpinned (a comparison within one
run, not a benchmark).

## 1. The paper

Haug, Negrut, Serban and Solis, "Numerical Methods for High Speed Vehicle
Dynamic Simulation" (1999), `documentation/Selección/Selección 2/`,
section 6: equations (32) to (34), the slowest-first method of Wells (1982),
the note that extrapolating the fast variables into the slow step "is
subject to numerical errors", the cost model (35) to (40), and Solis's
result: "very similar results were obtained with a ratio of 6 between slow
and fast integration step lengths" on the HMMWV-14. Section 7 runs the
10-body HMMWV in real time with a constant-step third-order Adams-Bashforth
method and dual-rate wheel rotation, at 5 m/s.

## 2. Three couplings, on the vehicles

A temporary hidden test case in `tests/test_drivetrain.cpp` (removed after
use), with a temporary switch for the coupling. Single rate is plain RK4.

**Braking at half pedal from 20 m/s** (the existing braking test's set-up,
deceleration over 1 s after 0.5 s to settle):

| run | fastest first | slowest first, held | slowest first, trend |
|---|---|---|---|
| single rate, 1 ms | 5.378671862 | (same) | (same) |
| single rate, 6 ms | 5.389398150 | (same) | (same) |
| multirate, 6 ms x 6 | 5.389402191 | 5.262499263 | 5.389402261 |
| multirate, 3 ms x 3 | 5.373309885 | 5.309215381 | 5.373309862 |

1200 evaluations and 7200 state evaluations for 6 ms x 6, against 7200
evaluations at 1 ms. Braking at 20 m/s is not stiff in the spins: multirate
at 6 ms equals single rate at 6 ms to 1e-6; the 0.2 % to 1 ms is the
bodies' step.

**The simple vehicle rolling at 0.5 m/s** (speed after 0.3 s and 2 s more;
worst |spin x rolling radius - speed| / speed):

| run | fastest first | held | trend |
|---|---|---|---|
| single rate, 0.5 ms | 0.500212 -> 0.500209, 3.736e-5 | | |
| multirate, 3 ms x 6 | 0.500373 -> 0.500367, 6.445e-5 | 0.500527 -> 0.500511, 9.488e-5 | 0.500342 -> 0.500337, 5.727e-5 |
| single rate, 3 ms | 0.494055 -> 0.493476, 4.761e-1 | | |
| multirate, 6 ms x 6 | 0.665205 -> 1.036258, 1.354e-2 | 0.614035 -> 1.030755, 1.552e-2 | 0.664162 -> 1.035170, 1.371e-2 |

18400 evaluations at 0.5 ms single rate; 3068 evaluations and 18408 state
evaluations at 3 ms x 6. At 6 ms the car gains speed in every coupling: the
hubs' side of the tyres' slip, explicit in the bodies' step, is unstable
there.

**The sedan through a drive** (double-wishbone, drivetrain; half throttle
1 s, brake 0.6 to a stop for 2 s, stand 1 s; top speed and speed at the
end; wall time of the whole run):

| run | top speed | at the end | evaluations | state evaluations | wall |
|---|---|---|---|---|---|
| single, 0.5 ms | 1.46849 | -9.59e-7 | 32000 | 0 | 3.9 s |
| single, 1 ms | 1.46854 | -4.00e-4 | 16000 | 0 | |
| single, 1.5 ms | 1.47028 | -9.14e-4 | 10668 | 0 | |
| single, 2 ms | 1.47056 | 2.90e-4 | 8000 | 0 | |
| single, 3 ms | 1.47953 | 5.85e-3 | 5332 | 0 | 0.7 s |
| fastest, 1.5 ms x 3 | 1.46907 | -9.56e-7 | 10668 | 32004 | 2.2 s |
| fastest, 3 ms x 6 | 1.47058 | -9.53e-7 | 5332 | 31992 | 1.4 s |
| fastest, 2 ms x 2 | 0.97021 | 5.13e-1 | 8000 | 16000 | |
| fastest, 3 ms x 3 | 1.30588 | 2.16e-1 | 5332 | 15996 | |
| fastest, 6 ms x 6 | 0.96021 | 5.01e-1 | 2668 | 16008 | |
| trend, 1.5 ms x 3 | 1.47108 | -9.54e-7 | 10668 | 32004 | 2.2 s |
| trend, 3 ms x 6 | 1.47449 | -9.50e-7 | 5332 | 31992 | 1.4 s |
| trend, 3 ms x 3 | 1.67870 | 5.23e-1 | 5332 | 15996 | |
| trend, 6 ms x 6 | 1.67910 | 5.22e-1 | 2668 | 16008 | |

Every run whose fast step is 1 ms fails (2 ms x 2, 3 ms x 3, 6 ms x 6),
every run whose fast step is 0.5 ms works: the micro-steps' length decides,
not the macro step's (single rate is fine at 2 ms). The sedan's front spins
at walking pace chatter at 1 ms; single rate averages the chatter, the
multirate bodies sample it at two instants per step and drift.

## 3. The tests

`scripts\build.ps1 -Test`: 457 tests, no compiler warning (one test failed
on its first run: its steps of 0.06 and 0.03 s did not divide 1 s, so the
runs ended at 1.02 and 0.99 s and the "error" was the motion of 0.02 s;
steps of 0.05 and 0.025 s since). Then `test_kernel.exe "[multirate]" -s`
and `test_vehicle.exe "[multirate]" -s`:

- **A lag driving a mass** (`tests/kernel/test_kernel_states.cpp`), RK4 with
  multirate 6: errors 6.52645e-7 m at H = 0.05 s and 4.08558e-8 at 0.025 s,
  ratio 15.9744; 20 steps, 80 evaluations, 23 state evaluations per step
  (four per fast step, the first given by the step's first evaluation).
- **The rolling car**: single rate at 0.5 ms 0.500209 m/s, wheels within
  3.73574e-5 (18400 evaluations); multirate 3 ms x 6 0.500367 m/s, within
  6.44522e-5 (3068 evaluations, 17641 state evaluations); single rate at 3
  ms within 0.476131.
- **The sedan's drive**: single rate at 0.5 ms top 1.46849 m/s, end
  -9.59e-7 (32000 evaluations); multirate 3 ms x 6 top 1.47058, end -9.53e-7
  (5332 evaluations).
- No allocation while stepping the sedan with multirate 6; K046 for a
  setting of 0, and of 6 with generalised-alpha.
