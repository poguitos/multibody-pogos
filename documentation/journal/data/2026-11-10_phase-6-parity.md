# Data: Phase 6, parity tests (task 6.3)

Raw measurements behind the journal entry `2026-11-10_phase-6-parity.md`.
The laptop (i7-13700H, 16 GB, Windows 11), on battery throughout
(`Win32_Battery.BatteryStatus` 1, 61 % falling to 59 %); the suite's build
(`build\`, RelWithDebInfo, MSVC, Ninja, one compiler), its Python the conda
environment `mbd-python` (Python 3.12).

## The ten, and what each compares

`parity_counts.py` (scratch): each test of `python/tests/test_parity.py` run
with the comparison wrapped to count what it compared; every number
identical.

| Test (C++ case) | Executable | Names | Numbers |
|---|---|---|---|
| Assembly: a four-bar given inconsistent values is assembled and reported | `test_kernel` | 17 | 27 |
| Force library: a limit stop holds a falling slider at m g / k past the limit | `test_forces` | 3 | 3 |
| Linearisation: a four-bar on a spring oscillates at sqrt(V'' / M) along its motion | `test_kernel` | 12 | 25 |
| Events: a bouncing ball's impact times match the closed form | `test_kernel` | 6 | 21 |
| State elements: a lag pushing a mass converges at the integrator's order | `test_kernel` | 4 | 16 |
| Kernel simulator: a constrained pendulum keeps its energy and its length | `test_kernel` | 12 | 32 |
| Outputs: a pendulum's hinge reaction is m a - m g, with no moment | `test_kernel` | 4 | 54 |
| Contact: a block on an incline creeps below the friction angle and slides above it | `test_forces` | 4 | 205 |
| Outputs: energy is conserved with springs and preloads | `test_kernel` | 7 | 9 |
| Radau IIA beats generalised-alpha on the bushed pendulum at tight tolerance | `test_kernel` | 7 | 124 |
| In all | | | 516 |

The first run of the ten found every number identical; nothing in a
reproduction was changed afterwards to make it so.

## Each comparison fails when the reproduction is off

`parity_mutations.py` (scratch) patches the reproductions in memory:

| Change | Test | Result |
|---|---|---|
| none | assembly | passes |
| the crank angle one double above 1.0 rad in the closed-form configuration | assembly | fails: 13 of 27 numbers differ (q[0] 1 ulp, q[1] 2 ulps, v[1] 1 ulp first) |
| the lag's time constant one double above 0.2 s | lag | fails: 11 of 16 differ (2 to 3 ulps first) |
| `v` recorded as `speed` | limit stop | fails: "names only from Python: ['speed']; only from C++: ['v']" |
| the hinge's frames from the rotation matrix instead of the axis and angle | hinge reaction | fails: 48 of 54 differ |

The two frames' quaternions [w, x, y, z], as hexadecimal:
`from_axis_angle([1, 0, 0], -pi/2)`: `0x1.6a09e667f3bcdp-1`,
`-0x1.6a09e667f3bcdp-1`, `-0x0.0p+0`, `-0x0.0p+0`; from the matrix
`[[1, 0, 0], [0, c, -s], [0, s, c]]`: `0x1.6a09e667f3bcdp-1`,
`-0x1.6a09e667f3bccp-1`, `0x0.0p+0`, `0x0.0p+0`: one ulp apart in x, and
the zeros of the other sign. Of the 48 numbers that moved, the median by
1024 ulps (`frame_mutation_spread.py`; its largest figure, 9.5e18, counts a
change of sign as ulps and means nothing).

## Times

| Run | `Python: test_parity.py` | Suite (534 tests) |
|---|---|---|
| `scripts\build.ps1 -Test`, the build of the change and the suite (527 s in all) | 218.39 s | 287.77 s |
| `ctest -R "Python: test_parity"`, alone | 8.06 s | |
| `pytest --durations=0 test_parity.py`, by hand | 14.82 s (the bushed pendulum 12.52 s; the others 0.07 to 0.70 s) | |
| `scripts\build.ps1 -Test`, after the change for MBD-P011 | 14.41 s | 73.89 s |

In the first suite run the other six Python files took their usual 0.7 to
2.5 s and the C++ case of the bushed pendulum 3.81 s (2.27 s in the run of
the morning); the cause of the 218 s was not found.

The bushed pendulum's four runs (`time_alternate.py`), alternated with its
C++ case, which makes the same runs:

| Round | C++ case | The same runs from Python |
|---|---|---|
| 1 | 3.40 s | 3.53 s |
| 2 | 3.14 s | 3.28 s |

An earlier, single timing (`time_bushed.py`) had Dormand-Prince alone at
6.69 s from Python (138701 steps, 970985 evaluations) against the whole C++
case at 2.8 s; alternated, the two take the same time.

## The build stopped by a line on stderr

`& scripts\build.ps1 -Test *> build.log` stopped before compiling:

```
cmake.exe : -- GLOB mismatch!
En C:\dev\multibody\scripts\build.ps1: 91 Carácter: 5
+     & $Exe @Arguments
    + FullyQualifiedErrorId : NativeCommandError
```

The same command started as `cmd /c "powershell -NoProfile
-ExecutionPolicy Bypass -File scripts\build.ps1 -Test > build.log 2>&1"`:
built and tested, exit 0.

## The author's environment

Before the reinstall, in `mbd-try`: `import multibody` raised
`AttributeError: module 'multibody._mbd' has no attribute 'ProjectionInfo'`
(the Python files from the repository, the module from the install of the
morning). After the change for MBD-P011 and `scripts\build.ps1 -Install
-Python %USERPROFILE%\miniforge3\envs\mbd-try\python.exe` (62 s:
`bind_simulator.cpp` compiled, the module linked): the import works;
`python -m pytest python/tests`: 44 passed, 10 skipped (the parity tests,
without the suite's build).
