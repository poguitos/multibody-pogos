# Data: timings taken again on mains power (24 November 2026)

The timings of task 6.3 (parity) were taken with the laptop in
energy-saving mode, and the suites of the Phase 6 gate and of tasks 7.1 and
7.2 on battery; the author asked for them again on mains power. Machine: the
development laptop (i7-13700H, 16 GB, Windows 11, MSVC, one compiler), on
mains power (`Win32_Battery.BatteryStatus` 2, before and after); the suite's
build at `7baaf3b` (563 tests), its Python `mbd-python` (3.12).

## The suite

`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1 -Test`
(the build current; tests only), from cmd with output to a file:

| Run | Tests | Total | Power |
|---|---|---|---|
| Phase 6 gate, 13 Nov | 544 | 297.76 s | battery |
| Task 7.1, 15 Nov | 546 | 121.48 s | battery |
| Task 7.2, 24 Nov | 563 | 101.99 s | battery |
| This run, 24 Nov | 563 | 54.02 s | mains |

The slowest in this run: `Python: test_parity.py` 5.78 s,
`Python: test_notebooks.py` 4.90 s (18.04 s at the gate, on battery),
`Python: test_howto.py` 2.81 s, `Python: test_viewer.py` 2.62 s, the C++
bushed pendulum's Radau IIA case 2.22 s, `Python: test_files.py` 2.21 s,
the two HMMWV bushing cases 1.78 and 1.39 s.

## Task 6.3's table, again

| Run | `Python: test_parity.py`, 6.3 (energy saving) | Now (mains) |
|---|---|---|
| In the suite (`scripts\build.ps1 -Test`) | 218.39 s, then 14.41 s | 5.78 s |
| `ctest -R "Python: test_parity"`, alone | 8.06 s | 5.91 s |
| `pytest --durations=0 test_parity.py`, by hand | 14.82 s (the bushed pendulum 12.52 s) | 5.08 s (the bushed pendulum 4.41 s; the others 0.03 to 0.17 s) |

The bushed pendulum's runs alternated with its C++ case (a scratch script,
`time_alternate.py`, rebuilt for this; the C++ case as a process, then the
same four runs from Python):

| Round | C++ case, 6.3 | From Python, 6.3 | C++ case, now | From Python, now |
|---|---|---|---|---|
| 1 | 3.40 s | 3.53 s | 2.36 s | 2.29 s |
| 2 | 3.14 s | 3.28 s | 2.35 s | 2.26 s |

As in 6.3, the two take the same time: the interface adds nothing to these
runs. The 218 s of 6.3's first suite run does not recur.
