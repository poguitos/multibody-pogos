# Data: Phase 6, examples (task 6.6, D58)

Machine: the development laptop (i7-13700H, 16 GB, Windows 11), 12 November
2026; its power state was not recorded. Timings are wall-clock seconds from
`time.perf_counter()` in one process each, for scale only.

## The environment change

```
conda install -n mbd-python -c conda-forge --dry-run "matplotlib-base=3.11.2" "libblas=*=*openblas"
```

listed 30 new packages (matplotlib-base 3.11.2 with fonttools, kiwisolver,
contourpy, freetype, cairo and fonts) and nothing updated, downgraded or
removed. Installed the same way without `--dry-run`. `mbd-try` already had
matplotlib 3.11.2. Both environments had ipykernel 7.4.0, ipywidgets 8.1.9
and anywidget 0.11.0, which came with rerun-sdk 0.43.0 from conda-forge.
`rerun-notebook` (Rerun's viewer inside a notebook) is not on conda-forge:
`conda search -c conda-forge rerun-notebook` found nothing.

## Prototypes: timings

| Run | Time |
|---|---|
| Pendulum, 4 s, Dormand-Prince at 1e-12 (915 steps) | 0.011 s |
| Pendulum, 15 releases to the first pass | 0.046 s |
| Four-bar, kinematics over 361 crank angles | 0.020 s |
| Four-bar, 3 s, Dormand-Prince (3000 steps, 21000 evaluations) | 0.12 s |
| Quarter car, 6 s on the shaker, RK4 at 1 ms, the driver in Python | 0.10 s |
| Quarter car, twelve shaker frequencies, 8 s each | 2.0 s |
| Quarter car over the bump, 2 s | 0.038 s |
| `pytest python/tests/test_notebooks.py` in `mbd-try` (4 tests) | 9.96 s |

## Semi-implicit Euler's energy, with and without an event

The notebook's pendulum (1 kg, 0.8 m, released at 1 rad), semi-implicit
Euler at h = 10 ms, `run_recorded(80.0, 0.01)`. E = m L^2 v^2 / 2 - m g L
cos q; the modified energy E - (h/2) v m g L sin q. The event, when present,
is a falling crossing of q = 0 with no action (42 crossings in 80 s).

| | Span | Largest abs(E - E0) [J] | Mean change of the modified energy [J] |
|---|---|---|---|
| no event | 0-20 s | 0.0604 | -0.00006 |
| no event | 20-40 s | 0.0604 | -0.00006 |
| no event | 40-60 s | 0.0604 | -0.00006 |
| no event | 60-80 s | 0.0604 | -0.00006 |
| event | 0-20 s | 0.0730 | -0.00857 |
| event | 20-40 s | 0.0897 | -0.02362 |
| event | 40-60 s | 0.1036 | -0.03867 |
| event | 60-80 s | 0.1203 | -0.05423 |

Without the event, over 20 s: the modified energy E + s (h/2) v V' varied by
0.238 J with s = +1 and by 1.7e-4 J with s = -1, so the engine's variant
keeps E - (h/2) th' V'. A hand-written step (v += -h g/L sin q, then q += h
v) matched the engine's q to 1.9e-15 and v to 4.9e-15 over 2000 steps; the
other order (q first) differed by 0.034 rad.

## The shaker: errors against the closed form

RK4 at h = 1 ms, plate amplitude 2 mm, 8 s per frequency, a sine, a cosine
and a constant fitted to the last 2 s. "Estimate" is (h max(w, 72.86))^4 /
120; the notebook allows ten times it.

| f [Hz] | body | wheel | estimate | smallest tyre deflection [m] |
|---|---|---|---|---|
| 0.50 | -3.6e-09 | -2.6e-10 | 2.3e-07 | 0.0142 |
| 0.70 | -9.0e-09 | -1.8e-09 | 2.3e-07 | 0.0141 |
| 0.98 | -9.9e-09 | -2.3e-10 | 2.3e-07 | 0.0140 |
| 1.37 | +2.2e-08 | +4.9e-09 | 2.3e-07 | 0.0138 |
| 1.91 | +5.4e-09 | -2.5e-09 | 2.3e-07 | 0.0138 |
| 2.67 | +7.2e-09 | +1.7e-09 | 2.3e-07 | 0.0138 |
| 3.74 | -1.6e-09 | +2.9e-09 | 2.3e-07 | 0.0137 |
| 5.23 | +2.1e-09 | +1.4e-08 | 2.3e-07 | 0.0133 |
| 7.31 | +2.5e-08 | +6.8e-08 | 2.3e-07 | 0.0125 |
| 10.23 | +4.8e-08 | +2.0e-07 | 2.3e-07 | 0.0109 |
| 14.30 | -1.6e-06 | -1.0e-06 | 5.4e-07 | 0.0102 |
| 20.00 | -5.2e-06 | -3.1e-06 | 2.1e-06 | 0.0111 |

An earlier fit over [5, 6] s of a 6 s run at 3 Hz gave 2.4e-7 for the body:
the transient, e^(-2.55 x 5) = 3e-6 of its start, was not yet gone.

## The bump

Raised cosine 5 cm high, 0.5 m long, at 10 m/s from t = 0.2 s; RK4 at 1 ms.
The tyre's force is zero (wheel in the air) from 0.237 s to 0.292 s, 56
contiguous samples. Largest tyre force 10228 N against 2845 N at rest; the
wheel rose 53.2 mm, the body 20.3 mm. Momentum lost between the first and
the last sample in the air: 145.0899 kg m/s in the prototype's window
(the third sample in the air to the third last), 1.1e-13 from (m_s + m_u) g dt; over
the notebook's window, 156.4695 kg m/s, 1.4e-13 from it.

## The notebooks in a kernel

Each notebook run in a Jupyter kernel of `mbd-try` through
`jupyter_client` (a script in the session's scratch folder, with
`mbd.replay` and `mbd.plot` redirected to files), every text output read and
every figure saved as a PNG and looked at. The printed checks:

```
pendulum
period [s]: 1.91330283523, expected 1.91330283523, difference 5.4e-13 (allowed 1.0e-09)
first pass, a quarter period [s]: 0.478325708808, expected 0.478325708808, difference 1.5e-14 (allowed 1.0e-09)
rate at the bottom [rad/s]: 3.35769056772, expected 3.35769056772, difference 1.6e-12 (allowed 1.0e-09)
pull of the rod [N]: 18.8292687589, expected 18.8292687589, difference 8.9e-12 (allowed 1.0e-08)
periods [s]: largest difference 1.2e-12 (allowed 1.0e-09)
SemiImplicitEuler: largest |E - E0| 6.0e-02 J
semi-implicit Euler: largest |E - E0| [J]: 0.0604390656603, expected 0.0604862691159, difference 4.7e-05 (allowed 3.0e-03)
semi-implicit Euler: change of the modified energy [J]: 0.000169558178533, expected 0, difference 1.7e-04 (allowed 3.0e-03)
RK4: largest |E - E0| 1.4e-07 J
DormandPrince: largest |E - E0| 4.6e-09 J
natural frequency [rad/s]: 3.50178525898, expected 3.50178525898, difference 2.9e-13 (allowed 3.5e-08)
damping ratio: -0, expected 0, difference 0.0e+00 (allowed 1.0e-08)

four_bar
5 closure equations, 2 independent; 1 degree of freedom
joint angles over the turn [rad]: largest difference 8.9e-16 (allowed 1.0e-09)
their rates per radian of crank: largest difference 1.1e-15 (allowed 1.0e-09)
rocker's angle at a crank angle of 0.5153 rad [rad]: 8.881784197e-16, expected 0, difference 8.9e-16 (allowed 1.0e-09)
  its rate [rad per rad]: -6.66133814775e-16, expected 0, difference 6.7e-16 (allowed 1.0e-09)
rocker's angle at a crank angle of 4.1386 rad [rad]: 4.48530101949e-14, expected 0, difference 4.5e-14 (allowed 1.0e-09)
  its rate [rad per rad]: 1.00253139124e-13, expected 0, difference 1.0e-13 (allowed 1.0e-09)
<StaticReport: converged in 4 iterations, residual 7.92e-07 (given 49.4); 1 degree of freedom; 0 warnings>
the loop closed at rest [rad]: largest difference 3.1e-15 (allowed 1.0e-09)
dV/dtheta at rest [N m]: -2.05258032793e-07, expected 0, difference 2.1e-07 (allowed 1.0e-06)
frequency of small swings [rad/s]: 9.833154624, expected 9.83315498228, difference 3.6e-07 (allowed 9.8e-06)
total energy 26.238656 J, changing by at most 4.0e-12 J

quarter_car
<StaticReport: converged in 7 iterations, residual 5.1e-10 (given 9.81); 2 degrees of freedom; 0 warnings>
wheel's centre at rest [m]: 0.2857755, expected 0.2857755, difference 3.9e-14 (allowed 1.0e-07)
body at rest [m]: 0.563150500001, expected 0.5631505, difference 5.9e-13 (allowed 1.0e-07)
body bounce, 1.382 Hz:
  frequency [rad/s]: 8.68040307287, expected 8.6804030728, difference 6.4e-11 (allowed 8.7e-08)
  damping ratio: 0.294125194663, expected 0.294125194698, difference 3.5e-11 (allowed 1.0e-08)
  wheel per unit of body: 0.0784077493556+0.054295626316j, expected 0.0784077493483+0.0542956263194j, difference 8.1e-12 (allowed 9.5e-10)
wheel hop, 11.596 Hz:
  frequency [rad/s]: 72.860157152, expected 72.8601571527, difference 7.2e-10 (allowed 7.3e-07)
  damping ratio: 0.263475615544, expected 0.263475615567, difference 2.3e-11 (allowed 1.0e-08)
  wheel per unit of body: -4.45662663386+11.2956406291j, expected -4.45662663347+11.2956406281j, difference 1.1e-09 (allowed 1.2e-07)
smallest tyre deflection in any run: 10.2 mm
in the air from 0.237 s to 0.292 s
momentum lost in the air [kg m/s]: 156.4695, expected 156.4695, difference 1.4e-13 (allowed 1.0e-10)
```

Recordings written by those runs: pendulum 73,931 B; four-bar 768,233 B;
body bounce 112,948 B; wheel hop 104,450 B; the bump 310,287 B.

## The suite

`powershell -NoProfile -ExecutionPolicy Bypass -File scriptsuild.ps1 -Test`
(from cmd, output redirected to a file): CMake configured again (a new test
file, "GLOB mismatch"), then 543 of 543 tests passed in 111.77 s;
`Python: test_notebooks.py` 20.21 s, `Python: test_viewer.py` 6.38 s.
The same two files by hand in `mbd-python` with `-rs`: 12 passed, none
skipped, 16.59 s. In `mbd-try` (`python -m pytest python/tests`): 65 passed,
10 skipped (the parity tests), 17.35 s.
