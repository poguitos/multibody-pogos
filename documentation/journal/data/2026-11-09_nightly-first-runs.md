# The nightly's first runs: measurements (Phase 5 gate, D50)

Machine: the author's laptop (13th Gen Intel Core i7-13700H, Windows 11
Home), MSVC 19.50.35717, RelWithDebInfo (`scripts\build.ps1`, the dev
preset), 9 November 2026. CI: GitHub's `ubuntu-latest`, GCC, read through
the public API (annotations only; the logs need a login).

## 1. The nightly's annotations

Run 42581491960 (commit `2dae1aa`): the long suite failed at "Long kernel
tests" (exit 1), the mutation job at "Build and run the unmutated tests"
(exit 3), with no annotation saying which tests. `scripts/ci_run_tests.sh`
(commit `93464db`) turns a Catch2 run's failure lines into annotations. Run
42586184000 (commit `93464db`):

```
long-suite  step Long kernel tests failure
  Long kernel tests | tests/kernel/test_kernel_iftomm.cpp:360: failed: e < 1e-6 for:
    0.00000161335804805 < 0.000001 with 1 message: 'largest error 1.61336e-06 rad'
mutation    step Build and run the unmutated tests failure
  Unmutated kernel tests | tests/kernel/test_kernel_assembly.cpp:362: failed:
    r.conflicting_equations == std::vector<int>{first, first + 1} for: {  } == { 5, 6 }
  Unmutated kernel tests | tests/kernel/test_kernel_assembly.cpp:366: failed: m, ContainsSubstring(
    "constraint 1 (distance), its equation 1 of 1, on body 0 (ground) and body 4 (bob)") for:
    "MBD-K062: Assembly: the positions were not assembled: |phi| = 9.62e-01 after 50 iterations
    (tolerance 1.00e-10). The linearised constraint equations can all hold there: no equations
    contradict each other; ..."
  Unmutated kernel tests | tests/kernel/test_kernel_assembly.cpp:367: (the same, "constraint 2")
```

The long suite is built RelWithDebInfo (-O2), the mutation job Release
(-O3). The fast suite passed in the long suite's -O2 build and in every CI
build.

## 2. The conflict search, probed

A temporary probe in `conflicting_equations` (`src/kernel/conflicts.cpp`),
removed after use, printed to stderr at the end of the Levenberg-Marquardt
loop: tries, steps accepted, why it stopped, lambda, |J^T phi|, |phi|, the
bob's position and its distance rho from the anchors' axis (y = 0, z = 2).
`build\tests\test_kernel.exe "Assembly: constraints that contradict each
other*"` with `MBD_PROBE=1`, Marquardt's scaling (the code before D50):

```
PROBE LM: 200 tries, 99 accepted, stop 200 tries, lambda 3.742e+10, |g| 9.032e-01,
  |phi| 9.60316958607702831e-01, bob 1.34359503305111772e+00 -1.58051533912224020e-06
  1.99999864980109399e+00, rho 2.079e-06
All tests passed (6 assertions in 1 test case)
```

The least possible |phi| is 0.625 sqrt(2) = 0.8838834764831844 (both
distances' phi = (d^2 - 1) / 2 = 0.625 at d = 1.5 m).

## 3. Forty-eight starts

The new test, "Assembly: contradicting constraints are named from any
starting position": the bob started at x in {-0.4, 0.5, 1.2, 1.5, 2.3, 3.5},
y in {0.3, -1.1}, z in {2.1, 1.2} m, the crank at 0.7 and 2.0 rad.

**Marquardt's scaling** (the code before D50, built again for this count):
`test_kernel.exe "Assembly: contradicting constraints are named from any*"
--reporter compact`:

```
assertions: 96 | 56 passed | 40 failed
```

The starts that passed (the conflict named): the bob at (1.2, 0.3, 2.1),
(1.2, -1.1, 1.2), (1.5, 0.3, 1.2) and (2.3, -1.1, 2.1), each with both crank
angles. With the probe (its first 80 lines of output kept, so the last four
places, x = 3.5, were cut off), where each search stopped; the crank's two
angles gave the same lines:

| Bob started at [m] | Tries | rho at the stop [m] | Named |
|---|---|---|---|
| -0.4, 0.3, 2.1 | 200 | 5.262e-03 | no |
| -0.4, 0.3, 1.2 | 200 | 1.036e-03 | no |
| -0.4, -1.1, 2.1 | 200 | 1.198e-02 | no |
| -0.4, -1.1, 1.2 | 200 | 7.397e-03 | no |
| 0.5, 0.3, 2.1 | 200 | 4.118e-03 | no |
| 0.5, 0.3, 1.2 | 200 | 2.393e-02 | no |
| 0.5, -1.1, 2.1 | 200 | 6.143e-02 | no |
| 0.5, -1.1, 1.2 | 200 | 1.108e-01 | no |
| 1.2, 0.3, 2.1 | 200 | 2.079e-06 | yes |
| 1.2, 0.3, 1.2 | 200 | 1.023e-02 | no |
| 1.2, -1.1, 2.1 | 200 | 7.369e-03 | no |
| 1.2, -1.1, 1.2 | 200 | 7.833e-04 | yes |
| 1.5, 0.3, 2.1 | 200 | 3.142e-02 | no |
| 1.5, 0.3, 1.2 | 200 | 6.437e-04 | yes |
| 1.5, -1.1, 2.1 | 200 | 1.096e-02 | no |
| 1.5, -1.1, 1.2 | 200 | 8.812e-03 | no |
| 2.3, 0.3, 2.1 | 200 | 7.406e-02 | no |
| 2.3, 0.3, 1.2 | 200 | 1.339e-02 | no |
| 2.3, -1.1, 2.1 | 171 (lambda above 1e12) | 2.776e-09 | yes |
| 2.3, -1.1, 1.2 | 200 | 1.512e-02 | no |

**Levenberg's damping** (D50), the same command, then with the probe:

```
All tests passed (96 assertions in 1 test case)
```

| Bob started at [m] | Tries | Accepted | lambda at the stop | rho at the stop [m] |
|---|---|---|---|---|
| -0.4, 0.3, 2.1 | 44 | 11 | 1.666e+12 | 1.945e-09 |
| -0.4, 0.3, 1.2 | 42 | 10 | 1.250e+12 | 9.053e-09 |
| -0.4, -1.1, 2.1 | 44 | 11 | 1.666e+12 | 1.159e-09 |
| -0.4, -1.1, 1.2 | 44 | 11 | 1.666e+12 | 6.433e-10 |
| 0.5, 0.3, 2.1 | 42 | 10 | 1.250e+12 | 6.175e-09 |
| 0.5, 0.3, 1.2 | 44 | 11 | 1.666e+12 | 8.861e-10 |
| 0.5, -1.1, 2.1 | 44 | 11 | 1.666e+12 | 1.947e-09 |
| 0.5, -1.1, 1.2 | 44 | 11 | 1.666e+12 | 2.514e-09 |
| 1.2, 0.3, 2.1 | 48 | 13 | 2.962e+12 | 1.147e-08 |
| 1.2, 0.3, 1.2 | 44 | 11 | 1.666e+12 | 4.122e-10 |
| 1.2, -1.1, 2.1 | 42 | 10 | 1.250e+12 | 1.164e-08 |
| 1.2, -1.1, 1.2 | 44 | 11 | 1.666e+12 | 6.130e-10 |
| 1.5, 0.3, 2.1 | 44 | 11 | 1.666e+12 | 1.052e-09 |
| 1.5, 0.3, 1.2 | 39 | 8 | 2.812e+12 | 1.552e-08 |
| 1.5, -1.1, 2.1 | 42 | 10 | 1.250e+12 | 1.402e-08 |
| 1.5, -1.1, 1.2 | 42 | 10 | 1.250e+12 | 1.668e-08 |
| 2.3, 0.3, 2.1 | 51 | 15 | 1.316e+12 | 1.295e-08 |
| 2.3, 0.3, 1.2 | 44 | 11 | 1.666e+12 | 1.233e-09 |
| 2.3, -1.1, 2.1 | 44 | 11 | 1.666e+12 | 6.342e-10 |
| 2.3, -1.1, 1.2 | 44 | 11 | 1.666e+12 | 8.302e-10 |
| 3.5, 0.3, 2.1 | 44 | 11 | 1.666e+12 | 4.315e-10 |
| 3.5, 0.3, 1.2 | 44 | 11 | 1.666e+12 | 9.346e-10 |
| 3.5, -1.1, 2.1 | 44 | 11 | 1.666e+12 | 7.392e-10 |
| 3.5, -1.1, 1.2 | 44 | 11 | 1.666e+12 | 5.840e-10 |

Every search stopped because no step lowered |phi| (lambda above 1e12),
every one at |phi| = 8.83883476483184438e-01, and every one named the two
distances. The crank's two angles again gave the same lines.

## 4. The 40-four-bar's error, and how much it moves

A temporary hidden test, removed after use, ran `NFourBar(40)` with
Dormand-Prince as `n_four_bar_error` does (10 s, the error against the
closed form at every output), varying one setting at a time from the
committed case (integration tolerance 1e-10, constraints 1e-12, output
every 10 ms). 237 s in all:

```
PROBE N 40 tol 1.0e-10 ctol 1.00e-12 dt 0.0100: largest error 4.3373e-07 rad at t = 8.850 s
PROBE N 40 tol 1.0e-10 ctol 1.00e-12 dt 0.0080: largest error 2.5683e-07 rad at t = 8.848 s
PROBE N 40 tol 1.0e-10 ctol 1.00e-12 dt 0.0125: largest error 3.2142e-07 rad at t = 8.850 s
PROBE N 40 tol 1.0e-10 ctol 1.00e-12 dt 0.0200: largest error 1.7332e-07 rad at t = 8.840 s
PROBE N 40 tol 1.0e-10 ctol 1.00e-12 dt 0.0250: largest error 2.6912e-07 rad at t = 8.850 s
PROBE N 40 tol 1.0e-10 ctol 8.00e-13 dt 0.0100: largest error 4.3725e-07 rad at t = 8.850 s
PROBE N 40 tol 1.0e-10 ctol 9.00e-13 dt 0.0100: largest error 4.2084e-07 rad at t = 8.850 s
PROBE N 40 tol 1.0e-10 ctol 1.10e-12 dt 0.0100: largest error 4.0303e-07 rad at t = 8.850 s
PROBE N 40 tol 1.0e-10 ctol 1.25e-12 dt 0.0100: largest error 8.5200e-07 rad at t = 8.850 s
PROBE N 40 tol 1.0e-11 ctol 1.00e-12 dt 0.0100: largest error 5.1516e-07 rad at t = 8.850 s
PROBE N 40 tol 1.0e-12 ctol 1.00e-12 dt 0.0100: largest error 9.7861e-07 rad at t = 8.850 s
```

The run at a 20 ms output also reported MBD-K051 once (t = 7.033 s,
residual 1.17e-12 after 20 iterations). For comparison: 3.4e-7 rad on
Linux in October (`2026-10-20_iftomm-benchmarks.md`), 1.6e-6 in GCC's
nightly build (section 1).

## 5. The suites after the changes

On the working tree, which also held the gate's how-to test (three of the
530 cases):

```
scripts\build.ps1 -Test                          100% tests passed, 0 tests failed out of 530 (43 s)
build\tests\test_kernel.exe                      All tests passed (15844 assertions in 159 test cases)
build\tests\test_kernel.exe "[long]"             All tests passed (1069 assertions in 2 test cases), 106 s
```
