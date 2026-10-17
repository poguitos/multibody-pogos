# Constraints per integrator: measurements (task 4.7)

Machine: the development laptop (i7-13700H, 16 GB, about 1.3 GB free during
these runs), Windows 11, MSVC RelWithDebInfo, one compiler. Wall times
unpinned (comparisons within one run, not benchmarks).

The model throughout: the planar four-bar of `tests/kernel/four_bar.hpp` with
the pendulum on its coupler (4 bodies, 4 coordinates, closed by a revolute
closure: 5 equations of rank 2), crank at 1 rad turning at 2 rad/s, the
pendulum at 0.3 rad at rest, gravity along -Y. No damping: its energy is
conserved.

## 1. Before any change: orders with and without projection

A temporary hidden test in `tests/kernel/test_kernel_integrators.cpp`
(removed after use): each method at fixed steps for 1 s, against
Dormand-Prince at tolerance 1e-13 (projected at 1e-10), with Newton's
tolerance at 1e-12; "proj" with the projection after each step, "free"
without. Errors |q - q_ref| and |v - v_ref| at 1 s; in brackets the ratio
when the step halves.

| method | steps | proj, q | free, q | proj, v | free, v |
|---|---|---|---|---|---|
| RK4 | 5, 2.5, 1.25 ms | 1.03e-5, 4.36e-7, 2.34e-8 (23.7, 18.7) | 2.22e-5, 1.74e-6, 1.20e-7 (12.8, 14.5) | 1.62e-5, 6.45e-7, 7.43e-8 | 1.27e-4, 9.79e-6, 6.71e-7 |
| semi-implicit Euler | 2, 1, 0.5 ms | 8.72e-2, 4.40e-2, 2.21e-2 (1.98, 1.99) | 1.77e-1, 9.00e-2, 4.54e-2 (1.97, 1.98) | 3.26e-1, ... | 9.96e-1, ... |
| Dormand-Prince | 10, 5, 2.5 ms | 1.29e-5, 5.14e-9, 7.47e-10 | 1.37e-5, 9.72e-8, 7.66e-10 | 5.25e-5, 8.01e-8, 4.50e-9 | 4.74e-5, 3.16e-7, 1.07e-9 |
| generalised-alpha | 4, 2, 1, 0.5, 0.25 ms | 1.43e-3, 2.87e-4, 6.27e-5, 1.46e-5, 3.50e-6 (4.99, 4.57, 4.31, 4.16) | 2.80e-4, 7.04e-5, 1.76e-5, 4.41e-6, 1.10e-6 (3.98, 3.99, 4.00, 4.00) | 5.74e-3, 1.10e-3, 2.35e-4, 5.36e-5, 1.28e-5 | 2.06e-3, 5.17e-4, 1.30e-4, 3.25e-5, 8.12e-6 |
| SDIRK4 | 10, 5, 2.5 ms | 1.57e-5, 5.63e-7, 3.09e-8 (27.9, 18.2) | 3.83e-5, 1.64e-6, 8.30e-8 (23.4, 19.7) | 6.08e-5, ... | 1.41e-4, ... |
| Radau IIA | 10, 5, 2.5 ms | 8.75e-7, 2.09e-8, 1.14e-9 (41.9, 18.3) | 4.44e-7, 5.65e-9, 2.56e-10 (78.6, 22.0) | 4.92e-6, ... | 3.03e-6, ... |
| linearly implicit Euler | 2, 1, 0.5 ms | identical to semi-implicit Euler | identical | | |

The linearly implicit step equals semi-implicit Euler here to every digit:
the four-bar has no force elements, so its K and C are zero and the step's
matrix is M. Radau IIA at fixed steps of 20 ms failed Newton (MBD-K058) at
t = 0.4 s; 10 ms was used. Dormand-Prince's and Radau IIA's smallest errors
are near the projection's own floor (residuals up to 1e-10 at every step).

The projection keeps or improves RK4, Dormand-Prince and SDIRK4, and
costs Radau IIA only at that floor. Generalised-alpha projected is second
order but 5.1 to 3.2 times less accurate than unprojected (at 4 to 0.25
ms); the unprojected runs are off the constraints, though (section 2).

## 2. Before any change: drift without projection

20 s at output steps of 2 ms (Dormand-Prince, SDIRK4, Radau IIA with error
control at 1e-6); |phi| and |J v - nu| measured by a separate solver at the
state.

| method | t = 1 s | 5 s | 20 s | energy change at 20 s |
|---|---|---|---|---|
| RK4 | 1.75e-7, 2.99e-7 | 1.62e-6, 4.83e-7 | 2.58e-5, 2.77e-6 | -7.4e-5 J |
| semi-implicit Euler | 2.42e-2, 8.10e-2 | 9.57e-2, 6.62e-3 | 4.63e-1, 5.62e-2 | -5.6 J |
| Dormand-Prince | 1.43e-10, 2.20e-10 | 8.75e-10, 1.60e-10 | 2.16e-8, 2.80e-9 | 5.1e-7 J |
| generalised-alpha | 3.03e-5, 1.38e-5 | 9.77e-5, 2.22e-4 | 1.30e-3, 1.32e-4 | 2.9e-2 J |
| SDIRK4 | 4.80e-9, 1.31e-8 | 2.00e-7, 6.41e-8 | 2.10e-6, 2.27e-7 | -4.8e-5 J |
| Radau IIA | 3.29e-10, 6.55e-10 | 2.07e-9, 1.34e-10 | 4.03e-8, 5.87e-9 | -8.7e-7 J |

## 3. Before any change: 200 s with projection

Output steps of 5 ms, the error-controlled methods at 1e-6. The drift is
what each step left before the projection (largest over the run), the
residual what the projection left; the projection's Gauss-Newton
iterations per step (0 / 1 / 2 / 3 or more).

| method | drift, position / velocity | residual, position / velocity | iterations 0/1/2/3+ | energy change, largest | evaluations | wall |
|---|---|---|---|---|---|---|
| RK4 | 1.82e-7 / 1.19e-5 | 1.0e-10 / 4.4e-15 | 7256 / 32744 / 0 / 0 | 2.2e-3 J | 160000 | 4.6 s |
| semi-implicit Euler | 7.31e-4 / 2.41e-2 | 1.0e-10 / 3.2e-15 | 3 / 17178 / 22819 / 0 | 12.8 J | 40000 | 2.3 s |
| Dormand-Prince | 2.83e-9 / 2.24e-7 | 1.0e-10 / 3.8e-15 | 38385 / 1615 / 0 / 0 | 5.0e-6 J | 280000 | 5.7 s |
| generalised-alpha | 2.56e-5 / 3.33e-3 | 1.0e-10 / 3.8e-15 | 25 / 38343 / 1632 / 0 | 0.113 J | 170433 | 4.7 s |
| SDIRK4 | 5.69e-9 / 1.45e-7 | 1.0e-10 / 3.6e-15 | 23034 / 16966 / 0 / 0 | 1.3e-5 J | 888488 | 18.0 s |
| Radau IIA | 4.34e-10 / 2.97e-9 | 1.0e-10 / 4.2e-15 | 39028 / 972 / 0 / 0 | 2.0e-6 J | 471852 | 9.5 s |
| linearly implicit Euler | as semi-implicit Euler | | | | 40000 | 2.3 s |

Generalised-alpha's velocity drift per step is 300 times RK4's at the same
step. Its energy falls by design (rho_infinity = 0.8).

## 4. Generalised-alpha in the stabilized index-2 form, three versions

The same probe, generalised-alpha with `enforce_constraints` meaning the
stabilized form ("proj" above became "stabilized").

**First version** (the iteration matrix's constraint rows from J and M^-1
J^T at the step's start, G = d(J v - nu)/dq left out, |J v - nu| held to
constraint_tolerance = 1e-10): Newton failed (MBD-K058, even with a fresh
matrix) at t = 0.408 s with steps of 4 ms and Newton's tolerance at 1e-12,
and at t = 34.5 s with steps of 5 ms at the default tolerances. A temporary
print of every Newton iteration (environment variable, removed after use),
the run at 4 ms:

| t | iteration: correction (in units of the tolerance), |phi|, |J v - nu| |
|---|---|
| 0 | 9.4e8, 3.1e-7, 2.3e-4; 9.1e6, 4.8e-9, 4.1e-6; 1.1e5, 3.4e-11, 3.8e-8; 1.1e3; 11.9; 0.13 (6 iterations) |
| 0.408 | 7.6e10, 2.4e-5, 1.2e-2; ... 1.17 at the 10th: not converged |

The constraints converge (|phi| to 1e-16); the corrections fall by only 100
per iteration at the start and 10 to 30 later, where the motion is faster:
J turns across a step by about h |v| / (link length), and the left-out G is
of the same size (beta / gamma h |G| / |J|, about 0.01 to 0.03 here).

**Second version** (J, M^-1 J^T and the independent equations from the
predictor's evaluation, Newton's first; G from the iteration matrix's
differences):

| t | iterations: correction, |phi|, |J v - nu| |
|---|---|
| 0 | 9.4e8, 3.1e-7, 2.3e-4; 1.8e5, 7.2e-13, 5.0e-8; 28, 4e-16, 5.0e-12; 4.0e-3 (4 iterations) |
| 0.400 | 2.7e10; 4.8e8; 1.2e7; 1.3e5; 1.8e3; 43; 0.64 (7 iterations) |

Orders (errors in q as in section 1; ratios when the step halves):

| steps | 4 ms | 2 ms | 1 ms | 0.5 ms | 0.25 ms |
|---|---|---|---|---|---|
| stabilized, q | 9.26e-4 | 2.23e-4 (4.15) | 5.48e-5 (4.07) | 1.36e-5 (4.03) | 3.38e-6 (4.02) |
| projected (section 1) | 1.43e-3 | 2.87e-4 (4.99) | 6.27e-5 (4.57) | 1.46e-5 (4.31) | 3.50e-6 (4.16) |
| free (section 1) | 2.80e-4 | 7.04e-5 | 1.76e-5 | 4.41e-6 | 1.10e-6 |

Clean second order; at short steps the same error as projected (3.4e-6
against 3.5e-6 at 0.25 ms), so the three-fold loss against free
generalised-alpha belongs to holding this method on the constraints, not to
the way it is held; at long steps, where the method is used, 1.5 to 1.1
times less error than projected (4 to 1 ms).

200 s at 5 ms, default tolerances (section 3's run): 213214 Newton iterations
in 40000 steps (5.3 a step; 253222 evaluations against 170433 projected),
8.9 s; position residual at most 2.1e-14, but velocity residual up to
3.96e-9, above the 1e-10 asked: the residual is measured at the iterate
before the last correction, which, through the simplified matrix, moves J v
by about 1 % of itself (1e-6). Energy change at most 0.109 J (projected:
0.113).

**Third version** (kept): |J v - nu| held to step_control.absolute_tolerance,
as Newton holds the velocities; positions still to constraint_tolerance.
First made generalised-alpha's default, then an option after section 6.

Its run over 1000 s on the four-bar (section 5): 654694 Newton iterations
in 200000 steps (3.27 a step, 4.27 evaluations; projected, section 3: 4.26),
position residual at most 6.7e-12, velocity residual at most 1.8e-7.

## 5. The tests

`scripts\build.ps1 -Test`: 461 tests, no compiler warning (two new tests
failed on their first run, both on wrong assertions: the top's closure at
its body origin, linear in the coordinates, so nothing drifted; and a
projection assumed to iterate at every step). Then `test_kernel.exe
"[constraints]" -s` and `test_kernel.exe "[long]" -s`.

**Generalised-alpha on the constraints** (four-bar, 1 s, Newton at 1e-12,
against Dormand-Prince at 1e-13 held to 1e-14): errors (q and v) 1.034e-3,
2.529e-4, 6.256e-5 at 2, 1, 0.5 ms, ratios 4.089 and 4.042; residuals at
most 1.2e-15 (positions) and 8.2e-15 (velocities); 4580 Newton iterations in
1000 steps at 1 ms. Left free at 2 ms: |phi| 6.7e-6, |J v - nu| 1.1e-5 after
1 s.

**1000 simulated seconds** (`[long]`, output steps of 5 ms, error control at
the default 1e-6; about twelve minutes on a quiet machine, 90 with a
compiler running beside it). Generalised-alpha here is the stabilized form,
which was the default when this ran; projected, it is section 3's run over
200 s, and the final run of the test (section 7). Largest drift per step
(positions; in brackets over the first and the last tenth) and velocities;
largest residuals after the projection; work:

| model, method | drift, positions (first tenth, last tenth) | drift, velocities | residuals | evaluations; Newton iterations; steps |
|---|---|---|---|---|
| four-bar, RK4 | 1.83e-7 (1.82e-7, 1.74e-7) | 1.21e-5 | 1.0e-10, 4.4e-15 | 800000; 0; 200000 |
| four-bar, semi-implicit Euler | 7.31e-4 (7.31e-4, 4.77e-5) | 2.40e-2 | 1.0e-10, 3.2e-15 | 200000; 0; 200000 |
| four-bar, Dormand-Prince | 2.98e-9 (2.60e-9, 2.87e-9) | 2.27e-7 | 1.0e-10, 4.4e-15 | 1400000; 0; 200000 |
| four-bar, generalised-alpha | 6.69e-12 (4.80e-12, 6.33e-12) | 1.85e-7 | the same (nothing projected) | 854702; 654694; 200000 |
| four-bar, SDIRK4 | 5.88e-9 (4.37e-9, 5.57e-9) | 1.91e-7 | 1.0e-10, 4.4e-15 | 4534948; 4531186; 220373 |
| four-bar, Radau IIA | 4.34e-10 (3.11e-10, 3.61e-10) | 3.02e-9 | 1.0e-10, 4.9e-15 | 2369614; 710780; 237266 |
| four-bar, linearly implicit Euler | as semi-implicit Euler | | | 200000; 0; 200000 |
| top, RK4 | 9.04e-9 (9.04e-9, 9.04e-9) | 2.0e-15 | 1.0e-10, 4.0e-16 | 800000; 0; 200000 |
| top, Dormand-Prince | 1.38e-10 (1.38e-10, 1.38e-10) | 2.2e-15 | 1.0e-10, 4.0e-16 | 1400000; 0; 200000 |
| top, generalised-alpha | 1.09e-16 (9.6e-17, 9.3e-17) | 1.8e-15 | the same | 660235; 460223; 200000 |
| top, SDIRK4 | 1.66e-9 (1.66e-9, 1.66e-9) | 1.4e-14 | 9.4e-17, 3.9e-16 | 4189369; 4189355; 200000 |
| top, Radau IIA | 1.20e-10 (1.17e-10, 1.17e-10) | 2.2e-15 | 1.0e-10, 4.1e-16 | 2621741; 792628; 243843 |

No residual above the tolerance at any step, no projection failure, no
drift growing. Semi-implicit Euler's (and the linearly implicit step's)
drift on the four-bar is largest at the start and falls as its energy error
changes the motion.

The top with semi-implicit Euler and with the linearly implicit step
(identical: no force elements), in the run before they were left out: 10 s
normal (drift at most 8.5e-14), but within the first 100 s of 1000 the drift
reached 2.4e56 and the residuals 2.2e11 and 8.9e52, with 193186 projection
failures in 200000 steps.

## 6. The cost on the sedan, and its accuracy

`build\bench\mbd_bench.exe`, with two columns added (evaluations and Newton
iterations per step over the drive), against the previous commit's
(`08099dd`, the same change applied to its benchmark, built from `git
archive` against the dependencies already fetched), both pinned to one
performance core at high priority, run alternately new, old, new, old. The
double-wishbone sedan with its drivetrain: n = nv + nx = 27 + 4, m = 17
constraint equations of rank 17. Median of one 1 ms step; then the median
of single steps over the drive.

**With the Newton matrix of n + 2 m** (stabilized generalised-alpha the
default):

| run | RK4 | generalised-alpha | linearly implicit Euler |
|---|---|---|---|
| new 1 | 144.7; 148.2 | 169.2; 168.7 | 130.0; 131.8 |
| old 1 | 149.4; 148.9 | 97.2; 99.4 | 129.2; 124.0 |
| new 2 | 152.2; 149.0 | 170.7; 174.6 | 131.0; 131.7 |
| old 2 | 148.2; 149.0 | 99.3; 99.2 | 134.3; 131.0 |

Evaluations per step 4.00, 2.41 and 1.00, generalised-alpha's Newton
iterations 1.37 per step, in both: the difference is linear algebra.

**With eta eliminated** (n + m; the iterates identical to the digit):

| run | RK4 | generalised-alpha | linearly implicit Euler |
|---|---|---|---|
| new 1 | 150.5; 149.8 | 145.7; 142.4 | 132.7; 130.9 |
| old 1 | 150.5; 150.0 | 98.4; 99.2 | 132.8; 130.4 |
| new 2 | 152.5; 149.1 | 146.6; 142.2 | 128.0; 130.0 |
| old 2 | 151.7; 148.6 | 110.9; 99.3 | 135.8; 130.7 |

**The pieces** (a temporary addition to the new benchmark, pinned, removed
after use): one LU of a random, well-conditioned matrix, ours against
Eigen's blocked `PartialPivLU` (each including a copy of 0.2 to 1.4 us):
31 rows 10.3 against 8.6 us, 48 rows 26.9 against 20.2 us, 65 rows 58.3
against 41.1 us. Generalised-alpha on the sedan at the drive's state: 144.9
us stabilized, 85.8 us with the constraints not enforced at all (so the
projection costs some 13 us of the old 98, and the stabilized form some 59
us, of which 17 the larger LU).

**Accuracy on a manoeuvre** (a temporary addition to both benchmarks,
removed after use): the sedan for 2 s, throttle 0.5 for 1 s then 0.2, the
steering rack's travel set before every step along a ramp from 0 at 0.5 s
to 0.02 m at 1 s; against RK4 at 0.1 ms, |q - q_ref| and |v - v_ref| at
the end, and the evaluations:

| step | stabilized (new) | projected (old) |
|---|---|---|
| 0.5 ms | 2.652e-5, 2.479e-5; 10738 | 3.029e-5, 2.481e-5; 8754 |
| 1 ms | 5.470e-5, 4.958e-5; 5876 | 6.312e-5, 4.962e-5; 5001 |
| 2 ms | 1.112e-4, 9.913e-5; 3201 | 1.289e-4, 9.928e-5; 2777 |
| 4 ms | 2.243e-4, 1.984e-4; 1914 | 2.606e-4, 1.986e-4; 1705 |

RK4 at 1 ms: 8.377e-5, 6.442e-5 (both builds). Every method is of first
order here: the steering is set once per step, and its sampling dominates.
The stabilized step is 1.13 to 1.16 times more accurate in positions, the
same in velocities, for 12 to 23 % more evaluations (each step's Newton
absorbs the driver's change) and its larger matrix.

## 7. The final runs

After the option (D37): `scripts\build.ps1 -Test`, 461 tests, no compiler
warning; `test_kernel.exe "[long]" -s`, all 60 checks passed (about 15
minutes). The methods other than generalised-alpha gave section 5's numbers
again; generalised-alpha both ways:

| model, method | drift, positions (first tenth, last tenth) | drift, velocities | residuals | evaluations; Newton iterations; steps |
|---|---|---|---|---|
| four-bar, generalised-alpha (projected) | 2.56e-5 (2.54e-5, 2.27e-5) | 3.33e-3 | 1.0e-10, 4.0e-15 | 849775; 649767; 200000 |
| four-bar, generalised-alpha, stabilized | 6.94e-12 (4.80e-12, 5.72e-12) | 1.70e-7 | the same (nothing projected) | 853553; 653545; 200000 |
| top, generalised-alpha (projected) | 2.0e-12 (2.1e-13, 2.0e-12) | 2.1e-13 | 2.0e-12, 4.2e-16 | 660235; 460223; 200000 |
| top, generalised-alpha, stabilized | 1.05e-16 (9.1e-17, 9.0e-17) | 2.0e-15 | the same | 660235; 460223; 200000 |

The stabilized-against-projected comparison of the order test (Newton at
1e-12, against Dormand-Prince at 1e-13), errors in q and v together: 2 ms
1.034e-3 against 1.391e-3, 1 ms 2.529e-4 against 2.975e-4, 0.5 ms 6.256e-5
against 6.813e-5 (1.34, 1.18, 1.09 times); Newton iterations 2613, 4580,
8173 against 2793, 4781, 8600.

The benchmark with the option, pinned, against the previous commit, new,
old, new, old: generalised-alpha 100.3 and 100.1 us (old 99.9 and 100.7),
stabilized 148.9 and 142.9 us; the full table in `docs/performance.md`,
task 4.7.
