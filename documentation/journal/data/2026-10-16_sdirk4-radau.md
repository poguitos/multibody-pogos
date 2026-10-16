# SDIRK4 and Radau IIA: method checks and measurements (task 4.4)

Machine: the development laptop (i7-13700H, 16 GB), Windows 11, MSVC
RelWithDebInfo, one compiler. No timings: every cost below is a count of
evaluations of the right-hand side (`IntegratorStats::evaluations`), which is
what the methods spend; the matrices are small (n = 26) and their LUs cheap.

Three versions of the code produced the tables, named where they are used:

- **A**: SDIRK4 with Newton stopping at a tenth of the tolerance (no Radau
  IIA yet, or Radau IIA with the same rule);
- **B**: Newton stopping at min(0.03, sqrt(tol)) of the tolerances (D34),
  floor at the rounding of the stage's terms only;
- **C**: the committed code, B plus the floor at the state's rounding
  amplified by J.

## 1. The tableaux, checked before use

Perl with `Math::Complex`, scripts in the session's scratch folder, reproduced
here in outline (the full checks are short loops over the coefficients).

**SDIRK4** (Hairer and Wanner, table IV.6.5, gamma = 1/4):

- row sums equal the nodes (largest residual 8.9e-16);
- the eight fourth-order conditions on b (the last row of A) hold to 1e-15;
- b_hat = (59/48, -17/96, 225/32, -85/12, 0) meets the four of third order;
- stage order 1 only: A c - c^2 / 2 = 0.031, 0.031, 0.041, 0.046, 0.000;
- max |R(iy)| over y in [0.01, 1e4] = 1.000000000000; |R(-1e8)| = 9.3e-8;
  |R(89.4 i)| = 0.1041;
- (e^z - R(z)) / z^5 = 8.45e-4 at z = 0.05 i.

**Radau IIA** (three stages, nodes (4 -+ sqrt 6)/10 and 1):

- row sums equal the nodes (2.8e-17); quadrature exact for degree 0 to 4
  (b . c^5 - 1/6 = 1.67e-3: order 5); stage conditions exact for k = 1 to 3,
  not 4 (stage order 3);
- R(z) from the coefficients equals the Pade (2, 3) form to 1e-15 at 0.3 i,
  -2 + i and 89.4 i; |R(89.4 i)| = 0.033589;
- (e^z - R(z)) / z^6 -> -1.3863e-4 at z = 0.05 i (-1/7200 = -1.3889e-4);
- the eigenvalues of A^-1: gamma = 30 / (6 + 81^(1/3) - 9^(1/3)) =
  3.6378342527445, alpha +- i beta = 2.68108287362775 +- 3.05043019924741 i,
  roots of z^3 - 9 z^2 + 36 z - 60 to 1e-14;
- RADAU5's error estimate for y' = lambda y, (gamma - z)^-1 (z + dd . Z(z)),
  dd = (-(13 + 7 sqrt 6)/3, (-13 + 7 sqrt 6)/3, -1/3): -0.004580 z^4 at small
  z; |estimate| -> 0.89, 0.999, 0.99999 at z = 10 i, 100 i, 1000 i (a stiff
  component away from equilibrium: hence the second evaluation after a
  first step or a rejection).
- For the oscillator test: N |e^(ih) - R(ih)| a over T = 10 s, a = 0.1:
  4.436395e-8 at h = 0.2, 1.388259e-9 at h = 0.1 (exact global errors
  4.436394e-8 and 1.388260e-9); ratio 31.9565.

## 2. The bushed pendulum

`tests/kernel/bushed_pendulum.hpp`: two 1 kg links, 1 m along X (solid
boxes 1 x 0.05 x 0.05 m, so the torsional inertia about the link's axis is
4.2e-4 kg m^2), each on a free joint; a bushing from the ground to link 1
at its end and from link 1's far end to link 2: stiffness k in translation,
k_r in rotation about X and Z, nothing about the hinge axis Y; damping ratio
0.2 for the translation on 1 kg and the rotation on 1/3 kg m^2 (which leaves
the torsion overdamped: rates -5.5e5 and -4.3e3 per second at k_r = 1e6).
Released level, at rest, gravity -Z; the tip of link 2 after 1 s.
n = nq + nv = 14 + 12 = 26.

Reference: Dormand-Prince at tolerance 1e-12, tip (-1.662764113350, 0,
-1.041248865058) after 1 s; 138701 steps, 970985 evaluations, 13 rejected
(k = 1e8, k_r = 1e6). With soft bushings (1e4, 1e2): (-1.663910940521, 0,
-1.054679908441), 25609 evaluations.

The tables come from a temporary hidden test case (removed after use; its
source is in section 8), run from `build/tests` as
`test_kernel.exe "DEBUG sdirk bushed"` with the settings in environment
variables, for example

```
MBD_BENCH_TIMES=0.2,1.0 MBD_BENCH_ALPHA= MBD_BENCH_SDIRK=4e-3,2e-3,1e-3,5e-4 \
MBD_BENCH_ADAPT= MBD_BENCH_RADAU=8e-3,4e-3,2e-3,1e-3 ./test_kernel.exe "DEBUG sdirk bushed"
```

Fixed-step runs use tolerances 1e-10 for Newton unless stated.

## 3. Why SDIRK4's Newton was slow: the Jacobian turns within a step

Code before A, J kept from step to step and rebuilt only when Newton
"slowed" (average contraction worse than one half): at h = 8 ms Newton took
4 to 40 iterations per stage, and with max_iterations = 10 the run stopped
with K058 at t = 0.024 s. With J rebuilt at every iterate (a temporary
switch), 3 to 4 iterations: the difference Jacobian is accurate; it is the
reuse that fails.

Spectral radius of the Newton iteration's error propagation with J frozen,
E = (I - gamma h J_0)^-1 gamma h (J_stage - J_0), at the last stage of each
step (h = 8 ms, J_0 as the policy left it), and the same with the
correction carried between the frames of J_0 and the stage, joint by joint
(quaternion parts by q_stage q_0^-1, positions by R_stage R_0^T):

| t [s] | frozen | carried |
|---|---|---|
| 0 | 0.041 | 0.041 |
| 0.008 | 0.060 | 0.065 |
| 0.016 | 0.070 | 0.078 |
| 0.024 | 0.160 | 0.186 |
| 0.040 | 0.279 | 0.327 |
| 0.056 | 0.411 | 0.484 |
| 0.080 | 0.628 | 0.748 |

Carrying the correction does not help: the change is not an effect of the
quaternion coordinates. The entries that change most are the derivatives of
each link's torsional acceleration with respect to its quaternion's z part
(rows 14 and 20, columns 5 and 12), growing as 2.9e8 omega_y^2 = 7.3e9 phi:
the bushing's rotation vector puts the tilt axis halfway between the two
markers, so a bending deflection delta makes a torsion moment of about k_r
phi delta / 2, divided by the torsional inertia 4.2e-4 kg m^2. At h = 4 ms
the radius grows by about 0.025 per step of J's age (0.020 fresh, 0.30 after
12 steps).

## 4. Newton's tests

Hairer and Wanner's tests (code before A, h = 8 ms): stop when theta / (1 -
theta) times the correction is below 0.1 of the tolerance; fail when theta
>= 1 or theta^(k_max - k) theta / (1 - theta) d > 0.1. Corrections (in
tolerance units) of a stage with J built at the step's start that was
abandoned, and of the same stage after J was rebuilt there:

```
k 1 d 5.975e+05   k 2 d 3.580e+04   k 3 d 3.591e+04   too slow, theta 0.245
(J at the stage) k 1 d 6.148e+05   k 2 d 2.485e+03   k 3 d 1.094e+02   k 4 d 1.059e-02
```

and one abandoned even with J built at its own predictor (then K058):

```
(J at the stage, t = 0.030) k 1 d 6.378e+05   k 2 d 1.955e+05   too slow, theta 0.306
```

The first correction from the predictor contracts by 0.25 to 0.8, the later
ones by 1e-2 to 1e-4: no constant rate. With these tests J was built 11
times in the first three steps (15 stages), and the run stopped at t =
0.024 s. Replaced by: stop at the fraction (section 6), fail
when a correction is not smaller than the one two iterations before or after
max_iterations.

## 5. SDIRK4's order on the pendulum

Code A. Errors of the tip [m] at t = 0.2 and 1 s, k = 1e8, k_r = 1e6:

| h | 0.2 s | 1 s | evaluations | iteration matrices |
|---|---|---|---|---|
| 4 ms | 4.03e-5 | 1.37e-3 | 18827 | 260 |
| 2 ms | 3.41e-5 | 1.25e-3 | 23932 | 150 |
| 1 ms | 6.01e-6 | 2.27e-4 | 40041 | 75 |
| 0.5 ms | 1.09e-6 | 4.21e-5 | 73536 | 40 |

Ratios at 1 s from 2 to 1 to 0.5 ms: 5.5 and 5.4 (order 2.45). Code C gives
the same errors to three digits (19761, 24353, 39998, 72949 evaluations).

At t = 0.2 s, h = 2, 1, 0.5 ms, generalised-alpha at 0.5 and 0.25 ms beside
(code A):

| bushings (k, k_r) | SDIRK4 | ratios | generalised-alpha |
|---|---|---|---|
| 1e6, 1e4 | 4.22e-7, 3.91e-8, 2.77e-9 | 10.8, 14.1 | 7.23e-8, 1.81e-8 |
| 1e8, 1e6 | 3.41e-5, 6.01e-6, 1.09e-6 | 5.7, 5.5 | 7.28e-8, 1.84e-8 |
| 1e10, 1e8 | 1.41e-5, 7.35e-6, 4.33e-6 | 1.9, 1.7 | 7.30e-8, 1.86e-8 |

The reference for 1e10 needed 532586 evaluations (tolerance 1e-11, to 0.2
s); for 1e6, 13835.

With soft bushings (1e4, 1e2), nothing stiff, code C:

| h | error at 1 s | evaluations |
|---|---|---|
| 10 ms | 1.273e-5 | 4419 |
| 5 ms | 9.063e-7 | 7949 |
| 2.5 ms | 5.644e-8 | 14510 |

Ratios 14.0 and 16.1: fourth order.

SDIRK4 with error control, k = 1e8 (code B):

| tolerance | error at 1 s | steps | evaluations |
|---|---|---|---|
| 1e-4 | 7.01e-4 | 1889 | 47661 |
| 1e-5 | 6.60e-5 | 5339 | 149539 |
| 1e-6 | 3.61e-6 | 11784 | 337364 |
| 1e-7 | 6.67e-8 | 22731 | 700695 |
| 1e-8 | 1.77e-9 | 41687 | 1254508 |

## 6. Radau IIA and Newton's stopping fraction

Radau IIA with error control, k = 1e8, error at 1 s and evaluations, by the
fraction of the tolerance at which Newton stops (a temporary switch on code
A, then code C, whose rule gives 0.01 at 1e-4, 1e-3 at 1e-6, 1e-4 at 1e-8):

| fraction | 1e-4 | 1e-6 | 1e-8 |
|---|---|---|---|
| 0.1 | 1.90e-4 (6648) | 4.06e-7 (9977) | 7.00e-8 (14061) |
| 0.01 | 9.94e-6 (8414) | 2.37e-8 (11716) | 2.78e-9 (15995) |
| 0.001 | 4.49e-7 (10075) | 6.44e-9 (13600) | 9.76e-11 (17575) |
| code C | 3.78e-6 (8399) | 1.04e-7 (11548) | 4.06e-9 (13986) |

Code C at 1e-5 and 1e-7: 1.24e-6 (10663) and 1.03e-7 (11298). In code C the
floor at the state's rounding amplified by J sets Newton's stop in the stiff
rows at tolerances of 1e-6 and below, which is why 1e-6 and 1e-7 end alike.

Code B (no J term in the floor) at tolerance 1e-10 and fixed steps: every
run, SDIRK4 at 4 to 0.5 ms and Radau IIA at 8 to 1 ms, stopped with K058 in
its first step (10 to 13 Newton iterations).

Radau IIA at fixed steps, code C, Newton at 1e-10:

| h | error at 0.2 s | at 1 s | evaluations | iteration matrices |
|---|---|---|---|---|
| 8 ms | K058 at t = 0 | | | |
| 4 ms | 1.25e-11 | K058 at t = 0.86 | | |
| 2 ms | 3.11e-12 | 5.33e-11 | 15591 | 83 |
| 1 ms | 5.76e-12 | 1.44e-10 | 21381 | 30 |

At 2 and 1 ms the errors are at the reference's own level. A J rebuilt at
the step's middle when Newton fails (temporary switch, code A): 4 ms ran to
the end (5.3e-11 m, 17163 evaluations), 8 ms stopped at t = 0.12 s.

With soft bushings (1e4, 1e2), code C:

| h | error at 1 s | evaluations |
|---|---|---|
| 20 ms | 6.052e-7 | 2919 |
| 10 ms | 1.920e-8 | 2940 |
| 5 ms | 5.903e-10 | 4485 |

Ratios 31.5 and 32.5: fifth order.

## 7. Generalised-alpha beside them, and the test values

Generalised-alpha, k = 1e8, error at 1 s (codes B and C, which do not differ
for it; the method is unchanged by task 4.4 apart from its Newton's
progress test):

| h | Newton at 1e-10 | evaluations | Newton at 1e-9 | evaluations |
|---|---|---|---|---|
| 1 ms | 6.806e-6 | 11742 | 6.811e-6 | 10488 |
| 0.5 ms | 1.703e-6 | 16979 | 1.724e-6 | 15856 |
| 0.25 ms | 4.280e-7 | 28783 | 4.590e-7 | 26057 |
| 0.125 ms | 1.027e-7 | 54711 | 7.780e-8 | 48595 |

The test "Radau IIA beats generalised-alpha on the bushed pendulum at tight
tolerance" (`test_kernel.exe "Radau IIA beats*" -s`, code C): reference
970985 evaluations; generalised-alpha at 0.5 ms 1.72437e-6 m (15856), at
0.25 ms 4.59018e-7 m (26057); Radau IIA at tolerance 1e-6, 1.04255e-7 m for
11548 evaluations. "SDIRK4 and Radau IIA converge at their orders on a
smooth spatial model": SDIRK4 at 5 and 2.5 ms 9.06333e-7, 5.64401e-8
(16.0583); Radau IIA at 10 and 5 ms 1.92047e-8, 5.90336e-10 (32.5318).

The other new tests (`test_kernel.exe "SDIRK4: *,Radau IIA: *" -s`):

- SDIRK4 on the oscillator (tolerance 1e-14): one step of 0.3 s within
  1.5e-16 of R(0.3 i) a; errors over 10 s 8.46096e-8 and 5.28931e-9 at 0.1
  and 0.05 s, as predicted to six digits, ratio 15.9963; 1000 Newton
  iterations in 100 steps, one matrix, 1003 evaluations.
- SDIRK4, spring of w h = 89.4: one step within 3e-17 of R(89.4 i) a, |R| =
  0.104017.
- SDIRK4, two masses with the stiff link, tolerance 1e-8: 161 steps (170
  predicted), 3 rejected, error 4.97e-9 m.
- Radau IIA on the oscillator: one step within 2e-17 of R(0.3 i) a; errors
  4.43639e-8 and 1.38826e-9 at 0.2 and 0.1 s, as predicted to six digits,
  ratio 31.9565; 100 Newton iterations in 50 steps, one matrix, 303
  evaluations.
- Radau IIA, spring of w h = 89.4: within 6e-17 of R(89.4 i) a, |R| =
  0.0335725.
- Radau IIA, two masses: 136 steps (140 predicted), 3 rejected, 278 Newton
  iterations, one matrix, 977 evaluations, error 1.65e-10 m (about 1e-10
  predicted).
- The double-wishbone sedan, 100 steps of 1 ms, no allocation: evaluations
  RK4 400, Dormand-Prince 700, generalised-alpha 300, SDIRK4 1403, Radau IIA
  526.

Full suite: `scripts\build.ps1 -Test`, 447 tests passed, no compiler
warning.

## 8. The measuring test case

Added to `tests/kernel/test_kernel_integrators.cpp` while measuring, hidden
(`[.debug]`), then removed. It needed `<cstdio>`, `<cstdlib>` and `<string>`.

```cpp
TEST_CASE("DEBUG sdirk bushed", "[.debug]")
{
    auto env = [](const char* name, Real fallback) {
        const char* s = std::getenv(name);
        return s ? std::atof(s) : fallback;
    };
    auto list = [](const char* name, const char* fallback) {
        const char* s = std::getenv(name);
        std::vector<Real> out;
        std::stringstream ss(s ? s : fallback);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) out.push_back(std::atof(item.c_str()));
        }
        return out;
    };
    const Real tol = env("MBD_BENCH_TOL", 1e-10), alpha_tol = env("MBD_BENCH_GATOL", tol);
    const int max_it = static_cast<int>(env("MBD_BENCH_MAXIT", 10));
    const std::vector<Real> times = list("MBD_BENCH_TIMES", "1.0");
    mbd_test::BushedPendulum bp(env("MBD_BENCH_K", 1e8), env("MBD_BENCH_KR", 1e6));
    auto run = [&](Integrator method, Real h, Real tl, bool adaptive, std::vector<Vec3>& tips) {
        Simulator sim(bp.sys);
        sim.method = method;
        bp.release(sim);
        sim.step_control.adaptive = adaptive;
        sim.step_control.relative_tolerance = tl;
        sim.step_control.absolute_tolerance = tl;
        sim.implicit_control.max_iterations = max_it;
        std::string message;
        tips.clear();
        try {
            for (const Real t : times) {
                sim.run(t - sim.time, h);
                tips.push_back(bp.tip(sim));
            }
        } catch (const std::exception& e) {
            message = e.what();
        }
        // prints the method, h, the time reached and integrator_stats(), and
        // the message if the run stopped
    };
    std::vector<Vec3> reference, tips;
    run(Integrator::DormandPrince, 0.01, env("MBD_BENCH_REFTOL", 1e-12), true, reference);
    // then generalised-alpha at the steps in MBD_BENCH_ALPHA (Newton at
    // alpha_tol), SDIRK4 at MBD_BENCH_SDIRK, SDIRK4 with error control at the
    // tolerances in MBD_BENCH_ADAPT, Radau IIA at MBD_BENCH_RADAU and with
    // error control at MBD_BENCH_RADAPT, each printing |tip - reference| at
    // every time in MBD_BENCH_TIMES
}
```
