# Generalised-alpha: method checks and measurements (task 4.3)

## 1. The phase error, from the recurrence

For q'' = -w^2 q the method's step is the recurrence written out in
`tests/kernel/test_kernel_integrators.cpp` (a0 = qdd0). Run in Windows
PowerShell 5.1 with h = 1 and w h as given, from (q, v) = (1, 0), for w T =
20; amplitude and phase of q - i v/(w h) against the exact e^(i w T):

```powershell
$am = (2*$rho - 1)/($rho + 1); $af = $rho/($rho + 1); $g = 0.5 - $am + $af; $b = 0.25*($g + 0.5)*($g + 0.5)
$W = $wh*$wh; $q = 1.0; $v = 0.0; $a = -$W*$q
# N times:
$qdd = -$W*$q
$a1 = (-(1-$af)*$W*($q + $v + (0.5-$b)*$a) - $am*$a + $af*$qdd) / ((1-$am) + (1-$af)*$b*$W)
$q = $q + $v + (0.5-$b)*$a + $b*$a1;  $v = $v + (1-$g)*$a + $g*$a1;  $a = $a1
```

| rho_infinity | w h | amplitude | phase error [rad] | phase error / (w T (w h)^2) |
|---|---|---|---|---|
| 0.8 | 0.02 | 0.9999993772 | -7.0388e-4 | -0.08799 |
| 0.8 | 0.01 | 0.9999998578 | -1.7598e-4 | -0.08799 |
| 0.5 | 0.02 | 0.9999924430 | -1.0018e-3 | -0.12523 |
| 0.5 | 0.01 | 0.9999984760 | -2.5049e-4 | -0.12525 |
| 1.0 | 0.02 | 1.0000000000 | -6.6663e-4 | -0.08333 |
| 1.0 | 0.01 | 1.0000000000 | -1.6666e-4 | -0.08333 |

At rho_infinity = 1 (the trapezoidal rule) the lag is (w h)^2 / 12 per radian,
the known value, and the amplitude is kept: a check of the recurrence. The
constant at 0.8 and 0.5 fits 1/12 + 0.377 (gamma - 1/2)^2, gamma - 1/2 = (1 -
rho) / (1 + rho); not derived in closed form.

A first attempt to compute the spectral radius by power iteration over a
fixed 49 steps gave 0.995 at w h = 0.1 for rho_infinity = 0.8, far too much
damping for a slow motion: the average of the norm's growth over a fraction
of a rotation is biased. Not used.

## 2. Stability for first-order states, by hand

For x' = lambda x with the velocity-level formulas, x+ = [(1 - alpha_m) x +
((1 - alpha_m)(1 - gamma) - gamma alpha_m) w + gamma alpha_f x_dot] / D with
D = (1 - alpha_m) - (1 - alpha_f) gamma lambda h: lambda h enters only
through D, so x+ = O(1 / |lambda h|). As lambda h -> -infinity the step
matrix on (x, w, x_dot) tends to the lower triangular

    [ 0                                0                                          0                  ]
    [ -1/gamma                         -(1 - gamma)/gamma                         0                  ]
    [ -(1 - alpha_m)/((1-alpha_f) gamma)  -((1-alpha_m)(1-gamma) - gamma alpha_m)/((1-alpha_f) gamma)  -alpha_f/(1-alpha_f) ]

with eigenvalues 0, -(1 - gamma)/gamma (-0.636 at rho_infinity = 0.8) and
-alpha_f/(1 - alpha_f) = -rho_infinity: inside the unit circle for gamma >=
1/2.

## 3. RK4's limit on the rolling car, measured

The simple vehicle (`build_simple_vehicle`, default `VehicleParams`, front
wheel inertia 2.45 kg m^2, driven rear 13.70 kg m^2 with the engine in gear 1,
rolling radius 0.3404 m, nv = 10) with its drivetrain, released at 0.5 m/s,
no throttle or brake; 0.5 s to settle, then the largest |spin x rolling
radius - speed| / speed over the next 0.5 s. A temporary hidden test case
(removed after use) in `tests/test_drivetrain.cpp`.

| method, step | largest deviation from free rolling | speed after 1 s |
|---|---|---|
| RK4 0.5 ms | 1.3e-5 | 0.500209 |
| RK4 0.6, 0.7, 0.8, 0.9 ms | 1.3e-5 | 0.500209 |
| RK4 1 ms | 2.84e-2 | 0.499817 |
| RK4 1.5 ms | 5.43e-2 | 0.499671 |
| RK4 2 ms | 3.65e-2 | 0.500072 |
| RK4 3 ms | 4.76e-1 | 0.494103 |
| RK4 5 ms | 1.42e-1 | 0.50279 |
| generalised-alpha 5 ms | 1.3e-5 | 0.500209 |

A first pass, the speed and spins after 1 s only: RK4 at 10 ms 0.5197 m/s
(front spin 1.020 against 1.469 rad/s for free rolling), 20 ms 0.5163, 40 ms
0.2140, 80 ms NaN. With RK4's limit 2.785 / h on the negative real axis,
stable at 0.9 ms and not at 1 ms puts the spins' relaxation rate between
2785 / 1e-3 and 2785 / 0.9e-3: 2800 to 3100 per second. 1.3e-5 is the tyres'
own slip in free rolling, the same for every converged run.

## 4. Measurements from the tests

Commands, from `build/tests` after `scripts\build.ps1 -Test` (439 tests
passed, no compiler warning): `test_kernel.exe "[implicit]" -s`,
`test_vehicle.exe "[implicit]" -s`, `test_alloc.exe -s`; the values are the
tests' INFO lines. rho_infinity = 0.8 unless stated.

**Oscillator (1 kg, 1 N/m, a = 0.1 m), 10 s, tolerance 1e-13.** The
simulator equals the recurrence within the test's 1e-12 at both steps (q
-0.0839263, v 0.0543726 at h = 0.02 s). Error against the exact solution
3.52104e-5 m at h = 0.02 s (predicted lag a 0.0880 T h^2 = 3.52e-5) and
8.80311e-6 at 0.01 s: ratio 3.9998. Two Newton iterations per step (1000 in
500 steps, 2000 in 1000), one iteration matrix per run.

**Spring of 8e9 N/m on 1 kg (w h = 89.4 at 1 ms), 1000 steps.** RK4 at 0.95
times its limit never grew; at 1.05 times it passed 1000 a within 100 steps.
Generalised-alpha, rho_infinity 0.8: largest amplitude 0.200599 m (the
recurrence: 0.200599 m), at the end 5.7e-78 m, energy 3.3e-153 of the start;
1234 Newton iterations, one matrix. rho_infinity 1: amplitude 0.1 m
throughout, energy equal to the start within the test's 1e-8; 2000 Newton
iterations, one matrix.

**Two masses, k_s = 78.96 N/m, k_t = 4e9 N/m, h = 1 ms, 2 s.** w1 = 6.28319,
w2 = 89442.7 rad/s; stiff mode's share 4.93e-9 a. The simulator equals the
recurrence run mode by mode within 1e-10 a. Slow motion's error 4.36379e-5 a
against the predicted lag 4.36568e-5 a (ratio 0.99957). One matrix, two
Newton iterations per step.

**Simple vehicle rolling at 0.5 m/s, 2 s after 0.3 s.** Generalised-alpha at
5 ms: speed 0.500212 to 0.500209 m/s, every wheel within 3.55e-5 of free
rolling; 460 steps, 482 Newton iterations, one matrix, 966 evaluations. RK4
at 0.5 ms: 0.500209 m/s, within 3.74e-5. RK4 at 5 ms: 0.50279 m/s, within
0.1417.

**Braked to a stop at 1 ms** (with RK4 and Dormand-Prince): all wheels
locked at zero with generalised-alpha too, once the drivetrain holds locked
spins at zero (before: -6.7e-7 rad/s).

**Double-wishbone sedan, 100 steps of 1 ms.** Generalised-alpha: 100 steps,
300 evaluations (RK4 400, Dormand-Prince 700); no allocation.
