# Dormand-Prince: coefficient checks and measurements (task 4.2)

## 1. The stability function and the error estimate, from the coefficients

For y' = lambda y a Runge-Kutta step multiplies y by R(z) = 1 + z b^T (I - z
A)^-1 1, z = lambda h, so the coefficient of z^(k+1) is b^T A^k 1; the error
estimate h e^T k has coefficients e^T A^k 1. The stage matrix is strictly
lower triangular (7 x 7), so both series end. Computed in Windows PowerShell
5.1 with the coefficients typed into the simulator (`src/kernel/simulator.cpp`):

```powershell
$A = @( @(0,0,0,0,0,0,0), @((1/5),0,0,0,0,0,0), @((3/40),(9/40),0,0,0,0,0),
 @((44/45),(-56/15),(32/9),0,0,0,0),
 @((19372/6561),(-25360/2187),(64448/6561),(-212/729),0,0,0),
 @((9017/3168),(-355/33),(46732/5247),(49/176),(-5103/18656),0,0),
 @((35/384),0,(500/1113),(125/192),(-2187/6784),(11/84),0))
$b = @((35/384),0,(500/1113),(125/192),(-2187/6784),(11/84),0)
$e = @((71/57600),0,(-71/16695),(71/1920),(-17253/339200),(22/525),(-1/40))
# w = 1; for m = 0..6: print b.w and e.w, then w = A w
```

| power | b^T A^k 1 | 1/(k+1)! | e^T A^k 1 |
|---|---|---|---|
| z^1 | 1 | 1 | 0 |
| z^2 | 0.5 | 0.5 | -1.4e-17 (zero) |
| z^3 | 1/6 | 1/6 | -1.7e-17 (zero) |
| z^4 | 1/24 | 1/24 | 4.3e-18 (zero) |
| z^5 | 1/120 | 1/120 | -8.0833333333e-4 = -97/120000 |
| z^6 | 1/600 | 1/720 | 3.25e-4 = 13/40000 |
| z^7 | 0 | 1/5040 | -4.1666666667e-5 = -1/24000 |

So R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 + z^5/120 + z^6/600 (fifth order:
it matches e^z to z^5), and E(z) = -97/120000 z^5 + 13/40000 z^6 - 1/24000
z^7 (the estimate is of order h^5, as a fourth-order embedded solution
gives). For an oscillator, z = i w h:

    e^(iwh) - R(iwh) = (wh)^6 / 3600 - i (wh)^7 / 5040 + ...

One step from (1, 0) with w = 1: the endpoint error at h = 0.2, 0.1, 0.05,
0.025 was 1.8021e-8, 2.7873e-10, 4.3441e-12, 6.7845e-14, against h^6 / 3600
= 1.7778e-8, 2.7778e-10, 4.3403e-12, 6.7817e-14.

## 2. The continuous extension

Hairer's DOPRI5 form, u(s) = y0 + s (y1 - y0 + (1 - s) (h k1 - (y1 - y0) +
s (... + (1 - s) h sum d_j k_j))), with his d_j. One step from w = 1 of w' =
i w, computed with System.Numerics.Complex; error at s against e^(i s h):

| s | h = 0.2 | h = 0.1 | h = 0.05 | h = 0.025 | ratio at the last halving | error / h^5 at h = 0.025 |
|---|---|---|---|---|---|---|
| 0.25 | 7.5023e-8 | 2.3307e-9 | 7.2727e-11 | 2.2719e-12 | 32.01 | 2.3264e-4 |
| 0.5 | 9.3649e-8 | 2.8568e-9 | 8.8723e-11 | 2.7683e-12 | 32.05 | 2.8347e-4 |
| 0.75 | 2.9885e-8 | 8.7975e-10 | 2.7060e-11 | 8.4221e-13 | 32.13 | 8.6243e-5 |

Fourth order: local error of order h^5. (A first attempt reported first
order: the scratch script built its "exact" value with `New-Object
System.Numerics.Complex(0, $x)`, which PowerShell's command-mode parsing
turned into a wrong number. Rebuilt with `-ArgumentList` and cos/sin, as
above.)

## 3. Measurements from the tests

Commands, from `build/tests` after `scripts\build.ps1 -Test` (434 tests
passed): `test_kernel.exe "[integrators]" -s`, `test_vehicle.exe
"[integrators]" -s`, `test_vehicle.exe "Drivetrain: braking to a stop locks
the wheels\, and the car stays stopped" -s`, `test_alloc.exe -s`; the values
are the tests' INFO lines. The oscillator: 1 kg, 1 N/m (w = 1 rad/s), released
at rest from a = 0.1 m; w = q - i v.

**One fixed step, h = 0.3 s.** w = (0.0955336, 0.029552), equal to R(0.3 i)
a to the digits printed and within the test's 1e-14; error measure 1.3926e-7
against |E(0.3 i)| a / sqrt(2) = 1.3926e-7 (test: within 1e-9 of itself).
Seven evaluations.

**Fixed steps over T = 10 s.**

| h [s] | steps | error | leading term a T sqrt((h^5/3600)^2 + (h^6/5040)^2) | error / leading |
|---|---|---|---|---|
| 0.1 | 100 | 2.78733e-9 | 2.78485e-9 | 1.0009 |
| 0.05 | 200 | 8.68805e-11 | 8.68609e-11 | 1.0002 |

Ratio 32.08 (theory 32, 32.06 with the phase term). The states equal R(ih)^N a
within 1e-13. Evaluations 7 per step.

**Continuous extension, one step, mid-step error.** h = 0.1: 2.85683e-10; h
= 0.05: 8.87233e-12; ratio 32.20. The same as part 2 times a = 0.1, to all
digits printed.

**Error control, relative tolerance 0, one call of step(10 s).**

| absolute tolerance | error | predicted 0.28697 T tol | ratio | steps | rejected |
|---|---|---|---|---|---|
| 1e-7 | 2.88656e-7 | 2.86969e-7 | 1.0059 | 40 | 3 |
| 1e-9 | 2.86824e-9 | 2.86969e-9 | 0.9995 | 100 | 3 |
| 1e-11 | 2.86504e-11 | 2.86969e-11 | 0.9984 | 250 | 4 |

Steps 250 / 40 = 6.25 against (10^4)^(1/5) = 6.31. The rejections are the
first tries, from h = 10 s down.

**Bouncing ball, 3 s in steps of 10 ms, events on the interpolant.** Worst
impact time error 2.2e-15 s (RK4 with re-integration: under the test's 1e-9);
2142 evaluations.

**Simple vehicle, half-pedal braking (the hand-calculation test).** RK4
5.378671862 m/s^2, 7200 evaluations over 1.8 s; Dormand-Prince 5.378671862
m/s^2, 1800 steps, 0 rejected, 12600 evaluations.

**Simple vehicle braked to a stop, 3.3 s.** RK4: largest |vx| over 2.0-2.5 s
6.9062e-4 m/s, over 2.5-3.0 s 1.64971e-4, at the end 4.36575e-7; 3300 steps.
Dormand-Prince: 6.9062e-4, 1.64970e-4, 4.36623e-7; 3300 steps, 0 rejected.
All wheels locked at zero with both.

**Double-wishbone sedan, 100 steps of 1 ms (throttle 0.3, steering 0.02).**
RK4: 100 steps, 400 evaluations; Dormand-Prince: 100 steps, 0 rejected, 700
evaluations (the seventh stage is not reused: the sedan has quaternions and
constraints). No allocation with either.
