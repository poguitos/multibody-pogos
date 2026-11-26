# Data: Phase 7, the suspensions (task 7.6)

The author's laptop (Windows 11, MSVC, RelWithDebInfo through `scripts\build.ps1`,
one compiler). Power: part 1's suite on battery (Win32_Battery.BatteryStatus 1),
110.7 s for 619 tests; part 2's suites on mains (status 2): 82.3 s for 630
tests before the two references were written again (those two failing),
74.7 s for 630 tests after, all passing.

## Part 1: hysteresis, the dyno, shaped curves

Command: `build\tests\test_forces.exe "[hysteresis]" -s`, the INFO lines
(15 cases, 3089 assertions, all passing):

```
  worst slope difference 4.99947e-05 N/m, worst force difference 2.40038e-07 N
  worst relative round trip 8.36207e-15
  largest force 2666.32 N, worst difference from the closed form 7.76794e-08 N
  loop against the steady state: worst 4.5977e-06 N; in phase 1415.22 N (closed
  form 1415.22), in quadrature 2252.39 N (2252.39); energy 176.902 J (176.902)
  2 Hz: 48.5809 J against 48.5809 J
  15.9155 Hz: 196.35 J against 196.35 J
  50 Hz: 113.5 J against 113.5 J
  rig's work 195.478 J, damper's loss 193.658 J, mount's energy 1.81958 J
  Radau IIA at h = 10 tau: worst 1.74778e-07 N of 3141.53 N
  after 5 s: -0.00981 m, Maxwell state 6.82689e-18 m
  worst 2.43073e-06 N, allowed 0.000230292 N; largest |z| 0.00249814 m (z_u 2.
  5e-3)
  worst 5.53854e-15 m in z
  worst difference 8.15675e-10 N
  worst 6.07682e-08 N m, allowed 5.75737e-06 N m
```

The energy balance with the tabulated damper, its residual printed after:
rig's work minus the damper's loss and the mount's energy = -1.66325e-4 J
(the estimate (w h)^2 / 12 of 200 J: 1.6e-4 J; allowed 1e-3 J).

### Radau IIA's steady state on the Maxwell element, from its coefficients

The script (Python, NumPy, environment mbd-crosscheck) and its output. The
system form (the rig's velocity at the stages from the driver's
acceleration, exact at each step's start) is the dyno's; the test measured
1.74778e-07 N.

```python
# Radau IIA (3 stages) on s' = u'(t) - s / tau, u = U sin(w t): the discrete
# steady-state response to the forcing, against the exact one. The forcing
# g(t) = U w cos(w t) = Re(U w e^{i w t}); with s_n = S e^{i w t_n}, the
# stages Y = s_n 1 + h A (lam Y + g(t_n + c h)), s_{n+1} = s_n + h b^T(lam Y + g).
import numpy as np
s6 = np.sqrt(6.0)
A = np.array([[(88-7*s6)/360, (296-169*s6)/1800, (-2+3*s6)/225],
              [(296+169*s6)/1800, (88+7*s6)/360, (-2-3*s6)/225],
              [(16-s6)/36, (16+s6)/36, 1/9]])
b = A[2]
c = np.array([(4-s6)/10, (4+s6)/10, 1.0])
tau, U, f = 1e-4, 0.025, 10.0
k = 2e7
w = 2*np.pi*f
for N in (100, 200):
    h = 1.0/(f*N)
    lam = -1.0/tau
    z = h*lam
    # forcing at stages relative to e^{i w t_n}: G = U w e^{i w c h}
    G = U*w*np.exp(1j*w*c*h)
    # Y = S 1 + h A (lam Y + G) -> (I - z A) Y = S 1 + h A G
    M = np.eye(3) - z*A
    # s_{n+1} = S + h b^T (lam Y + G) = S e^{i w h}
    # Y = M^-1 (S 1 + h A G): S e^{iwh} = S + z b^T M^-1 (S 1 + h A G) + h b^T G
    one = np.ones(3)
    coefS = 1 + z*b @ np.linalg.solve(M, one)
    rhs = z*b @ np.linalg.solve(M, h*A@G) + h*b@G
    S = rhs/(np.exp(1j*w*h) - coefS)
    S_exact = U*w/(1j*w - lam)
    print(N, "h/tau", h/tau, "rel error", abs(S-S_exact)/abs(S_exact), "force err N", k*abs(S-S_exact))

print("system form: v' = a(t) with v_n exact (the driver's projection), s' = v - s / tau")
for N in (100, 200):
    h = 1.0/(f*N)
    lam = -1.0/tau
    z = h*lam
    # relative to e^{i w t_n}: v_n = U w (u' = Re(U w e^{i w t}) taking u = Re(-i U e^{iwt})), a = i w U w e^{i w t}
    vn = U*w
    a = 1j*w*U*w*np.exp(1j*w*c*h)
    V = vn*np.ones(3) + h*A@a
    M = np.eye(3) - z*A
    one = np.ones(3)
    coefS = 1 + z*b @ np.linalg.solve(M, one)
    rhs = z*b @ np.linalg.solve(M, h*A@V) + h*b@V
    S = rhs/(np.exp(1j*w*h) - coefS)
    S_exact = U*w/(1j*w - lam)
    print(N, "rel error", abs(S-S_exact)/abs(S_exact), "force err N", k*abs(S-S_exact))
```

```
100 h/tau 10.0 rel error 8.853900933145095e-09 force err N 2.7814801090432652e-05
200 h/tau 5.0 rel error 5.293831639755854e-10 force err N 1.663073431450094e-06
system form: v' = a(t) with v_n exact (the driver's projection), s' = v - s / tau
100 rel error 5.563883621399157e-11 force err N 1.7479110890001813e-07
200 rel error 3.3262724780324626e-12 force err N 1.0449586916282395e-08
```

## Part 2: the suspensions

Command: `build\tests\test_vehicle.exe "[suspensions]" -s`, the INFO lines
(11 cases, all passing; the bar's relative difference from 4 k_arb, printed
after: 4.79482e-10):

```
  double wishbone, coil-over: motion ratio -0.989646 from the geometry, -0.
  989646 from virtual work; difference 1.11022e-16
  double wishbone, pushrod: motion ratio -0.703576 from the geometry, -0.703576
  from virtual work; difference -1.11022e-16
  McPherson: motion ratio -0.888835 from the geometry, -0.888835 from virtual
  work; difference 0
  five-link: motion ratio -1.00239 from the geometry, -1.00239 from virtual
  work; difference -8.88178e-16
  trailing arm: motion ratio -1.11111 from the geometry, -1.11111 from virtual
  work; difference -2.22045e-16
  solid axle: motion ratio -2 from the geometry, -2 from virtual work;
  difference -4.44089e-16
  arm's turn 3.09264 rad/m, rocker per arm turn 3.25, unit per arm turn -0.2275
  m/rad; motion ratio -0.703576
  roll rate 1.25 rad/m; centre at y -6.76075e-18, z 0.05 m; its distance from
  the rod's line 2.77556e-17 m
  double wishbone, coil-over: largest acceleration as placed 0.0115147, statics
  1 iterations, the chassis moved 2.79663e-06 m; Static equilibrium at t = 0 s:
  found.
  double wishbone, pushrod: largest acceleration as placed 0.0283242, statics 2
  iterations, the chassis moved 8.58781e-06 m; Static equilibrium at t = 0 s:
  McPherson: largest acceleration as placed 0.00310967, statics 1 iterations,
  the chassis moved 1.05498e-06 m; Static equilibrium at t = 0 s: found.
  Degrees of freedom 14: 7 with stiffness, 7 neutral.
  five-link: largest acceleration as placed 0.000604408, statics 1 iterations,
  the chassis moved 1.03725e-06 m; Static equilibrium at t = 0 s: found.
  trailing arm: largest acceleration as placed 0.00254979, statics 1
  iterations, the chassis moved 9.46189e-07 m; Static equilibrium at t = 0 s:
  solid axle: largest acceleration as placed 0.00287801, statics 1 iterations,
  the chassis moved 3.80824e-06 m; Static equilibrium at t = 0 s: found.
  sports car: largest acceleration as placed 0.0190684, statics 1 iterations,
  the chassis moved 5.67581e-06 m; Static equilibrium at t = 0 s: found.
  hatchback: largest acceleration as placed 0.0215127, statics 2 iterations,
  the chassis moved 9.7267e-06 m; Static equilibrium at t = 0 s: found.
  the half's turn -1.18948 rad/m, the bar 21203.5 N m/rad; roll stiffness
  120000 N/m against 4 k_arb = 120000
  largest difference 0
  chassis drift over 1 s: 2.2294e-14 m
  chassis drift over 1 s: 3.77822e-15 m
  chassis drift over 1 s: 1.58326e-12 m
```

## The regression scenarios that moved

Old (the references before task 7.6, `git show HEAD:...`) against new, at
the first row (50 ms), at 0.35 s (before the manoeuvre) and at the end:

```
sports-car-turn
  chassis x [m]     t=0.050000000000000037 old 0.000165556  new 2.74395e-08  | t=0.35000000000000026 old 0.00470919   new 0.00319366   | t=1.9999999999998905 old 3.83035      new 3.88263
  chassis z [m]     t=0.050000000000000037 old 0.299438     new 0.296566     | t=0.35000000000000026 old 0.326465     new 0.296454     | t=1.9999999999998905 old 0.314644     new 0.295146
  chassis vx [m/s]  t=0.050000000000000037 old 0.00350664   new 7.08125e-07  | t=0.35000000000000026 old 0.141571     new 0.139582     | t=1.9999999999998905 old 4.7445       new 4.82153
  FL Fz [N]         t=0.050000000000000037 old 4596.45      new 3755.16      | t=0.35000000000000026 old 3331.93      new 3621.57      | t=1.9999999999998905 old 3348.88      new 3358.68
  RL Fz [N]         t=0.050000000000000037 old 4213.88      new 3313.31      | t=0.35000000000000026 old 3140.49      new 3330         | t=1.9999999999998905 old 3571.1       new 3566.43
  FL toe [rad]      t=0.050000000000000037 old 0.00572145   new 1.28432e-06  | t=0.35000000000000026 old 0.0278633    new 0.0012227    | t=1.9999999999998905 old 0.132983     new 0.117193
  RL toe [rad]      t=0.050000000000000037 old 0.00744549   new 1.06097e-06  | t=0.35000000000000026 old 0.0294318    new -0.00220569  | t=1.9999999999998905 old 0.0712517    new 0.0493973
  FL camber [rad]   t=0.050000000000000037 old 0.000685106  new 1.83712e-07  | t=0.35000000000000026 old 0.0011546    new 0.000168661  | t=1.9999999999998905 old -0.000724547 new -0.00132338
  RL camber [rad]   t=0.050000000000000037 old 0.000841366  new 1.51765e-07  | t=0.35000000000000026 old 0.00107375   new -0.000336246 | t=1.9999999999998905 old -0.000276597 new -0.00335188
hatchback-brake
  chassis x [m]     t=0.050000000000000037 old 0.000469042  new 1.78549e-08  | t=0.35000000000000026 old 0.00488325   new 0.00162217   | t=1.9999999999998905 old 0.42235      new 0.407967
  chassis z [m]     t=0.050000000000000037 old 0.412897     new 0.297671     | t=0.35000000000000026 old 0.416629     new 0.297638     | t=1.9999999999998905 old 0.414049     new 0.29834
  chassis vx [m/s]  t=0.050000000000000037 old 0.0217974    new 4.62562e-07  | t=0.35000000000000026 old 0.0754316    new 0.0672895    | t=1.9999999999998905 old 0.00647037   new 0.000633889
  FL Fz [N]         t=0.050000000000000037 old 0            new 3830.65      | t=0.35000000000000026 old 3295.15      new 3725.13      | t=1.9999999999998905 old 3801.14      new 3800.19
  RL Fz [N]         t=0.050000000000000037 old 4498.28      new 2551.14      | t=0.35000000000000026 old 2846.2       new 2635.5       | t=1.9999999999998905 old 2481.51      new 2529.2
  FL toe [rad]      t=0.050000000000000037 old 0.0195776    new 9.26671e-07  | t=0.35000000000000026 old 0.0163689    new 0.000116466  | t=1.9999999999998905 old -0.0304249   new -0.0405878
  RL toe [rad]      t=0.050000000000000037 old -8.60824e-19 new -8.85053e-21 | t=0.35000000000000026 old -1.10531e-17 new 1.51226e-19  | t=1.9999999999998905 old -0.000984036 new -0.000913371
  FL camber [rad]   t=0.050000000000000037 old 0.0614758    new 3.56219e-06  | t=0.35000000000000026 old 0.0528673    new 0.000447034  | t=1.9999999999998905 old 0.0375092    new 0.000658513
  RL camber [rad]   t=0.050000000000000037 old -1.7823e-18  new 2.7243e-19   | t=0.35000000000000026 old -1.1555e-17  new 3.33049e-18  | t=1.9999999999998905 old -0.000385322 new 0.000191625
```

`wishbone-sweep`, written again by the update, had moved by at most
1.3e-15 (camber 6.4e-16, toe 1.3e-15, height 1.1e-16): restored, not
committed.

## The sensitivity probe after the update

Command: `build\tests\test_regression.exe "[sensitivity]" -s` (columns: the
scenario, the parameter changed, a 1e-12 change over the tolerance, a 1e-6
change over the tolerance, the worst channel):

```
four-bar             gravity                                  0.000693          693  rocker rate [rad/s]
heavy-top            gravity                                  0.000811          809  x [m]
bushed-pendulum      gravity                                  0.000168          177  link 2 x [m]
hmmwv-joints         gravity                                  5.95e-05         59.7  pitch [rad]
hmmwv-bushings       gravity                                  0.000238         59.7  pitch [rad]
sedan-drive          gravity                                  0.000992          992  chassis vz [m/s]
sedan-modes          gravity                                    0.0102         66.5  damping ratio
sports-car-turn      gravity                                    0.0108     1.08e+04  chassis vz [m/s]
sedan-rough-road     gravity                                   0.00061          610  chassis vz [m/s]
hatchback-brake      gravity                                    0.0031     3.11e+03  chassis vz [m/s]
wishbone-sweep       the upper ball joint's hardpoint         0.000681          682  camber [rad]
lap-oval             the friction                             4.16e-05         41.1  lap time [s]
hmmwv-joints-5s      gravity                                  6.56e-05         62.5  pitch [rad]
hmmwv-bushings-5s    gravity                                  7.17e-05         62.4  pitch [rad]
sedan-slalom         gravity                                  0.000988          990  chassis vz [m/s]
```
