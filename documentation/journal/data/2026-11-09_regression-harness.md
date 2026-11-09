# The regression harness: measurements (task 5.5, D49)

Machine: the author's laptop (13th Gen Intel Core i7-13700H, Windows 11
Home), MSVC 19.50.35717, RelWithDebInfo, 9 November 2026. Not pinned: no
timings are compared here, the durations are for orientation.

## 1. The sensitivity probe

`build\tests\test_regression.exe "[sensitivity]" -s`: for each scenario, the
largest change of any channel over its tolerance (1e-8 of the channel's
largest magnitude) when the parameter named changes by a relative 1e-12 and
by 1e-6, against the unchanged run:

```
scenario             perturbed                                   1e-12         1e-6
four-bar             gravity                                  0.000693          693
heavy-top            gravity                                  0.000811          809
bushed-pendulum      gravity                                  0.000168          177
hmmwv-joints         gravity                                  5.95e-05         59.7
hmmwv-bushings       gravity                                  0.000238         59.7
sedan-drive          gravity                                  0.000997          997
sedan-modes          gravity                                   0.00513         25.8
wishbone-sweep       the upper ball joint's hardpoint          0.00068          682
lap-oval             the friction                             4.16e-05         41.1
hmmwv-joints-5s      gravity                                  6.56e-05         62.5
hmmwv-bushings-5s    gravity                                  7.17e-05         62.4
sedan-slalom         gravity                                   0.00099          996
All tests passed (24 assertions in 1 test case)
```

The two columns stand in a ratio near 1e6 (a linear response) for every
scenario but `sedan-modes`, whose 1e-12 column (5e-11 relative) is the
linearisation's finite-difference noise rather than a response.

## 2. The scenarios' durations

`build\tests\test_regression.exe "[regression]~[sensitivity]" --durations yes`
(the filter also selects the hidden long ones):

```
0.062 s: Regression: the four-bar
0.028 s: Regression: the heavy top
0.052 s: Regression: the bushed pendulum
0.432 s: Regression: the HMMWV with joints
0.230 s: Regression: the HMMWV with bushings
0.055 s: Regression: the sedan driving
0.003 s: Regression: the sedan's modes at rest
0.001 s: Regression: a double wishbone's kinematics
0.001 s: Regression: a lap of an oval
4.182 s: Regression: the HMMWV with joints for 5 s
3.098 s: Regression: the HMMWV with bushings for 5 s
1.173 s: Regression: the sedan's slalom
1.111 s: Regression: a change of physics by one part in a million is caught
All tests passed (27 assertions in 13 test cases)
```

Writing the references (`MBD_REGRESSION_UPDATE=1`): the fast ones in 1.4 s,
the long ones in 7.5 s. Before keeping them, every channel of every
reference was checked to have a largest magnitude above 1e-6 (a channel of
rounding noise alone would have a tolerance no other compiler could meet):
none was below.

## 3. The mutation tests

`python -I scripts\mutation_test.py --build "powershell -NoProfile
-ExecutionPolicy Bypass -File scripts/build.ps1 -Target
test_kernel,test_regression" --test build/tests/test_regression.exe --test
build/tests/test_kernel.exe` (the regression executable first):

```
1. every joint's velocity-product acceleration with its sign flipped: caught by test_regression (41 s)
2. gravity 1 % weaker in the recursive algorithms: caught by test_kernel (34 s)
3. the articulated-body algorithm without the velocity-product bias of its inertias: caught by test_kernel (33 s)
4. RK4's velocity weights permuted (still consistent: first order): caught by test_regression (39 s)
5. point coincidence without its velocity-product term: caught by test_regression (18 s)
6. a spherical joint's rotation increment applied in the parent's axes, not the child's: caught by test_regression (15 s)
7. a spring-damper's rate from the sum of its ends' velocities: caught by test_regression (13 s)
8. a spring-damper's force on its second body 1 % short (action and reaction unequal): caught by test_regression (10 s)
9. the tyres' peak force 2 % higher: caught by test_regression (11 s)
9 of 9 mutations caught
```

The sources were unchanged afterwards (`git status`) and built again.

The same, with only the kernel tests (`--test build/tests/test_kernel.exe`)
and only the seven mutations the regression scenarios had caught first
(`--only 1 --only 4 --only 5 --only 6 --only 7 --only 8 --only 9`):

```
1. every joint's velocity-product acceleration with its sign flipped: caught by test_kernel (45 s)
4. RK4's velocity weights permuted (still consistent: first order): caught by test_kernel (45 s)
5. point coincidence without its velocity-product term: caught by test_kernel (19 s)
6. a spherical joint's rotation increment applied in the parent's axes, not the child's: caught by test_kernel (26 s)
7. a spring-damper's rate from the sum of its ends' velocities: caught by test_kernel (34 s)
8. a spring-damper's force on its second body 1 % short (action and reaction unequal): caught by test_kernel (32 s)
9. the tyres' peak force 2 % higher: SURVIVED every test passed (45 s)
6 of 7 mutations caught
```

(Run by hand, outside CI, to see which tests each mutation reaches; the
tyres' code also has tests in `test_forces`, which the runner does not build;
whether they would catch this mutation was not tried.)
