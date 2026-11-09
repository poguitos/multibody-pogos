# The Python interface (task 6.1, D52): measurements

Machine: the author's laptop (13th Gen Intel Core i7-13700H, Windows 11
Home), MSVC 19.50.35717, RelWithDebInfo, 9 November 2026. Python 3.12.16,
NumPy 2.5.4 (OpenBLAS), pytest 9.2.0, from the conda environment of
`python/environment.yml`. pybind11 3.1.0 (commit 97bf890).

Each value below was printed by a script that rebuilt the test's model
from `python/tests/models.py` (scripts in the session's scratchpad, not
kept). The bounds in the tests are set against these values.

## 1. The pendulum (test_pendulum.py)

1 kg on 0.8 m, released at 1 rad from rest: K(sin 0.5) =
1.6749939160926133, so T = 4 K / sqrt(g / L) = 1.9133028352300083 s, 1.0663
times the small-angle period. Dormand-Prince stepping T/4 at a time, the
angle against 0, -1, 0, 1 rad and the speed through the bottom against
sqrt(2 g L (1 - cos 1)):

```
tol 1e-10: angle errors ['-1.20e-12', '1.46e-10', '8.21e-11', '-2.92e-10'], speed error -1.68e-10, steps 185
tol 1e-12: angle errors ['4.08e-14', '1.44e-12', '6.45e-13', '-2.88e-12'], speed error -1.73e-12, steps 455
```

## 2. Constraints and forces (test_constraints_and_forces.py)

The oscillator, M = 2 kg, K = 50 N/m, from 0.1 m at rest, Dormand-Prince at
1e-12 to t = 1.3 s, against 0.1 cos(5 t):

```
linear_spring  position error -8.40e-13, velocity error 7.61e-13
joint_spring   position error -8.40e-13, velocity error 7.60e-13
python_spring  position error -8.40e-13, velocity error 7.60e-13
```

The same through the force callback (test_analyses.py): position error
-8.40e-13.

The parallelogram (crank 0.3 m, coupler 0.8 m), released at q = (1, -1, 1),
RK4 at 1 ms for 1 s:

```
parallelogram: max |q0 + q1| 9.54e-16, max |q2 - q0| 1.89e-15, range of q0 5.142, max residual 3.89e-16
```

The range is pi + 2 = 5.1416: from 1 rad above the horizontal down and up
to -(pi + 1).

## 3. The analyses (test_analyses.py)

```
assembly: q - (1,-1,1) = [0. 0. 0.] residual 0.0
statics: error -4.693467836602849e-13 residual 1.1734613281078051e-11 iterations 20
linearize: w - 5 = -6.689049314445583e-11  zeta - 0.2 = 2.6755542226197804e-12  Re + 1 = 2.220446049250313e-16
kinematics: q - 0.3 = 0.0  v - 1 = 0.0  a = 0.0
event: time error 4.996003610813204e-16  height error at 0.6 s -2.4424906541753444e-15  min height 0.0021088939352977066
```

Statics' 20 iterations: each Newton step is limited to max_translation =
0.02 m, and the rest position is 0.392 m away.

## 4. State elements and the trace (test_state_elements.py)

```
lag error -2.1460611066004276e-12
energy drift 1.0107470416187425e-12 E -4.240292496453161
```

The lag: x' = (2 - x) / 0.2 for 0.5 s, RK4 at 1 ms, against
2 (1 - e^(-2.5)). The energy: the pendulum's kinetic plus potential energy
from the trace, over 500 steps of RK4 at 1 ms, against -g 0.8 cos 1.

## 5. The suites

```
scripts\build.ps1 -Test     100% tests passed, 0 tests failed out of 532
```

That is 527 C++ tests (the template stub's three removed) and 5 Python test
files with 32 tests. CI run 42630744159 on `668b992` (the first part)
passed all five jobs.
