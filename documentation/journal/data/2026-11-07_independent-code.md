# Comparison with independent codes: measurements (task 5.3, D46)

Machine: the author's laptop (13th Gen Intel Core i7-13700H, Windows 11
Home), MSVC 19.50.35717, RelWithDebInfo. Python 3.12.16 from Miniforge
26.9.4 (installed per user in `C:\Users\margk\miniforge3`), environment
`mbd-crosscheck`: Pinocchio 4.1.0, PyChrono 10.0.0, NumPy 2.5.4, all from
conda-forge; NumPy's BLAS OpenBLAS 0.3.34 (section 7). No timings are
compared; the durations given are wall-clock, for orientation only.

Sections 1 and 8 are the committed scripts' output. Sections 2 to 6 come from
scratch probes written in the session's scratch folder and not committed;
each is described well enough to repeat. The commands for sections 1 and 8,
from the repository root:

```powershell
scripts\build.ps1 -Target mbd_crosscheck
build\crosscheck\mbd_crosscheck.exe trees trees.json 40 1
build\crosscheck\mbd_crosscheck.exe mechanisms mechanisms.json 12 1
C:\Users\margk\miniforge3\envs\mbd-crosscheck\python.exe -I crosscheck\compare_pinocchio.py trees.json
C:\Users\margk\miniforge3\envs\mbd-crosscheck\python.exe -I crosscheck\compare_chrono.py mechanisms.json
```

The mechanisms took 44 draws for 12 (the rest had redundant equations or no
motion left). `compare_pinocchio.py` takes a few seconds (not timed),
`compare_chrono.py` 115 s.

## 1. The committed comparisons

```
Pinocchio 4.1.0; 40 trees from mbd_crosscheck trees (seed 1)
563 bodies; joints: cylindrical 79, fixed 71, free 79, planar 63, prismatic 67, revolute 82, spherical 62, universal 60
  aba         largest relative difference 2.141e-13
  rnea        largest relative difference 1.397e-15
  crba        largest relative difference 1.066e-15
  placements  largest relative difference 6.179e-16
All within 1e-08.
```

```
PyChrono 10.0.0; 12 mechanisms from mbd_crosscheck mechanisms (seed 1)
  mechanism  0 (5 bodies,  4 dof, distance): motion differs by 8.06e-08 / 2.02e-08 / 2.26e-12 m and 6.84e-08 / 1.71e-08 / 1.40e-12 in rotation (h, h/2, extrapolated)
  mechanism  1 (5 bodies, 11 dof, cylindrical+distance+spherical): motion differs by 1.82e-07 / 4.56e-08 / 8.48e-11 m and 1.72e-06 / 4.30e-07 / 1.33e-10 in rotation (h, h/2, extrapolated)
  mechanism  2 (7 bodies,  6 dof, prismatic+spherical+universal): motion differs by 2.17e-07 / 5.42e-08 / 1.10e-10 m and 3.12e-06 / 7.81e-07 / 5.93e-10 in rotation (h, h/2, extrapolated)
  mechanism  3 (8 bodies, 10 dof, universal): motion differs by 4.68e-07 / 1.17e-07 / 1.80e-10 m and 1.13e-06 / 2.82e-07 / 5.67e-10 in rotation (h, h/2, extrapolated)
  mechanism  4 (7 bodies,  6 dof, revolute+spherical): motion differs by 1.17e-07 / 2.92e-08 / 3.45e-11 m and 7.15e-07 / 1.79e-07 / 1.19e-10 in rotation (h, h/2, extrapolated)
  mechanism  5 (6 bodies, 12 dof, cylindrical): motion differs by 1.77e-07 / 4.44e-08 / 4.72e-11 m and 6.11e-07 / 1.53e-07 / 2.16e-10 in rotation (h, h/2, extrapolated)
  mechanism  6 (8 bodies, 31 dof, distance): motion differs by 2.66e-06 / 6.66e-07 / 5.79e-11 m and 1.09e-05 / 2.71e-06 / 4.31e-10 in rotation (h, h/2, extrapolated)
  mechanism  7 (4 bodies,  7 dof, revolute): motion differs by 2.42e-08 / 6.06e-09 / 2.80e-12 m and 2.05e-08 / 5.13e-09 / 2.44e-12 in rotation (h, h/2, extrapolated)
  mechanism  8 (7 bodies, 11 dof, spherical+prismatic+distance): motion differs by 1.80e-07 / 4.50e-08 / 2.49e-10 m and 5.47e-07 / 1.37e-07 / 9.16e-10 in rotation (h, h/2, extrapolated)
  mechanism  9 (4 bodies,  9 dof, distance+universal): motion differs by 7.00e-07 / 1.76e-07 / 1.33e-09 m and 2.16e-06 / 5.41e-07 / 6.16e-09 in rotation (h, h/2, extrapolated)
  mechanism 10 (3 bodies,  5 dof, spherical): motion differs by 6.71e-08 / 1.68e-08 / 7.07e-12 m and 3.44e-07 / 8.59e-08 / 1.97e-11 in rotation (h, h/2, extrapolated)
  mechanism 11 (4 bodies,  5 dof, distance+cylindrical): motion differs by 5.92e-07 / 1.48e-07 / 1.11e-10 m and 9.34e-07 / 2.34e-07 / 4.38e-10 in rotation (h, h/2, extrapolated)
  at rest, angular acceleration         largest relative difference 1.026e-12
  at rest, centre-of-mass acceleration  largest relative difference 4.800e-13
  at rest, closure loads                largest relative difference 1.397e-12
  at rest, joint reactions              largest relative difference 3.597e-13
All within tolerance (at rest 1e-10; motion converging to ours at second order).
```

Chrono's steps: h = 2e-4 s and h/2 = 1e-4 s. The rotation column is the
largest difference between entries of the two rotation matrices (about the
angle, for small differences). Our own motion (RK4 at 1e-4 s, projected)
changed by at most 5.0e-10 (mechanism 6) when its step was doubled; the
others by 2e-14 to 3e-11.

The mechanisms (bodies, the joints of the tree, the closures):

| | Bodies | Joints | Closures | Velocities | Equations | Freedom |
|---|---|---|---|---|---|---|
| 0 | 5 | fixed, planar, prismatic, revolute | distance | 5 | 1 | 4 |
| 1 | 5 | cylindrical, free, planar, universal | cylindrical, distance, spherical | 19 | 8 | 11 |
| 2 | 7 | fixed, planar, spherical | prismatic, spherical, universal | 18 | 12 | 6 |
| 3 | 8 | cylindrical, free, prismatic, revolute | universal | 14 | 4 | 10 |
| 4 | 7 | fixed, free, planar, prismatic, universal | revolute, spherical | 14 | 8 | 6 |
| 5 | 6 | cylindrical, free, revolute, spherical, universal | cylindrical | 16 | 4 | 12 |
| 6 | 8 | free, planar, revolute, universal | distance | 32 | 1 | 31 |
| 7 | 4 | cylindrical, free, universal | revolute | 12 | 5 | 7 |
| 8 | 7 | fixed, free, prismatic, spherical, universal | spherical, prismatic, distance | 20 | 9 | 11 |
| 9 | 4 | free, spherical, universal | distance, universal | 14 | 5 | 9 |
| 10 | 3 | cylindrical, fixed, free | spherical | 8 | 3 | 5 |
| 11 | 4 | prismatic, spherical | distance, cylindrical | 10 | 5 | 5 |

"Equations" counts the closures' equations; Chrono also has one link for
each joint of the tree that is not free, and the script checks that its
count of equations matches.

## 2. Pinocchio's mass matrix on trees not numbered depth first

Before the fix, `compare_pinocchio.py` gave CRBA differences up to 0.657
relative (ABA 2.1e-13, RNEA 1.4e-15, placements 6.2e-16), on 111 of 120
states. A probe built each tree as the script did, and compared, at the first
state's q, Pinocchio's `crba` with the matrix whose columns are Pinocchio's
own `rnea(q, 0, e_k)` with gravity off, and ours with the same:

```
tree  0: crba-vs-rnea 7.70e+00  ours-vs-rnea 1.07e-14  [all eight joint types]
tree  1: crba-vs-rnea 2.63e+00  ours-vs-rnea 5.33e-15  ['cylindrical', 'free', 'prismatic', 'revolute']
tree  2: crba-vs-rnea 2.56e+00  ours-vs-rnea 1.78e-14
tree  3: crba-vs-rnea 3.65e+00  ours-vs-rnea 7.11e-15
tree  4: crba-vs-rnea 1.33e-15  ours-vs-rnea 3.11e-15  ['planar', 'spherical', 'universal']
tree  5: crba-vs-rnea 4.92e-01  ours-vs-rnea 8.88e-16
tree  6: crba-vs-rnea 2.32e+00  ours-vs-rnea 7.11e-15
tree  7: crba-vs-rnea 2.07e+00  ours-vs-rnea 2.13e-14
tree  8: crba-vs-rnea 3.75e-01  ours-vs-rnea 1.78e-15
tree  9: crba-vs-rnea 4.48e+00  ours-vs-rnea 3.55e-15
tree 10: crba-vs-rnea 2.40e-01  ours-vs-rnea 6.66e-16
tree 11: crba-vs-rnea 5.16e+00  ours-vs-rnea 3.55e-15
```

On tree 1, `crba` with `Convention.WORLD` differed from the RNEA columns by
8.69, with `Convention.LOCAL` (the default) by 2.63. Chains and stars of
single and composite joints built from scratch (RZ, PZ, free, spherical,
RZ-RZ-RZ, composites of RZ and RX, of RZ and PZ, of PX, PY and RZ, in every
order with each other and with free and spherical joints) all agreed to
7e-15. Classifying the 40 trees by whether every joint's subtree occupies the
joint ids just after it:

```
depth-first      agree   3
not depth-first  differ 37
```

## 3. Chrono's assembly at the acceleration level, on a pendulum

The pendulum: mass 2 kg, centre of mass 0.7 m from a revolute joint about the
world Z axis at the origin, inertia about the centre of mass diag(0.03, 0.04,
0.05) kg m^2, gravity 9.81 m/s^2 along -Y, at 0.4 rad above the horizontal;
`ChBody` at the centre of mass, `ChLinkLockRevolute`, `ChSolverSparseQR`
unless said. Closed form: alpha = -m g L cos(theta) / (I_zz + m L^2) =
-12.2814 rad/s^2; centre of mass acceleration alpha x r - omega^2 r.

Turning at 3 rad/s, after `DoAssembly(ACCELERATION)` (and `FULL`, the same):

```
exact [ -2.45486    -10.37168457   0.        ] -12.281409409354977
SparseQR ACCELERATION lin err 5.80e+00  ang err 7.89e-10  a=[ 3.34782427 -7.91834901  0.        ]
SparseLU ACCELERATION lin err 5.80e+00  ang err 9.87e-11
```

The difference, (5.80, 2.45) m/s^2, is omega^2 r = 9 x (0.645, 0.273):
the centripetal acceleration, left out. `ChLinkMateRevolute` gave the same
5.80. At rest, the same assembly is exact to rounding: angular 1.4e-16,
linear 4.5e-16 relative; a mass on a spring (k = 50 and 5000 N/m) 0 and
1.5e-16.

The same pendulum at rest, its joint opened along X by a gap before the
assembly:

```
gap 0e+00 m: angular acceleration error 1.78e-15 rad/s^2
gap 1e-16 m: angular acceleration error 5.88e-05 rad/s^2
gap 1e-15 m: angular acceleration error 5.29e-04 rad/s^2
gap 1e-14 m: angular acceleration error 5.29e-03 rad/s^2
gap 1e-12 m: angular acceleration error 5.29e-01 rad/s^2
```

5.29e-4 rad/s^2 per 1e-15 m is a gap over (1e-6 s)^2 times a factor of
order one: the step's right-hand side -C / dt, differentiated over dt.
The matrix blocks of the system descriptor after that assembly (gap 1e-15 m,
`WriteMatrixBlocks`, 13 significant digits) confirm it: H the mass matrix
(2, 2, 2, 0.03, 0.04, 0.05), Cq five rows, f = (0, -1.962e-5, 0, 0, 0, 0), that
is the step 1e-6 s times the weight, and b = (-9.992e-10, -4.9e-26, 0, 0, 0),
the gap over the step.

The first comparison of the 12 mechanisms at rest used these assembled
accelerations and the links' `GetReaction1/2`: differences from 1.5e-5 to
2.7e-4 relative, while the motions agreed to 1e-10.

## 4. Chrono's integrators started from rest or not, on the pendulum and a spring

theta at 1 s against RK4 at 1e-5 s on the pendulum's equation (wrapped to
plus or minus pi), HHT with alpha = 0, Newton tolerances 1e-13, no step
control.

Turning at 3 rad/s, stepping straight away (no assembly):

```
HHT          dt 2.00e-03  err +2.123e-02
HHT          dt 1.00e-03  err +1.057e-02  ratio   2.01
HHT          dt 5.00e-04  err +5.273e-03  ratio   2.00
HHT          dt 2.50e-04  err +2.633e-03  ratio   2.00
HHT          dt 1.25e-04  err +1.316e-03  ratio   2.00
TRAPEZOIDAL  dt 2.00e-03  err +1.149e-02  ... ratios 1.99 to 2.00, 7.236e-04 at 1.25e-4
NEWMARK      dt 2.00e-03  err +1.926e-02  ... ratios 2.00 to 2.01, 1.197e-03 at 1.25e-4
```

At rest: HHT 1.675e-3, 8.315e-4, 4.141e-4, 2.066e-4, 1.032e-4 (ratios 2.0);
trapezoidal 3.537e-2 to 2.256e-3 (ratios 2.0). A mass of 2 kg on a 50 N/m
spring (`ChLinkTSDA`, from 0.3 m, gravity off), at 1 s against x0 cos(omega
t): HHT with alpha 0, -0.1 and -0.3 all at first order (-7.227e-4, -3.605e-4,
-1.800e-4 for alpha 0 at 1e-3, 5e-4, 2.5e-4 s).

With `DoAssembly(ACCELERATION)` before the first step:

```
spring  alpha 0   dt 1.0e-03 err -1.558e-06   5.0e-04 +6.892e-07   2.5e-04 +1.251e-06
pendulum at rest  1.437e-05, 2.724e-06, -5.102e-07, -1.359e-06, -1.576e-06
pendulum at 3 rad/s  7.262e-05, 2.090e-06, -1.530e-05, -1.961e-05, -2.069e-05
```

The spring's state just before stepping, assembled and with the acceleration
set by hand:

```
assembly   err +1.251e-06   vel (-7.5e-06, 0, 0)  acc (-7.5, 0, 0)  step 1e-06
exact a0   err -1.873e-07   vel (0, 0, 0)         acc (-7.5, 0, 0)
```

With the acceleration set by hand and the velocity left at rest, at 0.25,
0.5 and 1 s: +7.41e-7, +9.35e-7, -3.00e-6 (dt 1e-3); +1.85e-7, +2.34e-7,
-7.49e-7 (5e-4); +4.63e-8, +5.84e-8, -1.87e-7 (2.5e-4): second order, the
same with `SparseLU` and with `ChSystemSMC`.

The pendulum at rest, assembled and then its velocities put back to zero:

```
alpha +0.0  dt 2.00e-03 err +1.602e-05
alpha +0.0  dt 1.00e-03 err +4.374e-06 ratio  3.66
alpha +0.0  dt 5.00e-04 err +1.140e-06 ratio  3.84
alpha +0.0  dt 2.50e-04 err +2.907e-07 ratio  3.92
alpha +0.0  dt 1.25e-04 err +7.350e-08 ratio  3.95
alpha -0.1  ratios 3.80, 3.90, 3.95, 3.97 (1.059e-07 at 1.25e-4)
```

A first attempt at putting the velocities back read them with
`GetLinVel()` and saved the result; PyChrono returns a reference to the
body's own vector, so the saved value changed with the body and the
"restore" restored nothing (the check of how far the assembly had moved
them printed 0). Copying the three numbers fixed it (8.6e-6 m/s moved).

## 5. Chrono's accelerations and reactions along the motion

Mechanism 2 of section 1, from rest to 0.1 s at h = 4e-4, 2e-4, 1e-4 and
5e-5 s, the largest change between successive steps, with the largest value
for scale:

```
HHT alpha 0.0:  positions     1.36e-08 3.42e-09 8.55e-10  ratios 3.98 4.00  scale 1.75e+00
HHT alpha 0.0:  accelerations 5.35e-03 3.53e-02 6.24e-01  ratios 0.15 0.06  scale 2.27e+01
HHT alpha 0.0:  reactions     4.29e-02 1.70e-01 6.54e+00  ratios 0.25 0.03  scale 1.91e+02
HHT alpha -0.2: positions     1.02e-08 2.56e-09 6.41e-10  ratios 3.99 3.99  scale 1.75e+00
HHT alpha -0.2: accelerations 3.63e-03 1.82e-03 8.93e-04  ratios 2.00 2.03  scale 2.29e+01
HHT alpha -0.2: reactions     1.07e-04 7.63e-06 6.37e-05  ratios 13.99 0.12  scale 1.97e+02
```

## 6. A comparison that can fail

Mechanisms 0 and 7 with one entry of body 2's inertia (I_zz about the centre
of mass) raised by one part in a million in what Chrono builds, our answers
unchanged:

```
  mechanism  0 (5 bodies,  4 dof, distance): motion differs by 8.06e-08 / 2.02e-08 / 2.31e-12 m and 6.84e-08 / 1.71e-08 / 1.42e-12 in rotation (h, h/2, extrapolated)
  mechanism  7 (4 bodies,  7 dof, revolute): motion differs by 2.34e-08 / 5.27e-09 / 7.88e-10 m and 1.99e-08 / 4.46e-09 / 6.68e-10 in rotation (h, h/2, extrapolated)
  at rest, angular acceleration         largest relative difference 1.926e-09
  at rest, centre-of-mass acceleration  largest relative difference 7.981e-10
  at rest, closure loads                largest relative difference 6.756e-10
  at rest, joint reactions              largest relative difference 9.817e-10
FAILED: 6 differences:
  mechanism 7: angular acceleration at rest differs by 1.926e-09
  mechanism 7: centre-of-mass acceleration at rest differs by 7.981e-10
  mechanism 7: closure loads at rest differs by 6.756e-10
  mechanism 7: joint reactions at rest differs by 9.817e-10
  mechanism 7: positions: extrapolated difference 7.88e-10 against 5.27e-09 at h/2
  mechanism 7: rotations: extrapolated difference 6.68e-10 against 4.46e-09 at h/2
```

Mechanism 0 is untouched because its body 2 is fixed to the ground. With the
first at-rest tolerance, 1e-8, the at-rest part let this through; it is now
1e-10.

## 7. NumPy with MKL on Windows

Conda resolved the environment's BLAS to MKL (`libblas 3.11.0-12_mkl`, `mkl
2026.1.0`). The first matrix product, even of two 3 x 3 arrays and before
PyChrono was imported, ended Python:

```
Windows fatal exception: code 0xc06d007f
  File "<string>", line 3 in <module>
```

0xc06d007f is a delay-loaded DLL's missing function. `compare_pinocchio.py`
had run because it never reaches BLAS. Neither Chrono nor PyChrono depends on
MKL (their conda records list eigen, the MSVC runtime and Python). After
`conda install -n mbd-crosscheck "libblas=*=*openblas"` (MKL removed,
libopenblas 0.3.34 added) a 50 x 50 product and inverse were right, and both
comparisons ran. `environment.yml` pins `libblas=*=*openblas`; a dry run of
`conda env create` with it resolves to OpenBLAS on win-64.

## 8. The full test suite

`scripts\build.ps1 -Test` after the changes: 508 of 508 passed (29.5 s), the
build without a warning (the MSVC warning C4458 in `tests/kernel/iftomm.hpp`
fixed, see the entry).

## 9. CI

Run 42365611807 on `643b5e6`, read from the public API
(`/repos/poguitos/multibody-pogos/actions/runs/42365611807/jobs`): windows,
linux-sanitizers, kernel-no-malloc, docs and crosscheck all `success`. The
crosscheck job's steps (UTC, 7 November 2026):

```
Install Pinocchio and PyChrono   10:50:28 to 10:50:54
Configure                        10:50:54 to 10:51:07
Build the exporter               10:51:07 to 10:53:04
Compare with Pinocchio           10:53:04 to 10:53:05
Compare with Project Chrono      10:53:05 to 10:54:27
```

The scripts' output on Linux was not read (the log download needs
authentication).
