# Comparison with independent codes

Plan task 5.3 (decision D46): the library's results compared with two codes
written independently of it.

| Script | Compares | With |
|---|---|---|
| `compare_pinocchio.py` | Tree dynamics on 40 random trees of every joint type: the accelerations of the articulated-body algorithm, the generalized forces of recursive Newton-Euler, the mass matrix, every body's placement. Tolerance 1e-8 relative | [Pinocchio](https://github.com/stack-of-tasks/pinocchio) |
| `compare_chrono.py` | 12 random closed mechanisms (trees of every joint type closed by spherical, revolute, universal, cylindrical, prismatic and distance constraints). At rest: accelerations, the closures' loads on each body, each joint's reaction (tolerance 1e-10). The motion from rest over 0.5 s: Chrono must converge to ours at its second order | [Project Chrono](https://projectchrono.org) (PyChrono) |

The C++ side, `mbd_crosscheck` (`main.cpp`, `export_trees.cpp`,
`export_mechanisms.cpp`), writes the models, the states and this library's
answers as JSON; the scripts rebuild the same models in the other code and
compare. Each script's header states the conventions it maps (frames, joint
types, the order of velocities), how the other code's numbers are obtained,
and the tolerances with their reasons. Two things matter when reading the
results:

- Pinocchio's mass matrix (CRBA) assumes joints numbered depth first; ours
  may be numbered in any order with parents first. The script adds the
  joints to Pinocchio depth first and maps the coordinates.
- Chrono's own accelerations (its assembly at the acceleration level) are
  good only to about 1e-4 and leave out the velocity products, and its
  integrators give accelerations and reaction forces no better. So the
  script solves Chrono's own equations at rest from the matrices Chrono
  assembles, and checks the motion by its convergence. The journal entry of
  7 November 2026 has the measurements.

## The environment

Both codes come from conda-forge. (The package called `pychrono` on PyPI is
an unrelated library; do not install it. Pinocchio's PyPI package, `pin`, has
no Windows wheels.) `environment.yml` pins the versions used, and NumPy's
BLAS to OpenBLAS: with MKL 2026.1.0, which conda chose on Windows, NumPy's
first matrix product stopped Python (exception 0xc06d007f).

```bash
conda env create -f crosscheck/environment.yml      # once
conda activate mbd-crosscheck
```

On the author's laptop, Miniforge is installed for the current user in
`C:\Users\margk\miniforge3`, not on the PATH; use its full path, for example
`C:\Users\margk\miniforge3\envs\mbd-crosscheck\python.exe`.

## Running it

```powershell
scripts\build.ps1 -Target mbd_crosscheck
build\crosscheck\mbd_crosscheck.exe trees trees.json 40 1            # count, seed
build\crosscheck\mbd_crosscheck.exe mechanisms mechanisms.json 12 1
<env python> crosscheck\compare_pinocchio.py trees.json                # a few seconds
<env python> crosscheck\compare_chrono.py mechanisms.json              # about two minutes
```

On Linux, as the CI job `crosscheck` does (`.github/workflows/ci.yml`): build
the target with the commands in `CLAUDE.md`, then run the same steps. Each
script prints its largest differences and exits with 1 if any is beyond its
tolerance.
