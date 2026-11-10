# Data: Phase 6, packaging (task 6.2)

Raw measurements behind the journal entry `2026-11-10_phase-6-packaging.md`.

## The install on the laptop

An empty virtual environment made from Miniforge's base Python (3.14.10;
`pip list` showed only pip 26.2.2), then
`scripts\build.ps1 -Install -Python <venv>\Scripts\python.exe` (pip install
-v --editable, `CMAKE_GENERATOR=Ninja`, MSVC 19.50.35717, CMake 4.2.0,
scikit-build-core 1.1.2).

- Started 09:48:04, ended 09:51:45: 3 min 41 s, the downloads of Eigen,
  pybind11 and spdlog included (Catch2 was not fetched: no line of the log
  names it).
- 58 build steps. `cl.exe` processes counted with `tasklist` four times
  during the compilation (at steps 5 and 19): 1, 1, 1, 1; and three times
  during the configuration: 0, 0, 0.
- The build folder's cache: `BUILD_TESTING=OFF`, `CMAKE_BUILD_TYPE=Release`,
  `MBD_BUILD_PYTHON=ON`, `MBD_COMPILE_JOBS=1`, generator Ninja.
- `build-python/cp314-cp314-win_amd64/`: 726 MB (`du -sh`).
- The editable wheel: `multibody-0.6.0-cp314-cp314-win_amd64.whl`, 668,768
  bytes. pip installed NumPy 2.5.4 with it.
- `multibody.__file__`: `C:\dev\multibody\python\multibody\__init__.py` (the
  source folder); `multibody._mbd.__file__`: the environment's
  `Lib\site-packages\multibody\_mbd.cp314-win_amd64.pyd`.
- `importlib.metadata.version("multibody")` and `multibody.__version__`:
  0.6.0 and 0.6.0.
- With pytest 9.2.0 added: `python -m pytest python/tests`: 33 passed in
  0.23 s.
- A second install with nothing changed: 16 s, no file compiled (the build
  ran "Re-checking globbed directories" and "Copying the multibody package's
  Python files").

## The guard against an installed package

From `build\python`, with the suite's interpreter (mbd-python) and
`PYTHONPATH` set there:

- `MBD_TEST_PACKAGE_DIR` set to the build's own package: `test_interface.py`
  9 passed, 1 skipped (the installed-version test: "multibody is not
  installed with pip").
- `MBD_TEST_PACKAGE_DIR` set to another folder: pytest stopped with the
  message, exit status 4.

## CI

| Commit | Job | Result |
|---|---|---|
| `f583c86` | package, Windows (Python 3.14) | passed in 5 min 22 s: plain `pip install -e .`, the tests, the wheel `multibody-0.6.0-cp314-cp314-win_amd64.whl` (668 kB) tested in an empty environment with NumPy 2.5.4 |
| `f583c86` | package, Linux (Python 3.11, NumPy 2.0) | the install failed after 91 s; the log needs a login |
| `60ee42d` | package, Linux | with the library position-independent: installed in 117 s; the tests failed within a second, the cause again only in the log |
| `254b403` | package, Linux | installed in 103 s; the annotations: `test_a_parallelogram_stays_a_parallelogram` failed, `assert np.float64(4.019842680946795e-09) < 3e-09` (max \|q2 - q0\|), 1 failed, 32 passed in 0.18 s |
| `6522fd7` | all seven jobs | passed (CI run 42790138563), the package job on both platforms included |
| `2537513` | all seven jobs | passed (CI run 42793160687); the wheels: `multibody-0.6.0-cp314-cp314-win_amd64.whl`, 668 kB, built with Visual Studio 18 2026 (plain pip's choice on Windows), tested with NumPy 2.5.4; `multibody-0.6.0-cp311-cp311-linux_x86_64.whl`, 1136 kB, built with Ninja, tested with NumPy 2.0.2 |

## The parallelogram's departure from its branch, on Windows

`parallelogram_probe.py` (the test's model and run: q = (1, -1, 1), RK4
1 ms for 1 s, 1001 samples), in the package's Release build and in the
suite's RelWithDebInfo build; the two printed the same digits:

```
q0 + q1: max 9.541e-16 at t = 0.291, theta = -0.0091, sin = -9.14e-03; where |sin theta| >= 0.5: max 2.220e-16 at theta = -1.0044
q2 - q0: max 1.894e-15 at t = 0.291, theta = -0.0091, sin = -9.14e-03; where |sin theta| >= 0.5: max 2.220e-16 at theta = -1.0044
drift: max position residual 3.885780586188048e-16
theta range -4.141592363078889 1.0 samples 1001
```

The bounds, a = 0.3 m, d = 0.8 m, r = 1e-10 m:
- the loop's Jacobian in (phi, psi) on the branch, [[0, a s], [d, -a c]]:
  at |s| = 0.5, J^T J has trace 0.73 and determinant (a d s)^2 = 0.0144,
  smaller eigenvalue 0.0203, smaller singular value 0.1425 m; r / 0.1425 =
  7.0e-10 rad on (phi, psi), sqrt(2) times that, 9.9e-10, on psi - phi;
- at the change point, null direction d dphi = a dpsi, closure to second
  order d (d - a) / (2 a) dphi^2 = 0.667 dphi^2: dphi = sqrt(1.5e-10) =
  1.22e-5 rad, q2 - q0 = (d / a - 1) dphi = 2.04e-5 rad.

## The interface's survey (for the tuning)

`api_survey.py` over `multibody._mbd` in the installed package:

- 59 classes; 3 with their own `__repr__`;
- callables with unnamed arguments (pybind11's `arg0`): 0;
- callables without a docstring: 71 (most of them constructors of
  parameter classes and short getters).
