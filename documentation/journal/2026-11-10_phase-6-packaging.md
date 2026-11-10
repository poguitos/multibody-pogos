# Phase 6: packaging (pip install -e .)

- **Dates:** 10 November 2026
- **Plan:** 6.2; D53
- **Commits:** `f583c86`, `60ee42d`, `254b403`, `6522fd7`, then see git log
- **Written:** at the time

## Goal

Task 6.2: "`pyproject.toml`, an editable install, and a wheel built in CI.
*Done when `pip install -e .` works on a clean machine.*" The author also
asked to tune the interface together once it can be installed; that comes
after this task.

## Starting point

Task 6.1 left the package `multibody` built only by the project's CMake
build, for the interpreter given with `scripts\build.ps1 -Python`, and
imported through `PYTHONPATH=<build>/python`. No install rule existed; the
build fetched Catch2 even with the tests off; no Linux job had built the
module.

## What was done

1. **Reading the backend before using it.** scikit-build-core 1.1.2 (the
   newest) downloaded and read, not run, for three questions that decide
   whether pip may build on the laptop:
   - it passes no parallelism to `cmake --build`, so the build system's
     job pools decide, but only under Ninja;
   - on Windows, with no `CMAKE_GENERATOR`, it takes CMake's default,
     Visual Studio's generator, which ignores job pools;
   - its editable finder is inserted at the head of `sys.meta_path`, so it
     comes before `PYTHONPATH`;
   - it adds CMake from PyPI when the machine has none of the version asked
     for, and Ninja on Linux.
2. **The package (D53):** `pyproject.toml` (the module only, Release, tests
   off, a build folder kept in `build-python/<wheel tag>/`, the version
   from `CMakeLists.txt`, NumPy 2.0 or later); an install rule for the
   module (component `python`); Catch2 fetched only with the tests;
   `scripts\build.ps1 -Install`, which installs with Ninja so that one
   compiler runs; `python/tests/conftest.py`, which stops the suite's
   Python tests if they import an installed package instead of the build;
   a test that pip's version is the module's.
3. **On the laptop**, into an empty environment from Miniforge's Python
   3.14: installed in 3 min 41 s with one compiler throughout, the 33
   Python tests passing against it; a second install compiled nothing
   (16 s). The backend warned that `tool.scikit-build.metadata` is
   deprecated; the version moved to a top-level `[[tool.dynamic-metadata]]`
   entry, and the warning went.
4. **CI's package job** (`f583c86`): plain `pip install -e .` on fresh
   runners, the tests, a wheel tested in an empty environment and kept.
   Windows (Python 3.14) passed at once. Linux (Python 3.11, NumPy 2.0)
   failed three times, each cause readable only in a log that needs a
   login:
   - the install (`f583c86`): the static library was not
     position-independent, which a shared module on Linux needs. Fixed in
     `python/CMakeLists.txt` (`60ee42d`); the install then succeeded, which
     is the evidence, since the error's text was never seen;
   - the tests (`60ee42d`), within a second: unknown until
     `scripts/ci_annotate.sh` turned each step's failure into annotations
     (`254b403`, with `.gitattributes` keeping shell scripts at LF for the
     Windows runner's bash);
   - then readable: the parallelogram test left its branch by 4.0e-9 rad
     against a bound of 3e-9. The bound was wrong, not the engine (below).
     Corrected with its derivation (`6522fd7`).
5. **The record:** D53, `docs/python.md` (installing; building for
   development), the README, CLAUDE.md, troubleshooting, this entry and its
   data, the plan, the outline.

## Decisions

D53: scikit-build-core, only the module built, a build folder kept between
installs, one compiler by default, the suite checking what it imports, a
package job in CI, nothing published. Smaller choices:
- the build script's `-Install` rather than letting pip run by hand on the
  laptop, whose rule stays: build only through the script;
- `scripts/ci_annotate.sh`, a general form of `ci_run_tests.sh`, so that any
  CI step's failure can be read without logging in;
- shell scripts at LF on every checkout (`.gitattributes`): the Windows
  runner runs them with bash, and a checkout with `core.autocrlf` writes
  them with CRLF, as it had on the laptop.

## What went wrong, and how it was found

- **The Linux module did not build.** On Linux a shared library cannot
  take code compiled for an executable, as the static library was.
  The library is now compiled as position-independent code when the
  interface is built. Found by CI's first Linux build of the module; the
  cause was inferred, and confirmed by the fix.
- **A bound derived for a mechanism that is singular on its path.** The
  6.1 test bounded the parallelogram's departure from its branch at 3e-9
  rad, from the projection's tolerance (1e-10 m) through the loop's
  Jacobian, as if it kept full rank. The swing passes the two change points,
  where the crank lies along the ground link and the Jacobian's
  determinant, -a d sin θ, vanishes: there a departure costs only second
  order in the closure, and the tolerance allows 2.0e-5 rad. Windows'
  1.9e-15 was rounding that happened to stay on the branch, at the change
  point itself (θ = -0.009); Linux's GCC with -O3 rounded to 4.0e-9, and
  the corrected test's pass on Linux shows that it lies where
  |sin θ| < 0.5, within 30 degrees of a change point.
  The test now bounds the departure at 5e-5 over the run and at 3e-9 where
  |sin θ| ≥ 0.5 (derived 1.0e-9). This corrects the 6.1 entry, which put
  the gap between 3e-10 and the measured 2e-15 down to the projection
  closing the loop to rounding: true, but the 3e-10 itself held only away
  from the change points.
- **CI's logs need a login**, and the first two Linux failures said only
  "exit code 1". `scripts/ci_annotate.sh` fixed that for every step of the
  package job.
- **Tooling:** a wait loop for the install matched pip's earlier line
  "Successfully installed" (its build tools) and ended early; a monitor of
  CI waited for 8 jobs where the run has 7; a scan of the bindings'
  includes matched every file for every symbol and was discarded unread.
  None changed a result.

## Verification

- On the laptop: the 33 Python tests against the installed package; the
  guard tried both ways (the build's own package: passes; another folder:
  pytest stops with its message, exit status 4); the installed version
  test skipped in the suite (nothing installed there) and passed in the
  environment; the suite's 532 tests after the CMake changes.
- In CI: every job of `6522fd7` (CI run 42790138563), the package job on
  both platforms included.
- The parallelogram's bounds: in the comment of
  `test_a_parallelogram_stays_a_parallelogram`, the Jacobian
  [[0, a sin θ], [d, -a cos θ]] in the coupler's and rocker's angles, its
  smaller singular value 0.1425 m at |sin θ| = 0.5 (7.0e-10 rad on the
  angles, 9.9e-10 on q2 - q0, allowed 3e-9), and the second-order closure
  d (d - a) / (2 a) φ² at a change point (1.22e-5 rad on φ, 2.04e-5 on
  q2 - q0, allowed 5e-5). Measured on Windows: 1.9e-15 at the change point,
  2.2e-16 away from it, in the package's Release build and the suite's
  RelWithDebInfo build alike.

## Measurements

In `data/2026-11-10_phase-6-packaging.md`: the laptop's install (3 min 41 s,
one compiler at each of four samples, 726 MB of build folder, a 668 kB
editable wheel), the reinstall (16 s), the guard, each CI run, the
parallelogram's departures and the arithmetic of its bounds, and the
survey of the interface for the tuning (59 classes, 3 with their own
`repr`; no unnamed argument; 71 functions without a docstring).

## Lessons

- **Check a Jacobian's rank along the whole motion before carrying a
  tolerance through it.** A loop that passes a singular configuration
  turns a linear bound into a square root.
- **A new platform's first build of a part is a test of its own.** The
  bindings had only been built with MSVC; on Linux the library had to be
  compiled differently before the module could be built.
- **Make a CI failure readable from the first push.** Two round trips were
  spent on failures that said only "exit code 1".
- **Read a tool's source before trusting it with a rule.** The laptop's
  limit of one compiler would have been broken by pip's default generator
  on Windows, silently.

## Open ends

- The tuning of the interface with the author: next, before 6.3.
- Publishing (a licence, manylinux wheels, PyPI): not planned; D53.
- An sdist and macOS are untested.
- Releasing the interpreter lock for long runs (D52) is still open.

## For the book

- Chapter on the Python interface (6): packaging a CMake project for pip,
  and the editable finder that comes before `PYTHONPATH`. *D53.*
- Chapter on verification: the parallelogram's two regimes, linear away
  from the change points and square-root through them, as a worked example
  of a derived bound and of how one platform's rounding can hide a wrong
  derivation that another's exposes.
