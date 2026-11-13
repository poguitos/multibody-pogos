# Phase 6 retrospective

- **Dates:** 9 to 13 November 2026 (the phase); 13 November (this entry)
- **Plan:** Phase 6, tasks 6.1 to 6.6, the tuning with the author (added by
  D54) and the viewer inside notebooks (D59); documentation track H, task H.6
  for this phase; decisions D52 to D59
- **Commits:** `668b992` (task 6.1) to the commit that adds this entry, on
  branch `phase-6`, which starts from `v0.6.0` with D51 (a second review of
  external sources, `5ff97ca`) and the workflows' move to Node.js 24
  (`dff802a`)
- **Written:** at the time

## What the phase achieved, against its goal

Goal: "everything the engine can do, callable from Python. Start after
Phase 2 and extend it in every later phase." The kernel is callable, and the
package is built, installed and checked; the vehicle layer is not, by
design (below).

- **Bindings** (6.1, D52): frames, inertias, the tree and its eight joints,
  constraints and drivers, the general force elements, the simulator with
  every integrator, events, state elements and forces written in Python,
  the analyses (assembly, statics, linearisation, kinematics, loads), the
  checks. Errors are Python exceptions and warnings Python warnings, each
  with its code; the state is the engine's own memory as NumPy views.
- **Packaging** (6.2, D53): `pyproject.toml` with scikit-build-core; `pip
  install -e .`, and wheels built and tested in CI on Linux (Python 3.11,
  NumPy 2.0) and Windows (Python 3.14). The author's first install failed
  and was fixed, with instructions tried as written (D53 amended).
- **The interface tuned with the author** (D54): C++ names kept, shorthands
  for frames and bodies, settings by keyword with misspellings refused,
  objects that print what they hold, a docstring for every function.
- **Parity** (6.3, D55): ten C++ test cases reproduced through Python, 516
  numbers identical bit for bit, so the C++ tests' checks against closed
  forms carry over to the package.
- **Files** (6.4, D56): model files as JSON in the C++ library, exact and
  strict (a misspelt key is an error, each error names its place); results
  as pandas tables, to Parquet and CSV, exactly.
- **Viewing** (6.5, D57; D58, D59): playback in Rerun of a run, a table or a
  C++ program's CSV, with its channels plotted on one time line, shown below
  the cell in a notebook or in a window; the four-bar and the HMMWV
  replayed.
- **Examples** (6.6, D58): three notebooks, a pendulum, a four-bar and a
  quarter car, each checking its results against closed forms and run from
  top to bottom as a test.
- **Help** (H.6 for this phase): [the Python interface's how-to
  guide](../../docs/help/howto/python.md), seven examples run and checked in
  `python/tests/test_howto.py`, quoted exactly (the page check now reads
  Python too).

Not callable yet: the vehicle layer, with its tyres, aerodynamic and
anti-roll-bar elements, which is still on the old Y-up axes and is rebuilt
in Phases 7 and 8 (bound then); the lap tools; the kernel's `Data` and
algorithms; the constraint solver's internals; joint forces written in
Python. The goal's second sentence stands: each later phase extends the
package.

On the way the phase found: a force callback that read the step's start
instead of the stage (now documented, and in troubleshooting); the
package's own example pendulum, which could not swing; a build folder
shared by two generators; an environment whose NumPy crashes on this laptop
(MKL); half-escaped names in the viewer; replays merged into one recording;
and that an event's cut steps make semi-implicit Euler's energy drift.

## What took longer than expected, and why

- **The author's first install** (after 6.2): the instructions had been
  written and tried in the assistant's shell, with its own environment and
  execution policy. In the author's miniforge3 Prompt the build script was
  refused and plain pip met a build folder made by the other generator.
  Fixed by a folder per route, and by trying every instruction for the
  author from a script that does what the author's prompt does.
- **Builds stopped by PowerShell, twice** (6.1 and 6.3): a redirected
  stderr line from CMake ended the script. The cause is now a pitfall in
  `CLAUDE.md`, with the form that works.
- **Timings on battery** (6.3): a test that takes 15 s took 218 s with the
  laptop in energy-saving mode; the times were noted as such.

## What went well

- **Parity at the first run.** The bindings were written to reproduce the
  C++ expressions, and 516 numbers agreed bit for bit without a fix.
- **Tuning with the author before more code was written against the
  interface** (D54): the names and shorthands were settled once.
- **Computing apart from drawing** (6.5): the viewer's scene is tested
  against closed forms without a viewer, and the drawing library is not
  mocked.
- **Examples that are tests** (6.6): a notebook that runs through has
  passed its checks, and the allowances derived before the first run found
  the effect of events on a symplectic method.
- **Reading a package before installing it** (D59): its code said where the
  viewer comes from, how long a cell waits and that data is queued.

## What to do differently

- **Try instructions in the user's own shell**, not the assistant's: the
  first install would not have failed.
- **Make every environment from its pinned file**, or with its pins: the
  author's, made by hand, lacked the one pin that mattered (OpenBLAS), and
  the fault showed only when new code first called linear algebra.
- **Never write a number before it is measured.** It happened five times in
  6.1, and again in later tasks (counts in the tuning's proposal and in
  6.4's first draft, a size in D59's), each corrected before the commit,
  none acceptable.
- **Think through how a feature will be used,** not only whether it works:
  notebooks re-run cells, and the viewer's recording ids were not made for
  that until 6.6.
- **Edit strings with backslashes in the editor, not through scripts:**
  Windows paths and regular expressions lost or kept backslashes wrongly in
  scripted edits several times in the phase (three in 6.1 alone); each was
  caught by reading the result.

## Open ends carried forward

- Not bound: the vehicle layer (Phases 7 and 8), the lap tools, the kernel's
  `Data` and algorithms, the constraint solver's multipliers and Jacobian,
  `JointForce` written in Python.
- The interpreter lock is held through every call into C++ (D52); releasing
  it for long runs without Python callbacks is for later.
- Publishing (a licence, manylinux wheels, PyPI) is not planned (D53); an
  sdist and macOS are untested.
- CI's Linux jobs build without the interface, so the parity tests run on
  Windows only.
- pandas 2.2.2, the oldest the extra allows, is untested; a run's tables
  name state elements' states by number only.
- The viewer: bodies are their inertia's boxes until models have shapes
  (Phases 7 and 8); forces written in Python and contact planes are not
  drawn; recordings are not read back in the tests; the viewer inside VS
  Code's notebooks has not been seen working (no display here).
- 6.3's timings, to be repeated on mains power.
- From Phase 5, unchanged: generalised-alpha with events on the HMMWV
  (MBD-K058), Dormand-Prince's event location across force jumps, SDIRK4
  across force jumps without events, RK4's irregular convergence over the
  bump; the linearly implicit step's MBD-K045 where the tyres turn; the
  comparison with Chrono limited to gravity and constraints; the validation
  report's version 2 (tasks 7.4 and 9.6); GCC's -O3 conflict search (D50);
  the note for Pinocchio's developers, for the author to send.
- GitHub's notice that `ubuntu-latest` moves to Ubuntu 26: the Linux jobs to
  be watched across the move.

## Gate

The gate's conditions hold. The retrospective is this entry. The how-to
guide of the phase is [the Python interface's](../../docs/help/howto/python.md):
seven examples run and checked against closed forms in
`python/tests/test_howto.py`, quoted exactly; the page check now reads
Python examples too, and was shown to fail on a changed quote. On `241c5a9`,
which carries the phase's work, the guide and the version 0.7.0, CI run
43414670199 passed all seven jobs: Windows (the suite, eleven Python test
files with Python 3.11), Linux with the sanitizers, Linux Debug with no
allocation in a step, the API reference, the comparison with Pinocchio and
Chrono, and the package on Linux (Python 3.11, NumPy 2.0) and on Windows
(Python 3.14), its wheels `multibody-0.7.0`. On the laptop 544 tests pass,
and in the author's environment, reinstalled at 0.7.0, 72 with the ten
parity tests skipped. With the author's go-ahead (13 November), `main` was
fast-forwarded to the commit that adds this section and tagged `v0.7.0`. The
nightly workflow, which runs on schedule from `main`, starts on this code.

## For the book

- The chapter on the Python interface: what a binding owes the engine
  (names, errors with codes, the state as the engine's own memory),
  parity as the proof that it carries the C++ checks, files that keep every
  number, a viewer for a model without shapes, examples that check
  themselves.
- The phase's case study: the author's first install, where instructions
  that passed in one shell failed in another, and the rule that came of it.
