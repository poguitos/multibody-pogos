# Working on this repository

A C++20 multibody engine for vehicle dynamics, built task by task to the plan
in `documentation/Master_plan.md`, with a record of every step kept for a book
about the project. These are the rules for anyone working on it, human or
agent (plan task H.7, decision D42). Read them before the first change.

## Where things are

| What | Where |
|---|---|
| The plan: phases, tasks with checkboxes and "done when", working rules | `documentation/Master_plan.md` |
| Design decisions D1 onwards, with context and alternatives | `documentation/decisions.md` |
| The journal: an entry per task, its index, the template | `documentation/journal/` (`README.md`, `TEMPLATE.md`), raw measurements in `journal/data/` |
| The book's outline and bibliography | `documentation/book/` |
| How the program works | `docs/conventions.md`, `docs/kernel.md`, `docs/integrators.md`, `docs/performance.md`, `docs/hmmwv.md`, `docs/python.md` |
| The Python interface: the package, its bindings, its tests, its packaging | `python/multibody/`, `python/src/`, `python/tests/` (D52), `pyproject.toml` (D53) |
| What is validated, against what, and what is not | `docs/validation.md` |
| Help: how-to guides, troubleshooting, the message catalogue | `docs/help/` |
| Sources reviewed for ideas, and what was taken | `documentation/review_2026-10-20/external_sources.md` |

**Where the work stands:** the plan's checkboxes and the journal index say it.
On 9 November 2026: Phase 5 merged into `main` (tag `v0.6.0`, milestone
M1), after Phase 4 (`v0.5.0`). Phase 5 brought Phase 3's how-to guides,
the D42 plan amendments, this file, task 5.1 (analytic cases and manufactured trajectories, D43), task 5.2
(the IFToMM benchmarks, D44, with the library's results for A04 and A05 in
`tests/kernel/data/iftomm/`), task 5.3 (the comparison with Pinocchio and
Project Chrono, D46, in `crosscheck/`), task 5.4 (the HMMWV benchmark, D47
and D48, `docs/hmmwv.md`), task 5.5 (the regression harness, D49,
`tests/regression/`) and task 5.6 (the validation report,
`docs/validation.md`), and the nightly's first runs (D50). Branch
`phase-6` starts from `v0.6.0` with D51 (a second review of external
sources: suspension kinematics, validation, target cascading; plan amended
in 8.1, 8.4, 9.1, 9.3 to 9.7 and 13.3), then Phase 6 (the Python
interface): task 6.1, the bindings, is done (D52, `docs/python.md`), and
6.2, packaging (D53, `pyproject.toml`), and the interface tuned with the
author (D54: shorthands, settings by keyword, readable objects; the author's
environment is conda `mbd-try`), and 6.3, parity tests (D55: ten C++ cases
reproduced through Python, 516 numbers identical, `python/tests/test_parity.py`);
6.4, files, is next. After a change to the bindings, reinstall `mbd-try`
(an editable install whose module is older than its Python files stops with
MBD-P011). The nightly workflow runs on schedule from `main`.

**Not in the repository:** the textbooks and papers in `documentation/` (git
ignores PDFs), and local copies of the reviewed sources (on the author's
laptop, `C:\dev\references\`). The plan's appendix says which reference serves
which phase; if a task needs one, ask the author.

## Building and testing

The project uses CMake, Ninja and the presets in `CMakePresets.json`; the
dependencies (Eigen, Catch2, spdlog, pybind11) are downloaded by CMake when it
configures. Build type RelWithDebInfo.

**On the author's laptop (Windows, MSVC): never more than one compiler at a
time.** Parallel builds restarted the machine twice. The limit is in the build
system (`MBD_COMPILE_JOBS`, default 1); do not raise it there. Build only
through the script, from PowerShell:

```powershell
scripts\build.ps1                    # configure if needed, build everything
scripts\build.ps1 -Test              # build and run the whole suite
scripts\build.ps1 -Target test_kernel
```

It loads the MSVC environment and pins the console code page (MSVC's
localized include notes otherwise stop Ninja tracking header changes). Never
start a build while another runs, and do not edit headers or test sources
during a build.

The Python interface is built for the interpreter given once with `-Python`
(kept in the build folder's cache). On the laptop that is the conda
environment `mbd-python` (`python/environment.yml`):
`scripts\build.ps1 -Python C:\Users\margk\miniforge3\envs\mbd-python\python.exe`.
Its tests run in the suite as `Python: <file>`; run one by hand from
`build\python` with `PYTHONPATH` set there and `python -m pytest`.

pip installs the package (`pyproject.toml`, scikit-build-core, D53) in its
own folder, `build-python\<wheel tag>`. On the laptop never run `pip install`
of this folder by hand: Visual Studio's generator, pip's default there,
ignores the build system's limit. Go through the script, which uses Ninja:
`scripts\build.ps1 -Install -Python <python.exe>`, into an environment other
than `mbd-python` (an installed package is imported before `PYTHONPATH`'s;
`python/tests/conftest.py` then stops the suite's Python tests). The
script's install builds in `build-python\ninja-<wheel tag>`: a CMake folder
serves one generator, and the two routes shared one until the author's
first install failed in it. The author runs scripts from a miniforge3
Prompt under Windows' default execution policy, so instructions for the
author give `powershell -NoProfile -ExecutionPolicy Bypass -File
scripts\build.ps1 ...` and are tried as written (`docs/python.md`).

**On Linux (a cloud machine, CI):** as the CI's Linux jobs do
(`.github/workflows/ci.yml`); three compile jobs are fine there:

```bash
cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DMBD_COMPILE_JOBS=3 -DMBD_PCH=OFF -DMBD_BUILD_PYTHON=OFF
cmake --build build-linux                       # or --target test_kernel ...
ctest --test-dir build-linux --output-on-failure
```

`MBD_PCH=OFF` keeps every file's includes honest (the precompiled header would
hide a missing one). A full build takes several minutes. CI builds only
`test_core`, `test_model` and `test_kernel` on Linux; the other executables
have so far been compiled only with MSVC, so a GCC error in them is a
portability fix to make (and record), not a sign that the build is broken.

**Tests.** One executable per area: `test_core`, `test_model`, `test_kernel`,
`test_forces`, `test_vehicle`, `test_analysis`, `test_alloc` (counts every
allocation in a step), `test_howto` (the how-to guides' examples),
`test_regression` (scenarios against stored references, D49). Run one
case by name with `-s` to read the values behind each check; a comma in a
test's name must be written `\,`. Hidden tests (`[.]`, such as `[alloc]` and
`[long]`) are not registered with ctest. `mbd_bench`, `mbd_verify` and
`mbd_hmmwv` (`bench/`) are built only on request (`--target`).

**Checks that run in the suite:** every message code used in the sources must
be in `docs/help/messages.md` and every catalogued code used
(`test_message_catalogue`); every example in `docs/help/howto/` must be quoted
exactly from `tests/howto/` (`test_howto_pages`).

**The comparison with independent codes** (`crosscheck/`, D46) needs the conda
environment in `crosscheck/environment.yml` (Pinocchio and PyChrono from
conda-forge; never PyPI's `pychrono`, an unrelated package). On the author's
laptop it is `C:\Users\margk\miniforge3\envs\mbd-crosscheck` (Miniforge, not
on the PATH). Build `mbd_crosscheck`, then run the two scripts as
`crosscheck/README.md` says; the Chrono one takes about two minutes.

**The regression harness** (`tests/regression/`, D49) compares nine
scenarios with references stored there, each channel within 1e-8 of its
largest magnitude. When one fails, find out why before anything else: a
reference is written again (`MBD_REGRESSION_UPDATE=1`, as its README says)
only for a change meant and understood, and the task's journal entry and
commit say which scenarios moved, by how much, and why.

**CI** (GitHub Actions, every push to any branch): Windows through the build
script; Linux with GCC and the address and undefined-behaviour sanitizers,
any report failing its test (D45); Linux Debug checking that nothing
allocates; the Doxygen reference with every warning an error; the comparison
with Pinocchio and Project Chrono (D46). Both test jobs report the
regression scenarios' worst differences as annotations. A task is finished
when CI is green on its commit. **Nightly** (`.github/workflows/nightly.yml`,
from `main` only; elsewhere when the workflow or the mutation script
changes): the hidden long tests, the long regression scenarios, and the
mutation tests (`scripts/mutation_test.py`, D49).

**Timings** compare only on one machine in one session: the laptop's speed
varies by a factor of two between days. Its benchmarks run pinned to one
performance core at high priority (`docs/performance.md`); numbers from a
cloud machine are not comparable with the recorded ones.

## Working rules

1. **Tests assert values derived independently.** Every tolerance comes with
   the hand calculation that justifies it, written in the test. A test that
   records what the code did, or only checks that it ran, is not a test.
   Report the margins actually seen.
2. **New joints, constraints and forces pass the invariant tests** in
   `tests/kernel/` (finite differences, conservation) before anything is
   built on them.
3. **One place per formula.**
4. **Measure before deciding**, on the model the choice is for, and verify a
   claim on the kind of model that could break it.
5. **Keep the record (D22).** Every task ends, in the commit that finishes
   it, with: a journal entry (`documentation/journal/TEMPLATE.md`; what went
   wrong and how it was found is the most valuable part), a row in the
   journal's index, decisions in `decisions.md` (numbered after the last, with
   an index row), raw data in `journal/data/` with the command and machine
   state, the plan's checkbox ticked with a short *Result* note, and the
   book outline updated when a topic worth a section appears.
6. **Keep the help current (D23).** A task that adds or changes a message,
   an option or a behaviour updates `docs/help/` in the same commit. Every
   error and warning carries a code (`MBD-` with an area letter and a number)
   with an entry in the catalogue.
7. **Each phase ends at its gate**: a retrospective entry, the how-to guides
   of what the phase delivered (H.6, examples run as tests), CI green.
   **Merging a phase into `main` and tagging needs the author's go-ahead,
   every time.**
8. **Work on the phase's branch** (now `phase-5`); `main` stays green.
   Commit and push each finished task; query the GitHub API sparingly.
9. **Commit messages:** a summary line, then what was done and why, the test
   count, the records touched; written to a file and committed with
   `git commit -F <file>`; ending with the co-author line of whoever helped.
10. **Report faithfully.** Never state a result that was not run and seen in
    the session; say what was skipped or failed, with the output.

## Writing

The documentation is plain English: short sentences, one idea each, numbers
with their units and how they were measured, no promotional words. The book
will quote the journal and the decisions, so write them for a reader who was
not there. Use British spelling, as the existing text does.

## Pitfalls met before

- Many files are CRLF. Scripted multi-line replacements (perl, sed) silently
  match nothing unless they handle `\r`; prefer an editor tool for such
  edits. In a perl replacement write `${1}` before a digit, and escape
  backslashes (a Windows path lost them once).
- Eigen's `LDLT` is not rank-revealing; use `kernel::RankRevealingLDLT`.
  `CompleteOrthogonalDecomposition` costs about 30 s of compile time per file
  that instantiates it.
- A long run of a chaotic model (the four-bar with its pendulum) depends on
  rounding; compare such runs by their statistics.
- A tight Newton tolerance can fail where a looser one, still far below the
  step's error, runs.
- `ctest --output-on-failure` prints nothing of a test that passes, so a
  sanitizer report in one is hidden from its summary. Build the sanitizer
  configuration as CI does (`-fno-sanitize-recover=undefined`, D45), or read
  `Testing/Temporary/LastTest.log`, before calling a run clean.
- An `Eigen::LLT` (or `LDLT`) built with its size only leaves its status
  unset; moving one copies that unset value (undefined behaviour). Size a
  factorization by computing it, on the identity if need be.
- MSVC warns where GCC does not: a parameter with a member's name (C4458)
  passed on Linux and warned on Windows, with CI green (a warning does not
  fail the Windows job). Read the Windows build's warnings (CI's log or the
  laptop's build) for code written on Linux.
- PyChrono's getters (`GetLinVel()`, `GetPos()`) return references to the
  body's own vectors: copy the numbers out before changing the body, or the
  saved value changes with it.
- `scripts\build.ps1` run under a PowerShell redirection (`*> log`) stops at
  the first line a native command writes to stderr, which a configuration
  that fetches a dependency does: Windows PowerShell turns each such line
  into an error, and the script's `$ErrorActionPreference = "Stop"` ends it.
  Redirect from outside instead: `powershell -File scripts\build.ps1 ... >
  log 2>&1`.
- The dev preset sets `FETCHCONTENT_UPDATES_DISCONNECTED`: a dependency whose
  pinned commit changes keeps its old sources in `build/_deps/` until its
  `<name>-src`, `-build` and `-subbuild` folders are removed.
