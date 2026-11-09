# Phase 5: the regression harness, its tolerance measured, and mutation tests

- **Dates:** 9 November 2026
- **Plan:** task 5.5 (regression harness, with the D42 option of mutation
  testing); decision D49
- **Commits:** see the index
- **Written:** at the time, on the author's laptop (Windows, MSVC), by
  Claude working with the author

## Goal

The plan's task 5.5: "Stored reference results with tolerances. A fast suite
for every commit and a long suite nightly. Optionally, mutation testing of
the kernel in a Linux CI job: a deliberately altered source line must make
some test fail (D42). Done when a deliberate physics change is caught."

## Starting point

516 tests, each checking a result against something known: a closed form,
an invariant, another code, a convergence order. Results with no such
reference (a vehicle's motion through a manoeuvre, its modes, a lap) were
checked only for being plausible, so a change in them would pass unnoticed.
CI ran every test executable on Windows and the kernel's on Linux with the
sanitizers; nothing ran on a schedule.

## What was done

1. **The harness** (`tests/regression/`): a scenario runs a model of the
   test suite deterministically and records named channels with their
   units; the test compares them with a CSV reference, each channel within
   a relative tolerance of its largest magnitude. Nine fast scenarios (the
   four-bar, the heavy top, the bushed pendulum, the HMMWV with joints and
   with bushings, the sedan driving, the sedan's modes, a wishbone's
   kinematics, a lap of an oval) and three long ones for the nightly suite.
   Every run uses fixed steps or a solve to a tight tolerance: a step
   controller's decision could turn on rounding.
2. **The references**, written by the test with `MBD_REGRESSION_UPDATE=1`
   on this laptop; the README says when that is allowed.
3. **The tolerance, measured.** A hidden case changes one physical
   parameter of each scenario by 1e-12 and by 1e-6 and compares with the
   unchanged run: the first moves no channel by more than 5e-3 of a
   tolerance of 1e-8, the second every scenario by 26 to 997 times it. A
   fast case requires the second for five scenarios.
4. **CI**: the Linux job builds and runs the regression executable too, and
   both jobs report each scenario's worst difference as an annotation, which
   can be read without logging in (the jobs' logs cannot).
5. **The nightly workflow** (`.github/workflows/nightly.yml`): the hidden
   long tests, the long scenarios and the sensitivity probe; and the mutation
   tests.
6. **Mutation tests** (`scripts/mutation_test.py`): nine one-line changes to
   the physics, each applied, built and tested, then undone. On the laptop
   all nine were caught, in 3.5 minutes: seven by the regression scenarios,
   which run first, two only by the kernel tests (gravity in the recursive
   Newton-Euler pass and the articulated-body pass, which the simulator does
   not use: it takes accelerations from the mass matrix and the bias
   forces). Run again against the kernel tests alone, six of the seven were
   caught there too; the tyres' peak force 2 % higher survived them, and
   only the regression scenarios (the sedan) notice it.

## Decisions

D49: deterministic scenarios against stored references at 1e-8 of each
channel's largest magnitude, the tolerance placed by measurement; platform
differences reported by annotations; a nightly long suite and mutation
tests from a short list of meaningful mutations.

## What went wrong, and how it was found

- **A claim made before it was measured.** The runner stops at the first
  executable that fails, and runs the regression tests first, so the first
  run showed which executable caught each mutation first, not which could.
  The draft decision said three mutations (the spring-damper's rate, its
  action and reaction, the tyres' peak force) were caught only by the
  regression scenarios. Run against the kernel tests alone, the first two
  were caught there as well (the HMMWV and the bushed models use
  spring-dampers); only the tyres' mutation survived. Corrected before the
  commit.
- **The runner would have changed the line endings.** Its first version read
  each source with Python's `read_text`, which turns CRLF into LF, and wrote
  it back so: every mutated file would have come back with other line
  endings. Caught on reading the script over before its first run; it now
  reads and writes bytes.
- **A date written from memory.** The README first dated the sensitivity
  table 7 November; the shell said 9 November.

## Verification

- `test_regression`: the nine fast scenarios against their references (all
  pass on the laptop), the long three (pass), the check that a change of one
  part in a million fails five scenarios by more than ten times their
  tolerances (it does: 26 to 997 in the probe), and the sensitivity probe.
- `scripts/mutation_test.py` on the laptop: nine of nine caught; against
  the kernel tests alone, six of the seven the regression scenarios caught
  first (the tyres' survived).
- `scripts\build.ps1 -Test`: 526 of 526 pass; no compiler warning.

## Measurements

`documentation/journal/data/2026-11-09_regression-harness.md`: the
sensitivity table, the mutation runs, the scenarios' durations.

## Lessons

- A tolerance for "has not changed" can be measured from both sides: what a
  change the size of rounding does, and what the smallest change worth
  catching does.
- A test of a tool's power belongs in the suite: the harness is checked to
  fail on a change it should catch.
- Mutation tests show which code the tests actually exercise: two
  algorithms checked by the kernel tests are not on the simulator's path.

## Open ends

- The Windows-Linux differences are reported by CI; if a scenario comes
  close to its tolerance across platforms, its tolerance or its run needs
  another look.
- The nightly schedule runs only from `main`, so it starts with the Phase 5
  gate; until then the workflow runs when it or the mutation script changes.
- Every planned change that moves results (the vehicle layer's axes in
  Phase 7, new tyres) will fail scenarios on purpose; the references are then
  written again, with the reasons recorded.

## For the book

Part X, "Testing a numerical code": the three kinds of check (against what
is known, against what was, against what a mutation breaks), and how the
regression tolerance is placed between rounding and the smallest change
worth catching. The sensitivity table as a figure.
