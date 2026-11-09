# The nightly's first runs: a search that stopped short by chance, and a bound set from one sample

- **Dates:** 9 November 2026
- **Plan:** the Phase 5 gate (the nightly workflow of task 5.5, D49); D50
- **Commits:** `93464db` (failures as annotations), then see git log
- **Written:** at the time

## Goal

The nightly workflow (D49) runs on schedule only from `main`. Before the
gate merges `phase-5`, it was run on the branch, started by a push of its
own files, and it had to pass.

## Starting point

The first run (42581491960, on `2dae1aa`) failed in both jobs. The long
suite failed at its "Long kernel tests" step (exit 1, about five minutes in),
and the mutation job failed in its unmutated tests (exit 3, in GCC's Release
build). The log that would name the tests needs a login. On the laptop the
long kernel tests passed (2 cases, 144 s), and the fast suite passed on
every CI build.

## What was done

1. **Failures as annotations** (`93464db`). `scripts/ci_run_tests.sh` runs a
   Catch2 executable with the compact reporter and, when it fails, writes
   its failure lines as `::error` annotations, which the public API serves.
   The second run (42586184000) named two tests (data, section 1). The
   40-four-bar's A02 error was 1.61e-6 rad against 1e-6 allowed (GCC,
   -O2). The test of contradicting constraints named nothing in GCC's
   Release build (-O3): MBD-K062 said the linearised equations could all
   hold.
2. **The conflict search, probed.** A temporary probe at the end of the
   Levenberg-Marquardt loop of `conflicting_equations` (D44) showed that on
   the laptop, from the test's one starting position, the search used all
   200 tries. It stopped with the gradient still 0.90, |phi| at 0.960
   against a least possible 0.884, lambda at 3.7e10, and the bob 2.1e-6 m
   from the anchors' axis (data, section 2). The filter's rank test drops
   one of the two distances' rows within about 7.5e-6 m of the axis, so the
   test had passed because the stop happened to land inside that.
3. **Forty-eight starts.** A new test starts the same contradiction from 24
   places of the bob and two crank angles. With the code as it was, 40 of
   the 48 named nothing; the searches stopped up to 0.11 m from the axis,
   after 200 tries (data, section 3). The cause (D50) is Marquardt's
   scaling. Each direction was damped by its own diagonal entry of J^T J,
   and the entry for the direction across the axis vanishes as rho^2. That
   direction keeps the curvature phi's second derivatives give it, so its
   steps overshoot unless lambda is huge, and a huge lambda stops every
   other direction too. Levenberg's damping, lambda times the largest
   diagonal entry in every direction, reaches the least-squares point from
   all 48 starts in 39 to 51 tries, at |phi| = 0.625 sqrt(2) to every digit
   printed.
4. **The 40-four-bar's bound.** A temporary probe ran the case with one
   setting changed at a time (data, section 4). The error ranged from
   1.7e-7 to 8.5e-7 rad when only the output interval or the constraint
   tolerance changed, and it grew as the integration tolerance fell (9.8e-7
   at 1e-12). It was always largest at t = 8.85 s. The error is the
   constraints' rounding amplified in the slow passes over the top
   (k^2 = 0.978, near the separatrix), not the integrator's error. The
   bound was 1e-6, set from one Linux measurement of 3.4e-7; it is now
   1e-5, six times the largest value seen on either platform, with the
   measurements in the test's comment. The branch stays checked apart from
   it at every sample, the couplers level to 1e-6.
5. **The nightly's regression steps** now also go through
   `ci_run_tests.sh`, and the workflow says that its mutation job is the
   only run of the tests built by GCC at -O3.
6. **The record:** D50; `docs/kernel.md`; the validation report's version
   1.1 (A02's 40 loops measured on both platforms; contradicting
   constraints named from 48 starts).

## Decisions

- D50: the conflict search damps its steps alike in every direction.
- The A02 bound is not a design decision; the test's comment carries its
  measurements.

## What went wrong, and how it was found

- **A test that passed by chance.** Task 5.2's test of contradicting
  constraints started from one position, and on three builds the stalled
  search stopped near enough to the axis. The 5.2 entry says that going
  "first to the least-squares point by Levenberg-Marquardt made the filter
  return exactly the two distances". The search never reached that point.
  It was found only because one more build, at another optimisation level,
  stopped it elsewhere.
- **A bound set from one sample.** The 40-loop bound of 1e-6 stood three
  times above one measurement of a quantity that moves fivefold between
  runs that differ only in their output interval. A tolerance is a claim
  about a spread, and the spread had not been measured.
- **A probe that could not be read the way it was first read.** The probe
  wrote to stderr and Catch2 to stdout. The two streams were first taken to
  be unpaired and the probe's stopping points quoted loosely, and the first
  draft of D50 said that the rank test's reach (7.5e-6 m) decided every
  case. A clean count showed that the probe's lines were in loop order, and
  that two starts had named the distances from 6.4e-4 and 7.8e-4 m.
  There both rows are kept, and the conflict is named only if the solve's
  rounding leaves more than 1e-10 of phi unexplained, which happens up to
  about 1e-3 m. D50 and the test's comment were corrected before the
  commit.
- **Not found:** why GCC's -O3 build stopped the search elsewhere than its
  -O2 build (no FMA or fast-math flag is set). On the laptop the same code
  fails from 40 of 48 starts, so no compiler difference was needed for it
  to go wrong; the nightly will show whether D50 is enough there.

## Verification

- "Assembly: contradicting constraints are named from any starting
  position" (`tests/kernel/test_kernel_assembly.cpp`). Its comment derives
  the least-squares point: along the axis the gradient is (u - v)(8 - u v),
  zero only at x = 1.5 m; off the axis both phi grow with rho; there the
  distances' rows are (1.5, 0, 0) and (-1.5, 0, 0) with phi = 0.625 each,
  asking for opposite moves. It fails from 40 of its 48 starts before D50
  and passes from all of them after.
- "IFToMM A02 with 40 four-bars" (hidden, `[long]`): bound 1e-5 rad, with
  the measured spread in its comment.
- On the laptop: all 530 tests; the kernel's fast suite (159 cases); the
  long kernel tests (2 cases, 106 s).

## Measurements

[data/2026-11-09_nightly-first-runs.md](data/2026-11-09_nightly-first-runs.md):
the nightly's annotations, the probe of the search, the 48 starts under
both dampings, the 40-four-bar's spread, and the suites.

## Lessons

- A search that may stop anywhere has to be tested from many starts; one
  start tests one path.
- Before setting a bound on a quantity that rounding moves, measure how
  far it moves under changes that leave the mathematics alone, as the
  regression harness's sensitivity probe does for its tolerance (D49).
- An optimisation level is another platform. The nightly's one -O3 build
  found what every other build had passed.

## Open ends

- Whether D50 holds in GCC's -O3 build: the nightly after this entry.
- The filter's verdict between about 7.5e-6 and 1e-3 m of such a point
  turns on rounding. Levenberg's damping keeps the search far inside that,
  but a search stopped by its 200 tries could still land there.

## For the book

- Part VII, the IFToMM benchmarks: the conflict search that reached its
  point by chance; Marquardt's scaling against Levenberg's damping at a
  least-squares point where a column of J vanishes. The figure: rho against
  tries for both, from a probe like section 3's.
- Part X, testing a numerical code: one start tests one path; a tolerance
  is a claim about a spread.
