# Phase 3 retrospective

- **Dates:** 5 October 2026 (the whole phase, and this entry)
- **Plan:** Phase 3 (tasks 3.1 to 3.9), documentation track H (H.1 to H.6 for
  this phase); decisions D23 to D31
- **Commits:** `47315ba` (H.1, H.2) to the commit of this entry, on
  `phase-3` (laptop) and then `claude/charming-cori-e6sbqf` (cloud), which
  carries everything; the gate is the author's (task 3.10)
- **Written:** at the time, in a cloud session

## What the phase achieved, against its goal

Goal: the analyses and elements a general multibody code is expected to
have. Every task's "done when" holds:

| Task | Done when | Result |
|---|---|---|
| 3.1 Force library | each derivative matches finite differences | all to 1e-6; closed-form motions of each element |
| 3.2 Prescribed motion | a driven pendulum follows a sine and reports its torque | 1e-10, 1e-8; a point on a circle to 1e-13 m |
| 3.3 Outputs | pendulum pivot reactions match the analytic values | 3e-14 N |
| 3.4 Assembly | a loop given inconsistent values is assembled and reported | onto the analytic closure to 6e-12 rad; the least correction verified |
| 3.5 Statics | the detailed sedan settles in under 20 iterations to 1e-6 | 3 iterations, 1.8e-8 |
| 3.6 Kinematic analysis | the bump sweep runs through it with identical results | bitwise |
| 3.7 Linearisation | quarter-car frequencies match the analytic ones | 12 digits |
| 3.8 Contact | a block on an incline slides at the friction angle | 2.5e-13 above it; creep exact below |
| 3.9 Events | a bouncing ball's impact times correct to the tolerance | 4.3e-10 s at any step |

The help grew alongside (D23): message codes on every error and warning
(H.2, now 93 codes), a troubleshooting guide that covers every problem the
journal records (H.3), debugging aids in the program with a step trace that
diagnoses broken models from its file alone (H.4), an API reference built in
CI (H.5), and nine how-to pages whose examples run as tests (H.6). Finding
F14 (no statics, assembly, linearisation, prescribed motions, limits,
friction, contact other than the tyre, reactions) is closed.

Tests: 368 at the start of the phase, 438 at its end, all passing with GCC
13; CI (MSVC, sanitizers, no allocation, and now the API reference) green on
every pushed commit whose run has finished.

## What took longer than expected, and why

- **Tests written from estimates.** The events task lost a round to an
  impact count estimated instead of computed, and two more to a chattering
  test whose premise was wrong. The statics task lost one to a tolerance that
  ignored roundoff. Each was found by reading the failure carefully before
  touching the code, which is why none became a wrong change to the engine;
  but each cost a build.
- **A fallback that hid a broken main method.** Statics converged from the
  first run, through dynamic relaxation, while every Newton step was garbage
  (the decomposition's threshold set too late). The relaxation count in the
  report showed it; the tests now assert that the sedan needs none.
- **Designing before coding paid.** Linearisation passed every test on its
  first build because the hard questions (which coordinates, the chart of
  rotations, points that move under projection) were settled first.

## What to do differently

- Compute every expected value in a test from its closed form, in code or by
  a probe, before the test is run; never by estimate.
- Test each diagnostic and each fallback on healthy cases too, and assert
  that the main method does the work (no relaxation, no chattering, no
  warnings), not only that the answer is right.
- Keep the probe programs: they became the raw data of the journal (D22) at
  no extra cost, and their output is what the numbers in the entries quote.

## The move to cloud sessions (D25)

From task 3.4 on, the work was done in a cloud session: a fresh clone, a
branch of its own, GCC on Linux, CI as the first MSVC build. Three things
made that work:

- **The rules moved into the repository** (`CLAUDE.md`). The rule "every task
  ends with its journal entry" had lived in a laptop session's memory, which
  a cloud session cannot see.
- **A commit and a push at the end of every task.** The session's worker
  process was restarted in the middle of task 3.7; nothing was lost, because
  3.4 and 3.5 were already pushed and 3.7's files were on disk.
- **CI read after every push.** Every task's run passed; each entry records
  its run when it had finished.

What remains different: timings taken in the container are not comparable
with the laptop's, and none were used for decisions; GCC 13 prints false
positive warnings from its own standard library, which are noted in the
troubleshooting guide.

## The gate

The phase's conditions hold. The gate itself, merging into `main` and tagging
`v0.4.0`, is the author's: `main` is at `9472d8d` (v0.3.0), and
`claude/charming-cori-e6sbqf` descends from it, so the merge is a fast-forward.
The branch `phase-3` on GitHub stops at task 3.3 and is superseded by it.

## Open ends

- The vehicle builders still start from `set_vehicle_equilibrium`; they
  should settle the car by statics (F9, Phase 8). The sedan's wheel hop at
  17.7 Hz is to be checked against its parameters (F9, Phase 7).
- Dense output (task 4.2) for events and for locating them without
  re-integration; a bristle friction state (task 4.1); Hunt and Crossley
  contact damping for impacts (Phase 9); the analytic tangent matrices (D24)
  when models grow.
- Linearisation about steady motions in a moving frame (Phase 9).
- How-to pages for what Phase 2 delivered (a mechanism with loops, vehicles,
  sweeps, performance).
- The author's account of the period before the review is still to be
  written (`2025-12-07_before-the-review.md`).

## For the book

- Part V of the outline is complete in its skeleton (chapters 17 to 23).
- The engineering-practice part gains: tests computed, not estimated; a
  fallback that hides a failure; diagnostics tested on healthy runs; the
  probe programs as raw data; and working through git from fresh clones.
