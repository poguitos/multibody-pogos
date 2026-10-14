# Phase 3 retrospective

- **Dates:** 5 to 13 October 2026 (the phase); 13 October (this entry)
- **Plan:** Phase 3, tasks 3.1 to 3.9; documentation track H, tasks H.1 to
  H.5; decisions D22 to D29
- **Commits:** `49b96bc` to `f811714` on branch `phase-3`
- **Written:** at the time

## What the phase achieved, against its goal

Goal: the analyses and elements a general multibody code is expected to
have. All nine tasks are done:

- **Elements:** a force library whose every law returns its derivatives
  (3.1, D24); prescribed motions (3.2); contact of points and spheres with a
  plane, with ADAMS's IMPACT law and stepped static and dynamic friction
  (3.8, D28).
- **Outputs:** joint reactions, constraint forces per body, energies, a
  recorder (3.3).
- **Analyses:** kinematic analysis along a prescribed motion (3.6);
  initial-condition assembly with held coordinates and the smallest
  correction (3.4, D25); static equilibrium by Newton with least-norm steps,
  step limits and relaxation (3.5, D26); linearisation to frequencies,
  damping and mode shapes (3.7, D27); events located within a step (3.9,
  D29).
- **Help (track H):** a help folder, codes on every message with a
  catalogue the tests keep honest, a troubleshooting guide built from the
  journal, `describe()`, a state dump and a step trace, and an API reference
  built in CI without warnings.
- 423 tests (from 366 at the end of Phase 2); CI with four jobs (Windows,
  Linux with sanitizers, no-malloc, docs) green.

Each analysis was checked against an independent result rather than against
itself: closed-form configurations and loop velocities (assembly), energy
minima found by search and tyre loads equal to the weight (statics),
characteristic determinants and energy curvature (linearisation), creep
speeds predicted from the friction step and the friction angle (contact),
closed-form impact times (events).

## What took longer than expected, and why

- **Statics** took three attempts. The first stalled on the sedan; the
  hypothesis (scaling) was measured and found wrong; the cause was Eigen's
  `LDLT`, which is not rank-revealing and was trusted to be. The second
  converged but a start with wheels in the air folded the template's rear
  suspension; step limits (as in ADAMS) and relaxation on the forces at rest
  fixed the method, and the fold turned out to be the model's (no rebound
  stops, a contact point fixed on the upright).
- **Compile time** grew twice where it was not expected (a decomposition that
  cost 31 s per file; the general eigensolver, accepted in one file). The
  build log is now read after every build.
- **The machine** ran a full rebuild in 5.7 minutes and, later the same day,
  in 28.8: its own state, not the code.

## What went well

- Tracing tests by hand before building found two problems on paper (the
  events' blind spot, a vague broken model for the trace) and saved builds.
- Shared pieces paid off: the least-change solve and projection served
  assembly, statics and linearisation; the rank-revealing factorization, once
  fixed, served all four.
- The record (D22) was written with each task, and the troubleshooting guide
  could be built from it.

## What to do differently

- Read a library's documentation for the property relied on ("rank-revealing")
  before relying on it.
- Write the bound with its derivation before the first run; two bounds this
  phase were guesses that the measurement showed to be thin.

## Open ends carried forward

- Vehicle template: rebound stops and a wheel-ground contact at the lowest
  point (tasks 7.x); corner-prefixed body names.
- Analytic tangent stiffness instead of differences (Phase 4 needs it for
  implicit integration).
- Dense output to locate events and see out-and-back crossings (Phase 4.2).
- The sedan step at 149 us against 124 us on 5 October: machine or code, not
  established.
- Sticking friction (a bristle model) once extra states exist (Phase 4.1).

## Gate

All tasks of the phase meet their "done when" conditions and CI is green on
`phase-3`. With the author's go-ahead (14 October), `main` was
fast-forwarded to `4a3a757` and tagged `v0.4.0`; CI run 38530469971 on
`main` passed all four jobs (Windows, Linux with sanitizers, no-malloc,
docs).

## For the book

- Part V of the outline: one chapter per analysis, each with its
  independent check.
- The statics story as the phase's case study in debugging a numerical
  method: hypothesis, measurement, the real cause, and a finding about the
  model rather than the method.
