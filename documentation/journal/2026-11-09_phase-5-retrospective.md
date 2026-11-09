# Phase 5 retrospective

- **Dates:** 20 October to 9 November 2026 (the phase); 9 November (this
  entry)
- **Plan:** Phase 5, tasks 5.1 to 5.6; documentation track H, task H.6 for
  this phase; decisions D43 to D50
- **Commits:** `8dd6919` (task 5.1) to the commit that adds this entry, on
  branch `phase-5` (which also carries Phase 3's how-to guides, the D42 plan
  amendments and H.7, from before the phase's first task)
- **Written:** at the time

## What the phase achieved, against its goal

Goal: "evidence, independent of our own tests, that the solver is right.
This closes milestone M1." All six tasks are done, and the evidence is
gathered in [the validation report](../../docs/validation.md), version 1:

- **Analytic cases and manufactured trajectories** (5.1, D43; by a cloud
  agent): Jacobi's functions for a torque-free body, a heavy top's
  precession and nod, a body on three bushings against eig(K, M) with each
  mode's energy shares (added to the linearisation), and smooth motions
  prescribed for trees and loops, returned by every integrator at its order.
- **The IFToMM benchmarks** (5.2, D44, D45; by a cloud agent, closed with
  the library's results the author downloaded): A01 to A05 against closed
  forms, the Test Set for IVP Solvers and the library; contradictory
  constraints now named; the sanitizer job now fails on any report.
- **Independent codes** (5.3, D46): Pinocchio for tree dynamics (2.1e-13 at
  worst on 563 bodies), Project Chrono for closed mechanisms (1.4e-12 at
  rest, converging at Chrono's order in motion), both in CI; a note for
  Pinocchio's developers on its depth-first numbering.
- **The HMMWV benchmark** (5.4, D47, D48): the paper's fourteen-body model
  with and without its bushings, every method's agreement and cost; the
  static solver no longer squares stiffness ratios.
- **The regression harness** (5.5, D49): nine scenarios against stored
  references on every commit, the tolerance placed by measurement; a nightly
  long suite; nine mutations of the physics, all caught.
- **The validation report** (5.6), version 1.
- **Help** (H.6 for this phase): [checking a model](../../docs/help/howto/checking-a-model.md)
  (contradicting constraints named, a mode's energy shares, statics of parts
  a millionfold apart in stiffness), its examples compiled, run and checked
  against closed forms.

On the way the phase found and fixed defects no earlier test had shown:
undefined behaviour in three factorizations (D45), a static solver that lost
the soft directions of a stiff model and a relaxation that blew up beside
it (D47), a curve that read past its table when given NaN, messages that
printed small numbers as 0.000000, a singular-position stall that a
tighter constraint tolerance removes (D44), and, in the nightly's first
runs, a search for contradicting constraints that had reached its point
only by chance (D50).

## What took longer than expected, and why

- **The comparison with Chrono** (5.3): three of Chrono's behaviours, each of
  which would have been read as our error, had to be measured first (its
  assembled accelerations leave out velocity products and carry the
  constraints' residual over its step squared; its integrators start from
  zero accelerations unless assembled). The comparison at rest became a
  solve of Chrono's own equations from its matrix blocks.
- **The HMMWV** (5.4): a published data set with two faults, a static solver
  that could not settle the bushed model, a road whose bump makes forces
  jump, a laptop that slept in the middle of the benchmark.
- **A failure visible on one platform only**: the curve's out-of-bounds read
  returned NaN anyway on Windows; only Linux's sanitizer saw it, and the log
  that would say which test failed needs a login. The nightly's first runs
  failed the same way, on Linux only and one of them only at -O3, until a
  script turned the failures into annotations.

## What went well

- **Measuring the other code before trusting it** (Chrono on a pendulum,
  Pinocchio against itself): every surprise was found there, not in a
  confusing comparison.
- **Probes in the code, removed after use** (the singular values of the
  static Jacobian) turned a stall into a measured cause and a threshold with
  margins on both sides.
- **Annotations as a channel CI can be read through** without logging in:
  the regression scenarios' cross-platform differences, the mutation
  results.
- **Work split between a cloud agent and the laptop** (5.1 and 5.2 on Linux,
  the rest here), joined through the repository's rules (H.7) and the
  record: the laptop's Windows build found the cloud agent's one warning.

## What to do differently

- **Check a claim before writing it down.** Three times in this phase a
  statement went further than its evidence and was corrected afterwards:
  Pinocchio's assumption called undocumented (its C++ documentation states
  it), mutations said to be caught "only" by the regression scenarios (the
  runner had stopped at the first executable to fail), and three numbers in
  the validation report's first draft. Each was caught before or soon after
  the commit, but should not have been written.
- **Test a search from many starts, and measure a spread before bounding
  it.** Two of 5.2's tests passed by luck: the conflict search from its one
  start (D50), and the 40-loop N-four-bar against a bound three times one
  measurement of an error that moves fivefold between equivalent runs.
- **Keep long runs awake and detached** from the start, with output to files
  that survive the session.
- **Read the Windows build's warnings for code written on Linux**, and
  Linux's sanitizer results for code written on Windows (now in the
  pitfalls of `CLAUDE.md`).

## Open ends carried forward

- Generalised-alpha fails with events on the HMMWV (MBD-K058 on the
  remainder of a cut step); Dormand-Prince locates events on its continuous
  extension, which assumes smooth dynamics across the crossing; SDIRK4 does
  not converge across force jumps without events; RK4 converges irregularly
  over the bump even with events (a remaining non-smoothness, not found).
- The linearly implicit step reports MBD-K045 where the tyres turn to the
  bump.
- The comparison with Chrono covers gravity and constraints, not force
  elements, drivers or contact.
- Tyres against reference curves (7.4), the vehicle against measurements and
  Chrono's HMMWV (9.6): the validation report's version 2.
- The nightly schedule runs only from `main`: it starts with this gate.
- GitHub warns that `actions/checkout@v4`, `actions/setup-python@v5` and
  `mamba-org/setup-micromamba@v2` target Node.js 20, now forced to run on
  Node.js 24, and that `ubuntu-latest` moves to Ubuntu 26: the workflows'
  actions to be updated, and the Linux jobs watched across the move.
- Why GCC's -O3 build stopped the conflict search elsewhere than its -O2
  build (D50): not found.
- The note for Pinocchio's developers, for the author to send.
- From Phase 4, still open: the linearly implicit step's energy gain on very
  stiff springs between turning bodies, its missing convective term,
  generalised-alpha's Newton at its longest steps, the dense factorizations
  beyond the mass matrix; from Phase 3, the vehicle template's rebound
  stops and wheel contact point, sticking friction.

## Gate

All six tasks meet their "done when" conditions, and the phase's how-to
guide (H.6) is written. On `phase-5` at `28ac8af`, CI run 42590299575
passed all five jobs (Windows; Linux with the sanitizers; the no-malloc
build; the API reference; the comparison with Pinocchio and Chrono), and
the nightly workflow's run 42590299594 passed both of its jobs. Its long
suite ran the hidden long tests, the long regression scenarios and the
sensitivity probe. Its mutation job ran the unmutated tests in GCC's
Release build and caught all nine mutations, seven by the regression
scenarios and two by the kernel tests, as on the laptop. The nightly's
first two runs had failed; their fixes are D50 and the A02 bound
([the nightly's first runs](2026-11-09_nightly-first-runs.md)). Milestone
M1, a trustworthy core, is reached with this gate. With the author's
go-ahead (9 November), `main` was fast-forwarded to the commit that adds
this entry (`9349c52`) and tagged `v0.6.0`. On `main`, CI run 42593897074
passed all five jobs and the nightly's run 42593897025 both of its jobs,
all nine mutations caught again; the tag's push started the nightly as
well (42593903425, green).

## For the book

- Part VII of the outline: what "validated" means, then the analytic cases,
  the benchmarks, the other codes, the HMMWV, the regression harness.
- The phase's case study: the HMMWV, where a data set, a static solver and a
  set of integrators were each found wanting by a model none of them had
  met, and each finding was measured before it was fixed.
