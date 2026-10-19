# Phase 4 retrospective

- **Dates:** 14 to 19 October 2026 (the phase); 19 October (this entry)
- **Plan:** Phase 4, tasks 4.1 to 4.9; documentation track H, task H.6 for
  this phase; decisions D30 to D41
- **Commits:** `aa8876c` to the commit that adds this entry, on branch
  `phase-4`
- **Written:** at the time

## What the phase achieved, against its goal

Goal: accurate integration for ordinary models and stable integration for
stiff ones. All nine tasks are done:

- **The interface** (4.1, D30): state elements, first-order states
  integrated with the bodies in the same stages; the drivetrain became one,
  and the step-callbacks went away. Braking became independent of the step
  to nine digits, which closed an open end of Phase 1.
- **Seven methods and a mode:** RK4 and semi-implicit Euler; Dormand-Prince
  with error control and dense output, on which events are now located
  (4.2, D31); generalised-alpha (4.3, D32); SDIRK4 and, by the author's
  decision when SDIRK4 could not meet its "done when", Radau IIA (4.4, D33,
  D34); the linearly implicit step for real time (4.5, D35); multirate RK4
  for fast states (4.6, D36).
- **Constraints per integrator** (4.7, D37): projection after every method,
  generalised-alpha's stabilized index-2 form as an option, a drift channel;
  1000 simulated seconds within the tolerance. Its open end, done as a task
  of its own: the inertia's velocity products implicit in the two Euler
  steps (D38), which a spinning top had blown up.
- **The mass matrix along the tree** (4.8, D39): 13 % off an RK4 step of the
  sedan; the crossover with a dense factorization measured.
- **Verification** (4.9, D40, D41): `docs/integrators.md`, every method's
  order, energy and cost on six models, written by a program; it found the
  real-time step diverging on stiff bushings (fixed, D40) and the limit
  that remains.
- **Help** (H.6 for this phase): a how-to guide for time integration whose
  examples are compiled, run and checked as tests, and quoted exactly.
- 478 tests (from 423 at the end of Phase 3); CI green on every commit of
  the phase.

Against the goal: on ordinary (smooth) models every method converges at its
order and the high-order explicit ones are cheapest; on stiff ones the
implicit methods are stable at any step, Radau IIA the cheapest at every
accuracy measured, and the real-time step stable at any step, with an
accuracy limit on very stiff springs between turning bodies that the
report states.

## What took longer than expected, and why

- **SDIRK4** (4.4) could not meet its "done when": its stage order of 1
  makes it lose its order beside stiff bushings. That was found by
  measuring, not by reading; Radau IIA was added in the same task.
- **Generalised-alpha on the constraints** (4.7): the stabilized form took
  two missing Newton terms, found by printing Newton's contraction, and then
  cost 1.45 times as much on the sedan, so it became an option.
- **The gyroscopic open end** (D38) became a task of its own: the first fix
  passed every test and was unstable on a closed loop, which only a probe of
  the energy over 100 s at three steps showed; six treatments were measured
  before one was kept.
- **The verification** (4.9) found what the tasks' own tests had not: the
  real-time step's divergence on stiff rotating bushings, and then, through
  the test written for the fix, its energy gain.
- **The machine** changed speed by 2.3 times between days, so every timing
  comparison was rerun within one session, versions alternated, often with
  an old commit built from an archive.

## What went well

- **Measuring before deciding.** Each choice between alternatives was made
  on numbers from the model it was for: the stabilized form on the sedan,
  not on the four-bar; the tree factorization's three versions on chains and
  trees alike.
- **Closed-form checks.** The oscillator, on which every method's answer is
  known from its stability function or recurrence, verified every method to
  rounding; the nutation of a free body, backward Euler's damping exactly.
- **Derived bounds** (energy cannot rise, the tip cannot leave the links'
  reach) caught what "it stays finite" would have passed.
- **Probes with temporary switches**, removed after use, separated causes
  quickly (the symmetric part, the rotational stiffness, damping or not).
- **The record** kept pace with the work: every task's journal entry, data
  and decisions in its commit.

## What to do differently

- **Check a claim on the kind of model that can break it**, at the time it is
  made: "stable at any step for stiff bushings" was tested on linear
  springs and failed on the first stiff spatial model.
- **Revisit a workaround when its reason goes away.** The symmetric parts
  (D35) were taken for a Cholesky; D38 replaced the Cholesky and the
  symmetrization stayed until 4.9.
- **Write each phase's guides at its gate (H.6).** Phase 3's gate did not;
  its guides are owed.
- **A Newton tolerance should follow the accuracy wanted**, not be as tight
  as possible: at 1e-12 generalised-alpha failed where 1e-9 ran.

## Open ends carried forward

- The linearly implicit step's energy gain on very stiff springs between
  turning bodies at long steps (k h^2 / m above about 25): one Newton
  iteration would remove it at twice the cost.
- Its stiffness matrix's missing convective term (about c |v| of a damper),
  and the joints' geometry it leaves explicit; analytic derivatives of the
  elements (D24, D35) where timing asks.
- Generalised-alpha's Newton at its longest steps (4 ms on the stiff
  pendulum, 10 ms on the sedan).
- The constrained solve beyond the mass matrix (J M^-1 J^T, dense) and the
  implicit steps' iteration matrices, factorized densely.
- A manoeuvre with events (braking, locking, contact) for the methods with
  error control.
- The how-to guides of Phase 3 (assembly, statics, linearisation, outputs,
  contact, events).
- From Phase 3, still open: the vehicle template's rebound stops and wheel
  contact point; sticking friction (a bristle model).

## Gate

All tasks of the phase meet their "done when" conditions; CI is green on
`phase-4`. With the author's go-ahead (19 October), `main` was
fast-forwarded to the commit that adds this entry and tagged `v0.5.0`.

## For the book

- Part VI of the outline (items 24 to 33): one chapter per method or
  question, each with its measurement.
- The phase's case study: a claim made on linear springs, a workaround that
  outlived its reason, and a fix that left a second limit, all found by a
  verification that put every method on the same models.
