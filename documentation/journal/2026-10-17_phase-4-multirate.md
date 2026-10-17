# Phase 4: multirate RK4 for the fast states

- **Dates:** 17 October 2026
- **Plan:** task 4.6; decision D36
- **Commits:** see the index
- **Written:** at the time

## Goal

Advance the fast states (wheel spin, tyre lag) several times per main step,
slowest first, as in Haug et al. section 6. *Done when a ratio of 6
reproduces the single-rate result.*

## Starting point

The state elements' states (task 4.1) are the simulator's fast part: the
drivetrain's wheel spins, which at walking pace relax at about 3000 per
second and hold RK4 under a millisecond (task 4.3's finding), while the
bodies' motions are far slower. Every RK4 stage evaluates everything,
including the constrained dynamics, the dear part.

## What was done

- **Read the paper's section 6 first** (`documentation/Selección/Selección
  2/`, found with the review's help). Its partition is the simulator's own
  (bodies; states x' = Gamma(q, q', x, t)); its slow integrator is a
  constant-step Adams-Bashforth method, one evaluation per step needing the
  fast states only at the step's start, which is why slowest first suits
  it; it warns that extrapolating the fast variables into the slow step "is
  subject to numerical errors". RK4's stages need the states inside the
  step.
- `Simulator::multirate`, `IntegratorStats::state_evaluations`, the fast
  steps' cheap evaluation (kinematics, forces, rates, no dynamics) and the
  step, first with three couplings behind a temporary switch: slowest first
  with the states held, slowest first with the states extrapolated along
  their last step's change, and fastest first (the bodies' first stage, the
  fast steps with the bodies extrapolated, then the bodies' other stages
  with the states the fast steps reached).
- Measured on the braking test, the car rolling at walking pace and the
  sedan's drive; chose fastest first (D36); removed the other two.
- Tests: fourth order on a lag driving a mass; the ratio of 6 against single
  rate at the fast step on the rolling car and the sedan; refusals (K046);
  no allocation.

## Decisions

- **D36**: multirate RK4, fastest first, the state elements' states as the
  fast part.

## What went wrong, and how it was found

- **Holding the fast states was wrong by a lot, as estimated.** Before
  coding, braking at 10 m/s with steps of 6 ms put the error of held spins
  at about 8 % of the braking force; measured from 20 m/s, 2.2 % on the
  deceleration. Each tyre's slip drifts over the step as the hub slows while
  the spin is held.
- **The sedan drifted, and the cause was not where it seemed.** With the
  first couplings that worked on the simple vehicle, the sedan's drive
  ended with the car moving at 0.5 m/s two seconds after it had stopped. It
  looked like the bodies' step being too long for the hubs' side of the
  tyres' slip at low speed (which is real: 6 ms on the simple vehicle at
  0.5 m/s was unstable). Sweeping the steps showed otherwise: single rate
  was fine at 2 ms, and every multirate run failed whose *fast* step was 1
  ms, whatever the step, and every one whose fast step was 0.5 ms worked. At
  1 ms the sedan's front spins chatter; single rate averages the chatter
  over its stages, but the multirate bodies see the spins only at two
  instants per step, so the chatter does not average out. The rule: each
  part inside its own stability.
- **A test that compared at the wrong time.** The order test's steps (0.06
  and 0.03 s) did not divide its span of 1 s; the simulator rounds the
  number of steps, so the runs ended at 1.02 and 0.99 s, and the test saw
  first order. Steps that divide the span gave 15.97.
- **A failed scripted edit** left a file unchanged instead of half-written;
  it was checked before going on.

## Verification

Three new test cases (two in the kernel, the refusal one of them, and one
on the vehicles), multirate added to the allocation test; 457 tests, no compiler warning.
Numbers and commands in the data file `2026-10-17_multirate.md`.

- **The method:** on a lag driving a mass, which does not depend on it,
  the step keeps RK4's fourth order: ratio 15.97 when the step halves, error
  6.5e-7 m against a Simpson's-rule estimate of a few 1e-7.
- **The "done when":** with a ratio of 6 (steps of 3 ms, fast steps of 0.5
  ms) the rolling car's wheels stay within 6.4e-5 of free rolling (single
  rate at 0.5 ms: 3.7e-5; at 3 ms they chatter by 48 %), and the sedan's
  drive gives the top speed of single rate at 0.5 ms within 0.14 % and the
  same rest at the end, for a sixth of the dynamics' evaluations and 2.8
  times less time.
- **Cost:** per step four evaluations of the dynamics and 4 m - 1 of the
  states' rates alone.

## Open ends

- The paper's own pairing, an explicit multistep method for the bodies with
  the fast states at the step's start only: one evaluation per step, but a
  history to restart at every lock, event and gear change; with task 12.4,
  which pairs multirate states with the linearly implicit step.
- The bodies' extrapolation uses their acceleration at the step's start
  only; the previous step's would make it one order better.
- All state elements are fast; a per-element choice when slow ones appear.
- The fast evaluation applies every force element; elements the states do
  not read could be skipped.

## For the book

- Multirate methods: which part is fast, and why the order of the two
  passes depends on what the slow integrator needs.
- Extrapolate the smooth part, never the stiff one: three couplings
  measured on one car.
- Aliasing inside an integrator: a chatter that single rate averages and a
  sampled slow step turns into drift.
