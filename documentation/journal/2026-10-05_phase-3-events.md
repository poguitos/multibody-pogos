# Phase 3: events

- **Dates:** 5 October 2026
- **Plan:** task 3.9; finding F14; decision D30
- **Commits:** this task's commit (the index gives the hash)
- **Written:** at the time, in a cloud session

## Goal

Detection of sign changes in user functions, with the step shortened to the
event in variable-step mode. *Done when a bouncing ball's impact times are
correct to the tolerance.*

## Starting point

The simulator had two fixed-step integrators and a step that integrated,
projected onto the constraints and refreshed the kinematics in one function.
Variable-step integration with dense output is task 4.2, not yet done, so
"the step shortened to the event" had to work with fixed steps.

## What was done

1. `Simulator::step` split: `advance(dt)` does one integrator step, the
   projection and the kinematics; `step(dt)` adds the events around it.
2. `Event` (name, function of the simulator, direction, action, stop) and
   `EventRecord`; `Simulator::events`, `event_tolerance`, `event_log()`,
   `stopped()`. `run()` returns early after a stopping event.
3. The search: the signs at the ends of the remaining step; for each
   crossing, Illinois' regula falsi on the length of a step from the saved
   start; the earliest event; the step to it, the action, the rest of the
   step. At most 100 events per step (K091). An event without a function is
   refused (K090).
4. Five tests (`tests/kernel/test_kernel_events.cpp`).

## Decisions

D30: re-integrate the step to the root (no dense output yet); regula falsi
in Illinois' form; the event lands strictly past its crossing; event
functions read the whole simulator.

## What went wrong, and how it was found

Three of the first five tests failed, and none of the failures was in the
event code, though one led to a change in it:

- **The expected impact times were miscounted.** The test ran 3.4 s
  expecting ten impacts; adding the flights by hand gives the tenth at
  3.58 s, and the run had eight, correctly. The run became 3.7 s.
- **A test asked for something sign changes cannot see.** "Two events in one
  step" was first written as one function rising and falling within a step:
  the signs at the two ends agree, and neither crossing is visible. That is a
  limit of the method, now stated in the help; the test now uses two
  functions crossing in the same step, which is what it meant to test.
- **The chattering test did not chatter, twice.** First a ball bouncing ever
  lower (restitution 0.5), expected to pile up events before its Zeno point:
  it stops bouncing after about 31 impacts instead, because each event lands
  up to a tolerance past the floor, and once the rebound speed is below about
  2e-9 m/s the ball never rises back above it, so the tolerance itself ends
  the sequence. The test became switched Coulomb friction at rest, the
  classic chattering case. That did not chatter either: the slider came to
  rest exactly on a step boundary, the event landed with the velocity exactly
  zero, and the friction switch, written to react to the velocity's sign,
  picked the wrong direction. Two changes: the test's numbers moved the stop
  off the step grid and its action became a plain switch of direction; and
  the event search now counts a point where the function is exactly zero as
  not yet crossed, so that an event always lands strictly past its crossing,
  where an action can rely on the switch having happened. With a velocity
  linear in time, regula falsi finds the root exactly, so this case is not
  rare.

## Verification

| Check | Measured | Bound, and why |
|---|---|---|
| Bouncing ball, the first ten impact times against the closed form, steps 0.01, 0.0137, 0.001 s | 4.3e-10, 3.2e-10, 1.5e-10 s | 1e-8 s: free flight is exact for RK4; each impact adds at most (1 + 2e) x 1e-10 s |
| run() ends on its time grid with events | to 1e-12 s | |
| Pendulum stopped at the bottom, against `K(sin(th0/2)) / w0` | 6.3e-13 s | 1e-8 s |
| Driven slider: two events in one step, in order, directions filtered | 5.3e-14 s | 1e-9 s: the event tolerance |
| Switched friction at rest: chattering reported once, first stop at `0.1 / (F0 - P)` | 100 events; 8.5e-11 s | the tolerance |
| An event without a function | K090 thrown | |

422 tests pass with GCC 13 (417 before).

## Measurements

[data/2026-10-05_events.md](data/2026-10-05_events.md).

## Lessons

- Write the expected values of a test from the closed form by computation,
  not by estimate: the miscounted impacts cost a round.
- When a test of a guard (chattering) does not trigger, find out why before
  changing the test: both times the reason taught something about the
  method (the tolerance ends the Zeno sequence; an exact root is not "past"
  the crossing).

## Open ends

- Dense output (task 4.2): locate events on the interpolant without
  re-integrating, and look inside a step for pairs of crossings.
- Events as the basis of hard stops and impacts with restitution in
  mechanisms (Phase 7, end stops; Phase 9, kerbs), where the penalty
  contact of task 3.8 is not wanted.

## For the book

- Events with fixed steps: cut the step at the event. The bouncing ball's
  impact times to 1e-10 s whatever the step, and the Zeno point the
  tolerance resolves.
- Chattering of switched friction, against the regularised friction of
  tasks 3.1 and 3.8: why the engine regularises.
