# Phase 3: events

- **Dates:** 13 October 2026
- **Plan:** task 3.9; decision D29
- **Commits:** see the index
- **Written:** at the time

## Goal

Detection of sign changes in user functions, with the step shortened to the
event in variable-step mode. *Done when a bouncing ball's impact times are
correct to the tolerance.*

## Starting point

The simulator stepped at a fixed length with RK4 or semi-implicit Euler, with
callbacks before the forces, for forces and after each step. Variable steps
and dense output arrive with Phase 4.

## What was done

- `Event` (name, a function of the simulator's state, an optional action, a
  watched direction) and `Simulator::events`, `event_tolerance`,
  `max_events_per_step`, `event_log()`.
- `Simulator::step` split: `advance()` is the old step (integrator,
  projection, time, kinematics); with events, `step_with_events()` compares
  each function's sign at the start and end of the stretch, locates the
  earliest crossing by the Illinois method on the step length, re-integrating
  from the start for each trial, stops just past it, runs the action, logs it
  and continues to the end of the step. Without events the old path runs,
  allocation-free as before.
- Code K052 (a chain of events that never ends); kernel document, help
  index, catalogue.

## Decisions

- **D29**: endpoint sign comparison and location by re-integration (dense
  output waits for Phase 4); the state left just past the crossing; a step of
  dt still advances by dt.

## What went wrong, and how it was found

- **A limitation found on paper.** Tracing the planned third test (a ball
  bouncing ever lower until the chain of impacts must be cut) by hand, before
  building, showed it could not work: with only the signs at the ends of
  each stretch compared, a bounce shorter than the rest of the step starts
  and ends below the ground and is not seen, so the ball would pass through
  after about seven impacts rather than chain a hundred. Leaving the state
  just before the crossing would fix bounces but fire every non-reversing
  event twice. The limitation is documented (the `Event` comment, the
  kernel document, K052's entry, D29); the guard against endless chains got a
  deterministic test instead.
- **A test killed by rounding.** That test first reflected the height
  across the watched level after each crossing. It fired five times and
  stopped: the located crossings land so close to 0.5 that `1 - q` rounds to
  0.5 itself, where the function has no sign. Putting the ball back a fixed
  micrometre above the level makes a chain that only the guard ends.

## Verification

Three test cases, 420 tests in all:

- **The bouncing ball (done when):** dropped from 1 m with restitution 0.8,
  10 ms steps: six impacts in 3 s at sqrt(2 h / g) and then flights of
  2 e^k v1 / g, matched to 1.8e-14 s against a tolerance of 1e-10 s (under
  gravity alone RK4 is exact, so the located times are the exact ones).
- **Two events in one step** (a falling ball crossing 0.5 m and 0.3 m between
  0.3 and 0.4 s): logged in the order they happen, to 2.3e-11 and 2.1e-13 s;
  the step still ends at its time and the free fall is untouched (8e-17 m).
- **The endless chain:** cut after exactly 100 events, one K052 warning, the
  step ending at its time.

## Lessons

- Trace a test by hand before writing it when it probes the edge of a
  method; here it found the method's limit before any build.
- Exact-zero function values are a real case after an event, not a
  curiosity; tests that place states at a crossing should keep clear of
  rounding.

## Open ends

- Dense output (Phase 4.2) to locate events without re-integration and to
  see a function go out and back within a step, by sampling the
  interpolant.
- The simulator's step trace (H.4) should log events.

## For the book

- Event location as root finding on the step length; why "just past the
  crossing"; the Illinois method.
- What endpoint sign detection cannot see, shown with the bouncing ball.
