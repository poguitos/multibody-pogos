# Phase 3's how-to guides

- **Dates:** 20 October 2026
- **Plan:** documentation track H, task H.6 for Phase 3 (owed since its
  gate); the start of branch `phase-5`
- **Commits:** see the index
- **Written:** at the time

## Goal

H.6: "Each phase adds the how-to pages of what it delivers, with examples
that are compiled and run as tests so they cannot go stale. Done at each
phase gate." Phase 3's gate (13 October) did not write them; Phase 4's gate
(19 October) set up the mechanism with the time-integration guide and
recorded Phase 3's as owed. The author asked for them before Phase 5.

## What was done

- **Three guides**, `docs/help/howto/`: forces and drives (tasks 3.1, 3.2),
  analyses (3.4 to 3.7), outputs, contact, events and checks (3.3, 3.8,
  3.9 and the debugging aids of H.4). Twelve examples, each compiled and run
  in `tests/howto/` (`test_howto_forces.cpp`, `test_howto_analyses.cpp`,
  `test_howto_outputs.cpp`), each checked against a value derived from the
  physics or the geometry:
  - a block hanging from a spring-damper settles 9.81 mm lower, where k (L
    - L0) = m g;
  - a rod held by a torsion spring settles where 50 (0.2 - q) = m g d cos q;
  - a driven hinge's torque is I q'' + m g d sin q;
  - the four-bar assembled with its crank held closes at its closed-form
    configuration, the other joints at their closed-form rates;
  - static equilibrium lands on m g / k;
  - the four-bar's kinematics along its crank angle gives the closed-form
    configuration and motion ratios;
  - linearisation gives sqrt(k / m) and c / (2 sqrt(k m));
  - a hinge carries the hanging rod's weight and no moment;
  - recorded angles follow the small-angle solution;
  - a ball rests at the contact's static penetration (m g / k)^(2/3);
  - an event fires at the free-fall time sqrt(2 x 0.5 / g);
  - `validate()` counts the four-bar's one degree of freedom and three
    redundant equations.
- **Shared models** in `tests/howto/howto_models.hpp` (the hanging rod, the
  hanging block, the spring block), the time-integration test moved onto
  them.
- **The help index** lists the four guides, and the rows of its table of
  tasks link to the guides' sections; a script checked that every anchored
  link in the help pages resolves.

## What went wrong, and how it was found

- **`validate()` at the neutral configuration.** The first checks example
  called `validate(fb.sys)`, which checks at every coordinate zero, where
  the four-bar's loop is open by a metre: it warned so (MBD-M040), and
  counted one independent equation there instead of two. The example now
  validates at a closed configuration, and the guide says why.
- **Nothing else failed**: the derived values held at the first run, the
  tightest by a factor of five (the recorded swing, whose bound is the
  nonlinearity's 2.4e-10 rad over 1 s).

## Verification

`test_howto`: 19 test cases (7 for time integration, 12 for Phase 3's
guides) and the check that every example is quoted exactly and every
quoted example exists: 20 examples. The whole suite: 490 tests, no compiler
warning.

## Lessons

- An example is the place where a user copies a call as written: an
  argument left at its default (`validate()`'s configuration) deserves a
  line saying what it means.

## For the book

- Documentation that cannot go stale: examples quoted from the tests that
  run them, and a test that compares the two.
