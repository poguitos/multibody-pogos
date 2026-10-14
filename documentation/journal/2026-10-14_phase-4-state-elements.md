# Phase 4: the integrator interface and state elements

- **Dates:** 14 October 2026
- **Plan:** task 4.1; decision D30; the Phase 3 gate before it (`main`
  fast-forwarded to `4a3a757`, tag `v0.4.0`)
- **Commits:** see the index
- **Written:** at the time

## Goal

One interface, with extra first-order states (tyre lag, actuators,
controllers) registered by the elements that own them. This replaces the
pre-step and post-step callbacks. *Done when the drivetrain no longer uses
callbacks.*

## Starting point

The simulator integrated (q, v) only. The drivetrain registered two
callbacks: before the forces it handed its wheel spins to the tyres; after
each step it computed its torques, shifted gear, and advanced the spins by a
linearly implicit Euler step of its own, with the brake as dry friction able
to hold a wheel at zero. That step was robust at standstill but first order
and outside the integrator: about 0.2 % of the braking result came from it
(Phase 1's open end).

## What was done

- `StateElement` (`kernel/state_element.hpp`): size, initial states,
  `begin_evaluation` before the forces, `rates` after the dynamics, `end_step`
  after each step. `Simulator::x`, `attach()`, `state_elements()`,
  `state_offset()`, `state_rates()`, and `acceleration(q, v, x, t)`. RK4 and
  semi-implicit Euler carry x; events save and restore it. The callbacks
  before the forces and after the step are gone; `force_callback` stays.
- The drivetrain is a state element: four integrated spins; engine speed and
  torque from the spins of each evaluation; brake as Coulomb friction on a
  turning wheel; a wheel that reaches rest within a step while the brake can
  hold it is locked at zero, and released when the other torques exceed the
  brake or the pedal is let go; gear shifts at the end of a step. Its callers
  (`initialize`, `connect`) are unchanged.
- Code K044 (an element attached twice).

## Decisions

- **D30**: states integrated with the bodies; the brake's hold as a discrete
  lock; elements attached, not owned.

## What went wrong, and how it was found

- **A test checking a decision against out-of-date state.** The auto-shift
  test compared each shift with the driven wheels' spin *before* the step.
  The old drivetrain decided its shift on that spin (it shifted before
  advancing its own spins); the new one decides at the end of the step, on
  the spins the gearbox then sees, so the check failed by one step (55.5527
  against a shift speed of 55.5656 rad/s, passed within the step). The test
  now checks each shift against the spin it was decided on.
- **A braked wheel that never locked.** A new test, braking to a stop,
  found the car creeping backwards at 0.9 mm/s three seconds later, with two
  wheels turning forwards. The first hypothesis was the tyre's low-speed
  regularisation (below V_low its slip passes through peak friction, which
  could release a lock). A printout of the spins over time disproved it: the
  front wheels never locked at all. Their brake beat the tyre, so each spin
  fell towards zero; but inside a step RK4's stages saw the spin cross zero,
  Coulomb friction flipped sign, and every step ended at a small positive
  spin, so the lock (on a spin that passed zero at a step's end) never fired.
  The wheel chattered about zero with the wrong average torque. A wheel that
  the brake will stop within the next step is now locked at once. With that,
  the fronts lock within 0.2 s of the pedal and slide, all four are locked
  by the time the car has stopped (printed every 0.5 s), and they stay
  locked to the end.
- **A bound from the wrong mechanism.** The same test's first bound on the
  speed left after the stop (1e-6 m/s) was derived from the locked tyres'
  damping (a decay of 150 per second); the measurement (4.4e-7) passed with
  little margin, and the printout showed the real decay, about 3.5 per
  second, set by the body's motion on its suspension. The test now checks
  what that mechanism implies: the residual speed at least halves from one
  half second to the next and ends below 0.1 mm/s.

## Verification

Three new test cases and one rewritten, 427 tests in all:

- **Order through the stages:** a lag state pushing a mass, against the
  closed form: RK4's error falls by a factor between 14 and 18 when the step
  halves, semi-implicit Euler's by about 2. Had the force seen the state at
  the start of each step instead of each stage, RK4 would have fallen to first
  order.
- **The hooks:** begin_evaluation, the forces, rates, four times per RK4
  step, then end_step once; a change made in end_step stays.
- **Attaching twice** is refused (K044).
- **The drivetrain** passes every earlier test (standing start, braking
  against the hand calculation, full braking at the grip limit, coasting,
  auto-shift, FWD, cornering, initialisation), and the new brake-to-stop
  test: every wheel locked at exactly zero, no wheel turning backwards, the
  residual speed decaying and below 0.1 mm/s.
- **The braking artefact is gone:** half-pedal braking gives 5.37867186
  m/s^2 at steps of 1, 0.5 and 0.25 ms, identical to nine digits, against
  5.36914 by hand (+0.18 %, from the hand calculation's static rolling radius
  and neglected slip). The old once-per-step spin update gave 5.357 (-0.22 %)
  at 1 ms.
- `test_alloc` still counts no allocation in 100 steps of the sedan with its
  drivetrain, now evaluated in every stage. The cost added per evaluation is
  a few arithmetic operations per wheel; it was not timed.
- CI run 38533479960 on `aa8876c`: Windows, Linux with sanitizers,
  no-malloc and docs all passed.

## Open ends

- The spin equation is stiff at low speed: RK4 at 1 ms carries the default
  wheels with little margin, explicit Euler does not. Tasks 4.3 to 4.5.
- Locking is decided once per step; an event (task 3.9's machinery inside an
  element) would find the instant.
- The tyre's relaxation length (a lag on slip) is the next natural state
  element (Phase 7).

## For the book

- Hybrid states: continuous spins integrated with the bodies, discrete
  decisions (lock, gear) at step boundaries, and why the brake's hold is not
  regularised.
