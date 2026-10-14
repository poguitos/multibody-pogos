# Phase 4: generalised-alpha, the implicit method for stiff models

- **Dates:** 14 October 2026
- **Plan:** task 4.3; decision D32
- **Commits:** see the index
- **Written:** at the time

## Goal

An implicit method for stiff models: start with generalised-alpha (HHT),
with finite-difference Jacobians first, then the analytic derivatives from
task 3.1. *Done when a spring 1,000 times too stiff for RK4 at 1 ms runs
stably.*

## Starting point

Three explicit methods, all limited by stability on stiff models: RK4 at
|w h| <= 2 sqrt(2) on an oscillator and about 2.8 on a decay, Dormand-Prince
about 3.3, semi-implicit Euler 2. The vehicles already sit near that limit
at low speed: a tyre below its V_low makes a wheel's spin relax fast, at
about 2700 per second by task 4.1's estimate (2800 to 3100 measured during
this task, see below), so RK4 needs steps of about 1 ms or less there.

## What was done

- `Integrator::GeneralizedAlpha` (`src/kernel/simulator_implicit.cpp`): the
  Chung-Hulbert generalised-alpha method in the Arnold-Bruls form,
  parameters from `ImplicitControl::rho_infinity`, positions through the
  joints' exponential maps, the state elements' states by the same formulas
  a level down.
- Newton's method on an iteration matrix from forward differences (2 nv +
  nx evaluations), kept across steps, re-formed for each step length,
  rebuilt when Newton slows, and once more before giving up (K058). A small
  allocation-free LU with partial pivoting.
- The accelerations at each step's start evaluated afresh; the method's
  history restarted whenever something other than its own step changed the
  state.
- `IntegratorStats` counts Newton iterations and iteration matrices. Codes
  K058 (a step not solved), K059 (settings).

## Decisions

- **D32**: generalised-alpha on the accelerations the simulator already
  computes (constraints as for the explicit methods), Newton on difference
  Jacobians, analytic derivatives deferred to task 4.5.

## How the tests were derived

As for Dormand-Prince, the method on a linear oscillator is a recurrence
that can be written out exactly (three lines, in the test file), so the
simulator's steps can be compared with it to rounding, on one mass and, mode
by mode, on two masses linked by a stiff spring. The recurrence also gives
the phase error (data file, part 1): 0.0880 (w h)^2 per radian at the
default rho_infinity = 0.8, and the trapezoidal rule's known 1/12 at
rho_infinity = 1, which checks the recurrence. RK4's limit comes from its
stability polynomial on the imaginary axis: |R(iy)|^2 = 1 - y^6/72 +
y^8/576, at most 1 for y <= 2 sqrt(2). For the state elements' states the
stability of the velocity-level formulas was worked out by hand before
coding (data file, part 2).

## What went wrong, and how it was found

- **A bound that was a guess.** The stiff-spring test first allowed the
  amplitude at most twice its start, from a hand calculation of the first
  step (1.3 times). It reached 2.006 times. At w h = 89 the step matrix's
  three eigenvalues lie close together near -rho_infinity, and such a matrix
  stretches a vector for several steps before it shrinks it. The test now
  runs the recurrence alongside and requires the same largest amplitude, to
  1e-9.
- **Newton against rounding.** At rho_infinity = 1 on the same spring, a
  step failed to converge at t = 0.28 s (K058). Estimating the noise in the
  corrections did not explain it until the size of the terms was written
  down: with accelerations of 8e8 m/s^2 the position update adds terms of
  h^2 |a| = 800 m that cancel to under 0.1 m, leaving about 1e-13 m of
  rounding, which the spring turns into corrections of about 5 eps h |a|
  (the stiffness cancels out of that estimate). Where the velocity passed
  zero, the test's absolute tolerance of 1e-12 m/s was below that noise, so
  Newton could never meet it. The convergence weights now have a floor at
  the rounding of the terms the update adds up, 64 eps (|v| + h |a|):
  negligible beside any sensible tolerance, and a guard against one set too
  fine for the problem.
- **A locked wheel that crept.** With generalised-alpha the brake-to-stop
  test found the locked front wheels turning backwards at 6.7e-7 rad/s. The
  iteration matrix had been built before the lock, so its rows still coupled
  the locked spin to the other states, and Newton's corrections, which only
  need to fall below the tolerance, leaked into it. A lock is a discrete
  constraint and must hold exactly whatever the integrator: the drivetrain
  now sets a locked spin to zero at the end of every step. For that not to
  restart the whole method every step, its history now restarts entry by
  entry: q and v together, each state on its own.
- **RK4 that did not blow up, and an estimate on the wrong side.** The
  vehicle test was written to show RK4 failing at 5 ms on the car rolling at
  0.5 m/s, from task 4.1's estimate that the spins relax at about 2700 per
  second (13.5 at 5 ms, far beyond RK4's 2.785). RK4 ran to the end with the
  right speed. A diagnostic run over steps from 0.5 to 80 ms showed why: past
  its limit RK4's spins do not grow without bound but chatter about free
  rolling, capped by the tyre's saturation (14 % of the speed at 5 ms, 48 %
  at 3 ms), and only at 80 ms does the run become NaN. Narrowing the steps
  located the limit: RK4 keeps the spins to the tyres' own slip (1.3e-5) at
  0.9 ms and chatters by 2.8 % at 1 ms. So the rate is 2800 to 3100 per
  second, and task 4.1's statement that it lay "just inside" RK4's limit at
  1 ms was wrong by a few per cent in the direction that matters: RK4 at 1 ms,
  the step every vehicle test uses, leaves the front spins chattering by up
  to 2.8 % whenever a car runs below V_low. D30 carries a correction, and the
  test now asserts what happens: generalised-alpha at 5 ms within 1e-3 of
  free rolling, RK4 at 5 ms off by more than 5 %.

Everything else passed at the first build.

## Verification

Five new test cases, two extended; 439 tests in all, no compiler warning.
Numbers and commands in the data file `2026-10-14_generalized-alpha.md`.

- **The method is the one intended:** on the oscillator the simulator
  equals the recurrence within 1e-12 over 1000 steps; on the two masses with
  the stiff link, mode by mode, within 1e-10 a.
- **Second order:** error 3.52104e-5 m at h = 0.02 s, the predicted phase lag
  to 0.03 %; ratio 3.9998 when the step halves.
- **The "done when":** a spring 1000 times too stiff for RK4 at 1 ms (RK4
  shown stable at 0.95 times its limit and diverging at 1.05 times) runs for
  1000 steps; with rho_infinity 0.8 the motion, after the transient the
  recurrence also shows (to 2.006 times its start), is gone (5.7e-78 m); with
  rho_infinity 1 its energy is kept.
- **Slow motion beside a stiff link:** error 4.36379e-5 a after 2 s against
  the predicted 4.36568e-5 a.
- **The vehicles:** the rolling car at 5 ms within 3.55e-5 of free rolling
  with the speed of RK4 at 0.5 ms, where RK4 at 5 ms chatters by 14 %; the
  brake-to-stop test passes with all three methods; on the sedan at 1 ms,
  three evaluations a step and no allocation.
- **Refusals:** K059 for rho_infinity 1.5, K058 for a force that is NaN.
- **The iteration matrix:** one per run in every test, the linear ones and
  the rolling car alike.

## Open ends

- Analytic Jacobians from the force elements' derivatives: task 4.5, where
  a fixed cost per step needs them.
- No error control: fixed steps. Task 4.4 (SDIRK) brings an embedded
  estimate.
- Constraints by projection after each step, as for the explicit methods;
  the index-3 or stabilised index-2 form natural to implicit methods is task
  4.7.

## For the book

- Stiffness as a stability limit, not an accuracy one: the spring that RK4
  cannot carry at 1 ms although nothing in the answer moves that fast.
- Numerical damping as a tool: rho_infinity, what it removes and what it
  costs the slow motions (the phase lag measured from the recurrence).
- The vehicle at walking pace: a stiff wheel-spin equation that limits an
  explicit method to 1 ms although the car hardly moves.
