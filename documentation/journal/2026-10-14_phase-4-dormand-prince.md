# Phase 4: the Dormand-Prince pair

- **Dates:** 14 October 2026
- **Plan:** task 4.2; decision D31
- **Commits:** see the index
- **Written:** at the time

## Goal

Keep RK4 and semi-implicit Euler. Add an embedded pair (Dormand-Prince) with
error control and dense output. *Done when measured convergence order
matches theory.*

## Starting point

The simulator had two fixed-step methods, RK4 and semi-implicit Euler, one
step per `step(dt)`. Since task 4.1 every method carries the state elements'
states x with (q, v). Events were located by re-integrating from the start of
the step. Nothing counted the work an integrator did.

## What was done

- `Integrator::DormandPrince`: the 5(4) pair of Dormand and Prince (1980),
  seven stages, the seventh at the new state. Within each `step(dt)` it takes
  steps of its own under error control and ends exactly at t + dt; the length
  reached carries over to the next call. Each step is projected; the seventh
  stage serves as the next first only when nothing changed the state.
- `StepControl` (adaptive on or off, tolerances, minimum step, maximum steps
  per call) and `IntegratorStats` (steps, rejections, evaluations, last step,
  last error measure), kept for every method.
- The continuous extension (Hairer's DOPRI5 form, fourth order):
  `Simulator::interpolate(t, q, v, x)` within the last step, and the event
  search, which now runs on it, at the end of each of Dormand-Prince's own
  steps, without integrating again.
- The drivetrain's brake keeps its direction through each step.
- Codes K054 (settings), K055 (step below the minimum), K056 (too many steps
  in one call), K057 (interpolation outside the last step).

## Decisions

- **D31**: Dormand-Prince within the fixed output step, events on its
  interpolant, the seventh stage reused only when the state is unchanged, the
  brake's direction fixed per step.

## How the tests were derived

The coefficients were checked before any code was written (data file, part
1): for y' = lambda y a Runge-Kutta step multiplies y by its stability
polynomial R(z), whose coefficients are b^T A^k 1, and the error estimate by
E(z), with e^T A^k 1. For Dormand-Prince R(z) ends with z^6/600 and E(z) =
-97/120000 z^5 + 13/40000 z^6 - 1/24000 z^7, both exact polynomials because
the stage matrix is nilpotent. An undamped oscillator therefore has a
numerical solution known in closed form, step by step, which turns the
tests from "the order looks right" into "the program computes exactly this
number":

- one step equals R(ih) times the initial state, and the error measure
  |E(ih)| a / sqrt(2), to rounding;
- N fixed steps equal R(ih)^N; against the exact solution the error is T
  (h^5 / 3600) to leading order, an amplitude loss of h^6 / 3600 per step;
- with error control, the measure goes as h^5, so the controller reaches
  its target 0.9^5 in one step; that fixes h, and with it the global error:
  0.287 T tol for any amplitude;
- the continuous extension's local error is of order h^5 (computed from its
  coefficients, data file part 2).

## What went wrong, and how it was found

- **A scratch calculation that misled.** The first check of the continuous
  extension's coefficients reported first order, as if the coefficients were
  wrong. The endpoint, computed in the same script, matched h^6/3600 to four
  digits, so the stages were right; printing the values showed the "exact"
  solution was wrong: PowerShell parsed `New-Object Complex(0, $x)` in
  command mode and built a different number. With the arguments passed
  explicitly the extension came out fourth order, as published.
- **A brake that would have stopped the integrator.** Before running
  anything, working through what error control does to the drivetrain: the
  brake's sign followed each evaluation's spin, so when a spin crosses zero
  within a step the stages disagree on the sign, the error estimate stops
  shrinking with the step, and the step would shrink to about 1e-9 s and run
  into the step limit at every stop. The brake now keeps the direction it
  had at the start of the step, and the lock and the new direction are
  decided at the step's end. The brake-to-stop test runs with both methods.
- **Bounds looser than the derivation.** The tolerance test was first
  written with a band of -15 % to +10 % around the predicted error; the
  measurements fell within 0.6 %. The band was narrowed to what the
  derivation supports, 5 % (the phase error, 1.6 % at the loosest tolerance,
  and one step's share for the first and last steps, 2.5 %).

Everything else passed at the first build.

## Verification

Seven new test cases, three extended; 434 tests in all, no compiler warning.
Numbers and commands in the data file `2026-10-14_dormand-prince.md`.

- **The method is the one intended:** one step equals R(ih) times the
  initial state within 1e-14, and the error measure |E(ih)| a / sqrt(2)
  within 1e-9 of itself.
- **Order (the "done when"):** fixed steps over 10 s, error 2.78733e-9 at h
  = 0.1 s and 8.68805e-11 at 0.05 s, within 0.1 % of the leading term; ratio
  32.08 against 32 (32.06 with the phase term). Fifth order. The states equal
  R(ih)^N within 1e-13.
- **The continuous extension:** mid-step error 2.85683e-10 and 8.87233e-12,
  ratio 32.20: fourth order, the same digits as the scratch calculation.
- **Error control:** global error 1.0059, 0.9995 and 0.9984 times the
  predicted 0.287 T tol at tolerances 1e-7, 1e-9, 1e-11; 40, 100 and 250
  steps (ratio 6.25 against 6.31).
- **Events on the interpolant:** the bouncing ball's impacts to 2.2e-15 s.
- **The vehicles:** Dormand-Prince's half-pedal braking equals RK4's to the
  ten digits printed (5.378671862 m/s^2); the brake-to-stop test passes with
  both, ending at 4.366e-7 m/s; on the simple vehicle and the sedan, one step
  per millisecond, no rejection, seven evaluations against RK4's four; no
  allocation per step with either.
- **The refusals:** K054 to K057 each raised by a case built to need it.

## Open ends

- Output steps shorter than the steps the error control would choose cut
  every step short; on the vehicles at 1 ms, Dormand-Prince costs seven
  evaluations per output step against RK4's four. Longer output steps, or
  output by interpolation where no input changes, would recover it.
- One tolerance for entries of different units; per-entry weights when a
  model needs them.
- The stiffness at standstill (task 4.1) limits Dormand-Prince as it limits
  RK4: its stability boundary on the negative real axis is about 3.3 / h.
  Tasks 4.3 to 4.5.

## For the book

- How a Runge-Kutta method's stability polynomial makes an exact test: the
  oscillator whose numerical solution is known to the last digit.
- Error control and tolerance proportionality: why the global error follows
  the tolerance, and the constant derived for one problem.
- Output steps and integrator steps: why the integrator's state must be its
  own at the output times when inputs change between them.
