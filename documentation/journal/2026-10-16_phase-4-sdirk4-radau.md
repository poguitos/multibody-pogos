# Phase 4: SDIRK4, its order on stiff bushings, and Radau IIA

- **Dates:** 14 to 16 October 2026
- **Plan:** task 4.4; decisions D33 and D34
- **Commits:** see the index
- **Written:** at the time on 16 October; the work of 14 October (SDIRK4's
  first version, the bushed pendulum, a draft of D33, two failing tests and
  debugging switches) was reconstructed from the uncommitted changes and
  that session's build logs, since the session ended before it was recorded

## Goal

The plan: the five-stage, fourth-order, L-stable SDIRK scheme described by
Haug et al. *Done when it beats 4.3 on the bushed benchmark at tight
tolerance.*

## Starting point

Generalised-alpha (task 4.3) carries stiff models at fixed steps, second
order. Dormand-Prince (4.2) has error control but is explicit. No benchmark
called "bushed" existed.

## What was done

- **14 October.** `Integrator::SDIRK4` on the stacked system y = (q, v, x)
  with the quaternions in their embedding, a filtered error estimate and
  Dormand-Prince's step control; the benchmark, a double pendulum whose
  hinges are stiff damped bushings between free bodies
  (`tests/kernel/bushed_pendulum.hpp`); tests of the stability function,
  L-stability and error control on the stiff two-mass system. At the end two
  tests failed: the error-controlled run on the two masses (step collapsed,
  K056) and the benchmark (K058 at 8 ms), and the code carried switches for
  the difference step, a full-Newton mode and a trace.
- **16 October, morning.** The two-mass test passed with the code as left
  (161 steps, error 5e-9 m, about 1e-8 predicted): its failure in the last
  log predated the last edits. The benchmark's K058 was Newton converging
  slowly: 4 to 40 iterations per stage with J kept from earlier steps, 3 to
  4 with J rebuilt at every iterate. A diagnostic printed the spectral
  radius of the frozen-J iteration and the entries of J that changed:
  torsion against bending, growing with the swing angle. A second
  experiment carried the correction between the frames of the two J's to
  test whether quaternion coordinates were to blame: no change (section 3 of
  the data). The cause is the bushing: its rotation vector puts the tilt
  axis halfway between the markers, so bending makes a torsion moment of
  about k_r phi delta / 2, stiff on a link of torsional inertia 4.2e-4 kg
  m^2, and turning with the swing.
- Hairer and Wanner's Newton tests (rate of contraction, predicted failure)
  replaced the old "slow" rule, with J rebuilt at a failing stage and kept
  across steps by a cost rule; then dropped (below); plain tests kept.
- **The order.** With Newton fixed, the benchmark ran, and SDIRK4 was 1e-3 m
  out after 1 s at 4 and 2 ms, where generalised-alpha at 1 ms was 7e-6. On
  the same pendulum with soft bushings it converged at fourth order, so the
  code was right. Error against time showed no initial transient; a sweep of
  the stiffness showed the order falling as the bushings stiffen (3.6, 2.5,
  1 at 1e6, 1e8, 1e10 N/m). The user was asked how to go on and chose to
  keep SDIRK4 and add Radau IIA within the task.
- **Radau IIA.** The tableau, the transformation and RADAU5's error estimate
  were checked numerically first (data, section 1); the transformation is
  computed from closed forms rather than copied. All its new tests passed at
  the first build, the predicted errors matching to six digits. On the
  benchmark: at fixed 2 ms within 1e-10 m, but Newton failing at 4 ms and
  above; with error control the global error stuck near 0.4 to 7 times the
  tolerance.
- **Newton's error.** A switch on the stopping fraction showed Newton's own
  error dominating the global error; RADAU5's rule, min(0.03, sqrt(tol)),
  for both methods. That broke every fixed-step run at tolerance 1e-10 in
  the first step; a floor at the state's rounding amplified by J mended it.
  A J rebuilt at the step's middle on failure was tried and dropped.
- The final benchmark test, a smooth-model order test, Radau IIA in the
  allocation and drivetrain tests, the debugging switches and the measuring
  test removed, and the documents.

## Decisions

- **D33**: SDIRK4 on the stacked system, kept for smooth models and
  first-order stiffness, with the measured limits in its documentation.
- **D34**: Radau IIA (RADAU5's scheme) for stiff mechanical models; Newton
  stopping at min(0.03, sqrt(tol)) of the tolerance with a floor that
  includes the state's rounding amplified by J; one J and one cost rule for
  both methods.
- Generalised-alpha's Newton now judges progress over two iterations and
  rebuilds its matrix when the average contraction is worse than a half
  (made on 14 October for the same alternating convergence; recorded in
  D34).

## What went wrong, and how it was found

- **A benchmark that could not be won.** The plan's "done when" was written
  before anyone had measured SDIRK4 on a stiff mechanical model. Its stage
  order is 1 (A c - c^2 / 2 is 0.031 to 0.046 in four of its five stages),
  and stiff bushings make the model nearly a constrained one, where such a
  method's order falls towards its stage order (Hairer and Wanner VII.4).
  It was found by measuring the order, not the cost: errors that hardly
  changed from 4 to 2 ms. The literature says this; the plan did not. The
  user decided the way on.
- **A wrong hypothesis, tested before acting on it.** The changing Jacobian
  looked like an effect of quaternions as coordinates (a turned body turns
  the meaning of each quaternion component). Carrying Newton's correction
  between frames would have fixed that; measured, it changed nothing, and
  the change was traced to the bushing's geometry instead. The experiment
  cost one build of one file.
- **The textbook's Newton tests.** Hairer and Wanner's stop (theta / (1 -
  theta) times the correction) and predicted-failure tests assume a steady
  rate of contraction. Here the first correction from the predictor
  contracts by 0.25 to 0.8 and the next by 1e-2 to 1e-4, so the tests gave
  up on solves one iteration from converging, even with a J built at the
  stage itself, and rebuilt J for most stages. Found in a trace of every
  correction.
- **Newton's error unseen by the error control.** With error control Radau
  IIA's global error was several times the tolerance, though its fixed
  steps were far more accurate. The estimate is of third order and the
  method of fifth, so the true truncation error is far below the
  estimate, while Newton, stopped at a tenth of the tolerance, left an
  error of that size in every step, adding up undamped over the swing.
  Found by varying only the stopping fraction (a tenth, a hundredth, a
  thousandth: error 4e-7, 2e-8, 6e-9 at tolerance 1e-6).
- **Newton against rounding, again.** RADAU5's rule at tolerance 1e-10 asks
  for corrections of 1e-15. Task 4.3 had met the same trap and put a floor
  at the rounding of the terms a step adds up; here the noise comes from
  elsewhere: a 1e8 N/m bushing turns the positions' rounding (1e-16 m) into
  forces of 1e-8 N, far above eps times the force. The floor now includes
  the state's rounding carried through J. Found because every fixed-step
  run failed at its first step right after the rule changed.
- **Fixed steps that Newton cannot solve.** Radau IIA's stages span the
  whole step, so on this model it cannot be run at fixed steps of 4 ms or
  more once the links swing fast (K058). Error control shortens the steps
  where needed and is how the method is meant to be used; the test uses it.
  Not resolved for fixed steps: documented (troubleshooting, K058).
- **Line endings.** Several scripted edits silently did nothing on files
  with CRLF line endings (the Markdown and some sources); each was noticed
  in the check that followed it and redone with the endings handled.

## Verification

Seven new test cases (SDIRK4: three; Radau IIA: three; the two together:
one) and the benchmark; Radau IIA added to the allocation, drivetrain and
refusal tests; 447 tests, no compiler warning. Numbers and commands in the
data file `2026-10-16_sdirk4-radau.md`.

- **The methods are the ones intended:** one step of each on the oscillator
  equals its stability function, computed in the test from the
  coefficients, to 2e-16; over 10 s the errors equal N |e^(ih) - R(ih)| a to
  six digits; ratios 15.9963 (SDIRK4) and 31.9565 (Radau IIA) when the step
  halves.
- **L-stability:** a spring at w h = 89.4 shrinks by |R| = 0.104 (SDIRK4)
  and 0.0336 (Radau IIA) in one step.
- **Error control on a stiff system follows the slow motion:** 161 steps
  (170 predicted) and 136 (140 predicted) over 2 s on the two masses, errors
  5e-9 m and 1.7e-10 m.
- **Order on a smooth spatial model:** 16.06 and 32.53 on the soft
  pendulum, two free bodies turning through large angles.
- **The "done when":** on the bushed benchmark Radau IIA with error control
  at 1e-6 ends 1.04e-7 m from the reference after 1 s for 11548
  evaluations; generalised-alpha needs fixed steps of 0.25 ms, 4.59e-7 m,
  26057 evaluations. At 1e-8: 4.1e-9 m for 13986 against about 48600 for
  7.8e-8 m.
- **No allocation** while stepping the sedan or the four-bar with either
  method; 526 evaluations per 100 steps of 1 ms on the sedan for Radau IIA
  (RK4 400, generalised-alpha 300), 1403 for SDIRK4.

## Open ends

- SDIRK4 is kept although no current model needs it; whether it earns its
  place will show when stiff first-order states (tyre lag, actuators) come
  in Phase 7.
- Radau IIA refactorizes its two matrices whenever the step length changes:
  cheap at n = 26, not measured on large models. RADAU5 keeps the factors
  when the step changes little; task 4.8 (sparse solve) is the place.
- No dense output for the implicit methods: events go through the generic
  re-integration (task 3.9). Radau IIA's collocation polynomial would give
  one.
- Fixed steps of 4 ms and more on the bushed pendulum fail in Newton for
  both Runge-Kutta methods; analytic Jacobians (task 4.5) would make
  rebuilding J cheaper, not the coupling slower.

## For the book

- Stage order: why a fourth-order method can be a first-order one on a
  stiff mechanical model, and why the error estimate does not notice; the
  stiffness sweep as the experiment that shows it.
- How a hypothesis is tested before it is acted on: the quaternion
  explanation that a one-file experiment refuted.
- Newton inside an integrator: its error as part of the global error, the
  tolerance-dependent stop, and two kinds of rounding floor.
- A plan's "done when" that measurement overturned, and the decision it
  needed from the author.
