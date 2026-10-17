# Phase 4: constraints per integrator

- **Dates:** 17 October 2026
- **Plan:** task 4.7; decision D37
- **Commits:** see the index
- **Written:** at the time

## Goal

Projection for explicit methods, a stabilised index-2 form for implicit
methods, and a drift channel in the output. *Done when drift stays under
tolerance for 1,000 simulated seconds.*

## Starting point

Every evaluation solves the loop constraints at the acceleration level
(index 1), and since task 2.7 every method projected q and v after each of
its steps (positions by Gauss-Newton to 1e-10, velocities exactly). The
step trace (task H.4) recorded the position drift the projection removed;
nothing recorded the velocity drift, and nothing kept the largest values.
Decision D32 had left generalised-alpha's constraints "as for the explicit
methods", the index-3 form being "the subject of task 4.7".

## What was done

- **Measured first.** A temporary probe on the four-bar with a pendulum on
  its coupler: each method's convergence with and without the projection,
  the drift without it over 20 s, and 200 s with it (data, sections 1 to
  3). The projection kept or improved RK4, Dormand-Prince and SDIRK4, and
  touched Radau IIA only at its own floor of 1e-10. Generalised-alpha,
  projected, stayed second order but its error at 4 ms was five times that
  of the same method unprojected, and its drift per step was 300 times
  RK4's in velocity (3.3e-3 m/s at 5 ms): its acceleration-like variable
  lags the accelerations, so each step leaves the constraints by far more
  than its error along them, and the projection takes that from the state
  but not from the history.
- **Read the literature** in the books the author keeps
  (`documentation/Selección`, read with `pdftotext`): Eich-Soellner and
  Fuhrer, section 5.3 (coordinate projection does not disturb a method's
  convergence when the projected values are the ones the method goes on
  from; the GGL "derivative projection" is index 2 and needs a method that
  converges on index-2 problems), and Hairer and Wanner, sections VI.5 and
  VI.9 (the GGL formulation; RADAU5 on a seven-body mechanism alike on the
  index-1, -2, -3 and GGL forms, GGL costing the most linear algebra).
- **First decision:** projection for the one-step methods, implicit ones
  included; the stabilized index-2 form for generalised-alpha.
- **The drift channel:** `ProjectionInfo::initial_velocity_residual`,
  `ConstraintDrift` and `Simulator::constraint_drift()` (the last step's
  drift and residuals and the largest of each), `add_drift_channels()` for a
  `Recorder`, a `velocity_drift` column in the trace; measured after every
  step also when nothing projects. `project_constraints` and
  `projection_tolerance` became `enforce_constraints` and
  `constraint_tolerance`; MBD-K047 refuses a tolerance that is not
  positive.
- **Generalised-alpha in the stabilized form**, in three versions (data,
  section 4): constraint rows of the iteration matrix from the step's
  start, without the derivative of J v (Newton failed at 4 and 5 ms); then
  from the predictor's evaluation, with that derivative from the
  differences that build the matrix (contraction of 5000 per iteration
  instead of 100); then with J v - nu held to Newton's own velocity
  tolerance instead of 1e-10.
- **Tests** in a new file, `tests/kernel/test_kernel_drift.cpp`, on the
  four-bar and on a heavy top held at its tip; the 1000-second case hidden.
  The whole suite passed.
- **Then the cost, measured on the sedan** (data, section 6): the
  benchmark gained evaluations and Newton iterations per step, and the
  previous commit's was built from an archive with the same change, both
  run pinned in one session. Generalised-alpha's step went from 98 to 169
  us for the same evaluations and iterations. Eliminating eta from the
  Newton system through the factor of J M^-1 J^T (exact: the same iterates,
  to the digit) brought it to 145 us. Timings of the pieces showed the LU
  only part of it, and Eigen's blocked LU only a quarter faster than ours
  at these sizes; most of the rest is forming the blocks. On a manoeuvre
  with a steering ramp the stabilized step took 12 to 23 % more evaluations
  and came out 1.15 times more accurate in positions, alike in velocities.
- **Second decision** (D37): the stabilized form stays, as an option
  (`implicit_control.stabilized_index2`, off), and generalised-alpha
  projects by default like the rest; the tests, the allocation test and the
  benchmark run both.

## Decisions

- **D37**: projection after every method's steps; generalised-alpha's
  stabilized index-2 form as an option; the drift channel.

## What went wrong, and how it was found

- **The plan's split did not survive the measurement.** "Stabilised index-2
  form for implicit methods" assumed the implicit methods were the ones a
  projection hurts; the probe showed the one-step implicit methods unhurt
  and the one method with a history hurt. And for that one method the form
  costs more than it gains on our models. The decision follows the
  measurements and the theory rather than the plan's wording.
- **A decision was taken before its cost was measured.** The stabilized form
  was made generalised-alpha's default on the four-bar's evidence, where its
  linear algebra is negligible (four velocities, five equations). Only the
  benchmark on the sedan, with 17 equations beside 31 unknowns, showed what
  its Newton matrix costs. The cost of a numerical choice must be measured
  on the model it is for, before the documents are written around it; two
  sets of documents were rewritten here.
- **A first explanation was half wrong.** Projected generalised-alpha was
  three to five times less accurate than unprojected, and I took that to be
  the projection's doing. Solved on the constraints, the method came out
  with the same error as projected at short steps (3.4e-6 against 3.5e-6
  at 0.25 ms): the three-fold loss belongs to holding this method on the
  constraints at all, whatever the way (the unprojected runs drift off
  them). The stabilized form's gain over the projection is real but
  smaller, 1.5 to 1.1 times at 4 to 1 ms.
- **Newton failed with the first stabilized version.** A temporary print of
  every iteration showed the constraints converging to 1e-16 while the
  corrections fell by only 100 per iteration, and by 10 to 30 when the
  motion was fast: the matrix's J was that of the step's start, and the
  derivative of J v with respect to q was left out, both errors of about h
  |v| / length. J from Newton's first evaluation (free: it is done anyway)
  and the derivative from the matrix's own differences (free as well) gave
  contractions of thousands.
- **Asking J v - nu for 1e-10 inside Newton cost two iterations a step and
  did not deliver.** The last correction goes through the simplified
  matrix, which moves J v by about one per cent of the correction: 4e-9 was
  left. Nothing needs it: the next step holds the positions to phi = 0
  again, so a velocity residual never adds up. It is held to the tolerance
  Newton applies to velocities.
- **The heavy top's first version could not drift.** With its body frame
  at the tip, the spherical closure's equations were the frame's position,
  linear in the coordinates, and every method keeps a linear constraint
  exactly: the "drift" was 1e-17 and a test that the projection had
  something to do failed. The tip moved 0.1 m from the frame.
- **A test assumed the projection always iterates.** When a step's drift is
  already below the tolerance, the projection leaves the positions alone,
  and the residual equals the drift; the test now asks for drift past the
  tolerance somewhere in the run and residuals within it everywhere.
- **Semi-implicit Euler and the linearly implicit step blew up on the
  top.** Over ten seconds they passed; within the first hundred of the
  thousand, their drift reached 1e56 and the projection failed at nearly
  every step. Both take the gyroscopic terms explicitly, and a body
  spinning at 30 rad/s gains energy under them step after step. It is no
  fault of the constraint handling, but it is a limit of the real-time
  integrator (D35 keeps the inertial terms explicit) worth knowing: a
  vehicle's wheels spin about their axles, where these terms are far
  weaker, but a free-spinning body needs another method.
- **A probe that measured the input, not the integrator.** The sedan
  manoeuvre's errors were of first order for every method, RK4 at 1 ms no
  better than generalised-alpha: the steering is set once per step, and its
  sampling dominates. The comparison between the two forms at equal steps
  stands; the orders do not.
- **A run slowed nine-fold.** The 1000-second test, about 12 minutes alone,
  took 90 while the old benchmark compiled beside it with 1.4 GB of memory
  free.
- **Tooling.** A scripted edit put a literal line break inside a C string,
  twice (the replacement's `\n`), and once matched nothing in a file with
  Windows line endings; `tar` read `C:` as a host name (needs
  `--force-local`); Git Bash's `grep -c $'\r'` reported no carriage returns
  in files that have them (perl counts them right); the long test also runs
  when a tag it carries is named, which is how it first ran.

## Verification

Four test cases in `tests/kernel/test_kernel_drift.cpp` and a hidden one;
the allocation test with the stabilized step on the sedan and the
four-bar; numbers in the data file, section 5.

- **Generalised-alpha, stabilized:** the four-bar for 1 s at 2, 1 and 0.5 ms
  against Dormand-Prince at 1e-13: ratios 4.09 and 4.04 (second order: 4),
  residuals at most 1.5e-15 and 1.1e-14, nothing projected; projected
  instead, less accurate at every step; left free, 6.7e-6 off the
  constraints after 1 s.
- **Every method keeps the state on the constraints** for 10 s on the
  four-bar and the top: residuals within 1e-10 in positions, below 1e-12 in
  velocities after a projection (rounding), within the Newton tolerance for
  the stabilized step, no projection failing.
- **The drift channel** gives the same numbers as the trace and the
  recorder at every step; left free, the drift equals what a separate
  solver measures at the state.
- **MBD-K047** for a tolerance of zero or NaN.
- **The "done when"** (hidden, about twelve minutes on a quiet machine):
  1000 s of both models with every method, residuals within the tolerance at
  every step and no projection failing, and the largest drift no larger at
  the end than at the start (semi-implicit Euler and the linearly implicit
  step on the four-bar only, above).

## Measurements

In the data file `2026-10-17_constraints-per-integrator.md`; the sedan's
timings also in `docs/performance.md`.

## Open ends

- The gyroscopic terms in semi-implicit Euler and the linearly implicit
  step (above): for a real-time model with free-spinning bodies, the
  linearly implicit step would need their derivative (the "inertial
  stiffness"), or such bodies need another method.
- The stabilized form's cost: forming its blocks takes more than its LU;
  the solver's own J M^-1 J^T and one triangular solve for M^-1 J^T would
  save part, and a factorization kept across steps while h and the matrix
  hold would save more, at the price of slower Newton. With sparse solves
  (task 4.8) the balance may change.
- Dormand-Prince's last stage could serve as the next step's first when
  the projection changed nothing (it changes the velocities at every
  step, to rounding or more, so today it never does).
- SDIRK4 needs some twenty Newton iterations a step on the four-bar at the
  default tolerances; the stopping fraction from D34 makes it pay for
  accuracy it cannot use there.

## For the book

- Drift and its two cures: projection after the step, and solving the
  step on the constraints (GGL); why a one-step method does not mind the
  first and a method with a history does.
- Generalised-alpha's lagging acceleration and the price of holding it on
  the constraints; the multipliers kappa and eta and what each holds.
- A plan's assumption overturned by one probe, and a choice overturned by
  its cost: measure on the model the choice is for.
- Newton's contraction as a diagnosis: printing each iteration found the
  two missing terms.
