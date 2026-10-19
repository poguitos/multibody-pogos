# Phase 4: the integrators verified

- **Dates:** 19 October 2026
- **Plan:** task 4.9 (verification); decisions D40 and D41
- **Commits:** see the index
- **Written:** at the time

## Goal

The plan's task 4.9: "convergence plots, energy drift tables and
work-against-precision curves for each method. Done when stored in
`docs/integrators.md`." Seven integrators and a multirate mode had been
built and tested one by one (tasks 4.1 to 4.7, D30 to D38); this task puts
them side by side on the same models, measures what each costs for a given
accuracy, and checks the claims made for each.

## Starting point

Each method had tests of its own order on one or two models (the oscillator
for most, the soft bushed pendulum for SDIRK4 and Radau IIA, the four-bar
for the projection) and the sedan benchmark's cost per step. Nothing
compared their accuracy against their cost, and some claims rested on a
single kind of model: the linearly implicit step was described as stable
at any step for stiff springs and bushings, checked on linear springs.

## What was done

- **A verification program** (D41): `bench/verify_integrators.cpp`
  (`mbd_verify`, built on request) runs every method on six models (the
  oscillator, the soft and the stiff bushed pendulum, the four-bar with its
  pendulum, the heavy top, the sedan), writes SVG figures with a small
  plotter of its own (`bench/svg_plot.hpp`; no plotting library is
  installed), CSV data, and the report's tables. Figures checked by
  rendering them to PNG with Edge headless.
- **The oscillator**: every method's state after N steps is known in
  closed form from its stability function or recurrence. Measured against
  that, every energy after 1000 s agreed to 3e-11 or better, and every order
  was exact. A test now keeps the energies (D41).
- **The smooth and closed models**: orders as theory says, through the
  projection (four-bar) and with quaternions (pendulum); the reference by
  Dormand-Prince at 1e-13, checked against Radau IIA at 1e-12 (2e-14 to
  2e-13 apart). The linearisation of each model printed alongside, to
  explain the explicit methods' limits.
- **The stiff pendulum showed the linearly implicit step diverging** at 2,
  1 and 0.5 ms. Probed (data, section 2): the bushings' rotational stiffness
  decides it; the force derivatives are right but far from symmetric; the
  step kept only their symmetric part. With them whole (D40) it runs.
- **Then it gained energy**: the test written for D40, built on a bound that
  holds by derivation (the energy can never rise above its zero start),
  failed at 1 and 2 ms. Probed again (data, section 3): a gain set by k
  h^2 alone, independent of the rotational stiffness: the limit of a single
  linearization per step on stiff springs between turning bodies. The test
  now checks boundedness at 2 and 1 ms and no gain at 0.5 ms, and the
  report says where the limit lies.
- **Generalised-alpha failed Newton** at long steps on the stiff pendulum
  with the tolerance at 1e-12; at 1e-9 (far below its error there) it runs
  at 2 ms. The stiff study uses 1e-9.
- **A discrepancy with D38's data** (the four-bar's energy at 1 ms after
  100 s: -0.528 of the start here, -0.638 there): the same run with the
  dense and the tree factorization separates after 20 s. The model is
  chaotic at that energy (data, section 4); both figures are right.
- **The final run** pinned, 2 minutes; the report written from it.

## Decisions

- **D40**: the linearly implicit step takes the force derivatives whole,
  not their symmetric parts.
- **D41**: the integrators verified by a program that writes the report's
  figures and tables.

## What went wrong, and how it was found

- **A claim checked on the wrong kind of model.** "Stable at any step for
  stiff springs, bushings, contact and tyres" (D35) had been tested on
  linear springs, whose stiffness is symmetric. A bushing carrying a large
  relative rotation is not. The verification found it on the first stiff
  spatial model it ran.
- **A workaround that outlived its reason.** The symmetric parts were taken
  so that a Cholesky could solve the step (D35); D38 replaced that solve by
  an LU for another reason, and the symmetrization stayed. When a change
  removes a constraint on the design, look for what was done only because
  of it.
- **The fix was not the end.** A bounded step can still be wrong: the test
  for D40, written on the energy bound rather than on "it does not diverge",
  found the energy gain the fix left. A bound derived from the physics
  catches more than a check that the numbers stay finite.
- **A first suspect that was real but not the cause.** The derivatives,
  compared with differences of the whole forces, missed about c |v| when
  the links moved (the convective part of a damper's stiffness); removing
  the damping showed the divergence without it.
- **Tight Newton tolerances fail where loose ones would do.** 1e-12 on a
  model of stiffness 1e8 cannot be met at 2 ms; the failure said nothing
  about the method.
- **An apparent contradiction with an earlier record** was chaos, shown by
  the smallest perturbation available (two factorizations that differ by
  rounding).
- **Tooling:** perl replacements wrote real newlines into C++ strings again
  (fixed with the Edit tool); a heredoc of 25 KB failed in Git Bash (the
  report was written with the Write tool); the program's output was
  buffered when redirected, so progress could only be seen at the end.

## Verification

- `tests/kernel/test_kernel_integrators.cpp`, two new test cases: every
  method's energy on the oscillator after 200 steps of w h = 0.5 equals its
  recurrence's (within 1e-10 relative; seen 9.3e-14), semi-implicit Euler
  keeping q^2 + v^2 - h q v (seen 1.2e-15) and its energy within 1 / (1 +-
  h / 2); and the linearly implicit step on the stiff bushed pendulum,
  bounded at 2 and 1 ms (the tip within the links' reach, seen 5e-6 m over),
  without energy gain at 0.5 ms (seen -2.1e-5 J).
- The whole suite: 471 tests, no compiler warning.
- The report's figures and tables: `mbd_verify`, its output in the data
  file.

## Measurements

`data/2026-10-19_integrators-verified.md`: the verification run's output,
the probes behind D40 (divergence by variant, the derivatives against
differences of the whole forces), the energy gain against k and h, and the
four-bar against rounding.

## Lessons

- Verify each claim on the kind of model that could break it: a nonlinear,
  spatial, stiff one, not only the linear one the method was derived on.
- A derived bound (energy cannot rise; the tip cannot leave the links'
  reach) makes a stronger test than "finite" or "close to the reference".
- Explain each stability limit from the model's eigenvalues before
  accepting it: every explicit limit here matched its prediction.

## Open ends

- The linearly implicit step's energy gain on stiff springs between
  turning bodies at long steps: one Newton iteration (a second evaluation)
  would remove it at twice the cost.
- Its stiffness matrix's missing convective term (about c |v|).
- Generalised-alpha's Newton at its longest steps.
- A manoeuvre with events (braking, locking, contact) for the methods with
  error control.

## For the book

- Work-precision diagrams as the way to choose an integrator, with the
  sedan's: on a smooth drive with stiff wheel spins the implicit methods
  with error control win (SDIRK4 37 ms for 2 s of driving at 1e-6 m,
  where RK4 needs 211 ms for 1e-4 m).
- The oscillator as a test with exact answers for every method.
- The symmetric-part workaround and the energy gain: a single linearization
  per step on stiff springs between turning bodies.
- Chaos found by the smallest perturbation at hand.
- Chapter: outline item 33.
