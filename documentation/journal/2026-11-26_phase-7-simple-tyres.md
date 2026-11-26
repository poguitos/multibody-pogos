# Phase 7: the simple tyres

- **Dates:** 26 November 2026
- **Plan:** task 7.5; D65
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 7.5: "Simple tyres. Linear with saturation, Fiala and brush models, kept
for real-time use and for teaching." The plan gives no "done when"; D65 sets
one: each model reproduces its closed forms, its stiffnesses and its
friction limit, carries a vehicle (the car held on its brakes on a slope
stays put on each), and has its cost per evaluation and per vehicle step
measured against the Magic Formula's.

## Starting point

Task 7.4 left the tyre on a wheel (`FullTireForce`) holding a Magic Formula,
its lag (two deflection states), low-speed terms and mirroring written
around it; the old simple formula of `mbd/forces/pacejka.hpp` served only the
bicycle model. Pacejka's chapter 3 gives the brush model in full; for the
Fiala tyre the book has only Fiala's 1954 model in words (a flexible
carcass, pure slip), while the multibody codes' Fiala tyre is MSC Adams's.

## What was done

1. Project Chrono's `ChFialaTire` (BSD-3) was fetched into
   `C:\dev\references\chrono-fiala\`, read-only, to fix what the industry's
   Fiala computes: the brush model's side force and torque at pure side slip
   (with the tyre's width for the half contact length), a longitudinal force
   linear then saturating, a friction falling with the combined slip, the
   two forces independent. PyChrono in the cross-check environment has no
   vehicle module, so Chrono's Fiala could not be run.
2. An interface, `TyreModel` (`mbd/forces/tyre_model.hpp`), took the Magic
   Formula's input and output as its own; the Magic Formula implements it,
   and `FullTireForce` holds any model, its lag, fade and mirroring unchanged.
3. The three tyres (`mbd/forces/simple_tyres.hpp`): linear with the friction
   circle, Adams's Fiala, the book's isotropic brush model with combined
   slip; MBD-F055.
4. Vehicles take a model (`VehicleParams::tyre_model`,
   `AxleConfig::tyre_model`), and the bicycle model reduces any model.
5. Tests against the closed forms, each written out in the test; the tyre on
   a wheel with each model; the slope test on each; a benchmark of each.

## Decisions

- D65: one interface; the three tyres and their sources; vehicles take any
  model; the "done when" above.
- Fiala as the industry has it, not the book's flexible-carcass model, whose
  formulas the book does not give.
- The brush model isotropic: its combined slip has a closed form only then.

## What went wrong, and how it was found

- **Chrono's Fiala could not be run.** PyChrono in the cross-check
  environment has no vehicle module. The Fiala tyre is checked against its
  closed forms only, written out from the source read.
- **A central difference that is not the slope.** The brush model's slope
  test first allowed 1e-8 for a central difference of 1e-7, assuming a
  smooth curve, and missed by 4.4e-7. The curve is odd but its quadratic
  term goes with |s|: the difference is C (1 - theta h + (theta h)^2 / 3),
  not C. The test now asserts that expression (and the stiffness itself
  through characteristics()).
- **Escapes in scripts.** A Python edit script, given through a shell
  here-document, turned the two characters backslash-n in a C++ string into
  a line break (the benchmark failed to compile) and backslash-b in a
  Windows path into a backspace (two docs, found by a search for control
  characters). Text with backslashes goes through the editor, or is written
  with doubled backslashes.
- **A count that caught a wrong replacement.** Replacing `axle.tyre.`
  everywhere would also have hit `t.front_axle.tyre.`; the edit script's
  asserted count (5 found, 3 meant) stopped it before it wrote anything.
- **On battery.** The suite (129 s against about 70 on mains) and the
  benchmark ran on battery power; the benchmark's rows are compared with
  each other only.

## Verification

- `test_simple_tyres` (10 cases): the linear tyre's stiffnesses, circle,
  trail and road friction; the rolling resistance, relaxation lengths and
  body of all three, and their refusals (MBD-F055); Fiala's longitudinal
  force on both branches, continuous at kappa_c (1e-3 N across 2e-9), its
  side force and torque against the brush model's (3.11) and (3.12) with the
  width (1e-13), its torque's peak, full sliding, the friction falling with
  the combined slip and the forces not reducing each other; the brush
  model's (3.11) to (3.14) at four slip angles, its torque's peak 27 mu Fz a
  / 256, isotropy against the book's Figure 3.8, its stiffnesses (the exact
  central difference), combined slip (3.48) to (3.50), the locked wheel, the
  scaling with load; the reduction to the simple formula for the bicycle
  model. Tolerances are rounding unless derived in the test.
- `test_full_tire`: on each simple tyre the lag's rates vanish at the
  steady deflections and the forces are the model's (1e-11).
- `test_drivetrain`: a car on each simple tyre, held on its brakes on the 10 %
  slope, stays put (means over 5-6 s and 9-10 s equal to 1e-5 m, speed at 10
  s under 1e-5 m/s) and rolls back by the hand calculation's 3.372 mm within
  -2 % and +10 %: 3.488 (linear), 3.487 (Fiala), 3.552 (brush).
- The Magic Formula's tests, now through the interface, unchanged.
- 604 tests pass.

## Measurements

In `data/2026-11-26_phase-7-simple-tyres.md`: the benchmark's tyre table
(one evaluation: Magic Formula 1.345 us, linear 0.034, Fiala 0.100, brush
0.067; the sedan's RK4 step 325.4 us against 289.1 to 290.6), on battery
power, pinned; the slope runs; the suite.

## Lessons

- A tyre's name is not its model: "Fiala" in the book and "Fiala" in the
  codes are two tyres. Read the code that users will compare with.
- A central difference measures the slope only where the function is
  smooth; for a curve with an |x|^2 term at zero, write what the difference
  gives and assert that.
- In the vehicle step the tyre model is a tenth of the cost: a real-time
  budget is the vehicle's, and the Magic Formula's double evaluation is the
  first saving to look at.

## Open ends

- The simple tyres are not compared with another code: PyChrono's vehicle
  module would let Chrono's Fiala be run on the same slips.
- The brush model has one stiffness; an anisotropic one needs a numerical
  combined slip.
- The Python interface does not expose tyres yet.

## For the book

Chapter 3's brush model as running code next to the Magic Formula: the
theoretical slip that makes combined slip one-dimensional, a central
difference that is not the slope (the brush curve's |s|^2 term), and what a
tyre called Fiala means in the books and in the codes. Candidate figure: the
four tyres' side force, aligning torque and friction ellipse at the nominal
load.
