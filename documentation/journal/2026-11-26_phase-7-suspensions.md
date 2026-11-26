# Phase 7: the suspensions

- **Dates:** 26 November 2026
- **Plan:** task 7.6; findings F9, F10; D66, D67
- **Commits:** `7de111b` (part 1: hysteresis, the dyno, shaped curves), then see git log
- **Written:** at the time

## Goal

Task 7.6: "Double wishbone, McPherson, five-link, trailing arm, with a solid
axle optional. All built from the constraint primitives with real masses and
centres of mass. Coil-over and pushrod layouts. Damper curves, bump stops,
and an anti-roll bar acting through drop links. Hysteresis as state
elements: a damper with a stiffness in series (a Maxwell element: top
mount, oil and gas compliance) and a Bouc-Wen bushing, checked on a virtual
damper dyno (a sinusoidal stroke; force against stroke and against
velocity). Bushing curves given by a few shape parameters (linear range,
progressive end, stop), so that optimisation can move their shape (D42)."
Done when the motion ratio from geometry equals the one from virtual work,
and the series damper's loop matches its closed-form steady-state response.

## Starting point

The double wishbone and McPherson were built on the kernel (task 2.7) with
each arm's centre of mass at its pivot and fixed boxes for inertias (F9);
their springs were linear spring-dampers preloaded for 4000 N whatever the
car; set_vehicle_equilibrium averaged the axles' heights, leaving the
double-wishbone sedan's chassis accelerating at 1.08 m/s^2 and a coordinate
at 73 (F9). The five-link was only a kinematic corner on the ground; there
was no trailing arm or solid axle. The anti-roll bar was a force element on
the wheels' heights. The force library had no element with states of its
own, and curves were lines or tables. F10's rack (a rigid rack moved as one
piece, input as angle or torque) is task 7.7's; here the tie rods and the
toe link sit on the existing rack.

## What was done

Part 1 (committed first, `7de111b`):

1. A bushing's curve by its shape (`Curve::bushing`): linear, then a cubic
   rise of the slope, flat at both ends, then the stop's line; its force,
   slope and energy in closed form. A rising curve's inverse.
2. `MaxwellDamper`, its state the series stiffness's extension, and
   `BoucWenBushing`, a state per axis with hysteresis; both take what their
   rates need in apply(), as the tyre does.
3. The virtual damper dyno (`run_damper_dyno`): a prismatic joint driven
   through a sine, any element, any integrator.
4. Model files and Python learned the shaped curve.

Part 2:

5. The parts' masses from their joints (`mass.hpp`): arms as rods, uprights
   as their joints' centroid.
6. One builder per corner mechanism (`src/vehicle/corners.cpp`), used by the
   vehicle, the rack's calibration and the analysis rig; the five-link on
   the chassis, the trailing arm and the solid axle new; the pushrod and
   rocker.
7. The spring unit: a `SpringDamper` with the damper's curve, a second for
   the stops measured from the design length, a `MaxwellDamper` for a top
   mount; the bar on drop links.
8. The springs set by virtual work at the design position (the motion's
   rates from Kinematics), and the car placed there.
9. Tests (`tests/vehicle/test_suspensions.cpp`, 11 cases) and the records.

## Decisions

- D66: hysteresis as state elements, the dyno, shaped curves.
- D67: the suspensions, their masses, the springs set by virtual work, the
  bar on drop links (opt-in), a mixed car's simple corners mounted at the
  design height.

## What went wrong, and how it was found

- **A mixed car could not be placed level.** The first placement, every
  wheel at its loaded radius, left the double wishbone with a simple rear
  accelerating at up to 5 m/s^2, and statics moved its chassis by 0.9 mm.
  The tyre loads matched their design values to the newton, and each
  corner's balance checked by hand to 0.1 N. Working the placement through
  by hand found it: a simple corner's mount was at the chassis frame's Z =
  0, its wheel 0.21 m below, while a linkage corner's wheel centre is at Z
  = 0; to put both on the road the chassis pitched by 4.5 degrees, and
  gravity turned into its axes pushed every part. A mixed car's simple
  corners are now mounted where their wheel centres too are at Z = 0.
- **The tyre's overturning moment at rest.** After that, the double
  wishbones and McPherson still began at 0.13 to 0.8 m/s^2. A run with the
  linear tyre (no overturning moment) brought them to 0.01 to 0.03: the
  Magic Formula's Mx at rest (R0 Fz QSX1, about 10 N m) does work along a
  corner whose camber changes with travel. The virtual work now includes
  it, from the tyre evaluated as the tyre on its wheel evaluates it at rest.
- **A damper table that could not be inverted.** The first digressive
  table for the Maxwell test fell so fast at its ends that the PCHIP end
  slope, extrapolated from the last two secants, came out negative and was
  set to zero: the curve stopped rising past its last point. Found by
  working the end slope by hand before the first build; the table was
  changed, and MBD-F057 says why such a curve is refused.
- **An energy rule that was not exact.** The dyno's energy a loop was first
  the sum of force times the stroke's changes, which is short by sin(w h) /
  (w h) (6.6e-6 at 1000 steps a cycle): found by working the sum out for a
  sine before testing it; now force times velocity by the trapezoidal rule,
  exact for a periodic loop.
- **A test that did not test what it said.** Radau IIA's test was meant to
  show it stepping at ten time constants, but its step control chooses its
  own steps; with the dyno's step control exposed and turned off the result
  was the same to the digit (it had taken whole steps), and the tolerance,
  first an order-of-magnitude estimate 7 decades loose, is now the error
  Radau IIA's coefficients give for this system, 1.748e-7 N, which the run
  reproduces to four digits.
- **A tolerance too loose by 4.5 decades.** The Bouc-Wen n = 2 test first
  allowed 2e-10 m from a stiffness estimate that does not apply to a smooth
  flow; the bound from the solution's fifth derivative gives 3e-14 m (5.5e-15
  measured); it now allows 1e-12.
- **Uncompiled files fail the catalogue.** Part 1's suite ran with part 2's
  new files on disk; the message catalogue's test scans the tree and failed
  on their two codes. Part 1 was committed without them.

## Verification

Part 1, `tests/forces/test_hysteresis.cpp` (15 cases):

- The shaped curve against its closed forms at a point in each range (5,
  57.5, 230, 440 N; energies 0.0125, 0.3, 1.65, 5.0 J) to rounding; its
  slope and energy against central differences, within 5.0e-5 N/m and
  2.4e-7 N (allowed 1e-4 and 1e-6 from the third derivatives); its second
  derivative continuous at both joints; inverses to 8.4e-15 relative; the
  refusals (MBD-F056, F057).
- The Maxwell damper on the dyno: from rest within 7.8e-8 N of 2666 N of
  its exact response, transient included (allowed 1e-6 from RK4's error on
  the relaxation, (h / tau)^5 / 120 a step over tau / h steps); its third
  loop against the steady state within 4.6e-6 N (the transient's exp(-20),
  predicted 4.6e-6; allowed 1e-5), its first harmonic's parts 1415.22 and
  2252.39 N and its energy 176.902 J to 1e-8; the energy a cycle at 2,
  15.92 and 50 Hz to 1e-8 (48.58, 196.35, 113.5 J, the peak at w tau = 1);
  with a tabulated damper the rig's work equals loss plus stored energy
  within 1.7e-4 J (allowed 1e-3); Radau IIA at h = 10 tau within 1.748e-7 N,
  its coefficients' prediction (allowed 2e-7); in parallel with a spring,
  the spring's static deflection in statics and after a released motion
  (1e-9 m); its loads balanced; its refusals (MBD-F058, A030).
- The Bouc-Wen bushing: its force on the exact branches (n = 1) within
  2.4e-6 N against a bound of 2.3e-4 N from RK4's error at the rate's kinks
  (J h^2 / 24 per kink, Simpson's worst case); n = 2's first loading within
  5.5e-15 m of z_u tanh (allowed 1e-12 from the fifth derivative); the same
  loop at 0.1 and 10 Hz within 8.2e-10 N (allowed 2.5e-8); about a rotation
  axis through a driven hinge within 6.1e-8 N m (allowed 5.8e-6); its loads
  balanced; its refusals (MBD-F059).

Part 2, `tests/vehicle/test_suspensions.cpp` (11 cases):

- The motion ratio from the geometry and from virtual work for six
  layouts, within 8.9e-16 (allowed 1e-10 of the larger).
- The trailing arm's -10/9, its wheel centre's recession 0.02/0.45 and its
  turn 1/0.45 to 1e-12; the pushrod's rocker chain to 1e-12; the solid
  axle's roll centre on its Panhard rod's line to 2.8e-17 m (roll rate
  2/1.6 exactly).
- Arms' and uprights' centres of mass and inertias from their joints to
  rounding; the template's total mass the model's for nine cars.
- Every linkage car placed within 0.028 m/s^2 of rest (allowed 0.05),
  statics moving the chassis by under 9.8e-6 m (allowed 1e-4) in at most
  two iterations, without relaxation.
- The drop-link bar's roll stiffness 4 k_arb within 4.8e-10 (allowed 1e-7;
  the even energy's fourth-derivative term, 2.5e-9).
- The spring unit's damper curve and stops from the design length; a top
  mount leaves the equilibrium unchanged bit for bit (allowed 1e-7 m).
- Three cars on every architecture: fourteen freedoms by validate(), no
  redundant equations, at rest within 1.6e-12 m over a second (allowed
  1e-6 m).
- The builder's refusals (MBD-V002, V004).

Existing tests: all 156 vehicle cases unchanged and passing.
`sports-car-turn` and `hatchback-brake` written again (D67's consequences
and the data file give old against new); `wishbone-sweep` moved by 1.3e-15
and was left. The sensitivity probe: the sports car 1.1e-2 and 10800, the
hatchback 3.1e-3 and 3110 (before, 4.6e-4 and 463, 4.1e-4 and 407): starting
at rest, their vertical speed is small, and so its tolerance. 630 tests pass.

## Measurements

In `data/2026-11-26_phase-7-suspensions.md`: every margin above with its
command, the Radau IIA analysis (script and output), the regression
scenarios old against new, the sensitivity probe. Part 1's suite ran on
battery (110.7 s, 619 tests), part 2's final run on mains (74.7 s, 630
tests, all passing).

## Lessons

- Before blaming a solver, work the setup through by hand: the placement's
  4.5 degrees were in the template's geometry, not in any equation, and a
  run with a different tyre separated the tyre's share of what was left.
- A test of an integrator's stiffness must hold its step: step control can
  quietly turn a stiff test into a smooth one. And a method's error on a
  linear system can be computed from its coefficients exactly: a tolerance
  from it is a prediction the run must meet, not an allowance.
- An identity checked two ways is only as good as the paths are
  different: here one solves J v = nu, the other the dynamics' multipliers.
- A scenario's tolerance follows its channels' sizes: a car that starts at
  rest has a small vertical speed and so a tight tolerance on it, which the
  sensitivity probe shows.

## Open ends

- Joints are ideal: the suspensions have no bushings yet; compliance and
  elastokinematics are task 9.1's (with suspension-explorer-core as the
  second code).
- The bar on drop links is opt-in; the scenario with bars keeps the force
  element.
- A car of simple corners keeps its approximate placement.
- F9's unread steering settings and the chassis's cg_offset: tasks 7.7 and
  7.11; F10's rigid rack and torque input: task 7.7.
- The new elements and builders are not in Python or in model files.
- The hysteresis elements are not compared with measured loops (Dobre et al.
  2026 would supply them).

## For the book

The motion ratio two ways, kinematics and virtual work, as one identity
seen from both sides of the constraint equations (J v = nu against J^T
lambda); springs set by virtual work to hold the design the hardpoints
describe, rather than found by statics; the mixed car whose geometry pitched
it by 4.5 degrees. The Maxwell element as a complex stiffness (its loss
peaking at w tau = 1) and an integrator's error on it computed from its
coefficients; Bouc-Wen's loop as exact branches, rate-independent; a
quadrature that is exact for periodic loops and one that is not. Candidate
figures: the damper's loops at three frequencies against the ellipse; the
Bouc-Wen loop with its branches; a corner's motion ratio along its travel.
