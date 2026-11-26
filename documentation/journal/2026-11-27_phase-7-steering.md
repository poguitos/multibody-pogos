# Phase 7: the steering

- **Dates:** 26 and 27 November 2026
- **Plan:** task 7.7; findings F9, F10; D68
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 7.7: "A rack body with tie rods as links, pinion ratio, and column
compliance. Input as angle or as torque." Done when Ackermann behaviour
emerges from geometry and the rack moves as one piece.

## Starting point

Task 2.7 had already answered the first half of F10: the rack is one body on
a prismatic joint, its tie rods distances to the uprights, so both inner
points move together. Its driver read a travel set between steps, with a
rate of zero: a change of steering reached the mechanism only through the
projection after the next step, and the steering's rate never entered the
equations (F10's second half). Linkage axles had no pinion, column or
steering wheel; `steering_ratio` and `max_steer_angle` were never read (F9).
The kernel had no constraint between two joints' coordinates. Steered
simple corners compute Ackermann angles by formula; linkage axles take
whatever their geometry gives, which no test had looked at.

## What was done

1. A joint coupler in the kernel (`JointCoupler`): q_a - r q_b = 0, each
   joint's coordinate read from its transform as the driver reads its own
   (the angle reading now one function for both), an angle on the branch
   nearest what the other side gives it; checked like the other
   constraints (J, nu and gamma as derivatives of phi) and on a rack and
   pinion turning 15.6 times.
2. A steering command, held or following a function of time; the rack's
   driver follows one (`set_steering` holds, `steer` follows).
3. The steering column (`add_steering_column`): the pinion and the steering
   wheel on hinges about the column's axis, the coupler, the column's
   torsion spring, the rack's stops, and the input (a driver on the wheel,
   or a torque on it as a state element without states, for the time); the
   pinion's ratio and the stops from the calibrated linkage and the two F9
   fields. The input chosen per car, the rack's travel by default.
4. Tests (`tests/vehicle/test_steering_system.cpp`, 5 cases; the coupler's
   2 in `tests/kernel/test_kernel_constraints.cpp`) and the records.

## Decisions

- D68: the coupler, the command, the three inputs, the ratios from the
  linkage, Ackermann left to the geometry.
- The column only on request: it adds a stiff freedom to every car, and the
  existing users steer by the rack.

## What went wrong, and how it was found

- **A test geometry that was not Ackermann's.** The Ackermann case was set
  up by the textbook's rule, the steering arms aimed at the rear axle's
  centre, expecting about 100 %. A closed-form calculation before the test
  was written gave 46 to 56 %: the rule is for a track rod behind the axle,
  and with a rack ahead of it the same arms give about half. The test now
  states the behaviour (the inner wheel steers more, the axes meet behind
  the front axle on the inside) and reports the share, rather than claiming
  the rule.
- **A sign hidden by an absolute value.** The first calculation of the
  Ackermann share took the larger angle as the inner one, which made the
  anti-Ackermann geometry read +56 %; computed with the left wheel as the
  inner one, it is -127 %.
- **Inertias that no body has.** The pinion and the steering wheel were
  first given inertia about their axis only; validate() refused them
  (MBD-M014, the triangle inequality). They now have 0.6 of it across.
- **A freedom forgotten.** The car with a steering wheel was expected to
  have fourteen freedoms like the others; validate() counted fifteen, and
  sixteen with a torque: the column's twist is a freedom of its own. The
  test was wrong.
- **A check that measured the jolt.** The rack's rate was compared with its
  travel's central difference while the wheel followed a sine from rest at
  its full rate: the column rang and the difference was 1e-4 m/s against an
  allowance of 1e-7. Started on a quintic ramp (rate and acceleration zero at
  the ends) it is 1.3e-8 m/s, near the estimate of 8e-9.
- **A Windows macro.** A variable named `near` did not compile in a test
  (windows.h defines it); renamed there and in the coupler.

## Verification

- The coupler: J, nu and gamma against central differences for four
  pairings (prismatic and revolute, either way, two revolutes, two
  prismatics), within 1e-7 and 1e-6; a pinion of 0.01 kg m^2 and a rack of 2
  kg under 0.5 N m turn 98.0392 rad in 2 s, the closed form, to 1e-9
  relative, the rack at 0.01 times that to 1e-10 m, the multiplier m r
  theta'' to 1e-9; its refusals (MBD-K023).
- The column alone, a spring of 2e4 N/m on the rack: a wheel held at 0.5
  rad turns the pinion to k_c theta / (k_c + k_r r^2) = 0.499680 rad, a
  torque of 2 N m moves the rack to 12.5 mm and the wheel 0.001 rad past
  the pinion, to 1e-9; its two modes, 5.6365 and 2762.35 rad/s from the
  quartic, to 1e-6.
- The sports car's front axle steered by its wheel through a ramp: the rack
  one piece within 2.8e-17 m, geared within 3.9e-18 m and 8.7e-19 m/s, its
  rate against its travel's central difference within 1.3e-8 m/s (allowed
  1e-7 from h^2 s''' / 6).
- Ackermann: for parallel, outward and inward arms, the toes are the planar
  linkage's closed form within 7.0e-12, 8.8e-12 and 9.4e-15 rad (allowed
  1e-10); at 60 mm of rack the shares of the ideal difference -7 %, +56 %,
  -127 %, the axes meeting 48 m ahead, 4.0 m behind the car's centre on the
  left, and 4.5 m ahead.
- The sports car by each input: 14, 15 and 16 freedoms, no redundant
  equations; steered by the wheel or by 1 N m, its left wheel turns left
  (0.010 and 0.106 rad in 0.5 s); the wrong command refused (MBD-V005).
- All 161 vehicle cases pass; the regression scenarios (the default input)
  unchanged. 637 tests pass.

## Measurements

In `data/2026-11-27_phase-7-steering.md`: every margin above with its
command, and the suite's run.

## Lessons

- Work a textbook rule through on the geometry at hand before writing it
  into a test: the rule had a condition (a track rod behind the axle) that
  the setup did not meet.
- A freedom count is a prediction: write down every body and equation, and
  a compliance is a freedom.
- A comparison of a state with its own finite difference measures the
  integration only if the input is smooth: start inputs from rest.

## Open ends

- The column is straight, without universal joints; the rack's ratio is
  constant; no power assistance.
- A steered rear linkage axle keeps a rack driven by its travel.
- The coupler is not in model files or Python.
- The presets' steering geometry is unchanged; its Ackermann share is not
  reported here.

## For the book

A gear as one equation between two joints, and the branch of an angle read
from a rotation; why a steering input must carry its rate into the
equations; Ackermann behaviour as the planar linkage's, with a rule from the
books that holds for one layout and halves for another; a compliance as a
freedom. Candidate figure: the three arm geometries' inner and outer angles
against rack travel, with the ideal Ackermann curve.
