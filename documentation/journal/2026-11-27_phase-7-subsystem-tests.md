# Phase 7: subsystem tests

- **Dates:** 27 November 2026
- **Plan:** task 7.12; D73
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 7.12: an analytic check for every subsystem.

## Starting point

Each of tasks 7.1 to 7.11 ended on a check against a closed form for what
it built: the road's shapes and spectra, the wheel's precession, the tyre's
stiffnesses and peaks, the simple tyres against Pacejka's chapter 3, the
hysteresis elements' closed-form loops, the suspensions' motion ratios, the
steering's Ackermann geometry, the powertrain's anti-squat, the brakes'
stopping distance, the aerodynamic map, the mass roll-up. No list said
which subsystems, or which parts of them, were left without one.

## What was done

1. A table in `docs/validation.md` (section 9): every subsystem of Phase 7,
   its analytic check, the file that runs it. Writing it showed two gaps:
   - the suspensions' kinematics beyond the motion ratio: the camber gain
     was checked only for its sign (double wishbone, McPherson) or for
     staying bounded (five-link);
   - the brakes' hook for a drive's torque (task 7.9) had never run with
     the powertrain.
2. `tests/vehicle/test_subsystems.cpp`, four cases closing them:
   - the double wishbone's and the McPherson's camber gain and wheel-centre
     path against the front-view instant centre;
   - a five-link laid out as a double wishbone against the double wishbone;
   - the brakes holding the car against the powertrain below the threshold
     and letting it go above.
3. The records; `docs/suspensions.md` gains the instant-centre checks.

## Decisions

- D73: a table of the checks first, then only the gaps it showed; the
  instant centre rather than the roll centre (which the library does not
  compute until task 9.1); what is checked in direction only, named.

## What went wrong, and how it was found

- **An allowance built on an incomplete estimate.** The brakes' case first
  held the car for 1.5 s after raising the throttle and allowed 1e-6 of the
  half-shafts' torque, from an estimate of the driveline's ringing that
  decayed as exp(-21) over the hold. The shafts read 367.987 N m against
  368, 3.4e-5 off, and the case failed. The estimate had left out the
  motor's and the gearbox's input inertia reflected through the ratio of 8,
  (0.05 + 0.02) x 64 = 4.48 kg m^2 of the 4.54 the carrier carries; with
  it the decay over 1.5 s is exp(-6.6), and the ringing left by the 0.5 s
  ramp, about 17 N m by an undamped estimate, falls to about 0.02 N m,
  the size seen. The physics was right and the allowance wrong; the hold is
  now 4 s (exp(-17.6)), and the shafts read 368 within 6.6e-10.
- **An overshoot misquoted.** The comment first said a throttle step would
  ring the shafts to 1.9 times their static torque; for a damping ratio of
  0.066 it is 1 + exp(-pi 0.066 / sqrt(1 - 0.066^2)) = 1.8. Found on
  reading the comment again against the numbers.

## Verification

- Double wishbone (the sports car's front corner): the instant centre at
  y = -6.946 m, z = 1.095 m (7.7 m inboard); the camber's rate 0.129428
  rad/m against the inverse of the swing arm within 1.9e-16, the wheel
  centre's sideways rate 0.141689 within 2.8e-17 (allowed 1e-10).
- McPherson (the hatchback's): the instant centre at y = -0.047 m, z =
  0.018 m; camber rate 1.23919 rad/m within 8.9e-16, sideways rate 0.0221
  within 8.7e-17.
- The five-link laid out as the double wishbone: every component of the
  upright's turn within 5.1e-15 and of the wheel centre's velocity within
  1.4e-15 of the double wishbone's (allowed 1e-12).
- The brakes against the powertrain (the sedan, a 100 N m motor through 8
  at 0.92, an open rear differential, rear brakes only): held at 1.2 times
  the shafts' torque, the car moved 0.16 mm over the ramp and 4 s, the
  rear wheels locked, the shafts at 368 N m within 6.6e-10 (allowed
  1e-6), the brakes carrying it within 3.2e-7 (allowed 1e-3: they also
  carry the tyres' moment, 1.2e-4 N m); at 0.8 times, the rear wheels
  turned, the car drove 0.094 m in 1 s, each brake at 294.4 N m within
  1.0e-9 (allowed 1e-6).
- The suite: see the data file.

## Measurements

In `data/2026-11-27_phase-7-subsystem-tests.md`: the tests' printed values
and the suite's run.

## Lessons

- An allowance is only as good as the estimate behind it: when a closed
  form is right and a margin fails, look first at what the estimate left
  out (here an inertia reflected through a gear ratio, larger than
  everything else in the driveline).
- A list of what each part is checked against finds gaps that the parts'
  own tests, each complete for its task, do not show.

## Open ends

- The anti-lock controller, and a car driving off once its brakes let go,
  are checked in direction only.
- The five-link's default layout has no closed form of its own; its
  equivalence to a double wishbone and its motion ratio cover its code.
- Roll centres and swing-arm lengths as outputs: task 9.1.

## For the book

The front-view instant centre as an exact statement about a mechanism's
velocities, when the axes allow it, rather than a drawing; two links to one
point as an arm; a dry-friction brake's threshold against a compliant
driveline, and the inertia a gear ratio multiplies.
