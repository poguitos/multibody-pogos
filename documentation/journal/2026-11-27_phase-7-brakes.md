# Phase 7: the brakes

- **Dates:** 27 November 2026
- **Plan:** task 7.9; findings F5, F6; D70
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 7.9: pedal to pressure to torque per wheel, the bias, locked-wheel
handling, and an optional anti-lock controller. Done when the stopping
distance matches a hand calculation.

## Starting point

The template's `Drivetrain` brakes inside it, without states: the pedal
times a maximum torque, split by a bias, acting as dry friction that locks
a wheel at rest (D62). The powertrain of task 7.8 (D69) has no brakes, so a
car on it could not stop.

## What was done

1. `BrakeSystem` (`mbd/vehicle/brakes.hpp`), a state element connected to
   a vehicle's four corners: four line pressures lagging their targets, a
   proportioning valve on the rear, torque from each axle's gain, the
   `Drivetrain`'s lock logic, and a two-threshold anti-lock controller on
   the tyres' slip. D70 gives the reasons.
2. Tests (`tests/vehicle/test_brakes.cpp`, 4 cases) on the simple car: the
   stopping distance against a closed form, locked wheels sliding at mu g,
   the anti-lock controller on the Magic Formula tyre, the valve's and the
   bias's arithmetic and the refusals.
3. The records; `docs/brakes.md`; MBD-V007.

## Decisions

- D70: a new element beside the `Drivetrain`'s brakes; pressures as
  states, because their lag is what separates the stopping distance from
  v^2 / (2 d) and what the controller cycles against; the lock logic
  copied from the `Drivetrain` rather than a smooth friction law.

## What went wrong, and how it was found

- **Nothing failed.** The four cases passed at their first build.
- **A comment with the wrong sign.** The header described the controller as
  releasing a wheel whose slip "falls below -abs_release_slip"; the code
  uses the braking slip Vsx / |Vx|, positive when the wheel turns slower
  than it rolls, and releases above +abs_release_slip. The test showed the
  code right (no wheel locked above 2 m/s); reading the header back against
  it found the comment wrong, and it was corrected.
- **An allowance called a bound.** The stopping-distance test's comment
  first said the last metre per second, where the linear tyre's force
  fades (below V_low = 1 m/s) and the wheels lock, could move the distance
  by "at most" 0.077 m. That is the distance the car covers in that metre
  per second at the nominal deceleration, not a bound on the error (a
  smaller deceleration there would take longer). The comment now says what
  the number is.

## Verification

- The stopping distance (the plan's test): from 20 m/s at half pedal (6
  MPa), the simple car on stiff linear tyres without rolling resistance or
  lag. By hand, d_inf = 2 (1.9e-4 + 0.9e-4) 6e6 / R / (m + 4 I / R^2) =
  6.53 m/s^2, rising as 1 - e^(-t / 0.05 s): 31.5954 m in 3.11036 s. The
  simulation: 31.6219 m in 3.109 s, 8.4e-4 and 4.4e-4 relative, allowed
  3e-3. The contributions to the difference (the tyres' slip, the loaded
  radius, the end of the stop) were not separated.
- Locked wheels on a road of friction 0.3 at full pedal: all four locked
  over the measured second, the car decelerating at 2.94202 m/s^2 against
  0.3 g = 2.943, 3.3e-4 relative, allowed 1e-3.
- The anti-lock controller on the Magic Formula tyre, friction 0.3, full
  pedal: without it a wheel was locked in 17880 of 20113 steps above 2 m/s
  and the car stopped in 100.504 m; with it no wheel was locked above 2 m/s
  and it stopped in 85.2558 m. This case has no closed form; it checks the
  direction of the effect.
- The valve: 4.6 MPa at the rear for 6 MPa at the master with a 4 MPa knee
  and a slope of 0.3; the bias 1.9 / 2.8; MBD-V007 for a zero response time
  and for thresholds in the wrong order.
- The suite: see the data file.

## Measurements

In `data/2026-11-27_phase-7-brakes.md`: the tests' printed values, and the
suite's run.

## Lessons

- A comment that states a sign convention is a claim like any other:
  check it against the test that settles it.
- An allowance should say what its number is (a distance at a nominal
  rate, a bound, a measured spread), not borrow the word "bound".

## Open ends

- The brakes and the powertrain together: `drive_torque` connects them,
  but no test runs them on one car.
- No fade, no hydraulic dead time, no inboard brakes; the controller is a
  slip switch, not a production algorithm.
- Not in model files or Python.

## For the book

Stopping distance as the textbook's v^2 / (2 d) corrected by the pressure's
lag and the wheels' inertia, both in closed form; the anti-lock
controller's gain on a tyre whose friction falls past its peak.
