# Phase 7: the powertrain

- **Dates:** 27 November 2026
- **Plan:** task 7.8; finding F6; D69
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 7.8: the engine's inertia back with its own speed; an engine map with
engine braking; a clutch or torque converter; a gearbox with shift time and
hysteresis; open, locked and limited-slip differentials; half-shaft
compliance; a centre differential; an electric motor option; a hysteretic
driveline isolator (Bouc-Wen) as a state element; reactions on the correct
bodies. Done when anti-squat matches the geometric value.

## Starting point

The template's `Drivetrain` has no states (task 7.3 made the wheels bodies
and left the engine's inertia out, D62): the engine's speed is the driven
wheels' mean times the ratio, the torque split equally, the brakes inside
it, the drive's reaction on the chassis. Every vehicle scenario uses it.

## What was done

1. `Powertrain` (`mbd/vehicle/powertrain.hpp`), a state element beside the
   `Drivetrain`: its states, laws and reactions as D69 says; `EngineMap`
   for engines and motors.
2. Tests (`tests/vehicle/test_powertrain.cpp`, 7 cases) on a bench (wheels
   on spin joints on a fixed chassis), on a trailing-arm rig for
   anti-squat, and on the sports car.
3. The records; `docs/powertrain.md`.

## Decisions

- D69: a new element rather than a rewrite of the `Drivetrain`; small
  inertias for the side gears rather than an algebraic split; the gearbox's
  input inertia reflected through the ratio.

## What went wrong, and how it was found

- **A car that went to NaN.** The first car run blew up. The converter's
  coupling, brought through a ratio of 12.25 onto a carrier of 0.06 kg m^2,
  is stiff: its rate grows with the engine's speed, past RK4's limit at 0.5
  ms. A real gearbox's input side (the turbine, the input shaft) appears at
  the carriers times the ratio squared; with it (0.05 kg m^2, about 7.5 kg
  m^2 at the carrier in first gear) the run is smooth.
- **Gravity set by a helper.** The anti-squat rig's multipliers were off by
  a constant in both routings; `build_corner_rig` sets gravity, which the
  test had zeroed before calling it.
- **A rig not at rest.** Then they were off by 5 %: `initialize()` had
  projected the velocities onto the bump drivers' rate of 1, the arms turned
  and the wheels spun relative to them, the powertrain took its carrier's
  speed from those spins, and the shafts' damping took 44 N m off their
  torque. With the rig at rest before connecting, the multipliers give the
  geometric value to rounding, both routings. The inboard case had matched
  even before: the wheel's spin joint passes only the balanced part of a
  shaft's torque to the arm, so the torque's size did not matter there.
- **An allowance from the wrong decay.** The locked differential's
  accelerations were tested to 1e-9 after 1.5 s; the slowest shaft mode
  decays to e^-16.5 = 7e-8 of its start by then. Allowed 1e-6.
- **A power limit applied in the wrong place.** The motor's torque was first
  clamped to P / w after blending throttle and regeneration; at half
  throttle above the base speed that gave 105 N m where 70 is right. Both
  curves are now scaled to the power limit before blending.
- **A saturation that was not reached.** With the isolator, the test
  expected Bouc-Wen's z at its bound; the shaft's last oscillations had
  unloaded it a little (0.0196 against 0.02). The test checks the torque
  law with the z there is.

## Verification

- Anti-squat on trailing arms: 40 % from the multipliers and the geometry
  (-(dx_wc / dz_wc) L / h), and 640 % both ways with the reaction on the
  arm, each to 1e-10.
- The torque converter at stall: 207.846 rad/s, 3381 N m a shaft, to 1e-9;
  with the isolator the shaft's torque is k a + k_h z to 1e-9, z in (0.9,
  1] of its bound.
- Open, locked and limited-slip differentials: the wheels' accelerations
  (141.219 and 70.8441, 101.657 rad/s^2) and the viscous coupling's steady
  speed difference (4.57459 rad/s), to 1e-6.
- Engine braking: 398.405 rad/s after 0.5 s, the exponential's, to 1e-9.
  The motor's map: 300, 200, -90, -60 and 70 N m, to rounding.
- A shift: begins at the upshift speed, changes gear 0.15 s later, its
  torque at its middle 5e-10 N m, and no downshift follows.
- The sports car accelerates through its converter to 7.0 m/s in 2 s, its
  shafts' torques equal.
- All 168 vehicle cases pass; 644 tests pass.

## Measurements

In `data/2026-11-27_phase-7-powertrain.md`: the tests' printed values, and
the suite's run.

## Lessons

- Where a driveline meets the bodies, the inertias decide the integrator:
  every ratio squares what it multiplies.
- Helpers that build a rig may set what a test means to control; set it
  after.
- A rig for statics must be at rest everywhere, not only where its
  equations look.

## Open ends

- The `Drivetrain` stays the vehicles' default; a car with the powertrain
  has no brakes until task 7.9 gives them.
- Torque roll of a longitudinal engine, a lock-up clutch, a solid axle's
  propshaft reaction, tables for the converter.
- Not in model files or Python.

## For the book

The driveline as states beside the bodies; anti-squat as virtual work, the
torque's reaction choosing between the wheel centre's path (40 %) and the
contact's (640 %) on the same geometry.
