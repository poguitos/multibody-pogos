# Brakes

Plan task 7.9 (decision D70): the brakes as a state element, from the
pedal to a pressure to a torque at each wheel. The template's simple
drivetrain (`mbd/vehicle/drivetrain.hpp`) keeps its own brakes, which the
existing scenarios use; this element is for a car on the powertrain of task
7.8, or for any car whose braking needs the pressure's lag, a proportioning
valve or an anti-lock controller. This page says what it computes, how it
was checked, and what it leaves out.

| What | Where |
|---|---|
| The brakes and their settings | `mbd/vehicle/brakes.hpp` (`BrakeSystem`, `BrakeSystemParams`) |
| Tests | `tests/vehicle/test_brakes.cpp` |

## What is modelled

- **The pressure.** The master cylinder's pressure is the pedal (0 to 1)
  times `max_pressure`. Each wheel's line pressure is a state that follows
  its target with the system's response time: p' = (target - p) / tau.
- **The proportioning valve.** The rear wheels' target follows the master's
  up to `knee_pressure`, then rises by `rear_slope` of it. With the default
  infinite knee there is no valve.
- **The torque and the bias.** The torque a brake can give is its axle's
  gain times its pressure (`front_gain`, `rear_gain`, in N m per Pa: the
  piston area times two pads times the pads' friction times the effective
  radius). The bias, the front's share at equal pressures, is
  front_gain / (front_gain + rear_gain): 0.679 with the defaults.
- **Dry friction and locking.** The brake opposes the wheel's spin relative
  to its upright, in the direction the spin had at the step's start. A
  wheel that comes to rest while its brake can hold the other torques on it
  is locked: its spin is set to rest at each step's end and the brake holds
  those torques. It is released when they exceed the brake. This is the
  template drivetrain's logic (D62). The torque reacts on the upright (an
  outboard brake).
- **The anti-lock controller** (`abs`). At each step's end, above
  `abs_min_speed`, each wheel's braking slip is read from its tyre (Vsx /
  |Vx|, positive when the wheel turns slower than it rolls). Past
  `abs_release_slip` the wheel's target pressure goes to zero; below
  `abs_apply_slip` it returns to the pedal's.

## Using it

```cpp
mbd::BrakeSystemParams bp;
bp.abs = true;
mbd::BrakeSystem brakes(bp);
brakes.connect(sim, vehicle);    // a VehicleHandle or a VehicleModel, after the simulator is placed
brakes.pedal = 0.5;
sim.run(3.0, 5e-4);
```

With the powertrain of task 7.8, give the brakes the shafts' torques, so a
wheel is not locked against its drive:
`brakes.drive_torque = [&pt](int c) { return pt.shaft_torque[c]; };`.

`locked`, `releasing` and `torque_applied` say what each wheel's brake did
in the last step.

## How it was checked

- **The stopping distance matches a hand calculation** (the plan's test).
  The simple car on stiff linear tyres without rolling resistance, from 20
  m/s at half pedal: the deceleration is the brakes' torque over the
  rolling radius, divided by the car's mass plus the wheels' inertias over
  R^2, and it rises with the pressure as 1 - e^(-t / tau). The closed form
  gives 31.5954 m in 3.11036 s; the simulation stopped in 31.6219 m and
  3.109 s (8.4e-4 and 4.4e-4 relative). The allowance, 3e-3, is sized to
  the last metre per second, where the tyre's force fades and the wheels
  lock: at the nominal deceleration it takes 0.077 m, 2.5e-3 of the
  distance.
- **Locked wheels slide at mu g.** On a road of friction 0.3 at full pedal
  every wheel locks and the car decelerates at 2.94202 m/s^2 against 0.3 g
  = 2.943 (3.3e-4 relative).
- **The anti-lock controller** on the Magic Formula tyre, friction 0.3,
  full pedal: without it the wheels lock and the car stops in 100.5 m; with
  it no wheel is locked above 2 m/s and the car stops in 85.3 m.
- The proportioning valve's and the bias's arithmetic, and the refusals
  (MBD-V007).

## What it leaves out

- No brake fade (the pads' friction is constant), no pressure limit per
  wheel beyond the valve, no hydraulic dead time.
- The anti-lock controller is a two-threshold switch on slip, not a
  production algorithm (no wheel deceleration thresholds, no select-low on
  the rear axle).
- The brakes act on the wheel's spin joint; an inboard brake (on the
  differential) is not modelled.
- The combination with the powertrain through `drive_torque` was not
  exercised by this task's tests.
- Not in model files or Python.
