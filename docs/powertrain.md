# Powertrain

Plan task 7.8 (decision D69): the powertrain as a state element, beside the
template's simple drivetrain (`mbd/vehicle/drivetrain.hpp`, which the
existing scenarios use). This page says what it computes, how it was
checked, and what it leaves out.

| What | Where |
|---|---|
| The powertrain, its settings and its engine map | `mbd/vehicle/powertrain.hpp` (`Powertrain`, `PowertrainParams`, `EngineMap`) |
| Tests | `tests/vehicle/test_powertrain.cpp` |

## What is modelled

- **The engine** has its own speed and inertia (a state), its torque from a
  map: a full-load curve and a drag curve against speed, blended by the
  throttle; the drag is the engine braking; above the redline the fuel is
  cut; below idle an idle controller adds torque. `EngineMap::from` makes a
  map from the template's `EngineParams`. An **electric motor**
  (`EngineMap::motor`): its peak torque to its base speed, then constant
  power, with regeneration at zero throttle; it is coupled directly
  (`Coupling::Direct`), its inertia carried by the driveline.
- **The coupling**: a torque converter (`Coupling::TorqueConverter`; its pump
  torque w_e (w_e - w_t) / K^2, (w_e / K)^2 at stall; its torque ratio
  falling linearly from its stall value to 1 at the coupling point), or a
  clutch (`Coupling::Clutch`; its torque the engagement times its capacity
  times tanh(slip / slip speed), a smooth dry friction).
- **The gearbox**: its ratios, final drive and efficiency; shifting by the
  engine's speed (`automatic`) with a gap between the up and down speeds
  (the hysteresis), or the driver's `gear`. A shift takes `shift_time`: the
  gearbox's torque falls to zero and back, the ratio changing midway.
- **The differentials**: each driven axle's open, locked or limited-slip
  (viscous, and a clutch pack with preload and a share of the input torque,
  smoothed); for four-wheel drive a centre differential splitting the
  torque, with an optional viscous coupling across it.
- **The half-shafts**: compliant, a torsion spring and damper each, their
  twists states; optionally a hysteretic isolator in each (Bouc-Wen on the
  twist, D42, as `mbd/forces/hysteresis.hpp`'s `BoucWen`).
- **The reactions**: each half-shaft's torque acts on its wheel and reacts
  on the differential's housing, the chassis (inboard) or a solid axle; the
  rate of change of the hidden parts' angular momentum (engine, gearbox,
  carriers) reacts on the chassis.

The states, the equations and their order are at the top of
`mbd/vehicle/powertrain.hpp`. The gearbox's input side (`input_inertia`) and
a direct motor's inertia appear at the carriers times the ratio squared:
without them, a coupling's stiffness brought through the ratio onto a light
carrier is beyond RK4's reach at the car's steps.

## Using it

```cpp
mbd::PowertrainParams p;
p.engine = mbd::EngineMap::from(tmpl.drivetrain.engine);
mbd::Powertrain pt(p);
pt.connect(sim, vehicle);     // after the simulator is made and placed
pt.throttle = 1.0;
sim.run(2.0, 5e-4);
```

## How it was checked

All in `tests/vehicle/test_powertrain.cpp`, against closed forms:

- **Anti-squat equals its geometric value** (the plan's test): on trailing
  arms, the half-shafts' torque reacting on the chassis, the bump drivers'
  multipliers give 40 %, -(dx_wc / dz_wc) L / h, to rounding; reacting on
  the arm instead (as a brake's would), 640 %, the contact's path.
- A torque converter at stall turns the engine at K sqrt(T) (207.846 rad/s)
  and gives each shaft half of eta g TR_0 T (3381 N m), to 1e-9; with the
  Bouc-Wen isolator the shaft's torque is its spring's and the hysteresis'.
- Open, locked and limited-slip differentials against their rigid limits
  (the accelerations, and the viscous coupling's steady speed difference),
  to 1e-6.
- Engine braking against the exponential of a linear drag, to 1e-9; the
  motor's map at its power limit and in regeneration, to rounding.
- A shift starts at the upshift speed, passes through zero torque, changes
  the ratio at its middle, and is not followed by a downshift.
- A car accelerates from rest through its converter, its shafts' torques
  equal.

## What it leaves out

- The hidden parts turn about the chassis's lateral axis (a transverse
  engine); a longitudinal engine's torque roll is not modelled.
- A shift's lost torque is dissipated in the gearbox, not modelled as
  clutch-to-clutch slip; there is no lock-up clutch in the converter.
- A directly coupled motor's inertia is shared between two axles by the
  torque split, an approximation for four-wheel drive.
- On a solid axle the half-shafts react on the axle; the propshaft's torque
  about the car's long axis (the axle's jacking in roll) is not applied.
- The template's `Drivetrain` stays the vehicles' default; the powertrain
  is connected by hand and is not in model files or Python.
