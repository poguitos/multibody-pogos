# Phase 7: mass properties

- **Dates:** 27 November 2026
- **Plan:** task 7.11; finding F9; D72
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 7.11: an explicit centre of mass and full inertia tensor for the
chassis; a lumped driver, fuel and payload; a roll-up report of total mass,
centre-of-gravity height, sprung and unsprung mass per corner, and yaw
inertia. Done when the report matches the inputs.

## Starting point

The chassis was a solid box of `half_extents` with its centre of mass at the
chassis frame's origin. `ChassisConfig::cg_offset` existed but only the lap
model's rough height estimate read it (F9's last open item). There was no
way to put a driver or fuel in the car, and no report. The builder already
computed, per corner, how each part rises with the wheel (`corner_motion`,
task 7.6), to set the springs by virtual work. Earlier the same day, with
the author's go-ahead, `main` was fast-forwarded to task 7.10 (`4d81e40`,
not tagged), and the author chose not to keep the old constant
aerodynamic model as an option (noted under D71).

## What was done

1. `ChassisConfig`: `com` (replacing `cg_offset`), an optional `inertia`
   about it, and `lumped_masses` (`LumpedMass`: name, mass, position, own
   inertia). `chassis_inertia` (`mbd/vehicle/mass_properties.hpp`) adds
   them to the chassis body by parallel axes; without them the chassis is
   the old box, bitwise.
2. The kernel's check of a body's inertia (symmetric, no negative principal
   moment, the triangle inequality) became one function, `inertia_fault`
   (`mbd/model/rigid_body.hpp`), used by `validate()` and by the builder;
   the builder's refusals are MBD-V008.
3. `CornerHandle::unsprung_mass`: by virtual work from `corner_motion`'s
   rates (the parts' weight's rise over the contact's), or the wheel's mass
   for a simple corner.
4. `mass_report(sim, vh)` and `MassReport`, printed as a table: from the
   built bodies as the simulator holds them.
5. The design loads use the lever rule's one function; the placement of a
   car of simple corners takes them instead of the whole mass at the
   origin; the lap model's `cg_height` is the report's;
   `VehicleTemplate::total_mass` counts the lumped masses.
6. Tests (`tests/vehicle/test_mass_properties.cpp`, 3 cases); the records;
   `docs/mass-properties.md`; MBD-V008.

## Decisions

- D72: the report reads the built model, not the template; the unsprung
  mass by virtual work (SAE J670's definition made exact); the sprung mass
  split by the lever rule along the road; the lumped masses part of the
  chassis body; no corner weighting.

## What went wrong, and how it was found

- **The lever rule along the wrong axis.** The first version of the report
  shared the sprung mass along the chassis's X and Y. The car at rest is
  pitched (7.3e-4 rad in the test, from the difference of its axles' tyre
  deflections), so a centre of mass 0.17 m above the chassis's origin lies
  1.2e-4 m further forward along the road than along X: a lever arm
  wrong by that much. Found while writing the hand calculation, which had
  to say where the car stands; the report now measures along the road
  beneath the centre of mass.
- **A contact put straight down.** The double-wishbone check first placed
  the contact straight below the wheel centre, where `corner_motion` takes
  the wheel circle's lowest point in the wheel plane. With static camber
  the two differ sideways by r sin(camber), and so does their rise as the
  upright turns: an estimate, not measured, for one degree of camber (5 mm)
  and a camber rate of the order of 0.5 rad/m, 2.5e-3 of the rise, far
  beyond the test's allowance. Found on reading the test before the build.
- **Values instead of differences.** The first run printed the report's
  values and the inputs' side by side, cut at the console's width, so the
  margins could not be read. The messages now print the differences; the
  test was built and run again (same results).
- **A direction written backwards.** D72's first draft said the centre of
  mass lies further back along the road; a nose-down pitch moves a point
  above the origin forward. Found on reading it again against the
  transformation.

## Verification

- The plan's test. The car: simple front corners, trailing arms at the
  back, axles at +1.2 m and -1.5 m, tracks 1.6 m and 1.52 m; a 1250 kg
  chassis with its centre of mass at (0.08, 0.012, 0.21) m and a tensor
  with products of inertia; a 75 kg driver with an inertia of its own,
  45 kg of fuel, a 60 kg payload. Every number computed again from the
  inputs part by part (wheels as discs, arms as rods, hubs as spheres):
  - total mass 1630 kg exactly (allowed 1e-9 kg);
  - centre of mass to 4.2e-17 m (allowed 1e-12 m);
  - inertia to 4.5e-13 kg m^2, its largest entry 3044 (allowed 1e-9);
  - contacts to 2.2e-16 m, the height of the centre of mass (0.4615 m)
    exactly (allowed 1e-12 m);
  - unsprung 40 kg and 57.5 kg exactly (allowed 1e-9 kg);
  - sprung per corner to 5.7e-14 kg (allowed 1e-9 kg); the corners carry
    the car's mass exactly and its moments to 6e-13 kg m.
- A double wishbone's unsprung mass from positions (the sports car's front
  corner moved by 1 mm and 2 mm, central differences, Richardson's
  extrapolation): 1.4e-9 kg from the builder's 54.7816 kg (allowed 1e-8 kg).
  The differences alone were 3.3e-7 and 1.3e-6 kg, in the ratio 4 that
  their delta^2 error predicts.
- The chassis body: the old box bitwise without the new inputs; a point
  mass's effect on the centre of mass and pitch inertia by hand; seven
  refusals with MBD-V008.
- Regression: no scenario written again. The sedan's (`sedan-drive`,
  `sedan-modes`, `sedan-rough-road`, and `sedan-slalom` of the long ones)
  and `sports-car-turn` are unchanged bitwise; `hatchback-brake` moved by
  1.2e-15 in one channel, 1e-5 of its tolerance; `wishbone-sweep` 6.4e-16,
  as before this task. The HMMWV's long scenarios use no vehicle template
  and were not run.
- The suite: 656 tests, all passing (data file).

## Measurements

In `data/2026-11-27_phase-7-mass-properties.md`: the two reports as
printed, the differences, the regression scenarios' worst channels, and the
suite's run.

## Lessons

- A statics convention needs its frame named. "Lever rule along X" was
  correct for the level car the design loads assume and wrong, by a
  measurable amount, for the same car standing on its tyres.
- When the builder already computes a quantity's ingredients for another
  purpose (here the springs' virtual work), deriving the reported number
  from them keeps the two consistent by construction.

## Open ends

- No corner weighting: a car whose centre of mass is off its centre line
  rests slightly rolled; `set_vehicle_equilibrium` still puts it at the
  design position.
- The presets keep their chassis's centre of mass at the wheel centres'
  height (the sports car's centre of mass 0.30 m above the road, a real
  car's 0.45 m to 0.55 m); changing them would move every vehicle scenario.
- The powertrain's hidden inertias are not in the report; fuel does not
  slosh.
- Not in model files or Python (the template is in neither).

## For the book

Sprung and unsprung mass as statics rather than a list of parts: virtual
work gives an arm's share and shows how an upright's turning moves it (55 kg
by the usual count, 54.78 kg by the motion); a car on four supports, and
the convention that splits its weight; the frame a lever rule must be
measured in once the car pitches.
