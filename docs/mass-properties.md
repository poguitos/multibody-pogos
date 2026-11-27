# Mass properties

Plan task 7.11 (decision D72): the chassis body is given its own centre of
mass and full inertia tensor, and carries lumped masses (a driver, fuel, a
payload); a roll-up report gives the built car's total mass, its centre of
mass and that point's height above the road, its inertia, and the sprung
and unsprung mass at each corner. This page says what each number means,
how it was checked, and what is left out.

| What | Where |
|---|---|
| The chassis's inputs | `ChassisConfig`'s `mass`, `com`, `inertia`, `lumped_masses` (`mbd/vehicle/vehicle_template.hpp`) |
| The chassis body from them; the report | `chassis_inertia`, `mass_report`, `MassReport` (`mbd/vehicle/mass_properties.hpp`) |
| Each corner's unsprung mass | `CornerHandle::unsprung_mass`, set by `build_vehicle` |
| Tests | `tests/vehicle/test_mass_properties.cpp` |

## The chassis

- **Its own mass, centre of mass and inertia.** `mass` is the chassis body
  alone. `com` is its centre of mass in the chassis frame (whose origin is
  the point the axles are placed from, at the wheel centres' height at the
  design position in a car with a linkage corner; a car of simple corners
  only hangs its wheels below it by its springs' lengths). `inertia` is its
  rotational inertia about `com`, in the
  chassis's axes, products of inertia included with the tensor's minus
  signs (I_xz = -sum(m x z)). Without `inertia` the chassis has the inertia
  of the solid box `half_extents`, about `com`. Without any of them the
  chassis is what it was before (a box with its centre of mass at the
  origin), bitwise.
- **Lumped masses** (`LumpedMass`: a name, a mass, its position in the
  chassis frame, its own inertia, zero for a point mass) move with the
  chassis. The builder adds them to the chassis body by parallel axes, so
  they add no body and no coordinate. Fuel is a rigid mass: it does not
  slosh.
- **Checked when the car is built** (MBD-V008): the masses not negative
  (the chassis's positive), the values finite, and each inertia one a real
  body can have, by the rule the kernel's `validate()` applies to every
  body (`inertia_fault`, `mbd/model/rigid_body.hpp`): symmetric, no
  principal moment negative, the two smaller adding up to at least the
  largest.

```cpp
mbd::VehicleTemplate t = mbd::VehicleTemplate::SportsCar();
t.chassis.com = mbd::Vec3(0.08, 0.0, 0.21);
mbd::Mat3 J;
J << 450.0, 0.0, -50.0,
     0.0, 1800.0, 0.0,
     -50.0, 0.0, 1950.0;
t.chassis.inertia = J;
t.chassis.lumped_masses = {
    {"driver", 75.0, mbd::Vec3(-0.2, 0.37, 0.1), mbd::Mat3::Zero()},
    {"fuel", 45.0, mbd::Vec3(-1.0, 0.0, -0.05), mbd::Mat3::Zero()},
};
mbd::kernel::System sys;
mbd::VehicleHandle vh = mbd::build_vehicle(sys, t);
mbd::kernel::Simulator sim(sys);
mbd::set_vehicle_equilibrium(sim, vh);
std::cout << mbd::mass_report(sim, vh);
```

## The report

`mass_report(sim, vh)` reads the bodies the builder made, in the
simulator's present state; after `set_vehicle_equilibrium` that is the car
at rest at its design position on its road. It does not read the template,
so it shows where the builder put each mass.

- **Total mass**: every body's.
- **Centre of mass**: in the chassis frame.
- **Its height above the road**: from the centre of mass to the road
  beneath it, along the road's normal there.
- **Inertia**: about the centre of mass, in the chassis's axes, the whole
  car as one rigid body (each body's own inertia turned into those axes,
  plus its mass's about the centre of mass). Roll, pitch and yaw inertia
  are its diagonal; the products of inertia are printed too.
- **Unsprung mass per corner**, as SAE J670 defines it: what the tyre
  carries directly. A part joined both to the chassis and to the wheel, an
  arm or a link, is shared. The builder finds the share by virtual work at
  the design position: with the chassis held and the spring taken away,
  the weight that the corner's own parts put on its tyre is the sum of
  each part's weight times the rise of its centre of mass per rise of the
  tyre's contact, from the corner's velocities (`corner_motion`). A wheel
  and an upright count about whole. An arm whose centre of mass is midway
  between its pivot and its outer joint counts about half, the usual rule,
  here not assumed but derived. A simple corner's unsprung mass is its
  wheel's. A solid axle's two corners share its heave equally.
- **Sprung mass per corner**: the rest of the car's mass. Its weight acts
  where the whole car's acts once each corner's unsprung weight is taken
  off at its contact. Positions are measured along the road beneath the
  centre of mass, forward (the chassis's X laid on the road) and to the
  left. Each axle's share is by the lever rule between the axles; the left
  and right shares on an axle by the lever rule between its two contacts.
  A corner's sprung and unsprung mass together is its tyre's load at rest
  over g in that split; the four together carry the car's weight and its
  moments.

The split side to side is a convention. A car on four supports is
statically indeterminate: how its weight goes across the diagonals depends
on the springs. The lever rule on each axle is the split of a car whose
axles each carry their weight as the centre of mass's lateral position
says.

## How it was checked

- **The report matches the inputs** (the plan's test). A car of simple
  front corners and trailing arms at the back, its axles 1.2 m ahead of and
  1.5 m behind the chassis frame's origin on tracks of 1.6 m and 1.52 m. Its
  chassis is 1250 kg with its centre of mass at (0.08, 0.012, 0.21) m and a
  tensor with products of inertia. It carries a driver of 75 kg with an
  inertia of its own, 45 kg of fuel and a 60 kg payload. Every number of the
  report is computed again in the test from these inputs alone, part by
  part: the wheels as discs, the trailing arms as rods with their hubs, as
  `mbd/vehicle/suspension/mass.hpp` describes them. The total mass (1630 kg)
  is exact. The centre of mass agrees to 4.2e-17 m and the inertia to
  4.5e-13 kg m^2 of 3044. The contacts agree to 2.2e-16 m, and the height
  of the centre of mass (0.4615 m) exactly. The car stands pitched by
  7.3e-4 rad, from the difference of its axles' tyre deflections. The
  unsprung masses are exact: 40 kg at the front (the wheel), 57.5 kg at the
  back (the wheel, the hub and half the arm, which turns about its hinge).
  The sprung masses agree to 5.7e-14 kg. The four corners' masses carry the
  car's mass exactly and its moments to 6e-13 kg m.
- **A double wishbone's unsprung mass from positions.** The sports car's
  front corner on its rig is moved by plus and minus 1 mm and 2 mm. The
  parts' weight's rise over the contact's rise, by central differences with
  Richardson's extrapolation, is the builder's value from velocities
  (54.7816 kg) within 1.4e-9 kg. A whole wheel and upright and half the
  arms would give 55 kg. The whole car's corners carry its 1441 kg.
- **The chassis body.** With no new input it is the old box bitwise. A
  100 kg point mass 2 m ahead moves its centre of mass and its pitch
  inertia as the parallel-axis theorem gives by hand. Seven bad inputs are
  refused with MBD-V008, one of them through `build_vehicle`.

## What it leaves out

- **No corner weighting.** The builder sets each axle's springs for the
  axle's mean load, alike on both sides, as a road car's are. A car whose
  centre of mass is off its centre line (a driver on one side) therefore
  rests slightly rolled, not exactly at its design position:
  `set_vehicle_equilibrium` puts it at the design position, and
  `kernel::static_equilibrium` (`mbd/kernel/statics.hpp`) finds where it
  rests.
- The design loads the springs are set for are taken with the chassis
  level; the report measures along the road with the car as placed,
  pitched by the difference of its axles' tyre deflections.
- The powertrain's hidden parts (the engine's inertia, `Powertrain`) are
  not bodies and are not in the report.
- Fuel does not slosh; lumped masses are rigid on the chassis.
- Not in model files or Python (the template is in neither).
