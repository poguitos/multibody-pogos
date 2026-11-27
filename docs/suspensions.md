# Suspensions

Plan tasks 7.6 and 7.7 (decisions D66, D67 and D68): the vehicle template's suspensions,
each built from the kernel's joints and constraints with its parts' masses
where their joints put them; their spring units (spring, damper curve,
stops, a damper on a top mount); an anti-roll bar through drop links; the
car placed at rest by springs set to hold it there; and the elements with
hysteresis, checked on a virtual damper dyno; and the steering, a rack geared to a pinion and a compliant column to
a steering wheel (see Steering, below). This page says what is built, how it
was checked, and what it leaves out.

| What | Where |
|---|---|
| The template: axles, hardpoints, spring units, bars | `mbd/vehicle/vehicle_template.hpp` (`AxleConfig`) |
| The corners' mechanisms, the motion at the design position, the spring units | `src/vehicle/corners.cpp` |
| Double wishbone, McPherson, five-link, trailing arm, solid axle | `mbd/vehicle/suspension/*.hpp` |
| Parts' masses from their joints | `mbd/vehicle/suspension/mass.hpp` |
| A damper on a top mount, a Bouc-Wen bushing; a bushing's curve by its shape | `mbd/forces/hysteresis.hpp`, `mbd/forces/curve.hpp` (`Curve::bushing`) |
| The virtual damper dyno | `mbd/analysis/damper_dyno.hpp` |
| The steering: the rack, its column, the commands | `mbd/vehicle/vehicle_template.hpp` (`SteeringConfig`, `SteeringCommand`, `SteeringColumn`); the gear, `JointCoupler` in `mbd/kernel/constraints.hpp` |
| Tests | `tests/vehicle/test_suspensions.cpp`, `tests/forces/test_hysteresis.cpp`, `tests/vehicle/test_steering_system.cpp` |

## The architectures

`AxleConfig::suspension_type` chooses one per axle. Every hardpoint is given
for the axle's left corner, relative to its wheel centre (a solid axle's to
the axle's centre), in ISO 8855 axes; the right corner is its mirror image.
At the design position every wheel centre is at the chassis frame's Z = 0.

| Type | Bodies and joints | Loop equations | Freedoms (and the wheel's spin) |
|---|---|---|---|
| `Simple` | a massless slider on a prismatic joint | none | 1 |
| `DoubleWishbone` | lower and upper arms on hinges, the upright on a ball joint on the lower arm | the upper ball joint (3), the tie rod (1) | 1 |
| `McPherson` | the lower arm on a hinge, the upright (with the strut's tube) on its ball joint | the top mount on the strut's axis (2), the tie rod (1) | 1 |
| `FiveLink` | the upright on a free joint | five links, each a distance (5); the fifth is the toe link | 1 |
| `TrailingArm` | the arm on a hinge, carrying the hub | none | 1 |
| `SolidAxle` | the axle, both wheels on it, on a free joint | two lower trailing links, a central upper link, a Panhard rod (4) | 2 for the axle: heave and roll |

A double wishbone, McPherson or five-link on a steered axle has its tie rod
(toe link) on the axle's steering rack. A trailing arm or solid axle cannot
be steered (MBD-V002).

**Masses** (finding F9). Each arm is a uniform rod from its pivot to its
outer joint, its centre of mass midway; each upright has its mass shared
among the points it joins and its wheel centre, its centre of mass their
centroid; a trailing arm is a rod with its hub at the wheel centre; a solid
axle a rod between its wheel centres; a five-link's links have no mass of
their own. `VehicleTemplate::total_mass()` counts every body the builder
makes.

## The spring unit

A linkage corner's spring unit acts between two points:

- **Coil-over** (`SpringLayout::CoilOver`): from the chassis to the lower
  arm of a double wishbone, the upright of a five-link, the trailing arm, or
  the solid axle; a McPherson's is along its strut, from the top mount to
  the strut's lower mount.
- **Pushrod** (`SpringLayout::Pushrod`, double wishbone only): a pushrod
  (a distance) from the lower arm turns a rocker on a hinge on the chassis;
  the unit acts between the rocker and the chassis.

The unit is a `SpringDamper` (the spring `k_spring`, the damper `c_damper`
or the table `damper_curve`), with a second one for the stops when
`bump_travel` or `rebound_travel` is set: each stop engages once the unit has
shortened (lengthened) by that travel from its design length. With
`top_mount_stiffness` above zero the damper leaves the unit and acts in
series with its top mount, a `MaxwellDamper` (D66).

**The design position and the spring's preload.** The builder finds each
axle's tyre load at rest from the masses it made and where they lie, then
sets each linkage corner's spring so that it holds the corner at its design
position by virtual work along the corner's motion there: with the motion's
rates per unit of wheel-centre rise (the corner alone on a fixed chassis,
`build_corner_rig`, through `Kinematics`),

    F sum(MR) + F_z sum(dz_contact/dt) + M_x sum(camber rate) - g sum(m_i dz_i/dt) = 0,

F the spring's force pushing apart, MR its rate of extension (the motion
ratio), F_z the tyre load, M_x the tyre's overturning moment at rest, m_i
and z_i each moving part's mass and the height of its centre of mass. The
spring's free length follows. `set_vehicle_equilibrium` then puts every
corner at its design position, the chassis raised and pitched so that each
wheel centre is at its tyre's loaded radius; a mixed car's simple corners
are mounted where their wheel centres too are at Z = 0 there. A car of
simple corners only keeps its earlier placement (its regression scenarios
start there).

The motion ratio's sign: the unit's rate of extension per unit of wheel
rise, negative for a unit that shortens in bump.

## The anti-roll bar through drop links

With `arb_links.enabled`, the bar is two halves, each turning in a bearing
on the chassis about the lateral axis, joined by a torsion spring
(`RotationalSpringDamper`); a drop link (a distance) from each half's arm to
the corner (the lower arm of a double wishbone or McPherson, else the
upright, trailing arm or axle). Its torsional stiffness and damping are set
from `k_arb` and `c_arb` (the force at each wheel per unit difference of
their travels): k_t = k_arb / r^2, r the half's turn per unit of its wheel's
travel at the design position. Without it, `install_anti_roll_bars` adds
the bar as a force element, as before.

## Steering

A steered linkage axle (double wishbone, McPherson, five-link) has a rack: a
body sliding across the car, carrying each tie rod's (or toe link's) inner
point; the tie rods are links, distances from the rack to the uprights. A
steered axle of simple corners turns its tyres by Ackermann's angles
instead; a trailing arm or solid axle is not steered.

`SteeringConfig::input` says what steers the front axle:

| Input | What is modelled | Command |
|---|---|---|
| `RackTravel` (default) | The rack, driven to its travel (an actuator, a robot); as before task 7.7 | `set_steering(delta)`: the travel delta times the linkage's calibrated rack travel per radian; `steer(f)`: a travel as a function of time |
| `WheelAngle` | The rack geared to a pinion (a `JointCoupler`), the pinion joined to the steering wheel by the column's torsion spring, the wheel's angle driven | `set_steering(delta)`: the wheel held at `steering_ratio` delta; `steer(f)`: its angle as a function of time |
| `WheelTorque` | As above, a torque on the wheel instead of its angle | `set_steering_torque(T)`, or `steer(f)`: the torque as a function of time |

A function of time's rate and acceleration enter the equations of motion
(finding F10: a value set between steps moved the rack with a rate of
zero). The pinion's ratio is the linkage's mean rack travel per radian of
toe (calibrated over the first 5 mm of travel) over `steering_ratio`, so
that the steering wheel turns `steering_ratio` times the road wheels there;
the rack's stops are at the travel that turns them by `max_steer_angle`.
The column's stiffness and damping, the wheel's and pinion's inertias and
the column's direction are in `SteeringConfig`. With a column the car has
one more freedom, the column's twist (and with a torque, the steering
itself). A steered rear linkage axle keeps a rack driven by its travel.

Ackermann behaviour is the linkage's own: the code has no Ackermann
formula for a linkage axle, and the wheels' angles follow from the steering
arms, the tie rods and the rack. With the rack ahead of the axle, arms
turned outward along the line from the rear axle's centre give the inner
wheel more lock (56 % of the ideal difference at 0.5 rad for the test's
geometry), parallel arms none to first order, arms turned inward the outer
wheel more.

## How it was checked

All in `tests/vehicle/test_suspensions.cpp`, against values derived in the
tests:

- **The motion ratio from the geometry equals the one from virtual work**
  (the plan's test): for the double wishbone (coil-over and pushrod),
  McPherson, five-link, trailing arm and solid axle, the spring unit's rate
  of extension from the mechanism's velocities and the force a bump driver
  needs against a known spring force, from the constraint multipliers, agree
  within 8.9e-16 (motion ratios -0.99, -0.70, -0.89, -1.00, -1.11, and -2
  for the axle's two units).
- A trailing arm's motion ratio is its lever arms' ratio, -10/9, and its
  wheel centre moves back by 0.02 / 0.45 per unit of travel; a pushrod's is
  the lower arm's turn times the rocker's levers; a solid axle rolls about a
  point on its Panhard rod's line (to 2.8e-17 m); all to rounding.
- Parts' centres of mass and inertias where their joints put them; the
  template's total mass is the model's for every architecture.
- Every linkage car is placed within 0.028 m/s^2 of rest (the double-wishbone
  sedan began at 73 before, F9); statics then moves the chassis by under
  1e-5 m, in one or two iterations.
- A bar on drop links gives 4 k_arb in roll at the wheels, as the force
  element does (120000 N/m).
- The spring unit's damper curve and stops, from the design length; a damper
  on a top mount leaves the car's equilibrium unchanged, bit for bit.
- Cars on every architecture have fourteen freedoms (validate()) and stay at
  rest (under 1.6e-12 m in a second).
- **Camber gain from the front-view instant centre** (task 7.12,
  `tests/vehicle/test_subsystems.cpp`). With the arms' axes along X and the
  joints in the wheel centre's front-view plane (the default hardpoints), the
  upright turns in that view about the instant centre. For the double
  wishbone that is where the arms' lines meet; for the McPherson, where the
  lower arm's line meets the perpendicular to the strut through its top
  mount. The camber's rate is the inverse of the front-view swing arm, and
  the wheel centre's sideways motion follows from the centre's height:
  0.1294 rad/m (centre 7.7 m inboard) and 1.239 rad/m (0.81 m) from the
  mechanism, equal to those predictions within 9e-16.
- A five-link laid out as the double wishbone (two links to each ball joint
  from two points of its arm's axis, the fifth along the tie rod) moves as
  the double wishbone, every component of the upright's turn and of the
  wheel centre's path within 5.1e-15.

The steering, in `tests/vehicle/test_steering_system.cpp`: with vertical
kingpins and level tie rods each side is a planar linkage, and the
mechanism's toes are its closed form's within 9e-12 rad for three arm
geometries, from which the Ackermann behaviour above follows; the rack moves
as one piece (its tie rods' inner points within 3e-17 m of its travel),
geared to the pinion to rounding at position and velocity, its rate its
travel's rate of change (within 1.3e-8 m/s); held angles and torques give the
column's and the rack's statics, and the column's two modes their closed
form; cars steered each way have fourteen, fifteen and sixteen freedoms and
turn left. The gear itself (`JointCoupler`) passes the kernel's
constraint checks and turns a rack and pinion through 15.6 turns as one
inertia, to rounding (`tests/kernel/test_kernel_constraints.cpp`).

The hysteresis elements and the dyno: `docs/kernel.md` (hysteresis) and D66.

## What it leaves out

- Joints are ideal: no bushings in the linkages yet, so no compliance under
  load (elastokinematics belong to task 9.1). A wishbone is one hinge with
  one pivot point, not two bushings.
- The parts' shapes are not known: rods and lumped points give their
  inertias, and their radii only keep the inertias positive.
- The solid axle is the three-link and Panhard kind only (no leaf springs,
  no Watt's linkage, no four-link).
- The tyre at rest is taken as it is evaluated there: its overturning moment
  is in the spring's setting; nothing else of it acts at rest.
- The steering column is straight, from the pinion to the wheel, without
  universal joints; the pinion's ratio is constant (no variable-ratio rack);
  no power assistance (task 7.x or later).
- The new elements and builders are not in the Python interface or in model
  files (a model file refuses the coupler by name).
