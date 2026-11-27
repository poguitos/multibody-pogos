# How to build and drive a vehicle

Build a car from a template, put a driver in it and read its mass report,
analyse a corner alone, and give the car brakes, a powertrain and
aerodynamics from a map (Phase 7, tasks 7.1 to 7.12). Every example is
compiled and run, its result checked against a hand calculation, in
`tests/howto/test_howto_vehicles.cpp`, and this page quotes that code
exactly (task H.6). Each subsystem has its own page for what it models and
leaves out: [suspensions and steering](../../suspensions.md),
[tyres](../../tyres.md), [powertrain](../../powertrain.md),
[brakes](../../brakes.md), [aerodynamics](../../aerodynamics.md),
[mass properties](../../mass-properties.md).

The examples use `using namespace mbd; using namespace mbd::kernel;`. The
vehicle layer's axes are ISO 8855's: X forward, Y left, Z up, a positive
steer a left turn ([conventions](../../conventions.md)).

## Build a car, put a driver in it, and read its mass report

A `VehicleTemplate` holds the whole car: its chassis, its two axles (each
with its suspension, springs, dampers, tyres, steering), its drivetrain.
Three presets start it; change any field before building. Here the sports
car (double wishbones) carries a 75 kg driver beside its centre line:

<!-- example: vehicle-build -->
```cpp
VehicleTemplate t = VehicleTemplate::SportsCar();
t.chassis.lumped_masses = {{"driver", 75.0, Vec3(-0.3, 0.37, 0.15), Mat3::Zero()}};
System sys;
VehicleHandle car = build_vehicle(sys, t);
Simulator sim(sys);
set_vehicle_equilibrium(sim, car);   // at rest at its design position, on a level road
sim.initialize();
const MassReport report = mass_report(sim, car);
std::cout << report;   // mass, centre of mass and its height, inertia, sprung and unsprung per corner
```

The report reads the bodies the builder made, so it shows where each mass
went. With the driver 0.37 m to the left, the car's centre of mass moves
75 x 0.37 / 1516 = 0.0183 m to the left, and the left corners carry more.
The chassis's own centre of mass and inertia tensor are
`t.chassis.com` and `t.chassis.inertia`; fuel and payload are lumped
masses too.

Other choices, each on its page:

- **The suspension**, per axle: `t.front_axle.suspension_type` (simple,
  double wishbone, McPherson, five-link, trailing arm, solid axle), its
  hardpoints (`dwb`, `mcpherson`, `five_link`, ...), the spring layout, the
  damper's curve, stops, a top mount, an anti-roll bar on drop links
  ([suspensions](../../suspensions.md)).
- **The tyre**, per axle: the Magic Formula from a `.tir` file (`tyre`), or
  any `TyreModel` in `tyre_model`, such as `LinearTyre`, `FialaTyre` or
  `BrushTyre` ([tyres](../../tyres.md)).
- **The steering**: by the rack's travel (`car.set_steering`), or through a
  column by the steering wheel's angle or a torque on it
  (`t.steering.input`), held or as a function of time (`car.steer`)
  ([steering](../../suspensions.md#steering)).
- **The road**: `car.set_road(road)` with any `Road`: level, a ramp, a
  bump, a kerb, a sine, a height map, an ISO 8608 random road
  (`mbd/forces/road.hpp`).

## Analyse a corner alone

`build_corner_rig` builds one corner on a fixed chassis, with a driver that
sets its wheel centre's height. The mechanism's velocities, per metre of
that height, are its kinematic rates: here the upright's turn about X, the
camber gain.

<!-- example: vehicle-corner -->
```cpp
System rig_sys;
const CornerRig rig = build_corner_rig(rig_sys, t, 0);   // the front left corner, the chassis fixed
Kinematics k(rig_sys);
k.solve();                          // the design position
const VecX v = k.velocities();      // per metre of the wheel centre's rise (its driver's rate)
Data data(rig_sys.model);
forward_kinematics(rig_sys.model, data, k.q, v);
const Real camber_gain = body_state(data, rig.corners[0].upright_body).w_WB.x();   // [rad/m]
```

The sports car's front corner gives 0.1294 rad/m. Its arms turn about axes
along X, so in the front view the upright turns about the point where the
arms' lines meet, 7.7 m inboard: the camber gain is the inverse of that
distance, and the test checks it to rounding. Moving the driver's target
(`k.t`) and solving again sweeps the corner through its travel
(`sweep_bump_travel`).

## Give the car brakes

`BrakeSystem` turns the pedal into a line pressure at each wheel, lagging
it by the system's response time, and the pressure into a torque by each
axle's gain; it locks a wheel that stops and holds it, and an anti-lock
controller can release one that slides ([brakes](../../brakes.md)). The
example's car is the sedan on linear tyres, from 20 m/s, its wheels
rolling (`roll_wheels`); `speed` is the chassis's velocity along its X:

<!-- example: vehicle-brakes -->
```cpp
BrakeSystem brakes(BrakeSystemParams{});   // 1.9e-4 and 0.9e-4 N m/Pa, 12 MPa at full pedal
brakes.connect(sim, car);
brakes.pedal = 0.5;
while (speed(sim, car) > 0.01) sim.step(5e-4);
```

It stops in 31.62 m. By hand: the deceleration rises as the pressure
does, to d = 2 (1.9e-4 + 0.9e-4) x 6e6 / R / (m + 4 I / R^2), m the car's
mass and I each wheel's inertia, and the distance follows from integrating
it twice; the two agree within 3e-3.

## Give the car a powertrain

`Powertrain` replaces the template's simple drivetrain with an engine (or a
motor) of its own speed, a torque converter or a clutch, a gearbox,
differentials and compliant half-shafts ([powertrain](../../powertrain.md)).
Here a motor of 100 N m drives the plain sedan's rear wheels through a
ratio of 8:

<!-- example: vehicle-powertrain -->
```cpp
PowertrainParams p;
p.coupling = Coupling::Direct;                        // a motor on the gearbox's input
p.engine = EngineMap::motor(100.0, 1e6, 1e6, 0.02);   // 100 N m at any speed here, 0.02 kg m^2
p.gear_ratios = {1.0};
p.final_drive = 8.0;
p.automatic = false;
Powertrain powertrain(p);   // rear-wheel drive, an open differential, compliant half-shafts
powertrain.connect(sim, car);
powertrain.throttle = 1.0;
sim.run(2.5, 5e-4);
```

The car accelerates at the axle's 0.92 x 8 x 100 N m over the wheels'
radius, divided by the car's mass and the rotating parts' inertia brought to
the road: the motor and the gearbox's input, 0.07 kg m^2 turning 8 times
faster than the wheels, weigh as 0.07 x 8^2 / R^2 = 46 kg more car. That is
1.3923 m/s^2; the car gives 1.3921 between 2.5 and 3 s. To use
the brakes with it, give them the drive's torque, so that a wheel held
against the motor is released only when the motor overcomes the brake:
`brakes.drive_torque = [&](int c) { return powertrain.shaft_torque[c]; };`.

## Give the car aerodynamics from a map

An `AeroMap` holds the force and moment over the dynamic pressure (A in
m^2, B in m^3), in the chassis's axes, over the front and rear ride heights,
pitch, roll and yaw; a variable with one node is one they do not depend on
([aerodynamics](../../aerodynamics.md)). Here a map over yaw alone:

<!-- example: vehicle-aero-map -->
```cpp
auto map = std::make_shared<AeroMap>();
map->nodes[4] = {-0.2, 0.0, 0.2};   // yaw [rad]; the ride heights, pitch and roll keep one node
Vec6 straight, yawed;               // A_x, A_y, A_z [m^2] and B_x, B_y, B_z [m^3], chassis axes
straight << -0.70, 0.0, -1.20, 0.0, -0.30, 0.0;
yawed << -0.75, 0.60, -1.00, 0.10, -0.25, 0.20;
Vec6 mirrored = yawed;
mirrored(1) = -yawed(1);
mirrored(3) = -yawed(3);
mirrored(5) = -yawed(5);
map->values = {mirrored, straight, yawed};
t.chassis.aero_map = map;
System sys;
VehicleHandle car = build_vehicle(sys, t);
AerodynamicForce* aero = car.install_aerodynamics(sys);
```

A drag area is a negative A_x, a downforce area a negative A_z. At 30 m/s
with the air 0.1 rad from the left, halfway between two nodes, the force is
q = 1.225 x 30^2 / 2 = 551 Pa times the mean of their A's. Wind is a
function of position, `aero->params.wind`; ride heights are measured above
the car's road when `aero->params.road` is set.
