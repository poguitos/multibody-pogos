# How to add forces and drives

Springs, dampers and stops between two points; a spring, damper, friction
or limit on a joint; a joint driven along a prescribed motion (Phase 3,
tasks 3.1 and 3.2). Every example is compiled and run, its result checked,
in `tests/howto/test_howto_forces.cpp`, and this page quotes that code
exactly (task H.6). The laws and their derivatives are described in
[the kernel](../../kernel.md#forces-and-simulation).

The examples use `using namespace mbd; using namespace mbd::kernel;`. A
`System sys` holds the model: its bodies and joints (`sys.model`), its
force elements, joint forces and constraints, all added before a
`Simulator` is built on it.

## A spring-damper between two points

A 10 kg block on a vertical slider (gravity along -Z, the slider's
coordinate its height), hung from a point 1 m up on the ground:

<!-- example: spring-damper -->
```cpp
const int block = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                     RigidBodyInertia::from_solid_box(10.0, Vec3(0.1, 0.1, 0.1)), "block");
SpringDamperParams p;
p.free_length = 0.8;                         // [m]
p.spring = Curve::linear(1e4);               // force pushing apart against the compression [N/m]
p.damper = Curve::linear(632.0);             // force against the rate of extension [N s/m]
p.rebound_clearance = 0.05;                  // a rebound stop 5 cm beyond the free length
p.rebound_stop = Curve::linear(1e6);
sys.force_elements.push_back(
    std::make_shared<SpringDamper>(0, block, Vec3(0.0, 0.0, 1.0), Vec3::Zero(), p));   // ground point, block point
```

Released at the free length, the block settles where k (L - L0) = m g,
9.81 mm lower (critically damped, here, within a second).

- **Curves:** `Curve::linear(slope)` or `Curve::table(x, y)` for a measured
  characteristic, interpolated, with its slope (the implicit integrators
  use it). A damper's table may differ in bump and rebound.
- **Preload, stops:** `p.preload` pushes apart at every length (an installed
  preload); `bump_clearance` and `bump_stop` engage when the compression
  exceeds the clearance, `rebound_clearance` and `rebound_stop` when the
  extension does.
- The points are in each body's frame; body 0 is the ground.
  `potential_energy(L)` gives the energy stored, for checks.

## A spring, damper, friction or limit on a joint

A torsion spring and damper on a hinge (a rod pointing along X on a hinge
along Z, held against gravity):

<!-- example: torsion-spring -->
```cpp
JointCoordinateForceParams hinge;
hinge.reference = 0.2;                       // where the spring carries no torque [rad]
hinge.spring = Curve::linear(50.0);          // [N m/rad]
hinge.damper = Curve::linear(8.0);           // [N m s/rad]
sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, rod, hinge));
```

The rod settles where the spring's torque, 50 (0.2 - q), balances gravity's,
m g d cos q.

- It acts on a revolute or prismatic joint's coordinate, given by the
  joint's child body (`rod` above).
- `preload` adds a constant torque or force; `friction` a Coulomb friction,
  regularised over `friction_velocity` (a slow creep replaces true
  sticking); `lower_limit`, `upper_limit`, `limit_stiffness` and
  `limit_damping` a stop that pushes back beyond the limits and never pulls.

**Other connectors.** `Bushing` (`mbd/forces/bushing.hpp`): six springs and
dampers between two markers, about and along each axis, the rotation
measured about an axis halfway between them. `RotationalSpringDamper`: a
torsion spring and damper about an axis between two bodies that are not
parent and child (a torsion bar). `UserForce` (`mbd/forces/user_force.hpp`): a
force of your own, a function that reads the bodies' world states and adds
world forces at their origins (without derivatives; statics and
linearisation take them by differences); a force that depends on time or
acts on joint coordinates goes in `Simulator::force_callback` instead.
Worked examples: `tests/forces/test_force_connectors.cpp`.

## Drive a joint along a prescribed motion

The hanging rod's hinge driven along a sine, and the torque it takes:

<!-- example: joint-driver -->
```cpp
const Real A = 0.5, w = 3.0;                 // amplitude [rad], frequency [rad/s]
sys.constraints.push_back(std::make_shared<JointDriver>(
    sys.model, rod,
    TimeFunction([=](Real t) { return A * std::sin(w * t); },              // the angle,
                 [=](Real t) { return A * w * std::cos(w * t); },          // its rate
                 [=](Real t) { return -A * w * w * std::sin(w * t); })));  // and its acceleration
Simulator sim(sys);
sim.initialize();                            // the velocity follows the drive: A w at t = 0
sim.run(1.0, 0.001);
const Loads loads = compute_loads(sim);      // at the state at 1 s
const Real torque = loads.lambda(0);         // what the drive exerts [N m]
```

The torque is what the motion needs: I q'' + m g d sin q.

- A driver is a constraint: the joint follows the motion exactly, and its
  multiplier (`loads.lambda`, in the order of `sys.constraints`'
  equations) is the force or torque that drives it.
- The rate and acceleration must be the function's derivatives: the
  velocity and acceleration levels of the constraint use them.
  `TimeFunction::constant(value)` holds the joint still.
- `initialize()` puts the velocities on the drive; to start with the joint
  elsewhere, call `assemble()` ([analyses](analyses.md)).
- Other constraints in `mbd/kernel/constraints.hpp` take a `TimeFunction`
  as their target: `Dot2`, a point's signed distance from a plane on
  another body (a point driven along a path, one coordinate each);
  `Distance`, the distance between two points; `Dot1`, the dot product of
  two axes (the cosine of the angle between them). Worked example: a point
  driven along a circle in `tests/kernel/test_kernel_simulator.cpp`.
