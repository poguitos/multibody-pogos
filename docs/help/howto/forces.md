# How to add springs, dampers, stops and joint friction

Force elements act between two bodies (or a body and the ground, body 0) and
go in `System::force_elements`; forces on one joint's coordinate go in
`System::joint_forces`. Each element's characteristic is a `Curve`: a straight
line (`Curve::linear`) or a table interpolated by a monotone cubic
(`Curve::table`), which never overshoots between its points.

## A spring-damper with a tabulated curve and a bump stop

<!-- example: tests/examples/ex_forces.cpp#spring -->

```cpp
System sys;
const int slider = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                      RigidBodyInertia::from_solid_box(10.0, Vec3(0.1, 0.1, 0.1)), "slider");
SpringDamperParams p;
p.free_length = 0.3;                                                  // [m]
p.spring = Curve::table({0.0, 0.02, 0.05, 0.1}, {0.0, 400.0, 1200.0, 3000.0});   // force [N] against compression [m]
p.damper = Curve::linear(150.0);                                      // [N per m/s] of extension rate
p.bump_clearance = 0.08;                                              // the stop engages beyond 8 cm of compression
p.bump_stop = Curve::linear(5e4);
auto spring = std::make_shared<SpringDamper>(0, slider, Vec3::Zero(), Vec3::Zero(), p);
sys.force_elements.push_back(spring);

Simulator sim(sys);
sim.q(0) = 0.3;                                    // the slider's height: the spring at its free length
const StaticsReport rest = static_equilibrium(sim);
const LawValue f = spring->law(sim.q(0), 0.0);     // force, d/dlength, d/drate
```

The spring pushes the two points apart against its compression
(`free_length - length`); the damper opposes the rate of extension; the bump
stop adds its own curve beyond `bump_clearance` of compression (a rebound
stop works the same way on the other side). `law(length, rate)` gives the
force with its two derivatives. Statics found where the spring carries the
slider's weight.

## A torsion spring with friction and limits on a hinge

<!-- example: tests/examples/ex_forces.cpp#joint -->

```cpp
JointCoordinateForceParams h;
h.spring = Curve::linear(20.0);       // [N m per rad] about the reference
h.reference = 0.0;
h.friction = 0.5;                     // [N m], regularised over friction_velocity (1e-3 rad/s)
h.lower_limit = -1.0;                 // [rad]: beyond the limits a stop pushes back
h.upper_limit = 1.0;
h.limit_stiffness = 1e4;
sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, hinge, h));
```

`JointCoordinateForce` acts on the coordinate of a revolute or prismatic
joint directly. Friction is regularised: it builds up as
`tanh(rate / friction_velocity)`, so a joint that should stick creeps very
slowly instead, and the time step must resolve that speed. The limit stops
push back beyond the limits and never pull.

Other elements: `RotationalSpringDamper` (twist between two bodies about an
axis), `Bushing` (six axes, each with its curves), `UserForce` (any function
of the body states), `PlaneContact` ([contact](contact.md)). The force
library is described in [the kernel](../../kernel.md#forces-and-simulation).
