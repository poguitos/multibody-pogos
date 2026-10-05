# How to prescribe a motion and analyse a mechanism's kinematics

## Drive a joint along a function of time, and read the drive's force

<!-- example: tests/examples/ex_motion.cpp#driver -->

```cpp
// The hinge follows s(t) = 0.3 sin(2 t): the function and its first two
// derivatives.
sys.constraints.push_back(std::make_shared<JointDriver>(
    sys.model, body,
    TimeFunction([](Real t) { return 0.3 * std::sin(2.0 * t); },
                 [](Real t) { return 0.6 * std::cos(2.0 * t); },
                 [](Real t) { return -1.2 * std::sin(2.0 * t); })));
Simulator sim(sys);
sim.initialize();                                  // velocities consistent with the drive
sim.run(1.0, 1e-3);
sim.acceleration(sim.q, sim.v, sim.time);          // evaluate at the state reached
const Real torque = sim.solver().lambda()(0);      // the drive's torque [N m]
```

A `JointDriver` is a constraint, `q - s(t) = 0`, on a revolute or prismatic
joint; its multiplier is the force or torque the drive applies. Give `s` with
its first two derivatives, exactly: the velocity and acceleration equations
use them (a wrong derivative shows as a drift the projection keeps
correcting). To drive a point along a path, use one `Dot2` constraint per
coordinate, each with its own `TimeFunction`.

## A mechanism's motion ratios

<!-- example: tests/examples/ex_motion.cpp#kinematics -->

```cpp
// A four-bar (crank 0.3 m, coupler 0.8, rocker 0.6, ground 0.7), its
// crank driven to the angle t: in a kinematic analysis t is the driving
// parameter, not a time.
System sys;
const auto hinge = std::make_shared<RevoluteJointModel>();
const RigidBodyInertia link = RigidBodyInertia::from_solid_box(1.0, Vec3(0.2, 0.02, 0.02));
const int crank = sys.model.add_body(0, hinge, Transform3(), Transform3(), link, "crank");
const int coupler = sys.model.add_body(crank, hinge, Transform3::FromTranslation(Vec3(0.3, 0, 0)), Transform3(), link, "coupler");
const int rocker = sys.model.add_body(coupler, hinge, Transform3::FromTranslation(Vec3(0.8, 0, 0)), Transform3(), link, "rocker");
sys.constraints.push_back(revolute_closure(Marker{0, Transform3::FromTranslation(Vec3(0.7, 0, 0))},
                                           Marker{rocker, Transform3::FromTranslation(Vec3(0.6, 0, 0))}));
sys.constraints.push_back(std::make_shared<JointDriver>(
    sys.model, crank, TimeFunction([](Real t) { return t; }, [](Real) { return 1.0; }, [](Real) { return 0.0; })));

Kinematics k(sys);
k.q << 1.0, -0.6, -2.3;             // near the closed loop
k.t = 1.0;                          // crank at 1 rad
const bool closed = k.solve();
const VecX& ratio = k.velocities(); // d(angles)/d(crank angle)
```

`Kinematics` solves `phi(q, t) = 0` with `t` as the driving parameter: here
the crank's angle. `velocities()` solves `J v = nu`: for a mechanism with no
freedom left, the derivatives of every coordinate with respect to `t`, its
motion ratios; `accelerations()` gives the second derivatives. A sweep sets
`t` step by step and calls `solve()` from the last configuration
(`sweep_bump_travel` does this for a suspension's bump travel).
