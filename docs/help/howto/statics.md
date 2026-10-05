# How to find a resting state

<!-- example: tests/examples/ex_phase3.cpp#statics -->

```cpp
auto tmpl = VehicleTemplate::DefaultSedan();
tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
System sys;
const VehicleHandle car = build_vehicle(sys, tmpl);
Simulator sim(sys);
set_vehicle_equilibrium(sim, car);     // an estimate of the resting state
sim.initialize();
const StaticsReport rest = static_equilibrium(sim);
out << rest.summary(sys.model);        // iterations, accelerations, free directions
```

`static_equilibrium` finds where nothing accelerates at rest, by Newton's
method on the constraint surface, falling back on dynamic relaxation when
Newton has nothing to work with (a body above the ground it will rest on).
Start near the resting state: Newton finds the nearest equilibrium, and the
report says if it is unstable (MBD-K072). Directions in which nothing pushes
back, such as a car's position and heading on a flat road, are left as given
and reported (MBD-K073). `StaticsOptions::hold` keeps chosen coordinates
fixed.

A simulation is not a way to settle a car: with free-rolling tyres nothing
stops it rolling (see [troubleshooting](../troubleshooting.md#3-starting-states)).

More: [the kernel, static equilibrium](../../kernel.md#static-equilibrium).
