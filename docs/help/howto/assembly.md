# How to start from values that do not satisfy the constraints

<!-- example: tests/examples/ex_phase3.cpp#assemble -->

```cpp
Simulator sim(sys);
sim.q << 1.0, -0.5, -2.0;          // the crank at 1 rad, the rest roughly
sim.v << 2.0, 0.0, 0.0;            // the crank turning at 2 rad/s
AssemblySpec spec;
spec.hold(sys.model, 1);           // keep the crank's angle and rate as given
const AssemblyReport report = assemble(sim, spec);
out << report.summary(sys.model);  // residuals before and after, what moved most
```

`assemble` moves the coordinates onto the constraints, then the velocities,
keeping the held ones exactly and changing the others as little as possible
(in the kinetic-energy metric: light parts move most). Hold as many
coordinates as the mechanism has degrees of freedom, chosen so that they fix
it: the crank of a four-bar, not its crank and rocker. The report gives each
level's residual before and after, the coordinate that changed most, and the
freedom left; held values that contradict the constraints are reported
(MBD-K062 to K065) with the closest state reached. `assemble(system, q, v,
a, t, spec)` also assembles accelerations.

More: [the kernel, assembly](../../kernel.md#assembly-of-initial-conditions).
