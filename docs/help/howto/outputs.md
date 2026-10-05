# How to record forces, reactions and energies over a run

<!-- example: tests/examples/ex_outputs.cpp#record -->

```cpp
Recorder rec;
rec.add("angle", [&] { return sim.q(0); });
rec.add("hinge_force", [&] {
    const Loads loads = compute_loads(sim);        // reactions, constraint forces, accelerations
    return loads.joint_reaction[static_cast<std::size_t>(body)].tail<3>().norm();
});
rec.add("energy", [&] {
    return kinetic_energy(sys.model, sim.data()) + potential_energy(sys.model, sim.data());
});
rec.sample(sim.time);
for (int n = 0; n < 1000; ++n) {
    sim.step(1e-3);
    rec.sample(sim.time);
}
rec.write_csv(path);                               // time, then one column per channel
```

A `Recorder` channel is any function returning a number, read at each
`sample()`. `compute_loads(sim)` gives, at the simulator's state, the
accelerations, the constraint multipliers, each constraint's wrench on each
body, and each joint's reaction: the force and moment it transmits to its
child body, in the child-side joint frame (`joint_reaction[i]` for the joint
of body `i`, `[moment; force]`). Energies come from `kinetic_energy` and
`potential_energy` (gravity), and from each spring element's
`potential_energy`. `write_csv` writes 17 significant digits, so values read
back exactly.
