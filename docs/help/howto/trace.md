# How to find out what went wrong in a simulation

<!-- example: tests/examples/ex_phase3.cpp#trace -->

```cpp
out << validate(sys, sim.q).summary();        // anything wrong with the model?
sim.trace = true;                             // a row per step
sim.run(1.0, 1e-3);
write_trace_csv(sim.trace_rows(), path);      // to look at, or to diagnose later
const TraceDiagnosis d = diagnose(read_trace_csv(path));
out << d.summary() << dump_state(sim);        // what went wrong, and where it ended
```

`validate` checks the model; the trace records each step (drift before the
projection, the projection, the rank, the energies, the work of the applied
forces and the energy balance), and `diagnose` says what it shows: failed
projections (MBD-K100), an energy balance that drifts, usually a step too
long (K101), energy put in by the applied forces (K102), redundant equations
(K103), drift within a step (K104). `dump_state` gives the state and each
constraint's residual. The trace file can be diagnosed later, on its own.

More: [troubleshooting](../troubleshooting.md), [the kernel, debugging
aids](../../kernel.md#debugging-aids).
