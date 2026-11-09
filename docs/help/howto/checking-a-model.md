# How to check a model

Find which constraints contradict each other, see where a mode's energy is,
and settle a model whose parts differ greatly in stiffness (Phase 5, tasks
5.1, 5.2 and 5.4). Every example is compiled and run, its result checked, in
`tests/howto/test_howto_checking.cpp`, and this page quotes that code exactly
(task H.6). Before any of these, `validate(sys)` reports what it can find in
a model on its own ([troubleshooting](../troubleshooting.md#models)).

The examples use `using namespace mbd; using namespace mbd::kernel;`.

## Find the constraints that contradict each other

A mechanism that will not close may be far from closing, or asked for the
impossible. The example's bob is held 1 m from two anchors 3 m apart: no
point is.

<!-- example: conflicts -->
```cpp
const AssemblyReport report = assemble(sim);
if (!report.positions_assembled) {
    // report.conflicting_equations: the rows (the constraints' equations, stacked in
    // order) that cannot all hold together; the MBD-K062 warning names their constraints.
    for (const std::string& w : report.warnings) std::cout << w << "\n";
}
```

The report names the two distances (rows 0 and 1) and nothing else: the
smallest set of equations that cannot hold together where the assembly
stopped. When the set is empty, the equations could all hold and the
iterations ran out instead: the positions given were too far from closing.
A projection during a run that cannot converge says the same in MBD-K051.

## See where a mode's energy is

Two blocks of 1 and 4 kg between walls on three equal springs. Each mode's
`energy_fraction` says how its kinetic energy is shared among the velocity
coordinates:

<!-- example: energy-shares -->
```cpp
const LinearModel lin = linearize(sim);
for (const Mode& mode : lin.modes) {
    // mode.natural_frequency [rad/s]; mode.energy_fraction(i): the share of the mode's
    // kinetic energy in velocity coordinate i, adding up to 1 over the coordinates.
    std::cout << mode.natural_frequency << " rad/s: " << mode.energy_fraction.transpose() << "\n";
}
```

In the slow mode (5.90 rad/s) the heavy block carries 92 % of the energy:
the mode is the heavy block's. The shares use the whole mass matrix, so a
coordinate coupled to others through it is counted with them. A mode with
most of its energy in one part is the place to look when that part's
frequency is wrong, and where a mount or a bushing changes it most.

## Settle a model whose parts differ greatly in stiffness

A block on a spring of 1e8 N/m carrying another on a spring of 100 N/m: a
millionfold apart, as a chassis on its tyres is from bushings of 2e7 N/m.

<!-- example: stiff-statics -->
```cpp
StaticOptions options;
options.max_translation = 1.0;               // let one step move the soft spring 9.8 cm
options.relaxation_step = 1e-4;              // below 2 / w for the stiffest part, should Newton stall
const StaticReport report = static_equilibrium(sim, options);
// report.supported: the directions with stiffness (here 2), down to 1e-7 of the stiffest;
// report.relaxation_undone: relaxation runs that grew instead of settling, and were undone.
```

- Statics counts a direction as having stiffness when it has at least 1e-7
  of the stiffest's (D47). Softer than that, it is neutral, left where it
  was (MBD-K072).
- Dynamic relaxation, the fallback when Newton cannot go on (a wheel above
  the ground), takes explicit steps: beside a stiff element it must step
  below 2 / w for that element's natural frequency w (`linearize()` gives
  them), or the motion grows. A run that ends worse than it began is undone
  and reported in MBD-K073.
- `max_translation` and `max_rotation` (0.02 m and 10 degrees by default)
  cap each Newton step, as ADAMS's TLIMIT and ALIMIT do; raise them when
  the equilibrium is far and the model is close to linear.
