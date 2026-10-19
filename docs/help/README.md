# Help

How to use the program, what its messages mean, and how to find out what went
wrong. This folder grows with the program (plan, documentation track H): each
task that adds or changes a message, an option or a behaviour updates it.

## Getting started

- **Build and test:** [README](../../README.md#building-and-testing). One
  compiler at a time, through `scripts\build.ps1`.
- **What the program is made of:** [README, layout](../../README.md#layout).

## Concepts

- [Conventions](../conventions.md): units, frames and transforms, bodies,
  joints and their coordinates, constraints, axes, signs, naming.
- [The multibody kernel](../kernel.md): spatial vectors, joint models, Model
  and Data, the recursive algorithms, constraints, the constrained solve, the
  simulator, initial-condition assembly, static equilibrium, linearisation,
  the checks and `validate()`.

## How-to guides

Planned with each phase (task H.6). Until then, the tests are the worked
examples:

| To do this | See |
|---|---|
| Build a mechanism with loops | `tests/kernel/test_kernel_constrained.cpp` (four-bar), `tests/kernel/test_kernel_validate.cpp` |
| Simulate it | `tests/kernel/test_kernel_simulator.cpp`, `tests/kernel/test_kernel_joints_physics.cpp` |
| Simulate a stiff model (very stiff springs, bushings, contact; tyres at low speed) at long steps, or to a tight tolerance; choose an integrator | [The kernel, forces and simulation](../kernel.md#forces-and-simulation) (integrators, and the table for choosing one); `Integrator::GeneralizedAlpha`, `Integrator::RadauIIA`, `Integrator::SDIRK4`, `Integrator::LinearlyImplicitEuler` (real time), `implicit_control`; `tests/kernel/test_kernel_integrators.cpp` |
| Simulate to a tolerance instead of a fixed step, see the steps taken, read the state between steps | [The kernel, forces and simulation](../kernel.md#forces-and-simulation) (integrators); `Integrator::DormandPrince`, `step_control`, `integrator_stats()`, `interpolate()`; `tests/kernel/test_kernel_integrators.cpp` |
| Add states of your own (an actuator lag, a controller) integrated with the bodies | [The kernel, forces and simulation](../kernel.md#forces-and-simulation); `StateElement`, `sim.attach()`; `tests/kernel/test_kernel_states.cpp` |
| Stop at the moment something happens (an impact, a threshold) and act on it | [The kernel, events](../kernel.md#events); `Simulator::events`; `tests/kernel/test_kernel_events.cpp` |
| Natural frequencies, damping and mode shapes about an equilibrium | [The kernel, linearisation](../kernel.md#linearisation); `linearize(sim)`; `tests/kernel/test_kernel_linearization.cpp`, `tests/test_quarter_car.cpp` |
| Find the static equilibrium (a vehicle settled on its tyres), with its loads | [The kernel, static equilibrium](../kernel.md#static-equilibrium); `static_equilibrium(sim)`; `tests/kernel/test_kernel_statics.cpp`, `tests/vehicle/test_vehicle_statics.cpp` |
| Make initial positions and velocities satisfy the constraints, holding some coordinates | [The kernel, initial conditions](../kernel.md#initial-conditions); `assemble(sim, options)`; `tests/kernel/test_kernel_assembly.cpp`, `tests/vehicle/test_vehicle_assembly.cpp` |
| Drive a joint along a prescribed motion | `tests/kernel/test_kernel_simulator.cpp` (driven pendulum) |
| Drive a point along a path | `tests/kernel/test_kernel_simulator.cpp` (point driven along a circle) |
| Velocities and accelerations of a mechanism along its motion (motion ratios) | `Kinematics::velocities()`, `accelerations()`; `tests/test_suspension_kinematics.cpp` |
| Add a spring or damper with a tabulated curve, preload or stops | `tests/forces/test_force_library.cpp` |
| Add a torsion spring, friction or a limit stop on a joint | `tests/forces/test_force_library.cpp` (`JointCoordinateForce`) |
| Add a bushing, a rotational spring between bodies, or a force of your own | `tests/forces/test_force_connectors.cpp` |
| Make points or spheres touch a plane, with friction | [The kernel, contact](../kernel.md#forces-and-simulation); `PlaneContact`; `tests/forces/test_contact.cpp` |
| Read joint reactions, constraint forces and energies | `compute_loads(sim)`; `tests/kernel/test_kernel_outputs.cpp` |
| Record channels over a run and write them as CSV | `Recorder`; `tests/kernel/test_kernel_outputs.cpp` |
| Watch how far a closed mechanism drifts from its constraints, and keep it on them | [The kernel, constraints per integrator](../kernel.md#forces-and-simulation); `constraint_drift()`, `add_drift_channels(recorder, sim)`, `enforce_constraints`, `constraint_tolerance`; `tests/kernel/test_kernel_drift.cpp` |
| Choose how the mass matrix is factorized (along the tree, or dense), or see which one a solve used | [The kernel, constrained dynamics](../kernel.md#constrained-dynamics); `sim.solver().mass_factorization`, `tree_factorization()`; `tests/kernel/test_kernel_sparse.cpp` |
| Check a model before simulating it | [The kernel, checks](../kernel.md#checks); `validate()` |
| See a model as text: bodies, joints, constraints, forces, counts | `describe(system)` or `describe(system, q)`; `tests/kernel/test_kernel_debugging.cpp` |
| Record how every step went (projection, residuals, rank, energy, events) and find what went wrong | `Simulator::trace_steps`, `write_trace(path)`, `dump_state(sim)`; [the kernel, checks](../kernel.md#checks); `tests/kernel/test_kernel_debugging.cpp` |
| Build and run a vehicle | `tests/vehicle/test_vehicle_template_dynamic.cpp`, `bench/bench_dynamics.cpp` |
| Sweep a suspension through bump travel | `tests/test_suspension_kinematics.cpp` |
| Measure performance | [Performance, task 2.10](../performance.md#task-210-the-sedan-step-5-october-2026) |

## Troubleshooting

[The troubleshooting guide](troubleshooting.md): from symptom to cause to
fix, for builds, models, initial conditions and statics, simulations and slow
runs, with the debugging tools (`validate()`, `describe()`, the step trace,
the state dump) and how to check a new element. Every problem met so far in
the journal has its entry. For a message, start with the code at its front,
in the catalogue below.

## Messages

[The message catalogue](messages.md): every error and warning the program
reports, by its code (`MBD-K001` and so on), with what it means, the usual
causes and what to do.

## Reference

- [The kernel](../kernel.md) and [conventions](../conventions.md).
- [Performance](../performance.md): timings, and how to measure them.
- The API reference generated from the header comments: with Doxygen
  installed, `cmake --build build --target docs` writes it to
  `build/docs/html/index.html`. CI builds it on every push and fails on any
  Doxygen warning (task H.5). The header comments under `include/mbd/` are its
  source.

## How the program came to be

The development record: the [journal](../../documentation/journal/README.md),
the [design decisions](../../documentation/decisions.md) and the
[plan](../../documentation/Master_plan.md).
