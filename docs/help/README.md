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
  simulator, the checks and `validate()`.

## How-to guides

Planned with each phase (task H.6). Until then, the tests are the worked
examples:

| To do this | See |
|---|---|
| Build a mechanism with loops | `tests/kernel/test_kernel_constrained.cpp` (four-bar), `tests/kernel/test_kernel_validate.cpp` |
| Simulate it | `tests/kernel/test_kernel_simulator.cpp`, `tests/kernel/test_kernel_joints_physics.cpp` |
| Drive a joint along a prescribed motion | `tests/kernel/test_kernel_simulator.cpp` (driven pendulum) |
| Check a model before simulating it | [The kernel, checks](../kernel.md#checks); `validate()` |
| Build and run a vehicle | `tests/vehicle/test_vehicle_template_dynamic.cpp`, `bench/bench_dynamics.cpp` |
| Sweep a suspension through bump travel | `tests/test_suspension_kinematics.cpp` |
| Measure performance | [Performance, task 2.10](../performance.md#task-210-the-sedan-step-5-october-2026) |

## Troubleshooting

Planned (task H.3): from symptom to cause to fix. Until then, start with the
code at the front of the message, in the catalogue below, and with
`validate(system)`, which lists every problem it finds in a model at once.

## Messages

[The message catalogue](messages.md): every error and warning the program
reports, by its code (`MBD-K001` and so on), with what it means, the usual
causes and what to do.

## Reference

- [The kernel](../kernel.md) and [conventions](../conventions.md).
- [Performance](../performance.md): timings, and how to measure them.
- The API reference generated from the headers: planned (task H.5). Until
  then, the header comments under `include/mbd/` are the reference.

## How the program came to be

The development record: the [journal](../../documentation/journal/README.md),
the [design decisions](../../documentation/decisions.md) and the
[plan](../../documentation/Master_plan.md).
