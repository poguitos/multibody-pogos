# The Python interface

The engine from Python: `import multibody as mbd` (plan Phase 6, task 6.1,
decision D52). This page says how the interface is built, the conventions it
follows, what is bound and what is not yet, and what to know about Python
code that the engine calls back.

## Building it

The interface is the package `multibody`. Its Python files are in
`python/multibody/` and its compiled module `multibody._mbd` is built from
`python/src/`, one file per part of the engine. A build lays the two out
together in `<build>/python/multibody/`, so with `PYTHONPATH=<build>/python`
the package imports.

It is built for one interpreter, CMake's `Python_EXECUTABLE`, which needs
its development files. On Windows the build script passes it:

```powershell
mamba env create -f python/environment.yml      # once: Python 3.12.16, NumPy 2.5.4, pytest 9.2.0
scripts\build.ps1 -Python C:\Users\<you>\miniforge3\envs\mbd-python\python.exe -Test
```

The choice is kept in the build folder's cache, so it is given once.
`-Python` also drops pybind11's old upper-case `PYTHON_*` cache entries,
which would otherwise keep a previous interpreter's module suffix.
`-DMBD_BUILD_PYTHON=OFF` leaves the interface out.

The tests are `python/tests/test_*.py`, with the small models they share in
`python/tests/models.py`. Each test file is one CTest test (`Python: <file>`),
run by pytest under the same interpreter. When that interpreter lacks NumPy
or pytest, CMake leaves the tests out and says so. CI's Windows job uses
Python 3.11; the laptop's environment uses 3.12.

## Conventions

**Names.** Classes, functions and members keep their C++ names
(`RevoluteJointModel`, `add_body`, `step_control.relative_tolerance`), so
[the kernel's documentation](kernel.md) and [the help](help/README.md) apply
as written. Python adds only what it needs and C++ lacks:
- `Transform3.from_axis_angle`;
- `Simulator.run_recorded`, which runs as `run()` and returns the states as
  arrays without a Python call per step;
- `Recorder.as_dict`;
- `kinetic_energy(sim)` and `potential_energy(sim)`.

One rename: a report's multipliers are `loads.lambda_`, because `lambda` is a
Python keyword.

**Arrays.**
- Vectors and matrices are NumPy arrays. A quaternion is `[w, x, y, z]`.
- The simulator's state (`sim.q`, `sim.v`, `sim.x`, `sim.tau`), a
  kinematic analysis's `q`, `model.gravity` and an inertia's `com_B` and
  `I_com_B` read as **views of the engine's own memory**, so
  `sim.q[0] = 0.3` writes the state. The state can also be assigned whole,
  but only at its size (MBD-P001): assigning an Eigen vector of another size
  would resize it.
- Results (reports, `states()`, `trace()`, columns) are copies.
- Rotation matrices and quaternions typed in Python are checked
  (MBD-P002 to P004).

**Lists.** A system's constraints, force elements and joint forces, a
bushing's curve triples and an assembly's held indices read as tuples.
Change them with `system.add_constraint(...)` and its kin, or by assigning
a whole sequence. A list would be a copy, and an append to it would be lost
without a word; a tuple refuses the append.

**Errors and warnings.**
- Every error of the engine raises `mbd.MbdError` (a `RuntimeError`) with
  the message unchanged and its code in `.code`, for example
  `"MBD-K010"`; [the catalogue](help/messages.md) explains each code.
- Warnings are Python warnings of class `mbd.MbdWarning`, also with
  `.code`, so the usual filters work:
  `warnings.simplefilter("error", mbd.MbdWarning)` makes them errors, and
  `pytest.warns(mbd.MbdWarning)` checks one.
- The interface's own codes are in area P of the catalogue.

**Lifetimes.**
- A simulator keeps its system alive, and so do a kinematic analysis and
  `attach()`'s state element. `system.model` is a reference into the
  system.
- The C++ rule stays: once a simulator exists, its system must not gain
  bodies or constraints.

## What is bound

| Part | Python |
|---|---|
| Frames, inertias, body states | `Transform3`, `RigidBodyInertia`, `RigidBodyState` |
| The tree | `Model`, `RevoluteJointModel` and the seven other joint models |
| Constraints | `Marker`, `TimeFunction`, `PointCoincidence`, `Dot1`, `Dot2`, `Distance`, `NoTwist`, `JointDriver`, the closures (`revolute_closure`, ...), `point_on_line`, `point_on_plane` |
| Force elements | `Curve`, `LinearSpringDamper`, `SpringDamper`, `RotationalSpringDamper`, `Bushing`, `PlaneContact`, `UserForce`, `JointCoordinateForce`, with their parameter structs |
| The simulator | `System`, `Simulator`, `Integrator`, `StepControl`, `ImplicitControl`, events, the force callback, state elements, the trace |
| Analyses | `assemble`, `static_equilibrium`, `linearize`, `Kinematics`, `point_height_driver`, `compute_loads`, the energies |
| Checks and output | `validate`, `describe`, `dump_state`, `Recorder`, `add_drift_channels` |

**Not bound yet:**
- The vehicle layer, with its tyre, aerodynamic and anti-roll-bar elements:
  they are still on the old Y-up axes, and Phases 7 and 8 rebuild them.
  They will be bound then.
- The lap tools.
- The kernel's `Data` and its algorithms (forward kinematics, inverse
  dynamics, the mass matrix, Jacobians).
- The constraint solver itself.

## Python code that the engine calls

User forces, prescribed functions, events, the force callback, recorder
channels and state elements can all be written in Python. The engine calls
each one at every evaluation or step, taking the interpreter lock, so they
run at Python's speed: use them for what has no C++ element, and use the C++
elements where one exists.

- **`UserForce(name, bodies, function)`.** `function(states, forces)` gets
  the bodies' world states (index 0 the ground) and a `BodyForces`:
  `forces.add_force(body, f_W, point_W)` and
  `forces.add_torque(body, tau_W)` add loads in world axes. The handle is
  valid during the call only (MBD-P005).
- **`sim.force_callback = f`.** `f(sim, t, tau)` adds generalized forces to
  `tau` in place (`tau[0] += ...`). **Read the state being evaluated from
  `sim.states()`:** `sim.q` and `sim.v` hold the start of the step, not the
  integrator's intermediate stage. A callback that read `sim.q` produced a
  constant force in the first draft of its test.
- **`TimeFunction(s, ds, dds)`.** Three callables of t that must agree: a
  driver is only as consistent as they are.
- **`sim.add_event(name, function, action, direction)`.**
  `function(sim)` is watched for zero crossings. `action(sim)` runs at a
  crossing and may change `sim.q` and `sim.v`.
- **`StateElement`.** Subclass it, call `super().__init__()`, and define
  `name()`, `size()`, `initial_states(x)` and `rates(sim, t, x, x_dot)`;
  `begin_evaluation` and `end_step` are optional (MBD-P007 names a missing
  one). `x_dot`, and the `x` of `initial_states` and `end_step`, are
  written in place. Then call `sim.attach(element)`.
- **A Python exception** raised in any of them reaches the caller of
  `step()` or `run()` unchanged.

## A pendulum

```python
import math
import numpy as np
import multibody as mbd

system = mbd.System()
system.model.gravity = [0.0, -9.81, 0.0]
system.model.add_body(0, mbd.RevoluteJointModel(), mbd.Transform3(),
                      mbd.Transform3.from_translation([0.0, 0.8, 0.0]),
                      mbd.RigidBodyInertia(1.0, np.zeros(3), np.zeros((3, 3))), "bob")
sim = mbd.Simulator(system)
sim.method = mbd.Integrator.DormandPrince
sim.q[0] = 1.0                              # released 1 rad from hanging
sim.initialize()
result = sim.run_recorded(2.0, 0.01)        # 'time', 'q', 'v', 'x' as arrays
```

`python/tests/test_pendulum.py` checks this pendulum, at tolerances of
1e-12, against its period from the elliptic integral, to 3e-11 rad.
