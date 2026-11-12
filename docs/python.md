# The Python interface

The engine from Python: `import multibody as mbd` (plan Phase 6, task 6.1,
decision D52). This page says how the interface is built, the conventions it
follows, what is bound and what is not yet, and what to know about Python
code that the engine calls back.

## Installing it

The interface is the package `multibody`. Its Python files are in
`python/multibody/` and its compiled module `multibody._mbd` is built from
`python/src/`, one file per part of the engine. pip compiles the module and
installs the package from the repository's folder (task 6.2, decision D53).
How pip is started depends on the machine: on the author's laptop through
the build script, as below; on other machines directly (further down).

### On the author's laptop, step by step

The laptop compiles with one compiler at a time: two builds that used every
core once restarted it (D9). pip's own build on Windows uses Visual Studio's
generator, which ignores that limit. So on the laptop the package is
installed through `scripts\build.ps1`, which builds with Ninja under the
limit. **Do not type `pip install -e .` there.**

The laptop already has what this needs: Miniforge in
`C:\Users\<you>\miniforge3`, Visual Studio with its C++ tools, CMake and
git. The first install also needs the network, to fetch Eigen, pybind11 and
spdlog.

1. **Open a "miniforge3 Prompt"** from the Start menu. It is a Command
   Prompt (`cmd`) with conda ready; its lines start with `(base)`.

2. **Activate the environment to install into:**

   ```bat
   conda activate mbd-try
   ```

   The lines now start with `(mbd-try)`. Install into `mbd-try` or another
   environment of your own, never into:
   - `base`, conda's own environment: conda itself runs from it;
   - `mbd-python`, the suite's: an installed package is imported before the
     build's own, so the suite would test the wrong one
     (`python/tests/conftest.py` then stops the suite's Python tests).

   An environment like `mbd-try` is made with

   ```bat
   conda create -n mbd-try -c conda-forge python=3.14 numpy matplotlib ipykernel pytest pandas pyarrow rerun-sdk "libblas=*=*openblas"
   ```

   under any name. The last item matters on this laptop: it gives NumPy the
   OpenBLAS library, where conda would otherwise choose Intel's MKL, whose
   build stops Python on its first linear algebra here (Windows error
   0xc06d007f). `mbd-try` itself was made without it on 10 November and
   switched to OpenBLAS on 12 November.

3. **Go to the repository's folder:**

   ```bat
   cd /d C:\dev\multibody
   ```

   (`/d` changes the drive as well, in case the prompt is on another one.)

4. **Start the install:**

   ```bat
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1 -Install -Python "%CONDA_PREFIX%\python.exe"
   ```

   Part by part:
   - `powershell ... -File scripts\build.ps1`: the script is a PowerShell
     script and the prompt is `cmd`, so PowerShell is started to run it.
   - `-ExecutionPolicy Bypass`: Windows refuses to run scripts under its
     default policy (on this laptop: "la ejecución de scripts está
     deshabilitada en este sistema"). This allows it for this one run and
     changes no setting.
   - `-NoProfile`: leaves out your PowerShell profile, so that the build
     starts from the same environment every time.
   - `-Install -Python "%CONDA_PREFIX%\python.exe"`: install into the active
     environment's Python. `%CONDA_PREFIX%` is that environment's folder,
     here `C:\Users\<you>\miniforge3\envs\mbd-try`.

   The script first prints the interpreter it installs into. It then loads
   Visual Studio's compiler and runs `pip install -e` with Ninja, one
   compiler at a time, in its own folder `build-python\ninja-<wheel tag>\`
   (for Python 3.14 the tag is `cp314-cp314-win_amd64`). pip prints its
   whole log, a hundred lines or more. The first install takes 4 to 11
   minutes: CMake downloads the dependencies (under a minute) and then
   compiles the whole library and the module, 59 steps one after another.
   Two first installs on 10 November 2026 took 3 min 41 s and 11 min, the
   second with the laptop on battery. The last line says

   ```
   Successfully installed multibody-0.6.0
   ```

   with NumPy added if the environment lacked it. The prompt then shows
   `(mbd-try)` again.

5. **Check it:**

   ```bat
   python -c "import multibody as mbd; print(mbd.__version__, mbd.__file__)"
   python -m pytest python/tests -q
   ```

   The first prints the version and where the package is read from, the
   repository itself:

   ```
   0.6.0 C:\dev\multibody\python\multibody\__init__.py
   ```

   The second runs the package's tests and ends with `43 passed in 0.37s`
   (the count grows with the package, the time varies).

**Using it in VS Code:** choose the environment as the interpreter
(Ctrl+Shift+P, "Python: Select Interpreter", then `mbd-try`; if it is not
listed, "Enter interpreter path..." and
`C:\Users\<you>\miniforge3\envs\mbd-try\python.exe`). A script then runs
with `import multibody as mbd`. In a notebook choose `mbd-try` as the
kernel.

**After a change:**
- to the Python files in `python/multibody/`: nothing to do. `-e`
  (editable) leaves them where they are, and a change shows at the next
  import (in a notebook, after restarting its kernel);
- to the C++ (`include/`, `src/`, `python/src/`): steps 1 to 4 again. Only
  what changed compiles. A second install with nothing changed took 16 s
  (37 s on battery); one after a change to the eight binding files, 161 s.
  Until then the Python files may name something the old compiled module
  lacks, and `import multibody` stops with MBD-P011, which says so.

**From PowerShell** instead (VS Code's terminal, for one), the activation is
not needed; give the interpreter's full path:

```powershell
cd C:\dev\multibody
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1 -Install -Python $env:USERPROFILE\miniforge3\envs\mbd-try\python.exe
```

**If it fails:**
- *"la ejecución de scripts está deshabilitada en este sistema"* ("running
  scripts is disabled on this system"): the script was started without
  `powershell -ExecutionPolicy Bypass -File`, for instance by typing
  `scripts\build.ps1`. Use step 4's line.
- *"\"conda\" no se reconoce como un comando interno o externo"* ("'conda'
  is not recognized as an internal or external command"): the prompt is a
  plain Command Prompt, not a miniforge3 Prompt. Open one (step 1).
- *"El argumento 'scripts\build.ps1' para el parámetro -File no existe"*
  ("the argument 'scripts\build.ps1' to the -File parameter does not
  exist"): the prompt is not in the repository's folder. Step 3.
- *"No Python interpreter at '%CONDA_PREFIX%\python.exe'"*, with
  `%CONDA_PREFIX%` written out: no conda environment is active. Steps 1 and 2.
- *"CMake configuration failed"*, after a CMake error that points at
  `FetchContent_MakeAvailable` in `CMakeLists.txt`: the build folder was
  configured with another generator. CMake says so a few lines above:

  ```
  CMake Error: Error: generator : Visual Studio 18 2026
  Does not match the generator used previously: Ninja
  ```

  A CMake folder serves one generator, and the dependencies' sub-builds keep
  the first. This happened when plain pip was run in the folder the script
  had used; since 10 November 2026 they use different folders. Delete the
  folder under `build-python\` (only build output is there) and start again;
  the next install takes as long as a first one.
- `docs/help/troubleshooting.md` has these and more under "Builds".

### On other machines: pip

From the repository's folder:

```bash
pip install -e .              # or pip install -e ".[test]" to add pytest
```

A machine needs:
- Python 3.11 or later; NumPy 2.0 or later is installed with the package;
- a C++20 compiler: on Windows, Visual Studio 2022 or later with its C++
  tools, which pip finds without a developer prompt; on Linux, GCC;
- git and the network for the first install, since CMake fetches the
  dependencies (Eigen, pybind11, spdlog) when it configures.

CMake 3.25 or later and, on Linux, Ninja are taken from PyPI when the machine
has none. The build is Release. With Ninja (Linux) it runs **one compiler at
a time**, the build system's limit; on a machine with memory to spare,
`pip install -e . -C cmake.define.MBD_COMPILE_JOBS=4` runs four (see "Build
limits" in the README). On Windows pip uses Visual Studio's generator, which
ignores the limit.

The build is kept in `build-python/<wheel tag>/`, so an install after a
change to the C++ compiles only what changed. A change to the Python files
needs no install, as above. Install into an environment other than the
suite's, for the reason given in step 2.

A wheel: `pip wheel --no-deps -w dist .`. CI builds one for Linux and one
for Windows at every push and keeps them 14 days. A wheel suits machines like
the one that built it; the package is not published.

## Building it for development

The suite builds the module in the project's own build folder. A build lays
the package out in `<build>/python/multibody/`, so with
`PYTHONPATH=<build>/python` the package imports.

It is built for one interpreter, CMake's `Python_EXECUTABLE`, which needs
its development files. On Windows the build script passes it:

```powershell
mamba env create -f python/environment.yml      # once: Python 3.12.16, NumPy 2.5.4, pytest 9.2.0
scripts\build.ps1 -Python C:\Users\<you>\miniforge3\envs\mbd-python\python.exe -Test
```

Under Windows' default execution policy start the script as in step 4 above:
`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1`,
then the same arguments.

The choice is kept in the build folder's cache, so it is given once.
`-Python` also drops pybind11's old upper-case `PYTHON_*` cache entries,
which would otherwise keep a previous interpreter's module suffix.
`-DMBD_BUILD_PYTHON=OFF` leaves the interface out.

The tests are `python/tests/test_*.py`, with the small models they share in
`python/tests/models.py`. Each test file is one CTest test (`Python: <file>`),
run by pytest under the same interpreter. When that interpreter lacks NumPy
or pytest, CMake leaves the tests out and says so. CI's Windows job uses
Python 3.11; the laptop's environment uses 3.12. The same tests run against
an installed package with `python -m pytest python/tests` from the
repository's folder; CI's package job does so with Python 3.11 and NumPy 2.0
on Linux and Python 3.14 with the newest NumPy on Windows.

**Parity with the C++ tests** (task 6.3, D55). `test_parity.py` reproduces
ten C++ test cases through the package and requires their numbers to be
identical, bit for bit: 516 numbers from assembly, statics, linearisation,
events, a state element and a force callback written in Python, constraints,
loads, contact with friction, nonlinear springs, bushings and five
integrators (RK4, semi-implicit Euler, Dormand-Prince, generalised-alpha,
Radau IIA). Each test runs its C++ case with `MBD_PARITY_FILE` set; the case
writes what it computed through `tests/support/parity.hpp` (each number in
hexadecimal, so that it reads back exactly), and the test records the same
names from Python and compares (`python/tests/parity.py`). The C++ cases
check their numbers against closed forms and independent methods; identical
numbers carry those checks over to the package. The setups are copied
expression for expression: an input one rounding apart changes nearly
every number after it. The tests need the C++ test executables of the same
build (`MBD_TEST_BIN_DIR`, set by CTest), so they run in the suite, on the
laptop and in CI's Windows job; against an installed package they are
skipped.

## Conventions

**Names.** Classes, functions and members keep their C++ names
(`RevoluteJointModel`, `add_body`, `step_control.relative_tolerance`), so
[the kernel's documentation](kernel.md) and [the help](help/README.md) apply
as written. Python adds only what it needs and C++ lacks:
- `Transform3.from_axis_angle`;
- `Simulator.run_recorded`, which runs as `run()` and returns the states as
  arrays without a Python call per step;
- `Recorder.as_dict`;
- `kinetic_energy(sim)` and `potential_energy(sim)`;
- the shorthands below.

One rename: a report's multipliers are `loads.lambda_`, because `lambda` is a
Python keyword.

**Shorthands** (tuned with the author, D54):
- A body's joint frames default to the identity, so
  `model.add_body(0, hinge, inertia=I, name="arm")` is enough for a joint at
  the body's origin.
- Three numbers stand for a translation wherever a `Transform3` is expected:
  `x_pj=[0.3, 0.0, 0.0]`, `mbd.Marker(body, [0.0, 0.1, 0.0])`,
  `mbd.Transform3([x, y, z])`. Anything else (two numbers, a matrix) is
  refused as before.
- Bodies by name: `add_body("crank", ...)` takes the parent by its name
  (`"ground"` for body 0), and `model.body("crank")` gives an index.
  MBD-P008 says no body has the name, MBD-P009 that several share it.
- The settings classes take their settings by keyword:
  `mbd.StaticOptions(max_iterations=50)`,
  `mbd.SpringDamperParams(free_length=0.3, spring=mbd.Curve.linear(3e4))`.
  The rest keep their defaults, each keyword goes through the setting's own
  checks, and a name that is not a setting raises MBD-P010 (a misspelling
  would otherwise leave the default in place). They are `AssemblyOptions`,
  `StaticOptions`, `LinearizationOptions`, `StepControl`, `ImplicitControl`
  and the parameters of the force elements (`SpringDamperParams`,
  `RotationalSpringDamperParams`, `BushingParams`, `ContactParams`,
  `JointCoordinateForceParams`). Each prints as the call that builds it.
- Every object prints what it holds: a record as its fields
  (`IntegratorStats(steps=500, rejected=0, evaluations=2000, ...)`), a
  report or a live object as a line of what matters
  (`<StaticReport: converged in 18 iterations, residual 5.61e-11 (given
  69.8); 1 degree of freedom; 0 warnings>`, `<Simulator: t = 0.5 s, RK4; 3
  coordinates, 3 velocities, 0 states>`). A report's
  `summary(model)` still gives the whole text. A curve prints as the call
  that makes it, every number in full (`Curve.table([0.0, 0.01], [0.0,
  2000.0])`, `Curve.linear(150.0)`, `Curve()`), and its points are readable
  (`table_x()`, `table_y()`, `line_slope()`, `line_offset()`; task 6.4).

These conveniences live in `python/multibody/_display.py` and
`_settings.py`, and in the bindings for frames and names. Two tests keep
them general: every class prints more than its address, and every function
has a docstring (`python/tests/test_shorthands.py`).

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
| The simulator | `System`, `Simulator`, `Integrator`, `StepControl`, `ImplicitControl`, events, the force callback, state elements, the trace, the last projection onto the constraints (`last_projection()`, a `ProjectionInfo`) |
| Analyses | `assemble`, `static_equilibrium`, `linearize`, `Kinematics`, `point_height_driver`, `compute_loads`, the energies |
| Checks and output | `validate`, `describe`, `dump_state`, `Recorder`, `add_drift_channels` |
| Files | `save_model`, `load_model`, `model_to_json`, `model_from_json` (`docs/model-files.md`); `dataframe`, `write_results`, `read_results`, `coordinate_names` (results as tables) |
| Viewing | `replay`, `plot` (Rerun) |

**Not bound yet:**
- The vehicle layer, with its tyre, aerodynamic and anti-roll-bar elements:
  they are still on the old Y-up axes, and Phases 7 and 8 rebuild them.
  They will be bound then.
- The lap tools.
- The kernel's `Data` and its algorithms (forward kinematics, inverse
  dynamics, the mass matrix, Jacobians).
- The constraint solver itself (its multipliers and Jacobian; the
  constraint forces are in `compute_loads`).
- Joint forces written in Python: `JointForce` cannot be subclassed; a force
  on joint coordinates goes through `JointCoordinateForce` or the force
  callback.

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

## Model files and results as tables

(Task 6.4, decision D56.)

**A model to a file and back:**

```python
mbd.save_model(system, "four_bar.json")       # a str or a pathlib.Path
system = mbd.load_model("four_bar.json")
text = mbd.model_to_json(system)              # and mbd.model_from_json(text)
```

The file is JSON, described in `docs/model-files.md`; the engine reads and
writes it in C++, so a C++ program reads the same files. A model read back
writes the same text again and runs bit for bit as the original. What a file
cannot hold (a `UserForce`, a time function given as code) is refused by
name, MBD-I008, and a misspelt key in a file is an error, MBD-I004.

**Results as tables** (pandas; pyarrow for Parquet; `pip install -e
".[results]"`, or `conda install pandas pyarrow`; the rest of the package
needs neither):

```python
run = sim.run_recorded(2.0, 1e-3)
df = mbd.dataframe(run, sim)                  # indexed by time
df["q"]["crank"]                              # a coordinate, by its body's name
df[("v", "crank")]                            # the same for its velocity
mbd.write_results(df, "run.parquet")          # or "run.csv"
df = mbd.read_results("run.parquet")
channels = mbd.dataframe(recorder)            # a Recorder: one column per channel
```

A run's columns have two levels, the vector (`q`, `v`, and `x` for the states
of state elements) and the coordinate. A coordinate is named after its body;
a joint with several coordinates numbers them (`block[0]` to `block[6]` for a
free joint: x, y, z, then the quaternion's x, y, z, w; its velocities
`block[0]` to `block[5]`); a body without a name is `body3`, and a name
several bodies share is followed by its index (`twin#4`). A state element's
states are numbered `0`, `1`, ... `mbd.coordinate_names(model, "q")` gives
the names alone.

Both formats keep every number exactly: Parquet in binary, CSV with each
number in full and read back with pandas's exact parser. `read_results` also
reads a `Recorder`'s own CSV (`recorder.write_csv`).

## Viewing: playback and plots

(Task 6.5, decision D57.) A motion played back in three dimensions with its
channels plotted beside it, in Rerun (rerun.io; `conda install rerun-sdk`,
or the package's extra `pip install -e ".[view]"`; the rest of the package
does not need it):

```python
run = sim.run_recorded(3.0, 1e-3)
mbd.replay(system, run)                         # opens the Rerun viewer
mbd.replay(system, run, path="four_bar.rrd")    # or saves the recording; open it with: rerun four_bar.rrd
mbd.replay(system, run, channels=mbd.dataframe(recorder))   # with a recorder's channels plotted too
mbd.plot(mbd.dataframe(recorder))               # channels alone
```

The view is on the left and the plots on the right, on one time line: drag
the time, play it, or pick a moment in a plot. The motion may be a run
(`run_recorded`), a table (`dataframe`, `read_results`, or a CSV with columns
`q[0]`, `q[1]`, ... as a C++ program writes it) or `(time, q)`. `fps=` sets
the frames drawn per second of motion (60 by default; the plots take every
sample).

The model has no shapes of its own, so the view draws what it has: each
body as the uniform box with its mass and moments of inertia (a point for a
point mass), lines from its centre of mass to its joints, springs, dampers,
bushings and distances as lines between their points, contact spheres and
the markers of closures as points. Up is against gravity. Forces written in
Python (`UserForce`) have no points and are not drawn.

`python/examples/` has two: `replay_four_bar.py`, the four-bar on a spring;
and `replay_hmmwv.py`, the HMMWV of the benchmark over its bump, whose run is
recorded in C++ (`mbd_hmmwv record <folder>`, `docs/hmmwv.md`) and played
back from its model file and its CSV.

## A pendulum

```python
import math
import numpy as np
import multibody as mbd

system = mbd.System()
system.model.gravity = [0.0, -9.81, 0.0]
system.model.add_body(0, mbd.RevoluteJointModel(), x_cj=[0.0, 0.8, 0.0],   # the hinge 0.8 m above
                      inertia=mbd.RigidBodyInertia(1.0, np.zeros(3), np.zeros((3, 3))), name="bob")
sim = mbd.Simulator(system)
sim.method = mbd.Integrator.DormandPrince
sim.q[0] = 1.0                              # released 1 rad from hanging
sim.initialize()
result = sim.run_recorded(2.0, 0.01)        # 'time', 'q', 'v', 'x' as arrays
```

`python/tests/test_pendulum.py` checks this pendulum, at tolerances of
1e-12, against its period from the elliptic integral, to 3e-11 rad.
