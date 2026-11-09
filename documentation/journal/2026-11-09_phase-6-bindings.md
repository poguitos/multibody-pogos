# Phase 6: the Python interface (bindings)

- **Dates:** 9 November 2026
- **Plan:** 6.1; D52
- **Commits:** `668b992` (first part), `72980ee` (second part), then see
  git log
- **Written:** at the time

## Goal

Task 6.1: "Model building, data, all analyses, recorder output as NumPy
arrays, errors as Python exceptions. *Done when a pendulum is built and
simulated from Python.*" This is the start of Phase 6, whose goal is
everything the engine can do, callable from Python, extended in every
later phase.

## Starting point

`python/` held the project template's stub: a module `pycore` with two
functions of no use (`compute_mean`, `make_greeting`) from `src/core.cpp`.
It was built on pybind11 2.12.0 for whatever Python the Windows Store alias
gave (3.13), with no test. The C++ API is snake_case already, and holds its
joints, constraints and forces as `shared_ptr<const T>`. Several of its
classes keep references or C pointers: a simulator its system, a
constraint set its name.

## What was done

1. **The design (D52):**
   - pybind11 raised to 3.1.0, for Python 3.14 support, FindPython,
     `py::smart_holder` and `py::native_enum`;
   - a package `multibody` around one module `multibody._mbd`, with the C++
     names kept;
   - the state as writable NumPy views, assigned only whole and at its size;
   - errors as `MbdError` and warnings as `MbdWarning`, each with its code;
   - a new catalogue area P;
   - lists read as tuples;
   - a pinned conda environment for the laptop (`python/environment.yml`),
     with Python 3.11 in CI's Windows job.
2. **The first part (`668b992`):** frames, inertias, body states, the joint
   models, the Model, the System, the simulator and its settings, and
   `run_recorded`. The pendulum test (task 6.1's condition), and the
   interface's own behaviour: errors, views, sizes, frames and lifetimes.
   The stub was removed with its three C++ tests.
3. **The second part (`72980ee`):**
   - markers, time functions from Python callables, every constraint
     primitive, drivers and closures;
   - curves and the general force elements, with `UserForce` taking a
     Python function and a `BodyForces` handle;
   - joint forces;
   - assembly, statics, linearisation, kinematic analysis, loads, energies,
     the checks and descriptions, the recorder, events and the force
     callback.
4. **The third part:** state elements written in Python (a trampoline),
   `attach`, and the step trace. Also `docs/python.md`, the help index, the
   troubleshooting entries, and a CI step that counts the Python tests in an
   annotation.

## Decisions

- D52, recorded when it was taken, with its scope corrected in the second
  part.
- Not bound now: the vehicle layer, whose tyre, aerodynamic and
  anti-roll-bar elements are still on the old Y-up axes and are rebuilt in
  Phases 7 and 8; the lap tools; the kernel's `Data`, algorithms and
  constraint solver. Each will be bound when a task needs it (6.3's parity
  tests may).

## What went wrong, and how it was found

- **The first build was stopped by PowerShell, not by the compiler.** Run
  under `*> log`, the build script ended at the first line CMake wrote to
  stderr: FetchContent's progress while it downloaded pybind11. Windows
  PowerShell turns such lines into errors when the output is redirected,
  and the script stops at errors. Running it as `powershell -File ... >
  log 2>&1` from outside avoids it.
- **pybind11 stayed at 2.12.0 after its pin changed.** The dev preset's
  `FETCHCONTENT_UPDATES_DISCONNECTED` never updates fetched sources. Found
  from the populate step's "disconnected update" line, and fixed by removing
  the three `_deps/pybind11-*` folders.
- **The module was built for Python 3.13 although 3.12 was found.** pybind11
  2's legacy finder had cached `PYTHON_MODULE_EXTENSION = .cp313-win_amd64.pyd`,
  and pybind11 3 reuses that value if it is set, so `import multibody`
  failed. The build script's `-Python` now drops the upper-case `PYTHON_*`
  entries. The compile commands, checked, already used 3.12's headers.
- **`def_readwrite` gives a read-only view of an Eigen member**, so
  `model.gravity[2] = -1.62` raised an error. It was safe, but not D52's
  rule. Fixed with a helper that gives writable views of fixed-size members.
- **A force callback that read `sim.q` produced a constant force.** It saw
  the start of the step, not the integrator's intermediate stages. The C++
  documentation says to read `states()`. The test was wrong, and the
  docstring, `docs/python.md` and the troubleshooting guide now say it
  plainly.
- **A state element's crossing time was miscalculated** (τ ln 4 instead of
  τ ln(4/3)). The test of the cap caught it.
- **Comments claimed measurements before they were made.** Five times a
  test's comment stated a "measured" error written ahead of the measurement:
  the pendulum's 3e-11, the oscillator's 2e-14 to 2e-13, the parallelogram's
  3e-10, the lag's 1.3e-12 and the energy's 1.5e-11. Each was measured and
  corrected before its commit. All five were off, by factors from 1.6 (the
  lag) to 160,000 (the parallelogram, because the projection closes the
  loop to rounding, not merely to its tolerance). The habit has to stop: a comment gets its number
  from the measurement, never the other way round.
- **Windows paths in Python strings.** Three scripted edits broke on `\r`
  or `\U` escapes in paths. Edits with paths now go through scripts written
  to a file, with raw strings.

## Verification

Five Python test files with 32 tests, each against a derived value. The
measurements are in the data file.

- `test_pendulum.py`: the period at 1 rad against 4 K(k)/ω₀ (measured
  2.9e-12 rad at tolerances of 1e-12 and 2.9e-10 at 1e-10; allowed 3e-11,
  so the check also shows that the tolerances reach the integrator); the
  speed through the bottom from the energy; the bob's place; `run_recorded`
  against `run()`, to the bit.
- `test_interface.py`: an engine error's code; views written through;
  wrong sizes refused (P001); the system kept alive; the integrators as an
  enum; settings written through; transforms composed, inverted and checked
  (P002 to P004); the model's tables.
- `test_constraints_and_forces.py`:
  - one oscillator through a linear spring, a joint spring and a Python
    force, each within 8.4e-13 of 0.1 cos 5t;
  - a parallelogram keeping (θ, −θ, θ) and swinging to −(π + 1), with
    MBD-K050 as a Python warning;
  - a joint driven by Python functions;
  - lists as tuples; the user force's handle closed after its call (P005)
    and checked (P006);
  - a Python exception reaching the caller.
- `test_analyses.py`: assembly with the crank held, statics at −Mg/K,
  linearisation's ω = 5 and ζ = 0.2, a driven point's kinematics, a bounce at
  √(2/g), the force callback's oscillator, a pendulum's energies and joint
  reaction, and `validate`'s count of the loop's freedom.
- `test_state_elements.py`: a lag against its exponential; `end_step`
  capping a state; a missing method named (P007); the trace's energy.
- The suite: 532 tests pass (527 C++, 5 Python files). CI on `668b992`
  passed all five jobs.

## Lessons

- Every list or array that crosses into Python is a copy unless it is
  deliberately made a view. Every place where a user would naturally write
  into one (the state, a list of constraints, a curve triple, the loads a
  Python force adds) has to be a view, a handle, or a tuple that refuses
  the write. Losing a write without a word is the failure to design out.
- An object that keeps a pointer (a constraint set's name, a simulator's
  system, a force handle) needs its lifetime decided at the binding, not
  left to the garbage collector.
- Build state that outlives a change, such as cached interpreter entries
  and fetched sources, deserves the same suspicion as code.

## Open ends

- CI's Python 3.11 run of the tests is confirmed by the annotation added in
  the third part, from its next run on.
- The interpreter lock is held through every call into C++; releasing it
  for long runs without Python callbacks is for later.
- Tasks 6.2 to 6.6: packaging (`pyproject.toml`, a wheel), parity tests,
  files, viewing, examples.

## For the book

- Part on the Python interface: the copy trap at a language boundary, and
  how each case was designed out (views, handles, tuples); the stage
  against the step in a callback.
- Part X, a lesson on the record: a number in a comment comes from a
  measurement, and five that didn't were each wrong.
