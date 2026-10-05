# multibody

A C++20 multibody dynamics engine aimed at vehicle and suspension simulation: joint-coordinate rigid-body dynamics with loop-closing constraints, tyre and suspension models, and the analyses built on them.

The project is in active development and its core is being revised. The roadmap, the findings of the October 2026 review, and the status of each task are in [documentation/Master_plan.md](documentation/Master_plan.md). Frames, units and sign conventions are in [docs/conventions.md](docs/conventions.md). The multibody kernel (spatial algebra, joint models, recursive algorithms, constraints, the simulator) is described in [docs/kernel.md](docs/kernel.md), and timings in [docs/performance.md](docs/performance.md).

## Layout

| Path | Contents |
|---|---|
| `include/mbd/core` | Math types, transforms, errors, logging |
| `include/mbd/spatial` | Spatial vectors, frame changes, spatial inertia |
| `include/mbd/kernel`, `src/kernel` | Joint models, model and data, recursive algorithms, constraints, constrained dynamics, simulator |
| `include/mbd/model` | Rigid-body inertia, state and forces |
| `include/mbd/forces` | Springs, tyres, aerodynamics, anti-roll bar |
| `include/mbd/vehicle` | Suspension corners, vehicle builders, drivetrain |
| `include/mbd/analysis` | Suspension kinematics, bicycle model, track, lap simulation, optimisation |
| `tests` | One executable per folder above |
| `documentation` | Plans, review evidence, reference notes |
| `docs` | Project documentation |
| `scripts` | Build helper |

The kernel is compiled (`src/kernel`); the force, vehicle and analysis layers are still header-only. Moving them into the compiled library is task 2.8 of the plan.

## Requirements

- Windows with Visual Studio 2022 or later, "Desktop development with C++" workload. That provides the compiler, CMake and Ninja.
- CMake 3.25 or later.
- Python 3.11 for the Python bindings. Configure with `-DMBD_BUILD_PYTHON=OFF` to leave them out.

Eigen 3.4.0, Catch2 3.6.0, spdlog 1.13.0 and pybind11 2.12.0 are downloaded by CMake when it configures. Each is pinned to an exact hash.

## Building and testing

From a PowerShell window in the repository root:

```powershell
scripts\build.ps1                               # build everything
scripts\build.ps1 -Test                         # build everything, run all tests
scripts\build.ps1 -Target test_model -Test      # one test executable
scripts\build.ps1 -Test -Filter "Drivetrain"    # tests whose name matches
```

The script loads the Visual Studio environment and then uses the presets in `CMakePresets.json`. From a "x64 Native Tools" prompt the same steps are `cmake --preset dev`, `cmake --build --preset dev` and `ctest --preset dev`.

A test executable can also be run directly, with Catch2's own filters:

```powershell
build\tests\test_vehicle.exe "Drivetrain*"
build\tests\test_kernel.exe "[constraints]"
```

## Build limits

**The build runs one compiler at a time, and that limit is part of the build system.** It is set by `MBD_COMPILE_JOBS` in the top-level `CMakeLists.txt` through a Ninja job pool, so a plain `cmake --build` is limited too, whatever Ninja would choose by itself.

Why: compiling this code is heavy, because the engine is header-only and every file that uses it compiles all of it. Measured on the development laptop (i7-13700H, 20 logical cores, 16 GB), October 2026:

| | Time | Peak compiler memory |
|---|---|---|
| One test file, no precompiled header | 92 s | 1.6 GB |
| The shared precompiled header (built once, rebuilt when an engine header changes) | 49 s | 1.7 GB |
| One test file using it | 23 s typical, 58 s at most | 1.1 GB |
| All 47 test files and the header, from scratch, one compiler | about 21 min | |

Without the shared header, a full rebuild took roughly an hour (estimated from the per-file times). The vehicle test executable alone went from about 16 minutes to 7.5.

A default Ninja build would start 22 compilers at once on this machine. At 1.6 GB each that is more than twice its memory.

By memory alone, 16 GB with 4 GB kept free would allow about six compilers. The default nevertheless stays at one: in mid-2026 two all-core builds restarted this PC, and whether memory or heat caused it was never established. To raise the limit, watch temperatures during a build and step up gradually:

```powershell
cmake --preset dev -DMBD_COMPILE_JOBS=3
```

The `ci` preset uses three, sized for a hosted runner.

## Non-English Visual Studio

Ninja learns which headers a source file includes from notes that the compiler prints, and recognises those notes by a prefix that CMake records when it configures. On a non-English Visual Studio the notes are translated, and their bytes depend on the console code page. If a build runs under a different code page from the one used to configure, Ninja stops recognising the notes and **header changes no longer trigger rebuilds, without any error**.

`scripts\build.ps1` prevents this: it fixes the code page, and configures again if the build directory was configured under another one. If you drive CMake by hand, use the same terminal settings for configuring and building. Installing the English language pack for Visual Studio removes the problem at its source.

## Tests

`ctest --preset dev` runs every Catch2 test case as its own test. The executables mirror the source folders: `test_core`, `test_model`, `test_kernel`, `test_forces`, `test_vehicle`, `test_analysis`.

The invariant tests in `test_kernel` are different in kind. They check relations that must hold for every joint and constraint at any state, by comparing the engine with finite differences of its own position-level quantities and with conservation laws:

- body velocities are the time derivative of body poses;
- the mass matrix, the inverse dynamics and Lagrange's equations describe the same system;
- a constraint's Jacobian and acceleration term are the first and second time derivatives of its equation, and its equations are independent;
- energy and momentum are conserved when nothing dissipates them.

Two rules follow from the October 2026 review, and are part of the plan:

1. A new joint, constraint or force is added to the invariant tests in `tests/kernel` and passes there before anything is built on it.
2. A test asserts a value that was derived, with the derivation in a comment. A threshold that only records what the code happened to do is not a test.
