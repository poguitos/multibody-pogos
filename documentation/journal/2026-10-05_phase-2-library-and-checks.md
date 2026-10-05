# Phase 2: the legacy core goes, one library, checks that stay on

- **Dates:** 5 October 2026
- **Plan:** tasks 2.7b, 2.8, 2.9; findings F16, F20; decision D5
- **Commits:** `860b264` (2.7b), `32f2075` (2.8), `253e788` (2.9)
- **Written:** at the time

## Goals

- **2.7b** Delete the legacy multibody core and its tests, keeping their
  closed-form cases on the kernel. *Done when nothing includes those files and
  every test passes.*
- **2.8** Move non-template code into `src/`. *Done when editing one source
  file recompiles one file, and a full rebuild at one job takes under ten
  minutes.*
- **2.9** Size checks in all builds, and a `validate()` that reports topological
  order, invalid inertias, unconnected bodies, degrees of freedom and
  redundant constraints. *Done when each malformed test model produces a
  specific message.*

## 2.7b: retiring the legacy core

Five headers went (`model/system.hpp`, `model/joint.hpp`,
`model/constraint.hpp`, `algorithms/dynamics.hpp`, `integrators/simulator.hpp`)
with the 18 test files that exercised them (176 test cases), the Phase 1
invariants of the legacy joints and the kernel-versus-legacy comparison. The
analytic content of those tests moved to `test_kernel_joints_physics.cpp`:
pendulum periods on revolute, spherical and universal joints, a slider on a
spring, sliders on inclines, the steady spin of a symmetric body, a fixed
joint's placement, the slow normal mode of a double pendulum
(`omega^2 = (2 - sqrt 2) g / L`), two bodies held together at a point under a
balanced load, and a height driver holding a point of a falling body.

**A behaviour changed and the test was rewritten, not loosened.** With two
contradictory constraints (one body point held at heights 0 and 0.5), the
legacy solver gave a least-squares compromise. The kernel finds the two
equations have the same Jacobian, keeps the first and drops the second as
redundant: the point stays at 0, the second residual stays at 0.5, and the
projection fails at every step. The new test asserts exactly that: 50 failed
projections in 50 steps, two warnings (redundancy, then the first failure),
height 0, residual 0.5.

A full build at one compiler took about nine minutes, so the new test file
was checked line by line against the APIs it uses before the build started;
it then compiled on the first try.

The documentation followed: `docs/conventions.md` still described the legacy
joints (rotation vectors) and constraint interface, and was rewritten for the
kernel. The plan's findings keep their links to deleted files, with a note
that git history holds them.

## 2.8: one compiled library

**Moving code without retyping it.** About 2,900 lines of function bodies had
to leave 17 headers. Retyping them risks silent changes (a constant mistyped
is not caught by any compiler), so a script did the move (`split.pl`, kept in
the session scratchpad): given the line where a signature starts, it finds the
body by matching braces at the same indentation, replaces it in the header by
`;`, and appends signature and body to the source file, de-indented, with the
class name inserted before the function name and the declaration-only parts
removed (default arguments, `static`, `virtual`, `override`, `inline`). Two
mistakes were caught on staged copies, before anything touched the tree:
default arguments written with exponents (`1e-6`) were not stripped, and moved
free functions kept `inline` on their header declarations, which would leave
inline functions without definitions. Two return types that are class members
(`Drivetrain::TireSet`, `BicycleModel::CorneringPoint`) were qualified by hand.
119 functions moved.

**Measuring before deciding.** The build's own log (`build/.ninja_log`) gave
the time of every object. With the shared test header most test files took 1
to 3 s, while each kernel source file took 10 to 46 s, nearly all of it
parsing Eigen. Moving 17 more headers into sources would have added minutes,
not saved them. So the library got its own precompiled header (standard
library, Eigen, core and kernel; the vehicle headers left out so that editing
one recompiles only its users): the kernel files dropped to 2 to 3 s each and
the new files take 0.4 to 2.2 s.

**Result.** A full rebuild at one compiler: 7.4 min for 65 files, against 7.8
min for 46 files before. A dry run of Ninja after editing one source file
lists that file, the library archive and the links, nothing else.

**What went wrong.** Compiling the moved code once brought out four unused
variables in `BicycleModel::max_lateral_acceleration`: an inline function no
test had ever called, so no compiler had ever compiled it. Removed.

## 2.9: checks that stay on

- **Size checks in every build.** The kernel's entry points used debug-only
  asserts; in a release build a vector of the wrong size was read out of
  bounds. Every entry point now checks its vectors against the model and its
  `Data` against the model. A check is one comparison, and its message (the
  function, the argument, both sizes) is built in a separate function only
  when it fails, so the fast path does not allocate.
- **References.** Constraints and force elements now report the bodies they
  act on (`bodies()`), and force elements their `name()`. Building a
  `ConstraintSolver` or a `Simulator` rejects one that refers to a body that
  does not exist.
- **`validate(system)`** reports without throwing: arrays out of step, bodies
  out of topological order, joint indices that do not match their joints,
  inertias no real body has (not finite, negative mass, not symmetric, a
  negative principal moment, principal moments breaking the triangle
  inequality), an inertia or joint frame changed after the body was added (the
  algorithms would still use the old one), a joint that moves nothing with
  mass (a singular mass matrix, naming the joint), constraints and force
  elements on missing bodies, a constraint with all its markers on one body,
  constraints not satisfied, floating bodies, and redundant equations. It
  counts the degrees of freedom with the solver's own rank, so its count
  agrees with what the simulator will see.

**"Unconnected bodies" needed reinterpreting.** In the kernel every body comes
with its joint, so no body can be unconnected. The check that remains useful
is a body floating on a free joint with nothing acting on it or on what it
carries, which usually means a forgotten connection; it is reported as a note.

## Verification

- `test_kernel_validate.cpp`: one malformed model per check, each asserting its
  specific message; a planar four-bar closed in 3D has 3 velocities, 5
  equations of rank 2, 3 redundant, and 1 degree of freedom; the summary text;
  and the size and reference checks of the algorithms, the solver and the
  simulator.
- Every template vehicle validates with 10 degrees of freedom whatever its
  suspension: double wishbone 27 velocities and 17 equations, McPherson 23 and
  13, simple 10 and none (chassis 6, plus one bump travel per wheel).
- 352, then 352, then 364 tests; CI green on all three jobs each time, including
  the Linux build without precompiled headers, which shows every file still
  includes what it uses.

## Lessons

- Move code mechanically; review the mechanism on copies first.
- Measure where build time goes before restructuring for build time.
- A function that no test calls is a function no compiler has checked.

## For the book

- The `split.pl` approach as an example of safe large refactoring.
- The anatomy of a C++ build: parsing shared headers versus generating code,
  measured per object, and what each kind of precompiled header saves.
- `validate()` as the program's first "help": messages that name the body or
  constraint at fault, written for the person who will read them.
