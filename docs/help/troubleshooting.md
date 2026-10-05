# Troubleshooting

From what you see to why, and what to do. Each entry gives the symptom, the
cause, the fix, and where to read more: a message code (in the
[catalogue](messages.md)) or the journal entry where the problem was first
met. The last section shows how to use the debugging tools; the table at the
end lists every problem the journal records and its entry here.

When a message has a code, start with the catalogue. When nothing is
reported but the result looks wrong, start with `validate(system)` and a
trace (section 8).

## 1. Building

**The PC restarts, or freezes, during a build.**
A parallel build of this project needs well over a gigabyte per compiler; a
default Ninja build on the development laptop would start 22 at once. Two
such builds restarted the laptop in 2026. The build system runs one compiler
at a time (`MBD_COMPILE_JOBS`, decision D9); do not raise it on that machine
without watching memory and temperature (README, "Build limits"). In a
cloud container or on CI, three is fine.

**Editing a header does not rebuild anything (Windows, non-English Visual Studio).**
Ninja learns a file's headers from the compiler's notes, which a Spanish
Visual Studio translates; if the console code page differs between configure
and build, Ninja stops recognising them, silently. Build through
`scripts\build.ps1`, which fixes the code page and reconfigures when it
changed (README, "Non-English Visual Studio"; journal, Phase 0).

**A build ignores a header edited while it was running (MSVC).**
The precompiled header was built before the edit, and the files compiled
after it use the old version. Do not edit sources a running build has yet to
compile; rebuild after editing (journal, Phase 2: simulator and vehicles).

**New warnings appear in code nobody touched.**
Inline functions no test calls are never compiled; moving or first calling
them brings out what was always there (four unused variables appeared that
way, journal: library and checks). Fix them where they appear. GCC 13 also
prints `-Warray-bounds` and `-Wstringop-overflow` warnings from inside its
own standard library, reached through spdlog and Catch2; they are false
positives and point at `/usr/include`, not at this project.

**A test fails only in the CI job `kernel-no-malloc`.**
Something in the kernel allocates on the heap after setup. Eigen evaluates
`x = y - A * b` through a temporary shaped like `y`, on the heap when `y` is
a block of a dynamic vector: write `x = y; x.noalias() -= A * b;` (the rule
in `docs/kernel.md`, "Memory"). To reproduce: configure a Debug build with
`-DCMAKE_CXX_FLAGS=-DEIGEN_RUNTIME_NO_MALLOC`, then run
`test_kernel "[alloc]"` and `test_alloc` (journal, Phase 2: constraints).

**Running one test by name runs nothing.**
Catch2 takes a comma in a name as a separator between names: escape it as
`\,` (journal, outputs).

## 2. Models

**A model behaves oddly, or the solver complains, and the reason is not
clear.** Run `validate(system)` (or `validate(system, q)` at the state in
question): it lists at once inertias no body has, bodies out of order,
constraints on missing bodies or on one body only, joints that move nothing
with mass, constraints not met, redundant equations and floating bodies, each
with its code (MBD-M001 to M051), and counts the degrees of freedom.
`describe(system)` lists what the model contains.

**The degrees of freedom are not what you expected.**
`describe(system, q)` gives the count at a configuration; at a loop's neutral
configuration (all angles zero) the loop may be singular, a four-bar laid
straight, and the count is wrong there. Count at a closed configuration.

**Redundant constraints (MBD-K050, M051, K103).**
A planar loop closed in three dimensions, or a joint closed twice. The
motion is unaffected; the redundant equations carry no force. Close planar
loops with fewer equations if their forces matter.

**A parameter seems to act twice, or not at all.**
Read its definition in the header. Example from Phase 1: the brakes'
`max_torque` was read three ways in three places, and the drivetrain applied
twice the stated total; it now has one definition (the total over the four
wheels at full pedal). Finding F9 lists vehicle-template parameters that are
declared and never read (`steering_ratio`, `max_steer_angle`,
`R_loaded_approx`, `cg_offset`).

## 3. Starting states

**A simulation starts with a jump.** The initial values do not satisfy the
constraints, and the first projection pulls them on. Use
`assemble(sim, spec)` (task 3.4), holding the coordinates that define the
intended state; its report says how far each level was from consistent.

**Assembly moved the wrong part.** Without holds, the least correction in the
kinetic-energy metric moves light parts most (MBD-K066): a four-bar given
all three angles wrong turned its light crank furthest. Hold as many
coordinates as the mechanism has degrees of freedom, chosen so that they fix
it (journal, assembly).

**Assembly fails (MBD-K062 to K065).** Held values contradict the
constraints (too many held, or a held coordinate the constraints determine),
or a loop cannot close. The report names the constraint left unsatisfied and
returns the closest state it reached.

**A vehicle does not settle: in a simulation it keeps moving, or creeps.**
Free-rolling tyres resist nothing at rest, so a car keeps the momentum of
its first transient and rolls (finding F7; the sedan rolls at about 4 mm/s
after 8 s); below 1 m/s the tyre's slip is clamped, so a car on its brakes
on a slope creeps. Do not settle a car by simulating it: use
`static_equilibrium(sim)` (task 3.5), which reaches accelerations of 1e-8 in a
few iterations. `set_vehicle_equilibrium` only places a car near rest
(accelerations up to 73, finding F9).

**Statics reports free directions (MBD-K073).** Directions along which
nothing pushes back: a car's position and heading on a flat road, a sphere's
rolling. Statics leaves them as given. If one should be held, check the
element that should hold it.

**Statics finds an unstable equilibrium (MBD-K072).** Newton finds the
nearest equilibrium, stable or not. Start nearer the resting state, or
disturb the result and solve again.

**Statics falls back on dynamic relaxation (MBD-K074), or fails (MBD-K070).**
Relaxation is normal when a force acts where nothing is stiff yet (a body
above the ground it will rest on). Every iteration falling back is not:
check `history` in the report. A failure means no equilibrium exists (an
unopposed force: a car on a slope with free-rolling tyres), or the
relaxation needs a shorter step or more steps (`StaticsOptions`).

## 4. Simulations

**The projection does not converge (MBD-K051, K100).** The constraints
cannot be met: a loop that cannot close, contradictory drivers, a driver
target out of reach, a singular position. `dump_state(sim)` at the first
failure names each constraint's residual; `validate(system, q)` there.

**Energy grows, or decays, where it should not (MBD-K101, K102).** Turn on
`Simulator::trace` and `diagnose` the trace. K101 (the energy balance drifts)
means the integration or the projection makes or loses energy, usually a
step too long for the stiffest motion: halve the step and compare, and find
the stiff element with `linearize` (its highest mode). K102 (the peaks grow
while the applied forces do work) means something applied puts energy in: a
motor or a driver, or a force with the wrong sign, such as a damper that
pushes.

**The constraints drift within a step (MBD-K104).** The step is long for the
motion; the projection hides the drift, but the step's dynamics were
computed off the constraints. Shorten the step.

**A check of energy or drift cannot tell numerical error from a defect.**
Make the step small enough that the integrator's error is well below the
effect sought: the Phase 1 invariant tests needed a 0.25 ms step before RK4's
truncation was smaller than a real defect (journal, Phase 1). For RK4 the
energy error scales as (omega dt)^4.

**A stiff element (contact, bushing, friction) makes the simulation blow up.**
An explicit integrator is stable only while the step resolves the fastest
motion: RK4 while omega dt < 2.8. Regularised friction has its own speed,
`mu F_n / (v_s m)`: with mu = 0.5 and a slip speed of 1 mm/s, a 10 kg block
on four contact points needs a step under 0.6 ms (journal, contact). Shorten the step, raise the slip speed
`v_s`, or wait for the implicit integrators of Phase 4.

## 5. Events and contact

**An event was missed.** Detection compares the signs at the ends of a step,
so two crossings of one function within one step (a rise and a fall) are
both invisible. Make the step shorter than the time between them (journal,
events).

**Events chatter (MBD-K091).** An action that makes its own function cross
again at once: switched friction at rest is the classic case. After 100
events in a step the rest of the step is taken without events, and what the
actions maintain is no longer updated: the motion after that point is not
to be trusted. Use a regularised law (the friction of `JointCoordinateForce`
or `PlaneContact`), a direction, or two events with hysteresis.

**A bouncing body stops bouncing after a few dozen impacts, then falls
through the floor.** Each impact is placed up to `event_tolerance` past the
floor; once the rebound speed is too small to climb back above it (about
2e-9 m/s for 1e-10 s), no further crossing is seen: the tolerance resolves
the Zeno point. For a body that should come to rest, use penalty contact
(`PlaneContact`) instead of events (journal, events).

**An event's action sees the state exactly on its switching surface.** It
does not: events land strictly past the crossing, where the function has its
new sign, so an action may rely on the switch having happened (decision
D30).

**A block that should stick on a slope creeps.** Regularised friction makes
sticking a creep at about the slip speed: exactly `v_s atanh(tan th / mu)` on
an incline of angle th. Lower `v_s` (and the step with it) if the creep
matters (decision D29).

**A contact pulls, or its force jumps at first touch.** It does neither: the
normal force is clamped at zero and its damping ramps from zero over
`damping_depth`. A force that stays zero while penetrating is a damper
outrunning its spring as the body separates fast, which is intended.

## 6. Slow runs and timings

**The same benchmark gives different times from run to run.**
On a hybrid CPU (the i7-13700H) an unpinned single-threaded run lands on a
performance or an efficiency core: times varied by more than two. Pin to a
performance core at high priority. Even pinned, the machine's own state
changed by 2.2 within a morning: compare two versions only alternately, in
one session (decision D21; journal, performance).

**A time measured in a cloud session disagrees with the laptop's.**
Different machines; never compare them (decision D25). Compare versions on
one machine, in one session.

**A simulation is slow.** Measure first (`bench/bench_dynamics.cpp`,
`docs/performance.md`). A trace costs a constraint and a force evaluation per
step; statics and linearisation cost about 2 nv force evaluations per
iteration (their stiffness is by differences).

## 7. Developing the engine

**A new joint, constraint or force element.** It passes the invariant tests
in `tests/kernel` before anything is built on it (working rule 2): its
Jacobian and acceleration terms against finite differences, its forces
against the gradient of its potential, energy and momentum conserved when
nothing dissipates them. A force element's `law()` returns its derivatives;
test them against central differences, away from kinks (a clamp at zero, a
stop's engagement).

**A tolerance in a test fails although the code is right.** Derive the
tolerance (working rule 3), including roundoff: a central difference of step
h has roundoff about eps |F| / h, so even a linear force's derivative by
differences is off by about 1e-11 relative at h = 1e-6 (journal, statics).
Write the expected value with the same arithmetic as the code when it
matters (`0.35 - 0.30` is not `0.05` in floating point; journal, force
library).

**A brute-force reference disagrees with the code.** A search is only as
good as its region: print the cost along the search before trusting a
failure (the assembly test's search interval missed the minimum; journal,
assembly). Compute expected values from the closed form, not by estimate
(the bouncing ball's impact count; journal, events).

**A test of a guard does not trigger.** Find out why before changing the
test: the reason is often a property of the method (journal, events).

**A rank-revealing solve gives absurd steps.** Eigen's
`CompleteOrthogonalDecomposition` decides its rank when it decomposes: set
the threshold before `compute()`, not after (journal, statics). A pivoted
LDLT that drops pivots solves the independent equations, not a least-squares
problem; the two differ when the right-hand side is not in the range
(journal, assembly).

**A test compares a quantity with itself.** It cannot fail. Each test must be
able to fail on a wrong engine (journal, Phase 2: constraints).

## 8. The debugging tools

| Tool | Gives | Where |
|---|---|---|
| `validate(system[, q, t])` | Every problem it can find in a model, with codes; the degrees of freedom | `kernel/validate.hpp` |
| `describe(system[, q])` | The model as text | `kernel/diagnostics.hpp` |
| `dump_state(sim)` | The state, each constraint's residual, the last projection | `kernel/diagnostics.hpp` |
| `Simulator::trace`, `write_trace_csv`, `diagnose` | A row per step (drift, projection, rank, energies, work, balance, events) and what it says (K100 to K104) | `kernel/diagnostics.hpp`, `kernel/simulator.hpp` |
| `assemble(...)`, `static_equilibrium(sim)`, `linearize(sim)` | Reports with residuals, iterations, free and unstable directions, modes | `kernel/assembly.hpp`, `statics.hpp`, `linearization.hpp` |
| `compute_loads(sim)` | Reactions, constraint forces, accelerations | `kernel/outputs.hpp` |
| `Recorder` | Any quantity over time, to CSV | `analysis/recorder.hpp` |
| The diagnostic sink | Every warning; `init_logging()` sends them to the logger, or install your own sink | `core/core.hpp`, `core/logging.hpp` |

A typical session with a misbehaving simulation:

```cpp
kernel::ValidationReport check = kernel::validate(sys);
std::cout << check.summary();             // anything wrong with the model?
sim.trace = true;
sim.run(2.0, 1e-3);
kernel::write_trace_csv(sim.trace_rows(), "trace.csv");
std::cout << kernel::diagnose(sim.trace_rows()).summary();
std::cout << kernel::dump_state(sim);     // where it ended
```

## The journal's problems, and their entries

| Journal entry | Problem | Entry above |
|---|---|---|
| Before the review | Restarts of the PC during builds | 1, "The PC restarts" |
| Before the review; review | Hand-written kinematics that disagreed (F1 to F4) | 7, "A new joint, constraint or force element" |
| Phase 0 | Header edits not rebuilding (code page) | 1, "Editing a header does not rebuild" |
| Phase 1 | Drift check needing a 0.25 ms step | 4, "A check of energy or drift" |
| Phase 1 | `max_torque` read three ways | 2, "A parameter seems to act twice" |
| Phase 2: constraints | Heap temporary in ABA | 1, "A test fails only in kernel-no-malloc" |
| Phase 2: constraints | Catch2 cannot report a skipped test | 1, "kernel-no-malloc" (the hidden test) |
| Phase 2: constraints | A tautological test | 7, "A test compares a quantity with itself" |
| Phase 2: simulator and vehicles | Header edited during a build (MSVC precompiled header) | 1, "A build ignores a header" |
| Phase 2: library and checks | Unused variables in never-compiled inline code | 1, "New warnings appear" |
| Phase 2: library and checks | "Unconnected bodies" reinterpreted as floating bodies | 2, `validate` (MBD-M050) |
| Phase 2: performance | Unpinned timings varying by two; the machine's slow state | 6, "The same benchmark gives different times" |
| Phase 3: force library | `0.35 - 0.30` is not `0.05`; a run shorter than a period | 7, "A tolerance in a test fails" |
| Phase 3: outputs | A comma in a test name | 1, "Running one test by name" |
| Phase 3: assembly | Light parts move most; a search region that missed the minimum | 3, "Assembly moved the wrong part"; 7, "A brute-force reference" |
| Phase 3: assembly | Contradicting holds stalling; pivot dropping is not least squares | 3, "Assembly fails"; 7, "A rank-revealing solve" |
| Phase 3: statics | The decomposition's threshold set too late | 7, "A rank-revealing solve gives absurd steps" |
| Phase 3: statics | A tolerance derived without roundoff | 7, "A tolerance in a test fails" |
| Phase 3: statics | A car that never settles in a simulation | 3, "A vehicle does not settle" |
| Phase 3: contact | A test that asked the law for a pull | 5, "A contact pulls" |
| Phase 3: contact | Friction's stability limit | 4, "A stiff element" |
| Phase 3: events | Miscounted impacts | 7, "A brute-force reference" |
| Phase 3: events | Two crossings in one step | 5, "An event was missed" |
| Phase 3: events | The tolerance ends a bouncing sequence | 5, "A bouncing body stops bouncing" |
| Phase 3: events | An event at an exact zero; chattering | 5, "An event's action"; "Events chatter" |
| Cloud sessions (D25) | Timings from different machines | 6, "A time measured in a cloud session" |
| Cloud sessions | GCC's false-positive warnings from its standard library | 1, "New warnings appear" |
